"""
fetch_hourly_ridership.py
-------------------------
Downloads per-station hourly boardings from the Chicago Data Portal.

Dataset: CTA - Ridership - 'L' Station Entries - Hourly Totals (t2rn-p8d7)

If the hourly dataset is unavailable or too large, falls back to the daily
per-station dataset (5neh-572f, already in data/raw/ridership.csv) and
distributes daily counts across hours using a typical ridership profile.

Output: data/raw/ridership_hourly.csv
Columns: station_id, station_name, date, hour, rides

Run:
    python scripts/fetch_hourly_ridership.py
"""

from __future__ import annotations

from pathlib import Path
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "data" / "raw" / "ridership_hourly.csv"

# Filter to 2022–2023 to match weather and sports data
DATE_FILTER = "date >= '2022-01-01' AND date <= '2023-12-31'"

HOURLY_URL = (
    "https://data.cityofchicago.org/resource/t2rn-p8d7.csv"
    f"?$where={DATE_FILTER.replace(' ', '%20').replace('>=','%3E%3D').replace('<=','%3C%3D')}"
    "&$limit=500000"
)

DAILY_URL = (
    "https://data.cityofchicago.org/resource/5neh-572f.csv"
    "?$where=date%20%3E%3D%20'2022-01-01T00:00:00.000'%20AND%20date%20%3C%3D%20'2023-12-31T23:59:59.000'"
    "&$limit=500000"
)

# Typical Chicago L ridership distribution by hour (weekday)
# Source shape: CTA ridership studies showing ~5% of daily load at AM peak,
# ~4% at PM peak, near-zero overnight. Normalised to sum to 1.
HOUR_PROFILE = np.array([
    0.005, 0.003, 0.002, 0.002, 0.004, 0.010,  # 0-5
    0.020, 0.055, 0.075, 0.060, 0.040, 0.038,  # 6-11
    0.045, 0.042, 0.040, 0.045, 0.058, 0.072,  # 12-17
    0.065, 0.050, 0.040, 0.032, 0.020, 0.012,  # 18-23
], dtype=float)
HOUR_PROFILE /= HOUR_PROFILE.sum()


def try_hourly_download() -> pd.DataFrame | None:
    """Attempts to download the hourly dataset; returns None on failure."""
    print(f"[ridership] Trying hourly dataset (t2rn-p8d7) …")
    try:
        resp = requests.get(
            "https://data.cityofchicago.org/resource/t2rn-p8d7.csv"
            "?$limit=5&$where=date%20%3E%3D%20'2022-01-01'",
            timeout=20,
        )
        resp.raise_for_status()
        probe = pd.read_csv(pd.io.common.StringIO(resp.text))
        print(f"[ridership] Hourly dataset columns: {probe.columns.tolist()}")

        # Looks good — fetch full range
        resp2 = requests.get(HOURLY_URL, timeout=120)
        resp2.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(resp2.text))
        print(f"[ridership] Hourly dataset: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"[ridership] Hourly dataset unavailable: {e}")
        return None


def build_from_daily() -> pd.DataFrame:
    """
    Falls back to daily per-station data (already known to work) and
    distributes each day's rides across hours using HOUR_PROFILE.
    This preserves real station × day variation but estimates the hour.
    """
    print(f"[ridership] Falling back to daily dataset (5neh-572f) …")
    daily_path = ROOT / "data" / "raw" / "ridership.csv"

    # Always re-fetch — cached file is likely 2001-era data without date filter
    print(f"[ridership] Fetching daily data 2022–2023 from Data Portal …")
    resp = requests.get(DAILY_URL, timeout=60)
    resp.raise_for_status()
    df_daily = pd.read_csv(pd.io.common.StringIO(resp.text), dtype={"station_id": str})
    print(f"[ridership] Downloaded daily data: {len(df_daily):,} rows")

    df_daily.columns = [c.lower().strip() for c in df_daily.columns]
    df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.strftime("%Y-%m-%d")
    df_daily["rides"] = pd.to_numeric(df_daily["rides"], errors="coerce").fillna(0)

    # Filter to 2022–2023
    df_daily = df_daily[df_daily["date"] >= "2022-01-01"]
    df_daily = df_daily[df_daily["date"] <= "2023-12-31"]
    print(f"[ridership] After date filter (2022–2023): {len(df_daily):,} rows")

    # Expand each daily row into 24 hourly rows
    records = []
    for _, row in df_daily.iterrows():
        for hour in range(24):
            records.append({
                "station_id":   row["station_id"],
                "station_name": row.get("stationname", ""),
                "date":         row["date"],
                "hour":         hour,
                "rides":        round(float(row["rides"]) * HOUR_PROFILE[hour], 2),
            })

    return pd.DataFrame(records)


def normalise_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises column names regardless of which dataset was used."""
    df.columns = [c.lower().strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        if "station_id" in c and c != "station_id":
            rename[c] = "station_id"
        if "station" in c and "name" in c and c != "station_name":
            rename[c] = "station_name"
        if c in ("boardings", "entries", "total_boardings"):
            rename[c] = "rides"
        # Socrata date column is sometimes named differently
        if c in ("service_date",) and "date" not in df.columns:
            rename[c] = "date"
    df = df.rename(columns=rename)
    # Date column may contain ISO timestamps like "2022-01-01T00:00:00.000"
    date_col = next((c for c in df.columns if "date" in c), None)
    if date_col and date_col != "date":
        df = df.rename(columns={date_col: "date"})
    df["date"]  = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if "hour" not in df.columns:
        df["hour"] = 0
    df["hour"]  = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
    df["rides"] = pd.to_numeric(df["rides"], errors="coerce").fillna(0)
    if "station_name" not in df.columns:
        df["station_name"] = df.get("stationname", "")
    return df[["station_id", "station_name", "date", "hour", "rides"]]


def main() -> None:
    if OUT.exists():
        print(f"[ridership] Already built — {OUT}")
        _print_stats()
        return

    df_hourly = try_hourly_download()
    if df_hourly is not None and len(df_hourly) > 100:
        df = normalise_hourly(df_hourly)
        source = "real hourly"
    else:
        df = normalise_hourly(build_from_daily())
        source = "daily→hourly (estimated)"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n[ridership] Saved → {OUT}  ({len(df):,} rows)  [{source}]")
    _print_stats(df)


def _print_stats(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = pd.read_csv(OUT)
    print(f"\n── Ridership coverage ─────────────────────────────────")
    print(f"  Stations : {df['station_id'].nunique()}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Total rows: {len(df):,}")
    print(f"\n── Top 5 stations by avg hourly rides ─────────────────")
    top = (
        df.groupby(["station_id","station_name"])["rides"]
        .mean().sort_values(ascending=False).head(5)
    )
    print(top.round(1).to_string())


if __name__ == "__main__":
    main()
