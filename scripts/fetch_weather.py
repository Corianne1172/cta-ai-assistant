"""
fetch_weather.py
----------------
Downloads hourly Chicago weather from Open-Meteo's free historical archive.
No API key required.

Variables pulled:
  precipitation_mm  — mm of rain per hour
  snowfall_cm       — cm of snow per hour
  temperature_2m    — °C at 2m height
  windspeed_10m     — km/h at 10m height
  weathercode       — WMO weather code (0=clear, 71-77=snow, 61-67=rain, etc.)

Derived:
  weather_idx       — 0.0 (clear) to 1.0 (blizzard/extreme), composite of all vars

Output: data/raw/weather_hourly.csv

Run:
    python scripts/fetch_weather.py
"""

from __future__ import annotations

from pathlib import Path
import requests
import pandas as pd
import numpy as np

ROOT    = Path(__file__).parent.parent
OUT     = ROOT / "data" / "raw" / "weather_hourly.csv"

# Chicago city centre
LAT, LON = 41.8781, -87.6298

# 2022 + 2023 — two full years covering all seasons
START_DATE = "2022-01-01"
END_DATE   = "2023-12-31"

API_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LAT}&longitude={LON}"
    f"&start_date={START_DATE}&end_date={END_DATE}"
    "&hourly=precipitation,snowfall,temperature_2m,windspeed_10m,weathercode"
    "&timezone=America%2FChicago"
    "&wind_speed_unit=kmh"
)


# ─────────────────────────────────────────────────────────────────────────────
# Weather index derivation
# Composite 0→1 score; each component capped so no single variable dominates
# ─────────────────────────────────────────────────────────────────────────────
def weather_idx(temp_c: float, precip_mm: float,
                snowfall_cm: float, windspeed_kmh: float) -> float:
    """
    Components and their physical basis for rail delays:

    Snow (max 0.45):  Track icing, switch freezing, reduced adhesion on elevated
                      sections. Most impactful variable for the L.
    Rain (max 0.25):  Signal interference, wet-rail slip. Less than snow.
    Cold (max 0.20):  Rail contraction causes switch failures below -10°C.
                      Wind chill on exposed elevated platforms.
    Wind (max 0.10):  Debris on tracks, speed restrictions on exposed sections.
    """
    snow_score  = min(0.45, float(snowfall_cm)  / 5.0)   # 5 cm/hr  → max
    rain_score  = min(0.25, float(precip_mm)    / 12.0)  # 12 mm/hr → max
    cold_score  = min(0.20, max(0.0, (-float(temp_c) - 5.0) / 75.0))  # −5°C→0, −80°C→0.2
    wind_score  = min(0.10, float(windspeed_kmh) / 80.0) # 80 km/h  → max
    return round(min(1.0, snow_score + rain_score + cold_score + wind_score), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if OUT.exists():
        print(f"[weather] Already downloaded — {OUT}")
        _print_stats()
        return

    print(f"[weather] Fetching hourly data {START_DATE} → {END_DATE} from Open-Meteo …")
    resp = requests.get(API_URL, timeout=60)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame({
        "datetime":     data["time"],
        "temp_c":       data["temperature_2m"],
        "precip_mm":    data["precipitation"],
        "snowfall_cm":  data["snowfall"],
        "windspeed_kmh":data["windspeed_10m"],
        "weathercode":  data["weathercode"],
    })

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"]     = df["datetime"].dt.date.astype(str)
    df["hour"]     = df["datetime"].dt.hour

    # Fill any nulls (rare at edge of forecast range)
    for col in ["temp_c","precip_mm","snowfall_cm","windspeed_kmh"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["weather_idx"] = df.apply(
        lambda r: weather_idx(r.temp_c, r.precip_mm, r.snowfall_cm, r.windspeed_kmh),
        axis=1,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"[weather] Saved → {OUT}  ({len(df):,} rows)")
    _print_stats(df)


def _print_stats(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = pd.read_csv(OUT)
    print(f"\n── Weather index distribution ──────────────────────────")
    bins   = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["clear (0-0.05)","light (0.05-0.2)","moderate (0.2-0.4)",
              "heavy (0.4-0.6)","severe (0.6-0.8)","extreme (0.8-1.0)"]
    counts = pd.cut(df["weather_idx"], bins=bins, labels=labels).value_counts().sort_index()
    for label, n in counts.items():
        pct = 100 * n / len(df)
        print(f"  {label:<25} {n:>6,}  ({pct:.1f}%)")

    print(f"\n── Extremes ────────────────────────────────────────────")
    worst = df.nlargest(5, "weather_idx")[["datetime","temp_c","snowfall_cm","precip_mm","weather_idx"]]
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
