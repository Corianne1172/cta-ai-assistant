"""
fetch_cta_data.py
-----------------
Downloads real CTA data from two official sources and produces
data/station_map.json — the single source of truth for station data
used by every other script in this project.

Sources:
  1. CTA GTFS static schedule
     https://www.transitchicago.com/downloads/sch_data/google_transit.zip
     Gives us: real station names, IDs, coordinates, and which L line serves each.

  2. CTA Ridership Daily Boarding Totals (Chicago Data Portal, dataset 6iiy-9s97)
     https://data.cityofchicago.org/resource/6iiy-9s97.csv
     Gives us: per-station average daily ridership (crowding signal for delay model).

Output files:
  data/raw/gtfs/          GTFS text files (stops, routes, trips, stop_times)
  data/raw/ridership.csv  raw ridership download
  data/station_map.json   cleaned station registry (consumed by all other scripts)

Run:
  python scripts/fetch_cta_data.py
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
GTFS_DIR = RAW_DIR / "gtfs"

GTFS_URL      = "https://www.transitchicago.com/downloads/sch_data/google_transit.zip"
# Per-station daily entries (dataset 5neh-572f)
# System-wide totals (6iiy-9s97) are not per-station — wrong dataset
RIDERSHIP_URL = (
    "https://data.cityofchicago.org/resource/5neh-572f.csv"
    "?$limit=100000"
)

# Sports venue coordinates for proximity tagging
VENUES: dict[str, tuple[float, float]] = {
    "cubs":        (41.9484, -87.6553),  # Wrigley Field
    "sox":         (41.8299, -87.6339),  # Guaranteed Rate Field
    "bears":       (41.8623, -87.6167),  # Soldier Field
    "bulls_hawks": (41.8806, -87.6742),  # United Center
}
VENUE_RADIUS_MILES = 0.4


# CTA route_id → friendly line name
ROUTE_NAME_MAP: dict[str, str] = {
    "Red":  "Red",   "Blue": "Blue",  "Brn":  "Brown",
    "G":    "Green", "Org":  "Orange","Pink": "Pink",
    "P":    "Purple","Pexp": "Purple","Y":    "Yellow",
}

# Station name → Chicago neighbourhood (best-effort; unmapped → empty string)
HOOD_MAP: dict[str, str] = {
    "Howard":             "Rogers Park",
    "Morse":              "Rogers Park",
    "Loyola":             "Rogers Park",
    "Granville":          "Edgewater",
    "Thorndale":          "Edgewater",
    "Bryn Mawr":          "Edgewater",
    "Berwyn":             "Andersonville",
    "Argyle":             "Uptown",
    "Lawrence":           "Uptown",
    "Wilson":             "Uptown",
    "Sheridan":           "Wrigleyville",
    "Addison":            "Wrigleyville",
    "Belmont":            "Lakeview",
    "Fullerton":          "Lincoln Park",
    "North/Clybourn":     "Old Town",
    "Clark/Division":     "Gold Coast",
    "Chicago":            "River North",
    "Grand":              "River North",
    "Lake":               "Loop",
    "Monroe":             "Loop",
    "Jackson":            "Loop",
    "Harrison":           "South Loop",
    "Cermak-Chinatown":   "Chinatown",
    "Sox-35th":           "Bridgeport",
    "O'Hare":             "O'Hare",
    "Rosemont":           "Rosemont",
    "Jefferson Park":     "Jefferson Park",
    "Logan Square":       "Logan Square",
    "California":         "Humboldt Park",
    "UIC-Halsted":        "Near West Side",
    "Forest Park":        "Forest Park",
    "Kimball":            "Albany Park",
    "Merchandise Mart":   "River North",
    "Midway":             "Midway",
    "Roosevelt":          "South Loop",
    "Davis":              "Evanston",
    "Dempster-Skokie":    "Skokie",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def nearest_venue(lat: float, lon: float) -> str | None:
    for key, (vlat, vlon) in VENUES.items():
        if haversine_miles(lat, lon, vlat, vlon) <= VENUE_RADIUS_MILES:
            return key
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Download GTFS
# ─────────────────────────────────────────────────────────────────────────────
def download_gtfs() -> None:
    GTFS_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded
    if (GTFS_DIR / "stops.txt").exists():
        print("[fetch] GTFS already present — skipping download")
        return

    print(f"[fetch] Downloading GTFS from {GTFS_URL} …")
    resp = requests.get(GTFS_URL, timeout=60)
    resp.raise_for_status()
    print(f"[fetch] Downloaded {len(resp.content) / 1e6:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            zf.extract(name, GTFS_DIR)
            print(f"[fetch]   extracted {name}  ({zf.getinfo(name).file_size:,} bytes)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Download ridership
# ─────────────────────────────────────────────────────────────────────────────
def download_ridership() -> None:
    out = RAW_DIR / "ridership.csv"
    if out.exists():
        print("[fetch] Ridership CSV already present — skipping download")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] Downloading ridership data …")
    resp = requests.get(RIDERSHIP_URL, timeout=60)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    rows = resp.content.count(b"\n")
    print(f"[fetch] Ridership saved → {out}  ({rows:,} rows, {out.stat().st_size // 1024} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Parse GTFS → station list with line assignments
# ─────────────────────────────────────────────────────────────────────────────
def parse_stations() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per parent L station.
    Columns: stop_id, stop_name, stop_lat, stop_lon, line
    """
    stops = pd.read_csv(GTFS_DIR / "stops.txt", dtype=str)
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)

    # Parent stations have location_type == 1
    # Some GTFS exports use int, some use str — normalise
    if "location_type" in stops.columns:
        stops["location_type"] = stops["location_type"].fillna("0")
        parents = stops[stops["location_type"].astype(str) == "1"].copy()
    else:
        # Fallback: use stop_ids that appear as parent_station for other stops
        child_stops = stops[stops["parent_station"].notna()]
        parent_ids  = child_stops["parent_station"].unique()
        parents     = stops[stops["stop_id"].isin(parent_ids)].copy()

    print(f"[fetch] Found {len(parents)} parent stations in stops.txt")

    # Map stop_id → primary route using stop_times + trips join
    print("[fetch] Mapping stations → L lines (reading stop_times.txt) …")
    trips    = pd.read_csv(GTFS_DIR / "trips.txt",      usecols=["trip_id", "route_id"], dtype=str)
    routes   = pd.read_csv(GTFS_DIR / "routes.txt",     usecols=["route_id", "route_short_name"], dtype=str)

    trip_to_route = dict(zip(trips["trip_id"], trips["route_id"]))

    # stop_times can be large — read in chunks, only keep stop_id + trip_id
    stop_route_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for chunk in pd.read_csv(
        GTFS_DIR / "stop_times.txt",
        usecols=["trip_id", "stop_id"],
        dtype=str,
        chunksize=100_000,
    ):
        for _, row in chunk.iterrows():
            route = trip_to_route.get(row["trip_id"])
            if route:
                stop_route_counts[row["stop_id"]][route] += 1

    # For child stops, roll up to parent
    child_to_parent = {}
    if "parent_station" in stops.columns:
        child_to_parent = dict(
            zip(
                stops[stops["parent_station"].notna()]["stop_id"],
                stops[stops["parent_station"].notna()]["parent_station"],
            )
        )

    parent_route_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for stop_id, rcounts in stop_route_counts.items():
        pid = child_to_parent.get(stop_id, stop_id)
        for route, cnt in rcounts.items():
            parent_route_counts[pid][route] += cnt

    # Assign primary line = most frequent route
    def primary_line(stop_id: str) -> str:
        rcounts = parent_route_counts.get(str(stop_id), {})
        if not rcounts:
            return "Unknown"
        best_route = max(rcounts, key=rcounts.get)
        return ROUTE_NAME_MAP.get(best_route, best_route)

    parents = parents.copy()
    parents["line"] = parents["stop_id"].apply(primary_line)

    # Filter out non-L entries (buses, etc.) — keep known line names only
    known_lines = set(ROUTE_NAME_MAP.values())
    parents = parents[parents["line"].isin(known_lines)].copy()
    print(f"[fetch] After line filtering: {len(parents)} L stations")

    return parents[["stop_id", "stop_name", "stop_lat", "stop_lon", "line"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Compute per-station ridership stats
# ─────────────────────────────────────────────────────────────────────────────
def compute_ridership_stats(stations: pd.DataFrame) -> dict[str, float]:
    """
    Returns {gtfs_stop_id: avg_weekday_rides} for every station we have data for.
    """
    ridership_path = RAW_DIR / "ridership.csv"
    if not ridership_path.exists():
        return {}

    df = pd.read_csv(ridership_path)
    print(f"[fetch] Ridership columns: {df.columns.tolist()}")

    # Normalise column names (Socrata exports vary)
    df.columns = [c.lower().strip() for c in df.columns]

    # station_id column
    sid_col = next((c for c in df.columns if "station_id" in c), None)
    # ridership column
    ride_col = next((c for c in df.columns if c in ("rides", "rail_boardings", "total_rides")), None)
    # day type column
    day_col = next((c for c in df.columns if "day" in c and "type" in c or c == "daytype"), None)

    if not ride_col or not sid_col:
        print(f"[fetch] WARNING: can't find station_id/rides columns, skipping stats")
        print(f"        Available: {df.columns.tolist()}")
        return {}

    df[ride_col] = pd.to_numeric(df[ride_col], errors="coerce")

    # Keep weekdays only if day_type column exists
    if day_col:
        df = df[df[day_col] == "W"]

    df[ride_col] = pd.to_numeric(df[ride_col], errors="coerce")
    stats = df.groupby(sid_col)[ride_col].mean().to_dict()
    return {str(k): float(v) for k, v in stats.items() if not math.isnan(v)}


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Build station_map.json
# ─────────────────────────────────────────────────────────────────────────────
def build_station_map(stations: pd.DataFrame, ridership_stats: dict[str, float]) -> dict:
    """
    Assigns compact integer token_ids (1-based) and enriches each station with:
      - sports venue proximity tag
      - neighbourhood label
      - average weekday ridership (0 if not available)
    """
    global_mean_rides = (
        sum(ridership_stats.values()) / len(ridership_stats)
        if ridership_stats else 1.0
    )

    records = []
    for token_id, (_, row) in enumerate(stations.iterrows(), start=1):
        stop_id  = str(row["stop_id"])
        name     = str(row["stop_name"]).strip()
        lat, lon = float(row["stop_lat"]), float(row["stop_lon"])
        line     = str(row["line"])

        sports = nearest_venue(lat, lon)

        # Neighbourhood: exact match, then strip "Station" suffix and retry
        hood = HOOD_MAP.get(name, "")
        if not hood:
            base = name.replace(" Station", "").strip()
            hood = HOOD_MAP.get(base, "")

        avg_rides   = ridership_stats.get(stop_id, global_mean_rides)
        # Normalise ridership to [0, 1] range relative to system mean
        # Values > 1.0 mean above-average crowding (used to scale delay in training)
        rides_norm  = avg_rides / global_mean_rides if global_mean_rides > 0 else 1.0

        records.append({
            "token_id":       token_id,
            "gtfs_stop_id":   stop_id,
            "name":           name,
            "line":           line,
            "lat":            round(lat, 6),
            "lon":            round(lon, 6),
            "hood":           hood,
            "sports":         sports,
            "avg_weekday_rides": round(avg_rides, 1),
            "rides_norm":     round(rides_norm, 4),
        })

    return {
        "num_stations": len(records),
        "stations":     records,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    download_gtfs()
    download_ridership()

    stations       = parse_stations()
    ridership_stats = compute_ridership_stats(stations)
    print(f"[fetch] Ridership stats available for {len(ridership_stats)} stations")

    station_map = build_station_map(stations, ridership_stats)

    out = ROOT / "data" / "station_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(station_map, f, indent=2)

    n = station_map["num_stations"]
    print(f"\n[fetch] station_map.json → {out}")
    print(f"         {n} stations  |  token IDs 1–{n}")

    # Summary
    from collections import Counter
    lines = Counter(s["line"] for s in station_map["stations"])
    sports_tagged = sum(1 for s in station_map["stations"] if s["sports"])
    rides_covered = sum(1 for s in station_map["stations"] if s["avg_weekday_rides"] > 0)

    print(f"\n── Line distribution ───────────────────────────────────")
    for line, cnt in sorted(lines.items()):
        print(f"  {line:<10} {cnt} stations")
    print(f"\n  Sports-tagged: {sports_tagged}")
    print(f"  Has ridership: {rides_covered}/{n}")
    print(f"\n── Sample stations ─────────────────────────────────────")
    for s in station_map["stations"][:5]:
        print(f"  [{s['token_id']:>3}] {s['name']:<30} {s['line']:<8} "
              f"rides={s['avg_weekday_rides']:>8,.0f}  sports={s['sports']}")


if __name__ == "__main__":
    main()
