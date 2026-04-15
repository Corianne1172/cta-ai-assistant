"""
fetch_sports.py
---------------
Downloads home game schedules for Chicago's four major sports teams.

Sources:
  MLB Stats API (official, free)  — Cubs + White Sox
  ESPN unofficial API (free)      — Bears + Bulls

Output: data/raw/sports_schedule.csv
Columns: date, game_date_local, team, venue, sports_key

Run:
    python scripts/fetch_sports.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "data" / "raw" / "sports_schedule.csv"

YEARS = [2022, 2023]

# ─────────────────────────────────────────────────────────────────────────────
# MLB — official stats API, no key required
# ─────────────────────────────────────────────────────────────────────────────
MLB_TEAMS = {
    112: ("Chicago Cubs",       "Wrigley Field",           "cubs"),
    145: ("Chicago White Sox",  "Guaranteed Rate Field",   "sox"),
}

def fetch_mlb(team_id: int, year: int) -> list[dict]:
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?teamId={team_id}&season={year}&sportId=1&gameType=R"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    _, venue, sports_key = MLB_TEAMS[team_id]
    games = []
    for date_entry in resp.json().get("dates", []):
        for game in date_entry.get("games", []):
            # Only home games
            if game["teams"]["home"]["team"]["id"] != team_id:
                continue
            # Game time is UTC ISO string: "2022-04-07T18:10:00Z"
            game_dt_utc = game.get("gameDate", "")
            try:
                dt_utc   = datetime.fromisoformat(game_dt_utc.replace("Z", "+00:00"))
                dt_local = dt_utc.astimezone(
                    timezone.utc
                ).strftime("%Y-%m-%dT%H:%M")
            except Exception:
                dt_local = date_entry["date"] + "T18:00"

            games.append({
                "date":            date_entry["date"],
                "game_date_local": dt_local,
                "team":            MLB_TEAMS[team_id][0],
                "venue":           venue,
                "sports_key":      sports_key,
            })
    return games


# ─────────────────────────────────────────────────────────────────────────────
# ESPN unofficial API — Bears + Bulls
# ─────────────────────────────────────────────────────────────────────────────
ESPN_TEAMS = {
    # (sport, league, espn_team_id, team_name, venue, sports_key)
    "bears": ("football",   "nfl", 3,  "Chicago Bears", "Soldier Field",    "bears"),
    "bulls": ("basketball", "nba", 4,  "Chicago Bulls", "United Center",    "bulls_hawks"),
}

def fetch_espn(key: str, year: int) -> list[dict]:
    sport, league, team_id, team_name, venue, sports_key = ESPN_TEAMS[key]
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports"
        f"/{sport}/{league}/teams/{team_id}/schedule?season={year}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    body = resp.json()

    games = []
    for event in body.get("events", []):
        competitions = event.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]
        competitors = comp.get("competitors", [])

        # Only home games
        home_team = next(
            (c for c in competitors if c.get("homeAway") == "home"),
            None,
        )
        if not home_team:
            continue
        # Check it's actually the Chicago team
        home_name = home_team.get("team", {}).get("displayName", "")
        if team_name not in home_name and key not in home_name.lower():
            continue

        raw_date = event.get("date", "")  # UTC ISO string
        try:
            dt_utc   = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            date_str = dt_utc.strftime("%Y-%m-%d")
            dt_local = dt_utc.strftime("%Y-%m-%dT%H:%M")
        except Exception:
            date_str = raw_date[:10]
            dt_local = raw_date[:16]

        games.append({
            "date":            date_str,
            "game_date_local": dt_local,
            "team":            team_name,
            "venue":           venue,
            "sports_key":      sports_key,
        })
    return games


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if OUT.exists():
        print(f"[sports] Already downloaded — {OUT}")
        _print_stats()
        return

    all_games: list[dict] = []

    # MLB
    for team_id, (name, _, _) in MLB_TEAMS.items():
        for year in YEARS:
            print(f"[sports] Fetching {name} {year} schedule …", end=" ")
            try:
                games = fetch_mlb(team_id, year)
                all_games.extend(games)
                print(f"{len(games)} home games")
            except Exception as e:
                print(f"FAILED: {e}")

    # ESPN
    for key in ["bears", "bulls"]:
        for year in YEARS:
            _, _, _, name, _, _ = ESPN_TEAMS[key]
            print(f"[sports] Fetching {name} {year} schedule …", end=" ")
            try:
                games = fetch_espn(key, year)
                all_games.extend(games)
                print(f"{len(games)} home games")
            except Exception as e:
                print(f"FAILED (non-critical): {e}")

    df = pd.DataFrame(all_games).drop_duplicates(subset=["date", "team"])
    df = df.sort_values("date").reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n[sports] Saved → {OUT}  ({len(df)} total home games)")
    _print_stats(df)


def _print_stats(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = pd.read_csv(OUT)
    print(f"\n── Home games by team ──────────────────────────────────")
    print(df.groupby(["team","sports_key"]).size().to_string())
    print(f"\n── Sample ──────────────────────────────────────────────")
    print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
