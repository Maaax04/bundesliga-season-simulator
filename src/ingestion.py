import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import CFG

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _is_cache_valid(filepath: Path, ttl_hours: int) -> bool:
    if not filepath.exists():
        return False
    age_seconds = time.time() - filepath.stat().st_mtime
    return age_seconds < ttl_hours * 3600


def fetch_league_overview(force_refresh: bool = False) -> dict:
    if not force_refresh and _is_cache_valid(CFG.RAW_LEAGUE_FILE, CFG.CACHE_TTL_HOURS):
        with open(CFG.RAW_LEAGUE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{CFG.FOTMOB_BASE_URL}/leagues"
    resp = requests.get(url, headers=HEADERS, params={"id": CFG.LEAGUE_ID}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    with open(CFG.RAW_LEAGUE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def fetch_match_details(match_id: int) -> dict:
    cache_file = CFG.MATCH_DETAILS_DIR / f"{match_id}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{CFG.FOTMOB_BASE_URL}/matchDetails"
    resp = requests.get(url, headers=HEADERS, params={"matchId": match_id}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    time.sleep(CFG.REQUEST_DELAY_SEC)
    return data


def _extract_match_xg(match_details: dict) -> tuple[float, float]:
    stats_groups = (
        match_details.get("content", {})
        .get("stats", {})
        .get("Periods", {})
        .get("All", {})
        .get("stats", [])
    )
    for group in stats_groups:
        for item in group.get("stats", []):
            if item.get("key") == "expected_goals" and item.get("type") != "title":
                raw = item.get("rawStats") or []
                if len(raw) == 2 and raw[0].get("value") is not None and raw[1].get("value") is not None:
                    return float(raw[0]["value"]), float(raw[1]["value"])
    raise ValueError("Could not extract 'expected_goals' from matchDetails JSON.")


def load_and_prepare_data(
        force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    CFG.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    CFG.MATCH_DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    CFG.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    league_data = fetch_league_overview(force_refresh=force_refresh)
    all_matches = league_data.get("fixtures", {}).get("allMatches", [])

    played_records = []
    remaining_records = []

    finished_matches = [
        m for m in all_matches
        if m.get("status", {}).get("finished") and not m.get("status", {}).get("cancelled")
    ]
    print(f"Syncing match details for {len(finished_matches)} finished matches (cached matches load instantly)...")

    for match in all_matches:
        match_id = int(match["id"])
        status = match.get("status", {})
        is_finished = bool(status.get("finished")) and not bool(status.get("cancelled"))

        base_row = {
            "match_id": match_id,
            "matchday": int(match["round"]),
            "date": status["utcTime"],
            "home_team": match["home"]["name"],
            "away_team": match["away"]["name"],
        }

        if is_finished:
            score_parts = status["scoreStr"].split("-")
            home_goals = int(score_parts[0].strip())
            away_goals = int(score_parts[1].strip())

            details = fetch_match_details(match_id)
            home_xg, away_xg = _extract_match_xg(details)

            played_records.append(
                {
                    **base_row,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_xg": home_xg,
                    "away_xg": away_xg,
                }
            )
        else:
            remaining_records.append(base_row)

    played_df = pd.DataFrame(played_records)
    remaining_df = pd.DataFrame(remaining_records)

    played_df["date"] = pd.to_datetime(played_df["date"], format="ISO8601", utc=True)
    remaining_df["date"] = pd.to_datetime(remaining_df["date"], format="ISO8601", utc=True)

    played_df = played_df.sort_values(["date", "match_id"]).reset_index(drop=True)
    remaining_df = remaining_df.sort_values(["date", "match_id"]).reset_index(drop=True)

    # Sanity checks
    total_matches = len(played_df) + len(remaining_df)
    assert total_matches == 306, f"Expected 306 matches, found: {total_matches}"
    unique_teams = set(played_df["home_team"]).union(set(remaining_df["home_team"]))
    assert len(unique_teams) == 18, f"Expected 18 teams, found: {len(unique_teams)}"

    played_df.to_csv(CFG.PLAYED_MATCHES_FILE, index=False)
    remaining_df.to_csv(CFG.REMAINING_MATCHES_FILE, index=False)

    return played_df, remaining_df


if __name__ == "__main__":
    played, remaining = load_and_prepare_data()
    print(f"\nDone! Played matches: {len(played)} | Remaining matches: {len(remaining)}")
    print("\nLast 5 played matches:")
    print(played.tail())
