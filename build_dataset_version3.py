#!/usr/bin/env python3
"""
Assemble Dataset_Version_3.csv by enriching the existing Dataset.csv with
shot-based features and pre-match Elo ratings generated in the
afdf0c3..e4e72bb data refresh.

The script keeps all original columns and appends:
    - home_shots_for
    - away_shots_for
    - elo_home_pre
    - elo_away_pre
    - elo_home_expectation

Columns are only populated when corresponding source data exists; otherwise
they remain blank so downstream pipelines can guard against missingness.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

BASE_DIR = Path("understat_data")
SOURCE_DATASET = BASE_DIR / "Dataset.csv"
OUTPUT_DATASET = BASE_DIR / "Dataset_Version_3.csv"

TEAM_RESULTS_SUBPATH = Path("Team_Results") / "team_results.csv"
TEAM_ELO_TIMESERIES_SUBPATH = Path("Team_Results") / "team_elos_timeseries.csv"

ShotsKey = Tuple[str, str, str]  # (league, match_id, team_name)
EloKey = Tuple[str, str]  # (league, match_id)


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_team_shot_counts() -> Dict[ShotsKey, Dict[str, Optional[int]]]:
    shot_map: Dict[ShotsKey, Dict[str, Optional[int]]] = {}
    for league_dir in BASE_DIR.iterdir():
        if not league_dir.is_dir():
            continue
        results_path = league_dir / TEAM_RESULTS_SUBPATH
        if not results_path.exists():
            continue

        league = league_dir.name
        with open(results_path, newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                match_id = row.get("match_id")
                team_name = row.get("team")
                if not match_id or not team_name:
                    continue

                shots_for = _safe_int(row.get("shots_for"))
                shots_against = _safe_int(row.get("shots_against"))
                shot_map[(league, match_id, team_name)] = {
                    "shots_for": shots_for,
                    "shots_against": shots_against,
                }
    return shot_map


def load_elo_timeseries() -> Dict[EloKey, Dict[str, Optional[float]]]:
    elo_map: Dict[EloKey, Dict[str, Optional[float]]] = {}
    for league_dir in BASE_DIR.iterdir():
        if not league_dir.is_dir():
            continue
        elo_path = league_dir / TEAM_ELO_TIMESERIES_SUBPATH
        if not elo_path.exists():
            continue

        league = league_dir.name
        with open(elo_path, newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                match_id = row.get("match_id")
                if not match_id:
                    continue
                elo_map[(league, match_id)] = {
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "home_elo_pre": _safe_float(row.get("home_elo_pre")),
                    "away_elo_pre": _safe_float(row.get("away_elo_pre")),
                    "elo_expectation_home": _safe_float(row.get("E_home")),
                }
    return elo_map


def main() -> None:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Base dataset missing at {SOURCE_DATASET}")

    shot_map = load_team_shot_counts()
    elo_map = load_elo_timeseries()

    with open(SOURCE_DATASET, newline="") as fp:
        reader = csv.DictReader(fp)
        base_rows = list(reader)
        base_fieldnames = reader.fieldnames or []

    new_columns = [
        "home_shots_for",
        "away_shots_for",
        "elo_home_pre",
        "elo_away_pre",
        "elo_home_expectation",
    ]

    enriched_fieldnames = base_fieldnames + [col for col in new_columns if col not in base_fieldnames]

    missing_shots = 0
    missing_elo = 0

    for row in base_rows:
        league = row.get("league", "").strip()
        match_id = row.get("match_id", "").strip()
        home_team = row.get("home_team_name", "").strip()
        away_team = row.get("away_team_name", "").strip()

        # Team shot counts
        home_shots = shot_map.get((league, match_id, home_team))
        away_shots = shot_map.get((league, match_id, away_team))

        if home_shots and home_shots.get("shots_for") is not None:
            row["home_shots_for"] = str(home_shots["shots_for"])
        else:
            row["home_shots_for"] = ""
            missing_shots += 1

        if away_shots and away_shots.get("shots_for") is not None:
            row["away_shots_for"] = str(away_shots["shots_for"])
        else:
            row["away_shots_for"] = ""
            missing_shots += 1

        # Elo features
        elo_key = (league, match_id)
        elo_entry = elo_map.get(elo_key)

        if elo_entry and (
            (not elo_entry.get("home_team") or elo_entry["home_team"] == home_team)
            and (not elo_entry.get("away_team") or elo_entry["away_team"] == away_team)
        ):
            row["elo_home_pre"] = (
                f"{elo_entry['home_elo_pre']:.3f}" if elo_entry.get("home_elo_pre") is not None else ""
            )
            row["elo_away_pre"] = (
                f"{elo_entry['away_elo_pre']:.3f}" if elo_entry.get("away_elo_pre") is not None else ""
            )
            row["elo_home_expectation"] = (
                f"{elo_entry['elo_expectation_home']:.6f}"
                if elo_entry.get("elo_expectation_home") is not None
                else ""
            )
        else:
            row["elo_home_pre"] = ""
            row["elo_away_pre"] = ""
            row["elo_home_expectation"] = ""
            missing_elo += 1

    with open(OUTPUT_DATASET, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=enriched_fieldnames)
        writer.writeheader()
        writer.writerows(base_rows)

    print(f"Enriched dataset written to {OUTPUT_DATASET}")
    print(f"Rows processed: {len(base_rows)}")
    print(f"Shot features missing in {missing_shots} team entries")
    print(f"Elo features missing in {missing_elo} matches")


if __name__ == "__main__":
    main()
