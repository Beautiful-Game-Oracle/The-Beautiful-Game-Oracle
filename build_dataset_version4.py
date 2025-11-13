#!/usr/bin/env python3
"""
Build Dataset_Version_4.csv by extending the Dataset_Version_3 enrichment flow
with Elo V2 artefacts for every league directory under understat_data/.

Steps:
    1) Start from understat_data/Dataset.csv (baseline curated table).
    2) Merge team shot counts sourced from Team_Results/*.csv files.
    3) Reconstruct pre-match Elo + expectation values from
       Team_Results/team_elos_timeseries.csv (produced by getTeamEloV2.py).
    4) Join per-team summary ratings from team_elos_v2.csv so downstream
       experiments can access the refreshed league standings.

Columns are left blank when the upstream file lacks coverage so modelling code
can treat missingness explicitly.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

BASE_DIR = Path("understat_data")
SOURCE_DATASET = BASE_DIR / "Dataset.csv"
OUTPUT_DATASET = BASE_DIR / "Dataset_Version_4.csv"

TEAM_RESULTS_SUBPATH = Path("Team_Results") / "team_results.csv"
TEAM_ELO_TIMESERIES_SUBPATH = Path("Team_Results") / "team_elos_timeseries.csv"
TEAM_ELO_SUMMARY_FILENAME = "team_elos_v2.csv"

ShotsKey = Tuple[str, str, str]  # (league, match_id, team_name)
EloKey = Tuple[str, str]  # (league, match_id)
SummaryKey = Tuple[str, str]  # (league, team_name)


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


def _format_float(value: Optional[float], decimals: int = 3) -> str:
    if value is None:
        return ""
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(value)


def _score_from_goals(home_goals: Optional[int], away_goals: Optional[int]) -> Optional[float]:
    if home_goals is None or away_goals is None:
        return None
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


def _points_pct(wins: Optional[int], draws: Optional[int], played: Optional[int]) -> Optional[float]:
    if wins is None or draws is None or played in (None, 0):
        return None
    total_points = wins * 3 + draws
    return total_points / (played * 3)


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
                match_key = (league, str(match_id).strip(), team_name.strip())
                shot_map[match_key] = {
                    "shots_for": _safe_int(row.get("shots_for")),
                    "shots_against": _safe_int(row.get("shots_against")),
                }
    return shot_map


def load_elo_timeseries() -> Dict[EloKey, Dict[str, Optional[float]]]:
    """
    The V2 timeseries exports post-match ratings and metadata. Reconstruct the
    pre-match ratings by reversing the final update.
    """
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

                home_goals = _safe_int(row.get("home_goals"))
                away_goals = _safe_int(row.get("away_goals"))
                expectation = None
                p_home = _safe_float(row.get("p_home"))
                p_draw = _safe_float(row.get("p_draw"))
                if p_home is not None and p_draw is not None:
                    expectation = p_home + 0.5 * p_draw

                k_eff = _safe_float(row.get("k_eff"))
                actual = _score_from_goals(home_goals, away_goals)

                home_post = _safe_float(row.get("home_elo_post"))
                away_post = _safe_float(row.get("away_elo_post"))
                delta = None
                if expectation is not None and actual is not None and k_eff is not None:
                    delta = k_eff * (actual - expectation)

                home_pre = away_pre = None
                if delta is not None and home_post is not None and away_post is not None:
                    home_pre = home_post - delta
                    away_pre = away_post + delta

                elo_map[(league, str(match_id).strip())] = {
                    "home_team": row.get("home_team", "").strip(),
                    "away_team": row.get("away_team", "").strip(),
                    "home_elo_pre": home_pre,
                    "away_elo_pre": away_pre,
                    "elo_expectation_home": expectation,
                }
    return elo_map


def load_elo_summary() -> Dict[SummaryKey, Dict[str, Optional[float]]]:
    summary: Dict[SummaryKey, Dict[str, Optional[float]]] = {}
    for league_dir in BASE_DIR.iterdir():
        if not league_dir.is_dir():
            continue
        summary_path = league_dir / TEAM_ELO_SUMMARY_FILENAME
        if not summary_path.exists():
            continue

        league = league_dir.name
        with open(summary_path, newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                team = row.get("team")
                if not team:
                    continue

                played = _safe_int(row.get("played"))
                wins = _safe_int(row.get("wins"))
                draws = _safe_int(row.get("draws"))
                losses = _safe_int(row.get("losses"))

                summary[(league, team.strip())] = {
                    "final_elo": _safe_float(row.get("final_elo")),
                    "played": played,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "points_pct": _points_pct(wins, draws, played),
                }
    return summary


def main() -> None:
    if not SOURCE_DATASET.exists():
        raise FileNotFoundError(f"Base dataset missing at {SOURCE_DATASET}")

    shot_map = load_team_shot_counts()
    elo_map = load_elo_timeseries()
    elo_summary = load_elo_summary()

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
        "home_elo_final",
        "home_elo_matches_played",
        "home_elo_wins",
        "home_elo_draws",
        "home_elo_losses",
        "home_elo_points_pct",
        "away_elo_final",
        "away_elo_matches_played",
        "away_elo_wins",
        "away_elo_draws",
        "away_elo_losses",
        "away_elo_points_pct",
    ]

    enriched_fieldnames = base_fieldnames + [col for col in new_columns if col not in base_fieldnames]

    missing_shots = 0
    missing_elo = 0
    missing_summary_home = 0
    missing_summary_away = 0

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

        # Match-level Elo features
        elo_key = (league, match_id)
        elo_entry = elo_map.get(elo_key)
        if elo_entry and (
            (not elo_entry.get("home_team") or elo_entry["home_team"] == home_team)
            and (not elo_entry.get("away_team") or elo_entry["away_team"] == away_team)
        ):
            row["elo_home_pre"] = _format_float(elo_entry.get("home_elo_pre"))
            row["elo_away_pre"] = _format_float(elo_entry.get("away_elo_pre"))
            row["elo_home_expectation"] = _format_float(elo_entry.get("elo_expectation_home"), decimals=6)
        else:
            row["elo_home_pre"] = ""
            row["elo_away_pre"] = ""
            row["elo_home_expectation"] = ""
            missing_elo += 1

        # Team summary Elo features
        home_summary = elo_summary.get((league, home_team))
        if home_summary:
            row["home_elo_final"] = _format_float(home_summary.get("final_elo"), decimals=2)
            row["home_elo_matches_played"] = (
                str(home_summary["played"]) if home_summary.get("played") is not None else ""
            )
            row["home_elo_wins"] = str(home_summary["wins"]) if home_summary.get("wins") is not None else ""
            row["home_elo_draws"] = str(home_summary["draws"]) if home_summary.get("draws") is not None else ""
            row["home_elo_losses"] = str(home_summary["losses"]) if home_summary.get("losses") is not None else ""
            row["home_elo_points_pct"] = _format_float(home_summary.get("points_pct"), decimals=4)
        else:
            row["home_elo_final"] = ""
            row["home_elo_matches_played"] = ""
            row["home_elo_wins"] = ""
            row["home_elo_draws"] = ""
            row["home_elo_losses"] = ""
            row["home_elo_points_pct"] = ""
            missing_summary_home += 1

        away_summary = elo_summary.get((league, away_team))
        if away_summary:
            row["away_elo_final"] = _format_float(away_summary.get("final_elo"), decimals=2)
            row["away_elo_matches_played"] = (
                str(away_summary["played"]) if away_summary.get("played") is not None else ""
            )
            row["away_elo_wins"] = str(away_summary["wins"]) if away_summary.get("wins") is not None else ""
            row["away_elo_draws"] = str(away_summary["draws"]) if away_summary.get("draws") is not None else ""
            row["away_elo_losses"] = str(away_summary["losses"]) if away_summary.get("losses") is not None else ""
            row["away_elo_points_pct"] = _format_float(away_summary.get("points_pct"), decimals=4)
        else:
            row["away_elo_final"] = ""
            row["away_elo_matches_played"] = ""
            row["away_elo_wins"] = ""
            row["away_elo_draws"] = ""
            row["away_elo_losses"] = ""
            row["away_elo_points_pct"] = ""
            missing_summary_away += 1

    with open(OUTPUT_DATASET, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=enriched_fieldnames)
        writer.writeheader()
        writer.writerows(base_rows)

    print(f"Enriched dataset written to {OUTPUT_DATASET}")
    print(f"Rows processed: {len(base_rows)}")
    print(f"Shot features missing in {missing_shots} team entries")
    print(f"Elo features missing in {missing_elo} matches")
    print(f"Elo summary missing for {missing_summary_home} home teams")
    print(f"Elo summary missing for {missing_summary_away} away teams")


if __name__ == "__main__":
    main()
