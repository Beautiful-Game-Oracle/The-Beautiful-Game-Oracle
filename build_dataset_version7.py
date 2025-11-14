#!/usr/bin/env python3
"""
Build Dataset_Version_7.csv by extending the Dataset_Version_4 enrichment flow
with cleaner market-vs-Elo comparisons, volatility diagnostics, and ready-to-use Elo gap features.

Steps:
    1) Rebuild the EPL base table directly from season-level exports inside
       understat_data/EPL/<season>/league_results.csv so new seasons + fixtures
       are always captured without relying on collated CSVs.
    2) Merge team shot counts sourced from Team_Results/*.csv files.
    3) Reconstruct pre-match Elo + expectation values from
       Team_Results/team_elos_timeseries.csv (produced by getTeamEloV2.py).
    4) Join per-team summary ratings from team_elos_v2.csv so downstream
       experiments can access the refreshed league standings.
    5) Add season-normalised Elo gaps plus a clipped market-vs-Elo edge column
       to stabilise modelling inputs.

Columns are left blank when the upstream file lacks coverage so modelling code
can treat missingness explicitly.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from analysis import build_league_results_v2 as league_v2

BASE_DIR = Path("understat_data")
OUTPUT_DATASET = BASE_DIR / "Dataset_Version_7.csv"
TARGET_LEAGUE = "EPL"
LEAGUE_ROOT = BASE_DIR / TARGET_LEAGUE

VOLATILITY_WINDOW = 5
EXP_DECAY_ALPHA = 0.55

TEAM_RESULTS_SUBPATH = Path("Team_Results") / "team_results.csv"
TEAM_ELO_TIMESERIES_SUBPATH = Path("Team_Results") / "team_elos_timeseries.csv"
TEAM_ELO_SUMMARY_FILENAME = "team_elos_v2.csv"

ShotsKey = Tuple[str, str, str]  # (league, match_id, team_name)
EloKey = Tuple[str, str]  # (league, match_id)
SummaryKey = Tuple[str, str]  # (league, team_name)


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        return int(float(value_str))
    except ValueError:
        return None


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        return float(value_str)
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


def _parse_nested(value: object) -> Dict[str, object]:
    if value in ("", None):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            try:
                return json.loads(text.replace("'", '"'))
            except (json.JSONDecodeError, TypeError):
                return {}
    return {}


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _parse_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _derive_outcome(home_goals: int, away_goals: int) -> Tuple[str, str, int, int, int]:
    if home_goals > away_goals:
        return "Home Win", "H", 1, 0, 0
    if home_goals < away_goals:
        return "Away Win", "A", 0, 0, 1
    return "Draw", "D", 0, 1, 0


def _build_match_record(row: Dict[str, object], season_label: str) -> Optional[Dict[str, object]]:
    match_id = _safe_int(row.get("id"))
    if match_id is None:
        return None

    if not _parse_bool(row.get("isResult")):
        return None

    season_val = _safe_int(season_label)
    if season_val is None:
        return None

    match_dt = _parse_datetime(row.get("datetime"))
    match_dt_str = match_dt.strftime("%Y-%m-%d %H:%M:%S") if match_dt else str(row.get("datetime", "")).strip()
    match_date_str = match_dt.date().isoformat() if match_dt else ""
    match_time_str = match_dt.time().isoformat() if match_dt else ""

    home_meta = _parse_nested(row.get("h"))
    away_meta = _parse_nested(row.get("a"))
    goals_meta = _parse_nested(row.get("goals"))
    xg_meta = _parse_nested(row.get("xG"))
    forecast_meta = _parse_nested(row.get("forecast"))

    home_goals = _safe_int(goals_meta.get("h")) if goals_meta else None
    away_goals = _safe_int(goals_meta.get("a")) if goals_meta else None
    home_xg = _safe_float(xg_meta.get("h")) if xg_meta else None
    away_xg = _safe_float(xg_meta.get("a")) if xg_meta else None

    home_goals = 0 if home_goals is None else home_goals
    away_goals = 0 if away_goals is None else away_goals
    home_xg = 0.0 if home_xg is None else home_xg
    away_xg = 0.0 if away_xg is None else away_xg

    match_outcome, outcome_code, home_flag, draw_flag, away_flag = _derive_outcome(home_goals, away_goals)

    forecast_home = _safe_float(forecast_meta.get("w")) if forecast_meta else None
    forecast_draw = _safe_float(forecast_meta.get("d")) if forecast_meta else None
    forecast_away = _safe_float(forecast_meta.get("l")) if forecast_meta else None

    return {
        "match_id": match_id,
        "league": TARGET_LEAGUE,
        "season": season_val,
        "match_datetime_utc": match_dt_str,
        "match_date": match_date_str,
        "match_time": match_time_str,
        "is_result": True,
        "home_team_id": _safe_int(home_meta.get("id")) if home_meta else None,
        "home_team_name": home_meta.get("title", "") if home_meta else "",
        "home_team_short": home_meta.get("short_title", "") if home_meta else "",
        "away_team_id": _safe_int(away_meta.get("id")) if away_meta else None,
        "away_team_name": away_meta.get("title", "") if away_meta else "",
        "away_team_short": away_meta.get("short_title", "") if away_meta else "",
        "home_goals": home_goals,
        "away_goals": away_goals,
        "total_goals": home_goals + away_goals,
        "goal_difference": home_goals - away_goals,
        "home_xg": round(home_xg, 6),
        "away_xg": round(away_xg, 6),
        "xg_difference": round(home_xg - away_xg, 6),
        "forecast_home_win": forecast_home if forecast_home is not None else 0.0,
        "forecast_draw": forecast_draw if forecast_draw is not None else 0.0,
        "forecast_away_win": forecast_away if forecast_away is not None else 0.0,
        "match_outcome": match_outcome,
        "match_outcome_code": outcome_code,
        "home_win_flag": home_flag,
        "draw_flag": draw_flag,
        "away_win_flag": away_flag,
    }


def _collect_league_results() -> List[Dict[str, object]]:
    if not LEAGUE_ROOT.exists():
        raise FileNotFoundError(f"League directory missing at {LEAGUE_ROOT}")

    records: List[Dict[str, object]] = []
    for season_dir in sorted(LEAGUE_ROOT.iterdir()):
        if not season_dir.is_dir():
            continue
        season_label = season_dir.name
        if not season_label.isdigit():
            continue
        results_path = season_dir / "league_results.csv"
        if not results_path.exists():
            continue
        with open(results_path, newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                record = _build_match_record(row, season_label)
                if record:
                    records.append(record)
    if not records:
        raise RuntimeError(f"No league results found under {LEAGUE_ROOT}")
    return records


def _load_cleaned_league_results() -> Tuple[List[Dict[str, object]], List[str]]:
    records = _collect_league_results()
    df = pd.DataFrame(records)

    df["match_datetime_utc"] = pd.to_datetime(df["match_datetime_utc"])
    df["match_date"] = df["match_datetime_utc"].dt.normalize()
    df["match_weekday"] = df["match_date"].dt.day_name()
    df["season"] = df["season"].astype(int)

    numeric_cols = [
        "home_goals",
        "away_goals",
        "total_goals",
        "goal_difference",
        "home_xg",
        "away_xg",
        "xg_difference",
        "forecast_home_win",
        "forecast_draw",
        "forecast_away_win",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("match_datetime_utc").reset_index(drop=True)

    long_df = league_v2.compute_team_view(df.copy())
    long_df = league_v2.add_rolling_features(long_df)
    enriched = league_v2.pivot_features(df.copy(), long_df)
    enriched = league_v2.add_market_features(enriched)
    enriched = league_v2.add_targets(enriched)
    enriched = league_v2.add_momentum_standardisation(enriched)
    enriched = league_v2.prune_inference_columns(enriched)
    enriched = league_v2.reorder_columns(enriched)

    for col, fmt in (("match_datetime_utc", "%Y-%m-%d %H:%M:%S"), ("match_date", "%Y-%m-%d")):
        if col in enriched.columns and pd.api.types.is_datetime64_any_dtype(enriched[col]):
            enriched[col] = enriched[col].dt.strftime(fmt)

    output_records: List[Dict[str, object]] = []
    for record in enriched.to_dict(orient="records"):
        cleaned_record: Dict[str, object] = {}
        for key, value in record.items():
            if pd.isna(value):
                cleaned_record[key] = ""
            else:
                cleaned_record[key] = value
        output_records.append(cleaned_record)

    return output_records, list(enriched.columns)


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
    base_rows, base_fieldnames = _load_cleaned_league_results()

    base_rows.sort(key=lambda r: (str(r.get("match_datetime_utc", "")), str(r.get("match_id", ""))))

    shot_map = load_team_shot_counts()
    elo_map = load_elo_timeseries()
    elo_summary = load_elo_summary()

    new_columns = [
        "home_shots_for",
        "away_shots_for",
        "elo_home_pre",
        "elo_away_pre",
        "elo_home_expectation",
        "elo_mean_pre",
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
        "elo_gap_pre",
        "elo_expectation_gap",
        "elo_gap_pre_season_z",
        "elo_expectation_gap_season_z",
        "market_vs_elo_edge",
        "home_goal_diff_std5",
        "away_goal_diff_std5",
        "goal_diff_std_gap5",
        "home_goal_diff_exp_decay",
        "away_goal_diff_exp_decay",
        "goal_diff_exp_decay_gap",
        "home_xg_diff_std5",
        "away_xg_diff_std5",
        "xg_diff_std_gap5",
        "home_xg_diff_exp_decay",
        "away_xg_diff_exp_decay",
        "xg_diff_exp_decay_gap",
        "home_shot_diff_std5",
        "away_shot_diff_std5",
        "shot_diff_std_gap5",
        "home_shot_diff_exp_decay",
        "away_shot_diff_exp_decay",
        "shot_diff_exp_decay_gap",
    ]

    enriched_fieldnames = base_fieldnames + [col for col in new_columns if col not in base_fieldnames]

    missing_shots = 0
    missing_elo = 0
    missing_summary_home = 0
    missing_summary_away = 0

    # Temporary containers for season-level stats
    gap_values: Dict[Tuple[str, str], list[float]] = {}
    expectation_gap_values: Dict[Tuple[str, str], list[float]] = {}

    def _register_value(bucket: Dict[Tuple[str, str], list[float]], key: Tuple[str, str], value: float) -> None:
        bucket.setdefault(key, []).append(value)

    def _rolling_std(values: deque[float]) -> float:
        if not values:
            return 0.0
        mean_val = sum(values) / len(values)
        variance = sum((val - mean_val) ** 2 for val in values) / len(values)
        return variance ** 0.5

    def _current_volatility(
        team: str,
        history: Dict[str, deque],
        exp_avgs: Dict[str, float],
    ) -> Tuple[Optional[float], Optional[float]]:
        buffer = history.get(team)
        std_val = _rolling_std(buffer) if buffer and len(buffer) > 0 else None
        exp_val = exp_avgs.get(team)
        return std_val, exp_val

    def _append_volatility_value(
        team: str,
        value: Optional[float],
        history: Dict[str, deque],
        exp_avgs: Dict[str, float],
    ) -> None:
        if value is None:
            return
        buffer = history.setdefault(team, deque(maxlen=VOLATILITY_WINDOW))
        buffer.append(value)
        prev = exp_avgs.get(team)
        if prev is None:
            exp_avgs[team] = value
        else:
            exp_avgs[team] = EXP_DECAY_ALPHA * value + (1.0 - EXP_DECAY_ALPHA) * prev

    goal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=VOLATILITY_WINDOW))
    xg_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=VOLATILITY_WINDOW))
    shot_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=VOLATILITY_WINDOW))
    goal_exp_avgs: Dict[str, float] = {}
    xg_exp_avgs: Dict[str, float] = {}
    shot_exp_avgs: Dict[str, float] = {}

    for row in base_rows:
        league = str(row.get("league", "")).strip()
        match_id = str(row.get("match_id", "")).strip()
        home_team = str(row.get("home_team_name", "")).strip()
        away_team = str(row.get("away_team_name", "")).strip()
        season = str(row.get("season", "")).strip()

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

        home_shot_diff = None
        away_shot_diff = None
        if home_shots:
            shots_for = home_shots.get("shots_for")
            shots_against = home_shots.get("shots_against")
            if shots_for is not None and shots_against is not None:
                home_shot_diff = shots_for - shots_against
        if away_shots:
            shots_for = away_shots.get("shots_for")
            shots_against = away_shots.get("shots_against")
            if shots_for is not None and shots_against is not None:
                away_shot_diff = shots_for - shots_against

        # Match-level Elo features
        elo_key = (league, match_id)
        elo_entry = elo_map.get(elo_key)
        gap_value = None
        expect_gap_value = None
        if elo_entry and (
            (not elo_entry.get("home_team") or elo_entry["home_team"] == home_team)
            and (not elo_entry.get("away_team") or elo_entry["away_team"] == away_team)
        ):
            row["elo_home_pre"] = _format_float(elo_entry.get("home_elo_pre"))
            row["elo_away_pre"] = _format_float(elo_entry.get("away_elo_pre"))
            row["elo_home_expectation"] = _format_float(elo_entry.get("elo_expectation_home"), decimals=6)

            home_pre = elo_entry.get("home_elo_pre")
            away_pre = elo_entry.get("away_elo_pre")
            expectation = elo_entry.get("elo_expectation_home")

            if home_pre is not None and away_pre is not None:
                gap_value = home_pre - away_pre
                row["_elo_gap_pre_value"] = gap_value
                _register_value(gap_values, (league, season), gap_value)

            if expectation is not None:
                expect_gap_value = 2 * expectation - 1.0
                row["_elo_expect_gap_value"] = expect_gap_value
                _register_value(expectation_gap_values, (league, season), expect_gap_value)
            if home_pre is not None and away_pre is not None:
                row["elo_mean_pre"] = _format_float((home_pre + away_pre) / 2.0, decimals=3)
            else:
                row["elo_mean_pre"] = ""
        else:
            row["elo_home_pre"] = ""
            row["elo_away_pre"] = ""
            row["elo_home_expectation"] = ""
            row["elo_mean_pre"] = ""
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

        # Market vs Elo edge (clipped to mitigate outliers)
        forecast_home = _safe_float(row.get("forecast_home_win"))
        if forecast_home is not None and expect_gap_value is not None:
            # expectation gap is 2*E -1, so recover E
            expectation_home = (expect_gap_value + 1.0) / 2.0
        else:
            expectation_home = None if row.get("elo_home_expectation") == "" else _safe_float(row.get("elo_home_expectation"))

        if forecast_home is not None and expectation_home is not None:
            diff = max(-0.35, min(0.35, forecast_home - expectation_home))
            row["market_vs_elo_edge"] = _format_float(diff, decimals=6)
        else:
            row["market_vs_elo_edge"] = ""

        # Rolling volatility + decay metrics
        goal_diff = _safe_float(row.get("goal_difference"))
        xg_diff = _safe_float(row.get("xg_difference"))
        home_goal_std, home_goal_exp = _current_volatility(home_team, goal_history, goal_exp_avgs)
        away_goal_std, away_goal_exp = _current_volatility(away_team, goal_history, goal_exp_avgs)
        home_xg_std, home_xg_exp = _current_volatility(home_team, xg_history, xg_exp_avgs)
        away_xg_std, away_xg_exp = _current_volatility(away_team, xg_history, xg_exp_avgs)
        home_shot_std, home_shot_exp = _current_volatility(home_team, shot_history, shot_exp_avgs)
        away_shot_std, away_shot_exp = _current_volatility(away_team, shot_history, shot_exp_avgs)

        def _assign_metric(key: str, value: Optional[float], decimals: int = 4) -> None:
            row[key] = _format_float(value, decimals=decimals) if value is not None else ""

        _assign_metric("home_goal_diff_std5", home_goal_std)
        _assign_metric("away_goal_diff_std5", away_goal_std)
        _assign_metric("home_goal_diff_exp_decay", home_goal_exp)
        _assign_metric("away_goal_diff_exp_decay", away_goal_exp)
        _assign_metric("home_xg_diff_std5", home_xg_std)
        _assign_metric("away_xg_diff_std5", away_xg_std)
        _assign_metric("home_xg_diff_exp_decay", home_xg_exp)
        _assign_metric("away_xg_diff_exp_decay", away_xg_exp)
        _assign_metric("home_shot_diff_std5", home_shot_std)
        _assign_metric("away_shot_diff_std5", away_shot_std)
        _assign_metric("home_shot_diff_exp_decay", home_shot_exp)
        _assign_metric("away_shot_diff_exp_decay", away_shot_exp)

        gap_pairs = [
            ("goal_diff_std_gap5", home_goal_std, away_goal_std),
            ("goal_diff_exp_decay_gap", home_goal_exp, away_goal_exp),
            ("xg_diff_std_gap5", home_xg_std, away_xg_std),
            ("xg_diff_exp_decay_gap", home_xg_exp, away_xg_exp),
            ("shot_diff_std_gap5", home_shot_std, away_shot_std),
            ("shot_diff_exp_decay_gap", home_shot_exp, away_shot_exp),
        ]
        for key, home_val, away_val in gap_pairs:
            if home_val is not None and away_val is not None:
                row[key] = _format_float(home_val - away_val, decimals=4)
            else:
                row[key] = ""

        _append_volatility_value(home_team, goal_diff, goal_history, goal_exp_avgs)
        _append_volatility_value(away_team, -goal_diff if goal_diff is not None else None, goal_history, goal_exp_avgs)
        _append_volatility_value(home_team, xg_diff, xg_history, xg_exp_avgs)
        _append_volatility_value(away_team, -xg_diff if xg_diff is not None else None, xg_history, xg_exp_avgs)
        _append_volatility_value(home_team, home_shot_diff, shot_history, shot_exp_avgs)
        _append_volatility_value(away_team, away_shot_diff, shot_history, shot_exp_avgs)

    def _compute_stats(values: Dict[Tuple[str, str], list[float]]) -> Dict[Tuple[str, str], Tuple[float, float]]:
        stats: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for key, series in values.items():
            if not series:
                continue
            mean_val = sum(series) / len(series)
            variance = sum((val - mean_val) ** 2 for val in series) / len(series)
            std_val = variance ** 0.5
            stats[key] = (mean_val, std_val)
        return stats

    gap_stats = _compute_stats(gap_values)
    expect_gap_stats = _compute_stats(expectation_gap_values)

    for row in base_rows:
        league = str(row.get("league", "")).strip()
        season = str(row.get("season", "")).strip()
        key = (league, season)

        gap_val = row.pop("_elo_gap_pre_value", None)
        if gap_val is not None:
            row["elo_gap_pre"] = _format_float(gap_val, decimals=3)
            mean_std = gap_stats.get(key)
            if mean_std:
                mean_val, std_val = mean_std
                row["elo_gap_pre_season_z"] = _format_float(
                    0.0 if std_val in (None, 0.0) else (gap_val - mean_val) / (std_val or 1.0),
                    decimals=4,
                )
            else:
                row["elo_gap_pre_season_z"] = ""
        else:
            row["elo_gap_pre"] = ""
            row["elo_gap_pre_season_z"] = ""

        expect_val = row.pop("_elo_expect_gap_value", None)
        if expect_val is not None:
            row["elo_expectation_gap"] = _format_float(expect_val, decimals=3)
            mean_std = expect_gap_stats.get(key)
            if mean_std:
                mean_val, std_val = mean_std
                row["elo_expectation_gap_season_z"] = _format_float(
                    0.0 if std_val in (None, 0.0) else (expect_val - mean_val) / (std_val or 1.0),
                    decimals=4,
                )
            else:
                row["elo_expectation_gap_season_z"] = ""
        else:
            row["elo_expectation_gap"] = ""
            row["elo_expectation_gap_season_z"] = ""

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
