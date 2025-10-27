#!/usr/bin/env python3
"""
Prepare model-specific feature views from the EPL Dataset v2 snapshot.

Loads `understat_data/Dataset.csv` and emits parquet tables for the
performance, momentum, and market modelling paradigms used in the
Football Predictor notebook. Each output stores metadata columns
(`match_id`, `season`, `match_datetime_utc`, `match_outcome_code`,
`outcome_id`) plus the designated feature subset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "understat_data" / "Dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "feature_views_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["match_datetime_utc", "match_date"])
    if {"match_id", "match_outcome_code", "outcome_id"}.issubset(df.columns) is False:
        raise ValueError("Dataset missing required outcome columns.")

    df.sort_values("match_datetime_utc", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["match_day_index"] = (df["match_date"] - df["match_date"].min()).dt.days.astype(float)
    df["match_day_of_year"] = df["match_date"].dt.dayofyear.astype(float)
    df["match_day_of_year_norm"] = df["match_day_of_year"] / 366.0
    df["match_weekday_index"] = df["match_date"].dt.weekday.astype(float)
    weekday_dummies = pd.get_dummies(df["match_weekday"], prefix="match_weekday", dtype=float)
    df = pd.concat([df, weekday_dummies], axis=1)
    return df


def get_weekday_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if col.startswith(WEEKDAY_PREFIX) and "match_weekday_index" not in col
    ]


FEATURE_VIEWS: Dict[str, Dict[str, List[str]]] = {
    "performance_dense": {
        "description": "Rolling goal/xG aggregates over 5–10 match windows",
        "columns": [
            "home_goals_for_last_5",
            "home_goals_against_last_5",
            "home_goal_diff_last_5",
            "home_xg_for_last_5",
            "home_xg_against_last_5",
            "home_xg_diff_last_5",
            "home_points_last_5",
            "away_goals_for_last_5",
            "away_goals_against_last_5",
            "away_goal_diff_last_5",
            "away_xg_for_last_5",
            "away_xg_against_last_5",
            "away_xg_diff_last_5",
            "away_points_last_5",
            "form_diff_last5",
            "xg_diff_last5",
            "match_day_index",
            "match_day_of_year_norm",
            "match_weekday_index",
        ],
    },
    "momentum_policy_rl": {
        "description": "Z-scored momentum deltas with congestion and odds drift",
        "columns": [
            "momentum_points_last3_delta_season_z",
            "momentum_points_last2_delta_season_z",
            "momentum_points_last8_delta_season_z",
            "momentum_points_pct_last3_delta_season_z",
            "momentum_goal_diff_last3_delta_season_z",
            "momentum_goal_diff_last2_delta_season_z",
            "momentum_goal_diff_last8_delta_season_z",
            "momentum_xg_diff_last3_delta_season_z",
            "momentum_xg_diff_last2_delta_season_z",
            "momentum_xg_diff_last8_delta_season_z",
            "momentum_points_exp_decay_delta_season_z",
            "momentum_xg_exp_decay_delta_season_z",
            "momentum_matches_last14_delta_season_z",
            "momentum_travel_rest_ratio_delta_season_z",
            "momentum_forecast_win_prev_delta_season_z",
            "momentum_forecast_trend_delta_season_z",
            "form_pct_diff_last5_season_z",
            "form_diff_last5_season_z",
            "rest_diff_season_z",
            "fixture_congestion_flag_pair",
            "momentum_fixture_congestion_delta",
            "rest_reset_flag_pair",
            "match_day_index_season_z",
            "match_day_of_year_norm_season_z",
            "match_weekday_index_season_z",
        ],
    },
    "market_gradient_boost": {
        "description": "Market odds derived statistics",
        "columns": [
            "forecast_home_win",
            "forecast_draw",
            "forecast_away_win",
            "market_home_edge",
            "market_expected_points_home",
            "market_expected_points_away",
            "market_entropy",
            "market_logit_home",
            "market_max_prob",
            "match_day_index",
            "match_day_of_year_norm",
            "match_weekday_index",
        ],
    },
}
WEEKDAY_PREFIX = "match_weekday_"


def validate_columns(df: pd.DataFrame, columns: List[str], view_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"View '{view_name}' is missing required columns: {missing}")


def export_views(df: pd.DataFrame) -> None:
    metadata_cols = [
        "match_id",
        "season",
        "match_datetime_utc",
        "match_outcome_code",
        "outcome_id",
    ]

    weekday_cols = get_weekday_columns(df)

    for view_name, spec in FEATURE_VIEWS.items():
        full_columns = metadata_cols + spec["columns"] + weekday_cols
        validate_columns(df, spec["columns"], view_name)
        view_df = df[full_columns].copy()
        output_path = OUTPUT_DIR / f"{view_name}.csv"
        view_df.to_csv(output_path, index=False)
        print(
            f"[{view_name}] Saved {len(view_df)} rows x {len(spec['columns'])} features "
            f"to {output_path}"
        )


def main() -> None:
    dataset = load_dataset()
    export_views(dataset)
    summary_rows = []
    weekday_cols = get_weekday_columns(dataset)
    for name, spec in FEATURE_VIEWS.items():
        summary_rows.append(
            {
                "view": name,
                "description": spec["description"],
                "num_features": len(spec["columns"]) + len(weekday_cols),
                "output_path": OUTPUT_DIR / f"{name}.csv",
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "feature_views_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote feature view summary to {summary_path}")


if __name__ == "__main__":
    main()
