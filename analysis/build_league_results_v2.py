#!/usr/bin/env python3
"""
Build EPL league results dataset (version 2) with engineered features.

Extends the cleaned Understat export with pre-match form, calendar
intensity, and market-derived diagnostics tailored for the project
baselines. Momentum-oriented views now include per-season z-scored
deltas, congestion markers, and bookmaker drift signals so that the
reinforcement learner can train on well-conditioned inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "understat_data" / "league_results_cleaned.csv"
OUTPUT_PATH = PROJECT_ROOT / "understat_data" / "league_results_v2.csv"
PRIMARY_DATASET_PATH = PROJECT_ROOT / "understat_data" / "Dataset.csv"

METADATA_COLUMNS = [
    "match_id",
    "league",
    "season",
    "match_datetime_utc",
    "match_date",
    "match_weekday",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
]

TARGET_COLUMNS = [
    "home_goals",
    "away_goals",
    "total_goals",
    "goal_difference",
    "home_xg",
    "away_xg",
    "xg_difference",
    "match_outcome",
    "match_outcome_code",
    "outcome_label",
    "outcome_id",
    "home_win_flag",
    "draw_flag",
    "away_win_flag",
    "home_points_actual",
    "away_points_actual",
    "is_result",
]

PERFORMANCE_FEATURES = [
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
]

MOMENTUM_INFERENCE_FEATURES = [
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
]

MARKET_FEATURES = [
    "forecast_home_win",
    "forecast_draw",
    "forecast_away_win",
    "market_home_edge",
    "market_expected_points_home",
    "market_expected_points_away",
    "market_entropy",
    "market_logit_home",
    "market_max_prob",
]


MOMENTUM_BASE_FEATURES = [
    "momentum_points_last3_delta",
    "momentum_points_last2_delta",
    "momentum_points_last8_delta",
    "momentum_points_pct_last3_delta",
    "momentum_goal_diff_last3_delta",
    "momentum_goal_diff_last2_delta",
    "momentum_goal_diff_last8_delta",
    "momentum_xg_diff_last3_delta",
    "momentum_xg_diff_last2_delta",
    "momentum_xg_diff_last8_delta",
    "momentum_points_exp_decay_delta",
    "momentum_xg_exp_decay_delta",
    "momentum_matches_last14_delta",
    "momentum_travel_rest_ratio_delta",
    "momentum_fixture_congestion_delta",
    "momentum_forecast_win_prev_delta",
    "momentum_forecast_trend_delta",
    "form_pct_diff_last5",
    "form_diff_last5",
    "rest_diff",
    "match_day_index",
    "match_day_of_year_norm",
    "match_weekday_index",
]


def _trailing_sum_counts(dates: np.ndarray, weights: np.ndarray, window_days: int) -> np.ndarray:
    """Return trailing-weighted counts over a time window excluding the current row."""

    if dates.dtype != "datetime64[ns]":
        dates = dates.astype("datetime64[ns]")

    counts = np.zeros(len(dates), dtype=float)
    left = 0
    window = np.timedelta64(window_days, "D")

    for idx, current in enumerate(dates):
        window_start = current - window
        while left < idx and dates[left] < window_start:
            left += 1
        if idx == 0:
            counts[idx] = 0.0
        else:
            counts[idx] = float(weights[left:idx].sum())
    return counts


def load_v1() -> pd.DataFrame:
    """Load version 1 dataset with consistent dtypes."""
    df = pd.read_csv(INPUT_PATH)
    df["is_result"] = df["is_result"].astype(bool)
    df = df[df["is_result"]].copy()
    df = df[df["league"] == "EPL"].copy()
    df["match_datetime_utc"] = pd.to_datetime(df["match_datetime_utc"])
    df["match_date"] = pd.to_datetime(df["match_date"])
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
    df["season"] = df["season"].astype(int)
    df = df[df["season"] <= 2024].copy()
    df["match_weekday"] = df["match_date"].dt.day_name()
    df = df.drop(columns=["match_time", "home_team_short", "away_team_short"], errors="ignore")
    return df.sort_values("match_datetime_utc").reset_index(drop=True)


def compute_team_view(matches: pd.DataFrame) -> pd.DataFrame:
    """Transform match table into a long team-centric view with shifts."""

    def outcome_points(code: str) -> tuple[int, int]:
        if code == "H":
            return 3, 0
        if code == "A":
            return 0, 3
        return 1, 1

    home_pts, away_pts = zip(*matches["match_outcome_code"].map(outcome_points))
    matches = matches.assign(home_points=home_pts, away_points=away_pts)

    home_cols = {
        "home_team_id": "team_id",
        "home_team_name": "team_name",
        "home_points": "points",
        "home_goals": "goals_for",
        "away_goals": "goals_against",
        "home_xg": "xg_for",
        "away_xg": "xg_against",
        "forecast_home_win": "win_prob",
    }
    away_cols = {
        "away_team_id": "team_id",
        "away_team_name": "team_name",
        "away_points": "points",
        "away_goals": "goals_for",
        "home_goals": "goals_against",
        "away_xg": "xg_for",
        "home_xg": "xg_against",
        "forecast_away_win": "win_prob",
    }

    home_df = matches[["match_id", "match_datetime_utc", "season", *home_cols.keys()]].rename(columns=home_cols)
    home_df["is_home"] = 1

    away_df = matches[["match_id", "match_datetime_utc", "season", *away_cols.keys()]].rename(columns=away_cols)
    away_df["is_home"] = 0

    long_df = pd.concat([home_df, away_df], ignore_index=True).sort_values(
        ["team_id", "match_datetime_utc"]
    )
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
    long_df["xg_diff"] = long_df["xg_for"] - long_df["xg_against"]
    long_df["match_number"] = long_df.groupby("team_id").cumcount()

    long_df["rest_days"] = (
        long_df.groupby("team_id")["match_datetime_utc"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )
    return long_df.reset_index(drop=True)


def add_rolling_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Attach pre-match rolling aggregates and calendar diagnostics."""

    feature_cols = (
        "points",
        "goals_for",
        "goals_against",
        "goal_diff",
        "xg_for",
        "xg_against",
        "xg_diff",
    )
    windows = (2, 3, 5, 8, 10)
    grouped = long_df.groupby("team_id", group_keys=False)

    for col in feature_cols:
        for window in windows:
            new_col = f"{col}_last_{window}"
            long_df[new_col] = grouped[col].transform(
                lambda s, w=window: s.shift().rolling(window=w, min_periods=1).sum()
            )

    long_df["rest_days_prev"] = grouped["rest_days"].transform(lambda s: s.shift())
    long_df["rest_days_prev"] = long_df["rest_days_prev"].fillna(0)
    long_df["rest_days_capped"] = long_df["rest_days_prev"].clip(upper=28.0)
    long_df["rest_reset_flag"] = (long_df["rest_days_prev"] > 35).astype(int)

    for window in windows:
        denom = 3 * long_df["match_number"].clip(upper=window)
        denom = denom.replace(0, np.nan)
        long_df[f"points_pct_last_{window}"] = long_df[f"points_last_{window}"] / denom

    decay_alpha = 0.5
    long_df["points_exp_decay"] = grouped["points"].transform(
        lambda s: s.shift().ewm(alpha=decay_alpha, adjust=False).mean()
    )
    long_df["goal_diff_exp_decay"] = grouped["goal_diff"].transform(
        lambda s: s.shift().ewm(alpha=decay_alpha, adjust=False).mean()
    )
    long_df["xg_diff_exp_decay"] = grouped["xg_diff"].transform(
        lambda s: s.shift().ewm(alpha=decay_alpha, adjust=False).mean()
    )

    matches_last14 = []
    away_matches_last14 = []
    for _, group in grouped:
        dates = group["match_datetime_utc"].to_numpy()
        all_weights = np.ones(len(group), dtype=float)
        away_weights = (1 - group["is_home"].to_numpy()).astype(float)
        all_counts = _trailing_sum_counts(dates, all_weights, window_days=14)
        away_counts = _trailing_sum_counts(dates, away_weights, window_days=14)
        matches_last14.append(pd.Series(all_counts, index=group.index))
        away_matches_last14.append(pd.Series(away_counts, index=group.index))

    long_df["matches_last_14_days"] = pd.concat(matches_last14).sort_index()
    long_df["away_matches_last_14_days"] = pd.concat(away_matches_last14).sort_index()
    long_df["fixture_congestion_flag"] = (
        (long_df["matches_last_14_days"] >= 3) | (long_df["rest_days_prev"] <= 3)
    ).astype(int)

    travel_denominator = (
        long_df["rest_days_prev"]
        + long_df["away_matches_last_14_days"]
        + 1.0
    )
    long_df["travel_rest_ratio"] = (
        long_df["rest_days_prev"] / travel_denominator.replace(0, np.nan)
    )
    long_df["travel_rest_ratio"] = long_df["travel_rest_ratio"].fillna(0.0)

    long_df["win_prob_prev"] = grouped["win_prob"].transform(lambda s: s.shift(1))
    long_df["win_prob_prev2"] = grouped["win_prob"].transform(lambda s: s.shift(2))
    long_df["win_prob_prev_delta"] = long_df["win_prob_prev"] - long_df["win_prob_prev2"]

    long_df.fillna(0, inplace=True)
    return long_df


def pivot_features(matches: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """Join team aggregates back to matches with home/away prefixes."""
    base_feature_cols = [
        "match_id",
        "match_number",
        "rest_days_prev",
        "rest_days_capped",
        "rest_reset_flag",
        "matches_last_14_days",
        "away_matches_last_14_days",
        "fixture_congestion_flag",
        "travel_rest_ratio",
        "win_prob_prev",
        "win_prob_prev_delta",
    ]
    rolling_cols = [
        col
        for col in long_df.columns
        if col.endswith(("last_2", "last_3", "last_5", "last_8", "last_10"))
        or col.startswith("points_pct_last_")
        or col.endswith("exp_decay")
    ]
    feature_cols = base_feature_cols + rolling_cols

    home_features = (
        long_df[long_df["is_home"] == 1][feature_cols]
        .set_index("match_id")
        .add_prefix("home_")
    )
    away_features = (
        long_df[long_df["is_home"] == 0][feature_cols]
        .set_index("match_id")
        .add_prefix("away_")
    )

    enriched = (
        matches.set_index("match_id")
        .join(home_features, how="left")
        .join(away_features, how="left")
        .reset_index()
    )

    enriched["home_matches_played"] = enriched["home_match_number"]
    enriched["away_matches_played"] = enriched["away_match_number"]

    enriched.rename(
        columns={
            "home_win_prob_prev": "home_forecast_win_prev",
            "away_win_prob_prev": "away_forecast_win_prev",
            "home_win_prob_prev_delta": "home_forecast_win_prev_delta",
            "away_win_prob_prev_delta": "away_forecast_win_prev_delta",
        },
        inplace=True,
    )

    enriched["form_diff_last5"] = (
        enriched["home_points_last_5"] - enriched["away_points_last_5"]
    )
    enriched["form_pct_diff_last5"] = (
        enriched["home_points_pct_last_5"] - enriched["away_points_pct_last_5"]
    )
    enriched["xg_diff_last5"] = (
        enriched["home_xg_diff_last_5"] - enriched["away_xg_diff_last_5"]
    )
    enriched["rest_diff"] = (
        enriched["home_rest_days_capped"] - enriched["away_rest_days_capped"]
    )
    enriched["rest_reset_flag_pair"] = (
        enriched["home_rest_reset_flag"] | enriched["away_rest_reset_flag"]
    )
    enriched["season_phase_home"] = (
        enriched["home_matches_played"] / 38.0
    ).clip(upper=1)
    enriched["season_phase_away"] = (
        enriched["away_matches_played"] / 38.0
    ).clip(upper=1)

    enriched["fixture_congestion_flag_pair"] = (
        (enriched["home_fixture_congestion_flag"] == 1)
        | (enriched["away_fixture_congestion_flag"] == 1)
    ).astype(int)

    enriched["momentum_points_last3_delta"] = (
        enriched["home_points_last_3"] - enriched["away_points_last_3"]
    )
    enriched["momentum_points_last2_delta"] = (
        enriched["home_points_last_2"] - enriched["away_points_last_2"]
    )
    enriched["momentum_points_last8_delta"] = (
        enriched["home_points_last_8"] - enriched["away_points_last_8"]
    )
    enriched["momentum_points_pct_last3_delta"] = (
        enriched["home_points_pct_last_3"] - enriched["away_points_pct_last_3"]
    )
    enriched["momentum_goal_diff_last3_delta"] = (
        enriched["home_goal_diff_last_3"] - enriched["away_goal_diff_last_3"]
    )
    enriched["momentum_goal_diff_last2_delta"] = (
        enriched["home_goal_diff_last_2"] - enriched["away_goal_diff_last_2"]
    )
    enriched["momentum_goal_diff_last8_delta"] = (
        enriched["home_goal_diff_last_8"] - enriched["away_goal_diff_last_8"]
    )
    enriched["momentum_xg_diff_last3_delta"] = (
        enriched["home_xg_diff_last_3"] - enriched["away_xg_diff_last_3"]
    )
    enriched["momentum_xg_diff_last2_delta"] = (
        enriched["home_xg_diff_last_2"] - enriched["away_xg_diff_last_2"]
    )
    enriched["momentum_xg_diff_last8_delta"] = (
        enriched["home_xg_diff_last_8"] - enriched["away_xg_diff_last_8"]
    )
    enriched["momentum_points_exp_decay_delta"] = (
        enriched["home_points_exp_decay"] - enriched["away_points_exp_decay"]
    )
    enriched["momentum_xg_exp_decay_delta"] = (
        enriched["home_xg_diff_exp_decay"] - enriched["away_xg_diff_exp_decay"]
    )
    enriched["momentum_matches_last14_delta"] = (
        enriched["home_matches_last_14_days"] - enriched["away_matches_last_14_days"]
    )
    enriched["momentum_travel_rest_ratio_delta"] = (
        enriched["home_travel_rest_ratio"] - enriched["away_travel_rest_ratio"]
    )
    enriched["momentum_fixture_congestion_delta"] = (
        enriched["home_fixture_congestion_flag"] - enriched["away_fixture_congestion_flag"]
    )
    enriched["momentum_forecast_win_prev_delta"] = (
        enriched["home_forecast_win_prev"] - enriched["away_forecast_win_prev"]
    )
    enriched["momentum_forecast_trend_delta"] = (
        enriched["home_forecast_win_prev_delta"]
        - enriched["away_forecast_win_prev_delta"]
    )

    delta_columns = [
        "momentum_points_last3_delta",
        "momentum_points_last2_delta",
        "momentum_points_last8_delta",
        "momentum_points_pct_last3_delta",
        "momentum_goal_diff_last3_delta",
        "momentum_goal_diff_last2_delta",
        "momentum_goal_diff_last8_delta",
        "momentum_xg_diff_last3_delta",
        "momentum_xg_diff_last2_delta",
        "momentum_xg_diff_last8_delta",
        "momentum_points_exp_decay_delta",
        "momentum_xg_exp_decay_delta",
        "momentum_matches_last14_delta",
        "momentum_travel_rest_ratio_delta",
        "momentum_fixture_congestion_delta",
        "momentum_forecast_win_prev_delta",
        "momentum_forecast_trend_delta",
    ]
    for col in delta_columns:
        enriched[col] = enriched[col].fillna(0.0)

    enriched["match_day_index"] = (
        enriched["match_date"] - enriched["match_date"].min()
    ).dt.days.astype(float)
    enriched["match_day_of_year"] = enriched["match_date"].dt.dayofyear.astype(float)
    enriched["match_day_of_year_norm"] = enriched["match_day_of_year"] / 366.0
    enriched["match_weekday_index"] = enriched["match_date"].dt.weekday.astype(float)

    return enriched


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer market-signal derived features."""
    epsilon = 1e-6
    for col in ["forecast_home_win", "forecast_draw", "forecast_away_win"]:
        df[col] = df[col].clip(epsilon, 1 - epsilon)

    total = df["forecast_home_win"] + df["forecast_draw"] + df["forecast_away_win"]
    df["forecast_home_win"] /= total
    df["forecast_draw"] /= total
    df["forecast_away_win"] /= total

    df["market_home_edge"] = df["forecast_home_win"] - df["forecast_away_win"]
    df["market_expected_points_home"] = (
        3 * df["forecast_home_win"] + df["forecast_draw"]
    )
    df["market_expected_points_away"] = (
        3 * df["forecast_away_win"] + df["forecast_draw"]
    )

    def entropy(row: pd.Series) -> float:
        probs = row[["forecast_home_win", "forecast_draw", "forecast_away_win"]].astype(float).values
        return float(-(probs * np.log(probs)).sum())

    df["market_entropy"] = df.apply(entropy, axis=1)
    df["market_logit_home"] = np.log(
        df["forecast_home_win"] / df["forecast_away_win"]
    )
    df["market_max_prob"] = df[
        ["forecast_home_win", "forecast_draw", "forecast_away_win"]
    ].max(axis=1)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create modelling-friendly target encodings."""
    mapping = {"H": 2, "D": 1, "A": 0}
    df["outcome_label"] = df["match_outcome_code"]
    df["outcome_id"] = df["match_outcome_code"].map(mapping).astype(int)
    df["home_points_actual"] = df["home_win_flag"] * 3 + df["draw_flag"] * 1
    df["away_points_actual"] = df["away_win_flag"] * 3 + df["draw_flag"] * 1
    return df


def add_momentum_standardisation(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-season statistics and z-scored momentum features."""

    if "season" not in df.columns:
        raise KeyError("Season column required for momentum standardisation.")

    available_features = [col for col in MOMENTUM_BASE_FEATURES if col in df.columns]
    if not available_features:
        return df

    season_groups = df.groupby("season", group_keys=False)
    new_columns = {}

    for col in available_features:
        mean_col = f"{col}_season_mean"
        std_col = f"{col}_season_std"
        var_col = f"{col}_season_var"
        z_col = f"{col}_season_z"

        mean_series = season_groups[col].transform("mean").astype(np.float32)
        std_series = season_groups[col].transform("std").astype(np.float32).fillna(0.0)
        denom = std_series.replace(0.0, 1.0)
        z_series = ((df[col] - mean_series) / denom).fillna(0.0).astype(np.float32)

        new_columns[mean_col] = mean_series
        new_columns[std_col] = std_series
        new_columns[var_col] = (std_series ** 2).astype(np.float32)
        new_columns[z_col] = z_series

    if new_columns:
        df = df.assign(**new_columns)

    return df


def prune_inference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant columns so saved dataset matches model feature needs."""

    required_cols = set(METADATA_COLUMNS + TARGET_COLUMNS + PERFORMANCE_FEATURES + MARKET_FEATURES + MOMENTUM_INFERENCE_FEATURES)
    weekday_cols = [
        col
        for col in df.columns
        if col.startswith("match_weekday_") and "match_weekday_index" not in col
    ]
    required_cols.update(weekday_cols)

    for base in MOMENTUM_BASE_FEATURES:
        for suffix in ("", "_season_mean", "_season_std", "_season_var", "_season_z"):
            col_name = f"{base}{suffix}" if suffix else base
            if col_name in df.columns:
                required_cols.add(col_name)

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing required columns after pruning pass: {missing}")

    drop_cols = [col for col in df.columns if col not in required_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns into core metadata, targets, market, and performance."""
    base_cols = [
        "match_id",
        "league",
        "season",
        "match_datetime_utc",
        "match_date",
        "match_weekday",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
    ]
    result_cols = [
        "home_goals",
        "away_goals",
        "total_goals",
        "goal_difference",
        "home_xg",
        "away_xg",
        "xg_difference",
        "match_outcome",
        "match_outcome_code",
        "outcome_label",
        "outcome_id",
        "home_win_flag",
        "draw_flag",
        "away_win_flag",
        "home_points_actual",
        "away_points_actual",
    ]
    market_cols = [
        "forecast_home_win",
        "forecast_draw",
        "forecast_away_win",
        "market_home_edge",
        "market_expected_points_home",
        "market_expected_points_away",
        "market_entropy",
        "market_logit_home",
        "market_max_prob",
    ]
    perf_cols = sorted(
        [
            col
            for col in df.columns
            if col.startswith(("home_", "away_", "form_", "xg_", "season_phase", "rest_", "momentum_"))
            and col not in base_cols
            and col not in result_cols
        ]
    )
    remaining = [col for col in df.columns if col not in base_cols + result_cols + market_cols + perf_cols]
    ordered = base_cols + result_cols + market_cols + perf_cols + remaining
    return df[ordered]


def main() -> None:
    matches_v1 = load_v1()
    long_df = compute_team_view(matches_v1)
    long_df = add_rolling_features(long_df)
    enriched = pivot_features(matches_v1, long_df)
    enriched = add_market_features(enriched)
    enriched = add_targets(enriched)
    enriched = add_momentum_standardisation(enriched)
    enriched = prune_inference_columns(enriched)
    enriched = reorder_columns(enriched)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_PATH, index=False)
    enriched.to_csv(PRIMARY_DATASET_PATH, index=False)
    print(
        "Saved version 2 dataset with "
        f"{len(enriched)} rows to {OUTPUT_PATH} and {PRIMARY_DATASET_PATH}"
    )


if __name__ == "__main__":
    main()
