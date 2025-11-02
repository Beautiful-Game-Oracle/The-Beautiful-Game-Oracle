#!/usr/bin/env python3
"""
Exploratory diagnostics for the next EPL dataset iteration.

Loads league-level match data, engineers pre-match rollups, runs basic
statistical checks, and saves diagnostic plots that inform feature and
cleaning priorities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "understat_data"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.switch_backend("Agg")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings(
    "ignore",
    message="The default of observed=False is deprecated",
)


def load_matches() -> pd.DataFrame:
    """Load and type-coerce the cleaned league results."""
    matches = pd.read_csv(DATA_DIR / "league_results_cleaned.csv")
    matches["is_result"] = (
        matches["is_result"].astype(str).str.lower().str.strip() == "true"
    )
    matches = matches[matches["is_result"]].copy()
    matches = matches[matches["league"] == "EPL"].copy()
    matches["match_datetime"] = pd.to_datetime(
        matches["match_datetime_utc"], errors="coerce"
    )
    matches["season"] = matches["season"].fillna(0).astype(int)
    return matches.sort_values("match_datetime")


def outcome_points(code: str) -> Tuple[int, int]:
    """Return home and away point allocation for the result code."""
    if code == "H":
        return 3, 0
    if code == "A":
        return 0, 3
    return 1, 1


def expand_team_view(matches: pd.DataFrame) -> pd.DataFrame:
    """Create a team-perspective table for rolling feature engineering."""
    home_points, away_points = zip(*matches["match_outcome_code"].map(outcome_points))
    matches = matches.assign(home_points=home_points, away_points=away_points)

    home_df = matches[
        [
            "match_id",
            "match_datetime",
            "season",
            "home_team_id",
            "home_team_name",
            "home_points",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
        ]
    ].rename(
        columns={
            "home_team_id": "team_id",
            "home_team_name": "team_name",
            "home_points": "points",
            "home_goals": "goals_for",
            "away_goals": "goals_against",
            "home_xg": "xg_for",
            "away_xg": "xg_against",
        }
    )
    home_df["is_home"] = 1

    away_df = matches[
        [
            "match_id",
            "match_datetime",
            "season",
            "away_team_id",
            "away_team_name",
            "away_points",
            "away_goals",
            "home_goals",
            "away_xg",
            "home_xg",
        ]
    ].rename(
        columns={
            "away_team_id": "team_id",
            "away_team_name": "team_name",
            "away_points": "points",
            "away_goals": "goals_for",
            "home_goals": "goals_against",
            "away_xg": "xg_for",
            "home_xg": "xg_against",
        }
    )
    away_df["is_home"] = 0

    long_df = (
        pd.concat([home_df, away_df], ignore_index=True)
        .sort_values(["team_id", "match_datetime"])
        .reset_index(drop=True)
    )

    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
    long_df["xg_diff"] = long_df["xg_for"] - long_df["xg_against"]
    long_df["match_number"] = long_df.groupby("team_id").cumcount()
    long_df["rest_days"] = (
        long_df.groupby("team_id")["match_datetime"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )
    return long_df


def add_rolling_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Engineer pre-match rolling aggregates for each team."""
    def rolling_transform(series: pd.Series, window: int, agg: str) -> pd.Series:
        windowed = series.shift().rolling(window=window, min_periods=1)
        if agg == "sum":
            return windowed.sum()
        if agg == "mean":
            return windowed.mean()
        if agg == "std":
            return windowed.std()
        raise ValueError(f"Unsupported agg {agg}")

    grouped = long_df.groupby("team_id", group_keys=False)

    for window in (3, 5, 10):
        suffix = f"{window}"
        long_df[f"points_last_{suffix}"] = grouped["points"].transform(
            rolling_transform, window=window, agg="sum"
        )
        long_df[f"goal_diff_last_{suffix}"] = grouped["goal_diff"].transform(
            rolling_transform, window=window, agg="sum"
        )
        long_df[f"xg_diff_last_{suffix}"] = grouped["xg_diff"].transform(
            rolling_transform, window=window, agg="sum"
        )
        long_df[f"goals_for_last_{suffix}"] = grouped["goals_for"].transform(
            rolling_transform, window=window, agg="sum"
        )

    long_df["avg_rest_days"] = grouped["rest_days"].transform(
        rolling_transform, window=5, agg="mean"
    )
    long_df["rest_days_prev"] = grouped["rest_days"].transform(
        lambda s: s.shift()
    )

    fill_cols = [col for col in long_df.columns if "last_" in col or "rest_days" in col]
    long_df[fill_cols] = long_df[fill_cols].fillna(0)
    return long_df


def pivot_back(matches: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """Attach engineered features back to the match-level table."""
    feature_cols = [
        "points_last_3",
        "points_last_5",
        "goal_diff_last_5",
        "goal_diff_last_10",
        "xg_diff_last_5",
        "goals_for_last_5",
        "avg_rest_days",
        "rest_days_prev",
    ]

    home_features = (
        long_df[long_df["is_home"] == 1][["match_id"] + feature_cols]
        .set_index("match_id")
        .add_prefix("home_")
    )
    away_features = (
        long_df[long_df["is_home"] == 0][["match_id"] + feature_cols]
        .set_index("match_id")
        .add_prefix("away_")
    )

    enriched = (
        matches.set_index("match_id")
        .join(home_features, how="left")
        .join(away_features, how="left")
        .reset_index()
    )

    enriched["form_diff_last5"] = (
        enriched["home_points_last_5"] - enriched["away_points_last_5"]
    )
    enriched["rest_diff"] = enriched["home_rest_days_prev"] - enriched["away_rest_days_prev"]

    form_cols = [
        "home_points_last_5",
        "away_points_last_5",
        "home_goal_diff_last_5",
        "away_goal_diff_last_5",
        "home_xg_diff_last_5",
        "away_xg_diff_last_5",
        "home_rest_days_prev",
        "away_rest_days_prev",
        "form_diff_last5",
        "rest_diff",
    ]
    enriched[form_cols] = enriched[form_cols].fillna(0)
    return enriched


def plot_correlation_matrix(data: pd.DataFrame, output_path: Path) -> None:
    """Save a correlation heatmap for selected numeric features."""
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_calibration_curve(matches: pd.DataFrame, output_path: Path) -> None:
    """Visualize how bookmaker forecasts align with actual home win rates."""
    bins = pd.interval_range(start=0, end=1, freq=0.1, closed="right")
    binned = pd.cut(matches["forecast_home_win"], bins=bins, include_lowest=True)
    calibration = (
        matches.assign(is_home_win=(matches["match_outcome_code"] == "H").astype(int))
        .groupby(binned)["is_home_win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "home_win_rate", "count": "sample_size"})
        .reset_index()
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [interval.mid for interval in calibration["forecast_home_win"]],
        calibration["home_win_rate"],
        marker="o",
        label="Empirical home win rate",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Forecasted home win probability")
    ax.set_ylabel("Observed home win rate")
    ax.set_title("Home Win Calibration vs Market Forecasts")

    for _, row in calibration.iterrows():
        ax.annotate(
            int(row["sample_size"]),
            (row["forecast_home_win"].mid, row["home_win_rate"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_form_vs_outcome(matches: pd.DataFrame, output_path: Path) -> None:
    """Show relationship between recent form differential and home win odds."""
    quantiles = np.linspace(0, 1, 11)
    bins = matches["form_diff_last5"].quantile(quantiles).unique()
    bins = np.unique(np.append(bins, [matches["form_diff_last5"].min(), matches["form_diff_last5"].max()]))
    bins.sort()

    binned = pd.cut(matches["form_diff_last5"], bins=np.unique(bins), include_lowest=True)
    grouped = (
        matches.assign(is_home_win=(matches["match_outcome_code"] == "H").astype(int))
        .groupby(binned)["is_home_win"]
        .agg(["mean", "count"])
        .reset_index()
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    centers = [interval.mid for interval in grouped["form_diff_last5"]]
    ax.bar(centers, grouped["mean"], width=1.5, color="#1f77b4", alpha=0.7)
    ax.set_xlabel("Form differential (home points last 5 - away points last 5)")
    ax.set_ylabel("Observed home win rate")
    ax.set_title("Recent Form Differential vs Home Win Rate")
    for idx, row in grouped.iterrows():
        ax.text(
            centers[idx],
            row["mean"] + 0.01,
            f"n={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_logistic_baseline(matches: pd.DataFrame) -> Dict[str, float]:
    """Run cross-validated logistic regression using pre-match features."""
    feature_cols = [
        "forecast_home_win",
        "forecast_draw",
        "forecast_away_win",
        "home_points_last_5",
        "away_points_last_5",
        "home_goal_diff_last_5",
        "away_goal_diff_last_5",
        "home_rest_days_prev",
        "away_rest_days_prev",
    ]

    feature_df = matches.dropna(subset=feature_cols).copy()
    X = feature_df[feature_cols]
    y = feature_df["match_outcome_code"]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="auto",
                    max_iter=500,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    season_counts = feature_df["season"].nunique()
    if season_counts >= 3:
        splits = min(5, season_counts)
        cv = GroupKFold(n_splits=splits)
        cv_args = {"groups": feature_df["season"]}
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_args = {}

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "neg_log_loss": "neg_log_loss",
        },
        n_jobs=None,
        **cv_args,
    )

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    balanced_acc = balanced_accuracy_score(y, preds)

    summary = {
        "cv_accuracy_mean": scores["test_accuracy"].mean(),
        "cv_accuracy_std": scores["test_accuracy"].std(),
        "cv_log_loss_mean": -scores["test_neg_log_loss"].mean(),
        "cv_log_loss_std": scores["test_neg_log_loss"].std(),
        "in_sample_balanced_accuracy": balanced_acc,
    }
    return summary


def main() -> None:
    matches = load_matches()
    print("Loaded matches:", len(matches))
    print("Season coverage:", matches["season"].unique())
    print("Outcome distribution:\n", matches["match_outcome"].value_counts(normalize=True))

    numeric_cols = [
        "home_xg",
        "away_xg",
        "xg_difference",
        "forecast_home_win",
        "forecast_draw",
        "forecast_away_win",
        "home_goals",
        "away_goals",
    ]
    missing = matches[numeric_cols].isna().mean().sort_values(ascending=False)
    print("\nMissingness (ratio) in key numeric fields:\n", missing[missing > 0])

    long_df = expand_team_view(matches)
    long_df = add_rolling_features(long_df)
    enriched = pivot_back(matches, long_df)

    correlation_features = enriched[
        [
            "forecast_home_win",
            "forecast_away_win",
            "form_diff_last5",
            "rest_diff",
            "home_goal_diff_last_5",
            "away_goal_diff_last_5",
            "home_xg",
            "away_xg",
            "xg_difference",
        ]
    ].fillna(0)

    plot_correlation_matrix(
        correlation_features,
        FIGURES_DIR / "correlation_matrix.png",
    )
    plot_calibration_curve(
        enriched,
        FIGURES_DIR / "forecast_calibration.png",
    )
    plot_form_vs_outcome(
        enriched,
        FIGURES_DIR / "form_diff_vs_outcome.png",
    )
    print("\nSaved figures to", FIGURES_DIR)

    scores = evaluate_logistic_baseline(enriched)
    print("\nLogistic regression diagnostics:")
    for key, value in scores.items():
        print(f"  {key}: {value:.4f}")

    enriched.to_csv(PROJECT_ROOT / "reports" / "enriched_matches_snapshot.csv", index=False)
    print("\nExported enriched snapshot to reports/enriched_matches_snapshot.csv")


if __name__ == "__main__":
    main()
