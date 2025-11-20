"""Train the Financial Lens XGBoost model and emit notebook-compatible artefacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

FEATURES = [
    "squad_value_ratio",
    "squad_value_diff",
    "avg_player_value_ratio",
    "avg_player_value_diff",
    "wage_bill_ratio",
    "wage_bill_diff",
    "avg_salary_ratio",
    "avg_salary_diff",
]
TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_NAMES = {0: "home", 1: "draw", 2: "away"}
DEFAULT_VAL_SEASON = 2024
DEFAULT_TEST_SEASON = 2025


def _ensure_columns(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise ValueError(
            "Financial dataset missing required columns: " + ", ".join(missing),
        )
    if "target" not in df.columns:
        raise ValueError("Financial dataset missing 'target' column")
    if "season" not in df.columns:
        raise ValueError("Financial dataset missing 'season' column")


def _train_model(df: pd.DataFrame, val_season: int, test_season: int) -> tuple[XGBClassifier, Dict[str, float]]:
    df = df.dropna(subset=FEATURES).copy()
    if df.empty:
        raise ValueError("Financial dataset has no rows after dropping NA feature rows")

    df["target_label"] = df["target"].map(TARGET_MAP)
    if df["target_label"].isnull().any():
        raise ValueError("Financial dataset contains unknown target labels; expected H/D/A")

    df["season"] = df["season"].astype(int)

    train_mask = df["season"] < val_season
    val_mask = df["season"] == val_season
    test_mask = df["season"] == test_season

    if train_mask.sum() == 0:
        raise ValueError(f"No training rows found for seasons before {val_season}")
    if val_mask.sum() == 0:
        raise ValueError(f"No validation rows found for season {val_season}")
    if test_mask.sum() == 0:
        raise ValueError(f"No test rows found for season {test_season}")

    X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, "target_label"]
    X_val, y_val = df.loc[val_mask, FEATURES], df.loc[val_mask, "target_label"]
    X_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, "target_label"]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(TARGET_MAP),
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
        early_stopping_rounds=40,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    metrics = {
        "train": _evaluate(model, X_train, y_train),
        "val": _evaluate(model, X_val, y_val),
        "test": _evaluate(model, X_test, y_test),
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "class_mapping": TARGET_NAMES,
    }
    return model, metrics


def _evaluate(model: XGBClassifier, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    probs = model.predict_proba(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "logloss": float(log_loss(y, probs, labels=list(TARGET_NAMES.keys()))),
    }


def _save_metrics(path: Path, metrics: Dict[str, object], dataset_version: str) -> None:
    payload = {
        "trainer": "xgboost",
        "feature_cols": FEATURES,
        "dataset_label": f"Dataset_Version_{dataset_version}",
        **metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Financial Lens model")
    parser.add_argument("dataset", type=Path, help="Path to financial_dataset.csv")
    parser.add_argument("run_dir", type=Path, help="Notebook run directory (e.g. artifacts/experiments/run_YYYYMMDD-HHMMSS)")
    parser.add_argument("--dataset-version", default="7")
    parser.add_argument("--val-season", type=int, default=DEFAULT_VAL_SEASON)
    parser.add_argument("--test-season", type=int, default=DEFAULT_TEST_SEASON)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Financial dataset not found at {args.dataset}")

    df = pd.read_csv(args.dataset)
    _ensure_columns(df)

    model, metrics = _train_model(df, args.val_season, args.test_season)

    view_dir = args.run_dir / "financial_lens"
    view_dir.mkdir(parents=True, exist_ok=True)

    model_path = view_dir / "model.json"
    model.get_booster().save_model(model_path)

    metrics_path = view_dir / "metrics.json"
    _save_metrics(metrics_path, metrics, str(args.dataset_version))

    (view_dir / "README.md").write_text(
        "Financial lens trained via pipelines/train_financial_lens.py\n",
        encoding="utf-8",
    )

    print(f"Saved XGBoost model to {model_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
