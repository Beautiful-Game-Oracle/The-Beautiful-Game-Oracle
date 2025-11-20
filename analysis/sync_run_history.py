#!/usr/bin/env python3
"""
Utility helpers for harmonising historical experiment run logs.

The sync_run_history() entry point scans artefact directories produced by the
notebook baselines, backfills missing metadata (e.g., dataset labels), and
upserts consolidated records into baseline_run_history.csv so that legacy runs
remain comparable with new dataset variants.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Canonical ordering expected by the notebook run log.
CANONICAL_COLUMNS = [
    "timestamp",
    "run_id",
    "baseline",
    "trainer",
    "feature_view",
    "train_accuracy",
    "val_accuracy",
    "test_accuracy",
    "train_loss",
    "val_loss",
    "test_loss",
    "val_logloss",
    "test_logloss",
    "epochs_trained",
    "seasons",
    "dataset_label",
    "notes",
]

# Default descriptions in case they are not supplied by the caller.
DEFAULT_BASELINE_DESCRIPTIONS = {
    "performance_dense": "Performance-based dense network using smoothed per-match aggregates",
    "momentum_policy_rl": "Momentum-policy REINFORCE agent leveraging short-horizon trends",
    "market_gradient_boost": "Market odds derived statistics",
}


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "nan", "NaN", "None"}
    return False


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _infer_dataset_label(feature_cols: Iterable[str]) -> str:
    cols = [str(col) for col in (feature_cols or [])]
    lowered = [col.lower() for col in cols]
    if any("shot" in col for col in lowered) or any(col.startswith("elo_") for col in lowered):
        return "Dataset_V3"
    if any("market_expected_points" in col for col in lowered):
        return "Dataset_V2"
    return "Dataset_V1"


def _extract_seasons(predictions_path: Path) -> str:
    if not predictions_path.exists():
        return ""
    seasons = set()
    with predictions_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            season_val = row.get("season")
            if season_val is not None and str(season_val).strip():
                seasons.add(str(season_val))
    return "|".join(sorted(seasons))


def _load_existing_log(path: Path) -> Tuple[Dict[Tuple[str, str], dict], list[str]]:
    if not path.exists():
        return {}, list(CANONICAL_COLUMNS)
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        columns = list(reader.fieldnames or CANONICAL_COLUMNS)
        records: Dict[Tuple[str, str], dict] = {}
        for row in reader:
            key = (row.get("run_id", "") or "", row.get("baseline", "") or "")
            records[key] = row
    return records, columns


def sync_run_history(
    experiments_dir: Path,
    run_log_path: Path,
    *,
    baseline_descriptions: Dict[str, str] | None = None,
    canonical_columns: Iterable[str] | None = None,
) -> dict:
    """
    Backfill baseline_run_history.csv from saved metrics.json artefacts.

    Returns a summary dict with counts of created/updated entries and the
    resolved column ordering.
    """
    descriptions = dict(DEFAULT_BASELINE_DESCRIPTIONS)
    if baseline_descriptions:
        descriptions.update(baseline_descriptions)

    canonical = list(canonical_columns or CANONICAL_COLUMNS)
    existing_records, existing_columns = _load_existing_log(run_log_path)

    # Ensure canonical order seeds the final column ordering.
    column_order: list[str] = []
    for col in canonical + existing_columns:
        if col not in column_order:
            column_order.append(col)

    created = 0
    updated = 0

    for run_dir in sorted(experiments_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name.replace("run_", "")
        timestamp = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()

        for baseline_dir in sorted(run_dir.iterdir()):
            if not baseline_dir.is_dir():
                continue

            metrics_path = baseline_dir / "metrics.json"
            if not metrics_path.exists():
                continue

            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                continue

            trainer = metrics.get("trainer", "")
            feature_cols = metrics.get("feature_cols") or []
            dataset_label = metrics.get("dataset_label")
            if _is_blank(dataset_label):
                dataset_label = _infer_dataset_label(feature_cols)
                metrics["dataset_label"] = dataset_label
                metrics_path.write_text(json.dumps(metrics, indent=2))
            dataset_label = _format_value(dataset_label)

            feature_view = descriptions.get(baseline_dir.name, baseline_dir.name)
            seasons = _extract_seasons(baseline_dir / "test_predictions.csv")

            record = {
                "timestamp": timestamp,
                "run_id": run_id,
                "baseline": baseline_dir.name,
                "trainer": trainer,
                "feature_view": feature_view,
                "train_accuracy": _format_value(metrics.get("train", {}).get("accuracy")),
                "val_accuracy": _format_value(metrics.get("val", {}).get("accuracy")),
                "test_accuracy": _format_value(metrics.get("test", {}).get("accuracy")),
                "train_loss": _format_value(metrics.get("train", {}).get("logloss")),
                "val_loss": _format_value(metrics.get("val", {}).get("logloss")),
                "test_loss": _format_value(metrics.get("test", {}).get("logloss")),
                "val_logloss": _format_value(metrics.get("val", {}).get("logloss")),
                "test_logloss": _format_value(metrics.get("test", {}).get("logloss")),
                "epochs_trained": _format_value(metrics.get("epochs_trained")),
                "seasons": seasons,
                "dataset_label": dataset_label,
                "notes": "",
            }

            for col in record:
                if col not in column_order:
                    column_order.append(col)

            key = (record["run_id"], record["baseline"])
            if key in existing_records:
                row = existing_records[key]
                row_changed = False
                for col, value in record.items():
                    if _is_blank(row.get(col)) and not _is_blank(value):
                        row[col] = value
                        row_changed = True
                if row_changed:
                    updated += 1
            else:
                new_row = {col: "" for col in column_order}
                new_row.update(record)
                existing_records[key] = new_row
                created += 1

    # Normalise rows to shared column ordering.
    final_columns: list[str] = []
    for col in canonical + column_order:
        if col not in final_columns:
            final_columns.append(col)

    rows = []
    for row in existing_records.values():
        normalised = {col: _format_value(row.get(col)) for col in final_columns}
        rows.append(normalised)

    def _sort_key(row: dict) -> datetime:
        ts = _format_value(row.get("timestamp", ""))
        try:
            parsed = datetime.fromisoformat(ts)
        except ValueError:
            parsed = datetime.min
        # csv rows may contain naive timestamps from older logs, so normalise to UTC
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed

    rows.sort(key=_sort_key)

    with run_log_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=final_columns)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "created": created,
        "updated": updated,
        "total": len(rows),
        "columns": final_columns,
    }


if __name__ == "__main__":
    base_dir = Path("artifacts/experiments")
    summary = sync_run_history(
        base_dir,
        base_dir / "baseline_run_history.csv",
    )
    print(
        f"Run history sync complete — "
        f"{summary['created']} new entries, {summary['updated']} updated entries, "
        f"{summary['total']} total rows."
    )
