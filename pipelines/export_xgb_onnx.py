"""Convert notebook XGBoost JSON models to ONNX for Node inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import xgboost as xgb
from onnxmltools.convert.xgboost import convert as convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


def export_model(run_dir: Path, view: str, output_dir: Path) -> None:
    view_dir = run_dir / view
    model_path = view_dir / "model.json"
    metrics_path = view_dir / "metrics.json"
    if not model_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Expected files not found for {view} in {run_dir}")

    metrics = json.loads(metrics_path.read_text())
    feature_names = metrics.get("feature_cols")
    if not feature_names:
        raise ValueError(f"metrics.json missing feature_cols for {view}")

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    original_save_config = booster.save_config

    def sanitized_save_config(*args, **kwargs):  # type: ignore[override]
        config = json.loads(original_save_config(*args, **kwargs))
        base_score = config["learner"]["learner_model_param"].get("base_score")
        if base_score and base_score.startswith("["):
            config["learner"]["learner_model_param"]["base_score"] = "0.5"
        return json.dumps(config)

    booster.save_config = sanitized_save_config  # type: ignore[assignment]
    initial_types = [("input", FloatTensorType([None, len(feature_names)]))]
    normalized_feature_names = [f"f{i}" for i in range(len(feature_names))]
    if booster.feature_names is None or len(booster.feature_names) != len(feature_names):
        booster.feature_names = normalized_feature_names  # type: ignore[assignment]
    else:
        try:
            [int(name[1:]) for name in booster.feature_names]
        except Exception:
            booster.feature_names = normalized_feature_names  # type: ignore[assignment]
    booster.feature_types = None  # type: ignore[assignment]

    onnx_model = convert_xgboost(booster, initial_types=initial_types, target_opset=15)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{view}.onnx"
    with onnx_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())

    bundle = {
        "feature_schema_version": metrics.get("dataset_label", "v7"),
        "feature_names": feature_names,
        "trainer": metrics.get("trainer", "xgboost"),
    }
    (output_dir / f"{view}_preprocessing.json").write_text(
        json.dumps(bundle, indent=2),
        encoding="utf-8",
    )
    print(f"Exported {view} ONNX to {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export run directory to ONNX")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--views", nargs="*", default=["performance_dense", "market_gradient_boost"], help="Model view directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for view in args.views:
        export_model(args.run_dir, view, args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
