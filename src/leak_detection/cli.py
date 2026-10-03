"""Command-line interface for leak model training and held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import (
    DEFAULT_LEAK_CLASSES,
    DIRECT_LEAK_COLUMNS,
    FeatureConfig,
    load_split,
)
from .model import (
    choose_threshold,
    classification_metrics,
    fit_calibrated_model,
    leak_probabilities,
)

ARTIFACT_VERSION = 1


def comma_separated(value: str) -> tuple[str, ...]:
    items = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Supply at least one comma-separated value")
    return items


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def predictions_frame(
    metadata: pd.DataFrame,
    targets: pd.Series,
    probabilities: Any,
    threshold: float,
) -> pd.DataFrame:
    result = metadata.copy()
    result["actual_is_leak"] = targets.to_numpy()
    result["leak_probability"] = probabilities
    result["predicted_is_leak"] = (probabilities >= threshold).astype(int)
    return result


def run_train(args: argparse.Namespace) -> None:
    excluded = tuple(sorted(set(DIRECT_LEAK_COLUMNS) | set(args.exclude_columns)))
    config = FeatureConfig(
        max_time_seconds=args.max_time_seconds,
        excluded_columns=excluded,
    )

    train_features, train_targets, _ = load_split(
        args.train_dir, args.positive_classes, config
    )
    validation_features, validation_targets, validation_metadata = load_split(
        args.validation_dir, args.positive_classes, config
    )

    feature_columns = train_features.columns.tolist()
    validation_features = validation_features.reindex(columns=feature_columns)
    model = fit_calibrated_model(train_features, train_targets, args.random_state)
    validation_probabilities = leak_probabilities(model, validation_features)
    threshold = choose_threshold(
        validation_targets, validation_probabilities, args.minimum_recall
    )
    metrics = classification_metrics(
        validation_targets, validation_probabilities, threshold
    )

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model": model,
        "threshold": threshold,
        "feature_columns": feature_columns,
        "feature_config": {
            "max_time_seconds": config.max_time_seconds,
            "time_column": config.time_column,
            "excluded_columns": list(config.excluded_columns),
        },
        "positive_classes": sorted(args.positive_classes),
        "validation_metrics": metrics,
        "random_state": args.random_state,
    }
    artifact_path = Path(args.output)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    metrics_path = artifact_path.with_suffix(".validation_metrics.json")
    predictions_path = artifact_path.with_suffix(".validation_predictions.csv")
    write_json(metrics_path, metrics)
    predictions_frame(
        validation_metadata,
        validation_targets,
        validation_probabilities,
        threshold,
    ).to_csv(predictions_path, index=False)

    print(f"Saved model: {artifact_path}")
    print(f"Saved validation metrics: {metrics_path}")
    print(f"Saved validation predictions: {predictions_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported or invalid model artifact")
    return artifact


def run_evaluate(args: argparse.Namespace) -> None:
    artifact = load_artifact(args.model)
    stored_config = artifact["feature_config"]
    config = FeatureConfig(
        max_time_seconds=float(stored_config["max_time_seconds"]),
        time_column=str(stored_config["time_column"]),
        excluded_columns=tuple(stored_config["excluded_columns"]),
    )
    test_features, test_targets, test_metadata = load_split(
        args.test_dir, artifact["positive_classes"], config
    )
    test_features = test_features.reindex(columns=artifact["feature_columns"])
    probabilities = leak_probabilities(artifact["model"], test_features)
    metrics = classification_metrics(test_targets, probabilities, artifact["threshold"])

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "test_metrics.json"
    predictions_path = output_directory / "test_predictions.csv"
    write_json(metrics_path, metrics)
    predictions_frame(
        test_metadata, test_targets, probabilities, artifact["threshold"]
    ).to_csv(predictions_path, index=False)

    print(f"Saved test metrics: {metrics_path}")
    print(f"Saved test predictions: {predictions_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate a calibrated gradient-boosting leak detector."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train", help="Fit on train and select the alert threshold on validation."
    )
    train.add_argument("--train-dir", required=True)
    train.add_argument("--validation-dir", required=True)
    train.add_argument("--output", default="artifacts/leak_model.joblib")
    train.add_argument(
        "--positive-classes",
        type=comma_separated,
        default=tuple(sorted(DEFAULT_LEAK_CLASSES)),
        help="Comma-separated scenario folders treated as leaks.",
    )
    train.add_argument(
        "--exclude-columns",
        type=comma_separated,
        default=(),
        help="Additional sensor columns to exclude beyond direct leak fields.",
    )
    train.add_argument("--max-time-seconds", type=float, default=1500.0)
    train.add_argument("--minimum-recall", type=float, default=0.95)
    train.add_argument("--random-state", type=int, default=42)
    train.set_defaults(handler=run_train)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate one finalized artifact on the untouched test split."
    )
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--test-dir", required=True)
    evaluate.add_argument("--output-dir", default="reports/final_test")
    evaluate.set_defaults(handler=run_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
