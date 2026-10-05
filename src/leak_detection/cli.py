"""Command-line interface for leak model training and held-out evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict

import joblib
import pandas as pd

from .features import (
    DEFAULT_LEAK_CLASSES,
    DIRECT_LEAK_COLUMNS,
    FeatureConfig,
    extract_run_features,
    load_split,
)
from .model import (
    choose_threshold,
    classification_metrics,
    fit_calibrated_model,
    fit_calibrated_type_model,
    leak_probabilities,
    multiclass_metrics,
)

ARTIFACT_VERSION = 1


def comma_separated(value: str) -> tuple[str, ...]:
    items = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("Supply at least one comma-separated value")
    return items


def write_json(path: Path, payload: object) -> None:
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


def load_type_split(
    directory: str | Path,
    leak_classes: tuple[str, ...] | list[str],
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, _, metadata = load_split(directory, leak_classes, config)
    mask = metadata["scenario"].isin(leak_classes).to_numpy()
    return (
        features.loc[mask].reset_index(drop=True),
        metadata.loc[mask].reset_index(drop=True),
    )


def type_predictions_frame(
    metadata: pd.DataFrame,
    targets: pd.Series,
    probabilities: Any,
    class_names: list[str],
) -> pd.DataFrame:
    result = metadata.copy()
    result["actual_type"] = [class_names[index] for index in targets]
    result["predicted_type"] = [
        class_names[index] for index in probabilities.argmax(axis=1)
    ]
    for index, class_name in enumerate(class_names):
        result[f"probability_{class_name}"] = probabilities[:, index]
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


def run_train_types(args: argparse.Namespace) -> None:
    excluded = tuple(sorted(set(DIRECT_LEAK_COLUMNS) | set(args.exclude_columns)))
    config = FeatureConfig(
        max_time_seconds=args.max_time_seconds,
        excluded_columns=excluded,
    )
    class_names = sorted(args.leak_classes)
    class_indices = {name: index for index, name in enumerate(class_names)}

    train_features, train_metadata = load_type_split(
        args.train_dir, args.leak_classes, config
    )
    validation_features, validation_metadata = load_type_split(
        args.validation_dir, args.leak_classes, config
    )
    missing = set(class_names) - set(train_metadata["scenario"])
    if missing:
        raise ValueError(f"Training data is missing leak types: {sorted(missing)}")

    train_targets = train_metadata["scenario"].map(class_indices).astype(int)
    validation_targets = validation_metadata["scenario"].map(class_indices).astype(int)
    feature_columns = train_features.columns.tolist()
    validation_features = validation_features.reindex(columns=feature_columns)
    model = fit_calibrated_type_model(
        train_features, train_targets, len(class_names), args.random_state
    )
    validation_probabilities = model.predict_proba(validation_features)
    metrics = multiclass_metrics(
        validation_targets, validation_probabilities, class_names
    )

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model_kind": "leak_type",
        "model": model,
        "feature_columns": feature_columns,
        "feature_config": {
            "max_time_seconds": config.max_time_seconds,
            "time_column": config.time_column,
            "excluded_columns": list(config.excluded_columns),
        },
        "class_names": class_names,
        "validation_metrics": metrics,
        "random_state": args.random_state,
    }
    artifact_path = Path(args.output)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)

    metrics_path = artifact_path.with_suffix(".validation_metrics.json")
    predictions_path = artifact_path.with_suffix(".validation_predictions.csv")
    write_json(metrics_path, metrics)
    type_predictions_frame(
        validation_metadata,
        validation_targets,
        validation_probabilities,
        class_names,
    ).to_csv(predictions_path, index=False)

    print(f"Saved leak-type model: {artifact_path}")
    print(f"Saved validation metrics: {metrics_path}")
    print(f"Saved validation predictions: {predictions_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("Unsupported or invalid model artifact")
    return artifact


def config_from_artifact(artifact: dict[str, Any]) -> FeatureConfig:
    stored_config = artifact["feature_config"]
    return FeatureConfig(
        max_time_seconds=float(stored_config["max_time_seconds"]),
        time_column=str(stored_config["time_column"]),
        excluded_columns=tuple(stored_config["excluded_columns"]),
    )


def evaluate_binary_artifact(
    test_directory: str | Path, artifact: dict[str, Any]
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    config = config_from_artifact(artifact)
    test_features, test_targets, test_metadata = load_split(
        test_directory, artifact["positive_classes"], config
    )
    test_features = test_features.reindex(columns=artifact["feature_columns"])
    probabilities = leak_probabilities(artifact["model"], test_features)
    threshold = float(artifact["threshold"])
    metrics = classification_metrics(test_targets, probabilities, threshold)
    predictions = predictions_frame(
        test_metadata, test_targets, probabilities, threshold
    )
    return metrics, predictions


def run_evaluate(args: argparse.Namespace) -> None:
    artifact = load_artifact(args.model)
    metrics, predictions = evaluate_binary_artifact(args.test_dir, artifact)

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "test_metrics.json"
    predictions_path = output_directory / "test_predictions.csv"
    write_json(metrics_path, metrics)
    predictions.to_csv(predictions_path, index=False)

    print(f"Saved test metrics: {metrics_path}")
    print(f"Saved test predictions: {predictions_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def evaluate_type_artifact(
    test_directory: str | Path, artifact: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    if artifact.get("model_kind") != "leak_type":
        raise ValueError("Artifact is not a leak-type model")
    config = config_from_artifact(artifact)
    class_names = artifact["class_names"]
    class_indices = {name: index for index, name in enumerate(class_names)}
    test_features, test_metadata = load_type_split(
        test_directory, class_names, config
    )
    test_targets = test_metadata["scenario"].map(class_indices).astype(int)
    test_features = test_features.reindex(columns=artifact["feature_columns"])
    probabilities = artifact["model"].predict_proba(test_features)
    metrics = multiclass_metrics(test_targets, probabilities, class_names)
    predictions = type_predictions_frame(
        test_metadata, test_targets, probabilities, class_names
    )
    return metrics, predictions


def run_evaluate_types(args: argparse.Namespace) -> None:
    artifact = load_artifact(args.model)
    metrics, predictions = evaluate_type_artifact(args.test_dir, artifact)

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "leak_type_test_metrics.json"
    predictions_path = output_directory / "leak_type_test_predictions.csv"
    write_json(metrics_path, metrics)
    predictions.to_csv(predictions_path, index=False)

    print(f"Saved leak-type test metrics: {metrics_path}")
    print(f"Saved leak-type test predictions: {predictions_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


class PredictionResult(TypedDict):
    input_csv: str
    leak_probability: float
    alert_threshold: float
    leak_alert: bool
    predicted_leak_type: str | None
    leak_type_probabilities: dict[str, float]
    binary_window_seconds: float
    type_window_seconds: float


def features_for_artifact(
    csv_path: str | Path, artifact: dict[str, Any]
) -> pd.DataFrame:
    config = config_from_artifact(artifact)
    features = extract_run_features(csv_path, config).to_frame().T
    return features.reindex(columns=artifact["feature_columns"])


def predict_scenario(
    csv_path: str | Path,
    binary_artifact: dict[str, Any],
    type_artifact: dict[str, Any],
    input_name: str | None = None,
) -> PredictionResult:
    if type_artifact.get("model_kind") != "leak_type":
        raise ValueError("Type artifact is not a leak-type model")

    binary_features = features_for_artifact(csv_path, binary_artifact)
    leak_probability = float(
        leak_probabilities(binary_artifact["model"], binary_features)[0]
    )
    threshold = float(binary_artifact["threshold"])
    leak_alert = leak_probability >= threshold

    type_results: dict[str, float] = {}
    most_likely_type = None
    if leak_alert:
        type_features = features_for_artifact(csv_path, type_artifact)
        type_probabilities = type_artifact["model"].predict_proba(type_features)[0]
        class_names = type_artifact["class_names"]
        type_results = {
            name: float(type_probabilities[index])
            for index, name in enumerate(class_names)
        }
        most_likely_type = max(type_results, key=lambda name: type_results[name])

    return {
        "input_csv": input_name or str(csv_path),
        "leak_probability": leak_probability,
        "alert_threshold": threshold,
        "leak_alert": leak_alert,
        "predicted_leak_type": most_likely_type,
        "leak_type_probabilities": type_results,
        "binary_window_seconds": float(
            binary_artifact["feature_config"]["max_time_seconds"]
        ),
        "type_window_seconds": float(
            type_artifact["feature_config"]["max_time_seconds"]
        ),
    }


def run_predict(args: argparse.Namespace) -> None:
    binary_artifact = load_artifact(args.binary_model)
    type_artifact = load_artifact(args.type_model)
    result = predict_scenario(args.input_csv, binary_artifact, type_artifact)
    if args.output:
        write_json(Path(args.output), result)
        print(f"Saved prediction: {args.output}")
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or run calibrated XGBoost leak models."
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
    train.add_argument("--max-time-seconds", type=float, default=120.0)
    train.add_argument("--minimum-recall", type=float, default=1.0)
    train.add_argument("--random-state", type=int, default=42)
    train.set_defaults(handler=run_train)

    train_types = subparsers.add_parser(
        "train-types", help="Fit a second-stage leak-type probability model."
    )
    train_types.add_argument("--train-dir", required=True)
    train_types.add_argument("--validation-dir", required=True)
    train_types.add_argument("--output", default="artifacts/leak_type_model.joblib")
    train_types.add_argument(
        "--leak-classes",
        type=comma_separated,
        default=tuple(sorted(DEFAULT_LEAK_CLASSES)),
    )
    train_types.add_argument(
        "--exclude-columns",
        type=comma_separated,
        default=(),
    )
    train_types.add_argument("--max-time-seconds", type=float, default=120.0)
    train_types.add_argument("--random-state", type=int, default=42)
    train_types.set_defaults(handler=run_train_types)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate one finalized artifact on the untouched test split."
    )
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--test-dir", required=True)
    evaluate.add_argument("--output-dir", default="reports/final_test")
    evaluate.set_defaults(handler=run_evaluate)

    evaluate_types = subparsers.add_parser(
        "evaluate-types", help="Evaluate a frozen leak-type model."
    )
    evaluate_types.add_argument("--model", required=True)
    evaluate_types.add_argument("--test-dir", required=True)
    evaluate_types.add_argument("--output-dir", default="reports/final_type_test")
    evaluate_types.set_defaults(handler=run_evaluate_types)

    predict = subparsers.add_parser(
        "predict", help="Analyze one simulation CSV with both saved models."
    )
    predict.add_argument("--binary-model", required=True)
    predict.add_argument("--type-model", required=True)
    predict.add_argument("--input-csv", required=True)
    predict.add_argument("--output")
    predict.set_defaults(handler=run_predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
