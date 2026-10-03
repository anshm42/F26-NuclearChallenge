"""Model fitting, threshold selection, and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def fit_calibrated_model(
    features: pd.DataFrame, targets: pd.Series, random_state: int = 42
) -> CalibratedClassifierCV:
    """Fit XGBoost and calibrate probabilities using training data only."""

    counts = targets.value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("Training data must contain both leak and non-leak runs")
    calibration_folds = min(5, int(counts.min()))
    if calibration_folds < 2:
        raise ValueError("Training data needs at least two runs from each class")

    estimator = XGBClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )
    cross_validation = StratifiedKFold(
        n_splits=calibration_folds, shuffle=True, random_state=random_state
    )
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=cross_validation,
    )
    calibrated.fit(features, targets)
    return calibrated


def fit_calibrated_type_model(
    features: pd.DataFrame,
    targets: pd.Series,
    class_count: int,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Fit calibrated multiclass XGBoost using leak runs only."""

    counts = targets.value_counts()
    if len(counts) != class_count or set(counts.index) != set(range(class_count)):
        raise ValueError("Training data must contain every configured leak type")
    calibration_folds = min(5, int(counts.min()))
    if calibration_folds < 2:
        raise ValueError("Training data needs at least two runs from each leak type")

    estimator = XGBClassifier(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=class_count,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )
    cross_validation = StratifiedKFold(
        n_splits=calibration_folds, shuffle=True, random_state=random_state
    )
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid",
        cv=cross_validation,
    )
    calibrated.fit(features, targets)
    return calibrated


def leak_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return the probability assigned to class 1 regardless of class ordering."""

    class_indices = np.flatnonzero(model.classes_ == 1)
    if class_indices.size != 1:
        raise ValueError("Model does not contain the leak class (1)")
    return model.predict_proba(features)[:, int(class_indices[0])]


def choose_threshold(
    targets: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float = 0.95,
) -> float:
    """Choose the highest-precision validation threshold meeting minimum recall."""

    if not 0.0 < minimum_recall <= 1.0:
        raise ValueError("minimum_recall must be in (0, 1]")
    targets_array = np.asarray(targets, dtype=int)
    if np.sum(targets_array == 1) == 0:
        raise ValueError("Validation data must contain at least one leak run")

    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    choices: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        true_positives = int(np.sum(predictions & (targets_array == 1)))
        predicted_positives = int(np.sum(predictions))
        recall = true_positives / int(np.sum(targets_array == 1))
        if recall >= minimum_recall:
            precision = (
                true_positives / predicted_positives if predicted_positives else 0.0
            )
            choices.append((precision, float(threshold), recall))

    # Safety-first: avoid missed leaks, then minimize false alarms.
    return max(choices, key=lambda item: (item[2], item[0], item[1]))[1]


def multiclass_metrics(
    targets: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute probability, accuracy, and per-type recall metrics."""

    targets_array = np.asarray(targets, dtype=int)
    predictions = np.argmax(probabilities, axis=1)
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(targets_array, predictions, labels=labels)
    top_k = min(2, len(class_names))
    top_predictions = np.argsort(probabilities, axis=1)[:, -top_k:]

    per_type_recall = {}
    for index, name in enumerate(class_names):
        actual_count = int(np.sum(targets_array == index))
        correct_count = int(matrix[index, index])
        per_type_recall[name] = correct_count / actual_count if actual_count else None

    return {
        "runs": int(targets_array.size),
        "accuracy": float(np.mean(predictions == targets_array)),
        "top_2_accuracy": float(
            np.mean([target in choices for target, choices in zip(targets_array, top_predictions)])
        ),
        "log_loss": float(log_loss(targets_array, probabilities, labels=labels)),
        "per_type_recall": per_type_recall,
        "confusion_matrix": matrix.astype(int).tolist(),
        "class_names": class_names,
    }


def classification_metrics(
    targets: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    """Compute probability and alert metrics for a data split."""

    targets_array = np.asarray(targets, dtype=int)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        targets_array, predictions, labels=[0, 1]
    ).ravel()
    both_classes = np.unique(targets_array).size == 2

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "threshold": float(threshold),
        "runs": int(targets_array.size),
        "leak_runs": int(np.sum(targets_array == 1)),
        "non_leak_runs": int(np.sum(targets_array == 0)),
        "roc_auc": float(roc_auc_score(targets_array, probabilities))
        if both_classes
        else None,
        "average_precision": float(
            average_precision_score(targets_array, probabilities)
        )
        if np.any(targets_array == 1)
        else None,
        "brier_score": float(brier_score_loss(targets_array, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
