"""
FinCo AI - Fraud Detection Evaluation Metrics

File:
    ml/evaluation/fraud_metrics.py

Purpose:
    Evaluation utilities for fraud detection models.

Supported metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Specificity
    - FPR
    - FNR
    - ROC-AUC
    - PR-AUC
    - Confusion Matrix
    - Fraud Capture Rate
    - False Alarm Rate
    - Classification Summary

Designed for:
    - Transaction fraud detection
    - Financial anomaly detection
    - Credit-card fraud
    - Payment risk detection
    - Suspicious transaction classification

Important:
    Fraud datasets are usually highly imbalanced, so precision,
    recall, F1, PR-AUC, FPR and FNR are generally more informative
    than accuracy alone.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:
    raise ImportError(
        "scikit-learn is required for fraud_metrics.py. "
        "Install it with: pip install scikit-learn"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.5
POSITIVE_CLASS = 1


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_labels(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate binary actual and predicted labels.
    """

    y_true = np.asarray(list(actual))
    y_pred = np.asarray(list(predicted))

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "actual and predicted labels must be one-dimensional."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "actual and predicted labels must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "actual and predicted labels cannot be empty."
        )

    valid_values = {0, 1}

    if not set(np.unique(y_true)).issubset(valid_values):
        raise ValueError(
            "actual labels must contain only 0 and 1."
        )

    if not set(np.unique(y_pred)).issubset(valid_values):
        raise ValueError(
            "predicted labels must contain only 0 and 1."
        )

    return y_true.astype(int), y_pred.astype(int)


def _validate_probabilities(
    actual: Iterable[int],
    probabilities: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate binary labels and fraud probabilities.
    """

    y_true = np.asarray(list(actual), dtype=int)
    y_prob = np.asarray(list(probabilities), dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError(
            "actual and probabilities must be one-dimensional."
        )

    if len(y_true) != len(y_prob):
        raise ValueError(
            "actual and probabilities must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "actual and probabilities cannot be empty."
        )

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError(
            "actual labels must contain only 0 and 1."
        )

    if not np.all(np.isfinite(y_prob)):
        raise ValueError(
            "probabilities contain NaN or infinite values."
        )

    if np.any(y_prob < 0.0) or np.any(y_prob > 1.0):
        raise ValueError(
            "probabilities must be between 0 and 1."
        )

    return y_true, y_prob


# ---------------------------------------------------------------------------
# Prediction Conversion
# ---------------------------------------------------------------------------

def probabilities_to_labels(
    probabilities: Iterable[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """
    Convert fraud probabilities into binary predictions.

    probability >= threshold -> fraud (1)
    probability < threshold  -> legitimate (0)
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities_array = np.asarray(
        list(probabilities),
        dtype=float,
    )

    if probabilities_array.ndim != 1:
        raise ValueError(
            "probabilities must be one-dimensional."
        )

    if len(probabilities_array) == 0:
        raise ValueError(
            "probabilities cannot be empty."
        )

    if not np.all(np.isfinite(probabilities_array)):
        raise ValueError(
            "probabilities contain NaN or infinite values."
        )

    if np.any(probabilities_array < 0.0) or np.any(
        probabilities_array > 1.0
    ):
        raise ValueError(
            "probabilities must be between 0 and 1."
        )

    return (
        probabilities_array >= threshold
    ).astype(int)


# ---------------------------------------------------------------------------
# Basic Classification Metrics
# ---------------------------------------------------------------------------

def accuracy(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """Calculate classification accuracy."""

    y_true, y_pred = _validate_labels(
        actual,
        predicted,
    )

    return float(
        accuracy_score(y_true, y_pred)
    )


def precision(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """
    Calculate fraud precision.

    Precision answers:

        Of all transactions predicted as fraud,
        how many were actually fraudulent?
    """

    y_true, y_pred = _validate_labels(
        actual,
        predicted,
    )

    return float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


def recall(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """
    Calculate fraud recall.

    Recall answers:

        Of all actual fraudulent transactions,
        how many did the model detect?
    """

    y_true, y_pred = _validate_labels(
        actual,
        predicted,
    )

    return float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


def f1(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """
    Calculate F1 score.

    F1 balances precision and recall.
    """

    y_true, y_pred = _validate_labels(
        actual,
        predicted,
    )

    return float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


# ---------------------------------------------------------------------------
# Confusion Matrix Metrics
# ---------------------------------------------------------------------------

def get_confusion_matrix(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> Dict[str, int]:
    """
    Return confusion-matrix components.

    Returns:
        {
            "true_negative": TN,
            "false_positive": FP,
            "false_negative": FN,
            "true_positive": TP
        }
    """

    y_true, y_pred = _validate_labels(
        actual,
        predicted,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def specificity(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """
    Specificity / True Negative Rate.

    Specificity =
        TN / (TN + FP)

    Measures how well legitimate transactions are identified.
    """

    matrix = get_confusion_matrix(
        actual,
        predicted,
    )

    tn = matrix["true_negative"]
    fp = matrix["false_positive"]

    denominator = tn + fp

    if denominator == 0:
        return 0.0

    return float(tn / denominator)


def false_positive_rate(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """
    False Positive Rate.

    FPR =
        FP / (FP + TN)

    Lower is better.
    """

    return float(
        1.0 - specificity(
            actual,
            predicted,
        )
    )


def false_negative_rate(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """
    False Negative Rate.

    FNR =
        FN / (FN + TP)

    Lower is better.

    In fraud detection, reducing FNR is often critical because
    false negatives represent fraud that the system failed to detect.
    """

    matrix = get_confusion_matrix(
        actual,
        predicted,
    )

    fn = matrix["false_negative"]
    tp = matrix["true_positive"]

    denominator = fn + tp

    if denominator == 0:
        return 0.0

    return float(fn / denominator)


# ---------------------------------------------------------------------------
# Fraud-Specific Metrics
# ---------------------------------------------------------------------------

def fraud_capture_rate(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """
    Fraud Capture Rate.

    Equivalent to fraud recall.

    Returns value between 0 and 1.
    """

    return recall(
        actual,
        predicted,
    )


def fraud_capture_percentage(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """
    Fraud Capture Rate as a percentage.
    """

    return float(
        fraud_capture_rate(
            actual,
            predicted,
        ) * 100.0
    )


def false_alarm_rate(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """
    False Alarm Rate.

    Equivalent to False Positive Rate.

    Measures the proportion of legitimate transactions
    incorrectly flagged as fraud.
    """

    return false_positive_rate(
        actual,
        predicted,
    )


def fraud_precision_percentage(
    actual: Iterable[int],
    predicted: Iterable[int],
) -> float:
    """Fraud precision as a percentage."""

    return float(
        precision(
            actual,
            predicted,
        ) * 100.0
    )


# ---------------------------------------------------------------------------
# Probability-Based Metrics
# ---------------------------------------------------------------------------

def roc_auc(
    actual: Iterable[int],
    probabilities: Iterable[float],
) -> float:
    """
    ROC-AUC score.

    Higher is better.

    Requires both positive and negative classes to be present.
    """

    y_true, y_prob = _validate_probabilities(
        actual,
        probabilities,
    )

    if len(np.unique(y_true)) < 2:
        return 0.0

    return float(
        roc_auc_score(
            y_true,
            y_prob,
        )
    )


def pr_auc(
    actual: Iterable[int],
    probabilities: Iterable[float],
) -> float:
    """
    Precision-Recall AUC / Average Precision.

    Particularly useful for highly imbalanced fraud datasets.
    """

    y_true, y_prob = _validate_probabilities(
        actual,
        probabilities,
    )

    if np.sum(y_true) == 0:
        return 0.0

    return float(
        average_precision_score(
            y_true,
            y_prob,
        )
    )


# ---------------------------------------------------------------------------
# Threshold Evaluation
# ---------------------------------------------------------------------------

def evaluate_threshold(
    actual: Iterable[int],
    probabilities: Iterable[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    Evaluate a fraud model at a specific classification threshold.
    """

    y_true, y_prob = _validate_probabilities(
        actual,
        probabilities,
    )

    predicted = probabilities_to_labels(
        y_prob,
        threshold=threshold,
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy(y_true, predicted),
        "precision": precision(y_true, predicted),
        "recall": recall(y_true, predicted),
        "f1": f1(y_true, predicted),
        "specificity": specificity(y_true, predicted),
        "false_positive_rate": false_positive_rate(
            y_true,
            predicted,
        ),
        "false_negative_rate": false_negative_rate(
            y_true,
            predicted,
        ),
        "fraud_capture_rate": fraud_capture_rate(
            y_true,
            predicted,
        ),
    }


def evaluate_thresholds(
    actual: Iterable[int],
    probabilities: Iterable[float],
    thresholds: Optional[Iterable[float]] = None,
) -> list[Dict[str, float]]:
    """
    Evaluate a fraud model over multiple thresholds.

    Useful for selecting a business-appropriate operating point.
    """

    if thresholds is None:
        thresholds = np.arange(
            0.10,
            1.00,
            0.05,
        )

    y_true, y_prob = _validate_probabilities(
        actual,
        probabilities,
    )

    results = []

    for threshold in thresholds:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Invalid threshold: {threshold}"
            )

        results.append(
            evaluate_threshold(
                y_true,
                y_prob,
                threshold=float(threshold),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Comprehensive Evaluation
# ---------------------------------------------------------------------------

def calculate_fraud_metrics(
    actual: Iterable[int],
    predicted: Optional[Iterable[int]] = None,
    probabilities: Optional[Iterable[float]] = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """
    Calculate a comprehensive fraud-model evaluation.

    Either:
        predicted

    or:
        probabilities

    must be supplied.

    If probabilities are supplied, predicted labels are generated
    using the supplied threshold.
    """

    y_true = np.asarray(
        list(actual),
        dtype=int,
    )

    if probabilities is not None:

        y_true, y_prob = _validate_probabilities(
            y_true,
            probabilities,
        )

        y_pred = probabilities_to_labels(
            y_prob,
            threshold=threshold,
        )

    elif predicted is not None:

        y_true, y_pred = _validate_labels(
            y_true,
            predicted,
        )

        y_prob = None

    else:
        raise ValueError(
            "Either predicted or probabilities must be supplied."
        )

    matrix = get_confusion_matrix(
        y_true,
        y_pred,
    )

    results: Dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "false_positive_rate": false_positive_rate(
            y_true,
            y_pred,
        ),
        "false_negative_rate": false_negative_rate(
            y_true,
            y_pred,
        ),
        "fraud_capture_rate": fraud_capture_rate(
            y_true,
            y_pred,
        ),
        "confusion_matrix": matrix,
    }

    if y_prob is not None:
        results["roc_auc"] = roc_auc(
            y_true,
            y_prob,
        )

        results["pr_auc"] = pr_auc(
            y_true,
            y_prob,
        )

    return results


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------

def compare_fraud_models(
    actual: Iterable[int],
    predictions: Mapping[str, Iterable[int]],
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple fraud classification models.

    Example:

        predictions = {
            "logistic_regression": lr_predictions,
            "random_forest": rf_predictions,
            "xgboost": xgb_predictions,
        }

        results = compare_fraud_models(
            actual,
            predictions,
        )
    """

    y_true = list(actual)

    results: Dict[str, Dict[str, float]] = {}

    for model_name, model_predictions in predictions.items():

        metrics = calculate_fraud_metrics(
            y_true,
            predicted=model_predictions,
        )

        results[model_name] = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float))
        }

    return results


def select_best_fraud_model(
    metrics: Mapping[str, Mapping[str, float]],
    metric: str = "pr_auc",
) -> str:
    """
    Select the best fraud model.

    Higher is better for:
        accuracy
        precision
        recall
        f1
        specificity
        pr_auc
        roc_auc
        fraud_capture_rate

    Lower is better for:
        false_positive_rate
        false_negative_rate
    """

    if not metrics:
        raise ValueError(
            "metrics cannot be empty."
        )

    higher_is_better = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "pr_auc",
        "roc_auc",
        "fraud_capture_rate",
    }

    lower_is_better = {
        "false_positive_rate",
        "false_negative_rate",
    }

    supported = (
        higher_is_better |
        lower_is_better
    )

    if metric not in supported:
        raise ValueError(
            f"Unsupported metric: {metric}"
        )

    for model_name, model_metrics in metrics.items():

        if metric not in model_metrics:
            raise ValueError(
                f"Metric '{metric}' missing for "
                f"model '{model_name}'."
            )

    if metric in higher_is_better:
        return max(
            metrics,
            key=lambda name: metrics[name][metric],
        )

    return min(
        metrics,
        key=lambda name: metrics[name][metric],
    )


# ---------------------------------------------------------------------------
# Human-Readable Report
# ---------------------------------------------------------------------------

def format_fraud_metrics(
    metrics: Mapping[str, Any],
) -> str:
    """
    Format fraud metrics for CLI output, logs, or reports.
    """

    lines = [
        "Fraud Model Evaluation",
        "=" * 28,
    ]

    percentage_metrics = {
        "precision",
        "recall",
        "f1",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "fraud_capture_rate",
    }

    display_order = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "fraud_capture_rate",
        "roc_auc",
        "pr_auc",
    ]

    for name in display_order:

        if name not in metrics:
            continue

        value = metrics[name]

        label = name.replace(
            "_",
            " ",
        ).title()

        if name in percentage_metrics:
            lines.append(
                f"{label:<25}: {value * 100:.2f}%"
            )
        else:
            lines.append(
                f"{label:<25}: {value:.4f}"
            )

    if "confusion_matrix" in metrics:

        matrix = metrics["confusion_matrix"]

        lines.extend(
            [
                "",
                "Confusion Matrix",
                "-" * 20,
                f"True Negative : {matrix['true_negative']}",
                f"False Positive: {matrix['false_positive']}",
                f"False Negative: {matrix['false_negative']}",
                f"True Positive : {matrix['true_positive']}",
            ]
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    actual = [
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
    ]

    probabilities = [
        0.05,
        0.10,
        0.20,
        0.15,
        0.90,
        0.30,
        0.80,
        0.10,
        0.25,
        0.70,
    ]

    metrics = calculate_fraud_metrics(
        actual,
        probabilities=probabilities,
        threshold=0.50,
    )

    print(
        format_fraud_metrics(metrics)
    )

    print("\nThreshold Analysis")
    print("-" * 28)

    threshold_results = evaluate_thresholds(
        actual,
        probabilities,
        thresholds=[
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
        ],
    )

    for result in threshold_results:
        print(
            f"Threshold={result['threshold']:.2f} | "
            f"Precision={result['precision']:.3f} | "
            f"Recall={result['recall']:.3f} | "
            f"F1={result['f1']:.3f} | "
            f"FPR={result['false_positive_rate']:.3f}"
        )