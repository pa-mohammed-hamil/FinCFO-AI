"""
FinCo AI - Fraud Detection Model Evaluation

File:
    ml/fraud/evaluate.py

Purpose:
    Evaluate trained fraud detection models using:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - ROC-AUC
        - PR-AUC
        - Specificity
        - False Positive Rate
        - False Negative Rate
        - Fraud Capture Rate
        - Confusion Matrix

Workflow:
    Saved Model
        ↓
    Test Dataset
        ↓
    Fraud Predictions
        ↓
    Threshold Evaluation
        ↓
    fraud_metrics.py
        ↓
    Model Evaluation
        ↓
    JSON + Text Report
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from ml.evaluation.fraud_metrics import (
    calculate_fraud_metrics,
    evaluate_threshold,
    evaluate_thresholds,
    format_fraud_metrics,
    select_best_fraud_model,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.50
DEFAULT_OUTPUT_DIR = Path("artifacts/fraud")
DEFAULT_JSON_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "fraud_evaluation.json"
)
DEFAULT_REPORT_OUTPUT = (
    DEFAULT_OUTPUT_DIR / "fraud_evaluation.txt"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(
    level: int = logging.INFO,
) -> None:
    """Configure application logging."""

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_test_data(
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Validate fraud test data."""

    if not isinstance(X_test, np.ndarray):
        raise TypeError(
            "X_test must be a numpy array."
        )

    if not isinstance(y_test, np.ndarray):
        raise TypeError(
            "y_test must be a numpy array."
        )

    if X_test.ndim != 2:
        raise ValueError(
            "X_test must be a 2-dimensional array."
        )

    y_test = y_test.reshape(-1)

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test must contain "
            "the same number of samples."
        )

    if len(X_test) == 0:
        raise ValueError(
            "Test dataset cannot be empty."
        )

    if not np.all(np.isfinite(X_test)):
        raise ValueError(
            "X_test contains NaN or infinite values."
        )

    unique_labels = np.unique(y_test)

    if not np.all(
        np.isin(unique_labels, [0, 1])
    ):
        raise ValueError(
            "Fraud labels must contain only 0 and 1."
        )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_numpy_data(
    x_path: str | Path,
    y_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load fraud test data from NumPy files.

    Expected:
        X_test.npy
        y_test.npy
    """

    x_file = Path(x_path)
    y_file = Path(y_path)

    if not x_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {x_file}"
        )

    if not y_file.exists():
        raise FileNotFoundError(
            f"Target file not found: {y_file}"
        )

    X_test = np.load(x_file)
    y_test = np.load(y_file).reshape(-1)

    validate_test_data(
        X_test,
        y_test,
    )

    logger.info(
        "Loaded fraud test data: "
        "samples=%d, features=%d, fraud=%d",
        X_test.shape[0],
        X_test.shape[1],
        int(np.sum(y_test == 1)),
    )

    return X_test, y_test


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str | Path,
) -> Any:
    """Load a trained fraud model."""

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    model = joblib.load(path)

    logger.info(
        "Loaded model from %s",
        path,
    )

    return model


# ---------------------------------------------------------------------------
# Probability Prediction
# ---------------------------------------------------------------------------

def generate_probabilities(
    model: Any,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Generate probability of fraud.

    Preferred:
        predict_proba()

    Fallback:
        decision_function()
    """

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            X_test
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if probabilities.ndim != 2:
            raise ValueError(
                "predict_proba() must return "
                "a 2D probability matrix."
            )

        if probabilities.shape[1] == 2:
            fraud_probabilities = (
                probabilities[:, 1]
            )

        else:
            raise ValueError(
                "Expected binary classification "
                "with two probability columns."
            )

    elif hasattr(model, "decision_function"):

        scores = np.asarray(
            model.decision_function(X_test),
            dtype=float,
        ).reshape(-1)

        # Convert decision scores to probabilities
        # using a numerically stable sigmoid.
        fraud_probabilities = np.empty_like(
            scores,
            dtype=float,
        )

        positive = scores >= 0

        fraud_probabilities[positive] = (
            1.0
            / (
                1.0
                + np.exp(-scores[positive])
            )
        )

        exp_scores = np.exp(
            scores[~positive]
        )

        fraud_probabilities[~positive] = (
            exp_scores
            / (
                1.0
                + exp_scores
            )
        )

    else:
        raise AttributeError(
            "Model must implement either "
            "predict_proba() or "
            "decision_function()."
        )

    fraud_probabilities = np.clip(
        fraud_probabilities,
        0.0,
        1.0,
    )

    if len(fraud_probabilities) != len(X_test):
        raise ValueError(
            "Number of predictions does not "
            "match number of test samples."
        )

    if not np.all(
        np.isfinite(fraud_probabilities)
    ):
        raise ValueError(
            "Model generated invalid probabilities."
        )

    return fraud_probabilities


# ---------------------------------------------------------------------------
# Prediction Labels
# ---------------------------------------------------------------------------

def probabilities_to_labels(
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """
    Convert fraud probabilities into binary labels.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    return (
        probabilities >= threshold
    ).astype(int)


# ---------------------------------------------------------------------------
# Single Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = "fraud_model",
) -> Dict[str, Any]:
    """
    Evaluate a trained fraud detection model.
    """

    validate_test_data(
        X_test,
        y_test,
    )

    probabilities = generate_probabilities(
        model,
        X_test,
    )

    predictions = probabilities_to_labels(
        probabilities,
        threshold=threshold,
    )

    metrics = calculate_fraud_metrics(
        y_test,
        predictions,
        probabilities,
    )

    return {
        "model_name": model_name,
        "threshold": float(threshold),
        "metrics": metrics,
        "prediction_count": int(
            len(predictions)
        ),
        "fraud_count": int(
            np.sum(y_test == 1)
        ),
        "predicted_fraud_count": int(
            np.sum(predictions == 1)
        ),
    }


# ---------------------------------------------------------------------------
# Threshold Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_thresholds(
    y_test: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Optional[list[float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance across thresholds.

    Useful for fraud systems because the optimal threshold
    depends on the cost of false positives vs false negatives.
    """

    if thresholds is None:
        thresholds = [
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
        ]

    return evaluate_thresholds(
        y_test,
        probabilities,
        thresholds,
    )


# ---------------------------------------------------------------------------
# Best Threshold Selection
# ---------------------------------------------------------------------------

def select_best_threshold(
    threshold_results: Dict[str, Dict[str, float]],
    metric: str = "f1",
) -> tuple[float, Dict[str, float]]:
    """
    Select the best fraud threshold according to a metric.

    Default:
        F1 score
    """

    if not threshold_results:
        raise ValueError(
            "threshold_results cannot be empty."
        )

    higher_is_better = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "specificity",
        "fraud_capture_rate",
    }

    if metric not in higher_is_better:
        raise ValueError(
            f"Unsupported threshold metric: {metric}"
        )

    valid_results = {
        threshold: values
        for threshold, values
        in threshold_results.items()
        if metric in values
    }

    if not valid_results:
        raise ValueError(
            f"Metric '{metric}' not found "
            "in threshold results."
        )

    best_threshold_string = max(
        valid_results,
        key=lambda key: (
            valid_results[key][metric]
        ),
    )

    best_threshold = float(
        best_threshold_string
    )

    return (
        best_threshold,
        valid_results[
            best_threshold_string
        ],
    )


# ---------------------------------------------------------------------------
# Evaluation Summary
# ---------------------------------------------------------------------------

def build_evaluation_summary(
    evaluation: Dict[str, Any],
    threshold_results: Optional[
        Dict[str, Dict[str, float]]
    ] = None,
    best_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a complete evaluation summary.
    """

    summary = {
        "model_name": evaluation[
            "model_name"
        ],
        "threshold": evaluation[
            "threshold"
        ],
        "metrics": evaluation[
            "metrics"
        ],
        "prediction_count": evaluation[
            "prediction_count"
        ],
        "fraud_count": evaluation[
            "fraud_count"
        ],
        "predicted_fraud_count": evaluation[
            "predicted_fraud_count"
        ],
    }

    if threshold_results is not None:
        summary[
            "threshold_evaluation"
        ] = threshold_results

    if best_threshold is not None:
        summary[
            "recommended_threshold"
        ] = best_threshold

    return summary


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

def save_json(
    data: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save evaluation results as JSON."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Saved evaluation JSON: %s",
        path,
    )

    return path


# ---------------------------------------------------------------------------
# Text Report
# ---------------------------------------------------------------------------

def create_text_report(
    evaluation: Dict[str, Any],
    threshold_results: Optional[
        Dict[str, Dict[str, float]]
    ] = None,
    best_threshold: Optional[float] = None,
) -> str:
    """
    Generate human-readable fraud evaluation report.
    """

    lines = [
        "FinCo AI - Fraud Detection Evaluation",
        "=" * 50,
        "",
        f"Model: {evaluation['model_name']}",
        f"Threshold: {evaluation['threshold']:.4f}",
        f"Test samples: {evaluation['prediction_count']}",
        f"Actual fraud: {evaluation['fraud_count']}",
        (
            "Predicted fraud: "
            f"{evaluation['predicted_fraud_count']}"
        ),
        "",
        "Performance Metrics",
        "-" * 50,
    ]

    metrics = evaluation["metrics"]

    lines.append(
        format_fraud_metrics(metrics)
    )

    if best_threshold is not None:

        lines.extend(
            [
                "",
                "Recommended Threshold",
                "-" * 50,
                f"{best_threshold:.4f}",
            ]
        )

    if threshold_results:

        lines.extend(
            [
                "",
                "Threshold Analysis",
                "-" * 50,
            ]
        )

        for threshold, values in (
            threshold_results.items()
        ):

            lines.append(
                f"Threshold {threshold}: "
                f"F1={values.get('f1', 0.0):.4f}, "
                f"Precision="
                f"{values.get('precision', 0.0):.4f}, "
                f"Recall="
                f"{values.get('recall', 0.0):.4f}"
            )

    return "\n".join(lines)


def save_text_report(
    report: str,
    output_path: str | Path,
) -> Path:
    """Save text evaluation report."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        report,
        encoding="utf-8",
    )

    logger.info(
        "Saved evaluation report: %s",
        path,
    )

    return path


# ---------------------------------------------------------------------------
# Complete Evaluation Pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    model_path: str | Path,
    x_test_path: str | Path,
    y_test_path: str | Path,
    model_name: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    evaluate_all_thresholds: bool = True,
    threshold_metric: str = "f1",
    json_output: str | Path = DEFAULT_JSON_OUTPUT,
    report_output: str | Path = DEFAULT_REPORT_OUTPUT,
) -> Dict[str, Any]:
    """
    Run the complete fraud model evaluation pipeline.
    """

    model = load_model(
        model_path
    )

    X_test, y_test = load_numpy_data(
        x_test_path,
        y_test_path,
    )

    if model_name is None:
        model_name = Path(
            model_path
        ).stem

    logger.info(
        "Evaluating fraud model: %s",
        model_name,
    )

    evaluation = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        model_name=model_name,
    )

    probabilities = generate_probabilities(
        model,
        X_test,
    )

    threshold_results = None
    best_threshold = None

    if evaluate_all_thresholds:

        threshold_results = (
            evaluate_model_thresholds(
                y_test,
                probabilities,
            )
        )

        best_threshold, _ = (
            select_best_threshold(
                threshold_results,
                metric=threshold_metric,
            )
        )

        logger.info(
            "Recommended threshold: %.4f",
            best_threshold,
        )

    summary = build_evaluation_summary(
        evaluation=evaluation,
        threshold_results=threshold_results,
        best_threshold=best_threshold,
    )

    save_json(
        summary,
        json_output,
    )

    report = create_text_report(
        evaluation=evaluation,
        threshold_results=threshold_results,
        best_threshold=best_threshold,
    )

    save_text_report(
        report,
        report_output,
    )

    return summary


# ---------------------------------------------------------------------------
# Console Output
# ---------------------------------------------------------------------------

def print_evaluation_summary(
    summary: Dict[str, Any],
) -> None:
    """Print evaluation results."""

    print()
    print(
        "FinCo AI Fraud Detection Evaluation"
    )
    print("=" * 45)

    print(
        f"Model     : "
        f"{summary['model_name']}"
    )

    print(
        f"Threshold : "
        f"{summary['threshold']:.4f}"
    )

    print(
        f"Samples   : "
        f"{summary['prediction_count']}"
    )

    print(
        f"Fraud     : "
        f"{summary['fraud_count']}"
    )

    print(
        f"Predicted : "
        f"{summary['predicted_fraud_count']}"
    )

    print()
    print("Metrics")
    print("-" * 45)

    for name, value in (
        summary["metrics"].items()
    ):

        if isinstance(value, (int, float)):

            print(
                f"{name:<25} "
                f"{value:.4f}"
            )

    if (
        "recommended_threshold"
        in summary
    ):

        print()
        print(
            "Recommended Threshold: "
            f"{summary['recommended_threshold']:.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FinCo AI fraud detection models."
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to trained fraud model.",
    )

    parser.add_argument(
        "--x-test",
        required=True,
        type=str,
        help="Path to X_test.npy.",
    )

    parser.add_argument(
        "--y-test",
        required=True,
        type=str,
        help="Path to y_test.npy.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model name.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Fraud classification threshold.",
    )

    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="f1",
        choices=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "fraud_capture_rate",
        ],
        help=(
            "Metric used to select the recommended "
            "threshold."
        ),
    )

    parser.add_argument(
        "--no-threshold-search",
        action="store_true",
        help=(
            "Disable evaluation across multiple "
            "classification thresholds."
        ),
    )

    parser.add_argument(
        "--json-output",
        type=str,
        default=str(
            DEFAULT_JSON_OUTPUT
        ),
        help="Evaluation JSON output path.",
    )

    parser.add_argument(
        "--report-output",
        type=str,
        default=str(
            DEFAULT_REPORT_OUTPUT
        ),
        help="Text report output path.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""

    configure_logging()

    parser = build_parser()
    args = parser.parse_args()

    summary = run_evaluation(
        model_path=args.model,
        x_test_path=args.x_test,
        y_test_path=args.y_test,
        model_name=args.model_name,
        threshold=args.threshold,
        evaluate_all_thresholds=(
            not args.no_threshold_search
        ),
        threshold_metric=args.threshold_metric,
        json_output=args.json_output,
        report_output=args.report_output,
    )

    print_evaluation_summary(
        summary
    )


if __name__ == "__main__":
    main()