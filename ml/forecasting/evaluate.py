```python
"""
FinCo AI - Forecasting Model Evaluation

File:
    ml/forecasting/evaluate.py

Purpose:
    Evaluate trained forecasting models against a test dataset.

Responsibilities:
    - Generate predictions
    - Calculate forecasting metrics
    - Compare multiple models
    - Select the best model
    - Save evaluation results
    - Produce human-readable reports

Expected model interface:
    model.predict(X)

Expected data:
    X_test -> feature matrix
    y_test -> actual target values
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from ml.evaluation.forecast_metrics import (
    calculate_forecast_metrics,
    compare_models,
    format_metrics,
    select_best_model,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


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
# Prediction
# ---------------------------------------------------------------------------

def generate_predictions(
    model: Any,
    X_test: Any,
) -> np.ndarray:
    """
    Generate predictions from a trained forecasting model.

    Parameters
    ----------
    model:
        Trained model exposing a predict() method.

    X_test:
        Test features.

    Returns
    -------
    np.ndarray
        One-dimensional prediction array.
    """

    if model is None:
        raise ValueError("model cannot be None.")

    if not hasattr(model, "predict"):
        raise TypeError(
            "model must provide a predict() method."
        )

    predictions = model.predict(X_test)

    predictions = np.asarray(
        predictions,
        dtype=float,
    ).reshape(-1)

    if len(predictions) == 0:
        raise ValueError(
            "Model returned empty predictions."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            "Model predictions contain NaN or infinite values."
        )

    return predictions


# ---------------------------------------------------------------------------
# Single Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: Iterable[float],
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Evaluate a single forecasting model.

    Returns a dictionary containing:
        model name
        sample count
        forecasting metrics
    """

    logger.info(
        "Evaluating forecasting model: %s",
        model_name,
    )

    y_true = np.asarray(
        list(y_test),
        dtype=float,
    ).reshape(-1)

    if len(y_true) == 0:
        raise ValueError(
            "y_test cannot be empty."
        )

    predictions = generate_predictions(
        model,
        X_test,
    )

    if len(y_true) != len(predictions):
        raise ValueError(
            "y_test and predictions must have "
            "the same number of samples."
        )

    metrics = calculate_forecast_metrics(
        y_true,
        predictions,
    )

    result: Dict[str, Any] = {
        "model": model_name,
        "samples": int(len(y_true)),
        "metrics": metrics,
    }

    logger.info(
        "Completed evaluation for %s",
        model_name,
    )

    return result


# ---------------------------------------------------------------------------
# Evaluate Predictions Directly
# ---------------------------------------------------------------------------

def evaluate_predictions(
    actual: Iterable[float],
    predicted: Iterable[float],
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Evaluate already-generated predictions.

    Useful when predictions have already been produced
    by a training/inference pipeline.
    """

    y_true = np.asarray(
        list(actual),
        dtype=float,
    )

    y_pred = np.asarray(
        list(predicted),
        dtype=float,
    )

    metrics = calculate_forecast_metrics(
        y_true,
        y_pred,
    )

    return {
        "model": model_name,
        "samples": int(len(y_true)),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Multiple Model Evaluation
# ---------------------------------------------------------------------------

def evaluate_models(
    actual: Iterable[float],
    predictions: Mapping[
        str,
        Iterable[float],
    ],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate predictions from multiple forecasting models.

    Example:

        predictions = {
            "linear_regression": lr_predictions,
            "random_forest": rf_predictions,
            "xgboost": xgb_predictions,
        }

        results = evaluate_models(
            actual,
            predictions,
        )
    """

    logger.info(
        "Evaluating %d forecasting models.",
        len(predictions),
    )

    results = compare_models(
        actual,
        predictions,
    )

    return results


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_models(
    results: Mapping[
        str,
        Mapping[str, float],
    ],
    metric: str = "rmse",
    ascending: Optional[bool] = None,
) -> list[Dict[str, Any]]:
    """
    Rank forecasting models by a selected metric.

    Parameters
    ----------
    results:
        Model metrics.

    metric:
        Metric used for ranking.

    ascending:
        If None, automatically determines direction.
    """

    lower_is_better = {
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "wape",
    }

    higher_is_better = {
        "r2",
        "accuracy",
    }

    if metric not in (
        lower_is_better | higher_is_better
    ):
        raise ValueError(
            f"Unsupported ranking metric: {metric}"
        )

    if ascending is None:
        ascending = metric in lower_is_better

    ranked = sorted(
        results.items(),
        key=lambda item: item[1][metric],
        reverse=not ascending,
    )

    output = []

    for position, (model_name, metrics) in enumerate(
        ranked,
        start=1,
    ):
        output.append(
            {
                "rank": position,
                "model": model_name,
                "metric": metric,
                "value": float(metrics[metric]),
                "metrics": dict(metrics),
            }
        )

    return output


# ---------------------------------------------------------------------------
# Best Model
# ---------------------------------------------------------------------------

def get_best_model(
    results: Mapping[
        str,
        Mapping[str, float],
    ],
    metric: str = "rmse",
) -> str:
    """
    Return the best model according to the selected metric.
    """

    return select_best_model(
        results,
        metric=metric,
    )


# ---------------------------------------------------------------------------
# Evaluation Summary
# ---------------------------------------------------------------------------

def build_evaluation_summary(
    results: Mapping[
        str,
        Mapping[str, float],
    ],
    selection_metric: str = "rmse",
) -> Dict[str, Any]:
    """
    Build a complete model evaluation summary.
    """

    if not results:
        raise ValueError(
            "No model results supplied."
        )

    best_model = get_best_model(
        results,
        metric=selection_metric,
    )

    ranking = rank_models(
        results,
        metric=selection_metric,
    )

    return {
        "selection_metric": selection_metric,
        "best_model": best_model,
        "model_count": len(results),
        "ranking": ranking,
        "models": {
            name: dict(metrics)
            for name, metrics in results.items()
        },
    }


# ---------------------------------------------------------------------------
# JSON Serialization
# ---------------------------------------------------------------------------

def save_results_json(
    results: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save evaluation results as JSON.
    """

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
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Evaluation results saved to %s",
        path,
    )

    return path


# ---------------------------------------------------------------------------
# Text Report
# ---------------------------------------------------------------------------

def create_text_report(
    results: Mapping[
        str,
        Mapping[str, float],
    ],
    selection_metric: str = "rmse",
) -> str:
    """
    Create a human-readable forecasting evaluation report.
    """

    summary = build_evaluation_summary(
        results,
        selection_metric=selection_metric,
    )

    lines = [
        "FinCo AI - Forecast Evaluation Report",
        "=" * 45,
        "",
        f"Selection Metric : {selection_metric}",
        f"Best Model      : {summary['best_model']}",
        f"Model Count     : {summary['model_count']}",
        "",
        "Model Ranking",
        "-" * 45,
    ]

    for item in summary["ranking"]:

        lines.append(
            f"{item['rank']}. "
            f"{item['model']} "
            f"({selection_metric}="
            f"{item['value']:.4f})"
        )

    lines.extend(
        [
            "",
            "Detailed Metrics",
            "-" * 45,
        ]
    )

    for model_name, metrics in results.items():

        lines.extend(
            [
                "",
                f"[{model_name}]",
                format_metrics(metrics),
            ]
        )

    return "\n".join(lines)


def save_text_report(
    results: Mapping[
        str,
        Mapping[str, float],
    ],
    output_path: str | Path,
    selection_metric: str = "rmse",
) -> Path:
    """
    Save a human-readable evaluation report.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = create_text_report(
        results,
        selection_metric=selection_metric,
    )

    path.write_text(
        report,
        encoding="utf-8",
    )

    logger.info(
        "Evaluation report saved to %s",
        path,
    )

    return path


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    actual: Iterable[float],
    predictions: Mapping[
        str,
        Iterable[float],
    ],
    selection_metric: str = "rmse",
    json_output: Optional[str | Path] = None,
    report_output: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Execute the complete forecasting evaluation pipeline.

    Steps:
        1. Validate predictions
        2. Calculate metrics
        3. Rank models
        4. Select best model
        5. Optionally save JSON
        6. Optionally save text report
    """

    logger.info(
        "Starting FinCo AI forecast evaluation."
    )

    results = evaluate_models(
        actual,
        predictions,
    )

    summary = build_evaluation_summary(
        results,
        selection_metric=selection_metric,
    )

    if json_output is not None:
        save_results_json(
            summary,
            json_output,
        )

    if report_output is not None:
        save_text_report(
            results,
            report_output,
            selection_metric=selection_metric,
        )

    logger.info(
        "Best forecasting model: %s",
        summary["best_model"],
    )

    return summary


# ---------------------------------------------------------------------------
# Console Output
# ---------------------------------------------------------------------------

def print_evaluation_summary(
    summary: Mapping[str, Any],
) -> None:
    """Print evaluation summary to the console."""

    print()
    print(
        "FinCo AI - Forecast Evaluation"
    )
    print("=" * 40)

    print(
        f"Selection metric: "
        f"{summary['selection_metric']}"
    )

    print(
        f"Best model: "
        f"{summary['best_model']}"
    )

    print(
        f"Models evaluated: "
        f"{summary['model_count']}"
    )

    print()
    print("Ranking")
    print("-" * 40)

    for item in summary["ranking"]:

        print(
            f"{item['rank']}. "
            f"{item['model']} "
            f"-> "
            f"{item['metric']}="
            f"{item['value']:.4f}"
        )


# ---------------------------------------------------------------------------
# Optional Joblib Model Loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str | Path,
) -> Any:
    """
    Load a serialized sklearn-compatible model.

    Joblib is imported lazily so that the evaluation module
    can still be imported without joblib when only metric
    evaluation is required.
    """

    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load serialized models. "
            "Install it with: pip install joblib"
        ) from exc

    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    logger.info(
        "Loading model from %s",
        path,
    )

    return joblib.load(path)


# ---------------------------------------------------------------------------
# Optional NumPy Data Loading
# ---------------------------------------------------------------------------

def load_test_data(
    x_path: str | Path,
    y_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load X_test and y_test from NumPy files.

    Expected:
        X_test.npy
        y_test.npy
    """

    x_file = Path(x_path)
    y_file = Path(y_path)

    if not x_file.exists():
        raise FileNotFoundError(
            f"X_test file not found: {x_file}"
        )

    if not y_file.exists():
        raise FileNotFoundError(
            f"y_test file not found: {y_file}"
        )

    X_test = np.load(x_file)
    y_test = np.load(y_file)

    return X_test, y_test


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FinCo AI forecasting models."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Path to serialized forecasting model.",
    )

    parser.add_argument(
        "--x-test",
        type=str,
        required=False,
        help="Path to X_test.npy.",
    )

    parser.add_argument(
        "--y-test",
        type=str,
        required=False,
        help="Path to y_test.npy.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="forecast_model",
        help="Name of the forecasting model.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="rmse",
        choices=[
            "mae",
            "mse",
            "rmse",
            "mape",
            "smape",
            "wape",
            "r2",
            "accuracy",
        ],
        help="Metric used to select the best model.",
    )

    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )

    parser.add_argument(
        "--report-output",
        type=str,
        default=None,
        help="Optional text report path.",
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

    if not args.model:
        parser.error(
            "--model is required when running the CLI."
        )

    if not args.x_test:
        parser.error(
            "--x-test is required when running the CLI."
        )

    if not args.y_test:
        parser.error(
            "--y-test is required when running the CLI."
        )

    model = load_model(
        args.model
    )

    X_test, y_test = load_test_data(
        args.x_test,
        args.y_test,
    )

    evaluation = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=args.model_name,
    )

    results = {
        args.model_name: evaluation["metrics"]
    }

    summary = build_evaluation_summary(
        results,
        selection_metric=args.metric,
    )

    if args.json_output:
        save_results_json(
            summary,
            args.json_output,
        )

    if args.report_output:
        save_text_report(
            results,
            args.report_output,
            selection_metric=args.metric,
        )

    print_evaluation_summary(
        summary
    )


if __name__ == "__main__":
    main()
```

