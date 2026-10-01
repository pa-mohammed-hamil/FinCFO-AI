"""
FinCo AI - Forecast Evaluation Metrics

File:
    ml/evaluation/forecast_metrics.py

Used for:
    - Revenue forecasting
    - Sales forecasting
    - Expense forecasting
    - Cash-flow forecasting
    - Financial time-series model evaluation
    - Model comparison

Metrics:
    MAE, MSE, RMSE, MAPE, sMAPE, WAPE, R2,
    forecast bias, forecast accuracy
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np


EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate actual and predicted forecast values."""

    y_true = np.asarray(list(actual), dtype=float)
    y_pred = np.asarray(list(predicted), dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "actual and predicted must be one-dimensional."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "actual and predicted must have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "actual and predicted cannot be empty."
        )

    if not np.all(np.isfinite(y_true)):
        raise ValueError(
            "actual contains NaN or infinite values."
        )

    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            "predicted contains NaN or infinite values."
        )

    return y_true, y_pred


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

def mae(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    """
    Mean Absolute Error.

    Lower is better.

    MAE = mean(|actual - predicted|)
    """

    y_true, y_pred = _validate_inputs(actual, predicted)

    return float(
        np.mean(np.abs(y_true - y_pred))
    )


# ---------------------------------------------------------------------------
# MSE
# ---------------------------------------------------------------------------

def mse(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    """
    Mean Squared Error.

    Lower is better.
    """

    y_true, y_pred = _validate_inputs(actual, predicted)

    return float(
        np.mean((y_true - y_pred) ** 2)
    )


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------

def rmse(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    """
    Root Mean Squared Error.

    Lower is better.
    """

    return float(
        np.sqrt(
            mse(actual, predicted)
        )
    )


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------

def mape(
    actual: Iterable[float],
    predicted: Iterable[float],
    epsilon: float = EPSILON,
) -> float:
    """
    Mean Absolute Percentage Error.

    Returns percentage.

    Actual values close to zero are excluded because
    percentage error becomes unstable around zero.
    """

    y_true, y_pred = _validate_inputs(
        actual,
        predicted,
    )

    mask = np.abs(y_true) > epsilon

    if not np.any(mask):
        return 0.0

    errors = np.abs(
        (y_true[mask] - y_pred[mask])
        / y_true[mask]
    )

    return float(
        np.mean(errors) * 100.0
    )


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------

def smape(
    actual: Iterable[float],
    predicted: Iterable[float],
    epsilon: float = EPSILON,
) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Returns percentage.

    sMAPE =
        mean(
            2 * |actual - predicted|
            / (|actual| + |predicted|)
        ) * 100
    """

    y_true, y_pred = _validate_inputs(
        actual,
        predicted,
    )

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
    )

    mask = denominator > epsilon

    if not np.any(mask):
        return 0.0

    errors = (
        2.0
        * np.abs(y_true[mask] - y_pred[mask])
        / denominator[mask]
    )

    return float(
        np.mean(errors) * 100.0
    )


# ---------------------------------------------------------------------------
# WAPE
# ---------------------------------------------------------------------------

def wape(
    actual: Iterable[float],
    predicted: Iterable[float],
    epsilon: float = EPSILON,
) -> float:
    """
    Weighted Absolute Percentage Error.

    WAPE =
        sum(|actual - predicted|)
        / sum(|actual|)

    Returns percentage.
    """

    y_true, y_pred = _validate_inputs(
        actual,
        predicted,
    )

    denominator = np.sum(
        np.abs(y_true)
    )

    if denominator <= epsilon:
        return 0.0

    numerator = np.sum(
        np.abs(y_true - y_pred)
    )

    return float(
        numerator / denominator * 100.0
    )


# ---------------------------------------------------------------------------
# R2
# ---------------------------------------------------------------------------

def r2_score(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    """
    Coefficient of Determination.

    Higher is better.

    R2 = 1 - SS_res / SS_tot
    """

    y_true, y_pred = _validate_inputs(
        actual,
        predicted,
    )

    ss_res = np.sum(
        (y_true - y_pred) ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if ss_tot <= EPSILON:
        return (
            1.0
            if ss_res <= EPSILON
            else 0.0
        )

    return float(
        1.0 - ss_res / ss_tot
    )


# ---------------------------------------------------------------------------
# Forecast Bias
# ---------------------------------------------------------------------------

def forecast_bias(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    """
    Forecast Bias.

    Positive:
        Over-forecasting.

    Negative:
        Under-forecasting.

    Zero:
        No systematic bias.

    Formula:
        mean(predicted - actual)
    """

    y_true, y_pred = _validate_inputs(
        actual,
        predicted,
    )

    return float(
        np.mean(y_pred - y_true)
    )


def mean_error(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    """Alias for forecast_bias."""

    return forecast_bias(
        actual,
        predicted,
    )


# ---------------------------------------------------------------------------
# Forecast Accuracy
# ---------------------------------------------------------------------------

def forecast_accuracy(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> float:
    """
    Simple forecast accuracy.

    Accuracy = 100 - MAPE

    Returns percentage.
    """

    value = 100.0 - mape(
        actual,
        predicted,
    )

    return float(
        max(0.0, value)
    )


# ---------------------------------------------------------------------------
# Complete Metrics
# ---------------------------------------------------------------------------

def calculate_forecast_metrics(
    actual: Iterable[float],
    predicted: Iterable[float],
) -> Dict[str, float]:
    """
    Calculate all forecasting metrics.
    """

    y_true, y_pred = _validate_inputs(
        actual,
        predicted,
    )

    return {
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "bias": forecast_bias(y_true, y_pred),
        "accuracy": forecast_accuracy(
            y_true,
            y_pred,
        ),
    }


# ---------------------------------------------------------------------------
# Compare Multiple Models
# ---------------------------------------------------------------------------

def compare_models(
    actual: Iterable[float],
    predictions: Mapping[
        str,
        Iterable[float],
    ],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple forecasting models.

    Example:

        predictions = {
            "linear_regression": lr_predictions,
            "random_forest": rf_predictions,
            "xgboost": xgb_predictions,
        }

        results = compare_models(
            actual,
            predictions,
        )
    """

    y_true = list(actual)

    results: Dict[
        str,
        Dict[str, float],
    ] = {}

    for model_name, model_predictions in predictions.items():

        results[model_name] = (
            calculate_forecast_metrics(
                y_true,
                model_predictions,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Select Best Model
# ---------------------------------------------------------------------------

def select_best_model(
    metrics: Mapping[
        str,
        Mapping[str, float],
    ],
    metric: str = "rmse",
) -> str:
    """
    Select the best forecasting model.

    Lower is better:
        mae
        mse
        rmse
        mape
        smape
        wape

    Higher is better:
        r2
        accuracy
    """

    if not metrics:
        raise ValueError(
            "metrics cannot be empty."
        )

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

    supported = (
        lower_is_better
        | higher_is_better
    )

    if metric not in supported:
        raise ValueError(
            f"Unsupported metric: {metric}"
        )

    for model_name, model_metrics in metrics.items():

        if metric not in model_metrics:
            raise ValueError(
                f"Metric '{metric}' is missing "
                f"for model '{model_name}'."
            )

    if metric in lower_is_better:

        return min(
            metrics,
            key=lambda name: metrics[name][metric],
        )

    return max(
        metrics,
        key=lambda name: metrics[name][metric],
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_metrics(
    metrics: Mapping[str, float],
) -> str:
    """
    Create a human-readable metrics report.
    """

    lines = [
        "Forecast Evaluation",
        "=" * 24,
    ]

    percentage_metrics = {
        "mape",
        "smape",
        "wape",
        "accuracy",
    }

    display_order = [
        "mae",
        "mse",
        "rmse",
        "mape",
        "smape",
        "wape",
        "r2",
        "bias",
        "accuracy",
    ]

    for name in display_order:

        if name not in metrics:
            continue

        label = name.replace(
            "_",
            " ",
        ).title()

        value = metrics[name]

        if name in percentage_metrics:
            lines.append(
                f"{label:<12}: {value:.2f}%"
            )
        else:
            lines.append(
                f"{label:<12}: {value:.4f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example / Local Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    actual = [
        100,
        120,
        130,
        150,
        170,
        180,
    ]

    predicted = [
        105,
        115,
        128,
        155,
        165,
        185,
    ]

    metrics = calculate_forecast_metrics(
        actual,
        predicted,
    )

    print(
        format_metrics(metrics)
    )

    models = {
        "baseline": [
            110,
            118,
            135,
            145,
            175,
            190,
        ],
        "xgboost": predicted,
    }

    comparison = compare_models(
        actual,
        models,
    )

    best_model = select_best_model(
        comparison,
        metric="rmse",
    )

    print(
        "\nBest Model:",
        best_model,
    )