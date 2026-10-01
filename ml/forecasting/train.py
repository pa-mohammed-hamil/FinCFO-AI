```python
"""
FinCo AI - Forecasting Model Training

File:
    ml/forecasting/train.py

Purpose:
    Train financial forecasting models and save the best model.

Supported models:
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - HistGradientBoosting
    - XGBoost (optional)

Workflow:
    1. Load training data
    2. Validate data
    3. Train multiple models
    4. Generate validation predictions
    5. Calculate forecasting metrics
    6. Select best model
    7. Save model and training metadata

Expected target:
    Continuous numerical value such as:
        revenue
        sales
        expenses
        cash flow
        transaction amount
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from ml.evaluation.forecast_metrics import (
    calculate_forecast_metrics,
    compare_models,
    select_best_model,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42

DEFAULT_MODEL_DIR = Path("artifacts/models")
DEFAULT_METADATA_DIR = Path("artifacts/metrics")

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
# Data Validation
# ---------------------------------------------------------------------------

def validate_training_data(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate feature and target arrays.
    """

    if not isinstance(X, np.ndarray):
        raise TypeError(
            "X must be a numpy array."
        )

    if not isinstance(y, np.ndarray):
        raise TypeError(
            "y must be a numpy array."
        )

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2-dimensional feature matrix."
        )

    if y.ndim != 1:
        y = y.reshape(-1)

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same number of samples."
        )

    if len(X) < 10:
        raise ValueError(
            "At least 10 samples are required for training."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains NaN or infinite values."
        )

    if not np.all(np.isfinite(y)):
        raise ValueError(
            "y contains NaN or infinite values."
        )


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_numpy_data(
    x_path: str | Path,
    y_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load X and y from .npy files.

    Expected:
        X.npy
        y.npy
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

    X = np.load(x_file)
    y = np.load(y_file).reshape(-1)

    validate_training_data(
        X,
        y,
    )

    logger.info(
        "Loaded dataset: samples=%d, features=%d",
        X.shape[0],
        X.shape[1],
    )

    return X, y


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_models(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Build the forecasting model collection.

    XGBoost is included when installed.
    """

    models: Dict[str, Any] = {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=4,
            min_samples_leaf=2,
            loss="huber",
            random_state=random_state,
        ),

        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=random_state,
        ),
    }

    # XGBoost is optional.
    try:
        from xgboost import XGBRegressor

        models["xgboost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=random_state,
            n_jobs=-1,
        )

        logger.info(
            "XGBoost detected and enabled."
        )

    except ImportError:
        logger.warning(
            "XGBoost is not installed. "
            "Continuing without XGBoost."
        )

    return models


# ---------------------------------------------------------------------------
# Train Single Model
# ---------------------------------------------------------------------------

def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """
    Train a single forecasting model.
    """

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate predictions from a trained model.
    """

    predictions = np.asarray(
        model.predict(X),
        dtype=float,
    ).reshape(-1)

    if not np.all(np.isfinite(predictions)):
        raise ValueError(
            "Model generated NaN or infinite predictions."
        )

    return predictions


# ---------------------------------------------------------------------------
# Evaluate Model
# ---------------------------------------------------------------------------

def evaluate_trained_model(
    model: Any,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> Dict[str, float]:
    """
    Generate validation predictions and calculate
    comprehensive forecasting metrics.
    """

    predictions = predict(
        model,
        X_validation,
    )

    return calculate_forecast_metrics(
        y_validation,
        predictions,
    )


# ---------------------------------------------------------------------------
# Train and Compare
# ---------------------------------------------------------------------------

def train_and_compare(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[
    Dict[str, Any],
    Dict[str, Dict[str, float]],
]:
    """
    Train all models and compare validation performance.

    Returns:
        trained_models
        evaluation_results
    """

    models = build_models(
        random_state=random_state,
    )

    trained_models: Dict[str, Any] = {}

    predictions: Dict[
        str,
        np.ndarray,
    ] = {}

    for model_name, model in models.items():

        logger.info(
            "Training model: %s",
            model_name,
        )

        try:
            trained_model = train_model(
                model,
                X_train,
                y_train,
            )

            model_predictions = predict(
                trained_model,
                X_validation,
            )

            trained_models[model_name] = (
                trained_model
            )

            predictions[model_name] = (
                model_predictions
            )

            logger.info(
                "Finished training: %s",
                model_name,
            )

        except Exception:
            logger.exception(
                "Failed to train model: %s",
                model_name,
            )

    if not predictions:
        raise RuntimeError(
            "No forecasting models were successfully trained."
        )

    results = compare_models(
        y_validation,
        predictions,
    )

    return trained_models, results


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------

def select_model(
    trained_models: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    metric: str = "rmse",
) -> tuple[str, Any]:
    """
    Select the best trained model.
    """

    best_name = select_best_model(
        metrics,
        metric=metric,
    )

    if best_name not in trained_models:
        raise KeyError(
            f"Selected model '{best_name}' "
            "is not available."
        )

    return (
        best_name,
        trained_models[best_name],
    )


# ---------------------------------------------------------------------------
# Save Model
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    model_name: str,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
) -> Path:
    """
    Save trained model using joblib.
    """

    directory = Path(output_dir)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        directory
        / f"{model_name}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    logger.info(
        "Saved model to %s",
        model_path,
    )

    return model_path


# ---------------------------------------------------------------------------
# Save Metrics
# ---------------------------------------------------------------------------

def save_metrics(
    metrics: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save evaluation metrics to JSON.
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
            metrics,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Saved metrics to %s",
        path,
    )

    return path


# ---------------------------------------------------------------------------
# Training Metadata
# ---------------------------------------------------------------------------

def build_training_metadata(
    best_model_name: str,
    metrics: Dict[str, Dict[str, float]],
    X_train: np.ndarray,
    X_validation: np.ndarray,
    selection_metric: str,
    random_state: int,
) -> Dict[str, Any]:
    """
    Build metadata describing the training run.
    """

    return {
        "project": "FinCo AI",
        "task": "financial_forecasting",
        "best_model": best_model_name,
        "selection_metric": selection_metric,
        "random_state": random_state,
        "training_samples": int(
            X_train.shape[0]
        ),
        "validation_samples": int(
            X_validation.shape[0]
        ),
        "feature_count": int(
            X_train.shape[1]
        ),
        "models": metrics,
    }


# ---------------------------------------------------------------------------
# Complete Training Pipeline
# ---------------------------------------------------------------------------

def run_training(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    selection_metric: str = "rmse",
    model_output_dir: str | Path = DEFAULT_MODEL_DIR,
    metrics_output: str | Path = (
        DEFAULT_METADATA_DIR / "forecast_metrics.json"
    ),
) -> Dict[str, Any]:
    """
    Execute the complete forecasting training pipeline.

    Steps:
        1. Validate dataset
        2. Split train/validation data
        3. Train candidate models
        4. Evaluate models
        5. Select best model
        6. Save best model
        7. Save metrics and metadata
    """

    validate_training_data(
        X,
        y,
    )

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    logger.info(
        "Starting FinCo AI forecasting training."
    )

    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )
    )

    logger.info(
        "Train samples: %d | Validation samples: %d",
        len(X_train),
        len(X_validation),
    )

    trained_models, results = train_and_compare(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        random_state=random_state,
    )

    best_model_name, best_model = select_model(
        trained_models,
        results,
        metric=selection_metric,
    )

    model_path = save_model(
        best_model,
        best_model_name,
        model_output_dir,
    )

    metadata = build_training_metadata(
        best_model_name=best_model_name,
        metrics=results,
        X_train=X_train,
        X_validation=X_validation,
        selection_metric=selection_metric,
        random_state=random_state,
    )

    metadata["model_path"] = str(
        model_path
    )

    save_metrics(
        metadata,
        metrics_output,
    )

    logger.info(
        "Best forecasting model: %s",
        best_model_name,
    )

    return metadata


# ---------------------------------------------------------------------------
# Synthetic Demo Data
# ---------------------------------------------------------------------------

def create_demo_data(
    samples: int = 500,
    features: int = 8,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic financial forecasting data.

    This is intended only for local testing/demo purposes.
    """

    if samples < 20:
        raise ValueError(
            "samples must be at least 20."
        )

    if features < 2:
        raise ValueError(
            "features must be at least 2."
        )

    rng = np.random.default_rng(
        random_state
    )

    X = rng.normal(
        0,
        1,
        size=(samples, features),
    )

    coefficients = np.linspace(
        5.0,
        20.0,
        features,
    )

    trend = np.arange(samples) * 0.15

    noise = rng.normal(
        0,
        8,
        size=samples,
    )

    y = (
        500
        + X @ coefficients
        + trend
        + noise
    )

    return X.astype(float), y.astype(float)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Train FinCo AI forecasting models."
        )
    )

    parser.add_argument(
        "--x-data",
        type=str,
        help="Path to X.npy.",
    )

    parser.add_argument(
        "--y-data",
        type=str,
        help="Path to y.npy.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Validation split size.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed.",
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
        help="Metric used for model selection.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Directory for saved models.",
    )

    parser.add_argument(
        "--metrics-output",
        type=str,
        default=str(
            DEFAULT_METADATA_DIR
            / "forecast_metrics.json"
        ),
        help="Path for evaluation metadata.",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Train using synthetic demo data.",
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

    if args.demo:

        logger.info(
            "Using synthetic demo forecasting data."
        )

        X, y = create_demo_data()

    else:

        if not args.x_data or not args.y_data:
            parser.error(
                "Provide --x-data and --y-data "
                "or use --demo."
            )

        X, y = load_numpy_data(
            args.x_data,
            args.y_data,
        )

    metadata = run_training(
        X=X,
        y=y,
        test_size=args.test_size,
        random_state=args.random_state,
        selection_metric=args.metric,
        model_output_dir=args.model_dir,
        metrics_output=args.metrics_output,
    )

    print()
    print(
        "FinCo AI Forecast Training Complete"
    )
    print("=" * 42)
    print(
        f"Best model : {metadata['best_model']}"
    )
    print(
        f"Metric     : {metadata['selection_metric']}"
    )
    print(
        f"Samples    : {metadata['training_samples']}"
    )
    print(
        f"Features   : {metadata['feature_count']}"
    )
    print(
        f"Model path : {metadata['model_path']}"
    )


if __name__ == "__main__":
    main()
```
