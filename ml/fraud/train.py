"""
FinCo AI - Fraud Detection Model Training

File:
    ml/fraud/train.py

Purpose:
    Train, compare, and save fraud detection models.

Models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - HistGradientBoosting
    - XGBoost (optional)

Evaluation:
    - Accuracy
    - Precision
    - Recall
    - F1
    - ROC-AUC
    - PR-AUC
    - Specificity
    - False Positive Rate
    - False Negative Rate
    - Fraud Capture Rate

Important:
    Fraud datasets are usually highly imbalanced.
    Therefore PR-AUC, recall, precision and F1 are
    considered alongside ROC-AUC and accuracy.
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
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from ml.evaluation.fraud_metrics import (
    calculate_fraud_metrics,
    compare_fraud_models,
    evaluate_thresholds,
    select_best_fraud_model,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50

DEFAULT_MODEL_DIR = Path("artifacts/models")
DEFAULT_METRICS_DIR = Path("artifacts/metrics")

DEFAULT_METRICS_FILE = (
    DEFAULT_METRICS_DIR / "fraud_metrics.json"
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

def validate_data(
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Validate fraud training data.
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
            "X must be a 2-dimensional array."
        )

    y = y.reshape(-1)

    if len(X) != len(y):
        raise ValueError(
            "X and y must contain the same "
            "number of samples."
        )

    if len(X) < 20:
        raise ValueError(
            "At least 20 samples are required."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "X contains NaN or infinite values."
        )

    unique_labels = np.unique(y)

    if not np.all(
        np.isin(unique_labels, [0, 1])
    ):
        raise ValueError(
            "Fraud target must contain only "
            "0 and 1."
        )

    if len(unique_labels) < 2:
        raise ValueError(
            "Training data must contain both "
            "fraud and non-fraud samples."
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

    validate_data(X, y)

    logger.info(
        "Loaded dataset: samples=%d, features=%d",
        X.shape[0],
        X.shape[1],
    )

    logger.info(
        "Fraud samples=%d | Non-fraud samples=%d",
        int(np.sum(y == 1)),
        int(np.sum(y == 0)),
    )

    return X, y


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_models(
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Build candidate fraud detection models.

    Imputation is included so the models can safely handle
    missing numeric values at inference time.
    """

    models: Dict[str, Any] = {}

    models["logistic_regression"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    models["random_forest"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=14,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models["gradient_boosting"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    max_depth=4,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                ),
            ),
        ]
    )

    models["hist_gradient_boosting"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=0.5,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Optional XGBoost.
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=400,
                        max_depth=6,
                        learning_rate=0.05,
                        min_child_weight=2,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective=(
                            "binary:logistic"
                        ),
                        eval_metric="aucpr",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
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
# Prediction
# ---------------------------------------------------------------------------

def predict_probabilities(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate fraud probabilities.
    """

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise AttributeError(
            "Model does not support predict_proba()."
        )

    probabilities = np.asarray(
        model.predict_proba(X),
        dtype=float,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "Expected a 2D probability matrix."
        )

    if probabilities.shape[1] != 2:
        raise ValueError(
            "Expected binary classification."
        )

    fraud_probabilities = probabilities[:, 1]

    fraud_probabilities = np.clip(
        fraud_probabilities,
        0.0,
        1.0,
    )

    if not np.all(
        np.isfinite(fraud_probabilities)
    ):
        raise ValueError(
            "Model generated invalid probabilities."
        )

    return fraud_probabilities


def probabilities_to_labels(
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """
    Convert fraud probabilities into binary predictions.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    return (
        np.asarray(probabilities)
        >= threshold
    ).astype(int)


# ---------------------------------------------------------------------------
# Train Single Model
# ---------------------------------------------------------------------------

def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """
    Train a single fraud detection model.
    """

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------------------------
# Evaluate Single Model
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    """
    Evaluate a trained fraud model.
    """

    probabilities = predict_probabilities(
        model,
        X_validation,
    )

    predictions = probabilities_to_labels(
        probabilities,
        threshold,
    )

    metrics = calculate_fraud_metrics(
        y_validation,
        predictions,
        probabilities,
    )

    return metrics


# ---------------------------------------------------------------------------
# Train and Compare Models
# ---------------------------------------------------------------------------

def train_and_compare(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    random_state: int = DEFAULT_RANDOM_STATE,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[
    Dict[str, Any],
    Dict[str, Dict[str, float]],
]:
    """
    Train all candidate models and compare them.
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
            "Training fraud model: %s",
            model_name,
        )

        try:

            trained_model = train_model(
                model,
                X_train,
                y_train,
            )

            probabilities = (
                predict_probabilities(
                    trained_model,
                    X_validation,
                )
            )

            predictions[
                model_name
            ] = probabilities

            trained_models[
                model_name
            ] = trained_model

            logger.info(
                "Completed: %s",
                model_name,
            )

        except Exception:
            logger.exception(
                "Failed to train %s",
                model_name,
            )

    if not predictions:
        raise RuntimeError(
            "No fraud models were successfully trained."
        )

    # Compare using the centralized fraud metrics.
    results = compare_fraud_models(
        y_validation,
        {
            name: probabilities_to_labels(
                probabilities,
                threshold,
            )
            for name, probabilities
            in predictions.items()
        },
        {
            name: probabilities
            for name, probabilities
            in predictions.items()
        },
    )

    return trained_models, results


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------

def select_model(
    trained_models: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    metric: str = "pr_auc",
) -> tuple[str, Any]:
    """
    Select the best fraud detection model.

    Default selection metric:
        PR-AUC

    PR-AUC is generally more informative than accuracy
    for highly imbalanced fraud datasets.
    """

    best_model_name = select_best_fraud_model(
        metrics,
        metric=metric,
    )

    if best_model_name not in trained_models:
        raise KeyError(
            f"Selected model '{best_model_name}' "
            "was not trained successfully."
        )

    return (
        best_model_name,
        trained_models[best_model_name],
    )


# ---------------------------------------------------------------------------
# Threshold Search
# ---------------------------------------------------------------------------

def find_best_threshold(
    model: Any,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    metric: str = "f1",
) -> tuple[
    float,
    Dict[str, Dict[str, float]],
]:
    """
    Find an appropriate fraud classification threshold.
    """

    probabilities = predict_probabilities(
        model,
        X_validation,
    )

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    threshold_results = evaluate_thresholds(
        y_validation,
        probabilities,
        thresholds,
    )

    if not threshold_results:
        raise RuntimeError(
            "Threshold evaluation returned no results."
        )

    if metric in {
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "specificity",
        "fraud_capture_rate",
        "accuracy",
    }:

        best_threshold_key = max(
            threshold_results,
            key=lambda key: (
                threshold_results[key]
                .get(metric, -np.inf)
            ),
        )

    else:

        raise ValueError(
            f"Unsupported threshold metric: {metric}"
        )

    return (
        float(best_threshold_key),
        threshold_results,
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
    Save trained model with joblib.
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
        "Saved model: %s",
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
    Save training/evaluation metadata.
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
        "Saved metrics: %s",
        path,
    )

    return path


# ---------------------------------------------------------------------------
# Training Metadata
# ---------------------------------------------------------------------------

def build_metadata(
    best_model_name: str,
    best_model_path: Path,
    model_metrics: Dict[
        str,
        Dict[str, float],
    ],
    best_threshold: float,
    threshold_metric: str,
    X_train: np.ndarray,
    X_validation: np.ndarray,
    random_state: int,
) -> Dict[str, Any]:
    """
    Build complete training metadata.
    """

    return {
        "project": "FinCo AI",
        "task": "fraud_detection",
        "best_model": best_model_name,
        "best_model_path": str(
            best_model_path
        ),
        "selection_metric": "pr_auc",
        "threshold_metric": threshold_metric,
        "recommended_threshold": (
            float(best_threshold)
        ),
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
        "models": model_metrics,
    }


# ---------------------------------------------------------------------------
# Complete Training Pipeline
# ---------------------------------------------------------------------------

def run_training(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    model_selection_metric: str = "pr_auc",
    threshold_metric: str = "f1",
    model_output_dir: str | Path = DEFAULT_MODEL_DIR,
    metrics_output: str | Path = DEFAULT_METRICS_FILE,
) -> Dict[str, Any]:
    """
    Run the complete fraud training pipeline.
    """

    validate_data(
        X,
        y,
    )

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    logger.info(
        "Starting FinCo AI fraud training."
    )

    # Stratification is critical for fraud datasets.
    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    )

    logger.info(
        "Train samples=%d | Validation samples=%d",
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

    # Select the best model.
    best_model_name, best_model = select_model(
        trained_models,
        results,
        metric=model_selection_metric,
    )

    logger.info(
        "Best model based on %s: %s",
        model_selection_metric,
        best_model_name,
    )

    # Optimize classification threshold.
    best_threshold, threshold_results = (
        find_best_threshold(
            best_model,
            X_validation,
            y_validation,
            metric=threshold_metric,
        )
    )

    logger.info(
        "Best threshold based on %s: %.4f",
        threshold_metric,
        best_threshold,
    )

    # Re-evaluate the selected model at the
    # optimized threshold.
    probabilities = predict_probabilities(
        best_model,
        X_validation,
    )

    best_predictions = probabilities_to_labels(
        probabilities,
        best_threshold,
    )

    best_metrics = calculate_fraud_metrics(
        y_validation,
        best_predictions,
        probabilities,
    )

    results[
        best_model_name
    ] = best_metrics

    # Save model.
    model_path = save_model(
        best_model,
        best_model_name,
        model_output_dir,
    )

    metadata = build_metadata(
        best_model_name=best_model_name,
        best_model_path=model_path,
        model_metrics=results,
        best_threshold=best_threshold,
        threshold_metric=threshold_metric,
        X_train=X_train,
        X_validation=X_validation,
        random_state=random_state,
    )

    metadata[
        "threshold_results"
    ] = threshold_results

    metadata[
        "validation_fraud_count"
    ] = int(
        np.sum(y_validation == 1)
    )

    metadata[
        "validation_non_fraud_count"
    ] = int(
        np.sum(y_validation == 0)
    )

    save_metrics(
        metadata,
        metrics_output,
    )

    logger.info(
        "Fraud training completed successfully."
    )

    return metadata


# ---------------------------------------------------------------------------
# Demo Data
# ---------------------------------------------------------------------------

def create_demo_data(
    samples: int = 2000,
    features: int = 15,
    fraud_rate: float = 0.08,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic fraud data for local testing.

    This is NOT intended for real model training.
    """

    if samples < 100:
        raise ValueError(
            "samples must be at least 100."
        )

    if features < 3:
        raise ValueError(
            "features must be at least 3."
        )

    if not 0.01 <= fraud_rate <= 0.50:
        raise ValueError(
            "fraud_rate must be between "
            "0.01 and 0.50."
        )

    rng = np.random.default_rng(
        random_state
    )

    X = rng.normal(
        0,
        1,
        size=(samples, features),
    )

    # Simulated risk signals.
    risk_score = (
        1.5 * X[:, 0]
        + 1.2 * X[:, 1]
        + 1.0 * X[:, 2]
        + 0.8 * X[:, 3]
        + 0.5 * X[:, 4]
        + rng.normal(
            0,
            1,
            samples,
        )
    )

    # Choose threshold to approximately achieve
    # requested fraud rate.
    threshold = np.quantile(
        risk_score,
        1.0 - fraud_rate,
    )

    y = (
        risk_score >= threshold
    ).astype(int)

    return (
        X.astype(float),
        y.astype(int),
    )


# ---------------------------------------------------------------------------
# Console Summary
# ---------------------------------------------------------------------------

def print_summary(
    metadata: Dict[str, Any],
) -> None:
    """Print training summary."""

    print()
    print(
        "FinCo AI Fraud Training Complete"
    )
    print("=" * 48)

    print(
        f"Best model        : "
        f"{metadata['best_model']}"
    )

    print(
        f"Selection metric  : "
        f"{metadata['selection_metric']}"
    )

    print(
        f"Recommended thresh: "
        f"{metadata['recommended_threshold']:.4f}"
    )

    print(
        f"Training samples  : "
        f"{metadata['training_samples']}"
    )

    print(
        f"Validation samples: "
        f"{metadata['validation_samples']}"
    )

    print(
        f"Feature count     : "
        f"{metadata['feature_count']}"
    )

    print(
        f"Model path        : "
        f"{metadata['best_model_path']}"
    )

    print()

    best_metrics = metadata[
        "models"
    ][
        metadata["best_model"]
    ]

    print("Best Model Metrics")
    print("-" * 48)

    for metric, value in (
        best_metrics.items()
    ):

        if isinstance(
            value,
            (int, float),
        ):

            print(
                f"{metric:<25} "
                f"{value:.4f}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Train FinCo AI fraud detection models."
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
        "--demo",
        action="store_true",
        help="Train using synthetic demo data.",
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
        "--selection-metric",
        type=str,
        default="pr_auc",
        choices=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "specificity",
            "false_positive_rate",
            "false_negative_rate",
            "fraud_capture_rate",
        ],
        help="Metric used for model selection.",
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
            "roc_auc",
            "pr_auc",
            "specificity",
            "fraud_capture_rate",
        ],
        help=(
            "Metric used for classification "
            "threshold optimization."
        ),
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(
            DEFAULT_MODEL_DIR
        ),
        help="Directory for trained models.",
    )

    parser.add_argument(
        "--metrics-output",
        type=str,
        default=str(
            DEFAULT_METRICS_FILE
        ),
        help="JSON metrics output path.",
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
            "Using synthetic fraud data."
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
        model_selection_metric=(
            args.selection_metric
        ),
        threshold_metric=(
            args.threshold_metric
        ),
        model_output_dir=args.model_dir,
        metrics_output=args.metrics_output,
    )

    print_summary(
        metadata
    )


if __name__ == "__main__":
    main()