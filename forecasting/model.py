"""
FinCo AI - Advanced Forecasting Model
=====================================

Path:
    backend/app/forecasting/model.py

Purpose:
    Provides a production-oriented forecasting model abstraction for
    financial time-series forecasting.

Supported use cases:
    - Revenue forecasting
    - Profit forecasting
    - Cash-flow forecasting
    - Liquidity forecasting
    - Generic financial KPI forecasting

Design:
    Historical Data
          |
          v
    preprocessing.py
          |
          v
       features.py
          |
          v
       model.py
          |
          +--> Linear Regression
          +--> Random Forest
          +--> Gradient Boosting
          +--> User supplied estimator
          |
          v
    evaluation.py
          |
          v
    Best Forecast Model

Notes:
    - No database dependency.
    - No FastAPI dependency.
    - No LLM dependency.
    - Models are lazy-loaded.
    - Supports both direct forecasting and supervised ML forecasting.
    - Baseline forecasting remains separate in baseline.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# Exceptions
# ============================================================================


class ForecastModelError(Exception):
    """Base exception for forecasting model errors."""


class InvalidModelInputError(ForecastModelError):
    """Raised when model input is invalid."""


class ModelNotFittedError(ForecastModelError):
    """Raised when prediction is attempted before fitting."""


class ModelTrainingError(ForecastModelError):
    """Raised when model training fails."""


class ModelPredictionError(ForecastModelError):
    """Raised when model prediction fails."""


class UnsupportedModelError(ForecastModelError):
    """Raised when an unsupported model type is requested."""


class InsufficientTrainingDataError(ForecastModelError):
    """Raised when there is not enough data to train."""


# ============================================================================
# Enums
# ============================================================================


class ForecastModelType(str, Enum):
    """Supported forecasting model families."""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    EXTRA_TREES = "extra_trees"
    HISTORICAL_AVERAGE = "historical_average"
    CUSTOM = "custom"


class ForecastDirection(str, Enum):
    """General direction of a forecast."""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ForecastModelConfig:
    """
    Configuration for the forecasting model.

    Values are intentionally conservative for a portfolio-grade financial
    forecasting system.
    """

    model_type: ForecastModelType = ForecastModelType.GRADIENT_BOOSTING

    random_state: int = 42

    n_estimators: int = 200

    max_depth: Optional[int] = 5

    learning_rate: float = 0.05

    min_samples_leaf: int = 2

    minimum_training_samples: int = 12

    minimum_features: int = 1

    clip_negative_predictions: bool = False

    prediction_lower_bound: Optional[float] = None

    prediction_upper_bound: Optional[float] = None

    n_jobs: int = -1

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""

        if self.random_state < 0:
            raise InvalidModelInputError(
                "random_state must be >= 0."
            )

        if self.n_estimators <= 0:
            raise InvalidModelInputError(
                "n_estimators must be > 0."
            )

        if self.max_depth is not None and self.max_depth <= 0:
            raise InvalidModelInputError(
                "max_depth must be > 0 when provided."
            )

        if not 0.0 < self.learning_rate <= 1.0:
            raise InvalidModelInputError(
                "learning_rate must be in the range (0, 1]."
            )

        if self.min_samples_leaf <= 0:
            raise InvalidModelInputError(
                "min_samples_leaf must be > 0."
            )

        if self.minimum_training_samples < 2:
            raise InvalidModelInputError(
                "minimum_training_samples must be >= 2."
            )

        if self.minimum_features <= 0:
            raise InvalidModelInputError(
                "minimum_features must be > 0."
            )

        if (
            self.prediction_lower_bound is not None
            and self.prediction_upper_bound is not None
            and self.prediction_lower_bound > self.prediction_upper_bound
        ):
            raise InvalidModelInputError(
                "prediction_lower_bound cannot exceed "
                "prediction_upper_bound."
            )


# ============================================================================
# Result Objects
# ============================================================================


@dataclass
class TrainingResult:
    """Information returned after model training."""

    fitted: bool

    model_type: str

    sample_count: int

    feature_count: int

    feature_names: List[str]

    target_name: str

    target_mean: float

    target_min: float

    target_max: float

    training_mae: Optional[float]

    training_rmse: Optional[float]

    training_r2: Optional[float]

    trained_at: datetime

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert training result to a serializable dictionary."""

        return {
            "fitted": self.fitted,
            "model_type": self.model_type,
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "target_mean": self.target_mean,
            "target_min": self.target_min,
            "target_max": self.target_max,
            "training_mae": self.training_mae,
            "training_rmse": self.training_rmse,
            "training_r2": self.training_r2,
            "trained_at": self.trained_at.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass
class ForecastPrediction:
    """Represents one or more model predictions."""

    predictions: List[float]

    model_type: str

    horizon: int

    direction: ForecastDirection

    confidence: float

    lower_bound: Optional[float]

    upper_bound: Optional[float]

    feature_names: List[str]

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def forecast(self) -> List[float]:
        """Alias for predictions."""

        return self.predictions

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary."""

        return {
            "predictions": list(self.predictions),
            "forecast": list(self.predictions),
            "model_type": self.model_type,
            "horizon": self.horizon,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "feature_names": list(self.feature_names),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Main Forecasting Model
# ============================================================================


class FinancialForecastModel:
    """
    Advanced supervised forecasting model.

    The class accepts a user-supplied estimator or creates a default
    scikit-learn estimator based on ForecastModelConfig.

    Typical workflow:

        model = FinancialForecastModel()

        model.fit(
            X=features,
            y=revenue,
            feature_names=[...],
            target_name="revenue",
        )

        result = model.predict(future_features)
    """

    def __init__(
        self,
        config: Optional[ForecastModelConfig] = None,
        estimator: Optional[Any] = None,
    ) -> None:

        self.config = config or ForecastModelConfig()
        self.config.validate()

        self._estimator = estimator
        self._feature_names: List[str] = []
        self._target_name: str = "target"
        self._training_result: Optional[TrainingResult] = None
        self._is_fitted = False

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""

        return self._is_fitted

    @property
    def feature_names(self) -> List[str]:
        """Return feature names used during training."""

        return list(self._feature_names)

    @property
    def target_name(self) -> str:
        """Return the target variable name."""

        return self._target_name

    @property
    def model_type(self) -> ForecastModelType:
        """Return configured model type."""

        return self.config.model_type

    @property
    def estimator(self) -> Any:
        """Return underlying estimator."""

        return self._estimator

    @property
    def training_result(self) -> Optional[TrainingResult]:
        """Return training metadata."""

        return self._training_result

    # ========================================================================
    # Model Creation
    # ========================================================================

    def _create_estimator(self) -> Any:
        """Create the configured scikit-learn estimator."""

        model_type = self.config.model_type

        if model_type == ForecastModelType.HISTORICAL_AVERAGE:
            return None

        try:
            if model_type == ForecastModelType.LINEAR_REGRESSION:
                from sklearn.linear_model import LinearRegression

                return LinearRegression()

            if model_type == ForecastModelType.RANDOM_FOREST:
                from sklearn.ensemble import RandomForestRegressor

                return RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                )

            if model_type == ForecastModelType.GRADIENT_BOOSTING:
                from sklearn.ensemble import GradientBoostingRegressor

                return GradientBoostingRegressor(
                    n_estimators=self.config.n_estimators,
                    learning_rate=self.config.learning_rate,
                    max_depth=self.config.max_depth or 3,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state,
                )

            if model_type == ForecastModelType.EXTRA_TREES:
                from sklearn.ensemble import ExtraTreesRegressor

                return ExtraTreesRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                )

        except ImportError as exc:
            raise UnsupportedModelError(
                "scikit-learn is required for the selected forecasting model."
            ) from exc

        if model_type == ForecastModelType.CUSTOM:
            raise UnsupportedModelError(
                "CUSTOM model requires a user-supplied estimator."
            )

        raise UnsupportedModelError(
            f"Unsupported forecasting model: {model_type}"
        )

    # ========================================================================
    # Training
    # ========================================================================

    def fit(
        self,
        X: Any,
        y: Sequence[float],
        feature_names: Optional[Sequence[str]] = None,
        target_name: str = "target",
    ) -> TrainingResult:
        """
        Train the forecasting model.

        Parameters
        ----------
        X:
            Feature matrix.

        y:
            Target values.

        feature_names:
            Optional feature names.

        target_name:
            Name of the forecast target.
        """

        try:
            X_array = self._prepare_features(X)
            y_array = self._prepare_target(y)

            if len(X_array) != len(y_array):
                raise InvalidModelInputError(
                    "X and y must contain the same number of samples."
                )

            if len(X_array) < self.config.minimum_training_samples:
                raise InsufficientTrainingDataError(
                    f"At least "
                    f"{self.config.minimum_training_samples} "
                    f"training samples are required; "
                    f"received {len(X_array)}."
                )

            if X_array.shape[1] < self.config.minimum_features:
                raise InvalidModelInputError(
                    f"At least {self.config.minimum_features} "
                    "feature(s) are required."
                )

            self._feature_names = self._resolve_feature_names(
                X_array.shape[1],
                feature_names,
            )

            self._target_name = str(target_name or "target")

            if self.config.model_type == ForecastModelType.HISTORICAL_AVERAGE:
                self._estimator = None

            else:
                if self._estimator is None:
                    self._estimator = self._create_estimator()

                if not hasattr(self._estimator, "fit"):
                    raise InvalidModelInputError(
                        "Estimator must provide a fit() method."
                    )

                self._estimator.fit(X_array, y_array)

            predictions = self._training_predictions(X_array, y_array)

            mae = self._mae(y_array, predictions)
            rmse = self._rmse(y_array, predictions)
            r2 = self._r2(y_array, predictions)

            self._is_fitted = True

            self._training_result = TrainingResult(
                fitted=True,
                model_type=self.config.model_type.value,
                sample_count=len(y_array),
                feature_count=X_array.shape[1],
                feature_names=list(self._feature_names),
                target_name=self._target_name,
                target_mean=float(np.mean(y_array)),
                target_min=float(np.min(y_array)),
                target_max=float(np.max(y_array)),
                training_mae=mae,
                training_rmse=rmse,
                training_r2=r2,
                trained_at=self._utc_now(),
                metadata={
                    **self.config.metadata,
                    "estimator_class": (
                        type(self._estimator).__name__
                        if self._estimator is not None
                        else None
                    ),
                },
            )

            return self._training_result

        except ForecastModelError:
            raise

        except Exception as exc:
            raise ModelTrainingError(
                f"Forecast model training failed: {exc}"
            ) from exc

    # ========================================================================
    # Prediction
    # ========================================================================

    def predict(
        self,
        X: Any,
        horizon: Optional[int] = None,
    ) -> ForecastPrediction:
        """
        Generate forecasts.

        X can contain:
            - one future observation
            - multiple future observations

        The number of rows determines the forecast horizon unless
        horizon is explicitly supplied.
        """

        if not self._is_fitted:
            raise ModelNotFittedError(
                "Forecast model must be fitted before prediction."
            )

        try:
            X_array = self._prepare_features(X)

            if X_array.shape[1] != len(self._feature_names):
                raise InvalidModelInputError(
                    "Feature count does not match training features. "
                    f"Expected {len(self._feature_names)}, "
                    f"received {X_array.shape[1]}."
                )

            if horizon is not None:
                if horizon <= 0:
                    raise InvalidModelInputError(
                        "horizon must be > 0."
                    )

                if len(X_array) != horizon:
                    raise InvalidModelInputError(
                        f"horizon={horizon}, but X contains "
                        f"{len(X_array)} rows."
                    )

            actual_horizon = len(X_array)

            if self.config.model_type == ForecastModelType.HISTORICAL_AVERAGE:
                if self._training_result is None:
                    raise ModelNotFittedError(
                        "Training metadata is unavailable."
                    )

                predictions = np.full(
                    actual_horizon,
                    self._training_result.target_mean,
                    dtype=float,
                )

            else:
                if self._estimator is None:
                    raise ModelNotFittedError(
                        "Underlying estimator is unavailable."
                    )

                if not hasattr(self._estimator, "predict"):
                    raise ModelPredictionError(
                        "Estimator does not provide predict()."
                    )

                predictions = np.asarray(
                    self._estimator.predict(X_array),
                    dtype=float,
                ).reshape(-1)

            predictions = self._postprocess_predictions(predictions)

            direction = self._forecast_direction(predictions)

            confidence = self._estimate_confidence(predictions)

            lower, upper = self._prediction_bounds(predictions)

            return ForecastPrediction(
                predictions=predictions.tolist(),
                model_type=self.config.model_type.value,
                horizon=actual_horizon,
                direction=direction,
                confidence=confidence,
                lower_bound=lower,
                upper_bound=upper,
                feature_names=list(self._feature_names),
                metadata={
                    "target_name": self._target_name,
                    "generated_at": self._utc_now().isoformat(),
                },
            )

        except ForecastModelError:
            raise

        except Exception as exc:
            raise ModelPredictionError(
                f"Forecast prediction failed: {exc}"
            ) from exc

    def predict_one(self, features: Any) -> float:
        """Predict a single future value."""

        result = self.predict(features, horizon=1)
        return float(result.predictions[0])

    def predict_batch(self, X: Any) -> List[float]:
        """Return predictions as a simple list."""

        return self.predict(X).predictions

    # ========================================================================
    # Recursive Forecasting
    # ========================================================================

    def recursive_forecast(
        self,
        last_features: Sequence[float],
        horizon: int,
        update_features: Optional[Any] = None,
    ) -> ForecastPrediction:
        """
        Generate recursive forecasts.

        This is useful when future features depend partly on the previous
        predicted target.

        `update_features` can be a callable:

            update_features(current_features, prediction, step)
                -> next_features
        """

        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model must be fitted before recursive forecasting."
            )

        if horizon <= 0:
            raise InvalidModelInputError(
                "horizon must be > 0."
            )

        current_features = np.asarray(
            list(last_features),
            dtype=float,
        ).reshape(1, -1)

        if current_features.shape[1] != len(self._feature_names):
            raise InvalidModelInputError(
                "last_features does not match the trained feature count."
            )

        forecasts: List[float] = []

        for step in range(horizon):
            prediction = self.predict_one(current_features)

            forecasts.append(prediction)

            if update_features is not None:
                if not callable(update_features):
                    raise InvalidModelInputError(
                        "update_features must be callable."
                    )

                current_features = np.asarray(
                    update_features(
                        current_features.flatten().tolist(),
                        prediction,
                        step,
                    ),
                    dtype=float,
                ).reshape(1, -1)

                if current_features.shape[1] != len(self._feature_names):
                    raise InvalidModelInputError(
                        "update_features returned an invalid "
                        "feature vector size."
                    )

        array = np.asarray(forecasts, dtype=float)

        return ForecastPrediction(
            predictions=forecasts,
            model_type=self.config.model_type.value,
            horizon=horizon,
            direction=self._forecast_direction(array),
            confidence=self._estimate_confidence(array),
            lower_bound=(
                float(np.min(array))
                if len(array)
                else None
            ),
            upper_bound=(
                float(np.max(array))
                if len(array)
                else None
            ),
            feature_names=list(self._feature_names),
            metadata={
                "recursive": True,
                "target_name": self._target_name,
                "generated_at": self._utc_now().isoformat(),
            },
        )

    # ========================================================================
    # Historical Average
    # ========================================================================

    def historical_average_forecast(
        self,
        historical_values: Sequence[float],
        horizon: int,
    ) -> ForecastPrediction:
        """
        Produce a simple historical-average forecast.

        This is useful as a fallback when an ML model is unavailable.
        """

        if horizon <= 0:
            raise InvalidModelInputError(
                "horizon must be > 0."
            )

        values = self._prepare_target(historical_values)

        if len(values) == 0:
            raise InvalidModelInputError(
                "historical_values cannot be empty."
            )

        average = float(np.mean(values))

        predictions = [average] * horizon

        return ForecastPrediction(
            predictions=predictions,
            model_type=ForecastModelType.HISTORICAL_AVERAGE.value,
            horizon=horizon,
            direction=ForecastDirection.STABLE,
            confidence=self._estimate_historical_confidence(values),
            lower_bound=average,
            upper_bound=average,
            feature_names=[],
            metadata={
                "historical_observations": len(values),
                "historical_mean": average,
                "generated_at": self._utc_now().isoformat(),
            },
        )

    # ========================================================================
    # Feature Importance
    # ========================================================================

    def feature_importance(self) -> Dict[str, float]:
        """
        Return feature importance when supported by the estimator.
        """

        if not self._is_fitted:
            raise ModelNotFittedError(
                "Model must be fitted before feature importance "
                "can be calculated."
            )

        if self._estimator is None:
            return {}

        if hasattr(self._estimator, "feature_importances_"):
            values = np.asarray(
                self._estimator.feature_importances_,
                dtype=float,
            )

        elif hasattr(self._estimator, "coef_"):
            coefficients = np.asarray(
                self._estimator.coef_,
                dtype=float,
            )

            if coefficients.ndim > 1:
                coefficients = coefficients[0]

            values = np.abs(coefficients)

        else:
            return {}

        if len(values) != len(self._feature_names):
            return {}

        total = float(np.sum(values))

        if total > 0:
            values = values / total

        return {
            name: float(value)
            for name, value in zip(
                self._feature_names,
                values,
            )
        }

    # ========================================================================
    # Model Management
    # ========================================================================

    def set_estimator(self, estimator: Any) -> None:
        """Replace the underlying estimator."""

        if estimator is None:
            raise InvalidModelInputError(
                "estimator cannot be None."
            )

        if not hasattr(estimator, "fit"):
            raise InvalidModelInputError(
                "estimator must provide fit()."
            )

        self._estimator = estimator
        self._is_fitted = False
        self._training_result = None

    def get_estimator(self) -> Any:
        """Return the underlying estimator."""

        return self._estimator

    # ========================================================================
    # Validation / Preparation
    # ========================================================================

    def _prepare_features(self, X: Any) -> np.ndarray:
        """Convert feature input to a clean numeric matrix."""

        if X is None:
            raise InvalidModelInputError(
                "Feature matrix cannot be None."
            )

        if isinstance(X, Mapping):
            X = [X]

        try:
            if isinstance(X, Sequence) and not isinstance(
                X,
                (str, bytes),
            ):
                if len(X) == 0:
                    raise InvalidModelInputError(
                        "Feature matrix cannot be empty."
                    )

            array = np.asarray(X, dtype=float)

        except Exception as exc:
            raise InvalidModelInputError(
                "Features must contain numeric values."
            ) from exc

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise InvalidModelInputError(
                "Features must be a 2-dimensional matrix."
            )

        if array.shape[0] == 0 or array.shape[1] == 0:
            raise InvalidModelInputError(
                "Feature matrix cannot have zero rows or columns."
            )

        if not np.all(np.isfinite(array)):
            raise InvalidModelInputError(
                "Features contain NaN or infinite values."
            )

        return array

    def _prepare_target(self, y: Sequence[float]) -> np.ndarray:
        """Convert target values into a validated numeric array."""

        if y is None:
            raise InvalidModelInputError(
                "Target values cannot be None."
            )

        try:
            array = np.asarray(y, dtype=float).reshape(-1)
        except Exception as exc:
            raise InvalidModelInputError(
                "Target values must be numeric."
            ) from exc

        if len(array) == 0:
            raise InvalidModelInputError(
                "Target values cannot be empty."
            )

        if not np.all(np.isfinite(array)):
            raise InvalidModelInputError(
                "Target values contain NaN or infinite values."
            )

        return array

    def _resolve_feature_names(
        self,
        feature_count: int,
        feature_names: Optional[Sequence[str]],
    ) -> List[str]:

        if feature_names is None:
            return [
                f"feature_{index}"
                for index in range(feature_count)
            ]

        names = [str(name) for name in feature_names]

        if len(names) != feature_count:
            raise InvalidModelInputError(
                "feature_names length does not match "
                "feature count."
            )

        if len(set(names)) != len(names):
            raise InvalidModelInputError(
                "feature_names must be unique."
            )

        return names

    # ========================================================================
    # Training Metrics
    # ========================================================================

    def _training_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:

        if self.config.model_type == ForecastModelType.HISTORICAL_AVERAGE:
            return np.full_like(
                y,
                np.mean(y),
                dtype=float,
            )

        if self._estimator is None:
            raise ModelTrainingError(
                "Estimator is unavailable."
            )

        predictions = np.asarray(
            self._estimator.predict(X),
            dtype=float,
        ).reshape(-1)

        return self._postprocess_predictions(predictions)

    @staticmethod
    def _mae(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """Mean absolute error."""

        return float(
            np.mean(np.abs(actual - predicted))
        )

    @staticmethod
    def _rmse(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """Root mean squared error."""

        return float(
            np.sqrt(
                np.mean(
                    np.square(actual - predicted)
                )
            )
        )

    @staticmethod
    def _r2(
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> Optional[float]:
        """R-squared."""

        denominator = float(
            np.sum(
                np.square(
                    actual - np.mean(actual)
                )
            )
        )

        if denominator == 0:
            return None

        numerator = float(
            np.sum(
                np.square(actual - predicted)
            )
        )

        return float(1.0 - numerator / denominator)

    # ========================================================================
    # Prediction Processing
    # ========================================================================

    def _postprocess_predictions(
        self,
        predictions: np.ndarray,
    ) -> np.ndarray:

        values = np.asarray(
            predictions,
            dtype=float,
        ).reshape(-1)

        if not np.all(np.isfinite(values)):
            raise ModelPredictionError(
                "Model generated NaN or infinite predictions."
            )

        if self.config.clip_negative_predictions:
            values = np.maximum(values, 0.0)

        if self.config.prediction_lower_bound is not None:
            values = np.maximum(
                values,
                self.config.prediction_lower_bound,
            )

        if self.config.prediction_upper_bound is not None:
            values = np.minimum(
                values,
                self.config.prediction_upper_bound,
            )

        return values

    # ========================================================================
    # Forecast Direction
    # ========================================================================

    @staticmethod
    def _forecast_direction(
        predictions: np.ndarray,
    ) -> ForecastDirection:

        if len(predictions) < 2:
            return ForecastDirection.STABLE

        first = float(predictions[0])
        last = float(predictions[-1])

        scale = max(abs(first), 1.0)

        change = (last - first) / scale

        if change > 0.05:
            return ForecastDirection.UP

        if change < -0.05:
            return ForecastDirection.DOWN

        return ForecastDirection.STABLE

    # ========================================================================
    # Confidence
    # ========================================================================

    def _estimate_confidence(
        self,
        predictions: np.ndarray,
    ) -> float:
        """
        Estimate confidence from forecast stability.

        This is not a statistical prediction interval. It is an operational
        confidence score for downstream FinCo AI workflows.
        """

        values = np.asarray(
            predictions,
            dtype=float,
        )

        if len(values) <= 1:
            return 0.75

        mean = float(np.mean(np.abs(values)))

        if mean <= 1e-12:
            return 0.70

        std = float(np.std(values))

        coefficient = std / mean

        confidence = 1.0 - min(
            coefficient,
            0.80,
        )

        confidence = 0.20 + (
            0.80 * confidence
        )

        return float(
            np.clip(confidence, 0.20, 0.98)
        )

    @staticmethod
    def _estimate_historical_confidence(
        values: np.ndarray,
    ) -> float:

        if len(values) < 2:
            return 0.50

        mean = float(np.mean(np.abs(values)))

        if mean <= 1e-12:
            return 0.60

        std = float(np.std(values))

        cv = std / mean

        confidence = 1.0 - min(cv, 1.0)

        return float(
            np.clip(
                0.30 + 0.60 * confidence,
                0.30,
                0.90,
            )
        )

    # ========================================================================
    # Prediction Bounds
    # ========================================================================

    @staticmethod
    def _prediction_bounds(
        predictions: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:

        if len(predictions) == 0:
            return None, None

        mean = float(np.mean(predictions))
        std = float(np.std(predictions))

        if std == 0:
            return mean, mean

        return (
            float(mean - 1.96 * std),
            float(mean + 1.96 * std),
        )

    # ========================================================================
    # Utilities
    # ========================================================================

    @staticmethod
    def _utc_now() -> datetime:
        """Return timezone-aware UTC timestamp."""

        return datetime.now(timezone.utc)


# ============================================================================
# Specialized Financial Forecast Helpers
# ============================================================================


class RevenueForecastModel(FinancialForecastModel):
    """Specialized model for revenue forecasting."""

    def __init__(
        self,
        config: Optional[ForecastModelConfig] = None,
        estimator: Optional[Any] = None,
    ) -> None:

        config = config or ForecastModelConfig()

        super().__init__(
            config=config,
            estimator=estimator,
        )

        self._target_name = "revenue"


class ProfitForecastModel(FinancialForecastModel):
    """Specialized model for profit forecasting."""

    def __init__(
        self,
        config: Optional[ForecastModelConfig] = None,
        estimator: Optional[Any] = None,
    ) -> None:

        config = config or ForecastModelConfig()

        super().__init__(
            config=config,
            estimator=estimator,
        )

        self._target_name = "profit"


class CashFlowForecastModel(FinancialForecastModel):
    """Specialized model for cash-flow forecasting."""

    def __init__(
        self,
        config: Optional[ForecastModelConfig] = None,
        estimator: Optional[Any] = None,
    ) -> None:

        config = config or ForecastModelConfig()

        super().__init__(
            config=config,
            estimator=estimator,
        )

        self._target_name = "cash_flow"


# ============================================================================
# Convenience Functions
# ============================================================================


def train_forecast_model(
    X: Any,
    y: Sequence[float],
    feature_names: Optional[Sequence[str]] = None,
    target_name: str = "target",
    config: Optional[ForecastModelConfig] = None,
    estimator: Optional[Any] = None,
) -> FinancialForecastModel:
    """
    Train and return a forecasting model.
    """

    model = FinancialForecastModel(
        config=config,
        estimator=estimator,
    )

    model.fit(
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target_name,
    )

    return model


def forecast_with_model(
    model: FinancialForecastModel,
    X_future: Any,
) -> List[float]:
    """
    Generate forecasts using an already-trained model.
    """

    return model.predict_batch(X_future)


def historical_average_forecast(
    values: Sequence[float],
    horizon: int,
) -> List[float]:
    """
    Convenience historical-average forecast.
    """

    model = FinancialForecastModel(
        ForecastModelConfig(
            model_type=ForecastModelType.HISTORICAL_AVERAGE,
            minimum_training_samples=2,
        )
    )

    return model.historical_average_forecast(
        values,
        horizon,
    ).predictions


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "ForecastModelError",
    "InvalidModelInputError",
    "ModelNotFittedError",
    "ModelTrainingError",
    "ModelPredictionError",
    "UnsupportedModelError",
    "InsufficientTrainingDataError",

    # Enums
    "ForecastModelType",
    "ForecastDirection",

    # Configuration
    "ForecastModelConfig",

    # Results
    "TrainingResult",
    "ForecastPrediction",

    # Models
    "FinancialForecastModel",
    "RevenueForecastModel",
    "ProfitForecastModel",
    "CashFlowForecastModel",

    # Helpers
    "train_forecast_model",
    "forecast_with_model",
    "historical_average_forecast",
]