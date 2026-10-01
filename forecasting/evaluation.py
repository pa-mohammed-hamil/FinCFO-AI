"""
FinCo AI - Forecasting Evaluation

File:
    backend/app/forecasting/evaluation.py

Purpose:
    Evaluate forecasting models consistently across:

        - Revenue
        - Profit
        - Cash Flow
        - Liquidity
        - Other financial time series

Supported metrics:

    MAE
    MSE
    RMSE
    MAPE
    sMAPE
    WAPE
    MASE
    Bias
    Forecast Accuracy

Evaluation flow:

    Historical Data
          |
          v
    Train / Validation Split
          |
          v
    Forecast Model
          |
          v
    Predictions
          |
          v
    Evaluation Metrics
          |
          v
    Model Comparison
          |
          v
    Best Model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
import math


# ============================================================================
# Exceptions
# ============================================================================


class ForecastEvaluationError(Exception):
    """Base exception for forecasting evaluation."""


class InvalidEvaluationInputError(
    ForecastEvaluationError
):
    """Raised when evaluation input is invalid."""


class InsufficientEvaluationDataError(
    ForecastEvaluationError
):
    """Raised when insufficient data is available."""


class MetricCalculationError(
    ForecastEvaluationError
):
    """Raised when a metric cannot be calculated."""


# ============================================================================
# Enums
# ============================================================================


class EvaluationMetric(str, Enum):
    """Supported forecasting evaluation metrics."""

    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    MAPE = "mape"
    SMAPE = "smape"
    WAPE = "wape"
    MASE = "mase"
    BIAS = "bias"
    ACCURACY = "accuracy"


class EvaluationDirection(str, Enum):
    """Whether lower or higher metric values are better."""

    LOWER = "lower"
    HIGHER = "higher"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvaluationConfig:
    """
    Forecast evaluation configuration.
    """

    default_metrics: Tuple[
        EvaluationMetric, ...
    ] = (
        EvaluationMetric.MAE,
        EvaluationMetric.RMSE,
        EvaluationMetric.MAPE,
        EvaluationMetric.WAPE,
    )

    validation_ratio: float = 0.20

    minimum_training_size: int = 3

    minimum_validation_size: int = 1

    zero_tolerance: float = 1e-8

    mape_zero_policy: str = "ignore"

    mase_seasonal_period: int = 1

    def validate(self) -> None:
        """Validate configuration."""

        if not (
            0.0
            < self.validation_ratio
            < 1.0
        ):
            raise ValueError(
                "validation_ratio must be between "
                "0 and 1."
            )

        if self.minimum_training_size < 1:
            raise ValueError(
                "minimum_training_size must be >= 1."
            )

        if self.minimum_validation_size < 1:
            raise ValueError(
                "minimum_validation_size must be >= 1."
            )

        if self.zero_tolerance <= 0:
            raise ValueError(
                "zero_tolerance must be > 0."
            )

        if self.mase_seasonal_period < 1:
            raise ValueError(
                "mase_seasonal_period must be >= 1."
            )

        if self.mape_zero_policy not in {
            "ignore",
            "zero",
        }:
            raise ValueError(
                "mape_zero_policy must be "
                "'ignore' or 'zero'."
            )


# ============================================================================
# Result Models
# ============================================================================


@dataclass
class MetricResult:
    """
    Result for one evaluation metric.
    """

    metric: str

    value: float

    direction: str

    interpretation: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""

        return {
            "metric": self.metric,
            "value": self.value,
            "direction": self.direction,
            "interpretation": self.interpretation,
            "metadata": dict(self.metadata),
        }


@dataclass
class ForecastEvaluationResult:
    """
    Complete evaluation result for one model.
    """

    model_name: str

    actuals: List[float]

    predictions: List[float]

    metrics: Dict[str, float]

    metric_results: List[
        MetricResult
    ]

    validation_size: int

    training_size: Optional[int]

    score: float

    rank: Optional[int]

    passed: bool

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result."""

        return {
            "model_name": self.model_name,
            "actuals": list(self.actuals),
            "predictions": list(self.predictions),
            "metrics": dict(self.metrics),
            "metric_results": [
                item.to_dict()
                for item in self.metric_results
            ],
            "validation_size":
                self.validation_size,
            "training_size":
                self.training_size,
            "score": self.score,
            "rank": self.rank,
            "passed": self.passed,
            "metadata": dict(self.metadata),
        }


@dataclass
class ModelComparisonResult:
    """
    Comparison result for multiple forecasting models.
    """

    results: List[
        ForecastEvaluationResult
    ]

    best_model: Optional[str]

    ranking: List[str]

    metric: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison to dictionary."""

        return {
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "best_model": self.best_model,
            "ranking": list(self.ranking),
            "metric": self.metric,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Forecast Evaluator
# ============================================================================


class ForecastEvaluator:
    """
    Production-style forecasting evaluation service.

    The evaluator is model-agnostic.

    A model can be:

        - baseline.py model
        - forecasting/model.py
        - sklearn model
        - statistical model
        - custom forecasting function
    """

    METRIC_DIRECTION = {
        EvaluationMetric.MAE: (
            EvaluationDirection.LOWER
        ),
        EvaluationMetric.MSE: (
            EvaluationDirection.LOWER
        ),
        EvaluationMetric.RMSE: (
            EvaluationDirection.LOWER
        ),
        EvaluationMetric.MAPE: (
            EvaluationDirection.LOWER
        ),
        EvaluationMetric.SMAPE: (
            EvaluationDirection.LOWER
        ),
        EvaluationMetric.WAPE: (
            EvaluationDirection.LOWER
        ),
        EvaluationMetric.MASE: (
            EvaluationDirection.LOWER
        ),
        EvaluationMetric.BIAS: (
            EvaluationDirection.LOWER
        ),
        EvaluationMetric.ACCURACY: (
            EvaluationDirection.HIGHER
        ),
    }

    def __init__(
        self,
        config: Optional[
            EvaluationConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or EvaluationConfig()
        )

        self.config.validate()

    # ========================================================================
    # MAIN EVALUATION API
    # ========================================================================

    def evaluate(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
        model_name: str = "model",
        training_size: Optional[int] = None,
        metrics: Optional[
            Sequence[
                EvaluationMetric | str
            ]
        ] = None,
    ) -> ForecastEvaluationResult:
        """
        Evaluate predictions against actual values.
        """

        actual_values = self._validate_series(
            actuals,
            "actuals",
        )

        predicted_values = self._validate_series(
            predictions,
            "predictions",
        )

        if len(actual_values) != len(
            predicted_values
        ):
            raise InvalidEvaluationInputError(
                "actuals and predictions must "
                "have equal lengths."
            )

        if not actual_values:
            raise InvalidEvaluationInputError(
                "Evaluation data cannot be empty."
            )

        selected_metrics = (
            self._normalize_metrics(
                metrics
            )
        )

        metric_values: Dict[
            str,
            float,
        ] = {}

        metric_results: List[
            MetricResult
        ] = []

        for metric in selected_metrics:

            value = self.calculate_metric(
                metric,
                actual_values,
                predicted_values,
            )

            metric_values[
                metric.value
            ] = value

            metric_results.append(
                MetricResult(
                    metric=metric.value,
                    value=value,
                    direction=(
                        self.METRIC_DIRECTION[
                            metric
                        ].value
                    ),
                    interpretation=(
                        self.interpret_metric(
                            metric,
                            value,
                        )
                    ),
                )
            )

        score = self.calculate_model_score(
            metric_values
        )

        passed = self.is_acceptable(
            metric_values
        )

        return ForecastEvaluationResult(
            model_name=model_name,
            actuals=actual_values,
            predictions=predicted_values,
            metrics=metric_values,
            metric_results=metric_results,
            validation_size=len(
                actual_values
            ),
            training_size=training_size,
            score=score,
            rank=None,
            passed=passed,
            metadata={
                "metric_count":
                    len(selected_metrics),
            },
        )

    # ========================================================================
    # MODEL EVALUATION
    # ========================================================================

    def evaluate_model(
        self,
        historical_values: Sequence[float],
        forecast_function: Callable[
            [Sequence[float], int],
            Sequence[float],
        ],
        model_name: str = "model",
        validation_size: Optional[int] = None,
        metrics: Optional[
            Sequence[
                EvaluationMetric | str
            ]
        ] = None,
    ) -> ForecastEvaluationResult:
        """
        Evaluate a forecasting function using
        a chronological holdout set.

        Important:
            No random train/test split is used.
            Time-series order is preserved.
        """

        values = self._validate_series(
            historical_values,
            "historical_values",
        )

        train, validation = (
            self.train_validation_split(
                values,
                validation_size,
            )
        )

        predictions = forecast_function(
            train,
            len(validation),
        )

        predictions = self._validate_series(
            predictions,
            "predictions",
        )

        if len(predictions) != len(
            validation
        ):
            raise InvalidEvaluationInputError(
                "Forecast function returned "
                "an incorrect horizon."
            )

        return self.evaluate(
            actuals=validation,
            predictions=predictions,
            model_name=model_name,
            training_size=len(train),
            metrics=metrics,
        )

    # ========================================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================================

    def train_validation_split(
        self,
        values: Sequence[float],
        validation_size: Optional[int] = None,
    ) -> Tuple[
        List[float],
        List[float],
    ]:
        """
        Chronological train/validation split.
        """

        series = self._validate_series(
            values,
            "values",
        )

        if len(series) < (
            self.config.minimum_training_size
            + self.config.minimum_validation_size
        ):
            raise InsufficientEvaluationDataError(
                "Insufficient observations for "
                "train/validation evaluation."
            )

        if validation_size is None:

            validation_size = max(
                self.config.minimum_validation_size,
                int(
                    len(series)
                    * self.config.validation_ratio
                ),
            )

        if validation_size < 1:
            raise InvalidEvaluationInputError(
                "validation_size must be >= 1."
            )

        if validation_size >= len(series):
            raise InvalidEvaluationInputError(
                "validation_size must be smaller "
                "than total history."
            )

        training_size = (
            len(series)
            - validation_size
        )

        if training_size < (
            self.config.minimum_training_size
        ):
            raise InsufficientEvaluationDataError(
                "Training set is too small."
            )

        return (
            series[:training_size],
            series[training_size:],
        )

    # ========================================================================
    # METRIC CALCULATION
    # ========================================================================

    def calculate_metric(
        self,
        metric: EvaluationMetric | str,
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> float:
        """Calculate a single evaluation metric."""

        metric = self._normalize_metric(
            metric
        )

        actual_values = self._validate_series(
            actuals,
            "actuals",
        )

        predicted_values = self._validate_series(
            predictions,
            "predictions",
        )

        self._validate_equal_length(
            actual_values,
            predicted_values,
        )

        if metric == EvaluationMetric.MAE:
            return self.mae(
                actual_values,
                predicted_values,
            )

        if metric == EvaluationMetric.MSE:
            return self.mse(
                actual_values,
                predicted_values,
            )

        if metric == EvaluationMetric.RMSE:
            return self.rmse(
                actual_values,
                predicted_values,
            )

        if metric == EvaluationMetric.MAPE:
            return self.mape(
                actual_values,
                predicted_values,
            )

        if metric == EvaluationMetric.SMAPE:
            return self.smape(
                actual_values,
                predicted_values,
            )

        if metric == EvaluationMetric.WAPE:
            return self.wape(
                actual_values,
                predicted_values,
            )

        if metric == EvaluationMetric.MASE:
            return self.mase(
                actual_values,
                predicted_values,
            )

        if metric == EvaluationMetric.BIAS:
            return self.bias(
                actual_values,
                predicted_values,
            )

        if metric == EvaluationMetric.ACCURACY:
            return self.accuracy(
                actual_values,
                predicted_values,
            )

        raise MetricCalculationError(
            f"Unsupported metric: {metric}"
        )

    # ========================================================================
    # MAE
    # ========================================================================

    @staticmethod
    def mae(
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> float:
        """
        Mean Absolute Error.

        Lower is better.
        """

        ForecastEvaluator._validate_equal_length(
            actuals,
            predictions,
        )

        return sum(
            abs(
                actual - prediction
            )
            for actual, prediction
            in zip(
                actuals,
                predictions,
            )
        ) / len(actuals)

    # ========================================================================
    # MSE
    # ========================================================================

    @staticmethod
    def mse(
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> float:
        """
        Mean Squared Error.

        Lower is better.
        """

        ForecastEvaluator._validate_equal_length(
            actuals,
            predictions,
        )

        return sum(
            (
                actual - prediction
            ) ** 2
            for actual, prediction
            in zip(
                actuals,
                predictions,
            )
        ) / len(actuals)

    # ========================================================================
    # RMSE
    # ========================================================================

    @staticmethod
    def rmse(
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> float:
        """Root Mean Squared Error."""

        return math.sqrt(
            ForecastEvaluator.mse(
                actuals,
                predictions,
            )
        )

    # ========================================================================
    # MAPE
    # ========================================================================

    def mape(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> float:
        """
        Mean Absolute Percentage Error.

        Returned as percentage.

        Example:
            12.5 means 12.5%.
        """

        self._validate_equal_length(
            actuals,
            predictions,
        )

        errors = []

        for actual, prediction in zip(
            actuals,
            predictions,
        ):

            if abs(actual) <= (
                self.config.zero_tolerance
            ):

                if (
                    self.config.mape_zero_policy
                    == "zero"
                ):

                    errors.append(
                        abs(
                            actual - prediction
                        )
                    )

                continue

            errors.append(
                abs(
                    actual - prediction
                )
                / abs(actual)
            )

        if not errors:
            return 0.0

        return (
            sum(errors)
            / len(errors)
        ) * 100.0

    # ========================================================================
    # sMAPE
    # ========================================================================

    def smape(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> float:
        """
        Symmetric Mean Absolute Percentage Error.

        Returned as percentage.
        """

        self._validate_equal_length(
            actuals,
            predictions,
        )

        errors = []

        for actual, prediction in zip(
            actuals,
            predictions,
        ):

            denominator = (
                abs(actual)
                + abs(prediction)
            )

            if denominator <= (
                self.config.zero_tolerance
            ):
                continue

            errors.append(
                2.0
                * abs(
                    actual - prediction
                )
                / denominator
            )

        if not errors:
            return 0.0

        return (
            sum(errors)
            / len(errors)
        ) * 100.0

    # ========================================================================
    # WAPE
    # ========================================================================

    def wape(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> float:
        """
        Weighted Absolute Percentage Error.

        WAPE = sum(abs(error)) / sum(abs(actual)).

        Returned as percentage.
        """

        self._validate_equal_length(
            actuals,
            predictions,
        )

        numerator = sum(
            abs(
                actual - prediction
            )
            for actual, prediction
            in zip(
                actuals,
                predictions,
            )
        )

        denominator = sum(
            abs(actual)
            for actual in actuals
        )

        if denominator <= (
            self.config.zero_tolerance
        ):
            return 0.0

        return (
            numerator
            / denominator
        ) * 100.0

    # ========================================================================
    # MASE
    # ========================================================================

    def mase(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
        training_values: Optional[
            Sequence[float]
        ] = None,
        seasonal_period: Optional[int] = None,
    ) -> float:
        """
        Mean Absolute Scaled Error.

        If training_values is not supplied, the actual
        validation history is used as the scaling series.
        """

        self._validate_equal_length(
            actuals,
            predictions,
        )

        training = (
            list(training_values)
            if training_values is not None
            else list(actuals)
        )

        training = self._validate_series(
            training,
            "training_values",
        )

        period = (
            seasonal_period
            if seasonal_period is not None
            else self.config.mase_seasonal_period
        )

        if len(training) <= period:
            return 0.0

        scale_errors = [
            abs(
                training[index]
                - training[
                    index - period
                ]
            )
            for index in range(
                period,
                len(training),
            )
        ]

        scale = (
            sum(scale_errors)
            / len(scale_errors)
        )

        if scale <= (
            self.config.zero_tolerance
        ):
            return 0.0

        forecast_error = self.mae(
            actuals,
            predictions,
        )

        return (
            forecast_error
            / scale
        )

    # ========================================================================
    # BIAS
    # ========================================================================

    @staticmethod
    def bias(
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> float:
        """
        Mean Forecast Error.

        Positive:
            over-forecasting.

        Negative:
            under-forecasting.
        """

        ForecastEvaluator._validate_equal_length(
            actuals,
            predictions,
        )

        return sum(
            prediction - actual
            for actual, prediction
            in zip(
                actuals,
                predictions,
            )
        ) / len(actuals)

    # ========================================================================
    # ACCURACY
    # ========================================================================

    def accuracy(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> float:
        """
        Simple forecast accuracy.

        Accuracy = 100 - WAPE.

        Clamped to [0, 100].
        """

        error = self.wape(
            actuals,
            predictions,
        )

        return max(
            0.0,
            min(
                100.0,
                100.0 - error,
            ),
        )

    # ========================================================================
    # MODEL COMPARISON
    # ========================================================================

    def compare_models(
        self,
        results: Sequence[
            ForecastEvaluationResult
        ],
        metric: EvaluationMetric | str = (
            EvaluationMetric.RMSE
        ),
    ) -> ModelComparisonResult:
        """
        Compare already evaluated models.
        """

        if not results:
            raise InvalidEvaluationInputError(
                "results cannot be empty."
            )

        metric = self._normalize_metric(
            metric
        )

        result_list = list(results)

        for result in result_list:

            if metric.value not in (
                result.metrics
            ):
                raise InvalidEvaluationInputError(
                    f"Metric '{metric.value}' "
                    f"is missing from model "
                    f"'{result.model_name}'."
                )

        direction = (
            self.METRIC_DIRECTION[
                metric
            ]
        )

        reverse = (
            direction
            == EvaluationDirection.HIGHER
        )

        ranked = sorted(
            result_list,
            key=lambda item:
                item.metrics[
                    metric.value
                ],
            reverse=reverse,
        )

        for rank, result in enumerate(
            ranked,
            start=1,
        ):
            result.rank = rank

        ranking = [
            result.model_name
            for result in ranked
        ]

        return ModelComparisonResult(
            results=ranked,
            best_model=(
                ranked[0].model_name
                if ranked
                else None
            ),
            ranking=ranking,
            metric=metric.value,
        )

    # ========================================================================
    # MODEL SCORING
    # ========================================================================

    def calculate_model_score(
        self,
        metrics: Mapping[str, float],
    ) -> float:
        """
        Convert model metrics into a normalized quality score.

        Higher is better.

        The score is intentionally a ranking score,
        not a statistical probability.
        """

        components = []

        if "accuracy" in metrics:

            accuracy = self._bounded(
                metrics["accuracy"],
                0.0,
                100.0,
            )

            components.append(
                accuracy
            )

        if "mape" in metrics:

            mape = max(
                0.0,
                metrics["mape"],
            )

            components.append(
                max(
                    0.0,
                    100.0 - mape,
                )
            )

        if "smape" in metrics:

            smape = max(
                0.0,
                metrics["smape"],
            )

            components.append(
                max(
                    0.0,
                    100.0 - smape,
                )
            )

        if "wape" in metrics:

            wape = max(
                0.0,
                metrics["wape"],
            )

            components.append(
                max(
                    0.0,
                    100.0 - wape,
                )
            )

        if not components:
            return 0.0

        return round(
            max(
                0.0,
                min(
                    100.0,
                    sum(components)
                    / len(components),
                ),
            ),
            4,
        )

    # ========================================================================
    # QUALITY CHECKS
    # ========================================================================

    def is_acceptable(
        self,
        metrics: Mapping[str, float],
        max_mape: float = 30.0,
        max_wape: float = 30.0,
    ) -> bool:
        """
        Determine whether a forecast meets basic quality thresholds.
        """

        if (
            "mape" in metrics
            and metrics["mape"] > max_mape
        ):
            return False

        if (
            "wape" in metrics
            and metrics["wape"] > max_wape
        ):
            return False

        return True

    def interpret_metric(
        self,
        metric: EvaluationMetric | str,
        value: float,
    ) -> str:
        """Return a human-readable interpretation."""

        metric = self._normalize_metric(
            metric
        )

        if metric == EvaluationMetric.MAE:
            return (
                "Average absolute forecast "
                "error in original units."
            )

        if metric == EvaluationMetric.MSE:
            return (
                "Average squared forecast "
                "error; lower is better."
            )

        if metric == EvaluationMetric.RMSE:
            return (
                "Forecast error in original "
                "units with greater penalty "
                "for large errors."
            )

        if metric == EvaluationMetric.MAPE:
            return (
                "Average absolute percentage "
                "forecast error."
            )

        if metric == EvaluationMetric.SMAPE:
            return (
                "Symmetric percentage forecast "
                "error."
            )

        if metric == EvaluationMetric.WAPE:
            return (
                "Total absolute error relative "
                "to total actual value."
            )

        if metric == EvaluationMetric.MASE:
            return (
                "Forecast error scaled against "
                "a naive benchmark."
            )

        if metric == EvaluationMetric.BIAS:

            if value > 0:
                return (
                    "Forecast tends to overestimate "
                    "actual values."
                )

            if value < 0:
                return (
                    "Forecast tends to underestimate "
                    "actual values."
                )

            return (
                "Forecast has approximately zero bias."
            )

        if metric == EvaluationMetric.ACCURACY:
            return (
                "Approximate forecast accuracy "
                "derived from WAPE."
            )

        return "No interpretation available."

    # ========================================================================
    # CONVENIENCE EVALUATIONS
    # ========================================================================

    def evaluate_baseline_models(
        self,
        historical_values: Sequence[float],
        models: Mapping[
            str,
            Callable[
                [Sequence[float], int],
                Sequence[float],
            ],
        ],
        validation_size: Optional[int] = None,
        metric: EvaluationMetric | str = (
            EvaluationMetric.RMSE
        ),
    ) -> ModelComparisonResult:
        """
        Evaluate multiple baseline models.

        Example models:

            {
                "naive": forecaster.naive,
                "moving_average":
                    forecaster.moving_average,
                "drift": forecaster.drift,
            }
        """

        if not models:
            raise InvalidEvaluationInputError(
                "models cannot be empty."
            )

        results = []

        for name, function in models.items():

            result = self.evaluate_model(
                historical_values=(
                    historical_values
                ),
                forecast_function=function,
                model_name=name,
                validation_size=validation_size,
            )

            results.append(
                result
            )

        return self.compare_models(
            results,
            metric=metric,
        )

    # ========================================================================
    # RESIDUAL ANALYSIS
    # ========================================================================

    def residuals(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> List[float]:
        """
        Calculate residuals.

        residual = actual - prediction
        """

        self._validate_equal_length(
            actuals,
            predictions,
        )

        return [
            actual - prediction
            for actual, prediction
            in zip(
                actuals,
                predictions,
            )
        ]

    def residual_summary(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> Dict[str, float]:
        """Return residual statistics."""

        residual_values = (
            self.residuals(
                actuals,
                predictions,
            )
        )

        if not residual_values:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "mae": 0.0,
            }

        mean = (
            sum(residual_values)
            / len(residual_values)
        )

        return {
            "mean": mean,
            "min": min(
                residual_values
            ),
            "max": max(
                residual_values
            ),
            "mae": (
                sum(
                    abs(value)
                    for value
                    in residual_values
                )
                / len(residual_values)
            ),
        }

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _normalize_metrics(
        self,
        metrics: Optional[
            Sequence[
                EvaluationMetric | str
            ]
        ],
    ) -> List[
        EvaluationMetric
    ]:

        if metrics is None:

            return list(
                self.config.default_metrics
            )

        if not metrics:
            raise InvalidEvaluationInputError(
                "metrics cannot be empty."
            )

        return [
            self._normalize_metric(
                metric
            )
            for metric in metrics
        ]

    @staticmethod
    def _normalize_metric(
        metric: EvaluationMetric | str,
    ) -> EvaluationMetric:

        if isinstance(
            metric,
            EvaluationMetric,
        ):
            return metric

        try:

            return EvaluationMetric(
                str(
                    metric
                ).strip().lower()
            )

        except ValueError as exc:

            raise InvalidEvaluationInputError(
                f"Unsupported evaluation metric: "
                f"{metric}"
            ) from exc

    @staticmethod
    def _validate_series(
        values: Sequence[float],
        name: str,
    ) -> List[float]:

        if values is None:
            raise InvalidEvaluationInputError(
                f"{name} cannot be None."
            )

        try:

            result = [
                float(value)
                for value in values
            ]

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidEvaluationInputError(
                f"{name} must contain numeric "
                "values."
            ) from exc

        if not result:
            raise InvalidEvaluationInputError(
                f"{name} cannot be empty."
            )

        for value in result:

            if not math.isfinite(
                value
            ):
                raise InvalidEvaluationInputError(
                    f"{name} contains non-finite "
                    "values."
                )

        return result

    @staticmethod
    def _validate_equal_length(
        actuals: Sequence[float],
        predictions: Sequence[float],
    ) -> None:

        if len(actuals) != len(
            predictions
        ):

            raise InvalidEvaluationInputError(
                "actuals and predictions must "
                "have equal lengths."
            )

        if len(actuals) == 0:

            raise InvalidEvaluationInputError(
                "Evaluation series cannot be empty."
            )

    @staticmethod
    def _bounded(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def evaluate_forecast(
    actuals: Sequence[float],
    predictions: Sequence[float],
    model_name: str = "model",
) -> ForecastEvaluationResult:
    """Convenience forecast evaluation API."""

    return ForecastEvaluator().evaluate(
        actuals=actuals,
        predictions=predictions,
        model_name=model_name,
    )


def calculate_mae(
    actuals: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Convenience MAE calculation."""

    return ForecastEvaluator.mae(
        actuals,
        predictions,
    )


def calculate_rmse(
    actuals: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Convenience RMSE calculation."""

    return ForecastEvaluator.rmse(
        actuals,
        predictions,
    )


def calculate_mape(
    actuals: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Convenience MAPE calculation."""

    return ForecastEvaluator().mape(
        actuals,
        predictions,
    )


def calculate_wape(
    actuals: Sequence[float],
    predictions: Sequence[float],
) -> float:
    """Convenience WAPE calculation."""

    return ForecastEvaluator().wape(
        actuals,
        predictions,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "ForecastEvaluationError",
    "InvalidEvaluationInputError",
    "InsufficientEvaluationDataError",
    "MetricCalculationError",
    "EvaluationMetric",
    "EvaluationDirection",
    "EvaluationConfig",
    "MetricResult",
    "ForecastEvaluationResult",
    "ModelComparisonResult",
    "ForecastEvaluator",
    "evaluate_forecast",
    "calculate_mae",
    "calculate_rmse",
    "calculate_mape",
    "calculate_wape",
]