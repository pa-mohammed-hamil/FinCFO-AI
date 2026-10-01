"""
FinCo AI - Forecasting Baseline Models

File:
    backend/app/forecasting/baseline.py

Purpose:
    Provides simple, reliable forecasting baselines for financial
    forecasting.

Supported baselines:
    - Naive
    - Seasonal Naive
    - Mean
    - Moving Average
    - Weighted Moving Average
    - Exponential Smoothing
    - Drift / Linear Trend

These models are intentionally simple.

They provide:
    1. A benchmark for advanced ML models.
    2. A fallback when insufficient historical data exists.
    3. A reference point for model evaluation.
    4. A transparent forecast that can be explained easily.

Architecture:

    Historical Financial Data
             |
             v
       baseline.py
             |
       +-----+-----+----------------+
       |     |     |                |
       v     v     v                v
     Naive  Mean  Moving Avg   Exponential
       |     |     |                |
       +-----+-----+----------------+
                     |
                     v
                 Forecast
                     |
                     v
              evaluation.py
                     |
                     v
                model.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence
import math


# ============================================================================
# Exceptions
# ============================================================================


class BaselineForecastError(Exception):
    """Base exception for baseline forecasting."""


class InvalidForecastInputError(
    BaselineForecastError
):
    """Raised when forecast input is invalid."""


class InsufficientDataError(
    BaselineForecastError
):
    """Raised when insufficient historical data is available."""


class UnsupportedBaselineError(
    BaselineForecastError
):
    """Raised when an unsupported baseline is requested."""


# ============================================================================
# Enums
# ============================================================================


class BaselineMethod(str, Enum):
    """
    Supported baseline forecasting methods.
    """

    NAIVE = "naive"

    SEASONAL_NAIVE = "seasonal_naive"

    MEAN = "mean"

    MOVING_AVERAGE = "moving_average"

    WEIGHTED_MOVING_AVERAGE = (
        "weighted_moving_average"
    )

    EXPONENTIAL_SMOOTHING = (
        "exponential_smoothing"
    )

    DRIFT = "drift"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class BaselineConfig:
    """
    Configuration for baseline forecasting.
    """

    moving_average_window: int = 3

    weighted_window: int = 3

    exponential_alpha: float = 0.30

    seasonal_period: int = 12

    min_history: int = 2

    allow_negative_forecast: bool = True

    clip_negative_forecast: bool = False

    def validate(self) -> None:
        """Validate configuration."""

        if self.moving_average_window < 1:
            raise ValueError(
                "moving_average_window must be >= 1."
            )

        if self.weighted_window < 1:
            raise ValueError(
                "weighted_window must be >= 1."
            )

        if not (
            0.0
            < self.exponential_alpha
            <= 1.0
        ):
            raise ValueError(
                "exponential_alpha must be "
                "greater than 0 and <= 1."
            )

        if self.seasonal_period < 1:
            raise ValueError(
                "seasonal_period must be >= 1."
            )

        if self.min_history < 1:
            raise ValueError(
                "min_history must be >= 1."
            )


# ============================================================================
# Forecast Result
# ============================================================================


@dataclass
class BaselineForecastResult:
    """
    Result returned by a baseline forecast.
    """

    method: str

    forecast: List[float]

    historical_values: List[float]

    horizon: int

    last_value: Optional[float]

    mean_value: Optional[float]

    trend: Optional[float]

    confidence: float

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """Convert result to dictionary."""

        return {
            "method":
                self.method,

            "forecast":
                list(self.forecast),

            "historical_values":
                list(self.historical_values),

            "horizon":
                self.horizon,

            "last_value":
                self.last_value,

            "mean_value":
                self.mean_value,

            "trend":
                self.trend,

            "confidence":
                self.confidence,

            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Baseline Forecaster
# ============================================================================


class BaselineForecaster:
    """
    Financial forecasting baseline engine.

    This class does not require external ML libraries.

    It is useful for:

        revenue forecasting
        profit forecasting
        cash-flow forecasting
        expense forecasting
        balance forecasting
        KPI forecasting
    """

    def __init__(
        self,
        config: Optional[
            BaselineConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or BaselineConfig()
        )

        self.config.validate()

    # ========================================================================
    # MAIN FORECAST API
    # ========================================================================

    def forecast(
        self,
        values: Sequence[float],
        horizon: int,
        method: BaselineMethod | str = (
            BaselineMethod.NAIVE
        ),
        seasonal_period: Optional[int] = None,
    ) -> BaselineForecastResult:
        """
        Generate a forecast using the requested baseline.

        Parameters:
            values:
                Historical numerical observations.

            horizon:
                Number of future periods.

            method:
                Baseline forecasting method.

            seasonal_period:
                Optional seasonal period for seasonal naive forecasting.
        """

        history = self._validate_values(
            values
        )

        self._validate_horizon(
            horizon
        )

        method = self._normalize_method(
            method
        )

        if method == BaselineMethod.NAIVE:

            forecast = self.naive(
                history,
                horizon,
            )

        elif (
            method
            == BaselineMethod.SEASONAL_NAIVE
        ):

            forecast = (
                self.seasonal_naive(
                    history,
                    horizon,
                    seasonal_period=(
                        seasonal_period
                        or self.config
                        .seasonal_period
                    ),
                )
            )

        elif method == BaselineMethod.MEAN:

            forecast = self.mean(
                history,
                horizon,
            )

        elif (
            method
            == BaselineMethod.MOVING_AVERAGE
        ):

            forecast = (
                self.moving_average(
                    history,
                    horizon,
                    window=(
                        self.config
                        .moving_average_window
                    ),
                )
            )

        elif (
            method
            == BaselineMethod.WEIGHTED_MOVING_AVERAGE
        ):

            forecast = (
                self.weighted_moving_average(
                    history,
                    horizon,
                    window=(
                        self.config
                        .weighted_window
                    ),
                )
            )

        elif (
            method
            == BaselineMethod.EXPONENTIAL_SMOOTHING
        ):

            forecast = (
                self.exponential_smoothing(
                    history,
                    horizon,
                    alpha=(
                        self.config
                        .exponential_alpha
                    ),
                )
            )

        elif method == BaselineMethod.DRIFT:

            forecast = self.drift(
                history,
                horizon,
            )

        else:

            raise UnsupportedBaselineError(
                f"Unsupported baseline method: "
                f"{method}"
            )

        forecast = self._post_process_forecast(
            forecast
        )

        trend = self.calculate_trend(
            history
        )

        confidence = self.estimate_confidence(
            history,
            forecast,
            method,
        )

        return BaselineForecastResult(
            method=method.value,
            forecast=forecast,
            historical_values=list(history),
            horizon=horizon,
            last_value=history[-1],
            mean_value=self._mean(history),
            trend=trend,
            confidence=confidence,
            metadata={
                "history_length":
                    len(history),

                "method":
                    method.value,

                "seasonal_period":
                    seasonal_period
                    or self.config
                    .seasonal_period,

                "moving_average_window":
                    self.config
                    .moving_average_window,
            },
        )

    # ========================================================================
    # NAIVE
    # ========================================================================

    def naive(
        self,
        values: Sequence[float],
        horizon: int,
    ) -> List[float]:
        """
        Naive forecast.

        Every future value equals the latest observation.

        Example:

            history = [100, 110, 120]

            horizon = 3

            forecast = [120, 120, 120]
        """

        history = self._validate_values(
            values
        )

        self._validate_horizon(
            horizon
        )

        last = history[-1]

        return [
            last
            for _ in range(
                horizon
            )
        ]

    # ========================================================================
    # SEASONAL NAIVE
    # ========================================================================

    def seasonal_naive(
        self,
        values: Sequence[float],
        horizon: int,
        seasonal_period: int,
    ) -> List[float]:
        """
        Seasonal naive forecast.

        Example for monthly data:

            seasonal_period = 12

        Each future month uses the corresponding value
        from the previous seasonal cycle.
        """

        history = self._validate_values(
            values
        )

        self._validate_horizon(
            horizon
        )

        if seasonal_period < 1:

            raise InvalidForecastInputError(
                "seasonal_period must be >= 1."
            )

        if len(history) < seasonal_period:

            raise InsufficientDataError(
                "Seasonal naive forecasting requires "
                f"at least {seasonal_period} observations."
            )

        forecast = []

        for step in range(
            horizon
        ):

            index = (
                len(history)
                - seasonal_period
                + (
                    step
                    % seasonal_period
                )
            )

            forecast.append(
                history[index]
            )

        return forecast

    # ========================================================================
    # MEAN
    # ========================================================================

    def mean(
        self,
        values: Sequence[float],
        horizon: int,
    ) -> List[float]:
        """
        Mean-value forecast.

        Every future observation equals the historical mean.
        """

        history = self._validate_values(
            values
        )

        self._validate_horizon(
            horizon
        )

        average = self._mean(
            history
        )

        return [
            average
            for _ in range(
                horizon
            )
        ]

    # ========================================================================
    # MOVING AVERAGE
    # ========================================================================

    def moving_average(
        self,
        values: Sequence[float],
        horizon: int,
        window: int,
    ) -> List[float]:
        """
        Moving-average forecast.

        Recursive forecasting is used:

            forecast(t+1)
                = mean(last window observations)

            forecast(t+2)
                = mean(updated last window)

        This is useful for relatively stable financial series.
        """

        history = self._validate_values(
            values
        )

        self._validate_horizon(
            horizon
        )

        if window < 1:

            raise InvalidForecastInputError(
                "window must be >= 1."
            )

        if len(history) < window:

            raise InsufficientDataError(
                f"Moving average requires at least "
                f"{window} observations."
            )

        working = list(
            history
        )

        forecast = []

        for _ in range(
            horizon
        ):

            recent = working[
                -window:
            ]

            prediction = self._mean(
                recent
            )

            forecast.append(
                prediction
            )

            working.append(
                prediction
            )

        return forecast

    # ========================================================================
    # WEIGHTED MOVING AVERAGE
    # ========================================================================

    def weighted_moving_average(
        self,
        values: Sequence[float],
        horizon: int,
        window: int,
    ) -> List[float]:
        """
        Weighted moving average.

        More recent observations receive larger weights.

        For window=3:

            oldest  -> weight 1
            middle  -> weight 2
            newest  -> weight 3
        """

        history = self._validate_values(
            values
        )

        self._validate_horizon(
            horizon
        )

        if window < 1:

            raise InvalidForecastInputError(
                "window must be >= 1."
            )

        if len(history) < window:

            raise InsufficientDataError(
                f"Weighted moving average requires "
                f"at least {window} observations."
            )

        working = list(
            history
        )

        forecast = []

        weights = list(
            range(
                1,
                window + 1,
            )
        )

        weight_sum = sum(
            weights
        )

        for _ in range(
            horizon
        ):

            recent = working[
                -window:
            ]

            weighted_sum = sum(
                value * weight
                for value, weight
                in zip(
                    recent,
                    weights,
                )
            )

            prediction = (
                weighted_sum
                / weight_sum
            )

            forecast.append(
                prediction
            )

            working.append(
                prediction
            )

        return forecast

    # ========================================================================
    # EXPONENTIAL SMOOTHING
    # ========================================================================

    def exponential_smoothing(
        self,
        values: Sequence[float],
        horizon: int,
        alpha: float,
    ) -> List[float]:
        """
        Simple exponential smoothing.

        Formula:

            S_t =
                alpha * X_t
                + (1-alpha) * S_(t-1)

        The final smoothed value is used for future periods.
        """

        history = self._validate_values(
            values
        )

        self._validate_horizon(
            horizon
        )

        if not (
            0.0
            < alpha
            <= 1.0
        ):

            raise InvalidForecastInputError(
                "alpha must be greater than "
                "0 and <= 1."
            )

        smoothed = history[0]

        for value in history[1:]:

            smoothed = (
                alpha * value
                + (
                    1.0 - alpha
                ) * smoothed
            )

        return [
            smoothed
            for _ in range(
                horizon
            )
        ]

    # ========================================================================
    # DRIFT
    # ========================================================================

    def drift(
        self,
        values: Sequence[float],
        horizon: int,
    ) -> List[float]:
        """
        Drift forecast.

        Estimates average change per historical period.

        Formula:

            drift =
                (last - first)
                / (n - 1)

        Future:

            forecast_t =
                last + drift * t
        """

        history = self._validate_values(
            values
        )

        self._validate_horizon(
            horizon
        )

        if len(history) < 2:

            raise InsufficientDataError(
                "Drift forecasting requires "
                "at least two observations."
            )

        change = (
            history[-1]
            - history[0]
        )

        periods = (
            len(history) - 1
        )

        drift_value = (
            change
            / periods
        )

        last = history[-1]

        return [
            last
            + drift_value
            * step
            for step in range(
                1,
                horizon + 1,
            )
        ]

    # ========================================================================
    # TREND
    # ========================================================================

    def calculate_trend(
        self,
        values: Sequence[float],
    ) -> float:
        """
        Calculate average absolute period-over-period change.
        """

        history = self._validate_values(
            values
        )

        if len(history) < 2:

            return 0.0

        return (
            history[-1]
            - history[0]
        ) / (
            len(history) - 1
        )

    def calculate_growth_rate(
        self,
        values: Sequence[float],
    ) -> float:
        """
        Calculate total growth rate from first to last value.

        Returns decimal form.

        Example:

            100 -> 120

            returns 0.20
        """

        history = self._validate_values(
            values
        )

        first = history[0]
        last = history[-1]

        if first == 0:

            return 0.0

        return (
            last - first
        ) / abs(first)

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def estimate_confidence(
        self,
        history: Sequence[float],
        forecast: Sequence[float],
        method: BaselineMethod,
    ) -> float:
        """
        Estimate a simple baseline confidence score.

        This is NOT a statistical prediction interval.

        It is an operational confidence indicator based on:

            - historical stability
            - amount of history
            - forecast consistency
        """

        values = self._validate_values(
            history
        )

        predictions = self._validate_values(
            forecast
        )

        if len(values) < 2:

            return 0.20

        mean = abs(
            self._mean(values)
        )

        if mean == 0:

            volatility = 0.0

        else:

            volatility = (
                self._standard_deviation(
                    values
                )
                / mean
            )

        stability = max(
            0.0,
            min(
                1.0,
                1.0 - volatility,
            ),
        )

        history_factor = min(
            1.0,
            len(values)
            / 24.0,
        )

        forecast_stability = (
            self._forecast_stability(
                predictions
            )
        )

        method_factor = {
            BaselineMethod.NAIVE: 0.75,
            BaselineMethod.SEASONAL_NAIVE: 0.80,
            BaselineMethod.MEAN: 0.65,
            BaselineMethod.MOVING_AVERAGE: 0.75,
            BaselineMethod.WEIGHTED_MOVING_AVERAGE: 0.78,
            BaselineMethod.EXPONENTIAL_SMOOTHING: 0.82,
            BaselineMethod.DRIFT: 0.70,
        }.get(
            method,
            0.70,
        )

        confidence = (
            0.35 * stability
            + 0.25 * history_factor
            + 0.20 * forecast_stability
            + 0.20 * method_factor
        )

        return round(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            4,
        )

    # ========================================================================
    # FORECAST COMPARISON
    # ========================================================================

    def forecast_all(
        self,
        values: Sequence[float],
        horizon: int,
        include_seasonal: bool = True,
    ) -> Dict[
        str,
        BaselineForecastResult
    ]:
        """
        Generate forecasts using all available baseline methods.
        """

        history = self._validate_values(
            values
        )

        results: Dict[
            str,
            BaselineForecastResult
        ] = {}

        methods = [
            BaselineMethod.NAIVE,
            BaselineMethod.MEAN,
            BaselineMethod.MOVING_AVERAGE,
            BaselineMethod.WEIGHTED_MOVING_AVERAGE,
            BaselineMethod.EXPONENTIAL_SMOOTHING,
            BaselineMethod.DRIFT,
        ]

        if (
            include_seasonal
            and len(history)
            >= self.config.seasonal_period
        ):

            methods.insert(
                1,
                BaselineMethod.SEASONAL_NAIVE,
            )

        for method in methods:

            try:

                results[
                    method.value
                ] = self.forecast(
                    history,
                    horizon,
                    method=method,
                )

            except InsufficientDataError:

                continue

        return results

    # ========================================================================
    # BEST SIMPLE BASELINE
    # ========================================================================

    def select_baseline(
        self,
        values: Sequence[float],
        horizon: int,
        validation_size: Optional[int] = None,
    ) -> BaselineForecastResult:
        """
        Select a baseline using simple holdout validation.

        The last portion of historical data is held out and each
        baseline is evaluated using MAE.

        This is useful before introducing advanced ML models.
        """

        history = self._validate_values(
            values
        )

        self._validate_horizon(
            horizon
        )

        if validation_size is None:

            validation_size = max(
                1,
                min(
                    horizon,
                    len(history) // 4,
                ),
            )

        if validation_size >= len(history):

            raise InsufficientDataError(
                "validation_size must be smaller "
                "than the historical dataset."
            )

        train = history[
            :-validation_size
        ]

        actual = history[
            -validation_size:
        ]

        candidates = [
            BaselineMethod.NAIVE,
            BaselineMethod.MEAN,
            BaselineMethod.MOVING_AVERAGE,
            BaselineMethod.WEIGHTED_MOVING_AVERAGE,
            BaselineMethod.EXPONENTIAL_SMOOTHING,
            BaselineMethod.DRIFT,
        ]

        if len(train) >= self.config.seasonal_period:

            candidates.insert(
                1,
                BaselineMethod.SEASONAL_NAIVE,
            )

        best_method = None
        best_error = float(
            "inf"
        )

        errors: Dict[
            str,
            float
        ] = {}

        for method in candidates:

            try:

                prediction = self.forecast(
                    train,
                    validation_size,
                    method=method,
                ).forecast

            except (
                BaselineForecastError
            ):

                continue

            error = self.mae(
                actual,
                prediction,
            )

            errors[
                method.value
            ] = error

            if error < best_error:

                best_error = error

                best_method = method

        if best_method is None:

            raise BaselineForecastError(
                "Unable to select a baseline "
                "from the available methods."
            )

        result = self.forecast(
            history,
            horizon,
            method=best_method,
        )

        result.metadata[
            "selection_method"
        ] = "holdout_mae"

        result.metadata[
            "validation_size"
        ] = validation_size

        result.metadata[
            "validation_errors"
        ] = errors

        return result

    # ========================================================================
    # METRICS
    # ========================================================================

    @staticmethod
    def mae(
        actual: Sequence[float],
        predicted: Sequence[float],
    ) -> float:
        """
        Mean Absolute Error.
        """

        a = BaselineForecaster._validate_values(
            actual
        )

        p = BaselineForecaster._validate_values(
            predicted
        )

        if len(a) != len(p):

            raise InvalidForecastInputError(
                "actual and predicted must "
                "have the same length."
            )

        if not a:

            return 0.0

        return sum(
            abs(
                x - y
            )
            for x, y in zip(
                a,
                p,
            )
        ) / len(a)

    @staticmethod
    def mse(
        actual: Sequence[float],
        predicted: Sequence[float],
    ) -> float:
        """
        Mean Squared Error.
        """

        a = BaselineForecaster._validate_values(
            actual
        )

        p = BaselineForecaster._validate_values(
            predicted
        )

        if len(a) != len(p):

            raise InvalidForecastInputError(
                "actual and predicted must "
                "have the same length."
            )

        if not a:

            return 0.0

        return sum(
            (
                x - y
            ) ** 2
            for x, y in zip(
                a,
                p,
            )
        ) / len(a)

    @staticmethod
    def rmse(
        actual: Sequence[float],
        predicted: Sequence[float],
    ) -> float:
        """
        Root Mean Squared Error.
        """

        return math.sqrt(
            BaselineForecaster.mse(
                actual,
                predicted,
            )
        )

    @staticmethod
    def mape(
        actual: Sequence[float],
        predicted: Sequence[float],
    ) -> float:
        """
        Mean Absolute Percentage Error.

        Zero actual values are ignored.
        """

        a = BaselineForecaster._validate_values(
            actual
        )

        p = BaselineForecaster._validate_values(
            predicted
        )

        if len(a) != len(p):

            raise InvalidForecastInputError(
                "actual and predicted must "
                "have the same length."
            )

        errors = []

        for actual_value, predicted_value in zip(
            a,
            p,
        ):

            if actual_value == 0:

                continue

            errors.append(
                abs(
                    (
                        actual_value
                        - predicted_value
                    )
                    / actual_value
                )
            )

        if not errors:

            return 0.0

        return (
            sum(errors)
            / len(errors)
            * 100.0
        )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _post_process_forecast(
        self,
        forecast: Sequence[float],
    ) -> List[float]:
        """
        Clean and optionally clip forecast values.
        """

        result = []

        for value in forecast:

            value = float(
                value
            )

            if not math.isfinite(
                value
            ):

                raise BaselineForecastError(
                    "Forecast contains "
                    "non-finite values."
                )

            if (
                self.config.clip_negative_forecast
                and not self.config
                .allow_negative_forecast
            ):

                value = max(
                    0.0,
                    value,
                )

            result.append(
                value
            )

        return result

    @staticmethod
    def _normalize_method(
        method: BaselineMethod | str,
    ) -> BaselineMethod:

        if isinstance(
            method,
            BaselineMethod,
        ):

            return method

        try:

            normalized = str(
                method
            ).strip().lower()

            return BaselineMethod(
                normalized
            )

        except ValueError as exc:

            raise UnsupportedBaselineError(
                f"Unsupported baseline method: "
                f"{method}"
            ) from exc

    @staticmethod
    def _validate_values(
        values: Sequence[float],
    ) -> List[float]:

        if values is None:

            raise InvalidForecastInputError(
                "values cannot be None."
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

            raise InvalidForecastInputError(
                "All historical values must "
                "be numeric."
            ) from exc

        if not result:

            raise InvalidForecastInputError(
                "Historical values cannot be empty."
            )

        for value in result:

            if not math.isfinite(
                value
            ):

                raise InvalidForecastInputError(
                    "Historical values must be finite."
                )

        return result

    @staticmethod
    def _validate_horizon(
        horizon: int,
    ) -> None:

        if not isinstance(
            horizon,
            int,
        ):

            raise InvalidForecastInputError(
                "horizon must be an integer."
            )

        if horizon < 1:

            raise InvalidForecastInputError(
                "horizon must be >= 1."
            )

    @staticmethod
    def _mean(
        values: Sequence[float],
    ) -> float:

        if not values:

            return 0.0

        return sum(values) / len(
            values
        )

    @staticmethod
    def _standard_deviation(
        values: Sequence[float],
    ) -> float:

        if len(values) < 2:

            return 0.0

        average = (
            BaselineForecaster._mean(
                values
            )
        )

        variance = sum(
            (
                value - average
            ) ** 2
            for value in values
        ) / len(values)

        return math.sqrt(
            variance
        )

    @staticmethod
    def _forecast_stability(
        forecast: Sequence[float],
    ) -> float:

        if len(forecast) < 2:

            return 1.0

        mean = abs(
            BaselineForecaster._mean(
                forecast
            )
        )

        if mean == 0:

            return 1.0

        deviation = (
            BaselineForecaster
            ._standard_deviation(
                forecast
            )
        )

        coefficient = (
            deviation
            / mean
        )

        return max(
            0.0,
            min(
                1.0,
                1.0 - coefficient,
            ),
        )


# ============================================================================
# Convenience API
# ============================================================================


def naive_forecast(
    values: Sequence[float],
    horizon: int,
) -> List[float]:
    """Convenience naive forecast."""

    return BaselineForecaster().naive(
        values,
        horizon,
    )


def moving_average_forecast(
    values: Sequence[float],
    horizon: int,
    window: int = 3,
) -> List[float]:
    """Convenience moving-average forecast."""

    return BaselineForecaster().moving_average(
        values,
        horizon,
        window,
    )


def exponential_smoothing_forecast(
    values: Sequence[float],
    horizon: int,
    alpha: float = 0.30,
) -> List[float]:
    """Convenience exponential-smoothing forecast."""

    return (
        BaselineForecaster()
        .exponential_smoothing(
            values,
            horizon,
            alpha,
        )
    )


def drift_forecast(
    values: Sequence[float],
    horizon: int,
) -> List[float]:
    """Convenience drift forecast."""

    return BaselineForecaster().drift(
        values,
        horizon,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "BaselineForecastError",
    "InvalidForecastInputError",
    "InsufficientDataError",
    "UnsupportedBaselineError",
    "BaselineMethod",
    "BaselineConfig",
    "BaselineForecastResult",
    "BaselineForecaster",
    "naive_forecast",
    "moving_average_forecast",
    "exponential_smoothing_forecast",
    "drift_forecast",
]