"""
FinCo AI - Revenue Forecasting Engine

File:
    backend/app/forecasting/revenue_forecast.py

Purpose:
    Specialized revenue forecasting service built on top of the generic
    forecasting preprocessing and baseline layers.

Responsibilities:
    - Revenue history validation
    - Revenue forecasting
    - Multiple baseline forecasting methods
    - Revenue growth analysis
    - Revenue trend analysis
    - Revenue target comparison
    - Forecast confidence estimation
    - Revenue risk assessment
    - Forecast result serialization

Architecture:

    Raw Revenue Data
            |
            v
    preprocessing.py
            |
            v
    revenue_forecast.py
            |
       +----+----------------+
       |                     |
       v                     v
    baseline.py          model.py
       |                     |
       +----------+----------+
                  |
                  v
          Revenue Forecast
                  |
        +---------+---------+
        |         |         |
        v         v         v
      Growth    Trend      Risk
        |         |         |
        +---------+---------+
                  |
                  v
             Alerts /
          Recommendations /
          Supervisor Agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from statistics import mean, pstdev
from typing import Any, Mapping, Optional, Sequence

try:
    from .baseline import (
        BaselineForecaster,
        BaselineMethod,
        BaselineForecastResult,
    )
except ImportError:
    from baseline import (  # type: ignore
        BaselineForecaster,
        BaselineMethod,
        BaselineForecastResult,
    )


# ============================================================================
# Exceptions
# ============================================================================


class RevenueForecastError(Exception):
    """Base exception for revenue forecasting errors."""


class InvalidRevenueInputError(RevenueForecastError):
    """Raised when revenue input data is invalid."""


class InsufficientRevenueDataError(RevenueForecastError):
    """Raised when insufficient historical revenue data is available."""


class RevenueCalculationError(RevenueForecastError):
    """Raised when a revenue calculation fails."""


class UnsupportedRevenueMethodError(RevenueForecastError):
    """Raised when an unsupported forecasting method is requested."""


# ============================================================================
# Enums
# ============================================================================


class RevenueForecastMethod(str, Enum):
    """Supported revenue forecasting methods."""

    NAIVE = "naive"
    SEASONAL_NAIVE = "seasonal_naive"
    MEAN = "mean"
    MOVING_AVERAGE = "moving_average"
    WEIGHTED_MOVING_AVERAGE = "weighted_moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    DRIFT = "drift"


class RevenueRiskLevel(str, Enum):
    """Revenue forecast risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RevenueTrend(str, Enum):
    """Revenue trend classification."""

    STRONGLY_GROWING = "strongly_growing"
    GROWING = "growing"
    STABLE = "stable"
    DECLINING = "declining"
    STRONGLY_DECLINING = "strongly_declining"


# ============================================================================
# Configuration
# ============================================================================


@dataclass(slots=True)
class RevenueForecastConfig:
    """Configuration for the revenue forecasting engine."""

    default_horizon: int = 6
    minimum_history: int = 3

    moving_average_window: int = 3
    weighted_window: int = 3
    exponential_alpha: float = 0.30
    seasonal_period: int = 12

    currency: str = "USD"

    allow_negative_revenue: bool = False
    clip_negative_forecast: bool = True

    growth_warning_threshold: float = -5.0
    growth_critical_threshold: float = -15.0

    forecast_decline_warning_threshold: float = -5.0
    forecast_decline_critical_threshold: float = -15.0

    volatility_warning_threshold: float = 20.0
    volatility_critical_threshold: float = 35.0

    confidence_high_threshold: float = 80.0
    confidence_medium_threshold: float = 60.0

    def validate(self) -> None:
        """Validate configuration values."""

        if self.default_horizon <= 0:
            raise ValueError("default_horizon must be greater than zero.")

        if self.minimum_history < 2:
            raise ValueError("minimum_history must be at least 2.")

        if self.moving_average_window <= 0:
            raise ValueError("moving_average_window must be greater than zero.")

        if self.weighted_window <= 0:
            raise ValueError("weighted_window must be greater than zero.")

        if not 0 < self.exponential_alpha <= 1:
            raise ValueError("exponential_alpha must be in (0, 1].")

        if self.seasonal_period <= 0:
            raise ValueError("seasonal_period must be greater than zero.")

        if not self.currency.strip():
            raise ValueError("currency must not be empty.")


# ============================================================================
# Data Models
# ============================================================================


@dataclass(slots=True)
class RevenuePeriod:
    """Normalized historical revenue period."""

    period: Any
    revenue: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period.isoformat()
            if isinstance(self.period, datetime)
            else self.period,
            "revenue": self.revenue,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RevenueForecastPoint:
    """Single projected revenue period."""

    period_index: int
    projected_revenue: float

    growth_rate: Optional[float] = None
    cumulative_revenue: Optional[float] = None
    confidence: Optional[float] = None

    risk_level: RevenueRiskLevel = RevenueRiskLevel.LOW

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_index": self.period_index,
            "projected_revenue": self.projected_revenue,
            "growth_rate": self.growth_rate,
            "cumulative_revenue": self.cumulative_revenue,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RevenueForecastResult:
    """Complete revenue forecasting result."""

    method: str
    horizon: int

    historical_periods: list[RevenuePeriod]
    forecast: list[RevenueForecastPoint]

    historical_revenue: list[float]

    projected_revenue: list[float]
    total_projected_revenue: float

    average_historical_revenue: float
    average_projected_revenue: float

    last_historical_revenue: float

    historical_growth_rate: Optional[float]
    projected_growth_rate: Optional[float]

    historical_trend: RevenueTrend
    projected_trend: RevenueTrend

    volatility: float

    revenue_risk: RevenueRiskLevel
    confidence: float

    revenue_gap: Optional[float] = None
    revenue_gap_percent: Optional[float] = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    currency: str = "USD"

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""

        return {
            "method": self.method,
            "horizon": self.horizon,
            "historical_periods": [
                period.to_dict() for period in self.historical_periods
            ],
            "forecast": [
                point.to_dict() for point in self.forecast
            ],
            "historical_revenue": list(self.historical_revenue),
            "projected_revenue": list(self.projected_revenue),
            "total_projected_revenue": self.total_projected_revenue,
            "average_historical_revenue": self.average_historical_revenue,
            "average_projected_revenue": self.average_projected_revenue,
            "last_historical_revenue": self.last_historical_revenue,
            "historical_growth_rate": self.historical_growth_rate,
            "projected_growth_rate": self.projected_growth_rate,
            "historical_trend": self.historical_trend.value,
            "projected_trend": self.projected_trend.value,
            "volatility": self.volatility,
            "revenue_risk": self.revenue_risk.value,
            "confidence": self.confidence,
            "revenue_gap": self.revenue_gap,
            "revenue_gap_percent": self.revenue_gap_percent,
            "created_at": self.created_at.isoformat(),
            "currency": self.currency,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Revenue Forecaster
# ============================================================================


class RevenueForecaster:
    """
    Specialized revenue forecasting service.

    The class intentionally keeps forecasting logic separate from:

        - preprocessing.py -> cleaning and normalization
        - baseline.py -> generic forecasting algorithms
        - model.py -> advanced ML models
        - evaluation.py -> model evaluation

    This module focuses on revenue-specific business interpretation.
    """

    def __init__(
        self,
        baseline_forecaster: Optional[BaselineForecaster] = None,
        config: Optional[RevenueForecastConfig] = None,
    ) -> None:
        self.config = config or RevenueForecastConfig()
        self.config.validate()

        self.baseline_forecaster = (
            baseline_forecaster or BaselineForecaster()
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def forecast(
        self,
        historical_data: Sequence[Any],
        horizon: Optional[int] = None,
        method: RevenueForecastMethod | str = RevenueForecastMethod.EXPONENTIAL_SMOOTHING,
        revenue_field: str = "revenue",
        target_revenue: Optional[float] = None,
    ) -> RevenueForecastResult:
        """
        Generate a revenue forecast.

        Supported input examples:

            [100000, 110000, 120000, 130000]

        or:

            [
                {"period": "2026-01", "revenue": 100000},
                {"period": "2026-02", "revenue": 110000},
            ]
        """

        normalized_method = self._normalize_method(method)
        forecast_horizon = (
            horizon
            if horizon is not None
            else self.config.default_horizon
        )

        self._validate_horizon(forecast_horizon)

        periods = self._normalize_periods(
            historical_data,
            revenue_field=revenue_field,
        )

        self._validate_history(periods)

        historical_values = [
            period.revenue
            for period in periods
        ]

        forecast_values = self._forecast_series(
            historical_values,
            horizon=forecast_horizon,
            method=normalized_method,
        )

        forecast_values = [
            self._sanitize_forecast_value(value)
            for value in forecast_values
        ]

        historical_growth = self.calculate_growth_rate(
            historical_values
        )

        projected_growth = self.calculate_projected_growth(
            historical_values,
            forecast_values,
        )

        historical_trend = self.calculate_trend(
            historical_values
        )

        projected_trend = self.calculate_projected_trend(
            historical_values,
            forecast_values,
        )

        volatility = self.calculate_volatility(
            historical_values
        )

        risk = self.assess_revenue_risk(
            historical_growth=historical_growth,
            projected_growth=projected_growth,
            volatility=volatility,
        )

        confidence = self.estimate_confidence(
            historical_values,
            forecast_values,
            volatility=volatility,
        )

        forecast_points = self._build_forecast_points(
            historical_values=historical_values,
            forecast_values=forecast_values,
            confidence=confidence,
            risk=risk,
        )

        revenue_gap = None
        revenue_gap_percent = None

        if target_revenue is not None:
            target = self._number(target_revenue)

            revenue_gap = (
                sum(forecast_values) - target
            )

            if target != 0:
                revenue_gap_percent = (
                    revenue_gap / abs(target)
                ) * 100.0

        return RevenueForecastResult(
            method=normalized_method.value,
            horizon=forecast_horizon,
            historical_periods=periods,
            forecast=forecast_points,
            historical_revenue=list(historical_values),
            projected_revenue=list(forecast_values),
            total_projected_revenue=sum(forecast_values),
            average_historical_revenue=mean(historical_values),
            average_projected_revenue=mean(forecast_values),
            last_historical_revenue=historical_values[-1],
            historical_growth_rate=historical_growth,
            projected_growth_rate=projected_growth,
            historical_trend=historical_trend,
            projected_trend=projected_trend,
            volatility=volatility,
            revenue_risk=risk,
            confidence=confidence,
            revenue_gap=revenue_gap,
            revenue_gap_percent=revenue_gap_percent,
            currency=self.config.currency,
            metadata={
                "forecast_engine": "RevenueForecaster",
                "baseline_engine": type(
                    self.baseline_forecaster
                ).__name__,
                "target_revenue_provided": target_revenue is not None,
            },
        )

    # ========================================================================
    # Forecasting Methods
    # ========================================================================

    def forecast_revenue(
        self,
        historical_data: Sequence[Any],
        horizon: Optional[int] = None,
        method: RevenueForecastMethod | str = RevenueForecastMethod.EXPONENTIAL_SMOOTHING,
        revenue_field: str = "revenue",
    ) -> RevenueForecastResult:
        """Alias for forecast()."""

        return self.forecast(
            historical_data=historical_data,
            horizon=horizon,
            method=method,
            revenue_field=revenue_field,
        )

    def forecast_naive(
        self,
        values: Sequence[float],
        horizon: int,
    ) -> list[float]:
        """Naive revenue forecast."""

        return self._forecast_series(
            values,
            horizon,
            RevenueForecastMethod.NAIVE,
        )

    def forecast_moving_average(
        self,
        values: Sequence[float],
        horizon: int,
    ) -> list[float]:
        """Moving-average revenue forecast."""

        return self._forecast_series(
            values,
            horizon,
            RevenueForecastMethod.MOVING_AVERAGE,
        )

    def forecast_exponential_smoothing(
        self,
        values: Sequence[float],
        horizon: int,
    ) -> list[float]:
        """Exponential-smoothing revenue forecast."""

        return self._forecast_series(
            values,
            horizon,
            RevenueForecastMethod.EXPONENTIAL_SMOOTHING,
        )

    def forecast_drift(
        self,
        values: Sequence[float],
        horizon: int,
    ) -> list[float]:
        """Drift-based revenue forecast."""

        return self._forecast_series(
            values,
            horizon,
            RevenueForecastMethod.DRIFT,
        )

    # ========================================================================
    # Revenue Analytics
    # ========================================================================

    def calculate_growth_rate(
        self,
        values: Sequence[float],
    ) -> Optional[float]:
        """
        Calculate growth between the first and last observations.

        Returns percentage.
        """

        if len(values) < 2:
            return None

        first = self._number(values[0])
        last = self._number(values[-1])

        if first == 0:
            return None

        return ((last - first) / abs(first)) * 100.0

    def calculate_period_growth_rates(
        self,
        values: Sequence[float],
    ) -> list[Optional[float]]:
        """Calculate period-over-period growth rates."""

        normalized = [
            self._number(value)
            for value in values
        ]

        result: list[Optional[float]] = [None]

        for previous, current in zip(
            normalized,
            normalized[1:],
        ):
            if previous == 0:
                result.append(None)
            else:
                result.append(
                    ((current - previous) / abs(previous))
                    * 100.0
                )

        return result

    def calculate_projected_growth(
        self,
        historical_values: Sequence[float],
        forecast_values: Sequence[float],
    ) -> Optional[float]:
        """Calculate growth from latest historical revenue to final forecast."""

        if not historical_values or not forecast_values:
            return None

        latest = self._number(historical_values[-1])
        final = self._number(forecast_values[-1])

        if latest == 0:
            return None

        return ((final - latest) / abs(latest)) * 100.0

    def calculate_trend(
        self,
        values: Sequence[float],
    ) -> RevenueTrend:
        """Classify historical revenue trend."""

        growth = self.calculate_growth_rate(values)

        if growth is None:
            return RevenueTrend.STABLE

        if growth >= 15:
            return RevenueTrend.STRONGLY_GROWING

        if growth >= 5:
            return RevenueTrend.GROWING

        if growth > -5:
            return RevenueTrend.STABLE

        if growth > -15:
            return RevenueTrend.DECLINING

        return RevenueTrend.STRONGLY_DECLINING

    def calculate_projected_trend(
        self,
        historical_values: Sequence[float],
        forecast_values: Sequence[float],
    ) -> RevenueTrend:
        """Classify projected revenue trend."""

        growth = self.calculate_projected_growth(
            historical_values,
            forecast_values,
        )

        if growth is None:
            return RevenueTrend.STABLE

        if growth >= 15:
            return RevenueTrend.STRONGLY_GROWING

        if growth >= 5:
            return RevenueTrend.GROWING

        if growth > -5:
            return RevenueTrend.STABLE

        if growth > -15:
            return RevenueTrend.DECLINING

        return RevenueTrend.STRONGLY_DECLINING

    def calculate_volatility(
        self,
        values: Sequence[float],
    ) -> float:
        """
        Calculate revenue volatility using coefficient of variation.

        Returns percentage.
        """

        normalized = [
            self._number(value)
            for value in values
        ]

        if len(normalized) < 2:
            return 0.0

        average = mean(normalized)

        if average == 0:
            return 0.0

        volatility = (
            pstdev(normalized) / abs(average)
        ) * 100.0

        return max(0.0, volatility)

    def calculate_average_revenue(
        self,
        values: Sequence[float],
    ) -> float:
        """Calculate average revenue."""

        if not values:
            raise InvalidRevenueInputError(
                "Revenue history cannot be empty."
            )

        return mean(
            self._number(value)
            for value in values
        )

    def calculate_total_revenue(
        self,
        values: Sequence[float],
    ) -> float:
        """Calculate total revenue."""

        return sum(
            self._number(value)
            for value in values
        )

    # ========================================================================
    # Revenue Risk
    # ========================================================================

    def assess_revenue_risk(
        self,
        historical_growth: Optional[float],
        projected_growth: Optional[float],
        volatility: float,
    ) -> RevenueRiskLevel:
        """
        Determine revenue risk.

        Risk is based on:
            - historical deterioration
            - projected deterioration
            - revenue volatility
        """

        score = 0.0

        if historical_growth is not None:
            if historical_growth <= self.config.growth_critical_threshold:
                score += 40.0
            elif historical_growth <= self.config.growth_warning_threshold:
                score += 20.0

        if projected_growth is not None:
            if (
                projected_growth
                <= self.config.forecast_decline_critical_threshold
            ):
                score += 40.0
            elif (
                projected_growth
                <= self.config.forecast_decline_warning_threshold
            ):
                score += 20.0

        if volatility >= self.config.volatility_critical_threshold:
            score += 30.0
        elif volatility >= self.config.volatility_warning_threshold:
            score += 15.0

        if score >= 70:
            return RevenueRiskLevel.CRITICAL

        if score >= 40:
            return RevenueRiskLevel.HIGH

        if score >= 20:
            return RevenueRiskLevel.MEDIUM

        return RevenueRiskLevel.LOW

    # ========================================================================
    # Forecast Confidence
    # ========================================================================

    def estimate_confidence(
        self,
        historical_values: Sequence[float],
        forecast_values: Sequence[float],
        volatility: Optional[float] = None,
    ) -> float:
        """
        Estimate forecast confidence.

        This is a heuristic confidence indicator, not a calibrated
        statistical prediction interval.
        """

        if len(historical_values) < 2:
            return 20.0

        if volatility is None:
            volatility = self.calculate_volatility(
                historical_values
            )

        confidence = 90.0

        # Penalize unstable historical revenue.
        if volatility >= 50:
            confidence -= 40
        elif volatility >= 35:
            confidence -= 25
        elif volatility >= 20:
            confidence -= 15
        elif volatility >= 10:
            confidence -= 5

        # Penalize very short histories.
        if len(historical_values) < 4:
            confidence -= 15
        elif len(historical_values) < 6:
            confidence -= 5

        # Penalize extreme forecast movement.
        if forecast_values:
            last_actual = historical_values[-1]
            final_forecast = forecast_values[-1]

            if last_actual != 0:
                movement = (
                    abs(final_forecast - last_actual)
                    / abs(last_actual)
                ) * 100

                if movement >= 50:
                    confidence -= 20
                elif movement >= 30:
                    confidence -= 10

        return max(0.0, min(100.0, confidence))

    # ========================================================================
    # Target Analysis
    # ========================================================================

    def compare_with_target(
        self,
        forecast_values: Sequence[float],
        target_revenue: float,
    ) -> dict[str, Any]:
        """
        Compare forecast revenue against a target.

        Returns:
            target
            forecast_total
            gap
            gap_percent
            achieved
        """

        target = self._number(target_revenue)
        total = self.calculate_total_revenue(
            forecast_values
        )

        gap = total - target

        gap_percent = None

        if target != 0:
            gap_percent = (
                gap / abs(target)
            ) * 100.0

        return {
            "target_revenue": target,
            "forecast_total": total,
            "gap": gap,
            "gap_percent": gap_percent,
            "achieved": total >= target,
        }

    # ========================================================================
    # Baseline Selection
    # ========================================================================

    def select_best_baseline(
        self,
        values: Sequence[float],
        validation_horizon: int = 1,
    ) -> dict[str, Any]:
        """
        Select the best baseline model using holdout MAE.

        This delegates model comparison to BaselineForecaster.
        """

        normalized = [
            self._number(value)
            for value in values
        ]

        self._validate_history_values(normalized)

        try:
            result = self.baseline_forecaster.select_baseline(
                normalized,
                horizon=validation_horizon,
            )
        except TypeError:
            # Compatibility with alternate baseline implementations.
            result = self.baseline_forecaster.select_baseline(
                normalized,
                validation_horizon,
            )

        if isinstance(result, dict):
            return result

        return {
            "selected_method": getattr(
                result,
                "method",
                None,
            ),
            "result": result,
        }

    # ========================================================================
    # Forecast Point Construction
    # ========================================================================

    def _build_forecast_points(
        self,
        historical_values: Sequence[float],
        forecast_values: Sequence[float],
        confidence: float,
        risk: RevenueRiskLevel,
    ) -> list[RevenueForecastPoint]:
        """Build structured forecast points."""

        points: list[RevenueForecastPoint] = []

        previous = (
            historical_values[-1]
            if historical_values
            else None
        )

        cumulative = 0.0

        for index, value in enumerate(
            forecast_values,
            start=1,
        ):
            growth = None

            if previous is not None and previous != 0:
                growth = (
                    (value - previous)
                    / abs(previous)
                ) * 100.0

            cumulative += value

            # Confidence gradually decreases for distant periods.
            point_confidence = max(
                0.0,
                confidence - max(0, index - 1) * 2.0,
            )

            points.append(
                RevenueForecastPoint(
                    period_index=index,
                    projected_revenue=value,
                    growth_rate=growth,
                    cumulative_revenue=cumulative,
                    confidence=point_confidence,
                    risk_level=risk,
                )
            )

            previous = value

        return points

    # ========================================================================
    # Input Normalization
    # ========================================================================

    def _normalize_periods(
        self,
        data: Sequence[Any],
        revenue_field: str,
    ) -> list[RevenuePeriod]:
        """Normalize supported revenue input formats."""

        if not data:
            raise InvalidRevenueInputError(
                "Revenue history cannot be empty."
            )

        periods: list[RevenuePeriod] = []

        for index, item in enumerate(data):
            if isinstance(item, Mapping):
                if revenue_field not in item:
                    # Support common aliases.
                    aliases = (
                        "revenue",
                        "amount",
                        "value",
                        "sales",
                        "total_revenue",
                    )

                    found = next(
                        (
                            alias
                            for alias in aliases
                            if alias in item
                        ),
                        None,
                    )

                    if found is None:
                        raise InvalidRevenueInputError(
                            f"Missing revenue field '{revenue_field}' "
                            f"at index {index}."
                        )

                    revenue = item[found]
                else:
                    revenue = item[revenue_field]

                period = item.get(
                    "period",
                    item.get(
                        "date",
                        item.get(
                            "timestamp",
                            index,
                        ),
                    ),
                )

                metadata = dict(
                    item.get("metadata", {})
                    if isinstance(
                        item.get("metadata", {}),
                        Mapping,
                    )
                    else {}
                )

                # Preserve useful dimensions.
                for key in (
                    "currency",
                    "region",
                    "country",
                    "business_unit",
                    "product",
                    "category",
                    "channel",
                ):
                    if key in item:
                        metadata[key] = item[key]

            elif isinstance(item, (int, float)):
                period = index
                revenue = item
                metadata = {}

            else:
                raise InvalidRevenueInputError(
                    f"Unsupported revenue record type at "
                    f"index {index}: {type(item).__name__}"
                )

            numeric_revenue = self._number(
                revenue
            )

            if (
                not self.config.allow_negative_revenue
                and numeric_revenue < 0
            ):
                raise InvalidRevenueInputError(
                    f"Negative revenue is not allowed: "
                    f"{numeric_revenue}"
                )

            periods.append(
                RevenuePeriod(
                    period=period,
                    revenue=numeric_revenue,
                    metadata=metadata,
                )
            )

        return periods

    # ========================================================================
    # Baseline Integration
    # ========================================================================

    def _forecast_series(
        self,
        values: Sequence[float],
        horizon: int,
        method: RevenueForecastMethod,
    ) -> list[float]:
        """Run the selected baseline forecasting algorithm."""

        normalized_values = [
            self._number(value)
            for value in values
        ]

        self._validate_history_values(
            normalized_values
        )

        try:
            baseline_method = BaselineMethod(
                method.value
            )
        except ValueError as exc:
            raise UnsupportedRevenueMethodError(
                f"Unsupported revenue forecasting method: "
                f"{method}"
            ) from exc

        try:
            result: BaselineForecastResult = (
                self.baseline_forecaster.forecast(
                    normalized_values,
                    horizon=horizon,
                    method=baseline_method,
                )
            )

            forecast = list(result.forecast)

        except Exception as exc:
            raise RevenueCalculationError(
                f"Failed to generate revenue forecast: {exc}"
            ) from exc

        return forecast

    # ========================================================================
    # Validation
    # ========================================================================

    def _validate_history(
        self,
        periods: Sequence[RevenuePeriod],
    ) -> None:
        """Validate normalized revenue history."""

        if len(periods) < self.config.minimum_history:
            raise InsufficientRevenueDataError(
                f"At least {self.config.minimum_history} "
                f"historical periods are required; "
                f"received {len(periods)}."
            )

        self._validate_history_values(
            [period.revenue for period in periods]
        )

    def _validate_history_values(
        self,
        values: Sequence[float],
    ) -> None:
        """Validate numerical revenue history."""

        if not values:
            raise InvalidRevenueInputError(
                "Revenue values cannot be empty."
            )

        for value in values:
            numeric = self._number(value)

            if (
                not self.config.allow_negative_revenue
                and numeric < 0
            ):
                raise InvalidRevenueInputError(
                    "Revenue values cannot be negative."
                )

    def _validate_horizon(
        self,
        horizon: int,
    ) -> None:
        """Validate forecast horizon."""

        if not isinstance(horizon, int):
            raise InvalidRevenueInputError(
                "Forecast horizon must be an integer."
            )

        if horizon <= 0:
            raise InvalidRevenueInputError(
                "Forecast horizon must be greater than zero."
            )

        if horizon > 120:
            raise InvalidRevenueInputError(
                "Forecast horizon cannot exceed 120 periods."
            )

    # ========================================================================
    # Helpers
    # ========================================================================

    def _normalize_method(
        self,
        method: RevenueForecastMethod | str,
    ) -> RevenueForecastMethod:
        """Normalize forecast method."""

        if isinstance(method, RevenueForecastMethod):
            return method

        try:
            return RevenueForecastMethod(
                str(method).strip().lower()
            )
        except ValueError as exc:
            supported = ", ".join(
                item.value
                for item in RevenueForecastMethod
            )

            raise UnsupportedRevenueMethodError(
                f"Unsupported method '{method}'. "
                f"Supported methods: {supported}"
            ) from exc

    def _sanitize_forecast_value(
        self,
        value: float,
    ) -> float:
        """Sanitize a forecast value."""

        numeric = self._number(value)

        if (
            self.config.clip_negative_forecast
            and not self.config.allow_negative_revenue
        ):
            numeric = max(0.0, numeric)

        return numeric

    @staticmethod
    def _number(value: Any) -> float:
        """Convert a value to a finite float."""

        if isinstance(value, bool):
            raise InvalidRevenueInputError(
                "Boolean values are not valid revenue values."
            )

        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidRevenueInputError(
                f"Invalid revenue value: {value!r}"
            ) from exc

        if not isfinite(numeric):
            raise InvalidRevenueInputError(
                f"Revenue value must be finite: {value!r}"
            )

        return numeric


# ============================================================================
# Convenience Functions
# ============================================================================


def forecast_revenue(
    historical_data: Sequence[Any],
    horizon: int = 6,
    method: RevenueForecastMethod | str = RevenueForecastMethod.EXPONENTIAL_SMOOTHING,
    config: Optional[RevenueForecastConfig] = None,
    revenue_field: str = "revenue",
) -> RevenueForecastResult:
    """
    Convenience function for revenue forecasting.
    """

    forecaster = RevenueForecaster(
        config=config
    )

    return forecaster.forecast(
        historical_data=historical_data,
        horizon=horizon,
        method=method,
        revenue_field=revenue_field,
    )


def calculate_revenue_growth(
    values: Sequence[float],
) -> Optional[float]:
    """Convenience function for revenue growth."""

    forecaster = RevenueForecaster()

    return forecaster.calculate_growth_rate(
        values
    )


def calculate_revenue_volatility(
    values: Sequence[float],
) -> float:
    """Convenience function for revenue volatility."""

    forecaster = RevenueForecaster()

    return forecaster.calculate_volatility(
        values
    )


def assess_revenue_risk(
    historical_growth: Optional[float],
    projected_growth: Optional[float],
    volatility: float,
    config: Optional[RevenueForecastConfig] = None,
) -> RevenueRiskLevel:
    """Convenience function for revenue risk assessment."""

    forecaster = RevenueForecaster(
        config=config
    )

    return forecaster.assess_revenue_risk(
        historical_growth=historical_growth,
        projected_growth=projected_growth,
        volatility=volatility,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "RevenueForecastError",
    "InvalidRevenueInputError",
    "InsufficientRevenueDataError",
    "RevenueCalculationError",
    "UnsupportedRevenueMethodError",

    # Enums
    "RevenueForecastMethod",
    "RevenueRiskLevel",
    "RevenueTrend",

    # Configuration
    "RevenueForecastConfig",

    # Data models
    "RevenuePeriod",
    "RevenueForecastPoint",
    "RevenueForecastResult",

    # Main service
    "RevenueForecaster",

    # Convenience functions
    "forecast_revenue",
    "calculate_revenue_growth",
    "calculate_revenue_volatility",
    "assess_revenue_risk",
]