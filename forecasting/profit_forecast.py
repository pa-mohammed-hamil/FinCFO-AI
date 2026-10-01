"""
FinCo AI - Profit Forecasting
=============================

Path:
    backend/app/forecasting/profit_forecast.py

Purpose:
    Profit-specific forecasting engine built on top of the generic
    forecasting model layer.

Responsibilities:
    - Forecast revenue
    - Forecast expenses
    - Calculate projected profit
    - Calculate profit margin
    - Detect projected losses
    - Calculate profit growth
    - Analyze margin trend
    - Estimate profit risk
    - Produce structured forecast results

Architecture:

    Historical Revenue
            |
            v
    Historical Expenses
            |
            v
    profit_forecast.py
       /            \
      v              v
 Revenue Model   Expense Model
       \            /
        \          /
         v        v
        Projected Profit
              |
              v
        Margin Analysis
              |
              v
         Profit Risk
              |
              v
       Alert / Recommendation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .model import (
    FinancialForecastModel,
    ForecastModelConfig,
    ForecastModelType,
)


# ============================================================================
# Exceptions
# ============================================================================


class ProfitForecastError(Exception):
    """Base exception for profit forecasting."""


class InvalidProfitForecastInputError(ProfitForecastError):
    """Raised when profit forecast input is invalid."""


class InsufficientProfitDataError(ProfitForecastError):
    """Raised when insufficient historical profit data is supplied."""


class ProfitCalculationError(ProfitForecastError):
    """Raised when projected profit cannot be calculated."""


# ============================================================================
# Enums
# ============================================================================


class ProfitForecastMethod(str, Enum):
    """Methods available for profit forecasting."""

    DIRECT = "direct"
    REVENUE_MINUS_EXPENSE = "revenue_minus_expense"
    HISTORICAL_AVERAGE = "historical_average"
    MODEL = "model"


class ProfitRiskLevel(str, Enum):
    """Profitability risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProfitTrend(str, Enum):
    """Profit trend classification."""

    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ProfitForecastConfig:
    """
    Configuration for profit forecasting.
    """

    default_horizon: int = 6

    minimum_history: int = 3

    currency: str = "USD"

    model_type: ForecastModelType = (
        ForecastModelType.GRADIENT_BOOSTING
    )

    revenue_weight: float = 1.0

    expense_weight: float = 1.0

    loss_probability_threshold: float = 70.0

    high_risk_margin: float = 5.0

    medium_risk_margin: float = 10.0

    critical_negative_profit: bool = True

    clip_negative_revenue: bool = True

    clip_negative_expenses: bool = True

    allow_negative_profit: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""

        if self.default_horizon <= 0:
            raise InvalidProfitForecastInputError(
                "default_horizon must be > 0."
            )

        if self.minimum_history < 2:
            raise InvalidProfitForecastInputError(
                "minimum_history must be >= 2."
            )

        if not self.currency:
            raise InvalidProfitForecastInputError(
                "currency cannot be empty."
            )

        if self.revenue_weight < 0:
            raise InvalidProfitForecastInputError(
                "revenue_weight cannot be negative."
            )

        if self.expense_weight < 0:
            raise InvalidProfitForecastInputError(
                "expense_weight cannot be negative."
            )

        if not 0 <= self.loss_probability_threshold <= 100:
            raise InvalidProfitForecastInputError(
                "loss_probability_threshold must be between 0 and 100."
            )

        if self.high_risk_margin < 0:
            raise InvalidProfitForecastInputError(
                "high_risk_margin cannot be negative."
            )

        if self.medium_risk_margin < self.high_risk_margin:
            raise InvalidProfitForecastInputError(
                "medium_risk_margin must be >= high_risk_margin."
            )


# ============================================================================
# Historical Period
# ============================================================================


@dataclass
class ProfitPeriod:
    """Represents one historical or forecast period."""

    period: Any

    revenue: float

    expenses: float

    profit: Optional[float] = None

    profit_margin: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        self.revenue = float(self.revenue)
        self.expenses = float(self.expenses)

        if self.profit is None:
            self.profit = self.revenue - self.expenses
        else:
            self.profit = float(self.profit)

        if self.revenue != 0:
            self.profit_margin = (
                self.profit / self.revenue
            ) * 100.0
        else:
            self.profit_margin = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "revenue": self.revenue,
            "expenses": self.expenses,
            "profit": self.profit,
            "profit_margin": self.profit_margin,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Forecast Point
# ============================================================================


@dataclass
class ProfitForecastPoint:
    """Represents one projected profit period."""

    period_index: int

    projected_revenue: float

    projected_expenses: float

    projected_profit: float

    projected_margin: Optional[float]

    profit_growth: Optional[float]

    risk_level: ProfitRiskLevel

    loss_probability: float

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_index": self.period_index,
            "projected_revenue": self.projected_revenue,
            "projected_expenses": self.projected_expenses,
            "projected_profit": self.projected_profit,
            "projected_margin": self.projected_margin,
            "profit_growth": self.profit_growth,
            "risk_level": self.risk_level.value,
            "loss_probability": self.loss_probability,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Forecast Result
# ============================================================================


@dataclass
class ProfitForecastResult:
    """Complete profit forecasting result."""

    method: str

    horizon: int

    historical_periods: List[ProfitPeriod]

    forecast: List[ProfitForecastPoint]

    historical_profit: List[float]

    historical_margin: List[Optional[float]]

    projected_profit: List[float]

    projected_revenue: List[float]

    projected_expenses: List[float]

    projected_margin: List[Optional[float]]

    total_projected_profit: float

    average_projected_profit: float

    minimum_projected_profit: float

    maximum_projected_profit: float

    ending_projected_profit: float

    profit_growth_rate: Optional[float]

    margin_change: Optional[float]

    profit_trend: ProfitTrend

    loss_periods: int

    loss_probability: float

    risk_level: ProfitRiskLevel

    confidence: float

    created_at: datetime

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "horizon": self.horizon,
            "historical_periods": [
                period.to_dict()
                for period in self.historical_periods
            ],
            "forecast": [
                point.to_dict()
                for point in self.forecast
            ],
            "historical_profit": list(
                self.historical_profit
            ),
            "historical_margin": list(
                self.historical_margin
            ),
            "projected_profit": list(
                self.projected_profit
            ),
            "projected_revenue": list(
                self.projected_revenue
            ),
            "projected_expenses": list(
                self.projected_expenses
            ),
            "projected_margin": list(
                self.projected_margin
            ),
            "total_projected_profit": self.total_projected_profit,
            "average_projected_profit": self.average_projected_profit,
            "minimum_projected_profit": self.minimum_projected_profit,
            "maximum_projected_profit": self.maximum_projected_profit,
            "ending_projected_profit": self.ending_projected_profit,
            "profit_growth_rate": self.profit_growth_rate,
            "margin_change": self.margin_change,
            "profit_trend": self.profit_trend.value,
            "loss_periods": self.loss_periods,
            "loss_probability": self.loss_probability,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Main Profit Forecaster
# ============================================================================


class ProfitForecaster:
    """
    Profit forecasting engine.

    The engine can forecast profit in two primary ways:

    1. Direct profit forecasting:
           historical profit -> ML model -> future profit

    2. Revenue-minus-expense forecasting:
           revenue model
                 +
           expense model
                 |
                 v
           projected profit

    The second approach is generally preferable for financial systems
    because it preserves the relationship:

        Profit = Revenue - Expenses
    """

    def __init__(
        self,
        config: Optional[ProfitForecastConfig] = None,
        revenue_model: Optional[FinancialForecastModel] = None,
        expense_model: Optional[FinancialForecastModel] = None,
        profit_model: Optional[FinancialForecastModel] = None,
    ) -> None:

        self.config = config or ProfitForecastConfig()
        self.config.validate()

        self.revenue_model = revenue_model
        self.expense_model = expense_model
        self.profit_model = profit_model

    # ========================================================================
    # Main Forecast
    # ========================================================================

    def forecast(
        self,
        historical_data: Sequence[Any],
        horizon: Optional[int] = None,
        method: ProfitForecastMethod = (
            ProfitForecastMethod.REVENUE_MINUS_EXPENSE
        ),
        future_features: Optional[Any] = None,
    ) -> ProfitForecastResult:
        """
        Forecast future profit.

        Supported historical formats:

            [
                {
                    "period": "2026-01",
                    "revenue": 100000,
                    "expenses": 70000,
                },
                ...
            ]

        Or:

            [
                ProfitPeriod(...),
                ...
            ]
        """

        periods = self._normalize_periods(historical_data)

        if len(periods) < self.config.minimum_history:
            raise InsufficientProfitDataError(
                f"At least {self.config.minimum_history} "
                f"historical periods are required."
            )

        actual_horizon = (
            self.config.default_horizon
            if horizon is None
            else horizon
        )

        self._validate_horizon(actual_horizon)

        method = self._normalize_method(method)

        try:
            if method == ProfitForecastMethod.HISTORICAL_AVERAGE:
                projected_revenue, projected_expenses = (
                    self._historical_average_forecast(
                        periods,
                        actual_horizon,
                    )
                )

            elif method == ProfitForecastMethod.DIRECT:
                projected_revenue, projected_expenses = (
                    self._direct_profit_forecast(
                        periods,
                        actual_horizon,
                        future_features,
                    )
                )

            elif method == ProfitForecastMethod.MODEL:
                projected_revenue, projected_expenses = (
                    self._model_profit_forecast(
                        periods,
                        actual_horizon,
                        future_features,
                    )
                )

            else:
                projected_revenue, projected_expenses = (
                    self._revenue_minus_expense_forecast(
                        periods,
                        actual_horizon,
                        future_features,
                    )
                )

            projected_revenue = self._sanitize_revenue(
                projected_revenue
            )

            projected_expenses = self._sanitize_expenses(
                projected_expenses
            )

            projected_profit = self.calculate_projected_profit(
                projected_revenue,
                projected_expenses,
            )

            forecast = self._build_forecast_points(
                periods=periods,
                projected_revenue=projected_revenue,
                projected_expenses=projected_expenses,
                projected_profit=projected_profit,
            )

            projected_margin = [
                self.calculate_margin(
                    revenue,
                    profit,
                )
                for revenue, profit in zip(
                    projected_revenue,
                    projected_profit,
                )
            ]

            historical_profit = [
                float(period.profit)
                for period in periods
            ]

            historical_margin = [
                period.profit_margin
                for period in periods
            ]

            profit_growth = self.calculate_growth_rate(
                historical_profit,
                projected_profit,
            )

            margin_change = self.calculate_margin_change(
                historical_margin,
                projected_margin,
            )

            trend = self.assess_profit_trend(
                historical_profit,
                projected_profit,
            )

            loss_periods = sum(
                1
                for value in projected_profit
                if value < 0
            )

            loss_probability = self.calculate_loss_probability(
                projected_profit,
                historical_profit,
            )

            risk_level = self.assess_profit_risk(
                projected_profit=projected_profit,
                projected_margin=projected_margin,
                loss_probability=loss_probability,
            )

            confidence = self.estimate_confidence(
                historical_profit,
                projected_profit,
            )

            return ProfitForecastResult(
                method=method.value,
                horizon=actual_horizon,
                historical_periods=periods,
                forecast=forecast,
                historical_profit=historical_profit,
                historical_margin=historical_margin,
                projected_profit=projected_profit,
                projected_revenue=projected_revenue,
                projected_expenses=projected_expenses,
                projected_margin=projected_margin,
                total_projected_profit=float(
                    np.sum(projected_profit)
                ),
                average_projected_profit=float(
                    np.mean(projected_profit)
                ),
                minimum_projected_profit=float(
                    np.min(projected_profit)
                ),
                maximum_projected_profit=float(
                    np.max(projected_profit)
                ),
                ending_projected_profit=float(
                    projected_profit[-1]
                ),
                profit_growth_rate=profit_growth,
                margin_change=margin_change,
                profit_trend=trend,
                loss_periods=loss_periods,
                loss_probability=loss_probability,
                risk_level=risk_level,
                confidence=confidence,
                created_at=self._utc_now(),
                metadata={
                    **self.config.metadata,
                    "currency": self.config.currency,
                    "revenue_model": (
                        type(self.revenue_model.estimator).__name__
                        if self.revenue_model
                        and self.revenue_model.estimator
                        else None
                    ),
                    "expense_model": (
                        type(self.expense_model.estimator).__name__
                        if self.expense_model
                        and self.expense_model.estimator
                        else None
                    ),
                },
            )

        except ProfitForecastError:
            raise

        except Exception as exc:
            raise ProfitForecastError(
                f"Profit forecasting failed: {exc}"
            ) from exc

    # ========================================================================
    # Revenue - Expense Forecasting
    # ========================================================================

    def _revenue_minus_expense_forecast(
        self,
        periods: List[ProfitPeriod],
        horizon: int,
        future_features: Optional[Any],
    ) -> tuple[List[float], List[float]]:

        historical_revenue = [
            period.revenue
            for period in periods
        ]

        historical_expenses = [
            period.expenses
            for period in periods
        ]

        revenue_forecast = self._forecast_series(
            historical_revenue,
            horizon,
            self.revenue_model,
            future_features,
        )

        expense_forecast = self._forecast_series(
            historical_expenses,
            horizon,
            self.expense_model,
            future_features,
        )

        return revenue_forecast, expense_forecast

    # ========================================================================
    # Direct Profit Forecasting
    # ========================================================================

    def _direct_profit_forecast(
        self,
        periods: List[ProfitPeriod],
        horizon: int,
        future_features: Optional[Any],
    ) -> tuple[List[float], List[float]]:

        historical_profit = [
            float(period.profit)
            for period in periods
        ]

        if self.profit_model is None:
            self.profit_model = FinancialForecastModel(
                ForecastModelConfig(
                    model_type=self.config.model_type,
                    minimum_training_samples=(
                        self.config.minimum_history
                    ),
                )
            )

            X = self._time_features(
                len(historical_profit)
            )

            self.profit_model.fit(
                X=X,
                y=historical_profit,
                feature_names=[
                    "time_index",
                ],
                target_name="profit",
            )

        X_future = future_features

        if X_future is None:
            X_future = self._future_time_features(
                len(historical_profit),
                horizon,
            )

        projected_profit = self.profit_model.predict_batch(
            X_future
        )

        projected_revenue = self._forecast_series(
            [
                period.revenue
                for period in periods
            ],
            horizon,
            self.revenue_model,
            future_features,
        )

        projected_expenses = [
            revenue - profit
            for revenue, profit in zip(
                projected_revenue,
                projected_profit,
            )
        ]

        return projected_revenue, projected_expenses

    # ========================================================================
    # Model Forecasting
    # ========================================================================

    def _model_profit_forecast(
        self,
        periods: List[ProfitPeriod],
        horizon: int,
        future_features: Optional[Any],
    ) -> tuple[List[float], List[float]]:

        if self.profit_model is None:
            self.profit_model = FinancialForecastModel(
                ForecastModelConfig(
                    model_type=self.config.model_type,
                    minimum_training_samples=(
                        self.config.minimum_history
                    ),
                )
            )

            X = self._time_features(
                len(periods)
            )

            y = [
                period.profit
                for period in periods
            ]

            self.profit_model.fit(
                X=X,
                y=y,
                feature_names=["time_index"],
                target_name="profit",
            )

        if future_features is None:
            future_features = self._future_time_features(
                len(periods),
                horizon,
            )

        projected_profit = self.profit_model.predict_batch(
            future_features
        )

        projected_revenue = self._forecast_series(
            [period.revenue for period in periods],
            horizon,
            self.revenue_model,
            future_features,
        )

        projected_expenses = [
            revenue - profit
            for revenue, profit in zip(
                projected_revenue,
                projected_profit,
            )
        ]

        return projected_revenue, projected_expenses

    # ========================================================================
    # Historical Average
    # ========================================================================

    def _historical_average_forecast(
        self,
        periods: List[ProfitPeriod],
        horizon: int,
    ) -> tuple[List[float], List[float]]:

        revenue = float(
            np.mean(
                [period.revenue for period in periods]
            )
        )

        expenses = float(
            np.mean(
                [period.expenses for period in periods]
            )
        )

        return (
            [revenue] * horizon,
            [expenses] * horizon,
        )

    # ========================================================================
    # Generic Series Forecasting
    # ========================================================================

    def _forecast_series(
        self,
        values: Sequence[float],
        horizon: int,
        model: Optional[FinancialForecastModel],
        future_features: Optional[Any],
    ) -> List[float]:

        values_array = self._validate_series(values)

        if model is not None and model.is_fitted:
            if future_features is not None:
                predictions = model.predict_batch(
                    future_features
                )
                return [float(value) for value in predictions]

        # Conservative fallback: trend-based projection.
        return self._trend_forecast(
            values_array,
            horizon,
        )

    def _trend_forecast(
        self,
        values: np.ndarray,
        horizon: int,
    ) -> List[float]:

        if len(values) < 2:
            return [
                float(values[-1])
            ] * horizon

        x = np.arange(len(values), dtype=float)

        slope, intercept = np.polyfit(
            x,
            values,
            1,
        )

        future_x = np.arange(
            len(values),
            len(values) + horizon,
            dtype=float,
        )

        predictions = (
            intercept + slope * future_x
        )

        return [
            float(value)
            for value in predictions
        ]

    # ========================================================================
    # Profit Calculations
    # ========================================================================

    @staticmethod
    def calculate_projected_profit(
        revenue: Sequence[float],
        expenses: Sequence[float],
    ) -> List[float]:
        """
        Calculate projected profit:

            Profit = Revenue - Expenses
        """

        if len(revenue) != len(expenses):
            raise ProfitCalculationError(
                "Revenue and expenses must have equal lengths."
            )

        return [
            float(r - e)
            for r, e in zip(
                revenue,
                expenses,
            )
        ]

    @staticmethod
    def calculate_margin(
        revenue: float,
        profit: float,
    ) -> Optional[float]:
        """Calculate profit margin percentage."""

        if revenue == 0:
            return None

        return float(
            (profit / revenue) * 100.0
        )

    @staticmethod
    def calculate_margin_change(
        historical_margin: Sequence[Optional[float]],
        projected_margin: Sequence[Optional[float]],
    ) -> Optional[float]:
        """Calculate latest projected margin change."""

        historical_valid = [
            value
            for value in historical_margin
            if value is not None
        ]

        projected_valid = [
            value
            for value in projected_margin
            if value is not None
        ]

        if not historical_valid or not projected_valid:
            return None

        return float(
            projected_valid[-1]
            - historical_valid[-1]
        )

    # ========================================================================
    # Growth
    # ========================================================================

    @staticmethod
    def calculate_growth_rate(
        historical_profit: Sequence[float],
        projected_profit: Sequence[float],
    ) -> Optional[float]:
        """
        Calculate projected profit growth against the latest
        historical profit.

        Returns percentage.
        """

        if not historical_profit or not projected_profit:
            return None

        baseline = float(historical_profit[-1])
        projected = float(projected_profit[-1])

        if baseline == 0:
            if projected > 0:
                return 100.0
            if projected < 0:
                return -100.0
            return 0.0

        return float(
            ((projected - baseline) / abs(baseline))
            * 100.0
        )

    # ========================================================================
    # Trend
    # ========================================================================

    @staticmethod
    def assess_profit_trend(
        historical_profit: Sequence[float],
        projected_profit: Sequence[float],
    ) -> ProfitTrend:

        if not historical_profit or not projected_profit:
            return ProfitTrend.STABLE

        historical_slope = ProfitForecaster._slope(
            historical_profit
        )

        projected_slope = ProfitForecaster._slope(
            projected_profit
        )

        if projected_slope > 0:
            return ProfitTrend.IMPROVING

        if projected_slope < 0:
            return ProfitTrend.DECLINING

        if historical_slope > 0:
            return ProfitTrend.IMPROVING

        if historical_slope < 0:
            return ProfitTrend.DECLINING

        return ProfitTrend.STABLE

    # ========================================================================
    # Loss Probability
    # ========================================================================

    @staticmethod
    def calculate_loss_probability(
        projected_profit: Sequence[float],
        historical_profit: Sequence[float],
    ) -> float:
        """
        Estimate probability of future loss.

        This is an operational risk score, not a calibrated statistical
        probability.
        """

        if not projected_profit:
            return 0.0

        negative_periods = sum(
            1
            for value in projected_profit
            if value < 0
        )

        base_probability = (
            negative_periods
            / len(projected_profit)
        ) * 100.0

        if historical_profit:
            historical_losses = sum(
                1
                for value in historical_profit
                if value < 0
            )

            historical_rate = (
                historical_losses
                / len(historical_profit)
            ) * 100.0

            probability = (
                0.70 * base_probability
                + 0.30 * historical_rate
            )
        else:
            probability = base_probability

        return float(
            np.clip(
                probability,
                0.0,
                100.0,
            )
        )

    # ========================================================================
    # Risk
    # ========================================================================

    def assess_profit_risk(
        self,
        projected_profit: Sequence[float],
        projected_margin: Sequence[Optional[float]],
        loss_probability: float,
    ) -> ProfitRiskLevel:
        """Classify projected profitability risk."""

        if not projected_profit:
            return ProfitRiskLevel.LOW

        minimum_profit = min(projected_profit)

        valid_margins = [
            value
            for value in projected_margin
            if value is not None
        ]

        minimum_margin = (
            min(valid_margins)
            if valid_margins
            else None
        )

        if (
            self.config.critical_negative_profit
            and minimum_profit < 0
        ):
            return ProfitRiskLevel.CRITICAL

        if loss_probability >= self.config.loss_probability_threshold:
            return ProfitRiskLevel.CRITICAL

        if (
            minimum_margin is not None
            and minimum_margin < self.config.high_risk_margin
        ):
            return ProfitRiskLevel.HIGH

        if (
            minimum_margin is not None
            and minimum_margin < self.config.medium_risk_margin
        ):
            return ProfitRiskLevel.MEDIUM

        if loss_probability >= 40:
            return ProfitRiskLevel.MEDIUM

        return ProfitRiskLevel.LOW

    # ========================================================================
    # Confidence
    # ========================================================================

    @staticmethod
    def estimate_confidence(
        historical_profit: Sequence[float],
        projected_profit: Sequence[float],
    ) -> float:
        """
        Estimate operational forecast confidence based on stability.
        """

        if len(historical_profit) < 2:
            return 0.50

        historical = np.asarray(
            historical_profit,
            dtype=float,
        )

        mean_abs = float(
            np.mean(np.abs(historical))
        )

        if mean_abs <= 1e-12:
            return 0.50

        std = float(
            np.std(historical)
        )

        cv = std / mean_abs

        stability = 1.0 - min(cv, 1.0)

        confidence = (
            0.40
            + 0.50 * stability
        )

        if projected_profit:
            projected = np.asarray(
                projected_profit,
                dtype=float,
            )

            projected_mean = float(
                np.mean(np.abs(projected))
            )

            if projected_mean > 0:
                confidence += 0.05

        return float(
            np.clip(
                confidence,
                0.30,
                0.95,
            )
        )

    # ========================================================================
    # Forecast Point Construction
    # ========================================================================

    def _build_forecast_points(
        self,
        periods: List[ProfitPeriod],
        projected_revenue: Sequence[float],
        projected_expenses: Sequence[float],
        projected_profit: Sequence[float],
    ) -> List[ProfitForecastPoint]:

        historical_profit = [
            float(period.profit)
            for period in periods
        ]

        points: List[ProfitForecastPoint] = []

        previous_profit = historical_profit[-1]

        for index, (
            revenue,
            expenses,
            profit,
        ) in enumerate(
            zip(
                projected_revenue,
                projected_expenses,
                projected_profit,
            ),
            start=1,
        ):

            margin = self.calculate_margin(
                revenue,
                profit,
            )

            if previous_profit == 0:
                growth = None
            else:
                growth = float(
                    (
                        (profit - previous_profit)
                        / abs(previous_profit)
                    )
                    * 100.0
                )

            loss_probability = (
                100.0
                if profit < 0
                else 0.0
            )

            if profit < 0:
                risk = ProfitRiskLevel.CRITICAL

            elif margin is not None and margin < 5:
                risk = ProfitRiskLevel.HIGH

            elif margin is not None and margin < 10:
                risk = ProfitRiskLevel.MEDIUM

            else:
                risk = ProfitRiskLevel.LOW

            points.append(
                ProfitForecastPoint(
                    period_index=index,
                    projected_revenue=float(revenue),
                    projected_expenses=float(expenses),
                    projected_profit=float(profit),
                    projected_margin=margin,
                    profit_growth=growth,
                    risk_level=risk,
                    loss_probability=loss_probability,
                    metadata={
                        "currency": self.config.currency,
                    },
                )
            )

            previous_profit = profit

        return points

    # ========================================================================
    # Input Normalization
    # ========================================================================

    def _normalize_periods(
        self,
        historical_data: Sequence[Any],
    ) -> List[ProfitPeriod]:

        if historical_data is None:
            raise InvalidProfitForecastInputError(
                "historical_data cannot be None."
            )

        periods: List[ProfitPeriod] = []

        for index, item in enumerate(historical_data):

            if isinstance(item, ProfitPeriod):
                periods.append(item)
                continue

            if isinstance(item, Mapping):
                revenue = self._number(
                    item.get("revenue", 0.0),
                    f"revenue[{index}]",
                )

                expenses = self._number(
                    item.get("expenses", 0.0),
                    f"expenses[{index}]",
                )

                period = item.get(
                    "period",
                    index,
                )

                profit = item.get("profit")

                if profit is not None:
                    profit = self._number(
                        profit,
                        f"profit[{index}]",
                    )

                periods.append(
                    ProfitPeriod(
                        period=period,
                        revenue=revenue,
                        expenses=expenses,
                        profit=profit,
                        metadata=dict(
                            item.get(
                                "metadata",
                                {},
                            )
                        ),
                    )
                )

                continue

            if isinstance(item, Sequence) and not isinstance(
                item,
                (str, bytes),
            ):

                if len(item) < 2:
                    raise InvalidProfitForecastInputError(
                        f"Historical row {index} must contain "
                        "at least revenue and expenses."
                    )

                revenue = self._number(
                    item[0],
                    f"revenue[{index}]",
                )

                expenses = self._number(
                    item[1],
                    f"expenses[{index}]",
                )

                periods.append(
                    ProfitPeriod(
                        period=index,
                        revenue=revenue,
                        expenses=expenses,
                    )
                )

                continue

            raise InvalidProfitForecastInputError(
                f"Unsupported historical data format at index {index}."
            )

        if not periods:
            raise InvalidProfitForecastInputError(
                "historical_data cannot be empty."
            )

        return periods

    # ========================================================================
    # Sanitization
    # ========================================================================

    def _sanitize_revenue(
        self,
        values: Sequence[float],
    ) -> List[float]:

        result = []

        for value in values:
            value = float(value)

            if not isfinite(value):
                raise InvalidProfitForecastInputError(
                    "Revenue forecast contains invalid values."
                )

            if self.config.clip_negative_revenue:
                value = max(0.0, value)

            result.append(value)

        return result

    def _sanitize_expenses(
        self,
        values: Sequence[float],
    ) -> List[float]:

        result = []

        for value in values:
            value = float(value)

            if not isfinite(value):
                raise InvalidProfitForecastInputError(
                    "Expense forecast contains invalid values."
                )

            if self.config.clip_negative_expenses:
                value = max(0.0, value)

            result.append(value)

        return result

    # ========================================================================
    # Time Features
    # ========================================================================

    @staticmethod
    def _time_features(
        count: int,
    ) -> np.ndarray:

        return np.arange(
            count,
            dtype=float,
        ).reshape(-1, 1)

    @staticmethod
    def _future_time_features(
        history_count: int,
        horizon: int,
    ) -> np.ndarray:

        return np.arange(
            history_count,
            history_count + horizon,
            dtype=float,
        ).reshape(-1, 1)

    # ========================================================================
    # Utilities
    # ========================================================================

    @staticmethod
    def _validate_series(
        values: Sequence[float],
    ) -> np.ndarray:

        if values is None:
            raise InvalidProfitForecastInputError(
                "Series cannot be None."
            )

        try:
            array = np.asarray(
                values,
                dtype=float,
            ).reshape(-1)
        except Exception as exc:
            raise InvalidProfitForecastInputError(
                "Series must contain numeric values."
            ) from exc

        if len(array) == 0:
            raise InvalidProfitForecastInputError(
                "Series cannot be empty."
            )

        if not np.all(np.isfinite(array)):
            raise InvalidProfitForecastInputError(
                "Series contains NaN or infinite values."
            )

        return array

    @staticmethod
    def _slope(
        values: Sequence[float],
    ) -> float:

        if len(values) < 2:
            return 0.0

        array = np.asarray(
            values,
            dtype=float,
        )

        x = np.arange(
            len(array),
            dtype=float,
        )

        slope, _ = np.polyfit(
            x,
            array,
            1,
        )

        return float(slope)

    @staticmethod
    def _number(
        value: Any,
        field_name: str,
    ) -> float:

        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidProfitForecastInputError(
                f"{field_name} must be numeric."
            ) from exc

        if not isfinite(number):
            raise InvalidProfitForecastInputError(
                f"{field_name} must be finite."
            )

        return number

    @staticmethod
    def _validate_horizon(
        horizon: int,
    ) -> None:

        if not isinstance(horizon, int):
            raise InvalidProfitForecastInputError(
                "horizon must be an integer."
            )

        if horizon <= 0:
            raise InvalidProfitForecastInputError(
                "horizon must be > 0."
            )

    @staticmethod
    def _normalize_method(
        method: ProfitForecastMethod,
    ) -> ProfitForecastMethod:

        if isinstance(method, ProfitForecastMethod):
            return method

        try:
            return ProfitForecastMethod(str(method))
        except ValueError as exc:
            raise InvalidProfitForecastInputError(
                f"Unsupported profit forecast method: {method}"
            ) from exc

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)


# ============================================================================
# Convenience Functions
# ============================================================================


def forecast_profit(
    historical_data: Sequence[Any],
    horizon: int = 6,
    method: ProfitForecastMethod = (
        ProfitForecastMethod.REVENUE_MINUS_EXPENSE
    ),
    config: Optional[ProfitForecastConfig] = None,
) -> ProfitForecastResult:
    """
    Convenience function for profit forecasting.
    """

    forecaster = ProfitForecaster(
        config=config
    )

    return forecaster.forecast(
        historical_data=historical_data,
        horizon=horizon,
        method=method,
    )


def calculate_profit(
    revenue: Sequence[float],
    expenses: Sequence[float],
) -> List[float]:
    """Convenience profit calculation."""

    return ProfitForecaster.calculate_projected_profit(
        revenue,
        expenses,
    )


def calculate_profit_margin(
    revenue: float,
    profit: float,
) -> Optional[float]:
    """Convenience margin calculation."""

    return ProfitForecaster.calculate_margin(
        revenue,
        profit,
    )


def calculate_loss_probability(
    projected_profit: Sequence[float],
    historical_profit: Sequence[float],
) -> float:
    """Convenience loss probability calculation."""

    return ProfitForecaster.calculate_loss_probability(
        projected_profit,
        historical_profit,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "ProfitForecastError",
    "InvalidProfitForecastInputError",
    "InsufficientProfitDataError",
    "ProfitCalculationError",

    # Enums
    "ProfitForecastMethod",
    "ProfitRiskLevel",
    "ProfitTrend",

    # Configuration
    "ProfitForecastConfig",

    # Data classes
    "ProfitPeriod",
    "ProfitForecastPoint",
    "ProfitForecastResult",

    # Main engine
    "ProfitForecaster",

    # Convenience functions
    "forecast_profit",
    "calculate_profit",
    "calculate_profit_margin",
    "calculate_loss_probability",
]