"""
FinCo AI - Cash Flow Forecasting

File:
    backend/app/forecasting/cashflow_forecast.py

Purpose:
    Forecast future cash flow using historical inflows, outflows,
    net cash flow, and optional business drivers.

Architecture:

    Historical Cash Flow
            |
            v
    CashFlowForecaster
            |
       +----+----+
       |         |
       v         v
    Inflows    Outflows
       |         |
       +----+----+
            |
            v
       Net Cash Flow
            |
            v
      Liquidity Analysis
            |
            v
       Forecast Result

This module is intentionally model-agnostic. Advanced ML models
can be plugged into the forecasting/model.py layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence
import math


# ============================================================================
# Exceptions
# ============================================================================


class CashFlowForecastError(Exception):
    """Base exception for cash-flow forecasting."""


class InvalidCashFlowInputError(
    CashFlowForecastError
):
    """Raised when cash-flow input is invalid."""


class InsufficientCashFlowDataError(
    CashFlowForecastError
):
    """Raised when insufficient history is available."""


class CashFlowCalculationError(
    CashFlowForecastError
):
    """Raised when cash-flow calculations fail."""


# ============================================================================
# Constants / Enums
# ============================================================================


class CashFlowType(str, Enum):
    """Supported cash-flow categories."""

    INFLOW = "inflow"
    OUTFLOW = "outflow"
    NET = "net"


class CashFlowMethod(str, Enum):
    """Forecasting methods supported by this module."""

    NAIVE = "naive"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    DRIFT = "drift"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class CashFlowForecastConfig:
    """
    Configuration for cash-flow forecasting.
    """

    default_horizon: int = 6

    moving_average_window: int = 3

    exponential_alpha: float = 0.30

    minimum_history: int = 3

    opening_cash_balance: float = 0.0

    currency: str = "USD"

    include_inflows: bool = True

    include_outflows: bool = True

    clip_negative_inflows: bool = True

    clip_negative_outflows: bool = True

    allow_negative_net_cash_flow: bool = True

    def validate(self) -> None:
        """Validate forecasting configuration."""

        if self.default_horizon < 1:
            raise ValueError(
                "default_horizon must be >= 1."
            )

        if self.moving_average_window < 1:
            raise ValueError(
                "moving_average_window must be >= 1."
            )

        if not (
            0.0
            < self.exponential_alpha
            <= 1.0
        ):
            raise ValueError(
                "exponential_alpha must be > 0 and <= 1."
            )

        if self.minimum_history < 1:
            raise ValueError(
                "minimum_history must be >= 1."
            )

        if not math.isfinite(
            self.opening_cash_balance
        ):
            raise ValueError(
                "opening_cash_balance must be finite."
            )


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class CashFlowPeriod:
    """
    Represents one historical cash-flow period.
    """

    period: str

    inflow: float

    outflow: float

    opening_balance: Optional[float] = None

    closing_balance: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def net_cash_flow(self) -> float:
        """Calculate net cash flow."""

        return (
            self.inflow
            - self.outflow
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert period to dictionary."""

        return {
            "period": self.period,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "net_cash_flow": self.net_cash_flow,
            "opening_balance": self.opening_balance,
            "closing_balance": self.closing_balance,
            "metadata": dict(self.metadata),
        }


@dataclass
class CashFlowForecastPoint:
    """
    One future cash-flow forecast period.
    """

    period_index: int

    inflow: float

    outflow: float

    net_cash_flow: float

    opening_balance: float

    closing_balance: float

    cumulative_cash_flow: float

    liquidity_status: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert forecast point to dictionary."""

        return {
            "period_index": self.period_index,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "net_cash_flow": self.net_cash_flow,
            "opening_balance": self.opening_balance,
            "closing_balance": self.closing_balance,
            "cumulative_cash_flow":
                self.cumulative_cash_flow,
            "liquidity_status":
                self.liquidity_status,
            "metadata": dict(self.metadata),
        }


@dataclass
class CashFlowForecastResult:
    """
    Complete cash-flow forecast result.
    """

    method: str

    horizon: int

    historical_periods: List[CashFlowPeriod]

    forecast: List[CashFlowForecastPoint]

    historical_inflow: float

    historical_outflow: float

    historical_net_cash_flow: float

    projected_inflow: float

    projected_outflow: float

    projected_net_cash_flow: float

    minimum_projected_balance: float

    ending_projected_balance: float

    liquidity_risk: str

    confidence: float

    created_at: datetime

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""

        return {
            "method": self.method,
            "horizon": self.horizon,
            "historical_periods": [
                item.to_dict()
                for item in self.historical_periods
            ],
            "forecast": [
                item.to_dict()
                for item in self.forecast
            ],
            "historical_inflow":
                self.historical_inflow,
            "historical_outflow":
                self.historical_outflow,
            "historical_net_cash_flow":
                self.historical_net_cash_flow,
            "projected_inflow":
                self.projected_inflow,
            "projected_outflow":
                self.projected_outflow,
            "projected_net_cash_flow":
                self.projected_net_cash_flow,
            "minimum_projected_balance":
                self.minimum_projected_balance,
            "ending_projected_balance":
                self.ending_projected_balance,
            "liquidity_risk":
                self.liquidity_risk,
            "confidence":
                self.confidence,
            "created_at":
                self.created_at.isoformat(),
            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Cash Flow Forecaster
# ============================================================================


class CashFlowForecaster:
    """
    Cash-flow forecasting service.

    Supports:

        - inflow forecasting
        - outflow forecasting
        - net cash-flow forecasting
        - cash-balance projection
        - liquidity risk detection
        - trend analysis
        - confidence estimation
        - multiple baseline methods
    """

    def __init__(
        self,
        config: Optional[
            CashFlowForecastConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or CashFlowForecastConfig()
        )

        self.config.validate()

    # ========================================================================
    # MAIN API
    # ========================================================================

    def forecast(
        self,
        historical_data: Sequence[
            CashFlowPeriod
            | Mapping[str, Any]
        ],
        horizon: Optional[int] = None,
        method: CashFlowMethod | str = (
            CashFlowMethod.MOVING_AVERAGE
        ),
        opening_cash_balance: Optional[float] = None,
    ) -> CashFlowForecastResult:
        """
        Generate a complete cash-flow forecast.

        Historical data can contain:

            {
                "period": "2026-01",
                "inflow": 100000,
                "outflow": 70000
            }

        or CashFlowPeriod objects.
        """

        periods = self._normalize_periods(
            historical_data
        )

        if len(periods) < (
            self.config.minimum_history
        ):
            raise InsufficientCashFlowDataError(
                "At least "
                f"{self.config.minimum_history} "
                "historical periods are required."
            )

        horizon = (
            horizon
            if horizon is not None
            else self.config.default_horizon
        )

        self._validate_horizon(
            horizon
        )

        method = self._normalize_method(
            method
        )

        opening_balance = (
            opening_cash_balance
            if opening_cash_balance is not None
            else self.config.opening_cash_balance
        )

        if not math.isfinite(
            float(opening_balance)
        ):
            raise InvalidCashFlowInputError(
                "opening_cash_balance must be finite."
            )

        inflows = [
            period.inflow
            for period in periods
        ]

        outflows = [
            period.outflow
            for period in periods
        ]

        inflow_forecast = (
            self._forecast_series(
                inflows,
                horizon,
                method,
            )
        )

        outflow_forecast = (
            self._forecast_series(
                outflows,
                horizon,
                method,
            )
        )

        forecast_points = (
            self._build_forecast_points(
                inflow_forecast,
                outflow_forecast,
                opening_balance,
            )
        )

        historical_inflow = sum(
            inflows
        )

        historical_outflow = sum(
            outflows
        )

        historical_net = (
            historical_inflow
            - historical_outflow
        )

        projected_inflow = sum(
            inflow_forecast
        )

        projected_outflow = sum(
            outflow_forecast
        )

        projected_net = (
            projected_inflow
            - projected_outflow
        )

        minimum_balance = min(
            point.closing_balance
            for point in forecast_points
        )

        ending_balance = (
            forecast_points[-1]
            .closing_balance
        )

        liquidity_risk = (
            self.assess_liquidity_risk(
                forecast_points
            )
        )

        confidence = (
            self._estimate_confidence(
                inflows,
                outflows,
                inflow_forecast,
                outflow_forecast,
            )
        )

        return CashFlowForecastResult(
            method=method.value,
            horizon=horizon,
            historical_periods=periods,
            forecast=forecast_points,
            historical_inflow=historical_inflow,
            historical_outflow=historical_outflow,
            historical_net_cash_flow=historical_net,
            projected_inflow=projected_inflow,
            projected_outflow=projected_outflow,
            projected_net_cash_flow=projected_net,
            minimum_projected_balance=minimum_balance,
            ending_projected_balance=ending_balance,
            liquidity_risk=liquidity_risk,
            confidence=confidence,
            created_at=datetime.now(
                timezone.utc
            ),
            metadata={
                "currency":
                    self.config.currency,
                "history_length":
                    len(periods),
                "method":
                    method.value,
            },
        )

    # ========================================================================
    # SERIES FORECASTING
    # ========================================================================

    def forecast_inflows(
        self,
        historical_inflows: Sequence[float],
        horizon: int,
        method: CashFlowMethod | str = (
            CashFlowMethod.MOVING_AVERAGE
        ),
    ) -> List[float]:
        """Forecast future cash inflows."""

        values = self._validate_series(
            historical_inflows,
            "historical_inflows",
        )

        return self._forecast_series(
            values,
            horizon,
            self._normalize_method(
                method
            ),
        )

    def forecast_outflows(
        self,
        historical_outflows: Sequence[float],
        horizon: int,
        method: CashFlowMethod | str = (
            CashFlowMethod.MOVING_AVERAGE
        ),
    ) -> List[float]:
        """Forecast future cash outflows."""

        values = self._validate_series(
            historical_outflows,
            "historical_outflows",
        )

        return self._forecast_series(
            values,
            horizon,
            self._normalize_method(
                method
            ),
        )

    def forecast_net_cash_flow(
        self,
        historical_net_cash_flow: Sequence[float],
        horizon: int,
        method: CashFlowMethod | str = (
            CashFlowMethod.MOVING_AVERAGE
        ),
    ) -> List[float]:
        """Forecast net cash flow directly."""

        values = self._validate_series(
            historical_net_cash_flow,
            "historical_net_cash_flow",
        )

        return self._forecast_series(
            values,
            horizon,
            self._normalize_method(
                method
            ),
        )

    # ========================================================================
    # BASELINE METHODS
    # ========================================================================

    def naive(
        self,
        values: Sequence[float],
        horizon: int,
    ) -> List[float]:
        """Forecast using the latest historical value."""

        series = self._validate_series(
            values,
            "values",
        )

        self._validate_horizon(
            horizon
        )

        return [
            series[-1]
            for _ in range(horizon)
        ]

    def moving_average(
        self,
        values: Sequence[float],
        horizon: int,
        window: Optional[int] = None,
    ) -> List[float]:
        """
        Recursive moving-average forecast.
        """

        series = self._validate_series(
            values,
            "values",
        )

        self._validate_horizon(
            horizon
        )

        window = (
            window
            if window is not None
            else self.config.moving_average_window
        )

        if window < 1:
            raise InvalidCashFlowInputError(
                "window must be >= 1."
            )

        if len(series) < window:
            raise InsufficientCashFlowDataError(
                f"At least {window} observations "
                "are required for moving average."
            )

        working = list(series)
        predictions = []

        for _ in range(horizon):

            recent = working[-window:]

            prediction = (
                sum(recent)
                / len(recent)
            )

            predictions.append(
                prediction
            )

            working.append(
                prediction
            )

        return predictions

    def exponential_smoothing(
        self,
        values: Sequence[float],
        horizon: int,
        alpha: Optional[float] = None,
    ) -> List[float]:
        """Simple exponential smoothing."""

        series = self._validate_series(
            values,
            "values",
        )

        self._validate_horizon(
            horizon
        )

        alpha = (
            alpha
            if alpha is not None
            else self.config.exponential_alpha
        )

        if not (
            0.0
            < alpha
            <= 1.0
        ):
            raise InvalidCashFlowInputError(
                "alpha must be > 0 and <= 1."
            )

        smoothed = series[0]

        for value in series[1:]:

            smoothed = (
                alpha * value
                + (1.0 - alpha)
                * smoothed
            )

        return [
            smoothed
            for _ in range(horizon)
        ]

    def drift(
        self,
        values: Sequence[float],
        horizon: int,
    ) -> List[float]:
        """Forecast using historical linear drift."""

        series = self._validate_series(
            values,
            "values",
        )

        self._validate_horizon(
            horizon
        )

        if len(series) < 2:
            raise InsufficientCashFlowDataError(
                "At least two observations "
                "are required for drift."
            )

        average_change = (
            series[-1]
            - series[0]
        ) / (
            len(series) - 1
        )

        last = series[-1]

        return [
            last
            + average_change * step
            for step in range(
                1,
                horizon + 1,
            )
        ]

    # ========================================================================
    # LIQUIDITY ANALYSIS
    # ========================================================================

    def assess_liquidity_risk(
        self,
        forecast: Sequence[
            CashFlowForecastPoint
        ],
    ) -> str:
        """
        Assess projected liquidity risk.

        Risk levels:

            LOW
            MEDIUM
            HIGH
            CRITICAL
        """

        if not forecast:
            return "LOW"

        balances = [
            point.closing_balance
            for point in forecast
        ]

        minimum_balance = min(
            balances
        )

        negative_periods = sum(
            1
            for point in forecast
            if point.closing_balance < 0
        )

        negative_cashflow_periods = sum(
            1
            for point in forecast
            if point.net_cash_flow < 0
        )

        if minimum_balance < 0:

            return "CRITICAL"

        if negative_periods > 0:

            return "CRITICAL"

        if negative_cashflow_periods >= (
            max(2, len(forecast) // 2)
        ):

            return "HIGH"

        if negative_cashflow_periods > 0:

            return "MEDIUM"

        return "LOW"

    def calculate_cash_runway(
        self,
        cash_balance: float,
        average_monthly_outflow: float,
    ) -> Optional[float]:
        """
        Calculate estimated cash runway in periods.

        Returns None when outflow is zero or negative.
        """

        cash_balance = float(
            cash_balance
        )

        average_monthly_outflow = float(
            average_monthly_outflow
        )

        if not math.isfinite(
            cash_balance
        ):
            raise InvalidCashFlowInputError(
                "cash_balance must be finite."
            )

        if not math.isfinite(
            average_monthly_outflow
        ):
            raise InvalidCashFlowInputError(
                "average_monthly_outflow must be finite."
            )

        if average_monthly_outflow <= 0:

            return None

        return max(
            0.0,
            cash_balance
            / average_monthly_outflow,
        )

    def calculate_burn_rate(
        self,
        outflows: Sequence[float],
        inflows: Optional[
            Sequence[float]
        ] = None,
    ) -> float:
        """
        Calculate average net cash burn.

        Positive result means the business is consuming cash.
        Negative result means net cash generation.
        """

        outflow_values = self._validate_series(
            outflows,
            "outflows",
        )

        average_outflow = (
            sum(outflow_values)
            / len(outflow_values)
        )

        if inflows is None:

            return average_outflow

        inflow_values = self._validate_series(
            inflows,
            "inflows",
        )

        if len(inflow_values) != len(
            outflow_values
        ):
            raise InvalidCashFlowInputError(
                "inflows and outflows must "
                "have equal lengths."
            )

        average_inflow = (
            sum(inflow_values)
            / len(inflow_values)
        )

        return (
            average_outflow
            - average_inflow
        )

    # ========================================================================
    # TREND ANALYSIS
    # ========================================================================

    def calculate_trend(
        self,
        values: Sequence[float],
    ) -> float:
        """Calculate average period-over-period change."""

        series = self._validate_series(
            values,
            "values",
        )

        if len(series) < 2:
            return 0.0

        return (
            series[-1]
            - series[0]
        ) / (
            len(series) - 1
        )

    def calculate_growth_rate(
        self,
        values: Sequence[float],
    ) -> float:
        """Calculate total growth rate."""

        series = self._validate_series(
            values,
            "values",
        )

        first = series[0]

        if first == 0:
            return 0.0

        return (
            series[-1] - first
        ) / abs(first)

    # ========================================================================
    # FORECAST POINTS
    # ========================================================================

    def _build_forecast_points(
        self,
        inflows: Sequence[float],
        outflows: Sequence[float],
        opening_balance: float,
    ) -> List[
        CashFlowForecastPoint
    ]:

        if len(inflows) != len(outflows):

            raise CashFlowCalculationError(
                "Inflow and outflow forecasts "
                "must have equal lengths."
            )

        points = []

        current_balance = (
            float(opening_balance)
        )

        cumulative_cash_flow = 0.0

        for index, (
            inflow,
            outflow,
        ) in enumerate(
            zip(
                inflows,
                outflows,
            ),
            start=1,
        ):

            net = (
                inflow
                - outflow
            )

            closing_balance = (
                current_balance
                + net
            )

            cumulative_cash_flow += net

            status = (
                self._liquidity_status(
                    closing_balance
                )
            )

            points.append(
                CashFlowForecastPoint(
                    period_index=index,
                    inflow=inflow,
                    outflow=outflow,
                    net_cash_flow=net,
                    opening_balance=current_balance,
                    closing_balance=closing_balance,
                    cumulative_cash_flow=(
                        cumulative_cash_flow
                    ),
                    liquidity_status=status,
                )
            )

            current_balance = (
                closing_balance
            )

        return points

    @staticmethod
    def _liquidity_status(
        balance: float,
    ) -> str:

        if balance < 0:
            return "CRITICAL"

        if balance == 0:
            return "HIGH"

        return "HEALTHY"

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def _estimate_confidence(
        self,
        historical_inflows: Sequence[float],
        historical_outflows: Sequence[float],
        forecast_inflows: Sequence[float],
        forecast_outflows: Sequence[float],
    ) -> float:
        """
        Estimate operational forecast confidence.

        This is not a statistical prediction interval.
        """

        inflow_stability = (
            self._series_stability(
                historical_inflows
            )
        )

        outflow_stability = (
            self._series_stability(
                historical_outflows
            )
        )

        history_factor = min(
            1.0,
            len(historical_inflows)
            / 24.0,
        )

        forecast_stability = (
            (
                self._series_stability(
                    forecast_inflows
                )
                + self._series_stability(
                    forecast_outflows
                )
            )
            / 2.0
        )

        confidence = (
            0.30 * inflow_stability
            + 0.30 * outflow_stability
            + 0.20 * history_factor
            + 0.20 * forecast_stability
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

    @staticmethod
    def _series_stability(
        values: Sequence[float],
    ) -> float:

        series = [
            float(value)
            for value in values
        ]

        if len(series) < 2:
            return 1.0

        mean = (
            sum(series)
            / len(series)
        )

        denominator = abs(mean)

        if denominator < 1e-12:
            return 1.0

        variance = sum(
            (
                value - mean
            ) ** 2
            for value in series
        ) / len(series)

        std = math.sqrt(
            variance
        )

        coefficient = (
            std
            / denominator
        )

        return max(
            0.0,
            min(
                1.0,
                1.0 - coefficient,
            ),
        )

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    def _normalize_periods(
        self,
        historical_data: Sequence[
            CashFlowPeriod
            | Mapping[str, Any]
        ],
    ) -> List[
        CashFlowPeriod
    ]:

        if not historical_data:
            raise InvalidCashFlowInputError(
                "historical_data cannot be empty."
            )

        result = []

        for index, item in enumerate(
            historical_data
        ):

            if isinstance(
                item,
                CashFlowPeriod,
            ):

                period = item

            elif isinstance(
                item,
                Mapping,
            ):

                period = (
                    self._period_from_mapping(
                        item,
                        index,
                    )
                )

            else:

                raise InvalidCashFlowInputError(
                    "Each historical cash-flow "
                    "item must be a mapping or "
                    "CashFlowPeriod."
                )

            result.append(
                period
            )

        return result

    def _period_from_mapping(
        self,
        item: Mapping[str, Any],
        index: int,
    ) -> CashFlowPeriod:

        period = item.get(
            "period",
            item.get(
                "date",
                item.get(
                    "month",
                    str(index + 1),
                ),
            ),
        )

        inflow = self._number(
            item.get(
                "inflow",
                item.get(
                    "cash_inflow",
                    item.get(
                        "cash_received",
                        0.0,
                    ),
                ),
            ),
            "inflow",
        )

        outflow = self._number(
            item.get(
                "outflow",
                item.get(
                    "cash_outflow",
                    item.get(
                        "cash_paid",
                        0.0,
                    ),
                ),
            ),
            "outflow",
        )

        if (
            self.config.clip_negative_inflows
        ):
            inflow = max(
                0.0,
                inflow,
            )

        if (
            self.config.clip_negative_outflows
        ):
            outflow = max(
                0.0,
                outflow,
            )

        opening_balance = (
            item.get(
                "opening_balance"
            )
        )

        closing_balance = (
            item.get(
                "closing_balance"
            )
        )

        if opening_balance is not None:
            opening_balance = self._number(
                opening_balance,
                "opening_balance",
            )

        if closing_balance is not None:
            closing_balance = self._number(
                closing_balance,
                "closing_balance",
            )

        return CashFlowPeriod(
            period=str(period),
            inflow=inflow,
            outflow=outflow,
            opening_balance=opening_balance,
            closing_balance=closing_balance,
            metadata={
                key: value
                for key, value in item.items()
                if key not in {
                    "period",
                    "date",
                    "month",
                    "inflow",
                    "cash_inflow",
                    "cash_received",
                    "outflow",
                    "cash_outflow",
                    "cash_paid",
                    "opening_balance",
                    "closing_balance",
                }
            },
        )

    # ========================================================================
    # SERIES DISPATCH
    # ========================================================================

    def _forecast_series(
        self,
        values: Sequence[float],
        horizon: int,
        method: CashFlowMethod,
    ) -> List[float]:

        if method == CashFlowMethod.NAIVE:

            return self.naive(
                values,
                horizon,
            )

        if (
            method
            == CashFlowMethod.MOVING_AVERAGE
        ):

            return self.moving_average(
                values,
                horizon,
            )

        if (
            method
            == CashFlowMethod.EXPONENTIAL_SMOOTHING
        ):

            return self.exponential_smoothing(
                values,
                horizon,
            )

        if method == CashFlowMethod.DRIFT:

            return self.drift(
                values,
                horizon,
            )

        raise CashFlowForecastError(
            f"Unsupported cash-flow method: "
            f"{method}"
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_series(
        values: Sequence[float],
        name: str,
    ) -> List[float]:

        if values is None:
            raise InvalidCashFlowInputError(
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

            raise InvalidCashFlowInputError(
                f"{name} must contain only "
                "numeric values."
            ) from exc

        if not result:
            raise InvalidCashFlowInputError(
                f"{name} cannot be empty."
            )

        for value in result:

            if not math.isfinite(
                value
            ):
                raise InvalidCashFlowInputError(
                    f"{name} contains "
                    "non-finite values."
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

            raise InvalidCashFlowInputError(
                "horizon must be an integer."
            )

        if horizon < 1:

            raise InvalidCashFlowInputError(
                "horizon must be >= 1."
            )

    @staticmethod
    def _number(
        value: Any,
        field_name: str,
    ) -> float:

        try:

            number = float(value)

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidCashFlowInputError(
                f"{field_name} must be numeric."
            ) from exc

        if not math.isfinite(
            number
        ):

            raise InvalidCashFlowInputError(
                f"{field_name} must be finite."
            )

        return number

    @staticmethod
    def _normalize_method(
        method: CashFlowMethod | str,
    ) -> CashFlowMethod:

        if isinstance(
            method,
            CashFlowMethod,
        ):
            return method

        try:

            return CashFlowMethod(
                str(
                    method
                ).strip().lower()
            )

        except ValueError as exc:

            raise CashFlowForecastError(
                f"Unsupported cash-flow "
                f"forecast method: {method}"
            ) from exc


# ============================================================================
# Convenience Functions
# ============================================================================


def forecast_cash_flow(
    historical_data: Sequence[
        CashFlowPeriod
        | Mapping[str, Any]
    ],
    horizon: int = 6,
    method: CashFlowMethod | str = (
        CashFlowMethod.MOVING_AVERAGE
    ),
    opening_cash_balance: float = 0.0,
) -> CashFlowForecastResult:
    """
    Convenience API for cash-flow forecasting.
    """

    forecaster = (
        CashFlowForecaster(
            CashFlowForecastConfig(
                default_horizon=horizon,
                opening_cash_balance=(
                    opening_cash_balance
                ),
            )
        )
    )

    return forecaster.forecast(
        historical_data,
        horizon=horizon,
        method=method,
        opening_cash_balance=(
            opening_cash_balance
        ),
    )


def forecast_inflows(
    historical_inflows: Sequence[float],
    horizon: int = 6,
    method: CashFlowMethod | str = (
        CashFlowMethod.MOVING_AVERAGE
    ),
) -> List[float]:
    """Convenience API for inflow forecasting."""

    return (
        CashFlowForecaster()
        .forecast_inflows(
            historical_inflows,
            horizon,
            method,
        )
    )


def forecast_outflows(
    historical_outflows: Sequence[float],
    horizon: int = 6,
    method: CashFlowMethod | str = (
        CashFlowMethod.MOVING_AVERAGE
    ),
) -> List[float]:
    """Convenience API for outflow forecasting."""

    return (
        CashFlowForecaster()
        .forecast_outflows(
            historical_outflows,
            horizon,
            method,
        )
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "CashFlowForecastError",
    "InvalidCashFlowInputError",
    "InsufficientCashFlowDataError",
    "CashFlowCalculationError",
    "CashFlowType",
    "CashFlowMethod",
    "CashFlowForecastConfig",
    "CashFlowPeriod",
    "CashFlowForecastPoint",
    "CashFlowForecastResult",
    "CashFlowForecaster",
    "forecast_cash_flow",
    "forecast_inflows",
    "forecast_outflows",
]