"""
FinCo AI - Liquidity Forecasting

File:
    backend/app/forecasting/liquidity_forecast.py

Purpose:
    Forecast future liquidity position and identify potential
    cash shortfalls, liquidity stress, and funding requirements.

Architecture:

    Historical Cash Flow
            |
            v
    cashflow_forecast.py
            |
            v
    liquidity_forecast.py
            |
       +----+-----+----------------+
       |          |                |
       v          v                v
    Balance    Runway          Burn Rate
       |          |                |
       +----------+----------------+
                  |
                  v
            Liquidity Risk
                  |
          +-------+-------+
          |               |
          v               v
      Alerts         Recommendations
          |               |
          +-------+-------+
                  v
           Supervisor Agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
)
import math
import statistics


# ============================================================================
# Exceptions
# ============================================================================


class LiquidityForecastError(Exception):
    """Base liquidity forecasting exception."""


class InvalidLiquidityInputError(
    LiquidityForecastError
):
    """Raised when liquidity input is invalid."""


class InsufficientLiquidityDataError(
    LiquidityForecastError
):
    """Raised when insufficient liquidity history exists."""


class LiquidityCalculationError(
    LiquidityForecastError
):
    """Raised when a liquidity calculation fails."""


# ============================================================================
# Enums
# ============================================================================


class LiquidityRiskLevel(str, Enum):
    """Liquidity risk classification."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class LiquidityStatus(str, Enum):
    """Point-in-time liquidity status."""

    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    STRESSED = "STRESSED"
    CRITICAL = "CRITICAL"


class LiquidityForecastMethod(str, Enum):
    """Liquidity forecasting methods."""

    CASH_FLOW = "cash_flow"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    DRIFT = "drift"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class LiquidityForecastConfig:
    """
    Configuration for liquidity forecasting.
    """

    default_horizon: int = 6

    minimum_history: int = 3

    opening_cash_balance: float = 0.0

    currency: str = "USD"

    minimum_cash_buffer: float = 0.0

    warning_cash_buffer: float = 0.0

    critical_cash_buffer: float = 0.0

    warning_runway_periods: float = 3.0

    critical_runway_periods: float = 1.0

    high_burn_rate_multiplier: float = 1.25

    critical_burn_rate_multiplier: float = 1.50

    negative_balance_is_critical: bool = True

    clip_negative_inflows: bool = True

    clip_negative_outflows: bool = True

    allow_negative_cash_balance: bool = True

    def validate(self) -> None:
        """Validate configuration."""

        if self.default_horizon < 1:
            raise ValueError(
                "default_horizon must be >= 1."
            )

        if self.minimum_history < 1:
            raise ValueError(
                "minimum_history must be >= 1."
            )

        if self.minimum_cash_buffer < 0:
            raise ValueError(
                "minimum_cash_buffer cannot be negative."
            )

        if self.warning_cash_buffer < 0:
            raise ValueError(
                "warning_cash_buffer cannot be negative."
            )

        if self.critical_cash_buffer < 0:
            raise ValueError(
                "critical_cash_buffer cannot be negative."
            )

        if (
            self.critical_cash_buffer
            > self.warning_cash_buffer
        ):
            raise ValueError(
                "critical_cash_buffer cannot be "
                "greater than warning_cash_buffer."
            )

        if self.warning_runway_periods < 0:
            raise ValueError(
                "warning_runway_periods cannot "
                "be negative."
            )

        if self.critical_runway_periods < 0:
            raise ValueError(
                "critical_runway_periods cannot "
                "be negative."
            )

        if (
            self.critical_runway_periods
            > self.warning_runway_periods
        ):
            raise ValueError(
                "critical_runway_periods cannot "
                "be greater than "
                "warning_runway_periods."
            )

        if self.high_burn_rate_multiplier <= 0:
            raise ValueError(
                "high_burn_rate_multiplier must be > 0."
            )

        if self.critical_burn_rate_multiplier <= 0:
            raise ValueError(
                "critical_burn_rate_multiplier "
                "must be > 0."
            )


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class LiquidityPeriod:
    """
    Historical or projected liquidity period.
    """

    period: Any

    inflow: float

    outflow: float

    opening_balance: float

    closing_balance: float

    net_cash_flow: float

    burn_rate: float

    runway_periods: Optional[float]

    status: LiquidityStatus

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "opening_balance":
                self.opening_balance,
            "closing_balance":
                self.closing_balance,
            "net_cash_flow":
                self.net_cash_flow,
            "burn_rate":
                self.burn_rate,
            "runway_periods":
                self.runway_periods,
            "status":
                self.status.value,
            "metadata":
                dict(self.metadata),
        }


@dataclass
class LiquidityForecastPoint:
    """
    One projected liquidity period.
    """

    period_index: int

    projected_inflow: float

    projected_outflow: float

    projected_net_cash_flow: float

    opening_balance: float

    closing_balance: float

    cumulative_net_cash_flow: float

    burn_rate: float

    runway_periods: Optional[float]

    cash_shortfall: float

    status: LiquidityStatus

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period_index":
                self.period_index,
            "projected_inflow":
                self.projected_inflow,
            "projected_outflow":
                self.projected_outflow,
            "projected_net_cash_flow":
                self.projected_net_cash_flow,
            "opening_balance":
                self.opening_balance,
            "closing_balance":
                self.closing_balance,
            "cumulative_net_cash_flow":
                self.cumulative_net_cash_flow,
            "burn_rate":
                self.burn_rate,
            "runway_periods":
                self.runway_periods,
            "cash_shortfall":
                self.cash_shortfall,
            "status":
                self.status.value,
            "metadata":
                dict(self.metadata),
        }


@dataclass
class LiquidityForecastResult:
    """
    Complete liquidity forecast result.
    """

    method: str

    horizon: int

    historical_periods: List[
        LiquidityPeriod
    ]

    forecast: List[
        LiquidityForecastPoint
    ]

    opening_cash_balance: float

    ending_cash_balance: float

    minimum_projected_balance: float

    average_projected_balance: float

    total_projected_inflow: float

    total_projected_outflow: float

    total_projected_net_cash_flow: float

    average_burn_rate: float

    cash_runway_periods: Optional[float]

    shortfall_periods: int

    maximum_cash_shortfall: float

    liquidity_risk: LiquidityRiskLevel

    confidence: float

    funding_required: float

    created_at: Any

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "horizon": self.horizon,
            "historical_periods": [
                period.to_dict()
                for period
                in self.historical_periods
            ],
            "forecast": [
                point.to_dict()
                for point in self.forecast
            ],
            "opening_cash_balance":
                self.opening_cash_balance,
            "ending_cash_balance":
                self.ending_cash_balance,
            "minimum_projected_balance":
                self.minimum_projected_balance,
            "average_projected_balance":
                self.average_projected_balance,
            "total_projected_inflow":
                self.total_projected_inflow,
            "total_projected_outflow":
                self.total_projected_outflow,
            "total_projected_net_cash_flow":
                self.total_projected_net_cash_flow,
            "average_burn_rate":
                self.average_burn_rate,
            "cash_runway_periods":
                self.cash_runway_periods,
            "shortfall_periods":
                self.shortfall_periods,
            "maximum_cash_shortfall":
                self.maximum_cash_shortfall,
            "liquidity_risk":
                self.liquidity_risk.value,
            "confidence":
                self.confidence,
            "funding_required":
                self.funding_required,
            "created_at":
                self.created_at,
            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Liquidity Forecaster
# ============================================================================


class LiquidityForecaster:
    """
    Liquidity forecasting engine.

    Responsibilities:

        1. Forecast cash inflows.
        2. Forecast cash outflows.
        3. Project cash balances.
        4. Calculate burn rate.
        5. Calculate runway.
        6. Detect cash shortfalls.
        7. Estimate funding requirements.
        8. Classify liquidity risk.
    """

    def __init__(
        self,
        config: Optional[
            LiquidityForecastConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or LiquidityForecastConfig()
        )

        self.config.validate()

    # ========================================================================
    # MAIN FORECAST
    # ========================================================================

    def forecast(
        self,
        historical_data: Sequence[
            Mapping[str, Any]
        ],
        horizon: Optional[int] = None,
        method: LiquidityForecastMethod | str = (
            LiquidityForecastMethod.CASH_FLOW
        ),
        opening_cash_balance: Optional[
            float
        ] = None,
    ) -> LiquidityForecastResult:
        """
        Generate a liquidity forecast.

        Each historical period should contain:

            period
            inflow
            outflow

        Optional fields:

            opening_balance
            closing_balance
        """

        periods = self._normalize_periods(
            historical_data
        )

        if len(periods) < (
            self.config.minimum_history
        ):
            raise InsufficientLiquidityDataError(
                "Insufficient liquidity history. "
                f"At least "
                f"{self.config.minimum_history} "
                "periods are required."
            )

        forecast_horizon = (
            horizon
            if horizon is not None
            else self.config.default_horizon
        )

        self._validate_horizon(
            forecast_horizon
        )

        normalized_method = (
            self._normalize_method(method)
        )

        opening_balance = (
            float(opening_cash_balance)
            if opening_cash_balance
            is not None
            else self._infer_opening_balance(
                periods
            )
        )

        historical_periods = (
            self._build_historical_periods(
                periods
            )
        )

        inflows = [
            item["inflow"]
            for item in periods
        ]

        outflows = [
            item["outflow"]
            for item in periods
        ]

        projected_inflows = (
            self._forecast_series(
                inflows,
                forecast_horizon,
                normalized_method,
                clip_negative=self.config.clip_negative_inflows,
            )
        )

        projected_outflows = (
            self._forecast_series(
                outflows,
                forecast_horizon,
                normalized_method,
                clip_negative=self.config.clip_negative_outflows,
            )
        )

        forecast_points = (
            self._build_forecast(
                projected_inflows,
                projected_outflows,
                opening_balance,
            )
        )

        ending_balance = (
            forecast_points[-1].closing_balance
        )

        balances = [
            point.closing_balance
            for point in forecast_points
        ]

        minimum_balance = min(
            balances
        )

        average_balance = (
            statistics.fmean(
                balances
            )
        )

        total_inflow = sum(
            projected_inflows
        )

        total_outflow = sum(
            projected_outflows
        )

        total_net = (
            total_inflow
            - total_outflow
        )

        burn_rates = [
            point.burn_rate
            for point in forecast_points
        ]

        average_burn_rate = (
            statistics.fmean(
                burn_rates
            )
            if burn_rates
            else 0.0
        )

        runway = self.calculate_runway(
            opening_balance,
            average_burn_rate,
        )

        shortfall_periods = sum(
            1
            for point in forecast_points
            if point.cash_shortfall > 0
        )

        maximum_shortfall = max(
            (
                point.cash_shortfall
                for point in forecast_points
            ),
            default=0.0,
        )

        funding_required = (
            self.calculate_funding_requirement(
                forecast_points
            )
        )

        risk = self.assess_liquidity_risk(
            minimum_balance=minimum_balance,
            runway_periods=runway,
            shortfall_periods=shortfall_periods,
            horizon=forecast_horizon,
            average_burn_rate=average_burn_rate,
            historical_outflows=outflows,
        )

        confidence = (
            self._estimate_confidence(
                inflows,
                outflows,
                projected_inflows,
                projected_outflows,
            )
        )

        return LiquidityForecastResult(
            method=normalized_method.value,
            horizon=forecast_horizon,
            historical_periods=historical_periods,
            forecast=forecast_points,
            opening_cash_balance=opening_balance,
            ending_cash_balance=ending_balance,
            minimum_projected_balance=minimum_balance,
            average_projected_balance=average_balance,
            total_projected_inflow=total_inflow,
            total_projected_outflow=total_outflow,
            total_projected_net_cash_flow=total_net,
            average_burn_rate=average_burn_rate,
            cash_runway_periods=runway,
            shortfall_periods=shortfall_periods,
            maximum_cash_shortfall=maximum_shortfall,
            liquidity_risk=risk,
            confidence=confidence,
            funding_required=funding_required,
            created_at=self._utc_now(),
            metadata={
                "currency":
                    self.config.currency,
                "minimum_cash_buffer":
                    self.config.minimum_cash_buffer,
                "warning_cash_buffer":
                    self.config.warning_cash_buffer,
                "critical_cash_buffer":
                    self.config.critical_cash_buffer,
            },
        )

    # ========================================================================
    # INFLOW FORECAST
    # ========================================================================

    def forecast_inflows(
        self,
        historical_inflows: Sequence[float],
        horizon: Optional[int] = None,
        method: LiquidityForecastMethod | str = (
            LiquidityForecastMethod.MOVING_AVERAGE
        ),
    ) -> List[float]:
        """Forecast future cash inflows."""

        values = self._validate_series(
            historical_inflows,
            "historical_inflows",
        )

        horizon = (
            horizon
            if horizon is not None
            else self.config.default_horizon
        )

        return self._forecast_series(
            values,
            horizon,
            self._normalize_method(
                method
            ),
            clip_negative=(
                self.config.clip_negative_inflows
            ),
        )

    # ========================================================================
    # OUTFLOW FORECAST
    # ========================================================================

    def forecast_outflows(
        self,
        historical_outflows: Sequence[float],
        horizon: Optional[int] = None,
        method: LiquidityForecastMethod | str = (
            LiquidityForecastMethod.MOVING_AVERAGE
        ),
    ) -> List[float]:
        """Forecast future cash outflows."""

        values = self._validate_series(
            historical_outflows,
            "historical_outflows",
        )

        horizon = (
            horizon
            if horizon is not None
            else self.config.default_horizon
        )

        return self._forecast_series(
            values,
            horizon,
            self._normalize_method(
                method
            ),
            clip_negative=(
                self.config.clip_negative_outflows
            ),
        )

    # ========================================================================
    # NET CASH FLOW
    # ========================================================================

    def forecast_net_cash_flow(
        self,
        historical_inflows: Sequence[float],
        historical_outflows: Sequence[float],
        horizon: Optional[int] = None,
        method: LiquidityForecastMethod | str = (
            LiquidityForecastMethod.MOVING_AVERAGE
        ),
    ) -> List[float]:
        """
        Forecast net cash flow.

        Net cash flow = inflow - outflow.
        """

        inflows = self._validate_series(
            historical_inflows,
            "historical_inflows",
        )

        outflows = self._validate_series(
            historical_outflows,
            "historical_outflows",
        )

        if len(inflows) != len(
            outflows
        ):
            raise InvalidLiquidityInputError(
                "Inflows and outflows must "
                "have equal lengths."
            )

        horizon = (
            horizon
            if horizon is not None
            else self.config.default_horizon
        )

        projected_inflows = (
            self.forecast_inflows(
                inflows,
                horizon,
                method,
            )
        )

        projected_outflows = (
            self.forecast_outflows(
                outflows,
                horizon,
                method,
            )
        )

        return [
            inflow - outflow
            for inflow, outflow
            in zip(
                projected_inflows,
                projected_outflows,
            )
        ]

    # ========================================================================
    # BURN RATE
    # ========================================================================

    def calculate_burn_rate(
        self,
        inflows: Sequence[float],
        outflows: Sequence[float],
    ) -> float:
        """
        Calculate average net cash burn.

        Positive result:
            Cash is being consumed.

        Zero/negative:
            No net cash burn.
        """

        inflow_values = (
            self._validate_series(
                inflows,
                "inflows",
            )
        )

        outflow_values = (
            self._validate_series(
                outflows,
                "outflows",
            )
        )

        if len(inflow_values) != len(
            outflow_values
        ):
            raise InvalidLiquidityInputError(
                "Inflows and outflows must "
                "have equal lengths."
            )

        burns = [
            max(
                0.0,
                outflow - inflow,
            )
            for inflow, outflow
            in zip(
                inflow_values,
                outflow_values,
            )
        ]

        if not burns:
            return 0.0

        return statistics.fmean(
            burns
        )

    # ========================================================================
    # RUNWAY
    # ========================================================================

    def calculate_runway(
        self,
        cash_balance: float,
        burn_rate: float,
    ) -> Optional[float]:
        """
        Calculate cash runway in periods.

        runway = cash balance / burn rate
        """

        cash = self._number(
            cash_balance,
            "cash_balance",
        )

        burn = self._number(
            burn_rate,
            "burn_rate",
        )

        if burn <= 0:

            return None

        if cash <= 0:

            return 0.0

        return cash / burn

    # ========================================================================
    # FUNDING REQUIREMENT
    # ========================================================================

    def calculate_funding_requirement(
        self,
        forecast: Sequence[
            LiquidityForecastPoint
        ],
    ) -> float:
        """
        Calculate minimum funding required to maintain
        the configured minimum cash buffer.
        """

        if not forecast:
            return 0.0

        minimum_balance = min(
            point.closing_balance
            for point in forecast
        )

        required = (
            self.config.minimum_cash_buffer
            - minimum_balance
        )

        return max(
            0.0,
            required,
        )

    # ========================================================================
    # LIQUIDITY RISK
    # ========================================================================

    def assess_liquidity_risk(
        self,
        minimum_balance: float,
        runway_periods: Optional[float],
        shortfall_periods: int,
        horizon: int,
        average_burn_rate: float = 0.0,
        historical_outflows: Optional[
            Sequence[float]
        ] = None,
    ) -> LiquidityRiskLevel:
        """
        Classify projected liquidity risk.
        """

        minimum_balance = self._number(
            minimum_balance,
            "minimum_balance",
        )

        average_burn_rate = self._number(
            average_burn_rate,
            "average_burn_rate",
        )

        if (
            self.config.negative_balance_is_critical
            and minimum_balance < 0
        ):
            return LiquidityRiskLevel.CRITICAL

        if shortfall_periods > 0:
            return LiquidityRiskLevel.CRITICAL

        if (
            minimum_balance
            <= self.config.critical_cash_buffer
        ):
            return LiquidityRiskLevel.CRITICAL

        if (
            runway_periods is not None
            and runway_periods
            <= self.config.critical_runway_periods
        ):
            return LiquidityRiskLevel.CRITICAL

        if (
            minimum_balance
            <= self.config.warning_cash_buffer
        ):
            return LiquidityRiskLevel.HIGH

        if (
            runway_periods is not None
            and runway_periods
            <= self.config.warning_runway_periods
        ):
            return LiquidityRiskLevel.HIGH

        if (
            historical_outflows
            and average_burn_rate > 0
        ):

            historical_burn = (
                self._estimate_historical_burn(
                    historical_outflows
                )
            )

            if (
                historical_burn > 0
                and average_burn_rate
                >= (
                    historical_burn
                    * self.config.critical_burn_rate_multiplier
                )
            ):
                return LiquidityRiskLevel.HIGH

            if (
                historical_burn > 0
                and average_burn_rate
                >= (
                    historical_burn
                    * self.config.high_burn_rate_multiplier
                )
            ):
                return LiquidityRiskLevel.MEDIUM

        if minimum_balance <= (
            self.config.minimum_cash_buffer
        ):
            return LiquidityRiskLevel.MEDIUM

        return LiquidityRiskLevel.LOW

    # ========================================================================
    # CASH SHORTFALL
    # ========================================================================

    def calculate_cash_shortfall(
        self,
        balance: float,
    ) -> float:
        """
        Calculate cash shortfall against minimum buffer.
        """

        balance = self._number(
            balance,
            "balance",
        )

        return max(
            0.0,
            self.config.minimum_cash_buffer
            - balance,
        )

    # ========================================================================
    # LIQUIDITY RATIO
    # ========================================================================

    def liquidity_ratio(
        self,
        cash_balance: float,
        expected_outflows: float,
    ) -> float:
        """
        Calculate cash liquidity ratio.

        cash / expected outflows
        """

        cash = self._number(
            cash_balance,
            "cash_balance",
        )

        outflow = self._number(
            expected_outflows,
            "expected_outflows",
        )

        if outflow <= 0:
            return math.inf

        return cash / outflow

    # ========================================================================
    # CASH BUFFER
    # ========================================================================

    def calculate_cash_buffer(
        self,
        cash_balance: float,
        expected_outflows: float,
    ) -> float:
        """
        Calculate cash remaining after expected outflows.
        """

        cash = self._number(
            cash_balance,
            "cash_balance",
        )

        outflow = self._number(
            expected_outflows,
            "expected_outflows",
        )

        return cash - outflow

    # ========================================================================
    # LIQUIDITY STATUS
    # ========================================================================

    def liquidity_status(
        self,
        balance: float,
        runway_periods: Optional[float] = None,
    ) -> LiquidityStatus:
        """Classify one liquidity observation."""

        if balance < 0:

            return LiquidityStatus.CRITICAL

        if (
            balance
            <= self.config.critical_cash_buffer
        ):

            return LiquidityStatus.CRITICAL

        if (
            runway_periods is not None
            and runway_periods
            <= self.config.critical_runway_periods
        ):

            return LiquidityStatus.CRITICAL

        if (
            balance
            <= self.config.warning_cash_buffer
        ):

            return LiquidityStatus.STRESSED

        if (
            runway_periods is not None
            and runway_periods
            <= self.config.warning_runway_periods
        ):

            return LiquidityStatus.STRESSED

        if (
            balance
            <= self.config.minimum_cash_buffer
        ):

            return LiquidityStatus.WATCH

        return LiquidityStatus.HEALTHY

    # ========================================================================
    # TREND
    # ========================================================================

    def calculate_liquidity_trend(
        self,
        balances: Sequence[float],
    ) -> float:
        """
        Calculate simple liquidity trend.

        Positive:
            improving liquidity.

        Negative:
            deteriorating liquidity.
        """

        values = self._validate_series(
            balances,
            "balances",
        )

        if len(values) < 2:
            return 0.0

        x = list(
            range(len(values))
        )

        x_mean = statistics.fmean(x)
        y_mean = statistics.fmean(values)

        denominator = sum(
            (
                item - x_mean
            ) ** 2
            for item in x
        )

        if denominator <= 0:
            return 0.0

        numerator = sum(
            (
                x_value - x_mean
            )
            * (
                y_value - y_mean
            )
            for x_value, y_value
            in zip(x, values)
        )

        return numerator / denominator

    # ========================================================================
    # INTERNAL FORECAST
    # ========================================================================

    def _forecast_series(
        self,
        values: Sequence[float],
        horizon: int,
        method: LiquidityForecastMethod,
        clip_negative: bool,
    ) -> List[float]:

        values = self._validate_series(
            values,
            "values",
        )

        if horizon < 1:
            raise InvalidLiquidityInputError(
                "horizon must be >= 1."
            )

        if method == (
            LiquidityForecastMethod.CASH_FLOW
        ):
            method = (
                LiquidityForecastMethod.MOVING_AVERAGE
            )

        if method == (
            LiquidityForecastMethod.MOVING_AVERAGE
        ):

            window = min(
                3,
                len(values),
            )

            baseline = statistics.fmean(
                values[-window:]
            )

            result = [
                baseline
                for _ in range(horizon)
            ]

        elif method == (
            LiquidityForecastMethod.EXPONENTIAL_SMOOTHING
        ):

            alpha = 0.30
            level = values[0]

            for value in values[1:]:
                level = (
                    alpha * value
                    + (1 - alpha) * level
                )

            result = [
                level
                for _ in range(horizon)
            ]

        elif method == (
            LiquidityForecastMethod.DRIFT
        ):

            if len(values) < 2:

                result = [
                    values[-1]
                    for _ in range(horizon)
                ]

            else:

                drift = (
                    values[-1]
                    - values[0]
                ) / (
                    len(values) - 1
                )

                result = [
                    max(
                        0.0
                        if clip_negative
                        else -math.inf,
                        values[-1]
                        + drift * step,
                    )
                    for step in range(
                        1,
                        horizon + 1,
                    )
                ]

        else:

            raise LiquidityCalculationError(
                f"Unsupported method: {method}"
            )

        if clip_negative:

            result = [
                max(
                    0.0,
                    value,
                )
                for value in result
            ]

        return result

    # ========================================================================
    # BUILD FORECAST
    # ========================================================================

    def _build_forecast(
        self,
        projected_inflows: Sequence[float],
        projected_outflows: Sequence[float],
        opening_balance: float,
    ) -> List[
        LiquidityForecastPoint
    ]:

        if len(projected_inflows) != len(
            projected_outflows
        ):
            raise InvalidLiquidityInputError(
                "Projected inflows and outflows "
                "must have equal lengths."
            )

        forecast = []

        balance = opening_balance
        cumulative_net = 0.0

        for index, (
            inflow,
            outflow,
        ) in enumerate(
            zip(
                projected_inflows,
                projected_outflows,
            ),
            start=1,
        ):

            net = (
                inflow
                - outflow
            )

            opening = balance

            balance = (
                balance + net
            )

            cumulative_net += net

            burn = max(
                0.0,
                outflow - inflow,
            )

            runway = self.calculate_runway(
                balance,
                burn,
            )

            shortfall = (
                self.calculate_cash_shortfall(
                    balance
                )
            )

            status = (
                self.liquidity_status(
                    balance,
                    runway,
                )
            )

            forecast.append(
                LiquidityForecastPoint(
                    period_index=index,
                    projected_inflow=inflow,
                    projected_outflow=outflow,
                    projected_net_cash_flow=net,
                    opening_balance=opening,
                    closing_balance=balance,
                    cumulative_net_cash_flow=(
                        cumulative_net
                    ),
                    burn_rate=burn,
                    runway_periods=runway,
                    cash_shortfall=shortfall,
                    status=status,
                    metadata={
                        "currency":
                            self.config.currency,
                    },
                )
            )

        return forecast

    # ========================================================================
    # HISTORICAL PERIODS
    # ========================================================================

    def _build_historical_periods(
        self,
        periods: Sequence[
            Mapping[str, Any]
        ],
    ) -> List[LiquidityPeriod]:

        result = []

        running_balance = (
            self._infer_opening_balance(
                periods
            )
        )

        for item in periods:

            opening = item.get(
                "opening_balance"
            )

            if opening is None:
                opening = running_balance

            opening = float(
                opening
            )

            inflow = float(
                item["inflow"]
            )

            outflow = float(
                item["outflow"]
            )

            net = (
                inflow
                - outflow
            )

            closing = item.get(
                "closing_balance"
            )

            if closing is None:
                closing = (
                    opening + net
                )

            closing = float(
                closing
            )

            burn = max(
                0.0,
                outflow - inflow,
            )

            runway = self.calculate_runway(
                closing,
                burn,
            )

            status = (
                self.liquidity_status(
                    closing,
                    runway,
                )
            )

            result.append(
                LiquidityPeriod(
                    period=item.get(
                        "period"
                    ),
                    inflow=inflow,
                    outflow=outflow,
                    opening_balance=opening,
                    closing_balance=closing,
                    net_cash_flow=net,
                    burn_rate=burn,
                    runway_periods=runway,
                    status=status,
                )
            )

            running_balance = closing

        return result

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    def _normalize_periods(
        self,
        data: Sequence[
            Mapping[str, Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:

        if data is None or not data:

            raise InvalidLiquidityInputError(
                "historical_data cannot be empty."
            )

        normalized = []

        for index, item in enumerate(
            data
        ):

            if not isinstance(
                item,
                Mapping,
            ):

                raise InvalidLiquidityInputError(
                    f"Period {index} must be "
                    "a mapping."
                )

            if "inflow" not in item:

                raise InvalidLiquidityInputError(
                    f"Period {index} is missing "
                    "'inflow'."
                )

            if "outflow" not in item:

                raise InvalidLiquidityInputError(
                    f"Period {index} is missing "
                    "'outflow'."
                )

            inflow = self._number(
                item["inflow"],
                f"inflow[{index}]",
            )

            outflow = self._number(
                item["outflow"],
                f"outflow[{index}]",
            )

            if self.config.clip_negative_inflows:
                inflow = max(
                    0.0,
                    inflow,
                )

            if self.config.clip_negative_outflows:
                outflow = max(
                    0.0,
                    outflow,
                )

            normalized.append(
                {
                    "period":
                        item.get(
                            "period",
                            index + 1,
                        ),
                    "inflow":
                        inflow,
                    "outflow":
                        outflow,
                    "opening_balance":
                        item.get(
                            "opening_balance"
                        ),
                    "closing_balance":
                        item.get(
                            "closing_balance"
                        ),
                }
            )

        return normalized

    def _infer_opening_balance(
        self,
        periods: Sequence[
            Mapping[str, Any]
        ],
    ) -> float:

        if not periods:
            return (
                self.config.opening_cash_balance
            )

        first = periods[0]

        if (
            first.get(
                "opening_balance"
            )
            is not None
        ):
            return float(
                first[
                    "opening_balance"
                ]
            )

        return (
            self.config.opening_cash_balance
        )

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def _estimate_confidence(
        self,
        historical_inflows: Sequence[float],
        historical_outflows: Sequence[float],
        projected_inflows: Sequence[float],
        projected_outflows: Sequence[float],
    ) -> float:
        """
        Estimate forecast confidence from historical stability.

        This is a heuristic confidence score, not a statistical
        probability.
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

        stability = (
            inflow_stability
            + outflow_stability
        ) / 2.0

        confidence = (
            0.50
            + 0.45 * stability
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

    def _series_stability(
        self,
        values: Sequence[float],
    ) -> float:

        if len(values) < 2:
            return 0.5

        mean = statistics.fmean(
            values
        )

        if abs(mean) <= 1e-8:
            return 0.5

        std = statistics.pstdev(
            values
        )

        coefficient = (
            std / abs(mean)
        )

        return max(
            0.0,
            min(
                1.0,
                1.0 - coefficient,
            ),
        )

    def _estimate_historical_burn(
        self,
        outflows: Sequence[float],
    ) -> float:

        if not outflows:
            return 0.0

        return statistics.fmean(
            outflows
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
            raise InvalidLiquidityInputError(
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

            raise InvalidLiquidityInputError(
                f"{name} must contain numeric "
                "values."
            ) from exc

        if not result:

            raise InvalidLiquidityInputError(
                f"{name} cannot be empty."
            )

        for value in result:

            if not math.isfinite(
                value
            ):

                raise InvalidLiquidityInputError(
                    f"{name} contains a "
                    "non-finite value."
                )

        return result

    @staticmethod
    def _number(
        value: Any,
        name: str,
    ) -> float:

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidLiquidityInputError(
                f"{name} must be numeric."
            ) from exc

        if not math.isfinite(
            number
        ):

            raise InvalidLiquidityInputError(
                f"{name} must be finite."
            )

        return number

    @staticmethod
    def _validate_horizon(
        horizon: int,
    ) -> None:

        if not isinstance(
            horizon,
            int,
        ):

            raise InvalidLiquidityInputError(
                "horizon must be an integer."
            )

        if horizon < 1:

            raise InvalidLiquidityInputError(
                "horizon must be >= 1."
            )

    @staticmethod
    def _normalize_method(
        method: LiquidityForecastMethod | str,
    ) -> LiquidityForecastMethod:

        if isinstance(
            method,
            LiquidityForecastMethod,
        ):
            return method

        try:

            return LiquidityForecastMethod(
                str(
                    method
                ).strip().lower()
            )

        except ValueError as exc:

            raise InvalidLiquidityInputError(
                f"Unsupported liquidity "
                f"forecast method: {method}"
            ) from exc

    @staticmethod
    def _utc_now() -> Any:
        from datetime import datetime, timezone

        return datetime.now(
            timezone.utc
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def forecast_liquidity(
    historical_data: Sequence[
        Mapping[str, Any]
    ],
    horizon: int = 6,
    opening_cash_balance: float = 0.0,
) -> LiquidityForecastResult:
    """Convenience liquidity forecasting API."""

    forecaster = LiquidityForecaster(
        LiquidityForecastConfig(
            default_horizon=horizon,
            opening_cash_balance=(
                opening_cash_balance
            ),
        )
    )

    return forecaster.forecast(
        historical_data=historical_data,
        horizon=horizon,
        opening_cash_balance=(
            opening_cash_balance
        ),
    )


def calculate_runway(
    cash_balance: float,
    burn_rate: float,
) -> Optional[float]:
    """Convenience runway calculation."""

    return LiquidityForecaster().calculate_runway(
        cash_balance,
        burn_rate,
    )


def calculate_burn_rate(
    inflows: Sequence[float],
    outflows: Sequence[float],
) -> float:
    """Convenience burn-rate calculation."""

    return LiquidityForecaster().calculate_burn_rate(
        inflows,
        outflows,
    )


def calculate_liquidity_risk(
    minimum_balance: float,
    runway_periods: Optional[float],
    shortfall_periods: int = 0,
    horizon: int = 6,
) -> LiquidityRiskLevel:
    """Convenience liquidity-risk calculation."""

    return LiquidityForecaster().assess_liquidity_risk(
        minimum_balance=minimum_balance,
        runway_periods=runway_periods,
        shortfall_periods=shortfall_periods,
        horizon=horizon,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "LiquidityForecastError",
    "InvalidLiquidityInputError",
    "InsufficientLiquidityDataError",
    "LiquidityCalculationError",
    "LiquidityRiskLevel",
    "LiquidityStatus",
    "LiquidityForecastMethod",
    "LiquidityForecastConfig",
    "LiquidityPeriod",
    "LiquidityForecastPoint",
    "LiquidityForecastResult",
    "LiquidityForecaster",
    "forecast_liquidity",
    "calculate_runway",
    "calculate_burn_rate",
    "calculate_liquidity_risk",
]