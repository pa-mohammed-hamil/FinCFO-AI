"""
FinCo AI - Historical Financial Analysis

File:
    backend/app/financial/historical.py

Purpose:
    Analyze historical financial performance across reporting periods.

Responsibilities:
    - Store normalized historical financial observations.
    - Sort and align periods chronologically.
    - Calculate period-over-period changes.
    - Calculate growth rates.
    - Calculate moving averages.
    - Calculate volatility.
    - Detect historical trends.
    - Identify highs, lows, peaks, and deterioration.
    - Compare current performance with historical baselines.
    - Generate historical evidence for forecasting, alerts, reports,
      recommendations, and agent workflows.

Pipeline:

    Financial Extraction / Normalization
                ↓
        Historical Metrics
                ↓
       historical.py
                ↓
    ┌───────────────────────────────┐
    │ Growth                        │
    │ Trend                         │
    │ Moving Average                │
    │ Volatility                    │
    │ High / Low                    │
    │ Period Comparison             │
    │ Baseline Deviation            │
    └───────────────────────────────┘
                ↓
       Forecasting / Alerts
                ↓
        Supervisor Agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence


# ============================================================================
# Constants
# ============================================================================

ZERO = Decimal("0")
ONE = Decimal("1")
ONE_HUNDRED = Decimal("100")

MONEY_QUANT = Decimal("0.01")
PERCENT_QUANT = Decimal("0.01")


# ============================================================================
# Enums
# ============================================================================


class HistoricalTrend(str, Enum):
    """Overall historical trend direction."""

    STRONG_GROWTH = "strong_growth"
    GROWTH = "growth"
    STABLE = "stable"
    DECLINE = "decline"
    STRONG_DECLINE = "strong_decline"
    VOLATILE = "volatile"
    INSUFFICIENT_DATA = "insufficient_data"


class ComparisonDirection(str, Enum):
    """Direction of a historical comparison."""

    INCREASE = "increase"
    DECREASE = "decrease"
    UNCHANGED = "unchanged"
    UNKNOWN = "unknown"


class PeriodFrequency(str, Enum):
    """Historical observation frequency."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    UNKNOWN = "unknown"


class HistoricalSignalType(str, Enum):
    """Types of historical signals."""

    GROWTH = "growth"
    DECLINE = "decline"
    PEAK = "peak"
    TROUGH = "trough"
    VOLATILITY = "volatility"
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"
    BASELINE_DEVIATION = "baseline_deviation"


# ============================================================================
# Utility Functions
# ============================================================================


def to_decimal(
    value: Any,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely convert a value into Decimal."""
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except Exception:
        return default


def round_money(value: Decimal) -> Decimal:
    """Round a monetary value to two decimal places."""
    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_percent(value: Decimal) -> Decimal:
    """Round a percentage to two decimal places."""
    return value.quantize(
        PERCENT_QUANT,
        rounding=ROUND_HALF_UP,
    )


def safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely divide two Decimal values."""
    if denominator == ZERO:
        return default

    return numerator / denominator


def percentage_change(
    current: Decimal,
    previous: Decimal,
) -> Decimal:
    """
    Calculate percentage change.

    Example:
        120 vs 100 -> +20%
        80 vs 100 -> -20%
    """
    if previous == ZERO:
        if current == ZERO:
            return ZERO

        return ONE_HUNDRED

    return round_percent(
        ((current - previous) / abs(previous))
        * ONE_HUNDRED
    )


def absolute_change(
    current: Decimal,
    previous: Decimal,
) -> Decimal:
    """Calculate absolute period-over-period change."""
    return round_money(current - previous)


def normalize_date(value: Any) -> date | None:
    """Normalize common date representations."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    text = str(value).strip()

    formats = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%Y/%m/%d",
    )

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    return None


# ============================================================================
# Data Models
# ============================================================================


@dataclass(slots=True)
class HistoricalObservation:
    """
    One historical financial observation.

    Example:
        period = 2025-03-31
        revenue = 1,500,000
        net_profit = 150,000
    """

    period: date

    revenue: Decimal = ZERO
    gross_profit: Decimal = ZERO
    operating_profit: Decimal = ZERO
    net_profit: Decimal = ZERO

    operating_expenses: Decimal = ZERO

    assets: Decimal = ZERO
    liabilities: Decimal = ZERO
    equity: Decimal = ZERO
    debt: Decimal = ZERO

    cash: Decimal = ZERO
    receivables: Decimal = ZERO
    inventory: Decimal = ZERO
    payables: Decimal = ZERO

    operating_cash_flow: Decimal = ZERO
    investing_cash_flow: Decimal = ZERO
    financing_cash_flow: Decimal = ZERO
    free_cash_flow: Decimal | None = None

    gross_margin: Decimal | None = None
    operating_margin: Decimal | None = None
    net_margin: Decimal | None = None

    current_ratio: Decimal | None = None
    debt_to_equity: Decimal | None = None

    currency: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def derive_metrics(self) -> None:
        """Derive basic financial ratios where possible."""

        if (
            self.gross_margin is None
            and self.revenue != ZERO
        ):
            self.gross_margin = round_percent(
                self.gross_profit
                / self.revenue
                * ONE_HUNDRED
            )

        if (
            self.operating_margin is None
            and self.revenue != ZERO
        ):
            self.operating_margin = round_percent(
                self.operating_profit
                / self.revenue
                * ONE_HUNDRED
            )

        if (
            self.net_margin is None
            and self.revenue != ZERO
        ):
            self.net_margin = round_percent(
                self.net_profit
                / self.revenue
                * ONE_HUNDRED
            )

        if self.free_cash_flow is None:
            self.free_cash_flow = (
                self.operating_cash_flow
                + self.investing_cash_flow
            )

        if (
            self.debt_to_equity is None
            and self.equity != ZERO
        ):
            self.debt_to_equity = round_percent(
                self.debt / self.equity
            )

    def metric_value(
        self,
        metric: str,
    ) -> Decimal | None:
        """Return a metric value by name."""
        value = getattr(self, metric, None)

        if value is None:
            return None

        return to_decimal(value)


@dataclass(slots=True)
class HistoricalComparison:
    """Comparison between two historical observations."""

    metric: str

    current_period: date
    previous_period: date

    current_value: Decimal
    previous_value: Decimal

    absolute_change: Decimal
    percentage_change: Decimal

    direction: ComparisonDirection

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "current_period": self.current_period.isoformat(),
            "previous_period": self.previous_period.isoformat(),
            "current_value": str(self.current_value),
            "previous_value": str(self.previous_value),
            "absolute_change": str(self.absolute_change),
            "percentage_change": str(self.percentage_change),
            "direction": self.direction.value,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class HistoricalSignal:
    """A detected historical pattern or signal."""

    signal_type: HistoricalSignalType

    metric: str

    period: date

    severity: str

    description: str

    value: Decimal | None = None

    reference_value: Decimal | None = None

    percentage_difference: Decimal | None = None

    evidence: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type.value,
            "metric": self.metric,
            "period": self.period.isoformat(),
            "severity": self.severity,
            "description": self.description,
            "value": (
                str(self.value)
                if self.value is not None
                else None
            ),
            "reference_value": (
                str(self.reference_value)
                if self.reference_value is not None
                else None
            ),
            "percentage_difference": (
                str(self.percentage_difference)
                if self.percentage_difference is not None
                else None
            ),
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class HistoricalMetricResult:
    """Historical analysis for one metric."""

    metric: str

    observations: list[HistoricalObservation]

    values: list[Decimal]

    changes: list[HistoricalComparison]

    moving_average: list[Decimal]

    trend: HistoricalTrend

    average: Decimal

    median: Decimal

    minimum: Decimal

    maximum: Decimal

    volatility: Decimal

    latest_value: Decimal | None

    previous_value: Decimal | None

    latest_growth: Decimal | None

    signals: list[HistoricalSignal] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "values": [str(value) for value in self.values],
            "changes": [
                change.to_dict()
                for change in self.changes
            ],
            "moving_average": [
                str(value)
                for value in self.moving_average
            ],
            "trend": self.trend.value,
            "average": str(self.average),
            "median": str(self.median),
            "minimum": str(self.minimum),
            "maximum": str(self.maximum),
            "volatility": str(self.volatility),
            "latest_value": (
                str(self.latest_value)
                if self.latest_value is not None
                else None
            ),
            "previous_value": (
                str(self.previous_value)
                if self.previous_value is not None
                else None
            ),
            "latest_growth": (
                str(self.latest_growth)
                if self.latest_growth is not None
                else None
            ),
            "signals": [
                signal.to_dict()
                for signal in self.signals
            ],
        }


@dataclass(slots=True)
class HistoricalAnalysisResult:
    """Complete historical analysis result."""

    observations: list[HistoricalObservation]

    metrics: dict[str, HistoricalMetricResult]

    period_frequency: PeriodFrequency

    overall_trend: HistoricalTrend

    signals: list[HistoricalSignal]

    summary: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def period_count(self) -> int:
        return len(self.observations)

    @property
    def latest_period(self) -> date | None:
        if not self.observations:
            return None

        return self.observations[-1].period

    @property
    def earliest_period(self) -> date | None:
        if not self.observations:
            return None

        return self.observations[0].period

    def metric(
        self,
        name: str,
    ) -> HistoricalMetricResult | None:
        return self.metrics.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_count": self.period_count,
            "earliest_period": (
                self.earliest_period.isoformat()
                if self.earliest_period
                else None
            ),
            "latest_period": (
                self.latest_period.isoformat()
                if self.latest_period
                else None
            ),
            "period_frequency": self.period_frequency.value,
            "overall_trend": self.overall_trend.value,
            "metrics": {
                name: result.to_dict()
                for name, result in self.metrics.items()
            },
            "signals": [
                signal.to_dict()
                for signal in self.signals
            ],
            "summary": self.summary,
            "metadata": self.metadata,
        }


# ============================================================================
# Historical Analyzer
# ============================================================================


class HistoricalAnalyzer:
    """
    Stateless historical financial analysis engine.

    The analyzer works with period-level observations and does not access
    databases or external services.
    """

    DEFAULT_METRICS = (
        "revenue",
        "gross_profit",
        "operating_profit",
        "net_profit",
        "operating_expenses",
        "assets",
        "liabilities",
        "equity",
        "debt",
        "cash",
        "receivables",
        "inventory",
        "payables",
        "operating_cash_flow",
        "free_cash_flow",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "current_ratio",
        "debt_to_equity",
    )

    def __init__(
        self,
        *,
        moving_average_window: int = 3,
        growth_threshold: Decimal | float = Decimal("5"),
        strong_growth_threshold: Decimal | float = Decimal("15"),
        decline_threshold: Decimal | float = Decimal("-5"),
        strong_decline_threshold: Decimal | float = Decimal("-15"),
        volatility_threshold: Decimal | float = Decimal("20"),
    ) -> None:
        if moving_average_window < 1:
            raise ValueError(
                "moving_average_window must be >= 1"
            )

        self.moving_average_window = moving_average_window
        self.growth_threshold = to_decimal(
            growth_threshold
        )
        self.strong_growth_threshold = to_decimal(
            strong_growth_threshold
        )
        self.decline_threshold = to_decimal(
            decline_threshold
        )
        self.strong_decline_threshold = to_decimal(
            strong_decline_threshold
        )
        self.volatility_threshold = to_decimal(
            volatility_threshold
        )

    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        observations: Iterable[
            HistoricalObservation | Mapping[str, Any]
        ],
        *,
        metrics: Sequence[str] | None = None,
        frequency: PeriodFrequency = PeriodFrequency.UNKNOWN,
    ) -> HistoricalAnalysisResult:
        """
        Analyze a complete historical time series.
        """
        normalized = self.normalize_observations(
            observations
        )

        selected_metrics = tuple(
            metrics or self.DEFAULT_METRICS
        )

        if not normalized:
            return HistoricalAnalysisResult(
                observations=[],
                metrics={},
                period_frequency=frequency,
                overall_trend=HistoricalTrend.INSUFFICIENT_DATA,
                signals=[],
                summary="Insufficient historical data.",
            )

        metric_results: dict[str, HistoricalMetricResult] = {}
        all_signals: list[HistoricalSignal] = []

        for metric in selected_metrics:
            result = self.analyze_metric(
                normalized,
                metric,
            )

            if result is None:
                continue

            metric_results[metric] = result
            all_signals.extend(result.signals)

        overall_trend = self.determine_overall_trend(
            metric_results
        )

        summary = self.generate_summary(
            normalized,
            metric_results,
            overall_trend,
        )

        return HistoricalAnalysisResult(
            observations=normalized,
            metrics=metric_results,
            period_frequency=frequency,
            overall_trend=overall_trend,
            signals=all_signals,
            summary=summary,
            metadata={
                "moving_average_window": (
                    self.moving_average_window
                ),
                "growth_threshold": str(
                    self.growth_threshold
                ),
                "decline_threshold": str(
                    self.decline_threshold
                ),
                "volatility_threshold": str(
                    self.volatility_threshold
                ),
            },
        )

    # ------------------------------------------------------------------
    # Observation Management
    # ------------------------------------------------------------------

    def normalize_observations(
        self,
        observations: Iterable[
            HistoricalObservation | Mapping[str, Any]
        ],
    ) -> list[HistoricalObservation]:
        """
        Normalize and sort observations chronologically.
        """
        result: list[HistoricalObservation] = []

        for item in observations:
            observation = self._normalize_observation(
                item
            )

            if observation is None:
                continue

            observation.derive_metrics()
            result.append(observation)

        # Keep the last observation for duplicate periods.
        by_period: dict[date, HistoricalObservation] = {}

        for observation in result:
            by_period[observation.period] = observation

        return sorted(
            by_period.values(),
            key=lambda item: item.period,
        )

    def add_observation(
        self,
        observations: Sequence[HistoricalObservation],
        observation: HistoricalObservation,
    ) -> list[HistoricalObservation]:
        """
        Add one observation and return a chronologically sorted series.
        """
        combined = list(observations)
        combined.append(observation)

        return self.normalize_observations(
            combined
        )

    # ------------------------------------------------------------------
    # Metric Analysis
    # ------------------------------------------------------------------

    def analyze_metric(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
    ) -> HistoricalMetricResult | None:
        """
        Analyze one metric over time.
        """
        if not observations:
            return None

        values: list[Decimal] = []

        valid_observations: list[
            HistoricalObservation
        ] = []

        for observation in observations:
            value = observation.metric_value(metric)

            if value is None:
                continue

            values.append(value)
            valid_observations.append(observation)

        if not values:
            return None

        changes = self.calculate_changes(
            valid_observations,
            metric,
        )

        moving_average = self.calculate_moving_average(
            values,
            window=self.moving_average_window,
        )

        average = self.average(values)
        median = self.median(values)
        minimum = min(values)
        maximum = max(values)

        volatility = self.calculate_volatility(
            values
        )

        trend = self.determine_trend(
            values,
            changes,
            volatility,
        )

        latest_value = values[-1]

        previous_value = (
            values[-2]
            if len(values) >= 2
            else None
        )

        latest_growth = (
            percentage_change(
                latest_value,
                previous_value,
            )
            if previous_value is not None
            else None
        )

        signals = self.detect_metric_signals(
            valid_observations,
            metric,
            values,
            changes,
            average,
            volatility,
        )

        return HistoricalMetricResult(
            metric=metric,
            observations=list(valid_observations),
            values=values,
            changes=changes,
            moving_average=moving_average,
            trend=trend,
            average=average,
            median=median,
            minimum=minimum,
            maximum=maximum,
            volatility=volatility,
            latest_value=latest_value,
            previous_value=previous_value,
            latest_growth=latest_growth,
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def calculate_changes(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
    ) -> list[HistoricalComparison]:
        """
        Calculate period-over-period changes.
        """
        comparisons: list[HistoricalComparison] = []

        if len(observations) < 2:
            return comparisons

        for index in range(1, len(observations)):
            previous = observations[index - 1]
            current = observations[index]

            current_value = current.metric_value(metric)
            previous_value = previous.metric_value(metric)

            if (
                current_value is None
                or previous_value is None
            ):
                continue

            absolute = current_value - previous_value

            growth = percentage_change(
                current_value,
                previous_value,
            )

            if absolute > ZERO:
                direction = ComparisonDirection.INCREASE
            elif absolute < ZERO:
                direction = ComparisonDirection.DECREASE
            else:
                direction = ComparisonDirection.UNCHANGED

            comparisons.append(
                HistoricalComparison(
                    metric=metric,
                    current_period=current.period,
                    previous_period=previous.period,
                    current_value=current_value,
                    previous_value=previous_value,
                    absolute_change=round_money(
                        absolute
                    ),
                    percentage_change=growth,
                    direction=direction,
                )
            )

        return comparisons

    def compare_periods(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
        current_index: int = -1,
        previous_index: int = -2,
    ) -> HistoricalComparison | None:
        """
        Compare two arbitrary observations.
        """
        if len(observations) < 2:
            return None

        try:
            current = observations[current_index]
            previous = observations[previous_index]
        except IndexError:
            return None

        current_value = current.metric_value(metric)
        previous_value = previous.metric_value(metric)

        if (
            current_value is None
            or previous_value is None
        ):
            return None

        absolute = current_value - previous_value

        if absolute > ZERO:
            direction = ComparisonDirection.INCREASE
        elif absolute < ZERO:
            direction = ComparisonDirection.DECREASE
        else:
            direction = ComparisonDirection.UNCHANGED

        return HistoricalComparison(
            metric=metric,
            current_period=current.period,
            previous_period=previous.period,
            current_value=current_value,
            previous_value=previous_value,
            absolute_change=round_money(absolute),
            percentage_change=percentage_change(
                current_value,
                previous_value,
            ),
            direction=direction,
        )

    # ------------------------------------------------------------------
    # Moving Averages
    # ------------------------------------------------------------------

    def calculate_moving_average(
        self,
        values: Sequence[Decimal],
        *,
        window: int | None = None,
    ) -> list[Decimal]:
        """
        Calculate trailing moving averages.

        The first values use the available observations rather than
        returning nulls.
        """
        if not values:
            return []

        effective_window = (
            window or self.moving_average_window
        )

        if effective_window < 1:
            raise ValueError(
                "Moving-average window must be >= 1."
            )

        result: list[Decimal] = []

        for index in range(len(values)):
            start = max(
                0,
                index - effective_window + 1,
            )

            window_values = values[start:index + 1]

            average = (
                sum(window_values, ZERO)
                / Decimal(len(window_values))
            )

            result.append(
                round_money(average)
            )

        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def average(
        values: Sequence[Decimal],
    ) -> Decimal:
        """Calculate arithmetic mean."""
        if not values:
            return ZERO

        return round_money(
            sum(values, ZERO)
            / Decimal(len(values))
        )

    @staticmethod
    def median(
        values: Sequence[Decimal],
    ) -> Decimal:
        """Calculate median."""
        if not values:
            return ZERO

        ordered = sorted(values)
        count = len(ordered)
        middle = count // 2

        if count % 2:
            return round_money(
                ordered[middle]
            )

        return round_money(
            (
                ordered[middle - 1]
                + ordered[middle]
            )
            / Decimal("2")
        )

    @staticmethod
    def calculate_volatility(
        values: Sequence[Decimal],
    ) -> Decimal:
        """
        Calculate coefficient of variation as a percentage.

        Uses absolute mean to support series that may contain negative
        values, such as net profit.
        """
        if len(values) < 2:
            return ZERO

        numeric_values = [
            float(value)
            for value in values
        ]

        average_value = mean(
            numeric_values
        )

        standard_deviation = pstdev(
            numeric_values
        )

        if average_value == 0:
            return ZERO

        return round_percent(
            Decimal(
                str(
                    (
                        standard_deviation
                        / abs(average_value)
                    )
                    * 100
                )
            )
        )

    # ------------------------------------------------------------------
    # Trend Detection
    # ------------------------------------------------------------------

    def determine_trend(
        self,
        values: Sequence[Decimal],
        changes: Sequence[HistoricalComparison],
        volatility: Decimal,
    ) -> HistoricalTrend:
        """
        Determine historical trend using recent growth and volatility.
        """
        if len(values) < 2:
            return HistoricalTrend.INSUFFICIENT_DATA

        if volatility >= self.volatility_threshold:
            return HistoricalTrend.VOLATILE

        recent_changes = [
            change.percentage_change
            for change in changes[-3:]
        ]

        if not recent_changes:
            return HistoricalTrend.STABLE

        average_growth = (
            sum(recent_changes, ZERO)
            / Decimal(len(recent_changes))
        )

        if (
            average_growth
            >= self.strong_growth_threshold
        ):
            return HistoricalTrend.STRONG_GROWTH

        if average_growth >= self.growth_threshold:
            return HistoricalTrend.GROWTH

        if (
            average_growth
            <= self.strong_decline_threshold
        ):
            return HistoricalTrend.STRONG_DECLINE

        if average_growth <= self.decline_threshold:
            return HistoricalTrend.DECLINE

        return HistoricalTrend.STABLE

    def determine_overall_trend(
        self,
        metric_results: Mapping[
            str,
            HistoricalMetricResult,
        ],
    ) -> HistoricalTrend:
        """
        Determine the overall business trend from major financial metrics.

        Revenue and net profit receive the greatest conceptual importance.
        """
        if not metric_results:
            return HistoricalTrend.INSUFFICIENT_DATA

        weighted_trends: list[
            tuple[HistoricalTrend, Decimal]
        ] = []

        importance = {
            "revenue": Decimal("0.30"),
            "net_profit": Decimal("0.30"),
            "operating_cash_flow": Decimal("0.20"),
            "gross_margin": Decimal("0.10"),
            "operating_margin": Decimal("0.10"),
        }

        for metric, weight in importance.items():
            result = metric_results.get(metric)

            if result is None:
                continue

            weighted_trends.append(
                (result.trend, weight)
            )

        if not weighted_trends:
            return HistoricalTrend.INSUFFICIENT_DATA

        score = ZERO

        trend_values = {
            HistoricalTrend.STRONG_GROWTH: Decimal("2"),
            HistoricalTrend.GROWTH: Decimal("1"),
            HistoricalTrend.STABLE: ZERO,
            HistoricalTrend.DECLINE: Decimal("-1"),
            HistoricalTrend.STRONG_DECLINE: Decimal("-2"),
            HistoricalTrend.VOLATILE: Decimal("0"),
            HistoricalTrend.INSUFFICIENT_DATA: ZERO,
        }

        total_weight = ZERO

        for trend, weight in weighted_trends:
            score += (
                trend_values[trend]
                * weight
            )
            total_weight += weight

        if total_weight == ZERO:
            return HistoricalTrend.INSUFFICIENT_DATA

        normalized = score / total_weight

        if normalized >= Decimal("1.5"):
            return HistoricalTrend.STRONG_GROWTH

        if normalized >= Decimal("0.5"):
            return HistoricalTrend.GROWTH

        if normalized <= Decimal("-1.5"):
            return HistoricalTrend.STRONG_DECLINE

        if normalized <= Decimal("-0.5"):
            return HistoricalTrend.DECLINE

        if any(
            trend == HistoricalTrend.VOLATILE
            for trend, _ in weighted_trends
        ):
            return HistoricalTrend.VOLATILE

        return HistoricalTrend.STABLE

    # ------------------------------------------------------------------
    # Signal Detection
    # ------------------------------------------------------------------

    def detect_metric_signals(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
        values: Sequence[Decimal],
        changes: Sequence[HistoricalComparison],
        average: Decimal,
        volatility: Decimal,
    ) -> list[HistoricalSignal]:
        """Detect historical patterns for a metric."""
        signals: list[HistoricalSignal] = []

        if not observations or not values:
            return signals

        # --------------------------------------------------------------
        # Growth / Decline
        # --------------------------------------------------------------

        for change in changes:
            if (
                change.percentage_change
                >= self.strong_growth_threshold
            ):
                signals.append(
                    HistoricalSignal(
                        signal_type=(
                            HistoricalSignalType.GROWTH
                        ),
                        metric=metric,
                        period=change.current_period,
                        severity="high",
                        description=(
                            f"{metric} increased by "
                            f"{change.percentage_change:.2f}%."
                        ),
                        value=change.current_value,
                        reference_value=change.previous_value,
                        percentage_difference=(
                            change.percentage_change
                        ),
                        evidence=[
                            (
                                f"Previous: "
                                f"{change.previous_value:.2f}"
                            ),
                            (
                                f"Current: "
                                f"{change.current_value:.2f}"
                            ),
                        ],
                    )
                )

            elif (
                change.percentage_change
                <= self.strong_decline_threshold
            ):
                signals.append(
                    HistoricalSignal(
                        signal_type=(
                            HistoricalSignalType.DECLINE
                        ),
                        metric=metric,
                        period=change.current_period,
                        severity="high",
                        description=(
                            f"{metric} decreased by "
                            f"{abs(change.percentage_change):.2f}%."
                        ),
                        value=change.current_value,
                        reference_value=change.previous_value,
                        percentage_difference=(
                            change.percentage_change
                        ),
                        evidence=[
                            (
                                f"Previous: "
                                f"{change.previous_value:.2f}"
                            ),
                            (
                                f"Current: "
                                f"{change.current_value:.2f}"
                            ),
                        ],
                    )
                )

        # --------------------------------------------------------------
        # Peak
        # --------------------------------------------------------------

        maximum = max(values)
        max_index = values.index(maximum)

        if max_index == len(values) - 1:
            signals.append(
                HistoricalSignal(
                    signal_type=HistoricalSignalType.PEAK,
                    metric=metric,
                    period=observations[max_index].period,
                    severity="info",
                    description=(
                        f"{metric} reached its historical high "
                        f"of {maximum:.2f}."
                    ),
                    value=maximum,
                    evidence=[
                        (
                            f"Historical maximum = "
                            f"{maximum:.2f}"
                        )
                    ],
                )
            )

        # --------------------------------------------------------------
        # Trough
        # --------------------------------------------------------------

        minimum = min(values)
        min_index = values.index(minimum)

        if min_index == len(values) - 1:
            signals.append(
                HistoricalSignal(
                    signal_type=HistoricalSignalType.TROUGH,
                    metric=metric,
                    period=observations[min_index].period,
                    severity="medium",
                    description=(
                        f"{metric} reached its historical low "
                        f"of {minimum:.2f}."
                    ),
                    value=minimum,
                    evidence=[
                        (
                            f"Historical minimum = "
                            f"{minimum:.2f}"
                        )
                    ],
                )
            )

        # --------------------------------------------------------------
        # Volatility
        # --------------------------------------------------------------

        if volatility >= self.volatility_threshold:
            latest = values[-1]

            signals.append(
                HistoricalSignal(
                    signal_type=(
                        HistoricalSignalType.VOLATILITY
                    ),
                    metric=metric,
                    period=observations[-1].period,
                    severity="high",
                    description=(
                        f"{metric} is historically volatile "
                        f"with volatility of "
                        f"{volatility:.2f}%."
                    ),
                    value=latest,
                    reference_value=average,
                    percentage_difference=(
                        round_percent(
                            safe_divide(
                                latest - average,
                                abs(average),
                            )
                            * ONE_HUNDRED
                        )
                        if average != ZERO
                        else ZERO
                    ),
                    evidence=[
                        f"Average = {average:.2f}",
                        f"Volatility = {volatility:.2f}%",
                    ],
                )
            )

        # --------------------------------------------------------------
        # Baseline Deviation
        # --------------------------------------------------------------

        latest = values[-1]

        if average != ZERO:
            deviation = round_percent(
                (
                    (latest - average)
                    / abs(average)
                )
                * ONE_HUNDRED
            )

            if abs(deviation) >= Decimal("20"):
                signals.append(
                    HistoricalSignal(
                        signal_type=(
                            HistoricalSignalType.BASELINE_DEVIATION
                        ),
                        metric=metric,
                        period=observations[-1].period,
                        severity="medium",
                        description=(
                            f"Latest {metric} is "
                            f"{abs(deviation):.2f}% "
                            f"{'above' if deviation > 0 else 'below'} "
                            f"its historical average."
                        ),
                        value=latest,
                        reference_value=average,
                        percentage_difference=deviation,
                        evidence=[
                            f"Latest = {latest:.2f}",
                            f"Historical average = {average:.2f}",
                        ],
                    )
                )

        return signals

    # ------------------------------------------------------------------
    # Baseline / Benchmark Analysis
    # ------------------------------------------------------------------

    def baseline(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
        *,
        periods: int | None = None,
    ) -> Decimal | None:
        """
        Calculate historical baseline.

        If periods is supplied, only the most recent N periods are used.
        """
        values = [
            observation.metric_value(metric)
            for observation in observations
        ]

        values = [
            value
            for value in values
            if value is not None
        ]

        if periods is not None:
            values = values[-periods:]

        if not values:
            return None

        return self.average(values)

    def deviation_from_baseline(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
        *,
        periods: int | None = None,
    ) -> Decimal | None:
        """Calculate latest value deviation from historical baseline."""
        if not observations:
            return None

        latest = observations[-1].metric_value(metric)

        if latest is None:
            return None

        baseline = self.baseline(
            observations,
            metric,
            periods=periods,
        )

        if baseline is None or baseline == ZERO:
            return None

        return round_percent(
            (
                (latest - baseline)
                / abs(baseline)
            )
            * ONE_HUNDRED
        )

    # ------------------------------------------------------------------
    # Acceleration / Deceleration
    # ------------------------------------------------------------------

    def calculate_growth_series(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
    ) -> list[Decimal]:
        """Return a period-by-period percentage growth series."""
        changes = self.calculate_changes(
            observations,
            metric,
        )

        return [
            change.percentage_change
            for change in changes
        ]

    def detect_growth_acceleration(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
    ) -> list[HistoricalSignal]:
        """
        Detect acceleration or deceleration in growth.
        """
        growth = self.calculate_growth_series(
            observations,
            metric,
        )

        signals: list[HistoricalSignal] = []

        if len(growth) < 2:
            return signals

        for index in range(1, len(growth)):
            previous_growth = growth[index - 1]
            current_growth = growth[index]

            change = current_growth - previous_growth

            period_index = index + 1

            if period_index >= len(observations):
                continue

            period = observations[
                period_index
            ].period

            if change >= Decimal("10"):
                signals.append(
                    HistoricalSignal(
                        signal_type=(
                            HistoricalSignalType.ACCELERATION
                        ),
                        metric=metric,
                        period=period,
                        severity="medium",
                        description=(
                            f"{metric} growth accelerated by "
                            f"{change:.2f} percentage points."
                        ),
                        value=current_growth,
                        reference_value=previous_growth,
                        percentage_difference=change,
                        evidence=[
                            (
                                f"Previous growth = "
                                f"{previous_growth:.2f}%"
                            ),
                            (
                                f"Current growth = "
                                f"{current_growth:.2f}%"
                            ),
                        ],
                    )
                )

            elif change <= Decimal("-10"):
                signals.append(
                    HistoricalSignal(
                        signal_type=(
                            HistoricalSignalType.DECELERATION
                        ),
                        metric=metric,
                        period=period,
                        severity="medium",
                        description=(
                            f"{metric} growth decelerated by "
                            f"{abs(change):.2f} percentage points."
                        ),
                        value=current_growth,
                        reference_value=previous_growth,
                        percentage_difference=change,
                        evidence=[
                            (
                                f"Previous growth = "
                                f"{previous_growth:.2f}%"
                            ),
                            (
                                f"Current growth = "
                                f"{current_growth:.2f}%"
                            ),
                        ],
                    )
                )

        return signals

    # ------------------------------------------------------------------
    # High / Low Analysis
    # ------------------------------------------------------------------

    def historical_high(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
    ) -> tuple[date, Decimal] | None:
        """Return the date and value of the historical high."""
        values: list[
            tuple[date, Decimal]
        ] = []

        for observation in observations:
            value = observation.metric_value(metric)

            if value is not None:
                values.append(
                    (observation.period, value)
                )

        if not values:
            return None

        return max(
            values,
            key=lambda item: item[1],
        )

    def historical_low(
        self,
        observations: Sequence[HistoricalObservation],
        metric: str,
    ) -> tuple[date, Decimal] | None:
        """Return the date and value of the historical low."""
        values: list[
            tuple[date, Decimal]
        ] = []

        for observation in observations:
            value = observation.metric_value(metric)

            if value is not None:
                values.append(
                    (observation.period, value)
                )

        if not values:
            return None

        return min(
            values,
            key=lambda item: item[1],
        )

    # ------------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------------

    @staticmethod
    def infer_frequency(
        observations: Sequence[HistoricalObservation],
    ) -> PeriodFrequency:
        """Infer observation frequency from period spacing."""
        if len(observations) < 2:
            return PeriodFrequency.UNKNOWN

        deltas = [
            (
                observations[index].period
                - observations[index - 1].period
            ).days
            for index in range(1, len(observations))
        ]

        if not deltas:
            return PeriodFrequency.UNKNOWN

        average_days = sum(deltas) / len(deltas)

        if average_days <= 2:
            return PeriodFrequency.DAILY

        if average_days <= 10:
            return PeriodFrequency.WEEKLY

        if average_days <= 45:
            return PeriodFrequency.MONTHLY

        if average_days <= 120:
            return PeriodFrequency.QUARTERLY

        if average_days <= 400:
            return PeriodFrequency.YEARLY

        return PeriodFrequency.UNKNOWN

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_summary(
        self,
        observations: Sequence[HistoricalObservation],
        metrics: Mapping[
            str,
            HistoricalMetricResult,
        ],
        overall_trend: HistoricalTrend,
    ) -> str:
        """Generate a concise historical performance summary."""
        if not observations:
            return "Insufficient historical data."

        parts = [
            (
                f"Historical analysis covers "
                f"{len(observations)} periods from "
                f"{observations[0].period.isoformat()} to "
                f"{observations[-1].period.isoformat()}."
            ),
            (
                f"Overall trend: "
                f"{overall_trend.value.replace('_', ' ')}."
            ),
        ]

        revenue = metrics.get("revenue")
        profit = metrics.get("net_profit")
        cash_flow = metrics.get(
            "operating_cash_flow"
        )

        if revenue and revenue.latest_growth is not None:
            parts.append(
                f"Latest revenue growth: "
                f"{revenue.latest_growth:.2f}%."
            )

        if profit and profit.latest_growth is not None:
            parts.append(
                f"Latest net-profit growth: "
                f"{profit.latest_growth:.2f}%."
            )

        if (
            cash_flow
            and cash_flow.latest_value is not None
        ):
            parts.append(
                f"Latest operating cash flow: "
                f"{cash_flow.latest_value:.2f}."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _normalize_observation(
        self,
        item: HistoricalObservation | Mapping[str, Any],
    ) -> HistoricalObservation | None:
        if isinstance(
            item,
            HistoricalObservation,
        ):
            return item

        if not isinstance(item, Mapping):
            return None

        period = normalize_date(
            item.get("period")
            or item.get("date")
            or item.get("period_end")
            or item.get("as_of_date")
        )

        if period is None:
            return None

        observation = HistoricalObservation(
            period=period,
            revenue=to_decimal(
                item.get("revenue")
            ),
            gross_profit=to_decimal(
                item.get("gross_profit")
            ),
            operating_profit=to_decimal(
                item.get("operating_profit")
            ),
            net_profit=to_decimal(
                item.get("net_profit")
            ),
            operating_expenses=to_decimal(
                item.get("operating_expenses")
            ),
            assets=to_decimal(
                item.get("assets")
                or item.get("total_assets")
            ),
            liabilities=to_decimal(
                item.get("liabilities")
                or item.get("total_liabilities")
            ),
            equity=to_decimal(
                item.get("equity")
                or item.get("total_equity")
            ),
            debt=to_decimal(
                item.get("debt")
                or item.get("total_debt")
            ),
            cash=to_decimal(
                item.get("cash")
            ),
            receivables=to_decimal(
                item.get("receivables")
            ),
            inventory=to_decimal(
                item.get("inventory")
            ),
            payables=to_decimal(
                item.get("payables")
            ),
            operating_cash_flow=to_decimal(
                item.get("operating_cash_flow")
            ),
            investing_cash_flow=to_decimal(
                item.get("investing_cash_flow")
            ),
            financing_cash_flow=to_decimal(
                item.get("financing_cash_flow")
            ),
            free_cash_flow=(
                to_decimal(
                    item.get("free_cash_flow")
                )
                if item.get("free_cash_flow")
                is not None
                else None
            ),
            gross_margin=(
                to_decimal(
                    item.get("gross_margin")
                )
                if item.get("gross_margin")
                is not None
                else None
            ),
            operating_margin=(
                to_decimal(
                    item.get("operating_margin")
                )
                if item.get("operating_margin")
                is not None
                else None
            ),
            net_margin=(
                to_decimal(
                    item.get("net_margin")
                )
                if item.get("net_margin")
                is not None
                else None
            ),
            current_ratio=(
                to_decimal(
                    item.get("current_ratio")
                )
                if item.get("current_ratio")
                is not None
                else None
            ),
            debt_to_equity=(
                to_decimal(
                    item.get("debt_to_equity")
                )
                if item.get("debt_to_equity")
                is not None
                else None
            ),
            currency=item.get("currency"),
            metadata=dict(
                item.get("metadata") or {}
            ),
        )

        return observation


# ============================================================================
# Convenience Functions
# ============================================================================


def analyze_historical_financials(
    observations: Iterable[
        HistoricalObservation | Mapping[str, Any]
    ],
    *,
    metrics: Sequence[str] | None = None,
    frequency: PeriodFrequency = PeriodFrequency.UNKNOWN,
    moving_average_window: int = 3,
) -> HistoricalAnalysisResult:
    """
    Convenience wrapper for historical analysis.
    """
    analyzer = HistoricalAnalyzer(
        moving_average_window=moving_average_window,
    )

    if frequency == PeriodFrequency.UNKNOWN:
        normalized = analyzer.normalize_observations(
            observations
        )

        frequency = analyzer.infer_frequency(
            normalized
        )

        return analyzer.analyze(
            normalized,
            metrics=metrics,
            frequency=frequency,
        )

    return analyzer.analyze(
        observations,
        metrics=metrics,
        frequency=frequency,
    )


def calculate_historical_growth(
    observations: Sequence[HistoricalObservation],
    metric: str,
) -> list[HistoricalComparison]:
    """Calculate period-over-period historical growth."""
    analyzer = HistoricalAnalyzer()

    normalized = analyzer.normalize_observations(
        observations
    )

    return analyzer.calculate_changes(
        normalized,
        metric,
    )


def calculate_moving_average(
    values: Sequence[Decimal | float | int],
    window: int = 3,
) -> list[Decimal]:
    """Calculate a moving average over numeric values."""
    analyzer = HistoricalAnalyzer(
        moving_average_window=window
    )

    normalized = [
        to_decimal(value)
        for value in values
    ]

    return analyzer.calculate_moving_average(
        normalized,
        window=window,
    )


def calculate_historical_volatility(
    values: Sequence[Decimal | float | int],
) -> Decimal:
    """Calculate coefficient-of-variation volatility."""
    normalized = [
        to_decimal(value)
        for value in values
    ]

    return HistoricalAnalyzer.calculate_volatility(
        normalized
    )


def detect_historical_trend(
    observations: Sequence[HistoricalObservation],
    metric: str,
) -> HistoricalTrend:
    """Determine the historical trend for a metric."""
    analyzer = HistoricalAnalyzer()

    normalized = analyzer.normalize_observations(
        observations
    )

    result = analyzer.analyze_metric(
        normalized,
        metric,
    )

    if result is None:
        return HistoricalTrend.INSUFFICIENT_DATA

    return result.trend


def historical_high(
    observations: Sequence[HistoricalObservation],
    metric: str,
) -> tuple[date, Decimal] | None:
    """Return historical high."""
    analyzer = HistoricalAnalyzer()

    normalized = analyzer.normalize_observations(
        observations
    )

    return analyzer.historical_high(
        normalized,
        metric,
    )


def historical_low(
    observations: Sequence[HistoricalObservation],
    metric: str,
) -> tuple[date, Decimal] | None:
    """Return historical low."""
    analyzer = HistoricalAnalyzer()

    normalized = analyzer.normalize_observations(
        observations
    )

    return analyzer.historical_low(
        normalized,
        metric,
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "HistoricalTrend",
    "ComparisonDirection",
    "PeriodFrequency",
    "HistoricalSignalType",
    "HistoricalObservation",
    "HistoricalComparison",
    "HistoricalSignal",
    "HistoricalMetricResult",
    "HistoricalAnalysisResult",
    "HistoricalAnalyzer",
    "analyze_historical_financials",
    "calculate_historical_growth",
    "calculate_moving_average",
    "calculate_historical_volatility",
    "detect_historical_trend",
    "historical_high",
    "historical_low",
]