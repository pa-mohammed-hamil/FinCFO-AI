"""
FinCo AI - Financial Comparison Engine

Provides reusable financial comparison logic for:

    - Current vs previous period
    - Current vs same period last year
    - Actual vs budget
    - Actual vs forecast
    - Company vs benchmark
    - Metric-level variance analysis
    - Trend detection
    - Significant-change detection
    - Alert-engine inputs
    - Agentic AI financial analysis

This module contains business logic only.
Database, API, and persistence concerns belong in their respective layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Iterable, Mapping, Optional


ZERO = Decimal("0")
HUNDRED = Decimal("100")


NumberLike = Decimal | int | float | str


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------

def to_decimal(value: NumberLike | None) -> Decimal:
    """Safely convert a numeric value to Decimal."""

    if value is None:
        return ZERO

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric value: {value!r}") from exc


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ComparisonType(str, Enum):
    """Supported financial comparison types."""

    PERIOD_OVER_PERIOD = "period_over_period"
    YEAR_OVER_YEAR = "year_over_year"
    ACTUAL_VS_BUDGET = "actual_vs_budget"
    ACTUAL_VS_FORECAST = "actual_vs_forecast"
    BENCHMARK = "benchmark"


class Trend(str, Enum):
    """Directional trend."""

    INCREASE = "increase"
    DECREASE = "decrease"
    STABLE = "stable"


class VarianceStatus(str, Enum):
    """Magnitude classification for a financial variance."""

    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparisonInput:
    """
    Input for comparing one financial metric.

    Example:

        ComparisonInput(
            metric="Revenue",
            current=120000,
            previous=100000,
            higher_is_better=True,
        )
    """

    metric: str
    current: NumberLike
    previous: NumberLike

    higher_is_better: bool = True

    # Optional materiality threshold.
    # Example: Decimal("10") means changes >= 10% are significant.
    significance_threshold_percent: NumberLike = Decimal("10")


@dataclass(frozen=True)
class ComparisonResult:
    """Result of comparing one financial metric."""

    metric: str

    current: Decimal
    previous: Decimal

    absolute_change: Decimal
    percentage_change: Optional[Decimal]

    trend: Trend
    status: VarianceStatus

    is_significant: bool

    direction: str

    explanation: str


@dataclass(frozen=True)
class BudgetVarianceResult:
    """Actual vs budget comparison."""

    metric: str

    actual: Decimal
    budget: Decimal

    variance: Decimal
    variance_percent: Optional[Decimal]

    status: VarianceStatus
    explanation: str


@dataclass(frozen=True)
class BenchmarkResult:
    """Company metric vs benchmark."""

    metric: str

    company_value: Decimal
    benchmark_value: Decimal

    difference: Decimal
    difference_percent: Optional[Decimal]

    status: VarianceStatus
    explanation: str


@dataclass(frozen=True)
class TrendResult:
    """Multi-period trend analysis."""

    metric: str

    values: tuple[Decimal, ...]

    direction: Trend

    first_value: Decimal
    latest_value: Decimal

    absolute_change: Decimal
    percentage_change: Optional[Decimal]

    periods_increasing: int
    periods_decreasing: int

    consistency: Optional[Decimal]

    explanation: str


@dataclass(frozen=True)
class ComparisonSummary:
    """Aggregated comparison results."""

    total_metrics: int
    significant_changes: int

    favorable_changes: int
    unfavorable_changes: int
    neutral_changes: int

    increasing_metrics: int
    decreasing_metrics: int
    stable_metrics: int

    overall_status: VarianceStatus


# ---------------------------------------------------------------------------
# Core comparison engine
# ---------------------------------------------------------------------------

class FinancialComparisonEngine:
    """
    Reusable financial comparison engine.

    Stateless and safe to use across API requests, agents,
    alert rules, reports, and ML pipelines.
    """

    # ------------------------------------------------------------------
    # Basic comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare(
        data: ComparisonInput,
    ) -> ComparisonResult:
        """Compare current value against a previous/reference value."""

        current = to_decimal(data.current)
        previous = to_decimal(data.previous)

        absolute_change = current - previous

        percentage_change = FinancialComparisonEngine.percentage_change(
            current,
            previous,
        )

        trend = FinancialComparisonEngine.detect_direction(
            absolute_change,
        )

        significance_threshold = abs(
            to_decimal(data.significance_threshold_percent)
        )

        is_significant = (
            percentage_change is not None
            and abs(percentage_change) >= significance_threshold
        )

        status = FinancialComparisonEngine.determine_variance_status(
            change=absolute_change,
            higher_is_better=data.higher_is_better,
        )

        direction = FinancialComparisonEngine.direction_label(
            trend=trend,
            percentage_change=percentage_change,
        )

        explanation = FinancialComparisonEngine.generate_explanation(
            metric=data.metric,
            current=current,
            previous=previous,
            absolute_change=absolute_change,
            percentage_change=percentage_change,
            trend=trend,
            status=status,
        )

        return ComparisonResult(
            metric=data.metric,
            current=current,
            previous=previous,
            absolute_change=absolute_change,
            percentage_change=percentage_change,
            trend=trend,
            status=status,
            is_significant=is_significant,
            direction=direction,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Percentage calculations
    # ------------------------------------------------------------------

    @staticmethod
    def percentage_change(
        current: NumberLike,
        previous: NumberLike,
    ) -> Optional[Decimal]:
        """
        Calculate percentage change.

            ((current - previous) / abs(previous)) * 100

        Returns None when the reference value is zero.
        """

        current = to_decimal(current)
        previous = to_decimal(previous)

        if previous == ZERO:
            return None

        return (
            (current - previous)
            / abs(previous)
        ) * HUNDRED

    @staticmethod
    def absolute_change(
        current: NumberLike,
        previous: NumberLike,
    ) -> Decimal:
        """Calculate absolute change."""

        return to_decimal(current) - to_decimal(previous)

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    @staticmethod
    def detect_direction(
        change: NumberLike,
        tolerance: NumberLike = Decimal("0"),
    ) -> Trend:
        """Determine whether a metric increased, decreased, or stayed stable."""

        change = to_decimal(change)
        tolerance = abs(to_decimal(tolerance))

        if abs(change) <= tolerance:
            return Trend.STABLE

        if change > ZERO:
            return Trend.INCREASE

        return Trend.DECREASE

    # ------------------------------------------------------------------
    # Favorability
    # ------------------------------------------------------------------

    @staticmethod
    def determine_variance_status(
        change: NumberLike,
        higher_is_better: bool = True,
    ) -> VarianceStatus:
        """
        Determine whether a change is favorable.

        For revenue/profit/cash:
            Higher is generally favorable.

        For expenses/debt/burn:
            Higher is generally unfavorable.
        """

        change = to_decimal(change)

        if change == ZERO:
            return VarianceStatus.NEUTRAL

        if higher_is_better:
            if change > ZERO:
                return VarianceStatus.FAVORABLE

            return VarianceStatus.UNFAVORABLE

        if change < ZERO:
            return VarianceStatus.FAVORABLE

        return VarianceStatus.UNFAVORABLE

    # ------------------------------------------------------------------
    # Budget comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_to_budget(
        metric: str,
        actual: NumberLike,
        budget: NumberLike,
        higher_is_better: bool = True,
    ) -> BudgetVarianceResult:
        """
        Compare actual financial performance against budget.

        Positive variance:
            Actual > Budget

        Negative variance:
            Actual < Budget
        """

        actual = to_decimal(actual)
        budget = to_decimal(budget)

        variance = actual - budget

        variance_percent = FinancialComparisonEngine.percentage_change(
            actual,
            budget,
        )

        status = FinancialComparisonEngine.determine_variance_status(
            variance,
            higher_is_better,
        )

        if variance > ZERO:
            direction = "above"
        elif variance < ZERO:
            direction = "below"
        else:
            direction = "on"

        if variance_percent is None:
            explanation = (
                f"{metric} is {direction} budget by "
                f"{abs(variance):,.2f}."
            )
        else:
            explanation = (
                f"{metric} is {direction} budget by "
                f"{abs(variance):,.2f} "
                f"({abs(variance_percent):.2f}%)."
            )

        return BudgetVarianceResult(
            metric=metric,
            actual=actual,
            budget=budget,
            variance=variance,
            variance_percent=variance_percent,
            status=status,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Forecast comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_to_forecast(
        metric: str,
        actual: NumberLike,
        forecast: NumberLike,
        higher_is_better: bool = True,
    ) -> ComparisonResult:
        """
        Compare actual performance against forecast.

        Reuses the core comparison model because the forecast is
        treated as the reference value.
        """

        return FinancialComparisonEngine.compare(
            ComparisonInput(
                metric=metric,
                current=actual,
                previous=forecast,
                higher_is_better=higher_is_better,
            )
        )

    # ------------------------------------------------------------------
    # Benchmark comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_to_benchmark(
        metric: str,
        company_value: NumberLike,
        benchmark_value: NumberLike,
        higher_is_better: bool = True,
    ) -> BenchmarkResult:
        """Compare a company metric against a benchmark."""

        company_value = to_decimal(company_value)
        benchmark_value = to_decimal(benchmark_value)

        difference = company_value - benchmark_value

        difference_percent = FinancialComparisonEngine.percentage_change(
            company_value,
            benchmark_value,
        )

        status = FinancialComparisonEngine.determine_variance_status(
            difference,
            higher_is_better,
        )

        if difference > ZERO:
            position = "above"
        elif difference < ZERO:
            position = "below"
        else:
            position = "equal to"

        if difference_percent is None:
            explanation = (
                f"{metric} is {position} the benchmark by "
                f"{abs(difference):,.2f}."
            )
        else:
            explanation = (
                f"{metric} is {position} the benchmark by "
                f"{abs(difference_percent):.2f}%."
            )

        return BenchmarkResult(
            metric=metric,
            company_value=company_value,
            benchmark_value=benchmark_value,
            difference=difference,
            difference_percent=difference_percent,
            status=status,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Multi-period trend
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_trend(
        metric: str,
        values: Iterable[NumberLike],
    ) -> TrendResult:
        """
        Analyze a metric across multiple chronological periods.

        Values must be ordered from oldest to newest.
        """

        normalized = tuple(to_decimal(value) for value in values)

        if not normalized:
            raise ValueError(
                "At least one value is required for trend analysis."
            )

        if len(normalized) == 1:
            return TrendResult(
                metric=metric,
                values=normalized,
                direction=Trend.STABLE,
                first_value=normalized[0],
                latest_value=normalized[0],
                absolute_change=ZERO,
                percentage_change=ZERO,
                periods_increasing=0,
                periods_decreasing=0,
                consistency=None,
                explanation=(
                    f"{metric} has only one available period; "
                    "no meaningful trend can be established."
                ),
            )

        changes = [
            normalized[index] - normalized[index - 1]
            for index in range(1, len(normalized))
        ]

        increasing = sum(
            1 for change in changes if change > ZERO
        )

        decreasing = sum(
            1 for change in changes if change < ZERO
        )

        stable = len(changes) - increasing - decreasing

        first_value = normalized[0]
        latest_value = normalized[-1]

        absolute_change = latest_value - first_value

        percentage_change = FinancialComparisonEngine.percentage_change(
            latest_value,
            first_value,
        )

        direction = FinancialComparisonEngine.detect_direction(
            absolute_change
        )

        if not changes:
            consistency = None
        else:
            consistency = (
                Decimal(max(increasing, decreasing))
                / Decimal(len(changes))
            ) * HUNDRED

        explanation = FinancialComparisonEngine.generate_trend_explanation(
            metric=metric,
            direction=direction,
            percentage_change=percentage_change,
            increasing=increasing,
            decreasing=decreasing,
            stable=stable,
        )

        return TrendResult(
            metric=metric,
            values=normalized,
            direction=direction,
            first_value=first_value,
            latest_value=latest_value,
            absolute_change=absolute_change,
            percentage_change=percentage_change,
            periods_increasing=increasing,
            periods_decreasing=decreasing,
            consistency=consistency,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Multiple metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compare_metrics(
        metrics: Iterable[ComparisonInput],
    ) -> list[ComparisonResult]:
        """Compare multiple financial metrics."""

        return [
            FinancialComparisonEngine.compare(metric)
            for metric in metrics
        ]

    @staticmethod
    def compare_metric_maps(
        current: Mapping[str, NumberLike],
        previous: Mapping[str, NumberLike],
        higher_is_better: Optional[Mapping[str, bool]] = None,
        significance_threshold_percent: NumberLike = Decimal("10"),
    ) -> list[ComparisonResult]:
        """
        Compare matching metrics from two dictionaries.

        Example:

            current = {
                "revenue": 150000,
                "expenses": 90000,
                "profit": 60000,
            }

            previous = {
                "revenue": 120000,
                "expenses": 80000,
                "profit": 40000,
            }
        """

        higher_is_better = higher_is_better or {}

        metric_names = sorted(
            set(current.keys()) & set(previous.keys())
        )

        results: list[ComparisonResult] = []

        for metric in metric_names:
            results.append(
                FinancialComparisonEngine.compare(
                    ComparisonInput(
                        metric=metric,
                        current=current[metric],
                        previous=previous[metric],
                        higher_is_better=higher_is_better.get(
                            metric,
                            True,
                        ),
                        significance_threshold_percent=(
                            significance_threshold_percent
                        ),
                    )
                )
            )

        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(
        results: Iterable[ComparisonResult],
    ) -> ComparisonSummary:
        """Create an aggregate summary from comparison results."""

        results = list(results)

        if not results:
            return ComparisonSummary(
                total_metrics=0,
                significant_changes=0,
                favorable_changes=0,
                unfavorable_changes=0,
                neutral_changes=0,
                increasing_metrics=0,
                decreasing_metrics=0,
                stable_metrics=0,
                overall_status=VarianceStatus.NEUTRAL,
            )

        significant_changes = sum(
            result.is_significant
            for result in results
        )

        favorable_changes = sum(
            result.status == VarianceStatus.FAVORABLE
            for result in results
        )

        unfavorable_changes = sum(
            result.status == VarianceStatus.UNFAVORABLE
            for result in results
        )

        neutral_changes = sum(
            result.status == VarianceStatus.NEUTRAL
            for result in results
        )

        increasing_metrics = sum(
            result.trend == Trend.INCREASE
            for result in results
        )

        decreasing_metrics = sum(
            result.trend == Trend.DECREASE
            for result in results
        )

        stable_metrics = sum(
            result.trend == Trend.STABLE
            for result in results
        )

        if unfavorable_changes > favorable_changes:
            overall_status = VarianceStatus.UNFAVORABLE
        elif favorable_changes > unfavorable_changes:
            overall_status = VarianceStatus.FAVORABLE
        else:
            overall_status = VarianceStatus.NEUTRAL

        return ComparisonSummary(
            total_metrics=len(results),
            significant_changes=significant_changes,
            favorable_changes=favorable_changes,
            unfavorable_changes=unfavorable_changes,
            neutral_changes=neutral_changes,
            increasing_metrics=increasing_metrics,
            decreasing_metrics=decreasing_metrics,
            stable_metrics=stable_metrics,
            overall_status=overall_status,
        )

    # ------------------------------------------------------------------
    # Alert-engine helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_significant_decline(
        current: NumberLike,
        previous: NumberLike,
        threshold_percent: NumberLike = Decimal("10"),
    ) -> bool:
        """
        Return True when a metric declined by at least the
        specified percentage.
        """

        change = FinancialComparisonEngine.percentage_change(
            current,
            previous,
        )

        if change is None:
            return False

        return change <= -abs(to_decimal(threshold_percent))

    @staticmethod
    def is_significant_increase(
        current: NumberLike,
        previous: NumberLike,
        threshold_percent: NumberLike = Decimal("10"),
    ) -> bool:
        """Return True when a metric increased significantly."""

        change = FinancialComparisonEngine.percentage_change(
            current,
            previous,
        )

        if change is None:
            return False

        return change >= abs(to_decimal(threshold_percent))

    @staticmethod
    def detect_repeated_decline(
        values: Iterable[NumberLike],
        periods: int = 3,
    ) -> bool:
        """
        Detect consecutive period declines.

        Example:
            [100, 90, 80, 70]

        periods=3 -> True
        """

        if periods <= 0:
            raise ValueError("periods must be greater than zero.")

        normalized = [
            to_decimal(value)
            for value in values
        ]

        if len(normalized) < periods + 1:
            return False

        recent = normalized[-(periods + 1):]

        return all(
            recent[index] < recent[index - 1]
            for index in range(1, len(recent))
        )

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    @staticmethod
    def direction_label(
        trend: Trend,
        percentage_change: Optional[Decimal],
    ) -> str:
        """Generate a concise direction label."""

        if trend == Trend.STABLE:
            return "stable"

        if percentage_change is None:
            return trend.value

        return (
            f"{trend.value} by "
            f"{abs(percentage_change):.2f}%"
        )

    @staticmethod
    def generate_explanation(
        metric: str,
        current: Decimal,
        previous: Decimal,
        absolute_change: Decimal,
        percentage_change: Optional[Decimal],
        trend: Trend,
        status: VarianceStatus,
    ) -> str:
        """Generate a human-readable explanation."""

        if trend == Trend.STABLE:
            return (
                f"{metric} remained unchanged at "
                f"{current:,.2f}."
            )

        if percentage_change is None:
            return (
                f"{metric} changed from "
                f"{previous:,.2f} to {current:,.2f}, "
                f"a change of {absolute_change:,.2f}."
            )

        return (
            f"{metric} {trend.value} from "
            f"{previous:,.2f} to {current:,.2f}, "
            f"a {abs(percentage_change):.2f}% "
            f"{trend.value}. "
            f"The change is classified as "
            f"{status.value}."
        )

    @staticmethod
    def generate_trend_explanation(
        metric: str,
        direction: Trend,
        percentage_change: Optional[Decimal],
        increasing: int,
        decreasing: int,
        stable: int,
    ) -> str:
        """Generate a human-readable trend explanation."""

        if percentage_change is None:
            change_text = "with no meaningful percentage baseline"
        else:
            change_text = (
                f"changing {abs(percentage_change):.2f}% "
                f"from the first to the latest period"
            )

        return (
            f"{metric} shows a {direction.value} trend, "
            f"{change_text}. "
            f"Across the observed periods there were "
            f"{increasing} increases, "
            f"{decreasing} decreases, and "
            f"{stable} stable transitions."
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def compare(
    metric: str,
    current: NumberLike,
    previous: NumberLike,
    higher_is_better: bool = True,
    significance_threshold_percent: NumberLike = Decimal("10"),
) -> ComparisonResult:
    """Convenience wrapper for a single metric comparison."""

    return FinancialComparisonEngine.compare(
        ComparisonInput(
            metric=metric,
            current=current,
            previous=previous,
            higher_is_better=higher_is_better,
            significance_threshold_percent=(
                significance_threshold_percent
            ),
        )
    )


def compare_periods(
    current: Mapping[str, NumberLike],
    previous: Mapping[str, NumberLike],
    higher_is_better: Optional[Mapping[str, bool]] = None,
) -> list[ComparisonResult]:
    """Compare two financial periods metric-by-metric."""

    return FinancialComparisonEngine.compare_metric_maps(
        current=current,
        previous=previous,
        higher_is_better=higher_is_better,
    )


def compare_budget(
    metric: str,
    actual: NumberLike,
    budget: NumberLike,
    higher_is_better: bool = True,
) -> BudgetVarianceResult:
    """Convenience wrapper for actual-vs-budget analysis."""

    return FinancialComparisonEngine.compare_to_budget(
        metric=metric,
        actual=actual,
        budget=budget,
        higher_is_better=higher_is_better,
    )


def compare_forecast(
    metric: str,
    actual: NumberLike,
    forecast: NumberLike,
    higher_is_better: bool = True,
) -> ComparisonResult:
    """Convenience wrapper for actual-vs-forecast analysis."""

    return FinancialComparisonEngine.compare_to_forecast(
        metric=metric,
        actual=actual,
        forecast=forecast,
        higher_is_better=higher_is_better,
    )


def compare_benchmark(
    metric: str,
    company_value: NumberLike,
    benchmark_value: NumberLike,
    higher_is_better: bool = True,
) -> BenchmarkResult:
    """Convenience wrapper for benchmark comparison."""

    return FinancialComparisonEngine.compare_to_benchmark(
        metric=metric,
        company_value=company_value,
        benchmark_value=benchmark_value,
        higher_is_better=higher_is_better,
    )


def analyze_trend(
    metric: str,
    values: Iterable[NumberLike],
) -> TrendResult:
    """Convenience wrapper for trend analysis."""

    return FinancialComparisonEngine.analyze_trend(
        metric=metric,
        values=values,
    )