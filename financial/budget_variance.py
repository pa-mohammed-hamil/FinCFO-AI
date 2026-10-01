"""
FinCo AI - Budget Variance Analysis
===================================

Location:
    backend/app/financial/budget_variance.py

Purpose:
    Compare budgeted financial values against actual values and produce
    structured variance analysis for:

    - Financial dashboards
    - Alert Engine
    - Recommendation Engine
    - What-If Analysis
    - Supervisor Agent
    - Executive reports
    - Financial health scoring

Design principles:
    - No database access
    - No API logic
    - No LLM calls
    - Deterministic calculations
    - Decimal-based financial arithmetic
    - Explicit revenue/expense semantics
    - Easy integration with Pydantic schemas and SQLAlchemy models
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence


# ============================================================================
# ENUMS
# ============================================================================


class MetricType(str, Enum):
    """Financial metric category."""

    REVENUE = "revenue"
    EXPENSE = "expense"
    PROFIT = "profit"
    CASH_FLOW = "cash_flow"
    GENERIC = "generic"


class VarianceType(str, Enum):
    """Business interpretation of a variance."""

    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    NEUTRAL = "neutral"


class VarianceSeverity(str, Enum):
    """Severity used by downstream alert/recommendation systems."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class VarianceResult:
    """
    Result for one budget-vs-actual comparison.
    """

    metric: str
    metric_type: MetricType

    budget: Decimal
    actual: Decimal

    variance: Decimal
    variance_percent: Decimal

    variance_type: VarianceType
    severity: VarianceSeverity

    threshold_percent: Decimal
    threshold_exceeded: bool

    period: Optional[str] = None
    category: Optional[str] = None

    explanation: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result into JSON-friendly dictionary."""

        return {
            "metric": self.metric,
            "metric_type": self.metric_type.value,
            "budget": float(self.budget),
            "actual": float(self.actual),
            "variance": float(self.variance),
            "variance_percent": float(self.variance_percent),
            "variance_type": self.variance_type.value,
            "severity": self.severity.value,
            "threshold_percent": float(self.threshold_percent),
            "threshold_exceeded": self.threshold_exceeded,
            "period": self.period,
            "category": self.category,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class PeriodVariance:
    """
    Aggregated variance for one accounting period.
    """

    period: str

    budget: Decimal
    actual: Decimal

    variance: Decimal
    variance_percent: Decimal

    variance_type: VarianceType
    severity: VarianceSeverity

    threshold_exceeded: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert period variance into JSON-friendly dictionary."""

        return {
            "period": self.period,
            "budget": float(self.budget),
            "actual": float(self.actual),
            "variance": float(self.variance),
            "variance_percent": float(self.variance_percent),
            "variance_type": self.variance_type.value,
            "severity": self.severity.value,
            "threshold_exceeded": self.threshold_exceeded,
        }


@dataclass(frozen=True)
class BudgetVarianceSummary:
    """
    Portfolio-level budget variance summary.
    """

    total_budget: Decimal
    total_actual: Decimal

    total_variance: Decimal
    total_variance_percent: Decimal

    favorable_count: int
    unfavorable_count: int
    neutral_count: int

    threshold_breaches: int

    high_risk_count: int

    results: Sequence[VarianceResult]

    def to_dict(self) -> dict[str, Any]:
        """Convert summary into JSON-friendly dictionary."""

        return {
            "total_budget": float(self.total_budget),
            "total_actual": float(self.total_actual),
            "total_variance": float(self.total_variance),
            "total_variance_percent": float(
                self.total_variance_percent
            ),
            "favorable_count": self.favorable_count,
            "unfavorable_count": self.unfavorable_count,
            "neutral_count": self.neutral_count,
            "threshold_breaches": self.threshold_breaches,
            "high_risk_count": self.high_risk_count,
            "results": [
                result.to_dict()
                for result in self.results
            ],
        }


# ============================================================================
# ANALYZER
# ============================================================================


class BudgetVarianceAnalyzer:
    """
    Core FinCo AI budget variance engine.

    Example:

        analyzer = BudgetVarianceAnalyzer()

        result = analyzer.calculate(
            metric="Operating Expenses",
            budget=100000,
            actual=115000,
            metric_type=MetricType.EXPENSE,
            period="2026-08",
        )

    Result:

        variance = 15000
        variance_percent = 15%
        variance_type = unfavorable
    """

    DEFAULT_THRESHOLD_PERCENT = Decimal("10.0")

    def __init__(
        self,
        threshold_percent: Decimal | float | int = DEFAULT_THRESHOLD_PERCENT,
    ) -> None:

        self.threshold_percent = self._to_decimal(
            threshold_percent
        )

        if self.threshold_percent < 0:
            raise ValueError(
                "threshold_percent cannot be negative."
            )

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    def calculate(
        self,
        metric: str,
        budget: Decimal | float | int,
        actual: Decimal | float | int,
        metric_type: MetricType | str = MetricType.GENERIC,
        period: Optional[str] = None,
        category: Optional[str] = None,
        threshold_percent: Optional[
            Decimal | float | int
        ] = None,
    ) -> VarianceResult:
        """
        Calculate budget variance for one metric.

        Formula:

            variance = actual - budget

            variance % =
                ((actual - budget) / abs(budget)) * 100

        Example:

            Budget = 100,000
            Actual = 115,000

            Variance = +15,000
            Variance % = +15%
        """

        if not metric or not metric.strip():
            raise ValueError(
                "metric must not be empty."
            )

        budget_decimal = self._to_decimal(budget)
        actual_decimal = self._to_decimal(actual)

        metric_type_enum = self._normalize_metric_type(
            metric_type
        )

        variance = actual_decimal - budget_decimal

        variance_percent = self._calculate_percentage_variance(
            budget=budget_decimal,
            actual=actual_decimal,
        )

        threshold = (
            self.threshold_percent
            if threshold_percent is None
            else self._to_decimal(threshold_percent)
        )

        if threshold < 0:
            raise ValueError(
                "threshold_percent cannot be negative."
            )

        variance_type = self._classify_variance(
            variance=variance,
            metric_type=metric_type_enum,
        )

        threshold_exceeded = (
            abs(variance_percent) >= threshold
        )

        severity = self._calculate_severity(
            variance_percent=variance_percent,
            variance_type=variance_type,
        )

        explanation = self._generate_explanation(
            metric=metric,
            metric_type=metric_type_enum,
            budget=budget_decimal,
            actual=actual_decimal,
            variance=variance,
            variance_percent=variance_percent,
            variance_type=variance_type,
        )

        return VarianceResult(
            metric=metric.strip(),
            metric_type=metric_type_enum,
            budget=budget_decimal,
            actual=actual_decimal,
            variance=variance,
            variance_percent=variance_percent,
            variance_type=variance_type,
            severity=severity,
            threshold_percent=threshold,
            threshold_exceeded=threshold_exceeded,
            period=period,
            category=category,
            explanation=explanation,
        )

    # ========================================================================

    def analyze(
        self,
        items: Iterable[Mapping[str, Any]],
        *,
        threshold_percent: Optional[
            Decimal | float | int
        ] = None,
    ) -> BudgetVarianceSummary:
        """
        Analyze multiple budget records.

        Expected input:

            [
                {
                    "metric": "Revenue",
                    "budget": 500000,
                    "actual": 550000,
                    "metric_type": "revenue",
                    "period": "2026-08"
                },
                {
                    "metric": "Marketing",
                    "budget": 50000,
                    "actual": 65000,
                    "metric_type": "expense",
                    "period": "2026-08"
                }
            ]
        """

        results: list[VarianceResult] = []

        for item in items:

            result = self.calculate(
                metric=item["metric"],
                budget=item["budget"],
                actual=item["actual"],
                metric_type=item.get(
                    "metric_type",
                    MetricType.GENERIC,
                ),
                period=item.get("period"),
                category=item.get("category"),
                threshold_percent=threshold_percent,
            )

            results.append(result)

        total_budget = sum(
            (result.budget for result in results),
            Decimal("0"),
        )

        total_actual = sum(
            (result.actual for result in results),
            Decimal("0"),
        )

        total_variance = (
            total_actual - total_budget
        )

        total_variance_percent = (
            self._calculate_percentage_variance(
                budget=total_budget,
                actual=total_actual,
            )
        )

        favorable_count = sum(
            result.variance_type
            == VarianceType.FAVORABLE
            for result in results
        )

        unfavorable_count = sum(
            result.variance_type
            == VarianceType.UNFAVORABLE
            for result in results
        )

        neutral_count = sum(
            result.variance_type
            == VarianceType.NEUTRAL
            for result in results
        )

        threshold_breaches = sum(
            result.threshold_exceeded
            for result in results
        )

        high_risk_count = sum(
            result.severity
            in {
                VarianceSeverity.HIGH,
                VarianceSeverity.CRITICAL,
            }
            for result in results
        )

        return BudgetVarianceSummary(
            total_budget=total_budget,
            total_actual=total_actual,
            total_variance=total_variance,
            total_variance_percent=total_variance_percent,
            favorable_count=favorable_count,
            unfavorable_count=unfavorable_count,
            neutral_count=neutral_count,
            threshold_breaches=threshold_breaches,
            high_risk_count=high_risk_count,
            results=results,
        )

    # ========================================================================

    def analyze_periods(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        metric_type: MetricType | str = MetricType.EXPENSE,
        threshold_percent: Optional[
            Decimal | float | int
        ] = None,
    ) -> list[PeriodVariance]:
        """
        Aggregate budget and actual values by period.

        Example:

            [
                {
                    "period": "2026-01",
                    "budget": 100000,
                    "actual": 105000,
                },
                {
                    "period": "2026-01",
                    "budget": 50000,
                    "actual": 55000,
                }
            ]

        Result:

            2026-01
            Budget = 150,000
            Actual = 160,000
            Variance = 10,000
            Variance % = 6.67%
        """

        grouped: dict[
            str,
            dict[str, Decimal],
        ] = {}

        for record in records:

            period = record.get("period")

            if not period:
                raise ValueError(
                    "Each record must contain a period."
                )

            if period not in grouped:

                grouped[period] = {
                    "budget": Decimal("0"),
                    "actual": Decimal("0"),
                }

            grouped[period]["budget"] += (
                self._to_decimal(record["budget"])
            )

            grouped[period]["actual"] += (
                self._to_decimal(record["actual"])
            )

        metric_type_enum = self._normalize_metric_type(
            metric_type
        )

        threshold = (
            self.threshold_percent
            if threshold_percent is None
            else self._to_decimal(threshold_percent)
        )

        results: list[PeriodVariance] = []

        for period, values in sorted(
            grouped.items()
        ):

            budget = values["budget"]
            actual = values["actual"]

            variance = actual - budget

            variance_percent = (
                self._calculate_percentage_variance(
                    budget=budget,
                    actual=actual,
                )
            )

            variance_type = self._classify_variance(
                variance=variance,
                metric_type=metric_type_enum,
            )

            severity = self._calculate_severity(
                variance_percent=variance_percent,
                variance_type=variance_type,
            )

            results.append(
                PeriodVariance(
                    period=period,
                    budget=budget,
                    actual=actual,
                    variance=variance,
                    variance_percent=variance_percent,
                    variance_type=variance_type,
                    severity=severity,
                    threshold_exceeded=(
                        abs(variance_percent)
                        >= threshold
                    ),
                )
            )

        return results

    # ========================================================================

    def find_major_variances(
        self,
        results: Sequence[VarianceResult],
        *,
        limit: int = 10,
        minimum_percent: Decimal | float | int = 10,
        unfavorable_only: bool = False,
    ) -> list[VarianceResult]:
        """
        Return the largest material variances.

        Useful for:

            - Executive dashboards
            - Alert Engine
            - Supervisor Agent
            - Recommendation Agent
        """

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        minimum = self._to_decimal(
            minimum_percent
        )

        filtered = [
            result
            for result in results
            if abs(result.variance_percent)
            >= minimum
        ]

        if unfavorable_only:

            filtered = [
                result
                for result in filtered
                if result.variance_type
                == VarianceType.UNFAVORABLE
            ]

        return sorted(
            filtered,
            key=lambda item: abs(
                item.variance_percent
            ),
            reverse=True,
        )[:limit]

    # ========================================================================

    def calculate_budget_utilization(
        self,
        budget: Decimal | float | int,
        actual: Decimal | float | int,
    ) -> Decimal:
        """
        Calculate percentage of budget consumed.

        Example:

            Budget = 100,000
            Actual = 75,000

            Utilization = 75%
        """

        budget_decimal = self._to_decimal(budget)
        actual_decimal = self._to_decimal(actual)

        if budget_decimal == 0:

            if actual_decimal == 0:
                return Decimal("0")

            return Decimal("100")

        return (
            actual_decimal
            / abs(budget_decimal)
            * Decimal("100")
        ).quantize(Decimal("0.01"))

    # ========================================================================

    def calculate_remaining_budget(
        self,
        budget: Decimal | float | int,
        actual: Decimal | float | int,
    ) -> Decimal:
        """
        Calculate remaining budget.

        Positive value:
            budget remaining

        Negative value:
            budget exceeded
        """

        return (
            self._to_decimal(budget)
            - self._to_decimal(actual)
        )

    # ========================================================================

    def detect_over_budget(
        self,
        budget: Decimal | float | int,
        actual: Decimal | float | int,
    ) -> bool:
        """Return True when actual spending exceeds budget."""

        return (
            self._to_decimal(actual)
            > self._to_decimal(budget)
        )

    # ========================================================================

    def forecast_budget_overrun(
        self,
        budget: Decimal | float | int,
        actual_to_date: Decimal | float | int,
        elapsed_ratio: Decimal | float,
    ) -> dict[str, Any]:
        """
        Estimate end-of-period actual value using current run rate.

        Example:

            Annual budget = 1,200,000
            Actual after 6 months = 700,000
            Elapsed ratio = 0.50

            Projected actual = 1,400,000

        This is a simple baseline projection and should not replace
        the dedicated forecasting module.
        """

        budget_decimal = self._to_decimal(budget)
        actual_decimal = self._to_decimal(
            actual_to_date
        )
        ratio = self._to_decimal(
            elapsed_ratio
        )

        if ratio <= 0:
            raise ValueError(
                "elapsed_ratio must be greater than zero."
            )

        if ratio > 1:
            raise ValueError(
                "elapsed_ratio cannot exceed 1."
            )

        projected_actual = (
            actual_decimal / ratio
        )

        projected_variance = (
            projected_actual - budget_decimal
        )

        projected_variance_percent = (
            self._calculate_percentage_variance(
                budget=budget_decimal,
                actual=projected_actual,
            )
        )

        return {
            "budget": float(budget_decimal),
            "actual_to_date": float(actual_decimal),
            "elapsed_ratio": float(ratio),
            "projected_actual": float(
                projected_actual
            ),
            "projected_variance": float(
                projected_variance
            ),
            "projected_variance_percent": float(
                projected_variance_percent
            ),
            "projected_overrun": (
                projected_actual
                > budget_decimal
            ),
        }

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    @staticmethod
    def _to_decimal(
        value: Decimal | float | int | str,
    ) -> Decimal:
        """
        Safely convert numeric values to Decimal.

        Strings are preferred for financial precision.
        """

        if isinstance(value, Decimal):
            return value

        if isinstance(value, bool):
            raise ValueError(
                "Boolean values are not valid financial numbers."
            )

        try:

            if isinstance(value, float):
                return Decimal(str(value))

            return Decimal(value)

        except (InvalidOperation, TypeError, ValueError) as exc:

            raise ValueError(
                f"Invalid numeric value: {value!r}"
            ) from exc

    # ========================================================================

    @staticmethod
    def _normalize_metric_type(
        metric_type: MetricType | str,
    ) -> MetricType:

        if isinstance(metric_type, MetricType):
            return metric_type

        try:
            return MetricType(
                str(metric_type).lower()
            )

        except ValueError as exc:

            raise ValueError(
                f"Unsupported metric_type: {metric_type}"
            ) from exc

    # ========================================================================

    @staticmethod
    def _calculate_percentage_variance(
        budget: Decimal,
        actual: Decimal,
    ) -> Decimal:
        """
        Calculate percentage variance.

        Uses absolute budget as denominator so negative
        financial metrics remain mathematically stable.
        """

        if budget == 0:

            if actual == 0:
                return Decimal("0")

            return Decimal("100")

        percentage = (
            (actual - budget)
            / abs(budget)
            * Decimal("100")
        )

        return percentage.quantize(
            Decimal("0.01")
        )

    # ========================================================================

    @staticmethod
    def _classify_variance(
        variance: Decimal,
        metric_type: MetricType,
    ) -> VarianceType:
        """
        Determine whether the variance is favorable.

        Revenue:
            actual > budget -> favorable

        Expense:
            actual < budget -> favorable

        Profit:
            actual > budget -> favorable

        Cash flow:
            actual > budget -> favorable

        Generic:
            positive -> unfavorable
            negative -> favorable
        """

        if variance == 0:
            return VarianceType.NEUTRAL

        if metric_type in {
            MetricType.REVENUE,
            MetricType.PROFIT,
            MetricType.CASH_FLOW,
        }:

            return (
                VarianceType.FAVORABLE
                if variance > 0
                else VarianceType.UNFAVORABLE
            )

        if metric_type == MetricType.EXPENSE:

            return (
                VarianceType.FAVORABLE
                if variance < 0
                else VarianceType.UNFAVORABLE
            )

        # Generic metric convention
        return (
            VarianceType.UNFAVORABLE
            if variance > 0
            else VarianceType.FAVORABLE
        )

    # ========================================================================

    @staticmethod
    def _calculate_severity(
        variance_percent: Decimal,
        variance_type: VarianceType,
    ) -> VarianceSeverity:
        """
        Determine variance severity.

        Severity is mainly intended for unfavorable variances.
        """

        if variance_type != VarianceType.UNFAVORABLE:

            return VarianceSeverity.NONE

        magnitude = abs(
            variance_percent
        )

        if magnitude >= Decimal("50"):
            return VarianceSeverity.CRITICAL

        if magnitude >= Decimal("25"):
            return VarianceSeverity.HIGH

        if magnitude >= Decimal("10"):
            return VarianceSeverity.MEDIUM

        if magnitude >= Decimal("5"):
            return VarianceSeverity.LOW

        return VarianceSeverity.NONE

    # ========================================================================

    @staticmethod
    def _generate_explanation(
        metric: str,
        metric_type: MetricType,
        budget: Decimal,
        actual: Decimal,
        variance: Decimal,
        variance_percent: Decimal,
        variance_type: VarianceType,
    ) -> str:
        """
        Generate deterministic explanation.

        This explanation can later be passed to an LLM,
        but the financial calculation itself remains deterministic.
        """

        direction = (
            "above"
            if variance > 0
            else "below"
            if variance < 0
            else "equal to"
        )

        if variance_type == VarianceType.FAVORABLE:

            interpretation = "favorable"

        elif variance_type == VarianceType.UNFAVORABLE:

            interpretation = "unfavorable"

        else:

            interpretation = "neutral"

        return (
            f"{metric} actual value of "
            f"{actual:,.2f} is {abs(variance):,.2f} "
            f"({abs(variance_percent):.2f}%) "
            f"{direction} the budget of "
            f"{budget:,.2f}. "
            f"The variance is {interpretation}."
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def calculate_budget_variance(
    budget: Decimal | float | int,
    actual: Decimal | float | int,
    metric_type: MetricType | str = MetricType.GENERIC,
    threshold_percent: Decimal | float | int = Decimal("10"),
) -> VarianceResult:
    """
    Convenience function for one-off calculations.
    """

    analyzer = BudgetVarianceAnalyzer(
        threshold_percent=threshold_percent
    )

    return analyzer.calculate(
        metric="Financial Metric",
        budget=budget,
        actual=actual,
        metric_type=metric_type,
    )


def analyze_budget_variance(
    records: Iterable[Mapping[str, Any]],
    threshold_percent: Decimal | float | int = Decimal("10"),
) -> BudgetVarianceSummary:
    """
    Convenience function for multiple records.
    """

    analyzer = BudgetVarianceAnalyzer(
        threshold_percent=threshold_percent
    )

    return analyzer.analyze(records)


# ============================================================================
# EXAMPLE
# ============================================================================

if __name__ == "__main__":

    analyzer = BudgetVarianceAnalyzer(
        threshold_percent=Decimal("10")
    )

    records = [
        {
            "metric": "Revenue",
            "budget": "500000",
            "actual": "550000",
            "metric_type": "revenue",
            "period": "2026-08",
            "category": "Sales",
        },
        {
            "metric": "Marketing Expense",
            "budget": "50000",
            "actual": "65000",
            "metric_type": "expense",
            "period": "2026-08",
            "category": "Marketing",
        },
        {
            "metric": "Payroll Expense",
            "budget": "100000",
            "actual": "95000",
            "metric_type": "expense",
            "period": "2026-08",
            "category": "Payroll",
        },
    ]

    summary = analyzer.analyze(records)

    print("=" * 70)
    print("FINCO AI - BUDGET VARIANCE ANALYSIS")
    print("=" * 70)

    print(
        f"Total Budget : {summary.total_budget:,.2f}"
    )

    print(
        f"Total Actual : {summary.total_actual:,.2f}"
    )

    print(
        f"Total Variance : {summary.total_variance:,.2f}"
    )

    print(
        f"Total Variance % : "
        f"{summary.total_variance_percent:.2f}%"
    )

    print(
        f"Favorable : {summary.favorable_count}"
    )

    print(
        f"Unfavorable : {summary.unfavorable_count}"
    )

    print(
        f"Threshold Breaches : "
        f"{summary.threshold_breaches}"
    )

    print(
        f"High/Critical : "
        f"{summary.high_risk_count}"
    )

    print("\nDetailed Variances:")

    for result in summary.results:

        print(
            f"\n{result.metric}"
        )

        print(
            f"  Budget   : {result.budget:,.2f}"
        )

        print(
            f"  Actual   : {result.actual:,.2f}"
        )

        print(
            f"  Variance : {result.variance:,.2f}"
        )

        print(
            f"  Variance % : "
            f"{result.variance_percent:.2f}%"
        )

        print(
            f"  Type     : "
            f"{result.variance_type.value}"
        )

        print(
            f"  Severity : "
            f"{result.severity.value}"
        )

        print(
            f"  Alert Threshold : "
            f"{result.threshold_exceeded}"
        )

        print(
            f"  Explanation : "
            f"{result.explanation}"
        )

### FinCo AI integration

This module should sit in your pipeline as:

Budget + Actual Financial Data
            ↓
   budget_variance.py
            ↓
 ┌────────────────────────────┐
 │ variance                   │
 │ variance_percent           │
 │ favorable/unfavorable      │
 │ severity                   │
 │ threshold_exceeded         │
 └────────────────────────────┘
            ↓
      health_score.py
            ↓
      risk_scoring.py
            ↓
       alert_engine
            ↓
      root_cause.py
            ↓
      forecast / RAG
            ↓
 recommendation_agent
            ↓
       Human Review
            ↓
         Audit Log

A particularly important design choice here is that **revenue and expenses are interpreted differently**: revenue above budget is favorable, while expenses above budget are unfavorable. That prevents a simplistic `actual - budget > 0` rule from generating incorrect financial alerts.

For example:

analyzer = BudgetVarianceAnalyzer()

result = analyzer.calculate(
    metric="Operating Expenses",
    budget=100000,
    actual=125000,
    metric_type=MetricType.EXPENSE,
    period="2026-08",
)

print(result.to_dict())