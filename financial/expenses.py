"""
FinCo AI - Expense Analysis Engine

Provides reusable expense-analysis logic for:

    - Expense classification
    - Expense totals
    - Expense categories
    - Expense ratios
    - Fixed / variable expenses
    - Direct / indirect expenses
    - Period-over-period comparison
    - Budget variance
    - Expense trends
    - Expense concentration
    - Expense anomaly screening
    - Expense efficiency
    - Expense reduction opportunities
    - Alert-engine inputs
    - Recommendation-engine inputs
    - Forecasting inputs
    - Agentic AI financial analysis

This module contains business logic only.
Database, API, ML model persistence, and notification concerns
belong in their respective layers.

Monetary calculations use Decimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Iterable, Mapping, Optional


ZERO = Decimal("0")
HUNDRED = Decimal("100")

NumberLike = Decimal | int | float | str


# ============================================================================
# Utility functions
# ============================================================================

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


def safe_divide(
    numerator: NumberLike,
    denominator: NumberLike,
) -> Optional[Decimal]:
    """Divide safely and return None when denominator is zero."""

    numerator = to_decimal(numerator)
    denominator = to_decimal(denominator)

    if denominator == ZERO:
        return None

    return numerator / denominator


def percentage(
    numerator: NumberLike,
    denominator: NumberLike,
) -> Optional[Decimal]:
    """Calculate numerator / denominator as a percentage."""

    result = safe_divide(numerator, denominator)

    if result is None:
        return None

    return result * HUNDRED


# ============================================================================
# Enums
# ============================================================================

class ExpenseType(str, Enum):
    """Behavioral classification of an expense."""

    FIXED = "fixed"
    VARIABLE = "variable"
    SEMI_VARIABLE = "semi_variable"
    UNKNOWN = "unknown"


class ExpenseCategory(str, Enum):
    """High-level expense category."""

    OPERATING = "operating"
    NON_OPERATING = "non_operating"
    FINANCING = "financing"
    TAX = "tax"
    OTHER = "other"


class ExpenseNature(str, Enum):
    """Direct or indirect expense classification."""

    DIRECT = "direct"
    INDIRECT = "indirect"


class ExpenseTrend(str, Enum):
    """Direction of expense movement."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class ExpenseSeverity(str, Enum):
    """Severity for unusual expense movement."""

    NORMAL = "normal"
    WATCH = "watch"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Data models
# ============================================================================

@dataclass(frozen=True)
class ExpenseItem:
    """
    Represents one expense item.

    Example:

        ExpenseItem(
            name="Cloud Hosting",
            amount=25000,
            category=ExpenseCategory.OPERATING,
            expense_type=ExpenseType.VARIABLE,
            nature=ExpenseNature.INDIRECT,
        )
    """

    name: str
    amount: NumberLike

    category: ExpenseCategory = ExpenseCategory.OPERATING

    expense_type: ExpenseType = ExpenseType.UNKNOWN

    nature: ExpenseNature = ExpenseNature.INDIRECT

    department: Optional[str] = None

    vendor: Optional[str] = None

    cost_center: Optional[str] = None

    budget: Optional[NumberLike] = None

    previous_period: Optional[NumberLike] = None


@dataclass(frozen=True)
class ExpenseMetric:
    """Calculated metrics for an individual expense."""

    name: str

    amount: Decimal

    percentage_of_total: Optional[Decimal]

    percentage_of_revenue: Optional[Decimal]

    budget_variance: Optional[Decimal]

    budget_variance_percent: Optional[Decimal]

    previous_period_change: Optional[Decimal]

    previous_period_change_percent: Optional[Decimal]

    trend: ExpenseTrend


@dataclass(frozen=True)
class ExpenseVariance:
    """Actual expense versus budget."""

    name: str

    actual: Decimal

    budget: Decimal

    variance: Decimal

    variance_percent: Optional[Decimal]

    favorable: bool

    explanation: str


@dataclass(frozen=True)
class ExpenseAnomaly:
    """Potentially unusual expense movement."""

    name: str

    current_amount: Decimal

    previous_amount: Decimal

    change: Decimal

    change_percent: Optional[Decimal]

    severity: ExpenseSeverity

    reason: str


@dataclass(frozen=True)
class ExpenseOpportunity:
    """Potential expense-reduction opportunity."""

    name: str

    current_amount: Decimal

    estimated_reduction_percent: Decimal

    estimated_savings: Decimal

    reason: str


@dataclass(frozen=True)
class ExpenseAnalysisResult:
    """Complete expense-analysis result."""

    total_expenses: Decimal

    operating_expenses: Decimal

    non_operating_expenses: Decimal

    financing_expenses: Decimal

    tax_expenses: Decimal

    other_expenses: Decimal

    fixed_expenses: Decimal

    variable_expenses: Decimal

    semi_variable_expenses: Decimal

    unknown_expenses: Decimal

    direct_expenses: Decimal

    indirect_expenses: Decimal

    expense_to_revenue_ratio: Optional[Decimal]

    operating_expense_ratio: Optional[Decimal]

    fixed_expense_ratio: Optional[Decimal]

    variable_expense_ratio: Optional[Decimal]

    average_expense: Optional[Decimal]

    largest_expense: Optional[Decimal]

    largest_expense_name: Optional[str]

    concentration_top_3: Optional[Decimal]

    metrics: tuple[ExpenseMetric, ...]

    anomalies: tuple[ExpenseAnomaly, ...]

    top_expenses: tuple[str, ...]


# ============================================================================
# Expense Analyzer
# ============================================================================

class ExpenseAnalyzer:
    """
    Stateless expense-analysis engine.

    Used by:

        financial/expenses.py
        financial/cost_analysis.py
        financial/pnl.py
        alerts/rules.py
        recommendations/cost_recommendations.py
        forecasting/
        agents/tools/financial_tools.py
    """

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        expenses: Iterable[ExpenseItem],
        revenue: NumberLike = ZERO,
        anomaly_threshold_percent: NumberLike = Decimal("20"),
    ) -> ExpenseAnalysisResult:
        """Perform complete expense analysis."""

        expenses = tuple(expenses)

        if not expenses:
            return ExpenseAnalysisResult(
                total_expenses=ZERO,
                operating_expenses=ZERO,
                non_operating_expenses=ZERO,
                financing_expenses=ZERO,
                tax_expenses=ZERO,
                other_expenses=ZERO,
                fixed_expenses=ZERO,
                variable_expenses=ZERO,
                semi_variable_expenses=ZERO,
                unknown_expenses=ZERO,
                direct_expenses=ZERO,
                indirect_expenses=ZERO,
                expense_to_revenue_ratio=None,
                operating_expense_ratio=None,
                fixed_expense_ratio=None,
                variable_expense_ratio=None,
                average_expense=None,
                largest_expense=None,
                largest_expense_name=None,
                concentration_top_3=None,
                metrics=(),
                anomalies=(),
                top_expenses=(),
            )

        revenue = to_decimal(revenue)

        total_expenses = self.total_expenses(expenses)

        operating_expenses = self.expenses_by_category(
            expenses,
            ExpenseCategory.OPERATING,
        )

        non_operating_expenses = self.expenses_by_category(
            expenses,
            ExpenseCategory.NON_OPERATING,
        )

        financing_expenses = self.expenses_by_category(
            expenses,
            ExpenseCategory.FINANCING,
        )

        tax_expenses = self.expenses_by_category(
            expenses,
            ExpenseCategory.TAX,
        )

        other_expenses = self.expenses_by_category(
            expenses,
            ExpenseCategory.OTHER,
        )

        fixed_expenses = self.expenses_by_type(
            expenses,
            ExpenseType.FIXED,
        )

        variable_expenses = self.expenses_by_type(
            expenses,
            ExpenseType.VARIABLE,
        )

        semi_variable_expenses = self.expenses_by_type(
            expenses,
            ExpenseType.SEMI_VARIABLE,
        )

        unknown_expenses = self.expenses_by_type(
            expenses,
            ExpenseType.UNKNOWN,
        )

        direct_expenses = self.expenses_by_nature(
            expenses,
            ExpenseNature.DIRECT,
        )

        indirect_expenses = self.expenses_by_nature(
            expenses,
            ExpenseNature.INDIRECT,
        )

        average_expense = (
            total_expenses / Decimal(len(expenses))
            if expenses
            else None
        )

        largest_item = max(
            expenses,
            key=lambda item: to_decimal(item.amount),
        )

        concentration_top_3 = self.expense_concentration(
            expenses,
            top_n=3,
        )

        metrics = self.build_metrics(
            expenses=expenses,
            total_expenses=total_expenses,
            revenue=revenue,
        )

        anomalies = self.detect_anomalies(
            expenses,
            threshold_percent=anomaly_threshold_percent,
        )

        top_expenses = self.find_top_expenses(
            expenses,
            limit=5,
        )

        return ExpenseAnalysisResult(
            total_expenses=total_expenses,
            operating_expenses=operating_expenses,
            non_operating_expenses=non_operating_expenses,
            financing_expenses=financing_expenses,
            tax_expenses=tax_expenses,
            other_expenses=other_expenses,
            fixed_expenses=fixed_expenses,
            variable_expenses=variable_expenses,
            semi_variable_expenses=semi_variable_expenses,
            unknown_expenses=unknown_expenses,
            direct_expenses=direct_expenses,
            indirect_expenses=indirect_expenses,
            expense_to_revenue_ratio=percentage(
                total_expenses,
                revenue,
            ),
            operating_expense_ratio=percentage(
                operating_expenses,
                revenue,
            ),
            fixed_expense_ratio=percentage(
                fixed_expenses,
                total_expenses,
            ),
            variable_expense_ratio=percentage(
                variable_expenses,
                total_expenses,
            ),
            average_expense=average_expense,
            largest_expense=to_decimal(
                largest_item.amount
            ),
            largest_expense_name=largest_item.name,
            concentration_top_3=concentration_top_3,
            metrics=tuple(metrics),
            anomalies=tuple(anomalies),
            top_expenses=tuple(top_expenses),
        )

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------

    @staticmethod
    def total_expenses(
        expenses: Iterable[ExpenseItem],
    ) -> Decimal:
        """Calculate total expenses."""

        return sum(
            (
                to_decimal(expense.amount)
                for expense in expenses
            ),
            ZERO,
        )

    # ------------------------------------------------------------------
    # Category classification
    # ------------------------------------------------------------------

    @staticmethod
    def expenses_by_category(
        expenses: Iterable[ExpenseItem],
        category: ExpenseCategory,
    ) -> Decimal:
        """Calculate expenses for a category."""

        return sum(
            (
                to_decimal(expense.amount)
                for expense in expenses
                if expense.category == category
            ),
            ZERO,
        )

    @staticmethod
    def expenses_by_type(
        expenses: Iterable[ExpenseItem],
        expense_type: ExpenseType,
    ) -> Decimal:
        """Calculate expenses for a behavioral type."""

        return sum(
            (
                to_decimal(expense.amount)
                for expense in expenses
                if expense.expense_type == expense_type
            ),
            ZERO,
        )

    @staticmethod
    def expenses_by_nature(
        expenses: Iterable[ExpenseItem],
        nature: ExpenseNature,
    ) -> Decimal:
        """Calculate direct or indirect expenses."""

        return sum(
            (
                to_decimal(expense.amount)
                for expense in expenses
                if expense.nature == nature
            ),
            ZERO,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def build_metrics(
        expenses: Iterable[ExpenseItem],
        total_expenses: NumberLike,
        revenue: NumberLike,
    ) -> list[ExpenseMetric]:
        """Build detailed metrics for each expense."""

        total_expenses = to_decimal(total_expenses)
        revenue = to_decimal(revenue)

        metrics: list[ExpenseMetric] = []

        for expense in expenses:
            amount = to_decimal(expense.amount)

            budget = (
                to_decimal(expense.budget)
                if expense.budget is not None
                else None
            )

            previous = (
                to_decimal(expense.previous_period)
                if expense.previous_period is not None
                else None
            )

            budget_variance = (
                amount - budget
                if budget is not None
                else None
            )

            budget_variance_percent = (
                percentage(amount, budget)
                if budget is not None
                else None
            )

            previous_change = (
                amount - previous
                if previous is not None
                else None
            )

            previous_change_percent = (
                percentage(amount, previous)
                if previous is not None
                else None
            )

            if previous_change is None:
                trend = ExpenseTrend.STABLE
            elif previous_change > ZERO:
                trend = ExpenseTrend.INCREASING
            elif previous_change < ZERO:
                trend = ExpenseTrend.DECREASING
            else:
                trend = ExpenseTrend.STABLE

            metrics.append(
                ExpenseMetric(
                    name=expense.name,
                    amount=amount,
                    percentage_of_total=percentage(
                        amount,
                        total_expenses,
                    ),
                    percentage_of_revenue=percentage(
                        amount,
                        revenue,
                    ),
                    budget_variance=budget_variance,
                    budget_variance_percent=(
                        budget_variance_percent
                    ),
                    previous_period_change=previous_change,
                    previous_period_change_percent=(
                        previous_change_percent
                    ),
                    trend=trend,
                )
            )

        return metrics

    # ------------------------------------------------------------------
    # Budget variance
    # ------------------------------------------------------------------

    @staticmethod
    def compare_to_budget(
        expense: ExpenseItem,
    ) -> Optional[ExpenseVariance]:
        """
        Compare an expense against budget.

        For expenses:

            Actual > Budget -> unfavorable
            Actual < Budget -> favorable
        """

        if expense.budget is None:
            return None

        actual = to_decimal(expense.amount)
        budget = to_decimal(expense.budget)

        variance = actual - budget

        variance_percent = percentage(
            actual,
            budget,
        )

        favorable = variance <= ZERO

        if variance > ZERO:
            explanation = (
                f"{expense.name} is over budget by "
                f"{variance:,.2f}."
            )
        elif variance < ZERO:
            explanation = (
                f"{expense.name} is under budget by "
                f"{abs(variance):,.2f}."
            )
        else:
            explanation = (
                f"{expense.name} is exactly on budget."
            )

        return ExpenseVariance(
            name=expense.name,
            actual=actual,
            budget=budget,
            variance=variance,
            variance_percent=variance_percent,
            favorable=favorable,
            explanation=explanation,
        )

    @staticmethod
    def budget_variances(
        expenses: Iterable[ExpenseItem],
    ) -> list[ExpenseVariance]:
        """Return budget variances for expenses that have budgets."""

        results: list[ExpenseVariance] = []

        for expense in expenses:
            result = ExpenseAnalyzer.compare_to_budget(
                expense
            )

            if result is not None:
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Period comparison
    # ------------------------------------------------------------------

    @staticmethod
    def percentage_change(
        current: NumberLike,
        previous: NumberLike,
    ) -> Optional[Decimal]:
        """Calculate expense percentage change."""

        current = to_decimal(current)
        previous = to_decimal(previous)

        if previous == ZERO:
            return None

        return (
            (current - previous)
            / abs(previous)
        ) * HUNDRED

    @staticmethod
    def expense_growth(
        current: NumberLike,
        previous: NumberLike,
    ) -> Optional[Decimal]:
        """Alias for percentage expense growth."""

        return ExpenseAnalyzer.percentage_change(
            current,
            previous,
        )

    @staticmethod
    def significant_increase(
        current: NumberLike,
        previous: NumberLike,
        threshold_percent: NumberLike = Decimal("10"),
    ) -> bool:
        """
        Determine whether expense growth is significant.

        Intended for alert rules.
        """

        growth = ExpenseAnalyzer.percentage_change(
            current,
            previous,
        )

        if growth is None:
            return False

        return growth >= abs(
            to_decimal(threshold_percent)
        )

    @staticmethod
    def significant_decrease(
        current: NumberLike,
        previous: NumberLike,
        threshold_percent: NumberLike = Decimal("10"),
    ) -> bool:
        """Determine whether expense reduction is significant."""

        growth = ExpenseAnalyzer.percentage_change(
            current,
            previous,
        )

        if growth is None:
            return False

        return growth <= -abs(
            to_decimal(threshold_percent)
        )

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_anomalies(
        expenses: Iterable[ExpenseItem],
        threshold_percent: NumberLike = Decimal("20"),
        high_threshold_percent: NumberLike = Decimal("50"),
        critical_threshold_percent: NumberLike = Decimal("100"),
    ) -> list[ExpenseAnomaly]:
        """
        Screen expenses for unusual period-over-period increases.

        This is a deterministic screening layer.

        It should not be confused with the ML anomaly model in:

            backend/app/fraud/
        """

        threshold = abs(
            to_decimal(threshold_percent)
        )

        high_threshold = abs(
            to_decimal(high_threshold_percent)
        )

        critical_threshold = abs(
            to_decimal(critical_threshold_percent)
        )

        anomalies: list[ExpenseAnomaly] = []

        for expense in expenses:
            if expense.previous_period is None:
                continue

            current = to_decimal(expense.amount)
            previous = to_decimal(
                expense.previous_period
            )

            change = current - previous

            change_percent = ExpenseAnalyzer.percentage_change(
                current,
                previous,
            )

            if change_percent is None:
                continue

            if change_percent < threshold:
                continue

            if change_percent >= critical_threshold:
                severity = ExpenseSeverity.CRITICAL
            elif change_percent >= high_threshold:
                severity = ExpenseSeverity.HIGH
            else:
                severity = ExpenseSeverity.WATCH

            anomalies.append(
                ExpenseAnomaly(
                    name=expense.name,
                    current_amount=current,
                    previous_amount=previous,
                    change=change,
                    change_percent=change_percent,
                    severity=severity,
                    reason=(
                        f"{expense.name} increased by "
                        f"{change_percent:.2f}% "
                        f"from the previous period."
                    ),
                )
            )

        anomalies.sort(
            key=lambda anomaly: (
                anomaly.change_percent or ZERO
            ),
            reverse=True,
        )

        return anomalies

    # ------------------------------------------------------------------
    # Concentration
    # ------------------------------------------------------------------

    @staticmethod
    def expense_concentration(
        expenses: Iterable[ExpenseItem],
        top_n: int = 3,
    ) -> Optional[Decimal]:
        """
        Calculate the percentage of total expenses represented
        by the largest N expenses.
        """

        expenses = tuple(expenses)

        if not expenses:
            return None

        if top_n <= 0:
            return ZERO

        total = ExpenseAnalyzer.total_expenses(
            expenses
        )

        if total == ZERO:
            return None

        largest = sorted(
            (
                to_decimal(expense.amount)
                for expense in expenses
            ),
            reverse=True,
        )[:top_n]

        return (
            sum(largest, ZERO)
            / total
        ) * HUNDRED

    # ------------------------------------------------------------------
    # Top expenses
    # ------------------------------------------------------------------

    @staticmethod
    def find_top_expenses(
        expenses: Iterable[ExpenseItem],
        limit: int = 5,
    ) -> list[str]:
        """Return the names of the largest expenses."""

        if limit <= 0:
            return []

        ordered = sorted(
            expenses,
            key=lambda expense: to_decimal(
                expense.amount
            ),
            reverse=True,
        )

        return [
            expense.name
            for expense in ordered[:limit]
        ]

    # ------------------------------------------------------------------
    # Vendor analysis
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_by_vendor(
        expenses: Iterable[ExpenseItem],
    ) -> dict[str, Decimal]:
        """Aggregate expenses by vendor."""

        result: dict[str, Decimal] = {}

        for expense in expenses:
            vendor = expense.vendor or "Unassigned"

            result[vendor] = (
                result.get(vendor, ZERO)
                + to_decimal(expense.amount)
            )

        return result

    @staticmethod
    def find_top_vendors(
        expenses: Iterable[ExpenseItem],
        limit: int = 5,
    ) -> list[tuple[str, Decimal]]:
        """Return the largest vendors by expense amount."""

        aggregated = ExpenseAnalyzer.aggregate_by_vendor(
            expenses
        )

        return sorted(
            aggregated.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

    # ------------------------------------------------------------------
    # Department analysis
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_by_department(
        expenses: Iterable[ExpenseItem],
    ) -> dict[str, Decimal]:
        """Aggregate expenses by department."""

        result: dict[str, Decimal] = {}

        for expense in expenses:
            department = (
                expense.department
                or "Unassigned"
            )

            result[department] = (
                result.get(department, ZERO)
                + to_decimal(expense.amount)
            )

        return result

    # ------------------------------------------------------------------
    # Cost-center analysis
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_by_cost_center(
        expenses: Iterable[ExpenseItem],
    ) -> dict[str, Decimal]:
        """Aggregate expenses by cost center."""

        result: dict[str, Decimal] = {}

        for expense in expenses:
            cost_center = (
                expense.cost_center
                or "Unassigned"
            )

            result[cost_center] = (
                result.get(cost_center, ZERO)
                + to_decimal(expense.amount)
            )

        return result

    # ------------------------------------------------------------------
    # Expense reduction
    # ------------------------------------------------------------------

    @staticmethod
    def identify_reduction_opportunities(
        expenses: Iterable[ExpenseItem],
        reduction_percent: NumberLike = Decimal("10"),
        concentration_threshold_percent: NumberLike = Decimal("10"),
    ) -> list[ExpenseOpportunity]:
        """
        Identify large expense items that may warrant review.

        This function does NOT recommend automatically cutting
        expenses. It identifies areas for further investigation.
        """

        expenses = tuple(expenses)

        total = ExpenseAnalyzer.total_expenses(
            expenses
        )

        if total <= ZERO:
            return []

        reduction_percent = abs(
            to_decimal(reduction_percent)
        )

        concentration_threshold = abs(
            to_decimal(concentration_threshold_percent)
        )

        opportunities: list[ExpenseOpportunity] = []

        for expense in expenses:
            amount = to_decimal(expense.amount)

            share = percentage(
                amount,
                total,
            )

            if share is None:
                continue

            if share < concentration_threshold:
                continue

            estimated_savings = (
                amount
                * reduction_percent
                / HUNDRED
            )

            opportunities.append(
                ExpenseOpportunity(
                    name=expense.name,
                    current_amount=amount,
                    estimated_reduction_percent=(
                        reduction_percent
                    ),
                    estimated_savings=estimated_savings,
                    reason=(
                        f"{expense.name} represents "
                        f"{share:.2f}% of total expenses "
                        f"and should be reviewed for "
                        f"efficiency opportunities."
                    ),
                )
            )

        opportunities.sort(
            key=lambda opportunity: (
                opportunity.estimated_savings
            ),
            reverse=True,
        )

        return opportunities

    # ------------------------------------------------------------------
    # Scenario analysis
    # ------------------------------------------------------------------

    @staticmethod
    def simulate_reduction(
        expenses: Iterable[ExpenseItem],
        reduction_percent: NumberLike,
        target_names: Optional[Iterable[str]] = None,
    ) -> dict[str, Decimal]:
        """
        Simulate expense reductions without modifying
        the original expense records.
        """

        expenses = tuple(expenses)

        reduction_percent = abs(
            to_decimal(reduction_percent)
        )

        if reduction_percent > HUNDRED:
            raise ValueError(
                "reduction_percent cannot exceed 100."
            )

        targets = (
            set(target_names)
            if target_names is not None
            else None
        )

        multiplier = (
            HUNDRED - reduction_percent
        ) / HUNDRED

        projected: dict[str, Decimal] = {}

        for expense in expenses:
            amount = to_decimal(expense.amount)

            if (
                targets is None
                or expense.name in targets
            ):
                projected[expense.name] = (
                    amount * multiplier
                )
            else:
                projected[expense.name] = amount

        return projected

    @staticmethod
    def projected_total_after_reduction(
        expenses: Iterable[ExpenseItem],
        reduction_percent: NumberLike,
        target_names: Optional[Iterable[str]] = None,
    ) -> Decimal:
        """Calculate total expenses after a simulated reduction."""

        projected = ExpenseAnalyzer.simulate_reduction(
            expenses=expenses,
            reduction_percent=reduction_percent,
            target_names=target_names,
        )

        return sum(
            projected.values(),
            ZERO,
        )

    @staticmethod
    def estimated_savings(
        expenses: Iterable[ExpenseItem],
        reduction_percent: NumberLike,
        target_names: Optional[Iterable[str]] = None,
    ) -> Decimal:
        """Calculate estimated savings from a scenario."""

        current_total = ExpenseAnalyzer.total_expenses(
            expenses
        )

        projected_total = (
            ExpenseAnalyzer.projected_total_after_reduction(
                expenses=expenses,
                reduction_percent=reduction_percent,
                target_names=target_names,
            )
        )

        return current_total - projected_total

    # ------------------------------------------------------------------
    # Trend analysis
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_trend(
        period_totals: Iterable[NumberLike],
    ) -> dict[str, object]:
        """
        Analyze total expense trend.

        Values must be ordered chronologically.
        """

        values = tuple(
            to_decimal(value)
            for value in period_totals
        )

        if not values:
            raise ValueError(
                "At least one period is required."
            )

        if len(values) == 1:
            return {
                "trend": ExpenseTrend.STABLE.value,
                "first_period": values[0],
                "latest_period": values[0],
                "change": ZERO,
                "change_percent": ZERO,
            }

        first = values[0]
        latest = values[-1]

        change = latest - first

        change_percent = ExpenseAnalyzer.percentage_change(
            latest,
            first,
        )

        if change > ZERO:
            trend = ExpenseTrend.INCREASING
        elif change < ZERO:
            trend = ExpenseTrend.DECREASING
        else:
            trend = ExpenseTrend.STABLE

        return {
            "trend": trend.value,
            "first_period": first,
            "latest_period": latest,
            "change": change,
            "change_percent": change_percent,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def generate_summary(
        result: ExpenseAnalysisResult,
    ) -> str:
        """Generate a concise natural-language expense summary."""

        if result.expense_to_revenue_ratio is None:
            ratio_text = "not available"
        else:
            ratio_text = (
                f"{result.expense_to_revenue_ratio:.2f}%"
            )

        summary = (
            f"Total expenses are "
            f"{result.total_expenses:,.2f}, representing "
            f"{ratio_text} of revenue."
        )

        if result.largest_expense_name is not None:
            summary += (
                f" The largest expense is "
                f"{result.largest_expense_name} "
                f"at {result.largest_expense:,.2f}."
            )

        if result.anomalies:
            summary += (
                f" {len(result.anomalies)} "
                f"potential expense anomaly/anomalies "
                f"were detected."
            )

        if result.top_expenses:
            summary += (
                " Top expenses: "
                + ", ".join(result.top_expenses)
                + "."
            )

        return summary


# ============================================================================
# Convenience functions
# ============================================================================

def analyze_expenses(
    expenses: Iterable[ExpenseItem],
    revenue: NumberLike = ZERO,
    anomaly_threshold_percent: NumberLike = Decimal("20"),
) -> ExpenseAnalysisResult:
    """Convenience wrapper for complete expense analysis."""

    analyzer = ExpenseAnalyzer()

    return analyzer.analyze(
        expenses=expenses,
        revenue=revenue,
        anomaly_threshold_percent=(
            anomaly_threshold_percent
        ),
    )


def calculate_total_expenses(
    expenses: Iterable[ExpenseItem],
) -> Decimal:
    """Calculate total expenses."""

    return ExpenseAnalyzer.total_expenses(
        expenses
    )


def calculate_expense_ratio(
    expenses: Iterable[ExpenseItem],
    revenue: NumberLike,
) -> Optional[Decimal]:
    """Calculate total expenses as a percentage of revenue."""

    total = ExpenseAnalyzer.total_expenses(
        expenses
    )

    return percentage(
        total,
        revenue,
    )


def detect_expense_anomalies(
    expenses: Iterable[ExpenseItem],
    threshold_percent: NumberLike = Decimal("20"),
) -> list[ExpenseAnomaly]:
    """Detect unusually large expense increases."""

    return ExpenseAnalyzer.detect_anomalies(
        expenses=expenses,
        threshold_percent=threshold_percent,
    )


def identify_expense_opportunities(
    expenses: Iterable[ExpenseItem],
    reduction_percent: NumberLike = Decimal("10"),
) -> list[ExpenseOpportunity]:
    """Identify potential expense-reduction areas."""

    return ExpenseAnalyzer.identify_reduction_opportunities(
        expenses=expenses,
        reduction_percent=reduction_percent,
    )