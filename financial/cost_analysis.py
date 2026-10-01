"""
FinCo AI - Cost Analysis Engine

Provides reusable cost-analysis logic for:

    - Fixed vs variable costs
    - Direct vs indirect costs
    - Cost ratios
    - Cost per unit
    - Cost variance
    - Cost growth
    - Cost concentration
    - Cost efficiency
    - Cost drivers
    - Break-even analysis
    - Operating leverage
    - Cost reduction opportunities
    - Alert-engine inputs
    - Recommendation-engine inputs
    - What-if analysis
    - Agentic AI financial analysis

This module contains business logic only.
Database, API, ML, and persistence concerns belong in their
respective layers.

Monetary calculations use Decimal to reduce floating-point
rounding errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Iterable, Mapping, Optional


ZERO = Decimal("0")
ONE = Decimal("1")
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
    """
    Divide safely.

    Returns None when the denominator is zero.
    """

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

class CostType(str, Enum):
    """Primary cost behavior classification."""

    FIXED = "fixed"
    VARIABLE = "variable"
    SEMI_VARIABLE = "semi_variable"
    UNKNOWN = "unknown"


class CostCategory(str, Enum):
    """High-level cost classification."""

    DIRECT = "direct"
    INDIRECT = "indirect"


class CostTrend(str, Enum):
    """Direction of cost movement."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class CostEfficiency(str, Enum):
    """Cost efficiency classification."""

    EFFICIENT = "efficient"
    ACCEPTABLE = "acceptable"
    INEFFICIENT = "inefficient"


class OpportunityPriority(str, Enum):
    """Priority of a cost-reduction opportunity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Data models
# ============================================================================

@dataclass(frozen=True)
class CostItem:
    """
    Represents one cost item.

    Example:
        CostItem(
            name="Cloud Infrastructure",
            amount=25000,
            cost_type=CostType.VARIABLE,
            category=CostCategory.INDIRECT,
        )
    """

    name: str
    amount: NumberLike

    cost_type: CostType = CostType.UNKNOWN
    category: CostCategory = CostCategory.INDIRECT

    department: Optional[str] = None
    cost_center: Optional[str] = None

    units: Optional[NumberLike] = None

    budget: Optional[NumberLike] = None
    previous_period: Optional[NumberLike] = None


@dataclass(frozen=True)
class CostMetric:
    """Calculated metric for an individual cost item."""

    name: str

    amount: Decimal

    percentage_of_total: Optional[Decimal]

    percentage_of_revenue: Optional[Decimal]

    budget_variance: Optional[Decimal]
    budget_variance_percent: Optional[Decimal]

    previous_period_change: Optional[Decimal]
    previous_period_change_percent: Optional[Decimal]

    trend: CostTrend


@dataclass(frozen=True)
class CostAnalysisResult:
    """Complete cost-analysis result."""

    total_cost: Decimal

    fixed_cost: Decimal
    variable_cost: Decimal
    semi_variable_cost: Decimal
    unknown_cost: Decimal

    direct_cost: Decimal
    indirect_cost: Decimal

    cost_to_revenue_ratio: Optional[Decimal]

    fixed_cost_ratio: Optional[Decimal]
    variable_cost_ratio: Optional[Decimal]

    direct_cost_ratio: Optional[Decimal]
    indirect_cost_ratio: Optional[Decimal]

    cost_per_unit: Optional[Decimal]

    gross_margin: Optional[Decimal]

    contribution_margin: Optional[Decimal]

    break_even_revenue: Optional[Decimal]

    operating_leverage: Optional[Decimal]

    metrics: tuple[CostMetric, ...]

    top_cost_drivers: tuple[str, ...]

    efficiency: CostEfficiency


@dataclass(frozen=True)
class CostVarianceResult:
    """Actual cost versus budget."""

    name: str

    actual: Decimal
    budget: Decimal

    variance: Decimal
    variance_percent: Optional[Decimal]

    favorable: bool

    explanation: str


@dataclass(frozen=True)
class CostOpportunity:
    """Potential cost-reduction opportunity."""

    name: str

    current_cost: Decimal

    estimated_reduction_percent: Decimal

    estimated_savings: Decimal

    priority: OpportunityPriority

    reason: str


@dataclass(frozen=True)
class BreakEvenResult:
    """Break-even analysis."""

    fixed_cost: Decimal
    variable_cost_per_unit: Decimal
    selling_price_per_unit: Decimal

    contribution_per_unit: Decimal

    break_even_units: Optional[Decimal]
    break_even_revenue: Optional[Decimal]

    contribution_margin_percent: Optional[Decimal]


# ============================================================================
# Main analyzer
# ============================================================================

class CostAnalyzer:
    """
    Stateless cost-analysis engine.

    Intended to be reused by:

        financial/cost_analysis.py
        financial/health_score.py
        alerts/rules.py
        recommendations/cost_recommendations.py
        what_if/financial_model.py
        agents/tools/financial_tools.py
    """

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        costs: Iterable[CostItem],
        revenue: NumberLike = ZERO,
        units_sold: NumberLike | None = None,
        gross_profit: NumberLike | None = None,
        operating_profit: NumberLike | None = None,
    ) -> CostAnalysisResult:
        """Perform comprehensive cost analysis."""

        costs = tuple(costs)

        if not costs:
            return CostAnalysisResult(
                total_cost=ZERO,
                fixed_cost=ZERO,
                variable_cost=ZERO,
                semi_variable_cost=ZERO,
                unknown_cost=ZERO,
                direct_cost=ZERO,
                indirect_cost=ZERO,
                cost_to_revenue_ratio=None,
                fixed_cost_ratio=None,
                variable_cost_ratio=None,
                direct_cost_ratio=None,
                indirect_cost_ratio=None,
                cost_per_unit=None,
                gross_margin=None,
                contribution_margin=None,
                break_even_revenue=None,
                operating_leverage=None,
                metrics=(),
                top_cost_drivers=(),
                efficiency=CostEfficiency.ACCEPTABLE,
            )

        revenue = to_decimal(revenue)

        total_cost = self.total_cost(costs)

        fixed_cost = self.cost_by_type(
            costs,
            CostType.FIXED,
        )

        variable_cost = self.cost_by_type(
            costs,
            CostType.VARIABLE,
        )

        semi_variable_cost = self.cost_by_type(
            costs,
            CostType.SEMI_VARIABLE,
        )

        unknown_cost = self.cost_by_type(
            costs,
            CostType.UNKNOWN,
        )

        direct_cost = self.cost_by_category(
            costs,
            CostCategory.DIRECT,
        )

        indirect_cost = self.cost_by_category(
            costs,
            CostCategory.INDIRECT,
        )

        cost_per_unit = self.calculate_cost_per_unit(
            total_cost,
            units_sold,
        )

        gross_margin = self.calculate_margin(
            gross_profit,
            revenue,
        )

        contribution_margin = self.calculate_contribution_margin(
            revenue,
            variable_cost,
        )

        break_even_revenue = self.calculate_break_even_revenue(
            fixed_cost=fixed_cost,
            contribution_margin=contribution_margin,
        )

        operating_leverage = self.calculate_operating_leverage(
            revenue=revenue,
            variable_cost=variable_cost,
            operating_profit=operating_profit,
        )

        metrics = self.build_metrics(
            costs=costs,
            total_cost=total_cost,
            revenue=revenue,
        )

        top_cost_drivers = self.find_top_cost_drivers(
            costs,
            limit=5,
        )

        efficiency = self.classify_efficiency(
            cost_to_revenue_ratio=safe_divide(
                total_cost,
                revenue,
            ),
        )

        return CostAnalysisResult(
            total_cost=total_cost,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            semi_variable_cost=semi_variable_cost,
            unknown_cost=unknown_cost,
            direct_cost=direct_cost,
            indirect_cost=indirect_cost,
            cost_to_revenue_ratio=percentage(
                total_cost,
                revenue,
            ),
            fixed_cost_ratio=percentage(
                fixed_cost,
                total_cost,
            ),
            variable_cost_ratio=percentage(
                variable_cost,
                total_cost,
            ),
            direct_cost_ratio=percentage(
                direct_cost,
                total_cost,
            ),
            indirect_cost_ratio=percentage(
                indirect_cost,
                total_cost,
            ),
            cost_per_unit=cost_per_unit,
            gross_margin=gross_margin,
            contribution_margin=contribution_margin,
            break_even_revenue=break_even_revenue,
            operating_leverage=operating_leverage,
            metrics=tuple(metrics),
            top_cost_drivers=tuple(top_cost_drivers),
            efficiency=efficiency,
        )

    # ------------------------------------------------------------------
    # Total cost
    # ------------------------------------------------------------------

    @staticmethod
    def total_cost(
        costs: Iterable[CostItem],
    ) -> Decimal:
        """Calculate total cost."""

        return sum(
            (
                to_decimal(cost.amount)
                for cost in costs
            ),
            ZERO,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def cost_by_type(
        costs: Iterable[CostItem],
        cost_type: CostType,
    ) -> Decimal:
        """Calculate total cost for a specific cost type."""

        return sum(
            (
                to_decimal(cost.amount)
                for cost in costs
                if cost.cost_type == cost_type
            ),
            ZERO,
        )

    @staticmethod
    def cost_by_category(
        costs: Iterable[CostItem],
        category: CostCategory,
    ) -> Decimal:
        """Calculate total direct or indirect cost."""

        return sum(
            (
                to_decimal(cost.amount)
                for cost in costs
                if cost.category == category
            ),
            ZERO,
        )

    # ------------------------------------------------------------------
    # Unit economics
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_cost_per_unit(
        total_cost: NumberLike,
        units_sold: NumberLike | None,
    ) -> Optional[Decimal]:
        """Calculate total cost per unit."""

        if units_sold is None:
            return None

        units_sold = to_decimal(units_sold)

        if units_sold <= ZERO:
            return None

        return to_decimal(total_cost) / units_sold

    @staticmethod
    def calculate_variable_cost_per_unit(
        variable_cost: NumberLike,
        units_sold: NumberLike,
    ) -> Optional[Decimal]:
        """Calculate variable cost per unit."""

        units_sold = to_decimal(units_sold)

        if units_sold <= ZERO:
            return None

        return to_decimal(variable_cost) / units_sold

    # ------------------------------------------------------------------
    # Margins
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_margin(
        profit: NumberLike | None,
        revenue: NumberLike,
    ) -> Optional[Decimal]:
        """Calculate profit margin."""

        if profit is None:
            return None

        return percentage(
            profit,
            revenue,
        )

    @staticmethod
    def calculate_contribution_margin(
        revenue: NumberLike,
        variable_cost: NumberLike,
    ) -> Optional[Decimal]:
        """
        Calculate contribution margin percentage.

            (Revenue - Variable Cost) / Revenue * 100
        """

        revenue = to_decimal(revenue)
        variable_cost = to_decimal(variable_cost)

        if revenue == ZERO:
            return None

        contribution = revenue - variable_cost

        return (
            contribution / revenue
        ) * HUNDRED

    # ------------------------------------------------------------------
    # Break-even
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_break_even_revenue(
        fixed_cost: NumberLike,
        contribution_margin: Optional[NumberLike],
    ) -> Optional[Decimal]:
        """
        Calculate break-even revenue.

            Fixed Cost / Contribution Margin

        contribution_margin is expected as a percentage.
        """

        if contribution_margin is None:
            return None

        fixed_cost = to_decimal(fixed_cost)
        margin_percent = to_decimal(contribution_margin)

        if margin_percent <= ZERO:
            return None

        margin_ratio = margin_percent / HUNDRED

        return fixed_cost / margin_ratio

    @staticmethod
    def calculate_break_even(
        fixed_cost: NumberLike,
        variable_cost_per_unit: NumberLike,
        selling_price_per_unit: NumberLike,
    ) -> BreakEvenResult:
        """Calculate unit and revenue break-even points."""

        fixed_cost = to_decimal(fixed_cost)
        variable_cost_per_unit = to_decimal(
            variable_cost_per_unit
        )
        selling_price_per_unit = to_decimal(
            selling_price_per_unit
        )

        contribution_per_unit = (
            selling_price_per_unit
            - variable_cost_per_unit
        )

        if contribution_per_unit <= ZERO:
            return BreakEvenResult(
                fixed_cost=fixed_cost,
                variable_cost_per_unit=variable_cost_per_unit,
                selling_price_per_unit=selling_price_per_unit,
                contribution_per_unit=contribution_per_unit,
                break_even_units=None,
                break_even_revenue=None,
                contribution_margin_percent=None,
            )

        break_even_units = (
            fixed_cost / contribution_per_unit
        )

        break_even_revenue = (
            break_even_units
            * selling_price_per_unit
        )

        contribution_margin_percent = (
            contribution_per_unit
            / selling_price_per_unit
            * HUNDRED
            if selling_price_per_unit != ZERO
            else None
        )

        return BreakEvenResult(
            fixed_cost=fixed_cost,
            variable_cost_per_unit=variable_cost_per_unit,
            selling_price_per_unit=selling_price_per_unit,
            contribution_per_unit=contribution_per_unit,
            break_even_units=break_even_units,
            break_even_revenue=break_even_revenue,
            contribution_margin_percent=(
                contribution_margin_percent
            ),
        )

    # ------------------------------------------------------------------
    # Operating leverage
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_operating_leverage(
        revenue: NumberLike,
        variable_cost: NumberLike,
        operating_profit: NumberLike | None,
    ) -> Optional[Decimal]:
        """
        Calculate degree of operating leverage.

            Contribution / Operating Profit
        """

        if operating_profit is None:
            return None

        revenue = to_decimal(revenue)
        variable_cost = to_decimal(variable_cost)
        operating_profit = to_decimal(operating_profit)

        if operating_profit == ZERO:
            return None

        contribution = revenue - variable_cost

        return contribution / operating_profit

    # ------------------------------------------------------------------
    # Cost metrics
    # ------------------------------------------------------------------

    @staticmethod
    def build_metrics(
        costs: Iterable[CostItem],
        total_cost: NumberLike,
        revenue: NumberLike,
    ) -> list[CostMetric]:
        """Build detailed metrics for every cost item."""

        total_cost = to_decimal(total_cost)
        revenue = to_decimal(revenue)

        metrics: list[CostMetric] = []

        for cost in costs:
            amount = to_decimal(cost.amount)

            budget = (
                to_decimal(cost.budget)
                if cost.budget is not None
                else None
            )

            previous = (
                to_decimal(cost.previous_period)
                if cost.previous_period is not None
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
                trend = CostTrend.STABLE
            elif previous_change > ZERO:
                trend = CostTrend.INCREASING
            elif previous_change < ZERO:
                trend = CostTrend.DECREASING
            else:
                trend = CostTrend.STABLE

            metrics.append(
                CostMetric(
                    name=cost.name,
                    amount=amount,
                    percentage_of_total=percentage(
                        amount,
                        total_cost,
                    ),
                    percentage_of_revenue=percentage(
                        amount,
                        revenue,
                    ),
                    budget_variance=budget_variance,
                    budget_variance_percent=budget_variance_percent,
                    previous_period_change=previous_change,
                    previous_period_change_percent=(
                        previous_change_percent
                    ),
                    trend=trend,
                )
            )

        return metrics

    # ------------------------------------------------------------------
    # Cost drivers
    # ------------------------------------------------------------------

    @staticmethod
    def find_top_cost_drivers(
        costs: Iterable[CostItem],
        limit: int = 5,
    ) -> list[str]:
        """Return the largest cost categories/items."""

        if limit <= 0:
            return []

        ordered = sorted(
            costs,
            key=lambda item: to_decimal(item.amount),
            reverse=True,
        )

        return [
            cost.name
            for cost in ordered[:limit]
        ]

    @staticmethod
    def cost_concentration(
        costs: Iterable[CostItem],
        top_n: int = 3,
    ) -> Optional[Decimal]:
        """
        Calculate what percentage of total cost is represented
        by the largest N cost items.
        """

        costs = tuple(costs)

        if not costs:
            return None

        if top_n <= 0:
            return ZERO

        total = CostAnalyzer.total_cost(costs)

        if total == ZERO:
            return None

        largest = sorted(
            (
                to_decimal(cost.amount)
                for cost in costs
            ),
            reverse=True,
        )[:top_n]

        return (
            sum(largest, ZERO)
            / total
        ) * HUNDRED

    # ------------------------------------------------------------------
    # Budget analysis
    # ------------------------------------------------------------------

    @staticmethod
    def compare_to_budget(
        cost: CostItem,
    ) -> Optional[CostVarianceResult]:
        """
        Compare a cost against its budget.

        For costs:
            Actual > Budget = unfavorable
            Actual < Budget = favorable
        """

        if cost.budget is None:
            return None

        actual = to_decimal(cost.amount)
        budget = to_decimal(cost.budget)

        variance = actual - budget

        variance_percent = percentage(
            actual,
            budget,
        )

        favorable = variance <= ZERO

        if variance > ZERO:
            explanation = (
                f"{cost.name} is over budget by "
                f"{variance:,.2f}"
            )
        elif variance < ZERO:
            explanation = (
                f"{cost.name} is under budget by "
                f"{abs(variance):,.2f}"
            )
        else:
            explanation = (
                f"{cost.name} is exactly on budget."
            )

        return CostVarianceResult(
            name=cost.name,
            actual=actual,
            budget=budget,
            variance=variance,
            variance_percent=variance_percent,
            favorable=favorable,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    @staticmethod
    def classify_efficiency(
        cost_to_revenue_ratio: Optional[NumberLike],
        efficient_threshold: NumberLike = Decimal("60"),
        inefficient_threshold: NumberLike = Decimal("80"),
    ) -> CostEfficiency:
        """
        Classify cost efficiency based on cost/revenue percentage.

        Default:
            <= 60% -> efficient
            <= 80% -> acceptable
            > 80%  -> inefficient

        These thresholds are generic defaults and should be
        overridden with industry-specific benchmarks in production.
        """

        if cost_to_revenue_ratio is None:
            return CostEfficiency.ACCEPTABLE

        ratio = to_decimal(cost_to_revenue_ratio)

        if ratio <= to_decimal(efficient_threshold):
            return CostEfficiency.EFFICIENT

        if ratio <= to_decimal(inefficient_threshold):
            return CostEfficiency.ACCEPTABLE

        return CostEfficiency.INEFFICIENT

    # ------------------------------------------------------------------
    # Cost reduction opportunities
    # ------------------------------------------------------------------

    @staticmethod
    def identify_reduction_opportunities(
        costs: Iterable[CostItem],
        reduction_percent: NumberLike = Decimal("10"),
        high_cost_threshold_percent: NumberLike = Decimal("15"),
    ) -> list[CostOpportunity]:
        """
        Identify large cost items that may represent
        cost-reduction opportunities.

        This is a screening mechanism, not an automatic
        recommendation to cut a cost.
        """

        costs = tuple(costs)

        total_cost = CostAnalyzer.total_cost(costs)

        if total_cost <= ZERO:
            return []

        reduction_percent = abs(
            to_decimal(reduction_percent)
        )

        threshold = abs(
            to_decimal(high_cost_threshold_percent)
        )

        opportunities: list[CostOpportunity] = []

        for cost in costs:
            amount = to_decimal(cost.amount)

            share = percentage(
                amount,
                total_cost,
            )

            if share is None or share < threshold:
                continue

            estimated_savings = (
                amount
                * reduction_percent
                / HUNDRED
            )

            if share >= Decimal("30"):
                priority = OpportunityPriority.HIGH
            elif share >= Decimal("20"):
                priority = OpportunityPriority.MEDIUM
            else:
                priority = OpportunityPriority.LOW

            opportunities.append(
                CostOpportunity(
                    name=cost.name,
                    current_cost=amount,
                    estimated_reduction_percent=reduction_percent,
                    estimated_savings=estimated_savings,
                    priority=priority,
                    reason=(
                        f"{cost.name} represents "
                        f"{share:.2f}% of total cost."
                    ),
                )
            )

        opportunities.sort(
            key=lambda opportunity: opportunity.estimated_savings,
            reverse=True,
        )

        return opportunities

    # ------------------------------------------------------------------
    # Cost growth
    # ------------------------------------------------------------------

    @staticmethod
    def cost_growth(
        current_cost: NumberLike,
        previous_cost: NumberLike,
    ) -> Optional[Decimal]:
        """Calculate cost growth percentage."""

        return percentage(
            to_decimal(current_cost),
            to_decimal(previous_cost),
        )

    @staticmethod
    def is_significant_cost_increase(
        current_cost: NumberLike,
        previous_cost: NumberLike,
        threshold_percent: NumberLike = Decimal("10"),
    ) -> bool:
        """
        Determine whether cost increased significantly.

        Intended for the alert engine.
        """

        growth = CostAnalyzer.cost_growth(
            current_cost,
            previous_cost,
        )

        if growth is None:
            return False

        return growth >= abs(
            to_decimal(threshold_percent)
        )

    @staticmethod
    def is_significant_cost_decrease(
        current_cost: NumberLike,
        previous_cost: NumberLike,
        threshold_percent: NumberLike = Decimal("10"),
    ) -> bool:
        """Determine whether cost decreased significantly."""

        growth = CostAnalyzer.cost_growth(
            current_cost,
            previous_cost,
        )

        if growth is None:
            return False

        return growth <= -abs(
            to_decimal(threshold_percent)
        )

    # ------------------------------------------------------------------
    # Department / cost-center analysis
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_by_department(
        costs: Iterable[CostItem],
    ) -> dict[str, Decimal]:
        """Aggregate costs by department."""

        result: dict[str, Decimal] = {}

        for cost in costs:
            department = cost.department or "Unassigned"

            result[department] = (
                result.get(department, ZERO)
                + to_decimal(cost.amount)
            )

        return result

    @staticmethod
    def aggregate_by_cost_center(
        costs: Iterable[CostItem],
    ) -> dict[str, Decimal]:
        """Aggregate costs by cost center."""

        result: dict[str, Decimal] = {}

        for cost in costs:
            cost_center = cost.cost_center or "Unassigned"

            result[cost_center] = (
                result.get(cost_center, ZERO)
                + to_decimal(cost.amount)
            )

        return result

    # ------------------------------------------------------------------
    # Scenario / what-if support
    # ------------------------------------------------------------------

    @staticmethod
    def simulate_cost_reduction(
        costs: Iterable[CostItem],
        reduction_percent: NumberLike,
        target_names: Optional[Iterable[str]] = None,
    ) -> dict[str, Decimal]:
        """
        Simulate a cost reduction scenario.

        Returns projected cost for each item.

        This function does not mutate the original CostItem objects.
        """

        costs = tuple(costs)

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

        for cost in costs:
            amount = to_decimal(cost.amount)

            if targets is None or cost.name in targets:
                projected[cost.name] = (
                    amount * multiplier
                )
            else:
                projected[cost.name] = amount

        return projected

    @staticmethod
    def calculate_projected_total(
        costs: Iterable[CostItem],
        reduction_percent: NumberLike,
        target_names: Optional[Iterable[str]] = None,
    ) -> Decimal:
        """Calculate total cost after a simulated reduction."""

        projected = CostAnalyzer.simulate_cost_reduction(
            costs=costs,
            reduction_percent=reduction_percent,
            target_names=target_names,
        )

        return sum(
            projected.values(),
            ZERO,
        )

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_summary(
        result: CostAnalysisResult,
    ) -> str:
        """Generate a concise natural-language cost summary."""

        total = result.total_cost

        if result.cost_to_revenue_ratio is None:
            ratio_text = "not available"
        else:
            ratio_text = (
                f"{result.cost_to_revenue_ratio:.2f}%"
            )

        summary = (
            f"Total cost is {total:,.2f}, representing "
            f"{ratio_text} of revenue."
        )

        if result.top_cost_drivers:
            summary += (
                " Top cost drivers: "
                + ", ".join(result.top_cost_drivers)
                + "."
            )

        summary += (
            f" Overall cost efficiency is "
            f"{result.efficiency.value}."
        )

        return summary


# ============================================================================
# Convenience functions
# ============================================================================

def analyze_costs(
    costs: Iterable[CostItem],
    revenue: NumberLike = ZERO,
    units_sold: NumberLike | None = None,
    gross_profit: NumberLike | None = None,
    operating_profit: NumberLike | None = None,
) -> CostAnalysisResult:
    """Convenience wrapper for complete cost analysis."""

    analyzer = CostAnalyzer()

    return analyzer.analyze(
        costs=costs,
        revenue=revenue,
        units_sold=units_sold,
        gross_profit=gross_profit,
        operating_profit=operating_profit,
    )


def calculate_total_cost(
    costs: Iterable[CostItem],
) -> Decimal:
    """Calculate total cost."""

    return CostAnalyzer.total_cost(costs)


def calculate_cost_ratio(
    total_cost: NumberLike,
    revenue: NumberLike,
) -> Optional[Decimal]:
    """Calculate cost-to-revenue percentage."""

    return percentage(
        total_cost,
        revenue,
    )


def calculate_cost_per_unit(
    total_cost: NumberLike,
    units: NumberLike,
) -> Optional[Decimal]:
    """Calculate cost per unit."""

    return CostAnalyzer.calculate_cost_per_unit(
        total_cost,
        units,
    )


def calculate_break_even(
    fixed_cost: NumberLike,
    variable_cost_per_unit: NumberLike,
    selling_price_per_unit: NumberLike,
) -> BreakEvenResult:
    """Calculate break-even analysis."""

    return CostAnalyzer.calculate_break_even(
        fixed_cost=fixed_cost,
        variable_cost_per_unit=variable_cost_per_unit,
        selling_price_per_unit=selling_price_per_unit,
    )


def identify_cost_opportunities(
    costs: Iterable[CostItem],
    reduction_percent: NumberLike = Decimal("10"),
) -> list[CostOpportunity]:
    """Identify potential cost-reduction opportunities."""

    return CostAnalyzer.identify_reduction_opportunities(
        costs=costs,
        reduction_percent=reduction_percent,
    )