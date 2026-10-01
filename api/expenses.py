# backend/app/financial/expenses.py

"""
FinCo AI - Expense Analysis Engine

Purpose
-------
Deterministic financial expense analysis for the FinCo AI platform.

Responsibilities
----------------
- Calculate total expenses
- Analyze expense categories
- Calculate operating expense metrics
- Compare expenses across periods
- Calculate expense growth
- Analyze expense concentration
- Compare actual expenses with budget
- Detect cost pressure
- Generate warnings and strengths
- Produce a normalized analysis result

Design
------
Financial calculations are deterministic and should NOT be delegated
to an LLM.

The output of this module can be consumed by:
    Financial Agent
    Alert Engine
    Forecasting Engine
    Recommendation Engine
    What-If Engine
    Reports
    Copilot API

Input convention
----------------
Expense amounts should normally be supplied as positive values.

Example:
    salaries = 50000
    rent = 10000
    marketing = 5000

The analyzer handles Decimal conversion internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Mapping, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZERO = Decimal("0")
ONE_HUNDRED = Decimal("100")
TWO_PLACES = Decimal("0.01")


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def to_decimal(value: Any, default: Decimal = ZERO) -> Decimal:
    """
    Safely convert a value to Decimal.

    Supports:
        int
        float
        str
        Decimal
        None
    """
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def safe_divide(
    numerator: Any,
    denominator: Any,
    default: Decimal = ZERO,
) -> Decimal:
    """
    Safely divide two numeric values.

    Returns default when denominator is zero.
    """
    num = to_decimal(numerator)
    den = to_decimal(denominator)

    if den == ZERO:
        return default

    return num / den


def percentage(
    numerator: Any,
    denominator: Any,
    decimals: int = 2,
) -> Decimal:
    """
    Calculate:

        numerator / denominator * 100
    """
    value = safe_divide(numerator, denominator) * ONE_HUNDRED
    return value.quantize(
        Decimal("1." + ("0" * decimals)),
        rounding=ROUND_HALF_UP,
    )


def round_decimal(
    value: Any,
    decimals: int = 2,
) -> Decimal:
    """Round a value to the requested decimal places."""
    number = to_decimal(value)

    quantizer = Decimal(
        "1." + ("0" * decimals)
    )

    return number.quantize(
        quantizer,
        rounding=ROUND_HALF_UP,
    )


def normalize_category(value: Any) -> str:
    """Normalize an expense category name."""
    if value is None:
        return "Uncategorized"

    category = str(value).strip()

    if not category:
        return "Uncategorized"

    return category


# ---------------------------------------------------------------------------
# Expense Data
# ---------------------------------------------------------------------------

@dataclass
class ExpenseData:
    """
    Represents expense information for one financial period.

    All expense amounts are expected to be positive.

    Example
    -------
    ExpenseData(
        salaries=50000,
        rent=10000,
        utilities=5000,
        marketing=8000,
        technology=7000,
        travel=3000,
        other_operating_expenses=2000,
        interest_expense=1500,
        tax_expense=4000,
        revenue=150000,
        budget_expenses=85000,
        period="2026-08",
    )
    """

    salaries: Decimal = ZERO
    wages: Decimal = ZERO
    rent: Decimal = ZERO
    utilities: Decimal = ZERO
    marketing: Decimal = ZERO
    advertising: Decimal = ZERO
    technology: Decimal = ZERO
    software: Decimal = ZERO
    travel: Decimal = ZERO
    professional_services: Decimal = ZERO
    insurance: Decimal = ZERO
    maintenance: Decimal = ZERO
    logistics: Decimal = ZERO
    depreciation: Decimal = ZERO
    amortization: Decimal = ZERO
    research_and_development: Decimal = ZERO
    other_operating_expenses: Decimal = ZERO

    interest_expense: Decimal = ZERO
    tax_expense: Decimal = ZERO
    other_non_operating_expenses: Decimal = ZERO

    revenue: Decimal = ZERO

    budget_expenses: Decimal = ZERO

    currency: str = "USD"
    period: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Normalize numeric values into Decimal.
        """
        numeric_fields = [
            "salaries",
            "wages",
            "rent",
            "utilities",
            "marketing",
            "advertising",
            "technology",
            "software",
            "travel",
            "professional_services",
            "insurance",
            "maintenance",
            "logistics",
            "depreciation",
            "amortization",
            "research_and_development",
            "other_operating_expenses",
            "interest_expense",
            "tax_expense",
            "other_non_operating_expenses",
            "revenue",
            "budget_expenses",
        ]

        for name in numeric_fields:
            setattr(
                self,
                name,
                to_decimal(getattr(self, name)),
            )


# ---------------------------------------------------------------------------
# Expense Analysis Result
# ---------------------------------------------------------------------------

@dataclass
class ExpenseAnalysis:
    """
    Complete deterministic expense analysis result.
    """

    total_operating_expenses: Decimal = ZERO
    total_non_operating_expenses: Decimal = ZERO
    total_expenses: Decimal = ZERO

    revenue: Decimal = ZERO

    operating_expense_ratio: Decimal = ZERO
    total_expense_ratio: Decimal = ZERO

    expense_growth: Optional[Decimal] = None

    budget: Decimal = ZERO
    budget_variance: Decimal = ZERO
    budget_variance_percentage: Decimal = ZERO

    category_breakdown: Dict[str, Decimal] = field(
        default_factory=dict
    )

    category_percentages: Dict[str, Decimal] = field(
        default_factory=dict
    )

    top_expense_categories: List[Dict[str, Any]] = field(
        default_factory=list
    )

    concentration_ratio: Decimal = ZERO

    cost_pressure_score: Decimal = ZERO
    health_score: Decimal = ZERO

    warnings: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    status: str = "healthy"

    currency: str = "USD"
    period: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Expense Analyzer
# ---------------------------------------------------------------------------

class ExpenseAnalyzer:
    """
    Deterministic expense analysis engine.
    """

    OPERATING_CATEGORIES = (
        "salaries",
        "wages",
        "rent",
        "utilities",
        "marketing",
        "advertising",
        "technology",
        "software",
        "travel",
        "professional_services",
        "insurance",
        "maintenance",
        "logistics",
        "depreciation",
        "amortization",
        "research_and_development",
        "other_operating_expenses",
    )

    NON_OPERATING_CATEGORIES = (
        "interest_expense",
        "tax_expense",
        "other_non_operating_expenses",
    )

    # -----------------------------------------------------------------------
    # Category calculations
    # -----------------------------------------------------------------------

    def operating_category_values(
        self,
        data: ExpenseData,
    ) -> Dict[str, Decimal]:
        """
        Return operating expense categories with their values.
        """
        return {
            category: to_decimal(
                getattr(data, category, ZERO)
            )
            for category in self.OPERATING_CATEGORIES
        }

    def non_operating_category_values(
        self,
        data: ExpenseData,
    ) -> Dict[str, Decimal]:
        """
        Return non-operating expense categories.
        """
        return {
            category: to_decimal(
                getattr(data, category, ZERO)
            )
            for category in self.NON_OPERATING_CATEGORIES
        }

    def total_operating_expenses(
        self,
        data: ExpenseData,
    ) -> Decimal:
        """Calculate total operating expenses."""
        values = self.operating_category_values(data)

        return sum(
            values.values(),
            ZERO,
        )

    def total_non_operating_expenses(
        self,
        data: ExpenseData,
    ) -> Decimal:
        """Calculate total non-operating expenses."""
        values = self.non_operating_category_values(data)

        return sum(
            values.values(),
            ZERO,
        )

    def total_expenses(
        self,
        data: ExpenseData,
    ) -> Decimal:
        """Calculate total expenses."""
        return (
            self.total_operating_expenses(data)
            + self.total_non_operating_expenses(data)
        )

    # -----------------------------------------------------------------------
    # Revenue ratios
    # -----------------------------------------------------------------------

    def operating_expense_ratio(
        self,
        data: ExpenseData,
    ) -> Decimal:
        """
        Operating expenses as a percentage of revenue.
        """
        return percentage(
            self.total_operating_expenses(data),
            data.revenue,
        )

    def total_expense_ratio(
        self,
        data: ExpenseData,
    ) -> Decimal:
        """
        Total expenses as a percentage of revenue.
        """
        return percentage(
            self.total_expenses(data),
            data.revenue,
        )

    # -----------------------------------------------------------------------
    # Category analysis
    # -----------------------------------------------------------------------

    def category_breakdown(
        self,
        data: ExpenseData,
        include_zero: bool = False,
    ) -> Dict[str, Decimal]:
        """
        Return expense category breakdown.

        By default, zero-value categories are excluded.
        """
        values = {
            **self.operating_category_values(data),
            **self.non_operating_category_values(data),
        }

        if not include_zero:
            values = {
                category: amount
                for category, amount in values.items()
                if amount != ZERO
            }

        return dict(
            sorted(
                values.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def category_percentages(
        self,
        data: ExpenseData,
    ) -> Dict[str, Decimal]:
        """
        Calculate each category's percentage of total expenses.
        """
        total = self.total_expenses(data)

        breakdown = self.category_breakdown(data)

        return {
            category: percentage(
                amount,
                total,
            )
            for category, amount in breakdown.items()
        }

    def top_categories(
        self,
        data: ExpenseData,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return the largest expense categories.
        """
        breakdown = self.category_breakdown(data)
        percentages = self.category_percentages(data)

        result: List[Dict[str, Any]] = []

        for category, amount in list(
            breakdown.items()
        )[:limit]:
            result.append(
                {
                    "category": category,
                    "amount": round_decimal(amount),
                    "percentage": percentages.get(
                        category,
                        ZERO,
                    ),
                }
            )

        return result

    # -----------------------------------------------------------------------
    # Expense concentration
    # -----------------------------------------------------------------------

    def concentration_ratio(
        self,
        data: ExpenseData,
        top_n: int = 3,
    ) -> Decimal:
        """
        Calculate the percentage of total expenses represented
        by the largest N expense categories.

        Example:
            Top 3 expenses = $70,000
            Total expenses = $100,000

            concentration = 70%
        """
        total = self.total_expenses(data)

        if total == ZERO:
            return ZERO

        top = self.top_categories(
            data,
            limit=top_n,
        )

        top_amount = sum(
            (
                to_decimal(item["amount"])
                for item in top
            ),
            ZERO,
        )

        return percentage(
            top_amount,
            total,
        )

    # -----------------------------------------------------------------------
    # Expense growth
    # -----------------------------------------------------------------------

    def expense_growth(
        self,
        current: ExpenseData,
        previous: ExpenseData,
    ) -> Decimal:
        """
        Calculate total expense growth percentage.
        """
        current_total = self.total_expenses(current)
        previous_total = self.total_expenses(previous)

        if previous_total == ZERO:
            return ZERO

        return percentage(
            current_total - previous_total,
            previous_total,
        )

    def operating_expense_growth(
        self,
        current: ExpenseData,
        previous: ExpenseData,
    ) -> Decimal:
        """
        Calculate operating expense growth percentage.
        """
        current_total = self.total_operating_expenses(
            current
        )

        previous_total = self.total_operating_expenses(
            previous
        )

        if previous_total == ZERO:
            return ZERO

        return percentage(
            current_total - previous_total,
            previous_total,
        )

    # -----------------------------------------------------------------------
    # Budget analysis
    # -----------------------------------------------------------------------

    def budget_variance(
        self,
        data: ExpenseData,
    ) -> Decimal:
        """
        Calculate actual expenses minus budget.

        Positive:
            over budget

        Negative:
            under budget
        """
        return (
            self.total_expenses(data)
            - data.budget_expenses
        )

    def budget_variance_percentage(
        self,
        data: ExpenseData,
    ) -> Decimal:
        """Calculate budget variance as a percentage."""
        if data.budget_expenses == ZERO:
            return ZERO

        return percentage(
            self.budget_variance(data),
            data.budget_expenses,
        )

    def is_over_budget(
        self,
        data: ExpenseData,
    ) -> bool:
        """Return True when actual expenses exceed budget."""
        return (
            data.budget_expenses > ZERO
            and self.total_expenses(data)
            > data.budget_expenses
        )

    # -----------------------------------------------------------------------
    # Cost pressure
    # -----------------------------------------------------------------------

    def cost_pressure_score(
        self,
        data: ExpenseData,
    ) -> Decimal:
        """
        Calculate a portfolio/demo cost-pressure score.

        Score:
            0   = low pressure
            100 = very high pressure

        This is NOT an accounting standard.
        """
        score = Decimal("0")

        expense_ratio = self.total_expense_ratio(data)

        if expense_ratio >= Decimal("100"):
            score += Decimal("40")
        elif expense_ratio >= Decimal("90"):
            score += Decimal("30")
        elif expense_ratio >= Decimal("80"):
            score += Decimal("20")
        elif expense_ratio >= Decimal("70"):
            score += Decimal("10")

        variance_pct = abs(
            self.budget_variance_percentage(data)
        )

        if self.is_over_budget(data):
            if variance_pct >= Decimal("20"):
                score += Decimal("30")
            elif variance_pct >= Decimal("10"):
                score += Decimal("20")
            elif variance_pct >= Decimal("5"):
                score += Decimal("10")

        concentration = self.concentration_ratio(
            data,
            top_n=3,
        )

        if concentration >= Decimal("80"):
            score += Decimal("20")
        elif concentration >= Decimal("70"):
            score += Decimal("15")
        elif concentration >= Decimal("60"):
            score += Decimal("10")

        return min(
            round_decimal(score),
            Decimal("100"),
        )

    # -----------------------------------------------------------------------
    # Warnings
    # -----------------------------------------------------------------------

    def warnings(
        self,
        data: ExpenseData,
        previous: Optional[ExpenseData] = None,
    ) -> List[str]:
        """
        Generate deterministic expense warnings.
        """
        warnings: List[str] = []

        total = self.total_expenses(data)
        expense_ratio = self.total_expense_ratio(data)

        # No revenue
        if data.revenue <= ZERO and total > ZERO:
            warnings.append(
                "Expenses are recorded but revenue is zero or unavailable."
            )

        # Expense-to-revenue pressure
        if expense_ratio >= Decimal("100"):
            warnings.append(
                "Total expenses are equal to or greater than revenue."
            )
        elif expense_ratio >= Decimal("90"):
            warnings.append(
                "Total expenses consume 90% or more of revenue."
            )
        elif expense_ratio >= Decimal("80"):
            warnings.append(
                "Total expenses consume 80% or more of revenue."
            )

        # Budget
        if self.is_over_budget(data):
            variance = self.budget_variance_percentage(data)

            if variance >= Decimal("20"):
                warnings.append(
                    "Expenses are more than 20% above budget."
                )
            elif variance >= Decimal("10"):
                warnings.append(
                    "Expenses are more than 10% above budget."
                )
            else:
                warnings.append(
                    "Expenses are above the approved budget."
                )

        # Growth
        if previous is not None:
            growth = self.expense_growth(
                data,
                previous,
            )

            if growth >= Decimal("20"):
                warnings.append(
                    "Total expenses increased by more than 20% "
                    "versus the previous period."
                )
            elif growth >= Decimal("10"):
                warnings.append(
                    "Total expenses increased significantly "
                    "versus the previous period."
                )

        # Concentration
        concentration = self.concentration_ratio(
            data,
            top_n=3,
        )

        if concentration >= Decimal("80"):
            warnings.append(
                "The top three expense categories represent "
                "80% or more of total expenses."
            )

        # Individual category pressure
        for item in self.top_categories(data, limit=3):
            if item["percentage"] >= Decimal("50"):
                warnings.append(
                    f"{item['category']} represents more than "
                    "50% of total expenses."
                )

        return warnings

    # -----------------------------------------------------------------------
    # Strengths
    # -----------------------------------------------------------------------

    def strengths(
        self,
        data: ExpenseData,
        previous: Optional[ExpenseData] = None,
    ) -> List[str]:
        """
        Generate positive expense-management indicators.
        """
        strengths: List[str] = []

        total = self.total_expenses(data)

        if total == ZERO:
            strengths.append(
                "No expenses were recorded for this period."
            )
            return strengths

        # Budget discipline
        if (
            data.budget_expenses > ZERO
            and not self.is_over_budget(data)
        ):
            strengths.append(
                "Total expenses are within the approved budget."
            )

        # Low expense ratio
        expense_ratio = self.total_expense_ratio(data)

        if (
            data.revenue > ZERO
            and expense_ratio < Decimal("70")
        ):
            strengths.append(
                "Total expenses remain below 70% of revenue."
            )

        # Expense reduction
        if previous is not None:
            growth = self.expense_growth(
                data,
                previous,
            )

            if growth < ZERO:
                strengths.append(
                    "Total expenses decreased versus "
                    "the previous period."
                )

        # Diversification
        concentration = self.concentration_ratio(
            data,
            top_n=3,
        )

        if concentration < Decimal("60"):
            strengths.append(
                "Expense concentration is relatively diversified."
            )

        return strengths

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def status(
        self,
        data: ExpenseData,
        previous: Optional[ExpenseData] = None,
    ) -> str:
        """
        Determine a simple expense health status.

        Values:
            healthy
            watch
            elevated
            critical
        """
        score = self.cost_pressure_score(data)

        if score >= Decimal("70"):
            return "critical"

        if score >= Decimal("45"):
            return "elevated"

        if score >= Decimal("20"):
            return "watch"

        # Previous-period deterioration can trigger watch.
        if previous is not None:
            growth = self.expense_growth(
                data,
                previous,
            )

            if growth >= Decimal("20"):
                return "watch"

        return "healthy"

    # -----------------------------------------------------------------------
    # Health score
    # -----------------------------------------------------------------------

    def calculate_health_score(
        self,
        data: ExpenseData,
        previous: Optional[ExpenseData] = None,
    ) -> Decimal:
        """
        Calculate a 0-100 expense health score.

        Higher is better.

        This is a portfolio/demo scoring model and is NOT
        a universal accounting standard.
        """
        pressure = self.cost_pressure_score(data)

        score = Decimal("100") - pressure

        # Reward expense reduction.
        if previous is not None:
            growth = self.expense_growth(
                data,
                previous,
            )

            if growth < ZERO:
                score += Decimal("5")

        return max(
            Decimal("0"),
            min(
                Decimal("100"),
                round_decimal(score),
            ),
        )

    # -----------------------------------------------------------------------
    # Main analysis
    # -----------------------------------------------------------------------

    def analyze(
        self,
        data: ExpenseData,
        previous: Optional[ExpenseData] = None,
    ) -> ExpenseAnalysis:
        """
        Perform complete expense analysis.
        """
        operating = self.total_operating_expenses(data)
        non_operating = self.total_non_operating_expenses(data)
        total = operating + non_operating

        budget = data.budget_expenses
        variance = self.budget_variance(data)

        growth: Optional[Decimal] = None

        if previous is not None:
            growth = self.expense_growth(
                data,
                previous,
            )

        analysis = ExpenseAnalysis(
            total_operating_expenses=round_decimal(
                operating
            ),
            total_non_operating_expenses=round_decimal(
                non_operating
            ),
            total_expenses=round_decimal(total),
            revenue=round_decimal(data.revenue),
            operating_expense_ratio=round_decimal(
                self.operating_expense_ratio(data)
            ),
            total_expense_ratio=round_decimal(
                self.total_expense_ratio(data)
            ),
            expense_growth=growth,
            budget=round_decimal(budget),
            budget_variance=round_decimal(variance),
            budget_variance_percentage=round_decimal(
                self.budget_variance_percentage(data)
            ),
            category_breakdown={
                key: round_decimal(value)
                for key, value
                in self.category_breakdown(data).items()
            },
            category_percentages={
                key: round_decimal(value)
                for key, value
                in self.category_percentages(data).items()
            },
            top_expense_categories=self.top_categories(
                data
            ),
            concentration_ratio=round_decimal(
                self.concentration_ratio(data)
            ),
            cost_pressure_score=round_decimal(
                self.cost_pressure_score(data)
            ),
            health_score=round_decimal(
                self.calculate_health_score(
                    data,
                    previous,
                )
            ),
            warnings=self.warnings(
                data,
                previous,
            ),
            strengths=self.strengths(
                data,
                previous,
            ),
            status=self.status(
                data,
                previous,
            ),
            currency=data.currency,
            period=data.period,
            metadata=dict(data.metadata),
        )

        return analysis


# ---------------------------------------------------------------------------
# Period Comparison
# ---------------------------------------------------------------------------

@dataclass
class ExpenseChange:
    """
    Expense comparison between two periods.
    """

    previous_total: Decimal
    current_total: Decimal

    absolute_change: Decimal
    percentage_change: Decimal

    previous_operating_expenses: Decimal
    current_operating_expenses: Decimal

    operating_absolute_change: Decimal
    operating_percentage_change: Decimal

    direction: str

    currency: str = "USD"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


def calculate_change(
    current: ExpenseData,
    previous: ExpenseData,
) -> ExpenseChange:
    """
    Compare current expenses with previous expenses.
    """
    analyzer = ExpenseAnalyzer()

    current_total = analyzer.total_expenses(current)
    previous_total = analyzer.total_expenses(previous)

    absolute_change = (
        current_total - previous_total
    )

    if previous_total == ZERO:
        percentage_change = ZERO
    else:
        percentage_change = percentage(
            absolute_change,
            previous_total,
        )

    current_operating = (
        analyzer.total_operating_expenses(current)
    )

    previous_operating = (
        analyzer.total_operating_expenses(previous)
    )

    operating_absolute_change = (
        current_operating
        - previous_operating
    )

    if previous_operating == ZERO:
        operating_percentage_change = ZERO
    else:
        operating_percentage_change = percentage(
            operating_absolute_change,
            previous_operating,
        )

    if absolute_change > ZERO:
        direction = "increased"
    elif absolute_change < ZERO:
        direction = "decreased"
    else:
        direction = "unchanged"

    return ExpenseChange(
        previous_total=round_decimal(
            previous_total
        ),
        current_total=round_decimal(
            current_total
        ),
        absolute_change=round_decimal(
            absolute_change
        ),
        percentage_change=round_decimal(
            percentage_change
        ),
        previous_operating_expenses=round_decimal(
            previous_operating
        ),
        current_operating_expenses=round_decimal(
            current_operating
        ),
        operating_absolute_change=round_decimal(
            operating_absolute_change
        ),
        operating_percentage_change=round_decimal(
            operating_percentage_change
        ),
        direction=direction,
        currency=current.currency,
    )


def compare_expenses(
    current: ExpenseData,
    previous: ExpenseData,
) -> ExpenseChange:
    """Alias for calculate_change()."""
    return calculate_change(
        current,
        previous,
    )


# ---------------------------------------------------------------------------
# Dictionary Conversion
# ---------------------------------------------------------------------------

def expense_data_from_dict(
    data: Mapping[str, Any],
) -> ExpenseData:
    """
    Build ExpenseData from a dictionary.

    Useful for:
        API payloads
        CSV/Excel normalization
        database records
        ML pipelines
        agent tools
    """
    allowed_fields = {
        "salaries",
        "wages",
        "rent",
        "utilities",
        "marketing",
        "advertising",
        "technology",
        "software",
        "travel",
        "professional_services",
        "insurance",
        "maintenance",
        "logistics",
        "depreciation",
        "amortization",
        "research_and_development",
        "other_operating_expenses",
        "interest_expense",
        "tax_expense",
        "other_non_operating_expenses",
        "revenue",
        "budget_expenses",
        "currency",
        "period",
        "metadata",
    }

    payload = {
        key: value
        for key, value in data.items()
        if key in allowed_fields
    }

    return ExpenseData(**payload)


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def analyze_expenses(
    data: Mapping[str, Any] | ExpenseData,
    previous: Optional[
        Mapping[str, Any] | ExpenseData
    ] = None,
) -> ExpenseAnalysis:
    """
    Convenience function for complete expense analysis.
    """
    if isinstance(data, ExpenseData):
        current_data = data
    else:
        current_data = expense_data_from_dict(data)

    previous_data: Optional[ExpenseData] = None

    if previous is not None:
        if isinstance(previous, ExpenseData):
            previous_data = previous
        else:
            previous_data = expense_data_from_dict(
                previous
            )

    analyzer = ExpenseAnalyzer()

    return analyzer.analyze(
        current_data,
        previous_data,
    )


def get_expense_metrics(
    data: Mapping[str, Any] | ExpenseData,
) -> Dict[str, Any]:
    """
    Return a compact metrics dictionary suitable for:

        API responses
        Agent context
        dashboards
        alerts
        reports
    """
    analysis = analyze_expenses(data)

    return {
        "total_operating_expenses": (
            analysis.total_operating_expenses
        ),
        "total_non_operating_expenses": (
            analysis.total_non_operating_expenses
        ),
        "total_expenses": analysis.total_expenses,
        "revenue": analysis.revenue,
        "operating_expense_ratio": (
            analysis.operating_expense_ratio
        ),
        "total_expense_ratio": (
            analysis.total_expense_ratio
        ),
        "budget": analysis.budget,
        "budget_variance": (
            analysis.budget_variance
        ),
        "budget_variance_percentage": (
            analysis.budget_variance_percentage
        ),
        "top_expense_categories": (
            analysis.top_expense_categories
        ),
        "concentration_ratio": (
            analysis.concentration_ratio
        ),
        "cost_pressure_score": (
            analysis.cost_pressure_score
        ),
        "health_score": analysis.health_score,
        "status": analysis.status,
        "warnings": analysis.warnings,
        "strengths": analysis.strengths,
        "currency": analysis.currency,
        "period": analysis.period,
    }


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    current = ExpenseData(
        salaries=Decimal("50000"),
        wages=Decimal("15000"),
        rent=Decimal("10000"),
        utilities=Decimal("4000"),
        marketing=Decimal("8000"),
        technology=Decimal("7000"),
        travel=Decimal("3000"),
        professional_services=Decimal("2000"),
        interest_expense=Decimal("1500"),
        tax_expense=Decimal("5000"),
        revenue=Decimal("150000"),
        budget_expenses=Decimal("100000"),
        currency="USD",
        period="2026-08",
    )

    previous = ExpenseData(
        salaries=Decimal("48000"),
        wages=Decimal("14000"),
        rent=Decimal("10000"),
        utilities=Decimal("4000"),
        marketing=Decimal("7000"),
        technology=Decimal("6000"),
        travel=Decimal("3000"),
        professional_services=Decimal("2000"),
        interest_expense=Decimal("1500"),
        tax_expense=Decimal("5000"),
        revenue=Decimal("145000"),
        budget_expenses=Decimal("95000"),
        currency="USD",
        period="2026-07",
    )

    analyzer = ExpenseAnalyzer()

    result = analyzer.analyze(
        current,
        previous,
    )

    print("Total Expenses:", result.total_expenses)
    print(
        "Operating Expenses:",
        result.total_operating_expenses,
    )
    print(
        "Expense Ratio:",
        result.total_expense_ratio,
    )
    print(
        "Budget Variance:",
        result.budget_variance,
    )
    print(
        "Health Score:",
        result.health_score,
    )
    print("Status:", result.status)

    print("\nTop Categories:")
    for category in result.top_expense_categories:
        print(category)

    print("\nWarnings:")
    for warning in result.warnings:
        print("-", warning)