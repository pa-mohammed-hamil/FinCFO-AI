"""
FinCo AI - What-If Financial Calculator

Purpose
-------
Deterministic financial calculations used by the What-If / Scenario
Simulation layer.

This module intentionally contains no LLM or agent logic. Agents can call
these functions through `what_if_tools.py`.

Example scenarios:
    - What if revenue increases by 10%?
    - What if expenses decrease by 5%?
    - What if both happen?
    - What if the company loses 15% of revenue?
    - What happens to profit margin?
    - What is the break-even revenue?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CalculatorError(ValueError):
    """Base exception for invalid financial calculations."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinancialSnapshot:
    """
    Represents the financial state before a scenario is applied.
    """

    revenue: float
    expenses: float
    cash_flow: Optional[float] = None

    def __post_init__(self) -> None:
        if self.revenue < 0:
            raise CalculatorError("Revenue cannot be negative.")

        if self.expenses < 0:
            raise CalculatorError("Expenses cannot be negative.")

    @property
    def profit(self) -> float:
        """Calculate operating profit."""
        return self.revenue - self.expenses

    @property
    def profit_margin(self) -> float:
        """Calculate profit margin as a percentage."""
        if self.revenue == 0:
            return 0.0

        return (self.profit / self.revenue) * 100


@dataclass(frozen=True)
class ScenarioResult:
    """
    Represents the result of applying a what-if scenario.
    """

    baseline_revenue: float
    baseline_expenses: float
    baseline_profit: float
    baseline_margin: float

    scenario_revenue: float
    scenario_expenses: float
    scenario_profit: float
    scenario_margin: float

    revenue_change: float
    expense_change: float
    profit_change: float
    margin_change: float

    revenue_change_pct: float
    expense_change_pct: float
    profit_change_pct: float


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_percentage(value: float, field_name: str = "percentage") -> None:
    """Validate a percentage value."""

    if not isinstance(value, (int, float)):
        raise CalculatorError(f"{field_name} must be numeric.")

    if value < -100:
        raise CalculatorError(
            f"{field_name} cannot be less than -100%."
        )


def _safe_percentage_change(
    baseline: float,
    current: float,
) -> float:
    """
    Calculate percentage change safely.

    Returns 0 when the baseline is zero.
    """

    if baseline == 0:
        return 0.0

    return ((current - baseline) / abs(baseline)) * 100


# ---------------------------------------------------------------------------
# Basic Financial Calculations
# ---------------------------------------------------------------------------

def calculate_profit(
    revenue: float,
    expenses: float,
) -> float:
    """
    Calculate profit.

    Formula:
        Profit = Revenue - Expenses
    """

    if revenue < 0:
        raise CalculatorError("Revenue cannot be negative.")

    if expenses < 0:
        raise CalculatorError("Expenses cannot be negative.")

    return revenue - expenses


def calculate_profit_margin(
    revenue: float,
    profit: float,
) -> float:
    """
    Calculate profit margin percentage.

    Formula:
        Margin = (Profit / Revenue) * 100
    """

    if revenue < 0:
        raise CalculatorError("Revenue cannot be negative.")

    if revenue == 0:
        return 0.0

    return (profit / revenue) * 100


def calculate_revenue_growth(
    previous_revenue: float,
    current_revenue: float,
) -> float:
    """
    Calculate revenue growth percentage.
    """

    if previous_revenue < 0 or current_revenue < 0:
        raise CalculatorError(
            "Revenue values cannot be negative."
        )

    return _safe_percentage_change(
        previous_revenue,
        current_revenue,
    )


def calculate_expense_growth(
    previous_expenses: float,
    current_expenses: float,
) -> float:
    """
    Calculate expense growth percentage.
    """

    if previous_expenses < 0 or current_expenses < 0:
        raise CalculatorError(
            "Expense values cannot be negative."
        )

    return _safe_percentage_change(
        previous_expenses,
        current_expenses,
    )


# ---------------------------------------------------------------------------
# Scenario Adjustments
# ---------------------------------------------------------------------------

def apply_percentage_change(
    value: float,
    percentage: float,
) -> float:
    """
    Apply a percentage change to a financial value.

    Example:
        apply_percentage_change(1000, 10)
        -> 1100

        apply_percentage_change(1000, -10)
        -> 900
    """

    if value < 0:
        raise CalculatorError("Value cannot be negative.")

    _validate_percentage(percentage)

    return value * (1 + percentage / 100)


def adjust_revenue(
    revenue: float,
    revenue_change_pct: float,
) -> float:
    """
    Apply a revenue percentage adjustment.
    """

    return apply_percentage_change(
        revenue,
        revenue_change_pct,
    )


def adjust_expenses(
    expenses: float,
    expense_change_pct: float,
) -> float:
    """
    Apply an expense percentage adjustment.
    """

    return apply_percentage_change(
        expenses,
        expense_change_pct,
    )


# ---------------------------------------------------------------------------
# What-If Scenario
# ---------------------------------------------------------------------------

def calculate_scenario(
    revenue: float,
    expenses: float,
    revenue_change_pct: float = 0.0,
    expense_change_pct: float = 0.0,
) -> ScenarioResult:
    """
    Calculate the financial impact of a what-if scenario.

    Parameters
    ----------
    revenue:
        Baseline revenue.

    expenses:
        Baseline expenses.

    revenue_change_pct:
        Percentage change applied to revenue.

    expense_change_pct:
        Percentage change applied to expenses.

    Example
    -------
    Revenue = 1,000,000
    Expenses = 800,000

    Revenue +10%
    Expenses -5%

    The function calculates the new revenue, expenses, profit,
    margins, and changes from baseline.
    """

    baseline = FinancialSnapshot(
        revenue=revenue,
        expenses=expenses,
    )

    scenario_revenue = adjust_revenue(
        revenue,
        revenue_change_pct,
    )

    scenario_expenses = adjust_expenses(
        expenses,
        expense_change_pct,
    )

    scenario_profit = calculate_profit(
        scenario_revenue,
        scenario_expenses,
    )

    scenario_margin = calculate_profit_margin(
        scenario_revenue,
        scenario_profit,
    )

    revenue_change = scenario_revenue - baseline.revenue
    expense_change = scenario_expenses - baseline.expenses
    profit_change = scenario_profit - baseline.profit
    margin_change = scenario_margin - baseline.profit_margin

    return ScenarioResult(
        baseline_revenue=baseline.revenue,
        baseline_expenses=baseline.expenses,
        baseline_profit=baseline.profit,
        baseline_margin=baseline.profit_margin,

        scenario_revenue=scenario_revenue,
        scenario_expenses=scenario_expenses,
        scenario_profit=scenario_profit,
        scenario_margin=scenario_margin,

        revenue_change=revenue_change,
        expense_change=expense_change,
        profit_change=profit_change,
        margin_change=margin_change,

        revenue_change_pct=_safe_percentage_change(
            baseline.revenue,
            scenario_revenue,
        ),

        expense_change_pct=_safe_percentage_change(
            baseline.expenses,
            scenario_expenses,
        ),

        profit_change_pct=_safe_percentage_change(
            baseline.profit,
            scenario_profit,
        ),
    )


# ---------------------------------------------------------------------------
# Break-Even Analysis
# ---------------------------------------------------------------------------

def calculate_break_even_revenue(
    fixed_costs: float,
    variable_cost_ratio: float,
) -> float:
    """
    Calculate break-even revenue.

    Formula:
        Break-even Revenue =
            Fixed Costs / Contribution Margin Ratio

    variable_cost_ratio:
        Variable costs as a decimal.

        Example:
            0.70 = 70% variable costs
            contribution margin = 30%
    """

    if fixed_costs < 0:
        raise CalculatorError(
            "Fixed costs cannot be negative."
        )

    if not 0 <= variable_cost_ratio < 1:
        raise CalculatorError(
            "Variable cost ratio must be between 0 and 1."
        )

    contribution_margin = 1 - variable_cost_ratio

    return fixed_costs / contribution_margin


def calculate_break_even_units(
    fixed_costs: float,
    selling_price: float,
    variable_cost_per_unit: float,
) -> float:
    """
    Calculate the number of units required to break even.
    """

    if fixed_costs < 0:
        raise CalculatorError(
            "Fixed costs cannot be negative."
        )

    if selling_price <= 0:
        raise CalculatorError(
            "Selling price must be greater than zero."
        )

    if variable_cost_per_unit < 0:
        raise CalculatorError(
            "Variable cost cannot be negative."
        )

    contribution_per_unit = (
        selling_price - variable_cost_per_unit
    )

    if contribution_per_unit <= 0:
        raise CalculatorError(
            "Selling price must exceed variable cost per unit."
        )

    return fixed_costs / contribution_per_unit


# ---------------------------------------------------------------------------
# Cash Flow Calculations
# ---------------------------------------------------------------------------

def calculate_cash_flow_change(
    operating_cash_flow: float,
    investing_cash_flow: float,
    financing_cash_flow: float,
) -> float:
    """
    Calculate net cash flow.

    Formula:
        Net Cash Flow =
            Operating + Investing + Financing
    """

    return (
        operating_cash_flow
        + investing_cash_flow
        + financing_cash_flow
    )


def project_cash_balance(
    current_cash: float,
    monthly_cash_flow: float,
    months: int,
) -> float:
    """
    Project future cash balance using a constant monthly cash flow.
    """

    if current_cash < 0:
        raise CalculatorError(
            "Current cash cannot be negative."
        )

    if months < 0:
        raise CalculatorError(
            "Months cannot be negative."
        )

    return current_cash + (monthly_cash_flow * months)


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

def calculate_sensitivity(
    revenue: float,
    expenses: float,
    revenue_changes: list[float],
    expense_changes: list[float],
) -> list[ScenarioResult]:
    """
    Generate multiple what-if scenarios.

    Example:

        revenue_changes = [-10, 0, 10]
        expense_changes = [-10, 0, 10]

    This produces 9 scenarios.
    """

    results: list[ScenarioResult] = []

    for revenue_change in revenue_changes:
        for expense_change in expense_changes:
            results.append(
                calculate_scenario(
                    revenue=revenue,
                    expenses=expenses,
                    revenue_change_pct=revenue_change,
                    expense_change_pct=expense_change,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Target Profit Analysis
# ---------------------------------------------------------------------------

def calculate_required_revenue_for_profit(
    target_profit: float,
    expense_ratio: float,
) -> float:
    """
    Calculate revenue required to achieve a target profit.

    Formula:

        Profit = Revenue - (Revenue * Expense Ratio)

        Revenue =
            Target Profit / (1 - Expense Ratio)
    """

    if not 0 <= expense_ratio < 1:
        raise CalculatorError(
            "Expense ratio must be between 0 and 1."
        )

    if target_profit < 0:
        raise CalculatorError(
            "Target profit cannot be negative."
        )

    margin = 1 - expense_ratio

    if margin <= 0:
        raise CalculatorError(
            "Expense ratio leaves no profit margin."
        )

    return target_profit / margin


# ---------------------------------------------------------------------------
# Loss Risk Analysis
# ---------------------------------------------------------------------------

def calculate_loss_risk(
    revenue: float,
    expenses: float,
    revenue_decline_pct: float = 0.0,
    expense_increase_pct: float = 0.0,
) -> dict:
    """
    Estimate whether a scenario produces a loss.

    This is a deterministic financial calculation and is not an ML
    probability model.
    """

    scenario = calculate_scenario(
        revenue=revenue,
        expenses=expenses,
        revenue_change_pct=-abs(revenue_decline_pct),
        expense_change_pct=abs(expense_increase_pct),
    )

    loss = scenario.scenario_profit < 0

    return {
        "loss_detected": loss,
        "baseline_profit": scenario.baseline_profit,
        "scenario_profit": scenario.scenario_profit,
        "profit_change": scenario.profit_change,
        "baseline_margin": scenario.baseline_margin,
        "scenario_margin": scenario.scenario_margin,
        "margin_change": scenario.margin_change,
    }


# ---------------------------------------------------------------------------
# Financial Health Helpers
# ---------------------------------------------------------------------------

def calculate_operating_margin(
    revenue: float,
    operating_expenses: float,
) -> float:
    """
    Calculate operating margin percentage.
    """

    if revenue <= 0:
        return 0.0

    operating_profit = revenue - operating_expenses

    return (
        operating_profit / revenue
    ) * 100


def calculate_cost_to_revenue_ratio(
    expenses: float,
    revenue: float,
) -> float:
    """
    Calculate expenses as a percentage of revenue.
    """

    if revenue <= 0:
        return 0.0

    return (expenses / revenue) * 100


def calculate_profit_change(
    baseline_profit: float,
    scenario_profit: float,
) -> dict:
    """
    Calculate absolute and percentage profit change.
    """

    change = scenario_profit - baseline_profit

    return {
        "absolute_change": change,
        "percentage_change": _safe_percentage_change(
            baseline_profit,
            scenario_profit,
        ),
        "improved": change > 0,
        "declined": change < 0,
        "unchanged": change == 0,
    }


# ---------------------------------------------------------------------------
# Result Serialization
# ---------------------------------------------------------------------------

def scenario_to_dict(
    result: ScenarioResult,
) -> dict:
    """
    Convert ScenarioResult into a JSON-friendly dictionary.

    Useful for FastAPI responses and agent tools.
    """

    return {
        "baseline": {
            "revenue": round(result.baseline_revenue, 2),
            "expenses": round(result.baseline_expenses, 2),
            "profit": round(result.baseline_profit, 2),
            "margin_pct": round(result.baseline_margin, 2),
        },
        "scenario": {
            "revenue": round(result.scenario_revenue, 2),
            "expenses": round(result.scenario_expenses, 2),
            "profit": round(result.scenario_profit, 2),
            "margin_pct": round(result.scenario_margin, 2),
        },
        "change": {
            "revenue": round(result.revenue_change, 2),
            "expenses": round(result.expense_change, 2),
            "profit": round(result.profit_change, 2),
            "margin_pct": round(result.margin_change, 2),
            "revenue_pct": round(result.revenue_change_pct, 2),
            "expense_pct": round(result.expense_change_pct, 2),
            "profit_pct": round(result.profit_change_pct, 2),
        },
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    result = calculate_scenario(
        revenue=1_000_000,
        expenses=800_000,
        revenue_change_pct=10,
        expense_change_pct=-5,
    )

    print("FinCo AI What-If Scenario")
    print("=" * 40)

    data = scenario_to_dict(result)

    print(f"Baseline Revenue : ${data['baseline']['revenue']:,.2f}")
    print(f"Baseline Expenses: ${data['baseline']['expenses']:,.2f}")
    print(f"Baseline Profit  : ${data['baseline']['profit']:,.2f}")
    print(f"Baseline Margin  : {data['baseline']['margin_pct']:.2f}%")

    print()

    print(f"Scenario Revenue : ${data['scenario']['revenue']:,.2f}")
    print(f"Scenario Expenses: ${data['scenario']['expenses']:,.2f}")
    print(f"Scenario Profit  : ${data['scenario']['profit']:,.2f}")
    print(f"Scenario Margin  : {data['scenario']['margin_pct']:.2f}%")

    print()

    print(f"Profit Change    : ${data['change']['profit']:,.2f}")
    print(f"Margin Change    : {data['change']['margin_pct']:.2f}%")