"""
FinCo AI - Financial Scenario Model

File:
    backend/app/what_if/financial_model.py

Purpose:
    Business-level financial modeling for the What-If engine.

Architecture:
    What-If Agent
          ↓
    What-If Tools
          ↓
    FinancialModel
          ↓
    calculator.py
          ↓
    ScenarioResult

This module is deterministic and contains no LLM logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


# ============================================================================
# Exceptions
# ============================================================================

class FinancialModelError(ValueError):
    """Raised when financial model inputs are invalid."""


# ============================================================================
# Data Models
# ============================================================================

@dataclass(frozen=True)
class FinancialInputs:
    """
    Baseline financial inputs.

    All monetary values use the same currency.
    """

    revenue: float
    cost_of_goods_sold: float = 0.0
    operating_expenses: float = 0.0

    taxes: float = 0.0
    interest_expense: float = 0.0

    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None

    cash_balance: float = 0.0
    debt: float = 0.0

    def validate(self) -> None:
        """Validate baseline financial inputs."""

        values = {
            "revenue": self.revenue,
            "cost_of_goods_sold": self.cost_of_goods_sold,
            "operating_expenses": self.operating_expenses,
            "taxes": self.taxes,
            "interest_expense": self.interest_expense,
            "cash_balance": self.cash_balance,
            "debt": self.debt,
        }

        for name, value in values.items():
            if value < 0:
                raise FinancialModelError(
                    f"{name} cannot be negative."
                )

        if self.revenue == 0:
            # Zero revenue is allowed for analysis,
            # but margins will safely return zero.
            return


@dataclass(frozen=True)
class FinancialMetrics:
    """Calculated financial metrics."""

    revenue: float
    cost_of_goods_sold: float
    gross_profit: float
    gross_margin_pct: float

    operating_expenses: float
    operating_profit: float
    operating_margin_pct: float

    interest_expense: float
    earnings_before_tax: float

    taxes: float
    net_profit: float
    net_margin_pct: float

    operating_cash_flow: float
    investing_cash_flow: float
    financing_cash_flow: float

    net_cash_flow: float
    ending_cash_balance: float

    debt: float
    debt_to_revenue_pct: float


@dataclass(frozen=True)
class ScenarioAssumptions:
    """
    Assumptions applied to the baseline model.

    Example:
        revenue_change_pct = -10
        cogs_change_pct = 5
        operating_expense_change_pct = 3
    """

    revenue_change_pct: float = 0.0
    cogs_change_pct: float = 0.0
    operating_expense_change_pct: float = 0.0

    tax_change_pct: float = 0.0
    interest_change_pct: float = 0.0

    investing_cash_flow_change_pct: float = 0.0
    financing_cash_flow_change_pct: float = 0.0

    debt_change_pct: float = 0.0


@dataclass(frozen=True)
class ScenarioComparison:
    """Baseline vs scenario comparison."""

    baseline: FinancialMetrics
    scenario: FinancialMetrics

    revenue_change: float
    revenue_change_pct: float

    gross_profit_change: float
    operating_profit_change: float
    net_profit_change: float

    gross_margin_change_pct: float
    operating_margin_change_pct: float
    net_margin_change_pct: float

    cash_change: float
    debt_change: float

    profit_improved: bool
    cash_improved: bool
    debt_reduced: bool
    scenario_is_loss: bool


# ============================================================================
# Validation Helpers
# ============================================================================

def _validate_percentage(
    value: float,
    field_name: str,
    minimum: float = -100.0,
) -> None:
    """Validate percentage input."""

    if not isinstance(value, (int, float)):
        raise FinancialModelError(
            f"{field_name} must be numeric."
        )

    if value < minimum:
        raise FinancialModelError(
            f"{field_name} cannot be below {minimum}%."
        )


def _apply_change(
    value: float,
    percentage: float,
) -> float:
    """Apply a percentage change."""

    return value * (1.0 + percentage / 100.0)


def _percentage_change(
    baseline: float,
    current: float,
) -> float:
    """Safely calculate percentage change."""

    if baseline == 0:
        return 0.0

    return ((current - baseline) / abs(baseline)) * 100.0


def _margin(
    numerator: float,
    denominator: float,
) -> float:
    """Safely calculate a percentage margin."""

    if denominator == 0:
        return 0.0

    return (numerator / denominator) * 100.0


# ============================================================================
# Financial Model
# ============================================================================

class FinancialModel:
    """
    Deterministic financial scenario model.

    Responsibilities:
        1. Calculate baseline metrics.
        2. Apply scenario assumptions.
        3. Calculate scenario metrics.
        4. Compare baseline and scenario.
        5. Expose risk-oriented financial indicators.
    """

    def __init__(self, inputs: FinancialInputs) -> None:
        inputs.validate()
        self.inputs = inputs

    # ------------------------------------------------------------------------
    # Baseline Calculations
    # ------------------------------------------------------------------------

    def calculate_metrics(
        self,
        inputs: Optional[FinancialInputs] = None,
    ) -> FinancialMetrics:
        """
        Calculate a complete financial statement model.
        """

        data = inputs or self.inputs
        data.validate()

        revenue = data.revenue
        cogs = data.cost_of_goods_sold
        operating_expenses = data.operating_expenses

        gross_profit = revenue - cogs

        operating_profit = (
            gross_profit - operating_expenses
        )

        earnings_before_tax = (
            operating_profit - data.interest_expense
        )

        net_profit = (
            earnings_before_tax - data.taxes
        )

        operating_cash_flow = (
            data.operating_cash_flow
            if data.operating_cash_flow is not None
            else operating_profit
        )

        investing_cash_flow = (
            data.investing_cash_flow
            if data.investing_cash_flow is not None
            else 0.0
        )

        financing_cash_flow = (
            data.financing_cash_flow
            if data.financing_cash_flow is not None
            else 0.0
        )

        net_cash_flow = (
            operating_cash_flow
            + investing_cash_flow
            + financing_cash_flow
        )

        ending_cash_balance = (
            data.cash_balance + net_cash_flow
        )

        return FinancialMetrics(
            revenue=revenue,

            cost_of_goods_sold=cogs,

            gross_profit=gross_profit,

            gross_margin_pct=_margin(
                gross_profit,
                revenue,
            ),

            operating_expenses=operating_expenses,

            operating_profit=operating_profit,

            operating_margin_pct=_margin(
                operating_profit,
                revenue,
            ),

            interest_expense=data.interest_expense,

            earnings_before_tax=earnings_before_tax,

            taxes=data.taxes,

            net_profit=net_profit,

            net_margin_pct=_margin(
                net_profit,
                revenue,
            ),

            operating_cash_flow=operating_cash_flow,

            investing_cash_flow=investing_cash_flow,

            financing_cash_flow=financing_cash_flow,

            net_cash_flow=net_cash_flow,

            ending_cash_balance=ending_cash_balance,

            debt=data.debt,

            debt_to_revenue_pct=_margin(
                data.debt,
                revenue,
            ),
        )

    def baseline(self) -> FinancialMetrics:
        """Return baseline financial metrics."""

        return self.calculate_metrics()

    # ------------------------------------------------------------------------
    # Scenario Application
    # ------------------------------------------------------------------------

    def apply_scenario(
        self,
        assumptions: ScenarioAssumptions,
    ) -> FinancialInputs:
        """
        Apply scenario assumptions to baseline inputs.

        Returns a new FinancialInputs object.
        """

        self._validate_assumptions(assumptions)

        revenue = _apply_change(
            self.inputs.revenue,
            assumptions.revenue_change_pct,
        )

        cogs = _apply_change(
            self.inputs.cost_of_goods_sold,
            assumptions.cogs_change_pct,
        )

        operating_expenses = _apply_change(
            self.inputs.operating_expenses,
            assumptions.operating_expense_change_pct,
        )

        taxes = _apply_change(
            self.inputs.taxes,
            assumptions.tax_change_pct,
        )

        interest = _apply_change(
            self.inputs.interest_expense,
            assumptions.interest_change_pct,
        )

        operating_cash_flow = self.inputs.operating_cash_flow

        if operating_cash_flow is not None:
            operating_cash_flow = _apply_change(
                operating_cash_flow,
                assumptions.revenue_change_pct,
            )

        investing_cash_flow = self.inputs.investing_cash_flow

        if investing_cash_flow is not None:
            investing_cash_flow = _apply_change(
                investing_cash_flow,
                assumptions.investing_cash_flow_change_pct,
            )

        financing_cash_flow = self.inputs.financing_cash_flow

        if financing_cash_flow is not None:
            financing_cash_flow = _apply_change(
                financing_cash_flow,
                assumptions.financing_cash_flow_change_pct,
            )

        debt = _apply_change(
            self.inputs.debt,
            assumptions.debt_change_pct,
        )

        return FinancialInputs(
            revenue=max(revenue, 0.0),

            cost_of_goods_sold=max(cogs, 0.0),

            operating_expenses=max(
                operating_expenses,
                0.0,
            ),

            taxes=max(taxes, 0.0),

            interest_expense=max(
                interest,
                0.0,
            ),

            operating_cash_flow=operating_cash_flow,

            investing_cash_flow=investing_cash_flow,

            financing_cash_flow=financing_cash_flow,

            cash_balance=self.inputs.cash_balance,

            debt=max(debt, 0.0),
        )

    # ------------------------------------------------------------------------
    # Scenario Comparison
    # ------------------------------------------------------------------------

    def compare(
        self,
        assumptions: ScenarioAssumptions,
    ) -> ScenarioComparison:
        """
        Calculate baseline and scenario metrics and compare them.
        """

        baseline = self.baseline()

        scenario_inputs = self.apply_scenario(
            assumptions
        )

        scenario = self.calculate_metrics(
            scenario_inputs
        )

        return ScenarioComparison(
            baseline=baseline,
            scenario=scenario,

            revenue_change=(
                scenario.revenue
                - baseline.revenue
            ),

            revenue_change_pct=_percentage_change(
                baseline.revenue,
                scenario.revenue,
            ),

            gross_profit_change=(
                scenario.gross_profit
                - baseline.gross_profit
            ),

            operating_profit_change=(
                scenario.operating_profit
                - baseline.operating_profit
            ),

            net_profit_change=(
                scenario.net_profit
                - baseline.net_profit
            ),

            gross_margin_change_pct=(
                scenario.gross_margin_pct
                - baseline.gross_margin_pct
            ),

            operating_margin_change_pct=(
                scenario.operating_margin_pct
                - baseline.operating_margin_pct
            ),

            net_margin_change_pct=(
                scenario.net_margin_pct
                - baseline.net_margin_pct
            ),

            cash_change=(
                scenario.ending_cash_balance
                - baseline.ending_cash_balance
            ),

            debt_change=(
                scenario.debt
                - baseline.debt
            ),

            profit_improved=(
                scenario.net_profit
                > baseline.net_profit
            ),

            cash_improved=(
                scenario.ending_cash_balance
                > baseline.ending_cash_balance
            ),

            debt_reduced=(
                scenario.debt
                < baseline.debt
            ),

            scenario_is_loss=(
                scenario.net_profit < 0
            ),
        )

    # ------------------------------------------------------------------------
    # Scenario Helpers
    # ------------------------------------------------------------------------

    def revenue_decline_scenario(
        self,
        decline_pct: float,
    ) -> ScenarioComparison:
        """
        Evaluate a revenue decline scenario.

        Example:
            15 means revenue falls by 15%.
        """

        return self.compare(
            ScenarioAssumptions(
                revenue_change_pct=-abs(decline_pct)
            )
        )

    def revenue_growth_scenario(
        self,
        growth_pct: float,
    ) -> ScenarioComparison:
        """Evaluate a revenue growth scenario."""

        return self.compare(
            ScenarioAssumptions(
                revenue_change_pct=abs(growth_pct)
            )
        )

    def expense_reduction_scenario(
        self,
        reduction_pct: float,
    ) -> ScenarioComparison:
        """
        Evaluate an expense reduction scenario.
        """

        return self.compare(
            ScenarioAssumptions(
                operating_expense_change_pct=-abs(
                    reduction_pct
                )
            )
        )

    def expense_increase_scenario(
        self,
        increase_pct: float,
    ) -> ScenarioComparison:
        """Evaluate an expense increase scenario."""

        return self.compare(
            ScenarioAssumptions(
                operating_expense_change_pct=abs(
                    increase_pct
                )
            )
        )

    def combined_downside_scenario(
        self,
        revenue_decline_pct: float,
        expense_increase_pct: float,
    ) -> ScenarioComparison:
        """
        Evaluate simultaneous revenue decline and
        expense increase.
        """

        return self.compare(
            ScenarioAssumptions(
                revenue_change_pct=-abs(
                    revenue_decline_pct
                ),

                operating_expense_change_pct=abs(
                    expense_increase_pct
                ),
            )
        )

    # ------------------------------------------------------------------------
    # Risk Analysis
    # ------------------------------------------------------------------------

    def risk_indicators(
        self,
        metrics: Optional[FinancialMetrics] = None,
    ) -> dict:
        """
        Generate deterministic risk indicators.

        These are indicators, not ML probabilities.
        """

        metrics = metrics or self.baseline()

        indicators = {
            "loss": metrics.net_profit < 0,

            "negative_operating_profit":
                metrics.operating_profit < 0,

            "negative_cash_flow":
                metrics.net_cash_flow < 0,

            "negative_cash_balance":
                metrics.ending_cash_balance < 0,

            "high_debt_to_revenue":
                metrics.debt_to_revenue_pct > 100,

            "low_gross_margin":
                metrics.gross_margin_pct < 20,

            "low_operating_margin":
                metrics.operating_margin_pct < 10,

            "low_net_margin":
                metrics.net_margin_pct < 5,
        }

        risk_count = sum(
            1 for value in indicators.values()
            if value
        )

        if risk_count >= 5:
            risk_level = "CRITICAL"

        elif risk_count >= 3:
            risk_level = "HIGH"

        elif risk_count >= 1:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "risk_indicator_count": risk_count,
            "indicators": indicators,
        }

    # ------------------------------------------------------------------------
    # Target Analysis
    # ------------------------------------------------------------------------

    def required_revenue_for_margin(
        self,
        target_margin_pct: float,
        projected_expenses: Optional[float] = None,
    ) -> float:
        """
        Calculate revenue required to achieve a target
        operating margin.

        Formula:

            Operating Profit = Revenue - Expenses

            Target Margin =
                (Revenue - Expenses) / Revenue

        Therefore:

            Revenue =
                Expenses / (1 - Target Margin)
        """

        if not 0 <= target_margin_pct < 100:
            raise FinancialModelError(
                "Target margin must be between 0 and 100."
            )

        expenses = (
            projected_expenses
            if projected_expenses is not None
            else (
                self.inputs.cost_of_goods_sold
                + self.inputs.operating_expenses
            )
        )

        if expenses < 0:
            raise FinancialModelError(
                "Projected expenses cannot be negative."
            )

        margin_decimal = target_margin_pct / 100.0

        return expenses / (1 - margin_decimal)

    def required_revenue_growth(
        self,
        target_revenue: float,
    ) -> float:
        """Calculate growth required to reach target revenue."""

        if target_revenue < 0:
            raise FinancialModelError(
                "Target revenue cannot be negative."
            )

        return _percentage_change(
            self.inputs.revenue,
            target_revenue,
        )

    # ------------------------------------------------------------------------
    # Internal Validation
    # ------------------------------------------------------------------------

    @staticmethod
    def _validate_assumptions(
        assumptions: ScenarioAssumptions,
    ) -> None:
        """Validate all scenario assumptions."""

        fields = {
            "revenue_change_pct":
                assumptions.revenue_change_pct,

            "cogs_change_pct":
                assumptions.cogs_change_pct,

            "operating_expense_change_pct":
                assumptions.operating_expense_change_pct,

            "tax_change_pct":
                assumptions.tax_change_pct,

            "interest_change_pct":
                assumptions.interest_change_pct,

            "investing_cash_flow_change_pct":
                assumptions.investing_cash_flow_change_pct,

            "financing_cash_flow_change_pct":
                assumptions.financing_cash_flow_change_pct,

            "debt_change_pct":
                assumptions.debt_change_pct,
        }

        for name, value in fields.items():
            _validate_percentage(
                value,
                name,
            )


# ============================================================================
# Serialization Helpers
# ============================================================================

def metrics_to_dict(
    metrics: FinancialMetrics,
) -> dict:
    """Convert financial metrics to JSON-compatible dictionary."""

    return {
        key: round(value, 2)
        if isinstance(value, float)
        else value
        for key, value in asdict(metrics).items()
    }


def comparison_to_dict(
    comparison: ScenarioComparison,
) -> dict:
    """
    Convert ScenarioComparison into a structure suitable
    for FastAPI / frontend / agent responses.
    """

    return {
        "baseline": metrics_to_dict(
            comparison.baseline
        ),

        "scenario": metrics_to_dict(
            comparison.scenario
        ),

        "changes": {
            "revenue": round(
                comparison.revenue_change,
                2,
            ),

            "revenue_pct": round(
                comparison.revenue_change_pct,
                2,
            ),

            "gross_profit": round(
                comparison.gross_profit_change,
                2,
            ),

            "operating_profit": round(
                comparison.operating_profit_change,
                2,
            ),

            "net_profit": round(
                comparison.net_profit_change,
                2,
            ),

            "gross_margin_pct": round(
                comparison.gross_margin_change_pct,
                2,
            ),

            "operating_margin_pct": round(
                comparison.operating_margin_change_pct,
                2,
            ),

            "net_margin_pct": round(
                comparison.net_margin_change_pct,
                2,
            ),

            "cash": round(
                comparison.cash_change,
                2,
            ),

            "debt": round(
                comparison.debt_change,
                2,
            ),
        },

        "interpretation": {
            "profit_improved":
                comparison.profit_improved,

            "cash_improved":
                comparison.cash_improved,

            "debt_reduced":
                comparison.debt_reduced,

            "scenario_is_loss":
                comparison.scenario_is_loss,
        },
    }


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":

    model = FinancialModel(
        FinancialInputs(
            revenue=1_000_000,
            cost_of_goods_sold=550_000,
            operating_expenses=250_000,
            taxes=40_000,
            interest_expense=20_000,
            cash_balance=300_000,
            debt=400_000,
            operating_cash_flow=180_000,
            investing_cash_flow=-50_000,
            financing_cash_flow=-20_000,
        )
    )

    print("=" * 60)
    print("FinCo AI - Financial Scenario Model")
    print("=" * 60)

    baseline = model.baseline()

    print("\nBASELINE")
    print("-" * 60)
    print(f"Revenue:          ${baseline.revenue:,.2f}")
    print(f"Gross Profit:     ${baseline.gross_profit:,.2f}")
    print(f"Operating Profit: ${baseline.operating_profit:,.2f}")
    print(f"Net Profit:       ${baseline.net_profit:,.2f}")
    print(f"Net Margin:       {baseline.net_margin_pct:.2f}%")
    print(f"Ending Cash:      ${baseline.ending_cash_balance:,.2f}")

    print("\nDOWNSIDE SCENARIO")
    print("-" * 60)
    print("Revenue: -15%")
    print("Operating Expenses: +8%")

    result = model.combined_downside_scenario(
        revenue_decline_pct=15,
        expense_increase_pct=8,
    )

    print(
        f"Scenario Revenue: "
        f"${result.scenario.revenue:,.2f}"
    )

    print(
        f"Scenario Net Profit: "
        f"${result.scenario.net_profit:,.2f}"
    )

    print(
        f"Profit Change: "
        f"${result.net_profit_change:,.2f}"
    )

    print(
        f"Net Margin: "
        f"{result.scenario.net_margin_pct:.2f}%"
    )

    print(
        f"Scenario Loss: "
        f"{result.scenario_is_loss}"
    )

    print("\nRISK INDICATORS")
    print("-" * 60)

    risk = model.risk_indicators(
        result.scenario
    )

    print(f"Risk Level: {risk['risk_level']}")
    print(
        f"Risk Indicators: "
        f"{risk['risk_indicator_count']}"
    )

    for name, triggered in risk["indicators"].items():
        if triggered:
            print(f"  ⚠ {name}")