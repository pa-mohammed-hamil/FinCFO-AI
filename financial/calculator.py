"""
FinCo AI - Financial Calculator
================================

Location:
    backend/app/financial/calculator.py

Purpose:
    Centralized, deterministic financial calculations.

Used by:
    - budget_variance.py
    - revenue.py
    - expenses.py
    - pnl.py
    - ratios.py
    - kpis.py
    - profitability.py
    - margin_analysis.py
    - cost_analysis.py
    - root_cause.py
    - health_score.py
    - what_if/
    - agents/tools/calculator_tools.py

Design:
    - No database access
    - No API logic
    - No LLM calls
    - Decimal-based arithmetic
    - Explicit validation
    - JSON-friendly results
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Optional


# ============================================================================
# ENUMS
# ============================================================================


class CalculationStatus(str, Enum):
    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    DIVISION_BY_ZERO = "division_by_zero"


class GrowthDirection(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"


# ============================================================================
# RESULT MODELS
# ============================================================================


@dataclass(frozen=True)
class CalculationResult:
    """Generic financial calculation result."""

    operation: str
    value: Decimal
    status: CalculationStatus = CalculationStatus.SUCCESS
    message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "value": float(self.value),
            "status": self.status.value,
            "message": self.message,
        }


@dataclass(frozen=True)
class PercentageResult:
    """Percentage-based calculation result."""

    operation: str
    numerator: Decimal
    denominator: Decimal
    percentage: Decimal
    status: CalculationStatus
    message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "numerator": float(self.numerator),
            "denominator": float(self.denominator),
            "percentage": float(self.percentage),
            "status": self.status.value,
            "message": self.message,
        }


@dataclass(frozen=True)
class GrowthResult:
    """Growth/change calculation."""

    previous: Decimal
    current: Decimal
    change: Decimal
    change_percent: Decimal
    direction: GrowthDirection
    status: CalculationStatus

    def to_dict(self) -> dict:
        return {
            "previous": float(self.previous),
            "current": float(self.current),
            "change": float(self.change),
            "change_percent": float(self.change_percent),
            "direction": self.direction.value,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class ProfitabilityResult:
    """Profitability calculation."""

    revenue: Decimal
    cost: Decimal
    gross_profit: Decimal
    margin_percent: Decimal
    status: CalculationStatus

    def to_dict(self) -> dict:
        return {
            "revenue": float(self.revenue),
            "cost": float(self.cost),
            "gross_profit": float(self.gross_profit),
            "margin_percent": float(self.margin_percent),
            "status": self.status.value,
        }


# ============================================================================
# FINANCIAL CALCULATOR
# ============================================================================


class FinancialCalculator:
    """
    Central deterministic financial calculator.

    All financial arithmetic should pass through this service where
    practical, so calculations remain consistent across FinCo AI.
    """

    QUANTIZE_2 = Decimal("0.01")
    QUANTIZE_4 = Decimal("0.0001")

    # ========================================================================
    # BASIC ARITHMETIC
    # ========================================================================

    @classmethod
    def add(
        cls,
        *values: Decimal | float | int | str,
    ) -> Decimal:
        """Add multiple financial values."""

        numbers = [
            cls.to_decimal(value)
            for value in values
        ]

        return sum(
            numbers,
            Decimal("0"),
        )

    # ------------------------------------------------------------------------

    @classmethod
    def subtract(
        cls,
        minuend: Decimal | float | int | str,
        subtrahend: Decimal | float | int | str,
    ) -> Decimal:
        """Subtract one financial value from another."""

        return (
            cls.to_decimal(minuend)
            - cls.to_decimal(subtrahend)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def multiply(
        cls,
        value: Decimal | float | int | str,
        multiplier: Decimal | float | int | str,
    ) -> Decimal:
        """Multiply financial values."""

        return (
            cls.to_decimal(value)
            * cls.to_decimal(multiplier)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def divide(
        cls,
        numerator: Decimal | float | int | str,
        denominator: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Divide two values safely.

        Raises:
            ZeroDivisionError
        """

        numerator_decimal = cls.to_decimal(
            numerator
        )

        denominator_decimal = cls.to_decimal(
            denominator
        )

        if denominator_decimal == 0:
            raise ZeroDivisionError(
                "Financial calculation cannot divide by zero."
            )

        quantizer = Decimal(
            "1"
        ).scaleb(-precision)

        return (
            numerator_decimal
            / denominator_decimal
        ).quantize(quantizer)

    # ========================================================================
    # PERCENTAGES
    # ========================================================================

    @classmethod
    def percentage(
        cls,
        numerator: Decimal | float | int | str,
        denominator: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate:

            numerator / denominator * 100
        """

        numerator_decimal = cls.to_decimal(
            numerator
        )

        denominator_decimal = cls.to_decimal(
            denominator
        )

        if denominator_decimal == 0:
            raise ZeroDivisionError(
                "Cannot calculate percentage with "
                "zero denominator."
            )

        quantizer = Decimal(
            "1"
        ).scaleb(-precision)

        return (
            numerator_decimal
            / denominator_decimal
            * Decimal("100")
        ).quantize(quantizer)

    # ------------------------------------------------------------------------

    @classmethod
    def safe_percentage(
        cls,
        numerator: Decimal | float | int | str,
        denominator: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> PercentageResult:
        """
        Percentage calculation that returns a structured result
        instead of raising on division by zero.
        """

        numerator_decimal = cls.to_decimal(
            numerator
        )

        denominator_decimal = cls.to_decimal(
            denominator
        )

        if denominator_decimal == 0:

            return PercentageResult(
                operation="percentage",
                numerator=numerator_decimal,
                denominator=denominator_decimal,
                percentage=Decimal("0"),
                status=CalculationStatus.DIVISION_BY_ZERO,
                message=(
                    "Percentage cannot be calculated "
                    "because denominator is zero."
                ),
            )

        percentage = cls.percentage(
            numerator_decimal,
            denominator_decimal,
            precision=precision,
        )

        return PercentageResult(
            operation="percentage",
            numerator=numerator_decimal,
            denominator=denominator_decimal,
            percentage=percentage,
            status=CalculationStatus.SUCCESS,
        )

    # ========================================================================
    # GROWTH
    # ========================================================================

    @classmethod
    def growth(
        cls,
        previous: Decimal | float | int | str,
        current: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> GrowthResult:
        """
        Calculate period-over-period growth.

        Formula:

            change = current - previous

            growth % =
                (current - previous) / abs(previous) * 100
        """

        previous_decimal = cls.to_decimal(
            previous
        )

        current_decimal = cls.to_decimal(
            current
        )

        change = (
            current_decimal
            - previous_decimal
        )

        if change > 0:
            direction = GrowthDirection.INCREASE

        elif change < 0:
            direction = GrowthDirection.DECREASE

        else:
            direction = GrowthDirection.NO_CHANGE

        if previous_decimal == 0:

            return GrowthResult(
                previous=previous_decimal,
                current=current_decimal,
                change=change,
                change_percent=Decimal("0"),
                direction=direction,
                status=(
                    CalculationStatus.DIVISION_BY_ZERO
                ),
            )

        change_percent = cls.percentage(
            change,
            abs(previous_decimal),
            precision=precision,
        )

        return GrowthResult(
            previous=previous_decimal,
            current=current_decimal,
            change=change,
            change_percent=change_percent,
            direction=direction,
            status=CalculationStatus.SUCCESS,
        )

    # ========================================================================
    # VARIANCE
    # ========================================================================

    @classmethod
    def variance(
        cls,
        expected: Decimal | float | int | str,
        actual: Decimal | float | int | str,
    ) -> Decimal:
        """
        Calculate absolute variance.

            actual - expected
        """

        return (
            cls.to_decimal(actual)
            - cls.to_decimal(expected)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def variance_percent(
        cls,
        expected: Decimal | float | int | str,
        actual: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate variance percentage.

            (actual - expected)
            ------------------- * 100
                 abs(expected)
        """

        expected_decimal = cls.to_decimal(
            expected
        )

        actual_decimal = cls.to_decimal(
            actual
        )

        if expected_decimal == 0:

            if actual_decimal == 0:
                return Decimal("0")

            return Decimal("100")

        return cls.percentage(
            actual_decimal - expected_decimal,
            abs(expected_decimal),
            precision=precision,
        )

    # ========================================================================
    # REVENUE
    # ========================================================================

    @classmethod
    def net_revenue(
        cls,
        gross_revenue: Decimal | float | int | str,
        returns: Decimal | float | int | str = 0,
        discounts: Decimal | float | int | str = 0,
        allowances: Decimal | float | int | str = 0,
    ) -> Decimal:
        """
        Calculate net revenue.

            Net Revenue =
                Gross Revenue
                - Returns
                - Discounts
                - Allowances
        """

        return (
            cls.to_decimal(gross_revenue)
            - cls.to_decimal(returns)
            - cls.to_decimal(discounts)
            - cls.to_decimal(allowances)
        )

    # ========================================================================
    # COST OF GOODS SOLD
    # ========================================================================

    @classmethod
    def cost_of_goods_sold(
        cls,
        opening_inventory: Decimal | float | int | str,
        purchases: Decimal | float | int | str,
        closing_inventory: Decimal | float | int | str,
    ) -> Decimal:
        """
        Calculate COGS.

            COGS =
                Opening Inventory
                + Purchases
                - Closing Inventory
        """

        return (
            cls.to_decimal(opening_inventory)
            + cls.to_decimal(purchases)
            - cls.to_decimal(closing_inventory)
        )

    # ========================================================================
    # GROSS PROFIT
    # ========================================================================

    @classmethod
    def gross_profit(
        cls,
        revenue: Decimal | float | int | str,
        cogs: Decimal | float | int | str,
    ) -> Decimal:
        """Calculate gross profit."""

        return (
            cls.to_decimal(revenue)
            - cls.to_decimal(cogs)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def gross_margin(
        cls,
        revenue: Decimal | float | int | str,
        cogs: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate gross profit margin.

            Gross Margin =
                Gross Profit / Revenue * 100
        """

        revenue_decimal = cls.to_decimal(
            revenue
        )

        gross_profit = cls.gross_profit(
            revenue,
            cogs,
        )

        if revenue_decimal == 0:
            return Decimal("0")

        return cls.percentage(
            gross_profit,
            revenue_decimal,
            precision=precision,
        )

    # ========================================================================
    # OPERATING PROFIT
    # ========================================================================

    @classmethod
    def operating_profit(
        cls,
        gross_profit: Decimal | float | int | str,
        operating_expenses: Decimal | float | int | str,
    ) -> Decimal:
        """Calculate operating profit."""

        return (
            cls.to_decimal(gross_profit)
            - cls.to_decimal(operating_expenses)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def operating_margin(
        cls,
        revenue: Decimal | float | int | str,
        operating_expenses: Decimal | float | int | str,
        cogs: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate operating margin.

            Operating Profit / Revenue * 100
        """

        revenue_decimal = cls.to_decimal(
            revenue
        )

        if revenue_decimal == 0:
            return Decimal("0")

        gross_profit = cls.gross_profit(
            revenue,
            cogs,
        )

        operating_profit = cls.operating_profit(
            gross_profit,
            operating_expenses,
        )

        return cls.percentage(
            operating_profit,
            revenue_decimal,
            precision=precision,
        )

    # ========================================================================
    # NET PROFIT
    # ========================================================================

    @classmethod
    def net_profit(
        cls,
        revenue: Decimal | float | int | str,
        cogs: Decimal | float | int | str,
        operating_expenses: Decimal | float | int | str,
        interest_expense: Decimal | float | int | str = 0,
        taxes: Decimal | float | int | str = 0,
        other_expenses: Decimal | float | int | str = 0,
        other_income: Decimal | float | int | str = 0,
    ) -> Decimal:
        """
        Calculate net profit.

            Gross Profit
            - Operating Expenses
            - Interest
            - Taxes
            - Other Expenses
            + Other Income
        """

        gross_profit = cls.gross_profit(
            revenue,
            cogs,
        )

        operating_profit = cls.operating_profit(
            gross_profit,
            operating_expenses,
        )

        return (
            operating_profit
            - cls.to_decimal(interest_expense)
            - cls.to_decimal(taxes)
            - cls.to_decimal(other_expenses)
            + cls.to_decimal(other_income)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def net_margin(
        cls,
        revenue: Decimal | float | int | str,
        net_profit: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """Calculate net profit margin."""

        revenue_decimal = cls.to_decimal(
            revenue
        )

        if revenue_decimal == 0:
            return Decimal("0")

        return cls.percentage(
            net_profit,
            revenue_decimal,
            precision=precision,
        )

    # ========================================================================
    # BREAK-EVEN ANALYSIS
    # ========================================================================

    @classmethod
    def contribution_margin(
        cls,
        selling_price: Decimal | float | int | str,
        variable_cost: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate contribution margin per unit.

            Selling Price - Variable Cost
        """

        return (
            cls.to_decimal(selling_price)
            - cls.to_decimal(variable_cost)
        ).quantize(
            Decimal("1").scaleb(-precision)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def contribution_margin_ratio(
        cls,
        selling_price: Decimal | float | int | str,
        variable_cost: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate contribution margin ratio.

            Contribution Margin / Selling Price * 100
        """

        price = cls.to_decimal(
            selling_price
        )

        contribution = cls.contribution_margin(
            selling_price,
            variable_cost,
        )

        if price == 0:
            return Decimal("0")

        return cls.percentage(
            contribution,
            price,
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def break_even_units(
        cls,
        fixed_costs: Decimal | float | int | str,
        selling_price: Decimal | float | int | str,
        variable_cost: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate break-even units.

            Fixed Costs
            --------------------------
            Selling Price - Variable Cost
        """

        fixed = cls.to_decimal(
            fixed_costs
        )

        contribution = cls.contribution_margin(
            selling_price,
            variable_cost,
        )

        if contribution == 0:
            raise ZeroDivisionError(
                "Break-even units cannot be calculated "
                "when contribution margin is zero."
            )

        return (
            fixed / contribution
        ).quantize(
            Decimal("1").scaleb(-precision)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def break_even_revenue(
        cls,
        fixed_costs: Decimal | float | int | str,
        contribution_margin_ratio: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate break-even revenue.

        contribution_margin_ratio can be supplied as:
            40 = 40%

        """

        fixed = cls.to_decimal(
            fixed_costs
        )

        ratio = (
            cls.to_decimal(
                contribution_margin_ratio
            )
            / Decimal("100")
        )

        if ratio == 0:
            raise ZeroDivisionError(
                "Break-even revenue cannot be calculated "
                "with zero contribution margin ratio."
            )

        return (
            fixed / ratio
        ).quantize(
            Decimal("1").scaleb(-precision)
        )

    # ========================================================================
    # CASH FLOW
    # ========================================================================

    @classmethod
    def operating_cash_flow(
        cls,
        net_income: Decimal | float | int | str,
        depreciation: Decimal | float | int | str = 0,
        working_capital_change: Decimal | float | int | str = 0,
    ) -> Decimal:
        """
        Simplified operating cash flow.

            Net Income
            + Depreciation
            - Increase in Working Capital
        """

        return (
            cls.to_decimal(net_income)
            + cls.to_decimal(depreciation)
            - cls.to_decimal(working_capital_change)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def free_cash_flow(
        cls,
        operating_cash_flow: Decimal | float | int | str,
        capital_expenditure: Decimal | float | int | str,
    ) -> Decimal:
        """
        Calculate free cash flow.

            Operating Cash Flow - CapEx
        """

        return (
            cls.to_decimal(
                operating_cash_flow
            )
            - cls.to_decimal(
                capital_expenditure
            )
        )

    # ========================================================================
    # LIQUIDITY
    # ========================================================================

    @classmethod
    def current_ratio(
        cls,
        current_assets: Decimal | float | int | str,
        current_liabilities: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """Calculate current ratio."""

        return cls.divide(
            current_assets,
            current_liabilities,
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def quick_ratio(
        cls,
        cash: Decimal | float | int | str,
        marketable_securities: Decimal | float | int | str,
        accounts_receivable: Decimal | float | int | str,
        current_liabilities: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate quick ratio.

            Cash
            + Marketable Securities
            + Accounts Receivable
            --------------------------------
            Current Liabilities
        """

        quick_assets = (
            cls.to_decimal(cash)
            + cls.to_decimal(
                marketable_securities
            )
            + cls.to_decimal(
                accounts_receivable
            )
        )

        return cls.divide(
            quick_assets,
            current_liabilities,
            precision=precision,
        )

    # ========================================================================
    # LEVERAGE
    # ========================================================================

    @classmethod
    def debt_to_equity(
        cls,
        total_debt: Decimal | float | int | str,
        shareholders_equity: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """Calculate debt-to-equity ratio."""

        return cls.divide(
            total_debt,
            shareholders_equity,
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def debt_ratio(
        cls,
        total_debt: Decimal | float | int | str,
        total_assets: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate debt ratio as percentage.

            Total Debt / Total Assets * 100
        """

        return cls.percentage(
            total_debt,
            total_assets,
            precision=precision,
        )

    # ========================================================================
    # RETURN METRICS
    # ========================================================================

    @classmethod
    def return_on_assets(
        cls,
        net_income: Decimal | float | int | str,
        average_assets: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        ROA.

            Net Income / Average Assets * 100
        """

        return cls.percentage(
            net_income,
            average_assets,
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def return_on_equity(
        cls,
        net_income: Decimal | float | int | str,
        average_equity: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        ROE.

            Net Income / Average Equity * 100
        """

        return cls.percentage(
            net_income,
            average_equity,
            precision=precision,
        )

    # ========================================================================
    # EFFICIENCY
    # ========================================================================

    @classmethod
    def asset_turnover(
        cls,
        revenue: Decimal | float | int | str,
        average_assets: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """Calculate asset turnover."""

        return cls.divide(
            revenue,
            average_assets,
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def inventory_turnover(
        cls,
        cogs: Decimal | float | int | str,
        average_inventory: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """Calculate inventory turnover."""

        return cls.divide(
            cogs,
            average_inventory,
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def receivables_turnover(
        cls,
        credit_sales: Decimal | float | int | str,
        average_accounts_receivable: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """Calculate receivables turnover."""

        return cls.divide(
            credit_sales,
            average_accounts_receivable,
            precision=precision,
        )

    # ========================================================================
    # DAYS METRICS
    # ========================================================================

    @classmethod
    def days_sales_outstanding(
        cls,
        accounts_receivable: Decimal | float | int | str,
        credit_sales: Decimal | float | int | str,
        days: int = 365,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate DSO.

            AR / Credit Sales * Number of Days
        """

        receivables = cls.to_decimal(
            accounts_receivable
        )

        sales = cls.to_decimal(
            credit_sales
        )

        if sales == 0:
            raise ZeroDivisionError(
                "DSO cannot be calculated with zero sales."
            )

        return (
            receivables
            / sales
            * cls.to_decimal(days)
        ).quantize(
            Decimal("1").scaleb(-precision)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def days_inventory_outstanding(
        cls,
        average_inventory: Decimal | float | int | str,
        cogs: Decimal | float | int | str,
        days: int = 365,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate DIO.

            Average Inventory / COGS * Days
        """

        inventory = cls.to_decimal(
            average_inventory
        )

        cogs_decimal = cls.to_decimal(
            cogs
        )

        if cogs_decimal == 0:
            raise ZeroDivisionError(
                "DIO cannot be calculated with zero COGS."
            )

        return (
            inventory
            / cogs_decimal
            * cls.to_decimal(days)
        ).quantize(
            Decimal("1").scaleb(-precision)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def days_payables_outstanding(
        cls,
        accounts_payable: Decimal | float | int | str,
        purchases: Decimal | float | int | str,
        days: int = 365,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate DPO.

            Accounts Payable / Purchases * Days
        """

        payable = cls.to_decimal(
            accounts_payable
        )

        purchases_decimal = cls.to_decimal(
            purchases
        )

        if purchases_decimal == 0:
            raise ZeroDivisionError(
                "DPO cannot be calculated with zero purchases."
            )

        return (
            payable
            / purchases_decimal
            * cls.to_decimal(days)
        ).quantize(
            Decimal("1").scaleb(-precision)
        )

    # ========================================================================
    # WORKING CAPITAL
    # ========================================================================

    @classmethod
    def working_capital(
        cls,
        current_assets: Decimal | float | int | str,
        current_liabilities: Decimal | float | int | str,
    ) -> Decimal:
        """Calculate working capital."""

        return (
            cls.to_decimal(current_assets)
            - cls.to_decimal(current_liabilities)
        )

    # ========================================================================
    # EBITDA
    # ========================================================================

    @classmethod
    def ebitda(
        cls,
        net_income: Decimal | float | int | str,
        interest: Decimal | float | int | str,
        taxes: Decimal | float | int | str,
        depreciation: Decimal | float | int | str,
        amortization: Decimal | float | int | str,
    ) -> Decimal:
        """
        Calculate EBITDA.

            Net Income
            + Interest
            + Taxes
            + Depreciation
            + Amortization
        """

        return (
            cls.to_decimal(net_income)
            + cls.to_decimal(interest)
            + cls.to_decimal(taxes)
            + cls.to_decimal(depreciation)
            + cls.to_decimal(amortization)
        )

    # ------------------------------------------------------------------------

    @classmethod
    def ebitda_margin(
        cls,
        ebitda: Decimal | float | int | str,
        revenue: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """Calculate EBITDA margin."""

        return cls.percentage(
            ebitda,
            revenue,
            precision=precision,
        )

    # ========================================================================
    # FINANCIAL HEALTH
    # ========================================================================

    @classmethod
    def interest_coverage_ratio(
        cls,
        ebit: Decimal | float | int | str,
        interest_expense: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate interest coverage.

            EBIT / Interest Expense
        """

        return cls.divide(
            ebit,
            interest_expense,
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def cash_ratio(
        cls,
        cash: Decimal | float | int | str,
        current_liabilities: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """Calculate cash ratio."""

        return cls.divide(
            cash,
            current_liabilities,
            precision=precision,
        )

    # ========================================================================
    # BUDGET UTILIZATION
    # ========================================================================

    @classmethod
    def budget_utilization(
        cls,
        budget: Decimal | float | int | str,
        actual: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Calculate percentage of budget consumed.

            Actual / Budget * 100
        """

        budget_decimal = cls.to_decimal(
            budget
        )

        actual_decimal = cls.to_decimal(
            actual
        )

        if budget_decimal == 0:

            if actual_decimal == 0:
                return Decimal("0")

            return Decimal("100")

        return cls.percentage(
            actual_decimal,
            abs(budget_decimal),
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def remaining_budget(
        cls,
        budget: Decimal | float | int | str,
        actual: Decimal | float | int | str,
    ) -> Decimal:
        """
        Calculate remaining budget.

        Positive:
            budget remains

        Negative:
            budget has been exceeded
        """

        return (
            cls.to_decimal(budget)
            - cls.to_decimal(actual)
        )

    # ========================================================================
    # UNIT ECONOMICS
    # ========================================================================

    @classmethod
    def average_revenue_per_customer(
        cls,
        revenue: Decimal | float | int | str,
        customers: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """Calculate average revenue per customer."""

        return cls.divide(
            revenue,
            customers,
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def customer_acquisition_cost(
        cls,
        sales_marketing_cost: Decimal | float | int | str,
        new_customers: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        CAC.

            Sales & Marketing Cost / New Customers
        """

        return cls.divide(
            sales_marketing_cost,
            new_customers,
            precision=precision,
        )

    # ------------------------------------------------------------------------

    @classmethod
    def lifetime_value(
        cls,
        average_revenue_per_customer: Decimal | float | int | str,
        gross_margin_percent: Decimal | float | int | str,
        customer_lifetime: Decimal | float | int | str,
        *,
        precision: int = 2,
    ) -> Decimal:
        """
        Simplified customer lifetime value.

            ARPC
            * Gross Margin %
            * Customer Lifetime
        """

        arpc = cls.to_decimal(
            average_revenue_per_customer
        )

        margin = (
            cls.to_decimal(
                gross_margin_percent
            )
            / Decimal("100")
        )

        lifetime = cls.to_decimal(
            customer_lifetime
        )

        return (
            arpc
            * margin
            * lifetime
        ).quantize(
            Decimal("1").scaleb(-precision)
        )

    # ========================================================================
    # ROUNDING / NORMALIZATION
    # ========================================================================

    @classmethod
    def round_money(
        cls,
        value: Decimal | float | int | str,
    ) -> Decimal:
        """Round value to standard two-decimal currency precision."""

        return cls.to_decimal(value).quantize(
            cls.QUANTIZE_2
        )

    # ------------------------------------------------------------------------

    @classmethod
    def round_percentage(
        cls,
        value: Decimal | float | int | str,
    ) -> Decimal:
        """Round percentage to two decimals."""

        return cls.to_decimal(value).quantize(
            cls.QUANTIZE_2
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    @staticmethod
    def validate_non_negative(
        value: Decimal | float | int | str,
        field_name: str = "value",
    ) -> Decimal:
        """
        Validate a value that should not be negative.

        Useful for:
            - revenue
            - assets
            - inventory
            - customer counts
        """

        decimal_value = FinancialCalculator.to_decimal(
            value
        )

        if decimal_value < 0:
            raise ValueError(
                f"{field_name} cannot be negative."
            )

        return decimal_value

    # ========================================================================
    # CONVERSION
    # ========================================================================

    @staticmethod
    def to_decimal(
        value: Decimal | float | int | str,
    ) -> Decimal:
        """
        Convert a numeric value into Decimal.

        Floats are converted through str() to avoid
        inheriting binary floating-point artifacts.
        """

        if isinstance(value, bool):
            raise ValueError(
                "Boolean values are not valid financial numbers."
            )

        if isinstance(value, Decimal):
            return value

        try:

            if isinstance(value, float):
                return Decimal(str(value))

            return Decimal(value)

        except (
            InvalidOperation,
            TypeError,
            ValueError,
        ) as exc:

            raise ValueError(
                f"Invalid financial value: {value!r}"
            ) from exc


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def calculate_percentage(
    numerator: Decimal | float | int | str,
    denominator: Decimal | float | int | str,
) -> Decimal:
    """Convenience wrapper for percentage calculation."""

    return FinancialCalculator.percentage(
        numerator,
        denominator,
    )


def calculate_growth(
    previous: Decimal | float | int | str,
    current: Decimal | float | int | str,
) -> GrowthResult:
    """Convenience wrapper for growth calculation."""

    return FinancialCalculator.growth(
        previous,
        current,
    )


def calculate_variance(
    expected: Decimal | float | int | str,
    actual: Decimal | float | int | str,
) -> Decimal:
    """Convenience wrapper for variance calculation."""

    return FinancialCalculator.variance(
        expected,
        actual,
    )


def calculate_net_profit(
    revenue: Decimal | float | int | str,
    cogs: Decimal | float | int | str,
    operating_expenses: Decimal | float | int | str,
    interest_expense: Decimal | float | int | str = 0,
    taxes: Decimal | float | int | str = 0,
) -> Decimal:
    """Convenience wrapper for net profit."""

    return FinancialCalculator.net_profit(
        revenue=revenue,
        cogs=cogs,
        operating_expenses=operating_expenses,
        interest_expense=interest_expense,
        taxes=taxes,
    )


# ============================================================================
# SELF TEST / DEMO
# ============================================================================


if __name__ == "__main__":

    calculator = FinancialCalculator()

    revenue = Decimal("1000000")
    cogs = Decimal("600000")
    operating_expenses = Decimal("200000")

    gross_profit = calculator.gross_profit(
        revenue,
        cogs,
    )

    gross_margin = calculator.gross_margin(
        revenue,
        cogs,
    )

    operating_profit = calculator.operating_profit(
        gross_profit,
        operating_expenses,
    )

    operating_margin = calculator.operating_margin(
        revenue,
        operating_expenses,
        cogs,
    )

    net_profit = calculator.net_profit(
        revenue=revenue,
        cogs=cogs,
        operating_expenses=operating_expenses,
        interest_expense=20000,
        taxes=30000,
    )

    net_margin = calculator.net_margin(
        revenue,
        net_profit,
    )

    growth = calculator.growth(
        previous=800000,
        current=1000000,
    )

    variance = calculator.variance(
        expected=900000,
        actual=1000000,
    )

    variance_percent = calculator.variance_percent(
        expected=900000,
        actual=1000000,
    )

    print("=" * 70)
    print("FINCO AI - FINANCIAL CALCULATOR")
    print("=" * 70)

    print(f"Revenue          : {revenue:,.2f}")
    print(f"COGS             : {cogs:,.2f}")
    print(f"Gross Profit     : {gross_profit:,.2f}")
    print(f"Gross Margin     : {gross_margin:.2f}%")
    print(f"Operating Profit : {operating_profit:,.2f}")
    print(f"Operating Margin : {operating_margin:.2f}%")
    print(f"Net Profit       : {net_profit:,.2f}")
    print(f"Net Margin       : {net_margin:.2f}%")

    print("\nGrowth Analysis")
    print(f"Change           : {growth.change:,.2f}")
    print(f"Growth %         : {growth.change_percent:.2f}%")
    print(f"Direction        : {growth.direction.value}")

    print("\nBudget Variance")
    print(f"Variance         : {variance:,.2f}")
    print(f"Variance %       : {variance_percent:.2f}%")