"""
FinCo AI - Cash Flow Analysis

Core cash-flow calculations used by:
    - Financial analysis
    - Alert engine
    - Forecasting
    - What-if analysis
    - Recommendation engine
    - Agentic AI tools

This module contains business logic only.
API, database, and persistence concerns belong in their respective layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Optional, Union


NumberLike = Union[Decimal, int, float, str]
ZERO = Decimal("0")


def to_decimal(value: NumberLike | None) -> Decimal:
    """Convert a numeric value to Decimal safely."""

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


@dataclass(frozen=True)
class CashFlowInput:
    """Input data required for cash-flow analysis."""

    beginning_cash: Decimal = ZERO

    # Operating activities
    net_income: Decimal = ZERO
    depreciation: Decimal = ZERO
    amortization: Decimal = ZERO
    change_in_working_capital: Decimal = ZERO
    other_operating_adjustments: Decimal = ZERO

    # Investing activities
    capital_expenditure: Decimal = ZERO
    asset_sales: Decimal = ZERO
    investments_purchased: Decimal = ZERO
    investments_sold: Decimal = ZERO
    other_investing_cash_flow: Decimal = ZERO

    # Financing activities
    debt_proceeds: Decimal = ZERO
    debt_repayment: Decimal = ZERO
    equity_raised: Decimal = ZERO
    dividends_paid: Decimal = ZERO
    other_financing_cash_flow: Decimal = ZERO


@dataclass(frozen=True)
class CashFlowResult:
    """Calculated cash-flow metrics."""

    beginning_cash: Decimal

    operating_cash_flow: Decimal
    investing_cash_flow: Decimal
    financing_cash_flow: Decimal

    net_change_in_cash: Decimal
    ending_cash: Decimal

    free_cash_flow: Decimal

    cash_burn: Decimal
    runway_months: Optional[Decimal]

    operating_cash_flow_margin: Optional[Decimal]
    free_cash_flow_margin: Optional[Decimal]

    investing_intensity: Optional[Decimal]
    debt_dependency: Optional[Decimal]

    status: str


class CashFlowAnalyzer:
    """
    FinCo AI cash-flow analyzer.

    This class is stateless and can safely be reused across requests.
    """

    def analyze(
        self,
        data: CashFlowInput,
        revenue: NumberLike = ZERO,
        monthly_cash_burn: NumberLike | None = None,
    ) -> CashFlowResult:
        """Calculate the complete cash-flow profile."""

        beginning_cash = to_decimal(data.beginning_cash)
        revenue = to_decimal(revenue)

        operating_cash_flow = self.calculate_operating_cash_flow(data)
        investing_cash_flow = self.calculate_investing_cash_flow(data)
        financing_cash_flow = self.calculate_financing_cash_flow(data)

        net_change_in_cash = (
            operating_cash_flow
            + investing_cash_flow
            + financing_cash_flow
        )

        ending_cash = beginning_cash + net_change_in_cash

        free_cash_flow = self.calculate_free_cash_flow(
            operating_cash_flow,
            data.capital_expenditure,
        )

        cash_burn = self.calculate_cash_burn(
            operating_cash_flow,
            investing_cash_flow,
            monthly_cash_burn,
        )

        runway_months = self.calculate_runway(
            ending_cash,
            cash_burn,
        )

        status = self.cash_flow_status(
            operating_cash_flow,
            ending_cash,
        )

        return CashFlowResult(
            beginning_cash=beginning_cash,
            operating_cash_flow=operating_cash_flow,
            investing_cash_flow=investing_cash_flow,
            financing_cash_flow=financing_cash_flow,
            net_change_in_cash=net_change_in_cash,
            ending_cash=ending_cash,
            free_cash_flow=free_cash_flow,
            cash_burn=cash_burn,
            runway_months=runway_months,
            operating_cash_flow_margin=safe_divide(
                operating_cash_flow,
                revenue,
            ),
            free_cash_flow_margin=safe_divide(
                free_cash_flow,
                revenue,
            ),
            investing_intensity=safe_divide(
                abs(to_decimal(data.capital_expenditure)),
                revenue,
            ),
            debt_dependency=self.calculate_debt_dependency(
                data.debt_proceeds,
                net_change_in_cash,
            ),
            status=status,
        )

    @staticmethod
    def calculate_operating_cash_flow(
        data: CashFlowInput,
    ) -> Decimal:
        """
        Calculate operating cash flow using a simplified
        indirect-method approach.

        OCF =
            Net Income
            + Depreciation
            + Amortization
            + Change in Working Capital
            + Other Operating Adjustments
        """

        return (
            to_decimal(data.net_income)
            + to_decimal(data.depreciation)
            + to_decimal(data.amortization)
            + to_decimal(data.change_in_working_capital)
            + to_decimal(data.other_operating_adjustments)
        )

    @staticmethod
    def calculate_investing_cash_flow(
        data: CashFlowInput,
    ) -> Decimal:
        """
        Calculate investing cash flow.

        Expected convention:
            Capital expenditure -> negative
            Asset sales -> positive
            Investments purchased -> negative
            Investments sold -> positive
        """

        return (
            to_decimal(data.capital_expenditure)
            + to_decimal(data.asset_sales)
            + to_decimal(data.investments_purchased)
            + to_decimal(data.investments_sold)
            + to_decimal(data.other_investing_cash_flow)
        )

    @staticmethod
    def calculate_financing_cash_flow(
        data: CashFlowInput,
    ) -> Decimal:
        """
        Calculate financing cash flow.

        Expected convention:
            Debt proceeds -> positive
            Debt repayment -> negative
            Equity raised -> positive
            Dividends paid -> negative
        """

        return (
            to_decimal(data.debt_proceeds)
            + to_decimal(data.debt_repayment)
            + to_decimal(data.equity_raised)
            + to_decimal(data.dividends_paid)
            + to_decimal(data.other_financing_cash_flow)
        )

    @staticmethod
    def calculate_free_cash_flow(
        operating_cash_flow: NumberLike,
        capital_expenditure: NumberLike,
    ) -> Decimal:
        """
        Calculate free cash flow.

        FCF = Operating Cash Flow + Capital Expenditure

        Capital expenditure is expected to be represented as
        a negative cash-flow value.
        """

        return (
            to_decimal(operating_cash_flow)
            + to_decimal(capital_expenditure)
        )

    @staticmethod
    def calculate_cash_burn(
        operating_cash_flow: NumberLike,
        investing_cash_flow: NumberLike,
        monthly_cash_burn: NumberLike | None = None,
    ) -> Decimal:
        """
        Calculate cash burn.

        If monthly_cash_burn is explicitly provided, it is used.

        Otherwise:
            Burn = -(Operating CF + Investing CF)

        Only negative cash generation is treated as burn.
        """

        if monthly_cash_burn is not None:
            return max(to_decimal(monthly_cash_burn), ZERO)

        operating_cash_flow = to_decimal(operating_cash_flow)
        investing_cash_flow = to_decimal(investing_cash_flow)

        cash_generation = operating_cash_flow + investing_cash_flow

        return max(-cash_generation, ZERO)

    @staticmethod
    def calculate_runway(
        available_cash: NumberLike,
        monthly_burn: NumberLike,
    ) -> Optional[Decimal]:
        """
        Estimate cash runway in months.

        Runway = Available Cash / Monthly Burn

        Returns None when there is no cash burn.
        """

        available_cash = to_decimal(available_cash)
        monthly_burn = to_decimal(monthly_burn)

        if monthly_burn <= ZERO:
            return None

        return max(available_cash, ZERO) / monthly_burn

    @staticmethod
    def calculate_debt_dependency(
        debt_proceeds: NumberLike,
        net_change_in_cash: NumberLike,
    ) -> Optional[Decimal]:
        """
        Estimate how dependent the period's cash position was
        on new debt.

        This is a simple analytical indicator, not a formal
        accounting ratio.
        """

        debt_proceeds = max(to_decimal(debt_proceeds), ZERO)
        net_change_in_cash = abs(to_decimal(net_change_in_cash))

        if net_change_in_cash == ZERO:
            return None

        return debt_proceeds / net_change_in_cash

    @staticmethod
    def period_change(
        current: NumberLike,
        previous: NumberLike,
    ) -> Optional[Decimal]:
        """
        Calculate percentage change between two periods.

        Returns None when the previous value is zero.

        Example:
            previous = 100
            current = 80

            result = -20%
        """

        current = to_decimal(current)
        previous = to_decimal(previous)

        if previous == ZERO:
            return None

        return ((current - previous) / abs(previous)) * Decimal("100")

    @staticmethod
    def cash_flow_status(
        operating_cash_flow: NumberLike,
        ending_cash: NumberLike,
    ) -> str:
        """
        Classify the overall cash-flow condition.

        Detailed risk scoring belongs in:
            backend/app/alerts/risk_scoring.py
        """

        operating_cash_flow = to_decimal(operating_cash_flow)
        ending_cash = to_decimal(ending_cash)

        if ending_cash < ZERO:
            return "CRITICAL"

        if operating_cash_flow < ZERO:
            return "WARNING"

        if operating_cash_flow == ZERO:
            return "NEUTRAL"

        return "HEALTHY"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def calculate_cash_flow(
    data: CashFlowInput,
    revenue: NumberLike = ZERO,
    monthly_cash_burn: NumberLike | None = None,
) -> CashFlowResult:
    """
    Convenience wrapper around CashFlowAnalyzer.

    Useful for services and agent tools that do not need to
    instantiate the analyzer explicitly.
    """

    analyzer = CashFlowAnalyzer()

    return analyzer.analyze(
        data=data,
        revenue=revenue,
        monthly_cash_burn=monthly_cash_burn,
    )


def calculate_net_cash_change(
    operating_cash_flow: NumberLike,
    investing_cash_flow: NumberLike,
    financing_cash_flow: NumberLike,
) -> Decimal:
    """Calculate total change in cash for a period."""

    return (
        to_decimal(operating_cash_flow)
        + to_decimal(investing_cash_flow)
        + to_decimal(financing_cash_flow)
    )


def calculate_ending_cash(
    beginning_cash: NumberLike,
    operating_cash_flow: NumberLike,
    investing_cash_flow: NumberLike,
    financing_cash_flow: NumberLike,
) -> Decimal:
    """Calculate ending cash."""

    return (
        to_decimal(beginning_cash)
        + calculate_net_cash_change(
            operating_cash_flow,
            investing_cash_flow,
            financing_cash_flow,
        )
    )