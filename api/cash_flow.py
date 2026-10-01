"""
FinCo AI - Cash Flow Analysis

Responsibilities:
- Operating cash flow analysis
- Investing cash flow analysis
- Financing cash flow analysis
- Net cash flow
- Free cash flow
- Cash conversion
- Operating cash flow margin
- CapEx analysis
- Debt financing analysis
- Cash burn / runway
- Liquidity analysis
- Period-over-period comparison
- Cash-flow health scoring

Design principle:
    Deterministic financial calculations belong here.
    LLMs/agents may explain the results, but should not
    calculate the underlying financial values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional


# ============================================================
# Types
# ============================================================

Number = int | float | Decimal | str


# ============================================================
# Helpers
# ============================================================

def to_decimal(value: Number | None) -> Decimal:
    """
    Convert a supported numeric value to Decimal.

    None is treated as zero.
    """

    if value is None:
        return Decimal("0")

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid financial value: {value!r}"
        ) from exc


def safe_divide(
    numerator: Number,
    denominator: Number,
) -> Decimal:
    """Safely divide two financial values."""

    numerator_decimal = to_decimal(numerator)
    denominator_decimal = to_decimal(denominator)

    if denominator_decimal == 0:
        return Decimal("0")

    return numerator_decimal / denominator_decimal


def percentage(
    value: Number,
    total: Number,
) -> Decimal:
    """Return percentage value."""

    return safe_divide(
        value,
        total,
    ) * Decimal("100")


def round_decimal(
    value: Decimal,
    places: int = 2,
) -> Decimal:
    """Round Decimal to requested decimal places."""

    quantum = Decimal("1").scaleb(-places)

    return value.quantize(
        quantum
    )


# ============================================================
# Cash Flow Data
# ============================================================

@dataclass
class CashFlowData:
    """
    Normalized cash-flow statement data.

    Positive values represent cash inflows.
    Negative values represent cash outflows.
    """

    # --------------------------------------------------------
    # Operating Activities
    # --------------------------------------------------------

    net_income: Number = 0

    depreciation: Number = 0
    amortization: Number = 0

    change_accounts_receivable: Number = 0
    change_inventory: Number = 0
    change_prepaid_expenses: Number = 0

    change_accounts_payable: Number = 0
    change_accrued_expenses: Number = 0
    change_taxes_payable: Number = 0

    other_operating_adjustments: Number = 0

    # --------------------------------------------------------
    # Investing Activities
    # --------------------------------------------------------

    capital_expenditures: Number = 0
    property_plant_equipment_purchases: Number = 0
    asset_sale_proceeds: Number = 0
    investment_purchases: Number = 0
    investment_sale_proceeds: Number = 0
    acquisition_spending: Number = 0
    other_investing_cash_flow: Number = 0

    # --------------------------------------------------------
    # Financing Activities
    # --------------------------------------------------------

    debt_proceeds: Number = 0
    debt_repayments: Number = 0

    equity_issuance: Number = 0
    share_buybacks: Number = 0

    dividends_paid: Number = 0

    other_financing_cash_flow: Number = 0

    # --------------------------------------------------------
    # Cash Position
    # --------------------------------------------------------

    beginning_cash: Number = 0
    ending_cash_reported: Optional[Number] = None

    # --------------------------------------------------------
    # Income Statement Context
    # --------------------------------------------------------

    revenue: Number = 0

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    currency: str = "USD"
    period: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Analysis Result
# ============================================================

@dataclass
class CashFlowAnalysis:
    """
    Complete cash-flow analysis result.
    """

    operating_cash_flow: Decimal
    investing_cash_flow: Decimal
    financing_cash_flow: Decimal

    net_cash_flow: Decimal

    beginning_cash: Decimal
    ending_cash: Decimal

    free_cash_flow: Decimal

    capital_expenditures: Decimal

    operating_cash_flow_margin: Decimal
    cash_conversion_ratio: Decimal

    investing_cash_flow_ratio: Decimal
    financing_cash_flow_ratio: Decimal

    debt_raised: Decimal
    debt_repaid: Decimal
    net_debt_cash_flow: Decimal

    dividends_paid: Decimal

    health_score: Decimal

    warnings: list[str]
    strengths: list[str]

    cash_position_status: str

    currency: str = "USD"
    period: Optional[str] = None


# ============================================================
# Cash Flow Analyzer
# ============================================================

class CashFlowAnalyzer:
    """
    Deterministic cash-flow analysis engine.
    """

    # ========================================================
    # Operating Cash Flow
    # ========================================================

    @staticmethod
    def operating_cash_flow(
        data: CashFlowData,
    ) -> Decimal:
        """
        Calculate cash generated from operating activities.

        Simplified indirect-method calculation:

        Net Income
        + Depreciation
        + Amortization
        + Working Capital Adjustments
        + Other Operating Adjustments
        """

        return sum(
            (
                to_decimal(data.net_income),

                to_decimal(
                    data.depreciation
                ),

                to_decimal(
                    data.amortization
                ),

                to_decimal(
                    data.change_accounts_receivable
                ),

                to_decimal(
                    data.change_inventory
                ),

                to_decimal(
                    data.change_prepaid_expenses
                ),

                to_decimal(
                    data.change_accounts_payable
                ),

                to_decimal(
                    data.change_accrued_expenses
                ),

                to_decimal(
                    data.change_taxes_payable
                ),

                to_decimal(
                    data.other_operating_adjustments
                ),
            ),
            Decimal("0"),
        )

    # ========================================================
    # Investing Cash Flow
    # ========================================================

    @staticmethod
    def investing_cash_flow(
        data: CashFlowData,
    ) -> Decimal:
        """
        Calculate cash generated/used by investing activities.

        Purchases are expected as negative values.
        """

        capex = (
            to_decimal(
                data.capital_expenditures
            )
            + to_decimal(
                data.property_plant_equipment_purchases
            )
        )

        investment_activity = (
            to_decimal(
                data.investment_purchases
            )
            + to_decimal(
                data.investment_sale_proceeds
            )
        )

        return sum(
            (
                capex,
                to_decimal(
                    data.asset_sale_proceeds
                ),
                investment_activity,
                to_decimal(
                    data.acquisition_spending
                ),
                to_decimal(
                    data.other_investing_cash_flow
                ),
            ),
            Decimal("0"),
        )

    # ========================================================
    # Financing Cash Flow
    # ========================================================

    @staticmethod
    def financing_cash_flow(
        data: CashFlowData,
    ) -> Decimal:
        """
        Calculate cash generated/used by financing activities.
        """

        return sum(
            (
                to_decimal(data.debt_proceeds),

                to_decimal(
                    data.debt_repayments
                ),

                to_decimal(
                    data.equity_issuance
                ),

                to_decimal(
                    data.share_buybacks
                ),

                to_decimal(
                    data.dividends_paid
                ),

                to_decimal(
                    data.other_financing_cash_flow
                ),
            ),
            Decimal("0"),
        )

    # ========================================================
    # Net Cash Flow
    # ========================================================

    @classmethod
    def net_cash_flow(
        cls,
        data: CashFlowData,
    ) -> Decimal:

        return (
            cls.operating_cash_flow(data)
            + cls.investing_cash_flow(data)
            + cls.financing_cash_flow(data)
        )

    # ========================================================
    # Ending Cash
    # ========================================================

    @classmethod
    def calculated_ending_cash(
        cls,
        data: CashFlowData,
    ) -> Decimal:

        return (
            to_decimal(data.beginning_cash)
            + cls.net_cash_flow(data)
        )

    @classmethod
    def ending_cash(
        cls,
        data: CashFlowData,
    ) -> Decimal:

        if data.ending_cash_reported is not None:

            return to_decimal(
                data.ending_cash_reported
            )

        return cls.calculated_ending_cash(
            data
        )

    # ========================================================
    # Free Cash Flow
    # ========================================================

    @classmethod
    def free_cash_flow(
        cls,
        data: CashFlowData,
    ) -> Decimal:
        """
        Free Cash Flow = Operating Cash Flow - CapEx

        CapEx is normalized as a positive spending amount
        for this calculation.
        """

        operating_cf = cls.operating_cash_flow(
            data
        )

        capex = cls.capex_amount(
            data
        )

        return operating_cf - capex

    # ========================================================
    # Capital Expenditure
    # ========================================================

    @staticmethod
    def capex_amount(
        data: CashFlowData,
    ) -> Decimal:
        """
        Return CapEx as a positive spending amount.

        Supports both:
        - capital_expenditures
        - property_plant_equipment_purchases

        If the source data stores purchases as negative values,
        their absolute value is used.
        """

        capex = (
            to_decimal(
                data.capital_expenditures
            )
            + to_decimal(
                data.property_plant_equipment_purchases
            )
        )

        return abs(capex)

    # ========================================================
    # Cash Conversion
    # ========================================================

    @classmethod
    def cash_conversion_ratio(
        cls,
        data: CashFlowData,
    ) -> Decimal:
        """
        Cash conversion ratio:

            Operating Cash Flow / Net Income

        Values above 1 generally indicate strong conversion
        of accounting earnings into operating cash.
        """

        return safe_divide(
            cls.operating_cash_flow(data),
            data.net_income,
        )

    # ========================================================
    # Operating Cash Flow Margin
    # ========================================================

    @classmethod
    def operating_cash_flow_margin(
        cls,
        data: CashFlowData,
    ) -> Decimal:

        return percentage(
            cls.operating_cash_flow(data),
            data.revenue,
        )

    # ========================================================
    # Cash Flow Ratios
    # ========================================================

    @classmethod
    def investing_cash_flow_ratio(
        cls,
        data: CashFlowData,
    ) -> Decimal:

        return safe_divide(
            cls.investing_cash_flow(data),
            cls.operating_cash_flow(data),
        )

    @classmethod
    def financing_cash_flow_ratio(
        cls,
        data: CashFlowData,
    ) -> Decimal:

        return safe_divide(
            cls.financing_cash_flow(data),
            cls.operating_cash_flow(data),
        )

    # ========================================================
    # Debt Cash Flow
    # ========================================================

    @staticmethod
    def debt_raised(
        data: CashFlowData,
    ) -> Decimal:

        return max(
            Decimal("0"),
            to_decimal(
                data.debt_proceeds
            ),
        )

    @staticmethod
    def debt_repaid(
        data: CashFlowData,
    ) -> Decimal:

        return abs(
            min(
                Decimal("0"),
                to_decimal(
                    data.debt_repayments
                ),
            )
        )

    @classmethod
    def net_debt_cash_flow(
        cls,
        data: CashFlowData,
    ) -> Decimal:

        return (
            cls.debt_raised(data)
            - cls.debt_repaid(data)
        )

    # ========================================================
    # Cash Burn
    # ========================================================

    @classmethod
    def cash_burn(
        cls,
        data: CashFlowData,
    ) -> Decimal:
        """
        Calculate cash burn when operating cash flow is negative.

        Returns a positive burn amount.
        """

        operating_cf = cls.operating_cash_flow(
            data
        )

        if operating_cf >= 0:
            return Decimal("0")

        return abs(operating_cf)

    @classmethod
    def cash_runway_months(
        cls,
        data: CashFlowData,
        monthly_burn: Optional[Number] = None,
    ) -> Decimal:
        """
        Estimate cash runway.

        If monthly_burn is supplied, it is used directly.

        Otherwise, this method assumes the supplied period
        represents one month.
        """

        ending_cash = cls.ending_cash(
            data
        )

        if monthly_burn is None:
            burn = cls.cash_burn(data)
        else:
            burn = abs(
                to_decimal(monthly_burn)
            )

        if burn <= 0:
            return Decimal("999")

        return safe_divide(
            ending_cash,
            burn,
        )

    # ========================================================
    # Cash Position
    # ========================================================

    @classmethod
    def cash_position_status(
        cls,
        data: CashFlowData,
    ) -> str:

        ending_cash = cls.ending_cash(
            data
        )

        operating_cf = cls.operating_cash_flow(
            data
        )

        if ending_cash <= 0:
            return "critical"

        if operating_cf < 0:
            return "under_pressure"

        if operating_cf > 0:
            return "healthy"

        return "stable"

    # ========================================================
    # Health Score
    # ========================================================

    @classmethod
    def calculate_health_score(
        cls,
        data: CashFlowData,
    ) -> Decimal:
        """
        Demo financial health scoring model.

        Score range:
            0 - 100

        This is a portfolio analytical score and not a
        regulated financial rating.
        """

        score = Decimal("50")

        operating_cf = cls.operating_cash_flow(
            data
        )

        free_cf = cls.free_cash_flow(
            data
        )

        cash_conversion = cls.cash_conversion_ratio(
            data
        )

        ending_cash = cls.ending_cash(
            data
        )

        # Positive operating cash flow.
        if operating_cf > 0:
            score += Decimal("15")
        elif operating_cf < 0:
            score -= Decimal("15")

        # Positive free cash flow.
        if free_cf > 0:
            score += Decimal("15")
        elif free_cf < 0:
            score -= Decimal("10")

        # Cash conversion.
        if cash_conversion >= Decimal("1"):
            score += Decimal("10")
        elif (
            cash_conversion > Decimal("0.5")
        ):
            score += Decimal("5")
        elif cash_conversion < Decimal("0"):
            score -= Decimal("10")

        # Ending cash.
        if ending_cash > 0:
            score += Decimal("5")
        else:
            score -= Decimal("15")

        return max(
            Decimal("0"),
            min(
                Decimal("100"),
                score,
            ),
        )

    # ========================================================
    # Warnings
    # ========================================================

    @classmethod
    def warnings(
        cls,
        data: CashFlowData,
    ) -> list[str]:

        warnings: list[str] = []

        operating_cf = cls.operating_cash_flow(
            data
        )

        free_cf = cls.free_cash_flow(
            data
        )

        ending_cash = cls.ending_cash(
            data
        )

        cash_conversion = cls.cash_conversion_ratio(
            data
        )

        financing_cf = cls.financing_cash_flow(
            data
        )

        if operating_cf < 0:
            warnings.append(
                "Operating activities are consuming cash."
            )

        if free_cf < 0:
            warnings.append(
                "Free cash flow is negative."
            )

        if ending_cash <= 0:
            warnings.append(
                "Ending cash position is zero or negative."
            )

        if cash_conversion < Decimal("0.5"):
            warnings.append(
                "Cash conversion from reported earnings is weak."
            )

        if (
            financing_cf > 0
            and operating_cf < 0
        ):
            warnings.append(
                "Positive financing cash flow is offsetting "
                "negative operating cash flow."
            )

        if (
            cls.cash_burn(data) > 0
            and cls.cash_runway_months(data)
            < Decimal("3")
        ):
            warnings.append(
                "Estimated cash runway is below three months."
            )

        return warnings

    # ========================================================
    # Strengths
    # ========================================================

    @classmethod
    def strengths(
        cls,
        data: CashFlowData,
    ) -> list[str]:

        strengths: list[str] = []

        operating_cf = cls.operating_cash_flow(
            data
        )

        free_cf = cls.free_cash_flow(
            data
        )

        cash_conversion = cls.cash_conversion_ratio(
            data
        )

        if operating_cf > 0:
            strengths.append(
                "Operations are generating positive cash flow."
            )

        if free_cf > 0:
            strengths.append(
                "Business is generating positive free cash flow."
            )

        if cash_conversion >= Decimal("1"):
            strengths.append(
                "Operating cash generation is at least as strong "
                "as reported net income."
            )

        if cls.net_debt_cash_flow(data) < 0:
            strengths.append(
                "Debt repayments exceed new debt raised."
            )

        if cls.ending_cash(data) > 0:
            strengths.append(
                "The period ends with a positive cash balance."
            )

        return strengths

    # ========================================================
    # Complete Analysis
    # ========================================================

    @classmethod
    def analyze(
        cls,
        data: CashFlowData,
    ) -> CashFlowAnalysis:

        operating_cf = cls.operating_cash_flow(
            data
        )

        investing_cf = cls.investing_cash_flow(
            data
        )

        financing_cf = cls.financing_cash_flow(
            data
        )

        net_cf = (
            operating_cf
            + investing_cf
            + financing_cf
        )

        beginning_cash = to_decimal(
            data.beginning_cash
        )

        ending_cash = cls.ending_cash(
            data
        )

        return CashFlowAnalysis(
            operating_cash_flow=round_decimal(
                operating_cf
            ),

            investing_cash_flow=round_decimal(
                investing_cf
            ),

            financing_cash_flow=round_decimal(
                financing_cf
            ),

            net_cash_flow=round_decimal(
                net_cf
            ),

            beginning_cash=round_decimal(
                beginning_cash
            ),

            ending_cash=round_decimal(
                ending_cash
            ),

            free_cash_flow=round_decimal(
                cls.free_cash_flow(data)
            ),

            capital_expenditures=round_decimal(
                cls.capex_amount(data)
            ),

            operating_cash_flow_margin=round_decimal(
                cls.operating_cash_flow_margin(data),
                2,
            ),

            cash_conversion_ratio=round_decimal(
                cls.cash_conversion_ratio(data),
                4,
            ),

            investing_cash_flow_ratio=round_decimal(
                cls.investing_cash_flow_ratio(data),
                4,
            ),

            financing_cash_flow_ratio=round_decimal(
                cls.financing_cash_flow_ratio(data),
                4,
            ),

            debt_raised=round_decimal(
                cls.debt_raised(data)
            ),

            debt_repaid=round_decimal(
                cls.debt_repaid(data)
            ),

            net_debt_cash_flow=round_decimal(
                cls.net_debt_cash_flow(data)
            ),

            dividends_paid=round_decimal(
                abs(
                    to_decimal(
                        data.dividends_paid
                    )
                )
            ),

            health_score=round_decimal(
                cls.calculate_health_score(data)
            ),

            warnings=cls.warnings(data),

            strengths=cls.strengths(data),

            cash_position_status=(
                cls.cash_position_status(data)
            ),

            currency=data.currency,

            period=data.period,
        )


# ============================================================
# Period Comparison
# ============================================================

@dataclass
class CashFlowChange:
    """Period-over-period cash-flow change."""

    metric: str

    previous: Decimal

    current: Decimal

    absolute_change: Decimal

    percentage_change: Decimal

    direction: str


def calculate_change(
    metric: str,
    previous: Number,
    current: Number,
) -> CashFlowChange:

    previous_decimal = to_decimal(
        previous
    )

    current_decimal = to_decimal(
        current
    )

    absolute_change = (
        current_decimal
        - previous_decimal
    )

    if previous_decimal == 0:

        percentage_change = Decimal("0")

    else:

        percentage_change = (
            absolute_change
            / abs(previous_decimal)
        ) * Decimal("100")

    if absolute_change > 0:
        direction = "increase"

    elif absolute_change < 0:
        direction = "decrease"

    else:
        direction = "unchanged"

    return CashFlowChange(
        metric=metric,
        previous=round_decimal(
            previous_decimal
        ),
        current=round_decimal(
            current_decimal
        ),
        absolute_change=round_decimal(
            absolute_change
        ),
        percentage_change=round_decimal(
            percentage_change
        ),
        direction=direction,
    )


def compare_cash_flows(
    previous: CashFlowData,
    current: CashFlowData,
) -> list[CashFlowChange]:
    """
    Compare key cash-flow metrics across periods.
    """

    previous_metrics = {
        "operating_cash_flow":
            CashFlowAnalyzer.operating_cash_flow(
                previous
            ),

        "investing_cash_flow":
            CashFlowAnalyzer.investing_cash_flow(
                previous
            ),

        "financing_cash_flow":
            CashFlowAnalyzer.financing_cash_flow(
                previous
            ),

        "net_cash_flow":
            CashFlowAnalyzer.net_cash_flow(
                previous
            ),

        "free_cash_flow":
            CashFlowAnalyzer.free_cash_flow(
                previous
            ),

        "ending_cash":
            CashFlowAnalyzer.ending_cash(
                previous
            ),

        "capital_expenditures":
            CashFlowAnalyzer.capex_amount(
                previous
            ),
    }

    current_metrics = {
        "operating_cash_flow":
            CashFlowAnalyzer.operating_cash_flow(
                current
            ),

        "investing_cash_flow":
            CashFlowAnalyzer.investing_cash_flow(
                current
            ),

        "financing_cash_flow":
            CashFlowAnalyzer.financing_cash_flow(
                current
            ),

        "net_cash_flow":
            CashFlowAnalyzer.net_cash_flow(
                current
            ),

        "free_cash_flow":
            CashFlowAnalyzer.free_cash_flow(
                current
            ),

        "ending_cash":
            CashFlowAnalyzer.ending_cash(
                current
            ),

        "capital_expenditures":
            CashFlowAnalyzer.capex_amount(
                current
            ),
    }

    return [
        calculate_change(
            metric,
            previous_metrics[metric],
            current_metrics[metric],
        )
        for metric in previous_metrics
    ]


# ============================================================
# Dictionary Conversion
# ============================================================

def cash_flow_from_dict(
    data: Mapping[str, Any],
) -> CashFlowData:
    """
    Create CashFlowData from a dictionary.

    Useful for:
    - CSV
    - Excel
    - database records
    - extracted PDF tables
    - API payloads
    """

    return CashFlowData(
        net_income=data.get(
            "net_income",
            0,
        ),

        depreciation=data.get(
            "depreciation",
            0,
        ),

        amortization=data.get(
            "amortization",
            0,
        ),

        change_accounts_receivable=data.get(
            "change_accounts_receivable",
            0,
        ),

        change_inventory=data.get(
            "change_inventory",
            0,
        ),

        change_prepaid_expenses=data.get(
            "change_prepaid_expenses",
            0,
        ),

        change_accounts_payable=data.get(
            "change_accounts_payable",
            0,
        ),

        change_accrued_expenses=data.get(
            "change_accrued_expenses",
            0,
        ),

        change_taxes_payable=data.get(
            "change_taxes_payable",
            0,
        ),

        other_operating_adjustments=data.get(
            "other_operating_adjustments",
            0,
        ),

        capital_expenditures=data.get(
            "capital_expenditures",
            0,
        ),

        property_plant_equipment_purchases=data.get(
            "property_plant_equipment_purchases",
            0,
        ),

        asset_sale_proceeds=data.get(
            "asset_sale_proceeds",
            0,
        ),

        investment_purchases=data.get(
            "investment_purchases",
            0,
        ),

        investment_sale_proceeds=data.get(
            "investment_sale_proceeds",
            0,
        ),

        acquisition_spending=data.get(
            "acquisition_spending",
            0,
        ),

        other_investing_cash_flow=data.get(
            "other_investing_cash_flow",
            0,
        ),

        debt_proceeds=data.get(
            "debt_proceeds",
            0,
        ),

        debt_repayments=data.get(
            "debt_repayments",
            0,
        ),

        equity_issuance=data.get(
            "equity_issuance",
            0,
        ),

        share_buybacks=data.get(
            "share_buybacks",
            0,
        ),

        dividends_paid=data.get(
            "dividends_paid",
            0,
        ),

        other_financing_cash_flow=data.get(
            "other_financing_cash_flow",
            0,
        ),

        beginning_cash=data.get(
            "beginning_cash",
            0,
        ),

        ending_cash_reported=data.get(
            "ending_cash_reported"
        ),

        revenue=data.get(
            "revenue",
            0,
        ),

        currency=data.get(
            "currency",
            "USD",
        ),

        period=data.get(
            "period"
        ),

        metadata=dict(
            data.get(
                "metadata",
                {},
            )
        ),
    )


# ============================================================
# Public API
# ============================================================

def analyze_cash_flow(
    data: CashFlowData
    | Mapping[str, Any],
) -> CashFlowAnalysis:
    """
    Public helper for services, APIs and agents.
    """

    if isinstance(data, Mapping):
        data = cash_flow_from_dict(data)

    return CashFlowAnalyzer.analyze(
        data
    )


def get_cash_flow_metrics(
    data: CashFlowData
    | Mapping[str, Any],
) -> dict[str, Any]:
    """
    Return JSON/API-friendly cash-flow metrics.
    """

    result = analyze_cash_flow(
        data
    )

    return {
        "operating_cash_flow": str(
            result.operating_cash_flow
        ),

        "investing_cash_flow": str(
            result.investing_cash_flow
        ),

        "financing_cash_flow": str(
            result.financing_cash_flow
        ),

        "net_cash_flow": str(
            result.net_cash_flow
        ),

        "beginning_cash": str(
            result.beginning_cash
        ),

        "ending_cash": str(
            result.ending_cash
        ),

        "free_cash_flow": str(
            result.free_cash_flow
        ),

        "capital_expenditures": str(
            result.capital_expenditures
        ),

        "operating_cash_flow_margin": str(
            result.operating_cash_flow_margin
        ),

        "cash_conversion_ratio": str(
            result.cash_conversion_ratio
        ),

        "investing_cash_flow_ratio": str(
            result.investing_cash_flow_ratio
        ),

        "financing_cash_flow_ratio": str(
            result.financing_cash_flow_ratio
        ),

        "debt_raised": str(
            result.debt_raised
        ),

        "debt_repaid": str(
            result.debt_repaid
        ),

        "net_debt_cash_flow": str(
            result.net_debt_cash_flow
        ),

        "dividends_paid": str(
            result.dividends_paid
        ),

        "health_score": str(
            result.health_score
        ),

        "cash_position_status": (
            result.cash_position_status
        ),

        "warnings": result.warnings,

        "strengths": result.strengths,

        "currency": result.currency,

        "period": result.period,
    }