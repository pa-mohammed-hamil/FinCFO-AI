"""
FinCo AI - Financial Ratios Engine

File:
    backend/app/financial/ratios.py

Purpose:
    Calculate standardized financial ratios from normalized
    financial statement data.

Responsibilities:
    - Liquidity ratios
    - Solvency ratios
    - Leverage ratios
    - Profitability ratios
    - Efficiency/activity ratios
    - Coverage ratios
    - Cash-flow ratios
    - Working-capital ratios
    - Ratio classification
    - Ratio health assessment
    - Period-over-period ratio comparison
    - Ratio trend detection
    - Risk signal generation
    - Benchmark evaluation
    - Financial-ratio summaries

Architecture:

    Normalized Financial Data
              │
              ▼
        Financial Statements
              │
              ▼
           ratios.py
              │
       ┌──────┼───────────────┐
       ▼      ▼               ▼
    Liquidity Leverage    Profitability
       │      │               │
       └──────┼───────────────┘
              ▼
          Efficiency
              │
              ▼
         Cash Flow / Coverage
              │
              ▼
        Ratio Health Signals
              │
       ┌──────┼───────────────┐
       ▼      ▼               ▼
     Risk    Alerts         KPIs
       │      │               │
       └──────┼───────────────┘
              ▼
       Supervisor Agent
              │
       ┌──────┼──────────┐
       ▼      ▼          ▼
    Root Cause Forecast What-If
              │
              ▼
       Recommendation
              │
              ▼
         Human Review
              │
              ▼
          Audit Log
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Sequence


# ============================================================================
# Constants
# ============================================================================

ZERO = Decimal("0")
ONE = Decimal("1")
ONE_HUNDRED = Decimal("100")

RATIO_QUANT = Decimal("0.01")
PERCENT_QUANT = Decimal("0.01")
MONEY_QUANT = Decimal("0.01")


# ============================================================================
# Utility Functions
# ============================================================================


def to_decimal(
    value: Any,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely convert a value to Decimal."""

    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    if isinstance(value, bool):
        return ONE if value else ZERO

    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except Exception:
            return default

    text = str(value).strip()

    if not text:
        return default

    negative = False

    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = (
        text.replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("₹", "")
        .replace("%", "")
        .strip()
    )

    if text.startswith("-"):
        negative = True
        text = text[1:]

    try:
        result = Decimal(text)

        if negative:
            result = -abs(result)

        return result

    except Exception:
        return default


def round_ratio(
    value: Decimal,
) -> Decimal:
    """Round ratio to two decimal places."""

    return value.quantize(
        RATIO_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_percent(
    value: Decimal,
) -> Decimal:
    """Round percentage to two decimal places."""

    return value.quantize(
        PERCENT_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_money(
    value: Decimal,
) -> Decimal:
    """Round monetary values."""

    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely divide two Decimal values."""

    if denominator == ZERO:
        return default

    return numerator / denominator


def calculate_ratio(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal:
    """Calculate a financial ratio."""

    return round_ratio(
        safe_divide(
            numerator,
            denominator,
        )
    )


def calculate_percentage(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal:
    """Calculate a percentage."""

    return round_percent(
        safe_divide(
            numerator,
            denominator,
        )
        * ONE_HUNDRED
    )


def percentage_change(
    current: Decimal,
    previous: Decimal,
) -> Decimal | None:
    """
    Calculate period-over-period percentage change.

    Returns None when the previous value is zero and the
    current value is non-zero because the change is undefined.
    """

    if previous == ZERO:
        if current == ZERO:
            return ZERO

        return None

    return round_percent(
        safe_divide(
            current - previous,
            abs(previous),
        )
        * ONE_HUNDRED
    )


# ============================================================================
# Enums
# ============================================================================


class RatioCategory(str, Enum):
    """Financial ratio categories."""

    LIQUIDITY = "liquidity"
    PROFITABILITY = "profitability"
    LEVERAGE = "leverage"
    SOLVENCY = "solvency"
    EFFICIENCY = "efficiency"
    COVERAGE = "coverage"
    CASH_FLOW = "cash_flow"
    WORKING_CAPITAL = "working_capital"


class RatioType(str, Enum):
    """Supported financial ratios."""

    # Liquidity
    CURRENT_RATIO = "current_ratio"
    QUICK_RATIO = "quick_ratio"
    CASH_RATIO = "cash_ratio"
    WORKING_CAPITAL = "working_capital"
    WORKING_CAPITAL_TO_REVENUE = "working_capital_to_revenue"

    # Profitability
    GROSS_MARGIN = "gross_margin"
    OPERATING_MARGIN = "operating_margin"
    EBITDA_MARGIN = "ebitda_margin"
    NET_MARGIN = "net_margin"
    RETURN_ON_ASSETS = "return_on_assets"
    RETURN_ON_EQUITY = "return_on_equity"
    RETURN_ON_CAPITAL_EMPLOYED = "return_on_capital_employed"

    # Leverage / Solvency
    DEBT_TO_EQUITY = "debt_to_equity"
    DEBT_TO_ASSETS = "debt_to_assets"
    DEBT_TO_CAPITAL = "debt_to_capital"
    EQUITY_RATIO = "equity_ratio"
    FINANCIAL_LEVERAGE = "financial_leverage"
    NET_DEBT_TO_EBITDA = "net_debt_to_ebitda"

    # Coverage
    INTEREST_COVERAGE = "interest_coverage"
    DEBT_SERVICE_COVERAGE = "debt_service_coverage"
    CASH_INTEREST_COVERAGE = "cash_interest_coverage"

    # Efficiency
    ASSET_TURNOVER = "asset_turnover"
    FIXED_ASSET_TURNOVER = "fixed_asset_turnover"
    INVENTORY_TURNOVER = "inventory_turnover"
    RECEIVABLE_TURNOVER = "receivable_turnover"
    PAYABLE_TURNOVER = "payable_turnover"

    # Days
    INVENTORY_DAYS = "inventory_days"
    RECEIVABLE_DAYS = "receivable_days"
    PAYABLE_DAYS = "payable_days"
    CASH_CONVERSION_CYCLE = "cash_conversion_cycle"

    # Cash flow
    OPERATING_CASH_FLOW_RATIO = "operating_cash_flow_ratio"
    CASH_FLOW_TO_DEBT = "cash_flow_to_debt"
    FREE_CASH_FLOW_MARGIN = "free_cash_flow_margin"
    CASH_FLOW_COVERAGE = "cash_flow_coverage"


class RatioHealth(str, Enum):
    """Health classification of a ratio."""

    EXCELLENT = "excellent"
    HEALTHY = "healthy"
    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    CRITICAL = "critical"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class RatioTrend(str, Enum):
    """Ratio trend direction."""

    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class RatioSeverity(str, Enum):
    """Risk severity."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Input Model
# ============================================================================


@dataclass(slots=True)
class FinancialRatioInput:
    """
    Normalized financial statement inputs.

    Monetary values should normally be positive magnitudes for
    balance-sheet assets/liabilities and expense/cost fields.

    Profit fields retain their natural sign.
    """

    # Income statement
    revenue: Decimal = ZERO
    cost_of_goods_sold: Decimal = ZERO
    gross_profit: Decimal | None = None

    operating_expenses: Decimal = ZERO
    ebitda: Decimal | None = None
    operating_profit: Decimal | None = None

    interest_expense: Decimal = ZERO
    tax_expense: Decimal = ZERO
    net_profit: Decimal | None = None

    # Balance sheet
    cash: Decimal = ZERO
    cash_equivalents: Decimal = ZERO

    accounts_receivable: Decimal = ZERO
    inventory: Decimal = ZERO
    prepaid_expenses: Decimal = ZERO
    other_current_assets: Decimal = ZERO

    current_assets: Decimal | None = None

    accounts_payable: Decimal = ZERO
    short_term_debt: Decimal = ZERO
    accrued_liabilities: Decimal = ZERO
    other_current_liabilities: Decimal = ZERO

    current_liabilities: Decimal | None = None

    total_assets: Decimal = ZERO
    fixed_assets: Decimal = ZERO

    total_liabilities: Decimal = ZERO
    total_debt: Decimal = ZERO
    long_term_debt: Decimal = ZERO
    equity: Decimal = ZERO

    # Cash flow
    operating_cash_flow: Decimal = ZERO
    investing_cash_flow: Decimal = ZERO
    financing_cash_flow: Decimal = ZERO
    free_cash_flow: Decimal | None = None

    # Debt service
    principal_repayment: Decimal = ZERO
    debt_service: Decimal | None = None

    # Period assumptions
    days_in_period: Decimal = Decimal("365")

    currency: str = "USD"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "FinancialRatioInput":
        """Create input from a generic mapping."""

        def first(*keys: str) -> Any:
            for key in keys:
                if key in data:
                    return data[key]

            return None

        return cls(
            revenue=to_decimal(
                first(
                    "revenue",
                    "sales",
                    "net_sales",
                )
            ),
            cost_of_goods_sold=to_decimal(
                first(
                    "cost_of_goods_sold",
                    "cogs",
                    "cost_of_sales",
                )
            ),
            gross_profit=(
                to_decimal(
                    first("gross_profit")
                )
                if first("gross_profit")
                is not None
                else None
            ),
            operating_expenses=to_decimal(
                first(
                    "operating_expenses",
                    "opex",
                )
            ),
            ebitda=(
                to_decimal(
                    first("ebitda")
                )
                if first("ebitda") is not None
                else None
            ),
            operating_profit=(
                to_decimal(
                    first(
                        "operating_profit",
                        "ebit",
                    )
                )
                if first(
                    "operating_profit",
                    "ebit",
                ) is not None
                else None
            ),
            interest_expense=to_decimal(
                first(
                    "interest_expense",
                    "interest",
                )
            ),
            tax_expense=to_decimal(
                first(
                    "tax_expense",
                    "tax",
                )
            ),
            net_profit=(
                to_decimal(
                    first(
                        "net_profit",
                        "net_income",
                    )
                )
                if first(
                    "net_profit",
                    "net_income",
                ) is not None
                else None
            ),
            cash=to_decimal(
                first("cash")
            ),
            cash_equivalents=to_decimal(
                first(
                    "cash_equivalents",
                    "cash_and_equivalents",
                )
            ),
            accounts_receivable=to_decimal(
                first(
                    "accounts_receivable",
                    "receivables",
                    "ar",
                )
            ),
            inventory=to_decimal(
                first("inventory")
            ),
            prepaid_expenses=to_decimal(
                first("prepaid_expenses")
            ),
            other_current_assets=to_decimal(
                first(
                    "other_current_assets"
                )
            ),
            current_assets=(
                to_decimal(
                    first("current_assets")
                )
                if first("current_assets")
                is not None
                else None
            ),
            accounts_payable=to_decimal(
                first(
                    "accounts_payable",
                    "payables",
                    "ap",
                )
            ),
            short_term_debt=to_decimal(
                first(
                    "short_term_debt",
                    "current_debt",
                )
            ),
            accrued_liabilities=to_decimal(
                first(
                    "accrued_liabilities"
                )
            ),
            other_current_liabilities=to_decimal(
                first(
                    "other_current_liabilities"
                )
            ),
            current_liabilities=(
                to_decimal(
                    first("current_liabilities")
                )
                if first(
                    "current_liabilities"
                ) is not None
                else None
            ),
            total_assets=to_decimal(
                first("total_assets")
            ),
            fixed_assets=to_decimal(
                first(
                    "fixed_assets",
                    "property_plant_equipment",
                    "ppe",
                )
            ),
            total_liabilities=to_decimal(
                first("total_liabilities")
            ),
            total_debt=to_decimal(
                first(
                    "total_debt",
                    "debt",
                )
            ),
            long_term_debt=to_decimal(
                first("long_term_debt")
            ),
            equity=to_decimal(
                first(
                    "equity",
                    "shareholders_equity",
                )
            ),
            operating_cash_flow=to_decimal(
                first(
                    "operating_cash_flow",
                    "cash_from_operations",
                    "cfo",
                )
            ),
            investing_cash_flow=to_decimal(
                first(
                    "investing_cash_flow",
                    "cfi",
                )
            ),
            financing_cash_flow=to_decimal(
                first(
                    "financing_cash_flow",
                    "cff",
                )
            ),
            free_cash_flow=(
                to_decimal(
                    first("free_cash_flow")
                )
                if first("free_cash_flow")
                is not None
                else None
            ),
            principal_repayment=to_decimal(
                first(
                    "principal_repayment",
                    "debt_repayment",
                )
            ),
            debt_service=(
                to_decimal(
                    first("debt_service")
                )
                if first("debt_service")
                is not None
                else None
            ),
            days_in_period=to_decimal(
                first(
                    "days_in_period",
                    "period_days",
                ),
                default=Decimal("365"),
            ),
            currency=str(
                first("currency")
                or "USD"
            ).upper(),
            metadata=dict(
                data.get(
                    "metadata",
                    {},
                )
                or {}
            ),
        )


# ============================================================================
# Ratio Result Models
# ============================================================================


@dataclass(slots=True)
class RatioResult:
    """Single financial ratio result."""

    ratio_type: RatioType
    category: RatioCategory

    value: Decimal | None

    unit: str

    health: RatioHealth

    trend: RatioTrend = RatioTrend.UNKNOWN

    severity: RatioSeverity = (
        RatioSeverity.INFO
    )

    benchmark_low: Decimal | None = None
    benchmark_high: Decimal | None = None

    explanation: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "ratio_type": self.ratio_type.value,
            "category": self.category.value,
            "value": (
                str(self.value)
                if self.value is not None
                else None
            ),
            "unit": self.unit,
            "health": self.health.value,
            "trend": self.trend.value,
            "severity": self.severity.value,
            "benchmark_low": (
                str(self.benchmark_low)
                if self.benchmark_low is not None
                else None
            ),
            "benchmark_high": (
                str(self.benchmark_high)
                if self.benchmark_high is not None
                else None
            ),
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RatioComparison:
    """Comparison of a ratio across periods."""

    ratio_type: RatioType

    current_value: Decimal | None
    previous_value: Decimal | None

    absolute_change: Decimal | None
    percentage_change: Decimal | None

    direction: str
    trend: RatioTrend

    explanation: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "ratio_type": self.ratio_type.value,
            "current_value": (
                str(self.current_value)
                if self.current_value is not None
                else None
            ),
            "previous_value": (
                str(self.previous_value)
                if self.previous_value is not None
                else None
            ),
            "absolute_change": (
                str(self.absolute_change)
                if self.absolute_change is not None
                else None
            ),
            "percentage_change": (
                str(self.percentage_change)
                if self.percentage_change is not None
                else None
            ),
            "direction": self.direction,
            "trend": self.trend.value,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RatioAlert:
    """Risk signal generated from ratio analysis."""

    code: str

    ratio_type: RatioType
    category: RatioCategory

    severity: RatioSeverity

    value: Decimal | None

    threshold: Decimal | None

    message: str

    recommendation: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "code": self.code,
            "ratio_type": self.ratio_type.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "value": (
                str(self.value)
                if self.value is not None
                else None
            ),
            "threshold": (
                str(self.threshold)
                if self.threshold is not None
                else None
            ),
            "message": self.message,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RatioAnalysisResult:
    """Complete financial ratio analysis."""

    ratios: dict[
        RatioType,
        RatioResult,
    ]

    comparisons: list[RatioComparison] = field(
        default_factory=list
    )

    alerts: list[RatioAlert] = field(
        default_factory=list
    )

    overall_health: RatioHealth = (
        RatioHealth.UNKNOWN
    )

    overall_trend: RatioTrend = (
        RatioTrend.UNKNOWN
    )

    summary: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def get(
        self,
        ratio_type: RatioType,
    ) -> RatioResult | None:
        """Return one ratio."""

        return self.ratios.get(
            ratio_type
        )

    def value(
        self,
        ratio_type: RatioType,
    ) -> Decimal | None:
        """Return one ratio value."""

        result = self.get(
            ratio_type
        )

        if result is None:
            return None

        return result.value

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "ratios": {
                ratio_type.value: result.to_dict()
                for ratio_type, result
                in self.ratios.items()
            },
            "comparisons": [
                item.to_dict()
                for item in self.comparisons
            ],
            "alerts": [
                item.to_dict()
                for item in self.alerts
            ],
            "overall_health": (
                self.overall_health.value
            ),
            "overall_trend": (
                self.overall_trend.value
            ),
            "summary": self.summary,
            "metadata": self.metadata,
        }


# ============================================================================
# Benchmark Configuration
# ============================================================================


@dataclass(frozen=True, slots=True)
class RatioBenchmark:
    """Benchmark configuration for a ratio."""

    low: Decimal | None = None
    high: Decimal | None = None

    healthy_min: Decimal | None = None
    healthy_max: Decimal | None = None

    higher_is_better: bool = True


DEFAULT_BENCHMARKS: dict[
    RatioType,
    RatioBenchmark,
] = {
    RatioType.CURRENT_RATIO: RatioBenchmark(
        low=Decimal("1.0"),
        high=Decimal("3.0"),
        healthy_min=Decimal("1.2"),
        healthy_max=Decimal("3.0"),
        higher_is_better=True,
    ),
    RatioType.QUICK_RATIO: RatioBenchmark(
        low=Decimal("0.8"),
        high=Decimal("2.0"),
        healthy_min=Decimal("1.0"),
        healthy_max=Decimal("2.0"),
        higher_is_better=True,
    ),
    RatioType.CASH_RATIO: RatioBenchmark(
        low=Decimal("0.2"),
        high=Decimal("1.0"),
        healthy_min=Decimal("0.3"),
        healthy_max=Decimal("1.0"),
        higher_is_better=True,
    ),
    RatioType.DEBT_TO_EQUITY: RatioBenchmark(
        low=Decimal("0.0"),
        high=Decimal("2.0"),
        healthy_min=Decimal("0.0"),
        healthy_max=Decimal("1.5"),
        higher_is_better=False,
    ),
    RatioType.DEBT_TO_ASSETS: RatioBenchmark(
        low=Decimal("0.0"),
        high=Decimal("0.7"),
        healthy_min=Decimal("0.0"),
        healthy_max=Decimal("0.6"),
        higher_is_better=False,
    ),
    RatioType.INTEREST_COVERAGE: RatioBenchmark(
        low=Decimal("1.5"),
        high=Decimal("20.0"),
        healthy_min=Decimal("2.5"),
        healthy_max=Decimal("20.0"),
        higher_is_better=True,
    ),
    RatioType.NET_DEBT_TO_EBITDA: RatioBenchmark(
        low=Decimal("0.0"),
        high=Decimal("4.0"),
        healthy_min=Decimal("0.0"),
        healthy_max=Decimal("3.0"),
        higher_is_better=False,
    ),
    RatioType.ASSET_TURNOVER: RatioBenchmark(
        low=Decimal("0.5"),
        high=Decimal("5.0"),
        healthy_min=Decimal("0.8"),
        healthy_max=Decimal("5.0"),
        higher_is_better=True,
    ),
    RatioType.INVENTORY_DAYS: RatioBenchmark(
        low=Decimal("10"),
        high=Decimal("120"),
        healthy_min=Decimal("10"),
        healthy_max=Decimal("90"),
        higher_is_better=False,
    ),
    RatioType.RECEIVABLE_DAYS: RatioBenchmark(
        low=Decimal("10"),
        high=Decimal("90"),
        healthy_min=Decimal("10"),
        healthy_max=Decimal("60"),
        higher_is_better=False,
    ),
    RatioType.PAYABLE_DAYS: RatioBenchmark(
        low=Decimal("10"),
        high=Decimal("120"),
        healthy_min=Decimal("20"),
        healthy_max=Decimal("90"),
        higher_is_better=True,
    ),
}


# ============================================================================
# Ratio Analyzer
# ============================================================================


class RatioAnalyzer:
    """
    Comprehensive financial ratio calculator.

    The analyzer is deterministic and database-independent.

    It should be called after financial data has been normalized.
    """

    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        current: (
            FinancialRatioInput
            | Mapping[str, Any]
        ),
        previous: (
            FinancialRatioInput
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> RatioAnalysisResult:

        current_input = self.normalize_input(
            current
        )

        previous_input = (
            self.normalize_input(
                previous
            )
            if previous is not None
            else None
        )

        ratios = self.calculate_all(
            current_input
        )

        comparisons = (
            self.compare(
                current_input,
                previous_input,
                ratios,
            )
            if previous_input is not None
            else []
        )

        self.apply_trends(
            ratios,
            comparisons,
        )

        alerts = self.generate_alerts(
            ratios,
            comparisons,
        )

        overall_health = (
            self.calculate_overall_health(
                ratios
            )
        )

        overall_trend = (
            self.calculate_overall_trend(
                comparisons
            )
        )

        summary = self.generate_summary(
            ratios,
            alerts,
            overall_health,
            overall_trend,
        )

        return RatioAnalysisResult(
            ratios=ratios,
            comparisons=comparisons,
            alerts=alerts,
            overall_health=overall_health,
            overall_trend=overall_trend,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_input(
        self,
        data: (
            FinancialRatioInput
            | Mapping[str, Any]
        ),
    ) -> FinancialRatioInput:

        if isinstance(
            data,
            FinancialRatioInput,
        ):
            return data

        return FinancialRatioInput.from_mapping(
            data
        )

    # ------------------------------------------------------------------
    # Derived Financial Values
    # ------------------------------------------------------------------

    def current_assets(
        self,
        data: FinancialRatioInput,
    ) -> Decimal:

        if data.current_assets is not None:
            return to_decimal(
                data.current_assets
            )

        return (
            to_decimal(data.cash)
            + to_decimal(data.cash_equivalents)
            + to_decimal(data.accounts_receivable)
            + to_decimal(data.inventory)
            + to_decimal(data.prepaid_expenses)
            + to_decimal(data.other_current_assets)
        )

    def current_liabilities(
        self,
        data: FinancialRatioInput,
    ) -> Decimal:

        if data.current_liabilities is not None:
            return to_decimal(
                data.current_liabilities
            )

        return (
            to_decimal(data.accounts_payable)
            + to_decimal(data.short_term_debt)
            + to_decimal(data.accrued_liabilities)
            + to_decimal(
                data.other_current_liabilities
            )
        )

    def gross_profit(
        self,
        data: FinancialRatioInput,
    ) -> Decimal:

        if data.gross_profit is not None:
            return to_decimal(
                data.gross_profit
            )

        return (
            to_decimal(data.revenue)
            - to_decimal(
                data.cost_of_goods_sold
            )
        )

    def operating_profit(
        self,
        data: FinancialRatioInput,
    ) -> Decimal:

        if data.operating_profit is not None:
            return to_decimal(
                data.operating_profit
            )

        return (
            self.gross_profit(data)
            - to_decimal(
                data.operating_expenses
            )
        )

    def ebitda(
        self,
        data: FinancialRatioInput,
    ) -> Decimal:

        if data.ebitda is not None:
            return to_decimal(
                data.ebitda
            )

        return (
            self.operating_profit(data)
            + abs(
                to_decimal(
                    data.depreciation
                )
            )
            + abs(
                to_decimal(
                    data.amortization
                )
            )
        )

    def net_profit(
        self,
        data: FinancialRatioInput,
    ) -> Decimal:

        if data.net_profit is not None:
            return to_decimal(
                data.net_profit
            )

        return (
            self.operating_profit(data)
            - abs(
                to_decimal(
                    data.interest_expense
                )
            )
            - abs(
                to_decimal(
                    data.tax_expense
                )
            )
        )

    def total_debt(
        self,
        data: FinancialRatioInput,
    ) -> Decimal:

        if data.total_debt != ZERO:
            return abs(
                to_decimal(
                    data.total_debt
                )
            )

        return (
            abs(
                to_decimal(
                    data.short_term_debt
                )
            )
            + abs(
                to_decimal(
                    data.long_term_debt
                )
            )
        )

    def free_cash_flow(
        self,
        data: FinancialRatioInput,
    ) -> Decimal:

        if data.free_cash_flow is not None:
            return to_decimal(
                data.free_cash_flow
            )

        # Generic FCF approximation.
        #
        # For precise accounting implementations, this should
        # preferably be supplied by cash_flow.py.
        return (
            to_decimal(
                data.operating_cash_flow
            )
            + to_decimal(
                data.investing_cash_flow
            )
        )

    # ------------------------------------------------------------------
    # Calculate All
    # ------------------------------------------------------------------

    def calculate_all(
        self,
        data: FinancialRatioInput,
    ) -> dict[
        RatioType,
        RatioResult,
    ]:

        ratios: dict[
            RatioType,
            RatioResult,
        ] = {}

        ratios.update(
            self.calculate_liquidity_ratios(
                data
            )
        )

        ratios.update(
            self.calculate_profitability_ratios(
                data
            )
        )

        ratios.update(
            self.calculate_leverage_ratios(
                data
            )
        )

        ratios.update(
            self.calculate_coverage_ratios(
                data
            )
        )

        ratios.update(
            self.calculate_efficiency_ratios(
                data
            )
        )

        ratios.update(
            self.calculate_cash_flow_ratios(
                data
            )
        )

        return ratios

    # ------------------------------------------------------------------
    # Liquidity Ratios
    # ------------------------------------------------------------------

    def calculate_liquidity_ratios(
        self,
        data: FinancialRatioInput,
    ) -> dict[
        RatioType,
        RatioResult,
    ]:

        current_assets = (
            self.current_assets(data)
        )

        current_liabilities = (
            self.current_liabilities(data)
        )

        liquid_assets = (
            current_assets
            - abs(
                to_decimal(
                    data.inventory
                )
            )
            - abs(
                to_decimal(
                    data.prepaid_expenses
                )
            )
        )

        cash = (
            abs(
                to_decimal(data.cash)
            )
            + abs(
                to_decimal(
                    data.cash_equivalents
                )
            )
        )

        working_capital = (
            current_assets
            - current_liabilities
        )

        return {
            RatioType.CURRENT_RATIO: self.make_ratio(
                RatioType.CURRENT_RATIO,
                RatioCategory.LIQUIDITY,
                calculate_ratio(
                    current_assets,
                    current_liabilities,
                ),
                unit="x",
            ),
            RatioType.QUICK_RATIO: self.make_ratio(
                RatioType.QUICK_RATIO,
                RatioCategory.LIQUIDITY,
                calculate_ratio(
                    liquid_assets,
                    current_liabilities,
                ),
                unit="x",
            ),
            RatioType.CASH_RATIO: self.make_ratio(
                RatioType.CASH_RATIO,
                RatioCategory.LIQUIDITY,
                calculate_ratio(
                    cash,
                    current_liabilities,
                ),
                unit="x",
            ),
            RatioType.WORKING_CAPITAL: self.make_ratio(
                RatioType.WORKING_CAPITAL,
                RatioCategory.WORKING_CAPITAL,
                round_money(
                    working_capital
                ),
                unit="currency",
            ),
            RatioType.WORKING_CAPITAL_TO_REVENUE: self.make_ratio(
                RatioType.WORKING_CAPITAL_TO_REVENUE,
                RatioCategory.WORKING_CAPITAL,
                calculate_percentage(
                    working_capital,
                    to_decimal(
                        data.revenue
                    ),
                ),
                unit="%",
            ),
        }

    # ------------------------------------------------------------------
    # Profitability Ratios
    # ------------------------------------------------------------------

    def calculate_profitability_ratios(
        self,
        data: FinancialRatioInput,
    ) -> dict[
        RatioType,
        RatioResult,
    ]:

        revenue = to_decimal(
            data.revenue
        )

        gross_profit = (
            self.gross_profit(data)
        )

        operating_profit = (
            self.operating_profit(data)
        )

        ebitda = self.ebitda(data)
        net_profit = self.net_profit(data)

        assets = abs(
            to_decimal(
                data.total_assets
            )
        )

        equity = abs(
            to_decimal(
                data.equity
            )
        )

        debt = self.total_debt(data)

        capital_employed = (
            equity + debt
        )

        return {
            RatioType.GROSS_MARGIN: self.make_ratio(
                RatioType.GROSS_MARGIN,
                RatioCategory.PROFITABILITY,
                calculate_percentage(
                    gross_profit,
                    revenue,
                ),
                unit="%",
            ),
            RatioType.EBITDA_MARGIN: self.make_ratio(
                RatioType.EBITDA_MARGIN,
                RatioCategory.PROFITABILITY,
                calculate_percentage(
                    ebitda,
                    revenue,
                ),
                unit="%",
            ),
            RatioType.OPERATING_MARGIN: self.make_ratio(
                RatioType.OPERATING_MARGIN,
                RatioCategory.PROFITABILITY,
                calculate_percentage(
                    operating_profit,
                    revenue,
                ),
                unit="%",
            ),
            RatioType.NET_MARGIN: self.make_ratio(
                RatioType.NET_MARGIN,
                RatioCategory.PROFITABILITY,
                calculate_percentage(
                    net_profit,
                    revenue,
                ),
                unit="%",
            ),
            RatioType.RETURN_ON_ASSETS: self.make_ratio(
                RatioType.RETURN_ON_ASSETS,
                RatioCategory.PROFITABILITY,
                calculate_percentage(
                    net_profit,
                    assets,
                ),
                unit="%",
            ),
            RatioType.RETURN_ON_EQUITY: self.make_ratio(
                RatioType.RETURN_ON_EQUITY,
                RatioCategory.PROFITABILITY,
                calculate_percentage(
                    net_profit,
                    equity,
                ),
                unit="%",
            ),
            RatioType.RETURN_ON_CAPITAL_EMPLOYED: self.make_ratio(
                RatioType.RETURN_ON_CAPITAL_EMPLOYED,
                RatioCategory.PROFITABILITY,
                calculate_percentage(
                    operating_profit,
                    capital_employed,
                ),
                unit="%",
            ),
        }

    # ------------------------------------------------------------------
    # Leverage / Solvency Ratios
    # ------------------------------------------------------------------

    def calculate_leverage_ratios(
        self,
        data: FinancialRatioInput,
    ) -> dict[
        RatioType,
        RatioResult,
    ]:

        debt = self.total_debt(data)

        liabilities = abs(
            to_decimal(
                data.total_liabilities
            )
        )

        assets = abs(
            to_decimal(
                data.total_assets
            )
        )

        equity = abs(
            to_decimal(
                data.equity
            )
        )

        cash = (
            abs(
                to_decimal(data.cash)
            )
            + abs(
                to_decimal(
                    data.cash_equivalents
                )
            )
        )

        ebitda = self.ebitda(data)

        debt_to_equity = calculate_ratio(
            debt,
            equity,
        )

        debt_to_assets = calculate_percentage(
            debt,
            assets,
        )

        capital = (
            debt + equity
        )

        debt_to_capital = calculate_percentage(
            debt,
            capital,
        )

        equity_ratio = calculate_percentage(
            equity,
            assets,
        )

        financial_leverage = calculate_ratio(
            assets,
            equity,
        )

        net_debt = debt - cash

        net_debt_to_ebitda = (
            calculate_ratio(
                net_debt,
                ebitda,
            )
        )

        return {
            RatioType.DEBT_TO_EQUITY: self.make_ratio(
                RatioType.DEBT_TO_EQUITY,
                RatioCategory.LEVERAGE,
                debt_to_equity,
                unit="x",
            ),
            RatioType.DEBT_TO_ASSETS: self.make_ratio(
                RatioType.DEBT_TO_ASSETS,
                RatioCategory.SOLVENCY,
                debt_to_assets,
                unit="%",
            ),
            RatioType.DEBT_TO_CAPITAL: self.make_ratio(
                RatioType.DEBT_TO_CAPITAL,
                RatioCategory.LEVERAGE,
                debt_to_capital,
                unit="%",
            ),
            RatioType.EQUITY_RATIO: self.make_ratio(
                RatioType.EQUITY_RATIO,
                RatioCategory.SOLVENCY,
                equity_ratio,
                unit="%",
            ),
            RatioType.FINANCIAL_LEVERAGE: self.make_ratio(
                RatioType.FINANCIAL_LEVERAGE,
                RatioCategory.LEVERAGE,
                financial_leverage,
                unit="x",
            ),
            RatioType.NET_DEBT_TO_EBITDA: self.make_ratio(
                RatioType.NET_DEBT_TO_EBITDA,
                RatioCategory.LEVERAGE,
                net_debt_to_ebitda,
                unit="x",
            ),
        }

    # ------------------------------------------------------------------
    # Coverage Ratios
    # ------------------------------------------------------------------

    def calculate_coverage_ratios(
        self,
        data: FinancialRatioInput,
    ) -> dict[
        RatioType,
        RatioResult,
    ]:

        operating_profit = (
            self.operating_profit(data)
        )

        ebitda = self.ebitda(data)

        interest = abs(
            to_decimal(
                data.interest_expense
            )
        )

        operating_cash_flow = (
            to_decimal(
                data.operating_cash_flow
            )
        )

        debt_service = (
            to_decimal(
                data.debt_service
            )
            if data.debt_service is not None
            else (
                interest
                + abs(
                    to_decimal(
                        data.principal_repayment
                    )
                )
            )
        )

        interest_coverage = calculate_ratio(
            operating_profit,
            interest,
        )

        cash_interest_coverage = calculate_ratio(
            operating_cash_flow,
            interest,
        )

        debt_service_coverage = calculate_ratio(
            operating_cash_flow,
            debt_service,
        )

        return {
            RatioType.INTEREST_COVERAGE: self.make_ratio(
                RatioType.INTEREST_COVERAGE,
                RatioCategory.COVERAGE,
                interest_coverage,
                unit="x",
            ),
            RatioType.DEBT_SERVICE_COVERAGE: self.make_ratio(
                RatioType.DEBT_SERVICE_COVERAGE,
                RatioCategory.COVERAGE,
                debt_service_coverage,
                unit="x",
            ),
            RatioType.CASH_INTEREST_COVERAGE: self.make_ratio(
                RatioType.CASH_INTEREST_COVERAGE,
                RatioCategory.COVERAGE,
                cash_interest_coverage,
                unit="x",
            ),
        }

    # ------------------------------------------------------------------
    # Efficiency Ratios
    # ------------------------------------------------------------------

    def calculate_efficiency_ratios(
        self,
        data: FinancialRatioInput,
    ) -> dict[
        RatioType,
        RatioResult,
    ]:

        revenue = abs(
            to_decimal(
                data.revenue
            )
        )

        cogs = abs(
            to_decimal(
                data.cost_of_goods_sold
            )
        )

        assets = abs(
            to_decimal(
                data.total_assets
            )
        )

        fixed_assets = abs(
            to_decimal(
                data.fixed_assets
            )
        )

        inventory = abs(
            to_decimal(
                data.inventory
            )
        )

        receivables = abs(
            to_decimal(
                data.accounts_receivable
            )
        )

        payables = abs(
            to_decimal(
                data.accounts_payable
            )
        )

        days = (
            to_decimal(
                data.days_in_period
            )
        )

        asset_turnover = calculate_ratio(
            revenue,
            assets,
        )

        fixed_asset_turnover = calculate_ratio(
            revenue,
            fixed_assets,
        )

        inventory_turnover = calculate_ratio(
            cogs,
            inventory,
        )

        receivable_turnover = calculate_ratio(
            revenue,
            receivables,
        )

        payable_turnover = calculate_ratio(
            cogs,
            payables,
        )

        inventory_days = calculate_ratio(
            days,
            inventory_turnover,
        )

        receivable_days = calculate_ratio(
            days,
            receivable_turnover,
        )

        payable_days = calculate_ratio(
            days,
            payable_turnover,
        )

        cash_conversion_cycle = (
            inventory_days
            + receivable_days
            - payable_days
        )

        return {
            RatioType.ASSET_TURNOVER: self.make_ratio(
                RatioType.ASSET_TURNOVER,
                RatioCategory.EFFICIENCY,
                asset_turnover,
                unit="x",
            ),
            RatioType.FIXED_ASSET_TURNOVER: self.make_ratio(
                RatioType.FIXED_ASSET_TURNOVER,
                RatioCategory.EFFICIENCY,
                fixed_asset_turnover,
                unit="x",
            ),
            RatioType.INVENTORY_TURNOVER: self.make_ratio(
                RatioType.INVENTORY_TURNOVER,
                RatioCategory.EFFICIENCY,
                inventory_turnover,
                unit="x",
            ),
            RatioType.RECEIVABLE_TURNOVER: self.make_ratio(
                RatioType.RECEIVABLE_TURNOVER,
                RatioCategory.EFFICIENCY,
                receivable_turnover,
                unit="x",
            ),
            RatioType.PAYABLE_TURNOVER: self.make_ratio(
                RatioType.PAYABLE_TURNOVER,
                RatioCategory.EFFICIENCY,
                payable_turnover,
                unit="x",
            ),
            RatioType.INVENTORY_DAYS: self.make_ratio(
                RatioType.INVENTORY_DAYS,
                RatioCategory.EFFICIENCY,
                inventory_days,
                unit="days",
            ),
            RatioType.RECEIVABLE_DAYS: self.make_ratio(
                RatioType.RECEIVABLE_DAYS,
                RatioCategory.EFFICIENCY,
                receivable_days,
                unit="days",
            ),
            RatioType.PAYABLE_DAYS: self.make_ratio(
                RatioType.PAYABLE_DAYS,
                RatioCategory.EFFICIENCY,
                payable_days,
                unit="days",
            ),
            RatioType.CASH_CONVERSION_CYCLE: self.make_ratio(
                RatioType.CASH_CONVERSION_CYCLE,
                RatioCategory.WORKING_CAPITAL,
                round_ratio(
                    cash_conversion_cycle
                ),
                unit="days",
            ),
        }

    # ------------------------------------------------------------------
    # Cash Flow Ratios
    # ------------------------------------------------------------------

    def calculate_cash_flow_ratios(
        self,
        data: FinancialRatioInput,
    ) -> dict[
        RatioType,
        RatioResult,
    ]:

        current_liabilities = (
            self.current_liabilities(data)
        )

        total_debt = (
            self.total_debt(data)
        )

        operating_cash_flow = (
            to_decimal(
                data.operating_cash_flow
            )
        )

        free_cash_flow = (
            self.free_cash_flow(data)
        )

        revenue = (
            to_decimal(
                data.revenue
            )
        )

        cash_flow_coverage = (
            calculate_ratio(
                operating_cash_flow,
                current_liabilities,
            )
        )

        operating_cash_flow_ratio = (
            calculate_ratio(
                operating_cash_flow,
                current_liabilities,
            )
        )

        cash_flow_to_debt = (
            calculate_percentage(
                operating_cash_flow,
                total_debt,
            )
        )

        free_cash_flow_margin = (
            calculate_percentage(
                free_cash_flow,
                revenue,
            )
        )

        return {
            RatioType.OPERATING_CASH_FLOW_RATIO: self.make_ratio(
                RatioType.OPERATING_CASH_FLOW_RATIO,
                RatioCategory.CASH_FLOW,
                operating_cash_flow_ratio,
                unit="x",
            ),
            RatioType.CASH_FLOW_TO_DEBT: self.make_ratio(
                RatioType.CASH_FLOW_TO_DEBT,
                RatioCategory.CASH_FLOW,
                cash_flow_to_debt,
                unit="%",
            ),
            RatioType.FREE_CASH_FLOW_MARGIN: self.make_ratio(
                RatioType.FREE_CASH_FLOW_MARGIN,
                RatioCategory.CASH_FLOW,
                free_cash_flow_margin,
                unit="%",
            ),
            RatioType.CASH_FLOW_COVERAGE: self.make_ratio(
                RatioType.CASH_FLOW_COVERAGE,
                RatioCategory.COVERAGE,
                cash_flow_coverage,
                unit="x",
            ),
        }

    # ------------------------------------------------------------------
    # Ratio Result Construction
    # ------------------------------------------------------------------

    def make_ratio(
        self,
        ratio_type: RatioType,
        category: RatioCategory,
        value: Decimal,
        *,
        unit: str,
    ) -> RatioResult:

        benchmark = DEFAULT_BENCHMARKS.get(
            ratio_type
        )

        health = self.classify_health(
            ratio_type,
            value,
        )

        severity = self.classify_severity(
            ratio_type,
            value,
        )

        explanation = self.explain_ratio(
            ratio_type,
            value,
            health,
        )

        return RatioResult(
            ratio_type=ratio_type,
            category=category,
            value=value,
            unit=unit,
            health=health,
            severity=severity,
            benchmark_low=(
                benchmark.low
                if benchmark
                else None
            ),
            benchmark_high=(
                benchmark.high
                if benchmark
                else None
            ),
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Health Classification
    # ------------------------------------------------------------------

    def classify_health(
        self,
        ratio_type: RatioType,
        value: Decimal | None,
    ) -> RatioHealth:

        if value is None:
            return RatioHealth.UNKNOWN

        if ratio_type in {
            RatioType.CURRENT_RATIO,
            RatioType.QUICK_RATIO,
            RatioType.CASH_RATIO,
        }:

            if value <= ZERO:
                return RatioHealth.CRITICAL

            if value < Decimal("0.75"):
                return RatioHealth.CRITICAL

            if value < Decimal("1.0"):
                return RatioHealth.WARNING

            if value <= Decimal("3.0"):
                return RatioHealth.HEALTHY

            return RatioHealth.ACCEPTABLE

        if ratio_type in {
            RatioType.DEBT_TO_EQUITY,
            RatioType.NET_DEBT_TO_EBITDA,
        }:

            if value < ZERO:
                return RatioHealth.UNKNOWN

            if ratio_type == (
                RatioType.DEBT_TO_EQUITY
            ):
                if value > Decimal("3"):
                    return RatioHealth.CRITICAL

                if value > Decimal("2"):
                    return RatioHealth.WARNING

                if value <= Decimal("1.5"):
                    return RatioHealth.HEALTHY

                return RatioHealth.ACCEPTABLE

            if value > Decimal("5"):
                return RatioHealth.CRITICAL

            if value > Decimal("3"):
                return RatioHealth.WARNING

            return RatioHealth.HEALTHY

        if ratio_type in {
            RatioType.DEBT_TO_ASSETS,
            RatioType.DEBT_TO_CAPITAL,
        }:

            if value > Decimal("80"):
                return RatioHealth.CRITICAL

            if value > Decimal("60"):
                return RatioHealth.WARNING

            if value <= Decimal("40"):
                return RatioHealth.HEALTHY

            return RatioHealth.ACCEPTABLE

        if ratio_type in {
            RatioType.INTEREST_COVERAGE,
            RatioType.DEBT_SERVICE_COVERAGE,
            RatioType.CASH_INTEREST_COVERAGE,
            RatioType.CASH_FLOW_COVERAGE,
        }:

            if value <= ZERO:
                return RatioHealth.CRITICAL

            if value < Decimal("1"):
                return RatioHealth.CRITICAL

            if value < Decimal("1.5"):
                return RatioHealth.WARNING

            if value >= Decimal("2.5"):
                return RatioHealth.HEALTHY

            return RatioHealth.ACCEPTABLE

        if ratio_type in {
            RatioType.GROSS_MARGIN,
            RatioType.EBITDA_MARGIN,
            RatioType.OPERATING_MARGIN,
            RatioType.NET_MARGIN,
            RatioType.RETURN_ON_ASSETS,
            RatioType.RETURN_ON_EQUITY,
            RatioType.RETURN_ON_CAPITAL_EMPLOYED,
            RatioType.FREE_CASH_FLOW_MARGIN,
        }:

            if value < ZERO:
                return RatioHealth.NEGATIVE

            if value < Decimal("5"):
                return RatioHealth.WARNING

            if value < Decimal("10"):
                return RatioHealth.ACCEPTABLE

            if value >= Decimal("20"):
                return RatioHealth.EXCELLENT

            return RatioHealth.HEALTHY

        if ratio_type in {
            RatioType.INVENTORY_DAYS,
            RatioType.RECEIVABLE_DAYS,
        }:

            if value > Decimal("120"):
                return RatioHealth.CRITICAL

            if value > Decimal("90"):
                return RatioHealth.WARNING

            if value <= Decimal("60"):
                return RatioHealth.HEALTHY

            return RatioHealth.ACCEPTABLE

        if ratio_type == (
            RatioType.CASH_CONVERSION_CYCLE
        ):

            if value > Decimal("120"):
                return RatioHealth.CRITICAL

            if value > Decimal("90"):
                return RatioHealth.WARNING

            if value <= Decimal("60"):
                return RatioHealth.HEALTHY

            return RatioHealth.ACCEPTABLE

        if ratio_type in {
            RatioType.ASSET_TURNOVER,
            RatioType.FIXED_ASSET_TURNOVER,
            RatioType.INVENTORY_TURNOVER,
            RatioType.RECEIVABLE_TURNOVER,
        }:

            if value <= ZERO:
                return RatioHealth.CRITICAL

            if value < Decimal("0.5"):
                return RatioHealth.WARNING

            if value >= Decimal("1"):
                return RatioHealth.HEALTHY

            return RatioHealth.ACCEPTABLE

        return RatioHealth.UNKNOWN

    # ------------------------------------------------------------------
    # Severity
    # ------------------------------------------------------------------

    def classify_severity(
        self,
        ratio_type: RatioType,
        value: Decimal | None,
    ) -> RatioSeverity:

        if value is None:
            return RatioSeverity.INFO

        health = self.classify_health(
            ratio_type,
            value,
        )

        mapping = {
            RatioHealth.EXCELLENT:
                RatioSeverity.INFO,
            RatioHealth.HEALTHY:
                RatioSeverity.INFO,
            RatioHealth.ACCEPTABLE:
                RatioSeverity.LOW,
            RatioHealth.WARNING:
                RatioSeverity.MEDIUM,
            RatioHealth.CRITICAL:
                RatioSeverity.HIGH,
            RatioHealth.NEGATIVE:
                RatioSeverity.CRITICAL,
            RatioHealth.UNKNOWN:
                RatioSeverity.INFO,
        }

        return mapping[health]

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_ratio(
        self,
        ratio_type: RatioType,
        value: Decimal | None,
        health: RatioHealth,
    ) -> str:

        if value is None:
            return (
                f"{ratio_type.value} could not be calculated."
            )

        if health == RatioHealth.CRITICAL:
            return (
                f"{ratio_type.value} is at a critical level "
                f"({value})."
            )

        if health == RatioHealth.WARNING:
            return (
                f"{ratio_type.value} requires attention "
                f"({value})."
            )

        if health == RatioHealth.HEALTHY:
            return (
                f"{ratio_type.value} is within a generally "
                f"healthy range ({value})."
            )

        if health == RatioHealth.EXCELLENT:
            return (
                f"{ratio_type.value} is strong ({value})."
            )

        if health == RatioHealth.NEGATIVE:
            return (
                f"{ratio_type.value} is negative ({value})."
            )

        return (
            f"{ratio_type.value} is {value}."
        )

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def compare(
        self,
        current: FinancialRatioInput,
        previous: FinancialRatioInput,
        current_ratios: dict[
            RatioType,
            RatioResult,
        ] | None = None,
    ) -> list[RatioComparison]:

        current_ratios = (
            current_ratios
            or self.calculate_all(
                current
            )
        )

        previous_ratios = (
            self.calculate_all(
                previous
            )
        )

        comparisons: list[
            RatioComparison
        ] = []

        for ratio_type, current_result in (
            current_ratios.items()
        ):

            previous_result = (
                previous_ratios.get(
                    ratio_type
                )
            )

            if previous_result is None:
                continue

            current_value = (
                current_result.value
            )

            previous_value = (
                previous_result.value
            )

            if (
                current_value is None
                or previous_value is None
            ):
                continue

            absolute_change = (
                current_value
                - previous_value
            )

            pct_change = percentage_change(
                current_value,
                previous_value,
            )

            direction = (
                "increase"
                if absolute_change > ZERO
                else (
                    "decrease"
                    if absolute_change < ZERO
                    else "unchanged"
                )
            )

            trend = self.determine_ratio_trend(
                ratio_type,
                current_value,
                previous_value,
            )

            explanation = (
                self.explain_comparison(
                    ratio_type,
                    current_value,
                    previous_value,
                    trend,
                )
            )

            comparisons.append(
                RatioComparison(
                    ratio_type=ratio_type,
                    current_value=current_value,
                    previous_value=previous_value,
                    absolute_change=(
                        round_ratio(
                            absolute_change
                        )
                    ),
                    percentage_change=(
                        pct_change
                    ),
                    direction=direction,
                    trend=trend,
                    explanation=explanation,
                )
            )

        return comparisons

    def determine_ratio_trend(
        self,
        ratio_type: RatioType,
        current: Decimal,
        previous: Decimal,
    ) -> RatioTrend:

        change = (
            current - previous
        )

        if change == ZERO:
            return RatioTrend.STABLE

        # Ratios where lower is generally better.
        lower_is_better = {
            RatioType.DEBT_TO_EQUITY,
            RatioType.DEBT_TO_ASSETS,
            RatioType.DEBT_TO_CAPITAL,
            RatioType.NET_DEBT_TO_EBITDA,
            RatioType.INVENTORY_DAYS,
            RatioType.RECEIVABLE_DAYS,
            RatioType.CASH_CONVERSION_CYCLE,
        }

        if ratio_type in lower_is_better:
            return (
                RatioTrend.IMPROVING
                if change < ZERO
                else RatioTrend.DETERIORATING
            )

        return (
            RatioTrend.IMPROVING
            if change > ZERO
            else RatioTrend.DETERIORATING
        )

    def explain_comparison(
        self,
        ratio_type: RatioType,
        current: Decimal,
        previous: Decimal,
        trend: RatioTrend,
    ) -> str:

        change = (
            current - previous
        )

        return (
            f"{ratio_type.value} changed from "
            f"{previous} to {current}, "
            f"a {abs(change)} point/unit change. "
            f"The resulting trend is "
            f"{trend.value}."
        )

    # ------------------------------------------------------------------
    # Apply Trends
    # ------------------------------------------------------------------

    def apply_trends(
        self,
        ratios: dict[
            RatioType,
            RatioResult,
        ],
        comparisons: Sequence[
            RatioComparison
        ],
    ) -> None:

        comparison_map = {
            item.ratio_type: item
            for item in comparisons
        }

        for ratio_type, result in ratios.items():

            comparison = (
                comparison_map.get(
                    ratio_type
                )
            )

            if comparison is None:
                continue

            result.trend = (
                comparison.trend
            )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def generate_alerts(
        self,
        ratios: dict[
            RatioType,
            RatioResult,
        ],
        comparisons: Sequence[
            RatioComparison
        ],
    ) -> list[RatioAlert]:

        alerts: list[RatioAlert] = []

        for ratio_type, result in ratios.items():

            if result.value is None:
                continue

            if result.health == (
                RatioHealth.CRITICAL
            ):
                alerts.append(
                    self.create_alert(
                        ratio_type,
                        result,
                        severity=(
                            RatioSeverity.CRITICAL
                        ),
                    )
                )

            elif result.health == (
                RatioHealth.WARNING
            ):
                alerts.append(
                    self.create_alert(
                        ratio_type,
                        result,
                        severity=(
                            RatioSeverity.MEDIUM
                        ),
                    )
                )

        comparison_map = {
            item.ratio_type: item
            for item in comparisons
        }

        for ratio_type, comparison in (
            comparison_map.items()
        ):

            if comparison.trend != (
                RatioTrend.DETERIORATING
            ):
                continue

            if (
                comparison.percentage_change
                is None
            ):
                continue

            if abs(
                comparison.percentage_change
            ) >= Decimal("20"):

                result = ratios.get(
                    ratio_type
                )

                if result is None:
                    continue

                alerts.append(
                    RatioAlert(
                        code=(
                            "RATIO_DETERIORATION"
                        ),
                        ratio_type=ratio_type,
                        category=result.category,
                        severity=(
                            RatioSeverity.HIGH
                        ),
                        value=result.value,
                        threshold=(
                            comparison.percentage_change
                        ),
                        message=(
                            f"{ratio_type.value} has "
                            f"deteriorated materially "
                            f"versus the previous period."
                        ),
                        recommendation=(
                            f"Investigate the drivers "
                            f"behind the deterioration "
                            f"in {ratio_type.value}."
                        ),
                    )
                )

        return self.deduplicate_alerts(
            alerts
        )

    def create_alert(
        self,
        ratio_type: RatioType,
        result: RatioResult,
        *,
        severity: RatioSeverity,
    ) -> RatioAlert:

        recommendations = {
            RatioType.CURRENT_RATIO: (
                "Review short-term liquidity, "
                "working capital, receivables, "
                "and near-term obligations."
            ),
            RatioType.QUICK_RATIO: (
                "Improve liquid asset coverage "
                "or reduce short-term liabilities."
            ),
            RatioType.CASH_RATIO: (
                "Review cash reserves and "
                "near-term cash obligations."
            ),
            RatioType.DEBT_TO_EQUITY: (
                "Review leverage, debt repayment "
                "capacity, and capital structure."
            ),
            RatioType.DEBT_TO_ASSETS: (
                "Assess debt dependence and "
                "balance-sheet solvency."
            ),
            RatioType.NET_DEBT_TO_EBITDA: (
                "Review debt reduction and "
                "operating cash generation."
            ),
            RatioType.INTEREST_COVERAGE: (
                "Review interest burden, debt "
                "structure, and operating profit."
            ),
            RatioType.DEBT_SERVICE_COVERAGE: (
                "Review debt repayment capacity "
                "and operating cash flow."
            ),
            RatioType.INVENTORY_DAYS: (
                "Investigate excess inventory, "
                "slow-moving stock, and purchasing."
            ),
            RatioType.RECEIVABLE_DAYS: (
                "Review customer collections, "
                "credit policy, and overdue invoices."
            ),
            RatioType.CASH_CONVERSION_CYCLE: (
                "Improve working-capital efficiency "
                "across inventory, receivables, "
                "and payables."
            ),
        }

        recommendation = recommendations.get(
            ratio_type,
            (
                f"Investigate the drivers behind "
                f"{ratio_type.value}."
            ),
        )

        return RatioAlert(
            code=(
                f"{ratio_type.value.upper()}_RISK"
            ),
            ratio_type=ratio_type,
            category=result.category,
            severity=severity,
            value=result.value,
            threshold=result.benchmark_low,
            message=(
                f"{ratio_type.value} is showing "
                f"{result.health.value} conditions."
            ),
            recommendation=recommendation,
        )

    @staticmethod
    def deduplicate_alerts(
        alerts: Sequence[RatioAlert],
    ) -> list[RatioAlert]:

        seen: set[
            tuple[str, str]
        ] = set()

        result: list[
            RatioAlert
        ] = []

        for alert in alerts:

            key = (
                alert.code,
                alert.ratio_type.value,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(alert)

        return result

    # ------------------------------------------------------------------
    # Overall Health
    # ------------------------------------------------------------------

    def calculate_overall_health(
        self,
        ratios: Mapping[
            RatioType,
            RatioResult,
        ],
    ) -> RatioHealth:

        if not ratios:
            return RatioHealth.UNKNOWN

        scores = {
            RatioHealth.EXCELLENT: 5,
            RatioHealth.HEALTHY: 4,
            RatioHealth.ACCEPTABLE: 3,
            RatioHealth.WARNING: 2,
            RatioHealth.CRITICAL: 1,
            RatioHealth.NEGATIVE: 1,
            RatioHealth.UNKNOWN: 0,
        }

        weighted_categories = {
            RatioCategory.LIQUIDITY: Decimal("0.20"),
            RatioCategory.PROFITABILITY: Decimal("0.25"),
            RatioCategory.LEVERAGE: Decimal("0.20"),
            RatioCategory.SOLVENCY: Decimal("0.15"),
            RatioCategory.EFFICIENCY: Decimal("0.10"),
            RatioCategory.COVERAGE: Decimal("0.10"),
        }

        total_score = ZERO
        total_weight = ZERO

        for result in ratios.values():

            weight = weighted_categories.get(
                result.category,
                Decimal("0.05"),
            )

            score = Decimal(
                scores[result.health]
            )

            total_score += (
                score * weight
            )

            total_weight += weight

        if total_weight == ZERO:
            return RatioHealth.UNKNOWN

        average = (
            total_score
            / total_weight
        )

        if average >= Decimal("4.5"):
            return RatioHealth.EXCELLENT

        if average >= Decimal("3.5"):
            return RatioHealth.HEALTHY

        if average >= Decimal("2.5"):
            return RatioHealth.ACCEPTABLE

        if average >= Decimal("1.5"):
            return RatioHealth.WARNING

        return RatioHealth.CRITICAL

    # ------------------------------------------------------------------
    # Overall Trend
    # ------------------------------------------------------------------

    def calculate_overall_trend(
        self,
        comparisons: Sequence[
            RatioComparison
        ],
    ) -> RatioTrend:

        if not comparisons:
            return RatioTrend.UNKNOWN

        improving = sum(
            1
            for item in comparisons
            if item.trend
            == RatioTrend.IMPROVING
        )

        deteriorating = sum(
            1
            for item in comparisons
            if item.trend
            == RatioTrend.DETERIORATING
        )

        total = (
            improving
            + deteriorating
        )

        if total == 0:
            return RatioTrend.STABLE

        if improving >= (
            deteriorating * 2
        ):
            return RatioTrend.IMPROVING

        if deteriorating >= (
            improving * 2
        ):
            return RatioTrend.DETERIORATING

        if improving > deteriorating:
            return RatioTrend.IMPROVING

        if deteriorating > improving:
            return RatioTrend.DETERIORATING

        return RatioTrend.STABLE

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_summary(
        self,
        ratios: Mapping[
            RatioType,
            RatioResult,
        ],
        alerts: Sequence[
            RatioAlert
        ],
        overall_health: RatioHealth,
        overall_trend: RatioTrend,
    ) -> str:

        summary = (
            f"Financial ratio analysis indicates "
            f"an overall {overall_health.value} "
            f"condition with a "
            f"{overall_trend.value} trend."
        )

        if alerts:
            critical_count = sum(
                1
                for alert in alerts
                if alert.severity
                == RatioSeverity.CRITICAL
            )

            high_count = sum(
                1
                for alert in alerts
                if alert.severity
                == RatioSeverity.HIGH
            )

            if critical_count:
                summary += (
                    f" {critical_count} critical "
                    f"ratio risk(s) were detected."
                )
            elif high_count:
                summary += (
                    f" {high_count} high-priority "
                    f"ratio risk(s) were detected."
                )

        current_ratio = ratios.get(
            RatioType.CURRENT_RATIO
        )

        net_margin = ratios.get(
            RatioType.NET_MARGIN
        )

        debt_to_equity = ratios.get(
            RatioType.DEBT_TO_EQUITY
        )

        if current_ratio is not None:
            summary += (
                f" Current ratio is "
                f"{current_ratio.value}."
            )

        if net_margin is not None:
            summary += (
                f" Net margin is "
                f"{net_margin.value}%."
            )

        if debt_to_equity is not None:
            summary += (
                f" Debt-to-equity is "
                f"{debt_to_equity.value}x."
            )

        return summary


# ============================================================================
# Convenience Functions
# ============================================================================


def analyze_ratios(
    current: (
        FinancialRatioInput
        | Mapping[str, Any]
    ),
    previous: (
        FinancialRatioInput
        | Mapping[str, Any]
        | None
    ) = None,
) -> RatioAnalysisResult:
    """Perform complete ratio analysis."""

    return RatioAnalyzer().analyze(
        current,
        previous,
    )


def calculate_all_ratios(
    data: (
        FinancialRatioInput
        | Mapping[str, Any]
    ),
) -> dict[
    RatioType,
    RatioResult,
]:
    """Calculate all supported financial ratios."""

    analyzer = RatioAnalyzer()

    normalized = (
        analyzer.normalize_input(
            data
        )
    )

    return analyzer.calculate_all(
        normalized
    )


def calculate_current_ratio(
    current_assets: Any,
    current_liabilities: Any,
) -> Decimal:
    """Calculate current ratio."""

    return calculate_ratio(
        to_decimal(current_assets),
        to_decimal(current_liabilities),
    )


def calculate_quick_ratio(
    current_assets: Any,
    inventory: Any,
    prepaid_expenses: Any,
    current_liabilities: Any,
) -> Decimal:
    """Calculate quick ratio."""

    liquid_assets = (
        to_decimal(current_assets)
        - to_decimal(inventory)
        - to_decimal(prepaid_expenses)
    )

    return calculate_ratio(
        liquid_assets,
        to_decimal(current_liabilities),
    )


def calculate_cash_ratio(
    cash: Any,
    current_liabilities: Any,
) -> Decimal:
    """Calculate cash ratio."""

    return calculate_ratio(
        to_decimal(cash),
        to_decimal(current_liabilities),
    )


def calculate_debt_to_equity(
    debt: Any,
    equity: Any,
) -> Decimal:
    """Calculate debt-to-equity ratio."""

    return calculate_ratio(
        to_decimal(debt),
        to_decimal(equity),
    )


def calculate_debt_to_assets(
    debt: Any,
    assets: Any,
) -> Decimal:
    """Calculate debt-to-assets percentage."""

    return calculate_percentage(
        to_decimal(debt),
        to_decimal(assets),
    )


def calculate_interest_coverage(
    operating_profit: Any,
    interest_expense: Any,
) -> Decimal:
    """Calculate interest coverage ratio."""

    return calculate_ratio(
        to_decimal(operating_profit),
        abs(
            to_decimal(interest_expense)
        ),
    )


def calculate_debt_service_coverage(
    operating_cash_flow: Any,
    debt_service: Any,
) -> Decimal:
    """Calculate debt service coverage ratio."""

    return calculate_ratio(
        to_decimal(operating_cash_flow),
        to_decimal(debt_service),
    )


def calculate_asset_turnover(
    revenue: Any,
    total_assets: Any,
) -> Decimal:
    """Calculate asset turnover."""

    return calculate_ratio(
        to_decimal(revenue),
        to_decimal(total_assets),
    )


def calculate_inventory_turnover(
    cogs: Any,
    inventory: Any,
) -> Decimal:
    """Calculate inventory turnover."""

    return calculate_ratio(
        to_decimal(cogs),
        to_decimal(inventory),
    )


def calculate_receivable_turnover(
    revenue: Any,
    receivables: Any,
) -> Decimal:
    """Calculate receivables turnover."""

    return calculate_ratio(
        to_decimal(revenue),
        to_decimal(receivables),
    )


def calculate_inventory_days(
    cogs: Any,
    inventory: Any,
    *,
    days: Any = Decimal("365"),
) -> Decimal:
    """Calculate inventory days."""

    turnover = calculate_inventory_turnover(
        cogs,
        inventory,
    )

    return calculate_ratio(
        to_decimal(days),
        turnover,
    )


def calculate_receivable_days(
    revenue: Any,
    receivables: Any,
    *,
    days: Any = Decimal("365"),
) -> Decimal:
    """Calculate receivable days."""

    turnover = calculate_receivable_turnover(
        revenue,
        receivables,
    )

    return calculate_ratio(
        to_decimal(days),
        turnover,
    )


def calculate_payable_days(
    cogs: Any,
    payables: Any,
    *,
    days: Any = Decimal("365"),
) -> Decimal:
    """Calculate payable days."""

    turnover = calculate_ratio(
        to_decimal(cogs),
        to_decimal(payables),
    )

    return calculate_ratio(
        to_decimal(days),
        turnover,
    )


def calculate_cash_conversion_cycle(
    inventory_days: Any,
    receivable_days: Any,
    payable_days: Any,
) -> Decimal:
    """Calculate cash conversion cycle."""

    return round_ratio(
        to_decimal(inventory_days)
        + to_decimal(receivable_days)
        - to_decimal(payable_days)
    )


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Constants
    "ZERO",
    "ONE",
    "ONE_HUNDRED",
    "RATIO_QUANT",
    "PERCENT_QUANT",
    "MONEY_QUANT",

    # Utilities
    "to_decimal",
    "round_ratio",
    "round_percent",
    "round_money",
    "safe_divide",
    "calculate_ratio",
    "calculate_percentage",
    "percentage_change",

    # Enums
    "RatioCategory",
    "RatioType",
    "RatioHealth",
    "RatioTrend",
    "RatioSeverity",

    # Input / output models
    "FinancialRatioInput",
    "RatioResult",
    "RatioComparison",
    "RatioAlert",
    "RatioAnalysisResult",

    # Benchmarks
    "RatioBenchmark",
    "DEFAULT_BENCHMARKS",

    # Analyzer
    "RatioAnalyzer",

    # Convenience functions
    "analyze_ratios",
    "calculate_all_ratios",
    "calculate_current_ratio",
    "calculate_quick_ratio",
    "calculate_cash_ratio",
    "calculate_debt_to_equity",
    "calculate_debt_to_assets",
    "calculate_interest_coverage",
    "calculate_debt_service_coverage",
    "calculate_asset_turnover",
    "calculate_inventory_turnover",
    "calculate_receivable_turnover",
    "calculate_inventory_days",
    "calculate_receivable_days",
    "calculate_payable_days",
    "calculate_cash_conversion_cycle",
]