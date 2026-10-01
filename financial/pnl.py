"""
FinCo AI - Profit & Loss (P&L) Analysis Engine

File:
    backend/app/financial/pnl.py

Purpose:
    Canonical Profit & Loss statement calculation and analysis layer.

Responsibilities:
    - Build normalized P&L statements.
    - Calculate revenue, COGS, gross profit.
    - Calculate operating expenses and operating profit.
    - Calculate EBITDA where sufficient data is available.
    - Calculate interest, taxes, and net profit.
    - Calculate profitability margins.
    - Calculate expense ratios.
    - Compare current and previous periods.
    - Detect profit growth/decline.
    - Detect margin compression/expansion.
    - Identify major P&L drivers.
    - Assess profitability health.
    - Support scenario/what-if analysis.
    - Produce deterministic outputs for downstream AI agents.

Architecture:

    Raw / Normalized Financial Data
                  │
                  ▼
          financial/normalization.py
                  │
                  ▼
              pnl.py
                  │
       ┌──────────┼───────────┐
       ▼          ▼           ▼
    Revenue      COGS        OPEX
       │          │           │
       └──────────┼───────────┘
                  ▼
             Gross Profit
                  │
                  ▼
          Operating Profit
                  │
           ┌──────┴──────┐
           ▼             ▼
        Interest        Tax
           │             │
           └──────┬──────┘
                  ▼
              Net Profit
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
     Margins     Trends    Drivers
       │          │          │
       └──────────┼──────────┘
                  ▼
       KPIs / Alerts / Forecast
                  │
                  ▼
          Supervisor Agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


# ============================================================================
# Constants
# ============================================================================

ZERO = Decimal("0")
ONE = Decimal("1")
ONE_HUNDRED = Decimal("100")

MONEY_QUANT = Decimal("0.01")
PERCENT_QUANT = Decimal("0.01")


# ============================================================================
# Utility Functions
# ============================================================================


def to_decimal(
    value: Any,
    default: Decimal = ZERO,
) -> Decimal:
    """
    Safely convert a value to Decimal.

    Handles:
        Decimal
        int
        float
        numeric strings
        comma-separated numbers
        accounting parentheses
        currency symbols
        percentage symbols
    """

    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    if isinstance(value, bool):
        return (
            ONE
            if value
            else ZERO
        )

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


def round_money(
    value: Decimal,
) -> Decimal:
    """Round a monetary value to two decimal places."""
    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_percent(
    value: Decimal,
) -> Decimal:
    """Round a percentage to two decimal places."""
    return value.quantize(
        PERCENT_QUANT,
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


def calculate_margin(
    profit: Decimal,
    revenue: Decimal,
) -> Decimal:
    """
    Calculate profit margin as a percentage.

    Example:
        profit = 200
        revenue = 1000

        margin = 20%
    """

    if revenue == ZERO:
        return ZERO

    return round_percent(
        safe_divide(
            profit,
            revenue,
        )
        * ONE_HUNDRED
    )


def percentage_change(
    current: Decimal,
    previous: Decimal,
) -> Decimal | None:
    """
    Calculate percentage change.

    Returns None when previous value is zero because
    percentage growth is undefined.
    """

    if previous == ZERO:
        if current == ZERO:
            return ZERO

        return None

    return round_percent(
        (
            safe_divide(
                current - previous,
                abs(previous),
            )
            * ONE_HUNDRED
        )
    )


def absolute_change(
    current: Decimal,
    previous: Decimal,
) -> Decimal:
    """Calculate absolute change."""
    return round_money(
        current - previous
    )


# ============================================================================
# Enums
# ============================================================================


class ProfitabilityTrend(str, Enum):
    """Overall profitability trend."""

    STRONG_IMPROVEMENT = "strong_improvement"
    IMPROVEMENT = "improvement"
    STABLE = "stable"
    DECLINE = "decline"
    SEVERE_DECLINE = "severe_decline"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class ProfitabilityHealth(str, Enum):
    """P&L health classification."""

    EXCELLENT = "excellent"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    LOSS = "loss"
    UNKNOWN = "unknown"


class ProfitMetric(str, Enum):
    """Supported P&L metrics."""

    REVENUE = "revenue"
    COGS = "cost_of_goods_sold"
    GROSS_PROFIT = "gross_profit"
    OPERATING_EXPENSES = "operating_expenses"
    EBITDA = "ebitda"
    OPERATING_PROFIT = "operating_profit"
    INTEREST_EXPENSE = "interest_expense"
    TAX_EXPENSE = "tax_expense"
    NET_PROFIT = "net_profit"


class PnLDriverType(str, Enum):
    """Types of P&L drivers."""

    REVENUE = "revenue"
    COGS = "cogs"
    OPEX = "opex"
    INTEREST = "interest"
    TAX = "tax"
    MIX = "mix"
    OTHER = "other"


class PnLSeverity(str, Enum):
    """Severity of P&L signals."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Data Models
# ============================================================================


@dataclass(slots=True)
class PnLInput:
    """
    Raw input used to construct a P&L.

    All expense/cost fields are represented as positive magnitudes.
    """

    period: str | None = None

    revenue: Decimal = ZERO

    cost_of_goods_sold: Decimal = ZERO

    operating_expenses: Decimal = ZERO

    interest_expense: Decimal = ZERO

    tax_expense: Decimal = ZERO

    depreciation: Decimal = ZERO

    amortization: Decimal = ZERO

    gross_profit: Decimal | None = None

    operating_profit: Decimal | None = None

    ebitda: Decimal | None = None

    net_profit: Decimal | None = None

    currency: str = "USD"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "PnLInput":
        """Create P&L input from a generic mapping."""

        def first(
            *keys: str,
        ) -> Any:
            for key in keys:
                if key in data:
                    return data[key]

            return None

        return cls(
            period=first(
                "period",
                "reporting_period",
            ),
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
            operating_expenses=to_decimal(
                first(
                    "operating_expenses",
                    "opex",
                    "operating_costs",
                )
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
                    "income_tax",
                )
            ),
            depreciation=to_decimal(
                first(
                    "depreciation",
                    "depreciation_expense",
                )
            ),
            amortization=to_decimal(
                first(
                    "amortization",
                    "amortization_expense",
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
            operating_profit=(
                to_decimal(
                    first("operating_profit")
                )
                if first("operating_profit")
                is not None
                else None
            ),
            ebitda=(
                to_decimal(
                    first("ebitda")
                )
                if first("ebitda")
                is not None
                else None
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
                )
                is not None
                else None
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

    def derive(
        self,
    ) -> "PnLInput":
        """
        Derive missing P&L values.

        Explicitly supplied values are preserved.
        """

        if self.gross_profit is None:
            self.gross_profit = (
                self.revenue
                - self.cost_of_goods_sold
            )

        if self.operating_profit is None:
            self.operating_profit = (
                self.gross_profit
                - self.operating_expenses
            )

        if self.ebitda is None:
            self.ebitda = (
                self.operating_profit
                + self.depreciation
                + self.amortization
            )

        if self.net_profit is None:
            self.net_profit = (
                self.operating_profit
                - self.interest_expense
                - self.tax_expense
            )

        return self


@dataclass(slots=True)
class PnLStatement:
    """Canonical calculated Profit & Loss statement."""

    period: str | None

    currency: str

    revenue: Decimal

    cost_of_goods_sold: Decimal

    gross_profit: Decimal

    operating_expenses: Decimal

    ebitda: Decimal

    operating_profit: Decimal

    interest_expense: Decimal

    tax_expense: Decimal

    net_profit: Decimal

    gross_margin: Decimal

    ebitda_margin: Decimal

    operating_margin: Decimal

    net_margin: Decimal

    cogs_ratio: Decimal

    operating_expense_ratio: Decimal

    interest_ratio: Decimal

    tax_ratio: Decimal

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        monetary_fields = (
            "revenue",
            "cost_of_goods_sold",
            "gross_profit",
            "operating_expenses",
            "ebitda",
            "operating_profit",
            "interest_expense",
            "tax_expense",
            "net_profit",
        )

        percentage_fields = (
            "gross_margin",
            "ebitda_margin",
            "operating_margin",
            "net_margin",
            "cogs_ratio",
            "operating_expense_ratio",
            "interest_ratio",
            "tax_ratio",
        )

        result: dict[str, Any] = {
            "period": self.period,
            "currency": self.currency,
            "metadata": self.metadata,
        }

        for field_name in monetary_fields:
            result[field_name] = str(
                getattr(
                    self,
                    field_name,
                )
            )

        for field_name in percentage_fields:
            result[field_name] = str(
                getattr(
                    self,
                    field_name,
                )
            )

        return result


@dataclass(slots=True)
class PnLComparison:
    """Comparison between two P&L periods."""

    metric: ProfitMetric

    current_value: Decimal

    previous_value: Decimal

    absolute_change: Decimal

    percentage_change: Decimal | None

    direction: str

    severity: PnLSeverity = PnLSeverity.INFO

    explanation: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "current_value": str(
                self.current_value
            ),
            "previous_value": str(
                self.previous_value
            ),
            "absolute_change": str(
                self.absolute_change
            ),
            "percentage_change": (
                str(self.percentage_change)
                if self.percentage_change
                is not None
                else None
            ),
            "direction": self.direction,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PnLDriver:
    """A detected driver of P&L change."""

    driver_type: PnLDriverType

    metric: ProfitMetric

    impact: Decimal

    direction: str

    severity: PnLSeverity

    explanation: str

    percentage_of_revenue: Decimal = ZERO

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "driver_type": self.driver_type.value,
            "metric": self.metric.value,
            "impact": str(self.impact),
            "direction": self.direction,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "percentage_of_revenue": str(
                self.percentage_of_revenue
            ),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PnLAlert:
    """P&L warning/signal."""

    code: str

    severity: PnLSeverity

    metric: ProfitMetric

    message: str

    value: Decimal

    threshold: Decimal | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "metric": self.metric.value,
            "message": self.message,
            "value": str(self.value),
            "threshold": (
                str(self.threshold)
                if self.threshold is not None
                else None
            ),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PnLAnalysisResult:
    """Complete P&L analysis result."""

    statement: PnLStatement

    health: ProfitabilityHealth

    trend: ProfitabilityTrend

    comparisons: list[PnLComparison] = field(
        default_factory=list
    )

    drivers: list[PnLDriver] = field(
        default_factory=list
    )

    alerts: list[PnLAlert] = field(
        default_factory=list
    )

    summary: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "statement": self.statement.to_dict(),
            "health": self.health.value,
            "trend": self.trend.value,
            "comparisons": [
                item.to_dict()
                for item in self.comparisons
            ],
            "drivers": [
                item.to_dict()
                for item in self.drivers
            ],
            "alerts": [
                item.to_dict()
                for item in self.alerts
            ],
            "summary": self.summary,
            "metadata": self.metadata,
        }


# ============================================================================
# P&L Analyzer
# ============================================================================


class PnLAnalyzer:
    """
    Production-oriented Profit & Loss analyzer.

    The analyzer is intentionally stateless and database-agnostic.
    """

    def __init__(
        self,
        *,
        excellent_margin: Decimal = Decimal("20"),
        healthy_margin: Decimal = Decimal("10"),
        warning_margin: Decimal = Decimal("0"),
        severe_decline_threshold: Decimal = Decimal("20"),
        decline_threshold: Decimal = Decimal("5"),
        improvement_threshold: Decimal = Decimal("5"),
        strong_improvement_threshold: Decimal = Decimal("20"),
    ) -> None:
        self.excellent_margin = (
            excellent_margin
        )

        self.healthy_margin = (
            healthy_margin
        )

        self.warning_margin = (
            warning_margin
        )

        self.severe_decline_threshold = (
            severe_decline_threshold
        )

        self.decline_threshold = (
            decline_threshold
        )

        self.improvement_threshold = (
            improvement_threshold
        )

        self.strong_improvement_threshold = (
            strong_improvement_threshold
        )

    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        current: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
        previous: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
            | None
        ) = None,
        *,
        period_history: Sequence[
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ]
        | None = None,
    ) -> PnLAnalysisResult:
        """
        Perform complete P&L analysis.
        """

        current_statement = (
            self.build_statement(
                current
            )
        )

        previous_statement = (
            self.build_statement(
                previous
            )
            if previous is not None
            else None
        )

        comparisons: list[PnLComparison] = []

        if previous_statement:
            comparisons = (
                self.compare(
                    current_statement,
                    previous_statement,
                )
            )

        health = self.classify_health(
            current_statement
        )

        trend = self.determine_trend(
            current_statement,
            previous_statement,
            period_history=period_history,
        )

        drivers = self.analyze_drivers(
            current_statement,
            previous_statement,
        )

        alerts = self.generate_alerts(
            current_statement,
            previous_statement,
        )

        summary = self.generate_summary(
            current_statement,
            health,
            trend,
            comparisons,
            drivers,
            alerts,
        )

        return PnLAnalysisResult(
            statement=current_statement,
            health=health,
            trend=trend,
            comparisons=comparisons,
            drivers=drivers,
            alerts=alerts,
            summary=summary,
            metadata={
                "has_previous_period": (
                    previous_statement is not None
                ),
                "history_length": (
                    len(period_history)
                    if period_history
                    else 0
                ),
            },
        )

    # ------------------------------------------------------------------
    # Statement Construction
    # ------------------------------------------------------------------

    def build_statement(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> PnLStatement:
        """Build canonical P&L statement."""

        if isinstance(
            data,
            PnLStatement,
        ):
            return data

        if isinstance(
            data,
            PnLInput,
        ):
            normalized = self.normalize_input(
                data
            )
        else:
            normalized = self.normalize_input(
                PnLInput.from_mapping(
                    data
                )
            )

        normalized.derive()

        revenue = round_money(
            normalized.revenue
        )

        cogs = round_money(
            normalized.cost_of_goods_sold
        )

        gross_profit = round_money(
            normalized.gross_profit
            if normalized.gross_profit
            is not None
            else revenue - cogs
        )

        operating_expenses = round_money(
            normalized.operating_expenses
        )

        operating_profit = round_money(
            normalized.operating_profit
            if normalized.operating_profit
            is not None
            else gross_profit
            - operating_expenses
        )

        ebitda = round_money(
            normalized.ebitda
            if normalized.ebitda
            is not None
            else operating_profit
            + normalized.depreciation
            + normalized.amortization
        )

        interest = round_money(
            normalized.interest_expense
        )

        tax = round_money(
            normalized.tax_expense
        )

        net_profit = round_money(
            normalized.net_profit
            if normalized.net_profit
            is not None
            else operating_profit
            - interest
            - tax
        )

        gross_margin = calculate_margin(
            gross_profit,
            revenue,
        )

        ebitda_margin = calculate_margin(
            ebitda,
            revenue,
        )

        operating_margin = calculate_margin(
            operating_profit,
            revenue,
        )

        net_margin = calculate_margin(
            net_profit,
            revenue,
        )

        cogs_ratio = calculate_margin(
            cogs,
            revenue,
        )

        operating_expense_ratio = (
            calculate_margin(
                operating_expenses,
                revenue,
            )
        )

        interest_ratio = calculate_margin(
            interest,
            revenue,
        )

        tax_ratio = calculate_margin(
            tax,
            revenue,
        )

        return PnLStatement(
            period=normalized.period,
            currency=normalized.currency,
            revenue=revenue,
            cost_of_goods_sold=cogs,
            gross_profit=gross_profit,
            operating_expenses=operating_expenses,
            ebitda=ebitda,
            operating_profit=operating_profit,
            interest_expense=interest,
            tax_expense=tax,
            net_profit=net_profit,
            gross_margin=gross_margin,
            ebitda_margin=ebitda_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            cogs_ratio=cogs_ratio,
            operating_expense_ratio=(
                operating_expense_ratio
            ),
            interest_ratio=interest_ratio,
            tax_ratio=tax_ratio,
            metadata=dict(
                normalized.metadata
            ),
        )

    # ------------------------------------------------------------------
    # Input Normalization
    # ------------------------------------------------------------------

    def normalize_input(
        self,
        data: PnLInput,
    ) -> PnLInput:
        """
        Normalize P&L inputs.

        Revenue and cost/expense values are stored as positive
        magnitudes. Profit values retain their natural sign.
        """

        normalized = PnLInput(
            period=data.period,
            revenue=abs(
                to_decimal(data.revenue)
            ),
            cost_of_goods_sold=abs(
                to_decimal(
                    data.cost_of_goods_sold
                )
            ),
            operating_expenses=abs(
                to_decimal(
                    data.operating_expenses
                )
            ),
            interest_expense=abs(
                to_decimal(
                    data.interest_expense
                )
            ),
            tax_expense=abs(
                to_decimal(
                    data.tax_expense
                )
            ),
            depreciation=abs(
                to_decimal(
                    data.depreciation
                )
            ),
            amortization=abs(
                to_decimal(
                    data.amortization
                )
            ),
            gross_profit=(
                to_decimal(
                    data.gross_profit
                )
                if data.gross_profit
                is not None
                else None
            ),
            operating_profit=(
                to_decimal(
                    data.operating_profit
                )
                if data.operating_profit
                is not None
                else None
            ),
            ebitda=(
                to_decimal(
                    data.ebitda
                )
                if data.ebitda
                is not None
                else None
            ),
            net_profit=(
                to_decimal(
                    data.net_profit
                )
                if data.net_profit
                is not None
                else None
            ),
            currency=(
                str(
                    data.currency
                    or "USD"
                ).upper()
            ),
            metadata=dict(
                data.metadata
            ),
        )

        return normalized

    # ------------------------------------------------------------------
    # Individual Metrics
    # ------------------------------------------------------------------

    def revenue(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        """Return revenue."""
        return self.build_statement(
            data
        ).revenue

    def cogs(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        """Return COGS."""
        return self.build_statement(
            data
        ).cost_of_goods_sold

    def gross_profit(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        """Return gross profit."""
        return self.build_statement(
            data
        ).gross_profit

    def operating_expenses(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        """Return operating expenses."""
        return self.build_statement(
            data
        ).operating_expenses

    def ebitda(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        """Return EBITDA."""
        return self.build_statement(
            data
        ).ebitda

    def operating_profit(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        """Return operating profit."""
        return self.build_statement(
            data
        ).operating_profit

    def net_profit(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        """Return net profit."""
        return self.build_statement(
            data
        ).net_profit

    # ------------------------------------------------------------------
    # Margins
    # ------------------------------------------------------------------

    def gross_margin(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        return self.build_statement(
            data
        ).gross_margin

    def ebitda_margin(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        return self.build_statement(
            data
        ).ebitda_margin

    def operating_margin(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        return self.build_statement(
            data
        ).operating_margin

    def net_margin(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
    ) -> Decimal:
        return self.build_statement(
            data
        ).net_margin

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def classify_health(
        self,
        statement: PnLStatement,
    ) -> ProfitabilityHealth:
        """
        Classify profitability health primarily from net margin.
        """

        if statement.revenue == ZERO:
            return ProfitabilityHealth.UNKNOWN

        if statement.net_profit < ZERO:
            return ProfitabilityHealth.LOSS

        if (
            statement.net_margin
            >= self.excellent_margin
        ):
            return ProfitabilityHealth.EXCELLENT

        if (
            statement.net_margin
            >= self.healthy_margin
        ):
            return ProfitabilityHealth.HEALTHY

        if (
            statement.net_margin
            > self.warning_margin
        ):
            return ProfitabilityHealth.WARNING

        return ProfitabilityHealth.CRITICAL

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    def determine_trend(
        self,
        current: PnLStatement,
        previous: PnLStatement | None = None,
        *,
        period_history: Sequence[
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ]
        | None = None,
    ) -> ProfitabilityTrend:
        """
        Determine profitability trend.

        Net-profit percentage change is the primary signal.
        Net-margin movement is used as a secondary signal.
        """

        if previous is None:
            if period_history:
                statements = [
                    self.build_statement(
                        item
                    )
                    for item in period_history
                ]

                if len(statements) >= 2:
                    previous = statements[-2]

            if previous is None:
                return ProfitabilityTrend.UNKNOWN

        profit_change = percentage_change(
            current.net_profit,
            previous.net_profit,
        )

        margin_change = (
            current.net_margin
            - previous.net_margin
        )

        # Loss-to-loss or profit-to-profit movement.
        if profit_change is not None:

            if (
                profit_change
                >= self.strong_improvement_threshold
            ):
                return (
                    ProfitabilityTrend.STRONG_IMPROVEMENT
                )

            if (
                profit_change
                >= self.improvement_threshold
            ):
                return ProfitabilityTrend.IMPROVEMENT

            if (
                profit_change
                <= -self.severe_decline_threshold
            ):
                return (
                    ProfitabilityTrend.SEVERE_DECLINE
                )

            if (
                profit_change
                <= -self.decline_threshold
            ):
                return ProfitabilityTrend.DECLINE

        if (
            margin_change
            >= self.improvement_threshold
        ):
            return ProfitabilityTrend.IMPROVEMENT

        if (
            margin_change
            <= -self.decline_threshold
        ):
            return ProfitabilityTrend.DECLINE

        return ProfitabilityTrend.STABLE

    # ------------------------------------------------------------------
    # Period Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        current: PnLStatement,
        previous: PnLStatement,
    ) -> list[PnLComparison]:
        """Compare all core P&L metrics."""

        metrics = (
            ProfitMetric.REVENUE,
            ProfitMetric.COGS,
            ProfitMetric.GROSS_PROFIT,
            ProfitMetric.OPERATING_EXPENSES,
            ProfitMetric.EBITDA,
            ProfitMetric.OPERATING_PROFIT,
            ProfitMetric.INTEREST_EXPENSE,
            ProfitMetric.TAX_EXPENSE,
            ProfitMetric.NET_PROFIT,
        )

        comparisons: list[PnLComparison] = []

        for metric in metrics:
            current_value = self.metric_value(
                current,
                metric,
            )

            previous_value = self.metric_value(
                previous,
                metric,
            )

            change = (
                current_value
                - previous_value
            )

            pct_change = percentage_change(
                current_value,
                previous_value,
            )

            if change > ZERO:
                direction = "increase"
            elif change < ZERO:
                direction = "decrease"
            else:
                direction = "unchanged"

            severity = (
                self.classify_change_severity(
                    metric,
                    pct_change,
                )
            )

            explanation = (
                self.explain_comparison(
                    metric,
                    change,
                    pct_change,
                )
            )

            comparisons.append(
                PnLComparison(
                    metric=metric,
                    current_value=current_value,
                    previous_value=previous_value,
                    absolute_change=round_money(
                        change
                    ),
                    percentage_change=pct_change,
                    direction=direction,
                    severity=severity,
                    explanation=explanation,
                )
            )

        return comparisons

    def metric_value(
        self,
        statement: PnLStatement,
        metric: ProfitMetric,
    ) -> Decimal:
        """Get metric value from a P&L statement."""

        mapping = {
            ProfitMetric.REVENUE: (
                statement.revenue
            ),
            ProfitMetric.COGS: (
                statement.cost_of_goods_sold
            ),
            ProfitMetric.GROSS_PROFIT: (
                statement.gross_profit
            ),
            ProfitMetric.OPERATING_EXPENSES: (
                statement.operating_expenses
            ),
            ProfitMetric.EBITDA: (
                statement.ebitda
            ),
            ProfitMetric.OPERATING_PROFIT: (
                statement.operating_profit
            ),
            ProfitMetric.INTEREST_EXPENSE: (
                statement.interest_expense
            ),
            ProfitMetric.TAX_EXPENSE: (
                statement.tax_expense
            ),
            ProfitMetric.NET_PROFIT: (
                statement.net_profit
            ),
        }

        return mapping[metric]

    # ------------------------------------------------------------------
    # Change Severity
    # ------------------------------------------------------------------

    def classify_change_severity(
        self,
        metric: ProfitMetric,
        change: Decimal | None,
    ) -> PnLSeverity:
        """Classify significance of a period-over-period change."""

        if change is None:
            return PnLSeverity.INFO

        magnitude = abs(change)

        if magnitude >= Decimal("30"):
            return PnLSeverity.CRITICAL

        if magnitude >= Decimal("20"):
            return PnLSeverity.HIGH

        if magnitude >= Decimal("10"):
            return PnLSeverity.MEDIUM

        if magnitude >= Decimal("5"):
            return PnLSeverity.LOW

        return PnLSeverity.INFO

    # ------------------------------------------------------------------
    # Driver Analysis
    # ------------------------------------------------------------------

    def analyze_drivers(
        self,
        current: PnLStatement,
        previous: PnLStatement | None = None,
    ) -> list[PnLDriver]:
        """
        Identify major P&L drivers.

        This is a deterministic driver analysis rather than a
        statistical causal attribution model.
        """

        if previous is None:
            return []

        drivers: list[PnLDriver] = []

        revenue_change = (
            current.revenue
            - previous.revenue
        )

        cogs_change = (
            current.cost_of_goods_sold
            - previous.cost_of_goods_sold
        )

        opex_change = (
            current.operating_expenses
            - previous.operating_expenses
        )

        interest_change = (
            current.interest_expense
            - previous.interest_expense
        )

        tax_change = (
            current.tax_expense
            - previous.tax_expense
        )

        drivers.append(
            self._build_driver(
                driver_type=PnLDriverType.REVENUE,
                metric=ProfitMetric.REVENUE,
                impact=revenue_change,
                revenue=current.revenue,
                positive_for_profit=True,
            )
        )

        drivers.append(
            self._build_driver(
                driver_type=PnLDriverType.COGS,
                metric=ProfitMetric.COGS,
                impact=cogs_change,
                revenue=current.revenue,
                positive_for_profit=False,
            )
        )

        drivers.append(
            self._build_driver(
                driver_type=PnLDriverType.OPEX,
                metric=ProfitMetric.OPERATING_EXPENSES,
                impact=opex_change,
                revenue=current.revenue,
                positive_for_profit=False,
            )
        )

        drivers.append(
            self._build_driver(
                driver_type=PnLDriverType.INTEREST,
                metric=ProfitMetric.INTEREST_EXPENSE,
                impact=interest_change,
                revenue=current.revenue,
                positive_for_profit=False,
            )
        )

        drivers.append(
            self._build_driver(
                driver_type=PnLDriverType.TAX,
                metric=ProfitMetric.TAX_EXPENSE,
                impact=tax_change,
                revenue=current.revenue,
                positive_for_profit=False,
            )
        )

        # Largest impacts first.
        drivers.sort(
            key=lambda item: abs(
                item.impact
            ),
            reverse=True,
        )

        return drivers

    def _build_driver(
        self,
        *,
        driver_type: PnLDriverType,
        metric: ProfitMetric,
        impact: Decimal,
        revenue: Decimal,
        positive_for_profit: bool,
    ) -> PnLDriver:
        """
        Build a P&L driver.

        For revenue:
            increase -> positive impact

        For costs:
            increase -> negative impact
        """

        if positive_for_profit:
            profit_impact = impact
        else:
            profit_impact = -impact

        if profit_impact > ZERO:
            direction = "positive"
        elif profit_impact < ZERO:
            direction = "negative"
        else:
            direction = "neutral"

        magnitude = abs(profit_impact)

        if magnitude >= Decimal("100000"):
            severity = PnLSeverity.CRITICAL
        elif magnitude >= Decimal("50000"):
            severity = PnLSeverity.HIGH
        elif magnitude >= Decimal("10000"):
            severity = PnLSeverity.MEDIUM
        elif magnitude > ZERO:
            severity = PnLSeverity.LOW
        else:
            severity = PnLSeverity.INFO

        percentage_of_revenue = (
            calculate_margin(
                magnitude,
                revenue,
            )
        )

        if direction == "positive":
            explanation = (
                f"{metric.value} contributed positively "
                f"to profitability by approximately "
                f"{abs(impact):,.2f}."
            )
        elif direction == "negative":
            explanation = (
                f"{metric.value} negatively affected "
                f"profitability by approximately "
                f"{abs(impact):,.2f}."
            )
        else:
            explanation = (
                f"{metric.value} had no material "
                f"period-over-period change."
            )

        return PnLDriver(
            driver_type=driver_type,
            metric=metric,
            impact=round_money(
                profit_impact
            ),
            direction=direction,
            severity=severity,
            explanation=explanation,
            percentage_of_revenue=(
                percentage_of_revenue
            ),
        )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def generate_alerts(
        self,
        current: PnLStatement,
        previous: PnLStatement | None = None,
    ) -> list[PnLAlert]:
        """Generate deterministic P&L alerts."""

        alerts: list[PnLAlert] = []

        # Loss alert.
        if current.net_profit < ZERO:
            alerts.append(
                PnLAlert(
                    code="NET_LOSS",
                    severity=PnLSeverity.CRITICAL,
                    metric=ProfitMetric.NET_PROFIT,
                    message=(
                        "The business is reporting a net loss "
                        "for the analyzed period."
                    ),
                    value=current.net_profit,
                )
            )

        # Negative operating profit.
        if current.operating_profit < ZERO:
            alerts.append(
                PnLAlert(
                    code="OPERATING_LOSS",
                    severity=PnLSeverity.HIGH,
                    metric=ProfitMetric.OPERATING_PROFIT,
                    message=(
                        "Operating expenses exceed gross profit."
                    ),
                    value=current.operating_profit,
                )
            )

        # Gross margin warning.
        if (
            current.gross_margin < Decimal("20")
            and current.revenue > ZERO
        ):
            alerts.append(
                PnLAlert(
                    code="LOW_GROSS_MARGIN",
                    severity=PnLSeverity.MEDIUM,
                    metric=ProfitMetric.GROSS_PROFIT,
                    message=(
                        "Gross margin is below the configured "
                        "20% warning threshold."
                    ),
                    value=current.gross_margin,
                    threshold=Decimal("20"),
                )
            )

        # Operating margin warning.
        if (
            current.operating_margin < Decimal("10")
            and current.revenue > ZERO
        ):
            alerts.append(
                PnLAlert(
                    code="LOW_OPERATING_MARGIN",
                    severity=PnLSeverity.MEDIUM,
                    metric=ProfitMetric.OPERATING_PROFIT,
                    message=(
                        "Operating margin is below the configured "
                        "10% warning threshold."
                    ),
                    value=current.operating_margin,
                    threshold=Decimal("10"),
                )
            )

        # Net margin warning.
        if (
            current.net_margin < Decimal("5")
            and current.revenue > ZERO
        ):
            alerts.append(
                PnLAlert(
                    code="LOW_NET_MARGIN",
                    severity=PnLSeverity.MEDIUM,
                    metric=ProfitMetric.NET_PROFIT,
                    message=(
                        "Net margin is below the configured "
                        "5% warning threshold."
                    ),
                    value=current.net_margin,
                    threshold=Decimal("5"),
                )
            )

        if previous is not None:
            net_profit_change = (
                percentage_change(
                    current.net_profit,
                    previous.net_profit,
                )
            )

            if (
                net_profit_change is not None
                and net_profit_change
                <= -20
            ):
                alerts.append(
                    PnLAlert(
                        code="SEVERE_PROFIT_DECLINE",
                        severity=PnLSeverity.HIGH,
                        metric=ProfitMetric.NET_PROFIT,
                        message=(
                            "Net profit declined by 20% or more "
                            "compared with the previous period."
                        ),
                        value=net_profit_change,
                        threshold=Decimal("-20"),
                    )
                )

            margin_change = (
                current.net_margin
                - previous.net_margin
            )

            if margin_change <= -5:
                alerts.append(
                    PnLAlert(
                        code="NET_MARGIN_COMPRESSION",
                        severity=PnLSeverity.HIGH,
                        metric=ProfitMetric.NET_PROFIT,
                        message=(
                            "Net margin compressed by at least "
                            "5 percentage points."
                        ),
                        value=margin_change,
                        threshold=Decimal("-5"),
                    )
                )

            gross_margin_change = (
                current.gross_margin
                - previous.gross_margin
            )

            if gross_margin_change <= -5:
                alerts.append(
                    PnLAlert(
                        code="GROSS_MARGIN_COMPRESSION",
                        severity=PnLSeverity.HIGH,
                        metric=ProfitMetric.GROSS_PROFIT,
                        message=(
                            "Gross margin compressed by at least "
                            "5 percentage points."
                        ),
                        value=gross_margin_change,
                        threshold=Decimal("-5"),
                    )
                )

        return alerts

    # ------------------------------------------------------------------
    # What-If Analysis
    # ------------------------------------------------------------------

    def simulate(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
        *,
        revenue_change_pct: Decimal = ZERO,
        cogs_change_pct: Decimal = ZERO,
        operating_expense_change_pct: Decimal = ZERO,
        interest_change_pct: Decimal = ZERO,
        tax_change_pct: Decimal = ZERO,
    ) -> PnLStatement:
        """
        Simulate a P&L scenario.

        Changes are percentages.

        Example:
            revenue_change_pct = 10
            -> revenue increases by 10%

            cogs_change_pct = -5
            -> COGS decreases by 5%
        """

        base = self.build_statement(
            data
        )

        revenue_multiplier = (
            ONE
            + (
                revenue_change_pct
                / ONE_HUNDRED
            )
        )

        cogs_multiplier = (
            ONE
            + (
                cogs_change_pct
                / ONE_HUNDRED
            )
        )

        opex_multiplier = (
            ONE
            + (
                operating_expense_change_pct
                / ONE_HUNDRED
            )
        )

        interest_multiplier = (
            ONE
            + (
                interest_change_pct
                / ONE_HUNDRED
            )
        )

        tax_multiplier = (
            ONE
            + (
                tax_change_pct
                / ONE_HUNDRED
            )
        )

        revenue = max(
            ZERO,
            base.revenue
            * revenue_multiplier,
        )

        cogs = max(
            ZERO,
            base.cost_of_goods_sold
            * cogs_multiplier,
        )

        opex = max(
            ZERO,
            base.operating_expenses
            * opex_multiplier,
        )

        interest = max(
            ZERO,
            base.interest_expense
            * interest_multiplier,
        )

        tax = max(
            ZERO,
            base.tax_expense
            * tax_multiplier,
        )

        simulated = PnLInput(
            period=base.period,
            revenue=revenue,
            cost_of_goods_sold=cogs,
            operating_expenses=opex,
            interest_expense=interest,
            tax_expense=tax,
            depreciation=(
                base.ebitda
                - base.operating_profit
            ),
            currency=base.currency,
            metadata={
                **base.metadata,
                "scenario": True,
            },
        )

        return self.build_statement(
            simulated
        )

    # ------------------------------------------------------------------
    # Break-Even
    # ------------------------------------------------------------------

    def break_even_revenue(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
        *,
        fixed_operating_expenses: Decimal
        | None = None,
    ) -> Decimal | None:
        """
        Calculate break-even revenue.

        Uses contribution margin ratio:

            Contribution Margin Ratio =
                (Revenue - COGS) / Revenue

            Break-even Revenue =
                Fixed Costs / Contribution Margin Ratio

        If fixed operating expenses are not supplied,
        operating expenses are treated as fixed.
        """

        statement = self.build_statement(
            data
        )

        if statement.revenue <= ZERO:
            return None

        contribution_profit = (
            statement.revenue
            - statement.cost_of_goods_sold
        )

        contribution_margin_ratio = safe_divide(
            contribution_profit,
            statement.revenue,
        )

        if contribution_margin_ratio <= ZERO:
            return None

        fixed_costs = (
            statement.operating_expenses
            if fixed_operating_expenses
            is None
            else abs(
                to_decimal(
                    fixed_operating_expenses
                )
            )
        )

        return round_money(
            safe_divide(
                fixed_costs,
                contribution_margin_ratio,
            )
        )

    # ------------------------------------------------------------------
    # Margin of Safety
    # ------------------------------------------------------------------

    def margin_of_safety(
        self,
        data: (
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ),
        *,
        fixed_operating_expenses: Decimal
        | None = None,
    ) -> Decimal | None:
        """
        Calculate margin of safety as a percentage.

            (Actual Revenue - Break-even Revenue)
            / Actual Revenue
        """

        statement = self.build_statement(
            data
        )

        break_even = self.break_even_revenue(
            statement,
            fixed_operating_expenses=(
                fixed_operating_expenses
            ),
        )

        if (
            break_even is None
            or statement.revenue == ZERO
        ):
            return None

        return round_percent(
            safe_divide(
                statement.revenue
                - break_even,
                statement.revenue,
            )
            * ONE_HUNDRED
        )

    # ------------------------------------------------------------------
    # Historical Analysis
    # ------------------------------------------------------------------

    def analyze_history(
        self,
        history: Sequence[
            PnLInput
            | PnLStatement
            | Mapping[str, Any]
        ],
    ) -> dict[str, Any]:
        """
        Analyze a sequence of P&L periods.
        """

        statements = [
            self.build_statement(
                item
            )
            for item in history
        ]

        if not statements:
            return {
                "periods": [],
                "trend": ProfitabilityTrend.UNKNOWN.value,
                "latest": None,
            }

        comparisons: list[PnLComparison] = []

        for index in range(
            1,
            len(statements),
        ):
            comparisons.extend(
                self.compare(
                    statements[index],
                    statements[index - 1],
                )
            )

        net_profit_series = [
            statement.net_profit
            for statement in statements
        ]

        net_margin_series = [
            statement.net_margin
            for statement in statements
        ]

        revenue_series = [
            statement.revenue
            for statement in statements
        ]

        return {
            "periods": [
                statement.to_dict()
                for statement in statements
            ],
            "trend": self.determine_history_trend(
                net_profit_series
            ).value,
            "latest": statements[-1].to_dict(),
            "net_profit_series": [
                str(value)
                for value in net_profit_series
            ],
            "net_margin_series": [
                str(value)
                for value in net_margin_series
            ],
            "revenue_series": [
                str(value)
                for value in revenue_series
            ],
            "comparisons": [
                comparison.to_dict()
                for comparison in comparisons
            ],
        }

    def determine_history_trend(
        self,
        values: Sequence[Decimal],
    ) -> ProfitabilityTrend:
        """Determine trend from a profitability series."""

        if len(values) < 2:
            return ProfitabilityTrend.UNKNOWN

        changes: list[Decimal] = []

        for index in range(
            1,
            len(values),
        ):
            change = percentage_change(
                values[index],
                values[index - 1],
            )

            if change is not None:
                changes.append(
                    change
                )

        if not changes:
            return ProfitabilityTrend.UNKNOWN

        average_change = (
            sum(changes, ZERO)
            / Decimal(len(changes))
        )

        if (
            average_change
            >= self.strong_improvement_threshold
        ):
            return (
                ProfitabilityTrend.STRONG_IMPROVEMENT
            )

        if (
            average_change
            >= self.improvement_threshold
        ):
            return ProfitabilityTrend.IMPROVEMENT

        if (
            average_change
            <= -self.severe_decline_threshold
        ):
            return (
                ProfitabilityTrend.SEVERE_DECLINE
            )

        if (
            average_change
            <= -self.decline_threshold
        ):
            return ProfitabilityTrend.DECLINE

        return ProfitabilityTrend.STABLE

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_summary(
        self,
        statement: PnLStatement,
        health: ProfitabilityHealth,
        trend: ProfitabilityTrend,
        comparisons: Sequence[PnLComparison],
        drivers: Sequence[PnLDriver],
        alerts: Sequence[PnLAlert],
    ) -> str:
        """Generate concise machine-readable/user-readable summary."""

        revenue = (
            f"{statement.revenue:,.2f}"
        )

        net_profit = (
            f"{statement.net_profit:,.2f}"
        )

        net_margin = (
            f"{statement.net_margin:.2f}%"
        )

        summary = (
            f"Revenue was {revenue} "
            f"with net profit of {net_profit} "
            f"and net margin of {net_margin}. "
            f"Profitability health is "
            f"{health.value}, with an overall trend of "
            f"{trend.value}."
        )

        if drivers:
            major_driver = drivers[0]

            summary += (
                f" The largest detected P&L driver was "
                f"{major_driver.metric.value}, "
                f"with a {major_driver.direction} "
                f"impact of "
                f"{abs(major_driver.impact):,.2f}."
            )

        if alerts:
            highest = max(
                alerts,
                key=self._severity_rank,
            )

            summary += (
                f" The highest-priority P&L signal is "
                f"{highest.code}."
            )

        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def explain_comparison(
        metric: ProfitMetric,
        change: Decimal,
        pct_change: Decimal | None,
    ) -> str:
        """Generate comparison explanation."""

        if change == ZERO:
            return (
                f"{metric.value} remained unchanged."
            )

        direction = (
            "increased"
            if change > ZERO
            else "decreased"
        )

        if pct_change is None:
            return (
                f"{metric.value} {direction} by "
                f"{abs(change):,.2f}; percentage "
                f"change is undefined because the "
                f"previous value was zero."
            )

        return (
            f"{metric.value} {direction} by "
            f"{abs(change):,.2f} "
            f"({abs(pct_change):.2f}%)."
        )

    @staticmethod
    def _severity_rank(
        alert: PnLAlert,
    ) -> int:
        ranks = {
            PnLSeverity.INFO: 1,
            PnLSeverity.LOW: 2,
            PnLSeverity.MEDIUM: 3,
            PnLSeverity.HIGH: 4,
            PnLSeverity.CRITICAL: 5,
        }

        return ranks[
            alert.severity
        ]


# ============================================================================
# Convenience Functions
# ============================================================================


def calculate_pnl(
    data: (
        PnLInput
        | PnLStatement
        | Mapping[str, Any]
    ),
) -> PnLStatement:
    """Calculate a canonical P&L statement."""

    return PnLAnalyzer().build_statement(
        data
    )


def analyze_pnl(
    current: (
        PnLInput
        | PnLStatement
        | Mapping[str, Any]
    ),
    previous: (
        PnLInput
        | PnLStatement
        | Mapping[str, Any]
        | None
    ) = None,
) -> PnLAnalysisResult:
    """Perform complete P&L analysis."""

    return PnLAnalyzer().analyze(
        current,
        previous,
    )


def calculate_gross_profit(
    revenue: Any,
    cost_of_goods_sold: Any,
) -> Decimal:
    """Calculate gross profit."""

    return round_money(
        to_decimal(revenue)
        - abs(
            to_decimal(
                cost_of_goods_sold
            )
        )
    )


def calculate_operating_profit(
    gross_profit: Any,
    operating_expenses: Any,
) -> Decimal:
    """Calculate operating profit."""

    return round_money(
        to_decimal(gross_profit)
        - abs(
            to_decimal(
                operating_expenses
            )
        )
    )


def calculate_net_profit(
    operating_profit: Any,
    interest_expense: Any = ZERO,
    tax_expense: Any = ZERO,
) -> Decimal:
    """Calculate net profit."""

    return round_money(
        to_decimal(
            operating_profit
        )
        - abs(
            to_decimal(
                interest_expense
            )
        )
        - abs(
            to_decimal(
                tax_expense
            )
        )
    )


def calculate_gross_margin(
    revenue: Any,
    cost_of_goods_sold: Any,
) -> Decimal:
    """Calculate gross margin."""

    gross_profit = calculate_gross_profit(
        revenue,
        cost_of_goods_sold,
    )

    return calculate_margin(
        gross_profit,
        to_decimal(revenue),
    )


def calculate_operating_margin(
    revenue: Any,
    gross_profit: Any,
    operating_expenses: Any,
) -> Decimal:
    """Calculate operating margin."""

    operating_profit = (
        calculate_operating_profit(
            gross_profit,
            operating_expenses,
        )
    )

    return calculate_margin(
        operating_profit,
        to_decimal(revenue),
    )


def calculate_net_margin(
    revenue: Any,
    operating_profit: Any,
    interest_expense: Any = ZERO,
    tax_expense: Any = ZERO,
) -> Decimal:
    """Calculate net margin."""

    net_profit = calculate_net_profit(
        operating_profit,
        interest_expense,
        tax_expense,
    )

    return calculate_margin(
        net_profit,
        to_decimal(revenue),
    )


def calculate_break_even_revenue(
    data: (
        PnLInput
        | PnLStatement
        | Mapping[str, Any]
    ),
    *,
    fixed_operating_expenses: Any = None,
) -> Decimal | None:
    """Calculate break-even revenue."""

    return PnLAnalyzer().break_even_revenue(
        data,
        fixed_operating_expenses=(
            to_decimal(
                fixed_operating_expenses
            )
            if fixed_operating_expenses
            is not None
            else None
        ),
    )


def calculate_margin_of_safety(
    data: (
        PnLInput
        | PnLStatement
        | Mapping[str, Any]
    ),
    *,
    fixed_operating_expenses: Any = None,
) -> Decimal | None:
    """Calculate margin of safety."""

    return PnLAnalyzer().margin_of_safety(
        data,
        fixed_operating_expenses=(
            to_decimal(
                fixed_operating_expenses
            )
            if fixed_operating_expenses
            is not None
            else None
        ),
    )


def simulate_pnl(
    data: (
        PnLInput
        | PnLStatement
        | Mapping[str, Any]
    ),
    *,
    revenue_change_pct: Any = ZERO,
    cogs_change_pct: Any = ZERO,
    operating_expense_change_pct: Any = ZERO,
    interest_change_pct: Any = ZERO,
    tax_change_pct: Any = ZERO,
) -> PnLStatement:
    """Run a P&L what-if scenario."""

    analyzer = PnLAnalyzer()

    return analyzer.simulate(
        data,
        revenue_change_pct=to_decimal(
            revenue_change_pct
        ),
        cogs_change_pct=to_decimal(
            cogs_change_pct
        ),
        operating_expense_change_pct=to_decimal(
            operating_expense_change_pct
        ),
        interest_change_pct=to_decimal(
            interest_change_pct
        ),
        tax_change_pct=to_decimal(
            tax_change_pct
        ),
    )


# ============================================================================
# Exports
# ============================================================================


__all__ = [
    # Constants
    "ZERO",
    "ONE",
    "ONE_HUNDRED",
    "MONEY_QUANT",
    "PERCENT_QUANT",

    # Utilities
    "to_decimal",
    "round_money",
    "round_percent",
    "safe_divide",
    "calculate_margin",
    "percentage_change",
    "absolute_change",

    # Enums
    "ProfitabilityTrend",
    "ProfitabilityHealth",
    "ProfitMetric",
    "PnLDriverType",
    "PnLSeverity",

    # Models
    "PnLInput",
    "PnLStatement",
    "PnLComparison",
    "PnLDriver",
    "PnLAlert",
    "PnLAnalysisResult",

    # Analyzer
    "PnLAnalyzer",

    # Convenience functions
    "calculate_pnl",
    "analyze_pnl",
    "calculate_gross_profit",
    "calculate_operating_profit",
    "calculate_net_profit",
    "calculate_gross_margin",
    "calculate_operating_margin",
    "calculate_net_margin",
    "calculate_break_even_revenue",
    "calculate_margin_of_safety",
    "simulate_pnl",
]