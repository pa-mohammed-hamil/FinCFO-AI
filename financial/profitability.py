"""
FinCo AI - Profitability Analysis Engine

File:
    backend/app/financial/profitability.py

Purpose:
    Comprehensive profitability intelligence layer.

Responsibilities:
    - Analyze gross, EBITDA, operating, and net profitability.
    - Analyze profitability margins.
    - Measure profit growth and profitability trends.
    - Assess profitability health.
    - Analyze revenue-to-profit conversion.
    - Analyze cost efficiency.
    - Detect profitability deterioration.
    - Identify profitability improvement opportunities.
    - Compare current vs previous periods.
    - Perform profitability what-if analysis.
    - Calculate break-even and margin of safety.
    - Produce deterministic outputs for alerts, forecasting,
      recommendations, and agent workflows.

Architecture:

    Revenue + COGS + OPEX + Financing + Tax
                     │
                     ▼
                   pnl.py
                     │
                     ▼
             profitability.py
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    Margins        Growth        Efficiency
       │             │             │
       └─────────────┼─────────────┘
                     ▼
            Profitability Health
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Drivers     Risks      Opportunities
          │          │          │
          └──────────┼──────────┘
                     ▼
           What-If / Forecasting
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

MONEY_QUANT = Decimal("0.01")
PERCENT_QUANT = Decimal("0.01")


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


def round_money(
    value: Decimal,
) -> Decimal:
    """Round monetary values to two decimal places."""

    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_percent(
    value: Decimal,
) -> Decimal:
    """Round percentages to two decimal places."""

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


def percentage(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal:
    """Return numerator as a percentage of denominator."""

    if denominator == ZERO:
        return ZERO

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
    Calculate percentage change.

    Returns None when the previous value is zero and the
    current value is non-zero because the percentage change
    is mathematically undefined.
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


def calculate_margin(
    profit: Decimal,
    revenue: Decimal,
) -> Decimal:
    """Calculate profit margin percentage."""

    return percentage(
        profit,
        revenue,
    )


# ============================================================================
# Enums
# ============================================================================


class ProfitabilityLevel(str, Enum):
    """Overall profitability classification."""

    EXCELLENT = "excellent"
    HEALTHY = "healthy"
    MODERATE = "moderate"
    WEAK = "weak"
    CRITICAL = "critical"
    LOSS = "loss"
    UNKNOWN = "unknown"


class ProfitabilityTrend(str, Enum):
    """Profitability direction."""

    STRONG_IMPROVEMENT = "strong_improvement"
    IMPROVEMENT = "improvement"
    STABLE = "stable"
    DECLINE = "decline"
    SEVERE_DECLINE = "severe_decline"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class ProfitabilityMetric(str, Enum):
    """Profitability metrics."""

    REVENUE = "revenue"
    GROSS_PROFIT = "gross_profit"
    EBITDA = "ebitda"
    OPERATING_PROFIT = "operating_profit"
    NET_PROFIT = "net_profit"

    GROSS_MARGIN = "gross_margin"
    EBITDA_MARGIN = "ebitda_margin"
    OPERATING_MARGIN = "operating_margin"
    NET_MARGIN = "net_margin"

    COGS_RATIO = "cogs_ratio"
    OPEX_RATIO = "operating_expense_ratio"
    PROFIT_CONVERSION = "profit_conversion"
    COST_EFFICIENCY = "cost_efficiency"


class ProfitabilityDriverType(str, Enum):
    """Drivers affecting profitability."""

    REVENUE = "revenue"
    GROSS_MARGIN = "gross_margin"
    COGS = "cogs"
    OPEX = "opex"
    OPERATING_MARGIN = "operating_margin"
    INTEREST = "interest"
    TAX = "tax"
    MIX = "mix"
    SCALE = "scale"
    OTHER = "other"


class ProfitabilitySeverity(str, Enum):
    """Severity of profitability signals."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OpportunityType(str, Enum):
    """Profitability improvement opportunity."""

    REVENUE_GROWTH = "revenue_growth"
    PRICE_IMPROVEMENT = "price_improvement"
    COGS_REDUCTION = "cogs_reduction"
    OPEX_REDUCTION = "opex_reduction"
    MARGIN_EXPANSION = "margin_expansion"
    PRODUCT_MIX = "product_mix"
    CUSTOMER_MIX = "customer_mix"
    SCALE_EFFICIENCY = "scale_efficiency"
    FINANCING = "financing"
    TAX_OPTIMIZATION = "tax_optimization"


# ============================================================================
# Data Models
# ============================================================================


@dataclass(slots=True)
class ProfitabilityInput:
    """
    Generic profitability input.

    Monetary costs are expected as positive magnitudes.
    Profit fields retain their natural sign.
    """

    period: str | None = None

    revenue: Decimal = ZERO
    cost_of_goods_sold: Decimal = ZERO
    operating_expenses: Decimal = ZERO

    depreciation: Decimal = ZERO
    amortization: Decimal = ZERO

    interest_expense: Decimal = ZERO
    tax_expense: Decimal = ZERO

    gross_profit: Decimal | None = None
    ebitda: Decimal | None = None
    operating_profit: Decimal | None = None
    net_profit: Decimal | None = None

    currency: str = "USD"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "ProfitabilityInput":
        """Create profitability input from a generic mapping."""

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
            depreciation=to_decimal(
                first("depreciation")
            ),
            amortization=to_decimal(
                first("amortization")
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
            gross_profit=(
                to_decimal(
                    first("gross_profit")
                )
                if first("gross_profit")
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
            operating_profit=(
                to_decimal(
                    first("operating_profit")
                )
                if first("operating_profit")
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


@dataclass(slots=True)
class ProfitabilitySnapshot:
    """Canonical profitability snapshot."""

    period: str | None
    currency: str

    revenue: Decimal

    gross_profit: Decimal
    ebitda: Decimal
    operating_profit: Decimal
    net_profit: Decimal

    gross_margin: Decimal
    ebitda_margin: Decimal
    operating_margin: Decimal
    net_margin: Decimal

    cogs_ratio: Decimal
    opex_ratio: Decimal

    profit_conversion: Decimal
    cost_efficiency: Decimal

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        monetary_fields = (
            "revenue",
            "gross_profit",
            "ebitda",
            "operating_profit",
            "net_profit",
        )

        percentage_fields = (
            "gross_margin",
            "ebitda_margin",
            "operating_margin",
            "net_margin",
            "cogs_ratio",
            "opex_ratio",
            "profit_conversion",
            "cost_efficiency",
        )

        result: dict[str, Any] = {
            "period": self.period,
            "currency": self.currency,
            "metadata": self.metadata,
        }

        for name in monetary_fields:
            result[name] = str(
                getattr(
                    self,
                    name,
                )
            )

        for name in percentage_fields:
            result[name] = str(
                getattr(
                    self,
                    name,
                )
            )

        return result


@dataclass(slots=True)
class ProfitabilityComparison:
    """Comparison between two profitability snapshots."""

    metric: ProfitabilityMetric

    current_value: Decimal
    previous_value: Decimal

    absolute_change: Decimal
    percentage_change: Decimal | None

    direction: str

    severity: ProfitabilitySeverity

    explanation: str

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
                str(
                    self.percentage_change
                )
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
class ProfitabilityDriver:
    """Detected profitability driver."""

    driver_type: ProfitabilityDriverType

    metric: ProfitabilityMetric

    impact: Decimal

    direction: str

    severity: ProfitabilitySeverity

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
class ProfitabilityAlert:
    """Profitability risk or warning."""

    code: str
    severity: ProfitabilitySeverity

    metric: ProfitabilityMetric

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
class ProfitabilityOpportunity:
    """Potential profitability improvement."""

    opportunity_type: OpportunityType

    title: str

    description: str

    estimated_impact: Decimal

    impact_as_percentage_of_revenue: Decimal

    priority: ProfitabilitySeverity

    assumptions: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "opportunity_type": (
                self.opportunity_type.value
            ),
            "title": self.title,
            "description": self.description,
            "estimated_impact": str(
                self.estimated_impact
            ),
            "impact_as_percentage_of_revenue": str(
                self.impact_as_percentage_of_revenue
            ),
            "priority": self.priority.value,
            "assumptions": self.assumptions,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ProfitabilityAnalysisResult:
    """Complete profitability intelligence result."""

    snapshot: ProfitabilitySnapshot

    level: ProfitabilityLevel

    trend: ProfitabilityTrend

    comparisons: list[ProfitabilityComparison] = field(
        default_factory=list
    )

    drivers: list[ProfitabilityDriver] = field(
        default_factory=list
    )

    alerts: list[ProfitabilityAlert] = field(
        default_factory=list
    )

    opportunities: list[
        ProfitabilityOpportunity
    ] = field(
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
            "snapshot": self.snapshot.to_dict(),
            "level": self.level.value,
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
            "opportunities": [
                item.to_dict()
                for item in self.opportunities
            ],
            "summary": self.summary,
            "metadata": self.metadata,
        }


# ============================================================================
# Profitability Analyzer
# ============================================================================


class ProfitabilityAnalyzer:
    """
    Comprehensive profitability analysis engine.

    This layer is intentionally stateless and independent of:
        - FastAPI
        - SQLAlchemy
        - PostgreSQL
        - vector databases
        - LLM providers

    It should receive normalized financial data and return
    deterministic profitability intelligence.
    """

    def __init__(
        self,
        *,
        excellent_net_margin: Decimal = Decimal("20"),
        healthy_net_margin: Decimal = Decimal("10"),
        moderate_net_margin: Decimal = Decimal("5"),
        weak_net_margin: Decimal = Decimal("0"),
        strong_improvement: Decimal = Decimal("20"),
        improvement: Decimal = Decimal("5"),
        decline: Decimal = Decimal("5"),
        severe_decline: Decimal = Decimal("20"),
        margin_compression_warning: Decimal = Decimal("5"),
        high_cogs_ratio: Decimal = Decimal("70"),
        high_opex_ratio: Decimal = Decimal("30"),
    ) -> None:

        self.excellent_net_margin = (
            excellent_net_margin
        )

        self.healthy_net_margin = (
            healthy_net_margin
        )

        self.moderate_net_margin = (
            moderate_net_margin
        )

        self.weak_net_margin = (
            weak_net_margin
        )

        self.strong_improvement = (
            strong_improvement
        )

        self.improvement = improvement
        self.decline = decline
        self.severe_decline = severe_decline

        self.margin_compression_warning = (
            margin_compression_warning
        )

        self.high_cogs_ratio = (
            high_cogs_ratio
        )

        self.high_opex_ratio = (
            high_opex_ratio
        )

    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        current: (
            ProfitabilityInput
            | ProfitabilitySnapshot
            | Mapping[str, Any]
        ),
        previous: (
            ProfitabilityInput
            | ProfitabilitySnapshot
            | Mapping[str, Any]
            | None
        ) = None,
        *,
        history: Sequence[
            ProfitabilityInput
            | ProfitabilitySnapshot
            | Mapping[str, Any]
        ]
        | None = None,
    ) -> ProfitabilityAnalysisResult:

        snapshot = self.build_snapshot(
            current
        )

        previous_snapshot = (
            self.build_snapshot(
                previous
            )
            if previous is not None
            else None
        )

        comparisons = (
            self.compare(
                snapshot,
                previous_snapshot,
            )
            if previous_snapshot is not None
            else []
        )

        level = self.classify_level(
            snapshot
        )

        trend = self.determine_trend(
            snapshot,
            previous_snapshot,
            history=history,
        )

        drivers = self.analyze_drivers(
            snapshot,
            previous_snapshot,
        )

        alerts = self.generate_alerts(
            snapshot,
            previous_snapshot,
        )

        opportunities = (
            self.identify_opportunities(
                snapshot,
                previous_snapshot,
            )
        )

        summary = self.generate_summary(
            snapshot,
            level,
            trend,
            drivers,
            alerts,
            opportunities,
        )

        return ProfitabilityAnalysisResult(
            snapshot=snapshot,
            level=level,
            trend=trend,
            comparisons=comparisons,
            drivers=drivers,
            alerts=alerts,
            opportunities=opportunities,
            summary=summary,
            metadata={
                "has_previous_period": (
                    previous_snapshot is not None
                ),
                "history_length": (
                    len(history)
                    if history
                    else 0
                ),
            },
        )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def build_snapshot(
        self,
        data: (
            ProfitabilityInput
            | ProfitabilitySnapshot
            | Mapping[str, Any]
        ),
    ) -> ProfitabilitySnapshot:

        if isinstance(
            data,
            ProfitabilitySnapshot,
        ):
            return data

        if isinstance(
            data,
            ProfitabilityInput,
        ):
            normalized = data
        else:
            normalized = (
                ProfitabilityInput.from_mapping(
                    data
                )
            )

        revenue = abs(
            to_decimal(
                normalized.revenue
            )
        )

        cogs = abs(
            to_decimal(
                normalized.cost_of_goods_sold
            )
        )

        opex = abs(
            to_decimal(
                normalized.operating_expenses
            )
        )

        interest = abs(
            to_decimal(
                normalized.interest_expense
            )
        )

        tax = abs(
            to_decimal(
                normalized.tax_expense
            )
        )

        depreciation = abs(
            to_decimal(
                normalized.depreciation
            )
        )

        amortization = abs(
            to_decimal(
                normalized.amortization
            )
        )

        if normalized.gross_profit is not None:
            gross_profit = to_decimal(
                normalized.gross_profit
            )
        else:
            gross_profit = (
                revenue - cogs
            )

        if normalized.operating_profit is not None:
            operating_profit = to_decimal(
                normalized.operating_profit
            )
        else:
            operating_profit = (
                gross_profit - opex
            )

        if normalized.ebitda is not None:
            ebitda = to_decimal(
                normalized.ebitda
            )
        else:
            ebitda = (
                operating_profit
                + depreciation
                + amortization
            )

        if normalized.net_profit is not None:
            net_profit = to_decimal(
                normalized.net_profit
            )
        else:
            net_profit = (
                operating_profit
                - interest
                - tax
            )

        gross_profit = round_money(
            gross_profit
        )

        ebitda = round_money(
            ebitda
        )

        operating_profit = round_money(
            operating_profit
        )

        net_profit = round_money(
            net_profit
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

        cogs_ratio = percentage(
            cogs,
            revenue,
        )

        opex_ratio = percentage(
            opex,
            revenue,
        )

        profit_conversion = calculate_margin(
            net_profit,
            revenue,
        )

        total_cost = (
            cogs
            + opex
        )

        cost_efficiency = (
            percentage(
                revenue,
                total_cost,
            )
            if total_cost > ZERO
            else ZERO
        )

        return ProfitabilitySnapshot(
            period=normalized.period,
            currency=(
                str(
                    normalized.currency
                    or "USD"
                ).upper()
            ),
            revenue=round_money(
                revenue
            ),
            gross_profit=gross_profit,
            ebitda=ebitda,
            operating_profit=operating_profit,
            net_profit=net_profit,
            gross_margin=gross_margin,
            ebitda_margin=ebitda_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            cogs_ratio=cogs_ratio,
            opex_ratio=opex_ratio,
            profit_conversion=(
                profit_conversion
            ),
            cost_efficiency=(
                cost_efficiency
            ),
            metadata=dict(
                normalized.metadata
            ),
        )

    # ------------------------------------------------------------------
    # Health Classification
    # ------------------------------------------------------------------

    def classify_level(
        self,
        snapshot: ProfitabilitySnapshot,
    ) -> ProfitabilityLevel:

        if snapshot.revenue <= ZERO:
            return ProfitabilityLevel.UNKNOWN

        if snapshot.net_profit < ZERO:
            return ProfitabilityLevel.LOSS

        if (
            snapshot.net_margin
            >= self.excellent_net_margin
        ):
            return ProfitabilityLevel.EXCELLENT

        if (
            snapshot.net_margin
            >= self.healthy_net_margin
        ):
            return ProfitabilityLevel.HEALTHY

        if (
            snapshot.net_margin
            >= self.moderate_net_margin
        ):
            return ProfitabilityLevel.MODERATE

        if (
            snapshot.net_margin
            > self.weak_net_margin
        ):
            return ProfitabilityLevel.WEAK

        return ProfitabilityLevel.CRITICAL

    # ------------------------------------------------------------------
    # Metric Access
    # ------------------------------------------------------------------

    def metric_value(
        self,
        snapshot: ProfitabilitySnapshot,
        metric: ProfitabilityMetric,
    ) -> Decimal:

        mapping = {
            ProfitabilityMetric.REVENUE:
                snapshot.revenue,

            ProfitabilityMetric.GROSS_PROFIT:
                snapshot.gross_profit,

            ProfitabilityMetric.EBITDA:
                snapshot.ebitda,

            ProfitabilityMetric.OPERATING_PROFIT:
                snapshot.operating_profit,

            ProfitabilityMetric.NET_PROFIT:
                snapshot.net_profit,

            ProfitabilityMetric.GROSS_MARGIN:
                snapshot.gross_margin,

            ProfitabilityMetric.EBITDA_MARGIN:
                snapshot.ebitda_margin,

            ProfitabilityMetric.OPERATING_MARGIN:
                snapshot.operating_margin,

            ProfitabilityMetric.NET_MARGIN:
                snapshot.net_margin,

            ProfitabilityMetric.COGS_RATIO:
                snapshot.cogs_ratio,

            ProfitabilityMetric.OPEX_RATIO:
                snapshot.opex_ratio,

            ProfitabilityMetric.PROFIT_CONVERSION:
                snapshot.profit_conversion,

            ProfitabilityMetric.COST_EFFICIENCY:
                snapshot.cost_efficiency,
        }

        return mapping[metric]

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        current: ProfitabilitySnapshot,
        previous: ProfitabilitySnapshot | None,
    ) -> list[ProfitabilityComparison]:

        if previous is None:
            return []

        metrics = (
            ProfitabilityMetric.REVENUE,
            ProfitabilityMetric.GROSS_PROFIT,
            ProfitabilityMetric.EBITDA,
            ProfitabilityMetric.OPERATING_PROFIT,
            ProfitabilityMetric.NET_PROFIT,
            ProfitabilityMetric.GROSS_MARGIN,
            ProfitabilityMetric.EBITDA_MARGIN,
            ProfitabilityMetric.OPERATING_MARGIN,
            ProfitabilityMetric.NET_MARGIN,
            ProfitabilityMetric.COGS_RATIO,
            ProfitabilityMetric.OPEX_RATIO,
        )

        results: list[
            ProfitabilityComparison
        ] = []

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
                self.change_severity(
                    pct_change
                )
            )

            explanation = (
                self.explain_comparison(
                    metric,
                    change,
                    pct_change,
                )
            )

            results.append(
                ProfitabilityComparison(
                    metric=metric,
                    current_value=(
                        current_value
                    ),
                    previous_value=(
                        previous_value
                    ),
                    absolute_change=(
                        round_money(
                            change
                        )
                    ),
                    percentage_change=(
                        pct_change
                    ),
                    direction=direction,
                    severity=severity,
                    explanation=explanation,
                )
            )

        return results

    def change_severity(
        self,
        change: Decimal | None,
    ) -> ProfitabilitySeverity:

        if change is None:
            return ProfitabilitySeverity.INFO

        magnitude = abs(change)

        if magnitude >= Decimal("30"):
            return ProfitabilitySeverity.CRITICAL

        if magnitude >= Decimal("20"):
            return ProfitabilitySeverity.HIGH

        if magnitude >= Decimal("10"):
            return ProfitabilitySeverity.MEDIUM

        if magnitude >= Decimal("5"):
            return ProfitabilitySeverity.LOW

        return ProfitabilitySeverity.INFO

    # ------------------------------------------------------------------
    # Trend
    # ------------------------------------------------------------------

    def determine_trend(
        self,
        current: ProfitabilitySnapshot,
        previous: ProfitabilitySnapshot | None = None,
        *,
        history: Sequence[
            ProfitabilityInput
            | ProfitabilitySnapshot
            | Mapping[str, Any]
        ]
        | None = None,
    ) -> ProfitabilityTrend:

        if previous is None and history:
            snapshots = [
                self.build_snapshot(
                    item
                )
                for item in history
            ]

            if len(snapshots) >= 2:
                previous = snapshots[-2]

        if previous is None:
            return ProfitabilityTrend.UNKNOWN

        net_profit_change = percentage_change(
            current.net_profit,
            previous.net_profit,
        )

        net_margin_change = (
            current.net_margin
            - previous.net_margin
        )

        if net_profit_change is not None:

            if (
                net_profit_change
                >= self.strong_improvement
            ):
                return (
                    ProfitabilityTrend
                    .STRONG_IMPROVEMENT
                )

            if (
                net_profit_change
                >= self.improvement
            ):
                return (
                    ProfitabilityTrend.IMPROVEMENT
                )

            if (
                net_profit_change
                <= -self.severe_decline
            ):
                return (
                    ProfitabilityTrend
                    .SEVERE_DECLINE
                )

            if (
                net_profit_change
                <= -self.decline
            ):
                return (
                    ProfitabilityTrend.DECLINE
                )

        if (
            net_margin_change
            >= self.improvement
        ):
            return (
                ProfitabilityTrend.IMPROVEMENT
            )

        if (
            net_margin_change
            <= -self.decline
        ):
            return (
                ProfitabilityTrend.DECLINE
            )

        return ProfitabilityTrend.STABLE

    # ------------------------------------------------------------------
    # Driver Analysis
    # ------------------------------------------------------------------

    def analyze_drivers(
        self,
        current: ProfitabilitySnapshot,
        previous: ProfitabilitySnapshot | None,
    ) -> list[ProfitabilityDriver]:

        if previous is None:
            return []

        drivers: list[
            ProfitabilityDriver
        ] = []

        revenue_change = (
            current.revenue
            - previous.revenue
        )

        gross_profit_change = (
            current.gross_profit
            - previous.gross_profit
        )

        ebitda_change = (
            current.ebitda
            - previous.ebitda
        )

        operating_profit_change = (
            current.operating_profit
            - previous.operating_profit
        )

        net_profit_change = (
            current.net_profit
            - previous.net_profit
        )

        gross_margin_change = (
            current.gross_margin
            - previous.gross_margin
        )

        cogs_ratio_change = (
            current.cogs_ratio
            - previous.cogs_ratio
        )

        opex_ratio_change = (
            current.opex_ratio
            - previous.opex_ratio
        )

        drivers.append(
            self._driver(
                ProfitabilityDriverType.REVENUE,
                ProfitabilityMetric.REVENUE,
                revenue_change,
                current.revenue,
                "Revenue",
            )
        )

        drivers.append(
            self._driver(
                ProfitabilityDriverType.GROSS_MARGIN,
                ProfitabilityMetric.GROSS_MARGIN,
                gross_margin_change,
                current.revenue,
                "Gross margin",
                percentage_metric=True,
            )
        )

        drivers.append(
            self._driver(
                ProfitabilityDriverType.COGS,
                ProfitabilityMetric.COGS_RATIO,
                -cogs_ratio_change,
                current.revenue,
                "COGS efficiency",
                percentage_metric=True,
            )
        )

        drivers.append(
            self._driver(
                ProfitabilityDriverType.OPEX,
                ProfitabilityMetric.OPEX_RATIO,
                -opex_ratio_change,
                current.revenue,
                "Operating expense efficiency",
                percentage_metric=True,
            )
        )

        drivers.append(
            self._driver(
                ProfitabilityDriverType.SCALE,
                ProfitabilityMetric.OPERATING_PROFIT,
                operating_profit_change,
                current.revenue,
                "Operating profit",
            )
        )

        drivers.append(
            self._driver(
                ProfitabilityDriverType.OTHER,
                ProfitabilityMetric.NET_PROFIT,
                net_profit_change,
                current.revenue,
                "Net profit",
            )
        )

        # Avoid returning an arbitrary large list.
        drivers.sort(
            key=lambda item: abs(
                item.impact
            ),
            reverse=True,
        )

        return drivers

    def _driver(
        self,
        driver_type: ProfitabilityDriverType,
        metric: ProfitabilityMetric,
        impact: Decimal,
        revenue: Decimal,
        label: str,
        *,
        percentage_metric: bool = False,
    ) -> ProfitabilityDriver:

        if impact > ZERO:
            direction = "positive"
        elif impact < ZERO:
            direction = "negative"
        else:
            direction = "neutral"

        magnitude = abs(impact)

        if percentage_metric:
            impact_amount = round_money(
                revenue
                * (
                    magnitude
                    / ONE_HUNDRED
                )
            )
        else:
            impact_amount = round_money(
                impact
            )

        severity = (
            ProfitabilitySeverity.INFO
        )

        if magnitude >= Decimal("20"):
            severity = (
                ProfitabilitySeverity.HIGH
            )
        elif magnitude >= Decimal("10"):
            severity = (
                ProfitabilitySeverity.MEDIUM
            )
        elif magnitude >= Decimal("5"):
            severity = (
                ProfitabilitySeverity.LOW
            )

        if direction == "positive":
            explanation = (
                f"{label} improved profitability."
            )
        elif direction == "negative":
            explanation = (
                f"{label} reduced profitability."
            )
        else:
            explanation = (
                f"{label} was broadly stable."
            )

        return ProfitabilityDriver(
            driver_type=driver_type,
            metric=metric,
            impact=impact_amount,
            direction=direction,
            severity=severity,
            explanation=explanation,
            percentage_of_revenue=(
                percentage(
                    abs(impact_amount),
                    revenue,
                )
            ),
        )

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def generate_alerts(
        self,
        current: ProfitabilitySnapshot,
        previous: ProfitabilitySnapshot | None,
    ) -> list[ProfitabilityAlert]:

        alerts: list[
            ProfitabilityAlert
        ] = []

        if current.net_profit < ZERO:
            alerts.append(
                ProfitabilityAlert(
                    code="PROFITABILITY_LOSS",
                    severity=(
                        ProfitabilitySeverity.CRITICAL
                    ),
                    metric=(
                        ProfitabilityMetric.NET_PROFIT
                    ),
                    message=(
                        "The business is operating at a net loss."
                    ),
                    value=current.net_profit,
                )
            )

        if (
            current.operating_profit < ZERO
        ):
            alerts.append(
                ProfitabilityAlert(
                    code="OPERATING_LOSS",
                    severity=(
                        ProfitabilitySeverity.HIGH
                    ),
                    metric=(
                        ProfitabilityMetric.OPERATING_PROFIT
                    ),
                    message=(
                        "Core operating activities "
                        "are generating a loss."
                    ),
                    value=current.operating_profit,
                )
            )

        if (
            current.gross_margin
            <= ZERO
            and current.revenue > ZERO
        ):
            alerts.append(
                ProfitabilityAlert(
                    code="NEGATIVE_GROSS_MARGIN",
                    severity=(
                        ProfitabilitySeverity.CRITICAL
                    ),
                    metric=(
                        ProfitabilityMetric.GROSS_MARGIN
                    ),
                    message=(
                        "Cost of goods sold exceeds revenue."
                    ),
                    value=current.gross_margin,
                )
            )

        if (
            current.cogs_ratio
            >= self.high_cogs_ratio
        ):
            alerts.append(
                ProfitabilityAlert(
                    code="HIGH_COGS_RATIO",
                    severity=(
                        ProfitabilitySeverity.HIGH
                    ),
                    metric=(
                        ProfitabilityMetric.COGS_RATIO
                    ),
                    message=(
                        "COGS consumes a high proportion "
                        "of revenue."
                    ),
                    value=current.cogs_ratio,
                    threshold=(
                        self.high_cogs_ratio
                    ),
                )
            )

        if (
            current.opex_ratio
            >= self.high_opex_ratio
        ):
            alerts.append(
                ProfitabilityAlert(
                    code="HIGH_OPEX_RATIO",
                    severity=(
                        ProfitabilitySeverity.HIGH
                    ),
                    metric=(
                        ProfitabilityMetric.OPEX_RATIO
                    ),
                    message=(
                        "Operating expenses consume a high "
                        "proportion of revenue."
                    ),
                    value=current.opex_ratio,
                    threshold=(
                        self.high_opex_ratio
                    ),
                )
            )

        if previous is not None:

            net_margin_change = (
                current.net_margin
                - previous.net_margin
            )

            if (
                net_margin_change
                <= -self.margin_compression_warning
            ):
                alerts.append(
                    ProfitabilityAlert(
                        code="MARGIN_COMPRESSION",
                        severity=(
                            ProfitabilitySeverity.HIGH
                        ),
                        metric=(
                            ProfitabilityMetric.NET_MARGIN
                        ),
                        message=(
                            "Net margin has compressed "
                            "materially compared with "
                            "the previous period."
                        ),
                        value=net_margin_change,
                        threshold=(
                            -self.margin_compression_warning
                        ),
                    )
                )

            net_profit_change = percentage_change(
                current.net_profit,
                previous.net_profit,
            )

            if (
                net_profit_change is not None
                and net_profit_change
                <= -self.severe_decline
            ):
                alerts.append(
                    ProfitabilityAlert(
                        code="SEVERE_PROFIT_DECLINE",
                        severity=(
                            ProfitabilitySeverity.CRITICAL
                        ),
                        metric=(
                            ProfitabilityMetric.NET_PROFIT
                        ),
                        message=(
                            "Net profit has declined "
                            "severely versus the "
                            "previous period."
                        ),
                        value=net_profit_change,
                        threshold=(
                            -self.severe_decline
                        ),
                    )
                )

            gross_margin_change = (
                current.gross_margin
                - previous.gross_margin
            )

            if (
                gross_margin_change
                <= -self.margin_compression_warning
            ):
                alerts.append(
                    ProfitabilityAlert(
                        code="GROSS_MARGIN_COMPRESSION",
                        severity=(
                            ProfitabilitySeverity.HIGH
                        ),
                        metric=(
                            ProfitabilityMetric.GROSS_MARGIN
                        ),
                        message=(
                            "Gross margin has compressed "
                            "materially."
                        ),
                        value=gross_margin_change,
                        threshold=(
                            -self.margin_compression_warning
                        ),
                    )
                )

        return alerts

    # ------------------------------------------------------------------
    # Opportunity Detection
    # ------------------------------------------------------------------

    def identify_opportunities(
        self,
        current: ProfitabilitySnapshot,
        previous: ProfitabilitySnapshot | None = None,
    ) -> list[ProfitabilityOpportunity]:

        opportunities: list[
            ProfitabilityOpportunity
        ] = []

        # COGS optimization.
        if (
            current.cogs_ratio
            >= self.high_cogs_ratio
        ):
            target_ratio = (
                self.high_cogs_ratio
                - Decimal("5")
            )

            potential_savings = (
                current.revenue
                * (
                    current.cogs_ratio
                    - target_ratio
                )
                / ONE_HUNDRED
            )

            opportunities.append(
                ProfitabilityOpportunity(
                    opportunity_type=(
                        OpportunityType.COGS_REDUCTION
                    ),
                    title=(
                        "Reduce COGS ratio"
                    ),
                    description=(
                        "Reducing COGS as a percentage "
                        "of revenue could materially "
                        "improve gross and net profit."
                    ),
                    estimated_impact=round_money(
                        potential_savings
                    ),
                    impact_as_percentage_of_revenue=(
                        percentage(
                            potential_savings,
                            current.revenue,
                        )
                    ),
                    priority=(
                        ProfitabilitySeverity.HIGH
                    ),
                    assumptions={
                        "target_cogs_ratio": str(
                            target_ratio
                        ),
                        "revenue_held_constant": True,
                    },
                )
            )

        # OPEX optimization.
        if (
            current.opex_ratio
            >= self.high_opex_ratio
        ):
            target_ratio = (
                self.high_opex_ratio
                - Decimal("5")
            )

            potential_savings = (
                current.revenue
                * (
                    current.opex_ratio
                    - target_ratio
                )
                / ONE_HUNDRED
            )

            opportunities.append(
                ProfitabilityOpportunity(
                    opportunity_type=(
                        OpportunityType.OPEX_REDUCTION
                    ),
                    title=(
                        "Improve operating expense efficiency"
                    ),
                    description=(
                        "Reducing operating expenses "
                        "relative to revenue can "
                        "expand operating and net margins."
                    ),
                    estimated_impact=round_money(
                        potential_savings
                    ),
                    impact_as_percentage_of_revenue=(
                        percentage(
                            potential_savings,
                            current.revenue,
                        )
                    ),
                    priority=(
                        ProfitabilitySeverity.HIGH
                    ),
                    assumptions={
                        "target_opex_ratio": str(
                            target_ratio
                        ),
                        "revenue_held_constant": True,
                    },
                )
            )

        # Margin expansion.
        if (
            current.net_margin
            < self.healthy_net_margin
        ):
            target_margin = (
                self.healthy_net_margin
            )

            required_profit = (
                current.revenue
                * target_margin
                / ONE_HUNDRED
            )

            potential_improvement = (
                required_profit
                - current.net_profit
            )

            if potential_improvement > ZERO:
                opportunities.append(
                    ProfitabilityOpportunity(
                        opportunity_type=(
                            OpportunityType.MARGIN_EXPANSION
                        ),
                        title=(
                            "Expand net margin"
                        ),
                        description=(
                            "Closing the gap toward the "
                            "healthy net-margin benchmark "
                            "would materially strengthen "
                            "profitability."
                        ),
                        estimated_impact=round_money(
                            potential_improvement
                        ),
                        impact_as_percentage_of_revenue=(
                            percentage(
                                potential_improvement,
                                current.revenue,
                            )
                        ),
                        priority=(
                            ProfitabilitySeverity.MEDIUM
                        ),
                        assumptions={
                            "target_net_margin": str(
                                target_margin
                            ),
                            "revenue_held_constant": True,
                        },
                    )
                )

        # Revenue growth opportunity.
        if (
            current.net_margin > ZERO
            and current.net_margin
            < self.healthy_net_margin
        ):
            additional_revenue = (
                current.revenue
                * Decimal("10")
                / ONE_HUNDRED
            )

            incremental_profit = (
                additional_revenue
                * current.gross_margin
                / ONE_HUNDRED
            )

            opportunities.append(
                ProfitabilityOpportunity(
                    opportunity_type=(
                        OpportunityType.REVENUE_GROWTH
                    ),
                    title=(
                        "Increase profitable revenue"
                    ),
                    description=(
                        "A 10% revenue increase at the "
                        "current gross-margin level could "
                        "increase gross profit."
                    ),
                    estimated_impact=round_money(
                        incremental_profit
                    ),
                    impact_as_percentage_of_revenue=(
                        percentage(
                            incremental_profit,
                            current.revenue,
                        )
                    ),
                    priority=(
                        ProfitabilitySeverity.MEDIUM
                    ),
                    assumptions={
                        "revenue_growth": "10%",
                        "gross_margin_held_constant": True,
                    },
                )
            )

        opportunities.sort(
            key=lambda item: abs(
                item.estimated_impact
            ),
            reverse=True,
        )

        return opportunities

    # ------------------------------------------------------------------
    # What-If Analysis
    # ------------------------------------------------------------------

    def simulate(
        self,
        data: (
            ProfitabilityInput
            | ProfitabilitySnapshot
            | Mapping[str, Any]
        ),
        *,
        revenue_change_pct: Decimal = ZERO,
        cogs_change_pct: Decimal = ZERO,
        opex_change_pct: Decimal = ZERO,
        interest_change_pct: Decimal = ZERO,
        tax_change_pct: Decimal = ZERO,
    ) -> ProfitabilitySnapshot:

        base = self.build_snapshot(
            data
        )

        revenue_multiplier = (
            ONE
            + revenue_change_pct
            / ONE_HUNDRED
        )

        cogs_multiplier = (
            ONE
            + cogs_change_pct
            / ONE_HUNDRED
        )

        opex_multiplier = (
            ONE
            + opex_change_pct
            / ONE_HUNDRED
        )

        interest_multiplier = (
            ONE
            + interest_change_pct
            / ONE_HUNDRED
        )

        tax_multiplier = (
            ONE
            + tax_change_pct
            / ONE_HUNDRED
        )

        revenue = max(
            ZERO,
            base.revenue
            * revenue_multiplier,
        )

        cogs = max(
            ZERO,
            (
                base.revenue
                - base.gross_profit
            )
            * cogs_multiplier,
        )

        opex = max(
            ZERO,
            (
                base.gross_profit
                - base.operating_profit
            )
            * opex_multiplier,
        )

        interest = max(
            ZERO,
            (
                base.operating_profit
                - base.net_profit
            )
            * interest_multiplier,
        )

        # Tax is not recoverable exactly from a snapshot,
        # so this scenario treats the residual after interest
        # as the tax base.
        estimated_tax = max(
            ZERO,
            (
                base.operating_profit
                - interest
                - base.net_profit
            ),
        )

        tax = max(
            ZERO,
            estimated_tax
            * tax_multiplier,
        )

        gross_profit = (
            revenue - cogs
        )

        operating_profit = (
            gross_profit - opex
        )

        net_profit = (
            operating_profit
            - interest
            - tax
        )

        return ProfitabilitySnapshot(
            period=base.period,
            currency=base.currency,
            revenue=round_money(
                revenue
            ),
            gross_profit=round_money(
                gross_profit
            ),
            ebitda=round_money(
                operating_profit
            ),
            operating_profit=round_money(
                operating_profit
            ),
            net_profit=round_money(
                net_profit
            ),
            gross_margin=calculate_margin(
                gross_profit,
                revenue,
            ),
            ebitda_margin=calculate_margin(
                operating_profit,
                revenue,
            ),
            operating_margin=calculate_margin(
                operating_profit,
                revenue,
            ),
            net_margin=calculate_margin(
                net_profit,
                revenue,
            ),
            cogs_ratio=percentage(
                cogs,
                revenue,
            ),
            opex_ratio=percentage(
                opex,
                revenue,
            ),
            profit_conversion=calculate_margin(
                net_profit,
                revenue,
            ),
            cost_efficiency=(
                percentage(
                    revenue,
                    cogs + opex,
                )
                if cogs + opex > ZERO
                else ZERO
            ),
            metadata={
                **base.metadata,
                "scenario": True,
            },
        )

    # ------------------------------------------------------------------
    # Break-Even
    # ------------------------------------------------------------------

    def break_even_revenue(
        self,
        data: (
            ProfitabilityInput
            | ProfitabilitySnapshot
            | Mapping[str, Any]
        ),
        *,
        fixed_costs: Decimal | None = None,
    ) -> Decimal | None:

        snapshot = self.build_snapshot(
            data
        )

        if snapshot.revenue <= ZERO:
            return None

        cogs = (
            snapshot.revenue
            - snapshot.gross_profit
        )

        contribution_ratio = safe_divide(
            snapshot.revenue - cogs,
            snapshot.revenue,
        )

        if contribution_ratio <= ZERO:
            return None

        if fixed_costs is None:
            fixed_costs = (
                snapshot.gross_profit
                - snapshot.operating_profit
            )

        fixed_costs = abs(
            to_decimal(
                fixed_costs
            )
        )

        return round_money(
            safe_divide(
                fixed_costs,
                contribution_ratio,
            )
        )

    # ------------------------------------------------------------------
    # Margin of Safety
    # ------------------------------------------------------------------

    def margin_of_safety(
        self,
        data: (
            ProfitabilityInput
            | ProfitabilitySnapshot
            | Mapping[str, Any]
        ),
        *,
        fixed_costs: Decimal | None = None,
    ) -> Decimal | None:

        snapshot = self.build_snapshot(
            data
        )

        break_even = (
            self.break_even_revenue(
                snapshot,
                fixed_costs=fixed_costs,
            )
        )

        if (
            break_even is None
            or snapshot.revenue <= ZERO
        ):
            return None

        return round_percent(
            safe_divide(
                snapshot.revenue
                - break_even,
                snapshot.revenue,
            )
            * ONE_HUNDRED
        )

    # ------------------------------------------------------------------
    # Historical Analysis
    # ------------------------------------------------------------------

    def analyze_history(
        self,
        history: Sequence[
            ProfitabilityInput
            | ProfitabilitySnapshot
            | Mapping[str, Any]
        ],
    ) -> dict[str, Any]:

        snapshots = [
            self.build_snapshot(
                item
            )
            for item in history
        ]

        if not snapshots:
            return {
                "periods": [],
                "trend": (
                    ProfitabilityTrend
                    .UNKNOWN
                    .value
                ),
                "latest": None,
            }

        comparisons: list[
            ProfitabilityComparison
        ] = []

        for index in range(
            1,
            len(snapshots),
        ):
            comparisons.extend(
                self.compare(
                    snapshots[index],
                    snapshots[index - 1],
                )
            )

        net_profit_series = [
            item.net_profit
            for item in snapshots
        ]

        net_margin_series = [
            item.net_margin
            for item in snapshots
        ]

        return {
            "periods": [
                item.to_dict()
                for item in snapshots
            ],
            "trend": (
                self.history_trend(
                    net_profit_series
                ).value
            ),
            "latest": (
                snapshots[-1].to_dict()
            ),
            "net_profit_series": [
                str(value)
                for value in net_profit_series
            ],
            "net_margin_series": [
                str(value)
                for value in net_margin_series
            ],
            "comparisons": [
                item.to_dict()
                for item in comparisons
            ],
        }

    def history_trend(
        self,
        values: Sequence[Decimal],
    ) -> ProfitabilityTrend:

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
            >= self.strong_improvement
        ):
            return (
                ProfitabilityTrend
                .STRONG_IMPROVEMENT
            )

        if (
            average_change
            >= self.improvement
        ):
            return (
                ProfitabilityTrend.IMPROVEMENT
            )

        if (
            average_change
            <= -self.severe_decline
        ):
            return (
                ProfitabilityTrend
                .SEVERE_DECLINE
            )

        if (
            average_change
            <= -self.decline
        ):
            return (
                ProfitabilityTrend.DECLINE
            )

        return ProfitabilityTrend.STABLE

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def generate_summary(
        self,
        snapshot: ProfitabilitySnapshot,
        level: ProfitabilityLevel,
        trend: ProfitabilityTrend,
        drivers: Sequence[ProfitabilityDriver],
        alerts: Sequence[ProfitabilityAlert],
        opportunities: Sequence[
            ProfitabilityOpportunity
        ],
    ) -> str:

        summary = (
            f"Revenue was "
            f"{snapshot.revenue:,.2f}, "
            f"generating net profit of "
            f"{snapshot.net_profit:,.2f} "
            f"at a net margin of "
            f"{snapshot.net_margin:.2f}%. "
            f"Profitability is classified as "
            f"{level.value} with a "
            f"{trend.value} trend."
        )

        if drivers:
            driver = drivers[0]

            summary += (
                f" The strongest detected driver "
                f"was {driver.metric.value}, "
                f"with a {driver.direction} impact."
            )

        if alerts:
            highest = max(
                alerts,
                key=self._alert_rank,
            )

            summary += (
                f" The highest-priority signal "
                f"is {highest.code}."
            )

        if opportunities:
            opportunity = opportunities[0]

            summary += (
                f" The largest modeled improvement "
                f"opportunity is "
                f"{opportunity.title.lower()}, "
                f"estimated at "
                f"{opportunity.estimated_impact:,.2f}."
            )

        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def explain_comparison(
        metric: ProfitabilityMetric,
        change: Decimal,
        pct_change: Decimal | None,
    ) -> str:

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
    def _alert_rank(
        alert: ProfitabilityAlert,
    ) -> int:

        return {
            ProfitabilitySeverity.INFO: 1,
            ProfitabilitySeverity.LOW: 2,
            ProfitabilitySeverity.MEDIUM: 3,
            ProfitabilitySeverity.HIGH: 4,
            ProfitabilitySeverity.CRITICAL: 5,
        }[alert.severity]


# ============================================================================
# Convenience Functions
# ============================================================================


def analyze_profitability(
    current: (
        ProfitabilityInput
        | ProfitabilitySnapshot
        | Mapping[str, Any]
    ),
    previous: (
        ProfitabilityInput
        | ProfitabilitySnapshot
        | Mapping[str, Any]
        | None
    ) = None,
) -> ProfitabilityAnalysisResult:
    """Perform complete profitability analysis."""

    return ProfitabilityAnalyzer().analyze(
        current,
        previous,
    )


def calculate_profitability_snapshot(
    data: (
        ProfitabilityInput
        | ProfitabilitySnapshot
        | Mapping[str, Any]
    ),
) -> ProfitabilitySnapshot:
    """Build a profitability snapshot."""

    return ProfitabilityAnalyzer().build_snapshot(
        data
    )


def calculate_profitability_level(
    data: (
        ProfitabilityInput
        | ProfitabilitySnapshot
        | Mapping[str, Any]
    ),
) -> ProfitabilityLevel:
    """Classify profitability health."""

    analyzer = ProfitabilityAnalyzer()

    snapshot = analyzer.build_snapshot(
        data
    )

    return analyzer.classify_level(
        snapshot
    )


def calculate_profitability_trend(
    current: (
        ProfitabilityInput
        | ProfitabilitySnapshot
        | Mapping[str, Any]
    ),
    previous: (
        ProfitabilityInput
        | ProfitabilitySnapshot
        | Mapping[str, Any]
        | None
    ) = None,
) -> ProfitabilityTrend:
    """Determine profitability trend."""

    analyzer = ProfitabilityAnalyzer()

    current_snapshot = (
        analyzer.build_snapshot(
            current
        )
    )

    previous_snapshot = (
        analyzer.build_snapshot(
            previous
        )
        if previous is not None
        else None
    )

    return analyzer.determine_trend(
        current_snapshot,
        previous_snapshot,
    )


def calculate_profitability_margin(
    profit: Any,
    revenue: Any,
) -> Decimal:
    """Calculate profitability margin."""

    return calculate_margin(
        to_decimal(profit),
        to_decimal(revenue),
    )


def calculate_profit_conversion(
    net_profit: Any,
    revenue: Any,
) -> Decimal:
    """
    Calculate revenue-to-net-profit conversion.

    Equivalent to net profit margin.
    """

    return calculate_margin(
        to_decimal(net_profit),
        to_decimal(revenue),
    )


def calculate_break_even_revenue(
    data: (
        ProfitabilityInput
        | ProfitabilitySnapshot
        | Mapping[str, Any]
    ),
    *,
    fixed_costs: Any = None,
) -> Decimal | None:
    """Calculate break-even revenue."""

    return ProfitabilityAnalyzer().break_even_revenue(
        data,
        fixed_costs=(
            to_decimal(fixed_costs)
            if fixed_costs is not None
            else None
        ),
    )


def calculate_margin_of_safety(
    data: (
        ProfitabilityInput
        | ProfitabilitySnapshot
        | Mapping[str, Any]
    ),
    *,
    fixed_costs: Any = None,
) -> Decimal | None:
    """Calculate margin of safety."""

    return ProfitabilityAnalyzer().margin_of_safety(
        data,
        fixed_costs=(
            to_decimal(fixed_costs)
            if fixed_costs is not None
            else None
        ),
    )


def simulate_profitability(
    data: (
        ProfitabilityInput
        | ProfitabilitySnapshot
        | Mapping[str, Any]
    ),
    *,
    revenue_change_pct: Any = ZERO,
    cogs_change_pct: Any = ZERO,
    opex_change_pct: Any = ZERO,
    interest_change_pct: Any = ZERO,
    tax_change_pct: Any = ZERO,
) -> ProfitabilitySnapshot:
    """Run a profitability what-if scenario."""

    analyzer = ProfitabilityAnalyzer()

    return analyzer.simulate(
        data,
        revenue_change_pct=to_decimal(
            revenue_change_pct
        ),
        cogs_change_pct=to_decimal(
            cogs_change_pct
        ),
        opex_change_pct=to_decimal(
            opex_change_pct
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
    "percentage",
    "percentage_change",
    "calculate_margin",

    # Enums
    "ProfitabilityLevel",
    "ProfitabilityTrend",
    "ProfitabilityMetric",
    "ProfitabilityDriverType",
    "ProfitabilitySeverity",
    "OpportunityType",

    # Models
    "ProfitabilityInput",
    "ProfitabilitySnapshot",
    "ProfitabilityComparison",
    "ProfitabilityDriver",
    "ProfitabilityAlert",
    "ProfitabilityOpportunity",
    "ProfitabilityAnalysisResult",

    # Analyzer
    "ProfitabilityAnalyzer",

    # Convenience functions
    "analyze_profitability",
    "calculate_profitability_snapshot",
    "calculate_profitability_level",
    "calculate_profitability_trend",
    "calculate_profitability_margin",
    "calculate_profit_conversion",
    "calculate_break_even_revenue",
    "calculate_margin_of_safety",
    "simulate_profitability",
]