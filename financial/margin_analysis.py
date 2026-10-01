"""
FinCo AI - Margin Analysis Engine

File:
    backend/app/financial/margin_analysis.py

Purpose:
    Analyze business margins and identify profitability expansion,
    compression, and root-cause drivers.

Responsibilities:
    - Calculate gross, operating, and net margins.
    - Analyze margin changes across periods.
    - Decompose margin movement into revenue, COGS, and OPEX effects.
    - Analyze product/category/customer/segment margins.
    - Detect margin compression and expansion.
    - Identify low-margin and high-margin areas.
    - Calculate contribution margin.
    - Estimate margin improvement opportunities.
    - Support alerts, recommendations, forecasting, what-if analysis,
      reports, and supervisor-agent workflows.

Architecture:

    Financial Data
          ↓
    Revenue + Cost Data
          ↓
    margin_analysis.py
          ↓
    ┌────────────────────────────────────┐
    │ Gross Margin                       │
    │ Operating Margin                   │
    │ Net Margin                         │
    │ Contribution Margin                │
    │ Margin Change                      │
    │ Margin Drivers                     │
    │ Segment Margins                    │
    │ Margin Opportunities               │
    └────────────────────────────────────┘
          ↓
    Health Score / Alerts / Root Cause
          ↓
    Forecast / What-If / Recommendations
          ↓
    Supervisor Agent
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

    try:
        return Decimal(str(value))
    except (TypeError, ValueError, ArithmeticError):
        return default


def round_money(value: Decimal) -> Decimal:
    """Round monetary values."""
    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_percent(value: Decimal) -> Decimal:
    """Round percentage values."""
    return value.quantize(
        PERCENT_QUANT,
        rounding=ROUND_HALF_UP,
    )


def safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely divide two values."""
    if denominator == ZERO:
        return default

    return numerator / denominator


def calculate_margin(
    profit: Decimal,
    revenue: Decimal,
) -> Decimal:
    """Calculate a profit margin as a percentage."""
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
) -> Decimal:
    """Calculate percentage change."""
    if previous == ZERO:
        if current == ZERO:
            return ZERO

        return ONE_HUNDRED

    return round_percent(
        (
            (current - previous)
            / abs(previous)
        )
        * ONE_HUNDRED
    )


# ============================================================================
# Enums
# ============================================================================


class MarginType(str, Enum):
    """Supported margin types."""

    GROSS = "gross"
    CONTRIBUTION = "contribution"
    OPERATING = "operating"
    NET = "net"


class MarginTrend(str, Enum):
    """Direction of margin movement."""

    EXPANDING = "expanding"
    STABLE = "stable"
    COMPRESSING = "compressing"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class MarginHealth(str, Enum):
    """Margin health classification."""

    EXCELLENT = "excellent"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class MarginDriverType(str, Enum):
    """Types of factors that can move margins."""

    REVENUE = "revenue"
    PRICE = "price"
    VOLUME = "volume"
    COGS = "cogs"
    OPEX = "opex"
    MIX = "mix"
    DISCOUNT = "discount"
    OTHER = "other"


class MarginSeverity(str, Enum):
    """Severity of a margin event."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Input Models
# ============================================================================


@dataclass(slots=True)
class MarginInput:
    """
    Normalized financial data for margin analysis.
    """

    revenue: Decimal = ZERO

    cost_of_goods_sold: Decimal = ZERO

    variable_costs: Decimal = ZERO

    fixed_costs: Decimal = ZERO

    operating_expenses: Decimal = ZERO

    interest_expense: Decimal = ZERO

    tax_expense: Decimal = ZERO

    gross_profit: Decimal | None = None

    contribution_profit: Decimal | None = None

    operating_profit: Decimal | None = None

    net_profit: Decimal | None = None

    discounts: Decimal = ZERO

    returns: Decimal = ZERO

    units_sold: Decimal = ZERO

    customers: Decimal = ZERO

    orders: Decimal = ZERO

    currency: str | None = None

    period: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def derive(self) -> None:
        """Derive common profitability measures."""

        if self.gross_profit is None:
            self.gross_profit = (
                self.revenue
                - self.cost_of_goods_sold
            )

        if self.contribution_profit is None:
            self.contribution_profit = (
                self.revenue
                - self.variable_costs
            )

        if self.operating_profit is None:
            self.operating_profit = (
                self.gross_profit
                - self.operating_expenses
            )

        if self.net_profit is None:
            self.net_profit = (
                self.operating_profit
                - self.interest_expense
                - self.tax_expense
            )


@dataclass(slots=True)
class MarginResult:
    """Result for a single margin metric."""

    margin_type: MarginType

    value: Decimal

    revenue: Decimal

    profit: Decimal

    health: MarginHealth = MarginHealth.UNKNOWN

    trend: MarginTrend = MarginTrend.UNKNOWN

    previous_value: Decimal | None = None

    change_points: Decimal | None = None

    percentage_change: Decimal | None = None

    description: str = ""

    interpretation: str = ""

    evidence: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "margin_type": self.margin_type.value,
            "value": str(self.value),
            "revenue": str(self.revenue),
            "profit": str(self.profit),
            "health": self.health.value,
            "trend": self.trend.value,
            "previous_value": (
                str(self.previous_value)
                if self.previous_value is not None
                else None
            ),
            "change_points": (
                str(self.change_points)
                if self.change_points is not None
                else None
            ),
            "percentage_change": (
                str(self.percentage_change)
                if self.percentage_change is not None
                else None
            ),
            "description": self.description,
            "interpretation": self.interpretation,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MarginDriver:
    """Represents a factor influencing margin movement."""

    driver_type: MarginDriverType

    impact: Decimal

    direction: str

    description: str

    severity: MarginSeverity = MarginSeverity.INFO

    evidence: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "driver_type": self.driver_type.value,
            "impact": str(self.impact),
            "direction": self.direction,
            "description": self.description,
            "severity": self.severity.value,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SegmentMargin:
    """Margin analysis for a product, customer, segment, or category."""

    name: str

    revenue: Decimal

    cost: Decimal

    profit: Decimal

    margin: Decimal

    revenue_share: Decimal = ZERO

    profit_share: Decimal = ZERO

    rank: int | None = None

    health: MarginHealth = MarginHealth.UNKNOWN

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "revenue": str(self.revenue),
            "cost": str(self.cost),
            "profit": str(self.profit),
            "margin": str(self.margin),
            "revenue_share": str(self.revenue_share),
            "profit_share": str(self.profit_share),
            "rank": self.rank,
            "health": self.health.value,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MarginOpportunity:
    """Potential margin-improvement opportunity."""

    name: str

    opportunity_type: MarginDriverType

    current_value: Decimal

    target_value: Decimal

    estimated_improvement: Decimal

    estimated_profit_impact: Decimal

    priority: MarginSeverity

    description: str

    recommendation: str

    evidence: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "opportunity_type": (
                self.opportunity_type.value
            ),
            "current_value": str(
                self.current_value
            ),
            "target_value": str(
                self.target_value
            ),
            "estimated_improvement": str(
                self.estimated_improvement
            ),
            "estimated_profit_impact": str(
                self.estimated_profit_impact
            ),
            "priority": self.priority.value,
            "description": self.description,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class MarginAnalysisResult:
    """Complete margin-analysis result."""

    gross_margin: MarginResult | None = None

    contribution_margin: MarginResult | None = None

    operating_margin: MarginResult | None = None

    net_margin: MarginResult | None = None

    drivers: list[MarginDriver] = field(
        default_factory=list
    )

    opportunities: list[MarginOpportunity] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    strongest_segments: list[SegmentMargin] = field(
        default_factory=list
    )

    weakest_segments: list[SegmentMargin] = field(
        default_factory=list
    )

    overall_trend: MarginTrend = MarginTrend.UNKNOWN

    summary: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gross_margin": (
                self.gross_margin.to_dict()
                if self.gross_margin
                else None
            ),
            "contribution_margin": (
                self.contribution_margin.to_dict()
                if self.contribution_margin
                else None
            ),
            "operating_margin": (
                self.operating_margin.to_dict()
                if self.operating_margin
                else None
            ),
            "net_margin": (
                self.net_margin.to_dict()
                if self.net_margin
                else None
            ),
            "drivers": [
                driver.to_dict()
                for driver in self.drivers
            ],
            "opportunities": [
                opportunity.to_dict()
                for opportunity in self.opportunities
            ],
            "warnings": self.warnings,
            "strongest_segments": [
                segment.to_dict()
                for segment in self.strongest_segments
            ],
            "weakest_segments": [
                segment.to_dict()
                for segment in self.weakest_segments
            ],
            "overall_trend": (
                self.overall_trend.value
            ),
            "summary": self.summary,
            "metadata": self.metadata,
        }


# ============================================================================
# Margin Analyzer
# ============================================================================


class MarginAnalyzer:
    """
    Stateless margin intelligence engine.
    """

    def __init__(
        self,
        *,
        stable_threshold: Decimal | float = Decimal("1"),
        warning_margin: Decimal | float = Decimal("5"),
        healthy_margin: Decimal | float = Decimal("10"),
        excellent_margin: Decimal | float = Decimal("20"),
    ) -> None:
        self.stable_threshold = to_decimal(
            stable_threshold
        )

        self.warning_margin = to_decimal(
            warning_margin
        )

        self.healthy_margin = to_decimal(
            healthy_margin
        )

        self.excellent_margin = to_decimal(
            excellent_margin
        )

    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        current: MarginInput | Mapping[str, Any],
        *,
        previous: MarginInput | Mapping[str, Any] | None = None,
        segments: Sequence[
            Mapping[str, Any]
        ]
        | None = None,
    ) -> MarginAnalysisResult:
        """
        Perform complete margin analysis.
        """

        current_input = self.normalize_input(
            current
        )
        current_input.derive()

        previous_input = None

        if previous is not None:
            previous_input = self.normalize_input(
                previous
            )
            previous_input.derive()

        gross = self.gross_margin(
            current_input,
            previous_input,
        )

        contribution = self.contribution_margin(
            current_input,
            previous_input,
        )

        operating = self.operating_margin(
            current_input,
            previous_input,
        )

        net = self.net_margin(
            current_input,
            previous_input,
        )

        drivers = self.analyze_drivers(
            current_input,
            previous_input,
        )

        opportunities = (
            self.identify_opportunities(
                current_input,
                gross,
                contribution,
                operating,
                net,
            )
        )

        segment_results: list[
            SegmentMargin
        ] = []

        if segments:
            segment_results = self.analyze_segments(
                segments
            )

        strongest = sorted(
            segment_results,
            key=lambda item: item.margin,
            reverse=True,
        )[:5]

        weakest = sorted(
            segment_results,
            key=lambda item: item.margin,
        )[:5]

        warnings = self.generate_warnings(
            gross,
            contribution,
            operating,
            net,
            drivers,
        )

        overall_trend = self.determine_overall_trend(
            [
                result
                for result in (
                    gross,
                    contribution,
                    operating,
                    net,
                )
                if result is not None
            ]
        )

        summary = self.generate_summary(
            gross,
            contribution,
            operating,
            net,
            overall_trend,
            drivers,
        )

        return MarginAnalysisResult(
            gross_margin=gross,
            contribution_margin=contribution,
            operating_margin=operating,
            net_margin=net,
            drivers=drivers,
            opportunities=opportunities,
            warnings=warnings,
            strongest_segments=strongest,
            weakest_segments=weakest,
            overall_trend=overall_trend,
            summary=summary,
            metadata={
                "currency": current_input.currency,
                "period": current_input.period,
                "segment_count": len(
                    segment_results
                ),
            },
        )

    # ------------------------------------------------------------------
    # Core Margins
    # ------------------------------------------------------------------

    def gross_margin(
        self,
        current: MarginInput,
        previous: MarginInput | None = None,
    ) -> MarginResult | None:
        if current.revenue == ZERO:
            return None

        profit = (
            current.gross_profit
            if current.gross_profit is not None
            else (
                current.revenue
                - current.cost_of_goods_sold
            )
        )

        value = calculate_margin(
            profit,
            current.revenue,
        )

        previous_value = None

        if previous is not None:
            previous_value = calculate_margin(
                previous.gross_profit or (
                    previous.revenue
                    - previous.cost_of_goods_sold
                ),
                previous.revenue,
            )

        return self._build_margin_result(
            margin_type=MarginType.GROSS,
            value=value,
            revenue=current.revenue,
            profit=profit,
            previous_value=previous_value,
            description=(
                "Gross profit retained after direct cost of goods sold."
            ),
            interpretation=(
                "Gross margin measures the economics of the core "
                "products or services before operating overhead."
            ),
        )

    def contribution_margin(
        self,
        current: MarginInput,
        previous: MarginInput | None = None,
    ) -> MarginResult | None:
        if current.revenue == ZERO:
            return None

        profit = (
            current.contribution_profit
            if current.contribution_profit is not None
            else (
                current.revenue
                - current.variable_costs
            )
        )

        value = calculate_margin(
            profit,
            current.revenue,
        )

        previous_value = None

        if previous is not None:
            previous_profit = (
                previous.contribution_profit
                if previous.contribution_profit is not None
                else (
                    previous.revenue
                    - previous.variable_costs
                )
            )

            previous_value = calculate_margin(
                previous_profit,
                previous.revenue,
            )

        return self._build_margin_result(
            margin_type=MarginType.CONTRIBUTION,
            value=value,
            revenue=current.revenue,
            profit=profit,
            previous_value=previous_value,
            description=(
                "Revenue remaining after variable costs."
            ),
            interpretation=(
                "Contribution margin shows how much revenue remains "
                "to cover fixed costs and generate operating profit."
            ),
        )

    def operating_margin(
        self,
        current: MarginInput,
        previous: MarginInput | None = None,
    ) -> MarginResult | None:
        if current.revenue == ZERO:
            return None

        profit = (
            current.operating_profit
            if current.operating_profit is not None
            else (
                current.gross_profit or ZERO
            )
            - current.operating_expenses
        )

        value = calculate_margin(
            profit,
            current.revenue,
        )

        previous_value = None

        if previous is not None:
            previous_profit = (
                previous.operating_profit
                if previous.operating_profit is not None
                else (
                    previous.gross_profit or ZERO
                )
                - previous.operating_expenses
            )

            previous_value = calculate_margin(
                previous_profit,
                previous.revenue,
            )

        return self._build_margin_result(
            margin_type=MarginType.OPERATING,
            value=value,
            revenue=current.revenue,
            profit=profit,
            previous_value=previous_value,
            description=(
                "Operating profit retained after operating expenses."
            ),
            interpretation=(
                "Operating margin measures the profitability of "
                "the core business after operating overhead."
            ),
        )

    def net_margin(
        self,
        current: MarginInput,
        previous: MarginInput | None = None,
    ) -> MarginResult | None:
        if current.revenue == ZERO:
            return None

        profit = (
            current.net_profit
            if current.net_profit is not None
            else (
                current.operating_profit or ZERO
            )
            - current.interest_expense
            - current.tax_expense
        )

        value = calculate_margin(
            profit,
            current.revenue,
        )

        previous_value = None

        if previous is not None:
            previous_profit = (
                previous.net_profit
                if previous.net_profit is not None
                else (
                    previous.operating_profit or ZERO
                )
                - previous.interest_expense
                - previous.tax_expense
            )

            previous_value = calculate_margin(
                previous_profit,
                previous.revenue,
            )

        return self._build_margin_result(
            margin_type=MarginType.NET,
            value=value,
            revenue=current.revenue,
            profit=profit,
            previous_value=previous_value,
            description=(
                "Net profit retained after all expenses."
            ),
            interpretation=(
                "Net margin represents the final percentage of "
                "revenue retained as profit."
            ),
        )

    # ------------------------------------------------------------------
    # Margin Construction
    # ------------------------------------------------------------------

    def _build_margin_result(
        self,
        *,
        margin_type: MarginType,
        value: Decimal,
        revenue: Decimal,
        profit: Decimal,
        previous_value: Decimal | None,
        description: str,
        interpretation: str,
    ) -> MarginResult:
        change_points = None
        pct_change = None
        trend = MarginTrend.UNKNOWN

        if previous_value is not None:
            change_points = round_percent(
                value - previous_value
            )

            pct_change = percentage_change(
                value,
                previous_value,
            )

            trend = self.determine_trend(
                change_points
            )

        health = self.classify_health(
            value
        )

        evidence = [
            f"{margin_type.value} margin = {value}%",
            f"Revenue = {round_money(revenue)}",
            f"Profit = {round_money(profit)}",
        ]

        if previous_value is not None:
            evidence.append(
                f"Previous margin = {previous_value}%"
            )
            evidence.append(
                f"Margin change = {change_points} percentage points"
            )

        return MarginResult(
            margin_type=margin_type,
            value=value,
            revenue=round_money(revenue),
            profit=round_money(profit),
            health=health,
            trend=trend,
            previous_value=previous_value,
            change_points=change_points,
            percentage_change=pct_change,
            description=description,
            interpretation=interpretation,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def classify_health(
        self,
        margin: Decimal,
    ) -> MarginHealth:
        if margin >= self.excellent_margin:
            return MarginHealth.EXCELLENT

        if margin >= self.healthy_margin:
            return MarginHealth.HEALTHY

        if margin >= self.warning_margin:
            return MarginHealth.WARNING

        if margin >= ZERO:
            return MarginHealth.CRITICAL

        return MarginHealth.NEGATIVE

    def determine_trend(
        self,
        change_points: Decimal,
    ) -> MarginTrend:
        if abs(change_points) <= self.stable_threshold:
            return MarginTrend.STABLE

        if change_points > self.stable_threshold:
            return MarginTrend.EXPANDING

        if change_points < -self.stable_threshold:
            return MarginTrend.COMPRESSING

        return MarginTrend.UNKNOWN

    def determine_overall_trend(
        self,
        margins: Sequence[MarginResult],
    ) -> MarginTrend:
        if not margins:
            return MarginTrend.UNKNOWN

        expanding = sum(
            1
            for margin in margins
            if margin.trend
            == MarginTrend.EXPANDING
        )

        compressing = sum(
            1
            for margin in margins
            if margin.trend
            == MarginTrend.COMPRESSING
        )

        if expanding > compressing:
            return MarginTrend.EXPANDING

        if compressing > expanding:
            return MarginTrend.COMPRESSING

        if expanding == compressing:
            return MarginTrend.STABLE

        return MarginTrend.UNKNOWN

    # ------------------------------------------------------------------
    # Driver Analysis
    # ------------------------------------------------------------------

    def analyze_drivers(
        self,
        current: MarginInput,
        previous: MarginInput | None,
    ) -> list[MarginDriver]:
        """
        Identify likely drivers of margin movement.

        This is deterministic financial decomposition. More sophisticated
        causal attribution can later be layered on top using ML or the
        supervisor agent.
        """

        if previous is None:
            return []

        drivers: list[MarginDriver] = []

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

        revenue_growth = percentage_change(
            current.revenue,
            previous.revenue,
        )

        cogs_growth = percentage_change(
            current.cost_of_goods_sold,
            previous.cost_of_goods_sold,
        )

        opex_growth = percentage_change(
            current.operating_expenses,
            previous.operating_expenses,
        )

        # Revenue driver
        if revenue_change != ZERO:
            direction = (
                "positive"
                if revenue_change > ZERO
                else "negative"
            )

            drivers.append(
                MarginDriver(
                    driver_type=MarginDriverType.REVENUE,
                    impact=round_money(
                        revenue_change
                    ),
                    direction=direction,
                    description=(
                        "Revenue changed relative to the previous period."
                    ),
                    severity=(
                        MarginSeverity.LOW
                        if abs(revenue_growth) < Decimal("10")
                        else MarginSeverity.MEDIUM
                    ),
                    evidence=[
                        f"Revenue change = {round_money(revenue_change)}",
                        f"Revenue growth = {revenue_growth}%",
                    ],
                )
            )

        # COGS driver
        if cogs_change != ZERO:
            direction = (
                "negative"
                if cogs_change > ZERO
                else "positive"
            )

            severity = (
                MarginSeverity.HIGH
                if cogs_growth > Decimal("15")
                else MarginSeverity.MEDIUM
                if cogs_growth > Decimal("5")
                else MarginSeverity.LOW
            )

            drivers.append(
                MarginDriver(
                    driver_type=MarginDriverType.COGS,
                    impact=round_money(
                        cogs_change
                    ),
                    direction=direction,
                    description=(
                        "Direct costs changed relative to the previous period."
                    ),
                    severity=severity,
                    evidence=[
                        f"COGS change = {round_money(cogs_change)}",
                        f"COGS growth = {cogs_growth}%",
                    ],
                )
            )

        # OPEX driver
        if opex_change != ZERO:
            direction = (
                "negative"
                if opex_change > ZERO
                else "positive"
            )

            severity = (
                MarginSeverity.HIGH
                if opex_growth > Decimal("15")
                else MarginSeverity.MEDIUM
                if opex_growth > Decimal("5")
                else MarginSeverity.LOW
            )

            drivers.append(
                MarginDriver(
                    driver_type=MarginDriverType.OPEX,
                    impact=round_money(
                        opex_change
                    ),
                    direction=direction,
                    description=(
                        "Operating expenses changed relative to the previous period."
                    ),
                    severity=severity,
                    evidence=[
                        f"OPEX change = {round_money(opex_change)}",
                        f"OPEX growth = {opex_growth}%",
                    ],
                )
            )

        # Discount driver
        discount_change = (
            current.discounts
            - previous.discounts
        )

        if discount_change != ZERO:
            direction = (
                "negative"
                if discount_change > ZERO
                else "positive"
            )

            drivers.append(
                MarginDriver(
                    driver_type=MarginDriverType.DISCOUNT,
                    impact=round_money(
                        discount_change
                    ),
                    direction=direction,
                    description=(
                        "Discount levels changed and may have affected realized margins."
                    ),
                    severity=MarginSeverity.MEDIUM,
                    evidence=[
                        f"Discount change = {round_money(discount_change)}",
                    ],
                )
            )

        return drivers

    # ------------------------------------------------------------------
    # Margin Compression / Expansion
    # ------------------------------------------------------------------

    def detect_margin_compression(
        self,
        current: MarginInput | Mapping[str, Any],
        previous: MarginInput | Mapping[str, Any],
        *,
        threshold_points: Decimal | float = Decimal("2"),
    ) -> list[MarginResult]:
        """
        Detect material margin compression.
        """
        current_input = self.normalize_input(
            current
        )
        current_input.derive()

        previous_input = self.normalize_input(
            previous
        )
        previous_input.derive()

        results = [
            self.gross_margin(
                current_input,
                previous_input,
            ),
            self.contribution_margin(
                current_input,
                previous_input,
            ),
            self.operating_margin(
                current_input,
                previous_input,
            ),
            self.net_margin(
                current_input,
                previous_input,
            ),
        ]

        threshold = to_decimal(
            threshold_points
        )

        return [
            result
            for result in results
            if result is not None
            and result.change_points is not None
            and result.change_points <= -threshold
        ]

    def detect_margin_expansion(
        self,
        current: MarginInput | Mapping[str, Any],
        previous: MarginInput | Mapping[str, Any],
        *,
        threshold_points: Decimal | float = Decimal("2"),
    ) -> list[MarginResult]:
        """
        Detect material margin expansion.
        """
        current_input = self.normalize_input(
            current
        )
        current_input.derive()

        previous_input = self.normalize_input(
            previous
        )
        previous_input.derive()

        results = [
            self.gross_margin(
                current_input,
                previous_input,
            ),
            self.contribution_margin(
                current_input,
                previous_input,
            ),
            self.operating_margin(
                current_input,
                previous_input,
            ),
            self.net_margin(
                current_input,
                previous_input,
            ),
        ]

        threshold = to_decimal(
            threshold_points
        )

        return [
            result
            for result in results
            if result is not None
            and result.change_points is not None
            and result.change_points >= threshold
        ]

    # ------------------------------------------------------------------
    # Segment Analysis
    # ------------------------------------------------------------------

    def analyze_segments(
        self,
        segments: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[SegmentMargin]:
        """
        Analyze margins by segment/product/category/customer.

        Expected fields:
            name
            revenue
            cost

        Optional:
            metadata
        """

        results: list[SegmentMargin] = []

        total_revenue = sum(
            (
                to_decimal(
                    segment.get("revenue")
                )
                for segment in segments
            ),
            ZERO,
        )

        total_profit = sum(
            (
                to_decimal(
                    segment.get("revenue")
                )
                - to_decimal(
                    segment.get("cost")
                )
                for segment in segments
            ),
            ZERO,
        )

        for segment in segments:
            name = str(
                segment.get(
                    "name",
                    "Unknown",
                )
            )

            revenue = to_decimal(
                segment.get("revenue")
            )

            cost = to_decimal(
                segment.get("cost")
            )

            profit = revenue - cost

            margin = calculate_margin(
                profit,
                revenue,
            )

            revenue_share = ZERO

            if total_revenue != ZERO:
                revenue_share = round_percent(
                    safe_divide(
                        revenue,
                        total_revenue,
                    )
                    * ONE_HUNDRED
                )

            profit_share = ZERO

            if total_profit != ZERO:
                profit_share = round_percent(
                    safe_divide(
                        profit,
                        total_profit,
                    )
                    * ONE_HUNDRED
                )

            results.append(
                SegmentMargin(
                    name=name,
                    revenue=round_money(
                        revenue
                    ),
                    cost=round_money(
                        cost
                    ),
                    profit=round_money(
                        profit
                    ),
                    margin=margin,
                    revenue_share=revenue_share,
                    profit_share=profit_share,
                    health=self.classify_health(
                        margin
                    ),
                    metadata=dict(
                        segment.get(
                            "metadata",
                            {},
                        )
                    ),
                )
            )

        results.sort(
            key=lambda item: item.margin,
            reverse=True,
        )

        for index, result in enumerate(
            results,
            start=1,
        ):
            result.rank = index

        return results

    def top_margin_segments(
        self,
        segments: Sequence[
            SegmentMargin
        ],
        limit: int = 5,
    ) -> list[SegmentMargin]:
        """Return highest-margin segments."""
        return sorted(
            segments,
            key=lambda item: item.margin,
            reverse=True,
        )[:limit]

    def bottom_margin_segments(
        self,
        segments: Sequence[
            SegmentMargin
        ],
        limit: int = 5,
    ) -> list[SegmentMargin]:
        """Return lowest-margin segments."""
        return sorted(
            segments,
            key=lambda item: item.margin,
        )[:limit]

    # ------------------------------------------------------------------
    # Opportunity Analysis
    # ------------------------------------------------------------------

    def identify_opportunities(
        self,
        data: MarginInput,
        gross: MarginResult | None,
        contribution: MarginResult | None,
        operating: MarginResult | None,
        net: MarginResult | None,
    ) -> list[MarginOpportunity]:
        """
        Identify deterministic margin-improvement opportunities.
        """

        opportunities: list[MarginOpportunity] = []

        # COGS opportunity
        if (
            gross is not None
            and data.revenue > ZERO
            and data.cost_of_goods_sold > ZERO
        ):
            target_cogs = (
                data.cost_of_goods_sold
                * Decimal("0.95")
            )

            improvement = (
                data.cost_of_goods_sold
                - target_cogs
            )

            margin_gain = percentage(
                improvement,
                data.revenue,
            )

            if improvement > ZERO:
                opportunities.append(
                    MarginOpportunity(
                        name="COGS optimization",
                        opportunity_type=(
                            MarginDriverType.COGS
                        ),
                        current_value=round_money(
                            data.cost_of_goods_sold
                        ),
                        target_value=round_money(
                            target_cogs
                        ),
                        estimated_improvement=(
                            round_percent(
                                margin_gain
                            )
                        ),
                        estimated_profit_impact=(
                            round_money(
                                improvement
                            )
                        ),
                        priority=(
                            MarginSeverity.HIGH
                            if gross.health
                            in {
                                MarginHealth.CRITICAL,
                                MarginHealth.NEGATIVE,
                            }
                            else MarginSeverity.MEDIUM
                        ),
                        description=(
                            "A 5% reduction in direct costs could improve gross profit."
                        ),
                        recommendation=(
                            "Review supplier pricing, procurement terms, "
                            "waste, production efficiency, and product-level costs."
                        ),
                        evidence=[
                            f"Current COGS = {data.cost_of_goods_sold}",
                            f"Estimated savings = {improvement}",
                            f"Potential margin improvement = {margin_gain} points",
                        ],
                    )
                )

        # OPEX opportunity
        if (
            data.operating_expenses > ZERO
            and data.revenue > ZERO
        ):
            target_opex = (
                data.operating_expenses
                * Decimal("0.95")
            )

            savings = (
                data.operating_expenses
                - target_opex
            )

            margin_gain = percentage(
                savings,
                data.revenue,
            )

            opportunities.append(
                MarginOpportunity(
                    name="Operating expense optimization",
                    opportunity_type=(
                        MarginDriverType.OPEX
                    ),
                    current_value=round_money(
                        data.operating_expenses
                    ),
                    target_value=round_money(
                        target_opex
                    ),
                    estimated_improvement=(
                        margin_gain
                    ),
                    estimated_profit_impact=round_money(
                        savings
                    ),
                    priority=(
                        MarginSeverity.HIGH
                        if operating
                        and operating.health
                        in {
                            MarginHealth.CRITICAL,
                            MarginHealth.NEGATIVE,
                        }
                        else MarginSeverity.MEDIUM
                    ),
                    description=(
                        "Reducing controllable operating expenses can expand operating margin."
                    ),
                    recommendation=(
                        "Review departmental spending, vendor contracts, "
                        "software subscriptions, staffing utilization, "
                        "and discretionary expenditure."
                    ),
                    evidence=[
                        f"Current OPEX = {data.operating_expenses}",
                        f"Estimated savings = {savings}",
                        f"Potential margin improvement = {margin_gain} points",
                    ],
                )
            )

        # Pricing opportunity
        if data.revenue > ZERO:
            price_improvement = (
                data.revenue
                * Decimal("0.03")
            )

            margin_gain = percentage(
                price_improvement,
                data.revenue,
            )

            opportunities.append(
                MarginOpportunity(
                    name="Pricing optimization",
                    opportunity_type=(
                        MarginDriverType.PRICE
                    ),
                    current_value=data.revenue,
                    target_value=(
                        data.revenue
                        + price_improvement
                    ),
                    estimated_improvement=margin_gain,
                    estimated_profit_impact=round_money(
                        price_improvement
                    ),
                    priority=MarginSeverity.MEDIUM,
                    description=(
                        "A modest pricing improvement may increase "
                        "profit without equivalent cost growth."
                    ),
                    recommendation=(
                        "Evaluate price elasticity, discount leakage, "
                        "premium offerings, and customer-specific pricing."
                    ),
                    evidence=[
                        "Scenario assumes a 3% revenue uplift.",
                        f"Estimated additional profit = {price_improvement}",
                    ],
                )
            )

        return opportunities

    # ------------------------------------------------------------------
    # What-If Margin Simulation
    # ------------------------------------------------------------------

    def simulate_margin(
        self,
        data: MarginInput | Mapping[str, Any],
        *,
        revenue_change_percent: Decimal | float = ZERO,
        cogs_change_percent: Decimal | float = ZERO,
        opex_change_percent: Decimal | float = ZERO,
        variable_cost_change_percent: Decimal | float = ZERO,
    ) -> MarginAnalysisResult:
        """
        Simulate margin results after controlled changes.

        Percent changes are expressed as percentage values:
            10 = +10%
            -5 = -5%
        """

        scenario = self.normalize_input(
            data
        )
        scenario.derive()

        revenue_factor = (
            ONE
            + to_decimal(
                revenue_change_percent
            )
            / ONE_HUNDRED
        )

        cogs_factor = (
            ONE
            + to_decimal(
                cogs_change_percent
            )
            / ONE_HUNDRED
        )

        opex_factor = (
            ONE
            + to_decimal(
                opex_change_percent
            )
            / ONE_HUNDRED
        )

        variable_factor = (
            ONE
            + to_decimal(
                variable_cost_change_percent
            )
            / ONE_HUNDRED
        )

        scenario.revenue = round_money(
            scenario.revenue
            * revenue_factor
        )

        scenario.cost_of_goods_sold = round_money(
            scenario.cost_of_goods_sold
            * cogs_factor
        )

        scenario.operating_expenses = round_money(
            scenario.operating_expenses
            * opex_factor
        )

        scenario.variable_costs = round_money(
            scenario.variable_costs
            * variable_factor
        )

        scenario.gross_profit = None
        scenario.contribution_profit = None
        scenario.operating_profit = None
        scenario.net_profit = None

        scenario.derive()

        return self.analyze(
            scenario
        )

    # ------------------------------------------------------------------
    # Break-Even
    # ------------------------------------------------------------------

    def break_even_revenue(
        self,
        data: MarginInput | Mapping[str, Any],
    ) -> Decimal | None:
        """
        Estimate break-even revenue using contribution margin.
        """

        normalized = self.normalize_input(
            data
        )
        normalized.derive()

        contribution_margin_ratio = (
            safe_divide(
                normalized.contribution_profit
                or ZERO,
                normalized.revenue,
            )
        )

        if (
            contribution_margin_ratio <= ZERO
            or normalized.fixed_costs <= ZERO
        ):
            return None

        return round_money(
            safe_divide(
                normalized.fixed_costs,
                contribution_margin_ratio,
            )
        )

    def margin_of_safety(
        self,
        data: MarginInput | Mapping[str, Any],
    ) -> Decimal | None:
        """
        Calculate margin of safety as a percentage of revenue.
        """

        normalized = self.normalize_input(
            data
        )

        normalized.derive()

        break_even = self.break_even_revenue(
            normalized
        )

        if (
            break_even is None
            or normalized.revenue == ZERO
        ):
            return None

        return round_percent(
            safe_divide(
                normalized.revenue
                - break_even,
                normalized.revenue,
            )
            * ONE_HUNDRED
        )

    # ------------------------------------------------------------------
    # Historical Margin Series
    # ------------------------------------------------------------------

    def analyze_history(
        self,
        periods: Sequence[
            MarginInput | Mapping[str, Any]
        ],
    ) -> list[MarginAnalysisResult]:
        """Analyze a chronological series of periods."""

        results: list[MarginAnalysisResult] = []

        previous: MarginInput | None = None

        for period in periods:
            current = self.normalize_input(
                period
            )

            result = self.analyze(
                current,
                previous=previous,
            )

            results.append(result)

            previous = current

        return results

    def margin_series(
        self,
        periods: Sequence[
            MarginInput | Mapping[str, Any]
        ],
        margin_type: MarginType,
    ) -> list[Decimal]:
        """Return a margin time series."""

        values: list[Decimal] = []

        for period in periods:
            normalized = self.normalize_input(
                period
            )
            normalized.derive()

            result: MarginResult | None

            if margin_type == MarginType.GROSS:
                result = self.gross_margin(
                    normalized
                )

            elif margin_type == MarginType.CONTRIBUTION:
                result = self.contribution_margin(
                    normalized
                )

            elif margin_type == MarginType.OPERATING:
                result = self.operating_margin(
                    normalized
                )

            else:
                result = self.net_margin(
                    normalized
                )

            if result is not None:
                values.append(
                    result.value
                )

        return values

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def generate_warnings(
        self,
        gross: MarginResult | None,
        contribution: MarginResult | None,
        operating: MarginResult | None,
        net: MarginResult | None,
        drivers: Sequence[MarginDriver],
    ) -> list[str]:
        """Generate margin-related warnings."""

        warnings: list[str] = []

        margins = [
            gross,
            contribution,
            operating,
            net,
        ]

        for margin in margins:
            if margin is None:
                continue

            if margin.health == MarginHealth.NEGATIVE:
                warnings.append(
                    f"{margin.margin_type.value.title()} margin is negative."
                )

            elif margin.health == MarginHealth.CRITICAL:
                warnings.append(
                    f"{margin.margin_type.value.title()} margin is critically low."
                )

            if margin.trend == MarginTrend.COMPRESSING:
                warnings.append(
                    f"{margin.margin_type.value.title()} margin is compressing."
                )

        for driver in drivers:
            if (
                driver.driver_type
                in {
                    MarginDriverType.COGS,
                    MarginDriverType.OPEX,
                }
                and driver.direction == "negative"
                and driver.severity
                in {
                    MarginSeverity.HIGH,
                    MarginSeverity.CRITICAL,
                }
            ):
                warnings.append(
                    driver.description
                )

        return list(
            dict.fromkeys(warnings)
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def generate_summary(
        gross: MarginResult | None,
        contribution: MarginResult | None,
        operating: MarginResult | None,
        net: MarginResult | None,
        overall_trend: MarginTrend,
        drivers: Sequence[MarginDriver],
    ) -> str:
        """Generate an executive-friendly margin summary."""

        available = [
            margin
            for margin in (
                gross,
                contribution,
                operating,
                net,
            )
            if margin is not None
        ]

        if not available:
            return "No margin metrics could be calculated."

        parts: list[str] = []

        if gross is not None:
            parts.append(
                f"gross margin is {gross.value}%"
            )

        if operating is not None:
            parts.append(
                f"operating margin is {operating.value}%"
            )

        if net is not None:
            parts.append(
                f"net margin is {net.value}%"
            )

        summary = (
            "The business has "
            + ", ".join(parts)
            + f". Overall margin trend is "
            f"{overall_trend.value}."
        )

        negative_drivers = [
            driver
            for driver in drivers
            if driver.direction == "negative"
        ]

        if negative_drivers:
            summary += (
                f" {len(negative_drivers)} negative "
                "margin driver(s) were identified."
            )

        return summary

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_input(
        self,
        data: MarginInput | Mapping[str, Any] | None,
    ) -> MarginInput:
        """Normalize mapping input."""

        if data is None:
            return MarginInput()

        if isinstance(
            data,
            MarginInput,
        ):
            return data

        if not isinstance(data, Mapping):
            raise TypeError(
                "Margin input must be MarginInput or a mapping."
            )

        return MarginInput(
            revenue=to_decimal(
                data.get("revenue")
            ),
            cost_of_goods_sold=to_decimal(
                data.get("cost_of_goods_sold")
                or data.get("cogs")
            ),
            variable_costs=to_decimal(
                data.get("variable_costs")
            ),
            fixed_costs=to_decimal(
                data.get("fixed_costs")
            ),
            operating_expenses=to_decimal(
                data.get("operating_expenses")
                or data.get("opex")
            ),
            interest_expense=to_decimal(
                data.get("interest_expense")
            ),
            tax_expense=to_decimal(
                data.get("tax_expense")
            ),
            gross_profit=(
                to_decimal(
                    data.get("gross_profit")
                )
                if data.get("gross_profit") is not None
                else None
            ),
            contribution_profit=(
                to_decimal(
                    data.get("contribution_profit")
                )
                if data.get("contribution_profit") is not None
                else None
            ),
            operating_profit=(
                to_decimal(
                    data.get("operating_profit")
                )
                if data.get("operating_profit") is not None
                else None
            ),
            net_profit=(
                to_decimal(
                    data.get("net_profit")
                    or data.get("net_income")
                )
                if (
                    data.get("net_profit") is not None
                    or data.get("net_income") is not None
                )
                else None
            ),
            discounts=to_decimal(
                data.get("discounts")
            ),
            returns=to_decimal(
                data.get("returns")
            ),
            units_sold=to_decimal(
                data.get("units_sold")
            ),
            customers=to_decimal(
                data.get("customers")
            ),
            orders=to_decimal(
                data.get("orders")
            ),
            currency=data.get(
                "currency"
            ),
            period=(
                str(data.get("period"))
                if data.get("period") is not None
                else None
            ),
            metadata=dict(
                data.get("metadata") or {}
            ),
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def analyze_margins(
    current: MarginInput | Mapping[str, Any],
    *,
    previous: MarginInput | Mapping[str, Any] | None = None,
    segments: Sequence[
        Mapping[str, Any]
    ]
    | None = None,
) -> MarginAnalysisResult:
    """Run complete margin analysis."""
    analyzer = MarginAnalyzer()

    return analyzer.analyze(
        current,
        previous=previous,
        segments=segments,
    )


def calculate_gross_margin(
    revenue: Decimal | float | int,
    cost_of_goods_sold: Decimal | float | int,
) -> Decimal:
    """Calculate gross margin."""
    revenue_decimal = to_decimal(
        revenue
    )

    cogs_decimal = to_decimal(
        cost_of_goods_sold
    )

    return calculate_margin(
        revenue_decimal - cogs_decimal,
        revenue_decimal,
    )


def calculate_contribution_margin(
    revenue: Decimal | float | int,
    variable_costs: Decimal | float | int,
) -> Decimal:
    """Calculate contribution margin."""
    revenue_decimal = to_decimal(
        revenue
    )

    variable_decimal = to_decimal(
        variable_costs
    )

    return calculate_margin(
        revenue_decimal - variable_decimal,
        revenue_decimal,
    )


def calculate_operating_margin(
    revenue: Decimal | float | int,
    operating_profit: Decimal | float | int,
) -> Decimal:
    """Calculate operating margin."""
    return calculate_margin(
        to_decimal(operating_profit),
        to_decimal(revenue),
    )


def calculate_net_margin(
    revenue: Decimal | float | int,
    net_profit: Decimal | float | int,
) -> Decimal:
    """Calculate net margin."""
    return calculate_margin(
        to_decimal(net_profit),
        to_decimal(revenue),
    )


def detect_margin_compression(
    current: MarginInput | Mapping[str, Any],
    previous: MarginInput | Mapping[str, Any],
    *,
    threshold_points: Decimal | float = Decimal("2"),
) -> list[MarginResult]:
    """Detect material margin compression."""
    return MarginAnalyzer().detect_margin_compression(
        current,
        previous,
        threshold_points=threshold_points,
    )


def detect_margin_expansion(
    current: MarginInput | Mapping[str, Any],
    previous: MarginInput | Mapping[str, Any],
    *,
    threshold_points: Decimal | float = Decimal("2"),
) -> list[MarginResult]:
    """Detect material margin expansion."""
    return MarginAnalyzer().detect_margin_expansion(
        current,
        previous,
        threshold_points=threshold_points,
    )


def calculate_break_even_revenue(
    data: MarginInput | Mapping[str, Any],
) -> Decimal | None:
    """Calculate estimated break-even revenue."""
    return MarginAnalyzer().break_even_revenue(
        data
    )


def calculate_margin_of_safety(
    data: MarginInput | Mapping[str, Any],
) -> Decimal | None:
    """Calculate margin of safety."""
    return MarginAnalyzer().margin_of_safety(
        data
    )


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Enums
    "MarginType",
    "MarginTrend",
    "MarginHealth",
    "MarginDriverType",
    "MarginSeverity",

    # Models
    "MarginInput",
    "MarginResult",
    "MarginDriver",
    "SegmentMargin",
    "MarginOpportunity",
    "MarginAnalysisResult",

    # Analyzer
    "MarginAnalyzer",

    # Utilities
    "to_decimal",
    "safe_divide",
    "calculate_margin",
    "percentage_change",

    # Convenience APIs
    "analyze_margins",
    "calculate_gross_margin",
    "calculate_contribution_margin",
    "calculate_operating_margin",
    "calculate_net_margin",
    "detect_margin_compression",
    "detect_margin_expansion",
    "calculate_break_even_revenue",
    "calculate_margin_of_safety",
]