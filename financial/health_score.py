"""
FinCo AI - Financial Health Score Engine

File:
    backend/app/financial/health_score.py

Purpose:
    Calculate an explainable 0-100 financial health score from normalized
    financial metrics.

Responsibilities:
    - Calculate component financial health scores.
    - Combine profitability, liquidity, leverage, cash-flow, growth,
      efficiency, and cost-management signals.
    - Classify overall financial health.
    - Identify strengths and weaknesses.
    - Generate risk indicators.
    - Produce evidence suitable for alerts, recommendations, reports,
      and agent workflows.
    - Support configurable scoring weights and thresholds.

Pipeline:

    Financial Data
        ↓
    Financial Extraction / Normalization
        ↓
    Financial Metrics
        ↓
    health_score.py
        ↓
    ┌───────────────────────────────────────────┐
    │ Profitability │ Liquidity │ Leverage       │
    │ Cash Flow     │ Growth    │ Efficiency     │
    │ Cost Control  │ Stability                  │
    └───────────────────────────────────────────┘
        ↓
    Overall Health Score (0-100)
        ↓
    Risk Signals / Alerts / Recommendations
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
ONE_HUNDRED = Decimal("100")
MONEY_QUANT = Decimal("0.01")
SCORE_QUANT = Decimal("0.01")


# ============================================================================
# Enums
# ============================================================================


class HealthCategory(str, Enum):
    """Overall financial health classification."""

    EXCELLENT = "excellent"
    HEALTHY = "healthy"
    STABLE = "stable"
    WEAK = "weak"
    CRITICAL = "critical"


class HealthComponent(str, Enum):
    """Financial health scoring components."""

    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    LEVERAGE = "leverage"
    CASH_FLOW = "cash_flow"
    GROWTH = "growth"
    EFFICIENCY = "efficiency"
    COST_CONTROL = "cost_control"
    STABILITY = "stability"


class RiskLevel(str, Enum):
    """Risk severity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrendDirection(str, Enum):
    """Metric trend direction."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    UNKNOWN = "unknown"


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
    except Exception:
        return default


def clamp(
    value: Decimal,
    minimum: Decimal = ZERO,
    maximum: Decimal = ONE_HUNDRED,
) -> Decimal:
    """Clamp a Decimal to a range."""
    return max(minimum, min(maximum, value))


def round_score(value: Decimal) -> Decimal:
    """Round a score to two decimal places."""
    return value.quantize(
        SCORE_QUANT,
        rounding=ROUND_HALF_UP,
    )


def safe_ratio(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely calculate a ratio."""
    if denominator == ZERO:
        return default

    return numerator / denominator


def percentage_change(
    current: Decimal,
    previous: Decimal,
) -> Decimal:
    """Calculate percentage change."""
    if previous == ZERO:
        if current == ZERO:
            return ZERO

        return ONE_HUNDRED

    return ((current - previous) / abs(previous)) * ONE_HUNDRED


def percentage_score(
    value: Decimal,
    *,
    poor: Decimal,
    excellent: Decimal,
    inverse: bool = False,
) -> Decimal:
    """
    Convert a metric into a 0-100 score.

    For normal metrics:
        poor -> 0
        excellent -> 100

    For inverse metrics:
        poor -> 100
        excellent -> 0
    """
    if poor == excellent:
        return Decimal("50")

    if inverse:
        score = (
            (poor - value)
            / (poor - excellent)
        ) * ONE_HUNDRED
    else:
        score = (
            (value - poor)
            / (excellent - poor)
        ) * ONE_HUNDRED

    return clamp(score)


# ============================================================================
# Data Models
# ============================================================================


@dataclass(slots=True)
class FinancialMetrics:
    """
    Input metrics for the health score engine.

    All percentage fields are expressed as percentages, not decimals.

    Example:
        gross_margin = 42.5
        net_margin = 11.2
        current_ratio = 1.8
        debt_to_equity = 0.75
    """

    # ------------------------------------------------------------------
    # Profitability
    # ------------------------------------------------------------------

    revenue: Decimal = ZERO
    gross_profit: Decimal = ZERO
    operating_profit: Decimal = ZERO
    net_profit: Decimal = ZERO

    gross_margin: Decimal | None = None
    operating_margin: Decimal | None = None
    net_margin: Decimal | None = None

    return_on_assets: Decimal | None = None
    return_on_equity: Decimal | None = None

    # ------------------------------------------------------------------
    # Liquidity
    # ------------------------------------------------------------------

    current_assets: Decimal = ZERO
    current_liabilities: Decimal = ZERO

    cash: Decimal = ZERO
    receivables: Decimal = ZERO
    inventory: Decimal = ZERO

    current_ratio: Decimal | None = None
    quick_ratio: Decimal | None = None
    cash_ratio: Decimal | None = None

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    total_assets: Decimal = ZERO
    total_liabilities: Decimal = ZERO
    total_equity: Decimal = ZERO
    total_debt: Decimal = ZERO

    debt_to_equity: Decimal | None = None
    debt_to_assets: Decimal | None = None
    liabilities_to_assets: Decimal | None = None
    interest_coverage: Decimal | None = None

    # ------------------------------------------------------------------
    # Cash Flow
    # ------------------------------------------------------------------

    operating_cash_flow: Decimal = ZERO
    investing_cash_flow: Decimal = ZERO
    financing_cash_flow: Decimal = ZERO

    free_cash_flow: Decimal | None = None
    cash_flow_margin: Decimal | None = None

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------

    revenue_growth: Decimal | None = None
    profit_growth: Decimal | None = None
    asset_growth: Decimal | None = None

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    asset_turnover: Decimal | None = None
    receivable_days: Decimal | None = None
    inventory_days: Decimal | None = None
    payable_days: Decimal | None = None

    # ------------------------------------------------------------------
    # Cost Management
    # ------------------------------------------------------------------

    operating_expenses: Decimal = ZERO
    cost_growth: Decimal | None = None
    expense_ratio: Decimal | None = None

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    revenue_volatility: Decimal | None = None
    profit_volatility: Decimal | None = None

    # ------------------------------------------------------------------
    # Historical / Comparative Data
    # ------------------------------------------------------------------

    previous_revenue: Decimal | None = None
    previous_profit: Decimal | None = None
    previous_cash: Decimal | None = None
    previous_debt: Decimal | None = None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    period: str | None = None
    currency: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def derive_missing_metrics(self) -> None:
        """Calculate basic metrics that were not supplied."""

        if (
            self.gross_margin is None
            and self.revenue != ZERO
        ):
            self.gross_margin = (
                self.gross_profit
                / self.revenue
                * ONE_HUNDRED
            )

        if (
            self.operating_margin is None
            and self.revenue != ZERO
        ):
            self.operating_margin = (
                self.operating_profit
                / self.revenue
                * ONE_HUNDRED
            )

        if (
            self.net_margin is None
            and self.revenue != ZERO
        ):
            self.net_margin = (
                self.net_profit
                / self.revenue
                * ONE_HUNDRED
            )

        if self.current_ratio is None:
            self.current_ratio = safe_ratio(
                self.current_assets,
                self.current_liabilities,
            )

        if self.quick_ratio is None:
            quick_assets = (
                self.current_assets
                - self.inventory
            )

            self.quick_ratio = safe_ratio(
                quick_assets,
                self.current_liabilities,
            )

        if self.cash_ratio is None:
            self.cash_ratio = safe_ratio(
                self.cash,
                self.current_liabilities,
            )

        if self.debt_to_equity is None:
            self.debt_to_equity = safe_ratio(
                self.total_debt,
                self.total_equity,
            )

        if self.debt_to_assets is None:
            self.debt_to_assets = safe_ratio(
                self.total_debt,
                self.total_assets,
            )

        if self.liabilities_to_assets is None:
            self.liabilities_to_assets = safe_ratio(
                self.total_liabilities,
                self.total_assets,
            )

        if self.cash_flow_margin is None and self.revenue != ZERO:
            self.cash_flow_margin = (
                self.operating_cash_flow
                / self.revenue
                * ONE_HUNDRED
            )

        if self.free_cash_flow is None:
            self.free_cash_flow = (
                self.operating_cash_flow
                + self.investing_cash_flow
            )

        if self.expense_ratio is None and self.revenue != ZERO:
            self.expense_ratio = (
                self.operating_expenses
                / self.revenue
                * ONE_HUNDRED
            )

        if (
            self.revenue_growth is None
            and self.previous_revenue is not None
        ):
            self.revenue_growth = percentage_change(
                self.revenue,
                self.previous_revenue,
            )

        if (
            self.profit_growth is None
            and self.previous_profit is not None
        ):
            self.profit_growth = percentage_change(
                self.net_profit,
                self.previous_profit,
            )


@dataclass(slots=True)
class ComponentScore:
    """Score for one financial health component."""

    component: HealthComponent

    score: Decimal

    weight: Decimal

    weighted_score: Decimal

    status: HealthCategory

    strengths: list[str] = field(default_factory=list)

    weaknesses: list[str] = field(default_factory=list)

    risk_level: RiskLevel = RiskLevel.LOW

    evidence: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component.value,
            "score": str(self.score),
            "weight": str(self.weight),
            "weighted_score": str(self.weighted_score),
            "status": self.status.value,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "risk_level": self.risk_level.value,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class HealthRisk:
    """Identified financial health risk."""

    name: str

    risk_level: RiskLevel

    component: HealthComponent

    score: Decimal

    description: str

    trigger_value: Decimal | None = None

    threshold: Decimal | None = None

    recommendation_hint: str | None = None

    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "risk_level": self.risk_level.value,
            "component": self.component.value,
            "score": str(self.score),
            "description": self.description,
            "trigger_value": (
                str(self.trigger_value)
                if self.trigger_value is not None
                else None
            ),
            "threshold": (
                str(self.threshold)
                if self.threshold is not None
                else None
            ),
            "recommendation_hint": self.recommendation_hint,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class HealthScoreResult:
    """Complete financial health score result."""

    overall_score: Decimal

    category: HealthCategory

    risk_level: RiskLevel

    components: list[ComponentScore]

    risks: list[HealthRisk]

    strengths: list[str]

    weaknesses: list[str]

    summary: str

    metrics: FinancialMetrics

    confidence: Decimal = Decimal("1.00")

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def weakest_component(self) -> ComponentScore | None:
        if not self.components:
            return None

        return min(
            self.components,
            key=lambda component: component.score,
        )

    @property
    def strongest_component(self) -> ComponentScore | None:
        if not self.components:
            return None

        return max(
            self.components,
            key=lambda component: component.score,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": str(self.overall_score),
            "category": self.category.value,
            "risk_level": self.risk_level.value,
            "components": [
                component.to_dict()
                for component in self.components
            ],
            "risks": [
                risk.to_dict()
                for risk in self.risks
            ],
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "summary": self.summary,
            "confidence": str(self.confidence),
            "metadata": self.metadata,
        }


# ============================================================================
# Default Weights
# ============================================================================


DEFAULT_WEIGHTS: dict[HealthComponent, Decimal] = {
    HealthComponent.PROFITABILITY: Decimal("0.20"),
    HealthComponent.LIQUIDITY: Decimal("0.15"),
    HealthComponent.LEVERAGE: Decimal("0.15"),
    HealthComponent.CASH_FLOW: Decimal("0.15"),
    HealthComponent.GROWTH: Decimal("0.10"),
    HealthComponent.EFFICIENCY: Decimal("0.10"),
    HealthComponent.COST_CONTROL: Decimal("0.10"),
    HealthComponent.STABILITY: Decimal("0.05"),
}


# ============================================================================
# Health Score Engine
# ============================================================================


class FinancialHealthScorer:
    """
    Deterministic financial health scoring engine.

    The scorer uses multiple independent components and combines them
    using configurable weights.

    All component scores are normalized to 0-100.
    """

    def __init__(
        self,
        *,
        weights: Mapping[
            HealthComponent,
            Decimal | float | int,
        ] | None = None,
    ) -> None:
        self.weights = self._normalize_weights(
            weights or DEFAULT_WEIGHTS
        )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def calculate(
        self,
        metrics: FinancialMetrics,
    ) -> HealthScoreResult:
        """
        Calculate the complete financial health score.
        """
        metrics.derive_missing_metrics()

        components = [
            self.score_profitability(metrics),
            self.score_liquidity(metrics),
            self.score_leverage(metrics),
            self.score_cash_flow(metrics),
            self.score_growth(metrics),
            self.score_efficiency(metrics),
            self.score_cost_control(metrics),
            self.score_stability(metrics),
        ]

        overall_score = round_score(
            sum(
                (
                    component.weighted_score
                    for component in components
                ),
                ZERO,
            )
        )

        category = self.classify_score(overall_score)

        risks = self.identify_risks(
            metrics,
            components,
        )

        strengths = self._collect_strengths(components)

        weaknesses = self._collect_weaknesses(components)

        risk_level = self.classify_risk(
            overall_score,
            risks,
        )

        confidence = self.calculate_confidence(metrics)

        summary = self.generate_summary(
            overall_score=overall_score,
            category=category,
            risk_level=risk_level,
            components=components,
            risks=risks,
        )

        return HealthScoreResult(
            overall_score=overall_score,
            category=category,
            risk_level=risk_level,
            components=components,
            risks=risks,
            strengths=strengths,
            weaknesses=weaknesses,
            summary=summary,
            metrics=metrics,
            confidence=confidence,
            metadata={
                "scoring_model": "deterministic_weighted_v1",
                "weights": {
                    component.value: str(weight)
                    for component, weight in self.weights.items()
                },
            },
        )

    # ------------------------------------------------------------------
    # Profitability
    # ------------------------------------------------------------------

    def score_profitability(
        self,
        metrics: FinancialMetrics,
    ) -> ComponentScore:
        """
        Score profitability.

        Main signals:
            - gross margin
            - operating margin
            - net margin
            - ROA
            - ROE
        """
        scores: list[Decimal] = []
        evidence: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        if metrics.gross_margin is not None:
            gross_score = percentage_score(
                metrics.gross_margin,
                poor=ZERO,
                excellent=Decimal("50"),
            )

            scores.append(gross_score)

            evidence.append(
                f"Gross margin: "
                f"{metrics.gross_margin:.2f}%"
            )

            if metrics.gross_margin >= Decimal("30"):
                strengths.append(
                    "Gross margin is strong."
                )
            elif metrics.gross_margin < Decimal("10"):
                weaknesses.append(
                    "Gross margin is weak."
                )

        if metrics.operating_margin is not None:
            operating_score = percentage_score(
                metrics.operating_margin,
                poor=Decimal("-10"),
                excellent=Decimal("30"),
            )

            scores.append(operating_score)

            evidence.append(
                f"Operating margin: "
                f"{metrics.operating_margin:.2f}%"
            )

            if metrics.operating_margin < ZERO:
                weaknesses.append(
                    "Operating margin is negative."
                )

        if metrics.net_margin is not None:
            net_score = percentage_score(
                metrics.net_margin,
                poor=Decimal("-20"),
                excellent=Decimal("25"),
            )

            scores.append(net_score)

            evidence.append(
                f"Net margin: "
                f"{metrics.net_margin:.2f}%"
            )

            if metrics.net_margin < ZERO:
                weaknesses.append(
                    "Net profitability is negative."
                )

        if metrics.return_on_assets is not None:
            roa_score = percentage_score(
                metrics.return_on_assets,
                poor=Decimal("-5"),
                excellent=Decimal("20"),
            )

            scores.append(roa_score)

            evidence.append(
                f"Return on assets: "
                f"{metrics.return_on_assets:.2f}%"
            )

        if metrics.return_on_equity is not None:
            roe_score = percentage_score(
                metrics.return_on_equity,
                poor=Decimal("-10"),
                excellent=Decimal("30"),
            )

            scores.append(roe_score)

            evidence.append(
                f"Return on equity: "
                f"{metrics.return_on_equity:.2f}%"
            )

        score = self._average_scores(
            scores,
            default=Decimal("50"),
        )

        if not scores:
            weaknesses.append(
                "Insufficient profitability metrics."
            )

        return self._build_component(
            HealthComponent.PROFITABILITY,
            score,
            strengths,
            weaknesses,
            evidence,
        )

    # ------------------------------------------------------------------
    # Liquidity
    # ------------------------------------------------------------------

    def score_liquidity(
        self,
        metrics: FinancialMetrics,
    ) -> ComponentScore:
        """
        Score short-term liquidity.
        """
        scores: list[Decimal] = []
        evidence: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        if metrics.current_ratio is not None:
            current_score = self._ratio_score(
                metrics.current_ratio,
                critical=Decimal("0.75"),
                healthy=Decimal("1.50"),
                excellent=Decimal("2.50"),
            )

            scores.append(current_score)

            evidence.append(
                f"Current ratio: "
                f"{metrics.current_ratio:.2f}"
            )

            if metrics.current_ratio < Decimal("1"):
                weaknesses.append(
                    "Current liabilities exceed current assets."
                )

        if metrics.quick_ratio is not None:
            quick_score = self._ratio_score(
                metrics.quick_ratio,
                critical=Decimal("0.50"),
                healthy=Decimal("1.00"),
                excellent=Decimal("2.00"),
            )

            scores.append(quick_score)

            evidence.append(
                f"Quick ratio: "
                f"{metrics.quick_ratio:.2f}"
            )

        if metrics.cash_ratio is not None:
            cash_score = self._ratio_score(
                metrics.cash_ratio,
                critical=Decimal("0.10"),
                healthy=Decimal("0.50"),
                excellent=Decimal("1.00"),
            )

            scores.append(cash_score)

            evidence.append(
                f"Cash ratio: "
                f"{metrics.cash_ratio:.2f}"
            )

        score = self._average_scores(
            scores,
            default=Decimal("50"),
        )

        if score >= Decimal("75"):
            strengths.append(
                "Short-term liquidity appears strong."
            )

        if score < Decimal("40"):
            weaknesses.append(
                "Liquidity position requires attention."
            )

        return self._build_component(
            HealthComponent.LIQUIDITY,
            score,
            strengths,
            weaknesses,
            evidence,
        )

    # ------------------------------------------------------------------
    # Leverage
    # ------------------------------------------------------------------

    def score_leverage(
        self,
        metrics: FinancialMetrics,
    ) -> ComponentScore:
        """
        Score debt and leverage.

        Lower debt ratios generally produce higher scores.
        """
        scores: list[Decimal] = []
        evidence: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        if metrics.debt_to_equity is not None:
            score = self._inverse_ratio_score(
                metrics.debt_to_equity,
                excellent=Decimal("0.25"),
                healthy=Decimal("1.00"),
                critical=Decimal("3.00"),
            )

            scores.append(score)

            evidence.append(
                f"Debt-to-equity: "
                f"{metrics.debt_to_equity:.2f}"
            )

            if metrics.debt_to_equity > Decimal("2"):
                weaknesses.append(
                    "Debt-to-equity is elevated."
                )

        if metrics.debt_to_assets is not None:
            score = self._inverse_ratio_score(
                metrics.debt_to_assets,
                excellent=Decimal("0.30"),
                healthy=Decimal("0.60"),
                critical=Decimal("0.90"),
            )

            scores.append(score)

            evidence.append(
                f"Debt-to-assets: "
                f"{metrics.debt_to_assets:.2f}"
            )

        if metrics.liabilities_to_assets is not None:
            score = self._inverse_ratio_score(
                metrics.liabilities_to_assets,
                excellent=Decimal("0.40"),
                healthy=Decimal("0.60"),
                critical=Decimal("0.90"),
            )

            scores.append(score)

            evidence.append(
                f"Liabilities-to-assets: "
                f"{metrics.liabilities_to_assets:.2f}"
            )

        if metrics.interest_coverage is not None:
            coverage_score = self._ratio_score(
                metrics.interest_coverage,
                critical=Decimal("1"),
                healthy=Decimal("3"),
                excellent=Decimal("8"),
            )

            scores.append(coverage_score)

            evidence.append(
                f"Interest coverage: "
                f"{metrics.interest_coverage:.2f}"
            )

            if metrics.interest_coverage < Decimal("1.5"):
                weaknesses.append(
                    "Interest coverage is weak."
                )

        score = self._average_scores(
            scores,
            default=Decimal("50"),
        )

        if score >= Decimal("75"):
            strengths.append(
                "Leverage is within a relatively healthy range."
            )

        return self._build_component(
            HealthComponent.LEVERAGE,
            score,
            strengths,
            weaknesses,
            evidence,
        )

    # ------------------------------------------------------------------
    # Cash Flow
    # ------------------------------------------------------------------

    def score_cash_flow(
        self,
        metrics: FinancialMetrics,
    ) -> ComponentScore:
        """
        Score operating cash generation and free cash flow.
        """
        scores: list[Decimal] = []
        evidence: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        if metrics.cash_flow_margin is not None:
            cash_margin_score = percentage_score(
                metrics.cash_flow_margin,
                poor=Decimal("-10"),
                excellent=Decimal("25"),
            )

            scores.append(cash_margin_score)

            evidence.append(
                f"Operating cash-flow margin: "
                f"{metrics.cash_flow_margin:.2f}%"
            )

        if metrics.operating_cash_flow != ZERO:
            ocf_score = (
                Decimal("75")
                if metrics.operating_cash_flow > ZERO
                else Decimal("20")
            )

            scores.append(ocf_score)

            evidence.append(
                f"Operating cash flow: "
                f"{metrics.operating_cash_flow:.2f}"
            )

            if metrics.operating_cash_flow < ZERO:
                weaknesses.append(
                    "Operating cash flow is negative."
                )
            else:
                strengths.append(
                    "Operations are generating positive cash flow."
                )

        if metrics.free_cash_flow is not None:
            fcf_score = (
                Decimal("80")
                if metrics.free_cash_flow >= ZERO
                else Decimal("25")
            )

            scores.append(fcf_score)

            evidence.append(
                f"Free cash flow: "
                f"{metrics.free_cash_flow:.2f}"
            )

        score = self._average_scores(
            scores,
            default=50,
        )

        if score < Decimal("40"):
            weaknesses.append(
                "Cash generation is a significant concern."
            )

        return self._build_component(
            HealthComponent.CASH_FLOW,
            score,
            strengths,
            weaknesses,
            evidence,
        )

    # ------------------------------------------------------------------
    # Growth
    # ------------------------------------------------------------------

    def score_growth(
        self,
        metrics: FinancialMetrics,
    ) -> ComponentScore:
        """Score revenue and profit growth."""
        scores: list[Decimal] = []
        evidence: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        if metrics.revenue_growth is not None:
            revenue_score = percentage_score(
                metrics.revenue_growth,
                poor=Decimal("-20"),
                excellent=Decimal("30"),
            )

            scores.append(revenue_score)

            evidence.append(
                f"Revenue growth: "
                f"{metrics.revenue_growth:.2f}%"
            )

            if metrics.revenue_growth < ZERO:
                weaknesses.append(
                    "Revenue is declining."
                )
            elif metrics.revenue_growth >= Decimal("10"):
                strengths.append(
                    "Revenue growth is strong."
                )

        if metrics.profit_growth is not None:
            profit_score = percentage_score(
                metrics.profit_growth,
                poor=Decimal("-30"),
                excellent=Decimal("40"),
            )

            scores.append(profit_score)

            evidence.append(
                f"Profit growth: "
                f"{metrics.profit_growth:.2f}%"
            )

            if metrics.profit_growth < ZERO:
                weaknesses.append(
                    "Profit is declining."
                )

        if metrics.asset_growth is not None:
            asset_score = percentage_score(
                metrics.asset_growth,
                poor=Decimal("-10"),
                excellent=Decimal("25"),
            )

            scores.append(asset_score)

            evidence.append(
                f"Asset growth: "
                f"{metrics.asset_growth:.2f}%"
            )

        score = self._average_scores(
            scores,
            default=50,
        )

        return self._build_component(
            HealthComponent.GROWTH,
            score,
            strengths,
            weaknesses,
            evidence,
        )

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def score_efficiency(
        self,
        metrics: FinancialMetrics,
    ) -> ComponentScore:
        """Score asset and working-capital efficiency."""
        scores: list[Decimal] = []
        evidence: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        if metrics.asset_turnover is not None:
            asset_score = self._ratio_score(
                metrics.asset_turnover,
                critical=Decimal("0.25"),
                healthy=Decimal("1.00"),
                excellent=Decimal("2.50"),
            )

            scores.append(asset_score)

            evidence.append(
                f"Asset turnover: "
                f"{metrics.asset_turnover:.2f}"
            )

        if metrics.receivable_days is not None:
            receivable_score = self._inverse_ratio_score(
                metrics.receivable_days,
                excellent=Decimal("30"),
                healthy=Decimal("60"),
                critical=Decimal("120"),
            )

            scores.append(receivable_score)

            evidence.append(
                f"Receivable days: "
                f"{metrics.receivable_days:.2f}"
            )

            if metrics.receivable_days > Decimal("90"):
                weaknesses.append(
                    "Receivables are taking too long to convert to cash."
                )

        if metrics.inventory_days is not None:
            inventory_score = self._inverse_ratio_score(
                metrics.inventory_days,
                excellent=Decimal("30"),
                healthy=Decimal("60"),
                critical=Decimal("150"),
            )

            scores.append(inventory_score)

            evidence.append(
                f"Inventory days: "
                f"{metrics.inventory_days:.2f}"
            )

        if metrics.payable_days is not None:
            payable_score = self._inverse_ratio_score(
                metrics.payable_days,
                excellent=Decimal("30"),
                healthy=Decimal("90"),
                critical=Decimal("180"),
            )

            scores.append(payable_score)

            evidence.append(
                f"Payable days: "
                f"{metrics.payable_days:.2f}"
            )

        score = self._average_scores(
            scores,
            default=50,
        )

        return self._build_component(
            HealthComponent.EFFICIENCY,
            score,
            strengths,
            weaknesses,
            evidence,
        )

    # ------------------------------------------------------------------
    # Cost Control
    # ------------------------------------------------------------------

    def score_cost_control(
        self,
        metrics: FinancialMetrics,
    ) -> ComponentScore:
        """Score operating cost management."""
        scores: list[Decimal] = []
        evidence: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        if metrics.expense_ratio is not None:
            expense_score = self._inverse_ratio_score(
                metrics.expense_ratio,
                excellent=Decimal("30"),
                healthy=Decimal("60"),
                critical=Decimal("100"),
            )

            scores.append(expense_score)

            evidence.append(
                f"Expense ratio: "
                f"{metrics.expense_ratio:.2f}%"
            )

            if metrics.expense_ratio > Decimal("80"):
                weaknesses.append(
                    "Operating expenses consume a large share of revenue."
                )

        if metrics.cost_growth is not None:
            cost_growth_score = percentage_score(
                metrics.cost_growth,
                poor=Decimal("20"),
                excellent=Decimal("0"),
                inverse=True,
            )

            scores.append(cost_growth_score)

            evidence.append(
                f"Cost growth: "
                f"{metrics.cost_growth:.2f}%"
            )

            if metrics.cost_growth > Decimal("15"):
                weaknesses.append(
                    "Costs are growing rapidly."
                )

        score = self._average_scores(
            scores,
            default=50,
        )

        if score >= Decimal("75"):
            strengths.append(
                "Cost structure appears well controlled."
            )

        return self._build_component(
            HealthComponent.COST_CONTROL,
            score,
            strengths,
            weaknesses,
            evidence,
        )

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def score_stability(
        self,
        metrics: FinancialMetrics,
    ) -> ComponentScore:
        """Score revenue and profit stability."""
        scores: list[Decimal] = []
        evidence: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        if metrics.revenue_volatility is not None:
            revenue_score = self._inverse_ratio_score(
                metrics.revenue_volatility,
                excellent=Decimal("5"),
                healthy=Decimal("15"),
                critical=Decimal("40"),
            )

            scores.append(revenue_score)

            evidence.append(
                f"Revenue volatility: "
                f"{metrics.revenue_volatility:.2f}%"
            )

        if metrics.profit_volatility is not None:
            profit_score = self._inverse_ratio_score(
                metrics.profit_volatility,
                excellent=Decimal("10"),
                healthy=Decimal("25"),
                critical=Decimal("60"),
            )

            scores.append(profit_score)

            evidence.append(
                f"Profit volatility: "
                f"{metrics.profit_volatility:.2f}%"
            )

        score = self._average_scores(
            scores,
            default=50,
        )

        if score >= Decimal("75"):
            strengths.append(
                "Financial performance appears relatively stable."
            )

        if score < Decimal("40"):
            weaknesses.append(
                "Financial performance is volatile."
            )

        return self._build_component(
            HealthComponent.STABILITY,
            score,
            strengths,
            weaknesses,
            evidence,
        )

    # ------------------------------------------------------------------
    # Risk Identification
    # ------------------------------------------------------------------

    def identify_risks(
        self,
        metrics: FinancialMetrics,
        components: Sequence[ComponentScore],
    ) -> list[HealthRisk]:
        """
        Identify actionable health risks.

        These risks are deliberately separate from the overall score so
        the alert engine can consume individual conditions.
        """
        risks: list[HealthRisk] = []

        # Liquidity
        if (
            metrics.current_ratio is not None
            and metrics.current_ratio < Decimal("1")
        ):
            risks.append(
                HealthRisk(
                    name="low_current_ratio",
                    risk_level=RiskLevel.HIGH,
                    component=HealthComponent.LIQUIDITY,
                    score=self._component_score(
                        components,
                        HealthComponent.LIQUIDITY,
                    ),
                    description=(
                        "Current liabilities exceed current assets."
                    ),
                    trigger_value=metrics.current_ratio,
                    threshold=Decimal("1"),
                    recommendation_hint=(
                        "Improve working capital and short-term liquidity."
                    ),
                    evidence=[
                        f"Current ratio = "
                        f"{metrics.current_ratio:.2f}"
                    ],
                )
            )

        # Negative operating cash flow
        if metrics.operating_cash_flow < ZERO:
            risks.append(
                HealthRisk(
                    name="negative_operating_cash_flow",
                    risk_level=RiskLevel.HIGH,
                    component=HealthComponent.CASH_FLOW,
                    score=self._component_score(
                        components,
                        HealthComponent.CASH_FLOW,
                    ),
                    description=(
                        "Core business operations are consuming cash."
                    ),
                    trigger_value=metrics.operating_cash_flow,
                    threshold=ZERO,
                    recommendation_hint=(
                        "Investigate collections, pricing, margins, "
                        "and operating costs."
                    ),
                    evidence=[
                        f"Operating cash flow = "
                        f"{metrics.operating_cash_flow:.2f}"
                    ],
                )
            )

        # Negative profit
        if metrics.net_profit < ZERO:
            risks.append(
                HealthRisk(
                    name="negative_net_profit",
                    risk_level=RiskLevel.HIGH,
                    component=HealthComponent.PROFITABILITY,
                    score=self._component_score(
                        components,
                        HealthComponent.PROFITABILITY,
                    ),
                    description=(
                        "The business is reporting a net loss."
                    ),
                    trigger_value=metrics.net_profit,
                    threshold=ZERO,
                    recommendation_hint=(
                        "Investigate revenue decline, gross margin "
                        "compression, and operating expenses."
                    ),
                    evidence=[
                        f"Net profit = "
                        f"{metrics.net_profit:.2f}"
                    ],
                )
            )

        # High leverage
        if (
            metrics.debt_to_equity is not None
            and metrics.debt_to_equity > Decimal("2")
        ):
            risks.append(
                HealthRisk(
                    name="high_debt_to_equity",
                    risk_level=RiskLevel.HIGH,
                    component=HealthComponent.LEVERAGE,
                    score=self._component_score(
                        components,
                        HealthComponent.LEVERAGE,
                    ),
                    description=(
                        "Debt is high relative to shareholder equity."
                    ),
                    trigger_value=metrics.debt_to_equity,
                    threshold=Decimal("2"),
                    recommendation_hint=(
                        "Review debt reduction, refinancing, and "
                        "capital structure options."
                    ),
                    evidence=[
                        f"Debt-to-equity = "
                        f"{metrics.debt_to_equity:.2f}"
                    ],
                )
            )

        # Revenue decline
        if (
            metrics.revenue_growth is not None
            and metrics.revenue_growth < Decimal("-10")
        ):
            risks.append(
                HealthRisk(
                    name="significant_revenue_decline",
                    risk_level=RiskLevel.HIGH,
                    component=HealthComponent.GROWTH,
                    score=self._component_score(
                        components,
                        HealthComponent.GROWTH,
                    ),
                    description=(
                        "Revenue has declined materially."
                    ),
                    trigger_value=metrics.revenue_growth,
                    threshold=Decimal("-10"),
                    recommendation_hint=(
                        "Investigate customer churn, pricing, sales "
                        "pipeline, and market conditions."
                    ),
                    evidence=[
                        f"Revenue growth = "
                        f"{metrics.revenue_growth:.2f}%"
                    ],
                )
            )

        # Expense pressure
        if (
            metrics.expense_ratio is not None
            and metrics.expense_ratio > Decimal("80")
        ):
            risks.append(
                HealthRisk(
                    name="high_expense_ratio",
                    risk_level=RiskLevel.MEDIUM,
                    component=HealthComponent.COST_CONTROL,
                    score=self._component_score(
                        components,
                        HealthComponent.COST_CONTROL,
                    ),
                    description=(
                        "Operating expenses consume a large proportion "
                        "of revenue."
                    ),
                    trigger_value=metrics.expense_ratio,
                    threshold=Decimal("80"),
                    recommendation_hint=(
                        "Identify major cost drivers and evaluate "
                        "cost-reduction opportunities."
                    ),
                    evidence=[
                        f"Expense ratio = "
                        f"{metrics.expense_ratio:.2f}%"
                    ],
                )
            )

        # Receivables
        if (
            metrics.receivable_days is not None
            and metrics.receivable_days > Decimal("90")
        ):
            risks.append(
                HealthRisk(
                    name="slow_receivables_collection",
                    risk_level=RiskLevel.MEDIUM,
                    component=HealthComponent.EFFICIENCY,
                    score=self._component_score(
                        components,
                        HealthComponent.EFFICIENCY,
                    ),
                    description=(
                        "Receivables are taking longer than expected "
                        "to convert into cash."
                    ),
                    trigger_value=metrics.receivable_days,
                    threshold=Decimal("90"),
                    recommendation_hint=(
                        "Review overdue invoices, credit terms, and "
                        "collection processes."
                    ),
                    evidence=[
                        f"Receivable days = "
                        f"{metrics.receivable_days:.2f}"
                    ],
                )
            )

        return risks

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_score(
        score: Decimal,
    ) -> HealthCategory:
        """Classify overall score."""
        if score >= Decimal("85"):
            return HealthCategory.EXCELLENT

        if score >= Decimal("70"):
            return HealthCategory.HEALTHY

        if score >= Decimal("55"):
            return HealthCategory.STABLE

        if score >= Decimal("35"):
            return HealthCategory.WEAK

        return HealthCategory.CRITICAL

    @staticmethod
    def classify_risk(
        score: Decimal,
        risks: Sequence[HealthRisk],
    ) -> RiskLevel:
        """Determine overall risk severity."""
        if any(
            risk.risk_level == RiskLevel.CRITICAL
            for risk in risks
        ):
            return RiskLevel.CRITICAL

        if any(
            risk.risk_level == RiskLevel.HIGH
            for risk in risks
        ):
            return RiskLevel.HIGH

        if score < Decimal("55"):
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_confidence(
        metrics: FinancialMetrics,
    ) -> Decimal:
        """
        Estimate score confidence based on metric availability.

        This does not measure statistical confidence.
        It measures how complete the supplied financial information is.
        """
        important_fields = (
            metrics.revenue,
            metrics.gross_profit,
            metrics.operating_profit,
            metrics.net_profit,
            metrics.current_assets,
            metrics.current_liabilities,
            metrics.total_assets,
            metrics.total_liabilities,
            metrics.total_equity,
            metrics.operating_cash_flow,
        )

        available = sum(
            1
            for value in important_fields
            if value != ZERO
        )

        ratio = (
            Decimal(available)
            / Decimal(len(important_fields))
        )

        # Keep a reasonable floor because a score can still be useful
        # with partial data, while making the incompleteness explicit.
        return round_score(
            Decimal("0.50")
            + ratio * Decimal("0.50")
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def generate_summary(
        *,
        overall_score: Decimal,
        category: HealthCategory,
        risk_level: RiskLevel,
        components: Sequence[ComponentScore],
        risks: Sequence[HealthRisk],
    ) -> str:
        """Generate a concise human-readable summary."""
        weakest = (
            min(
                components,
                key=lambda component: component.score,
            )
            if components
            else None
        )

        strongest = (
            max(
                components,
                key=lambda component: component.score,
            )
            if components
            else None
        )

        summary = (
            f"Financial health score is {overall_score:.2f}/100, "
            f"classified as {category.value} with "
            f"{risk_level.value} risk."
        )

        if strongest:
            summary += (
                f" Strongest area: "
                f"{strongest.component.value} "
                f"({strongest.score:.2f})."
            )

        if weakest:
            summary += (
                f" Weakest area: "
                f"{weakest.component.value} "
                f"({weakest.score:.2f})."
            )

        if risks:
            summary += (
                f" {len(risks)} material risk signal"
                f"{'s' if len(risks) != 1 else ''} detected."
            )

        return summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_component(
        self,
        component: HealthComponent,
        score: Decimal,
        strengths: list[str],
        weaknesses: list[str],
        evidence: list[str],
    ) -> ComponentScore:
        score = round_score(
            clamp(score)
        )

        weight = self.weights[component]

        weighted_score = round_score(
            score * weight
        )

        return ComponentScore(
            component=component,
            score=score,
            weight=weight,
            weighted_score=weighted_score,
            status=self.classify_score(score),
            strengths=strengths,
            weaknesses=weaknesses,
            risk_level=self._component_risk(score),
            evidence=evidence,
        )

    @staticmethod
    def _component_risk(
        score: Decimal,
    ) -> RiskLevel:
        if score < Decimal("30"):
            return RiskLevel.CRITICAL

        if score < Decimal("50"):
            return RiskLevel.HIGH

        if score < Decimal("70"):
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    @staticmethod
    def _average_scores(
        scores: Sequence[Decimal],
        *,
        default: Decimal,
    ) -> Decimal:
        if not scores:
            return default

        return sum(scores, ZERO) / Decimal(len(scores))

    @staticmethod
    def _ratio_score(
        value: Decimal,
        *,
        critical: Decimal,
        healthy: Decimal,
        excellent: Decimal,
    ) -> Decimal:
        if value <= critical:
            return Decimal("20")

        if value >= excellent:
            return Decimal("100")

        if value >= healthy:
            return (
                Decimal("70")
                + (
                    (value - healthy)
                    / (excellent - healthy)
                ) * Decimal("30")
            )

        return (
            Decimal("20")
            + (
                (value - critical)
                / (healthy - critical)
            ) * Decimal("50")
        )

    @staticmethod
    def _inverse_ratio_score(
        value: Decimal,
        *,
        excellent: Decimal,
        healthy: Decimal,
        critical: Decimal,
    ) -> Decimal:
        if value <= excellent:
            return Decimal("100")

        if value >= critical:
            return Decimal("10")

        if value <= healthy:
            return (
                Decimal("70")
                + (
                    (healthy - value)
                    / (healthy - excellent)
                ) * Decimal("30")
            )

        return (
            Decimal("10")
            + (
                (critical - value)
                / (critical - healthy)
            ) * Decimal("60")
        )

    @staticmethod
    def _component_score(
        components: Sequence[ComponentScore],
        component: HealthComponent,
    ) -> Decimal:
        for item in components:
            if item.component == component:
                return item.score

        return Decimal("50")

    @staticmethod
    def _collect_strengths(
        components: Sequence[ComponentScore],
    ) -> list[str]:
        strengths: list[str] = []

        for component in components:
            if component.score >= Decimal("75"):
                strengths.extend(component.strengths)

        return list(dict.fromkeys(strengths))

    @staticmethod
    def _collect_weaknesses(
        components: Sequence[ComponentScore],
    ) -> list[str]:
        weaknesses: list[str] = []

        for component in components:
            if component.score < Decimal("60"):
                weaknesses.extend(component.weaknesses)

        return list(dict.fromkeys(weaknesses))

    @staticmethod
    def _normalize_weights(
        weights: Mapping[
            HealthComponent,
            Decimal | float | int,
        ],
    ) -> dict[HealthComponent, Decimal]:
        normalized = {
            component: to_decimal(value)
            for component, value in weights.items()
        }

        for component in HealthComponent:
            normalized.setdefault(
                component,
                ZERO,
            )

        total = sum(
            normalized.values(),
            ZERO,
        )

        if total <= ZERO:
            raise ValueError(
                "Health score weights must sum to a positive value."
            )

        return {
            component: value / total
            for component, value in normalized.items()
        }


# ============================================================================
# Convenience Functions
# ============================================================================


def calculate_health_score(
    metrics: FinancialMetrics,
) -> HealthScoreResult:
    """
    Calculate financial health using the default scoring model.
    """
    scorer = FinancialHealthScorer()
    return scorer.calculate(metrics)


def calculate_financial_health(
    metrics: FinancialMetrics,
) -> HealthScoreResult:
    """Alias for calculate_health_score()."""
    return calculate_health_score(metrics)


def score_profitability(
    metrics: FinancialMetrics,
) -> ComponentScore:
    """Convenience profitability scorer."""
    return FinancialHealthScorer().score_profitability(metrics)


def score_liquidity(
    metrics: FinancialMetrics,
) -> ComponentScore:
    """Convenience liquidity scorer."""
    return FinancialHealthScorer().score_liquidity(metrics)


def score_leverage(
    metrics: FinancialMetrics,
) -> ComponentScore:
    """Convenience leverage scorer."""
    return FinancialHealthScorer().score_leverage(metrics)


def score_cash_flow(
    metrics: FinancialMetrics,
) -> ComponentScore:
    """Convenience cash-flow scorer."""
    return FinancialHealthScorer().score_cash_flow(metrics)


def identify_financial_risks(
    metrics: FinancialMetrics,
) -> list[HealthRisk]:
    """Return health risks without requiring callers to manage components."""
    scorer = FinancialHealthScorer()

    metrics.derive_missing_metrics()

    components = [
        scorer.score_profitability(metrics),
        scorer.score_liquidity(metrics),
        scorer.score_leverage(metrics),
        scorer.score_cash_flow(metrics),
        scorer.score_growth(metrics),
        scorer.score_efficiency(metrics),
        scorer.score_cost_control(metrics),
        scorer.score_stability(metrics),
    ]

    return scorer.identify_risks(
        metrics,
        components,
    )


def classify_health_score(
    score: Decimal | float | int,
) -> HealthCategory:
    """Classify a raw 0-100 health score."""
    return FinancialHealthScorer.classify_score(
        to_decimal(score)
    )


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "HealthCategory",
    "HealthComponent",
    "RiskLevel",
    "TrendDirection",
    "FinancialMetrics",
    "ComponentScore",
    "HealthRisk",
    "HealthScoreResult",
    "FinancialHealthScorer",
    "calculate_health_score",
    "calculate_financial_health",
    "score_profitability",
    "score_liquidity",
    "score_leverage",
    "score_cash_flow",
    "identify_financial_risks",
    "classify_health_score",
]