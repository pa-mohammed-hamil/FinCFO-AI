"""
FinCo AI - Financial Root Cause Analysis Engine

File:
    backend/app/financial/root_cause.py

Purpose:
    Deterministic financial root-cause analysis layer.

Responsibilities:
    - Identify likely financial drivers behind KPI changes
    - Decompose revenue, cost, margin, profit, cash-flow and ratio movements
    - Compare current vs previous periods
    - Rank competing root-cause hypotheses
    - Attach quantitative evidence
    - Calculate contribution / impact
    - Detect compound causes
    - Generate management-readable explanations
    - Provide structured inputs for:
        * Supervisor Agent
        * RAG evidence retrieval
        * Forecasting
        * What-If analysis
        * Recommendation Engine
        * Alert Engine
        * Audit logging

Important:
    This module performs deterministic financial attribution.

    It should NOT claim statistical causality unless supported by
    an appropriate causal inference / ML layer.

Architecture:

    Financial Data
          │
          ▼
    Financial Metrics
          │
          ▼
    Historical Comparison
          │
          ▼
    root_cause.py
          │
     ┌────┼─────────────┐
     ▼    ▼             ▼
  Drivers Evidence   Hypotheses
     │    │             │
     └────┼─────────────┘
          ▼
    Ranked Root Causes
          │
          ▼
    Supervisor Agent
          │
     ┌────┼─────────────┐
     ▼    ▼             ▼
    RAG Forecast     What-If
     │    │             │
     └────┼─────────────┘
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
SCORE_QUANT = Decimal("0.01")


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
    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_percent(
    value: Decimal,
) -> Decimal:
    return value.quantize(
        PERCENT_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_score(
    value: Decimal,
) -> Decimal:
    return value.quantize(
        SCORE_QUANT,
        rounding=ROUND_HALF_UP,
    )


def safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    if denominator == ZERO:
        return default

    return numerator / denominator


def percentage(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal:
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

    Returns None when the previous value is zero and
    the current value is non-zero.
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


class RootCauseCategory(str, Enum):
    """High-level financial root-cause category."""

    REVENUE = "revenue"
    COGS = "cogs"
    OPERATING_EXPENSE = "operating_expense"
    PRICING = "pricing"
    VOLUME = "volume"
    MIX = "mix"
    CUSTOMER = "customer"
    PRODUCT = "product"
    WORKING_CAPITAL = "working_capital"
    CASH_FLOW = "cash_flow"
    DEBT = "debt"
    INTEREST = "interest"
    TAX = "tax"
    MARGIN = "margin"
    OTHER = "other"


class RootCauseDirection(str, Enum):
    """Direction of the identified driver."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RootCauseSeverity(str, Enum):
    """Severity of root cause."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RootCauseConfidence(str, Enum):
    """Confidence classification."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class RootCauseMetric(str, Enum):
    """Financial metrics that can be explained."""

    REVENUE = "revenue"
    GROSS_PROFIT = "gross_profit"
    EBITDA = "ebitda"
    OPERATING_PROFIT = "operating_profit"
    NET_PROFIT = "net_profit"

    GROSS_MARGIN = "gross_margin"
    OPERATING_MARGIN = "operating_margin"
    NET_MARGIN = "net_margin"

    COGS = "cogs"
    OPERATING_EXPENSES = "operating_expenses"

    OPERATING_CASH_FLOW = "operating_cash_flow"
    FREE_CASH_FLOW = "free_cash_flow"

    RECEIVABLES = "receivables"
    INVENTORY = "inventory"
    PAYABLES = "payables"

    DEBT = "debt"
    INTEREST_EXPENSE = "interest_expense"


class RootCauseSignalType(str, Enum):
    """Types of evidence used in root-cause analysis."""

    PERIOD_CHANGE = "period_change"
    TREND = "trend"
    CONCENTRATION = "concentration"
    ANOMALY = "anomaly"
    RATIO_CHANGE = "ratio_change"
    THRESHOLD = "threshold"
    CONTRIBUTION = "contribution"
    CORRELATION = "correlation"
    STRUCTURAL = "structural"


# ============================================================================
# Input Models
# ============================================================================


@dataclass(slots=True)
class FinancialSnapshot:
    """
    Normalized financial snapshot for a single period.
    """

    revenue: Decimal = ZERO
    gross_profit: Decimal = ZERO
    ebitda: Decimal = ZERO
    operating_profit: Decimal = ZERO
    net_profit: Decimal = ZERO

    cogs: Decimal = ZERO
    operating_expenses: Decimal = ZERO

    interest_expense: Decimal = ZERO
    tax_expense: Decimal = ZERO

    operating_cash_flow: Decimal = ZERO
    investing_cash_flow: Decimal = ZERO
    financing_cash_flow: Decimal = ZERO
    free_cash_flow: Decimal = ZERO

    receivables: Decimal = ZERO
    inventory: Decimal = ZERO
    payables: Decimal = ZERO

    debt: Decimal = ZERO
    cash: Decimal = ZERO

    gross_margin: Decimal | None = None
    operating_margin: Decimal | None = None
    net_margin: Decimal | None = None

    current_ratio: Decimal | None = None
    debt_to_equity: Decimal | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "revenue",
            "gross_profit",
            "ebitda",
            "operating_profit",
            "net_profit",
            "cogs",
            "operating_expenses",
            "interest_expense",
            "tax_expense",
            "operating_cash_flow",
            "investing_cash_flow",
            "financing_cash_flow",
            "free_cash_flow",
            "receivables",
            "inventory",
            "payables",
            "debt",
            "cash",
        ):
            setattr(
                self,
                name,
                to_decimal(
                    getattr(self, name)
                ),
            )


@dataclass(slots=True)
class FinancialDriver:
    """
    A measurable financial driver.

    Example:
        Revenue declined by 12%.
        This contributes negatively to net profit.
    """

    code: str

    name: str

    category: RootCauseCategory

    direction: RootCauseDirection

    current_value: Decimal

    previous_value: Decimal

    absolute_change: Decimal

    percentage_change: Decimal | None

    impact: Decimal

    impact_percent: Decimal

    score: Decimal

    severity: RootCauseSeverity

    confidence: RootCauseConfidence

    explanation: str

    evidence: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "code": self.code,
            "name": self.name,
            "category": self.category.value,
            "direction": self.direction.value,
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
                if self.percentage_change is not None
                else None
            ),
            "impact": str(self.impact),
            "impact_percent": str(
                self.impact_percent
            ),
            "score": str(self.score),
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RootCauseHypothesis:
    """
    Ranked explanation for a financial change.
    """

    rank: int

    title: str

    category: RootCauseCategory

    metric: RootCauseMetric

    direction: RootCauseDirection

    impact: Decimal

    impact_percent: Decimal

    score: Decimal

    severity: RootCauseSeverity

    confidence: RootCauseConfidence

    explanation: str

    evidence: list[str] = field(
        default_factory=list
    )

    supporting_drivers: list[str] = field(
        default_factory=list
    )

    contradicting_evidence: list[str] = field(
        default_factory=list
    )

    recommended_next_checks: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "rank": self.rank,
            "title": self.title,
            "category": self.category.value,
            "metric": self.metric.value,
            "direction": self.direction.value,
            "impact": str(self.impact),
            "impact_percent": str(
                self.impact_percent
            ),
            "score": str(self.score),
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "explanation": self.explanation,
            "evidence": self.evidence,
            "supporting_drivers": (
                self.supporting_drivers
            ),
            "contradicting_evidence": (
                self.contradicting_evidence
            ),
            "recommended_next_checks": (
                self.recommended_next_checks
            ),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RootCauseEvidence:
    """
    Evidence item supplied to the root-cause engine.

    This can later be linked to:
        - database records
        - RAG chunks
        - invoices
        - financial statements
        - transactions
        - forecasts
        - documents
    """

    evidence_id: str

    signal_type: RootCauseSignalType

    title: str

    description: str

    value: Decimal | None = None

    reference_value: Decimal | None = None

    impact_percent: Decimal | None = None

    source_type: str = "financial_analysis"

    source_id: str | None = None

    citation: str | None = None

    reliability: Decimal = Decimal("1")

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.reliability = max(
            ZERO,
            min(
                ONE,
                to_decimal(
                    self.reliability
                ),
            ),
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "evidence_id": self.evidence_id,
            "signal_type": (
                self.signal_type.value
            ),
            "title": self.title,
            "description": self.description,
            "value": (
                str(self.value)
                if self.value is not None
                else None
            ),
            "reference_value": (
                str(self.reference_value)
                if self.reference_value is not None
                else None
            ),
            "impact_percent": (
                str(self.impact_percent)
                if self.impact_percent is not None
                else None
            ),
            "source_type": self.source_type,
            "source_id": self.source_id,
            "citation": self.citation,
            "reliability": str(
                self.reliability
            ),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class RootCauseAnalysisResult:
    """
    Complete root-cause analysis result.
    """

    target_metric: RootCauseMetric

    current_value: Decimal

    previous_value: Decimal

    change: Decimal

    change_percent: Decimal | None

    direction: RootCauseDirection

    drivers: list[FinancialDriver]

    hypotheses: list[RootCauseHypothesis]

    evidence: list[RootCauseEvidence]

    primary_root_cause: RootCauseHypothesis | None

    confidence: RootCauseConfidence

    severity: RootCauseSeverity

    explanation: str

    summary: str

    next_actions: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "target_metric": (
                self.target_metric.value
            ),
            "current_value": str(
                self.current_value
            ),
            "previous_value": str(
                self.previous_value
            ),
            "change": str(self.change),
            "change_percent": (
                str(self.change_percent)
                if self.change_percent is not None
                else None
            ),
            "direction": self.direction.value,
            "drivers": [
                driver.to_dict()
                for driver in self.drivers
            ],
            "hypotheses": [
                hypothesis.to_dict()
                for hypothesis in self.hypotheses
            ],
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "primary_root_cause": (
                self.primary_root_cause.to_dict()
                if self.primary_root_cause
                else None
            ),
            "confidence": self.confidence.value,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "summary": self.summary,
            "next_actions": self.next_actions,
            "metadata": self.metadata,
        }


# ============================================================================
# Root Cause Analyzer
# ============================================================================


class RootCauseAnalyzer:
    """
    Deterministic financial root-cause analyzer.

    The analyzer ranks explanations based on measurable financial
    movements. It does not claim causal certainty.
    """

    def __init__(
        self,
        *,
        material_change_threshold: Decimal = Decimal("5"),
        significant_change_threshold: Decimal = Decimal("10"),
        severe_change_threshold: Decimal = Decimal("20"),
        high_confidence_threshold: Decimal = Decimal("75"),
        medium_confidence_threshold: Decimal = Decimal("50"),
    ) -> None:

        self.material_change_threshold = (
            to_decimal(
                material_change_threshold
            )
        )

        self.significant_change_threshold = (
            to_decimal(
                significant_change_threshold
            )
        )

        self.severe_change_threshold = (
            to_decimal(
                severe_change_threshold
            )
        )

        self.high_confidence_threshold = (
            to_decimal(
                high_confidence_threshold
            )
        )

        self.medium_confidence_threshold = (
            to_decimal(
                medium_confidence_threshold
            )
        )

    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        current: FinancialSnapshot
        | Mapping[str, Any],
        previous: FinancialSnapshot
        | Mapping[str, Any],
        target_metric: RootCauseMetric
        | str,
        *,
        evidence: Sequence[
            RootCauseEvidence
            | Mapping[str, Any]
        ] | None = None,
    ) -> RootCauseAnalysisResult:

        current_snapshot = (
            self.normalize_snapshot(
                current
            )
        )

        previous_snapshot = (
            self.normalize_snapshot(
                previous
            )
        )

        metric = self.normalize_metric(
            target_metric
        )

        current_value = (
            self.metric_value(
                current_snapshot,
                metric,
            )
        )

        previous_value = (
            self.metric_value(
                previous_snapshot,
                metric,
            )
        )

        change = (
            current_value
            - previous_value
        )

        change_percent = (
            percentage_change(
                current_value,
                previous_value,
            )
        )

        direction = (
            self.determine_direction(
                change
            )
        )

        normalized_evidence = (
            self.normalize_evidence(
                evidence or []
            )
        )

        drivers = (
            self.identify_drivers(
                current_snapshot,
                previous_snapshot,
                metric,
            )
        )

        hypotheses = (
            self.build_hypotheses(
                current_snapshot,
                previous_snapshot,
                metric,
                drivers,
                normalized_evidence,
            )
        )

        hypotheses = (
            self.rank_hypotheses(
                hypotheses
            )
        )

        primary = (
            hypotheses[0]
            if hypotheses
            else None
        )

        confidence = (
            self.determine_overall_confidence(
                hypotheses
            )
        )

        severity = (
            self.determine_severity(
                change_percent
            )
        )

        explanation = (
            self.generate_explanation(
                metric=metric,
                current=current_value,
                previous=previous_value,
                change=change,
                change_percent=change_percent,
                direction=direction,
                primary=primary,
                hypotheses=hypotheses,
            )
        )

        summary = (
            self.generate_summary(
                metric=metric,
                change_percent=change_percent,
                primary=primary,
                confidence=confidence,
            )
        )

        next_actions = (
            self.generate_next_checks(
                metric,
                primary,
                hypotheses,
            )
        )

        return RootCauseAnalysisResult(
            target_metric=metric,
            current_value=round_money(
                current_value
            ),
            previous_value=round_money(
                previous_value
            ),
            change=round_money(
                change
            ),
            change_percent=change_percent,
            direction=direction,
            drivers=drivers,
            hypotheses=hypotheses,
            evidence=normalized_evidence,
            primary_root_cause=primary,
            confidence=confidence,
            severity=severity,
            explanation=explanation,
            summary=summary,
            next_actions=next_actions,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_snapshot(
        self,
        snapshot: FinancialSnapshot
        | Mapping[str, Any],
    ) -> FinancialSnapshot:

        if isinstance(
            snapshot,
            FinancialSnapshot,
        ):
            return snapshot

        data = dict(snapshot)

        return FinancialSnapshot(
            revenue=to_decimal(
                data.get("revenue")
            ),
            gross_profit=to_decimal(
                data.get("gross_profit")
            ),
            ebitda=to_decimal(
                data.get("ebitda")
            ),
            operating_profit=to_decimal(
                data.get("operating_profit")
            ),
            net_profit=to_decimal(
                data.get("net_profit")
            ),
            cogs=to_decimal(
                data.get(
                    "cogs",
                    data.get(
                        "cost_of_goods_sold"
                    ),
                )
            ),
            operating_expenses=to_decimal(
                data.get(
                    "operating_expenses",
                    data.get("opex"),
                )
            ),
            interest_expense=to_decimal(
                data.get(
                    "interest_expense",
                    data.get("interest"),
                )
            ),
            tax_expense=to_decimal(
                data.get(
                    "tax_expense",
                    data.get("tax"),
                )
            ),
            operating_cash_flow=to_decimal(
                data.get(
                    "operating_cash_flow"
                )
            ),
            investing_cash_flow=to_decimal(
                data.get(
                    "investing_cash_flow"
                )
            ),
            financing_cash_flow=to_decimal(
                data.get(
                    "financing_cash_flow"
                )
            ),
            free_cash_flow=to_decimal(
                data.get(
                    "free_cash_flow"
                )
            ),
            receivables=to_decimal(
                data.get("receivables")
            ),
            inventory=to_decimal(
                data.get("inventory")
            ),
            payables=to_decimal(
                data.get("payables")
            ),
            debt=to_decimal(
                data.get("debt")
            ),
            cash=to_decimal(
                data.get("cash")
            ),
            gross_margin=(
                to_decimal(
                    data["gross_margin"]
                )
                if data.get(
                    "gross_margin"
                )
                is not None
                else None
            ),
            operating_margin=(
                to_decimal(
                    data["operating_margin"]
                )
                if data.get(
                    "operating_margin"
                )
                is not None
                else None
            ),
            net_margin=(
                to_decimal(
                    data["net_margin"]
                )
                if data.get(
                    "net_margin"
                )
                is not None
                else None
            ),
            current_ratio=(
                to_decimal(
                    data["current_ratio"]
                )
                if data.get(
                    "current_ratio"
                )
                is not None
                else None
            ),
            debt_to_equity=(
                to_decimal(
                    data["debt_to_equity"]
                )
                if data.get(
                    "debt_to_equity"
                )
                is not None
                else None
            ),
            metadata=dict(
                data.get(
                    "metadata",
                    {},
                )
                or {}
            ),
        )

    def normalize_metric(
        self,
        metric: RootCauseMetric
        | str,
    ) -> RootCauseMetric:

        if isinstance(
            metric,
            RootCauseMetric,
        ):
            return metric

        return RootCauseMetric(
            str(metric).lower()
        )

    def normalize_evidence(
        self,
        evidence: Sequence[
            RootCauseEvidence
            | Mapping[str, Any]
        ],
    ) -> list[RootCauseEvidence]:

        result: list[
            RootCauseEvidence
        ] = []

        for index, item in enumerate(
            evidence
        ):

            if isinstance(
                item,
                RootCauseEvidence,
            ):
                result.append(item)
                continue

            data = dict(item)

            signal_raw = data.get(
                "signal_type",
                RootCauseSignalType.STRUCTURAL.value,
            )

            try:
                signal_type = (
                    signal_raw
                    if isinstance(
                        signal_raw,
                        RootCauseSignalType,
                    )
                    else RootCauseSignalType(
                        str(
                            signal_raw
                        ).lower()
                    )
                )
            except ValueError:
                signal_type = (
                    RootCauseSignalType.STRUCTURAL
                )

            result.append(
                RootCauseEvidence(
                    evidence_id=str(
                        data.get(
                            "evidence_id",
                            f"evidence-{index + 1}",
                        )
                    ),
                    signal_type=signal_type,
                    title=str(
                        data.get(
                            "title",
                            "Financial evidence",
                        )
                    ),
                    description=str(
                        data.get(
                            "description",
                            "",
                        )
                    ),
                    value=(
                        to_decimal(
                            data["value"]
                        )
                        if data.get(
                            "value"
                        )
                        is not None
                        else None
                    ),
                    reference_value=(
                        to_decimal(
                            data[
                                "reference_value"
                            ]
                        )
                        if data.get(
                            "reference_value"
                        )
                        is not None
                        else None
                    ),
                    impact_percent=(
                        to_decimal(
                            data[
                                "impact_percent"
                            ]
                        )
                        if data.get(
                            "impact_percent"
                        )
                        is not None
                        else None
                    ),
                    source_type=str(
                        data.get(
                            "source_type",
                            "financial_analysis",
                        )
                    ),
                    source_id=(
                        str(
                            data["source_id"]
                        )
                        if data.get(
                            "source_id"
                        )
                        is not None
                        else None
                    ),
                    citation=(
                        str(
                            data["citation"]
                        )
                        if data.get(
                            "citation"
                        )
                        is not None
                        else None
                    ),
                    reliability=to_decimal(
                        data.get(
                            "reliability",
                            ONE,
                        )
                    ),
                    metadata=dict(
                        data.get(
                            "metadata",
                            {},
                        )
                        or {}
                    ),
                )
            )

        return result

    # ------------------------------------------------------------------
    # Metric Values
    # ------------------------------------------------------------------

    def metric_value(
        self,
        snapshot: FinancialSnapshot,
        metric: RootCauseMetric,
    ) -> Decimal:

        direct = {
            RootCauseMetric.REVENUE: (
                snapshot.revenue
            ),
            RootCauseMetric.GROSS_PROFIT: (
                snapshot.gross_profit
            ),
            RootCauseMetric.EBITDA: (
                snapshot.ebitda
            ),
            RootCauseMetric.OPERATING_PROFIT: (
                snapshot.operating_profit
            ),
            RootCauseMetric.NET_PROFIT: (
                snapshot.net_profit
            ),
            RootCauseMetric.COGS: (
                snapshot.cogs
            ),
            RootCauseMetric.OPERATING_EXPENSES: (
                snapshot.operating_expenses
            ),
            RootCauseMetric.INTEREST_EXPENSE: (
                snapshot.interest_expense
            ),
            RootCauseMetric.OPERATING_CASH_FLOW: (
                snapshot.operating_cash_flow
            ),
            RootCauseMetric.FREE_CASH_FLOW: (
                snapshot.free_cash_flow
            ),
            RootCauseMetric.RECEIVABLES: (
                snapshot.receivables
            ),
            RootCauseMetric.INVENTORY: (
                snapshot.inventory
            ),
            RootCauseMetric.PAYABLES: (
                snapshot.payables
            ),
            RootCauseMetric.DEBT: (
                snapshot.debt
            ),
        }

        if metric in direct:
            return direct[metric]

        if metric == RootCauseMetric.GROSS_MARGIN:

            if snapshot.gross_margin is not None:
                return snapshot.gross_margin

            return percentage(
                snapshot.gross_profit,
                snapshot.revenue,
            )

        if metric == RootCauseMetric.OPERATING_MARGIN:

            if (
                snapshot.operating_margin
                is not None
            ):
                return snapshot.operating_margin

            return percentage(
                snapshot.operating_profit,
                snapshot.revenue,
            )

        if metric == RootCauseMetric.NET_MARGIN:

            if snapshot.net_margin is not None:
                return snapshot.net_margin

            return percentage(
                snapshot.net_profit,
                snapshot.revenue,
            )

        return ZERO

    # ------------------------------------------------------------------
    # Direction / Severity
    # ------------------------------------------------------------------

    def determine_direction(
        self,
        change: Decimal,
    ) -> RootCauseDirection:

        if change > ZERO:
            return RootCauseDirection.POSITIVE

        if change < ZERO:
            return RootCauseDirection.NEGATIVE

        return RootCauseDirection.NEUTRAL

    def determine_severity(
        self,
        change_percent: Decimal | None,
    ) -> RootCauseSeverity:

        if change_percent is None:
            return RootCauseSeverity.HIGH

        magnitude = abs(
            change_percent
        )

        if magnitude >= Decimal("30"):
            return RootCauseSeverity.CRITICAL

        if magnitude >= self.severe_change_threshold:
            return RootCauseSeverity.HIGH

        if magnitude >= self.significant_change_threshold:
            return RootCauseSeverity.MEDIUM

        if magnitude >= self.material_change_threshold:
            return RootCauseSeverity.LOW

        return RootCauseSeverity.INFO

    # ------------------------------------------------------------------
    # Driver Identification
    # ------------------------------------------------------------------

    def identify_drivers(
        self,
        current: FinancialSnapshot,
        previous: FinancialSnapshot,
        target_metric: RootCauseMetric,
    ) -> list[FinancialDriver]:

        if target_metric == RootCauseMetric.REVENUE:
            return self.revenue_drivers(
                current,
                previous,
            )

        if target_metric in {
            RootCauseMetric.GROSS_PROFIT,
            RootCauseMetric.GROSS_MARGIN,
        }:
            return self.gross_profit_drivers(
                current,
                previous,
            )

        if target_metric in {
            RootCauseMetric.EBITDA,
            RootCauseMetric.OPERATING_PROFIT,
            RootCauseMetric.OPERATING_MARGIN,
        }:
            return self.operating_profit_drivers(
                current,
                previous,
            )

        if target_metric in {
            RootCauseMetric.NET_PROFIT,
            RootCauseMetric.NET_MARGIN,
        }:
            return self.net_profit_drivers(
                current,
                previous,
            )

        if target_metric in {
            RootCauseMetric.OPERATING_CASH_FLOW,
            RootCauseMetric.FREE_CASH_FLOW,
        }:
            return self.cash_flow_drivers(
                current,
                previous,
            )

        if target_metric in {
            RootCauseMetric.RECEIVABLES,
            RootCauseMetric.INVENTORY,
            RootCauseMetric.PAYABLES,
            RootCauseMetric.DEBT,
        }:
            return self.balance_sheet_drivers(
                current,
                previous,
            )

        return self.generic_drivers(
            current,
            previous,
        )

    # ------------------------------------------------------------------
    # Revenue Drivers
    # ------------------------------------------------------------------

    def revenue_drivers(
        self,
        current: FinancialSnapshot,
        previous: FinancialSnapshot,
    ) -> list[FinancialDriver]:

        drivers: list[
            FinancialDriver
        ] = []

        drivers.append(
            self.create_driver(
                code="revenue_change",
                name="Revenue movement",
                category=(
                    RootCauseCategory.REVENUE
                ),
                current=current.revenue,
                previous=previous.revenue,
                impact=(
                    current.revenue
                    - previous.revenue
                ),
                explanation=(
                    "Revenue changed between the "
                    "current and previous periods."
                ),
            )
        )

        return self.clean_drivers(
            drivers
        )

    # ------------------------------------------------------------------
    # Gross Profit Drivers
    # ------------------------------------------------------------------

    def gross_profit_drivers(
        self,
        current: FinancialSnapshot,
        previous: FinancialSnapshot,
    ) -> list[FinancialDriver]:

        revenue_change = (
            current.revenue
            - previous.revenue
        )

        cogs_change = (
            current.cogs
            - previous.cogs
        )

        gross_margin_current = (
            percentage(
                current.gross_profit,
                current.revenue,
            )
        )

        gross_margin_previous = (
            percentage(
                previous.gross_profit,
                previous.revenue,
            )
        )

        margin_change = (
            gross_margin_current
            - gross_margin_previous
        )

        drivers = [
            self.create_driver(
                code="revenue_change",
                name="Revenue movement",
                category=(
                    RootCauseCategory.REVENUE
                ),
                current=current.revenue,
                previous=previous.revenue,
                impact=revenue_change,
                explanation=(
                    "Changes in revenue directly "
                    "affect gross profit."
                ),
            ),
            self.create_driver(
                code="cogs_change",
                name="Cost of goods sold movement",
                category=(
                    RootCauseCategory.COGS
                ),
                current=current.cogs,
                previous=previous.cogs,
                impact=-cogs_change,
                explanation=(
                    "Changes in COGS affect the amount "
                    "of revenue retained as gross profit."
                ),
            ),
            self.create_ratio_driver(
                code="gross_margin_change",
                name="Gross margin movement",
                category=(
                    RootCauseCategory.MARGIN
                ),
                current=gross_margin_current,
                previous=gross_margin_previous,
                impact=(
                    current.gross_profit
                    - (
                        current.revenue
                        * safe_divide(
                            gross_margin_previous,
                            ONE_HUNDRED,
                        )
                    )
                ),
                explanation=(
                    "Gross-margin movement indicates "
                    "changes in pricing, cost structure, "
                    "or revenue mix."
                ),
            ),
        ]

        return self.clean_drivers(
            drivers
        )

    # ------------------------------------------------------------------
    # Operating Profit Drivers
    # ------------------------------------------------------------------

    def operating_profit_drivers(
        self,
        current: FinancialSnapshot,
        previous: FinancialSnapshot,
    ) -> list[FinancialDriver]:

        drivers = [
            self.create_driver(
                code="revenue_change",
                name="Revenue movement",
                category=(
                    RootCauseCategory.REVENUE
                ),
                current=current.revenue,
                previous=previous.revenue,
                impact=(
                    current.revenue
                    - previous.revenue
                ),
                explanation=(
                    "Revenue movement changes the "
                    "available gross-profit base."
                ),
            ),
            self.create_driver(
                code="cogs_change",
                name="COGS movement",
                category=(
                    RootCauseCategory.COGS
                ),
                current=current.cogs,
                previous=previous.cogs,
                impact=-(
                    current.cogs
                    - previous.cogs
                ),
                explanation=(
                    "Higher COGS reduces operating "
                    "profit unless offset by revenue "
                    "or other improvements."
                ),
            ),
            self.create_driver(
                code="opex_change",
                name="Operating expense movement",
                category=(
                    RootCauseCategory.OPERATING_EXPENSE
                ),
                current=current.operating_expenses,
                previous=previous.operating_expenses,
                impact=-(
                    current.operating_expenses
                    - previous.operating_expenses
                ),
                explanation=(
                    "Operating expense changes directly "
                    "affect operating profitability."
                ),
            ),
        ]

        return self.clean_drivers(
            drivers
        )

    # ------------------------------------------------------------------
    # Net Profit Drivers
    # ------------------------------------------------------------------

    def net_profit_drivers(
        self,
        current: FinancialSnapshot,
        previous: FinancialSnapshot,
    ) -> list[FinancialDriver]:

        operating_drivers = (
            self.operating_profit_drivers(
                current,
                previous,
            )
        )

        interest_change = (
            current.interest_expense
            - previous.interest_expense
        )

        tax_change = (
            current.tax_expense
            - previous.tax_expense
        )

        operating_drivers.extend(
            [
                self.create_driver(
                    code="interest_change",
                    name="Interest expense movement",
                    category=(
                        RootCauseCategory.INTEREST
                    ),
                    current=current.interest_expense,
                    previous=previous.interest_expense,
                    impact=-interest_change,
                    explanation=(
                        "Higher interest expense reduces "
                        "pre-tax and net profit."
                    ),
                ),
                self.create_driver(
                    code="tax_change",
                    name="Tax expense movement",
                    category=(
                        RootCauseCategory.TAX
                    ),
                    current=current.tax_expense,
                    previous=previous.tax_expense,
                    impact=-tax_change,
                    explanation=(
                        "Changes in tax expense affect "
                        "net profit after operating results."
                    ),
                ),
            ]
        )

        return self.clean_drivers(
            operating_drivers
        )

    # ------------------------------------------------------------------
    # Cash Flow Drivers
    # ------------------------------------------------------------------

    def cash_flow_drivers(
        self,
        current: FinancialSnapshot,
        previous: FinancialSnapshot,
    ) -> list[FinancialDriver]:

        receivables_change = (
            current.receivables
            - previous.receivables
        )

        inventory_change = (
            current.inventory
            - previous.inventory
        )

        payables_change = (
            current.payables
            - previous.payables
        )

        drivers = [
            self.create_driver(
                code="profit_change",
                name="Profit movement",
                category=(
                    RootCauseCategory.REVENUE
                ),
                current=current.net_profit,
                previous=previous.net_profit,
                impact=(
                    current.net_profit
                    - previous.net_profit
                ),
                explanation=(
                    "Profitability changes can affect "
                    "operating cash generation."
                ),
            ),
            self.create_driver(
                code="receivables_change",
                name="Receivables movement",
                category=(
                    RootCauseCategory.WORKING_CAPITAL
                ),
                current=current.receivables,
                previous=previous.receivables,
                impact=-receivables_change,
                explanation=(
                    "An increase in receivables can "
                    "consume operating cash."
                ),
            ),
            self.create_driver(
                code="inventory_change",
                name="Inventory movement",
                category=(
                    RootCauseCategory.WORKING_CAPITAL
                ),
                current=current.inventory,
                previous=previous.inventory,
                impact=-inventory_change,
                explanation=(
                    "An increase in inventory can "
                    "consume operating cash."
                ),
            ),
            self.create_driver(
                code="payables_change",
                name="Payables movement",
                category=(
                    RootCauseCategory.WORKING_CAPITAL
                ),
                current=current.payables,
                previous=previous.payables,
                impact=payables_change,
                explanation=(
                    "An increase in payables can "
                    "temporarily support operating cash."
                ),
            ),
            self.create_driver(
                code="debt_change",
                name="Debt movement",
                category=(
                    RootCauseCategory.DEBT
                ),
                current=current.debt,
                previous=previous.debt,
                impact=(
                    current.debt
                    - previous.debt
                ),
                explanation=(
                    "Debt changes can influence "
                    "financing cash flows and liquidity."
                ),
            ),
        ]

        return self.clean_drivers(
            drivers
        )

    # ------------------------------------------------------------------
    # Balance Sheet Drivers
    # ------------------------------------------------------------------

    def balance_sheet_drivers(
        self,
        current: FinancialSnapshot,
        previous: FinancialSnapshot,
    ) -> list[FinancialDriver]:

        drivers = [
            self.create_driver(
                code="receivables_change",
                name="Receivables movement",
                category=(
                    RootCauseCategory.WORKING_CAPITAL
                ),
                current=current.receivables,
                previous=previous.receivables,
                impact=(
                    current.receivables
                    - previous.receivables
                ),
                explanation=(
                    "Receivables movement indicates "
                    "changes in outstanding customer balances."
                ),
            ),
            self.create_driver(
                code="inventory_change",
                name="Inventory movement",
                category=(
                    RootCauseCategory.WORKING_CAPITAL
                ),
                current=current.inventory,
                previous=previous.inventory,
                impact=(
                    current.inventory
                    - previous.inventory
                ),
                explanation=(
                    "Inventory movement may indicate "
                    "changes in demand, purchasing, or "
                    "working-capital efficiency."
                ),
            ),
            self.create_driver(
                code="payables_change",
                name="Payables movement",
                category=(
                    RootCauseCategory.WORKING_CAPITAL
                ),
                current=current.payables,
                previous=previous.payables,
                impact=(
                    current.payables
                    - previous.payables
                ),
                explanation=(
                    "Payables movement can indicate "
                    "changes in supplier obligations "
                    "and payment timing."
                ),
            ),
            self.create_driver(
                code="debt_change",
                name="Debt movement",
                category=(
                    RootCauseCategory.DEBT
                ),
                current=current.debt,
                previous=previous.debt,
                impact=(
                    current.debt
                    - previous.debt
                ),
                explanation=(
                    "Debt movement affects leverage "
                    "and financing requirements."
                ),
            ),
        ]

        return self.clean_drivers(
            drivers
        )

    # ------------------------------------------------------------------
    # Generic Drivers
    # ------------------------------------------------------------------

    def generic_drivers(
        self,
        current: FinancialSnapshot,
        previous: FinancialSnapshot,
    ) -> list[FinancialDriver]:

        candidates = [
            (
                "revenue_change",
                "Revenue movement",
                RootCauseCategory.REVENUE,
                current.revenue,
                previous.revenue,
            ),
            (
                "cogs_change",
                "COGS movement",
                RootCauseCategory.COGS,
                current.cogs,
                previous.cogs,
            ),
            (
                "opex_change",
                "Operating expense movement",
                RootCauseCategory.OPERATING_EXPENSE,
                current.operating_expenses,
                previous.operating_expenses,
            ),
            (
                "interest_change",
                "Interest expense movement",
                RootCauseCategory.INTEREST,
                current.interest_expense,
                previous.interest_expense,
            ),
        ]

        drivers: list[
            FinancialDriver
        ] = []

        for (
            code,
            name,
            category,
            current_value,
            previous_value,
        ) in candidates:

            change = (
                current_value
                - previous_value
            )

            drivers.append(
                self.create_driver(
                    code=code,
                    name=name,
                    category=category,
                    current=current_value,
                    previous=previous_value,
                    impact=change,
                    explanation=(
                        f"{name} changed between "
                        "the two periods."
                    ),
                )
            )

        return self.clean_drivers(
            drivers
        )

    # ------------------------------------------------------------------
    # Driver Construction
    # ------------------------------------------------------------------

    def create_driver(
        self,
        *,
        code: str,
        name: str,
        category: RootCauseCategory,
        current: Decimal,
        previous: Decimal,
        impact: Decimal,
        explanation: str,
    ) -> FinancialDriver:

        change = (
            current - previous
        )

        change_percent = (
            percentage_change(
                current,
                previous,
            )
        )

        direction = (
            self.determine_direction(
                impact
            )
        )

        magnitude = abs(impact)

        score = self.calculate_driver_score(
            magnitude=magnitude,
            percentage_change=(
                change_percent
            ),
        )

        severity = (
            self.determine_severity(
                change_percent
            )
        )

        confidence = (
            self.confidence_from_score(
                score
            )
        )

        return FinancialDriver(
            code=code,
            name=name,
            category=category,
            direction=direction,
            current_value=round_money(
                current
            ),
            previous_value=round_money(
                previous
            ),
            absolute_change=round_money(
                change
            ),
            percentage_change=change_percent,
            impact=round_money(
                impact
            ),
            impact_percent=(
                round_percent(
                    percentage_change(
                        current,
                        previous,
                    )
                )
                if previous != ZERO
                else ZERO
            ),
            score=score,
            severity=severity,
            confidence=confidence,
            explanation=explanation,
            evidence=[
                (
                    f"Previous value: "
                    f"{round_money(previous)}"
                ),
                (
                    f"Current value: "
                    f"{round_money(current)}"
                ),
            ],
        )

    def create_ratio_driver(
        self,
        *,
        code: str,
        name: str,
        category: RootCauseCategory,
        current: Decimal,
        previous: Decimal,
        impact: Decimal,
        explanation: str,
    ) -> FinancialDriver:

        change = (
            current - previous
        )

        direction = (
            self.determine_direction(
                impact
            )
        )

        score = self.calculate_driver_score(
            magnitude=abs(impact),
            percentage_change=abs(
                change
            ),
        )

        return FinancialDriver(
            code=code,
            name=name,
            category=category,
            direction=direction,
            current_value=round_percent(
                current
            ),
            previous_value=round_percent(
                previous
            ),
            absolute_change=round_percent(
                change
            ),
            percentage_change=round_percent(
                change
            ),
            impact=round_money(
                impact
            ),
            impact_percent=round_percent(
                change
            ),
            score=score,
            severity=self.determine_severity(
                change
            ),
            confidence=(
                self.confidence_from_score(
                    score
                )
            ),
            explanation=explanation,
            evidence=[
                (
                    f"Previous ratio: "
                    f"{round_percent(previous)}%"
                ),
                (
                    f"Current ratio: "
                    f"{round_percent(current)}%"
                ),
            ],
        )

    def calculate_driver_score(
        self,
        *,
        magnitude: Decimal,
        percentage_change: Decimal | None,
    ) -> Decimal:

        magnitude_score = min(
            Decimal("60"),
            magnitude / Decimal("100000")
            * Decimal("60"),
        )

        percentage_score = (
            min(
                Decimal("40"),
                abs(
                    percentage_change
                    or ZERO
                ),
            )
        )

        return round_score(
            min(
                Decimal("100"),
                magnitude_score
                + percentage_score,
            )
        )

    def confidence_from_score(
        self,
        score: Decimal,
    ) -> RootCauseConfidence:

        if score >= self.high_confidence_threshold:
            return RootCauseConfidence.HIGH

        if score >= self.medium_confidence_threshold:
            return RootCauseConfidence.MEDIUM

        return RootCauseConfidence.LOW

    def clean_drivers(
        self,
        drivers: Sequence[
            FinancialDriver
        ],
    ) -> list[FinancialDriver]:

        result = [
            driver
            for driver in drivers
            if (
                driver.absolute_change
                != ZERO
                or driver.current_value
                != ZERO
                or driver.previous_value
                != ZERO
            )
        ]

        result.sort(
            key=lambda item: abs(
                item.impact
            ),
            reverse=True,
        )

        return result

    # ------------------------------------------------------------------
    # Hypothesis Generation
    # ------------------------------------------------------------------

    def build_hypotheses(
        self,
        current: FinancialSnapshot,
        previous: FinancialSnapshot,
        metric: RootCauseMetric,
        drivers: Sequence[FinancialDriver],
        evidence: Sequence[RootCauseEvidence],
    ) -> list[RootCauseHypothesis]:

        hypotheses: list[
            RootCauseHypothesis
        ] = []

        target_change = (
            self.metric_value(
                current,
                metric,
            )
            - self.metric_value(
                previous,
                metric,
            )
        )

        target_direction = (
            self.determine_direction(
                target_change
            )
        )

        for driver in drivers:

            if driver.direction == (
                RootCauseDirection.NEUTRAL
            ):
                continue

            alignment = (
                self.driver_alignment(
                    driver.direction,
                    target_direction,
                    metric,
                )
            )

            if alignment <= ZERO:
                continue

            supporting_evidence = (
                self.match_evidence(
                    driver,
                    evidence,
                )
            )

            evidence_score = (
                self.evidence_score(
                    supporting_evidence
                )
            )

            score = round_score(
                min(
                    Decimal("100"),
                    (
                        driver.score
                        * Decimal("0.65")
                    )
                    + (
                        alignment
                        * Decimal("20")
                    )
                    + (
                        evidence_score
                        * Decimal("0.15")
                    ),
                )
            )

            severity = (
                self.determine_severity(
                    driver.percentage_change
                )
            )

            confidence = (
                self.confidence_from_score(
                    score
                )
            )

            title = self.hypothesis_title(
                driver
            )

            explanation = (
                self.hypothesis_explanation(
                    metric,
                    driver,
                    target_direction,
                )
            )

            evidence_text = list(
                driver.evidence
            )

            evidence_text.extend(
                item.description
                for item
                in supporting_evidence
            )

            hypotheses.append(
                RootCauseHypothesis(
                    rank=0,
                    title=title,
                    category=driver.category,
                    metric=metric,
                    direction=driver.direction,
                    impact=driver.impact,
                    impact_percent=(
                        abs(
                            driver.impact_percent
                        )
                    ),
                    score=score,
                    severity=severity,
                    confidence=confidence,
                    explanation=explanation,
                    evidence=evidence_text,
                    supporting_drivers=[
                        driver.code
                    ],
                    recommended_next_checks=(
                        self.recommended_checks(
                            driver.category
                        )
                    ),
                )
            )

        return hypotheses

    def driver_alignment(
        self,
        driver_direction: RootCauseDirection,
        target_direction: RootCauseDirection,
        metric: RootCauseMetric,
    ) -> Decimal:

        if target_direction == (
            RootCauseDirection.NEUTRAL
        ):
            return ZERO

        if driver_direction == (
            RootCauseDirection.NEUTRAL
        ):
            return ZERO

        # For expense drivers, an increase is normally
        # negative for profit-oriented targets.
        if metric in {
            RootCauseMetric.NET_PROFIT,
            RootCauseMetric.OPERATING_PROFIT,
            RootCauseMetric.EBITDA,
            RootCauseMetric.GROSS_PROFIT,
            RootCauseMetric.NET_MARGIN,
            RootCauseMetric.OPERATING_MARGIN,
            RootCauseMetric.GROSS_MARGIN,
        }:

            return (
                ONE
                if driver_direction
                == target_direction
                else ZERO
            )

        return (
            ONE
            if driver_direction
            == target_direction
            else Decimal("0.5")
        )

    def hypothesis_title(
        self,
        driver: FinancialDriver,
    ) -> str:

        return (
            f"{driver.name} is a material "
            f"financial driver"
        )

    def hypothesis_explanation(
        self,
        metric: RootCauseMetric,
        driver: FinancialDriver,
        target_direction: RootCauseDirection,
    ) -> str:

        direction_text = (
            "increased"
            if driver.absolute_change > ZERO
            else "decreased"
        )

        impact_text = (
            "supported"
            if driver.direction
            == target_direction
            else "pressured"
        )

        return (
            f"{driver.name} {direction_text} by "
            f"{abs(driver.absolute_change)} "
            f"({abs(driver.percentage_change or ZERO)}%). "
            f"This movement {impact_text} the change "
            f"in {metric.value}. This is a financial "
            f"attribution signal rather than proof of "
            f"causal certainty."
        )

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    def match_evidence(
        self,
        driver: FinancialDriver,
        evidence: Sequence[
            RootCauseEvidence
        ],
    ) -> list[RootCauseEvidence]:

        matches: list[
            RootCauseEvidence
        ] = []

        keywords = {
            driver.category.value,
            driver.code.replace(
                "_change",
                "",
            ),
            driver.name.lower(),
        }

        for item in evidence:

            searchable = (
                " ".join(
                    [
                        item.title.lower(),
                        item.description.lower(),
                        item.source_type.lower(),
                    ]
                )
            )

            if any(
                keyword.lower()
                in searchable
                for keyword in keywords
            ):
                matches.append(item)

        return matches

    def evidence_score(
        self,
        evidence: Sequence[
            RootCauseEvidence
        ],
    ) -> Decimal:

        if not evidence:
            return ZERO

        total = sum(
            (
                item.reliability
                for item in evidence
            ),
            ZERO,
        )

        return min(
            ONE,
            total
            / Decimal(
                len(evidence)
            ),
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_hypotheses(
        self,
        hypotheses: Sequence[
            RootCauseHypothesis
        ],
    ) -> list[RootCauseHypothesis]:

        ranked = sorted(
            hypotheses,
            key=lambda item: (
                item.score,
                abs(item.impact),
            ),
            reverse=True,
        )

        result: list[
            RootCauseHypothesis
        ] = []

        for index, hypothesis in enumerate(
            ranked,
            start=1,
        ):

            result.append(
                RootCauseHypothesis(
                    rank=index,
                    title=hypothesis.title,
                    category=hypothesis.category,
                    metric=hypothesis.metric,
                    direction=hypothesis.direction,
                    impact=hypothesis.impact,
                    impact_percent=hypothesis.impact_percent,
                    score=hypothesis.score,
                    severity=hypothesis.severity,
                    confidence=hypothesis.confidence,
                    explanation=hypothesis.explanation,
                    evidence=hypothesis.evidence,
                    supporting_drivers=(
                        hypothesis.supporting_drivers
                    ),
                    contradicting_evidence=(
                        hypothesis.contradicting_evidence
                    ),
                    recommended_next_checks=(
                        hypothesis.recommended_next_checks
                    ),
                    metadata=hypothesis.metadata,
                )
            )

        return result

    def determine_overall_confidence(
        self,
        hypotheses: Sequence[
            RootCauseHypothesis
        ],
    ) -> RootCauseConfidence:

        if not hypotheses:
            return RootCauseConfidence.VERY_LOW

        score = hypotheses[0].score

        if score >= Decimal("85"):
            return RootCauseConfidence.VERY_HIGH

        if score >= self.high_confidence_threshold:
            return RootCauseConfidence.HIGH

        if score >= self.medium_confidence_threshold:
            return RootCauseConfidence.MEDIUM

        if score >= Decimal("25"):
            return RootCauseConfidence.LOW

        return RootCauseConfidence.VERY_LOW

    # ------------------------------------------------------------------
    # Compound Root Causes
    # ------------------------------------------------------------------

    def detect_compound_causes(
        self,
        hypotheses: Sequence[
            RootCauseHypothesis
        ],
        *,
        limit: int = 3,
    ) -> list[
        RootCauseHypothesis
    ]:

        return list(
            hypotheses[: max(0, limit)]
        )

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------

    def generate_explanation(
        self,
        *,
        metric: RootCauseMetric,
        current: Decimal,
        previous: Decimal,
        change: Decimal,
        change_percent: Decimal | None,
        direction: RootCauseDirection,
        primary: RootCauseHypothesis | None,
        hypotheses: Sequence[
            RootCauseHypothesis
        ],
    ) -> str:

        if direction == (
            RootCauseDirection.NEUTRAL
        ):
            return (
                f"{metric.value} remained unchanged "
                "between the two periods."
            )

        direction_text = (
            "increased"
            if direction
            == RootCauseDirection.POSITIVE
            else "decreased"
        )

        percentage_text = (
            f"{abs(change_percent)}%"
            if change_percent is not None
            else "an undefined percentage"
        )

        text = (
            f"{metric.value} {direction_text} from "
            f"{round_money(previous)} to "
            f"{round_money(current)}, a change of "
            f"{round_money(change)} "
            f"({percentage_text}). "
        )

        if primary:

            text += (
                f"The leading financial driver is "
                f"{primary.title.lower()}, with a "
                f"root-cause score of {primary.score}/100. "
            )

        if len(hypotheses) > 1:

            text += (
                f"{len(hypotheses)} measurable driver "
                "hypotheses were identified. "
            )

        text += (
            "These findings represent financial "
            "attribution signals and should be validated "
            "with transaction-level data, operational "
            "context, and relevant source documents."
        )

        return text

    def generate_summary(
        self,
        *,
        metric: RootCauseMetric,
        change_percent: Decimal | None,
        primary: RootCauseHypothesis | None,
        confidence: RootCauseConfidence,
    ) -> str:

        if primary is None:
            return (
                f"No material root cause was identified "
                f"for {metric.value}."
            )

        magnitude = (
            abs(change_percent)
            if change_percent is not None
            else None
        )

        if magnitude is None:
            change_text = (
                "with an undefined percentage change"
            )
        else:
            change_text = (
                f"with a {magnitude}% change"
            )

        return (
            f"The leading explanation for the change "
            f"in {metric.value} is "
            f"{primary.title.lower()} "
            f"{change_text}. "
            f"Overall confidence is "
            f"{confidence.value.replace('_', ' ')}."
        )

    # ------------------------------------------------------------------
    # Next Checks
    # ------------------------------------------------------------------

    def recommended_checks(
        self,
        category: RootCauseCategory,
    ) -> list[str]:

        checks = {
            RootCauseCategory.REVENUE: [
                "Review revenue by customer.",
                "Review revenue by product or service.",
                "Review pricing and discount changes.",
                "Check customer churn and renewals.",
            ],
            RootCauseCategory.COGS: [
                "Review supplier pricing.",
                "Review unit cost changes.",
                "Review product/service mix.",
                "Check purchasing and inventory data.",
            ],
            RootCauseCategory.OPERATING_EXPENSE: [
                "Review expenses by category.",
                "Identify newly increased expense lines.",
                "Check fixed versus variable costs.",
                "Review departmental spending.",
            ],
            RootCauseCategory.PRICING: [
                "Review price changes.",
                "Review discount rates.",
                "Compare realized versus list prices.",
                "Analyze price-volume interaction.",
            ],
            RootCauseCategory.VOLUME: [
                "Review transaction counts.",
                "Review customer/order volume.",
                "Check conversion rates.",
                "Review demand trends.",
            ],
            RootCauseCategory.MIX: [
                "Review product mix.",
                "Review customer mix.",
                "Review geographic/channel mix.",
                "Compare segment margins.",
            ],
            RootCauseCategory.CUSTOMER: [
                "Review top customers.",
                "Check customer churn.",
                "Review customer concentration.",
                "Analyze customer-level revenue changes.",
            ],
            RootCauseCategory.PRODUCT: [
                "Review product revenue.",
                "Review product margins.",
                "Check product demand changes.",
                "Identify underperforming products.",
            ],
            RootCauseCategory.WORKING_CAPITAL: [
                "Review receivables aging.",
                "Review inventory turnover.",
                "Review supplier payment timing.",
                "Check working-capital policies.",
            ],
            RootCauseCategory.CASH_FLOW: [
                "Reconcile operating cash flow.",
                "Review working-capital movements.",
                "Review capital expenditures.",
                "Check financing flows.",
            ],
            RootCauseCategory.DEBT: [
                "Review debt maturity schedule.",
                "Review new borrowings.",
                "Review repayments.",
                "Check leverage ratios.",
            ],
            RootCauseCategory.INTEREST: [
                "Review interest-bearing debt.",
                "Check interest rates.",
                "Review refinancing activity.",
                "Check debt maturity changes.",
            ],
            RootCauseCategory.TAX: [
                "Review effective tax rate.",
                "Check tax provision changes.",
                "Review one-time tax items.",
                "Validate tax reconciliation.",
            ],
            RootCauseCategory.MARGIN: [
                "Review gross margin.",
                "Review operating margin.",
                "Analyze COGS and OPEX ratios.",
                "Review pricing and mix.",
            ],
            RootCauseCategory.OTHER: [
                "Review detailed transaction data.",
                "Compare against historical periods.",
                "Validate source documents.",
            ],
        }

        return checks.get(
            category,
            checks[RootCauseCategory.OTHER],
        )

    def generate_next_checks(
        self,
        metric: RootCauseMetric,
        primary: RootCauseHypothesis | None,
        hypotheses: Sequence[
            RootCauseHypothesis
        ],
    ) -> list[str]:

        if primary is None:
            return [
                "Retrieve transaction-level evidence.",
                "Compare additional historical periods.",
                "Validate source financial statements.",
            ]

        checks = list(
            primary.recommended_next_checks
        )

        if len(hypotheses) > 1:
            checks.append(
                "Compare the leading hypothesis "
                "against the second-ranked driver."
            )

        checks.append(
            "Retrieve supporting documents through "
            "the RAG evidence layer."
        )

        checks.append(
            "Run a what-if scenario to estimate "
            "the impact of the suspected driver."
        )

        return list(
            dict.fromkeys(checks)
        )

    # ------------------------------------------------------------------
    # Public Specialized Methods
    # ------------------------------------------------------------------

    def analyze_profit_decline(
        self,
        current: FinancialSnapshot
        | Mapping[str, Any],
        previous: FinancialSnapshot
        | Mapping[str, Any],
        *,
        evidence: Sequence[
            RootCauseEvidence
            | Mapping[str, Any]
        ] | None = None,
    ) -> RootCauseAnalysisResult:

        return self.analyze(
            current,
            previous,
            RootCauseMetric.NET_PROFIT,
            evidence=evidence,
        )

    def analyze_revenue_decline(
        self,
        current: FinancialSnapshot
        | Mapping[str, Any],
        previous: FinancialSnapshot
        | Mapping[str, Any],
        *,
        evidence: Sequence[
            RootCauseEvidence
            | Mapping[str, Any]
        ] | None = None,
    ) -> RootCauseAnalysisResult:

        return self.analyze(
            current,
            previous,
            RootCauseMetric.REVENUE,
            evidence=evidence,
        )

    def analyze_margin_decline(
        self,
        current: FinancialSnapshot
        | Mapping[str, Any],
        previous: FinancialSnapshot
        | Mapping[str, Any],
        *,
        margin: RootCauseMetric = (
            RootCauseMetric.GROSS_MARGIN
        ),
        evidence: Sequence[
            RootCauseEvidence
            | Mapping[str, Any]
        ] | None = None,
    ) -> RootCauseAnalysisResult:

        if margin not in {
            RootCauseMetric.GROSS_MARGIN,
            RootCauseMetric.OPERATING_MARGIN,
            RootCauseMetric.NET_MARGIN,
        }:
            raise ValueError(
                "margin must be a margin metric"
            )

        return self.analyze(
            current,
            previous,
            margin,
            evidence=evidence,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def analyze_root_cause(
    current: FinancialSnapshot
    | Mapping[str, Any],
    previous: FinancialSnapshot
    | Mapping[str, Any],
    target_metric: RootCauseMetric
    | str,
    *,
    evidence: Sequence[
        RootCauseEvidence
        | Mapping[str, Any]
    ] | None = None,
) -> RootCauseAnalysisResult:
    """Run financial root-cause analysis."""

    return RootCauseAnalyzer().analyze(
        current,
        previous,
        target_metric,
        evidence=evidence,
    )


def analyze_profit_decline(
    current: FinancialSnapshot
    | Mapping[str, Any],
    previous: FinancialSnapshot
    | Mapping[str, Any],
    *,
    evidence: Sequence[
        RootCauseEvidence
        | Mapping[str, Any]
    ] | None = None,
) -> RootCauseAnalysisResult:

    return RootCauseAnalyzer().analyze_profit_decline(
        current,
        previous,
        evidence=evidence,
    )


def analyze_revenue_decline(
    current: FinancialSnapshot
    | Mapping[str, Any],
    previous: FinancialSnapshot
    | Mapping[str, Any],
    *,
    evidence: Sequence[
        RootCauseEvidence
        | Mapping[str, Any]
    ] | None = None,
) -> RootCauseAnalysisResult:

    return RootCauseAnalyzer().analyze_revenue_decline(
        current,
        previous,
        evidence=evidence,
    )


def analyze_margin_decline(
    current: FinancialSnapshot
    | Mapping[str, Any],
    previous: FinancialSnapshot
    | Mapping[str, Any],
    *,
    margin: RootCauseMetric = (
        RootCauseMetric.GROSS_MARGIN
    ),
    evidence: Sequence[
        RootCauseEvidence
        | Mapping[str, Any]
    ] | None = None,
) -> RootCauseAnalysisResult:

    return RootCauseAnalyzer().analyze_margin_decline(
        current,
        previous,
        margin=margin,
        evidence=evidence,
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
    "SCORE_QUANT",

    # Utilities
    "to_decimal",
    "round_money",
    "round_percent",
    "round_score",
    "safe_divide",
    "percentage",
    "percentage_change",

    # Enums
    "RootCauseCategory",
    "RootCauseDirection",
    "RootCauseSeverity",
    "RootCauseConfidence",
    "RootCauseMetric",
    "RootCauseSignalType",

    # Models
    "FinancialSnapshot",
    "FinancialDriver",
    "RootCauseHypothesis",
    "RootCauseEvidence",
    "RootCauseAnalysisResult",

    # Analyzer
    "RootCauseAnalyzer",

    # Convenience functions
    "analyze_root_cause",
    "analyze_profit_decline",
    "analyze_revenue_decline",
    "analyze_margin_decline",
]