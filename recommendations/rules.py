"""
FinCo AI - Recommendation Rules
================================

Centralized rule definitions and rule-evaluation utilities for the
FinCo AI recommendation layer.

Architecture
------------

Financial / Risk / Cost Signals
              |
              v
        Recommendation Rules
              |
              +--> Threshold Evaluation
              +--> Severity Calculation
              +--> Confidence Calculation
              +--> Priority Calculation
              +--> Evidence Generation
              +--> Action Generation
              |
              v
    financial_recommendations.py
    risk_recommendations.py
    cost_recommendations.py
              |
              v
    recommendation_service.py

Design principles
-----------------
1. Explainable
2. Deterministic
3. Configurable
4. Financially conservative
5. Evidence-driven
6. Human-review aware
7. Independent from FastAPI/database/LLM
8. Compatible with dictionaries and Python objects
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ============================================================================
# CONSTANTS
# ============================================================================

PRIORITY_LEVELS: Dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

RISK_LEVELS: Dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "none": 0,
}

RULE_CATEGORIES = {
    "financial",
    "risk",
    "cost",
    "liquidity",
    "profitability",
    "working_capital",
    "leverage",
    "fraud",
    "forecast",
    "concentration",
    "budget",
    "cash_flow",
    "governance",
}

COMPARISON_OPERATORS = {
    "gt",
    "gte",
    "lt",
    "lte",
    "eq",
    "neq",
    "between",
    "outside",
}

DEFAULT_CURRENCY = "USD"


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def timestamp() -> str:
    """Return ISO-8601 UTC timestamp."""
    return utc_now().isoformat()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to integer."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_divide(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers."""

    num = safe_float(numerator)
    den = safe_float(denominator)

    if abs(den) < 1e-12:
        return default

    return num / den


def percentage(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """Calculate percentage."""

    return safe_divide(
        numerator,
        denominator,
        default,
    ) * 100.0


def normalize_text(value: Any) -> str:
    """Normalize text for comparison and display."""

    if value is None:
        return ""

    text = str(value).strip()

    return re.sub(
        r"\s+",
        " ",
        text,
    )


def normalize_level(
    value: Any,
    default: str = "medium",
) -> str:
    """Normalize risk/priority levels."""

    value = normalize_text(value).lower()

    aliases = {
        "urgent": "critical",
        "severe": "critical",
        "p0": "critical",
        "p1": "high",
        "p2": "medium",
        "p3": "low",
        "elevated": "high",
        "moderate": "medium",
        "minimal": "low",
    }

    value = aliases.get(
        value,
        value,
    )

    if value in PRIORITY_LEVELS:
        return value

    if value in RISK_LEVELS:
        return value

    return default


def get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a value from either a mapping or an object.
    """

    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


def first_value(
    obj: Any,
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """Return first available value."""

    for key in keys:
        value = get_value(
            obj,
            key,
            None,
        )

        if value is not None and value != "":
            return value

    return default


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare(
    value: float,
    operator: str,
    threshold: float,
    upper: Optional[float] = None,
) -> bool:
    """
    Generic numeric comparison.

    Supported operators:
        gt
        gte
        lt
        lte
        eq
        neq
        between
        outside
    """

    value = safe_float(value)
    threshold = safe_float(threshold)

    operator = normalize_text(
        operator
    ).lower()

    if operator == "gt":
        return value > threshold

    if operator == "gte":
        return value >= threshold

    if operator == "lt":
        return value < threshold

    if operator == "lte":
        return value <= threshold

    if operator == "eq":
        return math.isclose(
            value,
            threshold,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    if operator == "neq":
        return not math.isclose(
            value,
            threshold,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    if operator == "between":
        if upper is None:
            return False

        upper = safe_float(upper)

        return threshold <= value <= upper

    if operator == "outside":
        if upper is None:
            return False

        upper = safe_float(upper)

        return value < threshold or value > upper

    raise ValueError(
        f"Unsupported comparison operator: {operator}"
    )


# ============================================================================
# RULE MODEL
# ============================================================================

@dataclass(frozen=True)
class RecommendationRule:
    """
    Immutable recommendation rule definition.
    """

    rule_id: str

    category: str

    name: str

    description: str

    metric: str

    operator: str

    threshold: float

    upper_threshold: Optional[float] = None

    risk: str = "medium"

    priority: str = "medium"

    confidence: float = 0.80

    recommendation_type: str = "combined"

    action_template: str = ""

    evidence_template: str = ""

    rationale_template: str = ""

    human_review_required: bool = False

    enabled: bool = True

    tags: List[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if self.category not in RULE_CATEGORIES:
            raise ValueError(
                f"Unsupported rule category: "
                f"{self.category}"
            )

        if self.operator not in COMPARISON_OPERATORS:
            raise ValueError(
                f"Unsupported operator: "
                f"{self.operator}"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence must be between 0 and 1"
            )

        if self.priority not in PRIORITY_LEVELS:
            raise ValueError(
                f"Invalid priority: {self.priority}"
            )

        if self.risk not in RISK_LEVELS:
            raise ValueError(
                f"Invalid risk: {self.risk}"
            )

    def evaluate(
        self,
        value: Any,
    ) -> bool:
        """Evaluate the rule against a metric value."""

        return compare(
            value=safe_float(value),
            operator=self.operator,
            threshold=self.threshold,
            upper=self.upper_threshold,
        )


@dataclass
class RuleEvaluation:
    """
    Result of evaluating a recommendation rule.
    """

    rule_id: str

    category: str

    rule_name: str

    metric: str

    observed_value: float

    threshold: float

    triggered: bool

    risk: str

    priority: str

    confidence: float

    recommendation_type: str

    title: str

    description: str

    action: str

    evidence: List[str] = field(
        default_factory=list
    )

    rationale: str = ""

    human_review_required: bool = False

    evaluated_at: str = field(
        default_factory=timestamp
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evaluation."""

        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "rule_name": self.rule_name,
            "metric": self.metric,
            "observed_value": self.observed_value,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "risk": self.risk,
            "priority": self.priority,
            "confidence": self.confidence,
            "recommendation_type": (
                self.recommendation_type
            ),
            "title": self.title,
            "description": self.description,
            "action": self.action,
            "evidence": list(self.evidence),
            "rationale": self.rationale,
            "human_review_required": (
                self.human_review_required
            ),
            "evaluated_at": self.evaluated_at,
            "metadata": dict(self.metadata),
        }


# ============================================================================
# RULE ENGINE CONFIG
# ============================================================================

@dataclass
class RecommendationRuleConfig:
    """
    Central thresholds used by FinCo AI.

    These values are intentionally configurable because different
    industries and companies have different risk tolerances.
    """

    # Liquidity
    current_ratio_warning: float = 1.20
    current_ratio_critical: float = 1.00

    quick_ratio_warning: float = 1.00
    quick_ratio_critical: float = 0.80

    cash_runway_warning_months: float = 6.0
    cash_runway_critical_months: float = 3.0

    # Working capital
    dso_warning_days: float = 60.0
    dso_critical_days: float = 90.0

    dpo_warning_days: float = 20.0
    dpo_critical_days: float = 15.0

    inventory_days_warning: float = 75.0
    inventory_days_critical: float = 120.0

    # Profitability
    margin_decline_warning_percent: float = 3.0
    margin_decline_critical_percent: float = 7.0

    operating_margin_warning_percent: float = 8.0
    operating_margin_critical_percent: float = 3.0

    gross_margin_warning_percent: float = 20.0
    gross_margin_critical_percent: float = 10.0

    # Revenue
    revenue_decline_warning_percent: float = 5.0
    revenue_decline_critical_percent: float = 15.0

    revenue_volatility_warning_percent: float = 15.0
    revenue_volatility_critical_percent: float = 30.0

    # Debt
    debt_to_equity_warning: float = 2.0
    debt_to_equity_critical: float = 3.5

    debt_to_asset_warning_percent: float = 60.0
    debt_to_asset_critical_percent: float = 75.0

    interest_coverage_warning: float = 2.0
    interest_coverage_critical: float = 1.0

    # Concentration
    customer_concentration_warning_percent: float = 35.0
    customer_concentration_critical_percent: float = 50.0

    supplier_concentration_warning_percent: float = 35.0
    supplier_concentration_critical_percent: float = 50.0

    # Budget
    budget_variance_warning_percent: float = 10.0
    budget_variance_critical_percent: float = 20.0

    # Cost
    expense_growth_warning_percent: float = 10.0
    expense_growth_critical_percent: float = 20.0

    # Fraud/anomaly
    fraud_score_warning: float = 0.60
    fraud_score_critical: float = 0.85

    anomaly_score_warning: float = 0.70
    anomaly_score_critical: float = 0.90

    # Forecast
    forecast_downside_warning_percent: float = 10.0
    forecast_downside_critical_percent: float = 20.0

    forecast_probability_warning: float = 0.25
    forecast_probability_critical: float = 0.50

    # Cash flow
    operating_cash_flow_warning: float = 0.0
    free_cash_flow_warning: float = 0.0

    # Confidence
    default_confidence: float = 0.80
    high_risk_confidence: float = 0.90

    # Human review
    human_review_for_high_risk: bool = True
    human_review_for_critical: bool = True

    def validate(self) -> None:
        """Validate rule configuration."""

        numeric_fields = [
            self.current_ratio_warning,
            self.current_ratio_critical,
            self.quick_ratio_warning,
            self.quick_ratio_critical,
            self.cash_runway_warning_months,
            self.cash_runway_critical_months,
            self.dso_warning_days,
            self.dso_critical_days,
            self.debt_to_equity_warning,
            self.debt_to_equity_critical,
            self.customer_concentration_warning_percent,
            self.customer_concentration_critical_percent,
            self.supplier_concentration_warning_percent,
            self.supplier_concentration_critical_percent,
            self.default_confidence,
            self.high_risk_confidence,
        ]

        for value in numeric_fields:
            if not math.isfinite(
                safe_float(value)
            ):
                raise ValueError(
                    "Rule configuration contains "
                    "non-finite numeric value."
                )

        if not (
            0.0
            <= self.default_confidence
            <= 1.0
        ):
            raise ValueError(
                "default_confidence must be between 0 and 1"
            )

        if not (
            0.0
            <= self.high_risk_confidence
            <= 1.0
        ):
            raise ValueError(
                "high_risk_confidence must be between 0 and 1"
            )


# ============================================================================
# DEFAULT RULE CATALOG
# ============================================================================

def build_default_rules(
    config: Optional[
        RecommendationRuleConfig
    ] = None,
) -> List[RecommendationRule]:
    """
    Build FinCo AI's default rule catalog.
    """

    config = config or RecommendationRuleConfig()

    config.validate()

    rules: List[RecommendationRule] = []

    # ========================================================================
    # LIQUIDITY
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="LIQ-001",
            category="liquidity",
            name="Critical current ratio",
            description=(
                "Current assets are insufficient relative to "
                "current liabilities."
            ),
            metric="current_ratio",
            operator="lt",
            threshold=(
                config.current_ratio_critical
            ),
            risk="critical",
            priority="critical",
            confidence=config.high_risk_confidence,
            recommendation_type="liquidity",
            action_template=(
                "Protect short-term liquidity by accelerating "
                "receivables, reducing discretionary spending, "
                "and reviewing near-term obligations."
            ),
            evidence_template=(
                "Current ratio is {value:.2f}, below the "
                "critical threshold of {threshold:.2f}."
            ),
            rationale_template=(
                "A current ratio below the critical threshold "
                "indicates elevated short-term liquidity pressure."
            ),
            human_review_required=True,
            tags=[
                "liquidity",
                "working-capital",
                "critical",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="LIQ-002",
            category="liquidity",
            name="Warning current ratio",
            description=(
                "Short-term liquidity is below the preferred "
                "operating safety range."
            ),
            metric="current_ratio",
            operator="lt",
            threshold=(
                config.current_ratio_warning
            ),
            risk="high",
            priority="high",
            confidence=0.88,
            recommendation_type="liquidity",
            action_template=(
                "Review working capital and improve the "
                "conversion of current assets into cash."
            ),
            evidence_template=(
                "Current ratio is {value:.2f}, below the "
                "warning threshold of {threshold:.2f}."
            ),
            rationale_template=(
                "Current liquidity is weaker than the configured "
                "operating target."
            ),
            tags=[
                "liquidity",
                "working-capital",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="LIQ-003",
            category="liquidity",
            name="Critical quick ratio",
            description=(
                "Liquid assets excluding inventory are insufficient "
                "to cover short-term liabilities."
            ),
            metric="quick_ratio",
            operator="lt",
            threshold=(
                config.quick_ratio_critical
            ),
            risk="critical",
            priority="critical",
            confidence=config.high_risk_confidence,
            recommendation_type="liquidity",
            action_template=(
                "Increase liquid reserves, accelerate receivables "
                "collection, and review upcoming liabilities."
            ),
            evidence_template=(
                "Quick ratio is {value:.2f}, below the "
                "critical threshold of {threshold:.2f}."
            ),
            rationale_template=(
                "The company may not have sufficient immediately "
                "liquid assets to cover short-term obligations."
            ),
            human_review_required=True,
            tags=[
                "liquidity",
                "quick-ratio",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="LIQ-004",
            category="liquidity",
            name="Low cash runway",
            description=(
                "Projected cash runway is below the minimum "
                "recommended operating period."
            ),
            metric="cash_runway_months",
            operator="lt",
            threshold=(
                config.cash_runway_critical_months
            ),
            risk="critical",
            priority="critical",
            confidence=0.93,
            recommendation_type="cash_preservation",
            action_template=(
                "Activate a cash-preservation plan and review "
                "all non-essential expenditures."
            ),
            evidence_template=(
                "Cash runway is {value:.1f} months, below the "
                "critical threshold of {threshold:.1f} months."
            ),
            rationale_template=(
                "Low cash runway increases the probability of "
                "near-term funding or liquidity stress."
            ),
            human_review_required=True,
            tags=[
                "cash",
                "runway",
                "liquidity",
            ],
        )
    )

    # ========================================================================
    # WORKING CAPITAL
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="WC-001",
            category="working_capital",
            name="High receivables collection period",
            description=(
                "Receivables are taking too long to convert "
                "into cash."
            ),
            metric="dso",
            operator="gt",
            threshold=config.dso_critical_days,
            risk="high",
            priority="high",
            confidence=0.90,
            recommendation_type="working_capital",
            action_template=(
                "Prioritize overdue receivables, strengthen "
                "collection workflows, and review customer credit terms."
            ),
            evidence_template=(
                "DSO is {value:.1f} days, above the critical "
                "threshold of {threshold:.1f} days."
            ),
            rationale_template=(
                "Higher DSO delays cash conversion and can "
                "increase working-capital pressure."
            ),
            tags=[
                "receivables",
                "dso",
                "working-capital",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="WC-002",
            category="working_capital",
            name="Elevated inventory days",
            description=(
                "Inventory is being held longer than the "
                "configured operating target."
            ),
            metric="inventory_days",
            operator="gt",
            threshold=config.inventory_days_critical,
            risk="high",
            priority="high",
            confidence=0.87,
            recommendation_type="working_capital",
            action_template=(
                "Review slow-moving inventory, improve demand "
                "planning, and reduce excess stock."
            ),
            evidence_template=(
                "Inventory days are {value:.1f}, above the "
                "critical threshold of {threshold:.1f}."
            ),
            rationale_template=(
                "Excess inventory ties up working capital and "
                "increases holding and obsolescence risk."
            ),
            tags=[
                "inventory",
                "working-capital",
            ],
        )
    )

    # ========================================================================
    # PROFITABILITY
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="PROF-001",
            category="profitability",
            name="Critical operating margin",
            description=(
                "Operating margin is below the minimum "
                "recommended level."
            ),
            metric="operating_margin_percent",
            operator="lt",
            threshold=(
                config.operating_margin_critical_percent
            ),
            risk="critical",
            priority="critical",
            confidence=0.92,
            recommendation_type="margin_improvement",
            action_template=(
                "Launch a profitability recovery plan covering "
                "pricing, gross margin, operating expenses, and "
                "low-margin business segments."
            ),
            evidence_template=(
                "Operating margin is {value:.2f}%, below the "
                "critical threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "Very low operating margin reduces the company's "
                "ability to absorb revenue or cost shocks."
            ),
            human_review_required=True,
            tags=[
                "profitability",
                "margin",
                "critical",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="PROF-002",
            category="profitability",
            name="Low gross margin",
            description=(
                "Gross margin is below the configured "
                "profitability target."
            ),
            metric="gross_margin_percent",
            operator="lt",
            threshold=(
                config.gross_margin_warning_percent
            ),
            risk="high",
            priority="high",
            confidence=0.86,
            recommendation_type="margin_improvement",
            action_template=(
                "Review pricing, product mix, supplier costs, "
                "and direct operating costs."
            ),
            evidence_template=(
                "Gross margin is {value:.2f}%, below the "
                "threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "Weak gross margin limits the amount available "
                "to cover operating expenses and generate profit."
            ),
            tags=[
                "gross-margin",
                "profitability",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="PROF-003",
            category="profitability",
            name="Margin compression",
            description=(
                "Profit margin has deteriorated materially "
                "compared with the reference period."
            ),
            metric="margin_decline_percent",
            operator="gt",
            threshold=(
                config.margin_decline_critical_percent
            ),
            risk="high",
            priority="high",
            confidence=0.91,
            recommendation_type="margin_improvement",
            action_template=(
                "Investigate the drivers of margin compression "
                "and prioritize corrective actions by financial impact."
            ),
            evidence_template=(
                "Margin declined by {value:.2f}%, exceeding "
                "the threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "Persistent margin compression can materially "
                "reduce profitability even when revenue grows."
            ),
            tags=[
                "margin",
                "profitability",
                "trend",
            ],
        )
    )

    # ========================================================================
    # REVENUE
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="REV-001",
            category="financial",
            name="Critical revenue decline",
            description=(
                "Revenue has declined materially compared "
                "with the reference period."
            ),
            metric="revenue_decline_percent",
            operator="gt",
            threshold=(
                config.revenue_decline_critical_percent
            ),
            risk="critical",
            priority="critical",
            confidence=0.91,
            recommendation_type="revenue_growth",
            action_template=(
                "Perform a revenue root-cause analysis and "
                "prioritize customer retention, pricing, and "
                "pipeline recovery initiatives."
            ),
            evidence_template=(
                "Revenue decline is {value:.2f}%, exceeding "
                "the critical threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "A significant revenue decline can rapidly "
                "increase profitability and liquidity pressure."
            ),
            human_review_required=True,
            tags=[
                "revenue",
                "growth",
                "critical",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="REV-002",
            category="financial",
            name="High revenue volatility",
            description=(
                "Revenue volatility exceeds the preferred "
                "operating range."
            ),
            metric="revenue_volatility_percent",
            operator="gt",
            threshold=(
                config.revenue_volatility_critical_percent
            ),
            risk="high",
            priority="high",
            confidence=0.84,
            recommendation_type="revenue_growth",
            action_template=(
                "Investigate revenue volatility drivers and "
                "increase predictability through customer, "
                "product, and contract diversification."
            ),
            evidence_template=(
                "Revenue volatility is {value:.2f}%, above "
                "the threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "High revenue volatility makes forecasting, "
                "resource planning, and cash management more difficult."
            ),
            tags=[
                "revenue",
                "volatility",
                "forecasting",
            ],
        )
    )

    # ========================================================================
    # LEVERAGE
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="DEBT-001",
            category="leverage",
            name="Critical debt-to-equity ratio",
            description=(
                "Debt relative to shareholder equity is "
                "materially elevated."
            ),
            metric="debt_to_equity",
            operator="gt",
            threshold=(
                config.debt_to_equity_critical
            ),
            risk="critical",
            priority="critical",
            confidence=0.92,
            recommendation_type="debt_management",
            action_template=(
                "Review the debt maturity profile, refinancing "
                "options, repayment priorities, and capital structure."
            ),
            evidence_template=(
                "Debt-to-equity is {value:.2f}, above the "
                "critical threshold of {threshold:.2f}."
            ),
            rationale_template=(
                "High leverage increases financial risk and "
                "sensitivity to interest-rate or revenue shocks."
            ),
            human_review_required=True,
            tags=[
                "debt",
                "leverage",
                "capital-structure",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="DEBT-002",
            category="leverage",
            name="Weak interest coverage",
            description=(
                "Operating earnings provide limited coverage "
                "for interest obligations."
            ),
            metric="interest_coverage",
            operator="lt",
            threshold=(
                config.interest_coverage_critical
            ),
            risk="critical",
            priority="critical",
            confidence=0.94,
            recommendation_type="debt_management",
            action_template=(
                "Immediately review debt-service capacity and "
                "engage finance leadership on refinancing or "
                "debt-reduction options."
            ),
            evidence_template=(
                "Interest coverage is {value:.2f}x, below the "
                "critical threshold of {threshold:.2f}x."
            ),
            rationale_template=(
                "Interest coverage below the critical threshold "
                "indicates material debt-service pressure."
            ),
            human_review_required=True,
            tags=[
                "debt",
                "interest",
                "critical",
            ],
        )
    )

    # ========================================================================
    # CONCENTRATION
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="CONC-001",
            category="concentration",
            name="Critical customer concentration",
            description=(
                "A large share of revenue depends on a small "
                "number of customers."
            ),
            metric="customer_concentration_percent",
            operator="gt",
            threshold=(
                config.customer_concentration_critical_percent
            ),
            risk="critical",
            priority="high",
            confidence=0.89,
            recommendation_type="concentration_risk",
            action_template=(
                "Create customer diversification targets and "
                "reduce dependence on the largest accounts."
            ),
            evidence_template=(
                "Top customer concentration is {value:.2f}%, "
                "above the critical threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "High customer concentration creates material "
                "revenue dependency risk."
            ),
            human_review_required=True,
            tags=[
                "customer",
                "concentration",
                "revenue-risk",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="CONC-002",
            category="concentration",
            name="Critical supplier concentration",
            description=(
                "A significant proportion of purchases depends "
                "on a small supplier group."
            ),
            metric="supplier_concentration_percent",
            operator="gt",
            threshold=(
                config.supplier_concentration_critical_percent
            ),
            risk="high",
            priority="high",
            confidence=0.87,
            recommendation_type="supplier_optimization",
            action_template=(
                "Develop alternative suppliers and establish "
                "supplier concentration limits."
            ),
            evidence_template=(
                "Top supplier concentration is {value:.2f}%, "
                "above the critical threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "Supplier concentration can create disruption, "
                "pricing, and continuity risk."
            ),
            tags=[
                "supplier",
                "concentration",
                "procurement",
            ],
        )
    )

    # ========================================================================
    # BUDGET / EXPENSES
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="BUD-001",
            category="budget",
            name="Critical budget variance",
            description=(
                "Actual spending materially exceeds the "
                "approved budget."
            ),
            metric="budget_variance_percent",
            operator="gt",
            threshold=(
                config.budget_variance_critical_percent
            ),
            risk="high",
            priority="high",
            confidence=0.90,
            recommendation_type="budget_risk",
            action_template=(
                "Investigate the variance by category and "
                "department, then establish corrective spending controls."
            ),
            evidence_template=(
                "Budget variance is {value:.2f}%, above the "
                "critical threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "Persistent overspending can reduce margins and "
                "increase cash requirements."
            ),
            tags=[
                "budget",
                "variance",
                "expenses",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="COST-001",
            category="cost",
            name="Rapid expense growth",
            description=(
                "Operating expenses are growing faster than "
                "the configured tolerance."
            ),
            metric="expense_growth_percent",
            operator="gt",
            threshold=(
                config.expense_growth_critical_percent
            ),
            risk="high",
            priority="high",
            confidence=0.88,
            recommendation_type="expense_control",
            action_template=(
                "Review high-growth expense categories and "
                "introduce targeted cost controls."
            ),
            evidence_template=(
                "Expense growth is {value:.2f}%, above the "
                "critical threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "Rapid expense growth can cause margin compression "
                "if it outpaces revenue growth."
            ),
            tags=[
                "expenses",
                "cost",
                "growth",
            ],
        )
    )

    # ========================================================================
    # FRAUD / ANOMALY
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="FRAUD-001",
            category="fraud",
            name="Critical fraud score",
            description=(
                "A transaction or activity has a very high "
                "fraud probability."
            ),
            metric="fraud_score",
            operator="gte",
            threshold=(
                config.fraud_score_critical
            ),
            risk="critical",
            priority="critical",
            confidence=0.96,
            recommendation_type="fraud_risk",
            action_template=(
                "Place the affected transaction or workflow "
                "under immediate human review and preserve the "
                "supporting audit evidence."
            ),
            evidence_template=(
                "Fraud score is {value:.3f}, above the "
                "critical threshold of {threshold:.3f}."
            ),
            rationale_template=(
                "High fraud probability warrants human validation "
                "before financial action is finalized."
            ),
            human_review_required=True,
            tags=[
                "fraud",
                "anomaly",
                "human-review",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="FRAUD-002",
            category="fraud",
            name="Critical anomaly score",
            description=(
                "Transaction behavior is highly anomalous "
                "relative to the learned baseline."
            ),
            metric="anomaly_score",
            operator="gte",
            threshold=(
                config.anomaly_score_critical
            ),
            risk="critical",
            priority="critical",
            confidence=0.93,
            recommendation_type="fraud_risk",
            action_template=(
                "Investigate the anomalous activity, verify "
                "authorization, and preserve the transaction trail."
            ),
            evidence_template=(
                "Anomaly score is {value:.3f}, above the "
                "critical threshold of {threshold:.3f}."
            ),
            rationale_template=(
                "Highly anomalous financial activity may indicate "
                "fraud, process failure, or data-quality problems."
            ),
            human_review_required=True,
            tags=[
                "anomaly",
                "fraud",
                "investigation",
            ],
        )
    )

    # ========================================================================
    # FORECAST
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="FORECAST-001",
            category="forecast",
            name="Critical forecast downside",
            description=(
                "The forecast indicates a material downside "
                "relative to the baseline."
            ),
            metric="forecast_downside_percent",
            operator="gt",
            threshold=(
                config.forecast_downside_critical_percent
            ),
            risk="critical",
            priority="high",
            confidence=0.88,
            recommendation_type="forecast_risk",
            action_template=(
                "Run downside scenarios and prepare contingency "
                "actions for revenue, cash flow, and cost control."
            ),
            evidence_template=(
                "Forecast downside is {value:.2f}%, above the "
                "critical threshold of {threshold:.2f}%."
            ),
            rationale_template=(
                "A material forecast downside can affect "
                "liquidity, staffing, investment, and budgeting decisions."
            ),
            human_review_required=True,
            tags=[
                "forecast",
                "downside",
                "scenario",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="FORECAST-002",
            category="forecast",
            name="High downside probability",
            description=(
                "The probability of a material downside scenario "
                "is elevated."
            ),
            metric="forecast_downside_probability",
            operator="gte",
            threshold=(
                config.forecast_probability_critical
            ),
            risk="high",
            priority="high",
            confidence=0.86,
            recommendation_type="forecast_risk",
            action_template=(
                "Prepare contingency plans and monitor leading "
                "indicators associated with the downside scenario."
            ),
            evidence_template=(
                "Downside probability is {value:.1%}, above "
                "the critical probability threshold of {threshold:.1%}."
            ),
            rationale_template=(
                "Elevated downside probability warrants proactive "
                "scenario planning."
            ),
            tags=[
                "forecast",
                "probability",
                "risk",
            ],
        )
    )

    # ========================================================================
    # CASH FLOW
    # ========================================================================

    rules.append(
        RecommendationRule(
            rule_id="CASH-001",
            category="cash_flow",
            name="Negative operating cash flow",
            description=(
                "Core operations are consuming cash."
            ),
            metric="operating_cash_flow",
            operator="lt",
            threshold=(
                config.operating_cash_flow_warning
            ),
            risk="high",
            priority="high",
            confidence=0.91,
            recommendation_type="cash_flow",
            action_template=(
                "Investigate operating cash-flow drivers and "
                "prioritize receivables, inventory, margins, and "
                "operating expense improvements."
            ),
            evidence_template=(
                "Operating cash flow is {value:.2f}, below "
                "the configured zero-cash-flow threshold."
            ),
            rationale_template=(
                "Negative operating cash flow can create "
                "liquidity pressure even when reported earnings are positive."
            ),
            tags=[
                "cash-flow",
                "operations",
                "liquidity",
            ],
        )
    )

    rules.append(
        RecommendationRule(
            rule_id="CASH-002",
            category="cash_flow",
            name="Negative free cash flow",
            description=(
                "Free cash flow is negative."
            ),
            metric="free_cash_flow",
            operator="lt",
            threshold=(
                config.free_cash_flow_warning
            ),
            risk="medium",
            priority="medium",
            confidence=0.86,
            recommendation_type="cash_flow",
            action_template=(
                "Review operating cash generation and "
                "capital expenditure commitments."
            ),
            evidence_template=(
                "Free cash flow is {value:.2f}, below zero."
            ),
            rationale_template=(
                "Negative free cash flow may reduce flexibility "
                "for debt repayment, investment, and reserves."
            ),
            tags=[
                "free-cash-flow",
                "capex",
            ],
        )
    )

    return rules


# ============================================================================
# RULE ENGINE
# ============================================================================

class RecommendationRuleEngine:
    """
    Evaluates recommendation rules against financial signals.
    """

    def __init__(
        self,
        rules: Optional[
            Iterable[RecommendationRule]
        ] = None,
        config: Optional[
            RecommendationRuleConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or RecommendationRuleConfig()
        )

        self.config.validate()

        self.rules = list(
            rules
            if rules is not None
            else build_default_rules(
                self.config
            )
        )

        self._rule_index = {
            rule.rule_id: rule
            for rule in self.rules
        }

    # ---------------------------------------------------------------------
    # RULE MANAGEMENT
    # ---------------------------------------------------------------------

    def get_rule(
        self,
        rule_id: str,
    ) -> Optional[RecommendationRule]:
        """Return a rule by ID."""

        return self._rule_index.get(
            rule_id
        )

    def add_rule(
        self,
        rule: RecommendationRule,
    ) -> None:
        """Add or replace a rule."""

        self._rule_index[
            rule.rule_id
        ] = rule

        self.rules = list(
            self._rule_index.values()
        )

    def remove_rule(
        self,
        rule_id: str,
    ) -> bool:
        """Remove a rule."""

        if rule_id not in self._rule_index:
            return False

        del self._rule_index[
            rule_id
        ]

        self.rules = list(
            self._rule_index.values()
        )

        return True

    def enable_rule(
        self,
        rule_id: str,
    ) -> bool:
        """Enable a rule."""

        rule = self.get_rule(
            rule_id
        )

        if rule is None:
            return False

        replacement = RecommendationRule(
            **{
                **rule.__dict__,
                "enabled": True,
            }
        )

        self.add_rule(
            replacement
        )

        return True

    def disable_rule(
        self,
        rule_id: str,
    ) -> bool:
        """Disable a rule."""

        rule = self.get_rule(
            rule_id
        )

        if rule is None:
            return False

        replacement = RecommendationRule(
            **{
                **rule.__dict__,
                "enabled": False,
            }
        )

        self.add_rule(
            replacement
        )

        return True

    # ---------------------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------------------

    def evaluate_rule(
        self,
        rule: RecommendationRule,
        signals: Mapping[str, Any],
    ) -> RuleEvaluation:
        """Evaluate a single rule."""

        value = safe_float(
            signals.get(
                rule.metric,
                0.0,
            )
        )

        triggered = rule.evaluate(
            value
        )

        action = self._render_template(
            rule.action_template,
            value=value,
            threshold=rule.threshold,
        )

        evidence_text = self._render_template(
            rule.evidence_template,
            value=value,
            threshold=rule.threshold,
        )

        rationale = self._render_template(
            rule.rationale_template,
            value=value,
            threshold=rule.threshold,
        )

        evidence = []

        if evidence_text:
            evidence.append(
                evidence_text
            )

        human_review = (
            rule.human_review_required
            or (
                rule.risk
                in {"high", "critical"}
                and self.config.human_review_for_high_risk
            )
            or (
                rule.risk == "critical"
                and self.config.human_review_for_critical
            )
        )

        return RuleEvaluation(
            rule_id=rule.rule_id,
            category=rule.category,
            rule_name=rule.name,
            metric=rule.metric,
            observed_value=value,
            threshold=rule.threshold,
            triggered=triggered,
            risk=rule.risk,
            priority=rule.priority,
            confidence=rule.confidence,
            recommendation_type=(
                rule.recommendation_type
            ),
            title=rule.name,
            description=rule.description,
            action=action,
            evidence=evidence,
            rationale=rationale,
            human_review_required=human_review,
            metadata={
                "operator": rule.operator,
                "upper_threshold": (
                    rule.upper_threshold
                ),
                "tags": list(rule.tags),
            },
        )

    def evaluate(
        self,
        signals: Mapping[str, Any],
        categories: Optional[
            Sequence[str]
        ] = None,
        include_untriggered: bool = False,
    ) -> List[RuleEvaluation]:
        """
        Evaluate all enabled rules.

        Parameters
        ----------
        signals:
            Mapping of metric names to values.

        categories:
            Optional list of rule categories.

        include_untriggered:
            Include rules that did not trigger.
        """

        selected_categories = (
            set(categories)
            if categories
            else None
        )

        evaluations: List[
            RuleEvaluation
        ] = []

        for rule in self.rules:

            if not rule.enabled:
                continue

            if (
                selected_categories
                and rule.category
                not in selected_categories
            ):
                continue

            # Only evaluate a rule if the metric is supplied.
            if rule.metric not in signals:
                continue

            evaluation = (
                self.evaluate_rule(
                    rule,
                    signals,
                )
            )

            if (
                evaluation.triggered
                or include_untriggered
            ):
                evaluations.append(
                    evaluation
                )

        return self.sort_evaluations(
            evaluations
        )

    # ---------------------------------------------------------------------
    # SORTING
    # ---------------------------------------------------------------------

    def sort_evaluations(
        self,
        evaluations: List[
            RuleEvaluation
        ],
    ) -> List[RuleEvaluation]:
        """Sort triggered rules by risk, priority, and confidence."""

        return sorted(
            evaluations,
            key=lambda item: (
                RISK_LEVELS.get(
                    item.risk,
                    0,
                ),
                PRIORITY_LEVELS.get(
                    item.priority,
                    0,
                ),
                item.confidence,
            ),
            reverse=True,
        )

    # ---------------------------------------------------------------------
    # BATCH
    # ---------------------------------------------------------------------

    def evaluate_many(
        self,
        signal_records: Iterable[
            Mapping[str, Any]
        ],
    ) -> List[
        List[RuleEvaluation]
    ]:
        """Evaluate many signal records."""

        results = []

        for signals in signal_records:
            results.append(
                self.evaluate(
                    signals
                )
            )

        return results

    # ---------------------------------------------------------------------
    # CATEGORY HELPERS
    # ---------------------------------------------------------------------

    def evaluate_liquidity(
        self,
        signals: Mapping[str, Any],
    ) -> List[RuleEvaluation]:
        """Evaluate liquidity rules."""

        return self.evaluate(
            signals,
            categories=[
                "liquidity"
            ],
        )

    def evaluate_profitability(
        self,
        signals: Mapping[str, Any],
    ) -> List[RuleEvaluation]:
        """Evaluate profitability rules."""

        return self.evaluate(
            signals,
            categories=[
                "profitability"
            ],
        )

    def evaluate_working_capital(
        self,
        signals: Mapping[str, Any],
    ) -> List[RuleEvaluation]:
        """Evaluate working-capital rules."""

        return self.evaluate(
            signals,
            categories=[
                "working_capital"
            ],
        )

    def evaluate_leverage(
        self,
        signals: Mapping[str, Any],
    ) -> List[RuleEvaluation]:
        """Evaluate leverage rules."""

        return self.evaluate(
            signals,
            categories=[
                "leverage"
            ],
        )

    def evaluate_fraud(
        self,
        signals: Mapping[str, Any],
    ) -> List[RuleEvaluation]:
        """Evaluate fraud rules."""

        return self.evaluate(
            signals,
            categories=[
                "fraud"
            ],
        )

    def evaluate_forecast(
        self,
        signals: Mapping[str, Any],
    ) -> List[RuleEvaluation]:
        """Evaluate forecast rules."""

        return self.evaluate(
            signals,
            categories=[
                "forecast"
            ],
        )

    def evaluate_concentration(
        self,
        signals: Mapping[str, Any],
    ) -> List[RuleEvaluation]:
        """Evaluate concentration rules."""

        return self.evaluate(
            signals,
            categories=[
                "concentration"
            ],
        )

    # ---------------------------------------------------------------------
    # TEMPLATE RENDERING
    # ---------------------------------------------------------------------

    @staticmethod
    def _render_template(
        template: str,
        **kwargs: Any,
    ) -> str:
        """Safely render rule templates."""

        if not template:
            return ""

        try:
            return template.format(
                **kwargs
            )
        except (KeyError, ValueError):
            return template


# ============================================================================
# FINANCIAL SIGNAL CALCULATOR
# ============================================================================

class FinancialSignalCalculator:
    """
    Converts raw financial data into standardized recommendation signals.

    This keeps threshold rules independent from raw financial statements.
    """

    @staticmethod
    def calculate(
        data: Mapping[str, Any],
    ) -> Dict[str, float]:
        """
        Calculate standard signals.

        Supported raw fields include:
        - current_assets
        - current_liabilities
        - cash
        - receivables
        - inventory
        - revenue
        - previous_revenue
        - gross_profit
        - operating_income
        - previous_margin
        - current_margin
        - total_debt
        - total_equity
        - total_assets
        - interest_expense
        - ebit
        - customer_concentration
        - supplier_concentration
        - budget
        - actual_expenses
        - previous_expenses
        - fraud_score
        - anomaly_score
        - forecast_baseline
        - forecast_value
        - forecast_downside_probability
        - operating_cash_flow
        - free_cash_flow
        - monthly_cash_burn
        """

        signals: Dict[str, float] = {}

        current_assets = safe_float(
            data.get(
                "current_assets"
            )
        )

        current_liabilities = safe_float(
            data.get(
                "current_liabilities"
            )
        )

        cash = safe_float(
            data.get(
                "cash"
            )
        )

        receivables = safe_float(
            data.get(
                "receivables"
            )
        )

        inventory = safe_float(
            data.get(
                "inventory"
            )
        )

        revenue = safe_float(
            data.get(
                "revenue"
            )
        )

        previous_revenue = safe_float(
            data.get(
                "previous_revenue"
            )
        )

        gross_profit = safe_float(
            data.get(
                "gross_profit"
            )
        )

        operating_income = safe_float(
            data.get(
                "operating_income"
            )
        )

        previous_margin = safe_float(
            data.get(
                "previous_margin"
            )
        )

        current_margin = safe_float(
            data.get(
                "current_margin"
            )
        )

        total_debt = safe_float(
            data.get(
                "total_debt"
            )
        )

        total_equity = safe_float(
            data.get(
                "total_equity"
            )
        )

        total_assets = safe_float(
            data.get(
                "total_assets"
            )
        )

        ebit = safe_float(
            data.get(
                "ebit",
                operating_income,
            )
        )

        interest_expense = safe_float(
            data.get(
                "interest_expense"
            )
        )

        budget = safe_float(
            data.get(
                "budget"
            )
        )

        actual_expenses = safe_float(
            data.get(
                "actual_expenses"
            )
        )

        previous_expenses = safe_float(
            data.get(
                "previous_expenses"
            )
        )

        forecast_baseline = safe_float(
            data.get(
                "forecast_baseline"
            )
        )

        forecast_value = safe_float(
            data.get(
                "forecast_value"
            )
        )

        monthly_cash_burn = safe_float(
            data.get(
                "monthly_cash_burn"
            )
        )

        # --------------------------------------------------------------
        # Liquidity
        # --------------------------------------------------------------

        signals[
            "current_ratio"
        ] = safe_divide(
            current_assets,
            current_liabilities,
        )

        quick_assets = (
            cash
            + receivables
        )

        signals[
            "quick_ratio"
        ] = safe_divide(
            quick_assets,
            current_liabilities,
        )

        signals[
            "cash_runway_months"
        ] = safe_divide(
            cash,
            monthly_cash_burn,
        )

        # --------------------------------------------------------------
        # Working capital
        # --------------------------------------------------------------

        average_daily_revenue = (
            revenue / 365.0
            if revenue > 0
            else 0.0
        )

        signals[
            "dso"
        ] = safe_divide(
            receivables,
            average_daily_revenue,
        )

        cogs = safe_float(
            data.get(
                "cost_of_goods_sold"
            )
        )

        average_daily_cogs = (
            cogs / 365.0
            if cogs > 0
            else 0.0
        )

        signals[
            "inventory_days"
        ] = safe_divide(
            inventory,
            average_daily_cogs,
        )

        # --------------------------------------------------------------
        # Profitability
        # --------------------------------------------------------------

        signals[
            "gross_margin_percent"
        ] = percentage(
            gross_profit,
            revenue,
        )

        signals[
            "operating_margin_percent"
        ] = percentage(
            operating_income,
            revenue,
        )

        if previous_margin:
            signals[
                "margin_decline_percent"
            ] = max(
                0.0,
                previous_margin
                - current_margin,
            )
        else:
            signals[
                "margin_decline_percent"
            ] = safe_float(
                data.get(
                    "margin_decline_percent"
                )
            )

        # --------------------------------------------------------------
        # Revenue
        # --------------------------------------------------------------

        if previous_revenue > 0:
            revenue_change = (
                (
                    previous_revenue
                    - revenue
                )
                / previous_revenue
                * 100
            )

            signals[
                "revenue_decline_percent"
            ] = max(
                0.0,
                revenue_change,
            )
        else:
            signals[
                "revenue_decline_percent"
            ] = safe_float(
                data.get(
                    "revenue_decline_percent"
                )
            )

        signals[
            "revenue_volatility_percent"
        ] = safe_float(
            data.get(
                "revenue_volatility_percent"
            )
        )

        # --------------------------------------------------------------
        # Leverage
        # --------------------------------------------------------------

        signals[
            "debt_to_equity"
        ] = safe_divide(
            total_debt,
            total_equity,
        )

        signals[
            "debt_to_asset_percent"
        ] = percentage(
            total_debt,
            total_assets,
        )

        signals[
            "interest_coverage"
        ] = safe_divide(
            ebit,
            interest_expense,
        )

        # --------------------------------------------------------------
        # Concentration
        # --------------------------------------------------------------

        signals[
            "customer_concentration_percent"
        ] = safe_float(
            data.get(
                "customer_concentration_percent",
                data.get(
                    "customer_concentration",
                    0.0,
                ),
            )
        )

        signals[
            "supplier_concentration_percent"
        ] = safe_float(
            data.get(
                "supplier_concentration_percent",
                data.get(
                    "supplier_concentration",
                    0.0,
                ),
            )
        )

        # --------------------------------------------------------------
        # Budget
        # --------------------------------------------------------------

        signals[
            "budget_variance_percent"
        ] = percentage(
            max(
                0.0,
                actual_expenses
                - budget,
            ),
            budget,
        )

        # --------------------------------------------------------------
        # Expenses
        # --------------------------------------------------------------

        if previous_expenses > 0:
            signals[
                "expense_growth_percent"
            ] = (
                (
                    actual_expenses
                    - previous_expenses
                )
                / previous_expenses
                * 100
            )
        else:
            signals[
                "expense_growth_percent"
            ] = safe_float(
                data.get(
                    "expense_growth_percent"
                )
            )

        # --------------------------------------------------------------
        # Fraud / anomaly
        # --------------------------------------------------------------

        signals[
            "fraud_score"
        ] = safe_float(
            data.get(
                "fraud_score"
            )
        )

        signals[
            "anomaly_score"
        ] = safe_float(
            data.get(
                "anomaly_score"
            )
        )

        # --------------------------------------------------------------
        # Forecast
        # --------------------------------------------------------------

        if forecast_baseline > 0:
            signals[
                "forecast_downside_percent"
            ] = max(
                0.0,
                (
                    forecast_baseline
                    - forecast_value
                )
                / forecast_baseline
                * 100,
            )
        else:
            signals[
                "forecast_downside_percent"
            ] = safe_float(
                data.get(
                    "forecast_downside_percent"
                )
            )

        signals[
            "forecast_downside_probability"
        ] = safe_float(
            data.get(
                "forecast_downside_probability"
            )
        )

        # --------------------------------------------------------------
        # Cash flow
        # --------------------------------------------------------------

        signals[
            "operating_cash_flow"
        ] = safe_float(
            data.get(
                "operating_cash_flow"
            )
        )

        signals[
            "free_cash_flow"
        ] = safe_float(
            data.get(
                "free_cash_flow"
            )
        )

        return signals


# ============================================================================
# RULE FACTORY
# ============================================================================

class RecommendationRuleFactory:
    """
    Factory for creating custom recommendation rules.
    """

    @staticmethod
    def threshold(
        rule_id: str,
        category: str,
        name: str,
        metric: str,
        operator: str,
        threshold: float,
        *,
        risk: str = "medium",
        priority: str = "medium",
        confidence: float = 0.80,
        recommendation_type: str = "combined",
        description: str = "",
        action: str = "",
        evidence: str = "",
        rationale: str = "",
        human_review_required: bool = False,
        tags: Optional[List[str]] = None,
    ) -> RecommendationRule:
        """Create a standard threshold rule."""

        return RecommendationRule(
            rule_id=rule_id,
            category=category,
            name=name,
            description=description,
            metric=metric,
            operator=operator,
            threshold=threshold,
            risk=risk,
            priority=priority,
            confidence=confidence,
            recommendation_type=(
                recommendation_type
            ),
            action_template=action,
            evidence_template=evidence,
            rationale_template=rationale,
            human_review_required=(
                human_review_required
            ),
            tags=tags or [],
        )


# ============================================================================
# RULE SUMMARY / INSPECTION
# ============================================================================

def summarize_rules(
    rules: Iterable[
        RecommendationRule
    ],
) -> Dict[str, Any]:
    """Return a catalog summary."""

    rules = list(rules)

    category_counts: Dict[str, int] = {}
    risk_counts: Dict[str, int] = {}
    priority_counts: Dict[str, int] = {}

    enabled_count = 0

    for rule in rules:

        category_counts[
            rule.category
        ] = (
            category_counts.get(
                rule.category,
                0,
            )
            + 1
        )

        risk_counts[
            rule.risk
        ] = (
            risk_counts.get(
                rule.risk,
                0,
            )
            + 1
        )

        priority_counts[
            rule.priority
        ] = (
            priority_counts.get(
                rule.priority,
                0,
            )
            + 1
        )

        if rule.enabled:
            enabled_count += 1

    return {
        "total_rules": len(rules),
        "enabled_rules": enabled_count,
        "disabled_rules": (
            len(rules) - enabled_count
        ),
        "category_counts": category_counts,
        "risk_counts": risk_counts,
        "priority_counts": priority_counts,
    }


# ============================================================================
# DEMO
# ============================================================================

def create_demo_signals() -> Dict[str, Any]:
    """Create realistic financial signals."""

    return {
        "current_assets": 920_000,
        "current_liabilities": 1_000_000,

        "cash": 420_000,
        "receivables": 540_000,
        "inventory": 380_000,

        "revenue": 5_200_000,
        "previous_revenue": 6_100_000,

        "gross_profit": 780_000,
        "operating_income": 190_000,

        "previous_margin": 11.7,
        "current_margin": 8.4,

        "total_debt": 2_900_000,
        "total_equity": 800_000,
        "total_assets": 4_800_000,

        "ebit": 190_000,
        "interest_expense": 180_000,

        "customer_concentration_percent": 58.0,
        "supplier_concentration_percent": 52.0,

        "budget": 3_700_000,
        "actual_expenses": 4_100_000,

        "previous_expenses": 3_500_000,

        "fraud_score": 0.91,
        "anomaly_score": 0.94,

        "forecast_baseline": 5_600_000,
        "forecast_value": 4_650_000,
        "forecast_downside_probability": 0.56,

        "operating_cash_flow": -120_000,
        "free_cash_flow": -260_000,

        "monthly_cash_burn": 110_000,

        "cost_of_goods_sold": 4_420_000,

        "revenue_volatility_percent": 31.0,
    }


def create_demo_engine() -> RecommendationRuleEngine:
    """Create demo rule engine."""

    return RecommendationRuleEngine()


def run_demo() -> List[RuleEvaluation]:
    """Run default rules against demo data."""

    engine = create_demo_engine()

    raw_data = create_demo_signals()

    signals = FinancialSignalCalculator.calculate(
        raw_data
    )

    evaluations = engine.evaluate(
        signals
    )

    return evaluations


# ============================================================================
# PUBLIC HELPERS
# ============================================================================

def evaluate_recommendation_rules(
    signals: Mapping[str, Any],
    config: Optional[
        RecommendationRuleConfig
    ] = None,
) -> List[RuleEvaluation]:
    """
    Evaluate FinCo recommendation rules.

    Example
    -------
    signals = {
        "current_ratio": 0.82,
        "operating_margin_percent": 2.5,
    }

    evaluations = evaluate_recommendation_rules(
        signals
    )
    """

    engine = RecommendationRuleEngine(
        config=config
    )

    return engine.evaluate(
        signals
    )


def calculate_financial_signals(
    data: Mapping[str, Any],
) -> Dict[str, float]:
    """Public wrapper around FinancialSignalCalculator."""

    return FinancialSignalCalculator.calculate(
        data
    )


def get_default_rule_catalog(
    config: Optional[
        RecommendationRuleConfig
    ] = None,
) -> List[RecommendationRule]:
    """Return FinCo default recommendation rules."""

    return build_default_rules(
        config=config
    )


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """
    Command-line demonstration.

    Usage:

        python -m app.recommendations.rules

    """

    print()
    print("=" * 72)
    print("FinCo AI - Recommendation Rules")
    print("=" * 72)

    config = RecommendationRuleConfig()

    engine = RecommendationRuleEngine(
        config=config
    )

    summary = summarize_rules(
        engine.rules
    )

    print()
    print("RULE CATALOG")
    print("-" * 72)
    print(
        f"Total rules: "
        f"{summary['total_rules']}"
    )
    print(
        f"Enabled rules: "
        f"{summary['enabled_rules']}"
    )

    print()
    print("CATEGORY COUNTS")
    print("-" * 72)

    for category, count in sorted(
        summary["category_counts"].items()
    ):
        print(
            f"{category:<25} {count}"
        )

    raw_data = create_demo_signals()

    signals = FinancialSignalCalculator.calculate(
        raw_data
    )

    evaluations = engine.evaluate(
        signals
    )

    print()
    print("TRIGGERED RULES")
    print("-" * 72)

    if not evaluations:
        print(
            "No recommendation rules triggered."
        )

    for index, evaluation in enumerate(
        evaluations,
        start=1,
    ):
        print()
        print(
            f"{index}. "
            f"{evaluation.rule_id} - "
            f"{evaluation.rule_name}"
        )

        print(
            f"   Metric: "
            f"{evaluation.metric}"
        )

        print(
            f"   Observed: "
            f"{evaluation.observed_value:.4f}"
        )

        print(
            f"   Threshold: "
            f"{evaluation.threshold:.4f}"
        )

        print(
            f"   Risk: "
            f"{evaluation.risk.upper()}"
        )

        print(
            f"   Priority: "
            f"{evaluation.priority.upper()}"
        )

        print(
            f"   Confidence: "
            f"{evaluation.confidence:.2%}"
        )

        print(
            f"   Recommendation: "
            f"{evaluation.action}"
        )

        if evaluation.human_review_required:
            print(
                "   HUMAN REVIEW: REQUIRED"
            )

    print()
    print("=" * 72)
    print(
        "Rule evaluation complete."
    )
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()