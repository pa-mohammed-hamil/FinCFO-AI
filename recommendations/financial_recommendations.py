"""
FinCo AI - Financial Recommendations Engine
============================================

Path:
    backend/app/recommendations/financial_recommendations.py

Purpose:
    Generate explainable, prioritized financial recommendations from
    structured financial data.

Design:
    Financial Data
        ↓
    Financial Analysis
        ↓
    Rule / Threshold Detection
        ↓
    Recommendation Generation
        ↓
    Confidence + Impact + Risk
        ↓
    Human Review
        ↓
    Audit / Recommendation Service

This module intentionally does NOT execute financial actions.
It proposes recommendations for review.

Typical inputs:
    - Revenue
    - Expenses
    - Profit / EBITDA
    - Margins
    - Cash
    - Accounts receivable
    - Accounts payable
    - Budget vs actual
    - Growth rates
    - Liquidity ratios
    - Working-capital metrics

The implementation is dependency-light and can work with:
    - dicts
    - dataclasses
    - Pydantic models
    - ORM objects

Example:
    from app.recommendations.financial_recommendations import (
        FinancialRecommendationEngine,
    )

    engine = FinancialRecommendationEngine()

    result = engine.generate(
        financial_data={
            "revenue": 1_000_000,
            "previous_revenue": 1_100_000,
            "expenses": 720_000,
            "previous_expenses": 680_000,
            "gross_margin": 0.31,
            "previous_gross_margin": 0.36,
            "net_margin": 0.08,
            "cash": 150_000,
            "accounts_receivable": 260_000,
            "accounts_payable": 180_000,
            "budget_expenses": 650_000,
        }
    )

    for recommendation in result.recommendations:
        print(recommendation.title)
        print(recommendation.action)
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
import uuid

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ============================================================================
# Constants
# ============================================================================

DEFAULT_CURRENCY = "USD"

PRIORITY_LEVELS = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

RISK_LEVELS = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

RECOMMENDATION_TYPES = {
    "revenue",
    "expense",
    "profitability",
    "margin",
    "liquidity",
    "working_capital",
    "cash_flow",
    "budget",
    "growth",
    "cost_control",
    "financial_health",
}

# ============================================================================
# Utility Functions
# ============================================================================


def _utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    """Return ISO-8601 UTC timestamp."""
    return _utc_now().isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value safely to float."""
    if value is None:
        return default

    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value safely to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers."""
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)

    if abs(denominator) < 1e-12:
        return default

    return numerator / denominator


def _percentage(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Return percentage as 0-100."""
    return _safe_divide(numerator, denominator, default) * 100.0


def _round(value: Any, digits: int = 2) -> float:
    """Safely round a numeric value."""
    return round(_safe_float(value), digits)


def _normalize_text(value: Any) -> str:
    """Normalize text."""
    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def _normalize_priority(value: Any) -> str:
    """Normalize priority level."""
    value = _normalize_text(value).lower()

    if value in PRIORITY_LEVELS:
        return value

    return "medium"


def _normalize_risk(value: Any) -> str:
    """Normalize risk level."""
    value = _normalize_text(value).lower()

    if value in RISK_LEVELS:
        return value

    return "medium"


def _json_safe(value: Any) -> Any:
    """Convert arbitrary values to JSON-safe representations."""

    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value

        return None

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())

    if hasattr(value, "dict"):
        return _json_safe(value.dict())

    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))

    return str(value)


def _get(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Retrieve a field from dict-like or object-like data.
    """

    if source is None:
        return default

    if isinstance(source, Mapping):
        return source.get(key, default)

    return getattr(source, key, default)


def _first(
    source: Any,
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """Return first available value from a set of keys."""

    for key in keys:
        value = _get(source, key, None)

        if value is not None:
            return value

    return default


def _mean(values: Iterable[float]) -> float:
    """Safe arithmetic mean."""
    values = [
        _safe_float(v)
        for v in values
        if v is not None
    ]

    if not values:
        return 0.0

    return statistics.mean(values)


def _format_currency(
    amount: float,
    currency: str = DEFAULT_CURRENCY,
) -> str:
    """Format monetary value."""

    amount = _safe_float(amount)

    return f"{currency} {amount:,.2f}"


def _format_percent(value: float) -> str:
    """Format percentage."""

    return f"{_safe_float(value):,.2f}%"


def _hash_id(*parts: Any) -> str:
    """Generate deterministic identifier."""

    raw = "|".join(str(part) for part in parts)

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return f"FINREC-{digest.upper()}"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FinancialRecommendationConfig:
    """Configuration for the financial recommendation engine."""

    currency: str = DEFAULT_CURRENCY

    # Revenue
    revenue_decline_warning_pct: float = 5.0
    revenue_decline_high_pct: float = 10.0
    revenue_decline_critical_pct: float = 20.0

    # Expenses
    expense_growth_warning_pct: float = 5.0
    expense_growth_high_pct: float = 10.0
    expense_growth_critical_pct: float = 20.0

    # Margins
    gross_margin_warning_pct: float = 25.0
    gross_margin_critical_pct: float = 15.0

    margin_decline_warning_pct: float = 2.0
    margin_decline_high_pct: float = 5.0

    # Profitability
    net_margin_warning_pct: float = 5.0
    net_margin_critical_pct: float = 0.0

    # Budget
    budget_variance_warning_pct: float = 5.0
    budget_variance_high_pct: float = 10.0
    budget_variance_critical_pct: float = 20.0

    # Liquidity
    current_ratio_warning: float = 1.2
    current_ratio_critical: float = 1.0
    quick_ratio_warning: float = 1.0
    quick_ratio_critical: float = 0.8

    # Cash
    cash_runway_warning_months: float = 6.0
    cash_runway_critical_months: float = 3.0

    # Working capital
    dso_warning_days: float = 60.0
    dso_critical_days: float = 90.0

    dpo_warning_days: float = 15.0
    dpo_critical_days: float = 7.0

    # Debt
    debt_to_equity_warning: float = 2.0
    debt_to_equity_critical: float = 3.0

    # Concentration
    concentration_warning_pct: float = 30.0
    concentration_critical_pct: float = 50.0

    # Confidence
    minimum_confidence: float = 0.55

    # Result controls
    max_recommendations: int = 20

    # Human review
    require_human_review_for_high_risk: bool = True
    require_human_review_for_high_impact: bool = True
    high_impact_amount: float = 50_000.0

    # Output
    output_directory: str = "reports/recommendations"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class FinancialRecommendation:
    """Single financial recommendation."""

    recommendation_id: str
    recommendation_type: str

    title: str
    description: str
    action: str

    priority: str
    risk: str

    confidence: float
    impact_score: float

    current_value: Optional[float] = None
    benchmark_value: Optional[float] = None
    variance: Optional[float] = None
    variance_pct: Optional[float] = None

    estimated_impact: float = 0.0
    potential_savings: float = 0.0
    potential_revenue_uplift: float = 0.0

    currency: str = DEFAULT_CURRENCY

    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

    affected_metric: Optional[str] = None
    affected_department: Optional[str] = None
    affected_category: Optional[str] = None

    requires_human_review: bool = False

    generated_at: str = field(
        default_factory=_timestamp
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe dictionary."""

        return _json_safe(asdict(self))


@dataclass
class FinancialRecommendationResult:
    """Collection of financial recommendations."""

    result_id: str
    generated_at: str

    currency: str

    status: str

    recommendations: List[FinancialRecommendation]

    recommendations_count: int

    high_priority_count: int
    critical_count: int

    human_review_count: int

    total_estimated_impact: float
    total_potential_savings: float
    total_potential_revenue_uplift: float

    average_confidence: float

    summary: str

    metrics_analyzed: List[str]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-safe dictionary."""

        return _json_safe(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to JSON."""

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
        )


# ============================================================================
# Financial Recommendation Engine
# ============================================================================


class FinancialRecommendationEngine:
    """
    Rule-based financial recommendation engine.

    The engine is intentionally explainable.

    It does not make autonomous financial decisions.

    It detects financial conditions and proposes actions
    that can subsequently be reviewed by humans or higher-level
    recommendation agents.
    """

    def __init__(
        self,
        config: Optional[FinancialRecommendationConfig] = None,
    ) -> None:

        self.config = (
            config
            or FinancialRecommendationConfig()
        )

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def generate(
        self,
        financial_data: Optional[Any] = None,
        *,
        revenue_data: Optional[Any] = None,
        expense_data: Optional[Any] = None,
        budget_data: Optional[Any] = None,
        liquidity_data: Optional[Any] = None,
        working_capital_data: Optional[Any] = None,
        profitability_data: Optional[Any] = None,
    ) -> FinancialRecommendationResult:
        """
        Generate financial recommendations.

        Parameters may be dictionaries, dataclasses, Pydantic models,
        ORM objects, or sequences of records.
        """

        financial_data = financial_data or {}

        recommendations: List[
            FinancialRecommendation
        ] = []

        # --------------------------------------------------------------
        # Merge specialized data into the base financial context.
        # --------------------------------------------------------------

        context = self._build_context(
            financial_data=financial_data,
            revenue_data=revenue_data,
            expense_data=expense_data,
            budget_data=budget_data,
            liquidity_data=liquidity_data,
            working_capital_data=working_capital_data,
            profitability_data=profitability_data,
        )

        # --------------------------------------------------------------
        # Generate recommendation families.
        # --------------------------------------------------------------

        recommendations.extend(
            self._revenue_recommendations(context)
        )

        recommendations.extend(
            self._expense_recommendations(context)
        )

        recommendations.extend(
            self._profitability_recommendations(context)
        )

        recommendations.extend(
            self._margin_recommendations(context)
        )

        recommendations.extend(
            self._budget_recommendations(context)
        )

        recommendations.extend(
            self._liquidity_recommendations(context)
        )

        recommendations.extend(
            self._working_capital_recommendations(context)
        )

        recommendations.extend(
            self._cash_flow_recommendations(context)
        )

        recommendations.extend(
            self._growth_recommendations(context)
        )

        recommendations.extend(
            self._financial_health_recommendations(context)
        )

        # --------------------------------------------------------------
        # Post-processing.
        # --------------------------------------------------------------

        recommendations = self._filter_by_confidence(
            recommendations
        )

        recommendations = self._deduplicate(
            recommendations
        )

        recommendations = self._sort_recommendations(
            recommendations
        )

        recommendations = recommendations[
            : self.config.max_recommendations
        ]

        return self._build_result(
            recommendations=recommendations,
            context=context,
        )

    # ==========================================================================
    # Context
    # ==========================================================================

    def _build_context(
        self,
        financial_data: Any,
        revenue_data: Any,
        expense_data: Any,
        budget_data: Any,
        liquidity_data: Any,
        working_capital_data: Any,
        profitability_data: Any,
    ) -> Dict[str, Any]:

        context = {}

        if isinstance(financial_data, Mapping):
            context.update(financial_data)
        else:
            context.update(
                _json_safe(financial_data)
                if financial_data
                else {}
            )

        context["_revenue_data"] = revenue_data
        context["_expense_data"] = expense_data
        context["_budget_data"] = budget_data
        context["_liquidity_data"] = liquidity_data
        context["_working_capital_data"] = (
            working_capital_data
        )
        context["_profitability_data"] = (
            profitability_data
        )

        return context

    # ==========================================================================
    # Revenue Recommendations
    # ==========================================================================

    def _revenue_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        revenue = _safe_float(
            _first(
                data,
                ["revenue", "total_revenue", "sales"],
            )
        )

        previous_revenue = _safe_float(
            _first(
                data,
                [
                    "previous_revenue",
                    "prior_revenue",
                    "last_period_revenue",
                ],
            )
        )

        if revenue <= 0 or previous_revenue <= 0:
            return recommendations

        change_pct = _percentage(
            revenue - previous_revenue,
            previous_revenue,
        )

        decline_pct = abs(change_pct)

        if change_pct < -self.config.revenue_decline_critical_pct:

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="revenue",
                    title="Investigate critical revenue decline",
                    description=(
                        f"Revenue declined by {decline_pct:.2f}% "
                        f"compared with the previous period."
                    ),
                    action=(
                        "Perform a revenue root-cause analysis across "
                        "customer segments, products, pricing, geography, "
                        "sales pipeline and churn. Prioritize recovery "
                        "actions for the largest revenue drivers."
                    ),
                    priority="critical",
                    risk="critical",
                    confidence=0.94,
                    current_value=revenue,
                    benchmark_value=previous_revenue,
                    variance=revenue - previous_revenue,
                    variance_pct=change_pct,
                    potential_revenue_uplift=(
                        previous_revenue - revenue
                    ),
                    affected_metric="revenue",
                    evidence=[
                        f"Current revenue: "
                        f"{_format_currency(revenue, self.config.currency)}",
                        f"Previous revenue: "
                        f"{_format_currency(previous_revenue, self.config.currency)}",
                        f"Revenue change: "
                        f"{_format_percent(change_pct)}",
                    ],
                    assumptions=[
                        "Previous-period revenue is comparable.",
                        "Revenue values use the same accounting basis.",
                    ],
                    risks=[
                        "Aggressive recovery actions may increase acquisition costs.",
                        "Discounting may improve volume while reducing margin.",
                    ],
                    next_steps=[
                        "Segment revenue by product and customer.",
                        "Analyze churn and lost deals.",
                        "Review pricing changes.",
                        "Identify top three revenue-loss drivers.",
                        "Create a recovery scenario.",
                    ],
                    potential_impact=(
                        previous_revenue - revenue
                    ),
                )
            )

        elif change_pct < -self.config.revenue_decline_high_pct:

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="revenue",
                    title="Address significant revenue decline",
                    description=(
                        f"Revenue is down {decline_pct:.2f}% "
                        "from the previous period."
                    ),
                    action=(
                        "Review customer churn, sales conversion, pricing "
                        "and product-level performance, then prioritize "
                        "the highest-impact recovery opportunities."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.90,
                    current_value=revenue,
                    benchmark_value=previous_revenue,
                    variance=revenue - previous_revenue,
                    variance_pct=change_pct,
                    potential_revenue_uplift=(
                        previous_revenue - revenue
                    ),
                    affected_metric="revenue",
                    evidence=[
                        f"Revenue change: "
                        f"{_format_percent(change_pct)}",
                    ],
                    next_steps=[
                        "Analyze customer churn.",
                        "Review sales funnel conversion.",
                        "Review pricing and product mix.",
                    ],
                    potential_impact=(
                        previous_revenue - revenue
                    ),
                )
            )

        elif change_pct < -self.config.revenue_decline_warning_pct:

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="revenue",
                    title="Monitor revenue softness",
                    description=(
                        f"Revenue decreased by {decline_pct:.2f}%."
                    ),
                    action=(
                        "Monitor the trend and investigate early indicators "
                        "such as customer churn, pipeline conversion and "
                        "pricing pressure."
                    ),
                    priority="medium",
                    risk="medium",
                    confidence=0.84,
                    current_value=revenue,
                    benchmark_value=previous_revenue,
                    variance=revenue - previous_revenue,
                    variance_pct=change_pct,
                    potential_revenue_uplift=(
                        previous_revenue - revenue
                    ),
                    affected_metric="revenue",
                    evidence=[
                        f"Revenue change: "
                        f"{_format_percent(change_pct)}",
                    ],
                    next_steps=[
                        "Track weekly revenue trend.",
                        "Review customer retention.",
                        "Monitor pipeline conversion.",
                    ],
                    potential_impact=(
                        previous_revenue - revenue
                    ),
                )
            )

        return recommendations

    # ==========================================================================
    # Expense Recommendations
    # ==========================================================================

    def _expense_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        expenses = _safe_float(
            _first(
                data,
                ["expenses", "total_expenses", "operating_expenses"],
            )
        )

        previous_expenses = _safe_float(
            _first(
                data,
                [
                    "previous_expenses",
                    "prior_expenses",
                    "last_period_expenses",
                ],
            )
        )

        if expenses <= 0 or previous_expenses <= 0:
            return recommendations

        growth_pct = _percentage(
            expenses - previous_expenses,
            previous_expenses,
        )

        if growth_pct > self.config.expense_growth_critical_pct:

            excess = expenses - previous_expenses

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="expense",
                    title="Reduce critical expense growth",
                    description=(
                        f"Expenses increased by {growth_pct:.2f}% "
                        "compared with the previous period."
                    ),
                    action=(
                        "Launch an expense-control review. Prioritize "
                        "non-essential spending, supplier costs, recurring "
                        "subscriptions and operational overhead."
                    ),
                    priority="critical",
                    risk="high",
                    confidence=0.94,
                    current_value=expenses,
                    benchmark_value=previous_expenses,
                    variance=excess,
                    variance_pct=growth_pct,
                    potential_savings=max(excess * 0.25, 0),
                    affected_metric="expenses",
                    evidence=[
                        f"Current expenses: "
                        f"{_format_currency(expenses, self.config.currency)}",
                        f"Previous expenses: "
                        f"{_format_currency(previous_expenses, self.config.currency)}",
                        f"Expense growth: "
                        f"{_format_percent(growth_pct)}",
                    ],
                    risks=[
                        "Excessive cost cutting may reduce service quality.",
                        "Operationally critical spending should be protected.",
                    ],
                    next_steps=[
                        "Rank expense categories by growth.",
                        "Identify discretionary expenses.",
                        "Review supplier contracts.",
                        "Set category-level spending controls.",
                    ],
                    potential_impact=excess,
                )
            )

        elif growth_pct > self.config.expense_growth_high_pct:

            excess = expenses - previous_expenses

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="expense",
                    title="Control elevated expense growth",
                    description=(
                        f"Expenses increased by {growth_pct:.2f}%."
                    ),
                    action=(
                        "Review the fastest-growing expense categories "
                        "and introduce targeted cost controls."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.90,
                    current_value=expenses,
                    benchmark_value=previous_expenses,
                    variance=excess,
                    variance_pct=growth_pct,
                    potential_savings=max(excess * 0.15, 0),
                    affected_metric="expenses",
                    evidence=[
                        f"Expense growth: "
                        f"{_format_percent(growth_pct)}",
                    ],
                    next_steps=[
                        "Identify high-growth categories.",
                        "Review recurring costs.",
                        "Create category-level budgets.",
                    ],
                    potential_impact=excess,
                )
            )

        elif growth_pct > self.config.expense_growth_warning_pct:

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="expense",
                    title="Monitor rising expenses",
                    description=(
                        f"Expenses increased by {growth_pct:.2f}%."
                    ),
                    action=(
                        "Monitor expense growth and investigate categories "
                        "that are growing faster than revenue."
                    ),
                    priority="medium",
                    risk="medium",
                    confidence=0.82,
                    current_value=expenses,
                    benchmark_value=previous_expenses,
                    variance=expenses - previous_expenses,
                    variance_pct=growth_pct,
                    potential_savings=max(
                        expenses - previous_expenses,
                        0,
                    ) * 0.05,
                    affected_metric="expenses",
                    evidence=[
                        f"Expense growth: "
                        f"{_format_percent(growth_pct)}",
                    ],
                    next_steps=[
                        "Compare expense growth with revenue growth.",
                        "Monitor discretionary spending.",
                    ],
                    potential_impact=(
                        expenses - previous_expenses
                    ),
                )
            )

        return recommendations

    # ==========================================================================
    # Profitability Recommendations
    # ==========================================================================

    def _profitability_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        net_profit = _safe_float(
            _first(
                data,
                ["net_profit", "profit", "net_income"],
            )
        )

        revenue = _safe_float(
            _first(
                data,
                ["revenue", "total_revenue", "sales"],
            )
        )

        if revenue <= 0:
            return recommendations

        net_margin = _percentage(
            net_profit,
            revenue,
        )

        configured_margin = _first(
            data,
            ["net_margin", "net_profit_margin"],
        )

        if configured_margin is not None:
            configured_margin = _safe_float(
                configured_margin
            )

            if abs(configured_margin) <= 1:
                net_margin = configured_margin * 100
            else:
                net_margin = configured_margin

        if net_margin < self.config.net_margin_critical_pct:

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="profitability",
                    title="Restore positive profitability",
                    description=(
                        f"Net margin is {net_margin:.2f}%, indicating "
                        "that the business is currently operating at a loss."
                    ),
                    action=(
                        "Prioritize a profitability recovery plan covering "
                        "pricing, gross margin, operating expenses, product "
                        "mix and loss-making segments."
                    ),
                    priority="critical",
                    risk="critical",
                    confidence=0.96,
                    current_value=net_margin,
                    benchmark_value=0.0,
                    variance=net_margin,
                    variance_pct=net_margin,
                    affected_metric="net_margin",
                    evidence=[
                        f"Net profit: "
                        f"{_format_currency(net_profit, self.config.currency)}",
                        f"Revenue: "
                        f"{_format_currency(revenue, self.config.currency)}",
                        f"Net margin: "
                        f"{_format_percent(net_margin)}",
                    ],
                    risks=[
                        "Rapid cost reduction may damage growth.",
                        "Pricing changes may affect customer retention.",
                    ],
                    next_steps=[
                        "Identify loss-making products.",
                        "Analyze gross margin by segment.",
                        "Review operating expense ratios.",
                        "Evaluate pricing opportunities.",
                        "Build a profitability recovery scenario.",
                    ],
                    potential_impact=abs(net_profit),
                )
            )

        elif net_margin < self.config.net_margin_warning_pct:

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="profitability",
                    title="Improve low net profitability",
                    description=(
                        f"Net margin is only {net_margin:.2f}%."
                    ),
                    action=(
                        "Improve profitability through targeted cost control, "
                        "better pricing, product mix optimization and "
                        "reduction of low-margin activities."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.91,
                    current_value=net_margin,
                    benchmark_value=(
                        self.config.net_margin_warning_pct
                    ),
                    variance=(
                        net_margin
                        - self.config.net_margin_warning_pct
                    ),
                    variance_pct=(
                        net_margin
                        - self.config.net_margin_warning_pct
                    ),
                    affected_metric="net_margin",
                    evidence=[
                        f"Net margin: "
                        f"{_format_percent(net_margin)}",
                        f"Warning threshold: "
                        f"{_format_percent(self.config.net_margin_warning_pct)}",
                    ],
                    next_steps=[
                        "Review pricing.",
                        "Analyze product margins.",
                        "Identify avoidable operating expenses.",
                    ],
                    potential_impact=(
                        revenue
                        * max(
                            self.config.net_margin_warning_pct
                            - net_margin,
                            0,
                        )
                        / 100
                    ),
                )
            )

        return recommendations

    # ==========================================================================
    # Margin Recommendations
    # ==========================================================================

    def _margin_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        gross_margin = _safe_float(
            _first(
                data,
                ["gross_margin", "gross_profit_margin"],
            )
        )

        if abs(gross_margin) <= 1:
            gross_margin *= 100

        previous_margin = _safe_float(
            _first(
                data,
                [
                    "previous_gross_margin",
                    "prior_gross_margin",
                ],
            )
        )

        if previous_margin:
            if abs(previous_margin) <= 1:
                previous_margin *= 100

        if gross_margin <= 0:
            return recommendations

        if gross_margin < self.config.gross_margin_critical_pct:

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="margin",
                    title="Address critical gross-margin pressure",
                    description=(
                        f"Gross margin is {gross_margin:.2f}%, below the "
                        f"critical threshold of "
                        f"{self.config.gross_margin_critical_pct:.2f}%."
                    ),
                    action=(
                        "Investigate pricing, cost of goods sold, supplier "
                        "pricing, product mix and operational inefficiencies."
                    ),
                    priority="critical",
                    risk="critical",
                    confidence=0.95,
                    current_value=gross_margin,
                    benchmark_value=(
                        self.config.gross_margin_critical_pct
                    ),
                    variance=(
                        gross_margin
                        - self.config.gross_margin_critical_pct
                    ),
                    variance_pct=(
                        gross_margin
                        - self.config.gross_margin_critical_pct
                    ),
                    affected_metric="gross_margin",
                    evidence=[
                        f"Gross margin: "
                        f"{_format_percent(gross_margin)}",
                    ],
                    next_steps=[
                        "Analyze COGS by product.",
                        "Review supplier costs.",
                        "Evaluate pricing.",
                        "Review product mix.",
                    ],
                    potential_impact=0.0,
                )
            )

        elif gross_margin < self.config.gross_margin_warning_pct:

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="margin",
                    title="Improve gross margin",
                    description=(
                        f"Gross margin is {gross_margin:.2f}%."
                    ),
                    action=(
                        "Focus on supplier negotiations, pricing discipline, "
                        "product mix and reduction of direct costs."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.90,
                    current_value=gross_margin,
                    benchmark_value=(
                        self.config.gross_margin_warning_pct
                    ),
                    variance=(
                        gross_margin
                        - self.config.gross_margin_warning_pct
                    ),
                    variance_pct=(
                        gross_margin
                        - self.config.gross_margin_warning_pct
                    ),
                    affected_metric="gross_margin",
                    evidence=[
                        f"Gross margin: "
                        f"{_format_percent(gross_margin)}",
                        f"Target minimum: "
                        f"{_format_percent(self.config.gross_margin_warning_pct)}",
                    ],
                    next_steps=[
                        "Review direct-cost drivers.",
                        "Identify low-margin products.",
                        "Review supplier contracts.",
                        "Evaluate pricing changes.",
                    ],
                    potential_impact=0.0,
                )
            )

        if previous_margin:

            margin_change = (
                gross_margin - previous_margin
            )

            if (
                margin_change
                < -self.config.margin_decline_high_pct
            ):

                recommendations.append(
                    self._create_recommendation(
                        recommendation_type="margin",
                        title="Investigate sharp margin deterioration",
                        description=(
                            f"Gross margin declined by "
                            f"{abs(margin_change):.2f} percentage points."
                        ),
                        action=(
                            "Perform a margin bridge analysis to identify "
                            "whether pricing, COGS, discounts or product mix "
                            "caused the deterioration."
                        ),
                        priority="high",
                        risk="high",
                        confidence=0.93,
                        current_value=gross_margin,
                        benchmark_value=previous_margin,
                        variance=margin_change,
                        variance_pct=margin_change,
                        affected_metric="gross_margin",
                        evidence=[
                            f"Current margin: "
                            f"{_format_percent(gross_margin)}",
                            f"Previous margin: "
                            f"{_format_percent(previous_margin)}",
                            f"Change: "
                            f"{margin_change:.2f} percentage points",
                        ],
                        next_steps=[
                            "Build a margin bridge.",
                            "Analyze pricing changes.",
                            "Analyze COGS changes.",
                            "Review discounting.",
                        ],
                        potential_impact=0.0,
                    )
                )

        return recommendations

    # ==========================================================================
    # Budget Recommendations
    # ==========================================================================

    def _budget_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        actual_expenses = _safe_float(
            _first(
                data,
                [
                    "actual_expenses",
                    "expenses",
                    "total_expenses",
                ],
            )
        )

        budget_expenses = _safe_float(
            _first(
                data,
                [
                    "budget_expenses",
                    "expense_budget",
                    "budget",
                ],
            )
        )

        if actual_expenses <= 0 or budget_expenses <= 0:
            return recommendations

        variance = (
            actual_expenses - budget_expenses
        )

        variance_pct = _percentage(
            variance,
            budget_expenses,
        )

        if (
            variance_pct
            > self.config.budget_variance_critical_pct
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="budget",
                    title="Correct critical budget overrun",
                    description=(
                        f"Actual expenses exceed budget by "
                        f"{variance_pct:.2f}%."
                    ),
                    action=(
                        "Freeze or review non-essential spending and "
                        "perform category-level variance analysis."
                    ),
                    priority="critical",
                    risk="critical",
                    confidence=0.96,
                    current_value=actual_expenses,
                    benchmark_value=budget_expenses,
                    variance=variance,
                    variance_pct=variance_pct,
                    potential_savings=max(variance, 0),
                    affected_metric="budget_variance",
                    evidence=[
                        f"Budget: "
                        f"{_format_currency(budget_expenses, self.config.currency)}",
                        f"Actual: "
                        f"{_format_currency(actual_expenses, self.config.currency)}",
                        f"Variance: "
                        f"{_format_currency(variance, self.config.currency)}",
                        f"Variance %: "
                        f"{_format_percent(variance_pct)}",
                    ],
                    risks=[
                        "Spending freezes may disrupt critical operations.",
                    ],
                    next_steps=[
                        "Identify largest budget variances.",
                        "Classify controllable versus uncontrollable costs.",
                        "Freeze non-essential spending.",
                        "Create corrective budget.",
                    ],
                    potential_impact=variance,
                )
            )

        elif (
            variance_pct
            > self.config.budget_variance_high_pct
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="budget",
                    title="Reduce budget variance",
                    description=(
                        f"Expenses are {variance_pct:.2f}% above budget."
                    ),
                    action=(
                        "Investigate major variance drivers and establish "
                        "corrective spending controls."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.92,
                    current_value=actual_expenses,
                    benchmark_value=budget_expenses,
                    variance=variance,
                    variance_pct=variance_pct,
                    potential_savings=variance * 0.50,
                    affected_metric="budget_variance",
                    evidence=[
                        f"Budget variance: "
                        f"{_format_percent(variance_pct)}",
                    ],
                    next_steps=[
                        "Rank budget variances.",
                        "Review discretionary expenses.",
                        "Update spending forecasts.",
                    ],
                    potential_impact=variance,
                )
            )

        elif (
            variance_pct
            > self.config.budget_variance_warning_pct
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="budget",
                    title="Monitor budget variance",
                    description=(
                        f"Expenses are {variance_pct:.2f}% above budget."
                    ),
                    action=(
                        "Monitor the variance and review categories that "
                        "are trending above plan."
                    ),
                    priority="medium",
                    risk="medium",
                    confidence=0.82,
                    current_value=actual_expenses,
                    benchmark_value=budget_expenses,
                    variance=variance,
                    variance_pct=variance_pct,
                    potential_savings=variance * 0.25,
                    affected_metric="budget_variance",
                    evidence=[
                        f"Budget variance: "
                        f"{_format_percent(variance_pct)}",
                    ],
                    next_steps=[
                        "Monitor monthly budget variance.",
                        "Review top overspending categories.",
                    ],
                    potential_impact=variance,
                )
            )

        return recommendations

    # ==========================================================================
    # Liquidity Recommendations
    # ==========================================================================

    def _liquidity_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        current_ratio = _safe_float(
            _first(
                data,
                ["current_ratio"],
            ),
            default=-1,
        )

        quick_ratio = _safe_float(
            _first(
                data,
                ["quick_ratio"],
            ),
            default=-1,
        )

        if (
            current_ratio >= 0
            and current_ratio
            < self.config.current_ratio_critical
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="liquidity",
                    title="Protect short-term liquidity",
                    description=(
                        f"Current ratio is {current_ratio:.2f}, "
                        "indicating elevated short-term liquidity pressure."
                    ),
                    action=(
                        "Accelerate receivables, prioritize cash-preserving "
                        "spending, review payment terms and maintain a "
                        "short-term liquidity buffer."
                    ),
                    priority="critical",
                    risk="critical",
                    confidence=0.96,
                    current_value=current_ratio,
                    benchmark_value=(
                        self.config.current_ratio_critical
                    ),
                    variance=(
                        current_ratio
                        - self.config.current_ratio_critical
                    ),
                    affected_metric="current_ratio",
                    evidence=[
                        f"Current ratio: {current_ratio:.2f}",
                        f"Critical threshold: "
                        f"{self.config.current_ratio_critical:.2f}",
                    ],
                    risks=[
                        "Liquidity shortages can affect supplier and payroll obligations.",
                        "Aggressive collections may affect customer relationships.",
                    ],
                    next_steps=[
                        "Review cash position weekly.",
                        "Prioritize overdue receivables.",
                        "Review upcoming liabilities.",
                        "Build a short-term cash plan.",
                    ],
                )
            )

        elif (
            current_ratio >= 0
            and current_ratio
            < self.config.current_ratio_warning
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="liquidity",
                    title="Improve current liquidity",
                    description=(
                        f"Current ratio is {current_ratio:.2f}, "
                        "below the preferred warning threshold."
                    ),
                    action=(
                        "Improve working-capital efficiency and preserve "
                        "cash until the liquidity position strengthens."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.91,
                    current_value=current_ratio,
                    benchmark_value=(
                        self.config.current_ratio_warning
                    ),
                    variance=(
                        current_ratio
                        - self.config.current_ratio_warning
                    ),
                    affected_metric="current_ratio",
                    evidence=[
                        f"Current ratio: {current_ratio:.2f}",
                    ],
                    next_steps=[
                        "Review receivables.",
                        "Review payment schedules.",
                        "Reduce unnecessary cash commitments.",
                    ],
                )
            )

        if (
            quick_ratio >= 0
            and quick_ratio
            < self.config.quick_ratio_critical
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="liquidity",
                    title="Strengthen immediate liquidity",
                    description=(
                        f"Quick ratio is {quick_ratio:.2f}, "
                        "indicating limited liquid assets relative "
                        "to short-term obligations."
                    ),
                    action=(
                        "Accelerate cash collection and review short-term "
                        "liabilities and financing requirements."
                    ),
                    priority="high",
                    risk="critical",
                    confidence=0.93,
                    current_value=quick_ratio,
                    benchmark_value=(
                        self.config.quick_ratio_critical
                    ),
                    variance=(
                        quick_ratio
                        - self.config.quick_ratio_critical
                    ),
                    affected_metric="quick_ratio",
                    evidence=[
                        f"Quick ratio: {quick_ratio:.2f}",
                    ],
                    next_steps=[
                        "Accelerate receivables.",
                        "Review short-term debt.",
                        "Preserve operating cash.",
                    ],
                )
            )

        return recommendations

    # ==========================================================================
    # Working Capital Recommendations
    # ==========================================================================

    def _working_capital_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        dso = _safe_float(
            _first(
                data,
                [
                    "dso",
                    "days_sales_outstanding",
                ],
            ),
            default=-1,
        )

        if (
            dso >= 0
            and dso > self.config.dso_critical_days
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="working_capital",
                    title="Accelerate receivables collection",
                    description=(
                        f"Days sales outstanding is {dso:.1f} days, "
                        "indicating slow customer collections."
                    ),
                    action=(
                        "Prioritize overdue invoices, automate payment "
                        "reminders, review credit terms and segment "
                        "customers by collection risk."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.94,
                    current_value=dso,
                    benchmark_value=self.config.dso_critical_days,
                    variance=(
                        dso
                        - self.config.dso_critical_days
                    ),
                    affected_metric="dso",
                    evidence=[
                        f"DSO: {dso:.1f} days",
                        f"Critical threshold: "
                        f"{self.config.dso_critical_days:.1f} days",
                    ],
                    next_steps=[
                        "Rank overdue invoices.",
                        "Contact high-value overdue accounts.",
                        "Review customer payment terms.",
                        "Introduce collection alerts.",
                    ],
                )
            )

        elif (
            dso >= 0
            and dso > self.config.dso_warning_days
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="working_capital",
                    title="Improve receivables efficiency",
                    description=(
                        f"DSO is {dso:.1f} days."
                    ),
                    action=(
                        "Monitor collections and improve invoice-to-cash "
                        "processes."
                    ),
                    priority="medium",
                    risk="medium",
                    confidence=0.87,
                    current_value=dso,
                    benchmark_value=self.config.dso_warning_days,
                    variance=(
                        dso
                        - self.config.dso_warning_days
                    ),
                    affected_metric="dso",
                    evidence=[
                        f"DSO: {dso:.1f} days",
                    ],
                    next_steps=[
                        "Monitor overdue receivables.",
                        "Automate reminders.",
                        "Review payment terms.",
                    ],
                )
            )

        dpo = _safe_float(
            _first(
                data,
                [
                    "dpo",
                    "days_payables_outstanding",
                ],
            ),
            default=-1,
        )

        if (
            dpo >= 0
            and dpo < self.config.dpo_critical_days
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="working_capital",
                    title="Optimize supplier payment timing",
                    description=(
                        f"Days payable outstanding is only {dpo:.1f} days."
                    ),
                    action=(
                        "Review supplier payment terms and negotiate "
                        "appropriate payment windows without damaging "
                        "supplier relationships."
                    ),
                    priority="high",
                    risk="medium",
                    confidence=0.87,
                    current_value=dpo,
                    benchmark_value=self.config.dpo_critical_days,
                    variance=(
                        dpo
                        - self.config.dpo_critical_days
                    ),
                    affected_metric="dpo",
                    evidence=[
                        f"DPO: {dpo:.1f} days",
                    ],
                    risks=[
                        "Extending payments too aggressively can affect supplier relationships.",
                        "Early-payment discounts may be financially beneficial.",
                    ],
                    next_steps=[
                        "Review supplier contracts.",
                        "Identify early-payment discounts.",
                        "Negotiate appropriate payment terms.",
                    ],
                )
            )

        return recommendations

    # ==========================================================================
    # Cash Flow Recommendations
    # ==========================================================================

    def _cash_flow_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        operating_cash_flow = _safe_float(
            _first(
                data,
                [
                    "operating_cash_flow",
                    "cash_from_operations",
                ]
            )
        )

        net_income = _safe_float(
            _first(
                data,
                ["net_profit", "net_income", "profit"],
            )
        )

        if (
            operating_cash_flow < 0
            and net_income > 0
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="cash_flow",
                    title="Investigate profit-to-cash conversion",
                    description=(
                        "The business reports positive accounting profit "
                        "while operating cash flow is negative."
                    ),
                    action=(
                        "Perform a cash-conversion analysis focused on "
                        "receivables, inventory, payables, deferred revenue "
                        "and non-cash accounting items."
                    ),
                    priority="critical",
                    risk="critical",
                    confidence=0.96,
                    current_value=operating_cash_flow,
                    benchmark_value=0.0,
                    variance=operating_cash_flow,
                    affected_metric="operating_cash_flow",
                    evidence=[
                        f"Operating cash flow: "
                        f"{_format_currency(operating_cash_flow, self.config.currency)}",
                        f"Net profit: "
                        f"{_format_currency(net_income, self.config.currency)}",
                    ],
                    risks=[
                        "Persistent negative operating cash flow can create funding pressure.",
                    ],
                    next_steps=[
                        "Analyze working-capital movements.",
                        "Review receivables aging.",
                        "Review inventory levels.",
                        "Review supplier payment timing.",
                        "Build a 13-week cash-flow forecast.",
                    ],
                    potential_impact=abs(operating_cash_flow),
                )
            )

        elif operating_cash_flow < 0:

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="cash_flow",
                    title="Improve operating cash flow",
                    description=(
                        "Operating cash flow is negative."
                    ),
                    action=(
                        "Prioritize cash collection, working-capital "
                        "optimization and discretionary spending control."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.92,
                    current_value=operating_cash_flow,
                    benchmark_value=0.0,
                    variance=operating_cash_flow,
                    affected_metric="operating_cash_flow",
                    evidence=[
                        f"Operating cash flow: "
                        f"{_format_currency(operating_cash_flow, self.config.currency)}",
                    ],
                    next_steps=[
                        "Review receivables.",
                        "Review inventory.",
                        "Reduce discretionary cash outflows.",
                    ],
                    potential_impact=abs(operating_cash_flow),
                )
            )

        return recommendations

    # ==========================================================================
    # Growth Recommendations
    # ==========================================================================

    def _growth_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        revenue_growth = _safe_float(
            _first(
                data,
                [
                    "revenue_growth_pct",
                    "revenue_growth",
                ],
            ),
            default=999,
        )

        expense_growth = _safe_float(
            _first(
                data,
                [
                    "expense_growth_pct",
                    "expense_growth",
                ],
            ),
            default=999,
        )

        if (
            revenue_growth != 999
            and expense_growth != 999
            and expense_growth > revenue_growth + 5
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="growth",
                    title="Align expense growth with revenue growth",
                    description=(
                        f"Expense growth ({expense_growth:.2f}%) "
                        f"is materially higher than revenue growth "
                        f"({revenue_growth:.2f}%)."
                    ),
                    action=(
                        "Review operating leverage and identify expense "
                        "categories that are growing faster than the "
                        "revenue base."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.92,
                    current_value=expense_growth,
                    benchmark_value=revenue_growth,
                    variance=(
                        expense_growth
                        - revenue_growth
                    ),
                    variance_pct=(
                        expense_growth
                        - revenue_growth
                    ),
                    affected_metric="expense_to_revenue_growth",
                    evidence=[
                        f"Revenue growth: "
                        f"{_format_percent(revenue_growth)}",
                        f"Expense growth: "
                        f"{_format_percent(expense_growth)}",
                    ],
                    next_steps=[
                        "Compare category growth rates.",
                        "Identify low-return expenses.",
                        "Review operating leverage.",
                        "Set expense-growth guardrails.",
                    ],
                )
            )

        return recommendations

    # ==========================================================================
    # Financial Health Recommendations
    # ==========================================================================

    def _financial_health_recommendations(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialRecommendation]:

        recommendations = []

        debt_to_equity = _safe_float(
            _first(
                data,
                [
                    "debt_to_equity",
                    "debt_equity_ratio",
                ],
            ),
            default=-1,
        )

        if (
            debt_to_equity >= 0
            and debt_to_equity
            > self.config.debt_to_equity_critical
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="financial_health",
                    title="Reduce excessive leverage",
                    description=(
                        f"Debt-to-equity ratio is "
                        f"{debt_to_equity:.2f}, above the critical threshold."
                    ),
                    action=(
                        "Review debt maturity, refinancing options, "
                        "cash generation and capital structure."
                    ),
                    priority="critical",
                    risk="critical",
                    confidence=0.95,
                    current_value=debt_to_equity,
                    benchmark_value=(
                        self.config.debt_to_equity_critical
                    ),
                    variance=(
                        debt_to_equity
                        - self.config.debt_to_equity_critical
                    ),
                    affected_metric="debt_to_equity",
                    evidence=[
                        f"Debt-to-equity: "
                        f"{debt_to_equity:.2f}",
                    ],
                    risks=[
                        "High leverage increases financial risk.",
                        "Refinancing conditions may change.",
                    ],
                    next_steps=[
                        "Review debt maturity schedule.",
                        "Calculate interest coverage.",
                        "Evaluate refinancing options.",
                        "Build deleveraging scenarios.",
                    ],
                )
            )

        elif (
            debt_to_equity >= 0
            and debt_to_equity
            > self.config.debt_to_equity_warning
        ):

            recommendations.append(
                self._create_recommendation(
                    recommendation_type="financial_health",
                    title="Monitor leverage",
                    description=(
                        f"Debt-to-equity ratio is "
                        f"{debt_to_equity:.2f}."
                    ),
                    action=(
                        "Monitor leverage and prioritize sustainable "
                        "cash generation before taking on additional debt."
                    ),
                    priority="high",
                    risk="high",
                    confidence=0.89,
                    current_value=debt_to_equity,
                    benchmark_value=(
                        self.config.debt_to_equity_warning
                    ),
                    variance=(
                        debt_to_equity
                        - self.config.debt_to_equity_warning
                    ),
                    affected_metric="debt_to_equity",
                    evidence=[
                        f"Debt-to-equity: "
                        f"{debt_to_equity:.2f}",
                    ],
                    next_steps=[
                        "Monitor leverage monthly.",
                        "Review debt service capacity.",
                        "Avoid unnecessary borrowing.",
                    ],
                )
            )

        return recommendations

    # ==========================================================================
    # Recommendation Creation
    # ==========================================================================

    def _create_recommendation(
        self,
        *,
        recommendation_type: str,
        title: str,
        description: str,
        action: str,
        priority: str,
        risk: str,
        confidence: float,
        current_value: Optional[float] = None,
        benchmark_value: Optional[float] = None,
        variance: Optional[float] = None,
        variance_pct: Optional[float] = None,
        estimated_impact: float = 0.0,
        potential_savings: float = 0.0,
        potential_revenue_uplift: float = 0.0,
        evidence: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        risks: Optional[List[str]] = None,
        next_steps: Optional[List[str]] = None,
        affected_metric: Optional[str] = None,
        affected_department: Optional[str] = None,
        affected_category: Optional[str] = None,
        potential_impact: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FinancialRecommendation:
        """Create normalized recommendation."""

        priority = _normalize_priority(priority)
        risk = _normalize_risk(risk)

        confidence = max(
            0.0,
            min(
                1.0,
                _safe_float(confidence),
            ),
        )

        if potential_impact is not None:
            estimated_impact = max(
                estimated_impact,
                _safe_float(potential_impact),
            )

        high_impact = (
            abs(
                max(
                    _safe_float(estimated_impact),
                    _safe_float(potential_savings),
                    _safe_float(potential_revenue_uplift),
                )
            )
            >= self.config.high_impact_amount
        )

        requires_human_review = (
            (
                self.config.require_human_review_for_high_risk
                and risk in {"high", "critical"}
            )
            or (
                self.config.require_human_review_for_high_impact
                and high_impact
            )
        )

        recommendation_id = _hash_id(
            recommendation_type,
            title,
            affected_metric,
            affected_category,
            affected_department,
        )

        return FinancialRecommendation(
            recommendation_id=recommendation_id,
            recommendation_type=recommendation_type,
            title=_normalize_text(title),
            description=_normalize_text(description),
            action=_normalize_text(action),
            priority=priority,
            risk=risk,
            confidence=_round(confidence, 4),
            impact_score=self._calculate_impact_score(
                priority=priority,
                risk=risk,
                confidence=confidence,
            ),
            current_value=(
                _round(current_value)
                if current_value is not None
                else None
            ),
            benchmark_value=(
                _round(benchmark_value)
                if benchmark_value is not None
                else None
            ),
            variance=(
                _round(variance)
                if variance is not None
                else None
            ),
            variance_pct=(
                _round(variance_pct)
                if variance_pct is not None
                else None
            ),
            estimated_impact=_round(
                estimated_impact
            ),
            potential_savings=_round(
                potential_savings
            ),
            potential_revenue_uplift=_round(
                potential_revenue_uplift
            ),
            currency=self.config.currency,
            evidence=evidence or [],
            assumptions=assumptions or [
                "Recommendation is based on available financial data.",
                "Historical comparisons are assumed to be comparable.",
            ],
            risks=risks or [],
            next_steps=next_steps or [
                "Validate the recommendation with financial owners.",
                "Estimate implementation cost.",
                "Review expected business impact.",
            ],
            affected_metric=affected_metric,
            affected_department=affected_department,
            affected_category=affected_category,
            requires_human_review=requires_human_review,
            metadata=metadata or {},
        )

    def _calculate_impact_score(
        self,
        *,
        priority: str,
        risk: str,
        confidence: float,
    ) -> float:

        priority_score = (
            PRIORITY_LEVELS.get(priority, 2)
            / 4
        )

        risk_score = (
            RISK_LEVELS.get(risk, 2)
            / 4
        )

        confidence_score = (
            max(
                0.0,
                min(1.0, confidence),
            )
        )

        return _round(
            (
                priority_score * 0.35
                + risk_score * 0.25
                + confidence_score * 0.40
            ),
            4,
        )

    # ==========================================================================
    # Post Processing
    # ==========================================================================

    def _filter_by_confidence(
        self,
        recommendations: Sequence[
            FinancialRecommendation
        ],
    ) -> List[FinancialRecommendation]:

        return [
            recommendation
            for recommendation in recommendations
            if recommendation.confidence
            >= self.config.minimum_confidence
        ]

    def _deduplicate(
        self,
        recommendations: Sequence[
            FinancialRecommendation
        ],
    ) -> List[FinancialRecommendation]:

        seen = set()
        output = []

        for recommendation in recommendations:

            key = (
                recommendation.recommendation_type,
                recommendation.affected_metric,
                recommendation.title.lower(),
            )

            if key in seen:
                continue

            seen.add(key)
            output.append(recommendation)

        return output

    def _sort_recommendations(
        self,
        recommendations: Sequence[
            FinancialRecommendation
        ],
    ) -> List[FinancialRecommendation]:

        return sorted(
            recommendations,
            key=lambda item: (
                PRIORITY_LEVELS.get(
                    item.priority,
                    0,
                ),
                RISK_LEVELS.get(
                    item.risk,
                    0,
                ),
                item.impact_score,
                item.confidence,
                abs(item.estimated_impact),
            ),
            reverse=True,
        )

    # ==========================================================================
    # Result Builder
    # ==========================================================================

    def _build_result(
        self,
        recommendations: Sequence[
            FinancialRecommendation
        ],
        context: Mapping[str, Any],
    ) -> FinancialRecommendationResult:

        recommendations = list(recommendations)

        high_priority_count = sum(
            1
            for recommendation in recommendations
            if recommendation.priority == "high"
        )

        critical_count = sum(
            1
            for recommendation in recommendations
            if recommendation.priority == "critical"
        )

        human_review_count = sum(
            1
            for recommendation in recommendations
            if recommendation.requires_human_review
        )

        total_impact = sum(
            recommendation.estimated_impact
            for recommendation in recommendations
        )

        total_savings = sum(
            recommendation.potential_savings
            for recommendation in recommendations
        )

        total_revenue_uplift = sum(
            recommendation.potential_revenue_uplift
            for recommendation in recommendations
        )

        average_confidence = _mean(
            recommendation.confidence
            for recommendation in recommendations
        )

        metrics = sorted(
            {
                recommendation.affected_metric
                for recommendation in recommendations
                if recommendation.affected_metric
            }
        )

        if critical_count:
            status = "critical"
        elif high_priority_count:
            status = "attention_required"
        elif recommendations:
            status = "monitor"
        else:
            status = "healthy"

        if recommendations:

            summary = (
                f"Generated {len(recommendations)} financial "
                f"recommendation(s). "
                f"{critical_count} are critical and "
                f"{high_priority_count} are high priority. "
                f"{human_review_count} require human review."
            )

        else:

            summary = (
                "No financial recommendations exceeded the "
                "configured confidence threshold."
            )

        result_id = _hash_id(
            "financial-recommendations",
            _timestamp(),
            len(recommendations),
        )

        return FinancialRecommendationResult(
            result_id=result_id,
            generated_at=_timestamp(),
            currency=self.config.currency,
            status=status,
            recommendations=recommendations,
            recommendations_count=len(recommendations),
            high_priority_count=high_priority_count,
            critical_count=critical_count,
            human_review_count=human_review_count,
            total_estimated_impact=_round(
                total_impact
            ),
            total_potential_savings=_round(
                total_savings
            ),
            total_potential_revenue_uplift=_round(
                total_revenue_uplift
            ),
            average_confidence=_round(
                average_confidence,
                4,
            ),
            summary=summary,
            metrics_analyzed=metrics,
            metadata={
                "engine": self.__class__.__name__,
                "version": "1.0.0",
                "recommendation_policy": (
                    "advisory_only"
                ),
                "human_review_required": (
                    human_review_count > 0
                ),
            },
        )

    # ==========================================================================
    # Export - JSON
    # ==========================================================================

    def save_json(
        self,
        result: FinancialRecommendationResult,
        path: Optional[str] = None,
    ) -> Path:

        output_path = Path(
            path
            or (
                Path(self.config.output_directory)
                / "financial_recommendations.json"
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            result.to_json(),
            encoding="utf-8",
        )

        return output_path

    # ==========================================================================
    # Export - Markdown
    # ==========================================================================

    def to_markdown(
        self,
        result: FinancialRecommendationResult,
    ) -> str:

        lines = [
            "# FinCo AI - Financial Recommendations",
            "",
            f"**Result ID:** `{result.result_id}`",
            f"**Generated:** `{result.generated_at}`",
            f"**Status:** `{result.status}`",
            "",
            "## Summary",
            "",
            result.summary,
            "",
            "## Metrics",
            "",
            f"- Recommendations: {result.recommendations_count}",
            f"- Critical: {result.critical_count}",
            f"- High Priority: {result.high_priority_count}",
            f"- Human Review: {result.human_review_count}",
            f"- Average Confidence: "
            f"{result.average_confidence:.2%}",
            f"- Estimated Impact: "
            f"{_format_currency(result.total_estimated_impact, result.currency)}",
            f"- Potential Savings: "
            f"{_format_currency(result.total_potential_savings, result.currency)}",
            f"- Potential Revenue Uplift: "
            f"{_format_currency(result.total_potential_revenue_uplift, result.currency)}",
            "",
            "## Recommendations",
            "",
        ]

        if not result.recommendations:

            lines.append(
                "No recommendations were generated."
            )

        for index, recommendation in enumerate(
            result.recommendations,
            start=1,
        ):

            lines.extend(
                [
                    f"### {index}. {recommendation.title}",
                    "",
                    f"**Type:** {recommendation.recommendation_type}",
                    "",
                    f"**Priority:** {recommendation.priority}",
                    "",
                    f"**Risk:** {recommendation.risk}",
                    "",
                    f"**Confidence:** "
                    f"{recommendation.confidence:.2%}",
                    "",
                    f"**Description:** "
                    f"{recommendation.description}",
                    "",
                    f"**Recommended Action:** "
                    f"{recommendation.action}",
                    "",
                ]
            )

            if recommendation.current_value is not None:
                lines.extend(
                    [
                        f"**Current Value:** "
                        f"{recommendation.current_value:,.2f}",
                        "",
                    ]
                )

            if recommendation.benchmark_value is not None:
                lines.extend(
                    [
                        f"**Benchmark:** "
                        f"{recommendation.benchmark_value:,.2f}",
                        "",
                    ]
                )

            if recommendation.variance_pct is not None:
                lines.extend(
                    [
                        f"**Variance:** "
                        f"{recommendation.variance_pct:,.2f}%",
                        "",
                    ]
                )

            if recommendation.potential_savings:
                lines.extend(
                    [
                        f"**Potential Savings:** "
                        f"{_format_currency(recommendation.potential_savings, result.currency)}",
                        "",
                    ]
                )

            if recommendation.potential_revenue_uplift:
                lines.extend(
                    [
                        f"**Potential Revenue Uplift:** "
                        f"{_format_currency(recommendation.potential_revenue_uplift, result.currency)}",
                        "",
                    ]
                )

            if recommendation.evidence:

                lines.extend(
                    [
                        "#### Evidence",
                        "",
                    ]
                )

                for evidence in recommendation.evidence:
                    lines.append(
                        f"- {evidence}"
                    )

                lines.append("")

            if recommendation.next_steps:

                lines.extend(
                    [
                        "#### Next Steps",
                        "",
                    ]
                )

                for step in recommendation.next_steps:
                    lines.append(
                        f"1. {step}"
                    )

                lines.append("")

            if recommendation.risks:

                lines.extend(
                    [
                        "#### Risks",
                        "",
                    ]
                )

                for risk in recommendation.risks:
                    lines.append(
                        f"- {risk}"
                    )

                lines.append("")

            if recommendation.requires_human_review:

                lines.extend(
                    [
                        "> ⚠️ **Human review required before execution.**",
                        "",
                    ]
                )

        return "\n".join(lines)

    def save_markdown(
        self,
        result: FinancialRecommendationResult,
        path: Optional[str] = None,
    ) -> Path:

        output_path = Path(
            path
            or (
                Path(self.config.output_directory)
                / "financial_recommendations.md"
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.to_markdown(result),
            encoding="utf-8",
        )

        return output_path

    # ==========================================================================
    # Export - HTML
    # ==========================================================================

    def to_html(
        self,
        result: FinancialRecommendationResult,
    ) -> str:

        status_class = {
            "critical": "critical",
            "attention_required": "warning",
            "monitor": "info",
            "healthy": "success",
        }.get(
            result.status,
            "info",
        )

        cards = []

        for recommendation in result.recommendations:

            evidence_html = "".join(
                f"<li>{html.escape(item)}</li>"
                for item in recommendation.evidence
            )

            steps_html = "".join(
                f"<li>{html.escape(item)}</li>"
                for item in recommendation.next_steps
            )

            risks_html = "".join(
                f"<li>{html.escape(item)}</li>"
                for item in recommendation.risks
            )

            review_html = ""

            if recommendation.requires_human_review:

                review_html = """
                <div class="review">
                    Human review required before execution.
                </div>
                """

            cards.append(
                f"""
                <article class="card">
                    <div class="card-header">
                        <div>
                            <span class="badge">
                                {html.escape(recommendation.priority.upper())}
                            </span>
                            <span class="badge">
                                {html.escape(recommendation.risk.upper())}
                            </span>
                        </div>

                        <span class="confidence">
                            {recommendation.confidence:.0%}
                            confidence
                        </span>
                    </div>

                    <h2>
                        {html.escape(recommendation.title)}
                    </h2>

                    <p>
                        {html.escape(recommendation.description)}
                    </p>

                    <h3>Recommended Action</h3>

                    <p>
                        {html.escape(recommendation.action)}
                    </p>

                    <div class="metrics">
                        <div>
                            <span>Current</span>
                            <strong>
                                {
                                    (
                                        f"{recommendation.current_value:,.2f}"
                                        if recommendation.current_value is not None
                                        else "—"
                                    )
                                }
                            </strong>
                        </div>

                        <div>
                            <span>Variance</span>
                            <strong>
                                {
                                    (
                                        f"{recommendation.variance_pct:,.2f}%"
                                        if recommendation.variance_pct is not None
                                        else "—"
                                    )
                                }
                            </strong>
                        </div>

                        <div>
                            <span>Savings</span>
                            <strong>
                                {
                                    _format_currency(
                                        recommendation.potential_savings,
                                        result.currency,
                                    )
                                }
                            </strong>
                        </div>
                    </div>

                    <h3>Evidence</h3>
                    <ul>{evidence_html}</ul>

                    <h3>Next Steps</h3>
                    <ol>{steps_html}</ol>

                    {
                        (
                            f"<h3>Risks</h3><ul>{risks_html}</ul>"
                            if risks_html
                            else ""
                        )
                    }

                    {review_html}
                </article>
                """
            )

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
    FinCo AI - Financial Recommendations
</title>

<style>
    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        padding: 32px;
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
        background: #f4f7fb;
        color: #182230;
    }}

    .container {{
        max-width: 1200px;
        margin: 0 auto;
    }}

    .hero {{
        background: #ffffff;
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
    }}

    .hero h1 {{
        margin: 0 0 8px;
        font-size: 32px;
    }}

    .hero p {{
        color: #64748b;
        margin-bottom: 0;
    }}

    .status {{
        display: inline-block;
        margin-top: 18px;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
    }}

    .status.critical {{
        background: #fee2e2;
        color: #991b1b;
    }}

    .status.warning {{
        background: #fef3c7;
        color: #92400e;
    }}

    .status.info {{
        background: #dbeafe;
        color: #1e40af;
    }}

    .status.success {{
        background: #dcfce7;
        color: #166534;
    }}

    .summary {{
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }}

    .stat {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
    }}

    .stat span {{
        display: block;
        color: #64748b;
        font-size: 13px;
        margin-bottom: 8px;
    }}

    .stat strong {{
        font-size: 24px;
    }}

    .card {{
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow:
            0 8px 24px rgba(15, 23, 42, 0.04);
    }}

    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }}

    .badge {{
        display: inline-block;
        padding: 6px 10px;
        margin-right: 6px;
        border-radius: 999px;
        background: #eef2ff;
        font-size: 11px;
        font-weight: 800;
    }}

    .confidence {{
        color: #475569;
        font-size: 13px;
        font-weight: 700;
    }}

    .card h2 {{
        margin-bottom: 8px;
    }}

    .card h3 {{
        margin-top: 24px;
        font-size: 15px;
    }}

    .card p {{
        line-height: 1.7;
        color: #475569;
    }}

    .metrics {{
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin: 20px 0;
    }}

    .metrics > div {{
        padding: 14px;
        background: #f8fafc;
        border-radius: 12px;
    }}

    .metrics span {{
        display: block;
        color: #64748b;
        font-size: 12px;
        margin-bottom: 6px;
    }}

    .metrics strong {{
        font-size: 16px;
    }}

    li {{
        margin-bottom: 8px;
        color: #475569;
    }}

    .review {{
        margin-top: 20px;
        padding: 14px;
        border-radius: 12px;
        background: #fff7ed;
        color: #9a3412;
        font-weight: 700;
    }}

    .footer {{
        text-align: center;
        color: #64748b;
        padding: 24px;
        font-size: 13px;
    }}

    @media (max-width: 640px) {{
        body {{
            padding: 16px;
        }}

        .hero {{
            padding: 22px;
        }}

        .hero h1 {{
            font-size: 25px;
        }}
    }}
</style>
</head>

<body>

<div class="container">

    <section class="hero">

        <h1>
            FinCo AI
        </h1>

        <h2>
            Financial Recommendations
        </h2>

        <p>
            Explainable financial recommendations generated from
            financial performance, liquidity, profitability and
            working-capital indicators.
        </p>

        <div class="status {status_class}">
            {html.escape(result.status.upper())}
        </div>

    </section>

    <section class="summary">

        <div class="stat">
            <span>Recommendations</span>
            <strong>
                {result.recommendations_count}
            </strong>
        </div>

        <div class="stat">
            <span>Critical</span>
            <strong>
                {result.critical_count}
            </strong>
        </div>

        <div class="stat">
            <span>High Priority</span>
            <strong>
                {result.high_priority_count}
            </strong>
        </div>

        <div class="stat">
            <span>Human Review</span>
            <strong>
                {result.human_review_count}
            </strong>
        </div>

        <div class="stat">
            <span>Confidence</span>
            <strong>
                {result.average_confidence:.1%}
            </strong>
        </div>

        <div class="stat">
            <span>Potential Savings</span>
            <strong>
                {
                    _format_currency(
                        result.total_potential_savings,
                        result.currency,
                    )
                }
            </strong>
        </div>

    </section>

    <section>
        {"".join(cards)}
    </section>

    <div class="footer">
        FinCo AI · Advisory recommendations only ·
        Human validation required before financial execution
    </div>

</div>

</body>
</html>
"""

    def save_html(
        self,
        result: FinancialRecommendationResult,
        path: Optional[str] = None,
    ) -> Path:

        output_path = Path(
            path
            or (
                Path(self.config.output_directory)
                / "financial_recommendations.html"
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.to_html(result),
            encoding="utf-8",
        )

        return output_path

    # ==========================================================================
    # Export All
    # ==========================================================================

    def save_all(
        self,
        result: FinancialRecommendationResult,
        output_directory: Optional[str] = None,
    ) -> Dict[str, Path]:

        directory = Path(
            output_directory
            or self.config.output_directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return {
            "json": self.save_json(
                result,
                str(
                    directory
                    / "financial_recommendations.json"
                ),
            ),
            "markdown": self.save_markdown(
                result,
                str(
                    directory
                    / "financial_recommendations.md"
                ),
            ),
            "html": self.save_html(
                result,
                str(
                    directory
                    / "financial_recommendations.html"
                ),
            ),
        }


# ============================================================================
# Convenience API
# ============================================================================


def generate_financial_recommendations(
    financial_data: Optional[Any] = None,
    *,
    revenue_data: Optional[Any] = None,
    expense_data: Optional[Any] = None,
    budget_data: Optional[Any] = None,
    liquidity_data: Optional[Any] = None,
    working_capital_data: Optional[Any] = None,
    profitability_data: Optional[Any] = None,
    config: Optional[
        FinancialRecommendationConfig
    ] = None,
) -> FinancialRecommendationResult:
    """
    Convenience function for generating recommendations.
    """

    engine = FinancialRecommendationEngine(
        config=config
    )

    return engine.generate(
        financial_data=financial_data,
        revenue_data=revenue_data,
        expense_data=expense_data,
        budget_data=budget_data,
        liquidity_data=liquidity_data,
        working_capital_data=working_capital_data,
        profitability_data=profitability_data,
    )


# ============================================================================
# Demo Data
# ============================================================================


def create_demo_financial_data() -> Dict[str, Any]:
    """Create realistic demo financial data."""

    return {
        "revenue": 1_000_000,
        "previous_revenue": 1_130_000,

        "expenses": 760_000,
        "previous_expenses": 670_000,

        "net_profit": 38_000,

        "gross_margin": 0.23,
        "previous_gross_margin": 0.31,

        "net_margin": 0.038,

        "budget_expenses": 650_000,
        "actual_expenses": 760_000,

        "current_ratio": 0.96,
        "quick_ratio": 0.74,

        "dso": 96,
        "dpo": 5,

        "operating_cash_flow": -85_000,

        "revenue_growth_pct": -11.50,
        "expense_growth_pct": 13.43,

        "debt_to_equity": 2.7,
    }


def create_demo_result() -> FinancialRecommendationResult:
    """Generate demo recommendation result."""

    data = create_demo_financial_data()

    engine = FinancialRecommendationEngine()

    return engine.generate(
        financial_data=data
    )


# ============================================================================
# CLI
# ============================================================================


def _build_cli() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "FinCo AI financial recommendation engine"
        )
    )

    parser.add_argument(
        "--output",
        default="reports/recommendations",
        help="Output directory.",
    )

    parser.add_argument(
        "--format",
        choices=[
            "json",
            "markdown",
            "html",
            "all",
        ],
        default="all",
        help="Output format.",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate recommendations using demo data.",
    )

    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = _build_cli()

    args = parser.parse_args()

    if not args.demo:

        print(
            "No input dataset supplied. "
            "Use --demo to run the built-in demonstration."
        )

        return 0

    config = FinancialRecommendationConfig(
        output_directory=args.output
    )

    engine = FinancialRecommendationEngine(
        config=config
    )

    result = engine.generate(
        financial_data=create_demo_financial_data()
    )

    print()
    print("=" * 72)
    print("FinCo AI - Financial Recommendations")
    print("=" * 72)
    print()
    print(f"Result ID:             {result.result_id}")
    print(f"Status:                {result.status}")
    print(
        f"Recommendations:       "
        f"{result.recommendations_count}"
    )
    print(
        f"Critical:              "
        f"{result.critical_count}"
    )
    print(
        f"High Priority:         "
        f"{result.high_priority_count}"
    )
    print(
        f"Human Review:          "
        f"{result.human_review_count}"
    )
    print(
        f"Average Confidence:    "
        f"{result.average_confidence:.2%}"
    )
    print(
        f"Potential Savings:     "
        f"{_format_currency(result.total_potential_savings, result.currency)}"
    )
    print(
        f"Revenue Uplift:        "
        f"{_format_currency(result.total_potential_revenue_uplift, result.currency)}"
    )
    print()

    for index, recommendation in enumerate(
        result.recommendations,
        start=1,
    ):

        print(
            f"{index}. "
            f"[{recommendation.priority.upper()}] "
            f"{recommendation.title}"
        )

        print(
            f"   Type:       "
            f"{recommendation.recommendation_type}"
        )

        print(
            f"   Confidence: "
            f"{recommendation.confidence:.2%}"
        )

        print(
            f"   Action:     "
            f"{recommendation.action}"
        )

        if recommendation.requires_human_review:

            print(
                "   Review:     HUMAN REVIEW REQUIRED"
            )

        print()

    if args.format == "json":

        path = engine.save_json(
            result,
            str(
                Path(args.output)
                / "financial_recommendations.json"
            ),
        )

        print(f"JSON saved to: {path}")

    elif args.format == "markdown":

        path = engine.save_markdown(
            result,
            str(
                Path(args.output)
                / "financial_recommendations.md"
            ),
        )

        print(f"Markdown saved to: {path}")

    elif args.format == "html":

        path = engine.save_html(
            result,
            str(
                Path(args.output)
                / "financial_recommendations.html"
            ),
        )

        print(f"HTML saved to: {path}")

    else:

        paths = engine.save_all(
            result,
            args.output,
        )

        print("Files generated:")

        for file_type, path in paths.items():

            print(
                f"  {file_type:<10} {path}"
            )

    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())