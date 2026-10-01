"""
FinCo AI - Cost Recommendations
================================

File:
    backend/app/recommendations/cost_recommendations.py

Purpose:
    Generate explainable cost-optimization recommendations from
    financial, expense, budget, supplier, and operational data.

Architecture:

    Financial Data
          |
          v
    Cost Analysis
          |
          +--> Budget Variance
          +--> Expense Growth
          +--> Margin Pressure
          +--> Category Inefficiency
          +--> Supplier Cost
          +--> Fixed Cost
          +--> Variable Cost
          +--> Cost Concentration
          |
          v
    Recommendation Engine
          |
          +--> Priority
          +--> Impact
          +--> Confidence
          +--> Evidence
          +--> Expected Savings
          +--> Risk
          |
          v
    Human Review / Approval
          |
          v
    Action / Audit Log

Important:
    Recommendations are proposals, not automatic financial actions.
    High-impact recommendations should be reviewed by an authorized user.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIORITY_LEVELS = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}

RISK_LEVELS = {
    "critical",
    "high",
    "medium",
    "low",
    "minimal",
}

RECOMMENDATION_TYPES = {
    "budget_variance",
    "expense_growth",
    "category_optimization",
    "supplier_optimization",
    "fixed_cost_reduction",
    "variable_cost_optimization",
    "duplicate_expense",
    "cost_concentration",
    "margin_protection",
    "overhead_optimization",
    "cash_preservation",
    "general_cost_control",
}

DEFAULT_CURRENCY = "USD"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().isoformat()


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return default

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_divide(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)

    if denominator == 0:
        return default

    return numerator / denominator


def _percentage(
    numerator: Any,
    denominator: Any,
) -> float:
    return _safe_divide(
        numerator,
        denominator,
    ) * 100.0


def _round(
    value: Any,
    digits: int = 2,
) -> float:
    return round(
        _safe_float(value),
        digits,
    )


def _format_currency(
    value: Any,
    currency: str = DEFAULT_CURRENCY,
) -> str:
    return (
        f"{currency} "
        f"{_safe_float(value):,.2f}"
    )


def _format_percent(
    value: Any,
) -> str:
    return f"{_safe_float(value):.2f}%"


def _normalize_text(
    value: Any,
) -> str:
    return str(value or "").strip()


def _normalize_priority(
    value: Any,
) -> str:

    normalized = (
        _normalize_text(value)
        .lower()
    )

    if normalized in PRIORITY_LEVELS:
        return normalized

    return "medium"


def _normalize_risk(
    value: Any,
) -> str:

    normalized = (
        _normalize_text(value)
        .lower()
    )

    if normalized in RISK_LEVELS:
        return normalized

    return "medium"


def _json_safe(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "to_dict"):
        try:
            return _json_safe(
                value.to_dict()
            )
        except Exception:
            pass

    if hasattr(value, "model_dump"):
        try:
            return _json_safe(
                value.model_dump()
            )
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return _json_safe(
                value.dict()
            )
        except Exception:
            pass

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    if hasattr(value, "__dict__"):
        try:
            return _json_safe(
                vars(value)
            )
        except Exception:
            pass

    return str(value)


def _get(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:

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


def _first(
    obj: Any,
    keys: Sequence[str],
    default: Any = None,
) -> Any:

    for key in keys:
        value = _get(
            obj,
            key,
            None,
        )

        if value is not None:
            return value

    return default


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CostRecommendationConfig:
    """
    Configuration for cost recommendation generation.
    """

    currency: str = DEFAULT_CURRENCY

    # Budget variance
    budget_variance_threshold_percent: float = 10.0
    high_budget_variance_percent: float = 20.0
    critical_budget_variance_percent: float = 35.0

    # Expense growth
    expense_growth_threshold_percent: float = 10.0
    high_expense_growth_percent: float = 20.0
    critical_expense_growth_percent: float = 35.0

    # Cost concentration
    concentration_threshold_percent: float = 30.0
    high_concentration_percent: float = 50.0

    # Savings assumptions
    default_savings_percent: float = 5.0
    supplier_savings_percent: float = 8.0
    overhead_savings_percent: float = 5.0
    variable_cost_savings_percent: float = 4.0

    # Recommendation limits
    max_recommendations: int = 20

    # Human review
    require_human_review_for_high_impact: bool = True

    high_impact_savings_threshold: float = 50000.0

    # Confidence
    minimum_confidence: float = 0.55

    # Duplicate detection
    duplicate_amount_tolerance_percent: float = 1.0

    # Output
    output_directory: str = "reports/recommendations"


# ---------------------------------------------------------------------------
# Input data models
# ---------------------------------------------------------------------------


@dataclass
class ExpenseRecord:
    """
    Normalized expense record.
    """

    expense_id: str

    category: str

    amount: float

    date: Optional[str] = None

    supplier: Optional[str] = None

    department: Optional[str] = None

    expense_type: str = "variable"

    description: Optional[str] = None

    budget_amount: Optional[float] = None

    previous_amount: Optional[float] = None

    currency: str = DEFAULT_CURRENCY


@dataclass
class BudgetRecord:
    """
    Normalized budget record.
    """

    category: str

    budget_amount: float

    actual_amount: float

    period: Optional[str] = None

    department: Optional[str] = None


# ---------------------------------------------------------------------------
# Recommendation model
# ---------------------------------------------------------------------------


@dataclass
class CostRecommendation:
    """
    Individual explainable cost recommendation.
    """

    recommendation_id: str

    recommendation_type: str

    title: str

    description: str

    action: str

    priority: str

    risk_level: str

    confidence: float

    current_cost: float = 0.0

    benchmark_cost: float = 0.0

    potential_savings: float = 0.0

    savings_percent: float = 0.0

    affected_category: Optional[str] = None

    affected_supplier: Optional[str] = None

    affected_department: Optional[str] = None

    evidence: List[str] = field(
        default_factory=list
    )

    assumptions: List[str] = field(
        default_factory=list
    )

    risks: List[str] = field(
        default_factory=list
    )

    next_steps: List[str] = field(
        default_factory=list
    )

    requires_human_review: bool = False

    generated_at: str = field(
        default_factory=_timestamp
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(
            asdict(self)
        )


# ---------------------------------------------------------------------------
# Recommendation result
# ---------------------------------------------------------------------------


@dataclass
class CostRecommendationResult:
    """
    Complete output of the cost recommendation engine.
    """

    result_id: str

    generated_at: str

    currency: str

    status: str

    recommendations: List[
        CostRecommendation
    ] = field(
        default_factory=list
    )

    total_current_cost: float = 0.0

    estimated_total_savings: float = 0.0

    average_savings_percent: float = 0.0

    high_priority_count: int = 0

    human_review_count: int = 0

    categories_analyzed: int = 0

    suppliers_analyzed: int = 0

    expenses_analyzed: int = 0

    summary: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(
            asdict(self)
        )

    def to_json(
        self,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Main recommendation engine
# ---------------------------------------------------------------------------


class CostRecommendationEngine:
    """
    Explainable cost optimization engine.

    It does not automatically execute financial actions.

    It detects opportunities using:

        1. Budget variance
        2. Expense growth
        3. Supplier concentration
        4. Category concentration
        5. Fixed-cost pressure
        6. Variable-cost pressure
        7. Duplicate-looking expenses
        8. Margin pressure
        9. Cash preservation opportunities
    """

    def __init__(
        self,
        config: Optional[
            CostRecommendationConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or CostRecommendationConfig()
        )

        self.output_directory = Path(
            self.config.output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        expenses: Optional[
            Sequence[Any]
        ] = None,
        budgets: Optional[
            Sequence[Any]
        ] = None,
        financial_data: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> CostRecommendationResult:
        """
        Generate cost optimization recommendations.

        Parameters
        ----------
        expenses:
            Expense records.

        budgets:
            Budget records.

        financial_data:
            Optional high-level financial metrics.

        Returns
        -------
        CostRecommendationResult
        """

        normalized_expenses = (
            self._normalize_expenses(
                expenses
            )
        )

        normalized_budgets = (
            self._normalize_budgets(
                budgets
            )
        )

        recommendations: List[
            CostRecommendation
        ] = []

        # --------------------------------------------------------------
        # 1. Budget variance
        # --------------------------------------------------------------

        recommendations.extend(
            self._budget_variance_recommendations(
                normalized_budgets
            )
        )

        # --------------------------------------------------------------
        # 2. Expense growth
        # --------------------------------------------------------------

        recommendations.extend(
            self._expense_growth_recommendations(
                normalized_expenses
            )
        )

        # --------------------------------------------------------------
        # 3. Category optimization
        # --------------------------------------------------------------

        recommendations.extend(
            self._category_recommendations(
                normalized_expenses
            )
        )

        # --------------------------------------------------------------
        # 4. Supplier optimization
        # --------------------------------------------------------------

        recommendations.extend(
            self._supplier_recommendations(
                normalized_expenses
            )
        )

        # --------------------------------------------------------------
        # 5. Fixed / variable costs
        # --------------------------------------------------------------

        recommendations.extend(
            self._fixed_cost_recommendations(
                normalized_expenses
            )
        )

        recommendations.extend(
            self._variable_cost_recommendations(
                normalized_expenses
            )
        )

        # --------------------------------------------------------------
        # 6. Duplicate expenses
        # --------------------------------------------------------------

        recommendations.extend(
            self._duplicate_expense_recommendations(
                normalized_expenses
            )
        )

        # --------------------------------------------------------------
        # 7. Cost concentration
        # --------------------------------------------------------------

        recommendations.extend(
            self._cost_concentration_recommendations(
                normalized_expenses
            )
        )

        # --------------------------------------------------------------
        # 8. Margin pressure
        # --------------------------------------------------------------

        if financial_data:

            recommendations.extend(
                self._margin_recommendations(
                    financial_data
                )
            )

        # --------------------------------------------------------------
        # 9. Cash preservation
        # --------------------------------------------------------------

        if financial_data:

            recommendations.extend(
                self._cash_preservation_recommendations(
                    financial_data
                )
            )

        # --------------------------------------------------------------
        # Filter + rank
        # --------------------------------------------------------------

        recommendations = (
            self._filter_recommendations(
                recommendations
            )
        )

        recommendations = (
            self._deduplicate_recommendations(
                recommendations
            )
        )

        recommendations = sorted(
            recommendations,
            key=self._recommendation_sort_key,
        )

        recommendations = recommendations[
            : self.config.max_recommendations
        ]

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        total_current_cost = sum(
            expense.amount
            for expense
            in normalized_expenses
        )

        estimated_savings = sum(
            recommendation.potential_savings
            for recommendation
            in recommendations
        )

        average_savings = (
            _percentage(
                estimated_savings,
                total_current_cost,
            )
            if total_current_cost
            else 0.0
        )

        high_priority_count = sum(
            recommendation.priority
            in {
                "critical",
                "high",
            }
            for recommendation
            in recommendations
        )

        human_review_count = sum(
            recommendation.requires_human_review
            for recommendation
            in recommendations
        )

        categories = {
            expense.category
            for expense
            in normalized_expenses
        }

        suppliers = {
            expense.supplier
            for expense
            in normalized_expenses
            if expense.supplier
        }

        status = (
            "recommendations_available"
            if recommendations
            else "no_actionable_recommendations"
        )

        result = CostRecommendationResult(
            result_id=self._create_result_id(),
            generated_at=_timestamp(),
            currency=self.config.currency,
            status=status,
            recommendations=recommendations,
            total_current_cost=_round(
                total_current_cost
            ),
            estimated_total_savings=_round(
                estimated_savings
            ),
            average_savings_percent=_round(
                average_savings
            ),
            high_priority_count=(
                high_priority_count
            ),
            human_review_count=(
                human_review_count
            ),
            categories_analyzed=len(
                categories
            ),
            suppliers_analyzed=len(
                suppliers
            ),
            expenses_analyzed=len(
                normalized_expenses
            ),
            summary=self._build_summary(
                recommendations,
                total_current_cost,
                estimated_savings,
            ),
            metadata={
                "engine": (
                    "CostRecommendationEngine"
                ),
                "engine_version": "1.0.0",
                "recommendation_count": len(
                    recommendations
                ),
            },
        )

        return result

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_expenses(
        self,
        expenses: Optional[
            Sequence[Any]
        ],
    ) -> List[ExpenseRecord]:

        if not expenses:
            return []

        result = []

        for index, raw in enumerate(
            expenses
        ):

            amount = _safe_float(
                _first(
                    raw,
                    [
                        "amount",
                        "expense_amount",
                        "cost",
                        "value",
                    ],
                    0,
                )
            )

            category = _normalize_text(
                _first(
                    raw,
                    [
                        "category",
                        "expense_category",
                        "type",
                    ],
                    "Other",
                )
            )

            expense_type = _normalize_text(
                _first(
                    raw,
                    [
                        "expense_type",
                        "cost_type",
                    ],
                    "variable",
                )
            ).lower()

            if expense_type not in {
                "fixed",
                "variable",
            }:
                expense_type = "variable"

            result.append(
                ExpenseRecord(
                    expense_id=str(
                        _first(
                            raw,
                            [
                                "expense_id",
                                "id",
                                "transaction_id",
                            ],
                            f"EXP-{index + 1:05d}",
                        )
                    ),
                    category=category,
                    amount=amount,
                    date=_first(
                        raw,
                        [
                            "date",
                            "expense_date",
                            "transaction_date",
                        ],
                    ),
                    supplier=_first(
                        raw,
                        [
                            "supplier",
                            "vendor",
                            "merchant",
                        ],
                    ),
                    department=_first(
                        raw,
                        [
                            "department",
                            "business_unit",
                        ],
                    ),
                    expense_type=expense_type,
                    description=_first(
                        raw,
                        [
                            "description",
                            "memo",
                        ],
                    ),
                    budget_amount=(
                        _safe_float(
                            _first(
                                raw,
                                [
                                    "budget_amount",
                                    "budget",
                                ],
                                0,
                            )
                        )
                        or None
                    ),
                    previous_amount=(
                        _safe_float(
                            _first(
                                raw,
                                [
                                    "previous_amount",
                                    "prior_amount",
                                    "last_period_amount",
                                ],
                                0,
                            )
                        )
                        or None
                    ),
                    currency=_first(
                        raw,
                        ["currency"],
                        self.config.currency,
                    ),
                )
            )

        return result

    def _normalize_budgets(
        self,
        budgets: Optional[
            Sequence[Any]
        ],
    ) -> List[BudgetRecord]:

        if not budgets:
            return []

        result = []

        for raw in budgets:

            result.append(
                BudgetRecord(
                    category=_normalize_text(
                        _first(
                            raw,
                            [
                                "category",
                                "expense_category",
                            ],
                            "Other",
                        )
                    ),
                    budget_amount=_safe_float(
                        _first(
                            raw,
                            [
                                "budget_amount",
                                "budget",
                            ],
                            0,
                        )
                    ),
                    actual_amount=_safe_float(
                        _first(
                            raw,
                            [
                                "actual_amount",
                                "actual",
                                "expense",
                            ],
                            0,
                        )
                    ),
                    period=_first(
                        raw,
                        [
                            "period",
                            "month",
                            "year",
                        ],
                    ),
                    department=_first(
                        raw,
                        ["department"],
                    ),
                )
            )

        return result

    # ------------------------------------------------------------------
    # Budget variance
    # ------------------------------------------------------------------

    def _budget_variance_recommendations(
        self,
        budgets: Sequence[
            BudgetRecord
        ],
    ) -> List[
        CostRecommendation
    ]:

        recommendations = []

        for budget in budgets:

            if budget.budget_amount <= 0:
                continue

            variance = (
                budget.actual_amount
                - budget.budget_amount
            )

            variance_percent = _percentage(
                variance,
                budget.budget_amount,
            )

            if (
                variance_percent
                < self.config.budget_variance_threshold_percent
            ):
                continue

            if (
                variance_percent
                >= self.config.critical_budget_variance_percent
            ):
                priority = "critical"
                risk = "critical"

            elif (
                variance_percent
                >= self.config.high_budget_variance_percent
            ):
                priority = "high"
                risk = "high"

            else:
                priority = "medium"
                risk = "medium"

            savings = (
                max(variance, 0)
                * self.config.default_savings_percent
                / 100
            )

            recommendations.append(
                self._create_recommendation(
                    recommendation_type=(
                        "budget_variance"
                    ),
                    title=(
                        f"Reduce overspending in "
                        f"{budget.category}"
                    ),
                    description=(
                        f"{budget.category} is "
                        f"{_format_percent(variance_percent)} "
                        f"above its budget."
                    ),
                    action=(
                        "Review the largest cost drivers, "
                        "freeze non-essential spending, "
                        "and establish a corrective "
                        "spending limit."
                    ),
                    priority=priority,
                    risk_level=risk,
                    confidence=0.92,
                    current_cost=(
                        budget.actual_amount
                    ),
                    benchmark_cost=(
                        budget.budget_amount
                    ),
                    potential_savings=savings,
                    savings_percent=(
                        _percentage(
                            savings,
                            budget.actual_amount,
                        )
                    ),
                    affected_category=(
                        budget.category
                    ),
                    evidence=[
                        (
                            f"Budget: "
                            f"{_format_currency(budget.budget_amount, self.config.currency)}"
                        ),
                        (
                            f"Actual: "
                            f"{_format_currency(budget.actual_amount, self.config.currency)}"
                        ),
                        (
                            f"Variance: "
                            f"{_format_currency(variance, self.config.currency)}"
                        ),
                        (
                            f"Variance percentage: "
                            f"{_format_percent(variance_percent)}"
                        ),
                    ],
                    assumptions=[
                        "Budget figures are accurate.",
                        "Variance is not caused by a deliberate one-time investment.",
                    ],
                    risks=[
                        "Aggressive cost reduction may affect operations.",
                        "Some budget overruns may be strategically justified.",
                    ],
                    next_steps=[
                        "Identify the top three drivers of the variance.",
                        "Review non-essential purchases.",
                        "Create a corrective spending plan.",
                    ],
                    metadata={
                        "period": budget.period,
                        "department": budget.department,
                    },
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Expense growth
    # ------------------------------------------------------------------

    def _expense_growth_recommendations(
        self,
        expenses: Sequence[
            ExpenseRecord
        ],
    ) -> List[
        CostRecommendation
    ]:

        recommendations = []

        for expense in expenses:

            if (
                expense.previous_amount
                is None
                or expense.previous_amount <= 0
            ):
                continue

            growth = _percentage(
                expense.amount
                - expense.previous_amount,
                expense.previous_amount,
            )

            if (
                growth
                < self.config.expense_growth_threshold_percent
            ):
                continue

            if (
                growth
                >= self.config.critical_expense_growth_percent
            ):
                priority = "critical"
                risk = "critical"

            elif (
                growth
                >= self.config.high_expense_growth_percent
            ):
                priority = "high"
                risk = "high"

            else:
                priority = "medium"
                risk = "medium"

            incremental_cost = (
                expense.amount
                - expense.previous_amount
            )

            savings = (
                incremental_cost
                * self.config.default_savings_percent
                / 100
            )

            recommendations.append(
                self._create_recommendation(
                    recommendation_type=(
                        "expense_growth"
                    ),
                    title=(
                        f"Investigate rising "
                        f"{expense.category} costs"
                    ),
                    description=(
                        f"{expense.category} spending increased "
                        f"by {_format_percent(growth)} "
                        f"versus the previous period."
                    ),
                    action=(
                        "Perform a driver-level cost review "
                        "and negotiate or eliminate the "
                        "highest-growth expenses."
                    ),
                    priority=priority,
                    risk_level=risk,
                    confidence=0.87,
                    current_cost=expense.amount,
                    benchmark_cost=(
                        expense.previous_amount
                    ),
                    potential_savings=savings,
                    savings_percent=_percentage(
                        savings,
                        expense.amount,
                    ),
                    affected_category=(
                        expense.category
                    ),
                    affected_supplier=(
                        expense.supplier
                    ),
                    affected_department=(
                        expense.department
                    ),
                    evidence=[
                        (
                            f"Current cost: "
                            f"{_format_currency(expense.amount, self.config.currency)}"
                        ),
                        (
                            f"Previous cost: "
                            f"{_format_currency(expense.previous_amount, self.config.currency)}"
                        ),
                        (
                            f"Growth: "
                            f"{_format_percent(growth)}"
                        ),
                    ],
                    assumptions=[
                        "Previous-period value is comparable.",
                        "Cost increase is not entirely caused by volume growth.",
                    ],
                    risks=[
                        "Cost reductions may reduce service levels.",
                    ],
                    next_steps=[
                        "Break the increase into price and volume effects.",
                        "Review supplier pricing.",
                        "Identify avoidable spend.",
                    ],
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Category optimization
    # ------------------------------------------------------------------

    def _category_recommendations(
        self,
        expenses: Sequence[
            ExpenseRecord
        ],
    ) -> List[
        CostRecommendation
    ]:

        recommendations = []

        category_totals: Dict[
            str,
            float,
        ] = {}

        for expense in expenses:

            category_totals[
                expense.category
            ] = (
                category_totals.get(
                    expense.category,
                    0.0,
                )
                + expense.amount
            )

        total = sum(
            category_totals.values()
        )

        if total <= 0:
            return recommendations

        for category, amount in (
            category_totals.items()
        ):

            share = _percentage(
                amount,
                total,
            )

            if (
                share
                < self.config.concentration_threshold_percent
            ):
                continue

            if (
                share
                >= self.config.high_concentration_percent
            ):
                priority = "high"
                risk = "high"
            else:
                priority = "medium"
                risk = "medium"

            savings = (
                amount
                * self.config.default_savings_percent
                / 100
            )

            recommendations.append(
                self._create_recommendation(
                    recommendation_type=(
                        "category_optimization"
                    ),
                    title=(
                        f"Optimize concentrated "
                        f"{category} spending"
                    ),
                    description=(
                        f"{category} represents "
                        f"{_format_percent(share)} "
                        f"of analyzed expenses."
                    ),
                    action=(
                        "Perform category-level spend "
                        "segmentation and establish "
                        "preferred suppliers, approval "
                        "limits, and purchasing controls."
                    ),
                    priority=priority,
                    risk_level=risk,
                    confidence=0.82,
                    current_cost=amount,
                    potential_savings=savings,
                    savings_percent=(
                        self.config.default_savings_percent
                    ),
                    affected_category=category,
                    evidence=[
                        (
                            f"Category spend: "
                            f"{_format_currency(amount, self.config.currency)}"
                        ),
                        (
                            f"Share of analyzed spend: "
                            f"{_format_percent(share)}"
                        ),
                    ],
                    assumptions=[
                        "The analyzed expense sample is representative.",
                    ],
                    risks=[
                        "Category concentration may be operationally necessary.",
                    ],
                    next_steps=[
                        "Rank vendors by spend.",
                        "Review contract terms.",
                        "Identify consolidation opportunities.",
                    ],
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Supplier optimization
    # ------------------------------------------------------------------

    def _supplier_recommendations(
        self,
        expenses: Sequence[
            ExpenseRecord
        ],
    ) -> List[
        CostRecommendation
    ]:

        recommendations = []

        supplier_totals: Dict[
            str,
            float,
        ] = {}

        supplier_categories: Dict[
            str,
            set,
        ] = {}

        for expense in expenses:

            if not expense.supplier:
                continue

            supplier = str(
                expense.supplier
            )

            supplier_totals[
                supplier
            ] = (
                supplier_totals.get(
                    supplier,
                    0.0,
                )
                + expense.amount
            )

            supplier_categories.setdefault(
                supplier,
                set(),
            ).add(
                expense.category
            )

        if not supplier_totals:
            return recommendations

        total_supplier_spend = sum(
            supplier_totals.values()
        )

        for supplier, amount in (
            supplier_totals.items()
        ):

            share = _percentage(
                amount,
                total_supplier_spend,
            )

            if (
                share
                < self.config.concentration_threshold_percent
            ):
                continue

            savings = (
                amount
                * self.config.supplier_savings_percent
                / 100
            )

            priority = (
                "high"
                if share
                >= self.config.high_concentration_percent
                else "medium"
            )

            risk = (
                "high"
                if priority == "high"
                else "medium"
            )

            recommendations.append(
                self._create_recommendation(
                    recommendation_type=(
                        "supplier_optimization"
                    ),
                    title=(
                        f"Negotiate high-spend "
                        f"supplier: {supplier}"
                    ),
                    description=(
                        f"{supplier} represents "
                        f"{_format_percent(share)} "
                        f"of analyzed supplier spend."
                    ),
                    action=(
                        "Renegotiate pricing, volume "
                        "discounts, payment terms, "
                        "service levels, or evaluate "
                        "competitive alternatives."
                    ),
                    priority=priority,
                    risk_level=risk,
                    confidence=0.86,
                    current_cost=amount,
                    potential_savings=savings,
                    savings_percent=(
                        self.config.supplier_savings_percent
                    ),
                    affected_supplier=supplier,
                    evidence=[
                        (
                            f"Supplier spend: "
                            f"{_format_currency(amount, self.config.currency)}"
                        ),
                        (
                            f"Supplier spend share: "
                            f"{_format_percent(share)}"
                        ),
                        (
                            "Categories served: "
                            + ", ".join(
                                sorted(
                                    supplier_categories.get(
                                        supplier,
                                        set(),
                                    )
                                )
                            )
                        ),
                    ],
                    assumptions=[
                        "Supplier spend is comparable across the analyzed period.",
                        "Alternative pricing or contract negotiation is feasible.",
                    ],
                    risks=[
                        "Changing suppliers can create operational risk.",
                        "Aggressive negotiation may affect supplier service quality.",
                    ],
                    next_steps=[
                        "Review supplier contract.",
                        "Benchmark current pricing.",
                        "Request volume-based pricing.",
                        "Evaluate alternative suppliers.",
                    ],
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Fixed costs
    # ------------------------------------------------------------------

    def _fixed_cost_recommendations(
        self,
        expenses: Sequence[
            ExpenseRecord
        ],
    ) -> List[
        CostRecommendation
    ]:

        fixed_expenses = [
            expense
            for expense in expenses
            if expense.expense_type == "fixed"
        ]

        if not fixed_expenses:
            return []

        total = sum(
            expense.amount
            for expense in fixed_expenses
        )

        if total <= 0:
            return []

        savings = (
            total
            * self.config.overhead_savings_percent
            / 100
        )

        return [
            self._create_recommendation(
                recommendation_type=(
                    "fixed_cost_reduction"
                ),
                title=(
                    "Review fixed-cost structure"
                ),
                description=(
                    f"Fixed costs total "
                    f"{_format_currency(total, self.config.currency)} "
                    "in the analyzed dataset."
                ),
                action=(
                    "Review recurring contracts, "
                    "subscriptions, facilities, "
                    "insurance, and administrative "
                    "overheads for reduction opportunities."
                ),
                priority="medium",
                risk_level="medium",
                confidence=0.78,
                current_cost=total,
                potential_savings=savings,
                savings_percent=(
                    self.config.overhead_savings_percent
                ),
                evidence=[
                    (
                        f"Fixed costs: "
                        f"{_format_currency(total, self.config.currency)}"
                    ),
                    (
                        f"Target reduction: "
                        f"{_format_percent(self.config.overhead_savings_percent)}"
                    ),
                ],
                assumptions=[
                    "Some fixed costs contain negotiable or avoidable components.",
                ],
                risks=[
                    "Fixed-cost cuts can affect long-term capability.",
                ],
                next_steps=[
                    "Classify fixed costs as essential or discretionary.",
                    "Review renewal dates.",
                    "Renegotiate recurring contracts.",
                ],
            )
        ]

    # ------------------------------------------------------------------
    # Variable costs
    # ------------------------------------------------------------------

    def _variable_cost_recommendations(
        self,
        expenses: Sequence[
            ExpenseRecord
        ],
    ) -> List[
        CostRecommendation
    ]:

        variable_expenses = [
            expense
            for expense in expenses
            if expense.expense_type == "variable"
        ]

        if not variable_expenses:
            return []

        total = sum(
            expense.amount
            for expense in variable_expenses
        )

        if total <= 0:
            return []

        savings = (
            total
            * self.config.variable_cost_savings_percent
            / 100
        )

        return [
            self._create_recommendation(
                recommendation_type=(
                    "variable_cost_optimization"
                ),
                title=(
                    "Optimize variable spending"
                ),
                description=(
                    f"Variable costs total "
                    f"{_format_currency(total, self.config.currency)}."
                ),
                action=(
                    "Analyze usage-driven costs, "
                    "procurement rates, waste, "
                    "volume discounts, and process "
                    "efficiency."
                ),
                priority="medium",
                risk_level="low",
                confidence=0.74,
                current_cost=total,
                potential_savings=savings,
                savings_percent=(
                    self.config.variable_cost_savings_percent
                ),
                evidence=[
                    (
                        f"Variable costs: "
                        f"{_format_currency(total, self.config.currency)}"
                    ),
                ],
                assumptions=[
                    "Variable costs contain operational efficiency opportunities.",
                ],
                risks=[
                    "Reducing variable costs too aggressively may reduce output quality.",
                ],
                next_steps=[
                    "Identify top variable-cost drivers.",
                    "Compare price versus volume changes.",
                    "Optimize purchasing frequency.",
                ],
            )
        ]

    # ------------------------------------------------------------------
    # Duplicate expense detection
    # ------------------------------------------------------------------

    def _duplicate_expense_recommendations(
        self,
        expenses: Sequence[
            ExpenseRecord
        ],
    ) -> List[
        CostRecommendation
    ]:

        recommendations = []

        for index, first in enumerate(
            expenses
        ):

            for second in expenses[
                index + 1:
            ]:

                if (
                    first.supplier
                    and second.supplier
                    and str(
                        first.supplier
                    ).lower()
                    != str(
                        second.supplier
                    ).lower()
                ):
                    continue

                if (
                    first.category.lower()
                    != second.category.lower()
                ):
                    continue

                amount_difference = abs(
                    first.amount
                    - second.amount
                )

                denominator = max(
                    abs(first.amount),
                    abs(second.amount),
                    1,
                )

                difference_percent = (
                    amount_difference
                    / denominator
                    * 100
                )

                if (
                    difference_percent
                    > self.config.duplicate_amount_tolerance_percent
                ):
                    continue

                if (
                    first.description
                    and second.description
                    and first.description.lower()
                    != second.description.lower()
                ):
                    continue

                combined_amount = (
                    first.amount
                    + second.amount
                )

                recommendations.append(
                    self._create_recommendation(
                        recommendation_type=(
                            "duplicate_expense"
                        ),
                        title=(
                            "Review potentially "
                            "duplicate expenses"
                        ),
                        description=(
                            f"Expenses {first.expense_id} "
                            f"and {second.expense_id} have "
                            "matching category, supplier, "
                            "and similar amounts."
                        ),
                        action=(
                            "Validate the two transactions "
                            "against invoices, purchase "
                            "orders, and approval records "
                            "before payment or closure."
                        ),
                        priority="high",
                        risk_level="high",
                        confidence=0.72,
                        current_cost=combined_amount,
                        potential_savings=min(
                            first.amount,
                            second.amount,
                        ),
                        savings_percent=_percentage(
                            min(
                                first.amount,
                                second.amount,
                            ),
                            combined_amount,
                        ),
                        affected_category=(
                            first.category
                        ),
                        affected_supplier=(
                            first.supplier
                        ),
                        evidence=[
                            (
                                f"Transaction 1: "
                                f"{first.expense_id}"
                            ),
                            (
                                f"Transaction 2: "
                                f"{second.expense_id}"
                            ),
                            (
                                f"Amount 1: "
                                f"{_format_currency(first.amount, self.config.currency)}"
                            ),
                            (
                                f"Amount 2: "
                                f"{_format_currency(second.amount, self.config.currency)}"
                            ),
                        ],
                        assumptions=[
                            "Matching attributes may indicate duplicate billing.",
                        ],
                        risks=[
                            "False positives are possible for recurring transactions.",
                        ],
                        next_steps=[
                            "Compare invoice numbers.",
                            "Check purchase-order references.",
                            "Verify approval history.",
                        ],
                        requires_human_review=True,
                    )
                )

        return recommendations

    # ------------------------------------------------------------------
    # Cost concentration
    # ------------------------------------------------------------------

    def _cost_concentration_recommendations(
        self,
        expenses: Sequence[
            ExpenseRecord
        ],
    ) -> List[
        CostRecommendation
    ]:

        if not expenses:
            return []

        total = sum(
            expense.amount
            for expense in expenses
        )

        if total <= 0:
            return []

        sorted_expenses = sorted(
            expenses,
            key=lambda item: item.amount,
            reverse=True,
        )

        top_count = max(
            1,
            min(
                5,
                len(sorted_expenses),
            ),
        )

        top_expenses = sorted_expenses[
            :top_count
        ]

        top_total = sum(
            expense.amount
            for expense in top_expenses
        )

        concentration = _percentage(
            top_total,
            total,
        )

        if concentration < 60:
            return []

        savings = (
            top_total
            * self.config.default_savings_percent
            / 100
        )

        evidence = [
            (
                f"Top {top_count} expenses account for "
                f"{_format_percent(concentration)} "
                "of analyzed spend."
            )
        ]

        for expense in top_expenses:
            evidence.append(
                (
                    f"{expense.expense_id}: "
                    f"{_format_currency(expense.amount, self.config.currency)} "
                    f"({expense.category})"
                )
            )

        return [
            self._create_recommendation(
                recommendation_type=(
                    "cost_concentration"
                ),
                title=(
                    "Review highly concentrated spending"
                ),
                description=(
                    "A small number of expense records "
                    "represent a large share of analyzed "
                    "spending."
                ),
                action=(
                    "Perform management review of the "
                    "largest cost commitments and determine "
                    "whether contracts, purchasing terms, "
                    "or spending levels can be optimized."
                ),
                priority="high",
                risk_level="high",
                confidence=0.84,
                current_cost=top_total,
                potential_savings=savings,
                savings_percent=(
                    self.config.default_savings_percent
                ),
                evidence=evidence,
                assumptions=[
                    "The expense dataset contains material spending activity.",
                ],
                risks=[
                    "Large expenses may represent necessary strategic investments.",
                ],
                next_steps=[
                    "Review each top expense individually.",
                    "Validate business justification.",
                    "Compare against approved contracts and budgets.",
                ],
                requires_human_review=True,
            )
        ]

    # ------------------------------------------------------------------
    # Margin recommendations
    # ------------------------------------------------------------------

    def _margin_recommendations(
        self,
        financial_data: Mapping[str, Any],
    ) -> List[
        CostRecommendation
    ]:

        revenue = _safe_float(
            _first(
                financial_data,
                [
                    "revenue",
                    "total_revenue",
                    "sales",
                ],
                0,
            )
        )

        gross_profit = _safe_float(
            _first(
                financial_data,
                [
                    "gross_profit",
                ],
                0,
            )
        )

        operating_profit = _safe_float(
            _first(
                financial_data,
                [
                    "operating_profit",
                    "operating_income",
                ],
                0,
            )
        )

        if revenue <= 0:
            return []

        if gross_profit:

            gross_margin = _percentage(
                gross_profit,
                revenue,
            )

        else:

            gross_margin = 0.0

        if operating_profit:

            operating_margin = _percentage(
                operating_profit,
                revenue,
            )

        else:

            operating_margin = 0.0

        recommendations = []

        if (
            operating_margin > 0
            and operating_margin < 10
        ):

            cost_base = max(
                revenue
                - operating_profit,
                0,
            )

            savings = (
                cost_base
                * self.config.default_savings_percent
                / 100
            )

            recommendations.append(
                self._create_recommendation(
                    recommendation_type=(
                        "margin_protection"
                    ),
                    title=(
                        "Protect operating margin "
                        "through cost control"
                    ),
                    description=(
                        f"Operating margin is "
                        f"{_format_percent(operating_margin)}, "
                        "leaving limited room for cost shocks."
                    ),
                    action=(
                        "Prioritize discretionary cost "
                        "reductions, supplier optimization, "
                        "and process efficiency before "
                        "committing to additional overhead."
                    ),
                    priority="high",
                    risk_level="high",
                    confidence=0.88,
                    current_cost=cost_base,
                    potential_savings=savings,
                    savings_percent=(
                        self.config.default_savings_percent
                    ),
                    evidence=[
                        (
                            f"Revenue: "
                            f"{_format_currency(revenue, self.config.currency)}"
                        ),
                        (
                            f"Gross margin: "
                            f"{_format_percent(gross_margin)}"
                        ),
                        (
                            f"Operating margin: "
                            f"{_format_percent(operating_margin)}"
                        ),
                    ],
                    assumptions=[
                        "Reported revenue and operating profit are comparable.",
                    ],
                    risks=[
                        "Excessive cost cutting may damage growth.",
                    ],
                    next_steps=[
                        "Rank costs by business value.",
                        "Protect revenue-generating activities.",
                        "Reduce low-value discretionary spending.",
                    ],
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Cash preservation
    # ------------------------------------------------------------------

    def _cash_preservation_recommendations(
        self,
        financial_data: Mapping[str, Any],
    ) -> List[
        CostRecommendation
    ]:

        cash = _safe_float(
            _first(
                financial_data,
                [
                    "cash",
                    "cash_balance",
                    "cash_and_equivalents",
                ],
                0,
            )
        )

        current_liabilities = _safe_float(
            _first(
                financial_data,
                [
                    "current_liabilities",
                ],
                0,
            )
        )

        if (
            cash <= 0
            or current_liabilities <= 0
        ):
            return []

        cash_coverage = _percentage(
            cash,
            current_liabilities,
        )

        if cash_coverage >= 50:
            return []

        potential_savings = (
            current_liabilities
            * 0.03
        )

        return [
            self._create_recommendation(
                recommendation_type=(
                    "cash_preservation"
                ),
                title=(
                    "Strengthen cash preservation"
                ),
                description=(
                    f"Cash covers only "
                    f"{_format_percent(cash_coverage)} "
                    "of current liabilities."
                ),
                action=(
                    "Defer non-essential spending, "
                    "tighten purchase approvals, "
                    "and prioritize expenditures "
                    "with near-term business value."
                ),
                priority="high",
                risk_level="high",
                confidence=0.81,
                current_cost=current_liabilities,
                potential_savings=potential_savings,
                savings_percent=_percentage(
                    potential_savings,
                    current_liabilities,
                ),
                evidence=[
                    (
                        f"Cash: "
                        f"{_format_currency(cash, self.config.currency)}"
                    ),
                    (
                        f"Current liabilities: "
                        f"{_format_currency(current_liabilities, self.config.currency)}"
                    ),
                    (
                        f"Cash coverage: "
                        f"{_format_percent(cash_coverage)}"
                    ),
                ],
                assumptions=[
                    "Current liabilities are due within the normal operating cycle.",
                ],
                risks=[
                    "Overly aggressive spending controls can delay necessary operations.",
                ],
                next_steps=[
                    "Review upcoming payment commitments.",
                    "Prioritize essential operating expenses.",
                    "Create a short-term cash preservation plan.",
                ],
                requires_human_review=True,
            )
        ]

    # ------------------------------------------------------------------
    # Recommendation construction
    # ------------------------------------------------------------------

    def _create_recommendation(
        self,
        *,
        recommendation_type: str,
        title: str,
        description: str,
        action: str,
        priority: str,
        risk_level: str,
        confidence: float,
        current_cost: float = 0.0,
        benchmark_cost: float = 0.0,
        potential_savings: float = 0.0,
        savings_percent: float = 0.0,
        affected_category: Optional[str] = None,
        affected_supplier: Optional[str] = None,
        affected_department: Optional[str] = None,
        evidence: Optional[
            Sequence[str]
        ] = None,
        assumptions: Optional[
            Sequence[str]
        ] = None,
        risks: Optional[
            Sequence[str]
        ] = None,
        next_steps: Optional[
            Sequence[str]
        ] = None,
        requires_human_review: bool = False,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> CostRecommendation:

        priority = _normalize_priority(
            priority
        )

        risk_level = _normalize_risk(
            risk_level
        )

        confidence = max(
            0.0,
            min(
                1.0,
                _safe_float(
                    confidence
                ),
            ),
        )

        potential_savings = max(
            0.0,
            _safe_float(
                potential_savings
            ),
        )

        # High-impact recommendations
        # should normally receive human review.

        if (
            self.config.require_human_review_for_high_impact
            and potential_savings
            >= self.config.high_impact_savings_threshold
        ):
            requires_human_review = True

        if priority in {
            "critical",
            "high",
        }:
            requires_human_review = True

        return CostRecommendation(
            recommendation_id=(
                "COST-"
                + uuid.uuid4().hex[:12].upper()
            ),
            recommendation_type=(
                recommendation_type
            ),
            title=title,
            description=description,
            action=action,
            priority=priority,
            risk_level=risk_level,
            confidence=_round(
                confidence,
                3,
            ),
            current_cost=_round(
                current_cost
            ),
            benchmark_cost=_round(
                benchmark_cost
            ),
            potential_savings=_round(
                potential_savings
            ),
            savings_percent=_round(
                savings_percent
            ),
            affected_category=(
                affected_category
            ),
            affected_supplier=(
                affected_supplier
            ),
            affected_department=(
                affected_department
            ),
            evidence=list(
                evidence or []
            ),
            assumptions=list(
                assumptions or []
            ),
            risks=list(
                risks or []
            ),
            next_steps=list(
                next_steps or []
            ),
            requires_human_review=(
                requires_human_review
            ),
            metadata=dict(
                metadata or {}
            ),
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_recommendations(
        self,
        recommendations: Sequence[
            CostRecommendation
        ],
    ) -> List[
        CostRecommendation
    ]:

        result = []

        for recommendation in recommendations:

            if (
                recommendation.confidence
                < self.config.minimum_confidence
            ):
                continue

            result.append(
                recommendation
            )

        return result

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_recommendations(
        self,
        recommendations: Sequence[
            CostRecommendation
        ],
    ) -> List[
        CostRecommendation
    ]:

        result = []

        seen = set()

        for recommendation in recommendations:

            key = (
                recommendation.recommendation_type,
                (
                    recommendation.affected_category
                    or ""
                ).lower(),
                (
                    recommendation.affected_supplier
                    or ""
                ).lower(),
                recommendation.title.lower(),
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(
                recommendation
            )

        return result

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def _recommendation_sort_key(
        self,
        recommendation: CostRecommendation,
    ):

        return (
            PRIORITY_LEVELS.get(
                recommendation.priority,
                99,
            ),
            -recommendation.potential_savings,
            -recommendation.confidence,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        recommendations: Sequence[
            CostRecommendation
        ],
        total_cost: float,
        total_savings: float,
    ) -> Dict[str, Any]:

        priority_counts: Dict[
            str,
            int,
        ] = {}

        type_counts: Dict[
            str,
            int,
        ] = {}

        for recommendation in recommendations:

            priority_counts[
                recommendation.priority
            ] = (
                priority_counts.get(
                    recommendation.priority,
                    0,
                )
                + 1
            )

            type_counts[
                recommendation.recommendation_type
            ] = (
                type_counts.get(
                    recommendation.recommendation_type,
                    0,
                )
                + 1
            )

        savings_percent = _percentage(
            total_savings,
            total_cost,
        )

        return {
            "message": (
                "Cost optimization recommendations "
                "generated successfully."
                if recommendations
                else
                "No actionable cost recommendations "
                "were identified."
            ),
            "recommendation_count": len(
                recommendations
            ),
            "total_analyzed_cost": _round(
                total_cost
            ),
            "estimated_savings": _round(
                total_savings
            ),
            "estimated_savings_percent": _round(
                savings_percent
            ),
            "priority_counts": priority_counts,
            "recommendation_type_counts": (
                type_counts
            ),
            "human_review_required": sum(
                item.requires_human_review
                for item in recommendations
            ),
        }

    # ------------------------------------------------------------------
    # ID
    # ------------------------------------------------------------------

    def _create_result_id(self) -> str:

        return (
            "COST-RESULT-"
            + uuid.uuid4().hex[:12].upper()
        )

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def save_json(
        self,
        result: CostRecommendationResult,
        path: Optional[
            str | Path
        ] = None,
    ) -> Path:

        output_path = (
            Path(path)
            if path
            else self.output_directory
            / "cost_recommendations.json"
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

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def to_markdown(
        self,
        result: CostRecommendationResult,
    ) -> str:

        lines = [
            "# FinCo AI Cost Recommendations",
            "",
            f"**Result ID:** {result.result_id}",
            f"**Generated:** {result.generated_at}",
            f"**Status:** {result.status}",
            "",
            "## Summary",
            "",
            (
                f"- Analyzed cost: "
                f"{_format_currency(result.total_current_cost, result.currency)}"
            ),
            (
                f"- Estimated savings: "
                f"{_format_currency(result.estimated_total_savings, result.currency)}"
            ),
            (
                f"- Estimated savings rate: "
                f"{_format_percent(result.average_savings_percent)}"
            ),
            (
                f"- Recommendations: "
                f"{len(result.recommendations)}"
            ),
            (
                f"- High/Critical: "
                f"{result.high_priority_count}"
            ),
            (
                f"- Human review required: "
                f"{result.human_review_count}"
            ),
            "",
        ]

        if not result.recommendations:

            lines.extend(
                [
                    "## Result",
                    "",
                    "No actionable recommendations were identified.",
                    "",
                ]
            )

        for index, recommendation in enumerate(
            result.recommendations,
            start=1,
        ):

            lines.extend(
                [
                    f"## {index}. {recommendation.title}",
                    "",
                    (
                        f"**Priority:** "
                        f"{recommendation.priority.upper()}"
                    ),
                    (
                        f"**Risk:** "
                        f"{recommendation.risk_level.upper()}"
                    ),
                    (
                        f"**Confidence:** "
                        f"{recommendation.confidence:.0%}"
                    ),
                    (
                        f"**Estimated savings:** "
                        f"{_format_currency(recommendation.potential_savings, result.currency)}"
                    ),
                    "",
                    recommendation.description,
                    "",
                    "### Recommended Action",
                    "",
                    recommendation.action,
                    "",
                ]
            )

            if recommendation.evidence:

                lines.append(
                    "### Evidence"
                )

                lines.append("")

                for evidence in (
                    recommendation.evidence
                ):
                    lines.append(
                        f"- {evidence}"
                    )

                lines.append("")

            if recommendation.assumptions:

                lines.append(
                    "### Assumptions"
                )

                lines.append("")

                for assumption in (
                    recommendation.assumptions
                ):
                    lines.append(
                        f"- {assumption}"
                    )

                lines.append("")

            if recommendation.risks:

                lines.append(
                    "### Risks"
                )

                lines.append("")

                for risk in recommendation.risks:
                    lines.append(
                        f"- {risk}"
                    )

                lines.append("")

            if recommendation.next_steps:

                lines.append(
                    "### Next Steps"
                )

                lines.append("")

                for step in (
                    recommendation.next_steps
                ):
                    lines.append(
                        f"1. {step}"
                    )

                lines.append("")

            if recommendation.requires_human_review:

                lines.append(
                    "> **Human Review Required:** "
                    "This recommendation should be reviewed "
                    "by an authorized financial decision-maker "
                    "before execution."
                )

                lines.append("")

        lines.extend(
            [
                "---",
                "",
                (
                    "Generated by **FinCo AI — "
                    "Financial Intelligence & Decision Copilot**."
                ),
            ]
        )

        return "\n".join(lines)

    def save_markdown(
        self,
        result: CostRecommendationResult,
        path: Optional[
            str | Path
        ] = None,
    ) -> Path:

        output_path = (
            Path(path)
            if path
            else self.output_directory
            / "cost_recommendations.md"
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

    # ------------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------------

    def to_html(
        self,
        result: CostRecommendationResult,
    ) -> str:

        cards = []

        for recommendation in (
            result.recommendations
        ):

            evidence_html = "".join(
                f"<li>{evidence}</li>"
                for evidence
                in recommendation.evidence
            )

            next_steps_html = "".join(
                f"<li>{step}</li>"
                for step
                in recommendation.next_steps
            )

            review_html = (
                """
                <div class="review">
                    HUMAN REVIEW REQUIRED
                </div>
                """
                if recommendation.requires_human_review
                else ""
            )

            cards.append(
                f"""
                <article class="card">

                    <div class="top-row">
                        <span class="priority">
                            {recommendation.priority.upper()}
                        </span>

                        <span class="risk">
                            {recommendation.risk_level.upper()}
                        </span>
                    </div>

                    <h2>
                        {recommendation.title}
                    </h2>

                    <p>
                        {recommendation.description}
                    </p>

                    <div class="stats">

                        <div>
                            <span>Confidence</span>
                            <strong>
                                {recommendation.confidence:.0%}
                            </strong>
                        </div>

                        <div>
                            <span>Potential Savings</span>
                            <strong>
                                {_format_currency(
                                    recommendation.potential_savings,
                                    result.currency
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>Savings Rate</span>
                            <strong>
                                {_format_percent(
                                    recommendation.savings_percent
                                )}
                            </strong>
                        </div>

                    </div>

                    <h3>Recommended Action</h3>

                    <p>
                        {recommendation.action}
                    </p>

                    <h3>Evidence</h3>

                    <ul>
                        {evidence_html}
                    </ul>

                    <h3>Next Steps</h3>

                    <ol>
                        {next_steps_html}
                    </ol>

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
FinCo AI Cost Recommendations
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        Inter,
        Arial,
        Helvetica,
        sans-serif;
    background: #f4f7fb;
    color: #172033;
}}

.container {{
    width: min(1180px, 94%);
    margin: 40px auto;
}}

.header {{
    background: white;
    border: 1px solid #e4e8ef;
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 24px;
}}

.header h1 {{
    margin: 0 0 10px;
    font-size: 32px;
}}

.muted {{
    color: #697386;
}}

.summary {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(190px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}}

.summary-card {{
    background: white;
    border: 1px solid #e4e8ef;
    border-radius: 16px;
    padding: 22px;
}}

.summary-card span {{
    display: block;
    color: #697386;
    font-size: 13px;
    margin-bottom: 8px;
}}

.summary-card strong {{
    font-size: 24px;
}}

.card {{
    background: white;
    border: 1px solid #e4e8ef;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 18px;
}}

.top-row {{
    display: flex;
    gap: 10px;
    margin-bottom: 14px;
}}

.priority,
.risk {{
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
}}

.priority {{
    background: #fff7ed;
}}

.risk {{
    background: #fef3f2;
}}

.card h2 {{
    margin: 0 0 12px;
}}

.card h3 {{
    margin-top: 24px;
}}

.card p {{
    line-height: 1.7;
}}

.stats {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin: 22px 0;
}}

.stats div {{
    background: #f8fafc;
    border-radius: 12px;
    padding: 16px;
}}

.stats span {{
    display: block;
    color: #697386;
    font-size: 12px;
    margin-bottom: 5px;
}}

.review {{
    margin-top: 22px;
    padding: 14px;
    border-radius: 12px;
    background: #fff7ed;
    font-weight: 800;
}}

.footer {{
    text-align: center;
    color: #697386;
    padding: 30px;
}}

</style>

</head>

<body>

<div class="container">

<header class="header">

<h1>
FinCo AI Cost Recommendations
</h1>

<p class="muted">
Explainable cost optimization opportunities
for financial decision support.
</p>

<p class="muted">
Result ID:
<strong>
{result.result_id}
</strong>
</p>

</header>

<section class="summary">

<div class="summary-card">
<span>Analyzed Cost</span>
<strong>
{_format_currency(
    result.total_current_cost,
    result.currency
)}
</strong>
</div>

<div class="summary-card">
<span>Estimated Savings</span>
<strong>
{_format_currency(
    result.estimated_total_savings,
    result.currency
)}
</strong>
</div>

<div class="summary-card">
<span>Savings Rate</span>
<strong>
{_format_percent(
    result.average_savings_percent
)}
</strong>
</div>

<div class="summary-card">
<span>Recommendations</span>
<strong>
{len(result.recommendations)}
</strong>
</div>

</section>

<section>

{''.join(cards)
 if cards
 else '<div class="card"><h2>No actionable recommendations</h2></div>'}

</section>

<footer class="footer">

FinCo AI — Financial Intelligence
& Decision Copilot

</footer>

</div>

</body>

</html>
"""

    def save_html(
        self,
        result: CostRecommendationResult,
        path: Optional[
            str | Path
        ] = None,
    ) -> Path:

        output_path = (
            Path(path)
            if path
            else self.output_directory
            / "cost_recommendations.html"
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

    # ------------------------------------------------------------------
    # Save all
    # ------------------------------------------------------------------

    def save_all(
        self,
        result: CostRecommendationResult,
        output_directory: Optional[
            str | Path
        ] = None,
    ) -> Dict[str, Path]:

        directory = (
            Path(output_directory)
            if output_directory
            else self.output_directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return {
            "json": self.save_json(
                result,
                directory
                / "cost_recommendations.json",
            ),
            "markdown": self.save_markdown(
                result,
                directory
                / "cost_recommendations.md",
            ),
            "html": self.save_html(
                result,
                directory
                / "cost_recommendations.html",
            ),
        }


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def generate_cost_recommendations(
    *,
    expenses: Optional[
        Sequence[Any]
    ] = None,
    budgets: Optional[
        Sequence[Any]
    ] = None,
    financial_data: Optional[
        Mapping[str, Any]
    ] = None,
    config: Optional[
        CostRecommendationConfig
    ] = None,
) -> CostRecommendationResult:
    """
    Convenience function for application/service integration.
    """

    engine = CostRecommendationEngine(
        config=config
    )

    return engine.generate(
        expenses=expenses,
        budgets=budgets,
        financial_data=financial_data,
    )


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def create_demo_expenses() -> List[Dict[str, Any]]:

    return [
        {
            "expense_id": "EXP-001",
            "category": "Cloud Infrastructure",
            "amount": 180000,
            "previous_amount": 140000,
            "supplier": "Cloud Provider A",
            "department": "Technology",
            "expense_type": "variable",
            "description": "Cloud infrastructure",
        },
        {
            "expense_id": "EXP-002",
            "category": "Cloud Infrastructure",
            "amount": 175000,
            "previous_amount": 135000,
            "supplier": "Cloud Provider A",
            "department": "Technology",
            "expense_type": "variable",
            "description": "Cloud infrastructure",
        },
        {
            "expense_id": "EXP-003",
            "category": "Marketing",
            "amount": 90000,
            "previous_amount": 70000,
            "supplier": "Marketing Agency",
            "department": "Marketing",
            "expense_type": "variable",
            "description": "Digital marketing",
        },
        {
            "expense_id": "EXP-004",
            "category": "Facilities",
            "amount": 65000,
            "supplier": "Facilities Vendor",
            "department": "Operations",
            "expense_type": "fixed",
            "description": "Office facilities",
        },
        {
            "expense_id": "EXP-005",
            "category": "Software",
            "amount": 45000,
            "supplier": "Enterprise Software",
            "department": "Technology",
            "expense_type": "fixed",
            "description": "Software subscriptions",
        },
        {
            "expense_id": "EXP-006",
            "category": "Travel",
            "amount": 32000,
            "previous_amount": 20000,
            "supplier": "Travel Vendor",
            "department": "Sales",
            "expense_type": "variable",
            "description": "Business travel",
        },
    ]


def create_demo_budgets() -> List[Dict[str, Any]]:

    return [
        {
            "category": "Cloud Infrastructure",
            "budget_amount": 250000,
            "actual_amount": 355000,
            "period": "FY2025",
        },
        {
            "category": "Marketing",
            "budget_amount": 80000,
            "actual_amount": 90000,
            "period": "FY2025",
        },
        {
            "category": "Facilities",
            "budget_amount": 70000,
            "actual_amount": 65000,
            "period": "FY2025",
        },
    ]


def create_demo_financial_data() -> Dict[str, Any]:

    return {
        "revenue": 12500000,
        "gross_profit": 5300000,
        "operating_profit": 850000,
        "cash": 1100000,
        "current_liabilities": 3200000,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def create_demo_result() -> CostRecommendationResult:

    config = CostRecommendationConfig(
        currency="USD",
        output_directory="reports/recommendations",
    )

    engine = CostRecommendationEngine(
        config=config
    )

    return engine.generate(
        expenses=create_demo_expenses(),
        budgets=create_demo_budgets(),
        financial_data=create_demo_financial_data(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "FinCo AI cost recommendation engine"
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
            "md",
            "html",
            "all",
        ],
        default="all",
        help="Output format.",
    )

    return parser


def main() -> int:

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    parser = build_argument_parser()

    args = parser.parse_args()

    config = CostRecommendationConfig(
        output_directory=args.output,
    )

    engine = CostRecommendationEngine(
        config=config
    )

    result = engine.generate(
        expenses=create_demo_expenses(),
        budgets=create_demo_budgets(),
        financial_data=create_demo_financial_data(),
    )

    output_directory = Path(
        args.output
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.format == "all":

        paths = engine.save_all(
            result,
            output_directory,
        )

        for name, path in paths.items():
            print(
                f"[OK] {name}: {path}"
            )

    elif args.format == "json":

        path = engine.save_json(
            result,
            output_directory
            / "cost_recommendations.json",
        )

        print(
            f"[OK] JSON: {path}"
        )

    elif args.format in {
        "markdown",
        "md",
    }:

        path = engine.save_markdown(
            result,
            output_directory
            / "cost_recommendations.md",
        )

        print(
            f"[OK] Markdown: {path}"
        )

    elif args.format == "html":

        path = engine.save_html(
            result,
            output_directory
            / "cost_recommendations.html",
        )

        print(
            f"[OK] HTML: {path}"
        )

    print()
    print(
        "Result ID:",
        result.result_id,
    )

    print(
        "Status:",
        result.status,
    )

    print(
        "Recommendations:",
        len(result.recommendations),
    )

    print(
        "Estimated savings:",
        _format_currency(
            result.estimated_total_savings,
            result.currency,
        ),
    )

    print(
        "Human review:",
        result.human_review_count,
    )

    return 0


# ---------------------------------------------------------------------------
# Module execution
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    raise SystemExit(
        main()
    )