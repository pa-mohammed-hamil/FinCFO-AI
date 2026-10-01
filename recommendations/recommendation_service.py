"""
FinCo AI - Recommendation Service
=================================

Central orchestration service for financial, risk, and cost recommendations.

Architecture
------------

Financial Data
      |
      +--------------------+
      |                    |
      v                    v
Financial Engine       Risk Engine
      |                    |
      +---------+----------+
                |
                v
          Cost Engine
                |
                v
     Recommendation Service
                |
       +--------+--------+
       |        |        |
       v        v        v
    Filter   Rank/Dedupe  Aggregate
       |        |        |
       +--------+--------+
                |
                v
       Human Review / API
                |
                v
          Audit / Reports

Design goals
------------
- Explainable recommendations
- Risk-aware prioritization
- Confidence filtering
- Duplicate removal
- Human-review routing
- Financial impact aggregation
- JSON / Markdown / HTML / Text export
- Works with dictionaries or dataclass/model objects
- Minimal dependencies
- Safe for API/service integration
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

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
    "none": 0,
}

RECOMMENDATION_SOURCES = {
    "financial",
    "risk",
    "cost",
    "combined",
}

DEFAULT_CURRENCY = "USD"

STATUS_READY = "ready"
STATUS_REVIEW_REQUIRED = "review_required"
STATUS_NO_ACTION = "no_action"

RECOMMENDATION_TYPES = {
    "financial_performance",
    "revenue_growth",
    "margin_improvement",
    "liquidity",
    "working_capital",
    "cash_flow",
    "debt_management",
    "risk_mitigation",
    "fraud_risk",
    "forecast_risk",
    "concentration_risk",
    "budget_risk",
    "cost_optimization",
    "supplier_optimization",
    "expense_control",
    "cash_preservation",
    "governance",
    "combined",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    """Return ISO-8601 UTC timestamp."""
    return _utc_now().isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_divide(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """Safely divide two numeric values."""
    num = _safe_float(numerator)
    den = _safe_float(denominator)

    if abs(den) < 1e-12:
        return default

    return num / den


def _percentage(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """Return percentage."""
    return _safe_divide(numerator, denominator, default) * 100.0


def _round(value: Any, digits: int = 2) -> float:
    """Round safely."""
    return round(_safe_float(value), digits)


def _normalize_text(value: Any) -> str:
    """Normalize arbitrary text."""
    if value is None:
        return ""

    text = str(value).strip()

    text = re.sub(r"\s+", " ", text)

    return text


def _normalize_priority(value: Any) -> str:
    """Normalize priority."""
    value = _normalize_text(value).lower()

    if value in PRIORITY_LEVELS:
        return value

    aliases = {
        "urgent": "critical",
        "p0": "critical",
        "p1": "high",
        "p2": "medium",
        "p3": "low",
        "normal": "medium",
    }

    return aliases.get(value, "medium")


def _normalize_risk(value: Any) -> str:
    """Normalize risk."""
    value = _normalize_text(value).lower()

    if value in RISK_LEVELS:
        return value

    aliases = {
        "severe": "critical",
        "urgent": "critical",
        "elevated": "high",
        "moderate": "medium",
        "minimal": "low",
    }

    return aliases.get(value, "medium")


def _normalize_source(value: Any) -> str:
    """Normalize recommendation source."""
    value = _normalize_text(value).lower()

    if value in RECOMMENDATION_SOURCES:
        return value

    return "combined"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Read a value from either a dictionary or an object.

    Supports:
    - dict
    - dataclass
    - Pydantic-like objects
    - regular Python objects
    """
    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(key, default)

    try:
        return getattr(obj, key)
    except AttributeError:
        return default


def _first(
    obj: Any,
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first non-empty value from a list of keys."""
    for key in keys:
        value = _get(obj, key, None)

        if value is not None and value != "":
            return value

    return default


def _json_safe(value: Any) -> Any:
    """
    Convert arbitrary objects into JSON-safe structures.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None

        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))

    return str(value)


def _slug(value: str) -> str:
    """Create a safe identifier fragment."""
    value = _normalize_text(value).lower()

    value = re.sub(r"[^a-z0-9]+", "-", value)

    return value.strip("-")


def _format_currency(
    value: Any,
    currency: str = DEFAULT_CURRENCY,
) -> str:
    """Format monetary values."""
    amount = _safe_float(value)

    return f"{currency} {amount:,.2f}"


def _format_percent(value: Any) -> str:
    """Format percentage."""
    return f"{_safe_float(value):.2f}%"


def _priority_score(priority: str) -> int:
    return PRIORITY_LEVELS.get(
        _normalize_priority(priority),
        PRIORITY_LEVELS["medium"],
    )


def _risk_score(risk: str) -> int:
    return RISK_LEVELS.get(
        _normalize_risk(risk),
        RISK_LEVELS["medium"],
    )


def _now_date() -> str:
    return _utc_now().date().isoformat()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RecommendationServiceConfig:
    """
    Configuration for the recommendation orchestration service.
    """

    currency: str = DEFAULT_CURRENCY

    # Filtering
    minimum_confidence: float = 0.55

    include_low_confidence: bool = False

    # Output limits
    max_recommendations: int = 30

    # Human review
    human_review_for_critical: bool = True
    human_review_for_high_risk: bool = True

    # Financial impact
    high_impact_threshold: float = 50_000.0
    critical_impact_threshold: float = 250_000.0

    # Prioritization
    risk_weight: float = 0.35
    impact_weight: float = 0.30
    confidence_weight: float = 0.20
    priority_weight: float = 0.15

    # Deduplication
    duplicate_title_similarity: float = 0.90

    # Output
    output_directory: str = "reports/recommendations"

    # Safety
    require_evidence_for_high_risk: bool = True

    def validate(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be between 0 and 1"
            )

        if self.max_recommendations < 1:
            raise ValueError(
                "max_recommendations must be >= 1"
            )

        if self.high_impact_threshold < 0:
            raise ValueError(
                "high_impact_threshold must be >= 0"
            )

        if self.critical_impact_threshold < 0:
            raise ValueError(
                "critical_impact_threshold must be >= 0"
            )

        total_weight = (
            self.risk_weight
            + self.impact_weight
            + self.confidence_weight
            + self.priority_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "Recommendation priority weights must sum to > 0"
            )


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class UnifiedRecommendation:
    """
    Canonical recommendation representation.

    This model allows outputs from different engines to be consumed
    consistently by the service.
    """

    recommendation_id: str

    source: str

    type: str

    title: str

    description: str

    action: str

    priority: str = "medium"

    risk: str = "medium"

    confidence: float = 0.75

    current_value: float = 0.0

    benchmark_value: float = 0.0

    potential_impact: float = 0.0

    potential_savings: float = 0.0

    impact_percent: float = 0.0

    affected_area: str = ""

    affected_category: str = ""

    affected_department: str = ""

    evidence: List[Any] = field(default_factory=list)

    assumptions: List[str] = field(default_factory=list)

    risks: List[str] = field(default_factory=list)

    next_steps: List[str] = field(default_factory=list)

    human_review_required: bool = False

    generated_at: str = field(default_factory=_timestamp)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = _normalize_source(self.source)
        self.type = _normalize_text(self.type) or "combined"
        self.title = _normalize_text(self.title)
        self.description = _normalize_text(self.description)
        self.action = _normalize_text(self.action)

        self.priority = _normalize_priority(self.priority)
        self.risk = _normalize_risk(self.risk)

        self.confidence = max(
            0.0,
            min(1.0, _safe_float(self.confidence)),
        )

        self.current_value = _safe_float(self.current_value)
        self.benchmark_value = _safe_float(self.benchmark_value)
        self.potential_impact = _safe_float(self.potential_impact)
        self.potential_savings = _safe_float(self.potential_savings)
        self.impact_percent = _safe_float(self.impact_percent)

    @property
    def priority_score(self) -> int:
        return _priority_score(self.priority)

    @property
    def risk_score(self) -> int:
        return _risk_score(self.risk)

    @property
    def impact(self) -> float:
        """Return primary financial impact."""
        if abs(self.potential_impact) > 0:
            return abs(self.potential_impact)

        return abs(self.potential_savings)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize recommendation."""
        return _json_safe(asdict(self))


@dataclass
class RecommendationServiceResult:
    """
    Aggregate result returned by RecommendationService.
    """

    result_id: str

    generated_at: str

    status: str

    currency: str

    recommendations: List[UnifiedRecommendation]

    total_recommendations: int

    critical_recommendations: int

    high_priority_recommendations: int

    medium_priority_recommendations: int

    low_priority_recommendations: int

    high_risk_recommendations: int

    human_review_count: int

    estimated_savings: float

    estimated_financial_impact: float

    average_confidence: float

    overall_risk: str

    executive_summary: str

    source_counts: Dict[str, int]

    type_counts: Dict[str, int]

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete result."""
        return {
            "result_id": self.result_id,
            "generated_at": self.generated_at,
            "status": self.status,
            "currency": self.currency,
            "total_recommendations": self.total_recommendations,
            "critical_recommendations": self.critical_recommendations,
            "high_priority_recommendations": self.high_priority_recommendations,
            "medium_priority_recommendations": self.medium_priority_recommendations,
            "low_priority_recommendations": self.low_priority_recommendations,
            "high_risk_recommendations": self.high_risk_recommendations,
            "human_review_count": self.human_review_count,
            "estimated_savings": self.estimated_savings,
            "estimated_financial_impact": self.estimated_financial_impact,
            "average_confidence": self.average_confidence,
            "overall_risk": self.overall_risk,
            "executive_summary": self.executive_summary,
            "source_counts": self.source_counts,
            "type_counts": self.type_counts,
            "recommendations": [
                recommendation.to_dict()
                for recommendation in self.recommendations
            ],
            "metadata": _json_safe(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
        )


# ============================================================================
# RECOMMENDATION SERVICE
# ============================================================================

class RecommendationService:
    """
    Central recommendation orchestration service.

    Responsibilities
    ----------------
    1. Consume financial/risk/cost recommendation engines.
    2. Normalize heterogeneous recommendation structures.
    3. Apply safety and confidence filters.
    4. Calculate financial impact.
    5. Deduplicate overlapping recommendations.
    6. Prioritize recommendations.
    7. Determine human-review requirements.
    8. Produce an executive-level summary.
    9. Export recommendations for API/reporting layers.
    """

    def __init__(
        self,
        config: Optional[RecommendationServiceConfig] = None,
    ) -> None:
        self.config = config or RecommendationServiceConfig()

        self.config.validate()

        logger.info(
            "RecommendationService initialized"
        )

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------

    def generate(
        self,
        financial_recommendations: Optional[
            Iterable[Any]
        ] = None,
        risk_recommendations: Optional[
            Iterable[Any]
        ] = None,
        cost_recommendations: Optional[
            Iterable[Any]
        ] = None,
        financial_data: Optional[Any] = None,
        risk_data: Optional[Any] = None,
        cost_data: Optional[Any] = None,
        alerts: Optional[Iterable[Any]] = None,
        fraud_findings: Optional[Iterable[Any]] = None,
        forecast_data: Optional[Any] = None,
    ) -> RecommendationServiceResult:
        """
        Generate unified recommendations.

        Parameters
        ----------
        financial_recommendations:
            Recommendations generated by financial_recommendations.py.

        risk_recommendations:
            Recommendations generated by risk_recommendations.py.

        cost_recommendations:
            Recommendations generated by cost_recommendations.py.

        financial_data:
            Optional financial context.

        risk_data:
            Optional risk context.

        cost_data:
            Optional cost context.

        alerts:
            Optional alert records.

        fraud_findings:
            Optional fraud findings.

        forecast_data:
            Optional forecast information.

        Returns
        -------
        RecommendationServiceResult
        """

        raw_recommendations: List[Any] = []

        raw_recommendations.extend(
            self._extract_recommendations(
                financial_recommendations
            )
        )

        raw_recommendations.extend(
            self._extract_recommendations(
                risk_recommendations
            )
        )

        raw_recommendations.extend(
            self._extract_recommendations(
                cost_recommendations
            )
        )

        normalized = [
            self._normalize_recommendation(
                recommendation
            )
            for recommendation in raw_recommendations
        ]

        normalized = [
            recommendation
            for recommendation in normalized
            if recommendation is not None
        ]

        # Optional context enrichment.
        self._enrich_from_context(
            normalized,
            financial_data=financial_data,
            risk_data=risk_data,
            cost_data=cost_data,
            alerts=alerts,
            fraud_findings=fraud_findings,
            forecast_data=forecast_data,
        )

        # Apply safety rules.
        filtered = self._filter_recommendations(
            normalized
        )

        # Deduplicate.
        deduplicated = self._deduplicate_recommendations(
            filtered
        )

        # Apply ranking.
        ranked = sorted(
            deduplicated,
            key=self._recommendation_sort_key,
            reverse=True,
        )

        # Limit result size.
        ranked = ranked[
            : self.config.max_recommendations
        ]

        # Human-review routing.
        for recommendation in ranked:
            recommendation.human_review_required = (
                self._requires_human_review(
                    recommendation
                )
            )

        return self._build_result(ranked)

    # ---------------------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------------------

    def _extract_recommendations(
        self,
        source: Optional[Iterable[Any]],
    ) -> List[Any]:
        """
        Extract recommendation objects from:

        - list
        - tuple
        - result object with recommendations
        - single recommendation
        """
        if source is None:
            return []

        if hasattr(source, "recommendations"):
            try:
                recommendations = getattr(
                    source,
                    "recommendations",
                )

                if recommendations is None:
                    return []

                return list(recommendations)
            except Exception:
                return []

        if isinstance(source, (list, tuple, set)):
            return list(source)

        # Avoid treating strings as collections.
        if isinstance(source, str):
            return []

        try:
            return list(source)
        except TypeError:
            return [source]

    def _normalize_recommendation(
        self,
        recommendation: Any,
    ) -> Optional[UnifiedRecommendation]:
        """Convert an arbitrary recommendation into canonical form."""

        if recommendation is None:
            return None

        # Existing canonical object.
        if isinstance(
            recommendation,
            UnifiedRecommendation,
        ):
            return recommendation

        recommendation_id = _first(
            recommendation,
            [
                "recommendation_id",
                "id",
                "recommendationId",
            ],
            None,
        )

        source = _first(
            recommendation,
            [
                "source",
                "engine",
                "category_source",
            ],
            "combined",
        )

        rec_type = _first(
            recommendation,
            [
                "type",
                "recommendation_type",
                "category",
            ],
            "combined",
        )

        title = _first(
            recommendation,
            [
                "title",
                "name",
                "recommendation",
            ],
            "Financial recommendation",
        )

        description = _first(
            recommendation,
            [
                "description",
                "reason",
                "rationale",
            ],
            "",
        )

        action = _first(
            recommendation,
            [
                "action",
                "recommended_action",
                "recommendation_action",
            ],
            "",
        )

        priority = _first(
            recommendation,
            [
                "priority",
                "severity",
            ],
            "medium",
        )

        risk = _first(
            recommendation,
            [
                "risk",
                "risk_level",
            ],
            "medium",
        )

        confidence = _first(
            recommendation,
            [
                "confidence",
                "score",
                "recommendation_confidence",
            ],
            0.75,
        )

        current_value = _first(
            recommendation,
            [
                "current_value",
                "current_cost",
                "actual_value",
            ],
            0.0,
        )

        benchmark_value = _first(
            recommendation,
            [
                "benchmark_value",
                "benchmark_cost",
                "target_value",
            ],
            0.0,
        )

        potential_savings = _first(
            recommendation,
            [
                "potential_savings",
                "estimated_savings",
                "savings",
            ],
            0.0,
        )

        potential_impact = _first(
            recommendation,
            [
                "potential_impact",
                "financial_impact",
                "estimated_impact",
                "impact",
            ],
            potential_savings,
        )

        impact_percent = _first(
            recommendation,
            [
                "impact_percent",
                "savings_percent",
                "estimated_savings_percent",
            ],
            0.0,
        )

        affected_area = _first(
            recommendation,
            [
                "affected_area",
                "area",
                "scope",
            ],
            "",
        )

        affected_category = _first(
            recommendation,
            [
                "affected_category",
                "category",
            ],
            "",
        )

        affected_department = _first(
            recommendation,
            [
                "affected_department",
                "department",
            ],
            "",
        )

        evidence = _first(
            recommendation,
            [
                "evidence",
                "supporting_evidence",
                "signals",
            ],
            [],
        )

        assumptions = _first(
            recommendation,
            [
                "assumptions",
            ],
            [],
        )

        risks = _first(
            recommendation,
            [
                "risks",
                "risk_factors",
            ],
            [],
        )

        next_steps = _first(
            recommendation,
            [
                "next_steps",
                "actions",
                "implementation_steps",
            ],
            [],
        )

        metadata = _first(
            recommendation,
            [
                "metadata",
                "meta",
            ],
            {},
        )

        if not isinstance(evidence, list):
            evidence = [evidence] if evidence else []

        if not isinstance(assumptions, list):
            assumptions = (
                [assumptions]
                if assumptions
                else []
            )

        if not isinstance(risks, list):
            risks = [risks] if risks else []

        if not isinstance(next_steps, list):
            next_steps = (
                [next_steps]
                if next_steps
                else []
            )

        if not isinstance(metadata, dict):
            metadata = {
                "source_metadata": _json_safe(metadata)
            }

        if not recommendation_id:
            recommendation_id = self._create_recommendation_id(
                title=title,
                source=source,
                rec_type=rec_type,
            )

        return UnifiedRecommendation(
            recommendation_id=str(
                recommendation_id
            ),
            source=_normalize_source(source),
            type=_normalize_text(rec_type),
            title=_normalize_text(title),
            description=_normalize_text(description),
            action=_normalize_text(action),
            priority=_normalize_priority(priority),
            risk=_normalize_risk(risk),
            confidence=_safe_float(confidence, 0.75),
            current_value=_safe_float(current_value),
            benchmark_value=_safe_float(benchmark_value),
            potential_impact=_safe_float(potential_impact),
            potential_savings=_safe_float(potential_savings),
            impact_percent=_safe_float(impact_percent),
            affected_area=_normalize_text(
                affected_area
            ),
            affected_category=_normalize_text(
                affected_category
            ),
            affected_department=_normalize_text(
                affected_department
            ),
            evidence=evidence,
            assumptions=[
                _normalize_text(item)
                for item in assumptions
                if item is not None
            ],
            risks=[
                _normalize_text(item)
                for item in risks
                if item is not None
            ],
            next_steps=[
                _normalize_text(item)
                for item in next_steps
                if item is not None
            ],
            metadata=metadata,
        )

    # ---------------------------------------------------------------------
    # CONTEXT ENRICHMENT
    # ---------------------------------------------------------------------

    def _enrich_from_context(
        self,
        recommendations: List[UnifiedRecommendation],
        financial_data: Optional[Any],
        risk_data: Optional[Any],
        cost_data: Optional[Any],
        alerts: Optional[Iterable[Any]],
        fraud_findings: Optional[Iterable[Any]],
        forecast_data: Optional[Any],
    ) -> None:
        """
        Enrich recommendations with useful contextual information.

        This method deliberately does not create new recommendations.
        Recommendation generation remains the responsibility of the
        specialized engines.
        """

        alert_list = (
            list(alerts)
            if alerts is not None
            else []
        )

        fraud_list = (
            list(fraud_findings)
            if fraud_findings is not None
            else []
        )

        for recommendation in recommendations:
            recommendation.metadata.setdefault(
                "service_enriched_at",
                _timestamp(),
            )

            recommendation.metadata.setdefault(
                "has_financial_context",
                financial_data is not None,
            )

            recommendation.metadata.setdefault(
                "has_risk_context",
                risk_data is not None,
            )

            recommendation.metadata.setdefault(
                "has_cost_context",
                cost_data is not None,
            )

            recommendation.metadata.setdefault(
                "alert_count",
                len(alert_list),
            )

            recommendation.metadata.setdefault(
                "fraud_finding_count",
                len(fraud_list),
            )

            if forecast_data is not None:
                recommendation.metadata.setdefault(
                    "forecast_context_available",
                    True,
                )

            # Add evidence count.
            recommendation.metadata.setdefault(
                "evidence_count",
                len(recommendation.evidence),
            )

            # Calculate impact percent if missing.
            if (
                abs(recommendation.impact_percent)
                < 1e-12
                and abs(recommendation.current_value)
                > 1e-12
            ):
                recommendation.impact_percent = _percentage(
                    recommendation.potential_impact,
                    recommendation.current_value,
                )

    # ---------------------------------------------------------------------
    # FILTERING
    # ---------------------------------------------------------------------

    def _filter_recommendations(
        self,
        recommendations: List[UnifiedRecommendation],
    ) -> List[UnifiedRecommendation]:
        """Apply confidence and evidence safety filters."""

        filtered: List[UnifiedRecommendation] = []

        for recommendation in recommendations:

            # Basic validity.
            if not recommendation.title:
                continue

            if not recommendation.action:
                # A recommendation without an action is not operationally
                # useful, so provide a safe fallback.
                recommendation.action = (
                    "Review the identified issue and define "
                    "an owner, target, and implementation timeline."
                )

            # Confidence filtering.
            if (
                not self.config.include_low_confidence
                and recommendation.confidence
                < self.config.minimum_confidence
            ):
                continue

            # High-risk evidence safety.
            high_risk = recommendation.risk in {
                "high",
                "critical",
            }

            if (
                high_risk
                and self.config.require_evidence_for_high_risk
                and not recommendation.evidence
            ):
                recommendation.metadata[
                    "evidence_warning"
                ] = (
                    "High-risk recommendation has no explicit "
                    "supporting evidence."
                )

                recommendation.human_review_required = True

            filtered.append(recommendation)

        return filtered

    # ---------------------------------------------------------------------
    # DEDUPLICATION
    # ---------------------------------------------------------------------

    def _deduplicate_recommendations(
        self,
        recommendations: List[UnifiedRecommendation],
    ) -> List[UnifiedRecommendation]:
        """
        Remove duplicate or near-duplicate recommendations.

        If two recommendations overlap, the stronger one is retained.
        """

        if not recommendations:
            return []

        # First use normalized fingerprints.
        exact: Dict[str, UnifiedRecommendation] = {}

        for recommendation in recommendations:
            fingerprint = self._recommendation_fingerprint(
                recommendation
            )

            existing = exact.get(fingerprint)

            if existing is None:
                exact[fingerprint] = recommendation
            else:
                exact[fingerprint] = self._merge_recommendations(
                    existing,
                    recommendation,
                )

        unique = list(exact.values())

        # Second pass: title-level similarity.
        result: List[UnifiedRecommendation] = []

        for recommendation in sorted(
            unique,
            key=self._recommendation_sort_key,
            reverse=True,
        ):
            duplicate_index: Optional[int] = None

            for index, existing in enumerate(result):
                similarity = self._title_similarity(
                    recommendation.title,
                    existing.title,
                )

                if (
                    similarity
                    >= self.config.duplicate_title_similarity
                ):
                    duplicate_index = index
                    break

            if duplicate_index is None:
                result.append(recommendation)
            else:
                result[duplicate_index] = (
                    self._merge_recommendations(
                        result[duplicate_index],
                        recommendation,
                    )
                )

        return result

    def _recommendation_fingerprint(
        self,
        recommendation: UnifiedRecommendation,
    ) -> str:
        """Create a deterministic recommendation fingerprint."""

        parts = [
            _slug(recommendation.source),
            _slug(recommendation.type),
            _slug(recommendation.title),
            _slug(recommendation.affected_area),
            _slug(recommendation.affected_category),
            _slug(recommendation.affected_department),
        ]

        return "|".join(parts)

    def _title_tokens(
        self,
        title: str,
    ) -> set[str]:
        """Tokenize a title for simple similarity scoring."""
        words = re.findall(
            r"[a-zA-Z0-9]+",
            title.lower(),
        )

        stop_words = {
            "the",
            "a",
            "an",
            "to",
            "for",
            "and",
            "of",
            "in",
            "on",
            "with",
            "risk",
            "recommendation",
        }

        return {
            word
            for word in words
            if word not in stop_words
        }

    def _title_similarity(
        self,
        first: str,
        second: str,
    ) -> float:
        """Calculate Jaccard title similarity."""
        first_tokens = self._title_tokens(first)
        second_tokens = self._title_tokens(second)

        if not first_tokens or not second_tokens:
            return 0.0

        intersection = len(
            first_tokens & second_tokens
        )

        union = len(
            first_tokens | second_tokens
        )

        return intersection / union if union else 0.0

    def _merge_recommendations(
        self,
        first: UnifiedRecommendation,
        second: UnifiedRecommendation,
    ) -> UnifiedRecommendation:
        """
        Merge two overlapping recommendations.

        The stronger recommendation becomes the primary object.
        """
        if self._recommendation_sort_key(second) > (
            self._recommendation_sort_key(first)
        ):
            primary = second
            secondary = first
        else:
            primary = first
            secondary = second

        primary.confidence = max(
            primary.confidence,
            secondary.confidence,
        )

        primary.potential_savings = max(
            primary.potential_savings,
            secondary.potential_savings,
        )

        primary.potential_impact = max(
            primary.potential_impact,
            secondary.potential_impact,
        )

        primary.evidence = self._merge_unique_values(
            primary.evidence,
            secondary.evidence,
        )

        primary.assumptions = self._merge_unique_values(
            primary.assumptions,
            secondary.assumptions,
        )

        primary.risks = self._merge_unique_values(
            primary.risks,
            secondary.risks,
        )

        primary.next_steps = self._merge_unique_values(
            primary.next_steps,
            secondary.next_steps,
        )

        primary.metadata.setdefault(
            "merged_sources",
            [],
        )

        merged_sources = primary.metadata[
            "merged_sources"
        ]

        if primary.source not in merged_sources:
            merged_sources.append(primary.source)

        if secondary.source not in merged_sources:
            merged_sources.append(secondary.source)

        primary.metadata["merged"] = True

        if len(merged_sources) > 1:
            primary.source = "combined"

        return primary

    @staticmethod
    def _merge_unique_values(
        first: List[Any],
        second: List[Any],
    ) -> List[Any]:
        """Merge lists while preserving order."""
        result = list(first)

        serialized = {
            json.dumps(
                _json_safe(item),
                sort_keys=True,
            )
            for item in result
        }

        for item in second:
            marker = json.dumps(
                _json_safe(item),
                sort_keys=True,
            )

            if marker not in serialized:
                result.append(item)
                serialized.add(marker)

        return result

    # ---------------------------------------------------------------------
    # PRIORITIZATION
    # ---------------------------------------------------------------------

    def _recommendation_sort_key(
        self,
        recommendation: UnifiedRecommendation,
    ) -> float:
        """
        Calculate weighted priority score.

        Score components:
        - risk
        - financial impact
        - confidence
        - priority
        """

        risk_component = (
            recommendation.risk_score
            / max(RISK_LEVELS.values())
        )

        impact = recommendation.impact

        if self.config.critical_impact_threshold > 0:
            impact_component = min(
                1.0,
                impact
                / self.config.critical_impact_threshold,
            )
        else:
            impact_component = 0.0

        confidence_component = (
            recommendation.confidence
        )

        priority_component = (
            recommendation.priority_score
            / max(PRIORITY_LEVELS.values())
        )

        weighted_score = (
            risk_component * self.config.risk_weight
            + impact_component * self.config.impact_weight
            + confidence_component
            * self.config.confidence_weight
            + priority_component
            * self.config.priority_weight
        )

        return weighted_score

    # ---------------------------------------------------------------------
    # HUMAN REVIEW
    # ---------------------------------------------------------------------

    def _requires_human_review(
        self,
        recommendation: UnifiedRecommendation,
    ) -> bool:
        """
        Determine whether recommendation requires human review.
        """

        if (
            self.config.human_review_for_critical
            and (
                recommendation.priority == "critical"
                or recommendation.risk == "critical"
            )
        ):
            return True

        if (
            self.config.human_review_for_high_risk
            and recommendation.risk == "high"
        ):
            return True

        if (
            recommendation.impact
            >= self.config.high_impact_threshold
        ):
            return True

        if (
            recommendation.confidence
            < self.config.minimum_confidence
        ):
            return True

        if recommendation.metadata.get(
            "evidence_warning"
        ):
            return True

        return recommendation.human_review_required

    # ---------------------------------------------------------------------
    # RESULT BUILDING
    # ---------------------------------------------------------------------

    def _build_result(
        self,
        recommendations: List[UnifiedRecommendation],
    ) -> RecommendationServiceResult:
        """Build aggregate service result."""

        total = len(recommendations)

        critical = sum(
            1
            for recommendation in recommendations
            if recommendation.priority == "critical"
        )

        high = sum(
            1
            for recommendation in recommendations
            if recommendation.priority == "high"
        )

        medium = sum(
            1
            for recommendation in recommendations
            if recommendation.priority == "medium"
        )

        low = sum(
            1
            for recommendation in recommendations
            if recommendation.priority == "low"
        )

        high_risk = sum(
            1
            for recommendation in recommendations
            if recommendation.risk in {
                "high",
                "critical",
            }
        )

        human_review_count = sum(
            1
            for recommendation in recommendations
            if recommendation.human_review_required
        )

        estimated_savings = sum(
            max(
                0.0,
                recommendation.potential_savings,
            )
            for recommendation in recommendations
        )

        estimated_impact = sum(
            max(
                0.0,
                recommendation.potential_impact,
            )
            for recommendation in recommendations
        )

        average_confidence = (
            sum(
                recommendation.confidence
                for recommendation in recommendations
            )
            / total
            if total
            else 0.0
        )

        overall_risk = self._calculate_overall_risk(
            recommendations
        )

        source_counts = self._count_sources(
            recommendations
        )

        type_counts = self._count_types(
            recommendations
        )

        executive_summary = (
            self._build_executive_summary(
                recommendations=recommendations,
                estimated_savings=estimated_savings,
                estimated_impact=estimated_impact,
                overall_risk=overall_risk,
                human_review_count=human_review_count,
            )
        )

        status = self._calculate_status(
            recommendations
        )

        result_id = self._create_result_id(
            recommendations
        )

        return RecommendationServiceResult(
            result_id=result_id,
            generated_at=_timestamp(),
            status=status,
            currency=self.config.currency,
            recommendations=recommendations,
            total_recommendations=total,
            critical_recommendations=critical,
            high_priority_recommendations=high,
            medium_priority_recommendations=medium,
            low_priority_recommendations=low,
            high_risk_recommendations=high_risk,
            human_review_count=human_review_count,
            estimated_savings=_round(
                estimated_savings
            ),
            estimated_financial_impact=_round(
                estimated_impact
            ),
            average_confidence=_round(
                average_confidence,
                4,
            ),
            overall_risk=overall_risk,
            executive_summary=executive_summary,
            source_counts=source_counts,
            type_counts=type_counts,
            metadata={
                "service": (
                    "FinCo AI Recommendation Service"
                ),
                "generated_date": _now_date(),
                "max_recommendations": (
                    self.config.max_recommendations
                ),
                "minimum_confidence": (
                    self.config.minimum_confidence
                ),
                "deduplication_enabled": True,
                "human_review_routing_enabled": True,
            },
        )

    def _calculate_overall_risk(
        self,
        recommendations: List[UnifiedRecommendation],
    ) -> str:
        """Calculate aggregate risk level."""

        if not recommendations:
            return "none"

        critical_count = sum(
            1
            for recommendation in recommendations
            if recommendation.risk == "critical"
        )

        high_count = sum(
            1
            for recommendation in recommendations
            if recommendation.risk == "high"
        )

        medium_count = sum(
            1
            for recommendation in recommendations
            if recommendation.risk == "medium"
        )

        if critical_count >= 1:
            return "critical"

        if high_count >= 2:
            return "high"

        if high_count >= 1:
            return "high"

        if medium_count >= 2:
            return "medium"

        return "low"

    def _calculate_status(
        self,
        recommendations: List[UnifiedRecommendation],
    ) -> str:
        """Calculate aggregate result status."""

        if not recommendations:
            return STATUS_NO_ACTION

        if any(
            recommendation.human_review_required
            for recommendation in recommendations
        ):
            return STATUS_REVIEW_REQUIRED

        return STATUS_READY

    def _count_sources(
        self,
        recommendations: List[UnifiedRecommendation],
    ) -> Dict[str, int]:
        """Count recommendations by source."""

        counts = {
            source: 0
            for source in RECOMMENDATION_SOURCES
        }

        for recommendation in recommendations:
            source = recommendation.source

            counts.setdefault(source, 0)

            counts[source] += 1

        return counts

    def _count_types(
        self,
        recommendations: List[UnifiedRecommendation],
    ) -> Dict[str, int]:
        """Count recommendations by recommendation type."""

        counts: Dict[str, int] = {}

        for recommendation in recommendations:
            rec_type = (
                recommendation.type
                or "combined"
            )

            counts[rec_type] = (
                counts.get(rec_type, 0) + 1
            )

        return dict(
            sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def _build_executive_summary(
        self,
        recommendations: List[UnifiedRecommendation],
        estimated_savings: float,
        estimated_impact: float,
        overall_risk: str,
        human_review_count: int,
    ) -> str:
        """Build concise executive summary."""

        if not recommendations:
            return (
                "No actionable recommendations were identified "
                "from the supplied analysis."
            )

        total = len(recommendations)

        critical = sum(
            1
            for recommendation in recommendations
            if recommendation.priority == "critical"
        )

        high = sum(
            1
            for recommendation in recommendations
            if recommendation.priority == "high"
        )

        parts = [
            (
                f"{total} actionable recommendation"
                f"{'s' if total != 1 else ''} identified."
            ),
            (
                f"Overall recommendation risk is "
                f"{overall_risk}."
            ),
        ]

        if critical:
            parts.append(
                f"{critical} critical recommendation"
                f"{'s' if critical != 1 else ''} require immediate "
                "management attention."
            )

        if high:
            parts.append(
                f"{high} high-priority recommendation"
                f"{'s' if high != 1 else ''} should be reviewed "
                "promptly."
            )

        if estimated_savings > 0:
            parts.append(
                "Potential identified savings: "
                f"{_format_currency(estimated_savings, self.config.currency)}."
            )

        if estimated_impact > 0:
            parts.append(
                "Potential financial impact: "
                f"{_format_currency(estimated_impact, self.config.currency)}."
            )

        if human_review_count:
            parts.append(
                f"{human_review_count} recommendation"
                f"{'s' if human_review_count != 1 else ''} "
                "require human review before implementation."
            )

        return " ".join(parts)

    # ---------------------------------------------------------------------
    # IDS
    # ---------------------------------------------------------------------

    @staticmethod
    def _create_recommendation_id(
        title: Any,
        source: Any,
        rec_type: Any,
    ) -> str:
        """Create deterministic-ish recommendation ID."""

        raw = (
            f"{source}|{rec_type}|{title}"
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:12]

        return f"REC-{digest.upper()}"

    def _create_result_id(
        self,
        recommendations: List[UnifiedRecommendation],
    ) -> str:
        """Create result ID."""

        seed = "|".join(
            recommendation.recommendation_id
            for recommendation in recommendations
        )

        digest = hashlib.sha256(
            seed.encode("utf-8")
        ).hexdigest()[:12]

        return f"RECS-{digest.upper()}"

    # ---------------------------------------------------------------------
    # EXPORT: JSON
    # ---------------------------------------------------------------------

    def to_json(
        self,
        result: RecommendationServiceResult,
        indent: int = 2,
    ) -> str:
        """Convert result to JSON."""
        return result.to_json(indent=indent)

    def save_json(
        self,
        result: RecommendationServiceResult,
        path: Optional[str] = None,
    ) -> Path:
        """Save recommendation result as JSON."""

        output_path = self._resolve_output_path(
            path,
            "recommendations.json",
        )

        output_path.write_text(
            self.to_json(result),
            encoding="utf-8",
        )

        return output_path

    # ---------------------------------------------------------------------
    # EXPORT: MARKDOWN
    # ---------------------------------------------------------------------

    def to_markdown(
        self,
        result: RecommendationServiceResult,
    ) -> str:
        """Convert result to Markdown."""

        lines: List[str] = []

        lines.append(
            "# FinCo AI — Recommendation Report"
        )
        lines.append("")

        lines.append(
            f"**Result ID:** `{result.result_id}`"
        )
        lines.append(
            f"**Generated:** {result.generated_at}"
        )
        lines.append(
            f"**Status:** `{result.status}`"
        )
        lines.append(
            f"**Overall Risk:** `{result.overall_risk.upper()}`"
        )

        lines.append("")

        lines.append("## Executive Summary")
        lines.append("")
        lines.append(result.executive_summary)
        lines.append("")

        lines.append("## Summary Metrics")
        lines.append("")

        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(
            f"| Total recommendations | "
            f"{result.total_recommendations} |"
        )
        lines.append(
            f"| Critical | "
            f"{result.critical_recommendations} |"
        )
        lines.append(
            f"| High priority | "
            f"{result.high_priority_recommendations} |"
        )
        lines.append(
            f"| Medium priority | "
            f"{result.medium_priority_recommendations} |"
        )
        lines.append(
            f"| Low priority | "
            f"{result.low_priority_recommendations} |"
        )
        lines.append(
            f"| High/critical risk | "
            f"{result.high_risk_recommendations} |"
        )
        lines.append(
            f"| Human review | "
            f"{result.human_review_count} |"
        )
        lines.append(
            f"| Estimated savings | "
            f"{_format_currency(result.estimated_savings, result.currency)} |"
        )
        lines.append(
            f"| Financial impact | "
            f"{_format_currency(result.estimated_financial_impact, result.currency)} |"
        )
        lines.append(
            f"| Average confidence | "
            f"{_format_percent(result.average_confidence * 100)} |"
        )

        lines.append("")

        lines.append("## Recommendation Sources")
        lines.append("")

        lines.append("| Source | Count |")
        lines.append("|---|---:|")

        for source, count in result.source_counts.items():
            if count:
                lines.append(
                    f"| {source} | {count} |"
                )

        lines.append("")

        lines.append("## Recommendations")
        lines.append("")

        if not result.recommendations:
            lines.append(
                "No actionable recommendations."
            )
            lines.append("")
            return "\n".join(lines)

        for index, recommendation in enumerate(
            result.recommendations,
            start=1,
        ):
            lines.append(
                f"### {index}. {recommendation.title}"
            )
            lines.append("")

            lines.append(
                f"**Source:** "
                f"`{recommendation.source}`"
            )
            lines.append(
                f"**Type:** "
                f"`{recommendation.type}`"
            )
            lines.append(
                f"**Priority:** "
                f"`{recommendation.priority.upper()}`"
            )
            lines.append(
                f"**Risk:** "
                f"`{recommendation.risk.upper()}`"
            )
            lines.append(
                f"**Confidence:** "
                f"{recommendation.confidence:.2%}"
            )

            if recommendation.affected_area:
                lines.append(
                    f"**Affected area:** "
                    f"{recommendation.affected_area}"
                )

            lines.append("")

            lines.append(
                f"**Description:** "
                f"{recommendation.description}"
            )
            lines.append("")

            lines.append(
                f"**Recommended action:** "
                f"{recommendation.action}"
            )
            lines.append("")

            if recommendation.potential_savings:
                lines.append(
                    "**Potential savings:** "
                    f"{_format_currency(recommendation.potential_savings, result.currency)}"
                )
                lines.append("")

            if recommendation.potential_impact:
                lines.append(
                    "**Potential impact:** "
                    f"{_format_currency(recommendation.potential_impact, result.currency)}"
                )
                lines.append("")

            if recommendation.evidence:
                lines.append("**Evidence:**")
                for evidence in recommendation.evidence:
                    lines.append(
                        f"- {_normalize_text(evidence)}"
                    )
                lines.append("")

            if recommendation.assumptions:
                lines.append("**Assumptions:**")
                for assumption in recommendation.assumptions:
                    lines.append(
                        f"- {assumption}"
                    )
                lines.append("")

            if recommendation.risks:
                lines.append("**Risks:**")
                for risk in recommendation.risks:
                    lines.append(
                        f"- {risk}"
                    )
                lines.append("")

            if recommendation.next_steps:
                lines.append("**Next steps:**")
                for step in recommendation.next_steps:
                    lines.append(
                        f"1. {step}"
                    )
                lines.append("")

            if recommendation.human_review_required:
                lines.append(
                    "> ⚠️ **Human review required before implementation.**"
                )
                lines.append("")

        return "\n".join(lines)

    def save_markdown(
        self,
        result: RecommendationServiceResult,
        path: Optional[str] = None,
    ) -> Path:
        """Save result as Markdown."""

        output_path = self._resolve_output_path(
            path,
            "recommendations.md",
        )

        output_path.write_text(
            self.to_markdown(result),
            encoding="utf-8",
        )

        return output_path

    # ---------------------------------------------------------------------
    # EXPORT: TEXT
    # ---------------------------------------------------------------------

    def to_text(
        self,
        result: RecommendationServiceResult,
    ) -> str:
        """Create terminal-friendly text output."""

        lines = [
            "FINCO AI - RECOMMENDATION REPORT",
            "=" * 70,
            f"Result ID: {result.result_id}",
            f"Status: {result.status}",
            f"Overall Risk: {result.overall_risk.upper()}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 70,
            result.executive_summary,
            "",
            "METRICS",
            "-" * 70,
            f"Recommendations: {result.total_recommendations}",
            f"Critical: {result.critical_recommendations}",
            f"High: {result.high_priority_recommendations}",
            f"Medium: {result.medium_priority_recommendations}",
            f"Low: {result.low_priority_recommendations}",
            f"High/Critical Risk: {result.high_risk_recommendations}",
            f"Human Review: {result.human_review_count}",
            (
                "Estimated Savings: "
                f"{_format_currency(result.estimated_savings, result.currency)}"
            ),
            (
                "Financial Impact: "
                f"{_format_currency(result.estimated_financial_impact, result.currency)}"
            ),
            (
                "Average Confidence: "
                f"{result.average_confidence:.2%}"
            ),
            "",
            "RECOMMENDATIONS",
            "-" * 70,
        ]

        for index, recommendation in enumerate(
            result.recommendations,
            start=1,
        ):
            lines.extend(
                [
                    "",
                    f"{index}. {recommendation.title}",
                    f"   Source: {recommendation.source}",
                    f"   Type: {recommendation.type}",
                    f"   Priority: {recommendation.priority}",
                    f"   Risk: {recommendation.risk}",
                    (
                        f"   Confidence: "
                        f"{recommendation.confidence:.2%}"
                    ),
                    (
                        f"   Action: "
                        f"{recommendation.action}"
                    ),
                ]
            )

            if recommendation.potential_impact:
                lines.append(
                    "   Impact: "
                    + _format_currency(
                        recommendation.potential_impact,
                        result.currency,
                    )
                )

            if recommendation.human_review_required:
                lines.append(
                    "   HUMAN REVIEW: REQUIRED"
                )

        return "\n".join(lines)

    def save_text(
        self,
        result: RecommendationServiceResult,
        path: Optional[str] = None,
    ) -> Path:
        """Save result as plain text."""

        output_path = self._resolve_output_path(
            path,
            "recommendations.txt",
        )

        output_path.write_text(
            self.to_text(result),
            encoding="utf-8",
        )

        return output_path

    # ---------------------------------------------------------------------
    # EXPORT: HTML
    # ---------------------------------------------------------------------

    def to_html(
        self,
        result: RecommendationServiceResult,
    ) -> str:
        """Generate responsive HTML report."""

        risk_class = _slug(
            result.overall_risk
        )

        cards: List[str] = []

        for recommendation in result.recommendations:

            priority_class = _slug(
                recommendation.priority
            )

            risk_class_rec = _slug(
                recommendation.risk
            )

            evidence_html = ""

            if recommendation.evidence:
                evidence_items = "".join(
                    (
                        "<li>"
                        + html.escape(
                            _normalize_text(evidence)
                        )
                        + "</li>"
                    )
                    for evidence in recommendation.evidence
                )

                evidence_html = f"""
                <div class="section">
                    <h4>Evidence</h4>
                    <ul>{evidence_items}</ul>
                </div>
                """

            next_steps_html = ""

            if recommendation.next_steps:
                next_items = "".join(
                    (
                        "<li>"
                        + html.escape(
                            step
                        )
                        + "</li>"
                    )
                    for step in recommendation.next_steps
                )

                next_steps_html = f"""
                <div class="section">
                    <h4>Next steps</h4>
                    <ol>{next_items}</ol>
                </div>
                """

            review_html = ""

            if recommendation.human_review_required:
                review_html = """
                <div class="review">
                    ⚠ Human review required before implementation.
                </div>
                """

            cards.append(
                f"""
                <article class="card">
                    <div class="badges">
                        <span class="badge {priority_class}">
                            {html.escape(
                                recommendation.priority.upper()
                            )}
                        </span>
                        <span class="badge {risk_class_rec}">
                            {html.escape(
                                recommendation.risk.upper()
                            )} RISK
                        </span>
                        <span class="badge">
                            {html.escape(
                                recommendation.source
                            )}
                        </span>
                    </div>

                    <h3>
                        {html.escape(
                            recommendation.title
                        )}
                    </h3>

                    <p class="description">
                        {html.escape(
                            recommendation.description
                        )}
                    </p>

                    <div class="action">
                        <strong>Recommended action</strong>
                        <p>
                            {html.escape(
                                recommendation.action
                            )}
                        </p>
                    </div>

                    <div class="metrics">
                        <div>
                            <span>Confidence</span>
                            <strong>
                                {recommendation.confidence:.0%}
                            </strong>
                        </div>

                        <div>
                            <span>Impact</span>
                            <strong>
                                {html.escape(
                                    _format_currency(
                                        recommendation.potential_impact,
                                        result.currency,
                                    )
                                )}
                            </strong>
                        </div>

                        <div>
                            <span>Savings</span>
                            <strong>
                                {html.escape(
                                    _format_currency(
                                        recommendation.potential_savings,
                                        result.currency,
                                    )
                                )}
                            </strong>
                        </div>
                    </div>

                    {evidence_html}

                    {next_steps_html}

                    {review_html}
                </article>
                """
            )

        cards_html = "\n".join(cards)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>FinCo AI Recommendation Report</title>

<style>
:root {{
    --bg: #f5f7fb;
    --surface: #ffffff;
    --text: #172033;
    --muted: #667085;
    --border: #e4e7ec;
    --critical: #b42318;
    --high: #d92d20;
    --medium: #b54708;
    --low: #027a48;
    --shadow: 0 8px 30px rgba(16, 24, 40, .08);
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    background: var(--bg);
    color: var(--text);
}}

.container {{
    width: min(1200px, 94%);
    margin: 0 auto;
    padding: 32px 0 60px;
}}

.hero {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 30px;
    box-shadow: var(--shadow);
    margin-bottom: 24px;
}}

.hero h1 {{
    margin: 0 0 10px;
    font-size: 32px;
}}

.hero p {{
    color: var(--muted);
    line-height: 1.7;
}}

.risk {{
    display: inline-flex;
    padding: 7px 12px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 12px;
    margin-top: 10px;
}}

.risk.critical {{
    background: #fee4e2;
    color: var(--critical);
}}

.risk.high {{
    background: #fef3f2;
    color: var(--high);
}}

.risk.medium {{
    background: #fffaeb;
    color: var(--medium);
}}

.risk.low {{
    background: #ecfdf3;
    color: var(--low);
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 28px;
}}

.stat {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px;
    box-shadow: var(--shadow);
}}

.stat span {{
    display: block;
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 8px;
}}

.stat strong {{
    font-size: 24px;
}}

.summary {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 28px;
    box-shadow: var(--shadow);
}}

.summary p {{
    line-height: 1.7;
}}

.cards {{
    display: grid;
    gap: 18px;
}}

.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    box-shadow: var(--shadow);
}}

.card h3 {{
    font-size: 20px;
    margin: 16px 0 10px;
}}

.description {{
    color: var(--muted);
    line-height: 1.7;
}}

.badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}}

.badge {{
    display: inline-flex;
    padding: 6px 10px;
    border-radius: 999px;
    background: #f2f4f7;
    color: #344054;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: .04em;
}}

.badge.critical {{
    background: #fee4e2;
    color: var(--critical);
}}

.badge.high {{
    background: #fef3f2;
    color: var(--high);
}}

.badge.medium {{
    background: #fffaeb;
    color: var(--medium);
}}

.badge.low {{
    background: #ecfdf3;
    color: var(--low);
}}

.action {{
    background: #f8fafc;
    border-radius: 14px;
    padding: 16px;
    margin-top: 18px;
}}

.action p {{
    margin-bottom: 0;
    line-height: 1.6;
}}

.metrics {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-top: 18px;
}}

.metrics div {{
    background: #fafafa;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
}}

.metrics span {{
    display: block;
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 6px;
}}

.section {{
    margin-top: 20px;
}}

.section h4 {{
    margin-bottom: 8px;
}}

.section li {{
    margin-bottom: 7px;
    line-height: 1.5;
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
    margin-top: 40px;
    color: var(--muted);
    font-size: 13px;
    text-align: center;
}}
</style>
</head>

<body>

<div class="container">

    <section class="hero">
        <h1>FinCo AI — Recommendation Report</h1>

        <p>
            Unified financial, risk, and cost recommendations
            prioritized using financial impact, risk severity,
            confidence, and operational priority.
        </p>

        <div class="risk {risk_class}">
            Overall Risk:
            {html.escape(result.overall_risk.upper())}
        </div>
    </section>

    <section class="grid">

        <div class="stat">
            <span>Total Recommendations</span>
            <strong>
                {result.total_recommendations}
            </strong>
        </div>

        <div class="stat">
            <span>Critical</span>
            <strong>
                {result.critical_recommendations}
            </strong>
        </div>

        <div class="stat">
            <span>High Priority</span>
            <strong>
                {result.high_priority_recommendations}
            </strong>
        </div>

        <div class="stat">
            <span>Human Review</span>
            <strong>
                {result.human_review_count}
            </strong>
        </div>

        <div class="stat">
            <span>Estimated Savings</span>
            <strong>
                {html.escape(
                    _format_currency(
                        result.estimated_savings,
                        result.currency,
                    )
                )}
            </strong>
        </div>

        <div class="stat">
            <span>Financial Impact</span>
            <strong>
                {html.escape(
                    _format_currency(
                        result.estimated_financial_impact,
                        result.currency,
                    )
                )}
            </strong>
        </div>

        <div class="stat">
            <span>Average Confidence</span>
            <strong>
                {result.average_confidence:.0%}
            </strong>
        </div>

    </section>

    <section class="summary">
        <h2>Executive Summary</h2>
        <p>
            {html.escape(
                result.executive_summary
            )}
        </p>
    </section>

    <section>
        <h2>Recommendations</h2>

        <div class="cards">
            {cards_html}
        </div>
    </section>

    <div class="footer">
        FinCo AI • Financial Intelligence & Decision Copilot
        • Generated {html.escape(result.generated_at)}
    </div>

</div>

</body>
</html>
"""

    def save_html(
        self,
        result: RecommendationServiceResult,
        path: Optional[str] = None,
    ) -> Path:
        """Save result as HTML."""

        output_path = self._resolve_output_path(
            path,
            "recommendations.html",
        )

        output_path.write_text(
            self.to_html(result),
            encoding="utf-8",
        )

        return output_path

    # ---------------------------------------------------------------------
    # EXPORT ALL
    # ---------------------------------------------------------------------

    def save_all(
        self,
        result: RecommendationServiceResult,
        directory: Optional[str] = None,
    ) -> Dict[str, Path]:
        """Save JSON, Markdown, Text, and HTML reports."""

        if directory:
            output_directory = Path(directory)
            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
        else:
            output_directory = Path(
                self.config.output_directory
            )

            output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        result_slug = _slug(
            result.result_id
        )

        paths = {
            "json": output_directory
            / f"{result_slug}.json",

            "markdown": output_directory
            / f"{result_slug}.md",

            "text": output_directory
            / f"{result_slug}.txt",

            "html": output_directory
            / f"{result_slug}.html",
        }

        paths["json"].write_text(
            self.to_json(result),
            encoding="utf-8",
        )

        paths["markdown"].write_text(
            self.to_markdown(result),
            encoding="utf-8",
        )

        paths["text"].write_text(
            self.to_text(result),
            encoding="utf-8",
        )

        paths["html"].write_text(
            self.to_html(result),
            encoding="utf-8",
        )

        return paths

    # ---------------------------------------------------------------------
    # OUTPUT PATH
    # ---------------------------------------------------------------------

    def _resolve_output_path(
        self,
        path: Optional[str],
        default_filename: str,
    ) -> Path:
        """Resolve and create output directory."""

        if path:
            output_path = Path(path)

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            return output_path

        output_directory = Path(
            self.config.output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output_directory / default_filename


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_recommendations(
    financial_recommendations: Optional[
        Iterable[Any]
    ] = None,
    risk_recommendations: Optional[
        Iterable[Any]
    ] = None,
    cost_recommendations: Optional[
        Iterable[Any]
    ] = None,
    financial_data: Optional[Any] = None,
    risk_data: Optional[Any] = None,
    cost_data: Optional[Any] = None,
    alerts: Optional[Iterable[Any]] = None,
    fraud_findings: Optional[Iterable[Any]] = None,
    forecast_data: Optional[Any] = None,
    config: Optional[
        RecommendationServiceConfig
    ] = None,
) -> RecommendationServiceResult:
    """
    Convenience API for generating unified recommendations.
    """

    service = RecommendationService(
        config=config
    )

    return service.generate(
        financial_recommendations=(
            financial_recommendations
        ),
        risk_recommendations=(
            risk_recommendations
        ),
        cost_recommendations=(
            cost_recommendations
        ),
        financial_data=financial_data,
        risk_data=risk_data,
        cost_data=cost_data,
        alerts=alerts,
        fraud_findings=fraud_findings,
        forecast_data=forecast_data,
    )


# ============================================================================
# DEMO DATA
# ============================================================================

def create_demo_financial_recommendations() -> List[Dict[str, Any]]:
    """Create realistic financial recommendation examples."""

    return [
        {
            "recommendation_id": "FIN-001",
            "source": "financial",
            "type": "margin_improvement",
            "title": "Improve declining operating margin",
            "description": (
                "Operating margin has declined due to "
                "higher operating expenses relative to revenue."
            ),
            "action": (
                "Review high-growth expense categories and "
                "implement targeted cost controls."
            ),
            "priority": "high",
            "risk": "high",
            "confidence": 0.91,
            "current_value": 8.4,
            "benchmark_value": 12.0,
            "potential_impact": 125000,
            "potential_savings": 0,
            "impact_percent": 3.6,
            "affected_area": "Profitability",
            "evidence": [
                "Operating margin declined from 11.7% to 8.4%.",
                "Operating expenses increased faster than revenue.",
            ],
            "assumptions": [
                "Expense reductions can be achieved without material revenue impact."
            ],
            "risks": [
                "Aggressive cost reductions may affect service quality."
            ],
            "next_steps": [
                "Identify the five fastest-growing operating expense categories.",
                "Assign owners to each category.",
                "Set quarterly reduction targets.",
            ],
        },
        {
            "recommendation_id": "FIN-002",
            "source": "financial",
            "type": "working_capital",
            "title": "Reduce receivables collection cycle",
            "description": (
                "Receivables are converting to cash more slowly "
                "than the desired operating target."
            ),
            "action": (
                "Prioritize overdue invoices and introduce "
                "customer-specific collection workflows."
            ),
            "priority": "high",
            "risk": "medium",
            "confidence": 0.87,
            "current_value": 68,
            "benchmark_value": 45,
            "potential_impact": 180000,
            "potential_savings": 0,
            "impact_percent": 33.82,
            "affected_area": "Working Capital",
            "evidence": [
                "DSO is approximately 68 days.",
                "Target DSO is 45 days.",
            ],
            "next_steps": [
                "Segment receivables by age.",
                "Escalate high-value overdue invoices.",
                "Review customer credit terms.",
            ],
        },
    ]


def create_demo_risk_recommendations() -> List[Dict[str, Any]]:
    """Create realistic risk recommendations."""

    return [
        {
            "recommendation_id": "RISK-001",
            "source": "risk",
            "type": "liquidity",
            "title": "Strengthen short-term liquidity position",
            "description": (
                "Liquidity indicators show increased exposure "
                "to near-term cash pressure."
            ),
            "action": (
                "Increase minimum cash reserves and accelerate "
                "working-capital conversion."
            ),
            "priority": "critical",
            "risk": "critical",
            "confidence": 0.94,
            "current_value": 0.92,
            "benchmark_value": 1.50,
            "potential_impact": 310000,
            "potential_savings": 0,
            "affected_area": "Liquidity",
            "evidence": [
                "Current ratio is below the configured safety threshold.",
                "Cash runway is declining.",
                "Short-term obligations are increasing.",
            ],
            "assumptions": [
                "Management can accelerate receivables collection."
            ],
            "risks": [
                "Failure to improve liquidity may constrain operations."
            ],
            "next_steps": [
                "Review the 13-week cash-flow forecast.",
                "Freeze non-essential discretionary spending.",
                "Escalate large overdue receivables.",
            ],
        },
        {
            "recommendation_id": "RISK-002",
            "source": "risk",
            "type": "concentration_risk",
            "title": "Reduce customer concentration exposure",
            "description": (
                "A significant proportion of revenue is generated "
                "from a small number of customers."
            ),
            "action": (
                "Develop customer diversification targets and "
                "reduce dependency on the largest accounts."
            ),
            "priority": "high",
            "risk": "high",
            "confidence": 0.89,
            "current_value": 58,
            "benchmark_value": 35,
            "potential_impact": 210000,
            "potential_savings": 0,
            "impact_percent": 39.65,
            "affected_area": "Revenue Concentration",
            "evidence": [
                "Top five customers represent approximately 58% of revenue."
            ],
            "next_steps": [
                "Create account-level concentration limits.",
                "Build pipeline targets for new customers.",
                "Monitor monthly concentration ratios.",
            ],
        },
    ]


def create_demo_cost_recommendations() -> List[Dict[str, Any]]:
    """Create realistic cost recommendations."""

    return [
        {
            "recommendation_id": "COST-001",
            "source": "cost",
            "type": "cost_optimization",
            "title": "Optimize external service expenses",
            "description": (
                "External service expenses are materially above "
                "the historical operating baseline."
            ),
            "action": (
                "Renegotiate supplier contracts and consolidate "
                "overlapping service agreements."
            ),
            "priority": "high",
            "risk": "medium",
            "confidence": 0.88,
            "current_value": 620000,
            "benchmark_value": 560000,
            "potential_impact": 60000,
            "potential_savings": 60000,
            "impact_percent": 9.68,
            "affected_area": "Operating Expenses",
            "affected_category": "External Services",
            "evidence": [
                "External service expenses increased by 18%.",
                "Three suppliers provide overlapping services.",
            ],
            "next_steps": [
                "Review all active supplier contracts.",
                "Identify contract consolidation opportunities.",
                "Negotiate volume-based pricing.",
            ],
        },
        {
            "recommendation_id": "COST-002",
            "source": "cost",
            "type": "cash_preservation",
            "title": "Defer non-essential discretionary spending",
            "description": (
                "Discretionary spending can be temporarily reduced "
                "to protect cash reserves."
            ),
            "action": (
                "Delay low-priority discretionary purchases "
                "until liquidity conditions improve."
            ),
            "priority": "medium",
            "risk": "low",
            "confidence": 0.82,
            "current_value": 180000,
            "benchmark_value": 150000,
            "potential_impact": 30000,
            "potential_savings": 30000,
            "impact_percent": 16.67,
            "affected_area": "Cash Preservation",
            "evidence": [
                "Discretionary spending exceeds the internal target."
            ],
            "next_steps": [
                "Classify discretionary spending by business value.",
                "Defer low-value purchases.",
            ],
        },
    ]


def create_demo_result() -> RecommendationServiceResult:
    """Generate a complete demo recommendation result."""

    service = RecommendationService()

    return service.generate(
        financial_recommendations=(
            create_demo_financial_recommendations()
        ),
        risk_recommendations=(
            create_demo_risk_recommendations()
        ),
        cost_recommendations=(
            create_demo_cost_recommendations()
        ),
        financial_data={
            "revenue": 5_200_000,
            "operating_expenses": 4_100_000,
            "cash": 760_000,
        },
        risk_data={
            "current_ratio": 0.92,
            "customer_concentration": 58,
        },
        cost_data={
            "external_services": 620_000,
        },
        alerts=[
            {
                "severity": "high",
                "type": "liquidity",
            },
            {
                "severity": "medium",
                "type": "budget_variance",
            },
        ],
        fraud_findings=[],
        forecast_data={
            "revenue_growth": 0.04,
            "downside_probability": 0.21,
        },
    )


# ============================================================================
# CLI
# ============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "FinCo AI Recommendation Service"
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
            "text",
            "txt",
            "html",
            "all",
        ],
        default="all",
        help="Output format.",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum recommendation confidence.",
    )

    parser.add_argument(
        "--max-recommendations",
        type=int,
        default=30,
        help="Maximum recommendations.",
    )

    return parser


def main() -> None:
    """CLI entrypoint."""

    parser = build_argument_parser()

    args = parser.parse_args()

    config = RecommendationServiceConfig(
        minimum_confidence=args.min_confidence,
        max_recommendations=args.max_recommendations,
        output_directory=args.output,
    )

    service = RecommendationService(
        config=config
    )

    result = service.generate(
        financial_recommendations=(
            create_demo_financial_recommendations()
        ),
        risk_recommendations=(
            create_demo_risk_recommendations()
        ),
        cost_recommendations=(
            create_demo_cost_recommendations()
        ),
    )

    if args.format == "json":
        path = service.save_json(
            result,
            str(
                Path(args.output)
                / "recommendations.json"
            ),
        )

        print(f"Saved: {path}")

    elif args.format in {"markdown", "md"}:
        path = service.save_markdown(
            result,
            str(
                Path(args.output)
                / "recommendations.md"
            ),
        )

        print(f"Saved: {path}")

    elif args.format in {"text", "txt"}:
        path = service.save_text(
            result,
            str(
                Path(args.output)
                / "recommendations.txt"
            ),
        )

        print(f"Saved: {path}")

    elif args.format == "html":
        path = service.save_html(
            result,
            str(
                Path(args.output)
                / "recommendations.html"
            ),
        )

        print(f"Saved: {path}")

    else:
        paths = service.save_all(
            result,
            args.output,
        )

        for format_name, path in paths.items():
            print(
                f"{format_name.upper()}: {path}"
            )

    print()
    print(
        "FinCo AI Recommendation Service"
    )
    print(
        f"Result ID: {result.result_id}"
    )
    print(
        f"Status: {result.status}"
    )
    print(
        f"Recommendations: "
        f"{result.total_recommendations}"
    )
    print(
        f"Critical: "
        f"{result.critical_recommendations}"
    )
    print(
        f"High Priority: "
        f"{result.high_priority_recommendations}"
    )
    print(
        f"Human Review: "
        f"{result.human_review_count}"
    )
    print(
        f"Estimated Savings: "
        f"{_format_currency(result.estimated_savings, result.currency)}"
    )
    print(
        f"Financial Impact: "
        f"{_format_currency(result.estimated_financial_impact, result.currency)}"
    )
    print(
        f"Overall Risk: "
        f"{result.overall_risk}"
    )


if __name__ == "__main__":
    main()