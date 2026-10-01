"""
FinCo AI - Recommendation Agent
================================

Recommendation and decision-support agent for FinCo AI.

Responsibilities
----------------
- Analyze financial findings from other agents.
- Convert risks and KPIs into actionable recommendations.
- Consume forecast, fraud, RAG and what-if results.
- Prioritize recommendations by impact and urgency.
- Explain why a recommendation was generated.
- Attach supporting evidence.
- Estimate expected impact.
- Identify implementation effort.
- Detect recommendations requiring human approval.
- Avoid unsafe or unsupported financial claims.
- Return structured recommendations to the orchestrator.

Architecture
------------

Financial Agent
Forecast Agent
Fraud Agent
RAG Agent
What-If Agent
        |
        v
Recommendation Agent
        |
        +--> Rule Engine
        +--> Risk Analysis
        +--> Financial Impact
        +--> Evidence Mapping
        +--> Priority Scoring
        +--> Optional LLM Explanation
        +--> Guardrails
        |
        v
Prioritized Recommendations
        |
        v
Orchestrator / Human Decision Maker


Important
---------
This agent is a decision-support component.

It does NOT:
- execute payments,
- modify financial records,
- approve transactions,
- transfer money,
- change accounting data,
- override authorization,
- bypass human approval.
"""

from __future__ import annotations

import inspect
import re
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
)


# ============================================================================
# Exceptions
# ============================================================================


class RecommendationAgentError(Exception):
    """Base recommendation-agent exception."""


class InvalidRecommendationRequestError(
    RecommendationAgentError
):
    """Raised when recommendation input is invalid."""


class RecommendationConfigurationError(
    RecommendationAgentError
):
    """Raised when configuration is invalid."""


class RecommendationExecutionError(
    RecommendationAgentError
):
    """Raised when recommendation execution fails."""


class RecommendationAccessDeniedError(
    RecommendationAgentError
):
    """Raised when recommendation access is denied."""


class RecommendationGuardrailViolationError(
    RecommendationAgentError
):
    """Raised when recommendation guardrails block execution."""


class RecommendationGenerationError(
    RecommendationAgentError
):
    """Raised when recommendation generation fails."""


class RecommendationValidationError(
    RecommendationAgentError
):
    """Raised when recommendations cannot be validated."""


# ============================================================================
# Constants
# ============================================================================


DEFAULT_MAX_RECOMMENDATIONS = 10
DEFAULT_MAX_INPUT_LENGTH = 50_000
DEFAULT_MIN_CONFIDENCE = 0.50
DEFAULT_HIGH_IMPACT_THRESHOLD = 0.70
DEFAULT_CRITICAL_RISK_THRESHOLD = 0.85
DEFAULT_TIMEOUT_SECONDS = 60


# ============================================================================
# Enums
# ============================================================================


class RecommendationOperation(str, Enum):
    """Supported recommendation operations."""

    GENERATE = "generate"
    FINANCIAL = "financial"
    RISK = "risk"
    COST = "cost"
    FORECAST = "forecast"
    FRAUD = "fraud"
    LIQUIDITY = "liquidity"
    PROFITABILITY = "profitability"
    STRATEGIC = "strategic"
    PRIORITIZE = "prioritize"
    ACTION_PLAN = "action_plan"
    SUMMARY = "summary"


class RecommendationCategory(str, Enum):
    """Recommendation categories."""

    REVENUE = "revenue"
    COST = "cost"
    PROFITABILITY = "profitability"
    CASH_FLOW = "cash_flow"
    LIQUIDITY = "liquidity"
    FRAUD = "fraud"
    RISK = "risk"
    FORECAST = "forecast"
    WORKING_CAPITAL = "working_capital"
    BUDGET = "budget"
    OPERATIONS = "operations"
    DOCUMENT = "document"
    STRATEGIC = "strategic"


class RecommendationPriority(str, Enum):
    """Recommendation priority."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationUrgency(str, Enum):
    """Action urgency."""

    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class RecommendationImpact(str, Enum):
    """Expected business impact."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class RecommendationStatus(str, Enum):
    """Recommendation lifecycle state."""

    PROPOSED = "proposed"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class RiskLevel(str, Enum):
    """Risk level used by recommendation rules."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RecommendationAgentConfig:
    """Runtime configuration."""

    enabled: bool = True

    max_recommendations: int = (
        DEFAULT_MAX_RECOMMENDATIONS
    )

    max_input_length: int = (
        DEFAULT_MAX_INPUT_LENGTH
    )

    min_confidence: float = (
        DEFAULT_MIN_CONFIDENCE
    )

    high_impact_threshold: float = (
        DEFAULT_HIGH_IMPACT_THRESHOLD
    )

    critical_risk_threshold: float = (
        DEFAULT_CRITICAL_RISK_THRESHOLD
    )

    timeout_seconds: float = (
        DEFAULT_TIMEOUT_SECONDS
    )

    require_human_review_for_high_impact: bool = True

    require_human_review_for_financial_actions: bool = True

    enable_llm_explanation: bool = True

    enable_rule_engine: bool = True

    enable_impact_estimation: bool = True

    audit_enabled: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate configuration."""

        if self.max_recommendations < 1:
            raise RecommendationConfigurationError(
                "max_recommendations must be positive."
            )

        if self.max_input_length < 100:
            raise RecommendationConfigurationError(
                "max_input_length is too small."
            )

        if not 0 <= self.min_confidence <= 1:
            raise RecommendationConfigurationError(
                "min_confidence must be between 0 and 1."
            )

        if not 0 <= self.high_impact_threshold <= 1:
            raise RecommendationConfigurationError(
                "high_impact_threshold must be between 0 and 1."
            )

        if not 0 <= self.critical_risk_threshold <= 1:
            raise RecommendationConfigurationError(
                "critical_risk_threshold must be between 0 and 1."
            )


# ============================================================================
# Request Model
# ============================================================================


@dataclass
class RecommendationAgentRequest:
    """Input to the recommendation agent."""

    query: str

    operation: RecommendationOperation = (
        RecommendationOperation.GENERATE
    )

    company_id: Optional[str] = None

    user_id: Optional[str] = None

    session_id: Optional[str] = None

    workflow_id: Optional[str] = None

    financial_data: Dict[str, Any] = field(
        default_factory=dict
    )

    forecast_data: Dict[str, Any] = field(
        default_factory=dict
    )

    fraud_data: Dict[str, Any] = field(
        default_factory=dict
    )

    risk_data: Dict[str, Any] = field(
        default_factory=dict
    )

    what_if_data: Dict[str, Any] = field(
        default_factory=dict
    )

    rag_data: Dict[str, Any] = field(
        default_factory=dict
    )

    previous_results: List[Any] = field(
        default_factory=list
    )

    context: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# Recommendation Models
# ============================================================================


@dataclass
class RecommendationEvidence:
    """Evidence supporting a recommendation."""

    evidence_id: str

    source: str

    description: str

    confidence: float = 0.0

    document_id: Optional[str] = None

    citation_id: Optional[str] = None

    metric: Optional[str] = None

    value: Optional[Any] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "description": self.description,
            "confidence": self.confidence,
            "document_id": self.document_id,
            "citation_id": self.citation_id,
            "metric": self.metric,
            "value": _serialize(self.value),
            "metadata": _serialize(
                self.metadata
            ),
        }


@dataclass
class RecommendationImpactEstimate:
    """Estimated business impact."""

    metric: Optional[str] = None

    estimated_value: Optional[float] = None

    lower_bound: Optional[float] = None

    upper_bound: Optional[float] = None

    unit: Optional[str] = None

    confidence: float = 0.0

    assumptions: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "metric": self.metric,
            "estimated_value": self.estimated_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "unit": self.unit,
            "confidence": self.confidence,
            "assumptions": list(
                self.assumptions
            ),
            "metadata": _serialize(
                self.metadata
            ),
        }


@dataclass
class Recommendation:
    """Structured business recommendation."""

    recommendation_id: str

    title: str

    action: str

    rationale: str

    category: RecommendationCategory

    priority: RecommendationPriority

    urgency: RecommendationUrgency

    impact: RecommendationImpact

    confidence: float

    risk_level: RiskLevel = RiskLevel.LOW

    status: RecommendationStatus = (
        RecommendationStatus.PROPOSED
    )

    expected_benefit: Optional[str] = None

    impact_estimate: Optional[
        RecommendationImpactEstimate
    ] = None

    evidence: List[
        RecommendationEvidence
    ] = field(
        default_factory=list
    )

    dependencies: List[str] = field(
        default_factory=list
    )

    assumptions: List[str] = field(
        default_factory=list
    )

    risks: List[str] = field(
        default_factory=list
    )

    human_review_required: bool = False

    executable: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "recommendation_id": (
                self.recommendation_id
            ),
            "title": self.title,
            "action": self.action,
            "rationale": self.rationale,
            "category": self.category.value,
            "priority": self.priority.value,
            "urgency": self.urgency.value,
            "impact": self.impact.value,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "status": self.status.value,
            "expected_benefit": (
                self.expected_benefit
            ),
            "impact_estimate": (
                self.impact_estimate.to_dict()
                if self.impact_estimate
                else None
            ),
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "dependencies": list(
                self.dependencies
            ),
            "assumptions": list(
                self.assumptions
            ),
            "risks": list(
                self.risks
            ),
            "human_review_required": (
                self.human_review_required
            ),
            "executable": self.executable,
            "metadata": _serialize(
                self.metadata
            ),
        }


@dataclass
class RecommendationAgentResult:
    """Result returned to the orchestrator."""

    request_id: str

    operation: RecommendationOperation

    query: str

    recommendations: List[
        Recommendation
    ] = field(
        default_factory=list
    )

    top_recommendation: Optional[
        Recommendation
    ] = None

    action_plan: List[str] = field(
        default_factory=list
    )

    summary: str = ""

    confidence: float = 0.0

    risk_level: RiskLevel = RiskLevel.LOW

    human_review_required: bool = False

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    execution_time_ms: float = 0.0

    success: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "request_id": self.request_id,
            "operation": self.operation.value,
            "query": self.query,
            "recommendations": [
                item.to_dict()
                for item in self.recommendations
            ],
            "top_recommendation": (
                self.top_recommendation.to_dict()
                if self.top_recommendation
                else None
            ),
            "action_plan": list(
                self.action_plan
            ),
            "summary": self.summary,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "human_review_required": (
                self.human_review_required
            ),
            "warnings": list(
                self.warnings
            ),
            "errors": list(
                self.errors
            ),
            "execution_time_ms": (
                self.execution_time_ms
            ),
            "success": self.success,
            "metadata": _serialize(
                self.metadata
            ),
        }


# ============================================================================
# Recommendation Agent
# ============================================================================


class RecommendationAgent:
    """
    FinCo AI recommendation and decision-support agent.

    The agent converts observations into prioritized actions.

    It does not execute financial actions.
    """

    def __init__(
        self,
        recommendation_service: Optional[
            Any
        ] = None,
        rule_engine: Optional[
            Any
        ] = None,
        generator: Optional[
            Any
        ] = None,
        authorization_service: Optional[
            Any
        ] = None,
        guardrails: Optional[
            Any
        ] = None,
        audit_service: Optional[
            Any
        ] = None,
        config: Optional[
            RecommendationAgentConfig
        ] = None,
    ) -> None:

        self.recommendation_service = (
            recommendation_service
        )

        self.rule_engine = (
            rule_engine
        )

        self.generator = generator

        self.authorization_service = (
            authorization_service
        )

        self.guardrails = guardrails

        self.audit_service = audit_service

        self.config = (
            config
            or RecommendationAgentConfig()
        )

        self.config.validate()

    # ========================================================================
    # Main API
    # ========================================================================

    def run(
        self,
        request: RecommendationAgentRequest
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> RecommendationAgentResult:

        request = self.normalize_request(
            request,
            **kwargs,
        )

        request_id = (
            request.metadata.get(
                "request_id"
            )
            or self.generate_request_id()
        )

        started = time.perf_counter()

        try:

            self.validate_request(
                request
            )

            self.authorize(
                request
            )

            self.check_guardrails(
                request
            )

            recommendations = (
                self.generate_recommendations(
                    request
                )
            )

            recommendations = (
                self.validate_recommendations(
                    recommendations
                )
            )

            recommendations = (
                self.prioritize(
                    recommendations
                )
            )

            recommendations = (
                recommendations[
                    : self.config.max_recommendations
                ]
            )

            action_plan = (
                self.build_action_plan(
                    recommendations
                )
            )

            summary = (
                self.build_summary(
                    recommendations
                )
            )

            confidence = (
                self.calculate_confidence(
                    recommendations
                )
            )

            risk_level = (
                self.calculate_risk_level(
                    recommendations
                )
            )

            human_review_required = any(
                item.human_review_required
                for item in recommendations
            )

            result = RecommendationAgentResult(
                request_id=request_id,
                operation=request.operation,
                query=request.query,
                recommendations=recommendations,
                top_recommendation=(
                    recommendations[0]
                    if recommendations
                    else None
                ),
                action_plan=action_plan,
                summary=summary,
                confidence=confidence,
                risk_level=risk_level,
                human_review_required=(
                    human_review_required
                ),
                success=True,
            )

            self.finalize_result(
                result,
                started,
            )

            self.audit(
                request,
                result,
            )

            return result

        except RecommendationAgentError:
            raise

        except Exception as exc:

            raise RecommendationExecutionError(
                "Recommendation agent execution failed."
            ) from exc

    # ========================================================================
    # Specialized APIs
    # ========================================================================

    def recommend_financial(
        self,
        query: str,
        **kwargs: Any,
    ) -> RecommendationAgentResult:

        return self.run(
            RecommendationAgentRequest(
                query=query,
                operation=(
                    RecommendationOperation.FINANCIAL
                ),
                **kwargs,
            )
        )

    def recommend_risk(
        self,
        query: str,
        **kwargs: Any,
    ) -> RecommendationAgentResult:

        return self.run(
            RecommendationAgentRequest(
                query=query,
                operation=(
                    RecommendationOperation.RISK
                ),
                **kwargs,
            )
        )

    def recommend_cost(
        self,
        query: str,
        **kwargs: Any,
    ) -> RecommendationAgentResult:

        return self.run(
            RecommendationAgentRequest(
                query=query,
                operation=(
                    RecommendationOperation.COST
                ),
                **kwargs,
            )
        )

    def recommend_forecast(
        self,
        query: str,
        **kwargs: Any,
    ) -> RecommendationAgentResult:

        return self.run(
            RecommendationAgentRequest(
                query=query,
                operation=(
                    RecommendationOperation.FORECAST
                ),
                **kwargs,
            )
        )

    def recommend_liquidity(
        self,
        query: str,
        **kwargs: Any,
    ) -> RecommendationAgentResult:

        return self.run(
            RecommendationAgentRequest(
                query=query,
                operation=(
                    RecommendationOperation.LIQUIDITY
                ),
                **kwargs,
            )
        )

    # ========================================================================
    # Recommendation Generation
    # ========================================================================

    def generate_recommendations(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        recommendations: List[
            Recommendation
        ] = []

        # --------------------------------------------------------------
        # External recommendation service
        # --------------------------------------------------------------

        if self.recommendation_service is not None:

            service_results = (
                self.call_recommendation_service(
                    request
                )
            )

            recommendations.extend(
                self.normalize_recommendations(
                    service_results
                )
            )

        # --------------------------------------------------------------
        # Rule engine
        # --------------------------------------------------------------

        if (
            self.config.enable_rule_engine
            and self.rule_engine is not None
        ):

            rule_results = (
                self.call_rule_engine(
                    request
                )
            )

            recommendations.extend(
                self.normalize_recommendations(
                    rule_results
                )
            )

        # --------------------------------------------------------------
        # Built-in deterministic rules
        # --------------------------------------------------------------

        recommendations.extend(
            self.generate_rule_based_recommendations(
                request
            )
        )

        # --------------------------------------------------------------
        # LLM explanation / additional recommendations
        # --------------------------------------------------------------

        if (
            self.config.enable_llm_explanation
            and self.generator is not None
        ):

            generated = (
                self.generate_llm_recommendations(
                    request,
                    recommendations,
                )
            )

            recommendations.extend(
                self.normalize_recommendations(
                    generated
                )
            )

        return self.deduplicate_recommendations(
            recommendations
        )

    # ========================================================================
    # External Recommendation Service
    # ========================================================================

    def call_recommendation_service(
        self,
        request: RecommendationAgentRequest,
    ) -> Any:

        payload = self.build_service_payload(
            request
        )

        try:

            return self.invoke_component(
                self.recommendation_service,
                (
                    "recommend",
                    "generate",
                    "get_recommendations",
                    "analyze",
                    "run",
                ),
                payload,
            )

        except Exception as exc:

            raise RecommendationGenerationError(
                "Recommendation service failed."
            ) from exc

    # ========================================================================
    # Rule Engine
    # ========================================================================

    def call_rule_engine(
        self,
        request: RecommendationAgentRequest,
    ) -> Any:

        payload = self.build_service_payload(
            request
        )

        try:

            return self.invoke_component(
                self.rule_engine,
                (
                    "evaluate",
                    "evaluate_rules",
                    "recommend",
                    "run",
                ),
                payload,
            )

        except Exception:

            # Rules are supplemental. Built-in rules
            # remain available if external rules fail.
            return []

    # ========================================================================
    # Built-in Financial Rules
    # ========================================================================

    def generate_rule_based_recommendations(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        recommendations: List[
            Recommendation
        ] = []

        recommendations.extend(
            self.revenue_rules(
                request
            )
        )

        recommendations.extend(
            self.expense_rules(
                request
            )
        )

        recommendations.extend(
            self.profitability_rules(
                request
            )
        )

        recommendations.extend(
            self.liquidity_rules(
                request
            )
        )

        recommendations.extend(
            self.forecast_rules(
                request
            )
        )

        recommendations.extend(
            self.fraud_rules(
                request
            )
        )

        recommendations.extend(
            self.risk_rules(
                request
            )
        )

        recommendations.extend(
            self.what_if_rules(
                request
            )
        )

        return recommendations

    # ========================================================================
    # Revenue Rules
    # ========================================================================

    def revenue_rules(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        data = request.financial_data

        revenue_growth = self.get_numeric(
            data,
            (
                "revenue_growth",
                "revenue_change_pct",
                "revenue_growth_pct",
            ),
        )

        if revenue_growth is None:

            return []

        if revenue_growth <= -20:

            return [
                self.create_recommendation(
                    title=(
                        "Investigate and address "
                        "the revenue decline"
                    ),
                    action=(
                        "Perform a product, customer, "
                        "pricing and channel analysis to "
                        "identify the primary drivers of "
                        "the revenue decline."
                    ),
                    rationale=(
                        f"Revenue declined by "
                        f"{revenue_growth:.1f}%."
                    ),
                    category=(
                        RecommendationCategory.REVENUE
                    ),
                    priority=(
                        RecommendationPriority.CRITICAL
                    ),
                    urgency=(
                        RecommendationUrgency.IMMEDIATE
                    ),
                    impact=(
                        RecommendationImpact.VERY_HIGH
                    ),
                    confidence=0.90,
                    risk_level=RiskLevel.CRITICAL,
                    expected_benefit=(
                        "Reduce the probability of "
                        "continued revenue deterioration."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="financial_data",
                            metric="revenue_growth_pct",
                            value=revenue_growth,
                            description=(
                                "Reported revenue change."
                            ),
                        )
                    ],
                    human_review=True,
                )
            ]

        if revenue_growth <= -10:

            return [
                self.create_recommendation(
                    title=(
                        "Investigate revenue "
                        "deterioration"
                    ),
                    action=(
                        "Review customer retention, "
                        "pricing, product mix and "
                        "sales-channel performance."
                    ),
                    rationale=(
                        f"Revenue declined by "
                        f"{revenue_growth:.1f}%."
                    ),
                    category=(
                        RecommendationCategory.REVENUE
                    ),
                    priority=(
                        RecommendationPriority.HIGH
                    ),
                    urgency=(
                        RecommendationUrgency.SHORT_TERM
                    ),
                    impact=(
                        RecommendationImpact.HIGH
                    ),
                    confidence=0.84,
                    risk_level=RiskLevel.HIGH,
                    expected_benefit=(
                        "Identify controllable revenue "
                        "leakage and recovery opportunities."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="financial_data",
                            metric="revenue_growth_pct",
                            value=revenue_growth,
                            description=(
                                "Reported revenue change."
                            ),
                        )
                    ],
                )
            ]

        return []

    # ========================================================================
    # Expense Rules
    # ========================================================================

    def expense_rules(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        data = request.financial_data

        expense_growth = self.get_numeric(
            data,
            (
                "expense_growth",
                "expense_change_pct",
                "expenses_growth_pct",
            ),
        )

        revenue_growth = self.get_numeric(
            data,
            (
                "revenue_growth",
                "revenue_change_pct",
            ),
        )

        if expense_growth is None:

            return []

        if (
            expense_growth >= 20
            and (
                revenue_growth is None
                or expense_growth > revenue_growth
            )
        ):

            return [
                self.create_recommendation(
                    title=(
                        "Review rapidly increasing expenses"
                    ),
                    action=(
                        "Perform category-level expense "
                        "analysis and identify discretionary, "
                        "duplicated or inefficient spending."
                    ),
                    rationale=(
                        f"Expenses increased by "
                        f"{expense_growth:.1f}%."
                    ),
                    category=(
                        RecommendationCategory.COST
                    ),
                    priority=(
                        RecommendationPriority.HIGH
                    ),
                    urgency=(
                        RecommendationUrgency.SHORT_TERM
                    ),
                    impact=(
                        RecommendationImpact.HIGH
                    ),
                    confidence=0.86,
                    risk_level=RiskLevel.HIGH,
                    expected_benefit=(
                        "Improve cost efficiency and "
                        "protect operating margin."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="financial_data",
                            metric="expense_growth_pct",
                            value=expense_growth,
                            description=(
                                "Reported expense change."
                            ),
                        )
                    ],
                )
            ]

        return []

    # ========================================================================
    # Profitability Rules
    # ========================================================================

    def profitability_rules(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        data = request.financial_data

        profit_growth = self.get_numeric(
            data,
            (
                "profit_growth",
                "profit_change_pct",
                "profit_growth_pct",
            ),
        )

        margin = self.get_numeric(
            data,
            (
                "profit_margin",
                "net_margin",
                "operating_margin",
            ),
        )

        recommendations: List[
            Recommendation
        ] = []

        if (
            profit_growth is not None
            and profit_growth <= -20
        ):

            recommendations.append(
                self.create_recommendation(
                    title=(
                        "Protect profitability"
                    ),
                    action=(
                        "Run a margin bridge covering "
                        "revenue, pricing, volume, product "
                        "mix and operating expenses."
                    ),
                    rationale=(
                        f"Profit declined by "
                        f"{profit_growth:.1f}%."
                    ),
                    category=(
                        RecommendationCategory.PROFITABILITY
                    ),
                    priority=(
                        RecommendationPriority.CRITICAL
                    ),
                    urgency=(
                        RecommendationUrgency.IMMEDIATE
                    ),
                    impact=(
                        RecommendationImpact.VERY_HIGH
                    ),
                    confidence=0.90,
                    risk_level=RiskLevel.CRITICAL,
                    expected_benefit=(
                        "Identify the largest drivers "
                        "of profitability deterioration."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="financial_data",
                            metric="profit_growth_pct",
                            value=profit_growth,
                            description=(
                                "Reported profit change."
                            ),
                        )
                    ],
                    human_review=True,
                )
            )

        if (
            margin is not None
            and margin < 5
        ):

            recommendations.append(
                self.create_recommendation(
                    title=(
                        "Improve operating margin"
                    ),
                    action=(
                        "Review low-margin products, "
                        "pricing, variable costs and "
                        "customer-level profitability."
                    ),
                    rationale=(
                        f"Reported margin is "
                        f"{margin:.2f}%."
                    ),
                    category=(
                        RecommendationCategory.PROFITABILITY
                    ),
                    priority=(
                        RecommendationPriority.HIGH
                    ),
                    urgency=(
                        RecommendationUrgency.SHORT_TERM
                    ),
                    impact=(
                        RecommendationImpact.HIGH
                    ),
                    confidence=0.82,
                    risk_level=RiskLevel.HIGH,
                    expected_benefit=(
                        "Increase contribution and "
                        "operating profitability."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="financial_data",
                            metric="profit_margin",
                            value=margin,
                            description=(
                                "Reported profitability margin."
                            ),
                        )
                    ],
                )
            )

        return recommendations

    # ========================================================================
    # Liquidity Rules
    # ========================================================================

    def liquidity_rules(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        data = {
            **request.financial_data,
            **request.forecast_data,
        }

        runway = self.get_numeric(
            data,
            (
                "runway_months",
                "cash_runway_months",
            ),
        )

        cash_flow = self.get_numeric(
            data,
            (
                "cash_flow",
                "net_cash_flow",
            ),
        )

        recommendations: List[
            Recommendation
        ] = []

        if (
            runway is not None
            and runway <= 1
        ):

            recommendations.append(
                self.create_recommendation(
                    title=(
                        "Address immediate liquidity risk"
                    ),
                    action=(
                        "Review near-term cash inflows "
                        "and outflows, accelerate eligible "
                        "receivables and evaluate approved "
                        "funding or cost-control options."
                    ),
                    rationale=(
                        f"Estimated cash runway is "
                        f"{runway:.1f} month(s)."
                    ),
                    category=(
                        RecommendationCategory.LIQUIDITY
                    ),
                    priority=(
                        RecommendationPriority.CRITICAL
                    ),
                    urgency=(
                        RecommendationUrgency.IMMEDIATE
                    ),
                    impact=(
                        RecommendationImpact.VERY_HIGH
                    ),
                    confidence=0.93,
                    risk_level=RiskLevel.CRITICAL,
                    expected_benefit=(
                        "Reduce near-term liquidity "
                        "shortfall risk."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="forecast_data",
                            metric="runway_months",
                            value=runway,
                            description=(
                                "Estimated cash runway."
                            ),
                        )
                    ],
                    human_review=True,
                )
            )

        elif (
            runway is not None
            and runway <= 3
        ):

            recommendations.append(
                self.create_recommendation(
                    title=(
                        "Strengthen short-term liquidity"
                    ),
                    action=(
                        "Tighten working-capital monitoring "
                        "and review receivables, payables "
                        "and planned cash commitments."
                    ),
                    rationale=(
                        f"Estimated cash runway is "
                        f"{runway:.1f} months."
                    ),
                    category=(
                        RecommendationCategory.LIQUIDITY
                    ),
                    priority=(
                        RecommendationPriority.HIGH
                    ),
                    urgency=(
                        RecommendationUrgency.SHORT_TERM
                    ),
                    impact=(
                        RecommendationImpact.HIGH
                    ),
                    confidence=0.88,
                    risk_level=RiskLevel.HIGH,
                    expected_benefit=(
                        "Increase visibility and "
                        "resilience of short-term cash."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="forecast_data",
                            metric="runway_months",
                            value=runway,
                            description=(
                                "Estimated cash runway."
                            ),
                        )
                    ],
                )
            )

        if (
            cash_flow is not None
            and cash_flow < 0
        ):

            recommendations.append(
                self.create_recommendation(
                    title=(
                        "Investigate negative cash flow"
                    ),
                    action=(
                        "Separate operating, investing and "
                        "financing cash movements and identify "
                        "the largest recurring cash drains."
                    ),
                    rationale=(
                        f"Net cash flow is "
                        f"{cash_flow:.2f}."
                    ),
                    category=(
                        RecommendationCategory.CASH_FLOW
                    ),
                    priority=(
                        RecommendationPriority.HIGH
                    ),
                    urgency=(
                        RecommendationUrgency.SHORT_TERM
                    ),
                    impact=(
                        RecommendationImpact.HIGH
                    ),
                    confidence=0.83,
                    risk_level=RiskLevel.HIGH,
                    expected_benefit=(
                        "Improve cash-flow visibility "
                        "and reduce avoidable cash leakage."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="financial_data",
                            metric="net_cash_flow",
                            value=cash_flow,
                            description=(
                                "Reported net cash flow."
                            ),
                        )
                    ],
                )
            )

        return recommendations

    # ========================================================================
    # Forecast Rules
    # ========================================================================

    def forecast_rules(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        data = request.forecast_data

        revenue_change = self.get_numeric(
            data,
            (
                "forecast_revenue_change_pct",
                "revenue_forecast_change_pct",
            ),
        )

        profit_change = self.get_numeric(
            data,
            (
                "forecast_profit_change_pct",
                "profit_forecast_change_pct",
            ),
        )

        recommendations: List[
            Recommendation
        ] = []

        if (
            revenue_change is not None
            and revenue_change <= -20
        ):

            recommendations.append(
                self.create_recommendation(
                    title=(
                        "Prepare for forecast revenue decline"
                    ),
                    action=(
                        "Create a downside-response plan "
                        "covering sales pipeline, customer "
                        "retention, pricing and variable costs."
                    ),
                    rationale=(
                        f"Forecast revenue change is "
                        f"{revenue_change:.1f}%."
                    ),
                    category=(
                        RecommendationCategory.FORECAST
                    ),
                    priority=(
                        RecommendationPriority.HIGH
                    ),
                    urgency=(
                        RecommendationUrgency.SHORT_TERM
                    ),
                    impact=(
                        RecommendationImpact.VERY_HIGH
                    ),
                    confidence=0.82,
                    risk_level=RiskLevel.HIGH,
                    expected_benefit=(
                        "Improve preparedness for "
                        "forecast downside."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="forecast_data",
                            metric="forecast_revenue_change_pct",
                            value=revenue_change,
                            description=(
                                "Forecast revenue change."
                            ),
                        )
                    ],
                )
            )

        if (
            profit_change is not None
            and profit_change <= -20
        ):

            recommendations.append(
                self.create_recommendation(
                    title=(
                        "Build a profitability downside plan"
                    ),
                    action=(
                        "Run downside scenarios for "
                        "pricing, volume and controllable "
                        "expenses before committing resources."
                    ),
                    rationale=(
                        f"Forecast profit change is "
                        f"{profit_change:.1f}%."
                    ),
                    category=(
                        RecommendationCategory.PROFITABILITY
                    ),
                    priority=(
                        RecommendationPriority.HIGH
                    ),
                    urgency=(
                        RecommendationUrgency.SHORT_TERM
                    ),
                    impact=(
                        RecommendationImpact.VERY_HIGH
                    ),
                    confidence=0.84,
                    risk_level=RiskLevel.HIGH,
                    expected_benefit=(
                        "Improve resilience against "
                        "forecast profitability deterioration."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="forecast_data",
                            metric="forecast_profit_change_pct",
                            value=profit_change,
                            description=(
                                "Forecast profit change."
                            ),
                        )
                    ],
                )
            )

        return recommendations

    # ========================================================================
    # Fraud Rules
    # ========================================================================

    def fraud_rules(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        data = request.fraud_data

        fraud_score = self.get_numeric(
            data,
            (
                "fraud_score",
                "risk_score",
                "final_score",
            ),
        )

        suspicious_count = self.get_numeric(
            data,
            (
                "suspicious_count",
                "flagged_transactions",
                "anomaly_count",
            ),
        )

        recommendations: List[
            Recommendation
        ] = []

        if (
            fraud_score is not None
            and fraud_score >= 90
        ):

            recommendations.append(
                self.create_recommendation(
                    title=(
                        "Escalate high-risk fraud findings"
                    ),
                    action=(
                        "Route flagged transactions through "
                        "the organization's fraud-review and "
                        "human approval process."
                    ),
                    rationale=(
                        f"Fraud risk score is "
                        f"{fraud_score:.1f}/100."
                    ),
                    category=(
                        RecommendationCategory.FRAUD
                    ),
                    priority=(
                        RecommendationPriority.CRITICAL
                    ),
                    urgency=(
                        RecommendationUrgency.IMMEDIATE
                    ),
                    impact=(
                        RecommendationImpact.VERY_HIGH
                    ),
                    confidence=0.95,
                    risk_level=RiskLevel.CRITICAL,
                    expected_benefit=(
                        "Reduce exposure to potentially "
                        "fraudulent activity."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="fraud_data",
                            metric="fraud_score",
                            value=fraud_score,
                            description=(
                                "Fraud-risk score."
                            ),
                        )
                    ],
                    human_review=True,
                )
            )

        if (
            suspicious_count is not None
            and suspicious_count > 0
        ):

            recommendations.append(
                self.create_recommendation(
                    title=(
                        "Investigate suspicious transactions"
                    ),
                    action=(
                        "Review flagged transactions using "
                        "transaction history, customer context "
                        "and available fraud explanations."
                    ),
                    rationale=(
                        f"{suspicious_count:.0f} suspicious "
                        "transaction(s) were identified."
                    ),
                    category=(
                        RecommendationCategory.FRAUD
                    ),
                    priority=(
                        RecommendationPriority.HIGH
                    ),
                    urgency=(
                        RecommendationUrgency.IMMEDIATE
                    ),
                    impact=(
                        RecommendationImpact.HIGH
                    ),
                    confidence=0.88,
                    risk_level=RiskLevel.HIGH,
                    expected_benefit=(
                        "Improve fraud-detection response "
                        "and reduce unresolved exposure."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="fraud_data",
                            metric="suspicious_count",
                            value=suspicious_count,
                            description=(
                                "Number of suspicious transactions."
                            ),
                        )
                    ],
                    human_review=True,
                )
            )

        return recommendations

    # ========================================================================
    # Risk Rules
    # ========================================================================

    def risk_rules(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        data = request.risk_data

        risk_score = self.get_numeric(
            data,
            (
                "risk_score",
                "overall_risk_score",
                "score",
            ),
        )

        if risk_score is None:

            return []

        if risk_score >= 90:

            priority = (
                RecommendationPriority.CRITICAL
            )
            urgency = (
                RecommendationUrgency.IMMEDIATE
            )
            impact = (
                RecommendationImpact.VERY_HIGH
            )
            risk = RiskLevel.CRITICAL

        elif risk_score >= 70:

            priority = (
                RecommendationPriority.HIGH
            )
            urgency = (
                RecommendationUrgency.SHORT_TERM
            )
            impact = (
                RecommendationImpact.HIGH
            )
            risk = RiskLevel.HIGH

        elif risk_score >= 50:

            priority = (
                RecommendationPriority.MEDIUM
            )
            urgency = (
                RecommendationUrgency.MEDIUM_TERM
            )
            impact = (
                RecommendationImpact.MEDIUM
            )
            risk = RiskLevel.MEDIUM

        else:

            return []

        return [
            self.create_recommendation(
                title=(
                    "Address elevated financial risk"
                ),
                action=(
                    "Review the highest-contributing risk "
                    "drivers and assign mitigation owners "
                    "with measurable follow-up actions."
                ),
                rationale=(
                    f"Overall risk score is "
                    f"{risk_score:.1f}/100."
                ),
                category=(
                    RecommendationCategory.RISK
                ),
                priority=priority,
                urgency=urgency,
                impact=impact,
                confidence=0.86,
                risk_level=risk,
                expected_benefit=(
                    "Reduce the organization's exposure "
                    "to identified financial risks."
                ),
                evidence=[
                    self.numeric_evidence(
                        source="risk_data",
                        metric="risk_score",
                        value=risk_score,
                        description=(
                            "Overall risk score."
                        ),
                    )
                ],
                human_review=(
                    risk_score >= 70
                ),
            )
        ]

    # ========================================================================
    # What-If Rules
    # ========================================================================

    def what_if_rules(
        self,
        request: RecommendationAgentRequest,
    ) -> List[Recommendation]:

        data = request.what_if_data

        impact = self.get_numeric(
            data,
            (
                "profit_impact",
                "cash_flow_impact",
                "financial_impact",
                "impact_pct",
            ),
        )

        if impact is None:

            return []

        if impact < 0:

            return [
                self.create_recommendation(
                    title=(
                        "Review downside scenario before execution"
                    ),
                    action=(
                        "Compare the downside scenario against "
                        "base and upside cases and identify "
                        "mitigation actions before implementation."
                    ),
                    rationale=(
                        f"The evaluated scenario has an "
                        f"estimated impact of {impact:.2f}."
                    ),
                    category=(
                        RecommendationCategory.STRATEGIC
                    ),
                    priority=(
                        RecommendationPriority.HIGH
                    ),
                    urgency=(
                        RecommendationUrgency.SHORT_TERM
                    ),
                    impact=(
                        RecommendationImpact.HIGH
                    ),
                    confidence=0.78,
                    risk_level=RiskLevel.HIGH,
                    expected_benefit=(
                        "Reduce exposure to an unfavorable "
                        "scenario outcome."
                    ),
                    evidence=[
                        self.numeric_evidence(
                            source="what_if_data",
                            metric="scenario_impact",
                            value=impact,
                            description=(
                                "Scenario impact."
                            ),
                        )
                    ],
                    human_review=True,
                )
            ]

        return []

    # ========================================================================
    # LLM Recommendations
    # ========================================================================

    def generate_llm_recommendations(
        self,
        request: RecommendationAgentRequest,
        existing: Sequence[
            Recommendation
        ],
    ) -> Any:

        payload = {
            "query": request.query,
            "financial_data": request.financial_data,
            "forecast_data": request.forecast_data,
            "fraud_data": request.fraud_data,
            "risk_data": request.risk_data,
            "what_if_data": request.what_if_data,
            "rag_data": request.rag_data,
            "context": request.context,
            "existing_recommendations": [
                item.to_dict()
                for item in existing
            ],
            "instruction": (
                "Generate only evidence-supported "
                "decision-support recommendations. "
                "Do not invent financial facts. "
                "Do not execute or authorize actions. "
                "Clearly state assumptions and uncertainty. "
                "Recommendations involving financial "
                "commitments require human review."
            ),
        }

        try:

            return self.invoke_component(
                self.generator,
                (
                    "generate",
                    "recommend",
                    "generate_recommendations",
                    "run",
                ),
                payload,
            )

        except Exception as exc:

            raise RecommendationGenerationError(
                "LLM recommendation generation failed."
            ) from exc

    # ========================================================================
    # Recommendation Construction
    # ========================================================================

    def create_recommendation(
        self,
        *,
        title: str,
        action: str,
        rationale: str,
        category: RecommendationCategory,
        priority: RecommendationPriority,
        urgency: RecommendationUrgency,
        impact: RecommendationImpact,
        confidence: float,
        risk_level: RiskLevel = RiskLevel.LOW,
        expected_benefit: Optional[str] = None,
        impact_estimate: Optional[
            RecommendationImpactEstimate
        ] = None,
        evidence: Optional[
            Sequence[RecommendationEvidence]
        ] = None,
        dependencies: Optional[
            Sequence[str]
        ] = None,
        assumptions: Optional[
            Sequence[str]
        ] = None,
        risks: Optional[
            Sequence[str]
        ] = None,
        human_review: bool = False,
        executable: bool = False,
    ) -> Recommendation:

        requires_review = (
            human_review
            or (
                self.config.require_human_review_for_high_impact
                and impact
                in {
                    RecommendationImpact.HIGH,
                    RecommendationImpact.VERY_HIGH,
                }
            )
            or (
                self.config.require_human_review_for_financial_actions
                and self.looks_like_financial_action(
                    action
                )
            )
        )

        status = (
            RecommendationStatus.REVIEW_REQUIRED
            if requires_review
            else RecommendationStatus.PROPOSED
        )

        return Recommendation(
            recommendation_id=(
                "rec_"
                + uuid.uuid4().hex[:16]
            ),
            title=title.strip(),
            action=action.strip(),
            rationale=rationale.strip(),
            category=category,
            priority=priority,
            urgency=urgency,
            impact=impact,
            confidence=max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            ),
            risk_level=risk_level,
            status=status,
            expected_benefit=expected_benefit,
            impact_estimate=impact_estimate,
            evidence=list(
                evidence or []
            ),
            dependencies=list(
                dependencies or []
            ),
            assumptions=list(
                assumptions or []
            ),
            risks=list(
                risks or []
            ),
            human_review_required=(
                requires_review
            ),
            executable=(
                executable
                and not requires_review
            ),
        )

    def numeric_evidence(
        self,
        *,
        source: str,
        metric: str,
        value: Any,
        description: str,
        confidence: float = 0.90,
    ) -> RecommendationEvidence:

        return RecommendationEvidence(
            evidence_id=(
                "evidence_"
                + uuid.uuid4().hex[:12]
            ),
            source=source,
            description=description,
            confidence=confidence,
            metric=metric,
            value=value,
        )

    # ========================================================================
    # Normalization
    # ========================================================================

    def normalize_recommendations(
        self,
        raw: Any,
    ) -> List[Recommendation]:

        if raw is None:

            return []

        if isinstance(
            raw,
            Mapping,
        ):

            for key in (
                "recommendations",
                "results",
                "items",
                "data",
            ):

                if key in raw:

                    raw = raw[key]
                    break

        if isinstance(
            raw,
            Recommendation,
        ):

            return [
                raw
            ]

        if isinstance(
            raw,
            str,
        ):

            return []

        if not isinstance(
            raw,
            Sequence,
        ):

            raw = [
                raw
            ]

        result: List[
            Recommendation
        ] = []

        for item in raw:

            if isinstance(
                item,
                Recommendation,
            ):

                result.append(
                    item
                )
                continue

            data = (
                dict(item)
                if isinstance(
                    item,
                    Mapping,
                )
                else self.object_to_dict(
                    item
                )
            )

            title = (
                data.get(
                    "title"
                )
                or data.get(
                    "name"
                )
                or "Business recommendation"
            )

            action = (
                data.get(
                    "action"
                )
                or data.get(
                    "recommendation"
                )
                or data.get(
                    "description"
                )
                or ""
            )

            if not action:
                continue

            result.append(
                self.create_recommendation(
                    title=str(title),
                    action=str(action),
                    rationale=str(
                        data.get(
                            "rationale"
                        )
                        or data.get(
                            "reason"
                        )
                        or "Generated from available analysis."
                    ),
                    category=self.parse_enum(
                        data.get(
                            "category"
                        ),
                        RecommendationCategory,
                        RecommendationCategory.STRATEGIC,
                    ),
                    priority=self.parse_enum(
                        data.get(
                            "priority"
                        ),
                        RecommendationPriority,
                        RecommendationPriority.MEDIUM,
                    ),
                    urgency=self.parse_enum(
                        data.get(
                            "urgency"
                        ),
                        RecommendationUrgency,
                        RecommendationUrgency.MEDIUM_TERM,
                    ),
                    impact=self.parse_enum(
                        data.get(
                            "impact"
                        ),
                        RecommendationImpact,
                        RecommendationImpact.MEDIUM,
                    ),
                    confidence=self.safe_float(
                        data.get(
                            "confidence",
                            0.60,
                        )
                    ),
                    risk_level=self.parse_enum(
                        data.get(
                            "risk_level"
                        ),
                        RiskLevel,
                        RiskLevel.MEDIUM,
                    ),
                    expected_benefit=(
                        data.get(
                            "expected_benefit"
                        )
                    ),
                    assumptions=(
                        data.get(
                            "assumptions"
                        )
                        or []
                    ),
                    risks=(
                        data.get(
                            "risks"
                        )
                        or []
                    ),
                    human_review=bool(
                        data.get(
                            "human_review_required",
                            False,
                        )
                    ),
                )
            )

        return result

    # ========================================================================
    # Validation
    # ========================================================================

    def validate_recommendations(
        self,
        recommendations: Sequence[
            Recommendation
        ],
    ) -> List[Recommendation]:

        valid: List[
            Recommendation
        ] = []

        for recommendation in recommendations:

            if not recommendation.title.strip():

                continue

            if not recommendation.action.strip():

                continue

            if (
                recommendation.confidence
                < self.config.min_confidence
            ):

                continue

            recommendation.confidence = max(
                0.0,
                min(
                    1.0,
                    recommendation.confidence,
                ),
            )

            # Recommendations must not claim to
            # have executed financial actions.
            if self.looks_like_execution_claim(
                recommendation.action
            ):

                recommendation.warnings = (
                    getattr(
                        recommendation,
                        "warnings",
                        [],
                    )
                )

                recommendation.action = (
                    "Review and, if approved, "
                    + recommendation.action
                )

                recommendation.human_review_required = True
                recommendation.status = (
                    RecommendationStatus.REVIEW_REQUIRED
                )
                recommendation.executable = False

            valid.append(
                recommendation
            )

        return valid

    # ========================================================================
    # Prioritization
    # ========================================================================

    def prioritize(
        self,
        recommendations: Sequence[
            Recommendation
        ],
    ) -> List[Recommendation]:

        def score(
            item: Recommendation,
        ) -> float:

            priority_score = {
                RecommendationPriority.LOW: 1.0,
                RecommendationPriority.MEDIUM: 2.0,
                RecommendationPriority.HIGH: 3.0,
                RecommendationPriority.CRITICAL: 4.0,
            }[
                item.priority
            ]

            impact_score = {
                RecommendationImpact.LOW: 1.0,
                RecommendationImpact.MEDIUM: 2.0,
                RecommendationImpact.HIGH: 3.0,
                RecommendationImpact.VERY_HIGH: 4.0,
            }[
                item.impact
            ]

            urgency_score = {
                RecommendationUrgency.LONG_TERM: 1.0,
                RecommendationUrgency.MEDIUM_TERM: 2.0,
                RecommendationUrgency.SHORT_TERM: 3.0,
                RecommendationUrgency.IMMEDIATE: 4.0,
            }[
                item.urgency
            ]

            risk_score = {
                RiskLevel.LOW: 1.0,
                RiskLevel.MEDIUM: 2.0,
                RiskLevel.HIGH: 3.0,
                RiskLevel.CRITICAL: 4.0,
            }[
                item.risk_level
            ]

            return (
                0.30 * priority_score
                + 0.25 * impact_score
                + 0.20 * urgency_score
                + 0.15 * risk_score
                + 0.10 * (
                    item.confidence * 4
                )
            )

        return sorted(
            recommendations,
            key=score,
            reverse=True,
        )

    def deduplicate_recommendations(
        self,
        recommendations: Sequence[
            Recommendation
        ],
    ) -> List[Recommendation]:

        result: List[
            Recommendation
        ] = []

        seen: set[
            str
        ] = set()

        for item in recommendations:

            fingerprint = self.fingerprint(
                item.title,
                item.action,
            )

            if fingerprint in seen:
                continue

            seen.add(
                fingerprint
            )

            result.append(
                item
            )

        return result

    # ========================================================================
    # Action Plan
    # ========================================================================

    def build_action_plan(
        self,
        recommendations: Sequence[
            Recommendation
        ],
    ) -> List[str]:

        actions: List[
            str
        ] = []

        for index, item in enumerate(
            recommendations,
            start=1,
        ):

            review = (
                " [Human review required]"
                if item.human_review_required
                else ""
            )

            actions.append(
                f"{index}. {item.action}{review}"
            )

        return actions

    def build_summary(
        self,
        recommendations: Sequence[
            Recommendation
        ],
    ) -> str:

        if not recommendations:

            return (
                "No recommendation met the configured "
                "confidence threshold."
            )

        critical = sum(
            item.priority
            == RecommendationPriority.CRITICAL
            for item in recommendations
        )

        high = sum(
            item.priority
            == RecommendationPriority.HIGH
            for item in recommendations
        )

        review = sum(
            item.human_review_required
            for item in recommendations
        )

        parts = [
            f"{len(recommendations)} recommendation(s) generated."
        ]

        if critical:
            parts.append(
                f"{critical} critical."
            )

        if high:
            parts.append(
                f"{high} high priority."
            )

        if review:
            parts.append(
                f"{review} require human review."
            )

        top = recommendations[0]

        parts.append(
            f"Top priority: {top.title}."
        )

        return " ".join(
            parts
        )

    # ========================================================================
    # Confidence / Risk
    # ========================================================================

    def calculate_confidence(
        self,
        recommendations: Sequence[
            Recommendation
        ],
    ) -> float:

        if not recommendations:

            return 0.0

        return sum(
            item.confidence
            for item in recommendations
        ) / len(
            recommendations
        )

    def calculate_risk_level(
        self,
        recommendations: Sequence[
            Recommendation
        ],
    ) -> RiskLevel:

        if not recommendations:

            return RiskLevel.LOW

        levels = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }

        highest = max(
            recommendations,
            key=lambda item: levels[
                item.risk_level
            ],
        )

        return highest.risk_level

    # ========================================================================
    # Authorization / Guardrails
    # ========================================================================

    def authorize(
        self,
        request: RecommendationAgentRequest,
    ) -> None:

        if self.authorization_service is None:

            return

        payload = {
            "user_id": request.user_id,
            "company_id": request.company_id,
            "operation": request.operation.value,
        }

        try:

            result = self.invoke_component(
                self.authorization_service,
                (
                    "authorize",
                    "check_permission",
                    "can_access",
                    "has_permission",
                    "check",
                ),
                payload,
            )

        except Exception as exc:

            raise RecommendationAccessDeniedError(
                "Recommendation authorization failed."
            ) from exc

        if result is False:

            raise RecommendationAccessDeniedError(
                "Access to recommendation service denied."
            )

        if (
            isinstance(
                result,
                Mapping,
            )
            and result.get(
                "allowed"
            ) is False
        ):

            raise RecommendationAccessDeniedError(
                "Access to recommendation service denied."
            )

    def check_guardrails(
        self,
        request: RecommendationAgentRequest,
    ) -> None:

        if self.guardrails is None:

            return

        payload = {
            "query": request.query,
            "operation": request.operation.value,
            "company_id": request.company_id,
            "context": request.context,
        }

        try:

            result = self.invoke_component(
                self.guardrails,
                (
                    "check",
                    "validate",
                    "evaluate",
                    "run",
                ),
                payload,
            )

        except Exception as exc:

            raise RecommendationGuardrailViolationError(
                "Recommendation guardrail check failed."
            ) from exc

        if result is False:

            raise RecommendationGuardrailViolationError(
                "Recommendation request was blocked."
            )

        if (
            isinstance(
                result,
                Mapping,
            )
            and result.get(
                "allowed"
            ) is False
        ):

            raise RecommendationGuardrailViolationError(
                str(
                    result.get(
                        "reason"
                    )
                    or "Recommendation request was blocked."
                )
            )

    # ========================================================================
    # Request Handling
    # ========================================================================

    def normalize_request(
        self,
        request: RecommendationAgentRequest
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> RecommendationAgentRequest:

        if isinstance(
            request,
            RecommendationAgentRequest,
        ):

            return request

        if not isinstance(
            request,
            Mapping,
        ):

            raise InvalidRecommendationRequestError(
                "Request must be a "
                "RecommendationAgentRequest or mapping."
            )

        payload = dict(
            request
        )

        payload.update(
            kwargs
        )

        if "operation" in payload:

            try:

                payload["operation"] = (
                    RecommendationOperation(
                        payload["operation"]
                    )
                )

            except ValueError as exc:

                raise InvalidRecommendationRequestError(
                    "Unsupported recommendation operation."
                ) from exc

        return RecommendationAgentRequest(
            **payload
        )

    def validate_request(
        self,
        request: RecommendationAgentRequest,
    ) -> None:

        if not self.config.enabled:

            raise RecommendationConfigurationError(
                "Recommendation agent is disabled."
            )

        if not isinstance(
            request.query,
            str,
        ):

            raise InvalidRecommendationRequestError(
                "Query must be a string."
            )

        request.query = (
            request.query.strip()
        )

        if not request.query:

            raise InvalidRecommendationRequestError(
                "Query cannot be empty."
            )

        if len(
            request.query
        ) > self.config.max_input_length:

            raise InvalidRecommendationRequestError(
                "Recommendation input exceeds "
                "maximum length."
            )

    # ========================================================================
    # Payload Construction
    # ========================================================================

    def build_service_payload(
        self,
        request: RecommendationAgentRequest,
    ) -> Dict[str, Any]:

        return {
            "query": request.query,
            "operation": request.operation.value,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "workflow_id": request.workflow_id,
            "financial_data": request.financial_data,
            "forecast_data": request.forecast_data,
            "fraud_data": request.fraud_data,
            "risk_data": request.risk_data,
            "what_if_data": request.what_if_data,
            "rag_data": request.rag_data,
            "previous_results": request.previous_results,
            "context": request.context,
            "metadata": request.metadata,
        }

    # ========================================================================
    # Result Finalization
    # ========================================================================

    def finalize_result(
        self,
        result: RecommendationAgentResult,
        started: float,
    ) -> None:

        result.execution_time_ms = (
            time.perf_counter()
            - started
        ) * 1000

        if (
            result.execution_time_ms
            > self.config.timeout_seconds * 1000
        ):

            result.warnings.append(
                "Recommendation generation exceeded "
                "the configured latency target."
            )

    # ========================================================================
    # Audit
    # ========================================================================

    def audit(
        self,
        request: RecommendationAgentRequest,
        result: RecommendationAgentResult,
    ) -> None:

        if (
            not self.config.audit_enabled
            or self.audit_service is None
        ):

            return

        payload = {
            "event": (
                "recommendation_agent_execution"
            ),
            "request_id": result.request_id,
            "operation": request.operation.value,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "success": result.success,
            "recommendation_count": len(
                result.recommendations
            ),
            "human_review_required": (
                result.human_review_required
            ),
            "risk_level": result.risk_level.value,
            "confidence": result.confidence,
        }

        try:

            self.invoke_component(
                self.audit_service,
                (
                    "record",
                    "log",
                    "create",
                    "write",
                ),
                payload,
            )

        except Exception:
            pass

    # ========================================================================
    # Generic Component Invocation
    # ========================================================================

    def invoke_component(
        self,
        component: Any,
        methods: Sequence[str],
        payload: Mapping[str, Any],
    ) -> Any:

        if component is None:

            raise RecommendationConfigurationError(
                "Required component is not configured."
            )

        if callable(component):

            return self.invoke_callable(
                component,
                payload,
            )

        for method_name in methods:

            method = getattr(
                component,
                method_name,
                None,
            )

            if method is None:
                continue

            if not callable(method):
                continue

            return self.invoke_callable(
                method,
                payload,
            )

        raise RecommendationConfigurationError(
            "Configured component does not expose "
            "a supported interface."
        )

    @staticmethod
    def invoke_callable(
        callable_obj: Callable[..., Any],
        payload: Mapping[str, Any],
    ) -> Any:

        try:

            signature = inspect.signature(
                callable_obj
            )

            parameters = signature.parameters

            accepts_kwargs = any(
                parameter.kind
                == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )

            if accepts_kwargs:

                return callable_obj(
                    **dict(payload)
                )

            accepted = {
                name: value
                for name, value in payload.items()
                if name in parameters
            }

            if accepted:

                return callable_obj(
                    **accepted
                )

        except (
            TypeError,
            ValueError,
        ):

            pass

        return callable_obj(
            payload
        )

    # ========================================================================
    # Utility Helpers
    # ========================================================================

    @staticmethod
    def get_numeric(
        data: Mapping[str, Any],
        keys: Sequence[str],
    ) -> Optional[float]:

        for key in keys:

            if key not in data:
                continue

            value = data.get(
                key
            )

            if value is None:
                continue

            try:

                return float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

        return None

    @staticmethod
    def safe_float(
        value: Any,
    ) -> float:

        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    @staticmethod
    def object_to_dict(
        obj: Any,
    ) -> Dict[str, Any]:

        if isinstance(
            obj,
            Mapping,
        ):

            return dict(
                obj
            )

        if hasattr(
            obj,
            "model_dump",
        ):

            try:

                return dict(
                    obj.model_dump()
                )

            except Exception:
                pass

        if hasattr(
            obj,
            "dict",
        ):

            try:

                return dict(
                    obj.dict()
                )

            except Exception:
                pass

        if hasattr(
            obj,
            "__dict__",
        ):

            return dict(
                obj.__dict__
            )

        return {}

    @staticmethod
    def parse_enum(
        value: Any,
        enum_cls: Any,
        default: Any,
    ) -> Any:

        if isinstance(
            value,
            enum_cls,
        ):

            return value

        if value is None:

            return default

        try:

            return enum_cls(
                str(
                    value
                ).lower()
            )

        except (
            ValueError,
            TypeError,
        ):

            return default

    @staticmethod
    def fingerprint(
        title: str,
        action: str,
    ) -> str:

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            f"{title} {action}".lower(),
        ).strip()

        return normalized

    @staticmethod
    def looks_like_execution_claim(
        action: str,
    ) -> bool:

        patterns = (
            r"\bexecuted\b",
            r"\bcompleted\b",
            r"\btransferred\b",
            r"\bpaid\b",
            r"\bapproved\b",
            r"\bchanged\b",
            r"\bupdated the account\b",
            r"\bmodified the financial record\b",
        )

        lowered = action.lower()

        return any(
            re.search(
                pattern,
                lowered,
            )
            for pattern in patterns
        )

    @staticmethod
    def looks_like_financial_action(
        action: str,
    ) -> bool:

        financial_terms = (
            "transfer",
            "borrow",
            "loan",
            "invest",
            "payment",
            "pay",
            "financing",
            "funding",
            "purchase",
            "sell",
            "dividend",
            "capital expenditure",
            "credit",
            "debt",
        )

        lowered = action.lower()

        return any(
            term in lowered
            for term in financial_terms
        )

    @staticmethod
    def generate_request_id() -> str:

        return (
            "rec_agent_"
            + uuid.uuid4().hex[:20]
        )


# ============================================================================
# Serialization
# ============================================================================


def _serialize(
    value: Any,
) -> Any:

    if isinstance(
        value,
        Enum,
    ):

        return value.value

    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(key): _serialize(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):

        return [
            _serialize(
                item
            )
            for item in value
        ]

    if hasattr(
        value,
        "model_dump",
    ):

        try:

            return _serialize(
                value.model_dump()
            )

        except Exception:
            pass

    if hasattr(
        value,
        "to_dict",
    ):

        try:

            return _serialize(
                value.to_dict()
            )

        except Exception:
            pass

    return value


# ============================================================================
# Factory
# ============================================================================


def create_recommendation_agent(
    recommendation_service: Optional[
        Any
    ] = None,
    **kwargs: Any,
) -> RecommendationAgent:

    return RecommendationAgent(
        recommendation_service=recommendation_service,
        **kwargs,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "RecommendationAgentError",
    "InvalidRecommendationRequestError",
    "RecommendationConfigurationError",
    "RecommendationExecutionError",
    "RecommendationAccessDeniedError",
    "RecommendationGuardrailViolationError",
    "RecommendationGenerationError",
    "RecommendationValidationError",

    # Enums
    "RecommendationOperation",
    "RecommendationCategory",
    "RecommendationPriority",
    "RecommendationUrgency",
    "RecommendationImpact",
    "RecommendationStatus",
    "RiskLevel",

    # Configuration
    "RecommendationAgentConfig",

    # Models
    "RecommendationAgentRequest",
    "RecommendationEvidence",
    "RecommendationImpactEstimate",
    "Recommendation",
    "RecommendationAgentResult",

    # Agent
    "RecommendationAgent",

    # Factory
    "create_recommendation_agent",
]