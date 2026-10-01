"""
FinCo AI - Recommendation Schemas

Pydantic schemas for:

- AI-generated recommendations
- Financial recommendations
- Risk recommendations
- Cost optimization
- Recommendation priority
- Evidence and reasoning
- Expected impact
- Human approval
- Recommendation execution
- Recommendation feedback
- Recommendation dashboard
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# ENUMS
# ============================================================

class RecommendationType(str, Enum):
    """Recommendation categories."""

    FINANCIAL = "financial"
    RISK = "risk"
    COST = "cost"
    REVENUE = "revenue"
    CASH_FLOW = "cash_flow"
    LIQUIDITY = "liquidity"
    FRAUD = "fraud"
    FORECAST = "forecast"
    BUDGET = "budget"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    GENERAL = "general"


class RecommendationPriority(str, Enum):
    """Recommendation priority."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationStatus(str, Enum):
    """Recommendation lifecycle status."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RecommendationAction(str, Enum):
    """Recommended action type."""

    MONITOR = "monitor"
    REVIEW = "review"
    INVESTIGATE = "investigate"
    REDUCE = "reduce"
    INCREASE = "increase"
    OPTIMIZE = "optimize"
    RESTRUCTURE = "restructure"
    FREEZE = "freeze"
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    FORECAST = "forecast"
    SIMULATE = "simulate"
    OTHER = "other"


class ImpactDirection(str, Enum):
    """Expected impact direction."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


# ============================================================
# RECOMMENDATION BASE
# ============================================================

class RecommendationBase(BaseModel):
    """Common recommendation fields."""

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )

    company_id: int = Field(
        ...,
        gt=0,
    )

    recommendation_type: RecommendationType

    title: str = Field(
        ...,
        min_length=3,
        max_length=300,
    )

    description: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )


# ============================================================
# EXPECTED IMPACT
# ============================================================

class RecommendationImpact(BaseModel):
    """Expected business and financial impact."""

    direction: ImpactDirection = ImpactDirection.UNKNOWN

    expected_revenue_impact: Optional[Decimal] = None

    expected_cost_savings: Optional[Decimal] = None

    expected_profit_impact: Optional[Decimal] = None

    expected_cashflow_impact: Optional[Decimal] = None

    expected_risk_reduction: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    expected_roi: Optional[float] = None

    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    timeframe: Optional[str] = None

    explanation: Optional[str] = None


# ============================================================
# RECOMMENDATION EVIDENCE
# ============================================================

class RecommendationEvidence(BaseModel):
    """Evidence supporting a recommendation."""

    source_type: str

    source_id: Optional[str] = None

    title: Optional[str] = None

    content: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    relevance_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    citation: Optional[str] = None

    metadata: Dict[str, object] = Field(
        default_factory=dict,
    )


# ============================================================
# RECOMMENDATION REASONING
# ============================================================

class RecommendationReasoning(BaseModel):
    """AI reasoning and decision factors."""

    summary: Optional[str] = None

    key_findings: List[str] = Field(
        default_factory=list,
    )

    root_causes: List[str] = Field(
        default_factory=list,
    )

    risk_factors: List[str] = Field(
        default_factory=list,
    )

    assumptions: List[str] = Field(
        default_factory=list,
    )

    constraints: List[str] = Field(
        default_factory=list,
    )

    reasoning_steps: List[str] = Field(
        default_factory=list,
    )


# ============================================================
# RECOMMENDATION CREATE
# ============================================================

class RecommendationCreate(RecommendationBase):
    """Create a recommendation."""

    priority: RecommendationPriority = RecommendationPriority.MEDIUM

    action: RecommendationAction = RecommendationAction.REVIEW

    impact: Optional[RecommendationImpact] = None

    reasoning: Optional[RecommendationReasoning] = None

    evidence: List[RecommendationEvidence] = Field(
        default_factory=list,
    )

    recommended_steps: List[str] = Field(
        default_factory=list,
    )

    requires_human_review: bool = True

    auto_execute: bool = False

    expires_at: Optional[datetime] = None

    metadata: Dict[str, object] = Field(
        default_factory=dict,
    )


# ============================================================
# RECOMMENDATION RESPONSE
# ============================================================

class RecommendationResponse(RecommendationCreate):
    """Recommendation API response."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    status: RecommendationStatus

    confidence_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    created_at: datetime

    updated_at: Optional[datetime] = None

    approved_at: Optional[datetime] = None

    completed_at: Optional[datetime] = None


# ============================================================
# FINANCIAL RECOMMENDATION
# ============================================================

class FinancialRecommendation(BaseModel):
    """Financial-specific recommendation."""

    company_id: int

    title: str

    issue: str

    action: RecommendationAction

    priority: RecommendationPriority

    current_metric: Optional[Decimal] = None

    target_metric: Optional[Decimal] = None

    expected_impact: Optional[RecommendationImpact] = None

    rationale: Optional[str] = None

    evidence: List[RecommendationEvidence] = Field(
        default_factory=list,
    )

    next_steps: List[str] = Field(
        default_factory=list,
    )


# ============================================================
# RISK RECOMMENDATION
# ============================================================

class RiskRecommendation(BaseModel):
    """Risk mitigation recommendation."""

    company_id: int

    risk_type: str

    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    priority: RecommendationPriority

    title: str

    description: str

    mitigation_action: RecommendationAction

    mitigation_steps: List[str] = Field(
        default_factory=list,
    )

    expected_risk_reduction: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    evidence: List[RecommendationEvidence] = Field(
        default_factory=list,
    )

    requires_human_review: bool = True


# ============================================================
# COST RECOMMENDATION
# ============================================================

class CostRecommendation(BaseModel):
    """Cost optimization recommendation."""

    company_id: int

    category: str

    current_cost: Decimal = Field(
        ...,
        ge=0,
    )

    target_cost: Optional[Decimal] = Field(
        default=None,
        ge=0,
    )

    potential_savings: Optional[Decimal] = Field(
        default=None,
        ge=0,
    )

    savings_percentage: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    priority: RecommendationPriority

    action: RecommendationAction

    title: str

    explanation: Optional[str] = None

    implementation_steps: List[str] = Field(
        default_factory=list,
    )

    risks: List[str] = Field(
        default_factory=list,
    )


# ============================================================
# RECOMMENDATION ACTION
# ============================================================

class RecommendationActionRequest(BaseModel):
    """Approve/reject/execute recommendation."""

    recommendation_id: int = Field(
        ...,
        gt=0,
    )

    action: RecommendationStatus

    comment: Optional[str] = Field(
        default=None,
        max_length=3000,
    )


# ============================================================
# RECOMMENDATION FEEDBACK
# ============================================================

class RecommendationFeedback(BaseModel):
    """Human feedback on recommendation quality."""

    recommendation_id: int = Field(
        ...,
        gt=0,
    )

    rating: int = Field(
        ...,
        ge=1,
        le=5,
    )

    useful: Optional[bool] = None

    comment: Optional[str] = Field(
        default=None,
        max_length=3000,
    )

    actual_impact: Optional[Decimal] = None

    feedback_date: datetime


# ============================================================
# RECOMMENDATION EXECUTION
# ============================================================

class RecommendationExecution(BaseModel):
    """Execution tracking."""

    recommendation_id: int

    status: RecommendationStatus

    started_at: Optional[datetime] = None

    completed_at: Optional[datetime] = None

    executed_by: Optional[int] = None

    execution_notes: Optional[str] = None

    actual_revenue_impact: Optional[Decimal] = None

    actual_cost_savings: Optional[Decimal] = None

    actual_profit_impact: Optional[Decimal] = None

    actual_risk_reduction: Optional[float] = None

    success: Optional[bool] = None


# ============================================================
# RECOMMENDATION SUMMARY
# ============================================================

class RecommendationSummary(BaseModel):
    """Executive recommendation summary."""

    company_id: int

    total_recommendations: int = 0

    pending_review: int = 0

    approved: int = 0

    in_progress: int = 0

    completed: int = 0

    rejected: int = 0

    critical: int = 0

    high_priority: int = 0

    estimated_cost_savings: Decimal = Decimal("0")

    estimated_profit_impact: Decimal = Decimal("0")

    estimated_risk_reduction: Optional[float] = None


# ============================================================
# RECOMMENDATION DASHBOARD
# ============================================================

class RecommendationDashboardResponse(BaseModel):
    """Recommendation dashboard."""

    company_id: int

    summary: RecommendationSummary

    recommendations: List[RecommendationResponse] = Field(
        default_factory=list,
    )

    financial_recommendations: List[FinancialRecommendation] = Field(
        default_factory=list,
    )

    risk_recommendations: List[RiskRecommendation] = Field(
        default_factory=list,
    )

    cost_recommendations: List[CostRecommendation] = Field(
        default_factory=list,
    )

    generated_at: datetime


# ============================================================
# AI RECOMMENDATION REQUEST
# ============================================================

class AIRecommendationRequest(BaseModel):
    """Request recommendations from FinCo AI."""

    company_id: int = Field(
        ...,
        gt=0,
    )

    query: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    recommendation_types: List[RecommendationType] = Field(
        default_factory=list,
    )

    include_financial_analysis: bool = True

    include_risk_analysis: bool = True

    include_forecast: bool = True

    include_rag_evidence: bool = True

    include_what_if: bool = False

    max_recommendations: int = Field(
        default=10,
        ge=1,
        le=50,
    )


# ============================================================
# AI RECOMMENDATION RESPONSE
# ============================================================

class AIRecommendationResponse(BaseModel):
    """AI-generated recommendation response."""

    company_id: int

    summary: str

    recommendations: List[RecommendationResponse] = Field(
        default_factory=list,
    )

    key_findings: List[str] = Field(
        default_factory=list,
    )

    financial_impact: Optional[RecommendationImpact] = None

    overall_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    human_review_required: bool = True

    generated_at: datetime


# ============================================================
# RECOMMENDATION PRIORITY SCORE
# ============================================================

class RecommendationPriorityScore(BaseModel):
    """Calculated recommendation priority."""

    urgency_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    financial_impact_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    confidence_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    total_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    priority: RecommendationPriority


# ============================================================
# RECOMMENDATION GENERATION RESULT
# ============================================================

class RecommendationGenerationResult(BaseModel):
    """Internal recommendation generation result."""

    recommendations: List[RecommendationCreate] = Field(
        default_factory=list,
    )

    priority_scores: List[RecommendationPriorityScore] = Field(
        default_factory=list,
    )

    reasoning: Optional[RecommendationReasoning] = None

    evidence_count: int = 0

    generation_time_seconds: Optional[float] = None

    model_name: Optional[str] = None

    model_version: Optional[str] = None