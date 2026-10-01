"""
backend/app/api/recommendations.py

Recommendation API for FinCo AI.

Responsibilities:
    - Request validation
    - Authentication / authorization
    - Tenant isolation
    - Recommendation-service orchestration
    - Response serialization
    - Pagination / filtering
    - Safe error handling

Architecture:

    Client
       |
       v
    FastAPI API
       |
       v
    RecommendationService
       |
       +---- Financial Analysis
       +---- Forecasting
       +---- Risk/Fraud
       +---- What-If Analysis
       +---- RAG Evidence
       +---- Business Rules
       |
       v
    Recommendation
       |
       v
    Human Review / Action
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
)


# ============================================================================
# ENUMS
# ============================================================================


class RecommendationType(str, Enum):
    """
    Business category of a recommendation.
    """

    COST_REDUCTION = "cost_reduction"
    REVENUE_GROWTH = "revenue_growth"
    CASH_FLOW = "cash_flow"
    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    RISK_MITIGATION = "risk_mitigation"
    FRAUD_PREVENTION = "fraud_prevention"
    BUDGET_OPTIMIZATION = "budget_optimization"
    WORKING_CAPITAL = "working_capital"
    PRICING = "pricing"
    COLLECTIONS = "collections"
    EXPENSE_OPTIMIZATION = "expense_optimization"
    INVESTMENT = "investment"
    GENERAL = "general"


class RecommendationPriority(str, Enum):
    """
    Business priority.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationStatus(str, Enum):
    """
    Lifecycle status.
    """

    GENERATED = "generated"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DISMISSED = "dismissed"


class RecommendationConfidence(str, Enum):
    """
    Confidence classification.
    """

    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationSource(str, Enum):
    """
    Main source that contributed to a recommendation.
    """

    RULE = "rule"
    ML = "ml"
    FORECAST = "forecast"
    RAG = "rag"
    AGENT = "agent"
    HYBRID = "hybrid"


# ============================================================================
# COMMON SCHEMAS
# ============================================================================


class EvidenceItem(BaseModel):
    """
    Evidence supporting a recommendation.

    Evidence should be traceable to an actual financial,
    forecast, transaction, or document source.
    """

    model_config = ConfigDict(extra="forbid")

    source_type: str = Field(
        ...,
        description="Evidence category, e.g. financial_metric, forecast, document.",
    )

    source_id: str | None = Field(
        default=None,
        description="Identifier of the source record/document.",
    )

    title: str | None = None

    description: str

    metric: str | None = None

    value: float | None = None

    unit: str | None = None

    period: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    citation: str | None = None


class RecommendationImpact(BaseModel):
    """
    Estimated business impact.
    """

    model_config = ConfigDict(extra="forbid")

    expected_revenue_change: float | None = None

    expected_cost_change: float | None = None

    expected_profit_change: float | None = None

    expected_cashflow_change: float | None = None

    expected_margin_change: float | None = None

    expected_risk_reduction: float | None = None

    currency: str | None = None

    time_horizon_days: int | None = Field(
        default=None,
        ge=1,
    )

    assumptions: list[str] = Field(default_factory=list)


class RecommendationAction(BaseModel):
    """
    Concrete action proposed by the recommendation engine.
    """

    model_config = ConfigDict(extra="forbid")

    action_id: str

    title: str

    description: str

    sequence: int = Field(
        ...,
        ge=1,
    )

    owner: str | None = None

    estimated_effort: str | None = None

    expected_outcome: str | None = None

    requires_approval: bool = True


# ============================================================================
# REQUEST MODELS
# ============================================================================


class RecommendationRequest(BaseModel):
    """
    Generate recommendations for a company.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    recommendation_types: list[RecommendationType] = Field(
        default_factory=list,
        description="Recommendation categories to generate.",
    )

    priority_threshold: RecommendationPriority | None = None

    max_recommendations: int = Field(
        default=10,
        ge=1,
        le=50,
    )

    start_date: date | None = None

    end_date: date | None = None

    include_forecasts: bool = True

    include_risk_analysis: bool = True

    include_rag_evidence: bool = True

    include_what_if: bool = False

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=10,
    )

    @field_validator("end_date")
    @classmethod
    def validate_date_range(
        cls,
        value: date | None,
    ) -> date | None:
        return value


class RecommendationFeedbackRequest(BaseModel):
    """
    Human feedback on a generated recommendation.
    """

    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    status: Literal[
        "approved",
        "rejected",
        "dismissed",
        "in_progress",
        "completed",
    ]

    comment: str | None = Field(
        default=None,
        max_length=5000,
    )

    reviewer_id: str | None = Field(
        default=None,
        max_length=128,
    )


class RecommendationActionRequest(BaseModel):
    """
    Update execution state for a recommendation action.
    """

    model_config = ConfigDict(extra="forbid")

    recommendation_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    action_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    status: Literal[
        "approved",
        "in_progress",
        "completed",
        "rejected",
    ]

    comment: str | None = Field(
        default=None,
        max_length=5000,
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class RecommendationResponse(BaseModel):
    """
    Recommendation returned to the frontend.
    """

    model_config = ConfigDict(from_attributes=True)

    recommendation_id: str

    company_id: str

    type: RecommendationType

    priority: RecommendationPriority

    status: RecommendationStatus

    title: str

    summary: str

    rationale: str

    confidence: RecommendationConfidence

    confidence_score: float = Field(
        ...,
        ge=0,
        le=1,
    )

    source: RecommendationSource

    impact: RecommendationImpact | None = None

    actions: list[RecommendationAction] = Field(
        default_factory=list,
    )

    evidence: list[EvidenceItem] = Field(
        default_factory=list,
    )

    risk_factors: list[str] = Field(
        default_factory=list,
    )

    assumptions: list[str] = Field(
        default_factory=list,
    )

    expected_time_to_impact_days: int | None = Field(
        default=None,
        ge=0,
    )

    created_at: datetime

    updated_at: datetime


class RecommendationListResponse(BaseModel):
    """
    Paginated recommendation response.
    """

    items: list[RecommendationResponse]

    total: int

    page: int

    page_size: int

    has_next: bool


class RecommendationGenerationResponse(BaseModel):
    """
    Response from recommendation generation.
    """

    company_id: str

    generated_count: int

    recommendations: list[RecommendationResponse]

    generated_at: datetime

    processing_time_ms: float


class RecommendationSummaryResponse(BaseModel):
    """
    High-level recommendation summary.
    """

    company_id: str

    total: int

    critical: int

    high: int

    medium: int

    low: int

    pending_review: int

    approved: int

    in_progress: int

    completed: int

    estimated_profit_impact: float | None = None

    estimated_cost_impact: float | None = None

    estimated_cashflow_impact: float | None = None

    generated_at: datetime


class RecommendationFeedbackResponse(BaseModel):
    """
    Feedback operation response.
    """

    recommendation_id: str

    status: RecommendationStatus

    updated_at: datetime

    message: str


# ============================================================================
# DEPENDENCIES
# ============================================================================


def get_current_user() -> Any:
    """
    Import the application's authentication dependency lazily.

    Lazy imports keep this API module easier to test and prevent
    circular imports during application startup.
    """

    try:
        from backend.app.dependencies import get_current_user as dependency

        return dependency

    except ImportError as exc:
        logger.exception(
            "Authentication dependency could not be loaded."
        )
        raise RuntimeError(
            "Authentication dependency is not configured."
        ) from exc


def get_recommendation_service() -> Any:
    """
    Lazily construct the recommendation service.

    Expected implementation:

        backend.app.recommendations.recommendation_service.RecommendationService
    """

    try:
        from backend.app.recommendations.recommendation_service import (
            RecommendationService,
        )

        return RecommendationService()

    except ImportError as exc:
        logger.exception(
            "Recommendation service could not be loaded."
        )
        raise RuntimeError(
            "Recommendation service is not configured."
        ) from exc


# ============================================================================
# SECURITY
# ============================================================================


def _get_user_company_id(current_user: Any) -> str | None:
    """
    Extract company ID from the authenticated user.
    """

    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("company_id")

    return getattr(current_user, "company_id", None)


def _is_admin(current_user: Any) -> bool:
    """
    Determine whether the authenticated user has administrative access.
    """

    if current_user is None:
        return False

    if isinstance(current_user, dict):
        return bool(current_user.get("is_admin", False))

    return bool(getattr(current_user, "is_admin", False))


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce tenant isolation.

    A normal user can only access their own company.

    Administrators may access multiple companies depending on
    application-level authorization policy.
    """

    user_company_id = _get_user_company_id(current_user)

    if _is_admin(current_user):
        return

    if not user_company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not associated with a company.",
        )

    if user_company_id != company_id:
        logger.warning(
            "Unauthorized recommendation access attempt: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this company.",
        )


# ============================================================================
# SERVICE HELPERS
# ============================================================================


async def _call_service(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:
    """
    Call a recommendation-service method.

    Supports both async and synchronous service implementations.
    """

    method = getattr(service, method_name, None)

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Recommendation service does not implement "
                f"'{method_name}'."
            ),
        )

    try:
        result = method(**kwargs)

        if hasattr(result, "__await__"):
            result = await result

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid recommendation-service request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid recommendation request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Recommendation service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation service operation failed.",
        ) from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================================
# GENERATION
# ============================================================================


@router.post(
    "/generate",
    response_model=RecommendationGenerationResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate AI recommendations",
    description=(
        "Generates actionable financial recommendations using "
        "financial metrics, forecasts, risk analysis, what-if analysis, "
        "business rules, and optional RAG evidence."
    ),
)
async def generate_recommendations(
    request: RecommendationRequest,
    current_user: Any = Depends(get_current_user()),
) -> RecommendationGenerationResponse:
    """
    Generate recommendations.

    Typical flow:

        Financial Data
            ↓
        Financial Analysis
            ↓
        Forecasting
            ↓
        Risk Detection
            ↓
        What-If Analysis
            ↓
        RAG Evidence
            ↓
        Recommendation Engine
            ↓
        Human Review
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    if (
        request.start_date
        and request.end_date
        and request.start_date > request.end_date
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="start_date cannot be after end_date.",
        )

    started = datetime.now(timezone.utc)

    service = get_recommendation_service()

    result = await _call_service(
        service,
        "generate",
        company_id=request.company_id,
        recommendation_types=request.recommendation_types,
        priority_threshold=request.priority_threshold,
        max_recommendations=request.max_recommendations,
        start_date=request.start_date,
        end_date=request.end_date,
        include_forecasts=request.include_forecasts,
        include_risk_analysis=request.include_risk_analysis,
        include_rag_evidence=request.include_rag_evidence,
        include_what_if=request.include_what_if,
        currency=request.currency,
        user=current_user,
    )

    processing_time_ms = round(
        (datetime.now(timezone.utc) - started).total_seconds() * 1000,
        2,
    )

    if isinstance(result, RecommendationGenerationResponse):
        return result

    if isinstance(result, dict):
        return RecommendationGenerationResponse(
            company_id=request.company_id,
            generated_count=len(result.get("recommendations", [])),
            recommendations=result.get("recommendations", []),
            generated_at=result.get(
                "generated_at",
                _utc_now(),
            ),
            processing_time_ms=result.get(
                "processing_time_ms",
                processing_time_ms,
            ),
        )

    recommendations = result or []

    return RecommendationGenerationResponse(
        company_id=request.company_id,
        generated_count=len(recommendations),
        recommendations=recommendations,
        generated_at=_utc_now(),
        processing_time_ms=processing_time_ms,
    )


# ============================================================================
# LIST
# ============================================================================


@router.get(
    "",
    response_model=RecommendationListResponse,
    summary="List recommendations",
)
async def list_recommendations(
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    recommendation_type: RecommendationType | None = None,
    priority: RecommendationPriority | None = None,
    recommendation_status: RecommendationStatus | None = Query(
        default=None,
        alias="status",
    ),
    page: int = Query(
        default=1,
        ge=1,
        le=10000,
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RecommendationListResponse:
    """
    List generated recommendations with filtering and pagination.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_recommendation_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        recommendation_type=recommendation_type,
        priority=priority,
        status=recommendation_status,
        page=page,
        page_size=page_size,
        user=current_user,
    )

    if isinstance(result, RecommendationListResponse):
        return result

    if isinstance(result, dict):
        return RecommendationListResponse(**result)

    items = result or []

    return RecommendationListResponse(
        items=items,
        total=len(items),
        page=page,
        page_size=page_size,
        has_next=len(items) == page_size,
    )


# ============================================================================
# SINGLE RECOMMENDATION
# ============================================================================


@router.get(
    "/{recommendation_id}",
    response_model=RecommendationResponse,
    summary="Get recommendation",
)
async def get_recommendation(
    recommendation_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RecommendationResponse:
    """
    Retrieve one recommendation.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_recommendation_service()

    result = await _call_service(
        service,
        "get",
        recommendation_id=recommendation_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recommendation not found.",
        )

    if isinstance(result, RecommendationResponse):
        return result

    return RecommendationResponse.model_validate(result)


# ============================================================================
# SUMMARY
# ============================================================================


@router.get(
    "/summary/{company_id}",
    response_model=RecommendationSummaryResponse,
    summary="Get recommendation summary",
)
async def recommendation_summary(
    company_id: str,
    current_user: Any = Depends(get_current_user()),
) -> RecommendationSummaryResponse:
    """
    Return aggregated recommendation statistics.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_recommendation_service()

    result = await _call_service(
        service,
        "summary",
        company_id=company_id,
        user=current_user,
    )

    if isinstance(result, RecommendationSummaryResponse):
        return result

    if isinstance(result, dict):
        return RecommendationSummaryResponse(
            company_id=company_id,
            generated_at=result.get(
                "generated_at",
                _utc_now(),
            ),
            **{
                key: value
                for key, value in result.items()
                if key not in {
                    "company_id",
                    "generated_at",
                }
            },
        )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Invalid recommendation summary returned by service.",
    )


# ============================================================================
# FEEDBACK / HUMAN REVIEW
# ============================================================================


@router.post(
    "/feedback",
    response_model=RecommendationFeedbackResponse,
    summary="Submit recommendation feedback",
)
async def submit_feedback(
    request: RecommendationFeedbackRequest,
    current_user: Any = Depends(get_current_user()),
) -> RecommendationFeedbackResponse:
    """
    Submit human review feedback.

    This endpoint is intentionally separated from generation because
    recommendation approval is a human-in-the-loop operation.
    """

    service = get_recommendation_service()

    result = await _call_service(
        service,
        "update_status",
        recommendation_id=request.recommendation_id,
        status=request.status,
        comment=request.comment,
        reviewer_id=request.reviewer_id,
        user=current_user,
    )

    if isinstance(result, RecommendationFeedbackResponse):
        return result

    new_status = RecommendationStatus(request.status)

    if isinstance(result, dict):
        return RecommendationFeedbackResponse(
            recommendation_id=request.recommendation_id,
            status=RecommendationStatus(
                result.get("status", new_status.value)
            ),
            updated_at=result.get(
                "updated_at",
                _utc_now(),
            ),
            message=result.get(
                "message",
                "Recommendation status updated.",
            ),
        )

    return RecommendationFeedbackResponse(
        recommendation_id=request.recommendation_id,
        status=new_status,
        updated_at=_utc_now(),
        message="Recommendation status updated.",
    )


# ============================================================================
# ACTION EXECUTION
# ============================================================================


@router.post(
    "/actions",
    response_model=RecommendationFeedbackResponse,
    summary="Update recommendation action",
)
async def update_action(
    request: RecommendationActionRequest,
    current_user: Any = Depends(get_current_user()),
) -> RecommendationFeedbackResponse:
    """
    Update the execution state of an individual recommendation action.
    """

    service = get_recommendation_service()

    result = await _call_service(
        service,
        "update_action",
        recommendation_id=request.recommendation_id,
        action_id=request.action_id,
        status=request.status,
        comment=request.comment,
        user=current_user,
    )

    if isinstance(result, RecommendationFeedbackResponse):
        return result

    return RecommendationFeedbackResponse(
        recommendation_id=request.recommendation_id,
        status=RecommendationStatus(
            request.status
        ),
        updated_at=_utc_now(),
        message="Recommendation action updated.",
    )


# ============================================================================
# SPECIALIZED RECOMMENDATION ENDPOINTS
# ============================================================================


@router.post(
    "/cost-optimization/{company_id}",
    response_model=RecommendationGenerationResponse,
    summary="Generate cost optimization recommendations",
)
async def cost_optimization(
    company_id: str,
    max_recommendations: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RecommendationGenerationResponse:
    """
    Generate recommendations focused on reducing unnecessary costs.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    request = RecommendationRequest(
        company_id=company_id,
        recommendation_types=[
            RecommendationType.COST_REDUCTION,
            RecommendationType.EXPENSE_OPTIMIZATION,
            RecommendationType.BUDGET_OPTIMIZATION,
        ],
        max_recommendations=max_recommendations,
        include_forecasts=True,
        include_risk_analysis=True,
        include_rag_evidence=True,
    )

    return await generate_recommendations(
        request=request,
        current_user=current_user,
    )


@router.post(
    "/cash-flow/{company_id}",
    response_model=RecommendationGenerationResponse,
    summary="Generate cash-flow recommendations",
)
async def cash_flow_recommendations(
    company_id: str,
    max_recommendations: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RecommendationGenerationResponse:
    """
    Generate recommendations focused on cash flow and working capital.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    request = RecommendationRequest(
        company_id=company_id,
        recommendation_types=[
            RecommendationType.CASH_FLOW,
            RecommendationType.LIQUIDITY,
            RecommendationType.WORKING_CAPITAL,
            RecommendationType.COLLECTIONS,
        ],
        max_recommendations=max_recommendations,
        include_forecasts=True,
        include_risk_analysis=True,
        include_rag_evidence=True,
        include_what_if=True,
    )

    return await generate_recommendations(
        request=request,
        current_user=current_user,
    )


@router.post(
    "/risk/{company_id}",
    response_model=RecommendationGenerationResponse,
    summary="Generate risk mitigation recommendations",
)
async def risk_recommendations(
    company_id: str,
    max_recommendations: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RecommendationGenerationResponse:
    """
    Generate recommendations focused on financial/fraud risk.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    request = RecommendationRequest(
        company_id=company_id,
        recommendation_types=[
            RecommendationType.RISK_MITIGATION,
            RecommendationType.FRAUD_PREVENTION,
        ],
        max_recommendations=max_recommendations,
        include_forecasts=True,
        include_risk_analysis=True,
        include_rag_evidence=True,
        include_what_if=False,
    )

    return await generate_recommendations(
        request=request,
        current_user=current_user,
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    summary="Recommendation service health",
)
async def recommendation_health() -> dict[str, Any]:
    """
    Lightweight health check for the recommendation subsystem.

    Does not expose:
        - credentials
        - database URLs
        - API keys
        - model configuration secrets
    """

    try:
        service = get_recommendation_service()

        health_method = getattr(
            service,
            "health",
            None,
        )

        if health_method is None:
            return {
                "status": "healthy",
                "service": "recommendation_service",
                "timestamp": _utc_now(),
            }

        result = health_method()

        if hasattr(result, "__await__"):
            result = await result

        return {
            "status": "healthy",
            "service": "recommendation_service",
            "details": result,
            "timestamp": _utc_now(),
        }

    except Exception:
        logger.exception(
            "Recommendation service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is unavailable.",
        )