"""
backend/app/api/risk.py

Risk Intelligence API for FinCo AI.

Responsibilities
----------------
- Authentication and authorization
- Tenant/company isolation
- Request validation
- Risk-service orchestration
- Risk score retrieval
- Risk factor analysis
- Risk history
- Risk summary
- Risk assessment
- Health checks

Architecture
------------

    Client
       |
       v
    FastAPI Risk API
       |
       v
    RiskService
       |
       +---- Financial Metrics
       +---- Revenue
       +---- Expenses
       +---- Profitability
       +---- Liquidity
       +---- Cash Flow
       +---- Fraud Signals
       +---- Forecasts
       +---- Alerts
       |
       v
    Risk Scoring Engine
       |
       +---- Risk Score
       +---- Risk Level
       +---- Risk Factors
       +---- Explanation
       |
       v
    Supervisor Agent / Recommendations / Alerts
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/risk",
    tags=["Risk Intelligence"],
)


# ============================================================================
# TYPES
# ============================================================================


RiskLevel = Literal[
    "very_low",
    "low",
    "moderate",
    "high",
    "critical",
]

RiskCategory = Literal[
    "financial",
    "liquidity",
    "profitability",
    "revenue",
    "cash_flow",
    "fraud",
    "operational",
    "credit",
    "market",
    "compliance",
    "overall",
]

RiskTrend = Literal[
    "improving",
    "stable",
    "deteriorating",
    "volatile",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class RiskAssessmentRequest(BaseModel):
    """
    Request for a complete risk assessment.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    start_date: date

    end_date: date

    categories: list[RiskCategory] = Field(
        default_factory=lambda: ["overall"]
    )

    include_factors: bool = True

    include_recommendations: bool = True

    include_forecast_risk: bool = True

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=10,
    )


class RiskScoreRequest(BaseModel):
    """
    Request for a risk score.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    start_date: date

    end_date: date

    category: RiskCategory = "overall"

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=10,
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class RiskFactor(BaseModel):
    """
    Individual factor contributing to risk.
    """

    model_config = ConfigDict(extra="forbid")

    factor_id: str

    name: str

    category: RiskCategory

    score: float = Field(
        ge=0.0,
        le=100.0,
    )

    weight: float = Field(
        ge=0.0,
    )

    contribution: float

    direction: Literal[
        "positive",
        "negative",
        "neutral",
    ]

    severity: RiskLevel

    value: float | None = None

    benchmark: float | None = None

    explanation: str


class RiskScoreResponse(BaseModel):
    """
    Risk score returned by the risk engine.
    """

    company_id: str

    category: RiskCategory

    score: float = Field(
        ge=0.0,
        le=100.0,
    )

    level: RiskLevel

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    trend: RiskTrend

    period_start: date

    period_end: date

    currency: str | None = None

    factors: list[RiskFactor] = Field(
        default_factory=list
    )

    explanation: str | None = None

    generated_at: datetime


class RiskAssessmentResponse(BaseModel):
    """
    Complete company risk assessment.
    """

    company_id: str

    overall_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    overall_level: RiskLevel

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    trend: RiskTrend

    period_start: date

    period_end: date

    currency: str | None = None

    category_scores: dict[str, float] = Field(
        default_factory=dict
    )

    factors: list[RiskFactor] = Field(
        default_factory=list
    )

    critical_risks: list[str] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    recommendations: list[str] = Field(
        default_factory=list
    )

    forecast_risk: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    explanation: str | None = None

    generated_at: datetime


class RiskSummaryResponse(BaseModel):
    """
    Compact risk summary for dashboards.
    """

    company_id: str

    overall_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    overall_level: RiskLevel

    trend: RiskTrend

    highest_risk_category: RiskCategory | None = None

    highest_risk_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    category_scores: dict[str, float] = Field(
        default_factory=dict
    )

    critical_risk_count: int = Field(
        default=0,
        ge=0,
    )

    warning_count: int = Field(
        default=0,
        ge=0,
    )

    generated_at: datetime


class RiskHistoryPoint(BaseModel):
    """
    Historical risk score.
    """

    period_start: date

    period_end: date

    score: float = Field(
        ge=0.0,
        le=100.0,
    )

    level: RiskLevel

    category: RiskCategory


class RiskHistoryResponse(BaseModel):
    """
    Historical risk trajectory.
    """

    company_id: str

    category: RiskCategory

    periods: list[RiskHistoryPoint]

    current_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    previous_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    trend: RiskTrend

    generated_at: datetime


class RiskFactorResponse(BaseModel):
    """
    Risk factor analysis response.
    """

    company_id: str

    category: RiskCategory

    period_start: date

    period_end: date

    factors: list[RiskFactor]

    dominant_factor: RiskFactor | None = None

    generated_at: datetime


class RiskHealthResponse(BaseModel):
    """
    Risk subsystem health.
    """

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    service: str

    timestamp: datetime

    details: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================================
# AUTHENTICATION
# ============================================================================


def get_current_user() -> Any:
    """
    Lazily load authentication dependency.

    This avoids importing authentication infrastructure during
    module discovery and makes the API easier to test.
    """

    try:
        from backend.app.dependencies import (
            get_current_user as dependency,
        )

        return dependency

    except ImportError as exc:
        logger.exception(
            "Authentication dependency unavailable."
        )

        raise RuntimeError(
            "Authentication dependency is not configured."
        ) from exc


# ============================================================================
# SERVICE
# ============================================================================


def get_risk_service() -> Any:
    """
    Lazily construct the RiskService.

    Expected implementation:

        backend.app.risk.risk_service.RiskService

    A fallback to the existing alert risk-scoring engine is
    supported so the API can evolve incrementally.
    """

    try:
        from backend.app.risk.risk_service import (
            RiskService,
        )

        return RiskService()

    except ImportError:
        pass

    try:
        from backend.app.alerts.risk_scoring import (
            RiskScoringService,
        )

        return RiskScoringService()

    except ImportError as exc:
        logger.exception(
            "Risk service could not be loaded."
        )

        raise RuntimeError(
            "Risk service is not configured."
        ) from exc


# ============================================================================
# SECURITY
# ============================================================================


def _get_user_company_id(
    current_user: Any,
) -> str | None:
    """
    Extract company ID from authenticated user.
    """

    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("company_id")

    return getattr(
        current_user,
        "company_id",
        None,
    )


def _is_admin(
    current_user: Any,
) -> bool:
    """
    Determine whether the current user has administrative access.
    """

    if current_user is None:
        return False

    if isinstance(current_user, dict):
        return bool(
            current_user.get(
                "is_admin",
                False,
            )
        )

    return bool(
        getattr(
            current_user,
            "is_admin",
            False,
        )
    )


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce tenant isolation.

    Risk information is financially sensitive and must never
    cross company boundaries.
    """

    if _is_admin(current_user):
        return

    user_company_id = _get_user_company_id(
        current_user
    )

    if not user_company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "User is not associated with a company."
            ),
        )

    if user_company_id != company_id:
        logger.warning(
            "Unauthorized risk access attempt: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this "
                "company's risk information."
            ),
        )


# ============================================================================
# VALIDATION
# ============================================================================


def validate_period(
    start_date: date,
    end_date: date,
    max_days: int = 3660,
) -> None:
    """
    Validate analysis period.
    """

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be after end_date."
            ),
        )

    if (end_date - start_date).days > max_days:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Requested period cannot exceed "
                f"{max_days} days."
            ),
        )


def validate_categories(
    categories: list[RiskCategory],
) -> None:
    """
    Validate requested risk categories.
    """

    if not categories:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one risk category is required.",
        )

    if len(categories) > 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "A maximum of 10 risk categories can be "
                "requested at once."
            ),
        )


# ============================================================================
# SERVICE CALLER
# ============================================================================


async def _call_service(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:
    """
    Invoke a risk service method.

    Supports both synchronous and asynchronous service
    implementations.
    """

    method = getattr(
        service,
        method_name,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Risk service does not implement "
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
            "Invalid risk-service request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid risk assessment request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Risk service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk assessment failed.",
        ) from exc


# ============================================================================
# COMPLETE RISK ASSESSMENT
# ============================================================================


@router.post(
    "/assess",
    response_model=RiskAssessmentResponse,
    summary="Perform company risk assessment",
    description=(
        "Calculates overall and category-level financial risk "
        "for a company."
    ),
)
async def assess_risk(
    request: RiskAssessmentRequest,
    current_user: Any = Depends(get_current_user()),
) -> RiskAssessmentResponse:
    """
    Perform a complete company risk assessment.

    Typical downstream flow:

        Financial Data
            ↓
        Financial Metrics
            ↓
        Risk Assessment
            ↓
        Alert Engine
            ↓
        Supervisor Agent
            ↓
        Recommendations
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_period(
        request.start_date,
        request.end_date,
    )

    validate_categories(
        request.categories,
    )

    service = get_risk_service()

    result = await _call_service(
        service,
        "assess",
        company_id=request.company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        categories=request.categories,
        include_factors=request.include_factors,
        include_recommendations=(
            request.include_recommendations
        ),
        include_forecast_risk=(
            request.include_forecast_risk
        ),
        currency=request.currency,
        user=current_user,
    )

    if isinstance(
        result,
        RiskAssessmentResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Risk assessment could not be generated."
            ),
        )

    return RiskAssessmentResponse.model_validate(
        result
    )


# ============================================================================
# RISK SCORE
# ============================================================================


@router.post(
    "/score",
    response_model=RiskScoreResponse,
    summary="Calculate risk score",
)
async def get_risk_score(
    request: RiskScoreRequest,
    current_user: Any = Depends(get_current_user()),
) -> RiskScoreResponse:
    """
    Calculate a risk score for a specific category.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_period(
        request.start_date,
        request.end_date,
    )

    service = get_risk_service()

    result = await _call_service(
        service,
        "score",
        company_id=request.company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        category=request.category,
        currency=request.currency,
        user=current_user,
    )

    if isinstance(
        result,
        RiskScoreResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Risk score could not be calculated.",
        )

    return RiskScoreResponse.model_validate(
        result
    )


# ============================================================================
# RISK SUMMARY
# ============================================================================


@router.get(
    "/summary/{company_id}",
    response_model=RiskSummaryResponse,
    summary="Get risk summary",
)
async def get_risk_summary(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RiskSummaryResponse:
    """
    Return compact risk information for dashboards.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_risk_service()

    result = await _call_service(
        service,
        "summary",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        currency=currency,
        user=current_user,
    )

    if isinstance(
        result,
        RiskSummaryResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Risk summary not found.",
        )

    return RiskSummaryResponse.model_validate(
        result
    )


# ============================================================================
# RISK FACTORS
# ============================================================================


@router.get(
    "/factors/{company_id}",
    response_model=RiskFactorResponse,
    summary="Get risk factors",
)
async def get_risk_factors(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    category: RiskCategory = Query(
        default="overall"
    ),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RiskFactorResponse:
    """
    Return factors contributing to the company's risk.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_risk_service()

    result = await _call_service(
        service,
        "factors",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        category=category,
        currency=currency,
        user=current_user,
    )

    if isinstance(
        result,
        RiskFactorResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Risk factors not found.",
        )

    return RiskFactorResponse.model_validate(
        result
    )


# ============================================================================
# RISK HISTORY
# ============================================================================


@router.get(
    "/history/{company_id}",
    response_model=RiskHistoryResponse,
    summary="Get historical risk",
)
async def get_risk_history(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    category: RiskCategory = Query(
        default="overall"
    ),
    current_user: Any = Depends(get_current_user()),
) -> RiskHistoryResponse:
    """
    Return historical risk scores.

    Useful for detecting:

        improving
        stable
        deteriorating
        volatile
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
        max_days=3650,
    )

    service = get_risk_service()

    result = await _call_service(
        service,
        "history",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        category=category,
        user=current_user,
    )

    if isinstance(
        result,
        RiskHistoryResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Risk history not found.",
        )

    return RiskHistoryResponse.model_validate(
        result
    )


# ============================================================================
# CATEGORY RISK
# ============================================================================


@router.get(
    "/category/{company_id}",
    response_model=RiskScoreResponse,
    summary="Get category-specific risk",
)
async def get_category_risk(
    company_id: str,
    category: RiskCategory,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RiskScoreResponse:
    """
    Return risk for a specific category.

    Examples:

        /risk/category/{company_id}?category=liquidity
        /risk/category/{company_id}?category=profitability
        /risk/category/{company_id}?category=fraud
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_risk_service()

    result = await _call_service(
        service,
        "score",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        category=category,
        currency=currency,
        user=current_user,
    )

    if isinstance(
        result,
        RiskScoreResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Category risk score could not be calculated."
            ),
        )

    return RiskScoreResponse.model_validate(
        result
    )


# ============================================================================
# CRITICAL RISKS
# ============================================================================


@router.get(
    "/critical/{company_id}",
    summary="Get critical risks",
)
async def get_critical_risks(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> dict[str, Any]:
    """
    Return risks that exceed critical thresholds.

    This endpoint is intended to feed the alert subsystem.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_risk_service()

    result = await _call_service(
        service,
        "critical_risks",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        user=current_user,
    )

    return {
        "company_id": company_id,
        "period_start": start_date,
        "period_end": end_date,
        "critical_risks": result or [],
        "generated_at": datetime.now(
            timezone.utc
        ),
    }


# ============================================================================
# RISK ALERT SIGNAL
# ============================================================================


@router.get(
    "/alert-signal/{company_id}",
    summary="Generate risk alert signal",
)
async def get_risk_alert_signal(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> dict[str, Any]:
    """
    Generate a machine-readable risk signal for the alert engine.

    This does not itself create an alert.

    Alert creation remains the responsibility of:

        backend.app.alerts.detector
        backend.app.alerts.alert_generator
        backend.app.alerts.alert_service
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_risk_service()

    result = await _call_service(
        service,
        "alert_signal",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        user=current_user,
    )

    return {
        "company_id": company_id,
        "period_start": start_date,
        "period_end": end_date,
        "signal": result,
        "generated_at": datetime.now(
            timezone.utc
        ),
    }


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=RiskHealthResponse,
    summary="Risk service health",
)
async def risk_health() -> RiskHealthResponse:
    """
    Health check for the risk subsystem.
    """

    started = time.perf_counter()

    try:
        service = get_risk_service()

        health_method = getattr(
            service,
            "health",
            None,
        )

        details: dict[str, Any] = {}

        if health_method is not None:
            result = health_method()

            if hasattr(result, "__await__"):
                result = await result

            if isinstance(result, dict):
                details = result

        details["latency_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return RiskHealthResponse(
            status="healthy",
            service="risk",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Risk service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk service is unavailable.",
        )