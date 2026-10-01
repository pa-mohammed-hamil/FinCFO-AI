"""
backend/app/api/revenue.py

Revenue API for FinCo AI.

Responsibilities:
    - Request validation
    - Authentication / authorization
    - Tenant isolation
    - Revenue service orchestration
    - Revenue summary / trend / comparison endpoints
    - Safe error handling

Architecture:

    Client
       |
       v
    FastAPI Revenue API
       |
       v
    RevenueService
       |
       +---- Transactions
       +---- Revenue extraction
       +---- Normalization
       +---- Revenue aggregation
       +---- Growth analysis
       +---- Trend analysis
       +---- Forecast inputs
       |
       v
    Revenue Intelligence
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
    prefix="/revenue",
    tags=["Revenue"],
)


# ============================================================================
# ENUMS
# ============================================================================


class RevenueGranularity(str):
    """Supported revenue aggregation granularities."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class RevenueCategory(str):
    """Common revenue categories."""

    PRODUCT = "product"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    LICENSING = "licensing"
    OTHER = "other"


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class RevenueLineItem(BaseModel):
    """Revenue value for an individual category/source."""

    model_config = ConfigDict(extra="forbid")

    category: str

    amount: float

    percentage_of_total: float | None = None

    previous_amount: float | None = None

    variance: float | None = None

    variance_percentage: float | None = None


class RevenueStatement(BaseModel):
    """Revenue statement for a reporting period."""

    model_config = ConfigDict(extra="forbid")

    company_id: str

    period_start: date

    period_end: date

    currency: str

    total_revenue: float

    recurring_revenue: float = 0.0

    non_recurring_revenue: float = 0.0

    product_revenue: float = 0.0

    service_revenue: float = 0.0

    subscription_revenue: float = 0.0

    other_revenue: float = 0.0

    revenue_growth: float | None = None

    previous_period_revenue: float | None = None

    revenue_variance: float | None = None

    revenue_variance_percentage: float | None = None

    line_items: list[RevenueLineItem] = Field(
        default_factory=list
    )

    generated_at: datetime


class RevenuePeriod(BaseModel):
    """Revenue value for one historical period."""

    model_config = ConfigDict(extra="forbid")

    period_start: date

    period_end: date

    revenue: float

    recurring_revenue: float = 0.0

    non_recurring_revenue: float = 0.0

    growth_percentage: float | None = None


class RevenueTrendResponse(BaseModel):
    """Historical revenue trend."""

    company_id: str

    currency: str

    granularity: str

    periods: list[RevenuePeriod]

    total_revenue: float

    average_revenue: float

    growth_percentage: float | None = None

    trend_direction: Literal[
        "increasing",
        "decreasing",
        "stable",
        "volatile",
    ]

    generated_at: datetime


class RevenueComparisonResponse(BaseModel):
    """Revenue comparison between two periods."""

    company_id: str

    currency: str

    current_period_start: date

    current_period_end: date

    comparison_period_start: date

    comparison_period_end: date

    current_revenue: float

    comparison_revenue: float

    variance: float

    variance_percentage: float | None = None

    growth_percentage: float | None = None

    direction: Literal[
        "increase",
        "decrease",
        "unchanged",
    ]

    generated_at: datetime


class RevenueSourceResponse(BaseModel):
    """Revenue breakdown by source/category."""

    company_id: str

    currency: str

    period_start: date

    period_end: date

    sources: list[RevenueLineItem]

    total_revenue: float

    generated_at: datetime


class RevenueAnalysisResponse(BaseModel):
    """Analytical interpretation of revenue performance."""

    company_id: str

    period_start: date

    period_end: date

    currency: str

    total_revenue: float

    growth_percentage: float | None = None

    trend: Literal[
        "strong_growth",
        "moderate_growth",
        "stable",
        "moderate_decline",
        "severe_decline",
        "volatile",
    ]

    key_drivers: list[str] = Field(
        default_factory=list
    )

    concerns: list[str] = Field(
        default_factory=list
    )

    opportunities: list[str] = Field(
        default_factory=list
    )

    generated_at: datetime


class RevenueHealthResponse(BaseModel):
    """Revenue subsystem health."""

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
# REQUEST SCHEMAS
# ============================================================================


class RevenueRequest(BaseModel):
    """Request for revenue statement."""

    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    start_date: date

    end_date: date

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=10,
    )

    include_breakdown: bool = True


class RevenueTrendRequest(BaseModel):
    """Request for historical revenue trend."""

    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    start_date: date

    end_date: date

    granularity: Literal[
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "yearly",
    ] = "monthly"

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=10,
    )


# ============================================================================
# DEPENDENCIES
# ============================================================================


def get_current_user() -> Any:
    """
    Lazily load the application's authentication dependency.
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


def get_revenue_service() -> Any:
    """
    Lazily construct RevenueService.

    Expected implementation:

        backend.app.financial.revenue.RevenueService
    """

    try:
        from backend.app.financial.revenue import (
            RevenueService,
        )

        return RevenueService()

    except ImportError as exc:
        logger.exception(
            "Revenue service could not be loaded."
        )

        raise RuntimeError(
            "Revenue service is not configured."
        ) from exc


# ============================================================================
# SECURITY
# ============================================================================


def _get_user_company_id(
    current_user: Any,
) -> str | None:
    """Extract company ID from authenticated user."""

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
    """Determine administrative access."""

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
    Enforce company-level tenant isolation.

    Financial information must never be returned for another
    company merely because its ID was supplied to the API.
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
            "Unauthorized revenue access attempt: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this company's "
                "revenue data."
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
    """Validate reporting period."""

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


# ============================================================================
# SERVICE HELPER
# ============================================================================


async def _call_service(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:
    """
    Safely invoke a RevenueService method.

    Supports both async and synchronous implementations.
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
                f"Revenue service does not implement "
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
            "Invalid revenue-service request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid revenue request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Revenue service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Revenue calculation failed.",
        ) from exc


# ============================================================================
# REVENUE STATEMENT
# ============================================================================


@router.post(
    "/statement",
    response_model=RevenueStatement,
    summary="Get revenue statement",
    description=(
        "Returns revenue metrics for a company and reporting period."
    ),
)
async def get_revenue_statement(
    request: RevenueRequest,
    current_user: Any = Depends(get_current_user()),
) -> RevenueStatement:
    """
    Generate/retrieve a revenue statement.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_period(
        request.start_date,
        request.end_date,
    )

    service = get_revenue_service()

    result = await _call_service(
        service,
        "get_statement",
        company_id=request.company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        currency=request.currency,
        include_breakdown=request.include_breakdown,
        user=current_user,
    )

    if isinstance(
        result,
        RevenueStatement,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No revenue data found for the requested period."
            ),
        )

    return RevenueStatement.model_validate(result)


# ============================================================================
# REVENUE SUMMARY
# ============================================================================


@router.get(
    "/summary/{company_id}",
    summary="Get revenue summary",
)
async def get_revenue_summary(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> dict[str, Any]:
    """
    Return compact revenue metrics for dashboards.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_revenue_service()

    result = await _call_service(
        service,
        "get_summary",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        currency=currency,
        user=current_user,
    )

    return {
        "company_id": company_id,
        "period_start": start_date,
        "period_end": end_date,
        "data": result,
        "generated_at": datetime.now(
            timezone.utc
        ),
    }


# ============================================================================
# REVENUE TREND
# ============================================================================


@router.post(
    "/trend",
    response_model=RevenueTrendResponse,
    summary="Get revenue trend",
)
async def get_revenue_trend(
    request: RevenueTrendRequest,
    current_user: Any = Depends(get_current_user()),
) -> RevenueTrendResponse:
    """
    Retrieve historical revenue trend.

    Used by:
        - financial dashboards
        - anomaly detection
        - forecasting
        - health scoring
        - recommendation engine
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_period(
        request.start_date,
        request.end_date,
        max_days=3650,
    )

    service = get_revenue_service()

    result = await _call_service(
        service,
        "get_trend",
        company_id=request.company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        granularity=request.granularity,
        currency=request.currency,
        user=current_user,
    )

    if isinstance(
        result,
        RevenueTrendResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No revenue trend data found.",
        )

    return RevenueTrendResponse.model_validate(result)


# ============================================================================
# REVENUE COMPARISON
# ============================================================================


@router.get(
    "/comparison/{company_id}",
    response_model=RevenueComparisonResponse,
    summary="Compare revenue periods",
)
async def compare_revenue(
    company_id: str,
    current_start_date: date = Query(...),
    current_end_date: date = Query(...),
    comparison_start_date: date = Query(...),
    comparison_end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RevenueComparisonResponse:
    """
    Compare revenue across two periods.

    Supports:
        - Month-over-month
        - Quarter-over-quarter
        - Year-over-year
        - Previous-period decline detection
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        current_start_date,
        current_end_date,
    )

    validate_period(
        comparison_start_date,
        comparison_end_date,
    )

    service = get_revenue_service()

    result = await _call_service(
        service,
        "compare",
        company_id=company_id,
        current_start_date=current_start_date,
        current_end_date=current_end_date,
        comparison_start_date=comparison_start_date,
        comparison_end_date=comparison_end_date,
        currency=currency,
        user=current_user,
    )

    if isinstance(
        result,
        RevenueComparisonResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Revenue comparison could not be generated."
            ),
        )

    return RevenueComparisonResponse.model_validate(result)


# ============================================================================
# REVENUE SOURCES
# ============================================================================


@router.get(
    "/sources/{company_id}",
    response_model=RevenueSourceResponse,
    summary="Get revenue breakdown",
)
async def get_revenue_sources(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RevenueSourceResponse:
    """
    Return revenue broken down by category/source.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_revenue_service()

    result = await _call_service(
        service,
        "get_sources",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        currency=currency,
        user=current_user,
    )

    if isinstance(
        result,
        RevenueSourceResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Revenue source data not found.",
        )

    return RevenueSourceResponse.model_validate(result)


# ============================================================================
# REVENUE ANALYSIS
# ============================================================================


@router.get(
    "/analysis/{company_id}",
    response_model=RevenueAnalysisResponse,
    summary="Analyze revenue performance",
)
async def analyze_revenue(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> RevenueAnalysisResponse:
    """
    Analyze revenue growth, decline, volatility and drivers.

    This endpoint provides analytical output for downstream
    health-score, alert and recommendation components.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_revenue_service()

    result = await _call_service(
        service,
        "analyze",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        currency=currency,
        user=current_user,
    )

    if isinstance(
        result,
        RevenueAnalysisResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Revenue analysis could not be generated.",
        )

    return RevenueAnalysisResponse.model_validate(result)


# ============================================================================
# GROWTH
# ============================================================================


@router.get(
    "/growth/{company_id}",
    summary="Get revenue growth",
)
async def get_revenue_growth(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    comparison_start_date: date | None = Query(
        default=None
    ),
    comparison_end_date: date | None = Query(
        default=None
    ),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> dict[str, Any]:
    """
    Calculate revenue growth against a comparison period.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    if (
        comparison_start_date is not None
        and comparison_end_date is not None
    ):
        validate_period(
            comparison_start_date,
            comparison_end_date,
        )

    service = get_revenue_service()

    result = await _call_service(
        service,
        "growth",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        comparison_start_date=comparison_start_date,
        comparison_end_date=comparison_end_date,
        currency=currency,
        user=current_user,
    )

    return {
        "company_id": company_id,
        "period_start": start_date,
        "period_end": end_date,
        "data": result,
        "generated_at": datetime.now(
            timezone.utc
        ),
    }


# ============================================================================
# RECURRING REVENUE
# ============================================================================


@router.get(
    "/recurring/{company_id}",
    summary="Get recurring revenue",
)
async def get_recurring_revenue(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> dict[str, Any]:
    """
    Return recurring revenue metrics.

    Useful for SaaS/subscription-oriented businesses.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_revenue_service()

    result = await _call_service(
        service,
        "recurring_revenue",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        currency=currency,
        user=current_user,
    )

    return {
        "company_id": company_id,
        "period_start": start_date,
        "period_end": end_date,
        "data": result,
        "generated_at": datetime.now(
            timezone.utc
        ),
    }


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=RevenueHealthResponse,
    summary="Revenue service health",
)
async def revenue_health() -> RevenueHealthResponse:
    """
    Health check for the revenue subsystem.
    """

    started = time.perf_counter()

    try:
        service = get_revenue_service()

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

        return RevenueHealthResponse(
            status="healthy",
            service="revenue",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Revenue service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Revenue service is unavailable.",
        )