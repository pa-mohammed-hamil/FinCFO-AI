"""
backend/app/api/pnl.py

Profit & Loss (P&L) API for FinCo AI.

Responsibilities:
    - Request validation
    - Authentication / authorization
    - Company / tenant isolation
    - Period validation
    - P&L service orchestration
    - Response serialization
    - Safe error handling

Architecture:

    Client
       |
       v
    FastAPI P&L API
       |
       v
    P&L Service / FinancialRepository
       |
       +---- Revenue
       +---- COGS
       +---- Gross Profit
       +---- Operating Expenses
       +---- Operating Profit
       +---- EBITDA
       +---- Net Profit
       +---- Margins
       +---- Variance
       +---- Trend
       |
       v
    P&L Response
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/pnl",
    tags=["Profit & Loss"],
)


# ============================================================================
# ENUMS
# ============================================================================


class PnLGranularity(str):
    """
    Supported P&L reporting granularity.

    Kept as a string-compatible type so it remains easy to integrate
    with existing service/repository implementations.
    """

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class PnLPeriodType(str):
    """Supported comparison periods."""

    CURRENT = "current"
    PREVIOUS = "previous"
    YTD = "ytd"
    MTD = "mtd"
    QTD = "qtd"
    YOY = "yoy"


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class PnLLineItem(BaseModel):
    """Individual P&L line item."""

    model_config = ConfigDict(extra="forbid")

    name: str
    amount: float
    percentage_of_revenue: float | None = None
    previous_amount: float | None = None
    variance: float | None = None
    variance_percentage: float | None = None


class PnLStatement(BaseModel):
    """
    Complete profit and loss statement.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str

    period_start: date
    period_end: date

    currency: str

    revenue: float

    cost_of_goods_sold: float

    gross_profit: float

    operating_expenses: float

    operating_profit: float

    depreciation: float = 0.0

    amortization: float = 0.0

    ebitda: float | None = None

    interest_expense: float = 0.0

    interest_income: float = 0.0

    taxes: float = 0.0

    net_profit: float

    gross_margin: float

    operating_margin: float

    net_margin: float

    ebitda_margin: float | None = None

    line_items: list[PnLLineItem] = Field(
        default_factory=list
    )

    generated_at: datetime


class PnLPeriod(BaseModel):
    """P&L value for one reporting period."""

    model_config = ConfigDict(extra="forbid")

    period_start: date
    period_end: date

    revenue: float

    cost_of_goods_sold: float

    gross_profit: float

    operating_expenses: float

    operating_profit: float

    ebitda: float | None = None

    net_profit: float

    gross_margin: float

    operating_margin: float

    net_margin: float


class PnLTrendResponse(BaseModel):
    """Historical P&L trend."""

    company_id: str

    granularity: str

    currency: str

    periods: list[PnLPeriod]

    revenue_growth: float | None = None

    profit_growth: float | None = None

    margin_change: float | None = None

    generated_at: datetime


class PnLComparisonResponse(BaseModel):
    """Comparison between two P&L periods."""

    company_id: str

    currency: str

    current_period: PnLPeriod

    comparison_period: PnLPeriod

    revenue_variance: float

    revenue_variance_percentage: float | None = None

    gross_profit_variance: float

    gross_profit_variance_percentage: float | None = None

    operating_profit_variance: float

    operating_profit_variance_percentage: float | None = None

    net_profit_variance: float

    net_profit_variance_percentage: float | None = None

    gross_margin_change: float

    operating_margin_change: float

    net_margin_change: float

    generated_at: datetime


class PnLAnalysisResponse(BaseModel):
    """
    Analytical interpretation of the P&L.
    """

    company_id: str

    period_start: date

    period_end: date

    currency: str

    profitability_status: Literal[
        "profitable",
        "break_even",
        "loss_making",
    ]

    strongest_metric: str | None = None

    weakest_metric: str | None = None

    key_drivers: list[str] = Field(
        default_factory=list
    )

    concerns: list[str] = Field(
        default_factory=list
    )

    opportunities: list[str] = Field(
        default_factory=list
    )

    recommendations: list[str] = Field(
        default_factory=list
    )

    generated_at: datetime


class PnLSummaryResponse(BaseModel):
    """Compact P&L summary for dashboards."""

    company_id: str

    period_start: date

    period_end: date

    currency: str

    revenue: float

    gross_profit: float

    operating_profit: float

    net_profit: float

    gross_margin: float

    operating_margin: float

    net_margin: float

    revenue_growth: float | None = None

    profit_growth: float | None = None

    status: Literal[
        "profitable",
        "break_even",
        "loss_making",
    ]

    generated_at: datetime


class PnLHealthResponse(BaseModel):
    """Health information for the P&L subsystem."""

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


class PnLRequest(BaseModel):
    """Request for a P&L statement."""

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

    include_line_items: bool = True

    @field_validator("end_date")
    @classmethod
    def validate_end_date(
        cls,
        value: date,
    ) -> date:
        return value


class PnLTrendRequest(BaseModel):
    """Request for historical P&L trend."""

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


def get_pnl_service() -> Any:
    """
    Lazily construct the P&L service.

    Expected implementation:

        backend.app.financial.pnl.PnLService

    If your project uses FinancialRepository as the primary
    orchestration layer, this function can be changed to return
    that repository instead.
    """

    try:
        from backend.app.financial.pnl import PnLService

        return PnLService()

    except ImportError:
        try:
            from backend.app.database.repositories.financial_repository import (
                FinancialRepository,
            )

            return FinancialRepository()

        except ImportError as exc:
            logger.exception(
                "P&L service/repository could not be loaded."
            )

            raise RuntimeError(
                "P&L service is not configured."
            ) from exc


# ============================================================================
# SECURITY
# ============================================================================


def _get_user_company_id(
    current_user: Any,
) -> str | None:
    """Extract authenticated user's company ID."""

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
    """Determine whether the user has admin privileges."""

    if current_user is None:
        return False

    if isinstance(current_user, dict):
        return bool(
            current_user.get("is_admin", False)
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

    Normal users may only access their own company.
    """

    if _is_admin(current_user):
        return

    user_company_id = _get_user_company_id(
        current_user
    )

    if not user_company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is not associated with a company.",
        )

    if user_company_id != company_id:
        logger.warning(
            "Unauthorized P&L access attempt: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this company's "
                "financial data."
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
    Validate requested financial period.

    A maximum period protects the API from accidentally expensive
    historical aggregation queries.
    """

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="start_date cannot be after end_date.",
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
    Safely call a service method.

    Supports async and synchronous service implementations.
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
                f"P&L service does not implement "
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
            "Invalid P&L service request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid P&L request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "P&L service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="P&L calculation failed.",
        ) from exc


# ============================================================================
# P&L STATEMENT
# ============================================================================


@router.post(
    "/statement",
    response_model=PnLStatement,
    summary="Get P&L statement",
    description=(
        "Returns a complete profit and loss statement "
        "for a company and reporting period."
    ),
)
async def get_pnl_statement(
    request: PnLRequest,
    current_user: Any = Depends(get_current_user()),
) -> PnLStatement:
    """
    Generate/retrieve P&L statement.

    Financial flow:

        Transactions
             ↓
        Normalization
             ↓
        Revenue / COGS / Expenses
             ↓
        P&L Calculation
             ↓
        Margins / EBITDA / Net Profit
             ↓
        API
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_period(
        request.start_date,
        request.end_date,
    )

    service = get_pnl_service()

    result = await _call_service(
        service,
        "get_statement",
        company_id=request.company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        currency=request.currency,
        include_line_items=request.include_line_items,
        user=current_user,
    )

    if isinstance(result, PnLStatement):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No P&L data found for the requested period."
            ),
        )

    return PnLStatement.model_validate(result)


# ============================================================================
# SUMMARY
# ============================================================================


@router.get(
    "/summary/{company_id}",
    response_model=PnLSummaryResponse,
    summary="Get P&L summary",
)
async def get_pnl_summary(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> PnLSummaryResponse:
    """
    Return compact P&L metrics for dashboards.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_pnl_service()

    result = await _call_service(
        service,
        "get_summary",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        currency=currency,
        user=current_user,
    )

    if isinstance(result, PnLSummaryResponse):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="P&L summary not found.",
        )

    return PnLSummaryResponse.model_validate(result)


# ============================================================================
# TREND
# ============================================================================


@router.post(
    "/trend",
    response_model=PnLTrendResponse,
    summary="Get P&L trend",
    description=(
        "Returns historical P&L values for trend and profitability analysis."
    ),
)
async def get_pnl_trend(
    request: PnLTrendRequest,
    current_user: Any = Depends(get_current_user()),
) -> PnLTrendResponse:
    """
    Retrieve historical P&L trend.

    Used by:
        - financial dashboards
        - profitability analysis
        - forecasting
        - anomaly detection
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

    service = get_pnl_service()

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

    if isinstance(result, PnLTrendResponse):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No P&L trend data found.",
        )

    return PnLTrendResponse.model_validate(result)


# ============================================================================
# COMPARISON
# ============================================================================


@router.get(
    "/comparison/{company_id}",
    response_model=PnLComparisonResponse,
    summary="Compare P&L periods",
)
async def compare_pnl(
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
) -> PnLComparisonResponse:
    """
    Compare two P&L periods.

    Useful for:
        - MoM analysis
        - QoQ analysis
        - YoY analysis
        - previous-period decline detection
        - alert generation
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

    service = get_pnl_service()

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

    if isinstance(result, PnLComparisonResponse):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="P&L comparison could not be generated.",
        )

    return PnLComparisonResponse.model_validate(result)


# ============================================================================
# ANALYSIS
# ============================================================================


@router.get(
    "/analysis/{company_id}",
    response_model=PnLAnalysisResponse,
    summary="Analyze P&L",
    description=(
        "Analyzes profitability, margin movement, cost drivers, "
        "and potential business concerns."
    ),
)
async def analyze_pnl(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> PnLAnalysisResponse:
    """
    Perform analytical P&L interpretation.

    This endpoint should not generate autonomous business actions.
    It provides analytical inputs for the recommendation and
    supervisor-agent layers.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_pnl_service()

    result = await _call_service(
        service,
        "analyze",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        currency=currency,
        user=current_user,
    )

    if isinstance(result, PnLAnalysisResponse):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="P&L analysis could not be generated.",
        )

    return PnLAnalysisResponse.model_validate(result)


# ============================================================================
# SPECIALIZED METRIC ENDPOINTS
# ============================================================================


@router.get(
    "/profitability/{company_id}",
    summary="Get profitability metrics",
)
async def get_profitability(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> dict[str, Any]:
    """
    Return profitability-specific metrics.

    Includes:
        - gross profit
        - operating profit
        - EBITDA
        - net profit
        - gross margin
        - operating margin
        - net margin
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_pnl_service()

    result = await _call_service(
        service,
        "profitability",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
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


@router.get(
    "/margins/{company_id}",
    summary="Get P&L margins",
)
async def get_margins(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> dict[str, Any]:
    """
    Return gross, operating, EBITDA and net margins.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_pnl_service()

    result = await _call_service(
        service,
        "margins",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
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
    response_model=PnLHealthResponse,
    summary="P&L service health",
)
async def pnl_health() -> PnLHealthResponse:
    """
    Lightweight P&L subsystem health check.
    """

    started = time.perf_counter()

    try:
        service = get_pnl_service()

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

        return PnLHealthResponse(
            status="healthy",
            service="pnl",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "P&L health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="P&L service is unavailable.",
        )