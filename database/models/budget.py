"""
backend/app/api/budget.py

Budget Management API for FinCo AI.

Responsibilities
----------------
- Create and manage budgets
- Retrieve budget statements
- Budget vs actual analysis
- Variance analysis
- Department/category budget views
- Budget forecasting
- Budget approval workflow
- Budget versioning
- Tenant isolation
- Audit-friendly lifecycle management

Architecture
------------

    Client
       |
       v
    Budget API
       |
       +---- Authentication
       +---- Tenant Isolation
       +---- Validation
       |
       v
    BudgetService
       |
       +---- Budget Repository
       +---- Financial Engine
       +---- Transaction Data
       +---- Forecasting
       +---- Risk Engine
       |
       v
    Budget Result
       |
       +---- Alerts
       +---- Recommendations
       +---- Audit Log
"""

from __future__ import annotations

import inspect
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
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/budget",
    tags=["Budget"],
)


# ============================================================================
# TYPES
# ============================================================================

BudgetStatus = Literal[
    "draft",
    "pending_approval",
    "approved",
    "active",
    "closed",
    "archived",
]

BudgetPeriodType = Literal[
    "monthly",
    "quarterly",
    "yearly",
]

BudgetCategoryType = Literal[
    "revenue",
    "cost_of_goods_sold",
    "payroll",
    "marketing",
    "operations",
    "technology",
    "rent",
    "utilities",
    "tax",
    "capital_expenditure",
    "other",
]

VarianceStatus = Literal[
    "favorable",
    "unfavorable",
    "on_track",
]

ApprovalAction = Literal[
    "approve",
    "reject",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class BudgetLineItemRequest(BaseModel):
    """
    Individual budget line item.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    category: str = Field(
        ...,
        min_length=1,
        max_length=150,
    )

    category_type: BudgetCategoryType = "other"

    amount: float = Field(
        ...,
        ge=0,
    )

    notes: str | None = Field(
        default=None,
        max_length=1000,
    )

    @field_validator("category")
    @classmethod
    def normalize_category(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Category cannot be empty."
            )

        return value


class BudgetCreateRequest(BaseModel):
    """
    Create a new budget.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    period_type: BudgetPeriodType

    start_date: date

    end_date: date

    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
    )

    line_items: list[BudgetLineItemRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    notes: str | None = Field(
        default=None,
        max_length=5000,
    )

    auto_generate_forecast: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Budget name cannot be empty."
            )

        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str,
    ) -> str:
        return value.upper()

    @field_validator("end_date")
    @classmethod
    def validate_dates(
        cls,
        value: date,
        info,
    ) -> date:
        start_date = info.data.get(
            "start_date"
        )

        if (
            start_date is not None
            and value <= start_date
        ):
            raise ValueError(
                "end_date must be after start_date."
            )

        return value


class BudgetUpdateRequest(BaseModel):
    """
    Update a draft budget.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    line_items: list[BudgetLineItemRequest] | None = (
        None
    )

    notes: str | None = Field(
        default=None,
        max_length=5000,
    )


class BudgetApprovalRequest(BaseModel):
    """
    Approve or reject a budget.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    action: ApprovalAction

    reason: str | None = Field(
        default=None,
        max_length=2000,
    )


class BudgetActualsRequest(BaseModel):
    """
    Request budget-vs-actual analysis.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    budget_id: str

    as_of_date: date | None = None

    include_forecast: bool = True

    include_risk: bool = True

    include_recommendations: bool = True


class BudgetVarianceRequest(BaseModel):
    """
    Request variance analysis.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    budget_id: str

    threshold_percent: float = Field(
        default=10.0,
        ge=0,
        le=1000,
    )

    include_favorable: bool = True

    include_unfavorable: bool = True


class BudgetForecastRequest(BaseModel):
    """
    Request budget forecast.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    budget_id: str

    horizon_months: int = Field(
        default=12,
        ge=1,
        le=120,
    )

    include_risk: bool = True


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class BudgetLineItemResponse(BaseModel):
    """
    Budget line item.
    """

    id: str | None = None

    category: str

    category_type: BudgetCategoryType

    budget_amount: float

    actual_amount: float | None = None

    variance_amount: float | None = None

    variance_percent: float | None = None

    variance_status: VarianceStatus | None = None

    notes: str | None = None


class BudgetResponse(BaseModel):
    """
    Budget representation.
    """

    id: str

    company_id: str

    name: str

    version: int

    status: BudgetStatus

    period_type: BudgetPeriodType

    start_date: date

    end_date: date

    currency: str

    total_budget: float

    total_actual: float | None = None

    total_variance: float | None = None

    total_variance_percent: float | None = None

    line_items: list[BudgetLineItemResponse] = Field(
        default_factory=list
    )

    notes: str | None = None

    created_by: str | None = None

    approved_by: str | None = None

    approved_at: datetime | None = None

    created_at: datetime

    updated_at: datetime


class BudgetListResponse(BaseModel):
    """
    Paginated budget list.
    """

    company_id: str

    items: list[BudgetResponse]

    total: int

    limit: int

    offset: int

    has_more: bool


class BudgetSummaryResponse(BaseModel):
    """
    High-level budget summary.
    """

    company_id: str

    budget_id: str

    budget_name: str

    total_budget: float

    total_actual: float

    remaining_budget: float

    utilization_percent: float

    variance_amount: float

    variance_percent: float

    favorable_variance: float

    unfavorable_variance: float

    categories_over_budget: int

    categories_under_budget: int

    status: BudgetStatus


class BudgetActualResponse(BaseModel):
    """
    Budget vs actual analysis.
    """

    company_id: str

    budget_id: str

    as_of_date: date

    total_budget: float

    total_actual: float

    total_variance: float

    total_variance_percent: float

    utilization_percent: float

    line_items: list[BudgetLineItemResponse] = Field(
        default_factory=list
    )

    key_findings: list[str] = Field(
        default_factory=list
    )


class BudgetVarianceResponse(BaseModel):
    """
    Budget variance analysis.
    """

    company_id: str

    budget_id: str

    threshold_percent: float

    favorable_variances: list[
        BudgetLineItemResponse
    ] = Field(default_factory=list)

    unfavorable_variances: list[
        BudgetLineItemResponse
    ] = Field(default_factory=list)

    on_track_items: list[
        BudgetLineItemResponse
    ] = Field(default_factory=list)

    total_favorable: float = 0.0

    total_unfavorable: float = 0.0

    largest_unfavorable_category: str | None = None

    key_findings: list[str] = Field(
        default_factory=list
    )


class BudgetForecastPoint(BaseModel):
    """
    Forecasted budget utilization for one period.
    """

    period: str

    budget_amount: float

    projected_actual: float

    projected_variance: float

    projected_utilization_percent: float


class BudgetForecastResponse(BaseModel):
    """
    Budget forecast.
    """

    company_id: str

    budget_id: str

    horizon_months: int

    points: list[BudgetForecastPoint] = Field(
        default_factory=list
    )

    projected_total_budget: float

    projected_total_actual: float

    projected_variance: float

    risk_score: float | None = None

    risk_level: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ] | None = None

    key_findings: list[str] = Field(
        default_factory=list
    )


class BudgetApprovalResponse(BaseModel):
    """
    Budget approval workflow response.
    """

    budget_id: str

    status: BudgetStatus

    action: ApprovalAction

    approved_by: str | None = None

    reason: str | None = None

    updated_at: datetime


class BudgetVersionResponse(BaseModel):
    """
    Budget version information.
    """

    budget_id: str

    version: int

    status: BudgetStatus

    created_at: datetime

    created_by: str | None = None

    is_current: bool


class BudgetDeleteResponse(BaseModel):
    """
    Archive response.
    """

    budget_id: str

    status: Literal[
        "archived",
    ]

    message: str

    updated_at: datetime


class BudgetHealthResponse(BaseModel):
    """
    Budget subsystem health.
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


def get_budget_service() -> Any:
    """
    Lazily load BudgetService.

    Expected implementation:

        backend.app.financial.budget.BudgetService

    or:

        backend.app.budget.budget_service.BudgetService
    """

    try:
        from backend.app.financial.budget import (
            BudgetService,
        )

        return BudgetService()

    except ImportError:
        pass

    try:
        from backend.app.budget.budget_service import (
            BudgetService,
        )

        return BudgetService()

    except ImportError as exc:
        logger.exception(
            "BudgetService unavailable."
        )

        raise RuntimeError(
            "Budget service is not configured."
        ) from exc


# ============================================================================
# SECURITY HELPERS
# ============================================================================


def _get_user_id(
    current_user: Any,
) -> str | None:
    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return (
            current_user.get("id")
            or current_user.get("user_id")
        )

    return (
        getattr(current_user, "id", None)
        or getattr(
            current_user,
            "user_id",
            None,
        )
    )


def _get_company_id(
    current_user: Any,
) -> str | None:
    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("company_id")

    return getattr(
        current_user,
        "company_id",
        None,
    )


def _get_role(
    current_user: Any,
) -> str | None:
    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("role")

    return getattr(
        current_user,
        "role",
        None,
    )


def _is_admin(
    current_user: Any,
) -> bool:
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


def _can_manage_budget(
    current_user: Any,
) -> bool:
    """
    Roles allowed to create/update/approve budgets.
    """

    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
        "finance_manager",
    }


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce tenant isolation.
    """

    if _is_admin(current_user):
        return

    user_company_id = _get_company_id(
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
            "Unauthorized budget access: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this company."
            ),
        )


def require_budget_manager(
    current_user: Any,
) -> None:
    """
    Require budget-management privileges.
    """

    if not _can_manage_budget(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Budget-management privileges are required."
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
    Invoke sync/async budget service methods.
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
                f"Budget service does not implement "
                f"'{method_name}'."
            ),
        )

    try:
        result = method(**kwargs)

        if inspect.isawaitable(result):
            result = await result

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid budget request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid budget request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Budget service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Budget operation failed.",
        ) from exc


# ============================================================================
# CREATE
# ============================================================================


@router.post(
    "",
    response_model=BudgetResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create budget",
)
async def create_budget(
    request: BudgetCreateRequest,
    current_user: Any = Depends(get_current_user()),
) -> BudgetResponse:
    """
    Create a new budget.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    require_budget_manager(
        current_user
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "create",
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Budget creation returned no result.",
        )

    return BudgetResponse.model_validate(
        result
    )


# ============================================================================
# LIST
# ============================================================================


@router.get(
    "/company/{company_id}",
    response_model=BudgetListResponse,
    summary="List company budgets",
)
async def list_budgets(
    company_id: str,
    budget_status: BudgetStatus | None = Query(
        default=None,
        alias="status",
    ),
    period_type: BudgetPeriodType | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    current_user: Any = Depends(get_current_user()),
) -> BudgetListResponse:
    """
    List budgets belonging to a company.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        status=budget_status,
        period_type=period_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
        user=current_user,
    )

    if result is None:
        return BudgetListResponse(
            company_id=company_id,
            items=[],
            total=0,
            limit=limit,
            offset=offset,
            has_more=False,
        )

    return BudgetListResponse.model_validate(
        result
    )


# ============================================================================
# GET
# ============================================================================


@router.get(
    "/{budget_id}",
    response_model=BudgetResponse,
    summary="Get budget",
)
async def get_budget(
    budget_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> BudgetResponse:
    """
    Retrieve a budget.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "get",
        budget_id=budget_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found.",
        )

    return BudgetResponse.model_validate(
        result
    )


# ============================================================================
# UPDATE
# ============================================================================


@router.patch(
    "/{budget_id}",
    response_model=BudgetResponse,
    summary="Update budget",
)
async def update_budget(
    budget_id: str,
    request: BudgetUpdateRequest,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> BudgetResponse:
    """
    Update a budget.

    Approved/active budgets should normally be versioned
    rather than silently modified.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_budget_manager(
        current_user
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "update",
        budget_id=budget_id,
        company_id=company_id,
        updates=request.model_dump(
            exclude_unset=True
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found.",
        )

    return BudgetResponse.model_validate(
        result
    )


# ============================================================================
# SUMMARY
# ============================================================================


@router.get(
    "/{budget_id}/summary",
    response_model=BudgetSummaryResponse,
    summary="Get budget summary",
)
async def budget_summary(
    budget_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> BudgetSummaryResponse:
    """
    Get high-level budget performance.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "summary",
        budget_id=budget_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found.",
        )

    return BudgetSummaryResponse.model_validate(
        result
    )


# ============================================================================
# ACTUALS
# ============================================================================


@router.post(
    "/actuals",
    response_model=BudgetActualResponse,
    summary="Compare budget against actuals",
)
async def budget_actuals(
    request: BudgetActualsRequest,
    current_user: Any = Depends(get_current_user()),
) -> BudgetActualResponse:
    """
    Compare budgeted amounts with actual financial activity.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "actuals",
        request=request,
        user=current_user,
    )

    return BudgetActualResponse.model_validate(
        result
    )


# ============================================================================
# VARIANCE
# ============================================================================


@router.post(
    "/variance",
    response_model=BudgetVarianceResponse,
    summary="Analyze budget variance",
)
async def budget_variance(
    request: BudgetVarianceRequest,
    current_user: Any = Depends(get_current_user()),
) -> BudgetVarianceResponse:
    """
    Analyze favorable and unfavorable budget variances.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "variance",
        request=request,
        user=current_user,
    )

    return BudgetVarianceResponse.model_validate(
        result
    )


# ============================================================================
# FORECAST
# ============================================================================


@router.post(
    "/forecast",
    response_model=BudgetForecastResponse,
    summary="Forecast budget utilization",
)
async def budget_forecast(
    request: BudgetForecastRequest,
    current_user: Any = Depends(get_current_user()),
) -> BudgetForecastResponse:
    """
    Forecast future budget utilization.

    The actual forecasting model belongs in the forecasting
    subsystem, not this API module.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "forecast",
        request=request,
        user=current_user,
    )

    return BudgetForecastResponse.model_validate(
        result
    )


# ============================================================================
# APPROVAL
# ============================================================================


@router.post(
    "/{budget_id}/approval",
    response_model=BudgetApprovalResponse,
    summary="Approve or reject budget",
)
async def approve_budget(
    budget_id: str,
    request: BudgetApprovalRequest,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> BudgetApprovalResponse:
    """
    Approve or reject a budget.

    Approval should generate an audit event.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_budget_manager(
        current_user
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "approval",
        budget_id=budget_id,
        company_id=company_id,
        action=request.action,
        reason=request.reason,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found.",
        )

    return BudgetApprovalResponse.model_validate(
        result
    )


# ============================================================================
# VERSIONING
# ============================================================================


@router.post(
    "/{budget_id}/version",
    response_model=BudgetResponse,
    summary="Create budget version",
)
async def create_budget_version(
    budget_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> BudgetResponse:
    """
    Create a new version of a budget.

    Versioning preserves the historical budget instead of
    overwriting an approved budget.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_budget_manager(
        current_user
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "create_version",
        budget_id=budget_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found.",
        )

    return BudgetResponse.model_validate(
        result
    )


@router.get(
    "/{budget_id}/versions",
    response_model=list[BudgetVersionResponse],
    summary="List budget versions",
)
async def list_budget_versions(
    budget_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> list[BudgetVersionResponse]:
    """
    List all versions of a budget.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "versions",
        budget_id=budget_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        return []

    return [
        BudgetVersionResponse.model_validate(
            item
        )
        for item in result
    ]


# ============================================================================
# ARCHIVE
# ============================================================================


@router.delete(
    "/{budget_id}",
    response_model=BudgetDeleteResponse,
    summary="Archive budget",
)
async def archive_budget(
    budget_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> BudgetDeleteResponse:
    """
    Archive a budget.

    Hard deletion is intentionally avoided because financial
    planning history is auditable business data.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_budget_manager(
        current_user
    )

    service = get_budget_service()

    result = await _call_service(
        service,
        "archive",
        budget_id=budget_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Budget not found.",
        )

    return BudgetDeleteResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=BudgetHealthResponse,
    summary="Budget service health",
)
async def budget_health() -> BudgetHealthResponse:
    """
    Health check for budget infrastructure.
    """

    started = time.perf_counter()

    try:
        service = get_budget_service()

        details: dict[str, Any] = {}

        health_method = getattr(
            service,
            "health",
            None,
        )

        if health_method is not None:
            result = health_method()

            if inspect.isawaitable(result):
                result = await result

            if isinstance(result, dict):
                details = result

        details["latency_ms"] = round(
            (
                time.perf_counter() - started
            ) * 1000,
            2,
        )

        return BudgetHealthResponse(
            status="healthy",
            service="budget",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Budget service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Budget service is unavailable.",
        )