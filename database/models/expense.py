"""
backend/app/api/expense.py

Expense Management API for FinCo AI.

Responsibilities
----------------
- Expense creation and management
- Expense listing and filtering
- Expense summaries
- Category analysis
- Period comparisons
- Budget variance
- Expense trend analysis
- Anomaly/risk signals
- Tenant isolation
- Authorization

Architecture
------------

    Client
       |
       v
    Expense API
       |
       +---- Authentication
       +---- Authorization
       +---- Tenant Isolation
       |
       v
    ExpenseService
       |
       +---- ExpenseRepository
       +---- Financial Engine
       +---- Budget Engine
       +---- Risk / Anomaly Engine
       |
       v
    PostgreSQL
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
    prefix="/expenses",
    tags=["Expenses"],
)


# ============================================================================
# TYPES
# ============================================================================

ExpenseStatus = Literal[
    "draft",
    "pending",
    "approved",
    "paid",
    "rejected",
    "cancelled",
    "reversed",
]

ExpenseCategory = Literal[
    "cost_of_goods_sold",
    "payroll",
    "marketing",
    "sales",
    "operations",
    "technology",
    "rent",
    "utilities",
    "travel",
    "professional_services",
    "insurance",
    "tax",
    "interest",
    "depreciation",
    "maintenance",
    "office",
    "logistics",
    "other",
]

ExpenseType = Literal[
    "operating",
    "non_operating",
    "capital",
    "tax",
    "financial",
]

PaymentMethod = Literal[
    "cash",
    "bank_transfer",
    "credit_card",
    "debit_card",
    "direct_debit",
    "cheque",
    "upi",
    "other",
]

ExpenseRiskLevel = Literal[
    "very_low",
    "low",
    "medium",
    "high",
    "critical",
]

Granularity = Literal[
    "daily",
    "weekly",
    "monthly",
    "quarterly",
    "yearly",
]

SortField = Literal[
    "date",
    "amount",
    "category",
    "created_at",
]

SortOrder = Literal[
    "asc",
    "desc",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class ExpenseCreateRequest(BaseModel):
    """
    Create an expense transaction.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_date: date

    description: str = Field(
        ...,
        min_length=2,
        max_length=500,
    )

    category: ExpenseCategory

    expense_type: ExpenseType = "operating"

    amount: float = Field(
        ...,
        gt=0,
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
    )

    vendor_id: str | None = Field(
        default=None,
        max_length=150,
    )

    vendor_name: str | None = Field(
        default=None,
        max_length=300,
    )

    customer_id: str | None = Field(
        default=None,
        max_length=150,
    )

    payment_method: PaymentMethod = "bank_transfer"

    status: ExpenseStatus = "approved"

    invoice_number: str | None = Field(
        default=None,
        max_length=150,
    )

    tax_amount: float = Field(
        default=0,
        ge=0,
    )

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("description")
    @classmethod
    def normalize_description(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Description cannot be empty."
            )

        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str,
    ) -> str:
        return value.upper()


class ExpenseUpdateRequest(BaseModel):
    """
    Update mutable expense fields.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_date: date | None = None

    description: str | None = Field(
        default=None,
        min_length=2,
        max_length=500,
    )

    category: ExpenseCategory | None = None

    expense_type: ExpenseType | None = None

    amount: float | None = Field(
        default=None,
        gt=0,
    )

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    vendor_id: str | None = Field(
        default=None,
        max_length=150,
    )

    vendor_name: str | None = Field(
        default=None,
        max_length=300,
    )

    payment_method: PaymentMethod | None = None

    invoice_number: str | None = Field(
        default=None,
        max_length=150,
    )

    tax_amount: float | None = Field(
        default=None,
        ge=0,
    )

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )

    metadata: dict[str, Any] | None = None

    @field_validator("description")
    @classmethod
    def normalize_description(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError(
                "Description cannot be empty."
            )

        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


class ExpenseStatusRequest(BaseModel):
    """
    Change expense lifecycle state.
    """

    model_config = ConfigDict(extra="forbid")

    status: ExpenseStatus

    reason: str | None = Field(
        default=None,
        max_length=2000,
    )


class ExpenseVarianceRequest(BaseModel):
    """
    Request expense-vs-budget analysis.
    """

    start_date: date

    end_date: date

    category: ExpenseCategory | None = None

    budget_id: str | None = None


class ExpenseAnalysisRequest(BaseModel):
    """
    Request advanced expense analysis.
    """

    start_date: date

    end_date: date

    granularity: Granularity = "monthly"

    include_categories: bool = True

    include_vendor_analysis: bool = True

    include_anomalies: bool = True

    include_budget_variance: bool = True


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class ExpenseResponse(BaseModel):
    """
    Expense transaction response.
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
    )

    id: str

    company_id: str

    transaction_date: date

    description: str

    category: ExpenseCategory

    expense_type: ExpenseType

    amount: float

    currency: str

    vendor_id: str | None = None

    vendor_name: str | None = None

    customer_id: str | None = None

    payment_method: PaymentMethod

    status: ExpenseStatus

    invoice_number: str | None = None

    tax_amount: float

    notes: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: datetime

    updated_at: datetime


class ExpenseListResponse(BaseModel):
    """
    Paginated expense list.
    """

    items: list[ExpenseResponse]

    total: int

    page: int

    page_size: int

    pages: int


class ExpenseSummaryResponse(BaseModel):
    """
    Expense aggregate summary.
    """

    company_id: str

    currency: str

    total_expenses: float

    total_tax: float

    expense_count: int

    average_expense: float

    largest_expense: float

    operating_expenses: float

    non_operating_expenses: float

    capital_expenses: float

    expense_growth_percent: float | None = None

    previous_period_expenses: float | None = None

    period_start: date

    period_end: date


class ExpenseCategorySummary(BaseModel):
    """
    Expense category aggregation.
    """

    category: ExpenseCategory

    amount: float

    percentage: float

    transaction_count: int

    average_amount: float

    previous_amount: float | None = None

    growth_percent: float | None = None


class ExpenseCategoryResponse(BaseModel):
    """
    Category-level expense analysis.
    """

    company_id: str

    currency: str

    total_expenses: float

    categories: list[ExpenseCategorySummary]

    period_start: date

    period_end: date


class ExpenseTrendPoint(BaseModel):
    """
    Time-series expense point.
    """

    period: str

    amount: float

    transaction_count: int

    average_amount: float

    previous_amount: float | None = None

    growth_percent: float | None = None


class ExpenseTrendResponse(BaseModel):
    """
    Expense trend analysis.
    """

    company_id: str

    currency: str

    granularity: Granularity

    points: list[ExpenseTrendPoint]

    total_expenses: float

    average_period_expense: float

    period_start: date

    period_end: date


class ExpenseVendorSummary(BaseModel):
    """
    Vendor-level expense aggregation.
    """

    vendor_id: str | None = None

    vendor_name: str

    total_amount: float

    transaction_count: int

    average_amount: float

    percentage_of_total: float

    last_expense_date: date | None = None


class ExpenseVendorResponse(BaseModel):
    """
    Vendor analysis.
    """

    company_id: str

    currency: str

    vendors: list[ExpenseVendorSummary]

    total_expenses: float

    period_start: date

    period_end: date


class ExpenseVarianceResponse(BaseModel):
    """
    Budget vs actual expense variance.
    """

    company_id: str

    currency: str

    budget_id: str | None = None

    budget_amount: float

    actual_amount: float

    variance_amount: float

    variance_percent: float

    status: Literal[
        "favorable",
        "unfavorable",
        "on_track",
    ]

    category: ExpenseCategory | None = None

    period_start: date

    period_end: date


class ExpenseAnomaly(BaseModel):
    """
    Expense anomaly signal.
    """

    expense_id: str | None = None

    category: ExpenseCategory | None = None

    amount: float

    anomaly_score: float = Field(
        ge=0,
        le=100,
    )

    risk_level: ExpenseRiskLevel

    reason: str

    detected_at: datetime


class ExpenseAnomalyResponse(BaseModel):
    """
    Expense anomaly detection response.
    """

    company_id: str

    anomalies: list[ExpenseAnomaly]

    total_anomalies: int

    high_risk_count: int

    critical_count: int


class ExpenseAnalysisResponse(BaseModel):
    """
    Comprehensive expense analysis.
    """

    company_id: str

    currency: str

    period_start: date

    period_end: date

    total_expenses: float

    expense_growth_percent: float | None = None

    category_breakdown: list[ExpenseCategorySummary] = Field(
        default_factory=list,
    )

    trend: list[ExpenseTrendPoint] = Field(
        default_factory=list,
    )

    vendor_breakdown: list[ExpenseVendorSummary] = Field(
        default_factory=list,
    )

    anomalies: list[ExpenseAnomaly] = Field(
        default_factory=list,
    )

    budget_variance: ExpenseVarianceResponse | None = None

    insights: list[str] = Field(
        default_factory=list,
    )


class ExpenseStatusResponse(BaseModel):
    """
    Expense lifecycle update result.
    """

    expense_id: str

    status: ExpenseStatus

    message: str

    updated_at: datetime


class ExpenseDeleteResponse(BaseModel):
    """
    Expense void/reversal response.

    Financial records should not normally be hard deleted.
    """

    expense_id: str

    status: Literal["reversed", "cancelled"]

    message: str

    updated_at: datetime


class ExpenseHealthResponse(BaseModel):
    """
    Expense subsystem health.
    """

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    service: str

    timestamp: datetime

    details: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# AUTHENTICATION
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


# ============================================================================
# SERVICES
# ============================================================================


def get_expense_service() -> Any:
    """
    Preferred:

        backend.app.financial.expenses.ExpenseService

    Fallback:

        backend.app.database.repositories.financial_repository.FinancialRepository
    """

    try:
        from backend.app.financial.expenses import (
            ExpenseService,
        )

        return ExpenseService()

    except ImportError:
        pass

    try:
        from backend.app.database.repositories.financial_repository import (
            FinancialRepository,
        )

        return FinancialRepository()

    except ImportError as exc:
        logger.exception(
            "Expense service/repository unavailable."
        )

        raise RuntimeError(
            "Expense service is not configured."
        ) from exc


# ============================================================================
# SECURITY
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


def _can_manage_expenses(
    current_user: Any,
) -> bool:
    """
    Roles allowed to create/update/manage expenses.
    """

    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
        "finance_manager",
        "analyst",
    }


def _can_approve_expenses(
    current_user: Any,
) -> bool:
    """
    Roles allowed to approve/reverse financial expenses.
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
            detail="User is not associated with a company.",
        )

    if str(user_company_id) != str(company_id):
        logger.warning(
            "Unauthorized expense company access: "
            "user_company=%s requested_company=%s "
            "user=%s",
            user_company_id,
            company_id,
            _get_user_id(current_user),
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this company."
            ),
        )


def require_expense_manager(
    current_user: Any,
) -> None:
    if not _can_manage_expenses(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Expense management privileges are required."
            ),
        )


def require_expense_approver(
    current_user: Any,
) -> None:
    if not _can_approve_expenses(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Expense approval privileges are required."
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
    Support both synchronous and asynchronous services.
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
                f"Expense service does not implement "
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
            "Invalid expense request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid expense request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Expense service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Expense operation failed.",
        ) from exc


# ============================================================================
# CREATE
# ============================================================================


@router.post(
    "",
    response_model=ExpenseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create expense",
)
async def create_expense(
    request: ExpenseCreateRequest,
    current_user: Any = Depends(get_current_user()),
) -> ExpenseResponse:
    """
    Create a new expense.

    Financial calculations remain inside ExpenseService.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_expense_manager(
        current_user
    )

    service = get_expense_service()

    result = await _call_service(
        service,
        "create",
        company_id=company_id,
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Expense creation returned no result.",
        )

    return ExpenseResponse.model_validate(
        result
    )


# ============================================================================
# LIST
# ============================================================================


@router.get(
    "",
    response_model=ExpenseListResponse,
    summary="List expenses",
)
async def list_expenses(
    start_date: date | None = None,
    end_date: date | None = None,
    category: ExpenseCategory | None = None,
    expense_type: ExpenseType | None = None,
    expense_status: ExpenseStatus | None = Query(
        default=None,
        alias="status",
    ),
    payment_method: PaymentMethod | None = None,
    vendor_id: str | None = None,
    min_amount: float | None = Query(
        default=None,
        ge=0,
    ),
    max_amount: float | None = Query(
        default=None,
        ge=0,
    ),
    search: str | None = Query(
        default=None,
        min_length=1,
        max_length=200,
    ),
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    sort_by: SortField = "date",
    sort_order: SortOrder = "desc",
    current_user: Any = Depends(get_current_user()),
) -> ExpenseListResponse:
    """
    List expenses for the authenticated tenant.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if (
        start_date
        and end_date
        and start_date > end_date
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    if (
        min_amount is not None
        and max_amount is not None
        and min_amount > max_amount
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "min_amount cannot be greater than max_amount."
            ),
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        category=category,
        expense_type=expense_type,
        status=expense_status,
        payment_method=payment_method,
        vendor_id=vendor_id,
        min_amount=min_amount,
        max_amount=max_amount,
        search=search.strip()
        if search
        else None,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        user=current_user,
    )

    if result is None:
        return ExpenseListResponse(
            items=[],
            total=0,
            page=page,
            page_size=page_size,
            pages=0,
        )

    return ExpenseListResponse.model_validate(
        result
    )


# ============================================================================
# GET
# ============================================================================


@router.get(
    "/{expense_id}",
    response_model=ExpenseResponse,
    summary="Get expense",
)
async def get_expense(
    expense_id: str,
    current_user: Any = Depends(get_current_user()),
) -> ExpenseResponse:
    """
    Retrieve one expense.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "get",
        expense_id=expense_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense not found.",
        )

    return ExpenseResponse.model_validate(
        result
    )


# ============================================================================
# UPDATE
# ============================================================================


@router.patch(
    "/{expense_id}",
    response_model=ExpenseResponse,
    summary="Update expense",
)
async def update_expense(
    expense_id: str,
    request: ExpenseUpdateRequest,
    current_user: Any = Depends(get_current_user()),
) -> ExpenseResponse:
    """
    Update an expense.

    Posted financial records should normally be controlled by
    accounting rules and audit workflows.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_expense_manager(
        current_user
    )

    service = get_expense_service()

    result = await _call_service(
        service,
        "update",
        expense_id=expense_id,
        company_id=company_id,
        updates=request.model_dump(
            exclude_unset=True
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense not found.",
        )

    return ExpenseResponse.model_validate(
        result
    )


# ============================================================================
# SUMMARY
# ============================================================================


@router.get(
    "/summary/overview",
    response_model=ExpenseSummaryResponse,
    summary="Get expense summary",
)
async def expense_summary(
    start_date: date,
    end_date: date,
    current_user: Any = Depends(get_current_user()),
) -> ExpenseSummaryResponse:
    """
    Get total expense KPIs for a period.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "summary",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense summary unavailable.",
        )

    return ExpenseSummaryResponse.model_validate(
        result
    )


# ============================================================================
# CATEGORY ANALYSIS
# ============================================================================


@router.get(
    "/analysis/categories",
    response_model=ExpenseCategoryResponse,
    summary="Analyze expenses by category",
)
async def expense_categories(
    start_date: date,
    end_date: date,
    current_user: Any = Depends(get_current_user()),
) -> ExpenseCategoryResponse:
    """
    Analyze expense distribution by category.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "category_analysis",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense category analysis unavailable.",
        )

    return ExpenseCategoryResponse.model_validate(
        result
    )


# ============================================================================
# TREND
# ============================================================================


@router.get(
    "/analysis/trend",
    response_model=ExpenseTrendResponse,
    summary="Analyze expense trend",
)
async def expense_trend(
    start_date: date,
    end_date: date,
    granularity: Granularity = "monthly",
    current_user: Any = Depends(get_current_user()),
) -> ExpenseTrendResponse:
    """
    Analyze expense trends over time.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "trend",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense trend unavailable.",
        )

    return ExpenseTrendResponse.model_validate(
        result
    )


# ============================================================================
# VENDOR ANALYSIS
# ============================================================================


@router.get(
    "/analysis/vendors",
    response_model=ExpenseVendorResponse,
    summary="Analyze expenses by vendor",
)
async def expense_vendors(
    start_date: date,
    end_date: date,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    current_user: Any = Depends(get_current_user()),
) -> ExpenseVendorResponse:
    """
    Identify major expense vendors.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "vendor_analysis",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense vendor analysis unavailable.",
        )

    return ExpenseVendorResponse.model_validate(
        result
    )


# ============================================================================
# BUDGET VARIANCE
# ============================================================================


@router.post(
    "/analysis/variance",
    response_model=ExpenseVarianceResponse,
    summary="Analyze expense budget variance",
)
async def expense_variance(
    request: ExpenseVarianceRequest,
    current_user: Any = Depends(get_current_user()),
) -> ExpenseVarianceResponse:
    """
    Compare actual expenses against an approved budget.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.start_date > request.end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "budget_variance",
        company_id=company_id,
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense variance unavailable.",
        )

    return ExpenseVarianceResponse.model_validate(
        result
    )


# ============================================================================
# ANOMALY DETECTION
# ============================================================================


@router.get(
    "/analysis/anomalies",
    response_model=ExpenseAnomalyResponse,
    summary="Detect expense anomalies",
)
async def expense_anomalies(
    start_date: date,
    end_date: date,
    threshold: float = Query(
        default=70,
        ge=0,
        le=100,
    ),
    current_user: Any = Depends(get_current_user()),
) -> ExpenseAnomalyResponse:
    """
    Detect unusually large or unusual expenses.

    The actual anomaly model belongs to the financial/risk layer.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "detect_anomalies",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        threshold=threshold,
        user=current_user,
    )

    if result is None:
        return ExpenseAnomalyResponse(
            company_id=str(company_id),
            anomalies=[],
            total_anomalies=0,
            high_risk_count=0,
            critical_count=0,
        )

    return ExpenseAnomalyResponse.model_validate(
        result
    )


# ============================================================================
# COMPREHENSIVE ANALYSIS
# ============================================================================


@router.post(
    "/analysis",
    response_model=ExpenseAnalysisResponse,
    summary="Run comprehensive expense analysis",
)
async def expense_analysis(
    request: ExpenseAnalysisRequest,
    current_user: Any = Depends(get_current_user()),
) -> ExpenseAnalysisResponse:
    """
    Run comprehensive expense analysis.

    Intended as the main endpoint consumed by dashboards,
    reports, recommendation agents, and the financial copilot.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.start_date > request.end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "analyze",
        company_id=company_id,
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense analysis unavailable.",
        )

    return ExpenseAnalysisResponse.model_validate(
        result
    )


# ============================================================================
# STATUS
# ============================================================================


@router.patch(
    "/{expense_id}/status",
    response_model=ExpenseStatusResponse,
    summary="Change expense status",
)
async def change_expense_status(
    expense_id: str,
    request: ExpenseStatusRequest,
    current_user: Any = Depends(get_current_user()),
) -> ExpenseStatusResponse:
    """
    Change expense status.

    Approval/rejection/reversal should be audited.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.status in {
        "approved",
        "rejected",
        "paid",
        "reversed",
    }:
        require_expense_approver(
            current_user
        )
    else:
        require_expense_manager(
            current_user
        )

    service = get_expense_service()

    result = await _call_service(
        service,
        "change_status",
        expense_id=expense_id,
        company_id=company_id,
        status=request.status,
        reason=request.reason,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense not found.",
        )

    return ExpenseStatusResponse.model_validate(
        result
    )


# ============================================================================
# REVERSE / VOID
# ============================================================================


@router.delete(
    "/{expense_id}",
    response_model=ExpenseDeleteResponse,
    summary="Reverse or cancel expense",
)
async def reverse_expense(
    expense_id: str,
    reason: str = Query(
        ...,
        min_length=3,
        max_length=2000,
    ),
    current_user: Any = Depends(get_current_user()),
) -> ExpenseDeleteResponse:
    """
    Reverse/cancel an expense.

    This endpoint intentionally does not hard-delete financial
    history.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_expense_approver(
        current_user
    )

    service = get_expense_service()

    result = await _call_service(
        service,
        "reverse",
        expense_id=expense_id,
        company_id=company_id,
        reason=reason,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Expense not found.",
        )

    return ExpenseDeleteResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=ExpenseHealthResponse,
    summary="Expense service health",
)
async def expense_health() -> ExpenseHealthResponse:
    """
    Expense subsystem health check.
    """

    started = time.perf_counter()

    try:
        service = get_expense_service()

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

        return ExpenseHealthResponse(
            status="healthy",
            service="expense",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Expense service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Expense service is unavailable.",
        )