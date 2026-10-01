"""
backend/app/api/customer.py

Customer / CRM API for FinCo AI.

Responsibilities
----------------
- Customer creation and management
- Customer profile and contact information
- Customer financial summaries
- Customer transaction summaries
- Customer risk information
- Customer search and filtering
- Customer lifecycle management
- Tenant isolation
- Authorization
- Audit-friendly operations

Architecture
------------

    Client
       |
       v
    Customer API
       |
       +---- Authentication
       +---- Authorization
       +---- Tenant Isolation
       |
       v
    CustomerService
       |
       +---- CustomerRepository
       +---- TransactionRepository
       +---- Risk/Fraud Services
       +---- AuditService
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
    EmailStr,
    Field,
    field_validator,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/customers",
    tags=["Customers"],
)


# ============================================================================
# TYPES
# ============================================================================

CustomerType = Literal[
    "individual",
    "business",
    "enterprise",
    "government",
    "non_profit",
    "other",
]

CustomerStatus = Literal[
    "lead",
    "active",
    "inactive",
    "suspended",
    "blocked",
    "archived",
]

CustomerRiskLevel = Literal[
    "very_low",
    "low",
    "medium",
    "high",
    "critical",
]

CustomerTier = Literal[
    "standard",
    "silver",
    "gold",
    "platinum",
    "enterprise",
]

SortField = Literal[
    "name",
    "created_at",
    "updated_at",
    "total_revenue",
    "outstanding_balance",
    "risk_score",
]

SortOrder = Literal[
    "asc",
    "desc",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class CustomerCreateRequest(BaseModel):
    """
    Create a customer within the authenticated tenant.
    """

    model_config = ConfigDict(extra="forbid")

    external_id: str | None = Field(
        default=None,
        max_length=150,
    )

    customer_type: CustomerType = "individual"

    name: str = Field(
        ...,
        min_length=2,
        max_length=250,
    )

    legal_name: str | None = Field(
        default=None,
        max_length=300,
    )

    email: EmailStr | None = None

    phone: str | None = Field(
        default=None,
        max_length=50,
    )

    tax_id: str | None = Field(
        default=None,
        max_length=100,
    )

    industry: str | None = Field(
        default=None,
        max_length=150,
    )

    country: str | None = Field(
        default=None,
        min_length=2,
        max_length=2,
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
    )

    tier: CustomerTier = "standard"

    status: CustomerStatus = "active"

    credit_limit: float = Field(
        default=0,
        ge=0,
    )

    payment_terms_days: int = Field(
        default=30,
        ge=0,
        le=365,
    )

    address: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("name", "legal_name")
    @classmethod
    def normalize_names(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None

    @field_validator("external_id")
    @classmethod
    def normalize_external_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None

    @field_validator("country")
    @classmethod
    def normalize_country(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str,
    ) -> str:
        return value.upper()


class CustomerUpdateRequest(BaseModel):
    """
    Update mutable customer information.
    """

    model_config = ConfigDict(extra="forbid")

    external_id: str | None = Field(
        default=None,
        max_length=150,
    )

    customer_type: CustomerType | None = None

    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=250,
    )

    legal_name: str | None = Field(
        default=None,
        max_length=300,
    )

    email: EmailStr | None = None

    phone: str | None = Field(
        default=None,
        max_length=50,
    )

    tax_id: str | None = Field(
        default=None,
        max_length=100,
    )

    industry: str | None = Field(
        default=None,
        max_length=150,
    )

    country: str | None = Field(
        default=None,
        min_length=2,
        max_length=2,
    )

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    tier: CustomerTier | None = None

    credit_limit: float | None = Field(
        default=None,
        ge=0,
    )

    payment_terms_days: int | None = Field(
        default=None,
        ge=0,
        le=365,
    )

    address: dict[str, Any] | None = None

    metadata: dict[str, Any] | None = None

    @field_validator("name", "legal_name")
    @classmethod
    def normalize_names(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None

    @field_validator("country")
    @classmethod
    def normalize_country(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


class CustomerStatusRequest(BaseModel):
    """
    Change customer lifecycle state.
    """

    model_config = ConfigDict(extra="forbid")

    status: CustomerStatus

    reason: str | None = Field(
        default=None,
        max_length=2000,
    )


class CustomerRiskRequest(BaseModel):
    """
    Request a customer risk assessment.
    """

    model_config = ConfigDict(extra="forbid")

    include_transactions: bool = True

    include_payment_history: bool = True

    include_fraud_signals: bool = True

    include_credit_exposure: bool = True

    lookback_days: int = Field(
        default=365,
        ge=1,
        le=3650,
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class CustomerResponse(BaseModel):
    """
    Customer profile.
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
    )

    id: str

    company_id: str

    external_id: str | None = None

    customer_type: CustomerType

    name: str

    legal_name: str | None = None

    email: str | None = None

    phone: str | None = None

    tax_id: str | None = None

    industry: str | None = None

    country: str | None = None

    currency: str

    tier: CustomerTier

    status: CustomerStatus

    credit_limit: float

    payment_terms_days: int

    address: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: datetime

    updated_at: datetime


class CustomerListResponse(BaseModel):
    """
    Paginated customer list.
    """

    items: list[CustomerResponse]

    total: int

    page: int

    page_size: int

    pages: int


class CustomerFinancialSummary(BaseModel):
    """
    Financial summary for a customer.
    """

    customer_id: str

    currency: str

    total_revenue: float

    total_expenses: float

    total_transactions: int

    outstanding_balance: float

    overdue_balance: float

    paid_amount: float

    average_transaction_value: float

    revenue_growth_percent: float | None = None

    payment_delay_days: float | None = None

    last_transaction_date: date | None = None


class CustomerRiskResponse(BaseModel):
    """
    Customer financial/risk assessment.
    """

    customer_id: str

    risk_score: float = Field(
        ge=0,
        le=100,
    )

    risk_level: CustomerRiskLevel

    fraud_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    credit_utilization_percent: float | None = Field(
        default=None,
        ge=0,
    )

    outstanding_balance: float

    overdue_balance: float

    payment_delay_days: float | None = None

    risk_factors: list[str] = Field(
        default_factory=list,
    )

    assessed_at: datetime


class CustomerTransactionSummary(BaseModel):
    """
    Transaction statistics for a customer.
    """

    customer_id: str

    total_transactions: int

    income_transactions: int

    expense_transactions: int

    refund_transactions: int

    pending_transactions: int

    failed_transactions: int

    total_income: float

    total_expenses: float

    total_refunds: float

    average_transaction_amount: float


class CustomerStatusResponse(BaseModel):
    """
    Customer lifecycle update result.
    """

    customer_id: str

    status: CustomerStatus

    message: str

    updated_at: datetime


class CustomerDeleteResponse(BaseModel):
    """
    Archive operation result.

    Customers are archived rather than physically deleted.
    """

    customer_id: str

    status: Literal["archived"]

    message: str

    updated_at: datetime


class CustomerHealthResponse(BaseModel):
    """
    Customer subsystem health.
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
    Lazily load the authentication dependency.
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


def get_customer_service() -> Any:
    """
    Load CustomerService.

    Preferred:

        backend.app.customers.customer_service.CustomerService

    Fallback:

        backend.app.database.repositories.customer_repository.CustomerRepository
    """

    try:
        from backend.app.customers.customer_service import (
            CustomerService,
        )

        return CustomerService()

    except ImportError:
        pass

    try:
        from backend.app.database.repositories.customer_repository import (
            CustomerRepository,
        )

        return CustomerRepository()

    except ImportError as exc:
        logger.exception(
            "Customer service/repository unavailable."
        )

        raise RuntimeError(
            "Customer service is not configured."
        ) from exc


def get_customer_risk_service() -> Any:
    """
    Load the customer risk service.

    Preferred implementation:

        backend.app.risk.risk_service.RiskService
    """

    try:
        from backend.app.risk.risk_service import (
            RiskService,
        )

        return RiskService()

    except ImportError:
        return None


# ============================================================================
# SECURITY
# ============================================================================


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


def _is_customer_manager(
    current_user: Any,
) -> bool:
    """
    Roles allowed to manage customer records.
    """

    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
        "finance_manager",
        "analyst",
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

    if str(user_company_id) != str(company_id):
        logger.warning(
            "Unauthorized customer/company access: "
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


def require_customer_manager(
    current_user: Any,
) -> None:
    if not _is_customer_manager(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Customer management privileges are required."
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
    Supports both synchronous and asynchronous services.
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
                f"Customer service does not implement "
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
            "Invalid customer request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid customer request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Customer service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Customer operation failed.",
        ) from exc


# ============================================================================
# CREATE
# ============================================================================


@router.post(
    "",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create customer",
)
async def create_customer(
    request: CustomerCreateRequest,
    current_user: Any = Depends(get_current_user()),
) -> CustomerResponse:
    """
    Create a customer inside the authenticated tenant.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_customer_manager(
        current_user
    )

    service = get_customer_service()

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
            detail="Customer creation returned no result.",
        )

    return CustomerResponse.model_validate(
        result
    )


# ============================================================================
# LIST / SEARCH
# ============================================================================


@router.get(
    "",
    response_model=CustomerListResponse,
    summary="List customers",
)
async def list_customers(
    status_filter: CustomerStatus | None = Query(
        default=None,
        alias="status",
    ),
    customer_type: CustomerType | None = None,
    tier: CustomerTier | None = None,
    risk_level: CustomerRiskLevel | None = None,
    search: str | None = Query(
        default=None,
        min_length=1,
        max_length=200,
    ),
    page: int = Query(
        default=1,
        ge=1,
        le=100000,
    ),
    page_size: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    sort_by: SortField = "created_at",
    sort_order: SortOrder = "desc",
    current_user: Any = Depends(get_current_user()),
) -> CustomerListResponse:
    """
    List customers for the authenticated tenant.

    Tenant/company ID is never accepted from the client.
    It is derived from the authenticated user.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_customer_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        status=status_filter,
        customer_type=customer_type,
        tier=tier,
        risk_level=risk_level,
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
        return CustomerListResponse(
            items=[],
            total=0,
            page=page,
            page_size=page_size,
            pages=0,
        )

    return CustomerListResponse.model_validate(
        result
    )


# ============================================================================
# SEARCH
# ============================================================================


@router.get(
    "/search",
    response_model=CustomerListResponse,
    summary="Search customers",
)
async def search_customers(
    q: str = Query(
        ...,
        min_length=2,
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
    current_user: Any = Depends(get_current_user()),
) -> CustomerListResponse:
    """
    Dedicated customer search endpoint.

    Searches should be tenant-scoped by the service/repository.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_customer_service()

    result = await _call_service(
        service,
        "search",
        company_id=company_id,
        query=q.strip(),
        page=page,
        page_size=page_size,
        user=current_user,
    )

    if result is None:
        return CustomerListResponse(
            items=[],
            total=0,
            page=page,
            page_size=page_size,
            pages=0,
        )

    return CustomerListResponse.model_validate(
        result
    )


# ============================================================================
# CURRENT COMPANY CUSTOMER
# ============================================================================


@router.get(
    "/{customer_id}",
    response_model=CustomerResponse,
    summary="Get customer",
)
async def get_customer(
    customer_id: str,
    current_user: Any = Depends(get_current_user()),
) -> CustomerResponse:
    """
    Retrieve a customer.

    CustomerService must enforce company_id filtering at the
    repository level as a second layer of tenant protection.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_customer_service()

    result = await _call_service(
        service,
        "get",
        customer_id=customer_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    return CustomerResponse.model_validate(
        result
    )


# ============================================================================
# UPDATE
# ============================================================================


@router.patch(
    "/{customer_id}",
    response_model=CustomerResponse,
    summary="Update customer",
)
async def update_customer(
    customer_id: str,
    request: CustomerUpdateRequest,
    current_user: Any = Depends(get_current_user()),
) -> CustomerResponse:
    """
    Update customer information.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_customer_manager(
        current_user
    )

    service = get_customer_service()

    result = await _call_service(
        service,
        "update",
        customer_id=customer_id,
        company_id=company_id,
        updates=request.model_dump(
            exclude_unset=True
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    return CustomerResponse.model_validate(
        result
    )


# ============================================================================
# FINANCIAL SUMMARY
# ============================================================================


@router.get(
    "/{customer_id}/financial-summary",
    response_model=CustomerFinancialSummary,
    summary="Get customer financial summary",
)
async def get_customer_financial_summary(
    customer_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    current_user: Any = Depends(get_current_user()),
) -> CustomerFinancialSummary:
    """
    Get customer-level financial metrics.

    Actual financial calculations belong to CustomerService /
    FinancialService, not the API layer.
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
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_customer_service()

    result = await _call_service(
        service,
        "financial_summary",
        customer_id=customer_id,
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer financial data not found.",
        )

    return CustomerFinancialSummary.model_validate(
        result
    )


# ============================================================================
# TRANSACTION SUMMARY
# ============================================================================


@router.get(
    "/{customer_id}/transactions/summary",
    response_model=CustomerTransactionSummary,
    summary="Get customer transaction summary",
)
async def get_customer_transaction_summary(
    customer_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    current_user: Any = Depends(get_current_user()),
) -> CustomerTransactionSummary:
    """
    Aggregate transaction information for a customer.
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
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_customer_service()

    result = await _call_service(
        service,
        "transaction_summary",
        customer_id=customer_id,
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer transaction data not found.",
        )

    return CustomerTransactionSummary.model_validate(
        result
    )


# ============================================================================
# RISK
# ============================================================================


@router.post(
    "/{customer_id}/risk",
    response_model=CustomerRiskResponse,
    summary="Assess customer risk",
)
async def assess_customer_risk(
    customer_id: str,
    request: CustomerRiskRequest,
    current_user: Any = Depends(get_current_user()),
) -> CustomerRiskResponse:
    """
    Assess customer financial/risk exposure.

    The API only orchestrates the request. Risk calculations are
    delegated to the risk/customer service.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_customer_service()

    result = await _call_service(
        service,
        "assess_risk",
        customer_id=customer_id,
        company_id=company_id,
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer risk data not found.",
        )

    return CustomerRiskResponse.model_validate(
        result
    )


# ============================================================================
# STATUS
# ============================================================================


@router.patch(
    "/{customer_id}/status",
    response_model=CustomerStatusResponse,
    summary="Change customer status",
)
async def change_customer_status(
    customer_id: str,
    request: CustomerStatusRequest,
    current_user: Any = Depends(get_current_user()),
) -> CustomerStatusResponse:
    """
    Change customer lifecycle status.

    The service should generate an audit event for this operation.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_customer_manager(
        current_user
    )

    service = get_customer_service()

    result = await _call_service(
        service,
        "change_status",
        customer_id=customer_id,
        company_id=company_id,
        status=request.status,
        reason=request.reason,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    return CustomerStatusResponse.model_validate(
        result
    )


# ============================================================================
# ARCHIVE
# ============================================================================


@router.delete(
    "/{customer_id}",
    response_model=CustomerDeleteResponse,
    summary="Archive customer",
)
async def archive_customer(
    customer_id: str,
    reason: str | None = Query(
        default=None,
        max_length=2000,
    ),
    current_user: Any = Depends(get_current_user()),
) -> CustomerDeleteResponse:
    """
    Archive a customer.

    Hard deletion is intentionally avoided because historical
    transactions, invoices, reports, recommendations, fraud
    investigations, and audit records may reference the customer.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_customer_manager(
        current_user
    )

    service = get_customer_service()

    result = await _call_service(
        service,
        "archive",
        customer_id=customer_id,
        company_id=company_id,
        reason=reason,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    return CustomerDeleteResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=CustomerHealthResponse,
    summary="Customer service health",
)
async def customer_health() -> CustomerHealthResponse:
    """
    Customer service health check.
    """

    started = time.perf_counter()

    try:
        service = get_customer_service()

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

        return CustomerHealthResponse(
            status="healthy",
            service="customer",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Customer service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Customer service is unavailable.",
        )