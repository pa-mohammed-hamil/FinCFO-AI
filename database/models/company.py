"""
backend/app/api/company.py

Company / Tenant Management API for FinCo AI.

Responsibilities
----------------
- Company profile management
- Company settings
- Financial configuration
- Company user summary
- Company lifecycle
- Company metadata
- Tenant isolation
- Administrative authorization
- Audit-friendly operations

Architecture
------------

    Client
       |
       v
    Company API
       |
       +---- Authentication
       +---- Authorization
       +---- Tenant Isolation
       |
       v
    CompanyService
       |
       +---- CompanyRepository
       +---- UserRepository
       +---- AuditService
       |
       v
    PostgreSQL
"""

from __future__ import annotations

import inspect
import logging
import time
from datetime import datetime, timezone
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
    prefix="/companies",
    tags=["Companies"],
)


# ============================================================================
# TYPES
# ============================================================================

CompanyStatus = Literal[
    "active",
    "inactive",
    "suspended",
    "pending",
]

CompanyPlan = Literal[
    "free",
    "starter",
    "professional",
    "enterprise",
]

FiscalYearStartMonth = Literal[
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class CompanyCreateRequest(BaseModel):
    """
    Create a company/tenant.

    Normally used by platform administrators or onboarding flow.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=2,
        max_length=200,
    )

    legal_name: str | None = Field(
        default=None,
        max_length=300,
    )

    industry: str | None = Field(
        default=None,
        max_length=150,
    )

    country: str = Field(
        default="IN",
        min_length=2,
        max_length=2,
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
    )

    timezone: str = Field(
        default="Asia/Kolkata",
        max_length=100,
    )

    fiscal_year_start_month: FiscalYearStartMonth = 4

    plan: CompanyPlan = "starter"

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("name", "legal_name")
    @classmethod
    def clean_names(
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
        value: str,
    ) -> str:
        return value.upper()

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str,
    ) -> str:
        return value.upper()


class CompanyUpdateRequest(BaseModel):
    """
    Update company profile.

    Tenant identity and ownership are intentionally excluded.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        min_length=2,
        max_length=200,
    )

    legal_name: str | None = Field(
        default=None,
        max_length=300,
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

    timezone: str | None = Field(
        default=None,
        max_length=100,
    )

    fiscal_year_start_month: FiscalYearStartMonth | None = (
        None
    )

    metadata: dict[str, Any] | None = None

    @field_validator("name", "legal_name")
    @classmethod
    def clean_names(
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


class CompanySettingsRequest(BaseModel):
    """
    Update company-level application settings.
    """

    model_config = ConfigDict(extra="forbid")

    default_currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    timezone: str | None = Field(
        default=None,
        max_length=100,
    )

    fiscal_year_start_month: FiscalYearStartMonth | None = (
        None
    )

    default_forecast_horizon_months: int | None = Field(
        default=None,
        ge=1,
        le=120,
    )

    default_alert_threshold: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    enable_fraud_detection: bool | None = None

    enable_ai_recommendations: bool | None = None

    enable_what_if: bool | None = None

    metadata: dict[str, Any] | None = None

    @field_validator("default_currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


class CompanyStatusRequest(BaseModel):
    """
    Change company lifecycle status.
    """

    model_config = ConfigDict(extra="forbid")

    status: CompanyStatus

    reason: str | None = Field(
        default=None,
        max_length=2000,
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class CompanyResponse(BaseModel):
    """
    Company/tenant representation.
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
    )

    id: str

    name: str

    legal_name: str | None = None

    industry: str | None = None

    country: str

    currency: str

    timezone: str

    fiscal_year_start_month: int

    plan: CompanyPlan

    status: CompanyStatus

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: datetime

    updated_at: datetime


class CompanySettingsResponse(BaseModel):
    """
    Company configuration.
    """

    company_id: str

    default_currency: str

    timezone: str

    fiscal_year_start_month: int

    default_forecast_horizon_months: int

    default_alert_threshold: float

    enable_fraud_detection: bool

    enable_ai_recommendations: bool

    enable_what_if: bool

    updated_at: datetime


class CompanyUserSummary(BaseModel):
    """
    User-count summary for a company.
    """

    company_id: str

    total_users: int

    active_users: int

    pending_users: int

    suspended_users: int

    admins: int

    finance_managers: int

    analysts: int

    auditors: int

    viewers: int


class CompanyStatusResponse(BaseModel):
    """
    Company status update result.
    """

    company_id: str

    status: CompanyStatus

    message: str

    updated_at: datetime


class CompanyDeleteResponse(BaseModel):
    """
    Company archive response.

    Financial tenants should normally be archived rather than
    physically deleted.
    """

    company_id: str

    status: Literal["archived"]

    message: str

    updated_at: datetime


class CompanyHealthResponse(BaseModel):
    """
    Company subsystem health.
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
# SERVICE
# ============================================================================


def get_company_service() -> Any:
    """
    Lazily load CompanyService.

    Expected implementation:

        backend.app.company.company_service.CompanyService

    Fallback:

        backend.app.database.repositories.company_repository.CompanyRepository
    """

    try:
        from backend.app.company.company_service import (
            CompanyService,
        )

        return CompanyService()

    except ImportError:
        pass

    try:
        from backend.app.database.repositories.company_repository import (
            CompanyRepository,
        )

        return CompanyRepository()

    except ImportError as exc:
        logger.exception(
            "Company service/repository unavailable."
        )

        raise RuntimeError(
            "Company service is not configured."
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


def _is_company_admin(
    current_user: Any,
) -> bool:
    """
    Users allowed to administer their company.
    """

    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
    }


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Mandatory tenant-isolation check.
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
            "Unauthorized company access: "
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


def require_company_admin(
    current_user: Any,
) -> None:
    """
    Require company/platform administrative privileges.
    """

    if not _is_company_admin(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Company administrative privileges are required."
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
    Invoke sync or async service methods.
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
                f"Company service does not implement "
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
            "Invalid company request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid company request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Company service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Company operation failed.",
        ) from exc


# ============================================================================
# CREATE COMPANY
# ============================================================================


@router.post(
    "",
    response_model=CompanyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create company",
)
async def create_company(
    request: CompanyCreateRequest,
    current_user: Any = Depends(get_current_user()),
) -> CompanyResponse:
    """
    Create a company.

    In a multi-tenant production environment this should normally
    be restricted to platform admins or a controlled onboarding
    workflow.
    """

    if not _is_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Only platform administrators can create "
                "companies through this endpoint."
            ),
        )

    service = get_company_service()

    result = await _call_service(
        service,
        "create",
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Company creation returned no result.",
        )

    return CompanyResponse.model_validate(
        result
    )


# ============================================================================
# GET CURRENT COMPANY
# ============================================================================


@router.get(
    "/me",
    response_model=CompanyResponse,
    summary="Get current company",
)
async def get_current_company(
    current_user: Any = Depends(get_current_user()),
) -> CompanyResponse:
    """
    Retrieve the authenticated user's company.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not associated with a company.",
        )

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_company_service()

    result = await _call_service(
        service,
        "get",
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    return CompanyResponse.model_validate(
        result
    )


# ============================================================================
# GET COMPANY
# ============================================================================


@router.get(
    "/{company_id}",
    response_model=CompanyResponse,
    summary="Get company",
)
async def get_company(
    company_id: str,
    current_user: Any = Depends(get_current_user()),
) -> CompanyResponse:
    """
    Retrieve a company accessible to the authenticated user.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_company_service()

    result = await _call_service(
        service,
        "get",
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    return CompanyResponse.model_validate(
        result
    )


# ============================================================================
# UPDATE COMPANY
# ============================================================================


@router.patch(
    "/{company_id}",
    response_model=CompanyResponse,
    summary="Update company",
)
async def update_company(
    company_id: str,
    request: CompanyUpdateRequest,
    current_user: Any = Depends(get_current_user()),
) -> CompanyResponse:
    """
    Update company profile.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_company_admin(
        current_user
    )

    service = get_company_service()

    result = await _call_service(
        service,
        "update",
        company_id=company_id,
        updates=request.model_dump(
            exclude_unset=True
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    return CompanyResponse.model_validate(
        result
    )


# ============================================================================
# SETTINGS
# ============================================================================


@router.get(
    "/{company_id}/settings",
    response_model=CompanySettingsResponse,
    summary="Get company settings",
)
async def get_company_settings(
    company_id: str,
    current_user: Any = Depends(get_current_user()),
) -> CompanySettingsResponse:
    """
    Retrieve company-level FinCo AI settings.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_company_service()

    result = await _call_service(
        service,
        "get_settings",
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company settings not found.",
        )

    return CompanySettingsResponse.model_validate(
        result
    )


@router.patch(
    "/{company_id}/settings",
    response_model=CompanySettingsResponse,
    summary="Update company settings",
)
async def update_company_settings(
    company_id: str,
    request: CompanySettingsRequest,
    current_user: Any = Depends(get_current_user()),
) -> CompanySettingsResponse:
    """
    Update company-level FinCo AI settings.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_company_admin(
        current_user
    )

    service = get_company_service()

    result = await _call_service(
        service,
        "update_settings",
        company_id=company_id,
        settings=request.model_dump(
            exclude_unset=True
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company settings not found.",
        )

    return CompanySettingsResponse.model_validate(
        result
    )


# ============================================================================
# USER SUMMARY
# ============================================================================


@router.get(
    "/{company_id}/users/summary",
    response_model=CompanyUserSummary,
    summary="Get company user summary",
)
async def company_user_summary(
    company_id: str,
    current_user: Any = Depends(get_current_user()),
) -> CompanyUserSummary:
    """
    Get aggregate user information without exposing individual
    user records.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_company_admin(
        current_user
    )

    service = get_company_service()

    result = await _call_service(
        service,
        "user_summary",
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    return CompanyUserSummary.model_validate(
        result
    )


# ============================================================================
# STATUS
# ============================================================================


@router.patch(
    "/{company_id}/status",
    response_model=CompanyStatusResponse,
    summary="Change company status",
)
async def change_company_status(
    company_id: str,
    request: CompanyStatusRequest,
    current_user: Any = Depends(get_current_user()),
) -> CompanyStatusResponse:
    """
    Change company lifecycle status.

    This operation should always generate an audit event.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_company_admin(
        current_user
    )

    service = get_company_service()

    result = await _call_service(
        service,
        "change_status",
        company_id=company_id,
        status=request.status,
        reason=request.reason,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    return CompanyStatusResponse.model_validate(
        result
    )


# ============================================================================
# ARCHIVE
# ============================================================================


@router.delete(
    "/{company_id}",
    response_model=CompanyDeleteResponse,
    summary="Archive company",
)
async def archive_company(
    company_id: str,
    reason: str | None = Query(
        default=None,
        max_length=2000,
    ),
    current_user: Any = Depends(get_current_user()),
) -> CompanyDeleteResponse:
    """
    Archive a company.

    Physical deletion should not be exposed through the normal
    application API because financial records, audit trails,
    reports, and historical transactions may depend on the tenant.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_company_admin(
        current_user
    )

    service = get_company_service()

    result = await _call_service(
        service,
        "archive",
        company_id=company_id,
        reason=reason,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    return CompanyDeleteResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=CompanyHealthResponse,
    summary="Company service health",
)
async def company_health() -> CompanyHealthResponse:
    """
    Health check for company/tenant infrastructure.
    """

    started = time.perf_counter()

    try:
        service = get_company_service()

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

        return CompanyHealthResponse(
            status="healthy",
            service="company",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Company service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Company service is unavailable.",
        )