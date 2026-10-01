"""
backend/app/api/users.py

User Management API for FinCo AI.

Responsibilities
----------------
- Authentication / authorization
- User profile retrieval
- User profile updates
- Company-scoped user listing
- User activation/deactivation
- Role information
- User deletion/deactivation workflows
- Tenant isolation
- Health check

Architecture
------------

    Client
       |
       v
    FastAPI Users API
       |
       +---- Authentication
       +---- Permissions
       +---- Tenant Isolation
       |
       v
    UserService
       |
       +---- UserRepository
       +---- CompanyRepository
       +---- Audit Service
       |
       v
    PostgreSQL

Authentication/token issuance belongs to:
    backend/app/api/auth.py

Authorization rules belong to:
    backend/app/core/permissions.py
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
    EmailStr,
    Field,
    field_validator,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/users",
    tags=["Users"],
)


# ============================================================================
# TYPES
# ============================================================================

UserRole = Literal[
    "owner",
    "admin",
    "finance_manager",
    "analyst",
    "auditor",
    "viewer",
]

UserStatus = Literal[
    "active",
    "inactive",
    "suspended",
    "pending",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class UserUpdateRequest(BaseModel):
    """
    Update authenticated user's profile.
    """

    model_config = ConfigDict(extra="forbid")

    first_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    last_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    display_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    timezone: str | None = Field(
        default=None,
        max_length=100,
    )

    locale: str | None = Field(
        default=None,
        max_length=20,
    )

    preferences: dict[str, Any] | None = None


class AdminUserUpdateRequest(BaseModel):
    """
    Administrative user update.

    Role/status changes should be performed through the
    permission-aware service layer.
    """

    model_config = ConfigDict(extra="forbid")

    first_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    last_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )

    display_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    role: UserRole | None = None

    status: UserStatus | None = None

    timezone: str | None = Field(
        default=None,
        max_length=100,
    )

    locale: str | None = Field(
        default=None,
        max_length=20,
    )

    preferences: dict[str, Any] | None = None


class UserRoleUpdateRequest(BaseModel):
    """
    Change a user's role.
    """

    model_config = ConfigDict(extra="forbid")

    role: UserRole


class UserStatusUpdateRequest(BaseModel):
    """
    Change user account status.
    """

    model_config = ConfigDict(extra="forbid")

    status: UserStatus


class UserCompanyAssignmentRequest(BaseModel):
    """
    Assign a user to a company.

    This operation should normally be restricted to
    platform/company administrators.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class UserResponse(BaseModel):
    """
    Public-safe user representation.
    """

    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
    )

    id: str

    email: EmailStr

    first_name: str | None = None

    last_name: str | None = None

    display_name: str | None = None

    company_id: str | None = None

    role: UserRole

    status: UserStatus

    timezone: str | None = None

    locale: str | None = None

    preferences: dict[str, Any] = Field(
        default_factory=dict,
    )

    is_verified: bool = False

    last_login_at: datetime | None = None

    created_at: datetime

    updated_at: datetime


class UserListResponse(BaseModel):
    """
    Paginated company user list.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str

    items: list[UserResponse]

    total: int

    limit: int

    offset: int

    has_more: bool


class UserRoleResponse(BaseModel):
    """
    User role and authorization information.
    """

    user_id: str

    company_id: str | None

    role: UserRole

    permissions: list[str]

    is_admin: bool

    retrieved_at: datetime


class UserStatusResponse(BaseModel):
    """
    User status change result.
    """

    user_id: str

    company_id: str | None

    status: UserStatus

    message: str

    updated_at: datetime


class UserCompanyResponse(BaseModel):
    """
    Company assignment result.
    """

    user_id: str

    company_id: str

    role: UserRole

    message: str

    updated_at: datetime


class UserHealthResponse(BaseModel):
    """
    User subsystem health.
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


def get_user_service() -> Any:
    """
    Lazily construct UserService.

    Expected implementation:

        backend.app.users.user_service.UserService

    Repository fallback is provided for incremental development.
    """

    try:
        from backend.app.users.user_service import (
            UserService,
        )

        return UserService()

    except ImportError:
        pass

    try:
        from backend.app.database.repositories.user_repository import (
            UserRepository,
        )

        return UserRepository()

    except ImportError as exc:
        logger.exception(
            "User service/repository unavailable."
        )

        raise RuntimeError(
            "User service is not configured."
        ) from exc


# ============================================================================
# USER HELPERS
# ============================================================================


def _get_user_id(
    current_user: Any,
) -> str | None:
    """
    Extract authenticated user ID.
    """

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


def _get_user_company_id(
    current_user: Any,
) -> str | None:
    """
    Extract authenticated user's company ID.
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


def _get_user_role(
    current_user: Any,
) -> str | None:
    """
    Extract authenticated user's role.
    """

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
    """
    Determine administrative access.
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


def _is_owner_or_admin(
    current_user: Any,
) -> bool:
    """
    Determine whether user has owner/admin privileges.
    """

    if _is_admin(current_user):
        return True

    role = _get_user_role(current_user)

    return role in {
        "owner",
        "admin",
    }


# ============================================================================
# SECURITY
# ============================================================================


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce company-level tenant isolation.
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
            "Unauthorized user-management access: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to users "
                "outside your company."
            ),
        )


def require_admin(
    current_user: Any,
) -> None:
    """
    Require administrative privileges.
    """

    if not _is_owner_or_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Administrative privileges are required."
            ),
        )


def require_self_or_admin(
    user_id: str,
    current_user: Any,
) -> None:
    """
    Allow a user to access their own profile or allow admins
    to manage another user.
    """

    current_user_id = _get_user_id(
        current_user
    )

    if (
        current_user_id != user_id
        and not _is_admin(current_user)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You can only access your own profile."
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
    Invoke service method.

    Supports synchronous and asynchronous services.
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
                f"User service does not implement "
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
            "Invalid user-service request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "User service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User operation failed.",
        ) from exc


# ============================================================================
# CURRENT USER
# ============================================================================


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
)
async def get_current_user_profile(
    current_user: Any = Depends(get_current_user()),
) -> UserResponse:
    """
    Return the authenticated user's profile.
    """

    user_id = _get_user_id(
        current_user
    )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user ID is unavailable.",
        )

    service = get_user_service()

    result = await _call_service(
        service,
        "get",
        user_id=user_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found.",
        )

    return UserResponse.model_validate(
        result
    )


# ============================================================================
# UPDATE CURRENT USER
# ============================================================================


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update current user",
)
async def update_current_user_profile(
    request: UserUpdateRequest,
    current_user: Any = Depends(get_current_user()),
) -> UserResponse:
    """
    Update the authenticated user's profile.

    Sensitive fields such as:
        - role
        - company_id
        - account status
        - email
        - password

    are intentionally excluded from this endpoint.
    """

    user_id = _get_user_id(
        current_user
    )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user ID is unavailable.",
        )

    service = get_user_service()

    result = await _call_service(
        service,
        "update",
        user_id=user_id,
        updates=request.model_dump(
            exclude_unset=True
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User profile not found.",
        )

    return UserResponse.model_validate(
        result
    )


# ============================================================================
# GET USER
# ============================================================================


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user",
)
async def get_user(
    user_id: str,
    current_user: Any = Depends(get_current_user()),
) -> UserResponse:
    """
    Retrieve a user.

    A user can retrieve their own profile.
    Administrators can retrieve users within their company.
    """

    require_self_or_admin(
        user_id,
        current_user,
    )

    service = get_user_service()

    result = await _call_service(
        service,
        "get",
        user_id=user_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    company_id = (
        result.get("company_id")
        if isinstance(result, dict)
        else getattr(
            result,
            "company_id",
            None,
        )
    )

    if company_id:
        validate_company_access(
            company_id,
            current_user,
        )

    return UserResponse.model_validate(
        result
    )


# ============================================================================
# LIST COMPANY USERS
# ============================================================================


@router.get(
    "/company/{company_id}",
    response_model=UserListResponse,
    summary="List company users",
)
async def list_company_users(
    company_id: str,
    role: UserRole | None = Query(
        default=None
    ),
    user_status: UserStatus | None = Query(
        default=None,
        alias="status",
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    current_user: Any = Depends(get_current_user()),
) -> UserListResponse:
    """
    List users belonging to a company.

    Company user directories are restricted to administrators
    to avoid exposing employee/account information broadly.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    require_admin(
        current_user
    )

    service = get_user_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        role=role,
        status=user_status,
        limit=limit,
        offset=offset,
        user=current_user,
    )

    if result is None:
        return UserListResponse(
            company_id=company_id,
            items=[],
            total=0,
            limit=limit,
            offset=offset,
            has_more=False,
        )

    if isinstance(
        result,
        UserListResponse,
    ):
        return result

    return UserListResponse.model_validate(
        result
    )


# ============================================================================
# ADMIN UPDATE USER
# ============================================================================


@router.patch(
    "/{user_id}/admin",
    response_model=UserResponse,
    summary="Administratively update user",
)
async def admin_update_user(
    user_id: str,
    request: AdminUserUpdateRequest,
    current_user: Any = Depends(get_current_user()),
) -> UserResponse:
    """
    Administratively update another user's profile.
    """

    require_admin(
        current_user
    )

    service = get_user_service()

    target_user = await _call_service(
        service,
        "get",
        user_id=user_id,
        user=current_user,
    )

    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    target_company_id = (
        target_user.get("company_id")
        if isinstance(target_user, dict)
        else getattr(
            target_user,
            "company_id",
            None,
        )
    )

    if target_company_id:
        validate_company_access(
            target_company_id,
            current_user,
        )

    result = await _call_service(
        service,
        "admin_update",
        user_id=user_id,
        updates=request.model_dump(
            exclude_unset=True
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    return UserResponse.model_validate(
        result
    )


# ============================================================================
# ROLE
# ============================================================================


@router.get(
    "/{user_id}/role",
    response_model=UserRoleResponse,
    summary="Get user role and permissions",
)
async def get_user_role(
    user_id: str,
    current_user: Any = Depends(get_current_user()),
) -> UserRoleResponse:
    """
    Return role and effective permissions.
    """

    require_self_or_admin(
        user_id,
        current_user,
    )

    service = get_user_service()

    result = await _call_service(
        service,
        "get_role",
        user_id=user_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User role not found.",
        )

    if isinstance(
        result,
        UserRoleResponse,
    ):
        return result

    return UserRoleResponse.model_validate(
        result
    )


# ============================================================================
# CHANGE ROLE
# ============================================================================


@router.patch(
    "/{user_id}/role",
    response_model=UserRoleResponse,
    summary="Change user role",
)
async def change_user_role(
    user_id: str,
    request: UserRoleUpdateRequest,
    current_user: Any = Depends(get_current_user()),
) -> UserRoleResponse:
    """
    Change a user's role.

    This endpoint is deliberately admin-only.
    """

    require_admin(
        current_user
    )

    service = get_user_service()

    target_user = await _call_service(
        service,
        "get",
        user_id=user_id,
        user=current_user,
    )

    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    target_company_id = (
        target_user.get("company_id")
        if isinstance(target_user, dict)
        else getattr(
            target_user,
            "company_id",
            None,
        )
    )

    if target_company_id:
        validate_company_access(
            target_company_id,
            current_user,
        )

    result = await _call_service(
        service,
        "change_role",
        user_id=user_id,
        role=request.role,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    if isinstance(
        result,
        UserRoleResponse,
    ):
        return result

    return UserRoleResponse.model_validate(
        result
    )


# ============================================================================
# CHANGE STATUS
# ============================================================================


@router.patch(
    "/{user_id}/status",
    response_model=UserStatusResponse,
    summary="Change user status",
)
async def change_user_status(
    user_id: str,
    request: UserStatusUpdateRequest,
    current_user: Any = Depends(get_current_user()),
) -> UserStatusResponse:
    """
    Activate, deactivate, suspend, or mark a user pending.

    Account lifecycle changes should be audited by the service layer.
    """

    require_admin(
        current_user
    )

    service = get_user_service()

    target_user = await _call_service(
        service,
        "get",
        user_id=user_id,
        user=current_user,
    )

    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    target_company_id = (
        target_user.get("company_id")
        if isinstance(target_user, dict)
        else getattr(
            target_user,
            "company_id",
            None,
        )
    )

    if target_company_id:
        validate_company_access(
            target_company_id,
            current_user,
        )

    result = await _call_service(
        service,
        "change_status",
        user_id=user_id,
        status=request.status,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    if isinstance(
        result,
        UserStatusResponse,
    ):
        return result

    return UserStatusResponse.model_validate(
        result
    )


# ============================================================================
# COMPANY ASSIGNMENT
# ============================================================================


@router.patch(
    "/{user_id}/company",
    response_model=UserCompanyResponse,
    summary="Assign user to company",
)
async def assign_user_company(
    user_id: str,
    request: UserCompanyAssignmentRequest,
    current_user: Any = Depends(get_current_user()),
) -> UserCompanyResponse:
    """
    Assign a user to a company.

    This operation should normally be restricted to platform
    administrators or company owners with the appropriate
    permission.
    """

    require_admin(
        current_user
    )

    service = get_user_service()

    result = await _call_service(
        service,
        "assign_company",
        user_id=user_id,
        company_id=request.company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    if isinstance(
        result,
        UserCompanyResponse,
    ):
        return result

    return UserCompanyResponse.model_validate(
        result
    )


# ============================================================================
# DELETE / DEACTIVATE USER
# ============================================================================


@router.delete(
    "/{user_id}",
    response_model=UserStatusResponse,
    summary="Deactivate user",
)
async def deactivate_user(
    user_id: str,
    current_user: Any = Depends(get_current_user()),
) -> UserStatusResponse:
    """
    Deactivate a user account.

    Financial/audit systems should normally avoid hard-deleting
    user records because historical actions may reference the user.
    """

    require_admin(
        current_user
    )

    service = get_user_service()

    target_user = await _call_service(
        service,
        "get",
        user_id=user_id,
        user=current_user,
    )

    if target_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    target_company_id = (
        target_user.get("company_id")
        if isinstance(target_user, dict)
        else getattr(
            target_user,
            "company_id",
            None,
        )
    )

    if target_company_id:
        validate_company_access(
            target_company_id,
            current_user,
        )

    result = await _call_service(
        service,
        "deactivate",
        user_id=user_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    if isinstance(
        result,
        UserStatusResponse,
    ):
        return result

    return UserStatusResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=UserHealthResponse,
    summary="User service health",
)
async def user_health() -> UserHealthResponse:
    """
    Health check for user-management infrastructure.
    """

    started = time.perf_counter()

    try:
        service = get_user_service()

        health_method = getattr(
            service,
            "health",
            None,
        )

        details: dict[str, Any] = {}

        if health_method is not None:
            result = health_method()

            if inspect.isawaitable(result):
                result = await result

            if isinstance(result, dict):
                details = result

        details["latency_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return UserHealthResponse(
            status="healthy",
            service="users",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "User service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User service is unavailable.",
        )