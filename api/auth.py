"""
FinCo AI - Authentication API
Path: backend/app/api/auth.py

Responsibilities
----------------
- User registration
- User login
- JWT access-token generation
- JWT refresh-token rotation
- Current-user retrieval
- Logout / token revocation hook
- Authentication-related API contracts

Architecture
------------
Frontend
    ↓
/api/v1/auth/*
    ↓
auth.py
    ↓
Authentication Service
    ↓
User Repository
    ↓
Database

JWT tokens are used for stateless authentication.
Financial calculations, RAG, ML, and agent logic must never live here.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

try:
    from jose import JWTError, jwt
except ImportError:  # pragma: no cover
    JWTError = Exception
    jwt = None

try:
    from passlib.context import CryptContext
except ImportError:  # pragma: no cover
    CryptContext = None


# ============================================================================
# CONFIGURATION
# ============================================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False,
)


class AuthConfig:
    """
    Central authentication configuration.

    In production these values should come from environment variables
    through app.config.Settings.
    """

    SECRET_KEY = "CHANGE_ME_IN_ENVIRONMENT"
    ALGORITHM = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7

    PASSWORD_MIN_LENGTH = 8

    ISSUER = "finco-ai"
    AUDIENCE = "finco-ai-users"


# Password hashing context.
pwd_context = (
    CryptContext(
        schemes=["bcrypt"],
        deprecated="auto",
    )
    if CryptContext is not None
    else None
)


# ============================================================================
# ENUMS
# ============================================================================


class UserRole(str, Enum):
    ADMIN = "admin"
    EXECUTIVE = "executive"
    FINANCE_MANAGER = "finance_manager"
    FINANCIAL_ANALYST = "financial_analyst"
    AUDITOR = "auditor"
    VIEWER = "viewer"


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


# ============================================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================================


class AuthSchemaBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class RegisterRequest(AuthSchemaBase):
    email: EmailStr
    password: str = Field(
        min_length=AuthConfig.PASSWORD_MIN_LENGTH,
        max_length=128,
    )
    full_name: str = Field(
        min_length=2,
        max_length=150,
    )
    company_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )
    role: UserRole = UserRole.VIEWER

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> EmailStr:
        return EmailStr(str(value).lower().strip())

    @field_validator("full_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = " ".join(value.split())

        if len(value) < 2:
            raise ValueError("Full name must contain at least 2 characters.")

        return value


class LoginRequest(AuthSchemaBase):
    email: EmailStr
    password: str = Field(
        min_length=1,
        max_length=128,
    )

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> EmailStr:
        return EmailStr(str(value).lower().strip())


class RefreshTokenRequest(AuthSchemaBase):
    refresh_token: str = Field(
        min_length=20,
        max_length=4096,
    )


class LogoutRequest(AuthSchemaBase):
    refresh_token: Optional[str] = Field(
        default=None,
        min_length=20,
        max_length=4096,
    )


class UserResponse(AuthSchemaBase):
    id: str
    email: EmailStr
    full_name: str
    role: UserRole
    company_id: Optional[str] = None
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime


class TokenResponse(AuthSchemaBase):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class AccessTokenResponse(AuthSchemaBase):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class MessageResponse(AuthSchemaBase):
    message: str


# ============================================================================
# INTERNAL USER MODEL
# ============================================================================


class AuthUser:
    """
    Lightweight internal user representation.

    Replace this with your SQLAlchemy User model when connecting
    database repositories.
    """

    def __init__(
        self,
        user_id: str,
        email: str,
        full_name: str,
        password_hash: str,
        role: UserRole = UserRole.VIEWER,
        company_id: Optional[str] = None,
        is_active: bool = True,
        is_verified: bool = False,
        created_at: Optional[datetime] = None,
    ) -> None:
        self.id = user_id
        self.email = email
        self.full_name = full_name
        self.password_hash = password_hash
        self.role = role
        self.company_id = company_id
        self.is_active = is_active
        self.is_verified = is_verified
        self.created_at = created_at or datetime.now(timezone.utc)


# ============================================================================
# TEMPORARY IN-MEMORY REPOSITORY
# ============================================================================

"""
IMPORTANT
---------
This repository exists so the API can be demonstrated without requiring
the database layer to be completed.

For production:
    repositories/user_repository.py
should replace this implementation.
"""

_USERS: dict[str, AuthUser] = {}
_REVOKED_TOKENS: set[str] = set()


# ============================================================================
# SECURITY HELPERS
# ============================================================================


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""

    if pwd_context is None:
        raise RuntimeError(
            "passlib is required for password hashing. "
            "Install passlib[bcrypt]."
        )

    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    password_hash: str,
) -> bool:
    """Verify a plaintext password against its bcrypt hash."""

    if pwd_context is None:
        raise RuntimeError(
            "passlib is required for password verification."
        )

    try:
        return pwd_context.verify(
            plain_password,
            password_hash,
        )
    except Exception:
        return False


def generate_user_id() -> str:
    """Generate a cryptographically random user identifier."""

    return secrets.token_urlsafe(16)


def generate_jti() -> str:
    """Generate a unique JWT ID."""

    return secrets.token_urlsafe(24)


def hash_token(token: str) -> str:
    """
    Hash a token before storing it in a revocation store.

    Raw JWTs should not normally be persisted.
    """

    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


# ============================================================================
# JWT HELPERS
# ============================================================================


def _require_jwt() -> None:
    if jwt is None:
        raise RuntimeError(
            "python-jose is required for JWT authentication. "
            "Install python-jose[cryptography]."
        )


def create_token(
    *,
    user: AuthUser,
    token_type: TokenType,
    expires_delta: timedelta,
) -> str:
    """
    Create a signed JWT.

    Claims:
    - sub       user ID
    - email     user email
    - role      authorization role
    - company_id tenant/company scope
    - type      access/refresh
    - jti       unique token ID
    - iss       FinCo AI issuer
    - aud       intended audience
    - iat       issued-at timestamp
    - exp       expiration timestamp
    """

    _require_jwt()

    now = datetime.now(timezone.utc)
    expires_at = now + expires_delta

    payload: dict[str, Any] = {
        "sub": user.id,
        "email": user.email,
        "role": user.role.value,
        "company_id": user.company_id,
        "type": token_type.value,
        "jti": generate_jti(),
        "iss": AuthConfig.ISSUER,
        "aud": AuthConfig.AUDIENCE,
        "iat": now,
        "exp": expires_at,
    }

    return jwt.encode(
        payload,
        AuthConfig.SECRET_KEY,
        algorithm=AuthConfig.ALGORITHM,
    )


def create_access_token(user: AuthUser) -> str:
    return create_token(
        user=user,
        token_type=TokenType.ACCESS,
        expires_delta=timedelta(
            minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    )


def create_refresh_token(user: AuthUser) -> str:
    return create_token(
        user=user,
        token_type=TokenType.REFRESH,
        expires_delta=timedelta(
            days=AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS
        ),
    )


def decode_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT."""

    _require_jwt()

    try:
        payload = jwt.decode(
            token,
            AuthConfig.SECRET_KEY,
            algorithms=[AuthConfig.ALGORITHM],
            audience=AuthConfig.AUDIENCE,
            issuer=AuthConfig.ISSUER,
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    jti = payload.get("jti")

    if jti and jti in _REVOKED_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has been revoked.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


# ============================================================================
# USER HELPERS
# ============================================================================


def get_user_by_email(email: str) -> Optional[AuthUser]:
    normalized = email.lower().strip()

    for user in _USERS.values():
        if user.email.lower() == normalized:
            return user

    return None


def get_user_by_id(user_id: str) -> Optional[AuthUser]:
    return _USERS.get(user_id)


def user_to_response(user: AuthUser) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        company_id=user.company_id,
        is_active=user.is_active,
        is_verified=user.is_verified,
        created_at=user.created_at,
    )


def authenticate_user(
    email: str,
    password: str,
) -> Optional[AuthUser]:
    user = get_user_by_email(email)

    if user is None:
        return None

    if not user.is_active:
        return None

    if not verify_password(
        password,
        user.password_hash,
    ):
        return None

    return user


# ============================================================================
# DEPENDENCY: CURRENT USER
# ============================================================================


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> AuthUser:
    """
    Resolve the currently authenticated user.

    Other API modules can use:

        user = Depends(get_current_user)
    """

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token)

    if payload.get("type") != TokenType.ACCESS.value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication subject.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    return user


# ============================================================================
# ROLE-BASED ACCESS CONTROL
# ============================================================================


def require_roles(
    *allowed_roles: UserRole,
):
    """
    Dependency factory for role-based authorization.

    Example:

        Depends(
            require_roles(
                UserRole.ADMIN,
                UserRole.EXECUTIVE,
            )
        )
    """

    async def dependency(
        user: AuthUser = Depends(get_current_user),
    ) -> AuthUser:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions.",
            )

        return user

    return dependency


# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    request: RegisterRequest,
) -> TokenResponse:
    """
    Register a FinCo AI user.

    Production flow:
        request
            ↓
        validation
            ↓
        user repository
            ↓
        password hashing
            ↓
        database
            ↓
        JWT generation
    """

    existing_user = get_user_by_email(
        str(request.email)
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Prevent arbitrary privilege escalation during public registration.
    # New accounts should begin with the lowest privilege.
    role = UserRole.VIEWER

    user = AuthUser(
        user_id=generate_user_id(),
        email=str(request.email).lower(),
        full_name=request.full_name,
        password_hash=hash_password(request.password),
        role=role,
        company_id=request.company_id,
        is_active=True,
        is_verified=False,
    )

    _USERS[user.id] = user

    access_token = create_access_token(user)
    refresh_token = create_refresh_token(user)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_to_response(user),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user",
)
async def login(
    request: LoginRequest,
) -> TokenResponse:
    """
    Authenticate with email/password.

    Deliberately uses a generic error message so the API does not reveal
    whether a particular email address exists.
    """

    user = authenticate_user(
        email=str(request.email),
        password=request.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(user)
    refresh_token = create_refresh_token(user)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_to_response(user),
    )


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
    summary="Refresh an access token",
)
async def refresh_token(
    request: RefreshTokenRequest,
) -> AccessTokenResponse:
    """
    Exchange a valid refresh token for a new access token.

    The refresh token itself is also revoked to support rotation.
    """

    payload = decode_token(
        request.refresh_token
    )

    if payload.get("type") != TokenType.REFRESH.value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_id(user_id)

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is unavailable.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Revoke old refresh token.
    old_jti = payload.get("jti")

    if old_jti:
        _REVOKED_TOKENS.add(old_jti)

    new_access_token = create_access_token(user)

    return AccessTokenResponse(
        access_token=new_access_token,
        token_type="bearer",
        expires_in=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout current session",
)
async def logout(
    request: LogoutRequest,
    user: AuthUser = Depends(get_current_user),
) -> MessageResponse:
    """
    Revoke the supplied refresh token.

    Access tokens remain naturally short-lived.
    """

    if request.refresh_token:
        try:
            payload = decode_token(
                request.refresh_token
            )

            jti = payload.get("jti")

            if jti:
                _REVOKED_TOKENS.add(jti)

        except HTTPException:
            # Logout should remain idempotent.
            pass

    return MessageResponse(
        message="Successfully logged out."
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current authenticated user",
)
async def get_me(
    user: AuthUser = Depends(get_current_user),
) -> UserResponse:
    """Return the authenticated user's profile."""

    return user_to_response(user)


# ============================================================================
# TOKEN VALIDATION
# ============================================================================


@router.get(
    "/verify",
    response_model=UserResponse,
    summary="Verify access token",
)
async def verify_authentication(
    user: AuthUser = Depends(get_current_user),
) -> UserResponse:
    """
    Lightweight endpoint used by the frontend to verify authentication.
    """

    return user_to_response(user)


# ============================================================================
# ADMIN ENDPOINT
# ============================================================================


@router.get(
    "/admin/check",
    response_model=MessageResponse,
    summary="Check administrative permissions",
)
async def admin_check(
    user: AuthUser = Depends(
        require_roles(UserRole.ADMIN)
    ),
) -> MessageResponse:
    """Example RBAC-protected endpoint."""

    return MessageResponse(
        message=f"Administrative access granted to {user.email}."
    )


# ============================================================================
# HEALTH / AUTH METADATA
# ============================================================================


@router.get(
    "/config",
    response_model=dict[str, Any],
    summary="Authentication configuration",
)
async def auth_config() -> dict[str, Any]:
    """
    Return safe authentication metadata.

    Never expose SECRET_KEY or other sensitive configuration.
    """

    return {
        "issuer": AuthConfig.ISSUER,
        "algorithm": AuthConfig.ALGORITHM,
        "access_token_expire_minutes": (
            AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES
        ),
        "refresh_token_expire_days": (
            AuthConfig.REFRESH_TOKEN_EXPIRE_DAYS
        ),
        "password_min_length": (
            AuthConfig.PASSWORD_MIN_LENGTH
        ),
        "supported_roles": [
            role.value for role in UserRole
        ],
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "router",
    "UserRole",
    "TokenType",
    "RegisterRequest",
    "LoginRequest",
    "RefreshTokenRequest",
    "LogoutRequest",
    "UserResponse",
    "TokenResponse",
    "AccessTokenResponse",
    "MessageResponse",
    "AuthUser",
    "get_current_user",
    "require_roles",
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
]


# ============================================================================
# LOCAL DEMO
# ============================================================================

if __name__ == "__main__":
    print("FinCo AI Authentication API")
    print("=" * 40)
    print("Router prefix:", router.prefix)
    print("Endpoints:")
    print("  POST /auth/register")
    print("  POST /auth/login")
    print("  POST /auth/refresh")
    print("  POST /auth/logout")
    print("  GET  /auth/me")
    print("  GET  /auth/verify")
    print("  GET  /auth/admin/check")
    print("  GET  /auth/config")