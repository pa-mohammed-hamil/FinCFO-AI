"""
FinCo AI - Security Module

Responsibilities:
- Password hashing and verification
- JWT access-token creation
- JWT refresh-token creation
- JWT validation
- Token payload validation
- Secure random token generation
- Authentication-related security helpers

Does NOT handle:
- RBAC / permissions        -> permissions.py
- Prompt safety             -> guardrails.py
- Rate limiting             -> rate_limiter.py
- Audit logging             -> audit.py
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from jose import JWTError, ExpiredSignatureError, jwt
from passlib.context import CryptContext

from app.core.exceptions import (
    AuthenticationError,
    CredentialsError,
    InvalidTokenError,
)


logger = logging.getLogger("finco.security")


# ============================================================
# Configuration
# ============================================================

import os


JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY",
    "CHANGE_THIS_SECRET_IN_PRODUCTION",
)

JWT_ALGORITHM = os.getenv(
    "JWT_ALGORITHM",
    "HS256",
)

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv(
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        "30",
    )
)

REFRESH_TOKEN_EXPIRE_DAYS = int(
    os.getenv(
        "REFRESH_TOKEN_EXPIRE_DAYS",
        "7",
    )
)

PASSWORD_MIN_LENGTH = int(
    os.getenv(
        "PASSWORD_MIN_LENGTH",
        "8",
    )
)


# ============================================================
# Password Hashing
# ============================================================

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def hash_password(password: str) -> str:
    """
    Hash a password securely.

    Plain-text passwords must never be stored in the database.
    """

    if not password:
        raise CredentialsError(
            "Password cannot be empty."
        )

    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    Verify a plain-text password against its hash.
    """

    if not plain_password or not hashed_password:
        return False

    try:
        return pwd_context.verify(
            plain_password,
            hashed_password,
        )
    except Exception:
        logger.warning(
            "Password verification failed"
        )
        return False


def validate_password_strength(
    password: str,
) -> None:
    """
    Basic password policy.

    Stronger policies can be added later if required.
    """

    if len(password) < PASSWORD_MIN_LENGTH:
        raise CredentialsError(
            f"Password must contain at least "
            f"{PASSWORD_MIN_LENGTH} characters."
        )

    if password.lower() == password:
        raise CredentialsError(
            "Password must contain at least one uppercase letter."
        )

    if password.upper() == password:
        raise CredentialsError(
            "Password must contain at least one lowercase letter."
        )

    if not any(
        character.isdigit()
        for character in password
    ):
        raise CredentialsError(
            "Password must contain at least one number."
        )


# ============================================================
# Token Types
# ============================================================

class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"


# ============================================================
# JWT Helpers
# ============================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _create_token(
    subject: str,
    token_type: TokenType,
    expires_delta: timedelta,
    claims: Optional[dict[str, Any]] = None,
) -> str:
    """
    Create a signed JWT.
    """

    now = _utc_now()

    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": token_type.value,
        "iat": now,
        "exp": now + expires_delta,
        "jti": secrets.token_hex(16),
    }

    if claims:
        payload.update(claims)

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


def create_access_token(
    user_id: str,
    company_id: Optional[str] = None,
    roles: Optional[list[str]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create an access token.

    The JWT contains identity/context claims only.
    Final authorization should still be checked by
    permissions.py at the service/tool boundary.
    """

    if not user_id:
        raise CredentialsError(
            "User ID is required."
        )

    expires = (
        expires_delta
        or timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    )

    claims: dict[str, Any] = {}

    if company_id is not None:
        claims["company_id"] = str(company_id)

    if roles:
        claims["roles"] = roles

    return _create_token(
        subject=str(user_id),
        token_type=TokenType.ACCESS,
        expires_delta=expires,
        claims=claims,
    )


def create_refresh_token(
    user_id: str,
    company_id: Optional[str] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a refresh token.
    """

    if not user_id:
        raise CredentialsError(
            "User ID is required."
        )

    expires = (
        expires_delta
        or timedelta(
            days=REFRESH_TOKEN_EXPIRE_DAYS
        )
    )

    claims: dict[str, Any] = {}

    if company_id is not None:
        claims["company_id"] = str(company_id)

    return _create_token(
        subject=str(user_id),
        token_type=TokenType.REFRESH,
        expires_delta=expires,
        claims=claims,
    )


# ============================================================
# Token Decoding
# ============================================================

def decode_token(
    token: str,
) -> dict[str, Any]:
    """
    Decode and validate a JWT.

    Raises:
        InvalidTokenError
    """

    if not token:
        raise InvalidTokenError(
            "Token is missing."
        )

    try:

        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )

    except ExpiredSignatureError as exc:

        logger.info(
            "Expired authentication token"
        )

        raise InvalidTokenError(
            "Authentication token has expired."
        ) from exc

    except JWTError as exc:

        logger.warning(
            "Invalid JWT received"
        )

        raise InvalidTokenError(
            "Invalid authentication token."
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected token validation error"
        )

        raise InvalidTokenError(
            "Unable to validate authentication token."
        ) from exc

    if not payload.get("sub"):
        raise InvalidTokenError(
            "Token subject is missing."
        )

    if not payload.get("type"):
        raise InvalidTokenError(
            "Token type is missing."
        )

    return payload


def validate_token_type(
    payload: dict[str, Any],
    expected_type: TokenType,
) -> None:
    """
    Ensure the token is the expected type.
    """

    actual_type = payload.get("type")

    if actual_type != expected_type.value:
        raise InvalidTokenError(
            f"Expected {expected_type.value} token."
        )


# ============================================================
# Access Token Validation
# ============================================================

def validate_access_token(
    token: str,
) -> dict[str, Any]:
    """
    Decode and validate an access token.
    """

    payload = decode_token(token)

    validate_token_type(
        payload,
        TokenType.ACCESS,
    )

    return payload


def validate_refresh_token(
    token: str,
) -> dict[str, Any]:
    """
    Decode and validate a refresh token.
    """

    payload = decode_token(token)

    validate_token_type(
        payload,
        TokenType.REFRESH,
    )

    return payload


# ============================================================
# Token Information
# ============================================================

def get_token_subject(
    payload: dict[str, Any],
) -> str:
    """
    Get authenticated user ID from token payload.
    """

    subject = payload.get("sub")

    if not subject:
        raise InvalidTokenError(
            "Token subject is missing."
        )

    return str(subject)


def get_token_company_id(
    payload: dict[str, Any],
) -> Optional[str]:
    """
    Get company/tenant ID from token.
    """

    company_id = payload.get(
        "company_id"
    )

    if company_id is None:
        return None

    return str(company_id)


def get_token_roles(
    payload: dict[str, Any],
) -> list[str]:
    """
    Get roles embedded in token.
    """

    roles = payload.get(
        "roles",
        [],
    )

    if not isinstance(roles, list):
        return []

    return [
        str(role)
        for role in roles
    ]


def get_token_id(
    payload: dict[str, Any],
) -> Optional[str]:
    """
    Get JWT ID (jti).
    """

    token_id = payload.get("jti")

    if token_id is None:
        return None

    return str(token_id)


# ============================================================
# Authorization Header
# ============================================================

def extract_bearer_token(
    authorization: Optional[str],
) -> str:
    """
    Extract JWT from:

        Authorization: Bearer <token>
    """

    if not authorization:
        raise AuthenticationError(
            "Authorization header is missing."
        )

    parts = authorization.split()

    if len(parts) != 2:
        raise AuthenticationError(
            "Invalid authorization header."
        )

    scheme, token = parts

    if scheme.lower() != "bearer":
        raise AuthenticationError(
            "Authorization scheme must be Bearer."
        )

    if not token:
        raise AuthenticationError(
            "Authentication token is missing."
        )

    return token


# ============================================================
# Secure Random Values
# ============================================================

def generate_secure_token(
    length: int = 32,
) -> str:
    """
    Generate a cryptographically secure random token.
    """

    if length < 16:
        raise ValueError(
            "Secure token length must be at least 16."
        )

    return secrets.token_urlsafe(
        length
    )


def generate_session_id() -> str:
    """
    Generate a secure session identifier.
    """

    return secrets.token_urlsafe(32)


def generate_api_key(
    prefix: str = "finco",
) -> str:
    """
    Generate an API key.

    Example:
        finco_xxxxxxxxxxxxxxxxx
    """

    return (
        f"{prefix}_"
        f"{secrets.token_urlsafe(32)}"
    )


# ============================================================
# API Key Hashing
# ============================================================

def hash_api_key(
    api_key: str,
) -> str:
    """
    Hash an API key before storing it.

    Store the hash, not the original key.
    """

    return hashlib.sha256(
        api_key.encode("utf-8")
    ).hexdigest()


def verify_api_key(
    api_key: str,
    stored_hash: str,
) -> bool:
    """
    Verify an API key against its stored SHA-256 hash.
    """

    calculated_hash = hash_api_key(
        api_key
    )

    return secrets.compare_digest(
        calculated_hash,
        stored_hash,
    )


# ============================================================
# Password Reset Tokens
# ============================================================

def generate_password_reset_token() -> str:
    """
    Generate a secure password reset token.
    """

    return secrets.token_urlsafe(48)


def generate_email_verification_token() -> str:
    """
    Generate a secure email verification token.
    """

    return secrets.token_urlsafe(48)


# ============================================================
# Security Utilities
# ============================================================

def constant_time_compare(
    first: str,
    second: str,
) -> bool:
    """
    Constant-time string comparison.

    Useful for secrets/tokens.
    """

    return secrets.compare_digest(
        first,
        second,
    )


def mask_secret(
    value: Optional[str],
    visible_chars: int = 4,
) -> str:
    """
    Safely mask a secret for logs/UI.

    Example:
        abcdefghijkl -> abc...ijkl
    """

    if not value:
        return ""

    if len(value) <= visible_chars:
        return "*" * len(value)

    return (
        value[:visible_chars]
        + "..."
        + value[-visible_chars:]
    )


# ============================================================
# Security Validation
# ============================================================

def validate_jwt_configuration() -> None:
    """
    Validate security configuration at application startup.
    """

    if (
        not JWT_SECRET_KEY
        or JWT_SECRET_KEY
        == "CHANGE_THIS_SECRET_IN_PRODUCTION"
    ):

        if os.getenv(
            "ENVIRONMENT",
            "development",
        ).lower() in {
            "production",
            "prod",
        }:

            raise RuntimeError(
                "JWT_SECRET_KEY must be configured "
                "in production."
            )

        logger.warning(
            "Default JWT secret is being used. "
            "Configure JWT_SECRET_KEY before production."
        )

    if len(JWT_SECRET_KEY) < 32:

        logger.warning(
            "JWT_SECRET_KEY should contain at least "
            "32 characters."
        )


# ============================================================
# Security Startup
# ============================================================

validate_jwt_configuration()