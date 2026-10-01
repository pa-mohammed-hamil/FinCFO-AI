from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ============================================================
# User Registration
# ============================================================

class UserCreate(BaseModel):
    """Schema for creating a new user."""

    email: EmailStr

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )

    full_name: str = Field(
        ...,
        min_length=2,
        max_length=150,
    )

    company_id: Optional[int] = None


# ============================================================
# User Login
# ============================================================

class LoginRequest(BaseModel):
    """Schema for user authentication."""

    email: EmailStr

    password: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )


# ============================================================
# Token Response
# ============================================================

class TokenResponse(BaseModel):
    """JWT authentication response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ============================================================
# Refresh Token
# ============================================================

class RefreshTokenRequest(BaseModel):
    """Schema for refreshing an access token."""

    refresh_token: str = Field(
        ...,
        min_length=1,
    )


class RefreshTokenResponse(BaseModel):
    """Response returned after refreshing authentication."""

    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int


# ============================================================
# User Response
# ============================================================

class UserResponse(BaseModel):
    """Safe public representation of a user."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: str

    company_id: Optional[int] = None

    is_active: bool = True
    is_verified: bool = False

    created_at: datetime
    updated_at: Optional[datetime] = None


# ============================================================
# Current User
# ============================================================

class CurrentUserResponse(UserResponse):
    """Authenticated user's profile."""

    role: str = "user"


# ============================================================
# Password Change
# ============================================================

class PasswordChangeRequest(BaseModel):
    """Schema for changing an existing password."""

    current_password: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )


# ============================================================
# Password Reset Request
# ============================================================

class PasswordResetRequest(BaseModel):
    """Request a password reset."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Confirm password reset using a reset token."""

    token: str = Field(
        ...,
        min_length=1,
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )


# ============================================================
# Email Verification
# ============================================================

class EmailVerificationRequest(BaseModel):
    """Verify a user's email address."""

    token: str = Field(
        ...,
        min_length=1,
    )


class EmailVerificationResponse(BaseModel):
    """Email verification result."""

    message: str
    verified: bool


# ============================================================
# Authentication Response
# ============================================================

class AuthResponse(BaseModel):
    """Combined authentication response."""

    user: UserResponse
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ============================================================
# Generic Authentication Message
# ============================================================

class AuthMessage(BaseModel):
    """Generic authentication operation response."""

    message: str
    success: bool = True