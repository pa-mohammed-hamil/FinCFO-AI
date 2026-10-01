from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class User(Base):
    """
    Application user model for FinCo AI.

    Supports:
    - Authentication
    - Role-based access control
    - Company ownership / membership
    - Account security
    - Human-in-the-loop approvals
    - Audit logging
    - AI recommendation approvals
    """

    __tablename__ = "users"

    __table_args__ = (
        Index("ix_user_email", "email", unique=True),
        Index("ix_user_username", "username", unique=True),
        Index("ix_user_company_id", "company_id"),
        Index("ix_user_role", "role"),
        Index("ix_user_status", "status"),
        Index("ix_user_is_active", "is_active"),
    )

    # ------------------------------------------------------------------
    # Primary Key
    # ------------------------------------------------------------------

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ------------------------------------------------------------------
    # Company
    # ------------------------------------------------------------------

    company_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("companies.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    username: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    first_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    last_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    display_name: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Role / Permissions
    # ------------------------------------------------------------------

    role: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="USER",
    )
    # SUPER_ADMIN / ADMIN / FINANCE_MANAGER /
    # FINANCE_ANALYST / AUDITOR / USER / VIEWER

    permissions_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Account Status
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="ACTIVE",
    )
    # ACTIVE / INACTIVE / SUSPENDED / LOCKED / PENDING

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    failed_login_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    locked_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    password_changed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Email Verification
    # ------------------------------------------------------------------

    email_verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    email_verification_token: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Password Reset
    # ------------------------------------------------------------------

    password_reset_token: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    password_reset_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # MFA
    # ------------------------------------------------------------------

    mfa_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    mfa_secret: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Human-in-the-Loop
    # ------------------------------------------------------------------

    can_approve_recommendations: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    can_review_alerts: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------
    # Preferences
    # ------------------------------------------------------------------

    timezone: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="UTC",
    )

    language: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="en",
    )

    notification_preferences_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    phone: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    job_title: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    department: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    metadata_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    company = relationship(
        "Company",
        back_populates="users",
    )

    approved_recommendations = relationship(
        "Recommendation",
        back_populates="approver",
        foreign_keys="Recommendation.approved_by",
    )

    acknowledged_risk_alerts = relationship(
        "RiskAlert",
        back_populates="acknowledger",
        foreign_keys="RiskAlert.acknowledged_by",
    )

    resolved_risk_alerts = relationship(
        "RiskAlert",
        back_populates="resolver",
        foreign_keys="RiskAlert.resolved_by",
    )

    created_scenarios = relationship(
        "Scenario",
        back_populates="creator",
        foreign_keys="Scenario.created_by",
    )

    audit_logs = relationship(
        "AuditLog",
        back_populates="user",
        foreign_keys="AuditLog.user_id",
    )

    # ------------------------------------------------------------------
    # Business Logic
    # ------------------------------------------------------------------

    @property
    def full_name(self) -> str:
        """Return the user's full name."""

        if self.last_name:
            return f"{self.first_name} {self.last_name}"

        return self.first_name

    def activate(self) -> None:
        """Activate the user account."""

        self.status = "ACTIVE"
        self.is_active = True

    def deactivate(self) -> None:
        """Deactivate the user account."""

        self.status = "INACTIVE"
        self.is_active = False

    def suspend(self) -> None:
        """Suspend the user account."""

        self.status = "SUSPENDED"
        self.is_active = False

    def lock(self, until: Optional[datetime] = None) -> None:
        """Lock the user account."""

        self.status = "LOCKED"
        self.is_active = False
        self.locked_until = until

    def record_login(self) -> None:
        """Record a successful login."""

        self.last_login_at = datetime.utcnow()
        self.failed_login_attempts = 0

    def record_failed_login(self) -> None:
        """Record a failed authentication attempt."""

        self.failed_login_attempts += 1

    def verify_email(self) -> None:
        """Mark the user's email as verified."""

        self.is_verified = True
        self.email_verified_at = datetime.utcnow()
        self.email_verification_token = None

    def has_role(self, *roles: str) -> bool:
        """Check whether the user has one of the supplied roles."""

        normalized_role = self.role.upper()

        return normalized_role in {
            role.upper()
            for role in roles
        }

    def has_permission(self, permission: str) -> bool:
        """
        Check whether the user has a specific permission.

        permissions_json is expected to contain a JSON object/list
        managed by the authorization layer.
        """

        if self.is_superuser:
            return True

        if not self.permissions_json:
            return False

        # Permission parsing should normally be handled by the
        # permissions service rather than the ORM model.
        return permission in self.permissions_json

    def can_approve(self) -> bool:
        """Check whether the user can approve AI recommendations."""

        return (
            self.is_active
            and self.can_approve_recommendations
        )

    def can_review(self) -> bool:
        """Check whether the user can review financial alerts."""

        return (
            self.is_active
            and self.can_review_alerts
        )

    def is_locked(self) -> bool:
        """Determine whether the account is currently locked."""

        if self.status == "LOCKED":
            if (
                self.locked_until is not None
                and self.locked_until <= datetime.utcnow()
            ):
                return False

            return True

        return False

    def __repr__(self) -> str:
        return (
            f"<User("
            f"id={self.id}, "
            f"username='{self.username}', "
            f"email='{self.email}', "
            f"role='{self.role}', "
            f"status='{self.status}'"
            f")>"
        )