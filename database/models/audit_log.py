"""
backend/app/database/models/audit_log.py

Immutable audit log model for FinCo AI.

Purpose
-------
Provides an append-only audit trail for security-sensitive and
business-sensitive operations.

Examples
--------
- User login/logout
- User creation / role changes
- Financial data uploads
- Transaction creation/update/void
- Fraud investigations
- Risk assessments
- Alert generation
- AI recommendations
- What-if scenarios
- Report generation/export
- Administrative actions
- Permission changes
- Data access events

Security Principles
-------------------
1. Append-only
2. Tenant isolated
3. No passwords/tokens/secrets
4. Structured metadata
5. Request correlation
6. IP/user-agent tracking
7. Outcome tracking
8. Timestamped in UTC
9. Suitable for compliance/auditing
10. Never physically update/delete audit records through
    normal application workflows
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum as SAEnum,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.database.connection import Base


# ============================================================================
# ENUMS
# ============================================================================


class AuditAction(str, enum.Enum):
    """
    High-level operation performed by an actor.
    """

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"

    UPLOAD = "upload"
    DOWNLOAD = "download"
    EXPORT = "export"
    IMPORT = "import"

    APPROVE = "approve"
    REJECT = "reject"
    REVIEW = "review"

    EXECUTE = "execute"
    SIMULATE = "simulate"
    GENERATE = "generate"

    ENABLE = "enable"
    DISABLE = "disable"

    ASSIGN = "assign"
    REVOKE = "revoke"

    ARCHIVE = "archive"
    RESTORE = "restore"


class AuditResourceType(str, enum.Enum):
    """
    Resource affected by an audit event.
    """

    USER = "user"
    COMPANY = "company"

    DOCUMENT = "document"
    UPLOAD = "upload"

    TRANSACTION = "transaction"

    FINANCIAL_RECORD = "financial_record"
    REVENUE = "revenue"
    EXPENSE = "expense"
    PNL = "pnl"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"

    FORECAST = "forecast"
    FRAUD = "fraud"
    RISK = "risk"

    ALERT = "alert"
    RECOMMENDATION = "recommendation"

    WHAT_IF = "what_if"

    REPORT = "report"

    API_KEY = "api_key"
    PERMISSION = "permission"

    AUTHENTICATION = "authentication"
    SYSTEM = "system"


class AuditOutcome(str, enum.Enum):
    """
    Result of the audited operation.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"


class AuditSeverity(str, enum.Enum):
    """
    Security/business significance of an audit event.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditActorType(str, enum.Enum):
    """
    Entity that initiated the action.
    """

    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"
    AGENT = "agent"
    SERVICE = "service"
    API = "api"


# ============================================================================
# MODEL
# ============================================================================


class AuditLog(Base):
    """
    Immutable audit record.

    Each row represents one security/business event.

    IMPORTANT
    ---------
    AuditLog should be treated as append-only.

    Application code should never modify or delete an existing
    audit event after it has been persisted.
    """

    __tablename__ = "audit_logs"

    # ------------------------------------------------------------------------
    # PRIMARY KEY
    # ------------------------------------------------------------------------

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # ------------------------------------------------------------------------
    # TENANT
    # ------------------------------------------------------------------------

    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment=(
            "Tenant/company owning the audited event. "
            "Nullable for platform-level events."
        ),
    )

    # ------------------------------------------------------------------------
    # ACTOR
    # ------------------------------------------------------------------------

    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="User/service/agent responsible for the action.",
    )

    actor_type: Mapped[AuditActorType] = mapped_column(
        SAEnum(
            AuditActorType,
            name="audit_actor_type",
            native_enum=True,
            create_constraint=True,
        ),
        nullable=False,
        default=AuditActorType.USER,
    )

    actor_email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
        comment=(
            "Snapshot of actor email at event time. "
            "Never use this as the authorization source."
        ),
    )

    actor_role: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Actor role at event time.",
    )

    # ------------------------------------------------------------------------
    # EVENT
    # ------------------------------------------------------------------------

    action: Mapped[AuditAction] = mapped_column(
        SAEnum(
            AuditAction,
            name="audit_action",
            native_enum=True,
            create_constraint=True,
        ),
        nullable=False,
        index=True,
    )

    resource_type: Mapped[AuditResourceType] = mapped_column(
        SAEnum(
            AuditResourceType,
            name="audit_resource_type",
            native_enum=True,
            create_constraint=True,
        ),
        nullable=False,
        index=True,
    )

    resource_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="ID of affected resource when available.",
    )

    resource_name: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Human-readable resource identifier/name.",
    )

    # ------------------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------------------

    outcome: Mapped[AuditOutcome] = mapped_column(
        SAEnum(
            AuditOutcome,
            name="audit_outcome",
            native_enum=True,
            create_constraint=True,
        ),
        nullable=False,
        default=AuditOutcome.SUCCESS,
        index=True,
    )

    severity: Mapped[AuditSeverity] = mapped_column(
        SAEnum(
            AuditSeverity,
            name="audit_severity",
            native_enum=True,
            create_constraint=True,
        ),
        nullable=False,
        default=AuditSeverity.LOW,
        index=True,
    )

    # ------------------------------------------------------------------------
    # DESCRIPTION
    # ------------------------------------------------------------------------

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Human-readable description of the event.",
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Sanitized error description. "
            "Must never contain credentials or secrets."
        ),
    )

    # ------------------------------------------------------------------------
    # REQUEST / TRACE INFORMATION
    # ------------------------------------------------------------------------

    request_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="HTTP/request correlation ID.",
    )

    correlation_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="Cross-service workflow correlation ID.",
    )

    trace_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
        comment="Distributed tracing identifier.",
    )

    # ------------------------------------------------------------------------
    # NETWORK INFORMATION
    # ------------------------------------------------------------------------

    ip_address: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="Client IP address when available.",
    )

    user_agent: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # API INFORMATION
    # ------------------------------------------------------------------------

    endpoint: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="API route that generated the event.",
    )

    http_method: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
    )

    http_status_code: Mapped[int | None] = mapped_column(
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # SERVICE / AI INFORMATION
    # ------------------------------------------------------------------------

    service_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    service_version: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    agent_name: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
        comment="AI agent responsible for the action, if applicable.",
    )

    model_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment="AI/ML model used, if applicable.",
    )

    # ------------------------------------------------------------------------
    # STRUCTURED METADATA
    # ------------------------------------------------------------------------

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment=(
            "Structured non-sensitive event metadata. "
            "Secrets must never be stored here."
        ),
    )

    # ------------------------------------------------------------------------
    # DATA CHANGE INFORMATION
    # ------------------------------------------------------------------------

    changed_fields: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Names of changed fields; values should not contain secrets.",
    )

    previous_values: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment=(
            "Sanitized previous values for auditable changes. "
            "Do not store passwords, tokens, or secrets."
        ),
    )

    new_values: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
        comment=(
            "Sanitized new values for auditable changes. "
            "Do not store passwords, tokens, or secrets."
        ),
    )

    # ------------------------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    # ------------------------------------------------------------------------
    # INTEGRITY / RETENTION
    # ------------------------------------------------------------------------

    checksum: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "Optional tamper-evidence checksum/hash generated "
            "by the audit service."
        ),
    )

    retention_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Optional compliance retention deadline.",
    )

    # =========================================================================
    # TABLE INDEXES
    # =========================================================================

    __table_args__ = (
        Index(
            "ix_audit_logs_company_created",
            "company_id",
            "created_at",
        ),
        Index(
            "ix_audit_logs_company_resource",
            "company_id",
            "resource_type",
            "resource_id",
        ),
        Index(
            "ix_audit_logs_company_actor",
            "company_id",
            "actor_id",
            "created_at",
        ),
        Index(
            "ix_audit_logs_company_action",
            "company_id",
            "action",
            "created_at",
        ),
        Index(
            "ix_audit_logs_company_outcome",
            "company_id",
            "outcome",
            "created_at",
        ),
        Index(
            "ix_audit_logs_request",
            "request_id",
        ),
        Index(
            "ix_audit_logs_correlation",
            "correlation_id",
        ),
    )

    # =========================================================================
    # REPRESENTATION
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"<AuditLog("
            f"id={self.id}, "
            f"company_id={self.company_id}, "
            f"actor_id={self.actor_id}, "
            f"action={self.action.value}, "
            f"resource_type={self.resource_type.value}, "
            f"outcome={self.outcome.value}, "
            f"created_at={self.created_at}"
            f")>"
        )

    # =========================================================================
    # FACTORY
    # =========================================================================

    @classmethod
    def create_event(
        cls,
        *,
        action: AuditAction,
        resource_type: AuditResourceType,
        description: str,
        company_id: uuid.UUID | None = None,
        actor_id: uuid.UUID | None = None,
        actor_type: AuditActorType = AuditActorType.USER,
        actor_email: str | None = None,
        actor_role: str | None = None,
        resource_id: uuid.UUID | None = None,
        resource_name: str | None = None,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        severity: AuditSeverity = AuditSeverity.LOW,
        request_id: str | None = None,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        endpoint: str | None = None,
        http_method: str | None = None,
        http_status_code: int | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        agent_name: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        changed_fields: list[str] | None = None,
        previous_values: dict[str, Any] | None = None,
        new_values: dict[str, Any] | None = None,
        checksum: str | None = None,
        retention_until: datetime | None = None,
    ) -> "AuditLog":
        """
        Construct an audit event.

        Sanitization of sensitive values should preferably happen
        inside the dedicated audit service before calling this method.
        """

        return cls(
            company_id=company_id,
            actor_id=actor_id,
            actor_type=actor_type,
            actor_email=actor_email,
            actor_role=actor_role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            outcome=outcome,
            severity=severity,
            description=description,
            request_id=request_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            ip_address=ip_address,
            user_agent=user_agent,
            endpoint=endpoint,
            http_method=http_method,
            http_status_code=http_status_code,
            service_name=service_name,
            service_version=service_version,
            agent_name=agent_name,
            model_name=model_name,
            metadata_json=metadata or {},
            changed_fields=changed_fields or [],
            previous_values=previous_values,
            new_values=new_values,
            checksum=checksum,
            retention_until=retention_until,
        )

    # =========================================================================
    # IMMUTABILITY
    # =========================================================================

    def update_from_dict(
        self,
        values: dict[str, Any],
    ) -> None:
        """
        Prevent accidental mutation of persisted audit events.

        Audit records are append-only.

        Use a new audit event to record a correction or follow-up
        rather than modifying the original event.
        """

        raise RuntimeError(
            "AuditLog records are immutable. "
            "Create a new audit event instead."
        )

    def delete(self) -> None:
        """
        Prevent application-level deletion.
        """

        raise RuntimeError(
            "AuditLog records are immutable and cannot be deleted "
            "through the application service."
        )