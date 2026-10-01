"""
FinCo AI - Alert Escalation

Responsible for escalating financial alerts that require
additional attention or human intervention.

Responsibilities:
- Determine whether an alert requires escalation
- Determine escalation level
- Build escalation payloads
- Notify appropriate escalation targets
- Track escalation state
- Support manual escalation/de-escalation
- Maintain audit-friendly escalation history

Architecture:

    Alert
      ↓
    Severity + Risk Score
      ↓
    EscalationService
      ↓
    ┌─────────────────────────────┐
    │ L1 → Analyst                │
    │ L2 → Finance Manager        │
    │ L3 → CFO / Executive        │
    │ L4 → Emergency Response     │
    └─────────────────────────────┘
      ↓
    Notification
      ↓
    Human Review
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


# ============================================================================
# Exceptions
# ============================================================================

class EscalationError(Exception):
    """Base exception for escalation failures."""


class EscalationNotFoundError(EscalationError):
    """Raised when an escalation cannot be found."""


# ============================================================================
# Constants
# ============================================================================

class EscalationLevel:
    """Supported escalation levels."""

    NONE = "NONE"
    LEVEL_1 = "LEVEL_1"
    LEVEL_2 = "LEVEL_2"
    LEVEL_3 = "LEVEL_3"
    LEVEL_4 = "LEVEL_4"


class EscalationStatus:
    """Escalation lifecycle states."""

    PENDING = "PENDING"
    ESCALATED = "ESCALATED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    CANCELLED = "CANCELLED"


# ============================================================================
# Escalation Record
# ============================================================================

@dataclass
class EscalationRecord:
    """
    Represents an escalation event.

    This is separate from the financial Alert object so that
    one alert can have multiple escalation events.
    """

    escalation_id: str
    alert_id: str
    company_id: str

    level: str
    status: str

    reason: str

    target_role: str

    risk_score: float
    severity: str

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    escalated_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    escalated_to: Optional[str] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None

    notes: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation."""

        data = asdict(self)

        for key in (
            "created_at",
            "escalated_at",
            "acknowledged_at",
            "resolved_at",
        ):
            value = data.get(key)

            if isinstance(
                value,
                datetime,
            ):
                data[key] = value.isoformat()

        return data


# ============================================================================
# Escalation Service
# ============================================================================

class EscalationService:
    """
    Determines and manages alert escalation.

    Default policy:

        CRITICAL / >= 90
            → LEVEL_3

        HIGH / >= 75
            → LEVEL_2

        MEDIUM / >= 50
            → LEVEL_1

        LOW
            → NONE

    The policy is configurable.
    """

    def __init__(
        self,
        escalation_repository: Optional[Any] = None,
        notification_service: Optional[Any] = None,
        audit_service: Optional[Any] = None,
    ) -> None:

        self.escalation_repository = (
            escalation_repository
        )

        self.notification_service = (
            notification_service
        )

        self.audit_service = audit_service

    # ========================================================================
    # MAIN ESCALATION
    # ========================================================================

    def escalate(
        self,
        alert: Any,
        force: bool = False,
    ) -> Optional[EscalationRecord]:
        """
        Escalate an alert if escalation criteria are met.

        Parameters
        ----------
        alert:
            GeneratedAlert or compatible alert object.

        force:
            Force escalation even when the normal policy would not
            escalate the alert.
        """

        alert_id = self._get_value(
            alert,
            "alert_id",
        )

        company_id = self._get_value(
            alert,
            "company_id",
        )

        severity = str(
            self._get_value(
                alert,
                "severity",
                "LOW",
            )
        ).upper()

        risk_score = self._number(
            self._get_value(
                alert,
                "risk_score",
                0,
            )
        )

        requires_review = bool(
            self._get_value(
                alert,
                "requires_human_review",
                False,
            )
        )

        if not alert_id:
            raise EscalationError(
                "Alert must contain alert_id."
            )

        if not company_id:
            raise EscalationError(
                "Alert must contain company_id."
            )

        level = self.determine_level(
            severity=severity,
            risk_score=risk_score,
            requires_human_review=requires_review,
            force=force,
        )

        if level == EscalationLevel.NONE:
            return None

        # --------------------------------------------------------------
        # Prevent duplicate active escalations
        # --------------------------------------------------------------

        existing = self._find_active_escalation(
            alert_id
        )

        if existing is not None:
            return self._convert_record(
                existing
            )

        # --------------------------------------------------------------
        # Determine target
        # --------------------------------------------------------------

        target_role = self.target_role(
            level
        )

        reason = self.generate_reason(
            alert=alert,
            level=level,
        )

        # --------------------------------------------------------------
        # Create escalation
        # --------------------------------------------------------------

        record = EscalationRecord(
            escalation_id=self._generate_id(),
            alert_id=alert_id,
            company_id=company_id,
            level=level,
            status=EscalationStatus.ESCALATED,
            reason=reason,
            target_role=target_role,
            risk_score=risk_score,
            severity=severity,
            escalated_at=datetime.now(
                timezone.utc
            ),
            metadata={
                "alert_type": self._get_value(
                    alert,
                    "alert_type",
                ),
                "title": self._get_value(
                    alert,
                    "title",
                ),
            },
        )

        # --------------------------------------------------------------
        # Persist
        # --------------------------------------------------------------

        record = self._save(record)

        # --------------------------------------------------------------
        # Notify
        # --------------------------------------------------------------

        self._notify(record)

        # --------------------------------------------------------------
        # Audit
        # --------------------------------------------------------------

        self._audit(
            action="ALERT_ESCALATED",
            record=record,
        )

        return record

    # ========================================================================
    # DETERMINE LEVEL
    # ========================================================================

    def determine_level(
        self,
        severity: str,
        risk_score: float,
        requires_human_review: bool = False,
        force: bool = False,
    ) -> str:
        """
        Determine escalation level from severity and risk.

        Priority is intentionally conservative.
        """

        severity = str(
            severity
        ).upper()

        risk_score = self._number(
            risk_score
        )

        # --------------------------------------------------------------
        # Force escalation
        # --------------------------------------------------------------

        if force:
            return EscalationLevel.LEVEL_3

        # --------------------------------------------------------------
        # Critical
        # --------------------------------------------------------------

        if severity == "CRITICAL":
            return EscalationLevel.LEVEL_3

        if risk_score >= 90:
            return EscalationLevel.LEVEL_3

        # --------------------------------------------------------------
        # High
        # --------------------------------------------------------------

        if severity == "HIGH":
            return EscalationLevel.LEVEL_2

        if risk_score >= 75:
            return EscalationLevel.LEVEL_2

        # --------------------------------------------------------------
        # Human review
        # --------------------------------------------------------------

        if requires_human_review:
            return EscalationLevel.LEVEL_2

        # --------------------------------------------------------------
        # Medium
        # --------------------------------------------------------------

        if severity == "MEDIUM":
            return EscalationLevel.LEVEL_1

        if risk_score >= 50:
            return EscalationLevel.LEVEL_1

        return EscalationLevel.NONE

    # ========================================================================
    # TARGET ROLE
    # ========================================================================

    @staticmethod
    def target_role(
        level: str,
    ) -> str:
        """
        Map escalation level to responsible business role.
        """

        mapping = {
            EscalationLevel.LEVEL_1:
                "FINANCIAL_ANALYST",

            EscalationLevel.LEVEL_2:
                "FINANCE_MANAGER",

            EscalationLevel.LEVEL_3:
                "CFO",

            EscalationLevel.LEVEL_4:
                "EXECUTIVE_RISK_TEAM",

            EscalationLevel.NONE:
                "NONE",
        }

        return mapping.get(
            level,
            "FINANCE_MANAGER",
        )

    # ========================================================================
    # REASON
    # ========================================================================

    def generate_reason(
        self,
        alert: Any,
        level: str,
    ) -> str:
        """
        Generate a deterministic escalation reason.

        The Supervisor Agent can later provide a richer
        natural-language investigation.
        """

        severity = str(
            self._get_value(
                alert,
                "severity",
                "LOW",
            )
        ).upper()

        risk_score = self._number(
            self._get_value(
                alert,
                "risk_score",
                0,
            )
        )

        alert_type = self._get_value(
            alert,
            "alert_type",
            "FINANCIAL_RISK",
        )

        target = self.target_role(
            level
        )

        if severity == "CRITICAL":
            return (
                f"Critical financial alert '{alert_type}' "
                f"requires immediate review by {target}. "
                f"Risk score: {risk_score:.1f}/100."
            )

        if severity == "HIGH":
            return (
                f"High-risk financial alert '{alert_type}' "
                f"requires management review by {target}. "
                f"Risk score: {risk_score:.1f}/100."
            )

        return (
            f"Financial alert '{alert_type}' "
            f"requires review by {target}. "
            f"Risk score: {risk_score:.1f}/100."
        )

    # ========================================================================
    # ACKNOWLEDGE
    # ========================================================================

    def acknowledge(
        self,
        escalation_id: str,
        user_id: str,
        notes: Optional[str] = None,
    ) -> EscalationRecord:
        """
        Mark an escalation as acknowledged.
        """

        record = self._get(
            escalation_id
        )

        if record is None:
            raise EscalationNotFoundError(
                f"Escalation '{escalation_id}' "
                f"was not found."
            )

        now = datetime.now(
            timezone.utc
        )

        updates = {
            "status":
                EscalationStatus.ACKNOWLEDGED,

            "acknowledged_by":
                user_id,

            "acknowledged_at":
                now,
        }

        if notes:
            updates["notes"] = notes

        updated = self._update(
            escalation_id,
            updates,
        )

        result = self._convert_record(
            updated
        )

        self._audit(
            action="ESCALATION_ACKNOWLEDGED",
            record=result,
            metadata={
                "user_id": user_id,
                "notes": notes,
            },
        )

        return result

    # ========================================================================
    # RESOLVE
    # ========================================================================

    def resolve(
        self,
        escalation_id: str,
        user_id: str,
        resolution: Optional[str] = None,
    ) -> EscalationRecord:
        """
        Resolve an escalation after human review.
        """

        record = self._get(
            escalation_id
        )

        if record is None:
            raise EscalationNotFoundError(
                f"Escalation '{escalation_id}' "
                f"was not found."
            )

        now = datetime.now(
            timezone.utc
        )

        updates = {
            "status":
                EscalationStatus.RESOLVED,

            "resolved_by":
                user_id,

            "resolved_at":
                now,
        }

        if resolution:
            updates["resolution"] = resolution

        updated = self._update(
            escalation_id,
            updates,
        )

        result = self._convert_record(
            updated
        )

        self._audit(
            action="ESCALATION_RESOLVED",
            record=result,
            metadata={
                "user_id": user_id,
                "resolution": resolution,
            },
        )

        return result

    # ========================================================================
    # CANCEL
    # ========================================================================

    def cancel(
        self,
        escalation_id: str,
        user_id: str,
        reason: Optional[str] = None,
    ) -> EscalationRecord:
        """
        Cancel an escalation.
        """

        record = self._get(
            escalation_id
        )

        if record is None:
            raise EscalationNotFoundError(
                f"Escalation '{escalation_id}' "
                f"was not found."
            )

        updates = {
            "status":
                EscalationStatus.CANCELLED,
            "cancelled_by":
                user_id,
            "cancelled_at":
                datetime.now(
                    timezone.utc
                ),
        }

        if reason:
            updates["cancellation_reason"] = (
                reason
            )

        updated = self._update(
            escalation_id,
            updates,
        )

        result = self._convert_record(
            updated
        )

        self._audit(
            action="ESCALATION_CANCELLED",
            record=result,
            metadata={
                "user_id": user_id,
                "reason": reason,
            },
        )

        return result

    # ========================================================================
    # FIND ACTIVE
    # ========================================================================

    def _find_active_escalation(
        self,
        alert_id: str,
    ) -> Optional[Any]:
        """
        Find an existing active escalation.
        """

        if self.escalation_repository is None:
            return None

        method = getattr(
            self.escalation_repository,
            "find_active_by_alert",
            None,
        )

        if method is None:
            return None

        return method(
            alert_id
        )

    # ========================================================================
    # SAVE
    # ========================================================================

    def _save(
        self,
        record: EscalationRecord,
    ) -> EscalationRecord:

        if self.escalation_repository is None:
            return record

        method = getattr(
            self.escalation_repository,
            "create",
            None,
        )

        if method is None:
            return record

        try:

            saved = method(
                record.to_dict()
            )

            return self._convert_record(
                saved
            )

        except Exception as exc:

            raise EscalationError(
                f"Failed to save escalation "
                f"{record.escalation_id}: {exc}"
            ) from exc

    # ========================================================================
    # GET
    # ========================================================================

    def _get(
        self,
        escalation_id: str,
    ) -> Optional[Any]:

        if self.escalation_repository is None:
            return None

        method = getattr(
            self.escalation_repository,
            "get_by_id",
            None,
        )

        if method is None:
            return None

        return method(
            escalation_id
        )

    # ========================================================================
    # UPDATE
    # ========================================================================

    def _update(
        self,
        escalation_id: str,
        updates: Dict[str, Any],
    ) -> Any:

        if self.escalation_repository is None:
            raise EscalationError(
                "Escalation repository is not configured."
            )

        method = getattr(
            self.escalation_repository,
            "update",
            None,
        )

        if method is None:
            raise EscalationError(
                "Escalation repository does not "
                "implement update()."
            )

        return method(
            escalation_id,
            updates,
        )

    # ========================================================================
    # NOTIFICATION
    # ========================================================================

    def _notify(
        self,
        record: EscalationRecord,
    ) -> None:
        """
        Notify the responsible role.

        Notification service can later support:
        - Email
        - Slack
        - Microsoft Teams
        - Webhooks
        - In-app notifications
        """

        if self.notification_service is None:
            return

        try:

            method = getattr(
                self.notification_service,
                "send_escalation",
                None,
            )

            if method is not None:
                method(
                    record
                )

        except Exception as exc:

            self._audit(
                action="ESCALATION_NOTIFICATION_FAILED",
                record=record,
                metadata={
                    "error": str(exc)
                },
            )

    # ========================================================================
    # AUDIT
    # ========================================================================

    def _audit(
        self,
        action: str,
        record: EscalationRecord,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write escalation audit information.
        """

        if self.audit_service is None:
            return

        try:

            method = getattr(
                self.audit_service,
                "record",
                None,
            )

            if method is None:
                return

            payload = {
                "action": action,
                "escalation_id":
                    record.escalation_id,
                "alert_id":
                    record.alert_id,
                "company_id":
                    record.company_id,
                "level":
                    record.level,
                "status":
                    record.status,
                "severity":
                    record.severity,
                "risk_score":
                    record.risk_score,
                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                "metadata":
                    metadata or {},
            }

            method(
                payload
            )

        except Exception:
            # Audit failure should not break escalation.
            pass

    # ========================================================================
    # CONVERSION
    # ========================================================================

    def _convert_record(
        self,
        record: Any,
    ) -> EscalationRecord:

        if isinstance(
            record,
            EscalationRecord,
        ):
            return record

        if isinstance(
            record,
            dict,
        ):
            data = dict(record)

            datetime_fields = [
                "created_at",
                "escalated_at",
                "acknowledged_at",
                "resolved_at",
            ]

            for field_name in datetime_fields:

                value = data.get(
                    field_name
                )

                if isinstance(
                    value,
                    str,
                ):
                    try:
                        data[field_name] = (
                            datetime.fromisoformat(
                                value
                            )
                        )
                    except ValueError:
                        data[field_name] = None

            return EscalationRecord(
                **data
            )

        return EscalationRecord(
            escalation_id=record.escalation_id,
            alert_id=record.alert_id,
            company_id=record.company_id,
            level=record.level,
            status=record.status,
            reason=record.reason,
            target_role=record.target_role,
            risk_score=float(
                record.risk_score
            ),
            severity=record.severity,
            created_at=getattr(
                record,
                "created_at",
                datetime.now(
                    timezone.utc
                ),
            ),
            escalated_at=getattr(
                record,
                "escalated_at",
                None,
            ),
            acknowledged_at=getattr(
                record,
                "acknowledged_at",
                None,
            ),
            resolved_at=getattr(
                record,
                "resolved_at",
                None,
            ),
            escalated_to=getattr(
                record,
                "escalated_to",
                None,
            ),
            acknowledged_by=getattr(
                record,
                "acknowledged_by",
                None,
            ),
            resolved_by=getattr(
                record,
                "resolved_by",
                None,
            ),
            notes=getattr(
                record,
                "notes",
                None,
            ),
            metadata=getattr(
                record,
                "metadata",
                {},
            ) or {},
        )

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _get_value(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:

        if isinstance(
            obj,
            dict,
        ):
            return obj.get(
                key,
                default,
            )

        return getattr(
            obj,
            key,
            default,
        )

    @staticmethod
    def _number(
        value: Any,
    ) -> float:

        try:
            return max(
                0.0,
                min(
                    100.0,
                    float(value),
                ),
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _generate_id() -> str:
        return (
            f"ESC-{uuid4().hex[:12].upper()}"
        )


__all__ = [
    "EscalationError",
    "EscalationNotFoundError",
    "EscalationLevel",
    "EscalationStatus",
    "EscalationRecord",
    "EscalationService",
]