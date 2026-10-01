"""
FinCo AI - Alert Agent Tools

Path:
    backend/app/agents/tools/alert_tools.py

Purpose:
    Exposes alert-management capabilities to FinCo AI agents.

    The tools are intentionally thin orchestration wrappers around
    the application's AlertService.

Architecture:

    Supervisor / Risk / Financial Agent
                    |
                    v
              alert_tools.py
                    |
                    v
              AlertService
                    |
          +---------+---------+
          |         |         |
          v         v         v
      Repository Notification Escalation
          |
          v
       Audit Log

Supported operations:
    - create_alert
    - create_alerts
    - get_alert
    - get_company_alerts
    - get_open_alerts
    - get_alert_summary
    - acknowledge_alert
    - resolve_alert
    - reopen_alert
    - find_duplicate_alert

Design principles:
    - No database access directly from tools.
    - No notification implementation here.
    - No risk scoring implementation here.
    - No LLM reasoning here.
    - Validate agent inputs before calling services.
    - Return structured dictionaries suitable for tool calling.
    - Preserve auditability and traceability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


# ============================================================================
# Exceptions
# ============================================================================


class AlertToolError(Exception):
    """Base exception for alert tools."""


class InvalidAlertToolInputError(AlertToolError):
    """Raised when alert-tool input is invalid."""


class AlertToolExecutionError(AlertToolError):
    """Raised when an alert operation fails."""


class AlertToolConfigurationError(AlertToolError):
    """Raised when AlertService is not configured correctly."""


# ============================================================================
# Constants
# ============================================================================


DEFAULT_ALERT_TYPE = "financial_risk"

DEFAULT_SEVERITY = "medium"

DEFAULT_STATUS = "open"

SUPPORTED_SEVERITIES = {
    "info",
    "low",
    "medium",
    "high",
    "critical",
}

SUPPORTED_STATUSES = {
    "open",
    "acknowledged",
    "resolved",
    "closed",
}


# ============================================================================
# Tool Result
# ============================================================================


@dataclass(slots=True)
class AlertToolResult:
    """
    Standard response returned by alert tools.

    Keeping a consistent response envelope makes these tools easier
    to consume from agent workflows and API adapters.
    """

    success: bool

    operation: str

    message: str

    data: Any = None

    error: Optional[str] = None

    timestamp: datetime = None

    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(
                timezone.utc
            )

        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "message": self.message,
            "data": _serialize(self.data),
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Alert Tool Service
# ============================================================================


class AlertTools:
    """
    Agent-facing alert tool collection.

    Parameters
    ----------
    alert_service:
        Application AlertService instance.

    audit_service:
        Optional audit service. AlertService normally owns its
        own audit behavior, but this layer can record tool usage
        when required.

    authorization_service:
        Optional permission service used to restrict alert
        mutations such as acknowledge/resolve/reopen.
    """

    def __init__(
        self,
        *,
        alert_service: Any,
        audit_service: Any = None,
        authorization_service: Any = None,
    ) -> None:

        if alert_service is None:
            raise AlertToolConfigurationError(
                "alert_service is required."
            )

        self.alert_service = alert_service

        self.audit_service = audit_service

        self.authorization_service = (
            authorization_service
        )

    # ========================================================================
    # CREATE ALERT
    # ========================================================================

    def create_alert(
        self,
        *,
        company_id: str,
        alert_type: str = DEFAULT_ALERT_TYPE,
        severity: str = DEFAULT_SEVERITY,
        title: str,
        message: str,
        risk_score: float = 0.0,
        evidence: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
        recommendations: Optional[
            Sequence[Mapping[str, Any]]
        ] = None,
        source: str = "agent",
        requires_human_review: bool = False,
        user_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> dict[str, Any]:
        """
        Create a single alert.

        This method delegates creation to AlertService, which is
        responsible for duplicate detection, persistence,
        notifications, escalation and audit behavior.
        """

        self._validate_company_id(
            company_id
        )

        self._validate_text(
            title,
            "title",
        )

        self._validate_text(
            message,
            "message",
        )

        severity = self._normalize_severity(
            severity
        )

        risk_score = self._normalize_score(
            risk_score
        )

        self._authorize(
            user_id=user_id,
            action="create_alert",
            company_id=company_id,
        )

        payload = {
            "company_id": company_id,
            "alert_type": alert_type.strip(),
            "severity": severity,
            "title": title.strip(),
            "message": message.strip(),
            "risk_score": risk_score,
            "evidence": list(
                evidence or []
            ),
            "recommendations": list(
                recommendations or []
            ),
            "source": source,
            "requires_human_review":
                bool(
                    requires_human_review
                ),
            "metadata": dict(
                metadata or {}
            ),
        }

        try:

            result = self._call_service(
                (
                    "create_alert",
                    "create",
                    "generate_alert",
                ),
                payload,
            )

            self._audit_tool_usage(
                operation="create_alert",
                company_id=company_id,
                user_id=user_id,
                result=result,
            )

            return AlertToolResult(
                success=True,
                operation="create_alert",
                message="Alert created successfully.",
                data=result,
                metadata={
                    "company_id": company_id,
                    "alert_type": alert_type,
                    "severity": severity,
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to create alert: {exc}"
            ) from exc

    # ========================================================================
    # CREATE MULTIPLE ALERTS
    # ========================================================================

    def create_alerts(
        self,
        *,
        alerts: Sequence[
            Mapping[str, Any]
        ],
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create multiple alerts.

        Uses AlertService.create_alerts when available.
        Otherwise falls back to individual creation.
        """

        if not alerts:
            raise InvalidAlertToolInputError(
                "alerts cannot be empty."
            )

        normalized_alerts = []

        for alert in alerts:

            if not isinstance(
                alert,
                Mapping,
            ):
                raise InvalidAlertToolInputError(
                    "Each alert must be a mapping."
                )

            normalized = dict(
                alert
            )

            company_id = str(
                normalized.get(
                    "company_id",
                    "",
                )
            ).strip()

            title = str(
                normalized.get(
                    "title",
                    "",
                )
            ).strip()

            message = str(
                normalized.get(
                    "message",
                    "",
                )
            ).strip()

            if not company_id:
                raise InvalidAlertToolInputError(
                    "Each alert requires company_id."
                )

            if not title:
                raise InvalidAlertToolInputError(
                    "Each alert requires title."
                )

            if not message:
                raise InvalidAlertToolInputError(
                    "Each alert requires message."
                )

            normalized[
                "severity"
            ] = self._normalize_severity(
                normalized.get(
                    "severity",
                    DEFAULT_SEVERITY,
                )
            )

            normalized[
                "risk_score"
            ] = self._normalize_score(
                normalized.get(
                    "risk_score",
                    0,
                )
            )

            normalized_alerts.append(
                normalized
            )

        try:

            if hasattr(
                self.alert_service,
                "create_alerts",
            ):

                result = self.alert_service.create_alerts(
                    normalized_alerts
                )

            else:

                result = [
                    self.create_alert(
                        **alert,
                        user_id=user_id,
                    )
                    for alert
                    in normalized_alerts
                ]

            return AlertToolResult(
                success=True,
                operation="create_alerts",
                message=(
                    f"{len(normalized_alerts)} "
                    "alert(s) processed successfully."
                ),
                data=result,
                metadata={
                    "count":
                        len(normalized_alerts),
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to create alerts: {exc}"
            ) from exc

    # ========================================================================
    # GET ALERT
    # ========================================================================

    def get_alert(
        self,
        *,
        alert_id: str,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve a single alert."""

        self._validate_alert_id(
            alert_id
        )

        try:

            result = self._call_service(
                (
                    "get_alert",
                    "find_alert",
                    "retrieve_alert",
                ),
                {
                    "alert_id":
                        alert_id.strip(),
                },
            )

            return AlertToolResult(
                success=True,
                operation="get_alert",
                message="Alert retrieved successfully.",
                data=result,
                metadata={
                    "alert_id":
                        alert_id,
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to retrieve alert: {exc}"
            ) from exc

    # ========================================================================
    # GET COMPANY ALERTS
    # ========================================================================

    def get_company_alerts(
        self,
        *,
        company_id: str,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: int = 100,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve alerts belonging to a company."""

        self._validate_company_id(
            company_id
        )

        limit = self._validate_limit(
            limit
        )

        if status is not None:
            status = self._normalize_status(
                status
            )

        if severity is not None:
            severity = self._normalize_severity(
                severity
            )

        try:

            payload = {
                "company_id":
                    company_id,

                "status":
                    status,

                "severity":
                    severity,

                "alert_type":
                    alert_type,

                "limit":
                    limit,
            }

            result = self._call_service(
                (
                    "get_company_alerts",
                    "list_company_alerts",
                    "get_alerts",
                ),
                payload,
            )

            return AlertToolResult(
                success=True,
                operation="get_company_alerts",
                message="Company alerts retrieved.",
                data=result,
                metadata={
                    "company_id":
                        company_id,

                    "limit":
                        limit,
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to retrieve company alerts: {exc}"
            ) from exc

    # ========================================================================
    # GET OPEN ALERTS
    # ========================================================================

    def get_open_alerts(
        self,
        *,
        company_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve currently open alerts."""

        if company_id is not None:
            self._validate_company_id(
                company_id
            )

        limit = self._validate_limit(
            limit
        )

        if severity is not None:

            severity = self._normalize_severity(
                severity
            )

        payload = {
            "company_id":
                company_id,

            "severity":
                severity,

            "limit":
                limit,
        }

        try:

            result = self._call_service(
                (
                    "get_open_alerts",
                    "list_open_alerts",
                ),
                payload,
            )

            return AlertToolResult(
                success=True,
                operation="get_open_alerts",
                message="Open alerts retrieved.",
                data=result,
                metadata={
                    "company_id":
                        company_id,

                    "limit":
                        limit,
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to retrieve open alerts: {exc}"
            ) from exc

    # ========================================================================
    # ALERT SUMMARY
    # ========================================================================

    def get_alert_summary(
        self,
        *,
        company_id: str,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Retrieve an alert summary.

        Typically includes counts by severity/status/type.
        """

        self._validate_company_id(
            company_id
        )

        try:

            result = self._call_service(
                (
                    "get_alert_summary",
                    "alert_summary",
                    "summary",
                ),
                {
                    "company_id":
                        company_id,
                },
            )

            return AlertToolResult(
                success=True,
                operation="get_alert_summary",
                message="Alert summary retrieved.",
                data=result,
                metadata={
                    "company_id":
                        company_id,
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to retrieve alert summary: {exc}"
            ) from exc

    # ========================================================================
    # ACKNOWLEDGE ALERT
    # ========================================================================

    def acknowledge_alert(
        self,
        *,
        alert_id: str,
        user_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Acknowledge an alert."""

        self._validate_alert_id(
            alert_id
        )

        self._authorize(
            user_id=user_id,
            action="acknowledge_alert",
            alert_id=alert_id,
        )

        payload = {
            "alert_id":
                alert_id.strip(),

            "user_id":
                user_id,

            "note":
                note,
        }

        try:

            result = self._call_service(
                (
                    "acknowledge_alert",
                    "acknowledge",
                ),
                payload,
            )

            self._audit_tool_usage(
                operation="acknowledge_alert",
                company_id=None,
                user_id=user_id,
                result=result,
            )

            return AlertToolResult(
                success=True,
                operation="acknowledge_alert",
                message="Alert acknowledged successfully.",
                data=result,
                metadata={
                    "alert_id":
                        alert_id,
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to acknowledge alert: {exc}"
            ) from exc

    # ========================================================================
    # RESOLVE ALERT
    # ========================================================================

    def resolve_alert(
        self,
        *,
        alert_id: str,
        user_id: Optional[str] = None,
        resolution_note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Resolve an alert."""

        self._validate_alert_id(
            alert_id
        )

        self._authorize(
            user_id=user_id,
            action="resolve_alert",
            alert_id=alert_id,
        )

        payload = {
            "alert_id":
                alert_id.strip(),

            "user_id":
                user_id,

            "resolution_note":
                resolution_note,
        }

        try:

            result = self._call_service(
                (
                    "resolve_alert",
                    "resolve",
                ),
                payload,
            )

            self._audit_tool_usage(
                operation="resolve_alert",
                company_id=None,
                user_id=user_id,
                result=result,
            )

            return AlertToolResult(
                success=True,
                operation="resolve_alert",
                message="Alert resolved successfully.",
                data=result,
                metadata={
                    "alert_id":
                        alert_id,
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to resolve alert: {exc}"
            ) from exc

    # ========================================================================
    # REOPEN ALERT
    # ========================================================================

    def reopen_alert(
        self,
        *,
        alert_id: str,
        user_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Reopen a previously resolved alert."""

        self._validate_alert_id(
            alert_id
        )

        self._authorize(
            user_id=user_id,
            action="reopen_alert",
            alert_id=alert_id,
        )

        payload = {
            "alert_id":
                alert_id.strip(),

            "user_id":
                user_id,

            "reason":
                reason,
        }

        try:

            result = self._call_service(
                (
                    "reopen_alert",
                    "reopen",
                ),
                payload,
            )

            self._audit_tool_usage(
                operation="reopen_alert",
                company_id=None,
                user_id=user_id,
                result=result,
            )

            return AlertToolResult(
                success=True,
                operation="reopen_alert",
                message="Alert reopened successfully.",
                data=result,
                metadata={
                    "alert_id":
                        alert_id,
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to reopen alert: {exc}"
            ) from exc

    # ========================================================================
    # FIND DUPLICATE ALERT
    # ========================================================================

    def find_duplicate_alert(
        self,
        *,
        company_id: str,
        alert_type: str,
        title: Optional[str] = None,
        severity: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Check whether an active duplicate alert exists.

        Useful for agents before attempting alert creation.
        """

        self._validate_company_id(
            company_id
        )

        self._validate_text(
            alert_type,
            "alert_type",
        )

        if severity is not None:

            severity = self._normalize_severity(
                severity
            )

        payload = {
            "company_id":
                company_id,

            "alert_type":
                alert_type.strip(),

            "title":
                title,

            "severity":
                severity,
        }

        try:

            # Prefer public AlertService helper if available.
            method = getattr(
                self.alert_service,
                "find_active_duplicate",
                None,
            )

            if method is not None:

                duplicate = method(
                    **payload
                )

            else:

                method = getattr(
                    self.alert_service,
                    "_find_duplicate_alert",
                    None,
                )

                if method is None:

                    return AlertToolResult(
                        success=True,
                        operation="find_duplicate_alert",
                        message=(
                            "Duplicate lookup is not "
                            "available on AlertService."
                        ),
                        data={
                            "duplicate_found":
                                False,
                            "supported":
                                False,
                        },
                    ).to_dict()

                duplicate = method(
                    **payload
                )

            return AlertToolResult(
                success=True,
                operation="find_duplicate_alert",
                message="Duplicate alert lookup completed.",
                data={
                    "duplicate_found":
                        duplicate is not None,

                    "alert":
                        duplicate,
                },
                metadata={
                    "company_id":
                        company_id,

                    "alert_type":
                        alert_type,
                },
            ).to_dict()

        except Exception as exc:

            raise AlertToolExecutionError(
                f"Failed to find duplicate alert: {exc}"
            ) from exc

    # ========================================================================
    # Service Invocation
    # ========================================================================

    def _call_service(
        self,
        method_names: Sequence[str],
        payload: Mapping[str, Any],
    ) -> Any:
        """
        Call the first compatible AlertService method.

        This allows the tool layer to remain compatible with
        slightly different service implementations while keeping
        the public tool API stable.
        """

        last_error: Optional[
            Exception
        ] = None

        for method_name in method_names:

            method = getattr(
                self.alert_service,
                method_name,
                None,
            )

            if method is None:
                continue

            try:

                return method(
                    **payload
                )

            except TypeError as exc:

                last_error = exc

                # Some services accept a single payload object.
                try:

                    return method(
                        dict(payload)
                    )

                except TypeError as second_exc:

                    last_error = second_exc

        if callable(
            self.alert_service
        ):

            try:

                return self.alert_service(
                    dict(payload)
                )

            except Exception as exc:

                last_error = exc

        if last_error is not None:

            raise AlertToolConfigurationError(
                "No compatible AlertService method "
                f"could process the operation: "
                f"{last_error}"
            )

        raise AlertToolConfigurationError(
            "No compatible AlertService method found."
        )

    # ========================================================================
    # Authorization
    # ========================================================================

    def _authorize(
        self,
        *,
        user_id: Optional[str],
        action: str,
        company_id: Optional[str] = None,
        alert_id: Optional[str] = None,
    ) -> None:
        """
        Optional authorization hook.

        Production deployments should inject the existing
        permissions service rather than putting role logic here.
        """

        if self.authorization_service is None:
            return

        payload = {
            "user_id":
                user_id,

            "action":
                action,

            "company_id":
                company_id,

            "alert_id":
                alert_id,
        }

        method = getattr(
            self.authorization_service,
            "authorize",
            None,
        )

        if method is None:

            method = getattr(
                self.authorization_service,
                "check_permission",
                None,
            )

        if method is None:

            raise AlertToolConfigurationError(
                "authorization_service does not expose "
                "authorize/check_permission."
            )

        try:

            allowed = method(
                **payload
            )

        except TypeError:

            allowed = method(
                user_id,
                action,
                company_id,
                alert_id,
            )

        if allowed is False:

            raise AlertToolExecutionError(
                "User is not authorized to perform "
                f"alert operation: {action}."
            )

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_company_id(
        company_id: str,
    ) -> None:
        if not isinstance(
            company_id,
            str,
        ) or not company_id.strip():

            raise InvalidAlertToolInputError(
                "company_id is required."
            )

    @staticmethod
    def _validate_alert_id(
        alert_id: str,
    ) -> None:
        if not isinstance(
            alert_id,
            str,
        ) or not alert_id.strip():

            raise InvalidAlertToolInputError(
                "alert_id is required."
            )

    @staticmethod
    def _validate_text(
        value: str,
        field_name: str,
    ) -> None:
        if not isinstance(
            value,
            str,
        ) or not value.strip():

            raise InvalidAlertToolInputError(
                f"{field_name} is required."
            )

    @staticmethod
    def _validate_limit(
        limit: int,
    ) -> int:
        try:

            limit = int(
                limit
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidAlertToolInputError(
                "limit must be an integer."
            ) from exc

        if limit < 1:

            raise InvalidAlertToolInputError(
                "limit must be >= 1."
            )

        if limit > 1000:

            raise InvalidAlertToolInputError(
                "limit cannot exceed 1000."
            )

        return limit

    @staticmethod
    def _normalize_severity(
        severity: str,
    ) -> str:
        if not isinstance(
            severity,
            str,
        ):

            raise InvalidAlertToolInputError(
                "severity must be a string."
            )

        value = severity.strip().lower()

        if value not in SUPPORTED_SEVERITIES:

            raise InvalidAlertToolInputError(
                "Unsupported alert severity: "
                f"{severity}."
            )

        return value

    @staticmethod
    def _normalize_status(
        status: str,
    ) -> str:
        if not isinstance(
            status,
            str,
        ):

            raise InvalidAlertToolInputError(
                "status must be a string."
            )

        value = status.strip().lower()

        if value not in SUPPORTED_STATUSES:

            raise InvalidAlertToolInputError(
                "Unsupported alert status: "
                f"{status}."
            )

        return value

    @staticmethod
    def _normalize_score(
        score: Any,
    ) -> float:
        try:

            value = float(
                score
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidAlertToolInputError(
                "risk_score must be numeric."
            ) from exc

        if 0 <= value <= 1:
            value *= 100

        if not 0 <= value <= 100:

            raise InvalidAlertToolInputError(
                "risk_score must be between 0 and 100."
            )

        return round(
            value,
            4,
        )

    # ========================================================================
    # Audit
    # ========================================================================

    def _audit_tool_usage(
        self,
        *,
        operation: str,
        company_id: Optional[str],
        user_id: Optional[str],
        result: Any,
    ) -> None:
        """Record tool-level audit event when configured."""

        if self.audit_service is None:
            return

        payload = {
            "event":
                f"agent_alert_tool.{operation}",

            "company_id":
                company_id,

            "user_id":
                user_id,

            "result":
                _serialize(result),

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        try:

            method = getattr(
                self.audit_service,
                "record",
                None,
            )

            if method is not None:

                method(
                    **payload
                )

                return

            method = getattr(
                self.audit_service,
                "log",
                None,
            )

            if method is not None:

                method(
                    **payload
                )

        except Exception:
            # Tool auditing must not make the alert operation
            # itself fail.
            pass


# ============================================================================
# Standalone Tool Functions
# ============================================================================


def create_alert(
    alert_service: Any,
    *,
    company_id: str,
    alert_type: str,
    severity: str,
    title: str,
    message: str,
    risk_score: float = 0.0,
    evidence: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
    recommendations: Optional[
        Sequence[Mapping[str, Any]]
    ] = None,
    source: str = "agent",
    requires_human_review: bool = False,
    user_id: Optional[str] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> dict[str, Any]:
    """
    Standalone wrapper for agent/tool registries.
    """

    tools = AlertTools(
        alert_service=alert_service
    )

    return tools.create_alert(
        company_id=company_id,
        alert_type=alert_type,
        severity=severity,
        title=title,
        message=message,
        risk_score=risk_score,
        evidence=evidence,
        recommendations=recommendations,
        source=source,
        requires_human_review=
            requires_human_review,
        user_id=user_id,
        metadata=metadata,
    )


def get_alert(
    alert_service: Any,
    *,
    alert_id: str,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone alert retrieval wrapper."""

    tools = AlertTools(
        alert_service=alert_service
    )

    return tools.get_alert(
        alert_id=alert_id,
        user_id=user_id,
    )


def get_company_alerts(
    alert_service: Any,
    *,
    company_id: str,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = 100,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone company-alert lookup wrapper."""

    tools = AlertTools(
        alert_service=alert_service
    )

    return tools.get_company_alerts(
        company_id=company_id,
        status=status,
        severity=severity,
        alert_type=alert_type,
        limit=limit,
        user_id=user_id,
    )


def get_open_alerts(
    alert_service: Any,
    *,
    company_id: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone open-alert lookup wrapper."""

    tools = AlertTools(
        alert_service=alert_service
    )

    return tools.get_open_alerts(
        company_id=company_id,
        severity=severity,
        limit=limit,
        user_id=user_id,
    )


def get_alert_summary(
    alert_service: Any,
    *,
    company_id: str,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone alert-summary wrapper."""

    tools = AlertTools(
        alert_service=alert_service
    )

    return tools.get_alert_summary(
        company_id=company_id,
        user_id=user_id,
    )


def acknowledge_alert(
    alert_service: Any,
    *,
    alert_id: str,
    user_id: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone acknowledge wrapper."""

    tools = AlertTools(
        alert_service=alert_service
    )

    return tools.acknowledge_alert(
        alert_id=alert_id,
        user_id=user_id,
        note=note,
    )


def resolve_alert(
    alert_service: Any,
    *,
    alert_id: str,
    user_id: Optional[str] = None,
    resolution_note: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone resolve wrapper."""

    tools = AlertTools(
        alert_service=alert_service
    )

    return tools.resolve_alert(
        alert_id=alert_id,
        user_id=user_id,
        resolution_note=resolution_note,
    )


def reopen_alert(
    alert_service: Any,
    *,
    alert_id: str,
    user_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone reopen wrapper."""

    tools = AlertTools(
        alert_service=alert_service
    )

    return tools.reopen_alert(
        alert_id=alert_id,
        user_id=user_id,
        reason=reason,
    )


def find_duplicate_alert(
    alert_service: Any,
    *,
    company_id: str,
    alert_type: str,
    title: Optional[str] = None,
    severity: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone duplicate-alert lookup wrapper."""

    tools = AlertTools(
        alert_service=alert_service
    )

    return tools.find_duplicate_alert(
        company_id=company_id,
        alert_type=alert_type,
        title=title,
        severity=severity,
        user_id=user_id,
    )


# ============================================================================
# Serialization
# ============================================================================


def _serialize(
    value: Any,
) -> Any:
    """
    Convert application objects into JSON-compatible values.

    Supports:
        - Enum
        - datetime
        - mappings
        - lists/tuples
        - dataclasses
        - Pydantic-like model_dump()
        - to_dict()
    """

    if value is None:
        return None

    if isinstance(
        value,
        Enum,
    ):
        return value.value

    if isinstance(
        value,
        datetime,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(key):
                _serialize(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return [
            _serialize(item)
            for item
            in value
        ]

    if hasattr(
        value,
        "to_dict",
    ):

        return _serialize(
            value.to_dict()
        )

    if hasattr(
        value,
        "model_dump",
    ):

        return _serialize(
            value.model_dump()
        )

    if is_dataclass(
        value
    ):

        return _serialize(
            asdict(value)
        )

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return value

    return str(value)


# ============================================================================
# Tool Registry Metadata
# ============================================================================


ALERT_TOOL_DEFINITIONS = [
    {
        "name":
            "create_alert",

        "description":
            (
                "Create a financial, fraud, liquidity, "
                "risk or business alert."
            ),

        "category":
            "alerts",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_alert",

        "description":
            "Retrieve a specific alert by ID.",

        "category":
            "alerts",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_company_alerts",

        "description":
            "Retrieve alerts for a company.",

        "category":
            "alerts",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_open_alerts",

        "description":
            "Retrieve currently open alerts.",

        "category":
            "alerts",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_alert_summary",

        "description":
            "Return an alert summary for a company.",

        "category":
            "alerts",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "acknowledge_alert",

        "description":
            "Acknowledge an existing alert.",

        "category":
            "alerts",

        "requires_confirmation":
            True,
    },
    {
        "name":
            "resolve_alert",

        "description":
            "Resolve an existing alert.",

        "category":
            "alerts",

        "requires_confirmation":
            True,
    },
    {
        "name":
            "reopen_alert",

        "description":
            "Reopen a previously resolved alert.",

        "category":
            "alerts",

        "requires_confirmation":
            True,
    },
    {
        "name":
            "find_duplicate_alert",

        "description":
            "Check whether an active duplicate alert exists.",

        "category":
            "alerts",

        "requires_confirmation":
            False,
    },
]


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "AlertToolError",
    "InvalidAlertToolInputError",
    "AlertToolExecutionError",
    "AlertToolConfigurationError",

    # Result
    "AlertToolResult",

    # Main tool collection
    "AlertTools",

    # Standalone functions
    "create_alert",
    "get_alert",
    "get_company_alerts",
    "get_open_alerts",
    "get_alert_summary",
    "acknowledge_alert",
    "resolve_alert",
    "reopen_alert",
    "find_duplicate_alert",

    # Registry
    "ALERT_TOOL_DEFINITIONS",
]