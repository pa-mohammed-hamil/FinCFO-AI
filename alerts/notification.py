"""
FinCo AI - Alert Notification Service

Responsible for delivering financial alerts and escalation notifications.

Responsibilities:
- Route alerts to appropriate notification channels
- Support in-app, email, webhook and messaging integrations
- Apply severity-based notification policies
- Prevent duplicate notifications
- Retry failed deliveries
- Track notification status
- Provide audit-friendly delivery records

Architecture:

    Alert / Escalation
           │
           ▼
    NotificationService
           │
      ┌────┼───────────────┐
      ▼    ▼       ▼       ▼
    In-App Email  Webhook Messaging
      │    │       │       │
      └────┴───────┴───────┘
                 │
                 ▼
          Delivery Tracking
"""

from __future__ import annotations

import logging
import time

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================

class NotificationError(Exception):
    """Base exception for notification failures."""


class NotificationDeliveryError(NotificationError):
    """Raised when notification delivery fails."""


# ============================================================================
# Notification Channels
# ============================================================================

class NotificationChannel(str, Enum):
    """Supported notification channels."""

    IN_APP = "IN_APP"
    EMAIL = "EMAIL"
    WEBHOOK = "WEBHOOK"
    SLACK = "SLACK"
    TEAMS = "TEAMS"


# ============================================================================
# Notification Status
# ============================================================================

class NotificationStatus(str, Enum):
    """Notification delivery states."""

    PENDING = "PENDING"
    SENT = "SENT"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ============================================================================
# Notification Priority
# ============================================================================

class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================================
# Notification Record
# ============================================================================

@dataclass
class NotificationRecord:
    """
    Represents one notification delivery attempt.
    """

    notification_id: str

    alert_id: Optional[str]

    company_id: Optional[str]

    channel: str

    recipient: Optional[str]

    subject: Optional[str]

    message: str

    priority: str

    status: str

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    sent_at: Optional[datetime] = None

    failed_at: Optional[datetime] = None

    retry_count: int = 0

    error: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert notification record to dictionary."""

        data = asdict(self)

        for field_name in (
            "created_at",
            "sent_at",
            "failed_at",
        ):
            value = data.get(field_name)

            if isinstance(
                value,
                datetime,
            ):
                data[field_name] = value.isoformat()

        return data


# ============================================================================
# Notification Service
# ============================================================================

class NotificationService:
    """
    Central notification service for FinCo AI.

    This class intentionally keeps external integrations behind small
    adapter methods. Real providers can be connected later without
    changing the Alert Service.
    """

    def __init__(
        self,
        notification_repository: Optional[Any] = None,
        email_provider: Optional[Any] = None,
        webhook_provider: Optional[Any] = None,
        messaging_provider: Optional[Any] = None,
        default_email_recipient: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:

        self.notification_repository = (
            notification_repository
        )

        self.email_provider = email_provider

        self.webhook_provider = webhook_provider

        self.messaging_provider = messaging_provider

        self.default_email_recipient = (
            default_email_recipient
        )

        self.max_retries = max(
            0,
            int(max_retries),
        )

        self.retry_delay_seconds = max(
            0.0,
            float(retry_delay_seconds),
        )

    # ========================================================================
    # ALERT NOTIFICATION
    # ========================================================================

    def send_alert(
        self,
        alert: Any,
        channels: Optional[List[str]] = None,
        recipients: Optional[Dict[str, str]] = None,
    ) -> List[NotificationRecord]:
        """
        Send an alert through configured notification channels.

        Example:

            service.send_alert(
                alert,
                channels=[
                    "IN_APP",
                    "EMAIL",
                    "SLACK",
                ],
                recipients={
                    "EMAIL": "finance@example.com",
                    "SLACK": "#finance-alerts",
                },
            )
        """

        channels = channels or self.default_channels(
            alert
        )

        results: List[NotificationRecord] = []

        for channel in channels:

            recipient = None

            if recipients:
                recipient = recipients.get(
                    str(channel)
                )

            result = self.send(
                alert=alert,
                channel=channel,
                recipient=recipient,
            )

            results.append(result)

        return results

    # ========================================================================
    # ESCALATION NOTIFICATION
    # ========================================================================

    def send_escalation(
        self,
        escalation: Any,
        channels: Optional[List[str]] = None,
        recipients: Optional[Dict[str, str]] = None,
    ) -> List[NotificationRecord]:
        """
        Send an escalation notification.

        Critical escalations default to:
        - IN_APP
        - EMAIL
        - SLACK

        Lower-level escalations use fewer channels.
        """

        channels = channels or self.escalation_channels(
            escalation
        )

        results: List[NotificationRecord] = []

        for channel in channels:

            recipient = None

            if recipients:
                recipient = recipients.get(
                    str(channel)
                )

            result = self.send(
                alert=escalation,
                channel=channel,
                recipient=recipient,
                notification_type="ESCALATION",
            )

            results.append(result)

        return results

    # ========================================================================
    # GENERIC SEND
    # ========================================================================

    def send(
        self,
        alert: Any,
        channel: str,
        recipient: Optional[str] = None,
        notification_type: str = "ALERT",
    ) -> NotificationRecord:
        """
        Send a notification through one channel.
        """

        channel = self.normalize_channel(
            channel
        )

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
                "MEDIUM",
            )
        ).upper()

        priority = self.priority_from_severity(
            severity
        )

        subject = self.build_subject(
            alert,
            notification_type,
        )

        message = self.build_message(
            alert,
            notification_type,
        )

        record = NotificationRecord(
            notification_id=self._generate_id(),
            alert_id=alert_id,
            company_id=company_id,
            channel=channel,
            recipient=recipient,
            subject=subject,
            message=message,
            priority=priority,
            status=NotificationStatus.PENDING.value,
            metadata={
                "notification_type":
                    notification_type,
                "severity":
                    severity,
            },
        )

        # --------------------------------------------------------------
        # Duplicate protection
        # --------------------------------------------------------------

        if self.is_duplicate(
            record
        ):
            record.status = (
                NotificationStatus.SKIPPED.value
            )

            self._save(record)

            return record

        # --------------------------------------------------------------
        # Persist pending notification
        # --------------------------------------------------------------

        self._save(record)

        # --------------------------------------------------------------
        # Deliver with retry
        # --------------------------------------------------------------

        try:

            self._deliver_with_retry(
                record
            )

            record.status = (
                NotificationStatus.SENT.value
            )

            record.sent_at = datetime.now(
                timezone.utc
            )

            self._update(record)

            return record

        except Exception as exc:

            record.status = (
                NotificationStatus.FAILED.value
            )

            record.failed_at = datetime.now(
                timezone.utc
            )

            record.error = str(exc)

            self._update(record)

            logger.exception(
                "Notification delivery failed: %s",
                record.notification_id,
            )

            raise NotificationDeliveryError(
                f"Failed to deliver notification "
                f"{record.notification_id}: {exc}"
            ) from exc

    # ========================================================================
    # DEFAULT CHANNELS
    # ========================================================================

    def default_channels(
        self,
        alert: Any,
    ) -> List[str]:
        """
        Determine channels based on alert severity.
        """

        severity = str(
            self._get_value(
                alert,
                "severity",
                "LOW",
            )
        ).upper()

        requires_review = bool(
            self._get_value(
                alert,
                "requires_human_review",
                False,
            )
        )

        if severity == "CRITICAL":
            return [
                NotificationChannel.IN_APP.value,
                NotificationChannel.EMAIL.value,
                NotificationChannel.SLACK.value,
            ]

        if severity == "HIGH" or requires_review:
            return [
                NotificationChannel.IN_APP.value,
                NotificationChannel.EMAIL.value,
            ]

        if severity == "MEDIUM":
            return [
                NotificationChannel.IN_APP.value,
            ]

        return [
            NotificationChannel.IN_APP.value,
        ]

    # ========================================================================
    # ESCALATION CHANNELS
    # ========================================================================

    def escalation_channels(
        self,
        escalation: Any,
    ) -> List[str]:
        """
        Determine notification channels for escalation.
        """

        level = str(
            self._get_value(
                escalation,
                "level",
                "LEVEL_1",
            )
        ).upper()

        if level in (
            "LEVEL_3",
            "LEVEL_4",
        ):
            return [
                NotificationChannel.IN_APP.value,
                NotificationChannel.EMAIL.value,
                NotificationChannel.SLACK.value,
            ]

        if level == "LEVEL_2":
            return [
                NotificationChannel.IN_APP.value,
                NotificationChannel.EMAIL.value,
            ]

        return [
            NotificationChannel.IN_APP.value,
        ]

    # ========================================================================
    # DELIVERY
    # ========================================================================

    def _deliver_with_retry(
        self,
        record: NotificationRecord,
    ) -> None:
        """
        Attempt delivery multiple times.
        """

        last_error: Optional[Exception] = None

        attempts = self.max_retries + 1

        for attempt in range(
            attempts
        ):

            record.retry_count = attempt

            try:

                self._deliver(
                    record
                )

                return

            except Exception as exc:

                last_error = exc

                logger.warning(
                    "Notification attempt %s/%s failed "
                    "for %s: %s",
                    attempt + 1,
                    attempts,
                    record.notification_id,
                    exc,
                )

                if attempt < self.max_retries:

                    if self.retry_delay_seconds > 0:
                        time.sleep(
                            self.retry_delay_seconds
                        )

        if last_error:
            raise last_error

        raise NotificationError(
            "Notification delivery failed."
        )

    # ========================================================================
    # CHANNEL DELIVERY
    # ========================================================================

    def _deliver(
        self,
        record: NotificationRecord,
    ) -> None:
        """
        Route notification to the appropriate adapter.
        """

        channel = record.channel

        if channel == NotificationChannel.IN_APP.value:
            self._deliver_in_app(
                record
            )
            return

        if channel == NotificationChannel.EMAIL.value:
            self._deliver_email(
                record
            )
            return

        if channel == NotificationChannel.WEBHOOK.value:
            self._deliver_webhook(
                record
            )
            return

        if channel in (
            NotificationChannel.SLACK.value,
            NotificationChannel.TEAMS.value,
        ):
            self._deliver_messaging(
                record
            )
            return

        raise NotificationError(
            f"Unsupported notification channel: "
            f"{channel}"
        )

    # ========================================================================
    # IN-APP
    # ========================================================================

    def _deliver_in_app(
        self,
        record: NotificationRecord,
    ) -> None:
        """
        Store an in-app notification.

        The frontend can retrieve these notifications from
        the notification repository/API.
        """

        if self.notification_repository is None:
            logger.info(
                "In-app notification: %s",
                record.message,
            )
            return

        method = getattr(
            self.notification_repository,
            "create",
            None,
        )

        if method is None:
            logger.info(
                "In-app notification: %s",
                record.message,
            )
            return

        method(
            record.to_dict()
        )

    # ========================================================================
    # EMAIL
    # ========================================================================

    def _deliver_email(
        self,
        record: NotificationRecord,
    ) -> None:
        """
        Deliver an email using the configured email adapter.
        """

        recipient = (
            record.recipient
            or self.default_email_recipient
        )

        if not recipient:
            raise NotificationError(
                "No email recipient configured."
            )

        if self.email_provider is None:
            logger.info(
                "Email notification prepared for %s: %s",
                recipient,
                record.subject,
            )
            return

        method = getattr(
            self.email_provider,
            "send",
            None,
        )

        if method is None:
            raise NotificationError(
                "Email provider does not implement send()."
            )

        method(
            recipient=recipient,
            subject=record.subject,
            message=record.message,
        )

    # ========================================================================
    # WEBHOOK
    # ========================================================================

    def _deliver_webhook(
        self,
        record: NotificationRecord,
    ) -> None:
        """
        Deliver notification through a generic webhook.
        """

        if not record.recipient:
            raise NotificationError(
                "Webhook URL is required."
            )

        if self.webhook_provider is None:
            logger.info(
                "Webhook notification prepared for %s",
                record.recipient,
            )
            return

        method = getattr(
            self.webhook_provider,
            "send",
            None,
        )

        if method is None:
            raise NotificationError(
                "Webhook provider does not implement send()."
            )

        method(
            url=record.recipient,
            payload={
                "notification_id":
                    record.notification_id,
                "alert_id":
                    record.alert_id,
                "company_id":
                    record.company_id,
                "priority":
                    record.priority,
                "subject":
                    record.subject,
                "message":
                    record.message,
                "metadata":
                    record.metadata,
            },
        )

    # ========================================================================
    # SLACK / TEAMS
    # ========================================================================

    def _deliver_messaging(
        self,
        record: NotificationRecord,
    ) -> None:
        """
        Deliver Slack/Teams notification through a messaging adapter.
        """

        if self.messaging_provider is None:
            logger.info(
                "%s notification prepared: %s",
                record.channel,
                record.message,
            )
            return

        method = getattr(
            self.messaging_provider,
            "send",
            None,
        )

        if method is None:
            raise NotificationError(
                "Messaging provider does not implement send()."
            )

        method(
            channel=record.recipient,
            message=record.message,
            priority=record.priority,
            metadata=record.metadata,
        )

    # ========================================================================
    # MESSAGE BUILDING
    # ========================================================================

    def build_subject(
        self,
        alert: Any,
        notification_type: str = "ALERT",
    ) -> str:
        """
        Generate notification subject.
        """

        severity = str(
            self._get_value(
                alert,
                "severity",
                "MEDIUM",
            )
        ).upper()

        title = self._get_value(
            alert,
            "title",
            "Financial Alert",
        )

        if notification_type == "ESCALATION":
            return (
                f"[{severity}] FinCo AI Escalation - "
                f"{title}"
            )

        return (
            f"[{severity}] FinCo AI Alert - "
            f"{title}"
        )

    def build_message(
        self,
        alert: Any,
        notification_type: str = "ALERT",
    ) -> str:
        """
        Generate a concise notification message.

        Detailed investigation remains the responsibility
        of the Supervisor Agent.
        """

        alert_id = self._get_value(
            alert,
            "alert_id",
            "N/A",
        )

        company_id = self._get_value(
            alert,
            "company_id",
            "N/A",
        )

        alert_type = self._get_value(
            alert,
            "alert_type",
            "FINANCIAL_ALERT",
        )

        severity = self._get_value(
            alert,
            "severity",
            "MEDIUM",
        )

        risk_score = self._number(
            self._get_value(
                alert,
                "risk_score",
                0,
            )
        )

        title = self._get_value(
            alert,
            "title",
            "Financial Alert",
        )

        message = self._get_value(
            alert,
            "message",
            "",
        )

        explanation = self._get_value(
            alert,
            "explanation",
            None,
        )

        lines = [
            f"FinCo AI - {notification_type}",
            "",
            f"Title: {title}",
            f"Alert Type: {alert_type}",
            f"Severity: {severity}",
            f"Risk Score: {risk_score:.1f}/100",
            f"Company: {company_id}",
            f"Alert ID: {alert_id}",
        ]

        if message:
            lines.extend(
                [
                    "",
                    f"Message: {message}",
                ]
            )

        if explanation:
            lines.extend(
                [
                    "",
                    f"Explanation: {explanation}",
                ]
            )

        return "\n".join(
            lines
        )

    # ========================================================================
    # PRIORITY
    # ========================================================================

    @staticmethod
    def priority_from_severity(
        severity: str,
    ) -> str:
        """
        Convert alert severity into notification priority.
        """

        severity = str(
            severity
        ).upper()

        mapping = {
            "CRITICAL":
                NotificationPriority.CRITICAL.value,

            "HIGH":
                NotificationPriority.HIGH.value,

            "MEDIUM":
                NotificationPriority.MEDIUM.value,

            "LOW":
                NotificationPriority.LOW.value,
        }

        return mapping.get(
            severity,
            NotificationPriority.MEDIUM.value,
        )

    # ========================================================================
    # DUPLICATE DETECTION
    # ========================================================================

    def is_duplicate(
        self,
        record: NotificationRecord,
    ) -> bool:
        """
        Check whether the same notification has already been delivered.
        """

        if self.notification_repository is None:
            return False

        method = getattr(
            self.notification_repository,
            "find_recent_duplicate",
            None,
        )

        if method is None:
            return False

        result = method(
            alert_id=record.alert_id,
            channel=record.channel,
            recipient=record.recipient,
        )

        return result is not None

    # ========================================================================
    # REPOSITORY
    # ========================================================================

    def _save(
        self,
        record: NotificationRecord,
    ) -> None:
        """
        Save a notification record.
        """

        if self.notification_repository is None:
            return

        method = getattr(
            self.notification_repository,
            "create",
            None,
        )

        if method is None:
            return

        try:
            method(
                record.to_dict()
            )

        except Exception as exc:

            logger.warning(
                "Unable to persist notification %s: %s",
                record.notification_id,
                exc,
            )

    def _update(
        self,
        record: NotificationRecord,
    ) -> None:
        """
        Update notification delivery status.
        """

        if self.notification_repository is None:
            return

        method = getattr(
            self.notification_repository,
            "update",
            None,
        )

        if method is None:
            return

        try:

            method(
                record.notification_id,
                record.to_dict(),
            )

        except Exception as exc:

            logger.warning(
                "Unable to update notification %s: %s",
                record.notification_id,
                exc,
            )

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def normalize_channel(
        channel: Any,
    ) -> str:
        """
        Normalize Enum/string channel values.
        """

        if isinstance(
            channel,
            NotificationChannel,
        ):
            return channel.value

        return str(
            channel
        ).upper()

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
            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

    @staticmethod
    def _generate_id() -> str:
        return (
            f"NTF-{uuid4().hex[:12].upper()}"
        )


# ============================================================================
# Convenience Function
# ============================================================================

def send_alert_notification(
    alert: Any,
    channel: str = NotificationChannel.IN_APP.value,
    recipient: Optional[str] = None,
) -> NotificationRecord:
    """
    Convenience helper for sending one notification.
    """

    service = NotificationService()

    return service.send(
        alert=alert,
        channel=channel,
        recipient=recipient,
    )


__all__ = [
    "NotificationError",
    "NotificationDeliveryError",
    "NotificationChannel",
    "NotificationStatus",
    "NotificationPriority",
    "NotificationRecord",
    "NotificationService",
    "send_alert_notification",
]