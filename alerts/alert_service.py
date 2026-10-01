"""
FinCo AI - Alert Service

Application/service layer for financial alerts.

Responsibilities:
- Evaluate alert conditions
- Calculate risk/severity
- Generate standardized alerts
- Prevent duplicate alerts
- Persist alerts through AlertRepository
- Trigger notifications
- Trigger escalation for critical alerts
- Support alert acknowledgement/resolution
- Maintain an audit-friendly workflow

Architecture:

    Financial Analysis
            ↓
        Alert Rules
            ↓
       Risk Scoring
            ↓
       Alert Service
       ┌────┼─────┐
       ↓    ↓     ↓
    Generate Save Notify
       │    │     │
       └────┼─────┘
            ↓
       Human Review
            ↓
        Resolution
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.alerts.alert_generator import (
    AlertEvidence,
    AlertGenerator,
    AlertRecommendation,
    GeneratedAlert,
)


class AlertServiceError(Exception):
    """Base exception for alert service failures."""


class AlertNotFoundError(AlertServiceError):
    """Raised when an alert cannot be found."""


class AlertService:
    """
    Coordinates the complete FinCo AI alert lifecycle.

    Dependencies are injected so this service remains easy to:
    - test
    - replace repositories
    - replace notification providers
    - integrate with FastAPI
    """

    def __init__(
        self,
        alert_repository: Any,
        alert_generator: Optional[AlertGenerator] = None,
        notification_service: Optional[Any] = None,
        escalation_service: Optional[Any] = None,
        audit_service: Optional[Any] = None,
    ) -> None:

        self.alert_repository = alert_repository

        self.alert_generator = (
            alert_generator
            or AlertGenerator()
        )

        self.notification_service = notification_service
        self.escalation_service = escalation_service
        self.audit_service = audit_service

    # ======================================================================
    # CREATE ALERT
    # ======================================================================

    def create_alert(
        self,
        company_id: str,
        alert_type: str,
        severity: str,
        risk_score: float,
        metrics: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[AlertEvidence]] = None,
        recommendations: Optional[
            List[AlertRecommendation]
        ] = None,
        metadata: Optional[Dict[str, Any]] = None,
        notify: bool = True,
        escalate: bool = True,
    ) -> GeneratedAlert:
        """
        Create, persist and optionally distribute an alert.
        """

        metrics = metrics or {}
        evidence = evidence or []
        metadata = metadata or {}

        # --------------------------------------------------------------
        # Duplicate protection
        # --------------------------------------------------------------

        existing = self._find_duplicate_alert(
            company_id=company_id,
            alert_type=alert_type,
            metadata=metadata,
        )

        if existing is not None:
            return self._convert_repository_alert(
                existing
            )

        # --------------------------------------------------------------
        # Generate alert
        # --------------------------------------------------------------

        alert = self.alert_generator.generate_alert(
            company_id=company_id,
            alert_type=alert_type,
            severity=severity,
            risk_score=risk_score,
            metrics=metrics,
            evidence=evidence,
            recommendations=recommendations,
            metadata=metadata,
        )

        # --------------------------------------------------------------
        # Persist
        # --------------------------------------------------------------

        saved_alert = self._save_alert(alert)

        # --------------------------------------------------------------
        # Audit
        # --------------------------------------------------------------

        self._audit(
            action="ALERT_CREATED",
            company_id=company_id,
            alert=saved_alert,
        )

        # --------------------------------------------------------------
        # Notification
        # --------------------------------------------------------------

        if notify:
            self._notify(saved_alert)

        # --------------------------------------------------------------
        # Escalation
        # --------------------------------------------------------------

        if escalate:
            self._escalate_if_required(
                saved_alert
            )

        return saved_alert

    # ======================================================================
    # BULK CREATE
    # ======================================================================

    def create_alerts(
        self,
        alerts: List[Dict[str, Any]],
        notify: bool = True,
        escalate: bool = True,
    ) -> List[GeneratedAlert]:
        """
        Create multiple alerts.

        Useful after processing a financial statement or daily
        financial monitoring batch.
        """

        results: List[GeneratedAlert] = []

        for alert_data in alerts:

            alert = self.create_alert(
                company_id=alert_data["company_id"],
                alert_type=alert_data["alert_type"],
                severity=alert_data["severity"],
                risk_score=alert_data["risk_score"],
                metrics=alert_data.get("metrics"),
                evidence=alert_data.get("evidence"),
                recommendations=alert_data.get(
                    "recommendations"
                ),
                metadata=alert_data.get("metadata"),
                notify=notify,
                escalate=escalate,
            )

            results.append(alert)

        return results

    # ======================================================================
    # UPDATE ALERT
    # ======================================================================

    def update_alert(
        self,
        alert_id: str,
        updates: Dict[str, Any],
    ) -> GeneratedAlert:
        """
        Update an existing alert.

        Intended for controlled state changes such as:
        - severity
        - status
        - risk score
        - explanation
        - metadata
        """

        existing = self._get_alert(alert_id)

        if existing is None:
            raise AlertNotFoundError(
                f"Alert '{alert_id}' was not found."
            )

        updates = dict(updates)

        updates["updated_at"] = (
            datetime.now(timezone.utc)
        )

        updated = self.alert_repository.update(
            alert_id,
            updates,
        )

        result = self._convert_repository_alert(
            updated
        )

        self._audit(
            action="ALERT_UPDATED",
            company_id=result.company_id,
            alert=result,
            metadata={
                "updated_fields": list(
                    updates.keys()
                )
            },
        )

        return result

    # ======================================================================
    # ACKNOWLEDGE
    # ======================================================================

    def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
        comment: Optional[str] = None,
    ) -> GeneratedAlert:
        """
        Mark an alert as acknowledged by a human reviewer.
        """

        alert = self._get_alert(alert_id)

        if alert is None:
            raise AlertNotFoundError(
                f"Alert '{alert_id}' was not found."
            )

        now = datetime.now(timezone.utc)

        updates = {
            "status": "ACKNOWLEDGED",
            "acknowledged_by": user_id,
            "acknowledged_at": now,
        }

        if comment:
            updates["review_comment"] = comment

        updated = self.alert_repository.update(
            alert_id,
            updates,
        )

        result = self._convert_repository_alert(
            updated
        )

        self._audit(
            action="ALERT_ACKNOWLEDGED",
            company_id=result.company_id,
            alert=result,
            metadata={
                "user_id": user_id,
                "comment": comment,
            },
        )

        return result

    # ======================================================================
    # RESOLVE
    # ======================================================================

    def resolve_alert(
        self,
        alert_id: str,
        user_id: str,
        resolution: Optional[str] = None,
    ) -> GeneratedAlert:
        """
        Resolve an alert after investigation/corrective action.
        """

        alert = self._get_alert(alert_id)

        if alert is None:
            raise AlertNotFoundError(
                f"Alert '{alert_id}' was not found."
            )

        now = datetime.now(timezone.utc)

        updates = {
            "status": "RESOLVED",
            "resolved_by": user_id,
            "resolved_at": now,
        }

        if resolution:
            updates["resolution"] = resolution

        updated = self.alert_repository.update(
            alert_id,
            updates,
        )

        result = self._convert_repository_alert(
            updated
        )

        self._audit(
            action="ALERT_RESOLVED",
            company_id=result.company_id,
            alert=result,
            metadata={
                "user_id": user_id,
                "resolution": resolution,
            },
        )

        return result

    # ======================================================================
    # REOPEN
    # ======================================================================

    def reopen_alert(
        self,
        alert_id: str,
        user_id: str,
        reason: Optional[str] = None,
    ) -> GeneratedAlert:
        """
        Reopen a previously resolved alert.
        """

        alert = self._get_alert(alert_id)

        if alert is None:
            raise AlertNotFoundError(
                f"Alert '{alert_id}' was not found."
            )

        updates = {
            "status": "OPEN",
            "reopened_by": user_id,
            "reopened_at": datetime.now(
                timezone.utc
            ),
        }

        if reason:
            updates["reopen_reason"] = reason

        updated = self.alert_repository.update(
            alert_id,
            updates,
        )

        result = self._convert_repository_alert(
            updated
        )

        self._audit(
            action="ALERT_REOPENED",
            company_id=result.company_id,
            alert=result,
            metadata={
                "user_id": user_id,
                "reason": reason,
            },
        )

        return result

    # ======================================================================
    # QUERY ALERTS
    # ======================================================================

    def get_alert(
        self,
        alert_id: str,
    ) -> GeneratedAlert:
        """
        Retrieve one alert.
        """

        alert = self._get_alert(alert_id)

        if alert is None:
            raise AlertNotFoundError(
                f"Alert '{alert_id}' was not found."
            )

        return self._convert_repository_alert(
            alert
        )

    def get_company_alerts(
        self,
        company_id: str,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[GeneratedAlert]:
        """
        Retrieve alerts for a company.
        """

        if hasattr(
            self.alert_repository,
            "get_by_company",
        ):
            alerts = (
                self.alert_repository.get_by_company(
                    company_id=company_id,
                    status=status,
                    severity=severity,
                    alert_type=alert_type,
                    limit=limit,
                )
            )
        else:
            alerts = []

        return [
            self._convert_repository_alert(alert)
            for alert in alerts
        ]

    def get_open_alerts(
        self,
        company_id: str,
    ) -> List[GeneratedAlert]:
        """
        Retrieve unresolved alerts.
        """

        return self.get_company_alerts(
            company_id=company_id,
            status="OPEN",
        )

    # ======================================================================
    # STATISTICS
    # ======================================================================

    def get_alert_summary(
        self,
        company_id: str,
    ) -> Dict[str, Any]:
        """
        Generate a high-level alert summary for dashboards.
        """

        alerts = self.get_company_alerts(
            company_id=company_id
        )

        summary = {
            "company_id": company_id,
            "total": len(alerts),
            "open": 0,
            "acknowledged": 0,
            "resolved": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "average_risk_score": 0.0,
        }

        if not alerts:
            return summary

        total_risk = 0.0

        for alert in alerts:

            status = str(
                alert.status
            ).upper()

            severity = str(
                alert.severity
            ).upper()

            if status == "OPEN":
                summary["open"] += 1

            elif status == "ACKNOWLEDGED":
                summary["acknowledged"] += 1

            elif status == "RESOLVED":
                summary["resolved"] += 1

            if severity == "CRITICAL":
                summary["critical"] += 1

            elif severity == "HIGH":
                summary["high"] += 1

            elif severity == "MEDIUM":
                summary["medium"] += 1

            elif severity == "LOW":
                summary["low"] += 1

            total_risk += float(
                alert.risk_score
            )

        summary["average_risk_score"] = round(
            total_risk / len(alerts),
            2,
        )

        return summary

    # ======================================================================
    # DUPLICATE DETECTION
    # ======================================================================

    def _find_duplicate_alert(
        self,
        company_id: str,
        alert_type: str,
        metadata: Dict[str, Any],
    ) -> Optional[Any]:
        """
        Prevent repeated alerts for the same active condition.

        Repository implementations can provide more sophisticated
        deduplication using database constraints.
        """

        if not hasattr(
            self.alert_repository,
            "find_active_duplicate",
        ):
            return None

        period = metadata.get(
            "period"
        )

        metric = metadata.get(
            "metric"
        )

        return (
            self.alert_repository.find_active_duplicate(
                company_id=company_id,
                alert_type=alert_type,
                period=period,
                metric=metric,
            )
        )

    # ======================================================================
    # PERSISTENCE
    # ======================================================================

    def _save_alert(
        self,
        alert: GeneratedAlert,
    ) -> GeneratedAlert:
        """
        Persist an alert through the repository.
        """

        try:

            payload = alert.to_dict()

            saved = self.alert_repository.create(
                payload
            )

            return self._convert_repository_alert(
                saved
            )

        except Exception as exc:

            raise AlertServiceError(
                f"Failed to persist alert "
                f"{alert.alert_id}: {exc}"
            ) from exc

    def _get_alert(
        self,
        alert_id: str,
    ) -> Optional[Any]:

        if hasattr(
            self.alert_repository,
            "get_by_id",
        ):
            return self.alert_repository.get_by_id(
                alert_id
            )

        return None

    # ======================================================================
    # NOTIFICATIONS
    # ======================================================================

    def _notify(
        self,
        alert: GeneratedAlert,
    ) -> None:
        """
        Send alert notification.

        Notification implementation can later support:
        - email
        - Slack
        - Teams
        - in-app notifications
        - webhook
        """

        if self.notification_service is None:
            return

        try:

            if hasattr(
                self.notification_service,
                "send_alert",
            ):
                self.notification_service.send_alert(
                    alert
                )

        except Exception as exc:

            # Notification failure should not invalidate
            # an already persisted financial alert.
            self._audit(
                action="ALERT_NOTIFICATION_FAILED",
                company_id=alert.company_id,
                alert=alert,
                metadata={
                    "error": str(exc)
                },
            )

    # ======================================================================
    # ESCALATION
    # ======================================================================

    def _escalate_if_required(
        self,
        alert: GeneratedAlert,
    ) -> None:
        """
        Escalate high-risk alerts.

        Critical alerts and alerts explicitly marked for
        human review are escalated.
        """

        should_escalate = (
            alert.requires_human_review
            or alert.severity.upper()
            in {"HIGH", "CRITICAL"}
        )

        if not should_escalate:
            return

        if self.escalation_service is None:
            return

        try:

            if hasattr(
                self.escalation_service,
                "escalate",
            ):
                self.escalation_service.escalate(
                    alert
                )

            self._audit(
                action="ALERT_ESCALATED",
                company_id=alert.company_id,
                alert=alert,
            )

        except Exception as exc:

            self._audit(
                action="ALERT_ESCALATION_FAILED",
                company_id=alert.company_id,
                alert=alert,
                metadata={
                    "error": str(exc)
                },
            )

    # ======================================================================
    # AUDIT
    # ======================================================================

    def _audit(
        self,
        action: str,
        company_id: str,
        alert: GeneratedAlert,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write an audit event.

        Audit failures should not normally break the
        financial alert workflow.
        """

        if self.audit_service is None:
            return

        try:

            payload = {
                "action": action,
                "company_id": company_id,
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "risk_score": alert.risk_score,
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "metadata": metadata or {},
            }

            if hasattr(
                self.audit_service,
                "record",
            ):
                self.audit_service.record(
                    payload
                )

        except Exception:
            # Audit logging must not prevent the core
            # alert workflow from completing.
            pass

    # ======================================================================
    # CONVERSION
    # ======================================================================

    def _convert_repository_alert(
        self,
        alert: Any,
    ) -> GeneratedAlert:
        """
        Convert repository/database representation into
        the application-level GeneratedAlert object.
        """

        if isinstance(
            alert,
            GeneratedAlert,
        ):
            return alert

        if isinstance(
            alert,
            dict,
        ):
            data = dict(alert)

            evidence = [
                self._convert_evidence(item)
                for item in data.pop(
                    "evidence",
                    [],
                )
            ]

            recommendations = [
                self._convert_recommendation(item)
                for item in data.pop(
                    "recommendations",
                    [],
                )
            ]

            created_at = data.get(
                "created_at"
            )

            if isinstance(
                created_at,
                str,
            ):
                try:
                    data["created_at"] = (
                        datetime.fromisoformat(
                            created_at
                        )
                    )
                except ValueError:
                    data["created_at"] = (
                        datetime.now(
                            timezone.utc
                        )
                    )

            return GeneratedAlert(
                evidence=evidence,
                recommendations=recommendations,
                **data,
            )

        # SQLAlchemy-style object
        return GeneratedAlert(
            alert_id=alert.alert_id,
            company_id=alert.company_id,
            alert_type=alert.alert_type,
            severity=alert.severity,
            title=alert.title,
            message=alert.message,
            risk_score=float(
                alert.risk_score
            ),
            explanation=getattr(
                alert,
                "explanation",
                None,
            ),
            evidence=self._convert_evidence_list(
                getattr(
                    alert,
                    "evidence",
                    [],
                )
            ),
            recommendations=(
                self._convert_recommendation_list(
                    getattr(
                        alert,
                        "recommendations",
                        [],
                    )
                )
            ),
            source=getattr(
                alert,
                "source",
                "financial_analysis",
            ),
            status=getattr(
                alert,
                "status",
                "OPEN",
            ),
            requires_human_review=bool(
                getattr(
                    alert,
                    "requires_human_review",
                    False,
                )
            ),
            created_at=getattr(
                alert,
                "created_at",
                datetime.now(
                    timezone.utc
                ),
            ),
            metadata=getattr(
                alert,
                "metadata",
                {},
            ) or {},
        )

    # ======================================================================
    # CONVERSION HELPERS
    # ======================================================================

    @staticmethod
    def _convert_evidence(
        evidence: Any,
    ) -> AlertEvidence:

        if isinstance(
            evidence,
            AlertEvidence,
        ):
            return evidence

        if isinstance(
            evidence,
            dict,
        ):
            return AlertEvidence(
                **evidence
            )

        return AlertEvidence(
            metric=getattr(
                evidence,
                "metric",
                "unknown",
            ),
            value=getattr(
                evidence,
                "value",
                None,
            ),
            threshold=getattr(
                evidence,
                "threshold",
                None,
            ),
            comparison=getattr(
                evidence,
                "comparison",
                None,
            ),
            period=getattr(
                evidence,
                "period",
                None,
            ),
            source=getattr(
                evidence,
                "source",
                None,
            ),
            details=getattr(
                evidence,
                "details",
                None,
            ),
        )

    def _convert_evidence_list(
        self,
        evidence: Any,
    ) -> List[AlertEvidence]:

        if not evidence:
            return []

        return [
            self._convert_evidence(item)
            for item in evidence
        ]

    @staticmethod
    def _convert_recommendation(
        recommendation: Any,
    ) -> AlertRecommendation:

        if isinstance(
            recommendation,
            AlertRecommendation,
        ):
            return recommendation

        if isinstance(
            recommendation,
            dict,
        ):
            return AlertRecommendation(
                **recommendation
            )

        return AlertRecommendation(
            action=getattr(
                recommendation,
                "action",
                "Investigate financial risk",
            ),
            priority=getattr(
                recommendation,
                "priority",
                "MEDIUM",
            ),
            reason=getattr(
                recommendation,
                "reason",
                None,
            ),
            expected_impact=getattr(
                recommendation,
                "expected_impact",
                None,
            ),
        )

    def _convert_recommendation_list(
        self,
        recommendations: Any,
    ) -> List[AlertRecommendation]:

        if not recommendations:
            return []

        return [
            self._convert_recommendation(item)
            for item in recommendations
        ]


__all__ = [
    "AlertService",
    "AlertServiceError",
    "AlertNotFoundError",
]