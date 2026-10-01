"""
FinCo AI - Alert Repository

Database repository for:
- Financial alerts
- Risk alerts
- Fraud alerts
- Alert lifecycle management
- Severity filtering
- Company-specific alert retrieval
- Unresolved/critical alert queries
- Alert dashboard statistics

Architecture:

API
 ↓
Alert Service
 ↓
Alert Repository
 ↓
SQLAlchemy Session
 ↓
Database

The repository intentionally contains database-access logic only.
Business rules should remain in the alert service / alert engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import and_, case, desc, func, or_
from sqlalchemy.orm import Session

from app.database.models.fraud_alert import FraudAlert
from app.database.models.risk_alert import RiskAlert


class AlertRepository:
    """
    Repository for financial/risk/fraud alerts.

    Notes:
    - Uses SQLAlchemy ORM.
    - Keeps query logic centralized.
    - Does not contain AI/business decision logic.
    - Supports both RiskAlert and FraudAlert models.
    """

    # ========================================================
    # RISK ALERTS
    # ========================================================

    def create_risk_alert(
        self,
        db: Session,
        alert: RiskAlert,
    ) -> RiskAlert:
        """
        Persist a new risk alert.
        """

        db.add(alert)
        db.flush()
        db.refresh(alert)

        return alert

    def get_risk_alert_by_id(
        self,
        db: Session,
        alert_id: int,
    ) -> Optional[RiskAlert]:
        """
        Retrieve a risk alert by primary key.
        """

        return (
            db.query(RiskAlert)
            .filter(RiskAlert.id == alert_id)
            .first()
        )

    def get_risk_alerts_by_company(
        self,
        db: Session,
        company_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[RiskAlert]:
        """
        Retrieve risk alerts for a company.
        """

        return (
            db.query(RiskAlert)
            .filter(RiskAlert.company_id == company_id)
            .order_by(desc(RiskAlert.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_unresolved_risk_alerts(
        self,
        db: Session,
        company_id: int,
        *,
        limit: int = 100,
    ) -> List[RiskAlert]:
        """
        Retrieve unresolved risk alerts.

        The query intentionally uses common lifecycle fields.
        If your model uses a different status field, update this
        method to match the model definition.
        """

        query = db.query(RiskAlert).filter(
            RiskAlert.company_id == company_id
        )

        if hasattr(RiskAlert, "resolved"):
            query = query.filter(
                RiskAlert.resolved.is_(False)
            )

        elif hasattr(RiskAlert, "status"):
            query = query.filter(
                RiskAlert.status.notin_(
                    ["resolved", "closed"]
                )
            )

        return (
            query
            .order_by(desc(RiskAlert.created_at))
            .limit(limit)
            .all()
        )

    def get_critical_risk_alerts(
        self,
        db: Session,
        company_id: int,
        *,
        limit: int = 50,
    ) -> List[RiskAlert]:
        """
        Retrieve high-priority risk alerts.
        """

        query = db.query(RiskAlert).filter(
            RiskAlert.company_id == company_id
        )

        if hasattr(RiskAlert, "severity"):
            query = query.filter(
                RiskAlert.severity.in_(
                    ["critical", "high"]
                )
            )

        return (
            query
            .order_by(desc(RiskAlert.created_at))
            .limit(limit)
            .all()
        )

    # ========================================================
    # FRAUD ALERTS
    # ========================================================

    def create_fraud_alert(
        self,
        db: Session,
        alert: FraudAlert,
    ) -> FraudAlert:
        """
        Persist a new fraud alert.
        """

        db.add(alert)
        db.flush()
        db.refresh(alert)

        return alert

    def get_fraud_alert_by_id(
        self,
        db: Session,
        alert_id: int,
    ) -> Optional[FraudAlert]:
        """
        Retrieve a fraud alert by ID.
        """

        return (
            db.query(FraudAlert)
            .filter(FraudAlert.id == alert_id)
            .first()
        )

    def get_fraud_alerts_by_company(
        self,
        db: Session,
        company_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FraudAlert]:
        """
        Retrieve fraud alerts belonging to a company.
        """

        return (
            db.query(FraudAlert)
            .filter(
                FraudAlert.company_id == company_id
            )
            .order_by(desc(FraudAlert.created_at))
            .offset(offset)
            .limit(limit)
            .all()
        )

    def get_unresolved_fraud_alerts(
        self,
        db: Session,
        company_id: int,
        *,
        limit: int = 100,
    ) -> List[FraudAlert]:
        """
        Retrieve unresolved fraud alerts.
        """

        query = db.query(FraudAlert).filter(
            FraudAlert.company_id == company_id
        )

        if hasattr(FraudAlert, "resolved"):
            query = query.filter(
                FraudAlert.resolved.is_(False)
            )

        elif hasattr(FraudAlert, "status"):
            query = query.filter(
                FraudAlert.status.notin_(
                    ["resolved", "closed"]
                )
            )

        return (
            query
            .order_by(desc(FraudAlert.created_at))
            .limit(limit)
            .all()
        )

    # ========================================================
    # GENERIC COMPANY ALERTS
    # ========================================================

    def get_all_company_alerts(
        self,
        db: Session,
        company_id: int,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Any]:
        """
        Retrieve both risk and fraud alerts.

        Results are sorted in Python because the two alert
        models are separate database tables.
        """

        risk_alerts = (
            db.query(RiskAlert)
            .filter(
                RiskAlert.company_id == company_id
            )
            .order_by(desc(RiskAlert.created_at))
            .limit(limit)
            .all()
        )

        fraud_alerts = (
            db.query(FraudAlert)
            .filter(
                FraudAlert.company_id == company_id
            )
            .order_by(desc(FraudAlert.created_at))
            .limit(limit)
            .all()
        )

        alerts = [
            *risk_alerts,
            *fraud_alerts,
        ]

        alerts.sort(
            key=lambda alert: (
                getattr(
                    alert,
                    "created_at",
                    datetime.min,
                )
                or datetime.min
            ),
            reverse=True,
        )

        return alerts[offset: offset + limit]

    # ========================================================
    # RISK ALERT UPDATE
    # ========================================================

    def update_risk_alert(
        self,
        db: Session,
        alert_id: int,
        updates: Dict[str, Any],
    ) -> Optional[RiskAlert]:
        """
        Update a risk alert.

        Only existing model attributes are updated.
        """

        alert = self.get_risk_alert_by_id(
            db,
            alert_id,
        )

        if alert is None:
            return None

        for field, value in updates.items():
            if hasattr(alert, field):
                setattr(alert, field, value)

        db.flush()
        db.refresh(alert)

        return alert

    # ========================================================
    # FRAUD ALERT UPDATE
    # ========================================================

    def update_fraud_alert(
        self,
        db: Session,
        alert_id: int,
        updates: Dict[str, Any],
    ) -> Optional[FraudAlert]:
        """
        Update a fraud alert.
        """

        alert = self.get_fraud_alert_by_id(
            db,
            alert_id,
        )

        if alert is None:
            return None

        for field, value in updates.items():
            if hasattr(alert, field):
                setattr(alert, field, value)

        db.flush()
        db.refresh(alert)

        return alert

    # ========================================================
    # RESOLVE RISK ALERT
    # ========================================================

    def resolve_risk_alert(
        self,
        db: Session,
        alert_id: int,
        *,
        resolved_by: Optional[int] = None,
        resolution_note: Optional[str] = None,
    ) -> Optional[RiskAlert]:
        """
        Resolve a risk alert.

        Supports common model field conventions.
        """

        alert = self.get_risk_alert_by_id(
            db,
            alert_id,
        )

        if alert is None:
            return None

        if hasattr(alert, "resolved"):
            alert.resolved = True

        if hasattr(alert, "status"):
            alert.status = "resolved"

        if hasattr(alert, "resolved_at"):
            alert.resolved_at = datetime.utcnow()

        if (
            resolved_by is not None
            and hasattr(alert, "resolved_by")
        ):
            alert.resolved_by = resolved_by

        if (
            resolution_note is not None
            and hasattr(alert, "resolution_note")
        ):
            alert.resolution_note = resolution_note

        db.flush()
        db.refresh(alert)

        return alert

    # ========================================================
    # RESOLVE FRAUD ALERT
    # ========================================================

    def resolve_fraud_alert(
        self,
        db: Session,
        alert_id: int,
        *,
        resolved_by: Optional[int] = None,
        resolution_note: Optional[str] = None,
    ) -> Optional[FraudAlert]:
        """
        Resolve a fraud alert.
        """

        alert = self.get_fraud_alert_by_id(
            db,
            alert_id,
        )

        if alert is None:
            return None

        if hasattr(alert, "resolved"):
            alert.resolved = True

        if hasattr(alert, "status"):
            alert.status = "resolved"

        if hasattr(alert, "resolved_at"):
            alert.resolved_at = datetime.utcnow()

        if (
            resolved_by is not None
            and hasattr(alert, "resolved_by")
        ):
            alert.resolved_by = resolved_by

        if (
            resolution_note is not None
            and hasattr(alert, "resolution_note")
        ):
            alert.resolution_note = resolution_note

        db.flush()
        db.refresh(alert)

        return alert

    # ========================================================
    # DELETE / ARCHIVE
    # ========================================================

    def delete_risk_alert(
        self,
        db: Session,
        alert_id: int,
    ) -> bool:
        """
        Delete a risk alert.

        Prefer soft deletion/archive in production if your
        audit requirements require historical retention.
        """

        alert = self.get_risk_alert_by_id(
            db,
            alert_id,
        )

        if alert is None:
            return False

        db.delete(alert)
        db.flush()

        return True

    def delete_fraud_alert(
        self,
        db: Session,
        alert_id: int,
    ) -> bool:
        """
        Delete a fraud alert.
        """

        alert = self.get_fraud_alert_by_id(
            db,
            alert_id,
        )

        if alert is None:
            return False

        db.delete(alert)
        db.flush()

        return True

    # ========================================================
    # DATE FILTERING
    # ========================================================

    def get_risk_alerts_by_date_range(
        self,
        db: Session,
        company_id: int,
        start_date: datetime,
        end_date: datetime,
        *,
        limit: int = 500,
    ) -> List[RiskAlert]:
        """
        Retrieve risk alerts in a date range.
        """

        return (
            db.query(RiskAlert)
            .filter(
                and_(
                    RiskAlert.company_id == company_id,
                    RiskAlert.created_at >= start_date,
                    RiskAlert.created_at <= end_date,
                )
            )
            .order_by(desc(RiskAlert.created_at))
            .limit(limit)
            .all()
        )

    def get_fraud_alerts_by_date_range(
        self,
        db: Session,
        company_id: int,
        start_date: datetime,
        end_date: datetime,
        *,
        limit: int = 500,
    ) -> List[FraudAlert]:
        """
        Retrieve fraud alerts in a date range.
        """

        return (
            db.query(FraudAlert)
            .filter(
                and_(
                    FraudAlert.company_id == company_id,
                    FraudAlert.created_at >= start_date,
                    FraudAlert.created_at <= end_date,
                )
            )
            .order_by(desc(FraudAlert.created_at))
            .limit(limit)
            .all()
        )

    # ========================================================
    # COUNT
    # ========================================================

    def count_risk_alerts(
        self,
        db: Session,
        company_id: int,
    ) -> int:
        """
        Count risk alerts for a company.
        """

        return (
            db.query(func.count(RiskAlert.id))
            .filter(
                RiskAlert.company_id == company_id
            )
            .scalar()
            or 0
        )

    def count_fraud_alerts(
        self,
        db: Session,
        company_id: int,
    ) -> int:
        """
        Count fraud alerts for a company.
        """

        return (
            db.query(func.count(FraudAlert.id))
            .filter(
                FraudAlert.company_id == company_id
            )
            .scalar()
            or 0
        )

    # ========================================================
    # SEVERITY STATISTICS
    # ========================================================

    def get_risk_alert_severity_counts(
        self,
        db: Session,
        company_id: int,
    ) -> Dict[str, int]:
        """
        Count risk alerts by severity.
        """

        if not hasattr(RiskAlert, "severity"):
            return {}

        rows = (
            db.query(
                RiskAlert.severity,
                func.count(RiskAlert.id),
            )
            .filter(
                RiskAlert.company_id == company_id
            )
            .group_by(RiskAlert.severity)
            .all()
        )

        return {
            str(severity): count
            for severity, count in rows
        }

    def get_fraud_alert_severity_counts(
        self,
        db: Session,
        company_id: int,
    ) -> Dict[str, int]:
        """
        Count fraud alerts by severity/risk level.
        """

        field = None

        if hasattr(FraudAlert, "severity"):
            field = FraudAlert.severity
        elif hasattr(FraudAlert, "risk_level"):
            field = FraudAlert.risk_level

        if field is None:
            return {}

        rows = (
            db.query(
                field,
                func.count(FraudAlert.id),
            )
            .filter(
                FraudAlert.company_id == company_id
            )
            .group_by(field)
            .all()
        )

        return {
            str(level): count
            for level, count in rows
        }

    # ========================================================
    # ALERT DASHBOARD STATISTICS
    # ========================================================

    def get_alert_statistics(
        self,
        db: Session,
        company_id: int,
    ) -> Dict[str, Any]:
        """
        Generate dashboard-level alert statistics.

        This method aggregates both risk and fraud alerts.
        """

        risk_total = self.count_risk_alerts(
            db,
            company_id,
        )

        fraud_total = self.count_fraud_alerts(
            db,
            company_id,
        )

        risk_severity = (
            self.get_risk_alert_severity_counts(
                db,
                company_id,
            )
        )

        fraud_severity = (
            self.get_fraud_alert_severity_counts(
                db,
                company_id,
            )
        )

        return {
            "company_id": company_id,
            "total_alerts": risk_total + fraud_total,
            "risk_alerts": risk_total,
            "fraud_alerts": fraud_total,
            "risk_severity": risk_severity,
            "fraud_severity": fraud_severity,
        }

    # ========================================================
    # RECENT ALERTS
    # ========================================================

    def get_recent_alerts(
        self,
        db: Session,
        company_id: int,
        *,
        limit: int = 20,
    ) -> List[Any]:
        """
        Retrieve the most recent risk/fraud alerts.
        """

        return self.get_all_company_alerts(
            db,
            company_id,
            limit=limit,
        )

    # ========================================================
    # HIGH-RISK ALERTS
    # ========================================================

    def get_high_risk_alerts(
        self,
        db: Session,
        company_id: int,
        *,
        limit: int = 50,
    ) -> List[Any]:
        """
        Retrieve high/critical alerts across alert types.
        """

        risk_alerts = self.get_critical_risk_alerts(
            db,
            company_id,
            limit=limit,
        )

        fraud_alerts = (
            self.get_unresolved_fraud_alerts(
                db,
                company_id,
                limit=limit,
            )
        )

        alerts = [
            *risk_alerts,
            *fraud_alerts,
        ]

        alerts.sort(
            key=lambda alert: (
                getattr(
                    alert,
                    "risk_score",
                    0,
                )
                or 0
            ),
            reverse=True,
        )

        return alerts[:limit]

    # ========================================================
    # TRANSACTION-RELATED ALERTS
    # ========================================================

    def get_fraud_alerts_by_transaction(
        self,
        db: Session,
        transaction_id: int,
        *,
        limit: int = 50,
    ) -> List[FraudAlert]:
        """
        Retrieve fraud alerts associated with a transaction.
        """

        if not hasattr(
            FraudAlert,
            "transaction_id",
        ):
            return []

        return (
            db.query(FraudAlert)
            .filter(
                FraudAlert.transaction_id
                == transaction_id
            )
            .order_by(
                desc(FraudAlert.created_at)
            )
            .limit(limit)
            .all()
        )

    # ========================================================
    # EXISTENCE CHECKS
    # ========================================================

    def risk_alert_exists(
        self,
        db: Session,
        alert_id: int,
    ) -> bool:
        """
        Check whether a risk alert exists.
        """

        return (
            db.query(RiskAlert.id)
            .filter(RiskAlert.id == alert_id)
            .first()
            is not None
        )

    def fraud_alert_exists(
        self,
        db: Session,
        alert_id: int,
    ) -> bool:
        """
        Check whether a fraud alert exists.
        """

        return (
            db.query(FraudAlert.id)
            .filter(FraudAlert.id == alert_id)
            .first()
            is not None
        )

    # ========================================================
    # FLUSH / COMMIT HELPERS
    # ========================================================

    def flush(self, db: Session) -> None:
        """
        Flush pending repository changes.

        Transaction commit should normally be controlled by
        the service/unit-of-work layer.
        """

        db.flush()

### Recommended responsibility split

┌──────────────────────────────────────────────┐
│              ALERT ENGINE                    │
│                                              │
│ rules.py                                      │
│ detector.py                                   │
│ severity.py                                   │
│ risk_scoring.py                               │
│ alert_generator.py                            │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│             ALERT SERVICE                    │
│                                              │
│ alert_service.py                             │
│ notification.py                              │
│ escalation.py                                │
└──────────────────────┬───────────────────────┘
                       ↓
┌──────────────────────────────────────────────┐
│          ALERT REPOSITORY                    │
│                                              │
│ create                                        │
│ get                                            │
│ update                                         │
│ resolve                                        │
│ delete/archive                                 │
│ filter                                         │
│ count                                          │
│ statistics                                     │
└──────────────────────┬───────────────────────┘
                       ↓
                SQLAlchemy ORM
                       ↓
              PostgreSQL / Database

**Important:** this repository intentionally uses `RiskAlert` and `FraudAlert` separately because your current database architecture has both models. If your exact model fields differ—for example, `severity` is an Enum rather than a string, or the model uses `is_resolved` instead of `resolved`—those field names should be aligned with the corresponding model definitions before running migrations/tests.