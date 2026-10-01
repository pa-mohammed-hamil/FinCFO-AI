from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class RiskAlert(Base):
    """
    Financial/business risk alert for FinCo AI.

    Examples:
    - Financial profile deterioration
    - Previous-period decline
    - Severe loss risk
    - Liquidity risk
    - Cash-flow risk
    - Profitability deterioration
    - Revenue decline
    - Budget variance
    - Business continuity risk

    The alert engine creates the alert, while the Agentic AI layer can
    investigate the cause, retrieve evidence, forecast future impact,
    perform what-if analysis, and generate recommendations.
    """

    __tablename__ = "risk_alerts"

    __table_args__ = (
        Index("ix_risk_alert_company_id", "company_id"),
        Index("ix_risk_alert_alert_type", "alert_type"),
        Index("ix_risk_alert_severity", "severity"),
        Index("ix_risk_alert_status", "status"),
        Index("ix_risk_alert_risk_score", "risk_score"),
        Index("ix_risk_alert_created_at", "created_at"),
        Index("ix_risk_alert_detected_at", "detected_at"),
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
    # Ownership
    # ------------------------------------------------------------------

    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Optional user who acknowledged/reviewed the alert.
    acknowledged_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Optional user who resolved the alert.
    resolved_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Alert Identity
    # ------------------------------------------------------------------

    alert_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    alert_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    # financial_profile_down
    # previous_period_decline
    # severe_loss_risk
    # liquidity_risk
    # cash_flow_risk
    # profitability_risk
    # revenue_risk
    # budget_variance
    # business_continuity_risk

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="financial",
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Risk Scoring
    # ------------------------------------------------------------------

    risk_score: Mapped[Decimal] = mapped_column(
        Numeric(8, 5),
        nullable=False,
        default=Decimal("0.00000"),
    )

    severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="MEDIUM",
    )
    # INFO / LOW / MEDIUM / HIGH / CRITICAL

    confidence_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Risk Components
    # ------------------------------------------------------------------

    financial_health_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    liquidity_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    profitability_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    revenue_risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    cash_flow_risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    business_continuity_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Trigger Information
    # ------------------------------------------------------------------

    trigger_metric: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    trigger_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 5),
        nullable=True,
    )

    threshold_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 5),
        nullable=True,
    )

    trigger_operator: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
    )
    # < / <= / > / >= / ==

    rule_code: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Financial Context
    # ------------------------------------------------------------------

    affected_metric: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    current_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    previous_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    change_amount: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
    )

    # ------------------------------------------------------------------
    # Forecast Information
    # ------------------------------------------------------------------

    forecast_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    forecast_lower_bound: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    forecast_upper_bound: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    forecast_horizon: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------

    evidence: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    supporting_metrics: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    citations: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Agent Investigation
    # ------------------------------------------------------------------

    investigation_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="NOT_STARTED",
    )
    # NOT_STARTED / IN_PROGRESS / COMPLETED / FAILED

    root_cause: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    agent_analysis: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    agent_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    model_name: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    recommended_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    recommendation_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(
            "recommendations.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # What-If / Scenario
    # ------------------------------------------------------------------

    scenario_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey(
            "scenarios.id",
            ondelete="SET NULL",
        ),
        nullable=True,
    )

    what_if_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Alert Lifecycle
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="OPEN",
    )
    # OPEN
    # ACKNOWLEDGED
    # INVESTIGATING
    # RESOLVED
    # DISMISSED

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    resolution_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Notification / Escalation
    # ------------------------------------------------------------------

    notification_sent: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    escalation_level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    escalated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

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
        back_populates="risk_alerts",
    )

    acknowledger = relationship(
        "User",
        foreign_keys=[acknowledged_by],
    )

    resolver = relationship(
        "User",
        foreign_keys=[resolved_by],
    )

    recommendation = relationship(
        "Recommendation",
        foreign_keys=[recommendation_id],
    )

    scenario = relationship(
        "Scenario",
        foreign_keys=[scenario_id],
    )

    # ------------------------------------------------------------------
    # Business Logic
    # ------------------------------------------------------------------

    def acknowledge(self, user_id: int) -> None:
        """Acknowledge an active risk alert."""

        self.status = "ACKNOWLEDGED"
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.utcnow()

    def start_investigation(self) -> None:
        """Start AI/human investigation."""

        self.status = "INVESTIGATING"
        self.investigation_status = "IN_PROGRESS"

    def complete_investigation(
        self,
        root_cause: str,
        analysis: Optional[str] = None,
    ) -> None:
        """Complete the investigation."""

        self.investigation_status = "COMPLETED"
        self.root_cause = root_cause

        if analysis:
            self.agent_analysis = analysis

    def resolve(
        self,
        user_id: int,
        notes: Optional[str] = None,
    ) -> None:
        """Resolve the risk alert."""

        self.status = "RESOLVED"
        self.is_active = False
        self.resolved_by = user_id
        self.resolved_at = datetime.utcnow()

        if notes:
            self.resolution_notes = notes

    def dismiss(
        self,
        user_id: int,
        notes: Optional[str] = None,
    ) -> None:
        """Dismiss a risk alert."""

        self.status = "DISMISSED"
        self.is_active = False
        self.resolved_by = user_id
        self.resolved_at = datetime.utcnow()

        if notes:
            self.resolution_notes = notes

    def escalate(self) -> None:
        """Increase the escalation level."""

        self.escalation_level += 1
        self.escalated_at = datetime.utcnow()

    def is_critical(self) -> bool:
        """Check whether this is a critical risk."""

        return self.severity == "CRITICAL"

    def requires_immediate_action(self) -> bool:
        """Determine whether immediate human attention is required."""

        return (
            self.is_active
            and self.severity in {"HIGH", "CRITICAL"}
        )

    def calculate_change_percentage(self) -> Optional[Decimal]:
        """Calculate percentage change from the previous value."""

        if (
            self.current_value is None
            or self.previous_value is None
            or self.previous_value == Decimal("0")
        ):
            return None

        return (
            (
                self.current_value
                - self.previous_value
            )
            / self.previous_value
        ) * Decimal("100")

    def __repr__(self) -> str:
        return (
            f"<RiskAlert("
            f"id={self.id}, "
            f"code='{self.alert_code}', "
            f"type='{self.alert_type}', "
            f"severity='{self.severity}', "
            f"risk_score={self.risk_score}, "
            f"status='{self.status}'"
            f")>"
        )