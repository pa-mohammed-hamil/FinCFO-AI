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


class Recommendation(Base):
    """
    AI-generated financial recommendation.

    Supports:
    - Financial recommendations
    - Cost optimization
    - Risk mitigation
    - Cash-flow improvement
    - Fraud-related actions
    - Forecast-driven recommendations
    - What-if analysis
    - Human approval workflows
    - Evidence and explainability
    - Auditability
    """

    __tablename__ = "recommendations"

    __table_args__ = (
        Index("ix_recommendation_company_id", "company_id"),
        Index("ix_recommendation_alert_id", "alert_id"),
        Index("ix_recommendation_category", "category"),
        Index("ix_recommendation_priority", "priority"),
        Index("ix_recommendation_status", "status"),
        Index("ix_recommendation_confidence", "confidence_score"),
        Index("ix_recommendation_created_at", "created_at"),
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

    # Optional alert that caused this recommendation.
    alert_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("fraud_alerts.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Recommendation Identity
    # ------------------------------------------------------------------

    recommendation_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    # financial / cost / revenue / cash_flow / risk /
    # fraud / forecast / liquidity / profitability

    recommendation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    # rule_based / ml_based / agent_based / hybrid

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Recommendation Details
    # ------------------------------------------------------------------

    action: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    rationale: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    expected_outcome: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    implementation_steps: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Priority / Confidence
    # ------------------------------------------------------------------

    priority: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="MEDIUM",
    )
    # LOW / MEDIUM / HIGH / CRITICAL

    confidence_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Financial Impact
    # ------------------------------------------------------------------

    estimated_impact: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    estimated_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    estimated_savings: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    estimated_revenue_impact: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    impact_currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
    )

    impact_period: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    # monthly / quarterly / yearly

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
    # AI / Agent Information
    # ------------------------------------------------------------------

    generated_by: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    # financial_agent / recommendation_agent /
    # supervisor_agent / forecasting_model / rule_engine

    model_name: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    model_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    reasoning_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Scenario / Forecast Information
    # ------------------------------------------------------------------

    scenario_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("scenarios.id", ondelete="SET NULL"),
        nullable=True,
    )

    forecast_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("forecasts.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="PENDING",
    )
    # PENDING
    # REVIEW
    # APPROVED
    # REJECTED
    # IMPLEMENTED
    # DISMISSED

    requires_human_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    approved_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    reviewed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    implemented_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    rejection_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Outcome Tracking
    # ------------------------------------------------------------------

    actual_impact: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    outcome_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    outcome_status: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
    )
    # pending / successful / partially_successful / unsuccessful

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
        back_populates="recommendations",
    )

    alert = relationship(
        "FraudAlert",
        back_populates="recommendations",
    )

    scenario = relationship(
        "Scenario",
        back_populates="recommendations",
    )

    forecast = relationship(
        "Forecast",
        back_populates="recommendations",
    )

    approver = relationship(
        "User",
        foreign_keys=[approved_by],
    )

    # ------------------------------------------------------------------
    # Business Methods
    # ------------------------------------------------------------------

    def approve(self, user_id: int) -> None:
        """Approve the recommendation after human review."""

        self.status = "APPROVED"
        self.approved_by = user_id
        self.reviewed_at = datetime.utcnow()

    def reject(
        self,
        user_id: int,
        reason: str,
    ) -> None:
        """Reject the recommendation."""

        self.status = "REJECTED"
        self.approved_by = user_id
        self.rejection_reason = reason
        self.reviewed_at = datetime.utcnow()

    def mark_implemented(self) -> None:
        """Mark an approved recommendation as implemented."""

        self.status = "IMPLEMENTED"
        self.implemented_at = datetime.utcnow()

    def dismiss(self) -> None:
        """Dismiss the recommendation."""

        self.status = "DISMISSED"

    def calculate_realized_impact(self) -> Optional[Decimal]:
        """
        Compare actual impact with the estimated impact.
        """

        if (
            self.actual_impact is None
            or self.estimated_impact is None
        ):
            return None

        return self.actual_impact - self.estimated_impact

    def confidence_level(self) -> str:
        """Convert confidence score into a readable category."""

        if self.confidence_score is None:
            return "UNKNOWN"

        score = float(self.confidence_score)

        if score >= 0.90:
            return "VERY_HIGH"

        if score >= 0.75:
            return "HIGH"

        if score >= 0.50:
            return "MEDIUM"

        return "LOW"

    def is_actionable(self) -> bool:
        """Determine whether the recommendation can currently be acted upon."""

        return self.status in {
            "PENDING",
            "REVIEW",
            "APPROVED",
        }

    def __repr__(self) -> str:
        return (
            f"<Recommendation("
            f"id={self.id}, "
            f"code='{self.recommendation_code}', "
            f"title='{self.title}', "
            f"priority='{self.priority}', "
            f"status='{self.status}'"
            f")>"
        )