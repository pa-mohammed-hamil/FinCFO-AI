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


class Scenario(Base):
    """
    What-If / financial scenario model for FinCo AI.

    Examples:
    - Revenue decreases by 15%
    - Operating expenses increase by 10%
    - Customer churn increases to 8%
    - Pricing increases by 5%
    - Cash collection improves by 20%
    - Cost reduction of INR 500,000

    Supports:
    - Baseline comparison
    - Scenario simulation
    - Sensitivity analysis
    - Forecast integration
    - Risk analysis
    - Recommendation generation
    """

    __tablename__ = "scenarios"

    __table_args__ = (
        Index("ix_scenario_company_id", "company_id"),
        Index("ix_scenario_name", "name"),
        Index("ix_scenario_type", "scenario_type"),
        Index("ix_scenario_status", "status"),
        Index("ix_scenario_created_at", "created_at"),
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

    created_by: Mapped[Optional[int]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Scenario Identity
    # ------------------------------------------------------------------

    scenario_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    scenario_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    # what_if / stress_test / sensitivity / simulation /
    # best_case / base_case / worst_case / custom

    # ------------------------------------------------------------------
    # Scenario Status
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="DRAFT",
    )
    # DRAFT / RUNNING / COMPLETED / FAILED / ARCHIVED

    is_baseline: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------
    # Analysis Period
    # ------------------------------------------------------------------

    start_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    end_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    forecast_horizon_months: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Scenario Assumptions
    # ------------------------------------------------------------------

    revenue_change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    expense_change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    cost_change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    customer_growth_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    churn_change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    pricing_change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    collection_change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Custom Assumptions
    # ------------------------------------------------------------------

    assumptions_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    constraints_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    variables_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Baseline Financial Values
    # ------------------------------------------------------------------

    baseline_revenue: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    baseline_expenses: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    baseline_profit: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    baseline_cash_flow: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    baseline_cash_balance: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    baseline_margin: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Scenario Results
    # ------------------------------------------------------------------

    projected_revenue: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    projected_expenses: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    projected_profit: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    projected_cash_flow: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    projected_cash_balance: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    projected_margin: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Scenario Impact
    # ------------------------------------------------------------------

    revenue_impact: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    expense_impact: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    profit_impact: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    cash_flow_impact: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    margin_impact: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Risk Analysis
    # ------------------------------------------------------------------

    risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    financial_health_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    liquidity_risk: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    profitability_risk: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    business_continuity_risk: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Sensitivity Analysis
    # ------------------------------------------------------------------

    sensitivity_results_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    best_case_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    worst_case_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    expected_case_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    simulation_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    simulation_results_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    confidence_level: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # AI Interpretation
    # ------------------------------------------------------------------

    ai_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    ai_risk_analysis: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    ai_recommendation: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    model_name: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    model_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Currency / Metadata
    # ------------------------------------------------------------------

    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
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

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    company = relationship(
        "Company",
        back_populates="scenarios",
    )

    creator = relationship(
        "User",
        foreign_keys=[created_by],
    )

    recommendations = relationship(
        "Recommendation",
        back_populates="scenario",
    )

    # ------------------------------------------------------------------
    # Business Logic
    # ------------------------------------------------------------------

    def calculate_projected_revenue(self) -> Optional[Decimal]:
        """Calculate revenue after applying the revenue assumption."""

        if self.baseline_revenue is None:
            return None

        if self.revenue_change_percentage is None:
            return self.baseline_revenue

        multiplier = (
            Decimal("1")
            + self.revenue_change_percentage / Decimal("100")
        )

        return self.baseline_revenue * multiplier

    def calculate_projected_expenses(self) -> Optional[Decimal]:
        """Calculate expenses after applying the expense assumption."""

        if self.baseline_expenses is None:
            return None

        if self.expense_change_percentage is None:
            return self.baseline_expenses

        multiplier = (
            Decimal("1")
            + self.expense_change_percentage / Decimal("100")
        )

        return self.baseline_expenses * multiplier

    def calculate_projected_profit(self) -> Optional[Decimal]:
        """Calculate projected profit."""

        revenue = self.projected_revenue
        expenses = self.projected_expenses

        if revenue is None or expenses is None:
            return None

        return revenue - expenses

    def calculate_profit_impact(self) -> Optional[Decimal]:
        """Calculate change in profit against baseline."""

        if (
            self.projected_profit is None
            or self.baseline_profit is None
        ):
            return None

        return self.projected_profit - self.baseline_profit

    def calculate_margin(
        self,
        revenue: Optional[Decimal],
        profit: Optional[Decimal],
    ) -> Optional[Decimal]:
        """Calculate profit margin."""

        if (
            revenue is None
            or profit is None
            or revenue == Decimal("0")
        ):
            return None

        return (
            profit / revenue
        ) * Decimal("100")

    def calculate_impacts(self) -> None:
        """Calculate projected values and scenario impacts."""

        self.projected_revenue = (
            self.calculate_projected_revenue()
        )

        self.projected_expenses = (
            self.calculate_projected_expenses()
        )

        self.projected_profit = (
            self.calculate_projected_profit()
        )

        if self.baseline_revenue is not None:
            if self.projected_revenue is not None:
                self.revenue_impact = (
                    self.projected_revenue
                    - self.baseline_revenue
                )

        if self.baseline_expenses is not None:
            if self.projected_expenses is not None:
                self.expense_impact = (
                    self.projected_expenses
                    - self.baseline_expenses
                )

        self.profit_impact = self.calculate_profit_impact()

        self.baseline_margin = self.calculate_margin(
            self.baseline_revenue,
            self.baseline_profit,
        )

        self.projected_margin = self.calculate_margin(
            self.projected_revenue,
            self.projected_profit,
        )

        if (
            self.projected_margin is not None
            and self.baseline_margin is not None
        ):
            self.margin_impact = (
                self.projected_margin
                - self.baseline_margin
            )

    def start(self) -> None:
        """Mark scenario execution as started."""

        self.status = "RUNNING"

    def complete(self) -> None:
        """Mark scenario execution as completed."""

        self.status = "COMPLETED"
        self.completed_at = datetime.utcnow()

    def fail(self) -> None:
        """Mark scenario execution as failed."""

        self.status = "FAILED"

    def is_high_risk(
        self,
        threshold: Decimal = Decimal("0.75"),
    ) -> bool:
        """Determine whether the scenario represents high risk."""

        if self.risk_score is None:
            return False

        return self.risk_score >= threshold

    def is_loss_scenario(self) -> bool:
        """Determine whether projected profit becomes negative."""

        return (
            self.projected_profit is not None
            and self.projected_profit < Decimal("0")
        )

    def __repr__(self) -> str:
        return (
            f"<Scenario("
            f"id={self.id}, "
            f"code='{self.scenario_code}', "
            f"name='{self.name}', "
            f"type='{self.scenario_type}', "
            f"status='{self.status}'"
            f")>"
        )