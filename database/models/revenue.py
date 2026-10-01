from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Date,
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


class Revenue(Base):
    """
    Revenue record for FinCo AI.

    Supports:
    - Revenue tracking
    - Customer-level revenue analysis
    - Product/service revenue analysis
    - Period-over-period comparison
    - Forecasting
    - Financial health analysis
    - Anomaly detection
    - Alert generation
    - Profitability analysis
    """

    __tablename__ = "revenues"

    __table_args__ = (
        Index("ix_revenue_company_id", "company_id"),
        Index("ix_revenue_customer_id", "customer_id"),
        Index("ix_revenue_invoice_id", "invoice_id"),
        Index("ix_revenue_transaction_id", "transaction_id"),
        Index("ix_revenue_revenue_date", "revenue_date"),
        Index("ix_revenue_category", "category"),
        Index("ix_revenue_period", "period"),
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
    # Ownership / Relationships
    # ------------------------------------------------------------------

    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
    )

    customer_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL"),
        nullable=True,
    )

    invoice_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("invoices.id", ondelete="SET NULL"),
        nullable=True,
    )

    transaction_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("transactions.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Revenue Identification
    # ------------------------------------------------------------------

    revenue_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    revenue_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="operating",
    )
    # operating / non_operating / recurring / one_time

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    subcategory: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Revenue Source
    # ------------------------------------------------------------------

    source: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    # invoice / subscription / contract / transaction /
    # manual / imported / API

    product_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    service_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Customer Information
    # ------------------------------------------------------------------

    customer_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Financial Amounts
    # ------------------------------------------------------------------

    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
    )

    gross_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    discount_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    refund_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    tax_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    net_revenue: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    # ------------------------------------------------------------------
    # Cost / Margin
    # ------------------------------------------------------------------

    cost_of_revenue: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    gross_profit: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    gross_margin: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Quantity / Unit Economics
    # ------------------------------------------------------------------

    quantity: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 4),
        nullable=True,
    )

    unit_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 4),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Time Information
    # ------------------------------------------------------------------

    revenue_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    period: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )
    # daily / weekly / monthly / quarterly / yearly

    fiscal_year: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    fiscal_quarter: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    fiscal_month: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Comparison Metrics
    # ------------------------------------------------------------------

    previous_period_revenue: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    growth_amount: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    growth_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    forecast_value: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    forecast_variance: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Anomaly / Risk
    # ------------------------------------------------------------------

    anomaly_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    is_anomaly: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Description / Metadata
    # ------------------------------------------------------------------

    description: Mapped[Optional[str]] = mapped_column(
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
        back_populates="revenues",
    )

    customer = relationship(
        "Customer",
        back_populates="revenues",
    )

    invoice = relationship(
        "Invoice",
        back_populates="revenue",
    )

    transaction = relationship(
        "Transaction",
        back_populates="revenue",
    )

    # ------------------------------------------------------------------
    # Business Logic
    # ------------------------------------------------------------------

    def calculate_net_revenue(self) -> Decimal:
        """Calculate revenue after discounts and refunds."""

        return (
            self.gross_amount
            - self.discount_amount
            - self.refund_amount
        )

    def calculate_growth(
        self,
        previous_revenue: Decimal,
    ) -> Optional[Decimal]:
        """Calculate percentage growth from the previous period."""

        if previous_revenue == Decimal("0"):
            return None

        return (
            (self.net_revenue - previous_revenue)
            / previous_revenue
        ) * Decimal("100")

    def calculate_gross_profit(self) -> Optional[Decimal]:
        """Calculate gross profit."""

        if self.cost_of_revenue is None:
            return None

        return self.net_revenue - self.cost_of_revenue

    def calculate_gross_margin(self) -> Optional[Decimal]:
        """Calculate gross margin percentage."""

        if self.net_revenue == Decimal("0"):
            return None

        if self.cost_of_revenue is None:
            return None

        gross_profit = (
            self.net_revenue - self.cost_of_revenue
        )

        return (
            gross_profit / self.net_revenue
        ) * Decimal("100")

    def is_declining(
        self,
        threshold: Decimal = Decimal("0"),
    ) -> bool:
        """
        Determine whether revenue declined compared
        with the previous period.
        """

        if self.previous_period_revenue is None:
            return False

        return (
            self.net_revenue
            < self.previous_period_revenue - threshold
        )

    def update_financial_metrics(self) -> None:
        """Recalculate derived financial metrics."""

        self.net_revenue = self.calculate_net_revenue()

        if self.previous_period_revenue is not None:
            self.growth_amount = (
                self.net_revenue
                - self.previous_period_revenue
            )

            self.growth_percentage = self.calculate_growth(
                self.previous_period_revenue
            )

        if self.cost_of_revenue is not None:
            self.gross_profit = self.calculate_gross_profit()
            self.gross_margin = self.calculate_gross_margin()

    def __repr__(self) -> str:
        return (
            f"<Revenue("
            f"id={self.id}, "
            f"code='{self.revenue_code}', "
            f"net_revenue={self.net_revenue}, "
            f"date={self.revenue_date}"
            f")>"
        )