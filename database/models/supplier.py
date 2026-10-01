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


class Supplier(Base):
    """
    Supplier / vendor model for FinCo AI.

    Supports:
    - Supplier master data
    - Purchase and expense analysis
    - Invoice tracking
    - Payment behavior
    - Supplier risk scoring
    - Fraud/anomaly detection
    - Supplier concentration analysis
    - Cost optimization
    - AI recommendations
    """

    __tablename__ = "suppliers"

    __table_args__ = (
        Index("ix_supplier_company_id", "company_id"),
        Index("ix_supplier_name", "name"),
        Index("ix_supplier_tax_id", "tax_id"),
        Index("ix_supplier_email", "email"),
        Index("ix_supplier_status", "status"),
        Index("ix_supplier_risk_score", "risk_score"),
        Index("ix_supplier_category", "category"),
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

    # ------------------------------------------------------------------
    # Supplier Identity
    # ------------------------------------------------------------------

    supplier_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    legal_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    subcategory: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Contact Information
    # ------------------------------------------------------------------

    email: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    phone: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    website: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    contact_person: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Address
    # ------------------------------------------------------------------

    address_line1: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    address_line2: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    city: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    state: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    country: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    postal_code: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Tax / Compliance
    # ------------------------------------------------------------------

    tax_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    tax_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    registration_number: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    compliance_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="UNKNOWN",
    )
    # UNKNOWN / VERIFIED / PENDING / FAILED

    # ------------------------------------------------------------------
    # Payment Terms
    # ------------------------------------------------------------------

    payment_terms_days: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    preferred_payment_method: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
    )

    # ------------------------------------------------------------------
    # Financial Metrics
    # ------------------------------------------------------------------

    total_purchase_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_paid_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_outstanding_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_invoice_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    overdue_invoice_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ------------------------------------------------------------------
    # Payment Performance
    # ------------------------------------------------------------------

    average_payment_days: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2),
        nullable=True,
    )

    average_invoice_amount: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    on_time_payment_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    financial_risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    fraud_risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    concentration_risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    reliability_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Fraud / Anomaly Detection
    # ------------------------------------------------------------------

    anomaly_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    duplicate_invoice_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    suspicious_transaction_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    is_high_risk: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_anomalous: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------
    # Supplier Status
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="ACTIVE",
    )
    # ACTIVE / INACTIVE / SUSPENDED / BLOCKED

    is_preferred: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------
    # Procurement / Business Information
    # ------------------------------------------------------------------

    contract_reference: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    contract_start_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    contract_end_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # AI Analysis
    # ------------------------------------------------------------------

    ai_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    risk_analysis: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    recommended_action: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    notes: Mapped[Optional[str]] = mapped_column(
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
        back_populates="suppliers",
    )

    invoices = relationship(
        "Invoice",
        back_populates="supplier",
    )

    expenses = relationship(
        "Expense",
        back_populates="supplier",
    )

    # ------------------------------------------------------------------
    # Business Logic
    # ------------------------------------------------------------------

    def calculate_outstanding(self) -> Decimal:
        """Calculate supplier outstanding balance."""

        return max(
            Decimal("0.00"),
            self.total_purchase_amount
            - self.total_paid_amount,
        )

    def calculate_average_invoice_amount(
        self,
    ) -> Optional[Decimal]:
        """Calculate average supplier invoice value."""

        if self.total_invoice_count <= 0:
            return None

        return (
            self.total_purchase_amount
            / Decimal(self.total_invoice_count)
        )

    def calculate_payment_rate(self) -> Optional[Decimal]:
        """Calculate percentage of purchases already paid."""

        if self.total_purchase_amount == Decimal("0"):
            return None

        return (
            self.total_paid_amount
            / self.total_purchase_amount
        ) * Decimal("100")

    def calculate_risk_score(self) -> Optional[Decimal]:
        """
        Calculate a simple weighted supplier risk score.

        Individual ML risk models can replace this calculation later.
        """

        components = []

        if self.financial_risk_score is not None:
            components.append(
                self.financial_risk_score * Decimal("0.30")
            )

        if self.fraud_risk_score is not None:
            components.append(
                self.fraud_risk_score * Decimal("0.30")
            )

        if self.concentration_risk_score is not None:
            components.append(
                self.concentration_risk_score * Decimal("0.20")
            )

        if self.anomaly_score is not None:
            components.append(
                self.anomaly_score * Decimal("0.20")
            )

        if not components:
            return None

        return sum(components)

    def update_metrics(self) -> None:
        """Update derived supplier metrics."""

        self.total_outstanding_amount = (
            self.calculate_outstanding()
        )

        self.average_invoice_amount = (
            self.calculate_average_invoice_amount()
        )

        self.risk_score = self.calculate_risk_score()

        if self.risk_score is not None:
            self.is_high_risk = (
                self.risk_score >= Decimal("0.75")
            )

    def is_contract_expired(self) -> bool:
        """Check whether the supplier contract has expired."""

        if self.contract_end_date is None:
            return False

        return self.contract_end_date < datetime.utcnow()

    def __repr__(self) -> str:
        return (
            f"<Supplier("
            f"id={self.id}, "
            f"code='{self.supplier_code}', "
            f"name='{self.name}', "
            f"risk_score={self.risk_score}, "
            f"status='{self.status}'"
            f")>"
        )