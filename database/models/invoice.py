from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class Invoice(Base):
    """
    Invoice model for FinCo AI.

    Represents invoices issued by or received by a company and supports:
    - Revenue/expense tracking
    - Customer/supplier relationships
    - Payment tracking
    - Financial analysis
    - Cash-flow forecasting
    - Fraud/anomaly detection
    - Alert generation
    - Auditability
    """

    __tablename__ = "invoices"

    __table_args__ = (
        Index("ix_invoice_company_id", "company_id"),
        Index("ix_invoice_customer_id", "customer_id"),
        Index("ix_invoice_supplier_id", "supplier_id"),
        Index("ix_invoice_invoice_number", "invoice_number"),
        Index("ix_invoice_issue_date", "issue_date"),
        Index("ix_invoice_due_date", "due_date"),
        Index("ix_invoice_status", "status"),
        Index("ix_invoice_payment_status", "payment_status"),
    )

    # ------------------------------------------------------------------
    # Primary Key
    # ------------------------------------------------------------------

    id: Mapped[int] = mapped_column(
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

    supplier_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("suppliers.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Invoice Identification
    # ------------------------------------------------------------------

    invoice_number: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    invoice_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="sales",
    )
    # sales / purchase / credit_note / debit_note

    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
    )

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    issue_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    due_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    paid_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Amounts
    # ------------------------------------------------------------------

    subtotal: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    tax_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    discount_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    paid_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    outstanding_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="issued",
    )
    # draft / issued / sent / overdue / paid / cancelled

    payment_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="unpaid",
    )
    # unpaid / partially_paid / paid / overdue

    # ------------------------------------------------------------------
    # Customer / Supplier Information
    # ------------------------------------------------------------------

    counterparty_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    counterparty_email: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    counterparty_tax_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Financial Classification
    # ------------------------------------------------------------------

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    payment_method: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Document / Source
    # ------------------------------------------------------------------

    document_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    source: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )
    # upload / api / manual / email / OCR / imported

    # ------------------------------------------------------------------
    # Fraud / Anomaly Signals
    # ------------------------------------------------------------------

    anomaly_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    fraud_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    is_duplicate: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_anomaly: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------
    # Notes / Metadata
    # ------------------------------------------------------------------

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    extra_metadata: Mapped[Optional[str]] = mapped_column(
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
        back_populates="invoices",
    )

    customer = relationship(
        "Customer",
        back_populates="invoices",
    )

    supplier = relationship(
        "Supplier",
        back_populates="invoices",
    )

    document = relationship(
        "Document",
        back_populates="invoice",
    )

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def calculate_outstanding(self) -> Decimal:
        """Calculate the remaining unpaid amount."""

        return max(
            Decimal("0.00"),
            self.total_amount - self.paid_amount,
        )

    def update_payment_status(self) -> None:
        """Update payment status based on paid/outstanding amounts."""

        self.outstanding_amount = self.calculate_outstanding()

        if self.outstanding_amount <= Decimal("0.00"):
            self.payment_status = "paid"
            self.status = "paid"
            return

        if self.paid_amount > Decimal("0.00"):
            self.payment_status = "partially_paid"
            return

        if self.due_date < date.today():
            self.payment_status = "overdue"
            self.status = "overdue"
        else:
            self.payment_status = "unpaid"

    def days_overdue(self) -> int:
        """Return the number of days the invoice is overdue."""

        if self.payment_status != "overdue":
            return 0

        return max(
            0,
            (date.today() - self.due_date).days,
        )

    def is_overdue(self) -> bool:
        """Check whether the invoice is overdue."""

        return (
            self.outstanding_amount > Decimal("0.00")
            and self.due_date < date.today()
        )

    def __repr__(self) -> str:
        return (
            f"<Invoice("
            f"id={self.id}, "
            f"invoice_number='{self.invoice_number}', "
            f"total_amount={self.total_amount}, "
            f"status='{self.status}'"
            f")>"
        )