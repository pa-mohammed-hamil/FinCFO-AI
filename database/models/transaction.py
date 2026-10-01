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
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.connection import Base


class Transaction(Base):
    """
    Financial transaction model for FinCo AI.

    Represents a normalized financial transaction from:
    - Bank statements
    - Accounting systems
    - CSV/Excel uploads
    - Invoices
    - Payment systems
    - Other financial data sources

    Supports:
    - Revenue and expense analysis
    - Cash-flow analysis
    - Transaction categorization
    - Reconciliation
    - Duplicate detection
    - Fraud/anomaly detection
    - Risk scoring
    - Financial forecasting
    - AI-powered financial analysis
    """

    __tablename__ = "transactions"

    __table_args__ = (
        Index("ix_transaction_company_id", "company_id"),
        Index("ix_transaction_customer_id", "customer_id"),
        Index("ix_transaction_supplier_id", "supplier_id"),
        Index("ix_transaction_account_id", "account_id"),
        Index("ix_transaction_transaction_date", "transaction_date"),
        Index("ix_transaction_type", "transaction_type"),
        Index("ix_transaction_category", "category"),
        Index("ix_transaction_status", "status"),
        Index("ix_transaction_reference", "reference_number"),
        Index("ix_transaction_is_anomaly", "is_anomaly"),
        Index("ix_transaction_is_duplicate", "is_duplicate"),
        Index("ix_transaction_risk_score", "risk_score"),
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
    # Company
    # ------------------------------------------------------------------

    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Related Entities
    # ------------------------------------------------------------------

    customer_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("customers.id", ondelete="SET NULL"),
        nullable=True,
    )

    supplier_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("suppliers.id", ondelete="SET NULL"),
        nullable=True,
    )

    invoice_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("invoices.id", ondelete="SET NULL"),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Account / Ledger Information
    # ------------------------------------------------------------------

    account_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    account_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    ledger_code: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Transaction Identity
    # ------------------------------------------------------------------

    transaction_code: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    reference_number: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    external_transaction_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Transaction Classification
    # ------------------------------------------------------------------

    transaction_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    # SALE / PURCHASE / EXPENSE / PAYMENT / REFUND /
    # TRANSFER / DEPOSIT / WITHDRAWAL / FEE / TAX / OTHER

    direction: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    # CREDIT / DEBIT

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    subcategory: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Counterparty
    # ------------------------------------------------------------------

    counterparty_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    counterparty_account: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    counterparty_reference: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Monetary Values
    # ------------------------------------------------------------------

    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
    )

    amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    debit_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    credit_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    balance_after_transaction: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    tax_amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    transaction_date: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    value_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    posting_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Payment Information
    # ------------------------------------------------------------------

    payment_method: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    payment_reference: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    payment_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="COMPLETED",
    )
    # PENDING / COMPLETED / FAILED / REVERSED / CANCELLED

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    reconciliation_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="UNRECONCILED",
    )
    # UNRECONCILED / MATCHED / RECONCILED / DISPUTED

    reconciled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    reconciliation_reference: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Fraud / Anomaly Detection
    # ------------------------------------------------------------------

    anomaly_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    fraud_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    is_anomaly: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_duplicate: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    is_suspicious: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    duplicate_group_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    anomaly_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    fraud_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Financial Period
    # ------------------------------------------------------------------

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

    accounting_period: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Source / Import Information
    # ------------------------------------------------------------------

    source: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    # BANK / CSV / EXCEL / ERP / API / INVOICE / MANUAL / OTHER

    source_file: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    source_row_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    import_batch_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # AI / ML Analysis
    # ------------------------------------------------------------------

    ai_category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    ai_confidence_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    ai_summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="POSTED",
    )
    # PENDING / POSTED / COMPLETED / FAILED / REVERSED / CANCELLED

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
        back_populates="transactions",
    )

    customer = relationship(
        "Customer",
        back_populates="transactions",
    )

    supplier = relationship(
        "Supplier",
        back_populates="transactions",
    )

    invoice = relationship(
        "Invoice",
        back_populates="transactions",
    )

    revenue = relationship(
        "Revenue",
        back_populates="transaction",
        uselist=False,
    )

    expense = relationship(
        "Expense",
        back_populates="transaction",
        uselist=False,
    )

    # ------------------------------------------------------------------
    # Business Logic
    # ------------------------------------------------------------------

    def calculate_net_amount(self) -> Decimal:
        """Return the signed transaction amount."""

        if self.direction.upper() == "DEBIT":
            return -abs(self.amount)

        return abs(self.amount)

    def calculate_balance_impact(self) -> Decimal:
        """
        Calculate the transaction's impact on cash/bank balance.
        """

        return self.calculate_net_amount()

    def is_credit(self) -> bool:
        """Return True when transaction increases the account balance."""

        return self.direction.upper() == "CREDIT"

    def is_debit(self) -> bool:
        """Return True when transaction decreases the account balance."""

        return self.direction.upper() == "DEBIT"

    def mark_reconciled(
        self,
        reference: Optional[str] = None,
    ) -> None:
        """Mark the transaction as reconciled."""

        self.reconciliation_status = "RECONCILED"
        self.reconciled_at = datetime.utcnow()
        self.reconciliation_reference = reference

    def mark_duplicate(
        self,
        group_id: Optional[str] = None,
    ) -> None:
        """Mark transaction as a duplicate."""

        self.is_duplicate = True
        self.duplicate_group_id = group_id

    def mark_anomaly(
        self,
        score: Optional[Decimal] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Mark transaction as anomalous."""

        self.is_anomaly = True
        self.anomaly_score = score
        self.anomaly_reason = reason

    def mark_suspicious(
        self,
        fraud_score: Optional[Decimal] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Mark transaction as suspicious."""

        self.is_suspicious = True
        self.fraud_score = fraud_score
        self.fraud_reason = reason

    def calculate_risk_level(self) -> str:
        """
        Convert numerical risk score into a business risk level.
        """

        if self.risk_score is None:
            return "UNKNOWN"

        score = self.risk_score

        if score >= Decimal("0.80"):
            return "CRITICAL"

        if score >= Decimal("0.60"):
            return "HIGH"

        if score >= Decimal("0.30"):
            return "MEDIUM"

        return "LOW"

    def update_risk_score(self) -> Optional[Decimal]:
        """
        Calculate a basic combined risk score.

        Production ML scoring should eventually be handled by:
        backend/app/fraud/scoring.py
        """

        scores = []

        if self.anomaly_score is not None:
            scores.append(self.anomaly_score)

        if self.fraud_score is not None:
            scores.append(self.fraud_score)

        if not scores:
            return self.risk_score

        self.risk_score = sum(scores) / Decimal(len(scores))

        return self.risk_score

    def is_high_risk(self) -> bool:
        """Return True when transaction requires additional review."""

        if self.risk_score is None:
            return False

        return self.risk_score >= Decimal("0.60")

    def is_financially_significant(
        self,
        threshold: Decimal = Decimal("100000.00"),
    ) -> bool:
        """
        Check whether transaction exceeds a configurable
        financial significance threshold.
        """

        return abs(self.amount) >= threshold

    def __repr__(self) -> str:
        return (
            f"<Transaction("
            f"id={self.id}, "
            f"code='{self.transaction_code}', "
            f"type='{self.transaction_type}', "
            f"amount={self.amount}, "
            f"direction='{self.direction}', "
            f"date={self.transaction_date}"
            f")>"
        )