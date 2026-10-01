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


class BalanceSheet(Base):
    """
    Balance sheet snapshot for FinCo AI.

    Stores assets, liabilities, equity, liquidity metrics,
    financial health indicators, anomaly/risk scores, and
    AI-generated analysis for a reporting period.

    Detailed accounting calculations should live in:
        backend/app/financial/balance_sheet.py
    """

    __tablename__ = "balance_sheets"

    __table_args__ = (
        Index("ix_balance_sheet_company_id", "company_id"),
        Index("ix_balance_sheet_period_end", "period_end"),
        Index("ix_balance_sheet_fiscal_year", "fiscal_year"),
        Index("ix_balance_sheet_fiscal_period", "fiscal_period"),
        Index("ix_balance_sheet_status", "status"),
        Index("ix_balance_sheet_health_score", "financial_health_score"),
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
    # Reporting Period
    # ------------------------------------------------------------------

    period_start: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
    )

    period_end: Mapped[date] = mapped_column(
        Date,
        nullable=False,
    )

    fiscal_year: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    fiscal_period: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
    )
    # Example: Q1, Q2, Q3, Q4, FY, JAN, FEB, etc.

    currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="INR",
    )

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------

    cash_and_cash_equivalents: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    accounts_receivable: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    inventory: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    prepaid_expenses: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    other_current_assets: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_current_assets: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    property_plant_equipment: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    intangible_assets: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    long_term_investments: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    deferred_tax_assets: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    other_non_current_assets: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_non_current_assets: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_assets: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    # ------------------------------------------------------------------
    # Liabilities
    # ------------------------------------------------------------------

    accounts_payable: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    short_term_debt: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    accrued_expenses: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    tax_payable: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    other_current_liabilities: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_current_liabilities: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    long_term_debt: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    deferred_tax_liabilities: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    other_non_current_liabilities: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_non_current_liabilities: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_liabilities: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    # ------------------------------------------------------------------
    # Equity
    # ------------------------------------------------------------------

    share_capital: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    retained_earnings: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    additional_paid_in_capital: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    treasury_stock: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    other_equity: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    total_equity: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    # ------------------------------------------------------------------
    # Balance Sheet Validation
    # ------------------------------------------------------------------

    balance_check_difference: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    is_balanced: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Liquidity Metrics
    # ------------------------------------------------------------------

    working_capital: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=Decimal("0.00"),
    )

    current_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
    )

    quick_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
    )

    cash_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Leverage / Solvency Metrics
    # ------------------------------------------------------------------

    debt_to_equity: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
    )

    debt_to_assets: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
    )

    equity_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
    )

    solvency_ratio: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(12, 4),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Period-over-Period Metrics
    # ------------------------------------------------------------------

    previous_total_assets: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    previous_total_liabilities: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    previous_total_equity: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    assets_change: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    assets_change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    liabilities_change: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    liabilities_change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    equity_change: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    equity_change_percentage: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # Financial Health / Risk
    # ------------------------------------------------------------------

    financial_health_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    liquidity_risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    solvency_risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    leverage_risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    anomaly_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    risk_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 5),
        nullable=True,
    )

    # ------------------------------------------------------------------
    # AI Analysis
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

    # ------------------------------------------------------------------
    # Source / Status
    # ------------------------------------------------------------------

    source: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    source_document_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="DRAFT",
    )
    # DRAFT / VALIDATED / FINAL / RESTATED

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
        back_populates="balance_sheets",
    )

    source_document = relationship(
        "Document",
        back_populates="balance_sheets",
    )

    # ------------------------------------------------------------------
    # Business Logic
    # ------------------------------------------------------------------

    def calculate_total_current_assets(self) -> Decimal:
        """Calculate total current assets."""

        self.total_current_assets = (
            self.cash_and_cash_equivalents
            + self.accounts_receivable
            + self.inventory
            + self.prepaid_expenses
            + self.other_current_assets
        )

        return self.total_current_assets

    def calculate_total_non_current_assets(self) -> Decimal:
        """Calculate total non-current assets."""

        self.total_non_current_assets = (
            self.property_plant_equipment
            + self.intangible_assets
            + self.long_term_investments
            + self.deferred_tax_assets
            + self.other_non_current_assets
        )

        return self.total_non_current_assets

    def calculate_total_assets(self) -> Decimal:
        """Calculate total assets."""

        self.calculate_total_current_assets()
        self.calculate_total_non_current_assets()

        self.total_assets = (
            self.total_current_assets
            + self.total_non_current_assets
        )

        return self.total_assets

    def calculate_total_current_liabilities(self) -> Decimal:
        """Calculate total current liabilities."""

        self.total_current_liabilities = (
            self.accounts_payable
            + self.short_term_debt
            + self.accrued_expenses
            + self.tax_payable
            + self.other_current_liabilities
        )

        return self.total_current_liabilities

    def calculate_total_non_current_liabilities(self) -> Decimal:
        """Calculate total non-current liabilities."""

        self.total_non_current_liabilities = (
            self.long_term_debt
            + self.deferred_tax_liabilities
            + self.other_non_current_liabilities
        )

        return self.total_non_current_liabilities

    def calculate_total_liabilities(self) -> Decimal:
        """Calculate total liabilities."""

        self.calculate_total_current_liabilities()
        self.calculate_total_non_current_liabilities()

        self.total_liabilities = (
            self.total_current_liabilities
            + self.total_non_current_liabilities
        )

        return self.total_liabilities

    def calculate_total_equity(self) -> Decimal:
        """Calculate total shareholder equity."""

        self.total_equity = (
            self.share_capital
            + self.retained_earnings
            + self.additional_paid_in_capital
            + self.treasury_stock
            + self.other_equity
        )

        return self.total_equity

    def validate_balance(self) -> bool:
        """
        Validate the accounting equation:

            Assets = Liabilities + Equity
        """

        self.calculate_total_assets()
        self.calculate_total_liabilities()
        self.calculate_total_equity()

        self.balance_check_difference = (
            self.total_assets
            - self.total_liabilities
            - self.total_equity
        )

        self.is_balanced = (
            self.balance_check_difference == Decimal("0.00")
        )

        return self.is_balanced

    def calculate_liquidity_metrics(self) -> None:
        """Calculate liquidity ratios."""

        self.working_capital = (
            self.total_current_assets
            - self.total_current_liabilities
        )

        if self.total_current_liabilities > 0:
            self.current_ratio = (
                self.total_current_assets
                / self.total_current_liabilities
            )

            quick_assets = (
                self.cash_and_cash_equivalents
                + self.accounts_receivable
                + self.short_term_debt * Decimal("0")
            )

            self.quick_ratio = (
                quick_assets
                / self.total_current_liabilities
            )

            self.cash_ratio = (
                self.cash_and_cash_equivalents
                / self.total_current_liabilities
            )

    def calculate_leverage_metrics(self) -> None:
        """Calculate solvency and leverage ratios."""

        total_debt = (
            self.short_term_debt
            + self.long_term_debt
        )

        if self.total_equity > 0:
            self.debt_to_equity = (
                total_debt / self.total_equity
            )

        if self.total_assets > 0:
            self.debt_to_assets = (
                total_debt / self.total_assets
            )

            self.equity_ratio = (
                self.total_equity / self.total_assets
            )

            self.solvency_ratio = (
                self.total_assets
                / max(self.total_liabilities, Decimal("0.01"))
            )

    def calculate_period_changes(self) -> None:
        """Calculate changes against the previous reporting period."""

        if self.previous_total_assets is not None:
            self.assets_change = (
                self.total_assets
                - self.previous_total_assets
            )

            if self.previous_total_assets != 0:
                self.assets_change_percentage = (
                    self.assets_change
                    / abs(self.previous_total_assets)
                ) * Decimal("100")

        if self.previous_total_liabilities is not None:
            self.liabilities_change = (
                self.total_liabilities
                - self.previous_total_liabilities
            )

            if self.previous_total_liabilities != 0:
                self.liabilities_change_percentage = (
                    self.liabilities_change
                    / abs(self.previous_total_liabilities)
                ) * Decimal("100")

        if self.previous_total_equity is not None:
            self.equity_change = (
                self.total_equity
                - self.previous_total_equity
            )

            if self.previous_total_equity != 0:
                self.equity_change_percentage = (
                    self.equity_change
                    / abs(self.previous_total_equity)
                ) * Decimal("100")

    def calculate_metrics(self) -> None:
        """Calculate all deterministic balance-sheet metrics."""

        self.validate_balance()
        self.calculate_liquidity_metrics()
        self.calculate_leverage_metrics()
        self.calculate_period_changes()

    def is_financially_healthy(self) -> bool:
        """Return a basic liquidity/solvency health indicator."""

        if not self.is_balanced:
            return False

        if self.current_ratio is not None:
            if self.current_ratio < Decimal("1.00"):
                return False

        if self.debt_to_assets is not None:
            if self.debt_to_assets > Decimal("0.80"):
                return False

        return True

    def __repr__(self) -> str:
        return (
            f"<BalanceSheet("
            f"id={self.id}, "
            f"company_id={self.company_id}, "
            f"period_end={self.period_end}, "
            f"total_assets={self.total_assets}, "
            f"total_liabilities={self.total_liabilities}, "
            f"total_equity={self.total_equity}, "
            f"is_balanced={self.is_balanced}"
            f")>"
        )