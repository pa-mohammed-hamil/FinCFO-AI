"""
FinCo AI - Financial Data Normalization Engine

File:
    backend/app/financial/normalization.py

Purpose:
    Convert heterogeneous financial records into a canonical,
    validated, typed financial representation.

Responsibilities:
    - Normalize financial field names.
    - Normalize monetary values.
    - Normalize currencies.
    - Normalize dates and reporting periods.
    - Normalize accounting signs.
    - Normalize transaction types.
    - Normalize financial statement categories.
    - Normalize percentages and ratios.
    - Handle missing and malformed values.
    - Detect duplicate records.
    - Validate normalized financial observations.
    - Produce deterministic canonical records for downstream analytics.

Architecture:

    Raw Financial Data
          │
          ▼
    ingestion / extraction
          │
          ▼
    normalization.py
          │
          ├── Field Mapping
          ├── Type Conversion
          ├── Currency Normalization
          ├── Date Normalization
          ├── Sign Normalization
          ├── Category Normalization
          ├── Duplicate Detection
          └── Validation
          │
          ▼
    Canonical Financial Data
          │
    ┌─────┼────────┬────────┬────────┐
    ▼     ▼        ▼        ▼        ▼
  revenue expenses P&L   balance  cash flow
    │     │        │        │        │
    └─────┴────────┴────────┴────────┘
                    │
                    ▼
              KPIs / Margins
                    │
                    ▼
           Forecast / Fraud / RAG
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


# ============================================================================
# Constants
# ============================================================================

ZERO = Decimal("0")
ONE = Decimal("1")
ONE_HUNDRED = Decimal("100")

MONEY_QUANT = Decimal("0.01")
PERCENT_QUANT = Decimal("0.01")

DEFAULT_CURRENCY = "USD"

MISSING_VALUE_TOKENS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "-",
    "--",
    "unknown",
    "not available",
    "not_applicable",
}

POSITIVE_SIGN_TOKENS = {
    "positive",
    "credit",
    "cr",
    "income",
    "inflow",
    "receipt",
}

NEGATIVE_SIGN_TOKENS = {
    "negative",
    "debit",
    "dr",
    "expense",
    "outflow",
    "payment",
}


# ============================================================================
# Utility Functions
# ============================================================================


def to_decimal(
    value: Any,
    default: Decimal = ZERO,
) -> Decimal:
    """
    Convert a value to Decimal safely.

    Handles:
        - int
        - float
        - Decimal
        - numeric strings
        - accounting-style strings
        - commas
        - currency symbols
        - parentheses for negative values
        - percentage symbols
    """

    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    if isinstance(value, bool):
        return (
            Decimal("1")
            if value
            else ZERO
        )

    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return default

    text = str(value).strip()

    if not text:
        return default

    if text.lower() in MISSING_VALUE_TOKENS:
        return default

    negative = False

    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = (
        text.replace(",", "")
        .replace("$", "")
        .replace("€", "")
        .replace("£", "")
        .replace("₹", "")
        .replace("%", "")
        .strip()
    )

    # Handle explicit trailing negative sign.
    if text.endswith("-"):
        negative = True
        text = text[:-1]

    # Handle accounting sign.
    if text.startswith("-"):
        negative = True
        text = text[1:]

    try:
        result = Decimal(text)

        if negative:
            result = -abs(result)

        return result

    except (InvalidOperation, ValueError):
        return default


def round_money(
    value: Decimal,
) -> Decimal:
    """Round monetary values to two decimal places."""
    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_percent(
    value: Decimal,
) -> Decimal:
    """Round percentage values to two decimal places."""
    return value.quantize(
        PERCENT_QUANT,
        rounding=ROUND_HALF_UP,
    )


def safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely divide two Decimal values."""
    if denominator == ZERO:
        return default

    return numerator / denominator


def normalize_percent(
    value: Any,
    *,
    decimal_fraction: bool = False,
) -> Decimal:
    """
    Normalize percentage input.

    Examples:

        25      -> 25.00
        "25%"   -> 25.00
        0.25 with decimal_fraction=True -> 25.00
    """

    parsed = to_decimal(value)

    if decimal_fraction:
        parsed *= ONE_HUNDRED

    return round_percent(parsed)


def normalize_ratio(
    value: Any,
) -> Decimal:
    """Normalize a ratio without multiplying by 100."""
    return round_percent(
        to_decimal(value)
    )


def normalize_text(
    value: Any,
    *,
    default: str | None = None,
) -> str | None:
    """Normalize text values."""
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    return re.sub(
        r"\s+",
        " ",
        text,
    )


def normalize_key(
    value: Any,
) -> str:
    """
    Convert arbitrary field names into canonical lookup keys.

    Examples:
        "Gross Profit" -> "gross_profit"
        "gross-profit" -> "gross_profit"
        " Gross Profit " -> "gross_profit"
    """

    text = normalize_text(
        value,
        default="",
    ) or ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def parse_date(
    value: Any,
) -> date | None:
    """Parse common financial date formats."""

    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    text = str(value).strip()

    if not text:
        return None

    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
    )

    for fmt in formats:
        try:
            return datetime.strptime(
                text,
                fmt,
            ).date()
        except ValueError:
            continue

    # ISO datetime fallback.
    try:
        return datetime.fromisoformat(
            text.replace(
                "Z",
                "+00:00",
            )
        ).date()
    except ValueError:
        return None


def normalize_currency(
    value: Any,
    default: str = DEFAULT_CURRENCY,
) -> str:
    """Normalize a currency code or common currency symbol."""

    if value is None:
        return default

    text = str(value).strip().upper()

    mapping = {
        "$": "USD",
        "US$": "USD",
        "USD$": "USD",
        "€": "EUR",
        "£": "GBP",
        "₹": "INR",
        "RS": "INR",
        "RS.": "INR",
        "RMB": "CNY",
        "¥": "JPY",
    }

    return mapping.get(
        text,
        text if text else default,
    )


def normalize_sign(
    value: Any,
    *,
    sign_hint: Any = None,
    category: Any = None,
) -> Decimal:
    """
    Normalize a financial amount's sign.

    Explicit negative/positive values are preserved.

    Sign hints such as:
        debit / credit
        expense / income
        inflow / outflow

    can be used when the raw amount itself is unsigned.
    """

    amount = to_decimal(value)

    if amount == ZERO:
        return ZERO

    if amount < ZERO:
        return amount

    hint = normalize_key(sign_hint)

    if hint in NEGATIVE_SIGN_TOKENS:
        return -abs(amount)

    if hint in POSITIVE_SIGN_TOKENS:
        return abs(amount)

    category_key = normalize_key(category)

    if category_key in {
        "expense",
        "cost",
        "cogs",
        "operating_expense",
        "interest_expense",
        "tax_expense",
        "liability_payment",
        "cash_outflow",
    }:
        return -abs(amount)

    return amount


def canonical_hash(
    values: Sequence[Any],
) -> str:
    """
    Generate a deterministic SHA-256 fingerprint.
    """

    payload = "|".join(
        "" if value is None else str(value)
        for value in values
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


# ============================================================================
# Enums
# ============================================================================


class FinancialDataType(str, Enum):
    """Canonical financial data categories."""

    TRANSACTION = "transaction"
    REVENUE = "revenue"
    EXPENSE = "expense"
    PROFIT_LOSS = "profit_loss"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    BUDGET = "budget"
    FORECAST = "forecast"
    KPI = "kpi"
    UNKNOWN = "unknown"


class AccountType(str, Enum):
    """Accounting account classifications."""

    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    REVENUE = "revenue"
    EXPENSE = "expense"
    COST_OF_GOODS_SOLD = "cost_of_goods_sold"
    OTHER = "other"
    UNKNOWN = "unknown"


class TransactionType(str, Enum):
    """Normalized transaction types."""

    SALE = "sale"
    PURCHASE = "purchase"
    EXPENSE = "expense"
    REFUND = "refund"
    PAYMENT = "payment"
    RECEIPT = "receipt"
    TRANSFER = "transfer"
    PAYROLL = "payroll"
    TAX = "tax"
    INTEREST = "interest"
    FEE = "fee"
    ADJUSTMENT = "adjustment"
    OTHER = "other"
    UNKNOWN = "unknown"


class StatementType(str, Enum):
    """Financial statement classifications."""

    PROFIT_LOSS = "profit_loss"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    UNKNOWN = "unknown"


class CashFlowType(str, Enum):
    """Cash flow classifications."""

    OPERATING = "operating"
    INVESTING = "investing"
    FINANCING = "financing"
    UNKNOWN = "unknown"


class NormalizationSeverity(str, Enum):
    """Normalization issue severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Field Aliases
# ============================================================================


FIELD_ALIASES: dict[str, set[str]] = {
    "date": {
        "date",
        "transaction_date",
        "posting_date",
        "entry_date",
        "invoice_date",
        "document_date",
        "period_date",
        "as_of_date",
    },
    "description": {
        "description",
        "desc",
        "details",
        "narration",
        "memo",
        "particulars",
        "transaction_description",
    },
    "reference": {
        "reference",
        "reference_number",
        "ref",
        "ref_no",
        "transaction_id",
        "transaction_number",
        "invoice_number",
        "invoice_no",
        "document_number",
    },
    "account": {
        "account",
        "account_name",
        "ledger",
        "ledger_name",
        "gl_account",
        "general_ledger",
    },
    "account_type": {
        "account_type",
        "type",
        "account_category",
        "category",
    },
    "amount": {
        "amount",
        "value",
        "total",
        "net_amount",
        "transaction_amount",
        "balance",
    },
    "debit": {
        "debit",
        "debit_amount",
        "dr",
        "debits",
    },
    "credit": {
        "credit",
        "credit_amount",
        "cr",
        "credits",
    },
    "currency": {
        "currency",
        "currency_code",
        "ccy",
        "currency_symbol",
    },
    "customer": {
        "customer",
        "customer_name",
        "client",
        "client_name",
        "buyer",
    },
    "supplier": {
        "supplier",
        "supplier_name",
        "vendor",
        "vendor_name",
        "seller",
    },
    "department": {
        "department",
        "department_name",
        "dept",
    },
    "cost_center": {
        "cost_center",
        "cost_centre",
        "cost_center_name",
    },
    "quantity": {
        "quantity",
        "qty",
        "units",
        "units_sold",
    },
    "unit_price": {
        "unit_price",
        "price",
        "selling_price",
        "rate",
    },
    "revenue": {
        "revenue",
        "sales",
        "sales_revenue",
        "net_sales",
        "turnover",
        "income",
    },
    "gross_profit": {
        "gross_profit",
        "gross_income",
    },
    "operating_profit": {
        "operating_profit",
        "ebit",
    },
    "net_profit": {
        "net_profit",
        "net_income",
        "profit_after_tax",
        "pat",
    },
    "cost_of_goods_sold": {
        "cost_of_goods_sold",
        "cogs",
        "cost_of_sales",
        "cost_of_revenue",
        "direct_cost",
    },
    "operating_expenses": {
        "operating_expenses",
        "opex",
        "operating_costs",
        "operating_expense",
    },
    "assets": {
        "assets",
        "total_assets",
    },
    "liabilities": {
        "liabilities",
        "total_liabilities",
    },
    "equity": {
        "equity",
        "shareholders_equity",
        "owners_equity",
        "net_assets",
    },
    "cash": {
        "cash",
        "cash_and_equivalents",
        "cash_balance",
        "bank_balance",
    },
    "accounts_receivable": {
        "accounts_receivable",
        "receivables",
        "trade_receivables",
        "ar",
    },
    "accounts_payable": {
        "accounts_payable",
        "payables",
        "trade_payables",
        "ap",
    },
    "inventory": {
        "inventory",
        "stock",
        "inventory_balance",
    },
    "debt": {
        "debt",
        "borrowings",
        "loans",
        "total_debt",
    },
}


# Build reverse lookup once.
ALIAS_TO_CANONICAL: dict[str, str] = {}

for canonical_name, aliases in FIELD_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[
            normalize_key(alias)
        ] = canonical_name


# ============================================================================
# Normalization Issues
# ============================================================================


@dataclass(slots=True)
class NormalizationIssue:
    """Issue generated during normalization."""

    field: str | None

    severity: NormalizationSeverity

    code: str

    message: str

    original_value: Any = None

    normalized_value: Any = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "original_value": self.original_value,
            "normalized_value": self.normalized_value,
            "metadata": self.metadata,
        }


# ============================================================================
# Canonical Models
# ============================================================================


@dataclass(slots=True)
class NormalizedTransaction:
    """
    Canonical transaction record.

    This model is intentionally independent from the database ORM.
    """

    transaction_id: str | None = None

    transaction_date: date | None = None

    description: str | None = None

    reference: str | None = None

    account: str | None = None

    account_type: AccountType = AccountType.UNKNOWN

    transaction_type: TransactionType = (
        TransactionType.UNKNOWN
    )

    amount: Decimal = ZERO

    debit: Decimal = ZERO

    credit: Decimal = ZERO

    currency: str = DEFAULT_CURRENCY

    customer: str | None = None

    supplier: str | None = None

    department: str | None = None

    cost_center: str | None = None

    quantity: Decimal = ZERO

    unit_price: Decimal = ZERO

    source: str | None = None

    source_row: int | None = None

    fingerprint: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def generate_fingerprint(self) -> str:
        """Generate deterministic transaction fingerprint."""

        self.fingerprint = canonical_hash(
            [
                self.transaction_date,
                self.reference,
                self.account,
                self.description,
                self.amount,
                self.debit,
                self.credit,
                self.currency,
            ]
        )

        return self.fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "transaction_date": (
                self.transaction_date.isoformat()
                if self.transaction_date
                else None
            ),
            "description": self.description,
            "reference": self.reference,
            "account": self.account,
            "account_type": self.account_type.value,
            "transaction_type": (
                self.transaction_type.value
            ),
            "amount": str(self.amount),
            "debit": str(self.debit),
            "credit": str(self.credit),
            "currency": self.currency,
            "customer": self.customer,
            "supplier": self.supplier,
            "department": self.department,
            "cost_center": self.cost_center,
            "quantity": str(self.quantity),
            "unit_price": str(self.unit_price),
            "source": self.source,
            "source_row": self.source_row,
            "fingerprint": self.fingerprint,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class NormalizedFinancialPeriod:
    """
    Canonical period-level financial observation.
    """

    period: str | None = None

    start_date: date | None = None

    end_date: date | None = None

    currency: str = DEFAULT_CURRENCY

    revenue: Decimal = ZERO

    cost_of_goods_sold: Decimal = ZERO

    gross_profit: Decimal = ZERO

    operating_expenses: Decimal = ZERO

    operating_profit: Decimal = ZERO

    interest_expense: Decimal = ZERO

    tax_expense: Decimal = ZERO

    net_profit: Decimal = ZERO

    assets: Decimal = ZERO

    liabilities: Decimal = ZERO

    equity: Decimal = ZERO

    debt: Decimal = ZERO

    cash: Decimal = ZERO

    receivables: Decimal = ZERO

    inventory: Decimal = ZERO

    payables: Decimal = ZERO

    operating_cash_flow: Decimal = ZERO

    investing_cash_flow: Decimal = ZERO

    financing_cash_flow: Decimal = ZERO

    free_cash_flow: Decimal = ZERO

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def derive(self) -> None:
        """Derive missing profit and cash-flow metrics."""

        if (
            self.gross_profit == ZERO
            and (
                self.revenue != ZERO
                or self.cost_of_goods_sold != ZERO
            )
        ):
            self.gross_profit = (
                self.revenue
                - self.cost_of_goods_sold
            )

        if (
            self.operating_profit == ZERO
            and (
                self.gross_profit != ZERO
                or self.operating_expenses != ZERO
            )
        ):
            self.operating_profit = (
                self.gross_profit
                - self.operating_expenses
            )

        if (
            self.net_profit == ZERO
            and (
                self.operating_profit != ZERO
                or self.interest_expense != ZERO
                or self.tax_expense != ZERO
            )
        ):
            self.net_profit = (
                self.operating_profit
                - self.interest_expense
                - self.tax_expense
            )

        if self.free_cash_flow == ZERO:
            self.free_cash_flow = (
                self.operating_cash_flow
                + self.investing_cash_flow
            )

    def to_dict(self) -> dict[str, Any]:
        monetary_fields = (
            "revenue",
            "cost_of_goods_sold",
            "gross_profit",
            "operating_expenses",
            "operating_profit",
            "interest_expense",
            "tax_expense",
            "net_profit",
            "assets",
            "liabilities",
            "equity",
            "debt",
            "cash",
            "receivables",
            "inventory",
            "payables",
            "operating_cash_flow",
            "investing_cash_flow",
            "financing_cash_flow",
            "free_cash_flow",
        )

        result: dict[str, Any] = {
            "period": self.period,
            "start_date": (
                self.start_date.isoformat()
                if self.start_date
                else None
            ),
            "end_date": (
                self.end_date.isoformat()
                if self.end_date
                else None
            ),
            "currency": self.currency,
            "metadata": self.metadata,
        }

        for field_name in monetary_fields:
            result[field_name] = str(
                getattr(
                    self,
                    field_name,
                )
            )

        return result


@dataclass(slots=True)
class NormalizationResult:
    """
    Complete normalization result.
    """

    transactions: list[NormalizedTransaction] = field(
        default_factory=list
    )

    periods: list[NormalizedFinancialPeriod] = field(
        default_factory=list
    )

    issues: list[NormalizationIssue] = field(
        default_factory=list
    )

    duplicates: list[str] = field(
        default_factory=list
    )

    source: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def has_errors(self) -> bool:
        return any(
            issue.severity
            in {
                NormalizationSeverity.ERROR,
                NormalizationSeverity.CRITICAL,
            }
            for issue in self.issues
        )

    @property
    def error_count(self) -> int:
        return sum(
            1
            for issue in self.issues
            if issue.severity
            in {
                NormalizationSeverity.ERROR,
                NormalizationSeverity.CRITICAL,
            }
        )

    @property
    def warning_count(self) -> int:
        return sum(
            1
            for issue in self.issues
            if issue.severity
            == NormalizationSeverity.WARNING
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transactions": [
                item.to_dict()
                for item in self.transactions
            ],
            "periods": [
                item.to_dict()
                for item in self.periods
            ],
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "duplicates": self.duplicates,
            "source": self.source,
            "metadata": self.metadata,
            "has_errors": self.has_errors,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


# ============================================================================
# Financial Normalizer
# ============================================================================


class FinancialNormalizer:
    """
    Stateless canonical financial-data normalization engine.
    """

    def __init__(
        self,
        *,
        default_currency: str = DEFAULT_CURRENCY,
        strict: bool = False,
        infer_transaction_type: bool = True,
        normalize_negative_expenses: bool = True,
    ) -> None:
        self.default_currency = normalize_currency(
            default_currency
        )

        self.strict = strict

        self.infer_transaction_type = (
            infer_transaction_type
        )

        self.normalize_negative_expenses = (
            normalize_negative_expenses
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        records: Iterable[
            Mapping[str, Any]
            | NormalizedTransaction
        ],
        *,
        source: str | None = None,
    ) -> NormalizationResult:
        """
        Normalize transaction-like records.
        """

        result = NormalizationResult(
            source=source
        )

        seen: set[str] = set()

        for index, raw in enumerate(records, start=1):
            if isinstance(
                raw,
                NormalizedTransaction,
            ):
                transaction = raw

                if transaction.fingerprint is None:
                    transaction.generate_fingerprint()

            else:
                transaction, issues = (
                    self.normalize_transaction(
                        raw,
                        source=source,
                        source_row=index,
                    )
                )

                result.issues.extend(
                    issues
                )

            if transaction.fingerprint is None:
                transaction.generate_fingerprint()

            if transaction.fingerprint in seen:
                result.duplicates.append(
                    transaction.fingerprint
                )

                result.issues.append(
                    NormalizationIssue(
                        field=None,
                        severity=(
                            NormalizationSeverity.WARNING
                        ),
                        code="duplicate_record",
                        message=(
                            "Duplicate financial record detected."
                        ),
                        normalized_value=(
                            transaction.fingerprint
                        ),
                        metadata={
                            "source_row": index,
                        },
                    )
                )

                continue

            seen.add(
                transaction.fingerprint
            )

            result.transactions.append(
                transaction
            )

        return result

    # ------------------------------------------------------------------
    # Transaction Normalization
    # ------------------------------------------------------------------

    def normalize_transaction(
        self,
        raw: Mapping[str, Any],
        *,
        source: str | None = None,
        source_row: int | None = None,
    ) -> tuple[
        NormalizedTransaction,
        list[NormalizationIssue],
    ]:
        """Normalize one transaction record."""

        canonical = self.canonicalize_keys(
            raw
        )

        issues: list[NormalizationIssue] = []

        transaction_date = parse_date(
            canonical.get("date")
        )

        if (
            canonical.get("date") is not None
            and transaction_date is None
        ):
            issues.append(
                NormalizationIssue(
                    field="date",
                    severity=(
                        NormalizationSeverity.WARNING
                    ),
                    code="invalid_date",
                    message=(
                        "Could not parse transaction date."
                    ),
                    original_value=canonical.get(
                        "date"
                    ),
                )
            )

        currency = normalize_currency(
            canonical.get("currency"),
            self.default_currency,
        )

        account_type = self.normalize_account_type(
            canonical.get("account_type")
            or canonical.get("category")
        )

        transaction_type = self.normalize_transaction_type(
            canonical.get("transaction_type")
            or canonical.get("type")
        )

        amount = self._resolve_amount(
            canonical,
            account_type=account_type,
            transaction_type=transaction_type,
            issues=issues,
        )

        debit = to_decimal(
            canonical.get("debit")
        )

        credit = to_decimal(
            canonical.get("credit")
        )

        if debit == ZERO and credit == ZERO:
            if amount >= ZERO:
                credit = amount
            else:
                debit = abs(amount)

        elif debit != ZERO and credit != ZERO:
            issues.append(
                NormalizationIssue(
                    field="amount",
                    severity=(
                        NormalizationSeverity.WARNING
                    ),
                    code="both_debit_and_credit",
                    message=(
                        "Both debit and credit values were supplied."
                    ),
                    original_value={
                        "debit": canonical.get(
                            "debit"
                        ),
                        "credit": canonical.get(
                            "credit"
                        ),
                    },
                )
            )

        description = normalize_text(
            canonical.get("description")
        )

        reference = normalize_text(
            canonical.get("reference")
        )

        account = normalize_text(
            canonical.get("account")
        )

        customer = normalize_text(
            canonical.get("customer")
        )

        supplier = normalize_text(
            canonical.get("supplier")
        )

        department = normalize_text(
            canonical.get("department")
        )

        cost_center = normalize_text(
            canonical.get("cost_center")
        )

        quantity = to_decimal(
            canonical.get("quantity")
        )

        unit_price = to_decimal(
            canonical.get("unit_price")
        )

        transaction_id = normalize_text(
            canonical.get("transaction_id")
            or reference
        )

        transaction = NormalizedTransaction(
            transaction_id=transaction_id,
            transaction_date=transaction_date,
            description=description,
            reference=reference,
            account=account,
            account_type=account_type,
            transaction_type=transaction_type,
            amount=round_money(
                amount
            ),
            debit=round_money(
                abs(debit)
            ),
            credit=round_money(
                abs(credit)
            ),
            currency=currency,
            customer=customer,
            supplier=supplier,
            department=department,
            cost_center=cost_center,
            quantity=quantity,
            unit_price=round_money(
                unit_price
            ),
            source=source,
            source_row=source_row,
            metadata=self._extract_metadata(
                canonical
            ),
        )

        transaction.generate_fingerprint()

        issues.extend(
            self.validate_transaction(
                transaction
            )
        )

        return transaction, issues

    # ------------------------------------------------------------------
    # Amount Resolution
    # ------------------------------------------------------------------

    def _resolve_amount(
        self,
        data: Mapping[str, Any],
        *,
        account_type: AccountType,
        transaction_type: TransactionType,
        issues: list[NormalizationIssue],
    ) -> Decimal:
        amount_raw = data.get(
            "amount"
        )

        if amount_raw is not None:
            amount = to_decimal(
                amount_raw
            )

            sign_hint = (
                data.get("sign")
                or data.get("direction")
                or data.get("debit_credit")
            )

            return normalize_sign(
                amount,
                sign_hint=sign_hint,
                category=(
                    transaction_type.value
                    if transaction_type
                    else account_type.value
                ),
            )

        debit = to_decimal(
            data.get("debit")
        )

        credit = to_decimal(
            data.get("credit")
        )

        if debit != ZERO or credit != ZERO:
            return credit - debit

        # No monetary field supplied.
        issues.append(
            NormalizationIssue(
                field="amount",
                severity=(
                    NormalizationSeverity.WARNING
                ),
                code="missing_amount",
                message=(
                    "No amount, debit, or credit value was supplied."
                ),
            )
        )

        return ZERO

    # ------------------------------------------------------------------
    # Key Canonicalization
    # ------------------------------------------------------------------

    def canonicalize_keys(
        self,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Map raw field names to canonical names.
        Unknown fields are preserved.
        """

        result: dict[str, Any] = {}

        for raw_key, value in data.items():
            normalized_key = normalize_key(
                raw_key
            )

            canonical = ALIAS_TO_CANONICAL.get(
                normalized_key,
                normalized_key,
            )

            # Preserve first meaningful value.
            if canonical not in result:
                result[canonical] = value
                continue

            existing = result[canonical]

            if (
                existing is None
                or str(existing).strip() == ""
            ):
                result[canonical] = value

        return result

    # ------------------------------------------------------------------
    # Enum Normalization
    # ------------------------------------------------------------------

    def normalize_account_type(
        self,
        value: Any,
    ) -> AccountType:
        key = normalize_key(value)

        mapping = {
            "asset": AccountType.ASSET,
            "assets": AccountType.ASSET,
            "liability": AccountType.LIABILITY,
            "liabilities": AccountType.LIABILITY,
            "equity": AccountType.EQUITY,
            "capital": AccountType.EQUITY,
            "revenue": AccountType.REVENUE,
            "income": AccountType.REVENUE,
            "sales": AccountType.REVENUE,
            "expense": AccountType.EXPENSE,
            "expenses": AccountType.EXPENSE,
            "cost": AccountType.EXPENSE,
            "cogs": AccountType.COST_OF_GOODS_SOLD,
            "cost_of_goods_sold": (
                AccountType.COST_OF_GOODS_SOLD
            ),
            "cost_of_sales": (
                AccountType.COST_OF_GOODS_SOLD
            ),
            "other": AccountType.OTHER,
        }

        return mapping.get(
            key,
            AccountType.UNKNOWN,
        )

    def normalize_transaction_type(
        self,
        value: Any,
    ) -> TransactionType:
        key = normalize_key(value)

        mapping = {
            "sale": TransactionType.SALE,
            "sales": TransactionType.SALE,
            "revenue": TransactionType.SALE,
            "purchase": TransactionType.PURCHASE,
            "purchases": TransactionType.PURCHASE,
            "expense": TransactionType.EXPENSE,
            "expenses": TransactionType.EXPENSE,
            "refund": TransactionType.REFUND,
            "return": TransactionType.REFUND,
            "payment": TransactionType.PAYMENT,
            "receipt": TransactionType.RECEIPT,
            "transfer": TransactionType.TRANSFER,
            "payroll": TransactionType.PAYROLL,
            "salary": TransactionType.PAYROLL,
            "tax": TransactionType.TAX,
            "interest": TransactionType.INTEREST,
            "fee": TransactionType.FEE,
            "adjustment": TransactionType.ADJUSTMENT,
            "other": TransactionType.OTHER,
        }

        return mapping.get(
            key,
            TransactionType.UNKNOWN,
        )

    def normalize_statement_type(
        self,
        value: Any,
    ) -> StatementType:
        key = normalize_key(value)

        mapping = {
            "profit_loss": StatementType.PROFIT_LOSS,
            "profit_and_loss": StatementType.PROFIT_LOSS,
            "pnl": StatementType.PROFIT_LOSS,
            "income_statement": StatementType.PROFIT_LOSS,
            "balance_sheet": StatementType.BALANCE_SHEET,
            "balance": StatementType.BALANCE_SHEET,
            "cash_flow": StatementType.CASH_FLOW,
            "cashflow": StatementType.CASH_FLOW,
        }

        return mapping.get(
            key,
            StatementType.UNKNOWN,
        )

    def normalize_cash_flow_type(
        self,
        value: Any,
    ) -> CashFlowType:
        key = normalize_key(value)

        mapping = {
            "operating": CashFlowType.OPERATING,
            "operations": CashFlowType.OPERATING,
            "operating_activities": (
                CashFlowType.OPERATING
            ),
            "investing": CashFlowType.INVESTING,
            "investing_activities": (
                CashFlowType.INVESTING
            ),
            "financing": CashFlowType.FINANCING,
            "financing_activities": (
                CashFlowType.FINANCING
            ),
        }

        return mapping.get(
            key,
            CashFlowType.UNKNOWN,
        )

    # ------------------------------------------------------------------
    # Financial Period Normalization
    # ------------------------------------------------------------------

    def normalize_period(
        self,
        raw: Mapping[str, Any],
    ) -> tuple[
        NormalizedFinancialPeriod,
        list[NormalizationIssue],
    ]:
        """Normalize period-level financial data."""

        canonical = self.canonicalize_keys(
            raw
        )

        issues: list[NormalizationIssue] = []

        start_date = parse_date(
            canonical.get(
                "start_date"
            )
            or canonical.get(
                "period_start"
            )
        )

        end_date = parse_date(
            canonical.get(
                "end_date"
            )
            or canonical.get(
                "period_end"
            )
        )

        if start_date is None:
            start_date = parse_date(
                canonical.get("date")
            )

        if end_date is None:
            end_date = start_date

        currency = normalize_currency(
            canonical.get("currency"),
            self.default_currency,
        )

        period = normalize_text(
            canonical.get("period")
        )

        normalized = NormalizedFinancialPeriod(
            period=period,
            start_date=start_date,
            end_date=end_date,
            currency=currency,
            revenue=self._money(
                canonical.get("revenue")
            ),
            cost_of_goods_sold=self._money(
                canonical.get(
                    "cost_of_goods_sold"
                )
            ),
            gross_profit=self._money(
                canonical.get("gross_profit")
            ),
            operating_expenses=self._money(
                canonical.get(
                    "operating_expenses"
                )
            ),
            operating_profit=self._money(
                canonical.get(
                    "operating_profit"
                )
            ),
            interest_expense=self._money(
                canonical.get(
                    "interest_expense"
                )
            ),
            tax_expense=self._money(
                canonical.get(
                    "tax_expense"
                )
            ),
            net_profit=self._money(
                canonical.get("net_profit")
            ),
            assets=self._money(
                canonical.get("assets")
            ),
            liabilities=self._money(
                canonical.get("liabilities")
            ),
            equity=self._money(
                canonical.get("equity")
            ),
            debt=self._money(
                canonical.get("debt")
            ),
            cash=self._money(
                canonical.get("cash")
            ),
            receivables=self._money(
                canonical.get(
                    "accounts_receivable"
                )
            ),
            inventory=self._money(
                canonical.get("inventory")
            ),
            payables=self._money(
                canonical.get(
                    "accounts_payable"
                )
            ),
            operating_cash_flow=self._money(
                canonical.get(
                    "operating_cash_flow"
                )
            ),
            investing_cash_flow=self._money(
                canonical.get(
                    "investing_cash_flow"
                )
            ),
            financing_cash_flow=self._money(
                canonical.get(
                    "financing_cash_flow"
                )
            ),
            free_cash_flow=self._money(
                canonical.get(
                    "free_cash_flow"
                ),
            ),
            metadata=self._extract_metadata(
                canonical
            ),
        )

        normalized.derive()

        issues.extend(
            self.validate_period(
                normalized
            )
        )

        return normalized, issues

    # ------------------------------------------------------------------
    # Batch Period Normalization
    # ------------------------------------------------------------------

    def normalize_periods(
        self,
        records: Iterable[
            Mapping[str, Any]
        ],
    ) -> NormalizationResult:
        """Normalize period-level financial records."""

        result = NormalizationResult()

        for raw in records:
            period, issues = (
                self.normalize_period(
                    raw
                )
            )

            result.periods.append(
                period
            )

            result.issues.extend(
                issues
            )

        return result

    # ------------------------------------------------------------------
    # Money Normalization
    # ------------------------------------------------------------------

    def _money(
        self,
        value: Any,
    ) -> Decimal:
        return round_money(
            to_decimal(value)
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _extract_metadata(
        self,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Preserve non-canonical source fields."""

        canonical_fields = set(
            ALIAS_TO_CANONICAL.values()
        )

        metadata: dict[str, Any] = {}

        for key, value in data.items():
            if key not in canonical_fields:
                metadata[key] = value

        return metadata

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_transaction(
        self,
        transaction: NormalizedTransaction,
    ) -> list[NormalizationIssue]:
        """Validate normalized transaction."""

        issues: list[NormalizationIssue] = []

        if transaction.transaction_date is None:
            issues.append(
                NormalizationIssue(
                    field="transaction_date",
                    severity=(
                        NormalizationSeverity.WARNING
                    ),
                    code="missing_transaction_date",
                    message=(
                        "Transaction does not contain a valid date."
                    ),
                )
            )

        if (
            transaction.amount == ZERO
            and transaction.debit == ZERO
            and transaction.credit == ZERO
        ):
            issues.append(
                NormalizationIssue(
                    field="amount",
                    severity=(
                        NormalizationSeverity.WARNING
                    ),
                    code="zero_amount",
                    message=(
                        "Transaction amount is zero."
                    ),
                )
            )

        if (
            transaction.debit > ZERO
            and transaction.credit > ZERO
        ):
            issues.append(
                NormalizationIssue(
                    field="amount",
                    severity=(
                        NormalizationSeverity.ERROR
                    ),
                    code="invalid_debit_credit",
                    message=(
                        "Transaction contains both debit and credit values."
                    ),
                )
            )

        if (
            transaction.quantity < ZERO
        ):
            issues.append(
                NormalizationIssue(
                    field="quantity",
                    severity=(
                        NormalizationSeverity.WARNING
                    ),
                    code="negative_quantity",
                    message=(
                        "Transaction quantity is negative."
                    ),
                    normalized_value=str(
                        transaction.quantity
                    ),
                )
            )

        return issues

    def validate_period(
        self,
        period: NormalizedFinancialPeriod,
    ) -> list[NormalizationIssue]:
        """Validate period-level financial data."""

        issues: list[NormalizationIssue] = []

        if (
            period.start_date is not None
            and period.end_date is not None
            and period.start_date
            > period.end_date
        ):
            issues.append(
                NormalizationIssue(
                    field="period",
                    severity=(
                        NormalizationSeverity.ERROR
                    ),
                    code="invalid_period_range",
                    message=(
                        "Period start date occurs after end date."
                    ),
                )
            )

        # Accounting equation check.
        if (
            period.assets != ZERO
            and (
                period.liabilities
                + period.equity
            ) != ZERO
        ):
            difference = (
                period.assets
                - (
                    period.liabilities
                    + period.equity
                )
            )

            tolerance = Decimal("0.01")

            if abs(difference) > tolerance:
                issues.append(
                    NormalizationIssue(
                        field="balance_sheet",
                        severity=(
                            NormalizationSeverity.WARNING
                        ),
                        code="balance_sheet_imbalance",
                        message=(
                            "Assets do not equal liabilities plus equity "
                            "within the configured tolerance."
                        ),
                        normalized_value=str(
                            difference
                        ),
                    )
                )

        # Profit consistency.
        expected_gross = (
            period.revenue
            - period.cost_of_goods_sold
        )

        if (
            period.gross_profit != ZERO
            and abs(
                period.gross_profit
                - expected_gross
            ) > Decimal("0.01")
        ):
            issues.append(
                NormalizationIssue(
                    field="gross_profit",
                    severity=(
                        NormalizationSeverity.WARNING
                    ),
                    code="gross_profit_mismatch",
                    message=(
                        "Reported gross profit does not equal "
                        "revenue minus COGS."
                    ),
                    normalized_value=str(
                        period.gross_profit
                    ),
                    metadata={
                        "calculated": str(
                            expected_gross
                        )
                    },
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Sign Normalization
    # ------------------------------------------------------------------

    def normalize_expense(
        self,
        value: Any,
    ) -> Decimal:
        """
        Normalize an expense as a positive magnitude.

        Financial analytics modules generally benefit from expenses being
        represented as positive magnitudes while profit calculations
        explicitly subtract them.
        """

        amount = abs(
            to_decimal(value)
        )

        return round_money(
            amount
        )

    def normalize_revenue(
        self,
        value: Any,
    ) -> Decimal:
        """Normalize revenue as a positive magnitude."""
        return round_money(
            abs(
                to_decimal(value)
            )
        )

    def normalize_asset(
        self,
        value: Any,
    ) -> Decimal:
        """Normalize asset balance as a positive magnitude."""
        return round_money(
            abs(
                to_decimal(value)
            )
        )

    def normalize_liability(
        self,
        value: Any,
    ) -> Decimal:
        """Normalize liability balance as a positive magnitude."""
        return round_money(
            abs(
                to_decimal(value)
            )
        )

    # ------------------------------------------------------------------
    # Statement Aggregation
    # ------------------------------------------------------------------

    def aggregate_transactions(
        self,
        transactions: Sequence[
            NormalizedTransaction
        ],
    ) -> NormalizedFinancialPeriod:
        """
        Aggregate normalized transactions into a period-level
        financial representation.
        """

        period = NormalizedFinancialPeriod()

        for transaction in transactions:
            amount = transaction.amount

            account_type = (
                transaction.account_type
            )

            transaction_type = (
                transaction.transaction_type
            )

            if (
                transaction_type
                == TransactionType.SALE
                or account_type
                == AccountType.REVENUE
            ):
                period.revenue += abs(
                    amount
                )

            elif (
                transaction_type
                in {
                    TransactionType.PURCHASE,
                }
                or account_type
                == AccountType.COST_OF_GOODS_SOLD
            ):
                period.cost_of_goods_sold += abs(
                    amount
                )

            elif (
                transaction_type
                == TransactionType.EXPENSE
                or account_type
                == AccountType.EXPENSE
            ):
                period.operating_expenses += abs(
                    amount
                )

            elif (
                transaction_type
                == TransactionType.TAX
            ):
                period.tax_expense += abs(
                    amount
                )

            elif (
                transaction_type
                == TransactionType.INTEREST
            ):
                period.interest_expense += abs(
                    amount
                )

        period.revenue = round_money(
            period.revenue
        )

        period.cost_of_goods_sold = round_money(
            period.cost_of_goods_sold
        )

        period.operating_expenses = round_money(
            period.operating_expenses
        )

        period.tax_expense = round_money(
            period.tax_expense
        )

        period.interest_expense = round_money(
            period.interest_expense
        )

        period.derive()

        return period

    # ------------------------------------------------------------------
    # Duplicate Detection
    # ------------------------------------------------------------------

    def find_duplicates(
        self,
        transactions: Sequence[
            NormalizedTransaction
        ],
    ) -> list[
        tuple[
            str,
            list[int],
        ]
    ]:
        """
        Find duplicate transaction fingerprints.

        Returns:
            [
                (fingerprint, [row_indexes])
            ]
        """

        fingerprints: dict[
            str,
            list[int],
        ] = {}

        for index, transaction in enumerate(
            transactions
        ):
            fingerprint = (
                transaction.fingerprint
                or transaction.generate_fingerprint()
            )

            fingerprints.setdefault(
                fingerprint,
                [],
            ).append(index)

        return [
            (
                fingerprint,
                indexes,
            )
            for fingerprint, indexes
            in fingerprints.items()
            if len(indexes) > 1
        ]

    # ------------------------------------------------------------------
    # Field Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def canonical_field_name(
        value: Any,
    ) -> str:
        """Return canonical field name."""
        normalized = normalize_key(
            value
        )

        return ALIAS_TO_CANONICAL.get(
            normalized,
            normalized,
        )

    @staticmethod
    def supported_fields() -> list[str]:
        """Return canonical supported field names."""
        return sorted(
            set(
                ALIAS_TO_CANONICAL.values()
            )
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def normalize_financial_data(
    records: Iterable[
        Mapping[str, Any]
    ],
    *,
    source: str | None = None,
    default_currency: str = DEFAULT_CURRENCY,
) -> NormalizationResult:
    """Normalize financial transaction records."""

    normalizer = FinancialNormalizer(
        default_currency=default_currency
    )

    return normalizer.normalize(
        records,
        source=source,
    )


def normalize_transaction(
    record: Mapping[str, Any],
    *,
    default_currency: str = DEFAULT_CURRENCY,
) -> tuple[
    NormalizedTransaction,
    list[NormalizationIssue],
]:
    """Normalize a single transaction."""

    normalizer = FinancialNormalizer(
        default_currency=default_currency
    )

    return normalizer.normalize_transaction(
        record
    )


def normalize_financial_period(
    record: Mapping[str, Any],
    *,
    default_currency: str = DEFAULT_CURRENCY,
) -> tuple[
    NormalizedFinancialPeriod,
    list[NormalizationIssue],
]:
    """Normalize a single financial period."""

    normalizer = FinancialNormalizer(
        default_currency=default_currency
    )

    return normalizer.normalize_period(
        record
    )


def normalize_currency(
    value: Any,
    default: str = DEFAULT_CURRENCY,
) -> str:
    """Normalize currency."""
    return globals()["normalize_currency"](
        value,
        default,
    )


def parse_financial_date(
    value: Any,
) -> date | None:
    """Parse a financial date."""
    return parse_date(value)


def normalize_amount(
    value: Any,
) -> Decimal:
    """Normalize a monetary amount."""
    return round_money(
        to_decimal(value)
    )


def normalize_percentage(
    value: Any,
    *,
    decimal_fraction: bool = False,
) -> Decimal:
    """Normalize a percentage."""
    return normalize_percent(
        value,
        decimal_fraction=decimal_fraction,
    )


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Constants
    "ZERO",
    "ONE",
    "ONE_HUNDRED",
    "MONEY_QUANT",
    "PERCENT_QUANT",
    "DEFAULT_CURRENCY",

    # Utilities
    "to_decimal",
    "round_money",
    "round_percent",
    "safe_divide",
    "normalize_percent",
    "normalize_ratio",
    "normalize_text",
    "normalize_key",
    "parse_date",
    "normalize_currency",
    "normalize_sign",
    "canonical_hash",

    # Enums
    "FinancialDataType",
    "AccountType",
    "TransactionType",
    "StatementType",
    "CashFlowType",
    "NormalizationSeverity",

    # Issues
    "NormalizationIssue",

    # Models
    "NormalizedTransaction",
    "NormalizedFinancialPeriod",
    "NormalizationResult",

    # Engine
    "FinancialNormalizer",

    # Convenience functions
    "normalize_financial_data",
    "normalize_transaction",
    "normalize_financial_period",
    "parse_financial_date",
    "normalize_amount",
    "normalize_percentage",
]