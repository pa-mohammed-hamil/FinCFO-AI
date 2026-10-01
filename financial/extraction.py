"""
FinCo AI - Financial Extraction Engine

File:
    backend/app/financial/extraction.py

Purpose:
    Extract, normalize, classify, and validate financial facts from
    semi-structured document data.

Responsibilities:
    - Convert raw parser output into normalized financial facts.
    - Normalize labels, dates, currencies, and numeric values.
    - Identify financial statement categories.
    - Extract revenue, expense, asset, liability, equity, and cash-flow facts.
    - Preserve source/evidence metadata for downstream RAG and auditability.
    - Detect duplicate and conflicting financial facts.
    - Provide confidence scoring.
    - Produce a normalized extraction result for downstream analytics.

This module intentionally does NOT:
    - Perform database persistence.
    - Perform OCR.
    - Parse PDF/Excel/CSV/DOCX files directly.
    - Generate forecasts.
    - Calculate final financial KPIs.
    - Make business recommendations.

Pipeline position:

    Document
        ↓
    Ingestion / Parser
        ↓
    extraction.py
        ↓
    Financial Normalization
        ↓
    P&L / Balance Sheet / Cash Flow
        ↓
    Ratios / KPIs / Health Score
        ↓
    Alerts / Forecasting / Agents / Recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
import re
from typing import Any, Iterable, Mapping, Sequence


# ============================================================================
# Constants
# ============================================================================

MONEY_QUANT = Decimal("0.01")

DEFAULT_CONFIDENCE = Decimal("0.80")
HIGH_CONFIDENCE = Decimal("0.95")
MEDIUM_CONFIDENCE = Decimal("0.80")
LOW_CONFIDENCE = Decimal("0.60")


# ============================================================================
# Enums
# ============================================================================


class FinancialStatementType(str, Enum):
    """Supported financial statement types."""

    PROFIT_AND_LOSS = "profit_and_loss"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    TRIAL_BALANCE = "trial_balance"
    GENERAL_LEDGER = "general_ledger"
    BUDGET = "budget"
    UNKNOWN = "unknown"


class FinancialFactType(str, Enum):
    """High-level financial fact classifications."""

    REVENUE = "revenue"
    EXPENSE = "expense"

    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"

    OPERATING_CASH_FLOW = "operating_cash_flow"
    INVESTING_CASH_FLOW = "investing_cash_flow"
    FINANCING_CASH_FLOW = "financing_cash_flow"

    CASH = "cash"
    RECEIVABLE = "receivable"
    PAYABLE = "payable"

    TAX = "tax"
    INTEREST = "interest"
    DEPRECIATION = "depreciation"
    AMORTIZATION = "amortization"

    PROFIT = "profit"
    LOSS = "loss"

    UNKNOWN = "unknown"


class ExtractionSource(str, Enum):
    """Source from which a financial fact originated."""

    TEXT = "text"
    TABLE = "table"
    OCR = "ocr"
    EXCEL = "excel"
    CSV = "csv"
    API = "api"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class ExtractionStatus(str, Enum):
    """Overall extraction status."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Human-readable confidence levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Utility Functions
# ============================================================================


def to_decimal(
    value: Any,
    *,
    quantize: bool = True,
    default: Decimal | None = None,
) -> Decimal | None:
    """
    Convert a value into Decimal safely.

    Handles:
        - integers
        - floats
        - Decimal
        - strings
        - accounting negatives such as (1,250.00)
        - currency symbols
        - commas
        - percentage signs

    Examples:
        "₹1,250.50" -> Decimal("1250.50")
        "(2,500.00)" -> Decimal("-2500.00")
        "1,200" -> Decimal("1200.00")
    """
    if value is None:
        return default

    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, bool):
        return Decimal(int(value))
    else:
        text = str(value).strip()

        if not text:
            return default

        negative = False

        if text.startswith("(") and text.endswith(")"):
            negative = True
            text = text[1:-1].strip()

        # Remove common currency symbols and formatting characters.
        text = re.sub(
            r"[₹$€£¥₩]|USD|INR|EUR|GBP|JPY|"
            r"\b[A-Z]{3}\b",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = text.replace(",", "")
        text = text.replace(" ", "")
        text = text.replace("%", "")

        # Keep only numeric characters, decimal point and sign.
        text = re.sub(r"[^0-9.+\-]", "", text)

        if not text:
            return default

        try:
            result = Decimal(text)
        except (InvalidOperation, ValueError):
            return default

        if negative:
            result = -abs(result)

    if quantize:
        return result.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)

    return result


def normalize_label(value: Any) -> str:
    """
    Normalize a financial account or metric label.

    Example:
        "  Total   Operating Expenses " -> "total operating expenses"
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    text = text.replace("&", "and")
    text = re.sub(r"[_\-\/]+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_currency(value: Any) -> str | None:
    """Normalize currency identifiers."""
    if value is None:
        return None

    text = str(value).strip().upper()

    aliases = {
        "₹": "INR",
        "RS": "INR",
        "RS.": "INR",
        "RUPEE": "INR",
        "RUPEES": "INR",
        "$": "USD",
        "US$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
    }

    return aliases.get(text, text)


def normalize_date(value: Any) -> date | None:
    """Convert common date representations into datetime.date."""
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
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%b-%d-%Y",
        "%B-%d-%Y",
        "%Y/%m/%d",
    )

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    return None


def confidence_level(score: Decimal | float | int) -> ConfidenceLevel:
    """Map numerical confidence to a confidence level."""
    value = float(score)

    if value >= 0.90:
        return ConfidenceLevel.HIGH

    if value >= 0.70:
        return ConfidenceLevel.MEDIUM

    return ConfidenceLevel.LOW


def clamp_confidence(value: Any) -> Decimal:
    """Keep confidence between 0 and 1."""
    score = to_decimal(value, quantize=False, default=DEFAULT_CONFIDENCE)

    if score is None:
        return DEFAULT_CONFIDENCE

    return max(Decimal("0"), min(Decimal("1"), score))


# ============================================================================
# Financial Label Classification
# ============================================================================


LABEL_RULES: dict[FinancialFactType, tuple[str, ...]] = {
    FinancialFactType.REVENUE: (
        "revenue",
        "sales",
        "turnover",
        "net sales",
        "gross sales",
        "operating revenue",
        "income from operations",
    ),
    FinancialFactType.EXPENSE: (
        "expense",
        "expenses",
        "cost",
        "costs",
        "operating expenses",
        "administrative expenses",
        "selling expenses",
        "general expenses",
    ),
    FinancialFactType.ASSET: (
        "asset",
        "assets",
        "property",
        "equipment",
        "inventory",
        "investment",
        "receivable",
        "cash",
        "bank balance",
        "prepaid",
    ),
    FinancialFactType.LIABILITY: (
        "liability",
        "liabilities",
        "payable",
        "debt",
        "loan",
        "borrowing",
        "accrued",
        "provision",
    ),
    FinancialFactType.EQUITY: (
        "equity",
        "capital",
        "share capital",
        "retained earnings",
        "reserves",
        "shareholders funds",
        "owners equity",
    ),
    FinancialFactType.CASH: (
        "cash",
        "cash balance",
        "cash and cash equivalents",
        "bank",
        "bank balance",
    ),
    FinancialFactType.RECEIVABLE: (
        "accounts receivable",
        "trade receivable",
        "trade receivables",
        "receivables",
        "debtors",
    ),
    FinancialFactType.PAYABLE: (
        "accounts payable",
        "trade payable",
        "trade payables",
        "payables",
        "creditors",
    ),
    FinancialFactType.TAX: (
        "tax",
        "income tax",
        "corporate tax",
        "tax expense",
        "tax payable",
    ),
    FinancialFactType.INTEREST: (
        "interest",
        "interest expense",
        "finance cost",
        "finance costs",
    ),
    FinancialFactType.DEPRECIATION: (
        "depreciation",
        "depreciation expense",
    ),
    FinancialFactType.AMORTIZATION: (
        "amortization",
        "amortisation",
    ),
    FinancialFactType.PROFIT: (
        "profit",
        "net profit",
        "profit after tax",
        "profit before tax",
        "operating profit",
        "gross profit",
    ),
    FinancialFactType.LOSS: (
        "loss",
        "net loss",
        "loss after tax",
        "loss before tax",
        "operating loss",
    ),
    FinancialFactType.OPERATING_CASH_FLOW: (
        "cash from operating activities",
        "net cash from operating activities",
        "operating cash flow",
    ),
    FinancialFactType.INVESTING_CASH_FLOW: (
        "cash from investing activities",
        "net cash from investing activities",
        "investing cash flow",
    ),
    FinancialFactType.FINANCING_CASH_FLOW: (
        "cash from financing activities",
        "net cash from financing activities",
        "financing cash flow",
    ),
}


def classify_financial_label(label: str) -> FinancialFactType:
    """
    Classify a financial account label using deterministic rules.

    More specific rules should be added before generic rules when required.
    """
    normalized = normalize_label(label)

    if not normalized:
        return FinancialFactType.UNKNOWN

    # Check longer / more specific labels first.
    candidates: list[tuple[int, FinancialFactType]] = []

    for fact_type, keywords in LABEL_RULES.items():
        for keyword in keywords:
            normalized_keyword = normalize_label(keyword)

            if normalized == normalized_keyword:
                candidates.append((len(normalized_keyword), fact_type))
            elif normalized_keyword in normalized:
                candidates.append((len(normalized_keyword), fact_type))

    if not candidates:
        return FinancialFactType.UNKNOWN

    candidates.sort(key=lambda item: item[0], reverse=True)

    return candidates[0][1]


def infer_statement_type(
    document_name: str | None = None,
    headers: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> FinancialStatementType:
    """
    Infer financial statement type from document metadata and labels.
    """
    searchable = " ".join(
        [
            *(str(document_name or "").lower().split()),
            *(str(value).lower() for value in (headers or [])),
            *(str(value).lower() for value in (labels or [])),
        ]
    )

    if any(
        term in searchable
        for term in (
            "cash flow",
            "cash flows",
            "cash from operating",
            "cash from investing",
            "cash from financing",
        )
    ):
        return FinancialStatementType.CASH_FLOW

    if any(
        term in searchable
        for term in (
            "balance sheet",
            "statement of financial position",
            "assets",
            "liabilities",
            "shareholders equity",
        )
    ):
        return FinancialStatementType.BALANCE_SHEET

    if any(
        term in searchable
        for term in (
            "profit and loss",
            "profit & loss",
            "income statement",
            "statement of profit",
            "revenue",
            "gross profit",
            "operating profit",
        )
    ):
        return FinancialStatementType.PROFIT_AND_LOSS

    if "trial balance" in searchable:
        return FinancialStatementType.TRIAL_BALANCE

    if "general ledger" in searchable or "general journal" in searchable:
        return FinancialStatementType.GENERAL_LEDGER

    if "budget" in searchable or "forecast budget" in searchable:
        return FinancialStatementType.BUDGET

    return FinancialStatementType.UNKNOWN


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass(slots=True)
class SourceReference:
    """
    Evidence location for a financial fact.

    This is intentionally generic so it can connect to:
        - PDF pages
        - Excel sheets/cells
        - CSV rows
        - OCR regions
        - document IDs
        - RAG citations
    """

    document_id: str | None = None
    document_name: str | None = None

    page_number: int | None = None
    sheet_name: str | None = None
    row_number: int | None = None
    column_name: str | None = None
    cell_reference: str | None = None

    text: str | None = None

    source_type: ExtractionSource = ExtractionSource.UNKNOWN

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the source reference."""
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "sheet_name": self.sheet_name,
            "row_number": self.row_number,
            "column_name": self.column_name,
            "cell_reference": self.cell_reference,
            "text": self.text,
            "source_type": self.source_type.value,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class FinancialFact:
    """
    Normalized financial fact.

    Example:

        {
            "label": "Revenue",
            "normalized_label": "revenue",
            "fact_type": "revenue",
            "value": 1250000,
            "currency": "INR",
            "period_start": ...,
            "period_end": ...,
        }
    """

    label: str
    normalized_label: str

    value: Decimal | None

    fact_type: FinancialFactType = FinancialFactType.UNKNOWN
    statement_type: FinancialStatementType = FinancialStatementType.UNKNOWN

    currency: str | None = None

    period_start: date | None = None
    period_end: date | None = None

    as_of_date: date | None = None

    unit: str | None = None

    confidence: Decimal = DEFAULT_CONFIDENCE

    source: SourceReference | None = None

    raw_value: Any = None
    raw_label: Any = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.normalized_label = normalize_label(
            self.normalized_label or self.label
        )

        self.confidence = clamp_confidence(self.confidence)

        if self.currency:
            self.currency = normalize_currency(self.currency)

        if self.value is not None:
            self.value = to_decimal(self.value)

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return confidence_level(self.confidence)

    @property
    def is_negative(self) -> bool:
        return self.value is not None and self.value < 0

    @property
    def absolute_value(self) -> Decimal | None:
        if self.value is None:
            return None

        return abs(self.value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the financial fact."""
        return {
            "label": self.label,
            "normalized_label": self.normalized_label,
            "value": str(self.value) if self.value is not None else None,
            "fact_type": self.fact_type.value,
            "statement_type": self.statement_type.value,
            "currency": self.currency,
            "period_start": (
                self.period_start.isoformat()
                if self.period_start
                else None
            ),
            "period_end": (
                self.period_end.isoformat()
                if self.period_end
                else None
            ),
            "as_of_date": (
                self.as_of_date.isoformat()
                if self.as_of_date
                else None
            ),
            "unit": self.unit,
            "confidence": str(self.confidence),
            "confidence_level": self.confidence_level.value,
            "source": (
                self.source.to_dict()
                if self.source
                else None
            ),
            "raw_value": self.raw_value,
            "raw_label": self.raw_label,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ExtractionConflict:
    """Represents conflicting extracted values."""

    normalized_label: str

    values: list[Decimal] = field(default_factory=list)

    sources: list[SourceReference] = field(default_factory=list)

    message: str = ""

    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_label": self.normalized_label,
            "values": [str(value) for value in self.values],
            "sources": [
                source.to_dict()
                for source in self.sources
            ],
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(slots=True)
class ExtractionResult:
    """Complete output of the extraction engine."""

    status: ExtractionStatus

    statement_type: FinancialStatementType

    facts: list[FinancialFact] = field(default_factory=list)

    conflicts: list[ExtractionConflict] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fact_count(self) -> int:
        return len(self.facts)

    @property
    def high_confidence_facts(self) -> list[FinancialFact]:
        return [
            fact
            for fact in self.facts
            if fact.confidence >= Decimal("0.90")
        ]

    @property
    def low_confidence_facts(self) -> list[FinancialFact]:
        return [
            fact
            for fact in self.facts
            if fact.confidence < Decimal("0.70")
        ]

    def by_type(
        self,
        fact_type: FinancialFactType,
    ) -> list[FinancialFact]:
        return [
            fact
            for fact in self.facts
            if fact.fact_type == fact_type
        ]

    def by_label(
        self,
        label: str,
    ) -> list[FinancialFact]:
        normalized = normalize_label(label)

        return [
            fact
            for fact in self.facts
            if fact.normalized_label == normalized
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "statement_type": self.statement_type.value,
            "fact_count": self.fact_count,
            "facts": [
                fact.to_dict()
                for fact in self.facts
            ],
            "conflicts": [
                conflict.to_dict()
                for conflict in self.conflicts
            ],
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


# ============================================================================
# Financial Extraction Engine
# ============================================================================


class FinancialExtractor:
    """
    Stateless financial extraction engine.

    The extractor accepts parser output rather than files.

    Supported raw row shapes include:

        {
            "label": "Revenue",
            "value": "1,250,000",
            "currency": "INR",
            "period_end": "2025-03-31",
        }

    or:

        {
            "account": "Operating Expenses",
            "amount": 500000,
        }

    or:

        ["Revenue", "1,250,000"]
    """

    LABEL_KEYS = (
        "label",
        "name",
        "account",
        "account_name",
        "description",
        "particulars",
        "item",
        "metric",
        "category",
    )

    VALUE_KEYS = (
        "value",
        "amount",
        "balance",
        "total",
        "net_amount",
        "debit",
        "credit",
    )

    CURRENCY_KEYS = (
        "currency",
        "currency_code",
        "ccy",
    )

    PERIOD_START_KEYS = (
        "period_start",
        "start_date",
        "from_date",
        "date_from",
    )

    PERIOD_END_KEYS = (
        "period_end",
        "end_date",
        "to_date",
        "date_to",
    )

    AS_OF_KEYS = (
        "as_of",
        "as_of_date",
        "date",
        "reporting_date",
    )

    def __init__(
        self,
        *,
        default_currency: str | None = None,
        default_confidence: Decimal = DEFAULT_CONFIDENCE,
        strict: bool = False,
    ) -> None:
        self.default_currency = normalize_currency(default_currency)
        self.default_confidence = clamp_confidence(
            default_confidence
        )
        self.strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        rows: Iterable[Any],
        *,
        document_id: str | None = None,
        document_name: str | None = None,
        statement_type: FinancialStatementType | str | None = None,
        source_type: ExtractionSource = ExtractionSource.TABLE,
        currency: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExtractionResult:
        """
        Extract financial facts from raw rows.
        """
        normalized_statement_type = self._normalize_statement_type(
            statement_type
        )

        rows_list = list(rows)

        if normalized_statement_type == FinancialStatementType.UNKNOWN:
            normalized_statement_type = infer_statement_type(
                document_name=document_name,
                labels=self._collect_labels(rows_list),
            )

        facts: list[FinancialFact] = []
        warnings: list[str] = []
        errors: list[str] = []

        for index, row in enumerate(rows_list):
            try:
                fact = self.extract_row(
                    row,
                    row_number=index + 1,
                    document_id=document_id,
                    document_name=document_name,
                    statement_type=normalized_statement_type,
                    source_type=source_type,
                    currency=currency,
                )

                if fact is not None:
                    facts.append(fact)

            except Exception as exc:
                message = (
                    f"Failed to extract row {index + 1}: {exc}"
                )

                if self.strict:
                    errors.append(message)
                else:
                    warnings.append(message)

        facts = self.deduplicate_facts(facts)

        conflicts = self.detect_conflicts(facts)

        if errors and not facts:
            status = ExtractionStatus.FAILED
        elif warnings or errors or conflicts:
            status = ExtractionStatus.PARTIAL
        else:
            status = ExtractionStatus.SUCCESS

        return ExtractionResult(
            status=status,
            statement_type=normalized_statement_type,
            facts=facts,
            conflicts=conflicts,
            warnings=warnings,
            errors=errors,
            metadata={
                **dict(metadata or {}),
                "document_id": document_id,
                "document_name": document_name,
                "source_type": source_type.value,
                "default_currency": self.default_currency,
            },
        )

    def extract_row(
        self,
        row: Any,
        *,
        row_number: int | None = None,
        document_id: str | None = None,
        document_name: str | None = None,
        statement_type: FinancialStatementType = (
            FinancialStatementType.UNKNOWN
        ),
        source_type: ExtractionSource = ExtractionSource.TABLE,
        currency: str | None = None,
    ) -> FinancialFact | None:
        """
        Extract one normalized financial fact from one parser row.
        """
        label, value, row_metadata = self._extract_row_fields(row)

        if not label:
            return None

        if value is None:
            return None

        normalized_label = normalize_label(label)

        fact_type = classify_financial_label(normalized_label)

        resolved_currency = normalize_currency(
            row_metadata.get("currency")
            or currency
            or self.default_currency
        )

        period_start = normalize_date(
            self._first_value(
                row_metadata,
                self.PERIOD_START_KEYS,
            )
        )

        period_end = normalize_date(
            self._first_value(
                row_metadata,
                self.PERIOD_END_KEYS,
            )
        )

        as_of_date = normalize_date(
            self._first_value(
                row_metadata,
                self.AS_OF_KEYS,
            )
        )

        confidence = self._calculate_confidence(
            label=label,
            value=value,
            fact_type=fact_type,
            row=row,
        )

        source = SourceReference(
            document_id=document_id,
            document_name=document_name,
            row_number=row_number,
            sheet_name=row_metadata.get("sheet_name"),
            column_name=row_metadata.get("column_name"),
            cell_reference=row_metadata.get("cell_reference"),
            text=row_metadata.get("text"),
            source_type=source_type,
            metadata={
                key: value
                for key, value in row_metadata.items()
                if key not in self.VALUE_KEYS
            },
        )

        return FinancialFact(
            label=str(label),
            normalized_label=normalized_label,
            value=to_decimal(value),
            fact_type=fact_type,
            statement_type=statement_type,
            currency=resolved_currency,
            period_start=period_start,
            period_end=period_end,
            as_of_date=as_of_date,
            unit=row_metadata.get("unit"),
            confidence=confidence,
            source=source,
            raw_value=value,
            raw_label=label,
            metadata={
                "row_number": row_number,
                "extraction_method": "rule_based",
                **{
                    key: value
                    for key, value in row_metadata.items()
                    if key not in {
                        *self.LABEL_KEYS,
                        *self.VALUE_KEYS,
                    }
                },
            },
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_type(
        self,
        facts: Iterable[FinancialFact],
        fact_type: FinancialFactType,
    ) -> list[FinancialFact]:
        """Return facts belonging to a specific financial type."""
        return [
            fact
            for fact in facts
            if fact.fact_type == fact_type
        ]

    def filter_by_statement(
        self,
        facts: Iterable[FinancialFact],
        statement_type: FinancialStatementType,
    ) -> list[FinancialFact]:
        """Return facts belonging to a specific statement."""
        return [
            fact
            for fact in facts
            if fact.statement_type == statement_type
        ]

    def filter_confident(
        self,
        facts: Iterable[FinancialFact],
        minimum_confidence: Decimal | float = Decimal("0.70"),
    ) -> list[FinancialFact]:
        """Return facts meeting a confidence threshold."""
        threshold = clamp_confidence(minimum_confidence)

        return [
            fact
            for fact in facts
            if fact.confidence >= threshold
        ]

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_label(
        self,
        facts: Iterable[FinancialFact],
    ) -> dict[str, Decimal]:
        """
        Aggregate values by normalized financial label.
        """
        result: dict[str, Decimal] = {}

        for fact in facts:
            if fact.value is None:
                continue

            result[fact.normalized_label] = (
                result.get(fact.normalized_label, Decimal("0"))
                + fact.value
            )

        return {
            key: value.quantize(
                MONEY_QUANT,
                rounding=ROUND_HALF_UP,
            )
            for key, value in result.items()
        }

    def totals_by_type(
        self,
        facts: Iterable[FinancialFact],
    ) -> dict[FinancialFactType, Decimal]:
        """
        Aggregate values by fact type.
        """
        result: dict[FinancialFactType, Decimal] = {}

        for fact in facts:
            if fact.value is None:
                continue

            result[fact.fact_type] = (
                result.get(
                    fact.fact_type,
                    Decimal("0"),
                )
                + fact.value
            )

        return {
            key: value.quantize(
                MONEY_QUANT,
                rounding=ROUND_HALF_UP,
            )
            for key, value in result.items()
        }

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate_facts(
        self,
        facts: Sequence[FinancialFact],
    ) -> list[FinancialFact]:
        """
        Remove duplicate financial facts.

        Facts with the same normalized label, value, period, and source
        location are treated as duplicates.
        """
        unique: list[FinancialFact] = []
        seen: set[tuple[Any, ...]] = set()

        for fact in facts:
            key = (
                fact.normalized_label,
                fact.value,
                fact.period_start,
                fact.period_end,
                fact.as_of_date,
                fact.currency,
                fact.source.sheet_name if fact.source else None,
                fact.source.row_number if fact.source else None,
                fact.source.cell_reference if fact.source else None,
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(fact)

        return unique

    # ------------------------------------------------------------------
    # Conflict Detection
    # ------------------------------------------------------------------

    def detect_conflicts(
        self,
        facts: Sequence[FinancialFact],
    ) -> list[ExtractionConflict]:
        """
        Detect cases where the same financial label has multiple values.

        This is useful when:
            - PDF extraction duplicates a table.
            - Multiple source pages contain inconsistent figures.
            - OCR produces conflicting values.
            - Different source documents disagree.
        """
        grouped: dict[str, list[FinancialFact]] = {}

        for fact in facts:
            grouped.setdefault(
                fact.normalized_label,
                [],
            ).append(fact)

        conflicts: list[ExtractionConflict] = []

        for label, group in grouped.items():
            distinct_values = {
                fact.value
                for fact in group
                if fact.value is not None
            }

            if len(distinct_values) <= 1:
                continue

            values = sorted(
                distinct_values,
                key=lambda value: str(value),
            )

            sources = [
                fact.source
                for fact in group
                if fact.source is not None
            ]

            conflicts.append(
                ExtractionConflict(
                    normalized_label=label,
                    values=values,
                    sources=sources,
                    message=(
                        f"Multiple values found for "
                        f"'{label}': {', '.join(map(str, values))}"
                    ),
                    severity="high",
                )
            )

        return conflicts

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_facts(
        self,
        facts: Sequence[FinancialFact],
    ) -> list[str]:
        """
        Perform lightweight financial fact validation.

        This does not replace accounting validation.
        """
        errors: list[str] = []

        for index, fact in enumerate(facts, start=1):
            if not fact.label:
                errors.append(
                    f"Fact {index}: missing label."
                )

            if fact.value is None:
                errors.append(
                    f"Fact {index}: missing value."
                )

            if not fact.normalized_label:
                errors.append(
                    f"Fact {index}: missing normalized label."
                )

            if not Decimal("0") <= fact.confidence <= Decimal("1"):
                errors.append(
                    f"Fact {index}: invalid confidence."
                )

            if (
                fact.period_start
                and fact.period_end
                and fact.period_start > fact.period_end
            ):
                errors.append(
                    f"Fact {index}: period start occurs "
                    f"after period end."
                )

        return errors

    # ------------------------------------------------------------------
    # Statement-Specific Extraction
    # ------------------------------------------------------------------

    def extract_revenue(
        self,
        facts: Iterable[FinancialFact],
    ) -> list[FinancialFact]:
        """Extract revenue-related facts."""
        return self.filter_by_type(
            facts,
            FinancialFactType.REVENUE,
        )

    def extract_expenses(
        self,
        facts: Iterable[FinancialFact],
    ) -> list[FinancialFact]:
        """Extract expense-related facts."""
        return self.filter_by_type(
            facts,
            FinancialFactType.EXPENSE,
        )

    def extract_assets(
        self,
        facts: Iterable[FinancialFact],
    ) -> list[FinancialFact]:
        """Extract asset-related facts."""
        return self.filter_by_type(
            facts,
            FinancialFactType.ASSET,
        )

    def extract_liabilities(
        self,
        facts: Iterable[FinancialFact],
    ) -> list[FinancialFact]:
        """Extract liability-related facts."""
        return self.filter_by_type(
            facts,
            FinancialFactType.LIABILITY,
        )

    def extract_equity(
        self,
        facts: Iterable[FinancialFact],
    ) -> list[FinancialFact]:
        """Extract equity-related facts."""
        return self.filter_by_type(
            facts,
            FinancialFactType.EQUITY,
        )

    def extract_cash_flow(
        self,
        facts: Iterable[FinancialFact],
    ) -> list[FinancialFact]:
        """Extract cash-flow-related facts."""
        cash_flow_types = {
            FinancialFactType.OPERATING_CASH_FLOW,
            FinancialFactType.INVESTING_CASH_FLOW,
            FinancialFactType.FINANCING_CASH_FLOW,
            FinancialFactType.CASH,
        }

        return [
            fact
            for fact in facts
            if fact.fact_type in cash_flow_types
        ]

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _extract_row_fields(
        self,
        row: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """
        Extract label/value/metadata from common parser row formats.
        """
        if isinstance(row, Mapping):
            label = self._first_value(row, self.LABEL_KEYS)
            value = self._first_value(row, self.VALUE_KEYS)

            return label, value, dict(row)

        if isinstance(row, (list, tuple)):
            if len(row) == 0:
                return None, None, {}

            if len(row) == 1:
                return row[0], None, {}

            label = row[0]
            value = row[1]

            metadata: dict[str, Any] = {
                "raw_row": list(row),
            }

            return label, value, metadata

        return None, None, {}

    @staticmethod
    def _first_value(
        mapping: Mapping[str, Any],
        keys: Sequence[str],
    ) -> Any:
        """Return the first non-empty value matching candidate keys."""
        for key in keys:
            if key in mapping:
                value = mapping[key]

                if value is not None and value != "":
                    return value

        return None

    def _calculate_confidence(
        self,
        *,
        label: Any,
        value: Any,
        fact_type: FinancialFactType,
        row: Any,
    ) -> Decimal:
        """
        Estimate extraction confidence.

        This is intentionally conservative and deterministic.
        """
        score = self.default_confidence

        if label is None:
            return Decimal("0.20")

        if value is None:
            return Decimal("0.30")

        normalized_label = normalize_label(label)

        if not normalized_label:
            return Decimal("0.30")

        if fact_type != FinancialFactType.UNKNOWN:
            score += Decimal("0.10")

        if isinstance(row, Mapping):
            score += Decimal("0.03")

        if isinstance(value, (Decimal, int, float)):
            score += Decimal("0.02")

        return min(score, Decimal("1.00"))

    @staticmethod
    def _normalize_statement_type(
        statement_type: FinancialStatementType | str | None,
    ) -> FinancialStatementType:
        if statement_type is None:
            return FinancialStatementType.UNKNOWN

        if isinstance(statement_type, FinancialStatementType):
            return statement_type

        normalized = normalize_label(statement_type)

        aliases = {
            "pnl": FinancialStatementType.PROFIT_AND_LOSS,
            "profit loss": FinancialStatementType.PROFIT_AND_LOSS,
            "profit and loss": FinancialStatementType.PROFIT_AND_LOSS,
            "income statement": FinancialStatementType.PROFIT_AND_LOSS,

            "balance sheet": FinancialStatementType.BALANCE_SHEET,
            "financial position": FinancialStatementType.BALANCE_SHEET,

            "cash flow": FinancialStatementType.CASH_FLOW,
            "cash flows": FinancialStatementType.CASH_FLOW,

            "trial balance": FinancialStatementType.TRIAL_BALANCE,
            "general ledger": FinancialStatementType.GENERAL_LEDGER,
            "budget": FinancialStatementType.BUDGET,
        }

        return aliases.get(
            normalized,
            FinancialStatementType.UNKNOWN,
        )

    @staticmethod
    def _collect_labels(
        rows: Sequence[Any],
    ) -> list[str]:
        labels: list[str] = []

        for row in rows:
            if isinstance(row, Mapping):
                for key in (
                    "label",
                    "name",
                    "account",
                    "account_name",
                    "description",
                    "particulars",
                ):
                    value = row.get(key)

                    if value:
                        labels.append(str(value))
                        break

            elif isinstance(row, (list, tuple)) and row:
                labels.append(str(row[0]))

        return labels


# ============================================================================
# Convenience Functions
# ============================================================================


def extract_financial_facts(
    rows: Iterable[Any],
    *,
    document_id: str | None = None,
    document_name: str | None = None,
    statement_type: FinancialStatementType | str | None = None,
    source_type: ExtractionSource = ExtractionSource.TABLE,
    currency: str | None = None,
) -> ExtractionResult:
    """
    Convenience wrapper around FinancialExtractor.
    """
    extractor = FinancialExtractor(
        default_currency=currency,
    )

    return extractor.extract(
        rows,
        document_id=document_id,
        document_name=document_name,
        statement_type=statement_type,
        source_type=source_type,
        currency=currency,
    )


def classify_account(label: str) -> FinancialFactType:
    """Convenience wrapper for financial label classification."""
    return classify_financial_label(label)


def detect_statement_type(
    document_name: str | None = None,
    headers: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> FinancialStatementType:
    """Convenience wrapper for statement type detection."""
    return infer_statement_type(
        document_name=document_name,
        headers=headers,
        labels=labels,
    )


def normalize_financial_value(
    value: Any,
) -> Decimal | None:
    """Convenience wrapper for monetary normalization."""
    return to_decimal(value)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    "FinancialStatementType",
    "FinancialFactType",
    "ExtractionSource",
    "ExtractionStatus",
    "ConfidenceLevel",
    "SourceReference",
    "FinancialFact",
    "ExtractionConflict",
    "ExtractionResult",
    "FinancialExtractor",
    "classify_financial_label",
    "classify_account",
    "infer_statement_type",
    "detect_statement_type",
    "normalize_label",
    "normalize_currency",
    "normalize_date",
    "normalize_financial_value",
    "to_decimal",
    "extract_financial_facts",
]