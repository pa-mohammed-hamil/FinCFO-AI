"""
FinCo AI - CSV Parser

Responsible for:
- Loading CSV files
- Detecting common encodings
- Normalizing column names
- Parsing dates
- Converting numeric financial columns
- Handling missing values
- Removing duplicate rows
- Validating financial data
- Returning a clean pandas DataFrame

Expected usage:

    from app.ingestion.csv_parser import csv_parser

    result = csv_parser.parse("data/transactions.csv")

    dataframe = result.dataframe
    report = result.quality_report
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class CSVParserError(Exception):
    """Base exception for CSV parsing errors."""


class CSVFileNotFoundError(CSVParserError):
    """Raised when the CSV file cannot be found."""


class CSVValidationError(CSVParserError):
    """Raised when the CSV structure is invalid."""


class CSVEncodingError(CSVParserError):
    """Raised when the CSV encoding cannot be decoded."""


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class CSVParserConfig:
    """
    Configuration for CSV parsing.
    """

    encoding_candidates: tuple[str, ...] = (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    )

    delimiter_candidates: tuple[str, ...] = (
        ",",
        ";",
        "\t",
        "|",
    )

    default_encoding: str = "utf-8-sig"

    infer_datetime_format: bool = True

    drop_empty_rows: bool = True

    drop_duplicate_rows: bool = True

    normalize_columns: bool = True

    convert_numeric_columns: bool = True

    parse_date_columns: bool = True

    min_rows: int = 1

    max_rows: Optional[int] = None

    max_columns: Optional[int] = 200

    fail_on_missing_required_columns: bool = False

    required_columns: tuple[str, ...] = ()

    financial_numeric_columns: tuple[str, ...] = (
        "amount",
        "balance",
        "income",
        "revenue",
        "sales",
        "expense",
        "expenses",
        "cost",
        "profit",
        "loss",
        "debt",
        "credit",
        "debit",
        "cash_flow",
        "savings",
        "tax",
    )

    date_column_names: tuple[str, ...] = (
        "date",
        "transaction_date",
        "payment_date",
        "invoice_date",
        "created_at",
        "updated_at",
        "timestamp",
        "period",
        "month",
        "year",
    )


# ============================================================================
# Result Models
# ============================================================================


@dataclass
class CSVQualityReport:
    """
    Data-quality report generated during parsing.
    """

    source_name: Optional[str] = None

    original_row_count: int = 0

    final_row_count: int = 0

    original_column_count: int = 0

    final_column_count: int = 0

    duplicate_rows_removed: int = 0

    empty_rows_removed: int = 0

    missing_values_by_column: Dict[str, int] = field(
        default_factory=dict
    )

    missing_percentage_by_column: Dict[str, float] = field(
        default_factory=dict
    )

    converted_numeric_columns: List[str] = field(
        default_factory=list
    )

    parsed_date_columns: List[str] = field(
        default_factory=list
    )

    invalid_date_counts: Dict[str, int] = field(
        default_factory=dict
    )

    invalid_numeric_counts: Dict[str, int] = field(
        default_factory=dict
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    detected_encoding: Optional[str] = None

    detected_delimiter: Optional[str] = None

    is_valid: bool = True

    def add_warning(
        self,
        message: str,
    ) -> None:
        self.warnings.append(message)

    def add_error(
        self,
        message: str,
    ) -> None:
        self.errors.append(message)
        self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "original_row_count": self.original_row_count,
            "final_row_count": self.final_row_count,
            "original_column_count": self.original_column_count,
            "final_column_count": self.final_column_count,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "empty_rows_removed": self.empty_rows_removed,
            "missing_values_by_column": self.missing_values_by_column,
            "missing_percentage_by_column": (
                self.missing_percentage_by_column
            ),
            "converted_numeric_columns": (
                self.converted_numeric_columns
            ),
            "parsed_date_columns": self.parsed_date_columns,
            "invalid_date_counts": self.invalid_date_counts,
            "invalid_numeric_counts": self.invalid_numeric_counts,
            "warnings": self.warnings,
            "errors": self.errors,
            "detected_encoding": self.detected_encoding,
            "detected_delimiter": self.detected_delimiter,
            "is_valid": self.is_valid,
        }


@dataclass
class CSVParseResult:
    """
    Result returned by CSVParser.parse().
    """

    dataframe: pd.DataFrame

    quality_report: CSVQualityReport

    source_name: Optional[str] = None

    success: bool = True

    @property
    def row_count(self) -> int:
        return len(self.dataframe)

    @property
    def column_count(self) -> int:
        return len(self.dataframe.columns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "success": self.success,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.dataframe.columns),
            "quality_report": self.quality_report.to_dict(),
        }


# ============================================================================
# CSV Parser
# ============================================================================


class CSVParser:
    """
    Production-oriented CSV parser for FinCo AI.

    The parser does not assume that every CSV has the same schema.
    It performs generic cleaning and exposes validation warnings.
    Domain-specific validation should be performed by the financial
    ingestion or validation layer.
    """

    def __init__(
        self,
        config: CSVParserConfig = CSVParserConfig(),
    ) -> None:
        self.config = config

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def parse(
        self,
        source: str | Path | bytes | bytearray | BinaryIO,
        source_name: Optional[str] = None,
        required_columns: Optional[Sequence[str]] = None,
    ) -> CSVParseResult:
        """
        Parse a CSV source.

        Supported source types:
            - File path
            - pathlib.Path
            - bytes
            - bytearray
            - BinaryIO
        """

        resolved_source_name = (
            source_name
            or self._get_source_name(source)
        )

        report = CSVQualityReport(
            source_name=resolved_source_name,
        )

        raw_bytes = self._read_source(source)

        text, encoding = self._decode_bytes(raw_bytes)

        report.detected_encoding = encoding

        delimiter = self._detect_delimiter(text)

        report.detected_delimiter = delimiter

        dataframe = self._read_dataframe(
            text=text,
            delimiter=delimiter,
        )

        report.original_row_count = len(dataframe)
        report.original_column_count = len(dataframe.columns)

        if self.config.max_columns is not None:
            if len(dataframe.columns) > self.config.max_columns:
                raise CSVValidationError(
                    f"CSV contains {len(dataframe.columns)} columns. "
                    f"Maximum allowed is {self.config.max_columns}."
                )

        dataframe = self.clean_dataframe(
            dataframe=dataframe,
            report=report,
        )

        self.validate_dataframe(
            dataframe=dataframe,
            report=report,
            required_columns=required_columns,
        )

        report.final_row_count = len(dataframe)
        report.final_column_count = len(dataframe.columns)

        if len(dataframe) < self.config.min_rows:
            report.add_error(
                f"CSV contains {len(dataframe)} rows after cleaning. "
                f"Minimum required rows: {self.config.min_rows}."
            )

        if self.config.max_rows is not None:
            if len(dataframe) > self.config.max_rows:
                report.add_error(
                    f"CSV contains {len(dataframe)} rows. "
                    f"Maximum allowed rows: {self.config.max_rows}."
                )

        if report.errors:
            if self.config.fail_on_missing_required_columns:
                raise CSVValidationError(
                    "; ".join(report.errors)
                )

        return CSVParseResult(
            dataframe=dataframe,
            quality_report=report,
            source_name=resolved_source_name,
            success=report.is_valid,
        )

    def parse_dataframe(
        self,
        dataframe: pd.DataFrame,
        source_name: Optional[str] = None,
        required_columns: Optional[Sequence[str]] = None,
    ) -> CSVParseResult:
        """
        Clean and validate an already-loaded DataFrame.
        """

        report = CSVQualityReport(
            source_name=source_name,
            original_row_count=len(dataframe),
            original_column_count=len(dataframe.columns),
        )

        cleaned = self.clean_dataframe(
            dataframe=dataframe.copy(),
            report=report,
        )

        self.validate_dataframe(
            dataframe=cleaned,
            report=report,
            required_columns=required_columns,
        )

        report.final_row_count = len(cleaned)
        report.final_column_count = len(cleaned.columns)

        return CSVParseResult(
            dataframe=cleaned,
            quality_report=report,
            source_name=source_name,
            success=report.is_valid,
        )

    # ---------------------------------------------------------------------
    # Source handling
    # ---------------------------------------------------------------------

    def _get_source_name(
        self,
        source: Any,
    ) -> Optional[str]:
        if isinstance(source, (str, Path)):
            return Path(source).name

        return getattr(source, "name", None)

    def _read_source(
        self,
        source: str | Path | bytes | bytearray | BinaryIO,
    ) -> bytes:
        if isinstance(source, Path):
            if not source.exists():
                raise CSVFileNotFoundError(
                    f"CSV file not found: {source}"
                )

            if not source.is_file():
                raise CSVFileNotFoundError(
                    f"CSV source is not a file: {source}"
                )

            return source.read_bytes()

        if isinstance(source, str):
            path = Path(source)

            if not path.exists():
                raise CSVFileNotFoundError(
                    f"CSV file not found: {source}"
                )

            if not path.is_file():
                raise CSVFileNotFoundError(
                    f"CSV source is not a file: {source}"
                )

            return path.read_bytes()

        if isinstance(source, bytes):
            return source

        if isinstance(source, bytearray):
            return bytes(source)

        if hasattr(source, "read"):
            content = source.read()

            if isinstance(content, str):
                return content.encode(
                    self.config.default_encoding
                )

            return content

        raise TypeError(
            "Unsupported CSV source type. "
            "Use a path, bytes, bytearray, or binary file object."
        )

    def _decode_bytes(
        self,
        raw_bytes: bytes,
    ) -> tuple[str, str]:
        for encoding in self.config.encoding_candidates:
            try:
                return raw_bytes.decode(encoding), encoding
            except UnicodeDecodeError:
                continue

        raise CSVEncodingError(
            "Unable to decode CSV using supported encodings: "
            f"{self.config.encoding_candidates}"
        )

    def _detect_delimiter(
        self,
        text: str,
    ) -> str:
        """
        Detects the delimiter using csv.Sniffer with a fallback.
        """

        import csv

        sample = text[:10_000]

        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters="".join(
                    self.config.delimiter_candidates
                ),
            )

            return dialect.delimiter

        except csv.Error:
            first_line = text.splitlines()[0] if text else ""

            counts = {
                delimiter: first_line.count(delimiter)
                for delimiter in self.config.delimiter_candidates
            }

            return max(
                counts,
                key=counts.get,
            )

    def _read_dataframe(
        self,
        text: str,
        delimiter: str,
    ) -> pd.DataFrame:
        try:
            return pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                engine="python",
                dtype="object",
                keep_default_na=True,
            )

        except Exception as exc:
            raise CSVParserError(
                f"Unable to parse CSV content: {exc}"
            ) from exc

    # ---------------------------------------------------------------------
    # Cleaning
    # ---------------------------------------------------------------------

    def clean_dataframe(
        self,
        dataframe: pd.DataFrame,
        report: CSVQualityReport,
    ) -> pd.DataFrame:
        """
        Applies generic cleaning operations.
        """

        dataframe = dataframe.copy()

        if self.config.normalize_columns:
            dataframe = self.normalize_columns(
                dataframe
            )

        dataframe = self.remove_empty_columns(
            dataframe=dataframe,
            report=report,
        )

        if self.config.drop_empty_rows:
            dataframe = self.remove_empty_rows(
                dataframe=dataframe,
                report=report,
            )

        if self.config.drop_duplicate_rows:
            dataframe = self.remove_duplicate_rows(
                dataframe=dataframe,
                report=report,
            )

        dataframe = self.clean_string_values(
            dataframe
        )

        if self.config.convert_numeric_columns:
            dataframe = self.convert_numeric_columns(
                dataframe=dataframe,
                report=report,
            )

        if self.config.parse_date_columns:
            dataframe = self.parse_date_columns(
                dataframe=dataframe,
                report=report,
            )

        dataframe = self.replace_empty_strings_with_na(
            dataframe
        )

        self.update_missing_value_report(
            dataframe=dataframe,
            report=report,
        )

        return dataframe

    def normalize_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Converts column names into safe snake_case names.
        """

        dataframe = dataframe.copy()

        normalized_columns: List[str] = []
        used_names: Dict[str, int] = {}

        for column in dataframe.columns:
            name = str(column).strip().lower()

            name = re.sub(
                r"[^a-z0-9]+",
                "_",
                name,
            )

            name = re.sub(
                r"_+",
                "_",
                name,
            ).strip("_")

            if not name:
                name = "unnamed_column"

            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"

            else:
                used_names[name] = 0

            normalized_columns.append(name)

        dataframe.columns = normalized_columns

        return dataframe

    def remove_empty_columns(
        self,
        dataframe: pd.DataFrame,
        report: CSVQualityReport,
    ) -> pd.DataFrame:
        empty_columns = [
            column
            for column in dataframe.columns
            if dataframe[column].isna().all()
        ]

        if empty_columns:
            dataframe = dataframe.drop(
                columns=empty_columns
            )

            report.add_warning(
                "Removed completely empty columns: "
                f"{empty_columns}"
            )

        return dataframe

    def remove_empty_rows(
        self,
        dataframe: pd.DataFrame,
        report: CSVQualityReport,
    ) -> pd.DataFrame:
        before = len(dataframe)

        dataframe = dataframe.dropna(
            how="all"
        )

        removed = before - len(dataframe)

        report.empty_rows_removed += removed

        if removed:
            report.add_warning(
                f"Removed {removed} completely empty row(s)."
            )

        return dataframe

    def remove_duplicate_rows(
        self,
        dataframe: pd.DataFrame,
        report: CSVQualityReport,
    ) -> pd.DataFrame:
        before = len(dataframe)

        dataframe = dataframe.drop_duplicates()

        removed = before - len(dataframe)

        report.duplicate_rows_removed += removed

        if removed:
            report.add_warning(
                f"Removed {removed} duplicate row(s)."
            )

        return dataframe

    def clean_string_values(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()

        for column in dataframe.columns:
            if dataframe[column].dtype == "object":
                dataframe[column] = dataframe[column].map(
                    lambda value: (
                        value.strip()
                        if isinstance(value, str)
                        else value
                    )
                )

        return dataframe

    def replace_empty_strings_with_na(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()

        dataframe = dataframe.replace(
            to_replace=r"^\s*$",
            value=pd.NA,
            regex=True,
        )

        return dataframe

    # ---------------------------------------------------------------------
    # Type conversion
    # ---------------------------------------------------------------------

    def convert_numeric_columns(
        self,
        dataframe: pd.DataFrame,
        report: CSVQualityReport,
    ) -> pd.DataFrame:
        """
        Converts columns that appear to contain numeric values.

        Currency symbols and common separators are removed.
        """

        dataframe = dataframe.copy()

        for column in dataframe.columns:
            if not self._looks_like_numeric_column(
                column,
                dataframe[column],
            ):
                continue

            original = dataframe[column].copy()

            converted = self._to_numeric_series(
                dataframe[column]
            )

            non_empty_original = original.notna()

            invalid_values = (
                non_empty_original
                & converted.isna()
            )

            invalid_count = int(
                invalid_values.sum()
            )

            if invalid_count:
                report.invalid_numeric_counts[column] = (
                    invalid_count
                )

                report.add_warning(
                    f"Column '{column}' contains "
                    f"{invalid_count} invalid numeric value(s)."
                )

            valid_numeric_count = int(
                converted.notna().sum()
            )

            if valid_numeric_count > 0:
                dataframe[column] = converted

                report.converted_numeric_columns.append(
                    column
                )

        return dataframe

    def _looks_like_numeric_column(
        self,
        column: str,
        series: pd.Series,
    ) -> bool:
        normalized_column = column.lower()

        if normalized_column in self.config.financial_numeric_columns:
            return True

        numeric_keywords = (
            "amount",
            "balance",
            "income",
            "revenue",
            "sales",
            "expense",
            "cost",
            "profit",
            "loss",
            "debit",
            "credit",
            "cash",
            "saving",
            "tax",
            "price",
            "quantity",
            "count",
            "total",
        )

        if any(
            keyword in normalized_column
            for keyword in numeric_keywords
        ):
            return True

        sample = series.dropna().astype(str).head(100)

        if sample.empty:
            return False

        converted = self._to_numeric_series(
            sample
        )

        numeric_ratio = (
            converted.notna().sum()
            / len(sample)
        )

        return numeric_ratio >= 0.90

    def _to_numeric_series(
        self,
        series: pd.Series,
    ) -> pd.Series:
        cleaned = (
            series.astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False)
            .str.replace("INR", "", regex=False)
            .str.replace("USD", "", regex=False)
            .str.replace("%", "", regex=False)
        )

        negative_parentheses = cleaned.str.match(
            r"^\(.*\)$",
            na=False,
        )

        cleaned = cleaned.str.replace(
            r"^\((.*)\)$",
            r"-\1",
            regex=True,
        )

        converted = pd.to_numeric(
            cleaned,
            errors="coerce",
        )

        # Preserve the intended sign for parenthesized values.
        converted.loc[negative_parentheses] = (
            -converted.loc[negative_parentheses].abs()
        )

        return converted

    def parse_date_columns(
        self,
        dataframe: pd.DataFrame,
        report: CSVQualityReport,
    ) -> pd.DataFrame:
        """
        Parses columns whose names indicate date/time information.
        """

        dataframe = dataframe.copy()

        for column in dataframe.columns:
            if not self._looks_like_date_column(
                column,
                dataframe[column],
            ):
                continue

            original = dataframe[column].copy()

            parsed = pd.to_datetime(
                dataframe[column],
                errors="coerce",
                infer_datetime_format=(
                    self.config.infer_datetime_format
                ),
            )

            invalid_values = (
                original.notna()
                & parsed.isna()
            )

            invalid_count = int(
                invalid_values.sum()
            )

            if invalid_count:
                report.invalid_date_counts[column] = (
                    invalid_count
                )

                report.add_warning(
                    f"Column '{column}' contains "
                    f"{invalid_count} invalid date value(s)."
                )

            if parsed.notna().sum() > 0:
                dataframe[column] = parsed

                report.parsed_date_columns.append(
                    column
                )

        return dataframe

    def _looks_like_date_column(
        self,
        column: str,
        series: pd.Series,
    ) -> bool:
        normalized_column = column.lower()

        if normalized_column in self.config.date_column_names:
            return True

        date_keywords = (
            "date",
            "time",
            "timestamp",
            "period",
            "month",
            "year",
        )

        if any(
            keyword in normalized_column
            for keyword in date_keywords
        ):
            return True

        sample = series.dropna().astype(str).head(50)

        if sample.empty:
            return False

        parsed = pd.to_datetime(
            sample,
            errors="coerce",
        )

        return (
            parsed.notna().sum() / len(sample)
        ) >= 0.90

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def validate_dataframe(
        self,
        dataframe: pd.DataFrame,
        report: CSVQualityReport,
        required_columns: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Validates the cleaned DataFrame.
        """

        if dataframe.empty:
            report.add_error(
                "CSV contains no usable rows after cleaning."
            )
            return

        if len(dataframe.columns) == 0:
            report.add_error(
                "CSV contains no usable columns."
            )
            return

        configured_required = (
            required_columns
            if required_columns is not None
            else self.config.required_columns
        )

        if configured_required:
            normalized_required = {
                self._normalize_column_name(column)
                for column in configured_required
            }

            actual_columns = set(dataframe.columns)

            missing_columns = sorted(
                normalized_required - actual_columns
            )

            if missing_columns:
                report.add_error(
                    "Missing required column(s): "
                    f"{missing_columns}"
                )

        if dataframe.index.has_duplicates:
            report.add_warning(
                "DataFrame index contains duplicate values."
            )

        for column in dataframe.columns:
            if dataframe[column].isna().all():
                report.add_warning(
                    f"Column '{column}' contains only missing values."
                )

    def _normalize_column_name(
        self,
        column: str,
    ) -> str:
        name = str(column).strip().lower()

        name = re.sub(
            r"[^a-z0-9]+",
            "_",
            name,
        )

        return re.sub(
            r"_+",
            "_",
            name,
        ).strip("_")

    def update_missing_value_report(
        self,
        dataframe: pd.DataFrame,
        report: CSVQualityReport,
    ) -> None:
        if len(dataframe) == 0:
            return

        missing_counts = dataframe.isna().sum()

        for column, count in missing_counts.items():
            count = int(count)

            if count <= 0:
                continue

            percentage = round(
                count / len(dataframe) * 100,
                2,
            )

            report.missing_values_by_column[column] = count

            report.missing_percentage_by_column[column] = (
                percentage
            )

            if percentage >= 50:
                report.add_warning(
                    f"Column '{column}' has "
                    f"{percentage}% missing values."
                )


# ============================================================================
# Singleton
# ============================================================================


csv_parser = CSVParser()