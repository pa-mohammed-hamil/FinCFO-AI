"""
FinCo AI - Excel Parser

Provides production-oriented Excel parsing for:
- XLSX files
- XLS files when the required engine is installed
- Multiple worksheets
- Header detection
- Empty row/column removal
- Basic data-quality reporting
- Date and numeric normalization
- Sheet-level parsing
- Workbook metadata extraction

Dependencies:
    pandas
    openpyxl

Optional dependency for legacy .xls files:
    xlrd

Example:

    from app.ingestion.excel_parser import excel_parser

    result = excel_parser.parse(
        file_path="data/raw/financial_report.xlsx"
    )

    print(result.sheet_names)
    print(result.dataframes["Transactions"])
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class ExcelParserError(Exception):
    """Base exception for Excel parser errors."""


class ExcelFileNotFoundError(ExcelParserError):
    """Raised when the Excel file does not exist."""


class ExcelFormatError(ExcelParserError):
    """Raised when the file is not a supported Excel workbook."""


class ExcelSheetError(ExcelParserError):
    """Raised when a worksheet cannot be parsed."""


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class ExcelParserConfig:
    """
    Configuration for Excel parsing.
    """

    engine: Optional[str] = None

    header_row: Optional[int] = 0

    read_all_sheets: bool = True

    drop_empty_rows: bool = True

    drop_empty_columns: bool = True

    trim_column_names: bool = True

    normalize_column_names: bool = False

    infer_dates: bool = True

    infer_numeric_values: bool = True

    preserve_empty_sheets: bool = False

    max_rows_per_sheet: Optional[int] = None

    max_columns_per_sheet: Optional[int] = None

    supported_extensions: tuple[str, ...] = (
        ".xlsx",
        ".xlsm",
        ".xltx",
        ".xltm",
        ".xls",
    )


# ============================================================================
# Result Models
# ============================================================================


@dataclass
class ExcelQualityReport:
    """
    Data-quality report for one worksheet.
    """

    sheet_name: str

    original_rows: int = 0

    original_columns: int = 0

    final_rows: int = 0

    final_columns: int = 0

    empty_rows_removed: int = 0

    empty_columns_removed: int = 0

    duplicate_column_names: List[str] = field(
        default_factory=list
    )

    missing_value_counts: Dict[str, int] = field(
        default_factory=dict
    )

    inferred_date_columns: List[str] = field(
        default_factory=list
    )

    inferred_numeric_columns: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

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
            "sheet_name": self.sheet_name,
            "original_rows": self.original_rows,
            "original_columns": self.original_columns,
            "final_rows": self.final_rows,
            "final_columns": self.final_columns,
            "empty_rows_removed": self.empty_rows_removed,
            "empty_columns_removed": self.empty_columns_removed,
            "duplicate_column_names": (
                self.duplicate_column_names
            ),
            "missing_value_counts": (
                self.missing_value_counts
            ),
            "inferred_date_columns": (
                self.inferred_date_columns
            ),
            "inferred_numeric_columns": (
                self.inferred_numeric_columns
            ),
            "warnings": self.warnings,
            "errors": self.errors,
            "is_valid": self.is_valid,
        }


@dataclass
class ExcelSheetResult:
    """
    Parsed worksheet result.
    """

    sheet_name: str

    dataframe: pd.DataFrame

    quality_report: ExcelQualityReport

    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_name": self.sheet_name,
            "rows": len(self.dataframe),
            "columns": list(
                self.dataframe.columns
            ),
            "quality_report": (
                self.quality_report.to_dict()
            ),
            "success": self.success,
        }


@dataclass
class ExcelParseResult:
    """
    Final Excel parsing result.
    """

    file_path: str

    file_name: str

    file_extension: str

    sheet_names: List[str]

    dataframes: Dict[str, pd.DataFrame]

    sheet_results: Dict[str, ExcelSheetResult]

    workbook_metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    success: bool = True

    @property
    def total_rows(self) -> int:
        return sum(
            len(dataframe)
            for dataframe in self.dataframes.values()
        )

    @property
    def total_sheets(self) -> int:
        return len(self.dataframes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_extension": self.file_extension,
            "sheet_names": self.sheet_names,
            "total_sheets": self.total_sheets,
            "total_rows": self.total_rows,
            "workbook_metadata": self.workbook_metadata,
            "warnings": self.warnings,
            "errors": self.errors,
            "success": self.success,
            "sheet_results": {
                name: result.to_dict()
                for name, result in self.sheet_results.items()
            },
        }


# ============================================================================
# Excel Parser
# ============================================================================


class ExcelParser:
    """
    Parses Excel workbooks into pandas DataFrames.

    The parser:
        1. Validates the file.
        2. Loads one or all worksheets.
        3. Cleans empty rows and columns.
        4. Normalizes column names.
        5. Attempts safe date and numeric inference.
        6. Produces quality reports.
    """

    def __init__(
        self,
        config: ExcelParserConfig = ExcelParserConfig(),
    ) -> None:
        self.config = config

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def parse(
        self,
        file_path: str | Path,
        sheet_name: Optional[str] = None,
        usecols: Optional[str | Sequence[str]] = None,
        source_name: Optional[str] = None,
    ) -> ExcelParseResult:
        """
        Parse an Excel workbook.

        Args:
            file_path:
                Path to the Excel workbook.

            sheet_name:
                Optional worksheet name. If omitted, all worksheets
                are parsed when read_all_sheets=True.

            usecols:
                Optional pandas usecols expression or list of columns.

            source_name:
                Optional logical source name.

        Returns:
            ExcelParseResult
        """

        path = Path(file_path)

        self._validate_file(path)

        extension = path.suffix.lower()

        engine = self._resolve_engine(
            extension
        )

        workbook = self._load_workbook(
            path=path,
            engine=engine,
        )

        workbook_sheet_names = list(
            workbook.sheet_names
        )

        if sheet_name is not None:
            if sheet_name not in workbook_sheet_names:
                raise ExcelSheetError(
                    f"Worksheet '{sheet_name}' was not found. "
                    f"Available sheets: {workbook_sheet_names}"
                )

            selected_sheet_names = [
                sheet_name
            ]

        elif self.config.read_all_sheets:
            selected_sheet_names = (
                workbook_sheet_names
            )

        else:
            selected_sheet_names = (
                workbook_sheet_names[:1]
            )

        dataframes: Dict[
            str,
            pd.DataFrame,
        ] = {}

        sheet_results: Dict[
            str,
            ExcelSheetResult,
        ] = {}

        warnings: List[str] = []
        errors: List[str] = []

        for current_sheet_name in selected_sheet_names:
            try:
                dataframe = self._read_sheet(
                    path=path,
                    sheet_name=current_sheet_name,
                    engine=engine,
                    usecols=usecols,
                )

                sheet_result = self.parse_dataframe(
                    dataframe=dataframe,
                    sheet_name=current_sheet_name,
                )

                if (
                    not self.config.preserve_empty_sheets
                    and sheet_result.dataframe.empty
                ):
                    warnings.append(
                        f"Skipped empty worksheet: "
                        f"{current_sheet_name}"
                    )
                    continue

                dataframes[
                    current_sheet_name
                ] = sheet_result.dataframe

                sheet_results[
                    current_sheet_name
                ] = sheet_result

            except Exception as exc:
                message = (
                    f"Failed to parse worksheet "
                    f"'{current_sheet_name}': {exc}"
                )

                logger.exception(message)
                errors.append(message)

        success = len(errors) == 0

        return ExcelParseResult(
            file_path=str(path),
            file_name=path.name,
            file_extension=extension,
            sheet_names=list(dataframes.keys()),
            dataframes=dataframes,
            sheet_results=sheet_results,
            workbook_metadata={
                "source_name": source_name,
                "available_sheet_names": (
                    workbook_sheet_names
                ),
                "selected_sheet_names": (
                    selected_sheet_names
                ),
                "engine": engine,
                "file_size_bytes": path.stat().st_size,
            },
            warnings=warnings,
            errors=errors,
            success=success,
        )

    def parse_dataframe(
        self,
        dataframe: pd.DataFrame,
        sheet_name: str = "Sheet1",
    ) -> ExcelSheetResult:
        """
        Clean and analyze a DataFrame obtained from Excel.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        report = ExcelQualityReport(
            sheet_name=sheet_name,
            original_rows=len(dataframe),
            original_columns=len(dataframe.columns),
        )

        cleaned = dataframe.copy()

        cleaned = self._clean_column_names(
            cleaned,
            report,
        )

        if self.config.drop_empty_rows:
            before_rows = len(cleaned)

            cleaned = cleaned.dropna(
                axis=0,
                how="all",
            )

            report.empty_rows_removed = (
                before_rows - len(cleaned)
            )

        if self.config.drop_empty_columns:
            before_columns = len(cleaned.columns)

            cleaned = cleaned.dropna(
                axis=1,
                how="all",
            )

            report.empty_columns_removed = (
                before_columns - len(cleaned.columns)
            )

        if self.config.max_rows_per_sheet is not None:
            if len(cleaned) > self.config.max_rows_per_sheet:
                report.add_warning(
                    f"Worksheet exceeded max row limit of "
                    f"{self.config.max_rows_per_sheet}. "
                    "Rows were truncated."
                )

                cleaned = cleaned.iloc[
                    : self.config.max_rows_per_sheet
                ]

        if self.config.max_columns_per_sheet is not None:
            if (
                len(cleaned.columns)
                > self.config.max_columns_per_sheet
            ):
                report.add_warning(
                    f"Worksheet exceeded max column limit of "
                    f"{self.config.max_columns_per_sheet}. "
                    "Columns were truncated."
                )

                cleaned = cleaned.iloc[
                    :,
                    : self.config.max_columns_per_sheet,
                ]

        if self.config.infer_dates:
            cleaned = self._infer_date_columns(
                cleaned,
                report,
            )

        if self.config.infer_numeric_values:
            cleaned = self._infer_numeric_columns(
                cleaned,
                report,
            )

        report.missing_value_counts = {
            str(column): int(
                cleaned[column].isna().sum()
            )
            for column in cleaned.columns
        }

        report.final_rows = len(cleaned)
        report.final_columns = len(
            cleaned.columns
        )

        return ExcelSheetResult(
            sheet_name=sheet_name,
            dataframe=cleaned,
            quality_report=report,
            success=report.is_valid,
        )

    # ---------------------------------------------------------------------
    # File handling
    # ---------------------------------------------------------------------

    def _validate_file(
        self,
        path: Path,
    ) -> None:
        if not path.exists():
            raise ExcelFileNotFoundError(
                f"Excel file not found: {path}"
            )

        if not path.is_file():
            raise ExcelFormatError(
                f"Excel path is not a file: {path}"
            )

        if path.suffix.lower() not in (
            self.config.supported_extensions
        ):
            raise ExcelFormatError(
                f"Unsupported Excel extension: "
                f"{path.suffix}. Supported extensions: "
                f"{self.config.supported_extensions}"
            )

    def _resolve_engine(
        self,
        extension: str,
    ) -> Optional[str]:
        if self.config.engine:
            return self.config.engine

        if extension in {
            ".xlsx",
            ".xlsm",
            ".xltx",
            ".xltm",
        }:
            return "openpyxl"

        if extension == ".xls":
            return "xlrd"

        return None

    def _load_workbook(
        self,
        path: Path,
        engine: Optional[str],
    ) -> pd.ExcelFile:
        try:
            return pd.ExcelFile(
                path,
                engine=engine,
            )

        except ImportError as exc:
            raise ExcelParserError(
                "Required Excel engine is not installed. "
                "Install openpyxl for .xlsx files or xlrd "
                "for legacy .xls files."
            ) from exc

        except Exception as exc:
            raise ExcelFormatError(
                f"Unable to open Excel workbook '{path}': {exc}"
            ) from exc

    def _read_sheet(
        self,
        path: Path,
        sheet_name: str,
        engine: Optional[str],
        usecols: Optional[str | Sequence[str]],
    ) -> pd.DataFrame:
        try:
            return pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=self.config.header_row,
                engine=engine,
                usecols=usecols,
            )

        except Exception as exc:
            raise ExcelSheetError(
                f"Unable to read worksheet "
                f"'{sheet_name}': {exc}"
            ) from exc

    # ---------------------------------------------------------------------
    # Column cleaning
    # ---------------------------------------------------------------------

    def _clean_column_names(
        self,
        dataframe: pd.DataFrame,
        report: ExcelQualityReport,
    ) -> pd.DataFrame:
        cleaned = dataframe.copy()

        original_columns = [
            str(column)
            for column in cleaned.columns
        ]

        if self.config.trim_column_names:
            cleaned.columns = [
                column.strip()
                for column in original_columns
            ]

        if self.config.normalize_column_names:
            cleaned.columns = [
                self.normalize_column_name(
                    column
                )
                for column in cleaned.columns
            ]

        duplicate_columns = self._find_duplicates(
            list(cleaned.columns)
        )

        if duplicate_columns:
            report.duplicate_column_names = (
                duplicate_columns
            )

            report.add_warning(
                "Duplicate column names detected: "
                f"{duplicate_columns}"
            )

            cleaned.columns = (
                self._make_unique_column_names(
                    list(cleaned.columns)
                )
            )

        return cleaned

    def normalize_column_name(
        self,
        column_name: Any,
    ) -> str:
        """
        Converts a column name into a stable snake_case name.
        """

        value = str(column_name).strip()

        value = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            value,
        )

        value = re.sub(
            r"_+",
            "_",
            value,
        )

        return value.strip("_").lower()

    def _find_duplicates(
        self,
        values: Sequence[str],
    ) -> List[str]:
        seen = set()
        duplicates = []

        for value in values:
            if value in seen and value not in duplicates:
                duplicates.append(value)

            seen.add(value)

        return duplicates

    def _make_unique_column_names(
        self,
        columns: Sequence[str],
    ) -> List[str]:
        counts: Dict[str, int] = {}
        result: List[str] = []

        for column in columns:
            count = counts.get(column, 0)

            if count == 0:
                result.append(column)

            else:
                result.append(
                    f"{column}_{count}"
                )

            counts[column] = count + 1

        return result

    # ---------------------------------------------------------------------
    # Type inference
    # ---------------------------------------------------------------------

    def _infer_date_columns(
        self,
        dataframe: pd.DataFrame,
        report: ExcelQualityReport,
    ) -> pd.DataFrame:
        cleaned = dataframe.copy()

        for column in cleaned.columns:
            series = cleaned[column]

            if pd.api.types.is_datetime64_any_dtype(
                series
            ):
                report.inferred_date_columns.append(
                    str(column)
                )
                continue

            if not self._looks_like_date_column(
                str(column)
            ):
                continue

            converted = pd.to_datetime(
                series,
                errors="coerce",
                dayfirst=False,
            )

            non_empty_count = series.notna().sum()

            if non_empty_count == 0:
                continue

            valid_count = converted.notna().sum()

            conversion_ratio = (
                valid_count / non_empty_count
            )

            if conversion_ratio >= 0.80:
                cleaned[column] = converted

                report.inferred_date_columns.append(
                    str(column)
                )

        return cleaned

    def _looks_like_date_column(
        self,
        column_name: str,
    ) -> bool:
        normalized = (
            column_name.lower()
            .replace("_", " ")
            .strip()
        )

        date_terms = (
            "date",
            "time",
            "timestamp",
            "created",
            "updated",
            "period",
            "month",
            "year",
            "due",
            "expiry",
            "expiration",
        )

        return any(
            term in normalized
            for term in date_terms
        )

    def _infer_numeric_columns(
        self,
        dataframe: pd.DataFrame,
        report: ExcelQualityReport,
    ) -> pd.DataFrame:
        cleaned = dataframe.copy()

        for column in cleaned.columns:
            series = cleaned[column]

            if pd.api.types.is_numeric_dtype(
                series
            ):
                report.inferred_numeric_columns.append(
                    str(column)
                )
                continue

            if not self._looks_like_numeric_column(
                str(column)
            ):
                continue

            converted = self._convert_numeric_series(
                series
            )

            non_empty_count = series.notna().sum()

            if non_empty_count == 0:
                continue

            valid_count = converted.notna().sum()

            conversion_ratio = (
                valid_count / non_empty_count
            )

            if conversion_ratio >= 0.80:
                cleaned[column] = converted

                report.inferred_numeric_columns.append(
                    str(column)
                )

        return cleaned

    def _looks_like_numeric_column(
        self,
        column_name: str,
    ) -> bool:
        normalized = (
            column_name.lower()
            .replace("_", " ")
            .strip()
        )

        numeric_terms = (
            "amount",
            "balance",
            "revenue",
            "expense",
            "income",
            "profit",
            "loss",
            "price",
            "cost",
            "quantity",
            "qty",
            "total",
            "tax",
            "fee",
            "rate",
            "percentage",
            "percent",
            "debit",
            "credit",
        )

        return any(
            term in normalized
            for term in numeric_terms
        )

    def _convert_numeric_series(
        self,
        series: pd.Series,
    ) -> pd.Series:
        """
        Converts common financial numeric formats.

        Examples:
            ₹1,250.50
            $1,250.50
            (500.00)
            10%
            1,000
        """

        if pd.api.types.is_numeric_dtype(
            series
        ):
            return pd.to_numeric(
                series,
                errors="coerce",
            )

        cleaned = series.astype("string")

        cleaned = cleaned.str.strip()

        cleaned = cleaned.str.replace(
            ",",
            "",
            regex=False,
        )

        cleaned = cleaned.str.replace(
            r"^[₹$€£]\s*",
            "",
            regex=True,
        )

        cleaned = cleaned.str.replace(
            r"^\((.*)\)$",
            r"-\1",
            regex=True,
        )

        cleaned = cleaned.str.replace(
            "%",
            "",
            regex=False,
        )

        return pd.to_numeric(
            cleaned,
            errors="coerce",
        )


# ============================================================================
# Singleton
# ============================================================================


excel_parser = ExcelParser()