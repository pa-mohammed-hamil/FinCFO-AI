"""
FinCo AI - Duplicate Detector

Responsible for:
- Detecting duplicate financial records
- Detecting duplicate transactions by ID
- Detecting duplicate documents by file hash
- Detecting duplicate text content
- Detecting near-duplicate records
- Generating duplicate detection reports

This module is intentionally independent from:
    - CSVParser
    - DocumentParser
    - Database repositories
    - Ingestion pipelines

Example:

    from app.ingestion.duplicate_detector import duplicate_detector

    result = duplicate_detector.detect_dataframe(
        dataframe=df,
        subset=["transaction_id"],
    )

    print(result.duplicate_count)
    print(result.duplicate_indices)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class DuplicateDetectorError(Exception):
    """Base exception for duplicate detection errors."""


class DuplicateConfigurationError(DuplicateDetectorError):
    """Raised when duplicate detector configuration is invalid."""


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class DuplicateDetectorConfig:
    """
    Configuration for duplicate detection.
    """

    normalize_text: bool = True

    case_sensitive: bool = False

    strip_whitespace: bool = True

    remove_punctuation: bool = False

    treat_missing_values_as_equal: bool = True

    detect_exact_row_duplicates: bool = True

    detect_duplicate_ids: bool = True

    detect_near_duplicates: bool = False

    similarity_threshold: float = 0.95

    max_comparison_rows: int = 10_000

    default_id_columns: tuple[str, ...] = (
        "transaction_id",
        "document_id",
        "invoice_id",
        "payment_id",
        "reference_id",
        "record_id",
        "id",
    )

    default_text_columns: tuple[str, ...] = (
        "description",
        "details",
        "narration",
        "memo",
        "text",
        "content",
    )


# ============================================================================
# Result Models
# ============================================================================


@dataclass
class DuplicateGroup:
    """
    Represents a group of duplicate records.
    """

    group_id: str

    indices: List[Any] = field(
        default_factory=list
    )

    duplicate_type: str = "exact"

    matching_columns: List[str] = field(
        default_factory=list
    )

    representative_index: Optional[Any] = None

    similarity_score: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "indices": self.indices,
            "duplicate_type": self.duplicate_type,
            "matching_columns": self.matching_columns,
            "representative_index": self.representative_index,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
        }


@dataclass
class DuplicateDetectionReport:
    """
    Summary of duplicate detection.
    """

    source_name: Optional[str] = None

    total_records: int = 0

    unique_records: int = 0

    duplicate_records: int = 0

    duplicate_groups: int = 0

    duplicate_ids: int = 0

    exact_duplicate_groups: int = 0

    near_duplicate_groups: int = 0

    duplicate_indices: List[Any] = field(
        default_factory=list
    )

    groups: List[DuplicateGroup] = field(
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
            "source_name": self.source_name,
            "total_records": self.total_records,
            "unique_records": self.unique_records,
            "duplicate_records": self.duplicate_records,
            "duplicate_groups": self.duplicate_groups,
            "duplicate_ids": self.duplicate_ids,
            "exact_duplicate_groups": (
                self.exact_duplicate_groups
            ),
            "near_duplicate_groups": (
                self.near_duplicate_groups
            ),
            "duplicate_indices": self.duplicate_indices,
            "groups": [
                group.to_dict()
                for group in self.groups
            ],
            "warnings": self.warnings,
            "errors": self.errors,
            "is_valid": self.is_valid,
        }


@dataclass
class DuplicateDetectionResult:
    """
    Final result of duplicate detection.
    """

    dataframe: Optional[pd.DataFrame]

    report: DuplicateDetectionReport

    success: bool = True

    @property
    def has_duplicates(self) -> bool:
        return self.report.duplicate_records > 0

    @property
    def duplicate_count(self) -> int:
        return self.report.duplicate_records

    @property
    def duplicate_indices(self) -> List[Any]:
        return self.report.duplicate_indices

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "has_duplicates": self.has_duplicates,
            "duplicate_count": self.duplicate_count,
            "duplicate_indices": self.duplicate_indices,
            "report": self.report.to_dict(),
        }


# ============================================================================
# Duplicate Detector
# ============================================================================


class DuplicateDetector:
    """
    Detects duplicate financial records and documents.

    Supported operations:
        - DataFrame duplicate detection
        - ID-based duplicate detection
        - Exact text duplicate detection
        - Near-duplicate text detection
        - File hash comparison
        - Text hashing
        - Record hashing
    """

    def __init__(
        self,
        config: DuplicateDetectorConfig = DuplicateDetectorConfig(),
    ) -> None:
        self.config = config

        self._validate_config()

    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------

    def _validate_config(self) -> None:
        if not (
            0.0
            <= self.config.similarity_threshold
            <= 1.0
        ):
            raise DuplicateConfigurationError(
                "similarity_threshold must be between 0 and 1."
            )

        if self.config.max_comparison_rows <= 0:
            raise DuplicateConfigurationError(
                "max_comparison_rows must be greater than zero."
            )

    # ---------------------------------------------------------------------
    # DataFrame detection
    # ---------------------------------------------------------------------

    def detect_dataframe(
        self,
        dataframe: pd.DataFrame,
        subset: Optional[Sequence[str]] = None,
        source_name: Optional[str] = None,
        remove_duplicates: bool = False,
    ) -> DuplicateDetectionResult:
        """
        Detect duplicate rows in a DataFrame.

        Args:
            dataframe:
                Input DataFrame.

            subset:
                Columns used to determine duplicates.
                If omitted, full rows are compared.

            source_name:
                Optional source identifier.

            remove_duplicates:
                If True, returned DataFrame contains only the
                first record from each duplicate group.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        report = DuplicateDetectionReport(
            source_name=source_name,
            total_records=len(dataframe),
        )

        if dataframe.empty:
            report.unique_records = 0

            return DuplicateDetectionResult(
                dataframe=dataframe.copy(),
                report=report,
                success=True,
            )

        if len(dataframe) > self.config.max_comparison_rows:
            report.add_warning(
                f"DataFrame contains {len(dataframe)} rows, "
                f"which exceeds the recommended comparison limit "
                f"of {self.config.max_comparison_rows}."
            )

        normalized_subset = self._validate_subset(
            dataframe=dataframe,
            subset=subset,
        )

        duplicate_mask = dataframe.duplicated(
            subset=normalized_subset,
            keep="first",
        )

        duplicate_indices = list(
            dataframe.index[duplicate_mask]
        )

        report.duplicate_indices = duplicate_indices
        report.duplicate_records = len(
            duplicate_indices
        )

        report.unique_records = (
            report.total_records
            - report.duplicate_records
        )

        groups = self._build_dataframe_duplicate_groups(
            dataframe=dataframe,
            subset=normalized_subset,
        )

        report.groups.extend(groups)

        report.duplicate_groups = len(groups)

        report.exact_duplicate_groups = len(groups)

        if remove_duplicates:
            cleaned_dataframe = dataframe.loc[
                ~duplicate_mask
            ].copy()

        else:
            cleaned_dataframe = dataframe.copy()

        return DuplicateDetectionResult(
            dataframe=cleaned_dataframe,
            report=report,
            success=report.is_valid,
        )

    def _validate_subset(
        self,
        dataframe: pd.DataFrame,
        subset: Optional[Sequence[str]],
    ) -> Optional[List[str]]:
        if subset is None:
            return None

        normalized_subset = [
            str(column)
            for column in subset
        ]

        missing_columns = [
            column
            for column in normalized_subset
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise DuplicateDetectorError(
                "Duplicate detection columns not found: "
                f"{missing_columns}"
            )

        return normalized_subset

    def _build_dataframe_duplicate_groups(
        self,
        dataframe: pd.DataFrame,
        subset: Optional[Sequence[str]],
    ) -> List[DuplicateGroup]:
        """
        Builds duplicate groups using normalized record keys.
        """

        groups_by_key: Dict[
            Tuple[Any, ...],
            List[Any],
        ] = {}

        columns = (
            list(subset)
            if subset is not None
            else list(dataframe.columns)
        )

        for index, row in dataframe.iterrows():
            key = tuple(
                self.normalize_value(
                    row[column]
                )
                for column in columns
            )

            groups_by_key.setdefault(
                key,
                [],
            ).append(index)

        groups: List[DuplicateGroup] = []

        counter = 1

        for key, indices in groups_by_key.items():
            if len(indices) <= 1:
                continue

            groups.append(
                DuplicateGroup(
                    group_id=f"duplicate_group_{counter}",
                    indices=list(indices),
                    duplicate_type="exact",
                    matching_columns=columns,
                    representative_index=indices[0],
                    metadata={
                        "normalized_key": [
                            self.safe_serialize(value)
                            for value in key
                        ],
                    },
                )
            )

            counter += 1

        return groups

    # ---------------------------------------------------------------------
    # ID-based detection
    # ---------------------------------------------------------------------

    def detect_duplicate_ids(
        self,
        dataframe: pd.DataFrame,
        id_columns: Optional[Sequence[str]] = None,
        source_name: Optional[str] = None,
    ) -> DuplicateDetectionResult:
        """
        Detects duplicates using transaction/document identifiers.

        If multiple ID columns are provided, the first available
        column is used for each record.
        """

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        report = DuplicateDetectionReport(
            source_name=source_name,
            total_records=len(dataframe),
        )

        candidate_columns = list(
            id_columns
            or self.config.default_id_columns
        )

        available_columns = [
            column
            for column in candidate_columns
            if column in dataframe.columns
        ]

        if not available_columns:
            report.add_warning(
                "No ID columns were found for duplicate ID detection."
            )

            report.unique_records = len(dataframe)

            return DuplicateDetectionResult(
                dataframe=dataframe.copy(),
                report=report,
                success=True,
            )

        selected_column = available_columns[0]

        series = dataframe[selected_column].map(
            self.normalize_value
        )

        valid_mask = series.notna()

        duplicate_mask = (
            series.notna()
            & series.duplicated(
                keep="first"
            )
        )

        duplicate_indices = list(
            dataframe.index[duplicate_mask]
        )

        report.duplicate_indices = duplicate_indices
        report.duplicate_records = len(
            duplicate_indices
        )

        report.unique_records = (
            len(dataframe)
            - report.duplicate_records
        )

        report.duplicate_ids = (
            series[duplicate_mask].nunique()
        )

        groups_by_id: Dict[Any, List[Any]] = {}

        for index, value in series[valid_mask].items():
            groups_by_id.setdefault(
                value,
                [],
            ).append(index)

        group_counter = 1

        for identifier, indices in groups_by_id.items():
            if len(indices) <= 1:
                continue

            report.groups.append(
                DuplicateGroup(
                    group_id=f"duplicate_id_group_{group_counter}",
                    indices=list(indices),
                    duplicate_type="duplicate_id",
                    matching_columns=[
                        selected_column
                    ],
                    representative_index=indices[0],
                    metadata={
                        "duplicate_id": self.safe_serialize(
                            identifier
                        ),
                    },
                )
            )

            group_counter += 1

        report.duplicate_groups = len(
            report.groups
        )

        report.exact_duplicate_groups = len(
            report.groups
        )

        return DuplicateDetectionResult(
            dataframe=dataframe.copy(),
            report=report,
            success=report.is_valid,
        )

    # ---------------------------------------------------------------------
    # Text duplicate detection
    # ---------------------------------------------------------------------

    def detect_text_duplicates(
        self,
        texts: Sequence[str],
        source_name: Optional[str] = None,
        detect_near_duplicates: Optional[bool] = None,
    ) -> DuplicateDetectionReport:
        """
        Detects exact and optionally near-duplicate text values.
        """

        report = DuplicateDetectionReport(
            source_name=source_name,
            total_records=len(texts),
        )

        if not texts:
            return report

        normalized_texts = [
            self.normalize_text(text)
            for text in texts
        ]

        groups_by_hash: Dict[
            str,
            List[int],
        ] = {}

        for index, text in enumerate(normalized_texts):
            text_hash = self.hash_text(
                text
            )

            groups_by_hash.setdefault(
                text_hash,
                [],
            ).append(index)

        group_counter = 1

        for text_hash, indices in groups_by_hash.items():
            if len(indices) <= 1:
                continue

            report.groups.append(
                DuplicateGroup(
                    group_id=f"text_duplicate_group_{group_counter}",
                    indices=indices,
                    duplicate_type="exact_text",
                    matching_columns=["text"],
                    representative_index=indices[0],
                    similarity_score=1.0,
                    metadata={
                        "text_hash": text_hash,
                    },
                )
            )

            group_counter += 1

        if (
            detect_near_duplicates
            if detect_near_duplicates is not None
            else self.config.detect_near_duplicates
        ):
            self._detect_near_text_groups(
                normalized_texts=normalized_texts,
                report=report,
            )

        report.duplicate_groups = len(
            report.groups
        )

        report.exact_duplicate_groups = sum(
            1
            for group in report.groups
            if group.duplicate_type == "exact_text"
        )

        report.near_duplicate_groups = sum(
            1
            for group in report.groups
            if group.duplicate_type == "near_duplicate_text"
        )

        duplicate_indices = set()

        for group in report.groups:
            duplicate_indices.update(
                group.indices[1:]
            )

        report.duplicate_indices = sorted(
            duplicate_indices
        )

        report.duplicate_records = len(
            report.duplicate_indices
        )

        report.unique_records = (
            report.total_records
            - report.duplicate_records
        )

        return report

    def _detect_near_text_groups(
        self,
        normalized_texts: Sequence[str],
        report: DuplicateDetectionReport,
    ) -> None:
        """
        Uses token-based Jaccard similarity for near-duplicates.

        This avoids an external ML dependency and works well for
        moderately sized document collections.
        """

        exact_pairs = {
            tuple(sorted(group.indices))
            for group in report.groups
            if group.duplicate_type == "exact_text"
        }

        token_sets = [
            set(text.split())
            for text in normalized_texts
        ]

        group_counter = (
            len(report.groups) + 1
        )

        for left_index in range(
            len(token_sets)
        ):
            for right_index in range(
                left_index + 1,
                len(token_sets),
            ):
                pair = (
                    left_index,
                    right_index,
                )

                if pair in exact_pairs:
                    continue

                left_tokens = token_sets[left_index]
                right_tokens = token_sets[right_index]

                if not left_tokens or not right_tokens:
                    continue

                intersection = len(
                    left_tokens & right_tokens
                )

                union = len(
                    left_tokens | right_tokens
                )

                if union == 0:
                    continue

                similarity = (
                    intersection / union
                )

                if (
                    similarity
                    >= self.config.similarity_threshold
                ):
                    report.groups.append(
                        DuplicateGroup(
                            group_id=(
                                f"near_duplicate_group_"
                                f"{group_counter}"
                            ),
                            indices=[
                                left_index,
                                right_index,
                            ],
                            duplicate_type=(
                                "near_duplicate_text"
                            ),
                            matching_columns=["text"],
                            representative_index=left_index,
                            similarity_score=round(
                                similarity,
                                4,
                            ),
                        )
                    )

                    group_counter += 1

    # ---------------------------------------------------------------------
    # File and document hashes
    # ---------------------------------------------------------------------

    def hash_bytes(
        self,
        content: bytes,
        algorithm: str = "sha256",
    ) -> str:
        """
        Returns a hexadecimal hash for binary content.
        """

        try:
            digest = hashlib.new(
                algorithm
            )

        except ValueError as exc:
            raise DuplicateDetectorError(
                f"Unsupported hash algorithm: {algorithm}"
            ) from exc

        digest.update(content)

        return digest.hexdigest()

    def hash_file(
        self,
        file_path: str | Path,
        algorithm: str = "sha256",
        chunk_size: int = 1024 * 1024,
    ) -> str:
        """
        Hashes a file without loading the entire file into memory.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {path}"
            )

        try:
            digest = hashlib.new(
                algorithm
            )

        except ValueError as exc:
            raise DuplicateDetectorError(
                f"Unsupported hash algorithm: {algorithm}"
            ) from exc

        with path.open("rb") as file:
            while True:
                chunk = file.read(
                    chunk_size
                )

                if not chunk:
                    break

                digest.update(chunk)

        return digest.hexdigest()

    def compare_files(
        self,
        first_file: str | Path,
        second_file: str | Path,
    ) -> bool:
        """
        Compares two files using SHA-256 hashes.
        """

        return (
            self.hash_file(first_file)
            == self.hash_file(second_file)
        )

    # ---------------------------------------------------------------------
    # Record and text hashing
    # ---------------------------------------------------------------------

    def hash_text(
        self,
        text: str,
        algorithm: str = "sha256",
    ) -> str:
        normalized = self.normalize_text(
            text
        )

        return self.hash_bytes(
            normalized.encode("utf-8"),
            algorithm=algorithm,
        )

    def hash_record(
        self,
        record: Dict[str, Any],
        columns: Optional[Sequence[str]] = None,
        algorithm: str = "sha256",
    ) -> str:
        """
        Hashes a dictionary record deterministically.
        """

        selected_columns = (
            list(columns)
            if columns is not None
            else sorted(record.keys())
        )

        parts: List[str] = []

        for column in selected_columns:
            value = record.get(column)

            normalized_value = self.normalize_value(
                value
            )

            parts.append(
                f"{column}={self.safe_serialize(normalized_value)}"
            )

        canonical_record = "|".join(parts)

        return self.hash_bytes(
            canonical_record.encode("utf-8"),
            algorithm=algorithm,
        )

    # ---------------------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------------------

    def normalize_text(
        self,
        text: Any,
    ) -> str:
        """
        Normalizes text for reliable duplicate comparison.
        """

        if text is None:
            return ""

        if self._is_missing_value(text):
            return ""

        value = str(text)

        if self.config.strip_whitespace:
            value = value.strip()

        if self.config.normalize_text:
            value = re.sub(
                r"\s+",
                " ",
                value,
            )

        if self.config.remove_punctuation:
            value = re.sub(
                r"[^\w\s]",
                "",
                value,
            )

        if not self.config.case_sensitive:
            value = value.casefold()

        return value

    def normalize_value(
        self,
        value: Any,
    ) -> Any:
        """
        Normalizes values while preserving useful numeric types.
        """

        if value is None:
            return None

        if self._is_missing_value(value):
            return (
                None
                if self.config.treat_missing_values_as_equal
                else "<missing>"
            )

        if isinstance(value, str):
            return self.normalize_text(
                value
            )

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        if isinstance(value, float):
            if math.isnan(value):
                return None

            if math.isinf(value):
                return str(value)

            return round(value, 10)

        if isinstance(value, (list, tuple)):
            return tuple(
                self.normalize_value(item)
                for item in value
            )

        if isinstance(value, dict):
            return tuple(
                sorted(
                    (
                        str(key),
                        self.normalize_value(item),
                    )
                    for key, item in value.items()
                )
            )

        return value

    def _is_missing_value(
        self,
        value: Any,
    ) -> bool:
        if value is None:
            return True

        if value is pd.NA:
            return True

        try:
            result = pd.isna(value)

            if isinstance(result, bool):
                return result

        except Exception:
            return False

        return False

    def safe_serialize(
        self,
        value: Any,
    ) -> Any:
        """
        Converts values into JSON-friendly representations.
        """

        if value is None:
            return None

        if isinstance(value, tuple):
            return [
                self.safe_serialize(item)
                for item in value
            ]

        if isinstance(value, list):
            return [
                self.safe_serialize(item)
                for item in value
            ]

        if isinstance(value, dict):
            return {
                str(key): self.safe_serialize(item)
                for key, item in value.items()
            }

        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()

        if isinstance(value, (str, int, float, bool)):
            return value

        return str(value)


# ============================================================================
# Singleton
# ============================================================================


duplicate_detector = DuplicateDetector()