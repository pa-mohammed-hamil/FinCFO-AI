"""
FinCo AI - File Classifier

Classifies uploaded files based on:
- File extension
- MIME type
- File signature / magic bytes
- Filename patterns
- Content category
- Processing strategy

Supported file categories:
- CSV
- Excel
- PDF
- DOCX
- DOC
- TXT
- Markdown
- JSON
- XML
- Image
- Archive
- Unknown

This module does not parse file contents. It only determines the
most appropriate downstream parser and processing strategy.

Example:

    from app.ingestion.file_classifier import file_classifier

    result = file_classifier.classify(
        file_path="data/raw/financial_report.xlsx"
    )

    print(result.file_category)
    print(result.parser_name)
    print(result.confidence)
"""

from __future__ import annotations

import logging
import mimetypes
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class FileClassifierError(Exception):
    """Base exception for file-classification errors."""


class FileNotFoundForClassificationError(
    FileClassifierError
):
    """Raised when the file does not exist."""


# ============================================================================
# Enums
# ============================================================================


class FileCategory(str, Enum):
    """
    High-level file categories.
    """

    CSV = "csv"

    EXCEL = "excel"

    PDF = "pdf"

    DOCX = "docx"

    DOC = "doc"

    TEXT = "text"

    MARKDOWN = "markdown"

    JSON = "json"

    XML = "xml"

    IMAGE = "image"

    ARCHIVE = "archive"

    AUDIO = "audio"

    VIDEO = "video"

    UNKNOWN = "unknown"


class FileRiskLevel(str, Enum):
    """
    Basic file-processing risk classification.
    """

    LOW = "low"

    MEDIUM = "medium"

    HIGH = "high"

    UNKNOWN = "unknown"


# ============================================================================
# Constants
# ============================================================================


EXTENSION_CATEGORY_MAP: Dict[
    str,
    FileCategory,
] = {
    ".csv": FileCategory.CSV,
    ".tsv": FileCategory.CSV,
    ".xlsx": FileCategory.EXCEL,
    ".xlsm": FileCategory.EXCEL,
    ".xltx": FileCategory.EXCEL,
    ".xltm": FileCategory.EXCEL,
    ".xls": FileCategory.EXCEL,
    ".pdf": FileCategory.PDF,
    ".docx": FileCategory.DOCX,
    ".doc": FileCategory.DOC,
    ".txt": FileCategory.TEXT,
    ".text": FileCategory.TEXT,
    ".md": FileCategory.MARKDOWN,
    ".markdown": FileCategory.MARKDOWN,
    ".json": FileCategory.JSON,
    ".jsonl": FileCategory.JSON,
    ".xml": FileCategory.XML,
    ".html": FileCategory.TEXT,
    ".htm": FileCategory.TEXT,
    ".png": FileCategory.IMAGE,
    ".jpg": FileCategory.IMAGE,
    ".jpeg": FileCategory.IMAGE,
    ".gif": FileCategory.IMAGE,
    ".bmp": FileCategory.IMAGE,
    ".tiff": FileCategory.IMAGE,
    ".tif": FileCategory.IMAGE,
    ".webp": FileCategory.IMAGE,
    ".zip": FileCategory.ARCHIVE,
    ".tar": FileCategory.ARCHIVE,
    ".gz": FileCategory.ARCHIVE,
    ".bz2": FileCategory.ARCHIVE,
    ".7z": FileCategory.ARCHIVE,
    ".rar": FileCategory.ARCHIVE,
    ".mp3": FileCategory.AUDIO,
    ".wav": FileCategory.AUDIO,
    ".m4a": FileCategory.AUDIO,
    ".flac": FileCategory.AUDIO,
    ".mp4": FileCategory.VIDEO,
    ".avi": FileCategory.VIDEO,
    ".mov": FileCategory.VIDEO,
    ".mkv": FileCategory.VIDEO,
    ".webm": FileCategory.VIDEO,
}


MIME_CATEGORY_MAP: Dict[
    str,
    FileCategory,
] = {
    "text/csv": FileCategory.CSV,
    "text/tab-separated-values": FileCategory.CSV,
    "application/vnd.ms-excel": FileCategory.EXCEL,
    (
        "application/"
        "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ): FileCategory.EXCEL,
    (
        "application/"
        "vnd.ms-excel.sheet.macroenabled.12"
    ): FileCategory.EXCEL,
    "application/pdf": FileCategory.PDF,
    (
        "application/"
        "vnd.openxmlformats-officedocument.wordprocessingml.document"
    ): FileCategory.DOCX,
    "application/msword": FileCategory.DOC,
    "text/plain": FileCategory.TEXT,
    "text/markdown": FileCategory.MARKDOWN,
    "application/json": FileCategory.JSON,
    "application/xml": FileCategory.XML,
    "text/xml": FileCategory.XML,
    "text/html": FileCategory.TEXT,
    "image/png": FileCategory.IMAGE,
    "image/jpeg": FileCategory.IMAGE,
    "image/gif": FileCategory.IMAGE,
    "image/bmp": FileCategory.IMAGE,
    "image/tiff": FileCategory.IMAGE,
    "image/webp": FileCategory.IMAGE,
    "application/zip": FileCategory.ARCHIVE,
    "application/gzip": FileCategory.ARCHIVE,
    "application/x-tar": FileCategory.ARCHIVE,
    "audio/mpeg": FileCategory.AUDIO,
    "audio/wav": FileCategory.AUDIO,
    "video/mp4": FileCategory.VIDEO,
}


PARSER_MAP: Dict[
    FileCategory,
    str,
] = {
    FileCategory.CSV: "csv_parser",
    FileCategory.EXCEL: "excel_parser",
    FileCategory.PDF: "document_parser",
    FileCategory.DOCX: "docx_parser",
    FileCategory.DOC: "document_parser",
    FileCategory.TEXT: "document_parser",
    FileCategory.MARKDOWN: "document_parser",
    FileCategory.JSON: "json_parser",
    FileCategory.XML: "document_parser",
    FileCategory.IMAGE: "ocr_parser",
    FileCategory.ARCHIVE: "archive_parser",
    FileCategory.AUDIO: "audio_parser",
    FileCategory.VIDEO: "video_parser",
    FileCategory.UNKNOWN: "unsupported",
}


PROCESSING_STRATEGY_MAP: Dict[
    FileCategory,
    str,
] = {
    FileCategory.CSV: "tabular_ingestion",
    FileCategory.EXCEL: "tabular_ingestion",
    FileCategory.PDF: "document_ingestion",
    FileCategory.DOCX: "document_ingestion",
    FileCategory.DOC: "document_ingestion",
    FileCategory.TEXT: "text_ingestion",
    FileCategory.MARKDOWN: "text_ingestion",
    FileCategory.JSON: "structured_data_ingestion",
    FileCategory.XML: "structured_data_ingestion",
    FileCategory.IMAGE: "ocr_ingestion",
    FileCategory.ARCHIVE: "archive_inspection",
    FileCategory.AUDIO: "multimedia_transcription",
    FileCategory.VIDEO: "multimedia_transcription",
    FileCategory.UNKNOWN: "manual_review",
}


# ============================================================================
# File Signatures
# ============================================================================


@dataclass(frozen=True)
class FileSignature:
    """
    Magic-byte signature for a file format.
    """

    name: str

    category: FileCategory

    signature: bytes

    offset: int = 0

    description: str = ""


FILE_SIGNATURES: tuple[
    FileSignature,
    ...,
] = (
    FileSignature(
        name="PDF",
        category=FileCategory.PDF,
        signature=b"%PDF-",
        description="Portable Document Format",
    ),
    FileSignature(
        name="ZIP/OOXML",
        category=FileCategory.UNKNOWN,
        signature=b"PK\x03\x04",
        description="ZIP-based container such as XLSX or DOCX",
    ),
    FileSignature(
        name="OLE Compound File",
        category=FileCategory.UNKNOWN,
        signature=(
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        ),
        description="Legacy Microsoft Office compound file",
    ),
    FileSignature(
        name="PNG",
        category=FileCategory.IMAGE,
        signature=b"\x89PNG\r\n\x1a\n",
        description="Portable Network Graphics",
    ),
    FileSignature(
        name="JPEG",
        category=FileCategory.IMAGE,
        signature=b"\xff\xd8\xff",
        description="JPEG image",
    ),
    FileSignature(
        name="GIF",
        category=FileCategory.IMAGE,
        signature=b"GIF8",
        description="GIF image",
    ),
    FileSignature(
        name="GZIP",
        category=FileCategory.ARCHIVE,
        signature=b"\x1f\x8b",
        description="GZIP archive",
    ),
    FileSignature(
        name="RIFF",
        category=FileCategory.UNKNOWN,
        signature=b"RIFF",
        description="RIFF container",
    ),
)


# ============================================================================
# Result Models
# ============================================================================


@dataclass
class FileClassificationResult:
    """
    Classification result for one file.
    """

    file_path: str

    file_name: str

    extension: str

    mime_type: Optional[str]

    file_category: FileCategory

    parser_name: str

    processing_strategy: str

    confidence: float

    risk_level: FileRiskLevel

    detected_by: List[str] = field(
        default_factory=list
    )

    detected_signature: Optional[str] = None

    is_supported: bool = False

    is_binary: bool = False

    file_size_bytes: Optional[int] = None

    warnings: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "extension": self.extension,
            "mime_type": self.mime_type,
            "file_category": self.file_category.value,
            "parser_name": self.parser_name,
            "processing_strategy": (
                self.processing_strategy
            ),
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "detected_by": self.detected_by,
            "detected_signature": (
                self.detected_signature
            ),
            "is_supported": self.is_supported,
            "is_binary": self.is_binary,
            "file_size_bytes": self.file_size_bytes,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


# ============================================================================
# Classifier
# ============================================================================


class FileClassifier:
    """
    Classifies files using multiple signals.

    Classification priority:

        1. File signature
        2. MIME type
        3. File extension
        4. Filename/content heuristics
        5. Unknown fallback

    Signature detection is especially useful when the file extension
    is missing or incorrect.
    """

    def __init__(
        self,
        supported_categories: Optional[
            Sequence[FileCategory]
        ] = None,
        max_signature_read_bytes: int = 4096,
    ) -> None:
        self.supported_categories = set(
            supported_categories
            or {
                FileCategory.CSV,
                FileCategory.EXCEL,
                FileCategory.PDF,
                FileCategory.DOCX,
                FileCategory.DOC,
                FileCategory.TEXT,
                FileCategory.MARKDOWN,
                FileCategory.JSON,
                FileCategory.XML,
            }
        )

        self.max_signature_read_bytes = (
            max_signature_read_bytes
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        file_path: str | Path,
        declared_mime_type: Optional[str] = None,
        original_filename: Optional[str] = None,
    ) -> FileClassificationResult:
        """
        Classify a file.

        Args:
            file_path:
                Actual path to the file.

            declared_mime_type:
                MIME type supplied by an upload request.

            original_filename:
                Original filename supplied by the client.

        Returns:
            FileClassificationResult
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundForClassificationError(
                f"File not found: {path}"
            )

        if not path.is_file():
            raise FileClassifierError(
                f"Classification path is not a file: {path}"
            )

        extension = path.suffix.lower()

        mime_type = (
            declared_mime_type
            or mimetypes.guess_type(
                path.name
            )[0]
        )

        signature_result = (
            self._detect_signature(
                path
            )
        )

        extension_category = (
            EXTENSION_CATEGORY_MAP.get(
                extension,
                FileCategory.UNKNOWN,
            )
        )

        mime_category = (
            MIME_CATEGORY_MAP.get(
                self._normalize_mime_type(
                    mime_type
                ),
                FileCategory.UNKNOWN,
            )
        )

        signature_category = (
            signature_result["category"]
        )

        category, confidence, detected_by = (
            self._resolve_category(
                extension_category=extension_category,
                mime_category=mime_category,
                signature_category=signature_category,
                signature_name=signature_result.get(
                    "name"
                ),
                path=path,
            )
        )

        warnings: List[str] = []

        if (
            extension_category != FileCategory.UNKNOWN
            and mime_category != FileCategory.UNKNOWN
            and extension_category != mime_category
        ):
            warnings.append(
                "File extension and MIME type disagree."
            )

        if (
            signature_category != FileCategory.UNKNOWN
            and category != signature_category
            and signature_category
            not in {
                FileCategory.UNKNOWN,
                FileCategory.EXCEL,
                FileCategory.DOCX,
            }
        ):
            warnings.append(
                "File signature and resolved category disagree."
            )

        if (
            signature_result.get("name")
            == "ZIP/OOXML"
        ):
            package_category = (
                self._inspect_ooxml_package(
                    path
                )
            )

            if package_category != FileCategory.UNKNOWN:
                category = package_category
                confidence = 0.99

                if "package inspection" not in detected_by:
                    detected_by.append(
                        "package inspection"
                    )

        parser_name = PARSER_MAP.get(
            category,
            "unsupported",
        )

        processing_strategy = (
            PROCESSING_STRATEGY_MAP.get(
                category,
                "manual_review",
            )
        )

        is_binary = self._is_binary_file(
            path
        )

        risk_level = self._calculate_risk_level(
            category=category,
            path=path,
            warnings=warnings,
        )

        supported = (
            category in self.supported_categories
        )

        if not supported:
            warnings.append(
                f"Category '{category.value}' is not "
                "enabled for standard ingestion."
            )

        return FileClassificationResult(
            file_path=str(path),
            file_name=(
                original_filename
                or path.name
            ),
            extension=extension,
            mime_type=mime_type,
            file_category=category,
            parser_name=parser_name,
            processing_strategy=processing_strategy,
            confidence=round(
                max(0.0, min(confidence, 1.0)),
                4,
            ),
            risk_level=risk_level,
            detected_by=detected_by,
            detected_signature=(
                signature_result.get("name")
            ),
            is_supported=supported,
            is_binary=is_binary,
            file_size_bytes=path.stat().st_size,
            warnings=warnings,
            metadata={
                "suffix": extension,
                "stem": path.stem,
                "name": path.name,
                "declared_mime_type": (
                    declared_mime_type
                ),
                "detected_mime_type": mime_type,
            },
        )

    def classify_many(
        self,
        file_paths: Sequence[str | Path],
    ) -> List[FileClassificationResult]:
        """
        Classify multiple files.
        """

        results = []

        for file_path in file_paths:
            try:
                results.append(
                    self.classify(file_path)
                )
            except Exception:
                logger.exception(
                    "Failed to classify file: %s",
                    file_path,
                )

        return results

    def is_supported(
        self,
        file_path: str | Path,
    ) -> bool:
        """
        Return whether a file belongs to a supported category.
        """

        result = self.classify(file_path)

        return result.is_supported

    # ------------------------------------------------------------------
    # Category resolution
    # ------------------------------------------------------------------

    def _resolve_category(
        self,
        extension_category: FileCategory,
        mime_category: FileCategory,
        signature_category: FileCategory,
        signature_name: Optional[str],
        path: Path,
    ) -> tuple[
        FileCategory,
        float,
        List[str],
    ]:
        detected_by: List[str] = []

        if signature_category not in {
            FileCategory.UNKNOWN,
            FileCategory.EXCEL,
            FileCategory.DOCX,
        }:
            detected_by.append(
                "file signature"
            )

            return (
                signature_category,
                0.98,
                detected_by,
            )

        if signature_name == "OLE Compound File":
            if extension_category == FileCategory.EXCEL:
                detected_by.extend(
                    [
                        "file signature",
                        "extension",
                    ]
                )

                return (
                    FileCategory.EXCEL,
                    0.97,
                    detected_by,
                )

            if extension_category == FileCategory.DOC:
                detected_by.extend(
                    [
                        "file signature",
                        "extension",
                    ]
                )

                return (
                    FileCategory.DOC,
                    0.97,
                    detected_by,
                )

        if signature_name == "ZIP/OOXML":
            if extension_category in {
                FileCategory.EXCEL,
                FileCategory.DOCX,
            }:
                detected_by.extend(
                    [
                        "file signature",
                        "extension",
                    ]
                )

                return (
                    extension_category,
                    0.96,
                    detected_by,
                )

        if (
            mime_category != FileCategory.UNKNOWN
            and extension_category != FileCategory.UNKNOWN
            and mime_category == extension_category
        ):
            detected_by.extend(
                [
                    "MIME type",
                    "extension",
                ]
            )

            return (
                extension_category,
                0.95,
                detected_by,
            )

        if mime_category != FileCategory.UNKNOWN:
            detected_by.append(
                "MIME type"
            )

            return (
                mime_category,
                0.85,
                detected_by,
            )

        if extension_category != FileCategory.UNKNOWN:
            detected_by.append(
                "extension"
            )

            return (
                extension_category,
                0.80,
                detected_by,
            )

        heuristic_category = (
            self._classify_by_filename(
                path.name
            )
        )

        if heuristic_category != FileCategory.UNKNOWN:
            detected_by.append(
                "filename heuristic"
            )

            return (
                heuristic_category,
                0.55,
                detected_by,
            )

        return (
            FileCategory.UNKNOWN,
            0.0,
            detected_by,
        )

    # ------------------------------------------------------------------
    # Signature detection
    # ------------------------------------------------------------------

    def _detect_signature(
        self,
        path: Path,
    ) -> Dict[str, Any]:
        try:
            with path.open(
                "rb"
            ) as file:
                content = file.read(
                    self.max_signature_read_bytes
                )

        except OSError as exc:
            logger.warning(
                "Unable to read file signature for %s: %s",
                path,
                exc,
            )

            return {
                "name": None,
                "category": FileCategory.UNKNOWN,
            }

        for signature in FILE_SIGNATURES:
            start = signature.offset
            end = start + len(
                signature.signature
            )

            if content[start:end] == signature.signature:
                return {
                    "name": signature.name,
                    "category": signature.category,
                    "description": signature.description,
                }

        return {
            "name": None,
            "category": FileCategory.UNKNOWN,
        }

    def _inspect_ooxml_package(
        self,
        path: Path,
    ) -> FileCategory:
        """
        Inspects ZIP-based Office packages.

        XLSX packages contain:
            xl/workbook.xml

        DOCX packages contain:
            word/document.xml
        """

        try:
            import zipfile

            with zipfile.ZipFile(path) as archive:
                names = set(
                    archive.namelist()
                )

                if (
                    "xl/workbook.xml" in names
                    or any(
                        name.startswith(
                            "xl/"
                        )
                        for name in names
                    )
                ):
                    return FileCategory.EXCEL

                if (
                    "word/document.xml" in names
                    or any(
                        name.startswith(
                            "word/"
                        )
                        for name in names
                    )
                ):
                    return FileCategory.DOCX

        except Exception as exc:
            logger.debug(
                "Unable to inspect OOXML package %s: %s",
                path,
                exc,
            )

        return FileCategory.UNKNOWN

    # ------------------------------------------------------------------
    # Heuristics and utilities
    # ------------------------------------------------------------------

    def _classify_by_filename(
        self,
        filename: str,
    ) -> FileCategory:
        normalized = filename.lower()

        patterns = {
            FileCategory.FINANCIAL
            if False
            else FileCategory.CSV: (
                r"(csv|export|transactions?)"
            ),
        }

        for category, pattern in patterns.items():
            if re.search(
                pattern,
                normalized,
            ):
                return category

        return FileCategory.UNKNOWN

    def _normalize_mime_type(
        self,
        mime_type: Optional[str],
    ) -> Optional[str]:
        if not mime_type:
            return None

        return mime_type.split(
            ";",
            maxsplit=1,
        )[0].strip().lower()

    def _is_binary_file(
        self,
        path: Path,
    ) -> bool:
        try:
            with path.open(
                "rb"
            ) as file:
                sample = file.read(4096)

        except OSError:
            return True

        if not sample:
            return False

        if b"\x00" in sample:
            return True

        try:
            sample.decode(
                "utf-8"
            )
            return False

        except UnicodeDecodeError:
            return True

    def _calculate_risk_level(
        self,
        category: FileCategory,
        path: Path,
        warnings: Sequence[str],
    ) -> FileRiskLevel:
        if category in {
            FileCategory.ARCHIVE,
            FileCategory.AUDIO,
            FileCategory.VIDEO,
        }:
            return FileRiskLevel.MEDIUM

        if category == FileCategory.UNKNOWN:
            return FileRiskLevel.UNKNOWN

        if warnings:
            return FileRiskLevel.MEDIUM

        if path.stat().st_size > 100 * 1024 * 1024:
            return FileRiskLevel.MEDIUM

        return FileRiskLevel.LOW


# ============================================================================
# Singleton
# ============================================================================


file_classifier = FileClassifier()