"""
FinCo AI - DOCX Parser

Responsible for:
- Parsing Microsoft Word DOCX files
- Extracting paragraphs and headings
- Extracting tables
- Extracting headers and footers
- Extracting document metadata
- Extracting hyperlinks
- Preparing structured content for RAG
- Preserving document order as much as possible

Supported source types:
    - File path
    - pathlib.Path
    - bytes
    - bytearray
    - BinaryIO

Example:

    from app.ingestion.docx_parser import docx_parser

    result = docx_parser.parse(
        "data/raw/financial_report.docx"
    )

    print(result.text)
    print(result.metadata)
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class DOCXParserError(Exception):
    """Base exception for DOCX parsing errors."""


class DOCXFileNotFoundError(DOCXParserError):
    """Raised when a DOCX file cannot be found."""


class DOCXExtractionError(DOCXParserError):
    """Raised when DOCX extraction fails."""


class InvalidDOCXFileError(DOCXParserError):
    """Raised when the source is not a valid DOCX document."""


class DOCXSizeLimitError(DOCXParserError):
    """Raised when the DOCX file exceeds the size limit."""


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class DOCXParserConfig:
    """
    Configuration for DOCX parsing.
    """

    max_file_size_mb: int = 50

    include_tables: bool = True

    include_headers: bool = True

    include_footers: bool = True

    include_hyperlinks: bool = True

    include_empty_paragraphs: bool = False

    include_style_names: bool = True

    normalize_whitespace: bool = True

    preserve_heading_structure: bool = True

    include_table_headers: bool = True

    max_table_rows: Optional[int] = 10_000

    max_table_columns: Optional[int] = 200


# ============================================================================
# Result Models
# ============================================================================


@dataclass
class DOCXParagraph:
    """
    Represents one paragraph from a DOCX document.
    """

    index: int

    text: str

    style: Optional[str] = None

    is_heading: bool = False

    heading_level: Optional[int] = None

    is_bold: bool = False

    is_italic: bool = False

    hyperlinks: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "style": self.style,
            "is_heading": self.is_heading,
            "heading_level": self.heading_level,
            "is_bold": self.is_bold,
            "is_italic": self.is_italic,
            "hyperlinks": self.hyperlinks,
            "metadata": self.metadata,
        }


@dataclass
class DOCXTable:
    """
    Represents one table from a DOCX document.
    """

    index: int

    headers: List[str] = field(
        default_factory=list
    )

    rows: List[List[str]] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "headers": self.headers,
            "rows": self.rows,
            "metadata": self.metadata,
        }

    def to_text(self) -> str:
        lines: List[str] = []

        if self.headers:
            lines.append(
                " | ".join(self.headers)
            )

            lines.append(
                " | ".join(
                    "---"
                    for _ in self.headers
                )
            )

        for row in self.rows:
            lines.append(
                " | ".join(row)
            )

        return "\n".join(lines)


@dataclass
class DOCXMetadata:
    """
    Metadata extracted from DOCX core properties.
    """

    source_name: Optional[str] = None

    source_path: Optional[str] = None

    file_size_bytes: int = 0

    title: Optional[str] = None

    subject: Optional[str] = None

    author: Optional[str] = None

    keywords: Optional[str] = None

    comments: Optional[str] = None

    category: Optional[str] = None

    created: Optional[str] = None

    modified: Optional[str] = None

    last_modified_by: Optional[str] = None

    revision: Optional[int] = None

    language: Optional[str] = None

    extra: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_path": self.source_path,
            "file_size_bytes": self.file_size_bytes,
            "title": self.title,
            "subject": self.subject,
            "author": self.author,
            "keywords": self.keywords,
            "comments": self.comments,
            "category": self.category,
            "created": self.created,
            "modified": self.modified,
            "last_modified_by": self.last_modified_by,
            "revision": self.revision,
            "language": self.language,
            "extra": self.extra,
        }


@dataclass
class DOCXParseResult:
    """
    Final DOCX parsing result.
    """

    text: str

    paragraphs: List[DOCXParagraph]

    tables: List[DOCXTable]

    headers: List[str]

    footers: List[str]

    metadata: DOCXMetadata

    success: bool = True

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)

    @property
    def table_count(self) -> int:
        return len(self.tables)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "paragraphs": [
                paragraph.to_dict()
                for paragraph in self.paragraphs
            ],
            "tables": [
                table.to_dict()
                for table in self.tables
            ],
            "headers": self.headers,
            "footers": self.footers,
            "metadata": self.metadata.to_dict(),
            "success": self.success,
            "warnings": self.warnings,
            "errors": self.errors,
        }


# ============================================================================
# DOCX Parser
# ============================================================================


class DOCXParser:
    """
    Dedicated DOCX parser for FinCo AI.

    This parser is useful for:
    - Financial reports
    - Loan documents
    - Bank statements saved as Word files
    - Investment reports
    - Policy documents
    - Compliance documents
    - Internal finance documentation
    """

    def __init__(
        self,
        config: DOCXParserConfig = DOCXParserConfig(),
    ) -> None:
        self.config = config

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def parse(
        self,
        source: str | Path | bytes | bytearray | BinaryIO,
        source_name: Optional[str] = None,
    ) -> DOCXParseResult:
        """
        Parse a DOCX source.
        """

        resolved_name = (
            source_name
            or self._get_source_name(source)
            or "unknown_document.docx"
        )

        raw_bytes = self._read_source(source)

        self._validate_file_size(
            raw_bytes
        )

        self._validate_extension(
            source=source,
            source_name=resolved_name,
        )

        document = self._load_document(
            raw_bytes
        )

        metadata = self.extract_metadata(
            document=document,
            source_name=resolved_name,
            source=source,
            file_size_bytes=len(raw_bytes),
        )

        paragraphs = self.extract_paragraphs(
            document
        )

        tables = []

        if self.config.include_tables:
            tables = self.extract_tables(
                document
            )

        headers = []

        if self.config.include_headers:
            headers = self.extract_headers(
                document
            )

        footers = []

        if self.config.include_footers:
            footers = self.extract_footers(
                document
            )

        text = self.build_document_text(
            paragraphs=paragraphs,
            tables=tables,
            headers=headers,
            footers=footers,
        )

        warnings: List[str] = []

        if not text.strip():
            warnings.append(
                "DOCX document contains no extractable text."
            )

        if not paragraphs and not tables:
            warnings.append(
                "No paragraphs or tables were found."
            )

        return DOCXParseResult(
            text=text,
            paragraphs=paragraphs,
            tables=tables,
            headers=headers,
            footers=footers,
            metadata=metadata,
            success=bool(text.strip()),
            warnings=warnings,
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
                raise DOCXFileNotFoundError(
                    f"DOCX file not found: {source}"
                )

            if not source.is_file():
                raise DOCXFileNotFoundError(
                    f"DOCX source is not a file: {source}"
                )

            return source.read_bytes()

        if isinstance(source, str):
            path = Path(source)

            if not path.exists():
                raise DOCXFileNotFoundError(
                    f"DOCX file not found: {source}"
                )

            if not path.is_file():
                raise DOCXFileNotFoundError(
                    f"DOCX source is not a file: {source}"
                )

            return path.read_bytes()

        if isinstance(source, bytes):
            return source

        if isinstance(source, bytearray):
            return bytes(source)

        if hasattr(source, "read"):
            content = source.read()

            if isinstance(content, str):
                return content.encode("utf-8")

            return content

        raise TypeError(
            "Unsupported DOCX source type. "
            "Use a path, bytes, bytearray, or binary file object."
        )

    def _validate_file_size(
        self,
        raw_bytes: bytes,
    ) -> None:
        max_bytes = (
            self.config.max_file_size_mb
            * 1024
            * 1024
        )

        if len(raw_bytes) > max_bytes:
            raise DOCXSizeLimitError(
                f"DOCX file exceeds the maximum size of "
                f"{self.config.max_file_size_mb} MB."
            )

    def _validate_extension(
        self,
        source: Any,
        source_name: str,
    ) -> None:
        if isinstance(source, (str, Path)):
            extension = Path(source).suffix.lower()
        else:
            extension = Path(source_name).suffix.lower()

        if extension != ".docx":
            raise InvalidDOCXFileError(
                f"Expected a .docx file, received: {extension or 'unknown'}"
            )

    def _load_document(
        self,
        raw_bytes: bytes,
    ) -> Any:
        try:
            from docx import Document
        except ImportError as exc:
            raise DOCXExtractionError(
                "python-docx is required. "
                "Install it with: pip install python-docx"
            ) from exc

        try:
            return Document(
                io.BytesIO(raw_bytes)
            )

        except Exception as exc:
            raise InvalidDOCXFileError(
                f"Unable to open DOCX document: {exc}"
            ) from exc

    # ---------------------------------------------------------------------
    # Metadata extraction
    # ---------------------------------------------------------------------

    def extract_metadata(
        self,
        document: Any,
        source_name: str,
        source: Any,
        file_size_bytes: int,
    ) -> DOCXMetadata:
        properties = document.core_properties

        return DOCXMetadata(
            source_name=source_name,
            source_path=(
                str(source)
                if isinstance(source, (str, Path))
                else None
            ),
            file_size_bytes=file_size_bytes,
            title=properties.title,
            subject=properties.subject,
            author=properties.author,
            keywords=properties.keywords,
            comments=properties.comments,
            category=properties.category,
            created=(
                str(properties.created)
                if properties.created
                else None
            ),
            modified=(
                str(properties.modified)
                if properties.modified
                else None
            ),
            last_modified_by=properties.last_modified_by,
            revision=properties.revision,
            language=properties.language,
        )

    # ---------------------------------------------------------------------
    # Paragraph extraction
    # ---------------------------------------------------------------------

    def extract_paragraphs(
        self,
        document: Any,
    ) -> List[DOCXParagraph]:
        paragraphs: List[DOCXParagraph] = []

        for index, paragraph in enumerate(
            document.paragraphs
        ):
            text = self.clean_text(
                paragraph.text
            )

            if (
                not text
                and not self.config.include_empty_paragraphs
            ):
                continue

            style_name = None

            if paragraph.style is not None:
                style_name = paragraph.style.name

            heading_level = self.detect_heading_level(
                style_name
            )

            hyperlinks = []

            if self.config.include_hyperlinks:
                hyperlinks = self.extract_hyperlinks(
                    paragraph
                )

            paragraphs.append(
                DOCXParagraph(
                    index=index,
                    text=text,
                    style=style_name
                    if self.config.include_style_names
                    else None,
                    is_heading=heading_level is not None,
                    heading_level=heading_level,
                    is_bold=self.paragraph_is_bold(
                        paragraph
                    ),
                    is_italic=self.paragraph_is_italic(
                        paragraph
                    ),
                    hyperlinks=hyperlinks,
                )
            )

        return paragraphs

    def detect_heading_level(
        self,
        style_name: Optional[str],
    ) -> Optional[int]:
        if not style_name:
            return None

        normalized = style_name.strip().lower()

        match = re.search(
            r"heading\s*([1-9])",
            normalized,
        )

        if match:
            return int(match.group(1))

        if normalized in {
            "title",
            "subtitle",
        }:
            return 1

        return None

    def paragraph_is_bold(
        self,
        paragraph: Any,
    ) -> bool:
        runs = paragraph.runs

        if not runs:
            return False

        return any(
            run.bold is True
            for run in runs
        )

    def paragraph_is_italic(
        self,
        paragraph: Any,
    ) -> bool:
        runs = paragraph.runs

        if not runs:
            return False

        return any(
            run.italic is True
            for run in runs
        )

    def extract_hyperlinks(
        self,
        paragraph: Any,
    ) -> List[str]:
        """
        Extracts hyperlinks from a paragraph XML tree.
        """

        hyperlinks: List[str] = []

        try:
            relationships = (
                paragraph.part.rels
            )

            for relationship in relationships.values():
                if relationship.reltype.endswith(
                    "/hyperlink"
                ):
                    target = relationship.target_ref

                    if target:
                        hyperlinks.append(
                            str(target)
                        )

        except Exception as exc:
            logger.warning(
                "Unable to extract hyperlinks: %s",
                exc,
            )

        return list(
            dict.fromkeys(hyperlinks)
        )

    # ---------------------------------------------------------------------
    # Table extraction
    # ---------------------------------------------------------------------

    def extract_tables(
        self,
        document: Any,
    ) -> List[DOCXTable]:
        tables: List[DOCXTable] = []

        for table_index, table in enumerate(
            document.tables
        ):
            rows = table.rows

            if (
                self.config.max_table_rows is not None
                and len(rows) > self.config.max_table_rows
            ):
                raise DOCXExtractionError(
                    f"Table {table_index} exceeds the maximum "
                    f"row limit of {self.config.max_table_rows}."
                )

            table_rows: List[List[str]] = []

            for row in rows:
                cells = []

                for cell in row.cells:
                    cell_text = self.extract_cell_text(
                        cell
                    )

                    cells.append(cell_text)

                table_rows.append(cells)

            headers: List[str] = []

            data_rows = table_rows

            if (
                self.config.include_table_headers
                and table_rows
            ):
                headers = table_rows[0]
                data_rows = table_rows[1:]

            column_count = max(
                (
                    len(row)
                    for row in table_rows
                ),
                default=0,
            )

            if (
                self.config.max_table_columns is not None
                and column_count > self.config.max_table_columns
            ):
                raise DOCXExtractionError(
                    f"Table {table_index} exceeds the maximum "
                    f"column limit of {self.config.max_table_columns}."
                )

            tables.append(
                DOCXTable(
                    index=table_index,
                    headers=headers,
                    rows=data_rows,
                    metadata={
                        "row_count": len(table_rows),
                        "column_count": column_count,
                    },
                )
            )

        return tables

    def extract_cell_text(
        self,
        cell: Any,
    ) -> str:
        """
        Extracts all paragraph text from a table cell.
        """

        parts: List[str] = []

        for paragraph in cell.paragraphs:
            text = self.clean_text(
                paragraph.text
            )

            if text:
                parts.append(text)

        return " ".join(parts)

    # ---------------------------------------------------------------------
    # Header and footer extraction
    # ---------------------------------------------------------------------

    def extract_headers(
        self,
        document: Any,
    ) -> List[str]:
        headers: List[str] = []

        for section in document.sections:
            header = section.header

            for paragraph in header.paragraphs:
                text = self.clean_text(
                    paragraph.text
                )

                if text:
                    headers.append(text)

        return self.unique_strings(
            headers
        )

    def extract_footers(
        self,
        document: Any,
    ) -> List[str]:
        footers: List[str] = []

        for section in document.sections:
            footer = section.footer

            for paragraph in footer.paragraphs:
                text = self.clean_text(
                    paragraph.text
                )

                if text:
                    footers.append(text)

        return self.unique_strings(
            footers
        )

    # ---------------------------------------------------------------------
    # Text building and cleaning
    # ---------------------------------------------------------------------

    def build_document_text(
        self,
        paragraphs: List[DOCXParagraph],
        tables: List[DOCXTable],
        headers: List[str],
        footers: List[str],
    ) -> str:
        sections: List[str] = []

        if headers:
            sections.append(
                "DOCUMENT HEADERS\n"
                + "\n".join(headers)
            )

        for paragraph in paragraphs:
            if paragraph.is_heading:
                prefix = (
                    "#" * (
                        paragraph.heading_level or 1
                    )
                )

                sections.append(
                    f"{prefix} {paragraph.text}"
                )

            else:
                sections.append(
                    paragraph.text
                )

        for table in tables:
            sections.append(
                f"TABLE {table.index + 1}\n"
                f"{table.to_text()}"
            )

        if footers:
            sections.append(
                "DOCUMENT FOOTERS\n"
                + "\n".join(footers)
            )

        return self.clean_text(
            "\n\n".join(
                section
                for section in sections
                if section.strip()
            )
        )

    def clean_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace(
            "\x00",
            "",
        )

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        if self.config.normalize_whitespace:
            text = re.sub(
                r"[ \t]+",
                " ",
                text,
            )

            text = re.sub(
                r"\n{3,}",
                "\n\n",
                text,
            )

        return text.strip()

    def unique_strings(
        self,
        values: List[str],
    ) -> List[str]:
        seen = set()
        result = []

        for value in values:
            normalized = value.strip()

            if not normalized:
                continue

            if normalized.lower() in seen:
                continue

            seen.add(normalized.lower())
            result.append(normalized)

        return result


# ============================================================================
# Singleton
# ============================================================================


docx_parser = DOCXParser()