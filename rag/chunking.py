
"""
FinCo AI - RAG Chunking Module

File:
    backend/app/rag/chunking.py

Purpose:
    Split documents into meaningful chunks for retrieval,
    embedding generation, and LLM context construction.

Design:
    - Recursive paragraph/sentence chunking
    - Configurable chunk size and overlap
    - Metadata preservation
    - Stable chunk identifiers
    - Financial-document friendly processing
"""

from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ============================================================
# Constants
# ============================================================

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
]


# ============================================================
# Data Models
# ============================================================


@dataclass
class ChunkMetadata:
    """
    Metadata associated with a document chunk.
    """

    document_id: Optional[str] = None
    document_name: Optional[str] = None
    document_type: Optional[str] = None

    page_number: Optional[int] = None
    section: Optional[str] = None

    source: Optional[str] = None
    company_id: Optional[str] = None

    chunk_index: int = 0
    total_chunks: int = 0

    start_char: int = 0
    end_char: int = 0

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to a dictionary.
        """

        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_type": self.document_type,
            "page_number": self.page_number,
            "section": self.section,
            "source": self.source,
            "company_id": self.company_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            **self.extra,
        }


@dataclass
class DocumentChunk:
    """
    Represents a single chunk of a document.
    """

    chunk_id: str
    text: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to a dictionary.
        """

        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
        }


# ============================================================
# Text Cleaning
# ============================================================


def normalize_text(text: str) -> str:
    """
    Normalize extracted document text.

    Operations:
        - Convert line endings
        - Remove null bytes
        - Normalize whitespace
        - Preserve paragraph boundaries
        - Remove excessive blank lines
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\x00", "")

    # Normalize spaces and tabs without destroying newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)

    # Limit excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_chunk_text(text: str) -> str:
    """
    Clean individual chunk text.
    """

    text = text.strip()

    # Remove repeated whitespace while preserving newlines.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# Chunk ID Generation
# ============================================================


def generate_chunk_id(
    document_id: Optional[str],
    chunk_index: int,
    text: str,
) -> str:
    """
    Generate a deterministic chunk identifier.

    Same document ID, index, and text produce the same ID.
    """

    raw_value = (
        f"{document_id or 'unknown'}:"
        f"{chunk_index}:"
        f"{text}"
    )

    digest = hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:16]

    return f"chunk_{digest}"


# ============================================================
# Recursive Text Splitter
# ============================================================


class RecursiveTextSplitter:
    """
    Recursively split text using a hierarchy of separators.

    The splitter attempts to preserve:
        1. Paragraphs
        2. Lines
        3. Sentences
        4. Clauses
        5. Words
        6. Characters
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separators: Optional[Sequence[str]] = None,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be greater than zero"
            )

        if chunk_overlap < 0:
            raise ValueError(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.separators = list(
            separators or DEFAULT_SEPARATORS
        )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Returns:
            List[str]
        """

        text = normalize_text(text)

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = self._split_recursive(
            text=text,
            separators=self.separators,
        )

        return self._merge_chunks(chunks)

    # --------------------------------------------------------
    # Recursive Splitting
    # --------------------------------------------------------

    def _split_recursive(
        self,
        text: str,
        separators: Sequence[str],
    ) -> List[str]:
        """
        Recursively split text using separators.
        """

        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._split_by_characters(text)

        separator = separators[0]
        remaining = separators[1:]

        if separator and separator not in text:
            return self._split_recursive(
                text,
                remaining,
            )

        if separator:
            pieces = text.split(separator)
        else:
            pieces = list(text)

        pieces = [
            piece.strip()
            for piece in pieces
            if piece.strip()
        ]

        if len(pieces) <= 1:
            return self._split_recursive(
                text,
                remaining,
            )

        splits: List[str] = []

        for piece in pieces:

            if len(piece) <= self.chunk_size:
                splits.append(piece)
            else:
                splits.extend(
                    self._split_recursive(
                        piece,
                        remaining,
                    )
                )

        return splits

    # --------------------------------------------------------
    # Character Fallback
    # --------------------------------------------------------

    def _split_by_characters(
        self,
        text: str,
    ) -> List[str]:
        """
        Hard split text when no separator works.
        """

        return [
            text[i:i + self.chunk_size]
            for i in range(
                0,
                len(text),
                self.chunk_size,
            )
        ]

    # --------------------------------------------------------
    # Chunk Merging
    # --------------------------------------------------------

    def _merge_chunks(
        self,
        splits: List[str],
    ) -> List[str]:
        """
        Merge smaller splits into chunks.

        Adds overlap between neighboring chunks.
        """

        chunks: List[str] = []
        current = ""

        for split in splits:

            split = split.strip()

            if not split:
                continue

            if not current:
                current = split
                continue

            candidate = current + "\n" + split

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current)

                overlap_text = self._get_overlap(
                    current
                )

                current = (
                    overlap_text + "\n" + split
                    if overlap_text
                    else split
                )

                # Safety fallback if overlap makes chunk too long.
                if len(current) > self.chunk_size:
                    chunks.extend(
                        self._split_by_characters(current)
                    )
                    current = ""

        if current:
            chunks.append(current)

        return [
            clean_chunk_text(chunk)
            for chunk in chunks
            if clean_chunk_text(chunk)
        ]

    # --------------------------------------------------------
    # Overlap
    # --------------------------------------------------------

    def _get_overlap(self, text: str) -> str:
        """
        Return the trailing overlap portion.
        """

        if self.chunk_overlap <= 0:
            return ""

        if len(text) <= self.chunk_overlap:
            return text

        overlap = text[-self.chunk_overlap:]

        # Prefer starting at a word boundary.
        first_space = overlap.find(" ")

        if first_space != -1:
            overlap = overlap[first_space + 1:]

        return overlap.strip()


# ============================================================
# Financial Document Chunker
# ============================================================


class FinancialChunker:
    """
    High-level chunking service for financial documents.

    Handles:
        - Annual reports
        - Income statements
        - Balance sheets
        - Cash-flow statements
        - Transaction documents
        - Invoices
        - General financial text
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:

        self.splitter = RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # --------------------------------------------------------
    # Chunk Document
    # --------------------------------------------------------

    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a complete document.

        Args:
            text:
                Extracted document text.

            metadata:
                Document-level metadata.

        Returns:
            List[DocumentChunk]
        """

        metadata = metadata or {}

        document_id = metadata.get("document_id")

        chunks = self.splitter.split_text(text)

        total_chunks = len(chunks)

        results: List[DocumentChunk] = []

        current_position = 0

        for index, chunk_text in enumerate(chunks):

            start_position = text.find(
                chunk_text,
                current_position,
            )

            if start_position == -1:
                start_position = current_position

            end_position = (
                start_position + len(chunk_text)
            )

            chunk_metadata = ChunkMetadata(
                document_id=document_id,
                document_name=metadata.get(
                    "document_name"
                ),
                document_type=metadata.get(
                    "document_type"
                ),
                page_number=metadata.get(
                    "page_number"
                ),
                section=metadata.get("section"),
                source=metadata.get("source"),
                company_id=metadata.get("company_id"),
                chunk_index=index,
                total_chunks=total_chunks,
                start_char=start_position,
                end_char=end_position,
                extra={
                    key: value
                    for key, value in metadata.items()
                    if key not in {
                        "document_id",
                        "document_name",
                        "document_type",
                        "page_number",
                        "section",
                        "source",
                        "company_id",
                    }
                },
            )

            chunk_id = generate_chunk_id(
                document_id=document_id,
                chunk_index=index,
                text=chunk_text,
            )

            results.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_metadata.to_dict(),
                )
            )

            current_position = end_position

        return results


# ============================================================
# Convenience Functions
# ============================================================


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Simple text chunking function.
    """

    splitter = RecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_text(text)


def chunk_financial_document(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    """
    Convenience function for financial documents.
    """

    chunker = FinancialChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunker.chunk_document(
        text=text,
        metadata=metadata,
    )


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    sample_text = """
    Annual Financial Report 2025

    Revenue:
    The company generated revenue of ₹10 crore
    during the financial year 2025.

    Expenses:
    Operating expenses increased by 12 percent
    compared with the previous year.

    Profit:
    Net profit decreased due to higher operating costs.
    """

    chunks = chunk_financial_document(
        text=sample_text,
        metadata={
            "document_id": "annual_report_2025",
            "document_name": "Annual Report 2025",
            "document_type": "annual_report",
            "company_id": "company_001",
            "source": "annual_report.pdf",
        },
        chunk_size=300,
        chunk_overlap=50,
    )

    for chunk in chunks:
        print("=" * 60)
        print("Chunk ID:", chunk.chunk_id)
        print("Metadata:", chunk.metadata)
        print("Text:", chunk.text)