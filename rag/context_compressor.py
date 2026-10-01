
"""
FinCo AI - Context Compressor

File:
    backend/app/rag/context_compressor.py

Purpose:
    Compress and organize retrieved RAG context before
    passing it to the LLM.

Features:
    - Remove duplicate chunks
    - Remove redundant text
    - Preserve financial evidence
    - Compress long chunks
    - Keep metadata and citations
    - Token-budget-aware context construction
    - Query-aware relevance filtering
    - Financial number preservation
    - Context formatting for LLMs

Design:
    Retrieved Chunks
        ↓
    Deduplication
        ↓
    Relevance Filtering
        ↓
    Text Compression
        ↓
    Token Budget Control
        ↓
    Context Builder
        ↓
    LLM
"""

from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ============================================================
# Constants
# ============================================================

DEFAULT_MAX_TOKENS = 4000
DEFAULT_MIN_SCORE = 0.0
DEFAULT_MAX_CHUNKS = 20

DEFAULT_MAX_CHUNK_CHARS = 1500

# Approximate English token ratio.
# This is only an estimate, not a tokenizer.
APPROX_CHARS_PER_TOKEN = 4

FINANCIAL_NUMBER_PATTERN = re.compile(
    r"""
    (?:
        ₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP
    )?
    \s?
    -?
    \d[\d,]*(?:\.\d+)?
    \s?
    %?
    """,
    re.IGNORECASE | re.VERBOSE,
)

FINANCIAL_TERM_PATTERN = re.compile(
    r"""
    revenue|profit|loss|expense|expenses|income|
    cash\s+flow|ebitda|ebit|assets?|liabilities?|
    debt|margin|growth|sales|turnover|budget|
    forecast|financial|transaction|invoice|
    liquidity|working\s+capital|profitability|
    """,
    re.IGNORECASE | re.VERBOSE,
)

SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+|\n+"
)


# ============================================================
# Data Models
# ============================================================


@dataclass
class CompressedChunk:
    """
    Represents a compressed retrieved chunk.
    """

    chunk_id: str
    text: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    original_text: Optional[str] = None

    original_length: int = 0
    compressed_length: int = 0

    relevance_score: Optional[float] = None

    compression_ratio: float = 1.0

    citation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert compressed chunk into a dictionary.
        """

        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "original_text": self.original_text,
            "original_length": self.original_length,
            "compressed_length": self.compressed_length,
            "relevance_score": self.relevance_score,
            "compression_ratio": self.compression_ratio,
            "citation_id": self.citation_id,
        }


@dataclass
class CompressionResult:
    """
    Result of context compression.
    """

    chunks: List[CompressedChunk] = field(
        default_factory=list
    )

    formatted_context: str = ""

    original_chunks: int = 0
    compressed_chunks: int = 0

    original_characters: int = 0
    compressed_characters: int = 0

    estimated_tokens: int = 0
    max_tokens: int = DEFAULT_MAX_TOKENS

    removed_duplicates: int = 0
    removed_low_score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert compression result into a dictionary.
        """

        return {
            "chunks": [
                chunk.to_dict()
                for chunk in self.chunks
            ],
            "formatted_context": self.formatted_context,
            "original_chunks": self.original_chunks,
            "compressed_chunks": self.compressed_chunks,
            "original_characters": self.original_characters,
            "compressed_characters": self.compressed_characters,
            "estimated_tokens": self.estimated_tokens,
            "max_tokens": self.max_tokens,
            "removed_duplicates": self.removed_duplicates,
            "removed_low_score": self.removed_low_score,
        }


# ============================================================
# Text Utilities
# ============================================================


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving paragraphs.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

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


def normalize_for_comparison(
    text: str,
) -> str:
    """
    Normalize text for duplicate comparison.
    """

    text = normalize_text(text)

    return text.lower()


def estimate_tokens(
    text: str,
) -> int:
    """
    Estimate token count.

    This is approximate. For exact token budgets,
    use the tokenizer of the selected LLM.
    """

    if not text:
        return 0

    return max(
        1,
        (len(text) + APPROX_CHARS_PER_TOKEN - 1)
        // APPROX_CHARS_PER_TOKEN,
    )


def truncate_text(
    text: str,
    max_chars: int,
) -> str:
    """
    Truncate text to a maximum character count.
    """

    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Prefer ending at a word boundary.
    last_space = truncated.rfind(" ")

    if last_space > max_chars * 0.7:
        truncated = truncated[:last_space]

    return truncated.rstrip() + "..."


def extract_financial_numbers(
    text: str,
) -> Set[str]:
    """
    Extract financial-looking numeric values.

    Useful for preserving:
        - Revenue values
        - Percentages
        - Profit amounts
        - Financial ratios
    """

    if not text:
        return set()

    return {
        match.group(0).strip()
        for match in FINANCIAL_NUMBER_PATTERN.finditer(text)
    }


def contains_financial_terms(
    text: str,
) -> bool:
    """
    Determine whether text contains financial terminology.
    """

    return bool(
        FINANCIAL_TERM_PATTERN.search(text or "")
    )


def split_sentences(
    text: str,
) -> List[str]:
    """
    Split text into sentence-like units.
    """

    if not text:
        return []

    sentences = SENTENCE_SPLIT_PATTERN.split(text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def sentence_similarity(
    first: str,
    second: str,
) -> float:
    """
    Lightweight lexical similarity.

    Uses word-set Jaccard similarity.

    This is not semantic embedding similarity.
    """

    first_words = set(
        re.findall(
            r"\b\w+\b",
            first.lower(),
        )
    )

    second_words = set(
        re.findall(
            r"\b\w+\b",
            second.lower(),
        )
    )

    if not first_words or not second_words:
        return 0.0

    intersection = first_words & second_words
    union = first_words | second_words

    return len(intersection) / len(union)


# ============================================================
# Context Compressor
# ============================================================


class ContextCompressor:
    """
    Compress retrieved chunks into LLM-ready context.

    The compressor uses deterministic text processing.

    It does not call an LLM, so it is fast and predictable.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        min_score: float = DEFAULT_MIN_SCORE,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        preserve_financial_numbers: bool = True,
        remove_redundancy: bool = True,
    ) -> None:

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be positive"
            )

        if min_score < 0:
            raise ValueError(
                "min_score cannot be negative"
            )

        if max_chunks <= 0:
            raise ValueError(
                "max_chunks must be positive"
            )

        if max_chunk_chars <= 0:
            raise ValueError(
                "max_chunk_chars must be positive"
            )

        self.max_tokens = max_tokens
        self.min_score = min_score
        self.max_chunks = max_chunks
        self.max_chunk_chars = max_chunk_chars

        self.preserve_financial_numbers = (
            preserve_financial_numbers
        )

        self.remove_redundancy = remove_redundancy

    # ========================================================
    # Main API
    # ========================================================

    def compress(
        self,
        retrieved_chunks: Sequence[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> CompressionResult:
        """
        Compress retrieved chunks.

        Args:
            retrieved_chunks:
                Retrieved chunks from hybrid retriever.

            query:
                Optional user query.

        Returns:
            CompressionResult
        """

        original_chunks = len(retrieved_chunks)

        original_characters = sum(
            len(str(chunk.get("text", "")))
            for chunk in retrieved_chunks
            if isinstance(chunk, dict)
        )

        # Step 1: Normalize and filter.
        normalized = self._prepare_chunks(
            retrieved_chunks
        )

        removed_low_score = len(retrieved_chunks) - len(
            normalized
        )

        # Step 2: Remove duplicates.
        if self.remove_redundancy:

            normalized, removed_duplicates = (
                self._remove_duplicates(normalized)
            )

        else:
            removed_duplicates = 0

        # Step 3: Rank chunks.
        ranked = self._rank_chunks(
            normalized,
            query=query,
        )

        # Step 4: Compress and fit budget.
        compressed_chunks = self._compress_to_budget(
            ranked
        )

        formatted_context = self.format_context(
            compressed_chunks
        )

        compressed_characters = len(
            formatted_context
        )

        return CompressionResult(
            chunks=compressed_chunks,
            formatted_context=formatted_context,
            original_chunks=original_chunks,
            compressed_chunks=len(compressed_chunks),
            original_characters=original_characters,
            compressed_characters=compressed_characters,
            estimated_tokens=estimate_tokens(
                formatted_context
            ),
            max_tokens=self.max_tokens,
            removed_duplicates=removed_duplicates,
            removed_low_score=removed_low_score,
        )

    # ========================================================
    # Prepare Chunks
    # ========================================================

    def _prepare_chunks(
        self,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Normalize chunk objects and filter by score.
        """

        prepared: List[Dict[str, Any]] = []

        for chunk in chunks:

            if not isinstance(chunk, dict):
                continue

            text = normalize_text(
                str(chunk.get("text", ""))
            )

            if not text:
                continue

            score = self._get_score(chunk)

            if score < self.min_score:
                continue

            prepared.append(
                {
                    **chunk,
                    "text": text,
                    "score": score,
                }
            )

        return prepared

    # ========================================================
    # Score
    # ========================================================

    @staticmethod
    def _get_score(
        chunk: Dict[str, Any],
    ) -> float:
        """
        Extract relevance score.
        """

        score = chunk.get("score")

        if score is None:
            score = chunk.get("relevance_score")

        if score is None:
            return 0.0

        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    # ========================================================
    # Deduplication
    # ========================================================

    def _remove_duplicates(
        self,
        chunks: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Remove exact duplicate chunks.
        """

        seen: Set[str] = set()
        unique: List[Dict[str, Any]] = []

        removed = 0

        for chunk in chunks:

            text = chunk.get("text", "")

            normalized = normalize_for_comparison(
                text
            )

            digest = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()

            if digest in seen:
                removed += 1
                continue

            seen.add(digest)
            unique.append(chunk)

        return unique, removed

    # ========================================================
    # Ranking
    # ========================================================

    def _rank_chunks(
        self,
        chunks: Sequence[Dict[str, Any]],
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank chunks using retrieval score and
        lightweight query overlap.
        """

        query_words = set(
            re.findall(
                r"\b\w+\b",
                (query or "").lower(),
            )
        )

        def ranking_key(
            chunk: Dict[str, Any],
        ) -> Tuple[float, float]:

            score = self._get_score(chunk)

            text_words = set(
                re.findall(
                    r"\b\w+\b",
                    chunk.get("text", "").lower(),
                )
            )

            overlap = 0.0

            if query_words and text_words:

                overlap = len(
                    query_words & text_words
                ) / len(query_words)

            combined_score = (
                score * 0.7
                + overlap * 0.3
            )

            return combined_score, score

        ranked = sorted(
            chunks,
            key=ranking_key,
            reverse=True,
        )

        return ranked[:self.max_chunks]

    # ========================================================
    # Compress to Budget
    # ========================================================

    def _compress_to_budget(
        self,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[CompressedChunk]:
        """
        Compress chunks and fit within token budget.
        """

        results: List[CompressedChunk] = []

        used_tokens = 0

        for chunk in chunks:

            text = chunk.get("text", "")

            metadata = chunk.get("metadata") or {}

            if not isinstance(metadata, dict):
                metadata = {}

            score = self._get_score(chunk)

            # Calculate available token budget.
            remaining_tokens = (
                self.max_tokens - used_tokens
            )

            if remaining_tokens <= 0:
                break

            remaining_chars = (
                remaining_tokens
                * APPROX_CHARS_PER_TOKEN
            )

            allowed_chars = min(
                self.max_chunk_chars,
                remaining_chars,
            )

            if allowed_chars <= 0:
                break

            compressed_text = self._compress_text(
                text=text,
                max_chars=allowed_chars,
            )

            if not compressed_text:
                continue

            compressed_tokens = estimate_tokens(
                compressed_text
            )

            if compressed_tokens > remaining_tokens:

                compressed_text = truncate_text(
                    compressed_text,
                    max_chars=(
                        remaining_tokens
                        * APPROX_CHARS_PER_TOKEN
                    ),
                )

                compressed_tokens = estimate_tokens(
                    compressed_text
                )

            if compressed_tokens <= 0:
                continue

            chunk_id = str(
                chunk.get(
                    "chunk_id",
                    f"compressed_{len(results)}",
                )
            )

            citation_id = chunk.get(
                "citation_id"
            )

            if citation_id is None:
                citation_id = metadata.get(
                    "citation_id"
                )

            original_length = len(text)
            compressed_length = len(compressed_text)

            results.append(
                CompressedChunk(
                    chunk_id=chunk_id,
                    text=compressed_text,
                    metadata=metadata,
                    original_text=text,
                    original_length=original_length,
                    compressed_length=compressed_length,
                    relevance_score=score,
                    compression_ratio=(
                        compressed_length / original_length
                        if original_length > 0
                        else 1.0
                    ),
                    citation_id=citation_id,
                )
            )

            used_tokens += compressed_tokens

        return results

    # ========================================================
    # Text Compression
    # ========================================================

    def _compress_text(
        self,
        text: str,
        max_chars: int,
    ) -> str:
        """
        Compress text while preserving useful sentences.
        """

        text = normalize_text(text)

        if len(text) <= max_chars:
            return text

        sentences = split_sentences(text)

        if not sentences:
            return truncate_text(
                text,
                max_chars=max_chars,
            )

        selected: List[str] = []
        current_length = 0

        for sentence in sentences:

            sentence_length = len(sentence)

            if (
                current_length + sentence_length + 1
                <= max_chars
            ):

                selected.append(sentence)

                current_length += (
                    sentence_length + 1
                )

                continue

            # Preserve financial sentences if possible.
            if (
                self.preserve_financial_numbers
                and contains_financial_terms(sentence)
            ):

                selected.append(sentence)

                current_length += (
                    sentence_length + 1
                )

                if current_length >= max_chars:
                    break

            else:
                break

        compressed = " ".join(selected)

        if not compressed:
            compressed = truncate_text(
                text,
                max_chars=max_chars,
            )

        # Preserve financial numbers if truncation removed them.
        if self.preserve_financial_numbers:

            original_numbers = extract_financial_numbers(
                text
            )

            compressed_numbers = extract_financial_numbers(
                compressed
            )

            missing_numbers = (
                original_numbers - compressed_numbers
            )

            if missing_numbers:

                # Append missing numeric evidence
                # only when there is room.
                evidence = (
                    "\nFinancial values: "
                    + ", ".join(
                        sorted(missing_numbers)
                    )
                )

                if (
                    len(compressed) + len(evidence)
                    <= max_chars
                ):

                    compressed += evidence

        return compressed.strip()

    # ========================================================
    # Format Context
    # ========================================================

    def format_context(
        self,
        chunks: Sequence[CompressedChunk],
    ) -> str:
        """
        Format compressed chunks for LLM context.

        Every chunk receives a source marker.
        """

        if not chunks:
            return ""

        sections: List[str] = []

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            source_name = (
                chunk.metadata.get("document_name")
                or chunk.metadata.get("source")
                or chunk.metadata.get("document_id")
                or "Unknown source"
            )

            page_number = chunk.metadata.get(
                "page_number"
            )

            section_name = chunk.metadata.get(
                "section"
            )

            source_parts = [str(source_name)]

            if page_number is not None:
                source_parts.append(
                    f"Page {page_number}"
                )

            if section_name:
                source_parts.append(
                    f"Section: {section_name}"
                )

            source_label = ", ".join(
                source_parts
            )

            sections.append(
                f"[Context {index}]\n"
                f"Source: {source_label}\n"
                f"Chunk ID: {chunk.chunk_id}\n"
                f"Evidence:\n"
                f"{chunk.text}"
            )

        return "\n\n".join(sections)

    # ========================================================
    # Context Statistics
    # ========================================================

    def get_statistics(
        self,
        result: CompressionResult,
    ) -> Dict[str, Any]:
        """
        Return compression statistics.
        """

        compression_ratio = 1.0

        if result.original_characters > 0:

            compression_ratio = (
                result.compressed_characters
                / result.original_characters
            )

        return {
            "original_chunks": result.original_chunks,
            "compressed_chunks": result.compressed_chunks,
            "original_characters": result.original_characters,
            "compressed_characters": result.compressed_characters,
            "estimated_tokens": result.estimated_tokens,
            "max_tokens": result.max_tokens,
            "compression_ratio": round(
                compression_ratio,
                4,
            ),
            "removed_duplicates": result.removed_duplicates,
            "removed_low_score": result.removed_low_score,
        }


# ============================================================
# Convenience Functions
# ============================================================


def compress_context(
    retrieved_chunks: Sequence[Dict[str, Any]],
    query: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> CompressionResult:
    """
    Compress retrieved context using default settings.
    """

    compressor = ContextCompressor(
        max_tokens=max_tokens
    )

    return compressor.compress(
        retrieved_chunks=retrieved_chunks,
        query=query,
    )


def format_compressed_context(
    retrieved_chunks: Sequence[Dict[str, Any]],
    query: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Return formatted compressed context directly.
    """

    result = compress_context(
        retrieved_chunks=retrieved_chunks,
        query=query,
        max_tokens=max_tokens,
    )

    return result.formatted_context


# ============================================================
# Example Usage
# ============================================================


if __name__ == "__main__":

    retrieved_chunks = [
        {
            "chunk_id": "chunk_001",
            "text": (
                "The company generated revenue of "
                "₹10 crore during FY2025. "
                "Revenue increased by 15 percent "
                "compared with FY2024."
            ),
            "metadata": {
                "document_id": "doc_001",
                "document_name": "Annual Report 2025",
                "page_number": 12,
                "section": "Revenue",
                "source": "annual_report.pdf",
                "company_id": "company_001",
            },
            "score": 0.95,
        },
        {
            "chunk_id": "chunk_002",
            "text": (
                "Operating expenses increased by "
                "12 percent. Higher expenses reduced "
                "net profit during FY2025."
            ),
            "metadata": {
                "document_id": "doc_001",
                "document_name": "Annual Report 2025",
                "page_number": 15,
                "section": "Expenses",
                "source": "annual_report.pdf",
                "company_id": "company_001",
            },
            "score": 0.89,
        },
        {
            "chunk_id": "chunk_003",
            "text": (
                "The company generated revenue of "
                "₹10 crore during FY2025. "
                "Revenue increased by 15 percent "
                "compared with FY2024."
            ),
            "metadata": {
                "document_id": "doc_001",
                "document_name": "Annual Report 2025",
                "page_number": 12,
                "section": "Revenue",
                "source": "annual_report.pdf",
                "company_id": "company_001",
            },
            "score": 0.80,
        },
    ]

    compressor = ContextCompressor(
        max_tokens=1000,
        max_chunks=10,
        max_chunk_chars=1000,
    )

    result = compressor.compress(
        retrieved_chunks=retrieved_chunks,
        query="What happened to revenue and profit?",
    )

    print("=" * 60)
    print("Compressed Context")
    print("=" * 60)

    print(result.formatted_context)

    print("\nStatistics:")
    print(compressor.get_statistics(result))