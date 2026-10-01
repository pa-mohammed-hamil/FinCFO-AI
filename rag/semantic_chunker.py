"""
FinCo AI - Semantic Chunker
===========================

Semantic document chunking for financial RAG pipelines.

Features
--------
- Sentence-aware chunking.
- Embedding-based semantic boundary detection.
- Token and character limits.
- Configurable semantic similarity threshold.
- Optional paragraph and heading preservation.
- Metadata propagation.
- Deterministic chunk IDs.
- Lightweight lexical fallback when embeddings are unavailable.
- Support for financial tables, figures, and important numeric content.
- No mandatory dependency on a specific embedding provider.

The semantic chunking process:

Document
    |
    v
Paragraph / sentence segmentation
    |
    v
Sentence embeddings
    |
    v
Adjacent sentence similarity
    |
    v
Semantic boundary detection
    |
    v
Chunk size validation
    |
    v
Overlap handling
    |
    v
Semantic chunks with metadata
"""

from __future__ import annotations

import hashlib
import logging
import math
import re

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class SemanticChunkerError(RuntimeError):
    """Base exception for semantic chunking failures."""


class SemanticChunkerConfigurationError(SemanticChunkerError):
    """Raised when semantic chunker configuration is invalid."""


class SemanticChunkerEmbeddingError(SemanticChunkerError):
    """Raised when sentence embedding generation fails."""


# ============================================================================
# Protocols
# ============================================================================


class EmbeddingProviderProtocol(Protocol):
    """
    Minimal embedding-provider protocol.

    Compatible with providers exposing:

        embed(texts)

    or:

        embed_documents(texts)
    """

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


# ============================================================================
# Utility Functions
# ============================================================================


def normalize_text(value: Any) -> str:
    """
    Normalize whitespace while preserving readable text.
    """
    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def stable_hash(*values: Any, length: int = 16) -> str:
    """
    Generate a deterministic hash from multiple values.
    """
    payload = "||".join(
        normalize_text(value)
        for value in values
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:length]


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Clamp a value to a numeric range.
    """
    return max(minimum, min(maximum, value))


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for lexical fallback similarity.
    """
    return re.findall(
        r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*",
        text.lower(),
    )


def token_set(text: str) -> set[str]:
    """
    Return unique lexical tokens.
    """
    return set(tokenize(text))


def lexical_similarity(left: str, right: str) -> float:
    """
    Calculate Jaccard similarity between two texts.
    """
    left_tokens = token_set(left)
    right_tokens = token_set(right)

    if not left_tokens or not right_tokens:
        return 0.0

    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)

    if union == 0:
        return 0.0

    return intersection / union


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """
    if not left or not right:
        return 0.0

    if len(left) != len(right):
        raise ValueError(
            "Vectors must have the same dimensionality."
        )

    dot_product = sum(
        float(a) * float(b)
        for a, b in zip(left, right)
    )

    left_norm = math.sqrt(
        sum(float(value) ** 2 for value in left)
    )

    right_norm = math.sqrt(
        sum(float(value) ** 2 for value in right)
    )

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count.

    This is intentionally dependency-free. For exact model-specific
    token counts, inject a tokenizer into the chunker.
    """
    if not text:
        return 0

    return max(
        1,
        math.ceil(len(text) / 4),
    )


def split_sentences(text: str) -> list[str]:
    """
    Split text into readable sentences.

    The implementation protects common financial abbreviations and
    decimal numbers from accidental splitting.
    """
    text = text.strip()

    if not text:
        return []

    protected = text

    replacements = {
        "e.g.": "e<dot>g<dot>",
        "i.e.": "i<dot>e<dot>",
        "U.S.": "U<dot>S<dot>",
        "FY.": "FY<dot>",
        "Mr.": "Mr<dot>",
        "Mrs.": "Mrs<dot>",
        "Dr.": "Dr<dot>",
        "Inc.": "Inc<dot>",
        "Ltd.": "Ltd<dot>",
    }

    for source, replacement in replacements.items():
        protected = protected.replace(source, replacement)

    protected = re.sub(
        r"(?<=\d)\.(?=\d)",
        "<decimal>",
        protected,
    )

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9₹$€£(])",
        protected,
    )

    sentences: list[str] = []

    for part in parts:
        restored = (
            part.replace("<dot>", ".")
            .replace("<decimal>", ".")
        )

        restored = normalize_text(restored)

        if restored:
            sentences.append(restored)

    return sentences


def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs using blank lines.
    """
    return [
        normalize_text(paragraph)
        for paragraph in re.split(r"\n\s*\n+", text)
        if normalize_text(paragraph)
    ]


def is_heading(text: str) -> bool:
    """
    Heuristically detect a heading.
    """
    normalized = normalize_text(text)

    if not normalized:
        return False

    if len(normalized) > 140:
        return False

    if normalized.endswith((".", "?", "!")):
        return False

    if re.match(
        r"^(chapter|section|part|appendix|schedule|note)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True

    words = normalized.split()

    if len(words) <= 12 and normalized == normalized.title():
        return True

    if re.match(r"^\d+(?:\.\d+)*[\s.)-]+", normalized):
        return True

    return False


def contains_financial_signal(text: str) -> bool:
    """
    Detect financial content that should preferably remain intact.
    """
    financial_patterns = [
        r"\bFY\d{2,4}\b",
        r"\bQ[1-4]\b",
        r"\bEBITDA\b",
        r"\bEBIT\b",
        r"\bROI\b",
        r"\bROE\b",
        r"\bROA\b",
        r"\bCAGR\b",
        r"\bEPS\b",
        r"\bP&L\b",
        r"\bP\/L\b",
        r"\bbalance sheet\b",
        r"\bincome statement\b",
        r"\bcash flow\b",
        r"\brevenue\b",
        r"\bprofit\b",
        r"\bloss\b",
        r"\bmargin\b",
        r"\bexpense\b",
        r"\basset\b",
        r"\bliabilit(?:y|ies)\b",
        r"\bequity\b",
        r"\bdebt\b",
        r"\binterest rate\b",
        r"\b\d+(?:\.\d+)?\s*%",
        r"[$€£₹]\s?\d",
        r"\b\d+(?:,\d{3})+(?:\.\d+)?\b",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in financial_patterns
    )


def extract_numeric_tokens(text: str) -> set[str]:
    """
    Extract numeric and financial value tokens.
    """
    return set(
        re.findall(
            r"""
            [$€£₹]?\s?\d+(?:,\d{3})*(?:\.\d+)?%?
            |FY\d{2,4}
            |Q[1-4]
            """,
            text,
            flags=re.IGNORECASE | re.VERBOSE,
        )
    )


# ============================================================================
# Data Models
# ============================================================================


@dataclass(slots=True)
class SemanticChunk:
    """
    A semantically coherent text chunk.
    """

    chunk_id: str
    text: str

    document_id: str | None = None
    source: str | None = None

    index: int = 0
    start_sentence: int = 0
    end_sentence: int = 0

    token_count: int = 0
    character_count: int = 0

    semantic_score: float | None = None
    boundary_score: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the chunk to a serializable dictionary.
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_id": self.document_id,
            "source": self.source,
            "index": self.index,
            "start_sentence": self.start_sentence,
            "end_sentence": self.end_sentence,
            "token_count": self.token_count,
            "character_count": self.character_count,
            "semantic_score": self.semantic_score,
            "boundary_score": self.boundary_score,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SentenceUnit:
    """
    Internal sentence representation.
    """

    text: str
    index: int

    paragraph_index: int = 0
    is_heading: bool = False
    contains_financial_signal: bool = False

    token_count: int = 0
    embedding: Sequence[float] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SemanticBoundary:
    """
    Represents a potential semantic boundary between sentences.
    """

    left_sentence_index: int
    right_sentence_index: int

    similarity: float
    boundary_strength: float

    is_boundary: bool = False
    reason: str | None = None


@dataclass(slots=True)
class SemanticChunkingResult:
    """
    Result of a semantic chunking operation.
    """

    chunks: list[SemanticChunk] = field(default_factory=list)

    sentence_count: int = 0
    boundary_count: int = 0

    used_embeddings: bool = False
    used_lexical_fallback: bool = False

    average_chunk_tokens: float = 0.0
    average_chunk_characters: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_count(self) -> int:
        """
        Return the number of generated chunks.
        """
        return len(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result to a serializable dictionary.
        """
        return {
            "chunks": [
                chunk.to_dict()
                for chunk in self.chunks
            ],
            "sentence_count": self.sentence_count,
            "boundary_count": self.boundary_count,
            "used_embeddings": self.used_embeddings,
            "used_lexical_fallback": self.used_lexical_fallback,
            "average_chunk_tokens": self.average_chunk_tokens,
            "average_chunk_characters": (
                self.average_chunk_characters
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SemanticChunkerConfig:
    """
    Configuration for semantic chunking.
    """

    max_tokens: int = 450
    min_tokens: int = 80
    max_characters: int = 2_000

    similarity_threshold: float = 0.72
    boundary_percentile: float | None = None

    overlap_sentences: int = 1
    overlap_tokens: int = 50

    preserve_headings: bool = True
    preserve_paragraphs: bool = True
    preserve_financial_sentences: bool = True

    use_embeddings: bool = True
    use_lexical_fallback: bool = True

    fallback_similarity_threshold: float = 0.18

    include_sentence_metadata: bool = False
    include_chunk_statistics: bool = True

    def validate(self) -> None:
        """
        Validate configuration.
        """
        if self.max_tokens <= 0:
            raise SemanticChunkerConfigurationError(
                "max_tokens must be greater than zero."
            )

        if self.min_tokens < 0:
            raise SemanticChunkerConfigurationError(
                "min_tokens cannot be negative."
            )

        if self.min_tokens > self.max_tokens:
            raise SemanticChunkerConfigurationError(
                "min_tokens cannot exceed max_tokens."
            )

        if self.max_characters <= 0:
            raise SemanticChunkerConfigurationError(
                "max_characters must be greater than zero."
            )

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise SemanticChunkerConfigurationError(
                "similarity_threshold must be between 0 and 1."
            )

        if not 0.0 <= self.fallback_similarity_threshold <= 1.0:
            raise SemanticChunkerConfigurationError(
                "fallback_similarity_threshold must be between 0 and 1."
            )

        if self.boundary_percentile is not None:
            if not 0.0 <= self.boundary_percentile <= 100.0:
                raise SemanticChunkerConfigurationError(
                    "boundary_percentile must be between 0 and 100."
                )

        if self.overlap_sentences < 0:
            raise SemanticChunkerConfigurationError(
                "overlap_sentences cannot be negative."
            )

        if self.overlap_tokens < 0:
            raise SemanticChunkerConfigurationError(
                "overlap_tokens cannot be negative."
            )


# ============================================================================
# Semantic Chunker
# ============================================================================


class SemanticChunker:
    """
    Semantic chunker using sentence embeddings and similarity boundaries.

    Parameters
    ----------
    embedding_provider:
        Optional embedding provider.

    config:
        Semantic chunker configuration.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProviderProtocol | None = None,
        config: SemanticChunkerConfig | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.config = config or SemanticChunkerConfig()

        self.config.validate()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        *,
        document_id: str | None = None,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SemanticChunkingResult:
        """
        Chunk a document into semantically coherent segments.
        """
        normalized_text = text.strip()

        if not normalized_text:
            return SemanticChunkingResult()

        base_metadata = dict(metadata or {})

        paragraphs = split_paragraphs(normalized_text)

        sentences = self._build_sentence_units(
            paragraphs,
            base_metadata,
        )

        if not sentences:
            return SemanticChunkingResult()

        used_embeddings = False
        used_lexical_fallback = False

        if (
            self.config.use_embeddings
            and self.embedding_provider is not None
        ):
            try:
                self._embed_sentences(sentences)
                used_embeddings = True
            except Exception as exc:
                logger.warning(
                    "Sentence embedding failed; using lexical fallback. "
                    "error=%s",
                    exc,
                )

                if not self.config.use_lexical_fallback:
                    raise SemanticChunkerEmbeddingError(
                        "Unable to generate sentence embeddings."
                    ) from exc

                used_lexical_fallback = True
        else:
            used_lexical_fallback = True

        boundaries = self._detect_boundaries(
            sentences,
            used_embeddings=used_embeddings,
        )

        groups = self._build_sentence_groups(
            sentences,
            boundaries,
        )

        chunks = self._build_chunks(
            groups,
            document_id=document_id,
            source=source,
            metadata=base_metadata,
        )

        result = SemanticChunkingResult(
            chunks=chunks,
            sentence_count=len(sentences),
            boundary_count=sum(
                1
                for boundary in boundaries
                if boundary.is_boundary
            ),
            used_embeddings=used_embeddings,
            used_lexical_fallback=used_lexical_fallback,
        )

        if chunks:
            result.average_chunk_tokens = (
                sum(chunk.token_count for chunk in chunks)
                / len(chunks)
            )

            result.average_chunk_characters = (
                sum(chunk.character_count for chunk in chunks)
                / len(chunks)
            )

        if self.config.include_chunk_statistics:
            result.metadata.update(
                {
                    "max_tokens": self.config.max_tokens,
                    "min_tokens": self.config.min_tokens,
                    "max_characters": self.config.max_characters,
                    "similarity_threshold": (
                        self.config.similarity_threshold
                    ),
                    "overlap_sentences": (
                        self.config.overlap_sentences
                    ),
                }
            )

        return result

    def chunk_many(
        self,
        documents: Iterable[Mapping[str, Any]],
    ) -> list[SemanticChunkingResult]:
        """
        Chunk multiple documents.

        Expected document keys:
        - text or content
        - document_id
        - source
        - metadata
        """
        results: list[SemanticChunkingResult] = []

        for document in documents:
            text = (
                document.get("text")
                or document.get("content")
                or document.get("page_content")
                or ""
            )

            results.append(
                self.chunk(
                    text,
                    document_id=document.get("document_id"),
                    source=document.get("source"),
                    metadata=document.get("metadata"),
                )
            )

        return results

    # ---------------------------------------------------------------------
    # Sentence preparation
    # ---------------------------------------------------------------------

    def _build_sentence_units(
        self,
        paragraphs: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> list[SentenceUnit]:
        """
        Convert paragraphs into sentence units.
        """
        sentences: list[SentenceUnit] = []
        sentence_index = 0

        for paragraph_index, paragraph in enumerate(paragraphs):
            paragraph_sentences = split_sentences(paragraph)

            if not paragraph_sentences:
                continue

            for sentence_text in paragraph_sentences:
                sentence = SentenceUnit(
                    text=sentence_text,
                    index=sentence_index,
                    paragraph_index=paragraph_index,
                    is_heading=(
                        self.config.preserve_headings
                        and is_heading(sentence_text)
                    ),
                    contains_financial_signal=(
                        contains_financial_signal(sentence_text)
                    ),
                    token_count=estimate_tokens(sentence_text),
                    metadata=dict(metadata),
                )

                if self.config.include_sentence_metadata:
                    sentence.metadata.update(
                        {
                            "sentence_index": sentence_index,
                            "paragraph_index": paragraph_index,
                            "is_heading": sentence.is_heading,
                            "contains_financial_signal": (
                                sentence.contains_financial_signal
                            ),
                        }
                    )

                sentences.append(sentence)
                sentence_index += 1

        return sentences

    def _embed_sentences(
        self,
        sentences: Sequence[SentenceUnit],
    ) -> None:
        """
        Generate embeddings for all sentences.
        """
        if self.embedding_provider is None:
            raise SemanticChunkerEmbeddingError(
                "Embedding provider is not configured."
            )

        texts = [
            sentence.text
            for sentence in sentences
        ]

        if hasattr(self.embedding_provider, "embed"):
            embeddings = self.embedding_provider.embed(texts)
        elif hasattr(self.embedding_provider, "embed_documents"):
            embeddings = self.embedding_provider.embed_documents(texts)
        else:
            raise SemanticChunkerEmbeddingError(
                "Embedding provider must expose embed() or "
                "embed_documents()."
            )

        if len(embeddings) != len(sentences):
            raise SemanticChunkerEmbeddingError(
                "Embedding count does not match sentence count."
            )

        for sentence, embedding in zip(sentences, embeddings):
            sentence.embedding = list(embedding)

    # ---------------------------------------------------------------------
    # Boundary detection
    # ---------------------------------------------------------------------

    def _detect_boundaries(
        self,
        sentences: Sequence[SentenceUnit],
        *,
        used_embeddings: bool,
    ) -> list[SemanticBoundary]:
        """
        Detect semantic boundaries between adjacent sentences.
        """
        if len(sentences) <= 1:
            return []

        similarities: list[float] = []

        for left, right in zip(
            sentences,
            sentences[1:],
        ):
            if (
                used_embeddings
                and left.embedding is not None
                and right.embedding is not None
            ):
                similarity = cosine_similarity(
                    left.embedding,
                    right.embedding,
                )
            else:
                similarity = lexical_similarity(
                    left.text,
                    right.text,
                )

            similarities.append(clamp(similarity))

        threshold = self.config.similarity_threshold

        if not used_embeddings:
            threshold = self.config.fallback_similarity_threshold

        if self.config.boundary_percentile is not None:
            threshold = self._percentile_threshold(
                similarities,
                self.config.boundary_percentile,
                default=threshold,
            )

        boundaries: list[SemanticBoundary] = []

        for index, similarity in enumerate(similarities):
            left = sentences[index]
            right = sentences[index + 1]

            boundary_strength = 1.0 - similarity

            is_boundary = similarity < threshold
            reason = "low_similarity"

            if (
                self.config.preserve_paragraphs
                and left.paragraph_index != right.paragraph_index
            ):
                is_boundary = True
                reason = "paragraph_boundary"

            if (
                self.config.preserve_headings
                and right.is_heading
            ):
                is_boundary = True
                reason = "heading_boundary"

            boundaries.append(
                SemanticBoundary(
                    left_sentence_index=left.index,
                    right_sentence_index=right.index,
                    similarity=similarity,
                    boundary_strength=boundary_strength,
                    is_boundary=is_boundary,
                    reason=reason,
                )
            )

        return boundaries

    @staticmethod
    def _percentile_threshold(
        values: Sequence[float],
        percentile: float,
        default: float,
    ) -> float:
        """
        Calculate a simple percentile threshold.
        """
        if not values:
            return default

        sorted_values = sorted(values)

        if len(sorted_values) == 1:
            return sorted_values[0]

        position = (
            percentile / 100.0
        ) * (len(sorted_values) - 1)

        lower = math.floor(position)
        upper = math.ceil(position)

        if lower == upper:
            return sorted_values[lower]

        fraction = position - lower

        return (
            sorted_values[lower]
            + fraction
            * (
                sorted_values[upper]
                - sorted_values[lower]
            )
        )

    # ---------------------------------------------------------------------
    # Group construction
    # ---------------------------------------------------------------------

    def _build_sentence_groups(
        self,
        sentences: Sequence[SentenceUnit],
        boundaries: Sequence[SemanticBoundary],
    ) -> list[list[SentenceUnit]]:
        """
        Build initial sentence groups using semantic boundaries.
        """
        if not sentences:
            return []

        boundary_indexes = {
            boundary.right_sentence_index
            for boundary in boundaries
            if boundary.is_boundary
        }

        groups: list[list[SentenceUnit]] = []
        current_group: list[SentenceUnit] = []

        for sentence in sentences:
            if (
                current_group
                and sentence.index in boundary_indexes
            ):
                groups.append(current_group)
                current_group = []

            current_group.append(sentence)

        if current_group:
            groups.append(current_group)

        return groups

    # ---------------------------------------------------------------------
    # Chunk construction
    # ---------------------------------------------------------------------

    def _build_chunks(
        self,
        groups: Sequence[Sequence[SentenceUnit]],
        *,
        document_id: str | None,
        source: str | None,
        metadata: Mapping[str, Any],
    ) -> list[SemanticChunk]:
        """
        Convert sentence groups into size-constrained chunks.
        """
        chunks: list[SemanticChunk] = []

        for group in groups:
            if not group:
                continue

            subgroups = self._split_group_by_size(group)

            for subgroup in subgroups:
                if not subgroup:
                    continue

                text = " ".join(
                    sentence.text
                    for sentence in subgroup
                ).strip()

                if not text:
                    continue

                chunk_index = len(chunks)

                chunk_metadata = dict(metadata)

                chunk_metadata.update(
                    {
                        "chunking_strategy": "semantic",
                        "sentence_count": len(subgroup),
                        "start_sentence": subgroup[0].index,
                        "end_sentence": subgroup[-1].index,
                    }
                )

                if self.config.include_sentence_metadata:
                    chunk_metadata["sentences"] = [
                        sentence.text
                        for sentence in subgroup
                    ]

                semantic_score = self._calculate_group_score(
                    subgroup
                )

                boundary_score = self._calculate_boundary_score(
                    subgroup
                )

                chunk_id = stable_hash(
                    document_id or "",
                    source or "",
                    chunk_index,
                    text,
                )

                chunks.append(
                    SemanticChunk(
                        chunk_id=chunk_id,
                        text=text,
                        document_id=document_id,
                        source=source,
                        index=chunk_index,
                        start_sentence=subgroup[0].index,
                        end_sentence=subgroup[-1].index,
                        token_count=estimate_tokens(text),
                        character_count=len(text),
                        semantic_score=semantic_score,
                        boundary_score=boundary_score,
                        metadata=chunk_metadata,
                    )
                )

        return self._apply_overlap(
            chunks,
            document_id=document_id,
            source=source,
        )

    def _split_group_by_size(
        self,
        group: Sequence[SentenceUnit],
    ) -> list[list[SentenceUnit]]:
        """
        Split a semantic group when token or character limits are reached.
        """
        result: list[list[SentenceUnit]] = []
        current: list[SentenceUnit] = []

        current_tokens = 0
        current_characters = 0

        for sentence in group:
            sentence_tokens = sentence.token_count
            sentence_characters = len(sentence.text)

            exceeds_tokens = (
                current
                and current_tokens + sentence_tokens > self.config.max_tokens
            )

            exceeds_characters = (
                current
                and (
                    current_characters
                    + sentence_characters
                    + 1
                    > self.config.max_characters
                )
            )

            if exceeds_tokens or exceeds_characters:
                result.append(current)

                current = []
                current_tokens = 0
                current_characters = 0

            current.append(sentence)
            current_tokens += sentence_tokens
            current_characters += sentence_characters + 1

        if current:
            result.append(current)

        return result

    def _apply_overlap(
        self,
        chunks: Sequence[SemanticChunk],
        *,
        document_id: str | None,
        source: str | None,
    ) -> list[SemanticChunk]:
        """
        Apply sentence overlap between adjacent chunks.

        Overlap is applied conservatively. The original chunk content
        remains intact; overlap is added only when it does not violate
        configured size limits.
        """
        if (
            self.config.overlap_sentences <= 0
            or len(chunks) <= 1
        ):
            return list(chunks)

        result: list[SemanticChunk] = []

        for index, chunk in enumerate(chunks):
            if index == 0:
                result.append(chunk)
                continue

            previous = result[-1]

            previous_sentences = split_sentences(previous.text)

            if not previous_sentences:
                result.append(chunk)
                continue

            overlap_sentences = previous_sentences[
                -self.config.overlap_sentences:
            ]

            overlap_text = " ".join(overlap_sentences).strip()

            if not overlap_text:
                result.append(chunk)
                continue

            combined_text = (
                overlap_text
                + " "
                + chunk.text
            ).strip()

            combined_tokens = estimate_tokens(combined_text)

            if (
                combined_tokens > self.config.max_tokens
                or len(combined_text) > self.config.max_characters
            ):
                result.append(chunk)
                continue

            combined_metadata = dict(chunk.metadata)
            combined_metadata["has_overlap"] = True
            combined_metadata["overlap_text"] = overlap_text

            new_chunk = SemanticChunk(
                chunk_id=stable_hash(
                    document_id or chunk.document_id or "",
                    source or chunk.source or "",
                    index,
                    combined_text,
                ),
                text=combined_text,
                document_id=document_id or chunk.document_id,
                source=source or chunk.source,
                index=index,
                start_sentence=chunk.start_sentence,
                end_sentence=chunk.end_sentence,
                token_count=combined_tokens,
                character_count=len(combined_text),
                semantic_score=chunk.semantic_score,
                boundary_score=chunk.boundary_score,
                metadata=combined_metadata,
            )

            result.append(new_chunk)

        for index, chunk in enumerate(result):
            chunk.index = index

        return result

    # ---------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------

    def _calculate_group_score(
        self,
        group: Sequence[SentenceUnit],
    ) -> float | None:
        """
        Calculate the average adjacent sentence similarity.
        """
        similarities: list[float] = []

        for left, right in zip(
            group,
            group[1:],
        ):
            if (
                left.embedding is not None
                and right.embedding is not None
            ):
                similarities.append(
                    cosine_similarity(
                        left.embedding,
                        right.embedding,
                    )
                )
            else:
                similarities.append(
                    lexical_similarity(
                        left.text,
                        right.text,
                    )
                )

        if not similarities:
            return None

        return sum(similarities) / len(similarities)

    def _calculate_boundary_score(
        self,
        group: Sequence[SentenceUnit],
    ) -> float | None:
        """
        Calculate a rough boundary confidence score.
        """
        if not group:
            return None

        first = group[0]
        last = group[-1]

        if first.index == last.index:
            return 1.0

        if (
            first.embedding is not None
            and last.embedding is not None
        ):
            return clamp(
                cosine_similarity(
                    first.embedding,
                    last.embedding,
                )
            )

        return clamp(
            lexical_similarity(
                first.text,
                last.text,
            )
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def create_semantic_chunker(
    embedding_provider: EmbeddingProviderProtocol | None = None,
    config: SemanticChunkerConfig | None = None,
) -> SemanticChunker:
    """
    Create a configured SemanticChunker.
    """
    return SemanticChunker(
        embedding_provider=embedding_provider,
        config=config,
    )


def semantic_chunk_text(
    text: str,
    *,
    embedding_provider: EmbeddingProviderProtocol | None = None,
    document_id: str | None = None,
    source: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    config: SemanticChunkerConfig | None = None,
) -> list[SemanticChunk]:
    """
    Convenience function that returns only chunks.
    """
    chunker = create_semantic_chunker(
        embedding_provider=embedding_provider,
        config=config,
    )

    result = chunker.chunk(
        text,
        document_id=document_id,
        source=source,
        metadata=metadata,
    )

    return result.chunks


# ============================================================================
# Public Exports
# ============================================================================


__all__ = [
    "SemanticChunkerError",
    "SemanticChunkerConfigurationError",
    "SemanticChunkerEmbeddingError",
    "EmbeddingProviderProtocol",
    "SemanticChunk",
    "SentenceUnit",
    "SemanticBoundary",
    "SemanticChunkingResult",
    "SemanticChunkerConfig",
    "SemanticChunker",
    "create_semantic_chunker",
    "semantic_chunk_text",
    "normalize_text",
    "split_sentences",
    "split_paragraphs",
    "cosine_similarity",
    "lexical_similarity",
    "estimate_tokens",
    "contains_financial_signal",
]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.app.rag.embeddings import (
        HashEmbeddingProvider,
    )

    sample_text = """
    Revenue increased by 18% in FY2025 compared with FY2024.
    The increase was mainly driven by higher enterprise sales and
    improved customer retention.

    Operating expenses decreased by 6% during FY2025.
    Management expects margins to improve further in FY2026.

    The company maintains a conservative debt position.
    Its debt-to-equity ratio remained below 0.5.
    """

    embedding_provider = HashEmbeddingProvider(
        dimension=384
    )

    chunker = SemanticChunker(
        embedding_provider=embedding_provider,
        config=SemanticChunkerConfig(
            max_tokens=180,
            min_tokens=40,
            max_characters=1_000,
            similarity_threshold=0.70,
            overlap_sentences=1,
            preserve_paragraphs=True,
            preserve_headings=True,
        ),
    )

    result = chunker.chunk(
        sample_text,
        document_id="financial-report-001",
        source="annual_report.pdf",
        metadata={
            "tenant_id": "tenant-001",
            "document_type": "annual_report",
            "fiscal_year": 2025,
        },
    )

    for chunk in result.chunks:
        print("=" * 80)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Tokens: {chunk.token_count}")
        print(chunk.text)