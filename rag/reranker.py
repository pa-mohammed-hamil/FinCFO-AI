"""
Reranking utilities for FinCo AI.

Supports:
- Cross-encoder reranking
- Lightweight lexical reranking
- Metadata-aware reranking
- Score blending
- Duplicate removal
- Diversity-aware ranking
- Configurable top-k selection

Recommended production flow:

    Query
      ↓
    Hybrid Retriever
      ↓
    Candidate Documents
      ↓
    Reranker
      ↓
    Context Compressor
      ↓
    Citation Generator
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

from backend.app.rag.hybrid_retriever import (
    RetrievalResult,
    min_max_normalize,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class RerankerError(RuntimeError):
    """Base exception for reranking errors."""


class RerankerConfigurationError(RerankerError):
    """Raised when reranker configuration is invalid."""


# ============================================================================
# Data models
# ============================================================================


@dataclass
class RerankCandidate:
    """
    Candidate document prepared for reranking.
    """

    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    retrieval_rank: int | None = None

    @classmethod
    def from_retrieval_result(
        cls,
        result: RetrievalResult,
    ) -> "RerankCandidate":
        return cls(
            document_id=result.document_id,
            text=result.text,
            metadata=dict(result.metadata),
            retrieval_score=result.score,
            dense_score=result.dense_score,
            sparse_score=result.sparse_score,
            retrieval_rank=result.fused_rank,
        )


@dataclass
class RerankResult:
    """
    Final reranked document.
    """

    document_id: str
    text: str
    score: float
    rerank_score: float
    retrieval_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    original_rank: int | None = None
    rerank_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str | None:
        source = self.metadata.get("source")
        return str(source) if source is not None else None

    @property
    def page(self) -> Any:
        return self.metadata.get("page")

    @property
    def section(self) -> str | None:
        section = self.metadata.get("section")
        return str(section) if section is not None else None


@dataclass
class RerankResponse:
    """
    Structured reranking response.
    """

    query: str
    results: list[RerankResult]
    candidate_count: int
    returned_count: int
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankerConfig:
    """
    Reranker configuration.
    """

    top_k: int = 10
    retrieval_weight: float = 0.30
    rerank_weight: float = 0.70
    lexical_weight: float = 0.40
    metadata_weight: float = 0.10
    diversity_weight: float = 0.10
    max_text_length: int = 4_000
    remove_duplicates: bool = True
    use_metadata_boost: bool = True


# ============================================================================
# Protocols
# ============================================================================


class CrossEncoderProtocol(Protocol):
    """
    Protocol implemented by cross-encoder models.
    """

    def predict(
        self,
        sentence_pairs: Sequence[tuple[str, str]],
        **kwargs: Any,
    ) -> Sequence[float]:
        ...


class RerankerProtocol(Protocol):
    """
    Protocol for custom rerankers.
    """

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> Sequence[RerankResult]:
        ...


# ============================================================================
# Utility functions
# ============================================================================


def normalize_text(text: str) -> str:
    """
    Normalize text for lexical comparisons.
    """

    return " ".join(str(text or "").lower().split())


def tokenize(text: str) -> list[str]:
    """
    Tokenize text into alphanumeric terms.
    """

    return re.findall(
        r"[a-zA-Z0-9]+(?:[-./][a-zA-Z0-9]+)*",
        normalize_text(text),
    )


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def lexical_overlap(
    query: str,
    document: str,
) -> float:
    """
    Calculate query-document token overlap.

    Returns:
        Value between 0.0 and 1.0.
    """

    query_tokens = token_set(query)
    document_tokens = token_set(document)

    if not query_tokens or not document_tokens:
        return 0.0

    return len(query_tokens.intersection(document_tokens)) / len(
        query_tokens
    )


def phrase_overlap(
    query: str,
    document: str,
    *,
    phrase_length: int = 2,
) -> float:
    """
    Calculate overlap of consecutive query token phrases.
    """

    query_tokens = tokenize(query)
    document_text = normalize_text(document)

    if len(query_tokens) < phrase_length:
        return 0.0

    phrases = [
        " ".join(query_tokens[index : index + phrase_length])
        for index in range(
            len(query_tokens) - phrase_length + 1
        )
    ]

    if not phrases:
        return 0.0

    matches = sum(
        phrase in document_text
        for phrase in phrases
    )

    return matches / len(phrases)


def number_tokens(text: str) -> set[str]:
    """
    Extract numeric tokens, including decimal values and percentages.
    """

    return set(
        re.findall(
            r"\b\d+(?:[.,]\d+)?%?\b",
            str(text or ""),
        )
    )


def numeric_overlap(
    query: str,
    document: str,
) -> float:
    """
    Compare numeric values in the query and document.

    This is especially useful for financial queries involving:
    - Years
    - Percentages
    - Amounts
    - Account numbers
    - Transaction IDs
    """

    query_numbers = number_tokens(query)
    document_numbers = number_tokens(document)

    if not query_numbers:
        return 0.0

    return len(query_numbers.intersection(document_numbers)) / len(
        query_numbers
    )


def safe_score(value: Any) -> float:
    """
    Convert a score to a finite float.
    """

    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(score):
        return 0.0

    return score


def truncate_text(
    text: str,
    *,
    max_length: int,
) -> str:
    """
    Truncate text for model input.
    """

    value = str(text or "")

    if len(value) <= max_length:
        return value

    return value[:max_length].rstrip() + "..."


def deduplicate_candidates(
    candidates: Sequence[RerankCandidate],
) -> list[RerankCandidate]:
    """
    Remove duplicate candidates by document ID and normalized text.
    """

    result: list[RerankCandidate] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()

    for candidate in candidates:
        document_id = str(candidate.document_id)
        normalized_document = normalize_text(candidate.text)

        if document_id in seen_ids:
            continue

        if normalized_document and normalized_document in seen_texts:
            continue

        result.append(candidate)
        seen_ids.add(document_id)

        if normalized_document:
            seen_texts.add(normalized_document)

    return result


# ============================================================================
# Base reranker
# ============================================================================


class BaseReranker:
    """
    Base interface and common implementation for rerankers.
    """

    model_name = "base-reranker"

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        raise NotImplementedError


# ============================================================================
# Lexical reranker
# ============================================================================


class LexicalReranker(BaseReranker):
    """
    Lightweight reranker without external model dependencies.

    Score components:
    - Token overlap
    - Phrase overlap
    - Numeric overlap
    - Original retrieval score
    """

    model_name = "lexical-reranker"

    def __init__(
        self,
        *,
        lexical_weight: float = 0.55,
        phrase_weight: float = 0.20,
        numeric_weight: float = 0.15,
        retrieval_weight: float = 0.10,
        max_text_length: int = 4_000,
    ) -> None:
        weights = [
            lexical_weight,
            phrase_weight,
            numeric_weight,
            retrieval_weight,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError("Reranker weights cannot be negative.")

        if sum(weights) == 0:
            raise ValueError(
                "At least one reranker weight must be greater than zero."
            )

        self.lexical_weight = lexical_weight
        self.phrase_weight = phrase_weight
        self.numeric_weight = numeric_weight
        self.retrieval_weight = retrieval_weight
        self.max_text_length = max_text_length

    def _score_candidate(
        self,
        query: str,
        candidate: RerankCandidate,
        normalized_retrieval_score: float,
    ) -> float:
        document_text = truncate_text(
            candidate.text,
            max_length=self.max_text_length,
        )

        lexical_score = lexical_overlap(
            query,
            document_text,
        )

        phrase_score = phrase_overlap(
            query,
            document_text,
        )

        numeric_score = numeric_overlap(
            query,
            document_text,
        )

        return (
            self.lexical_weight * lexical_score
            + self.phrase_weight * phrase_score
            + self.numeric_weight * numeric_score
            + self.retrieval_weight * normalized_retrieval_score
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        retrieval_scores = [
            safe_score(candidate.retrieval_score)
            for candidate in candidates
        ]

        normalized_scores = min_max_normalize(
            retrieval_scores
        )

        results: list[RerankResult] = []

        for candidate, normalized_retrieval_score in zip(
            candidates,
            normalized_scores,
        ):
            score = self._score_candidate(
                query,
                candidate,
                normalized_retrieval_score,
            )

            results.append(
                RerankResult(
                    document_id=candidate.document_id,
                    text=candidate.text,
                    score=score,
                    rerank_score=score,
                    retrieval_score=candidate.retrieval_score,
                    dense_score=candidate.dense_score,
                    sparse_score=candidate.sparse_score,
                    original_rank=candidate.retrieval_rank,
                    metadata=dict(candidate.metadata),
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        for rank, result in enumerate(results, start=1):
            result.rerank_rank = rank

        if top_k is not None:
            return results[:max(0, top_k)]

        return results


# ============================================================================
# Cross-encoder reranker
# ============================================================================


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using sentence-transformers CrossEncoder.

    Install:
        pip install sentence-transformers

    Example:

        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int = 16,
        max_text_length: int = 4_000,
        retrieval_weight: float = 0.30,
        rerank_weight: float = 0.70,
        fallback_to_lexical: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if retrieval_weight < 0 or rerank_weight < 0:
            raise ValueError("Reranker weights cannot be negative.")

        if retrieval_weight + rerank_weight == 0:
            raise ValueError(
                "At least one score weight must be greater than zero."
            )

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerConfigurationError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.retrieval_weight = retrieval_weight
        self.rerank_weight = rerank_weight
        self.fallback_to_lexical = fallback_to_lexical

        self._model = CrossEncoder(model_name)

        self._fallback = LexicalReranker(
            max_text_length=max_text_length
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not candidates:
            return []

        try:
            pairs = [
                (
                    query,
                    truncate_text(
                        candidate.text,
                        max_length=self.max_text_length,
                    ),
                )
                for candidate in candidates
            ]

            raw_scores: list[float] = []

            for start in range(
                0,
                len(pairs),
                self.batch_size,
            ):
                batch = pairs[start : start + self.batch_size]

                predictions = self._model.predict(
                    batch,
                    show_progress_bar=False,
                )

                raw_scores.extend(
                    safe_score(score)
                    for score in predictions
                )

            if len(raw_scores) != len(candidates):
                raise RerankerError(
                    "Cross-encoder returned an unexpected number of scores."
                )

            normalized_rerank_scores = min_max_normalize(
                raw_scores
            )

            normalized_retrieval_scores = min_max_normalize(
                [
                    safe_score(candidate.retrieval_score)
                    for candidate in candidates
                ]
            )

            results: list[RerankResult] = []

            for (
                candidate,
                rerank_score,
                retrieval_score,
            ) in zip(
                candidates,
                normalized_rerank_scores,
                normalized_retrieval_scores,
            ):
                final_score = (
                    self.rerank_weight * rerank_score
                    + self.retrieval_weight * retrieval_score
                )

                results.append(
                    RerankResult(
                        document_id=candidate.document_id,
                        text=candidate.text,
                        score=final_score,
                        rerank_score=rerank_score,
                        retrieval_score=candidate.retrieval_score,
                        dense_score=candidate.dense_score,
                        sparse_score=candidate.sparse_score,
                        original_rank=candidate.retrieval_rank,
                        metadata=dict(candidate.metadata),
                    )
                )

            results.sort(
                key=lambda result: result.score,
                reverse=True,
            )

            for rank, result in enumerate(results, start=1):
                result.rerank_rank = rank

            if top_k is not None:
                return results[:max(0, top_k)]

            return results

        except Exception as exc:
            if not self.fallback_to_lexical:
                raise RerankerError(
                    f"Cross-encoder reranking failed: {exc}"
                ) from exc

            logger.warning(
                "Cross-encoder reranking failed; using lexical fallback: %s",
                exc,
            )

            return self._fallback.rerank(
                query,
                candidates,
                top_k=top_k,
            )


# ============================================================================
# Metadata-aware reranker
# ============================================================================


class MetadataAwareReranker(BaseReranker):
    """
    Adds metadata relevance to an existing reranker.

    Useful metadata:
    - document_type
    - section
    - source
    - year
    - quarter
    - currency
    - account_id
    - transaction_id
    """

    model_name = "metadata-aware-reranker"

    def __init__(
        self,
        base_reranker: BaseReranker | None = None,
        *,
        metadata_weight: float = 0.10,
        preferred_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if metadata_weight < 0:
            raise ValueError(
                "metadata_weight cannot be negative."
            )

        self.base_reranker = (
            base_reranker
            or LexicalReranker()
        )
        self.metadata_weight = metadata_weight
        self.preferred_metadata = dict(
            preferred_metadata or {}
        )

    def _metadata_score(
        self,
        metadata: Mapping[str, Any],
    ) -> float:
        if not self.preferred_metadata:
            return 0.0

        matched = 0

        for key, expected_value in self.preferred_metadata.items():
            actual_value = metadata.get(key)

            if isinstance(expected_value, (list, tuple, set)):
                if actual_value in expected_value:
                    matched += 1
            elif actual_value == expected_value:
                matched += 1

        return matched / len(self.preferred_metadata)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        base_results = list(
            self.base_reranker.rerank(
                query,
                candidates,
                top_k=None,
            )
        )

        if not base_results:
            return []

        base_scores = min_max_normalize(
            [
                safe_score(result.score)
                for result in base_results
            ]
        )

        final_results: list[RerankResult] = []

        for result, base_score in zip(
            base_results,
            base_scores,
        ):
            metadata_score = self._metadata_score(
                result.metadata
            )

            final_score = (
                (1.0 - self.metadata_weight) * base_score
                + self.metadata_weight * metadata_score
            )

            result.score = final_score
            result.rerank_score = final_score
            final_results.append(result)

        final_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        for rank, result in enumerate(final_results, start=1):
            result.rerank_rank = rank

        if top_k is not None:
            return final_results[:max(0, top_k)]

        return final_results


# ============================================================================
# Diversity-aware reranker
# ============================================================================


class DiversityAwareReranker(BaseReranker):
    """
    Adds a simple diversity penalty to avoid returning many nearly identical
    chunks from the same source or section.
    """

    model_name = "diversity-aware-reranker"

    def __init__(
        self,
        base_reranker: BaseReranker | None = None,
        *,
        diversity_weight: float = 0.10,
        same_source_penalty: float = 0.05,
        same_section_penalty: float = 0.03,
    ) -> None:
        if diversity_weight < 0:
            raise ValueError(
                "diversity_weight cannot be negative."
            )

        self.base_reranker = (
            base_reranker
            or LexicalReranker()
        )
        self.diversity_weight = diversity_weight
        self.same_source_penalty = same_source_penalty
        self.same_section_penalty = same_section_penalty

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        limit = top_k if top_k is not None else len(candidates)

        base_results = list(
            self.base_reranker.rerank(
                query,
                candidates,
                top_k=None,
            )
        )

        selected: list[RerankResult] = []
        remaining = list(base_results)

        while remaining and len(selected) < limit:
            best_result: RerankResult | None = None
            best_score = float("-inf")

            for result in remaining:
                penalty = 0.0

                for selected_result in selected:
                    if (
                        result.source
                        and selected_result.source
                        and result.source == selected_result.source
                    ):
                        penalty += self.same_source_penalty

                    if (
                        result.section
                        and selected_result.section
                        and result.section == selected_result.section
                    ):
                        penalty += self.same_section_penalty

                adjusted_score = (
                    result.score
                    - self.diversity_weight * penalty
                )

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_result = result

            if best_result is None:
                break

            best_result.score = best_score
            selected.append(best_result)
            remaining.remove(best_result)

        for rank, result in enumerate(selected, start=1):
            result.rerank_rank = rank

        return selected


# ============================================================================
# High-level service
# ============================================================================


class RerankingService:
    """
    High-level reranking service for FinCo AI.
    """

    def __init__(
        self,
        reranker: BaseReranker | None = None,
        *,
        config: RerankerConfig | None = None,
    ) -> None:
        self.config = config or RerankerConfig()

        self.reranker = (
            reranker
            or LexicalReranker(
                lexical_weight=0.55,
                phrase_weight=0.20,
                numeric_weight=0.15,
                retrieval_weight=0.10,
                max_text_length=self.config.max_text_length,
            )
        )

    def prepare_candidates(
        self,
        results: Sequence[RetrievalResult | RerankCandidate],
    ) -> list[RerankCandidate]:
        """
        Convert retrieval results into rerank candidates.
        """

        candidates: list[RerankCandidate] = []

        for result in results:
            if isinstance(result, RerankCandidate):
                candidates.append(result)
            else:
                candidates.append(
                    RerankCandidate.from_retrieval_result(
                        result
                    )
                )

        if self.config.remove_duplicates:
            candidates = deduplicate_candidates(
                candidates
            )

        return candidates

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult | RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> RerankResponse:
        """
        Rerank retrieval results.
        """

        query = str(query or "").strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        candidates = self.prepare_candidates(results)

        requested_top_k = (
            top_k
            if top_k is not None
            else self.config.top_k
        )

        if requested_top_k <= 0:
            return RerankResponse(
                query=query,
                results=[],
                candidate_count=len(candidates),
                returned_count=0,
                model_name=getattr(
                    self.reranker,
                    "model_name",
                    self.reranker.__class__.__name__,
                ),
            )

        reranked_results = list(
            self.reranker.rerank(
                query,
                candidates,
                top_k=requested_top_k,
            )
        )

        return RerankResponse(
            query=query,
            results=reranked_results,
            candidate_count=len(candidates),
            returned_count=len(reranked_results),
            model_name=getattr(
                self.reranker,
                "model_name",
                self.reranker.__class__.__name__,
            ),
            metadata={
                "remove_duplicates": self.config.remove_duplicates,
                "top_k": requested_top_k,
            },
        )

    def retrieve(
        self,
        query: str,
        results: Sequence[RetrievalResult | RerankCandidate],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """
        Convenience method returning only reranked documents.
        """

        return self.rerank(
            query,
            results,
            top_k=top_k,
        ).results


# ============================================================================
# Factory functions
# ============================================================================


def create_reranker(
    *,
    provider: str = "lexical",
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseReranker:
    """
    Create a reranker.

    Supported providers:
        - lexical
        - cross_encoder
        - cross-encoder
        - metadata
        - diversity
    """

    selected_provider = provider.strip().lower()

    if selected_provider in {
        "lexical",
        "keyword",
        "bm25",
    }:
        return LexicalReranker(**kwargs)

    if selected_provider in {
        "cross_encoder",
        "cross-encoder",
        "crossencoder",
    }:
        return CrossEncoderReranker(
            model_name=(
                model_name
                or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            **kwargs,
        )

    if selected_provider in {
        "metadata",
        "metadata-aware",
    }:
        return MetadataAwareReranker(**kwargs)

    if selected_provider in {
        "diversity",
        "diversity-aware",
    }:
        return DiversityAwareReranker(**kwargs)

    raise RerankerConfigurationError(
        f"Unsupported reranker provider: {provider}"
    )


def create_reranking_service(
    *,
    provider: str = "lexical",
    config: RerankerConfig | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> RerankingService:
    """
    Create a configured reranking service.
    """

    reranker = create_reranker(
        provider=provider,
        model_name=model_name,
        **kwargs,
    )

    return RerankingService(
        reranker=reranker,
        config=config,
    )


__all__ = [
    "RerankerError",
    "RerankerConfigurationError",
    "RerankCandidate",
    "RerankResult",
    "RerankResponse",
    "RerankerConfig",
    "BaseReranker",
    "LexicalReranker",
    "CrossEncoderReranker",
    "MetadataAwareReranker",
    "DiversityAwareReranker",
    "RerankingService",
    "lexical_overlap",
    "phrase_overlap",
    "numeric_overlap",
    "deduplicate_candidates",
    "create_reranker",
    "create_reranking_service",
]