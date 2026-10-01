"""
Hybrid retrieval for FinCo AI.

Combines:
- Dense vector similarity
- BM25 keyword retrieval
- Reciprocal Rank Fusion
- Optional metadata filtering
- Optional score normalization
- Duplicate document removal

The module is provider-agnostic and can work with:
- In-memory documents
- Chroma-like vector stores
- Custom vector databases
- Existing BM25 implementations
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from backend.app.rag.embeddings import (
    EmbeddingService,
    cosine_similarity,
    normalize_vector,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class RetrievalError(RuntimeError):
    """Base exception for retrieval failures."""


class RetrievalConfigurationError(RetrievalError):
    """Raised when retrieval is incorrectly configured."""


# ============================================================================
# Data models
# ============================================================================


@dataclass
class RetrievalDocument:
    """
    A searchable document or chunk.

    Attributes:
        document_id:
            Unique document/chunk identifier.
        text:
            Text content used for retrieval.
        metadata:
            Optional metadata such as source, page, section, account,
            transaction category, date, or tenant ID.
        embedding:
            Optional precomputed embedding.
    """

    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def __post_init__(self) -> None:
        self.document_id = str(self.document_id)
        self.text = str(self.text or "")

        if self.metadata is None:
            self.metadata = {}

        if self.embedding is not None:
            self.embedding = [float(value) for value in self.embedding]


@dataclass
class RetrievalResult:
    """
    Final hybrid retrieval result.
    """

    document_id: str
    text: str
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str | None:
        value = self.metadata.get("source")
        return str(value) if value is not None else None

    @property
    def page(self) -> Any:
        return self.metadata.get("page")

    @property
    def section(self) -> str | None:
        value = self.metadata.get("section")
        return str(value) if value is not None else None


@dataclass
class RetrievalResponse:
    """
    Structured response from hybrid retrieval.
    """

    query: str
    results: list[RetrievalResult]
    total_candidates: int
    dense_candidates: int
    sparse_candidates: int
    filters: dict[str, Any] = field(default_factory=dict)
    strategy: str = "hybrid_rrf"

    @property
    def top_result(self) -> RetrievalResult | None:
        return self.results[0] if self.results else None


# ============================================================================
# Protocols
# ============================================================================


class SparseRetriever(Protocol):
    """
    Protocol for BM25 or another keyword retriever.
    """

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Any]:
        ...


class DenseRetriever(Protocol):
    """
    Protocol for vector retrieval.
    """

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> Sequence[Any]:
        ...


# ============================================================================
# Text utilities
# ============================================================================


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+(?:[-./][A-Za-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for lightweight keyword retrieval.
    """

    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(str(text or ""))
    ]


def normalize_score(
    score: float,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Normalize a score to the supplied range.
    """

    if not math.isfinite(score):
        return minimum

    if maximum <= minimum:
        return minimum

    return max(minimum, min(maximum, score))


def min_max_normalize(scores: Sequence[float]) -> list[float]:
    """
    Min-max normalize a list of scores.

    If all scores are equal, every score is returned as 1.0.
    """

    if not scores:
        return []

    numeric_scores = [
        float(score) if math.isfinite(float(score)) else 0.0
        for score in scores
    ]

    minimum = min(numeric_scores)
    maximum = max(numeric_scores)

    if math.isclose(minimum, maximum):
        return [1.0 for _ in numeric_scores]

    return [
        (score - minimum) / (maximum - minimum)
        for score in numeric_scores
    ]


# ============================================================================
# In-memory BM25 implementation
# ============================================================================


class InMemoryBM25:
    """
    Lightweight BM25 retriever.

    Useful for:
    - Local development
    - Small and medium document collections
    - Testing
    - Fallback retrieval

    For very large collections, use a dedicated search engine or vector
    database with BM25 support.
    """

    def __init__(
        self,
        documents: Sequence[RetrievalDocument] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = k1
        self.b = b
        self.documents: list[RetrievalDocument] = []
        self._tokenized_documents: list[list[str]] = []
        self._document_frequencies: dict[str, int] = {}
        self._average_document_length = 0.0

        if documents:
            self.add_documents(documents)

    def add_documents(
        self,
        documents: Iterable[RetrievalDocument],
    ) -> None:
        """
        Add documents to the BM25 index.
        """

        for document in documents:
            self.documents.append(document)

        self._rebuild_index()

    def clear(self) -> None:
        """
        Clear the index.
        """

        self.documents.clear()
        self._tokenized_documents.clear()
        self._document_frequencies.clear()
        self._average_document_length = 0.0

    def _rebuild_index(self) -> None:
        self._tokenized_documents = [
            tokenize(document.text)
            for document in self.documents
        ]

        self._document_frequencies.clear()

        for tokens in self._tokenized_documents:
            for token in set(tokens):
                self._document_frequencies[token] = (
                    self._document_frequencies.get(token, 0) + 1
                )

        total_length = sum(
            len(tokens)
            for tokens in self._tokenized_documents
        )

        self._average_document_length = (
            total_length / len(self._tokenized_documents)
            if self._tokenized_documents
            else 0.0
        )

    def _idf(self, token: str) -> float:
        document_count = len(self.documents)
        frequency = self._document_frequencies.get(token, 0)

        if document_count == 0 or frequency == 0:
            return 0.0

        return math.log(
            1.0
            + (
                document_count - frequency + 0.5
            ) / (
                frequency + 0.5
            )
        )

    def _score_document(
        self,
        query_tokens: Sequence[str],
        document_tokens: Sequence[str],
    ) -> float:
        if not query_tokens or not document_tokens:
            return 0.0

        document_length = len(document_tokens)
        average_length = self._average_document_length or 1.0

        term_frequencies: dict[str, int] = {}

        for token in document_tokens:
            term_frequencies[token] = term_frequencies.get(token, 0) + 1

        score = 0.0

        for token in query_tokens:
            frequency = term_frequencies.get(token, 0)

            if frequency == 0:
                continue

            idf = self._idf(token)

            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + self.k1 * (
                1.0
                - self.b
                + self.b * document_length / average_length
            )

            score += idf * numerator / denominator

        return score

    @staticmethod
    def _matches_filters(
        document: RetrievalDocument,
        filters: Mapping[str, Any] | None,
    ) -> bool:
        if not filters:
            return True

        for key, expected_value in filters.items():
            actual_value = document.metadata.get(key)

            if isinstance(expected_value, (list, tuple, set, frozenset)):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False

        return True

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Search documents using BM25.
        """

        if top_k <= 0:
            return []

        query_tokens = tokenize(query)
        scored_results: list[RetrievalResult] = []

        for index, document in enumerate(self.documents):
            if not self._matches_filters(document, filters):
                continue

            score = self._score_document(
                query_tokens,
                self._tokenized_documents[index],
            )

            if score <= 0:
                continue

            scored_results.append(
                RetrievalResult(
                    document_id=document.document_id,
                    text=document.text,
                    score=score,
                    sparse_score=score,
                    metadata=dict(document.metadata),
                )
            )

        scored_results.sort(
            key=lambda result: result.sparse_score,
            reverse=True,
        )

        for rank, result in enumerate(scored_results, start=1):
            result.sparse_rank = rank

        return scored_results[:top_k]


# ============================================================================
# In-memory dense retriever
# ============================================================================


class InMemoryDenseRetriever:
    """
    Dense vector retriever over in-memory documents.

    Documents must contain embeddings. If embeddings are missing, the
    retriever raises a configuration error.
    """

    def __init__(
        self,
        documents: Sequence[RetrievalDocument] | None = None,
    ) -> None:
        self.documents: list[RetrievalDocument] = []

        if documents:
            self.add_documents(documents)

    def add_documents(
        self,
        documents: Iterable[RetrievalDocument],
    ) -> None:
        for document in documents:
            if document.embedding is None:
                raise RetrievalConfigurationError(
                    "Dense retrieval requires an embedding for every document: "
                    f"{document.document_id}"
                )

            document.embedding = normalize_vector(document.embedding)
            self.documents.append(document)

    def clear(self) -> None:
        self.documents.clear()

    @staticmethod
    def _matches_filters(
        document: RetrievalDocument,
        filters: Mapping[str, Any] | None,
    ) -> bool:
        if not filters:
            return True

        for key, expected_value in filters.items():
            actual_value = document.metadata.get(key)

            if isinstance(expected_value, (list, tuple, set, frozenset)):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False

        return True

    def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        query_vector = normalize_vector(query_vector)
        scored_results: list[RetrievalResult] = []

        for document in self.documents:
            if not self._matches_filters(document, filters):
                continue

            if document.embedding is None:
                continue

            score = cosine_similarity(
                query_vector,
                document.embedding,
            )

            scored_results.append(
                RetrievalResult(
                    document_id=document.document_id,
                    text=document.text,
                    score=score,
                    dense_score=score,
                    metadata=dict(document.metadata),
                )
            )

        scored_results.sort(
            key=lambda result: result.dense_score,
            reverse=True,
        )

        for rank, result in enumerate(scored_results, start=1):
            result.dense_rank = rank

        return scored_results[:top_k]


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """
    Combine ranked result lists using Reciprocal Rank Fusion.

    RRF formula:

        score(document) = sum(1 / (k + rank))

    Ranks start at 1.
    """

    if k <= 0:
        raise ValueError("RRF k must be greater than zero.")

    fused_scores: dict[str, float] = {}
    result_lookup: dict[str, RetrievalResult] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            document_id = result.document_id

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0.0)
                + 1.0 / (k + rank)
            )

            if document_id not in result_lookup:
                result_lookup[document_id] = RetrievalResult(
                    document_id=document_id,
                    text=result.text,
                    score=0.0,
                    dense_score=result.dense_score,
                    sparse_score=result.sparse_score,
                    dense_rank=result.dense_rank,
                    sparse_rank=result.sparse_rank,
                    metadata=dict(result.metadata),
                )
            else:
                current = result_lookup[document_id]

                current.dense_score = max(
                    current.dense_score,
                    result.dense_score,
                )
                current.sparse_score = max(
                    current.sparse_score,
                    result.sparse_score,
                )

                if current.dense_rank is None:
                    current.dense_rank = result.dense_rank

                if current.sparse_rank is None:
                    current.sparse_rank = result.sparse_rank

    fused_results: list[RetrievalResult] = []

    for document_id, score in fused_scores.items():
        result = result_lookup[document_id]
        result.score = score
        fused_results.append(result)

    fused_results.sort(
        key=lambda result: result.score,
        reverse=True,
    )

    for rank, result in enumerate(fused_results, start=1):
        result.fused_rank = rank

    if top_k is not None:
        if top_k <= 0:
            return []

        fused_results = fused_results[:top_k]

    return fused_results


# ============================================================================
# Hybrid retriever
# ============================================================================


class HybridRetriever:
    """
    Combines dense and sparse retrieval.

    Default strategy:
        Dense retrieval + BM25 + Reciprocal Rank Fusion

    Example:
        retriever = HybridRetriever(
            embedding_service=embedding_service,
            documents=documents,
        )

        response = retriever.search(
            "What was the quarterly revenue?",
            top_k=5,
        )
    """

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        documents: Sequence[RetrievalDocument] | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        default_candidate_k: int = 50,
        score_strategy: str = "rrf",
    ) -> None:
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError(
                "dense_weight and sparse_weight cannot be negative."
            )

        if dense_weight == 0 and sparse_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than zero."
            )

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if default_candidate_k <= 0:
            raise ValueError(
                "default_candidate_k must be greater than zero."
            )

        if score_strategy not in {"rrf", "weighted"}:
            raise ValueError(
                "score_strategy must be either 'rrf' or 'weighted'."
            )

        self.embedding_service = embedding_service
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.default_candidate_k = default_candidate_k
        self.score_strategy = score_strategy

        self.documents: list[RetrievalDocument] = []

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

        if documents:
            self.add_documents(documents)

    def add_documents(
        self,
        documents: Iterable[RetrievalDocument],
        *,
        generate_embeddings: bool = True,
    ) -> None:
        """
        Add documents to the hybrid retriever.

        If generate_embeddings is True, missing embeddings are generated
        using the configured embedding service.
        """

        incoming_documents = list(documents)

        if not incoming_documents:
            return

        if generate_embeddings:
            missing_documents = [
                document
                for document in incoming_documents
                if document.embedding is None
            ]

            if missing_documents:
                embeddings = (
                    self.embedding_service.embed_documents(
                        [
                            document.text
                            for document in missing_documents
                        ]
                    )
                )

                for document, embedding in zip(
                    missing_documents,
                    embeddings,
                ):
                    document.embedding = embedding

        self.documents.extend(incoming_documents)

        self._rebuild_retrievers()

    def _rebuild_retrievers(self) -> None:
        """
        Rebuild default in-memory retrievers.
        """

        self.dense_retriever = InMemoryDenseRetriever(
            self.documents
        )

        self.sparse_retriever = InMemoryBM25(
            self.documents
        )

    def clear(self) -> None:
        """
        Clear all indexed documents.
        """

        self.documents.clear()
        self.dense_retriever = InMemoryDenseRetriever()
        self.sparse_retriever = InMemoryBM25()

    def _dense_search(
        self,
        query: str,
        *,
        candidate_k: int,
        filters: Mapping[str, Any] | None,
    ) -> list[RetrievalResult]:
        if self.dense_retriever is None:
            return []

        query_vector = self.embedding_service.embed_query(query)

        return list(
            self.dense_retriever.search(
                query_vector,
                top_k=candidate_k,
                filters=filters,
            )
        )

    def _sparse_search(
        self,
        query: str,
        *,
        candidate_k: int,
        filters: Mapping[str, Any] | None,
    ) -> list[RetrievalResult]:
        if self.sparse_retriever is None:
            return []

        return list(
            self.sparse_retriever.search(
                query,
                top_k=candidate_k,
                filters=filters,
            )
        )

    def _weighted_merge(
        self,
        dense_results: Sequence[RetrievalResult],
        sparse_results: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Merge results using normalized dense and sparse scores.
        """

        dense_scores = min_max_normalize(
            [result.dense_score for result in dense_results]
        )
        sparse_scores = min_max_normalize(
            [result.sparse_score for result in sparse_results]
        )

        merged: dict[str, RetrievalResult] = {}

        for result, normalized_score in zip(
            dense_results,
            dense_scores,
        ):
            merged[result.document_id] = RetrievalResult(
                document_id=result.document_id,
                text=result.text,
                score=self.dense_weight * normalized_score,
                dense_score=result.dense_score,
                sparse_score=0.0,
                dense_rank=result.dense_rank,
                metadata=dict(result.metadata),
            )

        for result, normalized_score in zip(
            sparse_results,
            sparse_scores,
        ):
            weighted_score = self.sparse_weight * normalized_score

            if result.document_id in merged:
                merged_result = merged[result.document_id]
                merged_result.score += weighted_score
                merged_result.sparse_score = result.sparse_score
                merged_result.sparse_rank = result.sparse_rank
            else:
                merged[result.document_id] = RetrievalResult(
                    document_id=result.document_id,
                    text=result.text,
                    score=weighted_score,
                    dense_score=0.0,
                    sparse_score=result.sparse_score,
                    sparse_rank=result.sparse_rank,
                    metadata=dict(result.metadata),
                )

        results = list(merged.values())

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        for rank, result in enumerate(results, start=1):
            result.fused_rank = rank

        return results[:top_k]

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> RetrievalResponse:
        """
        Execute hybrid retrieval.
        """

        query = str(query or "").strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            return RetrievalResponse(
                query=query,
                results=[],
                total_candidates=0,
                dense_candidates=0,
                sparse_candidates=0,
                filters=dict(filters or {}),
                strategy=f"hybrid_{self.score_strategy}",
            )

        candidate_k = candidate_k or self.default_candidate_k

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        dense_results = self._dense_search(
            query,
            candidate_k=candidate_k,
            filters=filters,
        )

        sparse_results = self._sparse_search(
            query,
            candidate_k=candidate_k,
            filters=filters,
        )

        if self.score_strategy == "weighted":
            final_results = self._weighted_merge(
                dense_results,
                sparse_results,
                top_k=top_k,
            )
        else:
            final_results = reciprocal_rank_fusion(
                [dense_results, sparse_results],
                k=self.rrf_k,
                top_k=top_k,
            )

        candidate_ids = {
            result.document_id
            for result in dense_results
        }

        candidate_ids.update(
            result.document_id
            for result in sparse_results
        )

        return RetrievalResponse(
            query=query,
            results=final_results,
            total_candidates=len(candidate_ids),
            dense_candidates=len(dense_results),
            sparse_candidates=len(sparse_results),
            filters=dict(filters or {}),
            strategy=f"hybrid_{self.score_strategy}",
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Convenience method returning only result documents.
        """

        return self.search(
            query,
            top_k=top_k,
            candidate_k=candidate_k,
            filters=filters,
        ).results


# ============================================================================
# Factory function
# ============================================================================


def create_hybrid_retriever(
    *,
    embedding_service: EmbeddingService,
    documents: Sequence[RetrievalDocument] | None = None,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    score_strategy: str = "rrf",
    rrf_k: int = 60,
    candidate_k: int = 50,
) -> HybridRetriever:
    """
    Create a configured hybrid retriever.
    """

    return HybridRetriever(
        embedding_service=embedding_service,
        documents=documents,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        score_strategy=score_strategy,
        rrf_k=rrf_k,
        default_candidate_k=candidate_k,
    )


__all__ = [
    "RetrievalError",
    "RetrievalConfigurationError",
    "RetrievalDocument",
    "RetrievalResult",
    "RetrievalResponse",
    "SparseRetriever",
    "DenseRetriever",
    "InMemoryBM25",
    "InMemoryDenseRetriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "create_hybrid_retriever",
    "tokenize",
    "normalize_score",
    "min_max_normalize",
]