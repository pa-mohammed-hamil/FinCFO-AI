"""
FinCo AI - Retrieval Service
============================

Application-level retrieval orchestration for FinCo AI.

Responsibilities
----------------
- Rewrite and normalize user queries.
- Apply tenant and metadata filters.
- Execute hybrid retrieval.
- Apply reranking.
- Compress retrieved context.
- Generate citation-ready evidence.
- Return a consistent retrieval response.
- Support synchronous and asynchronous workflows.
- Provide observability-friendly retrieval metadata.

This module is intentionally provider-agnostic. It works with the
retrieval, reranking, query rewriting, compression, and citation
components already defined in the RAG package.

Suggested pipeline
------------------
User Query
    |
    v
Query Rewriter
    |
    v
Metadata Filter
    |
    v
Hybrid Retriever
    |
    v
Reranker
    |
    v
Context Compressor
    |
    v
Citation Generator
    |
    v
Retrieval Service Response
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Protocol, Sequence

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class RetrievalServiceError(RuntimeError):
    """Base exception for retrieval-service failures."""


class RetrievalConfigurationError(RetrievalServiceError):
    """Raised when the retrieval service is incorrectly configured."""


class RetrievalExecutionError(RetrievalServiceError):
    """Raised when retrieval execution fails."""


class RetrievalValidationError(RetrievalServiceError):
    """Raised when an invalid retrieval request is supplied."""


# ============================================================================
# Utility Functions
# ============================================================================


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def normalize_text(value: Any) -> str:
    """
    Convert a value into normalized text.

    Parameters
    ----------
    value:
        Any value that can be represented as text.

    Returns
    -------
    str
        Whitespace-normalized text.
    """
    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def stable_hash(*values: Any, length: int = 16) -> str:
    """
    Generate a deterministic hash from multiple values.
    """
    payload = "||".join(normalize_text(value) for value in values)

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:length]


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to float safely.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Convert a value to int safely.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Clamp a numeric value to a range.
    """
    return max(minimum, min(maximum, value))


def deduplicate_strings(values: Iterable[str]) -> list[str]:
    """
    Deduplicate strings while preserving order.
    """
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        normalized = normalize_text(value)

        if not normalized:
            continue

        key = normalized.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(normalized)

    return result


# ============================================================================
# Protocols
# ============================================================================


class QueryRewriterProtocol(Protocol):
    """
    Protocol for query-rewriting components.

    Supported implementations may return:
    - a string,
    - a mapping,
    - an object with a rewritten_query attribute.
    """

    def rewrite(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        ...


class RetrieverProtocol(Protocol):
    """
    Protocol for retrievers.

    The implementation may be:
    - HybridRetriever,
    - vector retriever,
    - BM25 retriever,
    - external search retriever.
    """

    def retrieve(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        ...


class RerankingServiceProtocol(Protocol):
    """
    Protocol for reranking services.
    """

    def rerank(
        self,
        query: str,
        results: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        ...


class ContextCompressorProtocol(Protocol):
    """
    Protocol for context-compression components.
    """

    def compress(
        self,
        query: str,
        chunks: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        ...


class CitationGeneratorProtocol(Protocol):
    """
    Protocol for citation generators.
    """

    def generate(
        self,
        query: str,
        sources: Sequence[Any],
        **kwargs: Any,
    ) -> Any:
        ...


class MetadataFilterProtocol(Protocol):
    """
    Protocol for metadata filters.
    """

    def apply(
        self,
        metadata: Mapping[str, Any],
    ) -> bool:
        ...


# ============================================================================
# Request Models
# ============================================================================


@dataclass(slots=True)
class RetrievalRequest:
    """
    Request object for the retrieval service.
    """

    query: str

    tenant_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    top_k: int = 8
    retrieval_k: int | None = None
    rerank_k: int | None = None

    filters: Mapping[str, Any] = field(default_factory=dict)
    metadata_filter: Any | None = None

    use_query_rewriting: bool = True
    use_reranking: bool = True
    use_compression: bool = True
    use_citations: bool = True

    include_metadata: bool = True
    include_scores: bool = True

    max_context_chars: int | None = None
    max_context_tokens: int | None = None

    retrieval_kwargs: Mapping[str, Any] = field(default_factory=dict)
    reranking_kwargs: Mapping[str, Any] = field(default_factory=dict)
    compression_kwargs: Mapping[str, Any] = field(default_factory=dict)
    citation_kwargs: Mapping[str, Any] = field(default_factory=dict)

    trace_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate the request.
        """
        if not normalize_text(self.query):
            raise RetrievalValidationError(
                "Retrieval query cannot be empty."
            )

        if self.top_k <= 0:
            raise RetrievalValidationError(
                "top_k must be greater than zero."
            )

        if self.retrieval_k is not None and self.retrieval_k <= 0:
            raise RetrievalValidationError(
                "retrieval_k must be greater than zero."
            )

        if self.rerank_k is not None and self.rerank_k <= 0:
            raise RetrievalValidationError(
                "rerank_k must be greater than zero."
            )

        if (
            self.max_context_chars is not None
            and self.max_context_chars <= 0
        ):
            raise RetrievalValidationError(
                "max_context_chars must be greater than zero."
            )

        if (
            self.max_context_tokens is not None
            and self.max_context_tokens <= 0
        ):
            raise RetrievalValidationError(
                "max_context_tokens must be greater than zero."
            )

        if self.tenant_id is not None and not normalize_text(self.tenant_id):
            raise RetrievalValidationError(
                "tenant_id cannot be blank."
            )

    @property
    def effective_retrieval_k(self) -> int:
        """
        Return the number of candidates to retrieve before reranking.
        """
        return self.retrieval_k or max(self.top_k * 4, 20)

    @property
    def effective_rerank_k(self) -> int:
        """
        Return the number of candidates to retain after reranking.
        """
        return self.rerank_k or max(self.top_k * 2, 10)


# ============================================================================
# Result Models
# ============================================================================


@dataclass(slots=True)
class RetrievedItem:
    """
    Normalized representation of one retrieved item.
    """

    item_id: str
    text: str

    score: float = 0.0
    original_score: float | None = None
    rerank_score: float | None = None

    rank: int = 0
    source: str | None = None
    document_id: str | None = None
    chunk_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
    citation: Any | None = None

    retrieval_method: str | None = None
    explanation: str | None = None

    def to_dict(
        self,
        include_metadata: bool = True,
        include_scores: bool = True,
    ) -> dict[str, Any]:
        """
        Convert the item into a serializable dictionary.
        """
        result: dict[str, Any] = {
            "item_id": self.item_id,
            "text": self.text,
            "rank": self.rank,
            "source": self.source,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "retrieval_method": self.retrieval_method,
        }

        if include_scores:
            result.update(
                {
                    "score": self.score,
                    "original_score": self.original_score,
                    "rerank_score": self.rerank_score,
                }
            )

        if include_metadata:
            result["metadata"] = dict(self.metadata)

        if self.citation is not None:
            result["citation"] = self.citation

        if self.explanation:
            result["explanation"] = self.explanation

        return result


@dataclass(slots=True)
class QueryRewriteResult:
    """
    Normalized query-rewriting result.
    """

    original_query: str
    rewritten_query: str

    alternative_queries: list[str] = field(default_factory=list)
    expanded_terms: list[str] = field(default_factory=list)
    detected_entities: list[str] = field(default_factory=list)

    was_rewritten: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_queries(self) -> list[str]:
        """
        Return the primary query plus unique alternatives.
        """
        return deduplicate_strings(
            [
                self.rewritten_query,
                *self.alternative_queries,
            ]
        )


@dataclass(slots=True)
class RetrievalMetrics:
    """
    Timing and count metrics for one retrieval request.
    """

    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None

    rewrite_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    compression_ms: float = 0.0
    citation_ms: float = 0.0
    total_ms: float = 0.0

    retrieved_count: int = 0
    reranked_count: int = 0
    compressed_count: int = 0
    citation_count: int = 0

    cache_hit: bool = False
    errors: list[str] = field(default_factory=list)

    def finish(self) -> None:
        """
        Mark metrics as completed.
        """
        self.completed_at = utc_now()

        self.total_ms = (
            self.completed_at - self.started_at
        ).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metrics into a serializable dictionary.
        """
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at
                else None
            ),
            "rewrite_ms": round(self.rewrite_ms, 3),
            "retrieval_ms": round(self.retrieval_ms, 3),
            "reranking_ms": round(self.reranking_ms, 3),
            "compression_ms": round(self.compression_ms, 3),
            "citation_ms": round(self.citation_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "retrieved_count": self.retrieved_count,
            "reranked_count": self.reranked_count,
            "compressed_count": self.compressed_count,
            "citation_count": self.citation_count,
            "cache_hit": self.cache_hit,
            "errors": list(self.errors),
        }


@dataclass(slots=True)
class RetrievalResponse:
    """
    Final response returned by the retrieval service.
    """

    request_id: str
    query: str
    rewritten_query: str

    items: list[RetrievedItem] = field(default_factory=list)
    context: str = ""

    citations: list[Any] = field(default_factory=list)
    query_rewrite: QueryRewriteResult | None = None

    metrics: RetrievalMetrics | None = None

    success: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_results(self) -> bool:
        """
        Return whether any items were retrieved.
        """
        return bool(self.items)

    @property
    def result_count(self) -> int:
        """
        Return the number of final results.
        """
        return len(self.items)

    def to_dict(
        self,
        include_metadata: bool = True,
        include_scores: bool = True,
    ) -> dict[str, Any]:
        """
        Convert the response into a serializable dictionary.
        """
        result: dict[str, Any] = {
            "request_id": self.request_id,
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "items": [
                item.to_dict(
                    include_metadata=include_metadata,
                    include_scores=include_scores,
                )
                for item in self.items
            ],
            "context": self.context,
            "citations": self.citations,
            "success": self.success,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

        if self.query_rewrite:
            result["query_rewrite"] = {
                "original_query": self.query_rewrite.original_query,
                "rewritten_query": self.query_rewrite.rewritten_query,
                "alternative_queries": (
                    self.query_rewrite.alternative_queries
                ),
                "expanded_terms": self.query_rewrite.expanded_terms,
                "detected_entities": (
                    self.query_rewrite.detected_entities
                ),
                "was_rewritten": self.query_rewrite.was_rewritten,
                "metadata": self.query_rewrite.metadata,
            }

        if self.metrics:
            result["metrics"] = self.metrics.to_dict()

        if include_metadata:
            result["metadata"] = dict(self.metadata)

        return result


# ============================================================================
# Configuration
# ============================================================================


@dataclass(slots=True)
class RetrievalServiceConfig:
    """
    Configuration for RetrievalService.
    """

    default_top_k: int = 8
    default_retrieval_k: int = 32
    default_rerank_k: int = 16

    enable_query_rewriting: bool = True
    enable_reranking: bool = True
    enable_compression: bool = True
    enable_citations: bool = True

    strict_tenant_isolation: bool = True
    fail_on_reranking_error: bool = False
    fail_on_compression_error: bool = False
    fail_on_citation_error: bool = False

    max_query_length: int = 4_000
    max_result_text_length: int = 20_000

    include_debug_metadata: bool = False

    def validate(self) -> None:
        """
        Validate service configuration.
        """
        if self.default_top_k <= 0:
            raise RetrievalConfigurationError(
                "default_top_k must be greater than zero."
            )

        if self.default_retrieval_k <= 0:
            raise RetrievalConfigurationError(
                "default_retrieval_k must be greater than zero."
            )

        if self.default_rerank_k <= 0:
            raise RetrievalConfigurationError(
                "default_rerank_k must be greater than zero."
            )

        if self.max_query_length <= 0:
            raise RetrievalConfigurationError(
                "max_query_length must be greater than zero."
            )

        if self.max_result_text_length <= 0:
            raise RetrievalConfigurationError(
                "max_result_text_length must be greater than zero."
            )


# ============================================================================
# Result Normalization
# ============================================================================


def _read_value(
    value: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Read a value from either an object or mapping.
    """
    if value is None:
        return default

    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]

    for name in names:
        if hasattr(value, name):
            return getattr(value, name)

    return default


def _extract_metadata(value: Any) -> dict[str, Any]:
    """
    Extract metadata from an arbitrary result object.
    """
    metadata = _read_value(
        value,
        "metadata",
        "meta",
        default={},
    )

    if isinstance(metadata, Mapping):
        return dict(metadata)

    return {}


def normalize_retrieved_item(
    value: Any,
    rank: int,
    max_text_length: int = 20_000,
) -> RetrievedItem:
    """
    Convert a retriever/reranker result into RetrievedItem.
    """
    text = normalize_text(
        _read_value(
            value,
            "text",
            "content",
            "page_content",
            "document",
            default="",
        )
    )

    if len(text) > max_text_length:
        text = text[:max_text_length].rstrip() + "..."

    metadata = _extract_metadata(value)

    item_id = normalize_text(
        _read_value(
            value,
            "item_id",
            "id",
            "chunk_id",
            "document_id",
            default="",
        )
    )

    if not item_id:
        item_id = stable_hash(
            text,
            metadata.get("source"),
            metadata.get("document_id"),
            metadata.get("chunk_id"),
        )

    score = safe_float(
        _read_value(
            value,
            "score",
            "similarity",
            "relevance_score",
            default=0.0,
        )
    )

    original_score = _read_value(
        value,
        "original_score",
        "retrieval_score",
        default=None,
    )

    rerank_score = _read_value(
        value,
        "rerank_score",
        "reranking_score",
        default=None,
    )

    source = _read_value(
        value,
        "source",
        "source_name",
        default=metadata.get("source"),
    )

    document_id = _read_value(
        value,
        "document_id",
        "doc_id",
        default=metadata.get("document_id"),
    )

    chunk_id = _read_value(
        value,
        "chunk_id",
        default=metadata.get("chunk_id"),
    )

    retrieval_method = _read_value(
        value,
        "retrieval_method",
        "method",
        default=None,
    )

    explanation = _read_value(
        value,
        "explanation",
        "reason",
        default=None,
    )

    citation = _read_value(
        value,
        "citation",
        "citation_text",
        default=None,
    )

    return RetrievedItem(
        item_id=item_id,
        text=text,
        score=score,
        original_score=(
            safe_float(original_score)
            if original_score is not None
            else None
        ),
        rerank_score=(
            safe_float(rerank_score)
            if rerank_score is not None
            else None
        ),
        rank=rank,
        source=str(source) if source is not None else None,
        document_id=(
            str(document_id)
            if document_id is not None
            else None
        ),
        chunk_id=(
            str(chunk_id)
            if chunk_id is not None
            else None
        ),
        metadata=metadata,
        citation=citation,
        retrieval_method=(
            str(retrieval_method)
            if retrieval_method is not None
            else None
        ),
        explanation=(
            str(explanation)
            if explanation is not None
            else None
        ),
    )


def normalize_result_collection(
    results: Any,
    max_text_length: int = 20_000,
) -> list[RetrievedItem]:
    """
    Normalize a collection returned by a retriever or reranker.
    """
    if results is None:
        return []

    if isinstance(results, Mapping):
        for key in (
            "results",
            "items",
            "documents",
            "matches",
            "candidates",
        ):
            if key in results:
                results = results[key]
                break

    if hasattr(results, "results"):
        results = getattr(results, "results")

    if hasattr(results, "items") and not isinstance(results, Mapping):
        possible_items = getattr(results, "items")

        if callable(possible_items):
            pass
        else:
            results = possible_items

    if isinstance(results, (str, bytes)):
        results = [results]

    try:
        values = list(results)
    except TypeError:
        values = [results]

    normalized: list[RetrievedItem] = []

    for index, value in enumerate(values, start=1):
        normalized.append(
            normalize_retrieved_item(
                value,
                rank=index,
                max_text_length=max_text_length,
            )
        )

    return normalized


# ============================================================================
# Query Rewriting
# ============================================================================


class DefaultQueryRewriter:
    """
    Dependency-free fallback query rewriter.

    This implementation performs only normalization and does not
    call an external language model.
    """

    def rewrite(
        self,
        query: str,
        **kwargs: Any,
    ) -> QueryRewriteResult:
        """
        Normalize a query without changing its meaning.
        """
        original_query = normalize_text(query)
        rewritten_query = original_query

        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten_query,
            was_rewritten=original_query != rewritten_query,
        )


def normalize_query_rewrite_result(
    original_query: str,
    value: Any,
) -> QueryRewriteResult:
    """
    Normalize different query-rewriter output formats.
    """
    original_query = normalize_text(original_query)

    if isinstance(value, QueryRewriteResult):
        return value

    if isinstance(value, str):
        rewritten = normalize_text(value)

        return QueryRewriteResult(
            original_query=original_query,
            rewritten_query=rewritten or original_query,
            was_rewritten=(
                rewritten.casefold() != original_query.casefold()
            ),
        )

    rewritten = normalize_text(
        _read_value(
            value,
            "rewritten_query",
            "query",
            "text",
            default=original_query,
        )
    )

    alternatives = _read_value(
        value,
        "alternative_queries",
        "alternatives",
        "queries",
        default=[],
    )

    expanded_terms = _read_value(
        value,
        "expanded_terms",
        "expansions",
        default=[],
    )

    entities = _read_value(
        value,
        "detected_entities",
        "entities",
        default=[],
    )

    return QueryRewriteResult(
        original_query=original_query,
        rewritten_query=rewritten or original_query,
        alternative_queries=list(alternatives or []),
        expanded_terms=list(expanded_terms or []),
        detected_entities=list(entities or []),
        was_rewritten=(
            rewritten.casefold() != original_query.casefold()
        ),
        metadata=dict(
            _read_value(
                value,
                "metadata",
                default={},
            )
            or {}
        ),
    )


# ============================================================================
# Tenant Isolation
# ============================================================================


def apply_tenant_isolation(
    items: Sequence[RetrievedItem],
    tenant_id: str | None,
    strict: bool = True,
) -> list[RetrievedItem]:
    """
    Enforce tenant isolation on normalized retrieval items.

    Accepted tenant metadata keys:
    - tenant_id
    - tenant
    - organization_id
    - org_id

    If strict is True, items without tenant metadata are excluded
    when a tenant_id is provided.
    """
    if not tenant_id:
        return list(items)

    expected_tenant = normalize_text(tenant_id)

    filtered: list[RetrievedItem] = []

    for item in items:
        actual_tenant = (
            item.metadata.get("tenant_id")
            or item.metadata.get("tenant")
            or item.metadata.get("organization_id")
            or item.metadata.get("org_id")
        )

        if actual_tenant is None:
            if strict:
                continue

            filtered.append(item)
            continue

        if normalize_text(actual_tenant) == expected_tenant:
            filtered.append(item)

    for index, item in enumerate(filtered, start=1):
        item.rank = index

    return filtered


def apply_basic_metadata_filters(
    items: Sequence[RetrievedItem],
    filters: Mapping[str, Any],
) -> list[RetrievedItem]:
    """
    Apply simple equality-based metadata filters.

    Supported forms
    ---------------
    {"account_id": "ACC-001"}

    {"department": ["finance", "audit"]}

    {"year": 2025}
    """
    if not filters:
        return list(items)

    filtered: list[RetrievedItem] = []

    for item in items:
        matches = True

        for key, expected in filters.items():
            actual = item.metadata.get(key)

            if isinstance(expected, (list, tuple, set, frozenset)):
                if actual not in expected:
                    matches = False
                    break
            elif actual != expected:
                matches = False
                break

        if matches:
            filtered.append(item)

    for index, item in enumerate(filtered, start=1):
        item.rank = index

    return filtered


# ============================================================================
# Retrieval Service
# ============================================================================


class RetrievalService:
    """
    Main orchestration service for FinCo AI retrieval.

    Parameters
    ----------
    retriever:
        Hybrid or vector retriever.

    query_rewriter:
        Optional query-rewriting component.

    reranking_service:
        Optional reranking service.

    context_compressor:
        Optional context-compression component.

    citation_generator:
        Optional citation generator.

    config:
        Retrieval service configuration.
    """

    def __init__(
        self,
        retriever: RetrieverProtocol,
        query_rewriter: QueryRewriterProtocol | None = None,
        reranking_service: RerankingServiceProtocol | None = None,
        context_compressor: ContextCompressorProtocol | None = None,
        citation_generator: CitationGeneratorProtocol | None = None,
        config: RetrievalServiceConfig | None = None,
    ) -> None:
        if retriever is None:
            raise RetrievalConfigurationError(
                "A retriever is required."
            )

        self.retriever = retriever
        self.query_rewriter = (
            query_rewriter or DefaultQueryRewriter()
        )
        self.reranking_service = reranking_service
        self.context_compressor = context_compressor
        self.citation_generator = citation_generator

        self.config = config or RetrievalServiceConfig()
        self.config.validate()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def retrieve(
        self,
        request: RetrievalRequest | str,
        **kwargs: Any,
    ) -> RetrievalResponse:
        """
        Execute the complete retrieval pipeline.

        Parameters
        ----------
        request:
            RetrievalRequest or raw query string.

        Returns
        -------
        RetrievalResponse
        """
        normalized_request = self._normalize_request(
            request,
            **kwargs,
        )

        normalized_request.validate()

        if len(normalized_request.query) > self.config.max_query_length:
            raise RetrievalValidationError(
                "Query exceeds the configured maximum length."
            )

        metrics = RetrievalMetrics()
        started = time.perf_counter()

        rewrite_result: QueryRewriteResult | None = None
        final_items: list[RetrievedItem] = []
        context = ""
        citations: list[Any] = []

        warnings: list[str] = []
        errors: list[str] = []

        try:
            # -------------------------------------------------------------
            # 1. Query rewriting
            # -------------------------------------------------------------
            rewrite_start = time.perf_counter()

            if (
                normalized_request.use_query_rewriting
                and self.config.enable_query_rewriting
                and self.query_rewriter is not None
            ):
                rewrite_output = self.query_rewriter.rewrite(
                    normalized_request.query,
                    tenant_id=normalized_request.tenant_id,
                    user_id=normalized_request.user_id,
                    session_id=normalized_request.session_id,
                    request_id=normalized_request.request_id,
                )

                rewrite_result = normalize_query_rewrite_result(
                    normalized_request.query,
                    rewrite_output,
                )
            else:
                rewrite_result = QueryRewriteResult(
                    original_query=normalized_request.query,
                    rewritten_query=normalized_request.query,
                )

            metrics.rewrite_ms = (
                time.perf_counter() - rewrite_start
            ) * 1000

            # -------------------------------------------------------------
            # 2. Retrieval
            # -------------------------------------------------------------
            retrieval_start = time.perf_counter()

            retrieval_kwargs = dict(
                normalized_request.retrieval_kwargs
            )

            retrieval_kwargs.setdefault(
                "top_k",
                normalized_request.effective_retrieval_k,
            )

            retrieval_kwargs.setdefault(
                "k",
                normalized_request.effective_retrieval_k,
            )

            if normalized_request.filters:
                retrieval_kwargs.setdefault(
                    "filters",
                    normalized_request.filters,
                )

            if normalized_request.metadata_filter is not None:
                retrieval_kwargs.setdefault(
                    "metadata_filter",
                    normalized_request.metadata_filter,
                )

            if normalized_request.tenant_id:
                retrieval_kwargs.setdefault(
                    "tenant_id",
                    normalized_request.tenant_id,
                )

            raw_results = self._retrieve_with_query_variants(
                rewrite_result,
                retrieval_kwargs,
            )

            items = normalize_result_collection(
                raw_results,
                max_text_length=self.config.max_result_text_length,
            )

            metrics.retrieved_count = len(items)

            metrics.retrieval_ms = (
                time.perf_counter() - retrieval_start
            ) * 1000

            # -------------------------------------------------------------
            # 3. Tenant isolation and metadata filtering
            # -------------------------------------------------------------
            items = apply_tenant_isolation(
                items,
                tenant_id=normalized_request.tenant_id,
                strict=self.config.strict_tenant_isolation,
            )

            items = apply_basic_metadata_filters(
                items,
                filters=normalized_request.filters,
            )

            # -------------------------------------------------------------
            # 4. Reranking
            # -------------------------------------------------------------
            rerank_start = time.perf_counter()

            if (
                normalized_request.use_reranking
                and self.config.enable_reranking
                and self.reranking_service is not None
                and items
            ):
                rerank_input = items[
                    : normalized_request.effective_rerank_k
                ]

                rerank_output = self.reranking_service.rerank(
                    query=rewrite_result.rewritten_query,
                    results=rerank_input,
                    **dict(normalized_request.reranking_kwargs),
                )

                items = normalize_result_collection(
                    rerank_output,
                    max_text_length=self.config.max_result_text_length,
                )

                metrics.reranked_count = len(items)

            else:
                metrics.reranked_count = len(items)

            metrics.reranking_ms = (
                time.perf_counter() - rerank_start
            ) * 1000

            # -------------------------------------------------------------
            # 5. Final top-k selection
            # -------------------------------------------------------------
            final_items = self._select_final_items(
                items,
                top_k=normalized_request.top_k,
            )

            # -------------------------------------------------------------
            # 6. Context compression
            # -------------------------------------------------------------
            compression_start = time.perf_counter()

            if (
                normalized_request.use_compression
                and self.config.enable_compression
                and self.context_compressor is not None
                and final_items
            ):
                compression_output = self.context_compressor.compress(
                    query=rewrite_result.rewritten_query,
                    chunks=final_items,
                    max_chars=normalized_request.max_context_chars,
                    max_tokens=normalized_request.max_context_tokens,
                    **dict(normalized_request.compression_kwargs),
                )

                context = self._extract_context(
                    compression_output
                )

                compressed_items = self._extract_items(
                    compression_output
                )

                if compressed_items:
                    final_items = compressed_items

                metrics.compressed_count = len(final_items)

            else:
                context = self._build_default_context(final_items)
                metrics.compressed_count = len(final_items)

            metrics.compression_ms = (
                time.perf_counter() - compression_start
            ) * 1000

            # -------------------------------------------------------------
            # 7. Citation generation
            # -------------------------------------------------------------
            citation_start = time.perf_counter()

            if (
                normalized_request.use_citations
                and self.config.enable_citations
                and self.citation_generator is not None
                and final_items
            ):
                citation_output = self.citation_generator.generate(
                    query=rewrite_result.rewritten_query,
                    sources=final_items,
                    **dict(normalized_request.citation_kwargs),
                )

                citations = self._extract_citations(
                    citation_output
                )

                self._attach_citations(
                    final_items,
                    citations,
                )

                metrics.citation_count = len(citations)

            metrics.citation_ms = (
                time.perf_counter() - citation_start
            ) * 1000

        except Exception as exc:
            logger.exception(
                "Retrieval pipeline failed. request_id=%s",
                normalized_request.request_id,
            )

            error_message = str(exc) or exc.__class__.__name__
            errors.append(error_message)
            metrics.errors.append(error_message)

            if isinstance(exc, RetrievalServiceError):
                raise

            raise RetrievalExecutionError(
                "Retrieval pipeline execution failed."
            ) from exc

        finally:
            metrics.total_ms = (
                time.perf_counter() - started
            ) * 1000
            metrics.finish()

        response_metadata = {
            "tenant_id": normalized_request.tenant_id,
            "user_id": normalized_request.user_id,
            "session_id": normalized_request.session_id,
            "trace_id": normalized_request.trace_id,
        }

        if self.config.include_debug_metadata:
            response_metadata.update(
                {
                    "filters": dict(normalized_request.filters),
                    "retrieval_k": (
                        normalized_request.effective_retrieval_k
                    ),
                    "rerank_k": (
                        normalized_request.effective_rerank_k
                    ),
                    "retriever": self.retriever.__class__.__name__,
                    "reranker": (
                        self.reranking_service.__class__.__name__
                        if self.reranking_service
                        else None
                    ),
                    "compressor": (
                        self.context_compressor.__class__.__name__
                        if self.context_compressor
                        else None
                    ),
                }
            )

        return RetrievalResponse(
            request_id=normalized_request.request_id,
            query=normalized_request.query,
            rewritten_query=(
                rewrite_result.rewritten_query
                if rewrite_result
                else normalized_request.query
            ),
            items=final_items,
            context=context,
            citations=citations,
            query_rewrite=rewrite_result,
            metrics=metrics,
            success=not errors,
            warnings=warnings,
            errors=errors,
            metadata=response_metadata,
        )

    async def aretrieve(
        self,
        request: RetrievalRequest | str,
        **kwargs: Any,
    ) -> RetrievalResponse:
        """
        Execute retrieval using async-compatible dependencies when
        available.

        The method supports both:
        - async dependency methods,
        - synchronous fallback methods.
        """
        normalized_request = self._normalize_request(
            request,
            **kwargs,
        )

        normalized_request.validate()

        if hasattr(self.retriever, "aretrieve"):
            return await self._aretrieve_pipeline(normalized_request)

        return self.retrieve(normalized_request)

    # ---------------------------------------------------------------------
    # Request handling
    # ---------------------------------------------------------------------

    def _normalize_request(
        self,
        request: RetrievalRequest | str,
        **kwargs: Any,
    ) -> RetrievalRequest:
        """
        Normalize a raw query or request object.
        """
        if isinstance(request, RetrievalRequest):
            return request

        if isinstance(request, str):
            return RetrievalRequest(
                query=request,
                **kwargs,
            )

        raise RetrievalValidationError(
            "request must be a RetrievalRequest or string."
        )

    # ---------------------------------------------------------------------
    # Retrieval internals
    # ---------------------------------------------------------------------

    def _retrieve_with_query_variants(
        self,
        rewrite_result: QueryRewriteResult,
        retrieval_kwargs: Mapping[str, Any],
    ) -> Any:
        """
        Retrieve using the rewritten query.

        Alternative queries are used only when the retriever supports
        multiple-query retrieval or when the primary query returns no
        results.
        """
        queries = rewrite_result.all_queries

        if not queries:
            queries = [rewrite_result.original_query]

        primary_query = queries[0]

        try:
            primary_results = self.retriever.retrieve(
                primary_query,
                **dict(retrieval_kwargs),
            )
        except TypeError:
            primary_results = self.retriever.retrieve(
                query=primary_query,
                **dict(retrieval_kwargs),
            )

        primary_items = normalize_result_collection(primary_results)

        if primary_items or len(queries) == 1:
            return primary_results

        merged_items: list[RetrievedItem] = []

        for alternative_query in queries[1:]:
            try:
                alternative_results = self.retriever.retrieve(
                    alternative_query,
                    **dict(retrieval_kwargs),
                )
            except TypeError:
                alternative_results = self.retriever.retrieve(
                    query=alternative_query,
                    **dict(retrieval_kwargs),
                )

            merged_items.extend(
                normalize_result_collection(alternative_results)
            )

        return self._merge_items(
            [
                *primary_items,
                *merged_items,
            ]
        )

    def _merge_items(
        self,
        items: Sequence[RetrievedItem],
    ) -> list[RetrievedItem]:
        """
        Merge and deduplicate retrieved items.
        """
        merged: dict[str, RetrievedItem] = {}

        for item in items:
            existing = merged.get(item.item_id)

            if existing is None or item.score > existing.score:
                merged[item.item_id] = item

        result = sorted(
            merged.values(),
            key=lambda item: item.score,
            reverse=True,
        )

        for index, item in enumerate(result, start=1):
            item.rank = index

        return result

    def _select_final_items(
        self,
        items: Sequence[RetrievedItem],
        top_k: int,
    ) -> list[RetrievedItem]:
        """
        Select and rank the final result set.
        """
        deduplicated = self._merge_items(items)

        selected = deduplicated[:top_k]

        for index, item in enumerate(selected, start=1):
            item.rank = index

        return selected

    # ---------------------------------------------------------------------
    # Context handling
    # ---------------------------------------------------------------------

    def _build_default_context(
        self,
        items: Sequence[RetrievedItem],
    ) -> str:
        """
        Build context when no compressor is configured.
        """
        sections: list[str] = []

        for index, item in enumerate(items, start=1):
            source = item.source or item.document_id or "Unknown source"

            sections.append(
                f"[Source {index}: {source}]\n"
                f"{item.text}"
            )

        return "\n\n".join(sections)

    def _extract_context(
        self,
        value: Any,
    ) -> str:
        """
        Extract context from a compressor response.
        """
        if isinstance(value, str):
            return value

        context = _read_value(
            value,
            "context",
            "compressed_context",
            "text",
            "content",
            default="",
        )

        return normalize_text(context)

    def _extract_items(
        self,
        value: Any,
    ) -> list[RetrievedItem]:
        """
        Extract compressed items from a compressor response.
        """
        if isinstance(value, Mapping):
            candidates = value.get(
                "chunks",
                value.get(
                    "items",
                    value.get("results", []),
                ),
            )
        else:
            candidates = _read_value(
                value,
                "chunks",
                "items",
                "results",
                default=[],
            )

        if not candidates:
            return []

        return normalize_result_collection(
            candidates,
            max_text_length=self.config.max_result_text_length,
        )

    # ---------------------------------------------------------------------
    # Citation handling
    # ---------------------------------------------------------------------

    def _extract_citations(
        self,
        value: Any,
    ) -> list[Any]:
        """
        Extract citations from a citation-generator response.
        """
        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, Mapping):
            for key in (
                "citations",
                "sources",
                "references",
                "items",
            ):
                if key in value:
                    extracted = value[key]

                    if isinstance(extracted, list):
                        return extracted

                    if extracted is None:
                        return []

                    return [extracted]

        extracted = _read_value(
            value,
            "citations",
            "sources",
            "references",
            "items",
            default=None,
        )

        if extracted is None:
            return [value]

        if isinstance(extracted, list):
            return extracted

        return [extracted]

    def _attach_citations(
        self,
        items: Sequence[RetrievedItem],
        citations: Sequence[Any],
    ) -> None:
        """
        Attach citations to matching retrieved items where possible.
        """
        citation_by_item_id: dict[str, Any] = {}

        for citation in citations:
            item_id = _read_value(
                citation,
                "item_id",
                "chunk_id",
                "document_id",
                "source_id",
                default=None,
            )

            if item_id is not None:
                citation_by_item_id[str(item_id)] = citation

        for item in items:
            if item.item_id in citation_by_item_id:
                item.citation = citation_by_item_id[item.item_id]
                continue

            if item.chunk_id and item.chunk_id in citation_by_item_id:
                item.citation = citation_by_item_id[item.chunk_id]
                continue

            if item.document_id and item.document_id in citation_by_item_id:
                item.citation = citation_by_item_id[item.document_id]

    # ---------------------------------------------------------------------
    # Async internals
    # ---------------------------------------------------------------------

    async def _aretrieve_pipeline(
        self,
        request: RetrievalRequest,
    ) -> RetrievalResponse:
        """
        Async retrieval path.

        This path delegates to aretrieve() when available and otherwise
        falls back to synchronous execution.
        """
        metrics = RetrievalMetrics()
        started = time.perf_counter()

        rewrite_result: QueryRewriteResult

        if (
            request.use_query_rewriting
            and self.config.enable_query_rewriting
            and self.query_rewriter is not None
        ):
            if hasattr(self.query_rewriter, "arewrite"):
                rewrite_output = await self.query_rewriter.arewrite(
                    request.query,
                    tenant_id=request.tenant_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                )
            else:
                rewrite_output = self.query_rewriter.rewrite(
                    request.query,
                    tenant_id=request.tenant_id,
                    user_id=request.user_id,
                    session_id=request.session_id,
                    request_id=request.request_id,
                )

            rewrite_result = normalize_query_rewrite_result(
                request.query,
                rewrite_output,
            )
        else:
            rewrite_result = QueryRewriteResult(
                original_query=request.query,
                rewritten_query=request.query,
            )

        retrieval_kwargs = dict(request.retrieval_kwargs)
        retrieval_kwargs.setdefault(
            "top_k",
            request.effective_retrieval_k,
        )
        retrieval_kwargs.setdefault(
            "k",
            request.effective_retrieval_k,
        )

        if hasattr(self.retriever, "aretrieve"):
            raw_results = await self.retriever.aretrieve(
                rewrite_result.rewritten_query,
                **retrieval_kwargs,
            )
        else:
            raw_results = self.retriever.retrieve(
                rewrite_result.rewritten_query,
                **retrieval_kwargs,
            )

        items = normalize_result_collection(
            raw_results,
            max_text_length=self.config.max_result_text_length,
        )

        items = apply_tenant_isolation(
            items,
            tenant_id=request.tenant_id,
            strict=self.config.strict_tenant_isolation,
        )

        items = apply_basic_metadata_filters(
            items,
            filters=request.filters,
        )

        if (
            request.use_reranking
            and self.config.enable_reranking
            and self.reranking_service is not None
        ):
            if hasattr(self.reranking_service, "arerank"):
                reranked = await self.reranking_service.arerank(
                    query=rewrite_result.rewritten_query,
                    results=items,
                    **dict(request.reranking_kwargs),
                )
            else:
                reranked = self.reranking_service.rerank(
                    query=rewrite_result.rewritten_query,
                    results=items,
                    **dict(request.reranking_kwargs),
                )

            items = normalize_result_collection(
                reranked,
                max_text_length=self.config.max_result_text_length,
            )

        final_items = self._select_final_items(
            items,
            top_k=request.top_k,
        )

        context = self._build_default_context(final_items)
        citations: list[Any] = []

        if (
            request.use_compression
            and self.config.enable_compression
            and self.context_compressor is not None
        ):
            if hasattr(self.context_compressor, "acompress"):
                compressed = await self.context_compressor.acompress(
                    query=rewrite_result.rewritten_query,
                    chunks=final_items,
                    **dict(request.compression_kwargs),
                )
            else:
                compressed = self.context_compressor.compress(
                    query=rewrite_result.rewritten_query,
                    chunks=final_items,
                    **dict(request.compression_kwargs),
                )

            context = self._extract_context(compressed)

            compressed_items = self._extract_items(compressed)

            if compressed_items:
                final_items = compressed_items

        if (
            request.use_citations
            and self.config.enable_citations
            and self.citation_generator is not None
        ):
            if hasattr(self.citation_generator, "agenerate"):
                citation_output = (
                    await self.citation_generator.agenerate(
                        query=rewrite_result.rewritten_query,
                        sources=final_items,
                        **dict(request.citation_kwargs),
                    )
                )
            else:
                citation_output = self.citation_generator.generate(
                    query=rewrite_result.rewritten_query,
                    sources=final_items,
                    **dict(request.citation_kwargs),
                )

            citations = self._extract_citations(citation_output)

            self._attach_citations(
                final_items,
                citations,
            )

        metrics.retrieved_count = len(items)
        metrics.reranked_count = len(items)
        metrics.compressed_count = len(final_items)
        metrics.citation_count = len(citations)

        metrics.total_ms = (
            time.perf_counter() - started
        ) * 1000
        metrics.finish()

        return RetrievalResponse(
            request_id=request.request_id,
            query=request.query,
            rewritten_query=rewrite_result.rewritten_query,
            items=final_items,
            context=context,
            citations=citations,
            query_rewrite=rewrite_result,
            metrics=metrics,
            metadata={
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "trace_id": request.trace_id,
            },
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def create_retrieval_service(
    retriever: RetrieverProtocol,
    query_rewriter: QueryRewriterProtocol | None = None,
    reranking_service: RerankingServiceProtocol | None = None,
    context_compressor: ContextCompressorProtocol | None = None,
    citation_generator: CitationGeneratorProtocol | None = None,
    config: RetrievalServiceConfig | None = None,
) -> RetrievalService:
    """
    Create a configured RetrievalService.
    """
    return RetrievalService(
        retriever=retriever,
        query_rewriter=query_rewriter,
        reranking_service=reranking_service,
        context_compressor=context_compressor,
        citation_generator=citation_generator,
        config=config,
    )


def retrieve(
    service: RetrievalService,
    query: str,
    *,
    tenant_id: str | None = None,
    user_id: str | None = None,
    top_k: int = 8,
    filters: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> RetrievalResponse:
    """
    Convenience wrapper around RetrievalService.retrieve().
    """
    request = RetrievalRequest(
        query=query,
        tenant_id=tenant_id,
        user_id=user_id,
        top_k=top_k,
        filters=filters or {},
        **kwargs,
    )

    return service.retrieve(request)


# ============================================================================
# Public Exports
# ============================================================================


__all__ = [
    "RetrievalServiceError",
    "RetrievalConfigurationError",
    "RetrievalExecutionError",
    "RetrievalValidationError",
    "RetrievalRequest",
    "RetrievedItem",
    "QueryRewriteResult",
    "RetrievalMetrics",
    "RetrievalResponse",
    "RetrievalServiceConfig",
    "QueryRewriterProtocol",
    "RetrieverProtocol",
    "RerankingServiceProtocol",
    "ContextCompressorProtocol",
    "CitationGeneratorProtocol",
    "MetadataFilterProtocol",
    "DefaultQueryRewriter",
    "normalize_retrieved_item",
    "normalize_result_collection",
    "normalize_query_rewrite_result",
    "apply_tenant_isolation",
    "apply_basic_metadata_filters",
    "RetrievalService",
    "create_retrieval_service",
    "retrieve",
]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from backend.app.rag.hybrid_retriever import (
        HybridRetriever,
        RetrievalDocument,
    )

    documents = [
        RetrievalDocument(
            id="doc-001",
            text=(
                "Revenue increased by 18% in FY2025 compared with FY2024."
            ),
            metadata={
                "tenant_id": "tenant-001",
                "source": "annual_report.pdf",
                "document_id": "doc-001",
                "chunk_id": "chunk-001",
            },
        ),
        RetrievalDocument(
            id="doc-002",
            text=(
                "Operating expenses decreased by 6% during FY2025."
            ),
            metadata={
                "tenant_id": "tenant-001",
                "source": "financial_statement.pdf",
                "document_id": "doc-002",
                "chunk_id": "chunk-002",
            },
        ),
    ]

    retriever = HybridRetriever.from_documents(
        documents
    )

    service = create_retrieval_service(
        retriever=retriever,
        config=RetrievalServiceConfig(
            default_top_k=5,
            strict_tenant_isolation=True,
            include_debug_metadata=True,
        ),
    )

    response = retrieve(
        service,
        query="How did revenue change in FY2025?",
        tenant_id="tenant-001",
        top_k=5,
    )

    print(response.to_dict())