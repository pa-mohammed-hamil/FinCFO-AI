"""
Embedding utilities for FinCo AI.

Supports:
- Local sentence-transformers embeddings
- OpenAI-compatible embeddings
- Deterministic hash embeddings for development/testing
- Batch embedding
- Cosine similarity
- Embedding normalization
- Simple embedding cache

Recommended production usage:
    SentenceTransformerEmbeddingProvider
or:
    OpenAIEmbeddingProvider
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class EmbeddingError(RuntimeError):
    """Base exception for embedding-related failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when an embedding provider is incorrectly configured."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when vectors have incompatible dimensions."""


# ============================================================================
# Data models
# ============================================================================


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Result of embedding one or more texts.
    """

    embeddings: list[list[float]]
    model: str
    dimension: int
    normalized: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.embeddings)

    def first(self) -> list[float]:
        if not self.embeddings:
            raise EmbeddingError("Embedding result is empty.")

        return self.embeddings[0]


@dataclass(frozen=True)
class EmbeddingDocument:
    """
    Text document prepared for embedding.
    """

    text: str
    document_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimilarityResult:
    """
    Similarity result between two vectors.
    """

    score: float
    index: int | None = None
    document_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ============================================================================
# Utility functions
# ============================================================================


def clean_text(text: Any) -> str:
    """
    Convert input to clean, whitespace-normalized text.
    """

    if text is None:
        return ""

    value = str(text)
    return " ".join(value.split()).strip()


def validate_texts(
    texts: Iterable[str],
    *,
    allow_empty: bool = False,
) -> list[str]:
    """
    Validate and normalize a collection of texts.
    """

    normalized: list[str] = []

    for index, text in enumerate(texts):
        value = clean_text(text)

        if not value and not allow_empty:
            raise ValueError(f"Text at index {index} is empty.")

        normalized.append(value)

    return normalized


def vector_dimension(vector: Sequence[float]) -> int:
    """
    Return the dimension of a vector.
    """

    return len(vector)


def validate_vector(vector: Sequence[float]) -> list[float]:
    """
    Validate a vector and convert all values to floats.
    """

    if vector is None:
        raise ValueError("Vector cannot be None.")

    result = [float(value) for value in vector]

    if not result:
        raise ValueError("Vector cannot be empty.")

    for index, value in enumerate(result):
        if not math.isfinite(value):
            raise ValueError(
                f"Vector contains a non-finite value at index {index}: {value}"
            )

    return result


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """
    L2-normalize a vector.
    """

    values = validate_vector(vector)
    magnitude = math.sqrt(sum(value * value for value in values))

    if magnitude == 0:
        return [0.0 for _ in values]

    return [value / magnitude for value in values]


def normalize_embeddings(
    embeddings: Iterable[Sequence[float]],
) -> list[list[float]]:
    """
    Normalize a collection of vectors.
    """

    return [normalize_vector(vector) for vector in embeddings]


def dot_product(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """
    Calculate the dot product of two vectors.
    """

    a = validate_vector(vector_a)
    b = validate_vector(vector_b)

    if len(a) != len(b):
        raise EmbeddingDimensionError(
            f"Vector dimensions do not match: {len(a)} != {len(b)}"
        )

    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.

    Returns:
        Value between -1.0 and 1.0.
    """

    a = validate_vector(vector_a)
    b = validate_vector(vector_b)

    if len(a) != len(b):
        raise EmbeddingDimensionError(
            f"Vector dimensions do not match: {len(a)} != {len(b)}"
        )

    magnitude_a = math.sqrt(sum(value * value for value in a))
    magnitude_b = math.sqrt(sum(value * value for value in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    similarity = dot_product(a, b) / (magnitude_a * magnitude_b)

    # Protect against small floating-point errors.
    return max(-1.0, min(1.0, similarity))


def euclidean_distance(
    vector_a: Sequence[float],
    vector_b: Sequence[float],
) -> float:
    """
    Calculate Euclidean distance between two vectors.
    """

    a = validate_vector(vector_a)
    b = validate_vector(vector_b)

    if len(a) != len(b):
        raise EmbeddingDimensionError(
            f"Vector dimensions do not match: {len(a)} != {len(b)}"
        )

    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def stable_text_hash(text: str) -> str:
    """
    Generate a stable SHA-256 hash for text.
    """

    return hashlib.sha256(clean_text(text).encode("utf-8")).hexdigest()


# ============================================================================
# Provider interface
# ============================================================================


class EmbeddingProvider(ABC):
    """
    Abstract embedding provider.

    Every provider must implement:
    - embed_text
    - embed_documents
    - dimension
    - model_name
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the embedding model name."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding vector dimension."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for one text."""

    @abstractmethod
    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

    def embed_query(self, query: str) -> list[float]:
        """
        Generate an embedding for a search query.

        Providers may override this when query and document embedding
        instructions differ.
        """

        return self.embed_text(query)

    def embed_result(
        self,
        texts: Sequence[str],
        *,
        normalized: bool = True,
    ) -> EmbeddingResult:
        """
        Generate a structured embedding result.
        """

        embeddings = self.embed_documents(texts)

        if normalized:
            embeddings = normalize_embeddings(embeddings)

        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model_name,
            dimension=self.dimension,
            normalized=normalized,
        )


# ============================================================================
# Hash embedding provider
# ============================================================================


class HashEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic, dependency-free embedding provider.

    This is useful for:
    - Unit tests
    - Local development
    - Offline development
    - Pipeline integration tests

    It is NOT recommended for semantic production search because it does
    not understand language semantics.
    """

    def __init__(
        self,
        dimension: int = 384,
        *,
        model: str = "hash-embedding-v1",
        normalize: bool = True,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")

        self._dimension = dimension
        self._model = model
        self._normalize = normalize

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        """
        Generate a deterministic pseudo-embedding.

        Each token contributes to multiple dimensions using a stable hash.
        """

        value = clean_text(text)

        if not value:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension
        tokens = value.lower().split()

        for token_index, token in enumerate(tokens):
            digest = hashlib.sha256(
                f"{token_index}:{token}".encode("utf-8")
            ).digest()

            for dimension_index in range(self.dimension):
                byte_index = dimension_index % len(digest)
                raw_value = digest[byte_index]

                contribution = (raw_value / 127.5) - 1.0

                # Slightly reduce the effect of later tokens.
                weight = 1.0 / math.sqrt(token_index + 1)

                vector[dimension_index] += contribution * weight

        if self._normalize:
            return normalize_vector(vector)

        return vector

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [self._embed(text) for text in texts]


# ============================================================================
# Sentence Transformers provider
# ============================================================================


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using sentence-transformers.

    Install:
        pip install sentence-transformers

    Example:
        provider = SentenceTransformerEmbeddingProvider(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingConfigurationError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }

        if device:
            model_kwargs["device"] = device

        logger.info(
            "Loading sentence-transformer model: %s",
            model_name,
        )

        self._model = SentenceTransformer(
            model_name,
            **model_kwargs,
        )

        dimension = self._model.get_sentence_embedding_dimension()

        if dimension is None:
            raise EmbeddingConfigurationError(
                "Unable to determine sentence-transformer embedding dimension."
            )

        self._dimension = int(dimension)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        value = clean_text(text)

        if not value:
            return [0.0] * self.dimension

        embeddings = self._model.encode(
            [value],
            batch_size=1,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        vector = embeddings[0].tolist()

        if self._normalize:
            return normalize_vector(vector)

        return validate_vector(vector)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        values = validate_texts(texts)

        if not values:
            return []

        embeddings = self._model.encode(
            values,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        result = [row.tolist() for row in embeddings]

        if self._normalize:
            return normalize_embeddings(result)

        return [validate_vector(row) for row in result]


# ============================================================================
# OpenAI-compatible provider
# ============================================================================


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI-compatible embedding provider.

    Install:
        pip install openai

    Environment:
        OPENAI_API_KEY=...
        OPENAI_EMBEDDING_MODEL=text-embedding-3-small

    Supports compatible gateways through:
        OPENAI_BASE_URL=https://your-compatible-endpoint/v1
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 64,
        dimensions: int | None = None,
        normalize: bool = True,
        timeout: float | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingConfigurationError(
                "openai is not installed. "
                "Install it with: pip install openai"
            ) from exc

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not resolved_api_key:
            raise EmbeddingConfigurationError(
                "Missing OpenAI API key. Set OPENAI_API_KEY or pass api_key."
            )

        self._model_name = (
            model_name
            or os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-small",
            )
        )
        self._batch_size = batch_size
        self._dimensions = dimensions
        self._normalize = normalize

        client_kwargs: dict[str, Any] = {
            "api_key": resolved_api_key,
        }

        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        if timeout is not None:
            client_kwargs["timeout"] = timeout

        self._client = OpenAI(**client_kwargs)

        # OpenAI embedding dimensions can be configured for some models.
        # We determine the actual dimension on the first request.
        self._dimension: int | None = dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise EmbeddingConfigurationError(
                "Embedding dimension is unknown until the first embedding "
                "request is completed."
            )

        return self._dimension

    def _request_embeddings(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        request: dict[str, Any] = {
            "model": self._model_name,
            "input": list(texts),
        }

        if self._dimensions is not None:
            request["dimensions"] = self._dimensions

        response = self._client.embeddings.create(**request)

        ordered_data = sorted(
            response.data,
            key=lambda item: item.index,
        )

        embeddings = [
            validate_vector(item.embedding)
            for item in ordered_data
        ]

        if embeddings:
            actual_dimension = len(embeddings[0])

            if self._dimension is None:
                self._dimension = actual_dimension
            elif self._dimension != actual_dimension:
                raise EmbeddingDimensionError(
                    "Embedding provider returned an unexpected dimension: "
                    f"{actual_dimension}; expected {self._dimension}"
                )

        if self._normalize:
            return normalize_embeddings(embeddings)

        return embeddings

    def embed_text(self, text: str) -> list[float]:
        value = clean_text(text)

        if not value:
            if self._dimension is None:
                raise EmbeddingConfigurationError(
                    "Cannot embed empty text before the embedding dimension "
                    "has been discovered."
                )

            return [0.0] * self.dimension

        return self._request_embeddings([value])[0]

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        values = validate_texts(texts)

        if not values:
            return []

        all_embeddings: list[list[float]] = []

        for start in range(0, len(values), self._batch_size):
            batch = values[start : start + self._batch_size]
            all_embeddings.extend(self._request_embeddings(batch))

        return all_embeddings


# ============================================================================
# Cached provider
# ============================================================================


class CachedEmbeddingProvider(EmbeddingProvider):
    """
    Thread-safe in-memory embedding cache.

    Recommended for:
    - Repeated RAG queries
    - Duplicate chunks
    - Development environments
    - Reducing API calls

    For distributed production deployments, replace this with Redis or
    another shared cache.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        max_size: int = 10_000,
    ) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than zero.")

        self._provider = provider
        self._max_size = max_size
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def _cache_key(self, text: str) -> str:
        return f"{self.model_name}:{stable_text_hash(text)}"

    def _set_cache(
        self,
        key: str,
        vector: list[float],
    ) -> None:
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key, None)

            self._cache[key] = list(vector)

    def embed_text(self, text: str) -> list[float]:
        value = clean_text(text)

        if not value:
            return [0.0] * self.dimension

        key = self._cache_key(value)

        with self._lock:
            cached = self._cache.get(key)

        if cached is not None:
            return list(cached)

        vector = self._provider.embed_text(value)
        self._set_cache(key, vector)

        return list(vector)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        values = validate_texts(texts)

        if not values:
            return []

        results: list[list[float] | None] = [None] * len(values)
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        for index, text in enumerate(values):
            key = self._cache_key(text)

            with self._lock:
                cached = self._cache.get(key)

            if cached is not None:
                results[index] = list(cached)
            else:
                missing_texts.append(text)
                missing_indices.append(index)

        if missing_texts:
            generated = self._provider.embed_documents(missing_texts)

            for index, text, vector in zip(
                missing_indices,
                missing_texts,
                generated,
            ):
                self._set_cache(self._cache_key(text), vector)
                results[index] = list(vector)

        return [
            vector if vector is not None else [0.0] * self.dimension
            for vector in results
        ]

    def clear(self) -> None:
        """
        Clear all cached embeddings.
        """

        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """
        Return the current cache size.
        """

        with self._lock:
            return len(self._cache)


# ============================================================================
# Embedding service
# ============================================================================


class EmbeddingService:
    """
    High-level embedding service for FinCo AI.

    Responsibilities:
    - Text cleaning
    - Query embedding
    - Document embedding
    - Similarity calculation
    - Batch processing
    - Provider management
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        normalize: bool = True,
    ) -> None:
        self.provider = provider
        self.normalize = normalize

    @property
    def model_name(self) -> str:
        return self.provider.model_name

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def embed_text(self, text: str) -> list[float]:
        vector = self.provider.embed_text(clean_text(text))

        if self.normalize:
            return normalize_vector(vector)

        return validate_vector(vector)

    def embed_query(self, query: str) -> list[float]:
        value = clean_text(query)

        if not value:
            raise ValueError("Query cannot be empty.")

        vector = self.provider.embed_query(value)

        if self.normalize:
            return normalize_vector(vector)

        return validate_vector(vector)

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        values = validate_texts(texts)

        embeddings = self.provider.embed_documents(values)

        if len(embeddings) != len(values):
            raise EmbeddingError(
                "Provider returned an unexpected number of embeddings: "
                f"{len(embeddings)} != {len(values)}"
            )

        if self.normalize:
            return normalize_embeddings(embeddings)

        return [validate_vector(vector) for vector in embeddings]

    def embed_document_objects(
        self,
        documents: Sequence[EmbeddingDocument],
    ) -> list[list[float]]:
        texts = [document.text for document in documents]
        return self.embed_documents(texts)

    def similarity(
        self,
        vector_a: Sequence[float],
        vector_b: Sequence[float],
    ) -> float:
        return cosine_similarity(vector_a, vector_b)

    def rank_by_similarity(
        self,
        query_vector: Sequence[float],
        document_vectors: Sequence[Sequence[float]],
        *,
        document_ids: Sequence[str | None] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[SimilarityResult]:
        """
        Rank document vectors against a query vector.
        """

        if document_ids is not None and len(document_ids) != len(
            document_vectors
        ):
            raise ValueError(
                "document_ids must have the same length as document_vectors."
            )

        if metadata is not None and len(metadata) != len(document_vectors):
            raise ValueError(
                "metadata must have the same length as document_vectors."
            )

        scored: list[SimilarityResult] = []

        for index, document_vector in enumerate(document_vectors):
            score = self.similarity(query_vector, document_vector)

            if min_score is not None and score < min_score:
                continue

            scored.append(
                SimilarityResult(
                    score=score,
                    index=index,
                    document_id=(
                        document_ids[index]
                        if document_ids is not None
                        else None
                    ),
                    metadata=(
                        metadata[index]
                        if metadata is not None
                        else {}
                    ),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)

        if top_k is not None:
            if top_k <= 0:
                return []

            scored = scored[:top_k]

        return scored

    def similarity_matrix(
        self,
        query_vectors: Sequence[Sequence[float]],
        document_vectors: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """
        Calculate a query-document similarity matrix.
        """

        return [
            [
                self.similarity(query_vector, document_vector)
                for document_vector in document_vectors
            ]
            for query_vector in query_vectors
        ]


# ============================================================================
# Factory functions
# ============================================================================


def create_embedding_provider(
    provider: str | None = None,
    *,
    model_name: str | None = None,
    dimension: int = 384,
    use_cache: bool = True,
    **kwargs: Any,
) -> EmbeddingProvider:
    """
    Create an embedding provider from configuration.

    Supported provider values:
        - hash
        - sentence_transformers
        - sentence-transformers
        - openai

    Environment:
        FINCO_EMBEDDING_PROVIDER=sentence_transformers
    """

    selected_provider = (
        provider
        or os.getenv("FINCO_EMBEDDING_PROVIDER")
        or "hash"
    ).strip().lower()

    if selected_provider == "hash":
        base_provider: EmbeddingProvider = HashEmbeddingProvider(
            dimension=dimension,
            model=model_name or "hash-embedding-v1",
            normalize=kwargs.pop("normalize", True),
        )

    elif selected_provider in {
        "sentence_transformers",
        "sentence-transformer",
        "sentence-transformers",
        "local",
    }:
        base_provider = SentenceTransformerEmbeddingProvider(
            model_name=(
                model_name
                or "sentence-transformers/all-MiniLM-L6-v2"
            ),
            **kwargs,
        )

    elif selected_provider in {"openai", "openai-compatible"}:
        base_provider = OpenAIEmbeddingProvider(
            model_name=model_name,
            **kwargs,
        )

    else:
        raise EmbeddingConfigurationError(
            f"Unsupported embedding provider: {selected_provider}"
        )

    if use_cache:
        return CachedEmbeddingProvider(base_provider)

    return base_provider


def create_embedding_service(
    provider: str | None = None,
    *,
    model_name: str | None = None,
    dimension: int = 384,
    use_cache: bool = True,
    normalize: bool = True,
    **kwargs: Any,
) -> EmbeddingService:
    """
    Create a ready-to-use embedding service.
    """

    embedding_provider = create_embedding_provider(
        provider=provider,
        model_name=model_name,
        dimension=dimension,
        use_cache=use_cache,
        normalize=normalize,
        **kwargs,
    )

    return EmbeddingService(
        provider=embedding_provider,
        normalize=normalize,
    )


# ============================================================================
# Convenience functions
# ============================================================================


_default_service: EmbeddingService | None = None
_default_service_lock = threading.Lock()


def get_default_embedding_service() -> EmbeddingService:
    """
    Lazily create the default embedding service.
    """

    global _default_service

    if _default_service is None:
        with _default_service_lock:
            if _default_service is None:
                _default_service = create_embedding_service()

    return _default_service


def embed_text(text: str) -> list[float]:
    """
    Embed a single text using the default service.
    """

    return get_default_embedding_service().embed_text(text)


def embed_query(query: str) -> list[float]:
    """
    Embed a query using the default service.
    """

    return get_default_embedding_service().embed_query(query)


def embed_documents(
    texts: Sequence[str],
) -> list[list[float]]:
    """
    Embed multiple documents using the default service.
    """

    return get_default_embedding_service().embed_documents(texts)


def get_embedding_dimension() -> int:
    """
    Return the default embedding dimension.
    """

    return get_default_embedding_service().dimension


__all__ = [
    "EmbeddingError",
    "EmbeddingConfigurationError",
    "EmbeddingDimensionError",
    "EmbeddingResult",
    "EmbeddingDocument",
    "SimilarityResult",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "CachedEmbeddingProvider",
    "EmbeddingService",
    "clean_text",
    "validate_texts",
    "validate_vector",
    "normalize_vector",
    "normalize_embeddings",
    "dot_product",
    "cosine_similarity",
    "euclidean_distance",
    "stable_text_hash",
    "create_embedding_provider",
    "create_embedding_service",
    "get_default_embedding_service",
    "embed_text",
    "embed_query",
    "embed_documents",
    "get_embedding_dimension",
]