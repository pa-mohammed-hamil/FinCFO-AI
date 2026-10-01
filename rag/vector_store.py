
"""
FinCo AI - Vector Store

Production-oriented vector database abstraction using Qdrant.

Responsibilities:
    - Manage Qdrant collections
    - Store document chunks and embeddings
    - Perform vector similarity search
    - Apply company and user access filters
    - Delete documents
    - Support batch upserts
    - Expose health checks

Expected embedding dimensions:
    Must match the embedding model used by FinCo AI.

Example payload:
{
    "chunk_id": "chunk_001",
    "document_id": "annual_report_2025",
    "company_id": "company_001",
    "user_id": "user_001",
    "text": "Revenue increased by 18%...",
    "source": "annual_report.pdf",
    "page_number": 12,
    "chunk_index": 0,
    "document_type": "financial_report",
    "access_level": "private",
    "metadata": {
        "fiscal_year": 2025,
        "department": "finance"
    }
}
"""

from __future__ import annotations

import logging
import uuid

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------


class VectorStoreError(Exception):
    """Base exception for vector store failures."""


class VectorStoreConnectionError(VectorStoreError):
    """Raised when Qdrant cannot be reached."""


class VectorStoreValidationError(VectorStoreError):
    """Raised when vector store input is invalid."""


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when a requested vector store resource is missing."""


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class VectorStoreConfig:
    """
    Qdrant configuration.

    url:
        Qdrant server URL.

    api_key:
        Optional API key for hosted Qdrant.

    collection_name:
        Default collection used by FinCo AI.

    vector_size:
        Must match the embedding model dimension.

    distance:
        Cosine, dot product, or Euclidean distance.

    batch_size:
        Maximum number of points per upsert request.
    """

    url: str = "http://localhost:6333"
    api_key: str | None = None

    collection_name: str = "finco_documents"

    vector_size: int = 1536

    distance: models.Distance = models.Distance.COSINE

    batch_size: int = 64

    timeout: float = 30.0

    prefer_grpc: bool = False

    grpc_port: int = 6334

    on_disk_payload: bool = True

    hnsw_ef_construct: int = 128

    def __post_init__(self) -> None:
        if self.vector_size <= 0:
            raise VectorStoreValidationError(
                "vector_size must be greater than zero."
            )

        if self.batch_size <= 0:
            raise VectorStoreValidationError(
                "batch_size must be greater than zero."
            )

        if self.timeout <= 0:
            raise VectorStoreValidationError(
                "timeout must be greater than zero."
            )

        if not self.collection_name.strip():
            raise VectorStoreValidationError(
                "collection_name cannot be empty."
            )


# ---------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class VectorDocument:
    """
    Represents a document chunk stored in Qdrant.
    """

    chunk_id: str
    document_id: str
    company_id: str

    text: str

    embedding: Sequence[float]

    user_id: str | None = None

    source: str | None = None

    page_number: int | None = None

    chunk_index: int | None = None

    document_type: str | None = None

    access_level: str = "private"

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id.strip():
            raise VectorStoreValidationError(
                "chunk_id cannot be empty."
            )

        if not self.document_id.strip():
            raise VectorStoreValidationError(
                "document_id cannot be empty."
            )

        if not self.company_id.strip():
            raise VectorStoreValidationError(
                "company_id cannot be empty."
            )

        if not self.text.strip():
            raise VectorStoreValidationError(
                "text cannot be empty."
            )

        if not self.embedding:
            raise VectorStoreValidationError(
                "embedding cannot be empty."
            )

        if any(
            not isinstance(value, (int, float))
            for value in self.embedding
        ):
            raise VectorStoreValidationError(
                "embedding must contain only numeric values."
            )

        if self.page_number is not None and self.page_number < 1:
            raise VectorStoreValidationError(
                "page_number must be >= 1."
            )

        if self.chunk_index is not None and self.chunk_index < 0:
            raise VectorStoreValidationError(
                "chunk_index must be >= 0."
            )


@dataclass(frozen=True)
class VectorSearchResult:
    """
    A single vector search result.
    """

    point_id: str

    score: float

    text: str

    document_id: str

    company_id: str

    chunk_id: str

    source: str | None = None

    page_number: int | None = None

    chunk_index: int | None = None

    document_type: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchFilters:
    """
    Security-aware search filters.

    company_id is mandatory for tenant isolation.

    user_id is optional, but can be used to restrict access
    to documents belonging to a specific user.

    document_ids can restrict retrieval to selected documents.

    document_types can restrict retrieval to financial reports,
    invoices, contracts, and other document types.
    """

    company_id: str

    user_id: str | None = None

    document_ids: Sequence[str] = field(default_factory=tuple)

    document_types: Sequence[str] = field(default_factory=tuple)

    access_levels: Sequence[str] = field(default_factory=tuple)

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.company_id.strip():
            raise VectorStoreValidationError(
                "company_id is required for secure retrieval."
            )


# ---------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------


class VectorStore:
    """
    Async Qdrant vector store for FinCo AI.

    Usage:

        config = VectorStoreConfig(
            url="http://localhost:6333",
            vector_size=1536,
        )

        store = VectorStore(config)

        await store.initialize()

        await store.upsert_documents(
            documents=[document]
        )

        results = await store.search(
            query_vector=[0.1] * 1536,
            filters=SearchFilters(
                company_id="company_001"
            ),
            top_k=5,
        )

        await store.close()
    """

    def __init__(
        self,
        config: VectorStoreConfig,
        client: AsyncQdrantClient | None = None,
    ) -> None:

        self.config = config

        self._client = client or AsyncQdrantClient(
            url=config.url,
            api_key=config.api_key,
            timeout=config.timeout,
            prefer_grpc=config.prefer_grpc,
            grpc_port=config.grpc_port,
        )

        self._owns_client = client is None

        self._initialized = False

        logger.info(
            "VectorStore created. collection=%s",
            config.collection_name,
        )

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize Qdrant collection.

        Safe to call multiple times.
        """

        if self._initialized:
            return

        try:
            exists = await self._client.collection_exists(
                collection_name=self.config.collection_name
            )

            if not exists:
                await self._create_collection()

                logger.info(
                    "Created collection: %s",
                    self.config.collection_name,
                )

            else:
                logger.info(
                    "Using existing collection: %s",
                    self.config.collection_name,
                )

            self._initialized = True

        except Exception as exc:
            logger.exception("Failed to initialize vector store.")

            raise VectorStoreConnectionError(
                "Could not initialize Qdrant collection."
            ) from exc

    async def close(self) -> None:
        """
        Close the underlying Qdrant client.
        """

        if self._owns_client:
            await self._client.close()

        self._initialized = False

        logger.info("Vector store closed.")

    async def __aenter__(self) -> "VectorStore":
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:

        await self.close()

    # -----------------------------------------------------------------
    # Collection Management
    # -----------------------------------------------------------------

    async def _create_collection(self) -> None:
        """
        Create the vector collection.
        """

        await self._client.create_collection(
            collection_name=self.config.collection_name,

            vectors_config=models.VectorParams(
                size=self.config.vector_size,
                distance=self.config.distance,
            ),

            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=self.config.hnsw_ef_construct,
            ),

            on_disk_payload=self.config.on_disk_payload,
        )

    async def recreate_collection(self) -> None:
        """
        WARNING:
            Deletes all vectors in the collection.

        Use only during development or controlled migrations.
        """

        try:
            await self._client.delete_collection(
                collection_name=self.config.collection_name
            )

            await self._create_collection()

            self._initialized = True

            logger.warning(
                "Recreated vector collection: %s",
                self.config.collection_name,
            )

        except Exception as exc:
            logger.exception("Failed to recreate collection.")

            raise VectorStoreError(
                "Could not recreate vector collection."
            ) from exc

    async def collection_info(self) -> dict[str, Any]:
        """
        Return collection information.
        """

        await self.initialize()

        try:
            info = await self._client.get_collection(
                collection_name=self.config.collection_name
            )

            return {
                "collection_name": self.config.collection_name,
                "status": str(info.status),
                "optimizer_status": str(
                    info.optimizer_status
                ),
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "indexed_vectors_count": (
                    info.indexed_vectors_count
                ),
            }

        except Exception as exc:
            raise VectorStoreError(
                "Failed to retrieve collection information."
            ) from exc

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _validate_embedding(
        self,
        embedding: Sequence[float],
    ) -> None:

        if len(embedding) != self.config.vector_size:
            raise VectorStoreValidationError(
                "Embedding dimension mismatch. "
                f"Expected {self.config.vector_size}, "
                f"received {len(embedding)}."
            )

    def _validate_documents(
        self,
        documents: Sequence[VectorDocument],
    ) -> None:

        for document in documents:
            self._validate_embedding(document.embedding)

    def _validate_query_vector(
        self,
        query_vector: Sequence[float],
    ) -> None:

        if not query_vector:
            raise VectorStoreValidationError(
                "query_vector cannot be empty."
            )

        self._validate_embedding(query_vector)

    # -----------------------------------------------------------------
    # Payload Construction
    # -----------------------------------------------------------------

    @staticmethod
    def _build_payload(
        document: VectorDocument,
    ) -> dict[str, Any]:
        """
        Convert a VectorDocument into Qdrant payload.

        Keep the text in the payload so retrieval can return
        the original chunk without another database query.
        """

        payload: dict[str, Any] = {
            "chunk_id": document.chunk_id,
            "document_id": document.document_id,
            "company_id": document.company_id,
            "user_id": document.user_id,
            "text": document.text,
            "source": document.source,
            "page_number": document.page_number,
            "chunk_index": document.chunk_index,
            "document_type": document.document_type,
            "access_level": document.access_level,
            "metadata": dict(document.metadata),
        }

        return payload

    # -----------------------------------------------------------------
    # Upsert
    # -----------------------------------------------------------------

    async def upsert_document(
        self,
        document: VectorDocument,
    ) -> str:
        """
        Insert or update one document chunk.

        Returns:
            The Qdrant point ID.
        """

        result = await self.upsert_documents(
            documents=[document]
        )

        return result[0]

    async def upsert_documents(
        self,
        documents: Sequence[VectorDocument],
    ) -> list[str]:
        """
        Batch upsert document chunks.

        Existing points with the same point ID are replaced.
        """

        await self.initialize()

        if not documents:
            return []

        self._validate_documents(documents)

        point_ids: list[str] = []

        try:
            for start in range(
                0,
                len(documents),
                self.config.batch_size,
            ):

                batch = documents[
                    start : start + self.config.batch_size
                ]

                points: list[models.PointStruct] = []

                for document in batch:

                    point_id = self._normalize_point_id(
                        document.chunk_id
                    )

                    points.append(
                        models.PointStruct(
                            id=point_id,

                            vector=list(document.embedding),

                            payload=self._build_payload(
                                document
                            ),
                        )
                    )

                    point_ids.append(point_id)

                await self._client.upsert(
                    collection_name=self.config.collection_name,
                    points=points,
                    wait=True,
                )

            logger.info(
                "Upserted %d document chunks.",
                len(documents),
            )

            return point_ids

        except Exception as exc:
            logger.exception(
                "Failed to upsert document chunks."
            )

            raise VectorStoreError(
                "Document upsert failed."
            ) from exc

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    async def search(
        self,
        query_vector: Sequence[float],
        filters: SearchFilters,
        top_k: int = 5,
        score_threshold: float | None = None,
        with_vectors: bool = False,
    ) -> list[VectorSearchResult]:
        """
        Perform secure vector similarity search.

        company_id is always applied.

        Args:
            query_vector:
                Query embedding.

            filters:
                Tenant and metadata filters.

            top_k:
                Number of results.

            score_threshold:
                Optional minimum similarity score.

            with_vectors:
                Return embeddings in payload result.
                Usually False for lower response size.
        """

        await self.initialize()

        self._validate_query_vector(query_vector)

        if top_k <= 0:
            raise VectorStoreValidationError(
                "top_k must be greater than zero."
            )

        query_filter = self._build_filter(filters)

        try:
            response = await self._client.query_points(
                collection_name=self.config.collection_name,

                query=list(query_vector),

                query_filter=query_filter,

                limit=top_k,

                score_threshold=score_threshold,

                with_payload=True,

                with_vectors=with_vectors,
            )

            results: list[VectorSearchResult] = []

            for point in response.points:

                payload = dict(point.payload or {})

                results.append(
                    self._to_search_result(
                        point=point,
                        payload=payload,
                    )
                )

            logger.debug(
                "Vector search returned %d results.",
                len(results),
            )

            return results

        except Exception as exc:
            logger.exception("Vector search failed.")

            raise VectorStoreError(
                "Vector similarity search failed."
            ) from exc

    # -----------------------------------------------------------------
    # Filtering
    # -----------------------------------------------------------------

    def _build_filter(
        self,
        filters: SearchFilters,
    ) -> models.Filter:
        """
        Build a Qdrant filter.

        Security:
            company_id is mandatory.

        All conditions are combined using AND.
        """

        must_conditions: list[models.FieldCondition] = []

        # Mandatory tenant isolation.
        must_conditions.append(
            models.FieldCondition(
                key="company_id",
                match=models.MatchValue(
                    value=filters.company_id
                ),
            )
        )

        if filters.user_id:

            must_conditions.append(
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(
                        value=filters.user_id
                    ),
                )
            )

        if filters.document_ids:

            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(
                        any=list(filters.document_ids)
                    ),
                )
            )

        if filters.document_types:

            must_conditions.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchAny(
                        any=list(filters.document_types)
                    ),
                )
            )

        if filters.access_levels:

            must_conditions.append(
                models.FieldCondition(
                    key="access_level",
                    match=models.MatchAny(
                        any=list(filters.access_levels)
                    ),
                )
            )

        for key, value in filters.metadata.items():

            must_conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(
                        value=value
                    ),
                )
            )

        return models.Filter(
            must=must_conditions
        )

    # -----------------------------------------------------------------
    # Result Conversion
    # -----------------------------------------------------------------

    @staticmethod
    def _to_search_result(
        point: Any,
        payload: dict[str, Any],
    ) -> VectorSearchResult:
        """
        Convert Qdrant ScoredPoint into application result.
        """

        return VectorSearchResult(
            point_id=str(point.id),

            score=float(point.score),

            text=str(payload.get("text", "")),

            document_id=str(
                payload.get("document_id", "")
            ),

            company_id=str(
                payload.get("company_id", "")
            ),

            chunk_id=str(
                payload.get("chunk_id", "")
            ),

            source=payload.get("source"),

            page_number=payload.get("page_number"),

            chunk_index=payload.get("chunk_index"),

            document_type=payload.get("document_type"),

            metadata=dict(
                payload.get("metadata") or {}
            ),

            payload=payload,
        )

    # -----------------------------------------------------------------
    # Delete Operations
    # -----------------------------------------------------------------

    async def delete_document(
        self,
        document_id: str,
        company_id: str,
    ) -> None:
        """
        Delete all chunks belonging to one document.

        company_id prevents cross-tenant deletion.
        """

        await self.initialize()

        if not document_id.strip():
            raise VectorStoreValidationError(
                "document_id cannot be empty."
            )

        if not company_id.strip():
            raise VectorStoreValidationError(
                "company_id cannot be empty."
            )

        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id
                    ),
                ),

                models.FieldCondition(
                    key="company_id",
                    match=models.MatchValue(
                        value=company_id
                    ),
                ),
            ]
        )

        try:
            await self._client.delete(
                collection_name=self.config.collection_name,

                points_selector=models.FilterSelector(
                    filter=delete_filter
                ),

                wait=True,
            )

            logger.info(
                "Deleted document chunks. "
                "document_id=%s company_id=%s",
                document_id,
                company_id,
            )

        except Exception as exc:
            logger.exception(
                "Failed to delete document."
            )

            raise VectorStoreError(
                "Document deletion failed."
            ) from exc

    async def delete_chunk(
        self,
        chunk_id: str,
        company_id: str,
    ) -> None:
        """
        Delete one chunk only if it belongs to the company.
        """

        await self.initialize()

        point_id = self._normalize_point_id(chunk_id)

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchValue(
                        value=chunk_id
                    ),
                ),

                models.FieldCondition(
                    key="company_id",
                    match=models.MatchValue(
                        value=company_id
                    ),
                ),
            ]
        )

        try:
            await self._client.delete(
                collection_name=self.config.collection_name,

                points_selector=models.FilterSelector(
                    filter=query_filter
                ),

                wait=True,
            )

            logger.info(
                "Deleted chunk: %s",
                point_id,
            )

        except Exception as exc:
            raise VectorStoreError(
                "Chunk deletion failed."
            ) from exc

    # -----------------------------------------------------------------
    # Retrieve by IDs
    # -----------------------------------------------------------------

    async def get_by_ids(
        self,
        chunk_ids: Sequence[str],
        company_id: str,
    ) -> list[VectorSearchResult]:
        """
        Retrieve chunks by point IDs.

        Applies company_id filter after retrieval.
        """

        await self.initialize()

        if not chunk_ids:
            return []

        if not company_id.strip():
            raise VectorStoreValidationError(
                "company_id is required."
            )

        point_ids = [
            self._normalize_point_id(chunk_id)
            for chunk_id in chunk_ids
        ]

        try:
            records = await self._client.retrieve(
                collection_name=self.config.collection_name,

                ids=point_ids,

                with_payload=True,

                with_vectors=False,
            )

            results: list[VectorSearchResult] = []

            for record in records:

                payload = dict(record.payload or {})

                if payload.get("company_id") != company_id:
                    continue

                results.append(
                    self._to_search_result(
                        point=record,
                        payload=payload,
                    )
                )

            return results

        except Exception as exc:
            raise VectorStoreError(
                "Failed to retrieve chunks."
            ) from exc

    # -----------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """
        Check Qdrant connectivity and collection availability.
        """

        try:
            await self.initialize()

            info = await self.collection_info()

            return {
                "status": "healthy",
                "database": "qdrant",
                "collection": self.config.collection_name,
                "details": info,
            }

        except Exception as exc:

            logger.exception(
                "Vector store health check failed."
            )

            return {
                "status": "unhealthy",
                "database": "qdrant",
                "collection": self.config.collection_name,
                "error": str(exc),
            }

    # -----------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize_point_id(value: str) -> str:
        """
        Normalize application chunk IDs into UUIDs.

        Qdrant accepts UUIDs or unsigned integer IDs.
        Using UUID5 gives deterministic IDs for repeated upserts.

        The original chunk_id remains in the payload.
        """

        if not value.strip():
            raise VectorStoreValidationError(
                "Point ID cannot be empty."
            )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"finco-ai:{value}",
            )
        )