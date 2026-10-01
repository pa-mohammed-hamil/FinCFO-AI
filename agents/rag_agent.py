"""
FinCo AI - RAG Agent
====================

Retrieval-Augmented Generation agent for financial documents.

Responsibilities
----------------
- Process document-oriented user questions.
- Rewrite/normalize queries when supported.
- Retrieve relevant document chunks.
- Support hybrid retrieval.
- Apply metadata filtering.
- Apply reranking/context compression when available.
- Generate grounded answers from retrieved evidence.
- Preserve document/page/chunk citations.
- Validate citations when a validator is available.
- Detect insufficient evidence.
- Prevent unsupported claims.
- Return structured results to the orchestrator.
- Support document search, summarization and evidence extraction.

Architecture
------------

User Query
    |
    v
RAG Agent
    |
    +--> Query Rewriter
    |
    +--> Metadata Filter
    |
    +--> Retrieval Service
    |       |
    |       +--> Vector Search
    |       +--> BM25
    |       +--> Hybrid Retrieval
    |
    +--> Reranker
    |
    +--> Context Compressor
    |
    +--> LLM / Generator
    |
    +--> Citation Generator
    |
    +--> Citation Validator
    |
    +--> Guardrails
    |
    v
Grounded RAG Result

Design Principles
-----------------
1. Retrieved evidence is the source of truth for document questions.
2. The agent must not fabricate citations.
3. Low-evidence queries should return an explicit evidence warning.
4. Authorization must remain outside or above this agent.
5. The agent must not bypass security/guardrails.
6. Financial calculations should use financial tools/services rather
   than arithmetic performed from unsupported text.
"""

from __future__ import annotations

import inspect
import re
import time
import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
)


# ============================================================================
# Exceptions
# ============================================================================


class RAGAgentError(Exception):
    """Base RAG agent exception."""


class InvalidRAGRequestError(
    RAGAgentError
):
    """Raised when a RAG request is invalid."""


class RAGAgentConfigurationError(
    RAGAgentError
):
    """Raised when RAG agent configuration is invalid."""


class RAGAgentExecutionError(
    RAGAgentError
):
    """Raised when RAG execution fails."""


class RAGAgentAccessDeniedError(
    RAGAgentError
):
    """Raised when access to requested information is denied."""


class RAGRetrievalError(
    RAGAgentError
):
    """Raised when retrieval fails."""


class RAGGenerationError(
    RAGAgentError
):
    """Raised when answer generation fails."""


class RAGCitationError(
    RAGAgentError
):
    """Raised when citation processing fails."""


class RAGGuardrailViolationError(
    RAGAgentError
):
    """Raised when a RAG request violates guardrails."""


class RAGInsufficientEvidenceError(
    RAGAgentError
):
    """Raised when evidence is insufficient for a grounded answer."""


# ============================================================================
# Constants
# ============================================================================


DEFAULT_TOP_K = 8
DEFAULT_RETRIEVAL_K = 20
DEFAULT_MAX_CONTEXT_LENGTH = 30_000
DEFAULT_MAX_QUERY_LENGTH = 10_000
DEFAULT_MIN_RELEVANCE_SCORE = 0.20
DEFAULT_MIN_GROUNDEDNESS_SCORE = 0.50
DEFAULT_MAX_CITATIONS = 20
DEFAULT_TIMEOUT_SECONDS = 60


# ============================================================================
# Enums
# ============================================================================


class RAGOperation(str, Enum):
    """Supported RAG operations."""

    ANSWER = "answer"
    SEARCH = "search"
    RETRIEVE = "retrieve"
    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    COMPARE = "compare"
    DOCUMENT_QA = "document_qa"
    EVIDENCE = "evidence"


class RetrievalMode(str, Enum):
    """Retrieval strategies."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    AUTO = "auto"


class RAGRiskLevel(str, Enum):
    """RAG answer risk."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceStatus(str, Enum):
    """Evidence quality."""

    SUFFICIENT = "sufficient"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"
    NONE = "none"


class CitationStatus(str, Enum):
    """Citation validation status."""

    VALID = "valid"
    PARTIAL = "partial"
    INVALID = "invalid"
    MISSING = "missing"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RAGAgentConfig:
    """Runtime configuration for the RAG agent."""

    enabled: bool = True

    top_k: int = DEFAULT_TOP_K

    retrieval_k: int = DEFAULT_RETRIEVAL_K

    max_context_length: int = (
        DEFAULT_MAX_CONTEXT_LENGTH
    )

    max_query_length: int = (
        DEFAULT_MAX_QUERY_LENGTH
    )

    min_relevance_score: float = (
        DEFAULT_MIN_RELEVANCE_SCORE
    )

    min_groundedness_score: float = (
        DEFAULT_MIN_GROUNDEDNESS_SCORE
    )

    max_citations: int = (
        DEFAULT_MAX_CITATIONS
    )

    timeout_seconds: float = (
        DEFAULT_TIMEOUT_SECONDS
    )

    retrieval_mode: RetrievalMode = (
        RetrievalMode.AUTO
    )

    use_query_rewriter: bool = True

    use_reranker: bool = True

    use_context_compressor: bool = True

    use_citation_generator: bool = True

    use_citation_validator: bool = True

    require_citations: bool = True

    allow_partial_evidence: bool = True

    fail_on_invalid_citations: bool = False

    enable_groundedness_check: bool = True

    audit_enabled: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate configuration."""

        if self.top_k < 1:
            raise RAGAgentConfigurationError(
                "top_k must be greater than zero."
            )

        if self.retrieval_k < self.top_k:
            raise RAGAgentConfigurationError(
                "retrieval_k must be >= top_k."
            )

        if self.max_context_length < 100:
            raise RAGAgentConfigurationError(
                "max_context_length is too small."
            )

        if self.max_query_length < 1:
            raise RAGAgentConfigurationError(
                "max_query_length must be positive."
            )

        if not 0 <= self.min_relevance_score <= 1:
            raise RAGAgentConfigurationError(
                "min_relevance_score must be between 0 and 1."
            )

        if not 0 <= self.min_groundedness_score <= 1:
            raise RAGAgentConfigurationError(
                "min_groundedness_score must be between 0 and 1."
            )

        if self.max_citations < 1:
            raise RAGAgentConfigurationError(
                "max_citations must be positive."
            )


# ============================================================================
# Request Models
# ============================================================================


@dataclass
class RAGAgentRequest:
    """Input to the RAG agent."""

    query: str

    operation: RAGOperation = (
        RAGOperation.ANSWER
    )

    company_id: Optional[str] = None

    user_id: Optional[str] = None

    session_id: Optional[str] = None

    workflow_id: Optional[str] = None

    document_ids: List[str] = field(
        default_factory=list
    )

    metadata_filters: Dict[str, Any] = field(
        default_factory=dict
    )

    top_k: Optional[int] = None

    retrieval_mode: Optional[
        RetrievalMode
    ] = None

    context: Dict[str, Any] = field(
        default_factory=dict
    )

    previous_results: List[
        Any
    ] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# Evidence Models
# ============================================================================


@dataclass
class RAGEvidence:
    """Normalized retrieved evidence."""

    evidence_id: str

    content: str

    score: float = 0.0

    rerank_score: Optional[float] = None

    document_id: Optional[str] = None

    document_name: Optional[str] = None

    page_number: Optional[int] = None

    chunk_id: Optional[str] = None

    section: Optional[str] = None

    source: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "evidence_id": self.evidence_id,
            "content": self.content,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "source": self.source,
            "metadata": _serialize(
                self.metadata
            ),
        }


@dataclass
class RAGCitation:
    """Citation attached to generated evidence."""

    citation_id: str

    evidence_id: Optional[str] = None

    document_id: Optional[str] = None

    document_name: Optional[str] = None

    page_number: Optional[int] = None

    chunk_id: Optional[str] = None

    section: Optional[str] = None

    quote: Optional[str] = None

    status: CitationStatus = (
        CitationStatus.VALID
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "citation_id": self.citation_id,
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "quote": self.quote,
            "status": self.status.value,
            "metadata": _serialize(
                self.metadata
            ),
        }


@dataclass
class RAGFinding:
    """Important finding extracted from evidence."""

    finding: str

    confidence: float = 0.0

    evidence_ids: List[str] = field(
        default_factory=list
    )

    citations: List[str] = field(
        default_factory=list
    )

    risk_level: RAGRiskLevel = (
        RAGRiskLevel.LOW
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "finding": self.finding,
            "confidence": self.confidence,
            "evidence_ids": list(
                self.evidence_ids
            ),
            "citations": list(
                self.citations
            ),
            "risk_level": (
                self.risk_level.value
            ),
            "metadata": _serialize(
                self.metadata
            ),
        }


# ============================================================================
# Result Model
# ============================================================================


@dataclass
class RAGAgentResult:
    """Structured result returned by the RAG agent."""

    request_id: str

    operation: RAGOperation

    answer: str = ""

    evidence: List[
        RAGEvidence
    ] = field(
        default_factory=list
    )

    citations: List[
        RAGCitation
    ] = field(
        default_factory=list
    )

    findings: List[
        RAGFinding
    ] = field(
        default_factory=list
    )

    evidence_status: EvidenceStatus = (
        EvidenceStatus.NONE
    )

    citation_status: CitationStatus = (
        CitationStatus.MISSING
    )

    groundedness_score: Optional[
        float
    ] = None

    confidence: float = 0.0

    risk_level: RAGRiskLevel = (
        RAGRiskLevel.LOW
    )

    query: str = ""

    rewritten_query: Optional[str] = None

    retrieval_mode: RetrievalMode = (
        RetrievalMode.AUTO
    )

    execution_time_ms: float = 0.0

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    success: bool = True

    def to_dict(self) -> Dict[str, Any]:

        return {
            "request_id": self.request_id,
            "operation": self.operation.value,
            "answer": self.answer,
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "citations": [
                item.to_dict()
                for item in self.citations
            ],
            "findings": [
                item.to_dict()
                for item in self.findings
            ],
            "evidence_status": (
                self.evidence_status.value
            ),
            "citation_status": (
                self.citation_status.value
            ),
            "groundedness_score": (
                self.groundedness_score
            ),
            "confidence": self.confidence,
            "risk_level": (
                self.risk_level.value
            ),
            "query": self.query,
            "rewritten_query": (
                self.rewritten_query
            ),
            "retrieval_mode": (
                self.retrieval_mode.value
            ),
            "execution_time_ms": (
                self.execution_time_ms
            ),
            "warnings": list(
                self.warnings
            ),
            "errors": list(
                self.errors
            ),
            "metadata": _serialize(
                self.metadata
            ),
            "success": self.success,
        }


# ============================================================================
# RAG Agent
# ============================================================================


class RAGAgent:
    """
    Financial document RAG agent.

    Dependencies are injected rather than imported directly so the agent
    remains testable and compatible with different vector stores, LLMs,
    rerankers and retrieval implementations.
    """

    def __init__(
        self,
        retrieval_service: Optional[
            Any
        ] = None,
        query_rewriter: Optional[
            Any
        ] = None,
        reranker: Optional[
            Any
        ] = None,
        context_compressor: Optional[
            Any
        ] = None,
        citation_generator: Optional[
            Any
        ] = None,
        citation_validator: Optional[
            Any
        ] = None,
        generator: Optional[
            Any
        ] = None,
        guardrails: Optional[
            Any
        ] = None,
        authorization_service: Optional[
            Any
        ] = None,
        audit_service: Optional[
            Any
        ] = None,
        config: Optional[
            RAGAgentConfig
        ] = None,
    ) -> None:

        self.retrieval_service = (
            retrieval_service
        )

        self.query_rewriter = (
            query_rewriter
        )

        self.reranker = reranker

        self.context_compressor = (
            context_compressor
        )

        self.citation_generator = (
            citation_generator
        )

        self.citation_validator = (
            citation_validator
        )

        self.generator = generator

        self.guardrails = guardrails

        self.authorization_service = (
            authorization_service
        )

        self.audit_service = (
            audit_service
        )

        self.config = (
            config
            or RAGAgentConfig()
        )

        self.config.validate()

    # ========================================================================
    # Public API
    # ========================================================================

    def run(
        self,
        request: RAGAgentRequest
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> RAGAgentResult:

        request = self.normalize_request(
            request,
            **kwargs,
        )

        request_id = (
            request.metadata.get(
                "request_id"
            )
            or self.generate_request_id()
        )

        started = time.perf_counter()

        try:

            self.validate_request(
                request
            )

            self.authorize(
                request
            )

            self.check_guardrails(
                request
            )

            rewritten_query = (
                self.rewrite_query(
                    request
                )
            )

            retrieval_mode = (
                request.retrieval_mode
                or self.config.retrieval_mode
            )

            evidence = (
                self.retrieve(
                    request,
                    rewritten_query,
                    retrieval_mode,
                )
            )

            evidence = self.filter_evidence(
                evidence
            )

            evidence_status = (
                self.assess_evidence(
                    evidence
                )
            )

            if (
                evidence_status
                == EvidenceStatus.NONE
            ):

                result = RAGAgentResult(
                    request_id=request_id,
                    operation=request.operation,
                    query=request.query,
                    rewritten_query=rewritten_query,
                    retrieval_mode=retrieval_mode,
                    evidence_status=evidence_status,
                    answer=(
                        "I could not find relevant "
                        "document evidence to answer "
                        "this question reliably."
                    ),
                    confidence=0.0,
                    risk_level=RAGRiskLevel.HIGH,
                    warnings=[
                        "No relevant evidence was retrieved."
                    ],
                    success=False,
                )

                self.finalize_result(
                    result,
                    started,
                )

                self.audit(
                    request,
                    result,
                )

                return result

            if (
                request.operation
                in {
                    RAGOperation.SEARCH,
                    RAGOperation.RETRIEVE,
                    RAGOperation.EVIDENCE,
                }
            ):

                result = self.build_retrieval_result(
                    request=request,
                    request_id=request_id,
                    query=rewritten_query,
                    evidence=evidence,
                    evidence_status=evidence_status,
                    retrieval_mode=retrieval_mode,
                )

            else:

                result = self.answer_question(
                    request=request,
                    request_id=request_id,
                    query=rewritten_query,
                    evidence=evidence,
                    evidence_status=evidence_status,
                    retrieval_mode=retrieval_mode,
                )

            self.check_output_guardrails(
                result
            )

            self.finalize_result(
                result,
                started,
            )

            self.audit(
                request,
                result,
            )

            return result

        except (
            RAGAgentError,
        ):

            raise

        except Exception as exc:

            raise RAGAgentExecutionError(
                "RAG agent execution failed."
            ) from exc

    # ========================================================================
    # Specialized APIs
    # ========================================================================

    def answer(
        self,
        query: str,
        **kwargs: Any,
    ) -> RAGAgentResult:

        return self.run(
            RAGAgentRequest(
                query=query,
                operation=RAGOperation.ANSWER,
                **kwargs,
            )
        )

    def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> RAGAgentResult:

        return self.run(
            RAGAgentRequest(
                query=query,
                operation=RAGOperation.SEARCH,
                **kwargs,
            )
        )

    def retrieve(
        self,
        request: RAGAgentRequest,
        rewritten_query: Optional[str] = None,
        retrieval_mode: Optional[
            RetrievalMode
        ] = None,
    ) -> List[RAGEvidence]:
        """
        Retrieve normalized evidence.

        Note:
        The overloaded public name 'retrieve' is intentionally avoided
        elsewhere; this method is the agent's retrieval API.
        """

        if self.retrieval_service is None:

            raise RAGAgentConfigurationError(
                "Retrieval service is not configured."
            )

        query = (
            rewritten_query
            or request.query
        )

        mode = (
            retrieval_mode
            or request.retrieval_mode
            or self.config.retrieval_mode
        )

        payload = {
            "query": query,
            "top_k": (
                request.top_k
                or self.config.retrieval_k
            ),
            "retrieval_mode": mode.value,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "document_ids": request.document_ids,
            "metadata_filters": (
                request.metadata_filters
            ),
            "context": request.context,
            "metadata": request.metadata,
        }

        try:

            raw_results = self.invoke_component(
                self.retrieval_service,
                (
                    "retrieve",
                    "search",
                    "hybrid_search",
                    "query",
                    "run",
                ),
                payload,
            )

        except Exception as exc:

            raise RAGRetrievalError(
                "Document retrieval failed."
            ) from exc

        return self.normalize_evidence(
            raw_results
        )

    # ========================================================================
    # Query Rewriting
    # ========================================================================

    def rewrite_query(
        self,
        request: RAGAgentRequest,
    ) -> str:

        query = request.query.strip()

        if (
            not self.config.use_query_rewriter
            or self.query_rewriter is None
        ):

            return query

        payload = {
            "query": query,
            "company_id": request.company_id,
            "context": request.context,
            "metadata": request.metadata,
        }

        try:

            rewritten = self.invoke_component(
                self.query_rewriter,
                (
                    "rewrite",
                    "rewrite_query",
                    "transform",
                    "run",
                ),
                payload,
            )

            value = self.extract_text(
                rewritten
            )

            if value:

                return value[: self.config.max_query_length]

        except Exception:

            # Query rewriting is an optimization.
            # Retrieval should continue with the original query.
            pass

        return query

    # ========================================================================
    # Evidence Processing
    # ========================================================================

    def filter_evidence(
        self,
        evidence: Sequence[
            RAGEvidence
        ],
    ) -> List[RAGEvidence]:

        filtered = [
            item
            for item in evidence
            if (
                item.content.strip()
                and item.score
                >= self.config.min_relevance_score
            )
        ]

        filtered.sort(
            key=lambda item: (
                item.rerank_score
                if item.rerank_score is not None
                else item.score
            ),
            reverse=True,
        )

        return filtered[
            : self.config.retrieval_k
        ]

    def assess_evidence(
        self,
        evidence: Sequence[
            RAGEvidence
        ],
    ) -> EvidenceStatus:

        if not evidence:

            return EvidenceStatus.NONE

        high_quality = [
            item
            for item in evidence
            if item.score
            >= self.config.min_relevance_score
        ]

        if len(high_quality) >= 3:

            return EvidenceStatus.SUFFICIENT

        if len(high_quality) >= 1:

            return EvidenceStatus.LIMITED

        return EvidenceStatus.INSUFFICIENT

    def normalize_evidence(
        self,
        raw_results: Any,
    ) -> List[RAGEvidence]:

        if raw_results is None:

            return []

        if isinstance(
            raw_results,
            Mapping,
        ):

            for key in (
                "results",
                "documents",
                "chunks",
                "evidence",
                "items",
                "data",
            ):

                if key in raw_results:

                    raw_results = (
                        raw_results[key]
                    )

                    break

        if not isinstance(
            raw_results,
            Sequence,
        ) or isinstance(
            raw_results,
            (str, bytes),
        ):

            raw_results = [
                raw_results
            ]

        normalized: List[
            RAGEvidence
        ] = []

        for index, item in enumerate(
            raw_results
        ):

            evidence = (
                self.normalize_single_evidence(
                    item,
                    index,
                )
            )

            if evidence.content:

                normalized.append(
                    evidence
                )

        return normalized

    def normalize_single_evidence(
        self,
        item: Any,
        index: int,
    ) -> RAGEvidence:

        if isinstance(
            item,
            RAGEvidence,
        ):

            return item

        data = (
            dict(item)
            if isinstance(
                item,
                Mapping,
            )
            else self.object_to_dict(
                item
            )
        )

        content = (
            data.get("content")
            or data.get("text")
            or data.get("page_content")
            or data.get("chunk")
            or ""
        )

        score = (
            data.get("score")
            or data.get("similarity")
            or data.get("relevance_score")
            or 0.0
        )

        rerank_score = (
            data.get("rerank_score")
            or data.get("reranker_score")
        )

        metadata = (
            data.get("metadata")
            or {}
        )

        document_id = (
            data.get("document_id")
            or metadata.get("document_id")
        )

        document_name = (
            data.get("document_name")
            or data.get("source_name")
            or metadata.get("document_name")
            or metadata.get("source")
        )

        page_number = (
            data.get("page_number")
            or data.get("page")
            or metadata.get("page_number")
            or metadata.get("page")
        )

        chunk_id = (
            data.get("chunk_id")
            or data.get("id")
            or metadata.get("chunk_id")
        )

        section = (
            data.get("section")
            or metadata.get("section")
        )

        source = (
            data.get("source")
            or metadata.get("source")
        )

        return RAGEvidence(
            evidence_id=(
                str(
                    data.get(
                        "evidence_id"
                    )
                    or chunk_id
                    or f"evidence_{index}"
                )
            ),
            content=str(
                content
            ).strip(),
            score=self.safe_float(
                score
            ),
            rerank_score=(
                self.safe_float(
                    rerank_score
                )
                if rerank_score is not None
                else None
            ),
            document_id=(
                str(document_id)
                if document_id is not None
                else None
            ),
            document_name=(
                str(document_name)
                if document_name is not None
                else None
            ),
            page_number=(
                self.safe_int(
                    page_number
                )
                if page_number is not None
                else None
            ),
            chunk_id=(
                str(chunk_id)
                if chunk_id is not None
                else None
            ),
            section=(
                str(section)
                if section is not None
                else None
            ),
            source=(
                str(source)
                if source is not None
                else None
            ),
            metadata=dict(
                metadata
            ),
        )

    # ========================================================================
    # Reranking
    # ========================================================================

    def rerank(
        self,
        query: str,
        evidence: Sequence[
            RAGEvidence
        ],
    ) -> List[RAGEvidence]:

        if (
            not self.config.use_reranker
            or self.reranker is None
            or not evidence
        ):

            return list(
                evidence[
                    : self.config.top_k
                ]
            )

        payload = {
            "query": query,
            "documents": [
                item.to_dict()
                for item in evidence
            ],
            "top_k": self.config.top_k,
        }

        try:

            result = self.invoke_component(
                self.reranker,
                (
                    "rerank",
                    "rank",
                    "rerank_documents",
                    "run",
                ),
                payload,
            )

            reranked = self.normalize_evidence(
                result
            )

            if reranked:

                return reranked[
                    : self.config.top_k
                ]

        except Exception:
            pass

        return list(
            evidence[
                : self.config.top_k
            ]
        )

    # ========================================================================
    # Context Compression
    # ========================================================================

    def compress_context(
        self,
        query: str,
        evidence: Sequence[
            RAGEvidence
        ],
    ) -> List[RAGEvidence]:

        if (
            not self.config.use_context_compressor
            or self.context_compressor is None
        ):

            return self.limit_context(
                evidence
            )

        payload = {
            "query": query,
            "evidence": [
                item.to_dict()
                for item in evidence
            ],
            "max_context_length": (
                self.config.max_context_length
            ),
        }

        try:

            result = self.invoke_component(
                self.context_compressor,
                (
                    "compress",
                    "compress_context",
                    "run",
                ),
                payload,
            )

            compressed = self.normalize_evidence(
                result
            )

            if compressed:

                return self.limit_context(
                    compressed
                )

        except Exception:
            pass

        return self.limit_context(
            evidence
        )

    def limit_context(
        self,
        evidence: Sequence[
            RAGEvidence
        ],
    ) -> List[RAGEvidence]:

        result: List[
            RAGEvidence
        ] = []

        total_length = 0

        for item in evidence:

            remaining = (
                self.config.max_context_length
                - total_length
            )

            if remaining <= 0:
                break

            content = item.content[
                :remaining
            ]

            if not content:
                continue

            if content != item.content:

                item = RAGEvidence(
                    **{
                        **item.__dict__,
                        "content": content,
                    }
                )

            result.append(
                item
            )

            total_length += len(
                content
            )

        return result

    # ========================================================================
    # Answer Generation
    # ========================================================================

    def answer_question(
        self,
        request: RAGAgentRequest,
        request_id: str,
        query: str,
        evidence: Sequence[
            RAGEvidence
        ],
        evidence_status: EvidenceStatus,
        retrieval_mode: RetrievalMode,
    ) -> RAGAgentResult:

        ranked = self.rerank(
            query,
            evidence,
        )

        compressed = self.compress_context(
            query,
            ranked,
        )

        if not compressed:

            return RAGAgentResult(
                request_id=request_id,
                operation=request.operation,
                query=request.query,
                rewritten_query=query,
                retrieval_mode=retrieval_mode,
                evidence_status=(
                    EvidenceStatus.NONE
                ),
                answer=(
                    "There is not enough relevant "
                    "document evidence to provide "
                    "a reliable answer."
                ),
                confidence=0.0,
                risk_level=RAGRiskLevel.HIGH,
                success=False,
            )

        if request.operation == (
            RAGOperation.SUMMARIZE
        ):

            answer = self.generate_summary(
                query,
                compressed,
                request,
            )

        elif request.operation == (
            RAGOperation.EXTRACT
        ):

            answer = self.generate_extraction(
                query,
                compressed,
                request,
            )

        elif request.operation == (
            RAGOperation.COMPARE
        ):

            answer = self.generate_comparison(
                query,
                compressed,
                request,
            )

        else:

            answer = self.generate_answer(
                query,
                compressed,
                request,
            )

        citations = self.generate_citations(
            answer=answer,
            evidence=compressed,
            request=request,
        )

        citation_status = (
            self.validate_citations(
                answer=answer,
                citations=citations,
                evidence=compressed,
            )
        )

        groundedness = (
            self.calculate_groundedness(
                answer,
                compressed,
                citations,
            )
        )

        confidence = (
            self.calculate_confidence(
                compressed,
                groundedness,
                citation_status,
            )
        )

        warnings: List[
            str
        ] = []

        if evidence_status == (
            EvidenceStatus.LIMITED
        ):

            warnings.append(
                "Only limited relevant evidence "
                "was retrieved."
            )

        if citation_status in {
            CitationStatus.PARTIAL,
            CitationStatus.MISSING,
        }:

            warnings.append(
                "Some answer claims may not have "
                "complete citation coverage."
            )

        if (
            groundedness is not None
            and groundedness
            < self.config.min_groundedness_score
        ):

            warnings.append(
                "Answer groundedness is below "
                "the configured threshold."
            )

        success = True

        if (
            self.config.require_citations
            and citation_status
            == CitationStatus.INVALID
            and self.config.fail_on_invalid_citations
        ):

            success = False

        findings = self.extract_findings(
            answer,
            compressed,
            citations,
        )

        return RAGAgentResult(
            request_id=request_id,
            operation=request.operation,
            answer=answer,
            evidence=list(
                compressed
            ),
            citations=citations[
                : self.config.max_citations
            ],
            findings=findings,
            evidence_status=evidence_status,
            citation_status=citation_status,
            groundedness_score=groundedness,
            confidence=confidence,
            risk_level=self.infer_risk_level(
                confidence,
                evidence_status,
                citation_status,
            ),
            query=request.query,
            rewritten_query=query,
            retrieval_mode=retrieval_mode,
            warnings=warnings,
            success=success,
        )

    def generate_answer(
        self,
        query: str,
        evidence: Sequence[
            RAGEvidence
        ],
        request: RAGAgentRequest,
    ) -> str:

        if self.generator is None:

            return self.fallback_grounded_answer(
                query,
                evidence,
            )

        context = self.build_prompt_context(
            evidence
        )

        system_instruction = (
            "You are FinCo AI's financial "
            "document intelligence agent. "
            "Answer only from the supplied evidence. "
            "Do not invent facts, numbers, dates, "
            "document references or citations. "
            "If evidence is insufficient, explicitly "
            "state that it is insufficient. "
            "Distinguish retrieved facts from inference. "
            "Preserve financial precision."
        )

        payload = {
            "query": query,
            "context": context,
            "system_instruction": system_instruction,
            "company_id": request.company_id,
            "metadata": request.metadata,
        }

        try:

            result = self.invoke_component(
                self.generator,
                (
                    "generate",
                    "generate_answer",
                    "complete",
                    "invoke",
                    "run",
                ),
                payload,
            )

            answer = self.extract_text(
                result
            )

            if answer:

                return answer.strip()

        except Exception as exc:

            raise RAGGenerationError(
                "Grounded answer generation failed."
            ) from exc

        raise RAGGenerationError(
            "Generator returned an empty answer."
        )

    def generate_summary(
        self,
        query: str,
        evidence: Sequence[
            RAGEvidence
        ],
        request: RAGAgentRequest,
    ) -> str:

        summary_query = (
            "Summarize the relevant information "
            "from the supplied financial documents. "
            f"User request: {query}"
        )

        return self.generate_answer(
            summary_query,
            evidence,
            request,
        )

    def generate_extraction(
        self,
        query: str,
        evidence: Sequence[
            RAGEvidence
        ],
        request: RAGAgentRequest,
    ) -> str:

        extraction_query = (
            "Extract the information requested by "
            "the user from the supplied evidence. "
            "Do not infer missing values. "
            f"User request: {query}"
        )

        return self.generate_answer(
            extraction_query,
            evidence,
            request,
        )

    def generate_comparison(
        self,
        query: str,
        evidence: Sequence[
            RAGEvidence
        ],
        request: RAGAgentRequest,
    ) -> str:

        comparison_query = (
            "Compare the relevant information in "
            "the supplied evidence. Clearly identify "
            "which document or period supports each "
            f"claim. User request: {query}"
        )

        return self.generate_answer(
            comparison_query,
            evidence,
            request,
        )

    def fallback_grounded_answer(
        self,
        query: str,
        evidence: Sequence[
            RAGEvidence
        ],
    ) -> str:

        """
        Deterministic fallback.

        This is intentionally conservative and does not
        fabricate a synthesized financial conclusion.
        """

        if not evidence:

            return (
                "No relevant document evidence "
                "was found."
            )

        top = evidence[
            : min(3, len(evidence))
        ]

        parts = [
            (
                "Relevant document evidence "
                f"for '{query}':"
            )
        ]

        for item in top:

            label = (
                item.document_name
                or item.document_id
                or item.source
                or item.evidence_id
            )

            page = (
                f", page {item.page_number}"
                if item.page_number is not None
                else ""
            )

            parts.append(
                f"- {label}{page}: "
                f"{item.content}"
            )

        return "\n".join(
            parts
        )

    # ========================================================================
    # Citations
    # ========================================================================

    def generate_citations(
        self,
        answer: str,
        evidence: Sequence[
            RAGEvidence
        ],
        request: RAGAgentRequest,
    ) -> List[RAGCitation]:

        if not evidence:

            return []

        if (
            not self.config.use_citation_generator
            or self.citation_generator is None
        ):

            return self.build_default_citations(
                evidence
            )

        payload = {
            "answer": answer,
            "evidence": [
                item.to_dict()
                for item in evidence
            ],
            "max_citations": (
                self.config.max_citations
            ),
        }

        try:

            result = self.invoke_component(
                self.citation_generator,
                (
                    "generate",
                    "generate_citations",
                    "create",
                    "run",
                ),
                payload,
            )

            citations = self.normalize_citations(
                result
            )

            if citations:

                return citations[
                    : self.config.max_citations
                ]

        except Exception as exc:

            if self.config.require_citations:

                raise RAGCitationError(
                    "Citation generation failed."
                ) from exc

        return self.build_default_citations(
            evidence
        )

    def build_default_citations(
        self,
        evidence: Sequence[
            RAGEvidence
        ],
    ) -> List[RAGCitation]:

        citations: List[
            RAGCitation
        ] = []

        for item in evidence[
            : self.config.max_citations
        ]:

            citations.append(
                RAGCitation(
                    citation_id=(
                        "citation_"
                        + uuid.uuid4().hex[:12]
                    ),
                    evidence_id=(
                        item.evidence_id
                    ),
                    document_id=(
                        item.document_id
                    ),
                    document_name=(
                        item.document_name
                    ),
                    page_number=(
                        item.page_number
                    ),
                    chunk_id=(
                        item.chunk_id
                    ),
                    section=(
                        item.section
                    ),
                    quote=self.build_quote(
                        item.content
                    ),
                    status=CitationStatus.VALID,
                )
            )

        return citations

    def normalize_citations(
        self,
        raw: Any,
    ) -> List[RAGCitation]:

        if raw is None:

            return []

        if isinstance(
            raw,
            Mapping,
        ):

            for key in (
                "citations",
                "results",
                "items",
                "data",
            ):

                if key in raw:

                    raw = raw[key]
                    break

        if not isinstance(
            raw,
            Sequence,
        ) or isinstance(
            raw,
            (str, bytes),
        ):

            raw = [
                raw
            ]

        result: List[
            RAGCitation
        ] = []

        for item in raw:

            if isinstance(
                item,
                RAGCitation,
            ):

                result.append(
                    item
                )
                continue

            data = (
                dict(item)
                if isinstance(
                    item,
                    Mapping,
                )
                else self.object_to_dict(
                    item
                )
            )

            status = data.get(
                "status",
                CitationStatus.VALID.value,
            )

            try:

                citation_status = (
                    CitationStatus(
                        status
                    )
                )

            except ValueError:

                citation_status = (
                    CitationStatus.VALID
                )

            result.append(
                RAGCitation(
                    citation_id=str(
                        data.get(
                            "citation_id"
                        )
                        or data.get(
                            "id"
                        )
                        or (
                            "citation_"
                            + uuid.uuid4().hex[:12]
                        )
                    ),
                    evidence_id=(
                        self.optional_string(
                            data.get(
                                "evidence_id"
                            )
                        )
                    ),
                    document_id=(
                        self.optional_string(
                            data.get(
                                "document_id"
                            )
                        )
                    ),
                    document_name=(
                        self.optional_string(
                            data.get(
                                "document_name"
                            )
                        )
                    ),
                    page_number=(
                        self.optional_int(
                            data.get(
                                "page_number"
                            )
                        )
                    ),
                    chunk_id=(
                        self.optional_string(
                            data.get(
                                "chunk_id"
                            )
                        )
                    ),
                    section=(
                        self.optional_string(
                            data.get(
                                "section"
                            )
                        )
                    ),
                    quote=(
                        self.optional_string(
                            data.get(
                                "quote"
                            )
                            or data.get(
                                "text"
                            )
                        )
                    ),
                    status=citation_status,
                    metadata=dict(
                        data.get(
                            "metadata"
                        )
                        or {}
                    ),
                )
            )

        return result

    def validate_citations(
        self,
        answer: str,
        citations: Sequence[
            RAGCitation
        ],
        evidence: Sequence[
            RAGEvidence
        ],
    ) -> CitationStatus:

        if not citations:

            return CitationStatus.MISSING

        evidence_ids = {
            item.evidence_id
            for item in evidence
        }

        valid_count = 0

        for citation in citations:

            if (
                citation.evidence_id
                and citation.evidence_id
                in evidence_ids
            ):

                valid_count += 1

        if self.citation_validator is not None:

            payload = {
                "answer": answer,
                "citations": [
                    item.to_dict()
                    for item in citations
                ],
                "evidence": [
                    item.to_dict()
                    for item in evidence
                ],
            }

            try:

                result = self.invoke_component(
                    self.citation_validator,
                    (
                        "validate",
                        "validate_citations",
                        "check",
                        "run",
                    ),
                    payload,
                )

                normalized = self.normalize_status(
                    result
                )

                if normalized:

                    return normalized

            except Exception:

                pass

        if valid_count == len(
            citations
        ):

            return CitationStatus.VALID

        if valid_count > 0:

            return CitationStatus.PARTIAL

        return CitationStatus.INVALID

    # ========================================================================
    # Groundedness
    # ========================================================================

    def calculate_groundedness(
        self,
        answer: str,
        evidence: Sequence[
            RAGEvidence
        ],
        citations: Sequence[
            RAGCitation
        ],
    ) -> Optional[float]:

        if not self.config.enable_groundedness_check:

            return None

        if not answer or not evidence:

            return 0.0

        answer_tokens = (
            self.tokenize(
                answer
            )
        )

        if not answer_tokens:

            return 0.0

        evidence_tokens = set()

        for item in evidence:

            evidence_tokens.update(
                self.tokenize(
                    item.content
                )
            )

        overlap = (
            len(
                answer_tokens
                & evidence_tokens
            )
            / len(answer_tokens)
        )

        citation_factor = (
            min(
                1.0,
                len(citations)
                / max(
                    1,
                    min(
                        5,
                        len(evidence),
                    ),
                ),
            )
        )

        # Conservative lexical groundedness heuristic.
        # A dedicated evaluation module should replace this
        # with NLI/LLM-based groundedness when available.
        score = (
            0.75 * overlap
            + 0.25 * citation_factor
        )

        return max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

    # ========================================================================
    # Confidence / Risk
    # ========================================================================

    def calculate_confidence(
        self,
        evidence: Sequence[
            RAGEvidence
        ],
        groundedness: Optional[
            float
        ],
        citation_status: CitationStatus,
    ) -> float:

        if not evidence:

            return 0.0

        scores = [
            max(
                0.0,
                min(
                    1.0,
                    item.rerank_score
                    if item.rerank_score is not None
                    else item.score,
                ),
            )
            for item in evidence
        ]

        retrieval_confidence = (
            sum(scores)
            / len(scores)
        )

        groundedness_score = (
            groundedness
            if groundedness is not None
            else retrieval_confidence
        )

        citation_factor = {
            CitationStatus.VALID: 1.0,
            CitationStatus.PARTIAL: 0.75,
            CitationStatus.INVALID: 0.35,
            CitationStatus.MISSING: 0.25,
        }[
            citation_status
        ]

        confidence = (
            0.45 * retrieval_confidence
            + 0.40 * groundedness_score
            + 0.15 * citation_factor
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    def infer_risk_level(
        self,
        confidence: float,
        evidence_status: EvidenceStatus,
        citation_status: CitationStatus,
    ) -> RAGRiskLevel:

        if (
            evidence_status
            == EvidenceStatus.NONE
            or citation_status
            == CitationStatus.INVALID
        ):

            return RAGRiskLevel.CRITICAL

        if confidence < 0.40:

            return RAGRiskLevel.HIGH

        if (
            confidence < 0.65
            or evidence_status
            == EvidenceStatus.LIMITED
            or citation_status
            == CitationStatus.PARTIAL
        ):

            return RAGRiskLevel.MEDIUM

        return RAGRiskLevel.LOW

    # ========================================================================
    # Findings
    # ========================================================================

    def extract_findings(
        self,
        answer: str,
        evidence: Sequence[
            RAGEvidence
        ],
        citations: Sequence[
            RAGCitation
        ],
    ) -> List[RAGFinding]:

        if not answer:

            return []

        citation_ids = [
            citation.citation_id
            for citation in citations
        ]

        finding = RAGFinding(
            finding=answer,
            confidence=self.calculate_confidence(
                evidence,
                self.calculate_groundedness(
                    answer,
                    evidence,
                    citations,
                ),
                self.validate_citations(
                    answer,
                    citations,
                    evidence,
                ),
            ),
            evidence_ids=[
                item.evidence_id
                for item in evidence[
                    :5
                ]
            ],
            citations=citation_ids[
                :5
            ],
            risk_level=(
                RAGRiskLevel.LOW
            ),
        )

        return [
            finding
        ]

    # ========================================================================
    # Retrieval Result
    # ========================================================================

    def build_retrieval_result(
        self,
        request: RAGAgentRequest,
        request_id: str,
        query: str,
        evidence: Sequence[
            RAGEvidence
        ],
        evidence_status: EvidenceStatus,
        retrieval_mode: RetrievalMode,
    ) -> RAGAgentResult:

        citations = self.build_default_citations(
            evidence[
                : self.config.top_k
            ]
        )

        return RAGAgentResult(
            request_id=request_id,
            operation=request.operation,
            answer="",
            evidence=list(
                evidence[
                    : self.config.top_k
                ]
            ),
            citations=citations,
            evidence_status=evidence_status,
            citation_status=(
                CitationStatus.VALID
                if citations
                else CitationStatus.MISSING
            ),
            confidence=self.calculate_confidence(
                evidence,
                None,
                CitationStatus.VALID
                if citations
                else CitationStatus.MISSING,
            ),
            risk_level=(
                RAGRiskLevel.LOW
                if evidence_status
                == EvidenceStatus.SUFFICIENT
                else RAGRiskLevel.MEDIUM
            ),
            query=request.query,
            rewritten_query=query,
            retrieval_mode=retrieval_mode,
        )

    # ========================================================================
    # Prompt Context
    # ========================================================================

    def build_prompt_context(
        self,
        evidence: Sequence[
            RAGEvidence
        ],
    ) -> str:

        blocks: List[
            str
        ] = []

        for index, item in enumerate(
            evidence,
            start=1,
        ):

            document = (
                item.document_name
                or item.document_id
                or "Unknown document"
            )

            location = ""

            if item.page_number is not None:

                location = (
                    f", page {item.page_number}"
                )

            section = ""

            if item.section:

                section = (
                    f", section {item.section}"
                )

            blocks.append(
                (
                    f"[Evidence {index} | "
                    f"ID: {item.evidence_id} | "
                    f"{document}"
                    f"{location}"
                    f"{section}]\n"
                    f"{item.content}"
                )
            )

        return "\n\n".join(
            blocks
        )

    # ========================================================================
    # Guardrails / Authorization
    # ========================================================================

    def authorize(
        self,
        request: RAGAgentRequest,
    ) -> None:

        if self.authorization_service is None:

            return

        payload = {
            "user_id": request.user_id,
            "company_id": request.company_id,
            "document_ids": request.document_ids,
            "operation": request.operation.value,
            "metadata": request.metadata,
        }

        try:

            result = self.invoke_component(
                self.authorization_service,
                (
                    "authorize",
                    "check_permission",
                    "can_access",
                    "has_permission",
                    "check",
                ),
                payload,
            )

        except Exception as exc:

            raise RAGAgentAccessDeniedError(
                "Document authorization check failed."
            ) from exc

        if result is False:

            raise RAGAgentAccessDeniedError(
                "Access to requested document information "
                "was denied."
            )

        if isinstance(
            result,
            Mapping,
        ) and result.get(
            "allowed"
        ) is False:

            raise RAGAgentAccessDeniedError(
                "Access to requested document information "
                "was denied."
            )

    def check_guardrails(
        self,
        request: RAGAgentRequest,
    ) -> None:

        if self.guardrails is None:

            return

        payload = {
            "query": request.query,
            "operation": request.operation.value,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "metadata": request.metadata,
        }

        try:

            result = self.invoke_component(
                self.guardrails,
                (
                    "check",
                    "validate",
                    "evaluate",
                    "run",
                ),
                payload,
            )

        except Exception as exc:

            raise RAGGuardrailViolationError(
                "RAG input guardrail check failed."
            ) from exc

        if result is False:

            raise RAGGuardrailViolationError(
                "RAG request was blocked by guardrails."
            )

        if isinstance(
            result,
            Mapping,
        ):

            if result.get(
                "allowed"
            ) is False:

                raise RAGGuardrailViolationError(
                    str(
                        result.get(
                            "reason"
                        )
                        or "RAG request was blocked."
                    )
                )

    def check_output_guardrails(
        self,
        result: RAGAgentResult,
    ) -> None:

        if not result.answer:

            return

        # Never allow obvious fake citation markers
        # to pass as validated evidence.
        fake_markers = (
            "[source]",
            "[citation]",
            "citation needed",
            "<citation>",
        )

        lowered = result.answer.lower()

        if any(
            marker in lowered
            for marker in fake_markers
        ):

            result.warnings.append(
                "Generated answer contains an "
                "unresolved citation marker."
            )

    # ========================================================================
    # Audit
    # ========================================================================

    def audit(
        self,
        request: RAGAgentRequest,
        result: RAGAgentResult,
    ) -> None:

        if (
            not self.config.audit_enabled
            or self.audit_service is None
        ):

            return

        payload = {
            "event": "rag_agent_execution",
            "request_id": result.request_id,
            "operation": request.operation.value,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "success": result.success,
            "confidence": result.confidence,
            "risk_level": result.risk_level.value,
            "evidence_count": len(
                result.evidence
            ),
            "citation_count": len(
                result.citations
            ),
        }

        try:

            self.invoke_component(
                self.audit_service,
                (
                    "record",
                    "log",
                    "create",
                    "write",
                ),
                payload,
            )

        except Exception:
            # Audit failure must not expose internal details
            # or destroy the user's RAG response.
            pass

    # ========================================================================
    # Request Validation
    # ========================================================================

    def normalize_request(
        self,
        request: RAGAgentRequest
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> RAGAgentRequest:

        if isinstance(
            request,
            RAGAgentRequest,
        ):

            return request

        if not isinstance(
            request,
            Mapping,
        ):

            raise InvalidRAGRequestError(
                "Request must be a RAGAgentRequest "
                "or mapping."
            )

        payload = dict(
            request
        )

        payload.update(
            kwargs
        )

        if "operation" in payload:

            try:

                payload["operation"] = (
                    RAGOperation(
                        payload["operation"]
                    )
                )

            except ValueError as exc:

                raise InvalidRAGRequestError(
                    "Unsupported RAG operation."
                ) from exc

        if "retrieval_mode" in payload:

            value = payload[
                "retrieval_mode"
            ]

            if value is not None:

                try:

                    payload["retrieval_mode"] = (
                        RetrievalMode(
                            value
                        )
                    )

                except ValueError as exc:

                    raise InvalidRAGRequestError(
                        "Unsupported retrieval mode."
                    ) from exc

        return RAGAgentRequest(
            **payload
        )

    def validate_request(
        self,
        request: RAGAgentRequest,
    ) -> None:

        if not self.config.enabled:

            raise RAGAgentConfigurationError(
                "RAG agent is disabled."
            )

        if not isinstance(
            request.query,
            str,
        ):

            raise InvalidRAGRequestError(
                "Query must be a string."
            )

        request.query = (
            request.query.strip()
        )

        if not request.query:

            raise InvalidRAGRequestError(
                "Query cannot be empty."
            )

        if len(
            request.query
        ) > self.config.max_query_length:

            raise InvalidRAGRequestError(
                "Query exceeds maximum length."
            )

        if request.top_k is not None:

            if request.top_k < 1:

                raise InvalidRAGRequestError(
                    "top_k must be positive."
                )

            if request.top_k > (
                self.config.retrieval_k
            ):

                request.top_k = (
                    self.config.retrieval_k
                )

    # ========================================================================
    # Finalization
    # ========================================================================

    def finalize_result(
        self,
        result: RAGAgentResult,
        started: float,
    ) -> None:

        result.execution_time_ms = (
            time.perf_counter()
            - started
        ) * 1000

        if (
            result.execution_time_ms
            > self.config.timeout_seconds * 1000
        ):

            result.warnings.append(
                "RAG execution exceeded the "
                "configured latency target."
            )

    # ========================================================================
    # Generic Component Invocation
    # ========================================================================

    def invoke_component(
        self,
        component: Any,
        methods: Sequence[str],
        payload: Mapping[str, Any],
    ) -> Any:

        if component is None:

            raise RAGAgentConfigurationError(
                "Required component is not configured."
            )

        # Callable components.
        if callable(component):

            return self.invoke_callable(
                component,
                payload,
            )

        for method_name in methods:

            method = getattr(
                component,
                method_name,
                None,
            )

            if method is None:
                continue

            if not callable(method):
                continue

            return self.invoke_callable(
                method,
                payload,
            )

        raise RAGAgentConfigurationError(
            "Configured component does not expose "
            "a supported interface."
        )

    @staticmethod
    def invoke_callable(
        callable_obj: Callable[..., Any],
        payload: Mapping[str, Any],
    ) -> Any:

        """
        Adapt to common dependency interfaces.

        Supported patterns:
        - fn(**payload)
        - fn(payload)
        """

        try:

            signature = inspect.signature(
                callable_obj
            )

            parameters = signature.parameters

            accepts_kwargs = any(
                parameter.kind
                == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )

            if accepts_kwargs:

                return callable_obj(
                    **dict(payload)
                )

            accepted = {
                name: value
                for name, value in payload.items()
                if name in parameters
            }

            if accepted:

                return callable_obj(
                    **accepted
                )

        except (
            TypeError,
            ValueError,
        ):

            pass

        return callable_obj(
            payload
        )

    # ========================================================================
    # Generic Helpers
    # ========================================================================

    @staticmethod
    def extract_text(
        result: Any,
    ) -> str:

        if result is None:

            return ""

        if isinstance(
            result,
            str,
        ):

            return result

        if isinstance(
            result,
            Mapping,
        ):

            for key in (
                "answer",
                "text",
                "content",
                "response",
                "output",
                "result",
            ):

                value = result.get(
                    key
                )

                if isinstance(
                    value,
                    str,
                ):

                    return value

        for attribute in (
            "answer",
            "text",
            "content",
            "response",
            "output",
            "result",
        ):

            value = getattr(
                result,
                attribute,
                None,
            )

            if isinstance(
                value,
                str,
            ):

                return value

        return str(
            result
        )

    @staticmethod
    def object_to_dict(
        obj: Any,
    ) -> Dict[str, Any]:

        if isinstance(
            obj,
            Mapping,
        ):

            return dict(
                obj
            )

        if hasattr(
            obj,
            "model_dump",
        ):

            try:

                return dict(
                    obj.model_dump()
                )

            except Exception:
                pass

        if hasattr(
            obj,
            "dict",
        ):

            try:

                return dict(
                    obj.dict()
                )

            except Exception:
                pass

        if hasattr(
            obj,
            "__dict__",
        ):

            return dict(
                obj.__dict__
            )

        return {}

    @staticmethod
    def safe_float(
        value: Any,
    ) -> float:

        try:

            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0.0

    @staticmethod
    def safe_int(
        value: Any,
    ) -> int:

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return 0

    @staticmethod
    def optional_string(
        value: Any,
    ) -> Optional[str]:

        if value is None:

            return None

        return str(
            value
        )

    @staticmethod
    def optional_int(
        value: Any,
    ) -> Optional[int]:

        if value is None:

            return None

        try:

            return int(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

    @staticmethod
    def build_quote(
        text: str,
        max_length: int = 500,
    ) -> str:

        normalized = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        if len(
            normalized
        ) <= max_length:

            return normalized

        return (
            normalized[
                : max_length - 3
            ]
            + "..."
        )

    @staticmethod
    def tokenize(
        text: str,
    ) -> set[str]:

        return {
            token
            for token in re.findall(
                r"\b[a-zA-Z0-9_]+\b",
                text.lower(),
            )
            if len(token) > 2
        }

    @staticmethod
    def normalize_status(
        result: Any,
    ) -> Optional[
        CitationStatus
    ]:

        if isinstance(
            result,
            CitationStatus,
        ):

            return result

        if isinstance(
            result,
            str,
        ):

            try:

                return CitationStatus(
                    result.lower()
                )

            except ValueError:
                return None

        if isinstance(
            result,
            Mapping,
        ):

            value = (
                result.get(
                    "status"
                )
                or result.get(
                    "citation_status"
                )
            )

            if value:

                try:

                    return CitationStatus(
                        str(
                            value
                        ).lower()
                    )

                except ValueError:
                    return None

        return None

    @staticmethod
    def generate_request_id() -> str:

        return (
            "rag_"
            + uuid.uuid4().hex[:20]
        )


# ============================================================================
# Serialization
# ============================================================================


def _serialize(
    value: Any,
) -> Any:

    if isinstance(
        value,
        Enum,
    ):

        return value.value

    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(key): _serialize(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):

        return [
            _serialize(
                item
            )
            for item in value
        ]

    if hasattr(
        value,
        "model_dump",
    ):

        try:

            return _serialize(
                value.model_dump()
            )

        except Exception:
            pass

    if hasattr(
        value,
        "to_dict",
    ):

        try:

            return _serialize(
                value.to_dict()
            )

        except Exception:
            pass

    return value


# ============================================================================
# Convenience API
# ============================================================================


def create_rag_agent(
    retrieval_service: Any,
    generator: Optional[
        Any
    ] = None,
    **kwargs: Any,
) -> RAGAgent:

    """
    Factory for dependency-injected RAG agents.
    """

    return RAGAgent(
        retrieval_service=retrieval_service,
        generator=generator,
        **kwargs,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "RAGAgentError",
    "InvalidRAGRequestError",
    "RAGAgentConfigurationError",
    "RAGAgentExecutionError",
    "RAGAgentAccessDeniedError",
    "RAGRetrievalError",
    "RAGGenerationError",
    "RAGCitationError",
    "RAGGuardrailViolationError",
    "RAGInsufficientEvidenceError",

    # Enums
    "RAGOperation",
    "RetrievalMode",
    "RAGRiskLevel",
    "EvidenceStatus",
    "CitationStatus",

    # Configuration
    "RAGAgentConfig",

    # Models
    "RAGAgentRequest",
    "RAGEvidence",
    "RAGCitation",
    "RAGFinding",
    "RAGAgentResult",

    # Agent
    "RAGAgent",

    # Factory
    "create_rag_agent",
]