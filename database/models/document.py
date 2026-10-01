"""
backend/app/api/document.py

Document Management API for FinCo AI.

Responsibilities
----------------
- Document metadata management
- Document retrieval
- Document listing/search
- Processing status
- Document access control
- Archive / restore
- Document statistics
- RAG-ready document metadata
- Tenant isolation

Architecture
------------

    Client
       |
       v
    Document API
       |
       +---- Authentication
       +---- Authorization
       +---- Tenant Isolation
       |
       v
    DocumentService
       |
       +---- DocumentRepository
       +---- Ingestion Pipeline
       +---- RAG / Vector Store
       +---- AuditService
       |
       v
    PostgreSQL + Object Storage + Vector DB
"""

from __future__ import annotations

import inspect
import logging
import time
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# ============================================================================
# TYPES
# ============================================================================

DocumentStatus = Literal[
    "uploaded",
    "queued",
    "processing",
    "processed",
    "failed",
    "archived",
]

DocumentType = Literal[
    "financial_statement",
    "income_statement",
    "balance_sheet",
    "cash_flow_statement",
    "bank_statement",
    "invoice",
    "receipt",
    "transaction_data",
    "budget",
    "forecast",
    "tax_document",
    "contract",
    "annual_report",
    "management_report",
    "other",
]

DocumentFormat = Literal[
    "pdf",
    "csv",
    "xlsx",
    "xls",
    "docx",
    "doc",
    "txt",
    "json",
    "image",
    "other",
]

AccessLevel = Literal[
    "private",
    "company",
    "finance",
    "management",
    "auditor",
]

ProcessingStage = Literal[
    "uploaded",
    "classified",
    "parsed",
    "normalized",
    "chunked",
    "embedded",
    "indexed",
    "completed",
    "failed",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class DocumentCreateRequest(BaseModel):
    """
    Register a document in the document system.

    Binary upload itself should normally be handled by the upload API.
    This endpoint is intended for metadata registration.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )

    document_type: DocumentType = "other"

    document_format: DocumentFormat = "other"

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    access_level: AccessLevel = "company"

    fiscal_year: int | None = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    reporting_period: str | None = Field(
        default=None,
        max_length=100,
    )

    source: str | None = Field(
        default=None,
        max_length=500,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Document name cannot be empty."
            )

        return value


class DocumentUpdateRequest(BaseModel):
    """
    Update document metadata.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
    )

    document_type: DocumentType | None = None

    description: str | None = Field(
        default=None,
        max_length=2000,
    )

    access_level: AccessLevel | None = None

    fiscal_year: int | None = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    reporting_period: str | None = Field(
        default=None,
        max_length=100,
    )

    source: str | None = Field(
        default=None,
        max_length=500,
    )

    metadata: dict[str, Any] | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError(
                "Document name cannot be empty."
            )

        return value


class DocumentAccessRequest(BaseModel):
    """
    Change document access level.
    """

    model_config = ConfigDict(extra="forbid")

    access_level: AccessLevel

    reason: str | None = Field(
        default=None,
        max_length=2000,
    )


class DocumentProcessingRequest(BaseModel):
    """
    Request document processing/reprocessing.
    """

    model_config = ConfigDict(extra="forbid")

    force_reprocess: bool = False

    rebuild_embeddings: bool = False

    rebuild_index: bool = False


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class DocumentResponse(BaseModel):
    """
    Document metadata response.
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
    )

    id: str

    company_id: str

    name: str

    document_type: DocumentType

    document_format: DocumentFormat

    description: str | None = None

    status: DocumentStatus

    access_level: AccessLevel

    source: str | None = None

    file_size_bytes: int | None = None

    checksum: str | None = None

    mime_type: str | None = None

    page_count: int | None = None

    chunk_count: int | None = None

    embedding_count: int | None = None

    fiscal_year: int | None = None

    reporting_period: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: datetime

    updated_at: datetime


class DocumentListResponse(BaseModel):
    """
    Paginated document response.
    """

    items: list[DocumentResponse]

    total: int

    page: int

    page_size: int

    pages: int


class DocumentProcessingResponse(BaseModel):
    """
    Document processing status.
    """

    document_id: str

    status: DocumentStatus

    stage: ProcessingStage

    progress_percent: float = Field(
        ge=0,
        le=100,
    )

    message: str | None = None

    chunks_created: int = 0

    embeddings_created: int = 0

    error: str | None = None

    started_at: datetime | None = None

    completed_at: datetime | None = None


class DocumentAccessResponse(BaseModel):
    """
    Document access-level update result.
    """

    document_id: str

    access_level: AccessLevel

    message: str

    updated_at: datetime


class DocumentArchiveResponse(BaseModel):
    """
    Archive result.

    Documents are archived instead of physically deleted.
    """

    document_id: str

    status: Literal["archived"]

    message: str

    updated_at: datetime


class DocumentRestoreResponse(BaseModel):
    """
    Restore result.
    """

    document_id: str

    status: Literal[
        "uploaded",
        "processed",
    ]

    message: str

    updated_at: datetime


class DocumentStatisticsResponse(BaseModel):
    """
    Tenant-level document statistics.
    """

    company_id: str

    total_documents: int

    processed_documents: int

    processing_documents: int

    failed_documents: int

    archived_documents: int

    total_size_bytes: int

    total_chunks: int

    total_embeddings: int

    by_type: dict[str, int] = Field(
        default_factory=dict,
    )

    by_format: dict[str, int] = Field(
        default_factory=dict,
    )


class DocumentHealthResponse(BaseModel):
    """
    Document subsystem health.
    """

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    service: str

    timestamp: datetime

    details: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# AUTHENTICATION
# ============================================================================


def get_current_user() -> Any:
    """
    Lazily import authentication dependency.
    """

    try:
        from backend.app.dependencies import (
            get_current_user as dependency,
        )

        return dependency

    except ImportError as exc:
        logger.exception(
            "Authentication dependency unavailable."
        )

        raise RuntimeError(
            "Authentication dependency is not configured."
        ) from exc


# ============================================================================
# SERVICES
# ============================================================================


def get_document_service() -> Any:
    """
    Preferred service:

        backend.app.documents.document_service.DocumentService

    Fallback:

        backend.app.database.repositories.document_repository.DocumentRepository
    """

    try:
        from backend.app.documents.document_service import (
            DocumentService,
        )

        return DocumentService()

    except ImportError:
        pass

    try:
        from backend.app.database.repositories.document_repository import (
            DocumentRepository,
        )

        return DocumentRepository()

    except ImportError as exc:
        logger.exception(
            "Document service/repository unavailable."
        )

        raise RuntimeError(
            "Document service is not configured."
        ) from exc


# ============================================================================
# SECURITY HELPERS
# ============================================================================


def _get_user_id(
    current_user: Any,
) -> str | None:
    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return (
            current_user.get("id")
            or current_user.get("user_id")
        )

    return (
        getattr(current_user, "id", None)
        or getattr(
            current_user,
            "user_id",
            None,
        )
    )


def _get_company_id(
    current_user: Any,
) -> str | None:
    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("company_id")

    return getattr(
        current_user,
        "company_id",
        None,
    )


def _get_role(
    current_user: Any,
) -> str | None:
    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("role")

    return getattr(
        current_user,
        "role",
        None,
    )


def _is_admin(
    current_user: Any,
) -> bool:
    if current_user is None:
        return False

    if isinstance(current_user, dict):
        return bool(
            current_user.get(
                "is_admin",
                False,
            )
        )

    return bool(
        getattr(
            current_user,
            "is_admin",
            False,
        )
    )


def _can_manage_documents(
    current_user: Any,
) -> bool:
    """
    Roles permitted to modify document metadata,
    processing state, and lifecycle.
    """

    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
        "finance_manager",
        "analyst",
    }


def _can_manage_access(
    current_user: Any,
) -> bool:
    """
    More restrictive permission for document access changes.
    """

    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
        "finance_manager",
    }


def require_document_manager(
    current_user: Any,
) -> None:
    if not _can_manage_documents(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Document management privileges "
                "are required."
            ),
        )


def require_access_manager(
    current_user: Any,
) -> None:
    if not _can_manage_access(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Document access-management "
                "privileges are required."
            ),
        )


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce tenant isolation.

    Admin users may access multiple companies only if
    the centralized authorization layer grants them that
    privilege.
    """

    if _is_admin(current_user):
        return

    user_company_id = _get_company_id(
        current_user
    )

    if not user_company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "User is not associated with a company."
            ),
        )

    if str(user_company_id) != str(company_id):
        logger.warning(
            "Unauthorized document company access: "
            "user_company=%s requested_company=%s "
            "user=%s",
            user_company_id,
            company_id,
            _get_user_id(current_user),
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this company."
            ),
        )


# ============================================================================
# SERVICE CALLER
# ============================================================================


async def _call_service(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:
    """
    Invoke synchronous or asynchronous service methods.
    """

    method = getattr(
        service,
        method_name,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Document service does not implement "
                f"'{method_name}'."
            ),
        )

    try:
        result = method(**kwargs)

        if inspect.isawaitable(result):
            result = await result

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid document request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Document service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document operation failed.",
        ) from exc


# ============================================================================
# CREATE METADATA
# ============================================================================


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create document metadata",
)
async def create_document(
    request: DocumentCreateRequest,
    current_user: Any = Depends(get_current_user()),
) -> DocumentResponse:
    """
    Register document metadata.

    Actual binary uploads should normally flow through
    /uploads and the ingestion pipeline.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_document_manager(
        current_user
    )

    service = get_document_service()

    result = await _call_service(
        service,
        "create",
        company_id=company_id,
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Document creation returned no result."
            ),
        )

    return DocumentResponse.model_validate(
        result
    )


# ============================================================================
# LIST
# ============================================================================


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents",
)
async def list_documents(
    document_type: DocumentType | None = None,
    document_status: DocumentStatus | None = Query(
        default=None,
        alias="status",
    ),
    document_format: DocumentFormat | None = None,
    access_level: AccessLevel | None = None,
    search: str | None = Query(
        default=None,
        min_length=1,
        max_length=200,
    ),
    fiscal_year: int | None = Query(
        default=None,
        ge=1900,
        le=2200,
    ),
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    current_user: Any = Depends(get_current_user()),
) -> DocumentListResponse:
    """
    List documents belonging to the authenticated tenant.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_document_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        document_type=document_type,
        status=document_status,
        document_format=document_format,
        access_level=access_level,
        search=search.strip()
        if search
        else None,
        fiscal_year=fiscal_year,
        page=page,
        page_size=page_size,
        user=current_user,
    )

    if result is None:
        return DocumentListResponse(
            items=[],
            total=0,
            page=page,
            page_size=page_size,
            pages=0,
        )

    return DocumentListResponse.model_validate(
        result
    )


# ============================================================================
# SEARCH
# ============================================================================


@router.get(
    "/search",
    response_model=DocumentListResponse,
    summary="Search documents",
)
async def search_documents(
    q: str = Query(
        ...,
        min_length=2,
        max_length=200,
    ),
    document_type: DocumentType | None = None,
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    current_user: Any = Depends(get_current_user()),
) -> DocumentListResponse:
    """
    Metadata search.

    This is not the semantic RAG search endpoint.
    Semantic retrieval belongs in the RAG retrieval service.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_document_service()

    result = await _call_service(
        service,
        "search",
        company_id=company_id,
        query=q.strip(),
        document_type=document_type,
        page=page,
        page_size=page_size,
        user=current_user,
    )

    if result is None:
        return DocumentListResponse(
            items=[],
            total=0,
            page=page,
            page_size=page_size,
            pages=0,
        )

    return DocumentListResponse.model_validate(
        result
    )


# ============================================================================
# GET
# ============================================================================


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document",
)
async def get_document(
    document_id: str,
    current_user: Any = Depends(get_current_user()),
) -> DocumentResponse:
    """
    Get document metadata.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_document_service()

    result = await _call_service(
        service,
        "get",
        document_id=document_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentResponse.model_validate(
        result
    )


# ============================================================================
# UPDATE
# ============================================================================


@router.patch(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Update document metadata",
)
async def update_document(
    document_id: str,
    request: DocumentUpdateRequest,
    current_user: Any = Depends(get_current_user()),
) -> DocumentResponse:
    """
    Update document metadata.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_document_manager(
        current_user
    )

    service = get_document_service()

    result = await _call_service(
        service,
        "update",
        document_id=document_id,
        company_id=company_id,
        updates=request.model_dump(
            exclude_unset=True
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentResponse.model_validate(
        result
    )


# ============================================================================
# PROCESSING STATUS
# ============================================================================


@router.get(
    "/{document_id}/processing",
    response_model=DocumentProcessingResponse,
    summary="Get document processing status",
)
async def get_processing_status(
    document_id: str,
    current_user: Any = Depends(get_current_user()),
) -> DocumentProcessingResponse:
    """
    Return ingestion/RAG processing state.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_document_service()

    result = await _call_service(
        service,
        "processing_status",
        document_id=document_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentProcessingResponse.model_validate(
        result
    )


# ============================================================================
# REPROCESS
# ============================================================================


@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessingResponse,
    summary="Process or reprocess document",
)
async def process_document(
    document_id: str,
    request: DocumentProcessingRequest,
    current_user: Any = Depends(get_current_user()),
) -> DocumentProcessingResponse:
    """
    Start document ingestion/reprocessing.

    Heavy processing should be asynchronous and handled by
    a worker/job queue rather than blocking the HTTP request.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_document_manager(
        current_user
    )

    service = get_document_service()

    result = await _call_service(
        service,
        "process",
        document_id=document_id,
        company_id=company_id,
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentProcessingResponse.model_validate(
        result
    )


# ============================================================================
# ACCESS CONTROL
# ============================================================================


@router.patch(
    "/{document_id}/access",
    response_model=DocumentAccessResponse,
    summary="Change document access level",
)
async def update_document_access(
    document_id: str,
    request: DocumentAccessRequest,
    current_user: Any = Depends(get_current_user()),
) -> DocumentAccessResponse:
    """
    Change document visibility/access level.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_access_manager(
        current_user
    )

    service = get_document_service()

    result = await _call_service(
        service,
        "update_access",
        document_id=document_id,
        company_id=company_id,
        access_level=request.access_level,
        reason=request.reason,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentAccessResponse.model_validate(
        result
    )


# ============================================================================
# ARCHIVE
# ============================================================================


@router.delete(
    "/{document_id}",
    response_model=DocumentArchiveResponse,
    summary="Archive document",
)
async def archive_document(
    document_id: str,
    reason: str | None = Query(
        default=None,
        max_length=2000,
    ),
    current_user: Any = Depends(get_current_user()),
) -> DocumentArchiveResponse:
    """
    Archive a document.

    Does not physically delete:
    - source object
    - document metadata
    - audit history
    - financial references
    - RAG provenance

    Cleanup can be handled asynchronously according to
    retention policies.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_document_manager(
        current_user
    )

    service = get_document_service()

    result = await _call_service(
        service,
        "archive",
        document_id=document_id,
        company_id=company_id,
        reason=reason,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentArchiveResponse.model_validate(
        result
    )


# ============================================================================
# RESTORE
# ============================================================================


@router.post(
    "/{document_id}/restore",
    response_model=DocumentRestoreResponse,
    summary="Restore archived document",
)
async def restore_document(
    document_id: str,
    current_user: Any = Depends(get_current_user()),
) -> DocumentRestoreResponse:
    """
    Restore an archived document.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    require_document_manager(
        current_user
    )

    service = get_document_service()

    result = await _call_service(
        service,
        "restore",
        document_id=document_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentRestoreResponse.model_validate(
        result
    )


# ============================================================================
# STATISTICS
# ============================================================================


@router.get(
    "/statistics/summary",
    response_model=DocumentStatisticsResponse,
    summary="Get document statistics",
)
async def document_statistics(
    current_user: Any = Depends(get_current_user()),
) -> DocumentStatisticsResponse:
    """
    Get tenant-level document statistics.
    """

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    service = get_document_service()

    result = await _call_service(
        service,
        "statistics",
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        return DocumentStatisticsResponse(
            company_id=str(company_id),
            total_documents=0,
            processed_documents=0,
            processing_documents=0,
            failed_documents=0,
            archived_documents=0,
            total_size_bytes=0,
            total_chunks=0,
            total_embeddings=0,
        )

    return DocumentStatisticsResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=DocumentHealthResponse,
    summary="Document service health",
)
async def document_health() -> DocumentHealthResponse:
    """
    Document service health check.
    """

    started = time.perf_counter()

    try:
        service = get_document_service()

        details: dict[str, Any] = {}

        health_method = getattr(
            service,
            "health",
            None,
        )

        if health_method is not None:
            result = health_method()

            if inspect.isawaitable(result):
                result = await result

            if isinstance(result, dict):
                details = result

        details["latency_ms"] = round(
            (
                time.perf_counter() - started
            ) * 1000,
            2,
        )

        return DocumentHealthResponse(
            status="healthy",
            service="document",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Document service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service is unavailable.",
        )