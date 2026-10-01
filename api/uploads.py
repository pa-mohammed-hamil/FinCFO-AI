"""
backend/app/api/uploads.py

Secure Upload API for FinCo AI.

Responsibilities
----------------
- Authentication / authorization
- Tenant isolation
- Upload validation
- File type validation
- File-size validation
- Safe filename handling
- Storage-service handoff
- Upload metadata
- Ingestion pipeline triggering
- Upload status tracking
- Health checks

Architecture
------------

    Client
       |
       v
    Upload API
       |
       +---- Authentication
       +---- Authorization
       +---- Validation
       |
       v
    Storage Service
       |
       v
    Ingestion Pipeline
       |
       +---- File Classification
       +---- PDF / DOCX / XLSX / CSV
       +---- OCR
       +---- Table Extraction
       +---- Normalization
       +---- Validation
       +---- Duplicate Detection
       |
       v
    Financial / RAG Pipelines
"""

from __future__ import annotations

import inspect
import logging
import mimetypes
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/uploads",
    tags=["Uploads"],
)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".csv",
    ".xlsx",
    ".xls",
    ".docx",
    ".doc",
    ".txt",
    ".json",
}

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    (
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    ),
    "application/msword",
    "text/plain",
    "application/json",
}

UploadStatus = Literal[
    "uploaded",
    "queued",
    "processing",
    "completed",
    "failed",
    "rejected",
]

UploadDocumentType = Literal[
    "financial_statement",
    "bank_statement",
    "invoice",
    "receipt",
    "transaction_data",
    "budget",
    "forecast",
    "tax_document",
    "contract",
    "other",
]


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class UploadResponse(BaseModel):
    """
    Upload metadata returned after successful upload.
    """

    model_config = ConfigDict(extra="forbid")

    upload_id: str

    company_id: str

    filename: str

    file_extension: str

    mime_type: str | None = None

    size_bytes: int

    document_type: UploadDocumentType

    status: UploadStatus

    ingestion_requested: bool

    ingestion_job_id: str | None = None

    duplicate_detected: bool = False

    checksum: str | None = None

    uploaded_at: datetime


class UploadStatusResponse(BaseModel):
    """
    Current processing status of an upload.
    """

    model_config = ConfigDict(extra="forbid")

    upload_id: str

    company_id: str

    filename: str

    status: UploadStatus

    progress_percentage: float = Field(
        ge=0.0,
        le=100.0,
    )

    current_stage: str | None = None

    ingestion_job_id: str | None = None

    records_processed: int = 0

    records_failed: int = 0

    chunks_created: int = 0

    errors: list[str] = Field(
        default_factory=list,
    )

    started_at: datetime | None = None

    completed_at: datetime | None = None

    updated_at: datetime


class UploadListResponse(BaseModel):
    """
    Paginated upload history.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str

    items: list[UploadResponse]

    total: int

    limit: int

    offset: int

    has_more: bool


class UploadCancelResponse(BaseModel):
    """
    Upload cancellation result.
    """

    upload_id: str

    company_id: str

    status: Literal[
        "cancelled",
        "already_completed",
        "already_failed",
        "not_found",
    ]

    message: str


class UploadHealthResponse(BaseModel):
    """
    Upload subsystem health.
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
    Lazily load authentication dependency.
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


def get_upload_service() -> Any:
    """
    Lazily construct UploadService.

    Expected implementation:

        backend.app.ingestion.upload_service.UploadService
    """

    try:
        from backend.app.ingestion.upload_service import (
            UploadService,
        )

        return UploadService()

    except ImportError:
        pass

    try:
        from backend.app.ingestion.pipeline import (
            IngestionPipeline,
        )

        return IngestionPipeline()

    except ImportError as exc:
        logger.exception(
            "Upload/Ingestion service unavailable."
        )

        raise RuntimeError(
            "Upload service is not configured."
        ) from exc


# ============================================================================
# SECURITY
# ============================================================================


def _get_user_company_id(
    current_user: Any,
) -> str | None:
    """
    Extract company ID from authenticated user.
    """

    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("company_id")

    return getattr(
        current_user,
        "company_id",
        None,
    )


def _is_admin(
    current_user: Any,
) -> bool:
    """
    Determine administrative access.
    """

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


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce tenant isolation.
    """

    if _is_admin(current_user):
        return

    user_company_id = _get_user_company_id(
        current_user
    )

    if not user_company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "User is not associated with a company."
            ),
        )

    if user_company_id != company_id:
        logger.warning(
            "Unauthorized upload access attempt: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this "
                "company's uploads."
            ),
        )


# ============================================================================
# FILE VALIDATION
# ============================================================================


def sanitize_filename(
    filename: str,
) -> str:
    """
    Sanitize a client-provided filename.

    Prevents path traversal and removes unsafe path
    components. The original filename should not be used
    as a storage path.
    """

    filename = filename.strip()

    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    filename = Path(filename).name

    if filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    return filename


def get_extension(
    filename: str,
) -> str:
    """
    Return normalized file extension.
    """

    extension = Path(filename).suffix.lower()

    if not extension:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "File extension is required."
            ),
        )

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: {extension}. "
                f"Allowed types: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    return extension


def validate_mime_type(
    mime_type: str | None,
) -> None:
    """
    Validate declared MIME type.

    MIME type alone is not trusted for security. The storage
    and ingestion layers should additionally inspect file
    signatures/content.
    """

    if not mime_type:
        return

    normalized = mime_type.lower().strip()

    if normalized not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported MIME type: {mime_type}."
            ),
        )


def infer_mime_type(
    filename: str,
) -> str | None:
    """
    Infer MIME type from filename when necessary.
    """

    mime_type, _ = mimetypes.guess_type(filename)

    return mime_type


async def read_upload_safely(
    file: UploadFile,
    max_size: int = MAX_UPLOAD_SIZE,
) -> bytes:
    """
    Read an uploaded file while enforcing a size limit.

    Reads in chunks instead of blindly loading arbitrarily
    large files into memory.
    """

    chunks: list[bytes] = []
    total_size = 0

    while True:
        chunk = await file.read(1024 * 1024)

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File exceeds maximum size of "
                    f"{max_size // (1024 * 1024)} MB."
                ),
            )

        chunks.append(chunk)

    return b"".join(chunks)


# ============================================================================
# DOCUMENT TYPE
# ============================================================================


def normalize_document_type(
    document_type: str | None,
) -> UploadDocumentType:
    """
    Normalize document type supplied by the client.
    """

    allowed = {
        "financial_statement",
        "bank_statement",
        "invoice",
        "receipt",
        "transaction_data",
        "budget",
        "forecast",
        "tax_document",
        "contract",
        "other",
    }

    if not document_type:
        return "other"

    normalized = document_type.lower().strip()

    if normalized not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported document_type: "
                f"{document_type}."
            ),
        )

    return normalized  # type: ignore[return-value]


# ============================================================================
# SERVICE CALLER
# ============================================================================


async def _call_service(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:
    """
    Invoke service method.

    Supports synchronous and asynchronous implementations.
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
                f"Upload service does not implement "
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
            "Invalid upload request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upload request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Upload service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload operation failed.",
        ) from exc


# ============================================================================
# UPLOAD FILE
# ============================================================================


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload financial document",
)
async def upload_file(
    company_id: str = Form(...),
    document_type: str = Form(default="other"),
    ingest: bool = Form(default=True),
    file: UploadFile = File(...),
    current_user: Any = Depends(get_current_user()),
) -> UploadResponse:
    """
    Upload a financial document.

    Supported examples:

        PDF
        CSV
        XLSX
        XLS
        DOCX
        DOC
        TXT
        JSON

    The API validates the request and passes the file to the
    storage/ingestion layer.

    It does NOT:
        - parse PDFs
        - perform OCR
        - calculate financial metrics
        - generate embeddings
        - execute LLM calls
    """

    validate_company_access(
        company_id,
        current_user,
    )

    filename = sanitize_filename(
        file.filename or ""
    )

    extension = get_extension(
        filename
    )

    declared_mime = (
        file.content_type
        or infer_mime_type(filename)
    )

    validate_mime_type(
        declared_mime
    )

    normalized_document_type = (
        normalize_document_type(
            document_type
        )
    )

    contents = await read_upload_safely(
        file
    )

    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    upload_id = str(uuid.uuid4())

    service = get_upload_service()

    result = await _call_service(
        service,
        "upload",
        upload_id=upload_id,
        company_id=company_id,
        filename=filename,
        extension=extension,
        mime_type=declared_mime,
        content=contents,
        size_bytes=len(contents),
        document_type=normalized_document_type,
        ingest=ingest,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed.",
        )

    if isinstance(
        result,
        UploadResponse,
    ):
        return result

    return UploadResponse.model_validate(
        result
    )


# ============================================================================
# UPLOAD STATUS
# ============================================================================


@router.get(
    "/{upload_id}/status",
    response_model=UploadStatusResponse,
    summary="Get upload processing status",
)
async def get_upload_status(
    upload_id: str,
    company_id: str = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> UploadStatusResponse:
    """
    Get current ingestion status.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_upload_service()

    result = await _call_service(
        service,
        "get_status",
        upload_id=upload_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload not found.",
        )

    if isinstance(
        result,
        UploadStatusResponse,
    ):
        return result

    return UploadStatusResponse.model_validate(
        result
    )


# ============================================================================
# LIST UPLOADS
# ============================================================================


@router.get(
    "",
    response_model=UploadListResponse,
    summary="List company uploads",
)
async def list_uploads(
    company_id: str = Query(...),
    upload_status: UploadStatus | None = Query(
        default=None,
        alias="status",
    ),
    document_type: UploadDocumentType | None = Query(
        default=None,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    current_user: Any = Depends(get_current_user()),
) -> UploadListResponse:
    """
    List uploads belonging to the authenticated company.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_upload_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        status=upload_status,
        document_type=document_type,
        limit=limit,
        offset=offset,
        user=current_user,
    )

    if result is None:
        return UploadListResponse(
            company_id=company_id,
            items=[],
            total=0,
            limit=limit,
            offset=offset,
            has_more=False,
        )

    if isinstance(
        result,
        UploadListResponse,
    ):
        return result

    return UploadListResponse.model_validate(
        result
    )


# ============================================================================
# GET UPLOAD
# ============================================================================


@router.get(
    "/{upload_id}",
    response_model=UploadResponse,
    summary="Get upload metadata",
)
async def get_upload(
    upload_id: str,
    company_id: str = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> UploadResponse:
    """
    Retrieve upload metadata.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_upload_service()

    result = await _call_service(
        service,
        "get",
        upload_id=upload_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload not found.",
        )

    if isinstance(
        result,
        UploadResponse,
    ):
        return result

    return UploadResponse.model_validate(
        result
    )


# ============================================================================
# CANCEL UPLOAD
# ============================================================================


@router.post(
    "/{upload_id}/cancel",
    response_model=UploadCancelResponse,
    summary="Cancel upload processing",
)
async def cancel_upload(
    upload_id: str,
    company_id: str = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> UploadCancelResponse:
    """
    Cancel a queued/processing upload.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_upload_service()

    result = await _call_service(
        service,
        "cancel",
        upload_id=upload_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload not found.",
        )

    if isinstance(
        result,
        UploadCancelResponse,
    ):
        return result

    return UploadCancelResponse.model_validate(
        result
    )


# ============================================================================
# RETRY INGESTION
# ============================================================================


@router.post(
    "/{upload_id}/retry",
    response_model=UploadStatusResponse,
    summary="Retry failed ingestion",
)
async def retry_upload(
    upload_id: str,
    company_id: str = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> UploadStatusResponse:
    """
    Retry ingestion for a failed upload.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_upload_service()

    result = await _call_service(
        service,
        "retry",
        upload_id=upload_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Upload not found.",
        )

    if isinstance(
        result,
        UploadStatusResponse,
    ):
        return result

    return UploadStatusResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=UploadHealthResponse,
    summary="Upload service health",
)
async def upload_health() -> UploadHealthResponse:
    """
    Health check for upload/ingestion infrastructure.
    """

    started = time.perf_counter()

    try:
        service = get_upload_service()

        health_method = getattr(
            service,
            "health",
            None,
        )

        details: dict[str, Any] = {}

        if health_method is not None:
            result = health_method()

            if inspect.isawaitable(result):
                result = await result

            if isinstance(result, dict):
                details = result

        details["latency_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return UploadHealthResponse(
            status="healthy",
            service="uploads",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Upload service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upload service is unavailable.",
        )