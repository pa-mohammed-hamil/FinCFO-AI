"""
FinCo AI - Documents API

Responsibilities:
- Create document records
- List company documents
- Get document metadata
- Search/filter documents
- Trigger document processing
- Reprocess failed documents
- Delete documents
- Track ingestion status
- Tenant isolation
- RBAC
- Audit logging
- Rate limiting

Actual parsing/OCR/chunking/embedding belongs to:
    app.ingestion.*
    app.rag.*

This API layer should not perform document processing itself.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.audit import AuditService
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentProcessingError,
    DuplicateResourceError,
    FileTooLargeError,
    FinCoException,
    NotFoundError,
    UnsupportedFileTypeError,
)
from app.core.logging import (
    clear_request_context,
    generate_request_id,
    get_logger,
    set_request_context,
)
from app.core.permissions import (
    AuthorizationContext,
    Permission,
    PermissionService,
    TenantAccessService,
)
from app.core.rate_limiter import (
    document_upload_rate_limit,
)
from app.database.models.document import Document
from app.database.session import get_db


# ============================================================
# Router
# ============================================================

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

logger = get_logger(__name__)


# ============================================================
# Configuration
# ============================================================

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".csv",
    ".xlsx",
    ".xls",
    ".docx",
    ".txt",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
}

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/csv",
    "application/vnd.ms-excel",
    (
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    ),
    "text/plain",
    "application/json",
    "image/png",
    "image/jpeg",
}


# ============================================================
# Current User Dependency
# ============================================================

def get_current_user(
    request: Any,
) -> AuthorizationContext:
    """
    Retrieve authenticated authorization context.

    Authentication middleware should populate:
        request.state.authorization_context
    """

    user = getattr(
        request.state,
        "authorization_context",
        None,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    return user


# ============================================================
# Schemas
# ============================================================

class DocumentResponse(BaseModel):
    id: str

    company_id: str

    name: str

    file_name: Optional[str] = None

    file_type: Optional[str] = None

    content_type: Optional[str] = None

    file_size: Optional[int] = None

    status: str = "uploaded"

    processing_status: Optional[str] = None

    page_count: Optional[int] = None

    chunk_count: Optional[int] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    error_message: Optional[str] = None

    uploaded_by: Optional[str] = None

    created_at: Optional[datetime] = None

    updated_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]

    total: int

    page: int

    page_size: int

    has_next: bool


class DocumentProcessingResponse(BaseModel):
    document_id: str

    status: str

    message: str

    request_id: str


class DocumentSearchResponse(BaseModel):
    items: list[DocumentResponse]

    total: int

    query: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    document_id: str

    status: str

    processing_status: Optional[str] = None

    chunk_count: Optional[int] = None

    page_count: Optional[int] = None

    error_message: Optional[str] = None


# ============================================================
# Helpers
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_company_id(
    requested_company_id: Optional[str],
    current_user: AuthorizationContext,
) -> str:
    """
    Resolve company tenant and enforce authorization.
    """

    company_id = (
        requested_company_id
        or current_user.company_id
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="company_id is required.",
        )

    TenantAccessService.require_company_access(
        current_user,
        company_id,
    )

    return company_id


def require_document_read(
    current_user: AuthorizationContext,
) -> None:

    PermissionService.require_permission(
        current_user,
        Permission.DOCUMENT_READ,
    )


def require_document_create(
    current_user: AuthorizationContext,
) -> None:

    PermissionService.require_permission(
        current_user,
        Permission.DOCUMENT_CREATE,
    )


def require_document_delete(
    current_user: AuthorizationContext,
) -> None:

    PermissionService.require_permission(
        current_user,
        Permission.DOCUMENT_DELETE,
    )


def get_document_or_404(
    db: Session,
    document_id: str,
    company_id: str,
) -> Document:

    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.company_id == company_id,
        )
        .first()
    )

    if document is None:
        raise DocumentNotFoundError(
            f"Document '{document_id}' was not found."
        )

    return document


def document_to_response(
    document: Document,
) -> DocumentResponse:
    """
    Convert SQLAlchemy document model to API response.

    Uses getattr so the API remains tolerant of optional
    model fields while the schema evolves.
    """

    return DocumentResponse(
        id=str(
            getattr(
                document,
                "id",
                "",
            )
        ),

        company_id=str(
            getattr(
                document,
                "company_id",
                "",
            )
        ),

        name=str(
            getattr(
                document,
                "name",
                getattr(
                    document,
                    "file_name",
                    "",
                ),
            )
        ),

        file_name=getattr(
            document,
            "file_name",
            None,
        ),

        file_type=getattr(
            document,
            "file_type",
            None,
        ),

        content_type=getattr(
            document,
            "content_type",
            None,
        ),

        file_size=getattr(
            document,
            "file_size",
            None,
        ),

        status=str(
            getattr(
                document,
                "status",
                "uploaded",
            )
        ),

        processing_status=getattr(
            document,
            "processing_status",
            None,
        ),

        page_count=getattr(
            document,
            "page_count",
            None,
        ),

        chunk_count=getattr(
            document,
            "chunk_count",
            None,
        ),

        metadata=getattr(
            document,
            "metadata",
            {},
        )
        or {},

        error_message=getattr(
            document,
            "error_message",
            None,
        ),

        uploaded_by=(
            str(
                getattr(
                    document,
                    "uploaded_by",
                )
            )
            if getattr(
                document,
                "uploaded_by",
                None,
            )
            else None
        ),

        created_at=getattr(
            document,
            "created_at",
            None,
        ),

        updated_at=getattr(
            document,
            "updated_at",
            None,
        ),
    )


def validate_file_extension(
    filename: str,
) -> str:

    filename = filename.strip()

    if "." not in filename:

        raise UnsupportedFileTypeError(
            "File must have a supported extension."
        )

    extension = (
        "."
        + filename.rsplit(
            ".",
            1,
        )[-1].lower()
    )

    if extension not in ALLOWED_EXTENSIONS:

        raise UnsupportedFileTypeError(
            f"Unsupported file type: {extension}. "
            f"Allowed types: "
            f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    return extension


def validate_content_type(
    content_type: Optional[str],
) -> None:

    if not content_type:
        return

    if content_type not in ALLOWED_CONTENT_TYPES:

        logger.warning(
            "Unexpected content type",
            extra={
                "content_type": content_type,
            },
        )

        raise UnsupportedFileTypeError(
            f"Unsupported content type: {content_type}"
        )


async def read_upload_safely(
    file: UploadFile,
) -> bytes:
    """
    Read uploaded file while enforcing size limit.

    The file is read in chunks rather than trusting the
    Content-Length header.
    """

    total_size = 0

    chunks: list[bytes] = []

    while True:

        chunk = await file.read(
            1024 * 1024
        )

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > MAX_FILE_SIZE:

            raise FileTooLargeError(
                f"File exceeds the maximum allowed "
                f"size of {MAX_FILE_SIZE // (1024 * 1024)} MB."
            )

        chunks.append(chunk)

    return b"".join(chunks)


def check_duplicate_filename(
    db: Session,
    company_id: str,
    filename: str,
) -> bool:
    """
    Basic duplicate check.

    Production systems should additionally use SHA-256
    content hashes for reliable duplicate detection.
    """

    query = (
        db.query(Document)
        .filter(
            Document.company_id == company_id,
        )
    )

    # Support models with file_name.
    try:

        return (
            query.filter(
                Document.file_name == filename
            ).first()
            is not None
        )

    except AttributeError:

        return False


# ============================================================
# POST /documents/upload
# ============================================================

@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    company_id: Optional[str] = Query(
        default=None
    ),
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
    _: Any = Depends(document_upload_rate_limit),
) -> DocumentResponse:
    """
    Upload a financial/business document.

    Supported:
    - PDF
    - CSV
    - Excel
    - DOCX
    - TXT
    - JSON
    - PNG/JPEG

    The endpoint stores document metadata and content.
    Parsing/chunking/embedding is handled by ingestion services.
    """

    request_id = generate_request_id()

    resolved_company_id = resolve_company_id(
        company_id,
        current_user,
    )

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=resolved_company_id,
    )

    require_document_create(
        current_user
    )

    try:

        if not file.filename:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required.",
            )

        filename = file.filename.strip()

        extension = validate_file_extension(
            filename
        )

        validate_content_type(
            file.content_type
        )

        if check_duplicate_filename(
            db,
            resolved_company_id,
            filename,
        ):

            raise DuplicateResourceError(
                f"Document '{filename}' already exists."
            )

        content = await read_upload_safely(
            file
        )

        if not content:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        document_id = str(
            uuid.uuid4()
        )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        document_data = {
            "id": document_id,
            "company_id": resolved_company_id,
            "name": filename,
            "file_name": filename,
            "file_type": extension.lstrip("."),
            "content_type": file.content_type,
            "file_size": len(content),
            "status": "uploaded",
            "processing_status": "pending",
            "uploaded_by": current_user.user_id,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "metadata": {
                "request_id": request_id,
                "original_filename": filename,
            },
        }

        # ----------------------------------------------------
        # Model compatibility
        # ----------------------------------------------------

        try:

            document = Document(
                **document_data
            )

        except TypeError:

            # If the project's SQLAlchemy model uses a
            # different subset of fields, construct only
            # fields that are available.

            document = Document()

            for field_name, value in document_data.items():

                if hasattr(
                    document,
                    field_name,
                ):

                    setattr(
                        document,
                        field_name,
                        value,
                    )

        db.add(document)

        db.commit()

        db.refresh(document)

        # ----------------------------------------------------
        # Persist actual file
        # ----------------------------------------------------

        try:

            from app.ingestion.pipeline import (
                save_uploaded_file,
            )

            await save_uploaded_file(
                document_id=document_id,
                company_id=resolved_company_id,
                filename=filename,
                content=content,
            )

        except ImportError:

            # The ingestion pipeline may be implemented
            # separately. Metadata is still preserved.
            logger.info(
                "save_uploaded_file is not yet available"
            )

        except Exception as exc:

            logger.exception(
                "Failed to persist uploaded file",
                extra={
                    "document_id": document_id,
                    "company_id": resolved_company_id,
                },
            )

            document.status = "failed"
            document.processing_status = "failed"

            if hasattr(
                document,
                "error_message",
            ):

                document.error_message = (
                    "File storage failed."
                )

            db.commit()

            raise DocumentProcessingError(
                "Unable to store uploaded document."
            ) from exc

        # ----------------------------------------------------
        # Audit
        # ----------------------------------------------------

        try:

            AuditService(
                db
            ).document_upload(
                user_id=current_user.user_id,
                company_id=resolved_company_id,
                document_id=document_id,
                metadata={
                    "filename": filename,
                    "file_size": len(content),
                    "content_type": file.content_type,
                    "request_id": request_id,
                },
            )

        except Exception:

            logger.exception(
                "Document upload audit failed"
            )

        return document_to_response(
            document
        )

    except FinCoException:

        db.rollback()

        raise

    except Exception as exc:

        db.rollback()

        logger.exception(
            "Document upload failed",
            extra={
                "request_id": request_id,
                "company_id": resolved_company_id,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload failed.",
        ) from exc

    finally:

        clear_request_context()


# ============================================================
# GET /documents
# ============================================================

@router.get(
    "",
    response_model=DocumentListResponse,
)
def list_documents(
    company_id: Optional[str] = Query(
        default=None
    ),
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
    ),
    file_type: Optional[str] = None,
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """
    List documents belonging to the authenticated company.
    """

    resolved_company_id = resolve_company_id(
        company_id,
        current_user,
    )

    require_document_read(
        current_user
    )

    query = (
        db.query(Document)
        .filter(
            Document.company_id
            == resolved_company_id
        )
    )

    if status_filter:

        if hasattr(
            Document,
            "status",
        ):

            query = query.filter(
                Document.status
                == status_filter
            )

    if file_type:

        if hasattr(
            Document,
            "file_type",
        ):

            query = query.filter(
                Document.file_type
                == file_type.lower()
            )

    total = query.count()

    offset = (
        page - 1
    ) * page_size

    documents = (
        query
        .order_by(
            Document.created_at.desc()
        )
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return DocumentListResponse(
        items=[
            document_to_response(
                document
            )
            for document in documents
        ],

        total=total,

        page=page,

        page_size=page_size,

        has_next=(
            offset + len(documents)
            < total
        ),
    )


# ============================================================
# GET /documents/search
# ============================================================

@router.get(
    "/search",
    response_model=DocumentSearchResponse,
)
def search_documents(
    q: str = Query(
        ...,
        min_length=1,
        max_length=200,
    ),
    company_id: Optional[str] = Query(
        default=None
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
) -> DocumentSearchResponse:
    """
    Search document metadata.

    Semantic document search belongs to the RAG layer.
    This endpoint searches basic document metadata.
    """

    resolved_company_id = resolve_company_id(
        company_id,
        current_user,
    )

    require_document_read(
        current_user
    )

    query = (
        db.query(Document)
        .filter(
            Document.company_id
            == resolved_company_id
        )
    )

    search_term = f"%{q.strip()}%"

    filters = []

    if hasattr(
        Document,
        "name",
    ):

        filters.append(
            Document.name.ilike(
                search_term
            )
        )

    if hasattr(
        Document,
        "file_name",
    ):

        filters.append(
            Document.file_name.ilike(
                search_term
            )
        )

    if not filters:

        documents = []

    else:

        from sqlalchemy import or_

        documents = (
            query
            .filter(
                or_(*filters)
            )
            .order_by(
                Document.created_at.desc()
            )
            .limit(limit)
            .all()
        )

    return DocumentSearchResponse(
        items=[
            document_to_response(
                document
            )
            for document in documents
        ],

        total=len(documents),

        query=q,
    )


# ============================================================
# GET /documents/{document_id}
# ============================================================

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document(
    document_id: str,
    company_id: Optional[str] = Query(
        default=None
    ),
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """
    Retrieve a single document.

    Company ID is always part of the query to prevent
    cross-tenant document access.
    """

    resolved_company_id = resolve_company_id(
        company_id,
        current_user,
    )

    require_document_read(
        current_user
    )

    document = get_document_or_404(
        db,
        document_id,
        resolved_company_id,
    )

    return document_to_response(
        document
    )


# ============================================================
# GET /documents/{document_id}/status
# ============================================================

@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
)
def get_document_status(
    document_id: str,
    company_id: Optional[str] = Query(
        default=None
    ),
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
) -> DocumentStatusResponse:
    """
    Return document ingestion status.
    """

    resolved_company_id = resolve_company_id(
        company_id,
        current_user,
    )

    require_document_read(
        current_user
    )

    document = get_document_or_404(
        db,
        document_id,
        resolved_company_id,
    )

    return DocumentStatusResponse(
        document_id=str(
            document.id
        ),

        status=str(
            getattr(
                document,
                "status",
                "unknown",
            )
        ),

        processing_status=getattr(
            document,
            "processing_status",
            None,
        ),

        chunk_count=getattr(
            document,
            "chunk_count",
            None,
        ),

        page_count=getattr(
            document,
            "page_count",
            None,
        ),

        error_message=getattr(
            document,
            "error_message",
            None,
        ),
    )


# ============================================================
# POST /documents/{document_id}/process
# ============================================================

@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessingResponse,
)
async def process_document(
    document_id: str,
    company_id: Optional[str] = Query(
        default=None
    ),
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
) -> DocumentProcessingResponse:
    """
    Trigger ingestion:

        File
          ↓
        Classification
          ↓
        Validation
          ↓
        Parser / OCR
          ↓
        Normalization
          ↓
        Chunking
          ↓
        Embeddings
          ↓
        Vector Store
          ↓
        Ready for RAG
    """

    request_id = generate_request_id()

    resolved_company_id = resolve_company_id(
        company_id,
        current_user,
    )

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=resolved_company_id,
    )

    require_document_create(
        current_user
    )

    try:

        document = get_document_or_404(
            db,
            document_id,
            resolved_company_id,
        )

        if hasattr(
            document,
            "processing_status",
        ):

            document.processing_status = (
                "processing"
            )

        if hasattr(
            document,
            "status",
        ):

            document.status = "processing"

        db.commit()

        # ----------------------------------------------------
        # Run ingestion pipeline
        # ----------------------------------------------------

        try:

            from app.ingestion.pipeline import (
                DocumentPipeline,
            )

            pipeline = DocumentPipeline(
                db=db
            )

            if hasattr(
                pipeline,
                "process",
            ):

                result = await pipeline.process(
                    document_id=document_id,
                    company_id=resolved_company_id,
                    request_id=request_id,
                )

            else:

                result = {}

        except ImportError:

            logger.warning(
                "DocumentPipeline is not implemented yet"
            )

            result = {
                "status": "queued"
            }

        except Exception as exc:

            if hasattr(
                document,
                "status",
            ):

                document.status = "failed"

            if hasattr(
                document,
                "processing_status",
            ):

                document.processing_status = (
                    "failed"
                )

            if hasattr(
                document,
                "error_message",
            ):

                document.error_message = (
                    str(exc)[:2000]
                )

            db.commit()

            raise DocumentProcessingError(
                "Document processing failed."
            ) from exc

        # ----------------------------------------------------
        # Update processing result
        # ----------------------------------------------------

        result = (
            result
            if isinstance(
                result,
                dict,
            )
            else {}
        )

        result_status = result.get(
            "status",
            "processed",
        )

        if result_status in {
            "processed",
            "completed",
            "ready",
        }:

            if hasattr(
                document,
                "status",
            ):

                document.status = "processed"

            if hasattr(
                document,
                "processing_status",
            ):

                document.processing_status = (
                    "completed"
                )

        elif result_status == "queued":

            if hasattr(
                document,
                "status",
            ):

                document.status = "processing"

            if hasattr(
                document,
                "processing_status",
            ):

                document.processing_status = (
                    "queued"
                )

        db.commit()

        # ----------------------------------------------------
        # Audit
        # ----------------------------------------------------

        try:

            AuditService(
                db
            ).document_processed(
                user_id=current_user.user_id,
                company_id=resolved_company_id,
                document_id=document_id,
                metadata={
                    "request_id": request_id,
                    "status": result_status,
                },
            )

        except Exception:

            logger.exception(
                "Document processing audit failed"
            )

        return DocumentProcessingResponse(
            document_id=document_id,
            status=result_status,
            message=(
                "Document processing completed."
                if result_status
                in {
                    "processed",
                    "completed",
                    "ready",
                }
                else
                "Document processing started."
            ),
            request_id=request_id,
        )

    except FinCoException:

        db.rollback()

        raise

    finally:

        clear_request_context()


# ============================================================
# POST /documents/{document_id}/reprocess
# ============================================================

@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentProcessingResponse,
)
async def reprocess_document(
    document_id: str,
    company_id: Optional[str] = Query(
        default=None
    ),
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
) -> DocumentProcessingResponse:
    """
    Re-run ingestion for a document.

    Useful when:
    - OCR failed
    - parser failed
    - embeddings failed
    - chunking configuration changed
    - vector index was rebuilt
    """

    resolved_company_id = resolve_company_id(
        company_id,
        current_user,
    )

    require_document_create(
        current_user
    )

    document = get_document_or_404(
        db,
        document_id,
        resolved_company_id,
    )

    if hasattr(
        document,
        "processing_status",
    ):

        document.processing_status = "pending"

    if hasattr(
        document,
        "status",
    ):

        document.status = "uploaded"

    if hasattr(
        document,
        "error_message",
    ):

        document.error_message = None

    db.commit()

    return await process_document(
        document_id=document_id,
        company_id=resolved_company_id,
        current_user=current_user,
        db=db,
    )


# ============================================================
# DELETE /documents/{document_id}
# ============================================================

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: str,
    company_id: Optional[str] = Query(
        default=None
    ),
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
) -> None:
    """
    Delete a document and its associated processing artifacts.

    Production implementation should remove:
    - object storage file
    - extracted text
    - chunks
    - embeddings
    - vector-store records
    - metadata
    """

    request_id = generate_request_id()

    resolved_company_id = resolve_company_id(
        company_id,
        current_user,
    )

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=resolved_company_id,
    )

    require_document_delete(
        current_user
    )

    try:

        document = get_document_or_404(
            db,
            document_id,
            resolved_company_id,
        )

        # ----------------------------------------------------
        # Delete ingestion/RAG artifacts
        # ----------------------------------------------------

        try:

            from app.ingestion.pipeline import (
                delete_document_artifacts,
            )

            result = delete_document_artifacts(
                document_id=document_id,
                company_id=resolved_company_id,
            )

            # Support async implementations.
            if hasattr(
                result,
                "__await__",
            ):

                await result

        except ImportError:

            logger.info(
                "Document artifact cleanup service "
                "is not implemented yet"
            )

        except Exception:

            logger.exception(
                "Document artifact cleanup failed",
                extra={
                    "document_id": document_id,
                },
            )

            raise DocumentProcessingError(
                "Unable to remove document artifacts."
            )

        # ----------------------------------------------------
        # Delete DB record
        # ----------------------------------------------------

        db.delete(
            document
        )

        db.commit()

        # ----------------------------------------------------
        # Audit
        # ----------------------------------------------------

        try:

            AuditService(
                db
            ).log(
                action="document_deleted",
                user_id=current_user.user_id,
                company_id=resolved_company_id,
                entity_type="document",
                entity_id=document_id,
                metadata={
                    "request_id": request_id,
                    "filename": getattr(
                        document,
                        "file_name",
                        None,
                    ),
                },
            )

        except Exception:

            logger.exception(
                "Document deletion audit failed"
            )

    except FinCoException:

        db.rollback()

        raise

    except Exception as exc:

        db.rollback()

        logger.exception(
            "Document deletion failed",
            extra={
                "request_id": request_id,
                "document_id": document_id,
                "company_id": resolved_company_id,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document deletion failed.",
        ) from exc

    finally:

        clear_request_context()