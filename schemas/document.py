"""
FinCo AI - Document Schemas

Pydantic schemas used for:

- Document upload
- Document creation
- Document updates
- Document classification
- Document processing
- RAG indexing
- Document search
- Document metadata
- Document responses
- Pagination
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


# ============================================================
# ENUMS
# ============================================================

class DocumentStatus(str, Enum):
    """Document processing status."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentType(str, Enum):
    """Supported financial/business document types."""

    ANNUAL_REPORT = "annual_report"
    INCOME_STATEMENT = "income_statement"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    INVOICE = "invoice"
    BANK_STATEMENT = "bank_statement"
    BUDGET = "budget"
    TRANSACTION_REPORT = "transaction_report"
    CONTRACT = "contract"
    OTHER = "other"


class FileType(str, Enum):
    """Supported uploaded file formats."""

    PDF = "pdf"
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"
    DOCX = "docx"
    DOC = "doc"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    JSON = "json"


# ============================================================
# DOCUMENT BASE
# ============================================================

class DocumentBase(BaseModel):
    """Common document fields."""

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )

    title: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Human-readable document title.",
    )

    description: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional document description.",
    )

    document_type: DocumentType = Field(
        default=DocumentType.OTHER,
        description="Financial/business document classification.",
    )

    fiscal_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2200,
        description="Fiscal year associated with the document.",
    )

    reporting_period: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Reporting period such as Q1, Q2, FY2025.",
    )

    currency: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="ISO-style three-letter currency code.",
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        """Normalize currency code."""

        if value is None:
            return None

        return value.upper()


# ============================================================
# CREATE DOCUMENT
# ============================================================

class DocumentCreate(DocumentBase):
    """Schema for creating a document record."""

    company_id: int = Field(
        ...,
        gt=0,
        description="Company that owns the document.",
    )


# ============================================================
# UPDATE DOCUMENT
# ============================================================

class DocumentUpdate(BaseModel):
    """Schema for updating document metadata."""

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )

    title: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=2000,
    )

    document_type: Optional[DocumentType] = None

    fiscal_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    reporting_period: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    currency: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        """Normalize currency code."""

        if value is None:
            return None

        return value.upper()


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

class DocumentUploadResponse(BaseModel):
    """Response returned after a document upload."""

    model_config = ConfigDict(
        use_enum_values=True,
    )

    id: int

    filename: str

    original_filename: Optional[str] = None

    file_type: str

    mime_type: Optional[str] = None

    file_size: int

    status: DocumentStatus

    message: str = "Document uploaded successfully."

    uploaded_at: datetime


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

class DocumentProcessingRequest(BaseModel):
    """Request to process a document."""

    force_reprocess: bool = Field(
        default=False,
        description="Whether to process an already processed document again.",
    )

    enable_ocr: bool = Field(
        default=True,
        description="Enable OCR for scanned/image documents.",
    )

    extract_tables: bool = Field(
        default=True,
        description="Extract tables from financial documents.",
    )

    create_embeddings: bool = Field(
        default=True,
        description="Create embeddings for RAG indexing.",
    )


class DocumentProcessingResponse(BaseModel):
    """Document processing result."""

    model_config = ConfigDict(
        use_enum_values=True,
    )

    document_id: int

    status: DocumentStatus

    page_count: Optional[int] = None

    chunk_count: int = 0

    tables_extracted: int = 0

    embeddings_created: int = 0

    ocr_used: bool = False

    processing_time_seconds: Optional[float] = None

    message: str


# ============================================================
# DOCUMENT CLASSIFICATION
# ============================================================

class DocumentClassification(BaseModel):
    """AI classification result for a document."""

    document_type: DocumentType

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    detected_title: Optional[str] = None

    detected_fiscal_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    detected_reporting_period: Optional[str] = None

    detected_currency: Optional[str] = None

    reasoning: Optional[str] = None


# ============================================================
# DOCUMENT METADATA
# ============================================================

class DocumentMetadata(BaseModel):
    """Extracted document metadata."""

    author: Optional[str] = None

    organization: Optional[str] = None

    language: Optional[str] = None

    page_count: Optional[int] = None

    created_date: Optional[datetime] = None

    modified_date: Optional[datetime] = None

    fiscal_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    reporting_period: Optional[str] = None

    currency: Optional[str] = None

    document_type: Optional[DocumentType] = None

    keywords: List[str] = Field(
        default_factory=list,
    )


# ============================================================
# RAG INDEXING
# ============================================================

class DocumentIndexRequest(BaseModel):
    """Request to index a document into the RAG system."""

    embedding_model: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    collection_name: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    force_reindex: bool = False


class DocumentIndexResponse(BaseModel):
    """RAG indexing result."""

    document_id: int

    indexed: bool

    embedding_model: Optional[str] = None

    vector_collection: Optional[str] = None

    chunk_count: int = 0

    embeddings_created: int = 0

    message: str


# ============================================================
# DOCUMENT RESPONSE
# ============================================================

class DocumentResponse(DocumentBase):
    """Complete document API response."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    company_id: int

    uploaded_by: Optional[int] = None

    filename: str

    original_filename: Optional[str] = None

    file_path: str

    file_type: str

    mime_type: Optional[str] = None

    file_size: int

    checksum: Optional[str] = None

    status: DocumentStatus

    processing_error: Optional[str] = None

    page_count: Optional[int] = None

    chunk_count: int = 0

    embedding_model: Optional[str] = None

    vector_collection: Optional[str] = None

    is_indexed: bool = False

    uploaded_at: datetime

    processed_at: Optional[datetime] = None

    updated_at: datetime


# ============================================================
# DOCUMENT SUMMARY
# ============================================================

class DocumentSummary(BaseModel):
    """Compact document representation for dashboards."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    filename: str

    title: Optional[str] = None

    document_type: DocumentType

    status: DocumentStatus

    fiscal_year: Optional[int] = None

    page_count: Optional[int] = None

    chunk_count: int = 0

    is_indexed: bool = False

    uploaded_at: datetime


# ============================================================
# DOCUMENT STATUS RESPONSE
# ============================================================

class DocumentStatusResponse(BaseModel):
    """Current processing/indexing status."""

    document_id: int

    status: DocumentStatus

    processing_error: Optional[str] = None

    page_count: Optional[int] = None

    chunk_count: int = 0

    is_indexed: bool = False

    processed_at: Optional[datetime] = None

    message: Optional[str] = None


# ============================================================
# DOCUMENT SEARCH
# ============================================================

class DocumentSearchRequest(BaseModel):
    """Search/filter documents."""

    query: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    company_id: Optional[int] = Field(
        default=None,
        gt=0,
    )

    document_type: Optional[DocumentType] = None

    status: Optional[DocumentStatus] = None

    fiscal_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    is_indexed: Optional[bool] = None

    page: int = Field(
        default=1,
        ge=1,
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
    )


# ============================================================
# DOCUMENT SEARCH RESULT
# ============================================================

class DocumentSearchResult(BaseModel):
    """Single document search result."""

    document: DocumentSummary

    score: Optional[float] = None

    matched_text: Optional[str] = None

    relevance_reason: Optional[str] = None


class DocumentSearchResponse(BaseModel):
    """Document search response."""

    results: List[DocumentSearchResult]

    total: int

    page: int

    page_size: int


# ============================================================
# DOCUMENT LIST RESPONSE
# ============================================================

class DocumentListResponse(BaseModel):
    """Paginated document listing."""

    items: List[DocumentResponse]

    total: int

    page: int = 1

    page_size: int = 20

    pages: int = 1


# ============================================================
# DOCUMENT DELETE
# ============================================================

class DocumentDeleteResponse(BaseModel):
    """Document deletion response."""

    document_id: int

    deleted: bool

    message: str


# ============================================================
# DOCUMENT ARCHIVE
# ============================================================

class DocumentArchiveResponse(BaseModel):
    """Document archive response."""

    document_id: int

    archived: bool

    message: str


# ============================================================
# DOCUMENT REPROCESS
# ============================================================

class DocumentReprocessRequest(BaseModel):
    """Request to reprocess a document."""

    enable_ocr: bool = True

    extract_tables: bool = True

    recreate_embeddings: bool = True

    force: bool = False


class DocumentReprocessResponse(BaseModel):
    """Reprocessing response."""

    document_id: int

    status: DocumentStatus

    message: str


# ============================================================
# RAG CHUNK
# ============================================================

class DocumentChunk(BaseModel):
    """A chunk generated from a document for RAG."""

    chunk_id: str

    document_id: int

    chunk_index: int

    text: str

    page_number: Optional[int] = None

    section: Optional[str] = None

    token_count: Optional[int] = None

    metadata: dict = Field(
        default_factory=dict,
    )


# ============================================================
# DOCUMENT CITATION
# ============================================================

class DocumentCitation(BaseModel):
    """Citation pointing back to source document evidence."""

    document_id: int

    filename: str

    chunk_id: Optional[str] = None

    page_number: Optional[int] = None

    section: Optional[str] = None

    text: Optional[str] = None

    relevance_score: Optional[float] = None


# ============================================================
# DOCUMENT HEALTH
# ============================================================

class DocumentHealthResponse(BaseModel):
    """Document ingestion/RAG health information."""

    document_id: int

    ingestion_complete: bool

    parsing_complete: bool

    chunking_complete: bool

    embedding_complete: bool

    indexed: bool

    chunk_count: int = 0

    errors: List[str] = Field(
        default_factory=list,
    )

### Fits directly into your architecture

schemas/document.py
        │
        ├── DocumentCreate
        ├── DocumentUpdate
        ├── DocumentResponse
        ├── DocumentUploadResponse
        │
        ├── DocumentProcessingRequest
        ├── DocumentProcessingResponse
        │
        ├── DocumentClassification
        ├── DocumentMetadata
        │
        ├── DocumentIndexRequest
        ├── DocumentIndexResponse
        │
        ├── DocumentSearchRequest
        ├── DocumentSearchResponse
        │
        ├── DocumentChunk
        └── DocumentCitation
                │
                ▼
        documents.py API
                │
                ▼
        ingestion/pipeline.py
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
      Parser   OCR    Tables
                │
                ▼
          Chunking
                │
                ▼
          Embeddings
                │
                ▼
        Hybrid Retrieval
                │
                ▼
        Reranking
                │
                ▼
        RAG / Financial Agent
                │
                ▼
            Citation

**One security recommendation:** keep `file_path` out of externally exposed responses if it contains internal filesystem/S3 locations. For a production API, you can replace it with a safe `download_url` or document identifier.