"""
backend/app/api/reports.py

Reports API for FinCo AI.

Responsibilities:
    - Request validation
    - Authentication / authorization
    - Tenant isolation
    - Report generation orchestration
    - Report retrieval
    - Report status
    - Report export
    - Safe error handling

Architecture:

    Client
       |
       v
    FastAPI Reports API
       |
       v
    ReportService
       |
       +---- Financial Analysis
       +---- P&L
       +---- Balance Sheet
       +---- Cash Flow
       +---- Forecasting
       +---- Fraud / Risk
       +---- Alerts
       +---- Recommendations
       +---- RAG Evidence
       |
       v
    Report Generator
       |
       +---- JSON
       +---- PDF
       +---- Excel
       +---- CSV
       |
       v
    Storage / Download
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)


# ============================================================================
# ENUMS
# ============================================================================


class ReportType(str, Enum):
    """Supported FinCo AI report types."""

    FINANCIAL_SUMMARY = "financial_summary"
    PROFIT_LOSS = "profit_loss"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"
    FINANCIAL_HEALTH = "financial_health"
    RISK = "risk"
    FRAUD = "fraud"
    FORECAST = "forecast"
    MANAGEMENT = "management"
    EXECUTIVE = "executive"
    ALERTS = "alerts"
    RECOMMENDATIONS = "recommendations"
    COMPREHENSIVE = "comprehensive"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    JSON = "json"
    PDF = "pdf"
    XLSX = "xlsx"
    CSV = "csv"


class ReportStatus(str, Enum):
    """Report lifecycle status."""

    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class ReportPeriod(str, Enum):
    """Common reporting periods."""

    MTD = "mtd"
    QTD = "qtd"
    YTD = "ytd"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


class ReportAccessLevel(str, Enum):
    """Report visibility level."""

    PRIVATE = "private"
    COMPANY = "company"
    ADMIN = "admin"


# ============================================================================
# COMMON SCHEMAS
# ============================================================================


class ReportSection(BaseModel):
    """A section contained inside a generated report."""

    model_config = ConfigDict(extra="forbid")

    section_id: str

    title: str

    description: str | None = None

    enabled: bool = True

    order: int = Field(
        default=1,
        ge=1,
    )


class ReportEvidence(BaseModel):
    """Traceable evidence included in a report."""

    model_config = ConfigDict(extra="forbid")

    source_type: str

    source_id: str | None = None

    title: str | None = None

    description: str

    metric: str | None = None

    value: float | None = None

    unit: str | None = None

    period: str | None = None

    page_number: int | None = Field(
        default=None,
        ge=1,
    )

    citation: str | None = None


class ReportMetadata(BaseModel):
    """Non-sensitive report metadata."""

    model_config = ConfigDict(extra="forbid")

    report_id: str

    company_id: str

    report_type: ReportType

    report_format: ReportFormat

    period_start: date

    period_end: date

    currency: str | None = None

    generated_at: datetime

    generated_by: str | None = None

    status: ReportStatus

    access_level: ReportAccessLevel = ReportAccessLevel.COMPANY


# ============================================================================
# REQUEST MODELS
# ============================================================================


class ReportRequest(BaseModel):
    """
    Generate a report for a company.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    report_type: ReportType = ReportType.COMPREHENSIVE

    report_format: ReportFormat = ReportFormat.JSON

    period: ReportPeriod = ReportPeriod.CUSTOM

    start_date: date

    end_date: date

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=10,
    )

    include_financials: bool = True

    include_pnl: bool = True

    include_balance_sheet: bool = True

    include_cash_flow: bool = True

    include_ratios: bool = True

    include_kpis: bool = True

    include_health_score: bool = True

    include_forecast: bool = True

    include_fraud: bool = True

    include_risk: bool = True

    include_alerts: bool = True

    include_recommendations: bool = True

    include_rag_evidence: bool = True

    include_citations: bool = True

    include_what_if: bool = False

    sections: list[ReportSection] = Field(
        default_factory=list
    )

    title: str | None = Field(
        default=None,
        max_length=200,
    )

    @field_validator("end_date")
    @classmethod
    def validate_end_date(
        cls,
        value: date,
    ) -> date:
        return value


class ReportExportRequest(BaseModel):
    """Request to export an existing report."""

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    format: ReportFormat

    include_citations: bool = True


class ReportRegenerateRequest(BaseModel):
    """Request to regenerate an existing report."""

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    report_format: ReportFormat | None = None

    include_citations: bool = True


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class ReportResponse(BaseModel):
    """Generated report response."""

    model_config = ConfigDict(extra="forbid")

    metadata: ReportMetadata

    title: str

    executive_summary: str | None = None

    content: dict[str, Any] = Field(
        default_factory=dict
    )

    evidence: list[ReportEvidence] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    recommendations: list[str] = Field(
        default_factory=list
    )


class ReportGenerationResponse(BaseModel):
    """Report generation operation response."""

    report_id: str

    company_id: str

    report_type: ReportType

    report_format: ReportFormat

    status: ReportStatus

    generated_at: datetime

    processing_time_ms: float

    report: ReportResponse | None = None

    download_available: bool = False


class ReportStatusResponse(BaseModel):
    """Report status response."""

    report_id: str

    company_id: str

    status: ReportStatus

    progress: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    message: str | None = None

    generated_at: datetime | None = None

    completed_at: datetime | None = None

    error: str | None = None


class ReportListResponse(BaseModel):
    """Paginated report listing."""

    items: list[ReportMetadata]

    total: int

    page: int

    page_size: int

    has_next: bool


class ReportSummaryResponse(BaseModel):
    """Summary of reports generated for a company."""

    company_id: str

    total_reports: int

    completed: int

    generating: int

    failed: int

    queued: int

    latest_report_id: str | None = None

    latest_report_type: ReportType | None = None

    latest_generated_at: datetime | None = None


class ReportHealthResponse(BaseModel):
    """Report subsystem health."""

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    service: str

    timestamp: datetime

    details: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================================
# DEPENDENCIES
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
            "Authentication dependency could not be loaded."
        )

        raise RuntimeError(
            "Authentication dependency is not configured."
        ) from exc


def get_report_service() -> Any:
    """
    Lazily construct ReportService.

    Expected implementation:

        backend.app.reports.report_service.ReportService
    """

    try:
        from backend.app.reports.report_service import (
            ReportService,
        )

        return ReportService()

    except ImportError as exc:
        logger.exception(
            "Report service could not be loaded."
        )

        raise RuntimeError(
            "Report service is not configured."
        ) from exc


# ============================================================================
# SECURITY
# ============================================================================


def _get_user_company_id(
    current_user: Any,
) -> str | None:
    """Extract company ID from authenticated user."""

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
    """Check administrative privileges."""

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

    Reports may contain highly sensitive financial information,
    so company isolation must be enforced at the API boundary
    as well as in the service/repository layer.
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
            "Unauthorized report access attempt: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this company's reports."
            ),
        )


# ============================================================================
# VALIDATION
# ============================================================================


def validate_period(
    start_date: date,
    end_date: date,
    max_days: int = 3660,
) -> None:
    """Validate report date range."""

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be after end_date."
            ),
        )

    if (end_date - start_date).days > max_days:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Report period cannot exceed "
                f"{max_days} days."
            ),
        )


# ============================================================================
# SERVICE HELPER
# ============================================================================


async def _call_service(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:
    """
    Call a report-service method.

    Supports both async and synchronous implementations.
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
                f"Report service does not implement "
                f"'{method_name}'."
            ),
        )

    try:
        result = method(**kwargs)

        if hasattr(result, "__await__"):
            result = await result

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid report-service request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Report service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report operation failed.",
        ) from exc


# ============================================================================
# GENERATE REPORT
# ============================================================================


@router.post(
    "/generate",
    response_model=ReportGenerationResponse,
    summary="Generate financial report",
    description=(
        "Generate a FinCo AI financial intelligence report "
        "using financial analytics, forecasting, risk, fraud, "
        "alerts, recommendations and RAG evidence."
    ),
)
async def generate_report(
    request: ReportRequest,
    current_user: Any = Depends(get_current_user()),
) -> ReportGenerationResponse:
    """
    Generate a report.

    Main FinCo AI reporting pipeline:

        Financial Data
              ↓
        Financial Analysis
              ↓
        P&L / Balance Sheet / Cash Flow
              ↓
        Ratios / KPIs / Health Score
              ↓
        Forecasting
              ↓
        Fraud / Risk
              ↓
        Alerts
              ↓
        Recommendations
              ↓
        RAG Evidence
              ↓
        Report Generator
              ↓
        PDF / XLSX / CSV / JSON
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_period(
        request.start_date,
        request.end_date,
    )

    started = time.perf_counter()

    service = get_report_service()

    result = await _call_service(
        service,
        "generate",
        company_id=request.company_id,
        report_type=request.report_type,
        report_format=request.report_format,
        period=request.period,
        start_date=request.start_date,
        end_date=request.end_date,
        currency=request.currency,
        include_financials=request.include_financials,
        include_pnl=request.include_pnl,
        include_balance_sheet=request.include_balance_sheet,
        include_cash_flow=request.include_cash_flow,
        include_ratios=request.include_ratios,
        include_kpis=request.include_kpis,
        include_health_score=request.include_health_score,
        include_forecast=request.include_forecast,
        include_fraud=request.include_fraud,
        include_risk=request.include_risk,
        include_alerts=request.include_alerts,
        include_recommendations=request.include_recommendations,
        include_rag_evidence=request.include_rag_evidence,
        include_citations=request.include_citations,
        include_what_if=request.include_what_if,
        sections=request.sections,
        title=request.title,
        user=current_user,
    )

    processing_time_ms = round(
        (time.perf_counter() - started) * 1000,
        2,
    )

    if isinstance(
        result,
        ReportGenerationResponse,
    ):
        return result

    if isinstance(result, dict):
        return ReportGenerationResponse(
            report_id=result["report_id"],
            company_id=result.get(
                "company_id",
                request.company_id,
            ),
            report_type=result.get(
                "report_type",
                request.report_type,
            ),
            report_format=result.get(
                "report_format",
                request.report_format,
            ),
            status=result.get(
                "status",
                ReportStatus.COMPLETED,
            ),
            generated_at=result.get(
                "generated_at",
                datetime.now(timezone.utc),
            ),
            processing_time_ms=result.get(
                "processing_time_ms",
                processing_time_ms,
            ),
            report=result.get("report"),
            download_available=result.get(
                "download_available",
                False,
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Invalid report response from service.",
    )


# ============================================================================
# LIST REPORTS
# ============================================================================


@router.get(
    "",
    response_model=ReportListResponse,
    summary="List company reports",
)
async def list_reports(
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    report_type: ReportType | None = None,
    report_status: ReportStatus | None = Query(
        default=None,
        alias="status",
    ),
    page: int = Query(
        default=1,
        ge=1,
        le=10000,
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    current_user: Any = Depends(get_current_user()),
) -> ReportListResponse:
    """
    List reports belonging to a company.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_report_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        report_type=report_type,
        status=report_status,
        page=page,
        page_size=page_size,
        user=current_user,
    )

    if isinstance(result, ReportListResponse):
        return result

    if isinstance(result, dict):
        return ReportListResponse(**result)

    items = result or []

    return ReportListResponse(
        items=items,
        total=len(items),
        page=page,
        page_size=page_size,
        has_next=len(items) == page_size,
    )


# ============================================================================
# GET REPORT
# ============================================================================


@router.get(
    "/{report_id}",
    response_model=ReportResponse,
    summary="Get generated report",
)
async def get_report(
    report_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> ReportResponse:
    """
    Retrieve a generated report.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_report_service()

    result = await _call_service(
        service,
        "get",
        report_id=report_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )

    if isinstance(result, ReportResponse):
        return result

    return ReportResponse.model_validate(result)


# ============================================================================
# REPORT STATUS
# ============================================================================


@router.get(
    "/{report_id}/status",
    response_model=ReportStatusResponse,
    summary="Get report generation status",
)
async def get_report_status(
    report_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> ReportStatusResponse:
    """
    Check report generation status.

    Useful when report generation is performed asynchronously.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_report_service()

    result = await _call_service(
        service,
        "get_status",
        report_id=report_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found.",
        )

    if isinstance(result, ReportStatusResponse):
        return result

    if isinstance(result, dict):
        return ReportStatusResponse(
            report_id=report_id,
            company_id=company_id,
            **result,
        )

    return ReportStatusResponse(
        report_id=report_id,
        company_id=company_id,
        status=ReportStatus.COMPLETED,
        progress=100,
    )


# ============================================================================
# EXPORT
# ============================================================================


@router.post(
    "/export",
    summary="Export report",
)
async def export_report(
    request: ReportExportRequest,
    current_user: Any = Depends(get_current_user()),
) -> Response:
    """
    Export an existing report.

    Supported formats:
        - JSON
        - PDF
        - XLSX
        - CSV

    The actual file generation remains inside ReportService.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_report_service()

    result = await _call_service(
        service,
        "export",
        report_id=request.report_id,
        company_id=request.company_id,
        report_format=request.format,
        include_citations=request.include_citations,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report export could not be generated.",
        )

    # File path returned by the service.
    if isinstance(result, str):
        media_types = {
            ReportFormat.PDF: "application/pdf",
            ReportFormat.XLSX: (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            ReportFormat.CSV: "text/csv",
            ReportFormat.JSON: "application/json",
        }

        return FileResponse(
            path=result,
            media_type=media_types.get(
                request.format,
                "application/octet-stream",
            ),
        )

    # Raw bytes returned by the service.
    if isinstance(result, bytes):
        media_types = {
            ReportFormat.PDF: "application/pdf",
            ReportFormat.XLSX: (
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            ReportFormat.CSV: "text/csv",
            ReportFormat.JSON: "application/json",
        }

        filename = (
            f"{request.report_id}."
            f"{request.format.value}"
        )

        return StreamingResponse(
            iter([result]),
            media_type=media_types.get(
                request.format,
                "application/octet-stream",
            ),
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{filename}"'
                )
            },
        )

    # Response returned directly by service.
    if isinstance(result, Response):
        return result

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Invalid report export returned by service.",
    )


# ============================================================================
# REGENERATE
# ============================================================================


@router.post(
    "/regenerate",
    response_model=ReportGenerationResponse,
    summary="Regenerate report",
)
async def regenerate_report(
    request: ReportRegenerateRequest,
    current_user: Any = Depends(get_current_user()),
) -> ReportGenerationResponse:
    """
    Regenerate an existing report.

    Useful when:
        - source financial data changed
        - forecast models were updated
        - recommendation logic changed
        - citations need refreshing
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_report_service()

    result = await _call_service(
        service,
        "regenerate",
        report_id=request.report_id,
        company_id=request.company_id,
        report_format=request.report_format,
        include_citations=request.include_citations,
        user=current_user,
    )

    if isinstance(
        result,
        ReportGenerationResponse,
    ):
        return result

    if isinstance(result, dict):
        return ReportGenerationResponse(**result)

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Invalid regeneration response.",
    )


# ============================================================================
# EXECUTIVE REPORT
# ============================================================================


@router.post(
    "/executive/{company_id}",
    response_model=ReportGenerationResponse,
    summary="Generate executive report",
)
async def executive_report(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    report_format: ReportFormat = Query(
        default=ReportFormat.JSON,
    ),
    current_user: Any = Depends(get_current_user()),
) -> ReportGenerationResponse:
    """
    Generate an executive-level report.

    Focus:
        - Revenue
        - Profitability
        - Cash flow
        - Financial health
        - Risk
        - Forecast
        - Key alerts
        - Top recommendations
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    request = ReportRequest(
        company_id=company_id,
        report_type=ReportType.EXECUTIVE,
        report_format=report_format,
        start_date=start_date,
        end_date=end_date,
        include_financials=True,
        include_pnl=True,
        include_balance_sheet=True,
        include_cash_flow=True,
        include_ratios=True,
        include_kpis=True,
        include_health_score=True,
        include_forecast=True,
        include_fraud=False,
        include_risk=True,
        include_alerts=True,
        include_recommendations=True,
        include_rag_evidence=True,
        include_citations=True,
    )

    return await generate_report(
        request=request,
        current_user=current_user,
    )


# ============================================================================
# COMPREHENSIVE REPORT
# ============================================================================


@router.post(
    "/comprehensive/{company_id}",
    response_model=ReportGenerationResponse,
    summary="Generate comprehensive financial report",
)
async def comprehensive_report(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    report_format: ReportFormat = Query(
        default=ReportFormat.PDF,
    ),
    include_what_if: bool = False,
    current_user: Any = Depends(get_current_user()),
) -> ReportGenerationResponse:
    """
    Generate the complete FinCo AI report.

    This is the broadest report and combines financial,
    predictive, risk and AI-generated intelligence.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    request = ReportRequest(
        company_id=company_id,
        report_type=ReportType.COMPREHENSIVE,
        report_format=report_format,
        start_date=start_date,
        end_date=end_date,
        include_financials=True,
        include_pnl=True,
        include_balance_sheet=True,
        include_cash_flow=True,
        include_ratios=True,
        include_kpis=True,
        include_health_score=True,
        include_forecast=True,
        include_fraud=True,
        include_risk=True,
        include_alerts=True,
        include_recommendations=True,
        include_rag_evidence=True,
        include_citations=True,
        include_what_if=include_what_if,
    )

    return await generate_report(
        request=request,
        current_user=current_user,
    )


# ============================================================================
# SUMMARY
# ============================================================================


@router.get(
    "/summary/{company_id}",
    response_model=ReportSummaryResponse,
    summary="Get report summary",
)
async def report_summary(
    company_id: str,
    current_user: Any = Depends(get_current_user()),
) -> ReportSummaryResponse:
    """
    Return report-generation statistics.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_report_service()

    result = await _call_service(
        service,
        "summary",
        company_id=company_id,
        user=current_user,
    )

    if isinstance(
        result,
        ReportSummaryResponse,
    ):
        return result

    if isinstance(result, dict):
        return ReportSummaryResponse(
            company_id=company_id,
            **{
                key: value
                for key, value in result.items()
                if key != "company_id"
            },
        )

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Invalid report summary.",
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=ReportHealthResponse,
    summary="Report service health",
)
async def report_health() -> ReportHealthResponse:
    """
    Health check for the report subsystem.

    Does not expose:
        - database credentials
        - API keys
        - storage credentials
        - connection strings
        - internal secrets
    """

    started = time.perf_counter()

    try:
        service = get_report_service()

        health_method = getattr(
            service,
            "health",
            None,
        )

        details: dict[str, Any] = {}

        if health_method is not None:
            result = health_method()

            if hasattr(result, "__await__"):
                result = await result

            if isinstance(result, dict):
                details = result

        details["latency_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return ReportHealthResponse(
            status="healthy",
            service="reports",
            timestamp=datetime.now(timezone.utc),
            details=details,
        )

    except Exception:
        logger.exception(
            "Report service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Report service is unavailable.",
        )