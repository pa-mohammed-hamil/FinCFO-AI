"""
backend/app/api/fraud_alert.py

Fraud Alert API for FinCo AI.

Architecture
------------

Transaction
    |
    v
Fraud Detection Engine
    |
    v
Fraud Risk Score
    |
    v
Fraud Alert Service
    |
    +---- Alert Creation
    +---- Alert Assignment
    +---- Investigation
    +---- Status Management
    +---- Evidence
    |
    v
Human Review
    |
    v
Audit Log

Important:
- API does not calculate fraud probability.
- API does not decide whether a transaction is fraudulent.
- API does not directly invoke an LLM.
- All operations are tenant isolated by company_id.
- Financial/fraud alerts are never hard-deleted.
"""

from __future__ import annotations

import inspect
import logging
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
    prefix="/fraud-alerts",
    tags=["Fraud Alerts"],
)


# ============================================================================
# TYPES
# ============================================================================

FraudAlertStatus = Literal[
    "open",
    "investigating",
    "confirmed_fraud",
    "false_positive",
    "resolved",
    "dismissed",
]

FraudAlertSeverity = Literal[
    "low",
    "medium",
    "high",
    "critical",
]

FraudRiskLevel = Literal[
    "very_low",
    "low",
    "medium",
    "high",
    "critical",
]

FraudAlertSource = Literal[
    "rule",
    "ml_model",
    "anomaly_detection",
    "manual",
    "hybrid",
]

FraudAlertSort = Literal[
    "created_at",
    "risk_score",
    "severity",
    "updated_at",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class FraudAlertCreateRequest(BaseModel):
    """
    Create a fraud alert from an already calculated fraud signal.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    transaction_id: str

    risk_score: float = Field(
        ge=0,
        le=100,
    )

    risk_level: FraudRiskLevel

    severity: FraudAlertSeverity

    source: FraudAlertSource

    title: str = Field(
        min_length=3,
        max_length=255,
    )

    description: str = Field(
        min_length=3,
        max_length=5000,
    )

    reason_codes: list[str] = Field(
        default_factory=list,
        max_length=50,
    )

    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=100,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class FraudAlertUpdateRequest(BaseModel):
    """
    Update operational fraud-alert fields.
    """

    model_config = ConfigDict(extra="forbid")

    status: FraudAlertStatus | None = None

    severity: FraudAlertSeverity | None = None

    assigned_to: str | None = None

    investigation_notes: str | None = Field(
        default=None,
        max_length=10000,
    )

    resolution_reason: str | None = Field(
        default=None,
        max_length=5000,
    )

    metadata: dict[str, Any] | None = None


class FraudAlertStatusRequest(BaseModel):
    """
    Explicit status transition.
    """

    model_config = ConfigDict(extra="forbid")

    status: FraudAlertStatus

    reason: str | None = Field(
        default=None,
        max_length=5000,
    )


class FraudAlertAssignRequest(BaseModel):
    """
    Assign alert to an investigator.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str

    note: str | None = Field(
        default=None,
        max_length=2000,
    )


class FraudInvestigationRequest(BaseModel):
    """
    Start or update an investigation.
    """

    model_config = ConfigDict(extra="forbid")

    alert_id: str

    action: Literal[
        "start",
        "review",
        "escalate",
        "resolve",
        "dismiss",
    ]

    notes: str | None = Field(
        default=None,
        max_length=10000,
    )

    evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=100,
    )


class FraudAlertFilterRequest(BaseModel):
    """
    Filter fraud alerts.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    status: FraudAlertStatus | None = None

    severity: FraudAlertSeverity | None = None

    risk_level: FraudRiskLevel | None = None

    source: FraudAlertSource | None = None

    transaction_id: str | None = None

    assigned_to: str | None = None

    min_risk_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    max_risk_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    start_date: datetime | None = None

    end_date: datetime | None = None

    limit: int = Field(
        default=50,
        ge=1,
        le=200,
    )

    offset: int = Field(
        default=0,
        ge=0,
    )

    sort_by: FraudAlertSort = "created_at"

    sort_order: Literal["asc", "desc"] = "desc"

    @field_validator("max_risk_score")
    @classmethod
    def validate_score_range(
        cls,
        value: float | None,
    ) -> float | None:
        return value


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class FraudAlertEvidence(BaseModel):
    """
    Evidence supporting a fraud alert.
    """

    evidence_id: str | None = None

    type: str

    title: str

    description: str

    source: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class FraudAlertResponse(BaseModel):
    """
    Fraud alert representation.
    """

    id: str

    company_id: str

    transaction_id: str

    risk_score: float

    risk_level: FraudRiskLevel

    severity: FraudAlertSeverity

    status: FraudAlertStatus

    source: FraudAlertSource

    title: str

    description: str

    reason_codes: list[str] = Field(
        default_factory=list,
    )

    evidence: list[FraudAlertEvidence] = Field(
        default_factory=list,
    )

    assigned_to: str | None = None

    investigator_name: str | None = None

    investigation_notes: str | None = None

    resolution_reason: str | None = None

    created_at: datetime

    updated_at: datetime

    resolved_at: datetime | None = None


class FraudAlertListResponse(BaseModel):
    """
    Paginated fraud alerts.
    """

    items: list[FraudAlertResponse]

    total: int

    limit: int

    offset: int

    has_more: bool


class FraudAlertSummaryResponse(BaseModel):
    """
    Fraud-alert dashboard summary.
    """

    company_id: str

    total_alerts: int

    open_alerts: int

    investigating_alerts: int

    confirmed_fraud: int

    false_positives: int

    resolved_alerts: int

    critical_alerts: int

    high_risk_alerts: int

    average_risk_score: float

    fraud_rate_percent: float | None = None

    generated_at: datetime


class FraudInvestigationResponse(BaseModel):
    """
    Investigation result.
    """

    alert_id: str

    status: FraudAlertStatus

    action: str

    investigator_id: str | None = None

    notes: str | None = None

    evidence: list[FraudAlertEvidence] = Field(
        default_factory=list,
    )

    next_action: str | None = None

    updated_at: datetime


class FraudAlertAssignmentResponse(BaseModel):
    """
    Assignment result.
    """

    alert_id: str

    assigned_to: str

    assigned_by: str

    note: str | None = None

    assigned_at: datetime


class FraudAlertStatusResponse(BaseModel):
    """
    Status-transition response.
    """

    alert_id: str

    previous_status: FraudAlertStatus

    status: FraudAlertStatus

    reason: str | None = None

    changed_by: str

    changed_at: datetime


class FraudAlertHealthResponse(BaseModel):
    """
    Fraud alert subsystem health.
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

    Lazy imports help avoid circular dependencies while the
    application is being assembled incrementally.
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


def get_fraud_alert_service() -> Any:
    """
    Preferred fraud alert service.

    Expected implementation:

        backend.app.fraud.fraud_alert_service.FraudAlertService

    Fallback:

        backend.app.alerts.fraud_alert_service.FraudAlertService
    """

    try:
        from backend.app.fraud.fraud_alert_service import (
            FraudAlertService,
        )

        return FraudAlertService()

    except ImportError:
        pass

    try:
        from backend.app.alerts.fraud_alert_service import (
            FraudAlertService,
        )

        return FraudAlertService()

    except ImportError as exc:
        logger.exception(
            "Fraud alert service unavailable."
        )

        raise RuntimeError(
            "Fraud alert service is not configured."
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


def _can_view_fraud(
    current_user: Any,
) -> bool:
    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
        "finance_manager",
        "analyst",
        "auditor",
    }


def _can_manage_fraud(
    current_user: Any,
) -> bool:
    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
        "finance_manager",
    }


def _can_investigate_fraud(
    current_user: Any,
) -> bool:
    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
        "finance_manager",
        "analyst",
        "auditor",
    }


def require_fraud_view_access(
    current_user: Any,
) -> None:
    if not _can_view_fraud(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Fraud-alert access is required."
            ),
        )


def require_fraud_management_access(
    current_user: Any,
) -> None:
    if not _can_manage_fraud(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Fraud-alert management permission is required."
            ),
        )


def require_investigation_access(
    current_user: Any,
) -> None:
    if not _can_investigate_fraud(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Fraud investigation permission is required."
            ),
        )


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Mandatory tenant boundary.
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
            "Unauthorized fraud-alert access: "
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
# SERVICE CALL
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
                f"Fraud alert service does not implement "
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
            "Invalid fraud-alert operation: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid fraud-alert request.",
        ) from exc

    except PermissionError as exc:
        logger.warning(
            "Fraud-alert service denied operation."
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation is not permitted.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Fraud-alert service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fraud-alert operation failed.",
        ) from exc


# ============================================================================
# CREATE
# ============================================================================


@router.post(
    "",
    response_model=FraudAlertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create fraud alert",
)
async def create_fraud_alert(
    request: FraudAlertCreateRequest,
    current_user: Any = Depends(get_current_user()),
) -> FraudAlertResponse:
    """
    Create an alert from an existing fraud signal.

    Fraud scoring itself belongs to FraudService.
    """

    require_fraud_management_access(
        current_user
    )

    company_id = _get_company_id(
        current_user
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.company_id:
        validate_company_access(
            request.company_id,
            current_user,
        )

        company_id = request.company_id

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "create_alert",
        company_id=company_id,
        transaction_id=request.transaction_id,
        risk_score=request.risk_score,
        risk_level=request.risk_level,
        severity=request.severity,
        source=request.source,
        title=request.title,
        description=request.description,
        reason_codes=request.reason_codes,
        evidence=request.evidence,
        metadata=request.metadata,
        actor_id=_get_user_id(
            current_user
        ),
        user=current_user,
    )

    return FraudAlertResponse.model_validate(
        result
    )


# ============================================================================
# LIST
# ============================================================================


@router.get(
    "",
    response_model=FraudAlertListResponse,
    summary="List fraud alerts",
)
async def list_fraud_alerts(
    company_id: str | None = None,
    status_filter: FraudAlertStatus | None = Query(
        default=None,
        alias="status",
    ),
    severity: FraudAlertSeverity | None = None,
    risk_level: FraudRiskLevel | None = None,
    source: FraudAlertSource | None = None,
    transaction_id: str | None = None,
    assigned_to: str | None = None,
    min_risk_score: float | None = Query(
        default=None,
        ge=0,
        le=100,
    ),
    max_risk_score: float | None = Query(
        default=None,
        ge=0,
        le=100,
    ),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    sort_by: FraudAlertSort = "created_at",
    sort_order: Literal["asc", "desc"] = "desc",
    current_user: Any = Depends(get_current_user()),
) -> FraudAlertListResponse:
    """
    List fraud alerts with tenant isolation.
    """

    require_fraud_view_access(
        current_user
    )

    user_company_id = _get_company_id(
        current_user
    )

    if not user_company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    target_company_id = (
        company_id or user_company_id
    )

    validate_company_access(
        target_company_id,
        current_user,
    )

    if (
        min_risk_score is not None
        and max_risk_score is not None
        and min_risk_score > max_risk_score
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "min_risk_score cannot exceed "
                "max_risk_score."
            ),
        )

    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "list_alerts",
        company_id=target_company_id,
        status=status_filter,
        severity=severity,
        risk_level=risk_level,
        source=source,
        transaction_id=transaction_id,
        assigned_to=assigned_to,
        min_risk_score=min_risk_score,
        max_risk_score=max_risk_score,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order,
        user=current_user,
    )

    return FraudAlertListResponse.model_validate(
        result
    )


# ============================================================================
# SUMMARY
# ============================================================================


@router.get(
    "/summary/{company_id}",
    response_model=FraudAlertSummaryResponse,
    summary="Get fraud alert summary",
)
async def fraud_alert_summary(
    company_id: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    current_user: Any = Depends(get_current_user()),
) -> FraudAlertSummaryResponse:
    """
    Fraud-alert dashboard summary.
    """

    require_fraud_view_access(
        current_user
    )

    validate_company_access(
        company_id,
        current_user,
    )

    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "summary",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        user=current_user,
    )

    return FraudAlertSummaryResponse.model_validate(
        result
    )


# ============================================================================
# GET SINGLE ALERT
# ============================================================================


@router.get(
    "/{alert_id}",
    response_model=FraudAlertResponse,
    summary="Get fraud alert",
)
async def get_fraud_alert(
    alert_id: str,
    current_user: Any = Depends(get_current_user()),
) -> FraudAlertResponse:
    """
    Get a single fraud alert.
    """

    require_fraud_view_access(
        current_user
    )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "get_alert",
        alert_id=alert_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fraud alert not found.",
        )

    company_id = (
        result.get("company_id")
        if isinstance(result, dict)
        else getattr(
            result,
            "company_id",
            None,
        )
    )

    if company_id:
        validate_company_access(
            str(company_id),
            current_user,
        )

    return FraudAlertResponse.model_validate(
        result
    )


# ============================================================================
# UPDATE
# ============================================================================


@router.patch(
    "/{alert_id}",
    response_model=FraudAlertResponse,
    summary="Update fraud alert",
)
async def update_fraud_alert(
    alert_id: str,
    request: FraudAlertUpdateRequest,
    current_user: Any = Depends(get_current_user()),
) -> FraudAlertResponse:
    """
    Update operational fraud-alert information.
    """

    require_fraud_management_access(
        current_user
    )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "update_alert",
        alert_id=alert_id,
        status=request.status,
        severity=request.severity,
        assigned_to=request.assigned_to,
        investigation_notes=(
            request.investigation_notes
        ),
        resolution_reason=(
            request.resolution_reason
        ),
        metadata=request.metadata,
        actor_id=_get_user_id(
            current_user
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fraud alert not found.",
        )

    return FraudAlertResponse.model_validate(
        result
    )


# ============================================================================
# STATUS TRANSITION
# ============================================================================


@router.patch(
    "/{alert_id}/status",
    response_model=FraudAlertStatusResponse,
    summary="Change fraud alert status",
)
async def change_fraud_alert_status(
    alert_id: str,
    request: FraudAlertStatusRequest,
    current_user: Any = Depends(get_current_user()),
) -> FraudAlertStatusResponse:
    """
    Change fraud-alert status.

    The service must validate legal state transitions.
    """

    require_investigation_access(
        current_user
    )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "change_status",
        alert_id=alert_id,
        new_status=request.status,
        reason=request.reason,
        actor_id=_get_user_id(
            current_user
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fraud alert not found.",
        )

    return FraudAlertStatusResponse.model_validate(
        result
    )


# ============================================================================
# ASSIGN
# ============================================================================


@router.post(
    "/{alert_id}/assign",
    response_model=FraudAlertAssignmentResponse,
    summary="Assign fraud alert",
)
async def assign_fraud_alert(
    alert_id: str,
    request: FraudAlertAssignRequest,
    current_user: Any = Depends(get_current_user()),
) -> FraudAlertAssignmentResponse:
    """
    Assign alert to a fraud investigator.
    """

    require_fraud_management_access(
        current_user
    )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "assign_alert",
        alert_id=alert_id,
        user_id=request.user_id,
        note=request.note,
        assigned_by=_get_user_id(
            current_user
        ),
        actor=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fraud alert not found.",
        )

    return FraudAlertAssignmentResponse.model_validate(
        result
    )


# ============================================================================
# INVESTIGATION
# ============================================================================


@router.post(
    "/{alert_id}/investigate",
    response_model=FraudInvestigationResponse,
    summary="Investigate fraud alert",
)
async def investigate_fraud_alert(
    alert_id: str,
    request: FraudInvestigationRequest,
    current_user: Any = Depends(get_current_user()),
) -> FraudInvestigationResponse:
    """
    Start, review, escalate, resolve, or dismiss an alert.

    Human investigation remains authoritative for final fraud decisions.
    """

    require_investigation_access(
        current_user
    )

    if request.alert_id != alert_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Path alert_id does not match "
                "request alert_id."
            ),
        )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "investigate",
        alert_id=alert_id,
        action=request.action,
        notes=request.notes,
        evidence_ids=request.evidence_ids,
        investigator_id=_get_user_id(
            current_user
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fraud alert not found.",
        )

    return FraudInvestigationResponse.model_validate(
        result
    )


# ============================================================================
# ESCALATE
# ============================================================================


@router.post(
    "/{alert_id}/escalate",
    response_model=FraudInvestigationResponse,
    summary="Escalate fraud alert",
)
async def escalate_fraud_alert(
    alert_id: str,
    notes: str | None = Query(
        default=None,
        max_length=5000,
    ),
    current_user: Any = Depends(get_current_user()),
) -> FraudInvestigationResponse:
    """
    Escalate an alert for higher-level review.
    """

    require_investigation_access(
        current_user
    )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "escalate",
        alert_id=alert_id,
        notes=notes,
        actor_id=_get_user_id(
            current_user
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fraud alert not found.",
        )

    return FraudInvestigationResponse.model_validate(
        result
    )


# ============================================================================
# EVIDENCE
# ============================================================================


@router.get(
    "/{alert_id}/evidence",
    response_model=list[FraudAlertEvidence],
    summary="Get fraud alert evidence",
)
async def get_fraud_alert_evidence(
    alert_id: str,
    current_user: Any = Depends(get_current_user()),
) -> list[FraudAlertEvidence]:
    """
    Retrieve evidence associated with an alert.
    """

    require_fraud_view_access(
        current_user
    )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "get_evidence",
        alert_id=alert_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fraud alert not found.",
        )

    return [
        FraudAlertEvidence.model_validate(
            item
        )
        for item in result
    ]


# ============================================================================
# TRANSACTION ALERTS
# ============================================================================


@router.get(
    "/transaction/{transaction_id}",
    response_model=list[FraudAlertResponse],
    summary="Get alerts for transaction",
)
async def transaction_fraud_alerts(
    transaction_id: str,
    current_user: Any = Depends(get_current_user()),
) -> list[FraudAlertResponse]:
    """
    Get all fraud alerts associated with a transaction.
    """

    require_fraud_view_access(
        current_user
    )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "get_transaction_alerts",
        transaction_id=transaction_id,
        user=current_user,
    )

    return [
        FraudAlertResponse.model_validate(
            item
        )
        for item in (result or [])
    ]


# ============================================================================
# HIGH-RISK ALERTS
# ============================================================================


@router.get(
    "/high-risk/{company_id}",
    response_model=FraudAlertListResponse,
    summary="Get high-risk fraud alerts",
)
async def high_risk_fraud_alerts(
    company_id: str,
    minimum_score: float = Query(
        default=70,
        ge=0,
        le=100,
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    current_user: Any = Depends(get_current_user()),
) -> FraudAlertListResponse:
    """
    Return open/high-risk fraud alerts.
    """

    require_fraud_view_access(
        current_user
    )

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "high_risk_alerts",
        company_id=company_id,
        minimum_score=minimum_score,
        limit=limit,
        user=current_user,
    )

    return FraudAlertListResponse.model_validate(
        result
    )


# ============================================================================
# OPEN ALERTS
# ============================================================================


@router.get(
    "/open/{company_id}",
    response_model=FraudAlertListResponse,
    summary="Get open fraud alerts",
)
async def open_fraud_alerts(
    company_id: str,
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    current_user: Any = Depends(get_current_user()),
) -> FraudAlertListResponse:
    """
    Return currently open/investigating alerts.
    """

    require_fraud_view_access(
        current_user
    )

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_fraud_alert_service()

    result = await _call_service(
        service,
        "open_alerts",
        company_id=company_id,
        limit=limit,
        user=current_user,
    )

    return FraudAlertListResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=FraudAlertHealthResponse,
    summary="Fraud alert service health",
)
async def fraud_alert_health() -> FraudAlertHealthResponse:
    """
    Fraud alert service health check.
    """

    try:
        service = get_fraud_alert_service()

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

        return FraudAlertHealthResponse(
            status="healthy",
            service="fraud-alert",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Fraud alert health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Fraud alert service is unavailable."
            ),
        )