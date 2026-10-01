"""
FinCo AI - Financial Copilot API

Responsibilities:
- Natural-language financial questions
- Copilot conversation requests
- Financial analysis requests
- RAG-grounded document questions
- Forecast requests
- Fraud/risk investigation requests
- What-if analysis requests
- Recommendation requests
- Citation-aware responses
- Guardrail enforcement
- RBAC + tenant isolation
- Rate limiting
- Audit logging

Important:
    This API layer does not perform financial calculations itself.
    Deterministic calculations belong to app.financial.* modules.
    Agent orchestration belongs to app.agents.*.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.core.audit import AuditService
from app.core.exceptions import (
    AgentError,
    AgentGuardrailError,
    AgentPlanningError,
    AgentRoutingError,
    AuthorizationError,
    FinCoException,
    GuardrailError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitExceededError,
)
from app.core.guardrails import GuardrailService
from app.core.logging import (
    get_logger,
    generate_request_id,
    set_request_context,
    clear_request_context,
)
from app.core.permissions import (
    AgentPermissionService,
    Permission,
    PermissionService,
    TenantAccessService,
    AuthorizationContext,
)
from app.core.rate_limiter import copilot_rate_limit
from app.database.session import get_db


# ============================================================
# Router
# ============================================================

router = APIRouter(
    prefix="/copilot",
    tags=["Financial Copilot"],
)

logger = get_logger(__name__)


# ============================================================
# Dependency: Current User
# ============================================================

def get_current_user(
    request: Request,
) -> AuthorizationContext:
    """
    Retrieve authorization context.

    The production implementation should populate this context
    from the authenticated JWT/database user.

    This dependency intentionally does not trust company_id
    supplied by the client as proof of authorization.
    """

    user = getattr(
        request.state,
        "authorization_context",
        None,
    )

    if user is None:
        # DEMO MODE: Create a demo user context
        # In production, this should validate JWT tokens
        auth_header = request.headers.get("Authorization", "")
        
        if auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            
            # Accept demo-token for testing
            if token == "demo-token" or not token:
                logger.info("Using demo authentication mode")
                user = AuthorizationContext(
                    user_id="demo-user-001",
                    company_id="demo-company-001",
                    roles=["admin", "analyst"],
                    permissions=[
                        Permission.COPILOT_USE,
                        Permission.FINANCIAL_READ,
                        Permission.DOCUMENT_READ,
                        Permission.FORECAST_READ,
                        Permission.FRAUD_READ,
                    ],
                )
                request.state.authorization_context = user
                return user
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Use 'Bearer demo-token' for testing.",
        )

    return user


# ============================================================
# Request Schemas
# ============================================================

class CopilotRequest(BaseModel):
    """
    Main financial copilot request.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=12000,
        description="Natural-language financial question.",
    )

    company_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    conversation_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    mode: str = Field(
        default="auto",
        description=(
            "Copilot mode: auto, financial, rag, forecast, "
            "fraud, risk, what_if, recommendation."
        ),
    )

    include_citations: bool = True

    include_metrics: bool = True

    include_recommendations: bool = False

    include_sources: bool = True

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @field_validator("message")
    @classmethod
    def validate_message(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Message cannot be empty."
            )

        return value

    @field_validator("mode")
    @classmethod
    def validate_mode(
        cls,
        value: str,
    ) -> str:

        allowed = {
            "auto",
            "financial",
            "rag",
            "forecast",
            "fraud",
            "risk",
            "what_if",
            "recommendation",
        }

        value = value.lower().strip()

        if value not in allowed:
            raise ValueError(
                f"Unsupported copilot mode: {value}"
            )

        return value


class FinancialAnalysisRequest(BaseModel):

    company_id: Optional[str] = None

    question: str = Field(
        ...,
        min_length=1,
        max_length=12000,
    )

    period: Optional[str] = None

    compare_period: Optional[str] = None

    include_ratios: bool = True

    include_root_cause: bool = True


class DocumentQuestionRequest(BaseModel):

    company_id: Optional[str] = None

    question: str = Field(
        ...,
        min_length=1,
        max_length=12000,
    )

    document_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
    )

    include_citations: bool = True


class ForecastRequest(BaseModel):

    company_id: Optional[str] = None

    metric: str = Field(
        ...,
        description=(
            "Revenue, profit, cash flow, liquidity, etc."
        ),
    )

    horizon: int = Field(
        default=6,
        ge=1,
        le=36,
    )

    frequency: str = Field(
        default="monthly",
    )

    include_confidence_interval: bool = True


class FraudInvestigationRequest(BaseModel):

    company_id: Optional[str] = None

    transaction_id: Optional[str] = None

    question: Optional[str] = None

    include_explanation: bool = True

    include_related_transactions: bool = True


class WhatIfRequest(BaseModel):

    company_id: Optional[str] = None

    scenario: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    horizon: int = Field(
        default=12,
        ge=1,
        le=60,
    )

    include_sensitivity: bool = True


class RecommendationRequest(BaseModel):

    company_id: Optional[str] = None

    area: str = Field(
        ...,
        description=(
            "financial, cost, risk, liquidity, "
            "fraud, forecast, general"
        ),
    )

    context: Optional[str] = Field(
        default=None,
        max_length=8000,
    )

    include_impact_estimate: bool = True

    require_human_review: bool = False


# ============================================================
# Response Schemas
# ============================================================

class Citation(BaseModel):

    citation_id: str

    document_id: Optional[str] = None

    document_name: Optional[str] = None

    page: Optional[int] = None

    chunk_id: Optional[str] = None

    text: Optional[str] = None

    relevance_score: Optional[float] = None


class CopilotResponse(BaseModel):

    request_id: str

    conversation_id: str

    company_id: str

    answer: str

    mode: str

    confidence: Optional[float] = None

    grounded: bool = False

    requires_human_review: bool = False

    metrics: dict[str, Any] = Field(
        default_factory=dict
    )

    citations: list[Citation] = Field(
        default_factory=list
    )

    sources: list[dict[str, Any]] = Field(
        default_factory=list
    )

    recommendations: list[dict[str, Any]] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    execution_time_ms: float = 0.0


class HealthResponse(BaseModel):

    status: str

    service: str

    timestamp: datetime


# ============================================================
# Internal Helpers
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_conversation_id(
    existing: Optional[str],
) -> str:

    if existing:
        return existing

    return str(uuid.uuid4())


def resolve_company_id(
    requested_company_id: Optional[str],
    current_user: AuthorizationContext,
) -> str:
    """
    Resolve and validate company tenant.

    Client-supplied company_id is never sufficient by itself.
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


def ensure_copilot_permission(
    current_user: AuthorizationContext,
) -> None:

    PermissionService.require_permission(
        current_user,
        Permission.COPILOT_USE,
    )


def normalize_agent_result(
    result: Any,
) -> dict[str, Any]:
    """
    Normalize different agent/service return formats.
    """

    if result is None:
        return {}

    if isinstance(result, dict):
        return result

    if hasattr(result, "model_dump"):
        return result.model_dump()

    if hasattr(result, "dict"):
        return result.dict()

    if hasattr(result, "__dict__"):
        return dict(result.__dict__)

    return {
        "answer": str(result)
    }


def extract_answer(
    result: dict[str, Any],
) -> str:

    for key in (
        "answer",
        "response",
        "message",
        "content",
        "output",
    ):

        value = result.get(key)

        if value:
            return str(value)

    return (
        "The copilot completed the requested analysis "
        "but did not return a textual explanation."
    )


def extract_citations(
    result: dict[str, Any],
) -> list[dict[str, Any]]:

    citations = result.get(
        "citations",
        [],
    )

    if not isinstance(citations, list):
        return []

    return citations


def extract_metrics(
    result: dict[str, Any],
) -> dict[str, Any]:

    metrics = result.get(
        "metrics",
        {},
    )

    if isinstance(metrics, dict):
        return metrics

    return {}


def extract_warnings(
    result: dict[str, Any],
) -> list[str]:

    warnings = result.get(
        "warnings",
        [],
    )

    if not isinstance(warnings, list):
        return []

    return [
        str(item)
        for item in warnings
    ]


# ============================================================
# Agent Service Loader
# ============================================================

def get_copilot_orchestrator(
    db: Session,
):
    """
    Load the application's agent orchestrator.

    Keeping the import inside the function avoids unnecessary
    startup coupling and makes the API easier to unit test.
    """

    try:

        from app.agents.orchestrator import (
            AgentOrchestrator,
        )

        return AgentOrchestrator(
            db=db
        )

    except ImportError as exc:

        logger.exception(
            "Unable to load AgentOrchestrator"
        )

        raise AgentError(
            "Copilot orchestration service is unavailable."
        ) from exc


# ============================================================
# Generic Agent Execution
# ============================================================

async def execute_copilot(
    *,
    db: Session,
    message: str,
    company_id: str,
    conversation_id: str,
    mode: str,
    current_user: AuthorizationContext,
    request_id: str,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:

    metadata = metadata or {}

    # --------------------------------------------------------
    # Agent permission boundary
    # --------------------------------------------------------

    AgentPermissionService.require_agent_access(
        current_user,
        "supervisor_agent",
    )

    # --------------------------------------------------------
    # Guardrails
    # --------------------------------------------------------

    guardrails = GuardrailService()

    input_result = guardrails.check_input(
        message
    )

    if input_result.blocked:
        raise AgentGuardrailError(
            input_result.message
        )

    if input_result.sanitized_text:
        message = input_result.sanitized_text

    # --------------------------------------------------------
    # Orchestrator
    # --------------------------------------------------------

    try:
        orchestrator = get_copilot_orchestrator(
            db
        )
    except AgentError as exc:
        # If orchestrator is not available, provide a helpful demo response
        logger.warning("Orchestrator unavailable, using fallback response: %s", exc)
        return {
            "answer": (
                "The AI copilot orchestration service is currently initializing. "
                f"Your question was: '{message}'. "
                "In a fully configured system, this would route to specialized agents "
                "for fraud detection, forecasting, financial analysis, and document intelligence."
            ),
            "mode": mode,
            "grounded": False,
            "confidence": 0.0,
            "requires_human_review": False,
            "metadata": {
                "fallback": True,
                "reason": "orchestrator_unavailable",
            },
        }

    payload = {
        "message": message,
        "company_id": company_id,
        "conversation_id": conversation_id,
        "mode": mode,
        "user_id": current_user.user_id,
        "request_id": request_id,
        "metadata": metadata,
    }

    try:

        if hasattr(
            orchestrator,
            "run",
        ):

            result = orchestrator.run(
                payload
            )

        elif hasattr(
            orchestrator,
            "execute",
        ):

            result = orchestrator.execute(
                payload
            )

        else:

            raise AgentError(
                "Agent orchestrator has no run/execute method."
            )

    except (
        AgentRoutingError,
        AgentPlanningError,
        AgentGuardrailError,
        AgentError,
    ) as exc:
        # Provide helpful error response
        logger.error("Agent execution error: %s", exc)
        return {
            "answer": (
                f"I encountered an issue processing your request: {str(exc)}. "
                "This is likely a configuration or setup issue. Please ensure all "
                "required services (RAG, agents, models) are properly initialized."
            ),
            "mode": mode,
            "grounded": False,
            "confidence": 0.0,
            "requires_human_review": True,
            "warnings": [str(exc)],
            "metadata": {
                "error": True,
                "error_type": type(exc).__name__,
            },
        }

    except Exception as exc:

        logger.exception(
            "Copilot execution failed",
            extra={
                "request_id": request_id,
                "company_id": company_id,
                "user_id": current_user.user_id,
            },
        )

        raise AgentError(
            "Unable to complete copilot request."
        ) from exc

    normalized = normalize_agent_result(
        result
    )

    # --------------------------------------------------------
    # Output guardrail
    # --------------------------------------------------------

    answer = extract_answer(
        normalized
    )

    output_result = guardrails.check_output(
        answer
    )

    if output_result.blocked:

        raise AgentGuardrailError(
            "Copilot response was blocked by output guardrails."
        )

    normalized["answer"] = (
        output_result.sanitized_text
        or answer
    )

    if output_result.warnings:

        normalized.setdefault(
            "warnings",
            []
        )

        normalized["warnings"].extend(
            output_result.warnings
        )

    return normalized


# ============================================================
# Audit Helper
# ============================================================

def audit_copilot_execution(
    *,
    db: Session,
    current_user: AuthorizationContext,
    company_id: str,
    request_id: str,
    conversation_id: str,
    mode: str,
    execution_time_ms: float,
    success: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> None:

    try:

        audit = AuditService(
            db
        )

        audit.agent_execution(
            user_id=current_user.user_id,
            company_id=company_id,
            agent="supervisor_agent",
            action="copilot_request",
            metadata={
                "request_id": request_id,
                "conversation_id": conversation_id,
                "mode": mode,
                "execution_time_ms": execution_time_ms,
                "success": success,
                **(metadata or {}),
            },
        )

    except Exception:

        # Audit failure should be logged loudly.
        # Whether it should block business execution depends
        # on the organization's compliance policy.
        logger.exception(
            "Failed to create copilot audit event",
            extra={
                "request_id": request_id,
                "company_id": company_id,
            },
        )


# ============================================================
# POST /copilot/chat
# ============================================================

@router.post(
    "/chat",
    response_model=CopilotResponse,
)
async def copilot_chat(
    payload: CopilotRequest,
    request: Request,
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
    _: Any = Depends(copilot_rate_limit),
) -> CopilotResponse:
    """
    Main financial copilot endpoint.

    Example questions:

    - "Why did profit decline this quarter?"
    - "What is causing our cash-flow problem?"
    - "Show me revenue forecast for the next six months."
    - "Investigate suspicious transactions."
    - "What happens if revenue drops 10%?"
    - "What recommendations can improve liquidity?"
    """

    request_id = generate_request_id()

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=current_user.company_id,
    )

    start = time.perf_counter()

    company_id = resolve_company_id(
        payload.company_id,
        current_user,
    )

    ensure_copilot_permission(
        current_user
    )

    conversation_id = create_conversation_id(
        payload.conversation_id
    )

    try:

        result = await execute_copilot(
            db=db,
            message=payload.message,
            company_id=company_id,
            conversation_id=conversation_id,
            mode=payload.mode,
            current_user=current_user,
            request_id=request_id,
            metadata=payload.metadata,
        )

        execution_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        audit_copilot_execution(
            db=db,
            current_user=current_user,
            company_id=company_id,
            request_id=request_id,
            conversation_id=conversation_id,
            mode=payload.mode,
            execution_time_ms=execution_time_ms,
            success=True,
            metadata={
                "include_citations":
                    payload.include_citations,
                "include_metrics":
                    payload.include_metrics,
            },
        )

        return CopilotResponse(
            request_id=request_id,
            conversation_id=conversation_id,
            company_id=company_id,
            answer=extract_answer(result),
            mode=result.get(
                "mode",
                payload.mode,
            ),
            confidence=result.get(
                "confidence"
            ),
            grounded=bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            requires_human_review=bool(
                result.get(
                    "requires_human_review",
                    False,
                )
            ),
            metrics=(
                extract_metrics(result)
                if payload.include_metrics
                else {}
            ),
            citations=(
                extract_citations(result)
                if payload.include_citations
                else []
            ),
            sources=(
                result.get(
                    "sources",
                    [],
                )
                if payload.include_sources
                else []
            ),
            recommendations=(
                result.get(
                    "recommendations",
                    [],
                )
                if payload.include_recommendations
                else []
            ),
            warnings=extract_warnings(
                result
            ),
            metadata=result.get(
                "metadata",
                {},
            ),
            execution_time_ms=round(
                execution_time_ms,
                2,
            ),
        )

    except FinCoException:

        execution_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        audit_copilot_execution(
            db=db,
            current_user=current_user,
            company_id=company_id,
            request_id=request_id,
            conversation_id=conversation_id,
            mode=payload.mode,
            execution_time_ms=execution_time_ms,
            success=False,
        )

        raise

    except Exception as exc:

        logger.exception(
            "Unexpected copilot API error",
            extra={
                "request_id": request_id,
                "company_id": company_id,
                "user_id": current_user.user_id,
            },
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Copilot request failed.",
        ) from exc

    finally:

        clear_request_context()


# ============================================================
# POST /copilot/financial-analysis
# ============================================================

@router.post(
    "/financial-analysis",
    response_model=CopilotResponse,
)
async def financial_analysis(
    payload: FinancialAnalysisRequest,
    request: Request,
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
    _: Any = Depends(copilot_rate_limit),
) -> CopilotResponse:
    """
    Financial-analysis focused copilot workflow.
    """

    company_id = resolve_company_id(
        payload.company_id,
        current_user,
    )

    PermissionService.require_permission(
        current_user,
        Permission.FINANCIAL_READ,
    )

    message = payload.question

    if payload.period:

        message += (
            f"\nAnalyze period: {payload.period}"
        )

    if payload.compare_period:

        message += (
            f"\nCompare against: "
            f"{payload.compare_period}"
        )

    if payload.include_ratios:

        message += (
            "\nInclude relevant financial ratios."
        )

    if payload.include_root_cause:

        message += (
            "\nPerform root-cause analysis "
            "for material changes."
        )

    request_id = generate_request_id()

    conversation_id = str(
        uuid.uuid4()
    )

    start = time.perf_counter()

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=company_id,
    )

    try:

        result = await execute_copilot(
            db=db,
            message=message,
            company_id=company_id,
            conversation_id=conversation_id,
            mode="financial",
            current_user=current_user,
            request_id=request_id,
            metadata={
                "period": payload.period,
                "compare_period":
                    payload.compare_period,
            },
        )

        execution_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return CopilotResponse(
            request_id=request_id,
            conversation_id=conversation_id,
            company_id=company_id,
            answer=extract_answer(result),
            mode="financial",
            confidence=result.get(
                "confidence"
            ),
            grounded=bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            requires_human_review=bool(
                result.get(
                    "requires_human_review",
                    False,
                )
            ),
            metrics=extract_metrics(
                result
            ),
            citations=extract_citations(
                result
            ),
            sources=result.get(
                "sources",
                [],
            ),
            recommendations=result.get(
                "recommendations",
                [],
            ),
            warnings=extract_warnings(
                result
            ),
            metadata=result.get(
                "metadata",
                {},
            ),
            execution_time_ms=round(
                execution_time_ms,
                2,
            ),
        )

    finally:

        clear_request_context()


# ============================================================
# POST /copilot/document-question
# ============================================================

@router.post(
    "/document-question",
    response_model=CopilotResponse,
)
async def document_question(
    payload: DocumentQuestionRequest,
    request: Request,
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
    _: Any = Depends(copilot_rate_limit),
) -> CopilotResponse:
    """
    Ask questions against company documents using RAG.
    """

    company_id = resolve_company_id(
        payload.company_id,
        current_user,
    )

    PermissionService.require_permission(
        current_user,
        Permission.DOCUMENT_READ,
    )

    message = payload.question

    if payload.document_ids:

        message += (
            "\nRestrict evidence to these documents: "
            + ", ".join(payload.document_ids)
        )

    message += (
        f"\nRetrieve up to {payload.top_k} relevant "
        "evidence chunks."
    )

    request_id = generate_request_id()

    conversation_id = str(
        uuid.uuid4()
    )

    start = time.perf_counter()

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=company_id,
    )

    try:

        result = await execute_copilot(
            db=db,
            message=message,
            company_id=company_id,
            conversation_id=conversation_id,
            mode="rag",
            current_user=current_user,
            request_id=request_id,
            metadata={
                "document_ids":
                    payload.document_ids,
                "top_k":
                    payload.top_k,
            },
        )

        execution_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return CopilotResponse(
            request_id=request_id,
            conversation_id=conversation_id,
            company_id=company_id,
            answer=extract_answer(result),
            mode="rag",
            confidence=result.get(
                "confidence"
            ),
            grounded=bool(
                result.get(
                    "grounded",
                    True,
                )
            ),
            requires_human_review=bool(
                result.get(
                    "requires_human_review",
                    False,
                )
            ),
            metrics=extract_metrics(
                result
            ),
            citations=(
                extract_citations(result)
                if payload.include_citations
                else []
            ),
            sources=result.get(
                "sources",
                [],
            ),
            recommendations=result.get(
                "recommendations",
                [],
            ),
            warnings=extract_warnings(
                result
            ),
            metadata=result.get(
                "metadata",
                {},
            ),
            execution_time_ms=round(
                execution_time_ms,
                2,
            ),
        )

    finally:

        clear_request_context()


# ============================================================
# POST /copilot/forecast
# ============================================================

@router.post(
    "/forecast",
    response_model=CopilotResponse,
)
async def forecast_analysis(
    payload: ForecastRequest,
    request: Request,
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
    _: Any = Depends(copilot_rate_limit),
) -> CopilotResponse:
    """
    Forecast financial metrics through the forecast agent.
    """

    company_id = resolve_company_id(
        payload.company_id,
        current_user,
    )

    PermissionService.require_permission(
        current_user,
        Permission.FORECAST_READ,
    )

    message = (
        f"Forecast {payload.metric} for the next "
        f"{payload.horizon} {payload.frequency} periods."
    )

    if payload.include_confidence_interval:

        message += (
            " Include confidence intervals and "
            "forecast uncertainty."
        )

    request_id = generate_request_id()

    conversation_id = str(
        uuid.uuid4()
    )

    start = time.perf_counter()

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=company_id,
    )

    try:

        result = await execute_copilot(
            db=db,
            message=message,
            company_id=company_id,
            conversation_id=conversation_id,
            mode="forecast",
            current_user=current_user,
            request_id=request_id,
            metadata={
                "metric": payload.metric,
                "horizon": payload.horizon,
                "frequency": payload.frequency,
            },
        )

        execution_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return CopilotResponse(
            request_id=request_id,
            conversation_id=conversation_id,
            company_id=company_id,
            answer=extract_answer(result),
            mode="forecast",
            confidence=result.get(
                "confidence"
            ),
            grounded=bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            requires_human_review=bool(
                result.get(
                    "requires_human_review",
                    False,
                )
            ),
            metrics=extract_metrics(
                result
            ),
            citations=extract_citations(
                result
            ),
            sources=result.get(
                "sources",
                [],
            ),
            recommendations=result.get(
                "recommendations",
                [],
            ),
            warnings=extract_warnings(
                result
            ),
            metadata=result.get(
                "metadata",
                {},
            ),
            execution_time_ms=round(
                execution_time_ms,
                2,
            ),
        )

    finally:

        clear_request_context()


# ============================================================
# POST /copilot/fraud-investigation
# ============================================================

@router.post(
    "/fraud-investigation",
    response_model=CopilotResponse,
)
async def fraud_investigation(
    payload: FraudInvestigationRequest,
    request: Request,
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
    _: Any = Depends(copilot_rate_limit),
) -> CopilotResponse:
    """
    Investigate suspicious financial activity.
    """

    company_id = resolve_company_id(
        payload.company_id,
        current_user,
    )

    PermissionService.require_permission(
        current_user,
        Permission.FRAUD_READ,
    )

    message = (
        payload.question
        or "Investigate suspicious financial activity."
    )

    if payload.transaction_id:

        message += (
            f"\nTransaction ID: "
            f"{payload.transaction_id}"
        )

    if payload.include_explanation:

        message += (
            "\nExplain the fraud/anomaly signals "
            "and supporting evidence."
        )

    if payload.include_related_transactions:

        message += (
            "\nIdentify relevant related transactions."
        )

    request_id = generate_request_id()

    conversation_id = str(
        uuid.uuid4()
    )

    start = time.perf_counter()

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=company_id,
    )

    try:

        result = await execute_copilot(
            db=db,
            message=message,
            company_id=company_id,
            conversation_id=conversation_id,
            mode="fraud",
            current_user=current_user,
            request_id=request_id,
            metadata={
                "transaction_id":
                    payload.transaction_id,
            },
        )

        execution_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return CopilotResponse(
            request_id=request_id,
            conversation_id=conversation_id,
            company_id=company_id,
            answer=extract_answer(result),
            mode="fraud",
            confidence=result.get(
                "confidence"
            ),
            grounded=bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            requires_human_review=True,
            metrics=extract_metrics(
                result
            ),
            citations=extract_citations(
                result
            ),
            sources=result.get(
                "sources",
                [],
            ),
            recommendations=result.get(
                "recommendations",
                [],
            ),
            warnings=extract_warnings(
                result
            ),
            metadata=result.get(
                "metadata",
                {},
            ),
            execution_time_ms=round(
                execution_time_ms,
                2,
            ),
        )

    finally:

        clear_request_context()


# ============================================================
# POST /copilot/what-if
# ============================================================

@router.post(
    "/what-if",
    response_model=CopilotResponse,
)
async def what_if_analysis(
    payload: WhatIfRequest,
    request: Request,
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
    _: Any = Depends(copilot_rate_limit),
) -> CopilotResponse:
    """
    Run scenario / what-if analysis.
    """

    company_id = resolve_company_id(
        payload.company_id,
        current_user,
    )

    PermissionService.require_permission(
        current_user,
        Permission.WHAT_IF_RUN,
    )

    message = (
        "Run a financial what-if scenario.\n"
        f"Scenario: {payload.scenario}\n"
        f"Horizon: {payload.horizon} periods."
    )

    if payload.include_sensitivity:

        message += (
            "\nInclude sensitivity analysis."
        )

    request_id = generate_request_id()

    conversation_id = str(
        uuid.uuid4()
    )

    start = time.perf_counter()

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=company_id,
    )

    try:

        result = await execute_copilot(
            db=db,
            message=message,
            company_id=company_id,
            conversation_id=conversation_id,
            mode="what_if",
            current_user=current_user,
            request_id=request_id,
            metadata={
                "horizon": payload.horizon,
            },
        )

        execution_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return CopilotResponse(
            request_id=request_id,
            conversation_id=conversation_id,
            company_id=company_id,
            answer=extract_answer(result),
            mode="what_if",
            confidence=result.get(
                "confidence"
            ),
            grounded=bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            requires_human_review=bool(
                result.get(
                    "requires_human_review",
                    False,
                )
            ),
            metrics=extract_metrics(
                result
            ),
            citations=extract_citations(
                result
            ),
            sources=result.get(
                "sources",
                [],
            ),
            recommendations=result.get(
                "recommendations",
                [],
            ),
            warnings=extract_warnings(
                result
            ),
            metadata=result.get(
                "metadata",
                {},
            ),
            execution_time_ms=round(
                execution_time_ms,
                2,
            ),
        )

    finally:

        clear_request_context()


# ============================================================
# POST /copilot/recommendations
# ============================================================

@router.post(
    "/recommendations",
    response_model=CopilotResponse,
)
async def recommendations(
    payload: RecommendationRequest,
    request: Request,
    current_user: AuthorizationContext = Depends(
        get_current_user
    ),
    db: Session = Depends(get_db),
    _: Any = Depends(copilot_rate_limit),
) -> CopilotResponse:
    """
    Generate finance/risk/cost recommendations.
    """

    company_id = resolve_company_id(
        payload.company_id,
        current_user,
    )

    PermissionService.require_permission(
        current_user,
        Permission.RECOMMENDATION_READ,
    )

    message = (
        f"Generate financial recommendations "
        f"for the following area: {payload.area}."
    )

    if payload.context:

        message += (
            f"\nContext: {payload.context}"
        )

    if payload.include_impact_estimate:

        message += (
            "\nEstimate financial impact where "
            "sufficient data exists."
        )

    if payload.require_human_review:

        message += (
            "\nMark recommendations for human review."
        )

    request_id = generate_request_id()

    conversation_id = str(
        uuid.uuid4()
    )

    start = time.perf_counter()

    set_request_context(
        request_id=request_id,
        user_id=current_user.user_id,
        company_id=company_id,
    )

    try:

        result = await execute_copilot(
            db=db,
            message=message,
            company_id=company_id,
            conversation_id=conversation_id,
            mode="recommendation",
            current_user=current_user,
            request_id=request_id,
            metadata={
                "area": payload.area,
            },
        )

        execution_time_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return CopilotResponse(
            request_id=request_id,
            conversation_id=conversation_id,
            company_id=company_id,
            answer=extract_answer(result),
            mode="recommendation",
            confidence=result.get(
                "confidence"
            ),
            grounded=bool(
                result.get(
                    "grounded",
                    False,
                )
            ),
            requires_human_review=(
                payload.require_human_review
                or bool(
                    result.get(
                        "requires_human_review",
                        False,
                    )
                )
            ),
            metrics=extract_metrics(
                result
            ),
            citations=extract_citations(
                result
            ),
            sources=result.get(
                "sources",
                [],
            ),
            recommendations=result.get(
                "recommendations",
                [],
            ),
            warnings=extract_warnings(
                result
            ),
            metadata=result.get(
                "metadata",
                {},
            ),
            execution_time_ms=round(
                execution_time_ms,
                2,
            ),
        )

    finally:

        clear_request_context()


# ============================================================
# GET /copilot/health
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
)
async def copilot_health() -> HealthResponse:
    """
    Lightweight copilot API health endpoint.
    """

    return HealthResponse(
        status="healthy",
        service="finco-copilot",
        timestamp=utc_now(),
    )