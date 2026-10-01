
"""
FinCo AI — Agent State

Shared state passed between the Supervisor Agent
and specialized financial agents.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypedDict


class AgentStatus(str, Enum):
    """Current status of the agent workflow."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_REVIEW = "waiting_for_review"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowType(str, Enum):
    """Types of FinCo AI workflows."""

    FINANCIAL_ANALYSIS = "financial_analysis"
    FRAUD_INVESTIGATION = "fraud_investigation"
    FORECAST_ANALYSIS = "forecast_analysis"
    RISK_INVESTIGATION = "risk_investigation"
    EXECUTIVE_ANALYSIS = "executive_analysis"
    FINANCIAL_CRISIS = "financial_crisis"


class FinCoState(TypedDict, total=False):
    """
    Shared state for the FinCo AI agent workflow.

    This state can be passed between:
        Supervisor Agent
        Financial Agent
        Fraud Agent
        Forecast Agent
        RAG Agent
        What-If Agent
        Recommendation Agent
    """

    # --------------------------------------------------
    # REQUEST CONTEXT
    # --------------------------------------------------

    request_id: str
    user_id: str
    company_id: str

    query: str
    normalized_query: str

    created_at: str
    updated_at: str

    # --------------------------------------------------
    # WORKFLOW CONTROL
    # --------------------------------------------------

    workflow_type: WorkflowType
    status: AgentStatus

    selected_agent: str
    current_agent: str

    completed_agents: list[str]
    failed_agents: list[str]

    execution_steps: list[str]

    error: str | None

    # --------------------------------------------------
    # USER / COMPANY CONTEXT
    # --------------------------------------------------

    company_context: dict[str, Any]
    user_context: dict[str, Any]

    permissions: list[str]

    # --------------------------------------------------
    # FINANCIAL DATA
    # --------------------------------------------------

    financial_data: dict[str, Any]

    revenue_data: dict[str, Any]
    expense_data: dict[str, Any]

    pnl_data: dict[str, Any]
    balance_sheet_data: dict[str, Any]
    cash_flow_data: dict[str, Any]

    financial_metrics: dict[str, Any]
    financial_ratios: dict[str, Any]
    kpis: dict[str, Any]

    financial_health_score: float | None

    # --------------------------------------------------
    # RAG / DOCUMENT INTELLIGENCE
    # --------------------------------------------------

    retrieved_documents: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]

    rag_context: str
    citations: list[dict[str, Any]]

    citation_valid: bool

    # --------------------------------------------------
    # FRAUD DETECTION
    # --------------------------------------------------

    fraud_results: list[dict[str, Any]]

    suspicious_transactions: list[dict[str, Any]]

    fraud_score: float | None

    fraud_explanation: str

    # --------------------------------------------------
    # FORECASTING
    # --------------------------------------------------

    forecast_results: dict[str, Any]

    revenue_forecast: dict[str, Any]
    profit_forecast: dict[str, Any]
    cashflow_forecast: dict[str, Any]

    forecast_confidence: float | None

    # --------------------------------------------------
    # ALERTS / RISK
    # --------------------------------------------------

    alerts: list[dict[str, Any]]

    active_alert: dict[str, Any]

    risk_score: float | None
    business_continuity_risk: float | None

    risk_explanation: str

    # --------------------------------------------------
    # WHAT-IF ANALYSIS
    # --------------------------------------------------

    scenario: dict[str, Any]

    scenario_results: dict[str, Any]

    sensitivity_results: dict[str, Any]

    # --------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------

    recommendations: list[dict[str, Any]]

    recommended_actions: list[str]

    recommendation_explanation: str

    # --------------------------------------------------
    # HUMAN REVIEW
    # --------------------------------------------------

    requires_human_review: bool

    human_review_status: str

    human_feedback: str

    approved_actions: list[str]

    # --------------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------------

    final_answer: str

    answer_sources: list[dict[str, Any]]

    confidence_score: float | None

    # --------------------------------------------------
    # AUDIT
    # --------------------------------------------------

    audit_events: list[dict[str, Any]]


def create_initial_state(
    query: str,
    user_id: str,
    company_id: str,
    request_id: str,
) -> FinCoState:
    """
    Create the initial state for a new agent workflow.
    """

    now = datetime.now(timezone.utc).isoformat()

    return FinCoState(
        request_id=request_id,
        user_id=user_id,
        company_id=company_id,

        query=query,
        normalized_query=query.lower().strip(),

        created_at=now,
        updated_at=now,

        workflow_type=WorkflowType.FINANCIAL_ANALYSIS,
        status=AgentStatus.IDLE,

        selected_agent="",
        current_agent="",

        completed_agents=[],
        failed_agents=[],
        execution_steps=[],

        error=None,

        company_context={},
        user_context={},
        permissions=[],

        financial_data={},

        revenue_data={},
        expense_data={},

        pnl_data={},
        balance_sheet_data={},
        cash_flow_data={},

        financial_metrics={},
        financial_ratios={},
        kpis={},

        financial_health_score=None,

        retrieved_documents=[],
        retrieved_chunks=[],

        rag_context="",
        citations=[],
        citation_valid=False,

        fraud_results=[],
        suspicious_transactions=[],

        fraud_score=None,
        fraud_explanation="",

        forecast_results={},
        revenue_forecast={},
        profit_forecast={},
        cashflow_forecast={},

        forecast_confidence=None,

        alerts=[],
        active_alert={},

        risk_score=None,
        business_continuity_risk=None,
        risk_explanation="",

        scenario={},
        scenario_results={},
        sensitivity_results={},

        recommendations=[],
        recommended_actions=[],
        recommendation_explanation="",

        requires_human_review=False,
        human_review_status="pending",
        human_feedback="",
        approved_actions=[],

        final_answer="",
        answer_sources=[],
        confidence_score=None,

        audit_events=[],
    )


def update_state(
    state: FinCoState,
    **updates: Any,
) -> FinCoState:
    """
    Update workflow state.

    Returns a new dictionary without mutating the original.
    """

    new_state = dict(state)

    new_state.update(updates)

    new_state["updated_at"] = (
        datetime.now(timezone.utc).isoformat()
    )

    return new_state


def add_execution_step(
    state: FinCoState,
    step: str,
) -> FinCoState:
    """Add an execution step to the workflow."""

    steps = list(state.get("execution_steps", []))
    steps.append(step)

    return update_state(
        state,
        execution_steps=steps,
    )


def mark_agent_completed(
    state: FinCoState,
    agent_name: str,
) -> FinCoState:
    """Mark an agent as completed."""

    completed = list(
        state.get("completed_agents", [])
    )

    if agent_name not in completed:
        completed.append(agent_name)

    return update_state(
        state,
        completed_agents=completed,
        current_agent="",
    )


def mark_agent_failed(
    state: FinCoState,
    agent_name: str,
    error: str,
) -> FinCoState:
    """Mark an agent as failed."""

    failed = list(
        state.get("failed_agents", [])
    )

    if agent_name not in failed:
        failed.append(agent_name)

    return update_state(
        state,
        failed_agents=failed,
        error=error,
        status=AgentStatus.FAILED,
        current_agent="",
    )


def add_audit_event(
    state: FinCoState,
    event_type: str,
    description: str,
    agent_name: str = "",
) -> FinCoState:
    """Add an audit event to the workflow."""

    events = list(
        state.get("audit_events", [])
    )

    events.append(
        {
            "event_type": event_type,
            "description": description,
            "agent_name": agent_name,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )

    return update_state(
        state,
        audit_events=events,
    )