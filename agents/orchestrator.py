"""
FinCo AI - Agent Orchestrator
=============================

Central orchestration engine for the FinCo AI agentic system.

Responsibilities
----------------
- Accept agent requests.
- Validate orchestration input.
- Authorize the request.
- Route the request.
- Build an execution plan.
- Execute specialized agents.
- Execute multi-step workflows.
- Apply guardrails.
- Manage agent memory.
- Handle agent/tool failures.
- Aggregate results.
- Produce a final structured response.
- Record audit events.
- Track execution metadata.

Architecture
------------

                    User Request
                         |
                         v
                 +---------------+
                 | Orchestrator  |
                 +-------+-------+
                         |
             +-----------+-----------+
             |                       |
             v                       v
          Security               Guardrails
             |                       |
             +-----------+-----------+
                         |
                         v
                       Router
                         |
                         v
                      Planner
                         |
              +----------+----------+
              |          |          |
              v          v          v
          Financial    Fraud     Forecast
            Agent      Agent       Agent
              |          |          |
              +----------+----------+
                         |
                         v
                   Tool Executor
                         |
                         v
                    RAG / Tools
                         |
                         v
                  Result Aggregator
                         |
              +----------+----------+
              |                     |
              v                     v
           Memory                 Audit
              |                     |
              +----------+----------+
                         |
                         v
                    Final Result


Important
---------
The orchestrator coordinates execution.

It must NOT:
- bypass authorization,
- directly modify financial records,
- trust agent output blindly,
- allow an agent to override system guardrails,
- treat memory as authority,
- expose internal errors to users,
- execute arbitrary tools.
"""

from __future__ import annotations

import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# ============================================================================
# Exceptions
# ============================================================================


class OrchestratorError(Exception):
    """Base exception for orchestration failures."""


class InvalidOrchestratorRequestError(
    OrchestratorError
):
    """Raised when an orchestration request is invalid."""


class OrchestratorConfigurationError(
    OrchestratorError
):
    """Raised when orchestrator configuration is invalid."""


class OrchestratorAuthorizationError(
    OrchestratorError
):
    """Raised when authorization fails."""


class OrchestratorExecutionError(
    OrchestratorError
):
    """Raised when orchestration execution fails."""


class OrchestratorTimeoutError(
    OrchestratorExecutionError
):
    """Raised when orchestration exceeds execution limits."""


class AgentNotRegisteredError(
    OrchestratorError
):
    """Raised when an agent is not registered."""


class AgentExecutionError(
    OrchestratorExecutionError
):
    """Raised when a specialized agent fails."""


class WorkflowExecutionError(
    OrchestratorExecutionError
):
    """Raised when a workflow fails."""


class GuardrailViolationError(
    OrchestratorError
):
    """Raised when guardrails reject execution."""


# ============================================================================
# Constants
# ============================================================================


DEFAULT_MAX_STEPS = 20

DEFAULT_MAX_AGENTS = 10

DEFAULT_TIMEOUT_SECONDS = 120

DEFAULT_MAX_CONTEXT_LENGTH = 30_000

DEFAULT_MAX_QUERY_LENGTH = 10_000

DEFAULT_MAX_RESULTS = 100


# ============================================================================
# Enums
# ============================================================================


class OrchestrationMode(str, Enum):
    """Execution modes."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    WORKFLOW = "workflow"
    AUTO = "auto"


class ExecutionStatus(str, Enum):
    """Overall execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"


class StepStatus(str, Enum):
    """Individual execution step status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ResultType(str, Enum):
    """Result classification."""

    ANSWER = "answer"
    ANALYSIS = "analysis"
    FORECAST = "forecast"
    FRAUD = "fraud"
    RAG = "rag"
    WHAT_IF = "what_if"
    RECOMMENDATION = "recommendation"
    ERROR = "error"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class OrchestratorConfig:
    """Configuration for the agent orchestrator."""

    enabled: bool = True

    max_steps: int = DEFAULT_MAX_STEPS

    max_agents: int = DEFAULT_MAX_AGENTS

    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    )

    max_context_length: int = (
        DEFAULT_MAX_CONTEXT_LENGTH
    )

    max_query_length: int = (
        DEFAULT_MAX_QUERY_LENGTH
    )

    max_results: int = DEFAULT_MAX_RESULTS

    continue_on_agent_error: bool = True

    continue_on_workflow_step_error: bool = False

    require_authorization: bool = True

    require_guardrails: bool = True

    enable_memory: bool = True

    enable_audit: bool = True

    allow_parallel_agents: bool = False

    fail_closed_on_authorization_error: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate configuration."""

        positive_values = {
            "max_steps": self.max_steps,
            "max_agents": self.max_agents,
            "timeout_seconds": (
                self.timeout_seconds
            ),
            "max_context_length": (
                self.max_context_length
            ),
            "max_query_length": (
                self.max_query_length
            ),
            "max_results": self.max_results,
        }

        for name, value in positive_values.items():

            if value < 1:

                raise OrchestratorConfigurationError(
                    f"{name} must be greater than zero."
                )


# ============================================================================
# Request Models
# ============================================================================


@dataclass
class OrchestratorRequest:
    """Incoming request to the orchestration layer."""

    query: str

    company_id: Optional[str] = None

    user_id: Optional[str] = None

    session_id: Optional[str] = None

    workflow_id: Optional[str] = None

    mode: OrchestrationMode = (
        OrchestrationMode.AUTO
    )

    requested_agent: Optional[str] = None

    requested_workflow: Optional[str] = None

    context: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    conversation_history: List[
        Mapping[str, Any]
    ] = field(
        default_factory=list
    )


@dataclass
class ExecutionStep:
    """Represents one orchestration step."""

    step_id: str

    name: str

    step_type: str

    status: StepStatus = (
        StepStatus.PENDING
    )

    agent_name: Optional[str] = None

    input_data: Dict[str, Any] = field(
        default_factory=dict
    )

    output_data: Any = None

    error: Optional[str] = None

    started_at: Optional[datetime] = None

    completed_at: Optional[datetime] = None

    duration_ms: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class AgentExecutionResult:
    """Normalized result from a specialized agent."""

    success: bool

    agent_name: str

    result_type: ResultType = (
        ResultType.ANSWER
    )

    answer: Optional[str] = None

    data: Any = None

    confidence: Optional[float] = None

    citations: List[Any] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    step_id: Optional[str] = None


@dataclass
class OrchestratorResult:
    """Final orchestrator response."""

    request_id: str

    success: bool

    status: ExecutionStatus

    answer: Optional[str] = None

    data: Any = None

    agents_used: List[str] = field(
        default_factory=list
    )

    steps: List[ExecutionStep] = field(
        default_factory=list
    )

    agent_results: List[
        AgentExecutionResult
    ] = field(
        default_factory=list
    )

    citations: List[Any] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    errors: List[str] = field(
        default_factory=list
    )

    confidence: Optional[float] = None

    started_at: Optional[datetime] = None

    completed_at: Optional[datetime] = None

    duration_ms: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "request_id": self.request_id,
            "success": self.success,
            "status": self.status.value,
            "answer": self.answer,
            "data": _serialize(self.data),
            "agents_used": self.agents_used,
            "steps": [
                _serialize(step)
                for step in self.steps
            ],
            "agent_results": [
                _serialize(result)
                for result in self.agent_results
            ],
            "citations": _serialize(
                self.citations
            ),
            "warnings": self.warnings,
            "errors": self.errors,
            "confidence": self.confidence,
            "started_at": (
                self.started_at.isoformat()
                if self.started_at
                else None
            ),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at
                else None
            ),
            "duration_ms": self.duration_ms,
            "metadata": _serialize(
                self.metadata
            ),
        }


# ============================================================================
# Orchestrator
# ============================================================================


class AgentOrchestrator:
    """
    Central FinCo AI agent orchestration engine.

    Dependencies are injected to keep the orchestration
    layer testable and independent of implementation details.
    """

    def __init__(
        self,
        router: Any = None,
        planner: Any = None,
        memory: Any = None,
        context_builder: Any = None,
        tool_registry: Any = None,
        tool_executor: Any = None,
        guardrails: Any = None,
        authorization_service: Any = None,
        audit_service: Any = None,
        agents: Optional[
            Mapping[str, Any]
        ] = None,
        workflows: Optional[
            Mapping[str, Any]
        ] = None,
        config: Optional[
            OrchestratorConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or OrchestratorConfig()
        )

        self.config.validate()

        self.router = router
        self.planner = planner
        self.memory = memory
        self.context_builder = (
            context_builder
        )
        self.tool_registry = (
            tool_registry
        )
        self.tool_executor = (
            tool_executor
        )
        self.guardrails = guardrails
        self.authorization_service = (
            authorization_service
        )
        self.audit_service = (
            audit_service
        )

        self.agents: Dict[
            str,
            Any,
        ] = dict(
            agents or {}
        )

        self.workflows: Dict[
            str,
            Any,
        ] = dict(
            workflows or {}
        )

    # ========================================================================
    # Registration
    # ========================================================================

    def register_agent(
        self,
        name: str,
        agent: Any,
    ) -> None:
        """Register a specialized agent."""

        normalized_name = (
            self._normalize_name(
                name
            )
        )

        if not normalized_name:

            raise InvalidOrchestratorRequestError(
                "Agent name cannot be empty."
            )

        if agent is None:

            raise InvalidOrchestratorRequestError(
                f"Agent '{name}' cannot be None."
            )

        self.agents[
            normalized_name
        ] = agent

    def register_workflow(
        self,
        name: str,
        workflow: Any,
    ) -> None:
        """Register an executable workflow."""

        normalized_name = (
            self._normalize_name(
                name
            )
        )

        if not normalized_name:

            raise InvalidOrchestratorRequestError(
                "Workflow name cannot be empty."
            )

        if workflow is None:

            raise InvalidOrchestratorRequestError(
                f"Workflow '{name}' cannot be None."
            )

        self.workflows[
            normalized_name
        ] = workflow

    def unregister_agent(
        self,
        name: str,
    ) -> bool:

        normalized_name = (
            self._normalize_name(
                name
            )
        )

        return (
            self.agents.pop(
                normalized_name,
                None,
            )
            is not None
        )

    def unregister_workflow(
        self,
        name: str,
    ) -> bool:

        normalized_name = (
            self._normalize_name(
                name
            )
        )

        return (
            self.workflows.pop(
                normalized_name,
                None,
            )
            is not None
        )

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def run(
        self,
        request: OrchestratorRequest
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> OrchestratorResult:
        """
        Execute an orchestration request.

        Main pipeline:

            validate
              ↓
            authorize
              ↓
            guardrails
              ↓
            memory/context
              ↓
            route
              ↓
            plan
              ↓
            execute
              ↓
            aggregate
              ↓
            memory
              ↓
            audit
              ↓
            response
        """

        started_at = datetime.now(
            timezone.utc
        )

        start_time = time.perf_counter()

        request_id = (
            self._generate_request_id()
        )

        try:

            request = (
                self._normalize_request(
                    request,
                    **kwargs,
                )
            )

            self._validate_request(
                request
            )

            if not self.config.enabled:

                raise OrchestratorConfigurationError(
                    "Agent orchestrator is disabled."
                )

            # --------------------------------------------------------------
            # Authorization
            # --------------------------------------------------------------

            self._authorize(
                request
            )

            # --------------------------------------------------------------
            # Guardrails - input
            # --------------------------------------------------------------

            self._check_guardrails(
                request,
                stage="input",
            )

            # --------------------------------------------------------------
            # Context
            # --------------------------------------------------------------

            context = self._build_context(
                request
            )

            request.context.update(
                context
            )

            # --------------------------------------------------------------
            # Routing
            # --------------------------------------------------------------

            route = self._route(
                request
            )

            # --------------------------------------------------------------
            # Planning
            # --------------------------------------------------------------

            plan = self._plan(
                request,
                route,
            )

            # --------------------------------------------------------------
            # Execution
            # --------------------------------------------------------------

            result = self._execute(
                request=request,
                request_id=request_id,
                route=route,
                plan=plan,
                started_at=started_at,
                start_time=start_time,
            )

            # --------------------------------------------------------------
            # Guardrails - output
            # --------------------------------------------------------------

            self._check_result_guardrails(
                result
            )

            # --------------------------------------------------------------
            # Memory
            # --------------------------------------------------------------

            self._store_memory(
                request,
                result,
            )

            # --------------------------------------------------------------
            # Audit
            # --------------------------------------------------------------

            self._audit(
                "orchestration_completed",
                request,
                result,
            )

            return result

        except GuardrailViolationError as exc:

            result = OrchestratorResult(
                request_id=request_id,
                success=False,
                status=ExecutionStatus.BLOCKED,
                errors=[
                    str(exc)
                ],
                started_at=started_at,
                completed_at=datetime.now(
                    timezone.utc
                ),
                duration_ms=(
                    time.perf_counter()
                    - start_time
                )
                * 1000,
            )

            self._audit(
                "orchestration_blocked",
                request
                if isinstance(
                    request,
                    OrchestratorRequest,
                )
                else None,
                result,
            )

            return result

        except OrchestratorAuthorizationError as exc:

            result = OrchestratorResult(
                request_id=request_id,
                success=False,
                status=ExecutionStatus.BLOCKED,
                errors=[
                    "Authorization failed."
                ],
                warnings=[
                    str(exc)
                ],
                started_at=started_at,
                completed_at=datetime.now(
                    timezone.utc
                ),
                duration_ms=(
                    time.perf_counter()
                    - start_time
                )
                * 1000,
            )

            self._audit(
                "orchestration_authorization_failed",
                request
                if isinstance(
                    request,
                    OrchestratorRequest,
                )
                else None,
                result,
            )

            return result

        except OrchestratorError as exc:

            result = OrchestratorResult(
                request_id=request_id,
                success=False,
                status=ExecutionStatus.FAILED,
                errors=[
                    str(exc)
                ],
                started_at=started_at,
                completed_at=datetime.now(
                    timezone.utc
                ),
                duration_ms=(
                    time.perf_counter()
                    - start_time
                )
                * 1000,
            )

            self._audit(
                "orchestration_failed",
                request
                if isinstance(
                    request,
                    OrchestratorRequest,
                )
                else None,
                result,
            )

            return result

        except Exception as exc:

            # Never expose internal implementation
            # details as the user-facing error.

            result = OrchestratorResult(
                request_id=request_id,
                success=False,
                status=ExecutionStatus.FAILED,
                errors=[
                    "An internal orchestration error occurred."
                ],
                warnings=[
                    str(exc)
                ],
                started_at=started_at,
                completed_at=datetime.now(
                    timezone.utc
                ),
                duration_ms=(
                    time.perf_counter()
                    - start_time
                )
                * 1000,
            )

            self._audit(
                "orchestration_internal_error",
                request
                if isinstance(
                    request,
                    OrchestratorRequest,
                )
                else None,
                result,
            )

            return result

    # ========================================================================
    # Validation
    # ========================================================================

    def _normalize_request(
        self,
        request: OrchestratorRequest
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> OrchestratorRequest:

        if isinstance(
            request,
            OrchestratorRequest,
        ):

            return request

        if not isinstance(
            request,
            Mapping,
        ):

            raise InvalidOrchestratorRequestError(
                "Request must be OrchestratorRequest "
                "or mapping."
            )

        payload = dict(
            request
        )

        payload.update(
            kwargs
        )

        mode = payload.get(
            "mode",
            OrchestrationMode.AUTO,
        )

        if not isinstance(
            mode,
            OrchestrationMode,
        ):

            mode = OrchestrationMode(
                mode
            )

        payload["mode"] = mode

        return OrchestratorRequest(
            **payload
        )

    def _validate_request(
        self,
        request: OrchestratorRequest,
    ) -> None:

        if not request.query:

            raise InvalidOrchestratorRequestError(
                "Query cannot be empty."
            )

        request.query = (
            request.query.strip()
        )

        if len(request.query) > (
            self.config.max_query_length
        ):

            raise InvalidOrchestratorRequestError(
                "Query exceeds maximum length."
            )

        if (
            not request.company_id
            and request.requested_agent
            in {
                "financial_agent",
                "fraud_agent",
                "forecast_agent",
                "recommendation_agent",
            }
        ):

            raise InvalidOrchestratorRequestError(
                "company_id is required for "
                "company-scoped financial agents."
            )

    # ========================================================================
    # Authorization
    # ========================================================================

    def _authorize(
        self,
        request: OrchestratorRequest,
    ) -> None:

        if not self.config.require_authorization:
            return

        if self.authorization_service is None:

            if (
                self.config.fail_closed_on_authorization_error
            ):

                raise OrchestratorAuthorizationError(
                    "Authorization service is unavailable."
                )

            return

        service = (
            self.authorization_service
        )

        method_names = (
            "authorize",
            "check_permission",
            "can_access",
            "has_permission",
        )

        payload = {
            "user_id": request.user_id,
            "company_id": request.company_id,
            "query": request.query,
            "requested_agent": (
                request.requested_agent
            ),
            "requested_workflow": (
                request.requested_workflow
            ),
        }

        for method_name in method_names:

            method = getattr(
                service,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                result = method(
                    **payload
                )

            except TypeError:

                try:
                    result = method(
                        request.user_id,
                        request.company_id,
                    )

                except TypeError as exc:

                    raise OrchestratorAuthorizationError(
                        "Authorization service signature "
                        "is incompatible."
                    ) from exc

            if result is False:

                raise OrchestratorAuthorizationError(
                    "User is not authorized "
                    "for this operation."
                )

            return

        if (
            self.config.fail_closed_on_authorization_error
        ):

            raise OrchestratorAuthorizationError(
                "Authorization service has no "
                "supported authorization method."
            )

    # ========================================================================
    # Guardrails
    # ========================================================================

    def _check_guardrails(
        self,
        request: OrchestratorRequest,
        stage: str,
    ) -> None:

        if not self.config.require_guardrails:
            return

        if self.guardrails is None:
            return

        payload = {
            "query": request.query,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "context": request.context,
            "stage": stage,
        }

        for method_name in (
            "check",
            "validate",
            "evaluate",
            "run",
        ):

            method = getattr(
                self.guardrails,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                result = method(
                    **payload
                )

            except TypeError:

                try:
                    result = method(
                        request.query
                    )

                except TypeError as exc:

                    raise GuardrailViolationError(
                        "Guardrail interface is incompatible."
                    ) from exc

            if result is False:

                raise GuardrailViolationError(
                    "Request blocked by guardrails."
                )

            if isinstance(
                result,
                Mapping,
            ):

                allowed = result.get(
                    "allowed",
                    result.get(
                        "safe",
                        True,
                    ),
                )

                if not allowed:

                    reason = result.get(
                        "reason",
                        "Request blocked by guardrails.",
                    )

                    raise GuardrailViolationError(
                        str(reason)
                    )

            return

    def _check_result_guardrails(
        self,
        result: OrchestratorResult,
    ) -> None:

        if not self.config.require_guardrails:
            return

        if self.guardrails is None:
            return

        payload = {
            "answer": result.answer,
            "data": result.data,
            "agent_results": [
                self._result_to_mapping(
                    item
                )
                for item in result.agent_results
            ],
            "stage": "output",
        }

        for method_name in (
            "check_output",
            "validate_output",
            "check",
            "validate",
        ):

            method = getattr(
                self.guardrails,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                validation = method(
                    **payload
                )

            except TypeError:
                continue

            if validation is False:

                raise GuardrailViolationError(
                    "Agent output blocked by guardrails."
                )

            if isinstance(
                validation,
                Mapping,
            ):

                if not validation.get(
                    "allowed",
                    True,
                ):

                    raise GuardrailViolationError(
                        str(
                            validation.get(
                                "reason",
                                "Agent output blocked "
                                "by guardrails.",
                            )
                        )
                    )

            return

    # ========================================================================
    # Routing
    # ========================================================================

    def _route(
        self,
        request: OrchestratorRequest,
    ) -> Dict[str, Any]:
        """Resolve the appropriate agent/workflow."""

        if request.requested_workflow:

            workflow_name = (
                self._normalize_name(
                    request.requested_workflow
                )
            )

            if (
                workflow_name
                not in self.workflows
            ):

                raise AgentNotRegisteredError(
                    f"Workflow not registered: "
                    f"{workflow_name}"
                )

            return {
                "mode": OrchestrationMode.WORKFLOW.value,
                "workflow": workflow_name,
                "agents": [],
            }

        if request.requested_agent:

            agent_name = (
                self._normalize_name(
                    request.requested_agent
                )
            )

            if (
                agent_name
                not in self.agents
            ):

                raise AgentNotRegisteredError(
                    f"Agent not registered: "
                    f"{agent_name}"
                )

            return {
                "mode": OrchestrationMode.SINGLE_AGENT.value,
                "agent": agent_name,
                "agents": [
                    agent_name
                ],
            }

        if self.router is None:

            raise OrchestratorExecutionError(
                "Router is required for automatic routing."
            )

        route = self._invoke_component(
            self.router,
            request,
            operation_names=(
                "route",
                "classify",
                "run",
                "invoke",
            ),
        )

        normalized = (
            self._normalize_route(
                route
            )
        )

        return normalized

    def _normalize_route(
        self,
        route: Any,
    ) -> Dict[str, Any]:

        if isinstance(
            route,
            str,
        ):

            agent_name = (
                self._normalize_name(
                    route
                )
            )

            return {
                "mode": OrchestrationMode.SINGLE_AGENT.value,
                "agent": agent_name,
                "agents": [
                    agent_name
                ],
            }

        if isinstance(
            route,
            Mapping,
        ):

            result = dict(
                route
            )

            if "agent" in result:

                result["agent"] = (
                    self._normalize_name(
                        str(
                            result["agent"]
                        )
                    )
                )

            agents = result.get(
                "agents",
                [],
            )

            if isinstance(
                agents,
                str,
            ):

                agents = [
                    agents
                ]

            normalized_agents = [
                self._normalize_name(
                    str(agent)
                )
                for agent in agents
            ]

            if result.get(
                "agent"
            ) and result[
                "agent"
            ] not in normalized_agents:

                normalized_agents.insert(
                    0,
                    result["agent"],
                )

            result["agents"] = (
                normalized_agents
            )

            result.setdefault(
                "mode",
                OrchestrationMode.SINGLE_AGENT.value,
            )

            self._validate_route(
                result
            )

            return result

        raise OrchestratorExecutionError(
            "Router returned an unsupported route."
        )

    def _validate_route(
        self,
        route: Mapping[str, Any],
    ) -> None:

        agents = route.get(
            "agents",
            [],
        )

        if len(agents) > (
            self.config.max_agents
        ):

            raise OrchestratorExecutionError(
                "Route exceeds maximum agent count."
            )

        for agent_name in agents:

            if agent_name not in self.agents:

                raise AgentNotRegisteredError(
                    f"Routed agent is not registered: "
                    f"{agent_name}"
                )

    # ========================================================================
    # Planning
    # ========================================================================

    def _plan(
        self,
        request: OrchestratorRequest,
        route: Mapping[str, Any],
    ) -> Any:

        if (
            request.mode
            == OrchestrationMode.SINGLE_AGENT
            or route.get(
                "mode"
            )
            == OrchestrationMode.SINGLE_AGENT.value
        ):

            return [
                {
                    "name": (
                        route.get(
                            "agent"
                        )
                        or route[
                            "agents"
                        ][0]
                    ),
                    "type": "agent",
                }
            ]

        if (
            route.get(
                "mode"
            )
            == OrchestrationMode.WORKFLOW.value
        ):

            return [
                {
                    "name": route[
                        "workflow"
                    ],
                    "type": "workflow",
                }
            ]

        if self.planner is None:

            return [
                {
                    "name": agent,
                    "type": "agent",
                }
                for agent in route.get(
                    "agents",
                    [],
                )
            ]

        plan = self._invoke_component(
            self.planner,
            request,
            extra={
                "route": route
            },
            operation_names=(
                "plan",
                "create_plan",
                "run",
                "invoke",
            ),
        )

        return self._normalize_plan(
            plan,
            route,
        )

    def _normalize_plan(
        self,
        plan: Any,
        route: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:

        if isinstance(
            plan,
            Mapping,
        ):

            if "steps" in plan:

                plan = plan[
                    "steps"
                ]

            else:

                plan = [
                    plan
                ]

        if isinstance(
            plan,
            str,
        ):

            plan = [
                {
                    "name": plan,
                    "type": "agent",
                }
            ]

        if not isinstance(
            plan,
            Sequence,
        ):

            raise OrchestratorExecutionError(
                "Planner returned an invalid plan."
            )

        normalized: List[
            Dict[str, Any]
        ] = []

        for step in plan:

            if isinstance(
                step,
                str,
            ):

                normalized.append(
                    {
                        "name": step,
                        "type": "agent",
                    }
                )

            elif isinstance(
                step,
                Mapping,
            ):

                normalized.append(
                    dict(
                        step
                    )
                )

            else:

                raise OrchestratorExecutionError(
                    "Planner returned an unsupported "
                    "step."
                )

        if len(normalized) > (
            self.config.max_steps
        ):

            raise OrchestratorExecutionError(
                "Execution plan exceeds maximum steps."
            )

        return normalized

    # ========================================================================
    # Context
    # ========================================================================

    def _build_context(
        self,
        request: OrchestratorRequest,
    ) -> Dict[str, Any]:

        context: Dict[
            str,
            Any,
        ] = dict(
            request.context
        )

        if (
            request.conversation_history
        ):

            context[
                "conversation_history"
            ] = list(
                request.conversation_history
            )

        if (
            self.config.enable_memory
            and self.memory is not None
        ):

            memory_context = self._get_memory_context(
                request
            )

            if memory_context:

                context[
                    "memory"
                ] = memory_context

        if self.context_builder is not None:

            built = self._invoke_component(
                self.context_builder,
                request,
                extra={
                    "context": context
                },
                operation_names=(
                    "build",
                    "build_context",
                    "create",
                    "run",
                    "invoke",
                ),
            )

            if isinstance(
                built,
                Mapping,
            ):

                context.update(
                    built
                )

            elif built is not None:

                context[
                    "built_context"
                ] = built

        serialized = str(
            _serialize(context)
        )

        if len(serialized) > (
            self.config.max_context_length
        ):

            context[
                "context_truncated"
            ] = True

        return context

    def _get_memory_context(
        self,
        request: OrchestratorRequest,
    ) -> Any:

        memory = self.memory

        if hasattr(
            memory,
            "build_context",
        ):

            return memory.build_context(
                query=request.query,
                company_id=request.company_id,
                user_id=request.user_id,
                session_id=request.session_id,
                workflow_id=request.workflow_id,
            )

        if hasattr(
            memory,
            "retrieve",
        ):

            return memory.retrieve(
                text=request.query,
                company_id=request.company_id,
                user_id=request.user_id,
                session_id=request.session_id,
                workflow_id=request.workflow_id,
                limit=10,
            )

        return None

    # ========================================================================
    # Execution
    # ========================================================================

    def _execute(
        self,
        request: OrchestratorRequest,
        request_id: str,
        route: Mapping[str, Any],
        plan: Sequence[
            Mapping[str, Any]
        ],
        started_at: datetime,
        start_time: float,
    ) -> OrchestratorResult:

        steps: List[
            ExecutionStep
        ] = []

        agent_results: List[
            AgentExecutionResult
        ] = []

        agents_used: List[str] = []

        warnings: List[str] = []

        errors: List[str] = []

        for index, plan_step in enumerate(
            plan
        ):

            if index >= (
                self.config.max_steps
            ):

                raise OrchestratorExecutionError(
                    "Maximum orchestration steps exceeded."
                )

            step_id = (
                self._generate_step_id(
                    index
                )
            )

            name = str(
                plan_step.get(
                    "name",
                    f"step_{index + 1}",
                )
            )

            step_type = str(
                plan_step.get(
                    "type",
                    "agent",
                )
            )

            step = ExecutionStep(
                step_id=step_id,
                name=name,
                step_type=step_type,
            )

            steps.append(
                step
            )

            if self._is_timed_out(
                start_time
            ):

                step.status = (
                    StepStatus.SKIPPED
                )

                errors.append(
                    "Orchestration timeout exceeded."
                )

                break

            try:

                step.status = (
                    StepStatus.RUNNING
                )

                step.started_at = (
                    datetime.now(
                        timezone.utc
                    )
                )

                if step_type == "workflow":

                    execution = (
                        self._execute_workflow(
                            name=name,
                            request=request,
                            step_id=step_id,
                        )
                    )

                    normalized_results = (
                        self._normalize_agent_results(
                            execution,
                            name,
                            step_id,
                        )
                    )

                else:

                    execution = (
                        self._execute_agent(
                            name=name,
                            request=request,
                            step_id=step_id,
                            previous_results=(
                                agent_results
                            ),
                        )
                    )

                    normalized_results = (
                        self._normalize_agent_results(
                            execution,
                            name,
                            step_id,
                        )
                    )

                for result in (
                    normalized_results
                ):

                    agent_results.append(
                        result
                    )

                    if (
                        result.agent_name
                        not in agents_used
                    ):

                        agents_used.append(
                            result.agent_name
                        )

                    warnings.extend(
                        result.warnings
                    )

                    errors.extend(
                        result.errors
                    )

                step.output_data = (
                    _serialize(
                        execution
                    )
                )

                step.status = (
                    StepStatus.COMPLETED
                )

            except GuardrailViolationError as exc:

                step.status = (
                    StepStatus.BLOCKED
                )

                step.error = str(
                    exc
                )

                errors.append(
                    str(exc)
                )

                if not self.config.continue_on_agent_error:

                    raise

            except Exception as exc:

                step.status = (
                    StepStatus.FAILED
                )

                step.error = (
                    "Agent execution failed."
                )

                errors.append(
                    f"{name}: "
                    f"Agent execution failed."
                )

                if not self.config.continue_on_agent_error:

                    raise AgentExecutionError(
                        f"Agent '{name}' failed."
                    ) from exc

            finally:

                step.completed_at = (
                    datetime.now(
                        timezone.utc
                    )
                )

                if step.started_at:

                    step.duration_ms = (
                        step.completed_at
                        - step.started_at
                    ).total_seconds() * 1000

        status = (
            self._determine_status(
                agent_results,
                steps,
                errors,
            )
        )

        answer = (
            self._aggregate_answer(
                request,
                agent_results,
            )
        )

        data = (
            self._aggregate_data(
                agent_results
            )
        )

        citations = (
            self._aggregate_citations(
                agent_results
            )
        )

        confidence = (
            self._aggregate_confidence(
                agent_results
            )
        )

        completed_at = datetime.now(
            timezone.utc
        )

        return OrchestratorResult(
            request_id=request_id,
            success=(
                status
                in {
                    ExecutionStatus.COMPLETED,
                    ExecutionStatus.PARTIAL,
                }
            ),
            status=status,
            answer=answer,
            data=data,
            agents_used=agents_used,
            steps=steps,
            agent_results=agent_results,
            citations=citations,
            warnings=warnings,
            errors=errors,
            confidence=confidence,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=(
                time.perf_counter()
                - start_time
            )
            * 1000,
            metadata={
                "route": _serialize(
                    route
                ),
                "step_count": len(
                    steps
                ),
            },
        )

    # ========================================================================
    # Agent Execution
    # ========================================================================

    def _execute_agent(
        self,
        name: str,
        request: OrchestratorRequest,
        step_id: str,
        previous_results: Sequence[
            AgentExecutionResult
        ],
    ) -> Any:

        normalized_name = (
            self._normalize_name(
                name
            )
        )

        agent = self.agents.get(
            normalized_name
        )

        if agent is None:

            raise AgentNotRegisteredError(
                f"Agent not registered: "
                f"{normalized_name}"
            )

        payload = {
            "query": request.query,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "workflow_id": request.workflow_id,
            "context": request.context,
            "metadata": request.metadata,
            "previous_results": [
                self._result_to_mapping(
                    result
                )
                for result in previous_results
            ],
            "request_id": (
                request.metadata.get(
                    "request_id"
                )
                if request.metadata
                else None
            ),
            "step_id": step_id,
        }

        return self._invoke_component(
            agent,
            request,
            extra=payload,
            operation_names=(
                "run",
                "execute",
                "invoke",
                "process",
                "forecast",
                "analyze",
                "detect_fraud",
            ),
        )

    # ========================================================================
    # Workflow Execution
    # ========================================================================

    def _execute_workflow(
        self,
        name: str,
        request: OrchestratorRequest,
        step_id: str,
    ) -> Any:

        workflow = self.workflows.get(
            self._normalize_name(
                name
            )
        )

        if workflow is None:

            raise WorkflowExecutionError(
                f"Workflow not registered: {name}"
            )

        payload = {
            "query": request.query,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "workflow_id": request.workflow_id,
            "context": request.context,
            "metadata": request.metadata,
            "step_id": step_id,
        }

        return self._invoke_component(
            workflow,
            request,
            extra=payload,
            operation_names=(
                "run",
                "execute",
                "invoke",
                "process",
            ),
        )

    # ========================================================================
    # Result Normalization
    # ========================================================================

    def _normalize_agent_results(
        self,
        result: Any,
        agent_name: str,
        step_id: str,
    ) -> List[
        AgentExecutionResult
    ]:

        if isinstance(
            result,
            AgentExecutionResult,
        ):

            result.step_id = step_id

            return [
                result
            ]

        if isinstance(
            result,
            Sequence,
        ) and not isinstance(
            result,
            (str, bytes, bytearray),
        ):

            normalized: List[
                AgentExecutionResult
            ] = []

            for item in result:

                normalized.extend(
                    self._normalize_agent_results(
                        item,
                        agent_name,
                        step_id,
                    )
                )

            return normalized

        if isinstance(
            result,
            Mapping,
        ):

            success = bool(
                result.get(
                    "success",
                    True,
                )
            )

            result_type = result.get(
                "result_type",
                ResultType.ANSWER.value,
            )

            try:

                result_type = ResultType(
                    result_type
                )

            except ValueError:

                result_type = (
                    ResultType.ANSWER
                )

            return [
                AgentExecutionResult(
                    success=success,
                    agent_name=str(
                        result.get(
                            "agent_name",
                            agent_name,
                        )
                    ),
                    result_type=result_type,
                    answer=result.get(
                        "answer",
                        result.get(
                            "message"
                        ),
                    ),
                    data=result.get(
                        "data",
                        result,
                    ),
                    confidence=_safe_float(
                        result.get(
                            "confidence"
                        )
                    ),
                    citations=list(
                        result.get(
                            "citations",
                            [],
                        )
                        or []
                    ),
                    warnings=list(
                        result.get(
                            "warnings",
                            [],
                        )
                        or []
                    ),
                    errors=list(
                        result.get(
                            "errors",
                            [],
                        )
                        or []
                    ),
                    metadata=dict(
                        result.get(
                            "metadata",
                            {},
                        )
                        or {}
                    ),
                    step_id=step_id,
                )
            ]

        return [
            AgentExecutionResult(
                success=True,
                agent_name=agent_name,
                result_type=(
                    ResultType.ANSWER
                ),
                answer=(
                    str(result)
                    if isinstance(
                        result,
                        str,
                    )
                    else None
                ),
                data=(
                    None
                    if isinstance(
                        result,
                        str,
                    )
                    else result
                ),
                step_id=step_id,
            )
        ]

    # ========================================================================
    # Aggregation
    # ========================================================================

    def _aggregate_answer(
        self,
        request: OrchestratorRequest,
        results: Sequence[
            AgentExecutionResult
        ],
    ) -> Optional[str]:

        answers = [
            result.answer
            for result in results
            if result.success
            and result.answer
        ]

        if not answers:
            return None

        if len(answers) == 1:
            return answers[0]

        sections: List[
            str
        ] = []

        for result in results:

            if not result.success:
                continue

            if not result.answer:
                continue

            sections.append(
                f"{result.agent_name}: "
                f"{result.answer}"
            )

        return "\n\n".join(
            sections
        )

    def _aggregate_data(
        self,
        results: Sequence[
            AgentExecutionResult
        ],
    ) -> Dict[str, Any]:

        data: Dict[
            str,
            Any,
        ] = {}

        for result in results:

            if not result.success:
                continue

            data[
                result.agent_name
            ] = result.data

        return data

    def _aggregate_citations(
        self,
        results: Sequence[
            AgentExecutionResult
        ],
    ) -> List[Any]:

        citations: List[Any] = []

        seen: set[str] = set()

        for result in results:

            for citation in (
                result.citations
            ):

                fingerprint = repr(
                    _serialize(
                        citation
                    )
                )

                if fingerprint in seen:
                    continue

                seen.add(
                    fingerprint
                )

                citations.append(
                    citation
                )

        return citations

    def _aggregate_confidence(
        self,
        results: Sequence[
            AgentExecutionResult
        ],
    ) -> Optional[float]:

        values = [
            result.confidence
            for result in results
            if result.success
            and result.confidence
            is not None
        ]

        if not values:
            return None

        return round(
            sum(values)
            / len(values),
            4,
        )

    # ========================================================================
    # Status
    # ========================================================================

    def _determine_status(
        self,
        results: Sequence[
            AgentExecutionResult
        ],
        steps: Sequence[
            ExecutionStep
        ],
        errors: Sequence[str],
    ) -> ExecutionStatus:

        if not steps:
            return ExecutionStatus.FAILED

        if all(
            step.status
            == StepStatus.BLOCKED
            for step in steps
        ):

            return ExecutionStatus.BLOCKED

        successful = any(
            result.success
            for result in results
        )

        failed_steps = any(
            step.status
            == StepStatus.FAILED
            for step in steps
        )

        blocked_steps = any(
            step.status
            == StepStatus.BLOCKED
            for step in steps
        )

        if successful and (
            failed_steps
            or blocked_steps
            or errors
        ):

            return ExecutionStatus.PARTIAL

        if successful:

            return ExecutionStatus.COMPLETED

        return ExecutionStatus.FAILED

    # ========================================================================
    # Memory
    # ========================================================================

    def _store_memory(
        self,
        request: OrchestratorRequest,
        result: OrchestratorResult,
    ) -> None:

        if not self.config.enable_memory:
            return

        if self.memory is None:
            return

        # Store only bounded, structured information.
        # Do not blindly persist entire raw tool outputs.

        try:

            if hasattr(
                self.memory,
                "add_conversation",
            ):

                if result.answer:

                    self.memory.add_conversation(
                        role="assistant",
                        content=result.answer,
                        company_id=request.company_id,
                        user_id=request.user_id,
                        session_id=request.session_id,
                        workflow_id=request.workflow_id,
                        metadata={
                            "request_id": result.request_id,
                            "status": (
                                result.status.value
                            ),
                            "agents_used": (
                                result.agents_used
                            ),
                        },
                    )

            if (
                hasattr(
                    self.memory,
                    "add_analysis",
                )
                and result.success
            ):

                summary = (
                    result.answer
                    or "Orchestration completed."
                )

                if len(summary) > 5000:
                    summary = summary[:5000]

                self.memory.add_analysis(
                    content=summary,
                    company_id=request.company_id,
                    workflow_id=request.workflow_id,
                    metadata={
                        "request_id": result.request_id,
                        "agents_used": (
                            result.agents_used
                        ),
                    },
                )

        except Exception:
            # Memory failure must not destroy the
            # financial-analysis response.
            return

    # ========================================================================
    # Audit
    # ========================================================================

    def _audit(
        self,
        event: str,
        request: Optional[
            OrchestratorRequest
        ],
        result: Optional[
            OrchestratorResult
        ],
    ) -> None:

        if not self.config.enable_audit:
            return

        if self.audit_service is None:
            return

        payload = {
            "event": event,
            "user_id": (
                request.user_id
                if request
                else None
            ),
            "company_id": (
                request.company_id
                if request
                else None
            ),
            "session_id": (
                request.session_id
                if request
                else None
            ),
            "workflow_id": (
                request.workflow_id
                if request
                else None
            ),
            "request_id": (
                result.request_id
                if result
                else None
            ),
            "status": (
                result.status.value
                if result
                else None
            ),
            "agents_used": (
                result.agents_used
                if result
                else []
            ),
        }

        for method_name in (
            "record",
            "log",
            "create",
            "write",
        ):

            method = getattr(
                self.audit_service,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                method(
                    **payload
                )

            except TypeError:

                try:
                    method(
                        payload
                    )

                except TypeError:
                    continue

            return

    # ========================================================================
    # Generic Component Invocation
    # ========================================================================

    def _invoke_component(
        self,
        component: Any,
        request: OrchestratorRequest,
        extra: Optional[
            Mapping[str, Any]
        ] = None,
        operation_names: Sequence[
            str
        ] = (
            "run",
            "execute",
            "invoke",
        ),
    ) -> Any:

        payload = {
            "query": request.query,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "workflow_id": request.workflow_id,
            "context": request.context,
            "metadata": request.metadata,
        }

        if extra:

            payload.update(
                dict(extra)
            )

        for method_name in operation_names:

            method = getattr(
                component,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                return method(
                    **payload
                )

            except TypeError:

                try:

                    return method(
                        request
                    )

                except TypeError:

                    try:

                        return method(
                            request.query
                        )

                    except TypeError:
                        continue

        if callable(component):

            try:

                return component(
                    **payload
                )

            except TypeError:

                return component(
                    request
                )

        raise OrchestratorExecutionError(
            "Component does not expose a supported "
            "execution interface."
        )

    # ========================================================================
    # Utilities
    # ========================================================================

    def _is_timed_out(
        self,
        start_time: float,
    ) -> bool:

        return (
            time.perf_counter()
            - start_time
        ) > self.config.timeout_seconds

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:

        return (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    @staticmethod
    def _generate_request_id() -> str:

        return (
            "orc_"
            + uuid.uuid4().hex[:24]
        )

    @staticmethod
    def _generate_step_id(
        index: int,
    ) -> str:

        return (
            f"step_{index + 1}_"
            f"{uuid.uuid4().hex[:12]}"
        )

    @staticmethod
    def _result_to_mapping(
        result: AgentExecutionResult,
    ) -> Dict[str, Any]:

        return {
            "success": result.success,
            "agent_name": result.agent_name,
            "result_type": (
                result.result_type.value
            ),
            "answer": result.answer,
            "data": _serialize(
                result.data
            ),
            "confidence": result.confidence,
            "citations": _serialize(
                result.citations
            ),
            "warnings": result.warnings,
            "errors": result.errors,
            "metadata": _serialize(
                result.metadata
            ),
            "step_id": result.step_id,
        }


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
        datetime,
    ):
        return value.isoformat()

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
            _serialize(item)
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
        "dict",
    ):

        try:

            return _serialize(
                value.dict()
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

    if hasattr(
        value,
        "__dict__",
    ):

        try:

            return _serialize(
                vars(value)
            )

        except Exception:
            pass

    return value


def _safe_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None

    if result < 0:
        return 0.0

    if result > 1:
        return 1.0

    return result


# ============================================================================
# Convenience API
# ============================================================================


def orchestrate(
    query: str,
    *,
    company_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    orchestrator: Optional[
        AgentOrchestrator
    ] = None,
    **kwargs: Any,
) -> OrchestratorResult:
    """
    Convenience function for orchestration.
    """

    engine = (
        orchestrator
        or AgentOrchestrator()
    )

    request = OrchestratorRequest(
        query=query,
        company_id=company_id,
        user_id=user_id,
        session_id=session_id,
        workflow_id=workflow_id,
        metadata=kwargs,
    )

    return engine.run(
        request
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "OrchestratorError",
    "InvalidOrchestratorRequestError",
    "OrchestratorConfigurationError",
    "OrchestratorAuthorizationError",
    "OrchestratorExecutionError",
    "OrchestratorTimeoutError",
    "AgentNotRegisteredError",
    "AgentExecutionError",
    "WorkflowExecutionError",
    "GuardrailViolationError",

    # Enums
    "OrchestrationMode",
    "ExecutionStatus",
    "StepStatus",
    "ResultType",

    # Models
    "OrchestratorConfig",
    "OrchestratorRequest",
    "ExecutionStep",
    "AgentExecutionResult",
    "OrchestratorResult",

    # Main class
    "AgentOrchestrator",

    # Convenience
    "orchestrate",
]