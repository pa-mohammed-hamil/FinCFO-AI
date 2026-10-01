"""
FinCo AI - Agent Planner
========================

Converts a routed user request into a structured, bounded execution plan.

Architecture
------------

User Request
     |
     v
  Router
     |
     v
  Planner
     |
     +-------------------------------+
     |                               |
     v                               v
Single Agent                    Multi-Agent Plan
     |                               |
     +---------------+---------------+
                     |
                     v
               Orchestrator
                     |
                     v
            Agent / Workflow Execution

Responsibilities
----------------
- Build execution plans from router output.
- Select appropriate specialized agents.
- Define step dependencies.
- Support sequential and parallel execution metadata.
- Add prerequisite/context steps.
- Enforce maximum steps and agents.
- Prevent duplicate agents.
- Validate plans.
- Estimate execution complexity.
- Preserve company/user/workflow scope.
- Never execute tools or agents itself.

Important
---------
The planner is NOT:
- an LLM execution engine,
- a tool executor,
- an authorization bypass,
- a replacement for guardrails,
- a replacement for the orchestrator.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ============================================================================
# Exceptions
# ============================================================================


class PlannerError(Exception):
    """Base planner exception."""


class InvalidPlannerRequestError(
    PlannerError
):
    """Raised when planner input is invalid."""


class PlannerConfigurationError(
    PlannerError
):
    """Raised when planner configuration is invalid."""


class UnsupportedPlanTypeError(
    PlannerError
):
    """Raised when a requested plan type is unsupported."""


class PlanValidationError(
    PlannerError
):
    """Raised when a generated plan is invalid."""


class PlanComplexityError(
    PlannerError
):
    """Raised when a plan exceeds configured complexity."""


class AgentSelectionError(
    PlannerError
):
    """Raised when the planner cannot select a valid agent."""


# ============================================================================
# Constants
# ============================================================================


DEFAULT_MAX_STEPS = 20
DEFAULT_MAX_AGENTS = 10
DEFAULT_MAX_DEPENDENCIES = 10
DEFAULT_MAX_QUERY_LENGTH = 10_000


# ============================================================================
# Enums
# ============================================================================


class PlanType(str, Enum):
    """Types of plans supported by FinCo AI."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    WORKFLOW = "workflow"
    ANALYSIS = "analysis"
    INVESTIGATION = "investigation"
    FORECAST = "forecast"
    DECISION = "decision"
    AUTO = "auto"


class PlanStepType(str, Enum):
    """Execution step categories."""

    AGENT = "agent"
    WORKFLOW = "workflow"
    CONTEXT = "context"
    VALIDATION = "validation"
    AGGREGATION = "aggregation"


class ExecutionMode(str, Enum):
    """How a step should be executed."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class PlanPriority(str, Enum):
    """Plan priority."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AgentCapability(str, Enum):
    """Known FinCo AI agent capabilities."""

    FINANCIAL = "financial"
    FRAUD = "fraud"
    FORECAST = "forecast"
    RAG = "rag"
    WHAT_IF = "what_if"
    RECOMMENDATION = "recommendation"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PlannerConfig:
    """Planner configuration."""

    enabled: bool = True

    max_steps: int = DEFAULT_MAX_STEPS

    max_agents: int = DEFAULT_MAX_AGENTS

    max_dependencies: int = (
        DEFAULT_MAX_DEPENDENCIES
    )

    max_query_length: int = (
        DEFAULT_MAX_QUERY_LENGTH
    )

    add_context_step: bool = True

    add_validation_step: bool = True

    add_aggregation_step: bool = True

    allow_parallel_steps: bool = False

    deduplicate_agents: bool = True

    require_registered_agents: bool = True

    allow_unknown_agents: bool = False

    enable_query_intent_detection: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate planner configuration."""

        numeric_values = {
            "max_steps": self.max_steps,
            "max_agents": self.max_agents,
            "max_dependencies": (
                self.max_dependencies
            ),
            "max_query_length": (
                self.max_query_length
            ),
        }

        for name, value in numeric_values.items():

            if value < 1:

                raise PlannerConfigurationError(
                    f"{name} must be greater than zero."
                )


# ============================================================================
# Request Model
# ============================================================================


@dataclass
class PlannerRequest:
    """Input supplied to the planner."""

    query: str

    route: Optional[
        Mapping[str, Any]
    ] = None

    company_id: Optional[str] = None

    user_id: Optional[str] = None

    session_id: Optional[str] = None

    workflow_id: Optional[str] = None

    requested_agent: Optional[str] = None

    requested_workflow: Optional[str] = None

    context: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# Plan Models
# ============================================================================


@dataclass
class PlanStep:
    """One executable orchestration step."""

    step_id: str

    name: str

    step_type: PlanStepType

    execution_mode: ExecutionMode = (
        ExecutionMode.SEQUENTIAL
    )

    agent_name: Optional[str] = None

    workflow_name: Optional[str] = None

    depends_on: List[str] = field(
        default_factory=list
    )

    description: str = ""

    input_mapping: Dict[str, Any] = field(
        default_factory=dict
    )

    output_key: Optional[str] = None

    priority: PlanPriority = (
        PlanPriority.NORMAL
    )

    required: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "step_id": self.step_id,
            "name": self.name,
            "step_type": self.step_type.value,
            "execution_mode": (
                self.execution_mode.value
            ),
            "agent_name": self.agent_name,
            "workflow_name": self.workflow_name,
            "depends_on": list(
                self.depends_on
            ),
            "description": self.description,
            "input_mapping": _serialize(
                self.input_mapping
            ),
            "output_key": self.output_key,
            "priority": self.priority.value,
            "required": self.required,
            "metadata": _serialize(
                self.metadata
            ),
        }


@dataclass
class ExecutionPlan:
    """Complete executable plan."""

    plan_id: str

    plan_type: PlanType

    query: str

    steps: List[PlanStep] = field(
        default_factory=list
    )

    agents: List[str] = field(
        default_factory=list
    )

    priority: PlanPriority = (
        PlanPriority.NORMAL
    )

    estimated_steps: int = 0

    estimated_agents: int = 0

    requires_human_review: bool = False

    company_id: Optional[str] = None

    user_id: Optional[str] = None

    session_id: Optional[str] = None

    workflow_id: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    warnings: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "plan_id": self.plan_id,
            "plan_type": self.plan_type.value,
            "query": self.query,
            "steps": [
                step.to_dict()
                for step in self.steps
            ],
            "agents": list(
                self.agents
            ),
            "priority": self.priority.value,
            "estimated_steps": (
                self.estimated_steps
            ),
            "estimated_agents": (
                self.estimated_agents
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "company_id": self.company_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "metadata": _serialize(
                self.metadata
            ),
            "warnings": list(
                self.warnings
            ),
        }


# ============================================================================
# Planner
# ============================================================================


class AgentPlanner:
    """
    FinCo AI execution planner.

    The planner determines WHAT should happen.

    The orchestrator determines HOW and WHEN
    those steps actually execute.
    """

    DEFAULT_AGENT_ALIASES = {
        "financial": "financial_agent",
        "finance": "financial_agent",
        "financial_analysis": "financial_agent",
        "fraud": "fraud_agent",
        "fraud_detection": "fraud_agent",
        "fraud_investigation": "fraud_agent",
        "forecast": "forecast_agent",
        "forecasting": "forecast_agent",
        "rag": "rag_agent",
        "documents": "rag_agent",
        "document": "rag_agent",
        "what_if": "what_if_agent",
        "scenario": "what_if_agent",
        "simulation": "what_if_agent",
        "recommendation": "recommendation_agent",
        "recommendations": "recommendation_agent",
    }

    def __init__(
        self,
        config: Optional[
            PlannerConfig
        ] = None,
        registered_agents: Optional[
            Sequence[str]
        ] = None,
        registered_workflows: Optional[
            Sequence[str]
        ] = None,
    ) -> None:

        self.config = (
            config
            or PlannerConfig()
        )

        self.config.validate()

        self.registered_agents = {
            self.normalize_name(
                name
            )
            for name in (
                registered_agents
                or []
            )
        }

        self.registered_workflows = {
            self.normalize_name(
                name
            )
            for name in (
                registered_workflows
                or []
            )
        }

    # ========================================================================
    # Registration
    # ========================================================================

    def register_agent(
        self,
        name: str,
    ) -> None:

        normalized = (
            self.normalize_name(
                name
            )
        )

        if not normalized:

            raise InvalidPlannerRequestError(
                "Agent name cannot be empty."
            )

        self.registered_agents.add(
            normalized
        )

    def register_workflow(
        self,
        name: str,
    ) -> None:

        normalized = (
            self.normalize_name(
                name
            )
        )

        if not normalized:

            raise InvalidPlannerRequestError(
                "Workflow name cannot be empty."
            )

        self.registered_workflows.add(
            normalized
        )

    # ========================================================================
    # Main API
    # ========================================================================

    def run(
        self,
        request: PlannerRequest
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> ExecutionPlan:
        """
        Build an execution plan.

        This is the main method expected by
        AgentOrchestrator.
        """

        request = self.normalize_request(
            request,
            **kwargs,
        )

        self.validate_request(
            request
        )

        if not self.config.enabled:

            raise PlannerConfigurationError(
                "Planner is disabled."
            )

        route = (
            dict(
                request.route
                or {}
            )
        )

        plan_type = (
            self.detect_plan_type(
                request,
                route,
            )
        )

        if plan_type == PlanType.SINGLE_AGENT:

            plan = self.plan_single_agent(
                request,
                route,
            )

        elif plan_type == PlanType.WORKFLOW:

            plan = self.plan_workflow(
                request,
                route,
            )

        elif plan_type in {
            PlanType.MULTI_AGENT,
            PlanType.ANALYSIS,
            PlanType.INVESTIGATION,
            PlanType.FORECAST,
            PlanType.DECISION,
        }:

            plan = self.plan_multi_agent(
                request,
                route,
                plan_type,
            )

        else:

            plan = self.plan_auto(
                request,
                route,
            )

        self.validate_plan(
            plan
        )

        return plan

    # ========================================================================
    # Request Normalization
    # ========================================================================

    def normalize_request(
        self,
        request: PlannerRequest
        | Mapping[str, Any],
        **kwargs: Any,
    ) -> PlannerRequest:

        if isinstance(
            request,
            PlannerRequest,
        ):

            return request

        if not isinstance(
            request,
            Mapping,
        ):

            raise InvalidPlannerRequestError(
                "Planner request must be a PlannerRequest "
                "or mapping."
            )

        payload = dict(
            request
        )

        payload.update(
            kwargs
        )

        return PlannerRequest(
            **payload
        )

    def validate_request(
        self,
        request: PlannerRequest,
    ) -> None:

        if not isinstance(
            request.query,
            str,
        ):

            raise InvalidPlannerRequestError(
                "Query must be a string."
            )

        request.query = (
            request.query.strip()
        )

        if not request.query:

            raise InvalidPlannerRequestError(
                "Query cannot be empty."
            )

        if len(request.query) > (
            self.config.max_query_length
        ):

            raise InvalidPlannerRequestError(
                "Query exceeds maximum length."
            )

    # ========================================================================
    # Plan Type Detection
    # ========================================================================

    def detect_plan_type(
        self,
        request: PlannerRequest,
        route: Mapping[str, Any],
    ) -> PlanType:

        explicit_type = route.get(
            "plan_type"
        )

        if explicit_type:

            try:

                return PlanType(
                    explicit_type
                )

            except ValueError:
                pass

        route_mode = str(
            route.get(
                "mode",
                ""
            )
        ).lower()

        if route_mode == "workflow":

            return PlanType.WORKFLOW

        if (
            request.requested_workflow
            or route.get(
                "workflow"
            )
        ):

            return PlanType.WORKFLOW

        agents = self.extract_agents(
            route
        )

        if (
            request.requested_agent
            or route.get(
                "agent"
            )
        ) and len(agents) <= 1:

            return PlanType.SINGLE_AGENT

        if len(agents) > 1:

            return PlanType.MULTI_AGENT

        if not self.config.enable_query_intent_detection:

            return PlanType.AUTO

        query = request.query.lower()

        if self._contains_any(
            query,
            (
                "fraud",
                "suspicious transaction",
                "fraud investigation",
                "fraud risk",
            ),
        ):

            return PlanType.INVESTIGATION

        if self._contains_any(
            query,
            (
                "forecast",
                "predict",
                "future revenue",
                "future profit",
                "cash flow forecast",
                "liquidity forecast",
            ),
        ):

            return PlanType.FORECAST

        if self._contains_any(
            query,
            (
                "what if",
                "scenario",
                "simulate",
                "sensitivity",
                "impact if",
            ),
        ):

            return PlanType.DECISION

        if self._contains_any(
            query,
            (
                "recommend",
                "recommendation",
                "how can we improve",
                "what should we do",
            ),
        ):

            return PlanType.DECISION

        if self._contains_any(
            query,
            (
                "why",
                "analyze",
                "analysis",
                "compare",
                "explain",
                "breakdown",
            ),
        ):

            return PlanType.ANALYSIS

        return PlanType.AUTO

    # ========================================================================
    # Single Agent Planning
    # ========================================================================

    def plan_single_agent(
        self,
        request: PlannerRequest,
        route: Mapping[str, Any],
    ) -> ExecutionPlan:

        agent_name = (
            request.requested_agent
            or route.get(
                "agent"
            )
        )

        if not agent_name:

            agents = self.extract_agents(
                route
            )

            if agents:

                agent_name = agents[0]

        if not agent_name:

            agent_name = (
                self.infer_primary_agent(
                    request.query
                )
            )

        agent_name = (
            self.resolve_agent_name(
                agent_name
            )
        )

        self.validate_agent_name(
            agent_name
        )

        steps: List[
            PlanStep
        ] = []

        if self.config.add_context_step:

            steps.append(
                self.create_context_step()
            )

        steps.append(
            self.create_agent_step(
                agent_name=agent_name,
                depends_on=(
                    [steps[-1].step_id]
                    if steps
                    else []
                ),
                query=request.query,
            )
        )

        if self.config.add_validation_step:

            steps.append(
                self.create_validation_step(
                    depends_on=[
                        steps[-1].step_id
                    ]
                )
            )

        if self.config.add_aggregation_step:

            steps.append(
                self.create_aggregation_step(
                    depends_on=[
                        steps[-1].step_id
                    ]
                )
            )

        return self.create_plan(
            request=request,
            plan_type=PlanType.SINGLE_AGENT,
            steps=steps,
            agents=[
                agent_name
            ],
        )

    # ========================================================================
    # Workflow Planning
    # ========================================================================

    def plan_workflow(
        self,
        request: PlannerRequest,
        route: Mapping[str, Any],
    ) -> ExecutionPlan:

        workflow_name = (
            request.requested_workflow
            or route.get(
                "workflow"
            )
        )

        if not workflow_name:

            raise UnsupportedPlanTypeError(
                "Workflow plan requires a workflow name."
            )

        workflow_name = (
            self.normalize_name(
                workflow_name
            )
        )

        if (
            self.config.require_registered_agents
            and self.registered_workflows
            and workflow_name
            not in self.registered_workflows
        ):

            raise PlanValidationError(
                f"Workflow is not registered: "
                f"{workflow_name}"
            )

        step = PlanStep(
            step_id=self.generate_step_id(
                1
            ),
            name=workflow_name,
            step_type=PlanStepType.WORKFLOW,
            execution_mode=(
                ExecutionMode.SEQUENTIAL
            ),
            workflow_name=workflow_name,
            description=(
                f"Execute FinCo AI workflow "
                f"'{workflow_name}'."
            ),
            output_key=workflow_name,
            priority=self.infer_priority(
                request.query
            ),
        )

        return self.create_plan(
            request=request,
            plan_type=PlanType.WORKFLOW,
            steps=[
                step
            ],
            agents=self.extract_agents(
                route
            ),
        )

    # ========================================================================
    # Multi-Agent Planning
    # ========================================================================

    def plan_multi_agent(
        self,
        request: PlannerRequest,
        route: Mapping[str, Any],
        plan_type: PlanType = PlanType.MULTI_AGENT,
    ) -> ExecutionPlan:

        agents = self.extract_agents(
            route
        )

        if not agents:

            agents = (
                self.infer_agents_from_query(
                    request.query
                )
            )

        agents = [
            self.resolve_agent_name(
                agent
            )
            for agent in agents
        ]

        agents = self.deduplicate(
            agents
        )

        if not agents:

            raise AgentSelectionError(
                "Planner could not select an agent."
            )

        if len(agents) > (
            self.config.max_agents
        ):

            raise PlanComplexityError(
                "Agent count exceeds planner limit."
            )

        for agent_name in agents:

            self.validate_agent_name(
                agent_name
            )

        steps: List[
            PlanStep
        ] = []

        if self.config.add_context_step:

            steps.append(
                self.create_context_step()
            )

        previous_id: Optional[
            str
        ] = (
            steps[-1].step_id
            if steps
            else None
        )

        parallel_group = (
            self.can_parallelize(
                agents,
                request.query,
            )
        )

        for index, agent_name in enumerate(
            agents,
            start=1,
        ):

            dependencies = (
                [previous_id]
                if previous_id
                and not parallel_group
                else (
                    [
                        steps[0].step_id
                    ]
                    if steps
                    and steps[0].step_type
                    == PlanStepType.CONTEXT
                    else []
                )
            )

            step = (
                self.create_agent_step(
                    agent_name=agent_name,
                    depends_on=dependencies,
                    query=request.query,
                    execution_mode=(
                        ExecutionMode.PARALLEL
                        if parallel_group
                        else ExecutionMode.SEQUENTIAL
                    ),
                    index=index,
                )
            )

            steps.append(
                step
            )

            if not parallel_group:

                previous_id = (
                    step.step_id
                )

        if self.config.add_validation_step:

            steps.append(
                self.create_validation_step(
                    depends_on=[
                        step.step_id
                        for step in steps
                        if step.step_type
                        == PlanStepType.AGENT
                    ]
                )
            )

        if self.config.add_aggregation_step:

            steps.append(
                self.create_aggregation_step(
                    depends_on=[
                        steps[-1].step_id
                    ]
                )
            )

        priority = self.infer_priority(
            request.query
        )

        human_review = (
            priority
            == PlanPriority.CRITICAL
        )

        return self.create_plan(
            request=request,
            plan_type=plan_type,
            steps=steps,
            agents=agents,
            priority=priority,
            requires_human_review=human_review,
        )

    # ========================================================================
    # Automatic Planning
    # ========================================================================

    def plan_auto(
        self,
        request: PlannerRequest,
        route: Mapping[str, Any],
    ) -> ExecutionPlan:

        agents = self.extract_agents(
            route
        )

        if not agents:

            agents = (
                self.infer_agents_from_query(
                    request.query
                )
            )

        if not agents:

            agents = [
                self.infer_primary_agent(
                    request.query
                )
            ]

        agents = self.deduplicate(
            [
                self.resolve_agent_name(
                    agent
                )
                for agent in agents
            ]
        )

        return self.plan_multi_agent(
            request,
            {
                "agents": agents
            },
            PlanType.MULTI_AGENT
            if len(agents) > 1
            else PlanType.SINGLE_AGENT,
        )

    # ========================================================================
    # Agent Selection
    # ========================================================================

    def infer_primary_agent(
        self,
        query: str,
    ) -> str:

        agents = (
            self.infer_agents_from_query(
                query
            )
        )

        if agents:

            return agents[0]

        return "financial_agent"

    def infer_agents_from_query(
        self,
        query: str,
    ) -> List[str]:

        text = query.lower()

        selected: List[
            str
        ] = []

        # --------------------------------------------------------------
        # Document / RAG
        # --------------------------------------------------------------

        if self._contains_any(
            text,
            (
                "document",
                "pdf",
                "contract",
                "annual report",
                "policy",
                "agreement",
                "according to",
                "according to the document",
                "citation",
                "source",
            ),
        ):

            selected.append(
                "rag_agent"
            )

        # --------------------------------------------------------------
        # Fraud
        # --------------------------------------------------------------

        if self._contains_any(
            text,
            (
                "fraud",
                "fraudulent",
                "suspicious",
                "transaction anomaly",
                "unusual transaction",
                "risk transaction",
            ),
        ):

            selected.append(
                "fraud_agent"
            )

        # --------------------------------------------------------------
        # Forecasting
        # --------------------------------------------------------------

        if self._contains_any(
            text,
            (
                "forecast",
                "forecasting",
                "predict",
                "prediction",
                "future revenue",
                "future profit",
                "future cash",
                "cash flow forecast",
                "liquidity forecast",
                "runway",
            ),
        ):

            selected.append(
                "forecast_agent"
            )

        # --------------------------------------------------------------
        # What-if
        # --------------------------------------------------------------

        if self._contains_any(
            text,
            (
                "what if",
                "scenario",
                "simulate",
                "simulation",
                "sensitivity",
                "impact of",
                "if revenue",
                "if expenses",
            ),
        ):

            selected.append(
                "what_if_agent"
            )

        # --------------------------------------------------------------
        # Recommendation
        # --------------------------------------------------------------

        if self._contains_any(
            text,
            (
                "recommend",
                "recommendation",
                "suggest",
                "improve",
                "optimize",
                "what should we do",
                "action plan",
            ),
        ):

            selected.append(
                "recommendation_agent"
            )

        # --------------------------------------------------------------
        # Financial analysis
        # --------------------------------------------------------------

        if self._contains_any(
            text,
            (
                "revenue",
                "expense",
                "profit",
                "loss",
                "margin",
                "p&l",
                "balance sheet",
                "cash flow",
                "financial health",
                "financial analysis",
                "financial performance",
                "liquidity",
                "working capital",
                "budget",
                "financial ratio",
            ),
        ):

            selected.append(
                "financial_agent"
            )

        return self.deduplicate(
            selected
        )

    def extract_agents(
        self,
        route: Mapping[str, Any],
    ) -> List[str]:

        agents: List[
            str
        ] = []

        direct_agent = route.get(
            "agent"
        )

        if direct_agent:

            agents.append(
                str(
                    direct_agent
                )
            )

        route_agents = route.get(
            "agents",
            [],
        )

        if isinstance(
            route_agents,
            str,
        ):

            route_agents = [
                route_agents
            ]

        if isinstance(
            route_agents,
            Sequence,
        ):

            agents.extend(
                str(agent)
                for agent in route_agents
            )

        return self.deduplicate(
            [
                self.resolve_agent_name(
                    agent
                )
                for agent in agents
            ]
        )

    def resolve_agent_name(
        self,
        name: str,
    ) -> str:

        normalized = (
            self.normalize_name(
                name
            )
        )

        return self.DEFAULT_AGENT_ALIASES.get(
            normalized,
            normalized,
        )

    # ========================================================================
    # Step Construction
    # ========================================================================

    def create_context_step(
        self,
    ) -> PlanStep:

        return PlanStep(
            step_id=self.generate_step_id(
                0
            ),
            name="build_context",
            step_type=PlanStepType.CONTEXT,
            execution_mode=(
                ExecutionMode.SEQUENTIAL
            ),
            description=(
                "Build bounded context from request, "
                "memory and available execution context."
            ),
            output_key="context",
            priority=PlanPriority.NORMAL,
            required=True,
        )

    def create_agent_step(
        self,
        agent_name: str,
        depends_on: Optional[
            Sequence[str]
        ] = None,
        query: Optional[str] = None,
        execution_mode: ExecutionMode = (
            ExecutionMode.SEQUENTIAL
        ),
        index: int = 1,
    ) -> PlanStep:

        description = (
            f"Execute {agent_name}"
        )

        if query:

            description += (
                f" for the requested analysis."
            )

        return PlanStep(
            step_id=self.generate_step_id(
                index
            ),
            name=agent_name,
            step_type=PlanStepType.AGENT,
            execution_mode=execution_mode,
            agent_name=agent_name,
            depends_on=list(
                depends_on
                or []
            ),
            description=description,
            input_mapping={
                "query": "request.query",
                "company_id": (
                    "request.company_id"
                ),
                "context": "request.context",
            },
            output_key=agent_name,
            priority=PlanPriority.NORMAL,
            required=True,
        )

    def create_validation_step(
        self,
        depends_on: Optional[
            Sequence[str]
        ] = None,
    ) -> PlanStep:

        return PlanStep(
            step_id=self.generate_step_id(
                900
            ),
            name="validate_results",
            step_type=PlanStepType.VALIDATION,
            execution_mode=(
                ExecutionMode.SEQUENTIAL
            ),
            depends_on=list(
                depends_on
                or []
            ),
            description=(
                "Validate agent results, "
                "errors, confidence and citations."
            ),
            output_key="validated_results",
            priority=PlanPriority.HIGH,
            required=True,
        )

    def create_aggregation_step(
        self,
        depends_on: Optional[
            Sequence[str]
        ] = None,
    ) -> PlanStep:

        return PlanStep(
            step_id=self.generate_step_id(
                1000
            ),
            name="aggregate_results",
            step_type=PlanStepType.AGGREGATION,
            execution_mode=(
                ExecutionMode.SEQUENTIAL
            ),
            depends_on=list(
                depends_on
                or []
            ),
            description=(
                "Aggregate validated agent outputs "
                "into a final response."
            ),
            output_key="final_response",
            priority=PlanPriority.HIGH,
            required=True,
        )

    # ========================================================================
    # Plan Creation
    # ========================================================================

    def create_plan(
        self,
        request: PlannerRequest,
        plan_type: PlanType,
        steps: Sequence[
            PlanStep
        ],
        agents: Sequence[str],
        priority: PlanPriority = (
            PlanPriority.NORMAL
        ),
        requires_human_review: bool = False,
    ) -> ExecutionPlan:

        unique_agents = self.deduplicate(
            [
                self.resolve_agent_name(
                    agent
                )
                for agent in agents
                if agent
            ]
        )

        plan = ExecutionPlan(
            plan_id=self.generate_plan_id(),
            plan_type=plan_type,
            query=request.query,
            steps=list(
                steps
            ),
            agents=unique_agents,
            priority=priority,
            estimated_steps=len(
                steps
            ),
            estimated_agents=len(
                unique_agents
            ),
            requires_human_review=(
                requires_human_review
            ),
            company_id=request.company_id,
            user_id=request.user_id,
            session_id=request.session_id,
            workflow_id=request.workflow_id,
            metadata={
                "planner_version": "1.0",
                "query_length": len(
                    request.query
                ),
            },
        )

        return plan

    # ========================================================================
    # Plan Validation
    # ========================================================================

    def validate_plan(
        self,
        plan: ExecutionPlan,
    ) -> None:

        if not plan.steps:

            raise PlanValidationError(
                "Plan must contain at least one step."
            )

        if len(
            plan.steps
        ) > self.config.max_steps:

            raise PlanComplexityError(
                "Plan exceeds maximum step count."
            )

        if len(
            plan.agents
        ) > self.config.max_agents:

            raise PlanComplexityError(
                "Plan exceeds maximum agent count."
            )

        step_ids = {
            step.step_id
            for step in plan.steps
        }

        if len(step_ids) != len(
            plan.steps
        ):

            raise PlanValidationError(
                "Plan contains duplicate step IDs."
            )

        for step in plan.steps:

            if (
                len(
                    step.depends_on
                )
                > self.config.max_dependencies
            ):

                raise PlanComplexityError(
                    f"Step '{step.name}' has too "
                    f"many dependencies."
                )

            for dependency in (
                step.depends_on
            ):

                if dependency not in step_ids:

                    raise PlanValidationError(
                        f"Step '{step.name}' depends "
                        f"on unknown step '{dependency}'."
                    )

            if step.step_type == (
                PlanStepType.AGENT
            ):

                if not step.agent_name:

                    raise PlanValidationError(
                        f"Agent step '{step.name}' "
                        f"has no agent name."
                    )

                self.validate_agent_name(
                    step.agent_name
                )

            if step.step_type == (
                PlanStepType.WORKFLOW
            ):

                if not step.workflow_name:

                    raise PlanValidationError(
                        f"Workflow step '{step.name}' "
                        f"has no workflow name."
                    )

        self._validate_no_cycles(
            plan.steps
        )

        self._validate_execution_modes(
            plan
        )

    def _validate_no_cycles(
        self,
        steps: Sequence[
            PlanStep
        ],
    ) -> None:

        graph = {
            step.step_id: list(
                step.depends_on
            )
            for step in steps
        }

        visiting: set[
            str
        ] = set()

        visited: set[
            str
        ] = set()

        def visit(
            node: str,
        ) -> None:

            if node in visiting:

                raise PlanValidationError(
                    "Circular dependency detected."
                )

            if node in visited:
                return

            visiting.add(
                node
            )

            for dependency in graph.get(
                node,
                [],
            ):

                visit(
                    dependency
                )

            visiting.remove(
                node
            )

            visited.add(
                node
            )

        for node in graph:

            visit(
                node
            )

    def _validate_execution_modes(
        self,
        plan: ExecutionPlan,
    ) -> None:

        if self.config.allow_parallel_steps:
            return

        for step in plan.steps:

            if (
                step.execution_mode
                == ExecutionMode.PARALLEL
            ):

                raise PlanValidationError(
                    "Parallel execution is disabled "
                    "by planner configuration."
                )

    # ========================================================================
    # Parallelization
    # ========================================================================

    def can_parallelize(
        self,
        agents: Sequence[str],
        query: str,
    ) -> bool:

        if not self.config.allow_parallel_steps:

            return False

        if len(agents) <= 1:

            return False

        # Financial analysis may depend on forecast,
        # fraud or RAG context, so default to sequential.
        #
        # Parallel execution is only metadata here.
        # The orchestrator decides actual scheduling.

        independent_agents = {
            "financial_agent",
            "fraud_agent",
            "forecast_agent",
            "rag_agent",
            "what_if_agent",
            "recommendation_agent",
        }

        return all(
            agent in independent_agents
            for agent in agents
        )

    # ========================================================================
    # Priority
    # ========================================================================

    def infer_priority(
        self,
        query: str,
    ) -> PlanPriority:

        text = query.lower()

        if self._contains_any(
            text,
            (
                "critical",
                "urgent",
                "crisis",
                "insolvency",
                "cash shortage",
                "fraud emergency",
                "immediate action",
            ),
        ):

            return PlanPriority.CRITICAL

        if self._contains_any(
            text,
            (
                "high risk",
                "high-risk",
                "serious",
                "major loss",
                "large loss",
                "significant decline",
            ),
        ):

            return PlanPriority.HIGH

        if self._contains_any(
            text,
            (
                "later",
                "optional",
                "overview",
            ),
        ):

            return PlanPriority.LOW

        return PlanPriority.NORMAL

    # ========================================================================
    # Validation Helpers
    # ========================================================================

    def validate_agent_name(
        self,
        agent_name: str,
    ) -> None:

        normalized = (
            self.resolve_agent_name(
                agent_name
            )
        )

        if not normalized:

            raise AgentSelectionError(
                "Agent name cannot be empty."
            )

        if (
            self.config.require_registered_agents
            and self.registered_agents
            and normalized
            not in self.registered_agents
        ):

            if not self.config.allow_unknown_agents:

                raise AgentSelectionError(
                    f"Agent '{normalized}' is not registered."
                )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @staticmethod
    def normalize_name(
        name: str,
    ) -> str:

        value = (
            str(name)
            .strip()
            .lower()
        )

        value = re.sub(
            r"[\s\-]+",
            "_",
            value,
        )

        return value

    @staticmethod
    def deduplicate(
        values: Iterable[str],
    ) -> List[str]:

        result: List[
            str
        ] = []

        seen: set[
            str
        ] = set()

        for value in values:

            normalized = (
                AgentPlanner.normalize_name(
                    value
                )
            )

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            result.append(
                normalized
            )

        return result

    @staticmethod
    def generate_plan_id() -> str:

        import uuid

        return (
            "plan_"
            + uuid.uuid4().hex[:20]
        )

    @staticmethod
    def generate_step_id(
        index: int,
    ) -> str:

        import uuid

        return (
            f"step_{index}_"
            f"{uuid.uuid4().hex[:10]}"
        )

    @staticmethod
    def _contains_any(
        text: str,
        terms: Sequence[str],
    ) -> bool:

        return any(
            term in text
            for term in terms
        )


# ============================================================================
# Convenience API
# ============================================================================


def create_plan(
    query: str,
    *,
    route: Optional[
        Mapping[str, Any]
    ] = None,
    company_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    planner: Optional[
        AgentPlanner
    ] = None,
    **kwargs: Any,
) -> ExecutionPlan:
    """
    Convenience function for creating an execution plan.
    """

    engine = (
        planner
        or AgentPlanner()
    )

    request = PlannerRequest(
        query=query,
        route=route,
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

    return value


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "PlannerError",
    "InvalidPlannerRequestError",
    "PlannerConfigurationError",
    "UnsupportedPlanTypeError",
    "PlanValidationError",
    "PlanComplexityError",
    "AgentSelectionError",

    # Enums
    "PlanType",
    "PlanStepType",
    "ExecutionMode",
    "PlanPriority",
    "AgentCapability",

    # Configuration
    "PlannerConfig",

    # Models
    "PlannerRequest",
    "PlanStep",
    "ExecutionPlan",

    # Planner
    "AgentPlanner",

    # Convenience
    "create_plan",
]