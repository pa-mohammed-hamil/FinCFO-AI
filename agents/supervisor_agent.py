
"""
FinCo AI — Supervisor Agent

Responsible for:
1. Receiving the user query.
2. Selecting the correct specialized agent.
3. Managing shared FinCoState.
4. Coordinating agent execution.
5. Handling failures.
6. Preparing the final response.
7. Supporting human review and audit logging.

This implementation is framework-independent.
It can later be integrated with LangGraph or another
agent orchestration framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import uuid4

from app.agents.router import (
    AgentRouter,
    AgentType,
    RouteDecision,
    agent_router,
)

from app.agents.state import (
    AgentStatus,
    FinCoState,
    add_audit_event,
    add_execution_step,
    create_initial_state,
    mark_agent_completed,
    mark_agent_failed,
    update_state,
)


# ============================================================
# TYPE DEFINITIONS
# ============================================================

AgentHandler = Callable[
    [FinCoState],
    Awaitable[FinCoState],
]


# ============================================================
# SUPERVISOR CONFIGURATION
# ============================================================

@dataclass
class SupervisorConfig:
    """
    Configuration for the Supervisor Agent.
    """

    max_iterations: int = 10

    enable_human_review: bool = True

    require_review_for_high_risk: bool = True

    high_risk_threshold: float = 0.75


# ============================================================
# SUPERVISOR AGENT
# ============================================================

class SupervisorAgent:
    """
    Main orchestrator for FinCo AI.

    Responsibilities:
        Query routing
        Agent coordination
        Shared state management
        Error handling
        Human review
        Final response preparation
    """

    def __init__(
        self,
        router: AgentRouter | None = None,
        config: SupervisorConfig | None = None,
    ) -> None:

        self.router = router or agent_router

        self.config = config or SupervisorConfig()

        self.agent_handlers: dict[
            AgentType,
            AgentHandler,
        ] = {}

    # ========================================================
    # REGISTER AGENTS
    # ========================================================

    def register_agent(
        self,
        agent_type: AgentType,
        handler: AgentHandler,
    ) -> None:
        """
        Register a specialized agent.

        Example:
            supervisor.register_agent(
                AgentType.FINANCIAL,
                financial_agent.run,
            )
        """

        self.agent_handlers[agent_type] = handler

    # ========================================================
    # INITIALIZE WORKFLOW
    # ========================================================

    def initialize_state(
        self,
        query: str,
        user_id: str,
        company_id: str,
        request_id: str | None = None,
    ) -> FinCoState:
        """
        Create initial shared state.
        """

        state = create_initial_state(
            query=query,
            user_id=user_id,
            company_id=company_id,
            request_id=request_id or str(uuid4()),
        )

        state = add_audit_event(
            state,
            event_type="workflow_started",
            description="FinCo AI workflow initialized.",
            agent_name="supervisor_agent",
        )

        return state

    # ========================================================
    # ROUTE QUERY
    # ========================================================

    def route_query(
        self,
        state: FinCoState,
    ) -> FinCoState:
        """
        Select the specialized agent for the query.
        """

        decision: RouteDecision = self.router.route(
            state["query"]
        )

        state = update_state(
            state,
            selected_agent=decision.agent.value,
            current_agent=decision.agent.value,
            status=AgentStatus.RUNNING,
        )

        state = add_execution_step(
            state,
            f"Supervisor routed query to {decision.agent.value}",
        )

        state = add_audit_event(
            state,
            event_type="query_routed",
            description=decision.reason,
            agent_name="supervisor_agent",
        )

        return state

    # ========================================================
    # GET NEXT AGENT
    # ========================================================

    def get_next_agent(
        self,
        state: FinCoState,
    ) -> AgentType | None:
        """
        Determine the next agent to execute.
        """

        selected_agent = state.get("selected_agent", "")

        if not selected_agent:
            return None

        try:
            return AgentType(selected_agent)

        except ValueError:
            return None

    # ========================================================
    # EXECUTE AGENT
    # ========================================================

    async def execute_agent(
        self,
        state: FinCoState,
        agent_type: AgentType,
    ) -> FinCoState:
        """
        Execute a registered specialized agent.
        """

        handler = self.agent_handlers.get(agent_type)

        if handler is None:
            return mark_agent_failed(
                state,
                agent_name=agent_type.value,
                error=(
                    f"No handler registered for "
                    f"{agent_type.value}"
                ),
            )

        state = update_state(
            state,
            current_agent=agent_type.value,
            status=AgentStatus.RUNNING,
        )

        state = add_execution_step(
            state,
            f"Started {agent_type.value}",
        )

        state = add_audit_event(
            state,
            event_type="agent_started",
            description=f"Executing {agent_type.value}",
            agent_name=agent_type.value,
        )

        try:
            updated_state = await handler(state)

            updated_state = mark_agent_completed(
                updated_state,
                agent_name=agent_type.value,
            )

            updated_state = add_execution_step(
                updated_state,
                f"Completed {agent_type.value}",
            )

            updated_state = add_audit_event(
                updated_state,
                event_type="agent_completed",
                description=f"{agent_type.value} completed.",
                agent_name=agent_type.value,
            )

            return updated_state

        except Exception as exc:

            failed_state = mark_agent_failed(
                state,
                agent_name=agent_type.value,
                error=str(exc),
            )

            return add_audit_event(
                failed_state,
                event_type="agent_failed",
                description=str(exc),
                agent_name=agent_type.value,
            )

    # ========================================================
    # HUMAN REVIEW CHECK
    # ========================================================

    def check_human_review(
        self,
        state: FinCoState,
    ) -> FinCoState:
        """
        Decide whether human review is required.

        High-risk financial situations should not
        automatically trigger financial actions.
        """

        risk_score = state.get("risk_score")

        requires_review = False

        if self.config.enable_human_review:

            if state.get("active_alert"):
                requires_review = True

            if (
                self.config.require_review_for_high_risk
                and risk_score is not None
                and risk_score >= self.config.high_risk_threshold
            ):
                requires_review = True

            if state.get("recommendations"):
                requires_review = True

        if requires_review:

            return update_state(
                state,
                requires_human_review=True,
                human_review_status="pending",
                status=AgentStatus.WAITING_FOR_REVIEW,
            )

        return state

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    def prepare_final_response(
        self,
        state: FinCoState,
    ) -> FinCoState:
        """
        Prepare the final response from agent outputs.

        A dedicated response generator can replace this
        method later.
        """

        if state.get("error"):
            final_answer = (
                "FinCo AI could not complete the analysis. "
                f"Error: {state['error']}"
            )

        elif state.get("requires_human_review"):
            final_answer = (
                "Your financial analysis is ready for "
                "human review before any recommended "
                "action is approved."
            )

        else:
            final_answer = (
                "FinCo AI completed the financial analysis."
            )

        return update_state(
            state,
            final_answer=final_answer,
            status=AgentStatus.COMPLETED,
        )

    # ========================================================
    # MAIN WORKFLOW
    # ========================================================

    async def run(
        self,
        query: str,
        user_id: str,
        company_id: str,
        request_id: str | None = None,
    ) -> FinCoState:
        """
        Run the complete Supervisor Agent workflow.
        """

        state = self.initialize_state(
            query=query,
            user_id=user_id,
            company_id=company_id,
            request_id=request_id,
        )

        # Step 1 — Route query
        state = self.route_query(state)

        # Step 2 — Find selected agent
        agent_type = self.get_next_agent(state)

        if agent_type is None:
            state = update_state(
                state,
                error="Unable to determine the correct agent.",
                status=AgentStatus.FAILED,
            )

            return self.prepare_final_response(state)

        # Step 3 — Execute agent
        state = await self.execute_agent(
            state,
            agent_type,
        )

        # Step 4 — Stop if agent failed
        if state.get("status") == AgentStatus.FAILED:
            return self.prepare_final_response(state)

        # Step 5 — Check human review
        state = self.check_human_review(state)

        # Step 6 — Final response
        if state.get("requires_human_review"):
            return state

        return self.prepare_final_response(state)


# ============================================================
# DEFAULT SUPERVISOR INSTANCE
# ============================================================

supervisor_agent = SupervisorAgent()