
"""
FinCo AI — Agent Router

Routes user requests to specialized financial agents.

Supported agents:
- financial_agent
- fraud_agent
- forecast_agent
- rag_agent
- what_if_agent
- recommendation_agent
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class AgentType(str, Enum):
    """Available FinCo AI specialized agents."""

    FINANCIAL = "financial_agent"
    FRAUD = "fraud_agent"
    FORECAST = "forecast_agent"
    RAG = "rag_agent"
    WHAT_IF = "what_if_agent"
    RECOMMENDATION = "recommendation_agent"


@dataclass(frozen=True)
class RouteDecision:
    """Result of the routing process."""

    agent: AgentType
    confidence: float
    reason: str
    requires_confirmation: bool = False


class AgentRouter:
    """
    Classifies financial queries and selects an appropriate agent.

    This first version uses deterministic keyword routing.
    It is easy to test and does not require an LLM.
    """

    def __init__(self) -> None:
        self._routing_rules: dict[AgentType, list[str]] = {
            AgentType.FRAUD: [
                "fraud",
                "suspicious",
                "suspicious transaction",
                "duplicate payment",
                "unusual transaction",
                "money laundering",
                "financial crime",
                "scam",
            ],
            AgentType.FORECAST: [
                "forecast",
                "predict",
                "prediction",
                "future revenue",
                "future profit",
                "cash flow forecast",
                "next month",
                "next quarter",
                "next year",
            ],
            AgentType.WHAT_IF: [
                "what if",
                "scenario",
                "simulate",
                "simulation",
                "sensitivity",
                "increase expenses",
                "reduce expenses",
                "increase revenue",
                "decrease revenue",
                "change pricing",
            ],
            AgentType.RECOMMENDATION: [
                "recommend",
                "recommendation",
                "suggest",
                "advice",
                "what should i do",
                "how can i improve",
                "cost saving",
                "cost savings",
                "business decision",
            ],
            AgentType.RAG: [
                "document",
                "annual report",
                "according to the report",
                "according to the document",
                "uploaded file",
                "policy",
                "invoice details",
                "contract",
                "explain the report",
                "what does the document say",
            ],
            AgentType.FINANCIAL: [
                "revenue",
                "profit",
                "loss",
                "expense",
                "expenses",
                "balance sheet",
                "income statement",
                "cash flow",
                "p&l",
                "pnl",
                "financial health",
                "financial profile",
                "financial ratio",
                "gross margin",
                "net margin",
                "ebitda",
                "financial analysis",
                "financial performance",
            ],
        }

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize user input for matching."""

        query = query.lower().strip()
        query = re.sub(r"\s+", " ", query)

        return query

    def _calculate_scores(
        self,
        query: str,
    ) -> dict[AgentType, int]:
        """Calculate keyword-match scores for each agent."""

        scores = {
            agent: 0
            for agent in AgentType
        }

        for agent, keywords in self._routing_rules.items():
            for keyword in keywords:
                if keyword in query:
                    # Longer phrases are more specific.
                    scores[agent] += len(keyword.split())

        return scores

    def route(self, query: str) -> RouteDecision:
        """
        Route a user query to the most appropriate agent.

        Args:
            query: User's financial question.

        Returns:
            RouteDecision containing selected agent and confidence.

        Raises:
            ValueError: If query is empty or not a string.
        """

        if not isinstance(query, str):
            raise ValueError("Query must be a string.")

        normalized_query = self._normalize_query(query)

        if not normalized_query:
            raise ValueError("Query cannot be empty.")

        scores = self._calculate_scores(normalized_query)

        best_agent = max(
            scores,
            key=scores.get,
        )

        best_score = scores[best_agent]
        total_score = sum(scores.values())

        # No keyword matched: use financial agent as safe default.
        if best_score == 0:
            return RouteDecision(
                agent=AgentType.FINANCIAL,
                confidence=0.30,
                reason=(
                    "No specialized intent detected. "
                    "Routing to the financial agent."
                ),
                requires_confirmation=False,
            )

        confidence = min(
            0.95,
            0.50 + (best_score / max(total_score, 1)) * 0.45,
        )

        return RouteDecision(
            agent=best_agent,
            confidence=round(confidence, 2),
            reason=(
                f"Matched financial intent for "
                f"{best_agent.value}."
            ),
            requires_confirmation=False,
        )

    def get_agent_name(self, query: str) -> str:
        """Return only the selected agent name."""

        return self.route(query).agent.value


# Singleton router for application-wide use.
agent_router = AgentRouter()


def route_query(query: str) -> RouteDecision:
    """Convenience function for routing queries."""

    return agent_router.route(query)