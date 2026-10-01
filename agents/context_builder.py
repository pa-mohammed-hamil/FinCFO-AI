"""
FinCo AI - Agent Context Builder
================================

Builds structured, safe, relevant context for FinCo AI agents.

Purpose
-------
The ContextBuilder sits between the application data layer and the
agent/orchestration layer.

Workflow
--------

User Request
    |
    v
ContextBuilder
    |
    +--> Conversation Context
    +--> Company Context
    +--> Financial Context
    +--> Risk Context
    +--> Fraud Context
    +--> Forecast Context
    +--> RAG Context
    +--> Alert Context
    +--> Scenario Context
    +--> Recommendation Context
    |
    v
Context Compression / Filtering
    |
    v
Guardrail Checks
    |
    v
Structured Agent Context
    |
    v
Supervisor / Specialist Agent

Design principles
-----------------
1. Least-context principle
2. Relevance-first retrieval
3. Token-conscious context construction
4. Financial safety
5. Explainability
6. Tenant/company isolation
7. No secret leakage
8. Deterministic behavior
9. Agent-friendly structured output
10. Auditability

This module intentionally has no dependency on FastAPI, LangChain,
database sessions, or a specific LLM provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
# CONSTANTS
# ============================================================================

DEFAULT_MAX_ITEMS = 20
DEFAULT_MAX_TEXT_LENGTH = 4000
DEFAULT_MAX_TOTAL_TEXT_LENGTH = 30000
DEFAULT_MAX_CONVERSATION_MESSAGES = 12
DEFAULT_MAX_RAG_CHUNKS = 8
DEFAULT_MAX_ALERTS = 10
DEFAULT_MAX_RECOMMENDATIONS = 10
DEFAULT_MAX_TRANSACTIONS = 20
DEFAULT_MAX_FORECAST_POINTS = 12

SENSITIVE_FIELD_NAMES = {
    "password",
    "password_hash",
    "secret",
    "api_key",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
    "private_key",
    "client_secret",
    "database_url",
    "connection_string",
}

PII_FIELD_NAMES = {
    "ssn",
    "social_security_number",
    "credit_card",
    "card_number",
    "cvv",
    "pan",
    "passport_number",
    "national_id",
    "aadhaar",
    "aadhaar_number",
    "tax_id",
    "bank_account",
    "account_number",
}

FINANCIAL_SIGNAL_FIELDS = {
    "revenue",
    "expenses",
    "net_income",
    "gross_profit",
    "operating_income",
    "cash",
    "cash_flow",
    "operating_cash_flow",
    "free_cash_flow",
    "current_assets",
    "current_liabilities",
    "total_assets",
    "total_liabilities",
    "total_debt",
    "equity",
    "total_equity",
    "accounts_receivable",
    "receivables",
    "inventory",
    "accounts_payable",
    "budget",
    "actual_expenses",
}

RISK_SIGNAL_FIELDS = {
    "risk_score",
    "risk_level",
    "fraud_score",
    "anomaly_score",
    "liquidity_risk",
    "credit_risk",
    "market_risk",
    "operational_risk",
    "concentration_risk",
    "forecast_risk",
}

ALLOWED_ROLES = {
    "viewer",
    "analyst",
    "manager",
    "finance_manager",
    "risk_manager",
    "auditor",
    "admin",
    "supervisor",
}

ROLE_CONTEXT_LIMITS = {
    "viewer": {
        "transactions": 5,
        "documents": 5,
        "alerts": 5,
        "recommendations": 5,
    },
    "analyst": {
        "transactions": 20,
        "documents": 8,
        "alerts": 10,
        "recommendations": 10,
    },
    "manager": {
        "transactions": 20,
        "documents": 10,
        "alerts": 15,
        "recommendations": 15,
    },
    "finance_manager": {
        "transactions": 30,
        "documents": 12,
        "alerts": 20,
        "recommendations": 20,
    },
    "risk_manager": {
        "transactions": 30,
        "documents": 12,
        "alerts": 20,
        "recommendations": 20,
    },
    "auditor": {
        "transactions": 50,
        "documents": 20,
        "alerts": 30,
        "recommendations": 30,
    },
    "admin": {
        "transactions": 50,
        "documents": 20,
        "alerts": 30,
        "recommendations": 30,
    },
    "supervisor": {
        "transactions": 30,
        "documents": 12,
        "alerts": 20,
        "recommendations": 20,
    },
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def utc_now() -> datetime:
    """Return the current UTC datetime."""

    return datetime.now(timezone.utc)


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return utc_now().isoformat()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    if value is None:
        return default

    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to integer."""

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def normalize_text(
    value: Any,
    max_length: int = DEFAULT_MAX_TEXT_LENGTH,
) -> str:
    """
    Normalize text and enforce a maximum length.
    """

    if value is None:
        return ""

    text = str(value)

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(text) > max_length:
        return (
            text[: max_length - 3]
            + "..."
        )

    return text


def get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read a value from a mapping or object."""

    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


def first_value(
    obj: Any,
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first available value."""

    for key in keys:
        value = get_value(
            obj,
            key,
            None,
        )

        if value is not None and value != "":
            return value

    return default


def normalize_role(
    role: Any,
) -> str:
    """Normalize a user role."""

    role = normalize_text(
        role
    ).lower()

    if role in ALLOWED_ROLES:
        return role

    return "viewer"


def is_sensitive_field(
    field_name: str,
) -> bool:
    """
    Detect secret/sensitive fields.
    """

    normalized = (
        field_name
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized in SENSITIVE_FIELD_NAMES:
        return True

    if normalized in PII_FIELD_NAMES:
        return True

    return any(
        token in normalized
        for token in (
            "password",
            "secret",
            "token",
            "private_key",
            "api_key",
            "credit_card",
            "cvv",
            "social_security",
        )
    )


def json_safe(
    value: Any,
    max_depth: int = 5,
    _depth: int = 0,
) -> Any:
    """
    Convert arbitrary objects into JSON-safe values while removing
    sensitive fields.
    """

    if _depth > max_depth:
        return "[MAX_DEPTH]"

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        if isinstance(value, float):
            if not math.isfinite(value):
                return None

        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Mapping):
        result = {}

        for key, item in value.items():

            key_text = str(key)

            if is_sensitive_field(
                key_text
            ):
                continue

            result[key_text] = json_safe(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )

        return result

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            json_safe(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if hasattr(
        value,
        "model_dump",
    ):
        try:
            return json_safe(
                value.model_dump(),
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        except Exception:
            pass

    if hasattr(
        value,
        "__dict__",
    ):
        return json_safe(
            vars(value),
            max_depth=max_depth,
            _depth=_depth + 1,
        )

    return normalize_text(
        value
    )


def stable_hash(
    value: Any,
) -> str:
    """
    Generate a deterministic short hash for audit/debug purposes.
    """

    payload = json.dumps(
        json_safe(value),
        sort_keys=True,
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]


def truncate_collection(
    values: Optional[Iterable[Any]],
    limit: int,
) -> List[Any]:
    """Convert an iterable to a bounded list."""

    if values is None:
        return []

    result = []

    for value in values:

        if len(result) >= limit:
            break

        result.append(value)

    return result


def deduplicate_text(
    values: Iterable[str],
) -> List[str]:
    """Remove duplicate text while preserving order."""

    seen = set()
    result = []

    for value in values:

        normalized = normalize_text(
            value
        ).lower()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(
            normalize_text(value)
        )

    return result


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ContextBuilderConfig:
    """
    Configuration controlling context construction.
    """

    max_total_text_length: int = (
        DEFAULT_MAX_TOTAL_TEXT_LENGTH
    )

    max_text_length: int = (
        DEFAULT_MAX_TEXT_LENGTH
    )

    max_conversation_messages: int = (
        DEFAULT_MAX_CONVERSATION_MESSAGES
    )

    max_rag_chunks: int = (
        DEFAULT_MAX_RAG_CHUNKS
    )

    max_alerts: int = (
        DEFAULT_MAX_ALERTS
    )

    max_recommendations: int = (
        DEFAULT_MAX_RECOMMENDATIONS
    )

    max_transactions: int = (
        DEFAULT_MAX_TRANSACTIONS
    )

    max_forecast_points: int = (
        DEFAULT_MAX_FORECAST_POINTS
    )

    include_raw_financial_data: bool = True

    include_transaction_details: bool = True

    include_document_metadata: bool = True

    include_citations: bool = True

    include_audit_metadata: bool = True

    redact_sensitive_fields: bool = True

    require_company_id: bool = True

    minimum_rag_score: float = 0.35

    minimum_recommendation_confidence: float = 0.55

    minimum_alert_score: float = 0.30

    enable_context_hash: bool = True

    enable_compression: bool = True

    max_context_items: int = 100

    def validate(self) -> None:
        """Validate configuration."""

        integer_fields = [
            self.max_total_text_length,
            self.max_text_length,
            self.max_conversation_messages,
            self.max_rag_chunks,
            self.max_alerts,
            self.max_recommendations,
            self.max_transactions,
            self.max_forecast_points,
            self.max_context_items,
        ]

        for value in integer_fields:

            if value <= 0:
                raise ValueError(
                    "Context limits must be positive."
                )

        if not (
            0.0
            <= self.minimum_rag_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_rag_score must be between 0 and 1."
            )

        if not (
            0.0
            <= self.minimum_recommendation_confidence
            <= 1.0
        ):
            raise ValueError(
                "minimum_recommendation_confidence "
                "must be between 0 and 1."
            )

        if not (
            0.0
            <= self.minimum_alert_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_alert_score "
                "must be between 0 and 1."
            )


# ============================================================================
# CONTEXT DATA MODELS
# ============================================================================

@dataclass
class ConversationMessage:
    """Normalized conversation message."""

    role: str

    content: str

    timestamp: Optional[str] = None

    message_id: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "metadata": json_safe(
                self.metadata
            ),
        }


@dataclass
class RAGContextItem:
    """Normalized retrieved document chunk."""

    chunk_id: str

    document_id: str

    content: str

    score: float = 0.0

    source: Optional[str] = None

    page: Optional[int] = None

    section: Optional[str] = None

    document_type: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    citation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "page": self.page,
            "section": self.section,
            "document_type": self.document_type,
            "metadata": json_safe(
                self.metadata
            ),
            "citation": self.citation,
        }


@dataclass
class FinancialContext:
    """Structured financial context."""

    metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    kpis: Dict[str, Any] = field(
        default_factory=dict
    )

    ratios: Dict[str, Any] = field(
        default_factory=dict
    )

    trends: Dict[str, Any] = field(
        default_factory=dict
    )

    transactions: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    statements: Dict[str, Any] = field(
        default_factory=dict
    )

    budget: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": json_safe(
                self.metrics
            ),
            "kpis": json_safe(
                self.kpis
            ),
            "ratios": json_safe(
                self.ratios
            ),
            "trends": json_safe(
                self.trends
            ),
            "transactions": json_safe(
                self.transactions
            ),
            "statements": json_safe(
                self.statements
            ),
            "budget": json_safe(
                self.budget
            ),
        }


@dataclass
class RiskContext:
    """Structured risk context."""

    overall_risk: Optional[str] = None

    risk_score: Optional[float] = None

    risk_signals: Dict[str, Any] = field(
        default_factory=dict
    )

    risk_alerts: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    top_risks: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_risk": self.overall_risk,
            "risk_score": self.risk_score,
            "risk_signals": json_safe(
                self.risk_signals
            ),
            "risk_alerts": json_safe(
                self.risk_alerts
            ),
            "top_risks": list(
                self.top_risks
            ),
        }


@dataclass
class FraudContext:
    """Structured fraud/anomaly context."""

    fraud_score: Optional[float] = None

    anomaly_score: Optional[float] = None

    findings: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    suspicious_transactions: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    explanations: List[str] = field(
        default_factory=list
    )

    human_review_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fraud_score": self.fraud_score,
            "anomaly_score": self.anomaly_score,
            "findings": json_safe(
                self.findings
            ),
            "suspicious_transactions": json_safe(
                self.suspicious_transactions
            ),
            "explanations": list(
                self.explanations
            ),
            "human_review_required": (
                self.human_review_required
            ),
        }


@dataclass
class ForecastContext:
    """Structured forecasting context."""

    horizon: Optional[str] = None

    model: Optional[str] = None

    metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    predictions: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    confidence_intervals: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    downside_risk: Optional[float] = None

    upside_risk: Optional[float] = None

    assumptions: List[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "model": self.model,
            "metrics": json_safe(
                self.metrics
            ),
            "predictions": json_safe(
                self.predictions
            ),
            "confidence_intervals": json_safe(
                self.confidence_intervals
            ),
            "downside_risk": self.downside_risk,
            "upside_risk": self.upside_risk,
            "assumptions": list(
                self.assumptions
            ),
        }


@dataclass
class AlertContext:
    """Structured alert context."""

    active_count: int = 0

    critical_count: int = 0

    high_count: int = 0

    alerts: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_count": self.active_count,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "alerts": json_safe(
                self.alerts
            ),
        }


@dataclass
class RecommendationContext:
    """Structured recommendation context."""

    recommendations: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    high_priority_count: int = 0

    human_review_count: int = 0

    estimated_impact: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommendations": json_safe(
                self.recommendations
            ),
            "high_priority_count": (
                self.high_priority_count
            ),
            "human_review_count": (
                self.human_review_count
            ),
            "estimated_impact": (
                self.estimated_impact
            ),
        }


@dataclass
class AgentContext:
    """
    Complete structured context supplied to an agent.
    """

    context_id: str

    created_at: str

    company_id: Optional[str]

    user_id: Optional[str]

    user_role: str

    task: str

    intent: Optional[str]

    conversation: List[
        ConversationMessage
    ] = field(
        default_factory=list
    )

    company: Dict[str, Any] = field(
        default_factory=dict
    )

    financial: FinancialContext = field(
        default_factory=FinancialContext
    )

    risk: RiskContext = field(
        default_factory=RiskContext
    )

    fraud: FraudContext = field(
        default_factory=FraudContext
    )

    forecast: ForecastContext = field(
        default_factory=ForecastContext
    )

    rag: List[
        RAGContextItem
    ] = field(
        default_factory=list
    )

    alerts: AlertContext = field(
        default_factory=AlertContext
    )

    recommendations: RecommendationContext = field(
        default_factory=RecommendationContext
    )

    what_if: Dict[str, Any] = field(
        default_factory=dict
    )

    tools_available: List[str] = field(
        default_factory=list
    )

    constraints: List[str] = field(
        default_factory=list
    )

    assumptions: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    citations: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    context_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to JSON-safe dictionary."""

        payload = {
            "context_id": self.context_id,
            "created_at": self.created_at,
            "company_id": self.company_id,
            "user_id": self.user_id,
            "user_role": self.user_role,
            "task": self.task,
            "intent": self.intent,
            "conversation": [
                item.to_dict()
                for item in self.conversation
            ],
            "company": json_safe(
                self.company
            ),
            "financial": self.financial.to_dict(),
            "risk": self.risk.to_dict(),
            "fraud": self.fraud.to_dict(),
            "forecast": self.forecast.to_dict(),
            "rag": [
                item.to_dict()
                for item in self.rag
            ],
            "alerts": self.alerts.to_dict(),
            "recommendations": (
                self.recommendations.to_dict()
            ),
            "what_if": json_safe(
                self.what_if
            ),
            "tools_available": list(
                self.tools_available
            ),
            "constraints": list(
                self.constraints
            ),
            "assumptions": list(
                self.assumptions
            ),
            "warnings": list(
                self.warnings
            ),
            "citations": list(
                self.citations
            ),
            "metadata": json_safe(
                self.metadata
            ),
            "context_hash": self.context_hash,
        }

        return payload

    def to_json(
        self,
        indent: int = 2,
    ) -> str:
        """Serialize context as JSON."""

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            default=str,
        )


# ============================================================================
# CONTEXT BUILDER
# ============================================================================

class ContextBuilder:
    """
    Production-oriented context builder for FinCo AI agents.
    """

    def __init__(
        self,
        config: Optional[
            ContextBuilderConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or ContextBuilderConfig()
        )

        self.config.validate()

    # ========================================================================
    # PUBLIC BUILD METHOD
    # ========================================================================

    def build(
        self,
        *,
        task: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_role: str = "viewer",
        intent: Optional[str] = None,
        conversation: Optional[
            Sequence[Any]
        ] = None,
        company: Optional[
            Mapping[str, Any]
        ] = None,
        financial_data: Optional[
            Mapping[str, Any]
        ] = None,
        risk_data: Optional[
            Mapping[str, Any]
        ] = None,
        risk_alerts: Optional[
            Sequence[Any]
        ] = None,
        fraud_data: Optional[
            Mapping[str, Any]
        ] = None,
        fraud_findings: Optional[
            Sequence[Any]
        ] = None,
        forecast_data: Optional[
            Mapping[str, Any]
        ] = None,
        rag_results: Optional[
            Sequence[Any]
        ] = None,
        alerts: Optional[
            Sequence[Any]
        ] = None,
        recommendations: Optional[
            Sequence[Any]
        ] = None,
        what_if: Optional[
            Mapping[str, Any]
        ] = None,
        transactions: Optional[
            Sequence[Any]
        ] = None,
        tools_available: Optional[
            Sequence[str]
        ] = None,
        constraints: Optional[
            Sequence[str]
        ] = None,
        assumptions: Optional[
            Sequence[str]
        ] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> AgentContext:
        """
        Build a complete AgentContext.

        All input collections are bounded and normalized.
        """

        task = normalize_text(
            task,
            self.config.max_text_length,
        )

        if not task:
            raise ValueError(
                "task is required."
            )

        role = normalize_role(
            user_role
        )

        if (
            self.config.require_company_id
            and not company_id
        ):
            raise ValueError(
                "company_id is required "
                "for FinCo tenant isolation."
            )

        context_id = self._create_context_id(
            company_id=company_id,
            user_id=user_id,
            task=task,
        )

        normalized_conversation = (
            self.build_conversation_context(
                conversation
            )
        )

        company_context = (
            self.build_company_context(
                company
            )
        )

        financial_context = (
            self.build_financial_context(
                financial_data,
                transactions=transactions,
                role=role,
            )
        )

        risk_context = (
            self.build_risk_context(
                risk_data,
                risk_alerts,
            )
        )

        fraud_context = (
            self.build_fraud_context(
                fraud_data,
                fraud_findings,
            )
        )

        forecast_context = (
            self.build_forecast_context(
                forecast_data
            )
        )

        rag_context = (
            self.build_rag_context(
                rag_results
            )
        )

        alert_context = (
            self.build_alert_context(
                alerts
            )
        )

        recommendation_context = (
            self.build_recommendation_context(
                recommendations
            )
        )

        what_if_context = (
            self.build_what_if_context(
                what_if
            )
        )

        context = AgentContext(
            context_id=context_id,
            created_at=timestamp(),
            company_id=(
                str(company_id)
                if company_id is not None
                else None
            ),
            user_id=(
                str(user_id)
                if user_id is not None
                else None
            ),
            user_role=role,
            task=task,
            intent=(
                normalize_text(intent)
                if intent
                else self.infer_intent(task)
            ),
            conversation=(
                normalized_conversation
            ),
            company=company_context,
            financial=financial_context,
            risk=risk_context,
            fraud=fraud_context,
            forecast=forecast_context,
            rag=rag_context,
            alerts=alert_context,
            recommendations=(
                recommendation_context
            ),
            what_if=what_if_context,
            tools_available=(
                self.normalize_tools(
                    tools_available
                )
            ),
            constraints=deduplicate_text(
                constraints or []
            ),
            assumptions=deduplicate_text(
                assumptions or []
            ),
            metadata=json_safe(
                metadata or {}
            ),
        )

        self._add_automatic_warnings(
            context
        )

        self._collect_citations(
            context
        )

        if self.config.enable_compression:
            self.compress_context(
                context
            )

        if self.config.enable_context_hash:
            context.context_hash = stable_hash(
                context.to_dict()
            )

        return context

    # ========================================================================
    # CONVERSATION
    # ========================================================================

    def build_conversation_context(
        self,
        messages: Optional[
            Sequence[Any]
        ],
    ) -> List[
        ConversationMessage
    ]:
        """
        Normalize and bound conversation history.

        Most recent messages are retained.
        """

        if not messages:
            return []

        normalized = []

        messages = list(messages)

        start_index = max(
            0,
            len(messages)
            - self.config.max_conversation_messages,
        )

        for message in messages[
            start_index:
        ]:

            role = normalize_text(
                first_value(
                    message,
                    [
                        "role",
                        "sender",
                        "author",
                    ],
                    "user",
                )
            ).lower()

            if role not in {
                "system",
                "user",
                "assistant",
                "tool",
            }:
                role = "user"

            content = normalize_text(
                first_value(
                    message,
                    [
                        "content",
                        "text",
                        "message",
                    ],
                    "",
                ),
                self.config.max_text_length,
            )

            if not content:
                continue

            normalized.append(
                ConversationMessage(
                    role=role,
                    content=content,
                    timestamp=first_value(
                        message,
                        [
                            "timestamp",
                            "created_at",
                        ],
                    ),
                    message_id=first_value(
                        message,
                        [
                            "message_id",
                            "id",
                        ],
                    ),
                    metadata=json_safe(
                        get_value(
                            message,
                            "metadata",
                            {},
                        )
                    ),
                )
            )

        return normalized

    # ========================================================================
    # COMPANY
    # ========================================================================

    def build_company_context(
        self,
        company: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """Build safe company context."""

        if not company:
            return {}

        allowed_fields = {
            "id",
            "company_id",
            "name",
            "legal_name",
            "industry",
            "sector",
            "country",
            "currency",
            "fiscal_year",
            "fiscal_year_end",
            "employee_count",
            "company_size",
            "description",
            "risk_tolerance",
            "reporting_period",
        }

        result = {}

        for field_name in allowed_fields:

            value = get_value(
                company,
                field_name,
                None,
            )

            if value is None:
                continue

            if (
                self.config.redact_sensitive_fields
                and is_sensitive_field(
                    field_name
                )
            ):
                continue

            result[field_name] = json_safe(
                value
            )

        return result

    # ========================================================================
    # FINANCIAL CONTEXT
    # ========================================================================

    def build_financial_context(
        self,
        financial_data: Optional[
            Mapping[str, Any]
        ],
        *,
        transactions: Optional[
            Sequence[Any]
        ] = None,
        role: str = "viewer",
    ) -> FinancialContext:
        """Build structured financial context."""

        context = FinancialContext()

        if financial_data:

            metrics = get_value(
                financial_data,
                "metrics",
                {},
            )

            kpis = get_value(
                financial_data,
                "kpis",
                {},
            )

            ratios = get_value(
                financial_data,
                "ratios",
                {},
            )

            trends = get_value(
                financial_data,
                "trends",
                {},
            )

            statements = get_value(
                financial_data,
                "statements",
                {},
            )

            budget = get_value(
                financial_data,
                "budget",
                {},
            )

            if not isinstance(
                metrics,
                Mapping,
            ):
                metrics = {}

            if not isinstance(
                kpis,
                Mapping,
            ):
                kpis = {}

            if not isinstance(
                ratios,
                Mapping,
            ):
                ratios = {}

            if not isinstance(
                trends,
                Mapping,
            ):
                trends = {}

            if not isinstance(
                statements,
                Mapping,
            ):
                statements = {}

            if not isinstance(
                budget,
                Mapping,
            ):
                budget = {}

            context.metrics = self._filter_financial_fields(
                metrics
            )

            context.kpis = json_safe(
                kpis
            )

            context.ratios = json_safe(
                ratios
            )

            context.trends = json_safe(
                trends
            )

            context.statements = json_safe(
                statements
            )

            context.budget = json_safe(
                budget
            )

            # Support flat financial dictionaries.
            for field_name in FINANCIAL_SIGNAL_FIELDS:

                if field_name in financial_data:

                    context.metrics[
                        field_name
                    ] = json_safe(
                        financial_data[
                            field_name
                        ]
                    )

        transaction_limit = self._role_limit(
            role,
            "transactions",
        )

        if not self.config.include_transaction_details:
            transaction_limit = 0

        context.transactions = (
            self._normalize_transactions(
                transactions,
                transaction_limit,
            )
        )

        return context

    def _filter_financial_fields(
        self,
        values: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Keep relevant financial metrics."""

        result = {}

        for key, value in values.items():

            key_text = str(key)

            if (
                self.config.redact_sensitive_fields
                and is_sensitive_field(
                    key_text
                )
            ):
                continue

            result[key_text] = json_safe(
                value
            )

        return result

    def _normalize_transactions(
        self,
        transactions: Optional[
            Sequence[Any]
        ],
        limit: int,
    ) -> List[
        Dict[str, Any]
    ]:
        """Normalize transaction context."""

        if not transactions or limit <= 0:
            return []

        result = []

        for transaction in list(
            transactions
        )[:limit]:

            normalized = {}

            allowed_fields = {
                "id",
                "transaction_id",
                "date",
                "timestamp",
                "amount",
                "currency",
                "type",
                "category",
                "description",
                "merchant",
                "supplier",
                "customer",
                "status",
                "risk_score",
                "fraud_score",
                "anomaly_score",
                "risk_level",
            }

            for field_name in allowed_fields:

                value = get_value(
                    transaction,
                    field_name,
                    None,
                )

                if value is None:
                    continue

                normalized[
                    field_name
                ] = json_safe(
                    value
                )

            if normalized:
                result.append(
                    normalized
                )

        return result

    # ========================================================================
    # RISK
    # ========================================================================

    def build_risk_context(
        self,
        risk_data: Optional[
            Mapping[str, Any]
        ],
        risk_alerts: Optional[
            Sequence[Any]
        ],
    ) -> RiskContext:
        """Build risk context."""

        context = RiskContext()

        if risk_data:

            context.overall_risk = (
                first_value(
                    risk_data,
                    [
                        "overall_risk",
                        "risk_level",
                        "level",
                    ],
                )
            )

            risk_score = first_value(
                risk_data,
                [
                    "risk_score",
                    "score",
                    "overall_score",
                ],
            )

            if risk_score is not None:
                context.risk_score = (
                    safe_float(
                        risk_score
                    )
                )

            signals = get_value(
                risk_data,
                "risk_signals",
                {},
            )

            if isinstance(
                signals,
                Mapping,
            ):
                context.risk_signals = (
                    json_safe(signals)
                )

            top_risks = get_value(
                risk_data,
                "top_risks",
                [],
            )

            if isinstance(
                top_risks,
                Sequence,
            ) and not isinstance(
                top_risks,
                (str, bytes),
            ):
                context.top_risks = (
                    deduplicate_text(
                        top_risks
                    )
                )

        context.risk_alerts = (
            self._normalize_alerts(
                risk_alerts
            )
        )

        return context

    # ========================================================================
    # FRAUD
    # ========================================================================

    def build_fraud_context(
        self,
        fraud_data: Optional[
            Mapping[str, Any]
        ],
        fraud_findings: Optional[
            Sequence[Any]
        ],
    ) -> FraudContext:
        """Build fraud/anomaly context."""

        context = FraudContext()

        if fraud_data:

            context.fraud_score = self._optional_float(
                first_value(
                    fraud_data,
                    [
                        "fraud_score",
                        "score",
                    ],
                )
            )

            context.anomaly_score = self._optional_float(
                first_value(
                    fraud_data,
                    [
                        "anomaly_score",
                        "anomaly_probability",
                    ],
                )
            )

            explanations = get_value(
                fraud_data,
                "explanations",
                [],
            )

            if isinstance(
                explanations,
                Sequence,
            ) and not isinstance(
                explanations,
                (str, bytes),
            ):
                context.explanations = (
                    deduplicate_text(
                        explanations
                    )
                )

            context.human_review_required = bool(
                get_value(
                    fraud_data,
                    "human_review_required",
                    False,
                )
            )

        context.findings = (
            self._normalize_findings(
                fraud_findings
            )
        )

        if any(
            normalize_text(
                get_value(
                    finding,
                    "risk",
                    get_value(
                        finding,
                        "severity",
                        "",
                    ),
                )
            ).lower()
            in {"critical", "high"}
            for finding in context.findings
        ):
            context.human_review_required = True

        return context

    # ========================================================================
    # FORECAST
    # ========================================================================

    def build_forecast_context(
        self,
        forecast_data: Optional[
            Mapping[str, Any]
        ],
    ) -> ForecastContext:
        """Build forecasting context."""

        context = ForecastContext()

        if not forecast_data:
            return context

        context.horizon = first_value(
            forecast_data,
            [
                "horizon",
                "forecast_horizon",
            ],
        )

        context.model = first_value(
            forecast_data,
            [
                "model",
                "model_name",
            ],
        )

        metrics = get_value(
            forecast_data,
            "metrics",
            {},
        )

        if isinstance(
            metrics,
            Mapping,
        ):
            context.metrics = json_safe(
                metrics
            )

        predictions = get_value(
            forecast_data,
            "predictions",
            get_value(
                forecast_data,
                "forecast",
                [],
            ),
        )

        if isinstance(
            predictions,
            Sequence,
        ) and not isinstance(
            predictions,
            (str, bytes),
        ):
            context.predictions = [
                json_safe(item)
                for item in list(
                    predictions
                )[
                    : self.config.max_forecast_points
                ]
            ]

        intervals = get_value(
            forecast_data,
            "confidence_intervals",
            [],
        )

        if isinstance(
            intervals,
            Sequence,
        ) and not isinstance(
            intervals,
            (str, bytes),
        ):
            context.confidence_intervals = [
                json_safe(item)
                for item in list(
                    intervals
                )[
                    : self.config.max_forecast_points
                ]
            ]

        context.downside_risk = (
            self._optional_float(
                first_value(
                    forecast_data,
                    [
                        "downside_risk",
                        "downside_probability",
                        "forecast_downside_probability",
                    ],
                )
            )
        )

        context.upside_risk = (
            self._optional_float(
                first_value(
                    forecast_data,
                    [
                        "upside_risk",
                        "upside_probability",
                    ],
                )
            )
        )

        assumptions = get_value(
            forecast_data,
            "assumptions",
            [],
        )

        if isinstance(
            assumptions,
            Sequence,
        ) and not isinstance(
            assumptions,
            (str, bytes),
        ):
            context.assumptions = (
                deduplicate_text(
                    assumptions
                )
            )

        return context

    # ========================================================================
    # RAG
    # ========================================================================

    def build_rag_context(
        self,
        rag_results: Optional[
            Sequence[Any]
        ],
    ) -> List[
        RAGContextItem
    ]:
        """
        Build bounded RAG evidence context.

        Results below minimum score are excluded.
        """

        if not rag_results:
            return []

        normalized = []

        for item in rag_results:

            score = safe_float(
                first_value(
                    item,
                    [
                        "score",
                        "similarity",
                        "rerank_score",
                    ],
                    0.0,
                )
            )

            if (
                score
                < self.config.minimum_rag_score
            ):
                continue

            content = normalize_text(
                first_value(
                    item,
                    [
                        "content",
                        "text",
                        "chunk",
                    ],
                    "",
                ),
                self.config.max_text_length,
            )

            if not content:
                continue

            chunk_id = normalize_text(
                first_value(
                    item,
                    [
                        "chunk_id",
                        "id",
                    ],
                    f"chunk-{len(normalized)+1}",
                )
            )

            document_id = normalize_text(
                first_value(
                    item,
                    [
                        "document_id",
                        "doc_id",
                    ],
                    "unknown-document",
                )
            )

            citation = self._build_citation(
                item
            )

            normalized.append(
                RAGContextItem(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    content=content,
                    score=score,
                    source=first_value(
                        item,
                        [
                            "source",
                            "filename",
                            "file_name",
                        ],
                    ),
                    page=self._optional_int(
                        first_value(
                            item,
                            [
                                "page",
                                "page_number",
                            ],
                        )
                    ),
                    section=first_value(
                        item,
                        [
                            "section",
                            "heading",
                        ],
                    ),
                    document_type=first_value(
                        item,
                        [
                            "document_type",
                            "type",
                        ],
                    ),
                    metadata=json_safe(
                        get_value(
                            item,
                            "metadata",
                            {},
                        )
                    ),
                    citation=citation,
                )
            )

        normalized.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return normalized[
            : self.config.max_rag_chunks
        ]

    def _build_citation(
        self,
        item: Any,
    ) -> Optional[str]:
        """Build a citation string from RAG metadata."""

        if not self.config.include_citations:
            return None

        existing = first_value(
            item,
            [
                "citation",
                "citation_text",
            ],
        )

        if existing:
            return normalize_text(
                existing
            )

        source = first_value(
            item,
            [
                "source",
                "filename",
                "file_name",
            ],
        )

        page = first_value(
            item,
            [
                "page",
                "page_number",
            ],
        )

        section = first_value(
            item,
            [
                "section",
                "heading",
            ],
        )

        parts = []

        if source:
            parts.append(
                str(source)
            )

        if page is not None:
            parts.append(
                f"p. {page}"
            )

        if section:
            parts.append(
                str(section)
            )

        if not parts:
            return None

        return " — ".join(parts)

    # ========================================================================
    # ALERTS
    # ========================================================================

    def build_alert_context(
        self,
        alerts: Optional[
            Sequence[Any]
        ],
    ) -> AlertContext:
        """Build alert context."""

        context = AlertContext()

        normalized = self._normalize_alerts(
            alerts
        )

        context.alerts = normalized
        context.active_count = len(
            normalized
        )

        for alert in normalized:

            severity = normalize_text(
                first_value(
                    alert,
                    [
                        "severity",
                        "priority",
                        "risk",
                    ],
                    "",
                )
            ).lower()

            if severity == "critical":
                context.critical_count += 1

            elif severity == "high":
                context.high_count += 1

        return context

    def _normalize_alerts(
        self,
        alerts: Optional[
            Sequence[Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """Normalize alert records."""

        if not alerts:
            return []

        result = []

        for alert in list(
            alerts
        ):

            score = safe_float(
                first_value(
                    alert,
                    [
                        "risk_score",
                        "score",
                        "confidence",
                    ],
                    1.0,
                ),
                1.0,
            )

            if score < self.config.minimum_alert_score:
                continue

            allowed = {
                "id",
                "alert_id",
                "type",
                "category",
                "title",
                "description",
                "severity",
                "priority",
                "risk",
                "risk_score",
                "score",
                "status",
                "created_at",
                "updated_at",
                "entity_id",
                "transaction_id",
                "recommendation",
            }

            normalized = {}

            for field_name in allowed:

                value = get_value(
                    alert,
                    field_name,
                    None,
                )

                if value is None:
                    continue

                normalized[
                    field_name
                ] = json_safe(
                    value
                )

            if normalized:
                result.append(
                    normalized
                )

        result.sort(
            key=lambda item: safe_float(
                item.get(
                    "risk_score",
                    item.get(
                        "score",
                        0,
                    ),
                )
            ),
            reverse=True,
        )

        return result[
            : self.config.max_alerts
        ]

    # ========================================================================
    # FINDINGS
    # ========================================================================

    def _normalize_findings(
        self,
        findings: Optional[
            Sequence[Any]
        ],
    ) -> List[
        Dict[str, Any]
    ]:
        """Normalize fraud findings."""

        if not findings:
            return []

        result = []

        for finding in list(
            findings
        )[
            : self.config.max_alerts
        ]:

            allowed = {
                "id",
                "finding_id",
                "transaction_id",
                "type",
                "category",
                "title",
                "description",
                "severity",
                "risk",
                "risk_score",
                "fraud_score",
                "anomaly_score",
                "evidence",
                "reason",
                "explanation",
                "recommendation",
                "status",
            }

            normalized = {}

            for field_name in allowed:

                value = get_value(
                    finding,
                    field_name,
                    None,
                )

                if value is None:
                    continue

                normalized[
                    field_name
                ] = json_safe(
                    value
                )

            if normalized:
                result.append(
                    normalized
                )

        return result

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================

    def build_recommendation_context(
        self,
        recommendations: Optional[
            Sequence[Any]
        ],
    ) -> RecommendationContext:
        """Build recommendation context."""

        context = RecommendationContext()

        if not recommendations:
            return context

        normalized = []

        for recommendation in list(
            recommendations
        )[
            : self.config.max_recommendations
        ]:

            confidence = safe_float(
                first_value(
                    recommendation,
                    [
                        "confidence",
                        "score",
                    ],
                    1.0,
                ),
                1.0,
            )

            if (
                confidence
                < self.config.minimum_recommendation_confidence
            ):
                continue

            allowed = {
                "id",
                "recommendation_id",
                "type",
                "category",
                "title",
                "description",
                "action",
                "priority",
                "risk",
                "confidence",
                "potential_savings",
                "estimated_impact",
                "impact",
                "evidence",
                "assumptions",
                "risks",
                "next_steps",
                "human_review_required",
                "status",
            }

            normalized_item = {}

            for field_name in allowed:

                value = get_value(
                    recommendation,
                    field_name,
                    None,
                )

                if value is None:
                    continue

                normalized_item[
                    field_name
                ] = json_safe(
                    value
                )

            if not normalized_item:
                continue

            normalized.append(
                normalized_item
            )

        normalized.sort(
            key=lambda item: (
                self._priority_rank(
                    item.get(
                        "priority"
                    )
                ),
                safe_float(
                    item.get(
                        "confidence",
                        0,
                    )
                ),
            ),
            reverse=True,
        )

        context.recommendations = normalized[
            : self.config.max_recommendations
        ]

        for recommendation in context.recommendations:

            priority = normalize_text(
                recommendation.get(
                    "priority",
                    "",
                )
            ).lower()

            if priority in {
                "critical",
                "high",
            }:
                context.high_priority_count += 1

            if bool(
                recommendation.get(
                    "human_review_required",
                    False,
                )
            ):
                context.human_review_count += 1

            context.estimated_impact += safe_float(
                recommendation.get(
                    "potential_savings",
                    recommendation.get(
                        "estimated_impact",
                        recommendation.get(
                            "impact",
                            0,
                        ),
                    ),
                )
            )

        return context

    # ========================================================================
    # WHAT-IF
    # ========================================================================

    def build_what_if_context(
        self,
        what_if: Optional[
            Mapping[str, Any]
        ],
    ) -> Dict[str, Any]:
        """Build scenario-analysis context."""

        if not what_if:
            return {}

        allowed_fields = {
            "scenario_id",
            "name",
            "description",
            "assumptions",
            "inputs",
            "outputs",
            "baseline",
            "scenario",
            "delta",
            "sensitivity",
            "impact",
            "risk",
            "confidence",
        }

        result = {}

        for field_name in allowed_fields:

            value = get_value(
                what_if,
                field_name,
                None,
            )

            if value is None:
                continue

            result[
                field_name
            ] = json_safe(
                value
            )

        return result

    # ========================================================================
    # INTENT
    # ========================================================================

    def infer_intent(
        self,
        task: str,
    ) -> str:
        """
        Lightweight deterministic intent classification.

        The Supervisor/Router can later replace or refine this.
        """

        text = normalize_text(
            task
        ).lower()

        intent_patterns = [
            (
                "fraud_investigation",
                [
                    "fraud",
                    "suspicious transaction",
                    "anomaly",
                    "scam",
                    "duplicate payment",
                ],
            ),
            (
                "forecast_analysis",
                [
                    "forecast",
                    "predict",
                    "prediction",
                    "future revenue",
                    "future profit",
                    "cash flow forecast",
                ],
            ),
            (
                "financial_analysis",
                [
                    "revenue",
                    "profit",
                    "expense",
                    "p&l",
                    "balance sheet",
                    "cash flow",
                    "financial",
                    "margin",
                    "ratio",
                ],
            ),
            (
                "risk_analysis",
                [
                    "risk",
                    "liquidity",
                    "debt",
                    "leverage",
                    "exposure",
                    "concentration",
                ],
            ),
            (
                "document_qa",
                [
                    "document",
                    "annual report",
                    "policy",
                    "contract",
                    "according to",
                    "according",
                    "pdf",
                ],
            ),
            (
                "what_if",
                [
                    "what if",
                    "scenario",
                    "simulate",
                    "simulation",
                    "sensitivity",
                ],
            ),
            (
                "recommendation",
                [
                    "recommend",
                    "recommendation",
                    "what should we do",
                    "action",
                    "optimize",
                    "improve",
                ],
            ),
            (
                "report_generation",
                [
                    "report",
                    "executive summary",
                    "generate report",
                    "management report",
                ],
            ),
        ]

        for intent, keywords in intent_patterns:

            if any(
                keyword in text
                for keyword in keywords
            ):
                return intent

        return "general_financial_analysis"

    # ========================================================================
    # TOOL NORMALIZATION
    # ========================================================================

    def normalize_tools(
        self,
        tools: Optional[
            Sequence[str]
        ],
    ) -> List[str]:
        """Normalize available agent tools."""

        if not tools:
            return []

        result = []

        for tool in tools:

            value = normalize_text(
                tool,
                200,
            )

            if not value:
                continue

            if is_sensitive_field(
                value
            ):
                continue

            if value not in result:
                result.append(
                    value
                )

        return result[
            : self.config.max_context_items
        ]

    # ========================================================================
    # AUTOMATIC WARNINGS
    # ========================================================================

    def _add_automatic_warnings(
        self,
        context: AgentContext,
    ) -> None:
        """Add safety and data-quality warnings."""

        if not context.company_id:
            context.warnings.append(
                "Company scope is missing."
            )

        if (
            context.fraud.human_review_required
        ):
            context.warnings.append(
                "Fraud-related findings require human review."
            )

        if (
            context.alerts.critical_count > 0
        ):
            context.warnings.append(
                "Critical financial/risk alerts are active."
            )

        if (
            context.recommendations.human_review_count
            > 0
        ):
            context.warnings.append(
                "One or more recommendations require human review."
            )

        if (
            not context.rag
            and context.intent
            in {
                "document_qa",
                "financial_analysis",
            }
        ):
            context.warnings.append(
                "No sufficiently relevant RAG evidence was retrieved."
            )

        if (
            context.forecast.downside_risk
            is not None
            and context.forecast.downside_risk
            >= 0.50
        ):
            context.warnings.append(
                "Forecast indicates elevated downside probability."
            )

        context.warnings = deduplicate_text(
            context.warnings
        )

    # ========================================================================
    # CITATIONS
    # ========================================================================

    def _collect_citations(
        self,
        context: AgentContext,
    ) -> None:
        """Collect unique RAG citations."""

        citations = []

        for item in context.rag:

            if item.citation:
                citations.append(
                    item.citation
                )

        context.citations = (
            deduplicate_text(
                citations
            )
        )

    # ========================================================================
    # COMPRESSION
    # ========================================================================

    def compress_context(
        self,
        context: AgentContext,
    ) -> AgentContext:
        """
        Compress context while retaining high-value information.

        This does not use an LLM. It performs deterministic reduction.
        """

        # --------------------------------------------------------------
        # Conversation
        # --------------------------------------------------------------

        if len(
            context.conversation
        ) > self.config.max_conversation_messages:

            context.conversation = (
                context.conversation[
                    -self.config.max_conversation_messages :
                ]
            )

        # --------------------------------------------------------------
        # RAG
        # --------------------------------------------------------------

        if len(
            context.rag
        ) > self.config.max_rag_chunks:

            context.rag.sort(
                key=lambda item: item.score,
                reverse=True,
            )

            context.rag = context.rag[
                : self.config.max_rag_chunks
            ]

        # --------------------------------------------------------------
        # Alerts
        # --------------------------------------------------------------

        if len(
            context.alerts.alerts
        ) > self.config.max_alerts:

            context.alerts.alerts = (
                context.alerts.alerts[
                    : self.config.max_alerts
                ]
            )

        # --------------------------------------------------------------
        # Recommendations
        # --------------------------------------------------------------

        if len(
            context.recommendations.recommendations
        ) > self.config.max_recommendations:

            context.recommendations.recommendations = (
                context.recommendations.recommendations[
                    : self.config.max_recommendations
                ]
            )

        # --------------------------------------------------------------
        # Transactions
        # --------------------------------------------------------------

        if len(
            context.financial.transactions
        ) > self.config.max_transactions:

            context.financial.transactions = (
                context.financial.transactions[
                    : self.config.max_transactions
                ]
            )

        # --------------------------------------------------------------
        # Text budget
        # --------------------------------------------------------------

        self._enforce_text_budget(
            context
        )

        return context

    def _enforce_text_budget(
        self,
        context: AgentContext,
    ) -> None:
        """
        Enforce an approximate text-size budget.

        Priority:
            task
            latest conversation
            high-scoring RAG
            alerts
            recommendations
            other metadata
        """

        current_length = len(
            context.task
        )

        for message in context.conversation:
            current_length += len(
                message.content
            )

        for item in context.rag:
            current_length += len(
                item.content
            )

        if (
            current_length
            <= self.config.max_total_text_length
        ):
            return

        # Reduce RAG content first.
        remaining_budget = max(
            1000,
            self.config.max_total_text_length
            - len(context.task),
        )

        for message in context.conversation:
            remaining_budget -= len(
                message.content
            )

        if remaining_budget <= 0:
            remaining_budget = 1000

        for item in context.rag:

            if len(item.content) <= remaining_budget:
                remaining_budget -= len(
                    item.content
                )
                continue

            if remaining_budget < 100:
                item.content = (
                    item.content[:100]
                    + "..."
                )
                continue

            item.content = (
                item.content[
                    :remaining_budget
                ]
                + "..."
            )

            remaining_budget = 0

    # ========================================================================
    # CONTEXT FOR LLM
    # ========================================================================

    def build_llm_context(
        self,
        context: AgentContext,
        *,
        include_metadata: bool = False,
    ) -> Dict[str, Any]:
        """
        Produce an agent-ready context payload.

        This is the recommended payload for Supervisor/Agent prompts.
        """

        payload = {
            "task": context.task,
            "intent": context.intent,

            "company": context.company,

            "conversation": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in context.conversation
            ],

            "financial": context.financial.to_dict(),

            "risk": context.risk.to_dict(),

            "fraud": context.fraud.to_dict(),

            "forecast": context.forecast.to_dict(),

            "retrieved_evidence": [
                {
                    "content": item.content,
                    "score": item.score,
                    "source": item.source,
                    "page": item.page,
                    "section": item.section,
                    "citation": item.citation,
                    "document_id": item.document_id,
                    "chunk_id": item.chunk_id,
                }
                for item in context.rag
            ],

            "alerts": context.alerts.to_dict(),

            "recommendations": (
                context.recommendations.to_dict()
            ),

            "what_if": context.what_if,

            "constraints": context.constraints,

            "assumptions": context.assumptions,

            "warnings": context.warnings,

            "citations": context.citations,
        }

        if include_metadata:
            payload[
                "metadata"
            ] = {
                "context_id": context.context_id,
                "created_at": context.created_at,
                "context_hash": context.context_hash,
                "user_role": context.user_role,
            }

        return payload

    def build_prompt_context(
        self,
        context: AgentContext,
    ) -> str:
        """
        Convert context into a readable prompt section.

        This is useful for agents that use plain-text prompts.
        """

        sections = []

        sections.append(
            "FINCO AI CONTEXT"
        )

        sections.append(
            f"Task: {context.task}"
        )

        if context.intent:
            sections.append(
                f"Intent: {context.intent}"
            )

        # --------------------------------------------------------------
        # Company
        # --------------------------------------------------------------

        if context.company:
            sections.append(
                "\nCOMPANY"
            )

            sections.append(
                json.dumps(
                    context.company,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        # --------------------------------------------------------------
        # Financial
        # --------------------------------------------------------------

        financial_payload = (
            context.financial.to_dict()
        )

        if any(
            financial_payload.values()
        ):
            sections.append(
                "\nFINANCIAL CONTEXT"
            )

            sections.append(
                json.dumps(
                    financial_payload,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        # --------------------------------------------------------------
        # Risk
        # --------------------------------------------------------------

        if (
            context.risk.risk_score
            is not None
            or context.risk.overall_risk
            or context.risk.risk_alerts
        ):
            sections.append(
                "\nRISK CONTEXT"
            )

            sections.append(
                json.dumps(
                    context.risk.to_dict(),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        # --------------------------------------------------------------
        # Fraud
        # --------------------------------------------------------------

        if (
            context.fraud.fraud_score
            is not None
            or context.fraud.anomaly_score
            is not None
            or context.fraud.findings
        ):
            sections.append(
                "\nFRAUD / ANOMALY CONTEXT"
            )

            sections.append(
                json.dumps(
                    context.fraud.to_dict(),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        # --------------------------------------------------------------
        # Forecast
        # --------------------------------------------------------------

        if (
            context.forecast.predictions
            or context.forecast.metrics
        ):
            sections.append(
                "\nFORECAST CONTEXT"
            )

            sections.append(
                json.dumps(
                    context.forecast.to_dict(),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        # --------------------------------------------------------------
        # RAG
        # --------------------------------------------------------------

        if context.rag:

            sections.append(
                "\nRETRIEVED EVIDENCE"
            )

            for index, item in enumerate(
                context.rag,
                start=1,
            ):

                citation = (
                    item.citation
                    or item.source
                    or item.document_id
                )

                sections.append(
                    f"[Evidence {index}] "
                    f"{citation}"
                )

                sections.append(
                    item.content
                )

        # --------------------------------------------------------------
        # Alerts
        # --------------------------------------------------------------

        if context.alerts.alerts:

            sections.append(
                "\nACTIVE ALERTS"
            )

            for alert in context.alerts.alerts:

                title = alert.get(
                    "title",
                    alert.get(
                        "type",
                        "Alert",
                    ),
                )

                severity = alert.get(
                    "severity",
                    alert.get(
                        "priority",
                        "unknown",
                    ),
                )

                description = alert.get(
                    "description",
                    "",
                )

                sections.append(
                    f"- [{severity}] "
                    f"{title}: "
                    f"{description}"
                )

        # --------------------------------------------------------------
        # Recommendations
        # --------------------------------------------------------------

        if (
            context.recommendations.recommendations
        ):

            sections.append(
                "\nEXISTING RECOMMENDATIONS"
            )

            for recommendation in (
                context.recommendations.recommendations
            ):

                title = recommendation.get(
                    "title",
                    "Recommendation",
                )

                action = recommendation.get(
                    "action",
                    recommendation.get(
                        "description",
                        "",
                    ),
                )

                sections.append(
                    f"- {title}: {action}"
                )

        # --------------------------------------------------------------
        # What-if
        # --------------------------------------------------------------

        if context.what_if:

            sections.append(
                "\nWHAT-IF SCENARIO"
            )

            sections.append(
                json.dumps(
                    context.what_if,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        # --------------------------------------------------------------
        # Safety
        # --------------------------------------------------------------

        if context.constraints:

            sections.append(
                "\nCONSTRAINTS"
            )

            for constraint in context.constraints:
                sections.append(
                    f"- {constraint}"
                )

        if context.assumptions:

            sections.append(
                "\nASSUMPTIONS"
            )

            for assumption in context.assumptions:
                sections.append(
                    f"- {assumption}"
                )

        if context.warnings:

            sections.append(
                "\nWARNINGS"
            )

            for warning in context.warnings:
                sections.append(
                    f"- {warning}"
                )

        return "\n".join(
            sections
        )

    # ========================================================================
    # ROLE-SPECIFIC CONTEXT
    # ========================================================================

    def build_role_context(
        self,
        context: AgentContext,
    ) -> Dict[str, Any]:
        """
        Return role-aware context.

        Viewers receive less transaction/detail context than
        auditors/managers.
        """

        role = normalize_role(
            context.user_role
        )

        payload = self.build_llm_context(
            context
        )

        limits = ROLE_CONTEXT_LIMITS.get(
            role,
            ROLE_CONTEXT_LIMITS[
                "viewer"
            ],
        )

        payload[
            "financial"
        ][
            "transactions"
        ] = payload[
            "financial"
        ][
            "transactions"
        ][: limits[
            "transactions"
        ]]

        payload[
            "retrieved_evidence"
        ] = payload[
            "retrieved_evidence"
        ][: limits[
            "documents"
        ]]

        payload[
            "alerts"
        ][
            "alerts"
        ] = payload[
            "alerts"
        ][
            "alerts"
        ][: limits[
            "alerts"
        ]]

        payload[
            "recommendations"
        ][
            "recommendations"
        ] = payload[
            "recommendations"
        ][
            "recommendations"
        ][: limits[
            "recommendations"
        ]]

        return payload

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate_context(
        self,
        context: AgentContext,
    ) -> Tuple[
        bool,
        List[str],
    ]:
        """
        Validate context before sending it to an agent.
        """

        errors = []

        if not context.context_id:
            errors.append(
                "context_id is missing."
            )

        if not context.task:
            errors.append(
                "task is missing."
            )

        if (
            self.config.require_company_id
            and not context.company_id
        ):
            errors.append(
                "company_id is missing."
            )

        if (
            context.user_role
            not in ALLOWED_ROLES
        ):
            errors.append(
                "Invalid user role."
            )

        # Validate RAG evidence.
        for item in context.rag:

            if not item.content:
                errors.append(
                    f"RAG chunk {item.chunk_id} "
                    "has empty content."
                )

            if not (
                0.0
                <= item.score
                <= 1.0
            ):
                errors.append(
                    f"RAG chunk {item.chunk_id} "
                    "has invalid score."
                )

        # Validate fraud scores.
        for score in (
            context.fraud.fraud_score,
            context.fraud.anomaly_score,
        ):

            if score is not None:

                if not (
                    0.0
                    <= score
                    <= 1.0
                ):
                    errors.append(
                        "Fraud/anomaly score "
                        "must be between 0 and 1."
                    )

        return (
            len(errors) == 0,
            errors,
        )

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _optional_float(
        value: Any,
    ) -> Optional[float]:
        """Convert optional value to float."""

        if value is None:
            return None

        return safe_float(
            value
        )

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> Optional[int]:
        """Convert optional value to int."""

        if value is None:
            return None

        try:
            return int(value)

        except (TypeError, ValueError):
            return None

    @staticmethod
    def _priority_rank(
        priority: Any,
    ) -> int:
        """Return priority ranking."""

        normalized = normalize_text(
            priority
        ).lower()

        return {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }.get(
            normalized,
            0,
        )

    @staticmethod
    def _role_limit(
        role: str,
        field_name: str,
    ) -> int:
        """Return role-specific limit."""

        limits = ROLE_CONTEXT_LIMITS.get(
            role,
            ROLE_CONTEXT_LIMITS[
                "viewer"
            ],
        )

        return safe_int(
            limits.get(
                field_name,
                DEFAULT_MAX_ITEMS,
            ),
            DEFAULT_MAX_ITEMS,
        )

    @staticmethod
    def _create_context_id(
        *,
        company_id: Optional[str],
        user_id: Optional[str],
        task: str,
    ) -> str:
        """Create deterministic context identifier."""

        seed = (
            f"{company_id or 'unknown'}:"
            f"{user_id or 'anonymous'}:"
            f"{task}:"
            f"{timestamp()}"
        )

        digest = hashlib.sha256(
            seed.encode("utf-8")
        ).hexdigest()[:16]

        return f"ctx_{digest}"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_agent_context(
    *,
    task: str,
    company_id: Optional[str],
    user_id: Optional[str] = None,
    user_role: str = "viewer",
    intent: Optional[str] = None,
    conversation: Optional[
        Sequence[Any]
    ] = None,
    company: Optional[
        Mapping[str, Any]
    ] = None,
    financial_data: Optional[
        Mapping[str, Any]
    ] = None,
    risk_data: Optional[
        Mapping[str, Any]
    ] = None,
    risk_alerts: Optional[
        Sequence[Any]
    ] = None,
    fraud_data: Optional[
        Mapping[str, Any]
    ] = None,
    fraud_findings: Optional[
        Sequence[Any]
    ] = None,
    forecast_data: Optional[
        Mapping[str, Any]
    ] = None,
    rag_results: Optional[
        Sequence[Any]
    ] = None,
    alerts: Optional[
        Sequence[Any]
    ] = None,
    recommendations: Optional[
        Sequence[Any]
    ] = None,
    what_if: Optional[
        Mapping[str, Any]
    ] = None,
    transactions: Optional[
        Sequence[Any]
    ] = None,
    tools_available: Optional[
        Sequence[str]
    ] = None,
    constraints: Optional[
        Sequence[str]
    ] = None,
    assumptions: Optional[
        Sequence[str]
    ] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
    config: Optional[
        ContextBuilderConfig
    ] = None,
) -> AgentContext:
    """
    Convenience wrapper for ContextBuilder.build().
    """

    builder = ContextBuilder(
        config=config
    )

    return builder.build(
        task=task,
        company_id=company_id,
        user_id=user_id,
        user_role=user_role,
        intent=intent,
        conversation=conversation,
        company=company,
        financial_data=financial_data,
        risk_data=risk_data,
        risk_alerts=risk_alerts,
        fraud_data=fraud_data,
        fraud_findings=fraud_findings,
        forecast_data=forecast_data,
        rag_results=rag_results,
        alerts=alerts,
        recommendations=recommendations,
        what_if=what_if,
        transactions=transactions,
        tools_available=tools_available,
        constraints=constraints,
        assumptions=assumptions,
        metadata=metadata,
    )


def build_prompt_context(
    context: AgentContext,
) -> str:
    """Convenience wrapper for prompt context generation."""

    return ContextBuilder().build_prompt_context(
        context
    )


def build_llm_context(
    context: AgentContext,
) -> Dict[str, Any]:
    """Convenience wrapper for LLM context."""

    return ContextBuilder().build_llm_context(
        context
    )


def validate_agent_context(
    context: AgentContext,
) -> Tuple[
    bool,
    List[str],
]:
    """Convenience wrapper for context validation."""

    return ContextBuilder().validate_context(
        context
    )


# ============================================================================
# DEMO DATA
# ============================================================================

def create_demo_financial_data() -> Dict[str, Any]:
    """Create realistic demo financial data."""

    return {
        "metrics": {
            "revenue": 5_200_000,
            "expenses": 4_900_000,
            "net_income": 300_000,
            "gross_profit": 1_050_000,
            "operating_income": 420_000,
            "cash": 820_000,
            "operating_cash_flow": 510_000,
            "free_cash_flow": 310_000,
            "current_assets": 1_900_000,
            "current_liabilities": 1_600_000,
            "total_assets": 8_400_000,
            "total_debt": 3_100_000,
            "total_equity": 3_900_000,
            "receivables": 1_050_000,
            "inventory": 920_000,
        },

        "kpis": {
            "revenue_growth": -0.08,
            "gross_margin": 0.202,
            "operating_margin": 0.081,
            "net_margin": 0.058,
            "current_ratio": 1.19,
            "quick_ratio": 0.91,
            "debt_to_equity": 0.79,
        },

        "ratios": {
            "current_ratio": 1.19,
            "quick_ratio": 0.91,
            "debt_to_equity": 0.79,
            "interest_coverage": 3.2,
            "dso": 73.0,
            "inventory_days": 81.0,
        },

        "trends": {
            "revenue": "declining",
            "gross_margin": "declining",
            "operating_expenses": "increasing",
            "cash_flow": "stable",
        },

        "budget": {
            "budget": 4_500_000,
            "actual": 4_900_000,
            "variance_percent": 8.89,
        },
    }


def create_demo_rag_results() -> List[Dict[str, Any]]:
    """Create demo RAG evidence."""

    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "annual-report-2025",
            "content": (
                "The company reported revenue of "
                "$5.2 million for the reporting period, "
                "representing an 8% decline from the "
                "previous period."
            ),
            "score": 0.94,
            "source": "annual_report.pdf",
            "page": 12,
            "section": "Revenue",
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "annual-report-2025",
            "content": (
                "Operating expenses increased during the "
                "period primarily due to higher technology "
                "and administrative costs."
            ),
            "score": 0.88,
            "source": "annual_report.pdf",
            "page": 18,
            "section": "Operating Expenses",
        },
        {
            "chunk_id": "chunk-003",
            "document_id": "finance-policy",
            "content": (
                "Capital expenditure above the approved "
                "threshold requires finance leadership approval."
            ),
            "score": 0.79,
            "source": "finance_policy.pdf",
            "page": 7,
            "section": "Capital Expenditure",
        },
    ]


def create_demo_alerts() -> List[Dict[str, Any]]:
    """Create demo alerts."""

    return [
        {
            "alert_id": "ALT-1001",
            "type": "margin_compression",
            "title": "Operating margin deterioration",
            "description": (
                "Operating margin declined compared "
                "with the previous period."
            ),
            "severity": "high",
            "priority": "high",
            "risk_score": 0.86,
            "status": "active",
        },
        {
            "alert_id": "ALT-1002",
            "type": "budget_variance",
            "title": "Operating expense variance",
            "description": (
                "Actual operating expenses are above budget."
            ),
            "severity": "medium",
            "priority": "medium",
            "risk_score": 0.62,
            "status": "active",
        },
    ]


def create_demo_recommendations() -> List[Dict[str, Any]]:
    """Create demo recommendations."""

    return [
        {
            "recommendation_id": "REC-001",
            "type": "cost_optimization",
            "title": "Review high-growth technology costs",
            "description": (
                "Technology expenses increased faster "
                "than overall revenue."
            ),
            "action": (
                "Review vendor contracts and discretionary "
                "technology spending."
            ),
            "priority": "high",
            "risk": "medium",
            "confidence": 0.91,
            "potential_savings": 125_000,
            "human_review_required": False,
        },
        {
            "recommendation_id": "REC-002",
            "type": "working_capital",
            "title": "Accelerate receivables collection",
            "action": (
                "Prioritize overdue customer invoices."
            ),
            "priority": "high",
            "risk": "low",
            "confidence": 0.87,
            "estimated_impact": 180_000,
            "human_review_required": False,
        },
    ]


def create_demo_context() -> AgentContext:
    """Create a complete demo agent context."""

    builder = ContextBuilder()

    context = builder.build(
        task=(
            "Analyze the company's declining revenue and "
            "operating margin, identify the main risks, "
            "and recommend actions supported by the documents."
        ),

        company_id="company-demo-001",

        user_id="user-demo-001",

        user_role="finance_manager",

        company={
            "company_id": "company-demo-001",
            "name": "FinCo Demo Corporation",
            "industry": "Financial Services",
            "country": "India",
            "currency": "USD",
            "company_size": "mid-market",
            "fiscal_year": 2025,
        },

        conversation=[
            {
                "role": "user",
                "content": (
                    "Revenue has fallen this year. "
                    "What is driving the decline?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I will analyze revenue, margins, "
                    "costs, and supporting documents."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Also identify the highest priority risks."
                ),
            },
        ],

        financial_data=(
            create_demo_financial_data()
        ),

        risk_data={
            "overall_risk": "high",
            "risk_score": 0.78,
            "risk_signals": {
                "liquidity_risk": 0.42,
                "margin_risk": 0.81,
                "revenue_risk": 0.84,
                "concentration_risk": 0.61,
            },
            "top_risks": [
                "Revenue decline",
                "Margin compression",
                "Expense growth",
            ],
        },

        fraud_data={
            "fraud_score": 0.17,
            "anomaly_score": 0.24,
            "human_review_required": False,
        },

        forecast_data={
            "horizon": "6 months",
            "model": "gradient_boosting_baseline",
            "metrics": {
                "mae": 82_000,
                "rmse": 119_000,
            },
            "predictions": [
                {
                    "period": "2026-10",
                    "revenue": 430_000,
                },
                {
                    "period": "2026-11",
                    "revenue": 445_000,
                },
                {
                    "period": "2026-12",
                    "revenue": 452_000,
                },
            ],
            "downside_risk": 0.34,
            "assumptions": [
                "No major customer loss.",
                "Operating expenses remain within historical range.",
            ],
        },

        rag_results=(
            create_demo_rag_results()
        ),

        alerts=(
            create_demo_alerts()
        ),

        recommendations=(
            create_demo_recommendations()
        ),

        what_if={
            "scenario_id": "SCN-001",
            "name": "Revenue recovery scenario",
            "description": (
                "Simulate a 10% revenue recovery."
            ),
            "inputs": {
                "revenue_change_percent": 10,
            },
            "outputs": {
                "estimated_operating_income_change": 230_000,
            },
            "risk": "medium",
            "confidence": 0.82,
        },

        tools_available=[
            "financial_calculator",
            "financial_analysis",
            "database_query",
            "rag_search",
            "forecast",
            "fraud_detection",
            "what_if_simulator",
            "recommendation_engine",
            "report_generator",
        ],

        constraints=[
            "Use retrieved evidence for document-based claims.",
            "Do not invent financial figures.",
            "High-risk financial actions require human approval.",
            "Cite supporting documents when available.",
        ],

        assumptions=[
            "Financial values are normalized to the company's reporting currency.",
            "Forecast values are model estimates, not guarantees.",
        ],
    )

    return context


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """Run a standalone context-builder demonstration."""

    print()
    print("=" * 78)
    print("FinCo AI - Agent Context Builder")
    print("=" * 78)

    context = create_demo_context()

    builder = ContextBuilder()

    valid, errors = builder.validate_context(
        context
    )

    print()
    print("CONTEXT")
    print("-" * 78)
    print(
        f"Context ID:       {context.context_id}"
    )
    print(
        f"Company ID:       {context.company_id}"
    )
    print(
        f"User Role:        {context.user_role}"
    )
    print(
        f"Intent:           {context.intent}"
    )
    print(
        f"RAG Chunks:       {len(context.rag)}"
    )
    print(
        f"Alerts:           {len(context.alerts.alerts)}"
    )
    print(
        "Recommendations:  "
        f"{len(context.recommendations.recommendations)}"
    )
    print(
        f"Context Hash:     {context.context_hash}"
    )

    print()
    print("VALIDATION")
    print("-" * 78)

    print(
        f"Valid: {valid}"
    )

    if errors:
        for error in errors:
            print(
                f"- {error}"
            )

    print()
    print("WARNINGS")
    print("-" * 78)

    if context.warnings:
        for warning in context.warnings:
            print(
                f"- {warning}"
            )
    else:
        print(
            "No warnings."
        )

    print()
    print("CITATIONS")
    print("-" * 78)

    if context.citations:
        for citation in context.citations:
            print(
                f"- {citation}"
            )
    else:
        print(
            "No citations."
        )

    print()
    print("PROMPT CONTEXT PREVIEW")
    print("-" * 78)

    prompt_context = (
        builder.build_prompt_context(
            context
        )
    )

    print(
        prompt_context[:6000]
    )

    print()
    print("=" * 78)
    print(
        "Context building complete."
    )
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()