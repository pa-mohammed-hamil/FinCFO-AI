"""
FinCo AI - Agent Memory
=======================

Controlled memory layer for Agentic AI workflows.

Purpose
-------
Provides safe, structured memory for:

- Conversation context
- Previous agent decisions
- Tool observations
- Financial-analysis context
- User preferences relevant to the workflow
- Workflow state summaries
- Retrieved evidence references
- Previous recommendations
- Important facts

Design principles
-----------------
1. Memory is scoped to a company.
2. Memory can be scoped to a user/session/workflow.
3. Memory has explicit types.
4. Memory can expire.
5. Sensitive values can be filtered.
6. Memory size is bounded.
7. Duplicate memories are avoided.
8. Memory retrieval is deterministic.
9. LLM-generated memory is treated as untrusted.
10. Memory must never override authorization or system policy.

Architecture
------------

Supervisor Agent
       |
       v
   Agent Memory
       |
       +-----------------------+
       |           |           |
       v           v           v
 Short-Term   Long-Term     Workflow
  Context      Memory        Memory
       |           |           |
       +-----------+-----------+
                   |
                   v
             Context Builder
                   |
                   v
              Agent Prompt

IMPORTANT
---------
Memory is contextual state, NOT authorization.

The agent must never use memory to:
- bypass permissions,
- change company scope,
- override system instructions,
- approve financial transactions,
- disable guardrails.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock
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


class AgentMemoryError(Exception):
    """Base exception for agent memory errors."""


class InvalidMemoryError(AgentMemoryError):
    """Raised when memory data is invalid."""


class MemoryNotFoundError(AgentMemoryError):
    """Raised when requested memory does not exist."""


class MemoryCapacityError(AgentMemoryError):
    """Raised when memory capacity is exceeded."""


class MemoryScopeError(AgentMemoryError):
    """Raised when memory scope is invalid."""


class MemorySecurityError(AgentMemoryError):
    """Raised when unsafe data is detected."""


class MemoryConfigurationError(AgentMemoryError):
    """Raised when memory configuration is invalid."""


# ============================================================================
# Constants
# ============================================================================


DEFAULT_MAX_MEMORIES = 500

DEFAULT_MAX_SHORT_TERM_MEMORIES = 20

DEFAULT_MAX_MEMORY_LENGTH = 10_000

DEFAULT_MAX_CONTEXT_LENGTH = 30_000

DEFAULT_TTL_HOURS = 24

DEFAULT_SESSION_TTL_HOURS = 12

DEFAULT_SUMMARY_MAX_LENGTH = 8_000

DEFAULT_RELEVANCE_THRESHOLD = 0.0


SENSITIVE_FIELD_PATTERNS = (
    r"password",
    r"passwd",
    r"secret",
    r"api[_-]?key",
    r"access[_-]?token",
    r"refresh[_-]?token",
    r"private[_-]?key",
    r"credit[_-]?card",
    r"card[_-]?number",
    r"cvv",
    r"cvc",
)


# ============================================================================
# Enums
# ============================================================================


class MemoryType(str, Enum):
    """Supported agent memory types."""

    CONVERSATION = "conversation"
    USER_PREFERENCE = "user_preference"
    FACT = "fact"
    DECISION = "decision"
    TOOL_OBSERVATION = "tool_observation"
    ANALYSIS = "analysis"
    RECOMMENDATION = "recommendation"
    EVIDENCE = "evidence"
    WORKFLOW = "workflow"
    SUMMARY = "summary"
    SYSTEM = "system"


class MemoryScope(str, Enum):
    """Memory ownership/scope."""

    SESSION = "session"
    USER = "user"
    COMPANY = "company"
    WORKFLOW = "workflow"


class MemoryImportance(str, Enum):
    """Memory importance."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemorySource(str, Enum):
    """Where memory originated."""

    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    SYSTEM = "system"
    RETRIEVER = "retriever"
    WORKFLOW = "workflow"


class MemoryStatus(str, Enum):
    """Memory lifecycle state."""

    ACTIVE = "active"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    DELETED = "deleted"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class MemoryConfig:
    """Configuration for agent memory."""

    enabled: bool = True

    max_memories: int = DEFAULT_MAX_MEMORIES

    max_short_term_memories: int = (
        DEFAULT_MAX_SHORT_TERM_MEMORIES
    )

    max_memory_length: int = (
        DEFAULT_MAX_MEMORY_LENGTH
    )

    max_context_length: int = (
        DEFAULT_MAX_CONTEXT_LENGTH
    )

    default_ttl_hours: int = DEFAULT_TTL_HOURS

    session_ttl_hours: int = (
        DEFAULT_SESSION_TTL_HOURS
    )

    summary_max_length: int = (
        DEFAULT_SUMMARY_MAX_LENGTH
    )

    relevance_threshold: float = (
        DEFAULT_RELEVANCE_THRESHOLD
    )

    deduplicate: bool = True

    sanitize_sensitive_data: bool = True

    allow_user_memory: bool = True

    allow_company_memory: bool = True

    allow_workflow_memory: bool = True

    allow_agent_memory: bool = True

    allow_tool_memory: bool = True

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate memory configuration."""

        positive_fields = {
            "max_memories": self.max_memories,
            "max_short_term_memories": (
                self.max_short_term_memories
            ),
            "max_memory_length": (
                self.max_memory_length
            ),
            "max_context_length": (
                self.max_context_length
            ),
            "default_ttl_hours": (
                self.default_ttl_hours
            ),
            "session_ttl_hours": (
                self.session_ttl_hours
            ),
            "summary_max_length": (
                self.summary_max_length
            ),
        }

        for name, value in positive_fields.items():

            if value < 1:

                raise MemoryConfigurationError(
                    f"{name} must be greater than zero."
                )

        if not (
            0.0
            <= self.relevance_threshold
            <= 1.0
        ):

            raise MemoryConfigurationError(
                "relevance_threshold must be "
                "between 0 and 1."
            )


# ============================================================================
# Memory Models
# ============================================================================


@dataclass
class MemoryItem:
    """
    Individual memory record.
    """

    memory_id: str

    content: str

    memory_type: MemoryType = (
        MemoryType.CONVERSATION
    )

    scope: MemoryScope = MemoryScope.SESSION

    source: MemorySource = MemorySource.AGENT

    importance: MemoryImportance = (
        MemoryImportance.MEDIUM
    )

    company_id: Optional[str] = None

    user_id: Optional[str] = None

    session_id: Optional[str] = None

    workflow_id: Optional[str] = None

    tags: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    updated_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    expires_at: Optional[datetime] = None

    status: MemoryStatus = MemoryStatus.ACTIVE

    access_count: int = 0

    relevance_score: float = 0.0

    fingerprint: Optional[str] = None

    def __post_init__(self) -> None:

        if not self.fingerprint:

            self.fingerprint = (
                calculate_memory_fingerprint(
                    self.content
                )
            )

    @property
    def is_expired(self) -> bool:

        if self.status != MemoryStatus.ACTIVE:
            return True

        if self.expires_at is None:
            return False

        return (
            _ensure_utc(self.expires_at)
            <= datetime.now(timezone.utc)
        )

    def touch(self) -> None:

        self.access_count += 1

        self.updated_at = datetime.now(
            timezone.utc
        )

    def to_dict(
        self,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:

        result = {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "scope": self.scope.value,
            "source": self.source.value,
            "importance": self.importance.value,
            "company_id": self.company_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "tags": list(self.tags),
            "created_at": (
                self.created_at.isoformat()
            ),
            "updated_at": (
                self.updated_at.isoformat()
            ),
            "expires_at": (
                self.expires_at.isoformat()
                if self.expires_at
                else None
            ),
            "status": self.status.value,
            "access_count": self.access_count,
            "relevance_score": self.relevance_score,
            "fingerprint": self.fingerprint,
        }

        if include_metadata:
            result["metadata"] = (
                _serialize(self.metadata)
            )

        return result


@dataclass
class MemoryQuery:
    """Query used to retrieve memories."""

    query: Optional[str] = None

    company_id: Optional[str] = None

    user_id: Optional[str] = None

    session_id: Optional[str] = None

    workflow_id: Optional[str] = None

    memory_types: Optional[
        Sequence[MemoryType]
    ] = None

    scopes: Optional[
        Sequence[MemoryScope]
    ] = None

    tags: Optional[
        Sequence[str]
    ] = None

    minimum_importance: (
        MemoryImportance
        | None
    ) = None

    limit: int = 10

    include_expired: bool = False


@dataclass
class MemoryResult:
    """Result of a memory operation."""

    success: bool

    memories: List[MemoryItem] = field(
        default_factory=list
    )

    count: int = 0

    context: str = ""

    warnings: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "success": self.success,
            "memories": [
                memory.to_dict()
                for memory in self.memories
            ],
            "count": self.count,
            "context": self.context,
            "warnings": self.warnings,
            "metadata": _serialize(
                self.metadata
            ),
        }


# ============================================================================
# Main Memory Manager
# ============================================================================


class AgentMemory:
    """
    Controlled in-process memory manager.

    The implementation is intentionally storage-agnostic.

    For production deployments, this class can sit behind:
        - PostgreSQL
        - Redis
        - vector database
        - dedicated memory service

    while retaining the same public interface.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        repository: Any = None,
        audit_service: Any = None,
    ) -> None:

        self.config = (
            config
            or MemoryConfig()
        )

        self.config.validate()

        self.repository = repository

        self.audit_service = audit_service

        self._memory: "OrderedDict[str, MemoryItem]" = (
            OrderedDict()
        )

        self._fingerprints: Dict[
            str,
            str,
        ] = {}

        self._lock = RLock()

    # ========================================================================
    # Create
    # ========================================================================

    def add(
        self,
        content: str,
        memory_type: MemoryType = (
            MemoryType.CONVERSATION
        ),
        scope: MemoryScope = MemoryScope.SESSION,
        source: MemorySource = MemorySource.AGENT,
        importance: MemoryImportance = (
            MemoryImportance.MEDIUM
        ),
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        tags: Optional[
            Sequence[str]
        ] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
        ttl_hours: Optional[int] = None,
        memory_id: Optional[str] = None,
    ) -> MemoryItem:
        """
        Add a memory item.

        Company/session/user scope must be explicitly supplied
        by the caller where applicable.
        """

        if not self.config.enabled:

            raise MemoryConfigurationError(
                "Agent memory is disabled."
            )

        content = self._validate_content(
            content
        )

        self._validate_scope(
            scope=scope,
            company_id=company_id,
            user_id=user_id,
            session_id=session_id,
            workflow_id=workflow_id,
        )

        self._validate_source(
            source
        )

        if ttl_hours is None:

            ttl_hours = (
                self.config.session_ttl_hours
                if scope == MemoryScope.SESSION
                else self.config.default_ttl_hours
            )

        if ttl_hours < 1:

            raise InvalidMemoryError(
                "ttl_hours must be greater than zero."
            )

        sanitized_content = (
            self.sanitize_content(
                content
            )
        )

        fingerprint = (
            calculate_memory_fingerprint(
                sanitized_content
            )
        )

        with self._lock:

            if self.config.deduplicate:

                existing_id = (
                    self._fingerprints.get(
                        fingerprint
                    )
                )

                if existing_id:

                    existing = (
                        self._memory.get(
                            existing_id
                        )
                    )

                    if existing:

                        existing.updated_at = (
                            datetime.now(
                                timezone.utc
                            )
                        )

                        existing.access_count += 1

                        self._audit(
                            "memory_deduplicated",
                            existing,
                        )

                        return existing

            if len(self._memory) >= (
                self.config.max_memories
            ):

                self._evict_oldest()

            created_at = datetime.now(
                timezone.utc
            )

            item = MemoryItem(
                memory_id=(
                    memory_id
                    or generate_memory_id()
                ),
                content=sanitized_content,
                memory_type=memory_type,
                scope=scope,
                source=source,
                importance=importance,
                company_id=company_id,
                user_id=user_id,
                session_id=session_id,
                workflow_id=workflow_id,
                tags=[
                    str(tag).strip()
                    for tag in (tags or [])
                    if str(tag).strip()
                ],
                metadata=dict(
                    metadata or {}
                ),
                created_at=created_at,
                updated_at=created_at,
                expires_at=(
                    created_at
                    + timedelta(
                        hours=ttl_hours
                    )
                ),
                fingerprint=fingerprint,
            )

            self._memory[
                item.memory_id
            ] = item

            self._fingerprints[
                fingerprint
            ] = item.memory_id

            self._persist(
                "add",
                item,
            )

            self._audit(
                "memory_added",
                item,
            )

            return item

    # ========================================================================
    # Conversation Memory
    # ========================================================================

    def add_conversation(
        self,
        role: str,
        content: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> MemoryItem:
        """Store a conversation turn."""

        role = (
            str(role)
            .strip()
            .lower()
        )

        if role not in {
            "user",
            "assistant",
            "system",
            "tool",
        }:

            raise InvalidMemoryError(
                "Unsupported conversation role."
            )

        return self.add(
            content=f"{role}: {content}",
            memory_type=MemoryType.CONVERSATION,
            scope=MemoryScope.SESSION,
            source=(
                MemorySource.USER
                if role == "user"
                else MemorySource.AGENT
            ),
            importance=MemoryImportance.MEDIUM,
            company_id=company_id,
            user_id=user_id,
            session_id=session_id,
            workflow_id=workflow_id,
            metadata=metadata,
        )

    # ========================================================================
    # Specialized Memory
    # ========================================================================

    def add_fact(
        self,
        content: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        importance: MemoryImportance = (
            MemoryImportance.HIGH
        ),
        tags: Optional[
            Sequence[str]
        ] = None,
    ) -> MemoryItem:

        return self.add(
            content=content,
            memory_type=MemoryType.FACT,
            scope=(
                MemoryScope.COMPANY
                if company_id
                else MemoryScope.USER
                if user_id
                else MemoryScope.WORKFLOW
            ),
            source=MemorySource.AGENT,
            importance=importance,
            company_id=company_id,
            user_id=user_id,
            workflow_id=workflow_id,
            tags=tags,
        )

    def add_decision(
        self,
        content: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> MemoryItem:

        return self.add(
            content=content,
            memory_type=MemoryType.DECISION,
            scope=MemoryScope.WORKFLOW,
            source=MemorySource.AGENT,
            importance=MemoryImportance.HIGH,
            company_id=company_id,
            user_id=user_id,
            workflow_id=workflow_id,
            metadata=metadata,
        )

    def add_tool_observation(
        self,
        tool_name: str,
        observation: str,
        company_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> MemoryItem:

        return self.add(
            content=(
                f"Tool: {tool_name}\n"
                f"Observation: {observation}"
            ),
            memory_type=MemoryType.TOOL_OBSERVATION,
            scope=MemoryScope.WORKFLOW,
            source=MemorySource.TOOL,
            importance=MemoryImportance.MEDIUM,
            company_id=company_id,
            session_id=session_id,
            workflow_id=workflow_id,
            metadata=metadata,
        )

    def add_analysis(
        self,
        content: str,
        company_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> MemoryItem:

        return self.add(
            content=content,
            memory_type=MemoryType.ANALYSIS,
            scope=MemoryScope.WORKFLOW,
            source=MemorySource.AGENT,
            importance=MemoryImportance.HIGH,
            company_id=company_id,
            workflow_id=workflow_id,
            metadata=metadata,
        )

    def add_recommendation(
        self,
        content: str,
        company_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> MemoryItem:

        return self.add(
            content=content,
            memory_type=MemoryType.RECOMMENDATION,
            scope=MemoryScope.WORKFLOW,
            source=MemorySource.AGENT,
            importance=MemoryImportance.HIGH,
            company_id=company_id,
            workflow_id=workflow_id,
            metadata=metadata,
        )

    def add_evidence(
        self,
        content: str,
        company_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> MemoryItem:

        return self.add(
            content=content,
            memory_type=MemoryType.EVIDENCE,
            scope=MemoryScope.WORKFLOW,
            source=MemorySource.RETRIEVER,
            importance=MemoryImportance.HIGH,
            company_id=company_id,
            workflow_id=workflow_id,
            metadata=metadata,
        )

    # ========================================================================
    # Retrieval
    # ========================================================================

    def retrieve(
        self,
        query: Optional[
            MemoryQuery
        ] = None,
        *,
        text: Optional[str] = None,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        limit: int = 10,
    ) -> MemoryResult:
        """
        Retrieve memories using deterministic filtering and
        lightweight lexical relevance scoring.

        A vector database can replace the scoring layer later.
        """

        if query is None:

            query = MemoryQuery(
                query=text,
                company_id=company_id,
                user_id=user_id,
                session_id=session_id,
                workflow_id=workflow_id,
                limit=limit,
            )

        self._validate_query(
            query
        )

        self.cleanup_expired()

        candidates: List[
            MemoryItem
        ] = []

        with self._lock:

            for item in self._memory.values():

                if (
                    item.status
                    != MemoryStatus.ACTIVE
                ):
                    continue

                if (
                    item.is_expired
                    and not query.include_expired
                ):
                    continue

                if not self._scope_matches(
                    item,
                    query,
                ):
                    continue

                if not self._type_matches(
                    item,
                    query,
                ):
                    continue

                if not self._importance_matches(
                    item,
                    query,
                ):
                    continue

                if not self._tags_match(
                    item,
                    query,
                ):
                    continue

                score = self._calculate_relevance(
                    item,
                    query.query,
                )

                if (
                    score
                    < self.config.relevance_threshold
                ):
                    continue

                item.relevance_score = score

                candidates.append(
                    item
                )

            candidates.sort(
                key=lambda item: (
                    item.relevance_score,
                    _importance_score(
                        item.importance
                    ),
                    item.updated_at,
                ),
                reverse=True,
            )

            selected = candidates[
                : query.limit
            ]

            for item in selected:
                item.touch()

        return MemoryResult(
            success=True,
            memories=selected,
            count=len(selected),
        )

    # ========================================================================
    # Context Building
    # ========================================================================

    def build_context(
        self,
        query: Optional[str] = None,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        limit: int = 10,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Build a bounded memory context suitable for an agent prompt.
        """

        result = self.retrieve(
            text=query,
            company_id=company_id,
            user_id=user_id,
            session_id=session_id,
            workflow_id=workflow_id,
            limit=limit,
        )

        memories = result.memories

        if not memories:
            return ""

        max_length = (
            max_length
            or self.config.max_context_length
        )

        sections: List[str] = []

        for memory in memories:

            section = (
                f"[{memory.memory_type.value}] "
                f"{memory.content}"
            )

            if memory.tags:

                section += (
                    "\nTags: "
                    + ", ".join(memory.tags)
                )

            sections.append(
                section
            )

        context = "\n\n".join(
            sections
        )

        if len(context) > max_length:

            context = context[
                :max_length
            ]

            context += (
                "\n[Memory context truncated]"
            )

        return context

    # ========================================================================
    # Short-Term Memory
    # ========================================================================

    def get_recent(
        self,
        session_id: str,
        company_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        """Return recent session memories."""

        limit = (
            limit
            or self.config.max_short_term_memories
        )

        result = self.retrieve(
            query=MemoryQuery(
                session_id=session_id,
                company_id=company_id,
                limit=limit,
            )
        )

        return sorted(
            result.memories,
            key=lambda item: item.created_at,
        )

    def get_conversation_history(
        self,
        session_id: str,
        company_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[MemoryItem]:
        """Return recent conversation turns."""

        result = self.retrieve(
            query=MemoryQuery(
                session_id=session_id,
                company_id=company_id,
                memory_types=[
                    MemoryType.CONVERSATION
                ],
                limit=limit,
            )
        )

        return sorted(
            result.memories,
            key=lambda item: item.created_at,
        )

    # ========================================================================
    # Summary
    # ========================================================================

    def create_summary(
        self,
        session_id: str,
        company_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        limit: int = 20,
    ) -> Optional[MemoryItem]:
        """
        Create a deterministic summary from recent memories.

        This deliberately does not call an LLM. An LLM summarizer
        can be injected by the orchestration layer if required.
        """

        memories = self.get_recent(
            session_id=session_id,
            company_id=company_id,
            limit=limit,
        )

        if not memories:
            return None

        lines: List[str] = []

        for memory in memories:

            lines.append(
                f"- "
                f"{memory.memory_type.value}: "
                f"{memory.content}"
            )

        summary = "\n".join(
            lines
        )

        if len(summary) > (
            self.config.summary_max_length
        ):

            summary = summary[
                : self.config.summary_max_length
            ]

        return self.add(
            content=summary,
            memory_type=MemoryType.SUMMARY,
            scope=MemoryScope.WORKFLOW,
            source=MemorySource.WORKFLOW,
            importance=MemoryImportance.HIGH,
            company_id=company_id,
            session_id=session_id,
            workflow_id=workflow_id,
        )

    # ========================================================================
    # Get / Update / Delete
    # ========================================================================

    def get(
        self,
        memory_id: str,
    ) -> MemoryItem:

        with self._lock:

            item = self._memory.get(
                memory_id
            )

            if item is None:

                if self.repository is not None:

                    persisted = self._repository_get(
                        memory_id
                    )

                    if persisted is not None:
                        return persisted

                raise MemoryNotFoundError(
                    f"Memory not found: {memory_id}"
                )

            if item.is_expired:

                item.status = (
                    MemoryStatus.EXPIRED
                )

                raise MemoryNotFoundError(
                    f"Memory expired: {memory_id}"
                )

            item.touch()

            return item

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        tags: Optional[
            Sequence[str]
        ] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
        importance: Optional[
            MemoryImportance
        ] = None,
    ) -> MemoryItem:

        with self._lock:

            item = self.get(
                memory_id
            )

            old_fingerprint = (
                item.fingerprint
            )

            if content is not None:

                content = self._validate_content(
                    content
                )

                content = self.sanitize_content(
                    content
                )

                item.content = content

                item.fingerprint = (
                    calculate_memory_fingerprint(
                        content
                    )
                )

            if tags is not None:

                item.tags = [
                    str(tag).strip()
                    for tag in tags
                    if str(tag).strip()
                ]

            if metadata is not None:

                item.metadata = dict(
                    metadata
                )

            if importance is not None:

                item.importance = importance

            item.updated_at = datetime.now(
                timezone.utc
            )

            if old_fingerprint:

                self._fingerprints.pop(
                    old_fingerprint,
                    None,
                )

            if item.fingerprint:

                self._fingerprints[
                    item.fingerprint
                ] = item.memory_id

            self._persist(
                "update",
                item,
            )

            self._audit(
                "memory_updated",
                item,
            )

            return item

    def delete(
        self,
        memory_id: str,
    ) -> bool:

        with self._lock:

            item = self._memory.get(
                memory_id
            )

            if item is None:
                return False

            item.status = (
                MemoryStatus.DELETED
            )

            self._memory.pop(
                memory_id,
                None,
            )

            if item.fingerprint:

                self._fingerprints.pop(
                    item.fingerprint,
                    None,
                )

            self._persist(
                "delete",
                item,
            )

            self._audit(
                "memory_deleted",
                item,
            )

            return True

    def clear_session(
        self,
        session_id: str,
        company_id: Optional[str] = None,
    ) -> int:
        """Delete all memories belonging to a session."""

        ids: List[str] = []

        with self._lock:

            for memory_id, item in (
                self._memory.items()
            ):

                if (
                    item.session_id
                    != session_id
                ):
                    continue

                if (
                    company_id
                    and item.company_id
                    != company_id
                ):
                    continue

                ids.append(
                    memory_id
                )

            for memory_id in ids:

                self.delete(
                    memory_id
                )

        return len(ids)

    def clear_workflow(
        self,
        workflow_id: str,
        company_id: Optional[str] = None,
    ) -> int:
        """Delete workflow memories."""

        ids: List[str] = []

        with self._lock:

            for memory_id, item in (
                self._memory.items()
            ):

                if (
                    item.workflow_id
                    != workflow_id
                ):
                    continue

                if (
                    company_id
                    and item.company_id
                    != company_id
                ):
                    continue

                ids.append(
                    memory_id
                )

            for memory_id in ids:

                self.delete(
                    memory_id
                )

        return len(ids)

    # ========================================================================
    # Expiration
    # ========================================================================

    def cleanup_expired(self) -> int:
        """Remove expired memories."""

        expired_ids: List[str] = []

        now = datetime.now(
            timezone.utc
        )

        with self._lock:

            for memory_id, item in (
                self._memory.items()
            ):

                if (
                    item.expires_at
                    and _ensure_utc(
                        item.expires_at
                    )
                    <= now
                ):

                    item.status = (
                        MemoryStatus.EXPIRED
                    )

                    expired_ids.append(
                        memory_id
                    )

            for memory_id in expired_ids:

                item = self._memory.pop(
                    memory_id
                )

                if item.fingerprint:

                    self._fingerprints.pop(
                        item.fingerprint,
                        None,
                    )

        return len(expired_ids)

    # ========================================================================
    # Statistics
    # ========================================================================

    def statistics(self) -> Dict[str, Any]:
        """Return memory statistics."""

        with self._lock:

            by_type: Dict[
                str,
                int,
            ] = {}

            by_scope: Dict[
                str,
                int,
            ] = {}

            by_source: Dict[
                str,
                int,
            ] = {}

            for item in self._memory.values():

                by_type[
                    item.memory_type.value
                ] = (
                    by_type.get(
                        item.memory_type.value,
                        0,
                    )
                    + 1
                )

                by_scope[
                    item.scope.value
                ] = (
                    by_scope.get(
                        item.scope.value,
                        0,
                    )
                    + 1
                )

                by_source[
                    item.source.value
                ] = (
                    by_source.get(
                        item.source.value,
                        0,
                    )
                    + 1
                )

            return {
                "enabled": self.config.enabled,
                "total_memories": len(
                    self._memory
                ),
                "max_memories": (
                    self.config.max_memories
                ),
                "by_type": by_type,
                "by_scope": by_scope,
                "by_source": by_source,
            }

    # ========================================================================
    # Sanitization
    # ========================================================================

    def sanitize_content(
        self,
        content: str,
    ) -> str:
        """
        Sanitize memory content.

        Sensitive key/value pairs are redacted.

        This does not attempt to understand every form of
        sensitive financial data. Domain-level controls remain
        responsible for PII/financial-data handling.
        """

        if not self.config.sanitize_sensitive_data:
            return content

        sanitized = content

        for pattern in SENSITIVE_FIELD_PATTERNS:

            sanitized = re.sub(
                rf"({pattern}\s*[:=]\s*)([^\s,;]+)",
                rf"\1[REDACTED]",
                sanitized,
                flags=re.IGNORECASE,
            )

        return sanitized

    # ========================================================================
    # Validation
    # ========================================================================

    def _validate_content(
        self,
        content: Any,
    ) -> str:

        if not isinstance(
            content,
            str,
        ):

            raise InvalidMemoryError(
                "Memory content must be a string."
            )

        content = content.strip()

        if not content:

            raise InvalidMemoryError(
                "Memory content cannot be empty."
            )

        if len(content) > (
            self.config.max_memory_length
        ):

            raise InvalidMemoryError(
                "Memory content exceeds the "
                "configured maximum length."
            )

        return content

    def _validate_scope(
        self,
        scope: MemoryScope,
        company_id: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        workflow_id: Optional[str],
    ) -> None:

        if (
            scope == MemoryScope.COMPANY
            and not company_id
        ):

            raise MemoryScopeError(
                "Company memory requires company_id."
            )

        if (
            scope == MemoryScope.USER
            and not user_id
        ):

            raise MemoryScopeError(
                "User memory requires user_id."
            )

        if (
            scope == MemoryScope.SESSION
            and not session_id
        ):

            raise MemoryScopeError(
                "Session memory requires session_id."
            )

        if (
            scope == MemoryScope.WORKFLOW
            and not workflow_id
        ):

            raise MemoryScopeError(
                "Workflow memory requires workflow_id."
            )

        if (
            scope == MemoryScope.COMPANY
            and not self.config.allow_company_memory
        ):

            raise MemoryScopeError(
                "Company memory is disabled."
            )

        if (
            scope == MemoryScope.USER
            and not self.config.allow_user_memory
        ):

            raise MemoryScopeError(
                "User memory is disabled."
            )

        if (
            scope == MemoryScope.WORKFLOW
            and not self.config.allow_workflow_memory
        ):

            raise MemoryScopeError(
                "Workflow memory is disabled."
            )

    def _validate_source(
        self,
        source: MemorySource,
    ) -> None:

        if (
            source == MemorySource.AGENT
            and not self.config.allow_agent_memory
        ):

            raise MemorySecurityError(
                "Agent-created memory is disabled."
            )

        if (
            source == MemorySource.TOOL
            and not self.config.allow_tool_memory
        ):

            raise MemorySecurityError(
                "Tool-created memory is disabled."
            )

    def _validate_query(
        self,
        query: MemoryQuery,
    ) -> None:

        if query.limit < 1:

            raise InvalidMemoryError(
                "Memory query limit must be positive."
            )

        if query.limit > (
            self.config.max_memories
        ):

            raise InvalidMemoryError(
                "Memory query limit exceeds "
                "maximum memory capacity."
            )

    # ========================================================================
    # Filtering
    # ========================================================================

    @staticmethod
    def _scope_matches(
        item: MemoryItem,
        query: MemoryQuery,
    ) -> bool:

        if (
            query.company_id
            and item.company_id
            != query.company_id
        ):
            return False

        if (
            query.user_id
            and item.user_id
            != query.user_id
        ):
            return False

        if (
            query.session_id
            and item.session_id
            != query.session_id
        ):
            return False

        if (
            query.workflow_id
            and item.workflow_id
            != query.workflow_id
        ):
            return False

        return True

    @staticmethod
    def _type_matches(
        item: MemoryItem,
        query: MemoryQuery,
    ) -> bool:

        if not query.memory_types:
            return True

        return (
            item.memory_type
            in query.memory_types
        )

    @staticmethod
    def _importance_matches(
        item: MemoryItem,
        query: MemoryQuery,
    ) -> bool:

        if not query.minimum_importance:
            return True

        return (
            _importance_score(
                item.importance
            )
            >= _importance_score(
                query.minimum_importance
            )
        )

    @staticmethod
    def _tags_match(
        item: MemoryItem,
        query: MemoryQuery,
    ) -> bool:

        if not query.tags:
            return True

        item_tags = {
            tag.lower()
            for tag in item.tags
        }

        return all(
            str(tag).lower()
            in item_tags
            for tag in query.tags
        )

    # ========================================================================
    # Relevance
    # ========================================================================

    @staticmethod
    def _calculate_relevance(
        item: MemoryItem,
        query: Optional[str],
    ) -> float:

        if not query:
            return 1.0

        query_tokens = _tokenize(
            query
        )

        content_tokens = _tokenize(
            item.content
        )

        if not query_tokens:
            return 1.0

        if not content_tokens:
            return 0.0

        overlap = (
            query_tokens
            & content_tokens
        )

        lexical_score = (
            len(overlap)
            / len(query_tokens)
        )

        importance_bonus = {
            MemoryImportance.LOW: 0.00,
            MemoryImportance.MEDIUM: 0.05,
            MemoryImportance.HIGH: 0.10,
            MemoryImportance.CRITICAL: 0.15,
        }[
            item.importance
        ]

        return min(
            1.0,
            lexical_score
            + importance_bonus,
        )

    # ========================================================================
    # Eviction
    # ========================================================================

    def _evict_oldest(self) -> None:

        if not self._memory:
            return

        candidates = sorted(
            self._memory.values(),
            key=lambda item: (
                _importance_score(
                    item.importance
                ),
                item.updated_at,
            ),
        )

        candidate = candidates[0]

        # Do not evict critical memories unless
        # absolutely necessary.
        if (
            candidate.importance
            == MemoryImportance.CRITICAL
        ):

            non_critical = [
                item
                for item
                in candidates
                if item.importance
                != MemoryImportance.CRITICAL
            ]

            if non_critical:
                candidate = non_critical[0]

        self._memory.pop(
            candidate.memory_id,
            None,
        )

        if candidate.fingerprint:

            self._fingerprints.pop(
                candidate.fingerprint,
                None,
            )

    # ========================================================================
    # Repository Integration
    # ========================================================================

    def _persist(
        self,
        operation: str,
        item: MemoryItem,
    ) -> None:

        if self.repository is None:
            return

        payload = item.to_dict()

        methods = {
            "add": (
                "create",
                "add",
                "save",
                "insert",
            ),
            "update": (
                "update",
                "save",
            ),
            "delete": (
                "delete",
                "remove",
            ),
        }

        for method_name in methods.get(
            operation,
            (),
        ):

            method = getattr(
                self.repository,
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
                        item
                    )

                except TypeError:
                    continue

            return

    def _repository_get(
        self,
        memory_id: str,
    ) -> Optional[MemoryItem]:

        if self.repository is None:
            return None

        for method_name in (
            "get",
            "find",
            "get_by_id",
        ):

            method = getattr(
                self.repository,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                result = method(
                    memory_id=memory_id
                )

            except TypeError:

                try:
                    result = method(
                        memory_id
                    )

                except TypeError:
                    continue

            if result is None:
                return None

            if isinstance(
                result,
                MemoryItem,
            ):
                return result

            if isinstance(
                result,
                Mapping,
            ):
                return memory_from_dict(
                    result
                )

        return None

    # ========================================================================
    # Audit
    # ========================================================================

    def _audit(
        self,
        event: str,
        item: MemoryItem,
    ) -> None:

        if self.audit_service is None:
            return

        payload = {
            "event": event,
            "memory_id": item.memory_id,
            "memory_type": (
                item.memory_type.value
            ),
            "scope": item.scope.value,
            "company_id": item.company_id,
            "user_id": item.user_id,
            "session_id": item.session_id,
            "workflow_id": item.workflow_id,
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


# ============================================================================
# Memory ID / Fingerprint Helpers
# ============================================================================


def generate_memory_id() -> str:
    """Generate a unique memory identifier."""

    now = datetime.now(
        timezone.utc
    ).isoformat()

    digest = hashlib.sha256(
        now.encode("utf-8")
    ).hexdigest()

    return f"mem_{digest[:24]}"


def calculate_memory_fingerprint(
    content: str,
) -> str:
    """Generate deterministic content fingerprint."""

    normalized = re.sub(
        r"\s+",
        " ",
        content.strip().lower(),
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================================
# Tokenization
# ============================================================================


def _tokenize(
    text: str,
) -> set[str]:

    return {
        token
        for token in re.findall(
            r"\b[a-zA-Z0-9_]+\b",
            text.lower(),
        )
        if len(token) > 1
    }


# ============================================================================
# Importance Helpers
# ============================================================================


def _importance_score(
    importance: MemoryImportance,
) -> int:

    return {
        MemoryImportance.LOW: 1,
        MemoryImportance.MEDIUM: 2,
        MemoryImportance.HIGH: 3,
        MemoryImportance.CRITICAL: 4,
    }[
        importance
    ]


# ============================================================================
# Datetime Helpers
# ============================================================================


def _ensure_utc(
    value: datetime,
) -> datetime:

    if value.tzinfo is None:

        return value.replace(
            tzinfo=timezone.utc
        )

    return value.astimezone(
        timezone.utc
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
        datetime,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _serialize(item)
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
# Memory Reconstruction
# ============================================================================


def memory_from_dict(
    data: Mapping[str, Any],
) -> MemoryItem:
    """Create MemoryItem from persisted data."""

    return MemoryItem(
        memory_id=str(
            data["memory_id"]
        ),
        content=str(
            data["content"]
        ),
        memory_type=MemoryType(
            data.get(
                "memory_type",
                MemoryType.CONVERSATION.value,
            )
        ),
        scope=MemoryScope(
            data.get(
                "scope",
                MemoryScope.SESSION.value,
            )
        ),
        source=MemorySource(
            data.get(
                "source",
                MemorySource.AGENT.value,
            )
        ),
        importance=MemoryImportance(
            data.get(
                "importance",
                MemoryImportance.MEDIUM.value,
            )
        ),
        company_id=data.get(
            "company_id"
        ),
        user_id=data.get(
            "user_id"
        ),
        session_id=data.get(
            "session_id"
        ),
        workflow_id=data.get(
            "workflow_id"
        ),
        tags=list(
            data.get(
                "tags",
                [],
            )
        ),
        metadata=dict(
            data.get(
                "metadata",
                {},
            )
        ),
        created_at=_parse_datetime(
            data.get(
                "created_at"
            )
        ),
        updated_at=_parse_datetime(
            data.get(
                "updated_at"
            )
        ),
        expires_at=_parse_datetime(
            data.get(
                "expires_at"
            )
        ),
        status=MemoryStatus(
            data.get(
                "status",
                MemoryStatus.ACTIVE.value,
            )
        ),
        access_count=int(
            data.get(
                "access_count",
                0,
            )
        ),
        relevance_score=float(
            data.get(
                "relevance_score",
                0.0,
            )
        ),
        fingerprint=data.get(
            "fingerprint"
        ),
    )


def _parse_datetime(
    value: Any,
) -> datetime:

    if isinstance(
        value,
        datetime,
    ):
        return _ensure_utc(
            value
        )

    if isinstance(
        value,
        str,
    ):

        try:

            return _ensure_utc(
                datetime.fromisoformat(
                    value
                )
            )

        except ValueError:
            pass

    return datetime.now(
        timezone.utc
    )


# ============================================================================
# Convenience Functions
# ============================================================================


def create_memory(
    content: str,
    company_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    memory_type: MemoryType = (
        MemoryType.CONVERSATION
    ),
    scope: MemoryScope = MemoryScope.SESSION,
    importance: MemoryImportance = (
        MemoryImportance.MEDIUM
    ),
) -> MemoryItem:
    """Create a standalone memory item."""

    manager = AgentMemory()

    return manager.add(
        content=content,
        company_id=company_id,
        user_id=user_id,
        session_id=session_id,
        workflow_id=workflow_id,
        memory_type=memory_type,
        scope=scope,
        importance=importance,
    )


def memory_context(
    query: Optional[str] = None,
    company_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Build memory context."""

    manager = AgentMemory()

    return manager.build_context(
        query=query,
        company_id=company_id,
        user_id=user_id,
        session_id=session_id,
        workflow_id=workflow_id,
        limit=limit,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "AgentMemoryError",
    "InvalidMemoryError",
    "MemoryNotFoundError",
    "MemoryCapacityError",
    "MemoryScopeError",
    "MemorySecurityError",
    "MemoryConfigurationError",

    # Enums
    "MemoryType",
    "MemoryScope",
    "MemoryImportance",
    "MemorySource",
    "MemoryStatus",

    # Configuration
    "MemoryConfig",

    # Models
    "MemoryItem",
    "MemoryQuery",
    "MemoryResult",

    # Main service
    "AgentMemory",

    # Helpers
    "generate_memory_id",
    "calculate_memory_fingerprint",
    "memory_from_dict",

    # Convenience
    "create_memory",
    "memory_context",
]