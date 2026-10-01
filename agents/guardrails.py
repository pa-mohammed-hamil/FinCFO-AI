"""
FinCo AI - Agent Guardrails
===========================

Agent-level safety, policy, and execution guardrails.

Purpose
-------
This module protects the Agentic AI layer from:

- Prompt injection
- Unsafe tool calls
- Unauthorized data access
- Cross-company data leakage
- Arbitrary SQL/code execution
- Dangerous financial actions
- Invalid tool arguments
- Excessive execution loops
- Excessive tool calls
- Untrusted instructions inside retrieved documents
- Attempts to override system policies
- Unsafe external actions

Architecture
------------

User
  |
  v
Supervisor Agent
  |
  v
Agent Guardrails
  |
  +-----------------------------+
  |             |               |
  v             v               v
Input       Tool Policy      Output
Validation  Validation       Validation
  |             |               |
  +-------------+---------------+
                |
                v
          Agent Execution
                |
                v
          Tool Registry
                |
                v
          Tool Executor
                |
                v
        Domain Services

IMPORTANT
---------
This module does not perform business calculations.

It does not:
- calculate fraud scores,
- calculate financial ratios,
- execute SQL,
- make financial decisions,
- replace authorization.py/security.py,
- replace core/guardrails.py.

It provides an additional safety boundary around agent execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


# ============================================================================
# Exceptions
# ============================================================================


class AgentGuardrailError(Exception):
    """Base exception for agent guardrail failures."""


class GuardrailViolationError(AgentGuardrailError):
    """Raised when a guardrail policy is violated."""


class PromptInjectionDetectedError(GuardrailViolationError):
    """Raised when prompt injection is detected."""


class UnauthorizedToolError(GuardrailViolationError):
    """Raised when a tool is not permitted."""


class UnsafeToolCallError(GuardrailViolationError):
    """Raised when a tool call is unsafe."""


class SensitiveDataAccessError(GuardrailViolationError):
    """Raised when sensitive data access violates policy."""


class CrossCompanyAccessError(GuardrailViolationError):
    """Raised when a request attempts cross-company access."""


class InvalidAgentInputError(GuardrailViolationError):
    """Raised when agent input is invalid."""


class InvalidAgentOutputError(GuardrailViolationError):
    """Raised when agent output is invalid."""


class AgentExecutionLimitError(GuardrailViolationError):
    """Raised when an execution limit is exceeded."""


class GuardrailConfigurationError(AgentGuardrailError):
    """Raised when guardrails are incorrectly configured."""


# ============================================================================
# Constants
# ============================================================================


DEFAULT_MAX_INPUT_LENGTH = 20_000
DEFAULT_MAX_OUTPUT_LENGTH = 50_000
DEFAULT_MAX_TOOL_CALLS = 25
DEFAULT_MAX_WORKFLOW_STEPS = 50
DEFAULT_MAX_AGENT_DEPTH = 10
DEFAULT_MAX_ARGUMENTS = 100
DEFAULT_MAX_STRING_LENGTH = 10_000

DEFAULT_MIN_CONFIDENCE = 0.0

REDACTED_VALUE = "[REDACTED]"


# ============================================================================
# Enums
# ============================================================================


class GuardrailAction(str, Enum):
    """Action taken when a guardrail is triggered."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REQUIRE_REVIEW = "require_review"
    REDACT = "redact"


class GuardrailSeverity(str, Enum):
    """Guardrail violation severity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class GuardrailCategory(str, Enum):
    """Guardrail violation categories."""

    INPUT = "input"
    PROMPT_INJECTION = "prompt_injection"
    AUTHORIZATION = "authorization"
    TOOL_SECURITY = "tool_security"
    DATA_SECURITY = "data_security"
    CROSS_COMPANY_ACCESS = "cross_company_access"
    FINANCIAL_SAFETY = "financial_safety"
    OUTPUT = "output"
    EXECUTION_LIMIT = "execution_limit"
    PRIVACY = "privacy"
    POLICY = "policy"


class ToolRiskLevel(str, Enum):
    """Risk level assigned to an agent tool."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolActionType(str, Enum):
    """Tool action classification."""

    READ = "read"
    ANALYZE = "analyze"
    CALCULATE = "calculate"
    RETRIEVE = "retrieve"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    EXTERNAL = "external"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class GuardrailConfig:
    """Configuration for agent-level guardrails."""

    enabled: bool = True

    max_input_length: int = DEFAULT_MAX_INPUT_LENGTH
    max_output_length: int = DEFAULT_MAX_OUTPUT_LENGTH

    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_workflow_steps: int = DEFAULT_MAX_WORKFLOW_STEPS
    max_agent_depth: int = DEFAULT_MAX_AGENT_DEPTH

    max_arguments: int = DEFAULT_MAX_ARGUMENTS
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH

    block_prompt_injection: bool = True
    block_unsafe_tools: bool = True
    block_cross_company_access: bool = True

    require_confirmation_for_writes: bool = True
    require_confirmation_for_external_actions: bool = True
    require_confirmation_for_deletes: bool = True

    allow_sql_tools: bool = False
    allow_shell_tools: bool = False
    allow_code_execution: bool = False

    redact_sensitive_data: bool = True

    minimum_confidence: float = DEFAULT_MIN_CONFIDENCE

    allowed_tools: Optional[Set[str]] = None

    blocked_tools: Set[str] = field(
        default_factory=lambda: {
            "execute_shell",
            "run_shell",
            "execute_code",
            "eval",
            "exec",
            "arbitrary_python",
            "drop_database",
            "delete_all",
        }
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate configuration."""

        positive_values = {
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "max_tool_calls": self.max_tool_calls,
            "max_workflow_steps": self.max_workflow_steps,
            "max_agent_depth": self.max_agent_depth,
            "max_arguments": self.max_arguments,
            "max_string_length": self.max_string_length,
        }

        for name, value in positive_values.items():
            if value < 1:
                raise GuardrailConfigurationError(
                    f"{name} must be greater than zero."
                )

        if not (
            0.0
            <= self.minimum_confidence
            <= 1.0
        ):
            raise GuardrailConfigurationError(
                "minimum_confidence must be between 0 and 1."
            )


# ============================================================================
# Policy Models
# ============================================================================


@dataclass
class ToolPolicy:
    """
    Security policy describing how an agent may use a tool.
    """

    name: str

    action_type: ToolActionType = ToolActionType.READ

    risk_level: ToolRiskLevel = ToolRiskLevel.LOW

    allowed: bool = True

    requires_confirmation: bool = False

    requires_authorization: bool = True

    allowed_companies: Optional[Set[str]] = None

    allowed_roles: Optional[Set[str]] = None

    description: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "action_type": self.action_type.value,
            "risk_level": self.risk_level.value,
            "allowed": self.allowed,
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "requires_authorization": (
                self.requires_authorization
            ),
            "allowed_companies": (
                sorted(self.allowed_companies)
                if self.allowed_companies
                else None
            ),
            "allowed_roles": (
                sorted(self.allowed_roles)
                if self.allowed_roles
                else None
            ),
            "description": self.description,
            "metadata": _serialize(self.metadata),
        }


@dataclass
class GuardrailViolation:
    """Structured guardrail violation."""

    category: GuardrailCategory

    severity: GuardrailSeverity

    action: GuardrailAction

    message: str

    rule: Optional[str] = None

    tool_name: Optional[str] = None

    company_id: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "action": self.action.value,
            "message": self.message,
            "rule": self.rule,
            "tool_name": self.tool_name,
            "company_id": self.company_id,
            "metadata": _serialize(self.metadata),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class GuardrailResult:
    """Result of a guardrail evaluation."""

    allowed: bool

    action: GuardrailAction = GuardrailAction.ALLOW

    severity: GuardrailSeverity = GuardrailSeverity.LOW

    violations: List[GuardrailViolation] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    redacted_fields: List[str] = field(
        default_factory=list
    )

    sanitized_input: Any = None

    sanitized_output: Any = None

    requires_confirmation: bool = False

    requires_human_review: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action.value,
            "severity": self.severity.value,
            "violations": [
                violation.to_dict()
                for violation in self.violations
            ],
            "warnings": self.warnings,
            "redacted_fields": self.redacted_fields,
            "sanitized_input": _serialize(
                self.sanitized_input
            ),
            "sanitized_output": _serialize(
                self.sanitized_output
            ),
            "requires_confirmation": (
                self.requires_confirmation
            ),
            "requires_human_review": (
                self.requires_human_review
            ),
            "metadata": _serialize(self.metadata),
        }


# ============================================================================
# Default Tool Policies
# ============================================================================


DEFAULT_TOOL_POLICIES: Dict[str, ToolPolicy] = {
    # Financial
    "financial_tools": ToolPolicy(
        name="financial_tools",
        action_type=ToolActionType.ANALYZE,
        risk_level=ToolRiskLevel.MEDIUM,
    ),

    # Fraud
    "fraud_tools": ToolPolicy(
        name="fraud_tools",
        action_type=ToolActionType.ANALYZE,
        risk_level=ToolRiskLevel.HIGH,
    ),

    # Forecast
    "forecast_tools": ToolPolicy(
        name="forecast_tools",
        action_type=ToolActionType.ANALYZE,
        risk_level=ToolRiskLevel.MEDIUM,
    ),

    # RAG
    "rag_tools": ToolPolicy(
        name="rag_tools",
        action_type=ToolActionType.RETRIEVE,
        risk_level=ToolRiskLevel.LOW,
    ),

    # What-if
    "what_if_tools": ToolPolicy(
        name="what_if_tools",
        action_type=ToolActionType.CALCULATE,
        risk_level=ToolRiskLevel.MEDIUM,
    ),

    # Reports
    "report_tools": ToolPolicy(
        name="report_tools",
        action_type=ToolActionType.READ,
        risk_level=ToolRiskLevel.LOW,
    ),

    # Database
    "database_tools": ToolPolicy(
        name="database_tools",
        action_type=ToolActionType.READ,
        risk_level=ToolRiskLevel.HIGH,
        description=(
            "Database access must remain read-only "
            "for agent workflows."
        ),
    ),

    # Alerts
    "alert_tools": ToolPolicy(
        name="alert_tools",
        action_type=ToolActionType.WRITE,
        risk_level=ToolRiskLevel.HIGH,
        requires_confirmation=True,
    ),

    # Calculator
    "calculator_tools": ToolPolicy(
        name="calculator_tools",
        action_type=ToolActionType.CALCULATE,
        risk_level=ToolRiskLevel.LOW,
    ),
}


# ============================================================================
# Prompt Injection Patterns
# ============================================================================


PROMPT_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(the\s+)?system\s+prompt",
    r"disregard\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"override\s+(the\s+)?system",
    r"override\s+(all\s+)?security",
    r"bypass\s+(all\s+)?security",
    r"bypass\s+(the\s+)?guardrails",
    r"disable\s+(the\s+)?guardrails",
    r"turn\s+off\s+(the\s+)?guardrails",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?the\s+system\s+prompt",
    r"print\s+(the\s+)?system\s+instructions",
    r"reveal\s+(your\s+)?hidden\s+instructions",
    r"developer\s+message",
    r"system\s+message",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"\bdan\b",
    r"act\s+as\s+an?\s+unrestricted",
    r"pretend\s+you\s+have\s+no\s+restrictions",
    r"you\s+are\s+now\s+unrestricted",
    r"bypass\s+authorization",
    r"bypass\s+permission",
    r"execute\s+arbitrary\s+code",
    r"run\s+arbitrary\s+python",
    r"execute\s+arbitrary\s+sql",
)


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
    r"bank[_-]?account",
    r"account[_-]?number",
)


DANGEROUS_OPERATION_PATTERNS = (
    r"\bdrop\s+table\b",
    r"\bdrop\s+database\b",
    r"\bdelete\s+from\b",
    r"\btruncate\s+table\b",
    r"\bshutdown\b",
    r"\brm\s+-rf\b",
    r"\bformat\s+disk\b",
    r"\bexecute\s+shell\b",
    r"\bsubprocess\b",
    r"\bos\.system\b",
    r"\beval\s*\(",
    r"\bexec\s*\(",
)


# ============================================================================
# Agent Guardrails
# ============================================================================


class AgentGuardrails:
    """
    Central policy enforcement layer for FinCo AI agents.

    Example
    -------

        guardrails = AgentGuardrails(
            config=GuardrailConfig()
        )

        result = guardrails.validate_input(
            text=user_message,
            company_id="COMP-001",
            user_id="USER-001",
        )

        guardrails.validate_tool_call(
            tool_name="fraud_tools",
            arguments={
                "company_id": "COMP-001",
                "transaction_id": "TX-001",
            },
            company_id="COMP-001",
        )
    """

    def __init__(
        self,
        config: Optional[GuardrailConfig] = None,
        tool_policies: Optional[
            Mapping[str, ToolPolicy]
        ] = None,
        authorization_service: Any = None,
        audit_service: Any = None,
    ) -> None:

        self.config = config or GuardrailConfig()
        self.config.validate()

        self.tool_policies = dict(
            tool_policies
            or DEFAULT_TOOL_POLICIES
        )

        self.authorization_service = (
            authorization_service
        )

        self.audit_service = audit_service

    # ========================================================================
    # Input Validation
    # ========================================================================

    def validate_input(
        self,
        text: Any,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> GuardrailResult:
        """
        Validate user/agent input.

        Detects:
        - malformed input,
        - oversized prompts,
        - prompt injection,
        - dangerous commands,
        - sensitive-data requests.
        """

        violations: List[
            GuardrailViolation
        ] = []

        warnings: List[str] = []

        if not isinstance(text, str):
            violations.append(
                GuardrailViolation(
                    category=GuardrailCategory.INPUT,
                    severity=GuardrailSeverity.HIGH,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Agent input must be a string."
                    ),
                    rule="input_type",
                    company_id=company_id,
                )
            )

            return self._result(
                violations=violations
            )

        text = text.strip()

        if not text:
            violations.append(
                GuardrailViolation(
                    category=GuardrailCategory.INPUT,
                    severity=GuardrailSeverity.MEDIUM,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Agent input cannot be empty."
                    ),
                    rule="empty_input",
                    company_id=company_id,
                )
            )

        if len(text) > self.config.max_input_length:
            violations.append(
                GuardrailViolation(
                    category=GuardrailCategory.INPUT,
                    severity=GuardrailSeverity.HIGH,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Agent input exceeds the "
                        "maximum permitted length."
                    ),
                    rule="max_input_length",
                    company_id=company_id,
                    metadata={
                        "length": len(text),
                        "maximum": (
                            self.config.max_input_length
                        ),
                    },
                )
            )

        if self.config.block_prompt_injection:

            injection_matches = (
                self.detect_prompt_injection(
                    text
                )
            )

            if injection_matches:

                violations.append(
                    GuardrailViolation(
                        category=(
                            GuardrailCategory.PROMPT_INJECTION
                        ),
                        severity=(
                            GuardrailSeverity.CRITICAL
                        ),
                        action=GuardrailAction.BLOCK,
                        message=(
                            "Potential prompt injection "
                            "detected."
                        ),
                        rule="prompt_injection",
                        company_id=company_id,
                        metadata={
                            "patterns": (
                                injection_matches
                            ),
                        },
                    )
                )

        dangerous_matches = (
            self.detect_dangerous_operations(
                text
            )
        )

        if dangerous_matches:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.POLICY
                    ),
                    severity=(
                        GuardrailSeverity.CRITICAL
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Potentially dangerous "
                        "operation detected."
                    ),
                    rule="dangerous_operation",
                    company_id=company_id,
                    metadata={
                        "patterns": dangerous_matches,
                    },
                )
            )

        return self._result(
            violations=violations,
            warnings=warnings,
            sanitized_input=text,
        )

    # ========================================================================
    # Prompt Injection Detection
    # ========================================================================

    def detect_prompt_injection(
        self,
        text: str,
    ) -> List[str]:
        """
        Detect common prompt-injection patterns.

        This is intentionally conservative and should be combined
        with model/provider-level safety controls.
        """

        if not isinstance(text, str):
            return []

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower(),
        ).strip()

        matches: List[str] = []

        for pattern in PROMPT_INJECTION_PATTERNS:

            if re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                matches.append(pattern)

        return matches

    # ========================================================================
    # Dangerous Operation Detection
    # ========================================================================

    def detect_dangerous_operations(
        self,
        text: str,
    ) -> List[str]:

        if not isinstance(text, str):
            return []

        normalized = text.lower()

        matches: List[str] = []

        for pattern in DANGEROUS_OPERATION_PATTERNS:

            if re.search(
                pattern,
                normalized,
                flags=re.IGNORECASE,
            ):
                matches.append(pattern)

        return matches

    # ========================================================================
    # Tool Validation
    # ========================================================================

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: Optional[
            Mapping[str, Any]
        ] = None,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        confirmation: bool = False,
    ) -> GuardrailResult:
        """
        Validate an agent tool invocation.

        This is the main security boundary before ToolExecutor.
        """

        violations: List[
            GuardrailViolation
        ] = []

        arguments = dict(
            arguments or {}
        )

        normalized_tool = (
            str(tool_name).strip().lower()
        )

        if not normalized_tool:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.TOOL_SECURITY
                    ),
                    severity=(
                        GuardrailSeverity.HIGH
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Tool name cannot be empty."
                    ),
                    rule="tool_name",
                    company_id=company_id,
                )
            )

            return self._result(
                violations=violations
            )

        if len(arguments) > self.config.max_arguments:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.TOOL_SECURITY
                    ),
                    severity=(
                        GuardrailSeverity.HIGH
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Too many tool arguments."
                    ),
                    rule="max_arguments",
                    tool_name=normalized_tool,
                    company_id=company_id,
                )
            )

        if normalized_tool in self.config.blocked_tools:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.TOOL_SECURITY
                    ),
                    severity=(
                        GuardrailSeverity.CRITICAL
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Tool is explicitly blocked "
                        "by agent policy."
                    ),
                    rule="blocked_tool",
                    tool_name=normalized_tool,
                    company_id=company_id,
                )
            )

        if (
            self.config.allowed_tools is not None
            and normalized_tool
            not in self.config.allowed_tools
        ):

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.AUTHORIZATION
                    ),
                    severity=(
                        GuardrailSeverity.HIGH
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Tool is not present in the "
                        "agent allowlist."
                    ),
                    rule="tool_allowlist",
                    tool_name=normalized_tool,
                    company_id=company_id,
                )
            )

        policy = self._get_tool_policy(
            normalized_tool
        )

        if policy is None:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.TOOL_SECURITY
                    ),
                    severity=(
                        GuardrailSeverity.HIGH
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "No security policy exists "
                        "for this tool."
                    ),
                    rule="unknown_tool_policy",
                    tool_name=normalized_tool,
                    company_id=company_id,
                )
            )

            return self._result(
                violations=violations
            )

        if not policy.allowed:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.TOOL_SECURITY
                    ),
                    severity=(
                        GuardrailSeverity.CRITICAL
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Tool is disabled by policy."
                    ),
                    rule="tool_disabled",
                    tool_name=normalized_tool,
                    company_id=company_id,
                )
            )

        # SQL/code/shell protections
        self._validate_dangerous_tool_type(
            policy,
            normalized_tool,
            company_id,
            violations,
        )

        # Company isolation
        self._validate_company_scope(
            arguments=arguments,
            company_id=company_id,
            tool_name=normalized_tool,
            violations=violations,
        )

        # Sensitive fields
        self._validate_sensitive_arguments(
            arguments=arguments,
            tool_name=normalized_tool,
            company_id=company_id,
            violations=violations,
        )

        # Role restrictions
        self._validate_role(
            policy=policy,
            role=role,
            tool_name=normalized_tool,
            company_id=company_id,
            violations=violations,
        )

        # Confirmation
        requires_confirmation = (
            self._requires_confirmation(
                policy
            )
        )

        if (
            requires_confirmation
            and not confirmation
        ):

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.FINANCIAL_SAFETY
                    ),
                    severity=(
                        GuardrailSeverity.HIGH
                    ),
                    action=(
                        GuardrailAction.REQUIRE_REVIEW
                    ),
                    message=(
                        "Explicit confirmation is "
                        "required before this tool "
                        "can execute."
                    ),
                    rule="confirmation_required",
                    tool_name=normalized_tool,
                    company_id=company_id,
                )
            )

        # External authorization service
        if (
            policy.requires_authorization
            and self.authorization_service is not None
        ):
            self._validate_external_authorization(
                tool_name=normalized_tool,
                company_id=company_id,
                user_id=user_id,
                role=role,
                arguments=arguments,
                violations=violations,
            )

        return self._result(
            violations=violations,
            requires_confirmation=(
                requires_confirmation
                and not confirmation
            ),
        )

    # ========================================================================
    # Tool Policy Registration
    # ========================================================================

    def register_tool_policy(
        self,
        policy: ToolPolicy,
    ) -> None:
        """Register or replace a tool policy."""

        if not policy.name.strip():
            raise GuardrailConfigurationError(
                "Tool policy name cannot be empty."
            )

        self.tool_policies[
            policy.name.strip().lower()
        ] = policy

    def remove_tool_policy(
        self,
        tool_name: str,
    ) -> None:
        """Remove a tool policy."""

        self.tool_policies.pop(
            tool_name.strip().lower(),
            None,
        )

    def get_tool_policy(
        self,
        tool_name: str,
    ) -> Optional[ToolPolicy]:
        """Return a tool policy."""

        return self._get_tool_policy(
            tool_name
        )

    def _get_tool_policy(
        self,
        tool_name: str,
    ) -> Optional[ToolPolicy]:

        normalized = (
            tool_name.strip().lower()
        )

        if normalized in self.tool_policies:
            return self.tool_policies[
                normalized
            ]

        # Support fully-qualified tool names such as:
        # fraud_tools.detect_fraud
        base_name = normalized.split(
            ".",
            1,
        )[0]

        return self.tool_policies.get(
            base_name
        )

    # ========================================================================
    # Company Isolation
    # ========================================================================

    def validate_company_access(
        self,
        requested_company_id: Optional[str],
        authorized_company_id: Optional[str],
    ) -> bool:
        """
        Prevent cross-company data access.

        Agents should operate within the authorized company context.
        """

        if not requested_company_id:
            return True

        if not authorized_company_id:
            return False

        return (
            str(requested_company_id).strip()
            == str(authorized_company_id).strip()
        )

    def _validate_company_scope(
        self,
        arguments: Mapping[str, Any],
        company_id: Optional[str],
        tool_name: str,
        violations: List[
            GuardrailViolation
        ],
    ) -> None:

        if not self.config.block_cross_company_access:
            return

        if not company_id:
            return

        requested_ids = (
            self._extract_company_ids(
                arguments
            )
        )

        for requested_id in requested_ids:

            if not self.validate_company_access(
                requested_id,
                company_id,
            ):

                violations.append(
                    GuardrailViolation(
                        category=(
                            GuardrailCategory
                            .CROSS_COMPANY_ACCESS
                        ),
                        severity=(
                            GuardrailSeverity.CRITICAL
                        ),
                        action=GuardrailAction.BLOCK,
                        message=(
                            "Cross-company data access "
                            "was blocked."
                        ),
                        rule="company_isolation",
                        tool_name=tool_name,
                        company_id=company_id,
                        metadata={
                            "requested_company_id": (
                                requested_id
                            ),
                        },
                    )
                )

    @staticmethod
    def _extract_company_ids(
        arguments: Mapping[str, Any],
    ) -> Set[str]:

        ids: Set[str] = set()

        for key, value in arguments.items():

            normalized_key = (
                str(key).lower()
            )

            if (
                normalized_key
                in {
                    "company_id",
                    "companyid",
                    "organization_id",
                    "organizationid",
                }
            ):

                if isinstance(
                    value,
                    str,
                ):
                    ids.add(
                        value.strip()
                    )

            elif isinstance(
                value,
                Mapping,
            ):

                ids.update(
                    AgentGuardrails
                    ._extract_company_ids(
                        value
                    )
                )

            elif isinstance(
                value,
                Sequence,
            ) and not isinstance(
                value,
                (str, bytes),
            ):

                for item in value:

                    if isinstance(
                        item,
                        Mapping,
                    ):
                        ids.update(
                            AgentGuardrails
                            ._extract_company_ids(
                                item
                            )
                        )

        return {
            value
            for value in ids
            if value
        }

    # ========================================================================
    # Sensitive Data
    # ========================================================================

    def redact_sensitive_data(
        self,
        data: Any,
    ) -> Any:
        """
        Recursively redact sensitive fields.

        This should be applied before exposing structured
        data to the LLM where appropriate.
        """

        if not self.config.redact_sensitive_data:
            return data

        return self._redact_recursive(
            data
        )

    def _redact_recursive(
        self,
        data: Any,
    ) -> Any:

        if isinstance(
            data,
            Mapping,
        ):

            result = {}

            for key, value in data.items():

                key_text = str(key).lower()

                if self._is_sensitive_field(
                    key_text
                ):
                    result[key] = REDACTED_VALUE
                else:
                    result[key] = (
                        self._redact_recursive(
                            value
                        )
                    )

            return result

        if isinstance(
            data,
            Sequence,
        ) and not isinstance(
            data,
            (str, bytes, bytearray),
        ):

            return [
                self._redact_recursive(
                    item
                )
                for item in data
            ]

        return data

    def _validate_sensitive_arguments(
        self,
        arguments: Mapping[str, Any],
        tool_name: str,
        company_id: Optional[str],
        violations: List[
            GuardrailViolation
        ],
    ) -> None:

        sensitive_fields = self.find_sensitive_fields(
            arguments
        )

        if not sensitive_fields:
            return

        violations.append(
            GuardrailViolation(
                category=GuardrailCategory.PRIVACY,
                severity=GuardrailSeverity.HIGH,
                action=GuardrailAction.REDACT,
                message=(
                    "Sensitive fields detected "
                    "in tool arguments."
                ),
                rule="sensitive_data",
                tool_name=tool_name,
                company_id=company_id,
                metadata={
                    "fields": sensitive_fields,
                },
            )
        )

    def find_sensitive_fields(
        self,
        data: Any,
        prefix: str = "",
    ) -> List[str]:

        fields: List[str] = []

        if isinstance(
            data,
            Mapping,
        ):

            for key, value in data.items():

                key_text = str(key)

                path = (
                    f"{prefix}.{key_text}"
                    if prefix
                    else key_text
                )

                if self._is_sensitive_field(
                    key_text.lower()
                ):
                    fields.append(path)

                else:
                    fields.extend(
                        self.find_sensitive_fields(
                            value,
                            path,
                        )
                    )

        elif isinstance(
            data,
            Sequence,
        ) and not isinstance(
            data,
            (str, bytes),
        ):

            for index, item in enumerate(
                data
            ):

                fields.extend(
                    self.find_sensitive_fields(
                        item,
                        f"{prefix}[{index}]",
                    )
                )

        return fields

    @staticmethod
    def _is_sensitive_field(
        field_name: str,
    ) -> bool:

        return any(
            re.search(
                pattern,
                field_name,
                flags=re.IGNORECASE,
            )
            for pattern in SENSITIVE_FIELD_PATTERNS
        )

    # ========================================================================
    # Dangerous Tool Types
    # ========================================================================

    def _validate_dangerous_tool_type(
        self,
        policy: ToolPolicy,
        tool_name: str,
        company_id: Optional[str],
        violations: List[
            GuardrailViolation
        ],
    ) -> None:

        if (
            policy.action_type
            == ToolActionType.EXECUTE
        ):

            if not self.config.allow_code_execution:

                violations.append(
                    GuardrailViolation(
                        category=(
                            GuardrailCategory.TOOL_SECURITY
                        ),
                        severity=(
                            GuardrailSeverity.CRITICAL
                        ),
                        action=GuardrailAction.BLOCK,
                        message=(
                            "Arbitrary code execution "
                            "is disabled."
                        ),
                        rule="code_execution_disabled",
                        tool_name=tool_name,
                        company_id=company_id,
                    )
                )

        if (
            "sql"
            in tool_name.lower()
            and not self.config.allow_sql_tools
        ):

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.TOOL_SECURITY
                    ),
                    severity=(
                        GuardrailSeverity.CRITICAL
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Direct SQL execution by "
                        "agents is disabled."
                    ),
                    rule="sql_execution_disabled",
                    tool_name=tool_name,
                    company_id=company_id,
                )
            )

        if (
            "shell"
            in tool_name.lower()
            and not self.config.allow_shell_tools
        ):

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.TOOL_SECURITY
                    ),
                    severity=(
                        GuardrailSeverity.CRITICAL
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Shell execution by "
                        "agents is disabled."
                    ),
                    rule="shell_execution_disabled",
                    tool_name=tool_name,
                    company_id=company_id,
                )
            )

    # ========================================================================
    # Role Validation
    # ========================================================================

    @staticmethod
    def _validate_role(
        policy: ToolPolicy,
        role: Optional[str],
        tool_name: str,
        company_id: Optional[str],
        violations: List[
            GuardrailViolation
        ],
    ) -> None:

        if not policy.allowed_roles:
            return

        if not role:
            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.AUTHORIZATION
                    ),
                    severity=GuardrailSeverity.HIGH,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "A role is required for "
                        "this tool."
                    ),
                    rule="role_required",
                    tool_name=tool_name,
                    company_id=company_id,
                )
            )
            return

        if role not in policy.allowed_roles:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.AUTHORIZATION
                    ),
                    severity=GuardrailSeverity.HIGH,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "User role is not authorized "
                        "for this tool."
                    ),
                    rule="role_not_allowed",
                    tool_name=tool_name,
                    company_id=company_id,
                    metadata={
                        "role": role,
                    },
                )
            )

    # ========================================================================
    # Confirmation
    # ========================================================================

    def _requires_confirmation(
        self,
        policy: ToolPolicy,
    ) -> bool:

        if policy.requires_confirmation:
            return True

        if (
            policy.action_type
            == ToolActionType.WRITE
            and self.config.require_confirmation_for_writes
        ):
            return True

        if (
            policy.action_type
            == ToolActionType.DELETE
            and self.config.require_confirmation_for_deletes
        ):
            return True

        if (
            policy.action_type
            == ToolActionType.EXTERNAL
            and self.config
            .require_confirmation_for_external_actions
        ):
            return True

        return False

    # ========================================================================
    # External Authorization
    # ========================================================================

    def _validate_external_authorization(
        self,
        tool_name: str,
        company_id: Optional[str],
        user_id: Optional[str],
        role: Optional[str],
        arguments: Mapping[str, Any],
        violations: List[
            GuardrailViolation
        ],
    ) -> None:

        try:

            result = self._invoke(
                self.authorization_service,
                (
                    "authorize_tool",
                    "authorize",
                    "check_permission",
                    "can_access",
                    "has_permission",
                ),
                {
                    "user_id": user_id,
                    "company_id": company_id,
                    "role": role,
                    "tool_name": tool_name,
                    "arguments": dict(arguments),
                },
            )

            allowed = True

            if result is False:
                allowed = False

            elif isinstance(
                result,
                Mapping,
            ):

                allowed = bool(
                    result.get(
                        "allowed",
                        result.get(
                            "authorized",
                            result.get(
                                "success",
                                True,
                            ),
                        ),
                    )
                )

            if not allowed:

                violations.append(
                    GuardrailViolation(
                        category=(
                            GuardrailCategory
                            .AUTHORIZATION
                        ),
                        severity=(
                            GuardrailSeverity.CRITICAL
                        ),
                        action=GuardrailAction.BLOCK,
                        message=(
                            "Tool authorization "
                            "was denied."
                        ),
                        rule="external_authorization",
                        tool_name=tool_name,
                        company_id=company_id,
                    )
                )

        except Exception as exc:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory
                        .AUTHORIZATION
                    ),
                    severity=(
                        GuardrailSeverity.CRITICAL
                    ),
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Tool authorization could "
                        "not be verified."
                    ),
                    rule="authorization_failure",
                    tool_name=tool_name,
                    company_id=company_id,
                    metadata={
                        "error": str(exc),
                    },
                )
            )

    # ========================================================================
    # Output Validation
    # ========================================================================

    def validate_output(
        self,
        output: Any,
        company_id: Optional[str] = None,
        expected_company_id: Optional[str] = None,
    ) -> GuardrailResult:
        """
        Validate agent output before returning it to the user.

        Protects against:
        - oversized responses,
        - cross-company information,
        - sensitive-data leakage,
        - accidental tool/system instruction leakage.
        """

        violations: List[
            GuardrailViolation
        ] = []

        sanitized_output = output

        serialized = _serialize(
            output
        )

        text = (
            serialized
            if isinstance(
                serialized,
                str,
            )
            else str(serialized)
        )

        if len(text) > self.config.max_output_length:

            violations.append(
                GuardrailViolation(
                    category=GuardrailCategory.OUTPUT,
                    severity=GuardrailSeverity.HIGH,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Agent output exceeds the "
                        "maximum permitted length."
                    ),
                    rule="max_output_length",
                    company_id=company_id,
                )
            )

        sensitive_fields = (
            self.find_sensitive_fields(
                serialized
            )
        )

        if sensitive_fields:

            if self.config.redact_sensitive_data:

                sanitized_output = (
                    self.redact_sensitive_data(
                        serialized
                    )
                )

            else:

                violations.append(
                    GuardrailViolation(
                        category=(
                            GuardrailCategory.PRIVACY
                        ),
                        severity=(
                            GuardrailSeverity.CRITICAL
                        ),
                        action=GuardrailAction.BLOCK,
                        message=(
                            "Sensitive data was "
                            "detected in output."
                        ),
                        rule="sensitive_output",
                        company_id=company_id,
                        metadata={
                            "fields": (
                                sensitive_fields
                            ),
                        },
                    )
                )

        if (
            expected_company_id
            and self.config.block_cross_company_access
        ):

            output_company_ids = (
                self._extract_company_ids(
                    serialized
                    if isinstance(
                        serialized,
                        Mapping,
                    )
                    else {}
                )
            )

            for output_company_id in (
                output_company_ids
            ):

                if not self.validate_company_access(
                    output_company_id,
                    expected_company_id,
                ):

                    violations.append(
                        GuardrailViolation(
                            category=(
                                GuardrailCategory
                                .CROSS_COMPANY_ACCESS
                            ),
                            severity=(
                                GuardrailSeverity.CRITICAL
                            ),
                            action=(
                                GuardrailAction.BLOCK
                            ),
                            message=(
                                "Output contains data "
                                "from an unauthorized "
                                "company."
                            ),
                            rule="output_company_isolation",
                            company_id=company_id,
                            metadata={
                                "output_company_id": (
                                    output_company_id
                                ),
                            },
                        )
                    )

        return self._result(
            violations=violations,
            sanitized_output=sanitized_output,
        )

    # ========================================================================
    # Agent Execution Limits
    # ========================================================================

    def validate_execution_state(
        self,
        tool_calls: int,
        workflow_steps: int,
        agent_depth: int,
    ) -> GuardrailResult:
        """Prevent runaway agent execution."""

        violations: List[
            GuardrailViolation
        ] = []

        if tool_calls > self.config.max_tool_calls:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.EXECUTION_LIMIT
                    ),
                    severity=GuardrailSeverity.HIGH,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Maximum tool-call limit "
                        "has been exceeded."
                    ),
                    rule="max_tool_calls",
                    metadata={
                        "tool_calls": tool_calls,
                        "maximum": (
                            self.config.max_tool_calls
                        ),
                    },
                )
            )

        if (
            workflow_steps
            > self.config.max_workflow_steps
        ):

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.EXECUTION_LIMIT
                    ),
                    severity=GuardrailSeverity.HIGH,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Maximum workflow-step "
                        "limit has been exceeded."
                    ),
                    rule="max_workflow_steps",
                    metadata={
                        "workflow_steps": (
                            workflow_steps
                        ),
                        "maximum": (
                            self.config.max_workflow_steps
                        ),
                    },
                )
            )

        if agent_depth > self.config.max_agent_depth:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.EXECUTION_LIMIT
                    ),
                    severity=GuardrailSeverity.CRITICAL,
                    action=GuardrailAction.BLOCK,
                    message=(
                        "Maximum agent recursion "
                        "depth has been exceeded."
                    ),
                    rule="max_agent_depth",
                    metadata={
                        "agent_depth": agent_depth,
                        "maximum": (
                            self.config.max_agent_depth
                        ),
                    },
                )
            )

        return self._result(
            violations=violations
        )

    # ========================================================================
    # Retrieved Content Validation
    # ========================================================================

    def validate_retrieved_content(
        self,
        content: Any,
    ) -> GuardrailResult:
        """
        Validate RAG content before it is supplied to an agent.

        Retrieved documents are treated as untrusted data,
        not as executable instructions.
        """

        violations: List[
            GuardrailViolation
        ] = []

        serialized = _serialize(
            content
        )

        text = (
            serialized
            if isinstance(
                serialized,
                str,
            )
            else str(serialized)
        )

        injection_matches = (
            self.detect_prompt_injection(
                text
            )
        )

        if injection_matches:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.PROMPT_INJECTION
                    ),
                    severity=(
                        GuardrailSeverity.HIGH
                    ),
                    action=GuardrailAction.WARN,
                    message=(
                        "Retrieved content contains "
                        "instruction-like text. "
                        "Treat it as untrusted evidence."
                    ),
                    rule="retrieved_content_injection",
                    metadata={
                        "patterns": injection_matches,
                    },
                )
            )

        return self._result(
            violations=violations
        )

    # ========================================================================
    # Financial Safety
    # ========================================================================

    def validate_financial_action(
        self,
        action: str,
        amount: Optional[float] = None,
        company_id: Optional[str] = None,
        requires_human_review: bool = True,
    ) -> GuardrailResult:
        """
        Validate potentially consequential financial actions.

        The agent cannot independently execute irreversible
        financial actions.
        """

        violations: List[
            GuardrailViolation
        ] = []

        normalized_action = (
            str(action).strip().lower()
        )

        irreversible_actions = {
            "transfer",
            "payment",
            "refund",
            "approve_payment",
            "reject_payment",
            "close_account",
            "write_off",
            "delete_transaction",
            "modify_transaction",
        }

        if (
            normalized_action
            in irreversible_actions
        ):

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory.FINANCIAL_SAFETY
                    ),
                    severity=(
                        GuardrailSeverity.CRITICAL
                    ),
                    action=(
                        GuardrailAction.REQUIRE_REVIEW
                    ),
                    message=(
                        "Irreversible financial actions "
                        "require human approval."
                    ),
                    rule="irreversible_financial_action",
                    company_id=company_id,
                    metadata={
                        "action": normalized_action,
                        "amount": amount,
                    },
                )
            )

        if amount is not None:

            numeric_amount = _safe_float(
                amount
            )

            if numeric_amount is None:

                violations.append(
                    GuardrailViolation(
                        category=(
                            GuardrailCategory
                            .FINANCIAL_SAFETY
                        ),
                        severity=(
                            GuardrailSeverity.HIGH
                        ),
                        action=(
                            GuardrailAction.BLOCK
                        ),
                        message=(
                            "Financial amount must "
                            "be numeric."
                        ),
                        rule="financial_amount",
                        company_id=company_id,
                    )
                )

            elif numeric_amount < 0:

                violations.append(
                    GuardrailViolation(
                        category=(
                            GuardrailCategory
                            .FINANCIAL_SAFETY
                        ),
                        severity=(
                            GuardrailSeverity.HIGH
                        ),
                        action=(
                            GuardrailAction.BLOCK
                        ),
                        message=(
                            "Negative financial amounts "
                            "are not permitted."
                        ),
                        rule="negative_amount",
                        company_id=company_id,
                    )
                )

        if requires_human_review:

            violations.append(
                GuardrailViolation(
                    category=(
                        GuardrailCategory
                        .FINANCIAL_SAFETY
                    ),
                    severity=(
                        GuardrailSeverity.HIGH
                    ),
                    action=(
                        GuardrailAction.REQUIRE_REVIEW
                    ),
                    message=(
                        "Human review is required "
                        "before consequential financial "
                        "actions."
                    ),
                    rule="human_review",
                    company_id=company_id,
                )
            )

        return self._result(
            violations=violations,
            requires_human_review=(
                requires_human_review
            ),
        )

    # ========================================================================
    # Complete Agent Request Validation
    # ========================================================================

    def validate_agent_request(
        self,
        user_input: Any,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_arguments: Optional[
            Mapping[str, Any]
        ] = None,
        role: Optional[str] = None,
        confirmation: bool = False,
    ) -> GuardrailResult:
        """
        Run the complete pre-execution guardrail chain.
        """

        input_result = self.validate_input(
            text=user_input,
            company_id=company_id,
            user_id=user_id,
        )

        all_violations = list(
            input_result.violations
        )

        if tool_name:

            tool_result = (
                self.validate_tool_call(
                    tool_name=tool_name,
                    arguments=tool_arguments,
                    company_id=company_id,
                    user_id=user_id,
                    role=role,
                    confirmation=confirmation,
                )
            )

            all_violations.extend(
                tool_result.violations
            )

        return self._result(
            violations=all_violations,
            sanitized_input=(
                input_result.sanitized_input
            ),
        )

    # ========================================================================
    # Enforcement
    # ========================================================================

    def enforce(
        self,
        result: GuardrailResult,
    ) -> GuardrailResult:
        """
        Raise an exception for blocking violations.

        Warnings/review requirements are returned without
        raising.
        """

        if not result.allowed:

            critical_or_high = [
                violation
                for violation
                in result.violations
                if violation.action
                == GuardrailAction.BLOCK
            ]

            if critical_or_high:

                first = critical_or_high[0]

                if (
                    first.category
                    == GuardrailCategory.PROMPT_INJECTION
                ):
                    raise PromptInjectionDetectedError(
                        first.message
                    )

                if (
                    first.category
                    == GuardrailCategory.AUTHORIZATION
                ):
                    raise UnauthorizedToolError(
                        first.message
                    )

                if (
                    first.category
                    == GuardrailCategory
                    .CROSS_COMPANY_ACCESS
                ):
                    raise CrossCompanyAccessError(
                        first.message
                    )

                raise GuardrailViolationError(
                    first.message
                )

        return result

    # ========================================================================
    # Internal Result Construction
    # ========================================================================

    def _result(
        self,
        violations: Optional[
            List[GuardrailViolation]
        ] = None,
        warnings: Optional[
            List[str]
        ] = None,
        sanitized_input: Any = None,
        sanitized_output: Any = None,
        requires_confirmation: bool = False,
        requires_human_review: bool = False,
    ) -> GuardrailResult:

        violations = violations or []
        warnings = warnings or []

        blocked = any(
            violation.action
            == GuardrailAction.BLOCK
            for violation in violations
        )

        review_required = (
            requires_human_review
            or any(
                violation.action
                == GuardrailAction.REQUIRE_REVIEW
                for violation in violations
            )
        )

        severity = self._highest_severity(
            violations
        )

        if blocked:
            action = GuardrailAction.BLOCK
        elif review_required:
            action = GuardrailAction.REQUIRE_REVIEW
        elif any(
            violation.action
            == GuardrailAction.REDACT
            for violation in violations
        ):
            action = GuardrailAction.REDACT
        elif warnings:
            action = GuardrailAction.WARN
        else:
            action = GuardrailAction.ALLOW

        result = GuardrailResult(
            allowed=not blocked,
            action=action,
            severity=severity,
            violations=violations,
            warnings=warnings,
            sanitized_input=sanitized_input,
            sanitized_output=sanitized_output,
            requires_confirmation=(
                requires_confirmation
            ),
            requires_human_review=review_required,
        )

        self._audit(result)

        return result

    @staticmethod
    def _highest_severity(
        violations: Sequence[
            GuardrailViolation
        ],
    ) -> GuardrailSeverity:

        priority = {
            GuardrailSeverity.LOW: 0,
            GuardrailSeverity.MEDIUM: 1,
            GuardrailSeverity.HIGH: 2,
            GuardrailSeverity.CRITICAL: 3,
        }

        highest = GuardrailSeverity.LOW

        for violation in violations:

            if (
                priority[
                    violation.severity
                ]
                > priority[highest]
            ):
                highest = (
                    violation.severity
                )

        return highest

    # ========================================================================
    # Audit
    # ========================================================================

    def _audit(
        self,
        result: GuardrailResult,
    ) -> None:

        if self.audit_service is None:
            return

        payload = {
            "event": "agent_guardrail_evaluation",
            "allowed": result.allowed,
            "action": result.action.value,
            "severity": result.severity.value,
            "requires_confirmation": (
                result.requires_confirmation
            ),
            "requires_human_review": (
                result.requires_human_review
            ),
            "violations": [
                violation.to_dict()
                for violation in result.violations
            ],
        }

        try:

            self._invoke(
                self.audit_service,
                (
                    "record",
                    "log",
                    "create",
                    "write",
                ),
                payload,
            )

        except Exception:
            # Guardrail auditing must never cause a
            # safety failure to become an execution failure.
            pass

    # ========================================================================
    # Generic Invocation
    # ========================================================================

    @staticmethod
    def _invoke(
        service: Any,
        method_names: Sequence[str],
        payload: Mapping[str, Any],
    ) -> Any:

        for method_name in method_names:

            method = getattr(
                service,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:
                return method(
                    **dict(payload)
                )

            except TypeError:

                try:
                    return method(
                        dict(payload)
                    )

                except TypeError:
                    continue

        if callable(service):

            try:
                return service(
                    **dict(payload)
                )

            except TypeError:
                return service(
                    dict(payload)
                )

        raise GuardrailConfigurationError(
            "No compatible authorization "
            "or service method found."
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def validate_agent_input(
    text: str,
    company_id: Optional[str] = None,
    config: Optional[GuardrailConfig] = None,
) -> GuardrailResult:
    """Validate agent input."""

    guardrails = AgentGuardrails(
        config=config
    )

    return guardrails.validate_input(
        text=text,
        company_id=company_id,
    )


def validate_tool_call(
    tool_name: str,
    arguments: Optional[
        Mapping[str, Any]
    ] = None,
    company_id: Optional[str] = None,
    config: Optional[GuardrailConfig] = None,
) -> GuardrailResult:
    """Validate an agent tool call."""

    guardrails = AgentGuardrails(
        config=config
    )

    return guardrails.validate_tool_call(
        tool_name=tool_name,
        arguments=arguments,
        company_id=company_id,
    )


def redact_sensitive_data(
    data: Any,
    config: Optional[GuardrailConfig] = None,
) -> Any:
    """Redact sensitive fields."""

    guardrails = AgentGuardrails(
        config=config
    )

    return guardrails.redact_sensitive_data(
        data
    )


def detect_prompt_injection(
    text: str,
) -> List[str]:
    """Detect prompt-injection patterns."""

    guardrails = AgentGuardrails()

    return guardrails.detect_prompt_injection(
        text
    )


def validate_company_access(
    requested_company_id: str,
    authorized_company_id: str,
) -> bool:
    """Check company-level isolation."""

    guardrails = AgentGuardrails()

    return guardrails.validate_company_access(
        requested_company_id,
        authorized_company_id,
    )


# ============================================================================
# Serialization Helpers
# ============================================================================


def _safe_float(
    value: Any,
) -> Optional[float]:

    if isinstance(
        value,
        bool,
    ):
        return None

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not isfinite(number):
        return None

    return number


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
        date,
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

    if hasattr(
        value,
        "__dict__",
    ):
        try:
            return {
                str(key): _serialize(item)
                for key, item in vars(
                    value
                ).items()
                if not key.startswith("_")
            }
        except Exception:
            pass

    return value


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "AgentGuardrailError",
    "GuardrailViolationError",
    "PromptInjectionDetectedError",
    "UnauthorizedToolError",
    "UnsafeToolCallError",
    "SensitiveDataAccessError",
    "CrossCompanyAccessError",
    "InvalidAgentInputError",
    "InvalidAgentOutputError",
    "AgentExecutionLimitError",
    "GuardrailConfigurationError",

    # Enums
    "GuardrailAction",
    "GuardrailSeverity",
    "GuardrailCategory",
    "ToolRiskLevel",
    "ToolActionType",

    # Dataclasses
    "GuardrailConfig",
    "ToolPolicy",
    "GuardrailViolation",
    "GuardrailResult",

    # Policies
    "DEFAULT_TOOL_POLICIES",

    # Main class
    "AgentGuardrails",

    # Convenience functions
    "validate_agent_input",
    "validate_tool_call",
    "redact_sensitive_data",
    "detect_prompt_injection",
    "validate_company_access",
]