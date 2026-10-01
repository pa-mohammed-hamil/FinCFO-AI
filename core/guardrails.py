# backend/app/core/guardrails.py

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence


# ======================================================================
# ENUMS
# ======================================================================


class GuardrailAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REVIEW = "review"
    SANITIZE = "sanitize"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ======================================================================
# RESULT
# ======================================================================


@dataclass
class GuardrailResult:
    """
    Result returned by every guardrail check.
    """

    action: GuardrailAction
    risk_level: RiskLevel = RiskLevel.LOW
    allowed: bool = True
    message: str = ""
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_text: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action == GuardrailAction.BLOCK

    @property
    def requires_review(self) -> bool:
        return self.action == GuardrailAction.REVIEW


# ======================================================================
# PATTERNS
# ======================================================================


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+the\s+system\s+prompt",
    r"forget\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+(me\s+)?your\s+hidden\s+instructions",
    r"print\s+your\s+system\s+message",
    r"bypass\s+(the\s+)?security",
    r"bypass\s+(the\s+)?guardrails",
    r"disable\s+(the\s+)?guardrails",
    r"override\s+(the\s+)?safety",
    r"act\s+as\s+an?\s+unrestricted",
    r"you\s+are\s+now\s+an?\s+unrestricted",
]


SENSITIVE_DATA_PATTERNS = {
    "credit_card": re.compile(
        r"\b(?:\d[ -]*?){13,19}\b"
    ),
    "bank_account": re.compile(
        r"\b\d{9,18}\b"
    ),
    "pan_india": re.compile(
        r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        re.IGNORECASE,
    ),
    "aadhaar_india": re.compile(
        r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"
    ),
    "email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@"
        r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
}


# ======================================================================
# CONFIGURATION
# ======================================================================


@dataclass
class GuardrailConfig:
    """
    Central configuration for FinCo AI guardrails.
    """

    max_input_length: int = 12000
    max_output_length: int = 20000

    block_prompt_injection: bool = True
    detect_sensitive_data: bool = True

    require_review_for_critical_risk: bool = True
    require_review_for_financial_actions: bool = True

    allow_external_urls: bool = False
    allow_code_execution: bool = False

    max_tool_calls: int = 10
    max_agent_steps: int = 15

    minimum_forecast_confidence: float = 0.60
    minimum_recommendation_confidence: float = 0.70


# ======================================================================
# MAIN SERVICE
# ======================================================================


class GuardrailService:
    """
    Central guardrail engine for FinCo AI.

    Protects:

        User input
        Documents
        RAG context
        LLM output
        Agent planning
        Tool execution
        Financial recommendations
        High-risk actions
    """

    def __init__(
        self,
        config: Optional[GuardrailConfig] = None,
    ):
        self.config = config or GuardrailConfig()

    # ------------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------------

    def check_input(
        self,
        text: str,
        *,
        user_id: Optional[int] = None,
        company_id: Optional[int] = None,
    ) -> GuardrailResult:
        """
        Validate user input before sending it to the LLM/agent system.
        """

        if not text or not text.strip():
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.LOW,
                allowed=False,
                message="Input cannot be empty.",
                reasons=["empty_input"],
            )

        if len(text) > self.config.max_input_length:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.MEDIUM,
                allowed=False,
                message="Input exceeds the maximum allowed length.",
                reasons=["input_too_long"],
                metadata={
                    "max_length": self.config.max_input_length,
                },
            )

        injection_result = self.check_prompt_injection(text)

        if injection_result.blocked:
            return injection_result

        sensitive_result = self.check_sensitive_data(text)

        if sensitive_result.blocked:
            return sensitive_result

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            risk_level=RiskLevel.LOW,
            allowed=True,
            message="Input passed guardrail checks.",
            metadata={
                "user_id": user_id,
                "company_id": company_id,
            },
        )

    # ------------------------------------------------------------------
    # PROMPT INJECTION
    # ------------------------------------------------------------------

    def check_prompt_injection(
        self,
        text: str,
    ) -> GuardrailResult:
        """
        Detect common prompt-injection attempts.

        This is a heuristic layer and should be combined with
        system-level isolation and tool authorization.
        """

        if not self.config.block_prompt_injection:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                allowed=True,
            )

        normalized = " ".join(text.lower().split())

        matches: list[str] = []

        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, normalized):
                matches.append(pattern)

        if matches:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                allowed=False,
                message="Potential prompt injection detected.",
                reasons=["prompt_injection_detected"],
                metadata={
                    "matched_patterns": len(matches),
                },
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            risk_level=RiskLevel.LOW,
            allowed=True,
        )

    # ------------------------------------------------------------------
    # SENSITIVE DATA
    # ------------------------------------------------------------------

    def check_sensitive_data(
        self,
        text: str,
    ) -> GuardrailResult:
        """
        Detect potentially sensitive financial/personal data.
        """

        if not self.config.detect_sensitive_data:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                allowed=True,
            )

        detected: list[str] = []

        for name, pattern in SENSITIVE_DATA_PATTERNS.items():
            if pattern.search(text):
                detected.append(name)

        if detected:
            return GuardrailResult(
                action=GuardrailAction.REVIEW,
                risk_level=RiskLevel.HIGH,
                allowed=True,
                message=(
                    "Potentially sensitive information detected. "
                    "Review or redact before external processing."
                ),
                reasons=["sensitive_data_detected"],
                warnings=detected,
                metadata={
                    "data_types": detected,
                },
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            allowed=True,
        )

    # ------------------------------------------------------------------
    # SANITIZATION
    # ------------------------------------------------------------------

    def sanitize_text(
        self,
        text: str,
    ) -> GuardrailResult:
        """
        Redact detected sensitive values.

        Intended for controlled preprocessing before sending
        content to external AI services.
        """

        sanitized = text

        replacements = 0
        detected_types: list[str] = []

        for name, pattern in SENSITIVE_DATA_PATTERNS.items():

            sanitized, count = pattern.subn(
                f"[REDACTED_{name.upper()}]",
                sanitized,
            )

            if count:
                replacements += count
                detected_types.append(name)

        if replacements == 0:
            return GuardrailResult(
                action=GuardrailAction.ALLOW,
                allowed=True,
                sanitized_text=text,
            )

        return GuardrailResult(
            action=GuardrailAction.SANITIZE,
            risk_level=RiskLevel.MEDIUM,
            allowed=True,
            message="Sensitive information was redacted.",
            reasons=["sensitive_data_redacted"],
            sanitized_text=sanitized,
            metadata={
                "replacement_count": replacements,
                "data_types": detected_types,
            },
        )

    # ------------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------------

    def check_output(
        self,
        text: str,
        *,
        contains_citations: bool = False,
        expected_citations: bool = False,
    ) -> GuardrailResult:
        """
        Validate LLM/agent output.
        """

        if not text or not text.strip():
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                allowed=False,
                message="AI output is empty.",
                reasons=["empty_output"],
            )

        if len(text) > self.config.max_output_length:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.MEDIUM,
                allowed=False,
                message="AI output exceeds the maximum allowed length.",
                reasons=["output_too_long"],
            )

        warnings: list[str] = []

        if expected_citations and not contains_citations:
            warnings.append("missing_citations")

        if warnings:
            return GuardrailResult(
                action=GuardrailAction.REVIEW,
                risk_level=RiskLevel.MEDIUM,
                allowed=True,
                message="AI output requires additional validation.",
                warnings=warnings,
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            allowed=True,
            message="AI output passed basic guardrail checks.",
        )

    # ------------------------------------------------------------------
    # RAG CONTEXT
    # ------------------------------------------------------------------

    def check_rag_context(
        self,
        documents: Sequence[dict[str, Any]],
        *,
        company_id: Optional[int] = None,
    ) -> GuardrailResult:
        """
        Validate retrieved RAG documents.

        Critical rule:
            Retrieved content must not cross company/tenant boundaries.
        """

        if not documents:
            return GuardrailResult(
                action=GuardrailAction.REVIEW,
                risk_level=RiskLevel.MEDIUM,
                allowed=True,
                message="No RAG evidence was retrieved.",
                reasons=["empty_retrieval"],
            )

        invalid_documents = []

        for document in documents:

            document_company_id = document.get("company_id")

            if (
                company_id is not None
                and document_company_id is not None
                and document_company_id != company_id
            ):
                invalid_documents.append(
                    document.get("document_id")
                )

        if invalid_documents:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.CRITICAL,
                allowed=False,
                message="Cross-company document access detected.",
                reasons=["tenant_boundary_violation"],
                metadata={
                    "invalid_documents": invalid_documents,
                },
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            allowed=True,
            metadata={
                "document_count": len(documents),
            },
        )

    # ------------------------------------------------------------------
    # CITATIONS
    # ------------------------------------------------------------------

    def check_citations(
        self,
        citations: Sequence[dict[str, Any]],
    ) -> GuardrailResult:
        """
        Validate basic citation completeness.

        Detailed citation verification belongs in citation_validator.py.
        """

        if not citations:
            return GuardrailResult(
                action=GuardrailAction.REVIEW,
                risk_level=RiskLevel.MEDIUM,
                allowed=True,
                message="No evidence citations were supplied.",
                reasons=["missing_citations"],
            )

        invalid = []

        for citation in citations:

            if not citation.get("document_id"):
                invalid.append("missing_document_id")

            if not citation.get("source"):
                invalid.append("missing_source")

        if invalid:
            return GuardrailResult(
                action=GuardrailAction.REVIEW,
                risk_level=RiskLevel.MEDIUM,
                allowed=True,
                message="One or more citations are incomplete.",
                reasons=["invalid_citations"],
                metadata={
                    "errors": invalid,
                },
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            allowed=True,
        )

    # ------------------------------------------------------------------
    # AGENT PLAN
    # ------------------------------------------------------------------

    def check_agent_plan(
        self,
        steps: Sequence[dict[str, Any]],
    ) -> GuardrailResult:
        """
        Validate an agent execution plan before execution.
        """

        if not steps:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                allowed=False,
                message="Agent plan cannot be empty.",
                reasons=["empty_agent_plan"],
            )

        if len(steps) > self.config.max_agent_steps:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                allowed=False,
                message="Agent plan exceeds maximum allowed steps.",
                reasons=["agent_step_limit_exceeded"],
                metadata={
                    "max_steps": self.config.max_agent_steps,
                },
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            allowed=True,
        )

    # ------------------------------------------------------------------
    # TOOL EXECUTION
    # ------------------------------------------------------------------

    def check_tool_execution(
        self,
        *,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        tool_calls_count: int = 0,
    ) -> GuardrailResult:
        """
        Validate an agent tool invocation.
        """

        arguments = arguments or {}

        if tool_calls_count >= self.config.max_tool_calls:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                allowed=False,
                message="Maximum tool-call limit exceeded.",
                reasons=["tool_call_limit_exceeded"],
            )

        if tool_name in {
            "execute_code",
            "shell",
            "terminal",
            "system_command",
        }:
            if not self.config.allow_code_execution:
                return GuardrailResult(
                    action=GuardrailAction.BLOCK,
                    risk_level=RiskLevel.CRITICAL,
                    allowed=False,
                    message="Code/system execution is not permitted.",
                    reasons=["unsafe_tool"],
                    metadata={
                        "tool": tool_name,
                    },
                )

        if tool_name in {
            "send_money",
            "transfer_funds",
            "execute_payment",
            "change_bank_account",
            "approve_loan",
        }:
            if self.config.require_review_for_financial_actions:
                return GuardrailResult(
                    action=GuardrailAction.REVIEW,
                    risk_level=RiskLevel.CRITICAL,
                    allowed=True,
                    message=(
                        "Financial action requires human approval."
                    ),
                    reasons=["financial_action_requires_review"],
                    metadata={
                        "tool": tool_name,
                    },
                )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            allowed=True,
            metadata={
                "tool": tool_name,
                "arguments_checked": list(arguments.keys()),
            },
        )

    # ------------------------------------------------------------------
    # FINANCIAL RECOMMENDATION
    # ------------------------------------------------------------------

    def check_recommendation(
        self,
        *,
        confidence: float,
        risk_score: float,
        requires_action: bool = False,
    ) -> GuardrailResult:
        """
        Decide whether an AI financial recommendation can proceed.

        High-risk or low-confidence recommendations should be
        reviewed by a human.
        """

        if not 0 <= confidence <= 1:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                allowed=False,
                message="Recommendation confidence is invalid.",
                reasons=["invalid_confidence"],
            )

        if not 0 <= risk_score <= 100:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                allowed=False,
                message="Risk score must be between 0 and 100.",
                reasons=["invalid_risk_score"],
            )

        if risk_score >= 90:
            return GuardrailResult(
                action=GuardrailAction.REVIEW,
                risk_level=RiskLevel.CRITICAL,
                allowed=True,
                message=(
                    "Critical financial risk requires human review."
                ),
                reasons=["critical_risk"],
            )

        if confidence < self.config.minimum_recommendation_confidence:
            return GuardrailResult(
                action=GuardrailAction.REVIEW,
                risk_level=RiskLevel.MEDIUM,
                allowed=True,
                message=(
                    "Recommendation confidence is below "
                    "the configured threshold."
                ),
                reasons=["low_recommendation_confidence"],
            )

        if requires_action:
            return GuardrailResult(
                action=GuardrailAction.REVIEW,
                risk_level=RiskLevel.HIGH,
                allowed=True,
                message=(
                    "Action-producing recommendation requires "
                    "human approval."
                ),
                reasons=["action_requires_review"],
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            risk_level=RiskLevel.LOW,
            allowed=True,
        )

    # ------------------------------------------------------------------
    # FORECAST
    # ------------------------------------------------------------------

    def check_forecast(
        self,
        *,
        confidence: float,
    ) -> GuardrailResult:
        """
        Validate forecast confidence.
        """

        if not 0 <= confidence <= 1:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                allowed=False,
                message="Invalid forecast confidence.",
                reasons=["invalid_confidence"],
            )

        if confidence < self.config.minimum_forecast_confidence:
            return GuardrailResult(
                action=GuardrailAction.REVIEW,
                risk_level=RiskLevel.MEDIUM,
                allowed=True,
                message="Forecast confidence is low.",
                reasons=["low_forecast_confidence"],
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            allowed=True,
        )

    # ------------------------------------------------------------------
    # URL VALIDATION
    # ------------------------------------------------------------------

    def check_url(
        self,
        url: str,
    ) -> GuardrailResult:
        """
        Prevent agents from freely accessing external URLs unless
        explicitly enabled.
        """

        if not url:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                allowed=False,
                message="URL cannot be empty.",
                reasons=["empty_url"],
            )

        if not self.config.allow_external_urls:
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                risk_level=RiskLevel.HIGH,
                allowed=False,
                message="External URL access is disabled.",
                reasons=["external_url_blocked"],
            )

        return GuardrailResult(
            action=GuardrailAction.ALLOW,
            allowed=True,
        )

    # ------------------------------------------------------------------
    # COMBINED CHECK
    # ------------------------------------------------------------------

    def validate_request(
        self,
        text: str,
        *,
        company_id: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> GuardrailResult:
        """
        Run the primary request guardrail pipeline.
        """

        input_result = self.check_input(
            text,
            user_id=user_id,
            company_id=company_id,
        )

        if input_result.blocked:
            return input_result

        return input_result