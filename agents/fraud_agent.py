"""
FinCo AI - Fraud Agent
======================

Agentic orchestration layer for fraud detection and investigation.

Architecture
------------

Supervisor Agent
       |
       v
   Fraud Agent
       |
       +-------------------+
       |                   |
       v                   v
 Fraud Tools         Fraud Services
       |                   |
       +---------+---------+
                 |
        +--------+---------+
        |        |         |
        v        v         v
   Preprocess  Features   ML Models
                          /       \
                         v         v
                    Anomaly     Supervised
                       \          /
                        \        /
                         v      v
                         Scoring
                            |
                            v
                       Thresholds
                            |
                            v
                     Explainability
                            |
                    +-------+-------+
                    |               |
                    v               v
                  Alert         Human Review
                    |               |
                    +-------+-------+
                            |
                            v
                         Audit Log

Responsibilities
----------------
- Route fraud requests.
- Detect suspicious transactions.
- Run batch fraud detection.
- Investigate individual transactions.
- Analyze fraud signals.
- Coordinate explainability.
- Coordinate fraud scoring and threshold evaluation.
- Produce structured fraud findings.
- Trigger alerts through the alert service.
- Escalate high/critical fraud cases.
- Prepare human-review packages.
- Record audit events.

The agent must NOT:
- invent fraud scores,
- override fraud thresholds,
- directly execute SQL,
- modify transaction records,
- bypass fraud_service,
- replace ML models,
- make irreversible financial decisions,
- approve/block transactions by LLM reasoning alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ============================================================================
# Exceptions
# ============================================================================


class FraudAgentError(Exception):
    """Base exception for fraud-agent failures."""


class InvalidFraudAgentInputError(FraudAgentError):
    """Raised when fraud-agent input is invalid."""


class FraudAgentExecutionError(FraudAgentError):
    """Raised when a fraud-agent operation fails."""


class FraudAgentConfigurationError(FraudAgentError):
    """Raised when the fraud agent is incorrectly configured."""


class FraudAgentAccessDeniedError(FraudAgentError):
    """Raised when fraud access is denied."""


class UnsupportedFraudOperationError(FraudAgentError):
    """Raised when an unsupported operation is requested."""


# ============================================================================
# Constants
# ============================================================================

DEFAULT_LIMIT = 100
MAX_LIMIT = 1000

DEFAULT_REVIEW_THRESHOLD = 50.0
DEFAULT_BLOCK_THRESHOLD = 80.0
DEFAULT_CRITICAL_THRESHOLD = 90.0

MAX_COMPANY_ID_LENGTH = 200
MAX_TRANSACTION_ID_LENGTH = 200
MAX_QUERY_LENGTH = 2_000


# ============================================================================
# Enums
# ============================================================================


class FraudOperation(str, Enum):
    """Supported fraud-agent operations."""

    DETECT = "detect"
    DETECT_BATCH = "detect_batch"
    INVESTIGATE = "investigate"
    ANALYZE_SIGNALS = "analyze_signals"
    PREPROCESS = "preprocess"
    ENGINEER_FEATURES = "engineer_features"
    DETECT_ANOMALY = "detect_anomaly"
    PREDICT_FRAUD = "predict_fraud"
    CALCULATE_SCORE = "calculate_score"
    EVALUATE_THRESHOLD = "evaluate_threshold"
    EXPLAIN = "explain"
    STATISTICS = "statistics"
    RISK_SUMMARY = "risk_summary"


class FraudDecision(str, Enum):
    """Fraud decision."""

    APPROVE = "approve"
    REVIEW = "review"
    BLOCK = "block"


class FraudRiskLevel(str, Enum):
    """Fraud risk level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FraudSignalSeverity(str, Enum):
    """Severity assigned to an individual fraud signal."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HumanReviewPriority(str, Enum):
    """Human-review priority."""

    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FraudAgentConfig:
    """Configuration for the fraud agent."""

    review_threshold: float = DEFAULT_REVIEW_THRESHOLD
    block_threshold: float = DEFAULT_BLOCK_THRESHOLD
    critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD

    default_limit: int = DEFAULT_LIMIT
    max_limit: int = MAX_LIMIT

    require_review_for_high: bool = True
    require_review_for_critical: bool = True

    create_alerts: bool = True
    enable_escalation: bool = True
    audit_enabled: bool = True

    continue_on_batch_error: bool = True

    enabled: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""

        thresholds = (
            self.review_threshold,
            self.block_threshold,
            self.critical_threshold,
        )

        if not all(isfinite(float(value)) for value in thresholds):
            raise FraudAgentConfigurationError(
                "Fraud thresholds must be finite."
            )

        if not (
            0 <= self.review_threshold
            <= self.block_threshold
            <= self.critical_threshold
            <= 100
        ):
            raise FraudAgentConfigurationError(
                "Fraud thresholds must satisfy "
                "0 <= review <= block <= critical <= 100."
            )

        if self.default_limit < 1:
            raise FraudAgentConfigurationError(
                "default_limit must be greater than zero."
            )

        if self.max_limit < self.default_limit:
            raise FraudAgentConfigurationError(
                "max_limit must be >= default_limit."
            )


# ============================================================================
# Request
# ============================================================================


@dataclass
class FraudAgentRequest:
    """Normalized request sent to the fraud agent."""

    company_id: str

    operation: FraudOperation = FraudOperation.DETECT

    transaction: Any = None
    transactions: Any = None
    history: Any = None

    transaction_id: Optional[str] = None

    anomaly_score: Optional[float] = None
    fraud_probability: Optional[float] = None
    fraud_score: Optional[float] = None
    confidence: Optional[float] = None

    features: Any = None

    limit: int = DEFAULT_LIMIT

    query: Optional[str] = None

    user_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        config: FraudAgentConfig,
    ) -> None:
        """Validate request."""

        if not isinstance(self.company_id, str):
            raise InvalidFraudAgentInputError(
                "company_id must be a string."
            )

        self.company_id = self.company_id.strip()

        if not self.company_id:
            raise InvalidFraudAgentInputError(
                "company_id cannot be empty."
            )

        if len(self.company_id) > MAX_COMPANY_ID_LENGTH:
            raise InvalidFraudAgentInputError(
                "company_id is too long."
            )

        if self.transaction_id is not None:
            if not isinstance(self.transaction_id, str):
                raise InvalidFraudAgentInputError(
                    "transaction_id must be a string."
                )

            self.transaction_id = self.transaction_id.strip()

            if len(self.transaction_id) > MAX_TRANSACTION_ID_LENGTH:
                raise InvalidFraudAgentInputError(
                    "transaction_id is too long."
                )

        if self.limit < 1 or self.limit > config.max_limit:
            raise InvalidFraudAgentInputError(
                f"limit must be between 1 and {config.max_limit}."
            )

        if self.query is not None:
            if not isinstance(self.query, str):
                raise InvalidFraudAgentInputError(
                    "query must be a string."
                )

            self.query = self.query.strip()

            if len(self.query) > MAX_QUERY_LENGTH:
                raise InvalidFraudAgentInputError(
                    "query is too long."
                )

        for name, value in (
            ("anomaly_score", self.anomaly_score),
            ("fraud_probability", self.fraud_probability),
            ("fraud_score", self.fraud_score),
            ("confidence", self.confidence),
        ):
            if value is not None:
                normalized = _normalize_score(value)

                if normalized < 0 or normalized > 100:
                    raise InvalidFraudAgentInputError(
                        f"{name} must be between 0 and 100 "
                        "or a normalized value between 0 and 1."
                    )


# ============================================================================
# Signal / Finding
# ============================================================================


@dataclass
class FraudFinding:
    """Structured fraud finding."""

    finding_type: str
    title: str
    message: str

    severity: FraudSignalSeverity = FraudSignalSeverity.LOW

    signal: Optional[str] = None

    score: Optional[float] = None

    evidence: List[Any] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_type": self.finding_type,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "signal": self.signal,
            "score": self.score,
            "evidence": _serialize(self.evidence),
            "metadata": _serialize(self.metadata),
        }


# ============================================================================
# Result
# ============================================================================


@dataclass
class FraudAgentResult:
    """Standard fraud-agent response."""

    success: bool

    operation: str

    company_id: str

    message: str = ""

    result: Any = None

    findings: List[FraudFinding] = field(
        default_factory=list
    )

    fraud_score: Optional[float] = None

    anomaly_score: Optional[float] = None

    fraud_probability: Optional[float] = None

    confidence: Optional[float] = None

    risk_level: FraudRiskLevel = FraudRiskLevel.LOW

    decision: FraudDecision = FraudDecision.APPROVE

    human_review_required: bool = False

    review_priority: HumanReviewPriority = (
        HumanReviewPriority.NORMAL
    )

    alert_created: bool = False

    alert_id: Optional[str] = None

    execution_time_ms: Optional[float] = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "company_id": self.company_id,
            "message": self.message,
            "result": _serialize(self.result),
            "findings": [
                finding.to_dict()
                for finding in self.findings
            ],
            "fraud_score": self.fraud_score,
            "anomaly_score": self.anomaly_score,
            "fraud_probability": self.fraud_probability,
            "confidence": self.confidence,
            "risk_level": self.risk_level.value,
            "decision": self.decision.value,
            "human_review_required": (
                self.human_review_required
            ),
            "review_priority": (
                self.review_priority.value
            ),
            "alert_created": self.alert_created,
            "alert_id": self.alert_id,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat(),
            "metadata": _serialize(self.metadata),
        }


# ============================================================================
# Fraud Agent
# ============================================================================


class FraudAgent:
    """
    Agentic orchestration layer for fraud detection.

    The agent coordinates deterministic fraud services and tools.

    Example:

        agent = FraudAgent(
            fraud_service=fraud_service,
            fraud_tools=fraud_tools,
            scoring_service=scorer,
            threshold_policy=threshold_policy,
            explainability_engine=explainability_engine,
            alert_service=alert_service,
            audit_service=audit_service,
        )

        result = agent.detect(
            company_id="COMP-001",
            transaction=transaction,
        )
    """

    def __init__(
        self,
        fraud_service: Any = None,
        fraud_tools: Any = None,
        preprocessor: Any = None,
        feature_engineer: Any = None,
        anomaly_model: Any = None,
        supervised_model: Any = None,
        scoring_service: Any = None,
        threshold_policy: Any = None,
        explainability_engine: Any = None,
        fraud_repository: Any = None,
        alert_service: Any = None,
        escalation_service: Any = None,
        audit_service: Any = None,
        authorization_service: Any = None,
        config: Optional[FraudAgentConfig] = None,
    ) -> None:

        self.fraud_service = fraud_service
        self.fraud_tools = fraud_tools

        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.anomaly_model = anomaly_model
        self.supervised_model = supervised_model

        self.scoring_service = scoring_service
        self.threshold_policy = threshold_policy
        self.explainability_engine = explainability_engine

        self.fraud_repository = fraud_repository

        self.alert_service = alert_service
        self.escalation_service = escalation_service
        self.audit_service = audit_service
        self.authorization_service = authorization_service

        self.config = config or FraudAgentConfig()
        self.config.validate()

    # ========================================================================
    # Public API
    # ========================================================================

    def run(
        self,
        request: FraudAgentRequest | Mapping[str, Any],
    ) -> FraudAgentResult:
        """Execute a fraud-agent request."""

        if not self.config.enabled:
            raise FraudAgentConfigurationError(
                "Fraud agent is disabled."
            )

        normalized_request = self._normalize_request(
            request
        )

        self._authorize(normalized_request)

        started = datetime.now(timezone.utc)

        try:
            result = self._dispatch(
                normalized_request
            )

            result.execution_time_ms = (
                datetime.now(timezone.utc) - started
            ).total_seconds() * 1000

            self._audit(
                normalized_request,
                result,
            )

            return result

        except FraudAgentError:
            raise

        except Exception as exc:
            self._audit_error(
                normalized_request,
                exc,
            )

            raise FraudAgentExecutionError(
                f"Fraud operation failed: {exc}"
            ) from exc

    # ========================================================================
    # Detection
    # ========================================================================

    def detect(
        self,
        company_id: str,
        transaction: Mapping[str, Any],
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Detect fraud for one transaction."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.DETECT,
            transaction=transaction,
            user_id=user_id,
        )

        return self.run(request)

    def detect_batch(
        self,
        company_id: str,
        transactions: Sequence[Mapping[str, Any]],
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Detect fraud for multiple transactions."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.DETECT_BATCH,
            transactions=transactions,
            user_id=user_id,
        )

        return self.run(request)

    # ========================================================================
    # Investigation
    # ========================================================================

    def investigate(
        self,
        company_id: str,
        transaction: Mapping[str, Any],
        history: Any = None,
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Run a complete fraud investigation."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.INVESTIGATE,
            transaction=transaction,
            history=history,
            user_id=user_id,
        )

        return self.run(request)

    # ========================================================================
    # Specialized Public Methods
    # ========================================================================

    def analyze_signals(
        self,
        company_id: str,
        transaction: Mapping[str, Any],
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Analyze deterministic transaction signals."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.ANALYZE_SIGNALS,
            transaction=transaction,
            user_id=user_id,
        )

        return self.run(request)

    def preprocess(
        self,
        company_id: str,
        transaction: Mapping[str, Any],
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Preprocess a transaction."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.PREPROCESS,
            transaction=transaction,
            user_id=user_id,
        )

        return self.run(request)

    def engineer_features(
        self,
        company_id: str,
        transaction: Mapping[str, Any],
        history: Any = None,
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Generate fraud features."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.ENGINEER_FEATURES,
            transaction=transaction,
            history=history,
            user_id=user_id,
        )

        return self.run(request)

    def detect_anomaly(
        self,
        company_id: str,
        transaction: Mapping[str, Any],
        features: Any = None,
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Run anomaly detection."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.DETECT_ANOMALY,
            transaction=transaction,
            features=features,
            user_id=user_id,
        )

        return self.run(request)

    def predict_fraud(
        self,
        company_id: str,
        transaction: Mapping[str, Any],
        features: Any = None,
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Run supervised fraud prediction."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.PREDICT_FRAUD,
            transaction=transaction,
            features=features,
            user_id=user_id,
        )

        return self.run(request)

    def calculate_score(
        self,
        company_id: str,
        anomaly_score: Optional[float] = None,
        fraud_probability: Optional[float] = None,
        features: Any = None,
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Calculate the final fraud score."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.CALCULATE_SCORE,
            anomaly_score=anomaly_score,
            fraud_probability=fraud_probability,
            features=features,
            user_id=user_id,
        )

        return self.run(request)

    def evaluate_threshold(
        self,
        company_id: str,
        fraud_score: float,
        confidence: Optional[float] = None,
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Evaluate a fraud score against the threshold policy."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.EVALUATE_THRESHOLD,
            fraud_score=fraud_score,
            confidence=confidence,
            user_id=user_id,
        )

        return self.run(request)

    def explain(
        self,
        company_id: str,
        transaction: Mapping[str, Any],
        fraud_score: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        fraud_probability: Optional[float] = None,
        features: Any = None,
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Generate a fraud explanation."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.EXPLAIN,
            transaction=transaction,
            fraud_score=fraud_score,
            anomaly_score=anomaly_score,
            fraud_probability=fraud_probability,
            features=features,
            user_id=user_id,
        )

        return self.run(request)

    def statistics(
        self,
        company_id: str,
        transactions: Sequence[Mapping[str, Any]],
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Calculate fraud statistics."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.STATISTICS,
            transactions=transactions,
            user_id=user_id,
        )

        return self.run(request)

    def risk_summary(
        self,
        company_id: str,
        fraud_results: Any,
        user_id: Optional[str] = None,
    ) -> FraudAgentResult:
        """Create a high-level fraud risk summary."""

        request = FraudAgentRequest(
            company_id=company_id,
            operation=FraudOperation.RISK_SUMMARY,
            transactions=fraud_results,
            user_id=user_id,
        )

        return self.run(request)

    # ========================================================================
    # Dispatch
    # ========================================================================

    def _dispatch(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        operation = request.operation

        if operation == FraudOperation.DETECT:
            return self._run_detect(request)

        if operation == FraudOperation.DETECT_BATCH:
            return self._run_detect_batch(request)

        if operation == FraudOperation.INVESTIGATE:
            return self._run_investigation(request)

        if operation == FraudOperation.ANALYZE_SIGNALS:
            return self._run_signal_analysis(request)

        if operation == FraudOperation.PREPROCESS:
            return self._run_preprocess(request)

        if operation == FraudOperation.ENGINEER_FEATURES:
            return self._run_feature_engineering(request)

        if operation == FraudOperation.DETECT_ANOMALY:
            return self._run_anomaly_detection(request)

        if operation == FraudOperation.PREDICT_FRAUD:
            return self._run_supervised_prediction(request)

        if operation == FraudOperation.CALCULATE_SCORE:
            return self._run_score_calculation(request)

        if operation == FraudOperation.EVALUATE_THRESHOLD:
            return self._run_threshold_evaluation(request)

        if operation == FraudOperation.EXPLAIN:
            return self._run_explanation(request)

        if operation == FraudOperation.STATISTICS:
            return self._run_statistics(request)

        if operation == FraudOperation.RISK_SUMMARY:
            return self._run_risk_summary(request)

        raise UnsupportedFraudOperationError(
            f"Unsupported fraud operation: {operation.value}"
        )

    # ========================================================================
    # Detection Implementation
    # ========================================================================

    def _run_detect(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transaction = self._require_transaction(
            request.transaction
        )

        service = self.fraud_service or self.fraud_tools

        if service is None:
            raise FraudAgentConfigurationError(
                "fraud_service or fraud_tools is required."
            )

        raw_result = self._invoke(
            service,
            (
                "detect_fraud",
                "detect",
                "analyze",
                "run",
            ),
            {
                "company_id": request.company_id,
                "transaction": transaction,
                "user_id": request.user_id,
            },
        )

        result = self._build_result_from_fraud_result(
            request,
            raw_result,
            "Fraud detection completed.",
        )

        return self._post_process_detection(
            request,
            result,
        )

    # ========================================================================
    # Batch Detection
    # ========================================================================

    def _run_detect_batch(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transactions = request.transactions

        if not isinstance(
            transactions,
            Sequence,
        ) or isinstance(
            transactions,
            (str, bytes),
        ):
            raise InvalidFraudAgentInputError(
                "transactions must be a sequence."
            )

        if not transactions:
            raise InvalidFraudAgentInputError(
                "transactions cannot be empty."
            )

        if len(transactions) > self.config.max_limit:
            raise InvalidFraudAgentInputError(
                f"Maximum batch size is "
                f"{self.config.max_limit}."
            )

        service = self.fraud_service or self.fraud_tools

        if service is None:
            raise FraudAgentConfigurationError(
                "fraud_service or fraud_tools is required."
            )

        raw_result = self._invoke(
            service,
            (
                "detect_fraud_batch",
                "detect_batch",
                "analyze_batch",
                "run_batch",
            ),
            {
                "company_id": request.company_id,
                "transactions": list(transactions),
                "user_id": request.user_id,
            },
        )

        findings = self._extract_findings(
            raw_result
        )

        risk_level = self._highest_risk(
            findings
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Batch fraud detection completed.",
            result=raw_result,
            findings=findings,
            risk_level=risk_level,
            decision=self._decision_from_risk(
                risk_level
            ),
            human_review_required=(
                risk_level
                in (
                    FraudRiskLevel.HIGH,
                    FraudRiskLevel.CRITICAL,
                )
            ),
            review_priority=self._review_priority(
                risk_level
            ),
            metadata={
                "batch_size": len(transactions),
            },
        )

    # ========================================================================
    # Investigation
    # ========================================================================

    def _run_investigation(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transaction = self._require_transaction(
            request.transaction
        )

        service = self.fraud_service or self.fraud_tools

        if service is None:
            raise FraudAgentConfigurationError(
                "fraud_service or fraud_tools is required."
            )

        raw_result = self._invoke(
            service,
            (
                "investigate_transaction",
                "investigate",
                "detect",
                "analyze",
                "run",
            ),
            {
                "company_id": request.company_id,
                "transaction": transaction,
                "history": request.history,
                "user_id": request.user_id,
            },
        )

        result = self._build_result_from_fraud_result(
            request,
            raw_result,
            "Fraud investigation completed.",
        )

        return self._post_process_detection(
            request,
            result,
        )

    # ========================================================================
    # Signal Analysis
    # ========================================================================

    def _run_signal_analysis(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transaction = self._require_transaction(
            request.transaction
        )

        service = self.fraud_tools

        if service is not None:
            raw_result = self._invoke(
                service,
                (
                    "analyze_fraud_signals",
                    "analyze_signals",
                    "signals",
                ),
                {
                    "transaction": transaction,
                },
            )
        else:
            raw_result = self._deterministic_signals(
                transaction
            )

        findings = self._findings_from_signals(
            raw_result
        )

        risk_level = self._highest_risk(
            findings
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Fraud signal analysis completed.",
            result=raw_result,
            findings=findings,
            risk_level=risk_level,
            decision=self._decision_from_risk(
                risk_level
            ),
            human_review_required=(
                risk_level
                in (
                    FraudRiskLevel.HIGH,
                    FraudRiskLevel.CRITICAL,
                )
            ),
        )

    # ========================================================================
    # Preprocessing
    # ========================================================================

    def _run_preprocess(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transaction = self._require_transaction(
            request.transaction
        )

        service = (
            self.preprocessor
            or self.fraud_tools
        )

        if service is None:
            raise FraudAgentConfigurationError(
                "No transaction preprocessor configured."
            )

        raw_result = self._invoke(
            service,
            (
                "preprocess_transaction",
                "process",
                "preprocess",
                "transform",
            ),
            {
                "transaction": transaction,
            },
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Transaction preprocessing completed.",
            result=raw_result,
        )

    # ========================================================================
    # Feature Engineering
    # ========================================================================

    def _run_feature_engineering(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transaction = self._require_transaction(
            request.transaction
        )

        service = (
            self.feature_engineer
            or self.fraud_tools
        )

        if service is None:
            raise FraudAgentConfigurationError(
                "No feature-engineering service configured."
            )

        raw_result = self._invoke(
            service,
            (
                "engineer_features",
                "transform",
                "engineer",
                "build",
                "extract",
            ),
            {
                "transaction": transaction,
                "history": request.history,
            },
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Fraud features generated.",
            result=raw_result,
        )

    # ========================================================================
    # Anomaly Detection
    # ========================================================================

    def _run_anomaly_detection(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transaction = self._require_transaction(
            request.transaction
        )

        service = (
            self.anomaly_model
            or self.fraud_tools
        )

        if service is None:
            raise FraudAgentConfigurationError(
                "No anomaly detection service configured."
            )

        raw_result = self._invoke(
            service,
            (
                "detect_anomaly",
                "detect",
                "predict",
                "score",
                "analyze",
            ),
            {
                "transaction": transaction,
                "features": request.features,
                "transaction_id": request.transaction_id,
            },
        )

        anomaly_score = self._extract_score(
            raw_result,
            (
                "anomaly_score",
                "score",
                "risk_score",
            ),
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Anomaly detection completed.",
            result=raw_result,
            anomaly_score=anomaly_score,
            risk_level=self._risk_from_score(
                anomaly_score
            ),
        )

    # ========================================================================
    # Supervised Prediction
    # ========================================================================

    def _run_supervised_prediction(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transaction = self._require_transaction(
            request.transaction
        )

        service = (
            self.supervised_model
            or self.fraud_tools
        )

        if service is None:
            raise FraudAgentConfigurationError(
                "No supervised fraud model configured."
            )

        raw_result = self._invoke(
            service,
            (
                "predict_fraud",
                "predict",
                "predict_proba",
                "score",
                "analyze",
            ),
            {
                "transaction": transaction,
                "features": request.features,
                "transaction_id": request.transaction_id,
            },
        )

        probability = self._extract_score(
            raw_result,
            (
                "fraud_probability",
                "probability",
                "fraud_score",
                "score",
            ),
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Supervised fraud prediction completed.",
            result=raw_result,
            fraud_probability=probability,
            risk_level=self._risk_from_score(
                probability
            ),
        )

    # ========================================================================
    # Score Calculation
    # ========================================================================

    def _run_score_calculation(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        service = (
            self.scoring_service
            or self.fraud_tools
        )

        if service is None:
            raise FraudAgentConfigurationError(
                "No fraud scoring service configured."
            )

        raw_result = self._invoke(
            service,
            (
                "calculate_fraud_score",
                "calculate",
                "calculate_score",
                "score",
            ),
            {
                "anomaly_score": request.anomaly_score,
                "fraud_probability": request.fraud_probability,
                "features": request.features,
            },
        )

        score = self._extract_score(
            raw_result,
            (
                "fraud_score",
                "score",
                "risk_score",
            ),
        )

        confidence = self._extract_score(
            raw_result,
            (
                "confidence",
            ),
        )

        risk_level = self._risk_from_score(
            score
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Fraud risk score calculated.",
            result=raw_result,
            fraud_score=score,
            confidence=confidence,
            risk_level=risk_level,
            decision=self._decision_from_score(
                score
            ),
            human_review_required=(
                risk_level
                in (
                    FraudRiskLevel.HIGH,
                    FraudRiskLevel.CRITICAL,
                )
            ),
            review_priority=self._review_priority(
                risk_level
            ),
        )

    # ========================================================================
    # Threshold Evaluation
    # ========================================================================

    def _run_threshold_evaluation(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        if request.fraud_score is None:
            raise InvalidFraudAgentInputError(
                "fraud_score is required."
            )

        score = _normalize_score(
            request.fraud_score
        )

        service = self.threshold_policy

        if service is not None:
            raw_result = self._invoke(
                service,
                (
                    "evaluate_score",
                    "evaluate_fraud_score",
                    "evaluate",
                    "assess",
                ),
                {
                    "score": score,
                    "confidence": request.confidence,
                },
            )
        else:
            raw_result = {
                "score": score,
                "risk_level": self._risk_from_score(
                    score
                ).value,
                "decision": self._decision_from_score(
                    score
                ).value,
            }

        risk_level = self._extract_risk_level(
            raw_result
        )

        decision = self._extract_decision(
            raw_result
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Fraud threshold evaluation completed.",
            result=raw_result,
            fraud_score=score,
            confidence=(
                _normalize_score(request.confidence)
                if request.confidence is not None
                else None
            ),
            risk_level=risk_level,
            decision=decision,
            human_review_required=(
                decision != FraudDecision.APPROVE
            ),
            review_priority=self._review_priority(
                risk_level
            ),
        )

    # ========================================================================
    # Explainability
    # ========================================================================

    def _run_explanation(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transaction = self._require_transaction(
            request.transaction
        )

        service = (
            self.explainability_engine
            or self.fraud_tools
        )

        if service is None:
            raise FraudAgentConfigurationError(
                "No fraud explainability service configured."
            )

        raw_result = self._invoke(
            service,
            (
                "explain_fraud",
                "explain",
                "analyze",
                "run",
            ),
            {
                "transaction": transaction,
                "fraud_score": request.fraud_score,
                "anomaly_score": request.anomaly_score,
                "fraud_probability": (
                    request.fraud_probability
                ),
                "features": request.features,
            },
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Fraud explanation generated.",
            result=raw_result,
            fraud_score=request.fraud_score,
            anomaly_score=request.anomaly_score,
            fraud_probability=request.fraud_probability,
        )

    # ========================================================================
    # Statistics
    # ========================================================================

    def _run_statistics(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        transactions = request.transactions

        if not isinstance(
            transactions,
            Sequence,
        ) or isinstance(
            transactions,
            (str, bytes),
        ):
            raise InvalidFraudAgentInputError(
                "transactions must be a sequence."
            )

        service = self.fraud_tools

        if service is not None:
            try:
                raw_result = self._invoke(
                    service,
                    (
                        "calculate_fraud_statistics",
                        "statistics",
                        "analyze_statistics",
                    ),
                    {
                        "transactions": list(
                            transactions
                        ),
                    },
                )
            except FraudAgentExecutionError:
                raw_result = (
                    self._deterministic_statistics(
                        transactions
                    )
                )
        else:
            raw_result = (
                self._deterministic_statistics(
                    transactions
                )
            )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Fraud statistics calculated.",
            result=raw_result,
        )

    # ========================================================================
    # Risk Summary
    # ========================================================================

    def _run_risk_summary(
        self,
        request: FraudAgentRequest,
    ) -> FraudAgentResult:

        data = request.transactions

        findings = self._extract_findings(
            data
        )

        if not findings and isinstance(
            data,
            Sequence,
        ):
            findings = self._findings_from_results(
                data
            )

        risk_level = self._highest_risk(
            findings
        )

        scores = self._extract_scores_from_results(
            data
        )

        average_score = (
            sum(scores) / len(scores)
            if scores
            else None
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message=self._risk_summary_message(
                risk_level
            ),
            result={
                "risk_level": risk_level.value,
                "average_fraud_score": average_score,
                "result_count": len(scores),
                "high_risk_count": sum(
                    1
                    for score in scores
                    if score >= self.config.block_threshold
                ),
                "critical_count": sum(
                    1
                    for score in scores
                    if score >= self.config.critical_threshold
                ),
            },
            findings=findings,
            risk_level=risk_level,
            decision=self._decision_from_risk(
                risk_level
            ),
            human_review_required=(
                risk_level
                in (
                    FraudRiskLevel.HIGH,
                    FraudRiskLevel.CRITICAL,
                )
            ),
            review_priority=self._review_priority(
                risk_level
            ),
        )

    # ========================================================================
    # Post Processing
    # ========================================================================

    def _post_process_detection(
        self,
        request: FraudAgentRequest,
        result: FraudAgentResult,
    ) -> FraudAgentResult:

        if (
            result.human_review_required
            and self.config.require_review_for_high
        ):
            self._create_alert_if_required(
                request,
                result,
            )

        if (
            result.risk_level
            == FraudRiskLevel.CRITICAL
            and self.config.require_review_for_critical
        ):
            result.human_review_required = True

        return result

    # ========================================================================
    # Alerting
    # ========================================================================

    def _create_alert_if_required(
        self,
        request: FraudAgentRequest,
        result: FraudAgentResult,
    ) -> None:

        if not self.config.create_alerts:
            return

        if self.alert_service is None:
            return

        if result.risk_level not in (
            FraudRiskLevel.HIGH,
            FraudRiskLevel.CRITICAL,
        ):
            return

        payload = {
            "company_id": request.company_id,
            "alert_type": "fraud_risk",
            "severity": result.risk_level.value,
            "title": (
                "Critical fraud risk detected"
                if result.risk_level
                == FraudRiskLevel.CRITICAL
                else "High fraud risk detected"
            ),
            "message": (
                "Fraud detection identified a "
                f"{result.risk_level.value} risk transaction."
            ),
            "risk_score": result.fraud_score,
            "evidence": [
                finding.to_dict()
                for finding in result.findings
            ],
            "requires_human_review": True,
            "metadata": {
                "transaction_id": (
                    request.transaction_id
                    or self._transaction_id(
                        request.transaction
                    )
                ),
                "fraud_probability": (
                    result.fraud_probability
                ),
                "anomaly_score": (
                    result.anomaly_score
                ),
            },
        }

        try:
            alert_result = self._invoke(
                self.alert_service,
                (
                    "create_alert",
                    "create",
                    "generate_alert",
                ),
                payload,
            )

            result.alert_created = True

            result.alert_id = self._extract_identifier(
                alert_result,
                (
                    "alert_id",
                    "id",
                ),
            )

        except Exception:
            # Fraud detection itself remains successful.
            # Alert failures should be monitored separately.
            result.metadata["alert_error"] = (
                "Failed to create fraud alert."
            )

    # ========================================================================
    # Authorization
    # ========================================================================

    def _authorize(
        self,
        request: FraudAgentRequest,
    ) -> None:

        if self.authorization_service is None:
            return

        result = self._invoke(
            self.authorization_service,
            (
                "authorize",
                "check_permission",
                "can_access",
                "has_permission",
            ),
            {
                "user_id": request.user_id,
                "company_id": request.company_id,
                "operation": request.operation.value,
                "resource": "fraud",
            },
        )

        if result is False:
            raise FraudAgentAccessDeniedError(
                "Fraud operation access denied."
            )

        if isinstance(result, Mapping):
            allowed = result.get(
                "allowed",
                result.get(
                    "authorized",
                    result.get("success", True),
                ),
            )

            if allowed is False:
                raise FraudAgentAccessDeniedError(
                    "Fraud operation access denied."
                )

    # ========================================================================
    # Audit
    # ========================================================================

    def _audit(
        self,
        request: FraudAgentRequest,
        result: FraudAgentResult,
    ) -> None:

        if not self.config.audit_enabled:
            return

        if self.audit_service is None:
            return

        payload = {
            "event": "fraud_agent_execution",
            "operation": request.operation.value,
            "company_id": request.company_id,
            "transaction_id": (
                request.transaction_id
                or self._transaction_id(
                    request.transaction
                )
            ),
            "user_id": request.user_id,
            "success": result.success,
            "fraud_score": result.fraud_score,
            "fraud_probability": (
                result.fraud_probability
            ),
            "anomaly_score": result.anomaly_score,
            "risk_level": result.risk_level.value,
            "decision": result.decision.value,
            "human_review_required": (
                result.human_review_required
            ),
            "alert_created": result.alert_created,
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
            pass

    def _audit_error(
        self,
        request: FraudAgentRequest,
        error: Exception,
    ) -> None:

        if self.audit_service is None:
            return

        payload = {
            "event": "fraud_agent_error",
            "operation": request.operation.value,
            "company_id": request.company_id,
            "transaction_id": (
                request.transaction_id
                or self._transaction_id(
                    request.transaction
                )
            ),
            "user_id": request.user_id,
            "error": str(error),
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
            pass

    # ========================================================================
    # Result Conversion
    # ========================================================================

    def _build_result_from_fraud_result(
        self,
        request: FraudAgentRequest,
        raw_result: Any,
        message: str,
    ) -> FraudAgentResult:

        score = self._extract_score(
            raw_result,
            (
                "fraud_score",
                "risk_score",
                "score",
            ),
        )

        anomaly_score = self._extract_score(
            raw_result,
            (
                "anomaly_score",
            ),
        )

        fraud_probability = self._extract_score(
            raw_result,
            (
                "fraud_probability",
                "probability",
            ),
        )

        confidence = self._extract_score(
            raw_result,
            (
                "confidence",
            ),
        )

        risk_level = self._extract_risk_level(
            raw_result
        )

        if risk_level == FraudRiskLevel.LOW:
            risk_level = self._risk_from_score(
                score
            )

        decision = self._extract_decision(
            raw_result
        )

        if decision == FraudDecision.APPROVE:
            decision = self._decision_from_score(
                score
            )

        findings = self._extract_findings(
            raw_result
        )

        return FraudAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message=message,
            result=raw_result,
            findings=findings,
            fraud_score=score,
            anomaly_score=anomaly_score,
            fraud_probability=fraud_probability,
            confidence=confidence,
            risk_level=risk_level,
            decision=decision,
            human_review_required=(
                decision
                != FraudDecision.APPROVE
            ),
            review_priority=self._review_priority(
                risk_level
            ),
        )

    # ========================================================================
    # Risk Helpers
    # ========================================================================

    def _risk_from_score(
        self,
        score: Optional[float],
    ) -> FraudRiskLevel:

        if score is None:
            return FraudRiskLevel.LOW

        score = _normalize_score(score)

        if score >= self.config.critical_threshold:
            return FraudRiskLevel.CRITICAL

        if score >= self.config.block_threshold:
            return FraudRiskLevel.HIGH

        if score >= self.config.review_threshold:
            return FraudRiskLevel.MEDIUM

        return FraudRiskLevel.LOW

    def _decision_from_score(
        self,
        score: Optional[float],
    ) -> FraudDecision:

        if score is None:
            return FraudDecision.APPROVE

        score = _normalize_score(score)

        if score >= self.config.block_threshold:
            return FraudDecision.BLOCK

        if score >= self.config.review_threshold:
            return FraudDecision.REVIEW

        return FraudDecision.APPROVE

    @staticmethod
    def _decision_from_risk(
        risk_level: FraudRiskLevel,
    ) -> FraudDecision:

        if risk_level == FraudRiskLevel.CRITICAL:
            return FraudDecision.BLOCK

        if risk_level == FraudRiskLevel.HIGH:
            return FraudDecision.BLOCK

        if risk_level == FraudRiskLevel.MEDIUM:
            return FraudDecision.REVIEW

        return FraudDecision.APPROVE

    @staticmethod
    def _review_priority(
        risk_level: FraudRiskLevel,
    ) -> HumanReviewPriority:

        if risk_level == FraudRiskLevel.CRITICAL:
            return HumanReviewPriority.CRITICAL

        if risk_level == FraudRiskLevel.HIGH:
            return HumanReviewPriority.URGENT

        if risk_level == FraudRiskLevel.MEDIUM:
            return HumanReviewPriority.HIGH

        return HumanReviewPriority.NORMAL

    @staticmethod
    def _highest_risk(
        findings: Iterable[FraudFinding],
    ) -> FraudRiskLevel:

        priority = {
            FraudRiskLevel.LOW: 0,
            FraudRiskLevel.MEDIUM: 1,
            FraudRiskLevel.HIGH: 2,
            FraudRiskLevel.CRITICAL: 3,
        }

        highest = FraudRiskLevel.LOW

        for finding in findings:

            level = FraudRiskLevel(
                finding.severity.value
            )

            if priority[level] > priority[highest]:
                highest = level

        return highest

    # ========================================================================
    # Signal Fallback
    # ========================================================================

    @staticmethod
    def _deterministic_signals(
        transaction: Mapping[str, Any],
    ) -> Dict[str, Any]:

        signals: Dict[str, Any] = {}

        if transaction.get("new_device") is True:
            signals["NEW_DEVICE"] = True

        if transaction.get("new_location") is True:
            signals["NEW_LOCATION"] = True

        if transaction.get("is_international") is True:
            signals["INTERNATIONAL_TRANSACTION"] = True

        if transaction.get("is_night") is True:
            signals["NIGHT_TRANSACTION"] = True

        velocity = _safe_float(
            transaction.get("velocity")
        )

        if velocity is not None and velocity >= 10:
            signals["HIGH_VELOCITY"] = velocity

        merchant_risk = _safe_float(
            transaction.get("merchant_risk_score")
        )

        if (
            merchant_risk is not None
            and merchant_risk >= 60
        ):
            signals["HIGH_MERCHANT_RISK"] = merchant_risk

        amount = _safe_float(
            transaction.get("amount")
        )

        if amount is not None:

            if amount >= 100_000:
                signals["VERY_HIGH_AMOUNT"] = amount

            elif amount >= 50_000:
                signals["HIGH_AMOUNT"] = amount

        return signals

    def _findings_from_signals(
        self,
        signals: Any,
    ) -> List[FraudFinding]:

        if not isinstance(signals, Mapping):
            return []

        findings: List[FraudFinding] = []

        for name, value in signals.items():

            signal_name = str(name).upper()

            if signal_name in (
                "VERY_HIGH_AMOUNT",
                "HIGH_VELOCITY",
                "HIGH_MERCHANT_RISK",
            ):
                severity = FraudSignalSeverity.HIGH
            elif signal_name in (
                "NEW_DEVICE",
                "NEW_LOCATION",
                "INTERNATIONAL_TRANSACTION",
            ):
                severity = FraudSignalSeverity.MEDIUM
            else:
                severity = FraudSignalSeverity.LOW

            findings.append(
                FraudFinding(
                    finding_type="fraud_signal",
                    title=signal_name.replace(
                        "_",
                        " ",
                    ).title(),
                    message=(
                        f"Fraud signal detected: "
                        f"{signal_name.replace('_', ' ').lower()}."
                    ),
                    severity=severity,
                    signal=signal_name,
                    score=(
                        _safe_float(value)
                        if not isinstance(
                            value,
                            bool,
                        )
                        else None
                    ),
                    metadata={
                        "raw_value": _serialize(value),
                    },
                )
            )

        return findings

    # ========================================================================
    # Statistics
    # ========================================================================

    @staticmethod
    def _deterministic_statistics(
        transactions: Sequence[Any],
    ) -> Dict[str, Any]:

        total = len(transactions)

        fraud_count = 0
        anomaly_count = 0
        review_count = 0
        block_count = 0
        approve_count = 0

        scores: List[float] = []

        for transaction in transactions:

            if not isinstance(
                transaction,
                Mapping,
            ):
                continue

            score = _safe_float(
                transaction.get(
                    "fraud_score"
                )
            )

            if score is not None:
                score = _normalize_score(
                    score
                )
                scores.append(score)

                if score >= DEFAULT_BLOCK_THRESHOLD:
                    block_count += 1
                    fraud_count += 1

                elif score >= DEFAULT_REVIEW_THRESHOLD:
                    review_count += 1
                    fraud_count += 1

                else:
                    approve_count += 1

            if transaction.get(
                "is_fraudulent"
            ) is True:
                fraud_count += 1

            if transaction.get(
                "is_anomaly"
            ) is True:
                anomaly_count += 1

        return {
            "total_transactions": total,
            "fraud_count": fraud_count,
            "anomaly_count": anomaly_count,
            "review_count": review_count,
            "block_count": block_count,
            "approve_count": approve_count,
            "average_fraud_score": (
                sum(scores) / len(scores)
                if scores
                else 0.0
            ),
            "fraud_rate": (
                fraud_count / total * 100
                if total
                else 0.0
            ),
            "anomaly_rate": (
                anomaly_count / total * 100
                if total
                else 0.0
            ),
        }

    # ========================================================================
    # Finding Extraction
    # ========================================================================

    def _extract_findings(
        self,
        data: Any,
    ) -> List[FraudFinding]:

        serialized = _serialize(data)

        if not isinstance(
            serialized,
            Mapping,
        ):
            return []

        raw_findings = serialized.get(
            "findings",
            serialized.get(
                "reasons",
                serialized.get(
                    "signals",
                    [],
                ),
            ),
        )

        if isinstance(
            raw_findings,
            Mapping,
        ):
            return self._findings_from_signals(
                raw_findings
            )

        if not isinstance(
            raw_findings,
            Sequence,
        ) or isinstance(
            raw_findings,
            (str, bytes),
        ):
            return []

        findings: List[FraudFinding] = []

        for item in raw_findings:

            if isinstance(
                item,
                Mapping,
            ):

                severity = item.get(
                    "severity",
                    "low",
                )

                try:
                    severity_enum = (
                        FraudSignalSeverity(
                            str(
                                severity
                            ).lower()
                        )
                    )
                except ValueError:
                    severity_enum = (
                        FraudSignalSeverity.LOW
                    )

                findings.append(
                    FraudFinding(
                        finding_type=str(
                            item.get(
                                "finding_type",
                                "fraud_signal",
                            )
                        ),
                        title=str(
                            item.get(
                                "title",
                                item.get(
                                    "signal",
                                    "Fraud finding",
                                ),
                            )
                        ),
                        message=str(
                            item.get(
                                "message",
                                item.get(
                                    "reason",
                                    "",
                                ),
                            )
                        ),
                        severity=severity_enum,
                        signal=item.get(
                            "signal"
                        ),
                        score=_safe_float(
                            item.get(
                                "score"
                            )
                        ),
                        evidence=list(
                            item.get(
                                "evidence",
                                [],
                            )
                            or []
                        ),
                        metadata=dict(
                            item.get(
                                "metadata",
                                {},
                            )
                            or {}
                        ),
                    )
                )

        return findings

    @staticmethod
    def _findings_from_results(
        results: Sequence[Any],
    ) -> List[FraudFinding]:

        findings: List[FraudFinding] = []

        for result in results:

            if isinstance(
                result,
                Mapping,
            ):

                raw_findings = result.get(
                    "findings",
                    [],
                )

                if isinstance(
                    raw_findings,
                    Sequence,
                ) and not isinstance(
                    raw_findings,
                    (str, bytes),
                ):

                    for item in raw_findings:

                        if isinstance(
                            item,
                            Mapping,
                        ):
                            severity = item.get(
                                "severity",
                                "low",
                            )

                            try:
                                severity_enum = (
                                    FraudSignalSeverity(
                                        str(
                                            severity
                                        ).lower()
                                    )
                                )
                            except ValueError:
                                severity_enum = (
                                    FraudSignalSeverity.LOW
                                )

                            findings.append(
                                FraudFinding(
                                    finding_type=str(
                                        item.get(
                                            "finding_type",
                                            "fraud_signal",
                                        )
                                    ),
                                    title=str(
                                        item.get(
                                            "title",
                                            "Fraud finding",
                                        )
                                    ),
                                    message=str(
                                        item.get(
                                            "message",
                                            "",
                                        )
                                    ),
                                    severity=severity_enum,
                                    signal=item.get(
                                        "signal"
                                    ),
                                    score=_safe_float(
                                        item.get(
                                            "score"
                                        )
                                    ),
                                )
                            )

        return findings

    # ========================================================================
    # Score Extraction
    # ========================================================================

    @staticmethod
    def _extract_score(
        data: Any,
        keys: Sequence[str],
    ) -> Optional[float]:

        serialized = _serialize(data)

        if isinstance(
            serialized,
            Mapping,
        ):

            for key in keys:

                if key in serialized:
                    value = _safe_float(
                        serialized[key]
                    )

                    if value is not None:
                        return _normalize_score(
                            value
                        )

            for value in serialized.values():

                result = FraudAgent._extract_score(
                    value,
                    keys,
                )

                if result is not None:
                    return result

        elif isinstance(
            serialized,
            Sequence,
        ) and not isinstance(
            serialized,
            (str, bytes),
        ):

            for value in serialized:

                result = FraudAgent._extract_score(
                    value,
                    keys,
                )

                if result is not None:
                    return result

        return None

    @staticmethod
    def _extract_scores_from_results(
        data: Any,
    ) -> List[float]:

        scores: List[float] = []

        serialized = _serialize(data)

        if isinstance(
            serialized,
            Sequence,
        ) and not isinstance(
            serialized,
            (str, bytes),
        ):

            for item in serialized:

                score = FraudAgent._extract_score(
                    item,
                    (
                        "fraud_score",
                        "risk_score",
                        "score",
                    ),
                )

                if score is not None:
                    scores.append(score)

        elif isinstance(
            serialized,
            Mapping,
        ):

            score = FraudAgent._extract_score(
                serialized,
                (
                    "fraud_score",
                    "risk_score",
                    "score",
                ),
            )

            if score is not None:
                scores.append(score)

        return scores

    # ========================================================================
    # Risk / Decision Extraction
    # ========================================================================

    @staticmethod
    def _extract_risk_level(
        data: Any,
    ) -> FraudRiskLevel:

        serialized = _serialize(data)

        if isinstance(
            serialized,
            Mapping,
        ):

            for key in (
                "risk_level",
                "risk",
                "fraud_risk",
            ):

                value = serialized.get(key)

                if value is not None:

                    try:
                        return FraudRiskLevel(
                            str(value).lower()
                        )
                    except ValueError:
                        pass

        return FraudRiskLevel.LOW

    @staticmethod
    def _extract_decision(
        data: Any,
    ) -> FraudDecision:

        serialized = _serialize(data)

        if isinstance(
            serialized,
            Mapping,
        ):

            value = serialized.get(
                "decision"
            )

            if value is not None:

                try:
                    return FraudDecision(
                        str(value).lower()
                    )
                except ValueError:
                    pass

        return FraudDecision.APPROVE

    # ========================================================================
    # Request Helpers
    # ========================================================================

    def _normalize_request(
        self,
        request: FraudAgentRequest
        | Mapping[str, Any],
    ) -> FraudAgentRequest:

        if isinstance(
            request,
            FraudAgentRequest,
        ):
            normalized = request

        elif isinstance(
            request,
            Mapping,
        ):

            payload = dict(request)

            operation = payload.get(
                "operation",
                FraudOperation.DETECT,
            )

            if not isinstance(
                operation,
                FraudOperation,
            ):
                operation = FraudOperation(
                    str(operation).lower()
                )

            normalized = FraudAgentRequest(
                company_id=payload.get(
                    "company_id",
                    "",
                ),
                operation=operation,
                transaction=payload.get(
                    "transaction"
                ),
                transactions=payload.get(
                    "transactions"
                ),
                history=payload.get(
                    "history"
                ),
                transaction_id=payload.get(
                    "transaction_id"
                ),
                anomaly_score=payload.get(
                    "anomaly_score"
                ),
                fraud_probability=payload.get(
                    "fraud_probability"
                ),
                fraud_score=payload.get(
                    "fraud_score"
                ),
                confidence=payload.get(
                    "confidence"
                ),
                features=payload.get(
                    "features"
                ),
                limit=payload.get(
                    "limit",
                    self.config.default_limit,
                ),
                query=payload.get(
                    "query"
                ),
                user_id=payload.get(
                    "user_id"
                ),
                metadata=dict(
                    payload.get(
                        "metadata",
                        {},
                    )
                    or {}
                ),
            )

        else:
            raise InvalidFraudAgentInputError(
                "request must be FraudAgentRequest "
                "or a mapping."
            )

        normalized.validate(
            self.config
        )

        return normalized

    @staticmethod
    def _require_transaction(
        transaction: Any,
    ) -> Mapping[str, Any]:

        if not isinstance(
            transaction,
            Mapping,
        ):
            raise InvalidFraudAgentInputError(
                "transaction must be a mapping."
            )

        if not transaction:
            raise InvalidFraudAgentInputError(
                "transaction cannot be empty."
            )

        return transaction

    @staticmethod
    def _transaction_id(
        transaction: Any,
    ) -> Optional[str]:

        if not isinstance(
            transaction,
            Mapping,
        ):
            return None

        value = transaction.get(
            "transaction_id",
            transaction.get(
                "id"
            ),
        )

        return (
            str(value)
            if value is not None
            else None
        )

    @staticmethod
    def _extract_identifier(
        data: Any,
        keys: Sequence[str],
    ) -> Optional[str]:

        serialized = _serialize(data)

        if isinstance(
            serialized,
            Mapping,
        ):

            for key in keys:

                value = serialized.get(
                    key
                )

                if value is not None:
                    return str(value)

        return None

    # ========================================================================
    # Summary Helpers
    # ========================================================================

    @staticmethod
    def _risk_summary_message(
        risk_level: FraudRiskLevel,
    ) -> str:

        if risk_level == FraudRiskLevel.CRITICAL:
            return (
                "Critical fraud risk detected. "
                "Immediate human investigation is required."
            )

        if risk_level == FraudRiskLevel.HIGH:
            return (
                "High fraud risk detected. "
                "Manual review is recommended."
            )

        if risk_level == FraudRiskLevel.MEDIUM:
            return (
                "Moderate fraud risk detected. "
                "Transaction monitoring is recommended."
            )

        return (
            "Fraud risk is currently low."
        )

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

        raise FraudAgentExecutionError(
            "No compatible service method found."
        )


# ============================================================================
# Utility Functions
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


def _normalize_score(
    value: Any,
) -> float:

    number = _safe_float(value)

    if number is None:
        raise InvalidFraudAgentInputError(
            "Score must be numeric."
        )

    if 0 <= number <= 1:
        return number * 100

    if 0 <= number <= 100:
        return number

    raise InvalidFraudAgentInputError(
        "Score must be between 0 and 100 "
        "or between 0 and 1."
    )


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
# Convenience Functions
# ============================================================================


def detect_fraud(
    agent: FraudAgent,
    company_id: str,
    transaction: Mapping[str, Any],
    user_id: Optional[str] = None,
) -> FraudAgentResult:
    """Convenience wrapper for fraud detection."""

    return agent.detect(
        company_id=company_id,
        transaction=transaction,
        user_id=user_id,
    )


def investigate_fraud(
    agent: FraudAgent,
    company_id: str,
    transaction: Mapping[str, Any],
    history: Any = None,
    user_id: Optional[str] = None,
) -> FraudAgentResult:
    """Convenience wrapper for fraud investigation."""

    return agent.investigate(
        company_id=company_id,
        transaction=transaction,
        history=history,
        user_id=user_id,
    )


def analyze_fraud_signals(
    agent: FraudAgent,
    company_id: str,
    transaction: Mapping[str, Any],
    user_id: Optional[str] = None,
) -> FraudAgentResult:
    """Convenience wrapper for signal analysis."""

    return agent.analyze_signals(
        company_id=company_id,
        transaction=transaction,
        user_id=user_id,
    )


def calculate_fraud_score(
    agent: FraudAgent,
    company_id: str,
    anomaly_score: Optional[float] = None,
    fraud_probability: Optional[float] = None,
    features: Any = None,
    user_id: Optional[str] = None,
) -> FraudAgentResult:
    """Convenience wrapper for fraud scoring."""

    return agent.calculate_score(
        company_id=company_id,
        anomaly_score=anomaly_score,
        fraud_probability=fraud_probability,
        features=features,
        user_id=user_id,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "FraudAgentError",
    "InvalidFraudAgentInputError",
    "FraudAgentExecutionError",
    "FraudAgentConfigurationError",
    "FraudAgentAccessDeniedError",
    "UnsupportedFraudOperationError",

    # Enums
    "FraudOperation",
    "FraudDecision",
    "FraudRiskLevel",
    "FraudSignalSeverity",
    "HumanReviewPriority",

    # Dataclasses
    "FraudAgentConfig",
    "FraudAgentRequest",
    "FraudFinding",
    "FraudAgentResult",

    # Agent
    "FraudAgent",

    # Convenience
    "detect_fraud",
    "investigate_fraud",
    "analyze_fraud_signals",
    "calculate_fraud_score",
]