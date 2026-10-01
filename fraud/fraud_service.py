"""
FinCo AI - Fraud Detection Service

File:
    backend/app/fraud/fraud_service.py

Purpose:
    Orchestrate the complete fraud detection workflow.

Pipeline:

    Transaction
        |
        v
    Preprocessing
        |
        v
    Feature Engineering
        |
        +----------------------+
        |                      |
        v                      v
    Anomaly Model        Supervised Model
        |                      |
        +----------+-----------+
                   |
                   v
                Scoring
                   |
                   v
             Explainability
                   |
                   v
             Fraud Decision
                   |
          +--------+--------+
          |                 |
          v                 v
      Repository        Alert Service
          |                 |
          +--------+--------+
                   |
                   v
              Audit Log

Design:
    - Service/orchestration layer
    - Model-agnostic
    - Dependency injection friendly
    - Repository independent
    - No direct database dependency
    - No LLM dependency
    - Deterministic orchestration
    - Suitable for API and Agentic AI workflows
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import math
import uuid


# ============================================================================
# Exceptions
# ============================================================================


class FraudServiceError(Exception):
    """Base exception for fraud service errors."""


class FraudInputError(FraudServiceError):
    """Raised when fraud input is invalid."""


class FraudModelError(FraudServiceError):
    """Raised when a fraud model fails."""


class FraudPersistenceError(FraudServiceError):
    """Raised when fraud results cannot be persisted."""


class FraudConfigurationError(FraudServiceError):
    """Raised when the service is incorrectly configured."""


# ============================================================================
# Fraud Status
# ============================================================================


class FraudDecision:
    """Fraud decision constants."""

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class FraudRiskLevel:
    """Fraud risk level constants."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================================
# Fraud Result
# ============================================================================


@dataclass
class FraudDetectionResult:
    """
    Final fraud detection result.

    This object is returned to:
        - API routes
        - fraud agents
        - alert service
        - dashboards
        - audit workflows
    """

    detection_id: str

    transaction_id: Optional[str]

    company_id: Optional[str]

    fraud_score: float

    risk_level: str

    decision: str

    is_fraudulent: bool

    is_anomaly: bool

    anomaly_score: float

    fraud_probability: float

    confidence: float

    reasons: List[str] = field(
        default_factory=list
    )

    features: Dict[str, float] = field(
        default_factory=dict
    )

    explanation: Optional[
        Dict[str, Any]
    ] = None

    model_results: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    alert_created: bool = False

    alert_id: Optional[str] = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result."""

        return {
            "detection_id":
                self.detection_id,

            "transaction_id":
                self.transaction_id,

            "company_id":
                self.company_id,

            "fraud_score":
                self.fraud_score,

            "risk_level":
                self.risk_level,

            "decision":
                self.decision,

            "is_fraudulent":
                self.is_fraudulent,

            "is_anomaly":
                self.is_anomaly,

            "anomaly_score":
                self.anomaly_score,

            "fraud_probability":
                self.fraud_probability,

            "confidence":
                self.confidence,

            "reasons":
                list(self.reasons),

            "features":
                dict(self.features),

            "explanation":
                self.explanation,

            "model_results":
                dict(self.model_results),

            "alert_created":
                self.alert_created,

            "alert_id":
                self.alert_id,

            "created_at":
                self.created_at.isoformat(),

            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FraudServiceConfig:
    """
    Runtime thresholds for fraud decisions.
    """

    review_threshold: float = 50.0

    block_threshold: float = 80.0

    critical_threshold: float = 90.0

    minimum_anomaly_score: float = 70.0

    minimum_fraud_probability: float = 70.0

    create_alert_for_review: bool = True

    create_alert_for_block: bool = True

    persist_results: bool = True

    def validate(self) -> None:

        if not (
            0 <= self.review_threshold <= 100
        ):
            raise FraudConfigurationError(
                "review_threshold must be "
                "between 0 and 100."
            )

        if not (
            0 <= self.block_threshold <= 100
        ):
            raise FraudConfigurationError(
                "block_threshold must be "
                "between 0 and 100."
            )

        if (
            self.block_threshold
            < self.review_threshold
        ):
            raise FraudConfigurationError(
                "block_threshold must be "
                "greater than or equal to "
                "review_threshold."
            )

        if not (
            0 <= self.critical_threshold <= 100
        ):
            raise FraudConfigurationError(
                "critical_threshold must be "
                "between 0 and 100."
            )

        if not (
            0 <= self.minimum_anomaly_score <= 100
        ):
            raise FraudConfigurationError(
                "minimum_anomaly_score must be "
                "between 0 and 100."
            )

        if not (
            0 <= self.minimum_fraud_probability <= 100
        ):
            raise FraudConfigurationError(
                "minimum_fraud_probability must "
                "be between 0 and 100."
            )


# ============================================================================
# Fraud Service
# ============================================================================


class FraudService:
    """
    Main fraud detection orchestration service.

    Expected dependencies are injected so this service can work with:

        preprocessing.py
        feature_engineering.py
        anomaly_model.py
        supervised_model.py
        scoring.py
        explainability.py
        repositories
        alert_service
        audit service

    No concrete implementation is required for every dependency.
    """

    def __init__(
        self,
        feature_engineer: Any,
        anomaly_model: Any = None,
        supervised_model: Any = None,
        scorer: Any = None,
        explainability_engine: Any = None,
        fraud_repository: Any = None,
        alert_service: Any = None,
        audit_service: Any = None,
        config: Optional[
            FraudServiceConfig
        ] = None,
    ) -> None:

        if feature_engineer is None:
            raise FraudConfigurationError(
                "feature_engineer is required."
            )

        self.feature_engineer = (
            feature_engineer
        )

        self.anomaly_model = (
            anomaly_model
        )

        self.supervised_model = (
            supervised_model
        )

        self.scorer = scorer

        self.explainability_engine = (
            explainability_engine
        )

        self.fraud_repository = (
            fraud_repository
        )

        self.alert_service = (
            alert_service
        )

        self.audit_service = (
            audit_service
        )

        self.config = (
            config
            or FraudServiceConfig()
        )

        self.config.validate()

    # ========================================================================
    # MAIN DETECTION API
    # ========================================================================

    def detect(
        self,
        transaction: Mapping[str, Any],
        history: Optional[
            Sequence[
                Mapping[str, Any]
            ]
        ] = None,
        company_id: Optional[str] = None,
        create_alert: bool = True,
        persist: Optional[bool] = None,
    ) -> FraudDetectionResult:
        """
        Execute the complete fraud detection pipeline.
        """

        self._validate_transaction(
            transaction
        )

        transaction_dict = dict(
            transaction
        )

        transaction_id = (
            self._transaction_id(
                transaction_dict
            )
        )

        company_id = (
            company_id
            or self._optional_string(
                transaction_dict.get(
                    "company_id"
                )
            )
        )

        # --------------------------------------------------------------
        # Step 1 - Feature engineering
        # --------------------------------------------------------------

        try:

            feature_vector = (
                self.feature_engineer.transform(
                    transaction_dict,
                    history or [],
                )
            )

        except Exception as exc:

            raise FraudServiceError(
                "Feature engineering failed."
            ) from exc

        features = dict(
            feature_vector.values
        )

        # --------------------------------------------------------------
        # Step 2 - Anomaly model
        # --------------------------------------------------------------

        anomaly_result = (
            self._run_anomaly_model(
                features=features,
                feature_vector=feature_vector,
                transaction=transaction_dict,
                transaction_id=transaction_id,
            )
        )

        # --------------------------------------------------------------
        # Step 3 - Supervised model
        # --------------------------------------------------------------

        supervised_result = (
            self._run_supervised_model(
                features=features,
                feature_vector=feature_vector,
                transaction=transaction_dict,
                transaction_id=transaction_id,
            )
        )

        # --------------------------------------------------------------
        # Step 4 - Calculate fraud score
        # --------------------------------------------------------------

        scoring_result = (
            self._calculate_score(
                features=features,
                anomaly_result=anomaly_result,
                supervised_result=supervised_result,
                transaction=transaction_dict,
            )
        )

        fraud_score = self._extract_score(
            scoring_result,
            anomaly_result,
            supervised_result,
        )

        # --------------------------------------------------------------
        # Step 5 - Risk classification
        # --------------------------------------------------------------

        risk_level = (
            self._risk_level(
                fraud_score
            )
        )

        decision = (
            self._decision(
                fraud_score
            )
        )

        # --------------------------------------------------------------
        # Step 6 - Determine model signals
        # --------------------------------------------------------------

        anomaly_score = self._number(
            anomaly_result.get(
                "anomaly_score"
            )
        )

        fraud_probability = (
            self._fraud_probability(
                supervised_result
            )
        )

        is_anomaly = (
            self._is_anomaly(
                anomaly_result
            )
        )

        is_fraudulent = (
            decision
            in {
                FraudDecision.REVIEW,
                FraudDecision.BLOCK,
            }
        )

        # --------------------------------------------------------------
        # Step 7 - Explainability
        # --------------------------------------------------------------

        explanation = (
            self._generate_explanation(
                transaction=transaction_dict,
                transaction_id=transaction_id,
                fraud_score=fraud_score,
                risk_level=risk_level,
                anomaly_result=anomaly_result,
                supervised_result=supervised_result,
                features=features,
                scoring_result=scoring_result,
            )
        )

        reasons = (
            self._extract_reasons(
                explanation,
                anomaly_result,
                supervised_result,
                scoring_result,
            )
        )

        confidence = (
            self._confidence(
                anomaly_result,
                supervised_result,
                scoring_result,
                fraud_score,
            )
        )

        # --------------------------------------------------------------
        # Step 8 - Create result
        # --------------------------------------------------------------

        result = FraudDetectionResult(
            detection_id=str(
                uuid.uuid4()
            ),
            transaction_id=transaction_id,
            company_id=company_id,
            fraud_score=round(
                fraud_score,
                4,
            ),
            risk_level=risk_level,
            decision=decision,
            is_fraudulent=is_fraudulent,
            is_anomaly=is_anomaly,
            anomaly_score=round(
                anomaly_score,
                4,
            ),
            fraud_probability=round(
                fraud_probability,
                4,
            ),
            confidence=round(
                confidence,
                4,
            ),
            reasons=reasons,
            features=features,
            explanation=explanation,
            model_results={
                "anomaly":
                    anomaly_result,
                "supervised":
                    supervised_result,
                "scoring":
                    scoring_result,
            },
            metadata={
                "service":
                    "FraudService",
                "version":
                    "1.0",
            },
        )

        # --------------------------------------------------------------
        # Step 9 - Persist
        # --------------------------------------------------------------

        should_persist = (
            self.config.persist_results
            if persist is None
            else bool(persist)
        )

        if (
            should_persist
            and self.fraud_repository is not None
        ):

            self._persist(
                result
            )

        # --------------------------------------------------------------
        # Step 10 - Alert
        # --------------------------------------------------------------

        should_alert = (
            create_alert
            and self._should_create_alert(
                result
            )
        )

        if should_alert:

            alert_id = (
                self._create_alert(
                    result
                )
            )

            if alert_id:

                result.alert_created = True
                result.alert_id = alert_id

        # --------------------------------------------------------------
        # Step 11 - Audit
        # --------------------------------------------------------------

        self._audit(
            "FRAUD_DETECTION",
            result,
        )

        return result

    # ========================================================================
    # ANOMALY MODEL
    # ========================================================================

    def _run_anomaly_model(
        self,
        features: Dict[str, float],
        feature_vector: Any,
        transaction: Dict[str, Any],
        transaction_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Execute anomaly model.

        Supports common interfaces:

            model.predict(...)
            model.detect(...)
            model.score(...)
        """

        if self.anomaly_model is None:
            return {}

        try:

            vector = self._feature_vector(
                feature_vector
            )

            if hasattr(
                self.anomaly_model,
                "predict",
            ):

                result = (
                    self.anomaly_model.predict(
                        vector,
                        transaction_id=transaction_id,
                        transaction=transaction,
                    )
                )

            elif hasattr(
                self.anomaly_model,
                "detect",
            ):

                result = (
                    self.anomaly_model.detect(
                        vector,
                        transaction_id=transaction_id,
                        transaction=transaction,
                    )
                )

            elif hasattr(
                self.anomaly_model,
                "score",
            ):

                result = {
                    "anomaly_score":
                        self.anomaly_model.score(
                            vector
                        )
                }

            else:

                raise FraudModelError(
                    "Anomaly model does not "
                    "provide predict(), detect(), "
                    "or score()."
                )

            return self._normalize_result(
                result
            )

        except Exception as exc:

            raise FraudModelError(
                "Anomaly model execution failed."
            ) from exc

    # ========================================================================
    # SUPERVISED MODEL
    # ========================================================================

    def _run_supervised_model(
        self,
        features: Dict[str, float],
        feature_vector: Any,
        transaction: Dict[str, Any],
        transaction_id: Optional[str],
    ) -> Dict[str, Any]:
        """
        Execute supervised fraud model.
        """

        if self.supervised_model is None:
            return {}

        try:

            vector = self._feature_vector(
                feature_vector
            )

            if hasattr(
                self.supervised_model,
                "predict_proba",
            ):

                result = (
                    self.supervised_model.predict_proba(
                        vector
                    )
                )

                return self._normalize_result(
                    result
                )

            if hasattr(
                self.supervised_model,
                "predict",
            ):

                result = (
                    self.supervised_model.predict(
                        vector,
                        transaction_id=transaction_id,
                        transaction=transaction,
                    )
                )

                return self._normalize_result(
                    result
                )

            if hasattr(
                self.supervised_model,
                "score",
            ):

                result = {
                    "fraud_probability":
                        self.supervised_model.score(
                            vector
                        )
                }

                return result

            raise FraudModelError(
                "Supervised model does not "
                "provide predict_proba(), "
                "predict(), or score()."
            )

        except Exception as exc:

            raise FraudModelError(
                "Supervised model execution failed."
            ) from exc

    # ========================================================================
    # SCORING
    # ========================================================================

    def _calculate_score(
        self,
        features: Dict[str, float],
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
        transaction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate final fraud score.

        Preferred interface:

            scorer.calculate(...)

        Compatible alternatives:

            scorer.score(...)
            scorer.calculate_score(...)
        """

        if self.scorer is None:

            return self._fallback_score(
                anomaly_result,
                supervised_result,
            )

        try:

            if hasattr(
                self.scorer,
                "calculate",
            ):

                result = (
                    self.scorer.calculate(
                        features=features,
                        anomaly_result=anomaly_result,
                        supervised_result=supervised_result,
                        transaction=transaction,
                    )
                )

            elif hasattr(
                self.scorer,
                "calculate_score",
            ):

                result = (
                    self.scorer.calculate_score(
                        features=features,
                        anomaly_result=anomaly_result,
                        supervised_result=supervised_result,
                        transaction=transaction,
                    )
                )

            elif hasattr(
                self.scorer,
                "score",
            ):

                result = (
                    self.scorer.score(
                        features=features,
                        anomaly_result=anomaly_result,
                        supervised_result=supervised_result,
                        transaction=transaction,
                    )
                )

            else:

                raise FraudModelError(
                    "Scorer does not provide "
                    "a supported scoring method."
                )

            return self._normalize_result(
                result
            )

        except Exception as exc:

            raise FraudModelError(
                "Fraud scoring failed."
            ) from exc

    def _fallback_score(
        self,
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Safe fallback when no external scorer is configured.

        Weighted combination:

            40% anomaly score
            60% supervised probability
        """

        anomaly_score = self._number(
            anomaly_result.get(
                "anomaly_score"
            )
        )

        fraud_probability = (
            self._fraud_probability(
                supervised_result
            )
        )

        if (
            anomaly_result
            and supervised_result
        ):

            score = (
                anomaly_score * 0.40
                + fraud_probability * 0.60
            )

        elif anomaly_result:

            score = anomaly_score

        elif supervised_result:

            score = fraud_probability

        else:

            score = 0.0

        return {
            "fraud_score":
                self._clamp(
                    score
                ),
            "method":
                "weighted_fallback",
        }

    # ========================================================================
    # EXPLAINABILITY
    # ========================================================================

    def _generate_explanation(
        self,
        transaction: Dict[str, Any],
        transaction_id: Optional[str],
        fraud_score: float,
        risk_level: str,
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
        features: Dict[str, float],
        scoring_result: Dict[str, Any],
    ) -> Optional[
        Dict[str, Any]
    ]:

        if (
            self.explainability_engine
            is None
        ):
            return None

        try:

            feature_importance = (
                self._feature_importance(
                    scoring_result,
                    anomaly_result,
                    supervised_result,
                )
            )

            explanation = (
                self.explainability_engine.explain(
                    transaction=transaction,
                    risk_score=fraud_score,
                    risk_label=risk_level,
                    transaction_id=transaction_id,
                    anomaly_result=anomaly_result,
                    supervised_result=supervised_result,
                    feature_importance=feature_importance,
                )
            )

            if hasattr(
                explanation,
                "to_dict",
            ):

                return explanation.to_dict()

            if isinstance(
                explanation,
                dict,
            ):

                return explanation

            return {
                "summary":
                    str(explanation)
            }

        except Exception as exc:

            # Explainability should not prevent
            # the core fraud detection result.

            return {
                "summary":
                    "Explanation generation "
                    "was unavailable.",
                "error":
                    str(exc),
            }

    # ========================================================================
    # RISK CLASSIFICATION
    # ========================================================================

    def _risk_level(
        self,
        score: float,
    ) -> str:

        if (
            score
            >= self.config.critical_threshold
        ):
            return FraudRiskLevel.CRITICAL

        if (
            score
            >= self.config.block_threshold
        ):
            return FraudRiskLevel.HIGH

        if (
            score
            >= self.config.review_threshold
        ):
            return FraudRiskLevel.MEDIUM

        return FraudRiskLevel.LOW

    def _decision(
        self,
        score: float,
    ) -> str:

        if (
            score
            >= self.config.block_threshold
        ):
            return FraudDecision.BLOCK

        if (
            score
            >= self.config.review_threshold
        ):
            return FraudDecision.REVIEW

        return FraudDecision.APPROVE

    # ========================================================================
    # ALERTS
    # ========================================================================

    def _should_create_alert(
        self,
        result: FraudDetectionResult,
    ) -> bool:

        if self.alert_service is None:
            return False

        if (
            result.decision
            == FraudDecision.BLOCK
            and self.config.create_alert_for_block
        ):
            return True

        if (
            result.decision
            == FraudDecision.REVIEW
            and self.config.create_alert_for_review
        ):
            return True

        return False

    def _create_alert(
        self,
        result: FraudDetectionResult,
    ) -> Optional[str]:
        """
        Create fraud alert through AlertService.

        AlertService remains responsible for:
            - duplicate detection
            - persistence
            - notifications
            - escalation
            - audit
        """

        if self.alert_service is None:
            return None

        payload = {
            "company_id":
                result.company_id,

            "alert_type":
                "FRAUD_RISK",

            "severity":
                self._alert_severity(
                    result.risk_level
                ),

            "title":
                (
                    "Critical fraud risk detected"
                    if result.risk_level
                    == FraudRiskLevel.CRITICAL
                    else "Potential fraud detected"
                ),

            "message":
                self._alert_message(
                    result
                ),

            "risk_score":
                result.fraud_score,

            "source":
                "FraudService",

            "requires_human_review":
                result.decision
                != FraudDecision.APPROVE,

            "metadata":
                {
                    "detection_id":
                        result.detection_id,
                    "transaction_id":
                        result.transaction_id,
                    "fraud_probability":
                        result.fraud_probability,
                    "anomaly_score":
                        result.anomaly_score,
                    "decision":
                        result.decision,
                },
        }

        try:

            if hasattr(
                self.alert_service,
                "create_alert",
            ):

                alert = (
                    self.alert_service.create_alert(
                        **payload
                    )
                )

            elif callable(
                self.alert_service
            ):

                alert = (
                    self.alert_service(
                        payload
                    )
                )

            else:

                return None

            return self._object_id(
                alert
            )

        except Exception:

            # Fraud detection should remain available
            # even if alert delivery fails.
            return None

    @staticmethod
    def _alert_severity(
        risk_level: str,
    ) -> str:

        mapping = {
            FraudRiskLevel.LOW:
                "LOW",

            FraudRiskLevel.MEDIUM:
                "MEDIUM",

            FraudRiskLevel.HIGH:
                "HIGH",

            FraudRiskLevel.CRITICAL:
                "CRITICAL",
        }

        return mapping.get(
            risk_level,
            "MEDIUM",
        )

    @staticmethod
    def _alert_message(
        result: FraudDetectionResult,
    ) -> str:

        reason = (
            result.reasons[0]
            if result.reasons
            else "Multiple fraud risk signals were detected."
        )

        return (
            f"Transaction "
            f"{result.transaction_id or 'unknown'} "
            f"received a fraud risk score of "
            f"{result.fraud_score:.1f}/100. "
            f"Decision: {result.decision}. "
            f"Primary reason: {reason}"
        )

    # ========================================================================
    # PERSISTENCE
    # ========================================================================

    def _persist(
        self,
        result: FraudDetectionResult,
    ) -> None:

        repository = (
            self.fraud_repository
        )

        try:

            payload = result.to_dict()

            if hasattr(
                repository,
                "create",
            ):

                repository.create(
                    payload
                )

            elif hasattr(
                repository,
                "save",
            ):

                repository.save(
                    payload
                )

            elif hasattr(
                repository,
                "insert",
            ):

                repository.insert(
                    payload
                )

            else:

                raise FraudPersistenceError(
                    "Fraud repository does not "
                    "provide create(), save(), "
                    "or insert()."
                )

        except FraudPersistenceError:
            raise

        except Exception as exc:

            raise FraudPersistenceError(
                "Failed to persist fraud result."
            ) from exc

    # ========================================================================
    # AUDIT
    # ========================================================================

    def _audit(
        self,
        action: str,
        result: FraudDetectionResult,
    ) -> None:

        if self.audit_service is None:
            return

        payload = {
            "action": action,
            "entity_type":
                "fraud_detection",
            "entity_id":
                result.detection_id,
            "company_id":
                result.company_id,
            "transaction_id":
                result.transaction_id,
            "risk_level":
                result.risk_level,
            "decision":
                result.decision,
            "fraud_score":
                result.fraud_score,
        }

        try:

            if hasattr(
                self.audit_service,
                "record",
            ):

                self.audit_service.record(
                    **payload
                )

            elif callable(
                self.audit_service
            ):

                self.audit_service(
                    payload
                )

        except Exception:
            # Audit failure should not silently
            # change the fraud decision.
            pass

    # ========================================================================
    # RESULT HELPERS
    # ========================================================================

    def _extract_score(
        self,
        scoring_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
    ) -> float:

        for key in (
            "fraud_score",
            "risk_score",
            "score",
        ):

            if key in scoring_result:

                return self._clamp(
                    self._number(
                        scoring_result[
                            key
                        ]
                    )
                )

        fallback = (
            self._fallback_score(
                anomaly_result,
                supervised_result,
            )
        )

        return self._clamp(
            self._number(
                fallback.get(
                    "fraud_score"
                )
            )
        )

    def _fraud_probability(
        self,
        supervised_result: Dict[str, Any],
    ) -> float:

        for key in (
            "fraud_probability",
            "probability",
            "fraud_score",
        ):

            if key in supervised_result:

                value = self._number(
                    supervised_result[
                        key
                    ]
                )

                # Support probabilities
                # represented as 0-1.

                if 0 <= value <= 1:
                    value *= 100

                return self._clamp(
                    value
                )

        return 0.0

    def _is_anomaly(
        self,
        anomaly_result: Dict[str, Any],
    ) -> bool:

        explicit = anomaly_result.get(
            "is_anomaly"
        )

        if explicit is not None:
            return bool(
                explicit
            )

        score = self._number(
            anomaly_result.get(
                "anomaly_score"
            )
        )

        return (
            score
            >= self.config.minimum_anomaly_score
        )

    def _extract_reasons(
        self,
        explanation: Optional[
            Dict[str, Any]
        ],
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
        scoring_result: Dict[str, Any],
    ) -> List[str]:

        reasons: List[
            str
        ] = []

        if explanation:

            explanation_reasons = (
                explanation.get(
                    "top_reasons",
                    []
                )
            )

            if isinstance(
                explanation_reasons,
                (list, tuple),
            ):

                for reason in (
                    explanation_reasons
                ):

                    text = str(
                        reason
                    ).strip()

                    if (
                        text
                        and text not in reasons
                    ):
                        reasons.append(
                            text
                        )

        for source in (
            anomaly_result,
            supervised_result,
            scoring_result,
        ):

            source_reasons = source.get(
                "reasons",
                []
            )

            if isinstance(
                source_reasons,
                (list, tuple),
            ):

                for reason in source_reasons:

                    text = str(
                        reason
                    ).strip()

                    if (
                        text
                        and text not in reasons
                    ):
                        reasons.append(
                            text
                        )

        return reasons

    def _confidence(
        self,
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
        scoring_result: Dict[str, Any],
        fraud_score: float,
    ) -> float:

        values: List[
            float
        ] = []

        for result in (
            anomaly_result,
            supervised_result,
            scoring_result,
        ):

            for key in (
                "confidence",
                "model_confidence",
            ):

                if key in result:

                    value = self._number(
                        result[key]
                    )

                    if (
                        0 <= value <= 1
                    ):
                        values.append(
                            value
                        )

                    elif (
                        0 < value <= 100
                    ):
                        values.append(
                            value / 100
                        )

        if values:

            return self._clamp(
                sum(values)
                / len(values),
                0.0,
                1.0,
            )

        if fraud_score >= 90:
            return 0.95

        if fraud_score >= 70:
            return 0.85

        if fraud_score >= 50:
            return 0.75

        return 0.65

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================

    @staticmethod
    def _feature_importance(
        scoring_result: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
    ) -> Dict[str, float]:

        for result in (
            scoring_result,
            supervised_result,
            anomaly_result,
        ):

            importance = result.get(
                "feature_importance"
            )

            if isinstance(
                importance,
                dict,
            ):

                cleaned = {}

                for key, value in (
                    importance.items()
                ):

                    try:

                        number = float(
                            value
                        )

                        if math.isfinite(
                            number
                        ):
                            cleaned[
                                str(key)
                            ] = number

                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                if cleaned:
                    return cleaned

        return {}

    # ========================================================================
    # FEATURE VECTOR
    # ========================================================================

    @staticmethod
    def _feature_vector(
        feature_vector: Any,
    ) -> List[float]:

        if hasattr(
            feature_vector,
            "values",
        ):

            values = (
                feature_vector.values
            )

            if isinstance(
                values,
                dict,
            ):

                return [
                    float(value)
                    for value
                    in values.values()
                ]

        if isinstance(
            feature_vector,
            (list, tuple),
        ):

            return [
                float(value)
                for value
                in feature_vector
            ]

        if isinstance(
            feature_vector,
            dict,
        ):

            return [
                float(value)
                for value
                in feature_vector.values()
            ]

        raise FraudModelError(
            "Unable to convert feature vector "
            "to numerical model input."
        )

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    @staticmethod
    def _normalize_result(
        result: Any,
    ) -> Dict[str, Any]:

        if result is None:
            return {}

        if isinstance(
            result,
            dict,
        ):
            return dict(
                result
            )

        if hasattr(
            result,
            "to_dict",
        ):

            converted = (
                result.to_dict()
            )

            if isinstance(
                converted,
                dict,
            ):
                return converted

        if hasattr(
            result,
            "__dict__",
        ):

            return dict(
                result.__dict__
            )

        if isinstance(
            result,
            (int, float),
        ):

            return {
                "score":
                    float(result)
            }

        if isinstance(
            result,
            (list, tuple),
        ):

            return {
                "values":
                    list(result)
            }

        return {
            "result":
                result
        }

    # ========================================================================
    # VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_transaction(
        transaction: Mapping[str, Any],
    ) -> None:

        if not isinstance(
            transaction,
            Mapping,
        ):
            raise FraudInputError(
                "Transaction must be a mapping."
            )

        if not transaction:
            raise FraudInputError(
                "Transaction cannot be empty."
            )

        if "amount" not in transaction:

            raise FraudInputError(
                "Transaction must contain "
                "an amount."
            )

        try:

            amount = float(
                transaction[
                    "amount"
                ]
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise FraudInputError(
                "Transaction amount must "
                "be numeric."
            ) from exc

        if not math.isfinite(
            amount
        ):

            raise FraudInputError(
                "Transaction amount must "
                "be finite."
            )

        if amount < 0:

            raise FraudInputError(
                "Transaction amount cannot "
                "be negative."
            )

    # ========================================================================
    # IDENTIFIERS
    # ========================================================================

    @staticmethod
    def _transaction_id(
        transaction: Dict[str, Any],
    ) -> Optional[str]:

        for key in (
            "transaction_id",
            "id",
        ):

            value = transaction.get(
                key
            )

            if value is not None:

                text = str(
                    value
                ).strip()

                if text:
                    return text

        return None

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        text = str(
            value
        ).strip()

        return (
            text
            if text
            else None
        )

    @staticmethod
    def _object_id(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        if isinstance(
            value,
            str,
        ):
            return value

        if isinstance(
            value,
            dict,
        ):

            for key in (
                "alert_id",
                "id",
                "identifier",
            ):

                if value.get(
                    key
                ) is not None:

                    return str(
                        value[key]
                    )

        for key in (
            "alert_id",
            "id",
            "identifier",
        ):

            if hasattr(
                value,
                key,
            ):

                identifier = getattr(
                    value,
                    key,
                )

                if identifier is not None:

                    return str(
                        identifier
                    )

        return None

    # ========================================================================
    # NUMERICAL HELPERS
    # ========================================================================

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            number = float(
                value
            )

            if not math.isfinite(
                number
            ):
                return default

            return number

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 100.0,
    ) -> float:

        try:
            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return minimum

        return max(
            minimum,
            min(
                maximum,
                number,
            ),
        )


# ============================================================================
# Convenience API
# ============================================================================


def detect_fraud(
    transaction: Mapping[str, Any],
    feature_engineer: Any,
    anomaly_model: Any = None,
    supervised_model: Any = None,
    scorer: Any = None,
    explainability_engine: Any = None,
    fraud_repository: Any = None,
    alert_service: Any = None,
    audit_service: Any = None,
    history: Optional[
        Sequence[
            Mapping[str, Any]
        ]
    ] = None,
    company_id: Optional[str] = None,
) -> FraudDetectionResult:
    """
    Convenience wrapper around FraudService.
    """

    service = FraudService(
        feature_engineer=feature_engineer,
        anomaly_model=anomaly_model,
        supervised_model=supervised_model,
        scorer=scorer,
        explainability_engine=(
            explainability_engine
        ),
        fraud_repository=fraud_repository,
        alert_service=alert_service,
        audit_service=audit_service,
    )

    return service.detect(
        transaction=transaction,
        history=history,
        company_id=company_id,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "FraudServiceError",
    "FraudInputError",
    "FraudModelError",
    "FraudPersistenceError",
    "FraudConfigurationError",
    "FraudDecision",
    "FraudRiskLevel",
    "FraudDetectionResult",
    "FraudServiceConfig",
    "FraudService",
    "detect_fraud",
]