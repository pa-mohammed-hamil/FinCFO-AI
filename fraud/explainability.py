"""
FinCo AI - Fraud Explainability Engine

File:
    backend/app/fraud/explainability.py

Purpose:
    Provide human-readable explanations for fraud and anomaly
    detection results.

Architecture:

    Transaction
         |
         +----------------------+
         |                      |
         v                      v
    anomaly_model.py      supervised_model.py
         |                      |
         +----------+-----------+
                    |
                    v
               scoring.py
                    |
                    v
           explainability.py
                    |
                    v
          Human-readable reasons
                    |
                    v
             fraud_service.py

Design principles:
    - Deterministic
    - Explainable
    - No LLM dependency
    - Model-agnostic
    - Suitable for UI/API responses
    - Supports feature importance
    - Supports anomaly explanations
    - Supports risk-factor explanations
    - Does not make the final fraud decision
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math


# ============================================================================
# Exceptions
# ============================================================================


class ExplainabilityError(Exception):
    """Base exception for explainability failures."""


class InvalidExplanationInputError(
    ExplainabilityError
):
    """Raised when invalid explanation input is supplied."""


# ============================================================================
# Explanation Types
# ============================================================================


class ExplanationType(str, Enum):
    """Types of explanations supported by FinCo AI."""

    ANOMALY = "ANOMALY"
    SUPERVISED = "SUPERVISED"
    RULE = "RULE"
    ENSEMBLE = "ENSEMBLE"
    TRANSACTION = "TRANSACTION"


# ============================================================================
# Risk Direction
# ============================================================================


class RiskDirection(str, Enum):
    """Direction of a feature's contribution to risk."""

    INCREASES_RISK = "INCREASES_RISK"
    DECREASES_RISK = "DECREASES_RISK"
    NEUTRAL = "NEUTRAL"


# ============================================================================
# Feature Contribution
# ============================================================================


@dataclass
class FeatureContribution:
    """
    Explain how an individual feature contributes to fraud risk.
    """

    feature: str

    value: Any

    contribution: float

    direction: str

    importance: float

    explanation: str

    baseline: Optional[float] = None

    threshold: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Explanation Evidence
# ============================================================================


@dataclass
class ExplanationEvidence:
    """
    Evidence supporting a fraud/anomaly explanation.
    """

    evidence_type: str

    feature: Optional[str]

    value: Any

    severity: str

    message: str

    confidence: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Complete Explanation
# ============================================================================


@dataclass
class FraudExplanation:
    """
    Complete explainability result.
    """

    transaction_id: Optional[str]

    risk_score: float

    risk_label: str

    explanation_type: str

    summary: str

    top_reasons: List[str] = field(
        default_factory=list
    )

    feature_contributions: List[
        FeatureContribution
    ] = field(
        default_factory=list
    )

    evidence: List[
        ExplanationEvidence
    ] = field(
        default_factory=list
    )

    model_factors: Dict[
        str,
        float
    ] = field(
        default_factory=dict
    )

    confidence: float = 0.0

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id":
                self.transaction_id,

            "risk_score":
                self.risk_score,

            "risk_label":
                self.risk_label,

            "explanation_type":
                self.explanation_type,

            "summary":
                self.summary,

            "top_reasons": list(
                self.top_reasons
            ),

            "feature_contributions": [
                item.to_dict()
                for item
                in self.feature_contributions
            ],

            "evidence": [
                item.to_dict()
                for item
                in self.evidence
            ],

            "model_factors":
                dict(
                    self.model_factors
                ),

            "confidence":
                self.confidence,

            "metadata":
                dict(
                    self.metadata
                ),
        }


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ExplainabilityConfig:
    """
    Configuration for fraud explanation generation.
    """

    high_amount_threshold: float = 50000.0

    critical_amount_threshold: float = 100000.0

    high_velocity_threshold: float = 10.0

    critical_velocity_threshold: float = 20.0

    high_balance_ratio_threshold: float = 0.75

    critical_balance_ratio_threshold: float = 1.0

    unusual_z_score_threshold: float = 3.0

    top_features: int = 5

    min_contribution_for_reason: float = 5.0

    def validate(self) -> None:

        if (
            self.high_amount_threshold
            < 0
        ):
            raise ExplainabilityError(
                "high_amount_threshold "
                "cannot be negative."
            )

        if (
            self.critical_amount_threshold
            < self.high_amount_threshold
        ):
            raise ExplainabilityError(
                "critical_amount_threshold "
                "must be greater than or equal "
                "to high_amount_threshold."
            )

        if (
            self.high_velocity_threshold
            < 0
        ):
            raise ExplainabilityError(
                "high_velocity_threshold "
                "cannot be negative."
            )

        if (
            self.critical_velocity_threshold
            < self.high_velocity_threshold
        ):
            raise ExplainabilityError(
                "critical_velocity_threshold "
                "must be greater than or equal "
                "to high_velocity_threshold."
            )

        if (
            self.top_features
            <= 0
        ):
            raise ExplainabilityError(
                "top_features must be greater than zero."
            )


# ============================================================================
# Main Explainability Engine
# ============================================================================


class FraudExplainabilityEngine:
    """
    Generates explanations for fraud detection results.

    Supported inputs:

        - Transaction attributes
        - Anomaly model output
        - Statistical signals
        - Rule signals
        - Supervised model probabilities
        - Feature importance values
        - Risk scores

    The engine does not change the underlying model prediction.
    """

    def __init__(
        self,
        config: Optional[
            ExplainabilityConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or ExplainabilityConfig()
        )

        self.config.validate()

    # ========================================================================
    # MAIN API
    # ========================================================================

    def explain(
        self,
        transaction: Optional[
            Dict[str, Any]
        ] = None,
        risk_score: float = 0.0,
        risk_label: str = "NORMAL",
        transaction_id: Optional[str] = None,
        anomaly_result: Optional[
            Dict[str, Any]
        ] = None,
        supervised_result: Optional[
            Dict[str, Any]
        ] = None,
        feature_importance: Optional[
            Dict[str, float]
        ] = None,
    ) -> FraudExplanation:
        """
        Generate a complete fraud explanation.
        """

        transaction = (
            transaction
            or {}
        )

        anomaly_result = (
            anomaly_result
            or {}
        )

        supervised_result = (
            supervised_result
            or {}
        )

        feature_importance = (
            feature_importance
            or {}
        )

        risk_score = self._clamp(
            risk_score
        )

        contributions = (
            self._build_feature_contributions(
                transaction=transaction,
                anomaly_result=anomaly_result,
                supervised_result=supervised_result,
                feature_importance=feature_importance,
            )
        )

        evidence = (
            self._build_evidence(
                transaction=transaction,
                anomaly_result=anomaly_result,
                supervised_result=supervised_result,
            )
        )

        model_factors = (
            self._build_model_factors(
                anomaly_result=anomaly_result,
                supervised_result=supervised_result,
            )
        )

        reasons = (
            self._build_top_reasons(
                contributions=contributions,
                evidence=evidence,
                anomaly_result=anomaly_result,
                supervised_result=supervised_result,
            )
        )

        summary = (
            self.generate_summary(
                risk_score=risk_score,
                risk_label=risk_label,
                reasons=reasons,
            )
        )

        confidence = (
            self._calculate_confidence(
                risk_score=risk_score,
                anomaly_result=anomaly_result,
                supervised_result=supervised_result,
            )
        )

        return FraudExplanation(
            transaction_id=transaction_id,
            risk_score=round(
                risk_score,
                4,
            ),
            risk_label=str(
                risk_label
            ).upper(),
            explanation_type=(
                ExplanationType.ENSEMBLE.value
            ),
            summary=summary,
            top_reasons=reasons[
                : self.config.top_features
            ],
            feature_contributions=(
                contributions[
                    : self.config.top_features
                ]
            ),
            evidence=evidence,
            model_factors=model_factors,
            confidence=round(
                confidence,
                4,
            ),
            metadata={
                "explainability_version":
                    "1.0",
                "deterministic":
                    True,
            },
        )

    # ========================================================================
    # FEATURE CONTRIBUTIONS
    # ========================================================================

    def _build_feature_contributions(
        self,
        transaction: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
        feature_importance: Dict[str, float],
    ) -> List[
        FeatureContribution
    ]:
        """
        Build feature-level explanations.
        """

        contributions: List[
            FeatureContribution
        ] = []

        # --------------------------------------------------------------
        # Amount
        # --------------------------------------------------------------

        if "amount" in transaction:

            amount = self._number(
                transaction[
                    "amount"
                ]
            )

            contribution = (
                self._amount_contribution(
                    amount
                )
            )

            direction = (
                self._direction(
                    contribution
                )
            )

            contributions.append(
                FeatureContribution(
                    feature="amount",
                    value=amount,
                    contribution=contribution,
                    direction=direction,
                    importance=(
                        abs(contribution)
                    ),
                    explanation=(
                        self._amount_explanation(
                            amount
                        )
                    ),
                    baseline=(
                        self.config.high_amount_threshold
                    ),
                    threshold=(
                        self.config.critical_amount_threshold
                    ),
                )
            )

        # --------------------------------------------------------------
        # Velocity
        # --------------------------------------------------------------

        if "velocity" in transaction:

            velocity = self._number(
                transaction[
                    "velocity"
                ]
            )

            contribution = (
                self._velocity_contribution(
                    velocity
                )
            )

            contributions.append(
                FeatureContribution(
                    feature="velocity",
                    value=velocity,
                    contribution=contribution,
                    direction=(
                        self._direction(
                            contribution
                        )
                    ),
                    importance=abs(
                        contribution
                    ),
                    explanation=(
                        self._velocity_explanation(
                            velocity
                        )
                    ),
                    baseline=(
                        self.config.high_velocity_threshold
                    ),
                    threshold=(
                        self.config.critical_velocity_threshold
                    ),
                )
            )

        # --------------------------------------------------------------
        # Amount-to-balance ratio
        # --------------------------------------------------------------

        if (
            "amount_to_balance_ratio"
            in transaction
        ):

            ratio = self._number(
                transaction[
                    "amount_to_balance_ratio"
                ]
            )

            contribution = (
                self._balance_ratio_contribution(
                    ratio
                )
            )

            contributions.append(
                FeatureContribution(
                    feature=(
                        "amount_to_balance_ratio"
                    ),
                    value=ratio,
                    contribution=contribution,
                    direction=(
                        self._direction(
                            contribution
                        )
                    ),
                    importance=abs(
                        contribution
                    ),
                    explanation=(
                        self._balance_ratio_explanation(
                            ratio
                        )
                    ),
                    baseline=(
                        self.config.high_balance_ratio_threshold
                    ),
                    threshold=(
                        self.config.critical_balance_ratio_threshold
                    ),
                )
            )

        # --------------------------------------------------------------
        # International transaction
        # --------------------------------------------------------------

        if transaction.get(
            "is_international"
        ):

            contributions.append(
                FeatureContribution(
                    feature=(
                        "is_international"
                    ),
                    value=True,
                    contribution=10.0,
                    direction=(
                        RiskDirection.INCREASES_RISK.value
                    ),
                    importance=10.0,
                    explanation=(
                        "The transaction is "
                        "international, which may "
                        "increase risk depending "
                        "on the customer's normal "
                        "transaction behavior."
                    ),
                )
            )

        # --------------------------------------------------------------
        # New device
        # --------------------------------------------------------------

        if transaction.get(
            "new_device"
        ):

            contributions.append(
                FeatureContribution(
                    feature="new_device",
                    value=True,
                    contribution=15.0,
                    direction=(
                        RiskDirection.INCREASES_RISK.value
                    ),
                    importance=15.0,
                    explanation=(
                        "The transaction originated "
                        "from a new or previously "
                        "unseen device."
                    ),
                )
            )

        # --------------------------------------------------------------
        # New location
        # --------------------------------------------------------------

        if transaction.get(
            "new_location"
        ):

            contributions.append(
                FeatureContribution(
                    feature="new_location",
                    value=True,
                    contribution=15.0,
                    direction=(
                        RiskDirection.INCREASES_RISK.value
                    ),
                    importance=15.0,
                    explanation=(
                        "The transaction originated "
                        "from a new or unusual "
                        "location."
                    ),
                )
            )

        # --------------------------------------------------------------
        # Model feature importance
        # --------------------------------------------------------------

        for feature, importance in (
            feature_importance.items()
        ):

            if not self._is_number(
                importance
            ):
                continue

            importance_value = abs(
                float(importance)
            )

            value = transaction.get(
                feature
            )

            contributions.append(
                FeatureContribution(
                    feature=feature,
                    value=value,
                    contribution=(
                        importance_value
                    ),
                    direction=(
                        RiskDirection.INCREASES_RISK.value
                        if importance_value > 0
                        else RiskDirection.NEUTRAL.value
                    ),
                    importance=(
                        importance_value
                    ),
                    explanation=(
                        f"{feature} was identified "
                        f"as an important model "
                        f"feature with relative "
                        f"importance "
                        f"{importance_value:.2f}."
                    ),
                )
            )

        # --------------------------------------------------------------
        # Sort by importance
        # --------------------------------------------------------------

        contributions.sort(
            key=lambda item: (
                item.importance
            ),
            reverse=True,
        )

        return contributions

    # ========================================================================
    # AMOUNT
    # ========================================================================

    def _amount_contribution(
        self,
        amount: float,
    ) -> float:

        if (
            amount
            >= self.config.critical_amount_threshold
        ):
            return 35.0

        if (
            amount
            >= self.config.high_amount_threshold
        ):
            return 20.0

        if amount > 0:
            ratio = (
                amount
                / max(
                    self.config.high_amount_threshold,
                    1.0,
                )
            )

            return min(
                10.0,
                ratio * 10.0,
            )

        return 0.0

    def _amount_explanation(
        self,
        amount: float,
    ) -> str:

        if (
            amount
            >= self.config.critical_amount_threshold
        ):
            return (
                "The transaction amount is "
                "extremely high compared with "
                "the configured risk threshold."
            )

        if (
            amount
            >= self.config.high_amount_threshold
        ):
            return (
                "The transaction amount is "
                "substantially above the normal "
                "high-value transaction threshold."
            )

        return (
            "The transaction amount does not "
            "independently represent a major "
            "risk signal."
        )

    # ========================================================================
    # VELOCITY
    # ========================================================================

    def _velocity_contribution(
        self,
        velocity: float,
    ) -> float:

        if (
            velocity
            >= self.config.critical_velocity_threshold
        ):
            return 35.0

        if (
            velocity
            >= self.config.high_velocity_threshold
        ):
            return 22.0

        if velocity > 0:
            ratio = (
                velocity
                / max(
                    self.config.high_velocity_threshold,
                    1.0,
                )
            )

            return min(
                10.0,
                ratio * 10.0,
            )

        return 0.0

    def _velocity_explanation(
        self,
        velocity: float,
    ) -> str:

        if (
            velocity
            >= self.config.critical_velocity_threshold
        ):
            return (
                "Transaction velocity is "
                "extremely high and may indicate "
                "automated or abnormal transaction "
                "activity."
            )

        if (
            velocity
            >= self.config.high_velocity_threshold
        ):
            return (
                "Transaction velocity is higher "
                "than the configured risk threshold."
            )

        return (
            "Transaction velocity does not "
            "independently represent a major "
            "risk signal."
        )

    # ========================================================================
    # BALANCE RATIO
    # ========================================================================

    def _balance_ratio_contribution(
        self,
        ratio: float,
    ) -> float:

        if (
            ratio
            >= self.config.critical_balance_ratio_threshold
        ):
            return 35.0

        if (
            ratio
            >= self.config.high_balance_ratio_threshold
        ):
            return 22.0

        if ratio > 0:
            return min(
                10.0,
                ratio * 10.0,
            )

        return 0.0

    def _balance_ratio_explanation(
        self,
        ratio: float,
    ) -> str:

        if (
            ratio
            >= self.config.critical_balance_ratio_threshold
        ):
            return (
                "The transaction amount exceeds "
                "or is approximately equal to the "
                "available account balance."
            )

        if (
            ratio
            >= self.config.high_balance_ratio_threshold
        ):
            return (
                "The transaction consumes a "
                "large proportion of the available "
                "account balance."
            )

        return (
            "The transaction-to-balance ratio "
            "does not independently represent "
            "a major risk signal."
        )

    # ========================================================================
    # EVIDENCE
    # ========================================================================

    def _build_evidence(
        self,
        transaction: Dict[str, Any],
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
    ) -> List[
        ExplanationEvidence
    ]:

        evidence: List[
            ExplanationEvidence
        ] = []

        # --------------------------------------------------------------
        # Anomaly result
        # --------------------------------------------------------------

        anomaly_score = self._number(
            anomaly_result.get(
                "anomaly_score"
            )
        )

        if anomaly_score >= 70:

            evidence.append(
                ExplanationEvidence(
                    evidence_type=(
                        ExplanationType.ANOMALY.value
                    ),
                    feature=None,
                    value=anomaly_score,
                    severity="HIGH",
                    message=(
                        "The anomaly detection "
                        "model identified the "
                        "transaction as significantly "
                        "different from historical "
                        "patterns."
                    ),
                    confidence=min(
                        1.0,
                        anomaly_score / 100.0,
                    ),
                )
            )

        # --------------------------------------------------------------
        # Loss / fraud probability
        # --------------------------------------------------------------

        fraud_probability = (
            self._number(
                supervised_result.get(
                    "fraud_probability"
                )
            )
        )

        if fraud_probability >= 70:

            evidence.append(
                ExplanationEvidence(
                    evidence_type=(
                        ExplanationType.SUPERVISED.value
                    ),
                    feature=None,
                    value=fraud_probability,
                    severity="HIGH",
                    message=(
                        "The supervised fraud model "
                        "assigned a high probability "
                        "of fraudulent behavior."
                    ),
                    confidence=min(
                        1.0,
                        fraud_probability / 100.0,
                    ),
                )
            )

        # --------------------------------------------------------------
        # Transaction signals
        # --------------------------------------------------------------

        if transaction.get(
            "new_device"
        ):

            evidence.append(
                ExplanationEvidence(
                    evidence_type=(
                        ExplanationType.TRANSACTION.value
                    ),
                    feature="new_device",
                    value=True,
                    severity="MEDIUM",
                    message=(
                        "The transaction originated "
                        "from a new device."
                    ),
                    confidence=0.70,
                )
            )

        if transaction.get(
            "new_location"
        ):

            evidence.append(
                ExplanationEvidence(
                    evidence_type=(
                        ExplanationType.TRANSACTION.value
                    ),
                    feature="new_location",
                    value=True,
                    severity="MEDIUM",
                    message=(
                        "The transaction originated "
                        "from a new or unusual location."
                    ),
                    confidence=0.70,
                )
            )

        if transaction.get(
            "is_international"
        ):

            evidence.append(
                ExplanationEvidence(
                    evidence_type=(
                        ExplanationType.TRANSACTION.value
                    ),
                    feature="is_international",
                    value=True,
                    severity="LOW",
                    message=(
                        "The transaction is international "
                        "and should be compared with "
                        "the customer's normal behavior."
                    ),
                    confidence=0.60,
                )
            )

        return evidence

    # ========================================================================
    # MODEL FACTORS
    # ========================================================================

    def _build_model_factors(
        self,
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
    ) -> Dict[str, float]:

        factors: Dict[
            str,
            float
        ] = {}

        anomaly_score = (
            self._number(
                anomaly_result.get(
                    "anomaly_score"
                )
            )
        )

        if anomaly_result:
            factors[
                "anomaly_score"
            ] = anomaly_score

        supervised_probability = (
            self._number(
                supervised_result.get(
                    "fraud_probability"
                )
            )
        )

        if supervised_result:
            factors[
                "fraud_probability"
            ] = supervised_probability

        supervised_score = (
            self._number(
                supervised_result.get(
                    "model_score"
                )
            )
        )

        if (
            supervised_result
            and "model_score"
            in supervised_result
        ):

            factors[
                "supervised_model_score"
            ] = supervised_score

        return factors

    # ========================================================================
    # TOP REASONS
    # ========================================================================

    def _build_top_reasons(
        self,
        contributions: List[
            FeatureContribution
        ],
        evidence: List[
            ExplanationEvidence
        ],
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
    ) -> List[str]:

        reasons: List[str] = []

        for contribution in contributions:

            if (
                contribution.contribution
                >= self.config.min_contribution_for_reason
            ):

                reasons.append(
                    contribution.explanation
                )

        for item in evidence:

            if item.message not in reasons:
                reasons.append(
                    item.message
                )

        # Add model-provided reasons if available.

        model_reasons = (
            anomaly_result.get(
                "reasons",
                []
            )
        )

        if isinstance(
            model_reasons,
            (list, tuple),
        ):

            for reason in model_reasons:

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

    # ========================================================================
    # SUMMARY
    # ========================================================================

    @staticmethod
    def generate_summary(
        risk_score: float,
        risk_label: str,
        reasons: Sequence[str],
    ) -> str:
        """
        Generate concise human-readable summary.
        """

        label = str(
            risk_label
        ).upper()

        if label == "CRITICAL":

            prefix = (
                "Critical fraud risk detected."
            )

        elif label in (
            "HIGH",
            "HIGH_RISK",
        ):

            prefix = (
                "High fraud risk detected."
            )

        elif label in (
            "MEDIUM",
            "SUSPICIOUS",
        ):

            prefix = (
                "Suspicious transaction "
                "behavior detected."
            )

        elif label == "LOW":

            prefix = (
                "Low-level transaction risk "
                "signals detected."
            )

        else:

            prefix = (
                "No significant fraud risk "
                "was identified."
            )

        if reasons:

            return (
                f"{prefix} Risk score is "
                f"{risk_score:.1f}/100. "
                f"Primary factor: "
                f"{reasons[0]}"
            )

        return (
            f"{prefix} Risk score is "
            f"{risk_score:.1f}/100."
        )

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    @staticmethod
    def _calculate_confidence(
        risk_score: float,
        anomaly_result: Dict[str, Any],
        supervised_result: Dict[str, Any],
    ) -> float:

        scores: List[
            float
        ] = []

        if anomaly_result:

            anomaly_confidence = (
                FraudExplainabilityEngine._number(
                    anomaly_result.get(
                        "confidence"
                    )
                )
            )

            if anomaly_confidence > 0:
                scores.append(
                    anomaly_confidence
                )

        if supervised_result:

            supervised_confidence = (
                FraudExplainabilityEngine._number(
                    supervised_result.get(
                        "confidence"
                    )
                )
            )

            if supervised_confidence > 0:
                scores.append(
                    supervised_confidence
                )

        if scores:

            return max(
                0.0,
                min(
                    1.0,
                    sum(scores)
                    / len(scores),
                ),
            )

        # Fallback based on risk score.

        if risk_score >= 90:
            return 0.95

        if risk_score >= 70:
            return 0.85

        if risk_score >= 40:
            return 0.70

        return 0.60

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _direction(
        contribution: float,
    ) -> str:

        if contribution > 1e-9:
            return (
                RiskDirection.INCREASES_RISK.value
            )

        if contribution < -1e-9:
            return (
                RiskDirection.DECREASES_RISK.value
            )

        return (
            RiskDirection.NEUTRAL.value
        )

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
    def _is_number(
        value: Any,
    ) -> bool:

        try:
            number = float(
                value
            )

            return math.isfinite(
                number
            )

        except (
            TypeError,
            ValueError,
        ):
            return False

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 100.0,
    ) -> float:

        try:
            value = float(
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
                value,
            ),
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def explain_transaction(
    transaction: Dict[str, Any],
    risk_score: float,
    risk_label: str = "NORMAL",
    transaction_id: Optional[str] = None,
    anomaly_result: Optional[
        Dict[str, Any]
    ] = None,
    supervised_result: Optional[
        Dict[str, Any]
    ] = None,
    feature_importance: Optional[
        Dict[str, float]
    ] = None,
) -> FraudExplanation:
    """
    Convenience function for generating a transaction explanation.
    """

    engine = (
        FraudExplainabilityEngine()
    )

    return engine.explain(
        transaction=transaction,
        risk_score=risk_score,
        risk_label=risk_label,
        transaction_id=transaction_id,
        anomaly_result=anomaly_result,
        supervised_result=supervised_result,
        feature_importance=feature_importance,
    )


def explain_anomaly(
    anomaly_result: Dict[str, Any],
    transaction: Optional[
        Dict[str, Any]
    ] = None,
    transaction_id: Optional[str] = None,
) -> FraudExplanation:
    """
    Convenience function specifically for anomaly explanations.
    """

    engine = (
        FraudExplainabilityEngine()
    )

    score = engine._number(
        anomaly_result.get(
            "anomaly_score"
        )
    )

    label = str(
        anomaly_result.get(
            "anomaly_label",
            "NORMAL",
        )
    )

    return engine.explain(
        transaction=transaction,
        risk_score=score,
        risk_label=label,
        transaction_id=transaction_id,
        anomaly_result=anomaly_result,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "ExplainabilityError",
    "InvalidExplanationInputError",
    "ExplanationType",
    "RiskDirection",
    "FeatureContribution",
    "ExplanationEvidence",
    "FraudExplanation",
    "ExplainabilityConfig",
    "FraudExplainabilityEngine",
    "explain_transaction",
    "explain_anomaly",
]