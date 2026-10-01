"""
FinCo AI - Fraud Detection Threshold Policy

File:
    backend/app/fraud/thresholds.py

Purpose:
    Centralized threshold management for fraud detection.

Responsibilities:
    - Fraud probability thresholds
    - Anomaly score thresholds
    - Review / block / critical thresholds
    - Transaction risk thresholds
    - Velocity thresholds
    - Balance-ratio thresholds
    - Merchant-risk thresholds
    - Model agreement thresholds
    - Risk-level classification
    - Fraud decision classification

Design principle:

    Models
       |
       v
    Raw Scores
       |
       v
    ThresholdPolicy
       |
       +------------------+
       |                  |
       v                  v
    Risk Level         Decision
       |                  |
       +--------+---------+
                |
                v
          Fraud Service

This module contains business thresholds only.
It does not perform model training, persistence,
notification, alert creation, or LLM reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
import math


# ============================================================================
# Exceptions
# ============================================================================


class ThresholdError(Exception):
    """Base exception for threshold policy errors."""


class InvalidThresholdError(
    ThresholdError
):
    """Raised when a threshold configuration is invalid."""


class InvalidScoreError(
    ThresholdError
):
    """Raised when a supplied score is invalid."""


# ============================================================================
# Decision Constants
# ============================================================================


class FraudDecision:
    """
    Business decision produced from fraud risk.
    """

    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class FraudRiskLevel:
    """
    Canonical fraud risk levels.

    These thresholds intentionally align with the
    fraud_service.py decision model.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ThresholdCategory:
    """
    Threshold categories.
    """

    MODEL = "MODEL"
    DECISION = "DECISION"
    TRANSACTION = "TRANSACTION"
    BEHAVIOR = "BEHAVIOR"
    MERCHANT = "MERCHANT"
    CONTEXT = "CONTEXT"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FraudThresholdConfig:
    """
    Centralized fraud threshold configuration.

    All percentages and scores use a 0-100 scale unless explicitly
    documented otherwise.

    Example:

        fraud_probability_threshold = 70

    means:

        70% fraud probability.
    """

    # ------------------------------------------------------------------------
    # Supervised model
    # ------------------------------------------------------------------------

    supervised_fraud_probability_threshold: float = 70.0

    supervised_high_probability_threshold: float = 80.0

    supervised_critical_probability_threshold: float = 90.0

    # ------------------------------------------------------------------------
    # Anomaly model
    # ------------------------------------------------------------------------

    anomaly_score_threshold: float = 70.0

    anomaly_high_score_threshold: float = 80.0

    anomaly_critical_score_threshold: float = 90.0

    # ------------------------------------------------------------------------
    # Final fraud score
    # ------------------------------------------------------------------------

    review_threshold: float = 50.0

    block_threshold: float = 80.0

    critical_threshold: float = 90.0

    # ------------------------------------------------------------------------
    # Transaction amount
    # ------------------------------------------------------------------------

    high_amount_threshold: float = 50_000.0

    critical_amount_threshold: float = 100_000.0

    extreme_amount_threshold: float = 500_000.0

    # ------------------------------------------------------------------------
    # Amount-to-average behavior
    # ------------------------------------------------------------------------

    amount_to_average_ratio_threshold: float = 2.0

    amount_to_average_high_ratio: float = 5.0

    amount_to_average_critical_ratio: float = 10.0

    # ------------------------------------------------------------------------
    # Amount-to-balance behavior
    # ------------------------------------------------------------------------

    amount_to_balance_ratio_threshold: float = 0.50

    amount_to_balance_high_ratio: float = 0.75

    amount_to_balance_critical_ratio: float = 1.00

    # ------------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------------

    velocity_review_threshold: float = 10.0

    velocity_high_threshold: float = 20.0

    velocity_critical_threshold: float = 30.0

    velocity_extreme_threshold: float = 50.0

    # ------------------------------------------------------------------------
    # Merchant risk
    # ------------------------------------------------------------------------

    merchant_risk_review_threshold: float = 60.0

    merchant_risk_high_threshold: float = 75.0

    merchant_risk_critical_threshold: float = 90.0

    # ------------------------------------------------------------------------
    # Behavioral signals
    # ------------------------------------------------------------------------

    suspicious_behavior_signal_threshold: int = 2

    high_behavior_signal_threshold: int = 3

    critical_behavior_signal_threshold: int = 4

    # ------------------------------------------------------------------------
    # Model agreement
    # ------------------------------------------------------------------------

    model_agreement_difference_threshold: float = 15.0

    model_disagreement_threshold: float = 30.0

    # ------------------------------------------------------------------------
    # Explainability / confidence
    # ------------------------------------------------------------------------

    minimum_confidence_for_auto_approval: float = 70.0

    minimum_confidence_for_auto_block: float = 80.0

    # ------------------------------------------------------------------------
    # Operational settings
    # ------------------------------------------------------------------------

    enabled: bool = True

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate(self) -> None:
        """
        Validate the complete threshold configuration.
        """

        percentage_fields = [
            (
                "supervised_fraud_probability_threshold",
                self.supervised_fraud_probability_threshold,
            ),
            (
                "supervised_high_probability_threshold",
                self.supervised_high_probability_threshold,
            ),
            (
                "supervised_critical_probability_threshold",
                self.supervised_critical_probability_threshold,
            ),
            (
                "anomaly_score_threshold",
                self.anomaly_score_threshold,
            ),
            (
                "anomaly_high_score_threshold",
                self.anomaly_high_score_threshold,
            ),
            (
                "anomaly_critical_score_threshold",
                self.anomaly_critical_score_threshold,
            ),
            (
                "review_threshold",
                self.review_threshold,
            ),
            (
                "block_threshold",
                self.block_threshold,
            ),
            (
                "critical_threshold",
                self.critical_threshold,
            ),
            (
                "merchant_risk_review_threshold",
                self.merchant_risk_review_threshold,
            ),
            (
                "merchant_risk_high_threshold",
                self.merchant_risk_high_threshold,
            ),
            (
                "merchant_risk_critical_threshold",
                self.merchant_risk_critical_threshold,
            ),
            (
                "minimum_confidence_for_auto_approval",
                self.minimum_confidence_for_auto_approval,
            ),
            (
                "minimum_confidence_for_auto_block",
                self.minimum_confidence_for_auto_block,
            ),
        ]

        for name, value in percentage_fields:

            self._validate_percentage(
                name,
                value,
            )

        # Ordered thresholds.

        self._validate_order(
            "supervised probability",
            self.supervised_fraud_probability_threshold,
            self.supervised_high_probability_threshold,
            self.supervised_critical_probability_threshold,
        )

        self._validate_order(
            "anomaly score",
            self.anomaly_score_threshold,
            self.anomaly_high_score_threshold,
            self.anomaly_critical_score_threshold,
        )

        self._validate_order(
            "final fraud score",
            self.review_threshold,
            self.block_threshold,
            self.critical_threshold,
        )

        self._validate_order(
            "merchant risk",
            self.merchant_risk_review_threshold,
            self.merchant_risk_high_threshold,
            self.merchant_risk_critical_threshold,
        )

        # Amount thresholds.

        amount_fields = [
            (
                "high_amount_threshold",
                self.high_amount_threshold,
            ),
            (
                "critical_amount_threshold",
                self.critical_amount_threshold,
            ),
            (
                "extreme_amount_threshold",
                self.extreme_amount_threshold,
            ),
            (
                "amount_to_average_ratio_threshold",
                self.amount_to_average_ratio_threshold,
            ),
            (
                "amount_to_average_high_ratio",
                self.amount_to_average_high_ratio,
            ),
            (
                "amount_to_average_critical_ratio",
                self.amount_to_average_critical_ratio,
            ),
            (
                "amount_to_balance_ratio_threshold",
                self.amount_to_balance_ratio_threshold,
            ),
            (
                "amount_to_balance_high_ratio",
                self.amount_to_balance_high_ratio,
            ),
            (
                "amount_to_balance_critical_ratio",
                self.amount_to_balance_critical_ratio,
            ),
        ]

        for name, value in amount_fields:

            self._validate_non_negative(
                name,
                value,
            )

        self._validate_order(
            "amount",
            self.high_amount_threshold,
            self.critical_amount_threshold,
            self.extreme_amount_threshold,
        )

        self._validate_order(
            "amount-to-average ratio",
            self.amount_to_average_ratio_threshold,
            self.amount_to_average_high_ratio,
            self.amount_to_average_critical_ratio,
        )

        self._validate_order(
            "amount-to-balance ratio",
            self.amount_to_balance_ratio_threshold,
            self.amount_to_balance_high_ratio,
            self.amount_to_balance_critical_ratio,
        )

        # Velocity.

        velocity_fields = [
            (
                "velocity_review_threshold",
                self.velocity_review_threshold,
            ),
            (
                "velocity_high_threshold",
                self.velocity_high_threshold,
            ),
            (
                "velocity_critical_threshold",
                self.velocity_critical_threshold,
            ),
            (
                "velocity_extreme_threshold",
                self.velocity_extreme_threshold,
            ),
        ]

        for name, value in velocity_fields:

            self._validate_non_negative(
                name,
                value,
            )

        self._validate_order(
            "velocity",
            self.velocity_review_threshold,
            self.velocity_high_threshold,
            self.velocity_critical_threshold,
            self.velocity_extreme_threshold,
        )

        # Behavioral thresholds.

        if (
            self.suspicious_behavior_signal_threshold
            < 1
        ):

            raise InvalidThresholdError(
                "suspicious_behavior_signal_threshold "
                "must be at least 1."
            )

        if (
            self.high_behavior_signal_threshold
            < self.suspicious_behavior_signal_threshold
        ):

            raise InvalidThresholdError(
                "high_behavior_signal_threshold must "
                "be >= suspicious_behavior_signal_threshold."
            )

        if (
            self.critical_behavior_signal_threshold
            < self.high_behavior_signal_threshold
        ):

            raise InvalidThresholdError(
                "critical_behavior_signal_threshold must "
                "be >= high_behavior_signal_threshold."
            )

        self._validate_percentage(
            "model_agreement_difference_threshold",
            self.model_agreement_difference_threshold,
        )

        self._validate_percentage(
            "model_disagreement_threshold",
            self.model_disagreement_threshold,
        )

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Return threshold configuration as a dictionary.
        """

        return {
            "supervised_fraud_probability_threshold":
                self.supervised_fraud_probability_threshold,

            "supervised_high_probability_threshold":
                self.supervised_high_probability_threshold,

            "supervised_critical_probability_threshold":
                self.supervised_critical_probability_threshold,

            "anomaly_score_threshold":
                self.anomaly_score_threshold,

            "anomaly_high_score_threshold":
                self.anomaly_high_score_threshold,

            "anomaly_critical_score_threshold":
                self.anomaly_critical_score_threshold,

            "review_threshold":
                self.review_threshold,

            "block_threshold":
                self.block_threshold,

            "critical_threshold":
                self.critical_threshold,

            "high_amount_threshold":
                self.high_amount_threshold,

            "critical_amount_threshold":
                self.critical_amount_threshold,

            "extreme_amount_threshold":
                self.extreme_amount_threshold,

            "amount_to_average_ratio_threshold":
                self.amount_to_average_ratio_threshold,

            "amount_to_average_high_ratio":
                self.amount_to_average_high_ratio,

            "amount_to_average_critical_ratio":
                self.amount_to_average_critical_ratio,

            "amount_to_balance_ratio_threshold":
                self.amount_to_balance_ratio_threshold,

            "amount_to_balance_high_ratio":
                self.amount_to_balance_high_ratio,

            "amount_to_balance_critical_ratio":
                self.amount_to_balance_critical_ratio,

            "velocity_review_threshold":
                self.velocity_review_threshold,

            "velocity_high_threshold":
                self.velocity_high_threshold,

            "velocity_critical_threshold":
                self.velocity_critical_threshold,

            "velocity_extreme_threshold":
                self.velocity_extreme_threshold,

            "merchant_risk_review_threshold":
                self.merchant_risk_review_threshold,

            "merchant_risk_high_threshold":
                self.merchant_risk_high_threshold,

            "merchant_risk_critical_threshold":
                self.merchant_risk_critical_threshold,

            "suspicious_behavior_signal_threshold":
                self.suspicious_behavior_signal_threshold,

            "high_behavior_signal_threshold":
                self.high_behavior_signal_threshold,

            "critical_behavior_signal_threshold":
                self.critical_behavior_signal_threshold,

            "model_agreement_difference_threshold":
                self.model_agreement_difference_threshold,

            "model_disagreement_threshold":
                self.model_disagreement_threshold,

            "minimum_confidence_for_auto_approval":
                self.minimum_confidence_for_auto_approval,

            "minimum_confidence_for_auto_block":
                self.minimum_confidence_for_auto_block,

            "enabled":
                self.enabled,

            "metadata":
                dict(self.metadata),
        }

    # ========================================================================
    # VALIDATION HELPERS
    # ========================================================================

    @staticmethod
    def _validate_percentage(
        name: str,
        value: float,
    ) -> None:

        try:

            number = float(value)

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidThresholdError(
                f"{name} must be numeric."
            ) from exc

        if not math.isfinite(
            number
        ):

            raise InvalidThresholdError(
                f"{name} must be finite."
            )

        if not (
            0.0
            <= number
            <= 100.0
        ):

            raise InvalidThresholdError(
                f"{name} must be between "
                "0 and 100."
            )

    @staticmethod
    def _validate_non_negative(
        name: str,
        value: float,
    ) -> None:

        try:

            number = float(value)

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidThresholdError(
                f"{name} must be numeric."
            ) from exc

        if not math.isfinite(
            number
        ):

            raise InvalidThresholdError(
                f"{name} must be finite."
            )

        if number < 0:

            raise InvalidThresholdError(
                f"{name} cannot be negative."
            )

    @staticmethod
    def _validate_order(
        name: str,
        *values: float,
    ) -> None:

        for index in range(
            len(values) - 1
        ):

            if (
                values[index]
                > values[index + 1]
            ):

                raise InvalidThresholdError(
                    f"{name} thresholds must "
                    "be ordered from low to high."
                )


# ============================================================================
# Threshold Evaluation Result
# ============================================================================


@dataclass
class ThresholdEvaluation:
    """
    Result of evaluating a score against the threshold policy.
    """

    score: float

    risk_level: str

    decision: str

    triggered: bool

    requires_review: bool

    requires_block: bool

    is_critical: bool

    matched_threshold: Optional[
        str
    ]

    reasons: list[str] = field(
        default_factory=list
    )

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "score":
                self.score,

            "risk_level":
                self.risk_level,

            "decision":
                self.decision,

            "triggered":
                self.triggered,

            "requires_review":
                self.requires_review,

            "requires_block":
                self.requires_block,

            "is_critical":
                self.is_critical,

            "matched_threshold":
                self.matched_threshold,

            "reasons":
                list(self.reasons),

            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Threshold Policy
# ============================================================================


class FraudThresholdPolicy:
    """
    Central business policy for fraud thresholds.

    This class should be used by:

        fraud_service.py
        scoring.py
        supervised_model.py
        anomaly_model.py
        alert generation
        fraud investigation workflows
    """

    def __init__(
        self,
        config: Optional[
            FraudThresholdConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or FraudThresholdConfig()
        )

        self.config.validate()

    # ========================================================================
    # FINAL FRAUD SCORE
    # ========================================================================

    def evaluate_score(
        self,
        score: float,
        confidence: Optional[
            float
        ] = None,
    ) -> ThresholdEvaluation:
        """
        Evaluate final fraud score.

        Score:
            0-100
        """

        score = self._validate_score(
            score
        )

        if (
            score
            >= self.config.critical_threshold
        ):

            risk_level = (
                FraudRiskLevel.CRITICAL
            )

            decision = (
                FraudDecision.BLOCK
            )

            matched = (
                "critical_threshold"
            )

            reasons = [
                "Fraud score reached "
                "the critical threshold."
            ]

            is_critical = True

        elif (
            score
            >= self.config.block_threshold
        ):

            risk_level = (
                FraudRiskLevel.HIGH
            )

            decision = (
                FraudDecision.BLOCK
            )

            matched = (
                "block_threshold"
            )

            reasons = [
                "Fraud score reached "
                "the block threshold."
            ]

            is_critical = False

        elif (
            score
            >= self.config.review_threshold
        ):

            risk_level = (
                FraudRiskLevel.MEDIUM
            )

            decision = (
                FraudDecision.REVIEW
            )

            matched = (
                "review_threshold"
            )

            reasons = [
                "Fraud score reached "
                "the manual-review threshold."
            ]

            is_critical = False

        else:

            risk_level = (
                FraudRiskLevel.LOW
            )

            decision = (
                FraudDecision.APPROVE
            )

            matched = None

            reasons = [
                "Fraud score is below "
                "the review threshold."
            ]

            is_critical = False

        requires_review = (
            decision
            == FraudDecision.REVIEW
            or decision
            == FraudDecision.BLOCK
        )

        requires_block = (
            decision
            == FraudDecision.BLOCK
        )

        if (
            confidence is not None
            and self._normalize_score(
                confidence
            )
            < self.config
            .minimum_confidence_for_auto_approval
            and decision
            == FraudDecision.APPROVE
        ):

            decision = (
                FraudDecision.REVIEW
            )

            risk_level = (
                FraudRiskLevel.MEDIUM
            )

            requires_review = True

            matched = (
                "minimum_confidence_for_auto_approval"
            )

            reasons.append(
                "Model confidence is too low "
                "for automatic approval."
            )

        return ThresholdEvaluation(
            score=score,
            risk_level=risk_level,
            decision=decision,
            triggered=(
                score
                >= self.config.review_threshold
            ),
            requires_review=requires_review,
            requires_block=requires_block,
            is_critical=is_critical,
            matched_threshold=matched,
            reasons=reasons,
            metadata={
                "review_threshold":
                    self.config.review_threshold,

                "block_threshold":
                    self.config.block_threshold,

                "critical_threshold":
                    self.config.critical_threshold,
            },
        )

    # ========================================================================
    # RISK LEVEL
    # ========================================================================

    def risk_level(
        self,
        score: float,
    ) -> str:
        """
        Return canonical risk level.
        """

        score = self._validate_score(
            score
        )

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

    # ========================================================================
    # DECISION
    # ========================================================================

    def decision(
        self,
        score: float,
        confidence: Optional[
            float
        ] = None,
    ) -> str:
        """
        Return APPROVE, REVIEW, or BLOCK.
        """

        return self.evaluate_score(
            score,
            confidence=confidence,
        ).decision

    # ========================================================================
    # SUPERVISED MODEL
    # ========================================================================

    def evaluate_supervised_probability(
        self,
        probability: float,
    ) -> Dict[str, Any]:
        """
        Evaluate supervised-model fraud probability.

        Input accepts either:
            0-1
        or:
            0-100
        """

        value = self._normalize_score(
            probability
        )

        if (
            value
            >= self.config
            .supervised_critical_probability_threshold
        ):

            level = FraudRiskLevel.CRITICAL

        elif (
            value
            >= self.config
            .supervised_high_probability_threshold
        ):

            level = FraudRiskLevel.HIGH

        elif (
            value
            >= self.config
            .supervised_fraud_probability_threshold
        ):

            level = FraudRiskLevel.MEDIUM

        else:

            level = FraudRiskLevel.LOW

        return {
            "fraud_probability":
                value,

            "risk_level":
                level,

            "triggered":
                value
                >= self.config
                .supervised_fraud_probability_threshold,

            "high_risk":
                value
                >= self.config
                .supervised_high_probability_threshold,

            "critical":
                value
                >= self.config
                .supervised_critical_probability_threshold,
        }

    # ========================================================================
    # ANOMALY MODEL
    # ========================================================================

    def evaluate_anomaly_score(
        self,
        score: float,
    ) -> Dict[str, Any]:
        """
        Evaluate anomaly score.
        """

        value = self._normalize_score(
            score
        )

        if (
            value
            >= self.config
            .anomaly_critical_score_threshold
        ):

            level = (
                FraudRiskLevel.CRITICAL
            )

        elif (
            value
            >= self.config
            .anomaly_high_score_threshold
        ):

            level = (
                FraudRiskLevel.HIGH
            )

        elif (
            value
            >= self.config
            .anomaly_score_threshold
        ):

            level = (
                FraudRiskLevel.MEDIUM
            )

        else:

            level = (
                FraudRiskLevel.LOW
            )

        return {
            "anomaly_score":
                value,

            "risk_level":
                level,

            "triggered":
                value
                >= self.config
                .anomaly_score_threshold,

            "high_risk":
                value
                >= self.config
                .anomaly_high_score_threshold,

            "critical":
                value
                >= self.config
                .anomaly_critical_score_threshold,
        }

    # ========================================================================
    # TRANSACTION AMOUNT
    # ========================================================================

    def evaluate_amount(
        self,
        amount: float,
    ) -> Dict[str, Any]:
        """
        Evaluate absolute transaction amount.
        """

        amount = self._validate_non_negative_value(
            amount,
            "amount",
        )

        if (
            amount
            >= self.config
            .extreme_amount_threshold
        ):

            level = (
                FraudRiskLevel.CRITICAL
            )

        elif (
            amount
            >= self.config
            .critical_amount_threshold
        ):

            level = (
                FraudRiskLevel.HIGH
            )

        elif (
            amount
            >= self.config
            .high_amount_threshold
        ):

            level = (
                FraudRiskLevel.MEDIUM
            )

        else:

            level = (
                FraudRiskLevel.LOW
            )

        return {
            "amount":
                amount,

            "risk_level":
                level,

            "high_amount":
                amount
                >= self.config
                .high_amount_threshold,

            "critical_amount":
                amount
                >= self.config
                .critical_amount_threshold,

            "extreme_amount":
                amount
                >= self.config
                .extreme_amount_threshold,
        }

    # ========================================================================
    # AMOUNT / AVERAGE
    # ========================================================================

    def evaluate_amount_to_average(
        self,
        ratio: float,
    ) -> Dict[str, Any]:
        """
        Evaluate transaction amount relative to
        historical average.
        """

        ratio = self._validate_non_negative_value(
            ratio,
            "amount_to_average_ratio",
        )

        if (
            ratio
            >= self.config
            .amount_to_average_critical_ratio
        ):

            level = (
                FraudRiskLevel.CRITICAL
            )

        elif (
            ratio
            >= self.config
            .amount_to_average_high_ratio
        ):

            level = (
                FraudRiskLevel.HIGH
            )

        elif (
            ratio
            >= self.config
            .amount_to_average_ratio_threshold
        ):

            level = (
                FraudRiskLevel.MEDIUM
            )

        else:

            level = (
                FraudRiskLevel.LOW
            )

        return {
            "ratio":
                ratio,

            "risk_level":
                level,

            "triggered":
                ratio
                >= self.config
                .amount_to_average_ratio_threshold,
        }

    # ========================================================================
    # AMOUNT / BALANCE
    # ========================================================================

    def evaluate_amount_to_balance(
        self,
        ratio: float,
    ) -> Dict[str, Any]:
        """
        Evaluate transaction amount relative to
        available account balance.
        """

        ratio = self._validate_non_negative_value(
            ratio,
            "amount_to_balance_ratio",
        )

        if (
            ratio
            >= self.config
            .amount_to_balance_critical_ratio
        ):

            level = (
                FraudRiskLevel.CRITICAL
            )

        elif (
            ratio
            >= self.config
            .amount_to_balance_high_ratio
        ):

            level = (
                FraudRiskLevel.HIGH
            )

        elif (
            ratio
            >= self.config
            .amount_to_balance_ratio_threshold
        ):

            level = (
                FraudRiskLevel.MEDIUM
            )

        else:

            level = (
                FraudRiskLevel.LOW
            )

        return {
            "ratio":
                ratio,

            "risk_level":
                level,

            "triggered":
                ratio
                >= self.config
                .amount_to_balance_ratio_threshold,
        }

    # ========================================================================
    # VELOCITY
    # ========================================================================

    def evaluate_velocity(
        self,
        velocity: float,
    ) -> Dict[str, Any]:
        """
        Evaluate transaction velocity.
        """

        velocity = self._validate_non_negative_value(
            velocity,
            "velocity",
        )

        if (
            velocity
            >= self.config
            .velocity_extreme_threshold
        ):

            level = (
                FraudRiskLevel.CRITICAL
            )

        elif (
            velocity
            >= self.config
            .velocity_critical_threshold
        ):

            level = (
                FraudRiskLevel.HIGH
            )

        elif (
            velocity
            >= self.config
            .velocity_high_threshold
        ):

            level = (
                FraudRiskLevel.MEDIUM
            )

        elif (
            velocity
            >= self.config
            .velocity_review_threshold
        ):

            level = (
                FraudRiskLevel.MEDIUM
            )

        else:

            level = (
                FraudRiskLevel.LOW
            )

        return {
            "velocity":
                velocity,

            "risk_level":
                level,

            "triggered":
                velocity
                >= self.config
                .velocity_review_threshold,

            "high_risk":
                velocity
                >= self.config
                .velocity_high_threshold,

            "critical":
                velocity
                >= self.config
                .velocity_critical_threshold,
        }

    # ========================================================================
    # MERCHANT RISK
    # ========================================================================

    def evaluate_merchant_risk(
        self,
        merchant_risk_score: float,
    ) -> Dict[str, Any]:
        """
        Evaluate merchant risk score.
        """

        score = self._validate_score(
            merchant_risk_score
        )

        if (
            score
            >= self.config
            .merchant_risk_critical_threshold
        ):

            level = (
                FraudRiskLevel.CRITICAL
            )

        elif (
            score
            >= self.config
            .merchant_risk_high_threshold
        ):

            level = (
                FraudRiskLevel.HIGH
            )

        elif (
            score
            >= self.config
            .merchant_risk_review_threshold
        ):

            level = (
                FraudRiskLevel.MEDIUM
            )

        else:

            level = (
                FraudRiskLevel.LOW
            )

        return {
            "merchant_risk_score":
                score,

            "risk_level":
                level,

            "triggered":
                score
                >= self.config
                .merchant_risk_review_threshold,

            "high_risk":
                score
                >= self.config
                .merchant_risk_high_threshold,

            "critical":
                score
                >= self.config
                .merchant_risk_critical_threshold,
        }

    # ========================================================================
    # BEHAVIORAL SIGNALS
    # ========================================================================

    def evaluate_behavior_signals(
        self,
        signal_count: int,
    ) -> Dict[str, Any]:
        """
        Evaluate the number of suspicious behavioral signals.
        """

        try:

            count = int(
                signal_count
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidThresholdError(
                "signal_count must be an integer."
            ) from exc

        if count < 0:

            raise InvalidThresholdError(
                "signal_count cannot be negative."
            )

        if (
            count
            >= self.config
            .critical_behavior_signal_threshold
        ):

            level = (
                FraudRiskLevel.CRITICAL
            )

        elif (
            count
            >= self.config
            .high_behavior_signal_threshold
        ):

            level = (
                FraudRiskLevel.HIGH
            )

        elif (
            count
            >= self.config
            .suspicious_behavior_signal_threshold
        ):

            level = (
                FraudRiskLevel.MEDIUM
            )

        else:

            level = (
                FraudRiskLevel.LOW
            )

        return {
            "signal_count":
                count,

            "risk_level":
                level,

            "triggered":
                count
                >= self.config
                .suspicious_behavior_signal_threshold,

            "high_risk":
                count
                >= self.config
                .high_behavior_signal_threshold,

            "critical":
                count
                >= self.config
                .critical_behavior_signal_threshold,
        }

    # ========================================================================
    # MODEL AGREEMENT
    # ========================================================================

    def evaluate_model_agreement(
        self,
        anomaly_score: Optional[
            float
        ],
        fraud_probability: Optional[
            float
        ],
    ) -> Dict[str, Any]:
        """
        Evaluate agreement/disagreement between anomaly
        and supervised models.
        """

        if (
            anomaly_score is None
            or fraud_probability is None
        ):

            return {
                "available":
                    False,

                "agreement":
                    None,

                "difference":
                    None,

                "high_disagreement":
                    False,
            }

        anomaly = self._normalize_score(
            anomaly_score
        )

        supervised = self._normalize_score(
            fraud_probability
        )

        difference = abs(
            anomaly
            - supervised
        )

        agreement = (
            difference
            <= self.config
            .model_agreement_difference_threshold
        )

        high_disagreement = (
            difference
            >= self.config
            .model_disagreement_threshold
        )

        return {
            "available":
                True,

            "agreement":
                agreement,

            "difference":
                difference,

            "high_disagreement":
                high_disagreement,

            "anomaly_score":
                anomaly,

            "fraud_probability":
                supervised,
        }

    # ========================================================================
    # TRANSACTION CONTEXT
    # ========================================================================

    def evaluate_transaction(
        self,
        transaction: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate all available transaction-level thresholds.

        This does not calculate a final fraud score.
        It only evaluates individual threshold signals.
        """

        if not isinstance(
            transaction,
            Mapping,
        ):

            raise InvalidThresholdError(
                "transaction must be a mapping."
            )

        results: Dict[
            str,
            Any
        ] = {}

        if "amount" in transaction:

            results["amount"] = (
                self.evaluate_amount(
                    transaction["amount"]
                )
            )

        if (
            "amount_to_average_ratio"
            in transaction
        ):

            results[
                "amount_to_average"
            ] = (
                self.evaluate_amount_to_average(
                    transaction[
                        "amount_to_average_ratio"
                    ]
                )
            )

        if (
            "amount_to_balance_ratio"
            in transaction
        ):

            results[
                "amount_to_balance"
            ] = (
                self.evaluate_amount_to_balance(
                    transaction[
                        "amount_to_balance_ratio"
                    ]
                )
            )

        if "velocity" in transaction:

            results["velocity"] = (
                self.evaluate_velocity(
                    transaction[
                        "velocity"
                    ]
                )
            )

        if (
            "merchant_risk_score"
            in transaction
        ):

            results[
                "merchant_risk"
            ] = (
                self.evaluate_merchant_risk(
                    transaction[
                        "merchant_risk_score"
                    ]
                )
            )

        if (
            "suspicious_behavior_count"
            in transaction
        ):

            results[
                "behavior"
            ] = (
                self.evaluate_behavior_signals(
                    transaction[
                        "suspicious_behavior_count"
                    ]
                )
            )

        return results

    # ========================================================================
    # SUMMARY
    # ========================================================================

    def summarize(
        self,
        fraud_score: Optional[
            float
        ] = None,
        anomaly_score: Optional[
            float
        ] = None,
        fraud_probability: Optional[
            float
        ] = None,
        transaction: Optional[
            Mapping[str, Any]
        ] = None,
        confidence: Optional[
            float
        ] = None,
    ) -> Dict[str, Any]:
        """
        Produce one consolidated threshold evaluation.
        """

        result: Dict[
            str,
            Any
        ] = {}

        if fraud_score is not None:

            result[
                "final_score"
            ] = self.evaluate_score(
                fraud_score,
                confidence=confidence,
            ).to_dict()

        if anomaly_score is not None:

            result[
                "anomaly"
            ] = self.evaluate_anomaly_score(
                anomaly_score
            )

        if fraud_probability is not None:

            result[
                "supervised"
            ] = (
                self.evaluate_supervised_probability(
                    fraud_probability
                )
            )

        if (
            anomaly_score is not None
            and fraud_probability is not None
        ):

            result[
                "model_agreement"
            ] = (
                self.evaluate_model_agreement(
                    anomaly_score,
                    fraud_probability,
                )
            )

        if transaction is not None:

            result[
                "transaction"
            ] = self.evaluate_transaction(
                transaction
            )

        return result

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    @staticmethod
    def _validate_score(
        score: float,
    ) -> float:

        try:

            value = float(
                score
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidScoreError(
                "Score must be numeric."
            ) from exc

        if not math.isfinite(
            value
        ):

            raise InvalidScoreError(
                "Score must be finite."
            )

        if not (
            0.0
            <= value
            <= 100.0
        ):

            raise InvalidScoreError(
                "Score must be between "
                "0 and 100."
            )

        return value

    @staticmethod
    def _normalize_score(
        value: float,
    ) -> float:

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidScoreError(
                "Score must be numeric."
            ) from exc

        if not math.isfinite(
            number
        ):

            raise InvalidScoreError(
                "Score must be finite."
            )

        if (
            0.0
            <= number
            <= 1.0
        ):

            number *= 100.0

        return max(
            0.0,
            min(
                100.0,
                number,
            ),
        )

    @staticmethod
    def _validate_non_negative_value(
        value: float,
        name: str,
    ) -> float:

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidThresholdError(
                f"{name} must be numeric."
            ) from exc

        if not math.isfinite(
            number
        ):

            raise InvalidThresholdError(
                f"{name} must be finite."
            )

        if number < 0:

            raise InvalidThresholdError(
                f"{name} cannot be negative."
            )

        return number


# ============================================================================
# Default Policy
# ============================================================================


DEFAULT_FRAUD_THRESHOLD_POLICY = (
    FraudThresholdPolicy()
)


# ============================================================================
# Convenience Functions
# ============================================================================


def get_risk_level(
    score: float,
) -> str:
    """
    Return risk level using the default policy.
    """

    return (
        DEFAULT_FRAUD_THRESHOLD_POLICY
        .risk_level(score)
    )


def get_fraud_decision(
    score: float,
    confidence: Optional[
        float
    ] = None,
) -> str:
    """
    Return APPROVE, REVIEW, or BLOCK.
    """

    return (
        DEFAULT_FRAUD_THRESHOLD_POLICY
        .decision(
            score,
            confidence=confidence,
        )
    )


def evaluate_fraud_score(
    score: float,
    confidence: Optional[
        float
    ] = None,
) -> Dict[str, Any]:
    """
    Evaluate a final fraud score using the default policy.
    """

    return (
        DEFAULT_FRAUD_THRESHOLD_POLICY
        .evaluate_score(
            score,
            confidence=confidence,
        )
        .to_dict()
    )


def evaluate_fraud_probability(
    probability: float,
) -> Dict[str, Any]:
    """
    Evaluate supervised fraud probability.
    """

    return (
        DEFAULT_FRAUD_THRESHOLD_POLICY
        .evaluate_supervised_probability(
            probability
        )
    )


def evaluate_anomaly(
    score: float,
) -> Dict[str, Any]:
    """
    Evaluate anomaly score.
    """

    return (
        DEFAULT_FRAUD_THRESHOLD_POLICY
        .evaluate_anomaly_score(
            score
        )
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "ThresholdError",
    "InvalidThresholdError",
    "InvalidScoreError",
    "FraudDecision",
    "FraudRiskLevel",
    "ThresholdCategory",
    "FraudThresholdConfig",
    "ThresholdEvaluation",
    "FraudThresholdPolicy",
    "DEFAULT_FRAUD_THRESHOLD_POLICY",
    "get_risk_level",
    "get_fraud_decision",
    "evaluate_fraud_score",
    "evaluate_fraud_probability",
    "evaluate_anomaly",
]