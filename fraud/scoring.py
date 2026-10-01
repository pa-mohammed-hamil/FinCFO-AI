"""
FinCo AI - Fraud Risk Scoring

File:
    backend/app/fraud/scoring.py

Purpose:
    Combine fraud detection signals into a normalized 0-100
    fraud risk score.

Pipeline:

    Feature Engineering
           |
           +------------------+
           |                  |
           v                  v
      Anomaly Model     Supervised Model
           |                  |
           +--------+---------+
                    |
                    v
              FraudScorer
                    |
                    v
            Fraud Risk Score
                 0 - 100
                    |
          +---------+---------+
          |                   |
          v                   v
       Decision          Explainability
          |
          v
        Alert

Design:
    - Deterministic
    - Model-agnostic
    - No LLM dependency
    - Explainable
    - Configurable weights
    - Supports 0-1 and 0-100 probabilities
    - Safe for online and batch inference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence
import math


# ============================================================================
# Exceptions
# ============================================================================


class FraudScoringError(Exception):
    """Base exception for fraud scoring."""


class InvalidScoreInputError(
    FraudScoringError
):
    """Raised when scoring input is invalid."""


class InvalidWeightError(
    FraudScoringError
):
    """Raised when scoring weights are invalid."""


# ============================================================================
# Constants
# ============================================================================


class ScoreBand:
    """Fraud score bands."""

    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ScoreSignal:
    """Supported scoring signal names."""

    ANOMALY = "anomaly"
    SUPERVISED = "supervised"
    AMOUNT = "amount"
    VELOCITY = "velocity"
    BALANCE_RATIO = "balance_ratio"
    MERCHANT_RISK = "merchant_risk"
    LOCATION = "location"
    DEVICE = "device"
    BEHAVIOR = "behavior"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FraudScoringConfig:
    """
    Configuration for the fraud scoring engine.

    Model signals receive the highest weights by default.
    Behavioral signals provide additional contextual risk.
    """

    anomaly_weight: float = 0.40

    supervised_weight: float = 0.60

    amount_weight: float = 0.05

    velocity_weight: float = 0.05

    balance_ratio_weight: float = 0.05

    merchant_risk_weight: float = 0.05

    location_weight: float = 0.05

    device_weight: float = 0.05

    behavior_weight: float = 0.05

    very_low_threshold: float = 20.0

    low_threshold: float = 40.0

    medium_threshold: float = 60.0

    high_threshold: float = 80.0

    critical_threshold: float = 90.0

    anomaly_override_threshold: float = 95.0

    supervised_override_threshold: float = 95.0

    clamp_output: bool = True

    def validate(self) -> None:
        """
        Validate configuration.
        """

        weights = {
            "anomaly_weight":
                self.anomaly_weight,

            "supervised_weight":
                self.supervised_weight,

            "amount_weight":
                self.amount_weight,

            "velocity_weight":
                self.velocity_weight,

            "balance_ratio_weight":
                self.balance_ratio_weight,

            "merchant_risk_weight":
                self.merchant_risk_weight,

            "location_weight":
                self.location_weight,

            "device_weight":
                self.device_weight,

            "behavior_weight":
                self.behavior_weight,
        }

        for name, weight in weights.items():

            if (
                not math.isfinite(
                    float(weight)
                )
                or weight < 0
            ):

                raise InvalidWeightError(
                    f"{name} must be a "
                    "finite non-negative number."
                )

        model_weight = (
            self.anomaly_weight
            + self.supervised_weight
        )

        if model_weight <= 0:

            raise InvalidWeightError(
                "At least one model weight "
                "must be greater than zero."
            )

        thresholds = (
            self.very_low_threshold,
            self.low_threshold,
            self.medium_threshold,
            self.high_threshold,
            self.critical_threshold,
        )

        if any(
            not math.isfinite(
                float(value)
            )
            for value in thresholds
        ):

            raise InvalidScoreInputError(
                "Score thresholds must be finite."
            )

        if not (
            0 <= self.very_low_threshold
            <= self.low_threshold
            <= self.medium_threshold
            <= self.high_threshold
            <= self.critical_threshold
            <= 100
        ):

            raise InvalidScoreInputError(
                "Score thresholds must be "
                "monotonically increasing "
                "between 0 and 100."
            )


# ============================================================================
# Score Contribution
# ============================================================================


@dataclass
class ScoreContribution:
    """
    Contribution of one signal to the final score.
    """

    signal: str

    raw_value: float

    normalized_value: float

    weight: float

    contribution: float

    reason: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "signal":
                self.signal,

            "raw_value":
                self.raw_value,

            "normalized_value":
                self.normalized_value,

            "weight":
                self.weight,

            "contribution":
                self.contribution,

            "reason":
                self.reason,

            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Score Result
# ============================================================================


@dataclass
class FraudScoreResult:
    """
    Complete scoring result.
    """

    fraud_score: float

    risk_band: str

    confidence: float

    contributions: list[
        ScoreContribution
    ] = field(
        default_factory=list
    )

    reasons: list[str] = field(
        default_factory=list
    )

    model_scores: Dict[
        str,
        float
    ] = field(
        default_factory=dict
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
            "fraud_score":
                self.fraud_score,

            "risk_band":
                self.risk_band,

            "confidence":
                self.confidence,

            "contributions": [
                item.to_dict()
                for item
                in self.contributions
            ],

            "reasons":
                list(self.reasons),

            "model_scores":
                dict(self.model_scores),

            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Main Scoring Engine
# ============================================================================


class FraudScorer:
    """
    Deterministic fraud risk scoring engine.

    Main responsibilities:

        1. Normalize model outputs.
        2. Extract contextual transaction signals.
        3. Apply configurable weights.
        4. Produce 0-100 fraud score.
        5. Assign risk band.
        6. Generate explainable contributions.
        7. Estimate scoring confidence.

    The scorer does NOT:

        - train models
        - persist results
        - create alerts
        - send notifications
        - call an LLM
    """

    def __init__(
        self,
        config: Optional[
            FraudScoringConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or FraudScoringConfig()
        )

        self.config.validate()

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def calculate(
        self,
        features: Optional[
            Mapping[str, Any]
        ] = None,
        anomaly_result: Optional[
            Mapping[str, Any]
        ] = None,
        supervised_result: Optional[
            Mapping[str, Any]
        ] = None,
        transaction: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Calculate fraud score.

        Returns a dictionary so it integrates directly with
        FraudService._calculate_score().
        """

        result = self.score(
            features=features,
            anomaly_result=anomaly_result,
            supervised_result=supervised_result,
            transaction=transaction,
        )

        return result.to_dict()

    def calculate_score(
        self,
        features: Optional[
            Mapping[str, Any]
        ] = None,
        anomaly_result: Optional[
            Mapping[str, Any]
        ] = None,
        supervised_result: Optional[
            Mapping[str, Any]
        ] = None,
        transaction: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Compatibility alias for calculate().
        """

        return self.calculate(
            features=features,
            anomaly_result=anomaly_result,
            supervised_result=supervised_result,
            transaction=transaction,
        )

    def score(
        self,
        features: Optional[
            Mapping[str, Any]
        ] = None,
        anomaly_result: Optional[
            Mapping[str, Any]
        ] = None,
        supervised_result: Optional[
            Mapping[str, Any]
        ] = None,
        transaction: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> FraudScoreResult:
        """
        Calculate the final fraud risk score.
        """

        features = dict(
            features or {}
        )

        anomaly_result = dict(
            anomaly_result or {}
        )

        supervised_result = dict(
            supervised_result or {}
        )

        transaction = dict(
            transaction or {}
        )

        anomaly_score = (
            self._extract_anomaly_score(
                anomaly_result
            )
        )

        supervised_score = (
            self._extract_supervised_score(
                supervised_result
            )
        )

        contributions: list[
            ScoreContribution
        ] = []

        # ----------------------------------------------------------------
        # Model signals
        # ----------------------------------------------------------------

        if anomaly_result:

            contributions.append(
                self._model_contribution(
                    signal=ScoreSignal.ANOMALY,
                    value=anomaly_score,
                    weight=(
                        self.config.anomaly_weight
                    ),
                    reason=(
                        "Anomaly model identified "
                        "unusual transaction behavior."
                        if anomaly_score >= 50
                        else
                        "Anomaly model identified "
                        "limited unusual behavior."
                    ),
                )
            )

        if supervised_result:

            contributions.append(
                self._model_contribution(
                    signal=ScoreSignal.SUPERVISED,
                    value=supervised_score,
                    weight=(
                        self.config.supervised_weight
                    ),
                    reason=(
                        "Supervised fraud model "
                        "estimated elevated fraud probability."
                        if supervised_score >= 50
                        else
                        "Supervised fraud model "
                        "estimated relatively low fraud probability."
                    ),
                )
            )

        # ----------------------------------------------------------------
        # Behavioral/contextual signals
        # ----------------------------------------------------------------

        contextual = (
            self._contextual_contributions(
                features=features,
                transaction=transaction,
            )
        )

        contributions.extend(
            contextual
        )

        # ----------------------------------------------------------------
        # Calculate weighted score
        # ----------------------------------------------------------------

        score = (
            self._weighted_score(
                contributions
            )
        )

        # ----------------------------------------------------------------
        # Override high-confidence model signals
        # ----------------------------------------------------------------

        score = (
            self._apply_model_overrides(
                score,
                anomaly_score,
                supervised_score,
            )
        )

        score = self._clamp(
            score
        )

        risk_band = (
            self.risk_band(
                score
            )
        )

        reasons = (
            self._top_reasons(
                contributions
            )
        )

        confidence = (
            self._calculate_confidence(
                anomaly_result=anomaly_result,
                supervised_result=supervised_result,
                score=score,
            )
        )

        return FraudScoreResult(
            fraud_score=round(
                score,
                4,
            ),

            risk_band=risk_band,

            confidence=round(
                confidence,
                4,
            ),

            contributions=contributions,

            reasons=reasons,

            model_scores={
                "anomaly_score":
                    round(
                        anomaly_score,
                        4,
                    ),

                "fraud_probability":
                    round(
                        supervised_score,
                        4,
                    ),
            },

            metadata={
                "scoring_method":
                    "weighted_signal_aggregation",

                "version":
                    "1.0",
            },
        )

    # ========================================================================
    # MODEL SCORE EXTRACTION
    # ========================================================================

    def _extract_anomaly_score(
        self,
        result: Mapping[str, Any],
    ) -> float:

        value = self._first_value(
            result,
            (
                "anomaly_score",
                "score",
                "risk_score",
            ),
        )

        return self._normalize_score(
            value
        )

    def _extract_supervised_score(
        self,
        result: Mapping[str, Any],
    ) -> float:

        value = self._first_value(
            result,
            (
                "fraud_probability",
                "probability",
                "fraud_score",
                "risk_score",
                "score",
            ),
        )

        return self._normalize_probability(
            value
        )

    # ========================================================================
    # CONTEXTUAL SIGNALS
    # ========================================================================

    def _contextual_contributions(
        self,
        features: Mapping[str, Any],
        transaction: Mapping[str, Any],
    ) -> list[
        ScoreContribution
    ]:

        contributions: list[
            ScoreContribution
        ] = []

        # --------------------------------------------------------------
        # Amount
        # --------------------------------------------------------------

        amount = self._number(
            features.get(
                "amount",
                transaction.get(
                    "amount"
                ),
            )
        )

        amount_signal = (
            self._amount_risk(
                amount,
                features,
                transaction,
            )
        )

        contributions.append(
            self._context_contribution(
                signal=ScoreSignal.AMOUNT,
                value=amount_signal,
                weight=(
                    self.config.amount_weight
                ),
                reason=self._amount_reason(
                    amount_signal
                ),
            )
        )

        # --------------------------------------------------------------
        # Velocity
        # --------------------------------------------------------------

        velocity = self._number(
            features.get(
                "velocity",
                transaction.get(
                    "velocity"
                ),
            )
        )

        velocity_signal = (
            self._velocity_risk(
                velocity
            )
        )

        contributions.append(
            self._context_contribution(
                signal=ScoreSignal.VELOCITY,
                value=velocity_signal,
                weight=(
                    self.config.velocity_weight
                ),
                reason=self._velocity_reason(
                    velocity_signal
                ),
            )
        )

        # --------------------------------------------------------------
        # Balance ratio
        # --------------------------------------------------------------

        ratio = self._number(
            features.get(
                "amount_to_balance_ratio"
            )
        )

        ratio_signal = (
            self._ratio_risk(
                ratio
            )
        )

        contributions.append(
            self._context_contribution(
                signal=ScoreSignal.BALANCE_RATIO,
                value=ratio_signal,
                weight=(
                    self.config.balance_ratio_weight
                ),
                reason=self._ratio_reason(
                    ratio_signal
                ),
            )
        )

        # --------------------------------------------------------------
        # Merchant risk
        # --------------------------------------------------------------

        merchant_risk = self._number(
            features.get(
                "merchant_risk_score",
                transaction.get(
                    "merchant_risk_score"
                ),
            )
        )

        merchant_signal = (
            self._clamp(
                merchant_risk
            )
        )

        contributions.append(
            self._context_contribution(
                signal=ScoreSignal.MERCHANT_RISK,
                value=merchant_signal,
                weight=(
                    self.config.merchant_risk_weight
                ),
                reason=(
                    "Merchant risk indicator "
                    "is elevated."
                    if merchant_signal >= 60
                    else
                    "Merchant risk indicator "
                    "is not elevated."
                ),
            )
        )

        # --------------------------------------------------------------
        # Location
        # --------------------------------------------------------------

        location_signal = (
            self._location_risk(
                features,
                transaction,
            )
        )

        contributions.append(
            self._context_contribution(
                signal=ScoreSignal.LOCATION,
                value=location_signal,
                weight=(
                    self.config.location_weight
                ),
                reason=(
                    "Transaction originated "
                    "from an unusual location."
                    if location_signal >= 60
                    else
                    "No strong location anomaly "
                    "was identified."
                ),
            )
        )

        # --------------------------------------------------------------
        # Device
        # --------------------------------------------------------------

        device_signal = (
            self._device_risk(
                features,
                transaction,
            )
        )

        contributions.append(
            self._context_contribution(
                signal=ScoreSignal.DEVICE,
                value=device_signal,
                weight=(
                    self.config.device_weight
                ),
                reason=(
                    "Transaction used a "
                    "new or unusual device."
                    if device_signal >= 60
                    else
                    "No strong device anomaly "
                    "was identified."
                ),
            )
        )

        # --------------------------------------------------------------
        # Behavioral
        # --------------------------------------------------------------

        behavior_signal = (
            self._behavior_risk(
                features,
                transaction,
            )
        )

        contributions.append(
            self._context_contribution(
                signal=ScoreSignal.BEHAVIOR,
                value=behavior_signal,
                weight=(
                    self.config.behavior_weight
                ),
                reason=(
                    "Multiple behavioral "
                    "risk indicators are present."
                    if behavior_signal >= 60
                    else
                    "Behavioral risk indicators "
                    "are relatively normal."
                ),
            )
        )

        return contributions

    # ========================================================================
    # CONTEXT RISK CALCULATORS
    # ========================================================================

    def _amount_risk(
        self,
        amount: float,
        features: Mapping[str, Any],
        transaction: Mapping[str, Any],
    ) -> float:

        zscore = self._number(
            features.get(
                "amount_zscore"
            )
        )

        if zscore > 0:

            return self._clamp(
                zscore * 20
            )

        average = self._number(
            features.get(
                "amount_to_avg_ratio"
            )
        )

        if average > 0:

            return self._clamp(
                (average - 1)
                * 50
            )

        # No historical baseline:
        # use conservative absolute amount
        # buckets rather than assuming fraud.

        if amount >= 100000:
            return 90.0

        if amount >= 50000:
            return 75.0

        if amount >= 10000:
            return 55.0

        if amount >= 5000:
            return 35.0

        return 10.0

    @staticmethod
    def _velocity_risk(
        velocity: float,
    ) -> float:

        if velocity >= 50:
            return 100.0

        if velocity >= 30:
            return 90.0

        if velocity >= 20:
            return 75.0

        if velocity >= 10:
            return 55.0

        if velocity >= 5:
            return 35.0

        return 10.0

    @staticmethod
    def _ratio_risk(
        ratio: float,
    ) -> float:

        if ratio >= 5:
            return 100.0

        if ratio >= 2:
            return 90.0

        if ratio >= 1:
            return 75.0

        if ratio >= 0.75:
            return 60.0

        if ratio >= 0.50:
            return 40.0

        if ratio > 0:
            return 20.0

        return 0.0

    def _location_risk(
        self,
        features: Mapping[str, Any],
        transaction: Mapping[str, Any],
    ) -> float:

        new_location = self._boolean(
            features.get(
                "new_location",
                transaction.get(
                    "new_location"
                ),
            )
        )

        international = self._boolean(
            features.get(
                "is_international",
                transaction.get(
                    "is_international"
                ),
            )
        )

        score = 0.0

        if new_location:
            score += 65.0

        if international:
            score += 25.0

        return self._clamp(
            score
        )

    def _device_risk(
        self,
        features: Mapping[str, Any],
        transaction: Mapping[str, Any],
    ) -> float:

        new_device = self._boolean(
            features.get(
                "new_device",
                transaction.get(
                    "new_device"
                ),
            )
        )

        if new_device:
            return 80.0

        return 10.0

    def _behavior_risk(
        self,
        features: Mapping[str, Any],
        transaction: Mapping[str, Any],
    ) -> float:

        signals = 0

        checks = (
            (
                "new_device",
                transaction.get(
                    "new_device"
                ),
            ),

            (
                "new_location",
                transaction.get(
                    "new_location"
                ),
            ),

            (
                "is_night",
                features.get(
                    "is_night"
                ),
            ),

            (
                "is_international",
                features.get(
                    "is_international"
                ),
            ),
        )

        for feature_name, fallback in checks:

            value = features.get(
                feature_name,
                fallback,
            )

            if self._boolean(
                value
            ):
                signals += 1

        velocity = self._number(
            features.get(
                "velocity"
            )
        )

        if velocity >= 10:
            signals += 1

        if signals >= 4:
            return 95.0

        if signals == 3:
            return 80.0

        if signals == 2:
            return 60.0

        if signals == 1:
            return 35.0

        return 10.0

    # ========================================================================
    # CONTRIBUTION BUILDERS
    # ========================================================================

    def _model_contribution(
        self,
        signal: str,
        value: float,
        weight: float,
        reason: str,
    ) -> ScoreContribution:

        return ScoreContribution(
            signal=signal,
            raw_value=value,
            normalized_value=value,
            weight=weight,
            contribution=(
                value * weight
            ),
            reason=reason,
        )

    def _context_contribution(
        self,
        signal: str,
        value: float,
        weight: float,
        reason: str,
    ) -> ScoreContribution:

        return ScoreContribution(
            signal=signal,
            raw_value=value,
            normalized_value=value,
            weight=weight,
            contribution=(
                value * weight
            ),
            reason=reason,
        )

    # ========================================================================
    # WEIGHTED SCORE
    # ========================================================================

    def _weighted_score(
        self,
        contributions: Sequence[
            ScoreContribution
        ],
    ) -> float:

        if not contributions:
            return 0.0

        total_weight = sum(
            item.weight
            for item in contributions
        )

        if total_weight <= 0:
            return 0.0

        total_contribution = sum(
            item.contribution
            for item in contributions
        )

        return (
            total_contribution
            / total_weight
        )

    # ========================================================================
    # MODEL OVERRIDES
    # ========================================================================

    def _apply_model_overrides(
        self,
        score: float,
        anomaly_score: float,
        supervised_score: float,
    ) -> float:

        if (
            anomaly_score
            >= self.config.anomaly_override_threshold
        ):

            score = max(
                score,
                anomaly_score,
            )

        if (
            supervised_score
            >= self.config.supervised_override_threshold
        ):

            score = max(
                score,
                supervised_score,
            )

        # If both independent models strongly
        # agree, raise the minimum risk.

        if (
            anomaly_score >= 80
            and supervised_score >= 80
        ):

            score = max(
                score,
                85.0,
            )

        return score

    # ========================================================================
    # RISK BAND
    # ========================================================================

    def risk_band(
        self,
        score: float,
    ) -> str:

        score = self._clamp(
            score
        )

        if (
            score
            >= self.config.critical_threshold
        ):
            return ScoreBand.CRITICAL

        if (
            score
            >= self.config.high_threshold
        ):
            return ScoreBand.HIGH

        if (
            score
            >= self.config.medium_threshold
        ):
            return ScoreBand.MEDIUM

        if (
            score
            >= self.config.low_threshold
        ):
            return ScoreBand.LOW

        return ScoreBand.VERY_LOW

    # ========================================================================
    # REASONS
    # ========================================================================

    @staticmethod
    def _top_reasons(
        contributions: Sequence[
            ScoreContribution
        ],
        limit: int = 5,
    ) -> list[str]:

        ranked = sorted(
            contributions,
            key=lambda item:
                abs(
                    item.contribution
                ),
            reverse=True,
        )

        reasons: list[
            str
        ] = []

        for item in ranked:

            reason = (
                item.reason
                .strip()
            )

            if (
                reason
                and reason not in reasons
            ):

                reasons.append(
                    reason
                )

            if len(
                reasons
            ) >= limit:
                break

        return reasons

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def _calculate_confidence(
        self,
        anomaly_result: Mapping[str, Any],
        supervised_result: Mapping[str, Any],
        score: float,
    ) -> float:

        model_confidences: list[
            float
        ] = []

        for result in (
            anomaly_result,
            supervised_result,
        ):

            for key in (
                "confidence",
                "model_confidence",
            ):

                if key not in result:
                    continue

                value = self._number(
                    result[key]
                )

                if (
                    0 <= value <= 1
                ):

                    model_confidences.append(
                        value
                    )

                elif (
                    0 <= value <= 100
                ):

                    model_confidences.append(
                        value / 100
                    )

                break

        if model_confidences:

            confidence = (
                sum(
                    model_confidences
                )
                / len(
                    model_confidences
                )
            )

        else:

            confidence = (
                0.90
                if (
                    anomaly_result
                    and supervised_result
                )
                else
                0.75
                if (
                    anomaly_result
                    or supervised_result
                )
                else
                0.50
            )

        # Very low or very high scores with
        # multiple signals provide stronger
        # decision confidence.

        if (
            anomaly_result
            and supervised_result
        ):

            anomaly_score = (
                self._extract_anomaly_score(
                    anomaly_result
                )
            )

            supervised_score = (
                self._extract_supervised_score(
                    supervised_result
                )
            )

            agreement = (
                1.0
                - abs(
                    anomaly_score
                    - supervised_score
                ) / 100.0
            )

            confidence = (
                confidence * 0.70
                + agreement * 0.30
            )

        if score >= 90:
            confidence += 0.03

        return self._clamp(
            confidence,
            0.0,
            1.0,
        )

    # ========================================================================
    # REASON HELPERS
    # ========================================================================

    @staticmethod
    def _amount_reason(
        signal: float,
    ) -> str:

        if signal >= 80:
            return (
                "Transaction amount is "
                "substantially unusual."
            )

        if signal >= 50:
            return (
                "Transaction amount is "
                "higher than the expected baseline."
            )

        return (
            "Transaction amount does not "
            "strongly indicate fraud."
        )

    @staticmethod
    def _velocity_reason(
        signal: float,
    ) -> str:

        if signal >= 80:
            return (
                "Transaction velocity is "
                "abnormally high."
            )

        if signal >= 50:
            return (
                "Transaction frequency "
                "is elevated."
            )

        return (
            "Transaction velocity is "
            "within a relatively normal range."
        )

    @staticmethod
    def _ratio_reason(
        signal: float,
    ) -> str:

        if signal >= 80:
            return (
                "Transaction amount is very large "
                "relative to available balance."
            )

        if signal >= 50:
            return (
                "Transaction consumes a significant "
                "portion of the account balance."
            )

        return (
            "Transaction-to-balance ratio "
            "does not strongly indicate fraud."
        )

    # ========================================================================
    # VALUE HELPERS
    # ========================================================================

    @staticmethod
    def _first_value(
        data: Mapping[str, Any],
        keys: Sequence[str],
    ) -> Any:

        for key in keys:

            if (
                key in data
                and data[key] is not None
            ):

                return data[key]

        return 0.0

    @staticmethod
    def _normalize_score(
        value: Any,
    ) -> float:

        number = FraudScorer._number(
            value
        )

        # Models sometimes return 0-1 scores.

        if 0 <= number <= 1:
            number *= 100

        return FraudScorer._clamp(
            number
        )

    @staticmethod
    def _normalize_probability(
        value: Any,
    ) -> float:

        number = FraudScorer._number(
            value
        )

        if 0 <= number <= 1:
            number *= 100

        return FraudScorer._clamp(
            number
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
    def _boolean(
        value: Any,
    ) -> bool:

        if isinstance(
            value,
            bool,
        ):

            return value

        if isinstance(
            value,
            str,
        ):

            return (
                value.strip().lower()
                in {
                    "true",
                    "1",
                    "yes",
                    "y",
                    "on",
                }
            )

        if isinstance(
            value,
            (int, float),
        ):

            return bool(
                value
            )

        return False

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

        if not math.isfinite(
            number
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
# Convenience Functions
# ============================================================================


def calculate_fraud_score(
    features: Optional[
        Mapping[str, Any]
    ] = None,
    anomaly_result: Optional[
        Mapping[str, Any]
    ] = None,
    supervised_result: Optional[
        Mapping[str, Any]
    ] = None,
    transaction: Optional[
        Mapping[str, Any]
    ] = None,
    config: Optional[
        FraudScoringConfig
    ] = None,
) -> FraudScoreResult:
    """
    Convenience function for calculating
    a fraud score.
    """

    scorer = FraudScorer(
        config=config
    )

    return scorer.score(
        features=features,
        anomaly_result=anomaly_result,
        supervised_result=supervised_result,
        transaction=transaction,
    )


def score_transaction(
    transaction: Mapping[str, Any],
    features: Optional[
        Mapping[str, Any]
    ] = None,
    anomaly_result: Optional[
        Mapping[str, Any]
    ] = None,
    supervised_result: Optional[
        Mapping[str, Any]
    ] = None,
    config: Optional[
        FraudScoringConfig
    ] = None,
) -> FraudScoreResult:
    """
    Convenience API for scoring a transaction.
    """

    scorer = FraudScorer(
        config=config
    )

    return scorer.score(
        features=features,
        anomaly_result=anomaly_result,
        supervised_result=supervised_result,
        transaction=transaction,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "FraudScoringError",
    "InvalidScoreInputError",
    "InvalidWeightError",
    "ScoreBand",
    "ScoreSignal",
    "FraudScoringConfig",
    "ScoreContribution",
    "FraudScoreResult",
    "FraudScorer",
    "calculate_fraud_score",
    "score_transaction",
]