"""
FinCo AI - Alert Severity Engine

Converts financial risk signals into operational alert severity.

Responsibilities:
- Convert risk scores into severity levels
- Consider alert type and business impact
- Consider human-review requirements
- Consider multiple financial risk factors
- Produce explainable severity decisions
- Keep severity logic deterministic

Architecture:

    Alert Rules
        ↓
    Detector
        ↓
    Risk Scoring
        ↓
    Severity Engine
        ↓
    ┌───────────────────────────────┐
    │ INFO                          │
    │ LOW                           │
    │ MEDIUM                        │
    │ HIGH                          │
    │ CRITICAL                      │
    └───────────────────────────────┘
        ↓
    Alert Generator
        ↓
    Notification / Escalation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# Exceptions
# ============================================================================

class SeverityError(Exception):
    """Base exception for severity calculation failures."""


class InvalidSeverityError(SeverityError):
    """Raised when an invalid severity is generated."""


# ============================================================================
# Severity Levels
# ============================================================================

class SeverityLevel:
    """
    FinCo AI operational severity levels.

    INFO
        Informational event. No immediate action.

    LOW
        Minor financial deviation.

    MEDIUM
        Meaningful financial deterioration requiring monitoring.

    HIGH
        Significant financial risk requiring management attention.

    CRITICAL
        Severe financial risk requiring immediate human review.
    """

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================================
# Alert Types
# ============================================================================

class SeverityAlertType:
    """Supported alert types."""

    FINANCIAL_PROFILE_DOWN = (
        "FINANCIAL_PROFILE_DOWN"
    )

    PREVIOUS_PERIOD_DECLINE = (
        "PREVIOUS_PERIOD_DECLINE"
    )

    SEVERE_LOSS_RISK = (
        "SEVERE_LOSS_RISK"
    )


# ============================================================================
# Severity Decision
# ============================================================================

@dataclass
class SeverityDecision:
    """
    Complete severity calculation result.
    """

    severity: str

    score: float

    risk_score: float

    alert_type: str

    reason: str

    requires_human_review: bool = False

    escalation_required: bool = False

    evidence: List[Dict[str, Any]] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Severity Engine
# ============================================================================

class SeverityEngine:
    """
    Deterministic severity engine.

    Default mapping:

        0-19    INFO
        20-39   LOW
        40-59   MEDIUM
        60-79   HIGH
        80-100  CRITICAL

    Additional business rules can raise severity when
    the underlying financial situation is particularly serious.
    """

    MIN_SCORE = 0.0
    MAX_SCORE = 100.0

    # ------------------------------------------------------------------------
    # Default score thresholds
    # ------------------------------------------------------------------------

    INFO_MAX = 19.99
    LOW_MAX = 39.99
    MEDIUM_MAX = 59.99
    HIGH_MAX = 79.99

    # ------------------------------------------------------------------------
    # Special thresholds
    # ------------------------------------------------------------------------

    HIGH_RISK_THRESHOLD = 60.0
    CRITICAL_RISK_THRESHOLD = 80.0

    HUMAN_REVIEW_THRESHOLD = 75.0

    CRITICAL_LOSS_PROBABILITY = 90.0

    CRITICAL_HEALTH_SCORE = 25.0

    CRITICAL_REVENUE_DECLINE = -30.0

    CRITICAL_MARGIN_DECLINE = -25.0

    def __init__(
        self,
        info_max: float = INFO_MAX,
        low_max: float = LOW_MAX,
        medium_max: float = MEDIUM_MAX,
        high_max: float = HIGH_MAX,
        human_review_threshold: float = (
            HUMAN_REVIEW_THRESHOLD
        ),
    ) -> None:

        self.info_max = float(
            info_max
        )

        self.low_max = float(
            low_max
        )

        self.medium_max = float(
            medium_max
        )

        self.high_max = float(
            high_max
        )

        self.human_review_threshold = float(
            human_review_threshold
        )

        self._validate_thresholds()

    # ========================================================================
    # MAIN API
    # ========================================================================

    def calculate(
        self,
        risk_score: float,
        alert_type: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> SeverityDecision:
        """
        Calculate severity from risk score and financial context.
        """

        metrics = metrics or {}

        risk_score = self._clamp(
            risk_score
        )

        alert_type = str(
            alert_type
        ).upper()

        base_severity = self.from_score(
            risk_score
        )

        severity = self._apply_business_rules(
            base_severity=base_severity,
            risk_score=risk_score,
            alert_type=alert_type,
            metrics=metrics,
        )

        human_review = (
            self.requires_human_review(
                severity=severity,
                risk_score=risk_score,
                alert_type=alert_type,
                metrics=metrics,
            )
        )

        escalation_required = (
            self.requires_escalation(
                severity=severity,
                risk_score=risk_score,
                requires_human_review=human_review,
            )
        )

        reason = self.generate_reason(
            severity=severity,
            risk_score=risk_score,
            alert_type=alert_type,
            metrics=metrics,
        )

        evidence = self.build_evidence(
            severity=severity,
            risk_score=risk_score,
            alert_type=alert_type,
            metrics=metrics,
        )

        return SeverityDecision(
            severity=severity,
            score=risk_score,
            risk_score=risk_score,
            alert_type=alert_type,
            reason=reason,
            requires_human_review=human_review,
            escalation_required=escalation_required,
            evidence=evidence,
            metadata={
                "base_severity":
                    base_severity,
            },
        )

    # ========================================================================
    # SCORE → SEVERITY
    # ========================================================================

    def from_score(
        self,
        risk_score: float,
    ) -> str:
        """
        Convert a 0-100 risk score into severity.
        """

        score = self._clamp(
            risk_score
        )

        if score <= self.info_max:
            return SeverityLevel.INFO

        if score <= self.low_max:
            return SeverityLevel.LOW

        if score <= self.medium_max:
            return SeverityLevel.MEDIUM

        if score <= self.high_max:
            return SeverityLevel.HIGH

        return SeverityLevel.CRITICAL

    # ========================================================================
    # BUSINESS RULES
    # ========================================================================

    def _apply_business_rules(
        self,
        base_severity: str,
        risk_score: float,
        alert_type: str,
        metrics: Dict[str, Any],
    ) -> str:
        """
        Apply financial context that can increase severity.
        """

        severity = base_severity

        # --------------------------------------------------------------
        # Severe loss risk
        # --------------------------------------------------------------

        if (
            alert_type
            == SeverityAlertType.SEVERE_LOSS_RISK
        ):

            loss_probability = self._number(
                metrics.get(
                    "loss_probability"
                )
            )

            projected_profit = self._number(
                metrics.get(
                    "projected_profit"
                )
            )

            projected_loss = self._number(
                metrics.get(
                    "projected_loss"
                )
            )

            if (
                loss_probability
                >= self.CRITICAL_LOSS_PROBABILITY
            ):
                severity = self._max_severity(
                    severity,
                    SeverityLevel.CRITICAL,
                )

            if projected_profit < 0 and projected_loss > 0:
                severity = self._max_severity(
                    severity,
                    SeverityLevel.HIGH,
                )

        # --------------------------------------------------------------
        # Financial profile
        # --------------------------------------------------------------

        if (
            alert_type
            == SeverityAlertType.FINANCIAL_PROFILE_DOWN
        ):

            health_score = self._number(
                metrics.get(
                    "health_score",
                    100,
                )
            )

            revenue_change = self._number(
                metrics.get(
                    "revenue_change_percent"
                )
            )

            margin_change = self._number(
                metrics.get(
                    "margin_change_percent"
                )
            )

            if (
                health_score
                <= self.CRITICAL_HEALTH_SCORE
            ):
                severity = self._max_severity(
                    severity,
                    SeverityLevel.CRITICAL,
                )

            if (
                revenue_change
                <= self.CRITICAL_REVENUE_DECLINE
            ):
                severity = self._max_severity(
                    severity,
                    SeverityLevel.CRITICAL,
                )

            if (
                margin_change
                <= self.CRITICAL_MARGIN_DECLINE
            ):
                severity = self._max_severity(
                    severity,
                    SeverityLevel.HIGH,
                )

        # --------------------------------------------------------------
        # Previous period decline
        # --------------------------------------------------------------

        if (
            alert_type
            == SeverityAlertType.PREVIOUS_PERIOD_DECLINE
        ):

            change_percent = self._number(
                metrics.get(
                    "change_percent"
                )
            )

            if (
                change_percent
                <= -30.0
            ):
                severity = self._max_severity(
                    severity,
                    SeverityLevel.CRITICAL,
                )

            elif (
                change_percent
                <= -20.0
            ):
                severity = self._max_severity(
                    severity,
                    SeverityLevel.HIGH,
                )

        return severity

    # ========================================================================
    # HUMAN REVIEW
    # ========================================================================

    def requires_human_review(
        self,
        severity: str,
        risk_score: float,
        alert_type: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Determine whether an alert requires human review.
        """

        metrics = metrics or {}

        severity = str(
            severity
        ).upper()

        if severity in (
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL,
        ):
            return True

        if (
            risk_score
            >= self.human_review_threshold
        ):
            return True

        if (
            alert_type
            == SeverityAlertType.SEVERE_LOSS_RISK
        ):

            loss_probability = self._number(
                metrics.get(
                    "loss_probability"
                )
            )

            if loss_probability >= 70:
                return True

        return False

    # ========================================================================
    # ESCALATION
    # ========================================================================

    @staticmethod
    def requires_escalation(
        severity: str,
        risk_score: float,
        requires_human_review: bool,
    ) -> bool:
        """
        Determine whether alert escalation should occur.
        """

        severity = str(
            severity
        ).upper()

        if severity in (
            SeverityLevel.HIGH,
            SeverityLevel.CRITICAL,
        ):
            return True

        if (
            risk_score >= 75
        ):
            return True

        if requires_human_review:
            return True

        return False

    # ========================================================================
    # REASON
    # ========================================================================

    def generate_reason(
        self,
        severity: str,
        risk_score: float,
        alert_type: str,
        metrics: Dict[str, Any],
    ) -> str:
        """
        Generate deterministic explanation for severity.
        """

        severity = str(
            severity
        ).upper()

        if (
            severity
            == SeverityLevel.CRITICAL
        ):
            return (
                f"Critical severity assigned because "
                f"financial risk is severe "
                f"(risk score {risk_score:.1f}/100) "
                f"for {alert_type}."
            )

        if (
            severity
            == SeverityLevel.HIGH
        ):
            return (
                f"High severity assigned because "
                f"the financial condition represents "
                f"significant risk "
                f"(risk score {risk_score:.1f}/100)."
            )

        if (
            severity
            == SeverityLevel.MEDIUM
        ):
            return (
                f"Medium severity assigned because "
                f"the financial condition requires "
                f"monitoring "
                f"(risk score {risk_score:.1f}/100)."
            )

        if (
            severity
            == SeverityLevel.LOW
        ):
            return (
                f"Low severity assigned because "
                f"the financial deviation is limited "
                f"(risk score {risk_score:.1f}/100)."
            )

        return (
            f"Informational event with minimal "
            f"financial risk "
            f"(risk score {risk_score:.1f}/100)."
        )

    # ========================================================================
    # EVIDENCE
    # ========================================================================

    def build_evidence(
        self,
        severity: str,
        risk_score: float,
        alert_type: str,
        metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Build evidence supporting the severity decision.
        """

        evidence: List[Dict[str, Any]] = [
            {
                "metric": "risk_score",
                "value": risk_score,
                "threshold": self._severity_threshold(
                    severity
                ),
                "severity": severity,
            }
        ]

        if (
            alert_type
            == SeverityAlertType.SEVERE_LOSS_RISK
        ):

            if "loss_probability" in metrics:

                evidence.append(
                    {
                        "metric":
                            "loss_probability",
                        "value":
                            metrics[
                                "loss_probability"
                            ],
                        "threshold":
                            self.CRITICAL_LOSS_PROBABILITY,
                    }
                )

            if "projected_loss" in metrics:

                evidence.append(
                    {
                        "metric":
                            "projected_loss",
                        "value":
                            metrics[
                                "projected_loss"
                            ],
                        "threshold":
                            0.0,
                    }
                )

        if (
            alert_type
            == SeverityAlertType.FINANCIAL_PROFILE_DOWN
        ):

            for metric_name in (
                "health_score",
                "revenue_change_percent",
                "margin_change_percent",
                "cash_flow",
                "profitability",
            ):

                if metric_name in metrics:

                    evidence.append(
                        {
                            "metric":
                                metric_name,
                            "value":
                                metrics[
                                    metric_name
                                ],
                        }
                    )

        if (
            alert_type
            == SeverityAlertType.PREVIOUS_PERIOD_DECLINE
        ):

            if "change_percent" in metrics:

                evidence.append(
                    {
                        "metric":
                            "change_percent",
                        "value":
                            metrics[
                                "change_percent"
                            ],
                        "threshold":
                            -10.0,
                    }
                )

        return evidence

    # ========================================================================
    # SEVERITY COMPARISON
    # ========================================================================

    @staticmethod
    def _max_severity(
        current: str,
        candidate: str,
    ) -> str:
        """
        Return the more severe level.
        """

        order = {
            SeverityLevel.INFO: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4,
        }

        if (
            candidate not in order
            or current not in order
        ):
            raise InvalidSeverityError(
                "Unknown severity level."
            )

        if (
            order[candidate]
            > order[current]
        ):
            return candidate

        return current

    # ========================================================================
    # THRESHOLD
    # ========================================================================

    def _severity_threshold(
        self,
        severity: str,
    ) -> float:

        mapping = {
            SeverityLevel.INFO:
                0.0,

            SeverityLevel.LOW:
                self.info_max + 0.01,

            SeverityLevel.MEDIUM:
                self.low_max + 0.01,

            SeverityLevel.HIGH:
                self.medium_max + 0.01,

            SeverityLevel.CRITICAL:
                self.high_max + 0.01,
        }

        return mapping.get(
            severity,
            0.0,
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_thresholds(
        self,
    ) -> None:

        values = [
            self.info_max,
            self.low_max,
            self.medium_max,
            self.high_max,
        ]

        if any(
            value < 0 or value > 100
            for value in values
        ):
            raise SeverityError(
                "Severity thresholds must "
                "be between 0 and 100."
            )

        if not (
            self.info_max
            < self.low_max
            < self.medium_max
            < self.high_max
        ):
            raise SeverityError(
                "Severity thresholds must "
                "be strictly increasing."
            )

        if not (
            0
            <= self.human_review_threshold
            <= 100
        ):
            raise SeverityError(
                "Human review threshold must "
                "be between 0 and 100."
            )

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _clamp(
        value: Any,
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

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return default


# ============================================================================
# Convenience Functions
# ============================================================================

def calculate_severity(
    risk_score: float,
    alert_type: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> SeverityDecision:
    """
    Convenience function for severity calculation.
    """

    engine = SeverityEngine()

    return engine.calculate(
        risk_score=risk_score,
        alert_type=alert_type,
        metrics=metrics,
    )


def severity_from_score(
    risk_score: float,
) -> str:
    """
    Convenience function for direct score-to-severity conversion.
    """

    engine = SeverityEngine()

    return engine.from_score(
        risk_score
    )


__all__ = [
    "SeverityError",
    "InvalidSeverityError",
    "SeverityLevel",
    "SeverityAlertType",
    "SeverityDecision",
    "SeverityEngine",
    "calculate_severity",
    "severity_from_score",
]