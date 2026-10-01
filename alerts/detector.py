"""
FinCo AI - Alert Detector

Detects financial conditions that may require an alert.

Responsibilities:
- Run configured alert rules
- Detect financial deterioration
- Detect previous-period decline
- Detect severe loss risk
- Produce structured detection results
- Avoid persistence and notifications

Architecture:

    Financial Metrics
          ↓
       detector.py
          ↓
    Detected Conditions
          ↓
    risk_scoring.py
          ↓
      severity.py
          ↓
    alert_generator.py
          ↓
    alert_service.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Exceptions
# ============================================================================

class AlertDetectionError(Exception):
    """Base exception for alert detection failures."""


# ============================================================================
# Detection Result
# ============================================================================

@dataclass
class DetectionResult:
    """
    Represents one detected financial condition.

    This is deliberately not an Alert object.

    DetectionResult says:
        "Something happened."

    AlertGenerator later converts it into:
        "This is the user-facing alert."
    """

    detected: bool

    alert_type: str

    company_id: str

    metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    evidence: List[Dict[str, Any]] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    detected_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "detected": self.detected,
            "alert_type": self.alert_type,
            "company_id": self.company_id,
            "metrics": self.metrics,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat(),
            "message": self.message,
        }


# ============================================================================
# Detector
# ============================================================================

class AlertDetector:
    """
    Financial alert condition detector.

    Default conditions:

    1. Financial profile deterioration
    2. Previous-period decline
    3. Severe loss risk

    Thresholds can be configured per instance.
    """

    FINANCIAL_PROFILE_DOWN = (
        "FINANCIAL_PROFILE_DOWN"
    )

    PREVIOUS_PERIOD_DECLINE = (
        "PREVIOUS_PERIOD_DECLINE"
    )

    SEVERE_LOSS_RISK = (
        "SEVERE_LOSS_RISK"
    )

    def __init__(
        self,
        financial_health_threshold: float = 50.0,
        previous_period_decline_threshold: float = 10.0,
        severe_loss_probability_threshold: float = 70.0,
        severe_loss_amount_threshold: float = 0.0,
    ) -> None:

        self.financial_health_threshold = (
            financial_health_threshold
        )

        self.previous_period_decline_threshold = (
            previous_period_decline_threshold
        )

        self.severe_loss_probability_threshold = (
            severe_loss_probability_threshold
        )

        self.severe_loss_amount_threshold = (
            severe_loss_amount_threshold
        )

    # ========================================================================
    # MAIN DETECTION
    # ========================================================================

    def detect(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> List[DetectionResult]:
        """
        Run all configured alert detectors.

        Parameters
        ----------
        company_id:
            Company being monitored.

        metrics:
            Normalized financial metrics.

        Returns
        -------
        List[DetectionResult]
            All conditions detected.
        """

        if not company_id:
            raise AlertDetectionError(
                "company_id is required."
            )

        if not isinstance(metrics, dict):
            raise AlertDetectionError(
                "metrics must be a dictionary."
            )

        results: List[DetectionResult] = []

        detection_functions: List[
            Callable[
                [str, Dict[str, Any]],
                Optional[DetectionResult],
            ]
        ] = [
            self.detect_financial_profile_down,
            self.detect_previous_period_decline,
            self.detect_severe_loss_risk,
        ]

        for detector in detection_functions:

            try:
                result = detector(
                    company_id,
                    metrics,
                )

                if result is not None and result.detected:
                    results.append(result)

            except Exception as exc:

                raise AlertDetectionError(
                    f"Alert detection failed in "
                    f"{detector.__name__}: {exc}"
                ) from exc

        return results

    # ========================================================================
    # FINANCIAL PROFILE DOWN
    # ========================================================================

    def detect_financial_profile_down(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> Optional[DetectionResult]:
        """
        Detect deterioration in the company's overall financial profile.

        Primary signal:
            health_score < configured threshold

        Additional supporting signals:
            - declining profitability
            - negative cash flow
            - declining revenue
            - increasing expenses
            - deteriorating margins
        """

        health_score = self._number(
            metrics.get("health_score")
        )

        profitability = self._number(
            metrics.get("profitability")
        )

        cash_flow = self._number(
            metrics.get("cash_flow")
        )

        revenue_change = self._number(
            metrics.get("revenue_change_percent")
        )

        expense_change = self._number(
            metrics.get("expense_change_percent")
        )

        margin_change = self._number(
            metrics.get("margin_change_percent")
        )

        signals: List[str] = []

        evidence: List[Dict[str, Any]] = []

        # --------------------------------------------------------------
        # Health score
        # --------------------------------------------------------------

        if (
            health_score is not None
            and health_score
            < self.financial_health_threshold
        ):

            signals.append(
                "financial_health_below_threshold"
            )

            evidence.append(
                {
                    "metric": "health_score",
                    "value": health_score,
                    "threshold": (
                        self.financial_health_threshold
                    ),
                    "comparison": "below",
                }
            )

        # --------------------------------------------------------------
        # Profitability
        # --------------------------------------------------------------

        if (
            profitability is not None
            and profitability < 0
        ):

            signals.append(
                "negative_profitability"
            )

            evidence.append(
                {
                    "metric": "profitability",
                    "value": profitability,
                    "threshold": 0,
                    "comparison": "below",
                }
            )

        # --------------------------------------------------------------
        # Cash flow
        # --------------------------------------------------------------

        if (
            cash_flow is not None
            and cash_flow < 0
        ):

            signals.append(
                "negative_cash_flow"
            )

            evidence.append(
                {
                    "metric": "cash_flow",
                    "value": cash_flow,
                    "threshold": 0,
                    "comparison": "below",
                }
            )

        # --------------------------------------------------------------
        # Revenue
        # --------------------------------------------------------------

        if (
            revenue_change is not None
            and revenue_change < 0
        ):

            signals.append(
                "revenue_decline"
            )

            evidence.append(
                {
                    "metric": "revenue_change_percent",
                    "value": revenue_change,
                    "threshold": 0,
                    "comparison": "below",
                }
            )

        # --------------------------------------------------------------
        # Expenses
        # --------------------------------------------------------------

        if (
            expense_change is not None
            and expense_change > 0
        ):

            signals.append(
                "expense_growth"
            )

            evidence.append(
                {
                    "metric": "expense_change_percent",
                    "value": expense_change,
                    "threshold": 0,
                    "comparison": "above",
                }
            )

        # --------------------------------------------------------------
        # Margin
        # --------------------------------------------------------------

        if (
            margin_change is not None
            and margin_change < 0
        ):

            signals.append(
                "margin_deterioration"
            )

            evidence.append(
                {
                    "metric": "margin_change_percent",
                    "value": margin_change,
                    "threshold": 0,
                    "comparison": "below",
                }
            )

        # --------------------------------------------------------------
        # Determine whether enough evidence exists
        # --------------------------------------------------------------

        if not self._profile_condition_met(
            health_score=health_score,
            signals=signals,
        ):
            return None

        return DetectionResult(
            detected=True,
            alert_type=self.FINANCIAL_PROFILE_DOWN,
            company_id=company_id,
            metrics={
                "health_score": health_score,
                "profitability": profitability,
                "cash_flow": cash_flow,
                "revenue_change_percent": revenue_change,
                "expense_change_percent": expense_change,
                "margin_change_percent": margin_change,
            },
            evidence=evidence,
            metadata={
                "signals": signals,
                "signal_count": len(signals),
                "period": metrics.get("period"),
            },
            message=(
                "The company's financial profile "
                "has deteriorated based on monitored "
                "financial indicators."
            ),
        )

    # ========================================================================
    # PREVIOUS PERIOD DECLINE
    # ========================================================================

    def detect_previous_period_decline(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> Optional[DetectionResult]:
        """
        Detect significant deterioration compared with the previous period.

        Expected inputs:

            current_value
            previous_value

        Optional:

            change_percent
            metric
            period
        """

        current_value = self._number(
            metrics.get("current_value")
        )

        previous_value = self._number(
            metrics.get("previous_value")
        )

        change_percent = self._number(
            metrics.get("change_percent")
        )

        metric_name = metrics.get(
            "metric",
            "financial_metric",
        )

        if (
            current_value is None
            or previous_value is None
        ):
            return None

        # --------------------------------------------------------------
        # Calculate change when not supplied
        # --------------------------------------------------------------

        if change_percent is None:

            if previous_value == 0:

                return None

            change_percent = (
                (
                    current_value
                    - previous_value
                )
                / abs(previous_value)
            ) * 100

        # --------------------------------------------------------------
        # Detect decline
        # --------------------------------------------------------------

        if (
            change_percent
            > -self.previous_period_decline_threshold
        ):
            return None

        evidence = [
            {
                "metric": metric_name,
                "value": current_value,
                "previous_value": previous_value,
                "change_percent": round(
                    change_percent,
                    2,
                ),
                "threshold": (
                    -self.previous_period_decline_threshold
                ),
                "comparison": "below",
                "period": metrics.get(
                    "period"
                ),
            }
        ]

        return DetectionResult(
            detected=True,
            alert_type=self.PREVIOUS_PERIOD_DECLINE,
            company_id=company_id,
            metrics={
                "metric": metric_name,
                "current_value": current_value,
                "previous_value": previous_value,
                "change_percent": round(
                    change_percent,
                    2,
                ),
            },
            evidence=evidence,
            metadata={
                "period": metrics.get(
                    "period"
                ),
                "previous_period": metrics.get(
                    "previous_period"
                ),
            },
            message=(
                f"{metric_name} declined by "
                f"{abs(change_percent):.2f}% "
                f"compared with the previous period."
            ),
        )

    # ========================================================================
    # SEVERE LOSS RISK
    # ========================================================================

    def detect_severe_loss_risk(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> Optional[DetectionResult]:
        """
        Detect elevated severe-loss risk.

        Expected inputs:

            loss_probability
            projected_loss

        Optional:

            forecast_period
            projected_profit
            cash_flow
        """

        loss_probability = self._number(
            metrics.get("loss_probability")
        )

        projected_loss = self._number(
            metrics.get("projected_loss")
        )

        projected_profit = self._number(
            metrics.get("projected_profit")
        )

        cash_flow = self._number(
            metrics.get("cash_flow")
        )

        signals: List[str] = []

        evidence: List[Dict[str, Any]] = []

        # --------------------------------------------------------------
        # Loss probability
        # --------------------------------------------------------------

        if (
            loss_probability is not None
            and loss_probability
            >= self.severe_loss_probability_threshold
        ):

            signals.append(
                "high_loss_probability"
            )

            evidence.append(
                {
                    "metric": "loss_probability",
                    "value": loss_probability,
                    "threshold": (
                        self.severe_loss_probability_threshold
                    ),
                    "comparison": "above_or_equal",
                }
            )

        # --------------------------------------------------------------
        # Projected loss
        # --------------------------------------------------------------

        if (
            projected_loss is not None
            and projected_loss
            > self.severe_loss_amount_threshold
        ):

            signals.append(
                "projected_loss"
            )

            evidence.append(
                {
                    "metric": "projected_loss",
                    "value": projected_loss,
                    "threshold": (
                        self.severe_loss_amount_threshold
                    ),
                    "comparison": "above",
                }
            )

        # --------------------------------------------------------------
        # Projected profit
        # --------------------------------------------------------------

        if (
            projected_profit is not None
            and projected_profit < 0
        ):

            signals.append(
                "negative_projected_profit"
            )

            evidence.append(
                {
                    "metric": "projected_profit",
                    "value": projected_profit,
                    "threshold": 0,
                    "comparison": "below",
                }
            )

        # --------------------------------------------------------------
        # Cash flow
        # --------------------------------------------------------------

        if (
            cash_flow is not None
            and cash_flow < 0
        ):

            signals.append(
                "negative_cash_flow"
            )

            evidence.append(
                {
                    "metric": "cash_flow",
                    "value": cash_flow,
                    "threshold": 0,
                    "comparison": "below",
                }
            )

        # --------------------------------------------------------------
        # Determine condition
        # --------------------------------------------------------------

        if not self._severe_loss_condition_met(
            loss_probability=loss_probability,
            projected_loss=projected_loss,
            projected_profit=projected_profit,
        ):
            return None

        return DetectionResult(
            detected=True,
            alert_type=self.SEVERE_LOSS_RISK,
            company_id=company_id,
            metrics={
                "loss_probability": loss_probability,
                "projected_loss": projected_loss,
                "projected_profit": projected_profit,
                "cash_flow": cash_flow,
                "forecast_period": metrics.get(
                    "forecast_period"
                ),
            },
            evidence=evidence,
            metadata={
                "signals": signals,
                "signal_count": len(signals),
                "forecast_model": metrics.get(
                    "forecast_model"
                ),
            },
            message=(
                "The financial forecast indicates "
                "an elevated risk of severe loss."
            ),
        )

    # ========================================================================
    # PROFILE CONDITION
    # ========================================================================

    @staticmethod
    def _profile_condition_met(
        health_score: Optional[float],
        signals: List[str],
    ) -> bool:
        """
        Decide whether financial profile deterioration
        is significant enough to produce an alert.

        A low health score alone is sufficient.

        Otherwise, require at least two deterioration signals.
        """

        if (
            health_score is not None
            and health_score < 50
        ):
            return True

        return len(signals) >= 2

    # ========================================================================
    # SEVERE LOSS CONDITION
    # ========================================================================

    def _severe_loss_condition_met(
        self,
        loss_probability: Optional[float],
        projected_loss: Optional[float],
        projected_profit: Optional[float],
    ) -> bool:
        """
        Determine whether severe-loss conditions are satisfied.
        """

        high_probability = (
            loss_probability is not None
            and loss_probability
            >= self.severe_loss_probability_threshold
        )

        projected_loss_detected = (
            projected_loss is not None
            and projected_loss
            > self.severe_loss_amount_threshold
        )

        negative_profit = (
            projected_profit is not None
            and projected_profit < 0
        )

        # Strong primary signals can trigger the alert individually.
        if high_probability:
            return True

        if (
            projected_loss_detected
            and negative_profit
        ):
            return True

        return False

    # ========================================================================
    # SINGLE DETECTOR
    # ========================================================================

    def detect_type(
        self,
        company_id: str,
        alert_type: str,
        metrics: Dict[str, Any],
    ) -> Optional[DetectionResult]:
        """
        Run one specific detector.

        Useful when an API endpoint or scheduled job only wants
        to evaluate a particular alert type.
        """

        detectors = {
            self.FINANCIAL_PROFILE_DOWN:
                self.detect_financial_profile_down,

            self.PREVIOUS_PERIOD_DECLINE:
                self.detect_previous_period_decline,

            self.SEVERE_LOSS_RISK:
                self.detect_severe_loss_risk,
        }

        detector = detectors.get(
            alert_type
        )

        if detector is None:
            raise AlertDetectionError(
                f"Unknown alert type: {alert_type}"
            )

        return detector(
            company_id,
            metrics,
        )

    # ========================================================================
    # UTILITY
    # ========================================================================

    @staticmethod
    def _number(
        value: Any,
    ) -> Optional[float]:
        """
        Safely convert numeric values.

        Handles:
        - int
        - float
        - numeric strings
        - None
        - invalid values
        """

        if value is None:
            return None

        if isinstance(
            value,
            bool,
        ):
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None


# ============================================================================
# Convenience Function
# ============================================================================

def detect_alerts(
    company_id: str,
    metrics: Dict[str, Any],
) -> List[DetectionResult]:
    """
    Functional wrapper for simple use cases.
    """

    detector = AlertDetector()

    return detector.detect(
        company_id=company_id,
        metrics=metrics,
    )


__all__ = [
    "AlertDetectionError",
    "DetectionResult",
    "AlertDetector",
    "detect_alerts",
]