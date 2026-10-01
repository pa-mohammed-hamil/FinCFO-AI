"""
FinCo AI - Alert Risk Scoring

Calculates a deterministic financial risk score for detected alerts.

Responsibilities:
- Calculate normalized 0-100 risk scores
- Combine multiple financial risk signals
- Score different alert types
- Explain scoring factors
- Provide risk bands
- Support configurable weights and thresholds
- Avoid LLM dependency

Architecture:

    Financial Metrics
          │
          ▼
    Alert Detector
          │
          ▼
    Risk Scoring
          │
       0 ─────────── 100
       │              │
      LOW          CRITICAL
          │
          ▼
    Severity Engine
          │
          ▼
    Alert Generator
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# Exceptions
# ============================================================================

class RiskScoringError(Exception):
    """Base exception for risk scoring failures."""


class InvalidRiskScoreError(RiskScoringError):
    """Raised when an invalid risk score is produced."""


# ============================================================================
# Risk Bands
# ============================================================================

class RiskBand:
    """Risk classification bands."""

    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================================
# Alert Types
# ============================================================================

class RiskAlertType:
    """Supported FinCo AI alert types."""

    FINANCIAL_PROFILE_DOWN = "FINANCIAL_PROFILE_DOWN"
    PREVIOUS_PERIOD_DECLINE = "PREVIOUS_PERIOD_DECLINE"
    SEVERE_LOSS_RISK = "SEVERE_LOSS_RISK"


# ============================================================================
# Risk Factor
# ============================================================================

@dataclass
class RiskFactor:
    """
    Individual contributor to the final risk score.
    """

    name: str
    value: float
    contribution: float
    weight: float

    description: str

    threshold: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Risk Score Result
# ============================================================================

@dataclass
class RiskScoreResult:
    """
    Complete risk scoring result.
    """

    company_id: str

    alert_type: str

    score: float

    band: str

    factors: List[RiskFactor] = field(
        default_factory=list
    )

    explanation: Optional[str] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "company_id": self.company_id,
            "alert_type": self.alert_type,
            "score": self.score,
            "band": self.band,
            "factors": [
                factor.to_dict()
                for factor in self.factors
            ],
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


# ============================================================================
# Risk Scoring Service
# ============================================================================

class RiskScorer:
    """
    Deterministic financial risk scoring engine.

    Score range:

        0-19   VERY_LOW
        20-39  LOW
        40-59  MEDIUM
        60-79  HIGH
        80-100 CRITICAL

    The scoring engine does not make a financial decision.
    It produces a standardized risk signal that can be consumed
    by the severity engine and Supervisor Agent.
    """

    FINANCIAL_PROFILE_DOWN = (
        RiskAlertType.FINANCIAL_PROFILE_DOWN
    )

    PREVIOUS_PERIOD_DECLINE = (
        RiskAlertType.PREVIOUS_PERIOD_DECLINE
    )

    SEVERE_LOSS_RISK = (
        RiskAlertType.SEVERE_LOSS_RISK
    )

    MIN_SCORE = 0.0
    MAX_SCORE = 100.0

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:

        self.weights = weights or {
            "financial_health": 0.25,
            "profitability": 0.15,
            "cash_flow": 0.20,
            "revenue_decline": 0.15,
            "expense_growth": 0.10,
            "margin_decline": 0.15,
        }

        self._validate_weights()

    # ========================================================================
    # MAIN API
    # ========================================================================

    def score(
        self,
        company_id: str,
        alert_type: str,
        metrics: Dict[str, Any],
    ) -> RiskScoreResult:
        """
        Calculate a risk score for an alert type.
        """

        if not company_id:
            raise RiskScoringError(
                "company_id is required."
            )

        if not isinstance(
            metrics,
            dict,
        ):
            raise RiskScoringError(
                "metrics must be a dictionary."
            )

        alert_type = str(
            alert_type
        ).upper()

        if alert_type == self.FINANCIAL_PROFILE_DOWN:
            return self.score_financial_profile_down(
                company_id,
                metrics,
            )

        if alert_type == self.PREVIOUS_PERIOD_DECLINE:
            return self.score_previous_period_decline(
                company_id,
                metrics,
            )

        if alert_type == self.SEVERE_LOSS_RISK:
            return self.score_severe_loss_risk(
                company_id,
                metrics,
            )

        return self.score_generic(
            company_id,
            alert_type,
            metrics,
        )

    # ========================================================================
    # FINANCIAL PROFILE DOWN
    # ========================================================================

    def score_financial_profile_down(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> RiskScoreResult:
        """
        Score deterioration in the overall financial profile.

        Expected metrics may include:

        - health_score
        - profitability
        - cash_flow
        - revenue_change_percent
        - expense_change_percent
        - margin_change_percent
        """

        factors: List[RiskFactor] = []

        # --------------------------------------------------------------
        # Financial health
        # --------------------------------------------------------------

        health_score = self._number(
            metrics.get(
                "health_score"
            ),
            default=100.0,
        )

        health_risk = self._inverse_score(
            health_score
        )

        factors.append(
            RiskFactor(
                name="financial_health",
                value=health_score,
                contribution=health_risk,
                weight=self.weights[
                    "financial_health"
                ],
                description=(
                    "Lower financial health increases risk."
                ),
                threshold=50.0,
            )
        )

        # --------------------------------------------------------------
        # Profitability
        # --------------------------------------------------------------

        profitability = self._number(
            metrics.get(
                "profitability"
            ),
            default=100.0,
        )

        profitability_risk = self._inverse_score(
            profitability
        )

        factors.append(
            RiskFactor(
                name="profitability",
                value=profitability,
                contribution=profitability_risk,
                weight=self.weights[
                    "profitability"
                ],
                description=(
                    "Lower profitability increases financial risk."
                ),
                threshold=50.0,
            )
        )

        # --------------------------------------------------------------
        # Cash flow
        # --------------------------------------------------------------

        cash_flow = self._number(
            metrics.get(
                "cash_flow"
            ),
            default=100.0,
        )

        cash_flow_risk = self._inverse_score(
            cash_flow
        )

        factors.append(
            RiskFactor(
                name="cash_flow",
                value=cash_flow,
                contribution=cash_flow_risk,
                weight=self.weights[
                    "cash_flow"
                ],
                description=(
                    "Weak cash flow increases liquidity risk."
                ),
                threshold=50.0,
            )
        )

        # --------------------------------------------------------------
        # Revenue change
        # --------------------------------------------------------------

        revenue_change = self._number_signed(
            metrics.get(
                "revenue_change_percent"
            )
        )

        revenue_risk = self._decline_score(
            revenue_change
        )

        factors.append(
            RiskFactor(
                name="revenue_decline",
                value=revenue_change,
                contribution=revenue_risk,
                weight=self.weights[
                    "revenue_decline"
                ],
                description=(
                    "Revenue contraction increases financial risk."
                ),
            )
        )

        # --------------------------------------------------------------
        # Expense growth
        # --------------------------------------------------------------

        expense_change = self._number_signed(
            metrics.get(
                "expense_change_percent"
            )
        )

        expense_risk = self._growth_score(
            expense_change
        )

        factors.append(
            RiskFactor(
                name="expense_growth",
                value=expense_change,
                contribution=expense_risk,
                weight=self.weights[
                    "expense_growth"
                ],
                description=(
                    "Rapid expense growth increases risk."
                ),
            )
        )

        # --------------------------------------------------------------
        # Margin change
        # --------------------------------------------------------------

        margin_change = self._number_signed(
            metrics.get(
                "margin_change_percent"
            )
        )

        margin_risk = self._decline_score(
            margin_change
        )

        factors.append(
            RiskFactor(
                name="margin_decline",
                value=margin_change,
                contribution=margin_risk,
                weight=self.weights[
                    "margin_decline"
                ],
                description=(
                    "Margin deterioration increases profitability risk."
                ),
            )
        )

        score = self._weighted_score(
            factors
        )

        return self._build_result(
            company_id=company_id,
            alert_type=self.FINANCIAL_PROFILE_DOWN,
            score=score,
            factors=factors,
        )

    # ========================================================================
    # PREVIOUS PERIOD DECLINE
    # ========================================================================

    def score_previous_period_decline(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> RiskScoreResult:
        """
        Score a decline relative to the previous period.

        Expected:

            current_value
            previous_value

        or:

            change_percent
        """

        change_percent = metrics.get(
            "change_percent"
        )

        if change_percent is None:

            current = self._number_signed(
                metrics.get(
                    "current_value"
                )
            )

            previous = self._number_signed(
                metrics.get(
                    "previous_value"
                )
            )

            if previous != 0:

                change_percent = (
                    (
                        current - previous
                    )
                    / abs(previous)
                ) * 100.0

            else:

                change_percent = (
                    -100.0
                    if current < 0
                    else 0.0
                )

        change_percent = self._number_signed(
            change_percent
        )

        risk = self._decline_score(
            change_percent
        )

        factor = RiskFactor(
            name="previous_period_decline",
            value=change_percent,
            contribution=risk,
            weight=1.0,
            description=(
                "Larger negative period-over-period "
                "changes produce higher risk."
            ),
            threshold=-10.0,
            metadata={
                "metric": metrics.get(
                    "metric"
                ),
                "period": metrics.get(
                    "period"
                ),
            },
        )

        return self._build_result(
            company_id=company_id,
            alert_type=self.PREVIOUS_PERIOD_DECLINE,
            score=risk,
            factors=[
                factor
            ],
        )

    # ========================================================================
    # SEVERE LOSS RISK
    # ========================================================================

    def score_severe_loss_risk(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> RiskScoreResult:
        """
        Score probability and magnitude of severe financial loss.

        Expected metrics:

        - loss_probability
        - projected_loss
        - projected_profit
        - cash_flow
        """

        factors: List[RiskFactor] = []

        # --------------------------------------------------------------
        # Loss probability
        # --------------------------------------------------------------

        loss_probability = self._number(
            metrics.get(
                "loss_probability"
            )
        )

        probability_risk = self._clamp(
            loss_probability
        )

        factors.append(
            RiskFactor(
                name="loss_probability",
                value=loss_probability,
                contribution=probability_risk,
                weight=0.50,
                description=(
                    "Higher predicted probability of loss "
                    "increases risk."
                ),
                threshold=70.0,
            )
        )

        # --------------------------------------------------------------
        # Projected loss
        # --------------------------------------------------------------

        projected_loss = self._number_signed(
            metrics.get(
                "projected_loss"
            )
        )

        loss_magnitude_risk = self._loss_magnitude_score(
            projected_loss,
            metrics,
        )

        factors.append(
            RiskFactor(
                name="projected_loss",
                value=projected_loss,
                contribution=loss_magnitude_risk,
                weight=0.30,
                description=(
                    "Larger projected financial losses "
                    "increase risk."
                ),
                threshold=0.0,
            )
        )

        # --------------------------------------------------------------
        # Projected profit
        # --------------------------------------------------------------

        projected_profit = self._number_signed(
            metrics.get(
                "projected_profit"
            )
        )

        profit_risk = (
            100.0
            if projected_profit < 0
            else 0.0
        )

        factors.append(
            RiskFactor(
                name="projected_profit",
                value=projected_profit,
                contribution=profit_risk,
                weight=0.10,
                description=(
                    "Negative projected profit increases risk."
                ),
                threshold=0.0,
            )
        )

        # --------------------------------------------------------------
        # Cash flow
        # --------------------------------------------------------------

        cash_flow = self._number_signed(
            metrics.get(
                "cash_flow"
            )
        )

        cash_flow_risk = (
            100.0
            if cash_flow < 0
            else 0.0
        )

        factors.append(
            RiskFactor(
                name="negative_cash_flow",
                value=cash_flow,
                contribution=cash_flow_risk,
                weight=0.10,
                description=(
                    "Negative cash flow increases severe-loss risk."
                ),
                threshold=0.0,
            )
        )

        score = self._weighted_score(
            factors
        )

        # A very high loss probability should dominate
        # the final result.

        if loss_probability >= 90:
            score = max(
                score,
                90.0,
            )

        elif loss_probability >= 80:
            score = max(
                score,
                80.0,
            )

        elif loss_probability >= 70:
            score = max(
                score,
                70.0,
            )

        return self._build_result(
            company_id=company_id,
            alert_type=self.SEVERE_LOSS_RISK,
            score=score,
            factors=factors,
        )

    # ========================================================================
    # GENERIC SCORING
    # ========================================================================

    def score_generic(
        self,
        company_id: str,
        alert_type: str,
        metrics: Dict[str, Any],
    ) -> RiskScoreResult:
        """
        Generic fallback scoring.

        Looks for commonly used risk-related metrics.
        """

        candidates = [
            "risk_score",
            "risk",
            "probability",
            "loss_probability",
            "severity_score",
        ]

        values: List[float] = []

        for key in candidates:

            if key in metrics:

                values.append(
                    self._clamp(
                        self._number(
                            metrics[key]
                        )
                    )
                )

        if values:
            score = max(
                values
            )

            factor = RiskFactor(
                name="generic_risk_signal",
                value=score,
                contribution=score,
                weight=1.0,
                description=(
                    "Generic risk signal supplied by the caller."
                ),
            )

            factors = [
                factor
            ]

        else:

            score = 0.0
            factors = []

        return self._build_result(
            company_id=company_id,
            alert_type=alert_type,
            score=score,
            factors=factors,
        )

    # ========================================================================
    # RISK BAND
    # ========================================================================

    @staticmethod
    def get_risk_band(
        score: float,
    ) -> str:
        """
        Convert a numerical score into a risk band.
        """

        score = RiskScorer._clamp(
            score
        )

        if score < 20:
            return RiskBand.VERY_LOW

        if score < 40:
            return RiskBand.LOW

        if score < 60:
            return RiskBand.MEDIUM

        if score < 80:
            return RiskBand.HIGH

        return RiskBand.CRITICAL

    # ========================================================================
    # EXPLANATION
    # ========================================================================

    @staticmethod
    def generate_explanation(
        score: float,
        factors: List[RiskFactor],
    ) -> str:
        """
        Produce a deterministic human-readable explanation.
        """

        band = RiskScorer.get_risk_band(
            score
        )

        if not factors:
            return (
                f"Risk score is {score:.1f}/100 "
                f"with a {band} risk classification."
            )

        sorted_factors = sorted(
            factors,
            key=lambda factor: (
                abs(
                    factor.contribution
                    * factor.weight
                )
            ),
            reverse=True,
        )

        top_factors = sorted_factors[
            :3
        ]

        descriptions = [
            factor.description
            for factor in top_factors
        ]

        return (
            f"Risk score is {score:.1f}/100 "
            f"({band}). Key risk drivers: "
            + " ".join(
                descriptions
            )
        )

    # ========================================================================
    # BUILD RESULT
    # ========================================================================

    def _build_result(
        self,
        company_id: str,
        alert_type: str,
        score: float,
        factors: List[RiskFactor],
    ) -> RiskScoreResult:

        score = self._validate_score(
            score
        )

        band = self.get_risk_band(
            score
        )

        explanation = self.generate_explanation(
            score,
            factors,
        )

        return RiskScoreResult(
            company_id=company_id,
            alert_type=alert_type,
            score=score,
            band=band,
            factors=factors,
            explanation=explanation,
        )

    # ========================================================================
    # WEIGHTED SCORE
    # ========================================================================

    @staticmethod
    def _weighted_score(
        factors: List[RiskFactor],
    ) -> float:

        if not factors:
            return 0.0

        total_weight = sum(
            factor.weight
            for factor in factors
        )

        if total_weight <= 0:
            return 0.0

        weighted = sum(
            factor.contribution
            * factor.weight
            for factor in factors
        )

        return weighted / total_weight

    # ========================================================================
    # SCORE FUNCTIONS
    # ========================================================================

    @staticmethod
    def _inverse_score(
        value: float,
    ) -> float:
        """
        Convert a health-style 0-100 metric into risk.

        100 health → 0 risk
        50 health  → 50 risk
        0 health   → 100 risk
        """

        return RiskScorer._clamp(
            100.0 - value
        )

    @staticmethod
    def _decline_score(
        change_percent: float,
    ) -> float:
        """
        Convert negative percentage change to risk.

        Examples:

            0%    → 0
            -10%  → ~33
            -20%  → ~67
            -30%  → 100
        """

        if change_percent >= 0:
            return 0.0

        decline = abs(
            change_percent
        )

        return RiskScorer._clamp(
            (
                decline
                / 30.0
            )
            * 100.0
        )

    @staticmethod
    def _growth_score(
        change_percent: float,
    ) -> float:
        """
        Convert expense growth into risk.

        0% growth   → 0
        10% growth  → ~33
        20% growth  → ~67
        30%+        → 100
        """

        if change_percent <= 0:
            return 0.0

        return RiskScorer._clamp(
            (
                change_percent
                / 30.0
            )
            * 100.0
        )

    @staticmethod
    def _loss_magnitude_score(
        projected_loss: float,
        metrics: Dict[str, Any],
    ) -> float:
        """
        Normalize projected loss against revenue.

        If revenue is available:

            loss / revenue × 100

        Otherwise use a conservative absolute interpretation.
        """

        if projected_loss <= 0:
            return 0.0

        revenue = RiskScorer._number_signed(
            metrics.get(
                "revenue"
            )
        )

        if revenue > 0:

            loss_percent = (
                projected_loss
                / revenue
            ) * 100.0

            return RiskScorer._clamp(
                loss_percent
                * 2.0
            )

        # If revenue isn't available, a positive projected
        # loss is considered significant but not automatically
        # critical.

        return 60.0

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_weights(
        self,
    ) -> None:

        required = {
            "financial_health",
            "profitability",
            "cash_flow",
            "revenue_decline",
            "expense_growth",
            "margin_decline",
        }

        missing = (
            required
            - set(
                self.weights.keys()
            )
        )

        if missing:
            raise RiskScoringError(
                "Missing risk weights: "
                + ", ".join(
                    sorted(missing)
                )
            )

        for name, weight in self.weights.items():

            if weight < 0:
                raise RiskScoringError(
                    f"Risk weight '{name}' "
                    f"cannot be negative."
                )

        total = sum(
            self.weights.values()
        )

        if total <= 0:
            raise RiskScoringError(
                "Risk weights must have "
                "a positive total."
            )

    @staticmethod
    def _validate_score(
        score: float,
    ) -> float:

        score = RiskScorer._clamp(
            score
        )

        if not (
            0.0
            <= score
            <= 100.0
        ):
            raise InvalidRiskScoreError(
                f"Invalid risk score: {score}"
            )

        return round(
            score,
            2,
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

    @staticmethod
    def _number_signed(
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
# Convenience Function
# ============================================================================

def calculate_risk_score(
    company_id: str,
    alert_type: str,
    metrics: Dict[str, Any],
) -> RiskScoreResult:
    """
    Convenience function for one-off risk calculations.
    """

    scorer = RiskScorer()

    return scorer.score(
        company_id=company_id,
        alert_type=alert_type,
        metrics=metrics,
    )


__all__ = [
    "RiskScoringError",
    "InvalidRiskScoreError",
    "RiskBand",
    "RiskAlertType",
    "RiskFactor",
    "RiskScoreResult",
    "RiskScorer",
    "calculate_risk_score",
]