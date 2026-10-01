"""
FinCo AI - Alert Rules Engine

Defines deterministic financial alert rules.

Responsibilities:
- Evaluate financial metrics against alert thresholds
- Detect financial deterioration patterns
- Produce rule evaluation results
- Keep business conditions separate from detection,
  risk scoring, severity, and alert generation

Architecture:

    Financial Data
         │
         ▼
      rules.py
         │
         ├── Financial Profile Down
         ├── Previous Period Decline
         └── Severe Loss Risk
         │
         ▼
      detector.py
         │
         ▼
   risk_scoring.py
         │
         ▼
     severity.py
         │
         ▼
 alert_generator.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# Exceptions
# ============================================================================

class AlertRuleError(Exception):
    """Base exception for alert rule failures."""


class InvalidMetricError(AlertRuleError):
    """Raised when financial metrics are invalid."""


# ============================================================================
# Alert Rule Types
# ============================================================================

class AlertRuleType:
    """Supported financial alert rule types."""

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
# Rule Result
# ============================================================================

@dataclass
class RuleResult:
    """
    Result produced by an individual alert rule.
    """

    rule_name: str

    alert_type: str

    triggered: bool

    company_id: str

    message: str

    score: float = 0.0

    evidence: List[Dict[str, Any]] = field(
        default_factory=list
    )

    metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Alert Rule
# ============================================================================

@dataclass
class AlertRule:
    """
    Configurable rule definition.
    """

    name: str

    alert_type: str

    description: str

    enabled: bool = True

    threshold: Optional[float] = None

    operator: Optional[str] = None

    priority: str = "MEDIUM"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Rules Engine
# ============================================================================

class AlertRulesEngine:
    """
    Deterministic financial alert rules engine.

    Important separation:

        rules.py
            What financial condition constitutes a problem?

        detector.py
            Did that condition occur?

        risk_scoring.py
            How risky is the condition?

        severity.py
            How urgent is the condition?

        alert_generator.py
            How should the alert be represented?
    """

    # ------------------------------------------------------------------------
    # Default thresholds
    # ------------------------------------------------------------------------

    DEFAULT_FINANCIAL_HEALTH_THRESHOLD = 50.0

    DEFAULT_PROFITABILITY_THRESHOLD = 50.0

    DEFAULT_CASH_FLOW_THRESHOLD = 50.0

    DEFAULT_REVENUE_DECLINE_THRESHOLD = -10.0

    DEFAULT_EXPENSE_GROWTH_THRESHOLD = 15.0

    DEFAULT_MARGIN_DECLINE_THRESHOLD = -10.0

    DEFAULT_PREVIOUS_PERIOD_DECLINE_THRESHOLD = -10.0

    DEFAULT_LOSS_PROBABILITY_THRESHOLD = 70.0

    DEFAULT_PROJECTED_LOSS_THRESHOLD = 0.0

    # ------------------------------------------------------------------------
    # Rule names
    # ------------------------------------------------------------------------

    RULE_FINANCIAL_PROFILE_DOWN = (
        "financial_profile_down_rule"
    )

    RULE_PREVIOUS_PERIOD_DECLINE = (
        "previous_period_decline_rule"
    )

    RULE_SEVERE_LOSS_RISK = (
        "severe_loss_risk_rule"
    )

    def __init__(
        self,
        financial_health_threshold: float = (
            DEFAULT_FINANCIAL_HEALTH_THRESHOLD
        ),
        profitability_threshold: float = (
            DEFAULT_PROFITABILITY_THRESHOLD
        ),
        cash_flow_threshold: float = (
            DEFAULT_CASH_FLOW_THRESHOLD
        ),
        revenue_decline_threshold: float = (
            DEFAULT_REVENUE_DECLINE_THRESHOLD
        ),
        expense_growth_threshold: float = (
            DEFAULT_EXPENSE_GROWTH_THRESHOLD
        ),
        margin_decline_threshold: float = (
            DEFAULT_MARGIN_DECLINE_THRESHOLD
        ),
        previous_period_decline_threshold: float = (
            DEFAULT_PREVIOUS_PERIOD_DECLINE_THRESHOLD
        ),
        loss_probability_threshold: float = (
            DEFAULT_LOSS_PROBABILITY_THRESHOLD
        ),
        projected_loss_threshold: float = (
            DEFAULT_PROJECTED_LOSS_THRESHOLD
        ),
    ) -> None:

        self.financial_health_threshold = (
            float(financial_health_threshold)
        )

        self.profitability_threshold = (
            float(profitability_threshold)
        )

        self.cash_flow_threshold = (
            float(cash_flow_threshold)
        )

        self.revenue_decline_threshold = (
            float(revenue_decline_threshold)
        )

        self.expense_growth_threshold = (
            float(expense_growth_threshold)
        )

        self.margin_decline_threshold = (
            float(margin_decline_threshold)
        )

        self.previous_period_decline_threshold = (
            float(previous_period_decline_threshold)
        )

        self.loss_probability_threshold = (
            float(loss_probability_threshold)
        )

        self.projected_loss_threshold = (
            float(projected_loss_threshold)
        )

        self._validate_thresholds()

    # ========================================================================
    # RULE REGISTRY
    # ========================================================================

    def get_rules(self) -> List[AlertRule]:
        """
        Return all configured rules.
        """

        return [
            AlertRule(
                name=self.RULE_FINANCIAL_PROFILE_DOWN,
                alert_type=(
                    AlertRuleType.FINANCIAL_PROFILE_DOWN
                ),
                description=(
                    "Detect significant deterioration "
                    "in the company's financial profile."
                ),
                threshold=(
                    self.financial_health_threshold
                ),
                operator="<",
                priority="HIGH",
            ),
            AlertRule(
                name=self.RULE_PREVIOUS_PERIOD_DECLINE,
                alert_type=(
                    AlertRuleType.PREVIOUS_PERIOD_DECLINE
                ),
                description=(
                    "Detect significant decline compared "
                    "with the previous reporting period."
                ),
                threshold=(
                    self.previous_period_decline_threshold
                ),
                operator="<=",
                priority="MEDIUM",
            ),
            AlertRule(
                name=self.RULE_SEVERE_LOSS_RISK,
                alert_type=(
                    AlertRuleType.SEVERE_LOSS_RISK
                ),
                description=(
                    "Detect high probability or magnitude "
                    "of severe financial loss."
                ),
                threshold=(
                    self.loss_probability_threshold
                ),
                operator=">=",
                priority="CRITICAL",
            ),
        ]

    # ========================================================================
    # MAIN EVALUATION
    # ========================================================================

    def evaluate(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> List[RuleResult]:
        """
        Evaluate all enabled rules.

        Returns one RuleResult for every rule.
        """

        self._validate_company_id(
            company_id
        )

        self._validate_metrics(
            metrics
        )

        return [
            self.evaluate_financial_profile_down(
                company_id,
                metrics,
            ),
            self.evaluate_previous_period_decline(
                company_id,
                metrics,
            ),
            self.evaluate_severe_loss_risk(
                company_id,
                metrics,
            ),
        ]

    # ========================================================================
    # FINANCIAL PROFILE DOWN
    # ========================================================================

    def evaluate_financial_profile_down(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> RuleResult:
        """
        Detect deterioration in the overall financial profile.

        Trigger conditions:

        1. Financial health is below threshold

        OR

        2. At least two significant deterioration signals exist.

        Signals:
        - profitability below threshold
        - cash flow below threshold
        - revenue decline
        - expense growth
        - margin decline
        """

        evidence: List[Dict[str, Any]] = []

        health_score = self._number(
            metrics.get(
                "health_score"
            ),
            default=100.0,
        )

        profitability = self._number(
            metrics.get(
                "profitability"
            ),
            default=100.0,
        )

        cash_flow = self._number(
            metrics.get(
                "cash_flow"
            ),
            default=100.0,
        )

        revenue_change = self._number(
            metrics.get(
                "revenue_change_percent"
            )
        )

        expense_change = self._number(
            metrics.get(
                "expense_change_percent"
            )
        )

        margin_change = self._number(
            metrics.get(
                "margin_change_percent"
            )
        )

        # --------------------------------------------------------------
        # Primary health condition
        # --------------------------------------------------------------

        health_triggered = (
            health_score
            < self.financial_health_threshold
        )

        if health_triggered:

            evidence.append(
                self._evidence(
                    metric="health_score",
                    value=health_score,
                    threshold=(
                        self.financial_health_threshold
                    ),
                    comparison="<",
                    description=(
                        "Financial health score is below "
                        "the configured threshold."
                    ),
                )
            )

        # --------------------------------------------------------------
        # Deterioration signals
        # --------------------------------------------------------------

        signals = 0

        if (
            profitability
            < self.profitability_threshold
        ):

            signals += 1

            evidence.append(
                self._evidence(
                    metric="profitability",
                    value=profitability,
                    threshold=(
                        self.profitability_threshold
                    ),
                    comparison="<",
                    description=(
                        "Profitability has deteriorated."
                    ),
                )
            )

        if (
            cash_flow
            < self.cash_flow_threshold
        ):

            signals += 1

            evidence.append(
                self._evidence(
                    metric="cash_flow",
                    value=cash_flow,
                    threshold=(
                        self.cash_flow_threshold
                    ),
                    comparison="<",
                    description=(
                        "Cash-flow health has deteriorated."
                    ),
                )
            )

        if (
            revenue_change
            <= self.revenue_decline_threshold
        ):

            signals += 1

            evidence.append(
                self._evidence(
                    metric="revenue_change_percent",
                    value=revenue_change,
                    threshold=(
                        self.revenue_decline_threshold
                    ),
                    comparison="<=",
                    description=(
                        "Revenue has declined significantly."
                    ),
                )
            )

        if (
            expense_change
            >= self.expense_growth_threshold
        ):

            signals += 1

            evidence.append(
                self._evidence(
                    metric="expense_change_percent",
                    value=expense_change,
                    threshold=(
                        self.expense_growth_threshold
                    ),
                    comparison=">=",
                    description=(
                        "Expenses have increased significantly."
                    ),
                )
            )

        if (
            margin_change
            <= self.margin_decline_threshold
        ):

            signals += 1

            evidence.append(
                self._evidence(
                    metric="margin_change_percent",
                    value=margin_change,
                    threshold=(
                        self.margin_decline_threshold
                    ),
                    comparison="<=",
                    description=(
                        "Profit margin has declined significantly."
                    ),
                )
            )

        # --------------------------------------------------------------
        # Final rule condition
        # --------------------------------------------------------------

        triggered = (
            health_triggered
            or signals >= 2
        )

        if triggered:

            message = (
                "Financial profile deterioration detected."
            )

        else:

            message = (
                "Financial profile is within configured "
                "risk thresholds."
            )

        score = self._rule_score(
            health_triggered=health_triggered,
            signal_count=signals,
            maximum_signals=5,
        )

        return RuleResult(
            rule_name=(
                self.RULE_FINANCIAL_PROFILE_DOWN
            ),
            alert_type=(
                AlertRuleType.FINANCIAL_PROFILE_DOWN
            ),
            triggered=triggered,
            company_id=company_id,
            message=message,
            score=score,
            evidence=evidence,
            metrics={
                "health_score": health_score,
                "profitability": profitability,
                "cash_flow": cash_flow,
                "revenue_change_percent":
                    revenue_change,
                "expense_change_percent":
                    expense_change,
                "margin_change_percent":
                    margin_change,
            },
            metadata={
                "deterioration_signals":
                    signals,
                "health_triggered":
                    health_triggered,
            },
        )

    # ========================================================================
    # PREVIOUS PERIOD DECLINE
    # ========================================================================

    def evaluate_previous_period_decline(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> RuleResult:
        """
        Detect significant period-over-period decline.

        Expected metrics:

            current_value
            previous_value

        Optional:

            change_percent
            metric
            period
        """

        current_value = self._number(
            metrics.get(
                "current_value"
            )
        )

        previous_value = self._number(
            metrics.get(
                "previous_value"
            )
        )

        supplied_change = metrics.get(
            "change_percent"
        )

        if supplied_change is not None:

            change_percent = self._number(
                supplied_change
            )

        elif previous_value != 0:

            change_percent = (
                (
                    current_value
                    - previous_value
                )
                / abs(previous_value)
            ) * 100.0

        else:

            change_percent = (
                -100.0
                if current_value < 0
                else 0.0
            )

        triggered = (
            change_percent
            <= self.previous_period_decline_threshold
        )

        evidence = [
            self._evidence(
                metric=(
                    metrics.get(
                        "metric",
                        "financial_metric",
                    )
                ),
                value=change_percent,
                threshold=(
                    self.previous_period_decline_threshold
                ),
                comparison="<=",
                description=(
                    "Period-over-period change "
                    "crossed the decline threshold."
                ),
            )
        ]

        if triggered:

            message = (
                "Significant previous-period decline detected."
            )

        else:

            message = (
                "Previous-period performance is "
                "within the configured threshold."
            )

        score = self._decline_rule_score(
            change_percent
        )

        return RuleResult(
            rule_name=(
                self.RULE_PREVIOUS_PERIOD_DECLINE
            ),
            alert_type=(
                AlertRuleType.PREVIOUS_PERIOD_DECLINE
            ),
            triggered=triggered,
            company_id=company_id,
            message=message,
            score=score,
            evidence=evidence,
            metrics={
                "current_value":
                    current_value,
                "previous_value":
                    previous_value,
                "change_percent":
                    change_percent,
            },
            metadata={
                "metric":
                    metrics.get(
                        "metric"
                    ),
                "period":
                    metrics.get(
                        "period"
                    ),
            },
        )

    # ========================================================================
    # SEVERE LOSS RISK
    # ========================================================================

    def evaluate_severe_loss_risk(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> RuleResult:
        """
        Detect severe projected financial loss.

        Trigger conditions:

            loss_probability >= 70%

        OR

            projected_loss > 0
            AND
            projected_profit < 0

        """

        loss_probability = self._number(
            metrics.get(
                "loss_probability"
            )
        )

        projected_loss = self._number(
            metrics.get(
                "projected_loss"
            )
        )

        projected_profit = self._number(
            metrics.get(
                "projected_profit"
            )
        )

        cash_flow = self._number(
            metrics.get(
                "cash_flow"
            )
        )

        probability_triggered = (
            loss_probability
            >= self.loss_probability_threshold
        )

        loss_triggered = (
            projected_loss
            > self.projected_loss_threshold
            and projected_profit < 0
        )

        cash_flow_triggered = (
            cash_flow < 0
        )

        triggered = (
            probability_triggered
            or loss_triggered
        )

        evidence: List[Dict[str, Any]] = []

        if probability_triggered:

            evidence.append(
                self._evidence(
                    metric="loss_probability",
                    value=loss_probability,
                    threshold=(
                        self.loss_probability_threshold
                    ),
                    comparison=">=",
                    description=(
                        "Predicted loss probability "
                        "has reached the severe-risk threshold."
                    ),
                )
            )

        if loss_triggered:

            evidence.append(
                self._evidence(
                    metric="projected_loss",
                    value=projected_loss,
                    threshold=(
                        self.projected_loss_threshold
                    ),
                    comparison=">",
                    description=(
                        "Projected loss is positive while "
                        "projected profit is negative."
                    ),
                )
            )

        if cash_flow_triggered:

            evidence.append(
                self._evidence(
                    metric="cash_flow",
                    value=cash_flow,
                    threshold=0.0,
                    comparison="<",
                    description=(
                        "Projected or current cash flow is negative."
                    ),
                )
            )

        if triggered:

            message = (
                "Severe financial loss risk detected."
            )

        else:

            message = (
                "Severe loss indicators are "
                "within configured thresholds."
            )

        score = self._loss_rule_score(
            loss_probability=loss_probability,
            projected_loss=projected_loss,
            projected_profit=projected_profit,
            cash_flow=cash_flow,
        )

        return RuleResult(
            rule_name=(
                self.RULE_SEVERE_LOSS_RISK
            ),
            alert_type=(
                AlertRuleType.SEVERE_LOSS_RISK
            ),
            triggered=triggered,
            company_id=company_id,
            message=message,
            score=score,
            evidence=evidence,
            metrics={
                "loss_probability":
                    loss_probability,
                "projected_loss":
                    projected_loss,
                "projected_profit":
                    projected_profit,
                "cash_flow":
                    cash_flow,
            },
            metadata={
                "probability_triggered":
                    probability_triggered,
                "loss_triggered":
                    loss_triggered,
                "negative_cash_flow":
                    cash_flow_triggered,
            },
        )

    # ========================================================================
    # SINGLE RULE EVALUATION
    # ========================================================================

    def evaluate_rule(
        self,
        company_id: str,
        rule_name: str,
        metrics: Dict[str, Any],
    ) -> RuleResult:
        """
        Evaluate one specific rule.
        """

        normalized = str(
            rule_name
        ).lower()

        if normalized in (
            self.RULE_FINANCIAL_PROFILE_DOWN,
            AlertRuleType.FINANCIAL_PROFILE_DOWN.lower(),
        ):
            return self.evaluate_financial_profile_down(
                company_id,
                metrics,
            )

        if normalized in (
            self.RULE_PREVIOUS_PERIOD_DECLINE,
            AlertRuleType.PREVIOUS_PERIOD_DECLINE.lower(),
        ):
            return self.evaluate_previous_period_decline(
                company_id,
                metrics,
            )

        if normalized in (
            self.RULE_SEVERE_LOSS_RISK,
            AlertRuleType.SEVERE_LOSS_RISK.lower(),
        ):
            return self.evaluate_severe_loss_risk(
                company_id,
                metrics,
            )

        raise AlertRuleError(
            f"Unknown alert rule: {rule_name}"
        )

    # ========================================================================
    # TRIGGERED RULES ONLY
    # ========================================================================

    def triggered_rules(
        self,
        company_id: str,
        metrics: Dict[str, Any],
    ) -> List[RuleResult]:
        """
        Return only rules that are currently triggered.
        """

        results = self.evaluate(
            company_id,
            metrics,
        )

        return [
            result
            for result in results
            if result.triggered
        ]

    # ========================================================================
    # SCORE HELPERS
    # ========================================================================

    @staticmethod
    def _rule_score(
        health_triggered: bool,
        signal_count: int,
        maximum_signals: int,
    ) -> float:
        """
        Produce a rule-level deterioration score.
        """

        if health_triggered:
            base = 60.0
        else:
            base = 0.0

        signal_score = (
            signal_count
            / max(
                1,
                maximum_signals,
            )
        ) * 40.0

        return min(
            100.0,
            base + signal_score,
        )

    @staticmethod
    def _decline_rule_score(
        change_percent: float,
    ) -> float:
        """
        Convert percentage decline to 0-100 rule score.
        """

        if change_percent >= 0:
            return 0.0

        decline = abs(
            change_percent
        )

        return min(
            100.0,
            (
                decline
                / 30.0
            ) * 100.0,
        )

    @staticmethod
    def _loss_rule_score(
        loss_probability: float,
        projected_loss: float,
        projected_profit: float,
        cash_flow: float,
    ) -> float:
        """
        Calculate severe-loss rule score.
        """

        score = loss_probability

        if projected_profit < 0:
            score = max(
                score,
                75.0,
            )

        if projected_loss > 0:
            score = max(
                score,
                70.0,
            )

        if cash_flow < 0:
            score = max(
                score,
                65.0,
            )

        return min(
            100.0,
            score,
        )

    # ========================================================================
    # EVIDENCE
    # ========================================================================

    @staticmethod
    def _evidence(
        metric: str,
        value: Any,
        threshold: Any,
        comparison: str,
        description: str,
    ) -> Dict[str, Any]:
        """
        Create standardized rule evidence.
        """

        return {
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "comparison": comparison,
            "description": description,
        }

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_thresholds(
        self,
    ) -> None:

        percentage_thresholds = [
            self.financial_health_threshold,
            self.profitability_threshold,
            self.cash_flow_threshold,
            self.loss_probability_threshold,
        ]

        for threshold in percentage_thresholds:

            if not 0 <= threshold <= 100:

                raise AlertRuleError(
                    "Percentage thresholds must "
                    "be between 0 and 100."
                )

        if self.revenue_decline_threshold > 0:
            raise AlertRuleError(
                "revenue_decline_threshold must "
                "be zero or negative."
            )

        if self.margin_decline_threshold > 0:
            raise AlertRuleError(
                "margin_decline_threshold must "
                "be zero or negative."
            )

        if (
            self.previous_period_decline_threshold
            > 0
        ):
            raise AlertRuleError(
                "previous_period_decline_threshold "
                "must be zero or negative."
            )

    @staticmethod
    def _validate_company_id(
        company_id: str,
    ) -> None:

        if not company_id:
            raise AlertRuleError(
                "company_id is required."
            )

    @staticmethod
    def _validate_metrics(
        metrics: Dict[str, Any],
    ) -> None:

        if not isinstance(
            metrics,
            dict,
        ):
            raise InvalidMetricError(
                "metrics must be a dictionary."
            )

    # ========================================================================
    # UTILITIES
    # ========================================================================

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

def evaluate_alert_rules(
    company_id: str,
    metrics: Dict[str, Any],
) -> List[RuleResult]:
    """
    Evaluate all FinCo AI alert rules.
    """

    engine = AlertRulesEngine()

    return engine.evaluate(
        company_id=company_id,
        metrics=metrics,
    )


def get_triggered_alert_rules(
    company_id: str,
    metrics: Dict[str, Any],
) -> List[RuleResult]:
    """
    Return only triggered financial alert rules.
    """

    engine = AlertRulesEngine()

    return engine.triggered_rules(
        company_id=company_id,
        metrics=metrics,
    )


__all__ = [
    "AlertRuleError",
    "InvalidMetricError",
    "AlertRuleType",
    "RuleResult",
    "AlertRule",
    "AlertRulesEngine",
    "evaluate_alert_rules",
    "get_triggered_alert_rules",
]