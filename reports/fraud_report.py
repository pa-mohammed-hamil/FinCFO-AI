"""
FinCo AI - Fraud Investigation Report
=====================================

Generates professional fraud / anomaly investigation reports.

Supports:
    - Fraud detection results
    - Anomaly scores
    - Risk scoring
    - Transaction analysis
    - Model explanations
    - Rule-based findings
    - Suspicious patterns
    - Investigation summaries
    - Risk levels
    - Recommended actions
    - Human-review requirements
    - Audit metadata
    - JSON / Markdown / HTML / Text output

Location:
    backend/app/reports/fraud_report.py

Architecture:

    Transaction Data
          |
          v
    Fraud Detection
          |
          +---- Supervised Model
          |
          +---- Anomaly Model
          |
          +---- Rule Engine
          |
          v
      Risk Scoring
          |
          v
    Explainability
          |
          v
    Fraud Report
          |
       +--+----------------+
       |                   |
       v                   v
    Human Review        Audit Log
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ============================================================================
# Constants
# ============================================================================

RISK_LEVELS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}

FRAUD_LABELS = {
    "fraud",
    "suspicious",
    "anomaly",
    "high_risk",
    "high-risk",
    "flagged",
    "potential_fraud",
}

DEFAULT_CURRENCY = "USD"


# ============================================================================
# Utility Functions
# ============================================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to integer."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safe division."""

    if abs(denominator) < 1e-12:
        return default

    return numerator / denominator


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0

    return statistics.mean(values)


def _round(
    value: Any,
    digits: int = 2,
) -> float:
    return round(
        _safe_float(value),
        digits,
    )


def _format_currency(
    value: float,
    currency: str = DEFAULT_CURRENCY,
) -> str:

    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "INR": "₹",
        "CAD": "C$",
        "AUD": "A$",
    }

    symbol = symbols.get(
        currency.upper(),
        currency.upper() + " ",
    )

    return f"{symbol}{value:,.2f}"


def _format_percent(
    value: Optional[float],
) -> str:

    if value is None:
        return "N/A"

    return f"{value:.2f}%"


def _risk_rank(
    level: str,
) -> int:

    return RISK_LEVELS.get(
        str(level).upper(),
        1,
    )


def _normalize_risk(
    level: Any,
) -> str:

    value = str(level or "LOW").upper()

    if value not in RISK_LEVELS:
        return "LOW"

    return value


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FraudReportConfig:
    """Fraud report configuration."""

    company_name: str = "FinCo Demo Company"

    currency: str = DEFAULT_CURRENCY

    report_title: str = (
        "Fraud & Anomaly Investigation Report"
    )

    high_risk_score: float = 70.0

    critical_risk_score: float = 90.0

    high_anomaly_score: float = 0.70

    critical_anomaly_score: float = 0.90

    high_transaction_amount: float = 100000.0

    require_human_review_for_high_risk: bool = True

    require_human_review_for_critical: bool = True

    include_recommendations: bool = True

    include_transaction_table: bool = True

    include_explanations: bool = True


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class FraudTransaction:
    """Represents a transaction evaluated for fraud."""

    transaction_id: str

    amount: float

    timestamp: Optional[str] = None

    customer_id: Optional[str] = None

    account_id: Optional[str] = None

    merchant: Optional[str] = None

    category: Optional[str] = None

    location: Optional[str] = None

    payment_method: Optional[str] = None

    fraud_probability: Optional[float] = None

    anomaly_score: Optional[float] = None

    risk_score: Optional[float] = None

    fraud_label: Optional[str] = None

    model_prediction: Optional[str] = None

    rule_flags: List[str] = field(
        default_factory=list
    )

    explanation: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_suspicious(self) -> bool:

        if self.fraud_label:
            label = self.fraud_label.lower()

            if label in FRAUD_LABELS:
                return True

        if (
            self.risk_score is not None
            and self.risk_score >= 70
        ):
            return True

        if (
            self.fraud_probability is not None
            and self.fraud_probability >= 0.70
        ):
            return True

        if (
            self.anomaly_score is not None
            and self.anomaly_score >= 0.70
        ):
            return True

        return bool(self.rule_flags)


@dataclass
class FraudFinding:
    """Individual investigation finding."""

    category: str

    severity: str

    title: str

    description: str

    evidence: List[str] = field(
        default_factory=list
    )

    risk_score: float = 0.0

    recommended_action: str = ""

    requires_human_review: bool = False


@dataclass
class FraudMetric:
    """Fraud report KPI."""

    name: str

    value: Any

    unit: str = ""

    status: str = "INFO"

    description: str = ""


@dataclass
class FraudRecommendation:
    """Recommended fraud response."""

    priority: str

    category: str

    action: str

    rationale: str

    expected_impact: str

    requires_human_approval: bool = False


@dataclass
class FraudReportSection:
    """Report section."""

    title: str

    content: str

    metrics: List[FraudMetric] = field(
        default_factory=list
    )


@dataclass
class FraudReport:
    """Complete fraud investigation report."""

    report_id: str

    company_name: str

    report_title: str

    generated_at: str

    investigation_type: str

    overall_status: str

    overall_risk: str

    overall_risk_score: float

    fraud_probability: float

    anomaly_score: float

    summary: str

    metrics: List[FraudMetric] = field(
        default_factory=list
    )

    suspicious_transactions: List[
        FraudTransaction
    ] = field(default_factory=list)

    findings: List[FraudFinding] = field(
        default_factory=list
    )

    recommendations: List[
        FraudRecommendation
    ] = field(default_factory=list)

    sections: List[FraudReportSection] = field(
        default_factory=list
    )

    requires_human_review: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(
        self,
        indent: int = 2,
    ) -> str:

        return json.dumps(
            self.to_dict(),
            indent=indent,
            default=str,
        )


# ============================================================================
# Fraud Report Generator
# ============================================================================

class FraudReportGenerator:
    """
    Main fraud report generator.

    Example:

        generator = FraudReportGenerator()

        report = generator.generate(
            transactions=[
                {
                    "transaction_id": "TX-001",
                    "amount": 150000,
                    "fraud_probability": 0.91,
                    "anomaly_score": 0.88,
                    "risk_score": 94,
                    "fraud_label": "suspicious",
                    "rule_flags": [
                        "Unusually high transaction amount"
                    ],
                }
            ]
        )

        print(report.to_json())
    """

    def __init__(
        self,
        config: Optional[FraudReportConfig] = None,
    ) -> None:

        self.config = (
            config
            or FraudReportConfig()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        transactions: Sequence[
            FraudTransaction
            | Mapping[str, Any]
        ],
        investigation_type: str = "transaction_review",
        model_metrics: Optional[
            Mapping[str, Any]
        ] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> FraudReport:
        """
        Generate a complete fraud report.
        """

        normalized = (
            self._normalize_transactions(
                transactions
            )
        )

        suspicious = [
            transaction
            for transaction in normalized
            if transaction.is_suspicious
        ]

        metrics = self._calculate_metrics(
            transactions=normalized,
            suspicious=suspicious,
            model_metrics=model_metrics or {},
        )

        findings = self._generate_findings(
            transactions=normalized,
            suspicious=suspicious,
        )

        recommendations = []

        if self.config.include_recommendations:

            recommendations = (
                self._generate_recommendations(
                    transactions=normalized,
                    suspicious=suspicious,
                    findings=findings,
                )
            )

        risk_score = self._calculate_overall_risk_score(
            transactions=normalized,
            suspicious=suspicious,
        )

        fraud_probability = (
            self._calculate_average_fraud_probability(
                suspicious
            )
        )

        anomaly_score = (
            self._calculate_average_anomaly_score(
                suspicious
            )
        )

        overall_risk = (
            self._risk_from_score(
                risk_score
            )
        )

        overall_status = (
            self._status_from_risk(
                overall_risk
            )
        )

        requires_human_review = (
            self._requires_human_review(
                overall_risk
            )
        )

        summary = self._build_summary(
            transactions=normalized,
            suspicious=suspicious,
            overall_risk=overall_risk,
            risk_score=risk_score,
            fraud_probability=fraud_probability,
            anomaly_score=anomaly_score,
        )

        sections = self._build_sections(
            metrics=metrics,
            findings=findings,
            recommendations=recommendations,
        )

        report_metadata = dict(
            metadata or {}
        )

        report_metadata.update(
            {
                "total_transactions": len(
                    normalized
                ),
                "suspicious_transactions": len(
                    suspicious
                ),
                "fraud_rate": _round(
                    _safe_divide(
                        len(suspicious),
                        len(normalized),
                    )
                    * 100
                ),
                "model_metrics_available": bool(
                    model_metrics
                ),
                "generator": (
                    "FinCoAI FraudReportGenerator"
                ),
                "version": "1.0.0",
            }
        )

        return FraudReport(
            report_id=self._generate_report_id(),
            company_name=self.config.company_name,
            report_title=self.config.report_title,
            generated_at=datetime.now(
                timezone.utc
            ).isoformat(),
            investigation_type=investigation_type,
            overall_status=overall_status,
            overall_risk=overall_risk,
            overall_risk_score=_round(
                risk_score
            ),
            fraud_probability=_round(
                fraud_probability * 100
            ),
            anomaly_score=_round(
                anomaly_score * 100
            ),
            summary=summary,
            metrics=metrics,
            suspicious_transactions=suspicious,
            findings=findings,
            recommendations=recommendations,
            sections=sections,
            requires_human_review=(
                requires_human_review
            ),
            metadata=report_metadata,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_transactions(
        self,
        transactions: Sequence[
            FraudTransaction
            | Mapping[str, Any]
        ],
    ) -> List[FraudTransaction]:

        result: List[FraudTransaction] = []

        for index, item in enumerate(
            transactions
        ):

            if isinstance(
                item,
                FraudTransaction,
            ):
                result.append(item)
                continue

            data = dict(item)

            result.append(
                FraudTransaction(
                    transaction_id=str(
                        data.get(
                            "transaction_id",
                            data.get(
                                "id",
                                f"TX-{index + 1:05d}",
                            ),
                        )
                    ),
                    amount=_safe_float(
                        data.get(
                            "amount",
                            data.get(
                                "transaction_amount",
                                0,
                            ),
                        )
                    ),
                    timestamp=self._optional_string(
                        data.get(
                            "timestamp",
                            data.get("date"),
                        )
                    ),
                    customer_id=self._optional_string(
                        data.get("customer_id")
                    ),
                    account_id=self._optional_string(
                        data.get("account_id")
                    ),
                    merchant=self._optional_string(
                        data.get("merchant")
                    ),
                    category=self._optional_string(
                        data.get("category")
                    ),
                    location=self._optional_string(
                        data.get("location")
                    ),
                    payment_method=self._optional_string(
                        data.get(
                            "payment_method"
                        )
                    ),
                    fraud_probability=(
                        self._normalize_probability(
                            data.get(
                                "fraud_probability",
                                data.get(
                                    "fraud_probability_score"
                                ),
                            )
                        )
                    ),
                    anomaly_score=(
                        self._normalize_probability(
                            data.get(
                                "anomaly_score"
                            )
                        )
                    ),
                    risk_score=(
                        self._normalize_risk_score(
                            data.get(
                                "risk_score"
                            )
                        )
                    ),
                    fraud_label=self._optional_string(
                        data.get(
                            "fraud_label",
                            data.get("label"),
                        )
                    ),
                    model_prediction=self._optional_string(
                        data.get(
                            "model_prediction"
                        )
                    ),
                    rule_flags=list(
                        data.get(
                            "rule_flags",
                            data.get(
                                "flags",
                                [],
                            ),
                        )
                        or []
                    ),
                    explanation=dict(
                        data.get(
                            "explanation",
                            {},
                        )
                        or {}
                    ),
                    metadata=dict(
                        data.get(
                            "metadata",
                            {},
                        )
                        or {}
                    ),
                )
            )

        return result

    @staticmethod
    def _optional_string(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        return str(value)

    @staticmethod
    def _normalize_probability(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        result = _safe_float(value)

        if result > 1:
            result /= 100.0

        return max(
            0.0,
            min(1.0, result),
        )

    @staticmethod
    def _normalize_risk_score(
        value: Any,
    ) -> Optional[float]:

        if value is None:
            return None

        result = _safe_float(value)

        if 0 <= result <= 1:
            result *= 100

        return max(
            0.0,
            min(100.0, result),
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_metrics(
        self,
        transactions: Sequence[FraudTransaction],
        suspicious: Sequence[FraudTransaction],
        model_metrics: Mapping[str, Any],
    ) -> List[FraudMetric]:

        total = len(transactions)

        suspicious_count = len(
            suspicious
        )

        suspicious_amount = sum(
            transaction.amount
            for transaction in suspicious
        )

        total_amount = sum(
            transaction.amount
            for transaction in transactions
        )

        fraud_rate = (
            _safe_divide(
                suspicious_count,
                total,
            )
            * 100
        )

        amount_exposure = (
            _safe_divide(
                suspicious_amount,
                total_amount,
            )
            * 100
        )

        metrics = [
            FraudMetric(
                name="Transactions Analysed",
                value=total,
                unit="transactions",
                description=(
                    "Total number of transactions "
                    "included in the investigation."
                ),
            ),
            FraudMetric(
                name="Suspicious Transactions",
                value=suspicious_count,
                unit="transactions",
                status=(
                    "WARNING"
                    if suspicious_count
                    else "GOOD"
                ),
                description=(
                    "Transactions flagged by fraud "
                    "models, anomaly detection, "
                    "or business rules."
                ),
            ),
            FraudMetric(
                name="Suspicious Transaction Rate",
                value=_round(fraud_rate),
                unit="%",
                status=(
                    "WARNING"
                    if fraud_rate >= 5
                    else "GOOD"
                ),
                description=(
                    "Percentage of analysed "
                    "transactions classified as suspicious."
                ),
            ),
            FraudMetric(
                name="Total Transaction Value",
                value=_round(
                    total_amount
                ),
                unit=self.config.currency,
                description=(
                    "Total value of analysed transactions."
                ),
            ),
            FraudMetric(
                name="Suspicious Exposure",
                value=_round(
                    suspicious_amount
                ),
                unit=self.config.currency,
                status=(
                    "WARNING"
                    if suspicious_amount > 0
                    else "GOOD"
                ),
                description=(
                    "Total transaction value associated "
                    "with suspicious activity."
                ),
            ),
            FraudMetric(
                name="Suspicious Exposure Ratio",
                value=_round(
                    amount_exposure
                ),
                unit="%",
                status=(
                    "WARNING"
                    if amount_exposure >= 10
                    else "GOOD"
                ),
                description=(
                    "Suspicious transaction value as "
                    "a percentage of total transaction value."
                ),
            ),
        ]

        fraud_probabilities = [
            transaction.fraud_probability
            for transaction in suspicious
            if transaction.fraud_probability
            is not None
        ]

        if fraud_probabilities:

            metrics.append(
                FraudMetric(
                    name="Average Fraud Probability",
                    value=_round(
                        _mean(
                            fraud_probabilities
                        )
                        * 100
                    ),
                    unit="%",
                    description=(
                        "Average fraud probability "
                        "among suspicious transactions."
                    ),
                )
            )

        anomaly_scores = [
            transaction.anomaly_score
            for transaction in suspicious
            if transaction.anomaly_score
            is not None
        ]

        if anomaly_scores:

            metrics.append(
                FraudMetric(
                    name="Average Anomaly Score",
                    value=_round(
                        _mean(
                            anomaly_scores
                        )
                        * 100
                    ),
                    unit="%",
                    description=(
                        "Average anomaly score among "
                        "suspicious transactions."
                    ),
                )
            )

        risk_scores = [
            transaction.risk_score
            for transaction in suspicious
            if transaction.risk_score
            is not None
        ]

        if risk_scores:

            metrics.append(
                FraudMetric(
                    name="Average Risk Score",
                    value=_round(
                        _mean(risk_scores)
                    ),
                    unit="/100",
                    description=(
                        "Average risk score among "
                        "suspicious transactions."
                    ),
                )
            )

        metrics.extend(
            self._model_metrics(
                model_metrics
            )
        )

        return metrics

    @staticmethod
    def _model_metrics(
        model_metrics: Mapping[str, Any],
    ) -> List[FraudMetric]:

        result = []

        supported = {
            "accuracy": (
                "Model Accuracy",
                "%",
            ),
            "precision": (
                "Precision",
                "%",
            ),
            "recall": (
                "Recall",
                "%",
            ),
            "f1": (
                "F1 Score",
                "%",
            ),
            "f1_score": (
                "F1 Score",
                "%",
            ),
            "roc_auc": (
                "ROC-AUC",
                "",
            ),
            "pr_auc": (
                "PR-AUC",
                "",
            ),
            "false_positive_rate": (
                "False Positive Rate",
                "%",
            ),
        }

        for key, value in model_metrics.items():

            normalized = str(key).lower()

            if normalized not in supported:
                continue

            name, unit = supported[
                normalized
            ]

            number = _safe_float(value)

            if (
                unit == "%"
                and number <= 1
            ):
                number *= 100

            result.append(
                FraudMetric(
                    name=name,
                    value=_round(number),
                    unit=unit,
                    description=(
                        f"Fraud model {name}."
                    ),
                )
            )

        return result

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    def _generate_findings(
        self,
        transactions: Sequence[FraudTransaction],
        suspicious: Sequence[FraudTransaction],
    ) -> List[FraudFinding]:

        findings: List[FraudFinding] = []

        if not suspicious:
            findings.append(
                FraudFinding(
                    category="Detection",
                    severity="LOW",
                    title="No Material Suspicious Activity",
                    description=(
                        "No transactions crossed the configured "
                        "fraud or anomaly detection thresholds."
                    ),
                    evidence=[
                        f"{len(transactions)} transactions analysed."
                    ],
                    risk_score=5.0,
                    recommended_action=(
                        "Continue routine monitoring."
                    ),
                )
            )

            return findings

        # --------------------------------------------------------------
        # High-value suspicious transactions
        # --------------------------------------------------------------

        high_value = [
            transaction
            for transaction in suspicious
            if transaction.amount
            >= self.config.high_transaction_amount
        ]

        if high_value:

            amount = sum(
                transaction.amount
                for transaction in high_value
            )

            findings.append(
                FraudFinding(
                    category="Transaction Amount",
                    severity="HIGH",
                    title=(
                        "High-Value Suspicious Transactions"
                    ),
                    description=(
                        f"{len(high_value)} suspicious "
                        "transaction(s) exceed the configured "
                        "high-value threshold."
                    ),
                    evidence=[
                        (
                            f"{transaction.transaction_id}: "
                            f"{_format_currency(transaction.amount, self.config.currency)}"
                        )
                        for transaction in high_value[:10]
                    ],
                    risk_score=80.0,
                    recommended_action=(
                        "Perform enhanced review of high-value "
                        "transactions and supporting documentation."
                    ),
                    requires_human_review=True,
                )
            )

        # --------------------------------------------------------------
        # Fraud probability
        # --------------------------------------------------------------

        high_probability = [
            transaction
            for transaction in suspicious
            if (
                transaction.fraud_probability
                is not None
                and transaction.fraud_probability
                >= 0.70
            )
        ]

        if high_probability:

            findings.append(
                FraudFinding(
                    category="Fraud Model",
                    severity="HIGH",
                    title="High Fraud Probability",
                    description=(
                        f"{len(high_probability)} transaction(s) "
                        "have fraud probabilities above 70%."
                    ),
                    evidence=[
                        (
                            f"{transaction.transaction_id}: "
                            f"{transaction.fraud_probability * 100:.1f}%"
                        )
                        for transaction in high_probability[:10]
                    ],
                    risk_score=82.0,
                    recommended_action=(
                        "Route high-probability transactions "
                        "to fraud investigation."
                    ),
                    requires_human_review=True,
                )
            )

        # --------------------------------------------------------------
        # Anomaly detection
        # --------------------------------------------------------------

        anomalies = [
            transaction
            for transaction in suspicious
            if (
                transaction.anomaly_score
                is not None
                and transaction.anomaly_score
                >= self.config.high_anomaly_score
            )
        ]

        if anomalies:

            findings.append(
                FraudFinding(
                    category="Anomaly Detection",
                    severity="HIGH",
                    title="Significant Transaction Anomalies",
                    description=(
                        f"{len(anomalies)} transaction(s) "
                        "show significant anomalous behaviour."
                    ),
                    evidence=[
                        (
                            f"{transaction.transaction_id}: "
                            f"anomaly score "
                            f"{transaction.anomaly_score * 100:.1f}%"
                        )
                        for transaction in anomalies[:10]
                    ],
                    risk_score=78.0,
                    recommended_action=(
                        "Investigate behavioural and "
                        "transaction-context anomalies."
                    ),
                    requires_human_review=True,
                )
            )

        # --------------------------------------------------------------
        # Rule violations
        # --------------------------------------------------------------

        flagged_rules = [
            transaction
            for transaction in suspicious
            if transaction.rule_flags
        ]

        if flagged_rules:

            all_flags = []

            for transaction in flagged_rules:

                all_flags.extend(
                    transaction.rule_flags
                )

            unique_flags = list(
                dict.fromkeys(
                    all_flags
                )
            )

            findings.append(
                FraudFinding(
                    category="Business Rules",
                    severity="MEDIUM",
                    title="Fraud Rule Violations",
                    description=(
                        f"{len(flagged_rules)} transaction(s) "
                        "triggered one or more fraud rules."
                    ),
                    evidence=[
                        str(flag)
                        for flag in unique_flags[:10]
                    ],
                    risk_score=70.0,
                    recommended_action=(
                        "Review rule-triggered transactions "
                        "against business policy and historical behaviour."
                    ),
                    requires_human_review=True,
                )
            )

        # --------------------------------------------------------------
        # Account concentration
        # --------------------------------------------------------------

        account_counts: Dict[str, int] = {}

        for transaction in suspicious:

            if not transaction.account_id:
                continue

            account_counts[
                transaction.account_id
            ] = (
                account_counts.get(
                    transaction.account_id,
                    0,
                )
                + 1
            )

        concentrated_accounts = [
            (
                account_id,
                count,
            )
            for account_id, count
            in account_counts.items()
            if count >= 3
        ]

        if concentrated_accounts:

            findings.append(
                FraudFinding(
                    category="Account Behaviour",
                    severity="HIGH",
                    title="Repeated Suspicious Activity",
                    description=(
                        "One or more accounts generated "
                        "multiple suspicious transactions."
                    ),
                    evidence=[
                        (
                            f"{account_id}: "
                            f"{count} suspicious transactions"
                        )
                        for account_id, count
                        in concentrated_accounts[:10]
                    ],
                    risk_score=80.0,
                    recommended_action=(
                        "Perform account-level behavioural "
                        "investigation and review recent activity."
                    ),
                    requires_human_review=True,
                )
            )

        # --------------------------------------------------------------
        # Geographic concentration
        # --------------------------------------------------------------

        location_counts: Dict[str, int] = {}

        for transaction in suspicious:

            if not transaction.location:
                continue

            location_counts[
                transaction.location
            ] = (
                location_counts.get(
                    transaction.location,
                    0,
                )
                + 1
            )

        suspicious_locations = [
            (
                location,
                count,
            )
            for location, count
            in location_counts.items()
            if count >= 2
        ]

        if suspicious_locations:

            findings.append(
                FraudFinding(
                    category="Geographic Pattern",
                    severity="MEDIUM",
                    title="Geographic Suspicious Pattern",
                    description=(
                        "Multiple suspicious transactions "
                        "are concentrated in specific locations."
                    ),
                    evidence=[
                        (
                            f"{location}: "
                            f"{count} suspicious transactions"
                        )
                        for location, count
                        in suspicious_locations[:10]
                    ],
                    risk_score=65.0,
                    recommended_action=(
                        "Compare geographic activity against "
                        "normal customer and account behaviour."
                    ),
                    requires_human_review=False,
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        transactions: Sequence[FraudTransaction],
        suspicious: Sequence[FraudTransaction],
        findings: Sequence[FraudFinding],
    ) -> List[FraudRecommendation]:

        recommendations: List[
            FraudRecommendation
        ] = []

        critical = any(
            finding.severity == "CRITICAL"
            for finding in findings
        )

        high = any(
            finding.severity == "HIGH"
            for finding in findings
        )

        if critical:

            recommendations.append(
                FraudRecommendation(
                    priority="CRITICAL",
                    category="Investigation",
                    action=(
                        "Immediately escalate critical "
                        "transactions for human investigation."
                    ),
                    rationale=(
                        "Critical fraud indicators were detected."
                    ),
                    expected_impact=(
                        "Reduces potential financial loss "
                        "and limits continued suspicious activity."
                    ),
                    requires_human_approval=True,
                )
            )

        elif high:

            recommendations.append(
                FraudRecommendation(
                    priority="HIGH",
                    category="Investigation",
                    action=(
                        "Route high-risk transactions to "
                        "the fraud investigation queue."
                    ),
                    rationale=(
                        "Multiple high-severity indicators "
                        "were detected."
                    ),
                    expected_impact=(
                        "Enables rapid review of the highest-risk activity."
                    ),
                    requires_human_approval=True,
                )
            )

        if suspicious:

            recommendations.append(
                FraudRecommendation(
                    priority="HIGH",
                    category="Transaction Monitoring",
                    action=(
                        "Increase monitoring of accounts and "
                        "customers associated with suspicious transactions."
                    ),
                    rationale=(
                        f"{len(suspicious)} suspicious transaction(s) "
                        "were detected."
                    ),
                    expected_impact=(
                        "Improves early detection of repeated "
                        "or escalating suspicious behaviour."
                    ),
                    requires_human_approval=False,
                )
            )

        if any(
            transaction.rule_flags
            for transaction in suspicious
        ):

            recommendations.append(
                FraudRecommendation(
                    priority="MEDIUM",
                    category="Rules",
                    action=(
                        "Review triggered fraud rules and "
                        "validate threshold effectiveness."
                    ),
                    rationale=(
                        "Business rules contributed to "
                        "fraud detection."
                    ),
                    expected_impact=(
                        "Improves detection quality while "
                        "reducing unnecessary alerts."
                    ),
                    requires_human_approval=False,
                )
            )

        recommendations.append(
            FraudRecommendation(
                priority="MEDIUM",
                category="Model Governance",
                action=(
                    "Compare fraud predictions with confirmed "
                    "investigation outcomes."
                ),
                rationale=(
                    "Continuous feedback is required to "
                    "maintain model effectiveness."
                ),
                expected_impact=(
                    "Improves future fraud detection "
                    "precision and recall."
                ),
                requires_human_approval=False,
            )
        )

        recommendations.append(
            FraudRecommendation(
                priority="LOW",
                category="Audit",
                action=(
                    "Record detection results, decisions, "
                    "and investigator outcomes in the audit trail."
                ),
                rationale=(
                    "Fraud decisions should remain traceable "
                    "for governance and compliance."
                ),
                expected_impact=(
                    "Improves accountability and "
                    "investigation reproducibility."
                ),
                requires_human_approval=False,
            )
        )

        return recommendations

    # ------------------------------------------------------------------
    # Overall risk
    # ------------------------------------------------------------------

    def _calculate_overall_risk_score(
        self,
        transactions: Sequence[FraudTransaction],
        suspicious: Sequence[FraudTransaction],
    ) -> float:

        if not suspicious:
            return 0.0

        scores = []

        for transaction in suspicious:

            components = []

            if transaction.risk_score is not None:
                components.append(
                    transaction.risk_score
                )

            if transaction.fraud_probability is not None:
                components.append(
                    transaction.fraud_probability
                    * 100
                )

            if transaction.anomaly_score is not None:
                components.append(
                    transaction.anomaly_score
                    * 100
                )

            if components:

                scores.append(
                    _mean(components)
                )

        if not scores:
            return 50.0

        average_score = _mean(scores)

        concentration = (
            _safe_divide(
                len(suspicious),
                len(transactions),
            )
        )

        # Increase overall risk when suspicious
        # activity is widespread.
        concentration_adjustment = min(
            15.0,
            concentration * 100,
        )

        return min(
            100.0,
            average_score
            + concentration_adjustment,
        )

    @staticmethod
    def _risk_from_score(
        score: float,
    ) -> str:

        if score >= 90:
            return "CRITICAL"

        if score >= 70:
            return "HIGH"

        if score >= 40:
            return "MEDIUM"

        return "LOW"

    @staticmethod
    def _status_from_risk(
        risk: str,
    ) -> str:

        return {
            "LOW": "HEALTHY",
            "MEDIUM": "ATTENTION",
            "HIGH": "WARNING",
            "CRITICAL": "CRITICAL",
        }.get(
            risk,
            "UNKNOWN",
        )

    def _requires_human_review(
        self,
        risk: str,
    ) -> bool:

        if (
            risk == "CRITICAL"
            and self.config
            .require_human_review_for_critical
        ):
            return True

        if (
            risk == "HIGH"
            and self.config
            .require_human_review_for_high_risk
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Probability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_average_fraud_probability(
        transactions: Sequence[
            FraudTransaction
        ],
    ) -> float:

        values = [
            transaction.fraud_probability
            for transaction in transactions
            if transaction.fraud_probability
            is not None
        ]

        return (
            _mean(values)
            if values
            else 0.0
        )

    @staticmethod
    def _calculate_average_anomaly_score(
        transactions: Sequence[
            FraudTransaction
        ],
    ) -> float:

        values = [
            transaction.anomaly_score
            for transaction in transactions
            if transaction.anomaly_score
            is not None
        ]

        return (
            _mean(values)
            if values
            else 0.0
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        transactions: Sequence[
            FraudTransaction
        ],
        suspicious: Sequence[
            FraudTransaction
        ],
        overall_risk: str,
        risk_score: float,
        fraud_probability: float,
        anomaly_score: float,
    ) -> str:

        total_amount = sum(
            transaction.amount
            for transaction in transactions
        )

        suspicious_amount = sum(
            transaction.amount
            for transaction in suspicious
        )

        if not suspicious:

            return (
                f"The investigation analysed "
                f"{len(transactions)} transaction(s) "
                f"with total value "
                f"{_format_currency(total_amount, self.config.currency)}. "
                "No material suspicious activity was detected "
                "under the configured thresholds."
            )

        return (
            f"The investigation analysed "
            f"{len(transactions)} transaction(s), "
            f"of which {len(suspicious)} were flagged as suspicious. "
            f"Suspicious exposure is "
            f"{_format_currency(suspicious_amount, self.config.currency)}. "
            f"The overall risk score is {risk_score:.1f}/100 "
            f"with an overall risk classification of "
            f"{overall_risk}. "
            f"Average fraud probability among suspicious "
            f"transactions is {fraud_probability * 100:.1f}%, "
            f"while the average anomaly score is "
            f"{anomaly_score * 100:.1f}%."
        )

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _build_sections(
        self,
        metrics: Sequence[FraudMetric],
        findings: Sequence[FraudFinding],
        recommendations: Sequence[
            FraudRecommendation
        ],
    ) -> List[FraudReportSection]:

        sections = [
            FraudReportSection(
                title="Fraud Overview",
                content=(
                    "Summary of transaction-level fraud "
                    "and anomaly detection results."
                ),
                metrics=list(metrics),
            ),
            FraudReportSection(
                title="Investigation Findings",
                content=(
                    f"{len(findings)} investigation finding(s) "
                    "were generated."
                ),
            ),
        ]

        if recommendations:

            sections.append(
                FraudReportSection(
                    title="Recommended Actions",
                    content=(
                        f"{len(recommendations)} recommended "
                        "action(s) were generated."
                    ),
                )
            )

        return sections

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_markdown(
        self,
        report: FraudReport,
    ) -> str:

        lines = [
            f"# {report.report_title}",
            "",
            f"**Company:** {report.company_name}",
            f"**Report ID:** {report.report_id}",
            f"**Generated:** {report.generated_at}",
            f"**Investigation:** "
            f"{report.investigation_type}",
            "",
            "## Executive Summary",
            "",
            report.summary,
            "",
            "## Risk Overview",
            "",
            f"- **Status:** {report.overall_status}",
            f"- **Risk:** {report.overall_risk}",
            f"- **Risk Score:** "
            f"{report.overall_risk_score:.2f}/100",
            f"- **Fraud Probability:** "
            f"{report.fraud_probability:.2f}%",
            f"- **Anomaly Score:** "
            f"{report.anomaly_score:.2f}%",
            f"- **Human Review Required:** "
            f"{'Yes' if report.requires_human_review else 'No'}",
            "",
        ]

        # Metrics
        if report.metrics:

            lines.extend(
                [
                    "## Fraud Metrics",
                    "",
                    "| Metric | Value | Unit | Status |",
                    "|---|---:|---|---|",
                ]
            )

            for metric in report.metrics:

                lines.append(
                    f"| {metric.name} | "
                    f"{metric.value} | "
                    f"{metric.unit} | "
                    f"{metric.status} |"
                )

            lines.append("")

        # Suspicious transactions
        if (
            report.suspicious_transactions
            and self.config.include_transaction_table
        ):

            lines.extend(
                [
                    "## Suspicious Transactions",
                    "",
                    "| ID | Amount | Fraud Probability | "
                    "Anomaly | Risk | Label |",
                    "|---|---:|---:|---:|---:|---|",
                ]
            )

            for transaction in (
                report.suspicious_transactions
            ):

                fraud_probability = (
                    _format_percent(
                        transaction.fraud_probability
                        * 100
                    )
                    if transaction.fraud_probability
                    is not None
                    else "N/A"
                )

                anomaly = (
                    _format_percent(
                        transaction.anomaly_score
                        * 100
                    )
                    if transaction.anomaly_score
                    is not None
                    else "N/A"
                )

                risk = (
                    f"{transaction.risk_score:.1f}"
                    if transaction.risk_score
                    is not None
                    else "N/A"
                )

                lines.append(
                    f"| {transaction.transaction_id} | "
                    f"{_format_currency(transaction.amount, self.config.currency)} | "
                    f"{fraud_probability} | "
                    f"{anomaly} | "
                    f"{risk} | "
                    f"{transaction.fraud_label or '-'} |"
                )

            lines.append("")

        # Findings
        if report.findings:

            lines.extend(
                [
                    "## Investigation Findings",
                    "",
                ]
            )

            for index, finding in enumerate(
                report.findings,
                start=1,
            ):

                lines.extend(
                    [
                        f"### {index}. {finding.title}",
                        "",
                        f"**Severity:** {finding.severity}",
                        "",
                        finding.description,
                        "",
                    ]
                )

                if finding.evidence:

                    lines.append(
                        "**Evidence:**"
                    )

                    for evidence in finding.evidence:

                        lines.append(
                            f"- {evidence}"
                        )

                    lines.append("")

                lines.extend(
                    [
                        f"**Recommended Action:** "
                        f"{finding.recommended_action}",
                        "",
                        f"**Human Review:** "
                        f"{'Yes' if finding.requires_human_review else 'No'}",
                        "",
                    ]
                )

        # Recommendations
        if report.recommendations:

            lines.extend(
                [
                    "## Recommended Actions",
                    "",
                ]
            )

            for index, recommendation in enumerate(
                report.recommendations,
                start=1,
            ):

                lines.extend(
                    [
                        f"{index}. **{recommendation.action}**",
                        f"   - Priority: "
                        f"{recommendation.priority}",
                        f"   - Category: "
                        f"{recommendation.category}",
                        f"   - Rationale: "
                        f"{recommendation.rationale}",
                        f"   - Expected Impact: "
                        f"{recommendation.expected_impact}",
                        f"   - Human Approval: "
                        f"{'Yes' if recommendation.requires_human_approval else 'No'}",
                        "",
                    ]
                )

        lines.extend(
            [
                "---",
                "",
                "*Generated by FinCo AI — "
                "Financial Intelligence & Decision Copilot.*",
            ]
        )

        return "\n".join(lines)

    def to_text(
        self,
        report: FraudReport,
    ) -> str:

        lines = [
            report.report_title,
            "=" * len(report.report_title),
            "",
            f"Company: {report.company_name}",
            f"Report ID: {report.report_id}",
            f"Investigation: "
            f"{report.investigation_type}",
            f"Status: {report.overall_status}",
            f"Risk: {report.overall_risk}",
            f"Risk Score: "
            f"{report.overall_risk_score:.2f}/100",
            f"Fraud Probability: "
            f"{report.fraud_probability:.2f}%",
            f"Anomaly Score: "
            f"{report.anomaly_score:.2f}%",
            f"Human Review Required: "
            f"{report.requires_human_review}",
            "",
            "SUMMARY",
            "-------",
            report.summary,
            "",
        ]

        if report.findings:

            lines.extend(
                [
                    "FINDINGS",
                    "--------",
                ]
            )

            for finding in report.findings:

                lines.append(
                    f"[{finding.severity}] "
                    f"{finding.title}: "
                    f"{finding.description}"
                )

            lines.append("")

        if report.recommendations:

            lines.extend(
                [
                    "RECOMMENDATIONS",
                    "---------------",
                ]
            )

            for recommendation in (
                report.recommendations
            ):

                lines.append(
                    f"[{recommendation.priority}] "
                    f"{recommendation.action}"
                )

        return "\n".join(lines)

    def to_html(
        self,
        report: FraudReport,
    ) -> str:

        metric_rows = ""

        for metric in report.metrics:

            metric_rows += f"""
            <tr>
                <td>{html.escape(metric.name)}</td>
                <td>{html.escape(str(metric.value))}</td>
                <td>{html.escape(metric.unit)}</td>
                <td>{html.escape(metric.status)}</td>
            </tr>
            """

        transaction_rows = ""

        for transaction in (
            report.suspicious_transactions
        ):

            fraud_probability = (
                f"{transaction.fraud_probability * 100:.1f}%"
                if transaction.fraud_probability
                is not None
                else "N/A"
            )

            anomaly = (
                f"{transaction.anomaly_score * 100:.1f}%"
                if transaction.anomaly_score
                is not None
                else "N/A"
            )

            risk = (
                f"{transaction.risk_score:.1f}"
                if transaction.risk_score
                is not None
                else "N/A"
            )

            transaction_rows += f"""
            <tr>
                <td>{html.escape(transaction.transaction_id)}</td>
                <td>
                    {html.escape(
                        _format_currency(
                            transaction.amount,
                            self.config.currency,
                        )
                    )}
                </td>
                <td>{fraud_probability}</td>
                <td>{anomaly}</td>
                <td>{risk}</td>
                <td>
                    {html.escape(
                        transaction.fraud_label or "-"
                    )}
                </td>
            </tr>
            """

        finding_html = ""

        for finding in report.findings:

            evidence_html = "".join(
                f"<li>{html.escape(str(evidence))}</li>"
                for evidence in finding.evidence
            )

            finding_html += f"""
            <article class="finding">
                <h3>{html.escape(finding.title)}</h3>

                <p>
                    <strong>Severity:</strong>
                    {html.escape(finding.severity)}
                </p>

                <p>
                    {html.escape(finding.description)}
                </p>

                <p>
                    <strong>Evidence</strong>
                </p>

                <ul>
                    {evidence_html}
                </ul>

                <p>
                    <strong>Recommended Action:</strong>
                    {html.escape(
                        finding.recommended_action
                    )}
                </p>

                <p>
                    <strong>Human Review:</strong>
                    {"Yes" if finding.requires_human_review else "No"}
                </p>
            </article>
            """

        recommendation_html = ""

        for recommendation in (
            report.recommendations
        ):

            recommendation_html += f"""
            <article class="recommendation">
                <h3>
                    {html.escape(
                        recommendation.action
                    )}
                </h3>

                <p>
                    <strong>Priority:</strong>
                    {html.escape(
                        recommendation.priority
                    )}
                </p>

                <p>
                    <strong>Category:</strong>
                    {html.escape(
                        recommendation.category
                    )}
                </p>

                <p>
                    <strong>Rationale:</strong>
                    {html.escape(
                        recommendation.rationale
                    )}
                </p>

                <p>
                    <strong>Expected Impact:</strong>
                    {html.escape(
                        recommendation.expected_impact
                    )}
                </p>

                <p>
                    <strong>Human Approval:</strong>
                    {"Yes" if recommendation.requires_human_approval else "No"}
                </p>
            </article>
            """

        return f"""<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
{html.escape(report.report_title)}
</title>

<style>

body {{
    margin: 0;
    padding: 32px;

    font-family:
        Inter,
        Arial,
        Helvetica,
        sans-serif;

    background: #f4f6f8;
    color: #17202a;
}}

.container {{
    max-width: 1200px;

    margin: auto;

    background: #ffffff;

    padding: 40px;

    border-radius: 18px;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.08);
}}

h1 {{
    margin-bottom: 8px;
}}

h2 {{
    margin-top: 38px;

    padding-bottom: 10px;

    border-bottom:
        1px solid #e5e7eb;
}}

.summary {{
    padding: 20px;

    margin-top: 20px;

    background: #f8fafc;

    border-radius: 12px;

    line-height: 1.7;
}}

.status-grid {{
    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 16px;

    margin: 24px 0;
}}

.card {{
    background: #f8fafc;

    padding: 20px;

    border-radius: 12px;
}}

.card-label {{
    color: #6b7280;

    font-size: 13px;
}}

.card-value {{
    display: block;

    margin-top: 8px;

    font-size: 24px;

    font-weight: 700;
}}

table {{
    width: 100%;

    border-collapse: collapse;

    margin-top: 16px;
}}

th,
td {{
    padding: 12px;

    border-bottom:
        1px solid #e5e7eb;

    text-align: left;
}}

th {{
    background: #f8fafc;
}}

.finding {{
    margin: 16px 0;

    padding: 20px;

    border-left:
        5px solid #dc2626;

    background: #fef2f2;

    border-radius: 10px;
}}

.recommendation {{
    margin: 16px 0;

    padding: 20px;

    border-left:
        5px solid #059669;

    background: #ecfdf5;

    border-radius: 10px;
}}

.footer {{
    margin-top: 40px;

    color: #6b7280;

    font-size: 13px;
}}

@media (max-width: 800px) {{

    body {{
        padding: 12px;
    }}

    .container {{
        padding: 20px;
    }}

    .status-grid {{
        grid-template-columns: 1fr;
    }}

    table {{
        display: block;
        overflow-x: auto;
    }}
}}

</style>

</head>

<body>

<div class="container">

<h1>
{html.escape(report.report_title)}
</h1>

<p>
<strong>Company:</strong>
{html.escape(report.company_name)}
</p>

<p>
<strong>Generated:</strong>
{html.escape(report.generated_at)}
</p>

<div class="summary">

<strong>Executive Summary</strong>

<p>
{html.escape(report.summary)}
</p>

</div>

<div class="status-grid">

<div class="card">
<span class="card-label">
Status
</span>

<span class="card-value">
{html.escape(report.overall_status)}
</span>
</div>

<div class="card">
<span class="card-label">
Risk
</span>

<span class="card-value">
{html.escape(report.overall_risk)}
</span>
</div>

<div class="card">
<span class="card-label">
Risk Score
</span>

<span class="card-value">
{report.overall_risk_score:.1f}/100
</span>
</div>

<div class="card">
<span class="card-label">
Human Review
</span>

<span class="card-value">
{"Required" if report.requires_human_review else "Not Required"}
</span>
</div>

</div>

<h2>
Fraud Metrics
</h2>

<table>

<thead>

<tr>
<th>Metric</th>
<th>Value</th>
<th>Unit</th>
<th>Status</th>
</tr>

</thead>

<tbody>
{metric_rows}
</tbody>

</table>

<h2>
Suspicious Transactions
</h2>

<table>

<thead>

<tr>
<th>Transaction</th>
<th>Amount</th>
<th>Fraud Probability</th>
<th>Anomaly</th>
<th>Risk</th>
<th>Label</th>
</tr>

</thead>

<tbody>
{transaction_rows}
</tbody>

</table>

<h2>
Investigation Findings
</h2>

{finding_html or "<p>No material findings.</p>"}

<h2>
Recommended Actions
</h2>

{recommendation_html or "<p>No recommendations.</p>"}

<div class="footer">
FinCo AI — Financial Intelligence &amp;
Decision Copilot
</div>

</div>

</body>

</html>
"""

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

    def save_json(
        self,
        report: FraudReport,
        path: str | Path,
    ) -> Path:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            report.to_json(),
            encoding="utf-8",
        )

        return target

    def save_markdown(
        self,
        report: FraudReport,
        path: str | Path,
    ) -> Path:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            self.to_markdown(report),
            encoding="utf-8",
        )

        return target

    def save_text(
        self,
        report: FraudReport,
        path: str | Path,
    ) -> Path:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            self.to_text(report),
            encoding="utf-8",
        )

        return target

    def save_html(
        self,
        report: FraudReport,
        path: str | Path,
    ) -> Path:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            self.to_html(report),
            encoding="utf-8",
        )

        return target

    # ------------------------------------------------------------------
    # IDs
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_report_id() -> str:

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d%H%M%S")

        return (
            "FINCO-FRAUD-"
            f"{timestamp}"
        )


# ============================================================================
# Convenience API
# ============================================================================

def generate_fraud_report(
    transactions: Sequence[
        FraudTransaction
        | Mapping[str, Any]
    ],
    investigation_type: str = "transaction_review",
    model_metrics: Optional[
        Mapping[str, Any]
    ] = None,
    company_name: str = "FinCo Demo Company",
    currency: str = DEFAULT_CURRENCY,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> FraudReport:
    """
    Convenience API for FastAPI, fraud_service,
    fraud_agent, and report_generator.
    """

    config = FraudReportConfig(
        company_name=company_name,
        currency=currency,
    )

    generator = FraudReportGenerator(
        config=config
    )

    return generator.generate(
        transactions=transactions,
        investigation_type=investigation_type,
        model_metrics=model_metrics,
        metadata=metadata,
    )


# ============================================================================
# Transaction Comparison Helper
# ============================================================================

def compare_fraud_results(
    predictions: Sequence[Any],
    actual_labels: Sequence[Any],
) -> Dict[str, Any]:
    """
    Lightweight classification metrics helper.

    Expected labels:
        0 / 1
        False / True
        normal / fraud
        legitimate / suspicious
    """

    pairs = list(
        zip(
            predictions,
            actual_labels,
        )
    )

    if not pairs:

        return {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
        }

    def to_binary(
        value: Any,
    ) -> int:

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, float)):
            return 1 if value else 0

        normalized = str(
            value
        ).strip().lower()

        return int(
            normalized in {
                "1",
                "true",
                "fraud",
                "suspicious",
                "anomaly",
                "positive",
                "yes",
            }
        )

    tp = tn = fp = fn = 0

    for prediction, actual in pairs:

        predicted = to_binary(
            prediction
        )

        actual_value = to_binary(
            actual
        )

        if predicted == 1 and actual_value == 1:
            tp += 1

        elif predicted == 0 and actual_value == 0:
            tn += 1

        elif predicted == 1 and actual_value == 0:
            fp += 1

        else:
            fn += 1

    precision = _safe_divide(
        tp,
        tp + fp,
    )

    recall = _safe_divide(
        tp,
        tp + fn,
    )

    f1 = _safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    accuracy = _safe_divide(
        tp + tn,
        len(pairs),
    )

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": _round(
            precision * 100
        ),
        "recall": _round(
            recall * 100
        ),
        "f1": _round(
            f1 * 100
        ),
        "accuracy": _round(
            accuracy * 100
        ),
    }


# ============================================================================
# Demo
# ============================================================================

def create_demo_report() -> FraudReport:
    """
    Create realistic demo fraud investigation data.
    """

    transactions = [
        {
            "transaction_id": "TX-1001",
            "amount": 12500,
            "timestamp": "2026-09-10T10:30:00Z",
            "customer_id": "CUS-001",
            "account_id": "ACC-001",
            "merchant": "Global Electronics",
            "category": "Electronics",
            "location": "Mumbai",
            "payment_method": "card",
            "fraud_probability": 0.12,
            "anomaly_score": 0.18,
            "risk_score": 15,
            "fraud_label": "normal",
        },
        {
            "transaction_id": "TX-1002",
            "amount": 185000,
            "timestamp": "2026-09-10T23:47:00Z",
            "customer_id": "CUS-002",
            "account_id": "ACC-002",
            "merchant": "International Services",
            "category": "Services",
            "location": "Delhi",
            "payment_method": "wire",
            "fraud_probability": 0.93,
            "anomaly_score": 0.89,
            "risk_score": 95,
            "fraud_label": "suspicious",
            "rule_flags": [
                "Unusually high transaction amount",
                "Unusual transaction time",
                "Behavioural deviation",
            ],
            "explanation": {
                "top_features": [
                    "transaction_amount",
                    "transaction_hour",
                    "customer_velocity",
                ]
            },
        },
        {
            "transaction_id": "TX-1003",
            "amount": 76000,
            "timestamp": "2026-09-11T02:15:00Z",
            "customer_id": "CUS-002",
            "account_id": "ACC-002",
            "merchant": "International Services",
            "category": "Services",
            "location": "Delhi",
            "payment_method": "wire",
            "fraud_probability": 0.82,
            "anomaly_score": 0.77,
            "risk_score": 84,
            "fraud_label": "suspicious",
            "rule_flags": [
                "Repeated high-value activity",
                "Unusual transaction time",
            ],
        },
        {
            "transaction_id": "TX-1004",
            "amount": 3500,
            "timestamp": "2026-09-11T11:15:00Z",
            "customer_id": "CUS-003",
            "account_id": "ACC-003",
            "merchant": "Retail Store",
            "category": "Retail",
            "location": "Kochi",
            "payment_method": "card",
            "fraud_probability": 0.04,
            "anomaly_score": 0.08,
            "risk_score": 8,
            "fraud_label": "normal",
        },
        {
            "transaction_id": "TX-1005",
            "amount": 142000,
            "timestamp": "2026-09-11T01:48:00Z",
            "customer_id": "CUS-004",
            "account_id": "ACC-004",
            "merchant": "Overseas Trading",
            "category": "Trading",
            "location": "Singapore",
            "payment_method": "wire",
            "fraud_probability": 0.88,
            "anomaly_score": 0.92,
            "risk_score": 91,
            "fraud_label": "suspicious",
            "rule_flags": [
                "Cross-border anomaly",
                "High transaction amount",
            ],
        },
    ]

    return generate_fraud_report(
        transactions=transactions,
        investigation_type="enterprise_fraud_review",
        model_metrics={
            "accuracy": 0.96,
            "precision": 0.91,
            "recall": 0.87,
            "f1": 0.89,
            "roc_auc": 0.95,
            "pr_auc": 0.92,
        },
        company_name="FinCo Demo Corporation",
        currency="USD",
        metadata={
            "fraud_model": (
                "FinCo Hybrid Fraud Detection Model"
            ),
            "model_version": "1.0.0",
            "detection_modes": [
                "supervised",
                "anomaly",
                "rules",
            ],
            "environment": "demo",
        },
    )


# ============================================================================
# CLI
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate FinCo AI fraud investigation reports."
        )
    )

    parser.add_argument(
        "--output",
        default="reports/fraud_report",
        help="Output file prefix.",
    )

    parser.add_argument(
        "--format",
        choices=[
            "json",
            "markdown",
            "text",
            "html",
            "all",
        ],
        default="all",
        help="Output report format.",
    )

    args = parser.parse_args()

    report = create_demo_report()

    generator = FraudReportGenerator()

    output = Path(
        args.output
    )

    if args.format in {
        "json",
        "all",
    }:

        path = generator.save_json(
            report,
            output.with_suffix(".json"),
        )

        print(
            f"JSON report: {path}"
        )

    if args.format in {
        "markdown",
        "all",
    }:

        path = generator.save_markdown(
            report,
            output.with_suffix(".md"),
        )

        print(
            f"Markdown report: {path}"
        )

    if args.format in {
        "text",
        "all",
    }:

        path = generator.save_text(
            report,
            output.with_suffix(".txt"),
        )

        print(
            f"Text report: {path}"
        )

    if args.format in {
        "html",
        "all",
    }:

        path = generator.save_html(
            report,
            output.with_suffix(".html"),
        )

        print(
            f"HTML report: {path}"
        )

    print()
    print(
        f"Report ID: {report.report_id}"
    )

    print(
        f"Status: {report.overall_status}"
    )

    print(
        f"Risk: {report.overall_risk}"
    )

    print(
        f"Risk Score: "
        f"{report.overall_risk_score:.1f}/100"
    )

    print(
        f"Suspicious Transactions: "
        f"{len(report.suspicious_transactions)}"
    )

    print(
        f"Human Review Required: "
        f"{report.requires_human_review}"
    )


if __name__ == "__main__":
    main()