"""
FinCo AI - Risk Recommendations Engine
=======================================

Path:
    backend/app/recommendations/risk_recommendations.py

Purpose:
    Generate explainable, risk-focused recommendations from financial,
    fraud, liquidity, transaction, and operational risk signals.

Design:
    Risk Signals
        ↓
    Risk Classification
        ↓
    Severity + Exposure
        ↓
    Recommendation Generation
        ↓
    Confidence + Impact
        ↓
    Human Review
        ↓
    Audit / Recommendation Service

Important:
    This module is advisory. It does not automatically approve, reject,
    block, transfer, or otherwise execute financial actions.

Typical risk areas:
    - Liquidity risk
    - Cash-flow risk
    - Credit / receivables risk
    - Leverage risk
    - Concentration risk
    - Fraud / anomaly risk
    - Revenue risk
    - Margin risk
    - Budget risk
    - Operational financial risk
    - Data-quality / control risk
    - Scenario / forecast risk
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ============================================================================
# Constants
# ============================================================================

DEFAULT_CURRENCY = "USD"

PRIORITY_LEVELS = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

RISK_LEVELS = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

RISK_TYPES = {
    "liquidity",
    "cash_flow",
    "credit",
    "receivables",
    "leverage",
    "concentration",
    "fraud",
    "anomaly",
    "revenue",
    "margin",
    "budget",
    "operational",
    "forecast",
    "data_quality",
    "financial_health",
    "compliance",
}


# ============================================================================
# Utility Functions
# ============================================================================


def _utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    """Return ISO-8601 UTC timestamp."""
    return _utc_now().isoformat()


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert value to float."""

    if value is None:
        return default

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
    """Safely convert value to integer."""

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def _safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safely divide two values."""

    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)

    if abs(denominator) < 1e-12:
        return default

    return numerator / denominator


def _percentage(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Return percentage value."""

    return _safe_divide(
        numerator,
        denominator,
        default,
    ) * 100.0


def _round(
    value: Any,
    digits: int = 2,
) -> float:
    """Safely round numeric value."""

    return round(
        _safe_float(value),
        digits,
    )


def _normalize_text(
    value: Any,
) -> str:
    """Normalize free-form text."""

    if value is None:
        return ""

    return " ".join(
        str(value).strip().split()
    )


def _normalize_priority(
    value: Any,
) -> str:
    """Normalize priority."""

    value = _normalize_text(value).lower()

    if value in PRIORITY_LEVELS:
        return value

    return "medium"


def _normalize_risk(
    value: Any,
) -> str:
    """Normalize risk severity."""

    value = _normalize_text(value).lower()

    if value in RISK_LEVELS:
        return value

    return "medium"


def _json_safe(
    value: Any,
) -> Any:
    """Convert arbitrary values into JSON-safe values."""

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, bool),
    ):
        return value

    if isinstance(value, float):

        if math.isfinite(value):
            return value

        return None

    if isinstance(
        value,
        (datetime, date),
    ):
        return value.isoformat()

    if isinstance(value, Mapping):

        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return [
            _json_safe(item)
            for item in value
        ]

    if hasattr(
        value,
        "model_dump",
    ):

        return _json_safe(
            value.model_dump()
        )

    if hasattr(
        value,
        "dict",
    ):

        return _json_safe(
            value.dict()
        )

    if hasattr(
        value,
        "__dict__",
    ):

        return _json_safe(
            vars(value)
        )

    return str(value)


def _get(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read a field from dict or object."""

    if source is None:
        return default

    if isinstance(
        source,
        Mapping,
    ):

        return source.get(
            key,
            default,
        )

    return getattr(
        source,
        key,
        default,
    )


def _first(
    source: Any,
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """Return first available field."""

    for key in keys:

        value = _get(
            source,
            key,
            None,
        )

        if value is not None:
            return value

    return default


def _mean(
    values: Iterable[float],
) -> float:
    """Calculate safe mean."""

    values = [
        _safe_float(value)
        for value in values
        if value is not None
    ]

    if not values:
        return 0.0

    return statistics.mean(values)


def _format_currency(
    amount: float,
    currency: str = DEFAULT_CURRENCY,
) -> str:
    """Format currency."""

    return (
        f"{currency} "
        f"{_safe_float(amount):,.2f}"
    )


def _format_percent(
    value: float,
) -> str:
    """Format percentage."""

    return (
        f"{_safe_float(value):,.2f}%"
    )


def _hash_id(
    *parts: Any,
) -> str:
    """Generate deterministic recommendation ID."""

    raw = "|".join(
        str(part)
        for part in parts
    )

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"RISKREC-{digest.upper()}"
    )


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RiskRecommendationConfig:
    """Configuration for risk recommendation generation."""

    currency: str = DEFAULT_CURRENCY

    # Liquidity
    current_ratio_warning: float = 1.20
    current_ratio_critical: float = 1.00

    quick_ratio_warning: float = 1.00
    quick_ratio_critical: float = 0.80

    cash_runway_warning_months: float = 6.0
    cash_runway_critical_months: float = 3.0

    # Receivables
    dso_warning_days: float = 60.0
    dso_high_days: float = 75.0
    dso_critical_days: float = 90.0

    overdue_receivables_warning_pct: float = 10.0
    overdue_receivables_critical_pct: float = 25.0

    # Leverage
    debt_to_equity_warning: float = 2.0
    debt_to_equity_critical: float = 3.0

    interest_coverage_warning: float = 2.0
    interest_coverage_critical: float = 1.0

    # Concentration
    concentration_warning_pct: float = 30.0
    concentration_critical_pct: float = 50.0

    # Fraud
    fraud_score_warning: float = 0.60
    fraud_score_high: float = 0.75
    fraud_score_critical: float = 0.90

    anomaly_rate_warning_pct: float = 2.0
    anomaly_rate_high_pct: float = 5.0
    anomaly_rate_critical_pct: float = 10.0

    # Revenue
    revenue_decline_warning_pct: float = 5.0
    revenue_decline_high_pct: float = 10.0
    revenue_decline_critical_pct: float = 20.0

    # Margin
    margin_decline_warning_pct: float = 2.0
    margin_decline_high_pct: float = 5.0
    margin_decline_critical_pct: float = 10.0

    # Budget
    budget_variance_warning_pct: float = 5.0
    budget_variance_high_pct: float = 10.0
    budget_variance_critical_pct: float = 20.0

    # Forecast
    forecast_error_warning_pct: float = 10.0
    forecast_error_high_pct: float = 20.0
    forecast_error_critical_pct: float = 30.0

    # Data quality
    missing_data_warning_pct: float = 5.0
    missing_data_critical_pct: float = 15.0

    # Engine
    minimum_confidence: float = 0.55
    max_recommendations: int = 25

    # Human review
    require_human_review_for_high_risk: bool = True
    require_human_review_for_high_exposure: bool = True

    high_exposure_amount: float = 50_000.0

    # Output
    output_directory: str = (
        "reports/recommendations"
    )


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class RiskSignal:
    """
    Normalized risk signal.

    Can represent:
        - liquidity warning
        - fraud score
        - anomaly
        - concentration
        - leverage
        - credit risk
        - forecast deviation
        - control issue
    """

    signal_id: str

    risk_type: str

    title: str

    description: str

    severity: str

    score: float

    confidence: float

    exposure: float = 0.0

    current_value: Optional[float] = None

    threshold: Optional[float] = None

    variance: Optional[float] = None

    variance_pct: Optional[float] = None

    affected_entity: Optional[str] = None

    affected_metric: Optional[str] = None

    evidence: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class RiskRecommendation:
    """Single explainable risk recommendation."""

    recommendation_id: str

    risk_type: str

    title: str

    description: str

    action: str

    priority: str

    risk: str

    confidence: float

    impact_score: float

    exposure: float = 0.0

    estimated_loss_avoidance: float = 0.0

    current_value: Optional[float] = None

    threshold: Optional[float] = None

    variance: Optional[float] = None

    variance_pct: Optional[float] = None

    currency: str = DEFAULT_CURRENCY

    affected_entity: Optional[str] = None

    affected_metric: Optional[str] = None

    evidence: List[str] = field(
        default_factory=list
    )

    assumptions: List[str] = field(
        default_factory=list
    )

    risks: List[str] = field(
        default_factory=list
    )

    controls: List[str] = field(
        default_factory=list
    )

    next_steps: List[str] = field(
        default_factory=list
    )

    requires_human_review: bool = False

    generated_at: str = field(
        default_factory=_timestamp
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dictionary."""

        return _json_safe(
            asdict(self)
        )


@dataclass
class RiskRecommendationResult:
    """Result containing risk recommendations."""

    result_id: str

    generated_at: str

    currency: str

    status: str

    recommendations: List[
        RiskRecommendation
    ]

    recommendations_count: int

    critical_count: int

    high_count: int

    medium_count: int

    human_review_count: int

    total_exposure: float

    estimated_loss_avoidance: float

    average_confidence: float

    average_risk_score: float

    risk_types_detected: List[str]

    summary: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""

        return _json_safe(
            asdict(self)
        )

    def to_json(
        self,
        indent: int = 2,
    ) -> str:
        """Serialize result to JSON."""

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
        )


# ============================================================================
# Risk Recommendation Engine
# ============================================================================


class RiskRecommendationEngine:
    """
    Explainable financial risk recommendation engine.

    Responsibilities:
        1. Detect risk conditions.
        2. Classify severity.
        3. Estimate exposure.
        4. Generate mitigation recommendations.
        5. Assign confidence.
        6. Route high-risk items for human review.

    It does not execute mitigation actions.
    """

    def __init__(
        self,
        config: Optional[
            RiskRecommendationConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or RiskRecommendationConfig()
        )

    # ==========================================================================
    # Public API
    # ==========================================================================

    def generate(
        self,
        financial_data: Optional[Any] = None,
        *,
        risk_data: Optional[Any] = None,
        fraud_data: Optional[Any] = None,
        forecast_data: Optional[Any] = None,
        transaction_data: Optional[Any] = None,
        control_data: Optional[Any] = None,
    ) -> RiskRecommendationResult:
        """
        Generate risk recommendations.

        Inputs may be:
            - dictionaries
            - dataclasses
            - Pydantic models
            - ORM objects
            - lists of records
        """

        context = self._build_context(
            financial_data=financial_data,
            risk_data=risk_data,
            fraud_data=fraud_data,
            forecast_data=forecast_data,
            transaction_data=transaction_data,
            control_data=control_data,
        )

        signals: List[RiskSignal] = []

        signals.extend(
            self._detect_liquidity_risks(
                context
            )
        )

        signals.extend(
            self._detect_cash_flow_risks(
                context
            )
        )

        signals.extend(
            self._detect_receivables_risks(
                context
            )
        )

        signals.extend(
            self._detect_leverage_risks(
                context
            )
        )

        signals.extend(
            self._detect_concentration_risks(
                context
            )
        )

        signals.extend(
            self._detect_fraud_risks(
                context
            )
        )

        signals.extend(
            self._detect_anomaly_risks(
                context
            )
        )

        signals.extend(
            self._detect_revenue_risks(
                context
            )
        )

        signals.extend(
            self._detect_margin_risks(
                context
            )
        )

        signals.extend(
            self._detect_budget_risks(
                context
            )
        )

        signals.extend(
            self._detect_forecast_risks(
                context
            )
        )

        signals.extend(
            self._detect_data_quality_risks(
                context
            )
        )

        signals.extend(
            self._detect_control_risks(
                context
            )
        )

        recommendations = [
            self._signal_to_recommendation(
                signal
            )
            for signal in signals
        ]

        recommendations = [
            recommendation
            for recommendation in recommendations
            if recommendation.confidence
            >= self.config.minimum_confidence
        ]

        recommendations = self._deduplicate(
            recommendations
        )

        recommendations = self._sort(
            recommendations
        )

        recommendations = recommendations[
            : self.config.max_recommendations
        ]

        return self._build_result(
            recommendations,
            signals,
        )

    # ==========================================================================
    # Context
    # ==========================================================================

    def _build_context(
        self,
        financial_data: Any,
        risk_data: Any,
        fraud_data: Any,
        forecast_data: Any,
        transaction_data: Any,
        control_data: Any,
    ) -> Dict[str, Any]:

        context: Dict[str, Any] = {}

        if isinstance(
            financial_data,
            Mapping,
        ):

            context.update(
                financial_data
            )

        elif financial_data is not None:

            context.update(
                _json_safe(
                    financial_data
                )
            )

        context["_risk_data"] = risk_data
        context["_fraud_data"] = fraud_data
        context["_forecast_data"] = forecast_data
        context["_transaction_data"] = (
            transaction_data
        )
        context["_control_data"] = control_data

        return context

    # ==========================================================================
    # Liquidity Risk
    # ==========================================================================

    def _detect_liquidity_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        current_ratio = _safe_float(
            _first(
                data,
                ["current_ratio"],
            ),
            default=-1,
        )

        if (
            current_ratio >= 0
            and current_ratio
            < self.config.current_ratio_critical
        ):

            signals.append(
                self._create_signal(
                    risk_type="liquidity",
                    title="Critical current-ratio risk",
                    description=(
                        "Short-term current assets are insufficient "
                        "relative to current liabilities."
                    ),
                    severity="critical",
                    score=0.95,
                    confidence=0.97,
                    current_value=current_ratio,
                    threshold=(
                        self.config.current_ratio_critical
                    ),
                    variance=(
                        current_ratio
                        - self.config.current_ratio_critical
                    ),
                    affected_metric="current_ratio",
                    evidence=[
                        f"Current ratio: "
                        f"{current_ratio:.2f}",
                        f"Critical threshold: "
                        f"{self.config.current_ratio_critical:.2f}",
                    ],
                )
            )

        elif (
            current_ratio >= 0
            and current_ratio
            < self.config.current_ratio_warning
        ):

            signals.append(
                self._create_signal(
                    risk_type="liquidity",
                    title="Current-ratio warning",
                    description=(
                        "Current liquidity is below the configured "
                        "warning threshold."
                    ),
                    severity="high",
                    score=0.76,
                    confidence=0.93,
                    current_value=current_ratio,
                    threshold=(
                        self.config.current_ratio_warning
                    ),
                    variance=(
                        current_ratio
                        - self.config.current_ratio_warning
                    ),
                    affected_metric="current_ratio",
                    evidence=[
                        f"Current ratio: "
                        f"{current_ratio:.2f}",
                    ],
                )
            )

        quick_ratio = _safe_float(
            _first(
                data,
                ["quick_ratio"],
            ),
            default=-1,
        )

        if (
            quick_ratio >= 0
            and quick_ratio
            < self.config.quick_ratio_critical
        ):

            signals.append(
                self._create_signal(
                    risk_type="liquidity",
                    title="Critical quick-ratio risk",
                    description=(
                        "Highly liquid assets may not be sufficient "
                        "to cover short-term obligations."
                    ),
                    severity="critical",
                    score=0.94,
                    confidence=0.96,
                    current_value=quick_ratio,
                    threshold=(
                        self.config.quick_ratio_critical
                    ),
                    variance=(
                        quick_ratio
                        - self.config.quick_ratio_critical
                    ),
                    affected_metric="quick_ratio",
                    evidence=[
                        f"Quick ratio: "
                        f"{quick_ratio:.2f}",
                    ],
                )
            )

        elif (
            quick_ratio >= 0
            and quick_ratio
            < self.config.quick_ratio_warning
        ):

            signals.append(
                self._create_signal(
                    risk_type="liquidity",
                    title="Weak immediate liquidity",
                    description=(
                        "Quick assets are below the preferred "
                        "liquidity threshold."
                    ),
                    severity="high",
                    score=0.72,
                    confidence=0.92,
                    current_value=quick_ratio,
                    threshold=(
                        self.config.quick_ratio_warning
                    ),
                    affected_metric="quick_ratio",
                    evidence=[
                        f"Quick ratio: "
                        f"{quick_ratio:.2f}",
                    ],
                )
            )

        cash_runway = _safe_float(
            _first(
                data,
                [
                    "cash_runway_months",
                    "cash_runway",
                ],
            ),
            default=-1,
        )

        if (
            cash_runway >= 0
            and cash_runway
            < self.config.cash_runway_critical_months
        ):

            signals.append(
                self._create_signal(
                    risk_type="liquidity",
                    title="Critical cash runway risk",
                    description=(
                        "Available cash may support operations for "
                        "less than the configured critical runway."
                    ),
                    severity="critical",
                    score=0.97,
                    confidence=0.96,
                    current_value=cash_runway,
                    threshold=(
                        self.config.cash_runway_critical_months
                    ),
                    affected_metric="cash_runway_months",
                    evidence=[
                        f"Cash runway: "
                        f"{cash_runway:.1f} months",
                    ],
                )
            )

        elif (
            cash_runway >= 0
            and cash_runway
            < self.config.cash_runway_warning_months
        ):

            signals.append(
                self._create_signal(
                    risk_type="liquidity",
                    title="Short cash runway",
                    description=(
                        "Cash runway is below the preferred "
                        "financial safety buffer."
                    ),
                    severity="high",
                    score=0.78,
                    confidence=0.91,
                    current_value=cash_runway,
                    threshold=(
                        self.config.cash_runway_warning_months
                    ),
                    affected_metric="cash_runway_months",
                    evidence=[
                        f"Cash runway: "
                        f"{cash_runway:.1f} months",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Cash Flow Risk
    # ==========================================================================

    def _detect_cash_flow_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        operating_cash_flow = _safe_float(
            _first(
                data,
                [
                    "operating_cash_flow",
                    "cash_from_operations",
                ],
            )
        )

        net_income = _safe_float(
            _first(
                data,
                [
                    "net_income",
                    "net_profit",
                    "profit",
                ],
            )
        )

        if (
            operating_cash_flow < 0
            and net_income > 0
        ):

            signals.append(
                self._create_signal(
                    risk_type="cash_flow",
                    title="Profit-to-cash conversion risk",
                    description=(
                        "Accounting profit is positive while operating "
                        "cash flow is negative."
                    ),
                    severity="critical",
                    score=0.91,
                    confidence=0.95,
                    exposure=abs(
                        operating_cash_flow
                    ),
                    current_value=operating_cash_flow,
                    threshold=0.0,
                    variance=operating_cash_flow,
                    affected_metric="operating_cash_flow",
                    evidence=[
                        f"Operating cash flow: "
                        f"{_format_currency(operating_cash_flow, self.config.currency)}",
                        f"Net income: "
                        f"{_format_currency(net_income, self.config.currency)}",
                    ],
                )
            )

        elif operating_cash_flow < 0:

            signals.append(
                self._create_signal(
                    risk_type="cash_flow",
                    title="Negative operating cash flow",
                    description=(
                        "Core business operations are consuming "
                        "rather than generating cash."
                    ),
                    severity="high",
                    score=0.82,
                    confidence=0.94,
                    exposure=abs(
                        operating_cash_flow
                    ),
                    current_value=operating_cash_flow,
                    threshold=0.0,
                    variance=operating_cash_flow,
                    affected_metric="operating_cash_flow",
                    evidence=[
                        f"Operating cash flow: "
                        f"{_format_currency(operating_cash_flow, self.config.currency)}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Receivables / Credit Risk
    # ==========================================================================

    def _detect_receivables_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        dso = _safe_float(
            _first(
                data,
                [
                    "dso",
                    "days_sales_outstanding",
                ],
            ),
            default=-1,
        )

        if (
            dso >= 0
            and dso
            > self.config.dso_critical_days
        ):

            signals.append(
                self._create_signal(
                    risk_type="receivables",
                    title="Critical receivables collection risk",
                    description=(
                        "Customer collection time is materially above "
                        "the configured critical threshold."
                    ),
                    severity="critical",
                    score=0.92,
                    confidence=0.95,
                    current_value=dso,
                    threshold=(
                        self.config.dso_critical_days
                    ),
                    variance=(
                        dso
                        - self.config.dso_critical_days
                    ),
                    affected_metric="dso",
                    evidence=[
                        f"DSO: {dso:.1f} days",
                        f"Critical threshold: "
                        f"{self.config.dso_critical_days:.1f} days",
                    ],
                )
            )

        elif (
            dso >= 0
            and dso
            > self.config.dso_high_days
        ):

            signals.append(
                self._create_signal(
                    risk_type="receivables",
                    title="Elevated receivables risk",
                    description=(
                        "Customer collections are taking longer "
                        "than the preferred threshold."
                    ),
                    severity="high",
                    score=0.77,
                    confidence=0.92,
                    current_value=dso,
                    threshold=(
                        self.config.dso_high_days
                    ),
                    variance=(
                        dso
                        - self.config.dso_high_days
                    ),
                    affected_metric="dso",
                    evidence=[
                        f"DSO: {dso:.1f} days",
                    ],
                )
            )

        overdue_pct = _safe_float(
            _first(
                data,
                [
                    "overdue_receivables_pct",
                    "overdue_ar_pct",
                ],
            ),
            default=-1,
        )

        receivables = _safe_float(
            _first(
                data,
                [
                    "accounts_receivable",
                    "receivables",
                    "total_receivables",
                ],
            )
        )

        if (
            overdue_pct >= 0
            and overdue_pct
            > self.config.overdue_receivables_critical_pct
        ):

            exposure = (
                receivables
                * overdue_pct
                / 100
            )

            signals.append(
                self._create_signal(
                    risk_type="credit",
                    title="Critical overdue receivables exposure",
                    description=(
                        "A significant share of receivables is overdue, "
                        "creating elevated collection and credit risk."
                    ),
                    severity="critical",
                    score=0.93,
                    confidence=0.95,
                    exposure=exposure,
                    current_value=overdue_pct,
                    threshold=(
                        self.config.overdue_receivables_critical_pct
                    ),
                    variance=(
                        overdue_pct
                        - self.config.overdue_receivables_critical_pct
                    ),
                    affected_metric="overdue_receivables_pct",
                    evidence=[
                        f"Overdue receivables: "
                        f"{_format_percent(overdue_pct)}",
                        f"Estimated exposed receivables: "
                        f"{_format_currency(exposure, self.config.currency)}",
                    ],
                )
            )

        elif (
            overdue_pct >= 0
            and overdue_pct
            > self.config.overdue_receivables_warning_pct
        ):

            exposure = (
                receivables
                * overdue_pct
                / 100
            )

            signals.append(
                self._create_signal(
                    risk_type="credit",
                    title="Elevated overdue receivables",
                    description=(
                        "Overdue receivables exceed the preferred "
                        "warning threshold."
                    ),
                    severity="high",
                    score=0.75,
                    confidence=0.91,
                    exposure=exposure,
                    current_value=overdue_pct,
                    threshold=(
                        self.config.overdue_receivables_warning_pct
                    ),
                    affected_metric="overdue_receivables_pct",
                    evidence=[
                        f"Overdue receivables: "
                        f"{_format_percent(overdue_pct)}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Leverage Risk
    # ==========================================================================

    def _detect_leverage_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        debt_to_equity = _safe_float(
            _first(
                data,
                [
                    "debt_to_equity",
                    "debt_equity_ratio",
                ],
            ),
            default=-1,
        )

        if (
            debt_to_equity >= 0
            and debt_to_equity
            > self.config.debt_to_equity_critical
        ):

            signals.append(
                self._create_signal(
                    risk_type="leverage",
                    title="Critical leverage risk",
                    description=(
                        "Debt relative to equity exceeds the configured "
                        "critical threshold."
                    ),
                    severity="critical",
                    score=0.95,
                    confidence=0.96,
                    current_value=debt_to_equity,
                    threshold=(
                        self.config.debt_to_equity_critical
                    ),
                    variance=(
                        debt_to_equity
                        - self.config.debt_to_equity_critical
                    ),
                    affected_metric="debt_to_equity",
                    evidence=[
                        f"Debt-to-equity: "
                        f"{debt_to_equity:.2f}",
                    ],
                )
            )

        elif (
            debt_to_equity >= 0
            and debt_to_equity
            > self.config.debt_to_equity_warning
        ):

            signals.append(
                self._create_signal(
                    risk_type="leverage",
                    title="Elevated leverage risk",
                    description=(
                        "Leverage exceeds the configured warning threshold."
                    ),
                    severity="high",
                    score=0.78,
                    confidence=0.93,
                    current_value=debt_to_equity,
                    threshold=(
                        self.config.debt_to_equity_warning
                    ),
                    variance=(
                        debt_to_equity
                        - self.config.debt_to_equity_warning
                    ),
                    affected_metric="debt_to_equity",
                    evidence=[
                        f"Debt-to-equity: "
                        f"{debt_to_equity:.2f}",
                    ],
                )
            )

        interest_coverage = _safe_float(
            _first(
                data,
                [
                    "interest_coverage",
                    "interest_coverage_ratio",
                ],
            ),
            default=-1,
        )

        if (
            interest_coverage >= 0
            and interest_coverage
            < self.config.interest_coverage_critical
        ):

            signals.append(
                self._create_signal(
                    risk_type="leverage",
                    title="Critical debt-service coverage risk",
                    description=(
                        "Operating earnings may be insufficient "
                        "to cover interest obligations."
                    ),
                    severity="critical",
                    score=0.96,
                    confidence=0.96,
                    current_value=interest_coverage,
                    threshold=(
                        self.config.interest_coverage_critical
                    ),
                    variance=(
                        interest_coverage
                        - self.config.interest_coverage_critical
                    ),
                    affected_metric="interest_coverage",
                    evidence=[
                        f"Interest coverage: "
                        f"{interest_coverage:.2f}x",
                    ],
                )
            )

        elif (
            interest_coverage >= 0
            and interest_coverage
            < self.config.interest_coverage_warning
        ):

            signals.append(
                self._create_signal(
                    risk_type="leverage",
                    title="Weak interest coverage",
                    description=(
                        "Debt-service coverage is below the "
                        "preferred threshold."
                    ),
                    severity="high",
                    score=0.81,
                    confidence=0.93,
                    current_value=interest_coverage,
                    threshold=(
                        self.config.interest_coverage_warning
                    ),
                    variance=(
                        interest_coverage
                        - self.config.interest_coverage_warning
                    ),
                    affected_metric="interest_coverage",
                    evidence=[
                        f"Interest coverage: "
                        f"{interest_coverage:.2f}x",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Concentration Risk
    # ==========================================================================

    def _detect_concentration_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        customer_concentration = _safe_float(
            _first(
                data,
                [
                    "top_customer_concentration_pct",
                    "customer_concentration_pct",
                ],
            ),
            default=-1,
        )

        if (
            customer_concentration >= 0
            and customer_concentration
            > self.config.concentration_critical_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="concentration",
                    title="Critical customer concentration risk",
                    description=(
                        "A large portion of revenue depends on "
                        "a small number of customers."
                    ),
                    severity="critical",
                    score=0.91,
                    confidence=0.94,
                    current_value=customer_concentration,
                    threshold=(
                        self.config.concentration_critical_pct
                    ),
                    variance=(
                        customer_concentration
                        - self.config.concentration_critical_pct
                    ),
                    affected_metric=(
                        "customer_concentration_pct"
                    ),
                    evidence=[
                        f"Top customer concentration: "
                        f"{_format_percent(customer_concentration)}",
                    ],
                )
            )

        elif (
            customer_concentration >= 0
            and customer_concentration
            > self.config.concentration_warning_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="concentration",
                    title="Elevated customer concentration",
                    description=(
                        "Revenue dependency on a limited number "
                        "of customers is above the warning threshold."
                    ),
                    severity="high",
                    score=0.76,
                    confidence=0.91,
                    current_value=customer_concentration,
                    threshold=(
                        self.config.concentration_warning_pct
                    ),
                    affected_metric=(
                        "customer_concentration_pct"
                    ),
                    evidence=[
                        f"Customer concentration: "
                        f"{_format_percent(customer_concentration)}",
                    ],
                )
            )

        supplier_concentration = _safe_float(
            _first(
                data,
                [
                    "top_supplier_concentration_pct",
                    "supplier_concentration_pct",
                ],
            ),
            default=-1,
        )

        if (
            supplier_concentration >= 0
            and supplier_concentration
            > self.config.concentration_critical_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="concentration",
                    title="Critical supplier concentration risk",
                    description=(
                        "A significant portion of procurement depends "
                        "on a concentrated supplier base."
                    ),
                    severity="critical",
                    score=0.89,
                    confidence=0.92,
                    current_value=supplier_concentration,
                    threshold=(
                        self.config.concentration_critical_pct
                    ),
                    affected_metric=(
                        "supplier_concentration_pct"
                    ),
                    evidence=[
                        f"Supplier concentration: "
                        f"{_format_percent(supplier_concentration)}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Fraud Risk
    # ==========================================================================

    def _detect_fraud_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        fraud_score = _safe_float(
            _first(
                data,
                [
                    "fraud_score",
                    "fraud_probability",
                    "risk_score",
                ],
            ),
            default=-1,
        )

        if fraud_score > 1:
            fraud_score /= 100.0

        if (
            fraud_score >= 0
            and fraud_score
            >= self.config.fraud_score_critical
        ):

            signals.append(
                self._create_signal(
                    risk_type="fraud",
                    title="Critical fraud-risk signal",
                    description=(
                        "The fraud detection model reports a "
                        "very high-risk transaction or entity."
                    ),
                    severity="critical",
                    score=fraud_score,
                    confidence=0.94,
                    exposure=_safe_float(
                        _first(
                            data,
                            [
                                "fraud_exposure",
                                "transaction_amount",
                                "amount",
                            ],
                        )
                    ),
                    current_value=fraud_score,
                    threshold=(
                        self.config.fraud_score_critical
                    ),
                    affected_metric="fraud_score",
                    evidence=[
                        f"Fraud score: "
                        f"{fraud_score:.2%}",
                    ],
                )
            )

        elif (
            fraud_score >= 0
            and fraud_score
            >= self.config.fraud_score_high
        ):

            signals.append(
                self._create_signal(
                    risk_type="fraud",
                    title="High fraud-risk signal",
                    description=(
                        "The fraud model has identified a high-risk "
                        "transaction or financial event."
                    ),
                    severity="high",
                    score=fraud_score,
                    confidence=0.91,
                    exposure=_safe_float(
                        _first(
                            data,
                            [
                                "fraud_exposure",
                                "transaction_amount",
                                "amount",
                            ],
                        )
                    ),
                    current_value=fraud_score,
                    threshold=(
                        self.config.fraud_score_high
                    ),
                    affected_metric="fraud_score",
                    evidence=[
                        f"Fraud score: "
                        f"{fraud_score:.2%}",
                    ],
                )
            )

        elif (
            fraud_score >= 0
            and fraud_score
            >= self.config.fraud_score_warning
        ):

            signals.append(
                self._create_signal(
                    risk_type="fraud",
                    title="Fraud-risk review required",
                    description=(
                        "A transaction or entity has exceeded the "
                        "configured fraud-risk review threshold."
                    ),
                    severity="medium",
                    score=fraud_score,
                    confidence=0.87,
                    exposure=_safe_float(
                        _first(
                            data,
                            [
                                "fraud_exposure",
                                "transaction_amount",
                                "amount",
                            ],
                        )
                    ),
                    current_value=fraud_score,
                    threshold=(
                        self.config.fraud_score_warning
                    ),
                    affected_metric="fraud_score",
                    evidence=[
                        f"Fraud score: "
                        f"{fraud_score:.2%}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Anomaly Risk
    # ==========================================================================

    def _detect_anomaly_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        anomaly_rate = _safe_float(
            _first(
                data,
                [
                    "anomaly_rate_pct",
                    "transaction_anomaly_rate_pct",
                ],
            ),
            default=-1,
        )

        if (
            anomaly_rate >= 0
            and anomaly_rate
            >= self.config.anomaly_rate_critical_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="anomaly",
                    title="Critical transaction anomaly rate",
                    description=(
                        "The proportion of anomalous transactions "
                        "is materially elevated."
                    ),
                    severity="critical",
                    score=0.93,
                    confidence=0.94,
                    current_value=anomaly_rate,
                    threshold=(
                        self.config.anomaly_rate_critical_pct
                    ),
                    variance=(
                        anomaly_rate
                        - self.config.anomaly_rate_critical_pct
                    ),
                    affected_metric="anomaly_rate_pct",
                    evidence=[
                        f"Anomaly rate: "
                        f"{_format_percent(anomaly_rate)}",
                    ],
                )
            )

        elif (
            anomaly_rate >= 0
            and anomaly_rate
            >= self.config.anomaly_rate_high_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="anomaly",
                    title="Elevated transaction anomalies",
                    description=(
                        "Transaction anomalies exceed the "
                        "configured high-risk threshold."
                    ),
                    severity="high",
                    score=0.80,
                    confidence=0.91,
                    current_value=anomaly_rate,
                    threshold=(
                        self.config.anomaly_rate_high_pct
                    ),
                    affected_metric="anomaly_rate_pct",
                    evidence=[
                        f"Anomaly rate: "
                        f"{_format_percent(anomaly_rate)}",
                    ],
                )
            )

        elif (
            anomaly_rate >= 0
            and anomaly_rate
            >= self.config.anomaly_rate_warning_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="anomaly",
                    title="Transaction anomaly warning",
                    description=(
                        "Transaction anomalies are above "
                        "the configured monitoring threshold."
                    ),
                    severity="medium",
                    score=0.65,
                    confidence=0.86,
                    current_value=anomaly_rate,
                    threshold=(
                        self.config.anomaly_rate_warning_pct
                    ),
                    affected_metric="anomaly_rate_pct",
                    evidence=[
                        f"Anomaly rate: "
                        f"{_format_percent(anomaly_rate)}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Revenue Risk
    # ==========================================================================

    def _detect_revenue_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        revenue = _safe_float(
            _first(
                data,
                [
                    "revenue",
                    "total_revenue",
                    "sales",
                ],
            )
        )

        previous_revenue = _safe_float(
            _first(
                data,
                [
                    "previous_revenue",
                    "prior_revenue",
                ],
            )
        )

        if (
            revenue <= 0
            or previous_revenue <= 0
        ):

            return signals

        change_pct = _percentage(
            revenue - previous_revenue,
            previous_revenue,
        )

        decline_pct = abs(change_pct)

        if (
            change_pct
            < -self.config.revenue_decline_critical_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="revenue",
                    title="Critical revenue decline",
                    description=(
                        "Revenue has declined materially relative "
                        "to the comparison period."
                    ),
                    severity="critical",
                    score=0.92,
                    confidence=0.95,
                    exposure=(
                        previous_revenue
                        - revenue
                    ),
                    current_value=revenue,
                    threshold=previous_revenue,
                    variance=(
                        revenue
                        - previous_revenue
                    ),
                    variance_pct=change_pct,
                    affected_metric="revenue",
                    evidence=[
                        f"Revenue decline: "
                        f"{_format_percent(decline_pct)}",
                    ],
                )
            )

        elif (
            change_pct
            < -self.config.revenue_decline_high_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="revenue",
                    title="High revenue decline risk",
                    description=(
                        "Revenue is declining at a rate that "
                        "requires management attention."
                    ),
                    severity="high",
                    score=0.78,
                    confidence=0.92,
                    exposure=(
                        previous_revenue
                        - revenue
                    ),
                    current_value=revenue,
                    threshold=previous_revenue,
                    variance=(
                        revenue
                        - previous_revenue
                    ),
                    variance_pct=change_pct,
                    affected_metric="revenue",
                    evidence=[
                        f"Revenue decline: "
                        f"{_format_percent(decline_pct)}",
                    ],
                )
            )

        elif (
            change_pct
            < -self.config.revenue_decline_warning_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="revenue",
                    title="Revenue trend warning",
                    description=(
                        "Revenue has started to weaken relative "
                        "to the comparison period."
                    ),
                    severity="medium",
                    score=0.63,
                    confidence=0.84,
                    exposure=(
                        previous_revenue
                        - revenue
                    ),
                    current_value=revenue,
                    threshold=previous_revenue,
                    variance=(
                        revenue
                        - previous_revenue
                    ),
                    variance_pct=change_pct,
                    affected_metric="revenue",
                    evidence=[
                        f"Revenue decline: "
                        f"{_format_percent(decline_pct)}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Margin Risk
    # ==========================================================================

    def _detect_margin_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        current_margin = _safe_float(
            _first(
                data,
                [
                    "gross_margin",
                    "gross_profit_margin",
                ],
            ),
            default=-1,
        )

        previous_margin = _safe_float(
            _first(
                data,
                [
                    "previous_gross_margin",
                    "prior_gross_margin",
                ],
            ),
            default=-1,
        )

        if current_margin < 0:
            return signals

        if abs(current_margin) <= 1:
            current_margin *= 100

        if previous_margin >= 0:

            if abs(previous_margin) <= 1:
                previous_margin *= 100

            decline = (
                previous_margin
                - current_margin
            )

            if (
                decline
                >= self.config.margin_decline_critical_pct
            ):

                signals.append(
                    self._create_signal(
                        risk_type="margin",
                        title="Critical margin deterioration",
                        description=(
                            "Gross margin has declined materially "
                            "compared with the previous period."
                        ),
                        severity="critical",
                        score=0.91,
                        confidence=0.94,
                        current_value=current_margin,
                        threshold=previous_margin,
                        variance=(
                            current_margin
                            - previous_margin
                        ),
                        variance_pct=-decline,
                        affected_metric="gross_margin",
                        evidence=[
                            f"Current gross margin: "
                            f"{_format_percent(current_margin)}",
                            f"Previous gross margin: "
                            f"{_format_percent(previous_margin)}",
                            f"Margin deterioration: "
                            f"{decline:.2f} percentage points",
                        ],
                    )
                )

            elif (
                decline
                >= self.config.margin_decline_high_pct
            ):

                signals.append(
                    self._create_signal(
                        risk_type="margin",
                        title="High margin deterioration risk",
                        description=(
                            "Gross margin has deteriorated "
                            "significantly."
                        ),
                        severity="high",
                        score=0.78,
                        confidence=0.91,
                        current_value=current_margin,
                        threshold=previous_margin,
                        variance=(
                            current_margin
                            - previous_margin
                        ),
                        variance_pct=-decline,
                        affected_metric="gross_margin",
                        evidence=[
                            f"Margin deterioration: "
                            f"{decline:.2f} percentage points",
                        ],
                    )
                )

            elif (
                decline
                >= self.config.margin_decline_warning_pct
            ):

                signals.append(
                    self._create_signal(
                        risk_type="margin",
                        title="Margin trend warning",
                        description=(
                            "Gross margin is weakening "
                            "compared with the prior period."
                        ),
                        severity="medium",
                        score=0.63,
                        confidence=0.84,
                        current_value=current_margin,
                        threshold=previous_margin,
                        variance=(
                            current_margin
                            - previous_margin
                        ),
                        variance_pct=-decline,
                        affected_metric="gross_margin",
                        evidence=[
                            f"Margin deterioration: "
                            f"{decline:.2f} percentage points",
                        ],
                    )
                )

        return signals

    # ==========================================================================
    # Budget Risk
    # ==========================================================================

    def _detect_budget_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        actual = _safe_float(
            _first(
                data,
                [
                    "actual_expenses",
                    "expenses",
                    "total_expenses",
                ],
            )
        )

        budget = _safe_float(
            _first(
                data,
                [
                    "budget_expenses",
                    "expense_budget",
                    "budget",
                ],
            )
        )

        if actual <= 0 or budget <= 0:
            return signals

        variance = (
            actual - budget
        )

        variance_pct = _percentage(
            variance,
            budget,
        )

        if (
            variance_pct
            > self.config.budget_variance_critical_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="budget",
                    title="Critical budget overrun",
                    description=(
                        "Actual expenses materially exceed "
                        "the approved budget."
                    ),
                    severity="critical",
                    score=0.94,
                    confidence=0.97,
                    exposure=variance,
                    current_value=actual,
                    threshold=budget,
                    variance=variance,
                    variance_pct=variance_pct,
                    affected_metric="budget_variance",
                    evidence=[
                        f"Budget: "
                        f"{_format_currency(budget, self.config.currency)}",
                        f"Actual: "
                        f"{_format_currency(actual, self.config.currency)}",
                        f"Variance: "
                        f"{_format_percent(variance_pct)}",
                    ],
                )
            )

        elif (
            variance_pct
            > self.config.budget_variance_high_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="budget",
                    title="High budget variance",
                    description=(
                        "Expenses are materially above budget."
                    ),
                    severity="high",
                    score=0.79,
                    confidence=0.94,
                    exposure=variance,
                    current_value=actual,
                    threshold=budget,
                    variance=variance,
                    variance_pct=variance_pct,
                    affected_metric="budget_variance",
                    evidence=[
                        f"Budget variance: "
                        f"{_format_percent(variance_pct)}",
                    ],
                )
            )

        elif (
            variance_pct
            > self.config.budget_variance_warning_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="budget",
                    title="Budget variance warning",
                    description=(
                        "Expenses have exceeded the warning "
                        "threshold relative to budget."
                    ),
                    severity="medium",
                    score=0.62,
                    confidence=0.86,
                    exposure=variance,
                    current_value=actual,
                    threshold=budget,
                    variance=variance,
                    variance_pct=variance_pct,
                    affected_metric="budget_variance",
                    evidence=[
                        f"Budget variance: "
                        f"{_format_percent(variance_pct)}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Forecast Risk
    # ==========================================================================

    def _detect_forecast_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        forecast_error = _safe_float(
            _first(
                data,
                [
                    "forecast_error_pct",
                    "mape",
                    "forecast_mape",
                ],
            ),
            default=-1,
        )

        if (
            forecast_error >= 0
            and forecast_error
            >= self.config.forecast_error_critical_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="forecast",
                    title="Critical forecast reliability risk",
                    description=(
                        "Forecast error is high enough that financial "
                        "planning based on the forecast may be unreliable."
                    ),
                    severity="critical",
                    score=0.90,
                    confidence=0.92,
                    current_value=forecast_error,
                    threshold=(
                        self.config.forecast_error_critical_pct
                    ),
                    variance=(
                        forecast_error
                        - self.config.forecast_error_critical_pct
                    ),
                    affected_metric="forecast_error_pct",
                    evidence=[
                        f"Forecast error: "
                        f"{_format_percent(forecast_error)}",
                    ],
                )
            )

        elif (
            forecast_error >= 0
            and forecast_error
            >= self.config.forecast_error_high_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="forecast",
                    title="High forecast uncertainty",
                    description=(
                        "Forecast accuracy is below the preferred "
                        "reliability level."
                    ),
                    severity="high",
                    score=0.76,
                    confidence=0.89,
                    current_value=forecast_error,
                    threshold=(
                        self.config.forecast_error_high_pct
                    ),
                    affected_metric="forecast_error_pct",
                    evidence=[
                        f"Forecast error: "
                        f"{_format_percent(forecast_error)}",
                    ],
                )
            )

        elif (
            forecast_error >= 0
            and forecast_error
            >= self.config.forecast_error_warning_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="forecast",
                    title="Forecast uncertainty warning",
                    description=(
                        "Forecast error has exceeded the monitoring "
                        "threshold."
                    ),
                    severity="medium",
                    score=0.61,
                    confidence=0.82,
                    current_value=forecast_error,
                    threshold=(
                        self.config.forecast_error_warning_pct
                    ),
                    affected_metric="forecast_error_pct",
                    evidence=[
                        f"Forecast error: "
                        f"{_format_percent(forecast_error)}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Data Quality Risk
    # ==========================================================================

    def _detect_data_quality_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        missing_pct = _safe_float(
            _first(
                data,
                [
                    "missing_data_pct",
                    "missing_values_pct",
                ],
            ),
            default=-1,
        )

        if (
            missing_pct >= 0
            and missing_pct
            >= self.config.missing_data_critical_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="data_quality",
                    title="Critical financial data-quality risk",
                    description=(
                        "A material share of required financial "
                        "data is missing."
                    ),
                    severity="critical",
                    score=0.94,
                    confidence=0.96,
                    current_value=missing_pct,
                    threshold=(
                        self.config.missing_data_critical_pct
                    ),
                    variance=(
                        missing_pct
                        - self.config.missing_data_critical_pct
                    ),
                    affected_metric="missing_data_pct",
                    evidence=[
                        f"Missing data: "
                        f"{_format_percent(missing_pct)}",
                    ],
                )
            )

        elif (
            missing_pct >= 0
            and missing_pct
            >= self.config.missing_data_warning_pct
        ):

            signals.append(
                self._create_signal(
                    risk_type="data_quality",
                    title="Financial data-quality warning",
                    description=(
                        "Missing financial data may reduce the "
                        "reliability of analysis and recommendations."
                    ),
                    severity="high",
                    score=0.72,
                    confidence=0.92,
                    current_value=missing_pct,
                    threshold=(
                        self.config.missing_data_warning_pct
                    ),
                    affected_metric="missing_data_pct",
                    evidence=[
                        f"Missing data: "
                        f"{_format_percent(missing_pct)}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Control Risk
    # ==========================================================================

    def _detect_control_risks(
        self,
        data: Mapping[str, Any],
    ) -> List[RiskSignal]:

        signals = []

        control_failures = _safe_int(
            _first(
                data,
                [
                    "control_failures",
                    "failed_controls",
                ],
            )
        )

        if control_failures <= 0:
            return signals

        high_risk_controls = _safe_int(
            _first(
                data,
                [
                    "high_risk_control_failures",
                    "critical_control_failures",
                ]
            )
        )

        if high_risk_controls > 0:

            signals.append(
                self._create_signal(
                    risk_type="compliance",
                    title="High-risk financial control failures",
                    description=(
                        f"{high_risk_controls} high-risk control "
                        "failure(s) require investigation."
                    ),
                    severity="critical",
                    score=0.94,
                    confidence=0.95,
                    affected_metric="control_failures",
                    evidence=[
                        f"Total control failures: "
                        f"{control_failures}",
                        f"High-risk failures: "
                        f"{high_risk_controls}",
                    ],
                )
            )

        else:

            signals.append(
                self._create_signal(
                    risk_type="operational",
                    title="Financial control weakness detected",
                    description=(
                        f"{control_failures} financial control "
                        "failure(s) were detected."
                    ),
                    severity="high",
                    score=0.76,
                    confidence=0.91,
                    affected_metric="control_failures",
                    evidence=[
                        f"Control failures: "
                        f"{control_failures}",
                    ],
                )
            )

        return signals

    # ==========================================================================
    # Signal Creation
    # ==========================================================================

    def _create_signal(
        self,
        *,
        risk_type: str,
        title: str,
        description: str,
        severity: str,
        score: float,
        confidence: float,
        exposure: float = 0.0,
        current_value: Optional[float] = None,
        threshold: Optional[float] = None,
        variance: Optional[float] = None,
        variance_pct: Optional[float] = None,
        affected_entity: Optional[str] = None,
        affected_metric: Optional[str] = None,
        evidence: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RiskSignal:

        signal_id = _hash_id(
            "risk-signal",
            risk_type,
            title,
            affected_entity,
            affected_metric,
        )

        return RiskSignal(
            signal_id=signal_id,
            risk_type=risk_type,
            title=_normalize_text(title),
            description=_normalize_text(
                description
            ),
            severity=_normalize_risk(
                severity
            ),
            score=max(
                0.0,
                min(
                    1.0,
                    _safe_float(score),
                ),
            ),
            confidence=max(
                0.0,
                min(
                    1.0,
                    _safe_float(confidence),
                ),
            ),
            exposure=max(
                0.0,
                _safe_float(exposure),
            ),
            current_value=(
                _round(current_value)
                if current_value is not None
                else None
            ),
            threshold=(
                _round(threshold)
                if threshold is not None
                else None
            ),
            variance=(
                _round(variance)
                if variance is not None
                else None
            ),
            variance_pct=(
                _round(variance_pct)
                if variance_pct is not None
                else None
            ),
            affected_entity=affected_entity,
            affected_metric=affected_metric,
            evidence=evidence or [],
            metadata=metadata or {},
        )

    # ==========================================================================
    # Signal -> Recommendation
    # ==========================================================================

    def _signal_to_recommendation(
        self,
        signal: RiskSignal,
    ) -> RiskRecommendation:

        action_map = {
            "liquidity": (
                "Create a short-term liquidity plan, accelerate "
                "cash collection, prioritize essential payments and "
                "review near-term financing requirements."
            ),
            "cash_flow": (
                "Perform a cash-conversion review covering receivables, "
                "inventory, payables and discretionary cash outflows."
            ),
            "receivables": (
                "Prioritize overdue accounts, improve collection "
                "monitoring and review customer payment terms."
            ),
            "credit": (
                "Review exposed customers, strengthen credit controls "
                "and prioritize collection of high-value overdue balances."
            ),
            "leverage": (
                "Review debt maturity, interest burden, refinancing "
                "options and the organization's deleveraging capacity."
            ),
            "concentration": (
                "Reduce dependency on concentrated customers or suppliers "
                "through diversification and contingency planning."
            ),
            "fraud": (
                "Place the identified event into human investigation, "
                "validate supporting evidence and review transaction "
                "authorization and account history."
            ),
            "anomaly": (
                "Investigate anomalous transactions, compare them with "
                "historical patterns and validate supporting records."
            ),
            "revenue": (
                "Perform revenue root-cause analysis across customers, "
                "products, pricing, churn and sales pipeline."
            ),
            "margin": (
                "Perform margin-bridge analysis across pricing, COGS, "
                "discounting and product mix."
            ),
            "budget": (
                "Perform category-level budget variance analysis and "
                "introduce corrective spending controls."
            ),
            "forecast": (
                "Review forecasting assumptions, model performance and "
                "input-data quality before relying on the forecast."
            ),
            "data_quality": (
                "Validate source systems, reconcile missing fields and "
                "block high-impact decisions that depend on unreliable data."
            ),
            "operational": (
                "Investigate the affected financial process and strengthen "
                "preventive and detective controls."
            ),
            "compliance": (
                "Escalate the control issue, document the exception and "
                "complete an appropriate compliance review."
            ),
        }

        action = action_map.get(
            signal.risk_type,
            (
                "Investigate the risk signal, validate the evidence "
                "and implement an appropriate mitigation plan."
            ),
        )

        priority = self._priority_from_severity(
            signal.severity
        )

        estimated_loss_avoidance = (
            signal.exposure
            * self._loss_avoidance_factor(
                signal.severity
            )
        )

        high_exposure = (
            signal.exposure
            >= self.config.high_exposure_amount
        )

        requires_human_review = (
            (
                self.config.require_human_review_for_high_risk
                and signal.severity
                in {"critical", "high"}
            )
            or (
                self.config.require_human_review_for_high_exposure
                and high_exposure
            )
            or signal.risk_type
            in {
                "fraud",
                "compliance",
            }
        )

        recommendation_id = _hash_id(
            "risk-recommendation",
            signal.risk_type,
            signal.title,
            signal.affected_metric,
            signal.affected_entity,
        )

        controls = self._controls_for_risk(
            signal.risk_type
        )

        next_steps = self._next_steps_for_risk(
            signal.risk_type,
            signal.severity,
        )

        assumptions = [
            "Risk thresholds are configurable policy defaults.",
            "Underlying source data is assumed to be reasonably accurate.",
            "Risk signals should be validated before financial action.",
        ]

        risks = self._risks_for_type(
            signal.risk_type
        )

        return RiskRecommendation(
            recommendation_id=recommendation_id,
            risk_type=signal.risk_type,
            title=self._recommendation_title(
                signal
            ),
            description=signal.description,
            action=action,
            priority=priority,
            risk=signal.severity,
            confidence=_round(
                signal.confidence,
                4,
            ),
            impact_score=self._impact_score(
                signal
            ),
            exposure=_round(
                signal.exposure
            ),
            estimated_loss_avoidance=_round(
                estimated_loss_avoidance
            ),
            current_value=signal.current_value,
            threshold=signal.threshold,
            variance=signal.variance,
            variance_pct=signal.variance_pct,
            currency=self.config.currency,
            affected_entity=signal.affected_entity,
            affected_metric=signal.affected_metric,
            evidence=signal.evidence,
            assumptions=assumptions,
            risks=risks,
            controls=controls,
            next_steps=next_steps,
            requires_human_review=(
                requires_human_review
            ),
            metadata={
                "signal_id": signal.signal_id,
                "risk_score": signal.score,
                "engine": self.__class__.__name__,
                "policy": "advisory_only",
            },
        )

    # ==========================================================================
    # Recommendation Helpers
    # ==========================================================================

    def _recommendation_title(
        self,
        signal: RiskSignal,
    ) -> str:

        prefixes = {
            "critical": "Mitigate Critical Risk:",
            "high": "Address High Risk:",
            "medium": "Monitor Risk:",
            "low": "Review Risk:",
        }

        prefix = prefixes.get(
            signal.severity,
            "Review Risk:",
        )

        return (
            f"{prefix} "
            f"{signal.title}"
        )

    def _priority_from_severity(
        self,
        severity: str,
    ) -> str:

        if severity == "critical":
            return "critical"

        if severity == "high":
            return "high"

        if severity == "medium":
            return "medium"

        return "low"

    def _loss_avoidance_factor(
        self,
        severity: str,
    ) -> float:

        return {
            "critical": 0.75,
            "high": 0.50,
            "medium": 0.25,
            "low": 0.10,
        }.get(
            severity,
            0.20,
        )

    def _impact_score(
        self,
        signal: RiskSignal,
    ) -> float:

        severity_score = (
            RISK_LEVELS.get(
                signal.severity,
                2,
            )
            / 4
        )

        return _round(
            (
                severity_score * 0.45
                + signal.score * 0.30
                + signal.confidence * 0.25
            ),
            4,
        )

    def _controls_for_risk(
        self,
        risk_type: str,
    ) -> List[str]:

        controls = {
            "liquidity": [
                "Daily/weekly cash monitoring",
                "Minimum liquidity threshold",
                "13-week cash-flow forecast",
            ],
            "cash_flow": [
                "Cash-flow reconciliation",
                "Working-capital monitoring",
                "Discretionary spending controls",
            ],
            "receivables": [
                "Receivables aging review",
                "Collection escalation rules",
                "Customer credit limits",
            ],
            "credit": [
                "Credit scoring",
                "Exposure limits",
                "Collections monitoring",
            ],
            "leverage": [
                "Debt covenant monitoring",
                "Interest coverage monitoring",
                "Debt maturity tracking",
            ],
            "concentration": [
                "Customer/supplier concentration monitoring",
                "Diversification thresholds",
                "Contingency supplier/customer planning",
            ],
            "fraud": [
                "Human investigation",
                "Transaction authorization controls",
                "Audit logging",
            ],
            "anomaly": [
                "Anomaly investigation",
                "Transaction monitoring",
                "Exception review",
            ],
            "revenue": [
                "Revenue trend monitoring",
                "Customer churn monitoring",
                "Sales pipeline monitoring",
            ],
            "margin": [
                "Margin bridge analysis",
                "Pricing controls",
                "COGS monitoring",
            ],
            "budget": [
                "Budget variance monitoring",
                "Approval controls",
                "Category spending limits",
            ],
            "forecast": [
                "Forecast accuracy monitoring",
                "Model drift checks",
                "Scenario analysis",
            ],
            "data_quality": [
                "Data validation",
                "Reconciliation checks",
                "Source-system monitoring",
            ],
            "operational": [
                "Control testing",
                "Exception monitoring",
                "Audit trail",
            ],
            "compliance": [
                "Compliance review",
                "Exception documentation",
                "Audit trail",
            ],
        }

        return controls.get(
            risk_type,
            [
                "Risk monitoring",
                "Human review",
                "Audit logging",
            ],
        )

    def _next_steps_for_risk(
        self,
        risk_type: str,
        severity: str,
    ) -> List[str]:

        base = {
            "liquidity": [
                "Review near-term cash inflows.",
                "Review upcoming liabilities.",
                "Build a short-term liquidity scenario.",
            ],
            "cash_flow": [
                "Analyze working-capital movements.",
                "Review operating cash drivers.",
                "Build a rolling cash-flow forecast.",
            ],
            "receivables": [
                "Review receivables aging.",
                "Prioritize high-value overdue accounts.",
                "Review payment terms.",
            ],
            "credit": [
                "Review customer exposure.",
                "Validate credit limits.",
                "Escalate material overdue balances.",
            ],
            "leverage": [
                "Review debt maturity.",
                "Calculate debt-service capacity.",
                "Evaluate refinancing/deleveraging scenarios.",
            ],
            "concentration": [
                "Identify concentrated exposures.",
                "Build diversification scenarios.",
                "Create contingency plans.",
            ],
            "fraud": [
                "Preserve relevant evidence.",
                "Assign human investigator.",
                "Review transaction and authorization history.",
            ],
            "anomaly": [
                "Inspect anomalous transactions.",
                "Compare against historical behavior.",
                "Validate source documentation.",
            ],
            "revenue": [
                "Analyze revenue by customer.",
                "Analyze revenue by product.",
                "Review churn and pipeline.",
            ],
            "margin": [
                "Perform margin bridge analysis.",
                "Review pricing.",
                "Review direct costs.",
            ],
            "budget": [
                "Identify largest budget variances.",
                "Classify controllable costs.",
                "Create corrective spending targets.",
            ],
            "forecast": [
                "Review forecast assumptions.",
                "Evaluate model accuracy.",
                "Run sensitivity scenarios.",
            ],
            "data_quality": [
                "Identify missing fields.",
                "Reconcile source systems.",
                "Validate affected financial metrics.",
            ],
            "operational": [
                "Investigate failed controls.",
                "Identify process root cause.",
                "Implement corrective controls.",
            ],
            "compliance": [
                "Escalate the exception.",
                "Document evidence.",
                "Complete required compliance review.",
            ],
        }

        steps = list(
            base.get(
                risk_type,
                [
                    "Validate the risk signal.",
                    "Investigate root cause.",
                    "Document mitigation.",
                ],
            )
        )

        if severity == "critical":

            steps.insert(
                0,
                "Escalate immediately to the appropriate financial owner.",
            )

        elif severity == "high":

            steps.insert(
                0,
                "Assign an accountable owner and review within the current reporting cycle.",
            )

        return steps

    def _risks_for_type(
        self,
        risk_type: str,
    ) -> List[str]:

        risks = {
            "liquidity": [
                "Liquidity shortages can affect critical obligations.",
                "Emergency financing may increase cost.",
            ],
            "cash_flow": [
                "Persistent negative cash flow can increase funding dependency.",
            ],
            "receivables": [
                "Aggressive collections may affect customer relationships.",
                "Some overdue balances may become uncollectible.",
            ],
            "credit": [
                "Customer defaults may reduce cash inflows.",
                "Credit tightening may affect sales.",
            ],
            "leverage": [
                "Higher leverage increases financial sensitivity.",
                "Refinancing conditions may change.",
            ],
            "concentration": [
                "Loss of a major customer or supplier may disrupt operations.",
            ],
            "fraud": [
                "False positives can consume investigation resources.",
                "Unauthorized activity may create financial and compliance exposure.",
            ],
            "anomaly": [
                "Not every anomaly indicates fraud.",
                "Investigation requires contextual validation.",
            ],
            "revenue": [
                "Recovery actions may increase acquisition costs.",
                "Discounting can reduce margins.",
            ],
            "margin": [
                "Pricing changes may affect demand.",
                "Cost reductions can affect quality.",
            ],
            "budget": [
                "Spending restrictions may affect operations.",
            ],
            "forecast": [
                "Model uncertainty may increase planning error.",
                "Forecast assumptions may become outdated.",
            ],
            "data_quality": [
                "Incomplete data can produce misleading analysis.",
            ],
            "operational": [
                "Control remediation may require process changes.",
            ],
            "compliance": [
                "Unresolved control issues may create regulatory exposure.",
            ],
        }

        return risks.get(
            risk_type,
            [
                "Risk should be validated before action."
            ],
        )

    # ==========================================================================
    # Post Processing
    # ==========================================================================

    def _deduplicate(
        self,
        recommendations: Sequence[
            RiskRecommendation
        ],
    ) -> List[RiskRecommendation]:

        seen = set()
        result = []

        for recommendation in recommendations:

            key = (
                recommendation.risk_type,
                recommendation.affected_metric,
                recommendation.title.lower(),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(
                recommendation
            )

        return result

    def _sort(
        self,
        recommendations: Sequence[
            RiskRecommendation
        ],
    ) -> List[RiskRecommendation]:

        return sorted(
            recommendations,
            key=lambda item: (
                PRIORITY_LEVELS.get(
                    item.priority,
                    0,
                ),
                RISK_LEVELS.get(
                    item.risk,
                    0,
                ),
                item.impact_score,
                item.confidence,
                item.exposure,
            ),
            reverse=True,
        )

    # ==========================================================================
    # Result
    # ==========================================================================

    def _build_result(
        self,
        recommendations: Sequence[
            RiskRecommendation
        ],
        signals: Sequence[RiskSignal],
    ) -> RiskRecommendationResult:

        recommendations = list(
            recommendations
        )

        critical_count = sum(
            1
            for item in recommendations
            if item.risk == "critical"
        )

        high_count = sum(
            1
            for item in recommendations
            if item.risk == "high"
        )

        medium_count = sum(
            1
            for item in recommendations
            if item.risk == "medium"
        )

        human_review_count = sum(
            1
            for item in recommendations
            if item.requires_human_review
        )

        total_exposure = sum(
            item.exposure
            for item in recommendations
        )

        estimated_loss_avoidance = sum(
            item.estimated_loss_avoidance
            for item in recommendations
        )

        average_confidence = _mean(
            item.confidence
            for item in recommendations
        )

        average_risk_score = _mean(
            _safe_float(
                item.metadata.get(
                    "risk_score",
                    0,
                )
            )
            for item in recommendations
        )

        risk_types = sorted(
            {
                item.risk_type
                for item in recommendations
            }
        )

        if critical_count:

            status = "critical"

        elif high_count:

            status = "attention_required"

        elif recommendations:

            status = "monitor"

        else:

            status = "healthy"

        if recommendations:

            summary = (
                f"Detected {len(recommendations)} financial risk "
                f"recommendation(s). "
                f"{critical_count} are critical, "
                f"{high_count} are high risk, and "
                f"{human_review_count} require human review."
            )

        else:

            summary = (
                "No risk conditions exceeded the configured "
                "confidence threshold."
            )

        result_id = _hash_id(
            "risk-result",
            _timestamp(),
            len(signals),
            len(recommendations),
        )

        return RiskRecommendationResult(
            result_id=result_id,
            generated_at=_timestamp(),
            currency=self.config.currency,
            status=status,
            recommendations=recommendations,
            recommendations_count=len(
                recommendations
            ),
            critical_count=critical_count,
            high_count=high_count,
            medium_count=medium_count,
            human_review_count=(
                human_review_count
            ),
            total_exposure=_round(
                total_exposure
            ),
            estimated_loss_avoidance=_round(
                estimated_loss_avoidance
            ),
            average_confidence=_round(
                average_confidence,
                4,
            ),
            average_risk_score=_round(
                average_risk_score,
                4,
            ),
            risk_types_detected=risk_types,
            summary=summary,
            metadata={
                "engine": self.__class__.__name__,
                "version": "1.0.0",
                "policy": "advisory_only",
                "signals_detected": len(
                    signals
                ),
                "human_review_required": (
                    human_review_count > 0
                ),
            },
        )

    # ==========================================================================
    # JSON Export
    # ==========================================================================

    def save_json(
        self,
        result: RiskRecommendationResult,
        path: Optional[str] = None,
    ) -> Path:

        output_path = Path(
            path
            or (
                Path(
                    self.config.output_directory
                )
                / "risk_recommendations.json"
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            result.to_json(),
            encoding="utf-8",
        )

        return output_path

    # ==========================================================================
    # Markdown Export
    # ==========================================================================

    def to_markdown(
        self,
        result: RiskRecommendationResult,
    ) -> str:

        lines = [
            "# FinCo AI - Risk Recommendations",
            "",
            f"**Result ID:** `{result.result_id}`",
            f"**Generated:** `{result.generated_at}`",
            f"**Status:** `{result.status}`",
            "",
            "## Risk Summary",
            "",
            result.summary,
            "",
            "## Risk Metrics",
            "",
            f"- Recommendations: {result.recommendations_count}",
            f"- Critical: {result.critical_count}",
            f"- High: {result.high_count}",
            f"- Medium: {result.medium_count}",
            f"- Human Review: {result.human_review_count}",
            f"- Average Confidence: "
            f"{result.average_confidence:.2%}",
            f"- Average Risk Score: "
            f"{result.average_risk_score:.2%}",
            f"- Total Exposure: "
            f"{_format_currency(result.total_exposure, result.currency)}",
            f"- Estimated Loss Avoidance: "
            f"{_format_currency(result.estimated_loss_avoidance, result.currency)}",
            "",
            "## Detected Risk Types",
            "",
        ]

        if result.risk_types_detected:

            for risk_type in result.risk_types_detected:

                lines.append(
                    f"- `{risk_type}`"
                )

        else:

            lines.append(
                "- None"
            )

        lines.extend(
            [
                "",
                "## Recommendations",
                "",
            ]
        )

        if not result.recommendations:

            lines.append(
                "No risk recommendations generated."
            )

        for index, item in enumerate(
            result.recommendations,
            start=1,
        ):

            lines.extend(
                [
                    f"### {index}. {item.title}",
                    "",
                    f"**Risk Type:** `{item.risk_type}`",
                    "",
                    f"**Priority:** `{item.priority}`",
                    "",
                    f"**Risk:** `{item.risk}`",
                    "",
                    f"**Confidence:** "
                    f"{item.confidence:.2%}",
                    "",
                    f"**Impact Score:** "
                    f"{item.impact_score:.2%}",
                    "",
                    f"**Description:** "
                    f"{item.description}",
                    "",
                    f"**Recommended Action:** "
                    f"{item.action}",
                    "",
                ]
            )

            if item.exposure:

                lines.extend(
                    [
                        f"**Exposure:** "
                        f"{_format_currency(item.exposure, result.currency)}",
                        "",
                    ]
                )

            if item.estimated_loss_avoidance:

                lines.extend(
                    [
                        f"**Estimated Loss Avoidance:** "
                        f"{_format_currency(item.estimated_loss_avoidance, result.currency)}",
                        "",
                    ]
                )

            if item.evidence:

                lines.extend(
                    [
                        "#### Evidence",
                        "",
                    ]
                )

                for evidence in item.evidence:

                    lines.append(
                        f"- {evidence}"
                    )

                lines.append("")

            if item.controls:

                lines.extend(
                    [
                        "#### Controls",
                        "",
                    ]
                )

                for control in item.controls:

                    lines.append(
                        f"- {control}"
                    )

                lines.append("")

            if item.next_steps:

                lines.extend(
                    [
                        "#### Next Steps",
                        "",
                    ]
                )

                for step in item.next_steps:

                    lines.append(
                        f"1. {step}"
                    )

                lines.append("")

            if item.risks:

                lines.extend(
                    [
                        "#### Mitigation Risks",
                        "",
                    ]
                )

                for risk in item.risks:

                    lines.append(
                        f"- {risk}"
                    )

                lines.append("")

            if item.requires_human_review:

                lines.extend(
                    [
                        "> ⚠️ **Human review required before execution.**",
                        "",
                    ]
                )

        lines.extend(
            [
                "---",
                "",
                "*FinCo AI risk recommendations are advisory "
                "and must be validated by authorized financial "
                "or risk personnel before action.*",
            ]
        )

        return "\n".join(lines)

    def save_markdown(
        self,
        result: RiskRecommendationResult,
        path: Optional[str] = None,
    ) -> Path:

        output_path = Path(
            path
            or (
                Path(
                    self.config.output_directory
                )
                / "risk_recommendations.md"
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.to_markdown(result),
            encoding="utf-8",
        )

        return output_path

    # ==========================================================================
    # HTML Export
    # ==========================================================================

    def to_html(
        self,
        result: RiskRecommendationResult,
    ) -> str:

        status_class = {
            "critical": "critical",
            "attention_required": "warning",
            "monitor": "info",
            "healthy": "success",
        }.get(
            result.status,
            "info",
        )

        cards = []

        for item in result.recommendations:

            evidence_html = "".join(
                f"<li>{html.escape(evidence)}</li>"
                for evidence in item.evidence
            )

            controls_html = "".join(
                f"<li>{html.escape(control)}</li>"
                for control in item.controls
            )

            steps_html = "".join(
                f"<li>{html.escape(step)}</li>"
                for step in item.next_steps
            )

            risks_html = "".join(
                f"<li>{html.escape(risk)}</li>"
                for risk in item.risks
            )

            review_html = ""

            if item.requires_human_review:

                review_html = """
                <div class="review">
                    ⚠ Human review required before execution.
                </div>
                """

            cards.append(
                f"""
                <article class="risk-card">

                    <div class="card-top">

                        <div>
                            <span class="badge">
                                {html.escape(item.priority.upper())}
                            </span>

                            <span class="badge">
                                {html.escape(item.risk.upper())}
                            </span>

                            <span class="type">
                                {html.escape(item.risk_type)}
                            </span>
                        </div>

                        <strong>
                            {item.confidence:.0%}
                        </strong>

                    </div>

                    <h2>
                        {html.escape(item.title)}
                    </h2>

                    <p class="description">
                        {html.escape(item.description)}
                    </p>

                    <div class="metrics">

                        <div>
                            <span>Risk Score</span>
                            <strong>
                                {
                                    float(
                                        item.metadata.get(
                                            "risk_score",
                                            0,
                                        )
                                    ):,.0%
                                }
                            </strong>
                        </div>

                        <div>
                            <span>Impact Score</span>
                            <strong>
                                {item.impact_score:.0%}
                            </strong>
                        </div>

                        <div>
                            <span>Exposure</span>
                            <strong>
                                {
                                    _format_currency(
                                        item.exposure,
                                        result.currency,
                                    )
                                }
                            </strong>
                        </div>

                        <div>
                            <span>Loss Avoidance</span>
                            <strong>
                                {
                                    _format_currency(
                                        item.estimated_loss_avoidance,
                                        result.currency,
                                    )
                                }
                            </strong>
                        </div>

                    </div>

                    <h3>Recommended Action</h3>

                    <p>
                        {html.escape(item.action)}
                    </p>

                    <h3>Evidence</h3>

                    <ul>
                        {evidence_html}
                    </ul>

                    <h3>Controls</h3>

                    <ul>
                        {controls_html}
                    </ul>

                    <h3>Next Steps</h3>

                    <ol>
                        {steps_html}
                    </ol>

                    {
                        (
                            f"<h3>Mitigation Risks</h3>"
                            f"<ul>{risks_html}</ul>"
                            if risks_html
                            else ""
                        )
                    }

                    {review_html}

                </article>
                """
            )

        return f"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
    FinCo AI - Risk Recommendations
</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{

    margin: 0;

    padding: 32px;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background: #f4f7fb;

    color: #182230;
}}

.container {{

    max-width: 1220px;

    margin: 0 auto;
}}

.hero {{

    background: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 22px;

    padding: 34px;

    margin-bottom: 24px;
}}

.hero h1 {{

    margin: 0;

    font-size: 32px;
}}

.hero h2 {{

    margin: 8px 0;

    font-size: 20px;
}}

.hero p {{

    color: #64748b;

    line-height: 1.7;
}}

.status {{

    display: inline-block;

    margin-top: 14px;

    padding: 8px 14px;

    border-radius: 999px;

    font-size: 12px;

    font-weight: 800;
}}

.status.critical {{

    background: #fee2e2;

    color: #991b1b;
}}

.status.warning {{

    background: #fef3c7;

    color: #92400e;
}}

.status.info {{

    background: #dbeafe;

    color: #1e40af;
}}

.status.success {{

    background: #dcfce7;

    color: #166534;
}}

.summary {{

    display: grid;

    grid-template-columns:
        repeat(auto-fit, minmax(180px, 1fr));

    gap: 16px;

    margin-bottom: 24px;
}}

.stat {{

    background: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 16px;

    padding: 20px;
}}

.stat span {{

    display: block;

    color: #64748b;

    font-size: 12px;

    margin-bottom: 8px;
}}

.stat strong {{

    font-size: 24px;
}}

.risk-card {{

    background: #ffffff;

    border: 1px solid #e2e8f0;

    border-radius: 20px;

    padding: 26px;

    margin-bottom: 18px;

    box-shadow:
        0 8px 24px rgba(
            15,
            23,
            42,
            0.04
        );
}}

.card-top {{

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 16px;

    flex-wrap: wrap;
}}

.badge {{

    display: inline-block;

    padding: 6px 10px;

    margin-right: 6px;

    border-radius: 999px;

    background: #eef2ff;

    font-size: 10px;

    font-weight: 800;
}}

.type {{

    color: #64748b;

    font-size: 12px;

    font-weight: 600;
}}

.card-top > strong {{

    font-size: 16px;
}}

.risk-card h2 {{

    margin-top: 20px;

    margin-bottom: 8px;
}}

.risk-card h3 {{

    margin-top: 24px;

    font-size: 15px;
}}

.description,
.risk-card p {{

    color: #475569;

    line-height: 1.7;
}}

.metrics {{

    display: grid;

    grid-template-columns:
        repeat(auto-fit, minmax(150px, 1fr));

    gap: 12px;

    margin: 22px 0;
}}

.metrics div {{

    background: #f8fafc;

    border-radius: 12px;

    padding: 15px;
}}

.metrics span {{

    display: block;

    color: #64748b;

    font-size: 11px;

    margin-bottom: 6px;
}}

.metrics strong {{

    font-size: 16px;
}}

li {{

    margin-bottom: 8px;

    color: #475569;
}}

.review {{

    margin-top: 22px;

    padding: 15px;

    border-radius: 12px;

    background: #fff7ed;

    color: #9a3412;

    font-weight: 800;
}}

.footer {{

    padding: 30px;

    text-align: center;

    color: #64748b;

    font-size: 12px;
}}

@media (max-width: 640px) {{

    body {{
        padding: 16px;
    }}

    .hero {{
        padding: 22px;
    }}

    .hero h1 {{
        font-size: 25px;
    }}

    .risk-card {{
        padding: 20px;
    }}

}}

</style>

</head>

<body>

<div class="container">

<section class="hero">

    <h1>
        FinCo AI
    </h1>

    <h2>
        Financial Risk Recommendations
    </h2>

    <p>
        Explainable risk detection and mitigation recommendations
        across liquidity, cash flow, credit, leverage, fraud,
        anomalies, revenue, margins, budgets and financial controls.
    </p>

    <div class="status {status_class}">
        {html.escape(result.status.upper())}
    </div>

</section>

<section class="summary">

    <div class="stat">
        <span>Recommendations</span>
        <strong>
            {result.recommendations_count}
        </strong>
    </div>

    <div class="stat">
        <span>Critical</span>
        <strong>
            {result.critical_count}
        </strong>
    </div>

    <div class="stat">
        <span>High</span>
        <strong>
            {result.high_count}
        </strong>
    </div>

    <div class="stat">
        <span>Human Review</span>
        <strong>
            {result.human_review_count}
        </strong>
    </div>

    <div class="stat">
        <span>Risk Confidence</span>
        <strong>
            {result.average_confidence:.0%}
        </strong>
    </div>

    <div class="stat">
        <span>Total Exposure</span>
        <strong>
            {
                _format_currency(
                    result.total_exposure,
                    result.currency,
                )
            }
        </strong>
    </div>

</section>

<section>

    {"".join(cards)}

</section>

<div class="footer">

    FinCo AI · Risk Intelligence ·
    Advisory recommendations only ·
    Human validation required

</div>

</div>

</body>

</html>
"""

    def save_html(
        self,
        result: RiskRecommendationResult,
        path: Optional[str] = None,
    ) -> Path:

        output_path = Path(
            path
            or (
                Path(
                    self.config.output_directory
                )
                / "risk_recommendations.html"
            )
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.to_html(result),
            encoding="utf-8",
        )

        return output_path

    # ==========================================================================
    # Save All
    # ==========================================================================

    def save_all(
        self,
        result: RiskRecommendationResult,
        output_directory: Optional[str] = None,
    ) -> Dict[str, Path]:

        directory = Path(
            output_directory
            or self.config.output_directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return {
            "json": self.save_json(
                result,
                str(
                    directory
                    / "risk_recommendations.json"
                ),
            ),
            "markdown": self.save_markdown(
                result,
                str(
                    directory
                    / "risk_recommendations.md"
                ),
            ),
            "html": self.save_html(
                result,
                str(
                    directory
                    / "risk_recommendations.html"
                ),
            ),
        }


# ============================================================================
# Convenience API
# ============================================================================


def generate_risk_recommendations(
    financial_data: Optional[Any] = None,
    *,
    risk_data: Optional[Any] = None,
    fraud_data: Optional[Any] = None,
    forecast_data: Optional[Any] = None,
    transaction_data: Optional[Any] = None,
    control_data: Optional[Any] = None,
    config: Optional[
        RiskRecommendationConfig
    ] = None,
) -> RiskRecommendationResult:
    """Convenience API."""

    engine = RiskRecommendationEngine(
        config=config
    )

    return engine.generate(
        financial_data=financial_data,
        risk_data=risk_data,
        fraud_data=fraud_data,
        forecast_data=forecast_data,
        transaction_data=transaction_data,
        control_data=control_data,
    )


# ============================================================================
# Demo Data
# ============================================================================


def create_demo_financial_data() -> Dict[str, Any]:
    """Create realistic risk-heavy demo data."""

    return {

        # Liquidity
        "current_ratio": 0.91,
        "quick_ratio": 0.72,
        "cash_runway_months": 2.8,

        # Cash flow
        "operating_cash_flow": -95_000,
        "net_income": 42_000,

        # Receivables
        "dso": 96,
        "accounts_receivable": 310_000,
        "overdue_receivables_pct": 28.0,

        # Leverage
        "debt_to_equity": 3.25,
        "interest_coverage": 0.92,

        # Concentration
        "top_customer_concentration_pct": 57.0,
        "top_supplier_concentration_pct": 43.0,

        # Fraud
        "fraud_score": 0.93,
        "fraud_exposure": 72_000,
        "transaction_amount": 72_000,

        # Anomaly
        "anomaly_rate_pct": 8.2,

        # Revenue
        "revenue": 1_000_000,
        "previous_revenue": 1_190_000,

        # Margin
        "gross_margin": 0.21,
        "previous_gross_margin": 0.32,

        # Budget
        "actual_expenses": 780_000,
        "budget_expenses": 650_000,

        # Forecast
        "forecast_error_pct": 24.0,

        # Data quality
        "missing_data_pct": 17.0,

        # Controls
        "control_failures": 6,
        "high_risk_control_failures": 2,
    }


def create_demo_result() -> RiskRecommendationResult:
    """Create demo risk recommendation result."""

    engine = RiskRecommendationEngine()

    return engine.generate(
        financial_data=(
            create_demo_financial_data()
        )
    )


# ============================================================================
# CLI
# ============================================================================


def _build_cli() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "FinCo AI risk recommendation engine"
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in demonstration.",
    )

    parser.add_argument(
        "--output",
        default="reports/recommendations",
        help="Output directory.",
    )

    parser.add_argument(
        "--format",
        choices=[
            "json",
            "markdown",
            "html",
            "all",
        ],
        default="all",
        help="Output format.",
    )

    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = _build_cli()

    args = parser.parse_args()

    if not args.demo:

        print(
            "No input data supplied. "
            "Use --demo to run the built-in demo."
        )

        return 0

    config = RiskRecommendationConfig(
        output_directory=args.output
    )

    engine = RiskRecommendationEngine(
        config=config
    )

    result = engine.generate(
        financial_data=(
            create_demo_financial_data()
        )
    )

    print()
    print("=" * 76)
    print(
        "FinCo AI - Financial Risk Recommendations"
    )
    print("=" * 76)
    print()

    print(
        f"Result ID:              "
        f"{result.result_id}"
    )

    print(
        f"Status:                 "
        f"{result.status}"
    )

    print(
        f"Recommendations:        "
        f"{result.recommendations_count}"
    )

    print(
        f"Critical:               "
        f"{result.critical_count}"
    )

    print(
        f"High:                   "
        f"{result.high_count}"
    )

    print(
        f"Medium:                 "
        f"{result.medium_count}"
    )

    print(
        f"Human Review:           "
        f"{result.human_review_count}"
    )

    print(
        f"Average Confidence:     "
        f"{result.average_confidence:.2%}"
    )

    print(
        f"Average Risk Score:     "
        f"{result.average_risk_score:.2%}"
    )

    print(
        f"Total Exposure:         "
        f"{_format_currency(result.total_exposure, result.currency)}"
    )

    print(
        f"Estimated Loss Avoid.:  "
        f"{_format_currency(result.estimated_loss_avoidance, result.currency)}"
    )

    print()

    for index, item in enumerate(
        result.recommendations,
        start=1,
    ):

        print(
            f"{index}. "
            f"[{item.risk.upper()}] "
            f"{item.title}"
        )

        print(
            f"   Type:       "
            f"{item.risk_type}"
        )

        print(
            f"   Confidence: "
            f"{item.confidence:.2%}"
        )

        print(
            f"   Risk Score: "
            f"{float(item.metadata.get('risk_score', 0)):.2%}"
        )

        print(
            f"   Exposure:   "
            f"{_format_currency(item.exposure, result.currency)}"
        )

        print(
            f"   Action:     "
            f"{item.action}"
        )

        if item.requires_human_review:

            print(
                "   Review:     "
                "HUMAN REVIEW REQUIRED"
            )

        print()

    if args.format == "json":

        path = engine.save_json(
            result,
            str(
                Path(args.output)
                / "risk_recommendations.json"
            ),
        )

        print(
            f"JSON saved to: {path}"
        )

    elif args.format == "markdown":

        path = engine.save_markdown(
            result,
            str(
                Path(args.output)
                / "risk_recommendations.md"
            ),
        )

        print(
            f"Markdown saved to: {path}"
        )

    elif args.format == "html":

        path = engine.save_html(
            result,
            str(
                Path(args.output)
                / "risk_recommendations.html"
            ),
        )

        print(
            f"HTML saved to: {path}"
        )

    else:

        paths = engine.save_all(
            result,
            args.output,
        )

        print("Files generated:")

        for file_type, path in paths.items():

            print(
                f"  {file_type:<10} {path}"
            )

    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )