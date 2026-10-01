"""
FinCo AI - Financial Agent
==========================

Specialized financial-analysis agent responsible for:

    Revenue Analysis
    Expense Analysis
    Profitability Analysis
    P&L Analysis
    Balance Sheet Analysis
    Cash Flow Analysis
    Financial Ratios
    KPI Analysis
    Margin Analysis
    Budget Variance
    Working Capital
    Liquidity
    Historical Comparison
    Root Cause Analysis
    Financial Health Assessment
    Evidence-based Financial Recommendations

Architecture
------------

                    Supervisor Agent
                           |
                           v
                  +----------------+
                  | FinancialAgent |
                  +-------+--------+
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
         Financial     Rule Engine   RAG Evidence
          Signals       / Analysis      |
              |             |            |
              +-------------+------------+
                            |
                            v
                    Financial Result
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Recommendation               Human Review
              |
              v
           Audit Log

Important principles
--------------------
- Do not invent financial figures.
- Prefer structured financial data over LLM assumptions.
- Use RAG evidence for document-based claims.
- Clearly distinguish calculated metrics from source values.
- Flag missing/inconsistent data.
- Never execute financial transactions.
- Recommendations are proposals, not approvals.
- High-impact recommendations require human review.
- Preserve evidence and calculation traceability.

This module intentionally avoids coupling itself to a specific
LLM provider. An LLM can be connected by the orchestrator later.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# ============================================================================
# CONSTANTS
# ============================================================================

AGENT_NAME = "financial_agent"
AGENT_VERSION = "1.0.0"

DEFAULT_CURRENCY = "USD"

RISK_LEVELS = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

PRIORITY_LEVELS = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

MAX_EVIDENCE_ITEMS = 10
MAX_RECOMMENDATIONS = 15
MAX_WARNINGS = 20
MAX_FINDINGS = 30


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def utc_now() -> datetime:
    """Return current UTC datetime."""

    return datetime.now(timezone.utc)


def timestamp() -> str:
    """Return current UTC timestamp."""

    return utc_now().isoformat()


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    if value is None:
        return default

    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def optional_float(
    value: Any,
) -> Optional[float]:
    """Convert a value to Optional[float]."""

    if value is None:
        return None

    try:
        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to integer."""

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def safe_divide(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """Safely divide two values."""

    numerator = safe_float(numerator)
    denominator = safe_float(denominator)

    if abs(denominator) < 1e-12:
        return default

    return numerator / denominator


def percentage(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """Return percentage."""

    return (
        safe_divide(
            numerator,
            denominator,
            default,
        )
        * 100.0
    )


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """Clamp a value to a range."""

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def round_value(
    value: Any,
    digits: int = 4,
) -> float:
    """Safely round a numeric value."""

    return round(
        safe_float(value),
        digits,
    )


def normalize_text(
    value: Any,
    max_length: int = 4000,
) -> str:
    """Normalize text."""

    if value is None:
        return ""

    text = str(value).strip()

    if len(text) > max_length:
        return text[: max_length - 3] + "..."

    return text


def normalize_level(
    value: Any,
    default: str = "medium",
) -> str:
    """Normalize risk/priority levels."""

    text = normalize_text(
        value
    ).lower()

    if text in RISK_LEVELS:
        return text

    if text in {
        "warn",
        "warning",
    }:
        return "medium"

    if text in {
        "urgent",
        "severe",
    }:
        return "critical"

    return default


def get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read a value from dict-like or object-like data."""

    if obj is None:
        return default

    if isinstance(
        obj,
        Mapping,
    ):
        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


def first_value(
    obj: Any,
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    """Return the first available field."""

    for key in keys:

        value = get_value(
            obj,
            key,
            None,
        )

        if value is not None:
            return value

    return default


def json_safe(
    value: Any,
) -> Any:
    """Convert values to JSON-safe representations."""

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, bool),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        return (
            value
            if math.isfinite(value)
            else None
        )

    if isinstance(
        value,
        datetime,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(
        value,
        "model_dump",
    ):
        try:
            return json_safe(
                value.model_dump()
            )
        except Exception:
            pass

    if hasattr(
        value,
        "to_dict",
    ):
        try:
            return json_safe(
                value.to_dict()
            )
        except Exception:
            pass

    return normalize_text(
        value
    )


def deduplicate(
    values: Iterable[str],
) -> List[str]:
    """Deduplicate strings while preserving order."""

    seen = set()
    result = []

    for value in values:

        text = normalize_text(
            value
        )

        if not text:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class FinancialFinding:
    """Represents an analytical financial finding."""

    finding_id: str

    category: str

    title: str

    description: str

    metric: Optional[str] = None

    value: Optional[float] = None

    benchmark: Optional[float] = None

    variance: Optional[float] = None

    severity: str = "medium"

    priority: str = "medium"

    confidence: float = 0.80

    evidence: List[str] = field(
        default_factory=list
    )

    rationale: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "finding_id": self.finding_id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "metric": self.metric,
            "value": self.value,
            "benchmark": self.benchmark,
            "variance": self.variance,
            "severity": self.severity,
            "priority": self.priority,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "rationale": self.rationale,
            "metadata": json_safe(
                self.metadata
            ),
        }


@dataclass
class FinancialRecommendation:
    """Represents a proposed financial action."""

    recommendation_id: str

    category: str

    title: str

    description: str

    action: str

    priority: str = "medium"

    risk: str = "low"

    confidence: float = 0.75

    estimated_impact: float = 0.0

    impact_type: str = "financial"

    evidence: List[str] = field(
        default_factory=list
    )

    assumptions: List[str] = field(
        default_factory=list
    )

    risks: List[str] = field(
        default_factory=list
    )

    next_steps: List[str] = field(
        default_factory=list
    )

    human_review_required: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "recommendation_id": (
                self.recommendation_id
            ),
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "action": self.action,
            "priority": self.priority,
            "risk": self.risk,
            "confidence": self.confidence,
            "estimated_impact": (
                self.estimated_impact
            ),
            "impact_type": self.impact_type,
            "evidence": list(self.evidence),
            "assumptions": list(
                self.assumptions
            ),
            "risks": list(self.risks),
            "next_steps": list(
                self.next_steps
            ),
            "human_review_required": (
                self.human_review_required
            ),
            "metadata": json_safe(
                self.metadata
            ),
        }


@dataclass
class FinancialAgentResult:
    """Complete result returned by FinancialAgent."""

    result_id: str

    agent_name: str

    agent_version: str

    generated_at: str

    company_id: Optional[str]

    task: str

    intent: str

    status: str

    summary: str

    financial_metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    findings: List[
        FinancialFinding
    ] = field(
        default_factory=list
    )

    recommendations: List[
        FinancialRecommendation
    ] = field(
        default_factory=list
    )

    calculations: Dict[str, Any] = field(
        default_factory=dict
    )

    evidence: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    assumptions: List[str] = field(
        default_factory=list
    )

    human_review_required: bool = False

    confidence: float = 0.0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:

        return {
            "result_id": self.result_id,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "generated_at": self.generated_at,
            "company_id": self.company_id,
            "task": self.task,
            "intent": self.intent,
            "status": self.status,
            "summary": self.summary,
            "financial_metrics": json_safe(
                self.financial_metrics
            ),
            "findings": [
                finding.to_dict()
                for finding in self.findings
            ],
            "recommendations": [
                recommendation.to_dict()
                for recommendation
                in self.recommendations
            ],
            "calculations": json_safe(
                self.calculations
            ),
            "evidence": json_safe(
                self.evidence
            ),
            "warnings": list(
                self.warnings
            ),
            "assumptions": list(
                self.assumptions
            ),
            "human_review_required": (
                self.human_review_required
            ),
            "confidence": self.confidence,
            "metadata": json_safe(
                self.metadata
            ),
        }

    def to_json(
        self,
        indent: int = 2,
    ) -> str:

        return json.dumps(
            self.to_dict(),
            indent=indent,
            ensure_ascii=False,
            default=str,
        )


@dataclass
class FinancialAgentConfig:
    """Configuration for FinancialAgent."""

    currency: str = DEFAULT_CURRENCY

    min_confidence: float = 0.55

    high_impact_threshold: float = 50_000.0

    critical_margin_threshold: float = 0.05

    low_margin_threshold: float = 0.10

    revenue_decline_warning: float = 0.05

    revenue_decline_critical: float = 0.15

    expense_growth_warning: float = 0.10

    expense_growth_critical: float = 0.20

    current_ratio_warning: float = 1.20

    current_ratio_critical: float = 1.00

    quick_ratio_warning: float = 1.00

    quick_ratio_critical: float = 0.80

    cash_runway_warning_months: float = 6.0

    cash_runway_critical_months: float = 3.0

    dso_warning: float = 60.0

    dso_critical: float = 90.0

    inventory_days_warning: float = 75.0

    inventory_days_critical: float = 120.0

    debt_equity_warning: float = 1.50

    debt_equity_critical: float = 2.50

    interest_coverage_warning: float = 2.0

    interest_coverage_critical: float = 1.0

    require_human_review_high_risk: bool = True

    require_human_review_high_impact: bool = True

    max_findings: int = MAX_FINDINGS

    max_recommendations: int = MAX_RECOMMENDATIONS

    max_evidence: int = MAX_EVIDENCE_ITEMS

    def validate(self) -> None:
        """Validate configuration."""

        if not (
            0.0
            <= self.min_confidence
            <= 1.0
        ):
            raise ValueError(
                "min_confidence must be "
                "between 0 and 1."
            )

        if self.max_findings <= 0:
            raise ValueError(
                "max_findings must be positive."
            )

        if self.max_recommendations <= 0:
            raise ValueError(
                "max_recommendations must be positive."
            )

        if self.max_evidence <= 0:
            raise ValueError(
                "max_evidence must be positive."
            )


# ============================================================================
# FINANCIAL AGENT
# ============================================================================

class FinancialAgent:
    """
    Specialized financial-analysis agent.

    The agent performs deterministic financial calculations and
    produces explainable findings/recommendations.

    It does not execute financial actions.
    """

    def __init__(
        self,
        config: Optional[
            FinancialAgentConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or FinancialAgentConfig()
        )

        self.config.validate()

    # ========================================================================
    # MAIN ENTRYPOINT
    # ========================================================================

    def run(
        self,
        *,
        context: Optional[Any] = None,
        task: Optional[str] = None,
        company_id: Optional[str] = None,
        financial_data: Optional[
            Mapping[str, Any]
        ] = None,
        rag_results: Optional[
            Sequence[Any]
        ] = None,
        risk_data: Optional[
            Mapping[str, Any]
        ] = None,
        forecast_data: Optional[
            Mapping[str, Any]
        ] = None,
        alerts: Optional[
            Sequence[Any]
        ] = None,
    ) -> FinancialAgentResult:
        """
        Execute financial analysis.

        Supports both:

            agent.run(context=context)

        and:

            agent.run(
                task="Analyze profitability",
                company_id="company-001",
                financial_data=data,
            )
        """

        normalized = self._extract_context(
            context=context,
            task=task,
            company_id=company_id,
            financial_data=financial_data,
            rag_results=rag_results,
            risk_data=risk_data,
            forecast_data=forecast_data,
            alerts=alerts,
        )

        task = normalized["task"]
        company_id = normalized["company_id"]
        financial_data = normalized[
            "financial_data"
        ]
        rag_results = normalized[
            "rag_results"
        ]
        risk_data = normalized[
            "risk_data"
        ]
        forecast_data = normalized[
            "forecast_data"
        ]
        alerts = normalized[
            "alerts"
        ]

        intent = self.infer_intent(
            task
        )

        result = FinancialAgentResult(
            result_id=self._create_result_id(),
            agent_name=AGENT_NAME,
            agent_version=AGENT_VERSION,
            generated_at=timestamp(),
            company_id=company_id,
            task=task,
            intent=intent,
            status="processing",
            summary="",
        )

        # --------------------------------------------------------------
        # Extract financial data
        # --------------------------------------------------------------

        metrics = self.calculate_metrics(
            financial_data
        )

        result.financial_metrics = metrics

        # --------------------------------------------------------------
        # Perform analyses
        # --------------------------------------------------------------

        findings = []

        findings.extend(
            self.analyze_revenue(
                financial_data
            )
        )

        findings.extend(
            self.analyze_expenses(
                financial_data
            )
        )

        findings.extend(
            self.analyze_profitability(
                financial_data
            )
        )

        findings.extend(
            self.analyze_liquidity(
                financial_data
            )
        )

        findings.extend(
            self.analyze_working_capital(
                financial_data
            )
        )

        findings.extend(
            self.analyze_leverage(
                financial_data
            )
        )

        findings.extend(
            self.analyze_cash_flow(
                financial_data
            )
        )

        findings.extend(
            self.analyze_budget(
                financial_data
            )
        )

        findings.extend(
            self.analyze_growth(
                financial_data
            )
        )

        findings.extend(
            self.analyze_financial_health(
                financial_data
            )
        )

        # --------------------------------------------------------------
        # Risk/forecast signals
        # --------------------------------------------------------------

        findings.extend(
            self.analyze_external_signals(
                risk_data=risk_data,
                forecast_data=forecast_data,
                alerts=alerts,
            )
        )

        findings = self.rank_findings(
            findings
        )[
            : self.config.max_findings
        ]

        result.findings = findings

        # --------------------------------------------------------------
        # Calculations
        # --------------------------------------------------------------

        result.calculations = (
            self.build_calculation_trace(
                financial_data
            )
        )

        # --------------------------------------------------------------
        # Evidence
        # --------------------------------------------------------------

        result.evidence = (
            self.normalize_evidence(
                rag_results
            )
        )

        # --------------------------------------------------------------
        # Recommendations
        # --------------------------------------------------------------

        result.recommendations = (
            self.generate_recommendations(
                findings=findings,
                metrics=metrics,
                financial_data=financial_data,
            )
        )

        # --------------------------------------------------------------
        # Warnings
        # --------------------------------------------------------------

        result.warnings = (
            self.build_warnings(
                financial_data=financial_data,
                findings=findings,
                rag_results=rag_results,
            )
        )

        # --------------------------------------------------------------
        # Assumptions
        # --------------------------------------------------------------

        result.assumptions = (
            self.build_assumptions(
                financial_data
            )
        )

        # --------------------------------------------------------------
        # Human review
        # --------------------------------------------------------------

        result.human_review_required = (
            self.requires_human_review(
                findings=findings,
                recommendations=result.recommendations,
            )
        )

        # --------------------------------------------------------------
        # Confidence
        # --------------------------------------------------------------

        result.confidence = (
            self.calculate_confidence(
                financial_data=financial_data,
                findings=findings,
                evidence=result.evidence,
            )
        )

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        result.summary = (
            self.build_summary(
                result
            )
        )

        result.status = "completed"

        result.metadata = {
            "analysis_categories": sorted(
                {
                    finding.category
                    for finding in findings
                }
            ),
            "finding_count": len(findings),
            "recommendation_count": len(
                result.recommendations
            ),
            "evidence_count": len(
                result.evidence
            ),
            "currency": self.config.currency,
        }

        return result

    # ========================================================================
    # CONTEXT EXTRACTION
    # ========================================================================

    def _extract_context(
        self,
        *,
        context: Optional[Any],
        task: Optional[str],
        company_id: Optional[str],
        financial_data: Optional[
            Mapping[str, Any]
        ],
        rag_results: Optional[
            Sequence[Any]
        ],
        risk_data: Optional[
            Mapping[str, Any]
        ],
        forecast_data: Optional[
            Mapping[str, Any]
        ],
        alerts: Optional[
            Sequence[Any]
        ],
    ) -> Dict[str, Any]:
        """Extract compatible data from AgentContext."""

        if context is not None:

            if not task:
                task = first_value(
                    context,
                    [
                        "task",
                    ],
                    "Perform financial analysis.",
                )

            if not company_id:
                company_id = first_value(
                    context,
                    [
                        "company_id",
                    ],
                )

            if financial_data is None:

                financial = get_value(
                    context,
                    "financial",
                    None,
                )

                if financial is not None:

                    if hasattr(
                        financial,
                        "to_dict",
                    ):
                        financial_data = (
                            financial.to_dict()
                        )

                    elif isinstance(
                        financial,
                        Mapping,
                    ):
                        financial_data = (
                            financial
                        )

            if rag_results is None:

                rag_results = get_value(
                    context,
                    "rag",
                    [],
                )

            if risk_data is None:

                risk = get_value(
                    context,
                    "risk",
                    None,
                )

                if risk is not None:

                    if hasattr(
                        risk,
                        "to_dict",
                    ):
                        risk_data = (
                            risk.to_dict()
                        )

                    elif isinstance(
                        risk,
                        Mapping,
                    ):
                        risk_data = risk

            if forecast_data is None:

                forecast = get_value(
                    context,
                    "forecast",
                    None,
                )

                if forecast is not None:

                    if hasattr(
                        forecast,
                        "to_dict",
                    ):
                        forecast_data = (
                            forecast.to_dict()
                        )

                    elif isinstance(
                        forecast,
                        Mapping,
                    ):
                        forecast_data = forecast

            if alerts is None:

                alert_context = get_value(
                    context,
                    "alerts",
                    None,
                )

                if alert_context is not None:

                    alerts = get_value(
                        alert_context,
                        "alerts",
                        [],
                    )

        return {
            "task": normalize_text(
                task
                or "Perform financial analysis."
            ),
            "company_id": (
                str(company_id)
                if company_id is not None
                else None
            ),
            "financial_data": (
                financial_data
                or {}
            ),
            "rag_results": (
                rag_results
                or []
            ),
            "risk_data": (
                risk_data
                or {}
            ),
            "forecast_data": (
                forecast_data
                or {}
            ),
            "alerts": (
                alerts
                or []
            ),
        }

    # ========================================================================
    # INTENT
    # ========================================================================

    def infer_intent(
        self,
        task: str,
    ) -> str:
        """Infer financial-analysis intent."""

        text = normalize_text(
            task
        ).lower()

        if any(
            word in text
            for word in [
                "revenue",
                "sales",
                "growth",
            ]
        ):
            return "revenue_analysis"

        if any(
            word in text
            for word in [
                "expense",
                "cost",
                "spending",
            ]
        ):
            return "expense_analysis"

        if any(
            word in text
            for word in [
                "profit",
                "margin",
                "profitability",
            ]
        ):
            return "profitability_analysis"

        if any(
            word in text
            for word in [
                "cash",
                "cashflow",
                "cash flow",
            ]
        ):
            return "cash_flow_analysis"

        if any(
            word in text
            for word in [
                "liquidity",
                "current ratio",
                "quick ratio",
            ]
        ):
            return "liquidity_analysis"

        if any(
            word in text
            for word in [
                "debt",
                "leverage",
                "interest coverage",
            ]
        ):
            return "leverage_analysis"

        if any(
            word in text
            for word in [
                "budget",
                "variance",
            ]
        ):
            return "budget_analysis"

        if any(
            word in text
            for word in [
                "health",
                "overall",
                "financial position",
            ]
        ):
            return "financial_health_analysis"

        return "general_financial_analysis"

    # ========================================================================
    # METRIC CALCULATION
    # ========================================================================

    def calculate_metrics(
        self,
        data: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Calculate standardized financial metrics."""

        revenue = self._field(
            data,
            [
                "revenue",
                "sales",
                "total_revenue",
            ],
        )

        previous_revenue = self._field(
            data,
            [
                "previous_revenue",
                "prior_revenue",
                "last_period_revenue",
            ],
        )

        expenses = self._field(
            data,
            [
                "expenses",
                "total_expenses",
                "operating_expenses",
            ],
        )

        previous_expenses = self._field(
            data,
            [
                "previous_expenses",
                "prior_expenses",
            ],
        )

        gross_profit = self._field(
            data,
            [
                "gross_profit",
            ],
        )

        operating_income = self._field(
            data,
            [
                "operating_income",
                "operating_profit",
            ],
        )

        net_income = self._field(
            data,
            [
                "net_income",
                "net_profit",
            ],
        )

        cash = self._field(
            data,
            [
                "cash",
                "cash_and_equivalents",
            ],
        )

        current_assets = self._field(
            data,
            [
                "current_assets",
            ],
        )

        current_liabilities = self._field(
            data,
            [
                "current_liabilities",
            ],
        )

        inventory = self._field(
            data,
            [
                "inventory",
            ],
        )

        receivables = self._field(
            data,
            [
                "receivables",
                "accounts_receivable",
            ],
        )

        payables = self._field(
            data,
            [
                "payables",
                "accounts_payable",
            ],
        )

        total_debt = self._field(
            data,
            [
                "total_debt",
                "debt",
            ],
        )

        equity = self._field(
            data,
            [
                "total_equity",
                "equity",
            ],
        )

        operating_cash_flow = self._field(
            data,
            [
                "operating_cash_flow",
                "cash_from_operations",
                "operating_cashflow",
            ],
        )

        free_cash_flow = self._field(
            data,
            [
                "free_cash_flow",
            ],
        )

        budget = self._field(
            data,
            [
                "budget",
                "budget_amount",
            ],
        )

        actual_expenses = self._field(
            data,
            [
                "actual_expenses",
                "actual",
            ],
        )

        interest_expense = self._field(
            data,
            [
                "interest_expense",
                "interest_expenses",
            ],
        )

        metrics = {
            "revenue": revenue,
            "previous_revenue": previous_revenue,
            "revenue_growth": (
                safe_divide(
                    revenue - previous_revenue,
                    previous_revenue,
                )
                if previous_revenue
                else None
            ),
            "expenses": expenses,
            "previous_expenses": previous_expenses,
            "expense_growth": (
                safe_divide(
                    expenses - previous_expenses,
                    previous_expenses,
                )
                if previous_expenses
                else None
            ),
            "gross_profit": gross_profit,
            "gross_margin": (
                safe_divide(
                    gross_profit,
                    revenue,
                )
                if revenue
                else None
            ),
            "operating_income": operating_income,
            "operating_margin": (
                safe_divide(
                    operating_income,
                    revenue,
                )
                if revenue
                else None
            ),
            "net_income": net_income,
            "net_margin": (
                safe_divide(
                    net_income,
                    revenue,
                )
                if revenue
                else None
            ),
            "cash": cash,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "current_ratio": (
                safe_divide(
                    current_assets,
                    current_liabilities,
                )
                if current_liabilities
                else None
            ),
            "quick_ratio": (
                safe_divide(
                    current_assets - inventory,
                    current_liabilities,
                )
                if current_liabilities
                else None
            ),
            "inventory": inventory,
            "receivables": receivables,
            "payables": payables,
            "total_debt": total_debt,
            "equity": equity,
            "debt_to_equity": (
                safe_divide(
                    total_debt,
                    equity,
                )
                if equity
                else None
            ),
            "operating_cash_flow": (
                operating_cash_flow
            ),
            "free_cash_flow": free_cash_flow,
            "budget": budget,
            "actual_expenses": actual_expenses,
            "budget_variance": (
                actual_expenses - budget
                if budget
                and actual_expenses
                else None
            ),
            "budget_variance_percent": (
                safe_divide(
                    actual_expenses - budget,
                    budget,
                )
                if budget
                else None
            ),
            "interest_expense": interest_expense,
            "interest_coverage": (
                safe_divide(
                    operating_income,
                    interest_expense,
                )
                if interest_expense
                else None
            ),
        }

        return {
            key: (
                round_value(value)
                if isinstance(
                    value,
                    (int, float),
                )
                else value
            )
            for key, value in metrics.items()
        }

    # ========================================================================
    # REVENUE
    # ========================================================================

    def analyze_revenue(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Analyze revenue performance."""

        findings = []

        revenue = self._field(
            data,
            [
                "revenue",
                "sales",
                "total_revenue",
            ],
        )

        previous = self._field(
            data,
            [
                "previous_revenue",
                "prior_revenue",
            ],
        )

        if not revenue:
            return findings

        if previous:

            growth = safe_divide(
                revenue - previous,
                previous,
            )

            if (
                growth
                <= -self.config.revenue_decline_critical
            ):

                findings.append(
                    self._finding(
                        category="revenue",
                        title="Critical revenue decline",
                        description=(
                            "Revenue has declined materially "
                            "compared with the previous period."
                        ),
                        metric="revenue_growth",
                        value=growth,
                        benchmark=(
                            -self.config.revenue_decline_critical
                        ),
                        variance=(
                            growth
                            + self.config.revenue_decline_critical
                        ),
                        severity="critical",
                        priority="critical",
                        confidence=0.94,
                        rationale=(
                            "Revenue deterioration can reduce "
                            "profitability, liquidity, and "
                            "future operating capacity."
                        ),
                    )
                )

            elif (
                growth
                <= -self.config.revenue_decline_warning
            ):

                findings.append(
                    self._finding(
                        category="revenue",
                        title="Revenue decline",
                        description=(
                            "Revenue is declining compared "
                            "with the previous period."
                        ),
                        metric="revenue_growth",
                        value=growth,
                        benchmark=(
                            -self.config.revenue_decline_warning
                        ),
                        variance=(
                            growth
                            + self.config.revenue_decline_warning
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.88,
                        rationale=(
                            "A sustained revenue decline "
                            "may pressure margins and cash flow."
                        ),
                    )
                )

            elif growth > 0.10:

                findings.append(
                    self._finding(
                        category="revenue",
                        title="Strong revenue growth",
                        description=(
                            "Revenue is growing strongly "
                            "compared with the previous period."
                        ),
                        metric="revenue_growth",
                        value=growth,
                        benchmark=0.10,
                        variance=growth - 0.10,
                        severity="low",
                        priority="medium",
                        confidence=0.86,
                        rationale=(
                            "Strong growth may support "
                            "profitability, provided costs "
                            "remain controlled."
                        ),
                    )
                )

        return findings

    # ========================================================================
    # EXPENSES
    # ========================================================================

    def analyze_expenses(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Analyze expense performance."""

        findings = []

        expenses = self._field(
            data,
            [
                "expenses",
                "total_expenses",
                "operating_expenses",
            ],
        )

        previous = self._field(
            data,
            [
                "previous_expenses",
                "prior_expenses",
            ],
        )

        revenue = self._field(
            data,
            [
                "revenue",
                "sales",
            ],
        )

        if not expenses:
            return findings

        if previous:

            growth = safe_divide(
                expenses - previous,
                previous,
            )

            if (
                growth
                >= self.config.expense_growth_critical
            ):

                findings.append(
                    self._finding(
                        category="expenses",
                        title="Critical expense growth",
                        description=(
                            "Expenses are increasing materially "
                            "faster than expected."
                        ),
                        metric="expense_growth",
                        value=growth,
                        benchmark=(
                            self.config.expense_growth_critical
                        ),
                        variance=(
                            growth
                            - self.config.expense_growth_critical
                        ),
                        severity="critical",
                        priority="critical",
                        confidence=0.93,
                        rationale=(
                            "Rapid expense growth can "
                            "compress operating margins."
                        ),
                    )
                )

            elif (
                growth
                >= self.config.expense_growth_warning
            ):

                findings.append(
                    self._finding(
                        category="expenses",
                        title="Elevated expense growth",
                        description=(
                            "Expenses are growing faster "
                            "than the preferred threshold."
                        ),
                        metric="expense_growth",
                        value=growth,
                        benchmark=(
                            self.config.expense_growth_warning
                        ),
                        variance=(
                            growth
                            - self.config.expense_growth_warning
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.87,
                        rationale=(
                            "Expense growth should be reviewed "
                            "against revenue growth."
                        ),
                    )
                )

        if revenue:

            expense_ratio = safe_divide(
                expenses,
                revenue,
            )

            if expense_ratio > 0.90:

                findings.append(
                    self._finding(
                        category="expenses",
                        title="High expense-to-revenue ratio",
                        description=(
                            "Expenses consume a large proportion "
                            "of reported revenue."
                        ),
                        metric="expense_ratio",
                        value=expense_ratio,
                        benchmark=0.90,
                        variance=(
                            expense_ratio - 0.90
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.89,
                        rationale=(
                            "A high expense ratio leaves "
                            "limited room for operating profit."
                        ),
                    )
                )

        return findings

    # ========================================================================
    # PROFITABILITY
    # ========================================================================

    def analyze_profitability(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Analyze gross, operating and net profitability."""

        findings = []

        revenue = self._field(
            data,
            [
                "revenue",
                "sales",
            ],
        )

        gross_profit = self._field(
            data,
            [
                "gross_profit",
            ],
        )

        operating_income = self._field(
            data,
            [
                "operating_income",
                "operating_profit",
            ],
        )

        net_income = self._field(
            data,
            [
                "net_income",
                "net_profit",
            ],
        )

        if not revenue:
            return findings

        if gross_profit:

            gross_margin = safe_divide(
                gross_profit,
                revenue,
            )

            if (
                gross_margin
                < self.config.critical_margin_threshold
            ):

                findings.append(
                    self._finding(
                        category="profitability",
                        title="Critical gross margin",
                        description=(
                            "Gross margin is below the "
                            "critical profitability threshold."
                        ),
                        metric="gross_margin",
                        value=gross_margin,
                        benchmark=(
                            self.config.critical_margin_threshold
                        ),
                        variance=(
                            gross_margin
                            - self.config.critical_margin_threshold
                        ),
                        severity="critical",
                        priority="critical",
                        confidence=0.93,
                        rationale=(
                            "Low gross margin can indicate "
                            "pricing pressure, high direct costs, "
                            "or unfavorable product mix."
                        ),
                    )
                )

            elif (
                gross_margin
                < self.config.low_margin_threshold
            ):

                findings.append(
                    self._finding(
                        category="profitability",
                        title="Low gross margin",
                        description=(
                            "Gross margin is below the "
                            "preferred profitability level."
                        ),
                        metric="gross_margin",
                        value=gross_margin,
                        benchmark=(
                            self.config.low_margin_threshold
                        ),
                        variance=(
                            gross_margin
                            - self.config.low_margin_threshold
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.89,
                        rationale=(
                            "Margin improvement may require "
                            "pricing, sourcing, or product-mix "
                            "changes."
                        ),
                    )
                )

        if operating_income is not None:

            operating_margin = safe_divide(
                operating_income,
                revenue,
            )

            if operating_margin < 0:

                findings.append(
                    self._finding(
                        category="profitability",
                        title="Negative operating margin",
                        description=(
                            "Operating activities are "
                            "currently generating a loss."
                        ),
                        metric="operating_margin",
                        value=operating_margin,
                        benchmark=0.0,
                        variance=operating_margin,
                        severity="critical",
                        priority="critical",
                        confidence=0.96,
                        rationale=(
                            "Persistent operating losses "
                            "can materially affect liquidity "
                            "and long-term viability."
                        ),
                    )
                )

        if net_income is not None:

            net_margin = safe_divide(
                net_income,
                revenue,
            )

            if net_margin < 0:

                findings.append(
                    self._finding(
                        category="profitability",
                        title="Negative net margin",
                        description=(
                            "The company is reporting "
                            "a net loss for the period."
                        ),
                        metric="net_margin",
                        value=net_margin,
                        benchmark=0.0,
                        variance=net_margin,
                        severity="critical",
                        priority="critical",
                        confidence=0.96,
                        rationale=(
                            "A net loss requires investigation "
                            "of revenue, operating costs, "
                            "financing costs, and exceptional items."
                        ),
                    )
                )

        return findings

    # ========================================================================
    # LIQUIDITY
    # ========================================================================

    def analyze_liquidity(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Analyze liquidity."""

        findings = []

        current_assets = self._field(
            data,
            [
                "current_assets",
            ],
        )

        current_liabilities = self._field(
            data,
            [
                "current_liabilities",
            ],
        )

        inventory = self._field(
            data,
            [
                "inventory",
            ],
        )

        if not current_liabilities:
            return findings

        current_ratio = safe_divide(
            current_assets,
            current_liabilities,
        )

        if (
            current_ratio
            < self.config.current_ratio_critical
        ):

            findings.append(
                self._finding(
                    category="liquidity",
                    title="Critical current ratio",
                    description=(
                        "Current assets are insufficient "
                        "relative to current liabilities."
                    ),
                    metric="current_ratio",
                    value=current_ratio,
                    benchmark=(
                        self.config.current_ratio_critical
                    ),
                    variance=(
                        current_ratio
                        - self.config.current_ratio_critical
                    ),
                    severity="critical",
                    priority="critical",
                    confidence=0.95,
                    rationale=(
                        "The company may face difficulty "
                        "meeting short-term obligations."
                    ),
                )
            )

        elif (
            current_ratio
            < self.config.current_ratio_warning
        ):

            findings.append(
                self._finding(
                    category="liquidity",
                    title="Weak current ratio",
                    description=(
                        "Short-term liquidity is below "
                        "the preferred threshold."
                    ),
                    metric="current_ratio",
                    value=current_ratio,
                    benchmark=(
                        self.config.current_ratio_warning
                    ),
                    variance=(
                        current_ratio
                        - self.config.current_ratio_warning
                    ),
                    severity="high",
                    priority="high",
                    confidence=0.89,
                    rationale=(
                        "Working-capital management "
                        "should be reviewed."
                    ),
                )
            )

        if inventory is not None:

            quick_ratio = safe_divide(
                current_assets - inventory,
                current_liabilities,
            )

            if (
                quick_ratio
                < self.config.quick_ratio_critical
            ):

                findings.append(
                    self._finding(
                        category="liquidity",
                        title="Critical quick ratio",
                        description=(
                            "Liquid current assets are "
                            "low relative to current liabilities."
                        ),
                        metric="quick_ratio",
                        value=quick_ratio,
                        benchmark=(
                            self.config.quick_ratio_critical
                        ),
                        variance=(
                            quick_ratio
                            - self.config.quick_ratio_critical
                        ),
                        severity="critical",
                        priority="critical",
                        confidence=0.94,
                        rationale=(
                            "Inventory may not be immediately "
                            "convertible into cash."
                        ),
                    )
                )

            elif (
                quick_ratio
                < self.config.quick_ratio_warning
            ):

                findings.append(
                    self._finding(
                        category="liquidity",
                        title="Weak quick ratio",
                        description=(
                            "Liquid assets excluding inventory "
                            "are below the preferred level."
                        ),
                        metric="quick_ratio",
                        value=quick_ratio,
                        benchmark=(
                            self.config.quick_ratio_warning
                        ),
                        variance=(
                            quick_ratio
                            - self.config.quick_ratio_warning
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.88,
                        rationale=(
                            "Cash and receivable conversion "
                            "should be monitored."
                        ),
                    )
                )

        return findings

    # ========================================================================
    # WORKING CAPITAL
    # ========================================================================

    def analyze_working_capital(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Analyze receivables, inventory and payables."""

        findings = []

        revenue = self._field(
            data,
            [
                "revenue",
                "sales",
            ],
        )

        receivables = self._field(
            data,
            [
                "receivables",
                "accounts_receivable",
            ],
        )

        inventory = self._field(
            data,
            [
                "inventory",
            ],
        )

        expenses = self._field(
            data,
            [
                "expenses",
                "cost_of_goods_sold",
            ],
        )

        payables = self._field(
            data,
            [
                "payables",
                "accounts_payable",
            ],
        )

        if revenue and receivables:

            dso = (
                safe_divide(
                    receivables,
                    revenue,
                )
                * 365
            )

            if dso >= self.config.dso_critical:

                findings.append(
                    self._finding(
                        category="working_capital",
                        title="Critical receivables days",
                        description=(
                            "Receivables collection time "
                            "is materially elevated."
                        ),
                        metric="dso",
                        value=dso,
                        benchmark=self.config.dso_critical,
                        variance=(
                            dso
                            - self.config.dso_critical
                        ),
                        severity="critical",
                        priority="critical",
                        confidence=0.92,
                        rationale=(
                            "Slow collection can restrict "
                            "cash availability."
                        ),
                    )
                )

            elif dso >= self.config.dso_warning:

                findings.append(
                    self._finding(
                        category="working_capital",
                        title="Elevated receivables days",
                        description=(
                            "Customer collection time "
                            "is above the preferred range."
                        ),
                        metric="dso",
                        value=dso,
                        benchmark=self.config.dso_warning,
                        variance=(
                            dso
                            - self.config.dso_warning
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.87,
                        rationale=(
                            "Accelerating collections can "
                            "improve operating liquidity."
                        ),
                    )
                )

        if expenses and inventory:

            inventory_days = (
                safe_divide(
                    inventory,
                    expenses,
                )
                * 365
            )

            if (
                inventory_days
                >= self.config.inventory_days_critical
            ):

                findings.append(
                    self._finding(
                        category="working_capital",
                        title="Excessive inventory days",
                        description=(
                            "Inventory is tied up for "
                            "an extended period."
                        ),
                        metric="inventory_days",
                        value=inventory_days,
                        benchmark=(
                            self.config.inventory_days_critical
                        ),
                        variance=(
                            inventory_days
                            - self.config.inventory_days_critical
                        ),
                        severity="critical",
                        priority="high",
                        confidence=0.88,
                        rationale=(
                            "Excess inventory can increase "
                            "carrying costs and reduce liquidity."
                        ),
                    )
                )

            elif (
                inventory_days
                >= self.config.inventory_days_warning
            ):

                findings.append(
                    self._finding(
                        category="working_capital",
                        title="Elevated inventory days",
                        description=(
                            "Inventory turnover appears "
                            "slower than preferred."
                        ),
                        metric="inventory_days",
                        value=inventory_days,
                        benchmark=(
                            self.config.inventory_days_warning
                        ),
                        variance=(
                            inventory_days
                            - self.config.inventory_days_warning
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.84,
                        rationale=(
                            "Inventory optimization can "
                            "release working capital."
                        ),
                    )
                )

        if (
            payables
            and expenses
        ):

            dpo = (
                safe_divide(
                    payables,
                    expenses,
                )
                * 365
            )

            if dpo < 15:

                findings.append(
                    self._finding(
                        category="working_capital",
                        title="Low supplier payment period",
                        description=(
                            "Supplier payments appear "
                            "to occur relatively quickly."
                        ),
                        metric="dpo",
                        value=dpo,
                        benchmark=15,
                        variance=dpo - 15,
                        severity="medium",
                        priority="medium",
                        confidence=0.78,
                        rationale=(
                            "Negotiating commercially appropriate "
                            "payment terms may improve working capital."
                        ),
                    )
                )

        return findings

    # ========================================================================
    # LEVERAGE
    # ========================================================================

    def analyze_leverage(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Analyze debt and financing risk."""

        findings = []

        debt = self._field(
            data,
            [
                "total_debt",
                "debt",
            ],
        )

        equity = self._field(
            data,
            [
                "total_equity",
                "equity",
            ],
        )

        operating_income = self._field(
            data,
            [
                "operating_income",
            ],
        )

        interest_expense = self._field(
            data,
            [
                "interest_expense",
                "interest_expenses",
            ],
        )

        if debt and equity:

            ratio = safe_divide(
                debt,
                equity,
            )

            if (
                ratio
                >= self.config.debt_equity_critical
            ):

                findings.append(
                    self._finding(
                        category="leverage",
                        title="Critical debt-to-equity",
                        description=(
                            "Debt is high relative "
                            "to shareholder equity."
                        ),
                        metric="debt_to_equity",
                        value=ratio,
                        benchmark=(
                            self.config.debt_equity_critical
                        ),
                        variance=(
                            ratio
                            - self.config.debt_equity_critical
                        ),
                        severity="critical",
                        priority="critical",
                        confidence=0.93,
                        rationale=(
                            "High leverage can increase "
                            "financial and refinancing risk."
                        ),
                    )
                )

            elif (
                ratio
                >= self.config.debt_equity_warning
            ):

                findings.append(
                    self._finding(
                        category="leverage",
                        title="Elevated debt-to-equity",
                        description=(
                            "Debt is elevated relative "
                            "to shareholder equity."
                        ),
                        metric="debt_to_equity",
                        value=ratio,
                        benchmark=(
                            self.config.debt_equity_warning
                        ),
                        variance=(
                            ratio
                            - self.config.debt_equity_warning
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.87,
                        rationale=(
                            "Debt reduction or stronger "
                            "equity generation may reduce risk."
                        ),
                    )
                )

        if operating_income and interest_expense:

            coverage = safe_divide(
                operating_income,
                interest_expense,
            )

            if (
                coverage
                <= self.config.interest_coverage_critical
            ):

                findings.append(
                    self._finding(
                        category="leverage",
                        title="Critical interest coverage",
                        description=(
                            "Operating income provides "
                            "limited coverage for interest expense."
                        ),
                        metric="interest_coverage",
                        value=coverage,
                        benchmark=(
                            self.config.interest_coverage_critical
                        ),
                        variance=(
                            coverage
                            - self.config.interest_coverage_critical
                        ),
                        severity="critical",
                        priority="critical",
                        confidence=0.94,
                        rationale=(
                            "Weak interest coverage can "
                            "increase default and refinancing risk."
                        ),
                    )
                )

            elif (
                coverage
                <= self.config.interest_coverage_warning
            ):

                findings.append(
                    self._finding(
                        category="leverage",
                        title="Weak interest coverage",
                        description=(
                            "Operating income provides "
                            "limited interest coverage."
                        ),
                        metric="interest_coverage",
                        value=coverage,
                        benchmark=(
                            self.config.interest_coverage_warning
                        ),
                        variance=(
                            coverage
                            - self.config.interest_coverage_warning
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.89,
                        rationale=(
                            "Financing costs should be "
                            "monitored closely."
                        ),
                    )
                )

        return findings

    # ========================================================================
    # CASH FLOW
    # ========================================================================

    def analyze_cash_flow(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Analyze operating and free cash flow."""

        findings = []

        operating_cash_flow = self._field(
            data,
            [
                "operating_cash_flow",
                "cash_from_operations",
            ],
        )

        free_cash_flow = self._field(
            data,
            [
                "free_cash_flow",
            ],
        )

        revenue = self._field(
            data,
            [
                "revenue",
                "sales",
            ],
        )

        if operating_cash_flow is not None:

            if operating_cash_flow < 0:

                findings.append(
                    self._finding(
                        category="cash_flow",
                        title="Negative operating cash flow",
                        description=(
                            "Core operating activities "
                            "are consuming cash."
                        ),
                        metric="operating_cash_flow",
                        value=operating_cash_flow,
                        benchmark=0.0,
                        variance=operating_cash_flow,
                        severity="critical",
                        priority="critical",
                        confidence=0.96,
                        rationale=(
                            "Persistent negative operating "
                            "cash flow may create liquidity pressure."
                        ),
                    )
                )

        if free_cash_flow is not None:

            if free_cash_flow < 0:

                findings.append(
                    self._finding(
                        category="cash_flow",
                        title="Negative free cash flow",
                        description=(
                            "Free cash flow is negative "
                            "for the analyzed period."
                        ),
                        metric="free_cash_flow",
                        value=free_cash_flow,
                        benchmark=0.0,
                        variance=free_cash_flow,
                        severity="high",
                        priority="high",
                        confidence=0.92,
                        rationale=(
                            "Management should review "
                            "operating cash generation and capital expenditure."
                        ),
                    )
                )

        if (
            revenue
            and operating_cash_flow is not None
        ):

            cash_conversion = safe_divide(
                operating_cash_flow,
                revenue,
            )

            if cash_conversion < 0:

                findings.append(
                    self._finding(
                        category="cash_flow",
                        title="Negative operating cash conversion",
                        description=(
                            "Operating cash generation is "
                            "negative relative to revenue."
                        ),
                        metric="operating_cash_conversion",
                        value=cash_conversion,
                        benchmark=0.0,
                        variance=cash_conversion,
                        severity="high",
                        priority="high",
                        confidence=0.90,
                        rationale=(
                            "Revenue growth alone may not "
                            "translate into available cash."
                        ),
                    )
                )

        return findings

    # ========================================================================
    # BUDGET
    # ========================================================================

    def analyze_budget(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Analyze budget variance."""

        findings = []

        budget = self._field(
            data,
            [
                "budget",
                "budget_amount",
            ],
        )

        actual = self._field(
            data,
            [
                "actual_expenses",
                "actual",
            ],
        )

        if not budget or actual is None:
            return findings

        variance = actual - budget

        variance_pct = safe_divide(
            variance,
            budget,
        )

        if variance_pct >= 0.20:

            findings.append(
                self._finding(
                    category="budget",
                    title="Significant unfavorable budget variance",
                    description=(
                        "Actual expenses are materially "
                        "above budget."
                    ),
                    metric="budget_variance_percent",
                    value=variance_pct,
                    benchmark=0.20,
                    variance=variance_pct - 0.20,
                    severity="critical",
                    priority="high",
                    confidence=0.93,
                    rationale=(
                        "Persistent unfavorable variance "
                        "can pressure profitability and cash flow."
                    ),
                )
            )

        elif variance_pct >= 0.10:

            findings.append(
                self._finding(
                    category="budget",
                    title="Unfavorable budget variance",
                    description=(
                        "Actual expenses exceed budget "
                        "by more than the preferred threshold."
                    ),
                    metric="budget_variance_percent",
                    value=variance_pct,
                    benchmark=0.10,
                    variance=variance_pct - 0.10,
                    severity="high",
                    priority="high",
                    confidence=0.88,
                    rationale=(
                        "Expense owners should investigate "
                        "the primary variance drivers."
                    ),
                )
            )

        elif variance_pct <= -0.20:

            findings.append(
                self._finding(
                    category="budget",
                    title="Material favorable budget variance",
                    description=(
                        "Actual expenses are significantly "
                        "below budget."
                    ),
                    metric="budget_variance_percent",
                    value=variance_pct,
                    benchmark=-0.20,
                    variance=variance_pct + 0.20,
                    severity="low",
                    priority="medium",
                    confidence=0.80,
                    rationale=(
                        "The variance may reflect genuine "
                        "efficiency or delayed spending."
                    ),
                )
            )

        return findings

    # ========================================================================
    # GROWTH
    # ========================================================================

    def analyze_growth(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Analyze growth quality."""

        findings = []

        revenue = self._field(
            data,
            [
                "revenue",
                "sales",
            ],
        )

        previous_revenue = self._field(
            data,
            [
                "previous_revenue",
                "prior_revenue",
            ],
        )

        expenses = self._field(
            data,
            [
                "expenses",
                "total_expenses",
            ],
        )

        previous_expenses = self._field(
            data,
            [
                "previous_expenses",
                "prior_expenses",
            ],
        )

        if (
            revenue
            and previous_revenue
            and expenses
            and previous_expenses
        ):

            revenue_growth = safe_divide(
                revenue - previous_revenue,
                previous_revenue,
            )

            expense_growth = safe_divide(
                expenses - previous_expenses,
                previous_expenses,
            )

            if (
                expense_growth
                > revenue_growth + 0.05
            ):

                findings.append(
                    self._finding(
                        category="growth",
                        title="Expense growth exceeds revenue growth",
                        description=(
                            "Expenses are increasing faster "
                            "than revenue."
                        ),
                        metric="growth_gap",
                        value=(
                            expense_growth
                            - revenue_growth
                        ),
                        benchmark=0.05,
                        variance=(
                            expense_growth
                            - revenue_growth
                            - 0.05
                        ),
                        severity="high",
                        priority="high",
                        confidence=0.91,
                        rationale=(
                            "This pattern can lead to "
                            "margin compression."
                        ),
                    )
                )

        return findings

    # ========================================================================
    # FINANCIAL HEALTH
    # ========================================================================

    def analyze_financial_health(
        self,
        data: Mapping[str, Any],
    ) -> List[FinancialFinding]:
        """Build an overall financial-health finding."""

        metrics = self.calculate_metrics(
            data
        )

        scores = []

        current_ratio = metrics.get(
            "current_ratio"
        )

        if current_ratio is not None:

            scores.append(
                clamp(
                    safe_divide(
                        current_ratio,
                        2.0,
                    )
                )
            )

        gross_margin = metrics.get(
            "gross_margin"
        )

        if gross_margin is not None:

            scores.append(
                clamp(
                    safe_divide(
                        gross_margin,
                        0.30,
                    )
                )
            )

        operating_margin = metrics.get(
            "operating_margin"
        )

        if operating_margin is not None:

            scores.append(
                clamp(
                    safe_divide(
                        operating_margin,
                        0.20,
                    )
                )
            )

        if not scores:
            return []

        health_score = (
            statistics.mean(
                scores
            )
        )

        if health_score < 0.30:

            severity = "critical"
            priority = "critical"

        elif health_score < 0.50:

            severity = "high"
            priority = "high"

        elif health_score < 0.70:

            severity = "medium"
            priority = "medium"

        else:

            severity = "low"
            priority = "low"

        return [
            self._finding(
                category="financial_health",
                title="Overall financial health assessment",
                description=(
                    "The standardized financial-health "
                    "indicator reflects the available "
                    "liquidity and profitability signals."
                ),
                metric="financial_health_score",
                value=health_score,
                benchmark=0.70,
                variance=health_score - 0.70,
                severity=severity,
                priority=priority,
                confidence=0.76,
                rationale=(
                    "The score combines selected liquidity "
                    "and profitability indicators. It should "
                    "not replace management or professional "
                    "financial judgment."
                ),
            )
        ]

    # ========================================================================
    # EXTERNAL SIGNALS
    # ========================================================================

    def analyze_external_signals(
        self,
        *,
        risk_data: Mapping[str, Any],
        forecast_data: Mapping[str, Any],
        alerts: Sequence[Any],
    ) -> List[FinancialFinding]:
        """Convert risk, forecast and alert signals into findings."""

        findings = []

        if risk_data:

            risk_score = optional_float(
                first_value(
                    risk_data,
                    [
                        "risk_score",
                        "overall_score",
                    ],
                )
            )

            risk_level = normalize_level(
                first_value(
                    risk_data,
                    [
                        "risk_level",
                        "overall_risk",
                    ],
                    "low",
                ),
                "low",
            )

            if (
                risk_score is not None
                and risk_score >= 0.80
            ):

                findings.append(
                    self._finding(
                        category="risk",
                        title="High overall financial risk signal",
                        description=(
                            "The risk engine reports "
                            "an elevated overall risk score."
                        ),
                        metric="risk_score",
                        value=risk_score,
                        benchmark=0.80,
                        variance=risk_score - 0.80,
                        severity=(
                            "critical"
                            if risk_score >= 0.90
                            else "high"
                        ),
                        priority="high",
                        confidence=0.88,
                        rationale=(
                            "The financial agent should "
                            "coordinate with the risk agent "
                            "for deeper investigation."
                        ),
                        metadata={
                            "risk_level": risk_level,
                        },
                    )
                )

        if forecast_data:

            downside = optional_float(
                first_value(
                    forecast_data,
                    [
                        "downside_risk",
                        "downside_probability",
                    ],
                )
            )

            if (
                downside is not None
                and downside >= 0.50
            ):

                findings.append(
                    self._finding(
                        category="forecast",
                        title="Elevated forecast downside risk",
                        description=(
                            "Forecast inputs indicate "
                            "a material downside probability."
                        ),
                        metric="forecast_downside_probability",
                        value=downside,
                        benchmark=0.50,
                        variance=downside - 0.50,
                        severity="high",
                        priority="high",
                        confidence=0.82,
                        rationale=(
                            "Forecast downside should be "
                            "tested using scenario analysis."
                        ),
                    )
                )

        for alert in list(
            alerts or []
        )[:10]:

            severity = normalize_level(
                first_value(
                    alert,
                    [
                        "severity",
                        "risk",
                    ],
                    "medium",
                )
            )

            if severity in {
                "high",
                "critical",
            }:

                title = normalize_text(
                    first_value(
                        alert,
                        [
                            "title",
                            "type",
                        ],
                        "Financial alert",
                    )
                )

                description = normalize_text(
                    first_value(
                        alert,
                        [
                            "description",
                            "message",
                        ],
                        "Active financial alert requires review.",
                    )
                )

                findings.append(
                    self._finding(
                        category="alerts",
                        title=title,
                        description=description,
                        severity=severity,
                        priority=severity,
                        confidence=0.80,
                        rationale=(
                            "An existing alert indicates "
                            "that additional investigation may be required."
                        ),
                    )
                )

        return findings

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================

    def generate_recommendations(
        self,
        *,
        findings: Sequence[
            FinancialFinding
        ],
        metrics: Mapping[str, Any],
        financial_data: Mapping[str, Any],
    ) -> List[
        FinancialRecommendation
    ]:
        """Generate explainable recommendations."""

        recommendations = []

        for finding in findings:

            category = finding.category

            recommendation = (
                self._recommendation_for_finding(
                    finding,
                    metrics,
                    financial_data,
                )
            )

            if recommendation:
                recommendations.append(
                    recommendation
                )

        recommendations = (
            self._deduplicate_recommendations(
                recommendations
            )
        )

        recommendations.sort(
            key=lambda item: (
                PRIORITY_LEVELS.get(
                    item.priority,
                    0,
                ),
                item.confidence,
                abs(item.estimated_impact),
            ),
            reverse=True,
        )

        return recommendations[
            : self.config.max_recommendations
        ]

    def _recommendation_for_finding(
        self,
        finding: FinancialFinding,
        metrics: Mapping[str, Any],
        financial_data: Mapping[str, Any],
    ) -> Optional[
        FinancialRecommendation
    ]:
        """Create recommendation based on finding."""

        category = finding.category

        if category == "revenue":

            return self._create_recommendation(
                category="revenue",
                title="Investigate revenue decline drivers",
                description=(
                    "Identify the customers, products, "
                    "regions, or channels responsible for "
                    "the revenue movement."
                ),
                action=(
                    "Perform revenue decomposition by "
                    "customer, product, geography, and channel "
                    "before approving corrective actions."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="revenue",
                assumptions=[
                    "Revenue data is sufficiently complete.",
                ],
                risks=[
                    "Corrective pricing or sales actions "
                    "may affect demand."
                ],
                next_steps=[
                    "Segment revenue by customer and product.",
                    "Compare current and prior-period performance.",
                    "Identify the largest negative contributors.",
                ],
            )

        if category == "expenses":

            impact = self._estimate_expense_savings(
                metrics
            )

            return self._create_recommendation(
                category="cost_optimization",
                title="Review expense growth",
                description=(
                    "Investigate expense categories growing "
                    "faster than revenue."
                ),
                action=(
                    "Review high-growth expense categories, "
                    "supplier contracts, and discretionary spending."
                ),
                finding=finding,
                estimated_impact=impact,
                impact_type="cost_savings",
                assumptions=[
                    "A portion of elevated expenses "
                    "may be controllable."
                ],
                risks=[
                    "Aggressive cost reduction may "
                    "affect service quality or growth."
                ],
                next_steps=[
                    "Rank expenses by growth.",
                    "Separate fixed and variable costs.",
                    "Review major supplier agreements.",
                ],
            )

        if category == "profitability":

            return self._create_recommendation(
                category="profitability",
                title="Improve operating profitability",
                description=(
                    "Address the largest drivers of "
                    "margin compression."
                ),
                action=(
                    "Evaluate pricing, product mix, direct costs, "
                    "and operating expenses before selecting "
                    "specific margin-improvement actions."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="profitability",
                assumptions=[
                    "Margin drivers are identifiable "
                    "from available financial data."
                ],
                risks=[
                    "Pricing changes can affect volume.",
                    "Cost reductions can affect operations.",
                ],
                next_steps=[
                    "Analyze gross margin by product.",
                    "Analyze operating expense ratios.",
                    "Run margin what-if scenarios.",
                ],
            )

        if category == "liquidity":

            return self._create_recommendation(
                category="liquidity",
                title="Strengthen short-term liquidity",
                description=(
                    "Improve the company's ability to "
                    "meet near-term obligations."
                ),
                action=(
                    "Prioritize receivables collection, "
                    "cash preservation, and appropriate "
                    "working-capital actions."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="liquidity",
                assumptions=[
                    "Liquidity metrics are based on "
                    "current-period data."
                ],
                risks=[
                    "Working-capital actions may affect "
                    "supplier or customer relationships."
                ],
                next_steps=[
                    "Review overdue receivables.",
                    "Review short-term obligations.",
                    "Model cash runway under downside scenarios.",
                ],
            )

        if category == "working_capital":

            return self._create_recommendation(
                category="working_capital",
                title="Optimize working capital",
                description=(
                    "Reduce cash tied up in receivables "
                    "and inventory where commercially appropriate."
                ),
                action=(
                    "Prioritize overdue receivables and "
                    "slow-moving inventory."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="working_capital",
                assumptions=[
                    "Receivables and inventory data "
                    "are representative."
                ],
                risks=[
                    "Tighter collection terms may affect "
                    "customer relationships."
                ],
                next_steps=[
                    "Create an aging analysis.",
                    "Identify slow-moving inventory.",
                    "Review supplier payment terms.",
                ],
            )

        if category == "leverage":

            return self._create_recommendation(
                category="leverage",
                title="Review debt and financing structure",
                description=(
                    "Evaluate leverage and debt-service "
                    "capacity."
                ),
                action=(
                    "Review debt maturities, interest costs, "
                    "coverage ratios, and refinancing exposure."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="risk_reduction",
                assumptions=[
                    "Debt balances and interest expense "
                    "are complete."
                ],
                risks=[
                    "Refinancing or debt restructuring "
                    "may have contractual implications."
                ],
                next_steps=[
                    "Map debt maturity dates.",
                    "Review interest-rate exposure.",
                    "Model debt-service scenarios.",
                ],
            )

        if category == "cash_flow":

            return self._create_recommendation(
                category="cash_flow",
                title="Improve cash generation",
                description=(
                    "Investigate the primary causes of "
                    "weak operating or free cash flow."
                ),
                action=(
                    "Review working capital, operating costs, "
                    "capital expenditure, and cash conversion."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="cash_flow",
                assumptions=[
                    "Cash-flow data reconciles to "
                    "the financial statements."
                ],
                risks=[
                    "Reducing capital expenditure "
                    "may affect future growth."
                ],
                next_steps=[
                    "Perform cash-flow bridge analysis.",
                    "Review working-capital movements.",
                    "Separate maintenance and growth capex.",
                ],
            )

        if category == "budget":

            return self._create_recommendation(
                category="budget",
                title="Investigate budget variance",
                description=(
                    "Identify the categories and owners "
                    "responsible for unfavorable variance."
                ),
                action=(
                    "Perform budget-versus-actual analysis "
                    "and establish corrective controls."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="cost_control",
                assumptions=[
                    "Budget and actual values use "
                    "consistent definitions."
                ],
                risks=[
                    "Overly aggressive budget controls "
                    "may delay necessary spending."
                ],
                next_steps=[
                    "Rank unfavorable variances.",
                    "Identify recurring variance drivers.",
                    "Update forecasts where appropriate.",
                ],
            )

        if category == "growth":

            return self._create_recommendation(
                category="growth",
                title="Align cost growth with revenue growth",
                description=(
                    "Review expense growth relative "
                    "to revenue growth."
                ),
                action=(
                    "Identify expenses growing faster than "
                    "revenue and evaluate their business value."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="margin",
                assumptions=[
                    "Prior-period figures are comparable."
                ],
                risks=[
                    "Cost controls may constrain growth "
                    "if applied indiscriminately."
                ],
                next_steps=[
                    "Compare revenue and expense growth.",
                    "Review cost categories by business impact.",
                    "Run margin sensitivity scenarios.",
                ],
            )

        if category == "risk":

            return self._create_recommendation(
                category="risk",
                title="Investigate elevated financial risk",
                description=(
                    "Coordinate financial analysis with "
                    "the dedicated risk engine."
                ),
                action=(
                    "Identify the financial drivers behind "
                    "the elevated risk score before taking action."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="risk_reduction",
                assumptions=[
                    "Risk-engine output is based on "
                    "current available data."
                ],
                risks=[
                    "Risk scores may change as new data arrives."
                ],
                next_steps=[
                    "Review top risk signals.",
                    "Inspect related alerts.",
                    "Run downside scenarios.",
                ],
            )

        if category == "forecast":

            return self._create_recommendation(
                category="forecast",
                title="Run downside financial scenarios",
                description=(
                    "Test the financial plan against "
                    "forecast downside conditions."
                ),
                action=(
                    "Use the what-if engine to evaluate "
                    "revenue, margin, and cash-flow downside scenarios."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="risk_reduction",
                assumptions=[
                    "Forecast model outputs are estimates.",
                ],
                risks=[
                    "Forecast uncertainty can be significant."
                ],
                next_steps=[
                    "Run base, upside, and downside scenarios.",
                    "Identify liquidity thresholds.",
                    "Prepare contingency actions.",
                ],
            )

        if category == "financial_health":

            return self._create_recommendation(
                category="financial_health",
                title="Monitor overall financial health",
                description=(
                    "Track the main liquidity and profitability "
                    "drivers rather than relying only on a composite score."
                ),
                action=(
                    "Monitor revenue, margins, liquidity, "
                    "cash flow, and leverage on a recurring basis."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="financial_health",
                assumptions=[
                    "The selected financial indicators "
                    "represent the current business condition."
                ],
                risks=[
                    "Composite indicators can hide "
                    "individual risk concentrations."
                ],
                next_steps=[
                    "Review monthly KPI trends.",
                    "Set thresholds for material changes.",
                    "Investigate deteriorating signals promptly.",
                ],
            )

        if category == "alerts":

            return self._create_recommendation(
                category="alerts",
                title="Investigate high-priority financial alert",
                description=(
                    "Review the underlying data and evidence "
                    "associated with the active alert."
                ),
                action=(
                    "Trace the alert to its source metric "
                    "before selecting corrective action."
                ),
                finding=finding,
                estimated_impact=0.0,
                impact_type="risk_reduction",
                assumptions=[
                    "The alert is generated from current data."
                ],
                risks=[
                    "The alert may represent a temporary condition."
                ],
                next_steps=[
                    "Open alert details.",
                    "Review source metrics.",
                    "Document investigation outcome.",
                ],
            )

        return None

    # ========================================================================
    # FINDING / RECOMMENDATION FACTORIES
    # ========================================================================

    def _finding(
        self,
        *,
        category: str,
        title: str,
        description: str,
        metric: Optional[str] = None,
        value: Optional[float] = None,
        benchmark: Optional[float] = None,
        variance: Optional[float] = None,
        severity: str = "medium",
        priority: str = "medium",
        confidence: float = 0.80,
        evidence: Optional[
            Sequence[str]
        ] = None,
        rationale: str = "",
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> FinancialFinding:

        return FinancialFinding(
            finding_id=(
                "FND-"
                + uuid.uuid4().hex[:10].upper()
            ),
            category=category,
            title=title,
            description=description,
            metric=metric,
            value=(
                round_value(value)
                if value is not None
                else None
            ),
            benchmark=(
                round_value(benchmark)
                if benchmark is not None
                else None
            ),
            variance=(
                round_value(variance)
                if variance is not None
                else None
            ),
            severity=normalize_level(
                severity
            ),
            priority=normalize_level(
                priority
            ),
            confidence=clamp(
                safe_float(
                    confidence
                )
            ),
            evidence=list(
                evidence or []
            ),
            rationale=normalize_text(
                rationale
            ),
            metadata=json_safe(
                metadata or {}
            ),
        )

    def _create_recommendation(
        self,
        *,
        category: str,
        title: str,
        description: str,
        action: str,
        finding: FinancialFinding,
        estimated_impact: float,
        impact_type: str,
        assumptions: Sequence[str],
        risks: Sequence[str],
        next_steps: Sequence[str],
    ) -> FinancialRecommendation:

        priority = finding.priority
        risk = finding.severity

        human_review = False

        if (
            self.config.require_human_review_high_risk
            and risk in {
                "high",
                "critical",
            }
        ):
            human_review = True

        if (
            self.config.require_human_review_high_impact
            and abs(
                estimated_impact
            )
            >= self.config.high_impact_threshold
        ):
            human_review = True

        return FinancialRecommendation(
            recommendation_id=(
                "REC-"
                + uuid.uuid4().hex[:10].upper()
            ),
            category=category,
            title=title,
            description=description,
            action=action,
            priority=priority,
            risk=risk,
            confidence=clamp(
                finding.confidence
            ),
            estimated_impact=round_value(
                estimated_impact
            ),
            impact_type=impact_type,
            evidence=list(
                finding.evidence
            ),
            assumptions=list(
                assumptions
            ),
            risks=list(
                risks
            ),
            next_steps=list(
                next_steps
            ),
            human_review_required=(
                human_review
            ),
            metadata={
                "source_finding_id": (
                    finding.finding_id
                ),
                "source_metric": (
                    finding.metric
                ),
            },
        )

    # ========================================================================
    # EVIDENCE
    # ========================================================================

    def normalize_evidence(
        self,
        rag_results: Sequence[Any],
    ) -> List[
        Dict[str, Any]
    ]:
        """Normalize RAG evidence for financial claims."""

        evidence = []

        for item in list(
            rag_results or []
        )[
            : self.config.max_evidence
        ]:

            content = normalize_text(
                first_value(
                    item,
                    [
                        "content",
                        "text",
                        "chunk",
                    ],
                    "",
                )
            )

            if not content:
                continue

            score = safe_float(
                first_value(
                    item,
                    [
                        "score",
                        "similarity",
                        "rerank_score",
                    ],
                    0.0,
                )
            )

            citation = first_value(
                item,
                [
                    "citation",
                    "citation_text",
                ],
            )

            source = first_value(
                item,
                [
                    "source",
                    "filename",
                    "file_name",
                ],
            )

            page = first_value(
                item,
                [
                    "page",
                    "page_number",
                ],
            )

            evidence.append(
                {
                    "chunk_id": first_value(
                        item,
                        [
                            "chunk_id",
                            "id",
                        ],
                    ),
                    "document_id": first_value(
                        item,
                        [
                            "document_id",
                            "doc_id",
                        ],
                    ),
                    "content": content,
                    "score": round_value(
                        score
                    ),
                    "source": source,
                    "page": page,
                    "citation": citation,
                }
            )

        evidence.sort(
            key=lambda item: safe_float(
                item.get(
                    "score",
                    0,
                )
            ),
            reverse=True,
        )

        return evidence

    # ========================================================================
    # CALCULATION TRACE
    # ========================================================================

    def build_calculation_trace(
        self,
        data: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """
        Build transparent calculation trace.

        This is useful for auditability and agent explanations.
        """

        metrics = self.calculate_metrics(
            data
        )

        formulas = {
            "revenue_growth": (
                "(revenue - previous_revenue) "
                "/ previous_revenue"
            ),
            "expense_growth": (
                "(expenses - previous_expenses) "
                "/ previous_expenses"
            ),
            "gross_margin": (
                "gross_profit / revenue"
            ),
            "operating_margin": (
                "operating_income / revenue"
            ),
            "net_margin": (
                "net_income / revenue"
            ),
            "current_ratio": (
                "current_assets / current_liabilities"
            ),
            "quick_ratio": (
                "(current_assets - inventory) "
                "/ current_liabilities"
            ),
            "debt_to_equity": (
                "total_debt / total_equity"
            ),
            "interest_coverage": (
                "operating_income / interest_expense"
            ),
            "dso": (
                "receivables / revenue * 365"
            ),
            "inventory_days": (
                "inventory / expenses * 365"
            ),
            "budget_variance_percent": (
                "(actual_expenses - budget) / budget"
            ),
        }

        return {
            "formulas": formulas,
            "calculated_metrics": metrics,
            "currency": self.config.currency,
        }

    # ========================================================================
    # WARNINGS / ASSUMPTIONS
    # ========================================================================

    def build_warnings(
        self,
        *,
        financial_data: Mapping[str, Any],
        findings: Sequence[
            FinancialFinding
        ],
        rag_results: Sequence[Any],
    ) -> List[str]:
        """Build data-quality and safety warnings."""

        warnings = []

        if not financial_data:
            warnings.append(
                "No structured financial data was supplied."
            )

        if not rag_results:
            warnings.append(
                "No document evidence was supplied; "
                "document-based claims should not be made."
            )

        if any(
            finding.severity == "critical"
            for finding in findings
        ):
            warnings.append(
                "Critical financial findings require "
                "management review before action."
            )

        if any(
            finding.confidence
            < self.config.min_confidence
            for finding in findings
        ):
            warnings.append(
                "Some findings have confidence below "
                "the configured preferred threshold."
            )

        return deduplicate(
            warnings
        )[:MAX_WARNINGS]

    def build_assumptions(
        self,
        data: Mapping[str, Any],
    ) -> List[str]:
        """Build explicit analysis assumptions."""

        assumptions = [
            (
                "Financial values are assumed to use "
                f"the configured currency ({self.config.currency}) "
                "unless explicitly specified."
            ),
            (
                "Ratios are calculated from the supplied "
                "period data and may differ from externally "
                "reported ratios."
            ),
            (
                "Thresholds are analytical defaults and "
                "should be adjusted for the company's industry "
                "and business model."
            ),
        ]

        if not self._field(
            data,
            [
                "previous_revenue",
                "prior_revenue",
            ],
        ):
            assumptions.append(
                "Revenue growth cannot be reliably "
                "calculated without prior-period revenue."
            )

        if not self._field(
            data,
            [
                "previous_expenses",
                "prior_expenses",
            ],
        ):
            assumptions.append(
                "Expense growth cannot be reliably "
                "calculated without prior-period expenses."
            )

        return deduplicate(
            assumptions
        )

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    def calculate_confidence(
        self,
        *,
        financial_data: Mapping[str, Any],
        findings: Sequence[
            FinancialFinding
        ],
        evidence: Sequence[
            Dict[str, Any]
        ],
    ) -> float:
        """Calculate overall analytical confidence."""

        data_score = 0.0

        important_fields = [
            "revenue",
            "expenses",
            "gross_profit",
            "operating_income",
            "cash",
            "current_assets",
            "current_liabilities",
        ]

        available = sum(
            1
            for field_name in important_fields
            if self._field(
                financial_data,
                [field_name],
            )
            is not None
        )

        data_score = (
            available
            / len(important_fields)
        )

        finding_score = (
            statistics.mean(
                [
                    finding.confidence
                    for finding in findings
                ]
            )
            if findings
            else 0.50
        )

        evidence_score = (
            min(
                1.0,
                len(evidence) / 3.0,
            )
            if evidence
            else 0.40
        )

        confidence = (
            data_score * 0.45
            + finding_score * 0.40
            + evidence_score * 0.15
        )

        return round(
            clamp(
                confidence
            ),
            4,
        )

    # ========================================================================
    # HUMAN REVIEW
    # ========================================================================

    def requires_human_review(
        self,
        *,
        findings: Sequence[
            FinancialFinding
        ],
        recommendations: Sequence[
            FinancialRecommendation
        ],
    ) -> bool:
        """Determine whether human review is required."""

        for finding in findings:

            if finding.severity in {
                "critical",
                "high",
            }:
                return True

        for recommendation in recommendations:

            if recommendation.human_review_required:
                return True

        return False

    # ========================================================================
    # SUMMARY
    # ========================================================================

    def build_summary(
        self,
        result: FinancialAgentResult,
    ) -> str:
        """Build concise executive summary."""

        critical = sum(
            1
            for finding in result.findings
            if finding.severity == "critical"
        )

        high = sum(
            1
            for finding in result.findings
            if finding.severity == "high"
        )

        recommendation_count = len(
            result.recommendations
        )

        if critical > 0:

            opening = (
                "Financial analysis identified "
                f"{critical} critical finding(s)"
            )

        elif high > 0:

            opening = (
                "Financial analysis identified "
                f"{high} high-priority finding(s)"
            )

        else:

            opening = (
                "Financial analysis did not identify "
                "critical financial deterioration "
                "from the supplied data"
            )

        return (
            f"{opening}. "
            f"{recommendation_count} recommendation(s) "
            "were generated. "
            f"Overall analytical confidence is "
            f"{result.confidence:.0%}. "
            + (
                "Human review is required before "
                "high-impact or high-risk actions."
                if result.human_review_required
                else
                "No mandatory human review flag was triggered."
            )
        )

    # ========================================================================
    # SORTING / DEDUPLICATION
    # ========================================================================

    def rank_findings(
        self,
        findings: Sequence[
            FinancialFinding
        ],
    ) -> List[
        FinancialFinding
    ]:
        """Rank findings by severity, priority and confidence."""

        result = list(
            findings
        )

        result.sort(
            key=lambda finding: (
                RISK_LEVELS.get(
                    finding.severity,
                    0,
                ),
                PRIORITY_LEVELS.get(
                    finding.priority,
                    0,
                ),
                finding.confidence,
                abs(
                    finding.variance
                    or 0.0
                ),
            ),
            reverse=True,
        )

        return result

    def _deduplicate_recommendations(
        self,
        recommendations: Sequence[
            FinancialRecommendation
        ],
    ) -> List[
        FinancialRecommendation
    ]:
        """Remove duplicate recommendations."""

        seen = set()
        result = []

        for recommendation in recommendations:

            key = (
                recommendation.category,
                recommendation.title.lower(),
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(
                recommendation
            )

        return result

    # ========================================================================
    # ESTIMATION
    # ========================================================================

    def _estimate_expense_savings(
        self,
        metrics: Mapping[str, Any],
    ) -> float:
        """
        Conservative savings estimate.

        This is a planning estimate, not a guaranteed saving.
        """

        expenses = safe_float(
            metrics.get(
                "expenses"
            )
        )

        if expenses <= 0:
            return 0.0

        return round(
            expenses * 0.05,
            2,
        )

    # ========================================================================
    # FIELD ACCESS
    # ========================================================================

    @staticmethod
    def _field(
        data: Mapping[str, Any],
        keys: Sequence[str],
    ) -> float:
        """Return first numeric financial field."""

        value = first_value(
            data,
            keys,
            None,
        )

        return safe_float(
            value,
            0.0,
        )

    # ========================================================================
    # IDs
    # ========================================================================

    @staticmethod
    def _create_result_id() -> str:
        """Create result identifier."""

        return (
            "FIN-"
            + uuid.uuid4().hex[:12].upper()
        )


# ============================================================================
# OUTPUT HELPERS
# ============================================================================

def to_markdown(
    result: FinancialAgentResult,
) -> str:
    """Convert result to Markdown."""

    lines = [
        "# FinCo AI Financial Analysis",
        "",
        f"**Result ID:** `{result.result_id}`",
        f"**Status:** {result.status}",
        f"**Company:** {result.company_id or 'N/A'}",
        f"**Intent:** {result.intent}",
        f"**Confidence:** {result.confidence:.1%}",
        "",
        "## Executive Summary",
        "",
        result.summary,
        "",
        "## Financial Metrics",
        "",
    ]

    for key, value in result.financial_metrics.items():

        if isinstance(
            value,
            float,
        ):
            lines.append(
                f"- **{key}:** {value:.4f}"
            )
        else:
            lines.append(
                f"- **{key}:** {value}"
            )

    lines.extend(
        [
            "",
            "## Findings",
            "",
        ]
    )

    if not result.findings:

        lines.append(
            "No material findings identified."
        )

    for finding in result.findings:

        lines.extend(
            [
                (
                    f"### [{finding.severity.upper()}] "
                    f"{finding.title}"
                ),
                "",
                finding.description,
                "",
                (
                    f"- Metric: `{finding.metric}`"
                    if finding.metric
                    else "- Metric: N/A"
                ),
                (
                    f"- Value: `{finding.value}`"
                    if finding.value is not None
                    else "- Value: N/A"
                ),
                (
                    f"- Benchmark: `{finding.benchmark}`"
                    if finding.benchmark is not None
                    else "- Benchmark: N/A"
                ),
                (
                    f"- Confidence: "
                    f"{finding.confidence:.1%}"
                ),
                "",
                f"**Rationale:** {finding.rationale}",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendations",
            "",
        ]
    )

    if not result.recommendations:

        lines.append(
            "No recommendations generated."
        )

    for recommendation in result.recommendations:

        lines.extend(
            [
                (
                    f"### [{recommendation.priority.upper()}] "
                    f"{recommendation.title}"
                ),
                "",
                recommendation.description,
                "",
                f"**Action:** {recommendation.action}",
                "",
                (
                    f"- Confidence: "
                    f"{recommendation.confidence:.1%}"
                ),
                (
                    f"- Estimated impact: "
                    f"{recommendation.estimated_impact:,.2f}"
                ),
                (
                    f"- Human review: "
                    f"{'Required' if recommendation.human_review_required else 'Not required'}"
                ),
                "",
                "#### Next Steps",
                "",
            ]
        )

        for step in recommendation.next_steps:
            lines.append(
                f"- {step}"
            )

        lines.append("")

    if result.evidence:

        lines.extend(
            [
                "## Evidence",
                "",
            ]
        )

        for evidence in result.evidence:

            citation = (
                evidence.get(
                    "citation"
                )
                or evidence.get(
                    "source"
                )
                or evidence.get(
                    "document_id"
                )
                or "Unknown source"
            )

            lines.append(
                f"- **{citation}** "
                f"(score: {safe_float(evidence.get('score')):.3f})"
            )

    if result.warnings:

        lines.extend(
            [
                "",
                "## Warnings",
                "",
            ]
        )

        for warning in result.warnings:
            lines.append(
                f"- {warning}"
            )

    if result.assumptions:

        lines.extend(
            [
                "",
                "## Assumptions",
                "",
            ]
        )

        for assumption in result.assumptions:
            lines.append(
                f"- {assumption}"
            )

    return "\n".join(
        lines
    )


def to_html(
    result: FinancialAgentResult,
) -> str:
    """Convert result to standalone HTML."""

    def escape(value: Any) -> str:

        text = normalize_text(
            value
        )

        return (
            text.replace(
                "&",
                "&amp;",
            )
            .replace(
                "<",
                "&lt;",
            )
            .replace(
                ">",
                "&gt;",
            )
            .replace(
                '"',
                "&quot;",
            )
        )

    finding_cards = []

    for finding in result.findings:

        finding_cards.append(
            f"""
            <article class="card">
                <div class="badge">
                    {escape(finding.severity.upper())}
                </div>

                <h3>{escape(finding.title)}</h3>

                <p>{escape(finding.description)}</p>

                <div class="meta">
                    <span>
                        Metric:
                        <strong>
                            {escape(finding.metric or "N/A")}
                        </strong>
                    </span>

                    <span>
                        Value:
                        <strong>
                            {escape(finding.value)}
                        </strong>
                    </span>

                    <span>
                        Confidence:
                        <strong>
                            {finding.confidence:.1%}
                        </strong>
                    </span>
                </div>

                <p>
                    <strong>Rationale:</strong>
                    {escape(finding.rationale)}
                </p>
            </article>
            """
        )

    recommendation_cards = []

    for recommendation in result.recommendations:

        recommendation_cards.append(
            f"""
            <article class="card">
                <div class="badge">
                    {escape(recommendation.priority.upper())}
                </div>

                <h3>{escape(recommendation.title)}</h3>

                <p>
                    {escape(recommendation.description)}
                </p>

                <p>
                    <strong>Action:</strong>
                    {escape(recommendation.action)}
                </p>

                <p>
                    <strong>Confidence:</strong>
                    {recommendation.confidence:.1%}
                </p>

                <p>
                    <strong>Estimated impact:</strong>
                    {recommendation.estimated_impact:,.2f}
                </p>

                <p>
                    <strong>Human review:</strong>
                    {
                        "Required"
                        if recommendation.human_review_required
                        else "Not required"
                    }
                </p>
            </article>
            """
        )

    metric_items = []

    for key, value in result.financial_metrics.items():

        metric_items.append(
            f"""
            <div class="metric">
                <span>{escape(key)}</span>
                <strong>{escape(value)}</strong>
            </div>
            """
        )

    warning_html = "".join(
        f"<li>{escape(warning)}</li>"
        for warning in result.warnings
    )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>FinCo AI Financial Analysis</title>

<style>

:root {{
    font-family:
        Inter,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background: #f4f7fb;
    color: #172033;
}}

body {{
    margin: 0;
    background: #f4f7fb;
}}

.container {{
    width: min(1180px, 94%);
    margin: 40px auto;
}}

.hero {{
    background: white;
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow:
        0 12px 35px rgba(20, 30, 50, 0.08);
}}

.hero h1 {{
    margin: 0 0 10px;
}}

.hero p {{
    color: #637083;
}}

.summary {{
    padding: 20px;
    border-radius: 16px;
    background: #f0f4fa;
    margin-top: 20px;
}}

.metrics {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 28px;
}}

.metric {{
    background: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow:
        0 8px 24px rgba(20, 30, 50, 0.06);
}}

.metric span {{
    display: block;
    font-size: 13px;
    color: #6b7788;
    margin-bottom: 8px;
}}

.metric strong {{
    font-size: 20px;
}}

.section {{
    margin-top: 34px;
}}

.cards {{
    display: grid;
    gap: 16px;
}}

.card {{
    background: white;
    border-radius: 18px;
    padding: 22px;
    box-shadow:
        0 8px 24px rgba(20, 30, 50, 0.06);
}}

.card h3 {{
    margin-top: 12px;
}}

.card p {{
    color: #566275;
    line-height: 1.6;
}}

.badge {{
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: #e9eef7;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .06em;
}}

.meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    margin: 15px 0;
    color: #657286;
    font-size: 13px;
}}

.meta strong {{
    color: #172033;
}}

.warning {{
    background: #fff7e6;
    padding: 20px;
    border-radius: 16px;
}}

.footer {{
    margin-top: 40px;
    color: #7a8494;
    font-size: 13px;
}}

</style>
</head>

<body>

<div class="container">

<section class="hero">

    <h1>FinCo AI Financial Analysis</h1>

    <p>
        Specialized financial-analysis agent output.
    </p>

    <div class="summary">
        <strong>Executive Summary</strong>
        <p>{escape(result.summary)}</p>
    </div>

</section>

<section class="metrics">

{''.join(metric_items)}

</section>

<section class="section">

<h2>Financial Findings</h2>

<div class="cards">

{
    ''.join(finding_cards)
    or "<p>No material findings identified.</p>"
}

</div>

</section>

<section class="section">

<h2>Recommendations</h2>

<div class="cards">

{
    ''.join(recommendation_cards)
    or "<p>No recommendations generated.</p>"
}

</div>

</section>

{
    f'''
    <section class="section">
        <h2>Warnings</h2>
        <div class="warning">
            <ul>{warning_html}</ul>
        </div>
    </section>
    '''
    if result.warnings
    else ""
}

<footer class="footer">

    Result ID:
    {escape(result.result_id)}
    <br>

    Generated:
    {escape(result.generated_at)}

</footer>

</div>

</body>
</html>
"""


def save_json(
    result: FinancialAgentResult,
    path: str | Path,
) -> Path:
    """Save result as JSON."""

    output = Path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        result.to_json(),
        encoding="utf-8",
    )

    return output


def save_markdown(
    result: FinancialAgentResult,
    path: str | Path,
) -> Path:
    """Save result as Markdown."""

    output = Path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        to_markdown(result),
        encoding="utf-8",
    )

    return output


def save_html(
    result: FinancialAgentResult,
    path: str | Path,
) -> Path:
    """Save result as HTML."""

    output = Path(path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        to_html(result),
        encoding="utf-8",
    )

    return output


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_financial_agent(
    *,
    task: str,
    company_id: Optional[str] = None,
    financial_data: Optional[
        Mapping[str, Any]
    ] = None,
    rag_results: Optional[
        Sequence[Any]
    ] = None,
    risk_data: Optional[
        Mapping[str, Any]
    ] = None,
    forecast_data: Optional[
        Mapping[str, Any]
    ] = None,
    alerts: Optional[
        Sequence[Any]
    ] = None,
    config: Optional[
        FinancialAgentConfig
    ] = None,
) -> FinancialAgentResult:
    """Convenience wrapper."""

    agent = FinancialAgent(
        config=config
    )

    return agent.run(
        task=task,
        company_id=company_id,
        financial_data=financial_data,
        rag_results=rag_results,
        risk_data=risk_data,
        forecast_data=forecast_data,
        alerts=alerts,
    )


# ============================================================================
# DEMO
# ============================================================================

def create_demo_financial_data() -> Dict[str, Any]:
    """Create realistic demo data."""

    return {
        "revenue": 5_200_000,
        "previous_revenue": 6_100_000,

        "expenses": 4_900_000,
        "previous_expenses": 4_100_000,

        "gross_profit": 780_000,
        "operating_income": 190_000,
        "net_income": 120_000,

        "cash": 420_000,

        "current_assets": 920_000,
        "current_liabilities": 1_000_000,

        "inventory": 310_000,
        "receivables": 540_000,
        "payables": 350_000,

        "total_debt": 2_900_000,
        "total_equity": 800_000,

        "operating_cash_flow": -120_000,
        "free_cash_flow": -240_000,

        "interest_expense": 180_000,

        "budget": 4_300_000,
        "actual_expenses": 4_900_000,
    }


def create_demo_rag_results() -> List[Dict[str, Any]]:
    """Create demo document evidence."""

    return [
        {
            "chunk_id": "chunk-001",
            "document_id": "annual-report-2025",
            "content": (
                "Revenue declined during the reporting period "
                "primarily due to lower demand in selected "
                "business segments."
            ),
            "score": 0.94,
            "source": "annual_report.pdf",
            "page": 12,
            "citation": (
                "annual_report.pdf, page 12"
            ),
        },
        {
            "chunk_id": "chunk-002",
            "document_id": "annual-report-2025",
            "content": (
                "Operating expenses increased due to "
                "higher administrative and technology costs."
            ),
            "score": 0.88,
            "source": "annual_report.pdf",
            "page": 18,
            "citation": (
                "annual_report.pdf, page 18"
            ),
        },
    ]


def create_demo_result() -> FinancialAgentResult:
    """Run demo FinancialAgent."""

    agent = FinancialAgent()

    return agent.run(
        task=(
            "Analyze the company's financial performance, "
            "identify the major risks, and recommend actions "
            "to improve profitability and liquidity."
        ),
        company_id="company-demo-001",
        financial_data=(
            create_demo_financial_data()
        ),
        rag_results=(
            create_demo_rag_results()
        ),
        risk_data={
            "risk_score": 0.84,
            "risk_level": "high",
        },
        forecast_data={
            "downside_risk": 0.61,
        },
        alerts=[
            {
                "alert_id": "ALT-001",
                "title": "Cash flow deterioration",
                "description": (
                    "Operating cash flow is negative."
                ),
                "severity": "high",
                "risk_score": 0.89,
            }
        ],
    )


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """CLI demonstration."""

    parser = argparse.ArgumentParser(
        description=(
            "FinCo AI Financial Agent"
        )
    )

    parser.add_argument(
        "--output",
        default="reports/financial_agent",
        help=(
            "Output directory."
        ),
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
    )

    args = parser.parse_args()

    print()
    print("=" * 78)
    print("FinCo AI - Financial Agent")
    print("=" * 78)

    result = create_demo_result()

    output_dir = Path(
        args.output
    )

    if args.format in {
        "json",
        "all",
    }:

        path = save_json(
            result,
            output_dir
            / "financial_agent_result.json",
        )

        print(
            f"JSON:     {path}"
        )

    if args.format in {
        "markdown",
        "all",
    }:

        path = save_markdown(
            result,
            output_dir
            / "financial_agent_report.md",
        )

        print(
            f"Markdown: {path}"
        )

    if args.format in {
        "html",
        "all",
    }:

        path = save_html(
            result,
            output_dir
            / "financial_agent_report.html",
        )

        print(
            f"HTML:     {path}"
        )

    print()
    print(
        f"Result ID:          {result.result_id}"
    )
    print(
        f"Status:             {result.status}"
    )
    print(
        f"Intent:             {result.intent}"
    )
    print(
        f"Findings:           {len(result.findings)}"
    )
    print(
        f"Recommendations:    {len(result.recommendations)}"
    )
    print(
        f"Evidence:           {len(result.evidence)}"
    )
    print(
        f"Confidence:         {result.confidence:.1%}"
    )
    print(
        "Human Review:       "
        f"{result.human_review_required}"
    )

    print()
    print("EXECUTIVE SUMMARY")
    print("-" * 78)
    print(
        result.summary
    )

    print()
    print("=" * 78)
    print(
        "Financial analysis complete."
    )
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()