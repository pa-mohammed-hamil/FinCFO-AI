"""
FinCo AI - Executive Report Generator

Path:
    backend/app/reports/executive_report.py

Purpose:
    Generate executive-level financial intelligence reports containing:

    - Executive summary
    - Financial KPIs
    - Revenue / expense / profit analysis
    - Financial health score
    - Risk alerts
    - Fraud/anomaly summary
    - Forecast outlook
    - Key business drivers
    - Recommendations
    - What-if insights
    - Management action items
    - Audit metadata

The module is intentionally framework-light so it can be used from:
    - FastAPI routes
    - Agent workflows
    - CLI scripts
    - Scheduled jobs
    - Tests
"""

from __future__ import annotations

import json
import math
import os
import statistics
import uuid

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_VERSION = "1.0.0"

DEFAULT_REPORT_DIR = Path(
    os.getenv(
        "FINCO_REPORT_DIR",
        str(Path(__file__).resolve().parents[3] / "reports"),
    )
)

DEFAULT_CURRENCY = os.getenv("FINCO_CURRENCY", "USD")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without raising exceptions."""
    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 2) -> float:
    """Safely round a numeric value."""
    return round(_safe_float(value), digits)


def _percentage(value: Any, digits: int = 2) -> float:
    """Convert a ratio to percentage."""
    return round(_safe_float(value) * 100, digits)


def _safe_divide(
    numerator: Any,
    denominator: Any,
    default: float = 0.0,
) -> float:
    """Safe division."""
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)

    if denominator == 0:
        return default

    return numerator / denominator


def _growth_rate(current: Any, previous: Any) -> float:
    """Return percentage growth rate."""
    current = _safe_float(current)
    previous = _safe_float(previous)

    if previous == 0:
        return 0.0

    return ((current - previous) / abs(previous)) * 100


def _change_direction(value: float) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return "flat"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _currency(value: Any, currency: str = DEFAULT_CURRENCY) -> str:
    """Human-readable currency formatting."""
    amount = _safe_float(value)

    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "INR": "₹",
        "CAD": "C$",
        "AUD": "A$",
    }

    symbol = symbols.get(currency.upper(), currency.upper() + " ")

    if abs(amount) >= 1_000_000_000:
        return f"{symbol}{amount / 1_000_000_000:.2f}B"

    if abs(amount) >= 1_000_000:
        return f"{symbol}{amount / 1_000_000:.2f}M"

    if abs(amount) >= 1_000:
        return f"{symbol}{amount / 1_000:.2f}K"

    return f"{symbol}{amount:,.2f}"


def _severity_rank(severity: Any) -> int:
    ranks = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
        "info": 0,
    }

    return ranks.get(
        _normalize_text(severity).lower(),
        0,
    )


def _status_from_score(score: float) -> str:
    if score >= 85:
        return "Excellent"

    if score >= 70:
        return "Healthy"

    if score >= 50:
        return "Watch"

    if score >= 30:
        return "At Risk"

    return "Critical"


def _ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return list(value)

    return [value]


def _get(item: Any, key: str, default: Any = None) -> Any:
    """Read from dict-like objects or regular objects."""
    if isinstance(item, Mapping):
        return item.get(key, default)

    return getattr(item, key, default)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ExecutiveKPI:
    name: str
    value: float
    formatted_value: str
    unit: str = ""
    previous_value: Optional[float] = None
    change_percent: Optional[float] = None
    direction: str = "flat"
    status: str = "neutral"
    description: str = ""


@dataclass
class FinancialInsight:
    title: str
    category: str
    severity: str
    message: str
    metric: Optional[str] = None
    value: Optional[float] = None
    impact: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ExecutiveReportInput:
    company_name: str = "FinCo Demo Company"
    company_id: Optional[str] = None
    reporting_period: str = "Current Period"
    currency: str = DEFAULT_CURRENCY

    revenue: float = 0.0
    previous_revenue: float = 0.0

    expenses: float = 0.0
    previous_expenses: float = 0.0

    profit: float = 0.0
    previous_profit: float = 0.0

    cash_flow: float = 0.0
    previous_cash_flow: float = 0.0

    assets: float = 0.0
    liabilities: float = 0.0
    equity: float = 0.0

    budget: float = 0.0
    previous_budget: float = 0.0

    current_ratio: float = 0.0
    debt_to_equity: float = 0.0
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    net_margin: float = 0.0

    financial_health_score: Optional[float] = None

    alerts: List[Any] = field(default_factory=list)
    fraud_alerts: List[Any] = field(default_factory=list)
    forecasts: List[Any] = field(default_factory=list)
    recommendations: List[Any] = field(default_factory=list)
    what_if_results: List[Any] = field(default_factory=list)
    root_causes: List[Any] = field(default_factory=list)

    top_revenue_drivers: List[Any] = field(default_factory=list)
    top_cost_drivers: List[Any] = field(default_factory=list)

    period_start: Optional[str] = None
    period_end: Optional[str] = None

    generated_by: str = "FinCo AI"
    user_id: Optional[str] = None


@dataclass
class ExecutiveReport:
    report_id: str
    version: str
    generated_at: str

    company_name: str
    company_id: Optional[str]

    reporting_period: str
    period_start: Optional[str]
    period_end: Optional[str]

    currency: str

    executive_summary: str

    financial_status: str
    financial_health_score: float

    kpis: List[ExecutiveKPI]
    insights: List[FinancialInsight]

    risk_summary: Dict[str, Any]
    fraud_summary: Dict[str, Any]
    forecast_summary: Dict[str, Any]

    recommendations: List[Dict[str, Any]]
    what_if_summary: List[Dict[str, Any]]

    action_items: List[Dict[str, Any]]

    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Executive report generator
# ---------------------------------------------------------------------------


class ExecutiveReportGenerator:
    """
    Main report-generation service.

    Example:

        generator = ExecutiveReportGenerator()

        report = generator.generate(
            ExecutiveReportInput(
                company_name="Acme Corp",
                revenue=10_000_000,
                previous_revenue=9_000_000,
                expenses=7_500_000,
                previous_expenses=7_000_000,
                profit=2_500_000,
                previous_profit=2_000_000,
            )
        )

        generator.save_json(report)
    """

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        self.output_dir = Path(output_dir or DEFAULT_REPORT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        data: ExecutiveReportInput | Mapping[str, Any],
    ) -> ExecutiveReport:
        """Generate a complete executive report."""

        data = self._normalize_input(data)

        health_score = self._calculate_health_score(data)

        kpis = self._build_kpis(data)

        insights = self._build_insights(
            data=data,
            health_score=health_score,
        )

        risk_summary = self._build_risk_summary(data.alerts)

        fraud_summary = self._build_fraud_summary(
            data.fraud_alerts
        )

        forecast_summary = self._build_forecast_summary(
            data.forecasts,
            data.currency,
        )

        recommendations = self._build_recommendations(
            data.recommendations
        )

        what_if_summary = self._build_what_if_summary(
            data.what_if_results,
            data.currency,
        )

        action_items = self._build_action_items(
            data=data,
            insights=insights,
            risk_summary=risk_summary,
            fraud_summary=fraud_summary,
        )

        executive_summary = self._generate_executive_summary(
            data=data,
            health_score=health_score,
            insights=insights,
            risk_summary=risk_summary,
            fraud_summary=fraud_summary,
            forecast_summary=forecast_summary,
        )

        report = ExecutiveReport(
            report_id=f"FINCO-EXEC-{uuid.uuid4().hex[:12].upper()}",
            version=REPORT_VERSION,
            generated_at=_now_iso(),
            company_name=data.company_name,
            company_id=data.company_id,
            reporting_period=data.reporting_period,
            period_start=data.period_start,
            period_end=data.period_end,
            currency=data.currency,
            executive_summary=executive_summary,
            financial_status=_status_from_score(health_score),
            financial_health_score=health_score,
            kpis=kpis,
            insights=insights,
            risk_summary=risk_summary,
            fraud_summary=fraud_summary,
            forecast_summary=forecast_summary,
            recommendations=recommendations,
            what_if_summary=what_if_summary,
            action_items=action_items,
            metadata={
                "generated_by": data.generated_by,
                "user_id": data.user_id,
                "report_type": "executive",
                "report_version": REPORT_VERSION,
                "generator": self.__class__.__name__,
            },
        )

        return report

    def to_dict(
        self,
        report: ExecutiveReport,
    ) -> Dict[str, Any]:
        """Convert report to JSON-compatible dictionary."""
        return asdict(report)

    def to_json(
        self,
        report: ExecutiveReport,
        indent: int = 2,
    ) -> str:
        """Serialize report to JSON."""
        return json.dumps(
            self.to_dict(report),
            indent=indent,
            ensure_ascii=False,
            default=str,
        )

    def save_json(
        self,
        report: ExecutiveReport,
        filename: Optional[str] = None,
    ) -> Path:
        """Save executive report as JSON."""

        if filename is None:
            filename = (
                f"{report.report_id.lower()}.json"
            )

        path = self.output_dir / filename

        path.write_text(
            self.to_json(report),
            encoding="utf-8",
        )

        return path

    def generate_and_save_json(
        self,
        data: ExecutiveReportInput | Mapping[str, Any],
        filename: Optional[str] = None,
    ) -> tuple[ExecutiveReport, Path]:
        """Generate and save JSON report."""

        report = self.generate(data)

        path = self.save_json(
            report,
            filename=filename,
        )

        return report, path

    def to_markdown(
        self,
        report: ExecutiveReport,
    ) -> str:
        """Create executive-friendly Markdown."""

        lines: List[str] = []

        lines.append(
            f"# {report.company_name} — Executive Financial Report"
        )

        lines.append("")

        lines.append(
            f"**Reporting Period:** {report.reporting_period}"
        )

        lines.append(
            f"**Report ID:** {report.report_id}"
        )

        lines.append(
            f"**Generated:** {report.generated_at}"
        )

        lines.append("")

        lines.append("## Executive Summary")
        lines.append("")
        lines.append(report.executive_summary)
        lines.append("")

        lines.append("## Financial Health")
        lines.append("")

        lines.append(
            f"- **Score:** "
            f"{report.financial_health_score:.1f}/100"
        )

        lines.append(
            f"- **Status:** {report.financial_status}"
        )

        lines.append("")

        lines.append("## Key Financial KPIs")
        lines.append("")

        lines.append(
            "| KPI | Value | Change | Direction | Status |"
        )

        lines.append(
            "|---|---:|---:|---|---|"
        )

        for kpi in report.kpis:
            change = (
                f"{kpi.change_percent:.2f}%"
                if kpi.change_percent is not None
                else "—"
            )

            lines.append(
                f"| {kpi.name} | "
                f"{kpi.formatted_value} | "
                f"{change} | "
                f"{kpi.direction} | "
                f"{kpi.status} |"
            )

        lines.append("")

        lines.append("## Key Insights")
        lines.append("")

        if report.insights:
            for insight in report.insights:
                lines.append(
                    f"- **[{insight.severity.upper()}] "
                    f"{insight.title}:** "
                    f"{insight.message}"
                )
        else:
            lines.append(
                "No material financial insights were detected."
            )

        lines.append("")

        lines.append("## Risk Summary")
        lines.append("")

        for key, value in report.risk_summary.items():
            lines.append(
                f"- **{key.replace('_', ' ').title()}:** {value}"
            )

        lines.append("")

        lines.append("## Fraud / Anomaly Summary")
        lines.append("")

        for key, value in report.fraud_summary.items():
            lines.append(
                f"- **{key.replace('_', ' ').title()}:** {value}"
            )

        lines.append("")

        lines.append("## Forecast Outlook")
        lines.append("")

        for key, value in report.forecast_summary.items():
            if isinstance(value, dict):
                lines.append(
                    f"- **{key.replace('_', ' ').title()}:**"
                )

                for nested_key, nested_value in value.items():
                    lines.append(
                        f"  - "
                        f"{nested_key.replace('_', ' ').title()}: "
                        f"{nested_value}"
                    )
            else:
                lines.append(
                    f"- **{key.replace('_', ' ').title()}:** "
                    f"{value}"
                )

        lines.append("")

        lines.append("## Recommendations")
        lines.append("")

        if report.recommendations:
            for index, recommendation in enumerate(
                report.recommendations,
                start=1,
            ):
                title = recommendation.get(
                    "title",
                    f"Recommendation {index}",
                )

                action = recommendation.get(
                    "action",
                    recommendation.get("description", ""),
                )

                priority = recommendation.get(
                    "priority",
                    "medium",
                )

                lines.append(
                    f"{index}. **{title}** "
                    f"({str(priority).upper()}) — {action}"
                )
        else:
            lines.append(
                "No recommendations were supplied."
            )

        lines.append("")

        lines.append("## Management Action Items")
        lines.append("")

        if report.action_items:
            for item in report.action_items:
                lines.append(
                    f"- **{item.get('priority', 'MEDIUM').upper()}** "
                    f"{item.get('action', '')}"
                )
        else:
            lines.append(
                "No immediate action items."
            )

        lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(
            "*Generated by FinCo AI — Financial Intelligence "
            "& Decision Copilot*"
        )

        return "\n".join(lines)

    def save_markdown(
        self,
        report: ExecutiveReport,
        filename: Optional[str] = None,
    ) -> Path:
        """Save report as Markdown."""

        if filename is None:
            filename = (
                f"{report.report_id.lower()}.md"
            )

        path = self.output_dir / filename

        path.write_text(
            self.to_markdown(report),
            encoding="utf-8",
        )

        return path

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    def _normalize_input(
        self,
        data: ExecutiveReportInput | Mapping[str, Any],
    ) -> ExecutiveReportInput:

        if isinstance(data, ExecutiveReportInput):
            return data

        if not isinstance(data, Mapping):
            raise TypeError(
                "data must be ExecutiveReportInput or mapping"
            )

        fields = {
            field_name
            for field_name in ExecutiveReportInput.__dataclass_fields__
        }

        cleaned = {
            key: value
            for key, value in data.items()
            if key in fields
        }

        return ExecutiveReportInput(**cleaned)

    # ------------------------------------------------------------------
    # Health score
    # ------------------------------------------------------------------

    def _calculate_health_score(
        self,
        data: ExecutiveReportInput,
    ) -> float:

        if data.financial_health_score is not None:
            return max(
                0.0,
                min(
                    100.0,
                    _safe_float(
                        data.financial_health_score
                    ),
                ),
            )

        scores: List[float] = []

        # Profitability
        if data.revenue != 0:
            net_margin = (
                data.profit / data.revenue
            ) * 100

            profitability_score = max(
                0,
                min(
                    100,
                    50 + net_margin * 5,
                ),
            )

            scores.append(profitability_score)

        # Revenue growth
        revenue_growth = _growth_rate(
            data.revenue,
            data.previous_revenue,
        )

        revenue_score = max(
            0,
            min(
                100,
                60 + revenue_growth * 3,
            ),
        )

        scores.append(revenue_score)

        # Expense control
        expense_growth = _growth_rate(
            data.expenses,
            data.previous_expenses,
        )

        expense_score = max(
            0,
            min(
                100,
                70 - max(expense_growth, 0) * 3,
            ),
        )

        scores.append(expense_score)

        # Liquidity
        if data.current_ratio > 0:
            liquidity_score = max(
                0,
                min(
                    100,
                    data.current_ratio * 40,
                ),
            )

            scores.append(liquidity_score)

        # Leverage
        if data.debt_to_equity > 0:
            leverage_score = max(
                0,
                min(
                    100,
                    100 - data.debt_to_equity * 30,
                ),
            )

            scores.append(leverage_score)

        # Cash flow
        if data.cash_flow != 0:
            cash_score = (
                80 if data.cash_flow > 0 else 30
            )

            scores.append(cash_score)

        if not scores:
            return 50.0

        return round(
            max(
                0,
                min(
                    100,
                    statistics.mean(scores),
                ),
            ),
            2,
        )

    # ------------------------------------------------------------------
    # KPI generation
    # ------------------------------------------------------------------

    def _build_kpis(
        self,
        data: ExecutiveReportInput,
    ) -> List[ExecutiveKPI]:

        kpis: List[ExecutiveKPI] = []

        def add_kpi(
            name: str,
            value: float,
            previous: Optional[float],
            description: str,
            unit: str = "",
        ) -> None:

            change = (
                _growth_rate(value, previous)
                if previous is not None
                else None
            )

            direction = (
                _change_direction(change)
                if change is not None
                else "flat"
            )

            if name.lower() in {
                "expenses",
                "debt",
            }:
                status = (
                    "warning"
                    if change is not None and change > 10
                    else "healthy"
                )
            else:
                status = (
                    "healthy"
                    if change is None or change >= 0
                    else "warning"
                )

            formatted = (
                f"{value:.2f}{unit}"
                if unit
                else _currency(
                    value,
                    data.currency,
                )
            )

            kpis.append(
                ExecutiveKPI(
                    name=name,
                    value=_round(value),
                    formatted_value=formatted,
                    unit=unit,
                    previous_value=(
                        _round(previous)
                        if previous is not None
                        else None
                    ),
                    change_percent=(
                        _round(change)
                        if change is not None
                        else None
                    ),
                    direction=direction,
                    status=status,
                    description=description,
                )
            )

        add_kpi(
            "Revenue",
            data.revenue,
            data.previous_revenue,
            "Total revenue generated during the reporting period.",
        )

        add_kpi(
            "Expenses",
            data.expenses,
            data.previous_expenses,
            "Total operating and non-operating expenses.",
        )

        add_kpi(
            "Profit",
            data.profit,
            data.previous_profit,
            "Net profit generated during the reporting period.",
        )

        add_kpi(
            "Cash Flow",
            data.cash_flow,
            data.previous_cash_flow,
            "Net cash generated or consumed.",
        )

        add_kpi(
            "Gross Margin",
            data.gross_margin,
            None,
            "Gross margin percentage.",
            unit="%",
        )

        add_kpi(
            "Operating Margin",
            data.operating_margin,
            None,
            "Operating margin percentage.",
            unit="%",
        )

        add_kpi(
            "Net Margin",
            data.net_margin,
            None,
            "Net profit margin percentage.",
            unit="%",
        )

        add_kpi(
            "Current Ratio",
            data.current_ratio,
            None,
            "Short-term liquidity ratio.",
            unit="x",
        )

        add_kpi(
            "Debt to Equity",
            data.debt_to_equity,
            None,
            "Leverage relative to shareholder equity.",
            unit="x",
        )

        return kpis

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def _build_insights(
        self,
        data: ExecutiveReportInput,
        health_score: float,
    ) -> List[FinancialInsight]:

        insights: List[FinancialInsight] = []

        revenue_growth = _growth_rate(
            data.revenue,
            data.previous_revenue,
        )

        expense_growth = _growth_rate(
            data.expenses,
            data.previous_expenses,
        )

        profit_growth = _growth_rate(
            data.profit,
            data.previous_profit,
        )

        # Revenue
        if revenue_growth >= 10:
            insights.append(
                FinancialInsight(
                    title="Strong Revenue Growth",
                    category="revenue",
                    severity="positive",
                    message=(
                        f"Revenue increased by "
                        f"{revenue_growth:.1f}% compared with "
                        "the previous period."
                    ),
                    metric="revenue_growth",
                    value=revenue_growth,
                    impact="positive",
                )
            )

        elif revenue_growth <= -10:
            insights.append(
                FinancialInsight(
                    title="Revenue Decline",
                    category="revenue",
                    severity="high",
                    message=(
                        f"Revenue declined by "
                        f"{abs(revenue_growth):.1f}% "
                        "versus the previous period."
                    ),
                    metric="revenue_growth",
                    value=revenue_growth,
                    impact="negative",
                )
            )

        # Expenses
        if expense_growth > revenue_growth + 5:
            insights.append(
                FinancialInsight(
                    title="Expense Growth Outpacing Revenue",
                    category="cost",
                    severity="high",
                    message=(
                        f"Expenses grew {expense_growth:.1f}% "
                        f"while revenue grew "
                        f"{revenue_growth:.1f}%."
                    ),
                    metric="expense_growth",
                    value=expense_growth,
                    impact="negative",
                )
            )

        # Profit
        if data.profit < 0:
            insights.append(
                FinancialInsight(
                    title="Net Loss Detected",
                    category="profitability",
                    severity="critical",
                    message=(
                        "The company reported a negative "
                        "net profit for the reporting period."
                    ),
                    metric="profit",
                    value=data.profit,
                    impact="negative",
                )
            )

        elif profit_growth >= 10:
            insights.append(
                FinancialInsight(
                    title="Profitability Improvement",
                    category="profitability",
                    severity="positive",
                    message=(
                        f"Profit increased by "
                        f"{profit_growth:.1f}% "
                        "compared with the previous period."
                    ),
                    metric="profit_growth",
                    value=profit_growth,
                    impact="positive",
                )
            )

        # Cash flow
        if data.cash_flow < 0:
            insights.append(
                FinancialInsight(
                    title="Negative Cash Flow",
                    category="cash_flow",
                    severity="high",
                    message=(
                        "Net cash flow is negative and "
                        "requires management attention."
                    ),
                    metric="cash_flow",
                    value=data.cash_flow,
                    impact="negative",
                )
            )

        # Liquidity
        if (
            data.current_ratio > 0
            and data.current_ratio < 1
        ):
            insights.append(
                FinancialInsight(
                    title="Liquidity Risk",
                    category="liquidity",
                    severity="high",
                    message=(
                        f"Current ratio is "
                        f"{data.current_ratio:.2f}x, "
                        "indicating potential short-term "
                        "liquidity pressure."
                    ),
                    metric="current_ratio",
                    value=data.current_ratio,
                    impact="negative",
                )
            )

        # Leverage
        if data.debt_to_equity >= 2:
            insights.append(
                FinancialInsight(
                    title="Elevated Leverage",
                    category="risk",
                    severity="high",
                    message=(
                        f"Debt-to-equity is "
                        f"{data.debt_to_equity:.2f}x, "
                        "indicating elevated leverage."
                    ),
                    metric="debt_to_equity",
                    value=data.debt_to_equity,
                    impact="negative",
                )
            )

        # Budget
        if data.budget:
            variance = (
                data.expenses - data.budget
            )

            variance_percent = (
                _safe_divide(
                    variance,
                    abs(data.budget),
                )
                * 100
            )

            if variance_percent > 10:
                insights.append(
                    FinancialInsight(
                        title="Budget Overrun",
                        category="budget",
                        severity="high",
                        message=(
                            f"Expenses exceed budget by "
                            f"{variance_percent:.1f}%."
                        ),
                        metric="budget_variance",
                        value=variance_percent,
                        impact="negative",
                    )
                )

        # Overall health
        if health_score < 40:
            insights.append(
                FinancialInsight(
                    title="Critical Financial Health",
                    category="financial_health",
                    severity="critical",
                    message=(
                        f"Overall financial health score "
                        f"is {health_score:.1f}/100."
                    ),
                    metric="financial_health_score",
                    value=health_score,
                    impact="negative",
                )
            )

        return insights

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    def _build_risk_summary(
        self,
        alerts: Iterable[Any],
    ) -> Dict[str, Any]:

        alerts = list(alerts)

        counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }

        for alert in alerts:
            severity = (
                _normalize_text(
                    _get(
                        alert,
                        "severity",
                        "info",
                    )
                ).lower()
            )

            if severity not in counts:
                severity = "info"

            counts[severity] += 1

        critical_high = (
            counts["critical"]
            + counts["high"]
        )

        sorted_alerts = sorted(
            alerts,
            key=lambda item: _severity_rank(
                _get(item, "severity", "info")
            ),
            reverse=True,
        )

        top_alerts = []

        for alert in sorted_alerts[:5]:
            top_alerts.append(
                {
                    "title": _get(
                        alert,
                        "title",
                        _get(
                            alert,
                            "name",
                            "Risk Alert",
                        ),
                    ),
                    "severity": _get(
                        alert,
                        "severity",
                        "info",
                    ),
                    "message": _get(
                        alert,
                        "message",
                        _get(
                            alert,
                            "description",
                            "",
                        ),
                    ),
                }
            )

        return {
            "total_alerts": len(alerts),
            "critical_alerts": counts["critical"],
            "high_alerts": counts["high"],
            "medium_alerts": counts["medium"],
            "low_alerts": counts["low"],
            "critical_high_alerts": critical_high,
            "risk_level": (
                "critical"
                if counts["critical"] > 0
                else "high"
                if counts["high"] > 0
                else "medium"
                if counts["medium"] > 0
                else "low"
            ),
            "top_alerts": top_alerts,
        }

    # ------------------------------------------------------------------
    # Fraud
    # ------------------------------------------------------------------

    def _build_fraud_summary(
        self,
        fraud_alerts: Iterable[Any],
    ) -> Dict[str, Any]:

        alerts = list(fraud_alerts)

        high_risk = 0
        medium_risk = 0
        low_risk = 0

        total_exposure = 0.0

        suspicious_transactions = 0

        for alert in alerts:
            severity = _normalize_text(
                _get(
                    alert,
                    "severity",
                    _get(
                        alert,
                        "risk_level",
                        "low",
                    ),
                )
            ).lower()

            if severity in {"critical", "high"}:
                high_risk += 1
            elif severity == "medium":
                medium_risk += 1
            else:
                low_risk += 1

            total_exposure += _safe_float(
                _get(
                    alert,
                    "amount",
                    _get(
                        alert,
                        "exposure",
                        0,
                    ),
                )
            )

            suspicious_transactions += 1

        return {
            "total_fraud_alerts": len(alerts),
            "high_risk_alerts": high_risk,
            "medium_risk_alerts": medium_risk,
            "low_risk_alerts": low_risk,
            "suspicious_transactions": suspicious_transactions,
            "estimated_exposure": _round(
                total_exposure
            ),
            "fraud_risk_level": (
                "high"
                if high_risk > 0
                else "medium"
                if medium_risk > 0
                else "low"
            ),
        }

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def _build_forecast_summary(
        self,
        forecasts: Iterable[Any],
        currency: str,
    ) -> Dict[str, Any]:

        forecasts = list(forecasts)

        if not forecasts:
            return {
                "forecast_available": False,
                "forecast_count": 0,
                "outlook": "Not available",
                "message": (
                    "No forecast results were supplied."
                ),
            }

        positive = 0
        negative = 0
        neutral = 0

        forecast_values: List[float] = []

        by_metric: Dict[str, Dict[str, Any]] = {}

        for forecast in forecasts:
            metric = _normalize_text(
                _get(
                    forecast,
                    "metric",
                    _get(
                        forecast,
                        "name",
                        "forecast",
                    ),
                )
            )

            value = _safe_float(
                _get(
                    forecast,
                    "predicted_value",
                    _get(
                        forecast,
                        "forecast",
                        _get(
                            forecast,
                            "value",
                            0,
                        ),
                    ),
                )
            )

            direction = _normalize_text(
                _get(
                    forecast,
                    "direction",
                    "",
                )
            ).lower()

            if not direction:
                if value > 0:
                    direction = "positive"
                elif value < 0:
                    direction = "negative"
                else:
                    direction = "neutral"

            if direction in {
                "positive",
                "up",
                "increase",
                "growth",
            }:
                positive += 1
            elif direction in {
                "negative",
                "down",
                "decrease",
                "decline",
            }:
                negative += 1
            else:
                neutral += 1

            forecast_values.append(value)

            by_metric[metric] = {
                "predicted_value": _round(value),
                "formatted_value": _currency(
                    value,
                    currency,
                ),
                "direction": direction,
            }

        if positive > negative:
            outlook = "Positive"
        elif negative > positive:
            outlook = "Negative"
        else:
            outlook = "Mixed"

        return {
            "forecast_available": True,
            "forecast_count": len(forecasts),
            "positive_forecasts": positive,
            "negative_forecasts": negative,
            "neutral_forecasts": neutral,
            "outlook": outlook,
            "average_forecast_value": _round(
                statistics.mean(forecast_values)
            )
            if forecast_values
            else 0.0,
            "metrics": by_metric,
        }

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _build_recommendations(
        self,
        recommendations: Iterable[Any],
    ) -> List[Dict[str, Any]]:

        recommendations = list(recommendations)

        normalized: List[Dict[str, Any]] = []

        for recommendation in recommendations:

            normalized.append(
                {
                    "id": _get(
                        recommendation,
                        "id",
                        None,
                    ),
                    "title": _get(
                        recommendation,
                        "title",
                        "Financial Recommendation",
                    ),
                    "action": _get(
                        recommendation,
                        "action",
                        _get(
                            recommendation,
                            "description",
                            "",
                        ),
                    ),
                    "priority": _get(
                        recommendation,
                        "priority",
                        "medium",
                    ),
                    "category": _get(
                        recommendation,
                        "category",
                        "financial",
                    ),
                    "expected_impact": _get(
                        recommendation,
                        "expected_impact",
                        None,
                    ),
                    "confidence": _safe_float(
                        _get(
                            recommendation,
                            "confidence",
                            0,
                        )
                    ),
                }
            )

        priority_order = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        normalized.sort(
            key=lambda item: priority_order.get(
                str(
                    item["priority"]
                ).lower(),
                2,
            )
        )

        return normalized[:10]

    # ------------------------------------------------------------------
    # What-if
    # ------------------------------------------------------------------

    def _build_what_if_summary(
        self,
        scenarios: Iterable[Any],
        currency: str,
    ) -> List[Dict[str, Any]]:

        scenarios = list(scenarios)

        output = []

        for scenario in scenarios[:10]:

            baseline = _safe_float(
                _get(
                    scenario,
                    "baseline",
                    _get(
                        scenario,
                        "baseline_profit",
                        0,
                    ),
                )
            )

            projected = _safe_float(
                _get(
                    scenario,
                    "projected",
                    _get(
                        scenario,
                        "projected_profit",
                        0,
                    ),
                )
            )

            impact = projected - baseline

            output.append(
                {
                    "name": _get(
                        scenario,
                        "name",
                        _get(
                            scenario,
                            "scenario",
                            "Scenario",
                        ),
                    ),
                    "baseline": _round(baseline),
                    "projected": _round(projected),
                    "impact": _round(impact),
                    "formatted_impact": _currency(
                        impact,
                        currency,
                    ),
                    "impact_percent": _round(
                        _safe_divide(
                            impact,
                            abs(baseline),
                        )
                        * 100
                    ),
                }
            )

        return output

    # ------------------------------------------------------------------
    # Action items
    # ------------------------------------------------------------------

    def _build_action_items(
        self,
        data: ExecutiveReportInput,
        insights: List[FinancialInsight],
        risk_summary: Dict[str, Any],
        fraud_summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:

        actions: List[Dict[str, Any]] = []

        critical_count = _safe_int(
            risk_summary.get(
                "critical_alerts",
                0,
            )
        )

        high_count = _safe_int(
            risk_summary.get(
                "high_alerts",
                0,
            )
        )

        if critical_count > 0:
            actions.append(
                {
                    "priority": "critical",
                    "action": (
                        "Review all critical financial risk alerts "
                        "and initiate management escalation."
                    ),
                    "owner": "Finance Leadership",
                    "source": "risk_engine",
                }
            )

        if high_count > 0:
            actions.append(
                {
                    "priority": "high",
                    "action": (
                        "Investigate high-severity financial "
                        "alerts and document remediation plans."
                    ),
                    "owner": "Finance / Risk Team",
                    "source": "risk_engine",
                }
            )

        if fraud_summary["high_risk_alerts"] > 0:
            actions.append(
                {
                    "priority": "critical",
                    "action": (
                        "Review high-risk suspicious transactions "
                        "and verify potential fraud exposure."
                    ),
                    "owner": "Fraud / Compliance Team",
                    "source": "fraud_engine",
                }
            )

        if data.cash_flow < 0:
            actions.append(
                {
                    "priority": "high",
                    "action": (
                        "Review cash preservation measures, "
                        "working capital, and upcoming obligations."
                    ),
                    "owner": "Treasury / Finance",
                    "source": "financial_analysis",
                }
            )

        if data.current_ratio > 0 and data.current_ratio < 1:
            actions.append(
                {
                    "priority": "high",
                    "action": (
                        "Assess short-term liquidity and "
                        "accelerate receivables where possible."
                    ),
                    "owner": "Treasury",
                    "source": "liquidity_analysis",
                }
            )

        if data.debt_to_equity >= 2:
            actions.append(
                {
                    "priority": "medium",
                    "action": (
                        "Review debt structure and evaluate "
                        "leverage-reduction opportunities."
                    ),
                    "owner": "CFO / Treasury",
                    "source": "risk_analysis",
                }
            )

        if data.budget:
            variance = (
                data.expenses - data.budget
            )

            if variance > 0:
                actions.append(
                    {
                        "priority": (
                            "high"
                            if variance
                            / abs(data.budget)
                            > 0.10
                            else "medium"
                        ),
                        "action": (
                            "Investigate budget variance and "
                            "identify controllable cost drivers."
                        ),
                        "owner": "Finance / Operations",
                        "source": "budget_analysis",
                    }
                )

        # Include severe insights as actions.
        for insight in insights:
            if insight.severity in {
                "critical",
                "high",
            }:
                actions.append(
                    {
                        "priority": insight.severity,
                        "action": insight.message,
                        "owner": "Finance Leadership",
                        "source": insight.category,
                    }
                )

        # Deduplicate.
        unique = []
        seen = set()

        for action in actions:
            key = action["action"].strip().lower()

            if key in seen:
                continue

            seen.add(key)
            unique.append(action)

        priority_order = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        unique.sort(
            key=lambda item: priority_order.get(
                item["priority"],
                2,
            )
        )

        return unique[:12]

    # ------------------------------------------------------------------
    # Executive summary
    # ------------------------------------------------------------------

    def _generate_executive_summary(
        self,
        data: ExecutiveReportInput,
        health_score: float,
        insights: List[FinancialInsight],
        risk_summary: Dict[str, Any],
        fraud_summary: Dict[str, Any],
        forecast_summary: Dict[str, Any],
    ) -> str:

        revenue_growth = _growth_rate(
            data.revenue,
            data.previous_revenue,
        )

        profit_growth = _growth_rate(
            data.profit,
            data.previous_profit,
        )

        expense_growth = _growth_rate(
            data.expenses,
            data.previous_expenses,
        )

        status = _status_from_score(
            health_score
        )

        paragraphs: List[str] = []

        paragraphs.append(
            f"{data.company_name} is currently assessed as "
            f"{status.lower()} with a financial health score "
            f"of {health_score:.1f}/100."
        )

        paragraphs.append(
            f"Revenue is "
            f"{'up' if revenue_growth >= 0 else 'down'} "
            f"{abs(revenue_growth):.1f}% compared with the "
            "previous period, while expenses are "
            f"{'up' if expense_growth >= 0 else 'down'} "
            f"{abs(expense_growth):.1f}%."
        )

        if data.profit >= 0:
            paragraphs.append(
                f"Net profit is {_currency(data.profit, data.currency)} "
                f"with a period-over-period change of "
                f"{profit_growth:.1f}%."
            )
        else:
            paragraphs.append(
                f"The business recorded a net loss of "
                f"{_currency(abs(data.profit), data.currency)}, "
                "which requires management attention."
            )

        critical_high = (
            risk_summary["critical_high_alerts"]
        )

        if critical_high > 0:
            paragraphs.append(
                f"There are {critical_high} critical/high "
                "financial risk alerts requiring investigation."
            )

        if fraud_summary["total_fraud_alerts"] > 0:
            paragraphs.append(
                f"The fraud monitoring layer identified "
                f"{fraud_summary['total_fraud_alerts']} "
                "suspicious event(s), including "
                f"{fraud_summary['high_risk_alerts']} "
                "high-risk event(s)."
            )

        if forecast_summary.get(
            "forecast_available",
            False,
        ):
            paragraphs.append(
                f"The current forecast outlook is "
                f"{forecast_summary['outlook'].lower()}."
            )

        # Key severe insights.
        severe = [
            insight
            for insight in insights
            if insight.severity
            in {"critical", "high"}
        ]

        if severe:
            top = severe[:2]

            titles = ", ".join(
                insight.title
                for insight in top
            )

            paragraphs.append(
                f"Priority management areas include: {titles}."
            )

        return " ".join(paragraphs)

    # ------------------------------------------------------------------
    # PDF generation
    # ------------------------------------------------------------------

    def save_pdf(
        self,
        report: ExecutiveReport,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate a professional PDF report.

        Requires:
            reportlab

        The import is intentionally local so that JSON/Markdown
        functionality does not require ReportLab.
        """

        try:
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import (
                ParagraphStyle,
                getSampleStyleSheet,
            )
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as exc:
            raise RuntimeError(
                "ReportLab is required for PDF generation. "
                "Install it with: pip install reportlab"
            ) from exc

        if filename is None:
            filename = (
                f"{report.report_id.lower()}.pdf"
            )

        path = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            rightMargin=16 * mm,
            leftMargin=16 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title=(
                f"{report.company_name} "
                "Executive Financial Report"
            ),
            author="FinCo AI",
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "FinCoTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=20,
            leading=24,
            spaceAfter=8,
        )

        heading_style = ParagraphStyle(
            "FinCoHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            spaceBefore=10,
            spaceAfter=6,
        )

        body_style = ParagraphStyle(
            "FinCoBody",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=14,
            spaceAfter=6,
        )

        small_style = ParagraphStyle(
            "FinCoSmall",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
        )

        story = []

        # Header
        story.append(
            Paragraph(
                "FINCO AI",
                title_style,
            )
        )

        story.append(
            Paragraph(
                "Financial Intelligence & Decision Copilot",
                body_style,
            )
        )

        story.append(
            Paragraph(
                f"<b>{report.company_name}</b><br/>"
                f"Executive Financial Report<br/>"
                f"Reporting Period: "
                f"{report.reporting_period}<br/>"
                f"Report ID: {report.report_id}",
                body_style,
            )
        )

        story.append(Spacer(1, 5 * mm))

        # Health score
        story.append(
            Paragraph(
                "Executive Summary",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                report.executive_summary,
                body_style,
            )
        )

        health_table = Table(
            [
                [
                    "Financial Health Score",
                    f"{report.financial_health_score:.1f}/100",
                ],
                [
                    "Overall Status",
                    report.financial_status,
                ],
            ],
            colWidths=[
                80 * mm,
                80 * mm,
            ],
        )

        health_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.lightgrey,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, -1),
                        "Helvetica",
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (0, -1),
                        "Helvetica-Bold",
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                ]
            )
        )

        story.append(health_table)

        # KPIs
        story.append(
            Paragraph(
                "Key Financial KPIs",
                heading_style,
            )
        )

        kpi_rows = [
            [
                "KPI",
                "Value",
                "Change",
                "Status",
            ]
        ]

        for kpi in report.kpis:
            change = (
                f"{kpi.change_percent:.1f}%"
                if kpi.change_percent is not None
                else "—"
            )

            kpi_rows.append(
                [
                    kpi.name,
                    kpi.formatted_value,
                    change,
                    kpi.status,
                ]
            )

        kpi_table = Table(
            kpi_rows,
            repeatRows=1,
            colWidths=[
                45 * mm,
                40 * mm,
                35 * mm,
                30 * mm,
            ],
        )

        kpi_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(kpi_table)

        # Insights
        story.append(
            Paragraph(
                "Key Insights",
                heading_style,
            )
        )

        if report.insights:
            for insight in report.insights:
                story.append(
                    Paragraph(
                        f"<b>[{insight.severity.upper()}] "
                        f"{insight.title}</b><br/>"
                        f"{insight.message}",
                        body_style,
                    )
                )
        else:
            story.append(
                Paragraph(
                    "No material financial insights detected.",
                    body_style,
                )
            )

        # Risk
        story.append(
            Paragraph(
                "Risk Summary",
                heading_style,
            )
        )

        risk_rows = [
            [
                "Metric",
                "Value",
            ],
            [
                "Total Alerts",
                str(
                    report.risk_summary[
                        "total_alerts"
                    ]
                ),
            ],
            [
                "Critical Alerts",
                str(
                    report.risk_summary[
                        "critical_alerts"
                    ]
                ),
            ],
            [
                "High Alerts",
                str(
                    report.risk_summary[
                        "high_alerts"
                    ]
                ),
            ],
            [
                "Risk Level",
                str(
                    report.risk_summary[
                        "risk_level"
                    ]
                ).upper(),
            ],
        ]

        risk_table = Table(
            risk_rows,
            colWidths=[
                80 * mm,
                70 * mm,
            ],
        )

        risk_table.setStyle(
            TableStyle(
                [
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.grey,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.lightgrey,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                ]
            )
        )

        story.append(risk_table)

        # Fraud
        story.append(
            Paragraph(
                "Fraud & Anomaly Monitoring",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                f"Total suspicious events: "
                f"{report.fraud_summary['total_fraud_alerts']}<br/>"
                f"High-risk events: "
                f"{report.fraud_summary['high_risk_alerts']}<br/>"
                f"Estimated exposure: "
                f"{_currency(report.fraud_summary['estimated_exposure'], report.currency)}",
                body_style,
            )
        )

        # Forecast
        story.append(
            Paragraph(
                "Forecast Outlook",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                f"Forecast available: "
                f"{report.forecast_summary.get('forecast_available', False)}<br/>"
                f"Outlook: "
                f"{report.forecast_summary.get('outlook', 'N/A')}<br/>"
                f"Forecast count: "
                f"{report.forecast_summary.get('forecast_count', 0)}",
                body_style,
            )
        )

        # Recommendations
        story.append(
            Paragraph(
                "Recommendations",
                heading_style,
            )
        )

        for index, recommendation in enumerate(
            report.recommendations,
            start=1,
        ):
            story.append(
                Paragraph(
                    f"<b>{index}. "
                    f"{recommendation.get('title', 'Recommendation')}</b>"
                    f" "
                    f"({str(recommendation.get('priority', 'medium')).upper()})"
                    f"<br/>"
                    f"{recommendation.get('action', '')}",
                    body_style,
                )
            )

        # Action items
        story.append(
            Paragraph(
                "Management Action Items",
                heading_style,
            )
        )

        if report.action_items:

            action_rows = [
                [
                    "Priority",
                    "Action",
                    "Owner",
                ]
            ]

            for item in report.action_items:
                action_rows.append(
                    [
                        str(
                            item.get(
                                "priority",
                                "medium",
                            )
                        ).upper(),
                        item.get(
                            "action",
                            "",
                        ),
                        item.get(
                            "owner",
                            "",
                        ),
                    ]
                )

            action_table = Table(
                action_rows,
                repeatRows=1,
                colWidths=[
                    25 * mm,
                    105 * mm,
                    30 * mm,
                ],
            )

            action_table.setStyle(
                TableStyle(
                    [
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.4,
                            colors.grey,
                        ),
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            colors.lightgrey,
                        ),
                        (
                            "FONTNAME",
                            (0, 0),
                            (-1, 0),
                            "Helvetica-Bold",
                        ),
                        (
                            "FONTSIZE",
                            (0, 0),
                            (-1, -1),
                            7.5,
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP",
                        ),
                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            5,
                        ),
                    ]
                )
            )

            story.append(action_table)

        # Footer
        story.append(Spacer(1, 8 * mm))

        story.append(
            Paragraph(
                "This report is generated by FinCo AI and is "
                "intended to support financial analysis and "
                "decision-making. Recommendations should be "
                "reviewed and approved by authorized personnel.",
                small_style,
            )
        )

        doc.build(story)

        return path


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def generate_executive_report(
    data: ExecutiveReportInput | Mapping[str, Any],
) -> ExecutiveReport:
    """Simple functional API."""
    return ExecutiveReportGenerator().generate(data)


def generate_executive_report_json(
    data: ExecutiveReportInput | Mapping[str, Any],
    filename: Optional[str] = None,
) -> Path:
    """Generate and save an executive report as JSON."""

    generator = ExecutiveReportGenerator()

    report = generator.generate(data)

    return generator.save_json(
        report,
        filename=filename,
    )


def generate_executive_report_markdown(
    data: ExecutiveReportInput | Mapping[str, Any],
    filename: Optional[str] = None,
) -> Path:
    """Generate and save Markdown executive report."""

    generator = ExecutiveReportGenerator()

    report = generator.generate(data)

    return generator.save_markdown(
        report,
        filename=filename,
    )


def generate_executive_report_pdf(
    data: ExecutiveReportInput | Mapping[str, Any],
    filename: Optional[str] = None,
) -> Path:
    """Generate and save PDF executive report."""

    generator = ExecutiveReportGenerator()

    report = generator.generate(data)

    return generator.save_pdf(
        report,
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Demo data
# ---------------------------------------------------------------------------


def create_demo_input() -> ExecutiveReportInput:
    """Create realistic FinCo demo data."""

    return ExecutiveReportInput(
        company_name="FinCo Demo Corporation",
        company_id="COMP-DEMO-001",
        reporting_period="Q3 2026",
        currency="USD",

        revenue=12_500_000,
        previous_revenue=11_200_000,

        expenses=9_300_000,
        previous_expenses=8_200_000,

        profit=2_350_000,
        previous_profit=2_050_000,

        cash_flow=1_480_000,
        previous_cash_flow=1_150_000,

        assets=31_500_000,
        liabilities=14_800_000,
        equity=16_700_000,

        budget=8_800_000,

        current_ratio=1.42,
        debt_to_equity=0.89,

        gross_margin=38.4,
        operating_margin=22.7,
        net_margin=18.8,

        alerts=[
            {
                "title": "Operating Expense Increase",
                "severity": "medium",
                "message": (
                    "Operating expenses increased faster "
                    "than revenue."
                ),
            },
            {
                "title": "Margin Compression Risk",
                "severity": "high",
                "message": (
                    "Input costs may negatively affect "
                    "future margins."
                ),
            },
        ],

        fraud_alerts=[
            {
                "transaction_id": "TX-1042",
                "severity": "high",
                "amount": 175000,
            },
            {
                "transaction_id": "TX-1091",
                "severity": "medium",
                "amount": 42000,
            },
        ],

        forecasts=[
            {
                "metric": "Revenue",
                "predicted_value": 13_100_000,
                "direction": "positive",
            },
            {
                "metric": "Profit",
                "predicted_value": 2_520_000,
                "direction": "positive",
            },
            {
                "metric": "Cash Flow",
                "predicted_value": 1_300_000,
                "direction": "positive",
            },
        ],

        recommendations=[
            {
                "title": "Control Operating Costs",
                "action": (
                    "Review top cost drivers and target "
                    "discretionary spending for optimization."
                ),
                "priority": "high",
                "category": "cost",
                "expected_impact": "Improve operating margin",
                "confidence": 0.91,
            },
            {
                "title": "Investigate High-Risk Transactions",
                "action": (
                    "Route suspicious transactions to the "
                    "fraud review queue."
                ),
                "priority": "critical",
                "category": "fraud",
                "expected_impact": "Reduce fraud exposure",
                "confidence": 0.95,
            },
        ],

        what_if_results=[
            {
                "name": "Reduce operating expenses by 5%",
                "baseline_profit": 2_350_000,
                "projected_profit": 2_815_000,
            },
            {
                "name": "Increase revenue by 8%",
                "baseline_profit": 2_350_000,
                "projected_profit": 3_350_000,
            },
        ],

        root_causes=[
            {
                "category": "cost",
                "driver": "Supplier costs",
                "impact": 350000,
            },
            {
                "category": "revenue",
                "driver": "Enterprise sales growth",
                "impact": 720000,
            },
        ],

        top_revenue_drivers=[
            {
                "name": "Enterprise Sales",
                "impact": 720000,
            },
            {
                "name": "Subscription Revenue",
                "impact": 480000,
            },
        ],

        top_cost_drivers=[
            {
                "name": "Supplier Costs",
                "impact": 350000,
            },
            {
                "name": "Cloud Infrastructure",
                "impact": 180000,
            },
        ],

        period_start="2026-07-01",
        period_end="2026-09-30",

        generated_by="FinCo AI",
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    generator = ExecutiveReportGenerator()

    report = generator.generate(
        create_demo_input()
    )

    assert report.report_id.startswith(
        "FINCO-EXEC-"
    )

    assert report.financial_health_score >= 0

    assert report.financial_health_score <= 100

    assert len(report.kpis) > 0

    assert isinstance(
        report.executive_summary,
        str,
    )

    markdown = generator.to_markdown(report)

    assert "# FinCo Demo Corporation" in markdown

    print(
        "Executive report self-test passed."
    )

    print(
        f"Health score: "
        f"{report.financial_health_score}/100"
    )

    print(
        f"Status: {report.financial_status}"
    )

    print(
        f"Report ID: {report.report_id}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Run:

        python -m backend.app.reports.executive_report

    or from backend:

        python -m app.reports.executive_report
    """

    generator = ExecutiveReportGenerator()

    demo = create_demo_input()

    report = generator.generate(demo)

    json_path = generator.save_json(report)

    markdown_path = generator.save_markdown(report)

    print("=" * 70)
    print("FINCO AI - EXECUTIVE REPORT")
    print("=" * 70)

    print(
        f"Company: {report.company_name}"
    )

    print(
        f"Period: {report.reporting_period}"
    )

    print(
        f"Health Score: "
        f"{report.financial_health_score:.1f}/100"
    )

    print(
        f"Status: {report.financial_status}"
    )

    print(
        f"Report ID: {report.report_id}"
    )

    print()

    print(
        f"JSON: {json_path}"
    )

    print(
        f"Markdown: {markdown_path}"
    )

    try:
        pdf_path = generator.save_pdf(report)

        print(
            f"PDF: {pdf_path}"
        )

    except RuntimeError as exc:
        print(
            f"PDF skipped: {exc}"
        )


if __name__ == "__main__":
    main()