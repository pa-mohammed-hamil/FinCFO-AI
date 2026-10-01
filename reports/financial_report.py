"""
FinCo AI - Detailed Financial Report Generator

Path:
    backend/app/reports/financial_report.py

Purpose:
    Generate a detailed financial-analysis report from:

    - Revenue
    - Expenses
    - Profit & Loss
    - Cash Flow
    - Balance Sheet
    - Financial Ratios
    - Margins
    - Budget Variance
    - Historical Comparison
    - Root Cause Analysis
    - Financial Health
    - Management Insights
    - Recommendations

Outputs:
    - Python dataclass
    - Dictionary
    - JSON
    - Markdown
    - PDF

This module is designed to work with:
    - FastAPI
    - FinCo AI agents
    - Financial analysis services
    - Dashboard APIs
    - Scheduled reporting
    - Executive reporting workflows
"""

from __future__ import annotations

import json
import math
import os
import statistics
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


# ============================================================================
# Configuration
# ============================================================================

REPORT_VERSION = "1.0.0"

DEFAULT_CURRENCY = os.getenv(
    "FINCO_CURRENCY",
    "USD",
)

DEFAULT_REPORT_DIR = Path(
    os.getenv(
        "FINCO_REPORT_DIR",
        str(
            Path(__file__).resolve().parents[3]
            / "reports"
        ),
    )
)


# ============================================================================
# Utility functions
# ============================================================================


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
    """Perform division safely."""

    numerator = safe_float(numerator)
    denominator = safe_float(denominator)

    if denominator == 0:
        return default

    return numerator / denominator


def growth_rate(
    current: Any,
    previous: Any,
) -> float:
    """Calculate percentage growth."""

    current = safe_float(current)
    previous = safe_float(previous)

    if previous == 0:
        return 0.0

    return (
        (current - previous)
        / abs(previous)
    ) * 100


def percentage(
    numerator: Any,
    denominator: Any,
) -> float:
    """Calculate percentage."""

    return safe_divide(
        numerator,
        denominator,
    ) * 100


def round_value(
    value: Any,
    digits: int = 2,
) -> float:
    return round(
        safe_float(value),
        digits,
    )


def now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read a value from dict or object."""

    if isinstance(obj, Mapping):
        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


def text(
    value: Any,
) -> str:
    if value is None:
        return ""

    return str(value).strip()


def currency(
    value: Any,
    currency_code: str = DEFAULT_CURRENCY,
) -> str:
    """Format monetary value."""

    value = safe_float(value)

    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "INR": "₹",
        "CAD": "C$",
        "AUD": "A$",
    }

    symbol = symbols.get(
        currency_code.upper(),
        currency_code.upper() + " ",
    )

    absolute = abs(value)

    if absolute >= 1_000_000_000:
        return (
            f"{symbol}"
            f"{value / 1_000_000_000:.2f}B"
        )

    if absolute >= 1_000_000:
        return (
            f"{symbol}"
            f"{value / 1_000_000:.2f}M"
        )

    if absolute >= 1_000:
        return (
            f"{symbol}"
            f"{value / 1_000:.2f}K"
        )

    return f"{symbol}{value:,.2f}"


def status_from_percentage(
    value: float,
    good_min: float = 0,
    warning_min: float = -10,
) -> str:
    if value >= good_min:
        return "healthy"

    if value >= warning_min:
        return "watch"

    return "critical"


# ============================================================================
# Data models
# ============================================================================


@dataclass
class FinancialMetric:
    """Single financial metric."""

    name: str
    value: float
    formatted_value: str

    previous_value: Optional[float] = None
    change_percent: Optional[float] = None

    unit: str = ""
    status: str = "neutral"
    interpretation: str = ""


@dataclass
class RatioAnalysis:
    """Financial ratio analysis."""

    name: str
    value: float
    unit: str
    status: str
    interpretation: str
    benchmark: Optional[str] = None


@dataclass
class FinancialInsight:
    """Financial report insight."""

    title: str
    category: str
    severity: str
    message: str

    metric: Optional[str] = None
    value: Optional[float] = None

    impact: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class BudgetVariance:
    """Budget vs actual analysis."""

    category: str

    budget: float
    actual: float
    variance: float
    variance_percent: float

    status: str
    interpretation: str


@dataclass
class FinancialReportInput:
    """Input model for financial report generation."""

    company_name: str = "FinCo Demo Corporation"
    company_id: Optional[str] = None

    reporting_period: str = "Current Period"

    period_start: Optional[str] = None
    period_end: Optional[str] = None

    currency: str = DEFAULT_CURRENCY

    # ------------------------------------------------------------------
    # Income statement
    # ------------------------------------------------------------------

    revenue: float = 0.0
    previous_revenue: float = 0.0

    cost_of_goods_sold: float = 0.0
    previous_cost_of_goods_sold: float = 0.0

    gross_profit: Optional[float] = None
    previous_gross_profit: Optional[float] = None

    operating_expenses: float = 0.0
    previous_operating_expenses: float = 0.0

    operating_income: Optional[float] = None
    previous_operating_income: Optional[float] = None

    interest_expense: float = 0.0
    taxes: float = 0.0

    net_profit: float = 0.0
    previous_net_profit: float = 0.0

    # ------------------------------------------------------------------
    # Balance sheet
    # ------------------------------------------------------------------

    cash: float = 0.0
    accounts_receivable: float = 0.0
    inventory: float = 0.0
    current_assets: float = 0.0

    fixed_assets: float = 0.0
    total_assets: float = 0.0

    accounts_payable: float = 0.0
    short_term_debt: float = 0.0
    current_liabilities: float = 0.0

    long_term_debt: float = 0.0
    total_liabilities: float = 0.0

    equity: float = 0.0

    # ------------------------------------------------------------------
    # Cash flow
    # ------------------------------------------------------------------

    operating_cash_flow: float = 0.0
    investing_cash_flow: float = 0.0
    financing_cash_flow: float = 0.0

    net_cash_flow: Optional[float] = None
    previous_net_cash_flow: Optional[float] = None

    # ------------------------------------------------------------------
    # Ratios
    # ------------------------------------------------------------------

    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None

    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None

    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None

    return_on_assets: Optional[float] = None
    return_on_equity: Optional[float] = None

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    revenue_budget: Optional[float] = None
    expense_budget: Optional[float] = None
    profit_budget: Optional[float] = None

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    historical_periods: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Drivers
    # ------------------------------------------------------------------

    revenue_drivers: List[Any] = field(
        default_factory=list
    )

    expense_drivers: List[Any] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # External analysis
    # ------------------------------------------------------------------

    alerts: List[Any] = field(
        default_factory=list
    )

    root_causes: List[Any] = field(
        default_factory=list
    )

    recommendations: List[Any] = field(
        default_factory=list
    )

    financial_health_score: Optional[float] = None

    generated_by: str = "FinCo AI"
    user_id: Optional[str] = None


@dataclass
class FinancialReport:
    """Complete generated financial report."""

    report_id: str
    version: str
    generated_at: str

    company_name: str
    company_id: Optional[str]

    reporting_period: str
    period_start: Optional[str]
    period_end: Optional[str]

    currency: str

    financial_health_score: float
    financial_status: str

    executive_summary: str

    income_statement: Dict[str, Any]
    balance_sheet: Dict[str, Any]
    cash_flow_statement: Dict[str, Any]

    metrics: List[FinancialMetric]
    ratios: List[RatioAnalysis]

    budget_variance: List[BudgetVariance]

    historical_analysis: Dict[str, Any]

    revenue_drivers: List[Dict[str, Any]]
    expense_drivers: List[Dict[str, Any]]

    insights: List[FinancialInsight]

    recommendations: List[Dict[str, Any]]

    action_items: List[Dict[str, Any]]

    metadata: Dict[str, Any]


# ============================================================================
# Financial report generator
# ============================================================================


class FinancialReportGenerator:
    """
    Detailed financial report generator.

    Example:

        generator = FinancialReportGenerator()

        report = generator.generate(
            FinancialReportInput(
                company_name="Acme Corp",
                revenue=10_000_000,
                cost_of_goods_sold=5_500_000,
                operating_expenses=2_000_000,
                net_profit=1_500_000,
            )
        )
    """

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
    ) -> None:

        self.output_dir = Path(
            output_dir
            or DEFAULT_REPORT_DIR
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def generate(
        self,
        data: (
            FinancialReportInput
            | Mapping[str, Any]
        ),
    ) -> FinancialReport:

        data = self._normalize_input(data)

        self._derive_financial_values(data)

        health_score = (
            self._calculate_financial_health(
                data
            )
        )

        income_statement = (
            self._build_income_statement(
                data
            )
        )

        balance_sheet = (
            self._build_balance_sheet(
                data
            )
        )

        cash_flow = (
            self._build_cash_flow(
                data
            )
        )

        metrics = (
            self._build_metrics(
                data
            )
        )

        ratios = (
            self._build_ratios(
                data
            )
        )

        budget_variance = (
            self._build_budget_variance(
                data
            )
        )

        historical_analysis = (
            self._build_historical_analysis(
                data
            )
        )

        revenue_drivers = (
            self._normalize_drivers(
                data.revenue_drivers
            )
        )

        expense_drivers = (
            self._normalize_drivers(
                data.expense_drivers
            )
        )

        insights = (
            self._build_insights(
                data=data,
                health_score=health_score,
                budget_variance=budget_variance,
            )
        )

        recommendations = (
            self._build_recommendations(
                data.recommendations
            )
        )

        action_items = (
            self._build_action_items(
                data=data,
                insights=insights,
                budget_variance=budget_variance,
            )
        )

        summary = (
            self._build_executive_summary(
                data=data,
                health_score=health_score,
                insights=insights,
            )
        )

        return FinancialReport(
            report_id=(
                "FINCO-FIN-"
                f"{uuid.uuid4().hex[:12].upper()}"
            ),
            version=REPORT_VERSION,
            generated_at=now_iso(),

            company_name=data.company_name,
            company_id=data.company_id,

            reporting_period=data.reporting_period,
            period_start=data.period_start,
            period_end=data.period_end,

            currency=data.currency,

            financial_health_score=health_score,
            financial_status=self._health_status(
                health_score
            ),

            executive_summary=summary,

            income_statement=income_statement,
            balance_sheet=balance_sheet,
            cash_flow_statement=cash_flow,

            metrics=metrics,
            ratios=ratios,

            budget_variance=budget_variance,

            historical_analysis=historical_analysis,

            revenue_drivers=revenue_drivers,
            expense_drivers=expense_drivers,

            insights=insights,

            recommendations=recommendations,

            action_items=action_items,

            metadata={
                "report_type": "financial",
                "report_version": REPORT_VERSION,
                "generated_by": data.generated_by,
                "user_id": data.user_id,
                "generator": (
                    self.__class__.__name__
                ),
            },
        )

    # ========================================================================
    # Input normalization
    # ========================================================================

    def _normalize_input(
        self,
        data: (
            FinancialReportInput
            | Mapping[str, Any]
        ),
    ) -> FinancialReportInput:

        if isinstance(
            data,
            FinancialReportInput,
        ):
            return data

        if not isinstance(
            data,
            Mapping,
        ):
            raise TypeError(
                "data must be "
                "FinancialReportInput or mapping"
            )

        allowed = set(
            FinancialReportInput.__dataclass_fields__
        )

        cleaned = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return FinancialReportInput(
            **cleaned
        )

    # ========================================================================
    # Derived values
    # ========================================================================

    def _derive_financial_values(
        self,
        data: FinancialReportInput,
    ) -> None:

        # Gross profit
        if data.gross_profit is None:
            data.gross_profit = (
                data.revenue
                - data.cost_of_goods_sold
            )

        # Previous gross profit
        if data.previous_gross_profit is None:
            data.previous_gross_profit = (
                data.previous_revenue
                - data.previous_cost_of_goods_sold
            )

        # Operating income
        if data.operating_income is None:
            data.operating_income = (
                data.gross_profit
                - data.operating_expenses
            )

        # Previous operating income
        if data.previous_operating_income is None:
            data.previous_operating_income = (
                data.previous_gross_profit
                - data.previous_operating_expenses
            )

        # Net cash flow
        if data.net_cash_flow is None:
            data.net_cash_flow = (
                data.operating_cash_flow
                + data.investing_cash_flow
                + data.financing_cash_flow
            )

        # Previous net cash flow
        if data.previous_net_cash_flow is None:
            data.previous_net_cash_flow = 0.0

        # Current assets
        if data.current_assets == 0:
            data.current_assets = (
                data.cash
                + data.accounts_receivable
                + data.inventory
            )

        # Current liabilities
        if data.current_liabilities == 0:
            data.current_liabilities = (
                data.accounts_payable
                + data.short_term_debt
            )

        # Total assets
        if data.total_assets == 0:
            data.total_assets = (
                data.current_assets
                + data.fixed_assets
            )

        # Total liabilities
        if data.total_liabilities == 0:
            data.total_liabilities = (
                data.current_liabilities
                + data.long_term_debt
            )

        # Equity
        if (
            data.equity == 0
            and data.total_assets != 0
        ):
            data.equity = (
                data.total_assets
                - data.total_liabilities
            )

        # Gross margin
        if data.gross_margin is None:
            data.gross_margin = percentage(
                data.gross_profit,
                data.revenue,
            )

        # Operating margin
        if data.operating_margin is None:
            data.operating_margin = percentage(
                data.operating_income,
                data.revenue,
            )

        # Net margin
        if data.net_margin is None:
            data.net_margin = percentage(
                data.net_profit,
                data.revenue,
            )

        # Liquidity ratios
        if data.current_ratio is None:
            data.current_ratio = safe_divide(
                data.current_assets,
                data.current_liabilities,
            )

        if data.quick_ratio is None:
            quick_assets = (
                data.cash
                + data.accounts_receivable
            )

            data.quick_ratio = safe_divide(
                quick_assets,
                data.current_liabilities,
            )

        # Leverage
        if data.debt_to_equity is None:
            total_debt = (
                data.short_term_debt
                + data.long_term_debt
            )

            data.debt_to_equity = safe_divide(
                total_debt,
                data.equity,
            )

        if data.debt_to_assets is None:
            total_debt = (
                data.short_term_debt
                + data.long_term_debt
            )

            data.debt_to_assets = safe_divide(
                total_debt,
                data.total_assets,
            )

        # Returns
        if data.return_on_assets is None:
            data.return_on_assets = percentage(
                data.net_profit,
                data.total_assets,
            )

        if data.return_on_equity is None:
            data.return_on_equity = percentage(
                data.net_profit,
                data.equity,
            )

    # ========================================================================
    # Income statement
    # ========================================================================

    def _build_income_statement(
        self,
        data: FinancialReportInput,
    ) -> Dict[str, Any]:

        gross_profit = safe_float(
            data.gross_profit
        )

        operating_income = safe_float(
            data.operating_income
        )

        return {
            "revenue": {
                "value": data.revenue,
                "formatted": currency(
                    data.revenue,
                    data.currency,
                ),
            },

            "cost_of_goods_sold": {
                "value": data.cost_of_goods_sold,
                "formatted": currency(
                    data.cost_of_goods_sold,
                    data.currency,
                ),
            },

            "gross_profit": {
                "value": gross_profit,
                "formatted": currency(
                    gross_profit,
                    data.currency,
                ),
            },

            "operating_expenses": {
                "value": data.operating_expenses,
                "formatted": currency(
                    data.operating_expenses,
                    data.currency,
                ),
            },

            "operating_income": {
                "value": operating_income,
                "formatted": currency(
                    operating_income,
                    data.currency,
                ),
            },

            "interest_expense": {
                "value": data.interest_expense,
                "formatted": currency(
                    data.interest_expense,
                    data.currency,
                ),
            },

            "taxes": {
                "value": data.taxes,
                "formatted": currency(
                    data.taxes,
                    data.currency,
                ),
            },

            "net_profit": {
                "value": data.net_profit,
                "formatted": currency(
                    data.net_profit,
                    data.currency,
                ),
            },

            "gross_margin": (
                f"{data.gross_margin:.2f}%"
            ),

            "operating_margin": (
                f"{data.operating_margin:.2f}%"
            ),

            "net_margin": (
                f"{data.net_margin:.2f}%"
            ),
        }

    # ========================================================================
    # Balance sheet
    # ========================================================================

    def _build_balance_sheet(
        self,
        data: FinancialReportInput,
    ) -> Dict[str, Any]:

        return {
            "assets": {
                "cash": currency(
                    data.cash,
                    data.currency,
                ),

                "accounts_receivable": currency(
                    data.accounts_receivable,
                    data.currency,
                ),

                "inventory": currency(
                    data.inventory,
                    data.currency,
                ),

                "current_assets": currency(
                    data.current_assets,
                    data.currency,
                ),

                "fixed_assets": currency(
                    data.fixed_assets,
                    data.currency,
                ),

                "total_assets": currency(
                    data.total_assets,
                    data.currency,
                ),
            },

            "liabilities": {
                "accounts_payable": currency(
                    data.accounts_payable,
                    data.currency,
                ),

                "short_term_debt": currency(
                    data.short_term_debt,
                    data.currency,
                ),

                "current_liabilities": currency(
                    data.current_liabilities,
                    data.currency,
                ),

                "long_term_debt": currency(
                    data.long_term_debt,
                    data.currency,
                ),

                "total_liabilities": currency(
                    data.total_liabilities,
                    data.currency,
                ),
            },

            "equity": {
                "value": data.equity,
                "formatted": currency(
                    data.equity,
                    data.currency,
                ),
            },

            "balance_check": round_value(
                data.total_assets
                - data.total_liabilities
                - data.equity
            ),
        }

    # ========================================================================
    # Cash flow
    # ========================================================================

    def _build_cash_flow(
        self,
        data: FinancialReportInput,
    ) -> Dict[str, Any]:

        return {
            "operating_cash_flow": {
                "value": data.operating_cash_flow,
                "formatted": currency(
                    data.operating_cash_flow,
                    data.currency,
                ),
            },

            "investing_cash_flow": {
                "value": data.investing_cash_flow,
                "formatted": currency(
                    data.investing_cash_flow,
                    data.currency,
                ),
            },

            "financing_cash_flow": {
                "value": data.financing_cash_flow,
                "formatted": currency(
                    data.financing_cash_flow,
                    data.currency,
                ),
            },

            "net_cash_flow": {
                "value": data.net_cash_flow,
                "formatted": currency(
                    data.net_cash_flow,
                    data.currency,
                ),
            },

            "cash_flow_status": (
                "positive"
                if data.net_cash_flow >= 0
                else "negative"
            ),
        }

    # ========================================================================
    # Metrics
    # ========================================================================

    def _build_metrics(
        self,
        data: FinancialReportInput,
    ) -> List[FinancialMetric]:

        metrics: List[FinancialMetric] = []

        def add(
            name: str,
            value: float,
            previous: Optional[float],
            interpretation: str,
            unit: str = "",
        ) -> None:

            change = (
                growth_rate(
                    value,
                    previous,
                )
                if previous is not None
                else None
            )

            if unit:
                formatted = (
                    f"{value:.2f}{unit}"
                )
            else:
                formatted = currency(
                    value,
                    data.currency,
                )

            status = (
                status_from_percentage(
                    change
                )
                if change is not None
                else "neutral"
            )

            metrics.append(
                FinancialMetric(
                    name=name,
                    value=round_value(value),
                    formatted_value=formatted,
                    previous_value=(
                        round_value(previous)
                        if previous is not None
                        else None
                    ),
                    change_percent=(
                        round_value(change)
                        if change is not None
                        else None
                    ),
                    unit=unit,
                    status=status,
                    interpretation=interpretation,
                )
            )

        add(
            "Revenue",
            data.revenue,
            data.previous_revenue,
            "Total revenue generated.",
        )

        add(
            "Cost of Goods Sold",
            data.cost_of_goods_sold,
            data.previous_cost_of_goods_sold,
            "Direct cost associated with revenue generation.",
        )

        add(
            "Gross Profit",
            safe_float(data.gross_profit),
            data.previous_gross_profit,
            "Revenue remaining after direct costs.",
        )

        add(
            "Operating Expenses",
            data.operating_expenses,
            data.previous_operating_expenses,
            "Operating costs incurred during the period.",
        )

        add(
            "Operating Income",
            safe_float(data.operating_income),
            data.previous_operating_income,
            "Profit generated from core operations.",
        )

        add(
            "Net Profit",
            data.net_profit,
            data.previous_net_profit,
            "Final profit after expenses, interest and taxes.",
        )

        add(
            "Net Cash Flow",
            safe_float(data.net_cash_flow),
            data.previous_net_cash_flow,
            "Net movement in cash during the period.",
        )

        add(
            "Gross Margin",
            safe_float(data.gross_margin),
            None,
            "Percentage of revenue remaining after direct costs.",
            "%",
        )

        add(
            "Operating Margin",
            safe_float(data.operating_margin),
            None,
            "Operating income as a percentage of revenue.",
            "%",
        )

        add(
            "Net Margin",
            safe_float(data.net_margin),
            None,
            "Net profit as a percentage of revenue.",
            "%",
        )

        return metrics

    # ========================================================================
    # Ratios
    # ========================================================================

    def _build_ratios(
        self,
        data: FinancialReportInput,
    ) -> List[RatioAnalysis]:

        ratios: List[RatioAnalysis] = []

        def add(
            name: str,
            value: float,
            unit: str,
            status: str,
            interpretation: str,
            benchmark: Optional[str] = None,
        ) -> None:

            ratios.append(
                RatioAnalysis(
                    name=name,
                    value=round_value(value),
                    unit=unit,
                    status=status,
                    interpretation=interpretation,
                    benchmark=benchmark,
                )
            )

        current_ratio = safe_float(
            data.current_ratio
        )

        if current_ratio >= 1.5:
            current_status = "healthy"
        elif current_ratio >= 1:
            current_status = "watch"
        else:
            current_status = "critical"

        add(
            "Current Ratio",
            current_ratio,
            "x",
            current_status,
            (
                "Measures the company's ability to "
                "cover short-term liabilities."
            ),
            "Generally > 1.5x is stronger",
        )

        quick_ratio = safe_float(
            data.quick_ratio
        )

        if quick_ratio >= 1:
            quick_status = "healthy"
        elif quick_ratio >= 0.7:
            quick_status = "watch"
        else:
            quick_status = "critical"

        add(
            "Quick Ratio",
            quick_ratio,
            "x",
            quick_status,
            (
                "Measures short-term liquidity excluding inventory."
            ),
            "Generally around 1.0x+",
        )

        debt_equity = safe_float(
            data.debt_to_equity
        )

        if debt_equity <= 1:
            debt_status = "healthy"
        elif debt_equity <= 2:
            debt_status = "watch"
        else:
            debt_status = "critical"

        add(
            "Debt to Equity",
            debt_equity,
            "x",
            debt_status,
            (
                "Measures financial leverage relative to equity."
            ),
            "Lower leverage is generally safer",
        )

        debt_assets = safe_float(
            data.debt_to_assets
        )

        if debt_assets <= 0.5:
            debt_assets_status = "healthy"
        elif debt_assets <= 0.7:
            debt_assets_status = "watch"
        else:
            debt_assets_status = "critical"

        add(
            "Debt to Assets",
            debt_assets,
            "x",
            debt_assets_status,
            (
                "Measures the proportion of assets financed by debt."
            ),
        )

        gross_margin = safe_float(
            data.gross_margin
        )

        add(
            "Gross Margin",
            gross_margin,
            "%",
            (
                "healthy"
                if gross_margin >= 30
                else "watch"
                if gross_margin >= 15
                else "critical"
            ),
            "Profit retained after direct production costs.",
        )

        operating_margin = safe_float(
            data.operating_margin
        )

        add(
            "Operating Margin",
            operating_margin,
            "%",
            (
                "healthy"
                if operating_margin >= 15
                else "watch"
                if operating_margin >= 5
                else "critical"
            ),
            "Profitability of core business operations.",
        )

        net_margin = safe_float(
            data.net_margin
        )

        add(
            "Net Margin",
            net_margin,
            "%",
            (
                "healthy"
                if net_margin >= 10
                else "watch"
                if net_margin >= 0
                else "critical"
            ),
            "Final profit generated per unit of revenue.",
        )

        roa = safe_float(
            data.return_on_assets
        )

        add(
            "Return on Assets",
            roa,
            "%",
            (
                "healthy"
                if roa >= 10
                else "watch"
                if roa >= 0
                else "critical"
            ),
            "Measures profitability relative to total assets.",
        )

        roe = safe_float(
            data.return_on_equity
        )

        add(
            "Return on Equity",
            roe,
            "%",
            (
                "healthy"
                if roe >= 15
                else "watch"
                if roe >= 0
                else "critical"
            ),
            "Measures profitability generated from shareholder equity.",
        )

        return ratios

    # ========================================================================
    # Budget variance
    # ========================================================================

    def _build_budget_variance(
        self,
        data: FinancialReportInput,
    ) -> List[BudgetVariance]:

        output: List[BudgetVariance] = []

        def add(
            category: str,
            budget: Optional[float],
            actual: float,
            favorable_when_positive: bool,
        ) -> None:

            if budget is None:
                return

            variance = (
                actual - budget
            )

            variance_percent = percentage(
                variance,
                abs(budget),
            )

            if favorable_when_positive:
                favorable = variance >= 0
            else:
                favorable = variance <= 0

            if favorable:
                status = "favorable"
            elif abs(variance_percent) <= 5:
                status = "watch"
            else:
                status = "unfavorable"

            if favorable:
                interpretation = (
                    "Actual performance is favorable "
                    "relative to budget."
                )
            else:
                interpretation = (
                    "Actual performance is below budget "
                    "expectations."
                )

            output.append(
                BudgetVariance(
                    category=category,
                    budget=round_value(budget),
                    actual=round_value(actual),
                    variance=round_value(variance),
                    variance_percent=round_value(
                        variance_percent
                    ),
                    status=status,
                    interpretation=interpretation,
                )
            )

        add(
            "Revenue",
            data.revenue_budget,
            data.revenue,
            True,
        )

        add(
            "Expenses",
            data.expense_budget,
            data.operating_expenses,
            False,
        )

        add(
            "Profit",
            data.profit_budget,
            data.net_profit,
            True,
        )

        return output

    # ========================================================================
    # Historical analysis
    # ========================================================================

    def _build_historical_analysis(
        self,
        data: FinancialReportInput,
    ) -> Dict[str, Any]:

        periods = list(
            data.historical_periods
        )

        if not periods:
            return {
                "available": False,
                "period_count": 0,
                "trend": "not_available",
                "periods": [],
            }

        revenue_values = []

        profit_values = []

        for period in periods:

            revenue_values.append(
                safe_float(
                    get_value(
                        period,
                        "revenue",
                        0,
                    )
                )
            )

            profit_values.append(
                safe_float(
                    get_value(
                        period,
                        "profit",
                        get_value(
                            period,
                            "net_profit",
                            0,
                        ),
                    )
                )
            )

        revenue_trend = (
            "increasing"
            if len(revenue_values) >= 2
            and revenue_values[-1]
            > revenue_values[0]
            else
            "decreasing"
            if len(revenue_values) >= 2
            and revenue_values[-1]
            < revenue_values[0]
            else "stable"
        )

        profit_trend = (
            "increasing"
            if len(profit_values) >= 2
            and profit_values[-1]
            > profit_values[0]
            else
            "decreasing"
            if len(profit_values) >= 2
            and profit_values[-1]
            < profit_values[0]
            else "stable"
        )

        return {
            "available": True,
            "period_count": len(periods),
            "trend": revenue_trend,
            "revenue_trend": revenue_trend,
            "profit_trend": profit_trend,
            "average_revenue": (
                round_value(
                    statistics.mean(
                        revenue_values
                    )
                )
                if revenue_values
                else 0.0
            ),
            "average_profit": (
                round_value(
                    statistics.mean(
                        profit_values
                    )
                )
                if profit_values
                else 0.0
            ),
            "periods": periods,
        }

    # ========================================================================
    # Drivers
    # ========================================================================

    def _normalize_drivers(
        self,
        drivers: Iterable[Any],
    ) -> List[Dict[str, Any]]:

        normalized = []

        for driver in list(drivers)[:10]:

            name = get_value(
                driver,
                "name",
                get_value(
                    driver,
                    "driver",
                    get_value(
                        driver,
                        "category",
                        "Unknown Driver",
                    ),
                ),
            )

            impact = safe_float(
                get_value(
                    driver,
                    "impact",
                    get_value(
                        driver,
                        "value",
                        0,
                    ),
                )
            )

            normalized.append(
                {
                    "name": text(name),
                    "impact": round_value(
                        impact
                    ),
                    "direction": (
                        "positive"
                        if impact > 0
                        else "negative"
                        if impact < 0
                        else "neutral"
                    ),
                }
            )

        normalized.sort(
            key=lambda item: abs(
                safe_float(
                    item["impact"]
                )
            ),
            reverse=True,
        )

        return normalized

    # ========================================================================
    # Health score
    # ========================================================================

    def _calculate_financial_health(
        self,
        data: FinancialReportInput,
    ) -> float:

        if data.financial_health_score is not None:
            return round(
                max(
                    0,
                    min(
                        100,
                        safe_float(
                            data.financial_health_score
                        ),
                    ),
                ),
                2,
            )

        components: List[float] = []

        # Profitability: 30%
        net_margin = safe_float(
            data.net_margin
        )

        profitability_score = max(
            0,
            min(
                100,
                50 + net_margin * 5,
            ),
        )

        components.extend(
            [profitability_score] * 3
        )

        # Growth: 20%
        revenue_growth = growth_rate(
            data.revenue,
            data.previous_revenue,
        )

        growth_score = max(
            0,
            min(
                100,
                60 + revenue_growth * 2.5,
            ),
        )

        components.extend(
            [growth_score] * 2
        )

        # Liquidity: 20%
        current_ratio = safe_float(
            data.current_ratio
        )

        liquidity_score = max(
            0,
            min(
                100,
                current_ratio * 50,
            ),
        )

        components.extend(
            [liquidity_score] * 2
        )

        # Leverage: 15%
        debt_equity = safe_float(
            data.debt_to_equity
        )

        leverage_score = max(
            0,
            min(
                100,
                100 - debt_equity * 30,
            ),
        )

        components.append(
            leverage_score
        )

        # Cash flow: 15%
        cash_score = (
            85
            if data.net_cash_flow >= 0
            else 30
        )

        components.append(
            cash_score
        )

        if not components:
            return 50.0

        return round(
            max(
                0,
                min(
                    100,
                    statistics.mean(
                        components
                    ),
                ),
            ),
            2,
        )

    def _health_status(
        self,
        score: float,
    ) -> str:

        if score >= 85:
            return "Excellent"

        if score >= 70:
            return "Healthy"

        if score >= 50:
            return "Watch"

        if score >= 30:
            return "At Risk"

        return "Critical"

    # ========================================================================
    # Insights
    # ========================================================================

    def _build_insights(
        self,
        data: FinancialReportInput,
        health_score: float,
        budget_variance: List[BudgetVariance],
    ) -> List[FinancialInsight]:

        insights: List[FinancialInsight] = []

        revenue_growth = growth_rate(
            data.revenue,
            data.previous_revenue,
        )

        expense_growth = growth_rate(
            data.operating_expenses,
            data.previous_operating_expenses,
        )

        profit_growth = growth_rate(
            data.net_profit,
            data.previous_net_profit,
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
                        f"{revenue_growth:.1f}% "
                        "versus the previous period."
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
                    recommendation=(
                        "Investigate customer, pricing, "
                        "volume and channel-level drivers."
                    ),
                )
            )

        # Expenses
        if (
            expense_growth
            > revenue_growth + 5
        ):

            insights.append(
                FinancialInsight(
                    title="Expense Growth Outpacing Revenue",
                    category="cost",
                    severity="high",
                    message=(
                        f"Operating expenses grew "
                        f"{expense_growth:.1f}% "
                        f"while revenue grew "
                        f"{revenue_growth:.1f}%."
                    ),
                    metric="expense_growth",
                    value=expense_growth,
                    impact="negative",
                    recommendation=(
                        "Review major cost drivers and "
                        "identify controllable expenses."
                    ),
                )
            )

        # Profit
        if data.net_profit < 0:

            insights.append(
                FinancialInsight(
                    title="Net Loss",
                    category="profitability",
                    severity="critical",
                    message=(
                        "The company reported a net loss "
                        "during the reporting period."
                    ),
                    metric="net_profit",
                    value=data.net_profit,
                    impact="negative",
                    recommendation=(
                        "Perform root-cause analysis on "
                        "revenue, costs and financing."
                    ),
                )
            )

        elif profit_growth >= 10:

            insights.append(
                FinancialInsight(
                    title="Profit Improvement",
                    category="profitability",
                    severity="positive",
                    message=(
                        f"Net profit increased by "
                        f"{profit_growth:.1f}%."
                    ),
                    metric="profit_growth",
                    value=profit_growth,
                    impact="positive",
                )
            )

        # Margin compression
        if (
            data.gross_margin < 20
            and data.revenue > 0
        ):

            insights.append(
                FinancialInsight(
                    title="Gross Margin Pressure",
                    category="margin",
                    severity="high",
                    message=(
                        f"Gross margin is "
                        f"{data.gross_margin:.1f}%, "
                        "indicating pressure on direct costs "
                        "or pricing."
                    ),
                    metric="gross_margin",
                    value=data.gross_margin,
                    impact="negative",
                )
            )

        # Cash flow
        if data.net_cash_flow < 0:

            insights.append(
                FinancialInsight(
                    title="Negative Net Cash Flow",
                    category="cash_flow",
                    severity="high",
                    message=(
                        "The business consumed more cash "
                        "than it generated during the period."
                    ),
                    metric="net_cash_flow",
                    value=data.net_cash_flow,
                    impact="negative",
                    recommendation=(
                        "Review working capital, capital "
                        "expenditure and financing activity."
                    ),
                )
            )

        # Liquidity
        if (
            data.current_ratio is not None
            and data.current_ratio < 1
        ):

            insights.append(
                FinancialInsight(
                    title="Short-Term Liquidity Risk",
                    category="liquidity",
                    severity="critical",
                    message=(
                        f"Current ratio is "
                        f"{data.current_ratio:.2f}x."
                    ),
                    metric="current_ratio",
                    value=data.current_ratio,
                    impact="negative",
                    recommendation=(
                        "Improve working capital and "
                        "review near-term liabilities."
                    ),
                )
            )

        # Leverage
        if (
            data.debt_to_equity is not None
            and data.debt_to_equity > 2
        ):

            insights.append(
                FinancialInsight(
                    title="High Financial Leverage",
                    category="leverage",
                    severity="high",
                    message=(
                        f"Debt-to-equity is "
                        f"{data.debt_to_equity:.2f}x."
                    ),
                    metric="debt_to_equity",
                    value=data.debt_to_equity,
                    impact="negative",
                    recommendation=(
                        "Evaluate debt reduction and "
                        "capital structure optimization."
                    ),
                )
            )

        # Budget
        for variance in budget_variance:

            if variance.status == "unfavorable":

                insights.append(
                    FinancialInsight(
                        title=(
                            f"{variance.category} "
                            "Budget Variance"
                        ),
                        category="budget",
                        severity="high",
                        message=(
                            f"{variance.category} has an "
                            f"unfavorable variance of "
                            f"{abs(variance.variance_percent):.1f}%."
                        ),
                        metric="budget_variance",
                        value=variance.variance_percent,
                        impact="negative",
                    )
                )

        # Health
        if health_score < 40:

            insights.append(
                FinancialInsight(
                    title="Critical Financial Health",
                    category="financial_health",
                    severity="critical",
                    message=(
                        f"Financial health score is "
                        f"{health_score:.1f}/100."
                    ),
                    metric="financial_health_score",
                    value=health_score,
                    impact="negative",
                )
            )

        return insights

    # ========================================================================
    # Recommendations
    # ========================================================================

    def _build_recommendations(
        self,
        recommendations: Iterable[Any],
    ) -> List[Dict[str, Any]]:

        output = []

        for recommendation in list(
            recommendations
        )[:10]:

            output.append(
                {
                    "id": get_value(
                        recommendation,
                        "id",
                    ),

                    "title": get_value(
                        recommendation,
                        "title",
                        "Financial Recommendation",
                    ),

                    "action": get_value(
                        recommendation,
                        "action",
                        get_value(
                            recommendation,
                            "description",
                            "",
                        ),
                    ),

                    "category": get_value(
                        recommendation,
                        "category",
                        "financial",
                    ),

                    "priority": get_value(
                        recommendation,
                        "priority",
                        "medium",
                    ),

                    "expected_impact": get_value(
                        recommendation,
                        "expected_impact",
                    ),

                    "confidence": round_value(
                        get_value(
                            recommendation,
                            "confidence",
                            0,
                        )
                    ),
                }
            )

        priority = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        output.sort(
            key=lambda item: priority.get(
                text(
                    item["priority"]
                ).lower(),
                2,
            )
        )

        return output

    # ========================================================================
    # Action items
    # ========================================================================

    def _build_action_items(
        self,
        data: FinancialReportInput,
        insights: List[FinancialInsight],
        budget_variance: List[BudgetVariance],
    ) -> List[Dict[str, Any]]:

        actions = []

        for insight in insights:

            if insight.severity not in {
                "critical",
                "high",
            }:
                continue

            actions.append(
                {
                    "priority": insight.severity,
                    "action": (
                        insight.recommendation
                        or insight.message
                    ),
                    "owner": (
                        "Finance Leadership"
                    ),
                    "source": insight.category,
                }
            )

        if data.net_cash_flow < 0:

            actions.append(
                {
                    "priority": "high",
                    "action": (
                        "Review working-capital requirements "
                        "and near-term cash obligations."
                    ),
                    "owner": "Treasury",
                    "source": "cash_flow",
                }
            )

        for variance in budget_variance:

            if variance.status == "unfavorable":

                actions.append(
                    {
                        "priority": "high",
                        "action": (
                            f"Investigate unfavorable "
                            f"{variance.category.lower()} "
                            "budget variance."
                        ),
                        "owner": "Finance / FP&A",
                        "source": "budget",
                    }
                )

        # Deduplicate
        unique = []

        seen = set()

        for action in actions:

            key = text(
                action["action"]
            ).lower()

            if key in seen:
                continue

            seen.add(key)

            unique.append(action)

        priority = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        unique.sort(
            key=lambda item: priority.get(
                item["priority"],
                2,
            )
        )

        return unique[:12]

    # ========================================================================
    # Executive summary
    # ========================================================================

    def _build_executive_summary(
        self,
        data: FinancialReportInput,
        health_score: float,
        insights: List[FinancialInsight],
    ) -> str:

        revenue_growth = growth_rate(
            data.revenue,
            data.previous_revenue,
        )

        profit_growth = growth_rate(
            data.net_profit,
            data.previous_net_profit,
        )

        expense_growth = growth_rate(
            data.operating_expenses,
            data.previous_operating_expenses,
        )

        status = self._health_status(
            health_score
        )

        paragraphs = []

        paragraphs.append(
            f"{data.company_name} is assessed as "
            f"{status.lower()} for "
            f"{data.reporting_period}, with a "
            f"financial health score of "
            f"{health_score:.1f}/100."
        )

        paragraphs.append(
            f"Revenue is "
            f"{'up' if revenue_growth >= 0 else 'down'} "
            f"{abs(revenue_growth):.1f}% versus the "
            "previous period. Operating expenses are "
            f"{'up' if expense_growth >= 0 else 'down'} "
            f"{abs(expense_growth):.1f}%."
        )

        paragraphs.append(
            f"Net profit is "
            f"{currency(data.net_profit, data.currency)}, "
            f"representing a "
            f"{profit_growth:.1f}% change from the "
            "previous period."
        )

        paragraphs.append(
            f"Gross margin is "
            f"{data.gross_margin:.1f}%, operating margin "
            f"is {data.operating_margin:.1f}%, and net "
            f"margin is {data.net_margin:.1f}%."
        )

        if data.net_cash_flow < 0:

            paragraphs.append(
                "Cash flow is negative and should be "
                "closely monitored."
            )

        if data.current_ratio < 1:

            paragraphs.append(
                "Short-term liquidity requires "
                "immediate attention."
            )

        severe = [
            insight
            for insight in insights
            if insight.severity
            in {
                "critical",
                "high",
            }
        ]

        if severe:

            titles = ", ".join(
                insight.title
                for insight in severe[:3]
            )

            paragraphs.append(
                f"Priority financial issues include: "
                f"{titles}."
            )

        return " ".join(
            paragraphs
        )

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(
        self,
        report: FinancialReport,
    ) -> Dict[str, Any]:

        return asdict(report)

    def to_json(
        self,
        report: FinancialReport,
        indent: int = 2,
    ) -> str:

        return json.dumps(
            self.to_dict(report),
            indent=indent,
            ensure_ascii=False,
            default=str,
        )

    # ========================================================================
    # JSON output
    # ========================================================================

    def save_json(
        self,
        report: FinancialReport,
        filename: Optional[str] = None,
    ) -> Path:

        filename = filename or (
            f"{report.report_id.lower()}.json"
        )

        path = (
            self.output_dir
            / filename
        )

        path.write_text(
            self.to_json(report),
            encoding="utf-8",
        )

        return path

    # ========================================================================
    # Markdown output
    # ========================================================================

    def to_markdown(
        self,
        report: FinancialReport,
    ) -> str:

        lines = []

        lines.append(
            f"# {report.company_name} "
            "— Financial Analysis Report"
        )

        lines.append("")

        lines.append(
            f"**Reporting Period:** "
            f"{report.reporting_period}"
        )

        lines.append(
            f"**Report ID:** "
            f"{report.report_id}"
        )

        lines.append(
            f"**Generated:** "
            f"{report.generated_at}"
        )

        lines.append("")

        # Summary
        lines.append(
            "## Executive Summary"
        )

        lines.append("")

        lines.append(
            report.executive_summary
        )

        lines.append("")

        # Health
        lines.append(
            "## Financial Health"
        )

        lines.append("")

        lines.append(
            f"- **Score:** "
            f"{report.financial_health_score:.1f}/100"
        )

        lines.append(
            f"- **Status:** "
            f"{report.financial_status}"
        )

        lines.append("")

        # Income statement
        lines.append(
            "## Income Statement"
        )

        lines.append("")

        lines.append(
            "| Metric | Value |"
        )

        lines.append(
            "|---|---:|"
        )

        for key, item in (
            report.income_statement.items()
        ):

            if isinstance(
                item,
                dict,
            ):
                value = item.get(
                    "formatted",
                    item.get(
                        "value",
                        "—",
                    ),
                )
            else:
                value = item

            lines.append(
                f"| {key.replace('_', ' ').title()} "
                f"| {value} |"
            )

        lines.append("")

        # Balance sheet
        lines.append(
            "## Balance Sheet"
        )

        lines.append("")

        for section, values in (
            report.balance_sheet.items()
        ):

            lines.append(
                f"### {section.replace('_', ' ').title()}"
            )

            if isinstance(
                values,
                dict,
            ):

                for key, value in values.items():

                    lines.append(
                        f"- **"
                        f"{key.replace('_', ' ').title()}"
                        f":** {value}"
                    )

        lines.append("")

        # Cash flow
        lines.append(
            "## Cash Flow Statement"
        )

        lines.append("")

        for key, value in (
            report.cash_flow_statement.items()
        ):

            if isinstance(
                value,
                dict,
            ):
                value = value.get(
                    "formatted",
                    value.get(
                        "value",
                        "—",
                    ),
                )

            lines.append(
                f"- **"
                f"{key.replace('_', ' ').title()}"
                f":** {value}"
            )

        lines.append("")

        # Metrics
        lines.append(
            "## Financial Metrics"
        )

        lines.append("")

        lines.append(
            "| Metric | Value | Change | Status |"
        )

        lines.append(
            "|---|---:|---:|---|"
        )

        for metric in report.metrics:

            change = (
                f"{metric.change_percent:.2f}%"
                if metric.change_percent
                is not None
                else "—"
            )

            lines.append(
                f"| {metric.name} | "
                f"{metric.formatted_value} | "
                f"{change} | "
                f"{metric.status} |"
            )

        lines.append("")

        # Ratios
        lines.append(
            "## Ratio Analysis"
        )

        lines.append("")

        lines.append(
            "| Ratio | Value | Status | Interpretation |"
        )

        lines.append(
            "|---|---:|---|---|"
        )

        for ratio in report.ratios:

            lines.append(
                f"| {ratio.name} | "
                f"{ratio.value:.2f}{ratio.unit} | "
                f"{ratio.status} | "
                f"{ratio.interpretation} |"
            )

        lines.append("")

        # Budget
        lines.append(
            "## Budget Variance"
        )

        lines.append("")

        if report.budget_variance:

            lines.append(
                "| Category | Budget | Actual | "
                "Variance | Status |"
            )

            lines.append(
                "|---|---:|---:|---:|---|"
            )

            for item in (
                report.budget_variance
            ):

                lines.append(
                    f"| {item.category} | "
                    f"{currency(item.budget, report.currency)} | "
                    f"{currency(item.actual, report.currency)} | "
                    f"{currency(item.variance, report.currency)} "
                    f"({item.variance_percent:.1f}%) | "
                    f"{item.status} |"
                )

        else:

            lines.append(
                "No budget data supplied."
            )

        lines.append("")

        # Historical
        lines.append(
            "## Historical Analysis"
        )

        lines.append("")

        historical = (
            report.historical_analysis
        )

        lines.append(
            f"- **Available:** "
            f"{historical.get('available')}"
        )

        lines.append(
            f"- **Periods:** "
            f"{historical.get('period_count')}"
        )

        lines.append(
            f"- **Revenue Trend:** "
            f"{historical.get('revenue_trend')}"
        )

        lines.append(
            f"- **Profit Trend:** "
            f"{historical.get('profit_trend')}"
        )

        lines.append("")

        # Insights
        lines.append(
            "## Financial Insights"
        )

        lines.append("")

        if report.insights:

            for insight in report.insights:

                lines.append(
                    f"- **"
                    f"[{insight.severity.upper()}] "
                    f"{insight.title}"
                    f":** "
                    f"{insight.message}"
                )

        else:

            lines.append(
                "No material financial insights detected."
            )

        lines.append("")

        # Drivers
        lines.append(
            "## Revenue Drivers"
        )

        lines.append("")

        for driver in (
            report.revenue_drivers
        ):

            lines.append(
                f"- **{driver['name']}** — "
                f"{currency(driver['impact'], report.currency)}"
            )

        lines.append("")

        lines.append(
            "## Expense Drivers"
        )

        lines.append("")

        for driver in (
            report.expense_drivers
        ):

            lines.append(
                f"- **{driver['name']}** — "
                f"{currency(driver['impact'], report.currency)}"
            )

        lines.append("")

        # Recommendations
        lines.append(
            "## Recommendations"
        )

        lines.append("")

        for index, recommendation in enumerate(
            report.recommendations,
            start=1,
        ):

            lines.append(
                f"{index}. **"
                f"{recommendation.get('title', 'Recommendation')}"
                f"** "
                f"({str(recommendation.get('priority', 'medium')).upper()}) "
                f"— "
                f"{recommendation.get('action', '')}"
            )

        lines.append("")

        # Actions
        lines.append(
            "## Management Action Items"
        )

        lines.append("")

        for item in (
            report.action_items
        ):

            lines.append(
                f"- **"
                f"{item['priority'].upper()}"
                f"** "
                f"{item['action']} "
                f"— Owner: "
                f"{item['owner']}"
            )

        lines.append("")

        lines.append("---")

        lines.append("")

        lines.append(
            "*Generated by FinCo AI — Financial "
            "Intelligence & Decision Copilot*"
        )

        return "\n".join(
            lines
        )

    def save_markdown(
        self,
        report: FinancialReport,
        filename: Optional[str] = None,
    ) -> Path:

        filename = filename or (
            f"{report.report_id.lower()}.md"
        )

        path = (
            self.output_dir
            / filename
        )

        path.write_text(
            self.to_markdown(report),
            encoding="utf-8",
        )

        return path

    # ========================================================================
    # PDF
    # ========================================================================

    def save_pdf(
        self,
        report: FinancialReport,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate a PDF financial report.

        Requires:
            reportlab
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
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )

        except ImportError as exc:

            raise RuntimeError(
                "ReportLab is required for PDF generation. "
                "Install with: pip install reportlab"
            ) from exc

        filename = filename or (
            f"{report.report_id.lower()}.pdf"
        )

        path = (
            self.output_dir
            / filename
        )

        document = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            rightMargin=14 * mm,
            leftMargin=14 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
            title=(
                f"{report.company_name} "
                "Financial Report"
            ),
            author="FinCo AI",
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "FinCoFinancialTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=19,
            leading=23,
            spaceAfter=8,
        )

        heading_style = ParagraphStyle(
            "FinCoFinancialHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            spaceBefore=10,
            spaceAfter=6,
        )

        body_style = ParagraphStyle(
            "FinCoFinancialBody",
            parent=styles["BodyText"],
            fontSize=8.7,
            leading=13,
            spaceAfter=5,
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
                "Detailed Financial Analysis Report",
                body_style,
            )
        )

        story.append(
            Paragraph(
                f"<b>{report.company_name}</b><br/>"
                f"Period: {report.reporting_period}<br/>"
                f"Report ID: {report.report_id}",
                body_style,
            )
        )

        # Summary
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

        # Health
        health_table = Table(
            [
                [
                    "Financial Health Score",
                    f"{report.financial_health_score:.1f}/100",
                ],
                [
                    "Financial Status",
                    report.financial_status,
                ],
            ],
            colWidths=[
                75 * mm,
                75 * mm,
            ],
        )

        health_table.setStyle(
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
                        (0, -1),
                        colors.lightgrey,
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
                        5,
                    ),
                ]
            )
        )

        story.append(
            health_table
        )

        # Income Statement
        story.append(
            Paragraph(
                "Income Statement",
                heading_style,
            )
        )

        rows = [
            [
                "Metric",
                "Amount",
            ]
        ]

        for key, item in (
            report.income_statement.items()
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            rows.append(
                [
                    key.replace(
                        "_",
                        " ",
                    ).title(),
                    item.get(
                        "formatted",
                        currency(
                            item.get(
                                "value",
                                0,
                            ),
                            report.currency,
                        ),
                    ),
                ]
            )

        table = Table(
            rows,
            repeatRows=1,
            colWidths=[
                90 * mm,
                60 * mm,
            ],
        )

        table.setStyle(
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
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                ]
            )
        )

        story.append(table)

        # Ratios
        story.append(
            Paragraph(
                "Financial Ratio Analysis",
                heading_style,
            )
        )

        ratio_rows = [
            [
                "Ratio",
                "Value",
                "Status",
            ]
        ]

        for ratio in report.ratios:

            ratio_rows.append(
                [
                    ratio.name,
                    f"{ratio.value:.2f}"
                    f"{ratio.unit}",
                    ratio.status,
                ]
            )

        ratio_table = Table(
            ratio_rows,
            repeatRows=1,
            colWidths=[
                65 * mm,
                35 * mm,
                50 * mm,
            ],
        )

        ratio_table.setStyle(
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

        story.append(
            ratio_table
        )

        # Cash Flow
        story.append(
            Paragraph(
                "Cash Flow",
                heading_style,
            )
        )

        story.append(
            Paragraph(
                f"Operating Cash Flow: "
                f"{currency(report.cash_flow_statement['operating_cash_flow']['value'], report.currency)}<br/>"
                f"Investing Cash Flow: "
                f"{currency(report.cash_flow_statement['investing_cash_flow']['value'], report.currency)}<br/>"
                f"Financing Cash Flow: "
                f"{currency(report.cash_flow_statement['financing_cash_flow']['value'], report.currency)}<br/>"
                f"Net Cash Flow: "
                f"{currency(report.cash_flow_statement['net_cash_flow']['value'], report.currency)}",
                body_style,
            )
        )

        # Budget
        if report.budget_variance:

            story.append(
                Paragraph(
                    "Budget Variance",
                    heading_style,
                )
            )

            budget_rows = [
                [
                    "Category",
                    "Budget",
                    "Actual",
                    "Variance",
                    "Status",
                ]
            ]

            for item in (
                report.budget_variance
            ):

                budget_rows.append(
                    [
                        item.category,
                        currency(
                            item.budget,
                            report.currency,
                        ),
                        currency(
                            item.actual,
                            report.currency,
                        ),
                        (
                            f"{currency(item.variance, report.currency)} "
                            f"({item.variance_percent:.1f}%)"
                        ),
                        item.status,
                    ]
                )

            budget_table = Table(
                budget_rows,
                repeatRows=1,
                colWidths=[
                    30 * mm,
                    30 * mm,
                    30 * mm,
                    40 * mm,
                    25 * mm,
                ],
            )

            budget_table.setStyle(
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
                            7,
                        ),
                        (
                            "PADDING",
                            (0, 0),
                            (-1, -1),
                            4,
                        ),
                    ]
                )
            )

            story.append(
                budget_table
            )

        # Insights
        story.append(
            Paragraph(
                "Financial Insights",
                heading_style,
            )
        )

        if report.insights:

            for insight in report.insights:

                story.append(
                    Paragraph(
                        f"<b>"
                        f"[{insight.severity.upper()}] "
                        f"{insight.title}"
                        f"</b><br/>"
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
                    f"{recommendation.get('title', 'Recommendation')}"
                    f"</b> "
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

            for item in (
                report.action_items
            ):

                action_rows.append(
                    [
                        item["priority"].upper(),
                        item["action"],
                        item["owner"],
                    ]
                )

            action_table = Table(
                action_rows,
                repeatRows=1,
                colWidths=[
                    25 * mm,
                    100 * mm,
                    35 * mm,
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
                            7.2,
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
                            4,
                        ),
                    ]
                )
            )

            story.append(
                action_table
            )

        story.append(
            Spacer(
                1,
                8 * mm,
            )
        )

        story.append(
            Paragraph(
                "FinCo AI generated this report to support "
                "financial analysis and decision-making. "
                "Financial recommendations should be reviewed "
                "and approved by authorized personnel.",
                body_style,
            )
        )

        document.build(
            story
        )

        return path


# ============================================================================
# Convenience functions
# ============================================================================


def generate_financial_report(
    data: (
        FinancialReportInput
        | Mapping[str, Any]
    ),
) -> FinancialReport:

    return FinancialReportGenerator().generate(
        data
    )


def generate_financial_report_json(
    data: (
        FinancialReportInput
        | Mapping[str, Any]
    ),
    filename: Optional[str] = None,
) -> Path:

    generator = FinancialReportGenerator()

    report = generator.generate(
        data
    )

    return generator.save_json(
        report,
        filename,
    )


def generate_financial_report_markdown(
    data: (
        FinancialReportInput
        | Mapping[str, Any]
    ),
    filename: Optional[str] = None,
) -> Path:

    generator = FinancialReportGenerator()

    report = generator.generate(
        data
    )

    return generator.save_markdown(
        report,
        filename,
    )


def generate_financial_report_pdf(
    data: (
        FinancialReportInput
        | Mapping[str, Any]
    ),
    filename: Optional[str] = None,
) -> Path:

    generator = FinancialReportGenerator()

    report = generator.generate(
        data
    )

    return generator.save_pdf(
        report,
        filename,
    )


# ============================================================================
# Demo data
# ============================================================================


def create_demo_input() -> FinancialReportInput:

    return FinancialReportInput(

        company_name=(
            "FinCo Demo Corporation"
        ),

        company_id="COMP-DEMO-001",

        reporting_period="Q3 2026",

        period_start="2026-07-01",
        period_end="2026-09-30",

        currency="USD",

        # Income statement
        revenue=12_500_000,
        previous_revenue=11_200_000,

        cost_of_goods_sold=7_700_000,
        previous_cost_of_goods_sold=7_100_000,

        operating_expenses=2_000_000,
        previous_operating_expenses=1_850_000,

        interest_expense=180_000,

        taxes=270_000,

        net_profit=2_350_000,
        previous_net_profit=2_050_000,

        # Balance sheet
        cash=3_500_000,

        accounts_receivable=4_200_000,

        inventory=2_100_000,

        current_assets=9_800_000,

        fixed_assets=21_700_000,

        total_assets=31_500_000,

        accounts_payable=2_100_000,

        short_term_debt=800_000,

        current_liabilities=2_900_000,

        long_term_debt=5_800_000,

        total_liabilities=8_700_000,

        equity=22_800_000,

        # Cash flow
        operating_cash_flow=1_750_000,

        investing_cash_flow=-600_000,

        financing_cash_flow=-250_000,

        net_cash_flow=900_000,

        previous_net_cash_flow=700_000,

        # Ratios
        current_ratio=3.38,

        quick_ratio=2.66,

        debt_to_equity=0.29,

        debt_to_assets=0.21,

        gross_margin=38.4,

        operating_margin=22.4,

        net_margin=18.8,

        return_on_assets=7.46,

        return_on_equity=10.31,

        # Budget
        revenue_budget=12_000_000,

        expense_budget=1_900_000,

        profit_budget=2_200_000,

        # Historical
        historical_periods=[
            {
                "period": "Q1 2026",
                "revenue": 10_800_000,
                "profit": 1_800_000,
            },
            {
                "period": "Q2 2026",
                "revenue": 11_200_000,
                "profit": 2_050_000,
            },
            {
                "period": "Q3 2026",
                "revenue": 12_500_000,
                "profit": 2_350_000,
            },
        ],

        # Drivers
        revenue_drivers=[
            {
                "name": "Enterprise Sales",
                "impact": 720_000,
            },
            {
                "name": "Subscription Revenue",
                "impact": 480_000,
            },
            {
                "name": "International Expansion",
                "impact": 210_000,
            },
        ],

        expense_drivers=[
            {
                "name": "Cloud Infrastructure",
                "impact": 180_000,
            },
            {
                "name": "Supplier Costs",
                "impact": 350_000,
            },
            {
                "name": "Payroll",
                "impact": 120_000,
            },
        ],

        # Alerts
        alerts=[
            {
                "title": "Expense Increase",
                "severity": "medium",
                "message": (
                    "Operating costs increased "
                    "faster than planned."
                ),
            },
            {
                "title": "Margin Risk",
                "severity": "high",
                "message": (
                    "Supplier costs could create "
                    "future margin pressure."
                ),
            },
        ],

        # Root causes
        root_causes=[
            {
                "category": "cost",
                "driver": "Supplier costs",
                "impact": 350_000,
            },
            {
                "category": "revenue",
                "driver": "Enterprise sales",
                "impact": 720_000,
            },
        ],

        # Recommendations
        recommendations=[
            {
                "title": "Optimize Supplier Costs",
                "action": (
                    "Renegotiate high-value supplier "
                    "contracts and identify alternatives."
                ),
                "category": "cost",
                "priority": "high",
                "expected_impact": (
                    "Improve gross margin"
                ),
                "confidence": 0.91,
            },
            {
                "title": "Scale Enterprise Revenue",
                "action": (
                    "Increase focus on high-value "
                    "enterprise customer segments."
                ),
                "category": "revenue",
                "priority": "medium",
                "expected_impact": (
                    "Increase revenue"
                ),
                "confidence": 0.87,
            },
        ],

        financial_health_score=82.5,

        generated_by="FinCo AI",
    )


# ============================================================================
# Self-test
# ============================================================================


def _self_test() -> None:

    generator = (
        FinancialReportGenerator()
    )

    report = generator.generate(
        create_demo_input()
    )

    assert report.report_id.startswith(
        "FINCO-FIN-"
    )

    assert (
        0
        <= report.financial_health_score
        <= 100
    )

    assert (
        report.income_statement[
            "revenue"
        ]["value"]
        == 12_500_000
    )

    assert len(
        report.metrics
    ) > 0

    assert len(
        report.ratios
    ) > 0

    assert isinstance(
        report.executive_summary,
        str,
    )

    markdown = (
        generator.to_markdown(
            report
        )
    )

    assert (
        "Income Statement"
        in markdown
    )

    assert (
        "Ratio Analysis"
        in markdown
    )

    print(
        "Financial report self-test passed."
    )

    print(
        f"Health Score: "
        f"{report.financial_health_score}/100"
    )

    print(
        f"Status: "
        f"{report.financial_status}"
    )

    print(
        f"Report ID: "
        f"{report.report_id}"
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> None:

    generator = (
        FinancialReportGenerator()
    )

    data = create_demo_input()

    report = generator.generate(
        data
    )

    json_path = (
        generator.save_json(
            report
        )
    )

    markdown_path = (
        generator.save_markdown(
            report
        )
    )

    print("=" * 70)

    print(
        "FINCO AI - FINANCIAL REPORT"
    )

    print("=" * 70)

    print(
        f"Company: "
        f"{report.company_name}"
    )

    print(
        f"Period: "
        f"{report.reporting_period}"
    )

    print(
        f"Health Score: "
        f"{report.financial_health_score:.1f}/100"
    )

    print(
        f"Status: "
        f"{report.financial_status}"
    )

    print(
        f"Revenue: "
        f"{currency(data.revenue, data.currency)}"
    )

    print(
        f"Gross Profit: "
        f"{currency(data.gross_profit, data.currency)}"
    )

    print(
        f"Net Profit: "
        f"{currency(data.net_profit, data.currency)}"
    )

    print(
        f"Net Cash Flow: "
        f"{currency(data.net_cash_flow, data.currency)}"
    )

    print()

    print(
        f"JSON: {json_path}"
    )

    print(
        f"Markdown: {markdown_path}"
    )

    try:

        pdf_path = (
            generator.save_pdf(
                report
            )
        )

        print(
            f"PDF: {pdf_path}"
        )

    except RuntimeError as exc:

        print(
            f"PDF skipped: {exc}"
        )


if __name__ == "__main__":
    main()