"""
FinCo AI - Financial KPI Engine

File:
    backend/app/financial/kpis.py

Purpose:
    Calculate and interpret financial and operational KPIs from normalized
    financial data.

Responsibilities:
    - Calculate core financial KPIs.
    - Calculate profitability ratios.
    - Calculate liquidity ratios.
    - Calculate leverage ratios.
    - Calculate efficiency ratios.
    - Calculate growth KPIs.
    - Calculate cash-flow KPIs.
    - Calculate margin KPIs.
    - Compare KPIs against previous periods.
    - Classify KPI health.
    - Generate KPI explanations and evidence.
    - Produce structured KPI results for:
        * health_score.py
        * alerts/
        * recommendations/
        * forecasting/
        * reports/
        * agents/
        * what_if/
        * frontend dashboards

Design:
    This module contains pure/stateless financial business logic.
    Database access, API handling, authentication, persistence, and
    notification logic belong elsewhere.

Typical pipeline:

    Raw Financial Data
            ↓
    Extraction / Normalization
            ↓
        kpis.py
            ↓
    KPI Metrics + Health + Trends
            ↓
    ┌─────────┬───────────┬─────────────┐
    ↓         ↓           ↓             ↓
 Health    Alerts    Recommendations  Reports
 Score
            ↓
      Supervisor Agent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping, Sequence


# ============================================================================
# Constants
# ============================================================================

ZERO = Decimal("0")
ONE = Decimal("1")
ONE_HUNDRED = Decimal("100")

MONEY_QUANT = Decimal("0.01")
PERCENT_QUANT = Decimal("0.01")
RATIO_QUANT = Decimal("0.01")


# ============================================================================
# Utility Functions
# ============================================================================


def to_decimal(
    value: Any,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely convert a value to Decimal."""
    if value is None:
        return default

    if isinstance(value, Decimal):
        return value

    try:
        return Decimal(str(value))
    except (TypeError, ValueError, ArithmeticError):
        return default


def round_money(value: Decimal) -> Decimal:
    """Round monetary values to two decimal places."""
    return value.quantize(
        MONEY_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_percent(value: Decimal) -> Decimal:
    """Round percentage values to two decimal places."""
    return value.quantize(
        PERCENT_QUANT,
        rounding=ROUND_HALF_UP,
    )


def round_ratio(value: Decimal) -> Decimal:
    """Round ratio values to two decimal places."""
    return value.quantize(
        RATIO_QUANT,
        rounding=ROUND_HALF_UP,
    )


def safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = ZERO,
) -> Decimal:
    """Safely divide two Decimal values."""
    if denominator == ZERO:
        return default

    return numerator / denominator


def percentage(
    numerator: Decimal,
    denominator: Decimal,
) -> Decimal:
    """Calculate numerator / denominator as a percentage."""
    if denominator == ZERO:
        return ZERO

    return round_percent(
        numerator
        / denominator
        * ONE_HUNDRED
    )


def percentage_change(
    current: Decimal,
    previous: Decimal,
) -> Decimal:
    """Calculate period-over-period percentage change."""
    if previous == ZERO:
        if current == ZERO:
            return ZERO

        return ONE_HUNDRED

    return round_percent(
        (
            (current - previous)
            / abs(previous)
        )
        * ONE_HUNDRED
    )


# ============================================================================
# Enums
# ============================================================================


class KPICategory(str, Enum):
    """Major KPI categories."""

    PROFITABILITY = "profitability"
    LIQUIDITY = "liquidity"
    LEVERAGE = "leverage"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"
    CASH_FLOW = "cash_flow"
    MARGIN = "margin"
    OPERATING = "operating"


class KPIHealth(str, Enum):
    """Interpretation of KPI health."""

    EXCELLENT = "excellent"
    GOOD = "good"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class KPITrend(str, Enum):
    """KPI trend direction."""

    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class KPISeverity(str, Enum):
    """Severity assigned to a KPI condition."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# Financial Input Model
# ============================================================================


@dataclass(slots=True)
class FinancialKPIInput:
    """
    Normalized financial inputs used by the KPI engine.

    All monetary values should use the same reporting currency and period.
    """

    period: date | None = None
    currency: str | None = None

    # Income statement
    revenue: Decimal = ZERO
    cost_of_goods_sold: Decimal = ZERO
    gross_profit: Decimal | None = None

    operating_expenses: Decimal = ZERO
    operating_profit: Decimal | None = None

    net_profit: Decimal = ZERO
    interest_expense: Decimal = ZERO
    tax_expense: Decimal = ZERO

    # Balance sheet
    total_assets: Decimal = ZERO
    current_assets: Decimal = ZERO

    total_liabilities: Decimal = ZERO
    current_liabilities: Decimal = ZERO

    equity: Decimal = ZERO
    debt: Decimal = ZERO

    cash: Decimal = ZERO
    accounts_receivable: Decimal = ZERO
    inventory: Decimal = ZERO
    accounts_payable: Decimal = ZERO

    # Cash flow
    operating_cash_flow: Decimal = ZERO
    investing_cash_flow: Decimal = ZERO
    financing_cash_flow: Decimal = ZERO
    capital_expenditure: Decimal = ZERO

    # Operating data
    customers: Decimal = ZERO
    employees: Decimal = ZERO
    orders: Decimal = ZERO
    units_sold: Decimal = ZERO

    # Working-capital assumptions
    average_inventory: Decimal | None = None
    average_receivables: Decimal | None = None
    average_payables: Decimal | None = None

    # Optional external inputs
    budget_revenue: Decimal | None = None
    budget_profit: Decimal | None = None
    budget_expenses: Decimal | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def derive(self) -> None:
        """Derive common financial values where possible."""

        if (
            self.gross_profit is None
            and (
                self.revenue != ZERO
                or self.cost_of_goods_sold != ZERO
            )
        ):
            self.gross_profit = (
                self.revenue
                - self.cost_of_goods_sold
            )

        if (
            self.operating_profit is None
            and (
                self.gross_profit is not None
                or self.operating_expenses != ZERO
            )
        ):
            self.operating_profit = (
                self.gross_profit or ZERO
            ) - self.operating_expenses

        if (
            self.average_inventory is None
            and self.inventory != ZERO
        ):
            self.average_inventory = self.inventory

        if (
            self.average_receivables is None
            and self.accounts_receivable != ZERO
        ):
            self.average_receivables = (
                self.accounts_receivable
            )

        if (
            self.average_payables is None
            and self.accounts_payable != ZERO
        ):
            self.average_payables = (
                self.accounts_payable
            )


# ============================================================================
# KPI Result Models
# ============================================================================


@dataclass(slots=True)
class KPIResult:
    """Result of a single KPI calculation."""

    name: str
    category: KPICategory
    value: Decimal

    unit: str = "ratio"

    health: KPIHealth = KPIHealth.UNKNOWN
    trend: KPITrend = KPITrend.UNKNOWN
    severity: KPISeverity = KPISeverity.INFO

    target: Decimal | None = None
    previous_value: Decimal | None = None
    change: Decimal | None = None

    description: str = ""
    interpretation: str = ""

    evidence: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "value": str(self.value),
            "unit": self.unit,
            "health": self.health.value,
            "trend": self.trend.value,
            "severity": self.severity.value,
            "target": (
                str(self.target)
                if self.target is not None
                else None
            ),
            "previous_value": (
                str(self.previous_value)
                if self.previous_value is not None
                else None
            ),
            "change": (
                str(self.change)
                if self.change is not None
                else None
            ),
            "description": self.description,
            "interpretation": self.interpretation,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class KPIAnalysisResult:
    """Complete KPI analysis."""

    kpis: dict[str, KPIResult]

    overall_health: KPIHealth

    overall_score: Decimal

    strongest_kpis: list[str] = field(
        default_factory=list
    )

    weakest_kpis: list[str] = field(
        default_factory=list
    )

    critical_kpis: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    recommendations: list[str] = field(
        default_factory=list
    )

    summary: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def get(
        self,
        name: str,
    ) -> KPIResult | None:
        """Retrieve one KPI."""
        return self.kpis.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kpis": {
                name: result.to_dict()
                for name, result in self.kpis.items()
            },
            "overall_health": (
                self.overall_health.value
            ),
            "overall_score": str(
                self.overall_score
            ),
            "strongest_kpis": self.strongest_kpis,
            "weakest_kpis": self.weakest_kpis,
            "critical_kpis": self.critical_kpis,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "summary": self.summary,
            "metadata": self.metadata,
        }


# ============================================================================
# KPI Analyzer
# ============================================================================


class KPIAnalyzer:
    """
    Financial KPI calculation and interpretation engine.

    This class is intentionally stateless. It can therefore be safely used
    by API services, agents, background jobs, tests, and report generation.
    """

    # ------------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        data: FinancialKPIInput | Mapping[str, Any],
        *,
        previous: FinancialKPIInput | Mapping[str, Any] | None = None,
    ) -> KPIAnalysisResult:
        """
        Calculate the full KPI set.
        """
        current = self.normalize_input(data)

        previous_input = (
            self.normalize_input(previous)
            if previous is not None
            else None
        )

        current.derive()

        if previous_input is not None:
            previous_input.derive()

        kpis = self.calculate_all(
            current,
            previous=previous_input,
        )

        overall_score = self.calculate_overall_score(
            kpis
        )

        overall_health = self.classify_score(
            overall_score
        )

        strongest = self.find_strongest(
            kpis
        )

        weakest = self.find_weakest(
            kpis
        )

        critical = [
            name
            for name, result in kpis.items()
            if result.health
            in {
                KPIHealth.CRITICAL,
                KPIHealth.NEGATIVE,
            }
        ]

        warnings = self.generate_warnings(
            kpis
        )

        recommendations = (
            self.generate_recommendations(
                kpis
            )
        )

        summary = self.generate_summary(
            kpis,
            overall_health,
            overall_score,
        )

        return KPIAnalysisResult(
            kpis=kpis,
            overall_health=overall_health,
            overall_score=overall_score,
            strongest_kpis=strongest,
            weakest_kpis=weakest,
            critical_kpis=critical,
            warnings=warnings,
            recommendations=recommendations,
            summary=summary,
            metadata={
                "period": (
                    current.period.isoformat()
                    if current.period
                    else None
                ),
                "currency": current.currency,
                "kpi_count": len(kpis),
            },
        )

    # ------------------------------------------------------------------
    # KPI Collection
    # ------------------------------------------------------------------

    def calculate_all(
        self,
        data: FinancialKPIInput,
        *,
        previous: FinancialKPIInput | None = None,
    ) -> dict[str, KPIResult]:
        """Calculate all supported KPIs."""

        kpis: dict[str, KPIResult] = {}

        # Profitability
        self._add(
            kpis,
            self.gross_margin(data, previous),
        )
        self._add(
            kpis,
            self.operating_margin(data, previous),
        )
        self._add(
            kpis,
            self.net_margin(data, previous),
        )
        self._add(
            kpis,
            self.roa(data, previous),
        )
        self._add(
            kpis,
            self.roe(data, previous),
        )
        self._add(
            kpis,
            self.roic(data, previous),
        )

        # Liquidity
        self._add(
            kpis,
            self.current_ratio(data, previous),
        )
        self._add(
            kpis,
            self.quick_ratio(data, previous),
        )
        self._add(
            kpis,
            self.cash_ratio(data, previous),
        )

        # Leverage
        self._add(
            kpis,
            self.debt_to_equity(data, previous),
        )
        self._add(
            kpis,
            self.debt_ratio(data, previous),
        )
        self._add(
            kpis,
            self.interest_coverage(data, previous),
        )

        # Efficiency
        self._add(
            kpis,
            self.asset_turnover(data, previous),
        )
        self._add(
            kpis,
            self.inventory_turnover(data, previous),
        )
        self._add(
            kpis,
            self.receivables_turnover(data, previous),
        )
        self._add(
            kpis,
            self.payables_turnover(data, previous),
        )

        # Working capital
        self._add(
            kpis,
            self.days_sales_outstanding(
                data,
                previous,
            ),
        )
        self._add(
            kpis,
            self.days_inventory_outstanding(
                data,
                previous,
            ),
        )
        self._add(
            kpis,
            self.days_payables_outstanding(
                data,
                previous,
            ),
        )
        self._add(
            kpis,
            self.cash_conversion_cycle(
                data,
                previous,
            ),
        )

        # Growth
        if previous is not None:
            self._add(
                kpis,
                self.revenue_growth(
                    data,
                    previous,
                ),
            )
            self._add(
                kpis,
                self.profit_growth(
                    data,
                    previous,
                ),
            )
            self._add(
                kpis,
                self.expense_growth(
                    data,
                    previous,
                ),
            )

        # Cash flow
        self._add(
            kpis,
            self.operating_cash_flow_margin(
                data,
                previous,
            ),
        )
        self._add(
            kpis,
            self.free_cash_flow(
                data,
                previous,
            ),
        )
        self._add(
            kpis,
            self.cash_flow_coverage(
                data,
                previous,
            ),
        )

        # Operating
        self._add(
            kpis,
            self.revenue_per_employee(
                data,
                previous,
            ),
        )
        self._add(
            kpis,
            self.revenue_per_customer(
                data,
                previous,
            ),
        )
        self._add(
            kpis,
            self.revenue_per_order(
                data,
                previous,
            ),
        )

        return kpis

    @staticmethod
    def _add(
        collection: dict[str, KPIResult],
        result: KPIResult | None,
    ) -> None:
        if result is not None:
            collection[result.name] = result

    # ------------------------------------------------------------------
    # Profitability KPIs
    # ------------------------------------------------------------------

    def gross_margin(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.revenue == ZERO:
            return None

        value = percentage(
            data.gross_profit or ZERO,
            data.revenue,
        )

        return self._build_result(
            name="gross_margin",
            category=KPICategory.MARGIN,
            value=value,
            unit="percent",
            previous=self._previous_value(
                previous,
                "gross_margin",
            ),
            description=(
                "Gross profit generated from each unit of revenue."
            ),
            interpretation=(
                "Higher gross margin generally indicates stronger "
                "pricing power, product economics, or cost control."
            ),
            health=self._margin_health(value),
        )

    def operating_margin(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.revenue == ZERO:
            return None

        value = percentage(
            data.operating_profit or ZERO,
            data.revenue,
        )

        return self._build_result(
            name="operating_margin",
            category=KPICategory.MARGIN,
            value=value,
            unit="percent",
            previous=self._previous_value(
                previous,
                "operating_margin",
            ),
            description=(
                "Operating profit generated from revenue."
            ),
            interpretation=(
                "Measures profitability after operating expenses."
            ),
            health=self._margin_health(
                value,
                healthy=15,
                warning=5,
            ),
        )

    def net_margin(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.revenue == ZERO:
            return None

        value = percentage(
            data.net_profit,
            data.revenue,
        )

        return self._build_result(
            name="net_margin",
            category=KPICategory.PROFITABILITY,
            value=value,
            unit="percent",
            previous=self._previous_value(
                previous,
                "net_margin",
            ),
            description=(
                "Net profit generated from revenue."
            ),
            interpretation=(
                "Shows the percentage of revenue retained as "
                "bottom-line profit after all expenses."
            ),
            health=self._margin_health(
                value,
                healthy=10,
                warning=3,
            ),
        )

    def roa(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.total_assets == ZERO:
            return None

        value = percentage(
            data.net_profit,
            data.total_assets,
        )

        return self._build_result(
            name="roa",
            category=KPICategory.PROFITABILITY,
            value=value,
            unit="percent",
            previous=self._previous_value(
                previous,
                "roa",
            ),
            description=(
                "Return generated on total assets."
            ),
            interpretation=(
                "Higher ROA indicates more efficient use of assets "
                "to generate profit."
            ),
            health=self._margin_health(
                value,
                healthy=8,
                warning=3,
            ),
        )

    def roe(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.equity == ZERO:
            return None

        value = percentage(
            data.net_profit,
            data.equity,
        )

        return self._build_result(
            name="roe",
            category=KPICategory.PROFITABILITY,
            value=value,
            unit="percent",
            previous=self._previous_value(
                previous,
                "roe",
            ),
            description=(
                "Return generated on shareholder equity."
            ),
            interpretation=(
                "Measures how efficiently shareholder capital "
                "is converted into profit."
            ),
            health=self._margin_health(
                value,
                healthy=15,
                warning=5,
            ),
        )

    def roic(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        invested_capital = (
            data.equity
            + data.debt
            - data.cash
        )

        if invested_capital <= ZERO:
            return None

        nopat = (
            data.operating_profit or ZERO
        ) - data.tax_expense

        value = percentage(
            nopat,
            invested_capital,
        )

        return self._build_result(
            name="roic",
            category=KPICategory.PROFITABILITY,
            value=value,
            unit="percent",
            previous=self._previous_value(
                previous,
                "roic",
            ),
            description=(
                "Return generated on invested capital."
            ),
            interpretation=(
                "ROIC evaluates whether the business generates "
                "sufficient returns from invested capital."
            ),
            health=self._margin_health(
                value,
                healthy=10,
                warning=5,
            ),
        )

    # ------------------------------------------------------------------
    # Liquidity KPIs
    # ------------------------------------------------------------------

    def current_ratio(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.current_liabilities == ZERO:
            return None

        value = round_ratio(
            safe_divide(
                data.current_assets,
                data.current_liabilities,
            )
        )

        health = (
            KPIHealth.HEALTHY
            if value >= Decimal("1.5")
            else KPIHealth.WARNING
            if value >= Decimal("1.0")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="current_ratio",
            category=KPICategory.LIQUIDITY,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "current_ratio",
            ),
            description=(
                "Current assets available to cover current liabilities."
            ),
            interpretation=(
                "A ratio above 1 generally indicates current assets "
                "exceed current liabilities."
            ),
            health=health,
        )

    def quick_ratio(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.current_liabilities == ZERO:
            return None

        quick_assets = (
            data.cash
            + data.accounts_receivable
        )

        value = round_ratio(
            safe_divide(
                quick_assets,
                data.current_liabilities,
            )
        )

        health = (
            KPIHealth.HEALTHY
            if value >= Decimal("1.0")
            else KPIHealth.WARNING
            if value >= Decimal("0.75")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="quick_ratio",
            category=KPICategory.LIQUIDITY,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "quick_ratio",
            ),
            description=(
                "Liquid assets available to cover current liabilities."
            ),
            interpretation=(
                "Quick ratio excludes inventory and focuses on "
                "more immediately realizable assets."
            ),
            health=health,
        )

    def cash_ratio(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.current_liabilities == ZERO:
            return None

        value = round_ratio(
            safe_divide(
                data.cash,
                data.current_liabilities,
            )
        )

        health = (
            KPIHealth.HEALTHY
            if value >= Decimal("0.5")
            else KPIHealth.WARNING
            if value >= Decimal("0.2")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="cash_ratio",
            category=KPICategory.LIQUIDITY,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "cash_ratio",
            ),
            description=(
                "Cash and cash equivalents relative to current liabilities."
            ),
            interpretation=(
                "Measures immediate liquidity without relying on "
                "receivables or inventory conversion."
            ),
            health=health,
        )

    # ------------------------------------------------------------------
    # Leverage KPIs
    # ------------------------------------------------------------------

    def debt_to_equity(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.equity == ZERO:
            return None

        value = round_ratio(
            safe_divide(
                data.debt,
                data.equity,
            )
        )

        health = (
            KPIHealth.HEALTHY
            if value <= Decimal("1.0")
            else KPIHealth.WARNING
            if value <= Decimal("2.0")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="debt_to_equity",
            category=KPICategory.LEVERAGE,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "debt_to_equity",
            ),
            description=(
                "Debt relative to shareholder equity."
            ),
            interpretation=(
                "Higher debt-to-equity generally indicates greater "
                "financial leverage and financing risk."
            ),
            health=health,
        )

    def debt_ratio(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.total_assets == ZERO:
            return None

        value = percentage(
            data.total_liabilities,
            data.total_assets,
        )

        health = (
            KPIHealth.HEALTHY
            if value <= Decimal("50")
            else KPIHealth.WARNING
            if value <= Decimal("70")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="debt_ratio",
            category=KPICategory.LEVERAGE,
            value=value,
            unit="percent",
            previous=self._previous_value(
                previous,
                "debt_ratio",
            ),
            description=(
                "Percentage of assets financed by liabilities."
            ),
            interpretation=(
                "A higher debt ratio indicates greater reliance "
                "on liabilities to finance assets."
            ),
            health=health,
        )

    def interest_coverage(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.interest_expense <= ZERO:
            return None

        operating_profit = (
            data.operating_profit or ZERO
        )

        value = round_ratio(
            safe_divide(
                operating_profit,
                data.interest_expense,
            )
        )

        health = (
            KPIHealth.HEALTHY
            if value >= Decimal("5")
            else KPIHealth.WARNING
            if value >= Decimal("2")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="interest_coverage",
            category=KPICategory.LEVERAGE,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "interest_coverage",
            ),
            description=(
                "Operating profit available to cover interest expense."
            ),
            interpretation=(
                "Low interest coverage indicates greater difficulty "
                "servicing debt obligations."
            ),
            health=health,
        )

    # ------------------------------------------------------------------
    # Efficiency KPIs
    # ------------------------------------------------------------------

    def asset_turnover(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.total_assets == ZERO:
            return None

        value = round_ratio(
            safe_divide(
                data.revenue,
                data.total_assets,
            )
        )

        return self._build_result(
            name="asset_turnover",
            category=KPICategory.EFFICIENCY,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "asset_turnover",
            ),
            description=(
                "Revenue generated per unit of assets."
            ),
            interpretation=(
                "Higher asset turnover generally indicates "
                "more efficient use of assets."
            ),
            health=self._higher_is_better(
                value,
                healthy=1.5,
                warning=0.75,
            ),
        )

    def inventory_turnover(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        inventory = (
            data.average_inventory
            or data.inventory
        )

        if inventory == ZERO:
            return None

        value = round_ratio(
            safe_divide(
                data.cost_of_goods_sold,
                inventory,
            )
        )

        return self._build_result(
            name="inventory_turnover",
            category=KPICategory.EFFICIENCY,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "inventory_turnover",
            ),
            description=(
                "Number of times inventory is converted during the period."
            ),
            interpretation=(
                "Higher turnover can indicate efficient inventory "
                "management, although excessively high turnover may "
                "also indicate understocking."
            ),
            health=self._higher_is_better(
                value,
                healthy=6,
                warning=3,
            ),
        )

    def receivables_turnover(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        receivables = (
            data.average_receivables
            or data.accounts_receivable
        )

        if receivables == ZERO:
            return None

        value = round_ratio(
            safe_divide(
                data.revenue,
                receivables,
            )
        )

        return self._build_result(
            name="receivables_turnover",
            category=KPICategory.EFFICIENCY,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "receivables_turnover",
            ),
            description=(
                "Number of times receivables are converted into revenue."
            ),
            interpretation=(
                "Higher turnover generally indicates faster collection "
                "of receivables."
            ),
            health=self._higher_is_better(
                value,
                healthy=8,
                warning=4,
            ),
        )

    def payables_turnover(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        payables = (
            data.average_payables
            or data.accounts_payable
        )

        if payables == ZERO:
            return None

        value = round_ratio(
            safe_divide(
                data.cost_of_goods_sold,
                payables,
            )
        )

        return self._build_result(
            name="payables_turnover",
            category=KPICategory.EFFICIENCY,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "payables_turnover",
            ),
            description=(
                "Number of times supplier payables are settled."
            ),
            interpretation=(
                "Payables turnover should be interpreted alongside "
                "supplier terms and cash-flow requirements."
            ),
            health=KPIHealth.HEALTHY,
        )

    # ------------------------------------------------------------------
    # Working Capital KPIs
    # ------------------------------------------------------------------

    def days_sales_outstanding(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
        *,
        days: Decimal = Decimal("365"),
    ) -> KPIResult | None:
        if data.revenue == ZERO:
            return None

        receivables = (
            data.average_receivables
            or data.accounts_receivable
        )

        value = round_ratio(
            safe_divide(
                receivables,
                data.revenue,
            )
            * days
        )

        health = (
            KPIHealth.HEALTHY
            if value <= Decimal("45")
            else KPIHealth.WARNING
            if value <= Decimal("75")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="days_sales_outstanding",
            category=KPICategory.EFFICIENCY,
            value=value,
            unit="days",
            previous=self._previous_value(
                previous,
                "days_sales_outstanding",
            ),
            description=(
                "Average number of days required to collect receivables."
            ),
            interpretation=(
                "Lower DSO generally indicates faster customer collections."
            ),
            health=health,
            lower_is_better=True,
        )

    def days_inventory_outstanding(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
        *,
        days: Decimal = Decimal("365"),
    ) -> KPIResult | None:
        if data.cost_of_goods_sold == ZERO:
            return None

        inventory = (
            data.average_inventory
            or data.inventory
        )

        value = round_ratio(
            safe_divide(
                inventory,
                data.cost_of_goods_sold,
            )
            * days
        )

        health = (
            KPIHealth.HEALTHY
            if value <= Decimal("60")
            else KPIHealth.WARNING
            if value <= Decimal("120")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="days_inventory_outstanding",
            category=KPICategory.EFFICIENCY,
            value=value,
            unit="days",
            previous=self._previous_value(
                previous,
                "days_inventory_outstanding",
            ),
            description=(
                "Average number of days inventory remains outstanding."
            ),
            interpretation=(
                "Lower inventory days generally indicate faster inventory "
                "conversion and lower carrying requirements."
            ),
            health=health,
            lower_is_better=True,
        )

    def days_payables_outstanding(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
        *,
        days: Decimal = Decimal("365"),
    ) -> KPIResult | None:
        if data.cost_of_goods_sold == ZERO:
            return None

        payables = (
            data.average_payables
            or data.accounts_payable
        )

        value = round_ratio(
            safe_divide(
                payables,
                data.cost_of_goods_sold,
            )
            * days
        )

        return self._build_result(
            name="days_payables_outstanding",
            category=KPICategory.EFFICIENCY,
            value=value,
            unit="days",
            previous=self._previous_value(
                previous,
                "days_payables_outstanding",
            ),
            description=(
                "Average number of days taken to pay suppliers."
            ),
            interpretation=(
                "Higher DPO can preserve cash but may indicate "
                "supplier-payment pressure if excessive."
            ),
            health=KPIHealth.HEALTHY,
        )

    def cash_conversion_cycle(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        dso = self.days_sales_outstanding(
            data
        )

        dio = self.days_inventory_outstanding(
            data
        )

        dpo = self.days_payables_outstanding(
            data
        )

        if not dso or not dio or not dpo:
            return None

        value = round_ratio(
            dso.value
            + dio.value
            - dpo.value
        )

        health = (
            KPIHealth.HEALTHY
            if value <= Decimal("60")
            else KPIHealth.WARNING
            if value <= Decimal("120")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="cash_conversion_cycle",
            category=KPICategory.LIQUIDITY,
            value=value,
            unit="days",
            previous=self._previous_value(
                previous,
                "cash_conversion_cycle",
            ),
            description=(
                "Time required to convert operating investment "
                "back into cash."
            ),
            interpretation=(
                "Lower CCC generally indicates stronger working-capital "
                "efficiency and faster cash recovery."
            ),
            health=health,
            lower_is_better=True,
        )

    # ------------------------------------------------------------------
    # Growth KPIs
    # ------------------------------------------------------------------

    def revenue_growth(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput,
    ) -> KPIResult:
        value = percentage_change(
            data.revenue,
            previous.revenue,
        )

        return self._build_result(
            name="revenue_growth",
            category=KPICategory.GROWTH,
            value=value,
            unit="percent",
            previous=ZERO,
            description=(
                "Period-over-period revenue growth."
            ),
            interpretation=(
                "Positive revenue growth indicates expansion relative "
                "to the previous period."
            ),
            health=self._growth_health(value),
        )

    def profit_growth(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput,
    ) -> KPIResult:
        value = percentage_change(
            data.net_profit,
            previous.net_profit,
        )

        return self._build_result(
            name="profit_growth",
            category=KPICategory.GROWTH,
            value=value,
            unit="percent",
            previous=ZERO,
            description=(
                "Period-over-period net profit growth."
            ),
            interpretation=(
                "Profit growth indicates whether bottom-line earnings "
                "are improving or deteriorating."
            ),
            health=self._growth_health(value),
        )

    def expense_growth(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput,
    ) -> KPIResult:
        value = percentage_change(
            data.operating_expenses,
            previous.operating_expenses,
        )

        # Expense growth is inverse: lower is generally better.
        health = (
            KPIHealth.HEALTHY
            if value <= Decimal("5")
            else KPIHealth.WARNING
            if value <= Decimal("15")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="expense_growth",
            category=KPICategory.GROWTH,
            value=value,
            unit="percent",
            previous=ZERO,
            description=(
                "Period-over-period operating expense growth."
            ),
            interpretation=(
                "Rapid expense growth without corresponding revenue "
                "growth can pressure margins."
            ),
            health=health,
        )

    # ------------------------------------------------------------------
    # Cash Flow KPIs
    # ------------------------------------------------------------------

    def operating_cash_flow_margin(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.revenue == ZERO:
            return None

        value = percentage(
            data.operating_cash_flow,
            data.revenue,
        )

        return self._build_result(
            name="operating_cash_flow_margin",
            category=KPICategory.CASH_FLOW,
            value=value,
            unit="percent",
            previous=self._previous_value(
                previous,
                "operating_cash_flow_margin",
            ),
            description=(
                "Operating cash flow generated per unit of revenue."
            ),
            interpretation=(
                "Higher operating cash-flow margin generally indicates "
                "stronger conversion of revenue into cash."
            ),
            health=self._margin_health(
                value,
                healthy=10,
                warning=3,
            ),
        )

    def free_cash_flow(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult:
        value = round_money(
            data.operating_cash_flow
            - abs(data.capital_expenditure)
        )

        health = (
            KPIHealth.HEALTHY
            if value > ZERO
            else KPIHealth.WARNING
            if value == ZERO
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="free_cash_flow",
            category=KPICategory.CASH_FLOW,
            value=value,
            unit="currency",
            previous=self._previous_value(
                previous,
                "free_cash_flow",
            ),
            description=(
                "Cash remaining after operating cash generation "
                "and capital expenditure."
            ),
            interpretation=(
                "Positive free cash flow provides flexibility for debt "
                "repayment, reinvestment, dividends, or reserves."
            ),
            health=health,
        )

    def cash_flow_coverage(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        debt_service = (
            abs(data.interest_expense)
            + abs(data.debt)
        )

        if debt_service == ZERO:
            return None

        value = round_ratio(
            safe_divide(
                data.operating_cash_flow,
                debt_service,
            )
        )

        health = (
            KPIHealth.HEALTHY
            if value >= Decimal("1.5")
            else KPIHealth.WARNING
            if value >= Decimal("1")
            else KPIHealth.CRITICAL
        )

        return self._build_result(
            name="cash_flow_coverage",
            category=KPICategory.CASH_FLOW,
            value=value,
            unit="ratio",
            previous=self._previous_value(
                previous,
                "cash_flow_coverage",
            ),
            description=(
                "Operating cash flow relative to debt obligations."
            ),
            interpretation=(
                "Higher coverage indicates stronger cash capacity "
                "to service financing obligations."
            ),
            health=health,
        )

    # ------------------------------------------------------------------
    # Operating KPIs
    # ------------------------------------------------------------------

    def revenue_per_employee(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.employees == ZERO:
            return None

        value = round_money(
            safe_divide(
                data.revenue,
                data.employees,
            )
        )

        return self._build_result(
            name="revenue_per_employee",
            category=KPICategory.OPERATING,
            value=value,
            unit="currency",
            previous=self._previous_value(
                previous,
                "revenue_per_employee",
            ),
            description=(
                "Revenue generated per employee."
            ),
            interpretation=(
                "Higher revenue per employee can indicate stronger "
                "workforce productivity, but should be interpreted "
                "within the company's operating model."
            ),
            health=KPIHealth.HEALTHY,
        )

    def revenue_per_customer(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.customers == ZERO:
            return None

        value = round_money(
            safe_divide(
                data.revenue,
                data.customers,
            )
        )

        return self._build_result(
            name="revenue_per_customer",
            category=KPICategory.OPERATING,
            value=value,
            unit="currency",
            previous=self._previous_value(
                previous,
                "revenue_per_customer",
            ),
            description=(
                "Average revenue generated per customer."
            ),
            interpretation=(
                "Useful for monitoring customer monetization and "
                "changes in customer value."
            ),
            health=KPIHealth.HEALTHY,
        )

    def revenue_per_order(
        self,
        data: FinancialKPIInput,
        previous: FinancialKPIInput | None = None,
    ) -> KPIResult | None:
        if data.orders == ZERO:
            return None

        value = round_money(
            safe_divide(
                data.revenue,
                data.orders,
            )
        )

        return self._build_result(
            name="revenue_per_order",
            category=KPICategory.OPERATING,
            value=value,
            unit="currency",
            previous=self._previous_value(
                previous,
                "revenue_per_order",
            ),
            description=(
                "Average revenue generated per order."
            ),
            interpretation=(
                "Tracks average order economics and changes in "
                "transaction value."
            ),
            health=KPIHealth.HEALTHY,
        )

    # ------------------------------------------------------------------
    # Health Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _margin_health(
        value: Decimal,
        *,
        healthy: int | Decimal = 10,
        warning: int | Decimal = 3,
    ) -> KPIHealth:
        if value >= to_decimal(healthy):
            return KPIHealth.HEALTHY

        if value >= to_decimal(warning):
            return KPIHealth.WARNING

        if value >= ZERO:
            return KPIHealth.WARNING

        return KPIHealth.NEGATIVE

    @staticmethod
    def _higher_is_better(
        value: Decimal,
        *,
        healthy: int | Decimal,
        warning: int | Decimal,
    ) -> KPIHealth:
        if value >= to_decimal(healthy):
            return KPIHealth.HEALTHY

        if value >= to_decimal(warning):
            return KPIHealth.WARNING

        return KPIHealth.CRITICAL

    @staticmethod
    def _growth_health(
        value: Decimal,
    ) -> KPIHealth:
        if value >= Decimal("15"):
            return KPIHealth.EXCELLENT

        if value >= Decimal("5"):
            return KPIHealth.HEALTHY

        if value >= ZERO:
            return KPIHealth.WARNING

        if value >= Decimal("-10"):
            return KPIHealth.WARNING

        return KPIHealth.CRITICAL

    @staticmethod
    def classify_score(
        score: Decimal,
    ) -> KPIHealth:
        if score >= Decimal("85"):
            return KPIHealth.EXCELLENT

        if score >= Decimal("70"):
            return KPIHealth.HEALTHY

        if score >= Decimal("50"):
            return KPIHealth.WARNING

        return KPIHealth.CRITICAL

    # ------------------------------------------------------------------
    # KPI Result Construction
    # ------------------------------------------------------------------

    def _build_result(
        self,
        *,
        name: str,
        category: KPICategory,
        value: Decimal,
        unit: str,
        previous: Decimal | None,
        description: str,
        interpretation: str,
        health: KPIHealth,
        lower_is_better: bool = False,
    ) -> KPIResult:
        change = None
        trend = KPITrend.UNKNOWN

        if previous is not None:
            change = percentage_change(
                value,
                previous,
            )

            if abs(change) < Decimal("2"):
                trend = KPITrend.STABLE
            elif lower_is_better:
                trend = (
                    KPITrend.IMPROVING
                    if change < ZERO
                    else KPITrend.DETERIORATING
                )
            else:
                trend = (
                    KPITrend.IMPROVING
                    if change > ZERO
                    else KPITrend.DETERIORATING
                )

        severity = self._severity_from_health(
            health
        )

        evidence = [
            f"{name} = {value}",
        ]

        if previous is not None:
            evidence.append(
                f"Previous period = {previous}"
            )

        if change is not None:
            evidence.append(
                f"Change = {change}%"
            )

        return KPIResult(
            name=name,
            category=category,
            value=value,
            unit=unit,
            health=health,
            trend=trend,
            severity=severity,
            previous_value=previous,
            change=change,
            description=description,
            interpretation=interpretation,
            evidence=evidence,
        )

    @staticmethod
    def _severity_from_health(
        health: KPIHealth,
    ) -> KPISeverity:
        mapping = {
            KPIHealth.EXCELLENT: KPISeverity.INFO,
            KPIHealth.GOOD: KPISeverity.INFO,
            KPIHealth.HEALTHY: KPISeverity.LOW,
            KPIHealth.WARNING: KPISeverity.MEDIUM,
            KPIHealth.CRITICAL: KPISeverity.HIGH,
            KPIHealth.NEGATIVE: KPISeverity.CRITICAL,
            KPIHealth.UNKNOWN: KPISeverity.INFO,
        }

        return mapping[health]

    # ------------------------------------------------------------------
    # Previous KPI Values
    # ------------------------------------------------------------------

    def _previous_value(
        self,
        previous: FinancialKPIInput | None,
        metric: str,
    ) -> Decimal | None:
        if previous is None:
            return None

        # Use the same KPI formula for the previous period.
        methods = {
            "gross_margin": self.gross_margin,
            "operating_margin": self.operating_margin,
            "net_margin": self.net_margin,
            "roa": self.roa,
            "roe": self.roe,
            "roic": self.roic,
            "current_ratio": self.current_ratio,
            "quick_ratio": self.quick_ratio,
            "cash_ratio": self.cash_ratio,
            "debt_to_equity": self.debt_to_equity,
            "debt_ratio": self.debt_ratio,
            "interest_coverage": self.interest_coverage,
            "asset_turnover": self.asset_turnover,
            "inventory_turnover": self.inventory_turnover,
            "receivables_turnover": self.receivables_turnover,
            "payables_turnover": self.payables_turnover,
            "days_sales_outstanding": (
                self.days_sales_outstanding
            ),
            "days_inventory_outstanding": (
                self.days_inventory_outstanding
            ),
            "days_payables_outstanding": (
                self.days_payables_outstanding
            ),
            "cash_conversion_cycle": (
                self.cash_conversion_cycle
            ),
            "operating_cash_flow_margin": (
                self.operating_cash_flow_margin
            ),
            "free_cash_flow": self.free_cash_flow,
            "cash_flow_coverage": (
                self.cash_flow_coverage
            ),
            "revenue_per_employee": (
                self.revenue_per_employee
            ),
            "revenue_per_customer": (
                self.revenue_per_customer
            ),
            "revenue_per_order": (
                self.revenue_per_order
            ),
        }

        method = methods.get(metric)

        if method is None:
            return None

        try:
            result = method(previous)

            if result is None:
                return None

            return result.value

        except (ArithmeticError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def calculate_overall_score(
        self,
        kpis: Mapping[str, KPIResult],
    ) -> Decimal:
        """
        Calculate a normalized 0-100 KPI health score.
        """
        if not kpis:
            return ZERO

        weights = {
            KPICategory.PROFITABILITY: Decimal("0.25"),
            KPICategory.LIQUIDITY: Decimal("0.20"),
            KPICategory.LEVERAGE: Decimal("0.15"),
            KPICategory.EFFICIENCY: Decimal("0.15"),
            KPICategory.GROWTH: Decimal("0.15"),
            KPICategory.CASH_FLOW: Decimal("0.10"),
        }

        health_scores = {
            KPIHealth.EXCELLENT: Decimal("100"),
            KPIHealth.GOOD: Decimal("90"),
            KPIHealth.HEALTHY: Decimal("80"),
            KPIHealth.WARNING: Decimal("55"),
            KPIHealth.CRITICAL: Decimal("25"),
            KPIHealth.NEGATIVE: Decimal("10"),
            KPIHealth.UNKNOWN: Decimal("50"),
        }

        category_totals: dict[
            KPICategory,
            Decimal,
        ] = {}

        category_counts: dict[
            KPICategory,
            Decimal,
        ] = {}

        for result in kpis.values():
            category = result.category

            category_totals[category] = (
                category_totals.get(
                    category,
                    ZERO,
                )
                + health_scores[result.health]
            )

            category_counts[category] = (
                category_counts.get(
                    category,
                    ZERO,
                )
                + ONE
            )

        weighted_score = ZERO
        total_weight = ZERO

        for category, total in category_totals.items():
            average = (
                total
                / category_counts[category]
            )

            weight = weights.get(
                category,
                Decimal("0.05"),
            )

            weighted_score += (
                average * weight
            )

            total_weight += weight

        if total_weight == ZERO:
            return ZERO

        return round_percent(
            weighted_score / total_weight
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    @staticmethod
    def find_strongest(
        kpis: Mapping[str, KPIResult],
        limit: int = 5,
    ) -> list[str]:
        """Return strongest KPI names."""
        ranking = {
            KPIHealth.EXCELLENT: 5,
            KPIHealth.GOOD: 4,
            KPIHealth.HEALTHY: 3,
            KPIHealth.WARNING: 2,
            KPIHealth.CRITICAL: 1,
            KPIHealth.NEGATIVE: 0,
            KPIHealth.UNKNOWN: 0,
        }

        ordered = sorted(
            kpis.items(),
            key=lambda item: (
                ranking[item[1].health],
                item[1].value,
            ),
            reverse=True,
        )

        return [
            name
            for name, _ in ordered[:limit]
        ]

    @staticmethod
    def find_weakest(
        kpis: Mapping[str, KPIResult],
        limit: int = 5,
    ) -> list[str]:
        """Return weakest KPI names."""
        ranking = {
            KPIHealth.EXCELLENT: 5,
            KPIHealth.GOOD: 4,
            KPIHealth.HEALTHY: 3,
            KPIHealth.WARNING: 2,
            KPIHealth.CRITICAL: 1,
            KPIHealth.NEGATIVE: 0,
            KPIHealth.UNKNOWN: 0,
        }

        ordered = sorted(
            kpis.items(),
            key=lambda item: (
                ranking[item[1].health],
                item[1].value,
            ),
        )

        return [
            name
            for name, _ in ordered[:limit]
        ]

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    @staticmethod
    def generate_warnings(
        kpis: Mapping[str, KPIResult],
    ) -> list[str]:
        """Generate human-readable KPI warnings."""
        warnings: list[str] = []

        for result in kpis.values():
            if result.health == KPIHealth.CRITICAL:
                warnings.append(
                    f"{result.name} is in a critical condition."
                )

            elif result.health == KPIHealth.NEGATIVE:
                warnings.append(
                    f"{result.name} is negative and requires attention."
                )

            elif result.health == KPIHealth.WARNING:
                warnings.append(
                    f"{result.name} is showing a warning condition."
                )

            if result.trend == KPITrend.DETERIORATING:
                warnings.append(
                    f"{result.name} is deteriorating versus the previous period."
                )

        return list(dict.fromkeys(warnings))

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def generate_recommendations(
        kpis: Mapping[str, KPIResult],
    ) -> list[str]:
        """Generate initial rule-based recommendations."""
        recommendations: list[str] = []

        current_ratio = kpis.get(
            "current_ratio"
        )

        if (
            current_ratio
            and current_ratio.health == KPIHealth.CRITICAL
        ):
            recommendations.append(
                "Review short-term liquidity and prioritize "
                "cash preservation."
            )

        dso = kpis.get(
            "days_sales_outstanding"
        )

        if (
            dso
            and dso.health == KPIHealth.CRITICAL
        ):
            recommendations.append(
                "Investigate overdue receivables and accelerate "
                "customer collections."
            )

        debt_to_equity = kpis.get(
            "debt_to_equity"
        )

        if (
            debt_to_equity
            and debt_to_equity.health
            == KPIHealth.CRITICAL
        ):
            recommendations.append(
                "Review leverage and evaluate debt-reduction options."
            )

        net_margin = kpis.get(
            "net_margin"
        )

        if (
            net_margin
            and net_margin.health
            in {
                KPIHealth.CRITICAL,
                KPIHealth.NEGATIVE,
            }
        ):
            recommendations.append(
                "Investigate margin compression through pricing, "
                "cost, and product-level analysis."
            )

        expense_growth = kpis.get(
            "expense_growth"
        )

        if (
            expense_growth
            and expense_growth.health
            == KPIHealth.CRITICAL
        ):
            recommendations.append(
                "Perform detailed operating-expense analysis "
                "and identify cost-reduction opportunities."
            )

        free_cash_flow = kpis.get(
            "free_cash_flow"
        )

        if (
            free_cash_flow
            and free_cash_flow.health
            == KPIHealth.CRITICAL
        ):
            recommendations.append(
                "Review capital expenditure and operating cash "
                "generation immediately."
            )

        return recommendations

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def generate_summary(
        kpis: Mapping[str, KPIResult],
        overall_health: KPIHealth,
        overall_score: Decimal,
    ) -> str:
        """Generate a concise KPI summary."""
        if not kpis:
            return "No KPIs could be calculated."

        critical = sum(
            1
            for result in kpis.values()
            if result.health
            in {
                KPIHealth.CRITICAL,
                KPIHealth.NEGATIVE,
            }
        )

        warnings = sum(
            1
            for result in kpis.values()
            if result.health
            == KPIHealth.WARNING
        )

        return (
            f"Financial KPI health is "
            f"{overall_health.value.replace('_', ' ')} "
            f"with a score of {overall_score:.2f}/100. "
            f"{critical} KPI(s) are critical and "
            f"{warnings} KPI(s) require attention."
        )

    # ------------------------------------------------------------------
    # Input Normalization
    # ------------------------------------------------------------------

    def normalize_input(
        self,
        data: FinancialKPIInput | Mapping[str, Any] | None,
    ) -> FinancialKPIInput:
        """Normalize dictionary-based data into FinancialKPIInput."""
        if data is None:
            return FinancialKPIInput()

        if isinstance(
            data,
            FinancialKPIInput,
        ):
            return data

        if not isinstance(data, Mapping):
            raise TypeError(
                "KPI input must be FinancialKPIInput "
                "or a mapping."
            )

        period = data.get("period")

        if isinstance(period, str):
            try:
                period = datetime.fromisoformat(
                    period
                ).date()
            except ValueError:
                period = None

        return FinancialKPIInput(
            period=period,
            currency=data.get("currency"),
            revenue=to_decimal(
                data.get("revenue")
            ),
            cost_of_goods_sold=to_decimal(
                data.get("cost_of_goods_sold")
                or data.get("cogs")
            ),
            gross_profit=(
                to_decimal(
                    data.get("gross_profit")
                )
                if data.get("gross_profit") is not None
                else None
            ),
            operating_expenses=to_decimal(
                data.get("operating_expenses")
                or data.get("opex")
            ),
            operating_profit=(
                to_decimal(
                    data.get("operating_profit")
                )
                if data.get("operating_profit") is not None
                else None
            ),
            net_profit=to_decimal(
                data.get("net_profit")
                or data.get("net_income")
            ),
            interest_expense=to_decimal(
                data.get("interest_expense")
            ),
            tax_expense=to_decimal(
                data.get("tax_expense")
            ),
            total_assets=to_decimal(
                data.get("total_assets")
                or data.get("assets")
            ),
            current_assets=to_decimal(
                data.get("current_assets")
            ),
            total_liabilities=to_decimal(
                data.get("total_liabilities")
                or data.get("liabilities")
            ),
            current_liabilities=to_decimal(
                data.get("current_liabilities")
            ),
            equity=to_decimal(
                data.get("equity")
                or data.get("total_equity")
            ),
            debt=to_decimal(
                data.get("debt")
                or data.get("total_debt")
            ),
            cash=to_decimal(
                data.get("cash")
                or data.get("cash_and_equivalents")
            ),
            accounts_receivable=to_decimal(
                data.get("accounts_receivable")
                or data.get("receivables")
            ),
            inventory=to_decimal(
                data.get("inventory")
            ),
            accounts_payable=to_decimal(
                data.get("accounts_payable")
                or data.get("payables")
            ),
            operating_cash_flow=to_decimal(
                data.get("operating_cash_flow")
                or data.get("cash_from_operations")
            ),
            investing_cash_flow=to_decimal(
                data.get("investing_cash_flow")
            ),
            financing_cash_flow=to_decimal(
                data.get("financing_cash_flow")
            ),
            capital_expenditure=to_decimal(
                data.get("capital_expenditure")
                or data.get("capex")
            ),
            customers=to_decimal(
                data.get("customers")
            ),
            employees=to_decimal(
                data.get("employees")
            ),
            orders=to_decimal(
                data.get("orders")
            ),
            units_sold=to_decimal(
                data.get("units_sold")
            ),
            average_inventory=(
                to_decimal(
                    data.get("average_inventory")
                )
                if data.get("average_inventory")
                is not None
                else None
            ),
            average_receivables=(
                to_decimal(
                    data.get("average_receivables")
                )
                if data.get("average_receivables")
                is not None
                else None
            ),
            average_payables=(
                to_decimal(
                    data.get("average_payables")
                )
                if data.get("average_payables")
                is not None
                else None
            ),
            budget_revenue=(
                to_decimal(
                    data.get("budget_revenue")
                )
                if data.get("budget_revenue")
                is not None
                else None
            ),
            budget_profit=(
                to_decimal(
                    data.get("budget_profit")
                )
                if data.get("budget_profit")
                is not None
                else None
            ),
            budget_expenses=(
                to_decimal(
                    data.get("budget_expenses")
                )
                if data.get("budget_expenses")
                is not None
                else None
            ),
            metadata=dict(
                data.get("metadata") or {}
            ),
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def calculate_kpis(
    data: FinancialKPIInput | Mapping[str, Any],
    *,
    previous: FinancialKPIInput | Mapping[str, Any] | None = None,
) -> KPIAnalysisResult:
    """Calculate the complete KPI set."""
    analyzer = KPIAnalyzer()

    return analyzer.analyze(
        data,
        previous=previous,
    )


def calculate_profitability_kpis(
    data: FinancialKPIInput | Mapping[str, Any],
) -> dict[str, KPIResult]:
    """Calculate profitability-focused KPIs."""
    analyzer = KPIAnalyzer()
    normalized = analyzer.normalize_input(data)
    normalized.derive()

    results = {
        "gross_margin": analyzer.gross_margin(
            normalized
        ),
        "operating_margin": analyzer.operating_margin(
            normalized
        ),
        "net_margin": analyzer.net_margin(
            normalized
        ),
        "roa": analyzer.roa(
            normalized
        ),
        "roe": analyzer.roe(
            normalized
        ),
        "roic": analyzer.roic(
            normalized
        ),
    }

    return {
        name: result
        for name, result in results.items()
        if result is not None
    }


def calculate_liquidity_kpis(
    data: FinancialKPIInput | Mapping[str, Any],
) -> dict[str, KPIResult]:
    """Calculate liquidity-focused KPIs."""
    analyzer = KPIAnalyzer()
    normalized = analyzer.normalize_input(data)
    normalized.derive()

    results = {
        "current_ratio": analyzer.current_ratio(
            normalized
        ),
        "quick_ratio": analyzer.quick_ratio(
            normalized
        ),
        "cash_ratio": analyzer.cash_ratio(
            normalized
        ),
        "cash_conversion_cycle": (
            analyzer.cash_conversion_cycle(
                normalized
            )
        ),
    }

    return {
        name: result
        for name, result in results.items()
        if result is not None
    }


def calculate_leverage_kpis(
    data: FinancialKPIInput | Mapping[str, Any],
) -> dict[str, KPIResult]:
    """Calculate leverage-focused KPIs."""
    analyzer = KPIAnalyzer()
    normalized = analyzer.normalize_input(data)
    normalized.derive()

    results = {
        "debt_to_equity": analyzer.debt_to_equity(
            normalized
        ),
        "debt_ratio": analyzer.debt_ratio(
            normalized
        ),
        "interest_coverage": analyzer.interest_coverage(
            normalized
        ),
    }

    return {
        name: result
        for name, result in results.items()
        if result is not None
    }


def calculate_cash_flow_kpis(
    data: FinancialKPIInput | Mapping[str, Any],
) -> dict[str, KPIResult]:
    """Calculate cash-flow-focused KPIs."""
    analyzer = KPIAnalyzer()
    normalized = analyzer.normalize_input(data)
    normalized.derive()

    results = {
        "operating_cash_flow_margin": (
            analyzer.operating_cash_flow_margin(
                normalized
            )
        ),
        "free_cash_flow": analyzer.free_cash_flow(
            normalized
        ),
        "cash_flow_coverage": (
            analyzer.cash_flow_coverage(
                normalized
            )
        ),
    }

    return {
        name: result
        for name, result in results.items()
        if result is not None
    }


def calculate_growth_kpis(
    current: FinancialKPIInput | Mapping[str, Any],
    previous: FinancialKPIInput | Mapping[str, Any],
) -> dict[str, KPIResult]:
    """Calculate growth KPIs between two periods."""
    analyzer = KPIAnalyzer()

    current_input = analyzer.normalize_input(
        current
    )
    previous_input = analyzer.normalize_input(
        previous
    )

    current_input.derive()
    previous_input.derive()

    return {
        "revenue_growth": analyzer.revenue_growth(
            current_input,
            previous_input,
        ),
        "profit_growth": analyzer.profit_growth(
            current_input,
            previous_input,
        ),
        "expense_growth": analyzer.expense_growth(
            current_input,
            previous_input,
        ),
    }


def calculate_current_ratio(
    current_assets: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> Decimal:
    """Calculate current ratio."""
    return round_ratio(
        safe_divide(
            to_decimal(current_assets),
            to_decimal(current_liabilities),
        )
    )


def calculate_quick_ratio(
    cash: Decimal | float | int,
    receivables: Decimal | float | int,
    current_liabilities: Decimal | float | int,
) -> Decimal:
    """Calculate quick ratio."""
    return round_ratio(
        safe_divide(
            to_decimal(cash)
            + to_decimal(receivables),
            to_decimal(current_liabilities),
        )
    )


def calculate_debt_to_equity(
    debt: Decimal | float | int,
    equity: Decimal | float | int,
) -> Decimal:
    """Calculate debt-to-equity ratio."""
    return round_ratio(
        safe_divide(
            to_decimal(debt),
            to_decimal(equity),
        )
    )


def calculate_net_margin(
    net_profit: Decimal | float | int,
    revenue: Decimal | float | int,
) -> Decimal:
    """Calculate net profit margin."""
    return percentage(
        to_decimal(net_profit),
        to_decimal(revenue),
    )


def calculate_gross_margin(
    gross_profit: Decimal | float | int,
    revenue: Decimal | float | int,
) -> Decimal:
    """Calculate gross margin."""
    return percentage(
        to_decimal(gross_profit),
        to_decimal(revenue),
    )


def calculate_operating_margin(
    operating_profit: Decimal | float | int,
    revenue: Decimal | float | int,
) -> Decimal:
    """Calculate operating margin."""
    return percentage(
        to_decimal(operating_profit),
        to_decimal(revenue),
    )


def calculate_revenue_growth(
    current_revenue: Decimal | float | int,
    previous_revenue: Decimal | float | int,
) -> Decimal:
    """Calculate revenue growth."""
    return percentage_change(
        to_decimal(current_revenue),
        to_decimal(previous_revenue),
    )


# ============================================================================
# Module Exports
# ============================================================================


__all__ = [
    # Enums
    "KPICategory",
    "KPIHealth",
    "KPITrend",
    "KPISeverity",

    # Models
    "FinancialKPIInput",
    "KPIResult",
    "KPIAnalysisResult",

    # Engine
    "KPIAnalyzer",

    # Utility functions
    "to_decimal",
    "safe_divide",
    "percentage",
    "percentage_change",

    # Main convenience API
    "calculate_kpis",
    "calculate_profitability_kpis",
    "calculate_liquidity_kpis",
    "calculate_leverage_kpis",
    "calculate_cash_flow_kpis",
    "calculate_growth_kpis",

    # Individual KPI helpers
    "calculate_current_ratio",
    "calculate_quick_ratio",
    "calculate_debt_to_equity",
    "calculate_net_margin",
    "calculate_gross_margin",
    "calculate_operating_margin",
    "calculate_revenue_growth",
]