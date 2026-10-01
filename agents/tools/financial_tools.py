"""
FinCo AI - Financial Agent Tools

Path:
    backend/app/agents/tools/financial_tools.py

Purpose:
    Agent-facing tools for financial analysis.

Architecture:

    Financial / Supervisor Agent
                |
                v
         financial_tools.py
                |
        +-------+--------+---------+---------+
        |       |        |         |         |
        v       v        v         v         v
      Revenue Expenses   P&L   Balance   Cash Flow
        |       |        |         |         |
        +-------+--------+---------+---------+
                        |
                        v
                Financial Services

Responsibilities:
    - Revenue analysis
    - Expense analysis
    - Profit & Loss analysis
    - Balance-sheet analysis
    - Cash-flow analysis
    - Financial ratios
    - KPI analysis
    - Profitability analysis
    - Margin analysis
    - Cost analysis
    - Budget variance
    - Historical comparison
    - Financial health
    - Root-cause analysis

Important:
    This module is an agent-tool boundary.

    It does NOT:
        - Execute arbitrary SQL
        - Let an LLM perform uncontrolled calculations
        - Access database sessions directly
        - Generate financial advice without analysis
        - Replace the financial domain services
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


# ============================================================================
# Exceptions
# ============================================================================


class FinancialToolError(Exception):
    """Base exception for financial agent tools."""


class InvalidFinancialToolInputError(
    FinancialToolError
):
    """Raised when financial tool input is invalid."""


class FinancialToolExecutionError(
    FinancialToolError
):
    """Raised when an underlying financial service fails."""


class FinancialToolConfigurationError(
    FinancialToolError
):
    """Raised when a required financial service is unavailable."""


# ============================================================================
# Constants
# ============================================================================


DEFAULT_LIMIT = 100

MAX_LIMIT = 1000


# ============================================================================
# Result
# ============================================================================


class FinancialToolResult:
    """
    Standard response envelope returned to agents.
    """

    def __init__(
        self,
        *,
        success: bool,
        operation: str,
        message: str,
        data: Any = None,
        error: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> None:

        self.success = bool(success)

        self.operation = operation

        self.message = message

        self.data = data

        self.error = error

        self.created_at = (
            datetime.now(
                timezone.utc
            )
        )

        self.metadata = dict(
            metadata or {}
        )

    def to_dict(self) -> dict[str, Any]:

        return {
            "success":
                self.success,

            "operation":
                self.operation,

            "message":
                self.message,

            "data":
                _serialize(self.data),

            "error":
                self.error,

            "created_at":
                self.created_at.isoformat(),

            "metadata":
                _serialize(
                    self.metadata
                ),
        }


# ============================================================================
# Financial Tools
# ============================================================================


class FinancialTools:
    """
    Agent-facing financial analysis toolkit.

    All domain logic is delegated to injected financial services.

    Example:

        tools = FinancialTools(
            revenue_service=revenue_service,
            expense_service=expense_service,
            pnl_service=pnl_service,
            balance_sheet_service=balance_sheet_service,
            cash_flow_service=cash_flow_service,
            ratio_service=ratio_service,
            kpi_service=kpi_service,
            profitability_service=profitability_service,
            margin_service=margin_service,
            cost_service=cost_service,
            budget_variance_service=budget_variance_service,
            historical_service=historical_service,
            comparison_service=comparison_service,
            health_score_service=health_score_service,
            root_cause_service=root_cause_service,
        )
    """

    def __init__(
        self,
        *,
        revenue_service: Any = None,
        expense_service: Any = None,
        pnl_service: Any = None,
        balance_sheet_service: Any = None,
        cash_flow_service: Any = None,
        ratio_service: Any = None,
        kpi_service: Any = None,
        profitability_service: Any = None,
        margin_service: Any = None,
        cost_service: Any = None,
        budget_variance_service: Any = None,
        historical_service: Any = None,
        comparison_service: Any = None,
        health_score_service: Any = None,
        root_cause_service: Any = None,
        financial_service: Any = None,
        audit_service: Any = None,
    ) -> None:

        self.revenue_service = revenue_service

        self.expense_service = expense_service

        self.pnl_service = pnl_service

        self.balance_sheet_service = (
            balance_sheet_service
        )

        self.cash_flow_service = (
            cash_flow_service
        )

        self.ratio_service = ratio_service

        self.kpi_service = kpi_service

        self.profitability_service = (
            profitability_service
        )

        self.margin_service = margin_service

        self.cost_service = cost_service

        self.budget_variance_service = (
            budget_variance_service
        )

        self.historical_service = (
            historical_service
        )

        self.comparison_service = (
            comparison_service
        )

        self.health_score_service = (
            health_score_service
        )

        self.root_cause_service = (
            root_cause_service
        )

        self.financial_service = (
            financial_service
        )

        self.audit_service = audit_service

    # ========================================================================
    # COMPREHENSIVE FINANCIAL ANALYSIS
    # ========================================================================

    def analyze_financials(
        self,
        *,
        company_id: str,
        financial_data: Any,
        previous_period_data: Any = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run comprehensive financial analysis.

        This is the preferred tool when an agent needs an overall
        financial picture rather than a single metric.
        """

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        payload = {
            "company_id":
                company_id,

            "financial_data":
                financial_data,

            "previous_period_data":
                previous_period_data,

            "user_id":
                user_id,
        }

        service = (
            self.financial_service
            or self.pnl_service
        )

        result = self._invoke(
            service,
            operation="analyze_financials",
            payload=payload,
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_financials",
            message=(
                "Financial analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    # ========================================================================
    # REVENUE
    # ========================================================================

    def analyze_revenue(
        self,
        *,
        company_id: str,
        financial_data: Any,
        previous_period_data: Any = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze revenue performance."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.revenue_service,
            operation="analyze_revenue",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "previous_period_data":
                    previous_period_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_revenue",
            message=(
                "Revenue analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    def calculate_revenue_growth(
        self,
        *,
        current_revenue: float,
        previous_revenue: float,
    ) -> dict[str, Any]:
        """
        Calculate revenue growth percentage.

        Formula:
            ((current - previous) / previous) * 100
        """

        current = self._number(
            current_revenue,
            "current_revenue",
        )

        previous = self._number(
            previous_revenue,
            "previous_revenue",
        )

        if previous == 0:

            raise InvalidFinancialToolInputError(
                "previous_revenue cannot be zero."
            )

        growth = (
            (
                current - previous
            )
            / previous
        ) * 100.0

        return self._success(
            operation="calculate_revenue_growth",
            message=(
                "Revenue growth calculated."
            ),
            data={
                "current_revenue":
                    current,

                "previous_revenue":
                    previous,

                "growth_percent":
                    growth,
            },
        )

    # ========================================================================
    # EXPENSES
    # ========================================================================

    def analyze_expenses(
        self,
        *,
        company_id: str,
        financial_data: Any,
        previous_period_data: Any = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze company expenses."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.expense_service,
            operation="analyze_expenses",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "previous_period_data":
                    previous_period_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_expenses",
            message=(
                "Expense analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    def calculate_expense_growth(
        self,
        *,
        current_expenses: float,
        previous_expenses: float,
    ) -> dict[str, Any]:
        """Calculate expense growth percentage."""

        current = self._number(
            current_expenses,
            "current_expenses",
        )

        previous = self._number(
            previous_expenses,
            "previous_expenses",
        )

        if previous == 0:

            raise InvalidFinancialToolInputError(
                "previous_expenses cannot be zero."
            )

        growth = (
            (
                current - previous
            )
            / previous
        ) * 100.0

        return self._success(
            operation="calculate_expense_growth",
            message=(
                "Expense growth calculated."
            ),
            data={
                "current_expenses":
                    current,

                "previous_expenses":
                    previous,

                "growth_percent":
                    growth,
            },
        )

    # ========================================================================
    # PROFIT & LOSS
    # ========================================================================

    def analyze_pnl(
        self,
        *,
        company_id: str,
        financial_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze the Profit & Loss statement."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.pnl_service,
            operation="analyze_pnl",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_pnl",
            message=(
                "P&L analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    def calculate_profit(
        self,
        *,
        revenue: float,
        expenses: float,
    ) -> dict[str, Any]:
        """
        Calculate profit.

        Formula:
            Profit = Revenue - Expenses
        """

        revenue_value = self._number(
            revenue,
            "revenue",
        )

        expense_value = self._number(
            expenses,
            "expenses",
        )

        profit = (
            revenue_value
            - expense_value
        )

        return self._success(
            operation="calculate_profit",
            message="Profit calculated.",
            data={
                "revenue":
                    revenue_value,

                "expenses":
                    expense_value,

                "profit":
                    profit,

                "is_profitable":
                    profit > 0,

                "is_loss":
                    profit < 0,
            },
        )

    # ========================================================================
    # BALANCE SHEET
    # ========================================================================

    def analyze_balance_sheet(
        self,
        *,
        company_id: str,
        balance_sheet_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze the balance sheet."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            balance_sheet_data,
            "balance_sheet_data",
        )

        result = self._invoke(
            self.balance_sheet_service,
            operation="analyze_balance_sheet",
            payload={
                "company_id":
                    company_id,

                "balance_sheet_data":
                    balance_sheet_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_balance_sheet",
            message=(
                "Balance-sheet analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    def calculate_net_assets(
        self,
        *,
        total_assets: float,
        total_liabilities: float,
    ) -> dict[str, Any]:
        """Calculate net assets/equity."""

        assets = self._number(
            total_assets,
            "total_assets",
        )

        liabilities = self._number(
            total_liabilities,
            "total_liabilities",
        )

        net_assets = (
            assets - liabilities
        )

        return self._success(
            operation="calculate_net_assets",
            message="Net assets calculated.",
            data={
                "total_assets":
                    assets,

                "total_liabilities":
                    liabilities,

                "net_assets":
                    net_assets,
            },
        )

    # ========================================================================
    # CASH FLOW
    # ========================================================================

    def analyze_cash_flow(
        self,
        *,
        company_id: str,
        cash_flow_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze company cash flow."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            cash_flow_data,
            "cash_flow_data",
        )

        result = self._invoke(
            self.cash_flow_service,
            operation="analyze_cash_flow",
            payload={
                "company_id":
                    company_id,

                "cash_flow_data":
                    cash_flow_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_cash_flow",
            message=(
                "Cash-flow analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    def calculate_net_cash_flow(
        self,
        *,
        cash_inflows: float,
        cash_outflows: float,
    ) -> dict[str, Any]:
        """Calculate net cash flow."""

        inflows = self._number(
            cash_inflows,
            "cash_inflows",
        )

        outflows = self._number(
            cash_outflows,
            "cash_outflows",
        )

        net_cash_flow = (
            inflows - outflows
        )

        return self._success(
            operation="calculate_net_cash_flow",
            message=(
                "Net cash flow calculated."
            ),
            data={
                "cash_inflows":
                    inflows,

                "cash_outflows":
                    outflows,

                "net_cash_flow":
                    net_cash_flow,

                "cash_positive":
                    net_cash_flow >= 0,
            },
        )

    # ========================================================================
    # RATIOS
    # ========================================================================

    def calculate_ratios(
        self,
        *,
        company_id: str,
        financial_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Calculate financial ratios."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.ratio_service,
            operation="calculate_ratios",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "user_id":
                    user_id,
            },
            methods=(
                "calculate",
                "calculate_ratios",
                "analyze",
                "run",
            ),
        )

        return self._success(
            operation="calculate_ratios",
            message=(
                "Financial ratios calculated."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    # ========================================================================
    # KPIs
    # ========================================================================

    def calculate_kpis(
        self,
        *,
        company_id: str,
        financial_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Calculate financial KPIs."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.kpi_service,
            operation="calculate_kpis",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "user_id":
                    user_id,
            },
            methods=(
                "calculate",
                "calculate_kpis",
                "analyze",
                "run",
            ),
        )

        return self._success(
            operation="calculate_kpis",
            message=(
                "Financial KPIs calculated."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    # ========================================================================
    # PROFITABILITY
    # ========================================================================

    def analyze_profitability(
        self,
        *,
        company_id: str,
        financial_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze profitability."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.profitability_service,
            operation="analyze_profitability",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_profitability",
            message=(
                "Profitability analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    # ========================================================================
    # MARGIN
    # ========================================================================

    def analyze_margins(
        self,
        *,
        company_id: str,
        financial_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze gross, operating and net margins."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.margin_service,
            operation="analyze_margins",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_margins",
            message=(
                "Margin analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    def calculate_margin(
        self,
        *,
        profit: float,
        revenue: float,
    ) -> dict[str, Any]:
        """Calculate profit margin percentage."""

        profit_value = self._number(
            profit,
            "profit",
        )

        revenue_value = self._number(
            revenue,
            "revenue",
        )

        if revenue_value == 0:

            raise InvalidFinancialToolInputError(
                "revenue cannot be zero."
            )

        margin = (
            profit_value
            / revenue_value
        ) * 100.0

        return self._success(
            operation="calculate_margin",
            message="Margin calculated.",
            data={
                "profit":
                    profit_value,

                "revenue":
                    revenue_value,

                "margin_percent":
                    margin,
            },
        )

    # ========================================================================
    # COST ANALYSIS
    # ========================================================================

    def analyze_costs(
        self,
        *,
        company_id: str,
        financial_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze cost structure."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.cost_service,
            operation="analyze_costs",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_costs",
            message=(
                "Cost analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    # ========================================================================
    # BUDGET VARIANCE
    # ========================================================================

    def analyze_budget_variance(
        self,
        *,
        company_id: str,
        actual_data: Any,
        budget_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze actual-vs-budget variance."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            actual_data,
            "actual_data",
        )

        self._require_data(
            budget_data,
            "budget_data",
        )

        result = self._invoke(
            self.budget_variance_service,
            operation="analyze_budget_variance",
            payload={
                "company_id":
                    company_id,

                "actual_data":
                    actual_data,

                "budget_data":
                    budget_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_budget_variance",
            message=(
                "Budget variance analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    def calculate_variance(
        self,
        *,
        actual: float,
        budget: float,
    ) -> dict[str, Any]:
        """Calculate absolute and percentage variance."""

        actual_value = self._number(
            actual,
            "actual",
        )

        budget_value = self._number(
            budget,
            "budget",
        )

        variance = (
            actual_value
            - budget_value
        )

        if budget_value != 0:

            variance_percent = (
                variance
                / abs(budget_value)
            ) * 100.0

        else:

            variance_percent = None

        return self._success(
            operation="calculate_variance",
            message="Variance calculated.",
            data={
                "actual":
                    actual_value,

                "budget":
                    budget_value,

                "variance":
                    variance,

                "variance_percent":
                    variance_percent,
            },
        )

    # ========================================================================
    # HISTORICAL ANALYSIS
    # ========================================================================

    def analyze_historical(
        self,
        *,
        company_id: str,
        financial_data: Any,
        metric: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyze historical financial performance."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.historical_service,
            operation="analyze_historical",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "metric":
                    metric,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "calculate",
                "compare",
                "run",
            ),
        )

        return self._success(
            operation="analyze_historical",
            message=(
                "Historical analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,

                "metric":
                    metric,
            },
        )

    # ========================================================================
    # PERIOD COMPARISON
    # ========================================================================

    def compare_periods(
        self,
        *,
        company_id: str,
        current_period: Any,
        previous_period: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compare current and previous financial periods."""

        self._validate_company_id(
            company_id
        )

        self._require_data(
            current_period,
            "current_period",
        )

        self._require_data(
            previous_period,
            "previous_period",
        )

        result = self._invoke(
            self.comparison_service,
            operation="compare_periods",
            payload={
                "company_id":
                    company_id,

                "current_period":
                    current_period,

                "previous_period":
                    previous_period,

                "user_id":
                    user_id,
            },
            methods=(
                "compare",
                "analyze",
                "calculate",
                "run",
            ),
        )

        return self._success(
            operation="compare_periods",
            message=(
                "Financial periods compared."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    # ========================================================================
    # FINANCIAL HEALTH
    # ========================================================================

    def calculate_health_score(
        self,
        *,
        company_id: str,
        financial_data: Any,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Calculate overall financial health score.

        The underlying health-score service owns the actual scoring
        methodology.
        """

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.health_score_service,
            operation="calculate_health_score",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "user_id":
                    user_id,
            },
            methods=(
                "calculate",
                "calculate_score",
                "assess",
                "analyze",
                "run",
            ),
        )

        return self._success(
            operation="calculate_health_score",
            message=(
                "Financial health score calculated."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,
            },
        )

    # ========================================================================
    # ROOT CAUSE
    # ========================================================================

    def analyze_root_cause(
        self,
        *,
        company_id: str,
        financial_data: Any,
        issue: Optional[str] = None,
        previous_period_data: Any = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Analyze the likely financial root cause.

        This delegates domain-specific root-cause logic to the
        financial root-cause service.
        """

        self._validate_company_id(
            company_id
        )

        self._require_data(
            financial_data,
            "financial_data",
        )

        result = self._invoke(
            self.root_cause_service,
            operation="analyze_root_cause",
            payload={
                "company_id":
                    company_id,

                "financial_data":
                    financial_data,

                "issue":
                    issue,

                "previous_period_data":
                    previous_period_data,

                "user_id":
                    user_id,
            },
            methods=(
                "analyze",
                "find_root_cause",
                "calculate",
                "assess",
                "run",
            ),
        )

        return self._success(
            operation="analyze_root_cause",
            message=(
                "Financial root-cause analysis completed."
            ),
            data=result,
            metadata={
                "company_id":
                    company_id,

                "issue":
                    issue,
            },
        )

    # ========================================================================
    # FINANCIAL METRIC EXTRACTION
    # ========================================================================

    def extract_metrics(
        self,
        *,
        financial_data: Any,
    ) -> dict[str, Any]:
        """
        Extract common financial metrics from a financial record.

        This method does not infer missing values.
        """

        self._require_data(
            financial_data,
            "financial_data",
        )

        data = _serialize(
            financial_data
        )

        if not isinstance(
            data,
            Mapping,
        ):

            raise InvalidFinancialToolInputError(
                "financial_data must be mapping-like."
            )

        metric_aliases = {
            "revenue": (
                "revenue",
                "total_revenue",
                "sales",
            ),

            "expenses": (
                "expenses",
                "total_expenses",
                "costs",
            ),

            "profit": (
                "profit",
                "net_profit",
                "net_income",
            ),

            "cash_flow": (
                "cash_flow",
                "net_cash_flow",
            ),

            "assets": (
                "assets",
                "total_assets",
            ),

            "liabilities": (
                "liabilities",
                "total_liabilities",
            ),

            "equity": (
                "equity",
                "shareholders_equity",
            ),
        }

        metrics: dict[str, Any] = {}

        for canonical_name, aliases in (
            metric_aliases.items()
        ):

            for alias in aliases:

                if alias in data:

                    metrics[
                        canonical_name
                    ] = data[alias]

                    break

        return self._success(
            operation="extract_metrics",
            message=(
                "Financial metrics extracted."
            ),
            data=metrics,
        )

    # ========================================================================
    # INTERNAL SERVICE INVOCATION
    # ========================================================================

    def _invoke(
        self,
        service: Any,
        *,
        operation: str,
        payload: Mapping[str, Any],
        methods: Sequence[str],
    ) -> Any:
        """
        Safely invoke an injected service.

        Services may expose different conventional entry points:
            analyze()
            calculate()
            assess()
            run()

        No dynamic code execution is performed.
        """

        if service is None:

            raise FinancialToolConfigurationError(
                f"No service configured for {operation}."
            )

        last_error: Optional[
            Exception
        ] = None

        for method_name in methods:

            method = getattr(
                service,
                method_name,
                None,
            )

            if method is None:
                continue

            try:

                return method(
                    **dict(payload)
                )

            except TypeError as exc:

                last_error = exc

                try:

                    return method(
                        dict(payload)
                    )

                except TypeError as second_exc:

                    last_error = second_exc

        if last_error is not None:

            raise FinancialToolExecutionError(
                (
                    f"Service invocation failed for "
                    f"{operation}: {last_error}"
                )
            ) from last_error

        raise FinancialToolConfigurationError(
            (
                f"No compatible service method found "
                f"for {operation}."
            )
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_company_id(
        company_id: str,
    ) -> None:

        if not isinstance(
            company_id,
            str,
        ):

            raise InvalidFinancialToolInputError(
                "company_id must be a string."
            )

        if not company_id.strip():

            raise InvalidFinancialToolInputError(
                "company_id cannot be empty."
            )

        if len(company_id.strip()) > 200:

            raise InvalidFinancialToolInputError(
                "company_id is too long."
            )

    @staticmethod
    def _require_data(
        data: Any,
        field_name: str,
    ) -> None:

        if data is None:

            raise InvalidFinancialToolInputError(
                f"{field_name} is required."
            )

    @staticmethod
    def _number(
        value: Any,
        field_name: str,
    ) -> float:

        if isinstance(
            value,
            bool,
        ):

            raise InvalidFinancialToolInputError(
                f"{field_name} must be numeric."
            )

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidFinancialToolInputError(
                f"{field_name} must be numeric."
            ) from exc

        if not (
            float("-inf")
            < number
            < float("inf")
        ):

            raise InvalidFinancialToolInputError(
                f"{field_name} must be finite."
            )

        return number

    # ========================================================================
    # RESPONSE
    # ========================================================================

    @staticmethod
    def _success(
        *,
        operation: str,
        message: str,
        data: Any,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> dict[str, Any]:

        return FinancialToolResult(
            success=True,
            operation=operation,
            message=message,
            data=data,
            metadata=metadata,
        ).to_dict()


# ============================================================================
# Standalone Convenience Functions
# ============================================================================


def calculate_profit(
    *,
    revenue: float,
    expenses: float,
) -> dict[str, Any]:
    """Standalone profit calculation."""

    return FinancialTools().calculate_profit(
        revenue=revenue,
        expenses=expenses,
    )


def calculate_margin(
    *,
    profit: float,
    revenue: float,
) -> dict[str, Any]:
    """Standalone margin calculation."""

    return FinancialTools().calculate_margin(
        profit=profit,
        revenue=revenue,
    )


def calculate_revenue_growth(
    *,
    current_revenue: float,
    previous_revenue: float,
) -> dict[str, Any]:
    """Standalone revenue-growth calculation."""

    return FinancialTools().calculate_revenue_growth(
        current_revenue=current_revenue,
        previous_revenue=previous_revenue,
    )


def calculate_expense_growth(
    *,
    current_expenses: float,
    previous_expenses: float,
) -> dict[str, Any]:
    """Standalone expense-growth calculation."""

    return FinancialTools().calculate_expense_growth(
        current_expenses=current_expenses,
        previous_expenses=previous_expenses,
    )


def calculate_net_cash_flow(
    *,
    cash_inflows: float,
    cash_outflows: float,
) -> dict[str, Any]:
    """Standalone net-cash-flow calculation."""

    return FinancialTools().calculate_net_cash_flow(
        cash_inflows=cash_inflows,
        cash_outflows=cash_outflows,
    )


def calculate_net_assets(
    *,
    total_assets: float,
    total_liabilities: float,
) -> dict[str, Any]:
    """Standalone net-assets calculation."""

    return FinancialTools().calculate_net_assets(
        total_assets=total_assets,
        total_liabilities=total_liabilities,
    )


def calculate_variance(
    *,
    actual: float,
    budget: float,
) -> dict[str, Any]:
    """Standalone budget-variance calculation."""

    return FinancialTools().calculate_variance(
        actual=actual,
        budget=budget,
    )


# ============================================================================
# Agent Tool Metadata
# ============================================================================


FINANCIAL_TOOL_DEFINITIONS = [
    {
        "name":
            "analyze_financials",

        "description":
            "Run comprehensive financial analysis for a company.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_revenue",

        "description":
            "Analyze revenue performance and changes.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_revenue_growth",

        "description":
            "Calculate revenue growth between two periods.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_expenses",

        "description":
            "Analyze company expenses and expense trends.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_expense_growth",

        "description":
            "Calculate expense growth between two periods.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_pnl",

        "description":
            "Analyze a company's Profit and Loss statement.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_profit",

        "description":
            "Calculate profit from revenue and expenses.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_balance_sheet",

        "description":
            "Analyze company balance-sheet data.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_net_assets",

        "description":
            "Calculate net assets from assets and liabilities.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_cash_flow",

        "description":
            "Analyze company cash-flow data.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_net_cash_flow",

        "description":
            "Calculate net cash flow from inflows and outflows.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_ratios",

        "description":
            "Calculate financial ratios.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_kpis",

        "description":
            "Calculate financial KPIs.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_profitability",

        "description":
            "Analyze profitability performance.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_margins",

        "description":
            "Analyze gross, operating and net margins.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_margin",

        "description":
            "Calculate a profit margin percentage.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_costs",

        "description":
            "Analyze company cost structure.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_budget_variance",

        "description":
            "Analyze actual versus budget performance.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_variance",

        "description":
            "Calculate actual versus budget variance.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_historical",

        "description":
            "Analyze historical financial performance.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "compare_periods",

        "description":
            "Compare financial performance across periods.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "calculate_health_score",

        "description":
            "Calculate the overall financial health score.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "analyze_root_cause",

        "description":
            "Analyze likely financial root causes.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },

    {
        "name":
            "extract_metrics",

        "description":
            "Extract common financial metrics from financial data.",

        "category":
            "financial",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
]


# ============================================================================
# Serialization
# ============================================================================


def _serialize(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        Enum,
    ):

        return value.value

    if isinstance(
        value,
        (datetime, date),
    ):

        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(key):
                _serialize(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return [
            _serialize(item)
            for item
            in value
        ]

    if hasattr(
        value,
        "model_dump",
    ):

        return _serialize(
            value.model_dump()
        )

    if hasattr(
        value,
        "to_dict",
    ):

        return _serialize(
            value.to_dict()
        )

    if hasattr(
        value,
        "__dict__",
    ):

        return {
            str(key):
                _serialize(item)
            for key, item
            in vars(value).items()
            if not key.startswith("_")
        }

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return value

    return str(value)


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "FinancialToolError",
    "InvalidFinancialToolInputError",
    "FinancialToolExecutionError",
    "FinancialToolConfigurationError",

    # Result
    "FinancialToolResult",

    # Toolkit
    "FinancialTools",

    # Convenience functions
    "calculate_profit",
    "calculate_margin",
    "calculate_revenue_growth",
    "calculate_expense_growth",
    "calculate_net_cash_flow",
    "calculate_net_assets",
    "calculate_variance",

    # Tool metadata
    "FINANCIAL_TOOL_DEFINITIONS",
]