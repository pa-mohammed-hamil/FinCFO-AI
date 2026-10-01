"""
FinCo AI - Financial Schemas

Pydantic schemas for:

- Financial records
- Revenue
- Expenses
- Profit & Loss
- Balance Sheet
- Cash Flow
- Financial Ratios
- KPIs
- Profitability
- Margin Analysis
- Budget Variance
- Historical Comparison
- Financial Health
- Financial Dashboard
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# ENUMS
# ============================================================

class FinancialPeriod(str, Enum):
    """Supported reporting periods."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class FinancialStatementType(str, Enum):
    """Financial statement types."""

    PNL = "pnl"
    BALANCE_SHEET = "balance_sheet"
    CASH_FLOW = "cash_flow"


class RevenueCategory(str, Enum):
    """Revenue categories."""

    PRODUCT = "product"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    LICENSING = "licensing"
    INTEREST = "interest"
    OTHER = "other"


class ExpenseCategory(str, Enum):
    """Expense categories."""

    OPERATING = "operating"
    SALARY = "salary"
    MARKETING = "marketing"
    RENT = "rent"
    UTILITIES = "utilities"
    TECHNOLOGY = "technology"
    INTEREST = "interest"
    TAX = "tax"
    OTHER = "other"


# ============================================================
# FINANCIAL PERIOD
# ============================================================

class FinancialPeriodRequest(BaseModel):
    """Common financial reporting period."""

    start_date: date
    end_date: date

    period: FinancialPeriod = FinancialPeriod.MONTHLY

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, value: date, info):
        start_date = info.data.get("start_date")

        if start_date and value < start_date:
            raise ValueError("end_date must be greater than or equal to start_date")

        return value


# ============================================================
# FINANCIAL BASE
# ============================================================

class FinancialBase(BaseModel):
    """Common financial fields."""

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )

    company_id: int = Field(
        ...,
        gt=0,
    )

    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
    )

    fiscal_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    reporting_period: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.upper()


# ============================================================
# REVENUE
# ============================================================

class RevenueCreate(FinancialBase):
    """Create a revenue record."""

    category: RevenueCategory = RevenueCategory.OTHER

    amount: Decimal = Field(
        ...,
        ge=0,
    )

    transaction_date: date

    description: Optional[str] = Field(
        default=None,
        max_length=1000,
    )


class RevenueResponse(RevenueCreate):
    """Revenue API response."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    created_at: datetime


class RevenueSummary(BaseModel):
    """Revenue summary."""

    total_revenue: Decimal = Decimal("0")

    product_revenue: Decimal = Decimal("0")

    service_revenue: Decimal = Decimal("0")

    subscription_revenue: Decimal = Decimal("0")

    other_revenue: Decimal = Decimal("0")

    growth_rate: Optional[float] = None


# ============================================================
# EXPENSES
# ============================================================

class ExpenseCreate(FinancialBase):
    """Create an expense record."""

    category: ExpenseCategory = ExpenseCategory.OTHER

    amount: Decimal = Field(
        ...,
        ge=0,
    )

    transaction_date: date

    description: Optional[str] = Field(
        default=None,
        max_length=1000,
    )


class ExpenseResponse(ExpenseCreate):
    """Expense API response."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    created_at: datetime


class ExpenseSummary(BaseModel):
    """Expense summary."""

    total_expenses: Decimal = Decimal("0")

    operating_expenses: Decimal = Decimal("0")

    salary_expenses: Decimal = Decimal("0")

    marketing_expenses: Decimal = Decimal("0")

    technology_expenses: Decimal = Decimal("0")

    interest_expenses: Decimal = Decimal("0")

    tax_expenses: Decimal = Decimal("0")

    other_expenses: Decimal = Decimal("0")

    growth_rate: Optional[float] = None


# ============================================================
# PROFIT & LOSS
# ============================================================

class ProfitLossCreate(FinancialBase):
    """Create a P&L statement."""

    period_start: date

    period_end: date

    revenue: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    cost_of_goods_sold: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    gross_profit: Decimal = Decimal("0")

    operating_expenses: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    operating_profit: Decimal = Decimal("0")

    interest_expense: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    taxes: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    net_profit: Decimal = Decimal("0")


class ProfitLossResponse(ProfitLossCreate):
    """P&L response."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    gross_margin: Optional[float] = None

    operating_margin: Optional[float] = None

    net_margin: Optional[float] = None

    created_at: datetime


# ============================================================
# BALANCE SHEET
# ============================================================

class BalanceSheetCreate(FinancialBase):
    """Create a balance sheet."""

    as_of_date: date

    cash: Decimal = Field(default=Decimal("0"), ge=0)

    accounts_receivable: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    inventory: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    other_current_assets: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    total_current_assets: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    property_plant_equipment: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    total_assets: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    accounts_payable: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    short_term_debt: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    long_term_debt: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    total_liabilities: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    shareholders_equity: Decimal = Decimal("0")


class BalanceSheetResponse(BalanceSheetCreate):
    """Balance sheet response."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    debt_to_equity: Optional[float] = None

    current_ratio: Optional[float] = None

    created_at: datetime


# ============================================================
# CASH FLOW
# ============================================================

class CashFlowCreate(FinancialBase):
    """Create cash flow statement."""

    period_start: date

    period_end: date

    operating_cash_flow: Decimal = Decimal("0")

    investing_cash_flow: Decimal = Decimal("0")

    financing_cash_flow: Decimal = Decimal("0")

    net_cash_flow: Decimal = Decimal("0")

    beginning_cash: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )

    ending_cash: Decimal = Field(
        default=Decimal("0"),
        ge=0,
    )


class CashFlowResponse(CashFlowCreate):
    """Cash flow response."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    cash_burn_rate: Optional[float] = None

    runway_months: Optional[float] = None

    created_at: datetime


# ============================================================
# FINANCIAL RATIOS
# ============================================================

class FinancialRatios(BaseModel):
    """Calculated financial ratios."""

    current_ratio: Optional[float] = None

    quick_ratio: Optional[float] = None

    debt_to_equity: Optional[float] = None

    debt_to_assets: Optional[float] = None

    gross_margin: Optional[float] = None

    operating_margin: Optional[float] = None

    net_margin: Optional[float] = None

    return_on_assets: Optional[float] = None

    return_on_equity: Optional[float] = None

    asset_turnover: Optional[float] = None

    inventory_turnover: Optional[float] = None

    receivables_turnover: Optional[float] = None


# ============================================================
# FINANCIAL KPIs
# ============================================================

class FinancialKPIs(BaseModel):
    """Executive financial KPIs."""

    revenue: Decimal = Decimal("0")

    revenue_growth: Optional[float] = None

    gross_profit: Decimal = Decimal("0")

    gross_margin: Optional[float] = None

    operating_profit: Decimal = Decimal("0")

    operating_margin: Optional[float] = None

    net_profit: Decimal = Decimal("0")

    net_margin: Optional[float] = None

    total_expenses: Decimal = Decimal("0")

    cash_balance: Decimal = Decimal("0")

    total_assets: Decimal = Decimal("0")

    total_liabilities: Decimal = Decimal("0")

    total_equity: Decimal = Decimal("0")

    current_ratio: Optional[float] = None

    debt_to_equity: Optional[float] = None

    cash_burn_rate: Optional[float] = None

    runway_months: Optional[float] = None


# ============================================================
# PROFITABILITY
# ============================================================

class ProfitabilityAnalysis(BaseModel):
    """Profitability analysis."""

    revenue: Decimal

    gross_profit: Decimal

    operating_profit: Decimal

    net_profit: Decimal

    gross_margin: Optional[float] = None

    operating_margin: Optional[float] = None

    net_margin: Optional[float] = None

    profit_growth: Optional[float] = None

    profitability_status: str


# ============================================================
# MARGIN ANALYSIS
# ============================================================

class MarginAnalysis(BaseModel):
    """Margin trend analysis."""

    gross_margin: Optional[float] = None

    operating_margin: Optional[float] = None

    net_margin: Optional[float] = None

    previous_gross_margin: Optional[float] = None

    previous_operating_margin: Optional[float] = None

    previous_net_margin: Optional[float] = None

    gross_margin_change: Optional[float] = None

    operating_margin_change: Optional[float] = None

    net_margin_change: Optional[float] = None

    interpretation: Optional[str] = None


# ============================================================
# BUDGET VARIANCE
# ============================================================

class BudgetVariance(BaseModel):
    """Budget vs actual analysis."""

    category: str

    budget_amount: Decimal

    actual_amount: Decimal

    variance_amount: Decimal

    variance_percentage: Optional[float] = None

    status: str

    explanation: Optional[str] = None


class BudgetVarianceSummary(BaseModel):
    """Overall budget variance."""

    total_budget: Decimal = Decimal("0")

    total_actual: Decimal = Decimal("0")

    total_variance: Decimal = Decimal("0")

    variance_percentage: Optional[float] = None

    favorable: bool = True

    items: List[BudgetVariance] = Field(
        default_factory=list,
    )


# ============================================================
# HISTORICAL COMPARISON
# ============================================================

class FinancialPeriodValue(BaseModel):
    """Metric value for a specific period."""

    period: str

    value: Decimal


class HistoricalComparison(BaseModel):
    """Historical financial comparison."""

    metric: str

    current_period: FinancialPeriodValue

    previous_period: FinancialPeriodValue

    absolute_change: Decimal

    percentage_change: Optional[float] = None

    direction: str

    interpretation: Optional[str] = None


# ============================================================
# FINANCIAL HEALTH
# ============================================================

class FinancialHealthScore(BaseModel):
    """Overall financial health score."""

    score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    rating: str

    profitability_score: Optional[float] = None

    liquidity_score: Optional[float] = None

    leverage_score: Optional[float] = None

    growth_score: Optional[float] = None

    cashflow_score: Optional[float] = None

    risk_score: Optional[float] = None

    explanation: Optional[str] = None


# ============================================================
# FINANCIAL ANALYSIS
# ============================================================

class FinancialAnalysisResponse(BaseModel):
    """Complete financial analysis."""

    company_id: int

    period_start: date

    period_end: date

    currency: str

    kpis: FinancialKPIs

    ratios: FinancialRatios

    profitability: ProfitabilityAnalysis

    margins: MarginAnalysis

    health: FinancialHealthScore

    revenue_summary: Optional[RevenueSummary] = None

    expense_summary: Optional[ExpenseSummary] = None

    cash_flow: Optional[CashFlowResponse] = None

    comparisons: List[HistoricalComparison] = Field(
        default_factory=list,
    )

    key_findings: List[str] = Field(
        default_factory=list,
    )


# ============================================================
# FINANCIAL DASHBOARD
# ============================================================

class FinancialDashboardResponse(BaseModel):
    """Executive financial dashboard response."""

    company_id: int

    company_name: Optional[str] = None

    currency: str

    reporting_period: Optional[str] = None

    kpis: FinancialKPIs

    health: FinancialHealthScore

    ratios: FinancialRatios

    revenue: Optional[RevenueSummary] = None

    expenses: Optional[ExpenseSummary] = None

    pnl: Optional[ProfitLossResponse] = None

    balance_sheet: Optional[BalanceSheetResponse] = None

    cash_flow: Optional[CashFlowResponse] = None

    active_alerts: int = 0

    critical_alerts: int = 0

    recommendations_count: int = 0

    last_updated: Optional[datetime] = None


# ============================================================
# FINANCIAL CALCULATION REQUEST
# ============================================================

class FinancialCalculationRequest(BaseModel):
    """Request for financial calculations."""

    metric: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    values: Dict[str, Decimal] = Field(
        default_factory=dict,
    )

    parameters: Dict[str, float] = Field(
        default_factory=dict,
    )


class FinancialCalculationResponse(BaseModel):
    """Financial calculation result."""

    metric: str

    result: Decimal

    unit: Optional[str] = None

    formula: Optional[str] = None

    explanation: Optional[str] = None


# ============================================================
# FINANCIAL STATEMENT RESPONSE
# ============================================================

class FinancialStatementResponse(BaseModel):
    """Generic financial statement response."""

    company_id: int

    statement_type: FinancialStatementType

    period_start: date

    period_end: date

    currency: str

    data: Dict[str, Decimal] = Field(
        default_factory=dict,
    )

    generated_at: datetime


# ============================================================
# FINANCIAL REPORT REQUEST
# ============================================================

class FinancialReportRequest(BaseModel):
    """Request to generate a financial report."""

    company_id: int = Field(
        ...,
        gt=0,
    )

    period_start: date

    period_end: date

    include_pnl: bool = True

    include_balance_sheet: bool = True

    include_cash_flow: bool = True

    include_ratios: bool = True

    include_health_score: bool = True

    include_comparison: bool = True

    include_recommendations: bool = True

    format: str = Field(
        default="pdf",
        max_length=20,
    )


class FinancialReportResponse(BaseModel):
    """Generated financial report response."""

    report_id: str

    company_id: int

    report_type: str

    format: str

    status: str

    generated_at: datetime

    download_url: Optional[str] = None