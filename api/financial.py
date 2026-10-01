"""
FinCo AI - Financial API

Central API endpoints for financial intelligence.

Responsibilities:
    - Financial overview
    - Revenue analysis
    - Expense analysis
    - Profitability analysis
    - Financial ratios
    - KPI analysis
    - Financial health score
    - Period comparison
    - Root-cause analysis
    - Historical financial data

Architecture:

    API
      ↓
    Financial Service
      ↓
    Financial Repository
      ↓
    Database

The API layer should NOT contain complex financial calculations.
Those belong in backend/app/financial/.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.dependencies import get_current_user
from backend.app.database.repositories.financial_repository import (
    FinancialRepository,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/financial",
    tags=["Financial Intelligence"],
)


# ============================================================
# Response Schemas
# ============================================================


class FinancialOverviewResponse(BaseModel):
    """High-level financial overview."""

    model_config = ConfigDict(from_attributes=True)

    company_id: str

    period_start: date
    period_end: date

    revenue: Decimal
    expenses: Decimal
    gross_profit: Decimal
    net_profit: Decimal

    profit_margin: float
    revenue_growth: float
    expense_growth: float

    cash_balance: Decimal

    financial_health_score: float

    currency: str = "USD"


class RevenueResponse(BaseModel):
    """Revenue analysis response."""

    company_id: str

    period_start: date
    period_end: date

    total_revenue: Decimal
    previous_period_revenue: Decimal | None = None

    growth_amount: Decimal | None = None
    growth_percentage: float | None = None

    recurring_revenue: Decimal | None = None
    non_recurring_revenue: Decimal | None = None

    currency: str = "USD"


class ExpenseResponse(BaseModel):
    """Expense analysis response."""

    company_id: str

    period_start: date
    period_end: date

    total_expenses: Decimal

    previous_period_expenses: Decimal | None = None

    expense_change: Decimal | None = None
    expense_change_percentage: float | None = None

    operating_expenses: Decimal | None = None
    fixed_expenses: Decimal | None = None
    variable_expenses: Decimal | None = None

    currency: str = "USD"


class ProfitLossResponse(BaseModel):
    """Profit and loss response."""

    company_id: str

    period_start: date
    period_end: date

    revenue: Decimal
    cost_of_goods_sold: Decimal
    gross_profit: Decimal

    operating_expenses: Decimal
    operating_profit: Decimal

    interest_expense: Decimal
    taxes: Decimal

    net_profit: Decimal

    gross_margin: float
    operating_margin: float
    net_margin: float

    currency: str = "USD"


class RatioResponse(BaseModel):
    """Financial ratio response."""

    company_id: str

    period_start: date
    period_end: date

    current_ratio: float | None = None
    quick_ratio: float | None = None

    debt_to_equity: float | None = None
    debt_to_assets: float | None = None

    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None

    return_on_assets: float | None = None
    return_on_equity: float | None = None

    asset_turnover: float | None = None


class KPIResponse(BaseModel):
    """Financial KPI response."""

    company_id: str

    period_start: date
    period_end: date

    revenue: Decimal
    revenue_growth: float

    gross_profit: Decimal
    gross_margin: float

    operating_profit: Decimal
    operating_margin: float

    net_profit: Decimal
    net_margin: float

    cash_balance: Decimal
    burn_rate: Decimal | None = None

    accounts_receivable: Decimal | None = None
    accounts_payable: Decimal | None = None

    currency: str = "USD"


class FinancialHealthResponse(BaseModel):
    """Financial health score."""

    company_id: str

    score: float = Field(
        ge=0,
        le=100,
    )

    rating: str

    profitability_score: float
    liquidity_score: float
    leverage_score: float
    growth_score: float
    cashflow_score: float

    risk_level: str

    warnings: list[str] = Field(
        default_factory=list
    )

    strengths: list[str] = Field(
        default_factory=list
    )


class ComparisonResponse(BaseModel):
    """Period-over-period comparison."""

    company_id: str

    current_period: dict[str, Any]

    previous_period: dict[str, Any]

    changes: dict[str, Any]

    overall_trend: str


class RootCauseResponse(BaseModel):
    """Financial root-cause analysis."""

    company_id: str

    metric: str

    direction: str

    change_percentage: float

    primary_causes: list[str]

    contributing_factors: list[str]

    supporting_metrics: dict[str, Any]

    confidence: float = Field(
        ge=0,
        le=1,
    )


# ============================================================
# Dependency Helpers
# ============================================================


def get_financial_repository() -> FinancialRepository:
    """
    Dependency factory for financial repository.

    Replace this implementation with your application's
    dependency-injection/container strategy when the database
    layer is fully implemented.
    """

    return FinancialRepository()


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Ensure the authenticated user can access the company.

    This is intentionally kept at the API boundary.

    In production this should call the permissions/security
    layer rather than relying on client-supplied company IDs.
    """

    if not company_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="company_id cannot be empty.",
        )

    user_company_id = getattr(
        current_user,
        "company_id",
        None,
    )

    is_admin = getattr(
        current_user,
        "is_admin",
        False,
    )

    if (
        not is_admin
        and user_company_id
        and user_company_id != company_id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this company.",
        )


def validate_period(
    period_start: date,
    period_end: date,
) -> None:
    """Validate requested financial period."""

    if period_start > period_end:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="period_start must be before period_end.",
        )

    if (
        period_end.year - period_start.year > 10
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Requested financial period is too large.",
        )


# ============================================================
# Financial Overview
# ============================================================


@router.get(
    "/overview",
    response_model=FinancialOverviewResponse,
    summary="Get financial overview",
)
async def get_financial_overview(
    company_id: str = Query(
        ...,
        description="Company identifier",
    ),
    period_start: date = Query(
        ...,
        description="Start of financial period",
    ),
    period_end: date = Query(
        ...,
        description="End of financial period",
    ),
    currency: str = Query(
        "USD",
        min_length=3,
        max_length=3,
    ),
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> FinancialOverviewResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        period_start,
        period_end,
    )

    try:
        data = await repository.get_financial_overview(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )

        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No financial data found.",
            )

        return FinancialOverviewResponse(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
            revenue=data.get(
                "revenue",
                Decimal("0"),
            ),
            expenses=data.get(
                "expenses",
                Decimal("0"),
            ),
            gross_profit=data.get(
                "gross_profit",
                Decimal("0"),
            ),
            net_profit=data.get(
                "net_profit",
                Decimal("0"),
            ),
            profit_margin=float(
                data.get("profit_margin", 0)
            ),
            revenue_growth=float(
                data.get("revenue_growth", 0)
            ),
            expense_growth=float(
                data.get("expense_growth", 0)
            ),
            cash_balance=data.get(
                "cash_balance",
                Decimal("0"),
            ),
            financial_health_score=float(
                data.get(
                    "financial_health_score",
                    0,
                )
            ),
            currency=currency.upper(),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to retrieve financial overview "
            "for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve financial overview.",
        ) from exc


# ============================================================
# Revenue
# ============================================================


@router.get(
    "/revenue",
    response_model=RevenueResponse,
    summary="Analyze company revenue",
)
async def get_revenue(
    company_id: str,
    period_start: date,
    period_end: date,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> RevenueResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        period_start,
        period_end,
    )

    try:
        data = await repository.get_revenue(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )

        return RevenueResponse(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
            total_revenue=data.get(
                "total_revenue",
                Decimal("0"),
            ),
            previous_period_revenue=data.get(
                "previous_period_revenue"
            ),
            growth_amount=data.get(
                "growth_amount"
            ),
            growth_percentage=data.get(
                "growth_percentage"
            ),
            recurring_revenue=data.get(
                "recurring_revenue"
            ),
            non_recurring_revenue=data.get(
                "non_recurring_revenue"
            ),
        )

    except Exception as exc:
        logger.exception(
            "Revenue analysis failed for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to analyze revenue.",
        ) from exc


# ============================================================
# Expenses
# ============================================================


@router.get(
    "/expenses",
    response_model=ExpenseResponse,
    summary="Analyze company expenses",
)
async def get_expenses(
    company_id: str,
    period_start: date,
    period_end: date,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> ExpenseResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        period_start,
        period_end,
    )

    try:
        data = await repository.get_expenses(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )

        return ExpenseResponse(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
            total_expenses=data.get(
                "total_expenses",
                Decimal("0"),
            ),
            previous_period_expenses=data.get(
                "previous_period_expenses"
            ),
            expense_change=data.get(
                "expense_change"
            ),
            expense_change_percentage=data.get(
                "expense_change_percentage"
            ),
            operating_expenses=data.get(
                "operating_expenses"
            ),
            fixed_expenses=data.get(
                "fixed_expenses"
            ),
            variable_expenses=data.get(
                "variable_expenses"
            ),
        )

    except Exception as exc:
        logger.exception(
            "Expense analysis failed for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to analyze expenses.",
        ) from exc


# ============================================================
# Profit & Loss
# ============================================================


@router.get(
    "/pnl",
    response_model=ProfitLossResponse,
    summary="Get profit and loss analysis",
)
async def get_pnl(
    company_id: str,
    period_start: date,
    period_end: date,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> ProfitLossResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        period_start,
        period_end,
    )

    try:
        data = await repository.get_pnl(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )

        return ProfitLossResponse(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
            revenue=data.get(
                "revenue",
                Decimal("0"),
            ),
            cost_of_goods_sold=data.get(
                "cost_of_goods_sold",
                Decimal("0"),
            ),
            gross_profit=data.get(
                "gross_profit",
                Decimal("0"),
            ),
            operating_expenses=data.get(
                "operating_expenses",
                Decimal("0"),
            ),
            operating_profit=data.get(
                "operating_profit",
                Decimal("0"),
            ),
            interest_expense=data.get(
                "interest_expense",
                Decimal("0"),
            ),
            taxes=data.get(
                "taxes",
                Decimal("0"),
            ),
            net_profit=data.get(
                "net_profit",
                Decimal("0"),
            ),
            gross_margin=float(
                data.get("gross_margin", 0)
            ),
            operating_margin=float(
                data.get("operating_margin", 0)
            ),
            net_margin=float(
                data.get("net_margin", 0)
            ),
        )

    except Exception as exc:
        logger.exception(
            "P&L analysis failed for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve P&L.",
        ) from exc


# ============================================================
# Financial Ratios
# ============================================================


@router.get(
    "/ratios",
    response_model=RatioResponse,
    summary="Calculate financial ratios",
)
async def get_ratios(
    company_id: str,
    period_start: date,
    period_end: date,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> RatioResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        period_start,
        period_end,
    )

    try:
        data = await repository.get_ratios(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )

        return RatioResponse(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
            current_ratio=data.get(
                "current_ratio"
            ),
            quick_ratio=data.get(
                "quick_ratio"
            ),
            debt_to_equity=data.get(
                "debt_to_equity"
            ),
            debt_to_assets=data.get(
                "debt_to_assets"
            ),
            gross_margin=data.get(
                "gross_margin"
            ),
            operating_margin=data.get(
                "operating_margin"
            ),
            net_margin=data.get(
                "net_margin"
            ),
            return_on_assets=data.get(
                "return_on_assets"
            ),
            return_on_equity=data.get(
                "return_on_equity"
            ),
            asset_turnover=data.get(
                "asset_turnover"
            ),
        )

    except Exception as exc:
        logger.exception(
            "Ratio calculation failed for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to calculate financial ratios.",
        ) from exc


# ============================================================
# KPIs
# ============================================================


@router.get(
    "/kpis",
    response_model=KPIResponse,
    summary="Get financial KPIs",
)
async def get_kpis(
    company_id: str,
    period_start: date,
    period_end: date,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> KPIResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        period_start,
        period_end,
    )

    try:
        data = await repository.get_kpis(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )

        return KPIResponse(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
            revenue=data.get(
                "revenue",
                Decimal("0"),
            ),
            revenue_growth=float(
                data.get("revenue_growth", 0)
            ),
            gross_profit=data.get(
                "gross_profit",
                Decimal("0"),
            ),
            gross_margin=float(
                data.get("gross_margin", 0)
            ),
            operating_profit=data.get(
                "operating_profit",
                Decimal("0"),
            ),
            operating_margin=float(
                data.get("operating_margin", 0)
            ),
            net_profit=data.get(
                "net_profit",
                Decimal("0"),
            ),
            net_margin=float(
                data.get("net_margin", 0)
            ),
            cash_balance=data.get(
                "cash_balance",
                Decimal("0"),
            ),
            burn_rate=data.get(
                "burn_rate"
            ),
            accounts_receivable=data.get(
                "accounts_receivable"
            ),
            accounts_payable=data.get(
                "accounts_payable"
            ),
        )

    except Exception as exc:
        logger.exception(
            "KPI analysis failed for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve KPIs.",
        ) from exc


# ============================================================
# Financial Health
# ============================================================


@router.get(
    "/health-score",
    response_model=FinancialHealthResponse,
    summary="Calculate financial health score",
)
async def get_financial_health(
    company_id: str,
    period_start: date,
    period_end: date,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> FinancialHealthResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        period_start,
        period_end,
    )

    try:
        data = await repository.get_financial_health(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )

        return FinancialHealthResponse(
            company_id=company_id,
            score=float(
                data.get("score", 0)
            ),
            rating=str(
                data.get(
                    "rating",
                    "UNKNOWN",
                )
            ),
            profitability_score=float(
                data.get(
                    "profitability_score",
                    0,
                )
            ),
            liquidity_score=float(
                data.get(
                    "liquidity_score",
                    0,
                )
            ),
            leverage_score=float(
                data.get(
                    "leverage_score",
                    0,
                )
            ),
            growth_score=float(
                data.get(
                    "growth_score",
                    0,
                )
            ),
            cashflow_score=float(
                data.get(
                    "cashflow_score",
                    0,
                )
            ),
            risk_level=str(
                data.get(
                    "risk_level",
                    "UNKNOWN",
                )
            ),
            warnings=list(
                data.get(
                    "warnings",
                    [],
                )
            ),
            strengths=list(
                data.get(
                    "strengths",
                    [],
                )
            ),
        )

    except Exception as exc:
        logger.exception(
            "Financial health calculation failed "
            "for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to calculate financial health.",
        ) from exc


# ============================================================
# Period Comparison
# ============================================================


@router.get(
    "/comparison",
    response_model=ComparisonResponse,
    summary="Compare financial periods",
)
async def compare_periods(
    company_id: str,
    current_start: date,
    current_end: date,
    previous_start: date,
    previous_end: date,
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> ComparisonResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        current_start,
        current_end,
    )

    validate_period(
        previous_start,
        previous_end,
    )

    try:
        data = await repository.compare_periods(
            company_id=company_id,
            current_start=current_start,
            current_end=current_end,
            previous_start=previous_start,
            previous_end=previous_end,
        )

        return ComparisonResponse(
            company_id=company_id,
            current_period=data.get(
                "current_period",
                {},
            ),
            previous_period=data.get(
                "previous_period",
                {},
            ),
            changes=data.get(
                "changes",
                {},
            ),
            overall_trend=data.get(
                "overall_trend",
                "UNKNOWN",
            ),
        )

    except Exception as exc:
        logger.exception(
            "Financial comparison failed "
            "for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to compare financial periods.",
        ) from exc


# ============================================================
# Root Cause
# ============================================================


@router.get(
    "/root-cause",
    response_model=RootCauseResponse,
    summary="Analyze financial root cause",
)
async def financial_root_cause(
    company_id: str,
    metric: str = Query(
        ...,
        description=(
            "Metric such as revenue, profit, "
            "expenses, or cashflow"
        ),
    ),
    period_start: date = Query(...),
    period_end: date = Query(...),
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> RootCauseResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        period_start,
        period_end,
    )

    allowed_metrics = {
        "revenue",
        "profit",
        "net_profit",
        "expenses",
        "cashflow",
        "gross_margin",
        "net_margin",
    }

    normalized_metric = metric.lower().strip()

    if normalized_metric not in allowed_metrics:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported metric '{metric}'. "
                f"Allowed metrics: "
                f"{sorted(allowed_metrics)}"
            ),
        )

    try:
        data = await repository.get_root_cause(
            company_id=company_id,
            metric=normalized_metric,
            period_start=period_start,
            period_end=period_end,
        )

        return RootCauseResponse(
            company_id=company_id,
            metric=normalized_metric,
            direction=str(
                data.get(
                    "direction",
                    "UNKNOWN",
                )
            ),
            change_percentage=float(
                data.get(
                    "change_percentage",
                    0,
                )
            ),
            primary_causes=list(
                data.get(
                    "primary_causes",
                    [],
                )
            ),
            contributing_factors=list(
                data.get(
                    "contributing_factors",
                    [],
                )
            ),
            supporting_metrics=dict(
                data.get(
                    "supporting_metrics",
                    {},
                )
            ),
            confidence=float(
                data.get(
                    "confidence",
                    0,
                )
            ),
        )

    except Exception as exc:
        logger.exception(
            "Root-cause analysis failed "
            "for company=%s metric=%s",
            company_id,
            normalized_metric,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to perform root-cause analysis.",
        ) from exc


# ============================================================
# Historical Financial Data
# ============================================================


@router.get(
    "/historical",
    summary="Get historical financial data",
)
async def get_historical_financials(
    company_id: str,
    start_date: date,
    end_date: date,
    granularity: str = Query(
        "monthly",
        pattern="^(daily|weekly|monthly|quarterly|yearly)$",
    ),
    current_user: Any = Depends(
        get_current_user
    ),
    repository: FinancialRepository = Depends(
        get_financial_repository
    ),
) -> dict[str, Any]:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    try:
        data = await repository.get_historical_financials(
            company_id=company_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
        )

        return {
            "company_id": company_id,
            "start_date": start_date,
            "end_date": end_date,
            "granularity": granularity,
            "data": data,
        }

    except Exception as exc:
        logger.exception(
            "Historical financial retrieval failed "
            "for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve historical financial data.",
        ) from exc


# ============================================================
# Health Endpoint
# ============================================================


@router.get(
    "/health",
    summary="Financial API health check",
)
async def financial_health_check() -> dict[str, str]:

    return {
        "service": "financial-api",
        "status": "healthy",
    }
