"""
backend/app/api/financial_statement.py

Financial Statement API for FinCo AI.

Responsibilities
----------------
- Financial statement retrieval
- P&L / Income Statement
- Balance Sheet
- Cash Flow Statement
- Comparative statements
- Statement summaries
- Financial health signals
- Period validation
- Tenant isolation
- Authorization

Architecture
------------

    Client
       |
       v
    Financial Statement API
       |
       +---- Authentication
       +---- Authorization
       +---- Tenant Isolation
       |
       v
    FinancialStatementService
       |
       +---- FinancialRepository
       +---- Revenue Engine
       +---- Expense Engine
       +---- P&L Engine
       +---- Balance Sheet Engine
       +---- Cash Flow Engine
       +---- Ratio / KPI Engine
       |
       v
    PostgreSQL
"""

from __future__ import annotations

import inspect
import logging
import time
from datetime import date, datetime, timezone
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/financial-statements",
    tags=["Financial Statements"],
)


# ============================================================================
# TYPES
# ============================================================================

StatementType = Literal[
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "comprehensive",
]

StatementPeriod = Literal[
    "monthly",
    "quarterly",
    "yearly",
]

CashFlowType = Literal[
    "operating",
    "investing",
    "financing",
]

StatementStatus = Literal[
    "draft",
    "final",
    "restated",
    "archived",
]

CurrencyCode = str


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class FinancialStatementRequest(BaseModel):
    """
    Request a financial statement for a specific period.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    start_date: date

    end_date: date

    period: StatementPeriod = "monthly"

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    include_zero_lines: bool = False

    include_comparatives: bool = False

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


class ComparativeStatementRequest(BaseModel):
    """
    Compare two financial periods.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    current_start_date: date

    current_end_date: date

    previous_start_date: date

    previous_end_date: date

    statement_type: StatementType = "comprehensive"

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


class FinancialStatementSummaryRequest(BaseModel):
    """
    Request consolidated financial statement KPIs.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    start_date: date

    end_date: date

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class StatementLineItem(BaseModel):
    """
    Generic financial statement line.
    """

    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
    )

    code: str | None = None

    name: str

    amount: float

    percentage: float | None = None

    previous_amount: float | None = None

    variance_amount: float | None = None

    variance_percent: float | None = None

    level: int = Field(
        default=0,
        ge=0,
    )

    children: list["StatementLineItem"] = Field(
        default_factory=list,
    )


class IncomeStatementResponse(BaseModel):
    """
    P&L / Income Statement.
    """

    company_id: str

    currency: str

    period_start: date

    period_end: date

    status: StatementStatus

    revenue: float

    cost_of_goods_sold: float

    gross_profit: float

    operating_expenses: float

    operating_profit: float

    other_income: float

    other_expenses: float

    profit_before_tax: float

    tax_expense: float

    net_profit: float

    gross_margin: float

    operating_margin: float

    net_margin: float

    line_items: list[StatementLineItem] = Field(
        default_factory=list,
    )

    generated_at: datetime


class BalanceSheetResponse(BaseModel):
    """
    Balance Sheet.
    """

    company_id: str

    currency: str

    as_of_date: date

    status: StatementStatus

    total_assets: float

    current_assets: float

    non_current_assets: float

    cash_and_equivalents: float

    accounts_receivable: float

    inventory: float

    total_liabilities: float

    current_liabilities: float

    non_current_liabilities: float

    accounts_payable: float

    total_equity: float

    retained_earnings: float

    balance_check: float

    is_balanced: bool

    line_items: list[StatementLineItem] = Field(
        default_factory=list,
    )

    generated_at: datetime


class CashFlowResponse(BaseModel):
    """
    Cash Flow Statement.
    """

    company_id: str

    currency: str

    period_start: date

    period_end: date

    status: StatementStatus

    operating_cash_flow: float

    investing_cash_flow: float

    financing_cash_flow: float

    net_cash_flow: float

    beginning_cash: float

    ending_cash: float

    free_cash_flow: float

    line_items: list[StatementLineItem] = Field(
        default_factory=list,
    )

    generated_at: datetime


class FinancialStatementResponse(BaseModel):
    """
    Comprehensive financial statement.
    """

    company_id: str

    currency: str

    period_start: date

    period_end: date

    statement_type: StatementType

    status: StatementStatus

    income_statement: IncomeStatementResponse | None = None

    balance_sheet: BalanceSheetResponse | None = None

    cash_flow: CashFlowResponse | None = None

    generated_at: datetime


class ComparativeLineItem(BaseModel):
    """
    Period comparison line.
    """

    name: str

    current_amount: float

    previous_amount: float

    variance_amount: float

    variance_percent: float | None = None

    direction: Literal[
        "increase",
        "decrease",
        "unchanged",
    ]


class ComparativeStatementResponse(BaseModel):
    """
    Comparative financial statement.
    """

    company_id: str

    currency: str

    statement_type: StatementType

    current_start_date: date

    current_end_date: date

    previous_start_date: date

    previous_end_date: date

    lines: list[ComparativeLineItem]

    current_total: float

    previous_total: float

    total_variance: float

    total_variance_percent: float | None = None


class FinancialStatementSummaryResponse(BaseModel):
    """
    Consolidated financial KPIs.
    """

    company_id: str

    currency: str

    period_start: date

    period_end: date

    revenue: float

    expenses: float

    gross_profit: float

    operating_profit: float

    net_profit: float

    cash_balance: float

    total_assets: float

    total_liabilities: float

    total_equity: float

    gross_margin: float

    operating_margin: float

    net_margin: float

    current_ratio: float | None = None

    quick_ratio: float | None = None

    debt_to_equity: float | None = None

    operating_cash_flow: float

    free_cash_flow: float

    financial_health_score: float = Field(
        ge=0,
        le=100,
    )

    generated_at: datetime


class FinancialStatementHealthResponse(BaseModel):
    """
    Financial statement subsystem health.
    """

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    service: str

    timestamp: datetime

    details: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# PYDANTIC FORWARD REFERENCES
# ============================================================================

StatementLineItem.model_rebuild()


# ============================================================================
# AUTHENTICATION
# ============================================================================


def get_current_user() -> Any:
    """
    Lazily load authentication dependency.
    """

    try:
        from backend.app.dependencies import (
            get_current_user as dependency,
        )

        return dependency

    except ImportError as exc:
        logger.exception(
            "Authentication dependency unavailable."
        )

        raise RuntimeError(
            "Authentication dependency is not configured."
        ) from exc


# ============================================================================
# SERVICE
# ============================================================================


def get_financial_statement_service() -> Any:
    """
    Preferred service:

        backend.app.financial.financial_statement.FinancialStatementService

    Fallback:

        backend.app.database.repositories.financial_repository.FinancialRepository
    """

    try:
        from backend.app.financial.financial_statement import (
            FinancialStatementService,
        )

        return FinancialStatementService()

    except ImportError:
        pass

    try:
        from backend.app.database.repositories.financial_repository import (
            FinancialRepository,
        )

        return FinancialRepository()

    except ImportError as exc:
        logger.exception(
            "Financial statement service unavailable."
        )

        raise RuntimeError(
            "Financial statement service is not configured."
        ) from exc


# ============================================================================
# SECURITY
# ============================================================================


def _get_user_id(
    current_user: Any,
) -> str | None:
    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return (
            current_user.get("id")
            or current_user.get("user_id")
        )

    return (
        getattr(current_user, "id", None)
        or getattr(
            current_user,
            "user_id",
            None,
        )
    )


def _get_company_id(
    current_user: Any,
) -> str | None:
    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("company_id")

    return getattr(
        current_user,
        "company_id",
        None,
    )


def _get_role(
    current_user: Any,
) -> str | None:
    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("role")

    return getattr(
        current_user,
        "role",
        None,
    )


def _is_admin(
    current_user: Any,
) -> bool:
    if current_user is None:
        return False

    if isinstance(current_user, dict):
        return bool(
            current_user.get(
                "is_admin",
                False,
            )
        )

    return bool(
        getattr(
            current_user,
            "is_admin",
            False,
        )
    )


def _can_view_financials(
    current_user: Any,
) -> bool:
    """
    Roles allowed to view financial statements.
    """

    if _is_admin(current_user):
        return True

    return _get_role(current_user) in {
        "owner",
        "admin",
        "finance_manager",
        "analyst",
        "auditor",
        "viewer",
    }


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce tenant isolation.
    """

    if _is_admin(current_user):
        return

    user_company_id = _get_company_id(
        current_user
    )

    if not user_company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "User is not associated with a company."
            ),
        )

    if str(user_company_id) != str(company_id):
        logger.warning(
            "Unauthorized financial statement access: "
            "user_company=%s requested_company=%s "
            "user=%s",
            user_company_id,
            company_id,
            _get_user_id(current_user),
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this company."
            ),
        )


def require_financial_access(
    current_user: Any,
) -> None:
    if not _can_view_financials(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Financial statement access is required."
            ),
        )


# ============================================================================
# VALIDATION
# ============================================================================


def validate_period(
    start_date: date,
    end_date: date,
) -> None:
    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be later than end_date."
            ),
        )

    # Protect API from accidentally requesting extreme ranges.
    if (
        end_date.year - start_date.year
    ) > 20:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Financial statement period cannot "
                "exceed 20 years."
            ),
        )


def validate_comparison_periods(
    current_start: date,
    current_end: date,
    previous_start: date,
    previous_end: date,
) -> None:
    validate_period(
        current_start,
        current_end,
    )

    validate_period(
        previous_start,
        previous_end,
    )

    if (
        current_start == previous_start
        and current_end == previous_end
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Current and previous periods cannot be identical."
            ),
        )


# ============================================================================
# SERVICE CALLER
# ============================================================================


async def _call_service(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:
    """
    Support synchronous and asynchronous services.
    """

    method = getattr(
        service,
        method_name,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Financial statement service does not "
                f"implement '{method_name}'."
            ),
        )

    try:
        result = method(**kwargs)

        if inspect.isawaitable(result):
            result = await result

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid financial statement request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid financial statement request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Financial statement operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Financial statement operation failed."
            ),
        ) from exc


# ============================================================================
# INCOME STATEMENT
# ============================================================================


@router.post(
    "/income-statement",
    response_model=IncomeStatementResponse,
    summary="Get income statement",
)
async def income_statement(
    request: FinancialStatementRequest,
    current_user: Any = Depends(get_current_user()),
) -> IncomeStatementResponse:
    """
    Generate/retrieve P&L for the requested period.
    """

    require_financial_access(
        current_user
    )

    company_id = (
        _get_company_id(current_user)
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.company_id:
        validate_company_access(
            request.company_id,
            current_user,
        )

        company_id = request.company_id

    validate_period(
        request.start_date,
        request.end_date,
    )

    service = get_financial_statement_service()

    result = await _call_service(
        service,
        "income_statement",
        company_id=company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        period=request.period,
        currency=request.currency,
        include_zero_lines=request.include_zero_lines,
        include_comparatives=request.include_comparatives,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Income statement unavailable.",
        )

    return IncomeStatementResponse.model_validate(
        result
    )


# ============================================================================
# BALANCE SHEET
# ============================================================================


@router.post(
    "/balance-sheet",
    response_model=BalanceSheetResponse,
    summary="Get balance sheet",
)
async def balance_sheet(
    request: FinancialStatementRequest,
    current_user: Any = Depends(get_current_user()),
) -> BalanceSheetResponse:
    """
    Generate/retrieve Balance Sheet.

    The end_date is used as the statement's as-of date.
    """

    require_financial_access(
        current_user
    )

    company_id = (
        _get_company_id(current_user)
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.company_id:
        validate_company_access(
            request.company_id,
            current_user,
        )

        company_id = request.company_id

    validate_period(
        request.start_date,
        request.end_date,
    )

    service = get_financial_statement_service()

    result = await _call_service(
        service,
        "balance_sheet",
        company_id=company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        period=request.period,
        currency=request.currency,
        include_zero_lines=request.include_zero_lines,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Balance sheet unavailable.",
        )

    return BalanceSheetResponse.model_validate(
        result
    )


# ============================================================================
# CASH FLOW
# ============================================================================


@router.post(
    "/cash-flow",
    response_model=CashFlowResponse,
    summary="Get cash flow statement",
)
async def cash_flow(
    request: FinancialStatementRequest,
    current_user: Any = Depends(get_current_user()),
) -> CashFlowResponse:
    """
    Generate/retrieve Cash Flow Statement.
    """

    require_financial_access(
        current_user
    )

    company_id = (
        _get_company_id(current_user)
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.company_id:
        validate_company_access(
            request.company_id,
            current_user,
        )

        company_id = request.company_id

    validate_period(
        request.start_date,
        request.end_date,
    )

    service = get_financial_statement_service()

    result = await _call_service(
        service,
        "cash_flow",
        company_id=company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        period=request.period,
        currency=request.currency,
        include_zero_lines=request.include_zero_lines,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cash flow statement unavailable.",
        )

    return CashFlowResponse.model_validate(
        result
    )


# ============================================================================
# COMPREHENSIVE STATEMENT
# ============================================================================


@router.post(
    "/comprehensive",
    response_model=FinancialStatementResponse,
    summary="Get comprehensive financial statements",
)
async def comprehensive_statement(
    request: FinancialStatementRequest,
    current_user: Any = Depends(get_current_user()),
) -> FinancialStatementResponse:
    """
    Retrieve P&L + Balance Sheet + Cash Flow.
    """

    require_financial_access(
        current_user
    )

    company_id = (
        _get_company_id(current_user)
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.company_id:
        validate_company_access(
            request.company_id,
            current_user,
        )

        company_id = request.company_id

    validate_period(
        request.start_date,
        request.end_date,
    )

    service = get_financial_statement_service()

    result = await _call_service(
        service,
        "comprehensive",
        company_id=company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        period=request.period,
        currency=request.currency,
        include_zero_lines=request.include_zero_lines,
        include_comparatives=request.include_comparatives,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Comprehensive financial statement unavailable."
            ),
        )

    return FinancialStatementResponse.model_validate(
        result
    )


# ============================================================================
# COMPARISON
# ============================================================================


@router.post(
    "/compare",
    response_model=ComparativeStatementResponse,
    summary="Compare financial statements",
)
async def compare_statements(
    request: ComparativeStatementRequest,
    current_user: Any = Depends(get_current_user()),
) -> ComparativeStatementResponse:
    """
    Compare two financial periods.
    """

    require_financial_access(
        current_user
    )

    company_id = (
        _get_company_id(current_user)
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.company_id:
        validate_company_access(
            request.company_id,
            current_user,
        )

        company_id = request.company_id

    validate_comparison_periods(
        current_start=request.current_start_date,
        current_end=request.current_end_date,
        previous_start=request.previous_start_date,
        previous_end=request.previous_end_date,
    )

    service = get_financial_statement_service()

    result = await _call_service(
        service,
        "compare",
        company_id=company_id,
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Financial statement comparison unavailable."
            ),
        )

    return ComparativeStatementResponse.model_validate(
        result
    )


# ============================================================================
# SUMMARY
# ============================================================================


@router.post(
    "/summary",
    response_model=FinancialStatementSummaryResponse,
    summary="Get financial statement summary",
)
async def financial_statement_summary(
    request: FinancialStatementSummaryRequest,
    current_user: Any = Depends(get_current_user()),
) -> FinancialStatementSummaryResponse:
    """
    Return consolidated financial KPIs.

    This endpoint is intended for dashboards, executive
    reports, alerts, and the FinCo Copilot.
    """

    require_financial_access(
        current_user
    )

    company_id = (
        _get_company_id(current_user)
    )

    if not company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no company.",
        )

    if request.company_id:
        validate_company_access(
            request.company_id,
            current_user,
        )

        company_id = request.company_id

    validate_period(
        request.start_date,
        request.end_date,
    )

    service = get_financial_statement_service()

    result = await _call_service(
        service,
        "summary",
        company_id=company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        currency=request.currency,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Financial statement summary unavailable."
            ),
        )

    return FinancialStatementSummaryResponse.model_validate(
        result
    )


# ============================================================================
# QUICK GET ENDPOINTS
# ============================================================================


@router.get(
    "/income-statement/{company_id}",
    response_model=IncomeStatementResponse,
    summary="Get latest income statement",
)
async def get_income_statement(
    company_id: str,
    start_date: date,
    end_date: date,
    period: StatementPeriod = "monthly",
    current_user: Any = Depends(get_current_user()),
) -> IncomeStatementResponse:
    """
    GET convenience endpoint for P&L.
    """

    require_financial_access(
        current_user
    )

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_financial_statement_service()

    result = await _call_service(
        service,
        "income_statement",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        period=period,
        currency=None,
        include_zero_lines=False,
        include_comparatives=False,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Income statement unavailable.",
        )

    return IncomeStatementResponse.model_validate(
        result
    )


@router.get(
    "/balance-sheet/{company_id}",
    response_model=BalanceSheetResponse,
    summary="Get latest balance sheet",
)
async def get_balance_sheet(
    company_id: str,
    as_of_date: date,
    current_user: Any = Depends(get_current_user()),
) -> BalanceSheetResponse:
    """
    GET convenience endpoint for Balance Sheet.
    """

    require_financial_access(
        current_user
    )

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_financial_statement_service()

    result = await _call_service(
        service,
        "balance_sheet",
        company_id=company_id,
        start_date=as_of_date,
        end_date=as_of_date,
        period="monthly",
        currency=None,
        include_zero_lines=False,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Balance sheet unavailable.",
        )

    return BalanceSheetResponse.model_validate(
        result
    )


@router.get(
    "/cash-flow/{company_id}",
    response_model=CashFlowResponse,
    summary="Get latest cash flow statement",
)
async def get_cash_flow(
    company_id: str,
    start_date: date,
    end_date: date,
    period: StatementPeriod = "monthly",
    current_user: Any = Depends(get_current_user()),
) -> CashFlowResponse:
    """
    GET convenience endpoint for Cash Flow.
    """

    require_financial_access(
        current_user
    )

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_financial_statement_service()

    result = await _call_service(
        service,
        "cash_flow",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        period=period,
        currency=None,
        include_zero_lines=False,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cash flow statement unavailable.",
        )

    return CashFlowResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=FinancialStatementHealthResponse,
    summary="Financial statement service health",
)
async def financial_statement_health() -> (
    FinancialStatementHealthResponse
):
    """
    Financial statement subsystem health check.
    """

    started = time.perf_counter()

    try:
        service = get_financial_statement_service()

        details: dict[str, Any] = {}

        health_method = getattr(
            service,
            "health",
            None,
        )

        if health_method is not None:
            result = health_method()

            if inspect.isawaitable(result):
                result = await result

            if isinstance(result, dict):
                details = result

        details["latency_ms"] = round(
            (
                time.perf_counter() - started
            ) * 1000,
            2,
        )

        return FinancialStatementHealthResponse(
            status="healthy",
            service="financial_statement",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Financial statement health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Financial statement service is unavailable."
            ),
        )