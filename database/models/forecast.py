"""
backend/app/api/forecast.py

Forecasting API for FinCo AI.

Responsibilities
----------------
- Revenue forecasting
- Profit forecasting
- Cash-flow forecasting
- Liquidity forecasting
- Generic metric forecasting
- Forecast model selection
- Forecast evaluation
- Forecast summaries
- Tenant isolation
- Authorization

Architecture
------------

    Client
       |
       v
    Forecast API
       |
       +---- Authentication
       +---- Authorization
       +---- Tenant Isolation
       |
       v
    ForecastingService
       |
       +---- Preprocessing
       +---- Feature Engineering
       +---- Baseline Model
       +---- ML Model
       +---- Evaluation
       |
       v
    Financial Data / Historical Data
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
    prefix="/forecast",
    tags=["Forecasting"],
)


# ============================================================================
# TYPES
# ============================================================================

ForecastMetric = Literal[
    "revenue",
    "profit",
    "cashflow",
    "liquidity",
]

ForecastGranularity = Literal[
    "daily",
    "weekly",
    "monthly",
    "quarterly",
]

ForecastModel = Literal[
    "auto",
    "baseline",
    "linear_regression",
    "random_forest",
    "xgboost",
    "prophet",
]

ForecastStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
]

ConfidenceLevel = Literal[
    "80",
    "90",
    "95",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class ForecastRequest(BaseModel):
    """
    Generic financial forecast request.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    metric: ForecastMetric

    start_date: date

    end_date: date

    horizon: int = Field(
        default=12,
        ge=1,
        le=365,
    )

    granularity: ForecastGranularity = "monthly"

    model: ForecastModel = "auto"

    confidence_level: ConfidenceLevel = "95"

    include_history: bool = True

    include_drivers: bool = True

    include_risk: bool = True

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


class RevenueForecastRequest(BaseModel):
    """
    Revenue forecast request.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    start_date: date

    end_date: date

    horizon: int = Field(
        default=12,
        ge=1,
        le=365,
    )

    granularity: ForecastGranularity = "monthly"

    model: ForecastModel = "auto"

    confidence_level: ConfidenceLevel = "95"

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    include_history: bool = True

    include_seasonality: bool = True

    include_drivers: bool = True

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


class ProfitForecastRequest(BaseModel):
    """
    Profit forecast request.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    start_date: date

    end_date: date

    horizon: int = Field(
        default=12,
        ge=1,
        le=365,
    )

    granularity: ForecastGranularity = "monthly"

    model: ForecastModel = "auto"

    confidence_level: ConfidenceLevel = "95"

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    include_revenue: bool = True

    include_expenses: bool = True

    include_history: bool = True

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


class CashFlowForecastRequest(BaseModel):
    """
    Cash-flow forecast request.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    start_date: date

    end_date: date

    horizon: int = Field(
        default=12,
        ge=1,
        le=365,
    )

    granularity: ForecastGranularity = "monthly"

    model: ForecastModel = "auto"

    confidence_level: ConfidenceLevel = "95"

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    include_history: bool = True

    include_inflows: bool = True

    include_outflows: bool = True

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


class LiquidityForecastRequest(BaseModel):
    """
    Liquidity forecast request.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    start_date: date

    end_date: date

    horizon: int = Field(
        default=12,
        ge=1,
        le=365,
    )

    granularity: ForecastGranularity = "monthly"

    model: ForecastModel = "auto"

    confidence_level: ConfidenceLevel = "95"

    currency: str | None = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    minimum_cash_threshold: float = Field(
        default=0,
        ge=0,
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str | None,
    ) -> str | None:
        return value.upper() if value else value


class ForecastEvaluationRequest(BaseModel):
    """
    Evaluate a forecasting model against historical data.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str | None = None

    metric: ForecastMetric

    start_date: date

    end_date: date

    model: ForecastModel = "auto"

    granularity: ForecastGranularity = "monthly"

    test_size: int = Field(
        default=3,
        ge=1,
        le=24,
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class ForecastPoint(BaseModel):
    """
    Single forecast point.
    """

    period: str

    predicted_value: float

    lower_bound: float | None = None

    upper_bound: float | None = None

    confidence_level: float | None = None

    is_forecast: bool = True


class HistoricalPoint(BaseModel):
    """
    Historical value used for forecasting.
    """

    period: str

    actual_value: float


class ForecastDriver(BaseModel):
    """
    Driver contributing to forecast movement.
    """

    name: str

    impact: Literal[
        "positive",
        "negative",
        "neutral",
    ]

    contribution_percent: float | None = None

    explanation: str


class ForecastRisk(BaseModel):
    """
    Forecast risk signal.
    """

    name: str

    severity: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ]

    probability: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    explanation: str


class ForecastMetrics(BaseModel):
    """
    Model evaluation metrics.
    """

    mae: float | None = None

    mse: float | None = None

    rmse: float | None = None

    mape: float | None = None

    smape: float | None = None

    r2: float | None = None


class ForecastResponse(BaseModel):
    """
    Generic forecast response.
    """

    company_id: str

    metric: ForecastMetric

    currency: str

    status: ForecastStatus

    model: str

    granularity: ForecastGranularity

    historical: list[HistoricalPoint] = Field(
        default_factory=list,
    )

    forecast: list[ForecastPoint]

    drivers: list[ForecastDriver] = Field(
        default_factory=list,
    )

    risks: list[ForecastRisk] = Field(
        default_factory=list,
    )

    evaluation: ForecastMetrics | None = None

    total_forecast: float | None = None

    average_forecast: float | None = None

    growth_percent: float | None = None

    generated_at: datetime


class RevenueForecastResponse(ForecastResponse):
    """
    Revenue-specific forecast.
    """

    metric: Literal["revenue"] = "revenue"

    recurring_revenue: float | None = None

    non_recurring_revenue: float | None = None


class ProfitForecastResponse(ForecastResponse):
    """
    Profit-specific forecast.
    """

    metric: Literal["profit"] = "profit"

    gross_profit_forecast: float | None = None

    operating_profit_forecast: float | None = None

    net_profit_forecast: float | None = None

    margin_forecast: float | None = None


class CashFlowForecastResponse(ForecastResponse):
    """
    Cash-flow-specific forecast.
    """

    metric: Literal["cashflow"] = "cashflow"

    operating_cash_flow: float | None = None

    investing_cash_flow: float | None = None

    financing_cash_flow: float | None = None

    ending_cash_balance: float | None = None


class LiquidityForecastResponse(ForecastResponse):
    """
    Liquidity forecast.
    """

    metric: Literal["liquidity"] = "liquidity"

    minimum_cash_balance: float | None = None

    minimum_cash_threshold: float | None = None

    liquidity_gap: float | None = None

    runway_months: float | None = None

    liquidity_risk: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ] | None = None


class ForecastEvaluationResponse(BaseModel):
    """
    Forecast evaluation response.
    """

    company_id: str

    metric: ForecastMetric

    model: str

    granularity: ForecastGranularity

    training_start_date: date

    training_end_date: date

    metrics: ForecastMetrics

    actual_values: list[float] = Field(
        default_factory=list,
    )

    predicted_values: list[float] = Field(
        default_factory=list,
    )

    model_rank: int | None = None

    evaluated_at: datetime


class ForecastSummaryResponse(BaseModel):
    """
    Executive forecasting summary.
    """

    company_id: str

    currency: str

    forecast_horizon: int

    revenue_growth_percent: float | None = None

    profit_growth_percent: float | None = None

    cashflow_growth_percent: float | None = None

    ending_cash_balance: float | None = None

    minimum_cash_balance: float | None = None

    runway_months: float | None = None

    highest_risk: str | None = None

    key_drivers: list[ForecastDriver] = Field(
        default_factory=list,
    )

    key_risks: list[ForecastRisk] = Field(
        default_factory=list,
    )

    recommendations: list[str] = Field(
        default_factory=list,
    )

    generated_at: datetime


class ForecastHealthResponse(BaseModel):
    """
    Forecasting subsystem health.
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


def get_forecasting_service() -> Any:
    """
    Preferred service:

        backend.app.forecasting.service.ForecastingService

    Fallback:

        backend.app.forecasting.model.ForecastingModel
    """

    try:
        from backend.app.forecasting.service import (
            ForecastingService,
        )

        return ForecastingService()

    except ImportError:
        pass

    try:
        from backend.app.forecasting.model import (
            ForecastingModel,
        )

        return ForecastingModel()

    except ImportError as exc:
        logger.exception(
            "Forecasting service unavailable."
        )

        raise RuntimeError(
            "Forecasting service is not configured."
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


def _can_view_forecasts(
    current_user: Any,
) -> bool:
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
            "Unauthorized forecast access: "
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


def require_forecast_access(
    current_user: Any,
) -> None:
    if not _can_view_forecasts(
        current_user
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Forecast access is required."
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

    if (
        end_date.year - start_date.year
    ) > 20:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Historical forecast period cannot "
                "exceed 20 years."
            ),
        )


def validate_horizon(
    horizon: int,
    granularity: ForecastGranularity,
) -> None:
    """
    Protect the forecasting engine from unreasonable horizons.
    """

    limits = {
        "daily": 365,
        "weekly": 260,
        "monthly": 120,
        "quarterly": 40,
    }

    maximum = limits[granularity]

    if horizon > maximum:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"{granularity} forecast horizon cannot "
                f"exceed {maximum} periods."
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
    Support both synchronous and asynchronous services.
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
                f"Forecasting service does not implement "
                f"'{method_name}'."
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
            "Invalid forecasting request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid forecasting request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Forecasting operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forecasting operation failed.",
        ) from exc


# ============================================================================
# GENERIC FORECAST
# ============================================================================


@router.post(
    "/generate",
    response_model=ForecastResponse,
    summary="Generate financial forecast",
)
async def generate_forecast(
    request: ForecastRequest,
    current_user: Any = Depends(get_current_user()),
) -> ForecastResponse:
    """
    Generate a forecast for revenue, profit, cash flow,
    or liquidity.
    """

    require_forecast_access(
        current_user
    )

    company_id = _get_company_id(
        current_user
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

    validate_horizon(
        request.horizon,
        request.granularity,
    )

    service = get_forecasting_service()

    result = await _call_service(
        service,
        "forecast",
        company_id=company_id,
        metric=request.metric,
        start_date=request.start_date,
        end_date=request.end_date,
        horizon=request.horizon,
        granularity=request.granularity,
        model=request.model,
        confidence_level=int(
            request.confidence_level
        ),
        currency=request.currency,
        include_history=request.include_history,
        include_drivers=request.include_drivers,
        include_risk=request.include_risk,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Forecast could not be generated.",
        )

    return ForecastResponse.model_validate(
        result
    )


# ============================================================================
# REVENUE FORECAST
# ============================================================================


@router.post(
    "/revenue",
    response_model=RevenueForecastResponse,
    summary="Forecast revenue",
)
async def revenue_forecast(
    request: RevenueForecastRequest,
    current_user: Any = Depends(get_current_user()),
) -> RevenueForecastResponse:
    """
    Forecast future revenue.
    """

    require_forecast_access(
        current_user
    )

    company_id = _get_company_id(
        current_user
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

    validate_horizon(
        request.horizon,
        request.granularity,
    )

    service = get_forecasting_service()

    result = await _call_service(
        service,
        "revenue_forecast",
        company_id=company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        horizon=request.horizon,
        granularity=request.granularity,
        model=request.model,
        confidence_level=int(
            request.confidence_level
        ),
        currency=request.currency,
        include_history=request.include_history,
        include_seasonality=request.include_seasonality,
        include_drivers=request.include_drivers,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Revenue forecast unavailable.",
        )

    return RevenueForecastResponse.model_validate(
        result
    )


# ============================================================================
# PROFIT FORECAST
# ============================================================================


@router.post(
    "/profit",
    response_model=ProfitForecastResponse,
    summary="Forecast profit",
)
async def profit_forecast(
    request: ProfitForecastRequest,
    current_user: Any = Depends(get_current_user()),
) -> ProfitForecastResponse:
    """
    Forecast gross, operating, and net profit.
    """

    require_forecast_access(
        current_user
    )

    company_id = _get_company_id(
        current_user
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

    validate_horizon(
        request.horizon,
        request.granularity,
    )

    service = get_forecasting_service()

    result = await _call_service(
        service,
        "profit_forecast",
        company_id=company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        horizon=request.horizon,
        granularity=request.granularity,
        model=request.model,
        confidence_level=int(
            request.confidence_level
        ),
        currency=request.currency,
        include_revenue=request.include_revenue,
        include_expenses=request.include_expenses,
        include_history=request.include_history,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Profit forecast unavailable.",
        )

    return ProfitForecastResponse.model_validate(
        result
    )


# ============================================================================
# CASH FLOW FORECAST
# ============================================================================


@router.post(
    "/cash-flow",
    response_model=CashFlowForecastResponse,
    summary="Forecast cash flow",
)
async def cash_flow_forecast(
    request: CashFlowForecastRequest,
    current_user: Any = Depends(get_current_user()),
) -> CashFlowForecastResponse:
    """
    Forecast future cash inflows, outflows, and balances.
    """

    require_forecast_access(
        current_user
    )

    company_id = _get_company_id(
        current_user
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

    validate_horizon(
        request.horizon,
        request.granularity,
    )

    service = get_forecasting_service()

    result = await _call_service(
        service,
        "cashflow_forecast",
        company_id=company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        horizon=request.horizon,
        granularity=request.granularity,
        model=request.model,
        confidence_level=int(
            request.confidence_level
        ),
        currency=request.currency,
        include_history=request.include_history,
        include_inflows=request.include_inflows,
        include_outflows=request.include_outflows,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cash-flow forecast unavailable.",
        )

    return CashFlowForecastResponse.model_validate(
        result
    )


# ============================================================================
# LIQUIDITY FORECAST
# ============================================================================


@router.post(
    "/liquidity",
    response_model=LiquidityForecastResponse,
    summary="Forecast liquidity",
)
async def liquidity_forecast(
    request: LiquidityForecastRequest,
    current_user: Any = Depends(get_current_user()),
) -> LiquidityForecastResponse:
    """
    Forecast cash liquidity, runway, and liquidity risk.
    """

    require_forecast_access(
        current_user
    )

    company_id = _get_company_id(
        current_user
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

    validate_horizon(
        request.horizon,
        request.granularity,
    )

    service = get_forecasting_service()

    result = await _call_service(
        service,
        "liquidity_forecast",
        company_id=company_id,
        start_date=request.start_date,
        end_date=request.end_date,
        horizon=request.horizon,
        granularity=request.granularity,
        model=request.model,
        confidence_level=int(
            request.confidence_level
        ),
        currency=request.currency,
        minimum_cash_threshold=(
            request.minimum_cash_threshold
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Liquidity forecast unavailable.",
        )

    return LiquidityForecastResponse.model_validate(
        result
    )


# ============================================================================
# EVALUATION
# ============================================================================


@router.post(
    "/evaluate",
    response_model=ForecastEvaluationResponse,
    summary="Evaluate forecasting model",
)
async def evaluate_forecast(
    request: ForecastEvaluationRequest,
    current_user: Any = Depends(get_current_user()),
) -> ForecastEvaluationResponse:
    """
    Evaluate forecasting accuracy using historical data.
    """

    require_forecast_access(
        current_user
    )

    company_id = _get_company_id(
        current_user
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

    service = get_forecasting_service()

    result = await _call_service(
        service,
        "evaluate",
        company_id=company_id,
        metric=request.metric,
        start_date=request.start_date,
        end_date=request.end_date,
        model=request.model,
        granularity=request.granularity,
        test_size=request.test_size,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Forecast evaluation unavailable.",
        )

    return ForecastEvaluationResponse.model_validate(
        result
    )


# ============================================================================
# SUMMARY
# ============================================================================


@router.get(
    "/summary/{company_id}",
    response_model=ForecastSummaryResponse,
    summary="Get forecast summary",
)
async def forecast_summary(
    company_id: str,
    start_date: date,
    end_date: date,
    horizon: int = Query(
        default=12,
        ge=1,
        le=120,
    ),
    granularity: ForecastGranularity = "monthly",
    current_user: Any = Depends(get_current_user()),
) -> ForecastSummaryResponse:
    """
    Generate an executive-level forecasting summary.
    """

    require_forecast_access(
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

    validate_horizon(
        horizon,
        granularity,
    )

    service = get_forecasting_service()

    result = await _call_service(
        service,
        "summary",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        granularity=granularity,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Forecast summary unavailable.",
        )

    return ForecastSummaryResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=ForecastHealthResponse,
    summary="Forecasting service health",
)
async def forecast_health() -> ForecastHealthResponse:
    """
    Forecasting subsystem health check.
    """

    started = time.perf_counter()

    try:
        service = get_forecasting_service()

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

        return ForecastHealthResponse(
            status="healthy",
            service="forecast",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Forecasting health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Forecasting service is unavailable."
            ),
        )