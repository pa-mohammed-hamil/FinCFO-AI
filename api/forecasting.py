"""
FinCo AI - Forecasting API

API endpoints for financial forecasting.

Responsibilities:
    - Revenue forecasting
    - Profit forecasting
    - Cash-flow forecasting
    - Liquidity forecasting
    - Forecast evaluation
    - Forecast summary

Architecture:

    API
      ↓
    Forecasting Service
      ↓
    Forecasting Models
      ↓
    Database / Historical Financial Data

The API layer must not contain model-training logic or
financial forecasting algorithms.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.dependencies import get_current_user
from backend.app.database.repositories.financial_repository import (
    FinancialRepository,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/forecasting",
    tags=["Financial Forecasting"],
)


# ============================================================
# Types
# ============================================================

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


# ============================================================
# Request Schemas
# ============================================================


class ForecastRequest(BaseModel):
    """
    Generic financial forecast request.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    metric: ForecastMetric

    forecast_periods: int = Field(
        ...,
        ge=1,
        le=120,
        description="Number of future periods to forecast.",
    )

    granularity: ForecastGranularity = "monthly"

    model: ForecastModel = "auto"

    confidence_level: float = Field(
        0.95,
        gt=0.50,
        lt=0.999,
    )


class RevenueForecastRequest(BaseModel):
    """
    Revenue-specific forecast request.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    forecast_periods: int = Field(
        12,
        ge=1,
        le=120,
    )

    granularity: ForecastGranularity = "monthly"

    model: ForecastModel = "auto"

    confidence_level: float = Field(
        0.95,
        gt=0.50,
        lt=0.999,
    )


class ProfitForecastRequest(BaseModel):
    """
    Profit-specific forecast request.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    forecast_periods: int = Field(
        12,
        ge=1,
        le=120,
    )

    granularity: ForecastGranularity = "monthly"

    model: ForecastModel = "auto"

    confidence_level: float = Field(
        0.95,
        gt=0.50,
        lt=0.999,
    )


class CashFlowForecastRequest(BaseModel):
    """
    Cash-flow forecast request.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    forecast_periods: int = Field(
        12,
        ge=1,
        le=120,
    )

    granularity: ForecastGranularity = "monthly"

    model: ForecastModel = "auto"

    confidence_level: float = Field(
        0.95,
        gt=0.50,
        lt=0.999,
    )


# ============================================================
# Response Schemas
# ============================================================


class ForecastPoint(BaseModel):
    """
    Single forecasted observation.
    """

    period: date

    predicted_value: float

    lower_bound: float | None = None

    upper_bound: float | None = None

    confidence_level: float | None = None


class ForecastResponse(BaseModel):
    """
    Generic forecast response.
    """

    model_config = ConfigDict(
        from_attributes=True
    )

    company_id: str

    metric: str

    granularity: str

    model: str

    generated_at: str

    forecast_periods: int

    forecast: list[ForecastPoint]

    historical_periods: int

    trend: str

    growth_rate: float | None = None

    confidence: float | None = None

    warnings: list[str] = Field(
        default_factory=list
    )


class RevenueForecastResponse(BaseModel):
    """
    Revenue forecast response.
    """

    company_id: str

    model: str

    granularity: str

    forecast: list[ForecastPoint]

    expected_revenue: float

    expected_growth: float

    trend: str

    confidence: float

    warnings: list[str] = Field(
        default_factory=list
    )


class ProfitForecastResponse(BaseModel):
    """
    Profit forecast response.
    """

    company_id: str

    model: str

    granularity: str

    forecast: list[ForecastPoint]

    expected_profit: float

    expected_margin: float

    expected_growth: float

    trend: str

    confidence: float

    loss_risk: float

    warnings: list[str] = Field(
        default_factory=list
    )


class CashFlowForecastResponse(BaseModel):
    """
    Cash-flow forecast response.
    """

    company_id: str

    model: str

    granularity: str

    forecast: list[ForecastPoint]

    expected_cashflow: float

    minimum_projected_cashflow: float

    cashflow_trend: str

    confidence: float

    cash_shortage_risk: float

    warnings: list[str] = Field(
        default_factory=list
    )


class LiquidityForecastResponse(BaseModel):
    """
    Liquidity forecast response.
    """

    company_id: str

    model: str

    granularity: str

    forecast: list[ForecastPoint]

    minimum_liquidity: float

    average_liquidity: float

    liquidity_trend: str

    confidence: float

    liquidity_risk: float

    warnings: list[str] = Field(
        default_factory=list
    )


class ForecastEvaluationResponse(BaseModel):
    """
    Forecast model evaluation metrics.
    """

    company_id: str

    metric: str

    model: str

    evaluation_periods: int

    mae: float | None = None

    mse: float | None = None

    rmse: float | None = None

    mape: float | None = None

    smape: float | None = None

    r2: float | None = None

    model_rank: int | None = None


class ForecastSummaryResponse(BaseModel):
    """
    Executive-level forecast summary.
    """

    company_id: str

    revenue_forecast: dict[str, Any]

    profit_forecast: dict[str, Any]

    cashflow_forecast: dict[str, Any]

    liquidity_forecast: dict[str, Any]

    overall_outlook: str

    major_risks: list[str]

    major_opportunities: list[str]

    recommended_actions: list[str]


# ============================================================
# Dependencies
# ============================================================


def get_forecasting_service() -> Any:
    """
    Return the forecasting service.

    Replace this factory with the project's actual dependency
    injection container once forecasting services are wired.
    """

    from backend.app.forecasting.service import ForecastingService

    return ForecastingService()


def get_financial_repository() -> FinancialRepository:
    """
    Financial repository dependency.
    """

    return FinancialRepository()


# ============================================================
# Security / Validation Helpers
# ============================================================


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce company-level access control.

    In production, this should be backed by the application's
    permissions service / RBAC layer.
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


def validate_forecast_horizon(
    forecast_periods: int,
    granularity: str,
) -> None:
    """
    Protect the forecasting service from unreasonable
    forecast horizons.
    """

    limits = {
        "daily": 365,
        "weekly": 260,
        "monthly": 120,
        "quarterly": 40,
    }

    maximum = limits[granularity]

    if forecast_periods > maximum:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Maximum forecast horizon for "
                f"{granularity} data is {maximum} periods."
            ),
        )


# ============================================================
# Generic Forecast
# ============================================================


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Generate a financial forecast",
)
async def generate_forecast(
    request: ForecastRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_forecasting_service
    ),
) -> ForecastResponse:

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_forecast_horizon(
        request.forecast_periods,
        request.granularity,
    )

    try:
        result = await service.forecast(
            company_id=request.company_id,
            metric=request.metric,
            forecast_periods=request.forecast_periods,
            granularity=request.granularity,
            model=request.model,
            confidence_level=request.confidence_level,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Insufficient financial data for forecasting.",
            )

        return ForecastResponse(
            company_id=request.company_id,
            metric=request.metric,
            granularity=request.granularity,
            model=result.get(
                "model",
                request.model,
            ),
            generated_at=str(
                result.get(
                    "generated_at",
                    "",
                )
            ),
            forecast_periods=request.forecast_periods,
            forecast=[
                ForecastPoint(**point)
                for point in result.get(
                    "forecast",
                    [],
                )
            ],
            historical_periods=int(
                result.get(
                    "historical_periods",
                    0,
                )
            ),
            trend=result.get(
                "trend",
                "UNKNOWN",
            ),
            growth_rate=result.get(
                "growth_rate"
            ),
            confidence=result.get(
                "confidence"
            ),
            warnings=result.get(
                "warnings",
                [],
            ),
        )

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid forecast request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Forecast generation failed "
            "for company=%s metric=%s",
            request.company_id,
            request.metric,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate financial forecast.",
        ) from exc


# ============================================================
# Revenue Forecast
# ============================================================


@router.post(
    "/revenue",
    response_model=RevenueForecastResponse,
    summary="Forecast future revenue",
)
async def forecast_revenue(
    request: RevenueForecastRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_forecasting_service
    ),
) -> RevenueForecastResponse:

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_forecast_horizon(
        request.forecast_periods,
        request.granularity,
    )

    try:
        result = await service.forecast_revenue(
            company_id=request.company_id,
            forecast_periods=request.forecast_periods,
            granularity=request.granularity,
            model=request.model,
            confidence_level=request.confidence_level,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Insufficient revenue history for forecasting.",
            )

        return RevenueForecastResponse(
            company_id=request.company_id,
            model=result.get(
                "model",
                request.model,
            ),
            granularity=request.granularity,
            forecast=[
                ForecastPoint(**point)
                for point in result.get(
                    "forecast",
                    [],
                )
            ],
            expected_revenue=float(
                result.get(
                    "expected_revenue",
                    0,
                )
            ),
            expected_growth=float(
                result.get(
                    "expected_growth",
                    0,
                )
            ),
            trend=result.get(
                "trend",
                "UNKNOWN",
            ),
            confidence=float(
                result.get(
                    "confidence",
                    0,
                )
            ),
            warnings=result.get(
                "warnings",
                [],
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Revenue forecast failed "
            "for company=%s",
            request.company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate revenue forecast.",
        ) from exc


# ============================================================
# Profit Forecast
# ============================================================


@router.post(
    "/profit",
    response_model=ProfitForecastResponse,
    summary="Forecast future profit",
)
async def forecast_profit(
    request: ProfitForecastRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_forecasting_service
    ),
) -> ProfitForecastResponse:

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_forecast_horizon(
        request.forecast_periods,
        request.granularity,
    )

    try:
        result = await service.forecast_profit(
            company_id=request.company_id,
            forecast_periods=request.forecast_periods,
            granularity=request.granularity,
            model=request.model,
            confidence_level=request.confidence_level,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Insufficient profit history for forecasting.",
            )

        return ProfitForecastResponse(
            company_id=request.company_id,
            model=result.get(
                "model",
                request.model,
            ),
            granularity=request.granularity,
            forecast=[
                ForecastPoint(**point)
                for point in result.get(
                    "forecast",
                    [],
                )
            ],
            expected_profit=float(
                result.get(
                    "expected_profit",
                    0,
                )
            ),
            expected_margin=float(
                result.get(
                    "expected_margin",
                    0,
                )
            ),
            expected_growth=float(
                result.get(
                    "expected_growth",
                    0,
                )
            ),
            trend=result.get(
                "trend",
                "UNKNOWN",
            ),
            confidence=float(
                result.get(
                    "confidence",
                    0,
                )
            ),
            loss_risk=float(
                result.get(
                    "loss_risk",
                    0,
                )
            ),
            warnings=result.get(
                "warnings",
                [],
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Profit forecast failed "
            "for company=%s",
            request.company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate profit forecast.",
        ) from exc


# ============================================================
# Cash Flow Forecast
# ============================================================


@router.post(
    "/cashflow",
    response_model=CashFlowForecastResponse,
    summary="Forecast future cash flow",
)
async def forecast_cashflow(
    request: CashFlowForecastRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_forecasting_service
    ),
) -> CashFlowForecastResponse:

    validate_company_access(
        request.company_id,
        current_user,
    )

    validate_forecast_horizon(
        request.forecast_periods,
        request.granularity,
    )

    try:
        result = await service.forecast_cashflow(
            company_id=request.company_id,
            forecast_periods=request.forecast_periods,
            granularity=request.granularity,
            model=request.model,
            confidence_level=request.confidence_level,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Insufficient cash-flow history for forecasting.",
            )

        return CashFlowForecastResponse(
            company_id=request.company_id,
            model=result.get(
                "model",
                request.model,
            ),
            granularity=request.granularity,
            forecast=[
                ForecastPoint(**point)
                for point in result.get(
                    "forecast",
                    [],
                )
            ],
            expected_cashflow=float(
                result.get(
                    "expected_cashflow",
                    0,
                )
            ),
            minimum_projected_cashflow=float(
                result.get(
                    "minimum_projected_cashflow",
                    0,
                )
            ),
            cashflow_trend=result.get(
                "cashflow_trend",
                "UNKNOWN",
            ),
            confidence=float(
                result.get(
                    "confidence",
                    0,
                )
            ),
            cash_shortage_risk=float(
                result.get(
                    "cash_shortage_risk",
                    0,
                )
            ),
            warnings=result.get(
                "warnings",
                [],
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Cash-flow forecast failed "
            "for company=%s",
            request.company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate cash-flow forecast.",
        ) from exc


# ============================================================
# Liquidity Forecast
# ============================================================


@router.post(
    "/liquidity",
    response_model=LiquidityForecastResponse,
    summary="Forecast future liquidity",
)
async def forecast_liquidity(
    request: ForecastRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_forecasting_service
    ),
) -> LiquidityForecastResponse:

    validate_company_access(
        request.company_id,
        current_user,
    )

    if request.metric != "liquidity":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Liquidity endpoint requires "
                "metric='liquidity'."
            ),
        )

    validate_forecast_horizon(
        request.forecast_periods,
        request.granularity,
    )

    try:
        result = await service.forecast_liquidity(
            company_id=request.company_id,
            forecast_periods=request.forecast_periods,
            granularity=request.granularity,
            model=request.model,
            confidence_level=request.confidence_level,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Insufficient liquidity history for forecasting.",
            )

        return LiquidityForecastResponse(
            company_id=request.company_id,
            model=result.get(
                "model",
                request.model,
            ),
            granularity=request.granularity,
            forecast=[
                ForecastPoint(**point)
                for point in result.get(
                    "forecast",
                    [],
                )
            ],
            minimum_liquidity=float(
                result.get(
                    "minimum_liquidity",
                    0,
                )
            ),
            average_liquidity=float(
                result.get(
                    "average_liquidity",
                    0,
                )
            ),
            liquidity_trend=result.get(
                "liquidity_trend",
                "UNKNOWN",
            ),
            confidence=float(
                result.get(
                    "confidence",
                    0,
                )
            ),
            liquidity_risk=float(
                result.get(
                    "liquidity_risk",
                    0,
                )
            ),
            warnings=result.get(
                "warnings",
                [],
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Liquidity forecast failed "
            "for company=%s",
            request.company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate liquidity forecast.",
        ) from exc


# ============================================================
# Model Evaluation
# ============================================================


@router.get(
    "/evaluate",
    response_model=ForecastEvaluationResponse,
    summary="Evaluate a forecasting model",
)
async def evaluate_forecast(
    company_id: str,
    metric: ForecastMetric,
    model: ForecastModel = "auto",
    evaluation_periods: int = Query(
        12,
        ge=1,
        le=60,
    ),
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_forecasting_service
    ),
) -> ForecastEvaluationResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    try:
        result = await service.evaluate(
            company_id=company_id,
            metric=metric,
            model=model,
            evaluation_periods=evaluation_periods,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Insufficient data for model evaluation.",
            )

        return ForecastEvaluationResponse(
            company_id=company_id,
            metric=metric,
            model=result.get(
                "model",
                model,
            ),
            evaluation_periods=evaluation_periods,
            mae=result.get("mae"),
            mse=result.get("mse"),
            rmse=result.get("rmse"),
            mape=result.get("mape"),
            smape=result.get("smape"),
            r2=result.get("r2"),
            model_rank=result.get("model_rank"),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Forecast evaluation failed "
            "for company=%s metric=%s",
            company_id,
            metric,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to evaluate forecast model.",
        ) from exc


# ============================================================
# Executive Forecast Summary
# ============================================================


@router.get(
    "/summary",
    response_model=ForecastSummaryResponse,
    summary="Get executive financial forecast summary",
)
async def forecast_summary(
    company_id: str,
    forecast_periods: int = Query(
        12,
        ge=1,
        le=120,
    ),
    granularity: ForecastGranularity = "monthly",
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_forecasting_service
    ),
) -> ForecastSummaryResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    validate_forecast_horizon(
        forecast_periods,
        granularity,
    )

    try:
        result = await service.generate_summary(
            company_id=company_id,
            forecast_periods=forecast_periods,
            granularity=granularity,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unable to generate forecast summary.",
            )

        return ForecastSummaryResponse(
            company_id=company_id,
            revenue_forecast=result.get(
                "revenue_forecast",
                {},
            ),
            profit_forecast=result.get(
                "profit_forecast",
                {},
            ),
            cashflow_forecast=result.get(
                "cashflow_forecast",
                {},
            ),
            liquidity_forecast=result.get(
                "liquidity_forecast",
                {},
            ),
            overall_outlook=result.get(
                "overall_outlook",
                "UNKNOWN",
            ),
            major_risks=result.get(
                "major_risks",
                [],
            ),
            major_opportunities=result.get(
                "major_opportunities",
                [],
            ),
            recommended_actions=result.get(
                "recommended_actions",
                [],
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Forecast summary failed "
            "for company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate forecast summary.",
        ) from exc


# ============================================================
# Forecasting API Health
# ============================================================


@router.get(
    "/health",
    summary="Forecasting API health check",
)
async def forecasting_health() -> dict[str, str]:

    return {
        "service": "forecasting-api",
        "status": "healthy",
    }