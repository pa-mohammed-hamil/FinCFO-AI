"""
FinCo AI - Forecast Schemas

Pydantic schemas for:

- Revenue forecasting
- Profit forecasting
- Cash-flow forecasting
- Liquidity forecasting
- Forecast requests/responses
- Time-series observations
- Forecast intervals
- Model configuration
- Model evaluation
- Forecast comparison
- Forecast dashboard
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# ENUMS
# ============================================================

class ForecastType(str, Enum):
    """Types of financial forecasts."""

    REVENUE = "revenue"
    PROFIT = "profit"
    CASH_FLOW = "cash_flow"
    LIQUIDITY = "liquidity"


class ForecastFrequency(str, Enum):
    """Forecast frequency."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class ForecastModelType(str, Enum):
    """Supported forecasting model families."""

    BASELINE = "baseline"
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ARIMA = "arima"
    SARIMA = "sarima"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class ForecastStatus(str, Enum):
    """Forecast execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# FORECAST BASE
# ============================================================

class ForecastBase(BaseModel):
    """Common forecast configuration."""

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )

    company_id: int = Field(
        ...,
        gt=0,
        description="Company being forecast.",
    )

    forecast_type: ForecastType

    frequency: ForecastFrequency = ForecastFrequency.MONTHLY

    horizon: int = Field(
        ...,
        ge=1,
        le=120,
        description="Number of future periods.",
    )


# ============================================================
# FORECAST REQUEST
# ============================================================

class ForecastRequest(ForecastBase):
    """Request to generate a financial forecast."""

    start_date: Optional[date] = None

    end_date: Optional[date] = None

    model_type: ForecastModelType = ForecastModelType.ENSEMBLE

    confidence_level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
    )

    include_confidence_interval: bool = True

    include_historical: bool = True

    include_explanation: bool = True

    include_risk_analysis: bool = True


# ============================================================
# HISTORICAL OBSERVATION
# ============================================================

class HistoricalObservation(BaseModel):
    """Historical time-series observation."""

    date: date

    value: Decimal

    label: Optional[str] = None

    metadata: Dict[str, str] = Field(
        default_factory=dict,
    )


# ============================================================
# FORECAST POINT
# ============================================================

class ForecastPoint(BaseModel):
    """Single future forecast value."""

    date: date

    predicted_value: Decimal

    lower_bound: Optional[Decimal] = None

    upper_bound: Optional[Decimal] = None

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @field_validator("upper_bound")
    @classmethod
    def validate_upper_bound(cls, value, info):
        lower = info.data.get("lower_bound")

        if value is not None and lower is not None and value < lower:
            raise ValueError(
                "upper_bound must be greater than or equal to lower_bound"
            )

        return value


# ============================================================
# FORECAST METRICS
# ============================================================

class ForecastMetrics(BaseModel):
    """Forecast model evaluation metrics."""

    mae: Optional[float] = Field(
        default=None,
        ge=0,
    )

    mse: Optional[float] = Field(
        default=None,
        ge=0,
    )

    rmse: Optional[float] = Field(
        default=None,
        ge=0,
    )

    mape: Optional[float] = Field(
        default=None,
        ge=0,
    )

    smape: Optional[float] = Field(
        default=None,
        ge=0,
    )

    r2: Optional[float] = None

    mase: Optional[float] = Field(
        default=None,
        ge=0,
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

class ForecastModelConfig(BaseModel):
    """Forecast model configuration."""

    model_type: ForecastModelType

    parameters: Dict[str, object] = Field(
        default_factory=dict,
    )

    features: List[str] = Field(
        default_factory=list,
    )

    training_window: Optional[int] = Field(
        default=None,
        ge=1,
    )

    validation_split: Optional[float] = Field(
        default=None,
        gt=0.0,
        lt=1.0,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

class ForecastModelEvaluation(BaseModel):
    """Evaluation result for a forecasting model."""

    model_type: ForecastModelType

    metrics: ForecastMetrics

    training_samples: int = Field(
        default=0,
        ge=0,
    )

    validation_samples: int = Field(
        default=0,
        ge=0,
    )

    training_time_seconds: Optional[float] = Field(
        default=None,
        ge=0,
    )

    best_model: bool = False


# ============================================================
# FORECAST RESPONSE
# ============================================================

class ForecastResponse(ForecastBase):
    """Complete forecast response."""

    model_config = ConfigDict(
        use_enum_values=True,
        from_attributes=True,
    )

    id: Optional[int] = None

    currency: str = "USD"

    model_type: ForecastModelType

    status: ForecastStatus

    generated_at: datetime

    historical_data: List[HistoricalObservation] = Field(
        default_factory=list,
    )

    predictions: List[ForecastPoint] = Field(
        default_factory=list,
    )

    metrics: Optional[ForecastMetrics] = None

    model_config_details: Optional[ForecastModelConfig] = None

    explanation: Optional[str] = None

    error: Optional[str] = None


# ============================================================
# REVENUE FORECAST
# ============================================================

class RevenueForecastRequest(ForecastRequest):
    """Revenue-specific forecast request."""

    forecast_type: ForecastType = ForecastType.REVENUE

    include_growth_analysis: bool = True

    include_seasonality: bool = True


class RevenueForecastResponse(ForecastResponse):
    """Revenue forecast result."""

    forecast_type: ForecastType = ForecastType.REVENUE

    historical_revenue: Optional[Decimal] = None

    forecasted_revenue: Optional[Decimal] = None

    expected_growth_rate: Optional[float] = None

    growth_trend: Optional[str] = None


# ============================================================
# PROFIT FORECAST
# ============================================================

class ProfitForecastRequest(ForecastRequest):
    """Profit-specific forecast request."""

    forecast_type: ForecastType = ForecastType.PROFIT

    include_margin_analysis: bool = True

    include_cost_impact: bool = True


class ProfitForecastResponse(ForecastResponse):
    """Profit forecast result."""

    forecast_type: ForecastType = ForecastType.PROFIT

    historical_profit: Optional[Decimal] = None

    forecasted_profit: Optional[Decimal] = None

    expected_margin: Optional[float] = None

    profit_growth_rate: Optional[float] = None

    loss_probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# CASH FLOW FORECAST
# ============================================================

class CashFlowForecastRequest(ForecastRequest):
    """Cash-flow forecast request."""

    forecast_type: ForecastType = ForecastType.CASH_FLOW

    include_burn_rate: bool = True

    include_runway: bool = True


class CashFlowForecastResponse(ForecastResponse):
    """Cash-flow forecast result."""

    forecast_type: ForecastType = ForecastType.CASH_FLOW

    current_cash: Optional[Decimal] = None

    forecasted_cash: Optional[Decimal] = None

    cash_burn_rate: Optional[float] = None

    expected_net_cash_flow: Optional[Decimal] = None

    runway_months: Optional[float] = None

    cash_shortage_probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# LIQUIDITY FORECAST
# ============================================================

class LiquidityForecastRequest(ForecastRequest):
    """Liquidity forecast request."""

    forecast_type: ForecastType = ForecastType.LIQUIDITY

    include_current_ratio: bool = True

    include_quick_ratio: bool = True

    include_liquidity_risk: bool = True


class LiquidityForecastResponse(ForecastResponse):
    """Liquidity forecast result."""

    forecast_type: ForecastType = ForecastType.LIQUIDITY

    current_liquidity: Optional[Decimal] = None

    forecasted_liquidity: Optional[Decimal] = None

    current_ratio: Optional[float] = None

    forecasted_current_ratio: Optional[float] = None

    liquidity_risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )

    liquidity_status: Optional[str] = None


# ============================================================
# FORECAST COMPARISON
# ============================================================

class ForecastComparison(BaseModel):
    """Compare forecast against previous period."""

    metric: str

    current_value: Decimal

    previous_value: Decimal

    forecast_value: Decimal

    historical_change_percentage: Optional[float] = None

    forecast_change_percentage: Optional[float] = None

    direction: str

    interpretation: Optional[str] = None


# ============================================================
# FORECAST RISK
# ============================================================

class ForecastRisk(BaseModel):
    """Risk derived from financial forecast."""

    risk_type: str

    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    probability: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    severity: str

    description: str

    affected_period: Optional[str] = None

    mitigation: Optional[str] = None


# ============================================================
# FORECAST EXPLANATION
# ============================================================

class ForecastExplanation(BaseModel):
    """Human-readable explanation of a forecast."""

    summary: str

    key_drivers: List[str] = Field(
        default_factory=list,
    )

    positive_factors: List[str] = Field(
        default_factory=list,
    )

    negative_factors: List[str] = Field(
        default_factory=list,
    )

    assumptions: List[str] = Field(
        default_factory=list,
    )

    limitations: List[str] = Field(
        default_factory=list,
    )


# ============================================================
# FORECAST DASHBOARD
# ============================================================

class ForecastDashboardResponse(BaseModel):
    """Executive forecasting dashboard."""

    company_id: int

    currency: str = "USD"

    revenue_forecast: Optional[RevenueForecastResponse] = None

    profit_forecast: Optional[ProfitForecastResponse] = None

    cash_flow_forecast: Optional[CashFlowForecastResponse] = None

    liquidity_forecast: Optional[LiquidityForecastResponse] = None

    risks: List[ForecastRisk] = Field(
        default_factory=list,
    )

    comparisons: List[ForecastComparison] = Field(
        default_factory=list,
    )

    overall_forecast_status: Optional[str] = None

    generated_at: datetime


# ============================================================
# FORECAST JOB
# ============================================================

class ForecastJobResponse(BaseModel):
    """Background forecast job status."""

    job_id: str

    company_id: int

    forecast_type: ForecastType

    status: ForecastStatus

    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    started_at: Optional[datetime] = None

    completed_at: Optional[datetime] = None

    error: Optional[str] = None


# ============================================================
# FORECAST SUMMARY
# ============================================================

class ForecastSummary(BaseModel):
    """Compact forecast summary for alerts and agents."""

    company_id: int

    forecast_type: ForecastType

    forecast_period: str

    current_value: Optional[Decimal] = None

    forecast_value: Optional[Decimal] = None

    growth_rate: Optional[float] = None

    risk_score: Optional[float] = None

    loss_probability: Optional[float] = None

    confidence: Optional[float] = None

    trend: Optional[str] = None


# ============================================================
# FORECAST ANALYSIS
# ============================================================

class ForecastAnalysisResponse(BaseModel):
    """Complete forecast intelligence response."""

    company_id: int

    currency: str

    forecasts: List[ForecastSummary] = Field(
        default_factory=list,
    )

    risks: List[ForecastRisk] = Field(
        default_factory=list,
    )

    explanations: List[ForecastExplanation] = Field(
        default_factory=list,
    )

    recommendations: List[str] = Field(
        default_factory=list,
    )

    generated_at: datetime