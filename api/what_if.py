"""
backend/app/api/what_if.py

What-If / Scenario Analysis API for FinCo AI.

Responsibilities
----------------
- Scenario simulation requests
- Revenue / expense / margin / cash-flow scenarios
- Multi-variable scenario analysis
- Scenario comparison
- Sensitivity analysis
- Break-even analysis
- Scenario listing/retrieval
- Tenant isolation
- Authorization
- Health checks

Architecture
------------

    Client
       |
       v
    What-If API
       |
       +---- Authentication
       +---- Tenant Isolation
       +---- Request Validation
       |
       v
    WhatIfService
       |
       +---- Financial Engine
       +---- Forecasting
       +---- Risk Engine
       +---- Recommendation Engine
       |
       v
    Scenario Result
       |
       +---- Human Review
       +---- Audit Log
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
    prefix="/what-if",
    tags=["What-If Analysis"],
)


# ============================================================================
# TYPES
# ============================================================================

ScenarioType = Literal[
    "revenue_change",
    "expense_change",
    "price_change",
    "volume_change",
    "headcount_change",
    "marketing_spend_change",
    "operating_cost_change",
    "cash_flow_change",
    "combined",
]

ScenarioStatus = Literal[
    "draft",
    "running",
    "completed",
    "failed",
]

MetricName = Literal[
    "revenue",
    "expenses",
    "gross_profit",
    "operating_profit",
    "net_profit",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "cashflow",
    "cash_balance",
    "liquidity",
    "runway",
    "risk_score",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class ScenarioVariable(BaseModel):
    """
    A single scenario input.

    Example:
        {
            "name": "revenue_growth",
            "value": 15,
            "unit": "percent"
        }
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    value: float = Field(
        ...,
        description="Scenario value.",
    )

    unit: Literal[
        "percent",
        "absolute",
        "currency",
        "count",
        "months",
        "days",
    ] = "percent"

    baseline: float | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(
        cls,
        value: str,
    ) -> str:
        return value.strip().lower()


class WhatIfRequest(BaseModel):
    """
    Generic what-if scenario request.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    scenario_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
    )

    scenario_type: ScenarioType

    variables: list[ScenarioVariable] = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    start_date: date | None = None

    end_date: date | None = None

    forecast_months: int = Field(
        default=12,
        ge=1,
        le=120,
    )

    include_risk: bool = True

    include_forecast: bool = True

    include_recommendations: bool = True

    include_baseline: bool = True

    save_scenario: bool = False

    notes: str | None = Field(
        default=None,
        max_length=2000,
    )

    @field_validator("scenario_name")
    @classmethod
    def validate_scenario_name(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Scenario name cannot be empty."
            )

        return value

    @field_validator("end_date")
    @classmethod
    def validate_dates(
        cls,
        value: date | None,
    ) -> date | None:
        return value


class RevenueScenarioRequest(BaseModel):
    """
    Revenue-focused scenario.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    revenue_change_percent: float = Field(
        ...,
        ge=-100,
        le=1000,
    )

    forecast_months: int = Field(
        default=12,
        ge=1,
        le=120,
    )

    include_risk: bool = True

    include_recommendations: bool = True


class ExpenseScenarioRequest(BaseModel):
    """
    Expense-focused scenario.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    expense_change_percent: float = Field(
        ...,
        ge=-100,
        le=1000,
    )

    forecast_months: int = Field(
        default=12,
        ge=1,
        le=120,
    )

    include_risk: bool = True

    include_recommendations: bool = True


class PriceVolumeScenarioRequest(BaseModel):
    """
    Price-volume scenario.

    Allows independent price and volume changes.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    price_change_percent: float = Field(
        default=0,
        ge=-100,
        le=1000,
    )

    volume_change_percent: float = Field(
        default=0,
        ge=-100,
        le=1000,
    )

    forecast_months: int = Field(
        default=12,
        ge=1,
        le=120,
    )

    include_risk: bool = True

    include_recommendations: bool = True


class CombinedScenarioRequest(BaseModel):
    """
    Multi-variable scenario.

    Useful for scenarios such as:

        +10% revenue
        -5% operating expenses
        +8% marketing spend
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    variables: list[ScenarioVariable] = Field(
        ...,
        min_length=1,
        max_length=50,
    )

    forecast_months: int = Field(
        default=12,
        ge=1,
        le=120,
    )

    include_risk: bool = True

    include_forecast: bool = True

    include_recommendations: bool = True


class ScenarioComparisonRequest(BaseModel):
    """
    Compare multiple scenarios.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    scenario_ids: list[str] = Field(
        ...,
        min_length=2,
        max_length=20,
    )

    metrics: list[MetricName] = Field(
        default_factory=lambda: [
            "revenue",
            "net_profit",
            "cashflow",
            "risk_score",
        ],
        min_length=1,
        max_length=20,
    )


class SensitivityRequest(BaseModel):
    """
    Sensitivity analysis request.

    Example:
        Revenue change from -20% to +20%
        Step = 5%
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    variable: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    minimum: float = Field(
        ...,
        ge=-1000,
        le=10000,
    )

    maximum: float = Field(
        ...,
        ge=-1000,
        le=10000,
    )

    step: float = Field(
        ...,
        gt=0,
        le=1000,
    )

    metric: MetricName = "net_profit"

    forecast_months: int = Field(
        default=12,
        ge=1,
        le=120,
    )

    @field_validator("maximum")
    @classmethod
    def validate_range(
        cls,
        value: float,
        info,
    ) -> float:
        minimum = info.data.get("minimum")

        if minimum is not None and value <= minimum:
            raise ValueError(
                "maximum must be greater than minimum."
            )

        return value


class BreakEvenRequest(BaseModel):
    """
    Break-even analysis request.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str

    forecast_months: int = Field(
        default=12,
        ge=1,
        le=120,
    )

    target_profit: float = Field(
        default=0,
    )

    include_sensitivity: bool = True


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class MetricValue(BaseModel):
    """
    Baseline vs scenario metric.
    """

    metric: MetricName

    baseline: float | None = None

    scenario: float | None = None

    absolute_change: float | None = None

    percentage_change: float | None = None

    unit: str = "value"


class ScenarioPeriodResult(BaseModel):
    """
    Scenario result for one forecast period.
    """

    period: str

    metrics: list[MetricValue]


class RiskImpact(BaseModel):
    """
    Risk impact caused by scenario.
    """

    metric: str

    baseline: float | None = None

    scenario: float | None = None

    change: float | None = None

    severity: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ] = "low"

    explanation: str | None = None


class ScenarioRecommendation(BaseModel):
    """
    Recommendation generated from scenario analysis.
    """

    title: str

    description: str

    priority: Literal[
        "low",
        "medium",
        "high",
        "critical",
    ] = "medium"

    expected_impact: str | None = None

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


class WhatIfResponse(BaseModel):
    """
    Complete scenario analysis response.
    """

    id: str | None = None

    company_id: str

    scenario_name: str

    scenario_type: ScenarioType

    status: ScenarioStatus

    variables: list[ScenarioVariable]

    baseline_metrics: list[MetricValue] = Field(
        default_factory=list
    )

    scenario_metrics: list[MetricValue] = Field(
        default_factory=list
    )

    periods: list[ScenarioPeriodResult] = Field(
        default_factory=list
    )

    risk_impacts: list[RiskImpact] = Field(
        default_factory=list
    )

    recommendations: list[
        ScenarioRecommendation
    ] = Field(
        default_factory=list
    )

    key_findings: list[str] = Field(
        default_factory=list
    )

    created_at: datetime

    completed_at: datetime | None = None


class ScenarioListResponse(BaseModel):
    """
    Paginated scenario list.
    """

    company_id: str

    items: list[WhatIfResponse]

    total: int

    limit: int

    offset: int

    has_more: bool


class ScenarioComparisonResponse(BaseModel):
    """
    Scenario comparison result.
    """

    company_id: str

    scenario_ids: list[str]

    metrics: list[MetricName]

    comparison: list[
        dict[str, Any]
    ]

    best_scenario: str | None = None

    worst_scenario: str | None = None

    key_findings: list[str] = Field(
        default_factory=list
    )


class SensitivityPoint(BaseModel):
    """
    One sensitivity-analysis point.
    """

    input_value: float

    output_value: float | None = None

    metric: MetricName

    risk_score: float | None = None


class SensitivityResponse(BaseModel):
    """
    Sensitivity analysis result.
    """

    company_id: str

    variable: str

    metric: MetricName

    points: list[SensitivityPoint]

    best_value: float | None = None

    worst_value: float | None = None

    optimal_value: float | None = None

    key_findings: list[str] = Field(
        default_factory=list
    )


class BreakEvenResponse(BaseModel):
    """
    Break-even analysis.
    """

    company_id: str

    break_even_revenue: float | None = None

    break_even_units: float | None = None

    contribution_margin: float | None = None

    fixed_costs: float | None = None

    target_profit: float = 0

    margin_of_safety: float | None = None

    current_revenue: float | None = None

    months_to_break_even: float | None = None

    key_findings: list[str] = Field(
        default_factory=list
    )


class ScenarioDeleteResponse(BaseModel):
    """
    Scenario deletion response.

    Scenarios should normally be archived rather than hard deleted.
    """

    scenario_id: str

    status: Literal[
        "archived",
        "deleted",
    ]

    message: str

    updated_at: datetime


class WhatIfHealthResponse(BaseModel):
    """
    What-if service health.
    """

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    service: str

    timestamp: datetime

    details: dict[str, Any] = Field(
        default_factory=dict
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


def get_what_if_service() -> Any:
    """
    Load WhatIfService lazily.

    Expected implementation:

        backend.app.what_if.what_if_service.WhatIfService
    """

    try:
        from backend.app.what_if.what_if_service import (
            WhatIfService,
        )

        return WhatIfService()

    except ImportError as exc:
        logger.exception(
            "WhatIfService unavailable."
        )

        raise RuntimeError(
            "What-if service is not configured."
        ) from exc


# ============================================================================
# AUTH HELPERS
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


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Mandatory tenant isolation.
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

    if user_company_id != company_id:
        logger.warning(
            "Unauthorized what-if access: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this company."
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
    Invoke sync/async service methods consistently.
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
                f"What-if service does not implement "
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
            "Invalid what-if request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scenario request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "What-if service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scenario analysis failed.",
        ) from exc


# ============================================================================
# GENERIC SCENARIO
# ============================================================================


@router.post(
    "/simulate",
    response_model=WhatIfResponse,
    summary="Run what-if scenario",
)
async def simulate_scenario(
    request: WhatIfRequest,
    current_user: Any = Depends(get_current_user()),
) -> WhatIfResponse:
    """
    Run a general financial scenario.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "simulate",
        request=request,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scenario simulation returned no result.",
        )

    return WhatIfResponse.model_validate(
        result
    )


# ============================================================================
# REVENUE SCENARIO
# ============================================================================


@router.post(
    "/revenue",
    response_model=WhatIfResponse,
    summary="Simulate revenue change",
)
async def revenue_scenario(
    request: RevenueScenarioRequest,
    current_user: Any = Depends(get_current_user()),
) -> WhatIfResponse:
    """
    Simulate revenue growth or decline.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "revenue_scenario",
        company_id=request.company_id,
        revenue_change_percent=(
            request.revenue_change_percent
        ),
        forecast_months=request.forecast_months,
        include_risk=request.include_risk,
        include_recommendations=(
            request.include_recommendations
        ),
        user=current_user,
    )

    return WhatIfResponse.model_validate(
        result
    )


# ============================================================================
# EXPENSE SCENARIO
# ============================================================================


@router.post(
    "/expenses",
    response_model=WhatIfResponse,
    summary="Simulate expense change",
)
async def expense_scenario(
    request: ExpenseScenarioRequest,
    current_user: Any = Depends(get_current_user()),
) -> WhatIfResponse:
    """
    Simulate expense increases or reductions.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "expense_scenario",
        company_id=request.company_id,
        expense_change_percent=(
            request.expense_change_percent
        ),
        forecast_months=request.forecast_months,
        include_risk=request.include_risk,
        include_recommendations=(
            request.include_recommendations
        ),
        user=current_user,
    )

    return WhatIfResponse.model_validate(
        result
    )


# ============================================================================
# PRICE / VOLUME
# ============================================================================


@router.post(
    "/price-volume",
    response_model=WhatIfResponse,
    summary="Simulate price and volume changes",
)
async def price_volume_scenario(
    request: PriceVolumeScenarioRequest,
    current_user: Any = Depends(get_current_user()),
) -> WhatIfResponse:
    """
    Simulate simultaneous price and volume changes.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "price_volume_scenario",
        company_id=request.company_id,
        price_change_percent=(
            request.price_change_percent
        ),
        volume_change_percent=(
            request.volume_change_percent
        ),
        forecast_months=request.forecast_months,
        include_risk=request.include_risk,
        include_recommendations=(
            request.include_recommendations
        ),
        user=current_user,
    )

    return WhatIfResponse.model_validate(
        result
    )


# ============================================================================
# COMBINED SCENARIO
# ============================================================================


@router.post(
    "/combined",
    response_model=WhatIfResponse,
    summary="Run combined scenario",
)
async def combined_scenario(
    request: CombinedScenarioRequest,
    current_user: Any = Depends(get_current_user()),
) -> WhatIfResponse:
    """
    Run a multi-variable financial scenario.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "combined_scenario",
        company_id=request.company_id,
        variables=[
            variable.model_dump()
            for variable in request.variables
        ],
        forecast_months=request.forecast_months,
        include_risk=request.include_risk,
        include_forecast=request.include_forecast,
        include_recommendations=(
            request.include_recommendations
        ),
        user=current_user,
    )

    return WhatIfResponse.model_validate(
        result
    )


# ============================================================================
# SCENARIO RETRIEVAL
# ============================================================================


@router.get(
    "/{scenario_id}",
    response_model=WhatIfResponse,
    summary="Get scenario",
)
async def get_scenario(
    scenario_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> WhatIfResponse:
    """
    Retrieve a saved scenario.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "get",
        scenario_id=scenario_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scenario not found.",
        )

    return WhatIfResponse.model_validate(
        result
    )


# ============================================================================
# LIST SCENARIOS
# ============================================================================


@router.get(
    "/company/{company_id}",
    response_model=ScenarioListResponse,
    summary="List company scenarios",
)
async def list_scenarios(
    company_id: str,
    scenario_status: ScenarioStatus | None = Query(
        default=None,
        alias="status",
    ),
    scenario_type: ScenarioType | None = Query(
        default=None,
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    current_user: Any = Depends(get_current_user()),
) -> ScenarioListResponse:
    """
    List saved scenarios for a company.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        status=scenario_status,
        scenario_type=scenario_type,
        limit=limit,
        offset=offset,
        user=current_user,
    )

    if result is None:
        return ScenarioListResponse(
            company_id=company_id,
            items=[],
            total=0,
            limit=limit,
            offset=offset,
            has_more=False,
        )

    return ScenarioListResponse.model_validate(
        result
    )


# ============================================================================
# COMPARE SCENARIOS
# ============================================================================


@router.post(
    "/compare",
    response_model=ScenarioComparisonResponse,
    summary="Compare scenarios",
)
async def compare_scenarios(
    request: ScenarioComparisonRequest,
    current_user: Any = Depends(get_current_user()),
) -> ScenarioComparisonResponse:
    """
    Compare multiple saved scenarios.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "compare",
        company_id=request.company_id,
        scenario_ids=request.scenario_ids,
        metrics=request.metrics,
        user=current_user,
    )

    return ScenarioComparisonResponse.model_validate(
        result
    )


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================


@router.post(
    "/sensitivity",
    response_model=SensitivityResponse,
    summary="Run sensitivity analysis",
)
async def sensitivity_analysis(
    request: SensitivityRequest,
    current_user: Any = Depends(get_current_user()),
) -> SensitivityResponse:
    """
    Evaluate how a financial metric changes as an input
    variable changes across a specified range.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "sensitivity",
        company_id=request.company_id,
        variable=request.variable,
        minimum=request.minimum,
        maximum=request.maximum,
        step=request.step,
        metric=request.metric,
        forecast_months=request.forecast_months,
        user=current_user,
    )

    return SensitivityResponse.model_validate(
        result
    )


# ============================================================================
# BREAK-EVEN
# ============================================================================


@router.post(
    "/break-even",
    response_model=BreakEvenResponse,
    summary="Calculate break-even scenario",
)
async def break_even_analysis(
    request: BreakEvenRequest,
    current_user: Any = Depends(get_current_user()),
) -> BreakEvenResponse:
    """
    Calculate break-even revenue/units and margin of safety.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "break_even",
        company_id=request.company_id,
        forecast_months=request.forecast_months,
        target_profit=request.target_profit,
        include_sensitivity=(
            request.include_sensitivity
        ),
        user=current_user,
    )

    return BreakEvenResponse.model_validate(
        result
    )


# ============================================================================
# ARCHIVE SCENARIO
# ============================================================================


@router.delete(
    "/{scenario_id}",
    response_model=ScenarioDeleteResponse,
    summary="Archive scenario",
)
async def archive_scenario(
    scenario_id: str,
    company_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
    current_user: Any = Depends(get_current_user()),
) -> ScenarioDeleteResponse:
    """
    Archive a scenario.

    Hard deletion is deliberately avoided because scenario
    history can be important for auditability and decision review.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_what_if_service()

    result = await _call_service(
        service,
        "archive",
        scenario_id=scenario_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scenario not found.",
        )

    if isinstance(
        result,
        ScenarioDeleteResponse,
    ):
        return result

    return ScenarioDeleteResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=WhatIfHealthResponse,
    summary="What-if service health",
)
async def what_if_health() -> WhatIfHealthResponse:
    """
    Health check for scenario-analysis infrastructure.
    """

    started = time.perf_counter()

    try:
        service = get_what_if_service()

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

        return WhatIfHealthResponse(
            status="healthy",
            service="what-if",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "What-if health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="What-if service is unavailable.",
        )