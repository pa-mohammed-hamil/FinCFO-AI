"""
FinCo AI - Forecast Agent
==========================

Agent responsible for financial forecasting workflows.

Responsibilities
----------------
- Understand and validate forecasting requests.
- Route requests to the appropriate forecasting tool/service.
- Coordinate revenue, profit, cash-flow, and liquidity forecasts.
- Compare forecasts and baselines.
- Provide structured forecast findings.
- Support what-if / scenario forecasting.
- Prepare context for the Supervisor Agent.
- Keep forecasting decisions deterministic by delegating calculations
  to the forecasting service layer.

The agent itself must NOT:
- invent forecast values,
- directly manipulate financial data,
- bypass forecasting models,
- change model thresholds,
- execute arbitrary SQL/Python,
- make irreversible financial decisions.

Expected architecture:

Supervisor Agent
       |
       v
 Forecast Agent
       |
       +----------------------+
       |                      |
       v                      v
forecast_tools.py       Forecast Services
                              |
                 +------------+------------+
                 |            |            |
                 v            v            v
              Revenue       Profit       Cash Flow
                 |            |            |
                 +------------+------------+
                              |
                              v
                       Liquidity Forecast
                              |
                              v
                       Forecast Evaluation
                              |
                              v
                     Recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


# ============================================================================
# Exceptions
# ============================================================================


class ForecastAgentError(Exception):
    """Base exception for forecast-agent failures."""


class InvalidForecastAgentInputError(ForecastAgentError):
    """Raised when agent input is invalid."""


class ForecastAgentExecutionError(ForecastAgentError):
    """Raised when a forecast operation fails."""


class ForecastAgentConfigurationError(ForecastAgentError):
    """Raised when the agent is incorrectly configured."""


class ForecastAgentAccessDeniedError(ForecastAgentError):
    """Raised when the caller is not authorized to run a forecast."""


class UnsupportedForecastOperationError(ForecastAgentError):
    """Raised when an unsupported forecasting operation is requested."""


# ============================================================================
# Constants
# ============================================================================

DEFAULT_HORIZON = 6
MAX_HORIZON = 120
DEFAULT_TOP_K = 10
MAX_TOP_K = 100

MAX_QUERY_LENGTH = 2_000
MAX_COMPANY_ID_LENGTH = 200


# ============================================================================
# Enums
# ============================================================================


class ForecastOperation(str, Enum):
    """Supported forecast-agent operations."""

    FORECAST = "forecast"
    REVENUE_FORECAST = "revenue_forecast"
    PROFIT_FORECAST = "profit_forecast"
    CASH_FLOW_FORECAST = "cash_flow_forecast"
    LIQUIDITY_FORECAST = "liquidity_forecast"
    COMPARE_FORECASTS = "compare_forecasts"
    FORECAST_ALL = "forecast_all"
    EVALUATE_FORECAST = "evaluate_forecast"
    FORECAST_RISK = "forecast_risk"
    FORECAST_SUMMARY = "forecast_summary"


class ForecastMetric(str, Enum):
    """Financial metrics supported by the forecast agent."""

    REVENUE = "revenue"
    EXPENSES = "expenses"
    PROFIT = "profit"
    CASH_FLOW = "cash_flow"
    LIQUIDITY = "liquidity"
    CASH_BALANCE = "cash_balance"


class ForecastRiskLevel(str, Enum):
    """High-level forecast risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ForecastTrend(str, Enum):
    """Forecast trend direction."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    UNKNOWN = "unknown"


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class ForecastAgentConfig:
    """
    Configuration for the forecast agent.

    Thresholds here are interpretation thresholds only.
    Actual forecasting model configuration belongs to the forecasting layer.
    """

    default_horizon: int = DEFAULT_HORIZON
    max_horizon: int = MAX_HORIZON
    default_top_k: int = DEFAULT_TOP_K
    max_top_k: int = MAX_TOP_K

    revenue_decline_warning: float = -10.0
    revenue_decline_critical: float = -20.0

    profit_decline_warning: float = -10.0
    profit_decline_critical: float = -20.0

    cash_flow_decline_warning: float = -10.0
    cash_flow_decline_critical: float = -20.0

    liquidity_warning_periods: float = 3.0
    liquidity_critical_periods: float = 1.0

    minimum_confidence: float = 50.0

    require_human_review_for_critical: bool = True
    continue_on_tool_error: bool = True

    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""

        if self.default_horizon < 1:
            raise ForecastAgentConfigurationError(
                "default_horizon must be greater than zero."
            )

        if self.max_horizon < self.default_horizon:
            raise ForecastAgentConfigurationError(
                "max_horizon must be >= default_horizon."
            )

        if self.default_top_k < 1:
            raise ForecastAgentConfigurationError(
                "default_top_k must be greater than zero."
            )

        if self.max_top_k < self.default_top_k:
            raise ForecastAgentConfigurationError(
                "max_top_k must be >= default_top_k."
            )

        for name, value in (
            ("revenue_decline_warning", self.revenue_decline_warning),
            ("revenue_decline_critical", self.revenue_decline_critical),
            ("profit_decline_warning", self.profit_decline_warning),
            ("profit_decline_critical", self.profit_decline_critical),
            ("cash_flow_decline_warning", self.cash_flow_decline_warning),
            ("cash_flow_decline_critical", self.cash_flow_decline_critical),
            ("liquidity_warning_periods", self.liquidity_warning_periods),
            ("liquidity_critical_periods", self.liquidity_critical_periods),
            ("minimum_confidence", self.minimum_confidence),
        ):
            if not isfinite(float(value)):
                raise ForecastAgentConfigurationError(
                    f"{name} must be finite."
                )

        if not (
            self.revenue_decline_critical
            <= self.revenue_decline_warning
            <= 0
        ):
            raise ForecastAgentConfigurationError(
                "Revenue decline thresholds are invalid."
            )

        if not (
            self.profit_decline_critical
            <= self.profit_decline_warning
            <= 0
        ):
            raise ForecastAgentConfigurationError(
                "Profit decline thresholds are invalid."
            )

        if not (
            self.cash_flow_decline_critical
            <= self.cash_flow_decline_warning
            <= 0
        ):
            raise ForecastAgentConfigurationError(
                "Cash-flow decline thresholds are invalid."
            )

        if self.liquidity_critical_periods > self.liquidity_warning_periods:
            raise ForecastAgentConfigurationError(
                "Critical liquidity runway must be <= warning runway."
            )

        if not 0 <= self.minimum_confidence <= 100:
            raise ForecastAgentConfigurationError(
                "minimum_confidence must be between 0 and 100."
            )


@dataclass
class ForecastAgentRequest:
    """Normalized request sent to the forecast agent."""

    company_id: str
    operation: ForecastOperation = ForecastOperation.FORECAST

    financial_data: Any = None
    historical_data: Any = None

    metric: Optional[ForecastMetric] = None
    horizon: int = DEFAULT_HORIZON

    method: Optional[str] = None
    query: Optional[str] = None

    previous_period_data: Any = None
    scenario: Optional[Mapping[str, Any]] = None

    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, config: ForecastAgentConfig) -> None:
        """Validate the request."""

        if not isinstance(self.company_id, str):
            raise InvalidForecastAgentInputError(
                "company_id must be a string."
            )

        self.company_id = self.company_id.strip()

        if not self.company_id:
            raise InvalidForecastAgentInputError(
                "company_id cannot be empty."
            )

        if len(self.company_id) > MAX_COMPANY_ID_LENGTH:
            raise InvalidForecastAgentInputError(
                "company_id is too long."
            )

        if self.horizon < 1 or self.horizon > config.max_horizon:
            raise InvalidForecastAgentInputError(
                f"horizon must be between 1 and {config.max_horizon}."
            )

        if self.query is not None:
            if not isinstance(self.query, str):
                raise InvalidForecastAgentInputError(
                    "query must be a string."
                )

            self.query = self.query.strip()

            if len(self.query) > MAX_QUERY_LENGTH:
                raise InvalidForecastAgentInputError(
                    "query is too long."
                )

        if self.user_id is not None and not isinstance(
            self.user_id, str
        ):
            raise InvalidForecastAgentInputError(
                "user_id must be a string."
            )


@dataclass
class ForecastFinding:
    """Structured finding produced by the agent."""

    finding_type: str
    title: str
    message: str
    severity: ForecastRiskLevel = ForecastRiskLevel.LOW
    metric: Optional[str] = None
    value: Optional[float] = None
    evidence: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_type": self.finding_type,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "metric": self.metric,
            "value": self.value,
            "evidence": _serialize(self.evidence),
            "metadata": _serialize(self.metadata),
        }


@dataclass
class ForecastAgentResult:
    """Standardized forecast-agent response."""

    success: bool
    operation: str
    company_id: str

    message: str = ""

    forecast: Any = None
    findings: List[ForecastFinding] = field(default_factory=list)
    risks: List[ForecastFinding] = field(default_factory=list)

    confidence: Optional[float] = None
    trend: ForecastTrend = ForecastTrend.UNKNOWN
    risk_level: ForecastRiskLevel = ForecastRiskLevel.LOW

    human_review_required: bool = False

    execution_time_ms: Optional[float] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "company_id": self.company_id,
            "message": self.message,
            "forecast": _serialize(self.forecast),
            "findings": [item.to_dict() for item in self.findings],
            "risks": [item.to_dict() for item in self.risks],
            "confidence": self.confidence,
            "trend": self.trend.value,
            "risk_level": self.risk_level.value,
            "human_review_required": self.human_review_required,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat(),
            "metadata": _serialize(self.metadata),
        }


# ============================================================================
# Forecast Agent
# ============================================================================


class ForecastAgent:
    """
    Agentic orchestration layer for financial forecasting.

    The agent routes requests to specialized forecasting services/tools.

    Example dependencies:

        ForecastAgent(
            forecast_service=...,
            revenue_forecast_service=...,
            profit_forecast_service=...,
            cash_flow_forecast_service=...,
            liquidity_forecast_service=...,
            forecast_tools=...,
            evaluation_service=...,
            audit_service=...,
            authorization_service=...,
        )
    """

    def __init__(
        self,
        forecast_service: Any = None,
        revenue_forecast_service: Any = None,
        profit_forecast_service: Any = None,
        cash_flow_forecast_service: Any = None,
        liquidity_forecast_service: Any = None,
        forecast_tools: Any = None,
        evaluation_service: Any = None,
        recommendation_service: Any = None,
        alert_service: Any = None,
        audit_service: Any = None,
        authorization_service: Any = None,
        config: Optional[ForecastAgentConfig] = None,
    ) -> None:

        self.forecast_service = forecast_service
        self.revenue_forecast_service = revenue_forecast_service
        self.profit_forecast_service = profit_forecast_service
        self.cash_flow_forecast_service = cash_flow_forecast_service
        self.liquidity_forecast_service = liquidity_forecast_service

        self.forecast_tools = forecast_tools
        self.evaluation_service = evaluation_service
        self.recommendation_service = recommendation_service
        self.alert_service = alert_service
        self.audit_service = audit_service
        self.authorization_service = authorization_service

        self.config = config or ForecastAgentConfig()
        self.config.validate()

    # ========================================================================
    # Public API
    # ========================================================================

    def run(
        self,
        request: ForecastAgentRequest | Mapping[str, Any],
    ) -> ForecastAgentResult:
        """
        Execute a forecast-agent request.
        """

        if not self.config.enabled:
            raise ForecastAgentConfigurationError(
                "Forecast agent is disabled."
            )

        normalized_request = self._normalize_request(request)

        self._authorize(normalized_request)

        started = datetime.now(timezone.utc)

        try:
            result = self._dispatch(normalized_request)

            result.execution_time_ms = (
                datetime.now(timezone.utc) - started
            ).total_seconds() * 1000

            self._audit(
                operation=normalized_request.operation.value,
                request=normalized_request,
                result=result,
            )

            return result

        except ForecastAgentError:
            raise

        except Exception as exc:
            self._audit_error(
                operation=normalized_request.operation.value,
                request=normalized_request,
                error=exc,
            )

            raise ForecastAgentExecutionError(
                f"Forecast operation failed: {exc}"
            ) from exc

    def forecast(
        self,
        company_id: str,
        financial_data: Any,
        horizon: Optional[int] = None,
        method: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ForecastAgentResult:
        """Run a general financial forecast."""

        request = ForecastAgentRequest(
            company_id=company_id,
            operation=ForecastOperation.FORECAST,
            financial_data=financial_data,
            historical_data=financial_data,
            horizon=horizon or self.config.default_horizon,
            method=method,
            user_id=user_id,
        )

        return self.run(request)

    def revenue_forecast(
        self,
        company_id: str,
        historical_data: Any,
        horizon: Optional[int] = None,
        method: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ForecastAgentResult:
        """Run a revenue forecast."""

        request = ForecastAgentRequest(
            company_id=company_id,
            operation=ForecastOperation.REVENUE_FORECAST,
            financial_data=historical_data,
            historical_data=historical_data,
            metric=ForecastMetric.REVENUE,
            horizon=horizon or self.config.default_horizon,
            method=method,
            user_id=user_id,
        )

        return self.run(request)

    def profit_forecast(
        self,
        company_id: str,
        historical_data: Any,
        horizon: Optional[int] = None,
        method: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ForecastAgentResult:
        """Run a profit forecast."""

        request = ForecastAgentRequest(
            company_id=company_id,
            operation=ForecastOperation.PROFIT_FORECAST,
            financial_data=historical_data,
            historical_data=historical_data,
            metric=ForecastMetric.PROFIT,
            horizon=horizon or self.config.default_horizon,
            method=method,
            user_id=user_id,
        )

        return self.run(request)

    def cash_flow_forecast(
        self,
        company_id: str,
        historical_data: Any,
        horizon: Optional[int] = None,
        method: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ForecastAgentResult:
        """Run a cash-flow forecast."""

        request = ForecastAgentRequest(
            company_id=company_id,
            operation=ForecastOperation.CASH_FLOW_FORECAST,
            financial_data=historical_data,
            historical_data=historical_data,
            metric=ForecastMetric.CASH_FLOW,
            horizon=horizon or self.config.default_horizon,
            method=method,
            user_id=user_id,
        )

        return self.run(request)

    def liquidity_forecast(
        self,
        company_id: str,
        historical_data: Any,
        horizon: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> ForecastAgentResult:
        """Run a liquidity forecast."""

        request = ForecastAgentRequest(
            company_id=company_id,
            operation=ForecastOperation.LIQUIDITY_FORECAST,
            financial_data=historical_data,
            historical_data=historical_data,
            metric=ForecastMetric.LIQUIDITY,
            horizon=horizon or self.config.default_horizon,
            user_id=user_id,
        )

        return self.run(request)

    def forecast_all(
        self,
        company_id: str,
        financial_data: Any,
        horizon: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> ForecastAgentResult:
        """
        Run the complete financial forecast suite.

        Includes:
        - revenue
        - profit
        - cash flow
        - liquidity
        """

        request = ForecastAgentRequest(
            company_id=company_id,
            operation=ForecastOperation.FORECAST_ALL,
            financial_data=financial_data,
            historical_data=financial_data,
            horizon=horizon or self.config.default_horizon,
            user_id=user_id,
        )

        return self.run(request)

    def compare_forecasts(
        self,
        company_id: str,
        forecasts: Any,
        metrics: Optional[Sequence[str]] = None,
        user_id: Optional[str] = None,
    ) -> ForecastAgentResult:
        """Compare multiple forecast outputs."""

        request = ForecastAgentRequest(
            company_id=company_id,
            operation=ForecastOperation.COMPARE_FORECASTS,
            financial_data=forecasts,
            historical_data=forecasts,
            horizon=self.config.default_horizon,
            user_id=user_id,
            metadata={
                "metrics": list(metrics or []),
            },
        )

        return self.run(request)

    def evaluate_forecast(
        self,
        company_id: str,
        actual_values: Sequence[float],
        predicted_values: Sequence[float],
        user_id: Optional[str] = None,
    ) -> ForecastAgentResult:
        """Evaluate forecast accuracy."""

        request = ForecastAgentRequest(
            company_id=company_id,
            operation=ForecastOperation.EVALUATE_FORECAST,
            financial_data={
                "actual": list(actual_values),
                "predicted": list(predicted_values),
            },
            historical_data={
                "actual": list(actual_values),
                "predicted": list(predicted_values),
            },
            user_id=user_id,
        )

        return self.run(request)

    def forecast_risk(
        self,
        company_id: str,
        forecast: Any,
        user_id: Optional[str] = None,
    ) -> ForecastAgentResult:
        """Analyze risk contained in an existing forecast."""

        request = ForecastAgentRequest(
            company_id=company_id,
            operation=ForecastOperation.FORECAST_RISK,
            financial_data=forecast,
            historical_data=forecast,
            user_id=user_id,
        )

        return self.run(request)

    # ========================================================================
    # Dispatch
    # ========================================================================

    def _dispatch(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        operation = request.operation

        if operation == ForecastOperation.FORECAST:
            return self._run_general_forecast(request)

        if operation == ForecastOperation.REVENUE_FORECAST:
            return self._run_revenue_forecast(request)

        if operation == ForecastOperation.PROFIT_FORECAST:
            return self._run_profit_forecast(request)

        if operation == ForecastOperation.CASH_FLOW_FORECAST:
            return self._run_cash_flow_forecast(request)

        if operation == ForecastOperation.LIQUIDITY_FORECAST:
            return self._run_liquidity_forecast(request)

        if operation == ForecastOperation.FORECAST_ALL:
            return self._run_all_forecasts(request)

        if operation == ForecastOperation.COMPARE_FORECASTS:
            return self._compare_forecasts(request)

        if operation == ForecastOperation.EVALUATE_FORECAST:
            return self._evaluate_forecast(request)

        if operation == ForecastOperation.FORECAST_RISK:
            return self._analyze_forecast_risk(request)

        if operation == ForecastOperation.FORECAST_SUMMARY:
            return self._build_forecast_summary(request)

        raise UnsupportedForecastOperationError(
            f"Unsupported operation: {operation.value}"
        )

    # ========================================================================
    # Forecast Operations
    # ========================================================================

    def _run_general_forecast(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        service = self.forecast_service or self.forecast_tools

        if service is None:
            raise ForecastAgentConfigurationError(
                "No general forecast service or forecast tools configured."
            )

        result = self._invoke(
            service,
            (
                "forecast",
                "predict",
                "analyze",
                "run",
            ),
            {
                "company_id": request.company_id,
                "financial_data": request.financial_data,
                "historical_data": request.historical_data,
                "horizon": request.horizon,
                "method": request.method,
            },
        )

        return self._build_result(
            request,
            result,
            "General financial forecast completed.",
        )

    def _run_revenue_forecast(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        service = (
            self.revenue_forecast_service
            or self.forecast_tools
        )

        if service is None:
            raise ForecastAgentConfigurationError(
                "Revenue forecast service is not configured."
            )

        result = self._invoke(
            service,
            (
                "forecast",
                "forecast_revenue",
                "predict",
                "analyze",
                "run",
            ),
            {
                "company_id": request.company_id,
                "historical_data": request.historical_data,
                "financial_data": request.financial_data,
                "horizon": request.horizon,
                "method": request.method,
            },
        )

        return self._build_result(
            request,
            result,
            "Revenue forecast completed.",
            metric=ForecastMetric.REVENUE,
        )

    def _run_profit_forecast(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        service = (
            self.profit_forecast_service
            or self.forecast_tools
        )

        if service is None:
            raise ForecastAgentConfigurationError(
                "Profit forecast service is not configured."
            )

        result = self._invoke(
            service,
            (
                "forecast",
                "forecast_profit",
                "predict",
                "analyze",
                "run",
            ),
            {
                "company_id": request.company_id,
                "historical_data": request.historical_data,
                "financial_data": request.financial_data,
                "horizon": request.horizon,
                "method": request.method,
            },
        )

        return self._build_result(
            request,
            result,
            "Profit forecast completed.",
            metric=ForecastMetric.PROFIT,
        )

    def _run_cash_flow_forecast(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        service = (
            self.cash_flow_forecast_service
            or self.forecast_tools
        )

        if service is None:
            raise ForecastAgentConfigurationError(
                "Cash-flow forecast service is not configured."
            )

        result = self._invoke(
            service,
            (
                "forecast",
                "forecast_cash_flow",
                "predict",
                "analyze",
                "run",
            ),
            {
                "company_id": request.company_id,
                "historical_data": request.historical_data,
                "financial_data": request.financial_data,
                "horizon": request.horizon,
                "method": request.method,
            },
        )

        return self._build_result(
            request,
            result,
            "Cash-flow forecast completed.",
            metric=ForecastMetric.CASH_FLOW,
        )

    def _run_liquidity_forecast(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        service = (
            self.liquidity_forecast_service
            or self.forecast_tools
        )

        if service is None:
            raise ForecastAgentConfigurationError(
                "Liquidity forecast service is not configured."
            )

        result = self._invoke(
            service,
            (
                "forecast",
                "forecast_liquidity",
                "analyze",
                "assess",
                "run",
            ),
            {
                "company_id": request.company_id,
                "historical_data": request.historical_data,
                "financial_data": request.financial_data,
                "horizon": request.horizon,
            },
        )

        return self._build_result(
            request,
            result,
            "Liquidity forecast completed.",
            metric=ForecastMetric.LIQUIDITY,
        )

    def _run_all_forecasts(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        operations = (
            (
                "revenue",
                self._run_revenue_forecast,
            ),
            (
                "profit",
                self._run_profit_forecast,
            ),
            (
                "cash_flow",
                self._run_cash_flow_forecast,
            ),
            (
                "liquidity",
                self._run_liquidity_forecast,
            ),
        )

        for name, operation in operations:
            try:
                results[name] = operation(request).to_dict()
            except Exception as exc:
                errors[name] = str(exc)

                if not self.config.continue_on_tool_error:
                    raise

        findings = self._extract_findings(results)

        risk_level = self._highest_risk(findings)

        return ForecastAgentResult(
            success=not errors,
            operation=request.operation.value,
            company_id=request.company_id,
            message=(
                "Complete forecast suite executed."
                if not errors
                else "Forecast suite completed with partial failures."
            ),
            forecast=results,
            findings=findings,
            risks=[
                finding
                for finding in findings
                if finding.severity
                in (
                    ForecastRiskLevel.HIGH,
                    ForecastRiskLevel.CRITICAL,
                )
            ],
            risk_level=risk_level,
            human_review_required=(
                risk_level == ForecastRiskLevel.CRITICAL
                and self.config.require_human_review_for_critical
            ),
            metadata={
                "errors": errors,
                "horizon": request.horizon,
            },
        )

    # ========================================================================
    # Forecast Comparison
    # ========================================================================

    def _compare_forecasts(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        service = (
            self.evaluation_service
            or self.forecast_tools
            or self.forecast_service
        )

        if service is None:
            comparison = self._deterministic_forecast_comparison(
                request.financial_data
            )

            return self._build_result(
                request,
                comparison,
                "Forecast comparison completed.",
            )

        result = self._invoke(
            service,
            (
                "compare_forecasts",
                "compare",
                "evaluate",
                "analyze",
                "run",
            ),
            {
                "company_id": request.company_id,
                "forecasts": request.financial_data,
                "metrics": request.metadata.get("metrics", []),
            },
        )

        return self._build_result(
            request,
            result,
            "Forecast comparison completed.",
        )

    # ========================================================================
    # Evaluation
    # ========================================================================

    def _evaluate_forecast(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        actual = request.financial_data.get("actual")
        predicted = request.financial_data.get("predicted")

        if not isinstance(actual, Sequence) or isinstance(
            actual, (str, bytes)
        ):
            raise InvalidForecastAgentInputError(
                "actual values must be a sequence."
            )

        if not isinstance(predicted, Sequence) or isinstance(
            predicted, (str, bytes)
        ):
            raise InvalidForecastAgentInputError(
                "predicted values must be a sequence."
            )

        if len(actual) != len(predicted):
            raise InvalidForecastAgentInputError(
                "actual and predicted values must have equal length."
            )

        if not actual:
            raise InvalidForecastAgentInputError(
                "actual and predicted values cannot be empty."
            )

        service = self.evaluation_service

        if service is not None:
            result = self._invoke(
                service,
                (
                    "evaluate",
                    "evaluate_forecast",
                    "compare",
                    "run",
                ),
                {
                    "actual": list(actual),
                    "predicted": list(predicted),
                },
            )

        else:
            result = self._deterministic_evaluation(
                actual,
                predicted,
            )

        return self._build_result(
            request,
            result,
            "Forecast evaluation completed.",
        )

    # ========================================================================
    # Forecast Risk
    # ========================================================================

    def _analyze_forecast_risk(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        findings = self._detect_forecast_risks(
            request.financial_data
        )

        risk_level = self._highest_risk(findings)

        confidence = self._extract_confidence(
            request.financial_data
        )

        return ForecastAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message="Forecast risk analysis completed.",
            forecast=request.financial_data,
            findings=findings,
            risks=findings,
            confidence=confidence,
            risk_level=risk_level,
            human_review_required=(
                risk_level == ForecastRiskLevel.CRITICAL
                and self.config.require_human_review_for_critical
            ),
        )

    # ========================================================================
    # Summary
    # ========================================================================

    def _build_forecast_summary(
        self,
        request: ForecastAgentRequest,
    ) -> ForecastAgentResult:

        findings = self._detect_forecast_risks(
            request.financial_data
        )

        risk_level = self._highest_risk(findings)
        trend = self._infer_trend(request.financial_data)

        return ForecastAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message=self._summary_message(
                risk_level,
                trend,
                findings,
            ),
            forecast=request.financial_data,
            findings=findings,
            risks=[
                item
                for item in findings
                if item.severity
                in (
                    ForecastRiskLevel.HIGH,
                    ForecastRiskLevel.CRITICAL,
                )
            ],
            confidence=self._extract_confidence(
                request.financial_data
            ),
            trend=trend,
            risk_level=risk_level,
            human_review_required=(
                risk_level == ForecastRiskLevel.CRITICAL
                and self.config.require_human_review_for_critical
            ),
        )

    # ========================================================================
    # Result Processing
    # ========================================================================

    def _build_result(
        self,
        request: ForecastAgentRequest,
        forecast: Any,
        message: str,
        metric: Optional[ForecastMetric] = None,
    ) -> ForecastAgentResult:

        confidence = self._extract_confidence(forecast)

        findings = self._detect_forecast_risks(
            forecast,
            metric=metric,
        )

        risk_level = self._highest_risk(findings)

        return ForecastAgentResult(
            success=True,
            operation=request.operation.value,
            company_id=request.company_id,
            message=message,
            forecast=forecast,
            findings=findings,
            risks=[
                finding
                for finding in findings
                if finding.severity
                in (
                    ForecastRiskLevel.HIGH,
                    ForecastRiskLevel.CRITICAL,
                )
            ],
            confidence=confidence,
            trend=self._infer_trend(forecast),
            risk_level=risk_level,
            human_review_required=(
                risk_level == ForecastRiskLevel.CRITICAL
                and self.config.require_human_review_for_critical
            ),
            metadata={
                "metric": (
                    metric.value
                    if metric is not None
                    else None
                ),
                "horizon": request.horizon,
            },
        )

    # ========================================================================
    # Risk Detection
    # ========================================================================

    def _detect_forecast_risks(
        self,
        forecast: Any,
        metric: Optional[ForecastMetric] = None,
    ) -> List[ForecastFinding]:

        findings: List[ForecastFinding] = []

        data = _serialize(forecast)

        revenue_decline = self._find_numeric(
            data,
            (
                "revenue_change",
                "revenue_growth",
                "revenue_growth_rate",
                "projected_revenue_change",
            ),
        )

        if revenue_decline is not None:
            if revenue_decline <= self.config.revenue_decline_critical:
                findings.append(
                    ForecastFinding(
                        finding_type="revenue_decline",
                        title="Critical revenue decline",
                        message=(
                            f"Forecast indicates revenue decline "
                            f"of {revenue_decline:.2f}%."
                        ),
                        severity=ForecastRiskLevel.CRITICAL,
                        metric=ForecastMetric.REVENUE.value,
                        value=revenue_decline,
                    )
                )

            elif revenue_decline <= self.config.revenue_decline_warning:
                findings.append(
                    ForecastFinding(
                        finding_type="revenue_decline",
                        title="Revenue decline risk",
                        message=(
                            f"Forecast indicates revenue decline "
                            f"of {revenue_decline:.2f}%."
                        ),
                        severity=ForecastRiskLevel.HIGH,
                        metric=ForecastMetric.REVENUE.value,
                        value=revenue_decline,
                    )
                )

        profit_decline = self._find_numeric(
            data,
            (
                "profit_change",
                "profit_growth",
                "profit_growth_rate",
                "projected_profit_change",
            ),
        )

        if profit_decline is not None:
            if profit_decline <= self.config.profit_decline_critical:
                findings.append(
                    ForecastFinding(
                        finding_type="profit_decline",
                        title="Critical profit deterioration",
                        message=(
                            f"Forecast indicates profit decline "
                            f"of {profit_decline:.2f}%."
                        ),
                        severity=ForecastRiskLevel.CRITICAL,
                        metric=ForecastMetric.PROFIT.value,
                        value=profit_decline,
                    )
                )

            elif profit_decline <= self.config.profit_decline_warning:
                findings.append(
                    ForecastFinding(
                        finding_type="profit_decline",
                        title="Profit deterioration risk",
                        message=(
                            f"Forecast indicates profit decline "
                            f"of {profit_decline:.2f}%."
                        ),
                        severity=ForecastRiskLevel.HIGH,
                        metric=ForecastMetric.PROFIT.value,
                        value=profit_decline,
                    )
                )

        cash_flow_decline = self._find_numeric(
            data,
            (
                "cash_flow_change",
                "cash_flow_growth",
                "cash_flow_growth_rate",
                "projected_cash_flow_change",
            ),
        )

        if cash_flow_decline is not None:
            if (
                cash_flow_decline
                <= self.config.cash_flow_decline_critical
            ):
                findings.append(
                    ForecastFinding(
                        finding_type="cash_flow_decline",
                        title="Critical cash-flow deterioration",
                        message=(
                            f"Forecast indicates cash-flow decline "
                            f"of {cash_flow_decline:.2f}%."
                        ),
                        severity=ForecastRiskLevel.CRITICAL,
                        metric=ForecastMetric.CASH_FLOW.value,
                        value=cash_flow_decline,
                    )
                )

            elif (
                cash_flow_decline
                <= self.config.cash_flow_decline_warning
            ):
                findings.append(
                    ForecastFinding(
                        finding_type="cash_flow_decline",
                        title="Cash-flow deterioration risk",
                        message=(
                            f"Forecast indicates cash-flow decline "
                            f"of {cash_flow_decline:.2f}%."
                        ),
                        severity=ForecastRiskLevel.HIGH,
                        metric=ForecastMetric.CASH_FLOW.value,
                        value=cash_flow_decline,
                    )
                )

        loss_probability = self._find_numeric(
            data,
            (
                "loss_probability",
                "projected_loss_probability",
            ),
        )

        if loss_probability is not None:
            loss_probability = self._normalize_percent(
                loss_probability
            )

            if loss_probability >= 90:
                findings.append(
                    ForecastFinding(
                        finding_type="loss_risk",
                        title="Critical loss probability",
                        message=(
                            f"Forecast loss probability is "
                            f"{loss_probability:.2f}%."
                        ),
                        severity=ForecastRiskLevel.CRITICAL,
                        metric=ForecastMetric.PROFIT.value,
                        value=loss_probability,
                    )
                )

            elif loss_probability >= 70:
                findings.append(
                    ForecastFinding(
                        finding_type="loss_risk",
                        title="High loss probability",
                        message=(
                            f"Forecast loss probability is "
                            f"{loss_probability:.2f}%."
                        ),
                        severity=ForecastRiskLevel.HIGH,
                        metric=ForecastMetric.PROFIT.value,
                        value=loss_probability,
                    )
                )

        runway = self._find_numeric(
            data,
            (
                "cash_runway_periods",
                "runway_periods",
                "projected_runway",
            ),
        )

        if runway is not None:
            if runway <= self.config.liquidity_critical_periods:
                findings.append(
                    ForecastFinding(
                        finding_type="liquidity",
                        title="Critical liquidity runway",
                        message=(
                            f"Projected cash runway is "
                            f"{runway:.2f} periods."
                        ),
                        severity=ForecastRiskLevel.CRITICAL,
                        metric=ForecastMetric.LIQUIDITY.value,
                        value=runway,
                    )
                )

            elif runway <= self.config.liquidity_warning_periods:
                findings.append(
                    ForecastFinding(
                        finding_type="liquidity",
                        title="Liquidity runway warning",
                        message=(
                            f"Projected cash runway is "
                            f"{runway:.2f} periods."
                        ),
                        severity=ForecastRiskLevel.HIGH,
                        metric=ForecastMetric.LIQUIDITY.value,
                        value=runway,
                    )
                )

        minimum_balance = self._find_numeric(
            data,
            (
                "minimum_projected_balance",
                "minimum_cash_balance",
                "minimum_balance",
            ),
        )

        if minimum_balance is not None and minimum_balance < 0:
            findings.append(
                ForecastFinding(
                    finding_type="negative_cash_balance",
                    title="Projected negative cash balance",
                    message=(
                        "Forecast indicates that cash balance may "
                        "become negative."
                    ),
                    severity=ForecastRiskLevel.CRITICAL,
                    metric=ForecastMetric.CASH_BALANCE.value,
                    value=minimum_balance,
                )
            )

        return findings

    # ========================================================================
    # Deterministic Evaluation
    # ========================================================================

    @staticmethod
    def _deterministic_evaluation(
        actual: Sequence[Any],
        predicted: Sequence[Any],
    ) -> Dict[str, float]:

        actual_values = [
            float(value)
            for value in actual
        ]

        predicted_values = [
            float(value)
            for value in predicted
        ]

        absolute_errors = [
            abs(a - p)
            for a, p in zip(
                actual_values,
                predicted_values,
            )
        ]

        squared_errors = [
            (a - p) ** 2
            for a, p in zip(
                actual_values,
                predicted_values,
            )
        ]

        mae = (
            sum(absolute_errors)
            / len(absolute_errors)
        )

        mse = (
            sum(squared_errors)
            / len(squared_errors)
        )

        rmse = mse ** 0.5

        percentage_errors = [
            abs(a - p) / abs(a) * 100
            for a, p in zip(
                actual_values,
                predicted_values,
            )
            if a != 0
        ]

        mape = (
            sum(percentage_errors)
            / len(percentage_errors)
            if percentage_errors
            else None
        )

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "sample_count": len(actual_values),
        }

    # ========================================================================
    # Deterministic Comparison
    # ========================================================================

    @staticmethod
    def _deterministic_forecast_comparison(
        forecasts: Any,
    ) -> Dict[str, Any]:

        if not isinstance(forecasts, Mapping):
            return {
                "forecasts": _serialize(forecasts),
                "comparison": {},
            }

        comparison: Dict[str, Any] = {}

        for name, forecast in forecasts.items():

            data = _serialize(forecast)

            if isinstance(data, Mapping):
                comparison[str(name)] = {
                    "forecast": data,
                    "confidence": data.get("confidence"),
                    "risk_level": data.get("risk_level"),
                }
            else:
                comparison[str(name)] = {
                    "forecast": data,
                }

        return {
            "forecasts": comparison,
            "model_count": len(comparison),
        }

    # ========================================================================
    # Request Normalization
    # ========================================================================

    def _normalize_request(
        self,
        request: ForecastAgentRequest | Mapping[str, Any],
    ) -> ForecastAgentRequest:

        if isinstance(request, ForecastAgentRequest):
            normalized = request

        elif isinstance(request, Mapping):
            payload = dict(request)

            operation = payload.get(
                "operation",
                ForecastOperation.FORECAST,
            )

            if not isinstance(operation, ForecastOperation):
                operation = ForecastOperation(
                    str(operation).lower()
                )

            metric = payload.get("metric")

            if metric is not None and not isinstance(
                metric,
                ForecastMetric,
            ):
                metric = ForecastMetric(
                    str(metric).lower()
                )

            normalized = ForecastAgentRequest(
                company_id=payload.get("company_id", ""),
                operation=operation,
                financial_data=payload.get(
                    "financial_data"
                ),
                historical_data=payload.get(
                    "historical_data"
                ),
                metric=metric,
                horizon=payload.get(
                    "horizon",
                    self.config.default_horizon,
                ),
                method=payload.get("method"),
                query=payload.get("query"),
                previous_period_data=payload.get(
                    "previous_period_data"
                ),
                scenario=payload.get("scenario"),
                user_id=payload.get("user_id"),
                metadata=dict(
                    payload.get("metadata") or {}
                ),
            )

        else:
            raise InvalidForecastAgentInputError(
                "request must be ForecastAgentRequest or mapping."
            )

        normalized.validate(self.config)

        return normalized

    # ========================================================================
    # Authorization
    # ========================================================================

    def _authorize(
        self,
        request: ForecastAgentRequest,
    ) -> None:

        if self.authorization_service is None:
            return

        result = self._invoke(
            self.authorization_service,
            (
                "authorize",
                "check_permission",
                "can_access",
                "has_permission",
            ),
            {
                "user_id": request.user_id,
                "company_id": request.company_id,
                "operation": request.operation.value,
                "resource": "forecast",
            },
        )

        if result is False:
            raise ForecastAgentAccessDeniedError(
                "User is not authorized to access forecasting."
            )

        if isinstance(result, Mapping):
            allowed = result.get(
                "allowed",
                result.get(
                    "authorized",
                    result.get("success", True),
                ),
            )

            if allowed is False:
                raise ForecastAgentAccessDeniedError(
                    "Forecasting access denied."
                )

    # ========================================================================
    # Audit
    # ========================================================================

    def _audit(
        self,
        operation: str,
        request: ForecastAgentRequest,
        result: ForecastAgentResult,
    ) -> None:

        if self.audit_service is None:
            return

        payload = {
            "event": "forecast_agent_execution",
            "operation": operation,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "success": result.success,
            "risk_level": result.risk_level.value,
            "human_review_required": (
                result.human_review_required
            ),
            "metadata": request.metadata,
        }

        try:
            self._invoke(
                self.audit_service,
                (
                    "record",
                    "log",
                    "create",
                    "write",
                ),
                payload,
            )
        except Exception:
            # Audit failure should not normally break a forecast.
            # A production deployment may instead route this to a
            # dedicated audit-failure monitoring channel.
            pass

    def _audit_error(
        self,
        operation: str,
        request: ForecastAgentRequest,
        error: Exception,
    ) -> None:

        if self.audit_service is None:
            return

        payload = {
            "event": "forecast_agent_error",
            "operation": operation,
            "company_id": request.company_id,
            "user_id": request.user_id,
            "error": str(error),
        }

        try:
            self._invoke(
                self.audit_service,
                (
                    "record",
                    "log",
                    "create",
                    "write",
                ),
                payload,
            )
        except Exception:
            pass

    # ========================================================================
    # Findings / Trend / Risk Helpers
    # ========================================================================

    @staticmethod
    def _highest_risk(
        findings: Iterable[ForecastFinding],
    ) -> ForecastRiskLevel:

        priority = {
            ForecastRiskLevel.LOW: 0,
            ForecastRiskLevel.MEDIUM: 1,
            ForecastRiskLevel.HIGH: 2,
            ForecastRiskLevel.CRITICAL: 3,
        }

        highest = ForecastRiskLevel.LOW

        for finding in findings:
            if priority[finding.severity] > priority[highest]:
                highest = finding.severity

        return highest

    @staticmethod
    def _extract_confidence(
        data: Any,
    ) -> Optional[float]:

        value = ForecastAgent._find_numeric(
            _serialize(data),
            (
                "confidence",
                "forecast_confidence",
                "model_confidence",
            ),
        )

        if value is None:
            return None

        return ForecastAgent._normalize_percent(value)

    @staticmethod
    def _infer_trend(
        data: Any,
    ) -> ForecastTrend:

        serialized = _serialize(data)

        growth = ForecastAgent._find_numeric(
            serialized,
            (
                "growth_rate",
                "forecast_growth_rate",
                "trend",
            ),
        )

        if growth is not None:

            if growth > 1:
                return ForecastTrend.UP

            if growth < -1:
                return ForecastTrend.DOWN

            return ForecastTrend.FLAT

        if isinstance(serialized, Mapping):

            values = serialized.get(
                "forecast",
                serialized.get(
                    "predictions",
                    serialized.get("values"),
                ),
            )

            if isinstance(values, Sequence) and not isinstance(
                values,
                (str, bytes),
            ):
                numbers = []

                for value in values:
                    number = _safe_float(value)

                    if number is not None:
                        numbers.append(number)

                if len(numbers) >= 2:

                    if numbers[-1] > numbers[0]:
                        return ForecastTrend.UP

                    if numbers[-1] < numbers[0]:
                        return ForecastTrend.DOWN

                    return ForecastTrend.FLAT

        return ForecastTrend.UNKNOWN

    @staticmethod
    def _summary_message(
        risk_level: ForecastRiskLevel,
        trend: ForecastTrend,
        findings: Sequence[ForecastFinding],
    ) -> str:

        if risk_level == ForecastRiskLevel.CRITICAL:
            return (
                "Forecast indicates critical financial risk and "
                "requires management review."
            )

        if risk_level == ForecastRiskLevel.HIGH:
            return (
                "Forecast indicates elevated financial risk and "
                "should be reviewed."
            )

        if trend == ForecastTrend.UP:
            return "Forecast indicates an improving financial trend."

        if trend == ForecastTrend.DOWN:
            return "Forecast indicates a declining financial trend."

        if findings:
            return "Forecast completed with moderate financial signals."

        return "Forecast indicates no material risk signals."

    @staticmethod
    def _extract_findings(
        results: Mapping[str, Any],
    ) -> List[ForecastFinding]:

        findings: List[ForecastFinding] = []

        for _, result in results.items():

            if not isinstance(result, Mapping):
                continue

            raw_findings = result.get("findings", [])

            if not isinstance(raw_findings, Sequence):
                continue

            for item in raw_findings:

                if isinstance(item, ForecastFinding):
                    findings.append(item)

                elif isinstance(item, Mapping):

                    severity = item.get(
                        "severity",
                        ForecastRiskLevel.LOW.value,
                    )

                    try:
                        severity = ForecastRiskLevel(
                            str(severity).lower()
                        )
                    except ValueError:
                        severity = ForecastRiskLevel.LOW

                    findings.append(
                        ForecastFinding(
                            finding_type=str(
                                item.get(
                                    "finding_type",
                                    "forecast_signal",
                                )
                            ),
                            title=str(
                                item.get(
                                    "title",
                                    "Forecast finding",
                                )
                            ),
                            message=str(
                                item.get(
                                    "message",
                                    "",
                                )
                            ),
                            severity=severity,
                            metric=item.get("metric"),
                            value=_safe_float(
                                item.get("value")
                            ),
                            evidence=list(
                                item.get(
                                    "evidence",
                                    [],
                                )
                                or []
                            ),
                            metadata=dict(
                                item.get(
                                    "metadata",
                                    {},
                                )
                                or {}
                            ),
                        )
                    )

        return findings

    # ========================================================================
    # Generic Service Invocation
    # ========================================================================

    @staticmethod
    def _invoke(
        service: Any,
        method_names: Sequence[str],
        payload: Mapping[str, Any],
    ) -> Any:

        for method_name in method_names:

            method = getattr(
                service,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:
                return method(**dict(payload))
            except TypeError:
                try:
                    return method(dict(payload))
                except TypeError:
                    continue

        if callable(service):
            try:
                return service(**dict(payload))
            except TypeError:
                return service(dict(payload))

        raise ForecastAgentExecutionError(
            "No compatible forecasting service method was found."
        )

    # ========================================================================
    # Numeric Extraction
    # ========================================================================

    @staticmethod
    def _find_numeric(
        data: Any,
        keys: Sequence[str],
    ) -> Optional[float]:

        if isinstance(data, Mapping):

            for key in keys:
                if key in data:
                    value = _safe_float(data[key])

                    if value is not None:
                        return value

            for value in data.values():

                result = ForecastAgent._find_numeric(
                    value,
                    keys,
                )

                if result is not None:
                    return result

        elif isinstance(data, Sequence) and not isinstance(
            data,
            (str, bytes),
        ):

            for value in data:

                result = ForecastAgent._find_numeric(
                    value,
                    keys,
                )

                if result is not None:
                    return result

        return None

    @staticmethod
    def _normalize_percent(
        value: float,
    ) -> float:

        if 0 <= value <= 1:
            return value * 100

        return max(
            0.0,
            min(100.0, value),
        )


# ============================================================================
# Serialization Helpers
# ============================================================================


def _safe_float(value: Any) -> Optional[float]:

    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not isfinite(number):
        return None

    return number


def _serialize(value: Any) -> Any:

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): _serialize(item)
            for key, item in value.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _serialize(item)
            for item in value
        ]

    if hasattr(value, "model_dump"):

        try:
            return _serialize(
                value.model_dump()
            )
        except Exception:
            pass

    if hasattr(value, "dict"):

        try:
            return _serialize(
                value.dict()
            )
        except Exception:
            pass

    if hasattr(value, "to_dict"):

        try:
            return _serialize(
                value.to_dict()
            )
        except Exception:
            pass

    if hasattr(value, "__dict__"):

        try:
            return {
                str(key): _serialize(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        except Exception:
            pass

    return value


# ============================================================================
# Convenience Functions
# ============================================================================


def forecast_financials(
    agent: ForecastAgent,
    company_id: str,
    financial_data: Any,
    horizon: int = DEFAULT_HORIZON,
    method: Optional[str] = None,
    user_id: Optional[str] = None,
) -> ForecastAgentResult:
    """Convenience wrapper for general financial forecasting."""

    return agent.forecast(
        company_id=company_id,
        financial_data=financial_data,
        horizon=horizon,
        method=method,
        user_id=user_id,
    )


def forecast_revenue(
    agent: ForecastAgent,
    company_id: str,
    historical_data: Any,
    horizon: int = DEFAULT_HORIZON,
    method: Optional[str] = None,
    user_id: Optional[str] = None,
) -> ForecastAgentResult:
    """Convenience wrapper for revenue forecasting."""

    return agent.revenue_forecast(
        company_id=company_id,
        historical_data=historical_data,
        horizon=horizon,
        method=method,
        user_id=user_id,
    )


def forecast_profit(
    agent: ForecastAgent,
    company_id: str,
    historical_data: Any,
    horizon: int = DEFAULT_HORIZON,
    method: Optional[str] = None,
    user_id: Optional[str] = None,
) -> ForecastAgentResult:
    """Convenience wrapper for profit forecasting."""

    return agent.profit_forecast(
        company_id=company_id,
        historical_data=historical_data,
        horizon=horizon,
        method=method,
        user_id=user_id,
    )


def forecast_cash_flow(
    agent: ForecastAgent,
    company_id: str,
    historical_data: Any,
    horizon: int = DEFAULT_HORIZON,
    method: Optional[str] = None,
    user_id: Optional[str] = None,
) -> ForecastAgentResult:
    """Convenience wrapper for cash-flow forecasting."""

    return agent.cash_flow_forecast(
        company_id=company_id,
        historical_data=historical_data,
        horizon=horizon,
        method=method,
        user_id=user_id,
    )


def forecast_liquidity(
    agent: ForecastAgent,
    company_id: str,
    historical_data: Any,
    horizon: int = DEFAULT_HORIZON,
    user_id: Optional[str] = None,
) -> ForecastAgentResult:
    """Convenience wrapper for liquidity forecasting."""

    return agent.liquidity_forecast(
        company_id=company_id,
        historical_data=historical_data,
        horizon=horizon,
        user_id=user_id,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "ForecastAgentError",
    "InvalidForecastAgentInputError",
    "ForecastAgentExecutionError",
    "ForecastAgentConfigurationError",
    "ForecastAgentAccessDeniedError",
    "UnsupportedForecastOperationError",

    # Enums
    "ForecastOperation",
    "ForecastMetric",
    "ForecastRiskLevel",
    "ForecastTrend",

    # Dataclasses
    "ForecastAgentConfig",
    "ForecastAgentRequest",
    "ForecastFinding",
    "ForecastAgentResult",

    # Agent
    "ForecastAgent",

    # Convenience functions
    "forecast_financials",
    "forecast_revenue",
    "forecast_profit",
    "forecast_cash_flow",
    "forecast_liquidity",
]