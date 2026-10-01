"""
FinCo AI - Forecast Report
===========================

Generates professional financial forecasting reports for:

- Revenue
- Profit
- Cash flow
- Liquidity
- Forecast accuracy
- Confidence intervals
- Trend analysis
- Risk assessment
- Forecast vs actual comparison
- Management recommendations

Location:
    backend/app/reports/forecast_report.py

The module is intentionally independent from the database and LLM layers.
It can therefore be used by:

    - FastAPI routes
    - Forecasting agents
    - Report generator
    - CLI scripts
    - Scheduled jobs
    - Power BI data preparation
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CURRENCY = "USD"

FORECAST_TYPES = {
    "revenue",
    "profit",
    "cash_flow",
    "liquidity",
    "expense",
    "custom",
}

RISK_LEVELS = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""

    if value is None:
        return default

    if isinstance(value, bool):
        return float(value)

    try:
        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def _safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers."""

    if abs(denominator) < 1e-12:
        return default

    return numerator / denominator


def _percentage_change(
    current: float,
    previous: float,
) -> Optional[float]:
    """Calculate percentage change."""

    if abs(previous) < 1e-12:
        return None

    return ((current - previous) / abs(previous)) * 100.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)

    if not values:
        return 0.0

    return statistics.mean(values)


def _round(value: Any, digits: int = 2) -> float:
    return round(_safe_float(value), digits)


def _format_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def _format_currency(
    value: float,
    currency: str = DEFAULT_CURRENCY,
) -> str:
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "INR": "₹",
        "CAD": "C$",
        "AUD": "A$",
    }

    symbol = symbols.get(currency.upper(), currency.upper() + " ")

    return f"{symbol}{value:,.2f}"


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "N/A"

    return f"{value:.2f}%"


def _normalize_period(value: Any) -> str:
    if value is None:
        return "Unknown"

    return str(value)


def _risk_rank(level: str) -> int:
    return RISK_LEVELS.get(str(level).upper(), 1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ForecastReportConfig:
    """Configuration for forecast report generation."""

    currency: str = DEFAULT_CURRENCY

    confidence_threshold: float = 0.70

    high_mape_threshold: float = 20.0
    medium_mape_threshold: float = 10.0

    high_decline_threshold: float = -10.0
    critical_decline_threshold: float = -20.0

    high_volatility_threshold: float = 0.20

    negative_cash_flow_risk: bool = True

    include_recommendations: bool = True
    include_risk_analysis: bool = True
    include_forecast_table: bool = True

    company_name: str = "FinCo Demo Company"

    report_title: str = "Financial Forecast Report"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ForecastPoint:
    """Represents one forecasted period."""

    period: str

    forecast: float

    lower_bound: Optional[float] = None

    upper_bound: Optional[float] = None

    actual: Optional[float] = None

    confidence: Optional[float] = None

    forecast_type: str = "custom"

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error(self) -> Optional[float]:
        if self.actual is None:
            return None

        return self.forecast - self.actual

    @property
    def absolute_error(self) -> Optional[float]:
        error = self.error

        if error is None:
            return None

        return abs(error)

    @property
    def percentage_error(self) -> Optional[float]:
        if self.actual is None or abs(self.actual) < 1e-12:
            return None

        return ((self.forecast - self.actual) / abs(self.actual)) * 100.0


@dataclass
class ForecastMetric:
    """A KPI used in the forecast report."""

    name: str

    value: Any

    unit: str = ""

    status: str = "INFO"

    description: str = ""

    benchmark: Optional[str] = None


@dataclass
class ForecastRisk:
    """Forecast-related risk."""

    category: str

    level: str

    score: float

    title: str

    description: str

    impact: str

    mitigation: str


@dataclass
class ForecastInsight:
    """Management insight derived from forecast data."""

    title: str

    category: str

    severity: str

    description: str

    evidence: List[str] = field(default_factory=list)


@dataclass
class ForecastRecommendation:
    """Actionable recommendation."""

    priority: str

    category: str

    action: str

    rationale: str

    expected_impact: str

    requires_human_approval: bool = False


@dataclass
class ForecastReportSection:
    """Generic report section."""

    title: str

    content: str

    metrics: List[ForecastMetric] = field(default_factory=list)


@dataclass
class ForecastReport:
    """Complete forecast report."""

    report_id: str

    company_name: str

    report_title: str

    generated_at: str

    forecast_type: str

    forecast_horizon: str

    summary: str

    overall_status: str

    overall_risk: str

    confidence_score: float

    metrics: List[ForecastMetric] = field(default_factory=list)

    forecast_points: List[ForecastPoint] = field(default_factory=list)

    risks: List[ForecastRisk] = field(default_factory=list)

    insights: List[ForecastInsight] = field(default_factory=list)

    recommendations: List[ForecastRecommendation] = field(
        default_factory=list
    )

    sections: List[ForecastReportSection] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            default=str,
        )


# ---------------------------------------------------------------------------
# Forecast Report Generator
# ---------------------------------------------------------------------------

class ForecastReportGenerator:
    """
    Main forecast report generator.

    Example:

        generator = ForecastReportGenerator()

        report = generator.generate(
            forecast_type="revenue",
            forecast_points=[
                {
                    "period": "2026-10",
                    "forecast": 120000,
                    "lower_bound": 108000,
                    "upper_bound": 132000,
                },
                {
                    "period": "2026-11",
                    "forecast": 126000,
                    "lower_bound": 112000,
                    "upper_bound": 140000,
                },
            ],
            historical_values=[
                95000,
                101000,
                108000,
                112000,
            ],
        )

        print(report.to_json())
    """

    def __init__(
        self,
        config: Optional[ForecastReportConfig] = None,
    ) -> None:
        self.config = config or ForecastReportConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        forecast_type: str,
        forecast_points: Sequence[
            ForecastPoint | Mapping[str, Any]
        ],
        historical_values: Optional[Sequence[float]] = None,
        actual_values: Optional[Sequence[float]] = None,
        baseline_values: Optional[Sequence[float]] = None,
        model_metrics: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ForecastReport:
        """
        Generate a complete forecast report.
        """

        forecast_type = self._normalize_forecast_type(forecast_type)

        points = self._normalize_forecast_points(
            forecast_points,
            forecast_type,
        )

        historical = [
            _safe_float(value)
            for value in (historical_values or [])
        ]

        actuals = [
            _safe_float(value)
            for value in (actual_values or [])
        ]

        baseline = [
            _safe_float(value)
            for value in (baseline_values or [])
        ]

        metrics = self._calculate_metrics(
            forecast_type=forecast_type,
            points=points,
            historical_values=historical,
            actual_values=actuals,
            baseline_values=baseline,
            model_metrics=model_metrics or {},
        )

        confidence = self._calculate_confidence(
            points=points,
            model_metrics=model_metrics or {},
        )

        risks = self._detect_risks(
            forecast_type=forecast_type,
            points=points,
            historical_values=historical,
            metrics=metrics,
            confidence=confidence,
        )

        insights = self._generate_insights(
            forecast_type=forecast_type,
            points=points,
            historical_values=historical,
            metrics=metrics,
            risks=risks,
        )

        recommendations = []

        if self.config.include_recommendations:
            recommendations = self._generate_recommendations(
                forecast_type=forecast_type,
                metrics=metrics,
                risks=risks,
                insights=insights,
            )

        summary = self._build_summary(
            forecast_type=forecast_type,
            points=points,
            metrics=metrics,
            risks=risks,
            confidence=confidence,
        )

        overall_risk = self._overall_risk(risks)

        overall_status = self._overall_status(
            risks=risks,
            confidence=confidence,
        )

        horizon = self._determine_horizon(points)

        sections = self._build_sections(
            forecast_type=forecast_type,
            points=points,
            metrics=metrics,
            risks=risks,
            insights=insights,
            recommendations=recommendations,
        )

        report_id = self._generate_report_id(
            forecast_type=forecast_type,
        )

        report_metadata = dict(metadata or {})

        report_metadata.update(
            {
                "forecast_points": len(points),
                "historical_points": len(historical),
                "model_metrics_available": bool(model_metrics),
                "generator": "FinCoAI ForecastReportGenerator",
                "version": "1.0.0",
            }
        )

        return ForecastReport(
            report_id=report_id,
            company_name=self.config.company_name,
            report_title=self.config.report_title,
            generated_at=datetime.now(timezone.utc).isoformat(),
            forecast_type=forecast_type,
            forecast_horizon=horizon,
            summary=summary,
            overall_status=overall_status,
            overall_risk=overall_risk,
            confidence_score=confidence,
            metrics=metrics,
            forecast_points=points,
            risks=risks,
            insights=insights,
            recommendations=recommendations,
            sections=sections,
            metadata=report_metadata,
        )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_forecast_type(
        self,
        forecast_type: str,
    ) -> str:
        value = str(forecast_type).strip().lower()

        aliases = {
            "revenue_forecast": "revenue",
            "profit_forecast": "profit",
            "cashflow": "cash_flow",
            "cashflow_forecast": "cash_flow",
            "liquidity_forecast": "liquidity",
            "expense_forecast": "expense",
        }

        value = aliases.get(value, value)

        if value not in FORECAST_TYPES:
            value = "custom"

        return value

    def _normalize_forecast_points(
        self,
        points: Sequence[
            ForecastPoint | Mapping[str, Any]
        ],
        forecast_type: str,
    ) -> List[ForecastPoint]:

        normalized: List[ForecastPoint] = []

        for item in points:

            if isinstance(item, ForecastPoint):
                item.forecast_type = forecast_type
                normalized.append(item)
                continue

            data = dict(item)

            normalized.append(
                ForecastPoint(
                    period=_normalize_period(
                        data.get("period")
                        or data.get("date")
                        or data.get("month")
                    ),
                    forecast=_safe_float(
                        data.get("forecast")
                        if "forecast" in data
                        else data.get("prediction")
                    ),
                    lower_bound=self._optional_float(
                        data.get("lower_bound")
                        if "lower_bound" in data
                        else data.get("lower")
                    ),
                    upper_bound=self._optional_float(
                        data.get("upper_bound")
                        if "upper_bound" in data
                        else data.get("upper")
                    ),
                    actual=self._optional_float(
                        data.get("actual")
                    ),
                    confidence=self._optional_float(
                        data.get("confidence")
                    ),
                    forecast_type=forecast_type,
                    metadata=dict(
                        data.get("metadata") or {}
                    ),
                )
            )

        return normalized

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None

        return _safe_float(value)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calculate_metrics(
        self,
        forecast_type: str,
        points: Sequence[ForecastPoint],
        historical_values: Sequence[float],
        actual_values: Sequence[float],
        baseline_values: Sequence[float],
        model_metrics: Mapping[str, Any],
    ) -> List[ForecastMetric]:

        metrics: List[ForecastMetric] = []

        forecasts = [
            point.forecast
            for point in points
        ]

        if forecasts:
            total_forecast = sum(forecasts)

            metrics.append(
                ForecastMetric(
                    name="Forecast Total",
                    value=_round(total_forecast),
                    unit=self.config.currency,
                    description="Total projected value across the forecast horizon.",
                )
            )

            metrics.append(
                ForecastMetric(
                    name="Average Forecast",
                    value=_round(_mean(forecasts)),
                    unit=self.config.currency,
                    description="Average forecasted value per period.",
                )
            )

            metrics.append(
                ForecastMetric(
                    name="Peak Forecast",
                    value=_round(max(forecasts)),
                    unit=self.config.currency,
                    description="Highest forecasted period.",
                )
            )

            metrics.append(
                ForecastMetric(
                    name="Minimum Forecast",
                    value=_round(min(forecasts)),
                    unit=self.config.currency,
                    description="Lowest forecasted period.",
                )
            )

        if historical_values and forecasts:
            historical_average = _mean(historical_values)
            forecast_average = _mean(forecasts)

            change = _percentage_change(
                forecast_average,
                historical_average,
            )

            metrics.append(
                ForecastMetric(
                    name="Forecast vs Historical",
                    value=_round(change)
                    if change is not None
                    else None,
                    unit="%",
                    status=self._change_status(change),
                    description=(
                        "Percentage change between the average "
                        "forecast and historical average."
                    ),
                )
            )

            trend = self._calculate_trend(historical_values)

            metrics.append(
                ForecastMetric(
                    name="Historical Trend",
                    value=_round(trend),
                    unit="%",
                    status=self._change_status(trend),
                    description=(
                        "Approximate percentage trend across "
                        "the supplied historical periods."
                    ),
                )
            )

        volatility = self._forecast_volatility(forecasts)

        metrics.append(
            ForecastMetric(
                name="Forecast Volatility",
                value=_round(volatility * 100),
                unit="%",
                status=(
                    "WARNING"
                    if volatility >= self.config.high_volatility_threshold
                    else "NORMAL"
                ),
                description=(
                    "Relative standard deviation of forecast values."
                ),
            )
        )

        interval_width = self._average_interval_width(points)

        if interval_width is not None:
            metrics.append(
                ForecastMetric(
                    name="Average Prediction Interval",
                    value=_round(interval_width * 100),
                    unit="%",
                    status=(
                        "WARNING"
                        if interval_width > 0.25
                        else "NORMAL"
                    ),
                    description=(
                        "Average uncertainty range relative to "
                        "the forecast value."
                    ),
                )
            )

        accuracy_metrics = self._calculate_accuracy(
            points=points,
            actual_values=actual_values,
        )

        metrics.extend(accuracy_metrics)

        metrics.extend(
            self._model_metric_objects(model_metrics)
        )

        return metrics

    def _calculate_accuracy(
        self,
        points: Sequence[ForecastPoint],
        actual_values: Sequence[float],
    ) -> List[ForecastMetric]:

        paired = [
            (
                point.forecast,
                point.actual,
            )
            for point in points
            if point.actual is not None
        ]

        if not paired and actual_values:

            paired = [
                (
                    points[index].forecast,
                    actual_values[index],
                )
                for index in range(
                    min(len(points), len(actual_values))
                )
            ]

        if not paired:
            return []

        errors = [
            abs(forecast - actual)
            for forecast, actual in paired
        ]

        percentage_errors = [
            abs(
                _safe_divide(
                    forecast - actual,
                    actual,
                )
            ) * 100
            for forecast, actual in paired
            if abs(actual) > 1e-12
        ]

        mae = _mean(errors)

        mape = _mean(percentage_errors)

        rmse = math.sqrt(
            _mean(
                [
                    (forecast - actual) ** 2
                    for forecast, actual in paired
                ]
            )
        )

        return [
            ForecastMetric(
                name="MAE",
                value=_round(mae),
                unit=self.config.currency,
                description="Mean Absolute Error.",
            ),
            ForecastMetric(
                name="RMSE",
                value=_round(rmse),
                unit=self.config.currency,
                description="Root Mean Squared Error.",
            ),
            ForecastMetric(
                name="MAPE",
                value=_round(mape),
                unit="%",
                status=self._mape_status(mape),
                description="Mean Absolute Percentage Error.",
                benchmark="<10% preferred",
            ),
            ForecastMetric(
                name="Forecast Accuracy",
                value=_round(max(0.0, 100.0 - mape)),
                unit="%",
                status=self._mape_status(mape),
                description=(
                    "Simplified accuracy indicator derived from MAPE."
                ),
            ),
        ]

    def _model_metric_objects(
        self,
        model_metrics: Mapping[str, Any],
    ) -> List[ForecastMetric]:

        metrics: List[ForecastMetric] = []

        known_metrics = {
            "mae": ("MAE", self.config.currency),
            "rmse": ("RMSE", self.config.currency),
            "mape": ("MAPE", "%"),
            "smape": ("sMAPE", "%"),
            "r2": ("R²", ""),
            "accuracy": ("Model Accuracy", "%"),
            "precision": ("Precision", "%"),
            "recall": ("Recall", "%"),
            "confidence": ("Model Confidence", "%"),
        }

        for key, value in model_metrics.items():

            normalized_key = str(key).lower()

            if normalized_key not in known_metrics:
                continue

            name, unit = known_metrics[normalized_key]

            number = _safe_float(value)

            if (
                unit == "%"
                and abs(number) <= 1.0
            ):
                number *= 100.0

            metrics.append(
                ForecastMetric(
                    name=name,
                    value=_round(number),
                    unit=unit,
                    description=f"Model-provided {name} metric.",
                )
            )

        return metrics

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        points: Sequence[ForecastPoint],
        model_metrics: Mapping[str, Any],
    ) -> float:

        confidence_values = [
            point.confidence
            for point in points
            if point.confidence is not None
        ]

        if confidence_values:
            confidence = _mean(confidence_values)

            if confidence > 1:
                confidence /= 100.0

            return max(0.0, min(1.0, confidence))

        model_confidence = model_metrics.get("confidence")

        if model_confidence is not None:

            confidence = _safe_float(model_confidence)

            if confidence > 1:
                confidence /= 100.0

            return max(
                0.0,
                min(1.0, confidence),
            )

        interval_confidence = self._interval_confidence(points)

        if interval_confidence is not None:
            return interval_confidence

        return 0.70

    def _interval_confidence(
        self,
        points: Sequence[ForecastPoint],
    ) -> Optional[float]:

        widths = []

        for point in points:

            if (
                point.lower_bound is None
                or point.upper_bound is None
                or abs(point.forecast) < 1e-12
            ):
                continue

            width = (
                point.upper_bound
                - point.lower_bound
            ) / abs(point.forecast)

            widths.append(width)

        if not widths:
            return None

        average_width = _mean(widths)

        return max(
            0.0,
            min(
                1.0,
                1.0 - min(average_width, 1.0),
            ),
        )

    # ------------------------------------------------------------------
    # Risk detection
    # ------------------------------------------------------------------

    def _detect_risks(
        self,
        forecast_type: str,
        points: Sequence[ForecastPoint],
        historical_values: Sequence[float],
        metrics: Sequence[ForecastMetric],
        confidence: float,
    ) -> List[ForecastRisk]:

        risks: List[ForecastRisk] = []

        # --------------------------------------------------------------
        # Low confidence
        # --------------------------------------------------------------

        if confidence < self.config.confidence_threshold:

            level = "HIGH" if confidence < 0.50 else "MEDIUM"

            risks.append(
                ForecastRisk(
                    category="Forecast Uncertainty",
                    level=level,
                    score=(
                        1.0 - confidence
                    ) * 100,
                    title="Low Forecast Confidence",
                    description=(
                        f"Forecast confidence is "
                        f"{confidence * 100:.1f}%, below the "
                        f"configured threshold."
                    ),
                    impact=(
                        "Management decisions based solely on the "
                        "forecast may carry elevated uncertainty."
                    ),
                    mitigation=(
                        "Use scenario analysis, prediction intervals, "
                        "and human review before major decisions."
                    ),
                )
            )

        # --------------------------------------------------------------
        # Revenue / profit decline
        # --------------------------------------------------------------

        if historical_values and points:

            historical_average = _mean(
                historical_values
            )

            forecast_average = _mean(
                point.forecast
                for point in points
            )

            change = _percentage_change(
                forecast_average,
                historical_average,
            )

            if change is not None:

                if change <= self.config.critical_decline_threshold:

                    risks.append(
                        ForecastRisk(
                            category="Performance",
                            level="CRITICAL",
                            score=95.0,
                            title="Severe Forecast Decline",
                            description=(
                                f"Projected average {forecast_type} "
                                f"is expected to decline by "
                                f"{abs(change):.1f}% versus history."
                            ),
                            impact=(
                                "Significant deterioration may affect "
                                "profitability, liquidity, or operating capacity."
                            ),
                            mitigation=(
                                "Trigger management review, stress testing, "
                                "cost controls, and corrective action planning."
                            ),
                        )
                    )

                elif change <= self.config.high_decline_threshold:

                    risks.append(
                        ForecastRisk(
                            category="Performance",
                            level="HIGH",
                            score=75.0,
                            title="Forecast Decline",
                            description=(
                                f"Projected average {forecast_type} "
                                f"is expected to decline by "
                                f"{abs(change):.1f}% versus history."
                            ),
                            impact=(
                                "The decline could pressure financial "
                                "performance if the trend persists."
                            ),
                            mitigation=(
                                "Investigate root causes and create "
                                "downside mitigation scenarios."
                            ),
                        )
                    )

        # --------------------------------------------------------------
        # Negative cash flow
        # --------------------------------------------------------------

        if (
            forecast_type == "cash_flow"
            and self.config.negative_cash_flow_risk
        ):

            negative_periods = [
                point
                for point in points
                if point.forecast < 0
            ]

            if negative_periods:

                level = (
                    "CRITICAL"
                    if len(negative_periods) >= len(points) / 2
                    else "HIGH"
                )

                risks.append(
                    ForecastRisk(
                        category="Liquidity",
                        level=level,
                        score=85.0,
                        title="Negative Cash Flow Forecast",
                        description=(
                            f"{len(negative_periods)} forecast period(s) "
                            "show negative cash flow."
                        ),
                        impact=(
                            "Persistent negative cash flow may create "
                            "liquidity and financing pressure."
                        ),
                        mitigation=(
                            "Review working capital, collections, "
                            "payment schedules, and financing options."
                        ),
                    )
                )

        # --------------------------------------------------------------
        # Volatility
        # --------------------------------------------------------------

        forecasts = [
            point.forecast
            for point in points
        ]

        volatility = self._forecast_volatility(
            forecasts
        )

        if volatility >= self.config.high_volatility_threshold:

            risks.append(
                ForecastRisk(
                    category="Volatility",
                    level="HIGH",
                    score=min(
                        90.0,
                        volatility * 200,
                    ),
                    title="High Forecast Volatility",
                    description=(
                        f"Forecast volatility is "
                        f"{volatility * 100:.1f}%."
                    ),
                    impact=(
                        "Large period-to-period movements may reduce "
                        "planning reliability."
                    ),
                    mitigation=(
                        "Use rolling forecasts, scenario ranges, "
                        "and closer monitoring."
                    ),
                )
            )

        # --------------------------------------------------------------
        # Prediction interval
        # --------------------------------------------------------------

        interval_width = self._average_interval_width(
            points
        )

        if (
            interval_width is not None
            and interval_width > 0.25
        ):

            risks.append(
                ForecastRisk(
                    category="Uncertainty",
                    level="MEDIUM",
                    score=60.0,
                    title="Wide Prediction Interval",
                    description=(
                        f"Average prediction interval width is "
                        f"{interval_width * 100:.1f}% of the forecast."
                    ),
                    impact=(
                        "Actual results may differ materially from "
                        "the point forecast."
                    ),
                    mitigation=(
                        "Use lower and upper bounds in decision-making "
                        "rather than relying on the point estimate."
                    ),
                )
            )

        # --------------------------------------------------------------
        # Poor model accuracy
        # --------------------------------------------------------------

        mape = self._metric_value(
            metrics,
            "MAPE",
        )

        if (
            mape is not None
            and mape >= self.config.high_mape_threshold
        ):

            risks.append(
                ForecastRisk(
                    category="Model Quality",
                    level="HIGH",
                    score=80.0,
                    title="Poor Forecast Accuracy",
                    description=(
                        f"MAPE is {mape:.1f}%, indicating "
                        "high forecast error."
                    ),
                    impact=(
                        "The forecast may not be reliable enough "
                        "for high-impact financial decisions."
                    ),
                    mitigation=(
                        "Retrain the model, review features, "
                        "investigate data drift, and compare baselines."
                    ),
                )
            )

        return risks

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def _generate_insights(
        self,
        forecast_type: str,
        points: Sequence[ForecastPoint],
        historical_values: Sequence[float],
        metrics: Sequence[ForecastMetric],
        risks: Sequence[ForecastRisk],
    ) -> List[ForecastInsight]:

        insights: List[ForecastInsight] = []

        forecasts = [
            point.forecast
            for point in points
        ]

        if not forecasts:
            return insights

        # Trend
        if len(forecasts) >= 2:

            trend = _percentage_change(
                forecasts[-1],
                forecasts[0],
            )

            if trend is not None:

                direction = (
                    "increase"
                    if trend > 0
                    else "decline"
                )

                severity = (
                    "POSITIVE"
                    if trend > 0
                    else "WARNING"
                )

                insights.append(
                    ForecastInsight(
                        title=(
                            f"Forecasted {forecast_type.title()} "
                            f"{direction}"
                        ),
                        category="Trend",
                        severity=severity,
                        description=(
                            f"The forecast moves from "
                            f"{_format_currency(forecasts[0], self.config.currency)} "
                            f"to "
                            f"{_format_currency(forecasts[-1], self.config.currency)}, "
                            f"a {abs(trend):.1f}% "
                            f"{direction} across the horizon."
                        ),
                        evidence=[
                            f"First forecast: {forecasts[0]:.2f}",
                            f"Last forecast: {forecasts[-1]:.2f}",
                        ],
                    )
                )

        # Peak period
        peak_index = max(
            range(len(forecasts)),
            key=lambda index: forecasts[index],
        )

        insights.append(
            ForecastInsight(
                title="Peak Forecast Period",
                category="Forecast",
                severity="INFO",
                description=(
                    f"The highest projected value is "
                    f"{_format_currency(forecasts[peak_index], self.config.currency)} "
                    f"in {points[peak_index].period}."
                ),
                evidence=[
                    f"Peak value: {forecasts[peak_index]:.2f}",
                    f"Period: {points[peak_index].period}",
                ],
            )
        )

        # Confidence
        low_confidence = [
            point
            for point in points
            if (
                point.confidence is not None
                and (
                    point.confidence < 0.70
                    if point.confidence <= 1
                    else point.confidence < 70
                )
            )
        ]

        if low_confidence:

            insights.append(
                ForecastInsight(
                    title="Uncertainty Requires Attention",
                    category="Confidence",
                    severity="WARNING",
                    description=(
                        f"{len(low_confidence)} forecast period(s) "
                        "have lower confidence."
                    ),
                    evidence=[
                        point.period
                        for point in low_confidence
                    ],
                )
            )

        # Risk-based insight
        if risks:

            highest_risk = max(
                risks,
                key=lambda risk: _risk_rank(risk.level),
            )

            insights.append(
                ForecastInsight(
                    title=highest_risk.title,
                    category="Risk",
                    severity=highest_risk.level,
                    description=highest_risk.description,
                    evidence=[
                        highest_risk.impact,
                    ],
                )
            )

        return insights

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        forecast_type: str,
        metrics: Sequence[ForecastMetric],
        risks: Sequence[ForecastRisk],
        insights: Sequence[ForecastInsight],
    ) -> List[ForecastRecommendation]:

        recommendations: List[
            ForecastRecommendation
        ] = []

        risk_categories = {
            risk.category.lower()
            for risk in risks
        }

        if "forecast uncertainty" in risk_categories:
            recommendations.append(
                ForecastRecommendation(
                    priority="HIGH",
                    category="Risk Management",
                    action=(
                        "Use forecast ranges and scenario analysis "
                        "for management decisions."
                    ),
                    rationale=(
                        "Forecast confidence is below the configured "
                        "decision threshold."
                    ),
                    expected_impact=(
                        "Reduces the risk of decisions based on "
                        "overconfident point estimates."
                    ),
                    requires_human_approval=True,
                )
            )

        if "model quality" in risk_categories:
            recommendations.append(
                ForecastRecommendation(
                    priority="HIGH",
                    category="Model Improvement",
                    action=(
                        "Review training data, features, "
                        "model drift, and baseline performance."
                    ),
                    rationale=(
                        "Forecast error exceeds the configured "
                        "acceptable range."
                    ),
                    expected_impact=(
                        "Improved forecast reliability and "
                        "decision support quality."
                    ),
                    requires_human_approval=False,
                )
            )

        if "performance" in risk_categories:

            recommendations.append(
                ForecastRecommendation(
                    priority="HIGH",
                    category="Financial Planning",
                    action=(
                        f"Investigate the drivers behind the projected "
                        f"{forecast_type} decline."
                    ),
                    rationale=(
                        "The forecast indicates deterioration "
                        "relative to historical performance."
                    ),
                    expected_impact=(
                        "Earlier corrective action and reduced "
                        "financial downside."
                    ),
                    requires_human_approval=True,
                )
            )

        if "liquidity" in risk_categories:

            recommendations.append(
                ForecastRecommendation(
                    priority="CRITICAL",
                    category="Liquidity",
                    action=(
                        "Review cash reserves, receivables, "
                        "payables, and financing capacity."
                    ),
                    rationale=(
                        "Negative cash flow is projected in one or "
                        "more forecast periods."
                    ),
                    expected_impact=(
                        "Reduced probability of liquidity stress."
                    ),
                    requires_human_approval=True,
                )
            )

        if "volatility" in risk_categories:

            recommendations.append(
                ForecastRecommendation(
                    priority="MEDIUM",
                    category="Planning",
                    action=(
                        "Adopt rolling forecasts and monitor "
                        "period-to-period variance."
                    ),
                    rationale=(
                        "High forecast volatility makes static "
                        "planning less reliable."
                    ),
                    expected_impact=(
                        "Faster response to changing financial conditions."
                    ),
                    requires_human_approval=False,
                )
            )

        if not recommendations:

            recommendations.append(
                ForecastRecommendation(
                    priority="LOW",
                    category="Monitoring",
                    action=(
                        "Continue monitoring forecast performance "
                        "against actual results."
                    ),
                    rationale=(
                        "No material forecast risks were detected."
                    ),
                    expected_impact=(
                        "Maintains forecast accountability and "
                        "early detection of deviations."
                    ),
                    requires_human_approval=False,
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        forecast_type: str,
        points: Sequence[ForecastPoint],
        metrics: Sequence[ForecastMetric],
        risks: Sequence[ForecastRisk],
        confidence: float,
    ) -> str:

        if not points:
            return (
                f"No forecast points were supplied for "
                f"{forecast_type} forecasting."
            )

        average = _mean(
            point.forecast
            for point in points
        )

        risk = self._overall_risk(risks)

        return (
            f"The {forecast_type.replace('_', ' ')} forecast covers "
            f"{len(points)} period(s) with an average projected value "
            f"of {_format_currency(average, self.config.currency)}. "
            f"Forecast confidence is {confidence * 100:.1f}%, "
            f"and the overall forecast risk is {risk}. "
            f"{len(risks)} material risk signal(s) were identified."
        )

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _build_sections(
        self,
        forecast_type: str,
        points: Sequence[ForecastPoint],
        metrics: Sequence[ForecastMetric],
        risks: Sequence[ForecastRisk],
        insights: Sequence[ForecastInsight],
        recommendations: Sequence[ForecastRecommendation],
    ) -> List[ForecastReportSection]:

        sections: List[ForecastReportSection] = []

        sections.append(
            ForecastReportSection(
                title="Forecast Overview",
                content=(
                    f"{forecast_type.replace('_', ' ').title()} "
                    "forecast generated from the supplied model output."
                ),
                metrics=list(metrics),
            )
        )

        if self.config.include_forecast_table:

            sections.append(
                ForecastReportSection(
                    title="Forecast Projection",
                    content=(
                        f"The report contains {len(points)} "
                        "forecast period(s)."
                    ),
                )
            )

        if risks and self.config.include_risk_analysis:

            sections.append(
                ForecastReportSection(
                    title="Risk Analysis",
                    content=(
                        f"{len(risks)} forecast risk(s) "
                        "were identified."
                    ),
                )
            )

        if insights:

            sections.append(
                ForecastReportSection(
                    title="Key Insights",
                    content=(
                        f"{len(insights)} management insight(s) "
                        "were generated."
                    ),
                )
            )

        if recommendations:

            sections.append(
                ForecastReportSection(
                    title="Recommended Actions",
                    content=(
                        f"{len(recommendations)} recommended "
                        "action(s) are available."
                    ),
                )
            )

        return sections

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _overall_risk(
        self,
        risks: Sequence[ForecastRisk],
    ) -> str:

        if not risks:
            return "LOW"

        return max(
            (
                risk.level.upper()
                for risk in risks
            ),
            key=_risk_rank,
        )

    def _overall_status(
        self,
        risks: Sequence[ForecastRisk],
        confidence: float,
    ) -> str:

        risk = self._overall_risk(risks)

        if risk == "CRITICAL":
            return "CRITICAL"

        if risk == "HIGH":
            return "WARNING"

        if confidence < self.config.confidence_threshold:
            return "WARNING"

        if risk == "MEDIUM":
            return "ATTENTION"

        return "HEALTHY"

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_trend(
        values: Sequence[float],
    ) -> float:

        if len(values) < 2:
            return 0.0

        first = _safe_float(values[0])
        last = _safe_float(values[-1])

        result = _percentage_change(
            last,
            first,
        )

        return result or 0.0

    @staticmethod
    def _forecast_volatility(
        values: Sequence[float],
    ) -> float:

        values = list(values)

        if len(values) < 2:
            return 0.0

        average = _mean(values)

        if abs(average) < 1e-12:
            return 0.0

        try:
            std = statistics.stdev(values)
        except statistics.StatisticsError:
            return 0.0

        return abs(std / average)

    @staticmethod
    def _average_interval_width(
        points: Sequence[ForecastPoint],
    ) -> Optional[float]:

        widths = []

        for point in points:

            if (
                point.lower_bound is None
                or point.upper_bound is None
                or abs(point.forecast) < 1e-12
            ):
                continue

            width = (
                point.upper_bound
                - point.lower_bound
            ) / abs(point.forecast)

            widths.append(width)

        if not widths:
            return None

        return _mean(widths)

    @staticmethod
    def _determine_horizon(
        points: Sequence[ForecastPoint],
    ) -> str:

        if not points:
            return "Unknown"

        if len(points) == 1:
            return "1 period"

        return f"{len(points)} periods"

    @staticmethod
    def _generate_report_id(
        forecast_type: str,
    ) -> str:

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d%H%M%S")

        return (
            f"FINCO-FORECAST-"
            f"{forecast_type.upper()}-"
            f"{timestamp}"
        )

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metric_value(
        metrics: Sequence[ForecastMetric],
        name: str,
    ) -> Optional[float]:

        for metric in metrics:

            if metric.name.lower() == name.lower():

                if metric.value is None:
                    return None

                return _safe_float(metric.value)

        return None

    @staticmethod
    def _mape_status(
        mape: float,
    ) -> str:

        if mape >= 20:
            return "CRITICAL"

        if mape >= 10:
            return "WARNING"

        return "GOOD"

    @staticmethod
    def _change_status(
        change: Optional[float],
    ) -> str:

        if change is None:
            return "INFO"

        if change >= 10:
            return "POSITIVE"

        if change <= -10:
            return "WARNING"

        return "NORMAL"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(
        self,
        report: ForecastReport,
        indent: int = 2,
    ) -> str:

        return report.to_json(indent=indent)

    def to_markdown(
        self,
        report: ForecastReport,
    ) -> str:

        lines: List[str] = []

        lines.append(
            f"# {report.report_title}"
        )

        lines.append("")

        lines.append(
            f"**Company:** {report.company_name}"
        )

        lines.append(
            f"**Report ID:** {report.report_id}"
        )

        lines.append(
            f"**Generated:** {report.generated_at}"
        )

        lines.append(
            f"**Forecast Type:** "
            f"{report.forecast_type.replace('_', ' ').title()}"
        )

        lines.append(
            f"**Horizon:** {report.forecast_horizon}"
        )

        lines.append("")

        lines.append("## Executive Summary")

        lines.append("")

        lines.append(report.summary)

        lines.append("")

        lines.append("## Status")

        lines.append("")

        lines.append(
            f"- **Overall Status:** {report.overall_status}"
        )

        lines.append(
            f"- **Overall Risk:** {report.overall_risk}"
        )

        lines.append(
            f"- **Confidence:** "
            f"{report.confidence_score * 100:.1f}%"
        )

        lines.append("")

        if report.metrics:

            lines.append("## Forecast Metrics")

            lines.append("")

            lines.append(
                "| Metric | Value | Unit | Status |"
            )

            lines.append(
                "|---|---:|---|---|"
            )

            for metric in report.metrics:

                value = metric.value

                if isinstance(value, float):
                    value = _format_number(value)

                lines.append(
                    f"| {metric.name} | "
                    f"{value} | "
                    f"{metric.unit} | "
                    f"{metric.status} |"
                )

            lines.append("")

        if report.forecast_points:

            lines.append("## Forecast Projection")

            lines.append("")

            lines.append(
                "| Period | Forecast | Lower | Upper | Actual | Confidence |"
            )

            lines.append(
                "|---|---:|---:|---:|---:|---:|"
            )

            for point in report.forecast_points:

                lower = (
                    _format_number(point.lower_bound)
                    if point.lower_bound is not None
                    else "-"
                )

                upper = (
                    _format_number(point.upper_bound)
                    if point.upper_bound is not None
                    else "-"
                )

                actual = (
                    _format_number(point.actual)
                    if point.actual is not None
                    else "-"
                )

                confidence = (
                    f"{point.confidence * 100:.1f}%"
                    if point.confidence is not None
                    and point.confidence <= 1
                    else (
                        f"{point.confidence:.1f}%"
                        if point.confidence is not None
                        else "-"
                    )
                )

                lines.append(
                    f"| {point.period} | "
                    f"{_format_number(point.forecast)} | "
                    f"{lower} | "
                    f"{upper} | "
                    f"{actual} | "
                    f"{confidence} |"
                )

            lines.append("")

        if report.risks:

            lines.append("## Risk Analysis")

            lines.append("")

            for index, risk in enumerate(
                report.risks,
                start=1,
            ):

                lines.append(
                    f"### {index}. {risk.title}"
                )

                lines.append("")

                lines.append(
                    f"**Level:** {risk.level}"
                )

                lines.append("")

                lines.append(
                    f"**Description:** {risk.description}"
                )

                lines.append("")

                lines.append(
                    f"**Impact:** {risk.impact}"
                )

                lines.append("")

                lines.append(
                    f"**Mitigation:** {risk.mitigation}"
                )

                lines.append("")

        if report.insights:

            lines.append("## Key Insights")

            lines.append("")

            for insight in report.insights:

                lines.append(
                    f"- **{insight.title}** "
                    f"({insight.severity}): "
                    f"{insight.description}"
                )

            lines.append("")

        if report.recommendations:

            lines.append("## Recommended Actions")

            lines.append("")

            for index, recommendation in enumerate(
                report.recommendations,
                start=1,
            ):

                approval = (
                    "Human approval required"
                    if recommendation.requires_human_approval
                    else "No approval required"
                )

                lines.append(
                    f"{index}. **"
                    f"{recommendation.action}"
                    f"**"
                )

                lines.append(
                    f"   - Priority: "
                    f"{recommendation.priority}"
                )

                lines.append(
                    f"   - Category: "
                    f"{recommendation.category}"
                )

                lines.append(
                    f"   - Rationale: "
                    f"{recommendation.rationale}"
                )

                lines.append(
                    f"   - Expected impact: "
                    f"{recommendation.expected_impact}"
                )

                lines.append(
                    f"   - {approval}"
                )

                lines.append("")

        lines.append("---")

        lines.append(
            "*Generated by FinCo AI Financial Intelligence "
            "& Decision Copilot.*"
        )

        return "\n".join(lines)

    def to_text(
        self,
        report: ForecastReport,
    ) -> str:

        lines = [
            report.report_title,
            "=" * len(report.report_title),
            "",
            f"Company: {report.company_name}",
            f"Report ID: {report.report_id}",
            f"Forecast Type: {report.forecast_type}",
            f"Horizon: {report.forecast_horizon}",
            f"Status: {report.overall_status}",
            f"Risk: {report.overall_risk}",
            f"Confidence: "
            f"{report.confidence_score * 100:.1f}%",
            "",
            "SUMMARY",
            "-------",
            report.summary,
            "",
        ]

        if report.metrics:

            lines.extend(
                [
                    "METRICS",
                    "-------",
                ]
            )

            for metric in report.metrics:

                lines.append(
                    f"{metric.name}: "
                    f"{metric.value} "
                    f"{metric.unit}"
                )

            lines.append("")

        if report.risks:

            lines.extend(
                [
                    "RISKS",
                    "-----",
                ]
            )

            for risk in report.risks:

                lines.append(
                    f"[{risk.level}] "
                    f"{risk.title}: "
                    f"{risk.description}"
                )

            lines.append("")

        if report.recommendations:

            lines.extend(
                [
                    "RECOMMENDATIONS",
                    "---------------",
                ]
            )

            for recommendation in report.recommendations:

                lines.append(
                    f"[{recommendation.priority}] "
                    f"{recommendation.action}"
                )

        return "\n".join(lines)

    def to_html(
        self,
        report: ForecastReport,
    ) -> str:

        metric_rows = ""

        for metric in report.metrics:

            metric_rows += f"""
            <tr>
                <td>{self._escape(metric.name)}</td>
                <td>{self._escape(str(metric.value))}</td>
                <td>{self._escape(metric.unit)}</td>
                <td>{self._escape(metric.status)}</td>
            </tr>
            """

        forecast_rows = ""

        for point in report.forecast_points:

            lower = (
                f"{point.lower_bound:,.2f}"
                if point.lower_bound is not None
                else "-"
            )

            upper = (
                f"{point.upper_bound:,.2f}"
                if point.upper_bound is not None
                else "-"
            )

            actual = (
                f"{point.actual:,.2f}"
                if point.actual is not None
                else "-"
            )

            forecast_rows += f"""
            <tr>
                <td>{self._escape(point.period)}</td>
                <td>{point.forecast:,.2f}</td>
                <td>{lower}</td>
                <td>{upper}</td>
                <td>{actual}</td>
            </tr>
            """

        risk_html = ""

        for risk in report.risks:

            risk_html += f"""
            <article class="risk">
                <h3>{self._escape(risk.title)}</h3>
                <p><strong>Level:</strong>
                    {self._escape(risk.level)}
                </p>
                <p>{self._escape(risk.description)}</p>
                <p><strong>Impact:</strong>
                    {self._escape(risk.impact)}
                </p>
                <p><strong>Mitigation:</strong>
                    {self._escape(risk.mitigation)}
                </p>
            </article>
            """

        recommendation_html = ""

        for recommendation in report.recommendations:

            approval = (
                "Human approval required"
                if recommendation.requires_human_approval
                else "No approval required"
            )

            recommendation_html += f"""
            <article class="recommendation">
                <h3>{self._escape(recommendation.action)}</h3>
                <p>
                    <strong>Priority:</strong>
                    {self._escape(recommendation.priority)}
                </p>
                <p>
                    <strong>Category:</strong>
                    {self._escape(recommendation.category)}
                </p>
                <p>
                    <strong>Rationale:</strong>
                    {self._escape(recommendation.rationale)}
                </p>
                <p>
                    <strong>Expected Impact:</strong>
                    {self._escape(recommendation.expected_impact)}
                </p>
                <p><strong>{self._escape(approval)}</strong></p>
            </article>
            """

        insight_html = ""

        for insight in report.insights:

            insight_html += f"""
            <article class="insight">
                <h3>{self._escape(insight.title)}</h3>
                <p>
                    <strong>{self._escape(insight.severity)}</strong>
                </p>
                <p>{self._escape(insight.description)}</p>
            </article>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>{self._escape(report.report_title)}</title>

<style>

body {{
    font-family:
        Inter, Arial, Helvetica, sans-serif;

    background: #f4f6f8;

    color: #17202a;

    margin: 0;

    padding: 32px;
}}

.container {{
    max-width: 1200px;

    margin: auto;

    background: white;

    padding: 40px;

    border-radius: 16px;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.08);
}}

h1 {{
    margin-bottom: 8px;
}}

h2 {{
    margin-top: 36px;

    border-bottom:
        1px solid #e5e7eb;

    padding-bottom: 8px;
}}

.summary {{
    background: #f8fafc;

    padding: 20px;

    border-radius: 12px;

    line-height: 1.7;
}}

.status-grid {{
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 16px;

    margin: 24px 0;
}}

.card {{
    padding: 20px;

    background: #f8fafc;

    border-radius: 12px;
}}

.card strong {{
    display: block;

    font-size: 24px;

    margin-top: 8px;
}}

table {{
    width: 100%;

    border-collapse: collapse;

    margin-top: 16px;
}}

th, td {{
    padding: 12px;

    border-bottom:
        1px solid #e5e7eb;

    text-align: left;
}}

th {{
    background: #f8fafc;
}}

.risk {{
    border-left:
        5px solid #dc2626;

    background: #fef2f2;

    padding: 16px;

    margin: 14px 0;

    border-radius: 8px;
}}

.insight {{
    border-left:
        5px solid #2563eb;

    background: #eff6ff;

    padding: 16px;

    margin: 14px 0;

    border-radius: 8px;
}}

.recommendation {{
    border-left:
        5px solid #059669;

    background: #ecfdf5;

    padding: 16px;

    margin: 14px 0;

    border-radius: 8px;
}}

.footer {{
    margin-top: 40px;

    color: #6b7280;

    font-size: 13px;
}}

@media (max-width: 700px) {{
    body {{
        padding: 12px;
    }}

    .container {{
        padding: 20px;
    }}

    .status-grid {{
        grid-template-columns: 1fr;
    }}
}}

</style>
</head>

<body>

<div class="container">

<h1>{self._escape(report.report_title)}</h1>

<p>
<strong>Company:</strong>
{self._escape(report.company_name)}
</p>

<p>
<strong>Generated:</strong>
{self._escape(report.generated_at)}
</p>

<div class="summary">
<strong>Executive Summary</strong>
<p>{self._escape(report.summary)}</p>
</div>

<div class="status-grid">

<div class="card">
Status
<strong>
{self._escape(report.overall_status)}
</strong>
</div>

<div class="card">
Risk
<strong>
{self._escape(report.overall_risk)}
</strong>
</div>

<div class="card">
Confidence
<strong>
{report.confidence_score * 100:.1f}%
</strong>
</div>

</div>

<h2>Forecast Metrics</h2>

<table>

<thead>
<tr>
<th>Metric</th>
<th>Value</th>
<th>Unit</th>
<th>Status</th>
</tr>
</thead>

<tbody>
{metric_rows}
</tbody>

</table>

<h2>Forecast Projection</h2>

<table>

<thead>
<tr>
<th>Period</th>
<th>Forecast</th>
<th>Lower Bound</th>
<th>Upper Bound</th>
<th>Actual</th>
</tr>
</thead>

<tbody>
{forecast_rows}
</tbody>

</table>

<h2>Risk Analysis</h2>

{risk_html or "<p>No material risks detected.</p>"}

<h2>Key Insights</h2>

{insight_html or "<p>No additional insights generated.</p>"}

<h2>Recommended Actions</h2>

{recommendation_html or "<p>No recommendations generated.</p>"}

<div class="footer">
FinCo AI — Financial Intelligence &amp;
Decision Copilot
</div>

</div>

</body>
</html>
"""

    @staticmethod
    def _escape(value: str) -> str:
        """Minimal HTML escaping without external dependencies."""

        replacements = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;",
        }

        text = str(value)

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------

    def save_json(
        self,
        report: ForecastReport,
        path: str | Path,
    ) -> Path:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            report.to_json(),
            encoding="utf-8",
        )

        return target

    def save_markdown(
        self,
        report: ForecastReport,
        path: str | Path,
    ) -> Path:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            self.to_markdown(report),
            encoding="utf-8",
        )

        return target

    def save_text(
        self,
        report: ForecastReport,
        path: str | Path,
    ) -> Path:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            self.to_text(report),
            encoding="utf-8",
        )

        return target

    def save_html(
        self,
        report: ForecastReport,
        path: str | Path,
    ) -> Path:

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        target.write_text(
            self.to_html(report),
            encoding="utf-8",
        )

        return target


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def generate_forecast_report(
    forecast_type: str,
    forecast_points: Sequence[
        ForecastPoint | Mapping[str, Any]
    ],
    historical_values: Optional[Sequence[float]] = None,
    actual_values: Optional[Sequence[float]] = None,
    baseline_values: Optional[Sequence[float]] = None,
    model_metrics: Optional[Mapping[str, Any]] = None,
    company_name: str = "FinCo Demo Company",
    currency: str = DEFAULT_CURRENCY,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ForecastReport:
    """
    Convenience function for FastAPI/services/agents.
    """

    config = ForecastReportConfig(
        company_name=company_name,
        currency=currency,
    )

    generator = ForecastReportGenerator(
        config=config
    )

    return generator.generate(
        forecast_type=forecast_type,
        forecast_points=forecast_points,
        historical_values=historical_values,
        actual_values=actual_values,
        baseline_values=baseline_values,
        model_metrics=model_metrics,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Forecast comparison helper
# ---------------------------------------------------------------------------

def compare_forecast_to_actual(
    forecast: Sequence[float],
    actual: Sequence[float],
) -> Dict[str, Any]:
    """
    Compare forecasted values with actual values.

    Returns MAE, RMSE, MAPE and directional accuracy.
    """

    pairs = [
        (
            _safe_float(f),
            _safe_float(a),
        )
        for f, a in zip(forecast, actual)
    ]

    if not pairs:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "directional_accuracy": 0.0,
            "observations": 0,
        }

    errors = [
        abs(f - a)
        for f, a in pairs
    ]

    squared_errors = [
        (f - a) ** 2
        for f, a in pairs
    ]

    percentage_errors = [
        abs(
            (f - a) / a
        ) * 100
        for f, a in pairs
        if abs(a) > 1e-12
    ]

    directional_correct = 0

    if len(pairs) >= 2:

        for index in range(1, len(pairs)):

            forecast_direction = (
                pairs[index][0]
                - pairs[index - 1][0]
            )

            actual_direction = (
                pairs[index][1]
                - pairs[index - 1][1]
            )

            if (
                forecast_direction == 0
                and actual_direction == 0
            ):
                directional_correct += 1

            elif (
                forecast_direction > 0
                and actual_direction > 0
            ):
                directional_correct += 1

            elif (
                forecast_direction < 0
                and actual_direction < 0
            ):
                directional_correct += 1

        directional_accuracy = (
            directional_correct
            / (len(pairs) - 1)
        ) * 100

    else:
        directional_accuracy = 0.0

    return {
        "mae": _round(_mean(errors)),
        "rmse": _round(
            math.sqrt(
                _mean(squared_errors)
            )
        ),
        "mape": _round(
            _mean(percentage_errors)
        ),
        "directional_accuracy": _round(
            directional_accuracy
        ),
        "observations": len(pairs),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def create_demo_report() -> ForecastReport:

    points = [
        ForecastPoint(
            period="2026-10",
            forecast=125000,
            lower_bound=112000,
            upper_bound=138000,
            confidence=0.84,
        ),
        ForecastPoint(
            period="2026-11",
            forecast=131000,
            lower_bound=116000,
            upper_bound=146000,
            confidence=0.82,
        ),
        ForecastPoint(
            period="2026-12",
            forecast=139000,
            lower_bound=121000,
            upper_bound=157000,
            confidence=0.79,
        ),
        ForecastPoint(
            period="2027-01",
            forecast=144000,
            lower_bound=124000,
            upper_bound=164000,
            confidence=0.76,
        ),
        ForecastPoint(
            period="2027-02",
            forecast=151000,
            lower_bound=128000,
            upper_bound=174000,
            confidence=0.73,
        ),
        ForecastPoint(
            period="2027-03",
            forecast=157000,
            lower_bound=131000,
            upper_bound=183000,
            confidence=0.71,
        ),
    ]

    return generate_forecast_report(
        forecast_type="revenue",
        forecast_points=points,
        historical_values=[
            102000,
            108000,
            111000,
            116000,
            119000,
            121000,
        ],
        model_metrics={
            "mae": 4200,
            "rmse": 6100,
            "mape": 6.8,
            "confidence": 0.78,
        },
        company_name="FinCo Demo Corporation",
        currency="USD",
        metadata={
            "model": "FinCo Revenue Forecast Model",
            "model_version": "1.0",
            "forecast_frequency": "monthly",
            "scenario": "base",
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate FinCo AI financial forecast reports."
        )
    )

    parser.add_argument(
        "--output",
        default="reports/forecast_report",
        help="Output file prefix.",
    )

    parser.add_argument(
        "--format",
        choices=[
            "json",
            "markdown",
            "text",
            "html",
            "all",
        ],
        default="all",
        help="Output format.",
    )

    args = parser.parse_args()

    report = create_demo_report()

    generator = ForecastReportGenerator()

    output = Path(args.output)

    if args.format in {"json", "all"}:

        path = generator.save_json(
            report,
            output.with_suffix(".json"),
        )

        print(
            f"JSON report: {path}"
        )

    if args.format in {"markdown", "all"}:

        path = generator.save_markdown(
            report,
            output.with_suffix(".md"),
        )

        print(
            f"Markdown report: {path}"
        )

    if args.format in {"text", "all"}:

        path = generator.save_text(
            report,
            output.with_suffix(".txt"),
        )

        print(
            f"Text report: {path}"
        )

    if args.format in {"html", "all"}:

        path = generator.save_html(
            report,
            output.with_suffix(".html"),
        )

        print(
            f"HTML report: {path}"
        )

    print()
    print(
        f"Report ID: {report.report_id}"
    )

    print(
        f"Status: {report.overall_status}"
    )

    print(
        f"Risk: {report.overall_risk}"
    )

    print(
        f"Confidence: "
        f"{report.confidence_score * 100:.1f}%"
    )


if __name__ == "__main__":
    main()