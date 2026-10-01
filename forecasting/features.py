"""
FinCo AI - Forecasting Feature Engineering

File:
    backend/app/forecasting/features.py

Purpose:
    Build reusable time-series features for financial forecasting.

Supported feature groups:

    - Lag features
    - Rolling statistics
    - Growth features
    - Trend features
    - Volatility features
    - Momentum features
    - Seasonal/calendar features
    - Ratio features
    - Difference features
    - Financial-domain features

Designed for:

    Revenue forecasting
    Profit forecasting
    Cash-flow forecasting
    Liquidity forecasting
    Expense forecasting
    Financial risk prediction

Architecture:

    Raw Financial Series
            |
            v
    preprocessing.py
            |
            v
        features.py
            |
       +----+----+
       |         |
       v         v
    baseline   model.py
       |         |
       +----+----+
            |
            v
       evaluation.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
import math
import statistics


# ============================================================================
# Exceptions
# ============================================================================


class ForecastFeatureError(Exception):
    """Base exception for forecasting feature engineering."""


class InvalidFeatureInputError(
    ForecastFeatureError
):
    """Raised when feature input is invalid."""


class InsufficientFeatureDataError(
    ForecastFeatureError
):
    """Raised when insufficient historical data exists."""


class FeatureCalculationError(
    ForecastFeatureError
):
    """Raised when a feature cannot be calculated."""


# ============================================================================
# Enums
# ============================================================================


class FeatureType(str, Enum):
    """Forecast feature categories."""

    LAG = "lag"
    ROLLING = "rolling"
    GROWTH = "growth"
    TREND = "trend"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    SEASONAL = "seasonal"
    DIFFERENCE = "difference"
    RATIO = "ratio"
    FINANCIAL = "financial"


# ============================================================================
# Feature Definition
# ============================================================================


@dataclass(frozen=True)
class FeatureDefinition:
    """
    Defines one forecasting feature.
    """

    name: str

    feature_type: FeatureType

    description: str

    window: Optional[int] = None

    lag: Optional[int] = None

    required_history: int = 1

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ForecastFeatureConfig:
    """
    Configuration for forecasting feature engineering.
    """

    lag_periods: Tuple[int, ...] = (
        1,
        2,
        3,
        6,
        12,
    )

    rolling_windows: Tuple[int, ...] = (
        3,
        6,
        12,
    )

    growth_periods: Tuple[int, ...] = (
        1,
        3,
        6,
        12,
    )

    volatility_windows: Tuple[int, ...] = (
        3,
        6,
        12,
    )

    trend_windows: Tuple[int, ...] = (
        3,
        6,
        12,
    )

    seasonal_period: int = 12

    include_lags: bool = True

    include_rolling: bool = True

    include_growth: bool = True

    include_trend: bool = True

    include_volatility: bool = True

    include_momentum: bool = True

    include_seasonality: bool = True

    include_differences: bool = True

    include_ratios: bool = True

    include_financial_features: bool = True

    minimum_history: int = 3

    fill_missing: bool = True

    fill_value: float = 0.0

    def validate(self) -> None:
        """Validate feature configuration."""

        if self.minimum_history < 1:
            raise ValueError(
                "minimum_history must be >= 1."
            )

        if self.seasonal_period < 1:
            raise ValueError(
                "seasonal_period must be >= 1."
            )

        self._validate_positive_sequence(
            self.lag_periods,
            "lag_periods",
        )

        self._validate_positive_sequence(
            self.rolling_windows,
            "rolling_windows",
        )

        self._validate_positive_sequence(
            self.growth_periods,
            "growth_periods",
        )

        self._validate_positive_sequence(
            self.volatility_windows,
            "volatility_windows",
        )

        self._validate_positive_sequence(
            self.trend_windows,
            "trend_windows",
        )

    @staticmethod
    def _validate_positive_sequence(
        values: Sequence[int],
        name: str,
    ) -> None:

        for value in values:

            if value < 1:
                raise ValueError(
                    f"{name} values must be >= 1."
                )


# ============================================================================
# Feature Result
# ============================================================================


@dataclass
class ForecastFeatureResult:
    """
    Complete feature engineering result.
    """

    feature_names: List[str]

    features: Dict[str, List[float]]

    row_count: int

    feature_count: int

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def matrix(self) -> List[List[float]]:
        """
        Convert feature dictionary into a row-wise matrix.
        """

        if not self.feature_names:
            return []

        return [
            [
                self.features[name][index]
                for name in self.feature_names
            ]
            for index in range(
                self.row_count
            )
        ]

    def latest(self) -> Dict[str, float]:
        """Return the latest feature row."""

        if self.row_count == 0:
            return {}

        index = self.row_count - 1

        return {
            name: self.features[name][index]
            for name in self.feature_names
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""

        return {
            "feature_names":
                list(self.feature_names),
            "features": {
                name: list(values)
                for name, values
                in self.features.items()
            },
            "row_count": self.row_count,
            "feature_count":
                self.feature_count,
            "metadata":
                dict(self.metadata),
        }


# ============================================================================
# Feature Engineer
# ============================================================================


class ForecastFeatureEngineer:
    """
    Time-series feature engineering engine.

    The implementation is intentionally dependency-light so that
    it can be used before advanced ML libraries such as:

        scikit-learn
        XGBoost
        LightGBM
        CatBoost
        PyTorch

    Features are generated without random operations.
    """

    def __init__(
        self,
        config: Optional[
            ForecastFeatureConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or ForecastFeatureConfig()
        )

        self.config.validate()

        self.feature_registry = (
            self._build_registry()
        )

    # ========================================================================
    # MAIN API
    # ========================================================================

    def transform(
        self,
        values: Sequence[float],
        *,
        timestamps: Optional[
            Sequence[Any]
        ] = None,
        revenue: Optional[
            Sequence[float]
        ] = None,
        expenses: Optional[
            Sequence[float]
        ] = None,
        profit: Optional[
            Sequence[float]
        ] = None,
        cash_inflow: Optional[
            Sequence[float]
        ] = None,
        cash_outflow: Optional[
            Sequence[float]
        ] = None,
    ) -> ForecastFeatureResult:
        """
        Generate forecasting features.

        Parameters
        ----------
        values:
            Primary time series.

        timestamps:
            Optional timestamps used for calendar features.

        revenue:
            Optional revenue series.

        expenses:
            Optional expense series.

        profit:
            Optional profit series.

        cash_inflow:
            Optional cash inflow series.

        cash_outflow:
            Optional cash outflow series.
        """

        series = self._validate_series(
            values,
            "values",
        )

        self._validate_minimum_history(
            series
        )

        length = len(series)

        self._validate_optional_series(
            revenue,
            length,
            "revenue",
        )

        self._validate_optional_series(
            expenses,
            length,
            "expenses",
        )

        self._validate_optional_series(
            profit,
            length,
            "profit",
        )

        self._validate_optional_series(
            cash_inflow,
            length,
            "cash_inflow",
        )

        self._validate_optional_series(
            cash_outflow,
            length,
            "cash_outflow",
        )

        if timestamps is not None:

            if len(timestamps) != length:
                raise InvalidFeatureInputError(
                    "timestamps must have the "
                    "same length as values."
                )

        features: Dict[
            str,
            List[float],
        ] = {}

        # --------------------------------------------------------------------
        # Lag Features
        # --------------------------------------------------------------------

        if self.config.include_lags:

            self._add_lag_features(
                features,
                series,
            )

        # --------------------------------------------------------------------
        # Rolling Features
        # --------------------------------------------------------------------

        if self.config.include_rolling:

            self._add_rolling_features(
                features,
                series,
            )

        # --------------------------------------------------------------------
        # Growth Features
        # --------------------------------------------------------------------

        if self.config.include_growth:

            self._add_growth_features(
                features,
                series,
            )

        # --------------------------------------------------------------------
        # Trend Features
        # --------------------------------------------------------------------

        if self.config.include_trend:

            self._add_trend_features(
                features,
                series,
            )

        # --------------------------------------------------------------------
        # Volatility
        # --------------------------------------------------------------------

        if self.config.include_volatility:

            self._add_volatility_features(
                features,
                series,
            )

        # --------------------------------------------------------------------
        # Momentum
        # --------------------------------------------------------------------

        if self.config.include_momentum:

            self._add_momentum_features(
                features,
                series,
            )

        # --------------------------------------------------------------------
        # Seasonality
        # --------------------------------------------------------------------

        if (
            self.config.include_seasonality
            and timestamps is not None
        ):

            self._add_seasonal_features(
                features,
                timestamps,
            )

        # --------------------------------------------------------------------
        # Difference Features
        # --------------------------------------------------------------------

        if self.config.include_differences:

            self._add_difference_features(
                features,
                series,
            )

        # --------------------------------------------------------------------
        # Ratio Features
        # --------------------------------------------------------------------

        if self.config.include_ratios:

            self._add_ratio_features(
                features,
                series,
            )

        # --------------------------------------------------------------------
        # Financial Features
        # --------------------------------------------------------------------

        if self.config.include_financial_features:

            self._add_financial_features(
                features=features,
                revenue=revenue,
                expenses=expenses,
                profit=profit,
                cash_inflow=cash_inflow,
                cash_outflow=cash_outflow,
                length=length,
            )

        if self.config.fill_missing:

            self._fill_missing(
                features
            )

        feature_names = list(
            features.keys()
        )

        return ForecastFeatureResult(
            feature_names=feature_names,
            features=features,
            row_count=length,
            feature_count=len(feature_names),
            metadata={
                "primary_series": "values",
                "seasonal_period":
                    self.config.seasonal_period,
            },
        )

    # ========================================================================
    # LAG FEATURES
    # ========================================================================

    def create_lag(
        self,
        values: Sequence[float],
        lag: int,
    ) -> List[float]:
        """Create a lag feature."""

        series = self._validate_series(
            values,
            "values",
        )

        if lag < 1:
            raise InvalidFeatureInputError(
                "lag must be >= 1."
            )

        result = []

        for index in range(len(series)):

            if index < lag:
                result.append(
                    self.config.fill_value
                )
            else:
                result.append(
                    series[index - lag]
                )

        return result

    # ========================================================================
    # ROLLING FEATURES
    # ========================================================================

    def rolling_mean(
        self,
        values: Sequence[float],
        window: int,
    ) -> List[float]:

        return self._rolling_statistic(
            values,
            window,
            statistics.fmean,
        )

    def rolling_std(
        self,
        values: Sequence[float],
        window: int,
    ) -> List[float]:

        series = self._validate_series(
            values,
            "values",
        )

        self._validate_window(
            window
        )

        result = []

        for index in range(len(series)):

            if index + 1 < window:

                result.append(
                    self.config.fill_value
                )

                continue

            current = series[
                index + 1 - window:
                index + 1
            ]

            if len(current) <= 1:
                result.append(0.0)
            else:
                result.append(
                    statistics.stdev(
                        current
                    )
                )

        return result

    def rolling_min(
        self,
        values: Sequence[float],
        window: int,
    ) -> List[float]:

        return self._rolling_statistic(
            values,
            window,
            min,
        )

    def rolling_max(
        self,
        values: Sequence[float],
        window: int,
    ) -> List[float]:

        return self._rolling_statistic(
            values,
            window,
            max,
        )

    # ========================================================================
    # GROWTH
    # ========================================================================

    def growth_rate(
        self,
        values: Sequence[float],
        periods: int = 1,
    ) -> List[float]:
        """
        Percentage growth rate.

        Example:

            previous = 100
            current = 120

            growth = 20%
        """

        series = self._validate_series(
            values,
            "values",
        )

        if periods < 1:
            raise InvalidFeatureInputError(
                "periods must be >= 1."
            )

        result = []

        for index, current in enumerate(
            series
        ):

            if index < periods:

                result.append(
                    self.config.fill_value
                )

                continue

            previous = series[
                index - periods
            ]

            if abs(previous) <= 1e-8:

                result.append(
                    self.config.fill_value
                )

            else:

                result.append(
                    (
                        (
                            current
                            - previous
                        )
                        / abs(previous)
                    )
                    * 100.0
                )

        return result

    # ========================================================================
    # TREND
    # ========================================================================

    def trend(
        self,
        values: Sequence[float],
        window: int = 3,
    ) -> List[float]:
        """
        Rolling linear trend slope.

        Positive:
            upward trend.

        Negative:
            downward trend.
        """

        series = self._validate_series(
            values,
            "values",
        )

        self._validate_window(
            window
        )

        result = []

        x = list(
            range(window)
        )

        x_mean = statistics.fmean(x)

        denominator = sum(
            (
                item - x_mean
            ) ** 2
            for item in x
        )

        for index in range(
            len(series)
        ):

            if index + 1 < window:

                result.append(
                    self.config.fill_value
                )

                continue

            y = series[
                index + 1 - window:
                index + 1
            ]

            y_mean = statistics.fmean(y)

            numerator = sum(
                (
                    x_value - x_mean
                )
                * (
                    y_value - y_mean
                )
                for x_value, y_value
                in zip(x, y)
            )

            slope = (
                numerator
                / denominator
                if denominator
                else 0.0
            )

            result.append(
                slope
            )

        return result

    # ========================================================================
    # VOLATILITY
    # ========================================================================

    def volatility(
        self,
        values: Sequence[float],
        window: int = 3,
    ) -> List[float]:
        """Rolling standard deviation."""

        return self.rolling_std(
            values,
            window,
        )

    # ========================================================================
    # MOMENTUM
    # ========================================================================

    def momentum(
        self,
        values: Sequence[float],
        periods: int = 1,
    ) -> List[float]:
        """
        Absolute momentum.

        momentum = current - previous
        """

        series = self._validate_series(
            values,
            "values",
        )

        if periods < 1:
            raise InvalidFeatureInputError(
                "periods must be >= 1."
            )

        result = []

        for index, current in enumerate(
            series
        ):

            if index < periods:

                result.append(
                    self.config.fill_value
                )

            else:

                result.append(
                    current
                    - series[
                        index - periods
                    ]
                )

        return result

    # ========================================================================
    # DIFFERENCE
    # ========================================================================

    def difference(
        self,
        values: Sequence[float],
        periods: int = 1,
    ) -> List[float]:
        """Difference between current and lagged values."""

        return self.momentum(
            values,
            periods,
        )

    # ========================================================================
    # RATIO
    # ========================================================================

    def ratio(
        self,
        numerator: Sequence[float],
        denominator: Sequence[float],
    ) -> List[float]:
        """Element-wise ratio."""

        numerator_values = (
            self._validate_series(
                numerator,
                "numerator",
            )
        )

        denominator_values = (
            self._validate_series(
                denominator,
                "denominator",
            )
        )

        self._validate_equal_length(
            numerator_values,
            denominator_values,
        )

        result = []

        for num, den in zip(
            numerator_values,
            denominator_values,
        ):

            if abs(den) <= 1e-8:

                result.append(
                    self.config.fill_value
                )

            else:

                result.append(
                    num / den
                )

        return result

    # ========================================================================
    # SEASONAL FEATURES
    # ========================================================================

    def seasonal_index(
        self,
        values: Sequence[float],
        period: Optional[int] = None,
    ) -> List[float]:
        """
        Simple seasonal index.

        Current value divided by the historical
        seasonal-period average.
        """

        series = self._validate_series(
            values,
            "values",
        )

        period = (
            period
            or self.config.seasonal_period
        )

        if period < 1:
            raise InvalidFeatureInputError(
                "period must be >= 1."
            )

        result = []

        for index, current in enumerate(
            series
        ):

            if index < period:

                result.append(
                    self.config.fill_value
                )

                continue

            previous = series[
                index - period:
                index
            ]

            baseline = (
                statistics.fmean(
                    previous
                )
            )

            if abs(baseline) <= 1e-8:

                result.append(
                    self.config.fill_value
                )

            else:

                result.append(
                    current / baseline
                )

        return result

    # ========================================================================
    # FINANCIAL FEATURES
    # ========================================================================

    def profit_margin(
        self,
        revenue: Sequence[float],
        profit: Sequence[float],
    ) -> List[float]:
        """
        Profit margin percentage.

        profit / revenue * 100
        """

        revenue_values = (
            self._validate_series(
                revenue,
                "revenue",
            )
        )

        profit_values = (
            self._validate_series(
                profit,
                "profit",
            )
        )

        self._validate_equal_length(
            revenue_values,
            profit_values,
        )

        return [
            (
                profit_value
                / revenue_value
                * 100.0
                if abs(revenue_value) > 1e-8
                else 0.0
            )
            for revenue_value,
            profit_value
            in zip(
                revenue_values,
                profit_values,
            )
        ]

    def expense_ratio(
        self,
        revenue: Sequence[float],
        expenses: Sequence[float],
    ) -> List[float]:
        """
        Expense-to-revenue ratio.
        """

        return self.ratio(
            expenses,
            revenue,
        )

    def cash_conversion_ratio(
        self,
        inflow: Sequence[float],
        outflow: Sequence[float],
    ) -> List[float]:
        """
        Cash conversion ratio.

        inflow / outflow
        """

        return self.ratio(
            inflow,
            outflow,
        )

    # ========================================================================
    # FEATURE REGISTRY
    # ========================================================================

    def get_feature_definitions(
        self,
    ) -> List[FeatureDefinition]:
        """Return all registered features."""

        return list(
            self.feature_registry.values()
        )

    def get_feature_names(
        self,
    ) -> List[str]:
        """Return registered feature names."""

        return list(
            self.feature_registry.keys()
        )

    # ========================================================================
    # INTERNAL FEATURE BUILDERS
    # ========================================================================

    def _add_lag_features(
        self,
        features: Dict[str, List[float]],
        series: Sequence[float],
    ) -> None:

        for lag in self.config.lag_periods:

            features[
                f"lag_{lag}"
            ] = self.create_lag(
                series,
                lag,
            )

    def _add_rolling_features(
        self,
        features: Dict[str, List[float]],
        series: Sequence[float],
    ) -> None:

        for window in (
            self.config.rolling_windows
        ):

            features[
                f"rolling_mean_{window}"
            ] = self.rolling_mean(
                series,
                window,
            )

            features[
                f"rolling_std_{window}"
            ] = self.rolling_std(
                series,
                window,
            )

            features[
                f"rolling_min_{window}"
            ] = self.rolling_min(
                series,
                window,
            )

            features[
                f"rolling_max_{window}"
            ] = self.rolling_max(
                series,
                window,
            )

    def _add_growth_features(
        self,
        features: Dict[str, List[float]],
        series: Sequence[float],
    ) -> None:

        for period in (
            self.config.growth_periods
        ):

            features[
                f"growth_{period}"
            ] = self.growth_rate(
                series,
                period,
            )

    def _add_trend_features(
        self,
        features: Dict[str, List[float]],
        series: Sequence[float],
    ) -> None:

        for window in (
            self.config.trend_windows
        ):

            features[
                f"trend_{window}"
            ] = self.trend(
                series,
                window,
            )

    def _add_volatility_features(
        self,
        features: Dict[str, List[float]],
        series: Sequence[float],
    ) -> None:

        for window in (
            self.config.volatility_windows
        ):

            features[
                f"volatility_{window}"
            ] = self.volatility(
                series,
                window,
            )

    def _add_momentum_features(
        self,
        features: Dict[str, List[float]],
        series: Sequence[float],
    ) -> None:

        for period in (
            self.config.growth_periods
        ):

            features[
                f"momentum_{period}"
            ] = self.momentum(
                series,
                period,
            )

    def _add_difference_features(
        self,
        features: Dict[str, List[float]],
        series: Sequence[float],
    ) -> None:

        features[
            "diff_1"
        ] = self.difference(
            series,
            1,
        )

        features[
            "diff_seasonal"
        ] = self.difference(
            series,
            self.config.seasonal_period,
        )

    def _add_ratio_features(
        self,
        features: Dict[str, List[float]],
        series: Sequence[float],
    ) -> None:

        lag_1 = self.create_lag(
            series,
            1,
        )

        features[
            "value_to_lag1_ratio"
        ] = self.ratio(
            series,
            lag_1,
        )

        rolling_mean = self.rolling_mean(
            series,
            min(
                3,
                len(series),
            ),
        )

        features[
            "value_to_rolling_mean_ratio"
        ] = self.ratio(
            series,
            rolling_mean,
        )

    def _add_seasonal_features(
        self,
        features: Dict[str, List[float]],
        timestamps: Sequence[Any],
    ) -> None:
        """
        Generate calendar features.

        Timestamp objects may be:

            datetime
            pandas.Timestamp-like
            objects with month/day attributes
        """

        month = []
        quarter = []
        day_of_week = []
        day_of_month = []

        for timestamp in timestamps:

            month_value = self._timestamp_part(
                timestamp,
                "month",
                0,
            )

            day_value = self._timestamp_part(
                timestamp,
                "day",
                0,
            )

            weekday = self._weekday(
                timestamp
            )

            month.append(
                float(month_value)
            )

            quarter.append(
                float(
                    (
                        (
                            month_value - 1
                        )
                        // 3
                    )
                    + 1
                    if month_value
                    else 0
                )
            )

            day_of_week.append(
                float(weekday)
            )

            day_of_month.append(
                float(day_value)
            )

        features[
            "month"
        ] = month

        features[
            "quarter"
        ] = quarter

        features[
            "day_of_week"
        ] = day_of_week

        features[
            "day_of_month"
        ] = day_of_month

        features[
            "month_sin"
        ] = [
            math.sin(
                2.0
                * math.pi
                * value
                / 12.0
            )
            if value
            else 0.0
            for value in month
        ]

        features[
            "month_cos"
        ] = [
            math.cos(
                2.0
                * math.pi
                * value
                / 12.0
            )
            if value
            else 0.0
            for value in month
        ]

    def _add_financial_features(
        self,
        *,
        features: Dict[str, List[float]],
        revenue: Optional[
            Sequence[float]
        ],
        expenses: Optional[
            Sequence[float]
        ],
        profit: Optional[
            Sequence[float]
        ],
        cash_inflow: Optional[
            Sequence[float]
        ],
        cash_outflow: Optional[
            Sequence[float]
        ],
        length: int,
    ) -> None:

        if (
            revenue is not None
            and profit is not None
        ):

            margin = self.profit_margin(
                revenue,
                profit,
            )

            features[
                "profit_margin"
            ] = margin

            features[
                "profit_margin_growth"
            ] = self.growth_rate(
                margin,
                1,
            )

        if (
            revenue is not None
            and expenses is not None
        ):

            expense_ratio = (
                self.expense_ratio(
                    revenue,
                    expenses,
                )
            )

            features[
                "expense_to_revenue_ratio"
            ] = expense_ratio

            features[
                "expense_growth"
            ] = self.growth_rate(
                expenses,
                1,
            )

        if (
            cash_inflow is not None
            and cash_outflow is not None
        ):

            net_cash_flow = [
                inflow - outflow
                for inflow, outflow
                in zip(
                    cash_inflow,
                    cash_outflow,
                )
            ]

            features[
                "net_cash_flow"
            ] = net_cash_flow

            features[
                "cash_burn"
            ] = [
                max(
                    0.0,
                    outflow - inflow,
                )
                for inflow, outflow
                in zip(
                    cash_inflow,
                    cash_outflow,
                )
            ]

            features[
                "cash_conversion_ratio"
            ] = self.cash_conversion_ratio(
                cash_inflow,
                cash_outflow,
            )

            features[
                "cash_flow_growth"
            ] = self.growth_rate(
                net_cash_flow,
                1,
            )

        if profit is not None:

            features[
                "profit_growth"
            ] = self.growth_rate(
                profit,
                1,
            )

            features[
                "profit_momentum"
            ] = self.momentum(
                profit,
                1,
            )

        if revenue is not None:

            features[
                "revenue_growth"
            ] = self.growth_rate(
                revenue,
                1,
            )

            features[
                "revenue_momentum"
            ] = self.momentum(
                revenue,
                1,
            )

    # ========================================================================
    # REGISTRY
    # ========================================================================

    def _build_registry(
        self,
    ) -> Dict[
        str,
        FeatureDefinition,
    ]:

        registry: Dict[
            str,
            FeatureDefinition,
        ] = {}

        for lag in self.config.lag_periods:

            registry[
                f"lag_{lag}"
            ] = FeatureDefinition(
                name=f"lag_{lag}",
                feature_type=(
                    FeatureType.LAG
                ),
                description=(
                    f"Value from {lag} "
                    "period(s) earlier."
                ),
                lag=lag,
                required_history=lag + 1,
            )

        for window in (
            self.config.rolling_windows
        ):

            for suffix, description in (
                (
                    "mean",
                    "Rolling mean.",
                ),
                (
                    "std",
                    "Rolling standard deviation.",
                ),
                (
                    "min",
                    "Rolling minimum.",
                ),
                (
                    "max",
                    "Rolling maximum.",
                ),
            ):

                registry[
                    f"rolling_{suffix}_{window}"
                ] = FeatureDefinition(
                    name=(
                        f"rolling_{suffix}_{window}"
                    ),
                    feature_type=(
                        FeatureType.ROLLING
                    ),
                    description=description,
                    window=window,
                    required_history=window,
                )

        return registry

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _rolling_statistic(
        self,
        values: Sequence[float],
        window: int,
        function: Any,
    ) -> List[float]:

        series = self._validate_series(
            values,
            "values",
        )

        self._validate_window(
            window
        )

        result = []

        for index in range(
            len(series)
        ):

            if index + 1 < window:

                result.append(
                    self.config.fill_value
                )

                continue

            current = series[
                index + 1 - window:
                index + 1
            ]

            result.append(
                float(
                    function(current)
                )
            )

        return result

    def _fill_missing(
        self,
        features: Dict[
            str,
            List[float],
        ],
    ) -> None:

        for name, values in (
            features.items()
        ):

            for index, value in enumerate(
                values
            ):

                if not math.isfinite(
                    value
                ):

                    values[index] = (
                        self.config.fill_value
                    )

    def _validate_minimum_history(
        self,
        series: Sequence[float],
    ) -> None:

        if len(series) < (
            self.config.minimum_history
        ):

            raise InsufficientFeatureDataError(
                "Insufficient historical data. "
                f"At least "
                f"{self.config.minimum_history} "
                "observations are required."
            )

    @staticmethod
    def _validate_series(
        values: Sequence[float],
        name: str,
    ) -> List[float]:

        if values is None:
            raise InvalidFeatureInputError(
                f"{name} cannot be None."
            )

        try:

            result = [
                float(value)
                for value in values
            ]

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidFeatureInputError(
                f"{name} must contain numeric "
                "values."
            ) from exc

        if not result:

            raise InvalidFeatureInputError(
                f"{name} cannot be empty."
            )

        for value in result:

            if not math.isfinite(
                value
            ):

                raise InvalidFeatureInputError(
                    f"{name} contains non-finite "
                    "values."
                )

        return result

    @staticmethod
    def _validate_optional_series(
        values: Optional[
            Sequence[float]
        ],
        expected_length: int,
        name: str,
    ) -> None:

        if values is None:
            return

        if len(values) != expected_length:

            raise InvalidFeatureInputError(
                f"{name} must have length "
                f"{expected_length}."
            )

        for value in values:

            try:
                numeric = float(value)
            except (
                TypeError,
                ValueError,
            ) as exc:

                raise InvalidFeatureInputError(
                    f"{name} contains a non-numeric "
                    "value."
                ) from exc

            if not math.isfinite(
                numeric
            ):

                raise InvalidFeatureInputError(
                    f"{name} contains a non-finite "
                    "value."
                )

    @staticmethod
    def _validate_equal_length(
        first: Sequence[float],
        second: Sequence[float],
    ) -> None:

        if len(first) != len(second):

            raise InvalidFeatureInputError(
                "Input series must have equal "
                "lengths."
            )

    @staticmethod
    def _validate_window(
        window: int,
    ) -> None:

        if window < 1:

            raise InvalidFeatureInputError(
                "window must be >= 1."
            )

    @staticmethod
    def _timestamp_part(
        timestamp: Any,
        attribute: str,
        default: int,
    ) -> int:

        value = getattr(
            timestamp,
            attribute,
            default,
        )

        try:

            return int(value)

        except (
            TypeError,
            ValueError,
        ):

            return default

    @staticmethod
    def _weekday(
        timestamp: Any,
    ) -> int:

        method = getattr(
            timestamp,
            "weekday",
            None,
        )

        if callable(method):

            try:
                return int(
                    method()
                )

            except Exception:
                return 0

        return 0


# ============================================================================
# Convenience Functions
# ============================================================================


def create_forecast_features(
    values: Sequence[float],
    **kwargs: Any,
) -> ForecastFeatureResult:
    """Convenience feature-engineering API."""

    return ForecastFeatureEngineer().transform(
        values,
        **kwargs,
    )


def create_lag_features(
    values: Sequence[float],
    lags: Sequence[int] = (
        1,
        2,
        3,
        6,
        12,
    ),
) -> Dict[str, List[float]]:
    """Create only lag features."""

    engineer = ForecastFeatureEngineer()

    return {
        f"lag_{lag}":
            engineer.create_lag(
                values,
                lag,
            )
        for lag in lags
    }


def create_growth_features(
    values: Sequence[float],
    periods: Sequence[int] = (
        1,
        3,
        6,
        12,
    ),
) -> Dict[str, List[float]]:
    """Create growth features."""

    engineer = ForecastFeatureEngineer()

    return {
        f"growth_{period}":
            engineer.growth_rate(
                values,
                period,
            )
        for period in periods
    }


def create_rolling_features(
    values: Sequence[float],
    windows: Sequence[int] = (
        3,
        6,
        12,
    ),
) -> Dict[str, List[float]]:
    """Create rolling statistical features."""

    engineer = ForecastFeatureEngineer()

    features: Dict[
        str,
        List[float],
    ] = {}

    for window in windows:

        features[
            f"rolling_mean_{window}"
        ] = engineer.rolling_mean(
            values,
            window,
        )

        features[
            f"rolling_std_{window}"
        ] = engineer.rolling_std(
            values,
            window,
        )

        features[
            f"rolling_min_{window}"
        ] = engineer.rolling_min(
            values,
            window,
        )

        features[
            f"rolling_max_{window}"
        ] = engineer.rolling_max(
            values,
            window,
        )

    return features


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "ForecastFeatureError",
    "InvalidFeatureInputError",
    "InsufficientFeatureDataError",
    "FeatureCalculationError",
    "FeatureType",
    "FeatureDefinition",
    "ForecastFeatureConfig",
    "ForecastFeatureResult",
    "ForecastFeatureEngineer",
    "create_forecast_features",
    "create_lag_features",
    "create_growth_features",
    "create_rolling_features",
]