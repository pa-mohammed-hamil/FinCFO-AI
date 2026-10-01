"""
FinCo AI - Forecasting Preprocessing
====================================

Path:
    backend/app/forecasting/preprocessing.py

Purpose:
    Clean, validate, normalize, and prepare financial time-series data
    before feature engineering and forecasting.

Supported financial series:
    - Revenue
    - Expenses
    - Profit
    - Cash Flow
    - Liquidity
    - Generic financial KPIs

Pipeline:

    Raw Financial Data
            |
            v
    preprocessing.py
            |
            +--> Validation
            +--> Missing-value handling
            +--> Duplicate handling
            +--> Sorting
            +--> Outlier handling
            +--> Period normalization
            +--> Frequency checks
            +--> Growth calculation
            |
            v
        features.py
            |
            v
        model.py / baseline.py
            |
            v
       evaluation.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


# ============================================================================
# Exceptions
# ============================================================================


class ForecastPreprocessingError(Exception):
    """Base exception for forecasting preprocessing."""


class InvalidForecastDataError(ForecastPreprocessingError):
    """Raised when input forecast data is invalid."""


class MissingForecastValueError(ForecastPreprocessingError):
    """Raised when required values are missing."""


class InvalidPeriodError(ForecastPreprocessingError):
    """Raised when a period cannot be normalized."""


class InsufficientForecastDataError(ForecastPreprocessingError):
    """Raised when insufficient data is available."""


class ForecastOutlierError(ForecastPreprocessingError):
    """Raised when outlier processing fails."""


# ============================================================================
# Enums
# ============================================================================


class ForecastFrequency(str, Enum):
    """Supported time-series frequencies."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    UNKNOWN = "unknown"


class MissingValueStrategy(str, Enum):
    """Missing value handling strategies."""

    DROP = "drop"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    ZERO = "zero"
    MEAN = "mean"
    MEDIAN = "median"


class OutlierStrategy(str, Enum):
    """Outlier handling strategies."""

    NONE = "none"
    CLIP = "clip"
    REMOVE = "remove"
    MEDIAN = "median"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ForecastPreprocessingConfig:
    """
    Configuration for financial time-series preprocessing.
    """

    period_field: str = "period"

    value_field: str = "value"

    revenue_field: str = "revenue"

    expense_field: str = "expenses"

    profit_field: str = "profit"

    cash_flow_field: str = "cash_flow"

    minimum_history: int = 3

    missing_value_strategy: MissingValueStrategy = (
        MissingValueStrategy.INTERPOLATE
    )

    outlier_strategy: OutlierStrategy = (
        OutlierStrategy.CLIP
    )

    outlier_zscore_threshold: float = 3.0

    outlier_iqr_multiplier: float = 1.5

    sort_periods: bool = True

    remove_duplicates: bool = True

    aggregate_duplicate_periods: bool = True

    fill_missing_periods: bool = False

    clip_negative_revenue: bool = True

    clip_negative_expenses: bool = True

    allow_negative_profit: bool = True

    allow_negative_cash_flow: bool = True

    calculate_growth_features: bool = True

    calculate_margin_features: bool = True

    normalize_numeric_values: bool = True

    currency: str = "USD"

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate preprocessing configuration."""

        if self.minimum_history < 2:
            raise InvalidForecastDataError(
                "minimum_history must be >= 2."
            )

        if not self.period_field:
            raise InvalidForecastDataError(
                "period_field cannot be empty."
            )

        if not self.value_field:
            raise InvalidForecastDataError(
                "value_field cannot be empty."
            )

        if self.outlier_zscore_threshold <= 0:
            raise InvalidForecastDataError(
                "outlier_zscore_threshold must be > 0."
            )

        if self.outlier_iqr_multiplier <= 0:
            raise InvalidForecastDataError(
                "outlier_iqr_multiplier must be > 0."
            )

        if not self.currency:
            raise InvalidForecastDataError(
                "currency cannot be empty."
            )


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class ForecastRecord:
    """
    Normalized financial time-series record.
    """

    period: Any

    value: float

    original_value: Optional[float] = None

    revenue: Optional[float] = None

    expenses: Optional[float] = None

    profit: Optional[float] = None

    cash_flow: Optional[float] = None

    growth_rate: Optional[float] = None

    margin: Optional[float] = None

    is_missing: bool = False

    is_outlier: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "value": self.value,
            "original_value": self.original_value,
            "revenue": self.revenue,
            "expenses": self.expenses,
            "profit": self.profit,
            "cash_flow": self.cash_flow,
            "growth_rate": self.growth_rate,
            "margin": self.margin,
            "is_missing": self.is_missing,
            "is_outlier": self.is_outlier,
            "metadata": dict(self.metadata),
        }


@dataclass
class ForecastPreprocessingResult:
    """
    Result returned by the preprocessing pipeline.
    """

    records: List[ForecastRecord]

    original_count: int

    processed_count: int

    removed_count: int

    missing_values_handled: int

    outliers_detected: int

    outliers_handled: int

    frequency: ForecastFrequency

    value_mean: float

    value_median: float

    value_std: float

    value_min: float

    value_max: float

    created_at: datetime

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def values(self) -> List[float]:
        """Return processed values."""

        return [
            record.value
            for record in self.records
        ]

    @property
    def periods(self) -> List[Any]:
        """Return processed periods."""

        return [
            record.period
            for record in self.records
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "records": [
                record.to_dict()
                for record in self.records
            ],
            "original_count": self.original_count,
            "processed_count": self.processed_count,
            "removed_count": self.removed_count,
            "missing_values_handled": self.missing_values_handled,
            "outliers_detected": self.outliers_detected,
            "outliers_handled": self.outliers_handled,
            "frequency": self.frequency.value,
            "value_mean": self.value_mean,
            "value_median": self.value_median,
            "value_std": self.value_std,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


# ============================================================================
# Main Preprocessor
# ============================================================================


class ForecastPreprocessor:
    """
    Production-oriented preprocessing engine for financial time series.

    The preprocessor intentionally does not perform model training.
    Its responsibility is to transform raw financial data into a stable,
    validated time series.
    """

    def __init__(
        self,
        config: Optional[ForecastPreprocessingConfig] = None,
    ) -> None:

        self.config = (
            config
            or ForecastPreprocessingConfig()
        )

        self.config.validate()

    # ========================================================================
    # Main Pipeline
    # ========================================================================

    def process(
        self,
        data: Sequence[Any],
        value_field: Optional[str] = None,
        frequency: Optional[ForecastFrequency] = None,
    ) -> ForecastPreprocessingResult:
        """
        Execute the complete preprocessing pipeline.

        Example input:

            [
                {"period": "2026-01", "value": 100000},
                {"period": "2026-02", "value": 105000},
                {"period": "2026-03", "value": 110000},
            ]
        """

        if data is None:
            raise InvalidForecastDataError(
                "data cannot be None."
            )

        if len(data) == 0:
            raise InvalidForecastDataError(
                "data cannot be empty."
            )

        selected_value_field = (
            value_field
            or self.config.value_field
        )

        original_count = len(data)

        records = self._normalize_records(
            data=data,
            value_field=selected_value_field,
        )

        if self.config.remove_duplicates:
            records = self._handle_duplicates(records)

        if self.config.sort_periods:
            records = self._sort_records(records)

        records, missing_handled = (
            self._handle_missing_values(records)
        )

        records, outliers_detected, outliers_handled = (
            self._handle_outliers(records)
        )

        records = self._calculate_derived_features(
            records
        )

        detected_frequency = (
            frequency
            if frequency is not None
            else self.detect_frequency(
                [record.period for record in records]
            )
        )

        self._validate_processed_history(records)

        values = np.asarray(
            [record.value for record in records],
            dtype=float,
        )

        return ForecastPreprocessingResult(
            records=records,
            original_count=original_count,
            processed_count=len(records),
            removed_count=(
                original_count - len(records)
            ),
            missing_values_handled=missing_handled,
            outliers_detected=outliers_detected,
            outliers_handled=outliers_handled,
            frequency=detected_frequency,
            value_mean=float(np.mean(values)),
            value_median=float(np.median(values)),
            value_std=float(np.std(values)),
            value_min=float(np.min(values)),
            value_max=float(np.max(values)),
            created_at=self._utc_now(),
            metadata={
                **self.config.metadata,
                "value_field": selected_value_field,
                "currency": self.config.currency,
            },
        )

    # ========================================================================
    # Financial Dataset Processing
    # ========================================================================

    def process_financial_data(
        self,
        data: Sequence[Any],
        frequency: Optional[ForecastFrequency] = None,
    ) -> ForecastPreprocessingResult:
        """
        Process a financial dataset containing revenue, expenses,
        profit and/or cash flow.

        Example:

            {
                "period": "2026-01",
                "revenue": 100000,
                "expenses": 70000,
                "profit": 30000,
                "cash_flow": 25000
            }
        """

        if not data:
            raise InvalidForecastDataError(
                "data cannot be empty."
            )

        records: List[ForecastRecord] = []

        for index, item in enumerate(data):

            if isinstance(item, ForecastRecord):
                records.append(item)
                continue

            if not isinstance(item, Mapping):
                raise InvalidForecastDataError(
                    f"Financial record {index} must be a mapping."
                )

            period = self._normalize_period(
                item.get(
                    self.config.period_field,
                    index,
                )
            )

            revenue = self._optional_number(
                item.get(
                    self.config.revenue_field
                )
            )

            expenses = self._optional_number(
                item.get(
                    self.config.expense_field
                )
            )

            profit = self._optional_number(
                item.get(
                    self.config.profit_field
                )
            )

            cash_flow = self._optional_number(
                item.get(
                    self.config.cash_flow_field
                )
            )

            if revenue is not None:
                if self.config.clip_negative_revenue:
                    revenue = max(0.0, revenue)

            if expenses is not None:
                if self.config.clip_negative_expenses:
                    expenses = max(0.0, expenses)

            if profit is None and (
                revenue is not None
                and expenses is not None
            ):
                profit = revenue - expenses

            if (
                profit is not None
                and not self.config.allow_negative_profit
            ):
                profit = max(0.0, profit)

            if (
                cash_flow is not None
                and not self.config.allow_negative_cash_flow
            ):
                cash_flow = max(0.0, cash_flow)

            primary_value = self._select_primary_value(
                item=item,
                revenue=revenue,
                expenses=expenses,
                profit=profit,
                cash_flow=cash_flow,
            )

            records.append(
                ForecastRecord(
                    period=period,
                    value=primary_value,
                    original_value=primary_value,
                    revenue=revenue,
                    expenses=expenses,
                    profit=profit,
                    cash_flow=cash_flow,
                    metadata={
                        "source_index": index,
                    },
                )
            )

        records = self._sort_records(records)

        if self.config.remove_duplicates:
            records = self._handle_duplicates(
                records
            )

        records, missing_handled = (
            self._handle_missing_values(records)
        )

        records, outliers_detected, outliers_handled = (
            self._handle_outliers(records)
        )

        records = self._calculate_financial_features(
            records
        )

        detected_frequency = (
            frequency
            if frequency is not None
            else self.detect_frequency(
                [record.period for record in records]
            )
        )

        self._validate_processed_history(records)

        values = np.asarray(
            [record.value for record in records],
            dtype=float,
        )

        return ForecastPreprocessingResult(
            records=records,
            original_count=len(data),
            processed_count=len(records),
            removed_count=(
                len(data) - len(records)
            ),
            missing_values_handled=missing_handled,
            outliers_detected=outliers_detected,
            outliers_handled=outliers_handled,
            frequency=detected_frequency,
            value_mean=float(np.mean(values)),
            value_median=float(np.median(values)),
            value_std=float(np.std(values)),
            value_min=float(np.min(values)),
            value_max=float(np.max(values)),
            created_at=self._utc_now(),
            metadata={
                **self.config.metadata,
                "dataset_type": "financial",
                "currency": self.config.currency,
            },
        )

    # ========================================================================
    # Normalization
    # ========================================================================

    def _normalize_records(
        self,
        data: Sequence[Any],
        value_field: str,
    ) -> List[ForecastRecord]:

        records: List[ForecastRecord] = []

        for index, item in enumerate(data):

            if isinstance(item, ForecastRecord):
                records.append(item)
                continue

            if isinstance(item, Mapping):

                period = self._normalize_period(
                    item.get(
                        self.config.period_field,
                        index,
                    )
                )

                raw_value = item.get(
                    value_field
                )

                value = self._number_or_missing(
                    raw_value,
                    f"value[{index}]",
                )

                records.append(
                    ForecastRecord(
                        period=period,
                        value=(
                            value
                            if value is not None
                            else 0.0
                        ),
                        original_value=value,
                        is_missing=value is None,
                        metadata={
                            "source_index": index,
                        },
                    )
                )

                continue

            if isinstance(item, Sequence) and not isinstance(
                item,
                (str, bytes),
            ):

                if len(item) < 2:
                    raise InvalidForecastDataError(
                        f"Record {index} must contain "
                        "period and value."
                    )

                period = self._normalize_period(
                    item[0]
                )

                value = self._number_or_missing(
                    item[1],
                    f"value[{index}]",
                )

                records.append(
                    ForecastRecord(
                        period=period,
                        value=(
                            value
                            if value is not None
                            else 0.0
                        ),
                        original_value=value,
                        is_missing=value is None,
                        metadata={
                            "source_index": index,
                        },
                    )
                )

                continue

            raise InvalidForecastDataError(
                f"Unsupported record format at index {index}."
            )

        return records

    def _select_primary_value(
        self,
        item: Mapping[str, Any],
        revenue: Optional[float],
        expenses: Optional[float],
        profit: Optional[float],
        cash_flow: Optional[float],
    ) -> float:

        if profit is not None:
            return profit

        if revenue is not None:
            return revenue

        if cash_flow is not None:
            return cash_flow

        if expenses is not None:
            return expenses

        explicit_value = item.get(
            self.config.value_field
        )

        if explicit_value is not None:
            return self._number(
                explicit_value,
                self.config.value_field,
            )

        raise MissingForecastValueError(
            "Financial record does not contain a usable "
            "financial value."
        )

    # ========================================================================
    # Missing Values
    # ========================================================================

    def _handle_missing_values(
        self,
        records: List[ForecastRecord],
    ) -> tuple[List[ForecastRecord], int]:

        missing_indices = [
            index
            for index, record in enumerate(records)
            if record.is_missing
            or record.original_value is None
        ]

        if not missing_indices:
            return records, 0

        strategy = self.config.missing_value_strategy

        if strategy == MissingValueStrategy.DROP:

            cleaned = [
                record
                for record in records
                if not record.is_missing
                and record.original_value is not None
            ]

            return cleaned, len(missing_indices)

        values = np.asarray(
            [
                np.nan
                if (
                    record.is_missing
                    or record.original_value is None
                )
                else record.value
                for record in records
            ],
            dtype=float,
        )

        valid_values = values[
            np.isfinite(values)
        ]

        if len(valid_values) == 0:
            raise MissingForecastValueError(
                "No valid values remain after missing-value detection."
            )

        if strategy == MissingValueStrategy.ZERO:
            replacement_values = np.zeros(
                len(values)
            )

        elif strategy == MissingValueStrategy.MEAN:
            replacement_values = np.full(
                len(values),
                float(np.mean(valid_values)),
            )

        elif strategy == MissingValueStrategy.MEDIAN:
            replacement_values = np.full(
                len(values),
                float(np.median(valid_values)),
            )

        elif strategy == MissingValueStrategy.FORWARD_FILL:
            replacement_values = self._forward_fill(
                values
            )

        elif strategy == MissingValueStrategy.BACKWARD_FILL:
            replacement_values = self._backward_fill(
                values
            )

        elif strategy == MissingValueStrategy.INTERPOLATE:
            replacement_values = self._interpolate(
                values
            )

        else:
            raise InvalidForecastDataError(
                f"Unsupported missing-value strategy: {strategy}"
            )

        handled = 0

        for index, record in enumerate(records):

            if record.is_missing or record.original_value is None:

                replacement = replacement_values[index]

                if not isfinite(float(replacement)):
                    replacement = float(
                        np.mean(valid_values)
                    )

                record.value = float(replacement)

                record.is_missing = True

                record.metadata[
                    "missing_value_strategy"
                ] = strategy.value

                handled += 1

        return records, handled

    @staticmethod
    def _forward_fill(
        values: np.ndarray,
    ) -> np.ndarray:

        result = values.copy()

        last_value: Optional[float] = None

        for index, value in enumerate(result):

            if np.isfinite(value):
                last_value = float(value)

            elif last_value is not None:
                result[index] = last_value

        if np.isnan(result[0]):
            valid = result[
                np.isfinite(result)
            ]

            if len(valid):
                first = float(valid[0])

                for index in range(len(result)):
                    if np.isnan(result[index]):
                        result[index] = first
                    else:
                        break

        return result

    @staticmethod
    def _backward_fill(
        values: np.ndarray,
    ) -> np.ndarray:

        result = values.copy()

        next_value: Optional[float] = None

        for index in range(
            len(result) - 1,
            -1,
            -1,
        ):

            value = result[index]

            if np.isfinite(value):
                next_value = float(value)

            elif next_value is not None:
                result[index] = next_value

        return result

    @staticmethod
    def _interpolate(
        values: np.ndarray,
    ) -> np.ndarray:

        result = values.copy()

        indices = np.arange(
            len(result)
        )

        valid = np.isfinite(result)

        if valid.sum() == 0:
            raise MissingForecastValueError(
                "Cannot interpolate without valid values."
            )

        if valid.sum() == 1:
            result[~valid] = result[valid][0]
            return result

        result[~valid] = np.interp(
            indices[~valid],
            indices[valid],
            result[valid],
        )

        return result

    # ========================================================================
    # Outlier Handling
    # ========================================================================

    def _handle_outliers(
        self,
        records: List[ForecastRecord],
    ) -> tuple[
        List[ForecastRecord],
        int,
        int,
    ]:

        if (
            self.config.outlier_strategy
            == OutlierStrategy.NONE
        ):
            return records, 0, 0

        values = np.asarray(
            [record.value for record in records],
            dtype=float,
        )

        if len(values) < 4:
            return records, 0, 0

        outlier_mask = self._detect_outliers(
            values
        )

        detected = int(
            np.sum(outlier_mask)
        )

        if detected == 0:
            return records, 0, 0

        strategy = self.config.outlier_strategy

        if strategy == OutlierStrategy.CLIP:

            lower, upper = self._iqr_bounds(
                values
            )

            for index, is_outlier in enumerate(
                outlier_mask
            ):
                if is_outlier:
                    records[index].is_outlier = True

                    records[index].value = float(
                        np.clip(
                            records[index].value,
                            lower,
                            upper,
                        )
                    )

                    records[index].metadata[
                        "outlier_strategy"
                    ] = strategy.value

            return records, detected, detected

        if strategy == OutlierStrategy.MEDIAN:

            median = float(
                np.median(values)
            )

            for index, is_outlier in enumerate(
                outlier_mask
            ):
                if is_outlier:
                    records[index].is_outlier = True
                    records[index].value = median
                    records[index].metadata[
                        "outlier_strategy"
                    ] = strategy.value

            return records, detected, detected

        if strategy == OutlierStrategy.REMOVE:

            cleaned = []

            for index, record in enumerate(records):

                if outlier_mask[index]:
                    record.is_outlier = True
                    continue

                cleaned.append(record)

            return cleaned, detected, detected

        raise ForecastOutlierError(
            f"Unsupported outlier strategy: {strategy}"
        )

    def _detect_outliers(
        self,
        values: np.ndarray,
    ) -> np.ndarray:

        mean = float(
            np.mean(values)
        )

        std = float(
            np.std(values)
        )

        zscore_mask = np.zeros(
            len(values),
            dtype=bool,
        )

        if std > 1e-12:

            zscores = np.abs(
                (values - mean) / std
            )

            zscore_mask = (
                zscores
                > self.config.outlier_zscore_threshold
            )

        lower, upper = self._iqr_bounds(
            values
        )

        iqr_mask = (
            (values < lower)
            | (values > upper)
        )

        return zscore_mask | iqr_mask

    def _iqr_bounds(
        self,
        values: np.ndarray,
    ) -> tuple[float, float]:

        q1 = float(
            np.percentile(values, 25)
        )

        q3 = float(
            np.percentile(values, 75)
        )

        iqr = q3 - q1

        lower = (
            q1
            - self.config.outlier_iqr_multiplier * iqr
        )

        upper = (
            q3
            + self.config.outlier_iqr_multiplier * iqr
        )

        return lower, upper

    # ========================================================================
    # Derived Features
    # ========================================================================

    def _calculate_derived_features(
        self,
        records: List[ForecastRecord],
    ) -> List[ForecastRecord]:

        if not records:
            return records

        for index, record in enumerate(records):

            if self.config.calculate_growth_features:

                if index == 0:
                    record.growth_rate = None

                else:
                    previous = records[
                        index - 1
                    ].value

                    if previous == 0:
                        record.growth_rate = None

                    else:
                        record.growth_rate = float(
                            (
                                (
                                    record.value
                                    - previous
                                )
                                / abs(previous)
                            )
                            * 100.0
                        )

        return records

    def _calculate_financial_features(
        self,
        records: List[ForecastRecord],
    ) -> List[ForecastRecord]:

        for index, record in enumerate(records):

            if (
                self.config.calculate_growth_features
                and index > 0
            ):

                previous = records[
                    index - 1
                ].value

                if previous != 0:
                    record.growth_rate = float(
                        (
                            (
                                record.value
                                - previous
                            )
                            / abs(previous)
                        )
                        * 100.0
                    )

            if self.config.calculate_margin_features:

                if (
                    record.revenue is not None
                    and record.profit is not None
                    and record.revenue != 0
                ):
                    record.margin = float(
                        (
                            record.profit
                            / record.revenue
                        )
                        * 100.0
                    )

        return records

    # ========================================================================
    # Duplicate Handling
    # ========================================================================

    def _handle_duplicates(
        self,
        records: List[ForecastRecord],
    ) -> List[ForecastRecord]:

        if not records:
            return records

        grouped: Dict[str, List[ForecastRecord]] = {}

        for record in records:

            key = str(record.period)

            grouped.setdefault(
                key,
                [],
            ).append(record)

        result: List[ForecastRecord] = []

        for group in grouped.values():

            if len(group) == 1:
                result.append(group[0])
                continue

            if not self.config.aggregate_duplicate_periods:
                result.append(group[-1])
                continue

            aggregated = self._aggregate_records(
                group
            )

            result.append(aggregated)

        return result

    def _aggregate_records(
        self,
        records: Sequence[ForecastRecord],
    ) -> ForecastRecord:

        first = records[0]

        values = [
            record.value
            for record in records
            if isfinite(record.value)
        ]

        if not values:
            raise InvalidForecastDataError(
                "Cannot aggregate duplicate records without values."
            )

        revenue_values = [
            record.revenue
            for record in records
            if record.revenue is not None
        ]

        expense_values = [
            record.expenses
            for record in records
            if record.expenses is not None
        ]

        profit_values = [
            record.profit
            for record in records
            if record.profit is not None
        ]

        cashflow_values = [
            record.cash_flow
            for record in records
            if record.cash_flow is not None
        ]

        return ForecastRecord(
            period=first.period,
            value=float(np.sum(values)),
            original_value=float(np.sum(values)),
            revenue=(
                float(np.sum(revenue_values))
                if revenue_values
                else None
            ),
            expenses=(
                float(np.sum(expense_values))
                if expense_values
                else None
            ),
            profit=(
                float(np.sum(profit_values))
                if profit_values
                else None
            ),
            cash_flow=(
                float(np.sum(cashflow_values))
                if cashflow_values
                else None
            ),
            metadata={
                "aggregated_duplicates": len(records),
            },
        )

    # ========================================================================
    # Sorting
    # ========================================================================

    @staticmethod
    def _sort_records(
        records: List[ForecastRecord],
    ) -> List[ForecastRecord]:

        try:
            return sorted(
                records,
                key=lambda record: record.period,
            )
        except TypeError:

            return sorted(
                records,
                key=lambda record: str(record.period),
            )

    # ========================================================================
    # Period Handling
    # ========================================================================

    def _normalize_period(
        self,
        period: Any,
    ) -> Any:
        """
        Normalize common period representations.

        Supported:
            datetime
            date
            strings such as:
                2026-01
                2026-01-15
                2026-01-15T00:00:00
        """

        if period is None:
            raise InvalidPeriodError(
                "Period cannot be None."
            )

        if isinstance(
            period,
            datetime,
        ):
            return period

        if hasattr(period, "isoformat"):
            return period

        if isinstance(period, str):

            text = period.strip()

            if not text:
                raise InvalidPeriodError(
                    "Period cannot be empty."
                )

            # Monthly representation remains monthly.
            if len(text) == 7 and text[4] == "-":
                try:
                    datetime.strptime(
                        text,
                        "%Y-%m",
                    )
                    return text
                except ValueError:
                    pass

            formats = (
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
            )

            for fmt in formats:
                try:
                    return datetime.strptime(
                        text,
                        fmt,
                    )
                except ValueError:
                    continue

            return text

        if isinstance(
            period,
            (int, float),
        ):
            if not isfinite(float(period)):
                raise InvalidPeriodError(
                    "Numeric period must be finite."
                )

            return period

        return period

    def detect_frequency(
        self,
        periods: Sequence[Any],
    ) -> ForecastFrequency:
        """Infer approximate time-series frequency."""

        if len(periods) < 2:
            return ForecastFrequency.UNKNOWN

        datetimes: List[datetime] = []

        for period in periods:

            if isinstance(
                period,
                datetime,
            ):
                datetimes.append(period)

            elif isinstance(period, str):

                parsed = self._try_parse_datetime(
                    period
                )

                if parsed is not None:
                    datetimes.append(parsed)

        if len(datetimes) < 2:
            return ForecastFrequency.UNKNOWN

        datetimes.sort()

        differences = [
            (
                datetimes[index + 1]
                - datetimes[index]
            ).days
            for index in range(
                len(datetimes) - 1
            )
        ]

        differences = [
            value
            for value in differences
            if value > 0
        ]

        if not differences:
            return ForecastFrequency.UNKNOWN

        median_days = float(
            np.median(differences)
        )

        if median_days <= 2:
            return ForecastFrequency.DAILY

        if median_days <= 10:
            return ForecastFrequency.WEEKLY

        if median_days <= 45:
            return ForecastFrequency.MONTHLY

        if median_days <= 120:
            return ForecastFrequency.QUARTERLY

        if median_days <= 400:
            return ForecastFrequency.YEARLY

        return ForecastFrequency.UNKNOWN

    @staticmethod
    def _try_parse_datetime(
        value: str,
    ) -> Optional[datetime]:

        formats = (
            "%Y-%m",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        )

        for fmt in formats:

            try:
                return datetime.strptime(
                    value,
                    fmt,
                )
            except ValueError:
                continue

        return None

    # ========================================================================
    # Validation
    # ========================================================================

    def _validate_processed_history(
        self,
        records: Sequence[ForecastRecord],
    ) -> None:

        if len(records) < self.config.minimum_history:
            raise InsufficientForecastDataError(
                f"At least "
                f"{self.config.minimum_history} "
                f"valid periods are required; "
                f"received {len(records)}."
            )

        for index, record in enumerate(records):

            if not isfinite(
                float(record.value)
            ):
                raise InvalidForecastDataError(
                    f"Record {index} contains an invalid value."
                )

    # ========================================================================
    # Numeric Helpers
    # ========================================================================

    @staticmethod
    def _number(
        value: Any,
        field_name: str,
    ) -> float:

        try:
            number = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise InvalidForecastDataError(
                f"{field_name} must be numeric."
            ) from exc

        if not isfinite(number):
            raise InvalidForecastDataError(
                f"{field_name} must be finite."
            )

        return number

    def _number_or_missing(
        self,
        value: Any,
        field_name: str,
    ) -> Optional[float]:

        if value is None:
            return None

        if isinstance(value, str):

            text = value.strip()

            if not text:
                return None

            if text.lower() in {
                "null",
                "none",
                "nan",
                "na",
                "n/a",
                "-",
            }:
                return None

        return self._number(
            value,
            field_name,
        )

    def _optional_number(
        self,
        value: Any,
    ) -> Optional[float]:

        return self._number_or_missing(
            value,
            "financial_value",
        )

    # ========================================================================
    # Utility Statistics
    # ========================================================================

    @staticmethod
    def calculate_growth_rates(
        values: Sequence[float],
    ) -> List[Optional[float]]:
        """Calculate period-over-period growth percentages."""

        if not values:
            return []

        result: List[Optional[float]] = [None]

        for index in range(
            1,
            len(values),
        ):

            previous = float(
                values[index - 1]
            )

            current = float(
                values[index]
            )

            if previous == 0:
                result.append(None)
            else:
                result.append(
                    (
                        (current - previous)
                        / abs(previous)
                    )
                    * 100.0
                )

        return result

    @staticmethod
    def calculate_statistics(
        values: Sequence[float],
    ) -> Dict[str, float]:
        """Calculate basic descriptive statistics."""

        if not values:
            raise InvalidForecastDataError(
                "values cannot be empty."
            )

        array = np.asarray(
            values,
            dtype=float,
        )

        if not np.all(
            np.isfinite(array)
        ):
            raise InvalidForecastDataError(
                "values contain invalid numbers."
            )

        return {
            "count": float(len(array)),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "std": float(np.std(array)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
        }

    @staticmethod
    def calculate_volatility(
        values: Sequence[float],
    ) -> float:
        """Calculate coefficient of variation as a percentage."""

        if len(values) < 2:
            return 0.0

        array = np.asarray(
            values,
            dtype=float,
        )

        mean_abs = float(
            np.mean(np.abs(array))
        )

        if mean_abs <= 1e-12:
            return 0.0

        return float(
            (
                np.std(array)
                / mean_abs
            )
            * 100.0
        )

    @staticmethod
    def calculate_trend(
        values: Sequence[float],
    ) -> float:
        """Calculate linear trend slope."""

        if len(values) < 2:
            return 0.0

        array = np.asarray(
            values,
            dtype=float,
        )

        x = np.arange(
            len(array),
            dtype=float,
        )

        slope, _ = np.polyfit(
            x,
            array,
            1,
        )

        return float(slope)

    # ========================================================================
    # Data Extraction Helpers
    # ========================================================================

    @staticmethod
    def to_arrays(
        result: ForecastPreprocessingResult,
    ) -> tuple[List[Any], List[float]]:
        """Return periods and values for downstream forecasting."""

        return (
            result.periods,
            result.values,
        )

    @staticmethod
    def to_matrix(
        result: ForecastPreprocessingResult,
        include_financial_fields: bool = True,
    ) -> np.ndarray:
        """
        Convert processed records into a numeric feature matrix.

        Matrix columns:

            value
            revenue
            expenses
            profit
            cash_flow
            growth_rate
            margin
        """

        rows: List[List[float]] = []

        for record in result.records:

            row = [
                float(record.value),
            ]

            if include_financial_fields:
                row.extend(
                    [
                        (
                            float(record.revenue)
                            if record.revenue is not None
                            else 0.0
                        ),
                        (
                            float(record.expenses)
                            if record.expenses is not None
                            else 0.0
                        ),
                        (
                            float(record.profit)
                            if record.profit is not None
                            else 0.0
                        ),
                        (
                            float(record.cash_flow)
                            if record.cash_flow is not None
                            else 0.0
                        ),
                        (
                            float(record.growth_rate)
                            if record.growth_rate is not None
                            else 0.0
                        ),
                        (
                            float(record.margin)
                            if record.margin is not None
                            else 0.0
                        ),
                    ]
                )

            rows.append(row)

        return np.asarray(
            rows,
            dtype=float,
        )

    # ========================================================================
    # Time-Series Validation
    # ========================================================================

    def validate_continuity(
        self,
        periods: Sequence[Any],
        frequency: ForecastFrequency,
    ) -> Dict[str, Any]:
        """
        Check whether periods appear continuous.

        This is diagnostic only. It does not modify the dataset.
        """

        if len(periods) < 2:
            return {
                "continuous": True,
                "missing_periods": 0,
                "frequency": frequency.value,
            }

        datetimes = []

        for period in periods:

            if isinstance(
                period,
                datetime,
            ):
                datetimes.append(period)

            elif isinstance(period, str):

                parsed = self._try_parse_datetime(
                    period
                )

                if parsed is not None:
                    datetimes.append(parsed)

        if len(datetimes) < 2:
            return {
                "continuous": True,
                "missing_periods": 0,
                "frequency": frequency.value,
                "diagnostic": "Non-date periods cannot be checked.",
            }

        datetimes.sort()

        expected_days = {
            ForecastFrequency.DAILY: 1,
            ForecastFrequency.WEEKLY: 7,
            ForecastFrequency.MONTHLY: 30,
            ForecastFrequency.QUARTERLY: 90,
            ForecastFrequency.YEARLY: 365,
        }.get(frequency)

        if expected_days is None:
            return {
                "continuous": True,
                "missing_periods": 0,
                "frequency": frequency.value,
            }

        gaps = []

        for index in range(
            len(datetimes) - 1
        ):

            difference = (
                datetimes[index + 1]
                - datetimes[index]
            ).days

            if difference > expected_days * 1.5:
                gaps.append(
                    {
                        "from": datetimes[index],
                        "to": datetimes[index + 1],
                        "days": difference,
                    }
                )

        return {
            "continuous": len(gaps) == 0,
            "missing_periods": len(gaps),
            "frequency": frequency.value,
            "gaps": gaps,
        }

    # ========================================================================
    # Timestamp
    # ========================================================================

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)


# ============================================================================
# Convenience Functions
# ============================================================================


def preprocess_forecast_data(
    data: Sequence[Any],
    value_field: str = "value",
    config: Optional[ForecastPreprocessingConfig] = None,
) -> ForecastPreprocessingResult:
    """
    Convenience preprocessing function.
    """

    processor = ForecastPreprocessor(
        config=config
    )

    return processor.process(
        data=data,
        value_field=value_field,
    )


def preprocess_financial_data(
    data: Sequence[Any],
    config: Optional[ForecastPreprocessingConfig] = None,
) -> ForecastPreprocessingResult:
    """
    Convenience function for financial datasets.
    """

    processor = ForecastPreprocessor(
        config=config
    )

    return processor.process_financial_data(
        data=data
    )


def calculate_growth_rates(
    values: Sequence[float],
) -> List[Optional[float]]:
    """Convenience growth-rate calculation."""

    return ForecastPreprocessor.calculate_growth_rates(
        values
    )


def calculate_forecast_statistics(
    values: Sequence[float],
) -> Dict[str, float]:
    """Convenience statistics calculation."""

    return ForecastPreprocessor.calculate_statistics(
        values
    )


def calculate_forecast_volatility(
    values: Sequence[float],
) -> float:
    """Convenience volatility calculation."""

    return ForecastPreprocessor.calculate_volatility(
        values
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "ForecastPreprocessingError",
    "InvalidForecastDataError",
    "MissingForecastValueError",
    "InvalidPeriodError",
    "InsufficientForecastDataError",
    "ForecastOutlierError",

    # Enums
    "ForecastFrequency",
    "MissingValueStrategy",
    "OutlierStrategy",

    # Configuration
    "ForecastPreprocessingConfig",

    # Data classes
    "ForecastRecord",
    "ForecastPreprocessingResult",

    # Main processor
    "ForecastPreprocessor",

    # Convenience functions
    "preprocess_forecast_data",
    "preprocess_financial_data",
    "calculate_growth_rates",
    "calculate_forecast_statistics",
    "calculate_forecast_volatility",
]