"""
FinCo AI - Feature Engineering

File:
    ml/fraud/feature_engineering.py

Purpose:
    Transform raw financial transaction data into ML-ready
    features for fraud detection.

Feature groups:
    - Transaction features
    - Temporal features
    - Amount features
    - Customer behavioral features
    - Velocity features
    - Merchant features
    - Location features
    - Device features
    - Risk indicators

Design goals:
    - Reusable
    - Production-friendly
    - Leakage-aware
    - Compatible with pandas / scikit-learn
    - Train/test consistent
    - Handles missing values
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("data/processed")

DEFAULT_INPUT_FILE = (
    Path("data/raw/transactions.csv")
)

DEFAULT_OUTPUT_FILE = (
    DEFAULT_OUTPUT_DIR / "fraud_features.csv"
)

EPSILON = 1e-8

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature Configuration
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """
    Configuration for FinCo AI fraud feature engineering.
    """

    transaction_id: str = "transaction_id"
    customer_id: str = "customer_id"
    merchant_id: str = "merchant_id"

    timestamp: str = "timestamp"
    amount: str = "amount"

    device_id: str = "device_id"
    location: str = "location"
    merchant_category: str = "merchant_category"
    payment_method: str = "payment_method"

    target: str = "is_fraud"

    rolling_windows_hours: Tuple[int, ...] = (
        1,
        6,
        24,
        72,
    )

    amount_zscore_window: int = 20

    drop_original_datetime: bool = False

    keep_identifier_columns: bool = True

    fill_missing_numeric: float = 0.0

    fill_missing_categorical: str = "unknown"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(
    level: int = logging.INFO,
) -> None:
    """Configure logging."""

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_input_dataframe(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> None:
    """
    Validate the raw transaction dataframe.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    required_columns = [
        config.customer_id,
        config.timestamp,
        config.amount,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if not pd.api.types.is_numeric_dtype(
        df[config.amount]
    ):
        raise TypeError(
            f"'{config.amount}' must be numeric."
        )

    if (df[config.amount] < 0).any():
        raise ValueError(
            "Transaction amount cannot be negative."
        )


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Numerically stable division.
    """

    denominator = denominator.replace(
        [np.inf, -np.inf, 0],
        np.nan,
    )

    result = numerator / (
        denominator + EPSILON
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a series to numeric safely.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _ensure_datetime(
    series: pd.Series,
) -> pd.Series:
    """
    Convert timestamps to pandas datetime.
    """

    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


# ---------------------------------------------------------------------------
# Timestamp Features
# ---------------------------------------------------------------------------

def add_time_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add temporal transaction features.
    """

    result = df.copy()

    timestamp = _ensure_datetime(
        result[config.timestamp]
    )

    result["_event_timestamp"] = timestamp

    result["transaction_year"] = (
        timestamp.dt.year
    )

    result["transaction_month"] = (
        timestamp.dt.month
    )

    result["transaction_day"] = (
        timestamp.dt.day
    )

    result["transaction_day_of_week"] = (
        timestamp.dt.dayofweek
    )

    result["transaction_hour"] = (
        timestamp.dt.hour
    )

    result["transaction_minute"] = (
        timestamp.dt.minute
    )

    result["transaction_day_of_year"] = (
        timestamp.dt.dayofyear
    )

    result["is_weekend"] = (
        timestamp.dt.dayofweek >= 5
    ).astype(int)

    result["is_night"] = (
        (timestamp.dt.hour < 6)
        | (timestamp.dt.hour >= 23)
    ).astype(int)

    result["is_business_hour"] = (
        (timestamp.dt.hour >= 9)
        & (timestamp.dt.hour < 18)
        & (timestamp.dt.dayofweek < 5)
    ).astype(int)

    # Cyclical encoding.
    result["hour_sin"] = np.sin(
        2 * np.pi * timestamp.dt.hour / 24
    )

    result["hour_cos"] = np.cos(
        2 * np.pi * timestamp.dt.hour / 24
    )

    result["day_of_week_sin"] = np.sin(
        2
        * np.pi
        * timestamp.dt.dayofweek
        / 7
    )

    result["day_of_week_cos"] = np.cos(
        2
        * np.pi
        * timestamp.dt.dayofweek
        / 7
    )

    if config.drop_original_datetime:
        result = result.drop(
            columns=[config.timestamp],
            errors="ignore",
        )

    return result


# ---------------------------------------------------------------------------
# Amount Features
# ---------------------------------------------------------------------------

def add_amount_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add transaction amount features.
    """

    result = df.copy()

    amount = _safe_numeric(
        result[config.amount]
    ).fillna(0.0)

    result["amount_log"] = np.log1p(
        amount.clip(lower=0)
    )

    result["amount_sqrt"] = np.sqrt(
        amount.clip(lower=0)
    )

    result["amount_is_zero"] = (
        amount <= EPSILON
    ).astype(int)

    result["amount_is_round"] = (
        np.isclose(
            amount % 100,
            0,
            atol=EPSILON,
        )
    ).astype(int)

    result["amount_bucket"] = pd.cut(
        amount,
        bins=[
            -np.inf,
            10,
            50,
            100,
            500,
            1000,
            5000,
            np.inf,
        ],
        labels=False,
    ).fillna(0).astype(int)

    return result


# ---------------------------------------------------------------------------
# Customer Aggregate Features
# ---------------------------------------------------------------------------

def add_customer_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add customer-level historical aggregate features.

    IMPORTANT:
        These features are based on transaction ordering and
        exclude the current transaction where appropriate.
    """

    result = df.copy()

    customer = config.customer_id

    result["_customer_transaction_count"] = (
        result.groupby(customer)
        .cumcount()
    )

    result["customer_transaction_count"] = (
        result["_customer_transaction_count"]
    )

    result["customer_transaction_count_log"] = (
        np.log1p(
            result[
                "customer_transaction_count"
            ]
        )
    )

    # Historical amount statistics.
    historical_amount_sum = (
        result.groupby(customer)[
            config.amount
        ]
        .transform(
            lambda s: s.shift(1).expanding().sum()
        )
    )

    historical_amount_count = (
        result.groupby(customer)[
            config.amount
        ]
        .transform(
            lambda s: s.shift(1).expanding().count()
        )
    )

    historical_amount_mean = _safe_divide(
        historical_amount_sum,
        historical_amount_count,
    )

    result[
        "customer_historical_amount_mean"
    ] = historical_amount_mean

    result[
        "amount_to_customer_mean"
    ] = _safe_divide(
        result[config.amount],
        historical_amount_mean,
    )

    # Historical maximum.
    result[
        "customer_historical_amount_max"
    ] = (
        result.groupby(customer)[
            config.amount
        ]
        .transform(
            lambda s: s.shift(1).expanding().max()
        )
    )

    result[
        "amount_to_customer_max"
    ] = _safe_divide(
        result[config.amount],
        result[
            "customer_historical_amount_max"
        ],
    )

    return result


# ---------------------------------------------------------------------------
# Transaction Velocity
# ---------------------------------------------------------------------------

def add_velocity_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add historical transaction velocity features.

    For every transaction, count previous transactions from
    the same customer within several time windows.
    """

    result = df.copy()

    timestamp_column = "_event_timestamp"

    if timestamp_column not in result.columns:
        result[timestamp_column] = (
            _ensure_datetime(
                result[config.timestamp]
            )
        )

    result = result.sort_values(
        [
            config.customer_id,
            timestamp_column,
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    for hours in config.rolling_windows_hours:

        window = f"{hours}h"

        counts = np.zeros(
            len(result),
            dtype=float,
        )

        amounts = np.zeros(
            len(result),
            dtype=float,
        )

        # Calculate rolling historical features
        # using only transactions before current event.
        for _, group in result.groupby(
            config.customer_id,
            sort=False,
        ):

            indices = group.index

            times = (
                group[timestamp_column]
                .astype("int64")
                .to_numpy()
                / 1e9
            )

            values = (
                group[config.amount]
                .astype(float)
                .to_numpy()
            )

            for position in range(
                len(group)
            ):

                current_time = (
                    times[position]
                )

                start_time = (
                    current_time
                    - hours * 3600
                )

                previous_mask = (
                    (times < current_time)
                    & (times >= start_time)
                )

                counts[
                    indices[position]
                ] = np.sum(
                    previous_mask
                )

                amounts[
                    indices[position]
                ] = np.sum(
                    values[previous_mask]
                )

        result[
            f"customer_txn_count_{hours}h"
        ] = counts

        result[
            f"customer_txn_amount_{hours}h"
        ] = amounts

    return result


# ---------------------------------------------------------------------------
# Merchant Features
# ---------------------------------------------------------------------------

def add_merchant_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add merchant-level historical features.
    """

    result = df.copy()

    if config.merchant_id not in result.columns:
        return result

    merchant = config.merchant_id

    result["merchant_transaction_count"] = (
        result.groupby(merchant)
        .cumcount()
    )

    result["merchant_transaction_count_log"] = (
        np.log1p(
            result[
                "merchant_transaction_count"
            ]
        )
    )

    historical_merchant_mean = (
        result.groupby(merchant)[
            config.amount
        ]
        .transform(
            lambda s: s.shift(1).expanding().mean()
        )
    )

    result[
        "merchant_historical_amount_mean"
    ] = historical_merchant_mean

    result[
        "amount_to_merchant_mean"
    ] = _safe_divide(
        result[config.amount],
        historical_merchant_mean,
    )

    return result


# ---------------------------------------------------------------------------
# Device Features
# ---------------------------------------------------------------------------

def add_device_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add device-level behavioral features.
    """

    result = df.copy()

    if config.device_id not in result.columns:
        return result

    device = config.device_id

    result["device_transaction_count"] = (
        result.groupby(device)
        .cumcount()
    )

    result["device_transaction_count_log"] = (
        np.log1p(
            result[
                "device_transaction_count"
            ]
        )
    )

    if config.customer_id in result.columns:

        result[
            "device_customer_count"
        ] = (
            result.groupby(device)[
                config.customer_id
            ]
            .transform(
                lambda s: (
                    s.expanding()
                    .apply(
                        lambda x: x.nunique(),
                        raw=False,
                    )
                )
                - 1
            )
        )

        result[
            "device_customer_count"
        ] = result[
            "device_customer_count"
        ].clip(
            lower=0
        )

    return result


# ---------------------------------------------------------------------------
# Location Features
# ---------------------------------------------------------------------------

def add_location_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add location-related features.
    """

    result = df.copy()

    if config.location not in result.columns:
        return result

    location = (
        result[config.location]
        .fillna(
            config.fill_missing_categorical
        )
        .astype(str)
    )

    result["location_missing"] = (
        location
        == config.fill_missing_categorical
    ).astype(int)

    result["location_frequency"] = (
        location.map(
            location.value_counts(
                normalize=True
            )
        )
    )

    if config.customer_id in result.columns:

        customer_location_count = (
            result.groupby(
                config.customer_id
            )[config.location]
            .transform(
                lambda s: s.fillna(
                    config.fill_missing_categorical
                ).ne(
                    s.fillna(
                        config.fill_missing_categorical
                    ).shift()
                ).cumsum()
            )
        )

        result[
            "customer_location_change_indicator"
        ] = (
            customer_location_count > 1
        ).astype(int)

    return result


# ---------------------------------------------------------------------------
# Payment Method Features
# ---------------------------------------------------------------------------

def add_payment_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add payment-method features.
    """

    result = df.copy()

    if config.payment_method not in result.columns:
        return result

    payment = (
        result[config.payment_method]
        .fillna(
            config.fill_missing_categorical
        )
        .astype(str)
    )

    result["payment_method_frequency"] = (
        payment.map(
            payment.value_counts(
                normalize=True
            )
        )
    )

    if config.customer_id in result.columns:

        customer_payment_count = (
            result.groupby(
                config.customer_id
            )[config.payment_method]
            .transform(
                lambda s: s.fillna(
                    config.fill_missing_categorical
                ).ne(
                    s.fillna(
                        config.fill_missing_categorical
                    ).shift()
                ).cumsum()
            )
        )

        result[
            "customer_payment_method_change"
        ] = (
            customer_payment_count > 1
        ).astype(int)

    return result


# ---------------------------------------------------------------------------
# Behavioral Features
# ---------------------------------------------------------------------------

def add_behavioral_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add customer behavioral anomaly features.
    """

    result = df.copy()

    customer = config.customer_id
    amount = _safe_numeric(
        result[config.amount]
    )

    # Historical standard deviation.
    historical_std = (
        result.groupby(customer)[
            config.amount
        ]
        .transform(
            lambda s: s.shift(1)
            .expanding()
            .std()
        )
    )

    historical_mean = (
        result.groupby(customer)[
            config.amount
        ]
        .transform(
            lambda s: s.shift(1)
            .expanding()
            .mean()
        )
    )

    result[
        "customer_historical_amount_std"
    ] = historical_std

    result[
        "customer_amount_zscore"
    ] = _safe_divide(
        amount - historical_mean,
        historical_std,
    )

    result[
        "customer_amount_deviation"
    ] = (
        amount
        - historical_mean
    ).abs()

    # Amount substantially above historical behavior.
    result[
        "unusual_amount_indicator"
    ] = (
        result[
            "customer_amount_zscore"
        ].abs()
        >= 3.0
    ).astype(int)

    return result


# ---------------------------------------------------------------------------
# Risk Indicators
# ---------------------------------------------------------------------------

def add_risk_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Add composite risk indicators.

    These are rule-based ML features, not final fraud decisions.
    """

    result = df.copy()

    risk_columns: List[str] = []

    candidate_columns = [
        "is_night",
        "is_weekend",
        "unusual_amount_indicator",
        "amount_is_round",
        "location_missing",
        "customer_payment_method_change",
        "customer_location_change_indicator",
    ]

    for column in candidate_columns:

        if column in result.columns:

            values = pd.to_numeric(
                result[column],
                errors="coerce",
            ).fillna(0)

            risk_columns.append(
                column
            )

    if risk_columns:

        result["rule_based_risk_score"] = (
            result[risk_columns]
            .sum(axis=1)
        )

    else:

        result["rule_based_risk_score"] = 0.0

    # High-risk indicator.
    result["high_risk_indicator"] = (
        result[
            "rule_based_risk_score"
        ] >= 3
    ).astype(int)

    return result


# ---------------------------------------------------------------------------
# Categorical Encoding
# ---------------------------------------------------------------------------

def encode_categorical_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    categorical_columns: Optional[
        Sequence[str]
    ] = None,
) -> pd.DataFrame:
    """
    Convert categorical values into stable integer category codes.

    For a production pipeline, fitted encoders should ideally be
    persisted separately. This function is intended for preprocessing
    a complete prepared dataset.
    """

    result = df.copy()

    if categorical_columns is None:

        categorical_columns = [
            column
            for column in [
                config.merchant_category,
                config.payment_method,
                config.location,
                config.device_id,
            ]
            if column in result.columns
        ]

    for column in categorical_columns:

        if column not in result.columns:
            continue

        values = (
            result[column]
            .fillna(
                config.fill_missing_categorical
            )
            .astype(str)
        )

        result[
            f"{column}_encoded"
        ] = pd.factorize(
            values,
            sort=True,
        )[0].astype(float)

    return result


# ---------------------------------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Handle missing and infinite feature values.
    """

    result = df.copy()

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    numeric_columns = (
        result.select_dtypes(
            include=[np.number]
        ).columns
    )

    categorical_columns = (
        result.select_dtypes(
            include=[
                "object",
                "category",
                "string",
            ]
        ).columns
    )

    for column in numeric_columns:

        result[column] = (
            result[column]
            .fillna(
                config.fill_missing_numeric
            )
        )

    for column in categorical_columns:

        result[column] = (
            result[column]
            .fillna(
                config.fill_missing_categorical
            )
        )

    return result


# ---------------------------------------------------------------------------
# Remove Temporary Columns
# ---------------------------------------------------------------------------

def cleanup_features(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """
    Remove internal processing columns.
    """

    result = df.copy()

    result = result.drop(
        columns=[
            "_customer_transaction_count",
        ],
        errors="ignore",
    )

    if (
        config.drop_original_datetime
        and "_event_timestamp"
        in result.columns
    ):

        result = result.drop(
            columns=["_event_timestamp"],
            errors="ignore",
        )

    return result


# ---------------------------------------------------------------------------
# Complete Feature Engineering
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    encode_categoricals: bool = True,
) -> pd.DataFrame:
    """
    Execute the complete feature-engineering pipeline.

    Returns:
        Feature-engineered dataframe.
    """

    if config is None:
        config = FeatureConfig()

    validate_input_dataframe(
        df,
        config,
    )

    logger.info(
        "Starting FinCo AI feature engineering."
    )

    result = df.copy()

    # Sort chronologically to ensure historical features
    # never use future transactions.
    timestamp = _ensure_datetime(
        result[config.timestamp]
    )

    result[config.timestamp] = timestamp

    result = result.sort_values(
        [
            config.customer_id,
            config.timestamp,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    # Feature pipeline.
    result = add_time_features(
        result,
        config,
    )

    result = add_amount_features(
        result,
        config,
    )

    result = add_customer_features(
        result,
        config,
    )

    result = add_velocity_features(
        result,
        config,
    )

    result = add_merchant_features(
        result,
        config,
    )

    result = add_device_features(
        result,
        config,
    )

    result = add_location_features(
        result,
        config,
    )

    result = add_payment_features(
        result,
        config,
    )

    result = add_behavioral_features(
        result,
        config,
    )

    result = add_risk_features(
        result,
        config,
    )

    if encode_categoricals:

        result = encode_categorical_features(
            result,
            config,
        )

    result = handle_missing_values(
        result,
        config,
    )

    result = cleanup_features(
        result,
        config,
    )

    logger.info(
        "Feature engineering complete: "
        "rows=%d, columns=%d",
        result.shape[0],
        result.shape[1],
    )

    return result


# ---------------------------------------------------------------------------
# ML Matrix Builder
# ---------------------------------------------------------------------------

def build_ml_matrix(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    drop_identifiers: bool = True,
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Convert engineered dataframe into X/y ML matrices.

    Returns:
        X, y

    Target:
        config.target if present.
    """

    if config is None:
        config = FeatureConfig()

    result = df.copy()

    y: Optional[pd.Series] = None

    if config.target in result.columns:

        y = pd.to_numeric(
            result[config.target],
            errors="coerce",
        )

        if y.isna().any():
            raise ValueError(
                "Target contains invalid values."
            )

        y = y.astype(int)

        invalid_labels = ~y.isin(
            [0, 1]
        )

        if invalid_labels.any():
            raise ValueError(
                "Fraud target must contain "
                "only 0 and 1."
            )

        result = result.drop(
            columns=[config.target]
        )

    if drop_identifiers:

        identifier_columns = [
            config.transaction_id,
            config.customer_id,
            config.merchant_id,
            config.device_id,
        ]

        result = result.drop(
            columns=[
                column
                for column in identifier_columns
                if column in result.columns
            ],
            errors="ignore",
        )

    # Datetime values cannot directly enter most ML models.
    datetime_columns = (
        result.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns
    )

    result = result.drop(
        columns=list(datetime_columns),
        errors="ignore",
    )

    # Convert remaining categoricals to numeric codes.
    categorical_columns = (
        result.select_dtypes(
            include=[
                "object",
                "category",
                "string",
            ]
        ).columns
    )

    for column in categorical_columns:

        result[column] = pd.factorize(
            result[column].astype(str),
            sort=True,
        )[0].astype(float)

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.fillna(0.0)

    X = result.astype(float)

    return X, y


# ---------------------------------------------------------------------------
# Feature Metadata
# ---------------------------------------------------------------------------

def generate_feature_metadata(
    df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
) -> Dict[str, Any]:
    """
    Generate metadata describing engineered features.
    """

    if config is None:
        config = FeatureConfig()

    numeric_columns = list(
        df.select_dtypes(
            include=[np.number]
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            include=[
                "object",
                "category",
                "string",
            ]
        ).columns
    )

    return {
        "project": "FinCo AI",
        "pipeline": "fraud_feature_engineering",
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_features": numeric_columns,
        "categorical_features": (
            categorical_columns
        ),
        "feature_count": int(
            len(numeric_columns)
            + len(categorical_columns)
        ),
        "target": config.target,
    }


# ---------------------------------------------------------------------------
# Save Data
# ---------------------------------------------------------------------------

def save_features(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save engineered features to CSV.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    logger.info(
        "Saved engineered features to %s",
        path,
    )

    return path


def save_metadata(
    metadata: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save feature metadata as JSON.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    return path


# ---------------------------------------------------------------------------
# File Pipeline
# ---------------------------------------------------------------------------

def run_feature_engineering(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
    metadata_path: Optional[
        str | Path
    ] = None,
    config: Optional[FeatureConfig] = None,
) -> Dict[str, Any]:
    """
    Run feature engineering from CSV to CSV.
    """

    if config is None:
        config = FeatureConfig()

    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )

    logger.info(
        "Loading raw transaction data: %s",
        input_file,
    )

    df = pd.read_csv(
        input_file
    )

    features = build_features(
        df,
        config=config,
    )

    output_file = save_features(
        features,
        output_path,
    )

    metadata = generate_feature_metadata(
        features,
        config=config,
    )

    metadata[
        "input_path"
    ] = str(input_file)

    metadata[
        "output_path"
    ] = str(output_file)

    if metadata_path is None:

        metadata_path = (
            Path(output_path).parent
            / "feature_metadata.json"
        )

    save_metadata(
        metadata,
        metadata_path,
    )

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "FinCo AI fraud feature engineering."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="Input transactions CSV.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output engineered features CSV.",
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Feature metadata JSON path.",
    )

    parser.add_argument(
        "--no-encoding",
        action="store_true",
        help="Disable categorical encoding.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""

    configure_logging()

    parser = build_parser()
    args = parser.parse_args()

    config = FeatureConfig()

    metadata = run_feature_engineering(
        input_path=args.input,
        output_path=args.output,
        metadata_path=args.metadata,
        config=config,
    )

    print()
    print(
        "FinCo AI Feature Engineering Complete"
    )
    print("=" * 48)
    print(
        f"Rows     : {metadata['rows']}"
    )
    print(
        f"Columns  : {metadata['columns']}"
    )
    print(
        f"Features : {metadata['feature_count']}"
    )
    print(
        f"Output   : {metadata['output_path']}"
    )


if __name__ == "__main__":
    main()