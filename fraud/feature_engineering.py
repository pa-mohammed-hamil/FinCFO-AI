"""
FinCo AI - Fraud Feature Engineering

File:
    backend/app/fraud/feature_engineering.py

Purpose:
    Transform raw financial transaction data into numerical features
    suitable for fraud detection and anomaly detection.

Pipeline:

    Raw Transaction
          |
          v
    preprocessing.py
          |
          v
    feature_engineering.py
          |
          +-------------------+
          |                   |
          v                   v
    anomaly_model.py   supervised_model.py
          |                   |
          +---------+---------+
                    |
                    v
                scoring.py
                    |
                    v
              fraud_service.py

Feature categories:
    - Transaction amount
    - Account balance
    - Transaction frequency
    - Transaction velocity
    - Amount/balance ratio
    - Account age
    - Time features
    - Merchant features
    - Location/device signals
    - Historical customer behavior
    - Rolling transaction statistics

Design principles:
    - Deterministic
    - No ML model dependency
    - No LLM dependency
    - Safe numerical handling
    - Reusable for batch and online inference
    - Explainable feature names
    - Production-friendly
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import statistics


# ============================================================================
# Exceptions
# ============================================================================


class FeatureEngineeringError(Exception):
    """Base exception for feature engineering failures."""


class InvalidTransactionError(
    FeatureEngineeringError
):
    """Raised when transaction data is invalid."""


class MissingFeatureError(
    FeatureEngineeringError
):
    """Raised when required feature data is missing."""


# ============================================================================
# Feature Categories
# ============================================================================


class FeatureCategory:
    """Feature category constants."""

    TRANSACTION = "transaction"
    ACCOUNT = "account"
    VELOCITY = "velocity"
    BEHAVIOR = "behavior"
    TIME = "time"
    DEVICE = "device"
    LOCATION = "location"
    MERCHANT = "merchant"
    RATIO = "ratio"
    ROLLING = "rolling"


# ============================================================================
# Feature Definition
# ============================================================================


@dataclass(frozen=True)
class FeatureDefinition:
    """
    Metadata describing a generated feature.
    """

    name: str
    category: str
    description: str
    default: float = 0.0


@dataclass
class FeatureVector:
    """
    Engineered numerical representation of a transaction.
    """

    transaction_id: Optional[str]

    values: Dict[str, float]

    categorical: Dict[str, str] = field(
        default_factory=dict
    )

    flags: Dict[str, bool] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id":
                self.transaction_id,
            "values":
                dict(self.values),
            "categorical":
                dict(self.categorical),
            "flags":
                dict(self.flags),
            "metadata":
                dict(self.metadata),
        }

    def vector(
        self,
        feature_names: Sequence[str],
    ) -> List[float]:
        """
        Return values in a deterministic feature order.
        """

        return [
            float(
                self.values.get(
                    name,
                    0.0,
                )
            )
            for name in feature_names
        ]


# ============================================================================
# Default Feature Definitions
# ============================================================================


DEFAULT_FEATURE_DEFINITIONS: Tuple[
    FeatureDefinition, ...
] = (
    FeatureDefinition(
        name="amount",
        category=FeatureCategory.TRANSACTION,
        description="Transaction amount.",
    ),

    FeatureDefinition(
        name="transaction_frequency",
        category=FeatureCategory.VELOCITY,
        description=(
            "Number of recent transactions "
            "for the account."
        ),
    ),

    FeatureDefinition(
        name="velocity",
        category=FeatureCategory.VELOCITY,
        description=(
            "Transaction activity intensity "
            "over a recent time window."
        ),
    ),

    FeatureDefinition(
        name="account_age_days",
        category=FeatureCategory.ACCOUNT,
        description=(
            "Age of the account in days."
        ),
    ),

    FeatureDefinition(
        name="balance",
        category=FeatureCategory.ACCOUNT,
        description=(
            "Available account balance."
        ),
    ),

    FeatureDefinition(
        name="amount_to_balance_ratio",
        category=FeatureCategory.RATIO,
        description=(
            "Transaction amount relative "
            "to available balance."
        ),
    ),

    FeatureDefinition(
        name="amount_to_avg_ratio",
        category=FeatureCategory.BEHAVIOR,
        description=(
            "Transaction amount relative "
            "to historical average."
        ),
    ),

    FeatureDefinition(
        name="amount_zscore",
        category=FeatureCategory.BEHAVIOR,
        description=(
            "Amount deviation from historical "
            "customer behavior."
        ),
    ),

    FeatureDefinition(
        name="frequency_zscore",
        category=FeatureCategory.BEHAVIOR,
        description=(
            "Frequency deviation from historical "
            "customer behavior."
        ),
    ),

    FeatureDefinition(
        name="velocity_zscore",
        category=FeatureCategory.BEHAVIOR,
        description=(
            "Velocity deviation from historical "
            "customer behavior."
        ),
    ),

    FeatureDefinition(
        name="hour",
        category=FeatureCategory.TIME,
        description=(
            "Hour of transaction."
        ),
    ),

    FeatureDefinition(
        name="day_of_week",
        category=FeatureCategory.TIME,
        description=(
            "Day of week of transaction."
        ),
    ),

    FeatureDefinition(
        name="is_weekend",
        category=FeatureCategory.TIME,
        description=(
            "Whether transaction occurred "
            "during the weekend."
        ),
    ),

    FeatureDefinition(
        name="is_night",
        category=FeatureCategory.TIME,
        description=(
            "Whether transaction occurred "
            "during night hours."
        ),
    ),

    FeatureDefinition(
        name="is_international",
        category=FeatureCategory.LOCATION,
        description=(
            "Whether transaction is international."
        ),
    ),

    FeatureDefinition(
        name="new_location",
        category=FeatureCategory.LOCATION,
        description=(
            "Whether transaction originated "
            "from a new location."
        ),
    ),

    FeatureDefinition(
        name="new_device",
        category=FeatureCategory.DEVICE,
        description=(
            "Whether transaction originated "
            "from a new device."
        ),
    ),

    FeatureDefinition(
        name="merchant_risk_score",
        category=FeatureCategory.MERCHANT,
        description=(
            "Risk score associated with merchant."
        ),
    ),

    FeatureDefinition(
        name="merchant_frequency",
        category=FeatureCategory.MERCHANT,
        description=(
            "Historical transaction frequency "
            "for the merchant."
        ),
    ),

    FeatureDefinition(
        name="recent_amount_sum",
        category=FeatureCategory.ROLLING,
        description=(
            "Total transaction amount "
            "over the recent window."
        ),
)

    FeatureDefinition(
        name="recent_amount_mean",
        category=FeatureCategory.ROLLING,
        description=(
            "Average transaction amount "
            "over the recent window."
        ),
    ),

    FeatureDefinition(
        name="recent_amount_max",
        category=FeatureCategory.ROLLING,
        description=(
            "Maximum transaction amount "
            "over the recent window."
        ),
    ),

    FeatureDefinition(
        name="recent_transaction_count",
        category=FeatureCategory.ROLLING,
        description=(
            "Number of transactions "
            "in the recent window."
        ),
    ),
)


# ============================================================================
# Main Feature Engineering Class
# ============================================================================


class FraudFeatureEngineer:
    """
    Generates fraud detection features.

    The class supports:

        1. Single transaction feature generation
        2. Batch feature generation
        3. Historical behavioral features
        4. Time-based features
        5. Device/location signals
        6. Merchant features
        7. Rolling transaction statistics
    """

    def __init__(
        self,
        definitions: Optional[
            Sequence[FeatureDefinition]
        ] = None,
        recent_window_size: int = 10,
        night_start_hour: int = 0,
        night_end_hour: int = 6,
    ) -> None:

        self.definitions = tuple(
            definitions
            or DEFAULT_FEATURE_DEFINITIONS
        )

        self.recent_window_size = int(
            recent_window_size
        )

        self.night_start_hour = int(
            night_start_hour
        )

        self.night_end_hour = int(
            night_end_hour
        )

        self._validate_config()

    # ========================================================================
    # MAIN API
    # ========================================================================

    def transform(
        self,
        transaction: Mapping[str, Any],
        history: Optional[
            Sequence[
                Mapping[str, Any]
            ]
        ] = None,
    ) -> FeatureVector:
        """
        Generate features for one transaction.
        """

        if not transaction:
            raise InvalidTransactionError(
                "Transaction cannot be empty."
            )

        history = list(
            history or []
        )

        transaction_id = (
            self._string_or_none(
                transaction.get(
                    "transaction_id"
                )
            )
        )

        values: Dict[
            str,
            float
        ] = {}

        categorical: Dict[
            str,
            str
        ] = {}

        flags: Dict[
            str,
            bool
        ] = {}

        # --------------------------------------------------------------
        # Transaction features
        # --------------------------------------------------------------

        amount = self._number(
            transaction.get(
                "amount"
            )
        )

        balance = self._number(
            transaction.get(
                "balance"
            )
        )

        values[
            "amount"
        ] = amount

        values[
            "balance"
        ] = balance

        # --------------------------------------------------------------
        # Account features
        # --------------------------------------------------------------

        account_age_days = (
            self._account_age_days(
                transaction
            )
        )

        values[
            "account_age_days"
        ] = account_age_days

        # --------------------------------------------------------------
        # Recent history
        # --------------------------------------------------------------

        recent_history = (
            self._recent_history(
                history
            )
        )

        recent_amounts = [
            self._number(
                item.get(
                    "amount"
                )
            )
            for item in recent_history
        ]

        transaction_frequency = (
            self._transaction_frequency(
                transaction,
                recent_history,
            )
        )

        velocity = (
            self._velocity(
                transaction,
                recent_history,
            )
        )

        values[
            "transaction_frequency"
        ] = transaction_frequency

        values[
            "velocity"
        ] = velocity

        # --------------------------------------------------------------
        # Ratios
        # --------------------------------------------------------------

        values[
            "amount_to_balance_ratio"
        ] = self._safe_ratio(
            amount,
            balance,
        )

        historical_average = (
            self._historical_average(
                recent_amounts
            )
        )

        values[
            "amount_to_avg_ratio"
        ] = self._safe_ratio(
            amount,
            historical_average,
        )

        # --------------------------------------------------------------
        # Statistical behavior
        # --------------------------------------------------------------

        values[
            "amount_zscore"
        ] = self._zscore(
            amount,
            recent_amounts,
        )

        frequencies = self._historical_frequencies(
            history
        )

        values[
            "frequency_zscore"
        ] = self._zscore(
            transaction_frequency,
            frequencies,
        )

        historical_velocities = (
            self._historical_velocities(
                history
            )
        )

        values[
            "velocity_zscore"
        ] = self._zscore(
            velocity,
            historical_velocities,
        )

        # --------------------------------------------------------------
        # Time features
        # --------------------------------------------------------------

        timestamp = (
            self._transaction_timestamp(
                transaction
            )
        )

        if timestamp:

            values[
                "hour"
            ] = float(
                timestamp.hour
            )

            values[
                "day_of_week"
            ] = float(
                timestamp.weekday()
            )

            flags[
                "is_weekend"
            ] = (
                timestamp.weekday()
                >= 5
            )

            flags[
                "is_night"
            ] = self._is_night(
                timestamp.hour
            )

            values[
                "is_weekend"
            ] = float(
                flags[
                    "is_weekend"
                ]
            )

            values[
                "is_night"
            ] = float(
                flags[
                    "is_night"
                ]
            )

        else:

            values[
                "hour"
            ] = 0.0

            values[
                "day_of_week"
            ] = 0.0

            values[
                "is_weekend"
            ] = 0.0

            values[
                "is_night"
            ] = 0.0

        # --------------------------------------------------------------
        # Location features
        # --------------------------------------------------------------

        is_international = (
            self._boolean(
                transaction.get(
                    "is_international"
                )
            )
        )

        new_location = (
            self._boolean(
                transaction.get(
                    "new_location"
                )
            )
        )

        flags[
            "is_international"
        ] = is_international

        flags[
            "new_location"
        ] = new_location

        values[
            "is_international"
        ] = float(
            is_international
        )

        values[
            "new_location"
        ] = float(
            new_location
        )

        # --------------------------------------------------------------
        # Device features
        # --------------------------------------------------------------

        new_device = (
            self._boolean(
                transaction.get(
                    "new_device"
                )
            )
        )

        flags[
            "new_device"
        ] = new_device

        values[
            "new_device"
        ] = float(
            new_device
        )

        # --------------------------------------------------------------
        # Merchant features
        # --------------------------------------------------------------

        merchant_risk = (
            self._number(
                transaction.get(
                    "merchant_risk_score"
                )
            )
        )

        merchant_frequency = (
            self._number(
                transaction.get(
                    "merchant_frequency"
                )
            )
        )

        values[
            "merchant_risk_score"
        ] = self._clamp(
            merchant_risk,
            0.0,
            100.0,
        )

        values[
            "merchant_frequency"
        ] = max(
            0.0,
            merchant_frequency,
        )

        # --------------------------------------------------------------
        # Rolling features
        # --------------------------------------------------------------

        values[
            "recent_amount_sum"
        ] = sum(
            recent_amounts
        )

        values[
            "recent_amount_mean"
        ] = (
            self._mean(
                recent_amounts
            )
        )

        values[
            "recent_amount_max"
        ] = (
            max(
                recent_amounts
            )
            if recent_amounts
            else 0.0
        )

        values[
            "recent_transaction_count"
        ] = float(
            len(
                recent_history
            )
        )

        # --------------------------------------------------------------
        # Categorical values
        # --------------------------------------------------------------

        for key in (
            "currency",
            "transaction_type",
            "merchant_category",
            "country",
            "device_type",
            "channel",
        ):

            value = transaction.get(
                key
            )

            if value is not None:

                categorical[
                    key
                ] = str(
                    value
                )

        # --------------------------------------------------------------
        # Sanitize
        # --------------------------------------------------------------

        values = self._sanitize_features(
            values
        )

        return FeatureVector(
            transaction_id=transaction_id,
            values=values,
            categorical=categorical,
            flags=flags,
            metadata={
                "feature_count":
                    len(values),
                "history_count":
                    len(history),
            },
        )

    # ========================================================================
    # BATCH TRANSFORM
    # ========================================================================

    def transform_batch(
        self,
        transactions: Sequence[
            Mapping[str, Any]
        ],
        histories: Optional[
            Sequence[
                Sequence[
                    Mapping[str, Any]
                ]
            ]
        ] = None,
    ) -> List[FeatureVector]:
        """
        Generate features for multiple transactions.
        """

        if not transactions:
            return []

        if histories is not None:

            if len(histories) != len(
                transactions
            ):
                raise InvalidTransactionError(
                    "histories length must "
                    "match transactions length."
                )

        results: List[
            FeatureVector
        ] = []

        for index, transaction in enumerate(
            transactions
        ):

            history = (
                histories[index]
                if histories is not None
                else []
            )

            results.append(
                self.transform(
                    transaction,
                    history,
                )
            )

        return results

    # ========================================================================
    # FEATURE MATRIX
    # ========================================================================

    def to_matrix(
        self,
        feature_vectors: Sequence[
            FeatureVector
        ],
        feature_names: Optional[
            Sequence[str]
        ] = None,
    ) -> Tuple[
        List[List[float]],
        List[str],
    ]:
        """
        Convert FeatureVector objects into an ML matrix.
        """

        if feature_names is None:

            feature_names = (
                self.numeric_feature_names()
            )

        names = list(
            feature_names
        )

        matrix = [
            vector.vector(
                names
            )
            for vector
            in feature_vectors
        ]

        return matrix, names

    # ========================================================================
    # FEATURE NAMES
    # ========================================================================

    def numeric_feature_names(
        self,
    ) -> List[str]:
        """
        Return deterministic numerical feature ordering.
        """

        return [
            definition.name
            for definition
            in self.definitions
        ]

    def feature_definitions(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Return feature metadata.
        """

        return [
            asdict(
                definition
            )
            for definition
            in self.definitions
        ]

    # ========================================================================
    # TRANSACTION FEATURES
    # ========================================================================

    def _transaction_frequency(
        self,
        transaction: Mapping[str, Any],
        history: Sequence[
            Mapping[str, Any]
        ],
    ) -> float:
        """
        Determine transaction frequency.

        If explicitly provided by upstream preprocessing,
        use that value. Otherwise count recent history.
        """

        explicit = transaction.get(
            "transaction_frequency"
        )

        if explicit is not None:

            return max(
                0.0,
                self._number(
                    explicit
                ),
            )

        return float(
            len(
                history
            )
        )

    # ========================================================================
    # VELOCITY
    # ========================================================================

    def _velocity(
        self,
        transaction: Mapping[str, Any],
        history: Sequence[
            Mapping[str, Any]
        ],
    ) -> float:
        """
        Calculate transaction velocity.

        Preference order:

            1. Explicit upstream velocity
            2. Transactions per hour
            3. Recent transaction count
        """

        explicit = transaction.get(
            "velocity"
        )

        if explicit is not None:

            return max(
                0.0,
                self._number(
                    explicit
                ),
            )

        current_time = (
            self._transaction_timestamp(
                transaction
            )
        )

        if current_time is None:
            return float(
                len(history)
            )

        recent_count = 0

        for item in history:

            timestamp = (
                self._transaction_timestamp(
                    item
                )
            )

            if timestamp is None:
                continue

            seconds = abs(
                (
                    current_time
                    - timestamp
                ).total_seconds()
            )

            if seconds <= 3600:
                recent_count += 1

        return float(
            recent_count
        )

    # ========================================================================
    # ACCOUNT AGE
    # ========================================================================

    def _account_age_days(
        self,
        transaction: Mapping[str, Any],
    ) -> float:

        explicit = transaction.get(
            "account_age_days"
        )

        if explicit is not None:

            return max(
                0.0,
                self._number(
                    explicit
                ),
            )

        created_at = (
            transaction.get(
                "account_created_at"
            )
        )

        timestamp = (
            self._transaction_timestamp(
                transaction
            )
        )

        if (
            created_at is None
            or timestamp is None
        ):
            return 0.0

        created = (
            self._parse_datetime(
                created_at
            )
        )

        if created is None:
            return 0.0

        return max(
            0.0,
            (
                timestamp
                - created
            ).total_seconds()
            / 86400.0,
        )

    # ========================================================================
    # HISTORICAL FEATURES
    # ========================================================================

    def _historical_average(
        self,
        amounts: Sequence[float],
    ) -> float:

        if not amounts:
            return 0.0

        return self._mean(
            amounts
        )

    def _historical_frequencies(
        self,
        history: Sequence[
            Mapping[str, Any]
        ],
    ) -> List[float]:

        frequencies: List[
            float
        ] = []

        for item in history:

            value = item.get(
                "transaction_frequency"
            )

            if value is not None:

                frequencies.append(
                    max(
                        0.0,
                        self._number(
                            value
                        ),
                    )
                )

        return frequencies

    def _historical_velocities(
        self,
        history: Sequence[
            Mapping[str, Any]
        ],
    ) -> List[float]:

        velocities: List[
            float
        ] = []

        for item in history:

            value = item.get(
                "velocity"
            )

            if value is not None:

                velocities.append(
                    max(
                        0.0,
                        self._number(
                            value
                        ),
                    )
                )

        return velocities

    # ========================================================================
    # RECENT HISTORY
    # ========================================================================

    def _recent_history(
        self,
        history: Sequence[
            Mapping[str, Any]
        ],
    ) -> List[
        Mapping[str, Any]
    ]:

        if not history:
            return []

        return list(
            history[
                -self.recent_window_size:
            ]
        )

    # ========================================================================
    # TIME FEATURES
    # ========================================================================

    def _transaction_timestamp(
        self,
        transaction: Mapping[str, Any],
    ) -> Optional[
        datetime
    ]:

        for key in (
            "timestamp",
            "transaction_timestamp",
            "created_at",
            "transaction_date",
        ):

            value = transaction.get(
                key
            )

            if value is None:
                continue

            parsed = (
                self._parse_datetime(
                    value
                )
            )

            if parsed:
                return parsed

        return None

    @staticmethod
    def _parse_datetime(
        value: Any,
    ) -> Optional[
        datetime
    ]:

        if isinstance(
            value,
            datetime,
        ):
            result = value

        elif isinstance(
            value,
            str,
        ):

            text = value.strip()

            if not text:
                return None

            try:

                result = (
                    datetime.fromisoformat(
                        text.replace(
                            "Z",
                            "+00:00",
                        )
                    )
                )

            except ValueError:
                return None

        else:
            return None

        if result.tzinfo is None:

            result = result.replace(
                tzinfo=timezone.utc
            )

        return result

    def _is_night(
        self,
        hour: int,
    ) -> bool:

        if (
            self.night_start_hour
            <= self.night_end_hour
        ):

            return (
                self.night_start_hour
                <= hour
                < self.night_end_hour
            )

        return (
            hour
            >= self.night_start_hour
            or hour
            < self.night_end_hour
        )

    # ========================================================================
    # STATISTICAL FEATURES
    # ========================================================================

    @staticmethod
    def _zscore(
        value: float,
        values: Sequence[float],
    ) -> float:

        if len(values) < 2:
            return 0.0

        mean = (
            FraudFeatureEngineer._mean(
                values
            )
        )

        try:

            std = float(
                statistics.stdev(
                    values
                )
            )

        except statistics.StatisticsError:
            return 0.0

        if std <= 1e-12:
            return 0.0

        return (
            value - mean
        ) / std

    # ========================================================================
    # SANITIZATION
    # ========================================================================

    @staticmethod
    def _sanitize_features(
        values: Dict[
            str,
            float
        ],
    ) -> Dict[
        str,
        float
    ]:

        sanitized: Dict[
            str,
            float
        ] = {}

        for key, value in values.items():

            try:
                number = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):
                number = 0.0

            if not math.isfinite(
                number
            ):
                number = 0.0

            sanitized[
                key
            ] = number

        return sanitized

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _safe_ratio(
        numerator: float,
        denominator: float,
    ) -> float:

        if abs(
            denominator
        ) <= 1e-12:
            return 0.0

        return (
            numerator
            / denominator
        )

    @staticmethod
    def _mean(
        values: Sequence[float],
    ) -> float:

        if not values:
            return 0.0

        return float(
            statistics.fmean(
                values
            )
        )

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            number = float(
                value
            )

            if not math.isfinite(
                number
            ):
                return default

            return number

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _boolean(
        value: Any,
    ) -> bool:

        if isinstance(
            value,
            bool,
        ):
            return value

        if isinstance(
            value,
            str,
        ):

            return value.strip().lower() in {
                "true",
                "1",
                "yes",
                "y",
                "on",
            }

        if isinstance(
            value,
            (int, float),
        ):
            return bool(
                value
            )

        return False

    @staticmethod
    def _string_or_none(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        text = str(
            value
        ).strip()

        return (
            text
            if text
            else None
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_config(
        self,
    ) -> None:

        if (
            self.recent_window_size
            <= 0
        ):
            raise FeatureEngineeringError(
                "recent_window_size must "
                "be greater than zero."
            )

        if not (
            0 <= self.night_start_hour <= 23
            and 0 <= self.night_end_hour <= 23
        ):
            raise FeatureEngineeringError(
                "Night hours must be between "
                "0 and 23."
            )


# ============================================================================
# Feature Registry
# ============================================================================


class FeatureRegistry:
    """
    Central registry for fraud feature definitions.

    Useful when the same feature ordering must be shared between:

        feature_engineering.py
        anomaly_model.py
        supervised_model.py
        model training jobs
        model inference
    """

    def __init__(
        self,
        definitions: Optional[
            Sequence[FeatureDefinition]
        ] = None,
    ) -> None:

        self._definitions = tuple(
            definitions
            or DEFAULT_FEATURE_DEFINITIONS
        )

        self._by_name = {
            item.name: item
            for item
            in self._definitions
        }

    def get(
        self,
        name: str,
    ) -> FeatureDefinition:

        try:
            return self._by_name[
                name
            ]

        except KeyError as exc:

            raise MissingFeatureError(
                f"Unknown feature: {name}"
            ) from exc

    def names(self) -> List[str]:

        return [
            item.name
            for item
            in self._definitions
        ]

    def definitions(
        self,
    ) -> List[FeatureDefinition]:

        return list(
            self._definitions
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def engineer_features(
    transaction: Mapping[str, Any],
    history: Optional[
        Sequence[
            Mapping[str, Any]
        ]
    ] = None,
) -> FeatureVector:
    """
    Generate features for a single transaction.
    """

    engineer = (
        FraudFeatureEngineer()
    )

    return engineer.transform(
        transaction,
        history,
    )


def engineer_batch(
    transactions: Sequence[
        Mapping[str, Any]
    ],
    histories: Optional[
        Sequence[
            Sequence[
                Mapping[str, Any]
            ]
        ]
    ] = None,
) -> List[FeatureVector]:
    """
    Generate features for a transaction batch.
    """

    engineer = (
        FraudFeatureEngineer()
    )

    return engineer.transform_batch(
        transactions,
        histories,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "FeatureEngineeringError",
    "InvalidTransactionError",
    "MissingFeatureError",
    "FeatureCategory",
    "FeatureDefinition",
    "FeatureVector",
    "DEFAULT_FEATURE_DEFINITIONS",
    "FraudFeatureEngineer",
    "FeatureRegistry",
    "engineer_features",
    "engineer_batch",
]