"""
FinCo AI - Fraud Anomaly Detection Model

File:
    backend/app/fraud/anomaly_model.py

Purpose:
    Detect potentially fraudulent or unusual financial transactions
    using unsupervised anomaly detection.

Supported approaches:
    - Isolation Forest
    - Statistical z-score detection
    - Rule-based anomaly detection
    - Ensemble anomaly scoring

Architecture:

    Transaction Data
          |
          v
    preprocessing.py
          |
          v
    feature_engineering.py
          |
          v
    anomaly_model.py
          |
          +--------------------+
          |                    |
          v                    v
    Isolation Forest      Statistical Signals
          |                    |
          +---------+----------+
                    |
                    v
              Anomaly Score
                    |
                    v
               scoring.py
                    |
                    v
              thresholds.py
                    |
                    v
              fraud_service.py

Design goals:
    - Deterministic
    - Explainable
    - No LLM dependency
    - Safe numerical handling
    - Easy model replacement
    - Batch and single-record inference
    - Production-friendly interfaces
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math
import statistics


# ============================================================================
# Optional sklearn dependency
# ============================================================================

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True

except ImportError:
    IsolationForest = None  # type: ignore
    SKLEARN_AVAILABLE = False


# ============================================================================
# Exceptions
# ============================================================================


class AnomalyModelError(Exception):
    """Base exception for anomaly model failures."""


class ModelNotFittedError(AnomalyModelError):
    """Raised when prediction is attempted before fitting."""


class InvalidFeatureError(AnomalyModelError):
    """Raised when invalid feature data is supplied."""


class InsufficientDataError(AnomalyModelError):
    """Raised when there is not enough data to train the model."""


# ============================================================================
# Enums
# ============================================================================


class AnomalyMethod(str, Enum):
    """Supported anomaly detection methods."""

    ISOLATION_FOREST = "isolation_forest"
    STATISTICAL = "statistical"
    RULE_BASED = "rule_based"
    ENSEMBLE = "ensemble"


class AnomalyLabel(str, Enum):
    """Human-readable anomaly classification."""

    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class AnomalyModelConfig:
    """
    Configuration for anomaly detection.
    """

    contamination: float = 0.02

    n_estimators: int = 200

    random_state: int = 42

    max_samples: str | int = "auto"

    statistical_z_threshold: float = 3.0

    suspicious_score_threshold: float = 40.0

    high_risk_score_threshold: float = 70.0

    critical_score_threshold: float = 90.0

    amount_z_weight: float = 0.35

    frequency_z_weight: float = 0.20

    velocity_weight: float = 0.20

    isolation_weight: float = 0.25

    rule_weight: float = 0.10

    minimum_training_samples: int = 20

    def validate(self) -> None:
        if not 0 < self.contamination < 0.5:
            raise AnomalyModelError(
                "contamination must be between 0 and 0.5."
            )

        if self.n_estimators <= 0:
            raise AnomalyModelError(
                "n_estimators must be greater than zero."
            )

        if self.statistical_z_threshold <= 0:
            raise AnomalyModelError(
                "statistical_z_threshold must be positive."
            )

        thresholds = (
            self.suspicious_score_threshold,
            self.high_risk_score_threshold,
            self.critical_score_threshold,
        )

        if not (
            0 <= thresholds[0]
            < thresholds[1]
            < thresholds[2]
            <= 100
        ):
            raise AnomalyModelError(
                "Anomaly score thresholds must be increasing "
                "and within 0-100."
            )

        if self.minimum_training_samples < 2:
            raise AnomalyModelError(
                "minimum_training_samples must be at least 2."
            )


# ============================================================================
# Result Models
# ============================================================================


@dataclass
class AnomalyResult:
    """
    Result for a single transaction.
    """

    transaction_id: Optional[str]

    anomaly_score: float

    anomaly_label: str

    is_anomaly: bool

    isolation_score: Optional[float] = None

    statistical_score: float = 0.0

    rule_score: float = 0.0

    confidence: float = 0.0

    reasons: List[str] = field(
        default_factory=list
    )

    signals: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetrics:
    """
    Training metadata.
    """

    samples: int

    features: int

    contamination: float

    method: str

    fitted: bool

    feature_names: List[str]

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Statistical Utility
# ============================================================================


class RobustStatistics:
    """
    Utility for calculating numerical anomaly signals.

    The implementation uses mean/std by default and protects against
    zero variance.
    """

    @staticmethod
    def mean(
        values: Sequence[float],
    ) -> float:

        if not values:
            return 0.0

        return float(
            statistics.fmean(values)
        )

    @staticmethod
    def std(
        values: Sequence[float],
    ) -> float:

        if len(values) < 2:
            return 0.0

        return float(
            statistics.stdev(values)
        )

    @classmethod
    def z_score(
        cls,
        value: float,
        values: Sequence[float],
    ) -> float:

        mean = cls.mean(values)

        std = cls.std(values)

        if std <= 1e-12:
            return 0.0

        return (
            value - mean
        ) / std

    @staticmethod
    def normalize_z_score(
        z_score: float,
        threshold: float = 3.0,
    ) -> float:
        """
        Convert absolute z-score into a 0-100 anomaly score.
        """

        magnitude = abs(
            float(z_score)
        )

        if magnitude <= threshold:
            return min(
                100.0,
                (
                    magnitude
                    / threshold
                ) * 50.0,
            )

        score = 50.0 + (
            (
                magnitude
                - threshold
            )
            / max(
                threshold,
                1e-12,
            )
        ) * 50.0

        return min(
            100.0,
            score,
        )


# ============================================================================
# Main Model
# ============================================================================


class AnomalyModel:
    """
    Fraud anomaly detection model.

    Primary model:
        Isolation Forest

    Additional signals:
        - Amount deviation
        - Transaction frequency deviation
        - Velocity
        - Rule-based indicators

    The class intentionally does not create fraud alerts itself.
    Alert creation belongs to fraud_service.py / alerts/.
    """

    DEFAULT_FEATURES = [
        "amount",
        "transaction_frequency",
        "velocity",
        "account_age_days",
        "balance",
        "amount_to_balance_ratio",
    ]

    def __init__(
        self,
        config: Optional[AnomalyModelConfig] = None,
        method: AnomalyMethod = AnomalyMethod.ENSEMBLE,
    ) -> None:

        self.config = (
            config
            or AnomalyModelConfig()
        )

        self.config.validate()

        self.method = method

        self.model: Any = None

        self.feature_names: List[str] = []

        self.training_statistics: Dict[
            str,
            Dict[str, float],
        ] = {}

        self._fitted = False

    # ========================================================================
    # FIT
    # ========================================================================

    def fit(
        self,
        X: Sequence[Sequence[float]],
        feature_names: Optional[
            Sequence[str]
        ] = None,
    ) -> ModelMetrics:
        """
        Train the Isolation Forest model.

        Parameters
        ----------
        X:
            Matrix of numerical features.

        feature_names:
            Names corresponding to columns in X.
        """

        rows = self._validate_matrix(
            X
        )

        if (
            len(rows)
            < self.config.minimum_training_samples
        ):
            raise InsufficientDataError(
                "At least "
                f"{self.config.minimum_training_samples} "
                "training samples are required."
            )

        width = len(rows[0])

        if feature_names is None:

            self.feature_names = [
                f"feature_{i}"
                for i in range(width)
            ]

        else:

            self.feature_names = list(
                feature_names
            )

            if len(
                self.feature_names
            ) != width:
                raise InvalidFeatureError(
                    "feature_names length must "
                    "match feature count."
                )

        self._fit_statistics(
            rows
        )

        if (
            self.method
            in (
                AnomalyMethod.ISOLATION_FOREST,
                AnomalyMethod.ENSEMBLE,
            )
        ):

            if not SKLEARN_AVAILABLE:
                raise AnomalyModelError(
                    "scikit-learn is required for "
                    "Isolation Forest detection."
                )

            self.model = IsolationForest(
                n_estimators=(
                    self.config.n_estimators
                ),
                contamination=(
                    self.config.contamination
                ),
                random_state=(
                    self.config.random_state
                ),
                max_samples=(
                    self.config.max_samples
                ),
            )

            self.model.fit(
                rows
            )

        self._fitted = True

        return ModelMetrics(
            samples=len(rows),
            features=width,
            contamination=(
                self.config.contamination
            ),
            method=self.method.value,
            fitted=True,
            feature_names=list(
                self.feature_names
            ),
        )

    # ========================================================================
    # SINGLE PREDICTION
    # ========================================================================

    def predict(
        self,
        features: Sequence[float],
        transaction_id: Optional[str] = None,
        transaction: Optional[
            Dict[str, Any]
        ] = None,
    ) -> AnomalyResult:
        """
        Detect anomaly for one transaction.
        """

        if not self._fitted:
            raise ModelNotFittedError(
                "Model must be fitted before prediction."
            )

        vector = self._validate_vector(
            features
        )

        isolation_score: Optional[
            float
        ] = None

        statistical_score = 0.0

        rule_score = 0.0

        # --------------------------------------------------------------
        # Isolation Forest
        # --------------------------------------------------------------

        if (
            self.method
            in (
                AnomalyMethod.ISOLATION_FOREST,
                AnomalyMethod.ENSEMBLE,
            )
        ):

            isolation_score = (
                self._isolation_score(
                    vector
                )
            )

        # --------------------------------------------------------------
        # Statistical model
        # --------------------------------------------------------------

        if (
            self.method
            in (
                AnomalyMethod.STATISTICAL,
                AnomalyMethod.ENSEMBLE,
            )
        ):

            statistical_score = (
                self._statistical_score(
                    vector
                )
            )

        # --------------------------------------------------------------
        # Rule model
        # --------------------------------------------------------------

        if transaction:

            rule_score = (
                self._rule_score(
                    transaction
                )
            )

        # --------------------------------------------------------------
        # Final score
        # --------------------------------------------------------------

        final_score = (
            self._combine_scores(
                isolation_score=
                    isolation_score,
                statistical_score=
                    statistical_score,
                rule_score=
                    rule_score,
            )
        )

        label = self.classify_score(
            final_score
        )

        reasons = (
            self.explain(
                vector=vector,
                transaction=transaction,
                isolation_score=
                    isolation_score,
                statistical_score=
                    statistical_score,
                rule_score=
                    rule_score,
            )
        )

        signals = (
            self._build_signals(
                vector=vector,
                transaction=transaction,
            )
        )

        confidence = (
            self._calculate_confidence(
                final_score
            )
        )

        return AnomalyResult(
            transaction_id=transaction_id,
            anomaly_score=round(
                final_score,
                4,
            ),
            anomaly_label=label,
            is_anomaly=(
                label
                != AnomalyLabel.NORMAL.value
            ),
            isolation_score=(
                None
                if isolation_score is None
                else round(
                    isolation_score,
                    4,
                )
            ),
            statistical_score=round(
                statistical_score,
                4,
            ),
            rule_score=round(
                rule_score,
                4,
            ),
            confidence=round(
                confidence,
                4,
            ),
            reasons=reasons,
            signals=signals,
            metadata={
                "method":
                    self.method.value,
                "model_fitted":
                    self._fitted,
            },
        )

    # ========================================================================
    # BATCH PREDICTION
    # ========================================================================

    def predict_batch(
        self,
        X: Sequence[Sequence[float]],
        transaction_ids: Optional[
            Sequence[str]
        ] = None,
        transactions: Optional[
            Sequence[Dict[str, Any]]
        ] = None,
    ) -> List[AnomalyResult]:
        """
        Predict anomalies for multiple transactions.
        """

        if transaction_ids is not None:
            if len(transaction_ids) != len(X):
                raise InvalidFeatureError(
                    "transaction_ids length must "
                    "match X length."
                )

        if transactions is not None:
            if len(transactions) != len(X):
                raise InvalidFeatureError(
                    "transactions length must "
                    "match X length."
                )

        results: List[
            AnomalyResult
        ] = []

        for index, vector in enumerate(X):

            transaction_id = (
                transaction_ids[index]
                if transaction_ids
                else None
            )

            transaction = (
                transactions[index]
                if transactions
                else None
            )

            results.append(
                self.predict(
                    features=vector,
                    transaction_id=
                        transaction_id,
                    transaction=
                        transaction,
                )
            )

        return results

    # ========================================================================
    # ISOLATION FOREST SCORE
    # ========================================================================

    def _isolation_score(
        self,
        vector: Sequence[float],
    ) -> float:
        """
        Convert Isolation Forest decision score to 0-100 anomaly score.

        IsolationForest:
            lower decision_function =
            more anomalous.

        We normalize the result into:
            0   = normal
            100 = highly anomalous
        """

        if self.model is None:
            raise ModelNotFittedError(
                "Isolation Forest is not fitted."
            )

        decision = float(
            self.model.decision_function(
                [list(vector)]
            )[0]
        )

        # Typical decision values are around
        # -0.5 to +0.5, but we intentionally
        # avoid assuming an exact range.

        score = (
            50.0
            - decision * 100.0
        )

        return max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

    # ========================================================================
    # STATISTICAL SCORE
    # ========================================================================

    def _statistical_score(
        self,
        vector: Sequence[float],
    ) -> float:
        """
        Calculate anomaly score from training distributions.
        """

        if not self.training_statistics:
            return 0.0

        z_scores: List[
            float
        ] = []

        for index, value in enumerate(
            vector
        ):

            if index >= len(
                self.feature_names
            ):
                continue

            name = (
                self.feature_names[index]
            )

            stats = (
                self.training_statistics.get(
                    name
                )
            )

            if not stats:
                continue

            mean = stats["mean"]

            std = stats["std"]

            if std <= 1e-12:
                z = 0.0

            else:
                z = (
                    value - mean
                ) / std

            z_scores.append(
                RobustStatistics.normalize_z_score(
                    z,
                    self.config.statistical_z_threshold,
                )
            )

        if not z_scores:
            return 0.0

        return max(
            z_scores
        )

    # ========================================================================
    # RULE SCORE
    # ========================================================================

    def _rule_score(
        self,
        transaction: Dict[str, Any],
    ) -> float:
        """
        Lightweight deterministic fraud signals.

        These are not final fraud decisions.
        They provide additional evidence for the ensemble.
        """

        score = 0.0

        # --------------------------------------------------------------
        # Large amount
        # --------------------------------------------------------------

        amount = self._number(
            transaction.get(
                "amount"
            )
        )

        if amount > 100000:
            score += 30.0

        elif amount > 50000:
            score += 20.0

        elif amount > 10000:
            score += 10.0

        # --------------------------------------------------------------
        # High transaction velocity
        # --------------------------------------------------------------

        velocity = self._number(
            transaction.get(
                "velocity"
            )
        )

        if velocity >= 20:
            score += 35.0

        elif velocity >= 10:
            score += 20.0

        elif velocity >= 5:
            score += 10.0

        # --------------------------------------------------------------
        # Amount / balance ratio
        # --------------------------------------------------------------

        ratio = self._number(
            transaction.get(
                "amount_to_balance_ratio"
            )
        )

        if ratio >= 1.0:
            score += 30.0

        elif ratio >= 0.75:
            score += 20.0

        elif ratio >= 0.50:
            score += 10.0

        # --------------------------------------------------------------
        # International / unusual transaction indicator
        # --------------------------------------------------------------

        if transaction.get(
            "is_international"
        ):

            score += 10.0

        # --------------------------------------------------------------
        # New device
        # --------------------------------------------------------------

        if transaction.get(
            "new_device"
        ):

            score += 15.0

        # --------------------------------------------------------------
        # New location
        # --------------------------------------------------------------

        if transaction.get(
            "new_location"
        ):

            score += 15.0

        # --------------------------------------------------------------
        # Final clamp
        # --------------------------------------------------------------

        return min(
            100.0,
            score,
        )

    # ========================================================================
    # ENSEMBLE
    # ========================================================================

    def _combine_scores(
        self,
        isolation_score: Optional[float],
        statistical_score: float,
        rule_score: float,
    ) -> float:
        """
        Combine anomaly signals.

        Ensemble weights:

            Isolation Forest : 25%
            Statistical      : 35%
            Rule signals     : 10%

        Remaining weight is normalized dynamically when
        one of the signals is unavailable.
        """

        weighted_values: List[
            Tuple[float, float]
        ] = []

        if isolation_score is not None:

            weighted_values.append(
                (
                    isolation_score,
                    self.config.isolation_weight,
                )
            )

        if (
            self.method
            in (
                AnomalyMethod.STATISTICAL,
                AnomalyMethod.ENSEMBLE,
            )
        ):

            weighted_values.append(
                (
                    statistical_score,
                    self.config.amount_z_weight
                    + self.config.frequency_z_weight,
                )
            )

        weighted_values.append(
            (
                rule_score,
                self.config.rule_weight,
            )
        )

        total_weight = sum(
            weight
            for _, weight
            in weighted_values
        )

        if total_weight <= 0:
            return 0.0

        score = sum(
            value * weight
            for value, weight
            in weighted_values
        ) / total_weight

        return max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

    # ========================================================================
    # CLASSIFICATION
    # ========================================================================

    def classify_score(
        self,
        score: float,
    ) -> str:
        """
        Convert anomaly score to classification.
        """

        score = max(
            0.0,
            min(
                100.0,
                float(score),
            ),
        )

        if (
            score
            >= self.config.critical_score_threshold
        ):
            return AnomalyLabel.CRITICAL.value

        if (
            score
            >= self.config.high_risk_score_threshold
        ):
            return AnomalyLabel.HIGH_RISK.value

        if (
            score
            >= self.config.suspicious_score_threshold
        ):
            return AnomalyLabel.SUSPICIOUS.value

        return AnomalyLabel.NORMAL.value

    # ========================================================================
    # EXPLAINABILITY
    # ========================================================================

    def explain(
        self,
        vector: Sequence[float],
        transaction: Optional[
            Dict[str, Any]
        ],
        isolation_score: Optional[float],
        statistical_score: float,
        rule_score: float,
    ) -> List[str]:
        """
        Generate deterministic anomaly reasons.
        """

        reasons: List[str] = []

        # --------------------------------------------------------------
        # Statistical deviations
        # --------------------------------------------------------------

        for index, value in enumerate(
            vector
        ):

            if index >= len(
                self.feature_names
            ):
                continue

            name = (
                self.feature_names[index]
            )

            stats = (
                self.training_statistics.get(
                    name
                )
            )

            if not stats:
                continue

            std = stats["std"]

            if std <= 1e-12:
                continue

            z = (
                value - stats["mean"]
            ) / std

            if abs(z) >= (
                self.config.statistical_z_threshold
            ):

                direction = (
                    "above"
                    if z > 0
                    else "below"
                )

                reasons.append(
                    f"{name} is significantly "
                    f"{direction} its historical "
                    f"baseline."
                )

        # --------------------------------------------------------------
        # Isolation Forest
        # --------------------------------------------------------------

        if (
            isolation_score is not None
            and isolation_score >= 70
        ):

            reasons.append(
                "Isolation Forest identified "
                "the transaction as unusually "
                "different from historical "
                "transaction patterns."
            )

        # --------------------------------------------------------------
        # Rule-based signals
        # --------------------------------------------------------------

        if (
            transaction
            and rule_score >= 40
        ):

            reasons.append(
                "Multiple transaction risk "
                "signals were detected."
            )

        if not reasons:

            reasons.append(
                "No significant anomaly "
                "signal was identified."
            )

        return reasons

    # ========================================================================
    # SIGNALS
    # ========================================================================

    def _build_signals(
        self,
        vector: Sequence[float],
        transaction: Optional[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        signals: Dict[
            str,
            Any
        ] = {}

        for index, value in enumerate(
            vector
        ):

            if index < len(
                self.feature_names
            ):

                signals[
                    self.feature_names[index]
                ] = value

        if transaction:

            for key in (
                "is_international",
                "new_device",
                "new_location",
                "velocity",
                "amount_to_balance_ratio",
            ):

                if key in transaction:

                    signals[key] = (
                        transaction[key]
                    )

        return signals

    # ========================================================================
    # CONFIDENCE
    # ========================================================================

    @staticmethod
    def _calculate_confidence(
        anomaly_score: float,
    ) -> float:
        """
        Convert distance from decision boundaries into confidence.

        This is model confidence, not probability of fraud.
        """

        score = max(
            0.0,
            min(
                100.0,
                anomaly_score,
            ),
        )

        if score < 40:
            return max(
                0.50,
                1.0 - score / 100.0,
            )

        if score < 70:
            return 0.70

        if score < 90:
            return 0.85

        return 0.95

    # ========================================================================
    # TRAINING STATISTICS
    # ========================================================================

    def _fit_statistics(
        self,
        rows: Sequence[
            Sequence[float]
        ],
    ) -> None:

        self.training_statistics = {}

        if not rows:
            return

        width = len(
            rows[0]
        )

        for index in range(
            width
        ):

            values = [
                float(row[index])
                for row in rows
            ]

            name = (
                self.feature_names[index]
            )

            self.training_statistics[
                name
            ] = {
                "mean":
                    RobustStatistics.mean(
                        values
                    ),
                "std":
                    RobustStatistics.std(
                        values
                    ),
                "min":
                    min(values),
                "max":
                    max(values),
            }

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_matrix(
        self,
        X: Sequence[
            Sequence[float]
        ],
    ) -> List[
        List[float]
    ]:

        if X is None or not X:
            raise InvalidFeatureError(
                "Feature matrix cannot be empty."
            )

        rows: List[
            List[float]
        ] = []

        expected_width: Optional[
            int
        ] = None

        for row in X:

            if row is None:
                raise InvalidFeatureError(
                    "Feature row cannot be None."
                )

            values = self._validate_vector(
                row
            )

            if expected_width is None:

                expected_width = len(
                    values
                )

            elif len(values) != expected_width:

                raise InvalidFeatureError(
                    "All feature rows must "
                    "have the same length."
                )

            rows.append(values)

        return rows

    def _validate_vector(
        self,
        vector: Sequence[float],
    ) -> List[float]:

        if vector is None:
            raise InvalidFeatureError(
                "Feature vector cannot be None."
            )

        if len(vector) == 0:
            raise InvalidFeatureError(
                "Feature vector cannot be empty."
            )

        result: List[
            float
        ] = []

        for value in vector:

            try:
                numeric = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise InvalidFeatureError(
                    f"Invalid numeric feature: {value!r}"
                ) from exc

            if not math.isfinite(
                numeric
            ):
                raise InvalidFeatureError(
                    f"Feature must be finite: {value!r}"
                )

            result.append(
                numeric
            )

        return result

    # ========================================================================
    # UTILITIES
    # ========================================================================

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            result = float(
                value
            )

            if not math.isfinite(
                result
            ):
                return default

            return result

        except (
            TypeError,
            ValueError,
        ):
            return default

    # ========================================================================
    # MODEL STATUS
    # ========================================================================

    @property
    def is_fitted(
        self,
    ) -> bool:
        return self._fitted

    def get_model_info(
        self,
    ) -> Dict[str, Any]:

        return {
            "model":
                "IsolationForest"
                if self.model is not None
                else None,

            "method":
                self.method.value,

            "fitted":
                self._fitted,

            "feature_names":
                list(
                    self.feature_names
                ),

            "training_statistics":
                self.training_statistics,

            "config":
                asdict(
                    self.config
                ),
        }


# ============================================================================
# Factory
# ============================================================================


def create_anomaly_model(
    method: AnomalyMethod = AnomalyMethod.ENSEMBLE,
    config: Optional[
        AnomalyModelConfig
    ] = None,
) -> AnomalyModel:
    """
    Factory for creating anomaly models.
    """

    return AnomalyModel(
        config=config,
        method=method,
    )


# ============================================================================
# Convenience Function
# ============================================================================


def detect_anomalies(
    X: Sequence[Sequence[float]],
    transaction_ids: Optional[
        Sequence[str]
    ] = None,
    transactions: Optional[
        Sequence[Dict[str, Any]]
    ] = None,
    feature_names: Optional[
        Sequence[str]
    ] = None,
    config: Optional[
        AnomalyModelConfig
    ] = None,
) -> List[AnomalyResult]:
    """
    Train an anomaly model and immediately
    perform batch detection.

    Useful for experiments and pipeline jobs.

    For long-running production services,
    prefer creating and fitting AnomalyModel once.
    """

    model = AnomalyModel(
        config=config,
        method=AnomalyMethod.ENSEMBLE,
    )

    model.fit(
        X,
        feature_names=feature_names,
    )

    return model.predict_batch(
        X,
        transaction_ids=transaction_ids,
        transactions=transactions,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "AnomalyModelError",
    "ModelNotFittedError",
    "InvalidFeatureError",
    "InsufficientDataError",
    "AnomalyMethod",
    "AnomalyLabel",
    "AnomalyModelConfig",
    "AnomalyResult",
    "ModelMetrics",
    "RobustStatistics",
    "AnomalyModel",
    "create_anomaly_model",
    "detect_anomalies",
]