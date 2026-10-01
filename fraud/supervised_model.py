"""
FinCo AI - Supervised Fraud Detection Model

File:
    backend/app/fraud/supervised_model.py

Purpose:
    Train and execute a supervised machine-learning model for
    transaction fraud classification.

Pipeline:

    Raw Transaction
          |
          v
    preprocessing.py
          |
          v
    feature_engineering.py
          |
          v
    supervised_model.py
          |
          +----------------------+
          |                      |
          v                      v
    Fraud Probability       Fraud Prediction
          |                      |
          +----------+-----------+
                     |
                     v
                 scoring.py
                     |
                     v
               fraud_service.py

Design:
    - Binary fraud classification
    - Model-agnostic estimator interface
    - Works with scikit-learn compatible estimators
    - Supports probability prediction
    - Supports batch prediction
    - Feature-name aware
    - Deterministic preprocessing
    - No LLM dependency
    - No persistence dependency
    - Suitable for online and batch inference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import math


# ============================================================================
# Exceptions
# ============================================================================


class SupervisedModelError(Exception):
    """Base exception for supervised fraud model errors."""


class ModelNotFittedError(
    SupervisedModelError
):
    """Raised when prediction is attempted before fitting."""


class ModelTrainingError(
    SupervisedModelError
):
    """Raised when model training fails."""


class ModelPredictionError(
    SupervisedModelError
):
    """Raised when model prediction fails."""


class InvalidTrainingDataError(
    SupervisedModelError
):
    """Raised when training data is invalid."""


class InvalidFeatureError(
    SupervisedModelError
):
    """Raised when feature input is invalid."""


# ============================================================================
# Constants
# ============================================================================


class FraudLabel:
    """Canonical binary fraud labels."""

    LEGITIMATE = 0
    FRAUD = 1


class PredictionLabel:
    """Human-readable prediction labels."""

    LEGITIMATE = "LEGITIMATE"
    FRAUD = "FRAUD"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class SupervisedModelConfig:
    """
    Configuration for supervised fraud detection.

    The default estimator is intentionally created lazily so importing
    this module does not require scikit-learn.
    """

    positive_label: int = FraudLabel.FRAUD

    negative_label: int = FraudLabel.LEGITIMATE

    fraud_probability_threshold: float = 0.50

    feature_names: Optional[
        Sequence[str]
    ] = None

    handle_missing_features: bool = True

    missing_feature_value: float = 0.0

    clip_probability: bool = True

    store_training_metadata: bool = True

    def validate(self) -> None:

        if not (
            0.0
            <= self.fraud_probability_threshold
            <= 1.0
        ):

            raise ValueError(
                "fraud_probability_threshold must "
                "be between 0 and 1."
            )

        if (
            self.positive_label
            == self.negative_label
        ):

            raise ValueError(
                "positive_label and negative_label "
                "must be different."
            )

        if not math.isfinite(
            float(
                self.missing_feature_value
            )
        ):

            raise ValueError(
                "missing_feature_value must be finite."
            )


# ============================================================================
# Training Result
# ============================================================================


@dataclass
class TrainingResult:
    """
    Training metadata returned by fit().
    """

    fitted: bool

    sample_count: int

    feature_count: int

    feature_names: List[str]

    positive_samples: int

    negative_samples: int

    fraud_rate: float

    model_type: str

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "fitted":
                self.fitted,

            "sample_count":
                self.sample_count,

            "feature_count":
                self.feature_count,

            "feature_names":
                list(
                    self.feature_names
                ),

            "positive_samples":
                self.positive_samples,

            "negative_samples":
                self.negative_samples,

            "fraud_rate":
                self.fraud_rate,

            "model_type":
                self.model_type,

            "metadata":
                dict(
                    self.metadata
                ),
        }


# ============================================================================
# Prediction Result
# ============================================================================


@dataclass
class FraudPrediction:
    """
    Prediction returned by the supervised model.
    """

    fraud_probability: float

    predicted_label: int

    predicted_class: str

    confidence: float

    transaction_id: Optional[str] = None

    feature_values: Dict[
        str,
        float
    ] = field(
        default_factory=dict
    )

    feature_importance: Dict[
        str,
        float
    ] = field(
        default_factory=dict
    )

    metadata: Dict[
        str,
        Any
    ] = field(
        default_factory=dict
    )

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {
            "fraud_probability":
                self.fraud_probability,

            "predicted_label":
                self.predicted_label,

            "predicted_class":
                self.predicted_class,

            "confidence":
                self.confidence,

            "transaction_id":
                self.transaction_id,

            "feature_values":
                dict(
                    self.feature_values
                ),

            "feature_importance":
                dict(
                    self.feature_importance
                ),

            "metadata":
                dict(
                    self.metadata
                ),
        }


# ============================================================================
# Main Supervised Model
# ============================================================================


class SupervisedFraudModel:
    """
    Supervised fraud classification model.

    The class accepts any estimator implementing a scikit-learn-like API.

    Supported estimator interfaces:

        fit(X, y)

        predict(X)

        predict_proba(X)

    Examples:

        LogisticRegression
        RandomForestClassifier
        GradientBoostingClassifier
        HistGradientBoostingClassifier
        XGBoost-compatible classifiers
        LightGBM-compatible classifiers

    The model itself does not select a business decision such as
    BLOCK or REVIEW. It only estimates fraud probability.
    """

    def __init__(
        self,
        estimator: Any = None,
        config: Optional[
            SupervisedModelConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or SupervisedModelConfig()
        )

        self.config.validate()

        self.estimator = (
            estimator
            if estimator is not None
            else self._create_default_estimator()
        )

        self._fitted = False

        self._feature_names: List[
            str
        ] = []

        self._training_metadata: Dict[
            str,
            Any
        ] = {}

    # ========================================================================
    # DEFAULT ESTIMATOR
    # ========================================================================

    @staticmethod
    def _create_default_estimator() -> Any:
        """
        Create a lightweight default classifier.

        Import is lazy so the module can still be imported in environments
        where scikit-learn is not installed.
        """

        try:

            from sklearn.linear_model import (
                LogisticRegression
            )

        except ImportError as exc:

            raise SupervisedModelError(
                "scikit-learn is required for the "
                "default supervised fraud model. "
                "Install scikit-learn or inject "
                "a compatible estimator."
            ) from exc

        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )

    # ========================================================================
    # FIT
    # ========================================================================

    def fit(
        self,
        X: Sequence[
            Sequence[float]
        ],
        y: Sequence[Any],
        feature_names: Optional[
            Sequence[str]
        ] = None,
    ) -> TrainingResult:
        """
        Train the supervised fraud model.

        Parameters:
            X:
                Numerical feature matrix.

            y:
                Binary fraud labels.

            feature_names:
                Names corresponding to columns in X.
        """

        matrix = self._validate_matrix(
            X
        )

        labels = self._validate_labels(
            y,
            len(matrix),
        )

        names = (
            list(feature_names)
            if feature_names is not None
            else self._configured_feature_names(
                len(matrix[0])
            )
        )

        self._validate_feature_names(
            names,
            len(matrix[0]),
        )

        try:

            self.estimator.fit(
                matrix,
                labels,
            )

        except Exception as exc:

            raise ModelTrainingError(
                "Supervised fraud model "
                "training failed."
            ) from exc

        self._feature_names = names

        self._fitted = True

        positive_count = sum(
            1
            for label in labels
            if self._is_positive_label(
                label
            )
        )

        negative_count = (
            len(labels)
            - positive_count
        )

        fraud_rate = (
            positive_count
            / len(labels)
        )

        self._training_metadata = {
            "sample_count":
                len(labels),

            "feature_count":
                len(names),

            "positive_samples":
                positive_count,

            "negative_samples":
                negative_count,

            "fraud_rate":
                fraud_rate,

            "model_type":
                type(
                    self.estimator
                ).__name__,
        }

        return TrainingResult(
            fitted=True,
            sample_count=len(labels),
            feature_count=len(names),
            feature_names=list(names),
            positive_samples=positive_count,
            negative_samples=negative_count,
            fraud_rate=fraud_rate,
            model_type=type(
                self.estimator
            ).__name__,
            metadata=dict(
                self._training_metadata
            ),
        )

    # ========================================================================
    # PREDICT
    # ========================================================================

    def predict(
        self,
        features: Sequence[float],
        transaction_id: Optional[
            str
        ] = None,
        transaction: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        """
        Predict fraud for one transaction.

        This method returns a dictionary compatible with
        FraudService._run_supervised_model().
        """

        self._ensure_fitted()

        vector = self._prepare_vector(
            features
        )

        probability = (
            self._predict_probability(
                vector
            )
        )

        predicted_label = (
            FraudLabel.FRAUD
            if (
                probability
                >= self.config.fraud_probability_threshold
            )
            else FraudLabel.LEGITIMATE
        )

        predicted_class = (
            PredictionLabel.FRAUD
            if predicted_label
            == FraudLabel.FRAUD
            else PredictionLabel.LEGITIMATE
        )

        confidence = max(
            probability,
            1.0 - probability,
        )

        importance = (
            self.feature_importance()
        )

        result = FraudPrediction(
            fraud_probability=round(
                probability,
                6,
            ),

            predicted_label=predicted_label,

            predicted_class=predicted_class,

            confidence=round(
                confidence,
                6,
            ),

            transaction_id=transaction_id,

            feature_values={
                name: value
                for name, value
                in zip(
                    self._feature_names,
                    vector,
                )
            },

            feature_importance=importance,

            metadata={
                "model_type":
                    type(
                        self.estimator
                    ).__name__,

                "threshold":
                    self.config
                    .fraud_probability_threshold,
            },
        )

        return result.to_dict()

    # ========================================================================
    # PREDICT PROBABILITY
    # ========================================================================

    def predict_proba(
        self,
        features: Sequence[float],
    ) -> Dict[str, Any]:
        """
        Predict fraud probability.

        Returns a dictionary rather than a raw probability so it integrates
        naturally with FraudService.
        """

        self._ensure_fitted()

        vector = self._prepare_vector(
            features
        )

        probability = (
            self._predict_probability(
                vector
            )
        )

        return {
            "fraud_probability":
                probability,

            "probability":
                probability,

            "confidence":
                max(
                    probability,
                    1.0 - probability,
                ),

            "model_type":
                type(
                    self.estimator
                ).__name__,
        }

    # ========================================================================
    # BATCH PREDICTION
    # ========================================================================

    def predict_batch(
        self,
        X: Sequence[
            Sequence[float]
        ],
        transaction_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> List[
        Dict[str, Any]
    ]:
        """
        Predict fraud for multiple transactions.
        """

        self._ensure_fitted()

        if transaction_ids is not None:

            if len(
                transaction_ids
            ) != len(X):

                raise InvalidFeatureError(
                    "transaction_ids length must "
                    "match X length."
                )

        results = []

        for index, vector in enumerate(
            X
        ):

            transaction_id = (
                transaction_ids[index]
                if transaction_ids
                is not None
                else None
            )

            results.append(
                self.predict(
                    vector,
                    transaction_id=(
                        transaction_id
                    ),
                )
            )

        return results

    # ========================================================================
    # RAW MODEL PREDICTION
    # ========================================================================

    def predict_labels(
        self,
        X: Sequence[
            Sequence[float]
        ],
    ) -> List[int]:
        """
        Return raw predicted labels.
        """

        self._ensure_fitted()

        matrix = self._validate_matrix(
            X
        )

        try:

            predictions = (
                self.estimator.predict(
                    matrix
                )
            )

        except Exception as exc:

            raise ModelPredictionError(
                "Raw model prediction failed."
            ) from exc

        return [
            self._normalize_label(
                value
            )
            for value in predictions
        ]

    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================

    def feature_importance(
        self,
    ) -> Dict[str, float]:
        """
        Extract feature importance from common estimator APIs.

        Supports:

            feature_importances_
            coef_

        If the estimator does not expose either,
        an empty dictionary is returned.
        """

        if not self._feature_names:
            return {}

        importance = None

        if hasattr(
            self.estimator,
            "feature_importances_",
        ):

            importance = (
                getattr(
                    self.estimator,
                    "feature_importances_",
                )
            )

        elif hasattr(
            self.estimator,
            "coef_",
        ):

            importance = (
                getattr(
                    self.estimator,
                    "coef_",
                )
            )

            # Binary logistic regression typically
            # returns [[coef1, coef2, ...]].

            try:

                if (
                    hasattr(
                        importance,
                        "ndim",
                    )
                    and importance.ndim > 1
                ):

                    importance = importance[0]

            except Exception:
                pass

        if importance is None:
            return {}

        values: Dict[
            str,
            float
        ] = {}

        try:

            for name, value in zip(
                self._feature_names,
                importance,
            ):

                number = self._number(
                    value
                )

                if math.isfinite(
                    number
                ):

                    values[
                        name
                    ] = abs(
                        number
                    )

        except Exception:

            return {}

        total = sum(
            values.values()
        )

        if total > 0:

            values = {
                name:
                    value / total
                for name, value
                in values.items()
            }

        return dict(
            sorted(
                values.items(),
                key=lambda item:
                    item[1],
                reverse=True,
            )
        )

    # ========================================================================
    # MODEL STATUS
    # ========================================================================

    @property
    def is_fitted(
        self,
    ) -> bool:

        return self._fitted

    @property
    def feature_names(
        self,
    ) -> List[str]:

        return list(
            self._feature_names
        )

    @property
    def training_metadata(
        self,
    ) -> Dict[str, Any]:

        return dict(
            self._training_metadata
        )

    # ========================================================================
    # MODEL SERIALIZATION HELPERS
    # ========================================================================

    def get_model(
        self,
    ) -> Any:
        """
        Return the underlying estimator.

        Useful for external persistence using joblib,
        pickle, MLflow, etc.
        """

        return self.estimator

    def set_model(
        self,
        estimator: Any,
        feature_names: Sequence[str],
    ) -> None:
        """
        Inject an already-trained estimator.
        """

        if estimator is None:
            raise ValueError(
                "estimator cannot be None."
            )

        names = list(
            feature_names
        )

        if not names:
            raise ValueError(
                "feature_names cannot be empty."
            )

        self.estimator = estimator

        self._feature_names = names

        self._fitted = True

    # ========================================================================
    # INTERNAL PROBABILITY LOGIC
    # ========================================================================

    def _predict_probability(
        self,
        vector: Sequence[float],
    ) -> float:

        matrix = [
            list(vector)
        ]

        # Preferred path.

        if hasattr(
            self.estimator,
            "predict_proba",
        ):

            try:

                probabilities = (
                    self.estimator.predict_proba(
                        matrix
                    )
                )

                probability = (
                    self._extract_positive_probability(
                        probabilities
                    )
                )

                return self._normalize_probability(
                    probability
                )

            except Exception as exc:

                raise ModelPredictionError(
                    "predict_proba() failed."
                ) from exc

        # Decision-function fallback.

        if hasattr(
            self.estimator,
            "decision_function",
        ):

            try:

                decision = (
                    self.estimator
                    .decision_function(
                        matrix
                    )
                )

                value = float(
                    decision[0]
                    if hasattr(
                        decision,
                        "__len__",
                    )
                    else decision
                )

                # Logistic transformation.

                probability = (
                    1.0
                    / (
                        1.0
                        + math.exp(
                            -max(
                                -50.0,
                                min(
                                    50.0,
                                    value,
                                ),
                            )
                        )
                    )
                )

                return self._normalize_probability(
                    probability
                )

            except Exception as exc:

                raise ModelPredictionError(
                    "decision_function() failed."
                ) from exc

        # Final fallback to predict().

        if hasattr(
            self.estimator,
            "predict",
        ):

            try:

                prediction = (
                    self.estimator.predict(
                        matrix
                    )[0]
                )

                return (
                    1.0
                    if self._is_positive_label(
                        prediction
                    )
                    else 0.0
                )

            except Exception as exc:

                raise ModelPredictionError(
                    "predict() fallback failed."
                ) from exc

        raise ModelPredictionError(
            "Estimator does not provide "
            "predict_proba(), decision_function(), "
            "or predict()."
        )

    # ========================================================================
    # PROBABILITY EXTRACTION
    # ========================================================================

    def _extract_positive_probability(
        self,
        probabilities: Any,
    ) -> float:

        try:

            row = probabilities[0]

        except (
            IndexError,
            TypeError,
        ) as exc:

            raise ModelPredictionError(
                "Invalid predict_proba() output."
            ) from exc

        # Determine positive-class column.

        classes = getattr(
            self.estimator,
            "classes_",
            None,
        )

        if classes is not None:

            for index, label in enumerate(
                classes
            ):

                if self._is_positive_label(
                    label
                ):

                    return self._number(
                        row[index]
                    )

        # Standard binary classifier fallback:
        # second probability = positive class.

        if len(row) >= 2:

            return self._number(
                row[1]
            )

        if len(row) == 1:

            return self._number(
                row[0]
            )

        raise ModelPredictionError(
            "Unable to extract positive-class "
            "probability."
        )

    # ========================================================================
    # FEATURE PREPARATION
    # ========================================================================

    def _prepare_vector(
        self,
        features: Sequence[float],
    ) -> List[float]:

        if isinstance(
            features,
            Mapping,
        ):

            return self._prepare_mapping(
                features
            )

        try:

            vector = [
                self._number(
                    value
                )
                for value in features
            ]

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidFeatureError(
                "Features must contain "
                "numeric values."
            ) from exc

        if not vector:

            raise InvalidFeatureError(
                "Feature vector cannot be empty."
            )

        if self._feature_names:

            expected = len(
                self._feature_names
            )

            if len(vector) != expected:

                raise InvalidFeatureError(
                    f"Expected {expected} features "
                    f"but received {len(vector)}."
                )

        return vector

    def _prepare_mapping(
        self,
        features: Mapping[str, Any],
    ) -> List[float]:

        if not self._feature_names:

            raise InvalidFeatureError(
                "Feature names are required when "
                "features are supplied as a mapping."
            )

        vector: List[
            float
        ] = []

        for name in (
            self._feature_names
        ):

            if name in features:

                vector.append(
                    self._number(
                        features[name]
                    )
                )

            elif (
                self.config
                .handle_missing_features
            ):

                vector.append(
                    self.config
                    .missing_feature_value
                )

            else:

                raise InvalidFeatureError(
                    f"Missing feature: {name}"
                )

        return vector

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _ensure_fitted(
        self,
    ) -> None:

        if not self._fitted:

            raise ModelNotFittedError(
                "Supervised fraud model has "
                "not been fitted."
            )

    @staticmethod
    def _validate_matrix(
        X: Sequence[
            Sequence[float]
        ],
    ) -> List[
        List[float]
    ]:

        if X is None:
            raise InvalidTrainingDataError(
                "Feature matrix cannot be None."
            )

        try:

            matrix = [
                [
                    float(value)
                    for value in row
                ]
                for row in X
            ]

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidTrainingDataError(
                "Feature matrix must contain "
                "numeric values."
            ) from exc

        if not matrix:

            raise InvalidTrainingDataError(
                "Feature matrix cannot be empty."
            )

        width = len(
            matrix[0]
        )

        if width == 0:

            raise InvalidTrainingDataError(
                "Feature vectors cannot be empty."
            )

        for row in matrix:

            if len(row) != width:

                raise InvalidTrainingDataError(
                    "All feature vectors must "
                    "have the same length."
                )

            for value in row:

                if not math.isfinite(
                    value
                ):

                    raise InvalidTrainingDataError(
                        "Feature values must be finite."
                    )

        return matrix

    def _validate_labels(
        self,
        y: Sequence[Any],
        expected_length: int,
    ) -> List[int]:

        if y is None:

            raise InvalidTrainingDataError(
                "Labels cannot be None."
            )

        if len(y) != expected_length:

            raise InvalidTrainingDataError(
                "X and y must contain the "
                "same number of samples."
            )

        labels = [
            self._normalize_label(
                value
            )
            for value in y
        ]

        unique = set(
            labels
        )

        if not unique.issubset(
            {
                FraudLabel.LEGITIMATE,
                FraudLabel.FRAUD,
            }
        ):

            raise InvalidTrainingDataError(
                "Labels must be binary "
                "0/1 values."
            )

        if len(unique) < 2:

            raise InvalidTrainingDataError(
                "Training data must contain "
                "both legitimate and fraud samples."
            )

        return labels

    def _validate_feature_names(
        self,
        names: Sequence[str],
        feature_count: int,
    ) -> None:

        if len(names) != feature_count:

            raise InvalidFeatureError(
                "Number of feature names must "
                "match number of feature columns."
            )

        cleaned = [
            str(name).strip()
            for name in names
        ]

        if any(
            not name
            for name in cleaned
        ):

            raise InvalidFeatureError(
                "Feature names cannot be empty."
            )

        if len(
            set(cleaned)
        ) != len(cleaned):

            raise InvalidFeatureError(
                "Feature names must be unique."
            )

    def _configured_feature_names(
        self,
        feature_count: int,
    ) -> List[str]:

        if self.config.feature_names:

            names = list(
                self.config.feature_names
            )

            self._validate_feature_names(
                names,
                feature_count,
            )

            return names

        return [
            f"feature_{index}"
            for index
            in range(
                feature_count
            )
        ]

    # ========================================================================
    # LABEL HELPERS
    # ========================================================================

    def _normalize_label(
        self,
        value: Any,
    ) -> int:

        if isinstance(
            value,
            str,
        ):

            normalized = (
                value.strip().lower()
            )

            if normalized in {
                "1",
                "fraud",
                "fraudulent",
                "true",
                "yes",
            }:

                return FraudLabel.FRAUD

            if normalized in {
                "0",
                "legitimate",
                "normal",
                "false",
                "no",
            }:

                return FraudLabel.LEGITIMATE

        try:

            number = int(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidTrainingDataError(
                f"Invalid fraud label: {value!r}"
            ) from exc

        if number not in {
            FraudLabel.LEGITIMATE,
            FraudLabel.FRAUD,
        }:

            raise InvalidTrainingDataError(
                f"Fraud label must be 0 or 1: "
                f"{value!r}"
            )

        return number

    @staticmethod
    def _is_positive_label(
        value: Any,
    ) -> bool:

        if isinstance(
            value,
            str,
        ):

            return (
                value.strip().lower()
                in {
                    "1",
                    "fraud",
                    "fraudulent",
                    "true",
                    "yes",
                }
            )

        try:

            return int(
                value
            ) == FraudLabel.FRAUD

        except (
            TypeError,
            ValueError,
        ):

            return False

    # ========================================================================
    # NUMERICAL HELPERS
    # ========================================================================

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

    def _normalize_probability(
        self,
        value: float,
    ) -> float:

        probability = float(
            value
        )

        # Support 0-100 probability outputs.

        if probability > 1.0:

            probability /= 100.0

        if self.config.clip_probability:

            probability = max(
                0.0,
                min(
                    1.0,
                    probability,
                ),
            )

        return probability


# ============================================================================
# Convenience API
# ============================================================================


def train_supervised_model(
    X: Sequence[
        Sequence[float]
    ],
    y: Sequence[Any],
    feature_names: Optional[
        Sequence[str]
    ] = None,
    estimator: Any = None,
    config: Optional[
        SupervisedModelConfig
    ] = None,
) -> tuple[
    SupervisedFraudModel,
    TrainingResult,
]:
    """
    Train and return a supervised fraud model.
    """

    model = SupervisedFraudModel(
        estimator=estimator,
        config=config,
    )

    result = model.fit(
        X,
        y,
        feature_names=feature_names,
    )

    return (
        model,
        result,
    )


def predict_fraud_probability(
    model: SupervisedFraudModel,
    features: Sequence[float],
) -> float:
    """
    Convenience probability prediction.
    """

    result = model.predict_proba(
        features
    )

    return float(
        result[
            "fraud_probability"
        ]
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "SupervisedModelError",
    "ModelNotFittedError",
    "ModelTrainingError",
    "ModelPredictionError",
    "InvalidTrainingDataError",
    "InvalidFeatureError",
    "FraudLabel",
    "PredictionLabel",
    "SupervisedModelConfig",
    "TrainingResult",
    "FraudPrediction",
    "SupervisedFraudModel",
    "train_supervised_model",
    "predict_fraud_probability",
]