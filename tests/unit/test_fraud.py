import pytest
from unittest.mock import Mock

from app.fraud.feature_engineering import create_features
from app.fraud.scoring import calculate_risk_score
from app.fraud.thresholds import classify_risk
from app.fraud.fraud_service import FraudService


# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------

def test_create_features_basic_transaction():
    transaction = {
        "amount": 5000,
        "vendor_frequency": 10,
        "days_since_last_payment": 7,
    }

    features = create_features(transaction)

    assert features["amount"] == 5000
    assert features["vendor_frequency"] == 10
    assert features["days_since_last_payment"] == 7


def test_create_features_handles_missing_values():
    transaction = {"amount": 1000}

    features = create_features(transaction)

    assert features["amount"] == 1000
    assert "vendor_frequency" in features
    assert "days_since_last_payment" in features


# ---------------------------------------------------------------------
# Risk Scoring
# ---------------------------------------------------------------------

def test_calculate_risk_score_high():
    probability = 0.92

    score = calculate_risk_score(probability)

    assert score >= 90


def test_calculate_risk_score_low():
    probability = 0.12

    score = calculate_risk_score(probability)

    assert score < 30


# ---------------------------------------------------------------------
# Threshold Classification
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "score,expected",
    [
        (10, "LOW"),
        (35, "MEDIUM"),
        (70, "HIGH"),
        (95, "CRITICAL"),
    ],
)
def test_classify_risk(score, expected):
    assert classify_risk(score) == expected


# ---------------------------------------------------------------------
# Fraud Service
# ---------------------------------------------------------------------

def test_fraud_service_detects_high_risk():
    model = Mock()
    model.predict_proba.return_value = [[0.08, 0.92]]

    explainer = Mock()
    explainer.explain.return_value = {
        "top_features": ["amount", "vendor_frequency"]
    }

    service = FraudService(model=model, explainer=explainer)

    transaction = {"amount": 50000}

    result = service.evaluate(transaction)

    assert result["risk_level"] in ("HIGH", "CRITICAL")
    assert result["risk_score"] >= 90
    assert "top_features" in result["explanation"]


def test_fraud_service_detects_low_risk():
    model = Mock()
    model.predict_proba.return_value = [[0.95, 0.05]]

    explainer = Mock()
    explainer.explain.return_value = {"top_features": []}

    service = FraudService(model=model, explainer=explainer)

    result = service.evaluate({"amount": 500})

    assert result["risk_level"] == "LOW"


# ---------------------------------------------------------------------
# Anomaly Handling
# ---------------------------------------------------------------------

def test_anomaly_detection_flags_large_transaction():
    anomaly_model = Mock()
    anomaly_model.predict.return_value = [-1]

    service = FraudService(anomaly_model=anomaly_model)

    result = service.detect_anomaly({"amount": 250000})

    assert result["is_anomaly"] is True


def test_anomaly_detection_normal_transaction():
    anomaly_model = Mock()
    anomaly_model.predict.return_value = [1]

    service = FraudService(anomaly_model=anomaly_model)

    result = service.detect_anomaly({"amount": 1200})

    assert result["is_anomaly"] is False


# ---------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------

def test_explanation_contains_top_features():
    model = Mock()
    model.predict_proba.return_value = [[0.2, 0.8]]

    explainer = Mock()
    explainer.explain.return_value = {
        "top_features": [
            "amount",
            "payment_time",
            "vendor_frequency",
        ]
    }

    service = FraudService(model=model, explainer=explainer)

    result = service.evaluate({"amount": 15000})

    assert len(result["explanation"]["top_features"]) == 3
    assert "amount" in result["explanation"]["top_features"]