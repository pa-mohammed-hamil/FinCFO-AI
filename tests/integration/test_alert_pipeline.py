import pytest
from unittest.mock import Mock

from app.alerts.alert_service import AlertService
from app.alerts.severity import HIGH, CRITICAL


# ---------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def financial_snapshot():
    return {
        "company_id": "finco-001",
        "revenue": 8_200_000,
        "previous_revenue": 10_100_000,
        "expenses": 7_900_000,
        "net_profit": 300_000,
        "previous_profit": 900_000,
        "health_score": 42,
    }


@pytest.fixture
def alert_service():
    risk_engine = Mock()
    rule_engine = Mock()
    generator = Mock()
    recommender = Mock()
    auditor = Mock()
    notifier = Mock()

    service = AlertService(
        risk_engine=risk_engine,
        rule_engine=rule_engine,
        generator=generator,
        recommender=recommender,
        auditor=auditor,
        notifier=notifier,
    )

    return service


# ---------------------------------------------------------------------
# Previous Period Decline
# ---------------------------------------------------------------------

def test_previous_period_decline_pipeline(
    alert_service,
    financial_snapshot,
):
    alert_service.risk_engine.calculate.return_value = 76

    alert_service.rule_engine.evaluate.return_value = {
        "triggered": True,
        "rule": "previous_period_decline",
        "severity": HIGH,
    }

    alert_service.generator.generate.return_value = {
        "title": "Revenue Decline",
        "severity": HIGH,
        "evidence": ["Revenue fell from 10.1M to 8.2M"],
    }

    alert_service.recommender.generate.return_value = [
        "Review marketing spend",
        "Improve customer retention",
    ]

    result = alert_service.process(financial_snapshot)

    assert result["triggered"] is True
    assert result["severity"] == HIGH

    alert_service.auditor.log.assert_called_once()
    alert_service.notifier.send.assert_called_once()


# ---------------------------------------------------------------------
# Financial Profile Down
# ---------------------------------------------------------------------

def test_financial_profile_down_pipeline(
    alert_service,
    financial_snapshot,
):
    financial_snapshot["health_score"] = 35

    alert_service.risk_engine.calculate.return_value = 82

    alert_service.rule_engine.evaluate.return_value = {
        "triggered": True,
        "rule": "financial_profile_down",
        "severity": HIGH,
    }

    alert_service.generator.generate.return_value = {
        "title": "Financial Health Declining",
        "severity": HIGH,
    }

    alert_service.recommender.generate.return_value = [
        "Reduce discretionary spending",
    ]

    result = alert_service.process(financial_snapshot)

    assert result["rule"] == "financial_profile_down"
    assert result["severity"] == HIGH

    alert_service.notifier.send.assert_called_once()


# ---------------------------------------------------------------------
# Severe Loss Risk
# ---------------------------------------------------------------------

def test_severe_loss_risk_pipeline(alert_service):
    snapshot = {
        "company_id": "finco-001",
        "forecast_profit": -1_500_000,
        "health_score": 18,
    }

    alert_service.risk_engine.calculate.return_value = 98

    alert_service.rule_engine.evaluate.return_value = {
        "triggered": True,
        "rule": "severe_loss_risk",
        "severity": CRITICAL,
    }

    alert_service.generator.generate.return_value = {
        "title": "Severe Loss Risk",
        "severity": CRITICAL,
    }

    alert_service.recommender.generate.return_value = [
        "Freeze discretionary spending",
        "Review cash flow immediately",
    ]

    result = alert_service.process(snapshot)

    assert result["severity"] == CRITICAL

    alert_service.auditor.log.assert_called_once()
    alert_service.notifier.send.assert_called_once()


# ---------------------------------------------------------------------
# No Alert Scenario
# ---------------------------------------------------------------------

def test_no_alert_pipeline(alert_service):
    snapshot = {
        "company_id": "finco-001",
        "revenue": 10_000_000,
        "previous_revenue": 9_900_000,
        "health_score": 88,
    }

    alert_service.risk_engine.calculate.return_value = 12

    alert_service.rule_engine.evaluate.return_value = {
        "triggered": False,
    }

    result = alert_service.process(snapshot)

    assert result["triggered"] is False

    alert_service.notifier.send.assert_not_called()
    alert_service.auditor.log.assert_not_called()


# ---------------------------------------------------------------------
# Recommendation & Evidence
# ---------------------------------------------------------------------

def test_alert_contains_evidence_and_recommendations(
    alert_service,
    financial_snapshot,
):
    alert_service.risk_engine.calculate.return_value = 74

    alert_service.rule_engine.evaluate.return_value = {
        "triggered": True,
        "rule": "previous_period_decline",
        "severity": HIGH,
    }

    alert_service.generator.generate.return_value = {
        "title": "Revenue Decline",
        "severity": HIGH,
        "evidence": [
            "Revenue dropped 18.8%",
            "Payroll costs increased",
        ],
    }

    alert_service.recommender.generate.return_value = [
        "Reduce operating costs",
        "Increase customer retention",
    ]

    result = alert_service.process(financial_snapshot)

    assert len(result["evidence"]) == 2
    assert len(result["recommendations"]) == 2