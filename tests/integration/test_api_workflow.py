import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def auth_token():
    """Authenticate and return a JWT token."""
    response = client.post(
        "/api/auth/login",
        json={
            "email": "admin@finco.ai",
            "password": "password123",
        },
    )

    assert response.status_code == 200
    return response.json()["access_token"]


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------

def test_login():
    response = client.post(
        "/api/auth/login",
        json={
            "email": "admin@finco.ai",
            "password": "password123",
        },
    )

    assert response.status_code == 200
    assert "access_token" in response.json()


# ---------------------------------------------------------------------
# Company Workflow
# ---------------------------------------------------------------------

def test_create_company(auth_token):
    response = client.post(
        "/api/companies",
        headers=auth_headers(auth_token),
        json={
            "name": "FinCo Demo",
            "industry": "Finance",
        },
    )

    assert response.status_code in (200, 201)
    assert response.json()["name"] == "FinCo Demo"


# ---------------------------------------------------------------------
# Document Upload
# ---------------------------------------------------------------------

def test_upload_csv(auth_token):
    csv_data = (
        "date,revenue,expenses\n"
        "2025-01-01,100000,70000\n"
    )

    response = client.post(
        "/api/uploads/csv",
        headers=auth_headers(auth_token),
        files={
            "file": (
                "financial.csv",
                io.BytesIO(csv_data.encode()),
                "text/csv",
            )
        },
    )

    assert response.status_code in (200, 201)
    assert "document_id" in response.json()


# ---------------------------------------------------------------------
# Financial Analysis
# ---------------------------------------------------------------------

def test_financial_analysis(auth_token):
    response = client.get(
        "/api/financial/summary",
        headers=auth_headers(auth_token),
    )

    assert response.status_code == 200

    data = response.json()

    assert "revenue" in data
    assert "expenses" in data
    assert "profit" in data


# ---------------------------------------------------------------------
# Forecast API
# ---------------------------------------------------------------------

@patch("app.forecasting.model.ForecastService.forecast")
def test_forecast_endpoint(mock_forecast, auth_token):
    mock_forecast.return_value = [120000, 130000, 140000]

    response = client.post(
        "/api/forecasting/revenue",
        headers=auth_headers(auth_token),
        json={"periods": 3},
    )

    assert response.status_code == 200
    assert len(response.json()["forecast"]) == 3


# ---------------------------------------------------------------------
# Fraud Detection
# ---------------------------------------------------------------------

@patch("app.fraud.fraud_service.FraudService.evaluate")
def test_fraud_endpoint(mock_evaluate, auth_token):
    mock_evaluate.return_value = {
        "risk_score": 92,
        "risk_level": "CRITICAL",
        "is_anomaly": True,
    }

    response = client.post(
        "/api/fraud/analyze",
        headers=auth_headers(auth_token),
        json={
            "amount": 50000,
            "vendor": "ABC Ltd",
        },
    )

    assert response.status_code == 200
    assert response.json()["risk_level"] == "CRITICAL"


# ---------------------------------------------------------------------
# What-If Simulation
# ---------------------------------------------------------------------

def test_what_if_endpoint(auth_token):
    response = client.post(
        "/api/what_if/simulate",
        headers=auth_headers(auth_token),
        json={
            "revenue_change": 10,
            "expense_change": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "profit" in data
    assert "revenue" in data


# ---------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------

@patch("app.alerts.alert_service.AlertService.process")
def test_alert_endpoint(mock_process, auth_token):
    mock_process.return_value = {
        "triggered": True,
        "severity": "HIGH",
        "title": "Revenue Decline",
    }

    response = client.get(
        "/api/alerts",
        headers=auth_headers(auth_token),
    )

    assert response.status_code == 200
    assert response.json()[0]["severity"] == "HIGH"


# ---------------------------------------------------------------------
# Copilot (RAG)
# ---------------------------------------------------------------------

@patch("app.agents.supervisor_agent.SupervisorAgent.run")
def test_copilot_endpoint(mock_run, auth_token):
    mock_run.return_value = {
        "answer": "Revenue declined due to higher payroll costs.",
        "citations": [
            {
                "source": "FY2025 Annual Report",
                "page": 42,
            }
        ],
    }

    response = client.post(
        "/api/copilot/chat",
        headers=auth_headers(auth_token),
        json={
            "question": "Why did revenue decline?",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "answer" in body
    assert len(body["citations"]) == 1


# ---------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------

def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"