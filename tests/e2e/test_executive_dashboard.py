import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def auth_headers():
    response = client.post(
        "/api/auth/login",
        json={
            "email": "admin@finco.ai",
            "password": "password123"
        },
    )

    assert response.status_code == 200

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------
# Dashboard Summary
# ---------------------------------------------------------------------

@patch("app.api.reports.get_dashboard_summary")
def test_dashboard_summary(mock_summary, auth_headers):
    mock_summary.return_value = {
        "revenue": 8_200_000,
        "expenses": 7_000_000,
        "profit": 1_200_000,
        "health_score": 82,
    }

    response = client.get(
        "/api/reports/dashboard",
        headers=auth_headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["revenue"] == 8_200_000
    assert data["profit"] == 1_200_000
    assert data["health_score"] == 82


# ---------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------

@patch("app.api.reports.get_kpis")
def test_dashboard_kpis(mock_kpis, auth_headers):
    mock_kpis.return_value = {
        "gross_margin": 61.5,
        "net_margin": 14.8,
        "current_ratio": 2.1,
        "debt_to_equity": 0.75,
    }

    response = client.get(
        "/api/reports/kpis",
        headers=auth_headers,
    )

    assert response.status_code == 200

    kpis = response.json()

    assert "gross_margin" in kpis
    assert "current_ratio" in kpis


# ---------------------------------------------------------------------
# Revenue Trend Chart
# ---------------------------------------------------------------------

@patch("app.api.reports.get_revenue_trend")
def test_revenue_trend(mock_trend, auth_headers):
    mock_trend.return_value = [
        {"month": "Jan", "revenue": 700000},
        {"month": "Feb", "revenue": 720000},
        {"month": "Mar", "revenue": 760000},
    ]

    response = client.get(
        "/api/reports/revenue-trend",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert len(response.json()) == 3


# ---------------------------------------------------------------------
# Forecast Widget
# ---------------------------------------------------------------------

@patch("app.forecasting.model.ForecastService.forecast")
def test_dashboard_forecast(mock_forecast, auth_headers):
    mock_forecast.return_value = [850000, 880000, 910000]

    response = client.get(
        "/api/reports/forecast",
        headers=auth_headers,
    )

    assert response.status_code == 200

    forecast = response.json()["forecast"]

    assert len(forecast) == 3
    assert forecast[-1] > forecast[0]


# ---------------------------------------------------------------------
# Active Alerts Widget
# ---------------------------------------------------------------------

@patch("app.alerts.alert_service.AlertService.list_active")
def test_dashboard_alerts(mock_alerts, auth_headers):
    mock_alerts.return_value = [
        {
            "title": "Revenue Decline",
            "severity": "HIGH",
        },
        {
            "title": "Cash Flow Risk",
            "severity": "CRITICAL",
        },
    ]

    response = client.get(
        "/api/reports/alerts",
        headers=auth_headers,
    )

    assert response.status_code == 200

    alerts = response.json()

    assert len(alerts) == 2
    assert alerts[1]["severity"] == "CRITICAL"


# ---------------------------------------------------------------------
# Executive Recommendations
# ---------------------------------------------------------------------

@patch("app.recommendations.recommendation_service.RecommendationService.generate")
def test_dashboard_recommendations(mock_recommendations, auth_headers):
    mock_recommendations.return_value = [
        "Reduce discretionary spending",
        "Improve customer retention",
        "Review payroll growth",
    ]

    response = client.get(
        "/api/reports/recommendations",
        headers=auth_headers,
    )

    assert response.status_code == 200

    recommendations = response.json()

    assert len(recommendations) == 3


# ---------------------------------------------------------------------
# Executive PDF Report
# ---------------------------------------------------------------------

@patch("app.reports.executive_report.generate_pdf")
def test_generate_executive_report(mock_pdf, auth_headers):
    mock_pdf.return_value = "/tmp/executive_report.pdf"

    response = client.post(
        "/api/reports/executive",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert "report_path" in response.json()


# ---------------------------------------------------------------------
# Dashboard Access Control
# ---------------------------------------------------------------------

def test_dashboard_requires_auth():
    response = client.get("/api/reports/dashboard")

    assert response.status_code in (401, 403)


# ---------------------------------------------------------------------
# Complete Dashboard Workflow
# ---------------------------------------------------------------------

@patch("app.api.reports.get_dashboard_summary")
@patch("app.api.reports.get_kpis")
@patch("app.alerts.alert_service.AlertService.list_active")
def test_complete_dashboard_workflow(
    mock_alerts,
    mock_kpis,
    mock_summary,
    auth_headers,
):
    mock_summary.return_value = {
        "revenue": 8_000_000,
        "profit": 1_000_000,
        "health_score": 80,
    }

    mock_kpis.return_value = {
        "gross_margin": 60,
        "net_margin": 15,
    }

    mock_alerts.return_value = [
        {"title": "Revenue Decline", "severity": "HIGH"}
    ]

    summary = client.get(
        "/api/reports/dashboard",
        headers=auth_headers,
    )

    kpis = client.get(
        "/api/reports/kpis",
        headers=auth_headers,
    )

    alerts = client.get(
        "/api/reports/alerts",
        headers=auth_headers,
    )

    assert summary.status_code == 200
    assert kpis.status_code == 200
    assert alerts.status_code == 200

    assert summary.json()["health_score"] == 80
    assert "gross_margin" in kpis.json()
    assert len(alerts.json()) == 1