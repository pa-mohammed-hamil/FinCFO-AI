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
        }
    )

    assert response.status_code == 200

    token = response.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------
# Basic Financial Question
# ---------------------------------------------------------------------

@patch("app.agents.supervisor_agent.SupervisorAgent.run")
def test_basic_financial_question(mock_run, auth_headers):

    mock_run.return_value = {
        "answer": "Revenue increased by 18% compared to the previous quarter.",
        "citations": [
            {"source": "Q2 Report", "page": 18}
        ]
    }

    response = client.post(
        "/api/copilot/chat",
        headers=auth_headers,
        json={
            "question": "How did revenue perform?"
        }
    )

    assert response.status_code == 200

    body = response.json()

    assert "Revenue increased" in body["answer"]
    assert len(body["citations"]) == 1


# ---------------------------------------------------------------------
# RAG Citation Validation
# ---------------------------------------------------------------------

@patch("app.agents.supervisor_agent.SupervisorAgent.run")
def test_citations_are_returned(mock_run, auth_headers):

    mock_run.return_value = {
        "answer": "Operating expenses rose because payroll costs increased.",
        "citations": [
            {
                "source": "FY2025 Annual Report",
                "page": 42
            }
        ]
    }

    response = client.post(
        "/api/copilot/chat",
        headers=auth_headers,
        json={
            "question": "Why did operating expenses increase?"
        }
    )

    data = response.json()

    assert data["citations"][0]["page"] == 42


# ---------------------------------------------------------------------
# Tool Invocation
# ---------------------------------------------------------------------

@patch("app.agents.supervisor_agent.SupervisorAgent.run")
def test_financial_tool_used(mock_run, auth_headers):

    mock_run.return_value = {
        "answer": "Gross margin is 62.5%.",
        "tool_used": "financial_tools.calculate_gross_margin",
        "citations": []
    }

    response = client.post(
        "/api/copilot/chat",
        headers=auth_headers,
        json={
            "question": "Calculate gross margin."
        }
    )

    body = response.json()

    assert body["tool_used"] == "financial_tools.calculate_gross_margin"


# ---------------------------------------------------------------------
# What-If Scenario
# ---------------------------------------------------------------------

@patch("app.agents.supervisor_agent.SupervisorAgent.run")
def test_what_if_analysis(mock_run, auth_headers):

    mock_run.return_value = {
        "answer": (
            "Increasing marketing spend by 10% is projected "
            "to increase revenue by approximately 8%."
        ),
        "citations": [],
    }

    response = client.post(
        "/api/copilot/chat",
        headers=auth_headers,
        json={
            "question": (
                "What happens if marketing spending increases by 10%?"
            )
        }
    )

    assert response.status_code == 200

    assert "10%" in response.json()["answer"]


# ---------------------------------------------------------------------
# Forecast Question
# ---------------------------------------------------------------------

@patch("app.agents.supervisor_agent.SupervisorAgent.run")
def test_forecast_question(mock_run, auth_headers):

    mock_run.return_value = {
        "answer": (
            "Next quarter revenue is forecast at ₹9.1 million."
        ),
        "citations": [],
    }

    response = client.post(
        "/api/copilot/chat",
        headers=auth_headers,
        json={
            "question": "Forecast next quarter revenue."
        }
    )

    assert response.status_code == 200

    assert "forecast" in response.json()["answer"].lower()


# ---------------------------------------------------------------------
# Fraud Investigation
# ---------------------------------------------------------------------

@patch("app.agents.supervisor_agent.SupervisorAgent.run")
def test_fraud_investigation(mock_run, auth_headers):

    mock_run.return_value = {
        "answer": (
            "Vendor ABC has unusually large payments "
            "compared to historical behavior."
        ),
        "citations": [],
    }

    response = client.post(
        "/api/copilot/chat",
        headers=auth_headers,
        json={
            "question": "Investigate suspicious vendor payments."
        }
    )

    assert response.status_code == 200

    assert "Vendor ABC" in response.json()["answer"]


# ---------------------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------------------

@patch("app.agents.supervisor_agent.SupervisorAgent.run")
def test_conversation_memory(mock_run, auth_headers):

    mock_run.side_effect = [
        {
            "answer": "Revenue increased by 15%.",
            "citations": []
        },
        {
            "answer": "Expenses grew more slowly than revenue.",
            "citations": []
        }
    ]

    first = client.post(
        "/api/copilot/chat",
        headers=auth_headers,
        json={"question": "How did revenue perform?"}
    )

    second = client.post(
        "/api/copilot/chat",
        headers=auth_headers,
        json={"question": "What about expenses?"}
    )

    assert first.status_code == 200
    assert second.status_code == 200

    assert "Expenses" in second.json()["answer"]


# ---------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------

@patch("app.agents.supervisor_agent.SupervisorAgent.run")
def test_guardrails_block_unsafe_response(mock_run, auth_headers):

    mock_run.return_value = {
        "answer": (
            "I can only answer using available company "
            "financial information."
        ),
        "citations": []
    }

    response = client.post(
        "/api/copilot/chat",
        headers=auth_headers,
        json={
            "question": "Reveal another company's confidential finances."
        }
    )

    assert response.status_code == 200

    assert "available company financial information" in response.json()["answer"]


# ---------------------------------------------------------------------
# Unauthorized Access
# ---------------------------------------------------------------------

def test_copilot_requires_authentication():

    response = client.post(
        "/api/copilot/chat",
        json={
            "question": "Show revenue."
        }
    )

    assert response.status_code in (401, 403)