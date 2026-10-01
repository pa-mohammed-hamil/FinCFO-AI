import pytest
from unittest.mock import Mock

from app.what_if.calculator import (
    calculate_profit,
    apply_revenue_change,
    apply_expense_change,
)
from app.what_if.sensitivity import run_sensitivity_analysis
from app.what_if.simulation_service import SimulationService


# ---------------------------------------------------------------------
# Profit Calculation
# ---------------------------------------------------------------------

def test_calculate_profit():
    assert calculate_profit(1_000_000, 700_000) == 300_000


def test_calculate_profit_loss():
    assert calculate_profit(500_000, 650_000) == -150_000


# ---------------------------------------------------------------------
# Revenue Adjustment
# ---------------------------------------------------------------------

def test_apply_revenue_increase():
    updated = apply_revenue_change(1_000_000, 10)

    assert updated == pytest.approx(1_100_000)


def test_apply_revenue_decrease():
    updated = apply_revenue_change(1_000_000, -15)

    assert updated == pytest.approx(850_000)


# ---------------------------------------------------------------------
# Expense Adjustment
# ---------------------------------------------------------------------

def test_apply_expense_increase():
    updated = apply_expense_change(500_000, 20)

    assert updated == pytest.approx(600_000)


def test_apply_expense_decrease():
    updated = apply_expense_change(500_000, -10)

    assert updated == pytest.approx(450_000)


# ---------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------

def test_run_sensitivity_analysis():
    base = {
        "revenue": 1_000_000,
        "expenses": 700_000,
    }

    result = run_sensitivity_analysis(
        base,
        variable="revenue",
        changes=[-10, 0, 10],
    )

    assert len(result) == 3
    assert result[1]["change"] == 0
    assert result[2]["profit"] > result[0]["profit"]


# ---------------------------------------------------------------------
# Simulation Service
# ---------------------------------------------------------------------

def test_simulation_service_runs_scenario():
    calculator = Mock()
    calculator.simulate.return_value = {
        "revenue": 1_100_000,
        "expenses": 700_000,
        "profit": 400_000,
    }

    service = SimulationService(calculator)

    scenario = {
        "revenue_change": 10,
        "expense_change": 0,
    }

    result = service.run(scenario)

    calculator.simulate.assert_called_once()
    assert result["profit"] == 400_000


def test_simulation_service_handles_loss():
    calculator = Mock()
    calculator.simulate.return_value = {
        "revenue": 800_000,
        "expenses": 950_000,
        "profit": -150_000,
    }

    service = SimulationService(calculator)

    result = service.run(
        {"revenue_change": -20, "expense_change": 5}
    )

    assert result["profit"] < 0


# ---------------------------------------------------------------------
# Multiple Scenario Comparison
# ---------------------------------------------------------------------

def test_compare_multiple_scenarios():
    calculator = Mock()

    calculator.simulate.side_effect = [
        {"profit": 300_000},
        {"profit": 350_000},
        {"profit": 250_000},
    ]

    service = SimulationService(calculator)

    scenarios = [
        {"revenue_change": 0},
        {"revenue_change": 10},
        {"revenue_change": -10},
    ]

    results = service.compare(scenarios)

    assert len(results) == 3
    assert results[1]["profit"] > results[0]["profit"]
    assert results[2]["profit"] < results[0]["profit"]


# ---------------------------------------------------------------------
# Executive Decision Scenario
# ---------------------------------------------------------------------

def test_marketing_budget_increase_improves_profit():
    calculator = Mock()

    calculator.simulate.return_value = {
        "revenue": 1_250_000,
        "expenses": 850_000,
        "profit": 400_000,
    }

    service = SimulationService(calculator)

    result = service.run(
        {
            "marketing_increase": 10,
            "expected_revenue_growth": 25,
        }
    )

    assert result["profit"] == 400_000