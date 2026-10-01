import pytest

from app.financial.calculator import (
    gross_profit,
    gross_margin,
    net_profit,
    net_margin,
    operating_margin,
    current_ratio,
    debt_to_equity_ratio,
    revenue_growth_rate,
)


# ---------- Gross Profit ----------

def test_gross_profit():
    assert gross_profit(1_000_000, 400_000) == 600_000


def test_gross_profit_zero_revenue():
    assert gross_profit(0, 0) == 0


# ---------- Gross Margin ----------

def test_gross_margin():
    assert gross_margin(1_000_000, 400_000) == pytest.approx(60.0)


def test_gross_margin_zero_revenue():
    assert gross_margin(0, 1000) == 0


# ---------- Net Profit ----------

def test_net_profit():
    assert net_profit(1_000_000, 800_000) == 200_000


# ---------- Net Margin ----------

def test_net_margin():
    assert net_margin(1_000_000, 200_000) == pytest.approx(20.0)


def test_net_margin_zero_revenue():
    assert net_margin(0, 1000) == 0


# ---------- Operating Margin ----------

def test_operating_margin():
    assert operating_margin(1_000_000, 150_000) == pytest.approx(15.0)


# ---------- Current Ratio ----------

def test_current_ratio():
    assert current_ratio(500_000, 250_000) == pytest.approx(2.0)


def test_current_ratio_zero_liabilities():
    assert current_ratio(500_000, 0) == float("inf")


# ---------- Debt to Equity ----------

def test_debt_to_equity_ratio():
    assert debt_to_equity_ratio(600_000, 300_000) == pytest.approx(2.0)


def test_debt_to_equity_zero_equity():
    assert debt_to_equity_ratio(100_000, 0) == float("inf")


# ---------- Revenue Growth ----------

def test_revenue_growth_positive():
    assert revenue_growth_rate(120_000, 100_000) == pytest.approx(20.0)


def test_revenue_growth_negative():
    assert revenue_growth_rate(80_000, 100_000) == pytest.approx(-20.0)


def test_revenue_growth_zero_previous():
    assert revenue_growth_rate(100_000, 0) == 0