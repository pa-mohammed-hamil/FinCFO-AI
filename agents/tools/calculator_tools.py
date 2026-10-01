"""
FinCo AI - Calculator Agent Tools

Path:
    backend/app/agents/tools/calculator_tools.py

Purpose:
    Provides deterministic financial calculation tools for FinCo AI agents.

Architecture:

    Agent
      |
      v
    Tool Registry
      |
      v
    calculator_tools.py
      |
      +--> Financial calculations
      +--> Percentage calculations
      +--> Growth calculations
      +--> Margin calculations
      +--> Ratio calculations
      +--> Break-even calculations
      +--> What-if calculations
      +--> Aggregations
      |
      v
    Structured Result

Design principles:
    - Deterministic calculations.
    - No LLM arithmetic.
    - No database access.
    - No external API calls.
    - Validate inputs before calculation.
    - Avoid division-by-zero.
    - Reject NaN and infinity.
    - Return JSON-friendly dictionaries.
    - Keep business orchestration outside this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence


# ============================================================================
# Exceptions
# ============================================================================


class CalculatorToolError(Exception):
    """Base exception for calculator tools."""


class InvalidCalculationInputError(
    CalculatorToolError
):
    """Raised when calculation input is invalid."""


class DivisionByZeroCalculationError(
    CalculatorToolError
):
    """Raised when a calculation would divide by zero."""


class UnsupportedCalculationError(
    CalculatorToolError
):
    """Raised when an unsupported calculation is requested."""


# ============================================================================
# Enums
# ============================================================================


class CalculationType(str, Enum):
    """Supported calculation categories."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

    PERCENTAGE = "percentage"
    PERCENTAGE_CHANGE = "percentage_change"
    GROWTH_RATE = "growth_rate"

    MARGIN = "margin"
    MARKUP = "markup"

    RATIO = "ratio"
    CURRENT_RATIO = "current_ratio"
    DEBT_TO_EQUITY = "debt_to_equity"

    BREAK_EVEN_UNITS = "break_even_units"
    BREAK_EVEN_REVENUE = "break_even_revenue"

    PROFIT = "profit"
    PROFIT_MARGIN = "profit_margin"

    CASH_RUNWAY = "cash_runway"

    WHAT_IF_REVENUE = "what_if_revenue"
    WHAT_IF_EXPENSE = "what_if_expense"
    WHAT_IF_PROFIT = "what_if_profit"


# ============================================================================
# Result Models
# ============================================================================


@dataclass(slots=True)
class CalculationResult:
    """
    Standardized calculator response.
    """

    calculation: str
    result: float

    inputs: dict[str, Any]

    formula: str

    unit: Optional[str] = None

    metadata: Optional[dict[str, Any]] = None

    calculated_at: Optional[datetime] = None

    def __post_init__(self) -> None:

        if self.metadata is None:
            self.metadata = {}

        if self.calculated_at is None:
            self.calculated_at = (
                datetime.now(timezone.utc)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calculation":
                self.calculation,

            "result":
                self.result,

            "inputs":
                _serialize(self.inputs),

            "formula":
                self.formula,

            "unit":
                self.unit,

            "metadata":
                _serialize(self.metadata),

            "calculated_at":
                self.calculated_at.isoformat(),
        }


# ============================================================================
# Calculator Tools
# ============================================================================


class CalculatorTools:
    """
    Deterministic financial calculator toolkit.

    This class is intended to be registered with the FinCo AI
    agent ToolRegistry.
    """

    # ========================================================================
    # Basic Arithmetic
    # ========================================================================

    def add(
        self,
        *values: float,
    ) -> dict[str, Any]:
        """Add multiple numeric values."""

        numbers = self._validate_values(
            values,
            minimum_count=1,
        )

        result = sum(numbers)

        return self._result(
            calculation="add",
            result=result,
            inputs={
                "values": numbers,
            },
            formula="sum(values)",
        )

    def subtract(
        self,
        minuend: float,
        *subtrahends: float,
    ) -> dict[str, Any]:
        """Subtract one or more values from a starting value."""

        first = self._number(
            minuend,
            "minuend",
        )

        values = self._validate_values(
            subtrahends,
            minimum_count=0,
        )

        result = first

        for value in values:
            result -= value

        return self._result(
            calculation="subtract",
            result=result,
            inputs={
                "minuend": first,
                "subtrahends": values,
            },
            formula="minuend - sum(subtrahends)",
        )

    def multiply(
        self,
        *values: float,
    ) -> dict[str, Any]:
        """Multiply multiple numeric values."""

        numbers = self._validate_values(
            values,
            minimum_count=1,
        )

        result = 1.0

        for value in numbers:
            result *= value

        return self._result(
            calculation="multiply",
            result=result,
            inputs={
                "values": numbers,
            },
            formula="product(values)",
        )

    def divide(
        self,
        numerator: float,
        denominator: float,
    ) -> dict[str, Any]:
        """Divide numerator by denominator."""

        numerator = self._number(
            numerator,
            "numerator",
        )

        denominator = self._number(
            denominator,
            "denominator",
        )

        self._require_nonzero(
            denominator,
            "denominator",
        )

        result = numerator / denominator

        return self._result(
            calculation="divide",
            result=result,
            inputs={
                "numerator": numerator,
                "denominator": denominator,
            },
            formula="numerator / denominator",
        )

    # ========================================================================
    # Percentage Calculations
    # ========================================================================

    def percentage(
        self,
        value: float,
        total: float,
    ) -> dict[str, Any]:
        """
        Calculate what percentage `value` represents of `total`.

        Example:
            value=250
            total=1000
            -> 25%
        """

        value = self._number(
            value,
            "value",
        )

        total = self._number(
            total,
            "total",
        )

        self._require_nonzero(
            total,
            "total",
        )

        result = (
            value / total
        ) * 100.0

        return self._result(
            calculation="percentage",
            result=result,
            inputs={
                "value": value,
                "total": total,
            },
            formula="(value / total) * 100",
            unit="percent",
        )

    def percentage_change(
        self,
        old_value: float,
        new_value: float,
    ) -> dict[str, Any]:
        """
        Calculate percentage change.

        Formula:

            ((new - old) / abs(old)) * 100
        """

        old_value = self._number(
            old_value,
            "old_value",
        )

        new_value = self._number(
            new_value,
            "new_value",
        )

        self._require_nonzero(
            old_value,
            "old_value",
        )

        result = (
            (new_value - old_value)
            / abs(old_value)
        ) * 100.0

        return self._result(
            calculation="percentage_change",
            result=result,
            inputs={
                "old_value": old_value,
                "new_value": new_value,
            },
            formula=(
                "((new_value - old_value) "
                "/ abs(old_value)) * 100"
            ),
            unit="percent",
        )

    def growth_rate(
        self,
        previous_value: float,
        current_value: float,
    ) -> dict[str, Any]:
        """Calculate financial growth rate."""

        result = self.percentage_change(
            old_value=previous_value,
            new_value=current_value,
        )

        result["calculation"] = "growth_rate"

        return result

    # ========================================================================
    # Profitability
    # ========================================================================

    def profit(
        self,
        revenue: float,
        expenses: float,
    ) -> dict[str, Any]:
        """Calculate profit as revenue minus expenses."""

        revenue = self._number(
            revenue,
            "revenue",
        )

        expenses = self._number(
            expenses,
            "expenses",
        )

        result = revenue - expenses

        return self._result(
            calculation="profit",
            result=result,
            inputs={
                "revenue": revenue,
                "expenses": expenses,
            },
            formula="revenue - expenses",
            unit="currency",
        )

    def margin(
        self,
        profit: float,
        revenue: float,
    ) -> dict[str, Any]:
        """
        Calculate profit margin.

        Formula:

            (profit / revenue) * 100
        """

        profit = self._number(
            profit,
            "profit",
        )

        revenue = self._number(
            revenue,
            "revenue",
        )

        self._require_nonzero(
            revenue,
            "revenue",
        )

        result = (
            profit / revenue
        ) * 100.0

        return self._result(
            calculation="margin",
            result=result,
            inputs={
                "profit": profit,
                "revenue": revenue,
            },
            formula="(profit / revenue) * 100",
            unit="percent",
        )

    def profit_margin(
        self,
        revenue: float,
        expenses: float,
    ) -> dict[str, Any]:
        """Calculate profit and profit margin together."""

        revenue = self._number(
            revenue,
            "revenue",
        )

        expenses = self._number(
            expenses,
            "expenses",
        )

        profit = revenue - expenses

        self._require_nonzero(
            revenue,
            "revenue",
        )

        margin = (
            profit / revenue
        ) * 100.0

        return self._result(
            calculation="profit_margin",
            result=margin,
            inputs={
                "revenue": revenue,
                "expenses": expenses,
                "profit": profit,
            },
            formula=(
                "((revenue - expenses) "
                "/ revenue) * 100"
            ),
            unit="percent",
            metadata={
                "profit": profit,
            },
        )

    def markup(
        self,
        selling_price: float,
        cost: float,
    ) -> dict[str, Any]:
        """
        Calculate markup percentage over cost.

        Formula:

            ((selling_price - cost) / cost) * 100
        """

        selling_price = self._number(
            selling_price,
            "selling_price",
        )

        cost = self._number(
            cost,
            "cost",
        )

        self._require_nonzero(
            cost,
            "cost",
        )

        result = (
            (selling_price - cost)
            / abs(cost)
        ) * 100.0

        return self._result(
            calculation="markup",
            result=result,
            inputs={
                "selling_price":
                    selling_price,
                "cost":
                    cost,
            },
            formula=(
                "((selling_price - cost) "
                "/ abs(cost)) * 100"
            ),
            unit="percent",
        )

    # ========================================================================
    # Financial Ratios
    # ========================================================================

    def ratio(
        self,
        numerator: float,
        denominator: float,
    ) -> dict[str, Any]:
        """Calculate a generic financial ratio."""

        numerator = self._number(
            numerator,
            "numerator",
        )

        denominator = self._number(
            denominator,
            "denominator",
        )

        self._require_nonzero(
            denominator,
            "denominator",
        )

        result = (
            numerator / denominator
        )

        return self._result(
            calculation="ratio",
            result=result,
            inputs={
                "numerator": numerator,
                "denominator": denominator,
            },
            formula="numerator / denominator",
            unit="ratio",
        )

    def current_ratio(
        self,
        current_assets: float,
        current_liabilities: float,
    ) -> dict[str, Any]:
        """Calculate current ratio."""

        result = self.ratio(
            numerator=current_assets,
            denominator=current_liabilities,
        )

        result["calculation"] = (
            "current_ratio"
        )

        result["formula"] = (
            "current_assets / "
            "current_liabilities"
        )

        result["inputs"] = {
            "current_assets":
                self._number(
                    current_assets,
                    "current_assets",
                ),

            "current_liabilities":
                self._number(
                    current_liabilities,
                    "current_liabilities",
                ),
        }

        return result

    def debt_to_equity(
        self,
        total_debt: float,
        shareholders_equity: float,
    ) -> dict[str, Any]:
        """Calculate debt-to-equity ratio."""

        result = self.ratio(
            numerator=total_debt,
            denominator=shareholders_equity,
        )

        result["calculation"] = (
            "debt_to_equity"
        )

        result["formula"] = (
            "total_debt / "
            "shareholders_equity"
        )

        result["inputs"] = {
            "total_debt":
                self._number(
                    total_debt,
                    "total_debt",
                ),

            "shareholders_equity":
                self._number(
                    shareholders_equity,
                    "shareholders_equity",
                ),
        }

        return result

    # ========================================================================
    # Break-Even Analysis
    # ========================================================================

    def break_even_units(
        self,
        fixed_costs: float,
        selling_price_per_unit: float,
        variable_cost_per_unit: float,
    ) -> dict[str, Any]:
        """
        Calculate break-even units.

        Formula:

            fixed_costs /
            (selling_price - variable_cost)
        """

        fixed_costs = self._number(
            fixed_costs,
            "fixed_costs",
        )

        selling_price_per_unit = (
            self._number(
                selling_price_per_unit,
                "selling_price_per_unit",
            )
        )

        variable_cost_per_unit = (
            self._number(
                variable_cost_per_unit,
                "variable_cost_per_unit",
            )
        )

        contribution_margin = (
            selling_price_per_unit
            - variable_cost_per_unit
        )

        if contribution_margin <= 0:

            raise InvalidCalculationInputError(
                "Selling price per unit must be "
                "greater than variable cost per unit."
            )

        result = (
            fixed_costs
            / contribution_margin
        )

        return self._result(
            calculation="break_even_units",
            result=result,
            inputs={
                "fixed_costs":
                    fixed_costs,

                "selling_price_per_unit":
                    selling_price_per_unit,

                "variable_cost_per_unit":
                    variable_cost_per_unit,
            },
            formula=(
                "fixed_costs / "
                "(selling_price_per_unit - "
                "variable_cost_per_unit)"
            ),
            unit="units",
        )

    def break_even_revenue(
        self,
        fixed_costs: float,
        contribution_margin_ratio: float,
    ) -> dict[str, Any]:
        """
        Calculate break-even revenue.

        contribution_margin_ratio may be supplied as:
            0.40
        or:
            40
        """

        fixed_costs = self._number(
            fixed_costs,
            "fixed_costs",
        )

        margin_ratio = (
            self._percentage_ratio(
                contribution_margin_ratio
            )
        )

        self._require_positive(
            margin_ratio,
            "contribution_margin_ratio",
        )

        result = (
            fixed_costs
            / margin_ratio
        )

        return self._result(
            calculation="break_even_revenue",
            result=result,
            inputs={
                "fixed_costs":
                    fixed_costs,

                "contribution_margin_ratio":
                    contribution_margin_ratio,
            },
            formula=(
                "fixed_costs / "
                "contribution_margin_ratio"
            ),
            unit="currency",
        )

    # ========================================================================
    # Cash Flow
    # ========================================================================

    def cash_runway(
        self,
        available_cash: float,
        monthly_burn_rate: float,
    ) -> dict[str, Any]:
        """
        Calculate cash runway in months.

        Formula:

            available_cash / monthly_burn_rate
        """

        available_cash = self._number(
            available_cash,
            "available_cash",
        )

        monthly_burn_rate = self._number(
            monthly_burn_rate,
            "monthly_burn_rate",
        )

        self._require_positive(
            monthly_burn_rate,
            "monthly_burn_rate",
        )

        if available_cash < 0:

            raise InvalidCalculationInputError(
                "available_cash cannot be negative."
            )

        result = (
            available_cash
            / monthly_burn_rate
        )

        return self._result(
            calculation="cash_runway",
            result=result,
            inputs={
                "available_cash":
                    available_cash,

                "monthly_burn_rate":
                    monthly_burn_rate,
            },
            formula=(
                "available_cash / "
                "monthly_burn_rate"
            ),
            unit="months",
        )

    # ========================================================================
    # What-If Analysis
    # ========================================================================

    def what_if_revenue(
        self,
        current_revenue: float,
        change_percent: float,
    ) -> dict[str, Any]:
        """
        Calculate revenue under a percentage scenario.

        Example:
            current_revenue = 1,000,000
            change_percent = 10
            projected = 1,100,000
        """

        current_revenue = self._number(
            current_revenue,
            "current_revenue",
        )

        change_percent = self._number(
            change_percent,
            "change_percent",
        )

        projected = (
            current_revenue
            * (1 + change_percent / 100)
        )

        change_amount = (
            projected
            - current_revenue
        )

        return self._result(
            calculation="what_if_revenue",
            result=projected,
            inputs={
                "current_revenue":
                    current_revenue,

                "change_percent":
                    change_percent,
            },
            formula=(
                "current_revenue * "
                "(1 + change_percent / 100)"
            ),
            unit="currency",
            metadata={
                "change_amount":
                    change_amount,
            },
        )

    def what_if_expense(
        self,
        current_expenses: float,
        change_percent: float,
    ) -> dict[str, Any]:
        """
        Calculate expenses under a percentage scenario.

        Negative change means expense reduction.
        """

        current_expenses = self._number(
            current_expenses,
            "current_expenses",
        )

        change_percent = self._number(
            change_percent,
            "change_percent",
        )

        projected = (
            current_expenses
            * (1 + change_percent / 100)
        )

        change_amount = (
            projected
            - current_expenses
        )

        return self._result(
            calculation="what_if_expense",
            result=projected,
            inputs={
                "current_expenses":
                    current_expenses,

                "change_percent":
                    change_percent,
            },
            formula=(
                "current_expenses * "
                "(1 + change_percent / 100)"
            ),
            unit="currency",
            metadata={
                "change_amount":
                    change_amount,
            },
        )

    def what_if_profit(
        self,
        current_revenue: float,
        current_expenses: float,
        revenue_change_percent: float = 0.0,
        expense_change_percent: float = 0.0,
    ) -> dict[str, Any]:
        """
        Calculate projected profit under simultaneous
        revenue and expense changes.
        """

        revenue = self._number(
            current_revenue,
            "current_revenue",
        )

        expenses = self._number(
            current_expenses,
            "current_expenses",
        )

        revenue_change = self._number(
            revenue_change_percent,
            "revenue_change_percent",
        )

        expense_change = self._number(
            expense_change_percent,
            "expense_change_percent",
        )

        projected_revenue = (
            revenue
            * (1 + revenue_change / 100)
        )

        projected_expenses = (
            expenses
            * (1 + expense_change / 100)
        )

        projected_profit = (
            projected_revenue
            - projected_expenses
        )

        current_profit = (
            revenue - expenses
        )

        profit_change = (
            projected_profit
            - current_profit
        )

        return self._result(
            calculation="what_if_profit",
            result=projected_profit,
            inputs={
                "current_revenue":
                    revenue,

                "current_expenses":
                    expenses,

                "revenue_change_percent":
                    revenue_change,

                "expense_change_percent":
                    expense_change,
            },
            formula=(
                "revenue * "
                "(1 + revenue_change / 100) "
                "- expenses * "
                "(1 + expense_change / 100)"
            ),
            unit="currency",
            metadata={
                "current_profit":
                    current_profit,

                "projected_revenue":
                    projected_revenue,

                "projected_expenses":
                    projected_expenses,

                "profit_change":
                    profit_change,
            },
        )

    # ========================================================================
    # Aggregate Calculations
    # ========================================================================

    def total(
        self,
        values: Sequence[float],
    ) -> dict[str, Any]:
        """Calculate total of a sequence."""

        numbers = self._validate_values(
            values,
            minimum_count=1,
        )

        result = sum(numbers)

        return self._result(
            calculation="total",
            result=result,
            inputs={
                "values": numbers,
            },
            formula="sum(values)",
        )

    def average(
        self,
        values: Sequence[float],
    ) -> dict[str, Any]:
        """Calculate arithmetic average."""

        numbers = self._validate_values(
            values,
            minimum_count=1,
        )

        result = (
            sum(numbers)
            / len(numbers)
        )

        return self._result(
            calculation="average",
            result=result,
            inputs={
                "values": numbers,
            },
            formula="sum(values) / count(values)",
        )

    def minimum(
        self,
        values: Sequence[float],
    ) -> dict[str, Any]:
        """Return minimum value."""

        numbers = self._validate_values(
            values,
            minimum_count=1,
        )

        result = min(numbers)

        return self._result(
            calculation="minimum",
            result=result,
            inputs={
                "values": numbers,
            },
            formula="min(values)",
        )

    def maximum(
        self,
        values: Sequence[float],
    ) -> dict[str, Any]:
        """Return maximum value."""

        numbers = self._validate_values(
            values,
            minimum_count=1,
        )

        result = max(numbers)

        return self._result(
            calculation="maximum",
            result=result,
            inputs={
                "values": numbers,
            },
            formula="max(values)",
        )

    # ========================================================================
    # Generic Calculation Dispatcher
    # ========================================================================

    def calculate(
        self,
        calculation: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generic calculation dispatcher.

        Useful when the Agent ToolRegistry receives a calculation
        name dynamically.
        """

        if not isinstance(
            calculation,
            str,
        ) or not calculation.strip():

            raise InvalidCalculationInputError(
                "calculation is required."
            )

        name = (
            calculation
            .strip()
            .lower()
        )

        aliases = {
            "percent":
                "percentage",

            "pct":
                "percentage",

            "growth":
                "growth_rate",

            "profitability_margin":
                "profit_margin",

            "current":
                "current_ratio",

            "d_e":
                "debt_to_equity",

            "break_even":
                "break_even_units",

            "runway":
                "cash_runway",
        }

        name = aliases.get(
            name,
            name,
        )

        method = getattr(
            self,
            name,
            None,
        )

        if method is None:
            raise UnsupportedCalculationError(
                f"Unsupported calculation: {calculation}"
            )

        if not callable(method):
            raise UnsupportedCalculationError(
                f"Invalid calculation handler: {calculation}"
            )

        try:

            return method(
                **kwargs
            )

        except CalculatorToolError:
            raise

        except TypeError as exc:

            raise InvalidCalculationInputError(
                f"Invalid arguments for "
                f"{calculation}: {exc}"
            ) from exc

        except Exception as exc:

            raise CalculatorToolError(
                f"Calculation failed: {exc}"
            ) from exc

    # ========================================================================
    # Internal Validation
    # ========================================================================

    @staticmethod
    def _number(
        value: Any,
        field_name: str,
    ) -> float:

        if isinstance(
            value,
            bool,
        ):

            raise InvalidCalculationInputError(
                f"{field_name} must be numeric."
            )

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidCalculationInputError(
                f"{field_name} must be numeric."
            ) from exc

        if not math.isfinite(
            number
        ):

            raise InvalidCalculationInputError(
                f"{field_name} must be finite."
            )

        return number

    def _validate_values(
        self,
        values: Iterable[Any],
        *,
        minimum_count: int,
    ) -> list[float]:

        numbers = [
            self._number(
                value,
                f"value_{index}",
            )
            for index, value
            in enumerate(values)
        ]

        if len(numbers) < minimum_count:

            raise InvalidCalculationInputError(
                f"At least {minimum_count} "
                "value(s) are required."
            )

        return numbers

    @staticmethod
    def _require_nonzero(
        value: float,
        field_name: str,
    ) -> None:

        if math.isclose(
            value,
            0.0,
            abs_tol=1e-12,
        ):

            raise DivisionByZeroCalculationError(
                f"{field_name} cannot be zero."
            )

    @staticmethod
    def _require_positive(
        value: float,
        field_name: str,
    ) -> None:

        if value <= 0:

            raise InvalidCalculationInputError(
                f"{field_name} must be greater than zero."
            )

    @staticmethod
    def _percentage_ratio(
        value: float,
    ) -> float:
        """
        Normalize either:
            0.40 -> 0.40
        or:
            40 -> 0.40
        """

        number = float(
            value
        )

        if not math.isfinite(
            number
        ):

            raise InvalidCalculationInputError(
                "Percentage ratio must be finite."
            )

        if abs(number) > 1:
            number /= 100.0

        return number

    @staticmethod
    def _result(
        *,
        calculation: str,
        result: float,
        inputs: Mapping[str, Any],
        formula: str,
        unit: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> dict[str, Any]:

        if not math.isfinite(
            float(result)
        ):

            raise CalculatorToolError(
                "Calculation produced a non-finite result."
            )

        calculation_result = (
            CalculationResult(
                calculation=calculation,
                result=round(
                    float(result),
                    10,
                ),
                inputs=dict(inputs),
                formula=formula,
                unit=unit,
                metadata=dict(
                    metadata or {}
                ),
            )
        )

        return calculation_result.to_dict()


# ============================================================================
# Standalone Tool Functions
# ============================================================================


def calculate(
    calculation: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Convenience entry point for ToolRegistry.
    """

    return CalculatorTools().calculate(
        calculation,
        **kwargs,
    )


def calculate_percentage(
    value: float,
    total: float,
) -> dict[str, Any]:
    """Calculate percentage."""

    return CalculatorTools().percentage(
        value,
        total,
    )


def calculate_growth_rate(
    previous_value: float,
    current_value: float,
) -> dict[str, Any]:
    """Calculate growth rate."""

    return CalculatorTools().growth_rate(
        previous_value,
        current_value,
    )


def calculate_profit(
    revenue: float,
    expenses: float,
) -> dict[str, Any]:
    """Calculate profit."""

    return CalculatorTools().profit(
        revenue,
        expenses,
    )


def calculate_margin(
    profit: float,
    revenue: float,
) -> dict[str, Any]:
    """Calculate profit margin."""

    return CalculatorTools().margin(
        profit,
        revenue,
    )


def calculate_current_ratio(
    current_assets: float,
    current_liabilities: float,
) -> dict[str, Any]:
    """Calculate current ratio."""

    return CalculatorTools().current_ratio(
        current_assets,
        current_liabilities,
    )


def calculate_debt_to_equity(
    total_debt: float,
    shareholders_equity: float,
) -> dict[str, Any]:
    """Calculate debt-to-equity ratio."""

    return CalculatorTools().debt_to_equity(
        total_debt,
        shareholders_equity,
    )


def calculate_break_even_units(
    fixed_costs: float,
    selling_price_per_unit: float,
    variable_cost_per_unit: float,
) -> dict[str, Any]:
    """Calculate break-even units."""

    return CalculatorTools().break_even_units(
        fixed_costs,
        selling_price_per_unit,
        variable_cost_per_unit,
    )


def calculate_cash_runway(
    available_cash: float,
    monthly_burn_rate: float,
) -> dict[str, Any]:
    """Calculate cash runway."""

    return CalculatorTools().cash_runway(
        available_cash,
        monthly_burn_rate,
    )


def calculate_what_if_profit(
    current_revenue: float,
    current_expenses: float,
    revenue_change_percent: float = 0.0,
    expense_change_percent: float = 0.0,
) -> dict[str, Any]:
    """Calculate what-if projected profit."""

    return CalculatorTools().what_if_profit(
        current_revenue=current_revenue,
        current_expenses=current_expenses,
        revenue_change_percent=
            revenue_change_percent,
        expense_change_percent=
            expense_change_percent,
    )


# ============================================================================
# Agent Tool Definitions
# ============================================================================


CALCULATOR_TOOL_DEFINITIONS = [
    {
        "name":
            "calculate",

        "description":
            (
                "Perform a deterministic financial calculation "
                "such as percentage, growth, margin, ratio, "
                "profit, break-even, runway or what-if analysis."
            ),

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "calculate_percentage",

        "description":
            "Calculate what percentage one value represents of another.",

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "calculate_growth_rate",

        "description":
            "Calculate percentage growth between two periods.",

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "calculate_profit",

        "description":
            "Calculate profit from revenue and expenses.",

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "calculate_margin",

        "description":
            "Calculate profit margin as a percentage.",

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "calculate_current_ratio",

        "description":
            "Calculate the current ratio.",

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "calculate_debt_to_equity",

        "description":
            "Calculate the debt-to-equity ratio.",

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "calculate_break_even_units",

        "description":
            "Calculate required units to reach break-even.",

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "calculate_cash_runway",

        "description":
            "Calculate available cash runway in months.",

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
    {
        "name":
            "calculate_what_if_profit",

        "description":
            (
                "Calculate projected profit under revenue "
                "and expense percentage changes."
            ),

        "category":
            "calculator",

        "requires_confirmation":
            False,
    },
]


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "CalculatorToolError",
    "InvalidCalculationInputError",
    "DivisionByZeroCalculationError",
    "UnsupportedCalculationError",

    # Enums
    "CalculationType",

    # Models
    "CalculationResult",

    # Main toolkit
    "CalculatorTools",

    # Convenience functions
    "calculate",
    "calculate_percentage",
    "calculate_growth_rate",
    "calculate_profit",
    "calculate_margin",
    "calculate_current_ratio",
    "calculate_debt_to_equity",
    "calculate_break_even_units",
    "calculate_cash_runway",
    "calculate_what_if_profit",

    # Registry metadata
    "CALCULATOR_TOOL_DEFINITIONS",
]


# ============================================================================
# Serialization Helper
# ============================================================================


def _serialize(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        Enum,
    ):
        return value.value

    if isinstance(
        value,
        datetime,
    ):
        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(key):
                _serialize(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return [
            _serialize(item)
            for item in value
        ]

    if isinstance(
       value,
        float,
    ):

        if not math.isfinite(
            value
        ):
            return None

        return value

    return value