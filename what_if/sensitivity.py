"""
FinCo AI - What-If Sensitivity Analysis

File:
    backend/app/what_if/sensitivity.py

Purpose:
    Multi-variable sensitivity analysis for financial scenarios.

Architecture:

        What-If Agent
              ↓
        Simulation Service
              ↓
        sensitivity.py
              ↓
        financial_model.py
              ↓
        calculator.py

Capabilities:
    - One-variable sensitivity analysis
    - Two-variable sensitivity analysis
    - Revenue sensitivity
    - Expense sensitivity
    - Profit sensitivity
    - Margin sensitivity
    - Downside analysis
    - Best/worst case detection
    - Loss-threshold detection
    - Impact ranking
    - Scenario matrix generation

This module contains no LLM logic.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Optional

from .financial_model import (
    FinancialInputs,
    FinancialModel,
    ScenarioAssumptions,
    ScenarioComparison,
)


# ============================================================================
# Exceptions
# ============================================================================

class SensitivityAnalysisError(ValueError):
    """Raised when sensitivity analysis inputs are invalid."""


# ============================================================================
# Data Models
# ============================================================================

@dataclass(frozen=True)
class SensitivityPoint:
    """
    Result of one sensitivity test.
    """

    variable: str
    change_pct: float

    revenue: float
    expenses: float
    gross_profit: float
    operating_profit: float
    net_profit: float

    gross_margin_pct: float
    operating_margin_pct: float
    net_margin_pct: float

    ending_cash_balance: float
    debt: float

    profit_change: float
    profit_change_pct: float

    is_loss: bool


@dataclass(frozen=True)
class TwoVariablePoint:
    """
    Result for a two-variable scenario combination.
    """

    revenue_change_pct: float
    expense_change_pct: float

    revenue: float
    expenses: float
    net_profit: float
    net_margin_pct: float

    profit_change: float
    ending_cash_balance: float

    is_loss: bool


@dataclass(frozen=True)
class SensitivitySummary:
    """
    Summary of a sensitivity analysis.
    """

    variable: str

    points: list[SensitivityPoint]

    best_profit: float
    worst_profit: float

    best_change_pct: float
    worst_change_pct: float

    loss_threshold_pct: Optional[float]

    highest_profit_change_pct: float
    lowest_profit_change_pct: float


@dataclass(frozen=True)
class SensitivityImpact:
    """
    Measures how strongly a variable affects profit.
    """

    variable: str

    tested_range_min: float
    tested_range_max: float

    baseline_profit: float

    best_profit: float
    worst_profit: float

    profit_range: float

    sensitivity_score: float

    direction: str


@dataclass(frozen=True)
class ScenarioMatrix:
    """
    Two-dimensional scenario matrix.
    """

    revenue_changes: list[float]
    expense_changes: list[float]

    results: list[TwoVariablePoint]

    best_case: TwoVariablePoint
    worst_case: TwoVariablePoint

    loss_scenarios: int
    total_scenarios: int


# ============================================================================
# Validation Helpers
# ============================================================================

def _validate_percentages(
    values: list[float],
    name: str,
) -> None:
    """Validate a list of percentage changes."""

    if not values:
        raise SensitivityAnalysisError(
            f"{name} cannot be empty."
        )

    for value in values:

        if not isinstance(value, (int, float)):
            raise SensitivityAnalysisError(
                f"{name} must contain numeric values."
            )

        if value < -100:
            raise SensitivityAnalysisError(
                f"{name} cannot contain values below -100%."
            )


def _unique_sorted(
    values: list[float],
) -> list[float]:
    """Return unique sorted percentages."""

    return sorted(
        set(float(value) for value in values)
    )


# ============================================================================
# Sensitivity Analyzer
# ============================================================================

class SensitivityAnalyzer:
    """
    Performs financial sensitivity analysis using FinancialModel.
    """

    def __init__(
        self,
        financial_model: FinancialModel,
    ) -> None:

        if not isinstance(
            financial_model,
            FinancialModel,
        ):
            raise SensitivityAnalysisError(
                "financial_model must be a FinancialModel."
            )

        self.model = financial_model

    # ------------------------------------------------------------------------
    # Revenue Sensitivity
    # ------------------------------------------------------------------------

    def revenue_sensitivity(
        self,
        changes: Optional[list[float]] = None,
    ) -> SensitivitySummary:
        """
        Test the impact of different revenue changes.

        Example:
            [-20, -10, 0, 10, 20]
        """

        changes = changes or [
            -20.0,
            -10.0,
            0.0,
            10.0,
            20.0,
        ]

        _validate_percentages(
            changes,
            "revenue changes",
        )

        points: list[SensitivityPoint] = []

        for change in _unique_sorted(changes):

            comparison = self.model.compare(
                ScenarioAssumptions(
                    revenue_change_pct=change
                )
            )

            points.append(
                self._comparison_to_point(
                    variable="revenue",
                    change_pct=change,
                    comparison=comparison,
                )
            )

        return self._build_summary(
            variable="revenue",
            points=points,
        )

    # ------------------------------------------------------------------------
    # Expense Sensitivity
    # ------------------------------------------------------------------------

    def expense_sensitivity(
        self,
        changes: Optional[list[float]] = None,
    ) -> SensitivitySummary:
        """
        Test the impact of operating expense changes.

        Example:
            [-20, -10, 0, 10, 20]
        """

        changes = changes or [
            -20.0,
            -10.0,
            0.0,
            10.0,
            20.0,
        ]

        _validate_percentages(
            changes,
            "expense changes",
        )

        points: list[SensitivityPoint] = []

        for change in _unique_sorted(changes):

            comparison = self.model.compare(
                ScenarioAssumptions(
                    operating_expense_change_pct=change
                )
            )

            points.append(
                self._comparison_to_point(
                    variable="operating_expenses",
                    change_pct=change,
                    comparison=comparison,
                )
            )

        return self._build_summary(
            variable="operating_expenses",
            points=points,
        )

    # ------------------------------------------------------------------------
    # COGS Sensitivity
    # ------------------------------------------------------------------------

    def cogs_sensitivity(
        self,
        changes: Optional[list[float]] = None,
    ) -> SensitivitySummary:
        """
        Test the impact of cost-of-goods-sold changes.
        """

        changes = changes or [
            -20.0,
            -10.0,
            0.0,
            10.0,
            20.0,
        ]

        _validate_percentages(
            changes,
            "COGS changes",
        )

        points: list[SensitivityPoint] = []

        for change in _unique_sorted(changes):

            comparison = self.model.compare(
                ScenarioAssumptions(
                    cogs_change_pct=change
                )
            )

            points.append(
                self._comparison_to_point(
                    variable="cogs",
                    change_pct=change,
                    comparison=comparison,
                )
            )

        return self._build_summary(
            variable="cogs",
            points=points,
        )

    # ------------------------------------------------------------------------
    # Revenue + Expense Matrix
    # ------------------------------------------------------------------------

    def revenue_expense_matrix(
        self,
        revenue_changes: Optional[list[float]] = None,
        expense_changes: Optional[list[float]] = None,
    ) -> ScenarioMatrix:
        """
        Generate a two-variable sensitivity matrix.

        Example:

            Revenue:
                -20%, -10%, 0%, +10%, +20%

            Expenses:
                -20%, -10%, 0%, +10%, +20%

        Result:
            25 combinations.
        """

        revenue_changes = revenue_changes or [
            -20.0,
            -10.0,
            0.0,
            10.0,
            20.0,
        ]

        expense_changes = expense_changes or [
            -20.0,
            -10.0,
            0.0,
            10.0,
            20.0,
        ]

        _validate_percentages(
            revenue_changes,
            "revenue changes",
        )

        _validate_percentages(
            expense_changes,
            "expense changes",
        )

        revenue_changes = _unique_sorted(
            revenue_changes
        )

        expense_changes = _unique_sorted(
            expense_changes
        )

        results: list[TwoVariablePoint] = []

        for revenue_change, expense_change in product(
            revenue_changes,
            expense_changes,
        ):

            comparison = self.model.compare(
                ScenarioAssumptions(
                    revenue_change_pct=revenue_change,
                    operating_expense_change_pct=(
                        expense_change
                    ),
                )
            )

            results.append(
                TwoVariablePoint(
                    revenue_change_pct=revenue_change,

                    expense_change_pct=expense_change,

                    revenue=(
                        comparison.scenario.revenue
                    ),

                    expenses=(
                        comparison.scenario.operating_expenses
                    ),

                    net_profit=(
                        comparison.scenario.net_profit
                    ),

                    net_margin_pct=(
                        comparison.scenario.net_margin_pct
                    ),

                    profit_change=(
                        comparison.net_profit_change
                    ),

                    ending_cash_balance=(
                        comparison.scenario
                        .ending_cash_balance
                    ),

                    is_loss=(
                        comparison.scenario.net_profit
                        < 0
                    ),
                )
            )

        best_case = max(
            results,
            key=lambda point: point.net_profit,
        )

        worst_case = min(
            results,
            key=lambda point: point.net_profit,
        )

        loss_scenarios = sum(
            point.is_loss
            for point in results
        )

        return ScenarioMatrix(
            revenue_changes=revenue_changes,
            expense_changes=expense_changes,
            results=results,
            best_case=best_case,
            worst_case=worst_case,
            loss_scenarios=loss_scenarios,
            total_scenarios=len(results),
        )

    # ------------------------------------------------------------------------
    # Combined Downside Sensitivity
    # ------------------------------------------------------------------------

    def downside_analysis(
        self,
        revenue_declines: Optional[list[float]] = None,
        expense_increases: Optional[list[float]] = None,
    ) -> ScenarioMatrix:
        """
        Analyze combinations of declining revenue and
        increasing expenses.

        Example:

            Revenue:
                5%, 10%, 15%, 20% decline

            Expenses:
                5%, 10%, 15%, 20% increase
        """

        revenue_declines = revenue_declines or [
            5.0,
            10.0,
            15.0,
            20.0,
        ]

        expense_increases = expense_increases or [
            5.0,
            10.0,
            15.0,
            20.0,
        ]

        revenue_changes = [
            -abs(value)
            for value in revenue_declines
        ]

        expense_changes = [
            abs(value)
            for value in expense_increases
        ]

        return self.revenue_expense_matrix(
            revenue_changes=revenue_changes,
            expense_changes=expense_changes,
        )

    # ------------------------------------------------------------------------
    # Variable Impact Ranking
    # ------------------------------------------------------------------------

    def rank_variable_impact(
        self,
        changes: Optional[list[float]] = None,
    ) -> list[SensitivityImpact]:
        """
        Determine which financial variable has the
        largest effect on profit.
        """

        changes = changes or [
            -20.0,
            -10.0,
            0.0,
            10.0,
            20.0,
        ]

        _validate_percentages(
            changes,
            "changes",
        )

        baseline_profit = (
            self.model.baseline().net_profit
        )

        analyses = [
            self.revenue_sensitivity(changes),
            self.expense_sensitivity(changes),
            self.cogs_sensitivity(changes),
        ]

        impacts: list[SensitivityImpact] = []

        for analysis in analyses:

            profit_range = (
                analysis.best_profit
                - analysis.worst_profit
            )

            # Normalize to baseline profit where possible.
            if baseline_profit != 0:
                sensitivity_score = (
                    abs(profit_range)
                    / abs(baseline_profit)
                ) * 100.0
            else:
                sensitivity_score = abs(
                    profit_range
                )

            if (
                analysis.points[-1].net_profit
                > analysis.points[0].net_profit
            ):
                direction = "positive_with_increase"

            else:
                direction = "negative_with_increase"

            impacts.append(
                SensitivityImpact(
                    variable=analysis.variable,

                    tested_range_min=min(
                        point.change_pct
                        for point in analysis.points
                    ),

                    tested_range_max=max(
                        point.change_pct
                        for point in analysis.points
                    ),

                    baseline_profit=baseline_profit,

                    best_profit=analysis.best_profit,

                    worst_profit=analysis.worst_profit,

                    profit_range=profit_range,

                    sensitivity_score=sensitivity_score,

                    direction=direction,
                )
            )

        impacts.sort(
            key=lambda item: item.sensitivity_score,
            reverse=True,
        )

        return impacts

    # ------------------------------------------------------------------------
    # Loss Threshold
    # ------------------------------------------------------------------------

    def find_revenue_loss_threshold(
        self,
        minimum_decline: float = 0.0,
        maximum_decline: float = 100.0,
        step: float = 1.0,
    ) -> Optional[float]:
        """
        Find the approximate revenue decline at which
        net profit becomes negative.

        Example:
            Returns 17 if profit becomes negative around
            a 17% revenue decline.
        """

        if minimum_decline < 0:
            raise SensitivityAnalysisError(
                "minimum_decline cannot be negative."
            )

        if maximum_decline > 100:
            raise SensitivityAnalysisError(
                "maximum_decline cannot exceed 100%."
            )

        if minimum_decline > maximum_decline:
            raise SensitivityAnalysisError(
                "minimum_decline must not exceed "
                "maximum_decline."
            )

        if step <= 0:
            raise SensitivityAnalysisError(
                "step must be greater than zero."
            )

        decline = minimum_decline

        while decline <= maximum_decline:

            comparison = self.model.compare(
                ScenarioAssumptions(
                    revenue_change_pct=-decline
                )
            )

            if comparison.scenario.net_profit < 0:
                return decline

            decline += step

        return None

    # ------------------------------------------------------------------------
    # Expense Loss Threshold
    # ------------------------------------------------------------------------

    def find_expense_loss_threshold(
        self,
        minimum_increase: float = 0.0,
        maximum_increase: float = 100.0,
        step: float = 1.0,
    ) -> Optional[float]:
        """
        Find the approximate expense increase at which
        net profit becomes negative.
        """

        if minimum_increase < 0:
            raise SensitivityAnalysisError(
                "minimum_increase cannot be negative."
            )

        if maximum_increase > 100:
            raise SensitivityAnalysisError(
                "maximum_increase cannot exceed 100%."
            )

        if minimum_increase > maximum_increase:
            raise SensitivityAnalysisError(
                "minimum_increase must not exceed "
                "maximum_increase."
            )

        if step <= 0:
            raise SensitivityAnalysisError(
                "step must be greater than zero."
            )

        increase = minimum_increase

        while increase <= maximum_increase:

            comparison = self.model.compare(
                ScenarioAssumptions(
                    operating_expense_change_pct=increase
                )
            )

            if comparison.scenario.net_profit < 0:
                return increase

            increase += step

        return None

    # ------------------------------------------------------------------------
    # Margin Sensitivity
    # ------------------------------------------------------------------------

    def margin_sensitivity(
        self,
        revenue_changes: Optional[list[float]] = None,
        expense_changes: Optional[list[float]] = None,
    ) -> list[dict]:
        """
        Analyze the impact of revenue and expense changes
        on net margin.
        """

        matrix = self.revenue_expense_matrix(
            revenue_changes=revenue_changes,
            expense_changes=expense_changes,
        )

        return [
            {
                "revenue_change_pct":
                    point.revenue_change_pct,

                "expense_change_pct":
                    point.expense_change_pct,

                "net_margin_pct":
                    round(
                        point.net_margin_pct,
                        2,
                    ),

                "profit_change":
                    round(
                        point.profit_change,
                        2,
                    ),

                "is_loss":
                    point.is_loss,
            }
            for point in matrix.results
        ]

    # ------------------------------------------------------------------------
    # Best / Worst Cases
    # ------------------------------------------------------------------------

    def best_and_worst_case(
        self,
        revenue_changes: Optional[list[float]] = None,
        expense_changes: Optional[list[float]] = None,
    ) -> dict:
        """
        Return best and worst combinations.
        """

        matrix = self.revenue_expense_matrix(
            revenue_changes=revenue_changes,
            expense_changes=expense_changes,
        )

        return {
            "best_case": asdict(
                matrix.best_case
            ),
            "worst_case": asdict(
                matrix.worst_case
            ),
            "loss_scenarios": matrix.loss_scenarios,
            "total_scenarios": matrix.total_scenarios,
        }

    # ------------------------------------------------------------------------
    # Internal Conversion
    # ------------------------------------------------------------------------

    @staticmethod
    def _comparison_to_point(
        variable: str,
        change_pct: float,
        comparison: ScenarioComparison,
    ) -> SensitivityPoint:
        """Convert ScenarioComparison into a sensitivity point."""

        return SensitivityPoint(
            variable=variable,

            change_pct=change_pct,

            revenue=(
                comparison.scenario.revenue
            ),

            expenses=(
                comparison.scenario
                .operating_expenses
            ),

            gross_profit=(
                comparison.scenario
                .gross_profit
            ),

            operating_profit=(
                comparison.scenario
                .operating_profit
            ),

            net_profit=(
                comparison.scenario
                .net_profit
            ),

            gross_margin_pct=(
                comparison.scenario
                .gross_margin_pct
            ),

            operating_margin_pct=(
                comparison.scenario
                .operating_margin_pct
            ),

            net_margin_pct=(
                comparison.scenario
                .net_margin_pct
            ),

            ending_cash_balance=(
                comparison.scenario
                .ending_cash_balance
            ),

            debt=(
                comparison.scenario.debt
            ),

            profit_change=(
                comparison.net_profit_change
            ),

            profit_change_pct=(
                (
                    comparison.net_profit_change
                    / abs(
                        comparison.baseline.net_profit
                    )
                    * 100.0
                )
                if comparison.baseline.net_profit != 0
                else 0.0
            ),

            is_loss=(
                comparison.scenario.net_profit < 0
            ),
        )

    @staticmethod
    def _build_summary(
        variable: str,
        points: list[SensitivityPoint],
    ) -> SensitivitySummary:
        """Build a sensitivity summary."""

        if not points:
            raise SensitivityAnalysisError(
                "Cannot build summary without points."
            )

        best_point = max(
            points,
            key=lambda point: point.net_profit,
        )

        worst_point = min(
            points,
            key=lambda point: point.net_profit,
        )

        loss_points = [
            point
            for point in points
            if point.is_loss
        ]

        loss_threshold = None

        if loss_points:
            loss_threshold = min(
                point.change_pct
                for point in loss_points
            )

        return SensitivitySummary(
            variable=variable,

            points=points,

            best_profit=best_point.net_profit,

            worst_profit=worst_point.net_profit,

            best_change_pct=best_point.change_pct,

            worst_change_pct=worst_point.change_pct,

            loss_threshold_pct=loss_threshold,

            highest_profit_change_pct=(
                max(
                    point.profit_change_pct
                    for point in points
                )
            ),

            lowest_profit_change_pct=(
                min(
                    point.profit_change_pct
                    for point in points
                )
            ),
        )


# ============================================================================
# Serialization Helpers
# ============================================================================

def sensitivity_summary_to_dict(
    summary: SensitivitySummary,
) -> dict:
    """Convert sensitivity summary into JSON-compatible data."""

    return {
        "variable": summary.variable,

        "points": [
            asdict(point)
            for point in summary.points
        ],

        "best_case": {
            "profit": summary.best_profit,
            "change_pct": summary.best_change_pct,
        },

        "worst_case": {
            "profit": summary.worst_profit,
            "change_pct": summary.worst_change_pct,
        },

        "loss_threshold_pct":
            summary.loss_threshold_pct,

        "profit_change_range": {
            "highest_pct":
                summary.highest_profit_change_pct,

            "lowest_pct":
                summary.lowest_profit_change_pct,
        },
    }


def scenario_matrix_to_dict(
    matrix: ScenarioMatrix,
) -> dict:
    """Convert scenario matrix into JSON-compatible data."""

    return {
        "revenue_changes":
            matrix.revenue_changes,

        "expense_changes":
            matrix.expense_changes,

        "results": [
            asdict(point)
            for point in matrix.results
        ],

        "best_case":
            asdict(matrix.best_case),

        "worst_case":
            asdict(matrix.worst_case),

        "loss_scenarios":
            matrix.loss_scenarios,

        "total_scenarios":
            matrix.total_scenarios,

        "loss_rate_pct": (
            (
                matrix.loss_scenarios
                / matrix.total_scenarios
            )
            * 100.0
            if matrix.total_scenarios
            else 0.0
        ),
    }


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_revenue_sensitivity(
    inputs: FinancialInputs,
    changes: Optional[list[float]] = None,
) -> SensitivitySummary:
    """
    Convenience function for revenue sensitivity.
    """

    model = FinancialModel(inputs)

    analyzer = SensitivityAnalyzer(model)

    return analyzer.revenue_sensitivity(changes)


def analyze_expense_sensitivity(
    inputs: FinancialInputs,
    changes: Optional[list[float]] = None,
) -> SensitivitySummary:
    """
    Convenience function for expense sensitivity.
    """

    model = FinancialModel(inputs)

    analyzer = SensitivityAnalyzer(model)

    return analyzer.expense_sensitivity(changes)


def analyze_downside(
    inputs: FinancialInputs,
    revenue_declines: Optional[list[float]] = None,
    expense_increases: Optional[list[float]] = None,
) -> ScenarioMatrix:
    """
    Convenience function for combined downside analysis.
    """

    model = FinancialModel(inputs)

    analyzer = SensitivityAnalyzer(model)

    return analyzer.downside_analysis(
        revenue_declines=revenue_declines,
        expense_increases=expense_increases,
    )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":

    print("=" * 75)
    print("FinCo AI - Sensitivity Analysis")
    print("=" * 75)

    inputs = FinancialInputs(
        revenue=1_000_000,
        cost_of_goods_sold=550_000,
        operating_expenses=250_000,
        taxes=40_000,
        interest_expense=20_000,
        cash_balance=300_000,
        debt=400_000,
        operating_cash_flow=180_000,
        investing_cash_flow=-50_000,
        financing_cash_flow=-20_000,
    )

    model = FinancialModel(inputs)

    analyzer = SensitivityAnalyzer(model)

    # ------------------------------------------------------------------------
    # Revenue sensitivity
    # ------------------------------------------------------------------------

    print("\nREVENUE SENSITIVITY")
    print("-" * 75)

    revenue_analysis = analyzer.revenue_sensitivity()

    for point in revenue_analysis.points:

        print(
            f"Revenue Change: "
            f"{point.change_pct:+6.1f}% | "
            f"Net Profit: "
            f"${point.net_profit:>12,.2f} | "
            f"Margin: "
            f"{point.net_margin_pct:>7.2f}% | "
            f"Loss: "
            f"{point.is_loss}"
        )

    print(
        f"\nBest Profit: "
        f"${revenue_analysis.best_profit:,.2f}"
    )

    print(
        f"Worst Profit: "
        f"${revenue_analysis.worst_profit:,.2f}"
    )

    print(
        f"Loss Threshold: "
        f"{revenue_analysis.loss_threshold_pct}"
    )

    # ------------------------------------------------------------------------
    # Expense sensitivity
    # ------------------------------------------------------------------------

    print("\nEXPENSE SENSITIVITY")
    print("-" * 75)

    expense_analysis = analyzer.expense_sensitivity()

    for point in expense_analysis.points:

        print(
            f"Expense Change: "
            f"{point.change_pct:+6.1f}% | "
            f"Net Profit: "
            f"${point.net_profit:>12,.2f} | "
            f"Margin: "
            f"{point.net_margin_pct:>7.2f}%"
        )

    # ------------------------------------------------------------------------
    # Two-variable matrix
    # ------------------------------------------------------------------------

    print("\nREVENUE + EXPENSE MATRIX")
    print("-" * 75)

    matrix = analyzer.revenue_expense_matrix(
        revenue_changes=[
            -20,
            -10,
            0,
            10,
            20,
        ],
        expense_changes=[
            -20,
            -10,
            0,
            10,
            20,
        ],
    )

    print(
        f"Total Scenarios: "
        f"{matrix.total_scenarios}"
    )

    print(
        f"Loss Scenarios: "
        f"{matrix.loss_scenarios}"
    )

    print(
        f"Best Case: "
        f"Revenue {matrix.best_case.revenue_change_pct:+.1f}%, "
        f"Expenses {matrix.best_case.expense_change_pct:+.1f}%, "
        f"Profit ${matrix.best_case.net_profit:,.2f}"
    )

    print(
        f"Worst Case: "
        f"Revenue {matrix.worst_case.revenue_change_pct:+.1f}%, "
        f"Expenses {matrix.worst_case.expense_change_pct:+.1f}%, "
        f"Profit ${matrix.worst_case.net_profit:,.2f}"
    )

    # ------------------------------------------------------------------------
    # Variable ranking
    # ------------------------------------------------------------------------

    print("\nVARIABLE IMPACT RANKING")
    print("-" * 75)

    rankings = analyzer.rank_variable_impact()

    for index, impact in enumerate(
        rankings,
        start=1,
    ):

        print(
            f"{index}. "
            f"{impact.variable:25} "
            f"Score: "
            f"{impact.sensitivity_score:.2f}%"
        )

    # ------------------------------------------------------------------------
    # Loss thresholds
    # ------------------------------------------------------------------------

    print("\nLOSS THRESHOLDS")
    print("-" * 75)

    revenue_threshold = (
        analyzer.find_revenue_loss_threshold()
    )

    expense_threshold = (
        analyzer.find_expense_loss_threshold()
    )

    print(
        "Revenue decline threshold: "
        f"{revenue_threshold}%"
    )

    print(
        "Expense increase threshold: "
        f"{expense_threshold}%"
    )