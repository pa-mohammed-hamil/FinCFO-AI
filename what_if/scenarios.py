"""
FinCo AI - What-If Scenario Definitions

File:
    backend/app/what_if/scenarios.py

Purpose:
    Define, validate, create, and manage reusable financial scenarios.

Architecture:

    Frontend
        ↓
    What-If API
        ↓
    What-If Agent
        ↓
    scenarios.py
        ↓
    financial_model.py
        ↓
    calculator.py
        ↓
    simulation_service.py

Examples:
    - Revenue Growth
    - Revenue Decline
    - Expense Reduction
    - Expense Increase
    - Margin Compression
    - Cost Shock
    - Severe Downside
    - Optimistic Growth
    - Cash Flow Stress
    - Custom Scenario

This module contains no LLM logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


# ============================================================================
# Exceptions
# ============================================================================

class ScenarioError(ValueError):
    """Base exception for scenario-related errors."""


# ============================================================================
# Scenario Types
# ============================================================================

class ScenarioType(str, Enum):
    """Supported FinCo AI scenario types."""

    REVENUE_GROWTH = "revenue_growth"
    REVENUE_DECLINE = "revenue_decline"

    EXPENSE_REDUCTION = "expense_reduction"
    EXPENSE_INCREASE = "expense_increase"

    MARGIN_COMPRESSION = "margin_compression"
    COST_SHOCK = "cost_shock"

    CASH_FLOW_STRESS = "cash_flow_stress"

    OPTIMISTIC = "optimistic"
    BASELINE = "baseline"
    MODERATE_DOWNSIDE = "moderate_downside"
    SEVERE_DOWNSIDE = "severe_downside"

    CUSTOM = "custom"


class ScenarioCategory(str, Enum):
    """High-level scenario categories."""

    GROWTH = "growth"
    DOWNSIDE = "downside"
    COST = "cost"
    LIQUIDITY = "liquidity"
    STRESS = "stress"
    CUSTOM = "custom"


class ScenarioSeverity(str, Enum):
    """Scenario severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ============================================================================
# Scenario Assumptions
# ============================================================================

@dataclass
class ScenarioAssumptions:
    """
    Financial assumptions applied by the scenario.

    Percentages are expressed as human-readable percentages.

    Example:
        revenue_change_pct = -15.0
        expense_change_pct = 8.0
    """

    revenue_change_pct: float = 0.0

    cogs_change_pct: float = 0.0

    operating_expense_change_pct: float = 0.0

    tax_change_pct: float = 0.0

    interest_change_pct: float = 0.0

    operating_cash_flow_change_pct: float = 0.0

    investing_cash_flow_change_pct: float = 0.0

    financing_cash_flow_change_pct: float = 0.0

    debt_change_pct: float = 0.0

    cash_balance_change_pct: float = 0.0

    # Optional operational assumptions
    customer_growth_pct: float = 0.0

    transaction_growth_pct: float = 0.0

    average_order_value_change_pct: float = 0.0

    price_change_pct: float = 0.0

    volume_change_pct: float = 0.0

    def validate(self) -> None:
        """Validate scenario assumptions."""

        values = asdict(self)

        for name, value in values.items():

            if not isinstance(value, (int, float)):
                raise ScenarioError(
                    f"{name} must be numeric."
                )

            if value < -100:
                raise ScenarioError(
                    f"{name} cannot be below -100%."
                )


# ============================================================================
# Scenario Definition
# ============================================================================

@dataclass
class Scenario:
    """
    Complete reusable scenario definition.
    """

    name: str

    scenario_type: ScenarioType

    category: ScenarioCategory

    description: str

    assumptions: ScenarioAssumptions

    severity: ScenarioSeverity = ScenarioSeverity.MEDIUM

    enabled: bool = True

    tags: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        """Validate the complete scenario."""

        if not self.name.strip():
            raise ScenarioError(
                "Scenario name cannot be empty."
            )

        if not self.description.strip():
            raise ScenarioError(
                "Scenario description cannot be empty."
            )

        self.assumptions.validate()

    def to_dict(self) -> dict[str, Any]:
        """Serialize scenario."""

        return {
            "name": self.name,
            "scenario_type": self.scenario_type.value,
            "category": self.category.value,
            "description": self.description,
            "severity": self.severity.value,
            "enabled": self.enabled,
            "tags": self.tags,
            "assumptions": asdict(
                self.assumptions
            ),
            "metadata": self.metadata,
        }


# ============================================================================
# Scenario Factory
# ============================================================================

class ScenarioFactory:
    """
    Creates standard FinCo AI scenarios.
    """

    @staticmethod
    def baseline() -> Scenario:
        """Create a baseline scenario."""

        return Scenario(
            name="Baseline",
            scenario_type=ScenarioType.BASELINE,
            category=ScenarioCategory.GROWTH,
            description=(
                "Current financial state with no changes "
                "to the baseline assumptions."
            ),
            assumptions=ScenarioAssumptions(),
            severity=ScenarioSeverity.LOW,
            tags=[
                "baseline",
                "reference",
            ],
        )

    @staticmethod
    def revenue_growth(
        growth_pct: float = 10.0,
    ) -> Scenario:
        """Create a revenue growth scenario."""

        _validate_non_negative(
            growth_pct,
            "growth_pct",
        )

        return Scenario(
            name=f"Revenue Growth {growth_pct:g}%",
            scenario_type=ScenarioType.REVENUE_GROWTH,
            category=ScenarioCategory.GROWTH,
            description=(
                f"Evaluate the impact of a "
                f"{growth_pct:g}% increase in revenue."
            ),
            assumptions=ScenarioAssumptions(
                revenue_change_pct=growth_pct
            ),
            severity=ScenarioSeverity.LOW,
            tags=[
                "revenue",
                "growth",
                "upside",
            ],
        )

    @staticmethod
    def revenue_decline(
        decline_pct: float = 10.0,
    ) -> Scenario:
        """Create a revenue decline scenario."""

        _validate_non_negative(
            decline_pct,
            "decline_pct",
        )

        return Scenario(
            name=f"Revenue Decline {decline_pct:g}%",
            scenario_type=ScenarioType.REVENUE_DECLINE,
            category=ScenarioCategory.DOWNSIDE,
            description=(
                f"Evaluate the impact of a "
                f"{decline_pct:g}% decrease in revenue."
            ),
            assumptions=ScenarioAssumptions(
                revenue_change_pct=-decline_pct
            ),
            severity=_severity_from_percentage(
                decline_pct
            ),
            tags=[
                "revenue",
                "decline",
                "downside",
            ],
        )

    @staticmethod
    def expense_reduction(
        reduction_pct: float = 10.0,
    ) -> Scenario:
        """Create an operating expense reduction scenario."""

        _validate_non_negative(
            reduction_pct,
            "reduction_pct",
        )

        return Scenario(
            name=f"Expense Reduction {reduction_pct:g}%",
            scenario_type=ScenarioType.EXPENSE_REDUCTION,
            category=ScenarioCategory.COST,
            description=(
                f"Evaluate the impact of reducing "
                f"operating expenses by "
                f"{reduction_pct:g}%."
            ),
            assumptions=ScenarioAssumptions(
                operating_expense_change_pct=(
                    -reduction_pct
                )
            ),
            severity=ScenarioSeverity.LOW,
            tags=[
                "expenses",
                "cost_reduction",
                "efficiency",
            ],
        )

    @staticmethod
    def expense_increase(
        increase_pct: float = 10.0,
    ) -> Scenario:
        """Create an operating expense increase scenario."""

        _validate_non_negative(
            increase_pct,
            "increase_pct",
        )

        return Scenario(
            name=f"Expense Increase {increase_pct:g}%",
            scenario_type=ScenarioType.EXPENSE_INCREASE,
            category=ScenarioCategory.COST,
            description=(
                f"Evaluate the impact of increasing "
                f"operating expenses by "
                f"{increase_pct:g}%."
            ),
            assumptions=ScenarioAssumptions(
                operating_expense_change_pct=increase_pct
            ),
            severity=_severity_from_percentage(
                increase_pct
            ),
            tags=[
                "expenses",
                "cost_increase",
                "downside",
            ],
        )

    @staticmethod
    def margin_compression(
        margin_decline_pct: float = 5.0,
    ) -> Scenario:
        """
        Create a margin compression scenario.

        This primarily models increasing cost pressure.
        """

        _validate_non_negative(
            margin_decline_pct,
            "margin_decline_pct",
        )

        return Scenario(
            name=(
                f"Margin Compression "
                f"{margin_decline_pct:g}%"
            ),
            scenario_type=ScenarioType.MARGIN_COMPRESSION,
            category=ScenarioCategory.DOWNSIDE,
            description=(
                "Evaluate the impact of declining "
                "profitability margins."
            ),
            assumptions=ScenarioAssumptions(
                operating_expense_change_pct=(
                    margin_decline_pct
                )
            ),
            severity=ScenarioSeverity.MEDIUM,
            tags=[
                "margin",
                "profitability",
                "downside",
            ],
        )

    @staticmethod
    def cost_shock(
        cost_increase_pct: float = 20.0,
    ) -> Scenario:
        """
        Create a sudden COGS/cost shock scenario.
        """

        _validate_non_negative(
            cost_increase_pct,
            "cost_increase_pct",
        )

        return Scenario(
            name=f"Cost Shock {cost_increase_pct:g}%",
            scenario_type=ScenarioType.COST_SHOCK,
            category=ScenarioCategory.STRESS,
            description=(
                f"Evaluate the impact of a sudden "
                f"{cost_increase_pct:g}% increase "
                f"in cost of goods sold."
            ),
            assumptions=ScenarioAssumptions(
                cogs_change_pct=cost_increase_pct
            ),
            severity=ScenarioSeverity.HIGH,
            tags=[
                "cogs",
                "cost_shock",
                "stress",
            ],
        )

    @staticmethod
    def cash_flow_stress(
        operating_cash_flow_decline_pct: float = 20.0,
    ) -> Scenario:
        """Create a cash-flow stress scenario."""

        _validate_non_negative(
            operating_cash_flow_decline_pct,
            "operating_cash_flow_decline_pct",
        )

        return Scenario(
            name=(
                "Cash Flow Stress "
                f"{operating_cash_flow_decline_pct:g}%"
            ),
            scenario_type=ScenarioType.CASH_FLOW_STRESS,
            category=ScenarioCategory.LIQUIDITY,
            description=(
                "Evaluate the impact of declining "
                "operating cash flow."
            ),
            assumptions=ScenarioAssumptions(
                operating_cash_flow_change_pct=(
                    -operating_cash_flow_decline_pct
                )
            ),
            severity=ScenarioSeverity.HIGH,
            tags=[
                "cash_flow",
                "liquidity",
                "stress",
            ],
        )

    @staticmethod
    def optimistic() -> Scenario:
        """
        Create an optimistic combined scenario.

        Assumptions:
            Revenue +15%
            COGS -3%
            Operating expenses -5%
        """

        return Scenario(
            name="Optimistic Growth",
            scenario_type=ScenarioType.OPTIMISTIC,
            category=ScenarioCategory.GROWTH,
            description=(
                "Strong growth scenario with higher revenue "
                "and improved cost efficiency."
            ),
            assumptions=ScenarioAssumptions(
                revenue_change_pct=15.0,
                cogs_change_pct=-3.0,
                operating_expense_change_pct=-5.0,
            ),
            severity=ScenarioSeverity.LOW,
            tags=[
                "optimistic",
                "growth",
                "efficiency",
            ],
        )

    @staticmethod
    def moderate_downside() -> Scenario:
        """
        Moderate downside scenario.

        Assumptions:
            Revenue -10%
            COGS +5%
            Operating expenses +5%
        """

        return Scenario(
            name="Moderate Downside",
            scenario_type=ScenarioType.MODERATE_DOWNSIDE,
            category=ScenarioCategory.DOWNSIDE,
            description=(
                "Moderate financial deterioration caused by "
                "lower revenue and higher costs."
            ),
            assumptions=ScenarioAssumptions(
                revenue_change_pct=-10.0,
                cogs_change_pct=5.0,
                operating_expense_change_pct=5.0,
            ),
            severity=ScenarioSeverity.MEDIUM,
            tags=[
                "downside",
                "stress",
                "revenue_decline",
            ],
        )

    @staticmethod
    def severe_downside() -> Scenario:
        """
        Severe downside scenario.

        Assumptions:
            Revenue -20%
            COGS +10%
            Operating expenses +10%
            Operating cash flow -25%
        """

        return Scenario(
            name="Severe Downside",
            scenario_type=ScenarioType.SEVERE_DOWNSIDE,
            category=ScenarioCategory.STRESS,
            description=(
                "Severe financial stress involving "
                "revenue contraction, cost escalation, "
                "and cash-flow deterioration."
            ),
            assumptions=ScenarioAssumptions(
                revenue_change_pct=-20.0,
                cogs_change_pct=10.0,
                operating_expense_change_pct=10.0,
                operating_cash_flow_change_pct=-25.0,
            ),
            severity=ScenarioSeverity.CRITICAL,
            tags=[
                "severe",
                "downside",
                "stress",
                "liquidity",
            ],
        )

    @staticmethod
    def custom(
        name: str,
        assumptions: ScenarioAssumptions,
        description: str = (
            "Custom financial what-if scenario."
        ),
        category: ScenarioCategory = (
            ScenarioCategory.CUSTOM
        ),
        severity: ScenarioSeverity = (
            ScenarioSeverity.MEDIUM
        ),
        tags: Optional[list[str]] = None,
    ) -> Scenario:
        """
        Create a custom user-defined scenario.
        """

        scenario = Scenario(
            name=name,
            scenario_type=ScenarioType.CUSTOM,
            category=category,
            description=description,
            assumptions=assumptions,
            severity=severity,
            tags=tags or ["custom"],
        )

        scenario.validate()

        return scenario


# ============================================================================
# Scenario Registry
# ============================================================================

class ScenarioRegistry:
    """
    In-memory registry of reusable scenarios.

    In production, custom scenarios can additionally be persisted
    through the database Scenario model.
    """

    def __init__(self) -> None:
        self._scenarios: dict[str, Scenario] = {}

    def register(
        self,
        scenario: Scenario,
    ) -> None:
        """Register a scenario."""

        scenario.validate()

        key = self._normalize_key(
            scenario.name
        )

        if key in self._scenarios:
            raise ScenarioError(
                f"Scenario already exists: "
                f"{scenario.name}"
            )

        self._scenarios[key] = scenario

    def upsert(
        self,
        scenario: Scenario,
    ) -> None:
        """Register or replace a scenario."""

        scenario.validate()

        key = self._normalize_key(
            scenario.name
        )

        self._scenarios[key] = scenario

    def get(
        self,
        name: str,
    ) -> Scenario:
        """Retrieve a scenario by name."""

        key = self._normalize_key(name)

        scenario = self._scenarios.get(key)

        if scenario is None:
            raise ScenarioError(
                f"Scenario not found: {name}"
            )

        return scenario

    def exists(
        self,
        name: str,
    ) -> bool:
        """Check whether a scenario exists."""

        return (
            self._normalize_key(name)
            in self._scenarios
        )

    def remove(
        self,
        name: str,
    ) -> None:
        """Remove a scenario."""

        key = self._normalize_key(name)

        if key not in self._scenarios:
            raise ScenarioError(
                f"Scenario not found: {name}"
            )

        del self._scenarios[key]

    def list_all(self) -> list[Scenario]:
        """Return all scenarios."""

        return list(self._scenarios.values())

    def list_enabled(self) -> list[Scenario]:
        """Return enabled scenarios only."""

        return [
            scenario
            for scenario in self._scenarios.values()
            if scenario.enabled
        ]

    def list_by_category(
        self,
        category: ScenarioCategory,
    ) -> list[Scenario]:
        """Return scenarios belonging to a category."""

        return [
            scenario
            for scenario in self._scenarios.values()
            if scenario.category == category
        ]

    def list_by_severity(
        self,
        severity: ScenarioSeverity,
    ) -> list[Scenario]:
        """Return scenarios with a specific severity."""

        return [
            scenario
            for scenario in self._scenarios.values()
            if scenario.severity == severity
        ]

    @staticmethod
    def _normalize_key(
        value: str,
    ) -> str:
        return value.strip().lower()


# ============================================================================
# Default Scenario Registry
# ============================================================================

def create_default_registry() -> ScenarioRegistry:
    """
    Create the default FinCo AI scenario registry.
    """

    registry = ScenarioRegistry()

    default_scenarios = [
        ScenarioFactory.baseline(),

        ScenarioFactory.revenue_growth(
            10
        ),

        ScenarioFactory.revenue_decline(
            10
        ),

        ScenarioFactory.expense_reduction(
            10
        ),

        ScenarioFactory.expense_increase(
            10
        ),

        ScenarioFactory.margin_compression(
            5
        ),

        ScenarioFactory.cost_shock(
            20
        ),

        ScenarioFactory.cash_flow_stress(
            20
        ),

        ScenarioFactory.optimistic(),

        ScenarioFactory.moderate_downside(),

        ScenarioFactory.severe_downside(),
    ]

    for scenario in default_scenarios:
        registry.register(scenario)

    return registry


# ============================================================================
# Scenario Matrix
# ============================================================================

def create_revenue_sensitivity_scenarios(
    percentages: Optional[list[float]] = None,
) -> list[Scenario]:
    """
    Create revenue sensitivity scenarios.

    Default:
        -20%, -10%, 0%, +10%, +20%
    """

    percentages = percentages or [
        -20.0,
        -10.0,
        0.0,
        10.0,
        20.0,
    ]

    scenarios: list[Scenario] = []

    for percentage in percentages:

        if percentage == 0:
            scenarios.append(
                ScenarioFactory.baseline()
            )

        elif percentage > 0:
            scenarios.append(
                ScenarioFactory.revenue_growth(
                    percentage
                )
            )

        else:
            scenarios.append(
                ScenarioFactory.revenue_decline(
                    abs(percentage)
                )
            )

    return scenarios


def create_expense_sensitivity_scenarios(
    percentages: Optional[list[float]] = None,
) -> list[Scenario]:
    """
    Create operating expense sensitivity scenarios.

    Default:
        -20%, -10%, 0%, +10%, +20%
    """

    percentages = percentages or [
        -20.0,
        -10.0,
        0.0,
        10.0,
        20.0,
    ]

    scenarios: list[Scenario] = []

    for percentage in percentages:

        if percentage == 0:
            scenarios.append(
                ScenarioFactory.baseline()
            )

        elif percentage < 0:
            scenarios.append(
                ScenarioFactory.expense_reduction(
                    abs(percentage)
                )
            )

        else:
            scenarios.append(
                ScenarioFactory.expense_increase(
                    percentage
                )
            )

    return scenarios


# ============================================================================
# Scenario Selection
# ============================================================================

def get_scenario(
    scenario_type: ScenarioType,
    value: Optional[float] = None,
) -> Scenario:
    """
    Convenience function for creating a scenario from
    a ScenarioType.
    """

    if scenario_type == ScenarioType.BASELINE:
        return ScenarioFactory.baseline()

    if scenario_type == ScenarioType.REVENUE_GROWTH:
        return ScenarioFactory.revenue_growth(
            value or 10.0
        )

    if scenario_type == ScenarioType.REVENUE_DECLINE:
        return ScenarioFactory.revenue_decline(
            value or 10.0
        )

    if scenario_type == ScenarioType.EXPENSE_REDUCTION:
        return ScenarioFactory.expense_reduction(
            value or 10.0
        )

    if scenario_type == ScenarioType.EXPENSE_INCREASE:
        return ScenarioFactory.expense_increase(
            value or 10.0
        )

    if scenario_type == ScenarioType.MARGIN_COMPRESSION:
        return ScenarioFactory.margin_compression(
            value or 5.0
        )

    if scenario_type == ScenarioType.COST_SHOCK:
        return ScenarioFactory.cost_shock(
            value or 20.0
        )

    if scenario_type == ScenarioType.CASH_FLOW_STRESS:
        return ScenarioFactory.cash_flow_stress(
            value or 20.0
        )

    if scenario_type == ScenarioType.OPTIMISTIC:
        return ScenarioFactory.optimistic()

    if scenario_type == ScenarioType.MODERATE_DOWNSIDE:
        return ScenarioFactory.moderate_downside()

    if scenario_type == ScenarioType.SEVERE_DOWNSIDE:
        return ScenarioFactory.severe_downside()

    raise ScenarioError(
        f"Unsupported scenario type: "
        f"{scenario_type}"
    )


# ============================================================================
# Helpers
# ============================================================================

def _validate_non_negative(
    value: float,
    field_name: str,
) -> None:
    """Validate non-negative percentage input."""

    if not isinstance(value, (int, float)):
        raise ScenarioError(
            f"{field_name} must be numeric."
        )

    if value < 0:
        raise ScenarioError(
            f"{field_name} cannot be negative."
        )

    if value > 100:
        raise ScenarioError(
            f"{field_name} cannot exceed 100%."
        )


def _severity_from_percentage(
    percentage: float,
) -> ScenarioSeverity:
    """Infer severity from scenario magnitude."""

    if percentage >= 30:
        return ScenarioSeverity.CRITICAL

    if percentage >= 20:
        return ScenarioSeverity.HIGH

    if percentage >= 10:
        return ScenarioSeverity.MEDIUM

    return ScenarioSeverity.LOW


# ============================================================================
# JSON-Friendly Export
# ============================================================================

def scenarios_to_dict(
    scenarios: list[Scenario],
) -> list[dict[str, Any]]:
    """Serialize a list of scenarios."""

    return [
        scenario.to_dict()
        for scenario in scenarios
    ]


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FinCo AI - What-If Scenario Engine")
    print("=" * 70)

    registry = create_default_registry()

    print("\nAVAILABLE SCENARIOS")
    print("-" * 70)

    for scenario in registry.list_enabled():

        print(
            f"{scenario.name:30} | "
            f"{scenario.category.value:10} | "
            f"{scenario.severity.value}"
        )

    print("\nCUSTOM SCENARIO")
    print("-" * 70)

    custom = ScenarioFactory.custom(
        name="Market Shock",
        description=(
            "Revenue declines while costs increase "
            "because of adverse market conditions."
        ),
        assumptions=ScenarioAssumptions(
            revenue_change_pct=-15.0,
            cogs_change_pct=8.0,
            operating_expense_change_pct=5.0,
            operating_cash_flow_change_pct=-20.0,
        ),
        category=ScenarioCategory.STRESS,
        severity=ScenarioSeverity.HIGH,
        tags=[
            "market",
            "stress",
            "revenue",
            "cost",
        ],
    )

    registry.register(custom)

    print(custom.to_dict())

    print("\nREVENUE SENSITIVITY")
    print("-" * 70)

    for scenario in create_revenue_sensitivity_scenarios():
        print(
            f"{scenario.name:30} "
            f"{scenario.assumptions.revenue_change_pct:+.1f}%"
        )