"""
FinCo AI - What-If Simulation Service

File:
    backend/app/what_if/simulation_service.py

Purpose:
    Orchestrates complete financial what-if simulations.

Architecture:

    What-If Agent
          ↓
    SimulationService
          ↓
    ┌─────────────────────────────┐
    │ Scenario Registry            │
    │ Financial Model              │
    │ Sensitivity Analyzer         │
    │ Deterministic Calculations   │
    └─────────────────────────────┘
          ↓
    Simulation Results
          ↓
    Recommendation / Risk / Agent
          ↓
    Human Review
          ↓
    Audit Log

Design principles:
    - Deterministic calculations
    - No LLM dependency
    - No financial decisions invented by AI
    - Structured results for agents and APIs
    - Supports single, batch, sensitivity and
      downside simulations
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from .financial_model import (
    FinancialInputs,
    FinancialModel,
    FinancialModelError,
    FinancialMetrics,
    ScenarioAssumptions,
    ScenarioComparison,
)

from .scenarios import (
    Scenario,
    ScenarioCategory,
    ScenarioSeverity,
    ScenarioRegistry,
    create_default_registry,
)

from .sensitivity import (
    SensitivityAnalyzer,
    SensitivityImpact,
    SensitivitySummary,
    ScenarioMatrix,
)


# ============================================================================
# Exceptions
# ============================================================================

class SimulationServiceError(Exception):
    """Base exception for simulation service errors."""


class SimulationValidationError(
    SimulationServiceError
):
    """Raised when simulation inputs are invalid."""


class SimulationNotFoundError(
    SimulationServiceError
):
    """Raised when a requested simulation cannot be found."""


# ============================================================================
# Enums
# ============================================================================

class SimulationStatus(str, Enum):
    """Lifecycle state of a simulation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RiskLevel(str, Enum):
    """Overall financial risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SimulationType(str, Enum):
    """Types of simulation supported."""

    SINGLE = "single"
    BATCH = "batch"
    SENSITIVITY = "sensitivity"
    DOWNSIDE = "downside"
    STRESS_TEST = "stress_test"
    CUSTOM = "custom"


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class SimulationConfig:
    """
    Configuration controlling simulation behavior.
    """

    loss_risk_threshold: float = 0.0

    critical_margin_pct: float = 5.0

    high_margin_pct: float = 10.0

    critical_cash_balance: float = 0.0

    high_debt_to_revenue_pct: float = 75.0

    critical_debt_to_revenue_pct: float = 100.0

    max_batch_size: int = 100

    default_sensitivity_range: tuple[
        float, ...
    ] = (
        -20.0,
        -10.0,
        0.0,
        10.0,
        20.0,
    )


# ============================================================================
# Result Models
# ============================================================================

@dataclass(frozen=True)
class RiskAssessment:
    """
    Deterministic risk assessment for a simulation.
    """

    level: RiskLevel

    score: float

    reasons: list[str]

    loss_risk: bool

    negative_cash_risk: bool

    margin_risk: bool

    leverage_risk: bool


@dataclass(frozen=True)
class SimulationResult:
    """
    Complete result of one scenario simulation.
    """

    simulation_id: str

    scenario_id: Optional[str]

    scenario_name: str

    simulation_type: SimulationType

    status: SimulationStatus

    created_at: str

    baseline: FinancialMetrics

    scenario: FinancialMetrics

    comparison: ScenarioComparison

    risk: RiskAssessment

    key_findings: list[str]

    assumptions: dict[str, Any]

    metadata: dict[str, Any]


@dataclass(frozen=True)
class BatchSimulationResult:
    """
    Results from multiple scenarios.
    """

    batch_id: str

    simulation_type: SimulationType

    status: SimulationStatus

    created_at: str

    results: list[SimulationResult]

    best_case_simulation_id: Optional[str]

    worst_case_simulation_id: Optional[str]

    profitable_scenarios: int

    loss_scenarios: int

    total_scenarios: int


@dataclass(frozen=True)
class StressTestResult:
    """
    Complete stress-test output.
    """

    simulation_id: str

    status: SimulationStatus

    created_at: str

    downside_matrix: ScenarioMatrix

    worst_case: dict[str, Any]

    best_case: dict[str, Any]

    loss_scenario_count: int

    total_scenario_count: int

    loss_rate_pct: float

    risk_level: RiskLevel


@dataclass(frozen=True)
class SensitivitySimulationResult:
    """
    Complete sensitivity-analysis output.
    """

    simulation_id: str

    status: SimulationStatus

    created_at: str

    revenue_sensitivity: Optional[
        SensitivitySummary
    ]

    expense_sensitivity: Optional[
        SensitivitySummary
    ]

    cogs_sensitivity: Optional[
        SensitivitySummary
    ]

    variable_ranking: list[
        SensitivityImpact
    ]

    revenue_loss_threshold: Optional[float]

    expense_loss_threshold: Optional[float]


# ============================================================================
# Simulation Service
# ============================================================================

class SimulationService:
    """
    Central orchestration service for FinCo AI What-If analysis.

    Responsibilities:
        1. Validate inputs.
        2. Execute financial scenarios.
        3. Run sensitivity analysis.
        4. Perform stress testing.
        5. Calculate deterministic risk.
        6. Generate structured findings.
        7. Return agent/API-friendly results.

    It intentionally does NOT:
        - generate natural-language financial advice using an LLM
        - approve transactions
        - execute financial trades
        - make autonomous financial decisions
    """

    def __init__(
        self,
        financial_inputs: FinancialInputs,
        scenario_registry: Optional[
            ScenarioRegistry
        ] = None,
        config: Optional[SimulationConfig] = None,
    ) -> None:

        if not isinstance(
            financial_inputs,
            FinancialInputs,
        ):
            raise SimulationValidationError(
                "financial_inputs must be "
                "FinancialInputs."
            )

        self.financial_inputs = financial_inputs

        self.model = FinancialModel(
            financial_inputs
        )

        self.registry = (
            scenario_registry
            or create_default_registry()
        )

        self.config = (
            config
            or SimulationConfig()
        )

        self.sensitivity = SensitivityAnalyzer(
            self.model
        )

        self._history: dict[
            str,
            SimulationResult
        ] = {}

    # ========================================================================
    # Single Simulation
    # ========================================================================

    def run_scenario(
        self,
        scenario: Scenario,
        simulation_type: SimulationType = (
            SimulationType.SINGLE
        ),
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> SimulationResult:
        """
        Execute one scenario.
        """

        if not isinstance(
            scenario,
            Scenario,
        ):
            raise SimulationValidationError(
                "scenario must be a Scenario."
            )

        if not scenario.enabled:
            raise SimulationValidationError(
                f"Scenario '{scenario.name}' "
                "is disabled."
            )

        simulation_id = self._generate_id()

        created_at = self._now()

        comparison = self.model.compare(
            scenario.assumptions
        )

        risk = self._assess_risk(
            comparison
        )

        findings = self._generate_findings(
            comparison,
            risk,
        )

        result = SimulationResult(
            simulation_id=simulation_id,

            scenario_id=(
                str(scenario.metadata.get("id"))
                if scenario.metadata.get("id")
                else None
            ),

            scenario_name=scenario.name,

            simulation_type=simulation_type,

            status=SimulationStatus.COMPLETED,

            created_at=created_at,

            baseline=comparison.baseline,

            scenario=comparison.scenario,

            comparison=comparison,

            risk=risk,

            key_findings=findings,

            assumptions=asdict(
                scenario.assumptions
            ),

            metadata={
                **scenario.metadata,
                **(
                    metadata
                    or {}
                ),
            },
        )

        self._history[
            simulation_id
        ] = result

        return result

    # ========================================================================
    # Scenario By Registry Name
    # ========================================================================

    def run_registered_scenario(
        self,
        scenario_name: str,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> SimulationResult:
        """
        Run a scenario registered in ScenarioRegistry.
        """

        if not scenario_name:
            raise SimulationValidationError(
                "scenario_name is required."
            )

        scenario = self.registry.get(
            scenario_name
        )

        if scenario is None:
            raise SimulationValidationError(
                f"Scenario '{scenario_name}' "
                "was not found."
            )

        return self.run_scenario(
            scenario,
            metadata=metadata,
        )

    # ========================================================================
    # Batch Simulation
    # ========================================================================

    def run_batch(
        self,
        scenarios: list[Scenario],
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> BatchSimulationResult:
        """
        Run multiple scenarios.
        """

        if not scenarios:
            raise SimulationValidationError(
                "scenarios cannot be empty."
            )

        if len(scenarios) > self.config.max_batch_size:
            raise SimulationValidationError(
                "Batch size exceeds configured "
                f"maximum of "
                f"{self.config.max_batch_size}."
            )

        batch_id = self._generate_id()

        results: list[
            SimulationResult
        ] = []

        for scenario in scenarios:

            result = self.run_scenario(
                scenario,
                simulation_type=(
                    SimulationType.BATCH
                ),
                metadata=metadata,
            )

            results.append(result)

        best_case = max(
            results,
            key=lambda result:
            result.scenario.net_profit,
        )

        worst_case = min(
            results,
            key=lambda result:
            result.scenario.net_profit,
        )

        profitable = sum(
            result.scenario.net_profit >= 0
            for result in results
        )

        losses = len(results) - profitable

        return BatchSimulationResult(
            batch_id=batch_id,

            simulation_type=(
                SimulationType.BATCH
            ),

            status=SimulationStatus.COMPLETED,

            created_at=self._now(),

            results=results,

            best_case_simulation_id=(
                best_case.simulation_id
            ),

            worst_case_simulation_id=(
                worst_case.simulation_id
            ),

            profitable_scenarios=profitable,

            loss_scenarios=losses,

            total_scenarios=len(results),
        )

    # ========================================================================
    # Default Scenario Portfolio
    # ========================================================================

    def run_default_scenarios(
        self,
    ) -> BatchSimulationResult:
        """
        Run the enabled default scenarios from
        ScenarioRegistry.
        """

        scenarios = (
            self.registry.list_enabled()
        )

        if not scenarios:
            raise SimulationValidationError(
                "No enabled scenarios "
                "are registered."
            )

        return self.run_batch(
            scenarios
        )

    # ========================================================================
    # Revenue Sensitivity
    # ========================================================================

    def run_revenue_sensitivity(
        self,
        changes: Optional[
            list[float]
        ] = None,
    ) -> SensitivitySimulationResult:
        """
        Run revenue sensitivity analysis.
        """

        simulation_id = self._generate_id()

        changes = (
            changes
            or list(
                self.config
                .default_sensitivity_range
            )
        )

        revenue = (
            self.sensitivity
            .revenue_sensitivity(changes)
        )

        ranking = (
            self.sensitivity
            .rank_variable_impact(changes)
        )

        revenue_threshold = (
            self.sensitivity
            .find_revenue_loss_threshold()
        )

        expense_threshold = (
            self.sensitivity
            .find_expense_loss_threshold()
        )

        return SensitivitySimulationResult(
            simulation_id=simulation_id,

            status=SimulationStatus.COMPLETED,

            created_at=self._now(),

            revenue_sensitivity=revenue,

            expense_sensitivity=None,

            cogs_sensitivity=None,

            variable_ranking=ranking,

            revenue_loss_threshold=(
                revenue_threshold
            ),

            expense_loss_threshold=(
                expense_threshold
            ),
        )

    # ========================================================================
    # Full Sensitivity Analysis
    # ========================================================================

    def run_full_sensitivity(
        self,
        changes: Optional[
            list[float]
        ] = None,
    ) -> SensitivitySimulationResult:
        """
        Run revenue, expense and COGS sensitivity
        plus variable-impact ranking.
        """

        simulation_id = self._generate_id()

        changes = (
            changes
            or list(
                self.config
                .default_sensitivity_range
            )
        )

        revenue = (
            self.sensitivity
            .revenue_sensitivity(changes)
        )

        expense = (
            self.sensitivity
            .expense_sensitivity(changes)
        )

        cogs = (
            self.sensitivity
            .cogs_sensitivity(changes)
        )

        ranking = (
            self.sensitivity
            .rank_variable_impact(changes)
        )

        revenue_threshold = (
            self.sensitivity
            .find_revenue_loss_threshold()
        )

        expense_threshold = (
            self.sensitivity
            .find_expense_loss_threshold()
        )

        return SensitivitySimulationResult(
            simulation_id=simulation_id,

            status=SimulationStatus.COMPLETED,

            created_at=self._now(),

            revenue_sensitivity=revenue,

            expense_sensitivity=expense,

            cogs_sensitivity=cogs,

            variable_ranking=ranking,

            revenue_loss_threshold=(
                revenue_threshold
            ),

            expense_loss_threshold=(
                expense_threshold
            ),
        )

    # ========================================================================
    # Stress Testing
    # ========================================================================

    def run_stress_test(
        self,
        revenue_declines: Optional[
            list[float]
        ] = None,
        expense_increases: Optional[
            list[float]
        ] = None,
    ) -> StressTestResult:
        """
        Run combined revenue-downside and
        expense-upside stress testing.
        """

        simulation_id = self._generate_id()

        matrix = (
            self.sensitivity
            .downside_analysis(
                revenue_declines=(
                    revenue_declines
                ),
                expense_increases=(
                    expense_increases
                ),
            )
        )

        risk_level = (
            self._assess_stress_risk(
                matrix
            )
        )

        loss_rate = (
            (
                matrix.loss_scenarios
                / matrix.total_scenarios
            )
            * 100.0
            if matrix.total_scenarios
            else 0.0
        )

        return StressTestResult(
            simulation_id=simulation_id,

            status=SimulationStatus.COMPLETED,

            created_at=self._now(),

            downside_matrix=matrix,

            worst_case=asdict(
                matrix.worst_case
            ),

            best_case=asdict(
                matrix.best_case
            ),

            loss_scenario_count=(
                matrix.loss_scenarios
            ),

            total_scenario_count=(
                matrix.total_scenarios
            ),

            loss_rate_pct=loss_rate,

            risk_level=risk_level,
        )

    # ========================================================================
    # Custom Simulation
    # ========================================================================

    def run_custom(
        self,
        name: str,
        assumptions: ScenarioAssumptions,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> SimulationResult:
        """
        Run a custom scenario without registering it.
        """

        if not name.strip():
            raise SimulationValidationError(
                "Scenario name cannot be empty."
            )

        if not isinstance(
            assumptions,
            ScenarioAssumptions,
        ):
            raise SimulationValidationError(
                "assumptions must be "
                "ScenarioAssumptions."
            )

        scenario = Scenario(
            name=name,
            assumptions=assumptions,
            category=ScenarioCategory.CUSTOM,
            severity=ScenarioSeverity.MEDIUM,
            metadata=metadata or {},
        )

        return self.run_scenario(
            scenario,
            simulation_type=(
                SimulationType.CUSTOM
            ),
        )

    # ========================================================================
    # Scenario Comparison
    # ========================================================================

    def compare_scenarios(
        self,
        scenarios: list[Scenario],
    ) -> dict[str, Any]:
        """
        Compare multiple scenarios and rank them
        by net profit.
        """

        batch = self.run_batch(
            scenarios
        )

        ranked = sorted(
            batch.results,
            key=lambda result:
            result.scenario.net_profit,
            reverse=True,
        )

        return {
            "batch_id": batch.batch_id,

            "ranking": [
                {
                    "rank": index,
                    "simulation_id":
                        result.simulation_id,
                    "scenario":
                        result.scenario_name,
                    "net_profit":
                        result.scenario.net_profit,
                    "net_margin_pct":
                        result.scenario.net_margin_pct,
                    "risk_level":
                        result.risk.level.value,
                }
                for index, result
                in enumerate(
                    ranked,
                    start=1,
                )
            ],

            "best_case":
                batch.best_case_simulation_id,

            "worst_case":
                batch.worst_case_simulation_id,
        }

    # ========================================================================
    # Retrieve Simulation
    # ========================================================================

    def get_simulation(
        self,
        simulation_id: str,
    ) -> SimulationResult:
        """
        Retrieve a previously completed simulation.
        """

        result = self._history.get(
            simulation_id
        )

        if result is None:
            raise SimulationNotFoundError(
                f"Simulation '{simulation_id}' "
                "was not found."
            )

        return result

    # ========================================================================
    # History
    # ========================================================================

    def list_history(
        self,
        limit: int = 50,
    ) -> list[SimulationResult]:
        """
        Return recent simulation results.
        """

        if limit <= 0:
            raise SimulationValidationError(
                "limit must be greater than zero."
            )

        results = list(
            self._history.values()
        )

        results.sort(
            key=lambda result:
            result.created_at,
            reverse=True,
        )

        return results[:limit]

    # ========================================================================
    # Risk Assessment
    # ========================================================================

    def _assess_risk(
        self,
        comparison: ScenarioComparison,
    ) -> RiskAssessment:
        """
        Deterministically classify scenario risk.
        """

        scenario = comparison.scenario

        reasons: list[str] = []

        score = 0.0

        loss_risk = (
            scenario.net_profit
            < self.config.loss_risk_threshold
        )

        negative_cash_risk = (
            scenario.ending_cash_balance
            < self.config.critical_cash_balance
        )

        margin_risk = (
            scenario.net_margin_pct
            < self.config.high_margin_pct
        )

        leverage_ratio = (
            (
                scenario.debt
                / scenario.revenue
            )
            * 100.0
            if scenario.revenue > 0
            else 100.0
        )

        leverage_risk = (
            leverage_ratio
            >= self.config.high_debt_to_revenue_pct
        )

        # ------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------

        if loss_risk:

            score += 40

            reasons.append(
                "Scenario produces a net loss."
            )

        # ------------------------------------------------------------
        # Negative cash
        # ------------------------------------------------------------

        if negative_cash_risk:

            score += 30

            reasons.append(
                "Scenario produces negative "
                "ending cash."
            )

        # ------------------------------------------------------------
        # Margin pressure
        # ------------------------------------------------------------

        if (
            scenario.net_margin_pct
            < self.config.critical_margin_pct
        ):

            score += 20

            reasons.append(
                "Net margin falls below the "
                "critical margin threshold."
            )

        elif margin_risk:

            score += 10

            reasons.append(
                "Net margin shows material "
                "pressure."
            )

        # ------------------------------------------------------------
        # Leverage
        # ------------------------------------------------------------

        if (
            leverage_ratio
            >= self.config.critical_debt_to_revenue_pct
        ):

            score += 20

            reasons.append(
                "Debt exceeds revenue."
            )

        elif leverage_risk:

            score += 10

            reasons.append(
                "Debt-to-revenue ratio is elevated."
            )

        # ------------------------------------------------------------
        # Profit deterioration
        # ------------------------------------------------------------

        if (
            comparison.net_profit_change
            < 0
        ):

            score += 5

            reasons.append(
                "Net profit deteriorates versus "
                "baseline."
            )

        # ------------------------------------------------------------
        # Final level
        # ------------------------------------------------------------

        if score >= 70:

            level = RiskLevel.CRITICAL

        elif score >= 45:

            level = RiskLevel.HIGH

        elif score >= 20:

            level = RiskLevel.MEDIUM

        else:

            level = RiskLevel.LOW

        return RiskAssessment(
            level=level,

            score=min(
                score,
                100.0,
            ),

            reasons=reasons,

            loss_risk=loss_risk,

            negative_cash_risk=(
                negative_cash_risk
            ),

            margin_risk=margin_risk,

            leverage_risk=leverage_risk,
        )

    # ========================================================================
    # Stress Risk
    # ========================================================================

    def _assess_stress_risk(
        self,
        matrix: ScenarioMatrix,
    ) -> RiskLevel:
        """
        Determine overall stress-test risk.
        """

        if matrix.total_scenarios == 0:
            return RiskLevel.LOW

        loss_rate = (
            matrix.loss_scenarios
            / matrix.total_scenarios
        )

        worst_profit = (
            matrix.worst_case.net_profit
        )

        if (
            worst_profit < 0
            and loss_rate >= 0.50
        ):
            return RiskLevel.CRITICAL

        if (
            worst_profit < 0
            or loss_rate >= 0.25
        ):
            return RiskLevel.HIGH

        if loss_rate > 0:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    # ========================================================================
    # Findings
    # ========================================================================

    def _generate_findings(
        self,
        comparison: ScenarioComparison,
        risk: RiskAssessment,
    ) -> list[str]:
        """
        Generate deterministic findings.

        These are factual observations, not AI-generated
        recommendations.
        """

        findings: list[str] = []

        baseline = comparison.baseline
        scenario = comparison.scenario

        # ------------------------------------------------------------
        # Revenue
        # ------------------------------------------------------------

        if (
            scenario.revenue
            < baseline.revenue
        ):

            change = (
                (
                    scenario.revenue
                    - baseline.revenue
                )
                / baseline.revenue
                * 100.0
                if baseline.revenue
                else 0.0
            )

            findings.append(
                "Revenue decreases by "
                f"{abs(change):.2f}% "
                "versus baseline."
            )

        elif (
            scenario.revenue
            > baseline.revenue
        ):

            change = (
                (
                    scenario.revenue
                    - baseline.revenue
                )
                / baseline.revenue
                * 100.0
                if baseline.revenue
                else 0.0
            )

            findings.append(
                "Revenue increases by "
                f"{change:.2f}% "
                "versus baseline."
            )

        # ------------------------------------------------------------
        # Profit
        # ------------------------------------------------------------

        if (
            scenario.net_profit
            < baseline.net_profit
        ):

            findings.append(
                "Net profit deteriorates "
                "versus baseline."
            )

        elif (
            scenario.net_profit
            > baseline.net_profit
        ):

            findings.append(
                "Net profit improves "
                "versus baseline."
            )

        # ------------------------------------------------------------
        # Margin
        # ------------------------------------------------------------

        if (
            scenario.net_margin_pct
            < baseline.net_margin_pct
        ):

            findings.append(
                "Net margin contracts "
                "under this scenario."
            )

        elif (
            scenario.net_margin_pct
            > baseline.net_margin_pct
        ):

            findings.append(
                "Net margin improves "
                "under this scenario."
            )

        # ------------------------------------------------------------
        # Cash
        # ------------------------------------------------------------

        if (
            scenario.ending_cash_balance
            < baseline.ending_cash_balance
        ):

            findings.append(
                "Ending cash balance decreases."
            )

        if risk.negative_cash_risk:

            findings.append(
                "Cash balance becomes negative."
            )

        # ------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------

        if risk.loss_risk:

            findings.append(
                "The scenario results in a "
                "negative net profit."
            )

        # ------------------------------------------------------------
        # Risk
        # ------------------------------------------------------------

        findings.append(
            f"Overall deterministic risk level: "
            f"{risk.level.value.upper()}."
        )

        return findings

    # ========================================================================
    # Helpers
    # ========================================================================

    @staticmethod
    def _generate_id() -> str:
        """Generate a unique simulation ID."""

        return (
            f"sim_{uuid4().hex[:16]}"
        )

    @staticmethod
    def _now() -> str:
        """Return UTC timestamp."""

        return datetime.now(
            timezone.utc
        ).isoformat()


# ============================================================================
# Serialization Helpers
# ============================================================================

def simulation_result_to_dict(
    result: SimulationResult,
) -> dict[str, Any]:
    """
    Convert SimulationResult into JSON-compatible
    dictionary data.
    """

    data = asdict(result)

    data["status"] = result.status.value
    data["simulation_type"] = (
        result.simulation_type.value
    )

    data["risk"]["level"] = (
        result.risk.level.value
    )

    return data


def batch_result_to_dict(
    result: BatchSimulationResult,
) -> dict[str, Any]:
    """
    Convert BatchSimulationResult into
    JSON-compatible dictionary data.
    """

    return {
        "batch_id": result.batch_id,

        "simulation_type":
            result.simulation_type.value,

        "status":
            result.status.value,

        "created_at":
            result.created_at,

        "results": [
            simulation_result_to_dict(
                item
            )
            for item in result.results
        ],

        "best_case_simulation_id":
            result.best_case_simulation_id,

        "worst_case_simulation_id":
            result.worst_case_simulation_id,

        "profitable_scenarios":
            result.profitable_scenarios,

        "loss_scenarios":
            result.loss_scenarios,

        "total_scenarios":
            result.total_scenarios,
    }


def stress_test_to_dict(
    result: StressTestResult,
) -> dict[str, Any]:
    """
    Convert StressTestResult into JSON-compatible
    dictionary data.
    """

    return {
        "simulation_id":
            result.simulation_id,

        "status":
            result.status.value,

        "created_at":
            result.created_at,

        "worst_case":
            result.worst_case,

        "best_case":
            result.best_case,

        "loss_scenario_count":
            result.loss_scenario_count,

        "total_scenario_count":
            result.total_scenario_count,

        "loss_rate_pct":
            result.loss_rate_pct,

        "risk_level":
            result.risk_level.value,
    }


def sensitivity_result_to_dict(
    result: SensitivitySimulationResult,
) -> dict[str, Any]:
    """
    Convert sensitivity results into API-friendly data.
    """

    return {
        "simulation_id":
            result.simulation_id,

        "status":
            result.status.value,

        "created_at":
            result.created_at,

        "revenue_sensitivity": (
            asdict(
                result.revenue_sensitivity
            )
            if result.revenue_sensitivity
            else None
        ),

        "expense_sensitivity": (
            asdict(
                result.expense_sensitivity
            )
            if result.expense_sensitivity
            else None
        ),

        "cogs_sensitivity": (
            asdict(
                result.cogs_sensitivity
            )
            if result.cogs_sensitivity
            else None
        ),

        "variable_ranking": [
            asdict(item)
            for item in result.variable_ranking
        ],

        "revenue_loss_threshold":
            result.revenue_loss_threshold,

        "expense_loss_threshold":
            result.expense_loss_threshold,
    }


# ============================================================================
# Convenience Factory
# ============================================================================

def create_simulation_service(
    financial_inputs: FinancialInputs,
    config: Optional[
        SimulationConfig
    ] = None,
) -> SimulationService:
    """
    Create a ready-to-use SimulationService.
    """

    return SimulationService(
        financial_inputs=financial_inputs,
        scenario_registry=(
            create_default_registry()
        ),
        config=config,
    )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("FinCo AI - What-If Simulation Service")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # Financial baseline
    # ------------------------------------------------------------------------

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

    service = create_simulation_service(
        inputs
    )

    # ------------------------------------------------------------------------
    # Single scenario
    # ------------------------------------------------------------------------

    print("\n1. SINGLE SCENARIO")
    print("-" * 80)

    result = (
        service
        .run_registered_scenario(
            "revenue_decline"
        )
    )

    print(
        f"Simulation ID: "
        f"{result.simulation_id}"
    )

    print(
        f"Scenario: "
        f"{result.scenario_name}"
    )

    print(
        f"Baseline Profit: "
        f"${result.baseline.net_profit:,.2f}"
    )

    print(
        f"Scenario Profit: "
        f"${result.scenario.net_profit:,.2f}"
    )

    print(
        f"Risk: "
        f"{result.risk.level.value.upper()}"
    )

    for finding in result.key_findings:
        print(f"- {finding}")

    # ------------------------------------------------------------------------
    # Default scenarios
    # ------------------------------------------------------------------------

    print("\n2. DEFAULT SCENARIO PORTFOLIO")
    print("-" * 80)

    batch = (
        service
        .run_default_scenarios()
    )

    print(
        f"Total scenarios: "
        f"{batch.total_scenarios}"
    )

    print(
        f"Profitable scenarios: "
        f"{batch.profitable_scenarios}"
    )

    print(
        f"Loss scenarios: "
        f"{batch.loss_scenarios}"
    )

    print(
        f"Best simulation: "
        f"{batch.best_case_simulation_id}"
    )

    print(
        f"Worst simulation: "
        f"{batch.worst_case_simulation_id}"
    )

    # ------------------------------------------------------------------------
    # Full sensitivity
    # ------------------------------------------------------------------------

    print("\n3. FULL SENSITIVITY ANALYSIS")
    print("-" * 80)

    sensitivity = (
        service
        .run_full_sensitivity()
    )

    print(
        "Variable impact ranking:"
    )

    for index, impact in enumerate(
        sensitivity.variable_ranking,
        start=1,
    ):

        print(
            f"{index}. "
            f"{impact.variable} -> "
            f"{impact.sensitivity_score:.2f}"
        )

    print(
        "Revenue loss threshold: "
        f"{sensitivity.revenue_loss_threshold}%"
    )

    print(
        "Expense loss threshold: "
        f"{sensitivity.expense_loss_threshold}%"
    )

    # ------------------------------------------------------------------------
    # Stress test
    # ------------------------------------------------------------------------

    print("\n4. DOWNSIDE STRESS TEST")
    print("-" * 80)

    stress = (
        service
        .run_stress_test(
            revenue_declines=[
                5,
                10,
                15,
                20,
            ],
            expense_increases=[
                5,
                10,
                15,
                20,
            ],
        )
    )

    print(
        f"Stress Risk: "
        f"{stress.risk_level.value.upper()}"
    )

    print(
        f"Loss scenarios: "
        f"{stress.loss_scenario_count}/"
        f"{stress.total_scenario_count}"
    )

    print(
        f"Loss rate: "
        f"{stress.loss_rate_pct:.2f}%"
    )

    print(
        "\nWorst Case:"
    )

    print(
        f"  Revenue change: "
        f"{stress.worst_case['revenue_change_pct']:+.1f}%"
    )

    print(
        f"  Expense change: "
        f"{stress.worst_case['expense_change_pct']:+.1f}%"
    )

    print(
        f"  Net profit: "
        f"${stress.worst_case['net_profit']:,.2f}"
    )

    print("\nSimulation service demo complete.")