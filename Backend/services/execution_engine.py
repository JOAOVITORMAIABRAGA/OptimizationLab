from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from algorithms.registry import AlgorithmAvailability, AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from domain.problem import OptimizationProblem
from validation.validator import ValidationEngine


@dataclass(frozen=True)
class ExecutionResult:
    algorithm_id: str
    solution: List[float]
    objective_value: float
    parameters: Dict[str, object]


class OptimizationExecutionEngine:
    """Validates, checks compatibility and executes a registered real solver."""

    def __init__(self, registry: AlgorithmRegistry | None = None) -> None:
        self.registry = registry or AlgorithmRegistry.from_builtin_algorithms()
        self.validator = ValidationEngine()
        self.compatibility = CompatibilityEngine()

    def execute(self, problem: OptimizationProblem, algorithm_id: str) -> ExecutionResult:
        report = self.validator.validate(problem)
        if not report.is_valid():
            raise ValueError("Invalid optimization problem: " + "; ".join(report.errors))

        descriptor = self.registry.get(algorithm_id)
        if descriptor.availability != AlgorithmAvailability.AVAILABLE:
            raise ValueError(f"Algorithm '{algorithm_id}' is not available for execution.")

        compatibility = self.compatibility.check(problem, descriptor)
        if compatibility.status != CompatibilityStatus.COMPATIBLE:
            raise ValueError(
                f"Algorithm '{algorithm_id}' is not directly compatible: "
                + "; ".join(compatibility.reasons or compatibility.failed_checks)
            )

        implementation = descriptor.implementation_class
        if implementation is None:
            raise ValueError(f"Algorithm '{algorithm_id}' has no executable implementation.")

        solver = implementation()
        solver.configure(None)
        solution, objective_value = solver.optimize(problem)
        return ExecutionResult(
            algorithm_id=algorithm_id,
            solution=solution,
            objective_value=objective_value,
            parameters=solver.get_params_report(),
        )
