from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from adapters.problem_adapters import BUILTIN_ADAPTER_REGISTRY
from algorithms.registry import AlgorithmAvailability, AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from domain.problem import OptimizationProblem
from evaluation.evaluator import ObjectiveEvaluator
from recommendation.recommendation_engine import RecommendationEngine
from validation.validator import ValidationEngine


@dataclass(frozen=True)
class ExecutionResult:
    algorithm_id: str
    solution: List[Any]
    variable_values: Dict[str, object]
    objective_value: float
    parameters: Dict[str, object]


class OptimizationExecutionEngine:
    """Execute a user-selected compatible algorithm from an explicit plan."""

    def __init__(self, registry: Optional[AlgorithmRegistry] = None) -> None:
        self.registry = registry or AlgorithmRegistry.from_builtin_algorithms()
        self.validator = ValidationEngine()
        self.compatibility = CompatibilityEngine(BUILTIN_ADAPTER_REGISTRY)
        self.recommendation = RecommendationEngine()

    def execute_auto(self, problem: OptimizationProblem) -> ExecutionResult:
        """Use the system recommendation only when the caller did not choose."""
        result = self.recommendation.recommend(
            problem,
            self.registry,
            compatibility_engine=self.compatibility,
            available_adapters={item.id for item in BUILTIN_ADAPTER_REGISTRY.all()},
        )
        if not result.recommendations:
            raise ValueError("No compatible executable algorithm is available for this problem.")
        return self.execute(problem, result.recommendations[0].algorithm_id)

    def execute(self, problem: OptimizationProblem, algorithm_id: str) -> ExecutionResult:
        report = self.validator.validate(problem)
        if not report.is_valid():
            raise ValueError("Invalid optimization problem: " + "; ".join(report.errors))

        descriptor = self.registry.get(algorithm_id)
        if descriptor.availability != AlgorithmAvailability.AVAILABLE:
            raise ValueError(f"Algorithm '{algorithm_id}' is not available for execution.")

        compatibility = self.compatibility.check(
            problem,
            descriptor,
            available_adapters={item.id for item in BUILTIN_ADAPTER_REGISTRY.all()},
        )
        if compatibility.status == CompatibilityStatus.INCOMPATIBLE:
            raise ValueError(
                f"Algorithm '{algorithm_id}' is incompatible with the problem: "
                + "; ".join(compatibility.reasons or compatibility.failed_checks)
            )

        implementation = descriptor.implementation_class
        if implementation is None:
            raise ValueError(f"Algorithm '{algorithm_id}' has no executable implementation.")

        solver = implementation()
        solver.configure(None)

        if compatibility.adaptation_plan is not None:
            if not hasattr(solver, "optimize_with_adapter"):
                raise ValueError(
                    f"Algorithm '{algorithm_id}' cannot execute an adaptation plan through the standard solver contract."
                )
            adapter_id = compatibility.adaptation_plan.adapter_ids[-1]
            adapter = BUILTIN_ADAPTER_REGISTRY.create(adapter_id, problem)
            result = solver.optimize_with_adapter(problem, adapter, ObjectiveEvaluator(problem))
        else:
            result = solver.optimize_problem_result(problem)

        candidate = result.solution
        if len(candidate.values) == 1:
            solution = next(iter(candidate.values.values()))
        else:
            solution = [candidate.values[v.name] for v in problem.variables]
        return ExecutionResult(
            algorithm_id=algorithm_id,
            solution=solution,
            variable_values=dict(candidate.values),
            objective_value=float(candidate.objective_value),
            parameters=solver.get_params_report(),
        )
