from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from domain.solutions import CandidateSolution, OptimizationResult
from adapters.problem_adapters import NumericSearchAdapter, SearchSpaceAdapter
from evaluation.evaluator import ObjectiveEvaluator
from schemas import AlgorithmConfig


@dataclass
class AlgorithmContext:
    """Runtime contract shared by native metaheuristic implementations."""

    fitness: Callable[[List[Any]], float]
    bounds: List[Tuple[float, float]]
    is_minimization: bool
    constraints: Optional[List[Callable]] = None


class OptimizationAlgorithm(ABC):
    """Common contract for native algorithms and semantic problem execution."""

    seed: int = 7
    problem_adapter_type = NumericSearchAdapter

    def __init__(self, seed: int | None = None):
        if seed is not None:
            self.seed = seed
        self._last_iterations = 0
        self._last_history: list[float] = []

    @abstractmethod
    def configure(self, config: AlgorithmConfig | None) -> None:
        pass

    @abstractmethod
    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] | None = None,
    ) -> Tuple[List[float], float]:
        """Low-level numeric execution kept for compatibility."""
        pass

    def optimize_representation(self, adapter: SearchSpaceAdapter, evaluator: ObjectiveEvaluator) -> Tuple[List[Any], float]:
        """Default bridge for numeric algorithms. Specialized algorithms override this."""
        vector, score = self.optimize(
            evaluator.fitness_for_adapter(adapter),
            adapter.bounds(),
            is_minimization=(self._sense(evaluator) == "minimize"),
            constraints=None,
        )
        return vector, score

    @staticmethod
    def _sense(evaluator: ObjectiveEvaluator) -> str:
        return evaluator.problem.objective.sense.value

    def create_problem_adapter(self, problem):
        """Adapt the domain problem to this solver family.

        The concrete solver declares the adapter type it needs; the base class
        only owns the lifecycle, never the representation selection policy.
        """
        return self.problem_adapter_type(problem)

    def optimize_problem(self, problem) -> CandidateSolution:
        return self.optimize_problem_result(problem).solution

    def optimize_problem_result(self, problem) -> OptimizationResult:
        adapter = self.create_problem_adapter(problem)
        return self._execute_with_adapter(problem, adapter)

    def optimize_with_adapter(self, problem, adapter: SearchSpaceAdapter, evaluator: ObjectiveEvaluator | None = None) -> OptimizationResult:
        """Execute using an adapter selected by the architecture's adaptation plan."""
        return self._execute_with_adapter(problem, adapter, evaluator=evaluator)

    def _execute_with_adapter(self, problem, adapter: SearchSpaceAdapter, evaluator: ObjectiveEvaluator | None = None) -> OptimizationResult:
        evaluator = evaluator or ObjectiveEvaluator(problem)
        start = perf_counter()
        representation, score = self.optimize_representation(adapter, evaluator)
        elapsed = perf_counter() - start
        candidate = adapter.decode(representation)
        objective_value = evaluator.objective(candidate, adapter)
        feasible = evaluator.is_feasible(candidate)
        history = tuple(getattr(self, "_last_history", ()))
        return OptimizationResult(
            solution=CandidateSolution(
                values=candidate.values,
                representation=candidate.representation,
                objective_value=objective_value,
                feasible=feasible,
                metadata={"algorithm": self.__class__.__name__},
            ),
            objective_value=objective_value,
            feasible=feasible,
            iterations=int(getattr(self, "_last_iterations", 0)),
            evaluations=evaluator.evaluations,
            runtime_seconds=elapsed,
            convergence_history=history,
            algorithm=self.__class__.__name__,
            parameters=self.get_params_report(),
        )

    @abstractmethod
    def get_params_report(self) -> Dict[str, Any]:
        pass
