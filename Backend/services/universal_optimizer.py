from __future__ import annotations

from typing import Callable, List

from domain.solutions import CandidateSolution
from adapters.problem_adapters import NumericSearchAdapter
from evaluation.evaluator import ObjectiveEvaluator


class UniversalProblemAdapter:
    """Compatibility facade for the legacy numeric problem adapter."""

    def __init__(self, problem):
        self.problem = problem
        self.representation = NumericSearchAdapter(problem)
        self.evaluator = ObjectiveEvaluator(problem)

    def bounds(self):
        return self.representation.bounds()

    def fitness(self) -> Callable[[List[float]], float]:
        return lambda vector: self.evaluator.fitness(vector, self.representation)

    def decode(self, vector: List[float]) -> CandidateSolution:
        candidate = self.representation.decode(vector)
        value = self.evaluator.objective(candidate, self.representation)
        feasible = self.evaluator.is_feasible(candidate)
        return CandidateSolution(
            values=candidate.values,
            representation=candidate.representation,
            objective_value=value,
            feasible=feasible,
        )


__all__ = ["ObjectiveEvaluator", "UniversalProblemAdapter"]
