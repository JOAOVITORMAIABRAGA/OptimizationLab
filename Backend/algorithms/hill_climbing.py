import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class HillClimbing(OptimizationAlgorithm):
    def __init__(self):
        self.iterations: int = 100
        self.step_size: float = 0.2

    def configure(self, config: AlgorithmConfig) -> None:
        if config is not None and config.hill_climbing_params:
            self.iterations = config.hill_climbing_params.iterations
            self.step_size = config.hill_climbing_params.step_size

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        rng = np.random.default_rng(31)
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        current = rng.uniform(low, high)
        current_score = self._evaluate(fitness_function, constraints, is_minimization, current)
        best_solution = current.copy()
        best_score = current_score

        for _ in range(self.iterations):
            candidate = current + rng.normal(0.0, self.step_size, size=len(bounds))
            candidate = np.clip(candidate, low, high)
            candidate_score = self._evaluate(fitness_function, constraints, is_minimization, candidate)
            if (candidate_score < current_score) if is_minimization else (candidate_score > current_score):
                current = candidate
                current_score = candidate_score
                if (candidate_score < best_score) if is_minimization else (candidate_score > best_score):
                    best_solution = candidate.copy()
                    best_score = candidate_score

        return best_solution.tolist(), float(best_score)

    def _evaluate(self, fitness_function, constraints, is_minimization, solution):
        if constraints:
            for constraint_func in constraints:
                if not constraint_func(solution.tolist()):
                    return float("inf") if is_minimization else float("-inf")
        value = float(fitness_function(solution.tolist()))
        if not np.isfinite(value):
            return float("inf") if is_minimization else float("-inf")
        return value

    def get_params_report(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "step_size": self.step_size,
            "engine": "Hill Climbing",
        }
