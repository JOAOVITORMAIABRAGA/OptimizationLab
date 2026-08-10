import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class TabuSearch(OptimizationAlgorithm):
    def __init__(self):
        self.iterations: int = 80
        self.tabu_size: int = 10
        self.neighborhood_size: int = 5

    def configure(self, config: AlgorithmConfig) -> None:
        if config is not None and config.tabu_params:
            self.iterations = config.tabu_params.iterations
            self.tabu_size = config.tabu_params.tabu_size
            self.neighborhood_size = config.tabu_params.neighborhood_size

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        rng = np.random.default_rng(29)
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        current = rng.uniform(low, high)
        current_score = self._evaluate(fitness_function, constraints, is_minimization, current)
        best_solution = current.copy()
        best_score = current_score
        tabu_list = []

        for _ in range(self.iterations):
            candidates = []
            for _ in range(self.neighborhood_size):
                candidate = current + rng.normal(0.0, 0.3, size=len(bounds))
                candidate = np.clip(candidate, low, high)
                candidates.append(candidate)

            best_candidate = None
            best_candidate_score = None
            for candidate in candidates:
                if candidate.tolist() in tabu_list:
                    continue
                score = self._evaluate(fitness_function, constraints, is_minimization, candidate)
                if best_candidate is None or ((score < best_candidate_score) if is_minimization else (score > best_candidate_score)):
                    best_candidate = candidate
                    best_candidate_score = score
            if best_candidate is None:
                break
            current = best_candidate
            current_score = best_candidate_score
            if (best_candidate_score < best_score) if is_minimization else (best_candidate_score > best_score):
                best_solution = best_candidate.copy()
                best_score = best_candidate_score
            tabu_list.append(current.tolist())
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)

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
            "tabu_size": self.tabu_size,
            "neighborhood_size": self.neighborhood_size,
            "engine": "Tabu Search",
        }
