import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class SimulatedAnnealing(OptimizationAlgorithm):
    def __init__(self):
        self.iterations: int = 120
        self.initial_temperature: float = 10.0
        self.cooling_rate: float = 0.95
        self.step_size: float = 0.5

    def configure(self, config: AlgorithmConfig) -> None:
        if config is not None and config.sa_params:
            self.iterations = config.sa_params.iterations
            self.initial_temperature = config.sa_params.initial_temperature
            self.cooling_rate = config.sa_params.cooling_rate
            self.step_size = config.sa_params.step_size

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(17)

        current = rng.uniform(low, high)
        current_score = self._evaluate(fitness_function, constraints, is_minimization, current)
        best_solution = current.copy()
        best_score = current_score
        temperature = self.initial_temperature

        for _ in range(self.iterations):
            candidate = current + rng.normal(0.0, self.step_size, size=len(bounds))
            candidate = np.clip(candidate, low, high)
            candidate_score = self._evaluate(fitness_function, constraints, is_minimization, candidate)

            delta = candidate_score - current_score if is_minimization else current_score - candidate_score
            if delta <= 0.0 or rng.random() < np.exp(-delta / max(temperature, 1e-12)):
                current = candidate
                current_score = candidate_score
                if (candidate_score < best_score) if is_minimization else (candidate_score > best_score):
                    best_solution = candidate.copy()
                    best_score = candidate_score

            temperature *= self.cooling_rate

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
            "initial_temperature": self.initial_temperature,
            "cooling_rate": self.cooling_rate,
            "step_size": self.step_size,
            "engine": "Simulated Annealing",
        }
