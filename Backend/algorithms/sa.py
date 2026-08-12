from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .base import OptimizationAlgorithm
from domain.representations import SolutionRepresentationKind
from adapters.problem_adapters import NeighborhoodSearchAdapter, SearchSpaceAdapter
from schemas import AlgorithmConfig


class SimulatedAnnealing(OptimizationAlgorithm):
    problem_adapter_type = NeighborhoodSearchAdapter
    def __init__(self, seed: int = 17):
        super().__init__(seed)
        self.iterations = 120
        self.initial_temperature = 10.0
        self.cooling_rate = 0.95
        self.step_size = 0.05
        self.neighborhood_size = 1

    def configure(self, config: AlgorithmConfig | None) -> None:
        if config is not None and config.sa_params:
            self.iterations = config.sa_params.iterations
            self.initial_temperature = config.sa_params.initial_temperature
            self.cooling_rate = config.sa_params.cooling_rate
            self.step_size = config.sa_params.step_size

    def optimize(self, fitness_function: Callable, bounds: List[Tuple[float, float]], is_minimization: bool = True, constraints=None):
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(self.seed)
        current = rng.uniform(low, high)
        current_score = self._evaluate(fitness_function, current, is_minimization)
        best = current.copy()
        best_score = current_score
        temperature = max(self.initial_temperature, 1e-12)
        history = [float(best_score)]

        for _ in range(self.iterations):
            candidate = current + rng.normal(0.0, self.step_size, size=len(bounds)) * np.maximum(high - low, 1.0)
            candidate = np.clip(candidate, low, high)
            candidate_score = self._evaluate(fitness_function, candidate, is_minimization)
            delta = candidate_score - current_score if is_minimization else current_score - candidate_score
            if delta <= 0.0 or rng.random() < np.exp(-min(delta / temperature, 700.0)):
                current, current_score = candidate, candidate_score
                if self._better(candidate_score, best_score, is_minimization):
                    best, best_score = candidate.copy(), candidate_score
            temperature *= self.cooling_rate
            history.append(float(best_score))

        self._last_iterations = self.iterations
        self._last_history = history
        return best.tolist(), float(best_score)

    def optimize_representation(self, adapter: SearchSpaceAdapter, evaluator):
        if adapter.kind == SolutionRepresentationKind.VECTOR:
            return super().optimize_representation(adapter, evaluator)

        rng = np.random.default_rng(self.seed)
        minimize = evaluator.problem.objective.sense.value == "minimize"
        current = adapter.random_solution(rng)
        current_score = evaluator.fitness(current, adapter)
        best, best_score = list(current), float(current_score)
        temperature = max(self.initial_temperature, 1e-12)
        history = [best_score]

        for _ in range(self.iterations):
            candidates = adapter.neighbors(current, rng, self.neighborhood_size, self.step_size)
            if not candidates:
                break
            candidate = candidates[0]
            candidate_score = evaluator.fitness(candidate, adapter)
            delta = candidate_score - current_score if minimize else current_score - candidate_score
            if delta <= 0.0 or rng.random() < np.exp(-min(delta / temperature, 700.0)):
                current, current_score = list(candidate), float(candidate_score)
                if self._better(current_score, best_score, minimize):
                    best, best_score = list(candidate), float(candidate_score)
            temperature *= self.cooling_rate
            history.append(best_score)

        self._last_iterations = self.iterations
        self._last_history = history
        return best, best_score

    @staticmethod
    def _better(a, b, minimize):
        return a < b if minimize else a > b

    @staticmethod
    def _evaluate(fitness_function, solution, is_minimization):
        value = float(fitness_function(solution.tolist()))
        return value if np.isfinite(value) else (np.inf if is_minimization else -np.inf)

    def get_params_report(self) -> Dict[str, Any]:
        return {"iterations": self.iterations, "initial_temperature": self.initial_temperature, "cooling_rate": self.cooling_rate, "step_size": self.step_size, "engine": "OptimizationLab native SA"}
