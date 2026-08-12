from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .base import OptimizationAlgorithm
from domain.representations import SolutionRepresentationKind
from adapters.problem_adapters import NeighborhoodSearchAdapter, SearchSpaceAdapter
from schemas import AlgorithmConfig


class HillClimbing(OptimizationAlgorithm):
    problem_adapter_type = NeighborhoodSearchAdapter
    def __init__(self, seed: int = 31):
        super().__init__(seed)
        self.iterations = 100
        self.step_size = 0.2
        self.neighborhood_size = 1

    def configure(self, config: AlgorithmConfig | None) -> None:
        if config is not None and config.hill_climbing_params:
            self.iterations = config.hill_climbing_params.iterations
            self.step_size = config.hill_climbing_params.step_size

    def optimize(self, fitness_function: Callable, bounds: List[Tuple[float, float]], is_minimization: bool = True, constraints=None):
        rng = np.random.default_rng(self.seed)
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        current = rng.uniform(low, high)
        current_score = float(fitness_function(current.tolist()))
        best, best_score = current.copy(), current_score
        history = [best_score]
        for _ in range(self.iterations):
            candidate = np.clip(current + rng.normal(0.0, self.step_size, len(bounds)) * np.maximum(high - low, 1.0), low, high)
            score = float(fitness_function(candidate.tolist()))
            if (score < current_score) if is_minimization else (score > current_score):
                current, current_score = candidate, score
                if (score < best_score) if is_minimization else (score > best_score):
                    best, best_score = candidate.copy(), score
            history.append(best_score)
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
        history = [best_score]
        for _ in range(self.iterations):
            neighbors = adapter.neighbors(current, rng, self.neighborhood_size, self.step_size)
            if not neighbors:
                break
            candidate, score = min(((n, evaluator.fitness(n, adapter)) for n in neighbors), key=lambda item: item[1]) if minimize else max(((n, evaluator.fitness(n, adapter)) for n in neighbors), key=lambda item: item[1])
            if (score < current_score) if minimize else (score > current_score):
                current, current_score = list(candidate), float(score)
                if (score < best_score) if minimize else (score > best_score):
                    best, best_score = list(candidate), float(score)
            history.append(best_score)
        self._last_iterations = self.iterations
        self._last_history = history
        return best, best_score

    def get_params_report(self) -> Dict[str, Any]:
        return {"iterations": self.iterations, "step_size": self.step_size, "engine": "OptimizationLab native Hill Climbing"}
