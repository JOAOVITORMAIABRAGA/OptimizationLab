from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .base import OptimizationAlgorithm
from domain.representations import SolutionRepresentationKind
from adapters.problem_adapters import NeighborhoodSearchAdapter, SearchSpaceAdapter
from schemas import AlgorithmConfig


class TabuSearch(OptimizationAlgorithm):
    problem_adapter_type = NeighborhoodSearchAdapter
    def __init__(self, seed: int = 29):
        super().__init__(seed)
        self.iterations = 80
        self.tabu_size = 10
        self.neighborhood_size = 8
        self.step_size = 0.2

    def configure(self, config: AlgorithmConfig | None) -> None:
        if config is not None and config.tabu_params:
            self.iterations = config.tabu_params.iterations
            self.tabu_size = config.tabu_params.tabu_size
            self.neighborhood_size = config.tabu_params.neighborhood_size

    def optimize(self, fitness_function: Callable, bounds: List[Tuple[float, float]], is_minimization: bool = True, constraints=None):
        rng = np.random.default_rng(self.seed)
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        current = rng.uniform(low, high)
        current_score = float(fitness_function(current.tolist()))
        best, best_score = current.copy(), current_score
        tabu: list[tuple] = []
        history = [best_score]
        for _ in range(self.iterations):
            candidates = []
            for _ in range(self.neighborhood_size):
                c = np.clip(current + rng.normal(0.0, self.step_size, len(bounds)) * np.maximum(high - low, 1.0), low, high)
                candidates.append(c)
            feasible = [c for c in candidates if tuple(np.round(c, 10)) not in tabu]
            if not feasible:
                feasible = candidates
            candidate = min(feasible, key=lambda x: fitness_function(x.tolist())) if is_minimization else max(feasible, key=lambda x: fitness_function(x.tolist()))
            score = float(fitness_function(candidate.tolist()))
            current, current_score = candidate, score
            key = tuple(np.round(candidate, 10))
            tabu.append(key)
            if len(tabu) > self.tabu_size:
                tabu.pop(0)
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
        tabu: list[tuple] = []
        history = [best_score]
        for _ in range(self.iterations):
            candidates = adapter.neighbors(current, rng, self.neighborhood_size, self.step_size)
            allowed = [c for c in candidates if tuple(c) not in tabu]
            if not allowed:
                allowed = candidates
            if not allowed:
                break
            scored = [(candidate, evaluator.fitness(candidate, adapter)) for candidate in allowed]
            candidate, score = (min(scored, key=lambda item: item[1]) if minimize else max(scored, key=lambda item: item[1]))
            current, current_score = list(candidate), float(score)
            tabu.append(tuple(current))
            if len(tabu) > self.tabu_size:
                tabu.pop(0)
            if (score < best_score) if minimize else (score > best_score):
                best, best_score = list(candidate), float(score)
            history.append(best_score)
        self._last_iterations = self.iterations
        self._last_history = history
        return best, best_score

    def get_params_report(self) -> Dict[str, Any]:
        return {"iterations": self.iterations, "tabu_size": self.tabu_size, "neighborhood_size": self.neighborhood_size, "step_size": self.step_size, "engine": "OptimizationLab native Tabu Search"}
