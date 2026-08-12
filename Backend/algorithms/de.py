from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class DifferentialEvolution(OptimizationAlgorithm):
    """Native Differential Evolution using the classic DE/rand/1/bin strategy."""

    def __init__(self, seed: int = 11):
        self.seed = seed
        self.population_size = 40
        self.iterations = 80
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def configure(self, config: AlgorithmConfig | None) -> None:
        if config is not None and config.de_params:
            self.population_size = config.de_params.population_size
            self.iterations = config.de_params.iterations
            self.mutation_factor = config.de_params.mutation_factor
            self.crossover_rate = config.de_params.crossover_rate

    def optimize(self, fitness_function: Callable, bounds: List[Tuple[float, float]], is_minimization: bool = True, constraints=None):
        if self.population_size < 4:
            raise ValueError("DE population size must be at least 4.")
        rng = np.random.default_rng(self.seed)
        low = np.asarray([b[0] for b in bounds], dtype=float)
        high = np.asarray([b[1] for b in bounds], dtype=float)
        n = len(bounds)

        def evaluate(x):
            if constraints and any(not c(x.tolist()) for c in constraints):
                return np.inf if is_minimization else -np.inf
            value = float(fitness_function(x.tolist()))
            return value if np.isfinite(value) else (np.inf if is_minimization else -np.inf)

        population = rng.uniform(low, high, size=(self.population_size, n))
        scores = np.asarray([evaluate(x) for x in population])
        best_idx = int(np.argmin(scores) if is_minimization else np.argmax(scores))
        best = population[best_idx].copy()
        best_score = float(scores[best_idx])
        history = [best_score]

        for _ in range(self.iterations):
            for i in range(self.population_size):
                choices = [j for j in range(self.population_size) if j != i]
                a, b, c = population[rng.choice(choices, size=3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), low, high)
                trial = population[i].copy()
                mask = rng.random(n) < self.crossover_rate
                mask[rng.integers(0, n)] = True
                trial[mask] = mutant[mask]
                score = evaluate(trial)
                if (score < scores[i]) if is_minimization else (score > scores[i]):
                    population[i] = trial
                    scores[i] = score
                    if (score < best_score) if is_minimization else (score > best_score):
                        best = trial.copy()
                        best_score = float(score)
            history.append(best_score)

        self._last_iterations = self.iterations
        self._last_history = history
        return best.tolist(), best_score

    def get_params_report(self) -> Dict[str, Any]:
        return {
            "population_size": self.population_size,
            "iterations": self.iterations,
            "mutation_factor": self.mutation_factor,
            "crossover_rate": self.crossover_rate,
            "engine": "OptimizationLab native Differential Evolution",
        }
