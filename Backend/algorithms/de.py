import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class DifferentialEvolution(OptimizationAlgorithm):
    def __init__(self):
        self.population_size: int = 40
        self.iterations: int = 80
        self.mutation_factor: float = 0.8
        self.crossover_rate: float = 0.7

    def configure(self, config: AlgorithmConfig) -> None:
        if config is not None and config.de_params:
            self.population_size = config.de_params.population_size
            self.iterations = config.de_params.iterations
            self.mutation_factor = config.de_params.mutation_factor
            self.crossover_rate = config.de_params.crossover_rate

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(11)

        population = rng.uniform(low, high, size=(self.population_size, len(bounds)))
        scores = np.array([self._evaluate(fitness_function, constraints, is_minimization, p) for p in population], dtype=float)

        best_idx = np.argmin(scores) if is_minimization else np.argmax(scores)
        best_solution = population[best_idx].copy()
        best_score = scores[best_idx]

        for _ in range(self.iterations):
            for i in range(self.population_size):
                candidates = [j for j in range(self.population_size) if j != i]
                a, b, c = population[rng.choice(candidates, size=3, replace=False)]
                mutant = a + self.mutation_factor * (b - c)
                trial = population[i].copy()
                cross_points = rng.random(len(bounds)) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[rng.integers(0, len(bounds))] = True
                trial[cross_points] = mutant[cross_points]
                trial = np.clip(trial, low, high)
                trial_score = self._evaluate(fitness_function, constraints, is_minimization, trial)

                if (trial_score < scores[i]) if is_minimization else (trial_score > scores[i]):
                    population[i] = trial
                    scores[i] = trial_score
                    if (trial_score < best_score) if is_minimization else (trial_score > best_score):
                        best_score = trial_score
                        best_solution = trial.copy()

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
            "population_size": self.population_size,
            "iterations": self.iterations,
            "mutation_factor": self.mutation_factor,
            "crossover_rate": self.crossover_rate,
            "engine": "Differential Evolution",
        }
