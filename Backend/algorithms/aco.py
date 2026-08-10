import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class AntColonyOptimization(OptimizationAlgorithm):
    def __init__(self):
        self.ants: int = 30
        self.iterations: int = 60
        self.alpha: float = 1.0
        self.beta: float = 2.0
        self.rho: float = 0.5

    def configure(self, config: AlgorithmConfig) -> None:
        if config is not None and config.aco_params:
            self.ants = config.aco_params.ants
            self.iterations = config.aco_params.iterations
            self.alpha = config.aco_params.alpha
            self.beta = config.aco_params.beta
            self.rho = config.aco_params.rho

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        n_vars = len(bounds)
        if n_vars < 2:
            return [0.0], float(fitness_function([0.0]))

        rng = np.random.default_rng(23)
        pheromone = np.ones((n_vars, 2))
        solutions = []
        best_solution = np.zeros(n_vars)
        best_score = float("inf") if is_minimization else float("-inf")

        for _ in range(self.iterations):
            for _ in range(self.ants):
                sol = []
                for i in range(n_vars):
                    low, high = bounds[i]
                    if i == 0:
                        choice = rng.choice([0, 1])
                    else:
                        choice = rng.choice([0, 1])
                    sol.append(float(low + (high - low) * choice))
                solutions.append(np.array(sol))

            scores = []
            for sol in solutions:
                score = self._evaluate(fitness_function, constraints, is_minimization, sol.tolist())
                scores.append(score)

            if is_minimization:
                best_idx = int(np.argmin(scores))
            else:
                best_idx = int(np.argmax(scores))
            if (scores[best_idx] < best_score) if is_minimization else (scores[best_idx] > best_score):
                best_score = scores[best_idx]
                best_solution = solutions[best_idx].copy()

            pheromone *= self.rho
            for idx, sol in enumerate(solutions):
                for j in range(n_vars):
                    pheromone[j, 0] += 1.0 / (1.0 + abs(sol[j] - bounds[j][0]))
                    pheromone[j, 1] += 1.0 / (1.0 + abs(sol[j] - bounds[j][1]))
            solutions = []

        return best_solution.tolist(), float(best_score)

    def _evaluate(self, fitness_function, constraints, is_minimization, solution):
        if constraints:
            for constraint_func in constraints:
                if not constraint_func(solution):
                    return float("inf") if is_minimization else float("-inf")
        value = float(fitness_function(solution))
        if not np.isfinite(value):
            return float("inf") if is_minimization else float("-inf")
        return value

    def get_params_report(self) -> Dict[str, Any]:
        return {
            "ants": self.ants,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "engine": "Ant Colony Optimization",
        }
