import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class BacterialForagingOptimization(OptimizationAlgorithm):
    def __init__(self):
        self.bacteria_count: int = 20
        self.chemotaxis_steps: int = 10
        self.reproduction_steps: int = 5
        self.step_size: float = 0.25

    def configure(self, config: AlgorithmConfig) -> None:
        if config is not None and config.bfo_params:
            self.bacteria_count = config.bfo_params.bacteria_count
            self.chemotaxis_steps = config.bfo_params.chemotaxis_steps
            self.reproduction_steps = config.bfo_params.reproduction_steps
            self.step_size = config.bfo_params.step_size

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(13)

        bacteria = rng.uniform(low, high, size=(self.bacteria_count, len(bounds)))
        scores = np.array([self._evaluate(fitness_function, constraints, is_minimization, b) for b in bacteria], dtype=float)
        best_idx = np.argmin(scores) if is_minimization else np.argmax(scores)
        best_solution = bacteria[best_idx].copy()
        best_score = scores[best_idx]

        for _ in range(self.reproduction_steps):
            for i in range(self.bacteria_count):
                for _ in range(self.chemotaxis_steps):
                    direction = rng.normal(size=len(bounds))
                    direction /= np.linalg.norm(direction) or 1.0
                    candidate = bacteria[i] + self.step_size * direction
                    candidate = np.clip(candidate, low, high)
                    candidate_score = self._evaluate(fitness_function, constraints, is_minimization, candidate)
                    if (candidate_score < scores[i]) if is_minimization else (candidate_score > scores[i]):
                        bacteria[i] = candidate
                        scores[i] = candidate_score
                        if (candidate_score < best_score) if is_minimization else (candidate_score > best_score):
                            best_solution = candidate.copy()
                            best_score = candidate_score

            survivors = self.bacteria_count // 2
            order = np.argsort(scores) if is_minimization else np.argsort(-scores)
            bacteria = bacteria[order[:survivors]]
            scores = scores[order[:survivors]]
            if len(bacteria) < self.bacteria_count:
                bacteria = np.vstack([bacteria, rng.uniform(low, high, size=(self.bacteria_count - len(bacteria), len(bounds)))])
                scores = np.concatenate([scores, np.full(self.bacteria_count - len(scores), np.inf if is_minimization else -np.inf)])

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
            "bacteria_count": self.bacteria_count,
            "chemotaxis_steps": self.chemotaxis_steps,
            "reproduction_steps": self.reproduction_steps,
            "step_size": self.step_size,
            "engine": "Bacterial Foraging Optimization",
        }
