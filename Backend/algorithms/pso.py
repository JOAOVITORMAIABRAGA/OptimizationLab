import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class ParticleSwarmOptimization(OptimizationAlgorithm):
    def __init__(self):
        self.swarm_size: int = 30
        self.iterations: int = 80
        self.inertia: float = 0.7
        self.c1: float = 1.5
        self.c2: float = 1.5

    def configure(self, config: AlgorithmConfig) -> None:
        if config is not None and config.pso_params:
            self.swarm_size = config.pso_params.swarm_size
            self.iterations = config.pso_params.iterations

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        low = np.array([b[0] for b in bounds], dtype=float)
        high = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(7)

        positions = rng.uniform(low, high, size=(self.swarm_size, len(bounds)))
        velocities = rng.uniform(-0.1 * (high - low), 0.1 * (high - low), size=(self.swarm_size, len(bounds)))

        pbest_positions = positions.copy()
        pbest_scores = np.full(self.swarm_size, np.inf if is_minimization else -np.inf, dtype=float)
        gbest_position = positions[0].copy()
        gbest_score = np.inf if is_minimization else -np.inf

        def evaluate(solution: np.ndarray) -> float:
            candidate = np.array(solution, dtype=float)
            if constraints:
                for constraint_func in constraints:
                    if not constraint_func(candidate.tolist()):
                        return float("inf") if is_minimization else float("-inf")
            value = float(fitness_function(candidate.tolist()))
            if not np.isfinite(value):
                return float("inf") if is_minimization else float("-inf")
            return value

        for _ in range(self.iterations):
            for i in range(self.swarm_size):
                score = evaluate(positions[i])
                if (score < pbest_scores[i]) if is_minimization else (score > pbest_scores[i]):
                    pbest_scores[i] = score
                    pbest_positions[i] = positions[i].copy()

                if (score < gbest_score) if is_minimization else (score > gbest_score):
                    gbest_score = score
                    gbest_position = positions[i].copy()

            for i in range(self.swarm_size):
                r1 = rng.random(len(bounds))
                r2 = rng.random(len(bounds))
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.c1 * r1 * (pbest_positions[i] - positions[i])
                    + self.c2 * r2 * (gbest_position - positions[i])
                )
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], low, high)

        return gbest_position.tolist(), float(gbest_score)

    def get_params_report(self) -> Dict[str, Any]:
        return {
            "swarm_size": self.swarm_size,
            "iterations": self.iterations,
            "inertia": self.inertia,
            "c1": self.c1,
            "c2": self.c2,
            "engine": "Particle Swarm Optimization",
        }
