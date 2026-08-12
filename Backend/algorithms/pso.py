from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class ParticleSwarmOptimization(OptimizationAlgorithm):
    """Native vector PSO with explicit position/velocity state."""

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.swarm_size = 30
        self.iterations = 80
        self.inertia = 0.7
        self.c1 = 1.5
        self.c2 = 1.5

    def configure(self, config: AlgorithmConfig | None) -> None:
        if config is not None and config.pso_params:
            self.swarm_size = config.pso_params.swarm_size
            self.iterations = config.pso_params.iterations

    def optimize(self, fitness_function: Callable, bounds: List[Tuple[float, float]], is_minimization: bool = True, constraints=None):
        if self.swarm_size < 2:
            raise ValueError("PSO swarm size must be at least 2.")
        rng = np.random.default_rng(self.seed)
        low = np.asarray([b[0] for b in bounds], dtype=float)
        high = np.asarray([b[1] for b in bounds], dtype=float)
        span = high - low
        positions = rng.uniform(low, high, size=(self.swarm_size, len(bounds)))
        velocities = rng.uniform(-0.1 * span, 0.1 * span, size=positions.shape)

        def evaluate(x):
            if constraints and any(not c(x.tolist()) for c in constraints):
                return np.inf if is_minimization else -np.inf
            value = float(fitness_function(x.tolist()))
            return value if np.isfinite(value) else (np.inf if is_minimization else -np.inf)

        pbest = positions.copy()
        pbest_scores = np.asarray([evaluate(p) for p in positions])
        idx = int(np.argmin(pbest_scores) if is_minimization else np.argmax(pbest_scores))
        gbest = pbest[idx].copy()
        gbest_score = float(pbest_scores[idx])
        history = [gbest_score]

        for _ in range(self.iterations):
            for i in range(self.swarm_size):
                r1 = rng.random(len(bounds))
                r2 = rng.random(len(bounds))
                velocities[i] = (
                    self.inertia * velocities[i]
                    + self.c1 * r1 * (pbest[i] - positions[i])
                    + self.c2 * r2 * (gbest - positions[i])
                )
                positions[i] = np.clip(positions[i] + velocities[i], low, high)
                score = evaluate(positions[i])
                if (score < pbest_scores[i]) if is_minimization else (score > pbest_scores[i]):
                    pbest[i] = positions[i].copy()
                    pbest_scores[i] = score
                    if (score < gbest_score) if is_minimization else (score > gbest_score):
                        gbest = positions[i].copy()
                        gbest_score = float(score)
            history.append(gbest_score)

        self._last_iterations = self.iterations
        self._last_history = history
        return gbest.tolist(), gbest_score

    def get_params_report(self) -> Dict[str, Any]:
        return {
            "swarm_size": self.swarm_size,
            "iterations": self.iterations,
            "inertia": self.inertia,
            "c1": self.c1,
            "c2": self.c2,
            "engine": "OptimizationLab native PSO",
        }
