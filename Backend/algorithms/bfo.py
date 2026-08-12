from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class BacterialForagingOptimization(OptimizationAlgorithm):
    """Native BFO kept as a first-class numeric metaheuristic."""

    def __init__(self, seed: int = 13):
        super().__init__(seed)
        self.bacteria_count = 20
        self.chemotaxis_steps = 10
        self.reproduction_steps = 5
        self.step_size = 0.25

    def configure(self, config: AlgorithmConfig | None) -> None:
        if config is not None and config.bfo_params:
            self.bacteria_count = config.bfo_params.bacteria_count
            self.chemotaxis_steps = config.bfo_params.chemotaxis_steps
            self.reproduction_steps = config.bfo_params.reproduction_steps
            self.step_size = config.bfo_params.step_size

    def optimize(self, fitness_function: Callable, bounds: List[Tuple[float, float]], is_minimization: bool = True, constraints=None):
        if self.bacteria_count < 2:
            raise ValueError("BFO bacteria_count must be at least 2.")
        rng = np.random.default_rng(self.seed)
        low = np.asarray([b[0] for b in bounds], dtype=float)
        high = np.asarray([b[1] for b in bounds], dtype=float)
        span = np.maximum(high - low, 1.0)
        bacteria = rng.uniform(low, high, size=(self.bacteria_count, len(bounds)))

        def evaluate(x):
            value = float(fitness_function(x.tolist()))
            return value if np.isfinite(value) else (np.inf if is_minimization else -np.inf)

        scores = np.asarray([evaluate(b) for b in bacteria])
        best_idx = int(np.argmin(scores) if is_minimization else np.argmax(scores))
        best, best_score = bacteria[best_idx].copy(), float(scores[best_idx])
        history = [best_score]

        for _ in range(self.reproduction_steps):
            for i in range(self.bacteria_count):
                for _ in range(self.chemotaxis_steps):
                    direction = rng.normal(size=len(bounds))
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction /= norm
                    candidate = np.clip(bacteria[i] + self.step_size * span * direction, low, high)
                    score = evaluate(candidate)
                    if (score < scores[i]) if is_minimization else (score > scores[i]):
                        bacteria[i], scores[i] = candidate, score
                        if (score < best_score) if is_minimization else (score > best_score):
                            best, best_score = candidate.copy(), score
            order = np.argsort(scores) if is_minimization else np.argsort(-scores)
            survivors = max(1, self.bacteria_count // 2)
            bacteria = bacteria[order[:survivors]]
            scores = scores[order[:survivors]]
            clones = rng.choice(len(bacteria), size=self.bacteria_count - len(bacteria), replace=True)
            if len(clones):
                bacteria = np.vstack([bacteria, np.clip(bacteria[clones] + rng.normal(0, 0.05, size=(len(clones), len(bounds))) * span, low, high)])
                scores = np.concatenate([scores, np.asarray([evaluate(b) for b in bacteria[-len(clones):]])])
            history.append(best_score)

        self._last_iterations = self.reproduction_steps
        self._last_history = history
        return best.tolist(), float(best_score)

    def get_params_report(self) -> Dict[str, Any]:
        return {"bacteria_count": self.bacteria_count, "chemotaxis_steps": self.chemotaxis_steps, "reproduction_steps": self.reproduction_steps, "step_size": self.step_size, "engine": "OptimizationLab native BFO"}
