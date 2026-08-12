from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from domain.objectives import ObjectiveSense
from domain.representations import SolutionRepresentationKind
from domain.solutions import CandidateSolution, OptimizationResult
from adapters.problem_adapters import PermutationSearchAdapter, SearchSpaceAdapter
from schemas import AlgorithmConfig
from evaluation.evaluator import ObjectiveEvaluator
from .base import OptimizationAlgorithm


class AntColonyOptimization(OptimizationAlgorithm):
    """Native Ant Colony Optimization over permutation representations.

    Each ant constructs a permutation position-by-position using pheromone and
    optional heuristic information. Pheromone is then evaporated and reinforced
    by the best-ranked ants from the iteration.
    """

    def __init__(self, seed: int = 23):
        super().__init__()
        self.seed = seed
        self.ants = 30
        self.iterations = 60
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.5
        self.elite_ants = 5
        self.q = 1.0

    problem_adapter_type = PermutationSearchAdapter

    def configure(self, config: AlgorithmConfig | None) -> None:
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
        constraints=None,
    ) -> Tuple[List[float], float]:
        """Low-level ACO over permutation indices.

        This compatibility API has no semantic element names, so a problem of
        ``n`` dimensions is represented as the permutation ``0..n-1``. The
        universal ``optimize_problem`` path should be preferred because its
        representation adapter translates those indices into domain values.
        """
        if len(bounds) < 2:
            raise ValueError("ACO requires at least two dimensions.")
        if self.ants < 1 or self.iterations < 1:
            raise ValueError("ACO ants and iterations must be positive.")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("ACO rho must be in [0, 1).")

        start = perf_counter()
        rng = np.random.default_rng(self.seed)
        n = len(bounds)
        pheromone = np.ones((n, n), dtype=float)
        best: List[int] | None = None
        best_score = float("inf") if is_minimization else float("-inf")
        history = [best_score]

        for _ in range(self.iterations):
            scored: List[Tuple[List[int], float]] = []
            for _ in range(self.ants):
                available = list(range(n))
                permutation: List[int] = []
                for position in range(n):
                    candidates = np.asarray(available, dtype=int)
                    weights = np.power(np.maximum(pheromone[position, candidates], 1e-12), self.alpha)
                    probabilities = weights / weights.sum()
                    selected = int(candidates[int(rng.choice(len(candidates), p=probabilities))])
                    permutation.append(selected)
                    available.remove(selected)

                if constraints and any(not c(permutation) for c in constraints):
                    continue
                score = float(fitness_function(permutation))
                if not np.isfinite(score):
                    continue
                scored.append((permutation, score))
                better = score < best_score if is_minimization else score > best_score
                if best is None or better:
                    best = permutation.copy()
                    best_score = score

            pheromone *= (1.0 - self.rho)
            ranked = self._quality_order(scored, is_minimization)
            for rank, (permutation, _) in enumerate(ranked[: min(self.elite_ants, len(ranked))], start=1):
                deposit = self.q / rank
                for position, element_index in enumerate(permutation):
                    pheromone[position, element_index] += deposit
            pheromone = np.clip(pheromone, 1e-12, 1e12)
            history.append(best_score)

        if best is None:
            raise ValueError("ACO could not construct a feasible permutation solution.")
        self._last_iterations = self.iterations
        self._last_history = history
        return [float(value) for value in best], float(best_score)

    def optimize_representation(self, adapter: SearchSpaceAdapter, evaluator):
        return self._run_permutation(adapter, evaluator)

    def _heuristic_matrix(self, adapter: SearchSpaceAdapter) -> np.ndarray:
        matrix = np.asarray(adapter.heuristic_matrix(), dtype=float)
        size = len(adapter.bounds())
        if matrix.shape != (size, size):
            raise ValueError(f"ACO heuristic matrix must have shape ({size}, {size}).")
        if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
            raise ValueError("ACO heuristic matrix must contain finite non-negative values.")
        return matrix

    def _construct_solution(self, pheromone: np.ndarray, heuristic: np.ndarray, rng: np.random.Generator) -> List[int]:
        n = pheromone.shape[0]
        available = list(range(n))
        permutation: List[int] = []
        for position in range(n):
            candidates = np.asarray(available, dtype=int)
            weights = np.power(np.maximum(pheromone[position, candidates], 1e-12), self.alpha)
            weights *= np.power(np.maximum(heuristic[position, candidates], 1e-12), self.beta)
            total = float(weights.sum())
            if not np.isfinite(total) or total <= 0:
                probabilities = np.full(len(candidates), 1.0 / len(candidates))
            else:
                probabilities = weights / total
            selected_offset = int(rng.choice(len(candidates), p=probabilities))
            selected = int(candidates[selected_offset])
            permutation.append(selected)
            available.remove(selected)
        return permutation

    def _quality_order(self, scored: List[Tuple[List[int], float]], is_minimization: bool) -> List[Tuple[List[int], float]]:
        return sorted(scored, key=lambda item: item[1], reverse=not is_minimization)

    def _run_permutation(self, adapter: SearchSpaceAdapter, evaluator: ObjectiveEvaluator) -> Tuple[List[int], float]:
        if self.ants < 1 or self.iterations < 1:
            raise ValueError("ACO ant count and iterations must be positive.")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("ACO rho must be in [0, 1).")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("ACO alpha and beta must be non-negative.")

        start = perf_counter()
        rng = np.random.default_rng(self.seed)
        n = len(adapter.bounds())
        pheromone = np.ones((n, n), dtype=float)
        heuristic = self._heuristic_matrix(adapter)
        is_minimization = evaluator.problem.objective.sense == ObjectiveSense.MINIMIZE
        best_indices: List[int] | None = None
        best_score = float("inf") if is_minimization else float("-inf")
        history: List[float] = [best_score]

        def better(value: float, reference: float) -> bool:
            return value < reference if is_minimization else value > reference

        for _ in range(self.iterations):
            scored: List[Tuple[List[int], float]] = []
            for _ in range(self.ants):
                indices = self._construct_solution(pheromone, heuristic, rng)
                score = evaluator.fitness(indices, adapter)
                if np.isfinite(score):
                    scored.append((indices, float(score)))
                    if best_indices is None or better(float(score), best_score):
                        best_indices = list(indices)
                        best_score = float(score)

            pheromone *= (1.0 - self.rho)
            if scored:
                ranked = self._quality_order(scored, is_minimization)
                for rank, (indices, score) in enumerate(ranked[: min(self.elite_ants, len(ranked))], start=1):
                    deposit = self.q / max(abs(score), 1e-12) / rank
                    for position, element_index in enumerate(indices):
                        pheromone[position, element_index] += deposit

            pheromone = np.clip(pheromone, 1e-12, 1e12)
            history.append(best_score)

        if best_indices is None:
            raise ValueError("ACO could not construct a feasible permutation solution.")
        self._last_iterations = self.iterations
        self._last_history = history
        _ = start  # retained for deterministic low-level timing compatibility
        return best_indices, float(best_score)

    def get_params_report(self) -> Dict[str, Any]:
        return {
            "ants": self.ants,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "elite_ants": self.elite_ants,
            "q": self.q,
            "engine": "OptimizationLab native ACO",
        }
