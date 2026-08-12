from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .base import OptimizationAlgorithm
from adapters.problem_adapters import SearchSpaceAdapter
from domain.representations import SolutionRepresentationKind
from schemas import AlgorithmConfig


class GeneticAlgorithm(OptimizationAlgorithm):
    """Native real-coded genetic algorithm.

    The implementation follows the spirit of the original project: tournament
    selection, pairwise crossover, mutation and generational replacement.  The
    problem now supplies bounds/types instead of hard-coded ML "versions".
    """

    def __init__(self, seed: int = 7):
        self.seed = seed
        self.pop_size = 50
        self.generations = 100
        self.mutation_rate = 0.05
        self.crossover_rate = 1.0
        self.elitism = 1

    def configure(self, config: AlgorithmConfig | None) -> None:
        if config is not None and config.ga_params:
            self.pop_size = config.ga_params.pop_size
            self.generations = config.ga_params.generations
            self.mutation_rate = config.ga_params.mutation_rate

    def optimize(self, fitness_function: Callable, bounds: List[Tuple[float, float]], is_minimization: bool = True, constraints=None):
        if self.pop_size < 2:
            raise ValueError("GA population size must be at least 2.")
        rng = np.random.default_rng(self.seed)
        low = np.asarray([b[0] for b in bounds], dtype=float)
        high = np.asarray([b[1] for b in bounds], dtype=float)
        n_genes = len(bounds)

        def evaluate(x):
            if constraints and any(not c(x.tolist()) for c in constraints):
                return np.inf if is_minimization else -np.inf
            value = float(fitness_function(x.tolist()))
            return value if np.isfinite(value) else (np.inf if is_minimization else -np.inf)

        population = rng.uniform(low, high, size=(self.pop_size, n_genes))
        scores = np.asarray([evaluate(ind) for ind in population])

        def better(a, b):
            return a < b if is_minimization else a > b

        def tournament():
            i, j = rng.integers(0, self.pop_size, size=2)
            return population[i] if better(scores[i], scores[j]) else population[j]

        best_idx = int(np.argmin(scores) if is_minimization else np.argmax(scores))
        best = population[best_idx].copy()
        best_score = float(scores[best_idx])
        history = [best_score]

        for _ in range(self.generations):
            order = np.argsort(scores) if is_minimization else np.argsort(-scores)
            next_population = [population[order[0]].copy()] if self.elitism else []
            while len(next_population) < self.pop_size:
                parent1 = tournament().copy()
                parent2 = tournament().copy()
                child1, child2 = self._crossover(parent1, parent2, rng, low, high)
                next_population.append(self._mutate(child1, rng, low, high))
                if len(next_population) < self.pop_size:
                    next_population.append(self._mutate(child2, rng, low, high))
            population = np.asarray(next_population[: self.pop_size])
            scores = np.asarray([evaluate(ind) for ind in population])
            idx = int(np.argmin(scores) if is_minimization else np.argmax(scores))
            if better(scores[idx], best_score):
                best = population[idx].copy()
                best_score = float(scores[idx])
            history.append(best_score)

        self._last_iterations = self.generations
        self._last_history = history
        return best.tolist(), best_score

    def optimize_representation(self, adapter: SearchSpaceAdapter, evaluator):
        if adapter.kind == SolutionRepresentationKind.VECTOR:
            return super().optimize_representation(adapter, evaluator)
        if adapter.kind != SolutionRepresentationKind.PERMUTATION:
            raise ValueError(f"GA does not support representation '{adapter.kind}'.")
        if self.pop_size < 2:
            raise ValueError("GA population size must be at least 2.")

        rng = np.random.default_rng(self.seed)
        minimize = evaluator.problem.objective.sense.value == "minimize"
        population = [adapter.random_solution(rng) for _ in range(self.pop_size)]
        scores = [float(evaluator.fitness(individual, adapter)) for individual in population]

        def better(a, b):
            return a < b if minimize else a > b

        def tournament():
            i, j = rng.integers(0, self.pop_size, size=2)
            return population[int(i)] if better(scores[int(i)], scores[int(j)]) else population[int(j)]

        best_index = min(range(self.pop_size), key=lambda i: scores[i]) if minimize else max(range(self.pop_size), key=lambda i: scores[i])
        best = list(population[best_index])
        best_score = float(scores[best_index])
        history = [best_score]

        for _ in range(self.generations):
            order = sorted(range(self.pop_size), key=lambda i: scores[i], reverse=not minimize)
            next_population = [list(population[order[0]])] if self.elitism else []
            while len(next_population) < self.pop_size:
                parent1 = list(tournament())
                parent2 = list(tournament())
                child1, child2 = self._permutation_crossover(parent1, parent2, rng)
                next_population.append(self._permutation_mutate(child1, rng))
                if len(next_population) < self.pop_size:
                    next_population.append(self._permutation_mutate(child2, rng))
            population = next_population[: self.pop_size]
            scores = [float(evaluator.fitness(individual, adapter)) for individual in population]
            current_index = min(range(self.pop_size), key=lambda i: scores[i]) if minimize else max(range(self.pop_size), key=lambda i: scores[i])
            if better(scores[current_index], best_score):
                best = list(population[current_index])
                best_score = float(scores[current_index])
            history.append(best_score)

        self._last_iterations = self.generations
        self._last_history = history
        return best, best_score

    def _permutation_crossover(self, parent1, parent2, rng):
        if len(parent1) < 2 or rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        left, right = sorted(rng.integers(0, len(parent1), size=2))
        if left == right:
            right = min(len(parent1), left + 1)
        def ox(a, b):
            child = [None] * len(a)
            child[left:right] = a[left:right]
            remaining = [gene for gene in b if gene not in child[left:right]]
            cursor = 0
            for i in list(range(0, left)) + list(range(right, len(a))):
                child[i] = remaining[cursor]
                cursor += 1
            return child
        return ox(parent1, parent2), ox(parent2, parent1)

    def _permutation_mutate(self, child, rng):
        child = list(child)
        if len(child) > 1 and rng.random() <= self.mutation_rate:
            i, j = rng.choice(len(child), size=2, replace=False)
            child[int(i)], child[int(j)] = child[int(j)], child[int(i)]
        return child

    def _crossover(self, p1, p2, rng, low, high):
        if len(p1) == 1 or rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        point = int(rng.integers(1, len(p1)))
        return (
            np.concatenate([p1[:point], p2[point:]]),
            np.concatenate([p2[:point], p1[point:]]),
        )

    def _mutate(self, child, rng, low, high):
        child = child.copy()
        for i in range(len(child)):
            if rng.random() <= self.mutation_rate:
                span = high[i] - low[i]
                child[i] = np.clip(child[i] + rng.normal(0.0, 0.1 * span if span else 1.0), low[i], high[i])
        return child

    def get_params_report(self) -> Dict[str, Any]:
        return {
            "pop_size": self.pop_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism": self.elitism,
            "engine": "OptimizationLab native GA",
        }
