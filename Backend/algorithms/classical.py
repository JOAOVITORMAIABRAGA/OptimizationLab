import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig


class LinearProgramming(OptimizationAlgorithm):
    def __init__(self):
        self.max_iter: int = 100

    def configure(self, config: AlgorithmConfig) -> None:
        pass

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        rng = np.random.default_rng(41)
        candidate = np.array([rng.uniform(low, high) for low, high in bounds], dtype=float)
        value = float(fitness_function(candidate.tolist()))
        return candidate.tolist(), value

    def get_params_report(self) -> Dict[str, Any]:
        return {"engine": "Linear Programming (heuristic fallback)"}


class IntegerProgramming(OptimizationAlgorithm):
    def __init__(self):
        self.max_iter: int = 100

    def configure(self, config: AlgorithmConfig) -> None:
        pass

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        rng = np.random.default_rng(43)
        candidate = np.array([round(rng.uniform(low, high)) for low, high in bounds], dtype=float)
        value = float(fitness_function(candidate.tolist()))
        return candidate.tolist(), value

    def get_params_report(self) -> Dict[str, Any]:
        return {"engine": "Integer Programming (heuristic fallback)"}


class ConstraintProgramming(OptimizationAlgorithm):
    def __init__(self):
        self.max_iter: int = 100

    def configure(self, config: AlgorithmConfig) -> None:
        pass

    def optimize(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None,
    ) -> Tuple[List[float], float]:
        rng = np.random.default_rng(47)
        candidate = np.array([rng.uniform(low, high) for low, high in bounds], dtype=float)
        if constraints:
            for _ in range(20):
                for constraint_func in constraints:
                    if not constraint_func(candidate.tolist()):
                        candidate += rng.normal(0.0, 0.1, size=len(bounds))
                        candidate = np.clip(candidate, [b[0] for b in bounds], [b[1] for b in bounds])
                        break
        value = float(fitness_function(candidate.tolist()))
        return candidate.tolist(), value

    def get_params_report(self) -> Dict[str, Any]:
        return {"engine": "Constraint Programming (heuristic fallback)"}
