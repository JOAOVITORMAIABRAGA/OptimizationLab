from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from schemas import AlgorithmConfig
from .base import OptimizationAlgorithm
from adapters.solver_adapter import OptimizationProblemAdapter, OrToolsConstraintProgrammingAdapter, ScipyLinearProgrammingAdapter, ScipyMixedIntegerAdapter, cp_model
from domain.problem import OptimizationProblem
from domain.variables import VariableType


class _ClassicalSolver(OptimizationAlgorithm):
    def __init__(self) -> None:
        self.adapter = OptimizationProblemAdapter()
        self.last_backend = ""

    def configure(self, config: Optional[AlgorithmConfig]) -> None:
        return None

    def _solve_problem(self, problem: OptimizationProblem) -> Tuple[List[float], float]:
        model = self.adapter.build(problem)
        return self._solve_model(model)

    def optimize(self, problem: OptimizationProblem, bounds=None, is_minimization=True, constraints=None) -> Tuple[List[float], float]:
        if not isinstance(problem, OptimizationProblem):
            raise TypeError(
                f"{self.__class__.__name__} requires an OptimizationProblem. "
                "Legacy fitness-function execution is intentionally unsupported for exact solvers."
            )
        return self._solve_problem(problem)

    def get_params_report(self) -> Dict[str, Any]:
        return {"engine": self.last_backend}

    def _solve_model(self, model):
        raise NotImplementedError


class LinearProgramming(_ClassicalSolver):
    def _solve_model(self, model):
        self.last_backend = "SciPy HiGHS linear programming"
        return ScipyLinearProgrammingAdapter().solve(model)


class IntegerProgramming(_ClassicalSolver):
    def _solve_model(self, model):
        self.last_backend = "SciPy HiGHS mixed-integer linear programming"
        return ScipyMixedIntegerAdapter().solve(model)


class ConstraintProgramming(_ClassicalSolver):
    def _solve_model(self, model):
        if any(value == 0 for value in model.integrality):
            raise ValueError("ConstraintProgramming backend currently requires integer/binary variables.")
        if cp_model is not None:
            self.last_backend = "OR-Tools CP-SAT"
            return OrToolsConstraintProgrammingAdapter().solve(model)
        self.last_backend = "SciPy HiGHS exact mixed-integer constraint backend"
        return ScipyMixedIntegerAdapter().solve(model)
