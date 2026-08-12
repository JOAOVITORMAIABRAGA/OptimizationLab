from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.optimize import LinearConstraint as ScipyLinearConstraint
from scipy.sparse import csc_matrix

from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveSense
from domain.problem import ConstraintSpec, OptimizationProblem
from domain.variables import VariableType


class SolverTranslationError(ValueError):
    """Raised when a domain model cannot be represented by a linear solver."""


@dataclass(frozen=True)
class LinearExpression:
    coefficients: Dict[str, float]
    constant: float = 0.0

    def add(self, other: "LinearExpression") -> "LinearExpression":
        coefficients = dict(self.coefficients)
        for name, value in other.coefficients.items():
            coefficients[name] = coefficients.get(name, 0.0) + value
        return LinearExpression({k: v for k, v in coefficients.items() if abs(v) > 1e-15}, self.constant + other.constant)

    def scale(self, factor: float) -> "LinearExpression":
        return LinearExpression({k: v * factor for k, v in self.coefficients.items()}, self.constant * factor)


class LinearExpressionTranslator:
    """Translates the structured expression AST into affine expressions."""

    def translate(self, expression: StructuredExpression) -> LinearExpression:
        if expression.kind == "literal":
            if not isinstance(expression.value, (int, float)) or isinstance(expression.value, bool):
                raise SolverTranslationError("Only numeric literals are supported by the linear solver backend.")
            return LinearExpression({}, float(expression.value))

        if expression.kind == "variable":
            if not expression.name:
                raise SolverTranslationError("Variable expression is missing its name.")
            return LinearExpression({expression.name: 1.0}, 0.0)

        if expression.kind == "unary":
            if expression.op != "neg" or len(expression.args) != 1:
                raise SolverTranslationError(f"Unsupported unary operation '{expression.op}'.")
            return self.translate(expression.args[0]).scale(-1.0)

        if expression.kind == "binary":
            if len(expression.args) != 2:
                raise SolverTranslationError(f"Binary operation '{expression.op}' requires two operands.")
            left = self.translate(expression.args[0])
            right = self.translate(expression.args[1])
            if expression.op == "add":
                return left.add(right)
            if expression.op == "sub":
                return left.add(right.scale(-1.0))
            if expression.op == "mul":
                if left.coefficients and right.coefficients:
                    raise SolverTranslationError("Nonlinear multiplication is not supported by the linear solver backend.")
                return left.scale(right.constant).add(right.scale(left.constant))
            if expression.op == "div":
                if right.coefficients:
                    raise SolverTranslationError("Division by a variable/expression is nonlinear and is not supported.")
                if abs(right.constant) < 1e-15:
                    raise SolverTranslationError("Division by zero is not supported.")
                return left.scale(1.0 / right.constant)
            if expression.op == "pow":
                if right.coefficients or right.constant != 1.0:
                    raise SolverTranslationError("Only power ^1 is affine; nonlinear powers are not supported.")
                return left
            raise SolverTranslationError(f"Unsupported binary operation '{expression.op}'.")

        if expression.kind == "function":
            function_name = expression.op or expression.name
            if function_name == "sum":
                result = LinearExpression({})
                for arg in expression.args:
                    result = result.add(self.translate(arg))
                return result
            raise SolverTranslationError(f"Function '{function_name}' is not supported by the linear solver backend.")

        raise SolverTranslationError(f"Unsupported expression kind '{expression.kind}'.")


@dataclass(frozen=True)
class SolverModel:
    variable_names: Tuple[str, ...]
    objective: np.ndarray
    objective_constant: float
    objective_sense: ObjectiveSense
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    integrality: np.ndarray
    constraint_matrix: np.ndarray
    constraint_lower: np.ndarray
    constraint_upper: np.ndarray


class ClassicalModelAdapter:
    """Adapt the domain model to the backend-neutral linear solver model."""

    def __init__(self) -> None:
        self.translator = LinearExpressionTranslator()

    def build(self, problem: OptimizationProblem) -> SolverModel:
        self._validate_solver_scope(problem)
        names = tuple(variable.name for variable in problem.variables)
        index = {name: i for i, name in enumerate(names)}

        objective_expr = self.translator.translate(problem.objective.expression)
        objective = self._vectorize(objective_expr, index)
        sense = problem.objective.sense
        objective_constant = objective_expr.constant
        if sense == ObjectiveSense.MAXIMIZE:
            objective = -objective
            objective_constant = -objective_constant
        elif sense != ObjectiveSense.MINIMIZE:
            raise SolverTranslationError(f"Unsupported objective sense '{sense}'.")

        lower_bounds: List[float] = []
        upper_bounds: List[float] = []
        integrality: List[int] = []
        for variable in problem.variables:
            low, high = self._variable_bounds(variable)
            lower_bounds.append(low)
            upper_bounds.append(high)
            if variable.variable_type in {VariableType.INTEGER, VariableType.BINARY}:
                integrality.append(1)
            elif variable.variable_type == VariableType.CONTINUOUS:
                integrality.append(0)
            else:
                raise SolverTranslationError(f"Variable type '{variable.variable_type}' is not supported by the linear solver backend.")

        rows: List[np.ndarray] = []
        constraint_lower: List[float] = []
        constraint_upper: List[float] = []
        for constraint in problem.constraints:
            if constraint.kind != "hard":
                raise SolverTranslationError(f"Soft constraint '{constraint.name}' is not supported by exact solver execution.")
            self._append_constraint(constraint, index, rows, constraint_lower, constraint_upper)

        matrix = np.vstack(rows) if rows else np.empty((0, len(names)), dtype=float)
        return SolverModel(
            variable_names=names,
            objective=np.asarray(objective, dtype=float),
            objective_constant=objective_constant,
            objective_sense=sense,
            lower_bounds=np.asarray(lower_bounds, dtype=float),
            upper_bounds=np.asarray(upper_bounds, dtype=float),
            integrality=np.asarray(integrality, dtype=int),
            constraint_matrix=matrix,
            constraint_lower=np.asarray(constraint_lower, dtype=float),
            constraint_upper=np.asarray(constraint_upper, dtype=float),
        )

    def _validate_solver_scope(self, problem: OptimizationProblem) -> None:
        if problem.objective is None or problem.objective.expression is None:
            raise SolverTranslationError("A single explicit objective expression is required.")
        if problem.objective.kind.value != "single":
            raise SolverTranslationError("Multiobjective execution is not implemented by this backend.")
        if problem.solution_representation is None:
            raise SolverTranslationError("A solution representation is required.")
        if problem.solution_representation.kind.value != "vector":
            raise SolverTranslationError("Classical linear solver execution currently requires vector representation.")

    @staticmethod
    def _variable_bounds(variable) -> Tuple[float, float]:
        if variable.variable_type == VariableType.BINARY:
            return 0.0, 1.0
        low = variable.lower_bound
        high = variable.upper_bound
        if low is None and variable.domain is not None:
            low = variable.domain.lower
        if high is None and variable.domain is not None:
            high = variable.domain.upper
        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf
        if low > high:
            raise SolverTranslationError(f"Invalid bounds for variable '{variable.name}'.")
        if variable.variable_type == VariableType.INTEGER and (not np.isfinite(low) or not np.isfinite(high)):
            raise SolverTranslationError(f"Integer variable '{variable.name}' requires finite bounds for this backend.")
        return float(low), float(high)

    def _vectorize(self, expression: LinearExpression, index: Dict[str, int]) -> np.ndarray:
        vector = np.zeros(len(index), dtype=float)
        for name, coefficient in expression.coefficients.items():
            if name not in index:
                raise SolverTranslationError(f"Unknown variable '{name}' in expression.")
            vector[index[name]] = coefficient
        return vector

    def _append_constraint(
        self,
        constraint: ConstraintSpec,
        index: Dict[str, int],
        rows: List[np.ndarray],
        lower: List[float],
        upper: List[float],
    ) -> None:
        if constraint.relation == "bound":
            if constraint.expression is None:
                raise SolverTranslationError(f"Bound constraint '{constraint.name}' requires an expression.")
            expression = self.translator.translate(constraint.expression)
            row = self._vectorize(expression, index)
            rows.append(row)
            lower.append(-np.inf if constraint.lower_bound is None else constraint.lower_bound - expression.constant)
            upper.append(np.inf if constraint.upper_bound is None else constraint.upper_bound - expression.constant)
            return

        if constraint.expression is None:
            raise SolverTranslationError(f"Constraint '{constraint.name}' requires an expression.")
        expression = self.translator.translate(constraint.expression)
        row = self._vectorize(expression, index)
        target = constraint.threshold
        if target is None and constraint.lower_bound is not None and constraint.upper_bound is not None and constraint.lower_bound == constraint.upper_bound:
            target = constraint.lower_bound
        if target is None:
            raise SolverTranslationError(f"Constraint '{constraint.name}' requires a threshold for relation '{constraint.relation}'.")
        target = float(target) - expression.constant
        rows.append(row)
        if constraint.relation == "le":
            lower.append(-np.inf)
            upper.append(target)
        elif constraint.relation == "ge":
            lower.append(target)
            upper.append(np.inf)
        elif constraint.relation == "eq":
            lower.append(target)
            upper.append(target)
        else:
            raise SolverTranslationError(f"Constraint relation '{constraint.relation}' is not supported by the solver backend.")


class ScipyLinearProgrammingAdapter:
    def solve(self, model: SolverModel) -> Tuple[List[float], float]:
        if np.any(model.integrality != 0):
            raise SolverTranslationError("LinearProgramming requires continuous variables only.")
        a_ub = []
        b_ub = []
        a_eq = []
        b_eq = []
        for row, low, high in zip(model.constraint_matrix, model.constraint_lower, model.constraint_upper):
            if np.isfinite(low) and np.isfinite(high) and abs(low - high) <= 1e-12:
                a_eq.append(row)
                b_eq.append(high)
            else:
                if np.isfinite(high):
                    a_ub.append(row)
                    b_ub.append(high)
                if np.isfinite(low):
                    a_ub.append(-row)
                    b_ub.append(-low)
        result = linprog(
            c=model.objective,
            A_ub=np.asarray(a_ub) if a_ub else None,
            b_ub=np.asarray(b_ub) if b_ub else None,
            A_eq=np.asarray(a_eq) if a_eq else None,
            b_eq=np.asarray(b_eq) if b_eq else None,
            bounds=list(zip(model.lower_bounds, model.upper_bounds)),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"LP solver failed: {result.message}")
        solution = [float(value) for value in result.x]
        # The minimization vector may have been negated for MAXIMIZE.
        transformed_objective = float(np.dot(model.objective, result.x) + model.objective_constant)
        objective = -transformed_objective if model.objective_sense == ObjectiveSense.MAXIMIZE else transformed_objective
        if objective == -0.0:
            objective = 0.0
        return solution, objective


try:
    from ortools.sat.python import cp_model
except ImportError:  # Optional runtime backend; SciPy remains a real exact fallback.
    cp_model = None


class OrToolsConstraintProgrammingAdapter:
    def solve(self, model: SolverModel) -> Tuple[List[float], float]:
        if cp_model is None:
            raise RuntimeError("OR-Tools CP-SAT is not installed.")
        solver_model = cp_model.CpModel()
        variables = []
        for name, low, high in zip(model.variable_names, model.lower_bounds, model.upper_bounds):
            if not np.isfinite(low) or not np.isfinite(high):
                raise SolverTranslationError("CP-SAT requires finite bounds for integer variables.")
            if low != int(low) or high != int(high):
                raise SolverTranslationError("CP-SAT integer bounds must be integral.")
            variables.append(solver_model.NewIntVar(int(low), int(high), name))

        for row, low, high in zip(model.constraint_matrix, model.constraint_lower, model.constraint_upper):
            expression = sum(int(coefficient) * variable for coefficient, variable in zip(row, variables))
            if any(abs(coefficient - round(coefficient)) > 1e-12 for coefficient in row):
                raise SolverTranslationError("CP-SAT backend requires integral linear coefficients.")
            if np.isfinite(low) and np.isfinite(high) and low == high:
                solver_model.Add(expression == int(round(low)))
            else:
                if np.isfinite(low):
                    solver_model.Add(expression >= int(round(low)))
                if np.isfinite(high):
                    solver_model.Add(expression <= int(round(high)))

        objective_terms = [int(coefficient) * variable for coefficient, variable in zip(model.objective, variables)]
        if any(abs(coefficient - round(coefficient)) > 1e-12 for coefficient in model.objective):
            raise SolverTranslationError("CP-SAT backend requires integral objective coefficients.")
        objective = sum(objective_terms) + int(round(model.objective_constant))
        if model.objective_sense == ObjectiveSense.MAXIMIZE:
            solver_model.Maximize(objective)
        else:
            solver_model.Minimize(objective)

        solver = cp_model.CpSolver()
        status = solver.Solve(solver_model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"CP-SAT solver failed with status {solver.StatusName(status)}")
        solution = [float(solver.Value(variable)) for variable in variables]
        objective_value = float(sum(coefficient * value for coefficient, value in zip(model.objective, solution)) + model.objective_constant)
        if model.objective_sense == ObjectiveSense.MAXIMIZE:
            objective_value = -objective_value
        return solution, objective_value


class ScipyMixedIntegerAdapter:
    def solve(self, model: SolverModel) -> Tuple[List[float], float]:
        constraints = None
        if model.constraint_matrix.shape[0]:
            constraints = ScipyLinearConstraint(model.constraint_matrix, model.constraint_lower, model.constraint_upper)
        result = milp(
            c=model.objective,
            integrality=model.integrality,
            bounds=Bounds(model.lower_bounds, model.upper_bounds),
            constraints=constraints,
            options={"presolve": True},
        )
        if not result.success:
            raise RuntimeError(f"Mixed-integer solver failed: {result.message}")
        solution = [float(value) for value in result.x]
        transformed_objective = float(np.dot(model.objective, result.x) + model.objective_constant)
        objective = -transformed_objective if model.objective_sense == ObjectiveSense.MAXIMIZE else transformed_objective
        return solution, objective


# Backward-compatible name for callers that imported the pre-refactor class.
OptimizationProblemAdapter = ClassicalModelAdapter
