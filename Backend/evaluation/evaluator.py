from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveMetric, ObjectiveSense
from domain.problem import ConstraintSpec, OptimizationProblem
from domain.solutions import CandidateSolution


MetricHandler = Callable[[CandidateSolution, Any], float]


class ObjectiveMetricRegistry:
    """Registry for semantic objective metrics.

    Algorithms do not know how a semantic objective is evaluated. The metric
    registry maps domain-level metrics to evaluation capabilities exposed by
    adapters. Adding a metric therefore does not require changing solvers.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, MetricHandler] = {}

    def register(self, metric: ObjectiveMetric | str, handler: MetricHandler) -> None:
        key = getattr(metric, "value", metric)
        if key in self._handlers:
            raise ValueError(f"Objective metric '{key}' already registered.")
        self._handlers[key] = handler

    def evaluate(self, metric: ObjectiveMetric | str, candidate: CandidateSolution, adapter: Any) -> float:
        key = getattr(metric, "value", metric)
        handler = self._handlers.get(key)
        if handler is None:
            raise ValueError(f"No evaluator registered for objective metric '{key}'.")
        return float(handler(candidate, adapter))

    def supports(self, metric: ObjectiveMetric | str) -> bool:
        return getattr(metric, "value", metric) in self._handlers


def _adapter_method(name: str, candidate: CandidateSolution, adapter: Any) -> float:
    method = getattr(adapter, name, None)
    if method is None:
        raise ValueError(
            f"The active representation adapter does not expose semantic metric '{name}'."
        )
    representation = adapter.encode(candidate)
    return float(method(representation))


def build_default_metric_registry() -> ObjectiveMetricRegistry:
    registry = ObjectiveMetricRegistry()
    registry.register(ObjectiveMetric.TOUR_LENGTH, lambda candidate, adapter: _adapter_method("route_cost", candidate, adapter))
    registry.register(ObjectiveMetric.TOTAL_DISTANCE, lambda candidate, adapter: _adapter_method("route_cost", candidate, adapter))
    registry.register(ObjectiveMetric.PATH_LENGTH, lambda candidate, adapter: _adapter_method("path_cost", candidate, adapter))
    registry.register(ObjectiveMetric.TOTAL_WEIGHT, lambda candidate, adapter: _adapter_method("edge_set_cost", candidate, adapter))
    return registry


class ObjectiveEvaluator:
    """Evaluate algebraic and semantic objectives through one stable contract."""

    def __init__(self, problem: OptimizationProblem, metric_registry: Optional[ObjectiveMetricRegistry] = None) -> None:
        self.problem = problem
        self.evaluations = 0
        self.metric_registry = metric_registry or build_default_metric_registry()
        objective = problem.objective
        if objective is None:
            raise ValueError("An optimization objective is required.")
        if objective.expression is None and objective.metric is None:
            raise ValueError("Objective requires an expression or semantic metric.")

    def _expression(self, expression: StructuredExpression, values: Dict[str, Any]) -> float:
        kind = expression.kind
        if kind == "literal":
            return float(expression.value)
        if kind == "variable":
            if expression.name not in values:
                raise ValueError(f"Unknown variable '{expression.name}'.")
            return float(values[expression.name])
        if kind == "unary":
            value = self._expression(expression.args[0], values)
            if expression.op == "neg":
                return -value
            raise ValueError(f"Unsupported unary operation '{expression.op}'.")
        if kind == "binary":
            left = self._expression(expression.args[0], values)
            right = self._expression(expression.args[1], values)
            if expression.op == "add": return left + right
            if expression.op == "sub": return left - right
            if expression.op == "mul": return left * right
            if expression.op == "div":
                if abs(right) < 1e-15:
                    raise ValueError("Division by zero.")
                return left / right
            if expression.op == "pow": return left ** right
            if expression.op == "mod": return left % right
            raise ValueError(f"Unsupported binary operation '{expression.op}'.")
        if kind == "function":
            name = expression.op or expression.name
            args = [self._expression(arg, values) for arg in expression.args]
            if name == "sum": return sum(args)
            if name == "abs": return abs(args[0])
            if name == "min": return min(args)
            if name == "max": return max(args)
            raise ValueError(f"Unsupported function '{name}'.")
        raise ValueError(f"Unsupported expression kind '{kind}'.")

    def objective(self, candidate: CandidateSolution, adapter: Any = None) -> float:
        objective = self.problem.objective
        if objective.expression is not None:
            return float(self._expression(objective.expression, candidate.values))
        if objective.metric is None:
            raise ValueError("Objective has neither expression nor semantic metric.")
        if adapter is None:
            raise ValueError("A representation adapter is required for semantic objective evaluation.")
        return self.metric_registry.evaluate(objective.metric, candidate, adapter)

    def is_feasible(self, candidate: CandidateSolution) -> bool:
        for constraint in self.problem.constraints:
            if constraint.kind != "hard" or constraint.scope == "structural":
                continue
            if not self._constraint_is_satisfied(constraint, candidate.values):
                return False
        return True

    def _constraint_is_satisfied(self, constraint: ConstraintSpec, values: Dict[str, Any]) -> bool:
        if constraint.relation == "bound":
            if constraint.expression is None:
                return False
            value = self._expression(constraint.expression, values)
            if constraint.lower_bound is not None and value < constraint.lower_bound - 1e-9:
                return False
            if constraint.upper_bound is not None and value > constraint.upper_bound + 1e-9:
                return False
            return True
        if constraint.expression is None:
            return False
        value = self._expression(constraint.expression, values)
        target = constraint.threshold
        if target is None and constraint.lower_bound is not None and constraint.upper_bound is not None and constraint.lower_bound == constraint.upper_bound:
            target = constraint.lower_bound
        if target is None:
            return False
        if constraint.relation == "le": return value <= target + 1e-9
        if constraint.relation == "ge": return value >= target - 1e-9
        if constraint.relation == "eq": return abs(value - target) <= 1e-9
        return False

    def fitness(self, representation: List[Any], adapter: Any) -> float:
        self.evaluations += 1
        candidate = adapter.decode(representation)
        if not self.is_feasible(candidate):
            return float("inf") if self.problem.objective.sense == ObjectiveSense.MINIMIZE else float("-inf")
        return self.objective(candidate, adapter)

    def fitness_for_adapter(self, adapter: Any, history: Optional[list[float]] = None) -> Callable[[List[Any]], float]:
        def evaluate(representation: List[Any]) -> float:
            value = self.fitness(representation, adapter)
            if history is not None and (not history or value != history[-1]):
                history.append(float(value))
            return value
        return evaluate
