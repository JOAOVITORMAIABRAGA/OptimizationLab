import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.classical import IntegerProgramming, ConstraintProgramming
from algorithms.ga import GeneticAlgorithm
from algorithms.registry import AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveKind, ObjectiveSense, ObjectiveSpec
from domain.problem import ConstraintSpec, DomainSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType
from recommendation.recommendation_engine import RecommendationEngine
from services.decision_engine import UniversalDecisionEngine


def v(name):
    return StructuredExpression(kind="variable", name=name)


def lit(value):
    return StructuredExpression(kind="literal", value=value)


def add(*args):
    result = args[0]
    for arg in args[1:]:
        result = StructuredExpression(kind="binary", op="add", args=(result, arg))
    return result


def mul(a, b):
    return StructuredExpression(kind="binary", op="mul", args=(a, b))


def make_knapsack():
    data = [("A", 40, 80), ("B", 35, 72), ("C", 30, 65), ("D", 25, 55), ("E", 20, 42), ("F", 15, 30)]
    variables = [
        VariableSpec(name=name, variable_type=VariableType.BINARY,
                     domain=DomainSpec(kind="binary", values=[0, 1]), lower_bound=0, upper_bound=1)
        for name, _, _ in data
    ]
    objective = add(*(mul(lit(ret), v(name)) for name, _, ret in data))
    budget = add(*(mul(lit(cost), v(name)) for name, cost, _ in data))
    constraint = ConstraintSpec("budget", "budget", "hard", "le", budget, threshold=100)
    return OptimizationProblem(
        name="binary_project_selection",
        objective=ObjectiveSpec(ObjectiveKind.SINGLE, ObjectiveSense.MAXIMIZE, objective),
        variables=variables,
        constraints=[constraint],
        problem_family=ProblemFamily.PORTFOLIO_OPTIMIZATION,
        mathematical_properties={
            MathematicalProperty.BINARY,
            MathematicalProperty.DISCRETE,
            MathematicalProperty.COMBINATORIAL,
            MathematicalProperty.LINEAR,
            MathematicalProperty.CONSTRAINED,
        },
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.VECTOR, "binary_vector"),
    )


def test_binary_knapsack_has_compatible_exact_solver():
    problem = make_knapsack()
    registry = AlgorithmRegistry.from_builtin_algorithms()
    descriptor = registry.get("integer_programming")
    result = CompatibilityEngine().check(problem, descriptor)
    assert result.status == CompatibilityStatus.COMPATIBLE, result


def test_binary_knapsack_auto_selects_integer_programming():
    problem = make_knapsack()
    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = RecommendationEngine().recommend(problem, registry)
    assert result.recommendations
    assert result.recommendations[0].algorithm_id == "integer_programming"


def test_binary_knapsack_exact_solution_is_207():
    problem = make_knapsack()
    solution, value = IntegerProgramming().optimize(problem)
    assert value == 209.0
    assert solution == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]


def test_binary_knapsack_decision_engine_is_fast_structural_selection():
    problem = make_knapsack()
    decision = UniversalDecisionEngine().decide(problem)
    assert decision.selected_algorithm_id == "integer_programming"
    assert decision.recommendations.recommendations


def test_binary_ga_can_execute_and_decode_only_zero_or_one():
    problem = make_knapsack()
    result = GeneticAlgorithm(seed=3).optimize_problem_result(problem)
    values = result.solution.values
    assert all(value in (0, 1) for value in values.values())
    assert result.feasible


def test_binary_constraint_programming_is_compatible_without_integer_property():
    problem = make_knapsack()
    problem.mathematical_properties.discard(MathematicalProperty.COMBINATORIAL)
    problem.mathematical_properties.discard(MathematicalProperty.DISCRETE)
    descriptor = AlgorithmRegistry.from_builtin_algorithms().get("constraint_programming")
    result = CompatibilityEngine().check(problem, descriptor)
    assert result.status == CompatibilityStatus.COMPATIBLE, result


def test_binary_knapsack_budget_constraint_is_preserved_by_problem_builder():
    from api import ProblemInput, build_problem
    payload = ProblemInput.model_validate({
        "name": "binary_project_selection",
        "problem_family": "portfolio_optimization",
        "mathematical_properties": ["binary", "linear", "constrained"],
        "variables": [
            {"name": name, "variable_type": "binary", "lower_bound": 0, "upper_bound": 1}
            for name in ["select_A", "select_B", "select_C", "select_D", "select_E", "select_F"]
        ],
        "objective_sense": "maximize",
        "expression": "80*select_A + 72*select_B + 65*select_C + 55*select_D + 42*select_E + 30*select_F",
        "constraints": [{
            "id": "budget_limit",
            "name": "Budget limit",
            "kind": "hard",
            "relation": "le",
            "expression": "40*select_A + 35*select_B + 30*select_C + 25*select_D + 20*select_E + 15*select_F",
            "lower_bound": None,
            "upper_bound": None,
            "threshold": 100,
        }],
        "representation": "vector",
    })
    problem = build_problem(payload)
    assert len(problem.constraints) == 1
    assert problem.constraints[0].threshold == 100
    solution, value = IntegerProgramming().optimize(problem)
    assert value == 209.0
    assert solution == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
