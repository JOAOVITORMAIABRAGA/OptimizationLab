import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.classical import ConstraintProgramming, IntegerProgramming, LinearProgramming
from algorithms.registry import AlgorithmAvailability, AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveKind, ObjectiveSense
from domain.problem import ConstraintSpec, DomainSpec, ObjectiveSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType
from recommendation.recommendation_engine import RecommendationEngine
from validation.validator import ValidationEngine


def var(name, kind, low, high):
    domain_kind = "binary" if kind == VariableType.BINARY else kind.value
    values = [0, 1] if kind == VariableType.BINARY else None
    return VariableSpec(
        name=name,
        variable_type=kind,
        domain=DomainSpec(kind=domain_kind, lower=low, upper=high, values=values),
        lower_bound=low,
        upper_bound=high,
    )


def v(name):
    return StructuredExpression(kind="variable", name=name)


def lit(value):
    return StructuredExpression(kind="literal", value=value)


def add(*args):
    result = args[0]
    for arg in args[1:]:
        result = StructuredExpression(kind="binary", op="add", args=(result, arg))
    return result


def mul(left, right):
    return StructuredExpression(kind="binary", op="mul", args=(left, right))


def problem(objective, variables, properties, constraints=None, sense=ObjectiveSense.MAXIMIZE):
    return OptimizationProblem(
        name="test",
        objective=ObjectiveSpec(ObjectiveKind.SINGLE, sense, objective),
        variables=variables,
        constraints=constraints or [],
        problem_family=ProblemFamily.CONTINUOUS_OPTIMIZATION,
        mathematical_properties=set(properties),
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.VECTOR, "vector"),
    )


def test_integer_production_planning_is_solved_exactly():
    variables = [var("prod_A", VariableType.INTEGER, 0, 40), var("prod_B", VariableType.INTEGER, 0, 50), var("prod_C", VariableType.INTEGER, 0, 35), var("prod_D", VariableType.INTEGER, 0, 45)]
    objective = add(mul(lit(20), v("prod_A")), mul(lit(15), v("prod_B")), mul(lit(30), v("prod_C")), mul(lit(10), v("prod_D")))
    p = problem(objective, variables, {MathematicalProperty.INTEGER, MathematicalProperty.LINEAR, MathematicalProperty.CONSTRAINED})
    report = ValidationEngine().validate(p)
    assert report.is_valid(), report.errors
    solution, value = IntegerProgramming().optimize(p)
    assert solution == [40.0, 50.0, 35.0, 45.0]
    assert value == 3050.0


def test_linear_programming_solves_continuous_problem():
    variables = [var("x", VariableType.CONTINUOUS, 0, 10), var("y", VariableType.CONTINUOUS, 0, 10)]
    objective = add(mul(lit(20), v("x")), mul(lit(15), v("y")))
    constraint = ConstraintSpec("c1", "capacity", "hard", "le", add(v("x"), v("y")), threshold=10)
    p = problem(objective, variables, {MathematicalProperty.CONTINUOUS, MathematicalProperty.LINEAR, MathematicalProperty.CONSTRAINED}, [constraint])
    solution, value = LinearProgramming().optimize(p)
    assert abs(solution[0] - 10.0) < 1e-7
    assert abs(solution[1]) < 1e-7
    assert abs(value - 200.0) < 1e-7


def test_constraint_programming_solves_binary_constraint_problem():
    variables = [var("x", VariableType.BINARY, 0, 1), var("y", VariableType.BINARY, 0, 1), var("z", VariableType.BINARY, 0, 1)]
    objective = add(mul(lit(10), v("x")), mul(lit(20), v("y")), mul(lit(15), v("z")))
    constraint = ConstraintSpec("c1", "cardinality", "hard", "le", add(v("x"), v("y"), v("z")), threshold=2)
    p = problem(objective, variables, {MathematicalProperty.BINARY, MathematicalProperty.INTEGER, MathematicalProperty.DISCRETE, MathematicalProperty.LINEAR, MathematicalProperty.CONSTRAINED}, [constraint])
    solution, value = ConstraintProgramming().optimize(p)
    assert solution == [0.0, 1.0, 1.0]
    assert value == 35.0


def test_registry_exposes_real_classical_implementations():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    for algorithm_id in ("linear_programming", "integer_programming", "constraint_programming"):
        descriptor = registry.get(algorithm_id)
        assert descriptor.availability == AlgorithmAvailability.AVAILABLE
        assert descriptor.implementation_class is not None
        assert descriptor.get_capability(SolutionRepresentationKind.VECTOR).status == "supported"


def test_integer_programming_is_directly_compatible_and_ranked_first():
    variables = [var("a", VariableType.INTEGER, 0, 10), var("b", VariableType.INTEGER, 0, 10)]
    objective = add(mul(lit(2), v("a")), mul(lit(3), v("b")))
    p = problem(objective, variables, {MathematicalProperty.INTEGER, MathematicalProperty.LINEAR, MathematicalProperty.CONSTRAINED})
    registry = AlgorithmRegistry.from_builtin_algorithms()
    descriptor = registry.get("integer_programming")
    compatibility = CompatibilityEngine().check(p, descriptor)
    assert compatibility.status == CompatibilityStatus.COMPATIBLE
    recommendations = RecommendationEngine().recommend(p, registry)
    assert recommendations.recommendations[0].algorithm_id == "integer_programming"


def test_unsupported_nonlinear_expression_fails_explicitly():
    expression = StructuredExpression(kind="binary", op="mul", args=(v("x"), v("x")))
    p = problem(expression, [var("x", VariableType.CONTINUOUS, 0, 10)], {MathematicalProperty.CONTINUOUS, MathematicalProperty.NONLINEAR})
    try:
        LinearProgramming().optimize(p)
    except ValueError as exc:
        assert "Nonlinear multiplication" in str(exc)
    else:
        raise AssertionError("Nonlinear expressions must not be silently accepted by LP")
