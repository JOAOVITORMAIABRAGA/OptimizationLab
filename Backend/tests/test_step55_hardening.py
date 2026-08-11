import ast
from pathlib import Path

import pytest

from algorithms.registry import AlgorithmAvailability, AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveComponent, ObjectiveKind, ObjectiveSense, ObjectiveSpec
from domain.problem import ConstraintSpec, DomainSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType
from operators import OperatorRegistry
from services.orchestrator import OptimizationOrchestrator
from validation.validator import ValidationEngine


BACKEND = Path(__file__).resolve().parents[1]


def make_problem(**kwargs):
    variables = kwargs.pop("variables", [
        VariableSpec("x", VariableType.CONTINUOUS, DomainSpec("continuous", lower=0.0, upper=1.0)),
    ])
    objective = kwargs.pop("objective", ObjectiveSpec(ObjectiveKind.SINGLE, ObjectiveSense.MINIMIZE, StructuredExpression("variable", name="x")))
    return OptimizationProblem(
        name="hardening",
        objective=objective,
        variables=variables,
        constraints=kwargs.pop("constraints", []),
        problem_family=kwargs.pop("family", ProblemFamily.CONTINUOUS_OPTIMIZATION),
        mathematical_properties=kwargs.pop("properties", {MathematicalProperty.CONTINUOUS}),
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.VECTOR, "vector"),
        **kwargs,
    )


def test_source_contains_no_dynamic_python_execution_calls():
    forbidden = {"exec", "eval", "compile"}
    violations = []
    for path in BACKEND.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden:
                violations.append((str(path), node.func.id, node.lineno))
    assert violations == []


def test_legacy_orchestrator_execution_is_disabled():
    orchestrator = object.__new__(OptimizationOrchestrator)
    with pytest.raises(RuntimeError, match="disabled"):
        orchestrator.execute_solution(object())


def test_optimization_problem_generates_non_empty_unique_ids():
    first = make_problem()
    second = make_problem()
    assert first.id and second.id and first.id != second.id


def test_multiobjective_requires_independent_objectives_and_preserves_each_expression():
    objective = ObjectiveSpec(
        kind=ObjectiveKind.MULTI,
        objectives=(
            ObjectiveComponent("cost", "Cost", ObjectiveSense.MINIMIZE, StructuredExpression("variable", name="x")),
            ObjectiveComponent("quality", "Quality", ObjectiveSense.MAXIMIZE, StructuredExpression("literal", value=2.0)),
        ),
    )
    problem = make_problem(objective=objective, properties={MathematicalProperty.CONTINUOUS, MathematicalProperty.MULTIOBJECTIVE})
    report = ValidationEngine().validate(problem)
    assert report.is_valid()
    assert len(problem.objective.objectives) == 2


def test_multiobjective_with_only_one_objective_is_invalid():
    objective = ObjectiveSpec(
        kind=ObjectiveKind.MULTI,
        objectives=(ObjectiveComponent("only", "Only", ObjectiveSense.MINIMIZE, StructuredExpression("literal", value=1.0)),),
    )
    assert not ValidationEngine().validate(make_problem(objective=objective)).is_valid()


def test_constraint_model_is_validated_without_execution():
    constraint = ConstraintSpec(
        id="c1",
        name="x upper bound",
        kind="hard",
        relation="le",
        expression=StructuredExpression("variable", name="x"),
        threshold=0.8,
    )
    problem = make_problem(constraints=[constraint], properties={MathematicalProperty.CONTINUOUS, MathematicalProperty.CONSTRAINED})
    report = ValidationEngine().validate(problem)
    assert report.is_valid()


def test_builtin_available_descriptors_have_structural_implementations():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    assert registry.validate() == []
    for descriptor in registry.get_all():
        if descriptor.availability == AlgorithmAvailability.AVAILABLE:
            assert descriptor.implementation_class is not None
            assert registry.validate_implementation(descriptor.implementation_class)


def test_numeric_algorithms_do_not_claim_integer_or_binary_support():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    for algorithm_id in ("ga", "pso", "de", "bfo", "sa", "aco", "tabu", "hill_climbing"):
        descriptor = registry.get(algorithm_id)
        assert VariableType.INTEGER not in descriptor.supported_variable_types
        assert VariableType.BINARY not in descriptor.supported_variable_types


def test_aco_is_honest_about_graph_and_permutation_support():
    aco = AlgorithmRegistry.from_builtin_algorithms().get("aco")
    assert aco.get_capability(SolutionRepresentationKind.GRAPH) is None or aco.get_capability(SolutionRepresentationKind.GRAPH).status == "unsupported"
    assert aco.get_capability(SolutionRepresentationKind.PERMUTATION) is None or aco.get_capability(SolutionRepresentationKind.PERMUTATION).status == "unsupported"
    assert aco.get_capability(SolutionRepresentationKind.VECTOR).status == "supported"


def test_operator_registry_is_runtime_source_of_truth():
    operators = OperatorRegistry.builtin()
    assert operators.has("mutation")
    assert operators.has("crossover")
    assert not operators.has("permutation_crossover")


def test_missing_runtime_operator_is_incompatible():
    descriptor = AlgorithmRegistry.from_builtin_algorithms().get("ga")
    problem = make_problem()
    result = CompatibilityEngine().check(problem, descriptor, available_operators={"mutation", "selection"})
    assert result.status == CompatibilityStatus.INCOMPATIBLE
    assert "required_operators" in result.failed_checks


def test_api_builder_transports_constraints_into_domain_model():
    from api import ConstraintInput, ProblemInput, VariableInput, build_problem

    payload = ProblemInput(
        name="constrained",
        variables=[VariableInput(name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0, upper_bound=1)],
        expression="x",
        constraints=[ConstraintInput(id="c1", name="limit", kind="hard", relation="le", expression="x", threshold=0.5)],
    )
    problem = build_problem(payload)
    assert len(problem.constraints) == 1
    assert problem.constraints[0].threshold == 0.5
    assert ValidationEngine().validate(problem).is_valid()


def test_api_builder_transports_multiple_objectives_into_domain_model():
    from api import ObjectiveInput, ProblemInput, VariableInput, build_problem

    payload = ProblemInput(
        name="multi",
        objective_kind=ObjectiveKind.MULTI,
        objectives=[
            ObjectiveInput(id="o1", name="cost", sense=ObjectiveSense.MINIMIZE, expression="x"),
            ObjectiveInput(id="o2", name="quality", sense=ObjectiveSense.MAXIMIZE, expression="x + 1"),
        ],
        variables=[VariableInput(name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0, upper_bound=1)],
    )
    problem = build_problem(payload)
    assert [item.id for item in problem.objective.objectives] == ["o1", "o2"]
    assert ValidationEngine().validate(problem).is_valid()


def test_existing_analysis_pipeline_stops_at_recommendation():
    from api import ProblemInput, VariableInput, analyze

    payload = ProblemInput(
        name="integration",
        variables=[VariableInput(name="x", variable_type=VariableType.CONTINUOUS, lower_bound=0, upper_bound=1)],
        expression="x * x",
    )
    response = analyze(payload)
    assert response.validation.valid
    assert response.recommendations
    assert not hasattr(response, "execution")
