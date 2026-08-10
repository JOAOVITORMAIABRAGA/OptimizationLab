import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest

from domain.problem import (
    OptimizationProblem,
    ObjectiveSpec,
    VariableSpec,
    DomainSpec,
    ConstraintSpec,
    DatasetSpec,
    SolutionRepresentationSpec,
)
from domain.variables import VariableType
from domain.problem_family import ProblemFamily, MathematicalProperty
from domain.representations import SolutionRepresentationKind
from domain.expressions import StructuredExpression
from validation.validator import ValidationEngine, ValidationReport


_DEFAULT_OBJECTIVE = object()


def make_problem(
    *,
    objective=_DEFAULT_OBJECTIVE,
    variables=None,
    constraints=None,
    representation=None,
    family=None,
    properties=None,
):
    if objective is _DEFAULT_OBJECTIVE:
        objective = ObjectiveSpec(kind="single", sense="minimize", expression=StructuredExpression(kind="literal", value=1.0))
    if variables is None:
        variables = [
            VariableSpec(name="x1", variable_type=VariableType.CONTINUOUS, domain=DomainSpec(kind="continuous", lower=-5.0, upper=5.0))
        ]
    if representation is None:
        representation = SolutionRepresentationSpec(kind=SolutionRepresentationKind.VECTOR, name="vector")
    if family is None:
        family = ProblemFamily.CONTINUOUS_OPTIMIZATION
    if properties is None:
        properties = {MathematicalProperty.CONTINUOUS, MathematicalProperty.UNCONSTRAINED}

    return OptimizationProblem(
        name="Test Problem",
        objective=objective,
        variables=variables,
        constraints=constraints or [],
        problem_family=family,
        mathematical_properties=properties,
        solution_representation=representation,
        dataset=DatasetSpec(source="memory", format="json", data={}),
    )


def test_continuous_problem_is_valid():
    expr = StructuredExpression(kind="binary", op="add", args=(
        StructuredExpression(kind="binary", op="pow", args=(StructuredExpression(kind="variable", name="x1"), StructuredExpression(kind="literal", value=2.0))),
        StructuredExpression(kind="binary", op="pow", args=(StructuredExpression(kind="variable", name="x2"), StructuredExpression(kind="literal", value=2.0))),
    ))
    objective = ObjectiveSpec(kind="single", sense="minimize", expression=expr)
    variables = [
        VariableSpec(name="x1", variable_type=VariableType.CONTINUOUS, domain=DomainSpec(kind="continuous", lower=-5.0, upper=5.0)),
        VariableSpec(name="x2", variable_type=VariableType.CONTINUOUS, domain=DomainSpec(kind="continuous", lower=-5.0, upper=5.0)),
    ]
    problem = make_problem(objective=objective, variables=variables)
    report = ValidationEngine().validate(problem)
    assert report.is_valid()


def test_binary_problem_is_valid():
    expr = StructuredExpression(kind="function", name="sum", args=(
        StructuredExpression(kind="variable", name="x1"),
        StructuredExpression(kind="variable", name="x2"),
    ))
    objective = ObjectiveSpec(kind="single", sense="maximize", expression=expr)
    variables = [
        VariableSpec(name="x1", variable_type=VariableType.BINARY, domain=DomainSpec(kind="binary", values=[0, 1])),
        VariableSpec(name="x2", variable_type=VariableType.BINARY, domain=DomainSpec(kind="binary", values=[0, 1])),
    ]
    representation = SolutionRepresentationSpec(kind=SolutionRepresentationKind.VECTOR, name="binary_vector")
    problem = make_problem(objective=objective, variables=variables, representation=representation)
    report = ValidationEngine().validate(problem)
    assert report.is_valid()


def test_tsp_permutation_problem_is_valid():
    objective = ObjectiveSpec(kind="single", sense="minimize", expression=StructuredExpression(kind="literal", value=1.0))
    variables = [
        VariableSpec(name="route", variable_type=VariableType.DISCRETE, domain=DomainSpec(kind="permutation", values=["A", "B", "C", "D"]))
    ]
    representation = SolutionRepresentationSpec(kind=SolutionRepresentationKind.PERMUTATION, name="permutation")
    problem = make_problem(
        objective=objective,
        variables=variables,
        representation=representation,
        family=ProblemFamily.ROUTING,
        properties={MathematicalProperty.COMBINATORIAL, MathematicalProperty.DISCRETE, MathematicalProperty.CONSTRAINED},
    )
    report = ValidationEngine().validate(problem)
    assert report.is_valid()


def test_problem_with_variable_without_domain_is_invalid():
    variables = [VariableSpec(name="x1", variable_type=VariableType.CONTINUOUS, domain=None)]
    problem = make_problem(variables=variables)
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()
    assert any("domain" in error.lower() for error in report.errors)


def test_problem_with_inconsistent_bounds_is_invalid():
    variables = [
        VariableSpec(name="x1", variable_type=VariableType.CONTINUOUS, domain=DomainSpec(kind="continuous", lower=5.0, upper=1.0))
    ]
    problem = make_problem(variables=variables)
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()


def test_invalid_constraint_is_invalid():
    constraint = ConstraintSpec(id="c1", name="bad", kind="hard", relation="le", expression=None, threshold=10.0)
    problem = make_problem(constraints=[constraint])
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()


def test_reference_to_unknown_variable_is_invalid():
    expr = StructuredExpression(kind="variable", name="missing")
    objective = ObjectiveSpec(kind="single", sense="minimize", expression=expr)
    problem = make_problem(objective=objective)
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()
    assert any("unknown variable" in error.lower() for error in report.errors)


def test_unsupported_operation_is_invalid():
    expr = StructuredExpression(kind="function", name="__import__", args=(StructuredExpression(kind="literal", value=1.0),))
    objective = ObjectiveSpec(kind="single", sense="minimize", expression=expr)
    problem = make_problem(objective=objective)
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()
    assert any("unsupported" in error.lower() or "forbidden" in error.lower() for error in report.errors)


def test_expression_with_incompatible_types_is_invalid():
    expr = StructuredExpression(kind="binary", op="add", args=(
        StructuredExpression(kind="literal", value=1.0),
        StructuredExpression(kind="literal", value="oops"),
    ))
    objective = ObjectiveSpec(kind="single", sense="minimize", expression=expr)
    problem = make_problem(objective=objective)
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()


def test_expression_attempting_arbitrary_operation_is_invalid():
    expr = StructuredExpression(kind="function", name="eval", args=(StructuredExpression(kind="literal", value="1"),))
    objective = ObjectiveSpec(kind="single", sense="minimize", expression=expr)
    problem = make_problem(objective=objective)
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()


def test_problem_without_objective_is_invalid():
    problem = make_problem(objective=None)
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()


def test_representation_incompatible_with_variables_is_invalid():
    variables = [VariableSpec(name="x1", variable_type=VariableType.CONTINUOUS, domain=DomainSpec(kind="continuous", lower=0.0, upper=1.0))]
    representation = SolutionRepresentationSpec(kind=SolutionRepresentationKind.PERMUTATION, name="permutation")
    problem = make_problem(variables=variables, representation=representation)
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()


def test_domain_incompatible_with_variable_type_is_invalid():
    variables = [VariableSpec(name="x1", variable_type=VariableType.BINARY, domain=DomainSpec(kind="continuous", lower=0.0, upper=1.0))]
    problem = make_problem(variables=variables)
    report = ValidationEngine().validate(problem)
    assert not report.is_valid()
