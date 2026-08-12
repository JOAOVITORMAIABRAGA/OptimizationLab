import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.base import OptimizationAlgorithm
from algorithms.registry import AlgorithmAvailability, AlgorithmDescriptor, AlgorithmRegistry, RepresentationCapability
from compatibility.compatibility_engine import CompatibilityEngine
from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveKind, ObjectiveSense
from domain.problem import ConstraintSpec, DomainSpec, ObjectiveSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType
from recommendation.recommendation_engine import RecommendationEngine


class DemoAlgorithm(OptimizationAlgorithm):
    def configure(self, config):
        pass

    def optimize(self, fitness_function, bounds, is_minimization=True, constraints=None):
        return [], 0.0

    def get_params_report(self):
        return {}


def make_problem(
    *,
    representation=SolutionRepresentationKind.VECTOR,
    variables=None,
    objective=None,
    family=ProblemFamily.CONTINUOUS_OPTIMIZATION,
    properties=None,
    constraints=None,
):
    if variables is None:
        variables = [
            VariableSpec(name="x1", variable_type=VariableType.CONTINUOUS, domain=DomainSpec(kind="continuous", lower=0.0, upper=1.0))
        ]
    if objective is None:
        objective = ObjectiveSpec(kind="single", sense="minimize", expression=StructuredExpression(kind="literal", value=1.0))
    if properties is None:
        properties = {MathematicalProperty.CONTINUOUS, MathematicalProperty.UNCONSTRAINED}
    return OptimizationProblem(
        name="demo",
        objective=objective,
        variables=variables,
        constraints=constraints or [],
        problem_family=family,
        mathematical_properties=properties,
        solution_representation=SolutionRepresentationSpec(kind=representation, name=representation.value),
    )


@pytest.fixture
def engine():
    return RecommendationEngine()


def test_continuous_problem_with_pso_remains_highly_recommended(engine):
    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.recommend(make_problem(), registry)

    assert result.recommendations
    assert any(rec.algorithm_id == "pso" for rec in result.recommendations)
    pso = next(rec for rec in result.recommendations if rec.algorithm_id == "pso")
    assert pso.score >= 0.75
    assert pso.rank >= 1


def test_tsp_problem_recommends_aco(engine):
    problem = make_problem(
        representation=SolutionRepresentationKind.PERMUTATION,
        variables=[VariableSpec(name=f"p{i}", variable_type=VariableType.DISCRETE, domain=DomainSpec(kind="permutation", values=[0, 1, 2, 3])) for i in range(4)],
        family=ProblemFamily.ROUTING,
        properties={MathematicalProperty.COMBINATORIAL},
    )
    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.recommend(problem, registry)

    assert result.recommendations
    assert result.recommendations[0].algorithm_id == "aco"
    assert any(excluded.algorithm_id == "pso" for excluded in result.excluded_algorithms)
    assert any(excluded.algorithm_id == "ga" for excluded in result.excluded_algorithms)


def test_binary_feature_selection_recommends_compatible_algorithm(engine):
    problem = make_problem(
        representation=SolutionRepresentationKind.VECTOR,
        variables=[VariableSpec(name="feature", variable_type=VariableType.BINARY, domain=DomainSpec(kind="binary", values=[0, 1]))],
        family=ProblemFamily.FEATURE_SELECTION,
        properties={MathematicalProperty.BINARY, MathematicalProperty.DISCRETE},
    )
    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.recommend(problem, registry)

    assert any(rec.algorithm_id == "ga" for rec in result.recommendations)
    ga = next(rec for rec in result.recommendations if rec.algorithm_id == "ga")
    assert ga.score > 0.0


def test_multiobjective_problem_only_recommends_multiobjective_algorithms(engine):
    objective = ObjectiveSpec(kind="multi", sense="minimize", expression=StructuredExpression(kind="literal", value=1.0))
    problem = make_problem(
        objective=objective,
        properties={MathematicalProperty.MULTIOBJECTIVE, MathematicalProperty.CONTINUOUS},
    )
    registry = AlgorithmRegistry.from_builtin_algorithms()
    registry.register(
        AlgorithmDescriptor(
            id="multi_custom",
            name="Multiobjective Demo",
            description="A demo multiobjective implementation",
            representation_capabilities=(
                RepresentationCapability(representation=SolutionRepresentationKind.VECTOR, status="supported", required_operators=(), required_adapters=(), notes=""),
            ),
            supported_variable_types=(VariableType.CONTINUOUS,),
            supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION,),
            supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.MULTIOBJECTIVE),
            supported_objectives=(ObjectiveSense.MINIMIZE, ObjectiveSense.MAXIMIZE),
            supports_constraints=True,
            supports_multiobjective=True,
            required_operators=(),
            optional_operators=(),
            required_adapters=(),
            limitations=(),
            implementation_class=DemoAlgorithm,
            availability=AlgorithmAvailability.AVAILABLE,
        )
    )

    result = engine.recommend(problem, registry)

    assert any(rec.algorithm_id == "multi_custom" for rec in result.recommendations)
    assert all(rec.algorithm_id != "pso" for rec in result.recommendations)


def test_constraints_exclude_algorithms_without_constraint_support(engine):
    problem = make_problem(
        constraints=[ConstraintSpec(id="c1", name="c1", kind="hard", relation="le", expression=None, threshold=1.0)]
    )
    registry = AlgorithmRegistry.from_builtin_algorithms()
    registry.register(
        AlgorithmDescriptor(
            id="no_constraints",
            name="No Constraints",
            description="Algorithm that cannot honor constraints",
            representation_capabilities=(
                RepresentationCapability(representation=SolutionRepresentationKind.VECTOR, status="supported", required_operators=(), required_adapters=(), notes=""),
            ),
            supported_variable_types=(VariableType.CONTINUOUS,),
            supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION,),
            supported_mathematical_properties=(MathematicalProperty.CONTINUOUS,),
            supported_objectives=(ObjectiveSense.MINIMIZE, ObjectiveSense.MAXIMIZE),
            supports_constraints=False,
            supports_multiobjective=False,
            required_operators=(),
            optional_operators=(),
            required_adapters=(),
            limitations=(),
            implementation_class=DemoAlgorithm,
            availability=AlgorithmAvailability.AVAILABLE,
        )
    )

    result = engine.recommend(problem, registry)

    assert any(excluded.algorithm_id == "no_constraints" for excluded in result.excluded_algorithms if excluded.reason)


def test_classical_algorithms_require_explicit_mathematical_structure(engine):
    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.recommend(make_problem(), registry)

    assert not any(rec.algorithm_id == "linear_programming" for rec in result.recommendations)
    assert not any(rec.algorithm_id == "integer_programming" for rec in result.recommendations)
    assert not any(rec.algorithm_id == "constraint_programming" for rec in result.recommendations)


def test_recommendations_are_deterministic(engine):
    registry = AlgorithmRegistry.from_builtin_algorithms()
    problem = make_problem()

    first = engine.recommend(problem, registry)
    second = engine.recommend(problem, registry)

    assert [rec.algorithm_id for rec in first.recommendations] == [rec.algorithm_id for rec in second.recommendations]
    assert [rec.score for rec in first.recommendations] == [rec.score for rec in second.recommendations]


def test_recommendations_are_explainable(engine):
    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.recommend(make_problem(), registry)

    assert result.recommendations
    rec = result.recommendations[0]
    assert rec.rationale
    assert rec.strengths
    assert rec.weaknesses
    assert rec.evidence
