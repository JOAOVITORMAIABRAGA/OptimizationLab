import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.registry import AlgorithmAvailability, AlgorithmDescriptor, RepresentationCapability
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from domain.expressions import StructuredExpression
from domain.problem import ConstraintSpec, DomainSpec, ObjectiveSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType


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
    return CompatibilityEngine()


def test_continuous_problem_with_pso_is_compatible(engine):
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(make_problem(), registry.get("pso"))
    assert result.status == CompatibilityStatus.COMPATIBLE


def test_continuous_problem_with_de_is_compatible(engine):
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(make_problem(), registry.get("de"))
    assert result.status == CompatibilityStatus.COMPATIBLE


def test_continuous_problem_with_bfo_is_compatible(engine):
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(make_problem(), registry.get("bfo"))
    assert result.status == "compatible"


def test_continuous_problem_with_ga_is_compatible(engine):
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(make_problem(), registry.get("ga"))
    assert result.status == CompatibilityStatus.COMPATIBLE


def test_binary_problem_with_algorithm_without_binary_support_is_incompatible(engine):
    problem = make_problem(variables=[VariableSpec(name="x1", variable_type=VariableType.BINARY, domain=DomainSpec(kind="binary", values=[0, 1]))])
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(problem, registry.get("pso"))
    assert result.status == CompatibilityStatus.INCOMPATIBLE


def test_permutation_problem_with_pso_is_incompatible(engine):
    problem = make_problem(representation=SolutionRepresentationKind.PERMUTATION, variables=[VariableSpec(name=f"p{i}", variable_type=VariableType.DISCRETE, domain=DomainSpec(kind="permutation", values=[0, 1, 2, 3])) for i in range(4)])
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(problem, registry.get("pso"))
    assert result.status == CompatibilityStatus.INCOMPATIBLE


def test_permutation_problem_with_ga_is_incompatible(engine):
    problem = make_problem(representation=SolutionRepresentationKind.PERMUTATION, variables=[VariableSpec(name=f"p{i}", variable_type=VariableType.DISCRETE, domain=DomainSpec(kind="permutation", values=[0, 1, 2, 3])) for i in range(4)])
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(problem, registry.get("ga"))
    assert result.status == CompatibilityStatus.INCOMPATIBLE


def test_permutation_problem_with_aco_is_compatible(engine):
    variables = [
        VariableSpec(name=f"p{i}", variable_type=VariableType.DISCRETE, domain=DomainSpec(kind="permutation", values=[0, 1, 2, 3]))
        for i in range(4)
    ]
    problem = make_problem(
        representation=SolutionRepresentationKind.PERMUTATION,
        variables=variables,
        family=ProblemFamily.ROUTING,
        properties={MathematicalProperty.COMBINATORIAL, MathematicalProperty.DISCRETE, MathematicalProperty.CONSTRAINED},
    )
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(problem, registry.get("aco"))
    assert result.status == CompatibilityStatus.COMPATIBLE


def test_graph_problem_with_aco_is_incompatible(engine):
    problem = make_problem(representation=SolutionRepresentationKind.GRAPH, variables=[VariableSpec(name="g", variable_type=VariableType.DISCRETE, domain=DomainSpec(kind="discrete", values=[0, 1]))])
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(problem, registry.get("aco"))
    assert result.status == CompatibilityStatus.INCOMPATIBLE


def test_problem_with_constraints_and_algorithm_without_support_is_incompatible(engine):
    problem = make_problem(constraints=[ConstraintSpec(id="c1", name="c1", kind="hard", relation="le", expression=None, threshold=1.0)])
    descriptor = AlgorithmDescriptor(
        id="custom",
        name="Custom",
        description="Custom algorithm",
        representation_capabilities=(RepresentationCapability(representation=SolutionRepresentationKind.VECTOR, status="supported", required_operators=(), required_adapters=(), notes=""),),
        supported_variable_types=(VariableType.CONTINUOUS,),
        supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION,),
        supported_mathematical_properties=(MathematicalProperty.CONTINUOUS,),
        supported_objectives=("minimize", "maximize"),
        supports_constraints=False,
        supports_multiobjective=False,
        required_operators=(),
        optional_operators=(),
        required_adapters=(),
        limitations=(),
        implementation_class=None,
        availability=AlgorithmAvailability.UNAVAILABLE,
    )
    result = engine.check(problem, descriptor)
    assert result.status == CompatibilityStatus.INCOMPATIBLE


def test_multiobjective_problem_with_monoobjective_algorithm_is_incompatible(engine):
    objective = ObjectiveSpec(kind="multi", sense="minimize", expression=StructuredExpression(kind="literal", value=1.0))
    problem = make_problem(objective=objective, properties={MathematicalProperty.MULTIOBJECTIVE})
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(problem, registry.get("pso"))
    assert result.status == "incompatible"


def test_multiobjective_problem_with_multiobjective_algorithm_is_compatible(engine):
    objective = ObjectiveSpec(kind="multi", sense="minimize", expression=StructuredExpression(kind="literal", value=1.0))
    problem = make_problem(objective=objective, properties={MathematicalProperty.MULTIOBJECTIVE})
    descriptor = AlgorithmDescriptor(
        id="multi",
        name="Multiobjective",
        description="Multiobjective algorithm",
        representation_capabilities=(RepresentationCapability(representation=SolutionRepresentationKind.VECTOR, status="supported", required_operators=(), required_adapters=(), notes=""),),
        supported_variable_types=(VariableType.CONTINUOUS,),
        supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION,),
        supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.MULTIOBJECTIVE),
        supported_objectives=("minimize", "maximize"),
        supports_constraints=True,
        supports_multiobjective=True,
        required_operators=(),
        optional_operators=(),
        required_adapters=(),
        limitations=(),
        implementation_class=None,
        availability=AlgorithmAvailability.UNAVAILABLE,
    )
    result = engine.check(problem, descriptor)
    assert result.status == CompatibilityStatus.COMPATIBLE


def test_variable_type_incompatibility_is_incompatible(engine):
    problem = make_problem(variables=[VariableSpec(name="cat", variable_type=VariableType.CATEGORICAL, domain=DomainSpec(kind="categorical", categories=["a", "b"]))])
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(problem, registry.get("pso"))
    assert result.status == CompatibilityStatus.INCOMPATIBLE


def test_problem_family_incompatibility_is_incompatible(engine):
    problem = make_problem(family=ProblemFamily.ROUTING)
    from algorithms.registry import AlgorithmRegistry

    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = engine.check(problem, registry.get("pso"))
    assert result.status == CompatibilityStatus.INCOMPATIBLE


def test_required_operator_missing_is_incompatible(engine):
    descriptor = AlgorithmDescriptor(
        id="custom-op",
        name="CustomOp",
        description="Custom op algorithm",
        representation_capabilities=(RepresentationCapability(representation=SolutionRepresentationKind.VECTOR, status="supported", required_operators=("mutation",), required_adapters=(), notes=""),),
        supported_variable_types=(VariableType.CONTINUOUS,),
        supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION,),
        supported_mathematical_properties=(MathematicalProperty.CONTINUOUS,),
        supported_objectives=("minimize", "maximize"),
        supports_constraints=True,
        supports_multiobjective=False,
        required_operators=("crossover",),
        optional_operators=(),
        required_adapters=(),
        limitations=(),
        implementation_class=None,
        availability=AlgorithmAvailability.UNAVAILABLE,
    )
    result = engine.check(make_problem(), descriptor, available_operators=set())
    assert result.status == "incompatible"


def test_adapter_available_is_compatible_with_adaptation(engine):
    descriptor = AlgorithmDescriptor(
        id="adapter-demo",
        name="Adapter Demo",
        description="Adapter-based algorithm",
        representation_capabilities=(RepresentationCapability(representation=SolutionRepresentationKind.PERMUTATION, status="supported_with_adapter", required_operators=(), required_adapters=("permutation_adapter",), notes=""),),
        supported_variable_types=(VariableType.DISCRETE,),
        supported_problem_families=(ProblemFamily.ROUTING,),
        supported_mathematical_properties=(MathematicalProperty.COMBINATORIAL,),
        supported_objectives=("minimize", "maximize"),
        supports_constraints=True,
        supports_multiobjective=False,
        required_operators=(),
        optional_operators=(),
        required_adapters=("permutation_adapter",),
        limitations=(),
        implementation_class=None,
        availability=AlgorithmAvailability.UNAVAILABLE,
    )
    problem = make_problem(representation=SolutionRepresentationKind.PERMUTATION, variables=[VariableSpec(name=f"p{i}", variable_type=VariableType.DISCRETE, domain=DomainSpec(kind="permutation", values=[0, 1, 2, 3])) for i in range(4)], family=ProblemFamily.ROUTING, properties={MathematicalProperty.COMBINATORIAL})
    result = engine.check(problem, descriptor, available_adapters={"permutation_adapter"})
    assert result.status == CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION


def test_adapter_missing_is_incompatible(engine):
    descriptor = AlgorithmDescriptor(
        id="adapter-demo-missing",
        name="Adapter Demo Missing",
        description="Adapter-based algorithm",
        representation_capabilities=(RepresentationCapability(representation=SolutionRepresentationKind.PERMUTATION, status="supported_with_adapter", required_operators=(), required_adapters=("permutation_adapter",), notes=""),),
        supported_variable_types=(VariableType.DISCRETE,),
        supported_problem_families=(ProblemFamily.ROUTING,),
        supported_mathematical_properties=(MathematicalProperty.COMBINATORIAL,),
        supported_objectives=("minimize", "maximize"),
        supports_constraints=True,
        supports_multiobjective=False,
        required_operators=(),
        optional_operators=(),
        required_adapters=("permutation_adapter",),
        limitations=(),
        implementation_class=None,
        availability=AlgorithmAvailability.UNAVAILABLE,
    )
    problem = make_problem(representation=SolutionRepresentationKind.PERMUTATION, variables=[VariableSpec(name=f"p{i}", variable_type=VariableType.DISCRETE, domain=DomainSpec(kind="permutation", values=[0, 1, 2, 3])) for i in range(4)], family=ProblemFamily.ROUTING, properties={MathematicalProperty.COMBINATORIAL})
    result = engine.check(problem, descriptor)
    assert result.status == CompatibilityStatus.INCOMPATIBLE


def test_multiple_reasons_are_collected(engine):
    problem = make_problem(representation=SolutionRepresentationKind.PERMUTATION, variables=[VariableSpec(name="cat", variable_type=VariableType.CATEGORICAL, domain=DomainSpec(kind="categorical", categories=["a", "b"]))], family=ProblemFamily.ROUTING, properties={MathematicalProperty.COMBINATORIAL})
    descriptor = AlgorithmDescriptor(
        id="multi-reason",
        name="Multi Reason",
        description="Multi reason algorithm",
        representation_capabilities=(RepresentationCapability(representation=SolutionRepresentationKind.PERMUTATION, status="unsupported", required_operators=("permutation_crossover",), required_adapters=(), notes=""),),
        supported_variable_types=(VariableType.CONTINUOUS,),
        supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION,),
        supported_mathematical_properties=(MathematicalProperty.CONTINUOUS,),
        supported_objectives=("minimize", "maximize"),
        supports_constraints=False,
        supports_multiobjective=False,
        required_operators=("mutation",),
        optional_operators=(),
        required_adapters=(),
        limitations=(),
        implementation_class=None,
        availability=AlgorithmAvailability.UNAVAILABLE,
    )
    result = engine.check(problem, descriptor)
    assert result.status == "incompatible"
    assert len(result.failed_checks) >= 3
