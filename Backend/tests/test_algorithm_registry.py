import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.base import OptimizationAlgorithm
from algorithms.registry import AlgorithmAvailability, AlgorithmDescriptor, AlgorithmRegistry, RepresentationCapability
from domain.objectives import ObjectiveKind, ObjectiveSense
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType


class DemoAlgorithm(OptimizationAlgorithm):
    def configure(self, config):
        pass

    def optimize(self, fitness_function, bounds, is_minimization=True, constraints=None):
        return [], 0.0

    def get_params_report(self):
        return {}


def build_descriptor(**overrides):
    data = {
        "id": "demo",
        "name": "Demo",
        "description": "Demo algorithm",
        "representation_capabilities": [
            RepresentationCapability(
                representation=SolutionRepresentationKind.VECTOR,
                status="supported",
                required_operators=["mutation"],
                required_adapters=[],
                notes="numeric vector",
            )
        ],
        "supported_variable_types": {VariableType.CONTINUOUS},
        "supported_problem_families": {ProblemFamily.CONTINUOUS_OPTIMIZATION},
        "supported_mathematical_properties": {MathematicalProperty.CONTINUOUS},
        "supported_objectives": (ObjectiveSense.MINIMIZE, ObjectiveSense.MAXIMIZE),
        "supports_constraints": True,
        "supports_multiobjective": False,
        "required_operators": ["mutation"],
        "optional_operators": [],
        "required_adapters": [],
        "limitations": [],
        "implementation_class": DemoAlgorithm,
        "availability": AlgorithmAvailability.AVAILABLE,
    }
    data.update(overrides)
    return AlgorithmDescriptor(**data)


def test_all_existing_algorithms_can_be_found_by_id():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    expected_ids = {
        "ga", "pso", "de", "bfo", "sa", "aco", "tabu", "hill_climbing",
        "chinese_postman", "shortest_path", "minimum_spanning_tree",
        "linear_programming", "integer_programming", "constraint_programming",
    }
    assert set(registry.get_all_ids()) == expected_ids


def test_duplicate_ids_are_rejected():
    registry = AlgorithmRegistry()
    registry.register(build_descriptor(id="demo"))
    with pytest.raises(ValueError):
        registry.register(build_descriptor(id="demo"))


def test_unknown_algorithm_raises_key_error():
    registry = AlgorithmRegistry()
    with pytest.raises(KeyError):
        registry.get("missing")


def test_capabilities_have_valid_types():
    capability = RepresentationCapability(
        representation=SolutionRepresentationKind.VECTOR,
        status="supported",
        required_operators=["mutation"],
        required_adapters=[],
        notes="ok",
    )
    assert capability.representation is SolutionRepresentationKind.VECTOR
    assert capability.status in {"supported", "supported_with_adapter", "unsupported"}


def test_invalid_representation_is_rejected():
    with pytest.raises(ValueError):
        AlgorithmDescriptor(
            id="demo",
            name="Demo",
            description="Demo",
            representation_capabilities=[
                RepresentationCapability(representation="not-a-representation", status="supported", required_operators=[], required_adapters=[], notes="")
            ],
            supported_variable_types={VariableType.CONTINUOUS},
            supported_problem_families={ProblemFamily.CONTINUOUS_OPTIMIZATION},
            supported_mathematical_properties={MathematicalProperty.CONTINUOUS},
            supported_objectives=("minimize", "maximize"),
            supports_constraints=True,
            supports_multiobjective=False,
            required_operators=[],
            optional_operators=[],
            required_adapters=[],
            limitations=[],
            implementation_class="DemoAlgorithm",
        )


def test_invalid_descriptor_is_rejected():
    with pytest.raises(ValueError):
        build_descriptor(id="", name="")


def test_registry_rejects_invalid_implementation_class():
    with pytest.raises(ValueError):
        AlgorithmDescriptor(
            id="demo-invalid",
            name="Demo",
            description="Demo",
            representation_capabilities=[
                RepresentationCapability(representation=SolutionRepresentationKind.VECTOR, status="supported", required_operators=(), required_adapters=(), notes="")
            ],
            supported_variable_types=(VariableType.CONTINUOUS,),
            supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION,),
            supported_mathematical_properties=(MathematicalProperty.CONTINUOUS,),
            supported_objectives=(ObjectiveSense.MINIMIZE, ObjectiveSense.MAXIMIZE),
            supports_constraints=True,
            supports_multiobjective=False,
            required_operators=(),
            optional_operators=(),
            required_adapters=(),
            limitations=(),
            implementation_class=str,
            availability=AlgorithmAvailability.AVAILABLE,
        )


def test_registry_validates_structure_and_availability():
    registry = AlgorithmRegistry()
    descriptor = build_descriptor(id="demo-availability", implementation_class=DemoAlgorithm, availability=AlgorithmAvailability.AVAILABLE)
    assert registry.validate_descriptor(descriptor)
    descriptor_unavailable = build_descriptor(id="demo-unavailable", implementation_class=None, availability=AlgorithmAvailability.UNAVAILABLE)
    assert registry.validate_descriptor(descriptor_unavailable)


def test_classical_algorithms_are_available_when_real_backends_exist():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    assert registry.get("linear_programming").availability == AlgorithmAvailability.AVAILABLE
    assert registry.get("integer_programming").availability == AlgorithmAvailability.AVAILABLE
    assert registry.get("constraint_programming").availability == AlgorithmAvailability.AVAILABLE


def test_registry_can_list_all_algorithms():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    assert len(registry.get_all()) == 14


def test_registry_can_filter_by_representation():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    filtered = registry.find_by_representation(SolutionRepresentationKind.VECTOR)
    assert filtered
    assert any(item.id == "ga" for item in filtered)


def test_registry_can_filter_by_problem_family():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    filtered = registry.find_by_problem_family(ProblemFamily.CONTINUOUS_OPTIMIZATION)
    assert filtered
    assert any(item.id == "ga" for item in filtered)


def test_ga_descriptor_reflects_current_implementation_state():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    ga = registry.get("ga")
    vector_capability = ga.get_capability(SolutionRepresentationKind.VECTOR)
    permutation_capability = ga.get_capability(SolutionRepresentationKind.PERMUTATION)
    assert vector_capability is not None and vector_capability.status == "supported"
    assert permutation_capability is not None and permutation_capability.status == "supported"
    graph_capability = ga.get_capability(SolutionRepresentationKind.GRAPH)
    assert graph_capability is not None and graph_capability.status == "supported_with_adapter"
    assert graph_capability.required_adapters == ("graph_to_permutation",)
    assert "crossover" in ga.required_operators and "mutation" in ga.required_operators
    assert ga.availability == AlgorithmAvailability.AVAILABLE


def test_aco_descriptor_supports_permutation_and_not_graph_or_vector():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    aco = registry.get("aco")
    graph_capability = aco.get_capability(SolutionRepresentationKind.GRAPH)
    permutation_capability = aco.get_capability(SolutionRepresentationKind.PERMUTATION)
    vector_capability = aco.get_capability(SolutionRepresentationKind.VECTOR)
    assert graph_capability is not None and graph_capability.status == "supported_with_adapter"
    assert graph_capability.required_adapters == ("graph_to_permutation",)
    assert permutation_capability is not None and permutation_capability.status == "supported"
    assert vector_capability is not None and vector_capability.status == "unsupported"
