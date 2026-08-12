from api import ProblemInput, build_problem
from adapters.problem_adapters import BUILTIN_ADAPTERS
from algorithms.registry import AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus


def binary_tsp_problem():
    nodes = ["A", "B", "C", "D"]
    edges = [
        {"id": "AB", "u": "A", "v": "B", "weight": 2},
        {"id": "AC", "u": "A", "v": "C", "weight": 9},
        {"id": "AD", "u": "A", "v": "D", "weight": 7},
        {"id": "BC", "u": "B", "v": "C", "weight": 3},
        {"id": "BD", "u": "B", "v": "D", "weight": 6},
        {"id": "CD", "u": "C", "v": "D", "weight": 4},
    ]
    payload = ProblemInput.model_validate({
        "name": "TSP binary encoding",
        "problem_family": "graph_optimization",
        "mathematical_properties": ["discrete", "binary", "combinatorial", "constrained"],
        "variables": [{"name": "edge_selected", "variable_type": "binary", "lower_bound": 0, "upper_bound": 1}],
        "objective_kind": "single",
        "objective_sense": "minimize",
        "objective_metric": "tour_length",
        "objective_status": "complete",
        "expression": "",
        "representation": "graph",
        "representation_metadata": {"nodes": nodes, "edges": edges, "directed": False, "graph_problem_type": "tsp"},
        "problem_structure": "graph",
        "problem_structure_metadata": {},
    })
    return build_problem(payload)


def test_adapter_boundary_ignores_source_binary_encoding_for_aco():
    problem = binary_tsp_problem()
    descriptor = AlgorithmRegistry.from_builtin_algorithms().get("aco")
    result = CompatibilityEngine().check(problem, descriptor, available_adapters=BUILTIN_ADAPTERS)
    assert result.status == CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION
    assert result.required_adapters == ("graph_to_permutation",)
    assert "variable_types" not in result.failed_checks
    assert "mathematical_properties" not in result.failed_checks


def test_adapter_boundary_rejects_wrong_graph_semantics():
    problem = binary_tsp_problem()
    problem.problem_structure.metadata["graph_problem_type"] = "shortest_path"
    descriptor = AlgorithmRegistry.from_builtin_algorithms().get("aco")
    result = CompatibilityEngine().check(problem, descriptor, available_adapters=BUILTIN_ADAPTERS)
    assert result.status == CompatibilityStatus.INCOMPATIBLE
    assert result.status == CompatibilityStatus.INCOMPATIBLE
    assert "representation_adapter" in result.failed_checks
