from api import ProblemInput, build_problem
from adapters.problem_adapters import BUILTIN_ADAPTERS, GraphRoutingAdapter
from algorithms.registry import AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from services.execution_engine import OptimizationExecutionEngine


def tsp_problem():
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
        "name": "TSP",
        "problem_family": "graph_optimization",
        "mathematical_properties": ["discrete", "combinatorial", "constrained"],
        "variables": [{"name": "route", "variable_type": "discrete"}],
        "objective_kind": "single",
        "objective_sense": "minimize",
        "objective_metric": "tour_length",
        "objective_status": "complete",
        "expression": "",
        "representation": "graph",
        "representation_metadata": {
            "nodes": nodes,
            "edges": edges,
            "directed": False,
            "graph_problem_type": "tsp",
        },
        "problem_structure": "graph",
        "problem_structure_metadata": {},
    })
    return build_problem(payload)


def test_tsp_uses_graph_to_permutation_adapter():
    problem = tsp_problem()
    adapter = GraphRoutingAdapter(problem)
    assert adapter.size == 4
    assert adapter.route_cost([0, 1, 2, 3]) == 16


def test_aco_is_compatible_with_tsp_through_adapter():
    problem = tsp_problem()
    descriptor = AlgorithmRegistry.from_builtin_algorithms().get("aco")
    result = CompatibilityEngine().check(problem, descriptor, available_adapters=BUILTIN_ADAPTERS)
    assert result.status == CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION
    assert result.required_adapters == ("graph_to_permutation",)


def test_aco_executes_tsp_through_adapter():
    result = OptimizationExecutionEngine().execute(tsp_problem(), "aco")
    assert result.algorithm_id == "aco"
    assert result.objective_value == 16
    assert len(result.variable_values["route"]) == 4
    assert set(result.variable_values["route"]) == {"A", "B", "C", "D"}
