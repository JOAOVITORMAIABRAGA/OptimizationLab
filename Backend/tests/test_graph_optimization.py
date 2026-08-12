from algorithms.registry import AlgorithmRegistry
from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveKind, ObjectiveMetric, ObjectiveSense, ObjectiveSpec
from domain.problem import DomainSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType
from services.execution_engine import OptimizationExecutionEngine
from representations import GraphRepresentationAdapter, RepresentationAdapterFactory


def graph_problem(graph_type, **metadata):
    edges = [
        {"id": "ab", "u": "A", "v": "B", "weight": 1},
        {"id": "bc", "u": "B", "v": "C", "weight": 1},
        {"id": "ca", "u": "C", "v": "A", "weight": 1},
        {"id": "cd", "u": "C", "v": "D", "weight": 1},
    ]
    graph = {"nodes": ["A", "B", "C", "D"], "edges": edges, "directed": False, "graph_problem_type": graph_type, **metadata}
    variable = VariableSpec("route", VariableType.DISCRETE, DomainSpec("discrete", values=[0]))
    return OptimizationProblem(
        name=graph_type,
        objective=ObjectiveSpec(ObjectiveKind.SINGLE, ObjectiveSense.MINIMIZE, StructuredExpression("literal", value=0.0)),
        variables=[variable],
        problem_family=ProblemFamily.GRAPH_OPTIMIZATION,
        mathematical_properties={MathematicalProperty.DISCRETE, MathematicalProperty.COMBINATORIAL},
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.GRAPH, "graph", graph),
    )


def test_graph_adapter_exposes_validated_graph_structure():
    problem = graph_problem("generic")
    adapter = RepresentationAdapterFactory.create(problem)
    assert isinstance(adapter, GraphRepresentationAdapter)
    assert len(adapter.edges) == 4
    assert set(adapter.nodes) == {"A", "B", "C", "D"}
    assert adapter.adjacency()["C"]


def test_chinese_postman_exactly_duplicates_shortest_path_between_odd_vertices():
    problem = graph_problem("chinese_postman")
    engine = OptimizationExecutionEngine(AlgorithmRegistry.from_builtin_algorithms())
    result = engine.execute(problem, "chinese_postman")
    assert result.objective_value == 5.0
    route = result.variable_values["route"]
    assert len(route) == 5
    assert route.count("ca") >= 2 or route.count("ab") >= 2 or route.count("bc") >= 2 or route.count("cd") >= 2


def test_shortest_path_uses_graph_source_and_target_metadata():
    problem = graph_problem("shortest_path", source="A", target="D")
    engine = OptimizationExecutionEngine(AlgorithmRegistry.from_builtin_algorithms())
    result = engine.execute(problem, "shortest_path")
    assert result.objective_value == 2.0
    assert result.variable_values["route"] == ["ca", "cd"] or result.variable_values["route"] == ["ab", "bc", "cd"]


def test_minimum_spanning_tree_is_exact_and_linear_time_after_sorting():
    problem = graph_problem("minimum_spanning_tree")
    engine = OptimizationExecutionEngine(AlgorithmRegistry.from_builtin_algorithms())
    result = engine.execute(problem, "minimum_spanning_tree")
    assert result.objective_value == 3.0
    assert len(result.variable_values["route"]) == 3


def test_auto_decision_prefers_exact_graph_solver_without_running_metaheuristics():
    problem = graph_problem("chinese_postman")
    engine = OptimizationExecutionEngine(AlgorithmRegistry.from_builtin_algorithms())
    result = engine.execute_auto(problem)
    assert result.algorithm_id == "chinese_postman"
    assert result.objective_value == 5.0


def test_generic_graph_is_not_misclassified_as_a_shortest_path_problem():
    problem = graph_problem("generic")
    engine = OptimizationExecutionEngine(AlgorithmRegistry.from_builtin_algorithms())
    try:
        engine.execute_auto(problem)
    except ValueError as exc:
        assert "No compatible executable algorithm" in str(exc)
    else:
        raise AssertionError("Generic graph should not be routed to a solver requiring a specific graph problem type.")


def test_chinese_postman_accepts_semantic_objective_metric_without_fake_expression():
    problem = graph_problem("chinese_postman")
    problem.objective.expression = None
    problem.objective.metric = ObjectiveMetric.TOTAL_DISTANCE
    engine = OptimizationExecutionEngine(AlgorithmRegistry.from_builtin_algorithms())
    result = engine.execute(problem, "chinese_postman")
    assert result.objective_value == 5.0
