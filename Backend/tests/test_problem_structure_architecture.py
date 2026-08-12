from domain.problem import DomainSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.structures import ProblemStructureKind, ProblemStructureSpec
from domain.variables import VariableType
from validation.validator import ValidationEngine
from algorithms.registry import AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from algorithms.graph import DijkstraShortestPath
from algorithms.chinese_postman import ChinesePostmanProblem
from domain.objectives import ObjectiveKind, ObjectiveMetric, ObjectiveSense, ObjectiveSpec
from domain.expressions import StructuredExpression


def _graph_problem(graph_type: str, representation=SolutionRepresentationKind.GRAPH):
    metadata = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"id": "AB", "u": "A", "v": "B", "weight": 1, "required": True},
            {"id": "BC", "u": "B", "v": "C", "weight": 1, "required": True},
            {"id": "CA", "u": "C", "v": "A", "weight": 1, "required": True},
        ],
        "directed": False,
        "graph_problem_type": graph_type,
    }
    if graph_type == "shortest_path":
        metadata.update({"source": "A", "target": "C"})
    objective = ObjectiveSpec(
        kind=ObjectiveKind.SINGLE,
        sense=ObjectiveSense.MINIMIZE,
        expression=StructuredExpression(kind="literal", value=0),
    )
    return OptimizationProblem(
        name=graph_type,
        objective=objective,
        variables=[VariableSpec("route", VariableType.DISCRETE, DomainSpec("discrete", elements=["AB", "BC", "CA"]))],
        problem_family=ProblemFamily.GRAPH_OPTIMIZATION,
        mathematical_properties={MathematicalProperty.DISCRETE, MathematicalProperty.COMBINATORIAL},
        problem_structure=ProblemStructureSpec(ProblemStructureKind.GRAPH, "Graph", metadata),
        solution_representation=SolutionRepresentationSpec(representation, representation.value, {}),
    )


def test_graph_structure_is_not_a_variable_domain():
    problem = _graph_problem("chinese_postman")
    report = ValidationEngine().validate(problem)
    assert report.is_valid(), report.errors


def test_graph_structure_is_distinct_from_solution_representation():
    problem = _graph_problem("shortest_path", SolutionRepresentationKind.GRAPH)
    assert problem.problem_structure.kind == ProblemStructureKind.GRAPH
    assert problem.solution_representation.kind == SolutionRepresentationKind.GRAPH
    # Same display value is allowed for the legacy graph-native representation;
    # the important invariant is that they are different concepts and stored
    # in separate fields.
    assert problem.problem_structure is not problem.solution_representation


def test_graph_exact_algorithms_declare_graph_structure_support():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    cp = registry.get("chinese_postman")
    dijkstra = registry.get("shortest_path")
    assert ProblemStructureKind.GRAPH in cp.supported_problem_structures
    assert ProblemStructureKind.GRAPH in dijkstra.supported_problem_structures


def test_structure_compatibility_rejects_wrong_structure_without_execution():
    registry = AlgorithmRegistry.from_builtin_algorithms()
    descriptor = registry.get("chinese_postman")
    problem = _graph_problem("chinese_postman")
    problem = OptimizationProblem(
        name=problem.name,
        objective=problem.objective,
        variables=problem.variables,
        constraints=problem.constraints,
        problem_family=problem.problem_family,
        mathematical_properties=problem.mathematical_properties,
        problem_structure=ProblemStructureSpec(ProblemStructureKind.TABULAR, "Tabular", {}),
        solution_representation=problem.solution_representation,
    )
    result = CompatibilityEngine().check(problem, descriptor)
    assert result.status == CompatibilityStatus.INCOMPATIBLE
    assert "problem_structure" in result.failed_checks


def test_graph_adapter_reads_new_structure_metadata():
    problem = _graph_problem("shortest_path")
    # No graph metadata is required in the legacy solution representation.
    adapter_problem = OptimizationProblem(
        name=problem.name,
        objective=problem.objective,
        variables=[VariableSpec("route", VariableType.DISCRETE, DomainSpec("discrete", elements=["AB", "BC"]))],
        problem_family=problem.problem_family,
        mathematical_properties=problem.mathematical_properties,
        problem_structure=problem.problem_structure,
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.GRAPH, "Graph", {}),
    )
    from representations import GraphRepresentationAdapter
    adapter = GraphRepresentationAdapter(adapter_problem)
    assert adapter.nodes == ["A", "B", "C"]
    assert len(adapter.edges) == 3
    assert adapter.graph_problem_type == "shortest_path"


def test_graph_completion_exposes_structure_separately():
    from services.llm_service import GroqLLMService
    service = GroqLLMService.__new__(GroqLLMService)
    result = service._complete_graph_model(
        {"representation": "graph", "representation_metadata": {}},
        "Quero encontrar a rota do carteiro chinês.",
        {
            "rows": [
                {"edge": "AB", "from": "A", "to": "B", "distance": 1},
                {"edge": "BA", "from": "B", "to": "A", "distance": 1},
            ],
            "rows_truncated": False,
        },
    )
    assert result["problem_structure"] == "graph"
    assert result["problem_structure_metadata"]["graph_problem_type"] == "chinese_postman"
    assert result["representation"] == "graph"


def test_graph_problem_can_have_metric_objective_without_algebraic_expression():
    problem = _graph_problem("chinese_postman")
    problem.objective.expression = None
    problem.objective.metric = ObjectiveMetric.TOTAL_DISTANCE
    report = ValidationEngine().validate(problem)
    assert report.is_valid(), report.errors


def test_graph_completion_declares_semantic_objective_metric():
    from services.llm_service import GroqLLMService
    service = GroqLLMService.__new__(GroqLLMService)
    result = service._complete_graph_model(
        {"representation": "graph", "representation_metadata": {}},
        "Quero resolver o Carteiro Chinês minimizando a distância total.",
        {"rows": [{"edge": "AB", "from": "A", "to": "B", "distance": 2}], "rows_truncated": False},
    )
    assert result["objective_metric"] == "total_distance"
    assert result["expression"] == ""
