from api import AIDraftProblem, ProblemInput, build_problem
from domain.objectives import ObjectiveMetric, ObjectiveStatus
from domain.representations import SolutionRepresentationKind
from validation.validator import ValidationEngine


def graph_payload(**overrides):
    payload = {
        "name": "Shortest path",
        "description": "Shortest route from A to F.",
        "problem_family": "graph_optimization",
        "mathematical_properties": ["discrete", "combinatorial", "constrained"],
        "variables": [{"name": "route", "variable_type": "discrete"}],
        "objective_kind": "single",
        "objective_sense": "minimize",
        "objective_metric": "path_length",
        "objective_status": "complete",
        "expression": "",
        "constraints": [{
            "id": "edges",
            "name": "Path edges",
            "kind": "hard",
            "relation": "custom",
            "expression": "e1 + e2 + e3",
        }],
        "representation": "graph",
        "representation_metadata": {
            "nodes": ["A", "B", "F"],
            "edges": [
                {"id": "e1", "u": "A", "v": "B", "weight": 5},
                {"id": "e2", "u": "B", "v": "F", "weight": 8},
            ],
            "directed": False,
            "graph_problem_type": "shortest_path",
            "source": "A",
            "target": "F",
        },
        "problem_structure": "graph",
        "problem_structure_metadata": {},
    }
    payload.update(overrides)
    return ProblemInput.model_validate(payload)


def test_native_graph_objective_is_semantic_without_expression():
    problem = build_problem(graph_payload())
    assert problem.objective.metric == ObjectiveMetric.PATH_LENGTH
    assert problem.objective.expression is None
    assert problem.objective.status == ObjectiveStatus.COMPLETE


def test_native_graph_does_not_parse_edge_ids_as_variables():
    problem = build_problem(graph_payload())
    assert problem.constraints[0].expression is None


def test_native_graph_validates_without_unknown_edge_variable_errors():
    problem = build_problem(graph_payload())
    report = ValidationEngine().validate(problem)
    assert report.errors == []


def test_ai_draft_exposes_canonical_objective_fields():
    draft = AIDraftProblem.model_validate({
        "name": "Shortest path",
        "problem_family": "graph_optimization",
        "variables": [{"name": "route", "variable_type": "discrete"}],
        "objective_kind": "single",
        "objective_sense": "minimize",
        "objective_metric": "path_length",
        "objective_status": "complete",
        "expression": "",
        "representation": "graph",
    })
    assert draft.objective_metric == ObjectiveMetric.PATH_LENGTH
    assert draft.objective_status == ObjectiveStatus.COMPLETE
