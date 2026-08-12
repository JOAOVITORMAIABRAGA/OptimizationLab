from services.llm_service import GroqLLMService


def test_graph_completion_detects_hamiltonian_tsp_and_normalizes_metric():
    service = GroqLLMService.__new__(GroqLLMService)
    result = service._complete_graph_model(
        {
            "problem_family": "graph_optimization",
            "representation": "graph",
            "variables": [{"name": "edge_selected", "variable_type": "binary"}],
            "objective_sense": "minimize",
            "objective_metric": "total_distance",
            "objective_status": "complete",
            "constraints": [],
            "representation_metadata": {},
            "problem_structure_metadata": {},
            "description": "The user wants a tour that visits each city once and returns to the origin while minimizing the sum of distances.",
        },
        "The user wants a tour that visits each city once and returns to the origin while minimizing the sum of distances.",
        {
            "rows": [
                {"from": "A", "to": "B", "distance": 2},
                {"from": "A", "to": "C", "distance": 9},
                {"from": "A", "to": "D", "distance": 7},
                {"from": "B", "to": "C", "distance": 3},
                {"from": "B", "to": "D", "distance": 6},
                {"from": "C", "to": "D", "distance": 4},
            ]
        },
    )
    assert result["problem_structure_metadata"]["graph_problem_type"] == "tsp"
    assert result["objective_metric"] == "tour_length"
    assert result["objective_status"] == "complete"


def test_tsp_semantic_completion_is_case_insensitive():
    service = GroqLLMService.__new__(GroqLLMService)
    result = service._complete_graph_model(
        {
            "problem_family": "graph_optimization",
            "representation": "graph",
            "objective_sense": "minimize",
            "objective_metric": "total_distance",
            "objective_status": "complete",
            "constraints": [],
            "representation_metadata": {},
            "problem_structure_metadata": {},
            "description": "A HAMILTONIAN CYCLE must visit every city once.",
        },
        "A HAMILTONIAN CYCLE must visit every city once.",
        {"rows": [
            {"from": "A", "to": "B", "distance": 1},
            {"from": "A", "to": "C", "distance": 2},
            {"from": "B", "to": "C", "distance": 3},
        ]},
    )
    assert result["problem_structure_metadata"]["graph_problem_type"] == "tsp"
    assert result["objective_metric"] == "tour_length"
