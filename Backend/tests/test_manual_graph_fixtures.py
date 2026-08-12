from __future__ import annotations

from unittest.mock import patch

from algorithms.registry import AlgorithmRegistry
from api import ProblemInput, build_problem
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from services.execution_engine import OptimizationExecutionEngine
from services.llm_service import GroqLLMService


MST_DESCRIPTION = (
    "Quero conectar todos os pontos de uma rede gastando o mínimo possível de cabo. "
    "Cada linha da planilha é uma possível conexão entre dois pontos e informa o comprimento do cabo. "
    "Todos os pontos precisam ficar conectados."
)
MST_ROWS = [
    {"edge": "AB", "from": "A", "to": "B", "distance": 4},
    {"edge": "AC", "from": "A", "to": "C", "distance": 3},
    {"edge": "BC", "from": "B", "to": "C", "distance": 1},
    {"edge": "BD", "from": "B", "to": "D", "distance": 2},
    {"edge": "CD", "from": "C", "to": "D", "distance": 4},
    {"edge": "CE", "from": "C", "to": "E", "distance": 2},
    {"edge": "DE", "from": "D", "to": "E", "distance": 3},
    {"edge": "DF", "from": "D", "to": "F", "distance": 2},
    {"edge": "EF", "from": "E", "to": "F", "distance": 5},
]

TSP_DESCRIPTION = (
    "Tenho uma lista de cidades e quero descobrir uma ordem para visitar todas elas uma vez "
    "e voltar para a primeira cidade, tentando minimizar a distância total. "
    "As distâncias entre as cidades estão na planilha."
)
TSP_ROWS = [
    {"from": "A", "to": "B", "distance": 10},
    {"from": "A", "to": "C", "distance": 15},
    {"from": "A", "to": "D", "distance": 20},
    {"from": "B", "to": "C", "distance": 35},
    {"from": "B", "to": "D", "distance": 25},
    {"from": "C", "to": "D", "distance": 30},
]


def _canonical(description, rows):
    service = GroqLLMService.__new__(GroqLLMService)
    model = service._deterministic_graph_fallback(description, {"rows": rows})
    model = service._complete_graph_model(model, description, {"rows": rows})
    return build_problem(ProblemInput.model_validate(model))


def test_manual_05_mst_is_classified_and_compatible():
    problem = _canonical(MST_DESCRIPTION, MST_ROWS)
    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = CompatibilityEngine().check(problem, registry.get("minimum_spanning_tree"))
    assert problem.problem_structure.metadata["graph_problem_type"] == "minimum_spanning_tree"
    assert result.status == CompatibilityStatus.COMPATIBLE

    execution = OptimizationExecutionEngine(registry).execute(problem, "minimum_spanning_tree")
    assert execution.objective_value == 10.0
    assert set(execution.solution) == {"BC", "BD", "CE", "DF", "AC"}


def test_manual_07_tsp_exposes_all_expected_adapted_candidates():
    problem = _canonical(TSP_DESCRIPTION, TSP_ROWS)
    registry = AlgorithmRegistry.from_builtin_algorithms()
    engine = CompatibilityEngine()

    assert problem.problem_structure.metadata["graph_problem_type"] == "tsp"
    for algorithm_id in ("aco", "ga", "sa", "tabu", "hill_climbing"):
        result = engine.check(problem, registry.get(algorithm_id))
        assert result.status == CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION
        assert result.adaptation_plan is not None
        assert result.adaptation_plan.adapter_ids == ("graph_to_permutation",)


def test_graph_modeling_has_provider_failure_fallback():
    service = GroqLLMService.__new__(GroqLLMService)
    with patch.object(service, "_request_model", side_effect=ValueError("json_validate_failed")):
        result = service.draft_model(MST_DESCRIPTION, {"rows": MST_ROWS})
    assert result["problem_structure_metadata"]["graph_problem_type"] == "minimum_spanning_tree"
    assert result["objective_metric"] == "total_weight"
