from algorithms.registry import AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from recommendation.recommendation_engine import RecommendationEngine
from services.execution_engine import OptimizationExecutionEngine
from tests.test_v14_graph_routing_adapter import tsp_problem


def test_tsp_exposes_multiple_compatible_candidates_without_algorithm_specific_selection():
    problem = tsp_problem()
    registry = AlgorithmRegistry.from_builtin_algorithms()
    result = RecommendationEngine().recommend(problem, registry)
    ids = [candidate.algorithm_id for candidate in result.candidates]
    assert {"aco", "ga", "sa"}.issubset(ids)
    assert all(candidate.compatibility == CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION for candidate in result.candidates)
    assert all(candidate.adaptation == ("graph_to_permutation",) for candidate in result.candidates)
    assert result.candidates[0].recommended is True
    assert sum(candidate.recommended for candidate in result.candidates) == 1


def test_tsp_can_execute_each_declared_adapted_candidate():
    problem = tsp_problem()
    engine = OptimizationExecutionEngine(AlgorithmRegistry.from_builtin_algorithms())
    for algorithm_id in ("aco", "ga", "sa"):
        result = engine.execute(problem, algorithm_id)
        assert result.algorithm_id == algorithm_id
        assert result.solution
        assert result.objective_value == 16.0


def test_compatibility_produces_explicit_adaptation_plan():
    problem = tsp_problem()
    descriptor = AlgorithmRegistry.from_builtin_algorithms().get("aco")
    result = CompatibilityEngine().check(problem, descriptor)
    assert result.status == CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION
    assert result.adaptation_plan is not None
    assert result.adaptation_plan.source_representation.value == "graph"
    assert result.adaptation_plan.target_representation.value == "permutation"
    assert result.adaptation_plan.adapter_ids == ("graph_to_permutation",)


def test_semantic_objective_does_not_require_an_algebraic_expression():
    problem = tsp_problem()
    assert problem.objective.expression is None
    assert problem.objective.metric.value == "tour_length"
    result = OptimizationExecutionEngine().execute(problem, "sa")
    assert result.objective_value == 16.0


def test_routing_family_does_not_force_a_permutation_representation():
    problem = tsp_problem()
    problem.problem_family = problem.problem_family.ROUTING
    assert CompatibilityEngine().check(problem, AlgorithmRegistry.from_builtin_algorithms().get("aco")).status == CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION
