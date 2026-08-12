import math

from algorithms.de import DifferentialEvolution
from algorithms.ga import GeneticAlgorithm
from algorithms.pso import ParticleSwarmOptimization
from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveKind, ObjectiveSense, ObjectiveSpec
from domain.problem import DomainSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType
from services.universal_optimizer import ObjectiveEvaluator, UniversalProblemAdapter


def v(name):
    return StructuredExpression(kind="variable", name=name)


def lit(value):
    return StructuredExpression(kind="literal", value=value)


def sub(a, b):
    return StructuredExpression(kind="binary", op="sub", args=(a, b))


def mul(a, b):
    return StructuredExpression(kind="binary", op="mul", args=(a, b))


def add(a, b):
    return StructuredExpression(kind="binary", op="add", args=(a, b))


def square(x):
    return mul(x, x)


def black_box_style_problem():
    # Same semantic problem can be passed to different algorithm-specific representations.
    objective = StructuredExpression(
        kind="unary",
        op="neg",
        args=(add(square(sub(v("x"), lit(3))), square(add(v("y"), lit(2)))),),
    )
    variables = [
        VariableSpec("x", VariableType.CONTINUOUS, DomainSpec("continuous", -5, 5), -5, 5),
        VariableSpec("y", VariableType.CONTINUOUS, DomainSpec("continuous", -5, 5), -5, 5),
    ]
    return OptimizationProblem(
        name="quadratic_peak",
        objective=ObjectiveSpec(ObjectiveKind.SINGLE, ObjectiveSense.MAXIMIZE, objective),
        variables=variables,
        problem_family=ProblemFamily.CONTINUOUS_OPTIMIZATION,
        mathematical_properties={MathematicalProperty.CONTINUOUS, MathematicalProperty.NONLINEAR},
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.VECTOR, "numeric vector"),
    )


def test_universal_adapter_decodes_semantic_solution():
    problem = black_box_style_problem()
    adapter = UniversalProblemAdapter(problem)
    candidate = adapter.decode([3.0, -2.0])
    assert candidate.values == {"x": 3.0, "y": -2.0}
    assert candidate.objective_value == 0.0
    assert candidate.feasible is True


def test_integer_vector_representation_normalizes_algorithm_output():
    variables = [VariableSpec("n", VariableType.INTEGER, DomainSpec("integer", 0, 10), 0, 10)]
    problem = OptimizationProblem(
        name="integer_vector",
        objective=ObjectiveSpec(ObjectiveKind.SINGLE, ObjectiveSense.MAXIMIZE, v("n")),
        variables=variables,
        problem_family=ProblemFamily.GENERIC,
        mathematical_properties={MathematicalProperty.INTEGER, MathematicalProperty.LINEAR},
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.VECTOR, "vector"),
    )
    candidate = UniversalProblemAdapter(problem).decode([7.8])
    assert candidate.values == {"n": 8.0}


def test_ga_uses_universal_problem_contract():
    result = GeneticAlgorithm().optimize_problem(black_box_style_problem())
    assert result.feasible is True
    assert result.objective_value > -1.0
    assert math.isclose(result.values["x"], 3.0, abs_tol=1.0)
    assert math.isclose(result.values["y"], -2.0, abs_tol=1.0)


def test_pso_uses_universal_problem_contract():
    result = ParticleSwarmOptimization().optimize_problem(black_box_style_problem())
    assert result.feasible is True
    assert result.objective_value > -2.0


def test_de_uses_universal_problem_contract():
    result = DifferentialEvolution().optimize_problem(black_box_style_problem())
    assert result.feasible is True
    assert result.objective_value > -1.0


def test_native_algorithms_return_standard_result_without_external_metaheuristic_library():
    problem = black_box_style_problem()
    for algorithm in (GeneticAlgorithm(seed=1), ParticleSwarmOptimization(seed=1), DifferentialEvolution(seed=1)):
        result = algorithm.optimize_problem_result(problem)
        assert result.algorithm
        assert result.feasible is True
        assert result.evaluations > 0
        assert result.iterations > 0
        assert len(result.convergence_history) == result.iterations + 1
        assert result.parameters["engine"].startswith("OptimizationLab native")


def test_integer_solution_is_semantic_integer_not_float():
    variables = [VariableSpec("n", VariableType.INTEGER, DomainSpec("integer", 0, 10), 0, 10)]
    problem = OptimizationProblem(
        name="integer_vector",
        objective=ObjectiveSpec(ObjectiveKind.SINGLE, ObjectiveSense.MAXIMIZE, v("n")),
        variables=variables,
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.VECTOR, "vector"),
    )
    candidate = UniversalProblemAdapter(problem).decode([7.8])
    assert candidate.values == {"n": 8}
    assert isinstance(candidate.values["n"], int)


def permutation_problem():
    # Maximize a weighted assignment of four elements to four ordered positions.
    # ACO must discover [3, 2, 1, 0] because the largest element should occupy
    # the largest coefficient.
    variables = [
        VariableSpec(
            f"p{i}",
            VariableType.DISCRETE,
            DomainSpec("permutation", values=[0, 1, 2, 3]),
        )
        for i in range(4)
    ]
    objective = add(
        add(mul(lit(4), v("p0")), mul(lit(3), v("p1"))),
        add(mul(lit(2), v("p2")), v("p3")),
    )
    return OptimizationProblem(
        name="permutation_weighted_assignment",
        objective=ObjectiveSpec(ObjectiveKind.SINGLE, ObjectiveSense.MAXIMIZE, objective),
        variables=variables,
        problem_family=ProblemFamily.ROUTING,
        mathematical_properties={MathematicalProperty.COMBINATORIAL, MathematicalProperty.DISCRETE},
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.PERMUTATION, "permutation"),
    )


def test_permutation_adapter_round_trips_semantic_solution():
    from representations import PermutationRepresentationAdapter
    from domain.solutions import CandidateSolution

    problem = permutation_problem()
    adapter = PermutationRepresentationAdapter(problem)
    candidate = adapter.decode([3, 2, 1, 0])
    assert candidate.values == {"p0": 3, "p1": 2, "p2": 1, "p3": 0}
    assert adapter.encode(candidate) == [3, 2, 1, 0]


def test_aco_uses_native_permutation_adapter():
    from algorithms.aco import AntColonyOptimization

    result = AntColonyOptimization(seed=23).optimize_problem_result(permutation_problem())
    assert result.feasible is True
    assert result.algorithm == "AntColonyOptimization"
    assert result.parameters["engine"] == "OptimizationLab native ACO"
    assert result.objective_value >= 19.0
    assert sorted(result.values.values()) == [0, 1, 2, 3]
    assert result.evaluations > 0
    assert len(result.convergence_history) == result.iterations + 1


def test_remaining_native_algorithms_use_universal_vector_contract():
    from algorithms.bfo import BacterialForagingOptimization
    from algorithms.sa import SimulatedAnnealing
    from algorithms.tabu import TabuSearch
    from algorithms.hill_climbing import HillClimbing

    problem = black_box_style_problem()
    for algorithm in (
        BacterialForagingOptimization(seed=1),
        SimulatedAnnealing(seed=1),
        TabuSearch(seed=1),
        HillClimbing(seed=1),
    ):
        result = algorithm.optimize_problem_result(problem)
        assert result.feasible is True
        assert result.evaluations > 0
        assert result.iterations > 0
        assert len(result.convergence_history) == result.iterations + 1
        assert result.parameters["engine"].startswith("OptimizationLab native")


def test_local_search_algorithms_use_permutation_adapter():
    from algorithms.sa import SimulatedAnnealing
    from algorithms.tabu import TabuSearch
    from algorithms.hill_climbing import HillClimbing

    problem = permutation_problem()
    for algorithm in (SimulatedAnnealing(seed=1), TabuSearch(seed=1), HillClimbing(seed=1)):
        result = algorithm.optimize_problem_result(problem)
        assert result.feasible is True
        assert sorted(result.values.values()) == [0, 1, 2, 3]
        assert result.objective_value == 20.0


def test_universal_decision_engine_selects_without_running_competitors():
    from services.decision_engine import UniversalDecisionEngine

    decision = UniversalDecisionEngine().decide(black_box_style_problem())
    assert decision.selected_algorithm_id == "de"
    assert decision.alternatives
    assert decision.recommendations.recommendations[0].algorithm_id == "de"
    assert "no competing algorithms were executed" in decision.rationale.lower()


def test_universal_decision_engine_prefers_aco_for_permutation_routing():
    from services.decision_engine import UniversalDecisionEngine

    decision = UniversalDecisionEngine().decide(permutation_problem())
    assert decision.selected_algorithm_id == "aco"
    assert decision.alternatives[0].algorithm_id == "tabu"
