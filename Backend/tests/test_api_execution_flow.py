from fastapi.testclient import TestClient

from api import app


def production_payload():
    return {
        "name": "production_planning",
        "description": "Decide how much of each product to produce to maximize profit.",
        "problem_family": "production_planning",
        "mathematical_properties": ["integer", "linear", "constrained"],
        "variables": [
            {"name": "produce_qty_A", "variable_type": "integer", "lower_bound": 0, "upper_bound": 40},
            {"name": "produce_qty_B", "variable_type": "integer", "lower_bound": 0, "upper_bound": 50},
            {"name": "produce_qty_C", "variable_type": "integer", "lower_bound": 0, "upper_bound": 35},
            {"name": "produce_qty_D", "variable_type": "integer", "lower_bound": 0, "upper_bound": 45},
        ],
        "objective_kind": "single",
        "objective_sense": "maximize",
        "expression": "20*produce_qty_A + 15*produce_qty_B + 30*produce_qty_C + 10*produce_qty_D",
        "representation": "vector",
    }


def test_analyze_accepts_complete_model():
    response = TestClient(app).post("/api/analyze", json=production_payload())
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["validation"]["valid"] is True
    assert data["recommendations"]


def test_solve_returns_named_variables():
    payload = production_payload()
    payload["algorithm_id"] = "integer_programming"
    response = TestClient(app).post("/api/solve", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["variable_values"] == {
        "produce_qty_A": 40.0,
        "produce_qty_B": 50.0,
        "produce_qty_C": 35.0,
        "produce_qty_D": 45.0,
    }
    assert data["objective_value"] == 3050.0


def test_aco_executes_permutation_problem():
    from algorithms.registry import AlgorithmRegistry
    from domain.expressions import StructuredExpression
    from domain.objectives import ObjectiveKind, ObjectiveSense, ObjectiveSpec
    from domain.problem import DomainSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
    from domain.problem_family import MathematicalProperty, ProblemFamily
    from domain.representations import SolutionRepresentationKind
    from domain.variables import VariableType
    from services.execution_engine import OptimizationExecutionEngine

    def v(name):
        return StructuredExpression(kind="variable", name=name)

    def lit(value):
        return StructuredExpression(kind="literal", value=value)

    def mul(a, b):
        return StructuredExpression(kind="binary", op="mul", args=(a, b))

    def add(a, b):
        return StructuredExpression(kind="binary", op="add", args=(a, b))

    objective = add(
        add(mul(lit(4), v("p0")), mul(lit(3), v("p1"))),
        add(mul(lit(2), v("p2")), v("p3")),
    )
    variables = [
        VariableSpec(f"p{i}", VariableType.DISCRETE, DomainSpec("permutation", values=[0, 1, 2, 3]))
        for i in range(4)
    ]
    problem = OptimizationProblem(
        name="aco_execution",
        objective=ObjectiveSpec(ObjectiveKind.SINGLE, ObjectiveSense.MAXIMIZE, objective),
        variables=variables,
        problem_family=ProblemFamily.ROUTING,
        mathematical_properties={MathematicalProperty.COMBINATORIAL, MathematicalProperty.DISCRETE},
        solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.PERMUTATION, "permutation"),
    )

    result = OptimizationExecutionEngine(AlgorithmRegistry.from_builtin_algorithms()).execute(problem, "aco")
    assert result.algorithm_id == "aco"
    assert result.objective_value >= 19.0
    assert sorted(result.variable_values.values()) == [0, 1, 2, 3]


def test_execution_engine_auto_selects_one_algorithm():
    from services.execution_engine import OptimizationExecutionEngine
    from algorithms.registry import AlgorithmRegistry
    from tests.test_universal_metaheuristics import black_box_style_problem

    engine = OptimizationExecutionEngine(AlgorithmRegistry.from_builtin_algorithms())
    result = engine.execute_auto(black_box_style_problem())
    assert result.algorithm_id == "de"
    assert result.objective_value > -1.0
