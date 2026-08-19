import io

import pytest

from adapters.solver_adapter import ClassicalModelAdapter, OrToolsConstraintProgrammingAdapter, cp_model
from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveKind, ObjectiveSense, ObjectiveSpec
from domain.problem import ConstraintSpec, DomainSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType
from services.dataset_service import DatasetService


def v(name):
    return StructuredExpression(kind="variable", name=name)

def lit(value):
    return StructuredExpression(kind="literal", value=value)

def mul(a, b):
    return StructuredExpression(kind="binary", op="mul", args=(a, b))

def add(*items):
    result = items[0]
    for item in items[1:]:
        result = StructuredExpression(kind="binary", op="add", args=(result, item))
    return result


def production_test_problem():
    data = [("P001", 0, 35, 325, 1800, 1.8), ("P002", 0, 50, 290, 700, 5.0), ("P003", 0, 80, 50, 250, 1.2), ("P004", 0, 100, 120, 180, 0.8), ("P005", 0, 60, 96, 320, 1.0), ("P006", 0, 45, 12.5, 280, 0.6)]
    variables = [VariableSpec(name, VariableType.INTEGER, DomainSpec("integer", lower=low, upper=high), low, high) for name, low, high, *_ in data]
    objective = add(*(mul(lit(coef), v(name)) for name, _, _, coef, _, _ in data))
    constraints = []
    for category, ids, budget, storage in [("electronics", {"P001", "P002", "P006"}, 50000, 180), ("peripherals", {"P003", "P004", "P005"}, 30000, 120)]:
        constraints.append(ConstraintSpec(f"budget_{category}", f"Budget {category}", "hard", "le", add(*(mul(lit(cost), v(name)) for name, _, _, _, cost, _ in data if name in ids)), threshold=budget))
        constraints.append(ConstraintSpec(f"storage_{category}", f"Storage {category}", "hard", "le", add(*(mul(lit(weight), v(name)) for name, _, _, _, _, weight in data if name in ids)), threshold=storage))
    return OptimizationProblem(name="production_planning", objective=ObjectiveSpec(ObjectiveKind.SINGLE, ObjectiveSense.MAXIMIZE, objective), variables=variables, constraints=constraints, problem_family=ProblemFamily.PRODUCTION_PLANNING, mathematical_properties={MathematicalProperty.INTEGER, MathematicalProperty.LINEAR, MathematicalProperty.CONSTRAINED}, solution_representation=SolutionRepresentationSpec(SolutionRepresentationKind.VECTOR, "vector"))


def test_multiple_csv_sources_are_kept_separate():
    result = DatasetService().summarize_files([("products.csv", b"product_id,cost\nP1,10\nP2,20\n"), ("demand.csv", b"product_id,demand\nP1,100\nP2,200\n")])
    assert result["multi_source"] is True
    assert [source["filename"] for source in result["sources"]] == ["products.csv", "demand.csv"]
    assert result["sources"][0]["columns"] == ["product_id", "cost"]
    assert result["sources"][1]["columns"] == ["product_id", "demand"]


def test_txt_plain_text_is_supported():
    result = DatasetService().summarize_files([("notes.txt", b"first line\nsecond line\n")])
    assert result["sources"][0]["format"] == "txt"
    assert result["sources"][0]["columns"] == ["line_number", "text"]


def test_xlsx_sheet_is_exposed_as_a_distinct_source():
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Products"
    sheet.append(["product_id", "cost"])
    sheet.append(["P1", 10])
    stream = io.BytesIO()
    workbook.save(stream)
    result = DatasetService().summarize_files([("data.xlsx", stream.getvalue())])
    assert result["sources"][0]["source_name"] == "data.xlsx::Products"
    assert result["sources"][0]["columns"] == ["product_id", "cost"]


def test_exact_integer_solvers_share_the_same_canonical_model():
    from algorithms.classical import ConstraintProgramming, IntegerProgramming
    problem = production_test_problem()
    integer_solution, integer_value = IntegerProgramming().optimize(problem)
    constraint_solution, constraint_value = ConstraintProgramming().optimize(problem)
    assert integer_value == pytest.approx(29452.0)
    assert constraint_value == pytest.approx(integer_value)
    assert constraint_solution == pytest.approx(integer_solution)


def test_scipy_integer_programming_matches_known_optimum():
    from algorithms.classical import IntegerProgramming
    solution, value = IntegerProgramming().optimize(production_test_problem())
    assert value == pytest.approx(29452.0)
    assert solution == pytest.approx([16, 30, 0, 100, 37, 0])


def test_cp_sat_integerization_preserves_fractional_model_exactly():
    from adapters.solver_adapter import OrToolsConstraintProgrammingAdapter
    import numpy as np

    coefficients, constant, scale = OrToolsConstraintProgrammingAdapter._integerize_objective([325, 12.5], 0)
    assert coefficients.tolist() == [26, 1]
    assert constant == 0
    assert scale == 1

    row, lower, upper, row_scale = OrToolsConstraintProgrammingAdapter._integerize_row([0.8, 1.2], -np.inf, 120)
    assert row.tolist() == [2, 3]
    assert lower == -np.inf
    assert upper == 300
    assert row_scale == 2


@pytest.mark.skipif(cp_model is None, reason="OR-Tools is not installed")
def test_cp_sat_accepts_fractional_linear_coefficients_and_preserves_maximization_direction():
    model = ClassicalModelAdapter().build(production_test_problem())
    solution, value = OrToolsConstraintProgrammingAdapter().solve(model)

    # The canonical model stores MAXIMIZE objectives as a negated minimization
    # vector. CP-SAT must preserve that direction at its own boundary.
    assert model.objective_sense == ObjectiveSense.MAXIMIZE
    assert model.objective.tolist() == pytest.approx([-325, -290, -50, -120, -96, -12.5])
    assert value == pytest.approx(29452.0)
    assert solution == pytest.approx([16, 30, 0, 100, 37, 0])


def test_api_integer_programming_returns_optimum_for_fractional_objective():
    from fastapi.testclient import TestClient
    from api import app

    variables = [
        {"name": f"purchase_qty_{pid}", "variable_type": "integer", "lower_bound": 0, "upper_bound": upper}
        for pid, upper in [("P001", 35), ("P002", 50), ("P003", 80), ("P004", 100), ("P005", 60), ("P006", 45)]
    ]
    payload = {
        "name": "production_planning",
        "problem_family": "production_planning",
        "mathematical_properties": ["integer", "linear", "constrained"],
        "variables": variables,
        "objective_kind": "single",
        "objective_sense": "maximize",
        "expression": "325*purchase_qty_P001 + 290*purchase_qty_P002 + 50*purchase_qty_P003 + 120*purchase_qty_P004 + 96*purchase_qty_P005 + 12.5*purchase_qty_P006",
        "representation": "vector",
        "constraints": [
            {"id": "budget_e", "name": "electronics budget", "kind": "hard", "relation": "le", "expression": "1800*purchase_qty_P001 + 700*purchase_qty_P002 + 280*purchase_qty_P006", "threshold": 50000},
            {"id": "storage_e", "name": "electronics storage", "kind": "hard", "relation": "le", "expression": "1.8*purchase_qty_P001 + 5*purchase_qty_P002 + 0.6*purchase_qty_P006", "threshold": 180},
            {"id": "budget_p", "name": "peripheral budget", "kind": "hard", "relation": "le", "expression": "250*purchase_qty_P003 + 180*purchase_qty_P004 + 320*purchase_qty_P005", "threshold": 30000},
            {"id": "storage_p", "name": "peripheral storage", "kind": "hard", "relation": "le", "expression": "1.2*purchase_qty_P003 + 0.8*purchase_qty_P004 + 1.0*purchase_qty_P005", "threshold": 120},
        ],
        "algorithm_id": "integer_programming",
    }
    response = TestClient(app).post("/api/solve", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["objective_value"] == pytest.approx(29452.0)
    assert data["variable_values"] == {
        "purchase_qty_P001": 16.0, "purchase_qty_P002": 30.0, "purchase_qty_P003": 0.0,
        "purchase_qty_P004": 100.0, "purchase_qty_P005": 37.0, "purchase_qty_P006": 0.0,
    }
