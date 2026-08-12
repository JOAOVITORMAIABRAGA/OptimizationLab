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
