from services.llm_service import GroqLLMService


DATASET = {
    "filename": "production_planning_test.csv",
    "row_count": 4,
    "column_count": 4,
    "columns": ["product", "profit_per_unit", "min_quantity", "max_quantity"],
    "sample_rows": [
        {"product": "A", "profit_per_unit": "20", "min_quantity": "0", "max_quantity": "40"},
        {"product": "B", "profit_per_unit": "15", "min_quantity": "0", "max_quantity": "50"},
        {"product": "C", "profit_per_unit": "30", "min_quantity": "0", "max_quantity": "35"},
        {"product": "D", "profit_per_unit": "10", "min_quantity": "0", "max_quantity": "45"},
    ],
    "rows": [
        {"product": "A", "profit_per_unit": "20", "min_quantity": "0", "max_quantity": "40"},
        {"product": "B", "profit_per_unit": "15", "min_quantity": "0", "max_quantity": "50"},
        {"product": "C", "profit_per_unit": "30", "min_quantity": "0", "max_quantity": "35"},
        {"product": "D", "profit_per_unit": "10", "min_quantity": "0", "max_quantity": "45"},
    ],
    "rows_truncated": False,
    "allowed_values": {
        "problem_families": ["production_planning", "generic"],
        "mathematical_properties": ["integer", "linear", "constrained"],
        "variable_types": ["integer", "continuous", "binary"],
        "representations": ["vector"],
        "objective_kinds": ["single", "multi"],
        "objective_senses": ["minimize", "maximize"],
    },
}


def test_tabular_production_model_is_completed_from_dataset():
    service = object.__new__(GroqLLMService)
    raw = {
        "name": "production planning",
        "description": "Choose production quantity for each product.",
        "problem_family": "generic",
        "mathematical_properties": ["integer", "linear", "constrained"],
        "variables": [{"name": "produce_qty", "variable_type": "integer", "lower_bound": 0, "upper_bound": None}],
        "objective_kind": "single",
        "objective_sense": "maximize",
        "expression": "",
        "representation": "vector",
        "explanation": "The model chooses production quantities.",
        "assumptions": [],
    }

    completed = service._complete_tabular_model(
        raw,
        "Tenho uma fábrica e quero decidir quanto produzir de cada produto para maximizar o lucro.",
        DATASET,
    )

    assert [item["name"] for item in completed["variables"]] == [
        "produce_qty_A", "produce_qty_B", "produce_qty_C", "produce_qty_D"
    ]
    assert [(item["lower_bound"], item["upper_bound"]) for item in completed["variables"]] == [
        (0.0, 40.0), (0.0, 50.0), (0.0, 35.0), (0.0, 45.0)
    ]
    assert completed["expression"] == "20*produce_qty_A + 15*produce_qty_B + 30*produce_qty_C + 10*produce_qty_D"
    assert completed["problem_family"] == "production_planning"
    assert completed["objective_sense"] == "maximize"


def test_tabular_completion_does_not_invent_missing_objective_parameter():
    service = object.__new__(GroqLLMService)
    raw = {
        "name": "production planning",
        "description": "Choose how much to produce.",
        "problem_family": "production_planning",
        "mathematical_properties": ["integer", "constrained"],
        "variables": [{"name": "produce_qty", "variable_type": "integer", "lower_bound": 0, "upper_bound": None}],
        "objective_kind": "single",
        "objective_sense": "maximize",
        "expression": "",
        "representation": "vector",
        "explanation": "Production decision.",
        "assumptions": [],
    }
    dataset = {**DATASET, "columns": ["product", "min_quantity", "max_quantity"]}
    dataset["rows"] = [{"product": "A", "min_quantity": "0", "max_quantity": "40"}]

    completed = service._complete_tabular_model(
        raw,
        "Quero produzir produtos para maximizar o lucro.",
        dataset,
    )

    assert completed == raw


def test_orchestrator_uses_declarative_llm_contract():
    from services.orchestrator import OptimizationOrchestrator

    orchestrator = object.__new__(OptimizationOrchestrator)
    class Stub:
        def draft_model(self, description, data):
            return {"name": "x", "variables": []}
    orchestrator.llm_service = Stub()
    result = orchestrator.generate_model(type("Request", (), {"problem_description": "x", "data": {}})())
    assert result["name"] == "x"
