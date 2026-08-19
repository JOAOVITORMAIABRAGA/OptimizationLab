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



def test_graph_completion_normalizes_edge_table_without_inventing_values():
    service = GroqLLMService.__new__(GroqLLMService)
    result = {
        "name": "Carteiro Chinês",
        "description": "Percorrer todas as ruas minimizando a distância.",
        "problem_family": "graph_optimization",
        "mathematical_properties": ["discrete", "combinatorial"],
        "variables": [{"name": "route", "variable_type": "discrete", "lower_bound": None, "upper_bound": None}],
        "objective_kind": "single",
        "objective_sense": "minimize",
        "expression": "0",
        "representation": "graph",
        "representation_metadata": {},
        "explanation": "",
        "assumptions": [],
    }
    dataset = {
        "rows": [
            {"edge_id": "ab", "from": "A", "to": "B", "distance": 2},
            {"edge_id": "bc", "from": "B", "to": "C", "distance": 3},
        ],
        "rows_truncated": False,
    }
    completed = service._complete_graph_model(result, "Quero resolver o Carteiro Chinês", dataset)
    metadata = completed["representation_metadata"]
    assert metadata["graph_problem_type"] == "chinese_postman"
    assert metadata["edges"][0] == {"id": "ab", "u": "A", "v": "B", "weight": 2.0, "required": True}
    assert set(metadata["nodes"]) == {"A", "B", "C"}


def test_tabular_completion_materializes_explicit_budget_constraint():
    service = GroqLLMService.__new__(GroqLLMService)
    raw = {
        "name": "project selection",
        "description": "Choose projects with maximum return under a budget.",
        "problem_family": "portfolio_optimization",
        "mathematical_properties": ["binary", "linear"],
        "variables": [
            {"name": "select_A", "variable_type": "binary", "lower_bound": 0, "upper_bound": 1},
            {"name": "select_B", "variable_type": "binary", "lower_bound": 0, "upper_bound": 1},
            {"name": "select_C", "variable_type": "binary", "lower_bound": 0, "upper_bound": 1},
            {"name": "select_D", "variable_type": "binary", "lower_bound": 0, "upper_bound": 1},
            {"name": "select_E", "variable_type": "binary", "lower_bound": 0, "upper_bound": 1},
            {"name": "select_F", "variable_type": "binary", "lower_bound": 0, "upper_bound": 1},
        ],
        "objective_kind": "single",
        "objective_sense": "maximize",
        "expression": "80*select_A + 72*select_B + 65*select_C + 55*select_D + 42*select_E + 30*select_F",
        "representation": "vector",
        "representation_metadata": {},
        "explanation": "Select projects.",
        "assumptions": [],
    }
    dataset = {
        "columns": ["project", "cost", "return"],
        "rows": [
            {"project": "A", "cost": "40", "return": "80"},
            {"project": "B", "cost": "35", "return": "72"},
            {"project": "C", "cost": "30", "return": "65"},
            {"project": "D", "cost": "25", "return": "55"},
            {"project": "E", "cost": "20", "return": "42"},
            {"project": "F", "cost": "15", "return": "30"},
            ],
        "rows_truncated": False,
    }
    completed = service._complete_tabular_constraints(
        raw,
        "Tenho uma verba limitada para investir nos projetos. Não posso gastar mais de 100.",
        dataset,
    )
    assert completed["constraints"] == [{
        "id": "budget_limit",
        "name": "Budget limit",
        "kind": "hard",
        "relation": "le",
        "expression": "40*select_A + 35*select_B + 30*select_C + 25*select_D + 20*select_E + 15*select_F",
        "lower_bound": None,
        "upper_bound": None,
        "threshold": 100.0,
    }]
    assert "constrained" in completed["mathematical_properties"]


def test_budget_limit_parser_supports_thousand_unit_without_silent_rescaling():
    service = GroqLLMService.__new__(GroqLLMService)
    assert service._extract_budget_limit("Não posso gastar mais de 100 mil.") == 100000.0


def test_ai_draft_contract_preserves_constraints_and_metadata():
    from api import validate_ai_draft
    draft = {
        "name": "project selection",
        "description": "Select projects under a budget.",
        "problem_family": "portfolio_optimization",
        "mathematical_properties": ["binary", "linear", "constrained"],
        "variables": [
            {"name": "select_A", "variable_type": "binary", "lower_bound": 0, "upper_bound": 1}
        ],
        "objective_kind": "single",
        "objective_sense": "maximize",
        "expression": "80*select_A",
        "constraints": [{
            "id": "budget_limit", "name": "Budget limit", "kind": "hard",
            "relation": "le", "expression": "40*select_A",
            "lower_bound": None, "upper_bound": None, "threshold": 100
        }],
        "representation": "vector",
        "representation_metadata": {},
    }
    model = validate_ai_draft(draft)
    assert len(model.constraints) == 1
    assert model.constraints[0].threshold == 100


def test_graph_completion_normalizes_llm_edge_count_variables():
    service = GroqLLMService.__new__(GroqLLMService)
    result = {
        "name": "Carteiro Chinês",
        "description": "Percorrer todas as ruas com menor distância.",
        "problem_family": "routing",
        "mathematical_properties": ["integer", "constrained"],
        "variables": [
            {"name": "edge_count_AB", "variable_type": "integer", "lower_bound": None, "upper_bound": None},
            {"name": "edge_count_BC", "variable_type": "integer", "lower_bound": None, "upper_bound": None},
        ],
        "objective_kind": "single",
        "objective_sense": "minimize",
        "expression": "edge_count_AB + edge_count_BC",
        "constraints": [],
        "representation": "graph",
        "representation_metadata": {"graph_problem_type": "chinese_postman"},
        "explanation": "graph",
        "assumptions": [],
    }
    dataset = {
        "rows": [
            {"edge": "AB", "from": "A", "to": "B", "distance": 1},
            {"edge": "BC", "from": "B", "to": "C", "distance": 1},
        ],
        "rows_truncated": False,
    }
    completed = service._complete_graph_model(result, "Quero resolver o Carteiro Chinês", dataset)
    assert completed["problem_family"] == "graph_optimization"
    assert completed["variables"] == [{
        "name": "route",
        "variable_type": "discrete",
        "lower_bound": None,
        "upper_bound": None,
    }]
    assert completed["expression"] == ""
    assert completed["objective_metric"] == "total_distance"
    assert completed["representation_metadata"]["graph_problem_type"] == "chinese_postman"


def test_graph_completion_infers_explicit_shortest_path_endpoints():
    service = GroqLLMService.__new__(GroqLLMService)
    result = {
        "name": "Shortest path",
        "description": "Find the shortest path.",
        "problem_family": "graph_optimization",
        "mathematical_properties": ["discrete", "combinatorial", "constrained"],
        "variables": [{"name": "route", "variable_type": "discrete", "lower_bound": None, "upper_bound": None}],
        "objective_kind": "single",
        "objective_sense": "minimize",
        "objective_metric": "path_length",
        "objective_status": "complete",
        "expression": "",
        "representation": "graph",
        "representation_metadata": {},
        "problem_structure": "graph",
        "problem_structure_metadata": {},
        "constraints": [],
        "explanation": "",
        "assumptions": [],
    }
    dataset = {
        "rows": [
            {"edge_id": "AB", "from": "A", "to": "B", "distance": 2},
            {"edge_id": "BC", "from": "B", "to": "C", "distance": 3},
        ],
        "rows_truncated": False,
    }
    completed = service._complete_graph_model(result, "Encontre o menor caminho de A para C.", dataset)
    assert completed["representation_metadata"]["source"] == "A"
    assert completed["representation_metadata"]["target"] == "C"


def test_json_transport_falls_back_after_groq_json_validation_error():
    service = GroqLLMService.__new__(GroqLLMService)
    service.model = "llama-3.3-70b-versatile"

    class Message:
        content = '{"name":"x"}'

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("Error code: 400 - code: json_validate_failed")
            return Response()

    class Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": Completions()})()

    service.client = Client()
    result = service._request_model("Return JSON.")
    assert result == {"name": "x"}
    assert len(service.client.chat.completions.calls) == 2
    assert "response_format" not in service.client.chat.completions.calls[1]


def test_multi_source_purchase_bounds_are_materialized_from_demand_table():
    service = GroqLLMService.__new__(GroqLLMService)
    raw = {
        "name": "production planning",
        "description": "",
        "problem_family": "production_planning",
        "mathematical_properties": ["integer", "linear", "constrained"],
        "variables": [
            {"name": f"purchase_qty_{pid}", "variable_type": "integer", "lower_bound": None, "upper_bound": None}
            for pid in ("P001", "P002", "P003")
        ],
        "objective_kind": "single",
        "objective_sense": "maximize",
        "expression": "325*purchase_qty_P001 + 290*purchase_qty_P002 + 50*purchase_qty_P003",
        "representation": "vector",
        "explanation": "Purchase planning.",
        "assumptions": [],
    }
    dataset = {
        "source_count": 2,
        "multi_source": True,
        "sources": [
            {
                "filename": "produtos.csv",
                "source_kind": "tabular",
                "columns": ["product_id", "cost"],
                "rows": [
                    {"product_id": "P001", "cost": 1800},
                    {"product_id": "P002", "cost": 700},
                    {"product_id": "P003", "cost": 250},
                ],
                "rows_truncated": False,
            },
            {
                "filename": "demanda.csv",
                "source_kind": "tabular",
                "columns": ["product_id", "monthly_demand", "expected_conversion_rate"],
                "rows": [
                    {"product_id": "P001", "monthly_demand": 35, "expected_conversion_rate": 0.85},
                    {"product_id": "P002", "monthly_demand": 50, "expected_conversion_rate": 0.90},
                    {"product_id": "P003", "monthly_demand": 80, "expected_conversion_rate": 0.75},
                ],
                "rows_truncated": False,
            },
        ],
    }

    completed = service._complete_multi_source_quantity_bounds(
        raw,
        "Não quero comprar mais unidades de um produto do que a demanda mensal estimada.",
        dataset,
    )

    assert [(item["name"], item["lower_bound"], item["upper_bound"]) for item in completed["variables"]] == [
        ("purchase_qty_P001", 0.0, 35.0),
        ("purchase_qty_P002", 0.0, 50.0),
        ("purchase_qty_P003", 0.0, 80.0),
    ]
