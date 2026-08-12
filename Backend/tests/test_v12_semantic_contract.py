from api import AIDraftProblem, ConstraintInput, ProblemInput, build_problem
from domain.objectives import ObjectiveStatus
from services.llm_service import GroqLLMService


def _base_model(**overrides):
    model = {
        "name": "test",
        "description": "test",
        "problem_family": "continuous_optimization",
        "mathematical_properties": ["continuous"],
        "variables": [{"name": "x", "variable_type": "continuous", "lower_bound": 0, "upper_bound": 10}],
        "objective_kind": "single",
        "objective_sense": "maximize",
        "objective_metric": None,
        "objective_status": "incomplete",
        "expression": "",
        "representation": "vector",
        "representation_metadata": {},
        "problem_structure": "tabular",
        "problem_structure_metadata": {},
        "constraints": [],
        "explanation": "Objective is missing.",
        "assumptions": ["The objective function was not supplied."],
    }
    model.update(overrides)
    return model


def test_objective_status_is_derived_for_incomplete_model():
    service = GroqLLMService.__new__(GroqLLMService)
    result = _base_model()
    service._validate_model(result, {})
    assert result["objective_status"] == "incomplete"


def test_objective_status_is_complete_for_semantic_metric():
    service = GroqLLMService.__new__(GroqLLMService)
    result = _base_model(objective_metric="path_length")
    service._validate_model(result, {})
    assert result["objective_status"] == "complete"


def test_objective_status_is_not_applicable_without_decisions():
    service = GroqLLMService.__new__(GroqLLMService)
    result = _base_model(variables=[], representation="", objective_sense="maximize")
    service._validate_model(result, {})
    assert result["objective_status"] == "not_applicable"


def test_constraint_does_not_require_legacy_bound_fields():
    service = GroqLLMService.__new__(GroqLLMService)
    result = _base_model(
        expression="x",
        objective_status="complete",
        constraints=[{
            "id": "limit",
            "name": "Limit",
            "kind": "hard",
            "relation": "le",
            "expression": "x",
            "threshold": 5,
        }],
    )
    service._validate_model(result, {})
    assert result["constraints"][0]["threshold"] == 5


def test_constraint_bound_is_accepted_and_normalized_for_execution():
    payload = ProblemInput.model_validate({
        "name": "Bound test",
        "variables": [{"name": "x", "variable_type": "continuous", "lower_bound": 0, "upper_bound": 10}],
        "objective_sense": "maximize",
        "expression": "x",
        "constraints": [{
            "id": "limit",
            "name": "Limit",
            "kind": "hard",
            "relation": "le",
            "expression": "x",
            "bound": 5,
        }],
        "representation": "vector",
    })
    problem = build_problem(payload)
    assert problem.constraints[0].bound == 5
    assert problem.constraints[0].threshold == 5


def test_ai_draft_accepts_objective_status():
    draft = AIDraftProblem.model_validate(_base_model())
    assert draft.objective_status == ObjectiveStatus.INCOMPLETE
