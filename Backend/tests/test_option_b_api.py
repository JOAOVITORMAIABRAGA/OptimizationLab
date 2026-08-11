from api import AIDraftProblem, ProblemInput


def test_ai_draft_allows_incomplete_model():
    draft = AIDraftProblem.model_validate({
        "name": "Delivery optimization",
        "description": "Reduce delivery delays.",
        "variables": [],
        "expression": "",
        "representation": None,
    })
    assert draft.variables == []
    assert draft.expression == ""
    assert draft.representation is None


def test_manual_problem_input_remains_strict():
    try:
        ProblemInput.model_validate({
            "name": "Incomplete",
            "variables": [],
            "expression": "",
        })
    except Exception as exc:
        assert "variables" in str(exc) or "expression" in str(exc)
    else:
        raise AssertionError("ProblemInput must remain strict for executable analysis")
