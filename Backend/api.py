from __future__ import annotations

import ast
import operator
from typing import Any, Dict, List, Optional

import csv
import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from algorithms.registry import AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from domain.expressions import StructuredExpression
from domain.objectives import ObjectiveKind, ObjectiveSense
from domain.problem import ConstraintSpec, DomainSpec, ObjectiveSpec, OptimizationProblem, SolutionRepresentationSpec, VariableSpec
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType
from recommendation.recommendation_engine import RecommendationEngine
from validation.validator import ValidationEngine
from services.llm_service import GroqLLMService
from services.execution_engine import OptimizationExecutionEngine


class VariableInput(BaseModel):
    name: str = Field(min_length=1)
    variable_type: VariableType
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ConstraintInput(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: str = "hard"
    relation: str
    expression: Optional[str] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    threshold: Optional[float] = None


class ObjectiveInput(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    sense: ObjectiveSense
    expression: str = Field(min_length=1)


class ProblemInput(BaseModel):
    name: str = Field(min_length=1)

    @field_validator("representation", mode="before")
    @classmethod
    def normalize_representation(cls, value):
        if value in ("", None):
            return SolutionRepresentationKind.VECTOR
        return value
    description: Optional[str] = None
    problem_family: ProblemFamily = ProblemFamily.GENERIC
    mathematical_properties: List[MathematicalProperty] = Field(default_factory=list)
    variables: List[VariableInput] = Field(min_length=1)
    objective_kind: ObjectiveKind = ObjectiveKind.SINGLE
    objective_sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    expression: str = ""
    objectives: List[ObjectiveInput] = Field(default_factory=list)
    constraints: List[ConstraintInput] = Field(default_factory=list)
    representation: SolutionRepresentationKind = SolutionRepresentationKind.VECTOR


class AIDraftProblem(BaseModel):
    """AI proposal contract. Unlike ProblemInput, this may be incomplete."""
    name: str = Field(min_length=1)
    description: Optional[str] = None
    problem_family: ProblemFamily = ProblemFamily.GENERIC
    mathematical_properties: List[MathematicalProperty] = Field(default_factory=list)
    variables: List[VariableInput] = Field(default_factory=list)
    objective_kind: ObjectiveKind = ObjectiveKind.SINGLE
    objective_sense: ObjectiveSense = ObjectiveSense.MINIMIZE
    expression: str = ""
    representation: Optional[SolutionRepresentationKind] = None


class AIModelResponse(BaseModel):
    problem: Dict[str, Any]
    explanation: str
    assumptions: List[str]
    dataset: Dict[str, Any]
    incomplete: bool = False


class ValidationResponse(BaseModel):
    valid: bool
    blocking: bool
    errors: List[str]
    warnings: List[str]


class CompatibilityResponse(BaseModel):
    algorithm_id: str
    algorithm_name: str
    status: str
    reasons: List[str]
    warnings: List[str]
    required_adapters: List[str]
    required_operators: List[str]


class RecommendationResponse(BaseModel):
    algorithm_id: str
    algorithm_name: str
    score: float
    rank: int
    rationale: str
    strengths: List[str]
    weaknesses: List[str]
    evidence: List[str]


class AnalysisResponse(BaseModel):
    problem: Dict[str, Any]
    validation: ValidationResponse
    compatibility: List[CompatibilityResponse]
    recommendations: List[RecommendationResponse]
    excluded_algorithms: List[Dict[str, Any]]


class ExpressionParser:
    """Small allow-list parser; user input is never evaluated as Python code."""

    _binary = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mul",
        ast.Div: "div",
        ast.Pow: "pow",
        ast.Mod: "mod",
    }
    _functions = {"abs", "min", "max", "sum"}

    def parse(self, text: str, variable_names: set[str]) -> StructuredExpression:
        try:
            tree = ast.parse(text.replace("^", "**"), mode="eval")
        except SyntaxError as exc:
            raise ValueError("Invalid mathematical expression.") from exc
        return self._node(tree.body, variable_names)

    def _node(self, node: ast.AST, variable_names: set[str]) -> StructuredExpression:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return StructuredExpression(kind="literal", value=node.value)
        if isinstance(node, ast.Name):
            if node.id not in variable_names:
                raise ValueError(f"Expression references unknown variable '{node.id}'.")
            return StructuredExpression(kind="variable", name=node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return StructuredExpression(kind="unary", op="neg", args=(self._node(node.operand, variable_names),))
        if isinstance(node, ast.BinOp):
            op = next((name for cls, name in self._binary.items() if isinstance(node.op, cls)), None)
            if op is None:
                raise ValueError("Unsupported expression operator.")
            return StructuredExpression(kind="binary", op=op, args=(self._node(node.left, variable_names), self._node(node.right, variable_names)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self._functions and not node.keywords:
            if not node.args:
                raise ValueError(f"Function '{node.func.id}' requires at least one argument.")
            return StructuredExpression(kind="function", op=node.func.id, args=tuple(self._node(arg, variable_names) for arg in node.args))
        raise ValueError("Unsupported expression. Use variables, numbers, +, -, *, /, ^ and supported functions (abs, min, max, sum).")


def expression_to_dict(expression: StructuredExpression) -> Dict[str, Any]:
    data: Dict[str, Any] = {"kind": expression.kind}
    if expression.op is not None:
        data["op"] = expression.op
    if expression.name is not None:
        data["name"] = expression.name
    if expression.kind == "literal":
        data["value"] = expression.value
    if expression.args:
        data["args"] = [expression_to_dict(arg) for arg in expression.args]
    return data


def build_problem(payload: ProblemInput) -> OptimizationProblem:
    variable_names = {item.name for item in payload.variables}
    if len(variable_names) != len(payload.variables):
        raise ValueError("Variable names must be unique.")

    expression = ExpressionParser().parse(payload.expression, variable_names) if payload.expression else None
    variables: List[VariableSpec] = []
    for item in payload.variables:
        if item.variable_type == VariableType.BINARY:
            domain = DomainSpec(kind="binary", values=[0, 1])
        elif item.variable_type == VariableType.INTEGER:
            domain = DomainSpec(kind="integer", lower=item.lower_bound, upper=item.upper_bound)
        elif item.variable_type == VariableType.CONTINUOUS:
            domain = DomainSpec(kind="continuous", lower=item.lower_bound, upper=item.upper_bound)
        elif item.variable_type == VariableType.DISCRETE:
            domain = DomainSpec(kind="discrete", lower=item.lower_bound, upper=item.upper_bound)
        else:
            domain = DomainSpec(kind="categorical")
        variables.append(VariableSpec(name=item.name, variable_type=item.variable_type, domain=domain, lower_bound=item.lower_bound, upper_bound=item.upper_bound))

    if payload.objective_kind == ObjectiveKind.MULTI:
        if len(payload.objectives) < 2:
            raise ValueError("Multiobjective problems require at least two objectives.")
        from domain.objectives import ObjectiveComponent
        objectives = tuple(
            ObjectiveComponent(item.id, item.name, item.sense, ExpressionParser().parse(item.expression, variable_names))
            for item in payload.objectives
        )
        objective = ObjectiveSpec(kind=ObjectiveKind.MULTI, sense=None, expression=None, objectives=objectives)
    else:
        if expression is None:
            raise ValueError("A single objective expression is required.")
        objective = ObjectiveSpec(kind=payload.objective_kind, sense=payload.objective_sense, expression=expression)

    constraints = []
    for item in payload.constraints:
        constraint_expression = ExpressionParser().parse(item.expression, variable_names) if item.expression else None
        constraints.append(
            ConstraintSpec(
                id=item.id,
                name=item.name,
                kind=item.kind,
                relation=item.relation,
                expression=constraint_expression,
                lower_bound=item.lower_bound,
                upper_bound=item.upper_bound,
                threshold=item.threshold,
            )
        )

    return OptimizationProblem(
        name=payload.name,
        description=payload.description,
        problem_family=payload.problem_family,
        mathematical_properties=set(payload.mathematical_properties),
        variables=variables,
        constraints=constraints,
        objective=objective,
        solution_representation=SolutionRepresentationSpec(kind=payload.representation, name=payload.representation.value),
    )


def problem_to_dict(problem: OptimizationProblem) -> Dict[str, Any]:
    return {
        "name": problem.name,
        "description": problem.description,
        "problem_family": problem.problem_family.value,
        "mathematical_properties": sorted(prop.value for prop in problem.mathematical_properties),
        "variables": [
            {
                "name": variable.name,
                "variable_type": variable.variable_type.value,
                "domain": {
                    "kind": variable.domain.kind if variable.domain else None,
                    "lower": variable.domain.lower if variable.domain else None,
                    "upper": variable.domain.upper if variable.domain else None,
                    "values": variable.domain.values if variable.domain else None,
                },
            }
            for variable in problem.variables
        ],
        "objective": {
            "kind": problem.objective.kind.value if hasattr(problem.objective.kind, "value") else problem.objective.kind,
            "sense": problem.objective.sense.value if hasattr(problem.objective.sense, "value") else problem.objective.sense,
            "expression": expression_to_dict(problem.objective.expression) if problem.objective.expression else None,
        },
        "solution_representation": problem.solution_representation.kind.value if problem.solution_representation else None,
    }


app = FastAPI(title="Optimization Lab MVP API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://optimization-lab-livid.vercel.app/"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

validation_engine = ValidationEngine()
compatibility_engine = CompatibilityEngine()
recommendation_engine = RecommendationEngine()
registry = AlgorithmRegistry.from_builtin_algorithms()


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "problem_families": [item.value for item in ProblemFamily],
        "mathematical_properties": [item.value for item in MathematicalProperty],
        "variable_types": [item.value for item in VariableType],
        "representations": [item.value for item in SolutionRepresentationKind],
        "objective_kinds": [item.value for item in ObjectiveKind],
        "objective_senses": [item.value for item in ObjectiveSense],
    }


def summarize_csv(content: bytes, filename: str) -> Dict[str, Any]:
    if not filename.lower().endswith(".csv"):
        raise ValueError("Only CSV datasets are supported in the MVP.")

    if len(content) > 10 * 1024 * 1024:
        raise ValueError("CSV file is too large. Maximum size is 10 MB.")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be encoded as UTF-8.") from exc

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV must contain a header row.")

    rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    row_count = 0
    max_model_rows = 5000
    for row in reader:
        row_count += 1
        if len(sample_rows) < 8:
            sample_rows.append(dict(row))
        if len(rows) < max_model_rows:
            rows.append(dict(row))

    columns = [name.strip() for name in reader.fieldnames if name and name.strip()]
    if not columns:
        raise ValueError("CSV must contain at least one named column.")

    return {
        "filename": filename,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "sample_rows": sample_rows,
        "rows": rows,
        "rows_truncated": row_count > max_model_rows,
    }


def validate_ai_draft(draft: Dict[str, Any]) -> AIDraftProblem:
    # AI proposals have a separate, intentionally permissive contract.
    # Manual / executable analysis remains strict through ProblemInput.

    allowed = {
        "name",
        "description",
        "problem_family",
        "mathematical_properties",
        "variables",
        "objective_kind",
        "objective_sense",
        "expression",
        "representation",
    }

    candidate = {
        key: value
        for key, value in draft.items()
        if key in allowed
    }

    # In an incomplete AI draft, representation may legitimately be
    # undefined. LLMs sometimes return "" instead of null.
    if candidate.get("representation") in ("", None):
        candidate["representation"] = None

    # Likewise, an incomplete model may have no mathematical expression.
    if candidate.get("expression") is None:
        candidate["expression"] = ""

    return AIDraftProblem.model_validate(candidate)


@app.post("/api/model", response_model=AIModelResponse)
async def model_problem(description: str = Form(...), file: UploadFile = File(...)) -> AIModelResponse:
    print(">>> /api/model FOI CHAMADO <<<")
    if not description.strip():
        raise HTTPException(status_code=422, detail="Problem description is required.")

    try:
        content = await file.read()
        dataset = summarize_csv(content, file.filename or "dataset.csv")
        dataset["allowed_values"] = {
            "problem_families": [item.value for item in ProblemFamily],
            "mathematical_properties": [item.value for item in MathematicalProperty],
            "variable_types": [item.value for item in VariableType],
            "representations": [item.value for item in SolutionRepresentationKind],
            "objective_kinds": [item.value for item in ObjectiveKind],
            "objective_senses": [item.value for item in ObjectiveSense],
        }
        draft = GroqLLMService().draft_model(description.strip(), dataset)
        problem_payload = validate_ai_draft(draft)
        incomplete = (
            not problem_payload.variables
            or not problem_payload.expression.strip()
        )

        # A draft with no defensible decision variables is intentionally an
        # intermediate state (Option B). Do not force it through the strict
        # executable ProblemInput/build_problem path. Complete drafts still
        # receive the exact same deterministic parser/builder validation used
        # by Advanced Mode.
        if not incomplete:
            build_problem(
                ProblemInput.model_validate(
                    problem_payload.model_dump(mode="json")
                )
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI modeling failed: {exc}") from exc

    return AIModelResponse(
        problem=problem_payload.model_dump(mode="json"),
        explanation=str(draft.get("explanation", "The AI proposed this model based on your description and dataset.")),
        assumptions=[str(item) for item in draft.get("assumptions", [])],
        dataset={key: value for key, value in dataset.items() if key != "allowed_values"},
        incomplete=incomplete,
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(payload: ProblemInput) -> AnalysisResponse:
    try:
        problem = build_problem(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    report = validation_engine.validate(problem)
    validation = ValidationResponse(valid=report.is_valid(), blocking=bool(report.errors), errors=report.errors, warnings=report.warnings)
    if not report.is_valid():
        return AnalysisResponse(problem=problem_to_dict(problem), validation=validation, compatibility=[], recommendations=[], excluded_algorithms=[])

    compatibility_results: List[CompatibilityResponse] = []
    for descriptor in sorted(registry.get_all(), key=lambda item: item.id):
        result = compatibility_engine.check(problem, descriptor)
        compatibility_results.append(CompatibilityResponse(
            algorithm_id=descriptor.id,
            algorithm_name=descriptor.name,
            status=result.status.value,
            reasons=list(result.reasons),
            warnings=list(result.warnings),
            required_adapters=list(result.required_adapters),
            required_operators=list(result.required_operators),
        ))

    recommendations_result = recommendation_engine.recommend(problem, registry, compatibility_engine=compatibility_engine)
    recommendations = [RecommendationResponse(
        algorithm_id=item.algorithm_id,
        algorithm_name=item.algorithm_name,
        score=item.score,
        rank=item.rank,
        rationale=item.rationale,
        strengths=list(item.strengths),
        weaknesses=list(item.weaknesses),
        evidence=list(item.evidence),
    ) for item in recommendations_result.recommendations]
    excluded = [{"algorithm_id": item.algorithm_id, "reason": item.reason, "compatibility_status": item.compatibility_status.value if item.compatibility_status else None, "evidence": list(item.evidence)} for item in recommendations_result.excluded_algorithms]

    return AnalysisResponse(problem=problem_to_dict(problem), validation=validation, compatibility=compatibility_results, recommendations=recommendations, excluded_algorithms=excluded)


class SolveProblemRequest(ProblemInput):
    algorithm_id: str = Field(min_length=1)


class SolveProblemResponse(BaseModel):
    algorithm_id: str
    solution: List[float]
    variable_values: Dict[str, float]
    objective_value: float
    parameters: Dict[str, Any]


execution_engine = OptimizationExecutionEngine(registry)


@app.post("/api/solve", response_model=SolveProblemResponse)
def solve(payload: SolveProblemRequest) -> SolveProblemResponse:
    try:
        problem = build_problem(payload)
        result = execution_engine.execute(problem, payload.algorithm_id)
    except (ValueError, KeyError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    variable_values = {
        variable.name: float(value)
        for variable, value in zip(problem.variables, result.solution)
    }
    return SolveProblemResponse(
        algorithm_id=result.algorithm_id,
        solution=result.solution,
        variable_values=variable_values,
        objective_value=result.objective_value,
        parameters=result.parameters,
    )
