from __future__ import annotations

import json
import os
import re
from typing import Optional
from pathlib import Path
from typing import Any

from services.graph_problem_classifier import GraphProblemClassifier

from dotenv import load_dotenv
try:
    from groq import Groq
except ImportError:  # Optional until an LLM call is actually requested.
    Groq = None  # type: ignore[assignment]


# ============================================================
# Environment
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BACKEND_DIR / ".env"

load_dotenv(ENV_FILE, override=True)


# ============================================================
# Constants
# ============================================================

DEFAULT_MODEL = "llama-3.3-70b-versatile"

ALLOWED_EXPRESSION_FUNCTIONS = {
    "abs",
    "min",
    "max",
    "sum",
}

REQUIRED_FIELDS = {
    "name",
    "description",
    "problem_family",
    "mathematical_properties",
    "variables",
    "objective_kind",
    "objective_sense",
    "expression",
    "representation",
    "explanation",
    "assumptions",
}

OPTIONAL_FIELDS = {
    "objective_metric",
    "objective_status",
    "constraints",
    "representation_metadata",
    "problem_structure",
    "problem_structure_metadata",
}

SAFE_IDENTIFIER_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*$"
)

EXPRESSION_IDENTIFIER_PATTERN = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*\b"
)


# ============================================================
# Service
# ============================================================

class GroqLLMService:
    """
    Serviço responsável por transformar uma descrição em linguagem
    natural + resumo do dataset em uma proposta de modelo de
    otimização.

    A IA:
    - interpreta a intenção do usuário;
    - identifica decisões controláveis;
    - identifica possíveis parâmetros;
    - propõe a função objetivo quando possível;
    - explica premissas e incertezas.

    A IA NÃO:
    - executa otimização;
    - executa código;
    - gera Python;
    - gera SQL;
    - acessa diretamente o otimizador.

    IMPORTANTE:

    Um modelo pode ser parcialmente determinado.

    Exemplo:

        usuário define claramente "quantidade a enviar";
        dataset não possui custo de envio.

    Nesse caso:

        variables = [ship_qty]
        expression = ""
        incomplete = True

    A ausência de uma expressão NÃO deve apagar uma variável de
    decisão que foi explicitamente definida pelo usuário.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured on the backend."
            )

        self.client = Groq(api_key=api_key)

        self.model = os.getenv(
            "GROQ_MODEL",
            DEFAULT_MODEL,
        )

    # ========================================================
    # Public API
    # ========================================================

    def draft_model(
        self,
        problem_description: str,
        dataset: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Gera e valida uma proposta de modelo.

        Se a primeira resposta falhar na validação local,
        uma única tentativa de reparo é feita.
        """

        prompt = self._build_prompt(
            problem_description=problem_description,
            dataset=dataset,
        )

        try:
            result = self._request_model(prompt)
        except ValueError:
            # Provider JSON failures must not make deterministic graph cases
            # unusable. The fallback is limited to graph-shaped tabular data
            # and still passes through the same canonical completion/validation
            # pipeline. It is not an algorithm-specific compatibility shortcut.
            result = self._deterministic_graph_fallback(
                problem_description=problem_description,
                dataset=dataset,
            )
        result = self._complete_tabular_model(
            result=result,
            problem_description=problem_description,
            dataset=dataset,
        )
        result = self._complete_multi_source_quantity_bounds(
            result=result,
            problem_description=problem_description,
            dataset=dataset,
        )
        result = self._complete_tabular_constraints(
            result=result,
            problem_description=problem_description,
            dataset=dataset,
        )
        result = self._complete_graph_model(
            result=result,
            problem_description=problem_description,
            dataset=dataset,
        )

        try:
            self._validate_model(
                result=result,
                dataset=dataset,
            )

            return result

        except ValueError as validation_error:
            repair_prompt = self._build_repair_prompt(
                original_result=result,
                validation_error=str(validation_error),
                dataset=dataset,
                problem_description=problem_description,
            )

            repaired_result = self._request_model(
                repair_prompt
            )
            repaired_result = self._complete_tabular_model(
                result=repaired_result,
                problem_description=problem_description,
                dataset=dataset,
            )
            repaired_result = self._complete_multi_source_quantity_bounds(
                result=repaired_result,
                problem_description=problem_description,
                dataset=dataset,
            )
            repaired_result = self._complete_tabular_constraints(
                result=repaired_result,
                problem_description=problem_description,
                dataset=dataset,
            )
            repaired_result = self._complete_graph_model(
                result=repaired_result,
                problem_description=problem_description,
                dataset=dataset,
            )

            self._validate_model(
                result=repaired_result,
                dataset=dataset,
            )

            return repaired_result

    # ========================================================
    # Groq
    # ========================================================

    def _request_model(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        """Request a JSON model with a resilient transport fallback.

        Groq has two relevant JSON mechanisms: JSON Object Mode, which is
        broadly supported, and Structured Outputs, which is available only
        on a subset of models. The modeling domain must not depend on one
        transport feature, so this method keeps the fallback policy here
        instead of spreading provider-specific branches through the domain.
        """
        attempts = []

        # GPT-OSS models support Groq Structured Outputs in strict mode.
        # For other models we deliberately keep JSON Object Mode as the
        # primary path because it has the broadest model compatibility.
        if self._supports_strict_structured_output():
            attempts.append((self._structured_response_format(), prompt))

        attempts.append(({"type": "json_object"}, prompt))

        # Last-resort transport: ask for JSON without response_format and
        # parse it locally. This is intentionally a provider-boundary fallback
        # and is never exposed to the rest of the application.
        attempts.append((None, self._build_compact_json_prompt(prompt)))

        last_error: Exception | None = None
        for response_format, request_prompt in attempts:
            try:
                kwargs: dict[str, Any] = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Return only one valid JSON object. Do not use markdown fences or commentary.",
                        },
                        {
                            "role": "user",
                            "content": request_prompt,
                        },
                    ],
                    "model": self.model,
                    "temperature": 0,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                result = self._parse_json_object(content)
                if not isinstance(result, dict):
                    raise ValueError("Groq returned an invalid problem model.")
                return result
            except Exception as exc:
                last_error = exc
                if not self._is_json_transport_error(exc):
                    # Non-format provider errors should not be hidden behind
                    # additional requests.
                    raise

        raise ValueError(
            "The AI provider could not return a valid JSON problem model after "
            "the supported JSON transport fallbacks were attempted."
        ) from last_error

    def _supports_strict_structured_output(self) -> bool:
        return self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}

    def _structured_response_format(self) -> dict[str, Any]:
        """Strict schema accepted by Groq Structured Outputs."""
        nullable_string = {"type": ["string", "null"]}
        nullable_number = {"type": ["number", "null"]}
        variable = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "variable_type": {"type": "string"},
                "lower_bound": nullable_number,
                "upper_bound": nullable_number,
            },
            "required": ["name", "variable_type", "lower_bound", "upper_bound"],
            "additionalProperties": False,
        }
        constraint = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "name": {"type": "string"},
                "kind": {"type": "string"},
                "relation": {"type": "string"},
                "expression": nullable_string,
                "bound": nullable_number,
                "lower_bound": nullable_number,
                "upper_bound": nullable_number,
                "threshold": nullable_number,
            },
            "required": ["id", "name", "kind", "relation", "expression", "bound", "lower_bound", "upper_bound", "threshold"],
            "additionalProperties": False,
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "optimization_problem_model",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "problem_family": {"type": "string"},
                        "mathematical_properties": {"type": "array", "items": {"type": "string"}},
                        "variables": {"type": "array", "items": variable},
                        "objective_kind": {"type": "string"},
                        "objective_sense": {"type": "string"},
                        "objective_metric": nullable_string,
                        "objective_status": {"type": "string"},
                        "expression": {"type": "string"},
                        "representation": {"type": "string"},
                        "representation_metadata": {"type": "object", "additionalProperties": True},
                        "problem_structure": {"type": "string"},
                        "problem_structure_metadata": {"type": "object", "additionalProperties": True},
                        "constraints": {"type": "array", "items": constraint},
                        "explanation": {"type": "string"},
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "name", "description", "problem_family", "mathematical_properties",
                        "variables", "objective_kind", "objective_sense", "objective_metric",
                        "objective_status", "expression", "representation", "representation_metadata",
                        "problem_structure", "problem_structure_metadata", "constraints",
                        "explanation", "assumptions",
                    ],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Some non-structured model paths prepend a short sentence.
            # Extract only the outermost JSON object without eval/exec.
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Groq returned invalid JSON.")
            parsed = json.loads(text[start:end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Groq returned an invalid problem model.")
        return parsed

    @staticmethod
    def _is_json_transport_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(marker in text for marker in (
            "json_validate_failed",
            "failed to validate json",
            "generated json does not match",
            "invalid json",
        ))

    @staticmethod
    def _build_compact_json_prompt(prompt: str) -> str:
        # Keep the original semantic instructions, but remove the enormous
        # decorative sections if a provider's JSON mode fails repeatedly.
        marker = "============================================================\nUSER REQUEST"
        if marker in prompt:
            prompt = prompt[prompt.index(marker):]
        return (
            "Return ONLY one JSON object matching the optimization model fields "
            "described in the user request. Preserve explicit decision variables; "
            "never invent missing numeric parameters.\n\n" + prompt
        )

    # ========================================================
    # Prompt
    # ========================================================

    def _build_prompt(
        self,
        problem_description: str,
        dataset: dict[str, Any],
    ) -> str:

        allowed = dataset.get(
            "allowed_values",
            {},
        )

        return f"""
You are the Optimization Lab modeling engine.

Your task is to transform:

1. the user's natural-language optimization request; and
2. the supplied dataset summary

into a structured optimization problem proposal.

You are NOT a chatbot.

Do not ask questions.
Do not provide conversational advice.
Do not execute optimization.
Do not write Python.
Do not write SQL.
Do not generate executable code.

Return ONLY valid JSON using the exact schema provided at the end.

============================================================
MOST IMPORTANT RULE
============================================================

The USER defines what the optimizer should decide.

The DATASET describes observed information and possible parameters.

These two concepts are different.

The dataset input may contain multiple independent sources. Each source is
identified by filename (and, for XLSX files, worksheet). Treat each source
as a distinct table or text source. Do NOT assume that sources are joined.
Only relate sources when the user description or the data clearly provides a
shared key/semantic relationship. Never fabricate a join, aggregation, or
matching rule. When several sources jointly determine the model, use the
information from all relevant sources and state the relationship in the
assumptions.

A decision variable does NOT need to exist as a column in the dataset.

If the user explicitly says that the optimizer should determine,
choose, assign, allocate, schedule, ship, produce, select, route,
or control something, that action is a legitimate decision variable.

NEVER remove an explicitly stated decision variable merely because
there is no corresponding column in the CSV.

============================================================
EXAMPLE: SHIPPING
============================================================

User:

"I want to determine how many products from each category should be
shipped in each period."

Correct decision variable:

ship_qty

Conceptual meaning:

ship_qty(category, period)

The dataset does NOT need to contain a "ship_qty" column.

The optimizer chooses ship_qty.

Historical dataset columns are observations or parameters.

Therefore, even if the dataset contains only historical orders and
shipping information, the variable ship_qty remains valid because
the USER explicitly requested that decision.

============================================================
OBSERVED DATA IS NOT AUTOMATICALLY A DECISION
============================================================

Dataset columns such as:

- actual shipping days;
- scheduled shipping days;
- sales;
- profit;
- delivery risk;
- customer information;
- historical order quantity;
- historical demand;
- historical shipment quantity;

are normally OBSERVED DATA.

Do NOT make them decision variables just because they exist.

For example:

Dataset:
"Days for shipping (real)"

User never says shipping time can be controlled.

Therefore:

shipping_days

is NOT a decision variable.

But:

User:
"Determine how many units should be shipped."

Then:

ship_qty

IS a decision variable, even if there is no ship_qty column.

============================================================
DECISION VARIABLE DIMENSIONS
============================================================

When the user gives dimensions, preserve their conceptual meaning.

For example:

"how many products from each category should be shipped in each
period"

means:

ship_qty(category, period)

However, the actual variable name MUST remain a safe identifier:

ship_qty

Do NOT use:

ship_qty(category, period)

as the variable name.

Do NOT use spaces, parentheses, brackets, or other invalid syntax
in variable names.

The dimensions can be described in the explanation or assumptions.

============================================================
TABULAR DATASETS AND INDEXED DECISIONS
============================================================

When the dataset contains one row per decision entity and the user asks
for a quantity for each entity, the scalar executable model MUST expand
the indexed concept into one safe variable per row when the current
OptimizationProblem contract requires scalar variables.

Example:

Dataset:
product,profit_per_unit,min_quantity,max_quantity
A,20,0,40
B,15,0,50
C,30,0,35
D,10,0,45

User:
"I need to decide how much to produce of each product to maximize profit."

Conceptual decision:
produce_qty(product)

Executable scalar variables:
produce_qty_A
produce_qty_B
produce_qty_C
produce_qty_D

Objective:
20*produce_qty_A + 15*produce_qty_B + 30*produce_qty_C + 10*produce_qty_D

Bounds:
produce_qty_A: 0..40
produce_qty_B: 0..50
produce_qty_C: 0..35
produce_qty_D: 0..45

The numeric dataset columns are parameters, not decision variables.
Their values may be embedded as coefficients or bounds in the declarative
model when the row/entity mapping is unambiguous. Never invent values.

If the dataset is insufficient to construct the complete expression, keep
the decision variable(s) and leave expression empty rather than fabricating
parameters.

============================================================
MODEL COMPLETENESS
============================================================

Determine each part of the model independently:

1. decision variables
2. parameters
3. constraints
4. objective
5. solution representation

A model can be PARTIALLY determined.

Missing information about one component must NOT erase information
already established about another component.

For example:

User:
"Determine how many products from each category should be shipped
in each period, minimizing shipping costs."

If the user clearly established ship_qty but the dataset does not
provide shipping-cost parameters:

variables MUST contain ship_qty.

The model may still be incomplete because the objective expression
cannot be constructed.

Correct:

variables:
[
    {{
        "name": "ship_qty",
        "variable_type": "integer",
        "lower_bound": 0,
        "upper_bound": null
    }}
]

expression:
""

representation:
"vector"

This is a valid PARTIAL MODEL.

============================================================
WHEN VARIABLES SHOULD BE EMPTY
============================================================

Return:

variables = []

ONLY when no controllable decision can be established from the user's
request.

Example:

User:
"Analyze historical shipping performance."

There is no explicit optimization decision.

Then:

variables = []
expression = ""
representation = ""

is appropriate.

But:

User:
"Determine how many products to ship in each period."

MUST NOT return variables=[] merely because the CSV lacks a shipment
quantity column.

============================================================
OBJECTIVE
============================================================

The objective must faithfully represent the user's requested goal.

Do NOT replace a cost objective with a quantity objective.

For example:

User:
"minimize shipping costs"

WRONG:

sum(ship_qty)

Correct concept:

sum(shipping_cost * ship_qty)

If shipping_cost is not available, DO NOT invent it.

Instead:

variables = [ship_qty]
expression = ""

and explain that the shipping-cost parameter is missing.

Similarly:

User:
"minimize delivery delays"

Do NOT use:

sum(ship_qty)

unless the user explicitly asked to minimize quantity.

If the required delay parameter is unavailable, leave expression empty.

============================================================
MULTIPLE OBJECTIVES
============================================================

If the user asks to minimize both shipping cost and delivery delays,
both goals must be represented.

Do NOT silently discard one of them.

If the required numerical parameters are unavailable, do not invent
weights or costs.

Instead, preserve the decision variables and leave expression empty.

The assumptions must explain what is missing.

============================================================
DEMAND
============================================================

Historical demand, orders, quantities, and sales are observations.

They may potentially be used as parameters.

Do not turn historical demand into a decision variable.

If demand can reasonably be derived from the dataset, state that as
an assumption.

Do not invent demand values.

============================================================
CONSTRAINTS
============================================================

Only infer constraints supported by the user or clearly supported by
the dataset.

If the user explicitly says:

"The quantity shipped cannot be negative."

Then:

lower_bound = 0

If the user explicitly requires whole quantities, use:

variable_type = integer

Do NOT invent capacity, budget, inventory, warehouse, vehicle, or
supply constraints without evidence.

============================================================
CONSTRAINTS — CRITICAL
============================================================

Constraints stated explicitly by the user are part of the mathematical
model and MUST be returned in the "constraints" field.

For a hard inequality such as:

"I cannot spend more than 100"
"budget is at most 100"
"cost cannot exceed 100"

construct a constraint with relation "le" and threshold 100.

When the constraint depends on a tabular parameter, use the concrete
expanded decision variables and dataset values. Example:

40*A + 35*B + 30*C <= 100

must be represented as an expression plus threshold, not merely described
in natural language.

Never invent a constraint. If a numeric threshold is not explicit or
reliably present in the dataset, leave the constraint unresolved and
state the uncertainty in assumptions.

Supported constraint relations:
- "le" = less than or equal
- "ge" = greater than or equal
- "eq" = equal

Each constraint object has:
{{
  "id": "safe_identifier",
  "name": "human readable name",
  "kind": "hard",
  "relation": "le|ge|eq",
  "expression": "...",
  "lower_bound": null,
  "upper_bound": null,
  "threshold": 100
}}

Use "threshold" for a scalar right-hand side.

============================================================
OBJECTIVE METRIC
============================================================

An objective has two layers:

1. objective_sense: minimize or maximize.
2. objective_metric: the semantic quantity being optimized.

Use an objective_metric when the problem has a native solver whose objective
is naturally computed from the problem structure and does not need a scalar
algebraic expression over decision variables.

For graph-native problems use:
- chinese_postman -> total_distance
- shortest_path -> path_length
- minimum_spanning_tree -> total_weight
- TSP/tour-length problems -> tour_length

For these native graph objectives, expression MUST remain an empty string.
Do NOT use expression = "0" merely to satisfy a generic expression contract.
The solver-specific adapter is responsible for evaluating the declared metric.

For expression-based problems such as production planning, knapsack, LP,
or MILP, objective_metric may be null and expression contains the algebraic
objective.

============================================================
SOLUTION REPRESENTATION
============================================================

If decision variables are established and their conceptual form is
a collection of quantities indexed by dimensions such as category
and period, "vector" is normally an appropriate representation.

If the objective is incomplete but the decision representation is
still clear, representation may still be "vector".

If no decision variables can be established, representation should
normally be an empty string.

============================================================
MATHEMATICAL PROPERTIES
============================================================

Only declare mathematical properties supported by the problem.

Do not declare properties merely because the CSV contains numeric,
integer, categorical, or bounded columns.

============================================================
PROBLEM FAMILY
============================================================

Choose the problem family from the allowed values according to the
actual optimization problem.

Do not choose a family merely because of dataset column types.

============================================================
EXPRESSION
============================================================

The "expression" field represents the mathematical objective.

For a non-empty expression:

- every variable identifier must exactly match a declared variable;
- only allowed mathematical functions may be used;
- dataset columns must NOT be referenced directly unless they are
  legitimate decision variables.

Allowed operators:

+
-
*
/
^

Allowed functions:

abs(...)
min(...)
max(...)
sum(...)

For a partial model, expression MUST be:

""

Do NOT invent a mathematical expression simply to make the model
appear complete.

============================================================
INCOMPLETE MODEL
============================================================

There are two valid kinds of incomplete model.

TYPE A — no decision identified:

variables = []
expression = ""
representation = ""

TYPE B — decision identified but another component is missing:

variables = [valid decision variables]
expression = ""
representation = justified representation

TYPE B is especially important.

Example:

User:
"Determine how many products to ship in each period, minimizing
shipping costs."

If shipping costs are unavailable:

variables = [ship_qty]
expression = ""
representation = "vector"

The model is incomplete because the OBJECTIVE is unresolved.

It is NOT incomplete because the decision variable is unresolved.

============================================================
FINAL CHECK
============================================================

Before returning the JSON, verify:

1. Did the user explicitly describe a controllable decision?
2. If yes, did I preserve that decision variable even if it is not
   present in the CSV?
3. Did I avoid turning historical dataset columns into decisions?
4. If objective parameters are missing, did I keep the decision
   variables instead of deleting them?
5. If the objective cannot be safely constructed, is expression empty?
6. Did I avoid inventing costs, weights, demand, capacities, or other
   parameters?
7. If the user explicitly requires non-negative quantities, is the
   lower bound 0?
8. Are variable names safe identifiers?
9. Does every identifier in a non-empty expression correspond to a
   declared variable or allowed function?
10. Are mathematical properties actually supported?
11. Are enum values taken from the allowed values?
12. If no controllable decision exists, only then should variables be
    empty.

============================================================
USER REQUEST
============================================================

{problem_description}

============================================================
DATASET
============================================================

{json.dumps(
    dataset,
    ensure_ascii=False,
    default=str,
    indent=2,
)}

============================================================
ALLOWED VALUES
============================================================

{json.dumps(
    allowed,
    ensure_ascii=False,
    default=str,
    indent=2,
)}

============================================================
GRAPH REPRESENTATIONS
============================================================

If the problem is naturally a graph problem (roads, vertices, edges,
routes, networks, shortest paths, Chinese Postman, spanning trees),
use representation = "graph".

When representation = "graph", representation_metadata should describe
the graph without inventing values. Prefer: nodes, edges, directed,
graph_problem_type, source, target. Each edge should contain id, u, v,
weight, and optionally required.

Supported graph_problem_type values include:
- chinese_postman
- shortest_path
- minimum_spanning_tree
- tsp
- generic

For a tabular edge dataset, use the row values as graph data. Common
column aliases are id/edge_id, u/from/source, v/to/target, and
weight/cost/distance. Do not invent missing edge weights.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly these top-level fields:

{{
    "name": "short problem name",
    "description": "concise interpretation of the user's problem",
    "problem_family": "one allowed problem_family",
    "mathematical_properties": [],
    "variables": [
        {{
            "name": "safe_identifier",
            "variable_type": "one allowed variable_type",
            "lower_bound": null,
            "upper_bound": null
        }}
    ],
    "objective_kind": "single or multi",
    "objective_sense": "minimize or maximize",
    "objective_metric": "semantic metric or null",
    "objective_status": "complete|incomplete|not_applicable",
    "expression": "mathematical expression or empty string",
    "representation": "one allowed representation or empty string",
    "representation_metadata": {{}},
    "problem_structure": "tabular|vector|graph|matrix|generic",
    "problem_structure_metadata": {{}},
    "constraints": [
        {{
            "id": "safe_identifier",
            "name": "human readable name",
            "kind": "hard",
            "relation": "le|ge|eq",
            "expression": "...",
            "bound": null,
            "lower_bound": null,
            "upper_bound": null,
            "threshold": null
        }}
    ],
    "explanation": "plain-language explanation",
    "assumptions": [
        "assumptions or uncertainties"
    ]
}}
"""

    # ========================================================
    # Deterministic tabular completion
    # ========================================================

    @staticmethod
    def _single_tabular_source(dataset: dict[str, Any]) -> dict[str, Any] | None:
        """Return the legacy flat source only when exactly one tabular source exists."""
        sources = dataset.get("sources")
        if isinstance(sources, list):
            if len(sources) != 1:
                return None
            source = sources[0]
            return source if isinstance(source, dict) and source.get("source_kind") == "tabular" else None
        return dataset if dataset.get("columns") else None

    def _complete_tabular_model(
        self,
        result: dict[str, Any],
        problem_description: str,
        dataset: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Completes a model when a tabular dataset provides all numerical
        parameters needed to expand an indexed decision into concrete
        scalar variables.

        The LLM remains responsible for understanding the user's intent.
        This layer only performs a deterministic, auditable transformation
        from rows/parameters to the scalar declarative model consumed by
        OptimizationProblem. It never invents numerical values.
        """
        single = self._single_tabular_source(dataset)
        if single is None or single.get("rows_truncated"):
            return result
        rows = single.get("rows") or single.get("sample_rows") or []
        if not isinstance(rows, list) or len(rows) < 1:
            return result
        if not isinstance(result, dict) or not result.get("variables"):
            return result

        columns = [str(item).strip() for item in single.get("columns", []) if str(item).strip()]
        normalized_columns = {self._normalize_text(column): column for column in columns}
        description_text = self._normalize_text(problem_description)

        # We only expand when the request clearly describes a row-wise
        # quantity decision (production, shipping, allocation, etc.).
        action = self._infer_rowwise_quantity_action(description_text)
        if action is None:
            return result

        entity_column = self._find_entity_column(rows, columns)
        if entity_column is None:
            return result

        coefficient_column, sense = self._find_objective_parameter(
            rows=rows,
            columns=columns,
            description_text=description_text,
            current_sense=result.get("objective_sense"),
        )
        if coefficient_column is None:
            return result

        lower_column = self._find_column(columns, ("min_quantity", "minimum_quantity", "min_qty", "lower_bound", "minimum", "min"))
        upper_column = self._find_column(columns, ("max_quantity", "maximum_quantity", "max_qty", "upper_bound", "maximum", "max", "capacity"))
        if lower_column is None and upper_column is None:
            return result

        # If the model already has multiple scalar variables, leave it alone.
        # This completion is intended for the common indexed-variable case.
        variables = result.get("variables", [])
        if len(variables) != 1:
            return result

        base_variable = variables[0]
        base_name = str(base_variable.get("name") or action)
        expanded_variables: list[dict[str, Any]] = []
        terms: list[str] = []
        seen_entities: set[str] = set()

        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_entity = row.get(entity_column)
            if raw_entity is None or str(raw_entity).strip() == "":
                continue
            entity = str(raw_entity).strip()
            if entity in seen_entities:
                continue
            coefficient = self._to_number(row.get(coefficient_column))
            if coefficient is None:
                return result

            lower = self._to_number(row.get(lower_column)) if lower_column else None
            upper = self._to_number(row.get(upper_column)) if upper_column else None
            if lower is None and upper is None:
                return result
            if lower is None:
                lower = 0.0
            if upper is None:
                return result
            if lower > upper:
                return result

            variable_name = self._safe_variable_name(f"{base_name}_{entity}")
            if variable_name in {item["name"] for item in expanded_variables}:
                return result
            expanded_variables.append({
                "name": variable_name,
                "variable_type": base_variable.get("variable_type") or "integer",
                "lower_bound": lower,
                "upper_bound": upper,
            })
            coefficient_text = self._format_number(coefficient)
            term = f"{coefficient_text}*{variable_name}"
            terms.append(term)
            seen_entities.add(entity)

        if not expanded_variables or not terms:
            return result

        # Preserve explicit model semantics, but correct the sense when the
        # request unambiguously says profit/revenue/cost and the draft was
        # incomplete or defaulted.
        objective_sense = sense or result.get("objective_sense") or "maximize"
        objective_expression = " + ".join(terms)

        completed = dict(result)
        completed["variables"] = expanded_variables
        completed["objective_sense"] = objective_sense
        completed["objective_kind"] = result.get("objective_kind") or "single"
        completed["expression"] = objective_expression
        completed["representation"] = result.get("representation") or "vector"
        completed["problem_family"] = self._infer_problem_family(
            result.get("problem_family"), action, description_text
        )

        properties = list(result.get("mathematical_properties") or [])
        variable_type = expanded_variables[0]["variable_type"]
        for prop in ("linear", "constrained"):
            if prop not in properties:
                properties.append(prop)
        if variable_type == "integer" and "integer" not in properties:
            properties.append("integer")
        elif variable_type == "binary" and "binary" not in properties:
            properties.append("binary")
        elif variable_type == "continuous" and "continuous" not in properties:
            properties.append("continuous")
        completed["mathematical_properties"] = properties

        assumptions = list(result.get("assumptions") or [])
        assumptions.append(
            f"The '{entity_column}' column defines the decision entities; "
            f"'{coefficient_column}' is used as the per-entity objective coefficient."
        )
        if lower_column:
            assumptions.append(f"'{lower_column}' supplies the lower production bound for each entity.")
        if upper_column:
            assumptions.append(f"'{upper_column}' supplies the upper production bound for each entity.")
        completed["assumptions"] = self._unique_strings(assumptions)

        explanation = str(result.get("explanation") or "").strip()
        completion_note = (
            f"The tabular dataset defines one decision quantity per {entity_column}; "
            f"the objective is expanded using {coefficient_column}."
        )
        completed["explanation"] = f"{explanation} {completion_note}".strip()
        return completed

    def _complete_multi_source_quantity_bounds(
        self,
        result: dict[str, Any],
        problem_description: str,
        dataset: dict[str, Any],
    ) -> dict[str, Any]:
        """Complete scalar quantity bounds from a related tabular source.

        Multi-source modeling intentionally keeps file parsing separate from
        semantic modeling. Once the LLM has identified the decision
        variables, however, a hard quantity limit explicitly stated by the
        user can be materialized deterministically from a related source.

        This closes the gap between an indexed decision (for example,
        ``purchase_qty_P001``) and a separate demand table containing
        ``product_id`` + ``monthly_demand``. No numeric value is invented:
        the bound must come directly from an uploaded source.
        """
        if not isinstance(result, dict) or not result.get("variables"):
            return result

        sources = dataset.get("sources") if isinstance(dataset, dict) else None
        if not isinstance(sources, list) or len(sources) < 2:
            return result
        if not self._description_explicitly_caps_quantity(problem_description):
            return result

        variables = result.get("variables") or []
        if not isinstance(variables, list):
            return result

        # Find a tabular source carrying a demand/capacity column and a key.
        demand_source = None
        entity_column = None
        upper_column = None
        for source in sources:
            if not isinstance(source, dict) or source.get("source_kind") != "tabular":
                continue
            if source.get("rows_truncated"):
                continue
            rows = source.get("rows") or source.get("sample_rows") or []
            columns = [str(c).strip() for c in source.get("columns", []) if str(c).strip()]
            if not rows or not columns:
                continue
            candidate_entity = self._find_entity_column(rows, columns)
            candidate_upper = self._find_column(
                columns,
                ("monthly_demand", "demand", "estimated_demand", "expected_demand", "max_quantity", "maximum_quantity", "max_qty", "capacity"),
            )
            if candidate_entity and candidate_upper:
                demand_source = source
                entity_column = candidate_entity
                upper_column = candidate_upper
                break

        if demand_source is None or entity_column is None or upper_column is None:
            return result

        rows = demand_source.get("rows") or demand_source.get("sample_rows") or []
        lookup: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = row.get(entity_column)
            value = self._to_number(row.get(upper_column))
            if key is None or value is None:
                continue
            lookup[str(key).strip()] = value

        if not lookup:
            return result

        changed = False
        completed_variables: list[dict[str, Any]] = []
        missing_keys: list[str] = []
        for variable in variables:
            if not isinstance(variable, dict):
                return result
            updated = dict(variable)
            name = str(updated.get("name") or "")
            # The LLM convention is <decision>_<entity>. Prefer the longest
            # exact entity suffix to avoid accidental partial matches.
            entity = next(
                (key for key in sorted(lookup, key=len, reverse=True) if name == key or name.endswith(f"_{key}")),
                None,
            )
            if entity is None:
                completed_variables.append(updated)
                continue

            if updated.get("upper_bound") is None:
                updated["upper_bound"] = lookup[entity]
                changed = True
            elif float(updated["upper_bound"]) != float(lookup[entity]):
                # Do not silently overwrite an explicit AI bound. A mismatch
                # is a modeling uncertainty and is left visible for review.
                completed_variables.append(updated)
                continue
            if updated.get("lower_bound") is None and updated.get("variable_type") in {"integer", "continuous"}:
                updated["lower_bound"] = 0.0
                changed = True
            completed_variables.append(updated)

        if not changed:
            return result

        completed = dict(result)
        completed["variables"] = completed_variables
        assumptions = list(result.get("assumptions") or [])
        assumptions.append(
            f"'{upper_column}' from the uploaded source '{demand_source.get('filename', 'dataset')}' "
            f"is used as the per-{entity_column} upper bound for purchase quantities."
        )
        completed["assumptions"] = self._unique_strings(assumptions)
        return completed

    @staticmethod
    def _description_explicitly_caps_quantity(description: str) -> bool:
        normalized = GroqLLMService._normalize_text(description)
        purchase_action = any(token in normalized for token in (
            "comprar", "compra", "purchase", "buy",
        ))
        demand_reference = any(token in normalized for token in (
            "demanda", "demand",
        ))
        cap_language = any(token in normalized for token in (
            "nao quero comprar mais",
            "nao posso comprar mais",
            "nao comprar mais",
            "nao ultrapassar",
            "nao exceder",
            "nao superar",
            "nao seja maior",
            "nao pode ser maior",
            "limite",
            "maximo",
            "maior do que",
            "maior que",
            "not buy more",
            "do not buy more",
            "cannot buy more",
            "not exceed",
            "do not exceed",
            "cannot exceed",
            "not greater than",
            "maximum",
        ))
        return purchase_action and demand_reference and cap_language

    def _complete_tabular_constraints(
        self,
        result: dict[str, Any],
        problem_description: str,
        dataset: dict[str, Any],
    ) -> dict[str, Any]:
        """Materialize explicit budget/cost limits using tabular parameters.

        This is deliberately conservative: the numeric limit must come from
        the user's text, while coefficients must come from the dataset.
        """
        if not isinstance(result, dict) or not result.get("variables"):
            return result
        single = self._single_tabular_source(dataset)
        if single is None or single.get("rows_truncated"):
            return result
        rows = single.get("rows") or single.get("sample_rows") or []
        columns = [str(item).strip() for item in single.get("columns", []) if str(item).strip()]
        if not rows or not columns:
            return result

        description = self._normalize_text(problem_description)
        threshold = self._extract_budget_limit(problem_description)
        if threshold is None:
            return result

        cost_column = self._find_column(columns, ("cost", "custo", "price", "preco", "expense", "despesa", "investment", "investimento"))
        entity_column = self._find_entity_column(rows, columns)
        if cost_column is None or entity_column is None:
            return result

        variables = result.get("variables", [])
        if not variables:
            return result

        terms: list[str] = []
        variable_names = {item.get("name") for item in variables if isinstance(item, dict)}
        for row in rows:
            if not isinstance(row, dict):
                return result
            entity = row.get(entity_column)
            cost = self._to_number(row.get(cost_column))
            if entity is None or cost is None:
                return result
            entity_text = str(entity).strip()
            exact = self._safe_variable_name(entity_text)
            matching = [
                name for name in variable_names
                if name == exact or name.endswith(f"_{exact}")
            ]
            if len(matching) != 1:
                return result
            name = matching[0]
            terms.append(f"{self._format_number(cost)}*{name}")

        if not terms:
            return result

        constraints = list(result.get("constraints") or [])
        if any(str(item.get("id", "")) == "budget_limit" for item in constraints if isinstance(item, dict)):
            return result

        constraints.append({
            "id": "budget_limit",
            "name": "Budget limit",
            "kind": "hard",
            "relation": "le",
            "expression": " + ".join(terms),
            "lower_bound": None,
            "upper_bound": None,
            "threshold": threshold,
        })

        completed = dict(result)
        completed["constraints"] = constraints
        properties = list(completed.get("mathematical_properties") or [])
        for prop in ("linear", "constrained"):
            if prop not in properties:
                properties.append(prop)
        completed["mathematical_properties"] = properties
        assumptions = list(completed.get("assumptions") or [])
        assumptions.append(f"The user's budget limit is modeled as a hard constraint of {self._format_number(threshold)} using the '{cost_column}' dataset column.")
        completed["assumptions"] = self._unique_strings(assumptions)
        return completed

    def _extract_budget_limit(self, text: str) -> float | None:
        patterns = (
            r"(?:nao posso|não posso|nao devo|não devo)\s+(?:gastar|investir)\s+(?:mais de|acima de)\s*([0-9]+(?:[.,][0-9]+)?)\s*(mil|milhao|milhoes|k|m)?",
            r"(?:orcamento|orçamento|budget|verba)\s+(?:e|é|de|igual a|no maximo de|no máximo de|ate|até)\s*([0-9]+(?:[.,][0-9]+)?)\s*(mil|milhao|milhoes|k|m)?",
            r"(?:nao posso|não posso|nao devo|não devo)\s+(?:ultrapassar|exceder)\s*([0-9]+(?:[.,][0-9]+)?)\s*(mil|milhao|milhoes|k|m)?",
            r"(?:at most|no more than|maximum of|cannot exceed|cannot spend more than)\s*([0-9]+(?:[.,][0-9]+)?)\s*(k|m)?",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            value = self._to_number(match.group(1))
            if value is None:
                continue
            unit = (match.group(2) or "").lower()
            if unit in {"mil", "k"}:
                value *= 1000
            elif unit in {"milhao", "milhoes", "m"}:
                value *= 1_000_000
            return value
        return None

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").strip().lower()
        replacements = {
            "á": "a", "à": "a", "ã": "a", "â": "a", "ä": "a",
            "é": "e", "ê": "e", "ë": "e",
            "í": "i", "ï": "i",
            "ó": "o", "õ": "o", "ô": "o", "ö": "o",
            "ú": "u", "ü": "u", "ç": "c",
        }
        return "".join(replacements.get(char, char) for char in text)

    def _infer_rowwise_quantity_action(self, description: str) -> str | None:
        if any(token in description for token in ("produzir", "producao", "fabricar", "production", "produce")):
            return "produce_qty"
        if any(token in description for token in ("enviar", "envio", "embarcar", "shipping", "ship")):
            return "ship_qty"
        if any(token in description for token in ("alocar", "alocacao", "allocate", "allocation")):
            return "allocate_qty"
        if any(token in description for token in ("comprar", "compra", "purchase", "procurement")):
            return "purchase_qty"
        return None

    def _find_objective_parameter(
        self,
        rows: list[dict[str, Any]],
        columns: list[str],
        description_text: str,
        current_sense: Any,
    ) -> tuple[str | None, str | None]:
        candidates: list[tuple[int, str, str]] = []
        objective_words = []
        sense = current_sense if current_sense in {"minimize", "maximize"} else None
        if any(token in description_text for token in ("lucro", "profit", "margem", "revenue", "receita", "faturamento")):
            objective_words.extend(("profit", "lucro", "revenue", "receita", "margin", "margem", "price", "preco"))
            sense = "maximize"
        if any(token in description_text for token in ("custo", "cost", "despesa", "expense")):
            objective_words.extend(("cost", "custo", "expense", "despesa"))
            sense = "minimize"
        if not objective_words:
            return None, sense

        for column in columns:
            normalized = self._normalize_text(column)
            if column.lower() in {"product", "produto", "item", "category", "categoria", "id", "name", "nome"}:
                continue
            values = [self._to_number(row.get(column)) for row in rows if isinstance(row, dict)]
            if not values or any(value is None for value in values):
                continue
            score = 0
            for word in objective_words:
                if self._normalize_text(word) in normalized:
                    score += 10
            if score:
                candidates.append((score, column, sense or "maximize"))
        if not candidates:
            return None, sense
        candidates.sort(reverse=True)
        _, column, inferred_sense = candidates[0]
        return column, inferred_sense

    def _find_entity_column(self, rows: list[dict[str, Any]], columns: list[str]) -> str | None:
        preferred = ("product", "produto", "item", "category", "categoria", "sku", "id", "name", "nome")
        for candidate in preferred:
            column = self._find_column(columns, (candidate,))
            if column:
                values = {str(row.get(column, "")).strip() for row in rows if isinstance(row, dict)}
                if values and any(value for value in values):
                    return column
        for column in columns:
            values = [row.get(column) for row in rows if isinstance(row, dict)]
            if len(values) >= 2 and all(self._to_number(value) is None for value in values):
                return column
        return None

    def _find_column(self, columns: list[str], aliases: tuple[str, ...]) -> str | None:
        normalized_aliases = {self._normalize_text(alias) for alias in aliases}
        for column in columns:
            normalized = self._normalize_text(column)
            if normalized in normalized_aliases:
                return column
        return None

    @staticmethod
    def _to_number(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(str(value).strip().replace(",", "."))
        except (TypeError, ValueError):
            return None
        return number

    @staticmethod
    def _format_number(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return format(value, ".15g")

    @staticmethod
    def _safe_variable_name(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
        normalized = re.sub(r"_+", "_", normalized).strip("_") or "x"
        if normalized[0].isdigit():
            normalized = f"v_{normalized}"
        return normalized

    @staticmethod
    def _infer_problem_family(current: Any, action: str, description: str) -> str:
        if current and current != "generic":
            return current
        if action == "produce_qty":
            return "production_planning"
        return "generic"

    @staticmethod
    def _unique_strings(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value and value not in seen:
                result.append(value)
                seen.add(value)
        return result

    # ========================================================
    # Repair prompt
    # ========================================================

    def _build_repair_prompt(
        self,
        original_result: dict[str, Any],
        validation_error: str,
        dataset: dict[str, Any],
        problem_description: str,
    ) -> str:

        allowed = dataset.get(
            "allowed_values",
            {},
        )

        return f"""
You are repairing an optimization model.

The previous AI response failed backend validation.

Your task is to correct the response WITHOUT inventing information.

============================================================
ORIGINAL USER REQUEST
============================================================

{problem_description}

============================================================
VALIDATION ERROR
============================================================

{validation_error}

============================================================
PREVIOUS MODEL
============================================================

{json.dumps(
    original_result,
    ensure_ascii=False,
    default=str,
    indent=2,
)}

============================================================
DATASET
============================================================

{json.dumps(
    dataset,
    ensure_ascii=False,
    default=str,
    indent=2,
)}

============================================================
CRITICAL REPAIR RULES
============================================================

1. The user's explicit decision has priority over the dataset.

2. If the user explicitly says what should be chosen, determined,
   assigned, allocated, scheduled, shipped, produced, or controlled,
   preserve that as a decision variable.

3. A decision variable does NOT need to exist as a CSV column.

4. Never delete an explicit decision variable merely because the
   objective parameters are missing.

5. Historical dataset columns are observations by default.

6. Do not turn observed historical outcomes into decisions.

7. A model can have decision variables while still being incomplete.

8. If a decision is known but the objective cannot be constructed,
   keep the variables and set:

   expression = ""

9. Do not invent shipping costs, delay costs, demand, weights,
   capacities, budgets, or other missing parameters.

10. If the user explicitly says quantities cannot be negative,
    use lower_bound = 0.

11. If the user explicitly requires whole quantities, use an integer
    variable.

12. If no controllable decision is established by the user, then and
    only then may variables be empty.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON with exactly these fields:

{{
    "name": "...",
    "description": "...",
    "problem_family": "...",
    "mathematical_properties": [],
    "variables": [],
    "objective_kind": "...",
    "objective_sense": "...",
    "objective_metric": null,
    "expression": "",
    "constraints": [],
    "representation": "",
    "representation_metadata": {{}},
    "problem_structure": "tabular|vector|graph|matrix|generic",
    "problem_structure_metadata": {{}},
    "explanation": "...",
    "assumptions": []
}}
"""

    def _deterministic_graph_fallback(
        self,
        problem_description: str,
        dataset: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a minimal graph model when the provider cannot emit JSON.

        This is intentionally data-driven: it recognizes the graph shape from
        the dataset and derives only the semantic family/metric from the
        centralized graph classifier. No solver or algorithm is selected here.
        """
        rows = dataset.get("rows") or dataset.get("sample_rows") or []
        if not isinstance(rows, list) or not rows:
            raise ValueError("The AI provider failed and the dataset is not graph-shaped enough for deterministic fallback.")

        classifier = GraphProblemClassifier()
        classification = classifier.classify(problem_description)
        if classification.kind == "generic":
            raise ValueError("The AI provider failed and the graph problem family could not be determined deterministically.")

        return {
            "name": classification.kind.replace("_", " ").title(),
            "description": problem_description.strip(),
            "problem_family": "graph_optimization",
            "mathematical_properties": ["discrete", "combinatorial"],
            "variables": [{"name": "route", "variable_type": "discrete", "lower_bound": None, "upper_bound": None}],
            "objective_kind": "single",
            "objective_sense": "minimize",
            "objective_metric": {
                "tsp": "tour_length",
                "shortest_path": "path_length",
                "minimum_spanning_tree": "total_weight",
                "chinese_postman": "total_distance",
            }.get(classification.kind, "custom"),
            "objective_status": "complete",
            "expression": "",
            "representation": "graph",
            "representation_metadata": {},
            "problem_structure": "graph",
            "problem_structure_metadata": {},
            "constraints": [],
            "explanation": "Graph problem modeled deterministically from the natural-language request and edge dataset after provider JSON generation failed.",
            "assumptions": ["The supplied edge table represents the graph instance."]
        }

    # ========================================================
    # Deterministic graph completion
    # ========================================================

    def _complete_graph_model(
        self,
        result: dict[str, Any],
        problem_description: str,
        dataset: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize an edge table into graph metadata without inventing data."""
        if not isinstance(result, dict) or (result.get("representation") != "graph" and result.get("problem_structure") != "graph"):
            return result
        rows = dataset.get("rows") or dataset.get("sample_rows") or []
        if not isinstance(rows, list) or not rows:
            return result
        if dataset.get("rows_truncated"):
            return result

        def pick(row, aliases):
            lowered = {str(key).lower(): key for key in row}
            for alias in aliases:
                if alias in lowered:
                    return row[lowered[alias]]
            return None

        edges = []
        nodes = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                return result
            u = pick(row, ("u", "from", "source", "origin"))
            v = pick(row, ("v", "to", "target", "destination"))
            weight = pick(row, ("weight", "cost", "distance", "length"))
            edge_id = pick(row, ("id", "edge_id", "edge", "arc"))
            if u is None or v is None or weight is None:
                return result
            if edge_id is None:
                edge_id = f"e{index + 1}"
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError):
                return result
            if numeric_weight < 0:
                return result
            edges.append({"id": edge_id, "u": u, "v": v, "weight": numeric_weight, "required": True})
            nodes.add(u)
            nodes.add(v)

        completed = dict(result)
        metadata = dict(completed.get("problem_structure_metadata") or {})
        legacy_metadata = dict(completed.get("representation_metadata") or {})
        metadata = {**legacy_metadata, **metadata}
        metadata.setdefault("nodes", list(nodes))
        metadata["edges"] = edges
        metadata.setdefault("directed", False)
        description = f"{problem_description} {completed.get('description', '')}".lower()

        # Source/target are semantic graph parameters, not edge-column names.
        # Preserve them when the model already identified them; otherwise
        # infer them only from explicit natural-language endpoint patterns.
        # This prevents the validation layer from turning a valid shortest-path
        # request into a misleading "0 compatible algorithms" result.
        if metadata.get("source") is None or metadata.get("target") is None:
            endpoint_patterns = (
                r"\bfrom\s+([^,.;]+?)\s+to\s+([^,.;]+)",
                r"\bde\s+([^,.;]+?)\s+para\s+([^,.;]+)",
                r"\bentre\s+([^,.;]+?)\s+e\s+([^,.;]+)",
                r"\bentre\s+([^,.;]+?)\s+até\s+([^,.;]+)",
                r"\bentre\s+([^,.;]+?)\s+ate\s+([^,.;]+)",
            )
            for pattern in endpoint_patterns:
                match = re.search(pattern, description, flags=re.IGNORECASE)
                if not match:
                    continue
                source_text = match.group(1).strip().strip('\"\'')
                target_text = match.group(2).strip().strip('\"\'')
                node_lookup = {str(node).strip().lower(): node for node in nodes}
                source_value = node_lookup.get(source_text.lower(), source_text)
                target_value = node_lookup.get(target_text.lower(), target_text)
                if source_value in nodes and target_value in nodes and source_value != target_value:
                    metadata["source"] = source_value
                    metadata["target"] = target_value
                break
        graph_type = metadata.get("graph_problem_type")
        if graph_type in (None, "", "generic"):
            classification = GraphProblemClassifier().classify(
                problem_description,
                completed.get("description", ""),
                completed.get("name", ""),
            )
            graph_type = classification.kind
        metadata["graph_problem_type"] = graph_type

        # Graph-native solvers do not optimize one variable per edge.
        # The graph itself is the instance; the decision is the traversal
        # returned by the graph solver. LLMs may otherwise hallucinate
        # variables such as edge_count_AB, which then collide with the
        # generic integer-domain and routing validators. Normalize the
        # semantic model here so graph problems have one discrete route
        # variable and the graph data remains exclusively in metadata.
        completed["problem_family"] = "graph_optimization"
        completed["representation"] = "graph"
        completed["variables"] = [{
            "name": "route",
            "variable_type": "discrete",
            "lower_bound": None,
            "upper_bound": None,
        }]
        # Native graph solvers calculate the objective from the graph.
        # Do not invent a fake algebraic expression just to satisfy
        # expression-based solvers. The semantic metric is the contract.
        if graph_type == "chinese_postman":
            objective_metric = "total_distance"
        elif graph_type == "shortest_path":
            objective_metric = "path_length"
        elif graph_type == "minimum_spanning_tree":
            objective_metric = "total_weight"
        elif graph_type == "tsp":
            # TSP is a tour problem, not a generic weighted graph objective.
            # Normalize even when the LLM supplied a generic metric such as
            # total_distance so capability matching can target tour solvers.
            objective_metric = "tour_length"
        else:
            objective_metric = completed.get("objective_metric")
        completed["objective_metric"] = objective_metric
        completed["expression"] = ""
        # Native graph constraints are structural (required edges, source/target,
        # connectivity, etc.), not algebraic constraints over scalar variables.
        # Preserve the LLM proposal in graph metadata instead of sending edge
        # identifiers such as e1/e2/e3 through the generic expression parser.
        graph_constraints = completed.get("constraints") or []
        if graph_constraints:
            metadata["graph_constraints"] = list(graph_constraints)
            completed["constraints"] = []
        completed["objective_status"] = "complete" if objective_metric else "incomplete"
        completed["mathematical_properties"] = [
            "discrete",
            "combinatorial",
        ]
        completed["problem_structure"] = "graph"
        completed["problem_structure_metadata"] = metadata
        # Keep the legacy metadata synchronized so older clients keep working.
        completed["representation_metadata"] = dict(metadata)
        return completed

    # ========================================================
    # Validation
    # ========================================================

    def _validate_model(
        self,
        result: dict[str, Any],
        dataset: dict[str, Any],
    ) -> None:

        result["objective_status"] = self._derive_objective_status(result)

        self._validate_top_level_fields(result)

        self._validate_basic_fields(result)

        self._validate_variables(
            result["variables"]
        )

        self._validate_expression(
            expression=result["expression"],
            variables=result["variables"],
        )

        self._validate_constraints(
            constraints=result.get("constraints", []),
            variables=result["variables"],
        )

        self._validate_completeness(result)

        self._validate_allowed_values(
            result=result,
            dataset=dataset,
        )

    # ========================================================
    # Top-level structure
    # ========================================================

    def _validate_top_level_fields(
        self,
        result: dict[str, Any],
    ) -> None:

        missing = REQUIRED_FIELDS - result.keys()

        if missing:
            raise ValueError(
                "Problem model is missing required fields: "
                + ", ".join(sorted(missing))
            )

        unexpected = result.keys() - REQUIRED_FIELDS - OPTIONAL_FIELDS

        if unexpected:
            raise ValueError(
                "Problem model contains unexpected fields: "
                + ", ".join(sorted(unexpected))
            )

    # ========================================================
    # Basic fields
    # ========================================================

    def _validate_basic_fields(
        self,
        result: dict[str, Any],
    ) -> None:

        if not isinstance(result["name"], str):
            raise ValueError(
                "'name' must be a string."
            )

        if not isinstance(result["description"], str):
            raise ValueError(
                "'description' must be a string."
            )

        if result["objective_kind"] not in {
            "single",
            "multi",
        }:
            raise ValueError(
                "objective_kind must be 'single' or 'multi'."
            )

        if result["objective_sense"] not in {
            "minimize",
            "maximize",
        }:
            raise ValueError(
                "objective_sense must be 'minimize' or 'maximize'."
            )

        if result.get("objective_status") not in {
            "complete",
            "incomplete",
            "not_applicable",
        }:
            raise ValueError(
                "objective_status must be 'complete', 'incomplete' or 'not_applicable'."
            )

        objective_metric = result.get("objective_metric")
        if objective_metric is not None:
            allowed_metrics = {
                "total_distance", "total_cost", "total_return", "tour_length",
                "path_length", "total_weight", "mse", "mae", "custom",
            }
            if objective_metric not in allowed_metrics:
                raise ValueError(f"Unsupported objective_metric '{objective_metric}'.")

        if not isinstance(
            result["mathematical_properties"],
            list,
        ):
            raise ValueError(
                "mathematical_properties must be a list."
            )

        if not isinstance(
            result["assumptions"],
            list,
        ):
            raise ValueError(
                "assumptions must be a list."
            )

        if not isinstance(
            result["explanation"],
            str,
        ):
            raise ValueError(
                "explanation must be a string."
            )

    # ========================================================
    # Variables
    # ========================================================

    def _validate_variables(
        self,
        variables: Any,
    ) -> None:

        if not isinstance(variables, list):
            raise ValueError(
                "'variables' must be a list."
            )

        # Empty variables are allowed for a model where no
        # controllable decision could be established.
        if not variables:
            return

        names: set[str] = set()

        for variable in variables:

            if not isinstance(variable, dict):
                raise ValueError(
                    "Each variable must be an object."
                )

            required = {
                "name",
                "variable_type",
                "lower_bound",
                "upper_bound",
            }

            missing = required - variable.keys()

            if missing:
                raise ValueError(
                    "Variable is missing fields: "
                    + ", ".join(sorted(missing))
                )

            name = variable["name"]

            if not isinstance(name, str):
                raise ValueError(
                    "Variable name must be a string."
                )

            if not SAFE_IDENTIFIER_PATTERN.match(name):
                raise ValueError(
                    f"Invalid variable name '{name}'. "
                    "Variable names must be safe identifiers."
                )

            if name in names:
                raise ValueError(
                    f"Duplicate decision variable '{name}'."
                )

            names.add(name)

    # ========================================================
    # Expression
    # ========================================================

    def _validate_expression(
        self,
        expression: Any,
        variables: list[dict[str, Any]],
    ) -> None:

        if not isinstance(expression, str):
            raise ValueError(
                "'expression' must be a string."
            )

        variable_names = {
            variable["name"]
            for variable in variables
        }

        # No variables means there cannot be an expression.
        if not variables:
            if expression.strip():
                raise ValueError(
                    "A model without decision variables cannot "
                    "contain an objective expression."
                )
            return

        # IMPORTANT:
        #
        # Variables may already be known even when the objective
        # cannot yet be constructed.
        #
        # Therefore an empty expression is valid for a partial model.
        if not expression.strip():
            return

        identifiers = set(
            EXPRESSION_IDENTIFIER_PATTERN.findall(
                expression
            )
        )

        unknown_identifiers = (
            identifiers
            - variable_names
            - ALLOWED_EXPRESSION_FUNCTIONS
        )

        if unknown_identifiers:
            unknown = ", ".join(
                sorted(unknown_identifiers)
            )

            raise ValueError(
                "Expression references unknown variable(s): "
                f"{unknown}."
            )

        forbidden_patterns = [
            "import ",
            "__",
            "lambda",
            "eval(",
            "exec(",
            "open(",
            "os.",
            "subprocess",
        ]

        expression_lower = expression.lower()

        for pattern in forbidden_patterns:
            if pattern in expression_lower:
                raise ValueError(
                    "Expression contains forbidden content."
                )

    def _validate_constraints(
        self,
        constraints: Any,
        variables: list[dict[str, Any]],
    ) -> None:
        if constraints is None:
            return
        if not isinstance(constraints, list):
            raise ValueError("'constraints' must be a list.")

        seen: set[str] = set()
        allowed_relations = {"le", "ge", "eq"}
        variable_names = {variable["name"] for variable in variables}

        for constraint in constraints:
            if not isinstance(constraint, dict):
                raise ValueError("Each constraint must be an object.")

            required = {"id", "name", "kind", "relation"}
            missing = required - constraint.keys()
            if missing:
                raise ValueError("Constraint is missing fields: " + ", ".join(sorted(missing)))

            cid = constraint["id"]
            if not isinstance(cid, str) or not SAFE_IDENTIFIER_PATTERN.match(cid):
                raise ValueError(f"Invalid constraint id '{cid}'.")
            if cid in seen:
                raise ValueError(f"Duplicate constraint '{cid}'.")
            seen.add(cid)

            if constraint["kind"] != "hard":
                raise ValueError("Only hard constraints are currently executable.")
            relation = constraint["relation"]
            if relation not in allowed_relations:
                raise ValueError("Constraint relation must be 'le', 'ge' or 'eq'.")

            expression = constraint.get("expression")
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError(f"Constraint '{cid}' must define an expression.")
            self._validate_expression(expression=expression, variables=variables)

            # A constraint has one semantic target. Legacy bound fields remain
            # accepted for compatibility, but none of them is mandatory when
            # another valid target is present.
            bound = constraint.get("bound")
            threshold = constraint.get("threshold")
            lower = constraint.get("lower_bound")
            upper = constraint.get("upper_bound")
            has_target = bound is not None or threshold is not None or lower is not None or upper is not None
            if not has_target:
                raise ValueError(f"Constraint '{cid}' needs a bound/threshold.")

            if relation == "le" and bound is None and threshold is None and upper is None:
                raise ValueError(f"Constraint '{cid}' with relation 'le' needs an upper bound or threshold.")
            if relation == "ge" and bound is None and threshold is None and lower is None:
                raise ValueError(f"Constraint '{cid}' with relation 'ge' needs a lower bound or threshold.")
            if relation == "eq" and bound is None and threshold is None and not (lower is not None and upper is not None and lower == upper):
                raise ValueError(f"Constraint '{cid}' with relation 'eq' needs a bound or threshold.")

    def _derive_objective_status(self, result: dict[str, Any]) -> str:
        """Derive objective completeness without fabricating missing math."""
        variables = result.get("variables") or []
        kind = result.get("objective_kind", "single")
        expression = str(result.get("expression") or "").strip()
        metric = result.get("objective_metric")

        if not variables:
            return "not_applicable"

        if kind == "multi":
            objectives = result.get("objectives") or []
            return "complete" if len(objectives) >= 2 else "incomplete"

        return "complete" if expression or metric else "incomplete"

    # ========================================================
    # Completeness
    # ========================================================

    def _validate_completeness(
        self,
        result: dict[str, Any],
    ) -> None:
        """
        Validates both complete and partially determined models.

        Valid states:

        1. No decision identified:
           variables=[]
           expression=""
           representation=""

        2. Decision identified but objective incomplete:
           variables=[...]
           expression=""
           representation="vector"

        3. Complete model:
           variables=[...]
           expression="..."
           representation="..."
        """

        variables = result["variables"]
        expression = result["expression"].strip()
        representation = result["representation"]
        objective_metric = result.get("objective_metric")
        objective_status = result.get("objective_status") or self._derive_objective_status(result)

        # ----------------------------------------------------
        # No decision variables
        # ----------------------------------------------------

        if not variables:

            if expression:
                raise ValueError(
                    "A model without decision variables cannot "
                    "contain an objective expression."
                )

            if representation not in {"", None}:
                raise ValueError(
                    "A model without decision variables should "
                    "not declare a solution representation."
                )

            if not result["assumptions"]:
                raise ValueError(
                    "An incomplete model without decision variables "
                    "must state its uncertainty in assumptions."
                )

            return

        # ----------------------------------------------------
        # Decision variables exist, but objective is incomplete
        # ----------------------------------------------------

        if not expression:

            if objective_status != "incomplete" and objective_metric is None:
                raise ValueError("Objective status is inconsistent with the objective fields.")

            if objective_metric is not None:
                if representation in {"", None}:
                    raise ValueError(
                        "A metric-based objective with decision variables "
                        "must declare a solution representation."
                    )
                return

            if representation in {"", None}:
                raise ValueError(
                    "A model with identified decision variables "
                    "must declare a solution representation."
                )

            if not result["assumptions"]:
                raise ValueError(
                    "A partially determined model must explain "
                    "what is still unresolved."
                )

            return

        # ----------------------------------------------------
        # Complete executable candidate
        # ----------------------------------------------------

        if representation in {"", None}:
            raise ValueError(
                "A model with decision variables and an objective "
                "expression must declare a solution representation."
            )
        if objective_status != "complete":
            raise ValueError("Objective status is inconsistent with a complete objective.")

    # ========================================================
    # Allowed values
    # ========================================================

    def _validate_allowed_values(
        self,
        result: dict[str, Any],
        dataset: dict[str, Any],
    ) -> None:

        allowed = dataset.get(
            "allowed_values",
            {},
        )

        if not isinstance(allowed, dict):
            return

        # ----------------------------------------------------
        # Problem family
        # ----------------------------------------------------

        allowed_problem_families = allowed.get(
            "problem_families"
        )

        if (
            isinstance(allowed_problem_families, list)
            and allowed_problem_families
            and result["problem_family"]
            not in allowed_problem_families
        ):
            raise ValueError(
                f"Invalid problem_family "
                f"'{result['problem_family']}'. "
                f"Allowed values: {allowed_problem_families}"
            )

        # ----------------------------------------------------
        # Representation
        # ----------------------------------------------------

        allowed_representations = allowed.get(
            "representations"
        )

        if (
            isinstance(allowed_representations, list)
            and allowed_representations
            and result["representation"] not in {
                "",
                None,
            }
            and result["representation"]
            not in allowed_representations
        ):
            raise ValueError(
                f"Invalid representation "
                f"'{result['representation']}'. "
                f"Allowed values: {allowed_representations}"
            )

        # ----------------------------------------------------
        # Problem structure
        # ----------------------------------------------------

        allowed_structures = allowed.get("problem_structures")
        if (
            isinstance(allowed_structures, list)
            and allowed_structures
            and result.get("problem_structure") not in {"", None}
            and result.get("problem_structure") not in allowed_structures
        ):
            raise ValueError(
                f"Invalid problem_structure '{result.get('problem_structure')}'. "
                f"Allowed values: {allowed_structures}"
            )

        # ----------------------------------------------------
        # Mathematical properties
        # ----------------------------------------------------

        allowed_properties = allowed.get(
            "mathematical_properties"
        )

        if (
            isinstance(allowed_properties, list)
            and allowed_properties
        ):
            invalid_properties = [
                value
                for value in result["mathematical_properties"]
                if value not in allowed_properties
            ]

            if invalid_properties:
                raise ValueError(
                    "Invalid mathematical_properties: "
                    + ", ".join(invalid_properties)
                )

        # ----------------------------------------------------
        # Variable types
        # ----------------------------------------------------

        allowed_variable_types = allowed.get(
            "variable_types"
        )

        if (
            isinstance(allowed_variable_types, list)
            and allowed_variable_types
        ):
            for variable in result["variables"]:
                if (
                    variable["variable_type"]
                    not in allowed_variable_types
                ):
                    raise ValueError(
                        f"Invalid variable_type "
                        f"'{variable['variable_type']}'. "
                        f"Allowed values: {allowed_variable_types}"
                    )