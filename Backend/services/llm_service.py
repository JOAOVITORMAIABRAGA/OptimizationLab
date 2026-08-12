from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

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

        result = self._request_model(prompt)
        result = self._complete_tabular_model(
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

        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model,
            temperature=0,
            response_format={
                "type": "json_object"
            },
        )

        content = (
            response.choices[0].message.content
            or "{}"
        )

        try:
            result = json.loads(content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Groq returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "Groq returned an invalid problem model."
            )

        return result

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
    "expression": "mathematical expression or empty string",
    "representation": "one allowed representation or empty string",
    "explanation": "plain-language explanation",
    "assumptions": [
        "assumptions or uncertainties"
    ]
}}
"""

    # ========================================================
    # Deterministic tabular completion
    # ========================================================

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
        if dataset.get("rows_truncated"):
            return result
        rows = dataset.get("rows") or dataset.get("sample_rows") or []
        if not isinstance(rows, list) or len(rows) < 1:
            return result
        if not isinstance(result, dict) or not result.get("variables"):
            return result

        columns = [str(item).strip() for item in dataset.get("columns", []) if str(item).strip()]
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
    "expression": "",
    "representation": "",
    "explanation": "...",
    "assumptions": []
}}
"""

    # ========================================================
    # Validation
    # ========================================================

    def _validate_model(
        self,
        result: dict[str, Any],
        dataset: dict[str, Any],
    ) -> None:

        self._validate_top_level_fields(result)

        self._validate_basic_fields(result)

        self._validate_variables(
            result["variables"]
        )

        self._validate_expression(
            expression=result["expression"],
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

        unexpected = result.keys() - REQUIRED_FIELDS

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