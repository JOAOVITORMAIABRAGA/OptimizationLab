from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq


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


class GroqLLMService:
    """
    Adapter responsável por transformar uma descrição natural de um
    problema de otimização em um modelo de domínio estruturado.

    A LLM:
    - interpreta o problema;
    - identifica possíveis variáveis de decisão;
    - propõe uma função objetivo;
    - explica suas premissas.

    A LLM NÃO:
    - executa código;
    - gera Python;
    - executa expressões;
    - recebe acesso direto ao otimizador;
    - decide arbitrariamente que colunas do dataset são variáveis
      de decisão.

    A resposta da LLM é validada antes de ser devolvida ao restante
    da aplicação.
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
        Generates and validates an optimization problem model.

        A primeira resposta da IA é validada localmente.
        Caso seja inconsistente, fazemos uma única tentativa
        automática de correção.
        """

        prompt = self._build_prompt(
            problem_description=problem_description,
            dataset=dataset,
        )

        result = self._request_model(prompt)

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
            )

            repaired_result = self._request_model(
                repair_prompt
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
You are the Optimization Lab problem-modeling assistant.

Your ONLY responsibility is to translate the user's natural-language
optimization problem and the supplied dataset summary into a proposed
domain-level optimization model.

You do NOT execute anything.

You do NOT write Python.

You do NOT write executable code.

You do NOT write constraints_code.

You do NOT write eval expressions.

You do NOT generate scripts.

You only produce a structured mathematical/domain interpretation.

============================================================
USER PROBLEM
============================================================

{problem_description}

============================================================
DATASET SUMMARY
============================================================

{json.dumps(
    dataset,
    ensure_ascii=False,
    default=str,
    indent=2,
)}

============================================================
ALLOWED DOMAIN VALUES
============================================================

{json.dumps(
    allowed,
    ensure_ascii=False,
    default=str,
    indent=2,
)}

============================================================
IMPORTANT CONCEPT: DATA VS DECISION VARIABLES
============================================================

The dataset contains observations, historical records, features,
parameters and other information about the problem.

Dataset columns are NOT decision variables by default.

A dataset column such as:

- Sales
- Shipping_Cost
- Order_Item_Profit_Ratio
- Customer_Latitude
- Customer_Longitude
- Order_Item_Quantity

must NOT automatically become a decision variable.

A decision variable represents something that the optimization
algorithm is allowed to CHOOSE.

For example:

- quantity to ship;
- quantity to produce;
- warehouse assignment;
- route selection;
- vehicle assignment;
- resource allocation.

Dataset columns should generally be treated as parameters,
observations or evidence that helps understand the problem.

Only treat a dataset field as a decision variable if the user's
description clearly indicates that the optimizer is allowed to
choose or control it.

============================================================
MATHEMATICAL EXPRESSION RULES
============================================================

The "expression" field represents the objective function.

Every identifier used inside "expression" MUST correspond exactly
to the "name" of a declared decision variable.

For example, if:

variables = [
    {{"name": "quantity", ...}},
    {{"name": "shipping_cost", ...}}
]

then:

"quantity * shipping_cost"

is valid.

But:

"quantity * Sales"

is INVALID if "Sales" is not declared as a decision variable.

Likewise:

"Order_Item_Profit_Ratio"

is INVALID if that name does not appear in "variables".

NEVER reference dataset columns directly inside "expression"
unless they have explicitly and legitimately been defined as
decision variables.

Allowed mathematical operators:

+ - * / ^

Allowed functions:

abs(...)
min(...)
max(...)
sum(...)

No other functions are allowed.

Do not use Python syntax.

Do not use SQL.

Do not use programming constructs.

============================================================
DECISION VARIABLE RULES
============================================================

Each variable must:

1. Have a safe identifier as its name.
2. Represent an actual decision that the optimizer can make.
3. Have a valid variable_type.
4. Have sensible bounds when they can be inferred.
5. Not simply be a copy of a dataset column.

Prefer a small number of meaningful decision variables.

Do not create dozens of variables merely because the CSV has many
columns.

============================================================
UNCERTAINTY
============================================================

If the user's problem description does not contain enough information
to determine the exact mathematical model:

- choose the simplest defensible interpretation;
- explicitly state the uncertainty in "assumptions";
- do NOT invent unsupported domain rules;
- do NOT invent arbitrary decision variables;
- do NOT pretend the model is more precise than the information allows.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

The JSON MUST contain exactly these top-level fields:

{{
    "name": "short problem name",

    "description": "concise interpretation of the user's problem",

    "problem_family": "one allowed problem_family",

    "mathematical_properties": [
        "zero or more allowed properties"
    ],

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

    "expression": "safe mathematical expression",

    "representation": "one allowed representation",

    "explanation": "plain-language explanation of your interpretation",

    "assumptions": [
        "explicit assumptions or uncertainties"
    ]
}}

============================================================
FINAL SELF-CHECK
============================================================

Before returning the JSON, verify all of the following:

1. Every expression identifier exists in variables.

2. No dataset column is used as an expression identifier unless
   it is intentionally and legitimately a decision variable.

3. Every variable has a safe identifier.

4. No executable code exists anywhere in the response.

5. The expression uses only:
   numbers,
   declared variable names,
   +,
   -,
   *,
   /,
   ^,
   abs,
   min,
   max,
   sum.

6. Every enum value comes from ALLOWED DOMAIN VALUES.

7. The model represents the user's actual problem rather than
   blindly converting CSV columns into variables.
"""

    # ========================================================
    # Repair prompt
    # ========================================================

    def _build_repair_prompt(
        self,
        original_result: dict[str, Any],
        validation_error: str,
        dataset: dict[str, Any],
    ) -> str:

        allowed = dataset.get(
            "allowed_values",
            {},
        )

        return f"""
You are repairing an optimization problem model generated by another
AI system.

The previous model failed backend validation.

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
ALLOWED VALUES
============================================================

{json.dumps(
    allowed,
    ensure_ascii=False,
    default=str,
    indent=2,
)}

Fix ONLY what is necessary.

IMPORTANT:

A dataset column is NOT automatically a decision variable.

Every identifier appearing in "expression" MUST exactly match a
variable name in "variables".

If the previous expression incorrectly referenced a dataset column,
do NOT simply add that column as a decision variable.

Instead, reinterpret the problem using legitimate decision variables.

Return ONLY valid JSON with exactly these fields:

{{
    "name": "...",
    "description": "...",
    "problem_family": "...",
    "mathematical_properties": [],
    "variables": [],
    "objective_kind": "...",
    "objective_sense": "...",
    "expression": "...",
    "representation": "...",
    "explanation": "...",
    "assumptions": []
}}

Do not output executable code.
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

        self._validate_allowed_values(
            result=result,
            dataset=dataset,
        )

    # --------------------------------------------------------
    # Top-level structure
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Basic fields
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    def _validate_variables(
        self,
        variables: Any,
    ) -> None:

        if not isinstance(variables, list):
            raise ValueError(
                "'variables' must be a list."
            )

        if not variables:
            raise ValueError(
                "At least one decision variable is required."
            )

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

    # --------------------------------------------------------
    # Expression
    # --------------------------------------------------------

    def _validate_expression(
        self,
        expression: Any,
        variables: list[dict[str, Any]],
    ) -> None:

        if not isinstance(expression, str):
            raise ValueError(
                "'expression' must be a string."
            )

        if not expression.strip():
            raise ValueError(
                "Optimization expression cannot be empty."
            )

        variable_names = {
            variable["name"]
            for variable in variables
        }

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

        # Prevent common programming constructs.
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

    # --------------------------------------------------------
    # Allowed domain values
    # --------------------------------------------------------

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

        # Validate fields only when the backend provides
        # corresponding allowed-value collections.
        for field in (
            "problem_family",
            "representation",
        ):
            allowed_values = allowed.get(field)

            if (
                isinstance(allowed_values, list)
                and allowed_values
                and result[field] not in allowed_values
            ):
                raise ValueError(
                    f"Invalid {field} "
                    f"'{result[field]}'. "
                    f"Allowed values: {allowed_values}"
                )

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