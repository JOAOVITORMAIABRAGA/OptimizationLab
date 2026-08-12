# OptimizationLab --- Backend

> FastAPI backend for dataset processing, AI-assisted problem modeling,
> validation, and future optimization execution.

## Overview

The backend orchestrates the core application workflow.

``` text
HTTP Request
     |
     v
FastAPI
     |
     +--> Dataset analysis
     |
     +--> LLM service
     |       |
     |       v
     |     Groq API
     |
     +--> Model validation
     |
     v
Structured Problem Model
     |
     v
Future Optimization Engine
```

The LLM is a modeling component, not an execution environment.

For tabular datasets, the modeling pipeline can recognize a decision
indexed by a dataset entity (for example, production quantity per product),
map numeric columns to objective coefficients/bounds, and expand the indexed
concept into the scalar declarative variables required by the current
`OptimizationProblem` contract. The numerical values always come from the
dataset; the system never fabricates missing parameters.

## Responsibilities

The backend currently handles:

-   REST API endpoints;
-   multipart CSV uploads;
-   dataset inspection;
-   dataset summaries;
-   environment configuration;
-   Groq integration;
-   structured LLM responses;
-   model validation;
-   CORS configuration.

The future backend will also orchestrate solver execution and result
serialization.

## Structure

``` text
Backend/
├── api.py
├── llm_service.py
├── requirements.txt
├── .env                  # local only
└── ...
```

### `api.py`

Application and API route layer.

Responsible for HTTP requests, request parsing, file uploads, validation
errors, and response serialization.

### `llm_service.py`

Groq/LLM adapter.

Responsible for translating:

``` text
Problem description + dataset summary
```

into:

``` text
Structured problem model
```

It must never execute arbitrary code returned by an LLM.

## LLM trust boundary

The intended architecture is:

``` text
LLM output
    |
    v
JSON parsing
    |
    v
Schema validation
    |
    v
Expression validation
    |
    v
Optimization engine
```

Never replace this with arbitrary `eval()` or `exec()` of LLM output.

## Problem model

A simplified response has this shape:

``` json
{
  "name": "Delivery Optimization",
  "problem_family": "routing",
  "mathematical_properties": ["discrete", "constrained"],
  "variables": [],
  "objective_kind": "single",
  "objective_sense": "minimize",
  "expression": "...",
  "representation": "vector",
  "explanation": "...",
  "assumptions": []
}
```

The model is deliberately declarative so the application can validate it
before it reaches a solver.

## Environment

Create:

``` text
Backend/.env
```

Example:

``` env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

The backend reads these values through environment variables.

Never commit the real `.env`.

Optional `.env.example`:

``` env
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
```

## Installation

``` powershell
cd Backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Start development server:

``` powershell
uvicorn api:app --reload
```

API:

``` text
http://127.0.0.1:8000
```

Swagger:

``` text
http://127.0.0.1:8000/docs
```

OpenAPI:

``` text
http://127.0.0.1:8000/openapi.json
```

## Dependencies

  Package            Purpose
  ------------------ ------------------------------
  FastAPI            REST API
  Uvicorn            ASGI server
  Pydantic           Validation
  Groq               LLM API client
  PyGAD              Genetic algorithm foundation
  NumPy              Numerical operations
  python-dotenv      Environment loading
  python-multipart   File uploads

## Modeling endpoint

The main modeling flow conceptually accepts:

``` text
POST /api/model
Content-Type: multipart/form-data

problem_description = "..."
file = dataset.csv
```

The backend then:

1.  receives the request;
2.  validates the file;
3.  parses the dataset;
4.  creates a dataset summary;
5.  sends relevant context to the LLM;
6.  receives structured JSON;
7.  validates the generated model;
8.  returns the model and dataset metadata.

Typical response shape:

``` json
{
  "problem": {
    "name": "Delivery Optimization",
    "problem_family": "routing",
    "objective_kind": "single",
    "objective_sense": "minimize",
    "variables": [],
    "expression": "..."
  },
  "explanation": "...",
  "assumptions": [],
  "dataset": {
    "filename": "dataset.csv",
    "row_count": 4999,
    "column_count": 53,
    "columns": []
  }
}
```

The current implementation remains the source of truth for the exact API
schema.

## Validation

The backend should reject invalid models before optimization.

Examples of invalid output include:

-   malformed JSON;
-   unsupported enum values;
-   expressions referencing unknown variables;
-   unsupported operators;
-   inconsistent variable definitions;
-   missing required fields.

Example error:

``` json
{
  "detail": "Expression references unknown variable 'x'."
}
```

This validation layer is essential because LLM output is probabilistic.

## CORS

Local development commonly uses:

``` text
Frontend: http://localhost:5173
Backend:  http://127.0.0.1:8000
```

The API therefore requires suitable CORS configuration.

In production, prefer allowing the deployed frontend origin instead of
an unrestricted wildcard.

## Testing

When tests are available:

``` powershell
pytest -q
```

Recommended test organization:

``` text
tests/
├── test_api.py
├── test_dataset.py
├── test_llm_service.py
├── test_validation.py
└── test_solver.py
```

High-value tests should verify that malformed or unsafe LLM output never
reaches solver execution.

## Deployment --- Render

Recommended configuration:

**Root directory**

``` text
Backend
```

**Build command**

``` bash
pip install -r requirements.txt
```

**Start command**

``` bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Environment variables**

``` env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

## Frontend integration

Local:

``` env
VITE_API_URL=http://127.0.0.1:8000
```

Production:

``` env
VITE_API_URL=https://your-backend.onrender.com
```

Correct:

``` text
React -> FastAPI -> Groq
```

Incorrect:

``` text
React -> Groq
```

Keeping Groq behind FastAPI prevents the API key from being exposed to
the browser.

## Error handling

The API should return useful errors for:

-   missing files;
-   invalid CSV;
-   missing fields;
-   missing Groq credentials;
-   LLM failure;
-   malformed LLM response;
-   failed model validation;
-   invalid objective expressions.

Errors should fail safely rather than bypass validation.

## Roadmap

### Current

-   [x] FastAPI application
-   [x] CSV upload
-   [x] Dataset summary
-   [x] Groq integration
-   [x] LLM problem modeling
-   [x] Structured response
-   [x] Initial validation
-   [x] Frontend integration

### Next

-   [ ] Formal optimization schema
-   [ ] Robust expression parser
-   [ ] Constraint modeling
-   [ ] Model normalization
-   [ ] Solver abstraction
-   [ ] PyGAD integration
-   [ ] Optimization execution
-   [ ] Result validation
-   [ ] Result API

### Future

-   [ ] Multiple solver backends
-   [ ] Multi-objective optimization
-   [ ] Persistent experiments
-   [ ] Async optimization jobs
-   [ ] Authentication
-   [ ] Rate limiting
-   [ ] Observability
-   [ ] Dataset privacy controls

## Engineering principles

### LLMs model; deterministic code validates and executes

AI output must pass application-level validation.

### No arbitrary code execution

Never execute LLM-generated Python.

### Provider isolation

Groq-specific implementation belongs inside the LLM service layer.

### Explicit domain models

Prefer typed schemas over loosely structured dictionaries.

### Fail safely

Reject invalid models before they reach a solver.

### Replaceable solvers

The domain model should not be tightly coupled to one optimization
library.

## Related documentation

-   `../README.md`
-   `../Frontend/README.md`

## V12 semantic modeling

V12 introduces explicit objective completeness (`complete`, `incomplete`,
`not_applicable`) and a semantic constraint target (`bound`) while preserving
legacy constraint bound fields for compatibility. Incomplete objectives are
modeling states, not solver incompatibilities.
