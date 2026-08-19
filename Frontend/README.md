# OptimizationLab --- Frontend

> React + TypeScript interface for submitting optimization problems,
> uploading datasets, and reviewing AI-generated problem models.

## Overview

The frontend is the user-facing layer of OptimizationLab.

It is intentionally designed around a simple default workflow:

``` text
Describe the problem
        +
Upload the dataset
        |
        v
      Backend
        |
        v
    AI modeling
        |
        v
Structured problem model
```

The browser handles interaction and presentation. Secrets, LLM provider
integration, validation, and optimization execution belong to the
backend.

## Technology stack

  Technology   Purpose
  ------------ ---------------------------
  React        UI
  TypeScript   Type safety
  Vite         Build/development tooling
  CSS          Styling
  Fetch API    Backend communication

## Project structure

``` text
Frontend/
├── src/
│   ├── App.tsx
│   ├── api.ts
│   ├── main.tsx
│   ├── styles.css
│   └── vite-env.d.ts
├── index.html
├── package.json
├── package-lock.json
├── vite.config.ts
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
└── ...
```

### Main files

**`src/App.tsx`**

Main application interface and workflow.

**`src/api.ts`**

Centralized backend API client.

**`src/main.tsx`**

React entry point.

**`src/styles.css`**

Global interface styling.

**`src/vite-env.d.ts`**

Vite TypeScript declarations:

``` typescript
/// <reference types="vite/client" />
```

This allows TypeScript to understand `import.meta.env` and Vite
asset/style imports.

## User workflow

### 1. Describe the problem

Example:

``` text
I have historical sales and delivery data and want
to improve delivery efficiency while reducing cost
and late-delivery risk.
```

### 2. Upload a dataset

The user can provide one or more CSV, TXT or XLSX sources relevant to the problem (up to 10 files).

### 3. Generate model

The frontend sends the description and file to:

``` text
POST /api/model
```

### 4. Review the interpretation

The backend returns a structured model containing information such as:

-   problem family;
-   decision variables;
-   objective;
-   mathematical representation;
-   assumptions.

### 5. Future optimization

Later versions will allow the validated model to be sent into an
optimization solver.

## Architecture

``` text
+-------------------------+
| React / TypeScript      |
|                         |
| Problem description     |
| Multi-file dataset upload |
| Model result            |
+------------+------------+
             |
             | HTTP
             v
+-------------------------+
| FastAPI Backend         |
|                         |
| Dataset processing      |
| LLM modeling            |
| Validation              |
+------------+------------+
             |
             v
          Groq API
```

The browser never receives the Groq API key.

## Environment variables

Create:

``` text
Frontend/.env
```

Local:

``` env
VITE_API_URL=http://127.0.0.1:8000
```

Production:

``` env
VITE_API_URL=https://your-backend.onrender.com
```

Vite exposes variables through:

``` typescript
import.meta.env
```

Only public configuration should use the `VITE_` prefix.

Never do this:

``` env
VITE_GROQ_API_KEY=...
```

The Groq key belongs exclusively in the backend.

## Installation

Requirements:

-   Node.js 20+
-   npm

Install dependencies:

``` powershell
cd Frontend
npm install
```

Start development:

``` powershell
npm run dev
```

Vite normally exposes:

``` text
http://localhost:5173
```

## Production build

Build:

``` powershell
npm run build
```

Output:

``` text
dist/
```

Preview:

``` powershell
npm run preview
```

## TypeScript

The project uses strict TypeScript configuration.

Important options include:

``` json
{
  "strict": true,
  "moduleResolution": "Bundler",
  "jsx": "react-jsx",
  "noEmit": true
}
```

React type dependencies include:

``` text
typescript
@types/react
@types/react-dom
```

## API client

Backend communication should remain centralized in `api.ts`.

Conceptually:

``` typescript
const API_URL = import.meta.env.VITE_API_URL;
```

The modeling request sends:

``` text
problem description
+
one or more CSV/TXT/XLSX files
```

to:

``` text
${API_URL}/api/model
```

Centralizing this logic prevents UI components from becoming tightly
coupled to HTTP details.

## Data flow

``` text
User input
    |
    +---- problem description
    |
    +---- CSV
    |
    v
React state
    |
    v
api.ts
    |
    | multipart/form-data
    v
FastAPI
    |
    v
LLM modeling
    |
    v
JSON response
    |
    v
React
    |
    v
Problem model UI
```

## Error handling

The UI should account for:

-   missing problem description;
-   missing dataset;
-   unsupported files;
-   failed API requests;
-   backend validation errors;
-   LLM failures;
-   network timeouts.

Example backend validation error:

``` json
{
  "detail": "Expression references unknown variable 'x'."
}
```

The frontend should expose useful information without leaking
implementation details or secrets.

## Local integration

Start the backend:

``` powershell
cd Backend
.venv\Scripts\activate
uvicorn api:app --reload
```

Start the frontend:

``` powershell
cd Frontend
npm run dev
```

Expected architecture:

``` text
Browser
http://localhost:5173
       |
       | HTTP
       v
FastAPI
http://127.0.0.1:8000
       |
       v
Groq API
```

If the frontend cannot reach the backend, check:

1.  `VITE_API_URL`;
2.  backend port;
3.  API route;
4.  CORS configuration;
5.  browser Network tab.

## Deployment --- Vercel

Recommended settings:

**Root directory**

``` text
Frontend
```

**Install command**

``` bash
npm install
```

**Build command**

``` bash
npm run build
```

**Output directory**

``` text
dist
```

**Environment variable**

``` env
VITE_API_URL=https://your-backend.onrender.com
```

After changing a Vercel environment variable, redeploy so the value is
included in the client build.

## Production integration

``` text
User
 |
 v
Vercel
React application
 |
 | HTTPS
 v
Render
FastAPI backend
 |
 v
Groq API
```

Frontend:

``` env
VITE_API_URL=https://your-backend.onrender.com
```

Backend:

``` env
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
```

The frontend must never receive `GROQ_API_KEY`.

## UI principles

### Simplicity

The default workflow should not require mathematical expertise.

### Transparency

Users should be able to see what the AI understood.

### Expert control

Advanced users should eventually be able to review and adjust the
generated model.

### Separation of concerns

The frontend collects and presents information. The backend owns
secrets, AI integration, validation, and optimization execution.

### Progressive complexity

Advanced controls should appear when they become useful rather than
overwhelming first-time users.

## Roadmap

### Current

-   [x] React + TypeScript
-   [x] Vite
-   [x] Problem description input
-   [x] Multiple dataset upload
-   [x] CSV, TXT and XLSX support
-   [x] Backend integration
-   [x] AI-generated problem model display
-   [x] Environment-based API URL
-   [x] Production build
-   [x] Vercel deployment support

### Next

-   [ ] Richer problem-model visualization
-   [ ] Expert-mode model editor
-   [ ] Constraint review
-   [ ] Optimization execution screen
-   [ ] Solver progress
-   [ ] Results dashboard
-   [ ] Objective/fitness charts

### Future

-   [ ] Saved projects
-   [ ] Optimization experiment history
-   [ ] Interactive model editor
-   [ ] Dataset profiling
-   [ ] Result comparison
-   [ ] Accessibility audit
-   [ ] Internationalization

## Related documentation

-   `../README.md`
-   `../Backend/README.md`
