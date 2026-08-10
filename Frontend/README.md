# Optimization Lab — GUI MVP

React + TypeScript + Vite frontend for the existing Optimization Lab backend.

## Run

Backend:

```bash
cd Backend
pip install -r requirements
uvicorn main:app --reload
```

Frontend, in another terminal:

```bash
cd Frontend
npm install
npm run dev
```

Optional: set `VITE_API_URL` when the API is not at `http://127.0.0.1:8000`.

## Scope

The MVP covers problem modeling, structured representation, validation, compatibility and recommendation. It deliberately does not implement Step 6 (Orchestrator integration) or Step 7 (Benchmark).
