# LLM constraint modeling hardening

The modeling contract now carries hard constraints from natural language into the executable mathematical model.

Pipeline:

Natural language + CSV
-> LLM draft
-> deterministic tabular completion
-> explicit constraints
-> AI draft validation
-> ProblemInput/build_problem
-> Validation/Compatibility/Recommendation
-> solver

For explicit budget limits, the deterministic completion layer extracts the numeric threshold from the user's text and the coefficient column from the dataset. It creates an auditable linear constraint instead of relying on the solver to infer missing semantics.

Example:

`I cannot spend more than 100.`

with `project,cost,return` becomes:

`40*select_A + 35*select_B + 30*select_C + 25*select_D + 20*select_E + 15*select_F <= 100`

The model does not silently rescale units. If the prompt says `100 mil`, the threshold is `100000` unless the dataset explicitly provides a compatible unit. This avoids silently inventing unit conversions.
