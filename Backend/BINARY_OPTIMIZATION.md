# Binary optimization

Binary optimization is implemented as a specialization of the native `VECTOR` representation rather than as a second incompatible representation.

A problem with `VariableType.BINARY` is materialized by `VectorRepresentationAdapter` with bounds `[0, 1]` and values normalized to exactly `0` or `1`.

## Supported execution paths

- Integer Programming: exact MILP through SciPy HiGHS.
- Constraint Programming: exact bounded integer/binary backend.
- Genetic Algorithm: native binary-compatible vector execution through the adapter.
- Simulated Annealing, Tabu Search and Hill Climbing: binary values are normalized by the vector adapter; these are available as heuristic alternatives when appropriate.

## Automatic decision

For a single-objective, linear, constrained binary model, the decision engine strongly prefers `integer_programming` and does not execute competing algorithms during selection.

The compatibility layer checks the actual variable types, objective, constraints, mathematical properties and representation. It does not require a redundant `INTEGER` property when the variables themselves are binary.

## Knapsack benchmark

The manual project-selection test uses six binary variables and a budget of 100. The exact optimum is:

- B = 1
- C = 1
- E = 1
- F = 1
- A = D = 0
- total cost = 100
- total return = 209
