# Universal Optimization Architecture

The OptimizationLab separates the semantic optimization problem from the internal representation used by an algorithm.

## Core flow

```text
OptimizationProblem
        |
        +-- Variables / Objective / Constraints
        |
        v
RepresentationAdapter
        |
        +--> VectorRepresentationAdapter --> GA / PSO / DE
        |
        +--> PermutationRepresentationAdapter --> ACO
        |
        v
CandidateSolution
        |
        v
ObjectiveEvaluator
        |
        v
Fitness
```

### Semantic model

`OptimizationProblem` describes what the user wants to optimize. It does not prescribe how GA, PSO, DE, ACO or a mathematical solver must store a candidate.

### Representation adapters

`RepresentationAdapter` translates between the semantic model and an algorithm-specific search space.

#### Vector

`VectorRepresentationAdapter` supports continuous, integer, discrete and binary variables for numeric metaheuristics:

- continuous: clipped to bounds
- integer/discrete: rounded and clipped
- binary: thresholded at 0.5

#### Permutation

`PermutationRepresentationAdapter` maps a semantic ordered assignment to permutation indices. A permutation problem has one discrete/integer variable per position and a shared finite element set. The adapter guarantees that every element is used exactly once.

ACO consumes the permutation indices internally and the adapter translates the result back into named semantic variables.

### Native metaheuristics

GA, PSO, Differential Evolution and Ant Colony Optimization are implemented directly in OptimizationLab. They do not delegate their search loops to third-party metaheuristic libraries. NumPy is used for numerical operations and random-number generation.

The algorithms intentionally keep different internal representations:

- GA: population of numeric chromosomes
- PSO: particle positions and velocities
- DE: numeric population and differential mutation
- ACO: ant permutations plus a position/element pheromone matrix

The common contract is the semantic problem and the resulting `CandidateSolution`, not an artificial universal internal vector.

### CandidateSolution

`CandidateSolution` is the common semantic output. Algorithms may use chromosomes, particles, differential vectors, ant paths or solver variables internally, but the application receives named decision values.

### ObjectiveEvaluator

`ObjectiveEvaluator` evaluates the explicit structured objective and hard constraints against a semantic candidate. This lets different algorithms evaluate the same problem without duplicating the mathematical interpretation.

## Current integration

The universal entry point is:

```python
algorithm.optimize_problem(problem)
```

The native metaheuristics currently integrated through the common contract are:

- Genetic Algorithm
- Particle Swarm Optimization
- Differential Evolution
- Ant Colony Optimization for permutation representations

LP, ILP and CP keep their specialized exact solver backends and return the same `CandidateSolution` abstraction through the execution engine.

## ACO representation contract

ACO is deliberately no longer advertised as a generic numeric-vector optimizer. Its executable representation is `PERMUTATION`.

For a four-position problem with elements `[0, 1, 2, 3]`, for example:

```text
semantic variables:
    p0, p1, p2, p3

ACO representation:
    [3, 2, 1, 0]

semantic candidate:
    p0 = 3
    p1 = 2
    p2 = 1
    p3 = 0
```

Pheromone is stored as a matrix indexed by `(position, element)`. Each ant constructs a valid permutation without repeating an element. Evaporation and ranked elite deposition update the pheromone matrix after each iteration.

A heuristic matrix can optionally be supplied through permutation representation metadata as `heuristic_matrix`. If none is supplied, the heuristic factor is uniform and the search relies on pheromone learning.

Graph-native construction is intentionally not claimed yet. A routing problem can use the permutation adapter when its semantic model exposes ordered positions, but a future graph adapter is still needed for native graph structures.

### OptimizationResult

`OptimizationResult` is the standard execution record for algorithms that expose telemetry. It contains the semantic solution, objective value, feasibility, iterations, objective evaluations, runtime, convergence history and algorithm parameters.

## Validation

The current test suite validates the full ACO path, including:

- permutation adapter encode/decode round-trip
- permutation validity
- ACO pheromone construction
- ACO optimization through `OptimizationProblem`
- registry compatibility and recommendation
- execution through `OptimizationExecutionEngine`

The current suite passes with **91 tests**.

## Universal decision layer

The execution path now separates **selection** from **optimization**:

```text
OptimizationProblem
      |
      v
ValidationEngine
      |
      v
CompatibilityEngine
      |
      v
RecommendationEngine
      |
      v
UniversalDecisionEngine
      |
      +--> one selected algorithm
      |
      v
OptimizationExecutionEngine
```

Selection is deterministic and structural. It does not execute several algorithms just to decide which one to use, keeping the default path inexpensive.

The tie-break policy favors specialized exact solvers for clearly linear/integer structures and fast domain-specific metaheuristics for nonlinear vectors and permutation/routing problems. Empirical comparison can be added later as an optional mode rather than as the default decision path.

## Native local-search algorithms

The native metaheuristic set is now:

- GA
- PSO
- Differential Evolution
- BFO
- ACO
- Simulated Annealing
- Tabu Search
- Hill Climbing

SA, Tabu Search and Hill Climbing share the representation adapter's neighborhood contract. Numeric vectors use bounded perturbations; permutations use swap moves. BFO remains intentionally numeric/vector-only.
