from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from domain.problem import OptimizationProblem
from domain.solutions import CandidateSolution
from domain.structures import ProblemStructureKind
from domain.variables import VariableType
from domain.representations import SolutionRepresentationKind


class ProblemAdapter(ABC):
    """Translate a domain OptimizationProblem into one solver's native input contract.

    Adapters belong at the solver boundary. They never choose algorithms and they
    never execute optimization; they only validate/translate the problem into the
    data shape expected by the concrete solver family.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        self.problem = problem


class SearchSpaceAdapter(ProblemAdapter, ABC):
    """Native search-space contract used by population/local-search algorithms."""

    @abstractmethod
    def bounds(self) -> List[Tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def encode(self, solution: CandidateSolution) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, representation: List[Any]) -> CandidateSolution:
        raise NotImplementedError

    @abstractmethod
    def random_solution(self, rng: np.random.Generator) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def neighbors(self, representation, rng, count: int, step_size: float = 0.2) -> List[List[Any]]:
        raise NotImplementedError

    def heuristic_matrix(self) -> np.ndarray:
        """Optional construction heuristic for algorithms such as ACO."""
        size = len(self.bounds())
        return np.ones((size, size), dtype=float)


class NumericSearchAdapter(SearchSpaceAdapter):
    """Adapt VECTOR/BINARY/INTEGER/DISCRETE models to numeric metaheuristics."""

    representation_kind = SolutionRepresentationKind.VECTOR

    @property
    def kind(self):
        return self.representation_kind

    def __init__(self, problem: OptimizationProblem) -> None:
        super().__init__(problem)
        representation = problem.solution_representation
        if representation is None or representation.kind != SolutionRepresentationKind.VECTOR:
            raise ValueError("Numeric search solvers require VECTOR solution representation.")

    def bounds(self) -> List[Tuple[float, float]]:
        result: List[Tuple[float, float]] = []
        for variable in self.problem.variables:
            if variable.variable_type == VariableType.BINARY:
                result.append((0.0, 1.0))
                continue
            low = variable.lower_bound
            high = variable.upper_bound
            if low is None and variable.domain is not None:
                low = variable.domain.lower
            if high is None and variable.domain is not None:
                high = variable.domain.upper
            if low is None or high is None:
                raise ValueError(
                    f"Variable '{variable.name}' requires finite bounds for numeric metaheuristics."
                )
            result.append((float(low), float(high)))
        return result

    def normalize(self, vector: List[float]) -> List[Any]:
        normalized: List[Any] = []
        for value, variable, (low, high) in zip(vector, self.problem.variables, self.bounds()):
            value = min(max(float(value), low), high)
            if variable.variable_type == VariableType.BINARY:
                value = 1 if value >= 0.5 else 0
            elif variable.variable_type in {VariableType.INTEGER, VariableType.DISCRETE}:
                value = int(round(value))
            normalized.append(value)
        return normalized

    # Compatibility name for callers written against the pre-refactor adapter.
    def normalize_vector(self, vector: List[float]) -> List[Any]:
        return self.normalize(vector)

    def encode(self, solution: CandidateSolution) -> List[Any]:
        return self.normalize([float(solution.values[v.name]) for v in self.problem.variables])

    def decode(self, representation: List[Any]) -> CandidateSolution:
        vector = self.normalize(representation)
        values = {variable.name: value for variable, value in zip(self.problem.variables, vector)}
        return CandidateSolution(values=values, representation=SolutionRepresentationKind.VECTOR)

    def random_solution(self, rng: np.random.Generator) -> List[Any]:
        low = np.asarray([item[0] for item in self.bounds()], dtype=float)
        high = np.asarray([item[1] for item in self.bounds()], dtype=float)
        return self.normalize(rng.uniform(low, high).tolist())

    def neighbors(self, representation, rng, count, step_size=0.2):
        low = np.asarray([item[0] for item in self.bounds()], dtype=float)
        high = np.asarray([item[1] for item in self.bounds()], dtype=float)
        current = np.asarray(self.normalize(representation), dtype=float)
        span = high - low
        neighbors: List[List[Any]] = []
        for _ in range(max(0, count)):
            candidate = current + rng.normal(0.0, step_size, size=len(current)) * np.where(span > 0, span, 1.0)
            neighbors.append(self.normalize(candidate.tolist()))
        return neighbors


class PermutationSearchAdapter(SearchSpaceAdapter):
    """Adapt a finite ordered assignment into permutation indices."""

    representation_kind = SolutionRepresentationKind.PERMUTATION

    @property
    def kind(self):
        return self.representation_kind

    def __init__(self, problem: OptimizationProblem) -> None:
        super().__init__(problem)
        self.elements = self._resolve_elements()
        self._element_to_index = {self._key(value): index for index, value in enumerate(self.elements)}
        if len(self.elements) != len(problem.variables):
            raise ValueError(
                "Permutation representation requires exactly one element per variable "
                f"(got {len(self.elements)} elements for {len(problem.variables)} variables)."
            )

    @staticmethod
    def _key(value: Any) -> Any:
        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    def _resolve_elements(self) -> List[Any]:
        metadata = self.problem.solution_representation.metadata if self.problem.solution_representation else None
        if metadata and metadata.get("elements") is not None:
            elements = list(metadata["elements"])
            if len(set(map(self._key, elements))) != len(elements):
                raise ValueError("Permutation elements must be unique.")
            return elements

        candidates: List[List[Any]] = []
        for variable in self.problem.variables:
            domain = variable.domain
            if domain is None:
                continue
            values = domain.values if domain.values is not None else domain.elements
            if values:
                candidates.append(list(values))
        if not candidates:
            raise ValueError("Permutation representation requires a finite element list.")

        first = candidates[0]
        first_keys = [self._key(value) for value in first]
        for other in candidates[1:]:
            if {self._key(value) for value in other} != set(first_keys):
                raise ValueError("All permutation position domains must contain the same elements.")
        if len(set(first_keys)) != len(first_keys):
            raise ValueError("Permutation elements must be unique.")
        return first

    @property
    def size(self) -> int:
        return len(self.elements)

    def bounds(self) -> List[Tuple[float, float]]:
        return [(0.0, float(self.size - 1)) for _ in self.problem.variables]

    def encode(self, solution: CandidateSolution) -> List[int]:
        try:
            return [self._element_to_index[self._key(solution.values[variable.name])] for variable in self.problem.variables]
        except KeyError as exc:
            raise ValueError("Solution contains an element outside the permutation domain.") from exc

    def decode(self, representation: List[int]) -> CandidateSolution:
        indices = [int(index) for index in representation]
        if len(indices) != self.size or sorted(indices) != list(range(self.size)):
            raise ValueError("A permutation representation must contain every element index exactly once.")
        values = {variable.name: self.elements[index] for variable, index in zip(self.problem.variables, indices)}
        return CandidateSolution(values=values, representation=SolutionRepresentationKind.PERMUTATION)

    def random_solution(self, rng: np.random.Generator) -> List[int]:
        return rng.permutation(self.size).astype(int).tolist()

    def neighbors(self, representation, rng, count, step_size=0.2):
        current = list(map(int, representation))
        if len(current) < 2:
            return [current.copy() for _ in range(max(0, count))]
        neighbors: List[List[int]] = []
        for _ in range(max(0, count)):
            candidate = current.copy()
            i, j = rng.choice(len(candidate), size=2, replace=False)
            candidate[int(i)], candidate[int(j)] = candidate[int(j)], candidate[int(i)]
            neighbors.append(candidate)
        return neighbors


class NeighborhoodSearchAdapter(SearchSpaceAdapter):
    """Adapt the two native search spaces supported by local-search solvers.

    Local search is intentionally the only solver family that needs this small
    composition: SA/Tabu/Hill Climbing share neighborhood semantics while the
    underlying vector/permutation spaces remain independent.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        super().__init__(problem)
        representation = problem.solution_representation
        if representation is None:
            raise ValueError("Local search requires a solution representation.")
        if representation.kind == SolutionRepresentationKind.VECTOR:
            self.delegate: SearchSpaceAdapter = NumericSearchAdapter(problem)
        elif representation.kind == SolutionRepresentationKind.PERMUTATION:
            self.delegate = PermutationSearchAdapter(problem)
        else:
            raise ValueError(
                "Local search currently supports VECTOR and PERMUTATION solution representations."
            )

    @property
    def kind(self):
        return self.delegate.kind

    def bounds(self):
        return self.delegate.bounds()

    def encode(self, solution):
        return self.delegate.encode(solution)

    def decode(self, representation):
        return self.delegate.decode(representation)

    def random_solution(self, rng):
        return self.delegate.random_solution(rng)

    def neighbors(self, representation, rng, count, step_size=0.2):
        return self.delegate.neighbors(representation, rng, count, step_size)


class GraphProblemAdapter(ProblemAdapter):
    """Adapt a graph-structured problem to graph-native solvers.

    Unlike search-space adapters, this adapter does not expose generic bounds,
    random solutions or neighborhoods. Graph algorithms own their solution
    construction (path, edge set, Euler walk, matching, ...).
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        super().__init__(problem)
        structure = problem.problem_structure
        legacy_representation = problem.solution_representation
        if structure is None:
            # Migration boundary for pre-V9 models. New models must declare the
            # graph as problem structure; this fallback exists only so persisted
            # legacy models remain executable during the migration window.
            if legacy_representation is None or legacy_representation.kind not in {
                SolutionRepresentationKind.GRAPH,
                SolutionRepresentationKind.EDGE_WALK,
                SolutionRepresentationKind.EDGE_SET,
            }:
                raise ValueError("Graph solvers require a GRAPH problem structure.")
            metadata = dict(legacy_representation.metadata or {})
        else:
            if structure.kind != ProblemStructureKind.GRAPH:
                raise ValueError("Graph solvers require a GRAPH problem structure.")
            metadata = dict(structure.metadata or {})
            metadata.update((legacy_representation.metadata if legacy_representation else {}) or {})
        self.nodes = list(metadata.get("nodes") or [])
        self.edges = list(metadata.get("edges") or [])
        self.directed = bool(metadata.get("directed", False))
        self.graph_problem_type = str(metadata.get("graph_problem_type", "generic"))
        self.source = metadata.get("source")
        self.target = metadata.get("target")
        if not self.nodes:
            self.nodes = self._infer_nodes()
        if not self.edges:
            raise ValueError("Graph structure requires a non-empty 'edges' list.")
        self._validate_edges()

    def _infer_nodes(self) -> List[Any]:
        nodes: List[Any] = []
        seen = set()
        for edge in self.edges:
            for node in (edge.get("u"), edge.get("v")):
                if node not in seen:
                    seen.add(node)
                    nodes.append(node)
        return nodes

    def _validate_edges(self) -> None:
        ids = set()
        node_set = set(self.nodes)
        for edge in self.edges:
            if not isinstance(edge, dict):
                raise ValueError("Each graph edge must be an object.")
            for field in ("id", "u", "v", "weight"):
                if field not in edge:
                    raise ValueError(f"Graph edge is missing '{field}'.")
            if edge["id"] in ids:
                raise ValueError(f"Duplicate graph edge id '{edge['id']}'.")
            ids.add(edge["id"])
            if edge["u"] not in node_set or edge["v"] not in node_set:
                raise ValueError(f"Graph edge '{edge['id']}' references an unknown node.")
            weight = float(edge["weight"])
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"Graph edge '{edge['id']}' must have a finite non-negative weight.")

    def edge_map(self) -> Dict[Any, Dict[str, Any]]:
        return {edge["id"]: edge for edge in self.edges}

    def adjacency(self, required_only: bool = False) -> Dict[Any, List[Tuple[Any, float, Any]]]:
        adjacency: Dict[Any, List[Tuple[Any, float, Any]]] = {node: [] for node in self.nodes}
        for edge in self.edges:
            if required_only and not bool(edge.get("required", True)):
                continue
            u, v, weight, edge_id = edge["u"], edge["v"], float(edge["weight"]), edge["id"]
            adjacency[u].append((v, weight, edge_id))
            if not self.directed:
                adjacency[v].append((u, weight, edge_id))
        return adjacency

    def candidate(self, edge_ids: List[Any], *, metadata: Dict[str, Any] | None = None, representation: SolutionRepresentationKind | None = None) -> CandidateSolution:
        representation = representation or self._default_solution_representation()
        variable_name = self._solution_variable_name()
        return CandidateSolution(
            values={variable_name: list(edge_ids)},
            representation=representation,
            metadata=dict(metadata or {}),
        )

    def _solution_variable_name(self) -> str:
        if len(self.problem.variables) == 1:
            return self.problem.variables[0].name
        return "route"

    def _default_solution_representation(self) -> SolutionRepresentationKind:
        if self.problem.solution_representation is not None:
            return self.problem.solution_representation.kind
        return SolutionRepresentationKind.EDGE_WALK


# Built-in representation adapters available to the decision/execution pipeline.
# Names are stable domain identifiers, not Python class names.
class GraphRoutingAdapter(GraphProblemAdapter, SearchSpaceAdapter):
    """Translate a graph-routing instance into a permutation search space.

    The graph remains the source of truth. This adapter only exposes the node
    ordering required by permutation-based metaheuristics such as ACO. It is
    intentionally limited to Hamiltonian-cycle/TSP semantics for now.
    """

    representation_kind = SolutionRepresentationKind.PERMUTATION

    def __init__(self, problem: OptimizationProblem) -> None:
        GraphProblemAdapter.__init__(self, problem)
        if self.graph_problem_type not in {"tsp", "traveling_salesman", "traveling_salesman_problem"}:
            raise ValueError(
                "GraphRoutingAdapter currently supports only graph_problem_type 'tsp'."
            )
        if len(self.nodes) < 3:
            raise ValueError("TSP requires at least three nodes.")
        self._node_index = {node: i for i, node in enumerate(self.nodes)}
        self._weights = self._build_weight_matrix()

    @property
    def kind(self):
        return self.representation_kind

    @property
    def elements(self) -> List[Any]:
        return list(self.nodes)

    @property
    def size(self) -> int:
        return len(self.nodes)

    def bounds(self) -> List[Tuple[float, float]]:
        return [(0.0, float(self.size - 1)) for _ in range(self.size)]

    def _build_weight_matrix(self) -> np.ndarray:
        matrix = np.full((self.size, self.size), np.inf, dtype=float)
        np.fill_diagonal(matrix, 0.0)
        for edge in self.edges:
            u, v = self._node_index[edge["u"]], self._node_index[edge["v"]]
            weight = float(edge["weight"])
            matrix[u, v] = min(matrix[u, v], weight)
            if not self.directed:
                matrix[v, u] = min(matrix[v, u], weight)
        if np.any(~np.isfinite(matrix[~np.eye(self.size, dtype=bool)])):
            raise ValueError("TSP graph must contain a finite edge between every pair of distinct nodes.")
        return matrix

    def encode(self, solution: CandidateSolution) -> List[int]:
        variable_name = self._solution_variable_name()
        route = list(solution.values.get(variable_name, []))
        if len(route) != self.size or len(set(route)) != self.size:
            raise ValueError("TSP route must visit every node exactly once.")
        try:
            return [self._node_index[node] for node in route]
        except KeyError as exc:
            raise ValueError("TSP route contains a node outside the graph.") from exc

    def decode(self, representation: List[Any]) -> CandidateSolution:
        indices = [int(index) for index in representation]
        if len(indices) != self.size or sorted(indices) != list(range(self.size)):
            raise ValueError("A TSP permutation must contain every node exactly once.")
        route = [self.nodes[index] for index in indices]
        return self.candidate(route, representation=SolutionRepresentationKind.PERMUTATION)

    def random_solution(self, rng: np.random.Generator) -> List[int]:
        return rng.permutation(self.size).astype(int).tolist()

    def neighbors(self, representation, rng, count, step_size=0.2):
        current = list(map(int, representation))
        neighbors: List[List[int]] = []
        for _ in range(max(0, count)):
            candidate = current.copy()
            i, j = rng.choice(self.size, size=2, replace=False)
            candidate[int(i)], candidate[int(j)] = candidate[int(j)], candidate[int(i)]
            neighbors.append(candidate)
        return neighbors

    def route_cost(self, representation: List[int]) -> float:
        indices = [int(index) for index in representation]
        if len(indices) != self.size or sorted(indices) != list(range(self.size)):
            raise ValueError("TSP representation must be a complete node permutation.")
        return float(sum(self._weights[indices[i], indices[(i + 1) % self.size]] for i in range(self.size)))

    def heuristic_matrix(self) -> np.ndarray:
        heuristic = np.zeros_like(self._weights)
        mask = np.isfinite(self._weights) & (self._weights > 0)
        heuristic[mask] = 1.0 / self._weights[mask]
        return heuristic


# Stable adapter registry used by compatibility and execution layers.
# The public names are domain contracts; callers do not depend on Python class names.
# Domain-level adapter registry. Kept here as compatibility aliases for older callers.
from adapters.registry import AdapterDescriptor, AdapterRegistry

BUILTIN_ADAPTER_REGISTRY = AdapterRegistry()
BUILTIN_ADAPTER_REGISTRY.register(
    AdapterDescriptor(
        id="graph_to_permutation",
        source_representation=SolutionRepresentationKind.GRAPH,
        target_representation=SolutionRepresentationKind.PERMUTATION,
        factory=GraphRoutingAdapter,
        description="Translate a TSP graph instance into a permutation search space.",
    )
)

BUILTIN_ADAPTERS = {descriptor.id for descriptor in BUILTIN_ADAPTER_REGISTRY.all()}
BUILTIN_ADAPTER_FACTORIES = {descriptor.id: descriptor.factory for descriptor in BUILTIN_ADAPTER_REGISTRY.all()}
