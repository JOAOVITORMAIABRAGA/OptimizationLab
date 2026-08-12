from __future__ import annotations

from functools import lru_cache
from heapq import heappop, heappush
from time import perf_counter
from typing import Any, Dict, List, Tuple

from algorithms.base import OptimizationAlgorithm
from domain.objectives import ObjectiveMetric, ObjectiveSense
from domain.representations import SolutionRepresentationKind
from domain.solutions import CandidateSolution, OptimizationResult
from adapters.problem_adapters import GraphProblemAdapter
from schemas import AlgorithmConfig


class ChinesePostmanProblem(OptimizationAlgorithm):
    """Exact undirected Chinese Postman solver for connected required-edge graphs.

    The implementation uses the classical formulation: traverse every required
    edge at least once, pair odd-degree vertices by shortest-path duplication,
    then construct an Euler tour. Minimum perfect matching is solved exactly by
    dynamic programming; therefore the implementation intentionally rejects
    instances with too many odd vertices instead of silently using a heuristic.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.max_odd_vertices = 18
        self._last_backend = "native exact undirected Chinese Postman"

    problem_adapter_type = GraphProblemAdapter

    def configure(self, config: AlgorithmConfig | None) -> None:
        if config is None:
            return
        params = getattr(config, "parameters", None) or {}
        if "max_odd_vertices" in params:
            self.max_odd_vertices = int(params["max_odd_vertices"])

    def optimize(self, problem, bounds=None, is_minimization=True, constraints=None):
        if not hasattr(problem, "solution_representation"):
            raise TypeError("ChinesePostmanProblem requires an OptimizationProblem.")
        result = self.optimize_problem_result(problem)
        route = result.solution.values[next(iter(result.solution.values))]
        return route, result.objective_value

    def optimize_problem(self, problem) -> CandidateSolution:
        return self.optimize_problem_result(problem).solution

    def optimize_problem_result(self, problem) -> OptimizationResult:
        adapter = self.create_problem_adapter(problem)
        if adapter.graph_problem_type != "chinese_postman":
            raise ValueError("Graph problem_type must be 'chinese_postman' for this solver.")
        if problem.objective.sense != ObjectiveSense.MINIMIZE:
            raise ValueError("Chinese Postman is currently a minimization problem.")
        if problem.objective.metric is not None and problem.objective.metric != ObjectiveMetric.TOTAL_DISTANCE:
            raise ValueError("Chinese Postman expects the total_distance objective metric.")

        start = perf_counter()
        route, cost, duplicated = self._solve(adapter)
        elapsed = perf_counter() - start
        candidate = adapter.candidate(route, representation=SolutionRepresentationKind.GRAPH)
        candidate = CandidateSolution(
            values=candidate.values,
            representation=candidate.representation,
            objective_value=cost,
            feasible=True,
            metadata={
                "algorithm": self.__class__.__name__,
                "graph_problem_type": adapter.graph_problem_type,
                "duplicated_edges": duplicated,
            },
        )
        self._last_iterations = 1
        self._last_history = [cost]
        return OptimizationResult(
            solution=candidate,
            objective_value=float(cost),
            feasible=True,
            iterations=1,
            evaluations=1,
            runtime_seconds=elapsed,
            convergence_history=(float(cost),),
            algorithm=self.__class__.__name__,
            parameters=self.get_params_report(),
            metadata={"exact": True, "duplicated_edges": duplicated},
        )

    def _solve(self, adapter: GraphProblemAdapter) -> Tuple[List[Any], float, List[Any]]:
        edges = adapter.edges
        required = [edge for edge in edges if bool(edge.get("required", True))]
        if not required:
            raise ValueError("Chinese Postman requires at least one required edge.")

        base_adjacency = adapter.adjacency(required_only=True)
        self._assert_connected_required_graph(adapter, base_adjacency)
        degree: Dict[Any, int] = {node: 0 for node in adapter.nodes}
        base_cost = 0.0
        for edge in required:
            degree[edge["u"]] += 1
            degree[edge["v"]] += 1
            base_cost += float(edge["weight"])
        odd = [node for node, value in degree.items() if value % 2]
        if len(odd) > self.max_odd_vertices:
            raise ValueError(
                f"Exact Chinese Postman matching would require {len(odd)} odd vertices; "
                f"the configured exact limit is {self.max_odd_vertices}."
            )

        if not odd:
            route = self._euler_tour(required, adapter.nodes[0])
            return route, base_cost, []

        full_adjacency = adapter.adjacency(required_only=False)
        pair_paths: Dict[Tuple[Any, Any], Tuple[float, List[Any]]] = {}
        for i, source in enumerate(odd):
            distances, paths = self._dijkstra(source, full_adjacency)
            for target in odd[i + 1 :]:
                if target not in distances:
                    raise ValueError("Graph is disconnected; no shortest path exists between odd vertices.")
                pair_paths[(source, target)] = (distances[target], paths[target])

        matching_cost, pairs = self._minimum_matching(odd, pair_paths)
        augmented = list(required)
        duplicated: List[Any] = []
        for left, right in pairs:
            _, path_edge_ids = pair_paths[(left, right) if (left, right) in pair_paths else (right, left)]
            for edge_id in path_edge_ids:
                edge = adapter.edge_map()[edge_id]
                augmented.append(dict(edge))
                duplicated.append(edge_id)

        route = self._euler_tour(augmented, required[0]["u"])
        return route, base_cost + matching_cost, duplicated

    def _assert_connected_required_graph(self, adapter, adjacency) -> None:
        start = next((node for node, links in adjacency.items() if links), None)
        if start is None:
            raise ValueError("Required graph has no edges.")
        visited = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor, _, _ in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        required_nodes = {edge["u"] for edge in adapter.edges if edge.get("required", True)} | {edge["v"] for edge in adapter.edges if edge.get("required", True)}
        if visited != required_nodes:
            raise ValueError("Chinese Postman currently requires the required-edge subgraph to be connected.")

    def _dijkstra(self, source, adjacency):
        distances = {source: 0.0}
        previous: Dict[Any, Tuple[Any, Any]] = {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance != distances.get(node):
                continue
            for neighbor, weight, edge_id in adjacency[node]:
                candidate = distance + weight
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = (node, edge_id)
                    heappush(queue, (candidate, neighbor))
        paths: Dict[Any, List[Any]] = {source: []}
        for target in distances:
            if target == source:
                continue
            path = []
            current = target
            while current != source:
                parent, edge_id = previous[current]
                path.append(edge_id)
                current = parent
            paths[target] = list(reversed(path))
        return distances, paths

    def _minimum_matching(self, vertices, pair_paths):
        vertices = tuple(vertices)

        @lru_cache(maxsize=None)
        def solve(remaining: Tuple[Any, ...]):
            if not remaining:
                return 0.0, ()
            first = remaining[0]
            best_cost = float("inf")
            best_pairs = ()
            for offset in range(1, len(remaining)):
                second = remaining[offset]
                key = (first, second)
                if key not in pair_paths:
                    key = (second, first)
                pair_cost = pair_paths[key][0]
                rest = remaining[1:offset] + remaining[offset + 1:]
                rest_cost, rest_pairs = solve(rest)
                total = pair_cost + rest_cost
                if total < best_cost:
                    best_cost = total
                    best_pairs = ((first, second),) + rest_pairs
            return best_cost, best_pairs

        return solve(vertices)

    def _euler_tour(self, edges: List[Dict[str, Any]], start_node: Any) -> List[Any]:
        adjacency: Dict[Any, List[Tuple[Any, Any, Any]]] = {}
        for index, edge in enumerate(edges):
            traversal_id = (edge["id"], index)
            adjacency.setdefault(edge["u"], []).append((edge["v"], traversal_id, edge["id"]))
            adjacency.setdefault(edge["v"], []).append((edge["u"], traversal_id, edge["id"]))

        used = set()
        stack = [start_node]
        edge_stack: List[Any] = []
        circuit: List[Any] = []
        while stack:
            node = stack[-1]
            while adjacency.get(node) and adjacency[node][-1][1] in used:
                adjacency[node].pop()
            if adjacency.get(node):
                neighbor, traversal_id, edge_id = adjacency[node].pop()
                if traversal_id in used:
                    continue
                used.add(traversal_id)
                stack.append(neighbor)
                edge_stack.append(edge_id)
            else:
                stack.pop()
                if edge_stack:
                    circuit.append(edge_stack.pop())
        circuit.reverse()
        if len(circuit) != len(edges):
            raise ValueError("Failed to construct a complete Euler tour for the augmented graph.")
        return circuit

    def get_params_report(self) -> Dict[str, Any]:
        return {
            "backend": self._last_backend,
            "exact": True,
            "max_odd_vertices": self.max_odd_vertices,
        }
