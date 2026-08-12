from __future__ import annotations

from heapq import heappop, heappush
from time import perf_counter
from typing import Any, Dict, List, Tuple

from domain.objectives import ObjectiveMetric, ObjectiveSense
from domain.representations import SolutionRepresentationKind
from domain.solutions import CandidateSolution, OptimizationResult
from adapters.problem_adapters import GraphProblemAdapter
from schemas import AlgorithmConfig
from .base import OptimizationAlgorithm


class _GraphExactBase(OptimizationAlgorithm):
    problem_adapter_type = GraphProblemAdapter

    def configure(self, config: AlgorithmConfig | None) -> None:
        return None

    def optimize(self, fitness_function, bounds, is_minimization=True, constraints=None):
        raise NotImplementedError("Graph-native solvers use optimize_problem_result().")

    def _adapter(self, problem) -> GraphProblemAdapter:
        return self.create_problem_adapter(problem)


class DijkstraShortestPath(_GraphExactBase):
    """Exact shortest path for non-negative directed or undirected graphs."""

    def optimize_problem_result(self, problem) -> OptimizationResult:
        adapter = self._adapter(problem)
        if problem.objective.metric is not None and problem.objective.metric != ObjectiveMetric.PATH_LENGTH:
            raise ValueError("Shortest path expects the path_length objective metric.")
        source = adapter.source
        target = adapter.target
        if source is None or target is None:
            raise ValueError("Shortest path requires graph metadata 'source' and 'target'.")
        start = perf_counter()
        adjacency = adapter.adjacency(required_only=False)
        distances = {node: float("inf") for node in adapter.nodes}
        previous: Dict[Any, Tuple[Any, Any]] = {}
        distances[source] = 0.0
        queue = [(0.0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            if node == target:
                break
            for neighbor, weight, edge_id in adjacency[node]:
                candidate = distance + weight
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    previous[neighbor] = (node, edge_id)
                    heappush(queue, (candidate, neighbor))
        if distances[target] == float("inf"):
            raise ValueError(f"No path exists between '{source}' and '{target}'.")
        edge_route: List[Any] = []
        node_route: List[Any] = [target]
        current = target
        while current != source:
            parent, edge_id = previous[current]
            edge_route.append(edge_id)
            node_route.append(parent)
            current = parent
        edge_route.reverse()
        node_route.reverse()
        candidate = self._candidate(adapter, edge_route, distances[target], {"nodes": node_route})
        elapsed = perf_counter() - start
        return OptimizationResult(candidate, distances[target], True, 1, len(previous) + 1, elapsed, (distances[target],), self.__class__.__name__, self.get_params_report())

    def optimize_problem(self, problem):
        return self.optimize_problem_result(problem).solution

    def _candidate(self, adapter, route, cost, metadata):
        return CandidateSolution(
            values=adapter.candidate(route, metadata=metadata, representation=SolutionRepresentationKind.GRAPH).values,
            representation=SolutionRepresentationKind.GRAPH,
            objective_value=float(cost),
            feasible=True,
            metadata=metadata,
        )

    def get_params_report(self):
        return {"exact": True, "backend": "native Dijkstra"}


class MinimumSpanningTree(_GraphExactBase):
    """Exact minimum spanning tree for connected undirected weighted graphs."""

    def optimize_problem_result(self, problem) -> OptimizationResult:
        adapter = self._adapter(problem)
        if problem.objective.metric is not None and problem.objective.metric != ObjectiveMetric.TOTAL_WEIGHT:
            raise ValueError("Minimum spanning tree expects the total_weight objective metric.")
        if adapter.directed:
            raise ValueError("Minimum spanning tree requires an undirected graph.")
        start = perf_counter()
        parent = {node: node for node in adapter.nodes}
        rank = {node: 0 for node in adapter.nodes}

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left, right):
            a, b = find(left), find(right)
            if a == b:
                return False
            if rank[a] < rank[b]:
                a, b = b, a
            parent[b] = a
            if rank[a] == rank[b]:
                rank[a] += 1
            return True

        edges = sorted(adapter.edges, key=lambda edge: float(edge["weight"]))
        selected = []
        cost = 0.0
        for edge in edges:
            if union(edge["u"], edge["v"]):
                selected.append(edge["id"])
                cost += float(edge["weight"])
        if len(selected) != max(0, len(adapter.nodes) - 1):
            raise ValueError("Minimum spanning tree requires a connected graph.")
        candidate = adapter.candidate(selected, metadata={"selected_edges": selected}, representation=SolutionRepresentationKind.GRAPH)
        candidate = CandidateSolution(values=candidate.values, representation=candidate.representation, objective_value=cost, feasible=True, metadata=candidate.metadata)
        elapsed = perf_counter() - start
        return OptimizationResult(candidate, cost, True, 1, len(edges), elapsed, (cost,), self.__class__.__name__, self.get_params_report())

    def optimize_problem(self, problem):
        return self.optimize_problem_result(problem).solution

    def get_params_report(self):
        return {"exact": True, "backend": "native Kruskal"}
