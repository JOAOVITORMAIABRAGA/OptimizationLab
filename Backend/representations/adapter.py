"""Backward-compatible facade for the pre-V10 representation adapter API.

The executable implementation now lives in ``adapters.problem_adapters``.
This module intentionally contains no algorithm-selection or execution logic.
New code should import adapters directly.
"""
from adapters.problem_adapters import (
    GraphProblemAdapter,
    NumericSearchAdapter,
    PermutationSearchAdapter,
    SearchSpaceAdapter,
)
from domain.representations import SolutionRepresentationKind

RepresentationAdapter = SearchSpaceAdapter
VectorRepresentationAdapter = NumericSearchAdapter
PermutationRepresentationAdapter = PermutationSearchAdapter
GraphRepresentationAdapter = GraphProblemAdapter


class RepresentationAdapterFactory:
    """Legacy factory retained only for external callers and old tests.

    Production solver code declares its adapter type directly and does not
    route through this factory.
    """

    @staticmethod
    def create(problem):
        representation = problem.solution_representation
        if representation is None:
            raise ValueError("A solution representation is required.")
        if representation.kind == SolutionRepresentationKind.VECTOR:
            return NumericSearchAdapter(problem)
        if representation.kind == SolutionRepresentationKind.PERMUTATION:
            return PermutationSearchAdapter(problem)
        if representation.kind in {
            SolutionRepresentationKind.GRAPH,
            SolutionRepresentationKind.EDGE_WALK,
            SolutionRepresentationKind.EDGE_SET,
        }:
            return GraphProblemAdapter(problem)
        raise ValueError(f"No execution adapter is implemented for representation '{representation.kind.value}'.")
