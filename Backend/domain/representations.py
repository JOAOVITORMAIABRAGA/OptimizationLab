from enum import Enum


class SolutionRepresentationKind(str, Enum):
    """How a candidate solution is encoded for a solver."""

    VECTOR = "vector"
    PERMUTATION = "permutation"
    GRAPH = "graph"  # legacy graph-native solution contract; kept for compatibility
    EDGE_WALK = "edge_walk"
    EDGE_SET = "edge_set"
    SET = "set"
    SEQUENCE = "sequence"
    MIXED = "mixed"
    MATRIX = "matrix"
