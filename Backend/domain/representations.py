from enum import Enum


class SolutionRepresentationKind(str, Enum):
    VECTOR = "vector"
    PERMUTATION = "permutation"
    GRAPH = "graph"
    SET = "set"
    SEQUENCE = "sequence"
    MIXED = "mixed"
    MATRIX = "matrix"
