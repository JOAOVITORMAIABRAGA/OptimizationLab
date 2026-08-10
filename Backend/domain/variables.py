from enum import Enum


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    DISCRETE = "discrete"
