from enum import Enum


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveKind(str, Enum):
    SINGLE = "single"
    MULTI = "multi"
