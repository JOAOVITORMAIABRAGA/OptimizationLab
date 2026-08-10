from enum import Enum


class ProblemFamily(str, Enum):
    CONTINUOUS_OPTIMIZATION = "continuous_optimization"
    ROUTING = "routing"
    SCHEDULING = "scheduling"
    ASSIGNMENT = "assignment"
    TRANSPORTATION = "transportation"
    PRODUCTION_PLANNING = "production_planning"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    FEATURE_SELECTION = "feature_selection"
    GRAPH_OPTIMIZATION = "graph_optimization"
    RESOURCE_ALLOCATION = "resource_allocation"
    GENERIC = "generic"


class MathematicalProperty(str, Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BINARY = "binary"
    COMBINATORIAL = "combinatorial"
    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    INTEGER = "integer"
    MIXED_INTEGER = "mixed_integer"
    BLACK_BOX = "black_box"
    MULTIOBJECTIVE = "multiobjective"
