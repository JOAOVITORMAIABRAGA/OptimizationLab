from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from .expressions import StructuredExpression


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveKind(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class ObjectiveStatus(str, Enum):
    """Semantic completeness of an objective in the declarative model."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    NOT_APPLICABLE = "not_applicable"


class ObjectiveMetric(str, Enum):
    """Semantic objective metrics that do not require an algebraic expression.

    An expression is the executable algebraic form used by expression-based
    solvers. A metric describes the meaning of the objective for native
    solvers such as graph algorithms.
    """

    TOTAL_DISTANCE = "total_distance"
    TOTAL_COST = "total_cost"
    TOTAL_RETURN = "total_return"
    TOUR_LENGTH = "tour_length"
    PATH_LENGTH = "path_length"
    TOTAL_WEIGHT = "total_weight"
    MSE = "mse"
    MAE = "mae"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ObjectiveComponent:
    id: str
    name: str
    sense: ObjectiveSense
    expression: StructuredExpression


@dataclass
class ObjectiveSpec:
    kind: Optional[ObjectiveKind]
    sense: Optional[ObjectiveSense] = None
    expression: Optional[StructuredExpression] = None
    weights: Optional[list[float]] = None
    description: Optional[str] = None
    objectives: Tuple[ObjectiveComponent, ...] = field(default_factory=tuple)
    metric: Optional[ObjectiveMetric | str] = None
    status: ObjectiveStatus = ObjectiveStatus.COMPLETE

    @property
    def is_complete(self) -> bool:
        return self.status == ObjectiveStatus.COMPLETE

    @property
    def is_semantic(self) -> bool:
        return self.metric is not None and self.expression is None

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = ObjectiveStatus(self.status)
        if isinstance(self.kind, str):
            self.kind = ObjectiveKind(self.kind)
        if isinstance(self.sense, str):
            self.sense = ObjectiveSense(self.sense)
        if isinstance(self.metric, str):
            try:
                self.metric = ObjectiveMetric(self.metric)
            except ValueError:
                # Preserve extensibility for domain-specific metrics.
                self.metric = self.metric
        if self.kind == ObjectiveKind.MULTI and not self.objectives and self.expression is not None:
            # Preserve legacy construction while allowing explicit multiobjective models.
            self.objectives = (ObjectiveComponent("objective", "Objective", self.sense, self.expression),)
