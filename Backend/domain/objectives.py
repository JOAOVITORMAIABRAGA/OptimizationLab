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

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            self.kind = ObjectiveKind(self.kind)
        if isinstance(self.sense, str):
            self.sense = ObjectiveSense(self.sense)
        if self.kind == ObjectiveKind.MULTI and not self.objectives and self.expression is not None:
            # Preserve legacy construction while allowing explicit multiobjective models.
            self.objectives = (ObjectiveComponent("objective", "Objective", self.sense, self.expression),)
