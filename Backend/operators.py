from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class OperatorRegistry:
    """Runtime catalog of operators that are actually implemented by the current algorithms."""

    operators: FrozenSet[str]

    @classmethod
    def builtin(cls) -> "OperatorRegistry":
        return cls(frozenset({
            "crossover",
            "mutation",
            "selection",
            "velocity_update",
            "position_update",
            "chemotaxis",
            "reproduction",
            "neighborhood",
            "acceptance_criterion",
            "pheromone",
            "construction",
            "tabu_list",
            "neighbor",
        }))

    def has(self, operator: str) -> bool:
        return operator in self.operators

    def available(self) -> set[str]:
        return set(self.operators)
