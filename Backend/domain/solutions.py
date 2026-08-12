from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .representations import SolutionRepresentationKind


@dataclass(frozen=True)
class CandidateSolution:
    """Semantic solution shared by all optimization algorithms."""

    values: Dict[str, Any]
    representation: SolutionRepresentationKind
    objective_value: Optional[float] = None
    feasible: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def vector(self) -> List[Any]:
        return list(self.values.values())


@dataclass(frozen=True)
class OptimizationResult:
    """Standard result returned by every optimization algorithm."""

    solution: CandidateSolution
    objective_value: float
    feasible: bool
    iterations: int
    evaluations: int
    runtime_seconds: float
    convergence_history: tuple[float, ...] = field(default_factory=tuple)
    algorithm: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def values(self) -> Dict[str, Any]:
        return self.solution.values
