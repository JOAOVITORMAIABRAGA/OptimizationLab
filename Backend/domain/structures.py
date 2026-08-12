from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ProblemStructureKind(str, Enum):
    """How the optimization instance itself is structurally organized."""

    TABULAR = "tabular"
    VECTOR = "vector"
    GRAPH = "graph"
    MATRIX = "matrix"
    GENERIC = "generic"


@dataclass(frozen=True)
class ProblemStructureSpec:
    """Problem-instance structure, deliberately separate from solution encoding."""

    kind: ProblemStructureKind
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
