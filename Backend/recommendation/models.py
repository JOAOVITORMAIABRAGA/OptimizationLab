from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from compatibility.compatibility_engine import CompatibilityStatus


@dataclass(frozen=True)
class Recommendation:
    algorithm_id: str
    algorithm_name: str
    score: float
    rank: int
    rationale: str
    strengths: Tuple[str, ...] = field(default_factory=tuple)
    weaknesses: Tuple[str, ...] = field(default_factory=tuple)
    evidence: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExcludedAlgorithm:
    algorithm_id: str
    reason: str
    compatibility_status: Optional[CompatibilityStatus] = None
    evidence: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: List[Recommendation]
    excluded_algorithms: List[ExcludedAlgorithm]
    explanation: str
