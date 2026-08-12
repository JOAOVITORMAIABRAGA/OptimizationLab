from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from compatibility.compatibility_engine import CompatibilityStatus


@dataclass(frozen=True)
class AlgorithmCandidate:
    """Executable candidate separated from the system's recommendation decision."""

    algorithm_id: str
    algorithm_name: str
    compatibility: CompatibilityStatus
    compatibility_score: float
    recommendation_score: float
    adaptation: Tuple[str, ...] = field(default_factory=tuple)
    estimated_cost: str = "unknown"
    algorithm_type: str = "heuristic"
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)
    recommended: bool = False


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
    candidates: List[AlgorithmCandidate]
    recommendations: List[Recommendation]
    excluded_algorithms: List[ExcludedAlgorithm]
    explanation: str
