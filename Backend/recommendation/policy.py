from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from compatibility.compatibility_engine import CompatibilityStatus


@dataclass(frozen=True)
class RecommendationWeights:
    representation_match: float = 0.40
    family_match: float = 0.18
    variable_type_match: float = 0.12
    mathematical_property_match: float = 0.12
    constraint_support: float = 0.08
    objective_match: float = 0.10
    adaptation_penalty: float = 0.06


@dataclass(frozen=True)
class RecommendationScoringPolicy:
    """Deterministic structural scoring with no algorithm-specific branches."""

    weights: RecommendationWeights = field(default_factory=RecommendationWeights)

    def score(
        self,
        *,
        representation_supported: bool,
        representation_supports_adapter: bool,
        family_match: bool,
        variable_types_match: bool,
        mathematical_property_match: float,
        constraints_supported: Optional[bool],
        constraints_present: bool,
        objective_match: bool,
        compatibility_status: CompatibilityStatus,
    ) -> float:
        score = 0.0
        if representation_supported:
            score += self.weights.representation_match
        elif representation_supports_adapter:
            score += self.weights.representation_match * 0.75

        if family_match:
            score += self.weights.family_match
        if variable_types_match:
            score += self.weights.variable_type_match
        score += self.weights.mathematical_property_match * mathematical_property_match

        if constraints_present and constraints_supported is True:
            score += self.weights.constraint_support
        elif constraints_present and constraints_supported is False:
            score -= self.weights.constraint_support

        if objective_match:
            score += self.weights.objective_match
        if compatibility_status == CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION:
            score -= self.weights.adaptation_penalty

        return max(0.0, min(1.0, score))
