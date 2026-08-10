from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from domain.problem_family import ProblemFamily
from domain.representations import SolutionRepresentationKind


@dataclass(frozen=True)
class RecommendationWeights:
    representation_match: float = 0.38
    family_match: float = 0.18
    variable_type_match: float = 0.14
    mathematical_property_match: float = 0.14
    constraint_support: float = 0.08
    objective_match: float = 0.08
    adaptation_penalty: float = 0.08
    specialization_bonus: float = 0.05


@dataclass(frozen=True)
class RecommendationScoringPolicy:
    weights: RecommendationWeights = field(default_factory=RecommendationWeights)
    specialization_bonus_by_algorithm_id: Dict[str, float] = field(
        default_factory=lambda: {
            "pso": 0.05,
            "de": 0.03,
            "bfo": 0.03,
            "ga": 0.02,
            "sa": 0.01,
        }
    )

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
        compatibility_is_direct: bool,
        algorithm_id: str,
        problem_family: ProblemFamily,
        representation: Optional[SolutionRepresentationKind],
    ) -> float:
        score = 0.0
        if representation_supported:
            score += self.weights.representation_match
        elif representation_supports_adapter:
            score += self.weights.representation_match * 0.6

        if family_match:
            score += self.weights.family_match

        if variable_types_match:
            score += self.weights.variable_type_match

        score += self.weights.mathematical_property_match * mathematical_property_match

        if constraints_present and constraints_supported is True:
            score += self.weights.constraint_support
        elif constraints_present and constraints_supported is False:
            score -= self.weights.constraint_support * 0.2

        if objective_match:
            score += self.weights.objective_match

        if not compatibility_is_direct:
            score -= self.weights.adaptation_penalty

        if (
            problem_family == ProblemFamily.CONTINUOUS_OPTIMIZATION
            and representation == SolutionRepresentationKind.VECTOR
            and algorithm_id in self.specialization_bonus_by_algorithm_id
        ):
            score += self.specialization_bonus_by_algorithm_id[algorithm_id]

        return max(0.0, min(1.0, score))
