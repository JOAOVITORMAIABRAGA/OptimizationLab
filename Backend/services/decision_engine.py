from __future__ import annotations

from dataclasses import dataclass
from typing import List

from algorithms.registry import AlgorithmRegistry
from adapters.problem_adapters import BUILTIN_ADAPTERS
from compatibility.compatibility_engine import CompatibilityEngine
from domain.problem import OptimizationProblem
from recommendation.models import Recommendation, RecommendationResult
from recommendation.recommendation_engine import RecommendationEngine
from validation.validator import ValidationEngine


@dataclass(frozen=True)
class OptimizationDecision:
    selected_algorithm_id: str
    selected_algorithm_name: str
    score: float
    rationale: str
    alternatives: List[Recommendation]
    recommendations: RecommendationResult


class UniversalDecisionEngine:
    """Fast, deterministic algorithm selection before any expensive execution.

    The decision stage only inspects the mathematical structure and declared
    representations. It never runs competing algorithms just to decide what
    to run, which keeps the default path cheap and predictable.
    """

    def __init__(
        self,
        registry: AlgorithmRegistry | None = None,
        recommendation_engine: RecommendationEngine | None = None,
    ) -> None:
        self.registry = registry or AlgorithmRegistry.from_builtin_algorithms()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.compatibility = CompatibilityEngine()
        self.validator = ValidationEngine()

    def decide(self, problem: OptimizationProblem, alternative_count: int = 3) -> OptimizationDecision:
        report = self.validator.validate(problem)
        if not report.is_valid():
            raise ValueError("Invalid optimization problem: " + "; ".join(report.errors))

        recommendations = self.recommendation_engine.recommend(
            problem,
            self.registry,
            compatibility_engine=self.compatibility,
            available_adapters=set(BUILTIN_ADAPTERS),
        )
        if not recommendations.recommendations:
            raise ValueError("No compatible executable algorithm was found for this problem.")

        selected = recommendations.recommendations[0]
        return OptimizationDecision(
            selected_algorithm_id=selected.algorithm_id,
            selected_algorithm_name=selected.algorithm_name,
            score=selected.score,
            rationale=(
                f"Selected {selected.algorithm_name} using structural fit only "
                f"(score {selected.score:.2f}); no competing algorithms were executed during selection."
            ),
            alternatives=recommendations.recommendations[1 : 1 + max(0, alternative_count)],
            recommendations=recommendations,
        )
