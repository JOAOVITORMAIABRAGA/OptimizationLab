from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from algorithms.registry import AlgorithmAvailability, AlgorithmDescriptor, AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from domain.objectives import ObjectiveKind
from domain.problem import OptimizationProblem
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from recommendation.models import ExcludedAlgorithm, Recommendation, RecommendationResult
from recommendation.policy import RecommendationScoringPolicy


class RecommendationEngine:
    def __init__(self, scoring_policy: Optional[RecommendationScoringPolicy] = None) -> None:
        self.scoring_policy = scoring_policy or RecommendationScoringPolicy()

    def recommend(
        self,
        problem: OptimizationProblem,
        registry: AlgorithmRegistry,
        compatibility_engine: Optional[CompatibilityEngine] = None,
        available_adapters: Optional[Set[str]] = None,
        available_operators: Optional[Set[str]] = None,
    ) -> RecommendationResult:
        compatibility_engine = compatibility_engine or CompatibilityEngine()
        if available_adapters is None:
            available_adapters = set()

        recommendations: List[Recommendation] = []
        excluded_algorithms: List[ExcludedAlgorithm] = []

        for descriptor in sorted(registry.get_all(), key=lambda item: item.id):
            if descriptor.availability != AlgorithmAvailability.AVAILABLE:
                excluded_algorithms.append(
                    ExcludedAlgorithm(
                        algorithm_id=descriptor.id,
                        reason=self._availability_reason(descriptor.availability),
                        compatibility_status=None,
                        evidence=(f"Availability: {descriptor.availability.value}",),
                    )
                )
                continue

            compatibility = compatibility_engine.check(
                problem,
                descriptor,
                available_adapters=available_adapters,
                available_operators=available_operators,
            )
            if compatibility.status == CompatibilityStatus.INCOMPATIBLE:
                excluded_algorithms.append(
                    ExcludedAlgorithm(
                        algorithm_id=descriptor.id,
                        reason=self._incompatibility_reason(problem, compatibility),
                        compatibility_status=compatibility.status,
                        evidence=compatibility.reasons,
                    )
                )
                continue

            score, strengths, weaknesses, evidence = self._score_candidate(problem, descriptor, compatibility)
            recommendations.append(
                Recommendation(
                    algorithm_id=descriptor.id,
                    algorithm_name=descriptor.name,
                    score=score,
                    rank=0,
                    rationale=self._build_rationale(descriptor, score, strengths, weaknesses),
                    strengths=tuple(strengths),
                    weaknesses=tuple(weaknesses),
                    evidence=tuple(evidence),
                )
            )

        scored_candidates = []
        for recommendation in recommendations:
            direct_support = 0 if any("direct support" in strength.lower() for strength in recommendation.strengths) else 1
            scored_candidates.append((recommendation.score, direct_support, recommendation.algorithm_id, recommendation))

        ranked = sorted(scored_candidates, key=lambda item: (-item[0], item[1], item[2]))
        ranked_recommendations = []
        for index, (_, _, _, recommendation) in enumerate(ranked, start=1):
            ranked_recommendations.append(
                Recommendation(
                    algorithm_id=recommendation.algorithm_id,
                    algorithm_name=recommendation.algorithm_name,
                    score=recommendation.score,
                    rank=index,
                    rationale=recommendation.rationale,
                    strengths=recommendation.strengths,
                    weaknesses=recommendation.weaknesses,
                    evidence=recommendation.evidence,
                )
            )

        explanation = self._build_result_explanation(problem, ranked_recommendations, excluded_algorithms)
        return RecommendationResult(
            recommendations=ranked_recommendations,
            excluded_algorithms=excluded_algorithms,
            explanation=explanation,
        )

    def _score_candidate(
        self,
        problem: OptimizationProblem,
        descriptor: AlgorithmDescriptor,
        compatibility: object,
    ) -> Tuple[float, List[str], List[str], List[str]]:
        capability = descriptor.get_capability(problem.solution_representation.kind if problem.solution_representation else None)
        representation_supported = capability is not None and capability.status == "supported"
        representation_supports_adapter = capability is not None and capability.status == "supported_with_adapter"

        problem_variables = problem.variables or []
        variable_types_match = all(variable.variable_type in descriptor.supported_variable_types for variable in problem_variables)

        problem_properties = set(problem.mathematical_properties)
        if not problem_properties:
            problem_properties = {MathematicalProperty.UNCONSTRAINED}

        matched_properties = [prop for prop in problem_properties if prop in descriptor.supported_mathematical_properties]
        mathematical_property_match = len(matched_properties) / max(1, len(problem_properties))

        family_match = problem.problem_family in descriptor.supported_problem_families
        constraints_present = bool(problem.constraints)
        constraints_supported = descriptor.supports_constraints if constraints_present else None
        objective_match = self._objective_match(problem, descriptor)
        compatibility_is_direct = compatibility.status == CompatibilityStatus.COMPATIBLE
        score = self.scoring_policy.score(
            representation_supported=representation_supported,
            representation_supports_adapter=representation_supports_adapter,
            family_match=family_match,
            variable_types_match=variable_types_match,
            mathematical_property_match=mathematical_property_match,
            constraints_supported=constraints_supported,
            constraints_present=constraints_present,
            objective_match=objective_match,
            compatibility_is_direct=compatibility_is_direct,
            algorithm_id=descriptor.id,
            problem_family=problem.problem_family,
            representation=problem.solution_representation.kind if problem.solution_representation else None,
        )

        strengths: List[str] = []
        weaknesses: List[str] = []
        evidence: List[str] = []

        if representation_supported:
            strengths.append("direct support for the declared solution representation")
        elif representation_supports_adapter:
            strengths.append("supports the representation with adaptation")
            weaknesses.append("requires adaptation for the target representation")
        else:
            weaknesses.append("does not natively support the required representation")

        if family_match:
            strengths.append(f"matches the problem family '{problem.problem_family.value}'")
        else:
            weaknesses.append("does not declare strong family specialization for this problem")

        if variable_types_match:
            strengths.append("supports the problem variable types")
        else:
            weaknesses.append("does not support all declared variable types")

        if matching_properties := [prop.value for prop in matched_properties]:
            strengths.append(f"covers mathematical properties: {', '.join(matching_properties)}")
        else:
            weaknesses.append("does not cover the problem mathematical properties")

        if constraints_present and descriptor.supports_constraints:
            strengths.append("supports constraint handling")
        elif constraints_present:
            weaknesses.append("does not declare constraint support")

        if objective_match:
            strengths.append("supports the declared objective semantics")
        else:
            weaknesses.append("does not fully support the declared objective semantics")

        if compatibility_is_direct:
            evidence.append("Compatibility status: compatible")
        else:
            evidence.append("Compatibility status: compatible_with_adaptation")
            strengths.append("remains viable through adaptation")

        if problem.problem_family == ProblemFamily.CONTINUOUS_OPTIMIZATION and problem.solution_representation and problem.solution_representation.kind == SolutionRepresentationKind.VECTOR:
            evidence.append("Structural fit favors continuous vector search")

        if not weaknesses:
            weaknesses.append("structural suitability does not guarantee empirical performance")

        evidence.append(f"Computed score: {score:.2f}")
        return score, strengths, weaknesses, evidence

    def _objective_match(self, problem: OptimizationProblem, descriptor: AlgorithmDescriptor) -> bool:
        if problem.objective is None:
            return False
        if problem.objective.kind == ObjectiveKind.MULTI:
            return descriptor.supports_multiobjective and problem.objective.sense in descriptor.supported_objectives
        return problem.objective.sense in descriptor.supported_objectives

    def _build_rationale(
        self,
        descriptor: AlgorithmDescriptor,
        score: float,
        strengths: Sequence[str],
        weaknesses: Sequence[str],
    ) -> str:
        strong_points = "; ".join(strengths[:3])
        weak_points = "; ".join(weaknesses[:2])
        if weak_points:
            return f"{descriptor.name} received a structural score of {score:.2f} because {strong_points}; limitations include {weak_points}."
        return f"{descriptor.name} received a structural score of {score:.2f} because {strong_points}."

    def _build_result_explanation(
        self,
        problem: OptimizationProblem,
        recommendations: Sequence[Recommendation],
        excluded_algorithms: Sequence[ExcludedAlgorithm],
    ) -> str:
        recommended = ", ".join(rec.algorithm_id for rec in recommendations) or "none"
        excluded = ", ".join(ex.algorithm_id for ex in excluded_algorithms) or "none"
        return (
            f"RecommendationEngine evaluated {problem.name} using compatibility filtering and deterministic structural scoring. "
            f"Recommended algorithms: {recommended}. Excluded algorithms: {excluded}."
        )

    def _availability_reason(self, availability: AlgorithmAvailability) -> str:
        return {
            AlgorithmAvailability.UNAVAILABLE: "algorithm is currently marked unavailable",
            AlgorithmAvailability.PLANNED: "algorithm is planned but not yet executable",
            AlgorithmAvailability.EXTERNAL: "algorithm is external and not part of the local executable set",
        }.get(availability, "algorithm is not available for execution")

    def _incompatibility_reason(self, problem: OptimizationProblem, compatibility: object) -> str:
        if compatibility.reasons:
            return "; ".join(compatibility.reasons)
        return "algorithm is incompatible with the problem"
