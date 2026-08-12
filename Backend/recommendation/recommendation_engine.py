from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Set, Tuple

from adapters.registry import AdapterRegistry
from adapters.problem_adapters import BUILTIN_ADAPTER_REGISTRY
from algorithms.registry import AlgorithmAvailability, AlgorithmDescriptor, AlgorithmRegistry
from compatibility.compatibility_engine import CompatibilityEngine, CompatibilityStatus
from domain.objectives import ObjectiveKind
from domain.problem import OptimizationProblem
from domain.problem_family import MathematicalProperty
from recommendation.models import AlgorithmCandidate, ExcludedAlgorithm, Recommendation, RecommendationResult
from recommendation.policy import RecommendationScoringPolicy


class RecommendationEngine:
    """Rank already-compatible candidates; never selects on behalf of the user."""

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
        compatibility_engine = compatibility_engine or CompatibilityEngine(BUILTIN_ADAPTER_REGISTRY)
        available_adapters = set(available_adapters) if available_adapters is not None else {
            descriptor.id for descriptor in BUILTIN_ADAPTER_REGISTRY.all()
        }

        candidates: List[AlgorithmCandidate] = []
        excluded: List[ExcludedAlgorithm] = []
        details: dict[str, tuple[list[str], list[str], list[str]]] = {}

        for descriptor in sorted(registry.get_all(), key=lambda item: item.id):
            if descriptor.availability != AlgorithmAvailability.AVAILABLE:
                excluded.append(
                    ExcludedAlgorithm(
                        algorithm_id=descriptor.id,
                        reason=f"algorithm is {descriptor.availability.value}",
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
            if not compatibility.is_compatible:
                excluded.append(
                    ExcludedAlgorithm(
                        algorithm_id=descriptor.id,
                        reason="; ".join(compatibility.reasons) or "algorithm is incompatible with the problem",
                        compatibility_status=compatibility.status,
                        evidence=compatibility.reasons,
                    )
                )
                continue

            score, strengths, weaknesses, evidence = self._score_candidate(problem, descriptor, compatibility)
            candidate = AlgorithmCandidate(
                algorithm_id=descriptor.id,
                algorithm_name=descriptor.name,
                compatibility=compatibility.status,
                compatibility_score=1.0 if compatibility.status == CompatibilityStatus.COMPATIBLE else 0.9,
                recommendation_score=score,
                adaptation=compatibility.required_adapters,
                estimated_cost=descriptor.estimated_cost,
                algorithm_type=descriptor.algorithm_type,
                reasons=compatibility.reasons,
                warnings=compatibility.warnings,
            )
            candidates.append(candidate)
            details[descriptor.id] = (strengths, weaknesses, evidence)

        ranked = sorted(candidates, key=lambda item: (-item.recommendation_score, -registry.get(item.algorithm_id).recommendation_priority, item.compatibility.value != CompatibilityStatus.COMPATIBLE.value, item.algorithm_id))
        recommendations: List[Recommendation] = []
        final_candidates: List[AlgorithmCandidate] = []
        for rank, candidate in enumerate(ranked, start=1):
            strengths, weaknesses, evidence = details[candidate.algorithm_id]
            recommended = rank == 1
            final_candidates.append(replace(candidate, recommended=recommended))
            recommendations.append(
                Recommendation(
                    algorithm_id=candidate.algorithm_id,
                    algorithm_name=candidate.algorithm_name,
                    score=candidate.recommendation_score,
                    rank=rank,
                    rationale=self._build_rationale(candidate, strengths, weaknesses),
                    strengths=tuple(strengths),
                    weaknesses=tuple(weaknesses),
                    evidence=tuple(evidence),
                )
            )

        return RecommendationResult(
            candidates=final_candidates,
            recommendations=recommendations,
            excluded_algorithms=excluded,
            explanation=(
                f"{len(final_candidates)} executable algorithms are compatible with {problem.name}. "
                "Recommendation score is structural suitability, not an empirical performance benchmark. "
                "The user remains responsible for the final algorithm selection."
            ),
        )

    def _score_candidate(self, problem, descriptor, compatibility):
        capability = descriptor.get_capability(problem.solution_representation.kind if problem.solution_representation else None)
        direct = capability is not None and capability.status == "supported"
        adapted = capability is not None and capability.status == "supported_with_adapter"
        variable_match = all(v.variable_type in descriptor.supported_variable_types for v in problem.variables)
        properties = set(problem.mathematical_properties) or {MathematicalProperty.UNCONSTRAINED}
        matched = [prop for prop in properties if prop in descriptor.supported_mathematical_properties]
        property_match = len(matched) / max(1, len(properties))
        family_match = problem.problem_family in descriptor.supported_problem_families
        constraints_present = bool(problem.constraints)
        constraints_supported = descriptor.supports_constraints if constraints_present else None
        objective_match = self._objective_match(problem, descriptor)
        score = self.scoring_policy.score(
            representation_supported=direct,
            representation_supports_adapter=adapted,
            family_match=family_match,
            variable_types_match=variable_match,
            mathematical_property_match=property_match,
            constraints_supported=constraints_supported,
            constraints_present=constraints_present,
            objective_match=objective_match,
            compatibility_status=compatibility.status,
        )
        score = min(1.0, score + descriptor.recommendation_priority)
        strengths, weaknesses, evidence = [], [], []
        strengths.append("direct support for the declared solution representation" if direct else "supports the representation through a declared adapter")
        if family_match: strengths.append("matches the problem family")
        if variable_match: strengths.append("supports the declared variable types")
        if matched: strengths.append("covers declared mathematical properties: " + ", ".join(prop.value for prop in matched))
        if constraints_present and descriptor.supports_constraints: strengths.append("supports constraint handling")
        if objective_match: strengths.append("supports the declared objective semantics")
        if not variable_match: weaknesses.append("does not natively cover every variable type")
        if constraints_present and not descriptor.supports_constraints: weaknesses.append("does not declare constraint support")
        if not objective_match: weaknesses.append("does not fully match the objective semantics")
        if adapted: weaknesses.append("requires representation adaptation")
        if not weaknesses: weaknesses.append("structural fit does not imply empirical superiority")
        evidence.append(f"Compatibility status: {compatibility.status.value}")
        evidence.append(f"Structural recommendation score: {score:.2f}")
        return score, strengths, weaknesses, evidence

    @staticmethod
    def _objective_match(problem, descriptor) -> bool:
        objective = problem.objective
        if objective is None:
            return False
        if objective.kind == ObjectiveKind.MULTI:
            return descriptor.supports_multiobjective and objective.sense in descriptor.supported_objectives
        if objective.sense not in descriptor.supported_objectives:
            return False
        if objective.metric is not None and descriptor.supported_objective_metrics:
            metric = getattr(objective.metric, "value", objective.metric)
            return metric in {getattr(item, "value", item) for item in descriptor.supported_objective_metrics}
        return True

    @staticmethod
    def _build_rationale(candidate, strengths, weaknesses) -> str:
        reason = "; ".join(strengths[:3]) or "matches the declared capabilities"
        limitation = "; ".join(weaknesses[:2])
        return f"{candidate.algorithm_name} is structurally suitable because {reason}." + (f" Limitations: {limitation}." if limitation else "")
