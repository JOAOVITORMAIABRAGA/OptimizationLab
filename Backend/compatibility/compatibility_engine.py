from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

from algorithms.registry import AlgorithmDescriptor, RepresentationCapability
from domain.objectives import ObjectiveKind, ObjectiveSense
from domain.problem import OptimizationProblem
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType


class CompatibilityStatus(str, Enum):
    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_ADAPTATION = "compatible_with_adaptation"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class CompatibilityResult:
    status: CompatibilityStatus
    algorithm_id: str
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    failed_checks: Tuple[str, ...] = field(default_factory=tuple)
    required_adapters: Tuple[str, ...] = field(default_factory=tuple)
    required_operators: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[str, ...] = field(default_factory=tuple)


class CompatibilityEngine:
    def check(
        self,
        problem: OptimizationProblem,
        algorithm: AlgorithmDescriptor,
        available_adapters: Optional[Set[str]] = None,
        available_operators: Optional[Set[str]] = None,
    ) -> CompatibilityResult:
        if available_adapters is None:
            available_adapters = set()
        if available_operators is None:
            available_operators = None

        reasons: List[str] = []
        failed_checks: List[str] = []
        required_adapters: List[str] = []
        required_operators: List[str] = []
        warnings: List[str] = []
        adaptation_possible = False

        representation_capability = algorithm.get_capability(problem.solution_representation.kind if problem.solution_representation else None)
        if problem.solution_representation is None:
            reasons.append("Problem has no declared solution representation.")
            failed_checks.append("representation")
        elif representation_capability is None:
            reasons.append(f"Algorithm does not declare support for representation '{problem.solution_representation.kind}'.")
            failed_checks.append("representation")
        else:
            if representation_capability.status == "unsupported":
                reasons.append(f"Representation '{problem.solution_representation.kind}' is not supported by the current implementation.")
                failed_checks.append("representation")
            elif representation_capability.status == "supported_with_adapter":
                adapter_names = list(representation_capability.required_adapters)
                failed_checks.append("representation")
                if not adapter_names:
                    reasons.append("Representation requires an adapter but none is declared.")
                elif not set(adapter_names).issubset(available_adapters):
                    reasons.append("Representation requires an adapter that is not available.")
                    required_adapters.extend(adapter_names)
                else:
                    required_adapters.extend(adapter_names)
                    adaptation_possible = True
                    warnings.append("Representation-specific adapter is available; compatibility is conditional.")

        variable_types = {variable.variable_type for variable in problem.variables}
        unsupported_variable_types = []
        for variable_type in variable_types:
            if variable_type not in algorithm.supported_variable_types:
                unsupported_variable_types.append(variable_type.value)
        if unsupported_variable_types:
            reasons.append(f"Variable types not supported by algorithm: {', '.join(unsupported_variable_types)}")
            failed_checks.append("variable_types")

        if problem.objective is None:
            reasons.append("Problem objective is missing.")
            failed_checks.append("objective")
        else:
            if problem.objective.kind == ObjectiveKind.MULTI and not algorithm.supports_multiobjective:
                reasons.append("Problem is multiobjective but the algorithm is not declared as multiobjective-capable.")
                failed_checks.append("objective")
            if problem.objective.sense not in algorithm.supported_objectives:
                reasons.append(f"Objective sense '{problem.objective.sense}' is not supported by the algorithm.")
                failed_checks.append("objective")

        if problem.constraints and not algorithm.supports_constraints:
            reasons.append("Problem has constraints but the algorithm does not declare constraint support.")
            failed_checks.append("constraints")

        if problem.problem_family not in algorithm.supported_problem_families:
            reasons.append(f"Problem family '{problem.problem_family}' is not supported by the algorithm.")
            failed_checks.append("problem_family")

        problem_properties = set(problem.mathematical_properties)
        unsupported_properties = []
        for property_name in problem_properties:
            if property_name == MathematicalProperty.UNCONSTRAINED:
                continue
            if property_name not in algorithm.supported_mathematical_properties:
                unsupported_properties.append(property_name.value)
        if unsupported_properties:
            reasons.append(f"Mathematical properties not supported by the algorithm: {', '.join(unsupported_properties)}")
            failed_checks.append("mathematical_properties")

        required_operator_names = []
        if representation_capability is not None:
            required_operator_names.extend(representation_capability.required_operators)
        required_operator_names.extend(algorithm.required_operators)
        if available_operators is not None:
            missing_operators = [operator for operator in required_operator_names if operator not in available_operators]
            if missing_operators:
                reasons.append("Algorithm requires operators that are not available: " + ", ".join(sorted(set(missing_operators))))
                failed_checks.append("required_operators")
                required_operators.extend(missing_operators)
        elif required_operator_names:
            warnings.append("Operator requirements are declared by the registry but no runtime operator availability set was supplied.")

        if algorithm.required_adapters:
            missing_adapters = [adapter for adapter in algorithm.required_adapters if adapter not in available_adapters]
            if missing_adapters:
                reasons.append("Algorithm requires adapters that are not available: " + ", ".join(missing_adapters))
                failed_checks.append("required_adapters")
                required_adapters.extend(missing_adapters)
            else:
                warnings.append("Algorithm-level adapter dependencies are available.")

        if failed_checks:
            if adaptation_possible and all(check in {"representation", "required_adapters", "required_operators"} for check in failed_checks):
                status = CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION
            else:
                status = CompatibilityStatus.INCOMPATIBLE
        else:
            status = CompatibilityStatus.COMPATIBLE

        return CompatibilityResult(
            status=status,
            algorithm_id=algorithm.id,
            reasons=tuple(reasons),
            failed_checks=tuple(failed_checks),
            required_adapters=tuple(required_adapters),
            required_operators=tuple(sorted(set(required_operators))),
            warnings=tuple(warnings),
        )
