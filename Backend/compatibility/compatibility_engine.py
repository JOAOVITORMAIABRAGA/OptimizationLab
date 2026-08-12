from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Tuple

from adapters.registry import AdaptationPlan, AdaptationStep, AdapterRegistry
from adapters.problem_adapters import BUILTIN_ADAPTER_REGISTRY
from algorithms.registry import AlgorithmDescriptor
from domain.objectives import ObjectiveKind
from domain.problem import OptimizationProblem
from domain.problem_family import MathematicalProperty
from domain.representations import SolutionRepresentationKind


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
    adaptation_plan: Optional[AdaptationPlan] = None

    @property
    def is_compatible(self) -> bool:
        return self.status != CompatibilityStatus.INCOMPATIBLE

    @property
    def is_direct(self) -> bool:
        return self.status == CompatibilityStatus.COMPATIBLE


class CompatibilityEngine:
    """Capability matcher for problems, algorithms and representation adapters.

    The engine never contains algorithm-specific branches. Algorithms describe
    their capabilities; adapters describe representation transformations.
    """

    def __init__(self, adapter_registry: Optional[AdapterRegistry] = None) -> None:
        self.adapter_registry = adapter_registry or BUILTIN_ADAPTER_REGISTRY

    def check(
        self,
        problem: OptimizationProblem,
        algorithm: AlgorithmDescriptor,
        available_adapters: Optional[Set[str]] = None,
        available_operators: Optional[Set[str]] = None,
    ) -> CompatibilityResult:
        available_adapters = set(available_adapters) if available_adapters is not None else {
            descriptor.id for descriptor in self.adapter_registry.all()
        }
        reasons: List[str] = []
        failed_checks: List[str] = []
        required_adapters: List[str] = []
        required_operators: List[str] = []
        warnings: List[str] = []
        adaptation_plan: Optional[AdaptationPlan] = None

        source_representation = problem.solution_representation.kind if problem.solution_representation else None
        capability = algorithm.get_capability(source_representation)
        if source_representation is None:
            reasons.append("Problem has no declared solution representation.")
            failed_checks.append("representation")
        elif capability is None:
            reasons.append(f"Algorithm does not declare support for representation '{source_representation.value}'.")
            failed_checks.append("representation")
        elif capability.status == "unsupported":
            reasons.append(f"Representation '{source_representation.value}' is not supported by the current implementation.")
            failed_checks.append("representation")
        elif capability.status == "supported":
            native_representation = capability.target_representation or source_representation
            if native_representation != source_representation:
                reasons.append("Algorithm capability declares a direct representation with a different target representation.")
                failed_checks.append("representation_contract")
            else:
                self._validate_native_adapter_materialization(problem, algorithm, reasons, failed_checks)
        elif capability.status == "supported_with_adapter":
            adapter_names = list(capability.required_adapters)
            if not adapter_names:
                reasons.append("Representation requires adaptation but no adapter is declared.")
                failed_checks.append("representation_adapter")
            else:
                adaptation_plan = self._build_adaptation_plan(
                    source_representation,
                    capability.target_representation,
                    adapter_names,
                    problem,
                    available_adapters,
                    reasons,
                    failed_checks,
                    required_adapters,
                )
                if adaptation_plan is not None:
                    warnings.append(
                        f"Representation is adapted from '{source_representation.value}' to "
                        f"'{adaptation_plan.target_representation.value}' before execution."
                    )

        adapted = adaptation_plan is not None

        self._check_problem_structure(problem, algorithm, reasons, failed_checks)
        self._check_objective(problem, algorithm, reasons, failed_checks)
        self._check_constraints(problem, algorithm, reasons, failed_checks)
        self._check_problem_family(problem, algorithm, reasons, failed_checks)

        if not adapted:
            self._check_variables(problem, algorithm, reasons, failed_checks)
            self._check_mathematical_properties(problem, algorithm, reasons, failed_checks)

        required_operator_names = list(algorithm.required_operators)
        if capability is not None:
            required_operator_names.extend(capability.required_operators)
        if available_operators is not None:
            missing = [name for name in required_operator_names if name not in available_operators]
            if missing:
                reasons.append("Algorithm requires operators that are not available: " + ", ".join(sorted(set(missing))))
                failed_checks.append("required_operators")
                required_operators.extend(missing)

        if algorithm.required_adapters:
            missing = [name for name in algorithm.required_adapters if name not in available_adapters]
            if missing:
                reasons.append("Algorithm requires adapters that are not available: " + ", ".join(missing))
                failed_checks.append("required_adapters")
                required_adapters.extend(missing)

        if adapted:
            status = CompatibilityStatus.COMPATIBLE_WITH_ADAPTATION if not failed_checks else CompatibilityStatus.INCOMPATIBLE
        else:
            status = CompatibilityStatus.COMPATIBLE if not failed_checks else CompatibilityStatus.INCOMPATIBLE

        return CompatibilityResult(
            status=status,
            algorithm_id=algorithm.id,
            reasons=tuple(dict.fromkeys(reasons)),
            failed_checks=tuple(dict.fromkeys(failed_checks)),
            required_adapters=tuple(dict.fromkeys(required_adapters)),
            required_operators=tuple(sorted(set(required_operators))),
            warnings=tuple(dict.fromkeys(warnings)),
            adaptation_plan=adaptation_plan if status != CompatibilityStatus.INCOMPATIBLE else None,
        )

    def _validate_native_adapter_materialization(self, problem, algorithm, reasons, failed_checks) -> None:
        adapter_type = getattr(algorithm.implementation_class, "problem_adapter_type", None)
        if adapter_type is None:
            return
        try:
            adapter_type(problem)
        except (TypeError, ValueError) as exc:
            reasons.append(f"Solver input adapter cannot materialize the problem: {exc}")
            failed_checks.append("problem_adapter")

    def _build_adaptation_plan(
        self,
        source: SolutionRepresentationKind,
        target: Optional[SolutionRepresentationKind],
        adapter_names: List[str],
        problem: OptimizationProblem,
        available_adapters: Set[str],
        reasons: List[str],
        failed_checks: List[str],
        required_adapters: List[str],
    ) -> Optional[AdaptationPlan]:
        current = source
        steps: List[AdaptationStep] = []
        for adapter_name in adapter_names:
            if adapter_name not in available_adapters:
                reasons.append(f"Required adapter '{adapter_name}' is not available.")
                failed_checks.append("representation_adapter")
                required_adapters.append(adapter_name)
                return None
            if not self.adapter_registry.has(adapter_name):
                # External providers may expose an adapter name without loading
                # its runtime factory into this process. The stable capability
                # contract remains valid; execution is delegated to the provider.
                if adapter_name not in available_adapters:
                    reasons.append(f"Required adapter '{adapter_name}' is not available.")
                    failed_checks.append("representation_adapter")
                    required_adapters.append(adapter_name)
                    return None
                target_representation = target or current
                steps.append(AdaptationStep(adapter_name, current, target_representation))
                current = target_representation
                required_adapters.append(adapter_name)
                continue
            descriptor = self.adapter_registry.get(adapter_name)
            if descriptor.source_representation != current:
                reasons.append(
                    f"Adapter '{adapter_name}' expects '{descriptor.source_representation.value}', "
                    f"but the current representation is '{current.value}'."
                )
                failed_checks.append("representation_adapter")
                return None
            try:
                self.adapter_registry.create(adapter_name, problem)
            except (TypeError, ValueError) as exc:
                reasons.append(f"Adapter '{adapter_name}' cannot materialize the problem: {exc}")
                failed_checks.append("representation_adapter")
                return None
            steps.append(AdaptationStep(adapter_name, descriptor.source_representation, descriptor.target_representation))
            current = descriptor.target_representation
            required_adapters.append(adapter_name)

        if target is not None and current != target:
            reasons.append(
                f"Adaptation ends at '{current.value}', but the algorithm requires '{target.value}'."
            )
            failed_checks.append("representation_contract")
            return None
        return AdaptationPlan(source, current, tuple(steps))

    def _check_problem_structure(self, problem, algorithm, reasons, failed_checks) -> None:
        structure = problem.problem_structure
        if algorithm.supported_problem_structures and structure is not None and structure.kind not in algorithm.supported_problem_structures:
            reasons.append(f"Algorithm does not support problem structure '{structure.kind.value}'.")
            failed_checks.append("problem_structure")
        if algorithm.supported_instance_kinds:
            metadata = dict((structure.metadata if structure is not None else {}) or {})
            if problem.solution_representation is not None:
                metadata.update(problem.solution_representation.metadata or {})
            instance_kind = str(metadata.get("graph_problem_type", ""))
            if instance_kind and instance_kind not in algorithm.supported_instance_kinds:
                reasons.append(f"Algorithm does not support problem instance kind '{instance_kind}'.")
                failed_checks.append("instance_kind")

    def _check_objective(self, problem, algorithm, reasons, failed_checks) -> None:
        objective = problem.objective
        if objective is None:
            reasons.append("Problem objective is missing.")
            failed_checks.append("objective")
            return
        if objective.kind == ObjectiveKind.MULTI and not algorithm.supports_multiobjective:
            reasons.append("Problem is multiobjective but the algorithm is not declared as multiobjective-capable.")
            failed_checks.append("objective")
        if objective.sense not in algorithm.supported_objectives:
            reasons.append(f"Objective sense '{objective.sense.value if objective.sense else objective.sense}' is not supported by the algorithm.")
            failed_checks.append("objective")
        if objective.metric is not None and algorithm.supported_objective_metrics:
            metric = getattr(objective.metric, "value", objective.metric)
            supported = {getattr(item, "value", item) for item in algorithm.supported_objective_metrics}
            if metric not in supported:
                reasons.append(f"Objective metric '{metric}' is not supported by the algorithm.")
                failed_checks.append("objective_metric")

    def _check_constraints(self, problem, algorithm, reasons, failed_checks) -> None:
        if problem.constraints and not algorithm.supports_constraints:
            reasons.append("Problem has constraints but the algorithm does not declare constraint support.")
            failed_checks.append("constraints")

    def _check_problem_family(self, problem, algorithm, reasons, failed_checks) -> None:
        if problem.problem_family not in algorithm.supported_problem_families:
            reasons.append(f"Problem family '{problem.problem_family.value}' is not supported by the algorithm.")
            failed_checks.append("problem_family")

    def _check_variables(self, problem, algorithm, reasons, failed_checks) -> None:
        variable_types = {variable.variable_type for variable in problem.variables}
        unsupported = [item.value for item in variable_types if item not in algorithm.supported_variable_types]
        if unsupported:
            reasons.append("Variable types not supported by algorithm: " + ", ".join(sorted(unsupported)))
            failed_checks.append("variable_types")

    def _check_mathematical_properties(self, problem, algorithm, reasons, failed_checks) -> None:
        properties = set(problem.mathematical_properties)
        missing = [prop.value for prop in algorithm.required_mathematical_properties if prop not in properties]
        if missing:
            reasons.append("Problem does not declare required mathematical properties: " + ", ".join(missing))
            failed_checks.append("required_mathematical_properties")
        unsupported = [prop.value for prop in properties if prop != MathematicalProperty.UNCONSTRAINED and prop not in algorithm.supported_mathematical_properties]
        if unsupported:
            reasons.append("Mathematical properties not supported by the algorithm: " + ", ".join(sorted(unsupported)))
            failed_checks.append("mathematical_properties")
