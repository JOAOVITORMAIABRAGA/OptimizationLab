from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type, Union

from algorithms.base import OptimizationAlgorithm
from algorithms.aco import AntColonyOptimization
from algorithms.bfo import BacterialForagingOptimization
from algorithms.de import DifferentialEvolution
from algorithms.ga import GeneticAlgorithm
from algorithms.hill_climbing import HillClimbing
from algorithms.pso import ParticleSwarmOptimization
from algorithms.sa import SimulatedAnnealing
from algorithms.tabu import TabuSearch
from algorithms.classical import ConstraintProgramming, IntegerProgramming, LinearProgramming
from domain.objectives import ObjectiveSense
from domain.problem_family import MathematicalProperty, ProblemFamily
from domain.representations import SolutionRepresentationKind
from domain.variables import VariableType


class AlgorithmAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    PLANNED = "planned"
    EXTERNAL = "external"


_IMPLEMENTATION_LOOKUP = {
    "GeneticAlgorithm": GeneticAlgorithm,
    "ParticleSwarmOptimization": ParticleSwarmOptimization,
    "DifferentialEvolution": DifferentialEvolution,
    "BacterialForagingOptimization": BacterialForagingOptimization,
    "SimulatedAnnealing": SimulatedAnnealing,
    "AntColonyOptimization": AntColonyOptimization,
    "TabuSearch": TabuSearch,
    "HillClimbing": HillClimbing,
    "LinearProgramming": LinearProgramming,
    "IntegerProgramming": IntegerProgramming,
    "ConstraintProgramming": ConstraintProgramming,
}


@dataclass(frozen=True)
class RepresentationCapability:
    representation: SolutionRepresentationKind
    status: str
    required_operators: Tuple[str, ...] = field(default_factory=tuple)
    required_adapters: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"supported", "supported_with_adapter", "unsupported"}:
            raise ValueError("status must be one of supported/supported_with_adapter/unsupported")
        if not isinstance(self.representation, SolutionRepresentationKind):
            raise ValueError("representation must be a SolutionRepresentationKind")


@dataclass(frozen=True)
class AlgorithmDescriptor:
    id: str
    name: str
    description: str
    representation_capabilities: Tuple[RepresentationCapability, ...]
    supported_variable_types: Tuple[VariableType, ...]
    supported_problem_families: Tuple[ProblemFamily, ...]
    supported_mathematical_properties: Tuple[MathematicalProperty, ...]
    supported_objectives: Tuple[ObjectiveSense, ...]
    supports_constraints: bool
    supports_multiobjective: bool
    required_operators: Tuple[str, ...]
    optional_operators: Tuple[str, ...]
    required_adapters: Tuple[str, ...]
    limitations: Tuple[str, ...]
    implementation_class: Optional[Type[OptimizationAlgorithm]] = None
    availability: AlgorithmAvailability = AlgorithmAvailability.AVAILABLE
    required_mathematical_properties: Tuple[MathematicalProperty, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.id or not self.name or not self.description:
            raise ValueError("id, name and description are required")
        if not self.representation_capabilities:
            raise ValueError("at least one representation capability is required")
        if not self.supported_variable_types:
            raise ValueError("supported_variable_types cannot be empty")
        if not self.supported_problem_families:
            raise ValueError("supported_problem_families cannot be empty")
        if not self.supported_mathematical_properties:
            raise ValueError("supported_mathematical_properties cannot be empty")
        if not self.supported_objectives:
            raise ValueError("supported_objectives cannot be empty")
        if not isinstance(self.availability, AlgorithmAvailability):
            raise ValueError("availability must be an AlgorithmAvailability")

        normalized_objectives = []
        for objective in self.supported_objectives:
            if isinstance(objective, ObjectiveSense):
                normalized_objectives.append(objective)
            elif isinstance(objective, str):
                try:
                    normalized_objectives.append(ObjectiveSense(objective))
                except ValueError as exc:
                    raise ValueError(f"unsupported objective sense '{objective}'") from exc
            else:
                raise ValueError("supported_objectives must contain ObjectiveSense values")
        object.__setattr__(self, "supported_objectives", tuple(normalized_objectives))

        normalized_implementation_class = self._normalize_implementation_class(self.implementation_class)
        if normalized_implementation_class is None and self.availability == AlgorithmAvailability.AVAILABLE:
            raise ValueError("available algorithms must declare a concrete implementation class")
        if isinstance(self.implementation_class, str) and self.availability != AlgorithmAvailability.AVAILABLE:
            object.__setattr__(self, "implementation_class", None)
        else:
            object.__setattr__(self, "implementation_class", normalized_implementation_class)

    def _normalize_implementation_class(self, implementation_class: Optional[Union[Type[OptimizationAlgorithm], str]]) -> Optional[Type[OptimizationAlgorithm]]:
        if implementation_class is None:
            return None
        if isinstance(implementation_class, str):
            class_name = implementation_class
            resolved = _IMPLEMENTATION_LOOKUP.get(class_name)
            if resolved is None:
                raise ValueError(f"unknown implementation class '{class_name}'")
            return resolved
        if isinstance(implementation_class, type) and issubclass(implementation_class, OptimizationAlgorithm):
            return implementation_class
        raise ValueError("implementation_class must be an OptimizationAlgorithm subclass or None")

    def get_capability(self, representation: SolutionRepresentationKind) -> Optional[RepresentationCapability]:
        for capability in self.representation_capabilities:
            if capability.representation == representation:
                return capability
        return None

    @property
    def implementation_name(self) -> Optional[str]:
        if self.implementation_class is None:
            return None
        return self.implementation_class.__name__


class AlgorithmRegistry:
    def __init__(self) -> None:
        self._descriptors: Dict[str, AlgorithmDescriptor] = {}

    def register(self, descriptor: AlgorithmDescriptor) -> None:
        if descriptor.id in self._descriptors:
            raise ValueError(f"Algorithm id '{descriptor.id}' already registered")
        if not self.validate_descriptor(descriptor):
            raise ValueError(f"Descriptor '{descriptor.id}' is structurally invalid")
        self._descriptors[descriptor.id] = descriptor

    def validate_descriptor(self, descriptor: AlgorithmDescriptor) -> bool:
        try:
            descriptor.__post_init__()
        except ValueError:
            return False
        if descriptor.availability == AlgorithmAvailability.AVAILABLE and descriptor.implementation_class is None:
            return False
        return True

    @staticmethod
    def validate_implementation(implementation_class: Type[OptimizationAlgorithm]) -> bool:
        try:
            if not isinstance(implementation_class, type) or not issubclass(implementation_class, OptimizationAlgorithm):
                return False
            instance = implementation_class()
            return callable(getattr(instance, "configure", None)) and callable(getattr(instance, "optimize", None)) and callable(getattr(instance, "get_params_report", None))
        except Exception:
            return False

    def validate(self) -> List[str]:
        errors: List[str] = []
        seen_ids: Set[str] = set()
        for descriptor in self._descriptors.values():
            if descriptor.id in seen_ids:
                errors.append(f"Duplicate algorithm id '{descriptor.id}'")
            seen_ids.add(descriptor.id)
            if not self.validate_descriptor(descriptor):
                errors.append(f"Descriptor '{descriptor.id}' is invalid")
        return errors

    def get(self, algorithm_id: str) -> AlgorithmDescriptor:
        try:
            return self._descriptors[algorithm_id]
        except KeyError as exc:
            raise KeyError(f"Unknown algorithm id '{algorithm_id}'") from exc

    def get_all(self) -> List[AlgorithmDescriptor]:
        return list(self._descriptors.values())

    def get_all_ids(self) -> List[str]:
        return sorted(self._descriptors.keys())

    def find_by_representation(self, representation: SolutionRepresentationKind) -> List[AlgorithmDescriptor]:
        return [descriptor for descriptor in self._descriptors.values() if descriptor.get_capability(representation) is not None]

    def find_by_problem_family(self, problem_family: ProblemFamily) -> List[AlgorithmDescriptor]:
        return [descriptor for descriptor in self._descriptors.values() if problem_family in descriptor.supported_problem_families]

    @classmethod
    def from_builtin_algorithms(cls) -> "AlgorithmRegistry":
        registry = cls()
        registry.register(
            AlgorithmDescriptor(
                id="ga",
                name="Genetic Algorithm",
                description="Population-based evolutionary search implemented through a PyGAD-backed numeric optimizer.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=("crossover", "mutation", "selection"),
                        required_adapters=(),
                        notes="The current implementation uses numeric vectors and a standard GA loop.",
                    ),
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.PERMUTATION,
                        status="unsupported",
                        required_operators=("permutation_crossover", "swap_mutation"),
                        required_adapters=(),
                        notes="No permutation-specific crossover or mutation operators are implemented in the current code.",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS, VariableType.INTEGER, VariableType.BINARY),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.FEATURE_SELECTION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.DISCRETE, MathematicalProperty.BINARY, MathematicalProperty.CONSTRAINED, MathematicalProperty.NONLINEAR),
                supported_objectives=("minimize", "maximize"),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=("crossover", "mutation", "selection"),
                optional_operators=("elitism",),
                required_adapters=(),
                limitations=("No dedicated permutation operators; no explicit multi-objective support.",),
                implementation_class="GeneticAlgorithm",
                availability=AlgorithmAvailability.AVAILABLE,
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="pso",
                name="Particle Swarm Optimization",
                description="Swarm-based optimizer over numeric vectors.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=("velocity_update", "position_update"),
                        required_adapters=(),
                        notes="The implementation uses continuous-valued positions and velocities directly.",
                    ),
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.PERMUTATION,
                        status="unsupported",
                        required_operators=("permutation_velocity",),
                        required_adapters=(),
                        notes="The current implementation is vector-only and does not define discrete permutation operators.",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS, VariableType.INTEGER),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.CONSTRAINED, MathematicalProperty.NONLINEAR),
                supported_objectives=("minimize", "maximize"),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=("velocity_update", "position_update"),
                optional_operators=(),
                required_adapters=(),
                limitations=("Not a permutation-based optimizer; uses numeric vectors only.",),
                implementation_class="ParticleSwarmOptimization",
                availability=AlgorithmAvailability.AVAILABLE,
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="de",
                name="Differential Evolution",
                description="Population-based numeric optimizer using differential mutation and crossover.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=("mutation", "crossover"),
                        required_adapters=(),
                        notes="The implementation operates on numeric vectors with clipping and crossover on real-valued genes.",
                    ),
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.PERMUTATION,
                        status="unsupported",
                        required_operators=("permutation_crossover",),
                        required_adapters=(),
                        notes="No permutation-specific mutation or crossover is implemented.",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS, VariableType.INTEGER),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.CONSTRAINED, MathematicalProperty.NONLINEAR),
                supported_objectives=("minimize", "maximize"),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=("mutation", "crossover"),
                optional_operators=(),
                required_adapters=(),
                limitations=("No real permutation support; relies on numeric vector arithmetic.",),
                implementation_class="DifferentialEvolution",
                availability=AlgorithmAvailability.AVAILABLE,
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="bfo",
                name="Bacterial Foraging Optimization",
                description="Chemotaxis-driven numeric optimizer for continuous search spaces.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=("chemotaxis", "reproduction"),
                        required_adapters=(),
                        notes="The implementation uses floating-point vectors and numerical movement steps.",
                    ),
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.PERMUTATION,
                        status="unsupported",
                        required_operators=("permutation_move",),
                        required_adapters=(),
                        notes="There is no permutation neighborhood or representation adapter.",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS, VariableType.INTEGER),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.CONSTRAINED, MathematicalProperty.NONLINEAR),
                supported_objectives=("minimize", "maximize"),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=("chemotaxis", "reproduction"),
                optional_operators=(),
                required_adapters=(),
                limitations=("No discrete or permutation representation support.",),
                implementation_class="BacterialForagingOptimization",
                availability=AlgorithmAvailability.AVAILABLE,
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="sa",
                name="Simulated Annealing",
                description="Single-solution local search with a generic numeric neighborhood.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=("neighborhood", "acceptance_criterion"),
                        required_adapters=(),
                        notes="The current implementation perturbs numeric vectors and evaluates them directly.",
                    ),
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.PERMUTATION,
                        status="unsupported",
                        required_operators=("swap_neighbor",),
                        required_adapters=(),
                        notes="There is no permutation-specific neighborhood logic in the current implementation.",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS, VariableType.INTEGER),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.DISCRETE, MathematicalProperty.CONSTRAINED, MathematicalProperty.NONLINEAR),
                supported_objectives=("minimize", "maximize"),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=("neighborhood", "acceptance_criterion"),
                optional_operators=(),
                required_adapters=(),
                limitations=("Neighborhood is numeric-only; no generic permutation support.",),
                implementation_class="SimulatedAnnealing",
                availability=AlgorithmAvailability.AVAILABLE,
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="aco",
                name="Ant Colony Optimization",
                description="Pheromone-based search implemented as a numeric heuristic over bounds.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=("pheromone", "construction"),
                        required_adapters=(),
                        notes="The current implementation randomly samples values within numeric bounds.",
                    ),
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.GRAPH,
                        status="unsupported",
                        required_operators=("graph_construction",),
                        required_adapters=(),
                        notes="There is no graph structure or graph-based move generation implemented in the current code.",
                    ),
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.PERMUTATION,
                        status="unsupported",
                        required_operators=("path_construction",),
                        required_adapters=(),
                        notes="The current implementation does not build an explicit permutation path representation.",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS, VariableType.INTEGER),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.CONSTRAINED, MathematicalProperty.NONLINEAR),
                supported_objectives=("minimize", "maximize"),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=("pheromone", "construction"),
                optional_operators=(),
                required_adapters=(),
                limitations=("No true graph or permutation representation support; current implementation is still a numeric heuristic.",),
                implementation_class="AntColonyOptimization",
                availability=AlgorithmAvailability.AVAILABLE,
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="tabu",
                name="Tabu Search",
                description="Tabu-based local improvement over numeric vectors.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=("neighborhood", "tabu_list"),
                        required_adapters=(),
                        notes="The implementation generates a numeric neighborhood around the current solution.",
                    ),
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.PERMUTATION,
                        status="unsupported",
                        required_operators=("swap_neighbor",),
                        required_adapters=(),
                        notes="No permutation-specific neighborhood or move operators are present.",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS, VariableType.INTEGER),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.DISCRETE, MathematicalProperty.CONSTRAINED, MathematicalProperty.NONLINEAR),
                supported_objectives=("minimize", "maximize"),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=("neighborhood", "tabu_list"),
                optional_operators=(),
                required_adapters=(),
                limitations=("Neighborhood is numeric-only; no adaptive permutation support.",),
                implementation_class="TabuSearch",
                availability=AlgorithmAvailability.AVAILABLE,
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="hill_climbing",
                name="Hill Climbing",
                description="Single-solution local search over numeric vectors.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=("neighbor",),
                        required_adapters=(),
                        notes="The implementation uses Gaussian perturbation of a numeric vector.",
                    ),
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.PERMUTATION,
                        status="unsupported",
                        required_operators=("swap_neighbor",),
                        required_adapters=(),
                        notes="No permutation neighborhood exists in the implementation.",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS, VariableType.INTEGER),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.DISCRETE, MathematicalProperty.CONSTRAINED, MathematicalProperty.NONLINEAR),
                supported_objectives=("minimize", "maximize"),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=("neighbor",),
                optional_operators=(),
                required_adapters=(),
                limitations=("Only numeric vector perturbations are implemented.",),
                implementation_class="HillClimbing",
                availability=AlgorithmAvailability.AVAILABLE,
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="linear_programming",
                name="Linear Programming",
                description="Exact linear programming solver backed by SciPy HiGHS.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=(),
                        required_adapters=(),
                        notes="Translates the structured optimization model to scipy.optimize.linprog (HiGHS).",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS,),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.PRODUCTION_PLANNING, ProblemFamily.RESOURCE_ALLOCATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.LINEAR, MathematicalProperty.CONSTRAINED),
                supported_objectives=(ObjectiveSense.MINIMIZE, ObjectiveSense.MAXIMIZE),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=(),
                optional_operators=(),
                required_adapters=(),
                limitations=("Current backend supports affine expressions and hard linear constraints only.",),
                implementation_class=LinearProgramming,
                availability=AlgorithmAvailability.AVAILABLE,
                required_mathematical_properties=(MathematicalProperty.LINEAR,),
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="integer_programming",
                name="Integer Programming",
                description="Exact mixed-integer linear programming solver backed by SciPy HiGHS.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=(),
                        required_adapters=(),
                        notes="Translates integer/binary/continuous affine models to scipy.optimize.milp (HiGHS).",
                    ),
                ),
                supported_variable_types=(VariableType.CONTINUOUS, VariableType.INTEGER, VariableType.BINARY),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.PRODUCTION_PLANNING, ProblemFamily.RESOURCE_ALLOCATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.CONTINUOUS, MathematicalProperty.INTEGER, MathematicalProperty.BINARY, MathematicalProperty.DISCRETE, MathematicalProperty.MIXED_INTEGER, MathematicalProperty.LINEAR, MathematicalProperty.CONSTRAINED),
                supported_objectives=(ObjectiveSense.MINIMIZE, ObjectiveSense.MAXIMIZE),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=(),
                optional_operators=(),
                required_adapters=(),
                limitations=("Current backend supports affine expressions and hard linear constraints only; multiobjective execution is not implemented.",),
                implementation_class=IntegerProgramming,
                availability=AlgorithmAvailability.AVAILABLE,
                required_mathematical_properties=(MathematicalProperty.LINEAR, MathematicalProperty.INTEGER),
            )
        )
        registry.register(
            AlgorithmDescriptor(
                id="constraint_programming",
                name="Constraint Programming",
                description="Exact integer constraint solver backed by SciPy HiGHS mixed-integer optimization.",
                representation_capabilities=(
                    RepresentationCapability(
                        representation=SolutionRepresentationKind.VECTOR,
                        status="supported",
                        required_operators=(),
                        required_adapters=(),
                        notes="Uses an exact mixed-integer backend for bounded linear integer/binary constraint models.",
                    ),
                ),
                supported_variable_types=(VariableType.INTEGER, VariableType.BINARY),
                supported_problem_families=(ProblemFamily.CONTINUOUS_OPTIMIZATION, ProblemFamily.PRODUCTION_PLANNING, ProblemFamily.RESOURCE_ALLOCATION, ProblemFamily.GENERIC),
                supported_mathematical_properties=(MathematicalProperty.INTEGER, MathematicalProperty.BINARY, MathematicalProperty.DISCRETE, MathematicalProperty.CONSTRAINED, MathematicalProperty.LINEAR),
                supported_objectives=(ObjectiveSense.MINIMIZE, ObjectiveSense.MAXIMIZE),
                supports_constraints=True,
                supports_multiobjective=False,
                required_operators=(),
                optional_operators=(),
                required_adapters=(),
                limitations=("This CP-compatible backend currently targets bounded linear integer/binary models; CP-SAT is not bundled in the runtime image.",),
                implementation_class=ConstraintProgramming,
                availability=AlgorithmAvailability.AVAILABLE,
                required_mathematical_properties=(MathematicalProperty.CONSTRAINED, MathematicalProperty.INTEGER),
            )
        )
        return registry
