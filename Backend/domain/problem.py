from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from .expressions import StructuredExpression
from .objectives import ObjectiveKind, ObjectiveSense, ObjectiveSpec
from .problem_family import MathematicalProperty, ProblemFamily
from .representations import SolutionRepresentationKind
from .variables import VariableType


@dataclass
class DomainSpec:
    kind: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    values: Optional[List[Any]] = None
    categories: Optional[List[str]] = None
    elements: Optional[List[Any]] = None


@dataclass
class VariableSpec:
    name: str
    variable_type: VariableType
    domain: Optional[DomainSpec]
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    categories: Optional[List[str]] = None
    values: Optional[List[Any]] = None
    required: bool = True
    description: Optional[str] = None


@dataclass
class ConstraintSpec:
    id: str
    name: str
    kind: str
    relation: str
    expression: Optional[StructuredExpression] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    threshold: Optional[float] = None
    weight: Optional[float] = None
    description: Optional[str] = None


@dataclass
class DatasetSpec:
    source: str
    format: str
    path: Optional[str] = None
    data: Optional[Any] = None
    schema: Optional[Dict[str, Any]] = None


@dataclass
class SolutionRepresentationSpec:
    kind: SolutionRepresentationKind
    name: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationProblem:
    name: str
    objective: ObjectiveSpec
    variables: List[VariableSpec]
    constraints: List[ConstraintSpec] = field(default_factory=list)
    problem_family: ProblemFamily = ProblemFamily.GENERIC
    mathematical_properties: Set[MathematicalProperty] = field(default_factory=set)
    solution_representation: Optional[SolutionRepresentationSpec] = None
    dataset: Optional[DatasetSpec] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    id: str = field(default_factory=lambda: uuid4().hex)
