from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from domain.representations import SolutionRepresentationKind


@dataclass(frozen=True)
class AdapterDescriptor:
    id: str
    source_representation: SolutionRepresentationKind
    target_representation: SolutionRepresentationKind
    factory: Callable[[Any], Any]
    description: str = ""


@dataclass(frozen=True)
class AdaptationStep:
    adapter_id: str
    source_representation: SolutionRepresentationKind
    target_representation: SolutionRepresentationKind


@dataclass(frozen=True)
class AdaptationPlan:
    source_representation: SolutionRepresentationKind
    target_representation: SolutionRepresentationKind
    steps: Tuple[AdaptationStep, ...] = field(default_factory=tuple)

    @property
    def adapter_ids(self) -> Tuple[str, ...]:
        return tuple(step.adapter_id for step in self.steps)


class AdapterRegistry:
    """Registry and factory boundary for representation adapters."""

    def __init__(self) -> None:
        self._descriptors: Dict[str, AdapterDescriptor] = {}

    def register(self, descriptor: AdapterDescriptor) -> None:
        if descriptor.id in self._descriptors:
            raise ValueError(f"Adapter id '{descriptor.id}' already registered.")
        self._descriptors[descriptor.id] = descriptor

    def get(self, adapter_id: str) -> AdapterDescriptor:
        try:
            return self._descriptors[adapter_id]
        except KeyError as exc:
            raise KeyError(f"Unknown adapter id '{adapter_id}'.") from exc

    def has(self, adapter_id: str) -> bool:
        return adapter_id in self._descriptors

    def create(self, adapter_id: str, problem: Any) -> Any:
        return self.get(adapter_id).factory(problem)

    def find(self, source: SolutionRepresentationKind, target: SolutionRepresentationKind, adapter_ids: Optional[List[str]] = None) -> Optional[AdapterDescriptor]:
        candidates = adapter_ids or list(self._descriptors)
        for adapter_id in candidates:
            descriptor = self._descriptors.get(adapter_id)
            if descriptor and descriptor.source_representation == source and descriptor.target_representation == target:
                return descriptor
        return None

    def all(self) -> List[AdapterDescriptor]:
        return list(self._descriptors.values())
