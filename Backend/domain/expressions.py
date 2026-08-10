from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Union


@dataclass(frozen=True)
class StructuredExpression:
    kind: str
    op: Optional[str] = None
    name: Optional[str] = None
    value: Any = None
    args: Tuple["StructuredExpression", ...] = field(default_factory=tuple)

    def is_literal(self) -> bool:
        return self.kind == "literal"

    def is_variable(self) -> bool:
        return self.kind == "variable"

    def is_unary(self) -> bool:
        return self.kind == "unary"

    def is_binary(self) -> bool:
        return self.kind == "binary"

    def is_function(self) -> bool:
        return self.kind == "function"
