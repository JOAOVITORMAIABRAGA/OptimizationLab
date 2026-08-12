from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GraphClassification:
    kind: str
    confidence: float


class GraphProblemClassifier:
    """Classifies common graph problem semantics from normalized language.

    This is intentionally isolated from the LLM transport and compatibility
    layers. Adding a graph family means adding one rule here, not spreading
    problem-specific branches across solvers or compatibility code.
    """

    _RULES = (
        ("chinese_postman", 0.99, ("carteiro chines", "chinese postman", "postman problem", "postman")),
        ("shortest_path", 0.98, ("menor caminho", "caminho minimo", "shortest path", "menor rota entre", "rota mais curta")),
        ("minimum_spanning_tree", 0.99, ("arvore geradora", "minimum spanning tree", "minimum spanning", "conectar todos os pontos", "conectar todos os nos", "todos os pontos precisam ficar conectados", "todos os nos precisam ficar conectados")),
        ("tsp", 0.99, ("caixeiro viajante", "traveling salesman", "travelling salesman", "tsp", "ciclo hamiltoniano", "circuito hamiltoniano", "hamiltonian cycle", "hamiltonian circuit")),
    )

    @staticmethod
    def normalize(text: str) -> str:
        value = unicodedata.normalize("NFKD", text or "")
        value = "".join(char for char in value if not unicodedata.combining(char))
        value = re.sub(r"[^a-z0-9]+", " ", value.lower())
        return re.sub(r"\s+", " ", value).strip()

    def classify(self, *texts: str) -> GraphClassification:
        normalized = self.normalize(" ".join(texts))
        best = GraphClassification("generic", 0.0)

        for kind, confidence, phrases in self._RULES:
            if any(phrase in normalized for phrase in phrases):
                if confidence > best.confidence:
                    best = GraphClassification(kind, confidence)

        # TSP descriptions frequently express the semantics without naming
        # the problem. Detect the invariant rather than a particular sentence.
        if (
            ("visitar todas" in normalized or "visitar cada" in normalized or "visita todas" in normalized or "visits each city" in normalized or "visits every city" in normalized or "visit each city" in normalized or "visit every city" in normalized)
            and ("uma vez" in normalized or "once" in normalized)
            and any(token in normalized for token in ("voltar", "retornar", "volta", "origem", "inicio", "primeira cidade", "returns", "return to", "returning"))
        ):
            return GraphClassification("tsp", 0.97)

        # MST descriptions often describe the optimization objective rather
        # than naming the algorithm: connect all vertices with minimum total
        # cable/cost/weight.
        if (
            any(token in normalized for token in ("conectar todos", "ligar todos", "todos os pontos", "todos os nos"))
            and any(token in normalized for token in ("minimo", "menor", "minimizar", "gastando o minimo", "menor possivel"))
            and any(token in normalized for token in ("cabo", "custo", "peso", "comprimento"))
        ):
            return GraphClassification("minimum_spanning_tree", 0.96)

        return best
