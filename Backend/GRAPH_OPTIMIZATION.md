# Graph Optimization in OptimizationLab

## Objetivo

A representação `GRAPH` permite que problemas de redes sejam descritos como dados sem obrigar os algoritmos a conhecer o formato original do CSV.

```text
Dataset de arestas
        ↓
GraphRepresentationAdapter
        ↓
OptimizationProblem
        ↓
UniversalDecisionEngine
        ↓
solver de grafo exato ou metaheurística compatível
```

## Estrutura mínima

```json
{
  "representation": "graph",
  "representation_metadata": {
    "nodes": ["A", "B", "C"],
    "edges": [
      {"id": "AB", "u": "A", "v": "B", "weight": 10, "required": true}
    ],
    "directed": false,
    "graph_problem_type": "chinese_postman"
  }
}
```

Para tabelas, o LLM reconhece aliases comuns:

- `id`, `edge_id`, `edge`, `arc`
- `u`, `from`, `source`, `origin`
- `v`, `to`, `target`, `destination`
- `weight`, `cost`, `distance`, `length`

Nenhum peso é inventado pelo completador determinístico.

## Solvers atuais

### Chinese Postman

Implementação exata para o Carteiro Chinês não-direcionado. O solver:

1. soma o custo das arestas obrigatórias;
2. encontra vértices de grau ímpar;
3. calcula menores caminhos entre os ímpares;
4. resolve o matching perfeito mínimo por programação dinâmica;
5. duplica as arestas necessárias;
6. constrói um circuito euleriano.

O matching exato possui crescimento exponencial no número de vértices ímpares. Por isso há um limite configurável (`max_odd_vertices`, padrão 18) para preservar previsibilidade de tempo.

### Shortest Path

Dijkstra nativo para pesos não-negativos.

Metadados adicionais:

```json
{"graph_problem_type": "shortest_path", "source": "A", "target": "D"}
```

### Minimum Spanning Tree

Kruskal nativo para grafos não-direcionados e conexos.

## Princípio de decisão

A seleção automática não executa todos os algoritmos para descobrir qual é melhor. Primeiro usa `problem_family`, `representation`, propriedades matemáticas e `graph_problem_type`. Assim, um PCC vai diretamente para o solver exato quando o modelo permite, evitando gastar avaliações de GA/ACO/PSO sem necessidade.

## Próximas extensões naturais

- TSP com ACO/GA e, para instâncias pequenas, Held-Karp;
- VRP/CVRP;
- Rural Postman;
- graph-aware ACO;
- grafos direcionados;
- múltiplos depósitos e janelas de tempo.
