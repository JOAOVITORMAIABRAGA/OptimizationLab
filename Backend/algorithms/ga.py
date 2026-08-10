import numpy as np
from typing import Callable, Tuple, List, Dict, Any
from .base import OptimizationAlgorithm
from schemas import AlgorithmConfig

try:
    import pygad  # type: ignore
except ImportError:  # pragma: no cover - depende do ambiente
    pygad = None

class GeneticAlgorithm(OptimizationAlgorithm):
    
    def __init__(self):
        self.pop_size: int = 50
        self.generations: int = 100
        self.mutation_rate: float = 0.05

    def configure(self, config: AlgorithmConfig) -> None:
        if config is not None and config.ga_params:
            self.pop_size = config.ga_params.pop_size
            self.generations = config.ga_params.generations
            self.mutation_rate = config.ga_params.mutation_rate

    def optimize(
        self, 
        fitness_function: Callable, 
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None
    ) -> Tuple[List[float], float]:
        
        num_genes = len(bounds)
        gene_space = [{'low': b[0], 'high': b[1]} for b in bounds]
        
        def pygad_fitness_wrapper(ga_instance, solution, solution_idx):
            # 1. Checagem Nativa de Restrições
            if constraints:
                for constraint_func in constraints:
                    if not constraint_func(solution):
                        # Se violar, no PyGAD podemos forçar o fitness a ser infinitamente ruim
                        # Em implementações mais profundas, descartamos o indivíduo antes
                        return -float('inf') if not is_minimization else float('inf')

            # 2. Avaliação Normal
            fit_value = fitness_function(solution)
            
            # O PyGAD sempre MAXIMIZA por padrão.
            # Se o usuário quer MINIMIZAR, invertemos o sinal do retorno.
            if is_minimization:
                return -fit_value 
            return fit_value

        if pygad is None:
            rng = np.random.default_rng(7)
            low = np.array([b[0] for b in bounds], dtype=float)
            high = np.array([b[1] for b in bounds], dtype=float)
            best_solution = np.array([0.5 * (l + h) for l, h in bounds], dtype=float)
            best_fitness = float('inf') if is_minimization else float('-inf')

            for _ in range(max(20, self.generations)):
                candidate = rng.uniform(low, high, size=num_genes)
                if constraints:
                    feasible = True
                    for constraint_func in constraints:
                        if not constraint_func(candidate.tolist()):
                            feasible = False
                            break
                    if not feasible:
                        continue
                fit_value = fitness_function(candidate.tolist())
                if not np.isfinite(fit_value):
                    continue
                if is_minimization:
                    if fit_value < best_fitness:
                        best_solution = candidate
                        best_fitness = fit_value
                else:
                    if fit_value > best_fitness:
                        best_solution = candidate
                        best_fitness = fit_value

            return best_solution.tolist(), float(best_fitness)

        ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=max(2, int(self.pop_size * 0.5)),
            fitness_func=pygad_fitness_wrapper,
            sol_per_pop=self.pop_size,
            num_genes=num_genes,
            gene_space=gene_space,
            mutation_percent_genes=self.mutation_rate * 100,
            mutation_type="random",
            crossover_type="single_point",
            parent_selection_type="rws",
            keep_elitism=1
        )

        ga_instance.run()
        best_solution, best_fitness, _ = ga_instance.best_solution()

        # Desinverte o fitness caso seja minimização
        final_fitness = -best_fitness if is_minimization else best_fitness

        return best_solution.tolist(), float(final_fitness)

    def get_params_report(self) -> Dict[str, Any]:
        return {
            "pop_size": self.pop_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "engine": "PyGAD"
        }