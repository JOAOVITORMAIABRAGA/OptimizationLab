from pydantic import BaseModel, Field
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

# --- 1. Novos Enums baseados no seu diagrama ---
class ProblemType(str, Enum):
    LP = "LP"
    NLP = "NLP"
    COMBINATORIAL = "Combinatorial"

class ProblemCharacteristic(str, Enum):
    MIXED = "Mixed"
    CONSTRAINED = "Constrained"
    BLACK_BOX = "Black-box"

# --- 2. Parâmetros Específicos de Cada Algoritmo ---
class GAParameters(BaseModel):
    pop_size: int = Field(50, description="Tamanho da população.")
    generations: int = Field(100, description="Número de gerações.")
    mutation_rate: float = Field(0.05, description="Taxa de mutação.")

class PSOParameters(BaseModel):
    swarm_size: int = Field(30, description="Quantidade de partículas.")
    iterations: int = Field(100, description="Número de iterações.")

class DEParameters(BaseModel):
    population_size: int = Field(40, description="Tamanho da população.")
    iterations: int = Field(80, description="Número de iterações.")
    mutation_factor: float = Field(0.8, description="Fator de mutação.")
    crossover_rate: float = Field(0.7, description="Taxa de crossover.")

class BFOParameters(BaseModel):
    bacteria_count: int = Field(20, description="Número de bactérias.")
    chemotaxis_steps: int = Field(10, description="Passos de quimiotaxia.")
    reproduction_steps: int = Field(5, description="Passos de reprodução.")
    step_size: float = Field(0.25, description="Tamanho do passo.")

class SAParameters(BaseModel):
    iterations: int = Field(120, description="Número de iterações.")
    initial_temperature: float = Field(10.0, description="Temperatura inicial.")
    cooling_rate: float = Field(0.95, description="Taxa de resfriamento.")
    step_size: float = Field(0.5, description="Tamanho do passo.")

class ACOParameters(BaseModel):
    ants: int = Field(30, description="Número de formigas.")
    iterations: int = Field(60, description="Número de iterações.")
    alpha: float = Field(1.0, description="Peso do feromônio.")
    beta: float = Field(2.0, description="Peso da heurística.")
    rho: float = Field(0.5, description="Taxa de evaporação.")

class TabuParameters(BaseModel):
    iterations: int = Field(80, description="Número de iterações.")
    tabu_size: int = Field(10, description="Tamanho da lista tabu.")
    neighborhood_size: int = Field(5, description="Tamanho da vizinhança.")

class HillClimbingParameters(BaseModel):
    iterations: int = Field(100, description="Número de iterações.")
    step_size: float = Field(0.2, description="Tamanho do passo.")

class AlgorithmConfig:
    def __init__(
        self,
        name: str = "AUTO",
        ga_params: Optional[GAParameters] = None,
        pso_params: Optional[PSOParameters] = None,
        de_params: Optional[DEParameters] = None,
        bfo_params: Optional[BFOParameters] = None,
        sa_params: Optional[SAParameters] = None,
        aco_params: Optional[ACOParameters] = None,
        tabu_params: Optional[TabuParameters] = None,
        hill_climbing_params: Optional[HillClimbingParameters] = None,
    ):
        self.name = name
        self.ga_params = ga_params
        self.pso_params = pso_params
        self.de_params = de_params
        self.bfo_params = bfo_params
        self.sa_params = sa_params
        self.aco_params = aco_params
        self.tabu_params = tabu_params
        self.hill_climbing_params = hill_climbing_params

# --- 3. Requisições e Respostas da Etapa 1 (Modelagem) ---
class ModelingRequest:
    def __init__(self, problem_description: str, data: Optional[Dict[str, Any]] = None):
        self.problem_description = problem_description
        self.data = data or {}

class ModelingResponse:
    def __init__(self, explanation: str, problem_type: ProblemType, characteristics: Optional[List[ProblemCharacteristic]] = None, generated_code: str = "", bounds: Optional[List[Tuple[float, float]]] = None, is_minimization: bool = True):
        self.explanation = explanation
        self.problem_type = problem_type
        self.characteristics = characteristics or []
        self.generated_code = generated_code
        self.bounds = bounds or []
        self.is_minimization = is_minimization

# --- 4. Requisições e Respostas da Etapa 2 (Resolução) ---
class SolveRequest:
    def __init__(self, generated_code: str, bounds: List[Tuple[float, float]], is_minimization: bool, problem_type: ProblemType, characteristics: List[ProblemCharacteristic], algorithm_config: Optional[AlgorithmConfig] = None):
        self.generated_code = generated_code
        self.bounds = bounds
        self.is_minimization = is_minimization
        self.problem_type = problem_type
        self.characteristics = characteristics
        self.algorithm_config = algorithm_config or AlgorithmConfig()

class SolutionResult:
    def __init__(self, algorithm_used: str, best_solution: List[float], best_fitness: float, parameters_used: Dict[str, Any]):
        self.algorithm_used = algorithm_used
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.parameters_used = parameters_used

class SolveResponse:
    def __init__(self, recommended_algorithm: str, results: List[SolutionResult]):
        self.recommended_algorithm = recommended_algorithm
        self.results = results