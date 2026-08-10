from schemas import ModelingRequest, ModelingResponse, SolveRequest, SolveResponse, SolutionResult
from services.llm_service import GroqLLMService
from services.selector import AlgorithmSelector
from algorithms.ga import GeneticAlgorithm
from algorithms.pso import ParticleSwarmOptimization
from algorithms.de import DifferentialEvolution
from algorithms.bfo import BacterialForagingOptimization
from algorithms.sa import SimulatedAnnealing
from algorithms.aco import AntColonyOptimization
from algorithms.tabu import TabuSearch
from algorithms.hill_climbing import HillClimbing
from algorithms.classical import LinearProgramming, IntegerProgramming, ConstraintProgramming

class OptimizationOrchestrator:
    def __init__(self):
        self.llm_service = GroqLLMService()
        self.selector = AlgorithmSelector()
        
        # Registro expansível
        self.algorithm_registry = {
            "GA": GeneticAlgorithm,
            "PSO": ParticleSwarmOptimization,
            "DE": DifferentialEvolution,
            "BFO": BacterialForagingOptimization,
            "SA": SimulatedAnnealing,
            "ACO": AntColonyOptimization,
            "TABU": TabuSearch,
            "HILL": HillClimbing,
            "LP": LinearProgramming,
            "IP": IntegerProgramming,
            "CP": ConstraintProgramming,
        }

    def generate_model(self, request: ModelingRequest) -> ModelingResponse:
        llm_output = self.llm_service.draft_model(request.problem_description, request.data)
        
        return ModelingResponse(
            explanation=llm_output["explanation"],
            problem_type=llm_output["problem_type"],
            characteristics=llm_output.get("characteristics", []),
            is_minimization=llm_output["is_minimization"],
            generated_code=llm_output["code"],
            bounds=llm_output["bounds"]
        )

    def execute_solution(self, request: SolveRequest) -> SolveResponse:
        # Compila a função de forma isolada
        local_scope = {}
        exec(request.generated_code, {}, local_scope)
        fitness_function = local_scope['fitness_function']

        # Seletor Automático e Identificação dos Algoritmos para rodar
        best_algo_name = self.selector.select_best(request.problem_type, request.characteristics)
        
        algos_to_run = []
        user_choice = request.algorithm_config.name.upper()
        
        if user_choice == "AUTO":
            algos_to_run = [best_algo_name]
        elif user_choice == "ALL":
            algos_to_run = list(self.algorithm_registry.keys())
        else:
            algos_to_run = [user_choice]

        results = []
        
        # Executa os algoritmos e alimenta o "Compare Solutions"
        for alg_name in algos_to_run:
            if alg_name not in self.algorithm_registry:
                continue
                
            alg_instance = self.algorithm_registry[alg_name]()
            alg_instance.configure(request.algorithm_config)
            
            # Passamos is_minimization para o algoritmo saber se inverte o fitness ou não
            best_sol, best_fit = alg_instance.optimize(
                fitness_function, 
                request.bounds, 
                request.is_minimization
            )
            
            results.append(SolutionResult(
                algorithm_used=alg_name,
                best_solution=best_sol,
                best_fitness=best_fit,
                parameters_used=alg_instance.get_params_report()
            ))

        # Ordena os resultados para mostrar o melhor primeiro (baseado se é minimização)
        results.sort(key=lambda x: x.best_fitness, reverse=not request.is_minimization)

        return SolveResponse(
            recommended_algorithm=best_algo_name,
            results=results
        )