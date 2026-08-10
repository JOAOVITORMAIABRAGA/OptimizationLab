from abc import ABC, abstractmethod
from typing import Callable, Tuple, List, Any, Dict
from schemas import AlgorithmConfig

class OptimizationAlgorithm(ABC):
    
    @abstractmethod
    def configure(self, config: AlgorithmConfig) -> None:
        pass

    @abstractmethod
    def optimize(
        self, 
        fitness_function: Callable, 
        bounds: List[Tuple[float, float]],
        is_minimization: bool = True,
        constraints: List[Callable] = None # <-- Novo parâmetro opcional
    ) -> Tuple[List[float], float]:
        """
        Executa a otimização matemática.
        :param constraints: Lista de funções que avaliam se a solução é viável (True/False).
        """
        pass

    @abstractmethod
    def get_params_report(self) -> Dict[str, Any]:
        pass