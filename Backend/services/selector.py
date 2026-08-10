from typing import List
from schemas import ProblemCharacteristic, ProblemType


class AlgorithmSelector:
    def select_best(self, problem_type: ProblemType, characteristics: List[ProblemCharacteristic]) -> str:
        if problem_type == ProblemType.COMBINATORIAL:
            return "TABU"

        if ProblemCharacteristic.CONSTRAINED in characteristics:
            return "DE"

        if ProblemCharacteristic.BLACK_BOX in characteristics:
            return "PSO"

        if ProblemType.LP == problem_type:
            return "LP"

        if ProblemType.NLP == problem_type:
            return "GA"

        return "GA"
