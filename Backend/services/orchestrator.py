from __future__ import annotations

from schemas import ModelingRequest, SolveRequest
from services.llm_service import GroqLLMService


class OptimizationOrchestrator:
    """Application orchestration facade for declarative problem modeling."""

    def __init__(self) -> None:
        self.llm_service = GroqLLMService()

    def generate_model(self, request: ModelingRequest) -> dict:
        """Return the current declarative OptimizationLab model contract."""
        return self.llm_service.draft_model(
            request.problem_description,
            request.data,
        )

    def execute_solution(self, request: SolveRequest) -> None:
        raise RuntimeError(
            "Legacy generated-code execution is disabled. "
            "Use OptimizationExecutionEngine with a validated OptimizationProblem."
        )
