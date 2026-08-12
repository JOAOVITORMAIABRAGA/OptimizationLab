"""Legacy compatibility facade.

Algorithm selection is no longer encoded as a problem-type switch. New code
must use RecommendationEngine + an explicit user selection. ``AUTO`` is the
only supported value for callers that still use this legacy API.
"""

from typing import List

from schemas import ProblemCharacteristic, ProblemType


class AlgorithmSelector:
    def select_best(self, problem_type: ProblemType, characteristics: List[ProblemCharacteristic]) -> str:
        """Return the neutral AUTO sentinel instead of hard-coding solver IDs."""
        return "AUTO"
