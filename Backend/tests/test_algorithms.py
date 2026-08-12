import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.ga import GeneticAlgorithm
from algorithms.pso import ParticleSwarmOptimization
from algorithms.de import DifferentialEvolution
from algorithms.bfo import BacterialForagingOptimization
from algorithms.sa import SimulatedAnnealing
from algorithms.aco import AntColonyOptimization
from algorithms.tabu import TabuSearch
from algorithms.hill_climbing import HillClimbing
from algorithms.classical import LinearProgramming, IntegerProgramming, ConstraintProgramming


def test_algorithms_run():
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    def fitness(x):
        return (x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2

    results = []
    for cls in [
        GeneticAlgorithm,
        ParticleSwarmOptimization,
        DifferentialEvolution,
        BacterialForagingOptimization,
        SimulatedAnnealing,
        AntColonyOptimization,
        TabuSearch,
        HillClimbing,
    ]:
        algo = cls()
        algo.configure(None)
        sol, fit = algo.optimize(fitness, bounds, is_minimization=True)
        results.append((cls.__name__, sol, fit))

    assert len(results) == 8
    assert all(sol is not None for _, sol, _ in results)
