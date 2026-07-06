from src.algorithms.continuous_algorithms import (
    AQEAOptimizer, QEAOptimizer,
    GeneticOptimizer,
    NSGA2Optimizer,
    PSOOptimizer,
    ContinuousRandomSearchOptimizer
)
from src.algorithms.base import Optimizer

from src.algorithms.discrete_algorithms import (
    DiscreteGeneticOptimizer,
    DiscretePSOOptimizer,
    DiscreteQEAOptimizer,
    DiscreteAQEAOptimizer,
    DiscreteNSGA2Optimizer,
    DiscreteAQEAFeasibleOptimizer,
    DiscreteMapElitesOptimizer,
    DiscreteRandomSearchOptimizer
)

REGISTRY = {
    # Continuous Algorithms
    "ga": {"id": "ga", "display_name": "Genetic Algorithm", "optimizer_class": GeneticOptimizer, "mode": "continuous"},
    "pso": {"id": "pso", "display_name": "Particle Swarm Optimization", "optimizer_class": PSOOptimizer, "mode": "continuous"},
    "qea": {"id": "qea", "display_name": "Quantum Evolutionary Algorithm", "optimizer_class": QEAOptimizer, "mode": "continuous"},
    "aqea": {"id": "aqea", "display_name": "Adaptive Quantum Evolutionary Algorithm", "optimizer_class": AQEAOptimizer, "mode": "continuous"},
    "nsga_ii": {"id": "nsga_ii", "display_name": "NSGA-II", "optimizer_class": NSGA2Optimizer, "mode": "continuous"},
    "random_search": {"id": "random_search", "display_name": "Random Search", "optimizer_class": ContinuousRandomSearchOptimizer, "mode": "continuous"},
    
    # Discrete Algorithms
    "ga_discrete": {"id": "ga_discrete", "display_name": "Discrete Genetic Algorithm", "optimizer_class": DiscreteGeneticOptimizer, "mode": "discrete"},
    "pso_discrete": {"id": "pso_discrete", "display_name": "Discrete PSO", "optimizer_class": DiscretePSOOptimizer, "mode": "discrete"},
    "qea_discrete": {"id": "qea_discrete", "display_name": "Discrete QEA", "optimizer_class": DiscreteQEAOptimizer, "mode": "discrete"},
    "aqea_discrete": {"id": "aqea_discrete", "display_name": "Discrete AQEA", "optimizer_class": DiscreteAQEAOptimizer, "mode": "discrete"},
    "nsga_ii_discrete": {"id": "nsga_ii_discrete", "display_name": "Discrete NSGA-II", "optimizer_class": DiscreteNSGA2Optimizer, "mode": "discrete"},
    "random_search_discrete": {"id": "random_search_discrete", "display_name": "Discrete Random Search", "optimizer_class": DiscreteRandomSearchOptimizer, "mode": "discrete"},
    "aqea_feasible_discrete": {"id": "aqea_feasible_discrete", "display_name": "Discrete AQEA Feasible", "optimizer_class": DiscreteAQEAFeasibleOptimizer, "mode": "discrete"},
    "map_elites_discrete": {"id": "map_elites_discrete", "display_name": "Discrete MAP-Elites", "optimizer_class": DiscreteMapElitesOptimizer, "mode": "discrete"}
}

def create_optimizer(name: str) -> Optimizer:
    """Factory method to instantiate an optimizer by name."""
    name = name.lower()
    
    # Aliases
    if name == "genetic": name = "ga"
    if name == "nsga2": name = "nsga_ii"
    if name == "nsga2_discrete": name = "nsga_ii_discrete"
        
    if name in REGISTRY:
        return REGISTRY[name]["optimizer_class"]()
        
    raise ValueError(f"Unknown optimizer or not yet refactored: {name}")

def get_supported_algorithms(mode: str) -> list:
    """Returns a list of primary supported algorithm keys for the given mode."""
    return [info["id"] for info in REGISTRY.values() if info["mode"] == mode]

def get_algorithm_info(name: str) -> dict:
    """Returns metadata for the algorithm."""
    name = name.lower()
    if name in REGISTRY:
        return REGISTRY[name]
    raise ValueError(f"Unknown algorithm: {name}")
