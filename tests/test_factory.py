import pytest
from src.algorithms.factory import create_optimizer
from src.algorithms.genetic import GeneticOptimizer
from src.algorithms.particle_swarm import PSOOptimizer

def test_create_optimizer():
    ga = create_optimizer("ga")
    assert isinstance(ga, GeneticOptimizer)
    
    pso = create_optimizer("pso")
    assert isinstance(pso, PSOOptimizer)
    
    with pytest.raises(ValueError):
        create_optimizer("unknown_algo")
