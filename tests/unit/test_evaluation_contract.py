import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from src.evaluation.backend import CPUBackend, GPUBackend, EvaluationResult
from src.datasets.csv_loader import load_csv_dataset
from src.algorithms.algorithms import GENES_PER_CHAMP
import random

def test_evaluation_contract():
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for contract test")
    except ImportError:
        pytest.skip("Torch not available for contract test")
        
    base_data = load_csv_dataset("data/raw/original_character_stats.csv")
    assert base_data is not None
    
    rng = random.Random(2026)
    num_genes = len(base_data) * GENES_PER_CHAMP
    candidates = [[rng.uniform(0.8, 1.2) for _ in range(num_genes)]]
    
    # Test Continuous Mode
    cpu_backend = CPUBackend()
    gpu_backend = GPUBackend()
    
    cpu_results = cpu_backend.evaluate(base_data, candidates, mode="continuous")
    gpu_results = gpu_backend.evaluate(base_data, candidates, mode="continuous")
    
    assert len(cpu_results) == 1
    assert len(gpu_results) == 1
    
    cpu_res = cpu_results[0]
    gpu_res = gpu_results[0]
    
    assert isinstance(cpu_res, EvaluationResult)
    assert isinstance(gpu_res, EvaluationResult)
    
    assert cpu_res.metrics.keys() == gpu_res.metrics.keys()
    
    # Assert values are close (CPU and GPU math might have tiny float differences)
    assert abs(cpu_res.fitness - gpu_res.fitness) < 1e-4
    for key in cpu_res.metrics:
        assert abs(cpu_res.metrics[key] - gpu_res.metrics[key]) < 1e-4
        
    # Test Discrete Mode
    from src.algorithms.discrete_algorithms import PATCH_LEVELS
    discrete_candidates = [[rng.choice(range(len(PATCH_LEVELS))) for _ in range(num_genes)]]
    
    cpu_disc = cpu_backend.evaluate(base_data, discrete_candidates, mode="discrete")
    gpu_disc = gpu_backend.evaluate(base_data, discrete_candidates, mode="discrete")
    
    assert cpu_disc[0].metrics.keys() == gpu_disc[0].metrics.keys()
    
    print("Contract Test Passed!")

if __name__ == "__main__":
    test_evaluation_contract()
