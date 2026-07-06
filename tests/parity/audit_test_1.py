import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.datasets.csv_loader import load_csv_dataset as load_champion_data
from src.evaluation.parallel_evaluator import evaluate_batch
import src.evaluation.parallel_evaluator as pe
from src.algorithms.algorithms import GENES_PER_CHAMP

def run_test_1():
    base_data = load_champion_data("riot_datadragon_champion_stats.csv")
    dim = len(base_data) * GENES_PER_CHAMP
    rng = random.Random(42)
    
    batch_size = 50
    candidates = [[rng.uniform(0.7, 1.3) for _ in range(dim)] for _ in range(batch_size)]
    
    print("--- Running CPU sequential mode ---")
    pe.USE_GPU = False
    cpu_results = evaluate_batch(base_data, candidates, mode="continuous", workers=1)
    
    print("--- Running GPU tensor mode ---")
    pe.USE_GPU = True
    gpu_results = evaluate_batch(base_data, candidates, mode="continuous", workers=1)
    
    metrics = ["fitness", "rbi", "mds"]
    print("\n--- Comparison Results ---")
    
    for metric in metrics:
        cpu_vals = np.array([res[metric] for res in cpu_results])
        gpu_vals = np.array([res[metric] for res in gpu_results])
        diff = np.abs(cpu_vals - gpu_vals)
        
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # 1e-4 tolerance due to PyTorch float32 accumulations vs Python double precision floats
        passed = max_diff < 1e-4 
        status = "PASS" if passed else "FAIL"
        
        print(f"Metric: {metric}")
        print(f"  Max Abs Diff:  {max_diff:.8f}")
        print(f"  Mean Abs Diff: {mean_diff:.8f}")
        print(f"  Status:        {status}")

if __name__ == "__main__":
    run_test_1()
