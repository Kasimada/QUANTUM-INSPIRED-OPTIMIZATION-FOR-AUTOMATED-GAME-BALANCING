import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.datasets.csv_loader import load_csv_dataset as load_champion_data
from src.evaluation.parallel_evaluator import evaluate_batch
import src.evaluation.parallel_evaluator as pe
from src.algorithms.algorithms import GENES_PER_CHAMP
from src.algorithms.discrete_algorithms import (
    repair_indices,
    patch_pressure_from_indices,
    indices_to_multipliers
)

def run_test_2():
    base_data = load_champion_data("riot_datadragon_champion_stats.csv")
    dim = len(base_data) * GENES_PER_CHAMP
    rng = random.Random(42)
    
    batch_size = 50
    # Generate random integer indices between 0 and 4
    candidates = [[rng.randint(0, 4) for _ in range(dim)] for _ in range(batch_size)]
    
    print("--- Testing Helper Functions ---")
    helpers_passed = True
    repaired_candidates = []
    
    for i, ind in enumerate(candidates):
        repaired = repair_indices(ind)
        repaired_candidates.append(repaired)
        
        pressure = patch_pressure_from_indices(repaired)
        if abs(pressure) > 1:
            print(f"FAIL: repair_indices left pressure = {pressure} for candidate {i}")
            helpers_passed = False
            
        mults = indices_to_multipliers(repaired)
        if len(mults) != dim:
            helpers_passed = False
            
    if helpers_passed:
        print("PASS: repair_indices and index conversions work correctly.")
    else:
        print("FAIL: helper functions failed.")
        
    print("\n--- Running CPU sequential discrete mode ---")
    pe.USE_GPU = False
    cpu_results = evaluate_batch(base_data, repaired_candidates, mode="discrete", workers=1)
    
    print("--- Running GPU tensor discrete mode ---")
    pe.USE_GPU = True
    gpu_results = evaluate_batch(base_data, repaired_candidates, mode="discrete", workers=1)
    
    metrics = ["fitness", "rbi", "mds", "pressure", "violation", "magnitude"]
    print("\n--- Comparison Results ---")
    
    for metric in metrics:
        cpu_vals = np.array([res[metric] for res in cpu_results])
        gpu_vals = np.array([res[metric] for res in gpu_results])
        diff = np.abs(cpu_vals - gpu_vals)
        
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        passed = max_diff < 1e-4 
        status = "PASS" if passed else "FAIL"
        
        print(f"Metric: {metric}")
        print(f"  Max Abs Diff:  {max_diff:.8f}")
        print(f"  Mean Abs Diff: {mean_diff:.8f}")
        print(f"  Status:        {status}")

if __name__ == "__main__":
    run_test_2()
