import sys
import os
import random
import math

sys.path.insert(0, os.path.abspath("."))
from src.datasets.csv_loader import load_csv_dataset as load_champion_data
from src.evaluation.parallel_evaluator import evaluate_batch
import src.evaluation.parallel_evaluator as pe
from src.algorithms.algorithms import GENES_PER_CHAMP

def run_verification():
    base_data = load_champion_data("riot_datadragon_champion_stats.csv")
    dim = len(base_data) * GENES_PER_CHAMP
    rng = random.Random(20260612)
    
    # 10 test candidates
    candidates = [[rng.uniform(0.7, 1.3) for _ in range(dim)] for _ in range(10)]
    
    print("Running CPU sequential mode...")
    pe.USE_GPU = False
    cpu_results = evaluate_batch(base_data, candidates, mode="continuous", workers=1)
    
    print("Running GPU tensor mode...")
    pe.USE_GPU = True
    gpu_results = evaluate_batch(base_data, candidates, mode="continuous", workers=1)
    
    print("\n--- Correctness Check ---")
    all_match = True
    for i in range(len(candidates)):
        c_fit, c_rbi, c_mds = cpu_results[i]["fitness"], cpu_results[i]["rbi"], cpu_results[i]["mds"]
        g_fit, g_rbi, g_mds = gpu_results[i]["fitness"], gpu_results[i]["rbi"], gpu_results[i]["mds"]
        
        match = math.isclose(c_fit, g_fit, abs_tol=1e-4) and \
                math.isclose(c_rbi, g_rbi, abs_tol=1e-4) and \
                math.isclose(c_mds, g_mds, abs_tol=1e-4)
                
        if not match:
            all_match = False
            print(f"Mismatch at index {i}:")
            print(f"  CPU -> fit: {c_fit:.6f}, rbi: {c_rbi:.6f}, mds: {c_mds:.6f}")
            print(f"  GPU -> fit: {g_fit:.6f}, rbi: {g_rbi:.6f}, mds: {g_mds:.6f}")
    
    if all_match:
        print("SUCCESS: CPU and GPU match perfectly within float tolerance!")
    else:
        print("FAILURE: There are mathematical differences.")

if __name__ == "__main__":
    run_verification()
