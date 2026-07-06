import time
import sys
import os
import multiprocessing
import random

sys.path.insert(0, os.path.abspath("."))
from src.datasets.csv_loader import load_csv_dataset as load_champion_data
from src.evaluation.parallel_evaluator import evaluate_batch
import src.evaluation.parallel_evaluator as pe
from src.algorithms.algorithms import GENES_PER_CHAMP

def run_trial(num_evals):
    base_data = load_champion_data("riot_datadragon_champion_stats.csv")
    dim = len(base_data) * GENES_PER_CHAMP
    rng = random.Random()
    candidates = [[rng.uniform(0.7, 1.3) for _ in range(dim)] for _ in range(100)]
    pe.USE_GPU = True
    # Do evaluations in chunks of 100
    for _ in range(num_evals):
        evaluate_batch(base_data, candidates, mode="continuous", workers=1)

def main():
    # Warm up GPU
    run_trial(1)
    
    print("--- GPU: 1 worker running 3 trials sequentially ---")
    start = time.perf_counter()
    run_trial(30) # 3 trials * 10 chunks = 3000 total candidates
    print(f"Time: {time.perf_counter() - start:.3f}s")
    
    print("--- GPU: 3 workers running 1 trial concurrently ---")
    start = time.perf_counter()
    processes = []
    for _ in range(3):
        p = multiprocessing.Process(target=run_trial, args=(10,)) # 10 chunks per worker = 3000 total
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print(f"Time: {time.perf_counter() - start:.3f}s")

if __name__ == "__main__":
    main()
