import time
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.datasets.csv_loader import load_csv_dataset as load_champion_data
from src.evaluation.parallel_evaluator import evaluate_batch
import src.evaluation.parallel_evaluator as pe
from src.algorithms.algorithms import GENES_PER_CHAMP
import random

def main():
    base_data = load_champion_data("riot_datadragon_champion_stats.csv")
    dim = len(base_data) * GENES_PER_CHAMP
    rng = random.Random(42)

    # Create 500 candidates
    candidates = [[rng.uniform(0.7, 1.3) for _ in range(dim)] for _ in range(500)]

    def test_cpu(workers):
        pe.USE_GPU = False
        start = time.perf_counter()
        evaluate_batch(base_data, candidates, mode="continuous", workers=workers)
        return time.perf_counter() - start

    print("--- CPU Workers Benchmark (500 candidates) ---")
    # Warmup
    test_cpu(1)

    for w in [1, 2, 4, 8, 12, 16]:
        t = test_cpu(w)
        print(f"CPU ({w} workers): {t:.3f}s")

    print("--- GPU Benchmark (500 candidates) ---")
    pe.USE_GPU = True
    # Warmup
    evaluate_batch(base_data, candidates[:10], mode="continuous", workers=1)

    start = time.perf_counter()
    evaluate_batch(base_data, candidates, mode="continuous", workers=1)
    gpu_time = time.perf_counter() - start
    print(f"GPU Time: {gpu_time:.3f}s")

if __name__ == "__main__":
    main()
