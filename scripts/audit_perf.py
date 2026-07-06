import time
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.csv_loader import load_csv_dataset as load_champion_data
from src.algorithms.algorithms import evaluate_batch_wrapper
from src.algorithms.discrete_algorithms import evaluate_batch_wrapper_discrete

def run_benchmark(mode, use_gpu, fes=5000):
    import src.evaluation.parallel_evaluator as pe
    pe.USE_GPU = use_gpu
    
    base_data = load_champion_data("data/raw/original_character_stats.csv")
    champs = len(base_data)
    
    if mode == "continuous":
        genes = champs * 8
        pop_size = 100
        batch = np.random.rand(pop_size, genes)
        wrapper = evaluate_batch_wrapper
    else:
        genes = champs * 4
        pop_size = 100
        batch = np.random.randint(0, 7, size=(pop_size, genes))
        wrapper = evaluate_batch_wrapper_discrete

    print(f"\n--- Benchmarking {mode.upper()} | GPU: {use_gpu} ---")
    
    # warmup
    _ = wrapper(base_data, batch, max_FEs=pop_size, scenario="symmetric")
    
    total_evals = 0
    start = time.time()
    
    while total_evals < fes:
        _ = wrapper(base_data, batch, max_FEs=pop_size, scenario="symmetric")
        total_evals += pop_size
        
    end = time.time()
    runtime = end - start
    fps = total_evals / runtime
    print(f"Total FEs: {total_evals} | Runtime: {runtime:.2f}s | Speed: {fps:.2f} FEs/sec")
    return fps

if __name__ == "__main__":
    from pathlib import Path
    out_dir = Path("results/archive_unused/perf_danger_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    results.append(("continuous", False, run_benchmark("continuous", False)))
    if torch.cuda.is_available():
        results.append(("continuous", True, run_benchmark("continuous", True)))
    
    results.append(("discrete", False, run_benchmark("discrete", False)))
    if torch.cuda.is_available():
        results.append(("discrete", True, run_benchmark("discrete", True)))
        
    with open(out_dir / "benchmark.md", "w") as f:
        f.write("# Performance Danger Audit\n\n")
        f.write("| Mode | Compute | FEs/sec |\n")
        f.write("|---|---|---|\n")
        for mode, gpu, fps in results:
            compute = "GPU" if gpu else "CPU"
            f.write(f"| {mode} | {compute} | {fps:.2f} |\n")
            
    print("\nBenchmark completed. Results saved to results/archive_unused/perf_danger_audit/benchmark.md")
