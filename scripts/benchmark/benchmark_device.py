import sys
import time
import json
import platform
import multiprocessing
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.csv_loader import load_csv_dataset
from src.algorithms.algorithms import evaluate, GENES_PER_CHAMP
import src.evaluation.parallel_evaluator as pe

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def run_benchmark(device_type="cpu", workers=1, batch_size=50):
    if device_type == "gpu" and (not HAS_TORCH or not torch.cuda.is_available()):
        return None
        
    base_data = load_csv_dataset("data/raw/original_character_stats.csv")
    if not base_data:
        return {"error": "Dataset missing"}
        
    # Generate some random candidates
    import random
    rng = random.Random(2026)
    num_genes = len(base_data) * GENES_PER_CHAMP
    candidates = [[rng.uniform(0.8, 1.2) for _ in range(num_genes)] for _ in range(batch_size)]
    
    # Configure evaluator
    pe.USE_GPU = (device_type == "gpu")
    if pe.USE_GPU:
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = platform.processor()

    start_ram = psutil.virtual_memory().used if HAS_PSUTIL else 0
    start_vram = 0
    if pe.USE_GPU:
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.perf_counter()
    
    # Run evaluation
    results = pe.evaluate_batch(base_data, candidates, mode="continuous", workers=workers)
    
    end_time = time.perf_counter()
    
    end_ram = psutil.virtual_memory().used if HAS_PSUTIL else 0
    peak_vram_gb = 0.0
    if pe.USE_GPU:
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
    runtime = max(end_time - start_time, 1e-6)
    throughput = len(candidates) / runtime
    
    return {
        "device_type": device_type,
        "device_name": device_name,
        "workers": workers,
        "batch_size": batch_size,
        "runtime_sec": runtime,
        "throughput_evals_per_sec": throughput,
        "peak_ram_gb": (end_ram - start_ram) / (1024**3) if HAS_PSUTIL else 0,
        "peak_vram_gb": peak_vram_gb
    }

def main():
    print("Starting Micro Benchmark...")
    results = []
    
    # CPU Tests
    cpu_cores = multiprocessing.cpu_count()
    test_workers = [1, 2, 4, 8, 12]
    for w in test_workers:
        if w > cpu_cores:
            continue
        print(f"Benchmarking CPU with {w} workers...")
        res = run_benchmark(device_type="cpu", workers=w, batch_size=100)
        if res and "error" not in res:
            results.append(res)
            
    # GPU Tests
    if HAS_TORCH and torch.cuda.is_available():
        gpu_workers = [1, 2, 4]
        for w in gpu_workers:
            print(f"Benchmarking GPU with {w} workers...")
            res = run_benchmark(device_type="gpu", workers=w, batch_size=100)
            if res and "error" not in res:
                results.append(res)
    
    # Export
    out_path = Path("scripts/benchmark/benchmark_final.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"Benchmark finished. Exported to {out_path.resolve()}")

if __name__ == "__main__":
    main()
