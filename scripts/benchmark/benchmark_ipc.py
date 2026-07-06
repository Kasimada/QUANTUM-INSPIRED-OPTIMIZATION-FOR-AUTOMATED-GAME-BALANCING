import time
import statistics
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.datasets.csv_loader import load_csv_dataset

def worker_disk(data_path: str):
    data = load_csv_dataset(data_path)
    return len(data)

def worker_ipc(data: list):
    return len(data)

def run_benchmark():
    data_path = "data/raw/original_character_stats.csv"
    base_data = load_csv_dataset(data_path)
    
    trials = 30
    workers = 6
    runs = 5
    
    disk_times = []
    ipc_times = []
    
    print(f"Benchmarking Dataset Loading (Disk vs IPC) over {runs} runs with {trials} trials and {workers} workers...\n")
    
    for i in range(runs):
        # 1. Disk run
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_disk, data_path) for _ in range(trials)]
            _ = [f.result() for f in futures]
        disk_times.append(time.perf_counter() - start)
        
        # 2. IPC run
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_ipc, base_data) for _ in range(trials)]
            _ = [f.result() for f in futures]
        ipc_times.append(time.perf_counter() - start)
        
        print(f"Run {i+1}: Disk={disk_times[-1]:.4f}s, IPC={ipc_times[-1]:.4f}s")
        
    avg_disk = statistics.mean(disk_times)
    std_disk = statistics.stdev(disk_times)
    avg_ipc = statistics.mean(ipc_times)
    std_ipc = statistics.stdev(ipc_times)
    
    print("\n=== Results ===")
    print(f"Disk (Current): {avg_disk:.4f}s ± {std_disk:.4f}s")
    print(f"IPC (Shared)  : {avg_ipc:.4f}s ± {std_ipc:.4f}s")
    
    if avg_ipc < avg_disk:
        improvement = (avg_disk - avg_ipc) / avg_disk * 100
        print(f"IPC is faster by {improvement:.2f}%")
        if improvement >= 10.0:
            print("Recommendation: USE IPC")
        else:
            print("Recommendation: KEEP DISK (Improvement < 10%)")
    else:
        degradation = (avg_ipc - avg_disk) / avg_disk * 100
        print(f"IPC is SLOWER by {degradation:.2f}%")
        print("Recommendation: KEEP DISK")

if __name__ == "__main__":
    run_benchmark()
