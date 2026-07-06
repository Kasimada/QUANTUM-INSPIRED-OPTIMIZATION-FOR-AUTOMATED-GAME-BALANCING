import json
import os
from pathlib import Path

def generate_report():
    base_file = Path("scripts/benchmark/benchmark_baseline.json")
    final_file = Path("scripts/benchmark/benchmark_final.json")
    out_file = Path("scripts/benchmark/benchmark.md")
    
    if not base_file.exists() or not final_file.exists():
        print("Missing benchmark JSON files.")
        return
        
    with open(base_file, "r") as f:
        base_data = json.load(f)
    with open(final_file, "r") as f:
        final_data = json.load(f)
        
    md = ["# Game Balancing - Architecture Refactoring Benchmark Report\n"]
    md.append("This report compares the performance of the evaluation system before and after the Strategy Pattern and Dataclass refactoring.\n")
    md.append("## CPU Benchmarks\n")
    
    md.append("| Workers | Baseline Runtime (s) | Final Runtime (s) | Baseline Throughput | Final Throughput |")
    md.append("|---|---|---|---|---|")
    
    # Simple matching by workers
    for b_run in base_data:
        w = b_run.get("workers")
        f_run = next((item for item in final_data if item.get("workers") == w), None)
        if f_run:
            brt = b_run.get("runtime_sec", 0)
            frt = f_run.get("runtime_sec", 0)
            btp = b_run.get("throughput_evals_per_sec", 0)
            ftp = f_run.get("throughput_evals_per_sec", 0)
            md.append(f"| {w} | {brt:.4f} | {frt:.4f} | {btp:.2f} | {ftp:.2f} |")
            
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"Generated {out_file}")

if __name__ == "__main__":
    generate_report()
