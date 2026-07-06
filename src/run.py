import shutil
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
from omegaconf import OmegaConf

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Entry Point for Game Balancing using AQEA.")
    
    parser.add_argument("--mode", type=str, choices=["continuous", "discrete", "both"], required=True,
                        help="Optimization space: 'continuous', 'discrete', or 'both'.")
    parser.add_argument("--dataset", type=str, default="data/raw/original_character_stats.csv",
                        help="Path to the character stats dataset CSV.")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of independent trials to run.")
    parser.add_argument("--fes", type=int, default=50000,
                        help="Maximum number of fitness evaluations per trial.")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel worker processes (limits concurrent CPU/GPU usage).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed for reproducibility.")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output directory. If not specified, defaults to results/...")
    parser.add_argument("--algorithm", type=str, default="ALL",
                        help="Comma-separated algorithms to run (e.g. AQEA, GA, ALL).")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: 'cpu', 'cuda', 'auto', 'cuda:0', 'cuda:1', etc.")
    parser.add_argument("--layout", type=str, default="date_config", choices=["default", "date", "config", "date_config", "flat"],
                        help="Output folder structure layout.")
    parser.add_argument("--launcher-profile", type=str, default=None,
                        help="Internal: pass launcher profile name if launched via UI.")
    parser.add_argument("--scenario", type=str, default="symmetric",
                        help="Simulation scenario (e.g. symmetric, asymmetric_support_handicap).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Perform a pre-flight health check without executing the optimization.")
    
    return parser.parse_args()

def run_mode(mode, args):
    import src.experiments.run_lol_parallel_trials as rpt
    
    # Process device
    use_gpu = False
    device_str = args.device.lower()
    if device_str.startswith("cuda"):
        use_gpu = True
        if ":" in device_str:
            device_id = device_str.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    elif device_str == "auto":
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            
    # Process algorithms
    algos_str = args.algorithm.upper()
    algorithms = []
    
    from src.algorithms.factory import get_supported_algorithms
    supported = get_supported_algorithms(mode)
    
    if algos_str == "ALL":
        algorithms = supported
    else:
        parts = [p.strip().lower() for p in algos_str.split(",")]
        for p in parts:
            if p in supported:
                algorithms.append(p)
            else:
                # Fallback for display names mapping without suffix
                mapped = p + "_discrete" if mode == "discrete" else p
                if mapped in supported:
                    algorithms.append(mapped)
                else:
                    # also handle nsga2 -> nsga_ii
                    if mapped.replace("nsga2", "nsga_ii") in supported:
                        algorithms.append(mapped.replace("nsga2", "nsga_ii"))
                    else:
                        print(f"Warning: Algorithm '{p}' not recognized for mode {mode}.")
                
    if not algorithms:
        raise ValueError(f"No valid algorithms selected for mode {mode}.")
        
    output_pattern = args.output if args.output else None
    

    if getattr(args, 'dry_run', False):
        validate_and_dry_run(args, algorithms, use_gpu, mode)
        return


    from src.utils.output_manager import OutputManager, OutputLayout
    
    layout_map = {
        "default": OutputLayout.DEFAULT,
        "date": OutputLayout.DATE,
        "config": OutputLayout.CONFIG,
        "date_config": OutputLayout.DATE_CONFIG,
        "flat": OutputLayout.FLAT
    }
    
    base_out_dir = output_pattern if output_pattern else "results"
    
    config_data = {
        "mode": mode,
        "dataset": args.dataset,
        "trials": args.trials,
        "fes": args.fes,
        "seed_start": args.seed if args.seed is not None else (2002 if mode == "discrete" else 1002),
        "workers": args.workers,
        "algorithms": algorithms,
        "use_gpu": use_gpu,
        "scenario": args.scenario,
    }
    
    cfg = OmegaConf.create({"experiment": config_data})
    
    print("\n=======================================================")
    print(f"Starting {mode.upper()} Optimization")
    print("=======================================================")
    
    out_manager = OutputManager(
        base_path=base_out_dir,
        layout=layout_map.get(args.layout, OutputLayout.DATE_CONFIG),
        config_data=config_data,
        launcher_profile=args.launcher_profile
    )
    
    # In CLI mode, if there's a collision and it's FLAT/DATE, we just print a warning and let it append
    # (Launcher handles prompt UX separately)
    collision = out_manager.detect_collision()
    if collision.exists and (collision.has_lock or collision.has_manifest):
        print(f"Warning: Output directory might contain previous results (status: {collision.status}).")
        
    with out_manager as out_info:
        cfg.experiment.output_pattern = out_info.absolute_path
        rpt.run_experiment(cfg)



def validate_and_dry_run(args, algorithms, use_gpu, mode):
    import platform
    import multiprocessing
    import os
    print("\n--- Pre-flight Health Check ---")
    
    print(f"[OK] Python version: {platform.python_version()}")
    
    try:
        import numpy as np
        print(f"[OK] numpy version: {np.__version__}")
        import pandas as pd
        print(f"[OK] pandas version: {pd.__version__}")
    except ImportError as e:
        raise ValueError(f"Missing essential data library: {e}")
        
    try:
        import torch
        print(f"[OK] torch version: {torch.__version__}")
        if use_gpu:
            if torch.cuda.is_available():
                print("[OK] CUDA available: Yes")
            else:
                raise ValueError("CUDA requested but not available.")
    except ImportError:
        print("[!] torch not installed. GPU acceleration disabled.")
        if use_gpu:
            raise ValueError("use_gpu=True but torch is missing")
            
    print(f"[OK] CPU cores available: {multiprocessing.cpu_count()}")
    
    try:
        import psutil
        free_ram = psutil.virtual_memory().available / (1024 ** 3)
        print(f"[OK] Free RAM: {free_ram:.1f} GB")
    except ImportError:
        print("[?] Free RAM: N/A (psutil not installed)")

    try:
        import pandas as pd
        df = pd.read_csv(args.dataset)
        print(f"[OK] Dataset OK ({len(df)} rows)")
    except Exception as e:
        raise ValueError(f"Dataset read failed: {e}")
        
    out = Path(args.output) if args.output else Path("results")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not os.access(out.parent, os.W_OK):
        raise ValueError(f"Output directory parent is not writable: {out.parent}")
    
    total, used, free = shutil.disk_usage(out.parent)
    if free < 100 * 1024 * 1024:
        print("[!] Warning: Less than 100MB free space on output drive.")
    else:
        print("[OK] Output Folder OK")
        
    print(f"[OK] Algorithms Configured: {', '.join(algorithms)}")
    
    from src.algorithms.factory import get_supported_algorithms, get_algorithm_info, create_optimizer
    supported = get_supported_algorithms(mode)
    if not supported:
        raise ValueError("Factory registry returned empty list of supported algorithms.")
    for algo in algorithms:
        if algo not in supported:
            raise ValueError(f"Algorithm {algo} not in supported algorithms: {supported}")
        info = get_algorithm_info(algo)
        if not info:
            raise ValueError(f"Failed to get info for {algo}")
        _ = create_optimizer(algo)
    print("[OK] Factory Registry OK")
    
    print("--- Dry Run Passed! Ready to Start ---\n")

def main():
    args = parse_args()
    
    modes_to_run = ["continuous", "discrete"] if args.mode == "both" else [args.mode]
    
    for mode in modes_to_run:
        run_mode(mode, args)

if __name__ == "__main__":
    main()


