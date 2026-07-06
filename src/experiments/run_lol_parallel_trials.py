"""Run LoL balancing trials in parallel at the independent-trial level.

This runner keeps the optimizer implementations unchanged.  Each worker process
executes one full trial and returns scalar result rows plus convergence history;
the parent process merges and writes the usual result package.
"""

from __future__ import annotations

import argparse
import json
import random
import os
import platform
import multiprocessing

# RESTRICT MULTITHREADING TO PREVENT MEMORY EXPLOSION
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import src.utils.evaluation_utils as evaluation_utils
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from omegaconf import DictConfig

class ExperimentCancelledError(Exception):
    pass



from src.algorithms.algorithms import GENES_PER_CHAMP, evaluate
from src.algorithms.discrete_algorithms import PATCH_LEVELS, evaluate_discrete
from src.datasets.csv_loader import load_csv_dataset as load_champion_data

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run continuous or discrete LoL trials with one CPU process per trial."
    )
    parser.add_argument("--mode", choices=["continuous", "discrete"], required=True)
    parser.add_argument("--data", default="riot_datadragon_champion_stats.csv")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--fes", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chart-name", default=None)
    parser.add_argument("--include-qea", action="store_true")
    return parser.parse_args()

def run_experiment(cfg: DictConfig) -> None:
    import time
    start_time = time.time()
    exp_cfg = cfg.experiment
    mode = exp_cfg.mode
    # Support Phase 5 dataset root config
    dataset_cfg = cfg.get("dataset", None)
    if dataset_cfg is not None:
        if isinstance(dataset_cfg, str):
            data = dataset_cfg
            dataset_display_name = "Static Character-Attribute Dataset"
            dataset_source_name = "Unspecified source"
        else:
            data = dataset_cfg.get("path", "data/raw/original_character_stats.csv")
            dataset_display_name = dataset_cfg.get("display_name", "Static Character-Attribute Dataset")
            dataset_source_name = dataset_cfg.get("source_name", "Unspecified source")
    else:
        # Fallback to older exp_cfg
        raw_dataset = exp_cfg.get("dataset", "data/raw/original_character_stats.csv")
        if isinstance(raw_dataset, str):
            data = raw_dataset
            dataset_display_name = "Static Character-Attribute Dataset"
            dataset_source_name = "Unspecified source"
        else:
            data = raw_dataset.get("path", "data/raw/original_character_stats.csv")
            dataset_display_name = raw_dataset.get("display_name", "Static Character-Attribute Dataset")
            dataset_source_name = raw_dataset.get("source_name", "Unspecified source")
            
    trials = exp_cfg.trials
    fes = exp_cfg.fes
    seed = exp_cfg.get("seed_start", 2002 if mode == "discrete" else 1002)
    workers = exp_cfg.get("workers", 6)
    out_dir_str = exp_cfg.get("output_pattern", None)
    
    include_qea = exp_cfg.get("include_qea", False)
    use_gpu = exp_cfg.get("use_gpu", False)
    from src.algorithms.factory import get_supported_algorithms
    configured_algorithms = list(exp_cfg.get("algorithms", get_supported_algorithms(mode)))
    algo_count = len(configured_algorithms)
    scenario = exp_cfg.get("scenario", "symmetric")

    if out_dir_str is None:
        raise ValueError("Absolute output directory must be provided via cfg.experiment.output_pattern")
    out_dir = Path(out_dir_str)
    
    # Setup run.log
    import logging
    log_file = out_dir / "run.log"
    logging.basicConfig(filename=str(log_file), level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info("=== Game Balancing Optimization Started ===")
    logging.info(f"Mode: {mode} | Dataset: {data} | Trials: {trials} | FEs: {fes}")


    # Setup run.log
    import logging
    
    log_file = out_dir / "run.log"
    logging.basicConfig(filename=str(log_file), level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info("=== Game Balancing Optimization Started ===")
    logging.info(f"Python: {platform.python_version()} | OS: {platform.system()} {platform.release()} | CPU: {multiprocessing.cpu_count()} cores")
    logging.info(f"Mode: {mode} | Dataset: {data} | Trials: {trials} | FEs: {fes}")


    base_data = load_champion_data(data)
    if not base_data:
        raise SystemExit("No champion data loaded.")

    eval_seed = cfg.get("evaluation_seed", 20260612)
    baseline = build_baseline(mode, base_data, seed, eval_seed)
    workers_count = max(1, min(int(workers), int(trials)))
    rows: list[dict] = []
    histories: list[dict] = []

    print(
        f"Running {mode.upper()} {fes:,} FEs x {trials} trials "
        f"with {workers_count} CPU worker processes.",
        flush=True,
    )
    print(f"Output folder: {out_dir.resolve()}", flush=True)

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    from src.utils.keyboard import KeyboardController
    import threading

    cancel_event = threading.Event()
    
    def keyboard_listener():
        while not cancel_event.is_set():
            if KeyboardController.check_key('c') or KeyboardController.check_key('\x1b'):
                cancel_event.set()
                print("\n[!] Cancellation requested by user. Waiting for current trial(s) to finish...", flush=True)
                break
            import time
            time.sleep(0.1)

    listener_thread = threading.Thread(target=keyboard_listener, daemon=True)
    listener_thread.start()

    with ProcessPoolExecutor(max_workers=workers_count) as executor:
        futures = [
            executor.submit(
                run_one_trial,
                mode,
                data,
                fes,
                seed + trial,
                eval_seed,
                trial,
                trials,
                configured_algorithms,
                use_gpu,
                str(out_dir),
                exp_cfg.get("scenario", "symmetric")
            )
            for trial in range(trials)
        ]
        
        if has_tqdm:
            pbar = tqdm(total=trials, desc=f"Optimizing ({mode})")
            
        for completed, future in enumerate(as_completed(futures), start=1):
            if cancel_event.is_set():
                for f in futures:
                    f.cancel()
                raise ExperimentCancelledError("User cancelled execution via keyboard.")
                
            try:
                trial_rows, trial_histories = future.result()
                rows.extend(trial_rows)
                histories.extend(trial_histories)
                sort_results(mode, rows, histories, configured_algorithms)
                
                # Write partial
                module = report_module(mode)
                module.write_csv(out_dir / "runs_partial.csv", rows)
                module.write_csv(out_dir / "summary_partial.csv", module.summarize(rows))
                
                finished_trial = trial_rows[0]["trial"] if trial_rows else "?"
                if has_tqdm:
                    pbar.update(1)
                    pbar.set_postfix({"Last trial": finished_trial})
                else:
                    print(f"[MERGE] completed trial {finished_trial}; {completed}/{trials} trials merged.", flush=True)
            except Exception as e:
                import traceback
                print(f"\n[ERROR] Trial failed with exception: {e}\n{traceback.format_exc()}", flush=True)
                logging.error(f"A trial failed: {e}\n{traceback.format_exc()}")
                if has_tqdm:
                    pbar.update(1)
                    
        if has_tqdm:
            pbar.close()
            
        # Clean shutdown for thread
        cancel_event.set()

    sort_results(mode, rows, histories, configured_algorithms)
    validate_complete_result(mode, rows, histories, trials, configured_algorithms)
    summary = report_module(mode).summarize(rows)
    best_trial = report_module(mode).select_best_trial(rows)
    report_module(mode).write_csv(out_dir / "runs.csv", rows)
    report_module(mode).write_csv(out_dir / "summary.csv", summary)
    report_module(mode).write_markdown_table(
        out_dir / "result_table.md", summary, baseline
    )
    
    chart_name = "chart.png"
    include_qea = "qea" in configured_algorithms
    
    # Readme args
    r_args = SimpleNamespace(
        data=data, trials=trials, fes=fes, seed=seed, out=out_dir, 
        chart_name=chart_name, include_qea=include_qea
    )
    report_module(mode).write_readme(
        out_dir / "README.md",
        r_args,
        summary,
        baseline,
        dataset_display_name,
        dataset_source_name,
        data,
        best_trial,
    )
    report_module(mode).write_best_chart(
        out_dir / chart_name, histories, best_trial
    )
    reference = "aqea" if mode == "continuous" else "aqea_discrete"
    report_module(mode).write_analysis(
        out_dir / "analysis.md", rows, reference=reference
    )


    print(f"\nWrote {out_dir / 'runs.csv'}", flush=True)
    print(f"Wrote {out_dir / 'summary.csv'}", flush=True)
    print(f"Wrote {out_dir / 'result_table.md'}", flush=True)
    print(f"Wrote {out_dir / 'README.md'}", flush=True)
    
    # Save best patch metadata for each algorithm
    from src.utils.patch_exporter import save_best_patch_metadata
    for row in summary:
        alg = row.get("algorithm", row.get("Algorithm"))
        bt = row.get("best_trial", row.get("Best Trial", 1))
        alg_patch_dir = out_dir / "patches" / alg
        if alg_patch_dir.exists():
            save_best_patch_metadata(alg_patch_dir, bt, f"trial_{bt:02d}.csv")

    print(f"Wrote {out_dir / chart_name}", flush=True)
    print(f"Wrote {out_dir / 'analysis.md'}", flush=True)




def run_one_trial(
    mode: str,
    data_path: str,
    fes: int,
    seed: int,
    eval_seed: int,
    trial: int,
    total_trials: int,
    configured_algorithms: list[str],
    use_gpu: bool,
    out_dir_str: str = "",
    scenario: str = "symmetric"
) -> tuple[list[dict], list[dict]]:
    if mode == "continuous":
        return run_one_continuous_trial(
            data_path, fes, seed, eval_seed, trial, total_trials, configured_algorithms, use_gpu, out_dir_str, scenario
        )
    return run_one_discrete_trial(
        data_path, fes, seed, eval_seed, trial, total_trials, configured_algorithms, use_gpu, out_dir_str, scenario
    )


def run_one_continuous_trial(
    data_path: str,
    fes: int,
    seed: int,
    eval_seed: int,
    trial: int,
    total_trials: int,
    configured_algorithms: list[str],
    use_gpu: bool,
    out_dir_str: str = "",
    scenario: str = "symmetric"
) -> tuple[list[dict], list[dict]]:
    from src.algorithms.factory import create_optimizer
    import src.evaluation.parallel_evaluator as pe
    pe.USE_GPU = use_gpu
    # PREVENT NESTED PARALLELISM MEMORY EXPLOSION
    pe.MAX_WORKERS = 1

    if use_gpu:
        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f'[Worker] use_gpu=True | torch.cuda.is_available={torch.cuda.is_available()} | device={device_name}', flush=True)

    base_data = load_champion_data(data_path)
    baseline_score, _, _, baseline_rbi, baseline_mds = evaluate(
        base_data, [1.0] * (len(base_data) * GENES_PER_CHAMP)
    )

    rows = []
    histories = []
    print(
        f'\n=== Continuous trial {trial + 1}/{total_trials} seed={seed} ===', flush=True
    )
    for algorithm_name in configured_algorithms:
        random.seed(seed)
        optimizer = create_optimizer(algorithm_name)
        from src.algorithms.algorithms import evaluate_batch_wrapper
        import time
        
        start_time = time.perf_counter()
        opt_config = {"max_FEs": fes, "scenario": scenario, "trial": trial, "seed": seed, "evaluation_seed": eval_seed, "out_dir": out_dir_str}
        best, _, hx, hy = optimizer.optimize(evaluate_batch_wrapper, base_data, opt_config)
        
        if use_gpu:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # EXPORT PATCH
        from src.utils.patch_exporter import export_patch
        import os
        patch_dir = os.path.join(out_dir_str, 'patches', algorithm_name)
        patch_path = os.path.join(patch_dir, f'trial_{trial+1:02d}.csv')
        export_patch(base_data, best, patch_path)
        
        runtime_seconds = end_time - start_time
        fes_per_sec = fes / max(runtime_seconds, 1e-9)
        # Recalculate score components
        score, _, _, rbi, mds, Sbalance, Pduration, Sentropy, PL2 = evaluate(base_data, best, return_components=True)
        
        optimizer_backend = getattr(optimizer, 'optimizer_backend', 'default')
        final_logging_backend = getattr(optimizer, 'final_logging_backend', 'default')
        cpu_legacy_relogged = getattr(optimizer, 'cpu_legacy_relogged', False)
        
        convergence_fe_95 = evaluation_utils.ContinuousReport.first_fe_at_fraction(hx, hy, 0.95)
        convergence_speed = max(0.0, 100.0 * (1.0 - convergence_fe_95 / max(1, fes)))
        row = {
            "trial": trial,
            "seed": seed,
            "evaluation_seed": eval_seed,
            "algorithm": algorithm_name,
            "fitness": score,
            "rbi": rbi,
            "mds": mds,
            "completion": max(0.0, 100.0 - rbi * 100.0),
            "balance_score": max(0.0, 100.0 - rbi * 100.0),
            "diversity_score": mds,
            "Sbalance": Sbalance,
            "Pduration": Pduration,
            "Sentropy": Sentropy,
            "PL2": PL2,
            "convergence_fe_95": convergence_fe_95,
            "convergence_speed": convergence_speed,
            "baseline_fitness": baseline_score,
            "baseline_rbi": baseline_rbi,
            "baseline_mds": baseline_mds,
            "fes": fes,
            "runtime_seconds": runtime_seconds,
            "fes_per_sec": fes_per_sec,
            "scenario": opt_config.get("scenario", "symmetric"),
            "optimizer_backend": optimizer_backend,
            "final_logging_backend": final_logging_backend,
            "cpu_legacy_relogged": cpu_legacy_relogged
        }
        rows.append(row)
        histories.append(
            {
                "trial": trial,
                "seed": seed,
                "evaluation_seed": eval_seed,
                "algorithm": algorithm_name,
                "fitness": score,
                "hx": hx,
                "hy": hy,
            }
        )
        print(
            f"[TRIAL {trial}] {algorithm_name} fitness={score:.4f} rbi={rbi:.4f}",
            flush=True,
        )
    return rows, histories

def run_one_discrete_trial(
    data_path: str,
    fes: int,
    seed: int,
    eval_seed: int,
    trial: int,
    total_trials: int,
    configured_algorithms: list[str],
    use_gpu: bool,
    out_dir_str: str = "",
    scenario: str = "symmetric"
) -> tuple[list[dict], list[dict]]:
    from src.algorithms.factory import create_optimizer
    from src.algorithms.discrete_algorithms import (
        multipliers_to_indices,
    )

    import src.evaluation.parallel_evaluator as pe
    pe.USE_GPU = use_gpu
    # PREVENT NESTED PARALLELISM MEMORY EXPLOSION
    pe.MAX_WORKERS = 1

    if use_gpu:
        import torch
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"[Worker] use_gpu=True | torch.cuda.is_available={torch.cuda.is_available()} | device={device_name}", flush=True)

    base_data = load_champion_data(data_path)
    num_genes = len(base_data) * GENES_PER_CHAMP
    baseline_indices = [PATCH_LEVELS.index(1.0)] * num_genes
    baseline_score, _, _, baseline_rbi, baseline_mds, _, _, _ = evaluate_discrete(
        base_data, baseline_indices, evaluate
    )

    rows = []
    histories = []
    print(
        f"\\n=== Discrete trial {trial + 1}/{total_trials} seed={seed} ===", flush=True
    )
    for algorithm_name in configured_algorithms:
        random.seed(seed)
        optimizer = create_optimizer(algorithm_name)
        from src.algorithms.discrete_algorithms import evaluate_batch_wrapper_discrete
        import time
        
        start_time = time.perf_counter()
        opt_config = {"max_FEs": fes, "scenario": scenario, "trial": trial, "seed": seed, "evaluation_seed": eval_seed, "out_dir": out_dir_str}
        best, _, hx, hy = optimizer.optimize(evaluate_batch_wrapper_discrete, base_data, opt_config)
        
        if use_gpu:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        # EXPORT PATCH
        from src.utils.patch_exporter import export_patch
        import os
        patch_dir = os.path.join(out_dir_str, 'patches', algorithm_name)
        patch_path = os.path.join(patch_dir, f'trial_{trial+1:02d}.csv')
        export_patch(base_data, best, patch_path)
        
        runtime_seconds = end_time - start_time
        fes_per_sec = fes / max(runtime_seconds, 1e-9)
        best_indices = multipliers_to_indices(best)
        score, _, _, rbi, mds, pressure, violation, magnitude, Sbalance, Pduration, Sentropy, PL2 = evaluate_discrete(base_data, best_indices, evaluate, return_components=True)
        
        convergence_fe_95 = evaluation_utils.ContinuousReport.first_fe_at_fraction(hx, hy, 0.95)
        convergence_speed = max(0.0, 100.0 * (1.0 - convergence_fe_95 / max(1, fes)))
        row = {
            "trial": trial,
            "seed": seed,
            "evaluation_seed": eval_seed,
            "algorithm": algorithm_name,
            "fitness": score,
            "rbi": rbi,
            "mds": mds,
            "net_patch_pressure": pressure,
            "constraint_violation": violation,
            "patch_magnitude": magnitude,
            "completion": max(0.0, 100.0 - rbi * 100.0),
            "balance_score": max(0.0, 100.0 - rbi * 100.0),
            "diversity_score": mds,
            "Sbalance": Sbalance,
            "Pduration": Pduration,
            "Sentropy": Sentropy,
            "PL2": PL2,
            "convergence_fe_95": convergence_fe_95,
            "convergence_speed": convergence_speed,
            "baseline_fitness": baseline_score,
            "baseline_rbi": baseline_rbi,
            "baseline_mds": baseline_mds,
            "fes": fes,
            "runtime_seconds": runtime_seconds,
            "fes_per_sec": fes_per_sec,
            "scenario": opt_config.get("scenario", "symmetric")
        }
        rows.append(row)
        histories.append(
            {
                "trial": trial,
                "seed": seed,
                "evaluation_seed": eval_seed,
                "algorithm": algorithm_name,
                "fitness": score,
                "hx": hx,
                "hy": hy,
            }
        )
        print(
            f"[TRIAL {trial}] {algorithm_name} fitness={score:.4f} rbi={rbi:.4f} pressure={pressure}",
            flush=True,
        )
    return rows, histories

def report_module(mode: str):
    if mode == 'continuous':
        return evaluation_utils.ContinuousReport
    else:
        return evaluation_utils.DiscreteReport

def build_baseline(mode: str, base_data, seed: int, eval_seed: int) -> dict:
    from src.algorithms.algorithms import evaluate, GENES_PER_CHAMP
    if mode == 'continuous':
        score, _, _, rbi, mds = evaluate(base_data, [1.0] * (len(base_data) * GENES_PER_CHAMP))
        return {'fitness': score, 'rbi': rbi, 'mds': mds, 'completion': max(0.0, 100.0 - rbi * 100.0), 'balance_score': max(0.0, 100.0 - rbi * 100.0), 'diversity_score': mds}
    else:
        from src.algorithms.discrete_algorithms import evaluate_discrete
        from src.algorithms.discrete_algorithms import PATCH_LEVELS
        num_genes = len(base_data) * GENES_PER_CHAMP
        baseline_indices = [PATCH_LEVELS.index(1.0)] * num_genes
        score, _, _, rbi, mds, pressure, violation, magnitude = evaluate_discrete(base_data, baseline_indices, evaluate)
        return {'fitness': score, 'rbi': rbi, 'mds': mds, 'completion': max(0.0, 100.0 - rbi * 100.0), 'balance_score': max(0.0, 100.0 - rbi * 100.0), 'diversity_score': mds, 'net_patch_pressure': pressure, 'constraint_violation': violation, 'patch_magnitude': magnitude}

def sort_results(mode: str, rows: list, histories: list, configured_algorithms: list):
    rows.sort(key=lambda r: (r['algorithm'], -r['fitness']))

def validate_complete_result(mode: str, rows: list, histories: list, trials: int, configured_algorithms: list):
    expected = trials * len(configured_algorithms)
    if len(rows) != expected:
        print(f'Warning: Expected {expected} rows, got {len(rows)}')
