from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Mapping
from numbers import Number
from typing import Any

@dataclass
class EvaluationResult:
    fitness: float
    metrics: Mapping[str, Number] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

class EvaluationBackend(ABC):
    @abstractmethod
    def evaluate(self, base_data: list[dict[str, Any]], candidates: list[list[float]] | list[list[int]], mode: str = "continuous", **kwargs) -> list[EvaluationResult]:
        """Evaluate a batch of candidates and return detailed EvaluationResults."""
        pass

import os
from concurrent.futures import ProcessPoolExecutor

_BASE_DATA = None
_MODE = "continuous"
_CONFIG = {}

def _init_worker(base_data, mode, config):
    global _BASE_DATA, _MODE, _CONFIG
    _BASE_DATA = base_data
    _MODE = mode
    _CONFIG = config

def _eval_worker(candidate):
    import time
    from src.algorithms.algorithms import evaluate as eval_cont
    from src.algorithms.discrete_algorithms import evaluate_discrete
    
    start_t = time.perf_counter()
    
    # Ensure return_components is set without duplicating kwargs
    worker_config = dict(_CONFIG)
    worker_config['return_components'] = True
    
    if _MODE == "continuous":
        res = eval_cont(_BASE_DATA, candidate, **worker_config)
        fitness, _, _, rbi, mds, Sbalance, Pduration, Sentropy, PL2 = res
        metrics = {
            "rbi": float(rbi), "mds": float(mds), "Sbalance": float(Sbalance),
            "Pduration": float(Pduration), "Sentropy": float(Sentropy), "PL2": float(PL2)
        }
    else:
        res = evaluate_discrete(_BASE_DATA, candidate, eval_cont, **worker_config)
        fitness, _, _, rbi, mds, pressure, violation, magnitude, Sbalance, Pduration, Sentropy, PL2 = res
        metrics = {
            "rbi": float(rbi), "mds": float(mds), "pressure": float(pressure),
            "violation": float(violation), "magnitude": float(magnitude),
            "Sbalance": float(Sbalance), "Pduration": float(Pduration),
            "Sentropy": float(Sentropy), "PL2": float(PL2)
        }
    elapsed = time.perf_counter() - start_t
    
    return EvaluationResult(
        fitness=float(fitness),
        metrics=metrics,
        metadata={"backend": "CPU", "device": "cpu", "elapsed_sec": elapsed}
    )

class CPUBackend(EvaluationBackend):
    def evaluate(self, base_data: list[dict[str, Any]], candidates: list[list[float]] | list[list[int]], mode: str = "continuous", workers: int | None = None, **kwargs) -> list[EvaluationResult]:
        candidate_list = list(candidates)
        if not candidate_list:
            return []
            
        if workers is None:
            workers = max(1, (os.cpu_count() or 2) - 1)
        workers = max(1, int(workers))
        
        if workers == 1 or len(candidate_list) == 1:
            _init_worker(base_data, mode, kwargs)
            return [_eval_worker(c) for c in candidate_list]
            
        chunksize = max(1, len(candidate_list) // (workers * 4))
        
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(base_data, mode, kwargs),
        ) as executor:
            return list(executor.map(_eval_worker, candidate_list, chunksize=chunksize))

class GPUBackend(EvaluationBackend):
    def evaluate(self, base_data: list[dict[str, Any]], candidates: list[list[float]] | list[list[int]], mode: str = "continuous", **kwargs) -> list[EvaluationResult]:
        from src.simulation.simulator_tensor import evaluate_batch_tensor
        import time
        import torch
        
        start_t = time.perf_counter()
        
        if mode == "discrete":
            from src.algorithms.discrete_algorithms import indices_to_multipliers, patch_pressure_deviation_ratio, patch_magnitude_ratio, patch_pressure_from_indices, PATCH_PRESSURE_PENALTY_WEIGHT, PATCH_MAGNITUDE_PENALTY_WEIGHT
            chromosomes = [indices_to_multipliers(ind) for ind in candidates]
        else:
            chromosomes = list(candidates)
            
        try:
            raw_results = evaluate_batch_tensor(base_data, chromosomes, mode=mode, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Retry once
            try:
                raw_results = evaluate_batch_tensor(base_data, chromosomes, mode=mode, **kwargs)
            except torch.cuda.OutOfMemoryError:
                print("⚠️ [GPUBackend] OOM error persistent after retry. Fallback to CPU.")
                return CPUBackend().evaluate(base_data, candidates, mode=mode, **kwargs)
            
        elapsed = time.perf_counter() - start_t
        avg_elapsed = elapsed / max(1, len(candidates))
        
        final_results = []
        for i, raw in enumerate(raw_results):
            # unpack full tuple from GPU
            # evaluate_batch_tensor will be updated to return 7 components
            if len(raw) == 3:
                raw_score, rbi, mds = raw
                Sbalance, Pduration, Sentropy, PL2 = 0.0, 0.0, 0.0, 0.0
            else:
                raw_score, rbi, mds, Sbalance, Pduration, Sentropy, PL2 = raw
                
            if mode == "discrete":
                indices = candidates[i]
                pressure_deviation = patch_pressure_deviation_ratio(indices)
                patch_magnitude = patch_magnitude_ratio(indices)
                score = max(
                    raw_score
                    - PATCH_PRESSURE_PENALTY_WEIGHT * pressure_deviation
                    - PATCH_MAGNITUDE_PENALTY_WEIGHT * patch_magnitude,
                    0.0001,
                )
                metrics = {
                    "rbi": float(rbi), "mds": float(mds),
                    "pressure": float(patch_pressure_from_indices(indices)),
                    "violation": float(pressure_deviation),
                    "magnitude": float(patch_magnitude),
                    "Sbalance": float(Sbalance), "Pduration": float(Pduration),
                    "Sentropy": float(Sentropy), "PL2": float(PL2)
                }
            else:
                score = raw_score
                metrics = {
                    "rbi": float(rbi), "mds": float(mds), "Sbalance": float(Sbalance),
                    "Pduration": float(Pduration), "Sentropy": float(Sentropy), "PL2": float(PL2)
                }
                
            final_results.append(EvaluationResult(
                fitness=float(score),
                metrics=metrics,
                metadata={"backend": "GPU", "device": "cuda", "elapsed_sec": avg_elapsed}
            ))
            
        return final_results
