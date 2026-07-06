"""Parallel candidate evaluation helpers for the LoL balancing experiments.

This module is intentionally non-invasive: it does not change the existing
GA/PSO/QEA/AQEA runners.  It provides a reusable multiprocessing layer that can
be wired into future runners after the current long experiment finishes.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal
from src.evaluation.backend import CPUBackend, GPUBackend, EvaluationResult, EvaluationBackend

Mode = Literal["continuous", "discrete"]

USE_GPU = False
MAX_WORKERS = None

def get_backend() -> EvaluationBackend:
    if USE_GPU:
        return GPUBackend()
    return CPUBackend()

def evaluate_batch(
    base_data: list[dict[str, Any]],
    candidates: Iterable[list[float]] | Iterable[list[int]],
    *,
    mode: Mode = "continuous",
    workers: int | None = None,
    **kwargs
) -> list[EvaluationResult]:
    """Evaluate candidates using the selected backend (CPU or GPU).
    
    Parameters
    ----------
    base_data:
        Champion stat rows loaded by the existing project loaders.
    candidates:
        Continuous chromosomes or discrete patch-index vectors.
    mode:
        "continuous" or "discrete".
    workers:
        Number of CPU worker processes (if using CPU backend).
    """
    candidate_list = list(candidates)
    if not candidate_list:
        return []
        
    if MAX_WORKERS is not None and workers is not None:
        workers = min(workers, MAX_WORKERS)
        
    backend = get_backend()
    
    # GPU backend ignores the workers parameter internally and handles batches via tensor ops
    return backend.evaluate(base_data, candidate_list, mode=mode, workers=workers, **kwargs)
