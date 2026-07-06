"""Benchmark the multiprocessing evaluator without modifying experiment code."""

from __future__ import annotations

import argparse
import random
import time

from src.algorithms.algorithms import GENES_PER_CHAMP
from src.datasets.csv_loader import load_csv_dataset as load_champion_data
from src.evaluation.parallel_evaluator import evaluate_batch


def build_continuous_candidates(
    count: int, dimension: int, seed: int
) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(0.7, 1.3) for _ in range(dimension)] for _ in range(count)]


def build_discrete_candidates(count: int, dimension: int, seed: int) -> list[list[int]]:
    from src.algorithms.discrete_algorithms import random_feasible_indices

    random.seed(seed)
    return [random_feasible_indices(dimension) for _ in range(count)]


from src.evaluation.backend import EvaluationResult

def timed_eval(
    base_data, candidates, mode: str, workers: int
) -> tuple[float, list[EvaluationResult]]:
    start = time.perf_counter()
    results = evaluate_batch(base_data, candidates, mode=mode, workers=workers)
    return time.perf_counter() - start, results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sequential and multiprocessing candidate evaluation speed."
    )
    parser.add_argument("--data", default="riot_datadragon_champion_stats.csv")
    parser.add_argument(
        "--mode", choices=["continuous", "discrete"], default="continuous"
    )
    parser.add_argument("--candidates", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    base_data = load_champion_data(args.data)
    dimension = len(base_data) * GENES_PER_CHAMP
    if args.mode == "continuous":
        candidates = build_continuous_candidates(args.candidates, dimension, args.seed)
    else:
        candidates = build_discrete_candidates(args.candidates, dimension, args.seed)

    sequential_time, sequential = timed_eval(
        base_data, candidates, args.mode, workers=1
    )
    parallel_time, parallel = timed_eval(
        base_data, candidates, args.mode, workers=args.workers
    )

    max_delta = max(
        abs(left["fitness"] - right["fitness"])
        for left, right in zip(sequential, parallel)
    )
    speedup = sequential_time / parallel_time if parallel_time > 0 else float("inf")

    print(f"Mode: {args.mode}")
    print(f"Candidates: {args.candidates}")
    print(f"Workers: {args.workers}")
    print(f"Sequential time: {sequential_time:.3f}s")
    print(f"Parallel time: {parallel_time:.3f}s")
    print(f"Estimated speedup: {speedup:.2f}x")
    print(f"Max fitness delta: {max_delta:.12f}")
    if max_delta > 1e-9:
        raise SystemExit("ERROR: parallel evaluation changed numerical results.")


if __name__ == "__main__":
    main()
