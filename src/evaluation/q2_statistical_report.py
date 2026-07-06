from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path

DEFAULT_METRICS = [
    "fitness",
    "balance_score",
    "diversity_score",
    "convergence_fe_95",
    "convergence_speed",
    "rbi",
    "mds",
    "constraint_violation",
    "patch_magnitude",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Q2-style statistical report from a result folder."
    )
    parser.add_argument("result_dir", type=Path, help="Folder containing runs.csv.")
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference algorithm for pairwise comparisons.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5000,
        help="Bootstrap samples for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Random seed for bootstrap/permutation tests.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output Markdown path. Defaults to result_dir/q2_statistical_report.md.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the report to stdout instead of writing a Markdown file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_path = args.result_dir / "runs.csv"
    if not runs_path.exists():
        raise SystemExit(f"Missing runs.csv: {runs_path}")

    rows = read_rows(runs_path)
    if not rows:
        raise SystemExit(f"No rows found in {runs_path}")

    metrics = [metric for metric in DEFAULT_METRICS if metric in rows[0]]
    algorithms = sorted({row["algorithm"] for row in rows})
    reference = args.reference or choose_reference(algorithms)

    random.seed(args.seed)
    report = render_report(rows, metrics, algorithms, reference, args.bootstrap)
    if args.stdout:
        print(report)
        return
    out_path = args.out or (args.result_dir / "q2_statistical_report.md")
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        row: dict[str, object] = dict(raw)
        for key, value in raw.items():
            if key in {"algorithm"}:
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
        rows.append(row)
    return rows


def choose_reference(algorithms: list[str]) -> str:
    preferred = [
        "aqea_discrete",
        "aqea",
        "balanced_qea_discrete",
        "balanced_qea",
    ]
    for name in preferred:
        if name in algorithms:
            return name
    return algorithms[0]


def render_report(
    rows: list[dict[str, object]],
    metrics: list[str],
    algorithms: list[str],
    reference: str,
    bootstrap_samples: int,
) -> str:
    lines = [
        "# Q2 Statistical Report",
        "",
        "This report is generated from `runs.csv` and is intended for Q2-style",
        "statistical screening. It should be reviewed before being copied into a paper.",
        "",
        f"Reference algorithm for pairwise comparisons: `{reference}`",
        "",
        "## Descriptive Statistics",
        "",
    ]

    for metric in metrics:
        lines.append(f"### {metric}")
        lines.append("")
        lines.append(
            "| Algorithm | n | Mean | Std | Median | IQR | 95% CI Mean | Min | Max |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
        for algorithm in algorithms:
            values = values_for(rows, algorithm, metric)
            if not values:
                continue
            summary = describe(values, bootstrap_samples)
            lines.append(
                f"| {algorithm} | {len(values)} | {summary['mean']:.6f} | "
                f"{summary['std']:.6f} | {summary['median']:.6f} | "
                f"{summary['iqr']:.6f} | [{summary['ci_low']:.6f}, {summary['ci_high']:.6f}] | "
                f"{min(values):.6f} | {max(values):.6f} |"
            )
        lines.append("")

    lines += [
        "## Pairwise Tests Against Reference",
        "",
        "Permutation p-values and Cliff's delta are non-parametric diagnostics.",
        "For small sample sizes, interpret them as screening evidence, not proof.",
        "",
        "| Metric | Comparison | Mean Diff. | Perm. p | Cliff's delta | Effect label | Holm p |",
        "|---|---|---:|---:|---:|---|---:|",
    ]

    comparisons = []
    for metric in metrics:
        ref_values = values_by_trial(rows, reference, metric)
        if not ref_values:
            continue
        for algorithm in algorithms:
            if algorithm == reference:
                continue
            other_values = values_by_trial(rows, algorithm, metric)
            common_trials = sorted(set(ref_values) & set(other_values))
            if len(common_trials) >= 2:
                a = [ref_values[t] for t in common_trials]
                b = [other_values[t] for t in common_trials]
            else:
                a = list(ref_values.values())
                b = list(other_values.values())
            if len(a) < 2 or len(b) < 2:
                continue
            mean_diff = statistics.fmean(b) - statistics.fmean(a)
            p_value = permutation_p_value(a, b, paired=len(common_trials) >= 2)
            delta = cliffs_delta(a, b)
            comparisons.append(
                {
                    "metric": metric,
                    "comparison": f"{reference} - {algorithm}",
                    "mean_diff": mean_diff,
                    "p_value": p_value,
                    "delta": delta,
                    "label": effect_label(delta),
                }
            )

    holm_values = holm_adjust([item["p_value"] for item in comparisons])
    for item, holm_p in zip(comparisons, holm_values):
        lines.append(
            f"| {item['metric']} | {item['comparison']} | {item['mean_diff']:.6f} | "
            f"{item['p_value']:.6f} | {item['delta']:.6f} | {item['label']} | {holm_p:.6f} |"
        )

    lines += [
        "",
        "## Interpretation Rules",
        "",
        "- `p < 0.05` is the conventional threshold for statistical significance, not `p < 0.5`.",
        "- p-values cannot be negative.",
        "- Effect size should be discussed together with p-values.",
        "- A strong Q2 claim should require stable mean/median differences, non-trivial effect size,",
        "  and consistent behavior across sensitivity/ablation experiments.",
        "",
    ]
    return "\n".join(lines)


def values_for(
    rows: list[dict[str, object]], algorithm: str, metric: str
) -> list[float]:
    values = []
    for row in rows:
        if row.get("algorithm") != algorithm:
            continue
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def values_by_trial(
    rows: list[dict[str, object]], algorithm: str, metric: str
) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        if row.get("algorithm") != algorithm:
            continue
        value = row.get(metric)
        trial = row.get("trial")
        if isinstance(value, (int, float)) and isinstance(trial, (int, float)):
            values[int(trial)] = float(value)
    return values


def describe(values: list[float], bootstrap_samples: int) -> dict[str, float]:
    sorted_values = sorted(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    q1 = percentile(sorted_values, 25)
    q3 = percentile(sorted_values, 75)
    ci_low, ci_high = bootstrap_ci(values, bootstrap_samples)
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "iqr": q3 - q1,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[low]
    weight = pos - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_ci(values: list[float], samples: int) -> tuple[float, float]:
    if len(values) <= 1:
        value = values[0] if values else float("nan")
        return value, value
    means = []
    n = len(values)
    for _ in range(samples):
        sample = [values[random.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    return percentile(means, 2.5), percentile(means, 97.5)


def permutation_p_value(a: list[float], b: list[float], paired: bool) -> float:
    observed = abs(statistics.fmean(a) - statistics.fmean(b))
    if paired and len(a) == len(b):
        diffs = [x - y for x, y in zip(a, b)]
        if len(diffs) <= 12:
            total = 0
            extreme = 0
            for mask in range(1 << len(diffs)):
                signed = [
                    diff if (mask >> i) & 1 else -diff for i, diff in enumerate(diffs)
                ]
                stat = abs(statistics.fmean(signed))
                total += 1
                if stat >= observed - 1e-12:
                    extreme += 1
            return extreme / total
    combined = a + b
    n_a = len(a)
    if len(combined) <= 12:
        from itertools import combinations

        total = 0
        extreme = 0
        indices = range(len(combined))
        for combo in combinations(indices, n_a):
            group_a = [combined[i] for i in combo]
            combo_set = set(combo)
            group_b = [combined[i] for i in indices if i not in combo_set]
            stat = abs(statistics.fmean(group_a) - statistics.fmean(group_b))
            total += 1
            if stat >= observed - 1e-12:
                extreme += 1
        return extreme / total
    samples = 10000
    extreme = 0
    for _ in range(samples):
        shuffled = combined[:]
        random.shuffle(shuffled)
        group_a = shuffled[:n_a]
        group_b = shuffled[n_a:]
        stat = abs(statistics.fmean(group_a) - statistics.fmean(group_b))
        if stat >= observed - 1e-12:
            extreme += 1
    return extreme / samples


def cliffs_delta(a: list[float], b: list[float]) -> float:
    greater = 0
    lesser = 0
    for x in a:
        for y in b:
            if x > y:
                greater += 1
            elif x < y:
                lesser += 1
    denom = len(a) * len(b)
    return (greater - lesser) / denom if denom else 0.0


def effect_label(delta: float) -> str:
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * p_value)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


if __name__ == "__main__":
    main()
