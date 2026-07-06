from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze LoL repeated optimizer runs.")
    parser.add_argument(
        "--runs",
        type=Path,
        default=ROOT / "results" / "lol_validation_tuned" / "runs.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "lol_validation_tuned" / "analysis.md",
    )
    parser.add_argument("--reference", default="aqea")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.runs)
    text = render_analysis(rows, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if "net_patch_pressure" in row and "abs_net_patch_pressure" not in row:
            row["abs_net_patch_pressure"] = str(abs(float(row["net_patch_pressure"])))
            
    # Merge nsga3 summary if it exists
    nsga3_path = path.parent / "nsga3_summary.csv"
    if nsga3_path.exists():
        with nsga3_path.open(newline="", encoding="utf-8") as f:
            nsga_rows = list(csv.DictReader(f))
            for nr in nsga_rows:
                # Merge into existing rows.
                # Find the baseline_fitness and other base metrics from trial 0 or a matching trial
                base_info = {}
                for r in rows:
                    if r["trial"] == nr["trial"] and r["seed"] == nr["seed"]:
                        base_info = r
                        break
                
                new_row = {
                    "trial": nr["trial"],
                    "seed": nr["seed"],
                    "algorithm": nr["algorithm"],
                    "fitness": nr["best_scalar_score"],
                    "baseline_fitness": base_info.get("baseline_fitness", "0"),
                    "baseline_rbi": base_info.get("baseline_rbi", "0"),
                    "baseline_mds": base_info.get("baseline_mds", "0"),
                }
                rows.append(new_row)
    return rows


def render_analysis(rows, reference="aqea"):
    reference = validate_rows(rows, reference)
    rows = normalize_rows(rows)
    algorithms = sorted({row["algorithm"] for row in rows})
    metrics = available_metrics(rows)
    trial_count = len(
        {int(row["trial"]) for row in rows if row["algorithm"] == reference}
    )
    lines = [
        "# LoL Optimizer Statistical Analysis",
        "",
        f"Reference algorithm: `{reference}`.",
        "",
        "P-values are computed from paired trials using:",
        "",
        "- `Wilcoxon p`: two-sided Wilcoxon signed-rank test.",
        "",
        f"Number of paired reference trials available: {trial_count}.",
        "",
        "| Baseline | Metric | Mean Difference (reference - baseline) | Wilcoxon p |",
        "|---|---|---:|---:|",
    ]
    for baseline in algorithms:
        if baseline == reference:
            continue
        for metric in metrics:
            differences = [
                a - b for a, b in paired_values(rows, reference, baseline, metric)
            ]
            if not differences:
                continue
            lines.append(
                f"| `{baseline}` | `{metric}` | {mean(differences):.6f} | "
                f"{format_p(wilcoxon_p(differences))} |"
            )

    lines.extend(render_baseline_lift(rows, reference))
    lines.extend(render_runtime_summary(rows))
    lines.extend(
        [
            "",
            "Interpretation: for `fitness`, `mds`, and `completion`, positive differences favor the reference algorithm. "
            "For `rbi`, `convergence_fe_95`, `constraint_violation`, `patch_magnitude`, `abs_net_patch_pressure`, and `runtime_seconds`, "
            "negative differences favor the reference algorithm. For `fes_per_sec`, positive differences indicate faster evaluation.",
        ]
    )
    return "\n".join(lines) + "\n"


def normalize_rows(rows):
    normalized = [dict(row) for row in rows]
    for row in normalized:
        if "net_patch_pressure" in row and "abs_net_patch_pressure" not in row:
            row["abs_net_patch_pressure"] = str(abs(float(row["net_patch_pressure"])))
    return normalized


def validate_rows(rows, reference):
    if not rows:
        raise ValueError("No rows found in runs.csv; cannot compute p-values.")
    algorithms = {row.get("algorithm") for row in rows}
    if reference not in algorithms:
        fallback = list(algorithms)[0]
        print(f"Warning: Reference algorithm `{reference}` not found. Falling back to `{fallback}`.")
        reference = fallback
    return reference


def available_metrics(rows):
    preferred = [
        "fitness",
        "rbi",
        "mds",
        "completion",
        "balance_score",
        "diversity_score",
        "convergence_fe_95",
        "convergence_speed",
        "abs_net_patch_pressure",
        "constraint_violation",
        "patch_magnitude",
        "runtime_seconds",
        "fes_per_sec",
    ]
    if not rows:
        return []
    keys = set(rows[0])
    return [metric for metric in preferred if metric in keys]


def render_baseline_lift(rows, reference):
    if not rows:
        return []
    reference_rows = [row for row in rows if row["algorithm"] == reference]
    if not reference_rows or "baseline_fitness" not in reference_rows[0]:
        return []
    base_fitness = float(reference_rows[0]["baseline_fitness"])
    base_rbi = float(reference_rows[0]["baseline_rbi"])
    base_mds = float(reference_rows[0]["baseline_mds"])
    ref_fitness = mean([float(row["fitness"]) for row in reference_rows])
    ref_rbi = mean([float(row["rbi"]) for row in reference_rows])
    ref_mds = mean([float(row["mds"]) for row in reference_rows])
    return [
        "",
        "## Reference-vs-Unbalanced Lift",
        "",
        "| Metric | Unbalanced | Reference Mean | Difference |",
        "|---|---:|---:|---:|",
        f"| `fitness` | {base_fitness:.6f} | {ref_fitness:.6f} | {ref_fitness - base_fitness:.6f} |",
        f"| `rbi` | {base_rbi:.6f} | {ref_rbi:.6f} | {ref_rbi - base_rbi:.6f} |",
        f"| `mds` | {base_mds:.6f} | {ref_mds:.6f} | {ref_mds - base_mds:.6f} |",
    ]

def render_runtime_summary(rows):
    if not rows:
        return []
    if "runtime_seconds" not in rows[0] or "fes_per_sec" not in rows[0]:
        return []
    
    algorithms = sorted({row["algorithm"] for row in rows})
    lines = [
        "",
        "## Runtime Summary",
        "",
        "| Algorithm | Mean Runtime (s) | Mean FEs/sec |",
        "|---|---:|---:|",
    ]
    for algo in algorithms:
        if algo == "unbalanced":
            continue
        algo_rows = [row for row in rows if row["algorithm"] == algo]
        if not algo_rows:
            continue
        mean_rt = mean([float(row["runtime_seconds"]) for row in algo_rows if "runtime_seconds" in row])
        mean_fps = mean([float(row["fes_per_sec"]) for row in algo_rows if "fes_per_sec" in row])
        lines.append(f"| `{algo}` | {mean_rt:.2f} | {mean_fps:.2f} |")
    return lines


def paired_values(rows, reference, baseline, metric):
    by_key = {(row["algorithm"], int(row["seed"])): row for row in rows}
    trials = sorted(
        int(row["seed"])
        for row in rows
        if row["algorithm"] == reference and (baseline, int(row["seed"])) in by_key
    )
    # pair strictly by seed
    return [
        (
            float(by_key[(reference, seed)][metric]),
            float(by_key[(baseline, seed)][metric]),
        )
        for seed in trials
        if metric in by_key[(reference, seed)] and metric in by_key[(baseline, seed)]
    ]


def mean(values):
    return sum(values) / len(values)


def wilcoxon_p(differences):
    non_zero = [value for value in differences if value != 0.0]
    n = len(non_zero)
    if n < 5:
        # Wilcoxon needs at least some samples, fallback or return 1.0 if not enough
        return 1.0
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(non_zero)
        return res.pvalue
    except ImportError:
        # Fallback to sign test if scipy not available
        positives = sum(1 for value in non_zero if value > 0)
        k = min(positives, n - positives)
        p = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
        return min(1.0, 2.0 * p)


def format_p(value):
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


if __name__ == "__main__":
    main()
