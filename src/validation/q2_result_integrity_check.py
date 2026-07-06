from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

CONTINUOUS_ALGORITHMS = ["ga", "pso", "qea", "aqea", "nsga_ii", "balanced_qea"]
DISCRETE_ALGORITHMS = [
    "ga_discrete",
    "pso_discrete",
    "qea_discrete",
    "aqea_discrete",
    "nsga_ii_discrete",
    "balanced_qea_discrete",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a result folder is complete and paper-usable."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--mode", choices=["auto", "continuous", "discrete"], default="auto"
    )
    parser.add_argument(
        "--trials", type=int, default=None, help="Expected number of trials."
    )
    parser.add_argument(
        "--stdout", action="store_true", help="Print Markdown report to stdout."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_result(args.result_dir, args.mode, args.trials)
    if args.stdout:
        print(report)
        return
    out_path = args.result_dir / "q2_integrity_check.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


def check_result(result_dir: Path, mode: str, expected_trials: int | None) -> str:
    runs_path = result_dir / "runs.csv"
    summary_path = result_dir / "summary.csv"
    partial_runs = result_dir / "runs_partial.csv"
    partial_summary = result_dir / "summary_partial.csv"
    rows = read_csv(runs_path) if runs_path.exists() else []

    detected_mode = mode
    if detected_mode == "auto":
        name = result_dir.name.lower()
        if "discrete" in name:
            detected_mode = "discrete"
        elif rows and any(
            str(row.get("algorithm", "")).endswith("_discrete") for row in rows
        ):
            detected_mode = "discrete"
        else:
            detected_mode = "continuous"

    expected_algorithms = (
        DISCRETE_ALGORITHMS if detected_mode == "discrete" else CONTINUOUS_ALGORITHMS
    )
    if expected_trials is None:
        expected_trials = infer_trials(rows)
    expected_rows = (
        expected_trials * len(expected_algorithms) if expected_trials else None
    )

    counts = Counter(row.get("algorithm", "") for row in rows)
    missing_algorithms = [
        name for name in expected_algorithms if counts.get(name, 0) == 0
    ]
    uneven = sorted({count for count in counts.values() if count})
    has_required_files = runs_path.exists() and summary_path.exists()
    row_count_ok = expected_rows is not None and len(rows) == expected_rows
    alg_count_ok = (
        all(counts.get(name, 0) == expected_trials for name in expected_algorithms)
        if expected_trials
        else False
    )
    complete = has_required_files and row_count_ok and alg_count_ok

    lines = [
        "# Q2 Result Integrity Check",
        "",
        f"Folder: `{result_dir}`",
        "",
        f"Detected mode: `{detected_mode}`",
        f"Expected trials: `{expected_trials if expected_trials is not None else 'unknown'}`",
        f"Expected algorithms: `{', '.join(expected_algorithms)}`",
        "",
        "## Required Files",
        "",
        "| File | Exists |",
        "|---|---:|",
        f"| runs.csv | {yes_no(runs_path.exists())} |",
        f"| summary.csv | {yes_no(summary_path.exists())} |",
        f"| runs_partial.csv | {yes_no(partial_runs.exists())} |",
        f"| summary_partial.csv | {yes_no(partial_summary.exists())} |",
        "",
        "## Row Counts",
        "",
        f"- Actual `runs.csv` rows: `{len(rows)}`",
        f"- Expected rows: `{expected_rows if expected_rows is not None else 'unknown'}`",
        f"- Row count OK: `{yes_no(row_count_ok)}`",
        "",
        "## Algorithm Counts",
        "",
        "| Algorithm | Rows | Expected |",
        "|---|---:|---:|",
    ]
    for algorithm in expected_algorithms:
        lines.append(
            f"| {algorithm} | {counts.get(algorithm, 0)} | {expected_trials or 0} |"
        )
    for algorithm, count in sorted(counts.items()):
        if algorithm and algorithm not in expected_algorithms:
            lines.append(f"| {algorithm} | {count} | unexpected |")

    lines += [
        "",
        "## Verdict",
        "",
        f"Complete for paper use: `{yes_no(complete)}`",
        "",
    ]
    if missing_algorithms:
        lines.append(f"Missing algorithms: `{', '.join(missing_algorithms)}`")
        lines.append("")
    if len(uneven) > 1:
        lines.append(f"Uneven algorithm row counts detected: `{uneven}`")
        lines.append("")
    if not complete:
        lines.append(
            "Use this folder as diagnostic evidence only until the missing/uneven outputs are resolved."
        )
    else:
        lines.append(
            "This folder passes the structural integrity check. Statistical interpretation is still required."
        )
    lines.append("")
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def infer_trials(rows: list[dict[str, str]]) -> int | None:
    trials = set()
    for row in rows:
        value = row.get("trial")
        if value in (None, ""):
            continue
        try:
            trials.add(int(float(value)))
        except ValueError:
            pass
    return len(trials) if trials else None


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


if __name__ == "__main__":
    main()
