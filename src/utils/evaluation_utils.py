import argparse
import statistics
try:
    import numpy as np
except ImportError:
    np = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from src.algorithms.discrete_algorithms import PATCH_LEVELS
from src.evaluation.analyze_lol_results import render_analysis
from statistics import mean, stdev
import sys
import csv
from pathlib import Path

def configure_output_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass

class ContinuousReport:
    @staticmethod
    def configure_output_streams() -> None:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass

    @staticmethod
    def write_checkpoint(out_dir: Path, rows: list[dict]) -> None:
        ContinuousReport.write_csv(out_dir / "runs_partial.csv", rows)
        ContinuousReport.write_csv(out_dir / "summary_partial.csv", ContinuousReport.summarize(rows))

    @staticmethod
    def summarize(rows: list[dict]) -> list[dict]:
        summary = []
        for algorithm in sorted({row["algorithm"] for row in rows}):
            subset = [row for row in rows if row["algorithm"] == algorithm]
            best_row = max(subset, key=lambda r: float(r["fitness"]))
            out = {
                "algorithm": algorithm, 
                "trials": len(subset),
                "best_trial": int(best_row["trial"]) + 1,
                "best_fitness": float(best_row["fitness"])
            }
            for metric in [
                "fitness",
                "rbi",
                "mds",
                "completion",
                "balance_score",
                "diversity_score",
                "convergence_fe_95",
                "convergence_speed",
            ]:
                values = [float(row[metric]) for row in subset]
                out[f"{metric}_mean"] = mean(values)
                out[f"{metric}_stdev"] = stdev(values) if len(values) > 1 else 0.0
            summary.append(out)
        summary.sort(key=lambda row: row["fitness_mean"], reverse=True)
        return summary

    @staticmethod
    def select_best_trial(rows: list[dict]) -> int:
        by_trial = {}
        for row in rows:
            by_trial.setdefault(int(row["trial"]), 0.0)
            by_trial[int(row["trial"])] += float(row["fitness"])
        return max(by_trial, key=by_trial.get)

    @staticmethod
    def first_fe_at_fraction(hx: list[int], hy: list[float], fraction: float) -> int:
        if not hx or not hy:
            return 0
        final_best = max(hy)
        if final_best <= 0:
            return int(hx[-1])
        threshold = final_best * fraction
        for fe, value in zip(hx, hy):
            if value >= threshold:
                return int(fe)
        return int(hx[-1])

    @staticmethod
    def write_best_chart(path: Path, histories: list[dict], trial: int) -> None:
        selected = {
            item["algorithm"]: item for item in histories if int(item["trial"]) == trial
        }
        series = []
        colors = {
            "ga": "#2563EB",
            "pso": "#F59E0B",
            "qea": "#7C3AED",
            "aqea": "#DC2626",
            "nsga_ii": "#059669",
            "balanced_qea": "#DB2777",
        }
        for algo_name in ["ga", "pso", "qea", "aqea", "nsga_ii", "balanced_qea"]:
            if algo_name in selected:
                # Format name: nsga_ii -> NSGA-II, balanced_qea -> Balanced-QEA
                display = (
                    algo_name.replace("nsga_ii", "NSGA-II")
                    .replace("balanced_qea", "Balanced-QEA")
                    .upper()
                )
                if display == "BALANCED-QEA":
                    display = "Balanced-QEA"
                if display == "GA_DISCRETE":
                    display = "GA"  # safety
                series.append(
                    (
                        display,
                        selected[algo_name]["hx"],
                        selected[algo_name]["hy"],
                        colors[algo_name],
                    )
                )
        ContinuousReport.write_convergence_svg(path, series)

    @staticmethod
    def write_convergence_svg(
        path: Path, series: list[tuple[str, list[int], list[float], str]]
    ) -> None:
        width, height = 920, 560
        left, right, top, bottom = 84, 32, 64, 72
        plot_w = width - left - right
        plot_h = height - top - bottom
        all_x = [x for _, xs, _, _ in series for x in xs]
        all_y = [y for _, _, ys, _ in series for y in ys]
        max_x = max(all_x) if all_x else 1
        min_y = min(all_y) if all_y else 0
        max_y = max(all_y) if all_y else 1
        if max_y == min_y:
            max_y += 1
    
        def sx(x):
            return left + (x / max_x) * plot_w
    
        def sy(y):
            return top + (1 - (y - min_y) / (max_y - min_y)) * plot_h
    
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}.small{font-size:12px;fill:#4B5563}.title{font-size:22px;font-weight:700}</style>",
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            '<text x="32" y="36" class="title">Convergence Curve: Game Balance Optimization</text>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9CA3AF"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9CA3AF"/>',
            f'<text x="{left + plot_w / 2 - 70}" y="{height - 24}" class="small">Function Evaluations (FEs)</text>',
            f'<text x="16" y="{top + plot_h / 2}" class="small" transform="rotate(-90 16 {top + plot_h / 2})">Fitness</text>',
        ]
        for tick in range(6):
            x = left + tick * plot_w / 5
            value = max_x * tick / 5
            lines.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#F3F4F6"/>'
            )
            lines.append(
                f'<text x="{x - 12:.2f}" y="{top + plot_h + 18}" class="small">{value:.0f}</text>'
            )
        for tick in range(5):
            y = top + tick * plot_h / 4
            value = max_y - (max_y - min_y) * tick / 4
            lines.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#F3F4F6"/>'
            )
            lines.append(
                f'<text x="{left - 56}" y="{y + 4:.2f}" class="small">{value:.2f}</text>'
            )
        legend_x = left + plot_w - 190
        for index, (name, xs, ys, color) in enumerate(series):
            if not xs or not ys:
                continue
            points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(xs, ys))
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>'
            )
            ly = top + 18 + index * 22
            lines.append(
                f'<rect x="{legend_x}" y="{ly - 10}" width="14" height="14" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{legend_x + 22}" y="{ly + 2}" class="small">{name}</text>'
            )
        lines.append("</svg>")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def write_analysis(path: Path, rows: list[dict], reference: str) -> None:
        path.write_text(render_analysis(rows, reference), encoding="utf-8")

    @staticmethod
    def write_markdown_table(path: Path, rows: list[dict], baseline: dict) -> None:
        lines = [
            "| Algorithm | Fitness Mean | Balance Score | Diversity Score | Convergence Speed | FE95 | Trials |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['algorithm']}` | {row['fitness_mean']:.4f} | {row['balance_score_mean']:.2f} | "
                f"{row['diversity_score_mean']:.2f} | {row['convergence_speed_mean']:.2f} | "
                f"{row['convergence_fe_95_mean']:.0f} | {row['trials']} |"
            )
        lines.append(
            f"| `unbalanced` | {baseline['fitness']:.4f} | {baseline['completion']:.2f} | "
            f"{baseline['mds']:.2f} | 0.00 | 0 | 1 |"
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def progress(iterable, **kwargs):
        if tqdm is None or not sys.stderr.isatty():
            return iterable
        return tqdm(iterable, **kwargs)

    @staticmethod
    def mean(values: list[float]) -> float:
        if np is not None:
            return float(np.mean(values))
        return statistics.mean(values)

    @staticmethod
    def stdev(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        if np is not None:
            return float(np.std(values, ddof=1))
        return statistics.stdev(values) if len(values) > 1 else 0.0

    @staticmethod
    def write_readme(
        path: Path,
        args: argparse.Namespace,
        rows: list[dict],
        baseline: dict,
        dataset_display_name: str,
        dataset_source_name: str,
        dataset_path: str,
        best_trial: int,
    ) -> None:
        lines = [
            f"# {dataset_display_name} Repeated Paper Experiment",
            "",
            "## Summary",
            "",
            f"- Dataset: {dataset_display_name}",
            f"- Source: {dataset_source_name}",
            f"- Dataset file: {dataset_path}",
            f"- Function Evaluations per algorithm per trial: {args.fes:,}",
            f"- Trials: {args.trials}",
            f"- Initial seed: {args.seed}",
            f"- Best trial used for convergence chart: {best_trial}",
            f"- Optional QEA ablation included: {args.include_qea}",
            "",
            "## Result Table",
            "",
            "| Algorithm | Fitness Mean | Balance Score | Diversity Score | Convergence Speed | FE95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['algorithm']}` | {row['fitness_mean']:.4f} | {row['balance_score_mean']:.2f} | "
                f"{row['diversity_score_mean']:.2f} | {row['convergence_speed_mean']:.2f} | "
                f"{row['convergence_fe_95_mean']:.0f} |"
            )
        lines.extend(
            [
                f"| `unbalanced` | {baseline['fitness']:.4f} | {baseline['completion']:.2f} | {baseline['mds']:.2f} | 0.00 | 0 |",
                "",
                "## Metric Definitions",
                "",
                "- `Balance Score`: `100 - RBI * 100`; higher is better.",
                "- `Diversity Score`: the normalized meta-diversity score (`MDS`); higher is better.",
                "- `FE95`: the first function-evaluation count where the algorithm reaches 95% of its final best fitness; lower is faster.",
                "- `Convergence Speed`: `100 * (1 - FE95 / max_FEs)`; higher is faster.",
                "- `QEA`: non-adaptive quantum-inspired ablation; included only when `--include-qea` is used.",
                "",
                "## Interpretation Rule",
                "",
                "This repeated run is stronger evidence than a single-seed 25K run. However, it still uses a LoL-inspired static-stat simulator, not the live League of Legends balance system.",
                "",
                "Safe claim:",
                "",
                "> The tested optimizers improve balance-oriented metrics over the unbalanced baseline in a LoL-inspired static-stat simulator. If QEA is included, the QEA-vs-AQEA comparison isolates the contribution of the adaptive mechanism.",
                "",
                "Avoid:",
                "",
                "> AQEA fully balances live League of Legends or universally outperforms classical optimizers.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

class DiscreteReport:
    @staticmethod
    def configure_output_streams() -> None:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except Exception:
                pass

    @staticmethod
    def write_checkpoint(out_dir: Path, rows: list[dict]) -> None:
        DiscreteReport.write_csv(out_dir / "runs_partial.csv", rows)
        DiscreteReport.write_csv(out_dir / "summary_partial.csv", DiscreteReport.summarize(rows))

    @staticmethod
    def summarize(rows: list[dict]) -> list[dict]:
        summary = []
        for algorithm in sorted({row["algorithm"] for row in rows}):
            subset = [row for row in rows if row["algorithm"] == algorithm]
            best_row = max(subset, key=lambda r: float(r["fitness"]))
            out = {
                "algorithm": algorithm, 
                "trials": len(subset),
                "best_trial": int(best_row["trial"]) + 1,
                "best_fitness": float(best_row["fitness"])
            }
            for metric in [
                "fitness",
                "rbi",
                "mds",
                "completion",
                "balance_score",
                "diversity_score",
                "convergence_fe_95",
                "convergence_speed",
                "net_patch_pressure",
                "constraint_violation",
                "patch_magnitude",
            ]:
                values = [float(row[metric]) for row in subset]
                out[f"{metric}_mean"] = mean(values)
                out[f"{metric}_stdev"] = stdev(values) if len(values) > 1 else 0.0
            summary.append(out)
        summary.sort(key=lambda row: row["fitness_mean"], reverse=True)
        return summary

    @staticmethod
    def select_best_trial(rows: list[dict]) -> int:
        by_trial = {}
        for row in rows:
            by_trial.setdefault(int(row["trial"]), 0.0)
            by_trial[int(row["trial"])] += float(row["fitness"])
        return max(by_trial, key=by_trial.get)

    @staticmethod
    def first_fe_at_fraction(hx: list[int], hy: list[float], fraction: float) -> int:
        if not hx or not hy:
            return 0
        final_best = max(hy)
        if final_best <= 0:
            return int(hx[-1])
        threshold = final_best * fraction
        for fe, value in zip(hx, hy):
            if value >= threshold:
                return int(fe)
        return int(hx[-1])

    @staticmethod
    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def write_analysis(path: Path, rows: list[dict], reference: str) -> None:
        path.write_text(render_analysis(rows, reference), encoding="utf-8")

    @staticmethod
    def write_markdown_table(path: Path, rows: list[dict], baseline: dict) -> None:
        lines = [
            "| Algorithm | Fitness Mean | Balance Score | Diversity Score | Conv. Speed | Net Patch Pressure | Budget Dev. | Patch Mag. | Trials |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['algorithm']}` | {row['fitness_mean']:.4f} | {row['balance_score_mean']:.2f} | "
                f"{row['diversity_score_mean']:.2f} | {row['convergence_speed_mean']:.2f} | "
                f"{row['net_patch_pressure_mean']:.1f} | {row['constraint_violation_mean']:.4f} | "
                f"{row['patch_magnitude_mean']:.4f} | {row['trials']} |"
            )
        lines.append(
            f"| `unbalanced` | {baseline['fitness']:.4f} | {baseline['balance_score']:.2f} | "
            f"{baseline['diversity_score']:.2f} | 0.00 | {baseline['net_patch_pressure']:.1f} | "
            f"{baseline['constraint_violation']:.4f} | {baseline['patch_magnitude']:.4f} | 1 |"
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def write_readme(
        path: Path,
        args: argparse.Namespace,
        rows: list[dict],
        baseline: dict,
        dataset_display_name: str,
        dataset_source_name: str,
        dataset_path: str,
        best_trial: int,
    ) -> None:
        lines = [
            "# LoL/Riot Discrete Stat Tuning Experiment",
            "",
            "## Summary",
            "",
            f"- Dataset: {dataset_display_name}",
            f"- Source: {dataset_source_name}",
            f"- Dataset file: {dataset_path}",
            f"- Function Evaluations per algorithm per trial: {args.fes:,}",
            f"- Trials: {args.trials}",
            f"- Initial seed: {args.seed}",
            f"- Best trial used for convergence chart: {best_trial}",
            f"- Patch levels: {PATCH_LEVELS}",
            "- Constraint: net patch pressure should remain close to zero; broad buff-only or nerf-only patches are penalized.",
            f"- Optional QEA ablation included: {args.include_qea}",
            "- Mathematical specification: see `METHODOLOGY_FORMULAS.md` in the project root.",
            "",
            "## Result Table",
            "",
            "| Algorithm | Fitness Mean | Balance Score | Diversity Score | Conv. Speed | Net Patch Pressure | Budget Dev. | Patch Mag. |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['algorithm']}` | {row['fitness_mean']:.4f} | {row['balance_score_mean']:.2f} | "
                f"{row['diversity_score_mean']:.2f} | {row['convergence_speed_mean']:.2f} | "
                f"{row['net_patch_pressure_mean']:.1f} | {row['constraint_violation_mean']:.4f} | "
                f"{row['patch_magnitude_mean']:.4f} |"
            )
        lines.extend(
            [
                f"| `unbalanced` | {baseline['fitness']:.4f} | {baseline['balance_score']:.2f} | {baseline['diversity_score']:.2f} | 0.00 | {baseline['net_patch_pressure']:.1f} | {baseline['constraint_violation']:.4f} | {baseline['patch_magnitude']:.4f} |",
                "",
                "## Metric Definitions",
                "",
                "- `Balance Score`: `100 - RBI * 100`; higher is better.",
                "- `Diversity Score`: normalized meta-diversity score (`MDS`); higher is better.",
                "- `Net Patch Pressure`: sum of discrete patch steps, where nerfs are negative and buffs are positive.",
                "- `Budget Dev.`: absolute net patch pressure normalized by maximum possible pressure; lower is better.",
                "- `Patch Mag.`: total absolute patch movement normalized by maximum possible pressure; lower means a smaller patch.",
                "- `Convergence Speed`: `100 * (1 - FE95 / max_FEs)`; higher is faster.",
                "",
                "## Interpretation Rule",
                "",
                "This experiment tests a discrete patch-like setting. AQEA is the proposed method; QEA is an optional ablation only. Any superiority claim must be based on repeated-run summaries, not a single seed.",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def write_best_chart(path: Path, histories: list[dict], trial: int) -> None:
        selected = [item for item in histories if int(item["trial"]) == trial]
        series = []
        colors = {
            "ga_discrete": "#2563EB",
            "pso_discrete": "#F59E0B",
            "qea_discrete": "#7C3AED",
            "aqea_discrete": "#DC2626",
            "nsga_ii_discrete": "#059669",
            "balanced_qea_discrete": "#DB2777",
        }
        for item in selected:
            series.append(
                (
                    item["algorithm"],
                    item["hx"],
                    item["hy"],
                    colors.get(item["algorithm"], "#111827"),
                )
            )
        DiscreteReport.write_convergence_png(path, series)

    @staticmethod
    def write_convergence_png(
        path: Path, series: list[tuple[str, list[int], list[float], str]]
    ) -> None:
        from PIL import Image, ImageDraw
    
        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = 1100, 640
        left, right, top, bottom = 88, 40, 58, 82
        plot_w = width - left - right
        plot_h = height - top - bottom
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        title_font = DiscreteReport.get_font(18)
        label_font = DiscreteReport.get_font(15)
        tick_font = DiscreteReport.get_font(12)
        legend_font = DiscreteReport.get_font(13)
    
        all_x = [x for _, xs, _, _ in series for x in xs]
        all_y = [y for _, _, ys, _ in series for y in ys]
        max_x = max(all_x) if all_x else 1
        min_y = min(all_y) if all_y else 0
        max_y = max(all_y) if all_y else 1
        padding = (max_y - min_y) * 0.08 if max_y > min_y else 1
        min_y -= padding
        max_y += padding
    
        def sx(x):
            return left + (x / max_x) * plot_w
    
        def sy(y):
            return top + (1 - (y - min_y) / (max_y - min_y)) * plot_h
    
        draw.text(
            (width / 2 - 250, 20),
            "Discrete Stat Tuning Convergence Curve",
            fill="black",
            font=title_font,
        )
        for tick in range(6):
            x = left + tick * plot_w / 5
            draw.line((x, top, x, top + plot_h), fill="#B8B8B8", width=1)
            draw.text(
                (x - 18, top + plot_h + 12),
                f"{max_x * tick / 5:.0f}",
                fill="black",
                font=tick_font,
            )
        for tick in range(6):
            y = top + tick * plot_h / 5
            draw.line((left, y, left + plot_w, y), fill="#B8B8B8", width=1)
            value = max_y - (max_y - min_y) * tick / 5
            draw.text((left - 60, y - 8), f"{value:.2f}", fill="black", font=tick_font)
    
        draw.rectangle((left, top, left + plot_w, top + plot_h), outline="black", width=1)
        draw.text(
            (width / 2 - 120, height - 36),
            "Function Evaluations (FEs)",
            fill="black",
            font=label_font,
        )
        draw.text(
            (16, height / 2 + 120),
            "Multi-objective Fitness",
            fill="black",
            font=label_font,
            anchor="ls",
        )
    
        dash_styles = {
            "ga_discrete": (8, 5),
            "pso_discrete": (10, 4, 2, 4),
            "qea_discrete": (5, 4),
            "aqea_discrete": None,
            "nsga_ii_discrete": (4, 4),
            "balanced_qea_discrete": (15, 5),
        }
        for name, xs, ys, color in series:
            points = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
            draw_polyline(draw, points, color, 4, dash_styles.get(name))
    
        legend_x, legend_y = width - 315, height - 165
        legend_h = 30 + 28 * len(series)
        draw.rounded_rectangle(
            (legend_x, legend_y, width - 28, legend_y + legend_h),
            radius=4,
            fill="white",
            outline="#D1D5DB",
        )
        for i, (name, _, _, color) in enumerate(series):
            y = legend_y + 24 + i * 28
            draw_polyline(
                draw,
                [(legend_x + 18, y), (legend_x + 55, y)],
                color,
                3,
                dash_styles.get(name),
            )
            draw.text((legend_x + 66, y - 9), name, fill="black", font=legend_font)
    
        image.save(path)

    @staticmethod
    def get_font(size):
        from PIL import ImageFont
    
        for candidate in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
        return ImageFont.load_default()

    @staticmethod
    def draw_polyline(draw, points, color, width, dash=None):
        if len(points) < 2:
            return
        if dash is None:
            draw.line(points, fill=color, width=width)
            return
        for start, end in zip(points, points[1:]):
            draw_dashed_segment(draw, start, end, color, width, dash)

    @staticmethod
    def draw_dashed_segment(draw, start, end, color, width, dash):
        import math
    
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        distance = 0.0
        dash_index = 0
        draw_on = True
        while distance < length:
            segment = dash[dash_index % len(dash)]
            next_distance = min(length, distance + segment)
            if draw_on:
                draw.line(
                    (
                        (x1 + dx * distance, y1 + dy * distance),
                        (x1 + dx * next_distance, y1 + dy * next_distance),
                    ),
                    fill=color,
                    width=width,
                )
            distance = next_distance
            dash_index += 1
            draw_on = not draw_on

    @staticmethod
    def progress(iterable, **kwargs):
        if tqdm is None or not sys.stderr.isatty():
            return iterable
        return tqdm(iterable, **kwargs)

    @staticmethod
    def mean(values: list[float]) -> float:
        if np is not None:
            return float(np.mean(values))
        return statistics.mean(values)

    @staticmethod
    def stdev(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        if np is not None:
            return float(np.std(values, ddof=1))
        return statistics.stdev(values) if len(values) > 1 else 0.0


def draw_polyline(draw, points, color, width, dash=None):
    if len(points) < 2:
        return
    if dash is None:
        draw.line(points, fill=color, width=width)
        return
    for start, end in zip(points, points[1:]):
        draw_dashed_segment(draw, start, end, color, width, dash)


def draw_dashed_segment(draw, start, end, color, width, dash):
    import math

    x1, y1 = start
    x2, y2 = end
    length = math.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    distance = 0.0
    dash_index = 0
    draw_on = True
    while distance < length:
        segment = dash[dash_index % len(dash)]
        next_distance = min(length, distance + segment)
        if draw_on:
            px1 = x1 + dx * distance
            py1 = y1 + dy * distance
            px2 = x1 + dx * next_distance
            py2 = y1 + dy * next_distance
            draw.line([(px1, py1), (px2, py2)], fill=color, width=width)
        distance = next_distance
        dash_index += 1
        draw_on = not draw_on