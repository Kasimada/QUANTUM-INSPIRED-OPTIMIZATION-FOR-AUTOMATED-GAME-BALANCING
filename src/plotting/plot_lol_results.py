from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent


METRICS = [
    ("fitness_mean", "Fitness", "higher"),
    ("rbi_mean", "Relative Balance Error (RBI)", "lower"),
    ("mds_mean", "Meta Diversity Score (MDS)", "higher"),
    ("completion_mean", "Balance Completion (%)", "higher"),
]


COLORS = {
    "ga": "#2563EB",
    "pso": "#F59E0B",
    "aqea": "#DC2626",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot LoL-inspired balance experiment summary as SVG."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "riot_lol_full" / "summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "riot_lol_full" / "lol_balance_stats.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_summary(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_svg(rows), encoding="utf-8")
    print(f"Wrote {args.out}")


def read_summary(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def render_svg(rows: list[dict]) -> str:
    width = 1040
    panel_height = 230
    header_height = 78
    footer_height = 42
    height = header_height + len(METRICS) * panel_height + footer_height
    left = 220
    right = 70
    chart_width = width - left - right
    bar_height = 30
    row_gap = 16

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}",
        ".title{font-size:24px;font-weight:700}",
        ".metric{font-size:18px;font-weight:700}",
        ".label{font-size:14px;font-weight:600}",
        ".small{font-size:13px;fill:#4B5563}",
        "</style>",
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="36" y="40" class="title">LoL-Inspired Riot Data Dragon Balance Experiment</text>',
        '<text x="36" y="62" class="small">GA vs PSO vs AQEA on official Riot static champion stats</text>',
    ]

    for metric_index, (field, title, direction) in enumerate(METRICS):
        y0 = header_height + metric_index * panel_height
        values = [float(row[field]) for row in rows]
        max_value = max(values)
        min_value = min(values)
        scale_max = max_value if max_value > 0 else 1.0
        best_value = max_value if direction == "higher" else min_value

        parts.extend(
            [
                f'<line x1="36" y1="{y0}" x2="{width - 36}" y2="{y0}" stroke="#E5E7EB"/>',
                f'<text x="36" y="{y0 + 32}" class="metric">{html.escape(title)}</text>',
                f'<text x="36" y="{y0 + 54}" class="small">Target: {direction} is better</text>',
            ]
        )

        for row_index, row in enumerate(rows):
            algorithm = row["algorithm"]
            value = float(row[field])
            y = y0 + 80 + row_index * (bar_height + row_gap)
            bar_width = 0 if scale_max == 0 else value / scale_max * chart_width
            color = COLORS.get(algorithm, "#111827")
            marker = " best" if value == best_value else ""
            stdev_field = field.replace("_mean", "_stdev")
            stdev = float(row.get(stdev_field, 0.0))

            parts.extend(
                [
                    f'<text x="52" y="{y + 21}" class="label">{html.escape(algorithm.upper())}</text>',
                    f'<rect x="{left}" y="{y}" width="{chart_width}" height="{bar_height}" rx="5" fill="#F3F4F6"/>',
                    f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" rx="5" fill="{color}"/>',
                    f'<text x="{left + bar_width + 10:.2f}" y="{y + 21}" class="small">{value:.4f} +/- {stdev:.4f}{marker}</text>',
                ]
            )

    parts.append(
        f'<text x="36" y="{height - 18}" class="small">Generated from repeated LoL-inspired experiment summary.csv.</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    main()
