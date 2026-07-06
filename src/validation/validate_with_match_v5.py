from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import scipy.stats as stats

from src.algorithms.algorithms import GENES_PER_CHAMP
from src.algorithms.discrete_algorithms import run_AQEA_discrete
from src.datasets.csv_loader import load_csv_dataset as load_champion_data

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate AQEA patches against Match-V5 empirical win rates."
    )
    parser.add_argument("--data", default="riot_datadragon_champion_stats.csv")
    parser.add_argument(
        "--match",
        type=Path,
        default=ROOT / "riot_match_data" / "processed" / "champion_balance_dataset.csv",
    )
    parser.add_argument("--fes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.match.exists():
        print(
            f"Error: Match dataset {args.match} not found. Please run collect and process scripts first."
        )
        return

    print("Loading empirical match data...")
    match_df = pd.read_csv(args.match)

    # Lọc ra các tướng có đủ số trận (ví dụ > 50 trận trong mẫu)
    match_df = match_df[match_df["games"] >= 50]
    if len(match_df) == 0:
        print("Not enough empirical data.")
        return

    print("Loading champion static data...")
    base_data = load_champion_data(args.data)
    if not base_data:
        print("No champion data loaded.")
        return

    print(f"Running AQEA for {args.fes} FEs to get a balanced patch...")
    random.seed(args.seed)
    hx, hy, best_chromosome = run_AQEA_discrete(base_data, args.fes)
    print(f"Best fitness achieved: {hy[-1]}")

    print("Extracting net patch pressure per champion...")
    results = []
    # best_chromosome là mảng độ dài len(base_data) * 8
    for i, champ in enumerate(base_data):
        champ_id = str(champ["id"])
        start_idx = i * GENES_PER_CHAMP
        end_idx = start_idx + GENES_PER_CHAMP
        multipliers = best_chromosome[start_idx:end_idx]

        # Áp lực bản vá (Net Patch Pressure): Tổng (multiplier - 1.0)
        net_pressure = sum(m - 1.0 for m in multipliers)

        results.append(
            {
                "champion_id": champ_id,
                "champion_name": champ["name"],
                "net_pressure": net_pressure,
            }
        )

    patch_df = pd.DataFrame(results)

    # Merge với match data
    merged = pd.merge(match_df, patch_df, on="champion_id", suffixes=("", "_patch"))

    print(f"\nMerged data for {len(merged)} champions.")

    # Tính tương quan Pearson giữa Tỷ lệ thắng (Win Rate) và Áp lực bản vá (Net Pressure)
    win_rates = merged["win_rate"].values
    pressures = merged["net_pressure"].values

    corr, p_value = stats.pearsonr(win_rates, pressures)

    print("==========================================================")
    print("               AQEA VALIDATION RESULTS                    ")
    print("==========================================================")
    print(f"Pearson Correlation (Win Rate vs Net Patch Pressure): {corr:.4f}")
    print(f"P-value: {p_value:.4f}")

    if corr < 0 and p_value < 0.05:
        print("\n=> SUCCESS! Negative correlation is significant.")
        print(
            "   This means AQEA implicitly nerfs champions with high empirical win rates"
        )
        print(
            "   and buffs champions with low empirical win rates, validating the simulator!"
        )
    elif corr < 0:
        print(
            "\n=> WEAK SUCCESS: Negative correlation observed, but p-value is not < 0.05."
        )
        print("   More FEs or more match data may be needed for stronger significance.")
    else:
        print("\n=> NO CORRELATION or POSITIVE CORRELATION.")
        print("   The proxy simulator might not fully align with Match-V5 reality.")


if __name__ == "__main__":
    main()
