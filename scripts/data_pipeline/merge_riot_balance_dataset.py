from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_STATIC = ROOT / "riot_datadragon_champion_stats.csv"
DEFAULT_MATCH = ROOT / "riot_match_data" / "processed" / "champion_match_metrics.csv"
DEFAULT_OUT = ROOT / "riot_match_data" / "processed" / "champion_balance_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Riot Data Dragon static champion stats with derived Riot Match-V5 champion metrics."
    )
    parser.add_argument("--static", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--match", type=Path, default=DEFAULT_MATCH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    static_df = pd.read_csv(args.static)
    match_df = pd.read_csv(args.match)

    static_df["champion_id"] = static_df["id"].astype(str)
    match_df["champion_id"] = match_df["champion_id"].astype(str)
    merged = static_df.merge(
        match_df, on="champion_id", how="left", suffixes=("", "_match")
    )

    numeric_defaults = [
        "games",
        "wins",
        "win_rate",
        "pick_count",
        "pick_rate",
        "ban_count",
        "ban_rate",
        "presence_rate",
        "avg_kills",
        "avg_deaths",
        "avg_assists",
        "avg_kda",
        "avg_damage_to_champions",
        "avg_damage_taken",
        "avg_gold_earned",
        "avg_cs",
        "avg_game_duration",
    ]
    for column in numeric_defaults:
        if column in merged:
            merged[column] = merged[column].fillna(0)

    for column in ["main_role", "role_distribution", "champion_name"]:
        if column in merged:
            merged[column] = merged[column].fillna("")

    merged["empirical_sampled"] = merged["games"].astype(float) > 0
    merged["win_rate_deviation"] = (merged["win_rate"].astype(float) - 0.5).abs()
    merged["meta_dominance_score"] = merged["pick_rate"].astype(float) + merged[
        "ban_rate"
    ].astype(float)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"[Merge] Static rows: {len(static_df)}")
    print(f"[Merge] Match metric rows: {len(match_df)}")
    print(f"[Merge] Merged rows: {len(merged)}")
    print(f"[Merge] Wrote {args.out}")


if __name__ == "__main__":
    main()
