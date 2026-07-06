from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT / "riot_match_data" / "raw" / "matches_json"
DEFAULT_OUT = ROOT / "riot_match_data" / "processed" / "champion_match_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate Riot Match-V5 JSON files into champion-level win/pick/ban/stat metrics."
    )
    parser.add_argument("--matches", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(args.matches.glob("*.json"))
    if not files:
        raise SystemExit(f"No match JSON files found in {args.matches}")

    champion = defaultdict(make_metrics)
    total_matches = 0
    patch_counter = Counter()
    queue_counter = Counter()

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        info = payload.get("info", {})
        participants = info.get("participants", [])
        if not participants:
            continue

        total_matches += 1
        patch = str(info.get("gameVersion", "")).split(".")
        if len(patch) >= 2:
            patch_counter[".".join(patch[:2])] += 1
        queue_counter[str(info.get("queueId", ""))] += 1
        duration = float(info.get("gameDuration", 0.0))

        for participant in participants:
            name = participant.get("championName", "")
            if not name:
                continue
            row = champion[name]
            row["champion_id"] = participant.get("championId", "")
            row["champion_name"] = name
            row["games"] += 1
            row["wins"] += int(bool(participant.get("win", False)))
            row["kills"] += float(participant.get("kills", 0))
            row["deaths"] += float(participant.get("deaths", 0))
            row["assists"] += float(participant.get("assists", 0))
            row["damage_to_champions"] += float(
                participant.get("totalDamageDealtToChampions", 0)
            )
            row["damage_taken"] += float(participant.get("totalDamageTaken", 0))
            row["gold_earned"] += float(participant.get("goldEarned", 0))
            row["cs"] += float(participant.get("totalMinionsKilled", 0)) + float(
                participant.get("neutralMinionsKilled", 0)
            )
            row["duration"] += duration
            lane = (
                participant.get("teamPosition")
                or participant.get("individualPosition")
                or "UNKNOWN"
            )
            row["lane_counts"][lane] += 1

        for team in info.get("teams", []):
            for ban in team.get("bans", []):
                champion_id = int(ban.get("championId", -1))
                if champion_id > 0:
                    champion[f"id:{champion_id}"]["champion_id"] = champion_id
                    champion[f"id:{champion_id}"]["ban_count"] += 1

    rows = finalize_rows(champion, total_matches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "match_files": len(files),
        "matches_processed": total_matches,
        "patch_distribution": dict(patch_counter),
        "queue_distribution": dict(queue_counter),
    }
    (args.out.parent / "match_metrics_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Match Metrics] Wrote {len(rows)} champion rows to {args.out}")


def make_metrics() -> dict:
    return {
        "champion_id": "",
        "champion_name": "",
        "games": 0,
        "wins": 0,
        "ban_count": 0,
        "kills": 0.0,
        "deaths": 0.0,
        "assists": 0.0,
        "damage_to_champions": 0.0,
        "damage_taken": 0.0,
        "gold_earned": 0.0,
        "cs": 0.0,
        "duration": 0.0,
        "lane_counts": Counter(),
    }


def finalize_rows(champion: dict, total_matches: int) -> list[dict]:
    rows = []
    total_picks = max(1, total_matches * 10)
    total_bans = max(1, total_matches * 10)
    for _, metrics in champion.items():
        games = int(metrics["games"])
        bans = int(metrics["ban_count"])
        if games == 0 and bans == 0:
            continue
        lane_counts = metrics["lane_counts"]
        main_role = lane_counts.most_common(1)[0][0] if lane_counts else ""
        row = {
            "champion_id": metrics["champion_id"],
            "champion_name": metrics["champion_name"],
            "games": games,
            "wins": int(metrics["wins"]),
            "win_rate": safe_div(metrics["wins"], games),
            "pick_count": games,
            "pick_rate": games / total_picks,
            "ban_count": bans,
            "ban_rate": bans / total_bans,
            "presence_rate": (games + bans) / max(1, total_matches * 20),
            "avg_kills": safe_div(metrics["kills"], games),
            "avg_deaths": safe_div(metrics["deaths"], games),
            "avg_assists": safe_div(metrics["assists"], games),
            "avg_kda": safe_div(
                metrics["kills"] + metrics["assists"], max(metrics["deaths"], 1.0)
            ),
            "avg_damage_to_champions": safe_div(metrics["damage_to_champions"], games),
            "avg_damage_taken": safe_div(metrics["damage_taken"], games),
            "avg_gold_earned": safe_div(metrics["gold_earned"], games),
            "avg_cs": safe_div(metrics["cs"], games),
            "avg_game_duration": safe_div(metrics["duration"], games),
            "main_role": main_role,
            "role_distribution": json.dumps(dict(lane_counts), ensure_ascii=False),
        }
        rows.append(row)
    rows.sort(
        key=lambda item: (float(item["presence_rate"]), int(item["games"])),
        reverse=True,
    )
    return rows


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    main()
