from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
VERSION_URL = "https://ddragon.leagueoflegends.com/api/versions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download official Riot Data Dragon champion stats and convert them to the AQEA CSV format."
    )
    parser.add_argument(
        "--version", default="latest", help="Data Dragon version, or 'latest'."
    )
    parser.add_argument("--language", default="en_US")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "riot_datadragon_champion_stats.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = resolve_version(args.version)
    champion_url = (
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/"
        f"{args.language}/champion.json"
    )
    print(f"[Data Dragon] Version: {version}")
    print(f"[Data Dragon] Downloading: {champion_url}")
    with urlopen(champion_url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = []
    for champion in payload["data"].values():
        tags = champion.get("tags", [])
        rows.append(
            {
                "source": "Riot Data Dragon",
                "version": version,
                "id": champion.get("key", ""),
                "apiname": champion.get("id", ""),
                "name": champion.get("name", ""),
                "title": champion.get("title", ""),
                "difficulty": champion.get("info", {}).get("difficulty", ""),
                "herotype": tags[0] if tags else "Fighter",
                "alttype": tags[1] if len(tags) > 1 else "",
                "resource": champion.get("partype", ""),
                "stats": json.dumps(champion.get("stats", {}), ensure_ascii=False),
                "rangetype": infer_range_type(champion.get("stats", {})),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Data Dragon] Wrote {len(rows)} champions to {args.out}")


def resolve_version(version: str) -> str:
    if version != "latest":
        return version
    with urlopen(VERSION_URL, timeout=30) as response:
        versions = json.loads(response.read().decode("utf-8"))
    return versions[0]


def infer_range_type(stats: dict) -> str:
    return "Ranged" if float(stats.get("attackrange", 125)) >= 350 else "Melee"


if __name__ == "__main__":
    main()
