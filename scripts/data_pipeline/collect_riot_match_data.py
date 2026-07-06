from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "riot_match_data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect official Riot Match-V5 match JSON files for LoL empirical "
            "win/pick/ban/stat aggregation."
        )
    )
    parser.add_argument(
        "--platform", default="vn2", help="Platform route, e.g. vn2, kr, na1."
    )
    parser.add_argument(
        "--regional", default="sea", help="Regional route, e.g. sea, asia, americas."
    )
    parser.add_argument("--queue", default="RANKED_SOLO_5x5")
    parser.add_argument(
        "--tier", default="challenger", choices=["challenger", "grandmaster", "master"]
    )
    parser.add_argument(
        "--seed-puuids",
        type=Path,
        default=None,
        help="Optional CSV/TXT containing one PUUID per line.",
    )
    parser.add_argument(
        "--players", type=int, default=40, help="Maximum seed players to use."
    )
    parser.add_argument("--matches-per-player", type=int, default=20)
    parser.add_argument("--match-limit", type=int, default=500)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--api-key-env", default="RIOT_API_KEY")
    parser.add_argument(
        "--sleep", type=float, default=1.2, help="Delay between API calls."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate the API key and route with lightweight Riot endpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"Missing Riot API key. Set it first, for example:\n"
            f'$env:{args.api_key_env}="RGAPI-..."'
        )

    if args.check_only:
        check_api_access(args, api_key)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    match_dir = args.out / "matches_json"
    match_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_puuids:
        puuids = read_seed_puuids(args.seed_puuids)[: args.players]
    else:
        puuids = fetch_ranked_puuids(args, api_key)[: args.players]

    write_csv(args.out / "seed_puuids.csv", [{"puuid": puuid} for puuid in puuids])
    print(f"[Riot API] Seed PUUIDs: {len(puuids)}")

    seen_match_ids = set()
    rows = []
    for index, puuid in enumerate(puuids, start=1):
        print(f"[Riot API] Fetching match ids for player {index}/{len(puuids)}")
        match_ids = fetch_match_ids(args, api_key, puuid)
        for match_id in match_ids:
            if len(seen_match_ids) >= args.match_limit:
                break
            if match_id in seen_match_ids:
                continue
            seen_match_ids.add(match_id)
            path = match_dir / f"{match_id}.json"
            if not path.exists():
                payload = riot_get(
                    f"https://{args.regional}.api.riotgames.com/lol/match/v5/matches/{match_id}",
                    api_key,
                    sleep=args.sleep,
                )
                path.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
            rows.append({"match_id": match_id, "path": str(path)})
        if len(seen_match_ids) >= args.match_limit:
            break

    write_csv(args.out / "match_ids.csv", rows)
    metadata = {
        "platform": args.platform,
        "regional": args.regional,
        "queue": args.queue,
        "tier": args.tier,
        "players": len(puuids),
        "matches": len(rows),
        "matches_per_player": args.matches_per_player,
        "source": "Riot Games API: League-V4 and Match-V5",
    }
    (args.out / "collection_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Riot API] Wrote {len(rows)} match files to {match_dir}")


def read_seed_puuids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in text:
        value = line.strip().split(",")[0]
        if value and value.lower() != "puuid":
            out.append(value)
    return out


def fetch_ranked_puuids(args: argparse.Namespace, api_key: str) -> list[str]:
    url = f"https://{args.platform}.api.riotgames.com/lol/league/v4/{args.tier}leagues/by-queue/{args.queue}"
    payload = riot_get(url, api_key, sleep=args.sleep)
    puuids = []
    for entry in payload.get("entries", []):
        if entry.get("puuid"):
            puuids.append(entry["puuid"])
        elif entry.get("summonerId"):
            summoner = riot_get(
                f"https://{args.platform}.api.riotgames.com/lol/summoner/v4/summoners/{entry['summonerId']}",
                api_key,
                sleep=args.sleep,
            )
            if summoner.get("puuid"):
                puuids.append(summoner["puuid"])
    return puuids


def check_api_access(args: argparse.Namespace, api_key: str) -> None:
    checks = [
        (
            "Platform status",
            f"https://{args.platform}.api.riotgames.com/lol/status/v4/platform-data",
        ),
        (
            "League challenger list",
            f"https://{args.platform}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{args.queue}",
        ),
    ]
    for name, url in checks:
        try:
            payload = riot_get(url, api_key, sleep=args.sleep)
            if isinstance(payload, dict):
                keys = ", ".join(list(payload)[:6])
            else:
                keys = type(payload).__name__
            print(f"[OK] {name}: {url} ({keys})")
        except Exception as exc:
            print(f"[FAIL] {name}: {url}")
            print(f"       {exc}")


def fetch_match_ids(args: argparse.Namespace, api_key: str, puuid: str) -> list[str]:
    query = urlencode({"start": args.start, "count": args.matches_per_player})
    url = f"https://{args.regional}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?{query}"
    return riot_get(url, api_key, sleep=args.sleep)


def riot_get(url: str, api_key: str, sleep: float):
    headers = {
        "X-Riot-Token": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    request = Request(url, headers=headers)
    while True:
        try:
            with urlopen(request, timeout=45) as response:
                time.sleep(sleep)
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 429:
                retry_after = float(exc.headers.get("Retry-After", "10"))
                print(f"[Riot API] Rate limited. Sleeping {retry_after:.1f}s")
                time.sleep(retry_after)
                continue
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Riot API error {exc.code} for {url}: {body}") from exc


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
