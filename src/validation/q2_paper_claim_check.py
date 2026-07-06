from __future__ import annotations

import argparse
import re
from pathlib import Path

RISK_PATTERNS = [
    (
        "overclaim_proof",
        r"\b(proves?|demonstrates?|guarantees?|universally superior)\b",
    ),
    (
        "live_lol_claim",
        r"\b(live League of Legends|fully balances|deployment engine)\b",
    ),
    ("match_10000", r"\b10,?000\b.*\b(match|matches|Match-V5)\b"),
    (
        "high_agreement",
        r"\b(high agreement|strong agreement|matches real-world meta)\b",
    ),
    ("completed_30_trials", r"\b30 independent trials\b|\b30-trial\b|\b30 trials\b"),
    ("stat_sig", r"\bstatistically significant\b|\bsignificance\b"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan a paper draft for risky claims.")
    parser.add_argument("paper", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.paper.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    print("# Q2 Paper Claim Check")
    print()
    print(f"File: `{args.paper}`")
    print()
    found = False
    for name, pattern in RISK_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        for idx, line in enumerate(lines, start=1):
            if regex.search(line):
                found = True
                print(f"- `{name}` line {idx}: {line.strip()}")
    if not found:
        print("No high-risk claim patterns found.")
    print()
    print(
        "Review context manually. This scanner is conservative and may flag safe planned/protocol language."
    )


if __name__ == "__main__":
    main()
