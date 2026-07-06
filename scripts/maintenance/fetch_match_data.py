import os
import time

import pandas as pd
import requests

REGION_LEAGUE = "vn2"
REGION_MATCH = "sea"


def get_headers():
    api_key = os.environ.get("RIOT_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing RIOT_API_KEY environment variable. "
            "Set it before running this collector."
        )
    return {"X-Riot-Token": api_key}


def make_request(url):
    headers = get_headers()
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 10))
            print(f"Rate limited. Sleeping for {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            print(f"Error {response.status_code} on {url}")
            return None


def fetch_data(target_matches=100):
    print(f"1. Lấy danh sách người chơi Thách Đấu tại {REGION_LEAGUE.upper()}...")
    url_challenger = f"https://{REGION_LEAGUE}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5"
    challenger_data = make_request(url_challenger)

    if not challenger_data:
        print("Không thể lấy danh sách Challenger. Kiểm tra lại API Key.")
        return

    entries = challenger_data.get("entries", [])
    print(f"Tìm thấy {len(entries)} người chơi. Lấy PUUID của họ...")

    puuids = []
    # Lấy tất cả người chơi Thách Đấu
    for entry in entries:
        if "puuid" in entry:
            puuids.append(entry["puuid"])

    print(f"Đã lấy {len(puuids)} PUUID. Bắt đầu tìm Match IDs...")
    match_ids = set()
    for puuid in puuids:
        if len(match_ids) >= target_matches:
            break
        url_matches = f"https://{REGION_MATCH}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&type=ranked&start=0&count=100"
        m_ids = make_request(url_matches)
        if m_ids:
            for mid in m_ids:
                match_ids.add(mid)
                if len(match_ids) >= target_matches:
                    break
        time.sleep(1.2)

    print(
        f"Thu thập thành công {len(match_ids)} trận đấu độc nhất. Bắt đầu tải chi tiết trận đấu..."
    )

    champion_stats = {}  # championName -> {'picks': 0, 'wins': 0}
    total_matches = 0

    for mid in list(match_ids)[:target_matches]:
        url_match = (
            f"https://{REGION_MATCH}.api.riotgames.com/lol/match/v5/matches/{mid}"
        )
        match_data = make_request(url_match)
        if not match_data:
            time.sleep(1)
            continue

        info = match_data.get("info")
        if not info:
            continue

        total_matches += 1
        participants = info.get("participants", [])

        for p in participants:
            c_name = p.get("championName")
            win = p.get("win")

            if c_name not in champion_stats:
                champion_stats[c_name] = {"picks": 0, "wins": 0}

            champion_stats[c_name]["picks"] += 1
            if win:
                champion_stats[c_name]["wins"] += 1

        # In tiến độ
        if total_matches % 10 == 0:
            print(f"Đã tải {total_matches}/{target_matches} trận...")

        time.sleep(1.2)  # Chờ 1.2s mỗi request để tránh limit 100 req/2 min

    print("Hoàn thành tải dữ liệu. Bắt đầu tổng hợp tỉ lệ...")

    # Tính toán
    results = []
    for c_name, stats in champion_stats.items():
        picks = stats["picks"]
        wins = stats["wins"]
        win_rate = (wins / picks) * 100 if picks > 0 else 0
        pick_rate = (picks / (total_matches * 10)) * 100  # 10 người mỗi trận
        results.append(
            {
                "Champion": c_name,
                "Picks": picks,
                "Wins": wins,
                "WinRate(%)": round(win_rate, 2),
                "PickRate(%)": round(pick_rate, 2),
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values(by="Picks", ascending=False)

    out_file = os.path.join(os.path.dirname(__file__), "vn2_champion_rates.csv")
    df.to_csv(out_file, index=False)
    print(f"\nĐã lưu kết quả tại: {out_file}")
    print(df.head(15).to_string())


if __name__ == "__main__":
    # Lấy 10000 trận theo yêu cầu của user
    fetch_data(target_matches=10000)
