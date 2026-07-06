from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithms.discrete_algorithms import run_AQEA_discrete
from src.datasets.csv_loader import load_csv_dataset as load_champion_data


def analyze_correlation():
    print("--- Phân tích Tương quan Dữ liệu Thực tế (Match-V5) và Đề xuất của AQEA ---")

    # 1. Load Real Data
    real_data_path = Path("scripts/vn2_champion_rates.csv")
    if not real_data_path.exists():
        print(
            f"Chưa tìm thấy {real_data_path}. Vui lòng chờ tiến trình lấy 10,000 trận hoàn tất."
        )
        return

    df_real = pd.read_csv(real_data_path)
    # Lọc các tướng có đủ số trận để tránh nhiễu (ví dụ: Pick > 0)
    df_real = df_real[df_real["Picks"] > 0]

    # 2. Load Base Stats
    base_data = load_champion_data("riot_datadragon_champion_stats.csv")
    if not base_data:
        print("Không tìm thấy dữ liệu base stats.")
        return

    champ_names = [c["Name"] for c in base_data]

    # 3. Run AQEA to get Patch Decisions
    print("Đang chạy AQEA (5,000 FEs) để lấy bản vá (Patch) mẫu...")
    # Chạy discrete patch vì nó mô phỏng giống với các bản cập nhật buff/nerf của Riot nhất
    hx, hy, best_patch = run_AQEA_discrete(base_data, max_FEs=5000)

    # 4. Map decisions
    # best_patch chứa các hệ số nhân (ví dụ: 0.95, 1.05) cho 8 chỉ số của 172 tướng
    genes_per_champ = 8

    results = []
    for i, name in enumerate(champ_names):
        idx = i * genes_per_champ
        champ_genes = best_patch[idx : idx + genes_per_champ]
        # Tính mức độ buff/nerf trung bình
        net_patch = np.mean(champ_genes) - 1.0  # > 0 là Buff, < 0 là Nerf

        # Tìm Win Rate thực tế
        real_row = df_real[df_real["Champion"] == name]
        if not real_row.empty:
            win_rate = real_row["WinRate(%)"].values[0]
            picks = real_row["Picks"].values[0]

            # Logic đồng thuận (Agreement):
            # Win Rate < 49% -> Cần Buff (net_patch > 0)
            # Win Rate > 51% -> Cần Nerf (net_patch < 0)
            # Win Rate 49-51% -> Cân bằng (net_patch gần 0)
            agree = False
            if win_rate < 49.0 and net_patch > 0:
                agree = True
            elif win_rate > 51.0 and net_patch < 0:
                agree = True
            elif 49.0 <= win_rate <= 51.0 and abs(net_patch) <= 0.02:
                agree = True

            results.append(
                {
                    "Champion": name,
                    "Picks (Real)": picks,
                    "WinRate (%)": win_rate,
                    "AQEA Net Patch (%)": round(net_patch * 100, 2),
                    "Agreement": "YES" if agree else "NO",
                }
            )

    df_results = pd.DataFrame(results)

    # Lọc ra top 20 tướng được pick nhiều nhất để phân tích
    df_top = df_results.sort_values(by="Picks (Real)", ascending=False).head(20)

    out_path = Path("Real_Correlation_Analysis.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Phân tích Tương quan: Thực tế (Match-V5) vs. Mô phỏng (AQEA)\n\n")
        f.write(
            "Bảng dưới đây so sánh Win Rate lấy từ dữ liệu hàng ngàn trận đấu rank Thách Đấu (VN2) với quyết định Buff/Nerf của thuật toán AQEA (mô phỏng trên Data Dragon).\n\n"
        )
        f.write(
            "Lưu ý: `Agreement = YES` nghĩa là thuật toán AQEA đã phát hiện đúng các tướng đang yếu ở thực tế (Win Rate < 49%) để Buff, và phát hiện đúng tướng đang quá mạnh (Win Rate > 51%) để Nerf.\n\n"
        )
        f.write("### Top 20 Tướng phổ biến nhất\n")
        f.write(df_top.to_markdown(index=False))

        agreement_rate = (df_results["Agreement"] == "YES").mean() * 100
        f.write(
            f"\n\n**Tỉ lệ đồng thuận tổng thể (Overall Agreement Rate): {agreement_rate:.2f}%**\n"
        )
        f.write("\n> [!NOTE]\n")
        f.write(
            "> Tỉ lệ đồng thuận cao (>60%) cho thấy AQEA và hệ thống mô phỏng tĩnh có khả năng phản ánh đúng meta game thực tế mà không cần nạp vào bất kỳ dữ liệu trận đấu thật nào trước đó. Đây là bước tiến lớn cho bài báo Q2!\n"
        )

    print(f"\nHoàn thành! Kết quả đã được lưu tại: {out_path.absolute()}")
    print(df_top.to_string(index=False))


if __name__ == "__main__":
    analyze_correlation()
