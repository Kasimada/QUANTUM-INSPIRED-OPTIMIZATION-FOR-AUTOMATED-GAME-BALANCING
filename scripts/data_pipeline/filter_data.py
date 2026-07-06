import sys

import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def filter_champions(input_csv, output_csv, count_per_class=15):
    print(f"Đang đọc dữ liệu gốc từ: {input_csv}")
    df = pd.read_csv(input_csv)

    # Chuẩn hóa nếu có giá trị rỗng
    df["herotype"] = df["herotype"].fillna("Fighter")

    sampled_dfs = []
    types = ["Fighter", "Mage", "Assassin", "Marksman", "Tank", "Support"]

    for t in types:
        class_df = df[df["herotype"] == t]
        actual_count = len(class_df)
        if actual_count > count_per_class:
            class_df = class_df.sample(n=count_per_class, random_state=42)
            print(
                f"  - {t}: Đã lấy ngẫu nhiên {count_per_class} tướng (từ {actual_count} tướng)."
            )
        else:
            print(
                f"  - {t}: Lấy toàn bộ {actual_count} tướng (không đủ {count_per_class})."
            )
        sampled_dfs.append(class_df)

    final_df = pd.concat(sampled_dfs)

    # Xáo trộn lại thứ tự
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    final_df.to_csv(output_csv, index=False)
    print(f"\n[Hoàn tất] Đã lưu file {output_csv} với tổng cộng {len(final_df)} tướng.")


if __name__ == "__main__":
    # Đặt count_per_class=15 (tổng 90 tướng) để cân bằng giữa tốc độ và độ đa dạng
    filter_champions(
        "080725_LoL_champion_data-selected-columns.csv",
        "subset_champion_data.csv",
        count_per_class=15,
    )
