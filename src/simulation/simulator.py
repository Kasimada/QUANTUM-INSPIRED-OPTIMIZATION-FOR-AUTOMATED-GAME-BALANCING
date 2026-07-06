import random

from src.simulation.type_chart import TYPE_CHART, TYPES


def run_simulation(population_stats, rng=None):
    rng = rng or random
    wins = {t1: {t2: 0 for t2 in TYPES} for t1 in TYPES}
    matches = {t1: {t2: 0 for t2 in TYPES} for t1 in TYPES}

    # Gom nhóm nhân vật theo Hệ (Type) để truy xuất nhanh
    champs_by_type = {t: [] for t in TYPES}
    for champ in population_stats:
        t = champ["Class"] if champ["Class"] in TYPES else "Fighter"
        champs_by_type[t].append(champ)

    total_duration = 0
    # 30 matches/pair x 36 pairs (6x6) = 1,080 simulated duels/evaluation.
    # In the paper, describe this as approximately 1,000 controlled simulated
    # matchups per candidate, not as live LoL matches.
    matches_per_pair = 30

    for t1 in TYPES:
        for t2 in TYPES:
            if not champs_by_type[t1] or not champs_by_type[t2]:
                continue

            for _ in range(matches_per_pair):
                champ1 = rng.choice(champs_by_type[t1])
                champ2 = rng.choice(champs_by_type[t2])

                mod1 = TYPE_CHART.get(t1, {}).get(t2, 1.0)
                mod2 = TYPE_CHART.get(t2, {}).get(t1, 1.0)

                hp1, hp2 = champ1["hp18"], champ2["hp18"]
                ad1, ad2 = champ1["ad18"], champ2["ad18"]
                arm1, arm2 = champ1["arm18"], champ2["arm18"]
                mr1, mr2 = champ1["mr18"], champ2["mr18"]
                as1, as2 = champ1["as18"], champ2["as18"]
                crit1, crit2 = champ1["crit18"], champ2["crit18"]
                hp_regen1, hp_regen2 = champ1["hp5_18"], champ2["hp5_18"]

                # [Support Proxy Mechanic] Đại diện cho utility heal/shield:
                if t1 == "Support":
                    hp_regen1 += champ1["mp5_18"] * 2.0
                if t2 == "Support":
                    hp_regen2 += champ2["mp5_18"] * 2.0

                mp1, mp2 = champ1["mp18"], champ2["mp18"]
                mp_regen1, mp_regen2 = champ1["mp5_18"], champ2["mp5_18"]

                crit_mult1 = 1.0 + crit1 * 0.75
                crit_mult2 = 1.0 + crit2 * 0.75

                phys_dps1 = (
                    ad1 * as1 * mod1 * crit_mult1 * (100.0 / (100.0 + max(0, arm2)))
                )
                phys_dps2 = (
                    ad2 * as2 * mod2 * crit_mult2 * (100.0 / (100.0 + max(0, arm1)))
                )

                # Giả lập sát thương kỹ năng dựa trên Mana để MR, MP, MP Regen có tác dụng
                magic_dps1 = (
                    (ad1 * 0.2 + mp1 * 0.01 + mp_regen1 * 2)
                    * mod1
                    * (100.0 / (100.0 + max(0, mr2)))
                    if mp1 > 0
                    else 0
                )
                magic_dps2 = (
                    (ad2 * 0.2 + mp2 * 0.01 + mp_regen2 * 2)
                    * mod2
                    * (100.0 / (100.0 + max(0, mr1)))
                    if mp2 > 0
                    else 0
                )

                dps1 = phys_dps1 + magic_dps1
                dps2 = phys_dps2 + magic_dps2

                # Lợi thế tầm đánh và tốc độ di chuyển
                range_adv1 = (champ1["attackrange"] - champ2["attackrange"]) / 1000.0
                range_adv2 = (champ2["attackrange"] - champ1["attackrange"]) / 1000.0

                accuracy1 = min(
                    0.95,
                    max(
                        0.40,
                        0.85
                        + (champ1["movespeed"] - champ2["movespeed"]) / 1000.0
                        + range_adv1,
                    ),
                )
                accuracy2 = min(
                    0.95,
                    max(
                        0.40,
                        0.85
                        + (champ2["movespeed"] - champ1["movespeed"]) / 1000.0
                        + range_adv2,
                    ),
                )

                turns = 0
                max_turns = 200

                # Lưu lại máu tối đa để tính % máu còn lại nếu đánh hết 200 lượt
                max_hp1 = hp1
                max_hp2 = hp2

                while hp1 > 0 and hp2 > 0 and turns < max_turns:
                    turns += 1
                    hp1 += hp_regen1
                    hp2 += hp_regen2

                    # [Assassin Burst Mechanic] Proxy cho burst damage kit của Sát Thủ:
                    # 5 lượt đầu, Sát Thủ nhận thêm 30% sát thương (combo kỹ năng mở đầu)
                    burst1 = 1.3 if (t1 == "Assassin" and turns <= 5) else 1.0
                    burst2 = 1.3 if (t2 == "Assassin" and turns <= 5) else 1.0

                    if rng.random() <= accuracy1:
                        hp2 -= dps1 * burst1 * rng.uniform(0.85, 1.15)
                    if rng.random() <= accuracy2:
                        hp1 -= dps2 * burst2 * rng.uniform(0.85, 1.15)

                matches[t1][t2] += 1
                if hp1 > 0 and hp2 <= 0:
                    wins[t1][t2] += 1
                elif hp2 > 0 and hp1 <= 0:
                    wins[t1][t2] += 0
                else:
                    # Hết 200 lượt chưa ai chết, người có % máu còn lại cao hơn sẽ thắng
                    hp1_pct = hp1 / max_hp1 if max_hp1 > 0 else 0
                    hp2_pct = hp2 / max_hp2 if max_hp2 > 0 else 0
                    if hp1_pct > hp2_pct:
                        wins[t1][t2] += 1
                    elif hp2_pct > hp1_pct:
                        wins[t1][t2] += 0
                    else:
                        wins[t1][t2] += 0.5

                total_duration += turns

    # Tính tỷ lệ thắng
    win_rate_matrix = {t1: {t2: 0.5 for t2 in TYPES} for t1 in TYPES}
    for t1 in TYPES:
        for t2 in TYPES:
            if matches[t1][t2] > 0:
                win_rate_matrix[t1][t2] = wins[t1][t2] / matches[t1][t2]

    total_matches = sum(sum(row.values()) for row in matches.values())
    avg_duration = total_duration / total_matches if total_matches > 0 else 0

    return win_rate_matrix, avg_duration
