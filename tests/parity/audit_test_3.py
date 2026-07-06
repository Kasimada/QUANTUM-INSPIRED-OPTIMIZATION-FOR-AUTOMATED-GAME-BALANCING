import os
import sys
import random
import numpy as np
import torch
import math

sys.path.insert(0, os.path.abspath("."))
from src.datasets.csv_loader import load_csv_dataset as load_champion_data
from src.simulation.type_chart import TYPE_CHART, TYPES, TARGET_MATRIX
from src.algorithms.algorithms import apply_chromosome
import src.simulation.simulator_tensor as st

# Pre-generate deterministic dice
rng = np.random.RandomState(42)
MAX_TURNS = 200
NUM_MATCHES = 36 * 30

R1_DICE = rng.uniform(0.0, 1.0, size=(MAX_TURNS, NUM_MATCHES))
R2_DICE = rng.uniform(0.0, 1.0, size=(MAX_TURNS, NUM_MATCHES))
VAR1_DICE = rng.uniform(0.85, 1.15, size=(MAX_TURNS, NUM_MATCHES))
VAR2_DICE = rng.uniform(0.85, 1.15, size=(MAX_TURNS, NUM_MATCHES))

def run_simulation_cpu(mutated_data):
    wins = {t1: {t2: 0 for t2 in TYPES} for t1 in TYPES}
    matches = {t1: {t2: 0 for t2 in TYPES} for t1 in TYPES}

    champs_by_type = {t: [] for t in TYPES}
    for champ in mutated_data:
        t = champ["Class"] if champ["Class"] in TYPES else "Fighter"
        champs_by_type[t].append(champ)

    matchups, type_pairs = st.get_precomputed_schedule(mutated_data)
    
    total_duration = 0
    match_idx = 0
    
    for (c1_idx, c2_idx), (t1, t2) in zip(matchups, type_pairs):
        champ1 = mutated_data[c1_idx]
        champ2 = mutated_data[c2_idx]

        mod1 = TYPE_CHART.get(t1, {}).get(t2, 1.0)
        mod2 = TYPE_CHART.get(t2, {}).get(t1, 1.0)

        hp1, hp2 = champ1["hp18"], champ2["hp18"]
        ad1, ad2 = champ1["ad18"], champ2["ad18"]
        arm1, arm2 = champ1["arm18"], champ2["arm18"]
        mr1, mr2 = champ1["mr18"], champ2["mr18"]
        as1, as2 = champ1["as18"], champ2["as18"]
        crit1, crit2 = champ1["crit18"], champ2["crit18"]
        hp_regen1, hp_regen2 = champ1["hp5_18"], champ2["hp5_18"]

        if t1 == "Support": hp_regen1 += champ1["mp5_18"] * 2.0
        if t2 == "Support": hp_regen2 += champ2["mp5_18"] * 2.0

        mp1, mp2 = champ1["mp18"], champ2["mp18"]
        mp_regen1, mp_regen2 = champ1["mp5_18"], champ2["mp5_18"]

        crit_mult1 = 1.0 + crit1 * 0.75
        crit_mult2 = 1.0 + crit2 * 0.75

        phys_dps1 = ad1 * as1 * mod1 * crit_mult1 * (100.0 / (100.0 + max(0, arm2)))
        phys_dps2 = ad2 * as2 * mod2 * crit_mult2 * (100.0 / (100.0 + max(0, arm1)))

        magic_dps1 = (ad1 * 0.2 + mp1 * 0.01 + mp_regen1 * 2) * mod1 * (100.0 / (100.0 + max(0, mr2))) if mp1 > 0 else 0
        magic_dps2 = (ad2 * 0.2 + mp2 * 0.01 + mp_regen2 * 2) * mod2 * (100.0 / (100.0 + max(0, mr1))) if mp2 > 0 else 0

        dps1 = phys_dps1 + magic_dps1
        dps2 = phys_dps2 + magic_dps2

        range_adv1 = (champ1["attackrange"] - champ2["attackrange"]) / 1000.0
        range_adv2 = (champ2["attackrange"] - champ1["attackrange"]) / 1000.0

        accuracy1 = min(0.95, max(0.40, 0.85 + (champ1["movespeed"] - champ2["movespeed"]) / 1000.0 + range_adv1))
        accuracy2 = min(0.95, max(0.40, 0.85 + (champ2["movespeed"] - champ1["movespeed"]) / 1000.0 + range_adv2))

        turns = 0
        max_hp1, max_hp2 = hp1, hp2

        while hp1 > 0 and hp2 > 0 and turns < MAX_TURNS:
            hp1 += hp_regen1
            hp2 += hp_regen2

            burst1 = 1.3 if (t1 == "Assassin" and turns < 5) else 1.0
            burst2 = 1.3 if (t2 == "Assassin" and turns < 5) else 1.0

            r1 = R1_DICE[turns, match_idx]
            r2 = R2_DICE[turns, match_idx]
            v1 = VAR1_DICE[turns, match_idx]
            v2 = VAR2_DICE[turns, match_idx]

            if r1 <= accuracy1: hp2 -= dps1 * burst1 * v1
            if r2 <= accuracy2: hp1 -= dps2 * burst2 * v2
            
            turns += 1

        matches[t1][t2] += 1
        if hp1 > 0 and hp2 <= 0: wins[t1][t2] += 1
        elif hp1 <= 0 and hp2 > 0: wins[t1][t2] += 0
        else:
            hp1_pct = hp1 / max_hp1 if max_hp1 > 0 else 0
            hp2_pct = hp2 / max_hp2 if max_hp2 > 0 else 0
            if hp1_pct > hp2_pct: wins[t1][t2] += 1
            elif hp2_pct > hp1_pct: wins[t1][t2] += 0
            else: wins[t1][t2] += 0.5

        total_duration += turns
        match_idx += 1

    win_rate_matrix = {t1: {t2: 0.5 for t2 in TYPES} for t1 in TYPES}
    for t1 in TYPES:
        for t2 in TYPES:
            if matches[t1][t2] > 0:
                win_rate_matrix[t1][t2] = wins[t1][t2] / matches[t1][t2]

    avg_duration = total_duration / NUM_MATCHES
    return win_rate_matrix, avg_duration

@torch.inference_mode()
def evaluate_gpu_deterministic(base_data, chromosome, device=None):
    if device is None: device = torch.device("cpu")
    T = st.init_tensors(base_data, device)
    
    chroms = torch.tensor([chromosome], dtype=torch.float32, device=device).view(1, len(base_data), 8)
    
    mults_linear = chroms[:, :, 0:7]
    mutated_linear = T["base_stats"][:, 0:7].unsqueeze(0) * mults_linear
    
    hp18 = mutated_linear[:, :, 0]
    mp18 = mutated_linear[:, :, 1]
    arm18 = mutated_linear[:, :, 2]
    mr18 = mutated_linear[:, :, 3]
    hp5_18 = mutated_linear[:, :, 4]
    mp5_18 = mutated_linear[:, :, 5]
    ad18 = mutated_linear[:, :, 6]
    
    as_mult = chroms[:, :, 7]
    as_base_mutated = T["base_stats"][:, 7].unsqueeze(0) * as_mult
    as_perlevel_mutated = T["base_stats"][:, 8].unsqueeze(0) * as_mult
    as18 = as_base_mutated * (1.0 + (as_perlevel_mutated / 100.0) * 17.0)
    
    idx1, idx2 = T["idx1"], T["idx2"]
    
    hp1, hp2 = hp18[:, idx1], hp18[:, idx2]
    ad1, ad2 = ad18[:, idx1], ad18[:, idx2]
    arm1, arm2 = arm18[:, idx1], arm18[:, idx2]
    mr1, mr2 = mr18[:, idx1], mr18[:, idx2]
    as1, as2 = as18[:, idx1], as18[:, idx2]
    hp_regen1, hp_regen2 = hp5_18[:, idx1], hp5_18[:, idx2]
    mp1, mp2 = mp18[:, idx1], mp18[:, idx2]
    mp_regen1, mp_regen2 = mp5_18[:, idx1], mp5_18[:, idx2]
    
    mod1, mod2 = T["mod1"], T["mod2"]
    
    hp_regen1 = torch.where(T["sup1_mask"], hp_regen1 + mp_regen1 * 2.0, hp_regen1)
    hp_regen2 = torch.where(T["sup2_mask"], hp_regen2 + mp_regen2 * 2.0, hp_regen2)
    
    phys_dps1 = ad1 * as1 * mod1 * T["crit_mult1"] * (100.0 / (100.0 + torch.clamp(arm2, min=0)))
    phys_dps2 = ad2 * as2 * mod2 * T["crit_mult2"] * (100.0 / (100.0 + torch.clamp(arm1, min=0)))
    
    magic_dps1 = torch.where(mp1 > 0, (ad1 * 0.2 + mp1 * 0.01 + mp_regen1 * 2) * mod1 * (100.0 / (100.0 + torch.clamp(mr2, min=0))), torch.zeros_like(mp1))
    magic_dps2 = torch.where(mp2 > 0, (ad2 * 0.2 + mp2 * 0.01 + mp_regen2 * 2) * mod2 * (100.0 / (100.0 + torch.clamp(mr1, min=0))), torch.zeros_like(mp2))
    
    dps1 = phys_dps1 + magic_dps1
    dps2 = phys_dps2 + magic_dps2
    
    max_hp1, max_hp2 = hp1.clone(), hp2.clone()
    
    turns = torch.zeros((1, NUM_MATCHES), dtype=torch.int32, device=device)
    active_mask = (hp1 > 0) & (hp2 > 0)
    
    acc1, acc2 = T["acc1"], T["acc2"]
    assassin1_mask, assassin2_mask = T["assassin1_mask"], T["assassin2_mask"]
    
    t_R1 = torch.tensor(R1_DICE, dtype=torch.float32, device=device).unsqueeze(1)
    t_R2 = torch.tensor(R2_DICE, dtype=torch.float32, device=device).unsqueeze(1)
    t_VAR1 = torch.tensor(VAR1_DICE, dtype=torch.float32, device=device).unsqueeze(1)
    t_VAR2 = torch.tensor(VAR2_DICE, dtype=torch.float32, device=device).unsqueeze(1)
    
    for t in range(1, MAX_TURNS + 1):
        if not active_mask.any(): break
        turns = torch.where(active_mask, turns + 1, turns)
        hp1 = torch.where(active_mask, hp1 + hp_regen1, hp1)
        hp2 = torch.where(active_mask, hp2 + hp_regen2, hp2)
        
        burst1 = torch.where(assassin1_mask & (turns <= 5), 1.3, 1.0)
        burst2 = torch.where(assassin2_mask & (turns <= 5), 1.3, 1.0)
        
        r1, r2 = t_R1[t-1], t_R2[t-1]
        hit1, hit2 = r1 <= acc1, r2 <= acc2
        var1, var2 = t_VAR1[t-1], t_VAR2[t-1]
        
        dmg1 = torch.where(active_mask & hit1, dps1 * burst1 * var1, torch.zeros_like(dps1))
        dmg2 = torch.where(active_mask & hit2, dps2 * burst2 * var2, torch.zeros_like(dps2))
        
        hp2 = hp2 - dmg1
        hp1 = hp1 - dmg2
        active_mask = (hp1 > 0) & (hp2 > 0)
        
    p1_wins = (hp1 > 0) & (hp2 <= 0)
    p2_wins = (hp2 > 0) & (hp1 <= 0)
    
    # FIXED LOGIC: Match CPU behavior exactly for both mutual destruction and timeout
    draws = ~(p1_wins | p2_wins)
    
    hp1_pct = torch.where(max_hp1 > 0, hp1 / max_hp1, torch.zeros_like(hp1))
    hp2_pct = torch.where(max_hp2 > 0, hp2 / max_hp2, torch.zeros_like(hp2))
    p1_score = p1_wins.float() + (draws & (hp1_pct > hp2_pct)).float() + (draws & (hp1_pct == hp2_pct)).float() * 0.5
    
    wins_tensor = p1_score.view(1, len(TYPES), len(TYPES), 30).sum(dim=-1)
    win_rate_matrix = wins_tensor / 30.0
    
    avg_duration = turns.float().sum() / NUM_MATCHES
    
    diff = T["target_matrix"] - win_rate_matrix
    total_abs_error = diff.abs().sum()
    rbi_error = total_abs_error / 36.0
    balance_score = torch.clamp(10.0 * (1.0 - (rbi_error / 0.30)), min=0.0)
    
    t_min, t_max, w_t = 35.0, 90.0, 0.2
    f2_penalty = torch.zeros_like(avg_duration)
    f2_penalty = torch.where(avg_duration < t_min, (t_min - avg_duration) * w_t, f2_penalty)
    f2_penalty = torch.where(avg_duration > t_max, (avg_duration - t_max) * w_t, f2_penalty)
    
    overall_win_rates = win_rate_matrix.sum(dim=2) / float(len(TYPES))
    total_wr = overall_win_rates.sum(dim=1, keepdim=True)
    p = torch.where(total_wr > 0, overall_win_rates / total_wr, torch.zeros_like(overall_win_rates))
    entropy = -torch.where(p > 0, p * torch.log2(p), torch.zeros_like(p)).sum(dim=1)
    
    entropy_score = (entropy / math.log2(len(TYPES))) * 5.0
    mds = (entropy / math.log2(len(TYPES))) * 100.0
    l2_penalty = ((chroms - 1.0) ** 2).view(1, -1).mean(dim=1)
    score = torch.clamp(1.0 * balance_score - 0.8 * f2_penalty + 0.5 * entropy_score - 0.2 * l2_penalty, min=0.0001)
    
    return win_rate_matrix.cpu().numpy()[0], float(avg_duration.cpu()), float(balance_score.cpu()), float(rbi_error.cpu()), float(mds.cpu()), float(entropy_score.cpu()), float(l2_penalty.cpu()), float(f2_penalty.cpu()), float(score.cpu())

def run_test():
    base_data = load_champion_data("riot_datadragon_champion_stats.csv")
    dim = len(base_data) * 8
    rng_cand = random.Random(42)
    chromosome = [rng_cand.uniform(0.7, 1.3) for _ in range(dim)]
    
    print("--- Running CPU Deterministic Mode ---")
    mutated_data = apply_chromosome(base_data, chromosome)
    cpu_wr_dict, cpu_dur = run_simulation_cpu(mutated_data)
    from src.fitness.fitness import calculate_fitness
    cpu_score, cpu_rbi, cpu_mds = calculate_fitness(cpu_wr_dict, cpu_dur, chromosome)
    
    # Need to manually get the other penalized metrics for print out
    # To do so I can run my own snippet matching fitness.py:
    target_matrix_list = [[TARGET_MATRIX[t1][t2] for t2 in TYPES] for t1 in TYPES]
    cpu_wr_mat = np.zeros((6, 6))
    for i, t1 in enumerate(TYPES):
        for j, t2 in enumerate(TYPES):
            cpu_wr_mat[i, j] = cpu_wr_dict[t1][t2]
            
    diff = target_matrix_list - cpu_wr_mat
    total_abs_error = np.sum(np.abs(diff))
    cpu_rbi2 = total_abs_error / 36.0
    cpu_bal = max(10.0 * (1.0 - (cpu_rbi2 / 0.30)), 0)
    
    t_min, t_max, w_t = 35.0, 90.0, 0.2
    cpu_f2 = 0.0
    if cpu_dur < t_min: cpu_f2 = (t_min - cpu_dur) * w_t
    elif cpu_dur > t_max: cpu_f2 = (cpu_dur - t_max) * w_t
    
    overall_win_rates = np.sum(cpu_wr_mat, axis=1) / 6.0
    total_wr = np.sum(overall_win_rates)
    p = np.where(total_wr > 0, overall_win_rates / total_wr, 0.0)
    entropy = -np.sum(np.where(p > 0, p * np.log2(p + 1e-12), 0.0))
    cpu_ent = (entropy / math.log2(6)) * 5.0
    
    chrom_arr = np.array(chromosome)
    cpu_l2 = np.mean((chrom_arr - 1.0) ** 2)
            
    print("--- Running GPU Deterministic Mode ---")
    gpu_wr_mat, gpu_dur, gpu_bal, gpu_rbi, gpu_mds, gpu_ent, gpu_l2, gpu_f2, gpu_score = evaluate_gpu_deterministic(base_data, chromosome)
    
    print("\n--- Correctness Diff ---")
    wr_diff = np.max(np.abs(cpu_wr_mat - gpu_wr_mat))
    dur_diff = abs(cpu_dur - gpu_dur)
    rbi_diff = abs(cpu_rbi - gpu_rbi)
    mds_diff = abs(cpu_mds - gpu_mds)
    bal_diff = abs(cpu_bal - gpu_bal)
    f2_diff = abs(cpu_f2 - gpu_f2)
    ent_diff = abs(cpu_ent - gpu_ent)
    l2_diff = abs(cpu_l2 - gpu_l2)
    score_diff = abs(cpu_score - gpu_score)
    
    print(f"WR Matrix Max Diff: {wr_diff:.8f}")
    print(f"Duration Diff:      {dur_diff:.8f}")
    print(f"RBI Diff:           {rbi_diff:.8f}")
    print(f"MDS Diff:           {mds_diff:.8f}")
    print(f"Balance Diff:       {bal_diff:.8f}")
    print(f"F2 Diff:            {f2_diff:.8f}")
    print(f"Entropy Diff:       {ent_diff:.8f}")
    print(f"L2 Diff:            {l2_diff:.8f}")
    print(f"Score Diff:         {score_diff:.8f}")
    
    passed = wr_diff <= 1e-5 and rbi_diff <= 1e-5 and mds_diff <= 1e-4 and score_diff <= 1e-4
    if passed:
        print("\nVERDICT: PASS")
    else:
        print("\nVERDICT: FAIL")

if __name__ == "__main__":
    run_test()
