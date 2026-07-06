import torch
import random
from src.simulation.type_chart import TYPE_CHART, TYPES

_PRECOMPUTED_SCHEDULE = None
_TENSORS_CACHE = {}

def get_precomputed_schedule(base_data, seed=20260612, matches_per_pair=30):
    global _PRECOMPUTED_SCHEDULE
    if _PRECOMPUTED_SCHEDULE is not None:
        return _PRECOMPUTED_SCHEDULE

    rng = random.Random(seed)
    
    champs_by_type = {t: [] for t in TYPES}
    for i, champ in enumerate(base_data):
        t = champ["Class"] if champ["Class"] in TYPES else "Fighter"
        champs_by_type[t].append(i)

    matchups = []
    type_pairs = []
    valid_pairs = []
    
    for t1 in TYPES:
        for t2 in TYPES:
            is_valid = bool(champs_by_type[t1] and champs_by_type[t2])
            valid_pairs.append(is_valid)
            
            for _ in range(matches_per_pair):
                c1_idx = rng.choice(champs_by_type[t1]) if is_valid else 0
                c2_idx = rng.choice(champs_by_type[t2]) if is_valid else 0
                matchups.append((c1_idx, c2_idx))
                type_pairs.append((t1, t2))
                
    _PRECOMPUTED_SCHEDULE = (matchups, type_pairs, valid_pairs)
    return _PRECOMPUTED_SCHEDULE

def init_tensors(base_data, device):
    global _TENSORS_CACHE
    if "base_stats" in _TENSORS_CACHE:
        return _TENSORS_CACHE

    num_champs = len(base_data)
    
    # 0: hp18_base
    # 1: mp18_base
    # 2: arm18_base
    # 3: mr18_base
    # 4: hp5_18_base
    # 5: mp5_18_base
    # 6: ad18_base
    # 7: as_base_raw
    # 8: as_perlevel_raw
    # 9: crit18
    # 10: movespeed
    # 11: attackrange
    
    base_stats = torch.zeros((num_champs, 12), dtype=torch.float32, device=device)
    
    for i, c in enumerate(base_data):
        base_stats[i, 0] = c.get("hp", 0) + c.get("hpperlevel", 0) * 17
        base_stats[i, 1] = c.get("mp", 0) + c.get("mpperlevel", 0) * 17
        base_stats[i, 2] = c.get("armor", 0) + c.get("armorperlevel", 0) * 17
        base_stats[i, 3] = c.get("spellblock", 0) + c.get("spellblockperlevel", 0) * 17
        base_stats[i, 4] = (c.get("hpregen", 0) + c.get("hpregenperlevel", 0) * 17) / 5.0
        base_stats[i, 5] = (c.get("mpregen", 0) + c.get("mpregenperlevel", 0) * 17) / 5.0
        base_stats[i, 6] = c.get("attackdamage", 0) + c.get("attackdamageperlevel", 0) * 17
        base_stats[i, 7] = c.get("attackspeed", 0)
        base_stats[i, 8] = c.get("attackspeedperlevel", 0)
        base_stats[i, 9] = c.get("crit", 0) / 100.0 + (c.get("critperlevel", 0) / 100.0) * 17
        base_stats[i, 10] = c.get("movespeed", 0)
        base_stats[i, 11] = c.get("attackrange", 0)
        
    _TENSORS_CACHE["base_stats"] = base_stats
    
    from src.simulation.type_chart import TARGET_MATRIX
    target_matrix_list = [[TARGET_MATRIX[t1][t2] for t2 in TYPES] for t1 in TYPES]
    _TENSORS_CACHE["target_matrix"] = torch.tensor(target_matrix_list, dtype=torch.float32, device=device).unsqueeze(0)
    
    matchups, type_pairs, valid_pairs = get_precomputed_schedule(base_data)
    num_matches = len(matchups)
    
    _TENSORS_CACHE["valid_mask"] = torch.tensor(valid_pairs, dtype=torch.bool, device=device).view(1, len(TYPES), len(TYPES))
    
    idx1 = torch.tensor([m[0] for m in matchups], dtype=torch.long, device=device)
    idx2 = torch.tensor([m[1] for m in matchups], dtype=torch.long, device=device)
    _TENSORS_CACHE["idx1"] = idx1
    _TENSORS_CACHE["idx2"] = idx2
    
    mod1_list = [TYPE_CHART.get(p[0], {}).get(p[1], 1.0) for p in type_pairs]
    mod2_list = [TYPE_CHART.get(p[1], {}).get(p[0], 1.0) for p in type_pairs]
    _TENSORS_CACHE["mod1"] = torch.tensor(mod1_list, dtype=torch.float32, device=device).unsqueeze(0)
    _TENSORS_CACHE["mod2"] = torch.tensor(mod2_list, dtype=torch.float32, device=device).unsqueeze(0)
    
    _TENSORS_CACHE["sup1_mask"] = torch.tensor([p[0] == "Support" for p in type_pairs], dtype=torch.bool, device=device).unsqueeze(0)
    _TENSORS_CACHE["sup2_mask"] = torch.tensor([p[1] == "Support" for p in type_pairs], dtype=torch.bool, device=device).unsqueeze(0)
    
    _TENSORS_CACHE["assassin1_mask"] = torch.tensor([p[0] == "Assassin" for p in type_pairs], dtype=torch.bool, device=device).unsqueeze(0)
    _TENSORS_CACHE["assassin2_mask"] = torch.tensor([p[1] == "Assassin" for p in type_pairs], dtype=torch.bool, device=device).unsqueeze(0)
    
    # Precompute static accuracy and crit multipliers
    crit_prob1 = base_stats[idx1, 9].unsqueeze(0)
    crit_prob2 = base_stats[idx2, 9].unsqueeze(0)
    ms1 = base_stats[idx1, 10].unsqueeze(0)
    ms2 = base_stats[idx2, 10].unsqueeze(0)
    ar1 = base_stats[idx1, 11].unsqueeze(0)
    ar2 = base_stats[idx2, 11].unsqueeze(0)
    
    _TENSORS_CACHE["crit_mult1"] = 1.0 + crit_prob1 * 0.75
    _TENSORS_CACHE["crit_mult2"] = 1.0 + crit_prob2 * 0.75
    
    range_adv1 = (ar1 - ar2) / 1000.0
    range_adv2 = (ar2 - ar1) / 1000.0
    
    _TENSORS_CACHE["acc1"] = torch.clamp(0.85 + (ms1 - ms2) / 1000.0 + range_adv1, min=0.40, max=0.95)
    _TENSORS_CACHE["acc2"] = torch.clamp(0.85 + (ms2 - ms1) / 1000.0 + range_adv2, min=0.40, max=0.95)
    
    _TENSORS_CACHE["num_matches"] = num_matches
    
    return _TENSORS_CACHE

@torch.inference_mode()
def evaluate_batch_tensor(base_data, chromosomes, mode="continuous", device=None, **kwargs):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    batch_size = len(chromosomes)
    num_champs = len(base_data)
    
    # Initialize base tensors (caches globally)
    T = init_tensors(base_data, device)
    
    # chromosomes shape: (batch_size, num_champs * 8)
    if isinstance(chromosomes[0], torch.Tensor):
        chroms = torch.stack(chromosomes).to(device)
    else:
        chroms = torch.tensor(chromosomes, dtype=torch.float32, device=device)
        
    chroms = chroms.view(batch_size, num_champs, 8)
    
    # 7 linear stats mapping
    mults_linear = chroms[:, :, 0:7]
    mutated_linear = T["base_stats"][:, 0:7].unsqueeze(0) * mults_linear
    
    hp18 = mutated_linear[:, :, 0]
    mp18 = mutated_linear[:, :, 1]
    arm18 = mutated_linear[:, :, 2]
    mr18 = mutated_linear[:, :, 3]
    hp5_18 = mutated_linear[:, :, 4]
    mp5_18 = mutated_linear[:, :, 5]
    ad18 = mutated_linear[:, :, 6]
    
    # Attack speed non-linear scaling
    as_mult = chroms[:, :, 7]
    as_base_mutated = T["base_stats"][:, 7].unsqueeze(0) * as_mult
    as_perlevel_mutated = T["base_stats"][:, 8].unsqueeze(0) * as_mult
    as18 = as_base_mutated * (1.0 + (as_perlevel_mutated / 100.0) * 17.0)
    
    # Indices
    idx1 = T["idx1"]
    idx2 = T["idx2"]
    
    # Extract match stats (batch_size, num_matches)
    hp1 = hp18[:, idx1]
    hp2 = hp18[:, idx2]
    ad1 = ad18[:, idx1]
    ad2 = ad18[:, idx2]
    arm1 = arm18[:, idx1]
    arm2 = arm18[:, idx2]
    mr1 = mr18[:, idx1]
    mr2 = mr18[:, idx2]
    as1 = as18[:, idx1]
    as2 = as18[:, idx2]
    hp_regen1 = hp5_18[:, idx1]
    hp_regen2 = hp5_18[:, idx2]
    mp1 = mp18[:, idx1]
    mp2 = mp18[:, idx2]
    mp_regen1 = mp5_18[:, idx1]
    mp_regen2 = mp5_18[:, idx2]
    
    mod1 = T["mod1"]
    mod2 = T["mod2"]
    
    hp_regen1 = torch.where(T["sup1_mask"], hp_regen1 + mp_regen1 * 2.0, hp_regen1)
    hp_regen2 = torch.where(T["sup2_mask"], hp_regen2 + mp_regen2 * 2.0, hp_regen2)
    
    # Phys DPS
    phys_dps1 = ad1 * as1 * mod1 * T["crit_mult1"] * (100.0 / (100.0 + torch.clamp(arm2, min=0)))
    phys_dps2 = ad2 * as2 * mod2 * T["crit_mult2"] * (100.0 / (100.0 + torch.clamp(arm1, min=0)))
    
    # Magic DPS
    magic_dps1 = torch.where(mp1 > 0, (ad1 * 0.2 + mp1 * 0.01 + mp_regen1 * 2) * mod1 * (100.0 / (100.0 + torch.clamp(mr2, min=0))), torch.zeros_like(mp1))
    magic_dps2 = torch.where(mp2 > 0, (ad2 * 0.2 + mp2 * 0.01 + mp_regen2 * 2) * mod2 * (100.0 / (100.0 + torch.clamp(mr1, min=0))), torch.zeros_like(mp2))
    
    dps1 = phys_dps1 + magic_dps1
    dps2 = phys_dps2 + magic_dps2
    
    # Max HP
    max_hp1 = hp1.clone()
    max_hp2 = hp2.clone()
    
    # Turn execution
    num_matches = T["num_matches"]
    turns = torch.zeros((batch_size, num_matches), dtype=torch.int32, device=device)
    max_turns = 200
    
    rng = torch.Generator(device=device)
    eval_seed = kwargs.get("evaluation_seed", 20260612)
    use_crn = kwargs.get("use_common_random_numbers", True)
    if not use_crn:
        import random
        eval_seed = random.randint(0, 1000000)
    rng.manual_seed(eval_seed)
    
    active_mask = (hp1 > 0) & (hp2 > 0)
    
    acc1 = T["acc1"]
    acc2 = T["acc2"]
    
    assassin1_mask = T["assassin1_mask"]
    assassin2_mask = T["assassin2_mask"]
    
    for t in range(1, max_turns + 1):
        if not active_mask.any():
            break
            
        turns = torch.where(active_mask, turns + 1, turns)
        
        hp1 = torch.where(active_mask, hp1 + hp_regen1, hp1)
        hp2 = torch.where(active_mask, hp2 + hp_regen2, hp2)
        
        burst1 = torch.where(assassin1_mask & (turns <= 5), 1.3, 1.0)
        burst2 = torch.where(assassin2_mask & (turns <= 5), 1.3, 1.0)
        
        r1 = torch.rand((batch_size, num_matches), generator=rng, device=device)
        r2 = torch.rand((batch_size, num_matches), generator=rng, device=device)
        hit1 = r1 <= acc1
        hit2 = r2 <= acc2
        
        var1 = torch.empty((batch_size, num_matches), device=device).uniform_(0.85, 1.15, generator=rng)
        var2 = torch.empty((batch_size, num_matches), device=device).uniform_(0.85, 1.15, generator=rng)
        
        dmg1 = torch.where(active_mask & hit1, dps1 * burst1 * var1, torch.zeros_like(dps1))
        dmg2 = torch.where(active_mask & hit2, dps2 * burst2 * var2, torch.zeros_like(dps2))
        
        hp2 = hp2 - dmg1
        hp1 = hp1 - dmg2
        
        active_mask = (hp1 > 0) & (hp2 > 0)
        
    p1_wins = (hp1 > 0) & (hp2 <= 0)
    p2_wins = (hp2 > 0) & (hp1 <= 0)
    
    # Matches the fallback `else` block in CPU logic: handles both mutual destruction AND timeouts
    draws = ~(p1_wins | p2_wins)
    
    hp1_pct = torch.where(max_hp1 > 0, hp1 / max_hp1, torch.zeros_like(hp1))
    hp2_pct = torch.where(max_hp2 > 0, hp2 / max_hp2, torch.zeros_like(hp2))
    
    p1_draw_wins = draws & (hp1_pct > hp2_pct)
    ties = draws & (hp1_pct == hp2_pct)
    
    p1_score = p1_wins.float() + p1_draw_wins.float() + ties.float() * 0.5
    
    p1_score = p1_score.view(batch_size, len(TYPES), len(TYPES), 30)
    wins_tensor = p1_score.sum(dim=-1)
    
    matches_tensor = torch.full((batch_size, len(TYPES), len(TYPES)), 30.0, device=device)
    
    valid_mask = T["valid_mask"]
    win_rate_matrix = torch.where(valid_mask, wins_tensor / matches_tensor, 0.5)
    
    total_duration = turns.float().sum(dim=1)
    total_matches = num_matches
    avg_duration = total_duration / total_matches
    
    diff = T["target_matrix"] - win_rate_matrix
    total_abs_error = diff.abs().sum(dim=(1, 2))
    rbi_error = total_abs_error / 36.0
    
    balance_tolerance = 0.30
    balance_score = torch.clamp(10.0 * (1.0 - (rbi_error / balance_tolerance)), min=0.0)
    
    t_min, t_max, w_t = 35.0, 90.0, 0.2
    f2_penalty = torch.zeros_like(avg_duration)
    f2_penalty = torch.where(avg_duration < t_min, (t_min - avg_duration) * w_t, f2_penalty)
    f2_penalty = torch.where(avg_duration > t_max, (avg_duration - t_max) * w_t, f2_penalty)
    
    overall_win_rates = win_rate_matrix.sum(dim=2) / float(len(TYPES))
    total_wr = overall_win_rates.sum(dim=1, keepdim=True)
    p = torch.where(total_wr > 0, overall_win_rates / total_wr, torch.zeros_like(overall_win_rates))
    entropy = -torch.where(p > 0, p * torch.log2(p), torch.zeros_like(p)).sum(dim=1)
    
    import math
    entropy_score = (entropy / math.log2(len(TYPES))) * 5.0
    mds = (entropy / math.log2(len(TYPES))) * 100.0
    
    l2_penalty = ((chroms - 1.0) ** 2).view(batch_size, -1).mean(dim=1)
    
    w1, w2, w3, l2_weight = 1.0, 0.8, 0.5, 0.2
    score = w1 * balance_score - w2 * f2_penalty + w3 * entropy_score - l2_penalty * l2_weight
    score = torch.clamp(score, min=0.0001)
    
    score_cpu = score.cpu().numpy()
    rbi_cpu = rbi_error.cpu().numpy()
    mds_cpu = mds.cpu().numpy()
    sbal_cpu = balance_score.cpu().numpy()
    pdur_cpu = avg_duration.cpu().numpy()
    sent_cpu = entropy_score.cpu().numpy()
    pl2_cpu = l2_penalty.cpu().numpy()
    
    results = []
    for b in range(batch_size):
        results.append((
            float(score_cpu[b]), float(rbi_cpu[b]), float(mds_cpu[b]),
            float(sbal_cpu[b]), float(pdur_cpu[b]), float(sent_cpu[b]), float(pl2_cpu[b])
        ))
        
    return results
