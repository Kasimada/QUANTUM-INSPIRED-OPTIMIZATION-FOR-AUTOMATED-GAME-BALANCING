import torch
import math
import numpy as np

# Mock data
from src.simulation.type_chart import TARGET_MATRIX, TYPES
from src.fitness.fitness import calculate_fitness

def run_test():
    batch_size = 1000
    num_types = len(TYPES)
    
    # 1. Generate fake deterministic win rates and durations
    np.random.seed(42)
    fake_win_rates = np.random.uniform(0.3, 0.7, (batch_size, num_types, num_types))
    fake_durations = np.random.uniform(20.0, 100.0, (batch_size,))
    fake_chroms = np.random.uniform(0.8, 1.2, (batch_size, 172 * 8))
    
    # 2. Run Python sequential calculate_fitness
    cpu_scores = []
    cpu_rbis = []
    cpu_mdss = []
    
    for b in range(batch_size):
        # build dict
        wr = {t1: {t2: float(fake_win_rates[b, i, j]) for j, t2 in enumerate(TYPES)} for i, t1 in enumerate(TYPES)}
        chrom_list = fake_chroms[b].tolist()
        dur = float(fake_durations[b])
        
        s, r, m = calculate_fitness(wr, dur, chrom_list)
        cpu_scores.append(s)
        cpu_rbis.append(r)
        cpu_mdss.append(m)
        
    # 3. Run PyTorch Vectorized Fitness
    win_rate_matrix = torch.tensor(fake_win_rates, dtype=torch.float32)
    avg_duration = torch.tensor(fake_durations, dtype=torch.float32)
    chroms = torch.tensor(fake_chroms, dtype=torch.float32)
    device = torch.device("cpu")
    
    target_matrix_list = [[TARGET_MATRIX[t1][t2] for t2 in TYPES] for t1 in TYPES]
    target_tensor = torch.tensor(target_matrix_list, dtype=torch.float32, device=device).unsqueeze(0)
    
    diff = target_tensor - win_rate_matrix
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
    
    entropy_score = (entropy / math.log2(len(TYPES))) * 5.0
    mds = (entropy / math.log2(len(TYPES))) * 100.0
    
    l2_penalty = ((chroms - 1.0) ** 2).mean(dim=1)
    
    w1, w2, w3, l2_weight = 1.0, 0.8, 0.5, 0.2
    score = w1 * balance_score - w2 * f2_penalty + w3 * entropy_score - l2_penalty * l2_weight
    score = torch.clamp(score, min=0.0001)
    
    gpu_scores = score.numpy()
    gpu_rbis = rbi_error.numpy()
    gpu_mdss = mds.numpy()
    
    # 4. Compare
    score_diff = np.max(np.abs(np.array(cpu_scores) - gpu_scores))
    rbi_diff = np.max(np.abs(np.array(cpu_rbis) - gpu_rbis))
    mds_diff = np.max(np.abs(np.array(cpu_mdss) - gpu_mdss))
    
    print(f"fitness max_abs_diff = {score_diff}")
    print(f"rbi max_abs_diff = {rbi_diff}")
    print(f"mds max_abs_diff = {mds_diff}")
    
    if score_diff < 1e-4 and rbi_diff < 1e-4 and mds_diff < 1e-4:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    run_test()
