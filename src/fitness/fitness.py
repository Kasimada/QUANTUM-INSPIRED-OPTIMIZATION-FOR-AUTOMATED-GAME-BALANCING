import math

from src.simulation.type_chart import get_target_matrix, TYPES


def calculate_fitness(
    win_rate_matrix, avg_duration, chromosome=None, w1=1.0, w2=0.8, w3=0.5, l2_weight=0.2,
    scenario="symmetric", return_components=False, **kwargs
):
    target_matrix = get_target_matrix(scenario)
    
    # 1. Sai sá»‘ cÃ¢n báº±ng tÆ°Æ¡ng Ä‘á»‘i (MSE)
    total_squared_error = 0
    total_abs_error = 0
    count = 0
    for t1 in TYPES:
        for t2 in TYPES:
            diff = target_matrix[t1][t2] - win_rate_matrix[t1][t2]
            total_squared_error += diff**2
            total_abs_error += abs(diff)
            count += 1

    f1_mse = total_squared_error / count
    rbi_error = total_abs_error / count
    # RBI is the primary balance error used in the paper tables. Scaling by a
    # domain tolerance avoids collapsing most candidates to a zero balance term.
    balance_tolerance = 0.30
    balance_score = max(10.0 * (1.0 - (rbi_error / balance_tolerance)), 0)

    # 2. Trá»«ng pháº¡t nhá»‹p Ä‘á»™ tráº­n Ä‘áº¥u
    # The simulator reports duel length in simulated turns, not real LoL minutes.
    # A broad target band prevents duration from dominating the balance metric.
    t_min = 35.0
    t_max = 90.0
    w_t = 0.2
    f2_penalty = 0.0
    if avg_duration < t_min:
        f2_penalty = (t_min - avg_duration) * w_t
    elif avg_duration > t_max:
        f2_penalty = (avg_duration - t_max) * w_t

    # 3. Äa dáº¡ng Há»‡ sinh thÃ¡i (Shannon Entropy)
    overall_win_rates = [sum(win_rate_matrix[t1].values()) / len(TYPES) for t1 in TYPES]
    total_wr = sum(overall_win_rates)
    entropy = 0
    if total_wr > 0:
        for wr in overall_win_rates:
            p = wr / total_wr
            if p > 0:
                entropy -= p * math.log2(p)

    entropy_score = (entropy / math.log2(len(TYPES))) * 5

    # Trá»ng sá»‘ Ä‘a má»¥c tiÃªu (Multi-objective Weights) â€” cÃ³ thá»ƒ tinh chá»‰nh thá»±c nghiá»‡m qua arguments
    score = w1 * balance_score - w2 * f2_penalty + w3 * entropy_score

    # 4. L2-Regularization: Pháº¡t láº¡m phÃ¡t chá»‰ sá»‘ (thay tháº¿ cho pháº¡t cÃ o báº±ng)
    if chromosome is not None:
        # Trung bÃ¬nh bÃ¬nh phÆ°Æ¡ng khoáº£ng cÃ¡ch tá»« chromosome tá»›i má»©c cÆ¡ báº£n (1.0)
        l2_penalty = sum([(x - 1.0) ** 2 for x in chromosome]) / len(chromosome)
        score -= l2_penalty * l2_weight

    score = max(score, 0.0001)

    if return_components:
        # Sbalance, Pduration, Sentropy, PL2
        Sbalance = w1 * balance_score
        Pduration = w2 * f2_penalty
        Sentropy = w3 * entropy_score
        pl2 = l2_penalty * l2_weight if chromosome is not None else 0.0
        return score, rbi_error, (entropy / math.log2(len(TYPES))) * 100, Sbalance, Pduration, Sentropy, pl2

    # Keep full precision for optimizers. Reporting code formats values to four
    # decimals, but rounding here creates artificial plateaus for adaptive search.
    return score, rbi_error, (entropy / math.log2(len(TYPES))) * 100
