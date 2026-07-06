import copy
import random
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
from src.fitness.fitness import calculate_fitness
from src.simulation.simulator import run_simulation

TUNABLE_STATS = [
    "hp",
    "mp",
    "armor",
    "spellblock",
    "hpregen",
    "mpregen",
    "attackdamage",
    "attackspeed",
]
# These are still used by the simulator, but not tuned by the optimizer.
# They define champion identity and do not scale by level.
LOCKED_STATS = ["movespeed", "attackrange"]
GENES_PER_CHAMP = len(TUNABLE_STATS)
EVALUATION_SEED = 20260612


def apply_chromosome(base_data, chromosome):
    mutated_data = copy.deepcopy(base_data)
    num_champs = len(mutated_data)

    # The optimizer tunes 8 core stats. Per-level stats are scaled together
    # with their corresponding base stat.
    # 1. Bỏ crit/critperlevel vì không đóng góp nhiều vào cân bằng tổng thể (chỉ vài tướng dùng)
    # 2. Các chỉ số 'perlevel' sẽ được scale chung với chỉ số gốc (1 gene điều khiển cả base lẫn level)
    core_stats = TUNABLE_STATS

    for i in range(num_champs):
        idx = i * GENES_PER_CHAMP
        for j, key in enumerate(core_stats):
            multiplier = chromosome[idx + j]
            mutated_data[i][key] *= multiplier

            # Gộp chung: Tăng base stat thì tăng luôn per_level tương ứng
            perlevel_key = key + "perlevel"
            if perlevel_key in mutated_data[i]:
                old_value = mutated_data[i][perlevel_key]
                try:
                    old_value = float(old_value)
                except (TypeError, ValueError):
                    print(f"[BAD VALUE] champion_index={i}, key={perlevel_key}, value={old_value}, type={type(old_value)}")
                    old_value = 0.0
                mutated_data[i][perlevel_key] = old_value * float(multiplier)

        # Tiền tính toán Level 18 Stats để tối ưu hóa simulator
        c = mutated_data[i]
        c["hp18"] = c["hp"] + c.get("hpperlevel", 0) * 17
        c["mp18"] = c["mp"] + c.get("mpperlevel", 0) * 17
        c["arm18"] = c["armor"] + c.get("armorperlevel", 0) * 17
        c["mr18"] = c["spellblock"] + c.get("spellblockperlevel", 0) * 17
        c["ad18"] = c["attackdamage"] + c.get("attackdamageperlevel", 0) * 17
        c["as18"] = c["attackspeed"] * (
            1 + (c.get("attackspeedperlevel", 0) / 100.0) * 17
        )
        c["crit18"] = c.get("crit", 0) / 100.0 + (c.get("critperlevel", 0) / 100.0) * 17
        c["hp5_18"] = (c["hpregen"] + c.get("hpregenperlevel", 0) * 17) / 5.0
        c["mp5_18"] = (c["mpregen"] + c.get("mpregenperlevel", 0) * 17) / 5.0

    return mutated_data


def evaluate(base_data, chromosome, **fitness_kwargs):
    mutated_data = apply_chromosome(base_data, chromosome)
    # Common random numbers: every candidate is evaluated on the same simulated
    # matchup stream, making optimizer comparisons fairer and less noisy.
    eval_seed = fitness_kwargs.get("evaluation_seed", EVALUATION_SEED)
    use_crn = fitness_kwargs.get("use_common_random_numbers", True)
    if not use_crn:
        eval_seed = random.randint(0, 1000000)
    win_rates, duration = run_simulation(mutated_data, random.Random(eval_seed))
    res = calculate_fitness(win_rates, duration, chromosome, **fitness_kwargs)
    if fitness_kwargs.get("return_components", False):
        score, rbi, mds, Sbalance, Pduration, Sentropy, PL2 = res
        return score, mutated_data, win_rates, rbi, mds, Sbalance, Pduration, Sentropy, PL2
    else:
        score, rbi, mds = res
        return score, mutated_data, win_rates, rbi, mds


def evaluate_batch_wrapper(base_data, batch, **kwargs):
    """
    Evaluates a batch of chromosomes using multiprocessing or GPU.
    Returns a list of tuples to match the expected signature.
    """
    from src.evaluation.parallel_evaluator import evaluate_batch
    
    results = evaluate_batch(base_data, batch, mode="continuous", **kwargs)
    if kwargs.get("return_components", False):
        return [(res.fitness, None, None, res.metrics.get("rbi", 0.0), res.metrics.get("mds", 0.0), res.metrics.get("Sbalance", 0.0), res.metrics.get("Pduration", 0.0), res.metrics.get("Sentropy", 0.0), res.metrics.get("PL2", 0.0)) for res in results]
    return [(res.fitness, None, None, res.metrics.get("rbi", 0.0), res.metrics.get("mds", 0.0)) for res in results]
