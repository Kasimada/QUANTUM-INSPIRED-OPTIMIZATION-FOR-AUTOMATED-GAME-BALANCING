import math
import random

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.algorithms import GENES_PER_CHAMP, evaluate
from src.datasets.csv_loader import load_csv_dataset as load_champion_data


def calc_diversity(population):
    # population is a list of lists (chromosomes)
    arr = np.array(population)
    # std deviation per gene, then average
    return np.mean(np.std(arr, axis=0))


def run_GA_tracked(base_data, max_FEs=3000):
    pop_size = 50
    num_genes = len(base_data) * GENES_PER_CHAMP
    population = [
        [random.uniform(0.7, 1.3) for _ in range(num_genes)] for _ in range(pop_size)
    ]
    history_fe, history_div = [], []
    evals = 0
    while evals < max_FEs:
        scores = []
        for chrom in population:
            if evals >= max_FEs:
                break
            score, _, _, _, _ = evaluate(base_data, chrom)
            evals += 1
            scores.append(score)

        history_fe.append(evals)
        history_div.append(calc_diversity(population))

        best_idx = np.argmax(scores)
        best_chromosome = population[best_idx]

        new_population = [list(best_chromosome)]
        while len(new_population) < pop_size:
            idx_pool1 = random.sample(range(len(scores)), min(2, len(scores)))
            p1_idx = max(idx_pool1, key=lambda i: scores[i])
            idx_pool2 = random.sample(range(len(scores)), min(2, len(scores)))
            p2_idx = max(idx_pool2, key=lambda i: scores[i])

            p1 = population[p1_idx]
            p2 = population[p2_idx]

            if random.random() < 0.9:
                pt = random.randint(1, num_genes - 1)
                child = p1[:pt] + p2[pt:]
            else:
                child = list(p1)

            eta_m = 20.0
            for i in range(num_genes):
                if random.random() < 0.05:
                    u = random.random()
                    if u <= 0.5:
                        delta = (2.0 * u) ** (1.0 / (eta_m + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta_m + 1.0))

                    if delta < 0:
                        child[i] += delta * (child[i] - 0.7)
                    else:
                        child[i] += delta * (1.3 - child[i])
                    child[i] = min(max(child[i], 0.7), 1.3)
            new_population.append(child)
        population = new_population
    return history_fe, history_div


def run_AQEA_tracked(base_data, max_FEs=3000):
    pop_size = 40
    num_genes = len(base_data) * GENES_PER_CHAMP
    angles = [
        [random.uniform(0.0, math.pi / 2.0) for _ in range(num_genes)]
        for _ in range(pop_size)
    ]
    history_fe, history_div = [], []
    evals = 0
    stagnant_generations = 0
    gbest_score = -1.0
    gbest = None

    while evals < max_FEs:
        previous_best = gbest_score
        generation = []
        pop_candidates = []

        for i in range(pop_size):
            if evals >= max_FEs - 1:
                break
            c_alpha = [
                0.7 + 0.6 * (math.cos(angles[i][j]) ** 2) for j in range(num_genes)
            ]
            c_beta = [
                0.7 + 0.6 * (math.sin(angles[i][j]) ** 2) for j in range(num_genes)
            ]
            score_alpha, _, _, _, _ = evaluate(base_data, c_alpha)
            score_beta, _, _, _, _ = evaluate(base_data, c_beta)
            evals += 2
            if score_alpha >= score_beta:
                score, solution = score_alpha, c_alpha
            else:
                score, solution = score_beta, c_beta
            generation.append((score, solution))
            pop_candidates.append(solution)
            if score > gbest_score:
                gbest_score = score
                gbest = list(solution)

        history_fe.append(evals)
        history_div.append(calc_diversity(pop_candidates))

        if gbest_score <= previous_best + 1e-9:
            stagnant_generations += 1
        else:
            stagnant_generations = 0

        generation.sort(key=lambda item: item[0], reverse=True)
        elites = generation[: max(1, min(6, len(generation) // 4))]
        theta_base = 0.055 * math.pi
        progress = min(1.0, evals / max(1, max_FEs))
        annealed_step = 0.55 + 0.45 * math.exp(-0.55 * progress)
        stagnation_boost = min(0.35, stagnant_generations * 0.025)
        rotation_factor = min(1.15, annealed_step + stagnation_boost)
        mutation_rate = 0.018 + min(0.075, stagnant_generations * 0.004)
        mutation_span = 0.08 * math.pi + min(
            0.14 * math.pi, stagnant_generations * 0.006 * math.pi
        )

        for i in range(pop_size):
            if not elites or gbest is None:
                break
            elite_solution = elites[i % len(elites)][1]
            for j in range(num_genes):
                target_value = 0.75 * gbest[j] + 0.25 * elite_solution[j]
                safe_target = min(1.3, max(0.7, target_value))
                target_cos_sq = min(1.0, max(0.0, (safe_target - 0.7) / 0.6))
                target_angle = math.acos(math.sqrt(target_cos_sq))

                if target_angle > angles[i][j]:
                    direction = 1.0
                elif target_angle < angles[i][j]:
                    direction = -1.0
                else:
                    direction = 0.0

                new_angle = angles[i][j] + direction * theta_base * rotation_factor
                if random.random() < mutation_rate:
                    new_angle += random.uniform(-mutation_span, mutation_span)
                angles[i][j] = min(max(new_angle, 0.0), math.pi / 2.0)

        if stagnant_generations >= 18:
            for i in range(pop_size // 2, pop_size):
                angles[i] = [
                    random.uniform(0.0, math.pi / 2.0) for _ in range(num_genes)
                ]
            stagnant_generations = 0

    return history_fe, history_div


if __name__ == "__main__":
    print("Loading data...")
    base_data = load_champion_data("riot_datadragon_champion_stats.csv")
    print("Running GA...")
    ga_fe, ga_div = run_GA_tracked(base_data, 5000)
    print("Running AQEA...")
    aqea_fe, aqea_div = run_AQEA_tracked(base_data, 5000)

    plt.figure(figsize=(10, 6))
    plt.plot(ga_fe, ga_div, label="GA (Classical)", color="#2563EB", linewidth=2)
    plt.plot(
        aqea_fe, aqea_div, label="AQEA (Quantum-Inspired)", color="#DC2626", linewidth=2
    )
    plt.title("Population Diversity Over Time (AQEA vs GA)", fontsize=16)
    plt.xlabel("Function Evaluations (FEs)", fontsize=12)
    plt.ylabel("Average Genetic Standard Deviation", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("AQEA_Diversity_Tracking.png")
    print("Saved AQEA_Diversity_Tracking.png")
