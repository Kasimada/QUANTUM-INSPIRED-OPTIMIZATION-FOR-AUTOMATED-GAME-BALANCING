from __future__ import annotations
from src.algorithms.algorithms import GENES_PER_CHAMP, evaluate
from src.algorithms.base import Optimizer
from typing import Any, Callable, Dict, Tuple
import json
import math
import os
import pandas as pd
import random




PATCH_LEVELS = [0.90, 0.95, 1.00, 1.05, 1.10]
PATCH_STEPS = [-2, -1, 0, 1, 2]
PATCH_LEVELS_TUPLE = tuple(PATCH_LEVELS)
PATCH_STEPS_TUPLE = tuple(PATCH_STEPS)
PATCH_INDEX_BY_LEVEL = {level: index for index, level in enumerate(PATCH_LEVELS)}
PATCH_PRESSURE_PENALTY_WEIGHT = 8.0
PATCH_MAGNITUDE_PENALTY_WEIGHT = 0.75
AQEA_MIN_PROBABILITY = 0.004
AQEA_DIVERSE_ELITE_DISTANCE = 0.08


def num_genes_for(base_data):
    return len(base_data) * GENES_PER_CHAMP


def index_to_multiplier(index):
    return PATCH_LEVELS[int(min(max(index, 0), len(PATCH_LEVELS) - 1))]


def nearest_patch_index(value):
    return min(
        range(len(PATCH_LEVELS)), key=lambda index: abs(PATCH_LEVELS[index] - value)
    )


def indices_to_multipliers(indices):
    return [PATCH_LEVELS_TUPLE[index] for index in indices]


def multipliers_to_indices(chromosome):
    return [nearest_patch_index(value) for value in chromosome]


def patch_pressure_from_indices(indices):
    return sum(PATCH_STEPS_TUPLE[index] for index in indices)


def patch_pressure_deviation_ratio(indices):
    if not indices:
        return 0.0
    net_pressure = patch_pressure_from_indices(indices)
    max_abs_pressure = max(abs(step) for step in PATCH_STEPS) * len(indices)
    return abs(net_pressure) / max(1, max_abs_pressure)


def patch_magnitude_ratio(indices):
    if not indices:
        return 0.0
    max_abs_pressure = 2 * len(indices)
    used_pressure = sum(abs(PATCH_STEPS_TUPLE[index]) for index in indices)
    return used_pressure / max(1, max_abs_pressure)


def discrete_audit_metrics(indices, max_abs_pressure):
    net_patch_pressure = patch_pressure_from_indices(indices)
    abs_net_patch_pressure = abs(net_patch_pressure)
    constraint_violation = max(0.0, abs_net_patch_pressure - max_abs_pressure)
    patch_magnitude = patch_magnitude_ratio(indices)
    return net_patch_pressure, abs_net_patch_pressure, constraint_violation, patch_magnitude


def evaluate_discrete(base_data, indices, objective_fn, **kwargs):
    multipliers = indices_to_multipliers(indices)
    res = objective_fn(base_data, multipliers, **kwargs)
    if kwargs.get("return_components", False):
        raw_score, mutated_data, win_rates, rbi, mds, Sbalance, Pduration, Sentropy, PL2 = res
    else:
        raw_score, mutated_data, win_rates, rbi, mds = res
        Sbalance = Pduration = Sentropy = PL2 = 0.0

    pressure_deviation = patch_pressure_deviation_ratio(indices)
    patch_magnitude = patch_magnitude_ratio(indices)
    score = max(
        raw_score
        - PATCH_PRESSURE_PENALTY_WEIGHT * pressure_deviation
        - PATCH_MAGNITUDE_PENALTY_WEIGHT * patch_magnitude,
        0.0001,
    )
    if kwargs.get("return_components", False):
        return (
            score, mutated_data, win_rates, rbi, mds,
            patch_pressure_from_indices(indices), pressure_deviation, patch_magnitude,
            Sbalance, Pduration, Sentropy, PL2
        )

    return (
        score, mutated_data, win_rates, rbi, mds,
        patch_pressure_from_indices(indices), pressure_deviation, patch_magnitude,
    )


def evaluate_batch_wrapper_discrete(base_data, batch, **kwargs):
    """
    Evaluates a batch of discrete chromosomes using multiprocessing.
    Returns a list of tuples to match the expected 8-element signature.
    """
    from src.evaluation.parallel_evaluator import evaluate_batch
    
    results = evaluate_batch(base_data, batch, mode="discrete", **kwargs)
    if kwargs.get("return_components", False):
        return [(res.fitness, None, None, res.metrics.get("rbi", 0.0), res.metrics.get("mds", 0.0), res.metrics.get("pressure", 0.0), res.metrics.get("violation", 0.0), res.metrics.get("magnitude", 0.0), res.metrics.get("Sbalance", 0.0), res.metrics.get("Pduration", 0.0), res.metrics.get("Sentropy", 0.0), res.metrics.get("PL2", 0.0)) for res in results]
    return [(res.fitness, None, None, res.metrics.get("rbi", 0.0), res.metrics.get("mds", 0.0), res.metrics.get("pressure", 0.0), res.metrics.get("violation", 0.0), res.metrics.get("magnitude", 0.0)) for res in results]



def random_feasible_indices(num_genes, max_abs_net_pressure=1.0):
    indices = [random.randrange(len(PATCH_LEVELS)) for _ in range(num_genes)]
    return repair_indices(indices, max_abs_net_pressure=max_abs_net_pressure)


def repair_indices(indices, max_abs_net_pressure=1.0):
    repaired = list(indices)
    net_pressure = patch_pressure_from_indices(repaired)
    while abs(net_pressure) > max_abs_net_pressure:
        if net_pressure > 0:
            candidates = [
                i
                for i, index in enumerate(repaired)
                if index > 0
            ]
            direction = -1
        else:
            candidates = [
                i
                for i, index in enumerate(repaired)
                if index < 4
            ]
            direction = 1
        if not candidates:
            break
        pos = random.choice(candidates)
        repaired[pos] += direction
        net_pressure += direction
    return repaired


def run_GA_discrete(base_data, max_FEs=20000, objective_fn=evaluate, config=None):
    if config is None: config = {}
    patch_cfg = config.get("patch_constraints", {})
    max_pressure = patch_cfg.get("max_abs_net_pressure", 1.0) if patch_cfg.get("use_net_patch_pressure", True) else 9999.0
    print(f"--- Starting discrete GA (budget: {max_FEs} FEs) ---", flush=True)
    pop_size = 50
    num_genes = num_genes_for(base_data)
    population = [random_feasible_indices(num_genes, max_pressure) for _ in range(pop_size)]
    history_x, history_y = [], []
    best_score = -1.0
    best_indices = None
    evals = 0
    gen = 0

    while evals < max_FEs:
        gen += 1
        scores = []
        batch = population[:max_FEs - evals]
        if not batch:
            break
            
        batch_results = objective_fn(base_data, batch)
        
        for chrom, result in zip(batch, batch_results):
            score = result[0]
            evals += 1
            scores.append(score)
            if score > best_score:
                best_score = score
                best_indices = list(chrom)

        history_x.append(evals)
        history_y.append(best_score)
        if not scores:
            break

        new_population = [list(best_indices)]
        while len(new_population) < pop_size:
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            child = []
            for a, b in zip(p1, p2):
                child.append(a if random.random() < 0.5 else b)
            for i in range(num_genes):
                if random.random() < 0.035:
                    if random.random() < 0.75:
                        child[i] = min(
                            max(child[i] + random.choice([-1, 1]), 0),
                            len(PATCH_LEVELS) - 1,
                        )
                    else:
                        child[i] = random.randrange(len(PATCH_LEVELS))
            new_population.append(repair_indices(child, max_pressure))
        population = new_population

        from src.utils.progress import print_progress_bar
        print_progress_bar("Discrete GA", min(evals, max_FEs), max_FEs, best_score)

    return history_x, history_y, indices_to_multipliers(best_indices)


def tournament_select(population, scores):
    limit = min(len(scores), len(population))
    candidates = random.sample(range(limit), min(3, limit))
    winner = max(candidates, key=lambda index: scores[index])
    return population[winner]


def run_PSO_discrete(base_data, max_FEs=20000, objective_fn=evaluate, config=None):
    if config is None: config = {}
    patch_cfg = config.get("patch_constraints", {})
    max_pressure = patch_cfg.get("max_abs_net_pressure", 1.0) if patch_cfg.get("use_net_patch_pressure", True) else 9999.0
    print(f"--- Starting discrete PSO (budget: {max_FEs} FEs) ---", flush=True)
    pop_size = 30
    num_genes = num_genes_for(base_data)
    particles = [
        [random.uniform(0.0, len(PATCH_LEVELS) - 1) for _ in range(num_genes)]
        for _ in range(pop_size)
    ]
    velocities = [
        [random.uniform(-0.5, 0.5) for _ in range(num_genes)] for _ in range(pop_size)
    ]
    pbest = [list(particle) for particle in particles]
    pbest_scores = [-1.0] * pop_size
    gbest = None
    gbest_indices = None
    gbest_score = -1.0
    history_x, history_y = [], []
    evals = 0
    gen = 0
    max_gen_estimate = max(1, max_FEs / pop_size)

    while evals < max_FEs:
        gen += 1
        inertia = max(0.4, 0.9 - 0.5 * (gen / max_gen_estimate))
        
        batch = particles[:max_FEs - evals]
        if not batch:
            break
            
        indices_batch = [repair_indices([round_particle_value(v) for v in p], max_pressure) for p in batch]
        results = objective_fn(base_data, indices_batch)
        
        for i, (result, indices) in enumerate(zip(results, indices_batch)):
            score = result[0]
            evals += 1
            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest[i] = [float(idx) for idx in indices]
            if score > gbest_score:
                gbest_score = score
                gbest_indices = list(indices)
                gbest = [float(idx) for idx in indices]

        history_x.append(evals)
        history_y.append(gbest_score)
        if gbest is None:
            break

        for i in range(pop_size):
            for j in range(num_genes):
                r1, r2 = random.random(), random.random()
                velocities[i][j] = (
                    inertia * velocities[i][j]
                    + 1.8 * r1 * (pbest[i][j] - particles[i][j])
                    + 1.8 * r2 * (gbest[j] - particles[i][j])
                )
                velocities[i][j] = min(max(velocities[i][j], -1.0), 1.0)
                particles[i][j] = min(
                    max(particles[i][j] + velocities[i][j], 0.0), len(PATCH_LEVELS) - 1
                )

        from src.utils.progress import print_progress_bar
        print_progress_bar("Discrete PSO", min(evals, max_FEs), max_FEs, gbest_score)

    if gbest_indices is None:
        gbest_indices = repair_indices([round_particle_value(value) for value in gbest], max_pressure)
    return history_x, history_y, indices_to_multipliers(gbest_indices)


def round_particle_value(value):
    return int(min(max(round(value), 0), len(PATCH_LEVELS) - 1))


def run_QEA_discrete(base_data, max_FEs=20000, objective_fn=evaluate, config=None):
    return run_probability_qea_discrete(base_data, max_FEs=max_FEs, adaptive=False, objective_fn=objective_fn, config=config)


def run_AQEA_discrete(base_data, max_FEs=20000, objective_fn=evaluate, config=None):
    return run_probability_qea_discrete(base_data, max_FEs=max_FEs, adaptive=True, objective_fn=objective_fn, config=config)


def run_probability_qea_discrete(
    base_data, max_FEs=20000, adaptive=True, balanced=False, objective_fn=evaluate, config=None
):
    if config is None: config = {}
    patch_cfg = config.get("patch_constraints", {})
    max_pressure = patch_cfg.get("max_abs_net_pressure", 1.0) if patch_cfg.get("use_net_patch_pressure", True) else 9999.0
    label = (
        "Discrete Balanced-QEA"
        if balanced
        else ("Discrete AQEA" if adaptive else "Discrete QEA")
    )
    print(f"--- Starting {label} (budget: {max_FEs} FEs) ---", flush=True)
    pop_size = 40
    num_genes = num_genes_for(base_data)
    probabilities = [
        [1.0 / len(PATCH_LEVELS)] * len(PATCH_LEVELS) for _ in range(num_genes)
    ]
    history_x, history_y = [], []
    gbest_indices = None
    gbest_score = -1.0
    evals = 0
    gen = 0
    stagnant_generations = 0
    max_gen_estimate = max(1, max_FEs / pop_size)

    while evals < max_FEs:
        gen += 1
        previous_best = gbest_score
        generation = []

        batch = []
        metadata = []
        for i in range(pop_size):
            if balanced:
                for k in range(3):
                    indices = repair_indices(sample_indices(probabilities), max_pressure)
                    batch.append(indices)
                    metadata.append(('balanced', i, k))
            else:
                indices = repair_indices(sample_indices(probabilities), max_pressure)
                batch.append(indices)
                metadata.append(('unbalanced', i))
                
        batch = batch[:max_FEs - evals]
        metadata = metadata[:max_FEs - evals]
        
        if not batch:
            break
            
        batch_results = objective_fn(base_data, batch)
        
        best_scores_per_individual = {i: -1.0 for i in range(pop_size)}
        best_indices_per_individual = {i: None for i in range(pop_size)}
        
        for m, res in zip(metadata, batch_results):
            score = res[0]
            evals += 1
            i = m[1]
            if score > best_scores_per_individual[i]:
                best_scores_per_individual[i] = score
                best_indices_per_individual[i] = batch[metadata.index(m)]
                
        for i in range(pop_size):
            sol = best_indices_per_individual[i]
            sc = best_scores_per_individual[i]
            if sol is not None:
                generation.append((sc, sol))
                if sc > gbest_score:
                    gbest_score = sc
                    gbest_indices = list(sol)

        local_trials = 0
        if (
            adaptive
            and gbest_indices is not None
            and (stagnant_generations >= 3 or evals >= max_FEs * 0.45)
        ):
            local_trials = min(6, max_FEs - evals)
        if local_trials > 0:
            local_batch = [local_patch_neighbor(gbest_indices, max_pressure) for _ in range(local_trials)]
            local_results = objective_fn(base_data, local_batch)
            for candidate, result in zip(local_batch, local_results):
                score = result[0]
                evals += 1
                generation.append((score, candidate))
                if score > gbest_score:
                    gbest_score = score
                    gbest_indices = list(candidate)

        history_x.append(evals)
        history_y.append(gbest_score)
        if gbest_score <= previous_best + 1e-9:
            stagnant_generations += 1
        else:
            stagnant_generations = 0

        if not generation or gbest_indices is None:
            break

        generation.sort(key=lambda item: item[0], reverse=True)
        elites = select_diverse_elites(
            generation,
            max_elites=max(1, min(6, len(generation) // 4)),
            min_distance=max(1, int(AQEA_DIVERSE_ELITE_DISTANCE * num_genes)),
        )

        if balanced or adaptive:
            learning_rate = 0.12 + min(0.06, max(0, stagnant_generations - 3) * 0.004)
            mutation_rate = 0.006 + min(
                0.040, max(0, stagnant_generations - 3) * 0.0025
            )
            probability_floor = (
                AQEA_MIN_PROBABILITY if stagnant_generations >= 4 else 0.0
            )
        else:
            learning_rate = 0.12
            mutation_rate = 0.006
            probability_floor = 0.0

        update_probabilities(
            probabilities, gbest_indices, learning_rate, probability_floor
        )
        for elite in elites:
            update_probabilities(
                probabilities, elite, learning_rate * 0.35, probability_floor
            )
        mutate_probabilities(probabilities, mutation_rate)

        if adaptive and stagnant_generations >= 14:
            soften_probabilities(probabilities, amount=0.08)
        if adaptive and stagnant_generations >= 30:
            reset_low_entropy_rows(probabilities, fraction=0.12)
            stagnant_generations = 0

        entropy = average_probability_entropy(probabilities)
        from src.utils.progress import print_progress_bar
        print_progress_bar(f"Discrete {label}", min(evals, max_FEs), max_FEs, f"{gbest_score:.4f} E:{entropy:.3f}")

    return history_x, history_y, indices_to_multipliers(gbest_indices)


def sample_indices(probabilities):
    return [sample_categorical(prob_vector) for prob_vector in probabilities]


def local_patch_neighbor(indices, max_abs_net_pressure=1.0):
    candidate = list(indices)
    edits = 1 if random.random() < 0.75 else 2
    for _ in range(edits):
        pos = random.randrange(len(candidate))
        candidate[pos] = min(
            max(candidate[pos] + random.choice([-1, 1]), 0), len(PATCH_LEVELS) - 1
        )
    return repair_indices(candidate, max_abs_net_pressure)


def sample_categorical(prob_vector):
    threshold = random.random()
    cumulative = 0.0
    for index, probability in enumerate(prob_vector):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(prob_vector) - 1


def select_diverse_elites(generation, max_elites, min_distance):
    elites = []
    for _, indices in generation:
        if not elites:
            elites.append(indices)
        elif all(hamming_distance(indices, elite) >= min_distance for elite in elites):
            elites.append(indices)
        if len(elites) >= max_elites:
            break
    if len(elites) < max_elites:
        for _, indices in generation:
            if indices not in elites:
                elites.append(indices)
            if len(elites) >= max_elites:
                break
    return elites


def hamming_distance(left, right):
    return sum(1 for a, b in zip(left, right) if a != b)


def update_probabilities(
    probabilities, target_indices, learning_rate, min_probability=0.0
):
    for j, target in enumerate(target_indices):
        row = probabilities[j]
        for index in range(len(row)):
            row[index] *= 1.0 - learning_rate
        row[target] += learning_rate
        normalize(row, min_probability)


def mutate_probabilities(probabilities, mutation_rate):
    if mutation_rate <= 0:
        return
    uniform = 1.0 / len(PATCH_LEVELS)
    for row in probabilities:
        if random.random() < mutation_rate:
            for index in range(len(row)):
                row[index] = row[index] * 0.75 + uniform * 0.25
            normalize(row)


def soften_probabilities(probabilities, amount):
    uniform = 1.0 / len(PATCH_LEVELS)
    for row in probabilities:
        for index in range(len(row)):
            row[index] = row[index] * (1.0 - amount) + uniform * amount
        normalize(row)


def reset_low_entropy_rows(probabilities, fraction):
    if not probabilities:
        return
    count = max(1, int(len(probabilities) * fraction))
    ranked = sorted(
        range(len(probabilities)),
        key=lambda index: probability_entropy(probabilities[index]),
    )
    uniform = 1.0 / len(PATCH_LEVELS)
    for row_index in ranked[:count]:
        for index in range(len(probabilities[row_index])):
            probabilities[row_index][index] = uniform


def average_probability_entropy(probabilities):
    if not probabilities:
        return 0.0
    return sum(probability_entropy(row) for row in probabilities) / len(probabilities)


def probability_entropy(row):
    max_entropy = math.log2(len(row))
    if max_entropy <= 0:
        return 0.0
    entropy = 0.0
    for probability in row:
        if probability > 0:
            entropy -= probability * math.log2(probability)
    return entropy / max_entropy


def normalize(row, min_probability=0.0):
    total = sum(row)
    if total <= 0:
        uniform = 1.0 / len(row)
        for index in range(len(row)):
            row[index] = uniform
        return
    for index in range(len(row)):
        row[index] /= total
    if min_probability > 0:
        max_floor = 1.0 / len(row)
        floor = min(min_probability, max_floor)
        for index in range(len(row)):
            row[index] = max(row[index], floor)
        total = sum(row)
        for index in range(len(row)):
            row[index] /= total


def non_dominated_sorting_discrete(population_scores):
    fronts = [[]]
    S = [[] for _ in range(len(population_scores))]
    n = [0] * len(population_scores)
    rank = [0] * len(population_scores)

    for p in range(len(population_scores)):
        p_obj = population_scores[p]
        for q in range(len(population_scores)):
            q_obj = population_scores[q]
            if (p_obj[0] <= q_obj[0] and p_obj[1] <= q_obj[1]) and (
                p_obj[0] < q_obj[0] or p_obj[1] < q_obj[1]
            ):
                S[p].append(q)
            elif (q_obj[0] <= p_obj[0] and q_obj[1] <= p_obj[1]) and (
                q_obj[0] < p_obj[0] or q_obj[1] < p_obj[1]
            ):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance_discrete(front, population_scores):
    distance = {i: 0 for i in front}
    if not front:
        return distance
    if len(front) <= 2:
        for i in front:
            distance[i] = float("inf")
        return distance

    for m in range(2):
        front_sorted = sorted(front, key=lambda i: population_scores[i][m])
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        m_min = population_scores[front_sorted[0]][m]
        m_max = population_scores[front_sorted[-1]][m]
        if m_max - m_min == 0:
            continue
        for i in range(1, len(front_sorted) - 1):
            distance[front_sorted[i]] += (
                population_scores[front_sorted[i + 1]][m]
                - population_scores[front_sorted[i - 1]][m]
            ) / (m_max - m_min)

    return distance


def run_NSGA_II_discrete(base_data, max_FEs=20000, objective_fn=evaluate, config=None):
    if config is None: config = {}
    patch_cfg = config.get("patch_constraints", {})
    max_pressure = patch_cfg.get("max_abs_net_pressure", 1.0) if patch_cfg.get("use_net_patch_pressure", True) else 9999.0
    print(f"--- Starting discrete NSGA-II (budget: {max_FEs} FEs) ---", flush=True)
    pop_size = 50
    num_genes = num_genes_for(base_data)
    population = [random_feasible_indices(num_genes, max_pressure) for _ in range(pop_size)]
    history_x, history_y = [], []
    best_score = -1.0
    best_indices = None
    evals = 0
    gen = 0

    while evals < max_FEs:
        gen += 1
        pop_objs = []
        pop_scores = []
        batch = population[:max_FEs - evals]
        if not batch:
            break
            
        batch_results = objective_fn(base_data, batch)
        
        for chrom, result in zip(batch, batch_results):
            score = result[0]
            rbi = result[3]
            mds = result[4]
            evals += 1
            pop_scores.append(score)
            pop_objs.append((rbi, -mds))
            if score > best_score:
                best_score = score
                best_indices = list(chrom)

        history_x.append(evals)
        history_y.append(best_score)
        if evals >= max_FEs:
            break

        fronts = non_dominated_sorting_discrete(pop_objs)
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) <= pop_size:
                new_population.extend([population[i] for i in front])
            else:
                dist = crowding_distance_discrete(front, pop_objs)
                front_sorted = sorted(front, key=lambda i: dist[i], reverse=True)
                needed = pop_size - len(new_population)
                new_population.extend([population[i] for i in front_sorted[:needed]])
                break

        offspring = []
        while len(offspring) < pop_size:
            p1_idx = random.randint(0, pop_size // 2)
            p2_idx = random.randint(0, pop_size // 2)
            p1 = new_population[p1_idx]
            p2 = new_population[p2_idx]

            child = []
            for a, b in zip(p1, p2):
                child.append(a if random.random() < 0.5 else b)
            for i in range(num_genes):
                if random.random() < 0.035:
                    if random.random() < 0.75:
                        child[i] = min(
                            max(child[i] + random.choice([-1, 1]), 0),
                            len(PATCH_LEVELS) - 1,
                        )
                    else:
                        child[i] = random.randrange(len(PATCH_LEVELS))
            offspring.append(repair_indices(child, max_pressure))
        population = offspring

        from src.utils.progress import print_progress_bar
        print_progress_bar("Discrete NSGA-II", min(evals, max_FEs), max_FEs, best_score)

    return history_x, history_y, indices_to_multipliers(best_indices)




class DiscreteGeneticOptimizer(Optimizer):
    def optimize(self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]) -> Tuple[Any, float, list, list]:
        max_FEs = config.get("max_FEs", 20000)
        history_x, history_y, best_multipliers = run_GA_discrete(initial_state, max_FEs=max_FEs, objective_fn=objective_fn, config=config)
        best_score = history_y[-1] if history_y else 0.0
        return best_multipliers, best_score, history_x, history_y

class DiscretePSOOptimizer(Optimizer):
    def optimize(self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]) -> Tuple[Any, float, list, list]:
        max_FEs = config.get("max_FEs", 20000)
        history_x, history_y, best_multipliers = run_PSO_discrete(initial_state, max_FEs=max_FEs, objective_fn=objective_fn, config=config)
        best_score = history_y[-1] if history_y else 0.0
        return best_multipliers, best_score, history_x, history_y

class DiscreteQEAOptimizer(Optimizer):
    def optimize(self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]) -> Tuple[Any, float, list, list]:
        max_FEs = config.get("max_FEs", 20000)
        history_x, history_y, best_multipliers = run_QEA_discrete(initial_state, max_FEs=max_FEs, objective_fn=objective_fn, config=config)
        best_score = history_y[-1] if history_y else 0.0
        return best_multipliers, best_score, history_x, history_y

class DiscreteAQEAOptimizer(Optimizer):
    def optimize(self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]) -> Tuple[Any, float, list, list]:
        max_FEs = config.get("max_FEs", 20000)
        history_x, history_y, best_multipliers = run_AQEA_discrete(initial_state, max_FEs=max_FEs, objective_fn=objective_fn, config=config)
        best_score = history_y[-1] if history_y else 0.0
        return best_multipliers, best_score, history_x, history_y

class DiscreteNSGA2Optimizer(Optimizer):
    def optimize(self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]) -> Tuple[Any, float, list, list]:
        max_FEs = config.get("max_FEs", 20000)
        history_x, history_y, best_multipliers = run_NSGA_II_discrete(initial_state, max_FEs=max_FEs, objective_fn=objective_fn, config=config)
        best_score = history_y[-1] if history_y else 0.0
        return best_multipliers, best_score, history_x, history_y

class DiscreteAQEAFeasibleOptimizer(Optimizer):
    def optimize(self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]) -> Tuple[Any, float, list, list]:
        max_FEs = config.get("max_FEs", 20000)
        history_x, history_y, best_multipliers = run_AQEA_feasible_discrete(initial_state, max_FEs=max_FEs, objective_fn=objective_fn, config=config)
        best_score = history_y[-1] if history_y else 0.0
        return best_multipliers, best_score, history_x, history_y

def run_AQEA_feasible_discrete(base_data, max_FEs=20000, objective_fn=evaluate, config=None):
    if config is None: config = {}
    patch_cfg = config.get("patch_constraints", {})
    max_pressure = patch_cfg.get("max_abs_net_pressure", 1.0) if patch_cfg.get("use_net_patch_pressure", True) else 9999.0
    print(f"--- Starting Discrete AQEA Feasible (budget: {max_FEs} FEs) ---", flush=True)
    pop_size = 40
    num_genes = num_genes_for(base_data)
    probabilities = [
        [1.0 / len(PATCH_LEVELS)] * len(PATCH_LEVELS) for _ in range(num_genes)
    ]
    history_x, history_y = [], []
    gbest_indices = None
    gbest_score = -1.0
    evals = 0
    gen = 0
    stagnant_generations = 0
    
    # Logging stats for ablation
    repair_count = 0
    fallback_to_repair_count = 0
    invalid_candidates_prevented = 0
    
    def generate_feasible_candidate(prob_matrix, max_attempts=50):
        nonlocal invalid_candidates_prevented, fallback_to_repair_count, repair_count
        for _ in range(max_attempts):
            candidate = sample_indices(prob_matrix)
            if abs(patch_pressure_from_indices(candidate)) <= max_pressure:
                return candidate
            invalid_candidates_prevented += 1
        
        fallback_to_repair_count += 1
        repair_count += 1
        return repair_indices(sample_indices(prob_matrix), max_pressure)

    def generate_feasible_neighbor(indices, max_attempts=50):
        nonlocal invalid_candidates_prevented, fallback_to_repair_count, repair_count
        for _ in range(max_attempts):
            candidate = list(indices)
            edits = 1 if random.random() < 0.75 else 2
            for _ in range(edits):
                pos = random.randrange(len(candidate))
                candidate[pos] = min(max(candidate[pos] + random.choice([-1, 1]), 0), len(PATCH_LEVELS) - 1)
            if abs(patch_pressure_from_indices(candidate)) <= max_pressure:
                return candidate
            invalid_candidates_prevented += 1
            
        fallback_to_repair_count += 1
        repair_count += 1
        return repair_indices(candidate, max_pressure) # repair the last failed attempt

    while evals < max_FEs:
        gen += 1
        previous_best = gbest_score
        generation = []
        batch = []
        metadata = []
        
        for i in range(pop_size):
            indices = generate_feasible_candidate(probabilities)
            batch.append(indices)
            metadata.append(('unbalanced', i))
                
        batch = batch[:max_FEs - evals]
        metadata = metadata[:max_FEs - evals]
        if not batch:
            break
            
        batch_results = objective_fn(base_data, batch)
        best_scores_per_individual = {i: -1.0 for i in range(pop_size)}
        best_indices_per_individual = {i: None for i in range(pop_size)}
        
        for m, res in zip(metadata, batch_results):
            score = res[0]
            evals += 1
            i = m[1]
            if score > best_scores_per_individual[i]:
                best_scores_per_individual[i] = score
                best_indices_per_individual[i] = batch[metadata.index(m)]
                
        for i in range(pop_size):
            sol = best_indices_per_individual[i]
            sc = best_scores_per_individual[i]
            if sol is not None:
                generation.append((sc, sol))
                if sc > gbest_score:
                    gbest_score = sc
                    gbest_indices = list(sol)

        local_trials = 0
        if gbest_indices is not None and (stagnant_generations >= 3 or evals >= max_FEs * 0.45):
            local_trials = min(6, max_FEs - evals)
        
        if local_trials > 0:
            local_batch = [generate_feasible_neighbor(gbest_indices) for _ in range(local_trials)]
            local_results = objective_fn(base_data, local_batch)
            for candidate, result in zip(local_batch, local_results):
                score = result[0]
                evals += 1
                generation.append((score, candidate))
                if score > gbest_score:
                    gbest_score = score
                    gbest_indices = list(candidate)

        history_x.append(evals)
        history_y.append(gbest_score)
        if gbest_score <= previous_best + 1e-9:
            stagnant_generations += 1
        else:
            stagnant_generations = 0

        if not generation or gbest_indices is None:
            break

        generation.sort(key=lambda item: item[0], reverse=True)
        elites = select_diverse_elites(
            generation,
            max_elites=max(1, min(6, len(generation) // 4)),
            min_distance=max(1, int(AQEA_DIVERSE_ELITE_DISTANCE * num_genes)),
        )

        learning_rate = 0.12 + min(0.06, max(0, stagnant_generations - 3) * 0.004)
        mutation_rate = 0.006 + min(0.040, max(0, stagnant_generations - 3) * 0.0025)
        probability_floor = AQEA_MIN_PROBABILITY if stagnant_generations >= 4 else 0.0

        update_probabilities(probabilities, gbest_indices, learning_rate, probability_floor)
        for elite in elites:
            update_probabilities(probabilities, elite, learning_rate * 0.35, probability_floor)
        mutate_probabilities(probabilities, mutation_rate)

        if stagnant_generations >= 14:
            soften_probabilities(probabilities, amount=0.08)
        if stagnant_generations >= 30:
            reset_low_entropy_rows(probabilities, fraction=0.12)
            stagnant_generations = 0

        entropy = average_probability_entropy(probabilities)
        from src.utils.progress import print_progress_bar
        print_progress_bar("Discrete Feasible AQEA", min(evals, max_FEs), max_FEs, f"{gbest_score:.4f} E:{entropy:.3f}")

    print(f"\n[Feasible Ablation Stats] repair_count: {repair_count}, fallback: {fallback_to_repair_count}, prevented: {invalid_candidates_prevented}")
    return history_x, history_y, indices_to_multipliers(gbest_indices)



class DiscreteMapElitesOptimizer(Optimizer):
    def optimize(self, objective_fn, base_data, config=None):
        if config is None:
            config = {}
        
        # Hyperparameters
        budget = config.get("fes", 1000)
        pop_size = config.get("population_size", 50)
        scenario = config.get("scenario", "symmetric")
        seed = config.get("seed", 4001)
        eval_seed = config.get("evaluation_seed", 20260612)
        trial = config.get("trial", 0)
        
        full_cfg = config
            
        patch_cfg = full_cfg.get("patch_constraints", {})
        max_abs_net_pressure = patch_cfg.get("max_abs_net_pressure", 1.0)
        
        me_cfg = full_cfg.get("map_elites", {})
        patch_magnitude_bins = me_cfg.get("patch_magnitude_bins", 10)
        diversity_bins = me_cfg.get("diversity_bins", 10)
        pm_min = me_cfg.get("patch_magnitude_min", 0.0)
        pm_max = me_cfg.get("patch_magnitude_max", 1.0)
        div_min = me_cfg.get("diversity_min", 90.0)
        div_max = me_cfg.get("diversity_max", 100.0)
        
        output_dir = full_cfg.get("experiment", {}).get("output_pattern", "results/runs/map_elites_discrete")
        
        from src.algorithms.discrete_algorithms import num_genes_for
        num_genes = num_genes_for(base_data)
        archive = {}
        clamping_events = 0
        
        def get_bin(val, v_min, v_max, bins):
            if val < v_min:
                return 0, True
            if val >= v_max:
                return bins - 1, True
            return math.floor((val - v_min) / (v_max - v_min) * bins), False
        
        def create_offspring():
            if len(archive) == 0:
                return random_feasible_indices(num_genes, max_abs_net_pressure)
            else:
                # Select random elite
                parent_indices = random.choice(list(archive.values()))["patch_vector"]
                # Mutate (flip a few indices)
                child_indices = list(parent_indices)
                num_mutations = max(1, int(num_genes * 0.05))
                from src.algorithms.discrete_algorithms import PATCH_LEVELS
                for _ in range(num_mutations):
                    idx = random.randrange(num_genes)
                    child_indices[idx] = random.randrange(len(PATCH_LEVELS))
                return repair_indices(child_indices, max_abs_net_pressure)

        fes = 0
        best_overall_score = -float("inf")
        best_overall_elite = None
        
        # Evaluate loop
        while fes < budget:
            remaining_fes = budget - fes
            batch_size = min(pop_size, remaining_fes)
            
            # Generate batch
            batch = [create_offspring() for _ in range(batch_size)]
            
            # Evaluate batch
            # evaluate_batch_wrapper_discrete returns:
            # (score, None, None, rbi, mds, pressure, violation, magnitude, Sbalance, Pduration, Sentropy, PL2)
            results = evaluate_batch_wrapper_discrete(
                base_data, batch, 
                return_components=True,
                evaluation_seed=eval_seed,
                scenario=scenario,
                patch_constraints=patch_cfg
            )
            
            fes += batch_size
            
            for i, res in enumerate(results):
                score = res[0]
                diversity_score = res[4]
                patch_magnitude = res[7]
                
                bin_x, clamped_x = get_bin(patch_magnitude, pm_min, pm_max, patch_magnitude_bins)
                bin_y, clamped_y = get_bin(diversity_score, div_min, div_max, diversity_bins)
                if clamped_x or clamped_y:
                    clamping_events += 1
                
                key = (bin_x, bin_y)
                
                # Check constraints explicitly
                net_pres, abs_net_pres, viol, patch_mag_check = discrete_audit_metrics(batch[i], max_abs_net_pressure)
                
                elite = {
                    "bin_x": bin_x,
                    "bin_y": bin_y,
                    "fitness": score,
                    "scalar_score": score,
                    "Sbalance": res[8],
                    "Pduration": res[9],
                    "diversity_score": diversity_score,
                    "rbi": res[3],
                    "PL2": res[11],
                    "patch_magnitude": patch_mag_check,
                    "net_patch_pressure": net_pres,
                    "abs_net_patch_pressure": abs_net_pres,
                    "constraint_violation": viol,
                    "max_abs_net_patch_pressure": max_abs_net_pressure,
                    "patch_vector": batch[i],
                    "seed": seed,
                    "evaluation_seed": eval_seed,
                    "scenario": scenario,
                    "fes": fes
                }
                
                if key not in archive or score > archive[key]["fitness"]:
                    archive[key] = elite
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_elite = elite
                    
        print(f"[MAP-Elites] Clamping events: {clamping_events}")
        
        # Calculate QD metrics
        total_bins = patch_magnitude_bins * diversity_bins
        occupied_bins = len(archive)
        coverage = occupied_bins / total_bins if total_bins > 0 else 0
        qd_score = sum(e["fitness"] for e in archive.values())
        mean_elite_fitness = qd_score / occupied_bins if occupied_bins > 0 else 0
        max_elite_fitness = max(e["fitness"] for e in archive.values()) if occupied_bins > 0 else 0
        
        # Prepare extra metrics for run_lol_parallel_trials.py
        self.extra_metrics = {
            "best_fitness": max_elite_fitness,
            "coverage": coverage,
            "qd_score": qd_score,
            "occupied_bins": occupied_bins,
            "total_bins": total_bins,
            "archive_size": occupied_bins,
            "mean_elite_fitness": mean_elite_fitness,
            "max_elite_fitness": max_elite_fitness
        }
        
        if best_overall_elite is not None:
            self.extra_metrics.update({
                "fitness": best_overall_elite["fitness"],
                "rbi": best_overall_elite["rbi"],
                "mds": best_overall_elite["diversity_score"],
                "diversity_score": best_overall_elite["diversity_score"],
                "completion": max(0.0, 100.0 - best_overall_elite["rbi"] * 100.0),
                "balance_score": max(0.0, 100.0 - best_overall_elite["rbi"] * 100.0),
                "net_patch_pressure": best_overall_elite["net_patch_pressure"],
                "abs_net_patch_pressure": best_overall_elite["abs_net_patch_pressure"],
                "constraint_violation": best_overall_elite["constraint_violation"],
                "patch_magnitude": best_overall_elite["patch_magnitude"]
            })
        
        # Export archive CSV
        archive_rows = []
        for elite in archive.values():
            elite_copy = dict(elite)
            elite_copy["patch_vector"] = json.dumps(elite_copy["patch_vector"])
            archive_rows.append(elite_copy)
            
        # Ensure it's a directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        archive_path = os.path.join(output_dir, f"map_elites_archive_trial_{trial}_seed_{seed}.csv")
        df_archive = pd.DataFrame(archive_rows)
        df_archive.to_csv(archive_path, index=False)
        print(f"[MAP-Elites] Exported archive of size {occupied_bins} to {archive_path}")
        
        # Return best
        best_indices = best_overall_elite["patch_vector"] if best_overall_elite else random_feasible_indices(num_genes, max_abs_net_pressure)
        best_score = best_overall_score
        
        # Dummy history to satisfy Optimizer interface
        hx = [fes]
        hy = [best_score]
        
        # Return indices instead of multipliers because evaluate_discrete handles indices inside run_lol_parallel_trials.py? 
        # Actually run_lol_parallel_trials.py converts best (multipliers) to indices using multipliers_to_indices.
        # So we MUST return multipliers!
        from src.algorithms.discrete_algorithms import indices_to_multipliers
        best_multipliers = indices_to_multipliers(best_indices)
        
        return best_multipliers, best_score, hx, hy


class DiscreteRandomSearchOptimizer(Optimizer):
    def __init__(self):
        self.label = "Discrete Random Search"

    def optimize(
        self, objective_fn, initial_state, config
    ):
        base_data = initial_state
        max_FEs = config.get("max_FEs", 20000)
        pop_size = config.get("population_size", 40)
        import random
        from src.algorithms.discrete_algorithms import PATCH_LEVELS

        print(f"--- Khoi dong {self.label} (Ngan sach: {max_FEs} FEs) ---")
        from src.algorithms.algorithms import GENES_PER_CHAMP
        num_genes = len(base_data) * GENES_PER_CHAMP
        gbest = None
        gbest_score = -1.0
        history_x, history_y = [], []
        evals = 0

        while evals < max_FEs:
            batch = []
            for _ in range(min(pop_size, max_FEs - evals)):
                batch.append([random.randint(0, len(PATCH_LEVELS) - 1) for _ in range(num_genes)])
            
            results = objective_fn(base_data, batch)
            
            for candidate, result in zip(batch, results):
                score = result[0] if isinstance(result, tuple) else result
                evals += 1
                if score > gbest_score:
                    gbest_score = score
                    gbest = candidate.copy()
            
            history_x.append(evals)
            history_y.append(gbest_score)

        return gbest, gbest_score, history_x, history_y
