from src.algorithms.algorithms import GENES_PER_CHAMP
from src.algorithms.base import Optimizer
from src.utils.progress import print_progress_bar
from typing import Any, Callable, Dict, Tuple
import math
import random




class QEAOptimizer(Optimizer):
    """Quantum-Inspired Evolutionary Algorithm (Base)."""

    def __init__(self, adaptive=False, balanced=False):
        self.adaptive = adaptive
        self.balanced = balanced
        self.label = (
            "Balanced AQEA" if balanced else ("Adaptive AQEA" if adaptive else "QEA")
        )

    def optimize(
        self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]
    ) -> Tuple[Any, float, list, list]:
        base_data = initial_state
        max_FEs = config.get("max_FEs", 20000)
        pop_size = config.get("population_size", 40)

        print(f"--- Khoi dong {self.label} (Ngan sach: {max_FEs} FEs) ---")
        num_genes = len(base_data) * GENES_PER_CHAMP
        angles = [
            [random.uniform(0.0, math.pi / 2.0) for _ in range(num_genes)]
            for _ in range(pop_size)
        ]
        gbest = None
        gbest_score = -1.0
        history_x, history_y = [], []
        evals = 0
        gen = 0
        stagnant_generations = 0
        max_gen_estimate = max(1, max_FEs / (pop_size * 3))

        while evals < max_FEs:
            gen += 1
            previous_best = gbest_score
            generation = []

            batch = []
            metadata = []
            
            for i in range(pop_size):
                if not self.balanced:
                    for k in range(3):
                        candidate = []
                        for j in range(num_genes):
                            # Quantum collapse using Beta distribution
                            # Mean = cos^2(theta)
                            alpha = (math.cos(angles[i][j]) ** 2) * 10.0 + 1.0
                            beta_param = (math.sin(angles[i][j]) ** 2) * 10.0 + 1.0
                            sampled = random.betavariate(alpha, beta_param)
                            # Map from [0, 1] to [0.7, 1.3]
                            candidate.append(0.7 + 0.6 * sampled)
                        batch.append(candidate)
                        metadata.append(('unbalanced', i, k))
                else:
                    c_alpha = [0.7 + 0.6 * (math.cos(angles[i][j]) ** 2) for j in range(num_genes)]
                    c_beta = [0.7 + 0.6 * (math.sin(angles[i][j]) ** 2) for j in range(num_genes)]
                    batch.extend([c_alpha, c_beta])
                    metadata.extend([('balanced_alpha', i), ('balanced_beta', i)])
                    
            # Trim to max_FEs
            batch = batch[:max_FEs - evals]
            metadata = metadata[:max_FEs - evals]
            
            if not batch:
                break
                
            batch_results = objective_fn(base_data, batch)
            
            best_scores_per_individual = {i: -1 for i in range(pop_size)}
            best_solutions_per_individual = {i: None for i in range(pop_size)}
            
            for m, res in zip(metadata, batch_results):
                score = res[0] if isinstance(res, tuple) else res
                evals += 1
                
                if m[0] == 'unbalanced':
                    i = m[1]
                    if score > best_scores_per_individual[i]:
                        best_scores_per_individual[i] = score
                        best_solutions_per_individual[i] = batch[metadata.index(m)]
                elif m[0].startswith('balanced'):
                    i = m[1]
                    # For balanced, we keep track of the best out of alpha/beta
                    if score > best_scores_per_individual[i]:
                        best_scores_per_individual[i] = score
                        best_solutions_per_individual[i] = batch[metadata.index(m)]
                        
            for i in range(pop_size):
                sol = best_solutions_per_individual[i]
                sc = best_scores_per_individual[i]
                if sol is not None:
                    generation.append((sc, sol))
                    if sc > gbest_score:
                        gbest_score = sc
                        gbest = list(sol)

            local_trials = 0
            if self.adaptive and (stagnant_generations >= 3 or evals >= max_FEs * 0.45):
                local_trials = min(6, max_FEs - evals)
            if gbest is not None and local_trials > 0:
                progress = evals / max(1, max_FEs)
                local_scale = 0.09 * (1.0 - progress) + 0.015
                local_batch = []
                for _ in range(local_trials):
                    candidate = [
                        min(1.3, max(0.7, value + random.gauss(0.0, local_scale)))
                        for value in gbest
                    ]
                    local_batch.append(candidate)
                    
                local_results = objective_fn(base_data, local_batch)
                
                for candidate, result in zip(local_batch, local_results):
                    score = result[0] if isinstance(result, tuple) else result
                    evals += 1
                    generation.append((score, candidate))
                    if score > gbest_score:
                        gbest_score = score
                        gbest = list(candidate)

            history_x.append(evals)
            history_y.append(gbest_score)
            if gbest_score <= previous_best + 1e-9:
                stagnant_generations += 1
            else:
                stagnant_generations = 0

            generation.sort(key=lambda item: item[0], reverse=True)
            elites = generation[: max(1, min(6, len(generation) // 4))]
            theta_base = 0.055 * math.pi
            if self.balanced or self.adaptive:
                progress = min(1.0, evals / max(1, max_FEs))
                annealed_step = 0.55 + 0.45 * math.exp(-0.55 * progress)
                stagnation_boost = min(0.35, stagnant_generations * 0.025)
                rotation_factor = min(1.15, annealed_step + stagnation_boost)
                mutation_rate = 0.018 + min(0.075, stagnant_generations * 0.004)
                mutation_span = 0.08 * math.pi + min(
                    0.14 * math.pi, stagnant_generations * 0.006 * math.pi
                )
            else:
                rotation_factor = 1.0
                mutation_rate = 0.018
                mutation_span = 0.06 * math.pi

            for i in range(pop_size):
                if not elites or gbest is None:
                    break
                elite_solution = elites[i % len(elites)][1]
                for j in range(num_genes):
                    target_value = 0.75 * gbest[j] + 0.25 * elite_solution[j]
                    safe_target = min(1.3, max(0.7, target_value))
                    target_cos_sq = (safe_target - 0.7) / 0.6
                    target_cos_sq = min(1.0, max(0.0, target_cos_sq))
                    target_angle = math.acos(math.sqrt(target_cos_sq))

                    if target_angle > angles[i][j]:
                        direction = 1.0
                    elif target_angle < angles[i][j]:
                        direction = -1.0
                    else:
                        direction = 0.0

                    # Aging Mechanism & Adaptive Rotation
                    aging_factor = math.exp(-0.55 * (evals / max(1, max_FEs)))
                    adaptive_rotation = theta_base * rotation_factor * aging_factor

                    new_angle = angles[i][j] + direction * adaptive_rotation
                    
                    if random.random() < mutation_rate:
                        new_angle += random.uniform(-mutation_span, mutation_span)
                    
                    angles[i][j] = min(max(new_angle, 0.0), math.pi / 2.0)

            if self.adaptive and stagnant_generations >= 18:
                for i in range(pop_size // 2, pop_size):
                    angles[i] = [
                        random.uniform(0.0, math.pi / 2.0) for _ in range(num_genes)
                    ]
                stagnant_generations = 0

            from src.utils.progress import print_progress_bar
            print_progress_bar(self.label, min(evals, max_FEs), max_FEs, gbest_score)

        return gbest, gbest_score, history_x, history_y


class AQEAOptimizer(QEAOptimizer):
    def __init__(self):
        super().__init__(adaptive=True, balanced=False)







class GeneticOptimizer(Optimizer):
    """Genetic Algorithm Optimizer."""

    def optimize(
        self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]
    ) -> Tuple[Any, float, list, list]:
        base_data = initial_state
        max_FEs = config.get("max_FEs", 20000)
        pop_size = config.get("population_size", 50)

        print(f"--- Khởi động GA (Ngân sách: {max_FEs} FEs) ---")
        num_genes = len(base_data) * GENES_PER_CHAMP
        population = [
            [random.uniform(0.7, 1.3) for _ in range(num_genes)]
            for _ in range(pop_size)
        ]

        history_x, history_y = [], []
        best_score = -1
        best_chromosome = None
        evals = 0
        gen = 0

        while evals < max_FEs:
            gen += 1
            scores = []
            # Determine batch size based on remaining FEs
            batch = population[:max_FEs - evals]
            if not batch:
                break
                
            batch_results = objective_fn(base_data, batch)
            
            for chrom, result in zip(batch, batch_results):
                score = result[0] if isinstance(result, tuple) else result
                evals += 1
                scores.append(score)
                if score > best_score:
                    best_score = score
                    best_chromosome = list(chrom)

            history_x.append(evals)
            history_y.append(best_score)

            new_population = [list(best_chromosome)]  # Elitism
            while len(new_population) < pop_size:
                idx_pool1 = random.sample(
                    range(min(len(scores), pop_size)), min(2, len(scores))
                )
                p1_idx = max(idx_pool1, key=lambda i: scores[i])
                idx_pool2 = random.sample(
                    range(min(len(scores), pop_size)), min(2, len(scores))
                )
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

            from src.utils.progress import print_progress_bar
            print_progress_bar("GA", min(evals, max_FEs), max_FEs, best_score)

        return best_chromosome, best_score, history_x, history_y





class PSOOptimizer(Optimizer):
    """Particle Swarm Optimization."""

    def optimize(
        self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]
    ) -> Tuple[Any, float, list, list]:
        base_data = initial_state
        max_FEs = config.get("max_FEs", 20000)
        pop_size = config.get("population_size", 30)

        print(f"--- Khởi động PSO (Ngân sách: {max_FEs} FEs) ---")
        num_genes = len(base_data) * GENES_PER_CHAMP

        particles = [
            [random.uniform(0.7, 1.3) for _ in range(num_genes)]
            for _ in range(pop_size)
        ]
        velocities = [
            [random.uniform(-0.1, 0.1) for _ in range(num_genes)]
            for _ in range(pop_size)
        ]
        pbest = [list(p) for p in particles]
        pbest_scores = [-1] * pop_size

        gbest = None
        gbest_score = -1
        history_x, history_y = [], []
        evals = 0
        gen = 0

        c1, c2 = 2.0, 2.0
        max_gen_estimate = max_FEs / pop_size

        while evals < max_FEs:
            gen += 1
            w = 0.9 - 0.5 * (gen / max_gen_estimate)
            w = max(w, 0.4)

            # Determine batch size based on remaining FEs
            batch = particles[:max_FEs - evals]
            if not batch:
                break
                
            batch_results = objective_fn(base_data, batch)
            
            for i, result in enumerate(batch_results):
                score = result[0] if isinstance(result, tuple) else result
                evals += 1

                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = list(particles[i])
                if score > gbest_score:
                    gbest_score = score
                    gbest = list(particles[i])

            history_x.append(evals)
            history_y.append(gbest_score)

            for i in range(pop_size):
                for j in range(num_genes):
                    r1, r2 = random.random(), random.random()
                    velocities[i][j] = (
                        w * velocities[i][j]
                        + c1 * r1 * (pbest[i][j] - particles[i][j])
                        + c2 * r2 * (gbest[j] - particles[i][j])
                    )

                    velocities[i][j] = min(max(velocities[i][j], -0.3), 0.3)
                    particles[i][j] = min(
                        max(particles[i][j] + velocities[i][j], 0.7), 1.3
                    )

            from src.utils.progress import print_progress_bar
            print_progress_bar("PSO", min(evals, max_FEs), max_FEs, gbest_score)

        return gbest, gbest_score, history_x, history_y





def non_dominated_sorting(population_scores):
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


def crowding_distance(front, population_scores):
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


class NSGA2Optimizer(Optimizer):
    """NSGA-II Multi-objective Evolutionary Algorithm."""

    def optimize(
        self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]
    ) -> Tuple[Any, float, list, list]:
        base_data = initial_state
        max_FEs = config.get("max_FEs", 20000)
        pop_size = config.get("population_size", 50)

        print(f"--- Khởi động NSGA-II (Ngân sách: {max_FEs} FEs) ---")
        num_genes = len(base_data) * GENES_PER_CHAMP
        population = [
            [random.uniform(0.7, 1.3) for _ in range(num_genes)]
            for _ in range(pop_size)
        ]

        history_x, history_y = [], []
        best_score = -1
        best_chromosome = None
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
                    best_chromosome = list(chrom)

            history_x.append(evals)
            history_y.append(best_score)

            if evals >= max_FEs:
                break

            fronts = non_dominated_sorting(pop_objs)
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= pop_size:
                    new_population.extend([population[i] for i in front])
                else:
                    dist = crowding_distance(front, pop_objs)
                    front_sorted = sorted(front, key=lambda i: dist[i], reverse=True)
                    needed = pop_size - len(new_population)
                    new_population.extend(
                        [population[i] for i in front_sorted[:needed]]
                    )
                    break

            offspring = []
            while len(offspring) < pop_size:
                p1_idx = random.randint(0, pop_size // 2)
                p2_idx = random.randint(0, pop_size // 2)
                p1 = new_population[p1_idx]
                p2 = new_population[p2_idx]

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
                offspring.append(child)
            population = offspring

            from src.utils.progress import print_progress_bar
            print_progress_bar("NSGA-II", min(evals, max_FEs), max_FEs, best_score)

        return best_chromosome, best_score, history_x, history_y




class ContinuousRandomSearchOptimizer(Optimizer):
    def optimize(
        self, objective_fn: Callable, initial_state: Any, config: Dict[str, Any]
    ) -> Tuple[Any, float, list, list]:
        max_FEs = config.get("max_FEs", 20000)
        scenario = config.get("scenario", "symmetric")
        num_genes = len(initial_state) * GENES_PER_CHAMP

        print(f"--- Khoi dong Continuous Random Search (Ngan sach: {max_FEs} FEs) ---")
        best_score = -1.0
        best_chromosome = None
        history_x, history_y = [], []
        evals = 0

        # We evaluate in batches for efficiency (matching other algorithms)
        batch_size = min(max_FEs, 100)
        
        while evals < max_FEs:
            current_batch_size = min(batch_size, max_FEs - evals)
            batch = [
                [random.uniform(0.7, 1.3) for _ in range(num_genes)]
                for _ in range(current_batch_size)
            ]
            
            # Pass scenario down via config (handled in wrapper)
            batch_results = objective_fn(initial_state, batch, scenario=scenario)
            
            for chrom, result in zip(batch, batch_results):
                score = result[0]
                evals += 1
                if score > best_score:
                    best_score = score
                    best_chromosome = list(chrom)
                    
            history_x.append(evals)
            history_y.append(best_score)
            
            print_progress_bar("Continuous Random Search", evals, max_FEs, best_score)
            
        return best_chromosome, best_score, history_x, history_y



