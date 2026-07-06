import cProfile
import pstats
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from src.datasets.csv_loader import load_csv_dataset as load_champion_data
from src.evaluation.parallel_evaluator import evaluate_batch
import src.evaluation.parallel_evaluator as pe
from src.algorithms.algorithms import GENES_PER_CHAMP
import random

base_data = load_champion_data("riot_datadragon_champion_stats.csv")
dim = len(base_data) * GENES_PER_CHAMP
rng = random.Random(42)
candidates = [[rng.uniform(0.7, 1.3) for _ in range(dim)] for _ in range(1000)]

pe.USE_GPU = True
# warmup
evaluate_batch(base_data, candidates[:10], mode="continuous", workers=1)

def run():
    evaluate_batch(base_data, candidates, mode="continuous", workers=1)

cProfile.run('run()', 'stats')
p = pstats.Stats('stats')
p.sort_stats('cumulative').print_stats(20)
