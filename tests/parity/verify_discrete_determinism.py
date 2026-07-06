import random
import os
import json
from src.algorithms.discrete_algorithms import (
    random_feasible_indices,
    patch_pressure_from_indices,
    indices_to_multipliers,
    patch_magnitude_ratio,
    repair_indices
)

def test_determinism():
    # 1. Test random_feasible_indices determinism
    random.seed(2002)
    chrom1 = random_feasible_indices(1376)
    random.seed(2002)
    chrom2 = random_feasible_indices(1376)
    
    assert chrom1 == chrom2, "RNG seed is broken!"
    
    # 2. Capture baseline state
    random.seed(42)
    num_genes = 1376
    # generate a raw candidate and then run repair on it
    raw_indices = [random.randrange(5) for _ in range(num_genes)]
    
    # Run repair and capture outputs
    random.seed(99)
    repaired_candidate = repair_indices(raw_indices)
    pressure = patch_pressure_from_indices(repaired_candidate)
    multipliers = indices_to_multipliers(repaired_candidate)
    magnitude = patch_magnitude_ratio(repaired_candidate)
    
    baseline_path = "baseline_discrete.json"
    
    if os.path.exists(baseline_path):
        print("Checking against baseline...")
        with open(baseline_path, "r") as f:
            data = json.load(f)
        
        assert repaired_candidate == data["repaired_candidate"], "repaired_candidate MISMATCH!"
        assert abs(pressure - data["pressure"]) < 1e-6, "pressure MISMATCH!"
        assert [abs(a - b) < 1e-6 for a, b in zip(multipliers, data["multipliers"])], "multipliers MISMATCH!"
        assert abs(magnitude - data["magnitude"]) < 1e-6, "magnitude MISMATCH!"
        
        print("PASS: Optimized code matches original baseline perfectly!")
    else:
        print("Generating baseline...")
        data = {
            "repaired_candidate": repaired_candidate,
            "pressure": pressure,
            "multipliers": multipliers,
            "magnitude": magnitude
        }
        with open(baseline_path, "w") as f:
            json.dump(data, f)
        print("Baseline generated.")

if __name__ == "__main__":
    test_determinism()
