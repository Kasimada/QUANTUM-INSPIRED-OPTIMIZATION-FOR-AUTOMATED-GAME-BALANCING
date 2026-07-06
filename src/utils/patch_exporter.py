import csv
import json
from pathlib import Path
from src.algorithms.algorithms import apply_chromosome

def export_patch(base_data, best_representation, out_path):
    """
    Exports the modified character stats (the patch) to a CSV file.
    
    CRITICAL: This function reuses the exact same decoding pipeline (`apply_chromosome`)
    used by the objective evaluation. It does NOT reimplement any decoding logic.
    """
    # 1. Reuse the exact decoding pipeline used in evaluate()
    mutated_data = apply_chromosome(base_data, best_representation)
    
    if not mutated_data:
        return
        
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Extract column headers from the first dictionary
    fieldnames = list(mutated_data[0].keys())
    
    # 3. Write to CSV
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mutated_data)

def save_best_patch_metadata(algorithm_dir, best_trial, trial_csv_name):
    """
    Saves a metadata file pointing to the best trial's CSV, avoiding file duplication.
    """
    meta_path = Path(algorithm_dir) / "best_patch.json"
    metadata = {
        "best_trial": best_trial,
        "best_file": trial_csv_name
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
