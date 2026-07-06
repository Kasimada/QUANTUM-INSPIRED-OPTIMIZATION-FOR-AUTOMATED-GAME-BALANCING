import ast
import json
import sys
from pathlib import Path
from typing import Union, Dict, List, Any

import pandas as pd
from src.datasets.schema import validate_dataset

def parse_stats(stats_str: str) -> dict:
    if pd.isna(stats_str):
        return {}
    try:
        return json.loads(stats_str)
    except Exception:
        try:
            return ast.literal_eval(stats_str)
        except Exception:
            return {}

def stat_get(stats_dict: dict, riot_key: str, legacy_key: str, default: float) -> float:
    if riot_key in stats_dict:
        return float(stats_dict.get(riot_key, default))
    return float(stats_dict.get(legacy_key, default))

def _load_legacy_riot_data(df: pd.DataFrame) -> List[Dict[str, Any]]:
    base_stats = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            stats_dict = parse_stats(row.get("stats", "{}"))
            champ_class = row.get("herotype")
            if pd.isna(champ_class):
                champ_class = "Fighter"
            
            champ = {
                "Name": row.get("apiname", "Unknown"),
                "Class": champ_class,
                "hp": stat_get(stats_dict, "hp", "hp_base", 500),
                "hpperlevel": stat_get(stats_dict, "hpperlevel", "hp_lvl", 0),
                "mp": stat_get(stats_dict, "mp", "mp_base", 0),
                "mpperlevel": stat_get(stats_dict, "mpperlevel", "mp_lvl", 0),
                "movespeed": stat_get(stats_dict, "movespeed", "ms", 330),
                "armor": stat_get(stats_dict, "armor", "arm_base", 30),
                "armorperlevel": stat_get(stats_dict, "armorperlevel", "arm_lvl", 0),
                "spellblock": stat_get(stats_dict, "spellblock", "mr_base", 30),
                "spellblockperlevel": stat_get(stats_dict, "spellblockperlevel", "mr_lvl", 0),
                "attackrange": stat_get(stats_dict, "attackrange", "range", 125),
                "hpregen": stat_get(stats_dict, "hpregen", "hp5_base", 0),
                "hpregenperlevel": stat_get(stats_dict, "hpregenperlevel", "hp5_lvl", 0),
                "mpregen": stat_get(stats_dict, "mpregen", "mp5_base", 0),
                "mpregenperlevel": stat_get(stats_dict, "mpregenperlevel", "mp5_lvl", 0),
                "crit": stat_get(stats_dict, "crit", "crit_base", 0),
                "critperlevel": stat_get(stats_dict, "critperlevel", "crit_lvl", 0),
                "attackdamage": stat_get(stats_dict, "attackdamage", "dam_base", 50),
                "attackdamageperlevel": stat_get(stats_dict, "attackdamageperlevel", "dam_lvl", 0),
                "attackspeed": stat_get(stats_dict, "attackspeed", "as_base", 0.625),
                "attackspeedperlevel": stat_get(stats_dict, "attackspeedperlevel", "as_lvl", 0),
            }
            base_stats.append(champ)
        except Exception:
            skipped += 1
            
    print(f"[CSV Loader] Legacy Riot data loaded. Skipped {skipped} rows.")
    return base_stats

def load_csv_dataset(dataset_config: Union[str, Dict] = "data/raw/original_character_stats.csv") -> List[Dict[str, Any]]:
    """
    Generic CSV loader that supports standard CSV files and legacy Riot dataset structures.
    """
    if isinstance(dataset_config, dict):
        filepath = dataset_config.get("path", "data/raw/original_character_stats.csv")
    else:
        filepath = dataset_config

    path_obj = Path(filepath)
    
    # Backward compatibility for old configs
    if filepath == "riot_datadragon_champion_stats.csv" and not path_obj.exists():
        fallback = Path("data/raw/original_character_stats.csv")
        if fallback.exists():
            print(f"[CSV Loader] Redirecting legacy path '{filepath}' to '{fallback}'")
            path_obj = fallback
            filepath = str(fallback)

    print(f"[CSV Loader] Reading data from: {filepath}")
    try:
        df = pd.read_csv(path_obj)
    except FileNotFoundError:
        print(f"[CSV Loader] Error: File not found {filepath}", file=sys.stderr)
        return []
        
    # Handle missing values broadly safely before conversion
    df = df.where(pd.notnull(df), None)

    # Detect if legacy Riot data format
    if "stats" in df.columns and "apiname" in df.columns:
        print(f"[CSV Loader] Detected legacy Riot data format in {filepath}")
        base_stats = _load_legacy_riot_data(df)
    else:
        # Generic CSV format mapped directly to records
        base_stats = []
        skipped = 0
        for _, row in df.iterrows():
            try:
                # Basic conversion for clean dict without None/NaN causing issues
                record = {k: v for k, v in row.to_dict().items() if v is not None}
                base_stats.append(record)
            except Exception:
                skipped += 1
        print(f"[CSV Loader] Loaded {len(base_stats)} rows generic CSV. Skipped {skipped} rows.")

    # Validate against expected schema
    is_valid, missing_keys = validate_dataset(base_stats)
    if not is_valid:
        error_msg = (
            f"Dataset validation failed:\n"
            f"Missing required columns: {', '.join(missing_keys)}\n"
            f"File: {filepath}"
        )
        raise ValueError(error_msg)
        
    return base_stats
