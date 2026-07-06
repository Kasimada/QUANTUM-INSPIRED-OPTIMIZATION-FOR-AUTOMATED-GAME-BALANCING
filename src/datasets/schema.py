from typing import Any, Dict, List, Tuple

REQUIRED_SIMULATION_KEYS = [
    "Name",
    "Class",
    "hp",
    "hpperlevel",
    "mp",
    "mpperlevel",
    "movespeed",
    "armor",
    "armorperlevel",
    "spellblock",
    "spellblockperlevel",
    "attackrange",
    "hpregen",
    "hpregenperlevel",
    "mpregen",
    "mpregenperlevel",
    "crit",
    "critperlevel",
    "attackdamage",
    "attackdamageperlevel",
    "attackspeed",
    "attackspeedperlevel",
]

def validate_dataset(champions: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Validates that the parsed champion list contains all keys required by the simulator.
    """
    if not champions:
        return False, []

    missing_keys = set()
    for champ in champions:
        for key in REQUIRED_SIMULATION_KEYS:
            if key not in champ:
                missing_keys.add(key)
                
    if missing_keys:
        return False, list(missing_keys)
        
    return True, []
