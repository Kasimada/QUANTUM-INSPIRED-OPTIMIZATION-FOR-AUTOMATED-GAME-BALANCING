import os

REQUIRED_KEYS = [
    "Name", "Class", "hp", "hpperlevel", "mp", "mpperlevel", "movespeed", 
    "armor", "armorperlevel", "spellblock", "spellblockperlevel", "attackrange", 
    "hpregen", "hpregenperlevel", "mpregen", "mpregenperlevel", "crit", 
    "critperlevel", "attackdamage", "attackdamageperlevel", "attackspeed", 
    "attackspeedperlevel"
]

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    
