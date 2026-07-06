# Dataset Directory

This directory contains static character attribute datasets used by the AQEA balancing simulator.

## Schemas
All data files fed into the simulator MUST adhere to the `src/data/schema.py` expected 22 keys.

## Raw vs Processed
* `raw/` contains original, un-mutated source files.
* `processed/` contains simulator-ready files that conform exactly to the required schema.

## Multi-Dataset Conversion Rules

### Pokemon Dataset
* `Name` -> `Name`
* `Class` -> `Type 1`
* `hp` -> `HP`
* `attackdamage` -> `Attack`
* `armor` -> `Defense`
* `spellblock` -> `Sp. Def`
* `movespeed` -> `Speed * 4` (Normalize to LoL average ~300-400)
* `attackspeed` -> `Speed / 100` (Normalize to 0.5 - 1.5)
* `attackrange` -> `150` (Melee if Attack >= Sp.Atk) or `500` (Ranged if Sp.Atk > Attack)
* `mp` -> `Sp. Atk` (proxy for magical capability)
* `hpregen`, `mpregen`, `crit`, and all `*perlevel` stats -> `0.0` (static game, no leveling during simulation).

### DND Dataset
* `Name` -> `alias` (fallback to `name`)
* `Class` -> `justClass`
* `hp` -> `HP * 5` (Normalize ~10 DND HP to ~50 LoL HP)
* `attackdamage` -> `Str * 4` (10 Str -> 40 AD)
* `armor` -> `AC * 2` (15 AC -> 30 Armor)
* `spellblock` -> `Wis * 2` (10 Wis -> 20 MR)
* `movespeed` -> `Dex * 20` (10 Dex -> 200 MS)
* `attackspeed` -> `Dex / 15`
* `attackrange` -> `500` if `weapons` contains `bow/crossbow`, else `150`
* `mp` -> `Int * 10`
* `hpregen`, `mpregen`, `crit`, and all `*perlevel` stats -> `0.0`.
*(Note: To match standard simulator scale ~172, we randomly sampled 200 DND characters for the processed dataset.)*
