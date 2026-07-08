# AQEA Game Balancer - Usage Guide

## Using the Launcher
The easiest way to start an optimization run is to use the interactive launcher:
```bash
Launch.bat
```
The launcher provides a live, interactive dashboard. To quickly start a run, you only need to configure the minimum required options:
1. Press `1` to select your **Dataset**.
2. Press `2` to select the **Mode** (Continuous/Discrete/Both).
3. (Optional) Press `3` to choose a **Scenario** (default is symmetric).
4. Press `4` to select the **Algorithms** (toggle numbers, then type `done` to finish).
5. Press `0` to **Review Configuration**.
6. Press `S` to **Start** the experiment!

You can also customize Trials (`5`), Max FEs (`6`), Workers (`7`), or configure profiles (Save/Load).

## Understanding Configuration Options
- **Workers**: The number of parallel processes spawned during evaluation. It is recommended to leave at least 2 cores free (e.g., if you have 8 cores, use 6 workers) to prevent system freezing.
- **Selecting Algorithms**: In the interactive menu, selecting algorithms acts as a toggle. Type the number to check `[x]`, and type it again to uncheck `[ ]`. Type `done` to confirm your selection.
- **Device (GPU/CPU)**: The launcher automatically detects if a compatible NVIDIA GPU (CUDA) is available. If not, it defaults to CPU and hides the GPU option to prevent crashes.
- **Seed**: A random seed ensures reproducibility. If you input a specific number (e.g., 42), the algorithms will produce the exact same results every time. If left blank, a default seed is used.
- **Output Folder & Layout**: By default, the system automatically saves results to `results/` using a smart layout. You can change the layout (Option `11`) to group by Date + Config (e.g. `2026-07-04/001_discrete`), Date only, Config only, or Flat. The system is equipped with an `OutputManager` that automatically detects collisions, warns you about interrupted runs (stale locks), and prevents accidental overwrites.
- **Import/Export Config**: You can use Option `14` to Export your current configuration to a JSON file (e.g., `benchmark.json`), and share it with colleagues. They can use Option `15` to Import the JSON file and instantly replicate your exact setup!

## Using the CLI directly
For automation, you can run the core runner directly:
```bash
# Continuous Balancing Mode
python src/run.py --mode continuous --trials 1 --fes 1000

# Discrete Patching Mode
python src/run.py --mode discrete --trials 1 --fes 1000
```
For more details on CLI arguments, run `python src/run.py --help`.

## Outputs
When an experiment finishes, a timestamped folder is created inside `results/`. This folder contains:
- `manifest.json`: Configuration and metadata for the run.
- `README.md`: A human-readable summary of the experiment's findings.
- `runs.csv`: Raw metrics for every trial and algorithm. **All comprehensive legacy columns (like optimizer_backend, final_logged_fe, etc.) are fully preserved.**
- `summary.csv`: Averaged metrics grouped by algorithm.
- `convergence.svg`: An aesthetic visual representation of the convergence speed.
- `patches/`: In discrete mode, this folder contains the generated patch configurations for each algorithm.

## Supported Algorithms
- **AQEA (Adaptive Quantum Evolutionary Algorithm)**: Our proposed hybrid mechanism (Recommended).
- **QEA**: Standard Quantum Evolutionary Algorithm.
- **PSO**: Particle Swarm Optimization.
- **GA**: Standard Genetic Algorithm.
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II.
- **Random Search**: Baseline Random Search.
- **MAP-Elites**: Quality-Diversity optimization (Discrete mode only).
