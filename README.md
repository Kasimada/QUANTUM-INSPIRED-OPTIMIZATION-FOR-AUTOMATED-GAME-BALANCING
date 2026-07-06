# AQEA Game Balancer

AQEA Game Balancer is a simulation-based optimization framework for automated game balancing.

It supports both:
- Continuous parameter optimization
- Discrete patch generation

using evolutionary and quantum-inspired algorithms.

---

## Features

- **AQEA** (Adaptive Quantum Evolutionary Algorithm)
- **QEA**
- **PSO**
- **GA**
- **NSGA-II**
- **MAP-Elites**
- CPU & GPU execution
- Interactive Launcher
- Automatic result management
- Multi-trial benchmarking

---

## Installation

```bash
git clone https://github.com/danhdz07082005/QUANTUM-INSPIRED-OPTIMIZATION-FOR-AUTOMATED-GAME-BALANCING.git
cd AQEA-Game-Balancer
pip install -r requirements.txt
```

---

## Quick Start

Launch interactive mode:
```bash
Launch.bat
```

Or run directly via CLI:
```bash
python src/run.py --mode continuous --trials 1 --fes 1000
```

---

## Dataset

Default dataset:
`data/raw/original_character_stats.csv`

You may replace it with your own game balance dataset (e.g. 167 champions).

---

## Results

Generated results are automatically managed and stored inside:
`results/`

Each run contains:
- `manifest.json`
- `runs.csv`
- `summary.csv`
- `README.md`
- `convergence.svg`

---

## Documentation

For detailed information, please refer to the dedicated documentation folder:

- 📖 **[Installation Guide](docs/installation.md)**
- 📖 **[Usage Guide](docs/usage.md)**
- 🏗️ **[Architecture](docs/architecture.md)**
- 🎮 **[Game Balance Guide](docs/game_balance_guide.md)**

See: `docs/`

---

## Project Structure
- `launcher/`: Interactive terminal UI
- `src/`: Core framework source code
- `data/`: Core datasets
- `docs/`: Additional documentation
- `scripts/`: Maintenance and validation utilities
- `tests/`: Integrated test suites ensuring simulation parity

---

## License

MIT License