# System Architecture

## Core Design
The project utilizes a modern **Strategy Pattern** for evaluations, ensuring that logic can easily switch between CPU multiprocessing and GPU tensor acceleration depending on the environment.

### Evaluation Backend
- **`EvaluationBackend`**: The abstract base class that defines how a simulation batch should be evaluated.
- **`CPUBackend`**: Handles continuous and discrete simulations via standard Python multiprocessing. Ideal for systems without dedicated GPUs or for small datasets.
- **`GPUBackend`**: Uses PyTorch for mass tensor operations. Features built-in Out-Of-Memory (OOM) safety and fallback mechanisms. Highly recommended when evaluating the full champion dataset.

### Supporting Components
- **`DeviceManager`**: Dynamically detects hardware capabilities (CUDA availability) and seamlessly routes work to the appropriate backend.
- **`OutputManager`**: Handles intelligent folder naming (e.g. `001_continuous_10T_1000FEs_6Algo`), prevents accidental overwrites of existing data, and warns about stale locks from interrupted runs.
- **`EvaluationResult`**: A robust dataclass returning `fitness`, `metrics` (like Relative Balance Index), and `metadata`.
- **Manifest Schema**: Every output folder contains a `manifest.json` ensuring 100% reproducibility. It stores the seed, configurations, layout choices, and timestamps.
