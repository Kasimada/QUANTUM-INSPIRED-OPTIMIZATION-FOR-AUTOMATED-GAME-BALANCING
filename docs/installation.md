# Installation Guide

Setting up the AQEA Game Balancer is simple.

## 1. Prerequisites
- **Python:** 3.9 - 3.14
- **OS:** Windows / Linux / macOS
- **Hardware:** CPU is fully supported. An NVIDIA GPU is optional but highly recommended if you are processing the full dataset (167+ champions).

## 2. Clone the Repository
Open your terminal and clone the repository:
```bash
git clone https://github.com/your-repo.git
cd AQEA-Game-Balancer
```

## 3. Install Dependencies
Run the following command to install all required libraries (including PyTorch, NumPy, Pandas, etc.):
```bash
pip install -r requirements.txt
```

## 4. CUDA Setup (Optional but Recommended)
If you have an NVIDIA GPU and want to leverage hardware acceleration (`--device auto`), ensure you have the appropriate NVIDIA drivers installed.

> [!WARNING]
> **Windows Installation Pitfall:**
> Running `pip install -r requirements.txt` on Windows will often install the **CPU-only** version of PyTorch by default. If your launcher says `Torch: 2.x.x+cpu (CUDA: No)`, you must manually install the CUDA-enabled version.
> 
> To fix this and enable GPU acceleration, run:
> ```bash
> pip uninstall torch torchvision torchaudio -y
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

To verify your installation, you can run:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

## 5. Next Steps
You are now ready to run the project. Head over to the [Usage Guide](usage.md) or launch the interactive dashboard:
```bash
Launch.bat
```
