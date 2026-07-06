import sys
import subprocess
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def run_step(name, cmd):
    print("\n=======================================================")
    print(f"STEP: {name}")
    print("=======================================================")
    
    try:
        result = subprocess.run(cmd, check=True, text=True, shell=True)
        print(f"[âœ“] PASS: {name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] FAIL: {name}")
        print(f"Command failed with exit code {e.returncode}")
        return False

def main():
    print("Starting Project Health Check (Mini CI)...")
    
    steps = [
        ("Compile Check", "python -m compileall src launcher -q"),
        ("Public API Import Check", "python -c \"import src.run; from src.algorithms import factory; from launcher import launcher; print('Imports OK')\""),
        ("Dry Run (Continuous)", "python src/run.py --mode continuous --trials 1 --fes 10 --algorithm GA --dry-run"),
        ("Dry Run (Discrete)", "python src/run.py --mode discrete --trials 1 --fes 10 --algorithm GA_discrete --dry-run"),
        ("Smoke Test (Continuous)", "python src/run.py --mode continuous --trials 1 --fes 100 --algorithm GA"),
        ("Smoke Test (Discrete)", "python src/run.py --mode discrete --trials 1 --fes 100 --algorithm GA_discrete")
    ]
    
    for name, cmd in steps:
        success = run_step(name, cmd)
        if not success:
            print("\n[!] Health Check FAILED. Halting pipeline.")
            sys.exit(1)
            
    print("\n[âœ“] All Health Checks Passed Successfully!")
    
if __name__ == "__main__":
    main()
