import os
import json
import time
import platform
import multiprocessing
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from src.version import APP_NAME, __version__
from src.utils.report_generator import ReportGenerator

class OutputLayout(Enum):
    DEFAULT = "default"
    DATE = "date"
    CONFIG = "config"
    DATE_CONFIG = "date_config"
    FLAT = "flat"

@dataclass
class OutputInfo:
    experiment_name: str
    relative_path: str
    absolute_path: str
    layout: str
    date: str
    run_index: Optional[int]
    created_new: bool

@dataclass
class CollisionResult:
    exists: bool
    has_lock: bool
    has_manifest: bool
    status: Optional[str]
    pid: Optional[int]

class OutputManager:
    """
    Manages the creation, locking, and manifesting of output directories.
    Provides a context manager for safe execution and cleanup.
    """
    def __init__(
        self, 
        base_path: str, 
        layout: OutputLayout, 
        config_data: Dict[str, Any],
        launcher_profile: Optional[str] = None
    ):
        self.base_path = Path(base_path).resolve()
        self.layout = layout
        self.config_data = config_data
        self.launcher_profile = launcher_profile
        
        self.out_info: Optional[OutputInfo] = None
        self._lock_file: Optional[Path] = None
        self._manifest_file: Optional[Path] = None
        self._start_time: float = 0.0

    def validate_path(self) -> None:
        """Validates the base_path for safety and constraints."""
        p = str(self.base_path).lower()
        if p.startswith("c:\\windows") or p.startswith("c:\\program files"):
            raise ValueError(f"Cannot output to system directory: {self.base_path}")
        
        # Test basic write permissions if exists, or parent if it doesn't
        test_dir = self.base_path
        while not test_dir.exists() and test_dir.parent != test_dir:
            test_dir = test_dir.parent
            
        if not os.access(test_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory: {test_dir}")

    def detect_collision(self) -> CollisionResult:
        """Checks if the proposed layout results in a collision at the target directory."""
        target_dir = self._resolve_target_dir()
        
        if not target_dir.exists():
            return CollisionResult(False, False, False, None, None)
            
        has_lock = (target_dir / ".running.lock").exists()
        has_manifest = (target_dir / "manifest.json").exists()
        
        status = None
        pid = None
        
        if has_manifest:
            try:
                with open(target_dir / "manifest.json", "r", encoding="utf-8") as f:
                    man = json.load(f)
                    status = man.get("status")
            except Exception:
                pass
                
        if has_lock:
            try:
                with open(target_dir / ".running.lock", "r", encoding="utf-8") as f:
                    l = json.load(f)
                    pid = l.get("pid")
            except Exception:
                pass
                
        return CollisionResult(True, has_lock, has_manifest, status, pid)

    def _resolve_target_dir(self) -> Path:
        """Determines the exact target directory based on the layout."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        if self.layout == OutputLayout.FLAT:
            return self.base_path
        elif self.layout == OutputLayout.DATE:
            return self.base_path / today_str
        elif self.layout == OutputLayout.CONFIG:
            return self.base_path / self._allocate_run_index(self.base_path)
        elif self.layout == OutputLayout.DATE_CONFIG or self.layout == OutputLayout.DEFAULT:
            date_dir = self.base_path / today_str
            return date_dir / self._allocate_run_index(date_dir)
        
        return self.base_path

    def _allocate_run_index(self, parent_dir: Path) -> str:
        """Scans the parent directory to allocate the next 00X_ prefix."""
        mode = self.config_data.get('mode', 'run')
        trials = self.config_data.get('trials', 1)
        fes = self.config_data.get('fes', 100)
        
        alg_val = self.config_data.get('algorithms', [])
        if isinstance(alg_val, list):
            algo_count = len(alg_val)
        elif isinstance(alg_val, str) and alg_val:
            algo_count = len(alg_val.split(','))
        else:
            algo_count = 1
            
        suffix = f"_{mode}_{trials}T_{fes}FEs_{algo_count}Algo"
        
        if not parent_dir.exists():
            return f"001{suffix}"
            
        max_idx = 0
        for item in parent_dir.iterdir():
            if item.is_dir():
                parts = item.name.split('_')
                if len(parts) > 1 and parts[0].isdigit() and parts[1] == mode:
                    idx = int(parts[0])
                    if idx > max_idx:
                        max_idx = idx
                        
        next_idx = max_idx + 1
        return f"{next_idx:03d}{suffix}"

    def get_latest_dir(self) -> Path:
        """Returns the most recently created run directory for the current mode."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = self.base_path / today_str
        if not date_dir.exists():
            return self.base_path
            
        mode = self.config_data.get('mode', 'run')
        max_idx = 0
        latest_dir = date_dir
        for item in date_dir.iterdir():
            if item.is_dir():
                parts = item.name.split('_')
                if len(parts) > 1 and parts[0].isdigit() and parts[1] == mode:
                    idx = int(parts[0])
                    if idx > max_idx:
                        max_idx = idx
                        latest_dir = item
        return latest_dir

    def prepare(self) -> OutputInfo:
        """Resolves the output directory and creates it."""
        self.validate_path()
        target_dir = self._resolve_target_dir()
        
        created_new = not target_dir.exists()
        target_dir.mkdir(parents=True, exist_ok=True)
        
        rel_path = str(target_dir.relative_to(self.base_path)) if self.base_path in target_dir.parents else target_dir.name
        
        # Extract run index if any
        run_index = None
        parts = target_dir.name.split('_')
        if len(parts) > 0 and parts[0].isdigit():
            run_index = int(parts[0])
            
        today_str = datetime.now().strftime("%Y-%m-%d")
            
        self.out_info = OutputInfo(
            experiment_name=target_dir.name,
            relative_path=rel_path,
            absolute_path=str(target_dir),
            layout=self.layout.value,
            date=today_str,
            run_index=run_index,
            created_new=created_new
        )
        
        return self.out_info

    def __enter__(self):
        if not self.out_info:
            self.prepare()
            
        out_dir = Path(self.out_info.absolute_path)
        self._lock_file = out_dir / ".running.lock"
        self._manifest_file = out_dir / "manifest.json"
        
        # Create lock
        import sys
        lock_data = {
            "pid": os.getpid(),
            "hostname": platform.node(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "launcher": self.launcher_profile is not None,
            "version": __version__,
            "command_line": " ".join(sys.argv),
            "working_directory": os.getcwd()
        }
        with open(self._lock_file, "w", encoding="utf-8") as f:
            json.dump(lock_data, f, indent=4)
            
        # Create immutable initial manifest
        self._write_manifest(status="running")
        self._start_time = time.time()
        
        return self.out_info

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._start_time
        
        if exc_type is None:
            self._update_manifest_status("completed", duration)
        else:
            self._update_manifest_status("failed", duration)
            
        # Remove lock
        if self._lock_file and self._lock_file.exists():
            try:
                self._lock_file.unlink()
            except Exception:
                pass
                
        # Generate Readme
        if self.out_info:
            ReportGenerator.generate(Path(self.out_info.absolute_path))
            
    def _get_system_info(self) -> dict:
        sys_info = {
            "os": platform.system() + " " + platform.release(),
            "cpu": platform.processor(),
            "cores": multiprocessing.cpu_count(),
            "python": platform.python_version(),
            "hostname": platform.node()
        }
        try:
            import subprocess
            model_name = subprocess.check_output('wmic csproduct get name', shell=True).decode('utf-8').strip().split('\n')[-1].strip()
            if model_name:
                sys_info["machine_model"] = model_name
        except Exception:
            pass
        try:
            import psutil
            sys_info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            pass
        return sys_info

    def _write_manifest(self, status: str) -> None:
        if not self._manifest_file or not self.out_info:
            return
            
        # Only rewrite the whole thing if it doesn't exist
        if not self._manifest_file.exists():
            manifest = {
                "schema_version": 1,
                "app_version": __version__,
                "app_name": APP_NAME,
                "layout": self.out_info.layout,
                "launcher": self.launcher_profile is not None,
                "launcher_profile": self.launcher_profile,
                "run_index": self.out_info.run_index,
                "date": self.out_info.date,
                "output_dir": self.out_info.absolute_path,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "system_info": self._get_system_info(),
                "status": status,
                "is_completed": False
            }
            # Merge in config data
            manifest.update(self.config_data)
            
            with open(self._manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4, ensure_ascii=False)
        else:
            self._update_manifest_status(status, 0)

    def _update_manifest_status(self, status: str, duration: float) -> None:
        if not self._manifest_file or not self._manifest_file.exists():
            return
            
        try:
            with open(self._manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                
            manifest["status"] = status
            manifest["is_completed"] = (status == "completed")
            if duration > 0:
                manifest["duration"] = round(duration, 2)
                manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
                
            with open(self._manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    @staticmethod
    def open_folder(path_str: str) -> None:
        import sys, subprocess
        if sys.platform == "win32":
            os.startfile(path_str)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path_str])
        else:
            subprocess.Popen(["xdg-open", path_str])
