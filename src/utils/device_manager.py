from dataclasses import dataclass
import os

@dataclass
class DeviceInfo:
    recommended_workers: int
    minimum_workers: int
    maximum_workers: int
    reason: str
    cpu_name: str
    gpu_name: str
    total_ram: int
    total_vram: int

class DeviceManager:
    @staticmethod
    def detect() -> dict:
        import platform
        try:
            import psutil
            total_ram = psutil.virtual_memory().total
        except ImportError:
            total_ram = 0
            
        cpu_name = platform.processor()
        gpu_name = "N/A"
        total_vram = 0
        has_gpu = False
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory
                has_gpu = True
        except ImportError:
            pass
            
        return {
            "has_gpu": has_gpu,
            "cpu_name": cpu_name,
            "gpu_name": gpu_name,
            "total_ram": total_ram,
            "total_vram": total_vram
        }

    @staticmethod
    def recommend_workers(device: str) -> DeviceInfo:
        info = DeviceManager.detect()
        
        if device == "gpu" and info["has_gpu"]:
            return DeviceInfo(
                recommended_workers=1,
                minimum_workers=1,
                maximum_workers=2,
                reason="CUDA Context memory footprint restricts multiple PyTorch processes.",
                cpu_name=info["cpu_name"],
                gpu_name=info["gpu_name"],
                total_ram=info["total_ram"],
                total_vram=info["total_vram"]
            )
        else:
            cores = os.cpu_count() or 2
            safe_max = max(1, cores - 1)
            # Recommend based on RAM if psutil is available
            if info["total_ram"] > 0:
                ram_gb = info["total_ram"] / (1024**3)
                if ram_gb < 8:
                    safe_max = min(safe_max, 4)
            return DeviceInfo(
                recommended_workers=safe_max,
                minimum_workers=1,
                maximum_workers=cores,
                reason="CPU threads constrained by total physical cores minus 1 to prevent system lockup.",
                cpu_name=info["cpu_name"],
                gpu_name=info["gpu_name"],
                total_ram=info["total_ram"],
                total_vram=info["total_vram"]
            )
