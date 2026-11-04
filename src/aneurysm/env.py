"""Where are we running? macOS (preparation) or Windows with a GPU (training)."""
from __future__ import annotations

import os
import platform
import sys


def os_name() -> str:
    """'windows', 'macos' or 'linux'."""
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin":
        return "macos"
    return "linux"


def get_device():
    """Best available torch device: CUDA GPU > Apple MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_num_workers() -> int:
    """Safe DataLoader worker count for notebooks.

    - macOS: 0 (worker processes in notebooks are slow to start and often unnecessary here)
    - Windows: 2 (workers use 'spawn'; our Dataset lives in a module, so this works)
    - Linux: up to 4
    """
    cpus = os.cpu_count() or 1
    name = os_name()
    if name == "macos":
        return 0
    if name == "windows":
        return min(2, cpus)
    return min(4, cpus)


def describe_environment() -> dict:
    """Short summary to print at the top of every notebook."""
    info = {
        "os": os_name(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpus": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
            info["cuda_version"] = torch.version.cuda
        info["device"] = str(get_device())
    except ImportError:
        info["torch"] = "not installed"
    return info
