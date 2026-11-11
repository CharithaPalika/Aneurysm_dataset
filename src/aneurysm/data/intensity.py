"""Intensity helpers: HU window <-> model values in [-1, 1], and the LeFusion histogram."""
from __future__ import annotations

import numpy as np
import torch

from ..config import DataConfig


def normalize_hu(x: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip HU to [hu_min, hu_max] and scale to [-1, 1]."""
    x = np.clip(x.astype(np.float32), hu_min, hu_max)
    return 2.0 * (x - hu_min) / (hu_max - hu_min) - 1.0


def norm_to_hu(x, cfg: DataConfig):
    """Model values in [-1, 1] -> Hounsfield units."""
    return (x + 1.0) / 2.0 * (cfg.hu_max - cfg.hu_min) + cfg.hu_min


def hu_to_norm(hu, cfg: DataConfig):
    """Hounsfield units -> model values in [-1, 1]."""
    return 2.0 * (hu - cfg.hu_min) / (cfg.hu_max - cfg.hu_min) - 1.0


def lesion_histogram(image: torch.Tensor, mask: torch.Tensor, bins: int) -> torch.Tensor:
    """Same as LeFusion: histogram over [-1, 1] of the voxels inside `mask`, sums to 1."""
    inside = image[mask > 0]
    if inside.numel() == 0:
        return torch.zeros(bins, dtype=torch.float32)
    return torch.histc(inside, bins=bins, min=-1.0, max=1.0) / inside.numel()
