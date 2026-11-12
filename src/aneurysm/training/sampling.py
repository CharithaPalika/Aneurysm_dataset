"""Generation with a trained model (used by training previews and evaluation).

load_trained_model(...)   checkpoint -> model + the configs it was trained with
generate(...)             LeFusion inpainting (RePaint loop) inside a mask, real background kept
grow_mask(...)            round dilation of a (B, 1, D, H, W) mask
fill_ratio(...)           fraction of mask voxels that are vessel-bright
"""
from __future__ import annotations

import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from ..config import DataConfig, RunPaths, TrainConfig, find_project_root
from .model import build_diffusion


# ----------------------------------------------------------------------
# loading a trained model
# ----------------------------------------------------------------------
def load_trained_model(run_dir: Path, checkpoint: str = "best", device=None, use_ema: bool = True,
                       project_root: Path | None = None):
    """Load outputs/<run>/checkpoints/<checkpoint>.pt.

    The model is rebuilt with the settings saved *inside* the checkpoint, so it
    always matches the weights, even if the YAML files were changed afterwards.

    Returns (diffusion, data_cfg, train_cfg, info) with info = {'step', 'best_val', 'path', 'weights'}.
    """
    project_root = Path(project_root or find_project_root())
    path = RunPaths(run_dir).checkpoint(checkpoint)
    if not path.exists():
        available = sorted(p.name for p in path.parent.glob("*.pt"))
        raise FileNotFoundError(f"{path} not found. Available: {available}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)   # our own file (holds RNG states)

    data_names = {f.name for f in fields(DataConfig)} - {"project_root"}
    data_cfg = DataConfig(project_root=project_root,
                          **{k: v for k, v in ck["data_cfg"].items() if k in data_names})
    train_names = {f.name for f in fields(TrainConfig)}
    train_cfg = TrainConfig(**{k: v for k, v in ck["train_cfg"].items() if k in train_names})

    diffusion = build_diffusion(data_cfg, train_cfg, device)
    weights = "ema" if use_ema else "model"
    diffusion.denoise_fn.load_state_dict(ck[weights])
    diffusion.eval()
    info = {"step": int(ck["step"]), "best_val": float(ck.get("best_val", float("nan"))),
            "path": path, "weights": weights}
    return diffusion, data_cfg, train_cfg, info


# ----------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------
@torch.no_grad()
def generate(diffusion, image: torch.Tensor, mask: torch.Tensor, hist: torch.Tensor,
             timesteps: int, jump_length: int = 5, jump_n_sample: int = 2,
             amp: bool = True, seed: int | None = 0) -> torch.Tensor:
    """Draw a new aneurysm inside `mask`, keep everything outside it.

    image : (B, 1, D, H, W) in [-1, 1]   real patch (background to keep)
    mask  : (B, 1, D, H, W) 0/1          where to draw
    hist  : (B, hist_bins)               brightness condition
    Returns (B, 1, D, H, W): generated inside the mask, exactly the real image outside.
    The global random state is restored afterwards (training is not disturbed).
    """
    from omegaconf import OmegaConf

    device = image.device
    conf = OmegaConf.create({
        "schedule_jump_params": {"t_T": timesteps, "n_sample": 1,
                                 "jump_length": jump_length, "jump_n_sample": jump_n_sample},
        "inpa_inj_sched_prev_cumnoise": False,
        "inpa_inj_time_shift": 1,
    })

    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    diffusion.eval()
    use_amp = bool(amp and device.type == "cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        out = diffusion.p_sample_loop_repaint(
            shape=tuple(image.shape), model_kwargs={"gt": image, "gt_keep_mask": mask},
            device=device, progress=False, conf=conf, cond=hist,
        )
    random.setstate(saved[0]); np.random.set_state(saved[1]); torch.set_rng_state(saved[2])
    if saved[3] is not None:
        torch.cuda.set_rng_state_all(saved[3])

    out = out.float().clamp(-1, 1)
    return out * mask + image * (1 - mask)          # real background outside the mask


def grow_mask(mask: torch.Tensor, voxels: int) -> torch.Tensor:
    """Grow a (B, 1, D, H, W) mask by `voxels` with a round (ball) kernel, so corners stay round."""
    if voxels <= 0:
        return mask
    r = int(voxels)
    ax = torch.arange(-r, r + 1, device=mask.device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(ax, ax, ax, indexing="ij")
    kernel = ((zz ** 2 + yy ** 2 + xx ** 2) <= r * r).float()[None, None]
    hits = torch.nn.functional.conv3d((mask > 0).float(), kernel, padding=r)
    return (hits > 0.5).to(mask.dtype)


def fill_ratio(gen: torch.Tensor, mask: torch.Tensor, threshold_norm: float) -> float:
    """Fraction of mask voxels that are vessel-bright in the generated patch."""
    inside = gen[mask > 0]
    return float((inside > threshold_norm).float().mean()) if inside.numel() else float("nan")
