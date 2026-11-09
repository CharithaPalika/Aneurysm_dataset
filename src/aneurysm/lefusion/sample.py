"""Generation with a trained LeFusion model (used by training previews and evaluation).

Main pieces
-----------
load_trained_model(...)   checkpoint -> model + the configs it was trained with
generate(...)             LeFusion inpainting (RePaint loop) inside a mask, real background kept
paste_patch_into_volume() put a generated patch back into the full CT scan
plot_rows(...)            real | (input) | generated | difference figure
"""
from __future__ import annotations

import random
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from ..config import DataConfig, find_project_root
from .config import TrainConfig
from .model import build_diffusion


# ----------------------------------------------------------------------
# intensity helpers
# ----------------------------------------------------------------------
def norm_to_hu(x, cfg: DataConfig):
    """Model values in [-1, 1] -> Hounsfield units."""
    return (x + 1.0) / 2.0 * (cfg.hu_max - cfg.hu_min) + cfg.hu_min


def hu_to_norm(hu, cfg: DataConfig):
    """Hounsfield units -> model values in [-1, 1]."""
    return 2.0 * (hu - cfg.hu_min) / (cfg.hu_max - cfg.hu_min) - 1.0


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
    path = Path(run_dir) / "checkpoints" / (checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt")
    if not path.exists():
        available = sorted(p.name for p in path.parent.glob("*.pt"))
        raise FileNotFoundError(f"{path} not found. Available: {available}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device)

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


# ----------------------------------------------------------------------
# back into the full scan
# ----------------------------------------------------------------------
def model_patch_affine(stored_affine: np.ndarray, margin: int) -> np.ndarray:
    """Affine of the centre-cropped model patch (x, y, z voxel -> world mm)."""
    shift = np.eye(4)
    shift[:3, 3] = margin
    return stored_affine @ shift


def paste_patch_into_volume(volume_hu: np.ndarray, volume_affine: np.ndarray,
                            patch_hu: np.ndarray, patch_mask: np.ndarray, patch_affine: np.ndarray,
                            blend_sigma_vox: float = 1.0) -> np.ndarray:
    """Paste a generated patch (x, y, z order, HU) back into the full scan (RAS, native grid).

    Only voxels near the mask change: weight = blurred, slightly grown mask, so the
    edge is soft. Returns a new int16 volume.
    """
    weight = ndimage.gaussian_filter(ndimage.binary_dilation(patch_mask > 0, iterations=2).astype(np.float32),
                                     blend_sigma_vox)
    # native voxels covered by the patch: map the patch corners into the volume
    corners = np.array([[i, j, k, 1] for i in (0, patch_hu.shape[0] - 1)
                        for j in (0, patch_hu.shape[1] - 1) for k in (0, patch_hu.shape[2] - 1)]).T
    vox = np.linalg.inv(volume_affine) @ patch_affine @ corners
    lo = np.maximum(np.floor(vox[:3].min(1)).astype(int), 0)
    hi = np.minimum(np.ceil(vox[:3].max(1)).astype(int) + 1, np.array(volume_hu.shape))

    grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"))
    pts = np.concatenate([grid.reshape(3, -1), np.ones((1, grid[0].size))])
    p = (np.linalg.inv(patch_affine) @ volume_affine @ pts)[:3]          # patch coordinates

    w = ndimage.map_coordinates(weight, p, order=1, mode="constant", cval=0).reshape(grid.shape[1:])
    g = ndimage.map_coordinates(patch_hu.astype(np.float32), p, order=1, mode="nearest").reshape(grid.shape[1:])

    out = volume_hu.copy()
    region = tuple(slice(a, b) for a, b in zip(lo, hi))
    out[region] = np.round(w * g + (1 - w) * out[region].astype(np.float32)).astype(np.int16)
    return out


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def _contour(ax, m2d: np.ndarray, color: str) -> None:
    if m2d.any():
        ax.contour(m2d, levels=[0.5], colors=color, linewidths=0.8, origin="lower")


def plot_rows(rows: list[dict], out_path: Path | None = None, show: bool = False, title: str = "",
              names: dict | None = None):
    """One row per sample: real | (input) | generated | |difference|, middle slice through the mask.

    rows: dicts with
        'real', 'gen', 'mask'  (D, H, W) numpy, [-1, 1] / 0-1
        'label'                str
        'input'      optional  background given to the model -> extra panel
        'outer_mask' optional  larger mask: then 'mask' is drawn yellow and 'outer_mask' red
                               (without it, 'mask' is drawn red)
    names: rename panels, e.g. {"generated": "pseudo-healthy"}
    """
    import matplotlib.pyplot as plt

    names = {"real": "real", "input": "input", "generated": "generated", "difference": "|difference|",
             **(names or {})}
    with_input = any("input" in r for r in rows)
    n, cols = len(rows), (4 if with_input else 3)
    fig, axs = plt.subplots(n, cols, figsize=(3 * cols, 3 * n), squeeze=False)
    for k, r in enumerate(rows):
        m, outer = r["mask"], r.get("outer_mask")
        ref = outer if outer is not None else m
        z = int(np.argmax(ref.sum(axis=(1, 2)))) if ref.any() else ref.shape[0] // 2
        panels = [(r["real"], names["real"], "gray", -1)]
        if with_input:
            panels.append((r.get("input", r["real"]), names["input"], "gray", -1))
        panels += [(r["gen"], names["generated"], "gray", -1),
                   (np.abs(r["gen"] - r["real"]), names["difference"], "magma", 0)]
        for j, (img, name, cmap, vmin) in enumerate(panels):
            ax = axs[k, j]
            ax.imshow(img[z], cmap=cmap, vmin=vmin, vmax=1, origin="lower")
            if outer is None:
                _contour(ax, m[z], "r")
            else:
                _contour(ax, outer[z], "r")
                _contour(ax, m[z], "y")
            ax.set_title(f"{r['label']}  {name}", fontsize=8)
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
