"""Build the LeFusion network and diffusion process from our configs.

The classes come unchanged from LeFusion/LeFusion/ddpm/diffusion.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..config import DataConfig
from .config import TrainConfig

LEFUSION_SUBDIR = Path("LeFusion") / "LeFusion"     # folder that contains ddpm/


def add_lefusion_to_path(project_root: Path) -> Path:
    """Make `import ddpm` find the cloned LeFusion code."""
    code_dir = (Path(project_root) / LEFUSION_SUBDIR).resolve()
    if not (code_dir / "ddpm" / "diffusion.py").exists():
        raise FileNotFoundError(f"LeFusion code not found at {code_dir} (expected ddpm/diffusion.py)")
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    return code_dir


def check_patch_shape(data_cfg: DataConfig) -> tuple[int, int]:
    """LeFusion needs square slices (H = W) and H divisible by 8. Depth can be anything."""
    x, y, z = data_cfg.model_patch_shape
    if x != y:
        raise ValueError(f"In-plane patch must be square, got {x}x{y}")
    if x % 8 != 0:
        raise ValueError(f"patch_size must be divisible by 8 (U-Net halves it 3 times), got {x}")
    return x, z            # (image_size, depth)


def build_diffusion(data_cfg: DataConfig, train_cfg: TrainConfig, device):
    """Returns LeFusion's GaussianDiffusion_Nolatent wrapping a Unet3D, on `device`."""
    add_lefusion_to_path(data_cfg.project_root)
    from ddpm.diffusion import GaussianDiffusion_Nolatent, Unet3D   # noqa: E402  (LeFusion code)

    image_size, depth = check_patch_shape(data_cfg)
    unet = Unet3D(
        dim=train_cfg.base_channels,              # LeFusion passes the image size here; we decouple it
        dim_mults=tuple(train_cfg.dim_mults),
        channels=1,
        cond_dim=data_cfg.hist_bins,
        attn_heads=train_cfg.attn_heads,
        attn_dim_head=train_cfg.attn_dim_head,
    )
    diffusion = GaussianDiffusion_Nolatent(
        unet,
        image_size=image_size,
        num_frames=depth,
        channels=1,
        timesteps=train_cfg.timesteps,
        loss_type=train_cfg.loss_type,
        device=device,
        data_type="lidc",                         # = single lesion class in LeFusion's code
    )
    return diffusion.to(device)


def count_parameters(module) -> float:
    """Millions of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6
