"""Model, training loop and generation.

The network and diffusion code are LeFusion's (Zhang et al., ICLR 2025, Apache-2.0),
imported unchanged from the cloned repository in LeFusion/LeFusion/ddpm.
Only data handling, the training loop and generation post-processing are ours.
"""
from .model import build_diffusion, count_parameters
from .sampling import fill_ratio, generate, grow_mask, load_trained_model
from .trainer import train

__all__ = ["build_diffusion", "count_parameters", "train", "load_trained_model", "generate",
           "grow_mask", "fill_ratio"]
