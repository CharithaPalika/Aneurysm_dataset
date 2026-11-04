"""LeFusion (Zhang et al., ICLR 2025) wrapped for our aneurysm patches.

The network and diffusion code are imported unchanged from the cloned
repository in LeFusion/LeFusion/ddpm. Only data handling, the training
loop and generation post-processing are ours.
"""
from .config import TrainConfig, load_train_config
from .model import build_diffusion
from .train import train

__all__ = ["TrainConfig", "load_train_config", "build_diffusion", "train"]
