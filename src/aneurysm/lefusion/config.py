"""Training settings for LeFusion. Values live in configs/lefusion_train.yaml."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import check_keys, find_project_root, read_yaml


@dataclass
class TrainConfig:
    # ---- run ----
    run_name: str = "pilot"
    output_dir: str = "outputs"           # everything of a run goes to <output_dir>/<run_name>/
    resume: bool = True
    seed: int = 42
    deterministic: bool = True            # deterministic cuDNN (slightly slower, more reproducible)

    # ---- network ----
    base_channels: int = 64
    dim_mults: list = field(default_factory=lambda: [1, 2, 4, 8])
    attn_heads: int = 8
    attn_dim_head: int = 32

    # ---- diffusion ----
    timesteps: int = 300
    loss_type: str = "l1"

    # ---- optimisation ----
    lr: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 4
    max_steps: int = 10000
    grad_clip: float | None = 1.0
    amp: bool = True
    num_workers: int | None = None

    # ---- EMA ----
    ema_decay: float = 0.995
    ema_start: int = 2000
    ema_every: int = 10

    # ---- logging / eval / checkpoints ----
    log_every: int = 50
    eval_every: int = 500
    eval_max_batches: int = 50
    eval_timesteps: list = field(default_factory=lambda: [30, 150, 270])
    sample_every: int = 2000
    n_samples: int = 2
    sample_jump_length: int = 5
    sample_jump_n_sample: int = 2
    save_every: int = 1000
    keep_last: int = 3
    memory_check: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        """Catch settings that would fail later in the run."""
        if self.loss_type not in ("l1", "l2"):
            raise ValueError("loss_type must be 'l1' or 'l2'")
        bad = [t for t in self.eval_timesteps if not 0 <= t < self.timesteps]
        if bad:
            raise ValueError(f"eval_timesteps {bad} must be in [0, {self.timesteps - 1}]")
        for name in ("batch_size", "grad_accum", "max_steps", "log_every", "eval_every", "save_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


def load_train_config(path: str | Path | None = None, overrides: dict | None = None,
                      project_root: str | Path | None = None) -> TrainConfig:
    """Read configs/lefusion_train.yaml, apply `overrides`, check for typos and bad values."""
    root = Path(project_root).resolve() if project_root else find_project_root()
    path = Path(path) if path else root / "configs" / "lefusion_train.yaml"
    values = read_yaml(path)
    values.update(overrides or {})
    check_keys(values, TrainConfig, path.name)
    cfg = TrainConfig(**values)
    cfg.validate()
    return cfg


# ======================================================================
# Evaluation settings (configs/lefusion_eval.yaml)
# ======================================================================
@dataclass
class EvalConfig:
    # ---- which model ----
    run_name: str = "pilot"
    output_dir: str = "outputs"
    checkpoint: str = "best"              # best | latest | step_001000 ...
    use_ema: bool = True                  # EMA weights (smoother samples)

    # ---- which data ----
    split: str = "val"                    # val | test
    max_patches: int = 8                  # patches to evaluate (0 = all)

    # ---- generation ----
    seed: int = 123
    jump_length: int = 5                  # RePaint settings (LeFusion script: 5 / 5)
    jump_n_sample: int = 2
    amp: bool = True
    vessel_hu: float = 200.0              # "bright" threshold for the mask-fill check

    # ---- what to run ----
    plot_curves: bool = True
    rebuild: bool = True
    new_shapes: bool = True
    grow_voxels: int = 2
    histogram_control: bool = True
    hist_patches: int = 2
    copy_check: bool = True
    full_scan: bool = True
    full_scan_patch: str | None = None    # patch_id, or null = first evaluated patch

    def to_dict(self) -> dict:
        return asdict(self)


def load_eval_config(path: str | Path | None = None, overrides: dict | None = None,
                     project_root: str | Path | None = None) -> EvalConfig:
    """Read configs/lefusion_eval.yaml, apply `overrides`, check for typos."""
    root = Path(project_root).resolve() if project_root else find_project_root()
    path = Path(path) if path else root / "configs" / "lefusion_eval.yaml"
    values = read_yaml(path)
    values.update(overrides or {})
    check_keys(values, EvalConfig, path.name)
    cfg = EvalConfig(**values)
    if cfg.split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    return cfg


# ======================================================================
# Background / mask test settings (configs/lefusion_eval_b.yaml, 03b notebook)
# ======================================================================
BACKGROUND_TEST_IDS = (1, 2, 3, 4)


@dataclass
class BackgroundEvalConfig:
    # ---- which model ----
    run_name: str = "pilot"
    output_dir: str = "outputs"
    checkpoint: str = "best"              # best | latest | step_001000 ...
    use_ema: bool = True

    # ---- which data ----
    split: str = "val"                    # val | test
    max_patches: int = 8                  # patches to evaluate (0 = all)

    # ---- generation ----
    seed: int = 123                       # same seed for every test -> fair comparison
    jump_length: int = 5
    jump_n_sample: int = 2
    amp: bool = True
    vessel_hu: float = 200.0              # "bright" threshold (vessel / aneurysm)

    # ---- dilated mask + pseudo-healthy patch ----
    dilate_voxels: int = 2                # rim removed and redrawn around the aneurysm
    tissue_ring_voxels: int = 4           # ring around the rim used for the fill value
    noise_blur_sigma: float = 0.7         # voxels; makes the fill noise grainy like CT

    # ---- which tests (order is kept) ----
    # 1 noisy real + actual mask        2 noisy real + dilated mask
    # 3 noisy pseudo-healthy + actual   4 noisy pseudo-healthy + dilated mask
    tests: list = field(default_factory=lambda: [1, 2, 3, 4])

    def to_dict(self) -> dict:
        return asdict(self)


def load_background_eval_config(path: str | Path | None = None, overrides: dict | None = None,
                                project_root: str | Path | None = None) -> BackgroundEvalConfig:
    """Read configs/lefusion_eval_b.yaml, apply `overrides`, check for typos."""
    root = Path(project_root).resolve() if project_root else find_project_root()
    path = Path(path) if path else root / "configs" / "lefusion_eval_b.yaml"
    values = read_yaml(path)
    values.update(overrides or {})
    check_keys(values, BackgroundEvalConfig, path.name)
    cfg = BackgroundEvalConfig(**values)
    if cfg.split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    bad = [t for t in cfg.tests if t not in BACKGROUND_TEST_IDS]
    if bad or not cfg.tests:
        raise ValueError(f"tests must be a non-empty list from {list(BACKGROUND_TEST_IDS)}, got {cfg.tests}")
    if cfg.dilate_voxels < 1:
        raise ValueError("dilate_voxels must be >= 1")
    return cfg
