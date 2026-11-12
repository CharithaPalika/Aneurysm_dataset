"""Training settings (configs/train/lefusion_train.yaml): run, network, diffusion, optimiser, logging."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from .base import CONFIG_DIR, check_keys, find_project_root, read_yaml, resolve_path

DEFAULT_TRAIN_CONFIG = Path(CONFIG_DIR) / "train" / "lefusion_train.yaml"


@dataclass
class TrainConfig:
    # ---- run ----
    run_name: str = "pilot"
    output_dir: str = "outputs"           # everything of a run goes to <output_dir>/<run_name>/
    resume: bool = True
    seed: int = 42
    deterministic: bool = True            # deterministic cuDNN (slightly slower, more reproducible)

    # ---- data source ----
    preprocess: bool = False              # True: cut missing patches from Dataset/ first; False: use work dir only
    work_dir: str | None = None           # preprocessed data (splits/ + patches/); None = work_subdir of data config

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
    """Read configs/train/lefusion_train.yaml (or `path`), apply `overrides`, check for typos and bad values."""
    root = Path(project_root).resolve() if project_root else find_project_root()
    path = resolve_path(path, root, DEFAULT_TRAIN_CONFIG)
    values = read_yaml(path)
    values.update(overrides or {})
    check_keys(values, TrainConfig, path.name)
    cfg = TrainConfig(**values)
    cfg.validate()
    return cfg
