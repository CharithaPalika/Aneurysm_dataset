"""Evaluation settings (configs/eval/lefusion_eval.yaml): which run, which checks, generation settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .base import CONFIG_DIR, check_keys, find_project_root, read_yaml, resolve_path

DEFAULT_EVAL_CONFIG = Path(CONFIG_DIR) / "eval" / "lefusion_eval.yaml"


@dataclass
class EvalConfig:
    # ---- which model ----
    run_name: str = "pilot"               # folder <output_dir>/<run_name>/ written by training
    run_dir: str | None = None            # or give the run folder directly (wins over run_name)
    output_dir: str = "outputs"
    checkpoint: str = "best"              # best | latest | step_001000 ...
    use_ema: bool = True                  # EMA weights (smoother samples)

    # ---- data source ----
    preprocess: bool = False              # True: cut missing patches from Dataset/ first; False: use work dir only
    work_dir: str | None = None           # preprocessed data (splits/ + patches/); None = the one used in training

    # ---- which data ----
    split: str = "val"                    # val | test
    max_patches: int = 8                  # patches to evaluate (0 = all)

    # ---- generation ----
    seed: int = 123                       # same seed in every case -> fair comparison
    jump_length: int = 5                  # RePaint back-and-forth settings
    jump_n_sample: int = 2
    amp: bool = True
    vessel_hu: float = 200.0              # "bright" threshold (vessel / filled aneurysm)

    # ---- dilated mask + pseudo-healthy patch (cases 2-4) ----
    dilate_voxels: int = 2                # rim around the aneurysm
    tissue_ring_voxels: int = 4           # ring around the rim used for the fill value
    noise_blur_sigma: float = 0.7         # grainy fill noise like CT (0 = white noise)

    # ---- checks (true / false) ----
    training_curves: bool = True
    case_1_real_actual: bool = True       # noisy real + actual mask
    case_2_real_dilated: bool = True      # noisy real + dilated mask
    case_3_pseudo_actual: bool = True     # noisy pseudo-healthy + actual mask
    case_4_pseudo_dilated: bool = True    # noisy pseudo-healthy + dilated mask
    histogram_control: bool = True        # low / medium / high contrast histograms
    hist_patches: int = 2
    copy_check: bool = True               # needs case 1
    full_scan: bool = True                # needs case 1
    full_scan_patch: str | None = None    # patch_id, or None = first evaluated patch

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        if self.dilate_voxels < 1:
            raise ValueError("dilate_voxels must be >= 1")
        needs_case_1 = [n for n in ("copy_check", "full_scan") if getattr(self, n)]
        if needs_case_1 and not self.case_1_real_actual:
            raise ValueError(f"{needs_case_1} use the case 1 results: set case_1_real_actual: true "
                             f"or switch them off")


def load_eval_config(path: str | Path | None = None, overrides: dict | None = None,
                     project_root: str | Path | None = None) -> EvalConfig:
    """Read configs/eval/lefusion_eval.yaml (or `path`), apply `overrides`, check for typos."""
    root = Path(project_root).resolve() if project_root else find_project_root()
    path = resolve_path(path, root, DEFAULT_EVAL_CONFIG)
    values = read_yaml(path)
    values.update(overrides or {})
    check_keys(values, EvalConfig, path.name)
    cfg = EvalConfig(**values)
    cfg.validate()
    return cfg
