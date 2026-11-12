"""Train the model (same as 02_train_lefusion.ipynb).

Examples (from the project folder):
    python scripts/run_train.py
    python scripts/run_train.py --config configs/train/lefusion_train.yaml --data-config configs/data/data.yaml
    python scripts/run_train.py --set run_name=pilot_small base_channels=32 "dim_mults=[1,2,4,4]" attn_heads=4
    python scripts/run_train.py --data-set patch_depth=24

Multi-GPU (one process per GPU; batch_size is per GPU):
    torchrun --standalone --nproc_per_node=4 scripts/run_train.py --config configs/train/lefusion_train_a100.yaml

Outputs go to outputs/<run_name>/. Ctrl+C saves checkpoints/latest.pt; the same command resumes.
Before a long run: python scripts/run_smoke_test.py (same options).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---- project folder = parent of scripts/ (works from any current folder) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "src" / "aneurysm").is_dir():
    raise SystemExit(f"Expected the project folder at {PROJECT_ROOT} (with src/aneurysm/)")
if str(PROJECT_ROOT / "src") not in sys.path:        # works with or without `pip install -e .`
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")                                # figures are saved to files, no windows

from aneurysm.config import (DEFAULT_DATA_CONFIG, DEFAULT_TRAIN_CONFIG, load_data_config,  # noqa: E402
                             load_train_config, resolve_path)
from aneurysm.training import train  # noqa: E402
from aneurysm.training.distributed import cleanup  # noqa: E402
from aneurysm.utils.cli import add_config_arg, add_set_arg, is_main_process, parse_overrides, print_header  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LeFusion on the aneurysm patches.")
    add_config_arg(parser, "--config", DEFAULT_TRAIN_CONFIG, "training")
    add_config_arg(parser, "--data-config", DEFAULT_DATA_CONFIG, "data")
    add_set_arg(parser, "--set", "training")
    add_set_arg(parser, "--data-set", "data")
    args = parser.parse_args()

    train_path = resolve_path(args.config, PROJECT_ROOT, DEFAULT_TRAIN_CONFIG)
    data_path = resolve_path(args.data_config, PROJECT_ROOT, DEFAULT_DATA_CONFIG)
    train_over, data_over = parse_overrides(args.set), parse_overrides(args.data_set, "--data-set")
    print_header("train", {"train config": train_path, "data config": data_path},
                 {"train set": train_over, "data set": data_over})

    data_cfg = load_data_config(data_path, project_root=PROJECT_ROOT, overrides=data_over)
    train_cfg = load_train_config(train_path, project_root=PROJECT_ROOT, overrides=train_over)
    try:
        result = train(data_cfg, train_cfg)
    finally:
        cleanup()
    if is_main_process():
        print(f"\nRun folder: {data_cfg.rel(result['run_dir'])}/")


if __name__ == "__main__":
    main()
