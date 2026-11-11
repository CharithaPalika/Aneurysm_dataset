"""Get the data ready: case list -> train/val/test split -> cached patches (same as the 01 notebook steps).

Examples (from the project folder):
    python scripts/run_prepare_data.py
    python scripts/run_prepare_data.py --data-config configs/data/data.yaml --set patch_depth=64
    python scripts/run_prepare_data.py --overwrite          # re-cut the patches of this setting
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

from aneurysm.config import DEFAULT_DATA_CONFIG, load_data_config, resolve_path  # noqa: E402
from aneurysm.data import prepare_data  # noqa: E402
from aneurysm.utils.cli import add_config_arg, add_set_arg, parse_overrides, print_header  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the split and the patch cache in work/.")
    add_config_arg(parser, "--data-config", DEFAULT_DATA_CONFIG, "data")
    add_set_arg(parser, "--set", "data")
    parser.add_argument("--threads", type=int, default=2, help="scans loaded in parallel (default: 2)")
    parser.add_argument("--overwrite", action="store_true", help="re-cut patches even if they exist")
    args = parser.parse_args()

    data_path = resolve_path(args.data_config, PROJECT_ROOT, DEFAULT_DATA_CONFIG)
    overrides = parse_overrides(args.set)
    print_header("prepare data", {"data config": data_path}, {"overrides": overrides})

    cfg = load_data_config(data_path, project_root=PROJECT_ROOT, overrides=overrides)
    prepare_data(cfg, n_threads=args.threads, verbose=True, overwrite=args.overwrite)
    print(f"\nPatches in {cfg.rel(cfg.patch_dir)}/")


if __name__ == "__main__":
    main()
