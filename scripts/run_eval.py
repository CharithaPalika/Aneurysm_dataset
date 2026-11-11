"""Evaluate a trained run (same as 03_eval_lefusion.ipynb). Checks are switched on / off in the config.

Examples (from the project folder):
    python scripts/run_eval.py
    python scripts/run_eval.py --config configs/eval/lefusion_eval.yaml --set run_name=pilot_small
    python scripts/run_eval.py --set run_dir=outputs/pilot_small checkpoint=latest max_patches=4
    python scripts/run_eval.py --set case_2_real_dilated=false histogram_control=false

Results go to <run folder>/eval/<checkpoint>_<split>/ (figures saved, not shown).
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

from aneurysm.config import DEFAULT_EVAL_CONFIG, load_eval_config, resolve_path  # noqa: E402
from aneurysm.evaluation import run_evaluation  # noqa: E402
from aneurysm.utils.cli import add_config_arg, add_set_arg, parse_overrides, print_header  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained LeFusion run.")
    add_config_arg(parser, "--config", DEFAULT_EVAL_CONFIG, "evaluation")
    add_set_arg(parser, "--set", "evaluation")
    args = parser.parse_args()

    eval_path = resolve_path(args.config, PROJECT_ROOT, DEFAULT_EVAL_CONFIG)
    overrides = parse_overrides(args.set)
    print_header("evaluate", {"eval config": eval_path}, {"overrides": overrides})

    cfg = load_eval_config(eval_path, project_root=PROJECT_ROOT, overrides=overrides)
    run_evaluation(cfg, project_root=PROJECT_ROOT, show=False)


if __name__ == "__main__":
    main()
