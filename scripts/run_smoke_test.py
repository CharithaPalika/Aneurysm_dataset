"""Smoke test: a few minutes of the real pipeline before a long run.

Uses the SAME train / data config as the real run (network size, batch, GPUs, data), but only a few
steps and its own run folder (outputs/smoke_tests/<name>/, never your real run). It checks:
  1. the preprocessed data loads (train + val)
  2. the model is built and fits in GPU memory (memory check)
  3. training steps run and the loss is finite (also on all GPUs with torchrun)
  4. validation, a preview sample and checkpoints are written
  5. resuming from latest.pt works
  6. evaluation runs on the smoke checkpoint (case 1 + case 4, one patch)
and prints PASS / FAIL for each. Exit code 0 = all passed.

Examples (from the project folder):
    python scripts/run_smoke_test.py                                        # pilot config, 1 GPU
    python scripts/run_smoke_test.py --tiny                                 # tiny network: laptop / CPU, pipeline only
    torchrun --standalone --nproc_per_node=4 scripts/run_smoke_test.py --config configs/train/lefusion_train_a100.yaml
    python scripts/run_smoke_test.py --config configs/train/lefusion_train_a100.yaml --steps 40 --set batch_size=4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---- project folder = parent of scripts/ (works from any current folder) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "src" / "aneurysm").is_dir():
    raise SystemExit(f"Expected the project folder at {PROJECT_ROOT} (with src/aneurysm/)")
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import math  # noqa: E402

import pandas as pd  # noqa: E402

from aneurysm.config import (DEFAULT_DATA_CONFIG, DEFAULT_TRAIN_CONFIG, RunPaths, load_data_config,  # noqa: E402
                             load_eval_config, load_train_config, resolve_path)
from aneurysm.training import train  # noqa: E402
from aneurysm.training.distributed import broadcast_object, cleanup, setup  # noqa: E402
from aneurysm.utils.cli import (add_config_arg, add_set_arg, is_main_process, parse_overrides,  # noqa: E402
                                print_header)
from aneurysm.utils.env import get_device  # noqa: E402

SMOKE_DIR = "outputs/smoke_tests"
TINY = {"base_channels": 8, "dim_mults": [1, 2], "attn_heads": 2, "attn_dim_head": 8,
        "timesteps": 20, "eval_timesteps": [2, 10, 18], "batch_size": 1, "grad_accum": 1}


def smoke_overrides(steps: int, name: str, tiny: bool, user: dict) -> tuple[dict, dict]:
    """Settings for the two training phases (phase 2 resumes phase 1)."""
    half = max(2, steps // 2)
    base = {
        "run_name": name, "output_dir": SMOKE_DIR, "resume": True,
        "log_every": max(1, half // 2), "eval_every": half, "eval_max_batches": 2,
        "sample_every": steps, "n_samples": 1, "save_every": half, "keep_last": 2,
        **(TINY if tiny else {}),
        **user,                                     # --set wins (e.g. batch_size), but not the smoke folder/steps
        "run_name": name, "output_dir": SMOKE_DIR,
    }
    return {**base, "max_steps": half}, {**base, "max_steps": max(steps, half + 1)}


def check(results: list, name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Short end-to-end test of training + evaluation.")
    add_config_arg(parser, "--config", DEFAULT_TRAIN_CONFIG, "training")
    add_config_arg(parser, "--data-config", DEFAULT_DATA_CONFIG, "data")
    add_set_arg(parser, "--set", "training")
    add_set_arg(parser, "--data-set", "data")
    parser.add_argument("--steps", type=int, default=20, help="training steps in total (default 20)")
    parser.add_argument("--tiny", action="store_true", help="tiny network + 20 diffusion steps (CPU / laptop)")
    parser.add_argument("--skip-eval", action="store_true", help="only test training")
    parser.add_argument("--name", default=None, help="run folder name (default smoke_<date>_<time>)")
    args = parser.parse_args()

    train_path = resolve_path(args.config, PROJECT_ROOT, DEFAULT_TRAIN_CONFIG)
    data_path = resolve_path(args.data_config, PROJECT_ROOT, DEFAULT_DATA_CONFIG)
    user, data_over = parse_overrides(args.set), parse_overrides(args.data_set, "--data-set")
    info = setup(get_device())                      # multi-GPU: join now so all processes share one name
    name = broadcast_object(args.name or "smoke_" + time.strftime("%Y%m%d_%H%M%S"), info)
    print_header("smoke test", {"train config": train_path, "data config": data_path},
                 {"train set": user, "data set": data_over})

    results: list = []
    data_cfg = load_data_config(data_path, project_root=PROJECT_ROOT, overrides=data_over)
    phase1, phase2 = smoke_overrides(args.steps, name, args.tiny, user)
    t0 = time.time()
    try:
        # ---- training: phase 1, then resume in phase 2 ----
        r1 = train(data_cfg, load_train_config(train_path, project_root=PROJECT_ROOT, overrides=phase1))
        r2 = train(data_cfg, load_train_config(train_path, project_root=PROJECT_ROOT, overrides=phase2))
    except Exception as exc:                        # report clearly, then fail
        cleanup()
        if is_main_process():
            check(results, "training runs", False, f"{type(exc).__name__}: {exc}")
            print("\nSMOKE TEST FAILED")
        raise SystemExit(1)
    cleanup()                                       # multi-GPU part done; checks + eval on the main process
    if not is_main_process():
        return

    run = RunPaths(r2["run_dir"])
    check(results, "training phase 1 reached its steps", r1["step"] == phase1["max_steps"], f"step {r1['step']}")
    check(results, "resume from latest.pt (phase 2)", r2["step"] == phase2["max_steps"],
          f"step {r1['step']} -> {r2['step']}")
    m = pd.read_csv(run.metrics) if run.metrics.exists() else pd.DataFrame(columns=["kind", "name", "value"])
    losses = m[(m.kind == "train") & (m.name == "loss")].value.astype(float).tolist()
    check(results, "train loss logged and finite", len(losses) > 0 and all(math.isfinite(x) for x in losses),
          f"{len(losses)} values, last {losses[-1]:.4f}" if losses else "none")
    vals = m[(m.kind == "val") & (m.name == "val_loss")].value.astype(float).tolist()
    check(results, "validation loss computed", len(vals) > 0 and all(math.isfinite(x) for x in vals),
          f"{len(vals)} values" if vals else "no val split / none")
    for ck in ("latest", "best"):
        check(results, f"checkpoint {ck}.pt written", run.checkpoint(ck).exists())
    check(results, "preview sample PNG written", any(run.samples.glob("*.png")))

    # ---- evaluation on the smoke checkpoint ----
    if not args.skip_eval:
        try:
            from aneurysm.evaluation import run_evaluation

            ecfg = load_eval_config(project_root=PROJECT_ROOT, overrides={
                "run_dir": str(run.root), "checkpoint": "latest", "max_patches": 1,
                "jump_length": 2, "jump_n_sample": 1,
                "case_2_real_dilated": False, "case_3_pseudo_actual": False,
                "histogram_control": False, "copy_check": False, "full_scan": False,
                "work_dir": phase2.get("work_dir")})
            res = run_evaluation(ecfg, project_root=PROJECT_ROOT, show=False)
            ok = (res["out_dir"] / "summary.csv").exists()
            check(results, "evaluation (case 1 + case 4, one patch)", ok, res["out_dir"].relative_to(PROJECT_ROOT).as_posix())
        except Exception as exc:
            check(results, "evaluation (case 1 + case 4, one patch)", False, f"{type(exc).__name__}: {exc}")

    failed = [r for r in results if not r[1]]
    print(f"\n{'SMOKE TEST PASSED' if not failed else 'SMOKE TEST FAILED'}: "
          f"{len(results) - len(failed)}/{len(results)} checks in {time.time() - t0:.0f} s | "
          f"run folder {run.root.relative_to(PROJECT_ROOT).as_posix()}/")
    if not failed:
        print("Look at the memory check line and s/step above to plan the real run.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
