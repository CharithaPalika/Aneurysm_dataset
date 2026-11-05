"""Evaluate a trained LeFusion run.

Usage (see 03_eval_lefusion.ipynb):
    results = evaluate_run(eval_cfg)

Tasks (each can be switched off in configs/lefusion_eval.yaml)
----------------------------------------------------------------
1. plot_curves        train and val loss from metrics.csv
2. rebuild            erase each real aneurysm and regenerate it with its own mask + histogram
3. new_shapes         grow the mask by `grow_voxels` and generate a bigger aneurysm
4. histogram_control  same patch + mask with low / medium / high contrast histograms
5. copy_check         is a generated aneurysm too similar to a training patch?
6. full_scan          paste one generated patch back into its CT scan (NIfTI + 3-plane figure)

Everything is saved to  outputs/<run_name>/eval/<checkpoint>_<split>/  and shown inline.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import find_project_root
from ..data.datasets import AneurysmPatchDataset
from ..data.pipeline import prepare_data
from ..data.volume import load_canonical
from ..env import get_device
from .config import EvalConfig
from .sample import (fill_ratio, generate, grow_mask, hu_to_norm, load_trained_model,
                     model_patch_affine, norm_to_hu, paste_patch_into_volume, plot_rows)


# ======================================================================
# helpers
# ======================================================================
def _batch(item: dict, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One dataset item -> (image, center_mask, hist) with a batch axis, on `device`."""
    return (item["image"][None].to(device), item["center_mask"][None].to(device), item["hist"][None].to(device))


def _ncc(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    """Normalised cross-correlation of a and b inside `region` (1 = identical pattern)."""
    x, y = a[region].astype(np.float64), b[region].astype(np.float64)
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std()))


def _patch_stats(real: np.ndarray, gen: np.ndarray, mask: np.ndarray, cfg, thr_norm: float) -> dict:
    inside = mask > 0
    r_hu, g_hu = norm_to_hu(real[inside], cfg), norm_to_hu(gen[inside], cfg)
    return {
        "mask_voxels": int(inside.sum()),
        "real_hu_mean": float(r_hu.mean()), "gen_hu_mean": float(g_hu.mean()),
        "real_hu_std": float(r_hu.std()), "gen_hu_std": float(g_hu.std()),
        "mae_hu_inside": float(np.abs(g_hu - r_hu).mean()),
        "real_fill": float((real[inside] > thr_norm).mean()),
        "gen_fill": float((gen[inside] > thr_norm).mean()),
    }


def _say(msg: str) -> None:
    print(msg, flush=True)


# ======================================================================
# tasks
# ======================================================================
def plot_training_curves(run_dir: Path, out_path: Path, show: bool = True) -> pd.DataFrame | None:
    """Train loss (smoothed) and val loss over steps."""
    import matplotlib.pyplot as plt

    path = run_dir / "metrics.csv"
    if not path.exists():
        _say("  no metrics.csv found")
        return None
    m = pd.read_csv(path)
    tr = m[(m.kind == "train") & (m.name == "loss")].sort_values("step")
    va = m[(m.kind == "val") & (m.name == "val_loss")].sort_values("step")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    if len(tr):
        ax[0].plot(tr.step, tr.value, alpha=0.35, label="train loss")
        ax[0].plot(tr.step, tr.value.rolling(10, min_periods=1).mean(), label="train (smoothed)")
    if len(va):
        ax[0].plot(va.step, va.value, "o-", label="val loss")
    ax[0].set(xlabel="step", ylabel="loss", title="Loss"); ax[0].legend()
    for name in sorted(m[m.kind == "val"].name.unique()):
        if name.startswith("val_t"):
            d = m[(m.kind == "val") & (m.name == name)].sort_values("step")
            ax[1].plot(d.step, d.value, "o-", label=f"t = {name[5:]}")
    ax[1].set(xlabel="step", ylabel="val loss", title="Val loss per noise level"); ax[1].legend()
    plt.tight_layout(); fig.savefig(out_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)
    return m


def _reference_histograms(index: pd.DataFrame, data_cfg) -> dict:
    """Low / medium / high contrast histograms from TRAIN patches (10th / 50th / 90th percentile)."""
    ds = AneurysmPatchDataset(index, data_cfg, "train", augment=False)
    means, hists = [], []
    for i in range(len(ds)):
        it = ds[i]
        inside = it["image"][it["center_mask"] > 0]
        if inside.numel():
            means.append(float(inside.mean())); hists.append(it["hist"])
    order = np.argsort(means)
    pick = {"low": order[int(0.1 * (len(order) - 1))], "medium": order[int(0.5 * (len(order) - 1))],
            "high": order[int(0.9 * (len(order) - 1))]}
    return {k: (hists[i], float(norm_to_hu(means[i], data_cfg))) for k, i in pick.items()}


def _plot_hist_control(rows: list[dict], out_path: Path, show: bool) -> None:
    import matplotlib.pyplot as plt

    n, cols = len(rows), 1 + len(rows[0]["gens"])
    fig, axs = plt.subplots(n, cols, figsize=(3 * cols, 3 * n), squeeze=False)
    for k, r in enumerate(rows):
        m = r["mask"]; z = int(np.argmax(m.sum(axis=(1, 2))))
        panels = [("real", r["real"], r["real_hu"])] + [(name, g, hu) for name, (g, hu) in r["gens"].items()]
        for j, (name, img, hu) in enumerate(panels):
            axs[k, j].imshow(img[z], cmap="gray", vmin=-1, vmax=1, origin="lower")
            axs[k, j].contour(m[z], levels=[0.5], colors="r", linewidths=0.8, origin="lower")
            axs[k, j].set_title(f"{r['label']}  {name}\n{hu:.0f} HU in mask", fontsize=8); axs[k, j].axis("off")
    plt.tight_layout(); fig.savefig(out_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)


def _plot_full_scan(orig: np.ndarray, syn: np.ndarray, mask: np.ndarray, centre, spacing,
                    title: str, out_path: Path, show: bool, crop_mm: float = 40.0) -> None:
    """Axial / coronal / sagittal zoom through the aneurysm: original vs synthetic."""
    import matplotlib.pyplot as plt

    c = np.round(centre).astype(int)
    h = [int(crop_mm / 2 / s) for s in spacing]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for col, (axis, name) in enumerate([(2, "axial"), (1, "coronal"), (0, "sagittal")]):
        sl = [slice(max(0, c[a] - h[a]), c[a] + h[a]) for a in range(3)]
        sl[axis] = c[axis]
        others = [a for a in range(3) if a != axis]
        aspect = spacing[others[1]] / spacing[others[0]]
        for row, (vol, lab) in enumerate([(orig, "original"), (syn, "synthetic")]):
            axs[row, col].imshow(vol[tuple(sl)].T, cmap="gray", vmin=-150, vmax=550, origin="lower", aspect=aspect)
            axs[row, col].contour(mask[tuple(sl)].T, levels=[0.5], colors="r", linewidths=0.8, origin="lower")
            axs[row, col].set_title(f"{lab} - {name}", fontsize=9); axs[row, col].axis("off")
    fig.suptitle(title, fontsize=10); plt.tight_layout(); fig.savefig(out_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)


# ======================================================================
# main entry point
# ======================================================================
def evaluate_run(eval_cfg: EvalConfig, project_root: Path | None = None, show: bool = True) -> dict:
    """Run the evaluation tasks. Returns a dict of result tables and the output folder."""
    import nibabel as nib

    root = Path(project_root or find_project_root())
    run_dir = root / eval_cfg.output_dir / eval_cfg.run_name
    out_dir = run_dir / "eval" / f"{eval_cfg.checkpoint}_{eval_cfg.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    results: dict = {"out_dir": out_dir}

    # ---- model + data (settings come from the checkpoint) ----
    diffusion, data_cfg, train_cfg, info = load_trained_model(run_dir, eval_cfg.checkpoint, device,
                                                              eval_cfg.use_ema, root)
    _say(f"Model: {info['path'].relative_to(root).as_posix()} | step {info['step']} | "
         f"{info['weights']} weights | best val loss {info['best_val']:.4f} | device {device}")
    data = prepare_data(data_cfg, verbose=False)
    index = data["index"]
    ds = AneurysmPatchDataset(index, data_cfg, eval_cfg.split, augment=False)
    if len(ds) == 0:
        raise RuntimeError(f"No patches in split '{eval_cfg.split}'.")
    n = len(ds) if eval_cfg.max_patches <= 0 else min(eval_cfg.max_patches, len(ds))
    items = [ds[i] for i in range(n)]
    thr = float(hu_to_norm(eval_cfg.vessel_hu, data_cfg))
    gen_kw = dict(timesteps=train_cfg.timesteps, jump_length=eval_cfg.jump_length,
                  jump_n_sample=eval_cfg.jump_n_sample, amp=eval_cfg.amp)
    _say(f"Split '{eval_cfg.split}': evaluating {n} of {len(ds)} patches | outputs: "
         f"{out_dir.relative_to(root).as_posix()}\n")

    # ---- 1. curves ----
    if eval_cfg.plot_curves:
        _say("1) Training curves")
        plot_training_curves(run_dir, out_dir / "1_curves.png", show)

    # ---- 2. rebuild ----
    rebuilt: dict[str, np.ndarray] = {}
    if eval_cfg.rebuild:
        _say("2) Rebuild: erase each real aneurysm and regenerate it")
        rows, stats = [], []
        for k, it in enumerate(items):
            t0 = time.time()
            img, msk, hist = _batch(it, device)
            gen = generate(diffusion, img, msk, hist, seed=eval_cfg.seed + k, **gen_kw)
            r, g, m = img[0, 0].cpu().numpy(), gen[0, 0].cpu().numpy(), msk[0, 0].cpu().numpy()
            rebuilt[it["patch_id"]] = g
            st = {"patch_id": it["patch_id"], **_patch_stats(r, g, m, data_cfg, thr)}
            stats.append(st)
            rows.append({"real": r, "gen": g, "mask": m,
                         "label": f"{it['patch_id']} ({st['real_hu_mean']:.0f}/{st['gen_hu_mean']:.0f} HU)"})
            _say(f"   {it['patch_id']}: HU in mask real {st['real_hu_mean']:.0f} / gen {st['gen_hu_mean']:.0f} | "
                 f"fill real {st['real_fill']:.2f} / gen {st['gen_fill']:.2f} | {time.time() - t0:.0f} s")
        results["rebuild"] = pd.DataFrame(stats)
        results["rebuild"].to_csv(out_dir / "2_rebuild.csv", index=False)
        plot_rows(rows, out_dir / "2_rebuild.png", show, title="Rebuild: real vs regenerated (red = mask)")

    # ---- 3. new shapes ----
    if eval_cfg.new_shapes:
        _say(f"3) New shapes: mask grown by {eval_cfg.grow_voxels} voxels")
        rows, stats = [], []
        for k, it in enumerate(items):
            img, msk, hist = _batch(it, device)
            big = grow_mask(msk, eval_cfg.grow_voxels)
            gen = generate(diffusion, img, big, hist, seed=eval_cfg.seed + 100 + k, **gen_kw)
            fr = fill_ratio(gen, big, thr)
            r, g, m = img[0, 0].cpu().numpy(), gen[0, 0].cpu().numpy(), big[0, 0].cpu().numpy()
            stats.append({"patch_id": it["patch_id"], "old_voxels": int(msk.sum()), "new_voxels": int(big.sum()),
                          "gen_fill": fr, "accept_(fill>=0.7)": fr >= 0.7})
            rows.append({"real": r, "gen": g, "mask": m, "label": f"{it['patch_id']} fill {fr:.2f}"})
            _say(f"   {it['patch_id']}: mask {int(msk.sum())} -> {int(big.sum())} voxels | fill {fr:.2f}"
                 f"{'' if fr >= 0.7 else '  <- would be rejected'}")
        results["new_shapes"] = pd.DataFrame(stats)
        results["new_shapes"].to_csv(out_dir / "3_new_shapes.csv", index=False)
        plot_rows(rows, out_dir / "3_new_shapes.png", show, title="New shapes: grown mask (red)")

    # ---- 4. histogram control ----
    if eval_cfg.histogram_control:
        _say("4) Histogram control: low / medium / high contrast (from training patches)")
        refs = _reference_histograms(index, data_cfg)
        _say("   reference mean HU: " + ", ".join(f"{k} {v[1]:.0f}" for k, v in refs.items()))
        rows = []
        for k, it in enumerate(items[:eval_cfg.hist_patches]):
            img, msk, _ = _batch(it, device)
            m = msk[0, 0].cpu().numpy()
            r = img[0, 0].cpu().numpy()
            gens = {}
            for name, (h, _hu) in refs.items():
                g = generate(diffusion, img, msk, h[None].to(device), seed=eval_cfg.seed + 200 + k, **gen_kw)
                g = g[0, 0].cpu().numpy()
                gens[name] = (g, float(norm_to_hu(g[m > 0].mean(), data_cfg)))
            rows.append({"real": r, "real_hu": float(norm_to_hu(r[m > 0].mean(), data_cfg)),
                         "mask": m, "gens": gens, "label": it["patch_id"]})
            _say(f"   {it['patch_id']}: " + ", ".join(f"{nm} -> {hu:.0f} HU" for nm, (_, hu) in gens.items()))
        _plot_hist_control(rows, out_dir / "4_histogram_control.png", show)

    # ---- 5. copy check ----
    if eval_cfg.copy_check and rebuilt:
        _say("5) Copy check: similarity to the closest TRAIN patch (inside the grown mask)")
        train_ds = AneurysmPatchDataset(index, data_cfg, "train", augment=False)
        train_imgs = [(train_ds.rows.iloc[i]["patch_id"], train_ds[i]["image"][0].numpy()) for i in range(len(train_ds))]
        stats = []
        for it in items:
            pid = it["patch_id"]
            if pid not in rebuilt:
                continue
            region = grow_mask(it["center_mask"][None], 3)[0, 0].numpy() > 0
            real, gen = it["image"][0].numpy(), rebuilt[pid]
            sims_gen = [(_ncc(gen, t, region), tid) for tid, t in train_imgs]
            sims_real = [_ncc(real, t, region) for _, t in train_imgs]
            best_gen = max(sims_gen, key=lambda s: -1 if np.isnan(s[0]) else s[0])
            stats.append({"patch_id": pid, "closest_train_patch": best_gen[1],
                          "max_ncc_generated": best_gen[0], "max_ncc_real": float(np.nanmax(sims_real)),
                          "ncc_generated_vs_real": _ncc(gen, real, region)})
        df = pd.DataFrame(stats)
        df["looks_copied"] = (df.max_ncc_generated > 0.9) & (df.max_ncc_generated > df.max_ncc_real + 0.1)
        results["copy_check"] = df
        df.to_csv(out_dir / "5_copy_check.csv", index=False)
        _say(df.round(3).to_string(index=False))
        _say("   'looks_copied' = generated is much closer to a training patch than the real aneurysm is.")

    # ---- 6. full scan ----
    if eval_cfg.full_scan and rebuilt:
        pid = eval_cfg.full_scan_patch or items[0]["patch_id"]
        if pid not in rebuilt:
            raise ValueError(f"full_scan_patch {pid} was not evaluated (choose one of {list(rebuilt)})")
        _say(f"6) Full scan: pasting the regenerated aneurysm of {pid} back into its CT scan")
        row = index[index.patch_id == pid].iloc[0]
        case = data["catalog"].set_index("case_id").loc[row.case_id]
        vol, aff, spacing = load_canonical(data_cfg.abs(case["image_path"]))
        full_mask, _, _ = load_canonical(data_cfg.abs(case["mask_path"]), is_mask=True)
        with np.load(data_cfg.abs(row.patch_path)) as f:
            stored_affine = f["affine"]
        item = next(it for it in items if it["patch_id"] == pid)
        # dataset order (z, y, x) -> volume order (x, y, z)
        gen_hu = norm_to_hu(rebuilt[pid], data_cfg).transpose(2, 1, 0)
        pm = item["center_mask"][0].numpy().transpose(2, 1, 0)
        syn = paste_patch_into_volume(vol, aff, gen_hu, pm, model_patch_affine(stored_affine, data_cfg.patch_margin))
        out_nii = out_dir / "6_full_scan"
        out_nii.mkdir(exist_ok=True)
        nib.save(nib.Nifti1Image(syn, aff), str(out_nii / f"Syn_{pid}.nii.gz"))
        nib.save(nib.Nifti1Image(full_mask, aff), str(out_nii / f"Syn_{pid}_mask.nii.gz"))
        centre = np.array([row.center_x, row.center_y, row.center_z])
        changed = int((syn != vol).sum())
        _plot_full_scan(vol, syn, full_mask, centre, spacing, f"{pid}: original vs synthetic ({changed} voxels changed)",
                        out_dir / "6_full_scan.png", show)
        _say(f"   saved {out_nii.relative_to(root).as_posix()}/Syn_{pid}.nii.gz (+ _mask) | {changed} voxels changed")

    # ---- summary ----
    if "rebuild" in results:
        rb = results["rebuild"]
        _say("\nSummary (rebuild)")
        _say(f"   mean HU in mask: real {rb.real_hu_mean.mean():.0f} / generated {rb.gen_hu_mean.mean():.0f}")
        _say(f"   HU spread in mask: real {rb.real_hu_std.mean():.0f} / generated {rb.gen_hu_std.mean():.0f}")
        _say(f"   mask filled (> {eval_cfg.vessel_hu:.0f} HU): real {rb.real_fill.mean():.2f} / "
             f"generated {rb.gen_fill.mean():.2f}")
    if "copy_check" in results:
        _say(f"   patches that look copied: {int(results['copy_check'].looks_copied.sum())} of "
             f"{len(results['copy_check'])}")
    _say(f"\nAll results in {out_dir.relative_to(root).as_posix()}/")
    return results
