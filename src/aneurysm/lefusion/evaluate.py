"""Evaluate a trained LeFusion run.

Usage (see 03_eval_lefusion.ipynb):
    results = evaluate_run(eval_cfg)

Tasks (each can be switched off in configs/lefusion_eval.yaml)
----------------------------------------------------------------
1. plot_curves        train and val loss from metrics.csv
2. rebuild            generate each aneurysm from noise inside its own mask, with its own histogram
3. new_shapes         grow the mask by `grow_voxels` and generate a bigger aneurysm
4. histogram_control  same patch + mask with low / medium / high contrast histograms
5. copy_check         is a generated aneurysm too similar to a training patch?
6. full_scan          paste one generated patch back into its CT scan (NIfTI + 3-plane figure)

Everything is saved to  outputs/<run_name>/eval/<checkpoint>_<split>/  and shown inline.

Background / mask tests (see 03b_eval_lefusion.ipynb):
    results = evaluate_backgrounds(cfg)
1. noisy real          + actual mask
2. noisy real          + dilated mask
3. noisy pseudo-healthy + actual mask
4. noisy pseudo-healthy + dilated mask
Saved to  outputs/<run_name>/eval_b/<checkpoint>_<split>/
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

from ..config import find_project_root
from ..data.datasets import AneurysmPatchDataset
from ..data.pipeline import prepare_data
from ..data.volume import load_canonical
from ..env import get_device
from .config import BackgroundEvalConfig, EvalConfig
from .pseudo_healthy import ball, dilate, make_pseudo_healthy
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


def _load_run(cfg, root: Path, device) -> dict:
    """Trained model + the evaluation patches of `cfg.split` (used by both entry points).

    The model and data settings come from the checkpoint, so they always match the weights.
    """
    run_dir = root / cfg.output_dir / cfg.run_name
    diffusion, data_cfg, train_cfg, info = load_trained_model(run_dir, cfg.checkpoint, device, cfg.use_ema, root)
    _say(f"Model: {info['path'].relative_to(root).as_posix()} | step {info['step']} | "
         f"{info['weights']} weights | best val loss {info['best_val']:.4f} | device {device}")
    data = prepare_data(data_cfg, verbose=False)
    ds = AneurysmPatchDataset(data["index"], data_cfg, cfg.split, augment=False)
    if len(ds) == 0:
        raise RuntimeError(f"No patches in split '{cfg.split}'.")
    n = len(ds) if cfg.max_patches <= 0 else min(cfg.max_patches, len(ds))
    gen_kw = dict(timesteps=train_cfg.timesteps, jump_length=cfg.jump_length,
                  jump_n_sample=cfg.jump_n_sample, amp=cfg.amp)
    return {"run_dir": run_dir, "diffusion": diffusion, "data_cfg": data_cfg, "data": data,
            "items": [ds[i] for i in range(n)], "n_total": len(ds), "gen_kw": gen_kw}


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
    device = get_device()

    # ---- model + data (settings come from the checkpoint) ----
    run = _load_run(eval_cfg, root, device)
    run_dir, diffusion, data_cfg, data = run["run_dir"], run["diffusion"], run["data_cfg"], run["data"]
    items, gen_kw, index = run["items"], run["gen_kw"], data["index"]
    n = len(items)
    out_dir = run_dir / "eval" / f"{eval_cfg.checkpoint}_{eval_cfg.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"out_dir": out_dir}
    thr = float(hu_to_norm(eval_cfg.vessel_hu, data_cfg))
    _say(f"Split '{eval_cfg.split}': evaluating {n} of {run['n_total']} patches | outputs: "
         f"{out_dir.relative_to(root).as_posix()}\n")

    # ---- 1. curves ----
    if eval_cfg.plot_curves:
        _say("1) Training curves")
        plot_training_curves(run_dir, out_dir / "1_curves.png", show)

    # ---- 2. rebuild ----
    rebuilt: dict[str, np.ndarray] = {}
    if eval_cfg.rebuild:
        _say("2) Rebuild: generate each aneurysm from noise inside its mask (real voxels inside are not used)")
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


# ======================================================================
# background / mask tests  (03b_eval_lefusion.ipynb)
# ======================================================================
#  id: (file name, title, background, generation mask)
BACKGROUND_TESTS = {
    1: ("1_real_actual", "noisy real + actual mask", "real", "actual"),
    2: ("2_real_dilated", "noisy real + dilated mask", "real", "dilated"),
    3: ("3_pseudo_actual", "noisy pseudo-healthy + actual mask", "pseudo", "actual"),
    4: ("4_pseudo_dilated", "noisy pseudo-healthy + dilated mask", "pseudo", "dilated"),
}


def _edge_step_hu(img: np.ndarray, mask: np.ndarray, cfg) -> float:
    """Mean HU on the inner border of `mask` minus mean HU just outside it.

    A real aneurysm has a blurred edge (moderate step). A generated aneurysm that
    stops sharply at the mask (cut-out look) gives a clearly bigger step.
    """
    m = mask > 0
    inner = m & ~ndimage.binary_erosion(m, structure=ball(1))
    outer = dilate(m, 1) & ~m
    if not inner.any() or not outer.any():
        return float("nan")
    return float(norm_to_hu(img[inner].mean(), cfg) - norm_to_hu(img[outer].mean(), cfg))


def _prepare_background_cases(items: list[dict], cfg: BackgroundEvalConfig, data_cfg) -> list[dict]:
    """For every patch: real image, pseudo-healthy image, actual mask, dilated mask, histogram."""
    vessel_norm = float(hu_to_norm(cfg.vessel_hu, data_cfg))
    cases = []
    for k, it in enumerate(items):
        real = it["image"][0].numpy()
        actual = it["center_mask"][0].numpy()
        ph = make_pseudo_healthy(real, actual, it["mask"][0].numpy(), cfg.dilate_voxels, vessel_norm,
                                 cfg.tissue_ring_voxels, cfg.noise_blur_sigma, seed=cfg.seed + k)
        dilated = ph["region"]
        cases.append({
            "patch_id": it["patch_id"], "real": real, "pseudo": ph["image"],
            "actual": actual, "dilated": dilated, "hist": it["hist"],
            "tissue_hu": float(norm_to_hu(ph["tissue_norm"], data_cfg)),
            "tissue_std_hu": float(ph["tissue_std"] * (data_cfg.hu_max - data_cfg.hu_min) / 2),
            "parent_voxels": int(ph["parent"].sum()),
            "left_bright": float((ph["image"][dilated > 0] > vessel_norm).mean()),
        })
    return cases


def _check_pseudo_healthy(cases: list[dict], out_path: Path, show: bool) -> pd.DataFrame:
    """Print + plot the pseudo-healthy patches (check them before trusting tests 3 and 4)."""
    stats, rows = [], []
    for c in cases:
        stats.append({"patch_id": c["patch_id"], "actual_voxels": int(c["actual"].sum()),
                      "dilated_voxels": int(c["dilated"].sum()), "parent_vessel_voxels_kept": c["parent_voxels"],
                      "fill_hu": c["tissue_hu"], "noise_std_hu": c["tissue_std_hu"],
                      "still_bright_in_dilated": c["left_bright"]})
        rows.append({"real": c["real"], "gen": c["pseudo"], "mask": c["actual"], "outer_mask": c["dilated"],
                     "label": f"{c['patch_id']} fill {c['tissue_hu']:.0f} HU"})
        _say(f"   {c['patch_id']}: actual {stats[-1]['actual_voxels']} -> dilated {stats[-1]['dilated_voxels']} "
             f"voxels | parent vessel kept {c['parent_voxels']} | fill {c['tissue_hu']:.0f} "
             f"+- {c['tissue_std_hu']:.0f} HU | still bright {c['left_bright']:.2f}")
    plot_rows(rows, out_path, show, names={"generated": "pseudo-healthy"},
              title="Pseudo-healthy check (yellow = actual mask, red = dilated mask = removed area)")
    return pd.DataFrame(stats)


def _run_background_test(test_id: int, cases: list[dict], diffusion, data_cfg, cfg: BackgroundEvalConfig,
                         gen_kw: dict, device, thr: float) -> tuple[pd.DataFrame, list[dict]]:
    """Generate every case with one background + mask combination; stats inside the actual mask."""
    _, title, bg_key, mask_key = BACKGROUND_TESTS[test_id]
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None].to(device)
    stats, rows = [], []
    for k, c in enumerate(cases):
        t0 = time.time()
        gen_mask = c[mask_key]
        gen = generate(diffusion, to_t(c[bg_key]), to_t(gen_mask), c["hist"][None].to(device),
                       seed=cfg.seed + k, **gen_kw)[0, 0].cpu().numpy()
        st = {"test": test_id, "name": title, "patch_id": c["patch_id"],
              **_patch_stats(c["real"], gen, c["actual"], data_cfg, thr),
              "gen_fill_generated_area": float((gen[gen_mask > 0] > thr).mean()),
              "real_edge_step_hu": _edge_step_hu(c["real"], c["actual"], data_cfg),
              "gen_edge_step_hu": _edge_step_hu(gen, c["actual"], data_cfg),
              "seconds": time.time() - t0}
        stats.append(st)
        rows.append({"real": c["real"], "input": c[bg_key], "gen": gen, "mask": c["actual"],
                     "outer_mask": c["dilated"], "label": c["patch_id"]})
        _say(f"   {c['patch_id']}: HU in mask real {st['real_hu_mean']:.0f} / gen {st['gen_hu_mean']:.0f} | "
             f"fill real {st['real_fill']:.2f} / gen {st['gen_fill']:.2f} | edge step real "
             f"{st['real_edge_step_hu']:.0f} / gen {st['gen_edge_step_hu']:.0f} HU | {st['seconds']:.0f} s")
    return pd.DataFrame(stats), rows


def _background_summary(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per test: averages over the patches."""
    cols = ["real_hu_mean", "gen_hu_mean", "real_hu_std", "gen_hu_std", "mae_hu_inside", "real_fill",
            "gen_fill", "gen_fill_generated_area", "real_edge_step_hu", "gen_edge_step_hu"]
    df = pd.concat(tables, ignore_index=True)
    summary = df.groupby(["test", "name"], sort=False)[cols].mean().reset_index()
    return summary


def evaluate_backgrounds(cfg: BackgroundEvalConfig, project_root: Path | None = None, show: bool = True) -> dict:
    """Run the background / mask tests in the order of `cfg.tests`.

    Same patches, same histogram and same seed in every test, so only the
    background and the generation mask change. All numbers are measured inside
    the actual mask and compared with the real aneurysm.
    Returns {"out_dir", "pseudo_check", "tests": {id: table}, "summary"}.
    """
    root = Path(project_root or find_project_root())
    device = get_device()
    run = _load_run(cfg, root, device)
    data_cfg, items = run["data_cfg"], run["items"]
    out_dir = run["run_dir"] / "eval_b" / f"{cfg.checkpoint}_{cfg.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = float(hu_to_norm(cfg.vessel_hu, data_cfg))
    _say(f"Split '{cfg.split}': {len(items)} of {run['n_total']} patches | tests {cfg.tests} | "
         f"dilated mask = actual + {cfg.dilate_voxels} voxels | outputs: {out_dir.relative_to(root).as_posix()}\n")

    # ---- 0. pseudo-healthy patches (needed by tests 3 and 4, always checked) ----
    _say(f"0) Pseudo-healthy patches: aneurysm + {cfg.dilate_voxels}-voxel rim removed, parent vessel kept")
    cases = _prepare_background_cases(items, cfg, data_cfg)
    results: dict = {"out_dir": out_dir, "tests": {}}
    results["pseudo_check"] = _check_pseudo_healthy(cases, out_dir / "0_pseudo_healthy_check.png", show)
    results["pseudo_check"].to_csv(out_dir / "0_pseudo_healthy_check.csv", index=False)
    _say("   'still bright' should be close to 0 (aneurysm fully removed).")

    # ---- 1-4. tests ----
    for test_id in cfg.tests:
        name, title, _, mask_key = BACKGROUND_TESTS[test_id]
        _say(f"\n{test_id}) {title}")
        table, rows = _run_background_test(test_id, cases, run["diffusion"], data_cfg, cfg,
                                           run["gen_kw"], device, thr)
        table.to_csv(out_dir / f"{name}.csv", index=False)
        plot_rows(rows, out_dir / f"{name}.png", show,
                  title=f"{test_id}) {title}  (yellow = actual mask, red = dilated mask; "
                        f"generated inside the {mask_key} mask)")
        results["tests"][test_id] = table

    # ---- summary ----
    summary = _background_summary(list(results["tests"].values()))
    summary.to_csv(out_dir / "summary.csv", index=False)
    results["summary"] = summary
    _say("\nSummary (mean over patches, inside the actual mask)")
    _say(summary.round(2).to_string(index=False))
    _say("\nHow to read it:")
    _say("   1 vs 2 : effect of the dilated mask on a real background")
    _say("   1 vs 4 : 4 close to 1 -> the model does not need the real rim; it works on a healthy-looking vessel")
    _say("   3      : gen edge step much bigger than real -> generating only inside the actual mask leaves a hard edge")
    _say(f"\nAll results in {out_dir.relative_to(root).as_posix()}/")
    return results
