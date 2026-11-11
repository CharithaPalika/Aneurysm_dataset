"""Evaluate a trained run.

Usage:
    results = run_evaluation(eval_cfg)
    (03_eval_lefusion.ipynb, or from a terminal: python scripts/run_eval.py)

Checks (each switched on / off in configs/eval/lefusion_eval.yaml)
------------------------------------------------------------------
training_curves        train and val loss from metrics.csv
case_1_real_actual     noisy real          + actual mask
case_2_real_dilated    noisy real          + dilated mask
case_3_pseudo_actual   noisy pseudo-healthy + actual mask
case_4_pseudo_dilated  noisy pseudo-healthy + dilated mask
histogram_control      same patch + mask with low / medium / high contrast histograms
copy_check             is a case-1 aneurysm too similar to a training patch?
full_scan              paste one case-1 patch back into its CT scan (NIfTI + 3-plane figure)

actual mask    = the annotated aneurysm (yellow in the figures)
dilated mask   = actual mask + `dilate_voxels` (round), without parent vessel / other aneurysms (red)
pseudo-healthy = aneurysm + its rim removed, parent vessel kept, filled with tissue
Cases 1-4 use the same patches, histogram and seed: only the background and the mask change.
All case numbers are measured inside the actual mask and compared with the real aneurysm.

Everything is saved to  <run folder>/eval/<checkpoint>_<split>/  with a copy of the config.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage

from ..config import EvalConfig, RunPaths, find_project_root, resolve_path
from ..data.datasets import AneurysmPatchDataset
from ..data.intensity import hu_to_norm, norm_to_hu
from ..data.pipeline import prepare_data
from ..data.pseudo_healthy import ball, dilate, make_pseudo_healthy
from ..data.volume import load_canonical, model_patch_affine, paste_patch_into_volume
from ..training.sampling import generate, load_trained_model
from ..utils.env import get_device
from ..utils.plotting import plot_panels

#  id: (config switch, title, background, generation mask)
CASES = {
    1: ("case_1_real_actual", "noisy real + actual mask", "real", "actual"),
    2: ("case_2_real_dilated", "noisy real + dilated mask", "real", "dilated"),
    3: ("case_3_pseudo_actual", "noisy pseudo-healthy + actual mask", "pseudo", "actual"),
    4: ("case_4_pseudo_dilated", "noisy pseudo-healthy + dilated mask", "pseudo", "dilated"),
}


# ======================================================================
# helpers
# ======================================================================
def _say(msg: str) -> None:
    print(msg, flush=True)


def _show(path: Path, root: Path) -> str:
    """Path relative to the project folder when possible (shorter prints)."""
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


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


# ======================================================================
# run folder, model and patches
# ======================================================================
def resolve_run_dir(cfg: EvalConfig, root: Path) -> Path:
    """`run_dir` if given, else <output_dir>/<run_name>. Lists the existing runs if not found."""
    run_dir = resolve_path(cfg.run_dir, root) if cfg.run_dir else (root / cfg.output_dir / cfg.run_name)
    if not RunPaths(run_dir).is_run():
        out = root / cfg.output_dir
        runs = sorted(p.name for p in out.iterdir() if RunPaths(p).is_run()) if out.is_dir() else []
        raise FileNotFoundError(f"No trained run at {run_dir} (no checkpoints/ folder). "
                                f"Runs in {_show(out, root)}/: {runs or 'none'}. Set run_name or run_dir.")
    return run_dir


def _load_run(cfg: EvalConfig, root: Path, device) -> dict:
    """Trained model + the evaluation patches of `cfg.split`.

    Model and data settings come from the checkpoint, so they always match the weights
    (also the patch setting: the patches of the trained setting are used, cut if needed).
    """
    run_dir = resolve_run_dir(cfg, root)
    diffusion, data_cfg, train_cfg, info = load_trained_model(run_dir, cfg.checkpoint, device, cfg.use_ema, root)
    _say(f"Model: {_show(info['path'], root)} | step {info['step']} | {info['weights']} weights | "
         f"best val loss {info['best_val']:.4f} | device {device}")
    data = prepare_data(data_cfg, verbose=False)
    ds = AneurysmPatchDataset(data["index"], data_cfg, cfg.split, augment=False)
    if len(ds) == 0:
        raise RuntimeError(f"No patches in split '{cfg.split}'.")
    n = len(ds) if cfg.max_patches <= 0 else min(cfg.max_patches, len(ds))
    gen_kw = dict(timesteps=train_cfg.timesteps, jump_length=cfg.jump_length,
                  jump_n_sample=cfg.jump_n_sample, amp=cfg.amp)
    return {"run_dir": run_dir, "diffusion": diffusion, "data_cfg": data_cfg, "data": data,
            "items": [ds[i] for i in range(n)], "n_total": len(ds), "gen_kw": gen_kw}


def _prepare_cases(items: list[dict], cfg: EvalConfig, data_cfg) -> list[dict]:
    """Per patch: real image, actual + dilated mask, pseudo-healthy image, histogram (model values)."""
    vessel_norm = float(hu_to_norm(cfg.vessel_hu, data_cfg))
    cases = []
    for k, it in enumerate(items):
        real, actual = it["image"][0].numpy(), it["center_mask"][0].numpy()
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


# ======================================================================
# training curves
# ======================================================================
def plot_training_curves(run_dir: Path, out_path: Path, show: bool = True) -> pd.DataFrame | None:
    """Train loss (smoothed) and val loss over steps."""
    import matplotlib.pyplot as plt

    path = RunPaths(run_dir).metrics
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


# ======================================================================
# cases 1-4
# ======================================================================
def _case_columns(case_id: int) -> list[dict]:
    """Figure layout of each case (yellow = actual mask, red = dilated mask)."""
    a, ad = ["actual"], ["actual", "dilated"]
    if case_id == 1:
        return [{"title": "real", "show": "real", "masks": a},
                {"title": "generated", "show": "gen", "masks": a},
                {"title": "|generated - real|", "diff": ("gen", "real"), "masks": a}]
    if case_id == 2:
        return [{"title": "real", "show": "real", "masks": ad},
                {"title": "generated", "show": "gen", "masks": ["dilated"]},
                {"title": "|generated - real| (rim expected)", "diff": ("gen", "real"), "masks": ad}]
    masks = a if case_id == 3 else ad
    return [{"title": "real", "show": "real", "masks": masks},
            {"title": "pseudo-healthy", "show": "pseudo", "masks": masks},
            {"title": "generated", "show": "gen", "masks": masks},
            {"title": "|generated - real|", "diff": ("gen", "real"), "masks": masks},
            {"title": "|generated - pseudo-healthy|", "diff": ("gen", "pseudo"), "masks": masks}]


def _report_pseudo_healthy(cases: list[dict], out_path: Path) -> pd.DataFrame:
    """Print + save how the pseudo-healthy patches were made (check before trusting cases 3 and 4)."""
    rows = []
    for c in cases:
        rows.append({"patch_id": c["patch_id"], "actual_voxels": int(c["actual"].sum()),
                     "dilated_voxels": int(c["dilated"].sum()), "parent_vessel_voxels_kept": c["parent_voxels"],
                     "fill_hu": c["tissue_hu"], "noise_std_hu": c["tissue_std_hu"],
                     "still_bright_in_dilated": c["left_bright"]})
        _say(f"   {c['patch_id']}: actual {rows[-1]['actual_voxels']} -> dilated {rows[-1]['dilated_voxels']} voxels | "
             f"parent vessel kept {c['parent_voxels']} | fill {c['tissue_hu']:.0f} +- {c['tissue_std_hu']:.0f} HU | "
             f"still bright {c['left_bright']:.2f}")
    _say("   'still bright' should be close to 0 (aneurysm fully removed).")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def run_case(case_id: int, cases: list[dict], diffusion, data_cfg, cfg: EvalConfig, gen_kw: dict,
             device, out_dir: Path, show: bool) -> tuple[pd.DataFrame, dict]:
    """Generate every patch for one case, save <switch>.csv / .png. Returns (table, {patch_id: generated})."""
    switch, title, bg_key, mask_key = CASES[case_id]
    thr = float(hu_to_norm(cfg.vessel_hu, data_cfg))
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None].to(device)
    hu = lambda a: norm_to_hu(a, data_cfg)
    stats, rows, gens = [], [], {}
    for k, c in enumerate(cases):
        t0 = time.time()
        gen_mask = c[mask_key]
        gen = generate(diffusion, to_t(c[bg_key]), to_t(gen_mask), c["hist"][None].to(device),
                       seed=cfg.seed + k, **gen_kw)[0, 0].cpu().numpy()
        gens[c["patch_id"]] = gen
        st = {"case": case_id, "name": title, "patch_id": c["patch_id"],
              **_patch_stats(c["real"], gen, c["actual"], data_cfg, thr),
              "gen_fill_generated_area": float((gen[gen_mask > 0] > thr).mean()),
              "real_edge_step_hu": _edge_step_hu(c["real"], c["actual"], data_cfg),
              "gen_edge_step_hu": _edge_step_hu(gen, c["actual"], data_cfg),
              "seconds": time.time() - t0}
        stats.append(st)
        rows.append({"real": hu(c["real"]), "pseudo": hu(c["pseudo"]), "gen": hu(gen),
                     "actual": c["actual"], "dilated": c["dilated"], "label": c["patch_id"]})
        _say(f"   {c['patch_id']}: HU in mask real {st['real_hu_mean']:.0f} / gen {st['gen_hu_mean']:.0f} | "
             f"fill real {st['real_fill']:.2f} / gen {st['gen_fill']:.2f} | edge step real "
             f"{st['real_edge_step_hu']:.0f} / gen {st['gen_edge_step_hu']:.0f} HU | {st['seconds']:.0f} s")
    table = pd.DataFrame(stats)
    table.to_csv(out_dir / f"{switch}.csv", index=False)
    plot_panels(rows, _case_columns(case_id), out_dir / f"{switch}.png", show,
                title=f"Case {case_id}: {title}  (yellow = actual mask, red = dilated mask; "
                      f"generated inside the {mask_key} mask)",
                value_range=(data_cfg.hu_min, data_cfg.hu_max))
    return table, gens


def _case_summary(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per case: averages over the patches."""
    cols = ["real_hu_mean", "gen_hu_mean", "real_hu_std", "gen_hu_std", "mae_hu_inside", "real_fill",
            "gen_fill", "gen_fill_generated_area", "real_edge_step_hu", "gen_edge_step_hu"]
    df = pd.concat(tables, ignore_index=True)
    return df.groupby(["case", "name"], sort=False)[cols].mean().reset_index()


# ======================================================================
# histogram control, copy check, full scan
# ======================================================================
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
            axs[k, j].contour(m[z], levels=[0.5], colors="yellow", linewidths=0.8, origin="lower")
            axs[k, j].set_title(f"{r['label']}  {name}\n{hu:.0f} HU in mask", fontsize=8); axs[k, j].axis("off")
    plt.tight_layout(); fig.savefig(out_path, dpi=110)
    if show:
        plt.show()
    plt.close(fig)


def run_histogram_control(cases: list[dict], index: pd.DataFrame, diffusion, data_cfg, cfg: EvalConfig,
                          gen_kw: dict, device, out_path: Path, show: bool) -> None:
    """Same patch + actual mask, generated with a low / medium / high contrast histogram."""
    refs = _reference_histograms(index, data_cfg)
    _say("   reference mean HU: " + ", ".join(f"{k} {v[1]:.0f}" for k, v in refs.items()))
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None].to(device)
    rows = []
    for k, c in enumerate(cases[:cfg.hist_patches]):
        m = c["actual"]
        gens = {}
        for name, (h, _hu) in refs.items():
            g = generate(diffusion, to_t(c["real"]), to_t(m), h[None].to(device),
                         seed=cfg.seed + 200 + k, **gen_kw)[0, 0].cpu().numpy()
            gens[name] = (g, float(norm_to_hu(g[m > 0].mean(), data_cfg)))
        rows.append({"real": c["real"], "real_hu": float(norm_to_hu(c["real"][m > 0].mean(), data_cfg)),
                     "mask": m, "gens": gens, "label": c["patch_id"]})
        _say(f"   {c['patch_id']}: " + ", ".join(f"{nm} -> {hu:.0f} HU" for nm, (_, hu) in gens.items()))
    _plot_hist_control(rows, out_path, show)


def run_copy_check(cases: list[dict], gens: dict, index: pd.DataFrame, data_cfg, out_path: Path) -> pd.DataFrame:
    """Similarity (NCC) of each case-1 aneurysm to the closest TRAIN patch, inside the grown mask."""
    train_ds = AneurysmPatchDataset(index, data_cfg, "train", augment=False)
    train_imgs = [(train_ds.rows.iloc[i]["patch_id"], train_ds[i]["image"][0].numpy()) for i in range(len(train_ds))]
    stats = []
    for c in cases:
        region = dilate(c["actual"], 3)
        real, gen = c["real"], gens[c["patch_id"]]
        sims_gen = [(_ncc(gen, t, region), tid) for tid, t in train_imgs]
        sims_real = [_ncc(real, t, region) for _, t in train_imgs]
        best_gen = max(sims_gen, key=lambda s: -1 if np.isnan(s[0]) else s[0])
        stats.append({"patch_id": c["patch_id"], "closest_train_patch": best_gen[1],
                      "max_ncc_generated": best_gen[0], "max_ncc_real": float(np.nanmax(sims_real)),
                      "ncc_generated_vs_real": _ncc(gen, real, region)})
    df = pd.DataFrame(stats)
    df["looks_copied"] = (df.max_ncc_generated > 0.9) & (df.max_ncc_generated > df.max_ncc_real + 0.1)
    df.to_csv(out_path, index=False)
    _say(df.round(3).to_string(index=False))
    _say("   'looks_copied' = generated is much closer to a training patch than the real aneurysm is.")
    return df


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


def run_full_scan(pid: str, cases: list[dict], gens: dict, data: dict, data_cfg, out_dir: Path, root: Path,
                  show: bool) -> Path:
    """Paste the case-1 aneurysm of `pid` back into its CT scan; save NIfTI + 3-plane figure."""
    import nibabel as nib

    index = data["index"]
    row = index[index.patch_id == pid].iloc[0]
    case = data["catalog"].set_index("case_id").loc[row.case_id]
    vol, aff, spacing = load_canonical(data_cfg.abs(case["image_path"]))
    full_mask, _, _ = load_canonical(data_cfg.abs(case["mask_path"]), is_mask=True)
    with np.load(data_cfg.abs(row.patch_path)) as f:
        stored_affine = f["affine"]
    c = next(c for c in cases if c["patch_id"] == pid)
    # dataset order (z, y, x) -> volume order (x, y, z)
    gen_hu = norm_to_hu(gens[pid], data_cfg).transpose(2, 1, 0)
    pm = c["actual"].transpose(2, 1, 0)
    syn = paste_patch_into_volume(vol, aff, gen_hu, pm, model_patch_affine(stored_affine, data_cfg.patch_margin))
    nii_dir = out_dir / "full_scan"
    nii_dir.mkdir(exist_ok=True)
    nib.save(nib.Nifti1Image(syn, aff), str(nii_dir / f"Syn_{pid}.nii.gz"))
    nib.save(nib.Nifti1Image(full_mask, aff), str(nii_dir / f"Syn_{pid}_mask.nii.gz"))
    centre = np.array([row.center_x, row.center_y, row.center_z])
    changed = int((syn != vol).sum())
    _plot_full_scan(vol, syn, full_mask, centre, spacing, f"{pid}: original vs synthetic ({changed} voxels changed)",
                    out_dir / "full_scan.png", show)
    _say(f"   saved {_show(nii_dir, root)}/Syn_{pid}.nii.gz (+ _mask) | {changed} voxels changed")
    return nii_dir


# ======================================================================
# main entry point
# ======================================================================
def run_evaluation(cfg: EvalConfig, project_root: Path | None = None, show: bool = True) -> dict:
    """Run every check switched on in `cfg`. Returns result tables and the output folder.

    show=False only saves the figures (used by scripts/run_eval.py).
    """
    root = Path(project_root or find_project_root())
    device = get_device()
    run = _load_run(cfg, root, device)
    diffusion, data_cfg, data, gen_kw = run["diffusion"], run["data_cfg"], run["data"], run["gen_kw"]
    out_dir = RunPaths(run["run_dir"]).eval_dir(cfg.checkpoint, cfg.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    results: dict = {"out_dir": out_dir, "cases": {}}
    case_ids = [i for i, (switch, *_rest) in CASES.items() if getattr(cfg, switch)]
    _say(f"Split '{cfg.split}': {len(run['items'])} of {run['n_total']} patches | cases {case_ids} | "
         f"dilated mask = actual + {cfg.dilate_voxels} voxels | outputs: {_show(out_dir, root)}/\n")

    # ---- training curves ----
    if cfg.training_curves:
        _say("Training curves")
        plot_training_curves(run["run_dir"], out_dir / "training_curves.png", show)

    cases = _prepare_cases(run["items"], cfg, data_cfg)
    if cfg.case_3_pseudo_actual or cfg.case_4_pseudo_dilated:
        _say(f"\nPseudo-healthy patches: aneurysm + {cfg.dilate_voxels}-voxel rim removed, parent vessel kept")
        results["pseudo_healthy"] = _report_pseudo_healthy(cases, out_dir / "pseudo_healthy.csv")

    # ---- cases 1-4 (fixed order) ----
    case_gens: dict[int, dict] = {}
    for case_id in case_ids:
        _say(f"\nCase {case_id}: {CASES[case_id][1]}")
        table, gens = run_case(case_id, cases, diffusion, data_cfg, cfg, gen_kw, device, out_dir, show)
        results["cases"][case_id] = table
        case_gens[case_id] = gens

    # ---- histogram control ----
    if cfg.histogram_control:
        _say("\nHistogram control: low / medium / high contrast (from training patches)")
        run_histogram_control(cases, data["index"], diffusion, data_cfg, cfg, gen_kw, device,
                              out_dir / "histogram_control.png", show)

    # ---- copy check + full scan (use case 1) ----
    if cfg.copy_check:
        _say("\nCopy check: case-1 aneurysms vs the closest TRAIN patch (inside the grown mask)")
        results["copy_check"] = run_copy_check(cases, case_gens[1], data["index"], data_cfg,
                                               out_dir / "copy_check.csv")
    if cfg.full_scan:
        pid = cfg.full_scan_patch or cases[0]["patch_id"]
        if pid not in case_gens[1]:
            raise ValueError(f"full_scan_patch {pid} was not evaluated (choose one of {list(case_gens[1])})")
        _say(f"\nFull scan: pasting the case-1 aneurysm of {pid} back into its CT scan")
        results["full_scan"] = run_full_scan(pid, cases, case_gens[1], data, data_cfg, out_dir, root, show)

    # ---- summary ----
    if results["cases"]:
        summary = _case_summary(list(results["cases"].values()))
        summary.to_csv(out_dir / "summary.csv", index=False)
        results["summary"] = summary
        _say("\nSummary (mean over patches, inside the actual mask)")
        _say(summary.round(2).to_string(index=False))
        _say("\nHow to read it:")
        _say("   1 vs 2 : effect of the dilated mask on a real background")
        _say("   1 vs 4 : 4 close to 1 -> the model does not need the real rim (works on a healthy-looking vessel)")
        _say("   3      : gen edge step much bigger than real -> only the actual mask leaves a hard edge")
    if "copy_check" in results:
        _say(f"   patches that look copied: {int(results['copy_check'].looks_copied.sum())} of "
             f"{len(results['copy_check'])}")
    _say(f"\nAll results in {_show(out_dir, root)}/")
    return results
