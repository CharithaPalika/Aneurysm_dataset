"""Cut aneurysm patches once and cache them on disk.

Loading a full CTA scan takes seconds (a few hundred MB, gzip). Training
reads each patch thousands of times, so we cut every aneurysm patch once
and save it as a small .npz file:

    work/patches/<patch_tag>/<split>/<case_id>_a<k>.npz   (k = aneurysm number in the case)
    work/patches/<patch_tag>/index.csv                    (one row per patch)

<patch_tag> encodes the patch settings (e.g. 64x64x32_sp0.45_hu-100to900_m8_min20),
so each setting has its own folder and changing configs/data.yaml never mixes them.

Each stored patch is the model patch + `patch_margin` voxels on every side, at
cfg.spacing_mm, centred on one aneurysm, in Hounsfield units (int16). The Dataset
crops the model patch from it (random shift for training, centre for val/test).

Each .npz holds:
    image       : CT values (HU, int16)
    mask        : all aneurysm voxels in the patch (the label)
    center_mask : only the aneurysm the patch is centred on (LeFusion loss + histogram)
    labels      : 0 = background, 1..K = aneurysm number in the case
    affine      : patch voxel -> world (mm), used to paste results back into the scan

Running again only processes cases that are not in the index yet
(e.g. after adding internal_Tr02). overwrite=True re-cuts everything.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DataConfig
from .volume import extract_patch, find_components, load_canonical

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_):
        return x

INDEX_COLUMNS = [
    "patch_id", "case_id", "split", "aneurysm_idx", "patch_path",
    "component_voxels", "volume_mm3", "diameter_mm", "n_aneurysms_in_case",
    "center_x", "center_y", "center_z",       # native voxel coords (RAS) of the patch centre
    "mask_voxels_in_patch", "center_voxels_in_patch",
]


def _process_case(case: pd.Series, cfg: DataConfig) -> list[dict]:
    image, affine, spacing = load_canonical(cfg.abs(case["image_path"]))
    mask, mask_affine, _ = load_canonical(cfg.abs(case["mask_path"]), is_mask=True)
    if image.shape != mask.shape or not np.allclose(affine, mask_affine, atol=1e-3):
        raise ValueError(f"{case['case_id']}: scan and mask do not line up "
                         f"({image.shape} vs {mask.shape})")

    comps, labels = find_components(mask, spacing, cfg.min_component_voxels)
    out_dir = cfg.patch_dir / case["split"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, comp in enumerate(comps):
        img_p, lab_p, aff_p = extract_patch(
            image, labels, affine, spacing, comp.centroid_vox,
            size=cfg.stored_patch_shape, out_spacing=cfg.spacing_mm, pad_value=cfg.air_hu,
        )
        msk_p = (lab_p > 0).astype(np.uint8)
        ctr_p = (lab_p == comp.label).astype(np.uint8)
        patch_id = f"{case['case_id']}_a{k:02d}"
        path = out_dir / f"{patch_id}.npz"
        np.savez_compressed(
            path, image=img_p, mask=msk_p, center_mask=ctr_p, labels=lab_p, affine=aff_p,
            spacing_mm=np.float32(cfg.spacing_mm), case_id=case["case_id"],
        )
        rows.append({
            "patch_id": patch_id,
            "case_id": case["case_id"],
            "split": case["split"],
            "aneurysm_idx": k,
            "patch_path": cfg.rel(path),
            "component_voxels": comp.voxels,
            "volume_mm3": round(comp.volume_mm3, 2),
            "diameter_mm": round(comp.diameter_mm, 2),
            "n_aneurysms_in_case": len(comps),
            "center_x": round(float(comp.centroid_vox[0]), 2),
            "center_y": round(float(comp.centroid_vox[1]), 2),
            "center_z": round(float(comp.centroid_vox[2]), 2),
            "mask_voxels_in_patch": int(msk_p.sum()),
            "center_voxels_in_patch": int(ctr_p.sum()),
        })
    return rows


def load_patch_index(cfg: DataConfig) -> pd.DataFrame:
    if not cfg.patch_index_path.exists():
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.read_csv(cfg.patch_index_path, dtype={"case_id": str, "patch_id": str})


def build_patch_index(catalog: pd.DataFrame, splits: pd.DataFrame, cfg: DataConfig,
                      overwrite: bool = False, n_threads: int = 2) -> pd.DataFrame:
    """Extract patches for all split cases that are not cached yet.

    n_threads: cases processed in parallel. Each thread holds one full scan in
    memory (up to ~0.5 GB), so keep this small (1-4).
    """
    cases = splits.merge(catalog[["case_id", "image_path", "mask_path"]], on="case_id")
    cases = cases[(cases["image_path"] != "") & (cases["mask_path"] != "")]

    index = pd.DataFrame(columns=INDEX_COLUMNS) if overwrite else load_patch_index(cfg)
    done = set(index["case_id"])
    todo = cases[~cases["case_id"].isin(done)]

    # a case whose split changed would sit in the wrong folder: redo it
    if not index.empty:
        moved = index.merge(splits[["case_id", "split"]], on="case_id", suffixes=("", "_now"))
        moved = moved[moved["split"] != moved["split_now"]]["case_id"].unique()
        if len(moved):
            print(f"{len(moved)} cases changed split; re-extracting them.")
            index = index[~index["case_id"].isin(moved)]
            todo = pd.concat([todo, cases[cases["case_id"].isin(moved)]])

    print(f"Cached cases: {len(done)}  |  to process: {len(todo)}")
    new_rows: list[dict] = []
    failed: list[tuple[str, str]] = []
    if len(todo):
        def run(row):
            try:
                return _process_case(row, cfg), None
            except Exception as exc:  # keep going; report at the end
                return [], (row["case_id"], str(exc))

        rows_iter = (r for _, r in todo.iterrows())
        with ThreadPoolExecutor(max_workers=max(1, n_threads)) as pool:
            for rows, err in tqdm(pool.map(run, rows_iter), total=len(todo), desc="extract patches"):
                new_rows.extend(rows)
                if err:
                    failed.append(err)

    index = pd.concat([index, pd.DataFrame(new_rows, columns=INDEX_COLUMNS)], ignore_index=True)
    index = index.sort_values("patch_id").reset_index(drop=True)
    cfg.patch_index_path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(cfg.patch_index_path, index=False)

    if failed:
        print(f"{len(failed)} cases failed:")
        for cid, msg in failed:
            print(f"  {cid}: {msg}")
    return index
