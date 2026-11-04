"""Find every CTA scan and mask, and join them with the case CSV.

Scans can be spread over several folders:
    Dataset/CTA_scans/internal_Tr01/Tr0001.nii.gz
    Dataset/CTA_scans/internal_Tr02/Tr0100.nii.gz
    ...
All folders matching `cfg.scan_folder_glob` are read, so new folders are
picked up automatically.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import DataConfig

CASE_ID_RE = re.compile(r"^(Tr|Ts)\d{4}$")
NII_SUFFIX = ".nii.gz"


def _case_id(path: Path) -> str | None:
    name = path.name
    if not name.endswith(NII_SUFFIX):
        return None
    cid = name[: -len(NII_SUFFIX)]
    return cid if CASE_ID_RE.match(cid) else None


def find_scan_folders(cfg: DataConfig) -> list[Path]:
    """All scan folders, e.g. [internal_Tr01, internal_Tr02, ...], sorted by name."""
    if not cfg.scans_dir.is_dir():
        raise FileNotFoundError(f"Scan folder not found: {cfg.scans_dir}")
    return sorted(p for p in cfg.scans_dir.glob(cfg.scan_folder_glob) if p.is_dir())


def _index_files(folders: list[Path]) -> dict[str, Path]:
    """case_id -> file path. Raises if the same case appears twice."""
    found: dict[str, Path] = {}
    for folder in folders:
        for path in sorted(folder.glob(f"*{NII_SUFFIX}")):
            cid = _case_id(path)
            if cid is None:
                continue
            if cid in found:
                raise ValueError(f"Case {cid} found twice:\n  {found[cid]}\n  {path}")
            found[cid] = path
    return found


def build_catalog(cfg: DataConfig) -> pd.DataFrame:
    """One row per case with paths (relative to project root) and CSV metadata.

    Columns: case_id, official_subset, scan_folder, image_path, mask_path,
             has_image, has_mask, paired, institution_id, age, gender,
             is_ruptured, num_IAs
    """
    scan_folders = find_scan_folders(cfg)
    images = _index_files(scan_folders)
    masks = _index_files([cfg.masks_dir])

    meta = pd.read_csv(cfg.instances_path, dtype={"instance_id": str})
    meta = meta.rename(columns={"instance_id": "case_id", "subset": "official_subset"})

    all_ids = sorted(set(meta["case_id"]) | set(images) | set(masks))
    df = pd.DataFrame({"case_id": all_ids})
    df["scan_folder"] = df["case_id"].map(lambda c: images[c].parent.name if c in images else "")
    df["image_path"] = df["case_id"].map(lambda c: cfg.rel(images[c]) if c in images else "")
    df["mask_path"] = df["case_id"].map(lambda c: cfg.rel(masks[c]) if c in masks else "")
    df["has_image"] = df["image_path"] != ""
    df["has_mask"] = df["mask_path"] != ""
    df["paired"] = df["has_image"] & df["has_mask"]

    df = df.merge(meta, on="case_id", how="left")
    df["official_subset"] = df["official_subset"].fillna(
        df["case_id"].str[:2].map({"Tr": "train", "Ts": "test"})
    )
    cols = ["case_id", "official_subset", "scan_folder", "image_path", "mask_path",
            "has_image", "has_mask", "paired",
            "institution_id", "age", "gender", "is_ruptured", "num_IAs"]
    return df[cols].sort_values("case_id").reset_index(drop=True)


def summarize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per official subset: total, with image, with mask, paired, normal (num_IAs == 0)."""
    g = df.groupby("official_subset")
    return pd.DataFrame({
        "cases": g.size(),
        "with_image": g["has_image"].sum(),
        "with_mask": g["has_mask"].sum(),
        "paired": g["paired"].sum(),
        "normal_cases": g["num_IAs"].apply(lambda s: int((s == 0).sum())),
    })
