"""Case table of all used datasets (one row per case).

Each dataset has its own reader in data/sources/ that knows its folder layout.
This module only combines the readers of the datasets switched on in
configs/data/data.yaml (`datasets: {zenodo_la: true, ...}`).
Scan / mask paths in the table are relative to each dataset's root, so the same
table works on every machine (only the root in configs/data/sources/<name>.yaml differs).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import DataConfig
from .sources import get_source


def build_catalog(cfg: DataConfig) -> pd.DataFrame:
    """Case table of every used dataset (columns: see data/sources/base.py)."""
    if not cfg.enabled_datasets:
        raise ValueError("No dataset is switched on (datasets: in configs/data/data.yaml)")
    tables = [get_source(name, cfg).build_catalog() for name in cfg.enabled_datasets]
    return pd.concat(tables, ignore_index=True)


def resolve_case_path(cfg: DataConfig, row, column: str = "image_path") -> Path:
    """Absolute path of a scan / mask of a case row (needs that dataset's root on this machine)."""
    return get_source(row["dataset"], cfg).resolve(row[column])


def summarize_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per dataset and pool: total, with image, with mask, paired, normal (num_IAs == 0)."""
    g = df.groupby(["dataset", "pool"])
    out = pd.DataFrame({
        "cases": g.size(),
        "with_image": g["has_image"].sum(),
        "with_mask": g["has_mask"].sum(),
        "paired": g["paired"].sum(),
    })
    if "num_IAs" in df:
        out["normal_cases"] = g["num_IAs"].apply(lambda s: int((s == 0).sum()))
    return out
