"""One call that gets the data ready: catalog -> split -> cached patches.

Used by the training notebook so it does not repeat the dataset notebook.
Patches are cached per patch setting (see DataConfig.patch_dir), so this is
fast after the first run and re-cuts automatically when data.yaml changes.
"""
from __future__ import annotations

import pandas as pd

from ..config import DataConfig
from .catalog import build_catalog
from .patches import build_patch_index
from .splits import load_or_create_splits


def prepare_data(cfg: DataConfig, n_threads: int = 2, verbose: bool = True) -> dict:
    """Returns {'catalog', 'splits', 'index'} (pandas DataFrames)."""
    catalog = build_catalog(cfg)
    splits = load_or_create_splits(catalog, cfg)
    index = build_patch_index(catalog, splits, cfg, overwrite=False, n_threads=n_threads)

    if verbose:
        counts = index.groupby("split").agg(patches=("patch_id", "size"), cases=("case_id", "nunique"))
        print(f"Patch setting: {cfg.patch_tag}")
        print(counts.to_string())
        empty = int((index["center_voxels_in_patch"] == 0).sum()) if "center_voxels_in_patch" in index else 0
        if empty:
            print(f"Warning: {empty} patches have no centred aneurysm voxels.")
    return {"catalog": catalog, "splits": splits, "index": index}
