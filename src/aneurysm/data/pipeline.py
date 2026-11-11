"""One call that gets the data ready: catalog -> split -> cached patches.

Used by training, evaluation and scripts/run_prepare_data.py.
Patches are cached per patch setting (see DataConfig.patch_dir), so this is
fast after the first run and re-cuts automatically when data.yaml changes.
"""
from __future__ import annotations

import pandas as pd

from ..config import DataConfig
from .catalog import build_catalog
from .patches import build_patch_index, load_patch_index
from .splits import load_or_create_splits


def prepare_data(cfg: DataConfig, n_threads: int = 2, verbose: bool = True, overwrite: bool = False) -> dict:
    """Returns {'catalog', 'splits', 'index'} (pandas DataFrames).

    overwrite=True re-cuts the patches of this setting even if they already exist.
    """
    catalog = build_catalog(cfg)
    splits = load_or_create_splits(catalog, cfg)
    index = build_patch_index(catalog, splits, cfg, overwrite=overwrite, n_threads=n_threads)

    if verbose:
        counts = index.groupby("split").agg(patches=("patch_id", "size"), cases=("case_id", "nunique"))
        print(f"Patch setting: {cfg.patch_tag}")
        print(counts.to_string())
        empty = int((index["center_voxels_in_patch"] == 0).sum()) if "center_voxels_in_patch" in index else 0
        if empty:
            print(f"Warning: {empty} patches have no centred aneurysm voxels.")
    return {"catalog": catalog, "splits": splits, "index": index}


def load_cached_data(cfg: DataConfig) -> dict | None:
    """The patches already cut for this setting, read from work/ only (Dataset/ is not needed).

    Returns {'catalog': None, 'splits', 'index'}, or None if the cache is missing or
    incomplete (then use prepare_data, which needs Dataset/).
    """
    index = load_patch_index(cfg)
    if index.empty:
        return None
    if not all(cfg.abs(p).exists() for p in index["patch_path"]):
        return None
    splits = (pd.read_csv(cfg.splits_path, dtype={"case_id": str}) if cfg.splits_path.exists()
              else index[["case_id", "split"]].drop_duplicates().reset_index(drop=True))
    return {"catalog": None, "splits": splits, "index": index}
