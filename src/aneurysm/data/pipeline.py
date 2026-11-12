"""Get the data ready: case table -> split -> cached patches, for every used dataset.

Used by training, evaluation and scripts/run_preprocess.py.
Each dataset is preprocessed into its own folder <work>/<dataset>/ (splits.csv +
patches/<patch_tag>/), so datasets never mix and each can have its own reader.
The tables returned here combine all used datasets (column `dataset`).
"""
from __future__ import annotations

import pandas as pd

from ..config import DataConfig
from .catalog import build_catalog
from .patches import build_patch_index, load_patch_index
from .splits import load_or_create_splits


def make_splits(catalog: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Split of every used dataset (each saved to <work>/<dataset>/splits.csv)."""
    return pd.concat([load_or_create_splits(catalog, cfg, name) for name in cfg.enabled_datasets],
                     ignore_index=True)


def make_patches(catalog: pd.DataFrame, splits: pd.DataFrame, cfg: DataConfig,
                 overwrite: bool = False, n_threads: int = 2) -> pd.DataFrame:
    """Cut (or reuse) the patches of every used dataset; returns the combined patch index."""
    return pd.concat([build_patch_index(catalog, splits, cfg, name, overwrite=overwrite, n_threads=n_threads)
                      for name in cfg.enabled_datasets], ignore_index=True)


def _print_counts(index: pd.DataFrame, cfg: DataConfig, note: str = "") -> None:
    counts = index.groupby(["dataset", "split"]).agg(patches=("patch_id", "size"), cases=("case_id", "nunique"))
    print(f"Patch setting: {cfg.patch_tag}{note}")
    print(counts.to_string())
    empty = int((index["center_voxels_in_patch"] == 0).sum()) if "center_voxels_in_patch" in index else 0
    if empty:
        print(f"Warning: {empty} patches have no centred aneurysm voxels.")


def prepare_data(cfg: DataConfig, n_threads: int = 2, verbose: bool = True, overwrite: bool = False) -> dict:
    """Preprocess (needs the raw datasets). Returns {'catalog', 'splits', 'index'} (pandas DataFrames).

    Patches already cut are reused; overwrite=True re-cuts them.
    """
    catalog = build_catalog(cfg)
    splits = make_splits(catalog, cfg)
    index = make_patches(catalog, splits, cfg, overwrite=overwrite, n_threads=n_threads)
    if verbose:
        _print_counts(index, cfg)
    return {"catalog": catalog, "splits": splits, "index": index}


def load_cached_data(cfg: DataConfig) -> dict | None:
    """The preprocessed data of every used dataset, from the work dir only (raw data not needed).

    Returns {'catalog': None, 'splits', 'index'}, or None if any dataset's cache is missing
    or incomplete (then use prepare_data, which needs the raw data).
    """
    indexes, splits = [], []
    for name in cfg.enabled_datasets:
        index = load_patch_index(cfg, name)
        if index.empty or not all(cfg.abs(p).exists() for p in index["patch_path"]):
            return None
        indexes.append(index)
        path = cfg.splits_path(name)
        splits.append(pd.read_csv(path, dtype={"case_id": str}) if path.exists()
                      else index[["dataset", "case_id", "split"]].drop_duplicates())
    return {"catalog": None, "splits": pd.concat(splits, ignore_index=True),
            "index": pd.concat(indexes, ignore_index=True)}


def get_data(cfg: DataConfig, preprocess: bool, n_threads: int = 2, verbose: bool = True) -> dict:
    """Data for training / evaluation.

    preprocess=True : prepare_data (needs the raw datasets; patches already cut are reused)
    preprocess=False: only the preprocessed data in the work dir (raw datasets not needed)
    """
    if preprocess:
        return prepare_data(cfg, n_threads=n_threads, verbose=verbose)
    data = load_cached_data(cfg)
    if data is None:
        folders = ", ".join(cfg.rel(cfg.patch_dir(n)) for n in cfg.enabled_datasets)
        raise FileNotFoundError(
            f"No complete preprocessed data for patch setting '{cfg.patch_tag}' (expected in: {folders}). "
            f"Run  python scripts/run_preprocess.py  with the same data config, point work_dir to the "
            f"right folder, or set preprocess: true.")
    if verbose:
        _print_counts(data["index"], cfg, " (preprocessed, from the work dir)")
    return data
