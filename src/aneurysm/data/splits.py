"""Train / val / test split by patient (case), per dataset.

Rules
-----
1. Only cases that have both a scan and a mask are split.
2. Cases in the dataset's "test" pool (e.g. internal_Ts*) always go to "test".
3. Cases in the "train" pool (e.g. internal_Tr*) are split into train / val with a
   hash of the case id. The hash makes the split *stable*: adding scan folders later
   keeps old cases where they were and only adds new ones (no leaks).
4. The split is saved to <work>/<dataset>/splits.csv. If that file exists, its
   assignments are kept and only new cases are added.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from ..config import DataConfig

SPLITS = ("train", "val", "test")


def _hash_unit(case_id: str, seed: int) -> float:
    """Deterministic number in [0, 1) for a case id."""
    h = hashlib.sha1(f"{seed}:{case_id}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


def _split_for(case_id: str, pool: str, cfg: DataConfig) -> str:
    if pool == "test":
        return "test"
    return "val" if _hash_unit(case_id, cfg.split_seed) < cfg.val_frac else "train"


def assign_splits(catalog: pd.DataFrame, cfg: DataConfig,
                  existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a table case_id -> split for all paired cases."""
    paired = catalog[catalog["paired"]].copy()
    old = {} if existing is None else dict(zip(existing["case_id"], existing["split"]))
    paired["split"] = [old.get(cid) or _split_for(cid, pool, cfg)
                       for cid, pool in zip(paired["case_id"], paired["pool"])]
    paired["is_new"] = ~paired["case_id"].isin(old)
    cols = ["dataset", "case_id", "source_case_id", "split", "pool", "scan_folder"]
    cols += [c for c in ("institution_id", "num_IAs") if c in paired]          # metadata, if the dataset has it
    return paired[cols + ["is_new"]].reset_index(drop=True)


def load_or_create_splits(catalog: pd.DataFrame, cfg: DataConfig, dataset: str,
                          save: bool = True) -> pd.DataFrame:
    """Split of one dataset: load the saved one (if any), add new cases, save it again."""
    catalog = catalog[catalog["dataset"] == dataset]
    path = cfg.splits_path(dataset)
    existing = pd.read_csv(path, dtype={"case_id": str}) if path.exists() else None
    splits = assign_splits(catalog, cfg, existing)

    if existing is not None:
        missing = set(existing["case_id"]) - set(splits["case_id"])
        if missing:
            print(f"Note: {len(missing)} cases in the saved split no longer have scan+mask "
                  f"(e.g. {sorted(missing)[:3]}). They are kept in the file but not used.")
            keep = existing[existing["case_id"].isin(missing)].assign(is_new=False)
            splits = pd.concat([splits, keep[splits.columns.intersection(keep.columns)]],
                               ignore_index=True)

    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        splits.drop(columns=["is_new"]).sort_values("case_id").to_csv(path, index=False)
    return splits.sort_values("case_id").reset_index(drop=True)
