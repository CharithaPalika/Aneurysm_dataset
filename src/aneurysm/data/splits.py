"""Train / val / test split by patient (case).

Rules
-----
1. Only cases that have both a scan and a mask are split.
2. Official test cases (Ts...) go to "test" when `official_test_as_test` is True.
3. Training cases (Tr...) are split into train / val / test with a hash of the
   case id. The hash makes the split *stable*: when you add internal_Tr02,
   internal_Tr03, ... later, old cases keep their split and only new cases
   are added. This avoids leaking a case from train into val/test.
4. The split is saved to work/splits/splits.csv. If that file exists, its
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


def _split_for(case_id: str, official_subset: str, cfg: DataConfig) -> str:
    if cfg.official_test_as_test and official_subset == "test":
        return "test"
    u = _hash_unit(case_id, cfg.split_seed)
    if u < cfg.test_frac:
        return "test"
    if u < cfg.test_frac + cfg.val_frac:
        return "val"
    return "train"


def assign_splits(catalog: pd.DataFrame, cfg: DataConfig,
                  existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a table case_id -> split for all paired cases."""
    paired = catalog[catalog["paired"]].copy()
    old = {} if existing is None else dict(zip(existing["case_id"], existing["split"]))
    paired["split"] = [
        old.get(cid) or _split_for(cid, sub, cfg)
        for cid, sub in zip(paired["case_id"], paired["official_subset"])
    ]
    paired["is_new"] = ~paired["case_id"].isin(old)
    cols = ["case_id", "split", "official_subset", "scan_folder", "institution_id", "num_IAs", "is_new"]
    return paired[cols].reset_index(drop=True)


def load_or_create_splits(catalog: pd.DataFrame, cfg: DataConfig, save: bool = True) -> pd.DataFrame:
    """Load the saved split (if any), add new cases, and save it again."""
    existing = pd.read_csv(cfg.splits_path, dtype={"case_id": str}) if cfg.splits_path.exists() else None
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
        cfg.splits_path.parent.mkdir(parents=True, exist_ok=True)
        splits.drop(columns=["is_new"]).sort_values("case_id").to_csv(cfg.splits_path, index=False)
    return splits.sort_values("case_id").reset_index(drop=True)
