"""What every dataset reader must provide.

A dataset reader turns one raw dataset (read-only, anywhere on disk) into a case table
with the same columns for every dataset. Everything after that (split, patches,
training, evaluation) only uses this table, so adding a dataset = one new reader +
one config file (configs/data/sources/<name>.yaml) + `<name>: true` in data.yaml.

Case table columns (CATALOG_COLUMNS):
    dataset         reader name, e.g. "zenodo_la"
    case_id         unique over all datasets: "<dataset>_<source_case_id>"
    source_case_id  id inside the dataset, e.g. "Tr0001"
    pool            "train" (split into train / val) or "test" (always test)
    scan_folder     folder of the scan inside the dataset root (information)
    image_path      scan path, relative to the dataset root ("" = no scan)
    mask_path       mask path, relative to the dataset root ("" = no mask)
    has_image, has_mask, paired
Extra columns (metadata such as age or num_IAs) are allowed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CATALOG_COLUMNS = ["dataset", "case_id", "source_case_id", "pool", "scan_folder",
                   "image_path", "mask_path", "has_image", "has_mask", "paired"]
POOLS = ("train", "test")


class DatasetSource:
    """Base class. Subclasses set NAME and DEFAULTS and implement build_catalog()."""

    NAME: str = ""
    DEFAULTS: dict = {"root": None}       # allowed keys of configs/data/sources/<NAME>.yaml

    def __init__(self, settings: dict):
        self.settings = {**self.DEFAULTS, **(settings or {})}

    # -- paths ---------------------------------------------------------------
    @property
    def root(self) -> Path:
        """The dataset folder (read-only). Raises a clear error if it is not set or missing."""
        root = self.settings.get("root")
        if not root:
            raise FileNotFoundError(f"Dataset '{self.NAME}': set `root` in configs/data/sources/{self.NAME}.yaml "
                                    f"(or override sources.{self.NAME}.root=PATH: --set in run_preprocess.py, --data-set in run_train.py)")
        root = Path(root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset '{self.NAME}': root folder not found: {root}")
        return root

    def rel(self, path: Path) -> str:
        """Path inside the dataset root, forward slashes (same on every machine)."""
        return Path(path).relative_to(self.root).as_posix()

    def resolve(self, rel_path: str) -> Path:
        """Inverse of rel(): dataset-relative path -> absolute path on this machine."""
        return self.root / rel_path

    # -- to implement ----------------------------------------------------------
    def build_catalog(self) -> pd.DataFrame:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------
    def finish_catalog(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the common columns, check them, put them first."""
        df = df.copy()
        df["dataset"] = self.NAME
        df["case_id"] = self.NAME + "_" + df["source_case_id"].astype(str)
        df["has_image"] = df["image_path"] != ""
        df["has_mask"] = df["mask_path"] != ""
        df["paired"] = df["has_image"] & df["has_mask"]
        bad = set(df["pool"]) - set(POOLS)
        if bad:
            raise ValueError(f"Dataset '{self.NAME}': pool must be one of {POOLS}, got {sorted(bad)}")
        extra = [c for c in df.columns if c not in CATALOG_COLUMNS]
        return df[CATALOG_COLUMNS + extra].sort_values("case_id").reset_index(drop=True)
