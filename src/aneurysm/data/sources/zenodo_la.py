"""Reader for the intracranial aneurysm CTA dataset from Zenodo (Bo et al., Patterns 2021).

Layout under `root` (the Data/ folder, read-only):
    internal_Tr01/ ... internal_Tr12/    training scans   Tr0001.nii.gz ...   -> pool "train" (train / val)
    internal_Ts01/, internal_Ts02/       test scans       Ts0001.nii.gz ...   -> pool "test"
    internal_annotation/                 masks            Tr0001.nii.gz ... Ts0152.nii.gz
    internal_instances.csv               subset, instance_id, institution_id, age, gender, is_ruptured, num_IAs
    external_*                           not used (yet)

Only `.nii.gz` files are read (every case also exists as `.nii`; those are ignored).
Other files (e.g. job logs) are ignored.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .base import DatasetSource

CASE_ID_RE = re.compile(r"^(Tr|Ts)\d{4}$")


class ZenodoLA(DatasetSource):
    NAME = "zenodo_la"
    DEFAULTS = {
        "root": None,                            # path up to .../Data/  (read-only)
        "train_folders": "internal_Tr*",         # scans -> pool "train" (split into train / val)
        "test_folders": "internal_Ts*",          # scans -> pool "test"
        "masks_folder": "internal_annotation",
        "instances_csv": "internal_instances.csv",
        "file_suffix": ".nii.gz",
    }

    # -- helpers ---------------------------------------------------------------
    def _case_id(self, path: Path) -> str | None:
        suffix = self.settings["file_suffix"]
        if not path.name.endswith(suffix):
            return None
        cid = path.name[: -len(suffix)]
        return cid if CASE_ID_RE.match(cid) else None

    def _index(self, folders: list[Path]) -> dict[str, Path]:
        """case id -> file. Raises if a case appears twice."""
        found: dict[str, Path] = {}
        for folder in folders:
            for path in sorted(folder.glob(f"*{self.settings['file_suffix']}")):
                cid = self._case_id(path)
                if cid is None:
                    continue
                if cid in found:
                    raise ValueError(f"Case {cid} found twice:\n  {found[cid]}\n  {path}")
                found[cid] = path
        return found

    def _folders(self, pattern: str) -> list[Path]:
        return sorted(p for p in self.root.glob(pattern) if p.is_dir())

    # -- main ------------------------------------------------------------------
    def build_catalog(self) -> pd.DataFrame:
        root = self.root
        train_imgs = self._index(self._folders(self.settings["train_folders"]))
        test_imgs = self._index(self._folders(self.settings["test_folders"]))
        both = set(train_imgs) & set(test_imgs)
        if both:
            raise ValueError(f"Cases in both train and test folders: {sorted(both)[:5]}")
        images = {**train_imgs, **test_imgs}
        masks_dir = root / self.settings["masks_folder"]
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Mask folder not found: {masks_dir}")
        masks = self._index([masks_dir])

        meta = pd.read_csv(root / self.settings["instances_csv"], dtype={"instance_id": str})
        meta = meta.rename(columns={"instance_id": "source_case_id", "subset": "official_subset"})

        ids = sorted(set(meta["source_case_id"]) | set(images) | set(masks))
        df = pd.DataFrame({"source_case_id": ids})
        df["pool"] = ["test" if (c in test_imgs or (c not in train_imgs and c.startswith("Ts"))) else "train"
                      for c in ids]
        df["scan_folder"] = [images[c].parent.name if c in images else "" for c in ids]
        df["image_path"] = [self.rel(images[c]) if c in images else "" for c in ids]
        df["mask_path"] = [self.rel(masks[c]) if c in masks else "" for c in ids]
        df = df.merge(meta, on="source_case_id", how="left")
        return self.finish_catalog(df)
