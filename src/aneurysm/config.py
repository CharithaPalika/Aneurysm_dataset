"""Project configuration.

Values live in YAML files in the configs/ folder (e.g. configs/data.yaml).
This module defines what each config contains and loads it.

All paths are built from one project root, so the same code runs on macOS
and Windows. Paths written to CSV files are stored *relative* to the
project root (with forward slashes), so caches made on one machine can be
copied to another.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path

import yaml


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start` until a folder containing `Dataset/` and `src/` is found."""
    start = Path(start or Path.cwd()).resolve()
    for folder in (start, *start.parents):
        if (folder / "Dataset").is_dir() and (folder / "src").is_dir():
            return folder
    raise FileNotFoundError(
        f"Could not find the project root (a folder with 'Dataset/' and 'src/') above {start}"
    )


@dataclass
class DataConfig:
    """Everything the data pipeline needs. Values come from configs/data.yaml;
    the defaults below are only used if a key is missing from the file."""

    project_root: Path

    # ---- Raw data (inside project_root) ----
    scans_subdir: str = "Dataset/CTA_scans"                      # holds internal_Tr01, internal_Tr02, ...
    masks_subdir: str = "Dataset/Annotations/internal_annotation"
    instances_csv: str = "Dataset/internal_instances.csv"
    scan_folder_glob: str = "internal_*"                         # which sub-folders of CTA_scans to read

    # ---- Outputs made by our code ----
    work_subdir: str = "work"                                    # splits, patch cache, ...

    # ---- Split ----
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 42
    official_test_as_test: bool = True   # Ts cases (when their scans exist) always go to "test"

    # ---- Preprocessing ----
    spacing_mm: float = 0.45             # isotropic target voxel size
    hu_min: float = -100.0               # intensity window for the model
    hu_max: float = 900.0
    air_hu: float = -1024.0              # padding value outside the scan
    min_component_voxels: int = 20       # smaller mask blobs are treated as noise

    # ---- Patches ----
    patch_size: int = 64                 # in-plane size given to the model (H = W)
    patch_depth: int | None = None       # number of slices (D); None = same as patch_size.
                                         # Pilot on a 4 GB GPU: 32  ->  patches of 32 x 64 x 64
    patch_margin: int = 8                # extra voxels per side stored on disk (for random shifts)
    hist_bins: int = 16                  # LeFusion histogram size

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def scans_dir(self) -> Path:
        return self.project_root / self.scans_subdir

    @property
    def masks_dir(self) -> Path:
        return self.project_root / self.masks_subdir

    @property
    def instances_path(self) -> Path:
        return self.project_root / self.instances_csv

    @property
    def work_dir(self) -> Path:
        return self.project_root / self.work_subdir

    @property
    def splits_path(self) -> Path:
        return self.work_dir / "splits" / "splits.csv"

    @property
    def patch_tag(self) -> str:
        """Name that encodes every setting that changes the stored patches,
        e.g. '64x64x32_sp0.45_hu-100to900_m8_min20'."""
        x, y, z = self.model_patch_shape
        return (f"{x}x{y}x{z}_sp{self.spacing_mm:g}_hu{self.hu_min:g}to{self.hu_max:g}"
                f"_m{self.patch_margin}_min{self.min_component_voxels}")

    @property
    def patch_dir(self) -> Path:
        """Each patch setting gets its own folder, so changing data.yaml
        (e.g. 64x64x32 -> 64x64x64) never mixes old and new patches."""
        return self.work_dir / "patches" / self.patch_tag

    @property
    def patch_index_path(self) -> Path:
        return self.patch_dir / "index.csv"

    @property
    def depth(self) -> int:
        return self.patch_depth or self.patch_size

    @property
    def model_patch_shape(self) -> tuple[int, int, int]:
        """(x, y, z) size of the patch given to the model. z = depth (axial slices)."""
        return (self.patch_size, self.patch_size, self.depth)

    @property
    def stored_patch_shape(self) -> tuple[int, int, int]:
        """(x, y, z) size saved on disk: model shape + margin on every side."""
        return tuple(s + 2 * self.patch_margin for s in self.model_patch_shape)

    # ------------------------------------------------------------------
    def rel(self, path: Path) -> str:
        """Path relative to the project root, with forward slashes (portable)."""
        return Path(path).resolve().relative_to(self.project_root.resolve()).as_posix()

    def abs(self, rel_path: str) -> Path:
        """Inverse of `rel`."""
        return self.project_root / Path(rel_path)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["project_root"] = str(self.project_root)
        return d


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_keys(values: dict, cls, source: str, skip: tuple = ()) -> None:
    """Raise if `values` has keys that are not fields of dataclass `cls` (catches typos)."""
    allowed = {f.name for f in fields(cls)} - set(skip)
    unknown = set(values) - allowed
    if unknown:
        raise KeyError(f"Unknown keys in {source}: {sorted(unknown)}. Allowed: {sorted(allowed)}")


def load_data_config(path: str | Path | None = None, project_root: str | Path | None = None,
                     overrides: dict | None = None) -> DataConfig:
    """Read configs/data.yaml (or `path`) into a DataConfig.

    overrides: optional dict applied on top of the file, e.g. {"patch_depth": 64}.
    Unknown keys raise an error, so typos are caught early.
    """
    root = Path(project_root).resolve() if project_root else find_project_root()
    path = Path(path) if path else root / "configs" / "data.yaml"
    values = read_yaml(path)
    values.update(overrides or {})
    check_keys(values, DataConfig, path.name, skip=("project_root",))
    return DataConfig(project_root=root, **values)
