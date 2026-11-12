"""Data settings.

configs/data/data.yaml               which datasets are used + split, preprocessing and patch settings
configs/data/sources/<name>.yaml     one file per dataset: its (read-only) root folder and layout

Preprocessed data is written inside the project:  <work_subdir>/<dataset>/splits.csv
                                                  <work_subdir>/<dataset>/patches/<patch_tag>/
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .base import CONFIG_DIR, check_keys, find_project_root, read_yaml, resolve_path
from .paths import SOURCES_CONFIG_DIR

DEFAULT_DATA_CONFIG = Path(CONFIG_DIR) / "data" / "data.yaml"


@dataclass
class DataConfig:
    """Everything the data pipeline needs. Values come from configs/data/data.yaml;
    the defaults below are only used if a key is missing from the file."""

    project_root: Path

    # ---- Datasets ----
    datasets: dict = field(default_factory=lambda: {"zenodo_la": True})   # name -> used (true / false)
    sources: dict = field(default_factory=dict)  # name -> settings from configs/data/sources/<name>.yaml (filled by the loader)

    # ---- Outputs made by our code (inside project_root) ----
    work_subdir: str = "work"                                    # <work_subdir>/<dataset>/{splits.csv, patches/}

    # ---- Split ----
    val_frac: float = 0.15               # share of each dataset's training pool that goes to "val"
    split_seed: int = 42

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
    def enabled_datasets(self) -> list[str]:
        return [name for name, used in self.datasets.items() if used]

    @property
    def work_dir(self) -> Path:
        return self.project_root / self.work_subdir

    def dataset_work_dir(self, dataset: str) -> Path:
        return self.work_dir / dataset

    def splits_path(self, dataset: str) -> Path:
        return self.dataset_work_dir(dataset) / "splits.csv"

    @property
    def patch_tag(self) -> str:
        """Name that encodes every setting that changes the stored patches,
        e.g. '64x64x32_sp0.45_hu-100to900_m8_min20'."""
        x, y, z = self.model_patch_shape
        return (f"{x}x{y}x{z}_sp{self.spacing_mm:g}_hu{self.hu_min:g}to{self.hu_max:g}"
                f"_m{self.patch_margin}_min{self.min_component_voxels}")

    def patch_dir(self, dataset: str) -> Path:
        """Each dataset and each patch setting gets its own folder, so changing data.yaml
        (e.g. 64x64x32 -> 64x64x64) never mixes old and new patches."""
        return self.dataset_work_dir(dataset) / "patches" / self.patch_tag

    def patch_index_path(self, dataset: str) -> Path:
        return self.patch_dir(dataset) / "index.csv"

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
        """Path relative to the project root, with forward slashes (portable).
        A path outside the project (e.g. data on another drive) stays absolute."""
        p = Path(path).resolve()
        try:
            return p.relative_to(self.project_root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    def with_work_dir(self, work_dir: str | Path | None) -> "DataConfig":
        """Copy that reads / writes the preprocessed data in `work_dir` (absolute, or relative to the project)."""
        if not work_dir:
            return self
        return replace(self, work_subdir=str(work_dir))

    def with_sources(self, sources: dict) -> "DataConfig":
        """Copy with dataset settings (e.g. this machine's dataset roots)."""
        return replace(self, sources=sources)

    def abs(self, rel_path: str) -> Path:
        """Inverse of `rel`."""
        return self.project_root / Path(rel_path)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["project_root"] = str(self.project_root)
        return d


def _split_overrides(overrides: dict | None) -> tuple[dict, dict]:
    """{'patch_depth': 24, 'sources.zenodo_la.root': 'D:/x'} -> plain values, {'zenodo_la': {'root': 'D:/x'}}"""
    plain, per_source = {}, {}
    for key, value in (overrides or {}).items():
        if key == "sources" and isinstance(value, dict):
            for name, vals in value.items():
                per_source.setdefault(name, {}).update(vals)
        elif key.startswith("sources."):
            parts = key.split(".")
            if len(parts) != 3:
                raise KeyError(f"Use sources.<dataset>.<key>=value, got '{key}'")
            per_source.setdefault(parts[1], {})[parts[2]] = value
        else:
            plain[key] = value
    return plain, per_source


def load_source_settings(datasets: list[str], project_root: Path, overrides: dict | None = None) -> dict:
    """Read configs/data/sources/<name>.yaml for each dataset (+ overrides). Unknown keys raise an error."""
    from ..data.sources import get_source_class          # local import: data -> config dependency

    out = {}
    for name in datasets:
        cls = get_source_class(name)
        path = Path(project_root) / SOURCES_CONFIG_DIR / f"{name}.yaml"
        values = {**cls.DEFAULTS, **(read_yaml(path) if path.exists() else {}), **(overrides or {}).get(name, {})}
        unknown = set(values) - set(cls.DEFAULTS)
        if unknown:
            raise KeyError(f"Unknown keys for dataset '{name}' ({path.name}): {sorted(unknown)}. "
                           f"Allowed: {sorted(cls.DEFAULTS)}")
        out[name] = values
    return out


def load_data_config(path: str | Path | None = None, project_root: str | Path | None = None,
                     overrides: dict | None = None) -> DataConfig:
    """Read configs/data/data.yaml (or `path`) + the files of the used datasets into a DataConfig.

    overrides: applied on top of the files, e.g. {"patch_depth": 64} or
               {"sources.zenodo_la.root": "D:/data/Data"}.
    Unknown keys raise an error, so typos are caught early.
    """
    root = Path(project_root).resolve() if project_root else find_project_root()
    path = resolve_path(path, root, DEFAULT_DATA_CONFIG)
    plain, per_source = _split_overrides(overrides)
    values = read_yaml(path)
    values.update(plain)
    check_keys(values, DataConfig, path.name, skip=("project_root", "sources"))
    cfg = DataConfig(project_root=root, **values)
    unknown = set(per_source) - set(cfg.enabled_datasets)
    if unknown:
        raise KeyError(f"Settings given for datasets that are not used: {sorted(unknown)}")
    return cfg.with_sources(load_source_settings(cfg.enabled_datasets, root, per_source))
