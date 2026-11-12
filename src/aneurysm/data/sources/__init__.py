"""Dataset readers. To add a dataset: write a DatasetSource subclass, register it below,
add configs/data/sources/<name>.yaml and set `<name>: true` under `datasets` in data.yaml."""
from __future__ import annotations

from .base import CATALOG_COLUMNS, DatasetSource
from .zenodo_la import ZenodoLA

SOURCES: dict[str, type[DatasetSource]] = {
    ZenodoLA.NAME: ZenodoLA,
    # "kaggle_xyz": KaggleXYZ,
}


def get_source_class(name: str) -> type[DatasetSource]:
    if name not in SOURCES:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(SOURCES)} (see src/aneurysm/data/sources/)")
    return SOURCES[name]


def get_source(name: str, data_cfg) -> DatasetSource:
    """Reader for `name` with the settings loaded into data_cfg.sources."""
    return get_source_class(name)(data_cfg.sources.get(name, {}))


__all__ = ["DatasetSource", "CATALOG_COLUMNS", "SOURCES", "get_source_class", "get_source", "ZenodoLA"]
