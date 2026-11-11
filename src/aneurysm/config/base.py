"""Shared config helpers: find the project folder, read YAML, catch typos, resolve paths.

All paths are built from one project root, so the same code runs on macOS and
Windows. Paths written to CSV files are stored relative to the project root
(forward slashes), so caches made on one machine can be copied to another.
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from .paths import CONFIG_DIR, ROOT_MARKERS  # noqa: F401  (re-exported)


def find_project_root(start: Path | None = None) -> Path:
    """The folder that contains src/aneurysm/ and configs/.

    Searches upwards from `start`; without `start`, from the current folder and then
    from this file's location (so it also works when called from outside the project).
    """
    starts = [Path(start)] if start else [Path.cwd(), Path(__file__)]
    for s in starts:
        s = s.resolve()
        for folder in (s, *s.parents):
            if all((folder / m).is_dir() for m in ROOT_MARKERS):
                return folder
    raise FileNotFoundError(f"Could not find the project root (a folder with src/aneurysm/ and configs/) "
                            f"at or above {', '.join(str(x) for x in starts)}")


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_keys(values: dict, cls, source: str, skip: tuple = ()) -> None:
    """Raise if `values` has keys that are not fields of dataclass `cls` (catches typos)."""
    allowed = {f.name for f in fields(cls)} - set(skip)
    unknown = set(values) - allowed
    if unknown:
        raise KeyError(f"Unknown keys in {source}: {sorted(unknown)}. Allowed: {sorted(allowed)}")


def resolve_path(path: str | Path | None, project_root: Path, default: str | Path | None = None) -> Path:
    """Turn a user-given path into an absolute one.

    - None            -> <project_root>/<default>
    - absolute path   -> unchanged
    - relative path   -> relative to the current folder if it exists there,
                         otherwise relative to the project root
    So `configs/train/x.yaml` works from the project folder and from anywhere else.
    """
    if path is None:
        if default is None:
            raise ValueError("No path given and no default")
        return (Path(project_root) / default).resolve()
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (Path(project_root) / p).resolve()
