"""Data pipeline: find files -> split -> extract patches -> torch Dataset/DataLoader."""
from .catalog import build_catalog, find_scan_folders
from .splits import assign_splits, load_or_create_splits
from .patches import build_patch_index, load_patch_index
from .datasets import AneurysmPatchDataset, make_dataloaders
from .pipeline import prepare_data

__all__ = [
    "build_catalog",
    "find_scan_folders",
    "assign_splits",
    "load_or_create_splits",
    "build_patch_index",
    "load_patch_index",
    "AneurysmPatchDataset",
    "make_dataloaders",
    "prepare_data",
]
