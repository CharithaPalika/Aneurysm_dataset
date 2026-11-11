"""Everything done to the data: find files -> split -> patches -> torch Dataset,
plus intensity scaling, pseudo-healthy patches and pasting patches back into scans."""
from .catalog import build_catalog, find_scan_folders
from .datasets import AneurysmPatchDataset, make_dataloaders
from .intensity import hu_to_norm, lesion_histogram, norm_to_hu, normalize_hu
from .patches import build_patch_index, load_patch_index
from .pipeline import load_cached_data, prepare_data
from .pseudo_healthy import dilate, make_pseudo_healthy
from .splits import assign_splits, load_or_create_splits
from .volume import extract_patch, load_canonical, model_patch_affine, paste_patch_into_volume

__all__ = [
    "build_catalog", "find_scan_folders",
    "assign_splits", "load_or_create_splits",
    "build_patch_index", "load_patch_index",
    "AneurysmPatchDataset", "make_dataloaders",
    "prepare_data", "load_cached_data",
    "normalize_hu", "norm_to_hu", "hu_to_norm", "lesion_histogram",
    "make_pseudo_healthy", "dilate",
    "load_canonical", "extract_patch", "model_patch_affine", "paste_patch_into_volume",
]
