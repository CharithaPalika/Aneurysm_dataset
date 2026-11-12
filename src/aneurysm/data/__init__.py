"""Everything done to the data: dataset readers -> case table -> split -> patches -> torch Dataset,
plus intensity scaling, pseudo-healthy patches and pasting patches back into scans."""
from .catalog import build_catalog, resolve_case_path, summarize_catalog
from .datasets import AneurysmPatchDataset, make_dataloaders
from .intensity import hu_to_norm, lesion_histogram, norm_to_hu, normalize_hu
from .patches import build_patch_index, load_patch_index
from .pipeline import get_data, load_cached_data, make_patches, make_splits, prepare_data
from .pseudo_healthy import dilate, make_pseudo_healthy
from .sources import SOURCES, DatasetSource, get_source
from .splits import assign_splits, load_or_create_splits
from .volume import extract_patch, load_canonical, model_patch_affine, paste_patch_into_volume

__all__ = [
    "SOURCES", "DatasetSource", "get_source",
    "build_catalog", "summarize_catalog", "resolve_case_path",
    "assign_splits", "load_or_create_splits", "make_splits",
    "build_patch_index", "load_patch_index", "make_patches",
    "AneurysmPatchDataset", "make_dataloaders",
    "prepare_data", "load_cached_data", "get_data",
    "normalize_hu", "norm_to_hu", "hu_to_norm", "lesion_histogram",
    "make_pseudo_healthy", "dilate",
    "load_canonical", "extract_patch", "model_patch_affine", "paste_patch_into_volume",
]
