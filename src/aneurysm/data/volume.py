"""Reading volumes, finding aneurysms, and cutting resampled 3D patches."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

STRUCT_26 = np.ones((3, 3, 3), dtype=bool)   # 26-connectivity in 3D


def load_canonical(path: Path, is_mask: bool = False):
    """Load a NIfTI file in RAS orientation.

    Returns (array, affine, spacing_mm). Arrays stay in their small dtype
    (int16 for CT, uint8 for masks) to save memory.
    """
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(img.dataobj)
    data = (data > 0).astype(np.uint8) if is_mask else data.astype(np.int16, copy=False)
    spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    return data, img.affine.astype(np.float64), spacing


@dataclass
class Component:
    """One connected aneurysm blob in a mask."""
    label: int
    voxels: int
    centroid_vox: np.ndarray      # (x, y, z) in native voxel coordinates
    volume_mm3: float

    @property
    def diameter_mm(self) -> float:
        """Diameter of a sphere with the same volume."""
        return float((6.0 * self.volume_mm3 / np.pi) ** (1.0 / 3.0))


def find_components(mask: np.ndarray, spacing: np.ndarray, min_voxels: int):
    """Label aneurysm blobs. Returns (components, label_volume).

    label_volume (uint8, same shape as mask): 0 = background, 1 = largest
    aneurysm, 2 = next largest, ... Blobs with fewer than `min_voxels` voxels
    are removed (set to 0). Component.label is the value in label_volume.
    Work is done inside the mask's bounding box, which is much faster
    than labelling the whole scan.
    """
    label_vol = np.zeros_like(mask, dtype=np.uint8)
    if not mask.any():
        return [], label_vol

    bbox = ndimage.find_objects(mask.astype(np.uint8))[0]
    sub = mask[bbox] > 0
    labels, n = ndimage.label(sub, structure=STRUCT_26)
    offset = np.array([s.start for s in bbox], dtype=np.float64)
    voxel_mm3 = float(np.prod(spacing))

    comps: list[Component] = []
    for lab in range(1, n + 1):
        blob = labels == lab
        count = int(blob.sum())
        if count < min_voxels:
            continue
        centroid = np.array(ndimage.center_of_mass(blob)) + offset
        comps.append(Component(lab, count, centroid, count * voxel_mm3))

    comps.sort(key=lambda c: -c.voxels)
    comps = comps[:255]                                  # uint8 labels
    sub_out = label_vol[bbox]                            # view into label_vol
    for new_id, comp in enumerate(comps, start=1):
        sub_out[labels == comp.label] = new_id
        comp.label = new_id
    return comps, label_vol


def extract_patch(image: np.ndarray, mask: np.ndarray, affine: np.ndarray, spacing: np.ndarray,
                  center_vox: np.ndarray, size, out_spacing: float, pad_value: float):
    """Cut a patch of `size` voxels (int for a cube, or (sx, sy, sz)) at `out_spacing` mm
    around `center_vox`.

    Only a small crop around the centre is resampled, so memory stays low even
    for 900-slice scans. Image: linear interpolation. Mask: nearest neighbour,
    so label values (1, 2, 3, ...) are kept.

    Returns (image_patch int16, mask_patch uint8 [same values as `mask`], patch_affine 4x4).
    Patch voxel (i, j, k) maps to native voxel  center + (idx - (size-1)/2) * out_spacing / spacing.
    """
    size = np.array([size] * 3 if np.isscalar(size) else size, dtype=int)
    step = out_spacing / spacing                          # native voxels per patch voxel, per axis
    half_native = np.ceil((size / 2.0) * step).astype(int) + 2

    lo = np.floor(center_vox).astype(int) - half_native
    hi = np.floor(center_vox).astype(int) + half_native + 1
    shape = np.array(image.shape)
    pad_lo = np.maximum(0, -lo)
    pad_hi = np.maximum(0, hi - shape)
    sl = tuple(slice(max(l, 0), min(h, s)) for l, h, s in zip(lo, hi, shape))

    img_crop = np.pad(image[sl].astype(np.float32), list(zip(pad_lo, pad_hi)), constant_values=pad_value)
    msk_crop = np.pad(mask[sl], list(zip(pad_lo, pad_hi)), constant_values=0)

    # coordinates of every patch voxel inside the crop
    start_in_crop = center_vox - lo                       # centre position inside the padded crop
    axes = [start_in_crop[a] + (np.arange(size[a]) - (size[a] - 1) / 2.0) * step[a] for a in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    coords = np.stack(grid)

    img_patch = ndimage.map_coordinates(img_crop, coords, order=1, mode="constant", cval=pad_value)
    msk_patch = ndimage.map_coordinates(msk_crop.astype(np.float32), coords, order=0, mode="constant", cval=0)

    # affine of the patch (patch voxel -> world mm)
    to_native = np.eye(4)
    to_native[:3, :3] = np.diag(step)
    to_native[:3, 3] = center_vox - (size - 1) / 2.0 * step   # size is an array here
    patch_affine = affine @ to_native

    return (np.round(img_patch).astype(np.int16),
            np.rint(msk_patch).astype(np.uint8),
            patch_affine)


# ----------------------------------------------------------------------
# generated patch -> back into the full scan
# ----------------------------------------------------------------------
def model_patch_affine(stored_affine: np.ndarray, margin: int) -> np.ndarray:
    """Affine of the centre-cropped model patch (x, y, z voxel -> world mm)."""
    shift = np.eye(4)
    shift[:3, 3] = margin
    return stored_affine @ shift


def paste_patch_into_volume(volume_hu: np.ndarray, volume_affine: np.ndarray,
                            patch_hu: np.ndarray, patch_mask: np.ndarray, patch_affine: np.ndarray,
                            blend_sigma_vox: float = 1.0) -> np.ndarray:
    """Paste a generated patch (x, y, z order, HU) back into the full scan (RAS, native grid).

    Only voxels near the mask change: weight = blurred, slightly grown mask, so the
    edge is soft. Returns a new int16 volume.
    """
    weight = ndimage.gaussian_filter(ndimage.binary_dilation(patch_mask > 0, iterations=2).astype(np.float32),
                                     blend_sigma_vox)
    # native voxels covered by the patch: map the patch corners into the volume
    corners = np.array([[i, j, k, 1] for i in (0, patch_hu.shape[0] - 1)
                        for j in (0, patch_hu.shape[1] - 1) for k in (0, patch_hu.shape[2] - 1)]).T
    vox = np.linalg.inv(volume_affine) @ patch_affine @ corners
    lo = np.maximum(np.floor(vox[:3].min(1)).astype(int), 0)
    hi = np.minimum(np.ceil(vox[:3].max(1)).astype(int) + 1, np.array(volume_hu.shape))

    grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"))
    pts = np.concatenate([grid.reshape(3, -1), np.ones((1, grid[0].size))])
    p = (np.linalg.inv(patch_affine) @ volume_affine @ pts)[:3]          # patch coordinates

    w = ndimage.map_coordinates(weight, p, order=1, mode="constant", cval=0).reshape(grid.shape[1:])
    g = ndimage.map_coordinates(patch_hu.astype(np.float32), p, order=1, mode="nearest").reshape(grid.shape[1:])

    out = volume_hu.copy()
    region = tuple(slice(a, b) for a, b in zip(lo, hi))
    out[region] = np.round(w * g + (1 - w) * out[region].astype(np.float32)).astype(np.int16)
    return out
