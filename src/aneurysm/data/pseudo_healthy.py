"""Pseudo-healthy patches: remove an aneurysm (and its blurred rim) from a real patch.

Used by the evaluation (cases 3 and 4) to test generation on a vessel that has no aneurysm.

One patch, model values in [-1, 1], axes (D, H, W):

1. region  = aneurysm mask grown by `dilate_voxels` (round kernel)
             minus other aneurysms in the patch (they stay as they are)
             minus the parent vessel: bright rim voxels connected to bright vessel outside the region
   `region` is also the "dilated mask" used for generation.
2. tissue  = a ring around the region without bright (vessel) voxels
             -> its median and spread give the fill value and the noise level
3. fill    = tissue median + smooth noise with the same spread (CT noise is grainy, not white)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def ball(radius: int) -> np.ndarray:
    """Round 3D structuring element of the given radius (in voxels)."""
    r = int(radius)
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r


def dilate(mask: np.ndarray, voxels: int) -> np.ndarray:
    """Grow a 3D mask by `voxels` with a round kernel (same shape as sample.grow_mask)."""
    if voxels <= 0:
        return mask > 0
    return ndimage.binary_dilation(mask > 0, structure=ball(voxels))


def _tissue_values(image: np.ndarray, region: np.ndarray, blocked: np.ndarray, bright: np.ndarray,
                   ring_voxels: int) -> np.ndarray:
    """Non-vessel voxels in a ring around `region` (whole patch if the ring is too small)."""
    ring = dilate(region, ring_voxels) & ~blocked & ~bright
    if ring.sum() < 20:
        ring = ~dilate(region, 1) & ~blocked & ~bright
    return image[ring]


def make_pseudo_healthy(image: np.ndarray, center_mask: np.ndarray, all_mask: np.ndarray,
                        dilate_voxels: int, vessel_norm: float, ring_voxels: int = 4,
                        noise_sigma: float = 0.7, seed: int = 0) -> dict:
    """Remove the centred aneurysm + its rim from one patch.

    image       : (D, H, W) model values in [-1, 1]
    center_mask : (D, H, W) 0/1, the aneurysm to remove
    all_mask    : (D, H, W) 0/1, all aneurysms in the patch (others are kept)
    vessel_norm : "bright" threshold in model values (vessel / aneurysm)

    Returns dict:
        image        pseudo-healthy patch (D, H, W)
        region       removed area = dilated mask for generation (float32 0/1)
        parent       rim voxels kept as parent vessel (bool)
        tissue_norm  fill value (median of the tissue ring), model values
        tissue_std   noise level (robust spread of the tissue ring), model values
    """
    img = image.astype(np.float32)
    center = center_mask > 0
    others = (all_mask > 0) & ~center
    bright = img > vessel_norm

    # 1) area to remove: grown aneurysm, without other aneurysms and without the parent vessel
    grown = dilate(center, dilate_voxels) & ~others
    rim = grown & ~center
    vessel_outside = bright & ~grown & ~others
    # parent vessel = bright rim voxels reachable from the vessel outside in <= dilate_voxels steps
    parent = ndimage.binary_dilation(vessel_outside, structure=ball(1), iterations=max(dilate_voxels, 1),
                                     mask=(rim & bright) | vessel_outside) & rim
    region = grown & ~parent

    # 2) fill value and noise level from the surrounding tissue
    tissue = _tissue_values(img, grown, others, bright, ring_voxels)
    med = float(np.median(tissue))
    spread = float(1.4826 * np.median(np.abs(tissue - med)))     # robust std (MAD)

    # 3) tissue median + smooth noise with the same spread
    noise = np.random.default_rng(seed).standard_normal(img.shape).astype(np.float32)
    if noise_sigma > 0:
        noise = ndimage.gaussian_filter(noise, noise_sigma)
    noise /= max(float(noise.std()), 1e-6)
    fill = np.clip(med + spread * noise, -1.0, 1.0)

    out = img.copy()
    out[region] = fill[region]
    return {"image": out, "region": region.astype(np.float32), "parent": parent,
            "tissue_norm": med, "tissue_std": spread}
