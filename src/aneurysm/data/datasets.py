"""PyTorch Dataset and DataLoaders for the cached aneurysm patches.

Each item is a dict:
    image       : float32 tensor (1, D, H, W), HU window scaled to [-1, 1]
    mask        : float32 tensor (1, D, H, W), 1 = any aneurysm in the patch (the label)
    center_mask : float32 tensor (1, D, H, W), 1 = the aneurysm the patch is centred on.
                  This is the LeFusion target: it is used for the LOSS and the histogram.
                  Other aneurysms in the patch are real context (not in the loss), exactly
                  as at generation time, where they are copied, not drawn.
    hist        : float32 tensor (hist_bins,), histogram inside center_mask (sums to 1)
    case_id, patch_id : str

Axis order (D, H, W) = (z, y, x) of the RAS patch: depth (axial slices) first.
This is what the LeFusion U-Net expects (depth is its "frame" axis).

Reproducibility: random shifts/flips use torch's random generator, which the
DataLoader seeds from `seed` (also inside worker processes). The same seed
gives the same sequence of training batches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig
from ..env import default_num_workers


def normalize_hu(x: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """Clip HU to [hu_min, hu_max] and scale to [-1, 1]."""
    x = np.clip(x.astype(np.float32), hu_min, hu_max)
    return 2.0 * (x - hu_min) / (hu_max - hu_min) - 1.0


def lesion_histogram(image: torch.Tensor, mask: torch.Tensor, bins: int) -> torch.Tensor:
    """Same as LeFusion: histogram over [-1, 1] of the voxels inside `mask`, sums to 1."""
    inside = image[mask > 0]
    if inside.numel() == 0:
        return torch.zeros(bins, dtype=torch.float32)
    return torch.histc(inside, bins=bins, min=-1.0, max=1.0) / inside.numel()


class AneurysmPatchDataset(Dataset):
    """Aneurysm-centred 3D patches of one split ("train", "val" or "test").

    augment=True  : random shift (inside the stored margin) + random flips  (default for train)
    augment=False : centre crop, no flips                                    (default for val/test)
    """

    def __init__(self, index: pd.DataFrame, cfg: DataConfig, split: str, augment: bool | None = None):
        self.cfg = cfg
        self.split = split
        self.rows = index[index["split"] == split].reset_index(drop=True)
        self.augment = (split == "train") if augment is None else augment
        self.shape = cfg.model_patch_shape            # (x, y, z) given to the model
        self.margin = cfg.patch_margin
        self.stored_shape = cfg.stored_patch_shape    # (x, y, z) saved on disk

    def __len__(self) -> int:
        return len(self.rows)

    # -- helpers ------------------------------------------------------------
    def _crop(self, a: np.ndarray, offset: tuple[int, int, int]) -> np.ndarray:
        (ox, oy, oz), (sx, sy, sz) = offset, self.shape
        return a[ox:ox + sx, oy:oy + sy, oz:oz + sz]

    def _random_offset(self, gen: torch.Generator) -> tuple[int, int, int]:
        hi = 2 * self.margin + 1
        return tuple(int(v) for v in torch.randint(0, hi, (3,), generator=gen))

    # -- main ---------------------------------------------------------------
    def __getitem__(self, i: int) -> dict:
        row = self.rows.iloc[i]
        with np.load(self.cfg.abs(row["patch_path"])) as f:
            image, mask = f["image"], f["mask"]
            center = f["center_mask"] if "center_mask" in f.files else mask

        if image.shape != self.stored_shape:
            raise ValueError(f"{row['patch_id']}: stored patch {image.shape} does not match the "
                             f"current settings {self.stored_shape}. Check configs/data.yaml.")

        # 1) crop the model-sized patch (random shift for training, centre otherwise)
        centre = (self.margin,) * 3
        offset = centre
        if self.augment:
            # seed from torch's RNG -> reproducible with a seeded DataLoader
            gen = torch.Generator().manual_seed(int(torch.randint(0, 2**31 - 1, (1,))))
            offset = self._random_offset(gen) if self.margin > 0 else centre
            if self._crop(center, offset).sum() == 0:         # shift lost the aneurysm
                offset = centre
        img, msk, ctr = (self._crop(a, offset) for a in (image, mask, center))

        # 2) random flips (training only)
        if self.augment:
            flips = torch.rand(3, generator=gen) < 0.5
            for axis in range(3):
                if flips[axis]:
                    img, msk, ctr = np.flip(img, axis), np.flip(msk, axis), np.flip(ctr, axis)

        # 3) scale HU, reorder axes (x, y, z) -> (z, y, x), add channel axis
        img = normalize_hu(img, self.cfg.hu_min, self.cfg.hu_max)
        to_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 1, 0)).astype(np.float32)[None])
        img_t, msk_t, ctr_t = to_tensor(img), to_tensor(msk), to_tensor(ctr)

        return {
            "image": img_t,
            "mask": msk_t,
            "center_mask": ctr_t,
            "hist": lesion_histogram(img_t, ctr_t, self.cfg.hist_bins),
            "case_id": row["case_id"],
            "patch_id": row["patch_id"],
        }


def _seed_worker(worker_id: int) -> None:
    """Give numpy/random in each worker a seed derived from torch's (already seeded) worker seed."""
    import random
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_dataloaders(index: pd.DataFrame, cfg: DataConfig, batch_size: int = 2,
                     num_workers: int | None = None, pin_memory: bool | None = None,
                     seed: int = 42) -> dict:
    """DataLoaders for "train", "val" and "test" (empty splits are skipped).

    train: shuffled + augmented; val/test: fixed order, no augmentation.
    `seed` fixes the shuffle order and the augmentations.
    """
    num_workers = default_num_workers() if num_workers is None else num_workers
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders = {}
    for k, split in enumerate(("train", "val", "test")):
        ds = AneurysmPatchDataset(index, cfg, split)
        if len(ds) == 0:
            continue
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train" and len(ds) > batch_size),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(seed + k),
        )
    return loaders
