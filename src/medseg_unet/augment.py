from __future__ import annotations

import random
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = [
    "paired_flip",
    "draw_flip_flags",
    "random_paired_flip",
    "should_augment",
    "PairedFlipDataset",
]

_VALID_SPLITS = ("train", "validation", "test")
_SPLIT_ALIASES = {"val": "validation"}


def _normalize_split(split: str) -> str:
    key = str(split).strip().lower()
    key = _SPLIT_ALIASES.get(key, key)
    if key not in _VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {_VALID_SPLITS} (alias 'val' accepted)")
    return key


def should_augment(split: str) -> bool:
    """Return True only for the training split.

    Validation and test data must never be augmented so metrics stay
    comparable across runs. Case-level split policy is unchanged; this
    helper only gates per-sample transforms.
    """
    return _normalize_split(split) == "train"


def paired_flip(image: Tensor, mask: Tensor, *, horizontal: bool = False, vertical: bool = False) -> Tuple[Tensor, Tensor]:
    """Apply the same spatial flip to an image and its mask.

    Horizontal flips the width axis (last dim), vertical flips the height
    axis (second-to-last dim). Both tensors must share the same shape with
    at least two spatial dims; leading dims (channel/batch) are preserved.
    """
    if not isinstance(image, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("image and mask must be torch Tensors")
    if image.shape != mask.shape:
        raise ValueError(f"image and mask shapes must match, got {tuple(image.shape)} vs {tuple(mask.shape)}")
    if image.ndim < 2:
        raise ValueError("image and mask must have at least 2 dimensions (H, W)")
    dims: list[int] = []
    if vertical:
        dims.append(-2)
    if horizontal:
        dims.append(-1)
    if not dims:
        return image.clone(), mask.clone()
    return torch.flip(image, dims=dims), torch.flip(mask, dims=dims)


def draw_flip_flags(rng: random.Random, p_horizontal: float = 0.5, p_vertical: float = 0.5) -> Tuple[bool, bool]:
    """Draw (horizontal, vertical) flags from a seeded ``random.Random``.

    Using an explicit RNG instance keeps behaviour reproducible: the same
    seed always yields the same flags.
    """
    if not 0.0 <= p_horizontal <= 1.0:
        raise ValueError(f"p_horizontal must be in [0, 1], got {p_horizontal}")
    if not 0.0 <= p_vertical <= 1.0:
        raise ValueError(f"p_vertical must be in [0, 1], got {p_vertical}")
    return (rng.random() < p_horizontal, rng.random() < p_vertical)


def random_paired_flip(
    image: Tensor,
    mask: Tensor,
    rng: random.Random,
    *,
    p_horizontal: float = 0.5,
    p_vertical: float = 0.5,
) -> Tuple[Tensor, Tensor, Tuple[bool, bool]]:
    """Apply a seeded random paired flip; returns (image, mask, flags)."""
    flags = draw_flip_flags(rng, p_horizontal, p_vertical)
    flipped_image, flipped_mask = paired_flip(image, mask, horizontal=flags[0], vertical=flags[1])
    return flipped_image, flipped_mask, flags


class PairedFlipDataset(Dataset):
    """Dataset wrapper applying deterministic paired flips to train only.

    Each training index derives its own ``random.Random(seed + index)`` RNG,
    so the same (seed, index) always yields the same transform regardless of
    access order or DataLoader shuffling. ``validation``/``test`` splits
    return the original pair untouched.
    """

    def __init__(
        self,
        images: Tensor,
        masks: Tensor,
        *,
        split: str = "train",
        seed: int = 17,
        p_horizontal: float = 0.5,
        p_vertical: float = 0.5,
    ) -> None:
        if not isinstance(images, Tensor) or not isinstance(masks, Tensor):
            raise TypeError("images and masks must be torch Tensors")
        if images.shape != masks.shape:
            raise ValueError(f"images and masks shapes must match, got {tuple(images.shape)} vs {tuple(masks.shape)}")
        if images.ndim < 3:
            raise ValueError("images/masks must be batched as [N, ..., H, W] (at least 3 dims)")
        if len(images) != len(masks):
            raise ValueError("images and masks must contain the same number of samples")
        if not 0.0 <= p_horizontal <= 1.0:
            raise ValueError(f"p_horizontal must be in [0, 1], got {p_horizontal}")
        if not 0.0 <= p_vertical <= 1.0:
            raise ValueError(f"p_vertical must be in [0, 1], got {p_vertical}")
        self.images = images
        self.masks = masks
        self.split = _normalize_split(split)
        self.seed = int(seed)
        self.p_horizontal = float(p_horizontal)
        self.p_vertical = float(p_vertical)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def flip_flags_for_index(self, index: int) -> Tuple[bool, bool]:
        if not should_augment(self.split):
            return (False, False)
        rng = random.Random(self.seed + int(index))
        return draw_flip_flags(rng, self.p_horizontal, self.p_vertical)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = self.images[int(index)]
        mask = self.masks[int(index)]
        if not should_augment(self.split):
            return image.clone(), mask.clone()
        rng = random.Random(self.seed + int(index))
        flipped_image, flipped_mask, _ = random_paired_flip(
            image, mask, rng, p_horizontal=self.p_horizontal, p_vertical=self.p_vertical
        )
        return flipped_image, flipped_mask
