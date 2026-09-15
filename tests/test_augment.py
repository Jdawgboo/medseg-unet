from __future__ import annotations

import random
import unittest

import torch

from medseg_unet.augment import (
    PairedFlipDataset,
    draw_flip_flags,
    paired_flip,
    random_paired_flip,
    should_augment,
)
from medseg_unet.pipeline import pair_case_ids, run_synthetic_demo, split_case_ids


def geometric_pair(size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic asymmetric image/mask pair with exact foreground alignment."""
    mask = torch.zeros(1, size, size)
    # Off-center rectangle + corner dot: asymmetric under both flip axes.
    mask[:, 2:6, 1:5] = 1.0
    mask[:, size - 3, size - 2] = 1.0
    image = mask.clone() * 0.9 + 0.05
    return image, mask


class PairedFlipTests(unittest.TestCase):
    def test_paired_flip_matches_torch_flip(self):
        image, mask = geometric_pair()
        flipped_image, flipped_mask = paired_flip(image, mask, horizontal=True)
        self.assertTrue(torch.equal(flipped_image, torch.flip(image, dims=[-1])))
        self.assertTrue(torch.equal(flipped_mask, torch.flip(mask, dims=[-1])))
        flipped_image, flipped_mask = paired_flip(image, mask, horizontal=True, vertical=True)
        self.assertTrue(torch.equal(flipped_image, torch.flip(image, dims=[-2, -1])))
        self.assertTrue(torch.equal(flipped_mask, torch.flip(mask, dims=[-2, -1])))

    def test_geometric_alignment_preserved(self):
        """Same transform on image+mask keeps foreground pixels coincident."""
        image, mask = geometric_pair()
        for horizontal in (False, True):
            for vertical in (False, True):
                flipped_image, flipped_mask = paired_flip(
                    image, mask, horizontal=horizontal, vertical=vertical
                )
                # Foreground coincidence survives the joint transform.
                self.assertTrue(
                    torch.equal((flipped_image > 0.5), (flipped_mask > 0.5)),
                    f"misaligned for h={horizontal} v={vertical}",
                )
                # And it equals the expected geometric result.
                expected = torch.flip(
                    mask,
                    dims=[d for d, f in ((-2, vertical), (-1, horizontal)) if f],
                ) if (horizontal or vertical) else mask
                self.assertTrue(torch.equal(flipped_mask, expected))
        # A naive image-only flip would break alignment (guard against regressions).
        image_only = torch.flip(image, dims=[-1])
        self.assertFalse(torch.equal((image_only > 0.5), (mask > 0.5)))

    def test_seeded_reproducible(self):
        image, mask = geometric_pair()
        first = draw_flip_flags(random.Random(1234))
        second = draw_flip_flags(random.Random(1234))
        self.assertEqual(first, second)
        out_a = random_paired_flip(image, mask, random.Random(99))
        out_b = random_paired_flip(image, mask, random.Random(99))
        self.assertTrue(torch.equal(out_a[0], out_b[0]))
        self.assertTrue(torch.equal(out_a[1], out_b[1]))
        self.assertEqual(out_a[2], out_b[2])

    def test_dataset_train_reproducible_per_index(self):
        images = torch.stack([geometric_pair()[0] for _ in range(4)])
        masks = torch.stack([geometric_pair()[1] for _ in range(4)])
        train_a = PairedFlipDataset(images, masks, split="train", seed=7)
        train_b = PairedFlipDataset(images, masks, split="train", seed=7)
        for index in (3, 0, 2, 1):  # out-of-order access still deterministic
            image_a, mask_a = train_a[index]
            image_b, mask_b = train_b[index]
            self.assertTrue(torch.equal(image_a, image_b))
            self.assertTrue(torch.equal(mask_a, mask_b))
            self.assertTrue(torch.equal((image_a > 0.5), (mask_a > 0.5)))

    def test_augmentation_disabled_for_val_test(self):
        images = torch.stack([geometric_pair()[0] for _ in range(2)])
        masks = torch.stack([geometric_pair()[1] for _ in range(2)])
        # Force-every-flip probabilities: train flips, val/test must not.
        for split in ("validation", "val", "test"):
            dataset = PairedFlipDataset(
                images, masks, split=split, seed=3, p_horizontal=1.0, p_vertical=1.0
            )
            for index in range(len(dataset)):
                image_t, mask_t = dataset[index]
                self.assertTrue(torch.equal(image_t, images[index]))
                self.assertTrue(torch.equal(mask_t, masks[index]))
            self.assertFalse(should_augment(split))
        self.assertTrue(should_augment("train"))
        train = PairedFlipDataset(images, masks, split="train", seed=3, p_horizontal=1.0, p_vertical=1.0)
        image_t, mask_t = train[0]
        self.assertTrue(torch.equal(image_t, torch.flip(images[0], dims=[-2, -1])))
        self.assertTrue(torch.equal(mask_t, torch.flip(masks[0], dims=[-2, -1])))

    def test_case_split_policy_unchanged(self):
        ids = pair_case_ids({"a", "b", "c", "d", "e"}, {"a", "b", "c", "d", "e"})
        first = split_case_ids(ids, seed=17)
        second = split_case_ids(ids, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first["test"]), 1)
        self.assertEqual(len(first["validation"]), 1)
        self.assertTrue(set(first["train"]).isdisjoint(first["test"]))
        self.assertTrue(set(first["train"]).isdisjoint(first["validation"]))
        self.assertEqual(
            sorted(first["train"] + first["validation"] + first["test"]), sorted(ids)
        )

    def test_pipeline_smoke_with_augmentation(self):
        _, result = run_synthetic_demo(epochs=1, augment_train=True)
        self.assertTrue(0 <= result["metrics"]["test_dice"] <= 1)
        _, plain = run_synthetic_demo(epochs=1, augment_train=False)
        self.assertTrue(0 <= plain["metrics"]["test_dice"] <= 1)


if __name__ == "__main__":
    unittest.main()
