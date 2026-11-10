"""Filesystem fixtures for pairing, label encoding and synchronized transforms."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from m2m_linet.data import ChangeDetectionDataset, mask_to_edge


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, directory, name, values, split="train"):
        path = self.root / split / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(values).save(path)
        return path

    def pair(self, name="sample", size=(8, 8), dirs=("A", "B"), encoding=255):
        mask = np.zeros(size, dtype=np.uint8)
        mask[1:3, 2:4] = 1
        image = np.repeat((mask * 255)[..., None], 3, axis=2)
        self.write(dirs[0], name + ".png", image)
        self.write(dirs[1], name + ".bmp", image)
        self.write("label", name + ".png", mask * encoding)
        return mask

    def test_encodings_normalization_and_different_extensions(self):
        mask = self.pair("a", encoding=1)
        self.pair("b", encoding=255)
        dataset = ChangeDetectionDataset(self.root)
        first, second = dataset[0], dataset[1]
        self.assertEqual(first["name"], "a.png")
        self.assertEqual(tuple(first["image_a"].shape), (3, 8, 8))
        torch.testing.assert_close(first["mask"], torch.from_numpy(mask.astype(np.float32))[None])
        torch.testing.assert_close(first["mask"], second["mask"])
        self.assertEqual(float(first["image_a"].min()), -1.0)
        self.assertEqual(float(first["image_a"].max()), 1.0)
        torch.testing.assert_close(first["edge"], mask_to_edge(first["mask"]))

    def test_shared_augmentation_and_rectangular_shape(self):
        self.pair(size=(8, 12))
        dataset = ChangeDetectionDataset(self.root, augment=True)
        for seed in range(8):
            torch.manual_seed(seed)
            item = dataset[0]
            self.assertEqual(tuple(item["mask"].shape), (1, 8, 12))
            torch.testing.assert_close((item["image_a"][:1] + 1) / 2, item["mask"])
            torch.testing.assert_close(item["image_a"], item["image_b"])
            torch.testing.assert_close(item["edge"], mask_to_edge(item["mask"]))

    def test_palette_mask_and_provided_edge(self):
        mask = self.pair()
        path = self.root / "train" / "label" / "sample.png"
        palette_mask = Image.fromarray(mask, mode="P")
        palette_mask.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
        palette_mask.save(path)
        self.write("edge", "sample.png", np.zeros_like(mask))
        item = ChangeDetectionDataset(self.root)[0]
        self.assertEqual(item["mask"].sum().item(), 4)
        self.assertEqual(item["edge"].sum().item(), 0)

    def test_unlabeled_and_time_directories(self):
        self.pair(dirs=("time1", "time2"))
        (self.root / "train" / "label" / "sample.png").unlink()
        item = ChangeDetectionDataset(self.root, require_labels=False)[0]
        self.assertEqual(set(item), {"image_a", "image_b", "name"})
        with self.assertRaises(ValueError):
            ChangeDetectionDataset(self.root)

    def test_reject_missing_duplicate_and_ambiguous_pairs(self):
        self.pair()
        self.write("B", "extra.png", np.zeros((8, 8), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            ChangeDetectionDataset(self.root)
        (self.root / "train" / "B" / "extra.png").unlink()
        self.write("A", "sample.bmp", np.zeros((8, 8), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ChangeDetectionDataset(self.root)
        (self.root / "train" / "A" / "sample.bmp").unlink()
        self.write("time1", "sample.png", np.zeros((8, 8), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            ChangeDetectionDataset(self.root)

    def test_reject_bad_mask_and_size_mismatch(self):
        self.pair()
        self.write("label", "sample.png", np.full((8, 8), 128, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "0/1 or 0/255"):
            ChangeDetectionDataset(self.root)[0]
        colored = np.zeros((8, 8, 3), dtype=np.uint8)
        colored[..., 0] = 255
        self.write("label", "sample.png", colored)
        with self.assertRaisesRegex(ValueError, "colored RGB"):
            ChangeDetectionDataset(self.root)[0]
        self.write("label", "sample.png", np.zeros((4, 8), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "sizes differ"):
            ChangeDetectionDataset(self.root)[0]

    def test_replicate_padding_has_no_artificial_border(self):
        for value in (0.0, 1.0):
            self.assertEqual(mask_to_edge(torch.full((1, 8, 8), value)).sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
