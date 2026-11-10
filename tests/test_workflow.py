"""Regression tests for gradient modes, portable checkpoints, and mask export."""

import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from m2m_linet.checkpoint import (
    capture_rng,
    load_model_weights,
    read_checkpoint,
    restore_rng,
    save_checkpoint,
    seed_everything,
)
from m2m_linet.cli import build_parser
from m2m_linet.engine import predict_masks, run_epoch
from m2m_linet.losses import ChangeDetectionLoss


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 2, 1)
        self.dropout = nn.Dropout2d(0.3)
        self.gradient_modes = []

    def forward(self, a, b):
        self.gradient_modes.append(torch.is_grad_enabled())
        output = self.conv(self.dropout(torch.cat((a, b), dim=1)))
        return output[:, :1], output[:, 1:]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_everything(7)
        self.model = TinyModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.loader = DataLoader(
            [
                {
                    "image_a": torch.rand(3, 8, 8),
                    "image_b": torch.rand(3, 8, 8),
                    "mask": torch.zeros(1, 8, 8),
                    "edge": torch.zeros(1, 8, 8),
                    "name": "scene.png",
                }
            ],
            batch_size=1,
        )

    def test_optimizer_update_and_gradient_free_validation(self):
        before = self.model.conv.weight.detach().clone()
        result = run_epoch(
            self.model, self.loader, torch.device("cpu"), ChangeDetectionLoss(), self.optimizer
        )
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertFalse(torch.equal(before, self.model.conv.weight))
        before = self.model.conv.weight.detach().clone()
        run_epoch(self.model, self.loader, torch.device("cpu"), ChangeDetectionLoss())
        self.assertTrue(torch.equal(before, self.model.conv.weight))
        self.assertEqual(self.model.gradient_modes, [True, False])

    def test_checkpoint_roundtrip_optimizer_and_rng(self):
        run_epoch(
            self.model, self.loader, torch.device("cpu"), ChangeDetectionLoss(), self.optimizer
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            generator = torch.Generator().manual_seed(13)
            save_checkpoint(
                path, self.model, self.optimizer, self.scaler, 1, 0.0, {"seed": 7}, generator
            )
            loaded = TinyModel()
            checkpoint = load_model_weights(loaded, path)
            self.assertEqual(checkpoint["epoch"], 1)
            for original, restored in zip(self.model.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(original, restored))
            restored_optimizer = torch.optim.AdamW(loaded.parameters())
            restored_optimizer.load_state_dict(checkpoint["optimizer"])
            self.assertEqual(restored_optimizer.param_groups[0]["lr"], 0.01)
            restore_rng(checkpoint["rng"])
            first = (random.random(), np.random.rand(), torch.rand(1))
            restore_rng(read_checkpoint(path)["rng"])
            second = (random.random(), np.random.rand(), torch.rand(1))
            self.assertEqual(first[:2], second[:2])
            self.assertTrue(torch.equal(first[2], second[2]))
            self.assertFalse(path.with_name("last.pt.tmp").exists())

    def test_rng_capture_replays(self):
        state = capture_rng()
        expected = torch.rand(4)
        restore_rng(state)
        self.assertTrue(torch.equal(expected, torch.rand(4)))

    def test_resumed_training_matches_uninterrupted_trajectory(self):
        generator = torch.Generator().manual_seed(29)
        loader = DataLoader(
            list(self.loader.dataset) * 4, batch_size=2, shuffle=True, generator=generator
        )
        criterion = ChangeDetectionLoss()
        device = torch.device("cpu")
        run_epoch(self.model, loader, device, criterion, self.optimizer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_checkpoint(path, self.model, self.optimizer, self.scaler, 1, 0.0, {}, generator)
            run_epoch(self.model, loader, device, criterion, self.optimizer)
            expected = {key: value.clone() for key, value in self.model.state_dict().items()}
            restored = TinyModel()
            checkpoint = load_model_weights(restored, path)
            optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
            optimizer.load_state_dict(checkpoint["optimizer"])
            restore_rng(checkpoint["rng"])
            generator.set_state(checkpoint["loader_rng"])
            run_epoch(restored, loader, device, criterion, optimizer)
            for key, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)
            for original, resumed in zip(self.optimizer.state.values(), optimizer.state.values()):
                for key in original:
                    self.assertTrue(torch.equal(original[key], resumed[key]), key)

    def test_export_binary_mask_and_probabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            count = predict_masks(
                self.model, self.loader, torch.device("cpu"), directory, save_probabilities=True
            )
            self.assertEqual(count, 1)
            with Image.open(Path(directory) / "scene.png") as mask:
                array = np.asarray(mask)
                self.assertEqual(array.shape, (8, 8))
                self.assertTrue(set(np.unique(array)).issubset({0, 255}))
            probability = np.load(Path(directory) / "probabilities/scene.npy")
            self.assertEqual(probability.dtype, np.float32)
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())
            self.assertEqual(self.model.gradient_modes, [False])

    def test_cli_default_uses_validation_and_rejects_typos(self):
        parser = build_parser("train")
        args = parser.parse_args(["--data-root", "dummy"])
        self.assertEqual(args.val_split, "val")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--data-root", "dummy", "--batch-szie", "16"])

    def test_help_creates_no_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for script in ("train.py", "evaluate.py", "predict.py"):
                result = subprocess.run(
                    [sys.executable, str(root / script), "--help"],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("--data-root", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
