"""Model shape, gradient and pretrained checkpoint contracts."""

import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from m2m_linet.models import M2MLINet


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_sizes_and_input_validation(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = M2MLINet().eval().to(device)
        with torch.inference_mode():
            for height, width in ((64, 64), (64, 96), (256, 256)):
                x1 = torch.randn(1, 3, height, width, device=device)
                x2 = torch.randn_like(x1)
                outputs = model(x1, x2)
                for output in outputs:
                    self.assertEqual(output.shape, (1, 1, height, width))
                    self.assertTrue(torch.isfinite(output).all().item())
            for shape in ((1, 3, 32, 32), (1, 3, 65, 64), (1, 1, 64, 64)):
                x = torch.randn(shape, device=device)
                with self.assertRaises(ValueError):
                    model(x, x)
            with self.assertRaises(ValueError):
                model(
                    torch.zeros(1, 3, 64, 64, device=device),
                    torch.zeros(1, 3, 64, 96, device=device),
                )

    def test_backward_uses_all_parameters(self):
        torch.manual_seed(17)
        model = M2MLINet().train()
        x1 = torch.randn(2, 3, 64, 64)
        x2 = torch.randn_like(x1)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        change, edge = model(x1, x2)
        loss = F.binary_cross_entropy_with_logits(change, target)
        loss = loss + F.binary_cross_entropy_with_logits(edge, target)
        loss.backward()
        unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
        self.assertEqual(unused, [])
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters()))
        keys = model.state_dict()
        self.assertFalse(any(key.startswith(("refinement1.", "ups.3.")) for key in keys))
        self.assertFalse(any("MASA.project_out" in key for key in keys))

    def test_backbone_loading_is_complete_and_atomic(self):
        model = M2MLINet()
        state = {key: value.clone() for key, value in model.backbone.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "backbone.pth"
            # ImageNet classification head is intentionally ignored by a feature backbone.
            torch.save(
                {**state, "head.weight": torch.zeros(1000, 256), "head.bias": torch.zeros(1000)},
                checkpoint,
            )
            model.load_backbone_weights(checkpoint)
            for actual, expected in zip(model.backbone.state_dict().values(), state.values()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            first = next(iter(state))
            wrong_cases = [
                {key: value for key, value in state.items() if key != first},
                {**state, "unexpected": torch.zeros(1)},
                {**state, first: torch.zeros(1)},
                {**state, first: "not a tensor"},
            ]
            for wrong in wrong_cases:
                torch.save(wrong, checkpoint)
                with self.assertRaisesRegex(ValueError, "Incompatible PVT-v2 B0"):
                    model.load_backbone_weights(checkpoint)
                for key, value in model.backbone.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
