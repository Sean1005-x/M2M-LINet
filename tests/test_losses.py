"""The stable-logit loss must preserve the original probability objective."""

import unittest

import torch
from torch.nn import functional as F

from m2m_linet.losses import ChangeDetectionLoss, dice_loss


class LossTests(unittest.TestCase):
    def test_matches_original_probability_objective(self):
        change = torch.tensor([[-2.0, 0.0], [1.0, 3.0]], requires_grad=True)
        edge_logits = torch.tensor([[-1.0, 2.0], [0.0, 1.0]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        edge = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        expected = (
            F.binary_cross_entropy(change.sigmoid(), target)
            + dice_loss(change.sigmoid(), target)
            + dice_loss(edge_logits.sigmoid(), edge)
        )
        actual = ChangeDetectionLoss()((change, edge_logits), target, edge)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(change.grad).all())
        self.assertTrue(torch.isfinite(edge_logits.grad).all())
        self.assertGreater(change.grad.abs().sum().item(), 0)

    def test_extreme_logits_remain_finite(self):
        logits = torch.tensor([-1000.0, 1000.0], requires_grad=True)
        target = torch.tensor([1.0, 0.0])
        loss = ChangeDetectionLoss()((logits, logits), target, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_global_dice_formula_and_weight(self):
        probabilities = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        self.assertAlmostEqual(dice_loss(probabilities, target).item(), 1 - 3.8 / 5, places=6)
        logits = torch.zeros(2, 2)
        expected = F.binary_cross_entropy_with_logits(logits, target) + dice_loss(
            logits.sigmoid(), target
        )
        torch.testing.assert_close(
            ChangeDetectionLoss(0)((logits, logits), target, target), expected
        )
        with self.assertRaises(ValueError):
            ChangeDetectionLoss(-1)
        with self.assertRaises(ValueError):
            ChangeDetectionLoss()((logits, logits), target[:1], target)


if __name__ == "__main__":
    unittest.main()
