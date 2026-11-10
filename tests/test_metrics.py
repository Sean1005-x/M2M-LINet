"""Regression tests for confusion orientation, aggregation and kappa."""

import unittest

import torch

from m2m_linet.metrics import BinaryMetrics


class BinaryMetricsTests(unittest.TestCase):
    def test_known_confusion_and_input_preservation(self):
        truth = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
        probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9, 0.1, 0.2, 0.8])
        before = probabilities.clone()
        metrics = BinaryMetrics()
        metrics.update(probabilities[:5], truth[:5])
        metrics.update(probabilities[5:], truth[5:])
        self.assertTrue(torch.equal(metrics.confusion, torch.tensor([[4, 1], [2, 1]])))
        for key, expected in {
            "precision": 0.5,
            "recall": 1 / 3,
            "f1": 0.4,
            "iou": 0.25,
            "accuracy": 0.625,
            "kappa": 1 / 7,
        }.items():
            self.assertAlmostEqual(metrics.compute()[key], expected)
        self.assertTrue(torch.equal(probabilities, before))
        metrics.reset()
        with self.assertRaises(ValueError):
            metrics.compute()

    def test_degenerate_background_and_threshold(self):
        metrics = BinaryMetrics()
        metrics.update(torch.zeros(4), torch.zeros(4))
        self.assertEqual(metrics.compute()["accuracy"], 1.0)
        self.assertEqual(metrics.compute()["f1"], 0.0)
        self.assertEqual(metrics.compute()["kappa"], 0.0)
        metrics.reset()
        metrics.update(torch.tensor([0.5]), torch.ones(1))
        self.assertEqual(metrics.compute()["f1"], 1.0)

    def test_reject_invalid_inputs(self):
        for threshold in [-1, 2, float("nan")]:
            with self.assertRaises(ValueError):
                BinaryMetrics(threshold)
        for predictions, target in [
            (torch.tensor([2.0]), torch.ones(1)),
            (torch.tensor([float("nan")]), torch.ones(1)),
            (torch.ones(1), torch.tensor([255])),
            (torch.ones(2), torch.ones(1)),
        ]:
            with self.assertRaises(ValueError):
                BinaryMetrics().update(predictions, target)


if __name__ == "__main__":
    unittest.main()
