"""Dataset-level binary metrics with rows=truth and columns=prediction."""

import math
from typing import Dict

import torch
from torch import Tensor


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


class BinaryMetrics:
    """Accumulate counts before computing metrics; never average batch F1.

    Undefined precision/recall/F1/IoU and kappa are reported as zero. An empty
    accumulator raises an error instead of silently reporting meaningful scores.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be finite and between zero and one")
        self.threshold = threshold
        self.confusion = torch.zeros((2, 2), dtype=torch.int64)

    def reset(self) -> None:
        self.confusion.zero_()

    @torch.no_grad()
    def update(self, probabilities: Tensor, target: Tensor) -> None:
        if probabilities.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        if not torch.isfinite(probabilities).all() or not torch.isfinite(target).all():
            raise ValueError("Metrics require finite predictions and targets")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Metrics expect probabilities in [0,1], not logits")
        if ((target != 0) & (target != 1)).any():
            raise ValueError("Metrics expect binary targets encoded as 0/1")
        predicted = (probabilities.detach() >= self.threshold).to(torch.int64)
        truth = target.detach().to(device=predicted.device, dtype=torch.int64)
        counts = torch.bincount((truth.reshape(-1) * 2 + predicted.reshape(-1)), minlength=4)
        self.confusion += counts.reshape(2, 2).cpu()

    def compute(self) -> Dict[str, float]:
        tn, fp, fn, tp = self.confusion.reshape(-1).tolist()
        total = tn + fp + fn + tp
        if not total:
            raise ValueError("Cannot compute metrics without any pixels")
        accuracy = (tp + tn) / total
        expected = ((tn + fp) * (tn + fn) + (fn + tp) * (fp + tp)) / total**2
        return {
            "precision": _safe_divide(tp, tp + fp),
            "recall": _safe_divide(tp, tp + fn),
            "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
            "iou": _safe_divide(tp, tp + fp + fn),
            "accuracy": accuracy,
            "kappa": _safe_divide(accuracy - expected, 1 - expected),
        }
