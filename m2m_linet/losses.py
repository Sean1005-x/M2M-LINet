"""Losses for change and boundary logits, matching the local training objective."""

from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def dice_loss(probabilities: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    """Batch-global soft Dice, preserving the original code's smooth=1 formula."""
    if probabilities.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {probabilities.shape} vs {target.shape}"
        )
    prediction = probabilities.float().reshape(-1)
    truth = target.float().reshape(-1)
    intersection = (prediction * truth).sum()
    return 1.0 - (2.0 * intersection + smooth) / (prediction.sum() + truth.sum() + smooth)


class ChangeDetectionLoss(nn.Module):
    """BCE(change) + Dice(change) + edge_weight * Dice(edge).

    Both outputs are logits. BCEWithLogitsLoss is numerically stable while the
    sigmoid Dice terms retain the local implementation's batch-global objective.
    This is the supplied code's objective; it does not assert paper reproduction.
    """

    def __init__(self, edge_weight: float = 1.0) -> None:
        super().__init__()
        if not torch.isfinite(torch.tensor(edge_weight)) or edge_weight < 0:
            raise ValueError("edge_weight must be finite and nonnegative")
        self.edge_weight = float(edge_weight)

    def forward(self, outputs: Tuple[Tensor, Tensor], mask: Tensor, edge: Tensor) -> Tensor:
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise ValueError("Model outputs must be a pair of change and edge logits")
        change_logits, edge_logits = outputs
        if change_logits.shape != mask.shape or edge_logits.shape != edge.shape:
            raise ValueError("Each output must have the same shape as its target")
        change_logits = change_logits.float()
        edge_logits = edge_logits.float()
        mask = mask.float()
        edge = edge.float()
        return (
            F.binary_cross_entropy_with_logits(change_logits, mask)
            + dice_loss(change_logits.sigmoid(), mask)
            + self.edge_weight * dice_loss(edge_logits.sigmoid(), edge)
        )
