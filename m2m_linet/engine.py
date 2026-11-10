"""Shared training, evaluation, and prediction loops with no global device state."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .metrics import BinaryMetrics


def _move(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, amp=False):
    """Accumulate pixel-level metrics over the entire split, without rounding.

    Validation runs with inference_mode; training retains the differentiable
    logits for the loss and detaches probabilities before measuring accuracy.
    """
    training = optimizer is not None
    model.train(training)
    meter = BinaryMetrics()
    loss_total = 0.0
    samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            batch = _move(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                outputs = model(batch["image_a"], batch["image_b"])
                loss = criterion(outputs, batch["mask"], batch["edge"])
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss encountered; checkpoint was not overwritten."
                )
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            size = batch["image_a"].shape[0]
            samples += size
            loss_total += loss.detach().item() * size
            meter.update(outputs[0].detach().float().sigmoid(), batch["mask"])
    if samples == 0:
        raise ValueError("Data loader yielded no samples.")
    return {"loss": loss_total / samples, **meter.compute()}


@torch.inference_mode()
def predict_masks(model, loader, device, output_dir, threshold=0.5, save_probabilities=False):
    """Save one binary PNG per input stem, including unlabeled image pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_dir = output_dir / "probabilities"
    if save_probabilities:
        probability_dir.mkdir(exist_ok=True)
    model.eval()
    count = 0
    for batch in loader:
        moved = _move(batch, device)
        logits, _ = model(moved["image_a"], moved["image_b"])
        if not torch.isfinite(logits).all():
            raise FloatingPointError(
                "Non-finite prediction logits; masks were not saved for this batch."
            )
        probabilities = logits.float().sigmoid().cpu().numpy()[:, 0]
        for name, probability in zip(batch["name"], probabilities):
            # Construct a separate array; saving must never alter model outputs.
            mask = (probability >= threshold).astype(np.uint8) * 255
            Image.fromarray(mask).save(output_dir / name)
            if save_probabilities:
                np.save(probability_dir / (Path(name).stem + ".npy"), probability)
            count += 1
    return count
