"""Portable, atomic checkpoints and explicit weight compatibility checks."""

import random
import re
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed each generator; deterministic algorithms remain opt-in at the CLI."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Give spawned workers reproducible, distinct augmentation streams."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng() -> dict:
    """Use only tensors and builtin types so weights_only loading stays safe."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])


def read_checkpoint(path) -> dict:
    """Load tensors on CPU without executing pickled application objects."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dictionary.")
    return checkpoint


def _unused_upstream_key(key: str) -> bool:
    # These parameters were registered in upstream but never used by forward().
    return (
        key.startswith(("refinement1.", "ups.3."))
        or re.fullmatch(
            r"(?:ltb_encoders|decoders)\.\d+\.\d+\.MASA\.project_out\.(?:weight|bias)", key
        )
        is not None
    )


def load_model_weights(model, path) -> dict:
    """Accept this project's checkpoints or official raw state dictionaries.

    Only known unused upstream parameters may be discarded. Other mismatches,
    including local AgentFormer ablation weights, fail with PyTorch's strict check.
    """
    checkpoint = read_checkpoint(path)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model/state_dict must be a dictionary.")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if "format_version" not in checkpoint:
        state = {key: value for key, value in state.items() if not _unused_upstream_key(key)}
    model.load_state_dict(state, strict=True)
    return checkpoint


def save_checkpoint(path, model, optimizer, scaler, epoch, best_f1, config, generator) -> None:
    """Replace a checkpoint only after its temporary file has been fully written."""
    path = Path(path)
    state = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_f1": best_f1,
        "config": config,
        "rng": capture_rng(),
        "loader_rng": generator.get_state(),
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
