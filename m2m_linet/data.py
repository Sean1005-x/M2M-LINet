"""Paired image loading with explicit binary labels and shared augmentation."""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _index_images(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in files:
            raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
        files[key] = path
    if not files:
        raise ValueError(f"No supported images found in {directory}")
    return files


def _check_pairs(reference: Dict[str, Path], other: Dict[str, Path], directory: Path) -> None:
    missing = sorted(reference.keys() - other.keys())
    extra = sorted(other.keys() - reference.keys())
    if missing or extra:
        raise ValueError(
            f"Unmatched image stems in {directory}; missing={missing[:5]}, extra={extra[:5]}"
        )


def _read_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous().div_(127.5).sub_(1.0)


def _read_binary(path: Path) -> Tensor:
    with Image.open(path) as image:
        # Keep palette indices: paletted masks often encode classes as indices 0/1.
        values = np.array(image, copy=True)
    if values.ndim == 3:
        if values.shape[2] == 4:
            if not np.all(values[..., 3] == 255):
                raise ValueError(f"Binary mask has transparent pixels: {path}")
            values = values[..., :3]
        if values.shape[2] != 3 or not np.all(values == values[..., :1]):
            raise ValueError(f"Binary mask must be grayscale, not a colored RGB label: {path}")
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Binary mask must be a two-dimensional image: {path}")
    unique = set(np.unique(values).tolist())
    if not (unique <= {0, 1} or unique <= {0, 255}):
        raise ValueError(
            f"Binary mask must use 0/1 or 0/255 values: {path}; got {sorted(unique)[:10]}"
        )
    return torch.from_numpy((values > 0).astype(np.float32)).unsqueeze(0)


def mask_to_edge(mask: Tensor) -> Tensor:
    """Return a 3x3 morphological gradient using replicated boundary pixels.

    Replicate padding prevents a fully foreground mask from acquiring a false
    contour along the image border. Input and output both have shape [1,H,W].
    """
    padded = F.pad(mask.unsqueeze(0), (1, 1, 1, 1), mode="replicate")
    dilated = F.max_pool2d(padded, kernel_size=3, stride=1)
    eroded = -F.max_pool2d(-padded, kernel_size=3, stride=1)
    return (dilated - eroded).squeeze(0)


class ChangeDetectionDataset(Dataset):
    """Read ``root/split/{A,B,label[,edge]}`` (or ``time1,time2``).

    Pairing uses case-insensitive stems and allows different image extensions.
    Every required directory must contain exactly the same stems. Labels can
    use either 0/1 or 0/255; RGB masks must have equal color channels. Spatial
    sizes are checked without resizing. Set ``require_labels=False`` for paired
    image inference. If edge/ is absent, edges are derived from the mask.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        augment: bool = False,
        require_labels: bool = True,
    ) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.augment = augment
        self.require_labels = require_labels
        split_dir = self.root / split
        schemes = [
            names
            for names in (("A", "B"), ("time1", "time2"))
            if any((split_dir / name).exists() for name in names)
        ]
        if len(schemes) > 1:
            raise ValueError(
                f"Ambiguous image directories in {split_dir}: use only A/B or time1/time2"
            )
        image_dirs: Tuple[str, str] = schemes[0] if schemes else ("A", "B")
        images_a = _index_images(split_dir / image_dirs[0])
        images_b = _index_images(split_dir / image_dirs[1])
        _check_pairs(images_a, images_b, split_dir / image_dirs[1])
        labels = None
        edges = None
        if require_labels:
            labels = _index_images(split_dir / "label")
            _check_pairs(images_a, labels, split_dir / "label")
            if (split_dir / "edge").exists():
                edges = _index_images(split_dir / "edge")
                _check_pairs(images_a, edges, split_dir / "edge")
        self.samples = [
            (
                images_a[key],
                images_b[key],
                labels[key] if labels else None,
                edges[key] if edges else None,
            )
            for key in sorted(images_a)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str]]:
        path_a, path_b, path_mask, path_edge = self.samples[index]
        tensors = {"image_a": _read_rgb(path_a), "image_b": _read_rgb(path_b)}
        if path_mask is not None:
            tensors["mask"] = _read_binary(path_mask)
            tensors["edge"] = (
                _read_binary(path_edge) if path_edge else mask_to_edge(tensors["mask"])
            )
        size = tensors["image_a"].shape[-2:]
        if any(tensor.shape[-2:] != size for tensor in tensors.values()):
            shapes = {key: tuple(value.shape[-2:]) for key, value in tensors.items()}
            raise ValueError(f"Paired image/mask sizes differ for {path_a.stem}: {shapes}")
        if self.augment:
            flip_horizontal = bool(torch.rand(()) < 0.5)
            flip_vertical = bool(torch.rand(()) < 0.5)
            rotations = int(torch.randint(4 if size[0] == size[1] else 2, ()).item())
            if size[0] != size[1]:
                rotations *= 2
            for key, tensor in tensors.items():
                if flip_horizontal:
                    tensor = tensor.flip(-1)
                if flip_vertical:
                    tensor = tensor.flip(-2)
                tensors[key] = torch.rot90(tensor, rotations, (-2, -1)).contiguous()
        return {**tensors, "name": path_a.stem + ".png"}
