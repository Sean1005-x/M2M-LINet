"""Side-effect-free command parsing and explicit experiment configuration."""

import argparse
import json
import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _split(value):
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("must be a single split directory name")
    return value


def build_parser(command):
    parser = argparse.ArgumentParser(description=f"M2M-LINet {command}")
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Dataset root with split subdirectories"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-workers", type=_nonnegative, default=4)
    parser.add_argument("--batch-size", type=_positive, default=16 if command == "train" else 1)
    if command == "train":
        parser.add_argument("--train-split", type=_split, default="train")
        parser.add_argument("--val-split", type=_split, default="val")
        parser.add_argument("--epochs", type=_positive, default=200)
        parser.add_argument("--lr", type=_positive_float, default=1e-4)
        parser.add_argument("--weight-decay", type=float, default=5e-4)
        parser.add_argument("--edge-weight", type=float, default=1.0)
        parser.add_argument("--seed", type=_nonnegative, default=2022)
        parser.add_argument(
            "--augment", action="store_true", help="Synchronized flips and right-angle rotations"
        )
        parser.add_argument("--amp", action="store_true", help="Opt-in CUDA mixed precision")
        parser.add_argument(
            "--deterministic", action="store_true", help="Require deterministic PyTorch algorithms"
        )
        parser.add_argument("--output-dir", type=Path, default=Path("runs/m2m_linet"))
        weights = parser.add_mutually_exclusive_group()
        weights.add_argument(
            "--backbone-weights", type=Path, help="Local PVTv2-B0 pretrained weights"
        )
        weights.add_argument(
            "--from-scratch",
            action="store_true",
            help="Explicitly disable pretrained initialization",
        )
        weights.add_argument("--resume", type=Path, help="Resume a standardized last.pt checkpoint")
    else:
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--split", type=_split, default="test")
        if command == "evaluate":
            parser.add_argument("--metrics-output", type=Path, help="Optional JSON output")
        else:
            parser.add_argument("--output-dir", type=Path, required=True)
            parser.add_argument("--threshold", type=float, default=0.5)
            parser.add_argument("--save-probabilities", action="store_true")
    return parser


def resolve_device(value):
    import torch

    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable. Use --device cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device index does not exist.")
    return device


def _loader(dataset, args, device, shuffle=False, generator=None):
    from torch.utils.data import DataLoader

    from .checkpoint import seed_worker

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def train_main(argv=None):
    args = build_parser("train").parse_args(argv)
    import torch

    from .checkpoint import load_model_weights, restore_rng, save_checkpoint, seed_everything
    from .data import ChangeDetectionDataset
    from .engine import run_epoch
    from .losses import ChangeDetectionLoss
    from .models import M2MLINet

    if not 0 <= args.weight_decay < float("inf") or not 0 <= args.edge_weight < float("inf"):
        raise ValueError("weight-decay and edge-weight must be finite and non-negative.")
    if args.train_split == args.val_split:
        raise ValueError("Training and validation splits must be different.")
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA.")
    seed_everything(args.seed)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    train_data = ChangeDetectionDataset(args.data_root, args.train_split, augment=args.augment)
    val_data = ChangeDetectionDataset(args.data_root, args.val_split)
    if (args.data_root / args.train_split).samefile(args.data_root / args.val_split):
        raise ValueError("Training and validation must use distinct physical directories.")
    if args.resume and args.output_dir.resolve() != args.resume.resolve().parent:
        raise ValueError(
            "Resume must use the checkpoint's existing --output-dir "
            "to preserve best.pt and history."
        )
    if args.resume and args.resume.name != "last.pt":
        raise ValueError("Resume from last.pt; best.pt is intended for evaluation and prediction.")
    if args.output_dir.exists() and not args.resume and any(args.output_dir.iterdir()):
        raise FileExistsError(
            "Output directory is not empty. Choose a new --output-dir or use --resume."
        )
    backbone_weights = args.backbone_weights
    if not args.resume and not args.from_scratch and backbone_weights is None:
        backbone_weights = PROJECT_ROOT / "pretrained/pvt_v2_b0.pth"
    if backbone_weights is not None and not backbone_weights.is_file():
        raise FileNotFoundError(
            f"Missing PVT weights: {backbone_weights}. See pretrained/README.md or --from-scratch."
        )
    model = M2MLINet(backbone_weights=backbone_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    criterion = ChangeDetectionLoss(edge_weight=args.edge_weight)
    generator = torch.Generator().manual_seed(args.seed)
    start_epoch, best_f1 = 0, -1.0
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "model": "M2MLINet",
            "output": "logits",
            "normalization": "[-1,1]",
            "lr_schedule": "constant",
        }
    )
    if args.resume:
        checkpoint = load_model_weights(model, args.resume)
        required = {"optimizer", "scaler", "epoch", "best_f1", "config", "rng", "loader_rng"}
        if checkpoint.get("format_version") != 1 or not required.issubset(checkpoint):
            raise ValueError("--resume requires a complete standardized training checkpoint.")
        # Epoch count and file locations may change when extending/moving a run.
        for key in (
            "batch_size",
            "lr",
            "weight_decay",
            "edge_weight",
            "seed",
            "augment",
            "amp",
            "deterministic",
            "train_split",
            "val_split",
            "num_workers",
        ):
            if checkpoint["config"].get(key) != config[key]:
                raise ValueError(
                    f"Resume configuration differs for {key}; use the original setting."
                )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_f1 = checkpoint["epoch"], checkpoint["best_f1"]
        restore_rng(checkpoint["rng"])
        generator.set_state(checkpoint["loader_rng"])
    if start_epoch >= args.epochs:
        raise ValueError("Checkpoint already reached the requested epoch count.")
    train_loader = _loader(train_data, args, device, shuffle=True, generator=generator)
    # Use a separate generator: validation worker seeds cannot alter training RNG.
    val_loader = _loader(val_data, args, device, generator=torch.Generator().manual_seed(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "config.json", config)
    logger = logging.getLogger("m2m_linet.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(args.output_dir / "train.log", encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    try:
        logger.info(
            "device=%s train=%d val=%d split=%s epochs=%d batch=%d",
            device,
            len(train_data),
            len(val_data),
            args.val_split,
            args.epochs,
            args.batch_size,
        )
        for epoch in range(start_epoch + 1, args.epochs + 1):
            train = run_epoch(model, train_loader, device, criterion, optimizer, scaler, args.amp)
            validation = run_epoch(model, val_loader, device, criterion)
            improved = validation["f1"] > best_f1
            best_f1 = max(best_f1, validation["f1"])
            row = {"epoch": epoch, "train": train, "validation": validation, "best_f1": best_f1}
            with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            save_checkpoint(
                args.output_dir / "last.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_f1,
                config,
                generator,
            )
            if improved:
                save_checkpoint(
                    args.output_dir / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_f1,
                    config,
                    generator,
                )
            logger.info(
                "epoch=%d/%d loss=%.5f val_loss=%.5f val_f1=%.6f val_iou=%.6f best_f1=%.6f",
                epoch,
                args.epochs,
                train["loss"],
                validation["loss"],
                validation["f1"],
                validation["iou"],
                best_f1,
            )
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def evaluate_main(argv=None):
    args = build_parser("evaluate").parse_args(argv)
    from .checkpoint import load_model_weights
    from .data import ChangeDetectionDataset
    from .engine import run_epoch
    from .losses import ChangeDetectionLoss
    from .models import M2MLINet

    device = resolve_device(args.device)
    dataset = ChangeDetectionDataset(args.data_root, args.split)
    model = M2MLINet().to(device)
    checkpoint = load_model_weights(model, args.checkpoint)
    edge_weight = checkpoint.get("config", {}).get("edge_weight", 1.0)
    scores = run_epoch(
        model, _loader(dataset, args, device), device, ChangeDetectionLoss(edge_weight=edge_weight)
    )
    result = {"split": args.split, "samples": len(dataset), "threshold": 0.5, "metrics": scores}
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.metrics_output:
        _write_json(args.metrics_output, result)


def predict_main(argv=None):
    args = build_parser("predict").parse_args(argv)
    if not 0 < args.threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")
    from .checkpoint import load_model_weights
    from .data import ChangeDetectionDataset
    from .engine import predict_masks
    from .models import M2MLINet

    device = resolve_device(args.device)
    dataset = ChangeDetectionDataset(args.data_root, args.split, require_labels=False)
    model = M2MLINet().to(device)
    load_model_weights(model, args.checkpoint)
    count = predict_masks(
        model,
        _loader(dataset, args, device),
        device,
        args.output_dir,
        threshold=args.threshold,
        save_probabilities=args.save_probabilities,
    )
    print(f"Saved {count} masks to {args.output_dir.resolve()}")
