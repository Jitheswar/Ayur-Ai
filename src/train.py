"""Train the leaf classifier with transfer learning.

Examples
--------
    python -m src.train
    python -m src.train --backbone efficientnet_b0 --epochs 30 --lr 1e-4
    python -m src.train --freeze-backbone          # train only the head (fast)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import ROOT, Config
from .data import build_dataloaders
from .model import SUPPORTED_BACKBONES, build_model, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Ayurvedic leaf classifier.")
    p.add_argument("--backbone", choices=SUPPORTED_BACKBONES)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--image-size", type=int)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--device", default=None, help="cuda | cpu (auto-detected if unset)")
    return p.parse_args()


def args_to_overrides(args: argparse.Namespace) -> dict:
    # Use ``is not None`` so explicit zeros (lr=0, epochs=0) are not dropped.
    o = {}
    if args.backbone is not None:
        o["model.backbone"] = args.backbone
    if args.freeze_backbone:  # store_true flag: only set when passed
        o["model.freeze_backbone"] = True
    if args.epochs is not None:
        o["train.epochs"] = args.epochs
    if args.lr is not None:
        o["train.lr"] = args.lr
    if args.batch_size is not None:
        o["data.batch_size"] = args.batch_size
    if args.image_size is not None:
        o["data.image_size"] = args.image_size
    return o


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) for reproducible runs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    desc = "train" if train else "val  "
    torch.set_grad_enabled(train)
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    cfg = Config.load(overrides=args_to_overrides(args))
    set_seed(cfg.data.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, class_names = build_dataloaders(cfg)
    n_train = len(loaders["train"].dataset)
    n_val = len(loaders["val"].dataset)
    has_val = n_val > 0
    print(f"Classes: {len(class_names)} | train={n_train} val={n_val}")
    if not has_val:
        print("No validation split -- selecting the best model on train accuracy.")

    model = build_model(
        backbone=cfg.model.backbone,
        num_classes=len(class_names),
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        dropout=cfg.model.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    history = []
    # Start below any real accuracy so the first epoch always checkpoints --
    # this also guarantees a model is saved even when there is no val split.
    best_metric, epochs_no_improve = -1.0, 0
    start = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        scheduler.step()
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
             "val_loss": va_loss, "val_acc": va_acc}
        )
        print(
            f"Epoch {epoch:02d}/{cfg.train.epochs} | "
            f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.3f} acc {va_acc:.3f}"
        )

        # Select on val accuracy when a val split exists, else on train accuracy.
        metric = va_acc if has_val else tr_acc
        if metric > best_metric:
            best_metric = metric
            epochs_no_improve = 0
            save_checkpoint(
                cfg.checkpoint_path, model, class_names, cfg,
                extra={"val_acc": va_acc, "train_acc": tr_acc, "epoch": epoch},
            )
            label = "val" if has_val else "train"
            print(f"   ↳ saved new best model ({label} acc {metric:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement).")
                break

    elapsed = time.time() - start
    label = "val" if has_val else "train"
    print(f"\nDone in {elapsed/60:.1f} min. Best {label} acc: {best_metric:.3f}")
    print(f"Best model: {cfg.checkpoint_path}")

    # Persist training history + class names for later inspection / plotting.
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    _maybe_plot(history, out_dir)


def _maybe_plot(history, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, [h["train_acc"] for h in history], label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    print(f"Saved training curves to {out_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
