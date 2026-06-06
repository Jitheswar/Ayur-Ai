"""Evaluate a trained checkpoint on the held-out test split.

    python -m src.evaluate
    python -m src.evaluate --checkpoint models/best_model.pt

Prints per-class precision/recall/F1 and writes a confusion-matrix image to
``outputs/confusion_matrix.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import ROOT, Config
from .data import build_dataloaders
from .model import load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the leaf classifier.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        preds = model(images).argmax(1).cpu()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred


def main() -> None:
    args = parse_args()
    cfg = Config.load()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_path

    model, class_names, meta = load_checkpoint(ckpt_path, device)
    # Evaluate at the resolution the model was trained on, not whatever
    # config.yaml currently says, so metrics aren't skewed by a size mismatch.
    cfg.data.image_size = meta["image_size"]
    loaders, data_classes = build_dataloaders(cfg)

    if data_classes != class_names:
        print(
            "WARNING: dataset classes differ from the checkpoint's classes. "
            "Make sure data/raw matches what the model was trained on."
        )

    y_true, y_pred = collect_predictions(model, loaders["test"], device)
    if not y_true:
        print("Test split is empty -- add more images or lower test_split.")
        return

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest accuracy: {acc:.4f}  (n={len(y_true)})\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    _plot_confusion(cm, class_names, ROOT / "outputs" / "confusion_matrix.png")


def _plot_confusion(cm, class_names, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(7, n * 0.5)))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    main()
