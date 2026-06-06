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
    """Seed Python, NumPy and torch, and enable deterministic kernels.

    Reproducibility is a hard project guardrail. cuDNN's default autotuner picks
    nondeterministic algorithms, which — feeding a *saturated* val set — made the
    checkpointed epoch (and therefore the test metric) swing run-to-run between
    ~0.978 and 1.000 for an identical config+seed (see journal iter 3). We trade
    the autotuner for deterministic kernels; the small speed cost is worth a
    metric we can actually trust. ``warn_only=True`` keeps any op lacking a
    deterministic implementation from hard-crashing the run.
    """
    import os
    import random

    import numpy as np

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # PYTHONHASHSEED only takes effect at interpreter startup, so this fixes hash
    # randomization for child processes spawned *after* here (e.g. spawn-mode
    # DataLoader workers), not this process — which the explicit seeds below cover.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def run_epoch(model, loader, criterion, optimizer, device, train: bool, ema_model=None):
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
            if ema_model is not None:
                ema_model.update_parameters(model)
        loss_sum += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_model(cfg, device, verbose: bool = True) -> dict:
    """Train per ``cfg`` on ``device``, saving the best checkpoint to
    ``cfg.checkpoint_path``. Returns a result dict (history, class_names,
    best_metric, best_epoch, timings).

    Pure training only: the caller owns ``set_seed()`` and any artifact writing
    (history JSON / plots). Extracted from ``main()`` so the multi-seed evaluator
    can drive the *real* training loop instead of a copy.
    """
    loaders, class_names = build_dataloaders(cfg)
    n_train = len(loaders["train"].dataset)
    n_val = len(loaders["val"].dataset)
    has_val = n_val > 0
    if verbose:
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

    # Optional EMA of weights: a separate averaged copy of the model that often
    # generalises / calibrates better than the raw SGD iterate. When enabled we
    # select on and serve the EMA weights. use_buffers=True also averages the
    # BatchNorm running stats so the served model is self-consistent.
    ema_model = None
    if getattr(cfg.train, "ema", False):
        from torch.optim.swa_utils import AveragedModel

        decay = cfg.train.ema_decay

        def _ema_avg(avg_p, cur_p, _num):
            return decay * avg_p + (1.0 - decay) * cur_p

        ema_model = AveragedModel(model, avg_fn=_ema_avg, use_buffers=True)
        if verbose:
            print(f"EMA enabled (decay={decay}); selecting + serving the EMA weights.")

    history = []
    # Start below any real accuracy so the first epoch always checkpoints --
    # this also guarantees a model is saved even when there is no val split.
    best_metric, best_epoch, epochs_no_improve = -1.0, 0, 0
    start = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, True, ema_model=ema_model
        )
        # Validate (and later serve) the EMA weights when EMA is enabled.
        eval_model = ema_model.module if ema_model is not None else model
        va_loss, va_acc = run_epoch(eval_model, loaders["val"], criterion, optimizer, device, False)
        scheduler.step()
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
             "val_loss": va_loss, "val_acc": va_acc}
        )
        if verbose:
            print(
                f"Epoch {epoch:02d}/{cfg.train.epochs} | "
                f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                f"val loss {va_loss:.3f} acc {va_acc:.3f}"
            )

        # Checkpoint-selection metric (higher=better). Strategy is config-driven
        # so iterations stay comparable: "val_acc" = original behaviour;
        # "val_acc_loss" additionally breaks the saturated-val plateau by loss.
        metric = _selection_metric(
            va_acc, va_loss, tr_acc, tr_loss, has_val,
            strategy=getattr(cfg.train, "select_on", "val_acc"),
        )
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                cfg.checkpoint_path, eval_model, class_names, cfg,
                extra={"val_acc": va_acc, "train_acc": tr_acc, "epoch": epoch},
            )
            if verbose:
                label = "val" if has_val else "train"
                print(f"   ↳ saved new best model ({label} acc {va_acc if has_val else tr_acc:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement).")
                break

    elapsed = time.time() - start
    return {
        "history": history,
        "class_names": class_names,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "n_train": n_train,
        "n_val": n_val,
        "elapsed": elapsed,
    }


def _selection_metric(va_acc, va_loss, tr_acc, tr_loss, has_val, strategy="val_acc") -> float:
    """Single scalar (higher=better) used to pick the checkpoint.

    ``val_acc`` (default): accuracy only — the original behaviour.
    ``val_acc_loss``: accuracy minus a small ``loss`` term, so on a saturated val
    set (many epochs at acc=1.0) we keep the *lowest-loss* (best-calibrated) one
    instead of an arbitrary first-to-plateau epoch.
    """
    acc, loss = (va_acc, va_loss) if has_val else (tr_acc, tr_loss)
    if strategy == "val_acc_loss":
        return acc - 1e-3 * loss
    return acc


def main() -> None:
    args = parse_args()
    cfg = Config.load(overrides=args_to_overrides(args))
    set_seed(cfg.data.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    result = train_model(cfg, device, verbose=True)

    label = "val" if result["n_val"] > 0 else "train"
    print(f"\nDone in {result['elapsed']/60:.1f} min. Best {label} sel-metric: "
          f"{result['best_metric']:.4f} (epoch {result['best_epoch']})")
    print(f"Best model: {cfg.checkpoint_path}")

    # Persist training history + class names for later inspection / plotting.
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(result["history"], indent=2))
    (out_dir / "class_names.json").write_text(json.dumps(result["class_names"], indent=2))
    _maybe_plot(result["history"], out_dir)


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
