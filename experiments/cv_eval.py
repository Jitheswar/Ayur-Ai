"""Cross-validated evaluation — the canonical, low-variance metric for this loop.

Why this exists (journal iters 3–4): the single random split is noise-limited.
1 test error = 0.0036, and the per-split test_acc swung 0.872–1.000 across seeds
because near-duplicate *components* (kept together to avoid leakage) make test
errors correlated — the effective number of independent test units is the
component count, not the image count.

This harness removes that variance by **component K-fold cross-validation**:
every component is held out exactly once, so every image is predicted once by a
model that never saw its near-duplicates. The pooled accuracy / macro-F1 over all
folds is the honest number; per-fold mean±std shows the residual spread.

Fold rotation (K=5): for fold i, ``test = fold[i]``, ``val = fold[(i+1)%K]``,
``train = the other K-2 folds``. So each fold is the test set once and the val
set once → ~60/20/20 train/val/test, every image tested exactly once.

Selection A/B (free): each fold is trained the FULL epoch budget while recording
per-epoch (val_acc, val_loss) and the test-fold predictions at every epoch. Then,
*offline*, we replay the real production selection + early-stopping logic
(``src.train._selection_metric``) for each strategy and pool the predictions at
the epoch that strategy would have checkpointed. This compares
``select_on=val_acc`` vs ``val_acc_loss`` from ONE set of training runs, with no
test-set peeking ever feeding back into training.

Usage:
    .venv/bin/python -m experiments.cv_eval --k 5 --seed 42 --tag cv_resnet18
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torchvision.datasets import ImageFolder

from src.config import ROOT, Config
from src.data import build_dataloaders, component_kfold
from src.model import build_model
import src.train as T

STRATEGIES = ["val_acc", "val_acc_loss"]


@torch.no_grad()
def _eval_collect(model, loader, criterion, device):
    """No-grad eval pass → (loss, acc, y_pred, y_true)."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    y_pred, y_true = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss_sum += criterion(out, labels).item() * images.size(0)
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
        y_pred.extend(pred.cpu().tolist())
        y_true.extend(labels.cpu().tolist())
    return loss_sum / max(total, 1), correct / max(total, 1), y_pred, y_true


def _train_fold(cfg, device, splits):
    """Train one fold the FULL epoch budget (no early stop), recording per-epoch
    val_acc/val_loss and the test-fold predictions. Returns the per-epoch log and
    the fold's fixed test y_true."""
    loaders, class_names = build_dataloaders(cfg, splits=splits)
    model = build_model(
        backbone=cfg.model.backbone, num_classes=len(class_names),
        pretrained=cfg.model.pretrained, freeze_backbone=cfg.model.freeze_backbone,
        dropout=cfg.model.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    epochs_log = []
    y_true_ref = None
    for _epoch in range(1, cfg.train.epochs + 1):
        T.run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        scheduler.step()
        va_loss, va_acc, _, _ = _eval_collect(model, loaders["val"], criterion, device)
        te_loss, te_acc, te_pred, te_true = _eval_collect(model, loaders["test"], criterion, device)
        if y_true_ref is None:
            y_true_ref = te_true
        epochs_log.append({"va_acc": va_acc, "va_loss": va_loss,
                           "te_acc": te_acc, "te_pred": te_pred})
    return epochs_log, y_true_ref


def _select_epoch(epochs_log, strategy, patience):
    """Replay production selection + early-stopping over the recorded epochs and
    return the index (0-based) of the epoch this strategy would checkpoint."""
    best_metric, best_idx, no_improve = -1.0, 0, 0
    for idx, e in enumerate(epochs_log):
        # has_val=True; tr_* unused for these val strategies (pass va_* as filler)
        metric = T._selection_metric(e["va_acc"], e["va_loss"], e["va_acc"],
                                     e["va_loss"], True, strategy=strategy)
        if metric > best_metric:
            best_metric, best_idx, no_improve = metric, idx, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    return best_idx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="cv")
    p.add_argument("--epochs", type=int, default=None, help="override epoch budget (smoke tests)")
    p.add_argument("--label_smoothing", type=float, default=None, help="override train.label_smoothing")
    p.add_argument("--weight_decay", type=float, default=None, help="override train.weight_decay")
    p.add_argument("--lr", type=float, default=None, help="override train.lr")
    p.add_argument("--backbone", type=str, default=None, help="override model.backbone")
    args = p.parse_args()

    if args.k < 3:
        raise SystemExit("--k must be >= 3 (rotation uses 1 test + 1 val fold, "
                         "leaving k-2 >= 1 folds for training).")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _apply_overrides(cfg):
        """Apply optional hyperparameter overrides (None = leave config default)."""
        if args.epochs is not None:
            cfg.train.epochs = args.epochs
        if args.label_smoothing is not None:
            cfg.train.label_smoothing = args.label_smoothing
        if args.weight_decay is not None:
            cfg.train.weight_decay = args.weight_decay
        if args.lr is not None:
            cfg.train.lr = args.lr
        if args.backbone is not None:
            cfg.model.backbone = args.backbone
        return cfg

    base = _apply_overrides(Config.load())
    patience = base.train.early_stopping_patience
    samples = ImageFolder(str(base.raw_dir)).samples
    folds = component_kfold(samples, args.k, seed=args.seed)
    print(f"CV eval | device={device} | backbone={base.model.backbone} | k={args.k} "
          f"| seed={args.seed} | fold sizes={[len(f) for f in folds]} | epochs={base.train.epochs} "
          f"| lr={base.train.lr} | label_smoothing={base.train.label_smoothing} | weight_decay={base.train.weight_decay}")

    # Train every fold once, recording everything needed for offline selection A/B.
    fold_logs, fold_true = [], []
    t0 = time.time()
    for i in range(args.k):
        test_idx = folds[i]
        val_idx = folds[(i + 1) % args.k]
        train_idx = [ix for j in range(args.k) if j not in (i, (i + 1) % args.k) for ix in folds[j]]
        cfg = _apply_overrides(Config.load())
        cfg.data.seed = args.seed
        T.set_seed(args.seed)
        elog, ytrue = _train_fold(cfg, device, (train_idx, val_idx, test_idx))
        fold_logs.append(elog)
        fold_true.append(ytrue)
        print(f"  fold {i}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
              f"| last-epoch test_acc={elog[-1]['te_acc']:.4f} | {time.time()-t0:.0f}s elapsed")
        if device == "cuda":
            torch.cuda.empty_cache()

    # Offline: for each strategy, pick each fold's checkpoint epoch and pool.
    results = {}
    for strat in STRATEGIES:
        pooled_pred, pooled_true, per_fold = [], [], []
        sel_epochs = []
        for i in range(args.k):
            e = _select_epoch(fold_logs[i], strat, patience)
            sel_epochs.append(e + 1)  # 1-based for reporting
            pred = fold_logs[i][e]["te_pred"]
            true = fold_true[i]
            pooled_pred.extend(pred)
            pooled_true.extend(true)
            per_fold.append({
                "acc": round(accuracy_score(true, pred), 4),
                "f1": round(f1_score(true, pred, average="macro", zero_division=0), 4),
            })
        accs = [f["acc"] for f in per_fold]
        f1s = [f["f1"] for f in per_fold]
        std = statistics.stdev if len(accs) > 1 else (lambda x: 0.0)
        results[strat] = {
            "pooled_acc": round(accuracy_score(pooled_true, pooled_pred), 4),
            "pooled_macro_f1": round(f1_score(pooled_true, pooled_pred, average="macro", zero_division=0), 4),
            "per_fold_acc_mean": round(statistics.mean(accs), 4),
            "per_fold_acc_std": round(std(accs), 4),
            "per_fold_f1_mean": round(statistics.mean(f1s), 4),
            "selected_epochs": sel_epochs,
            "per_fold": per_fold,
        }

    summary = {
        "tag": args.tag, "backbone": base.model.backbone, "k": args.k, "seed": args.seed,
        "n_images": len(samples), "fold_sizes": [len(f) for f in folds],
        "epochs_budget": base.train.epochs, "patience": patience,
        "lr": base.train.lr,
        "label_smoothing": base.train.label_smoothing, "weight_decay": base.train.weight_decay,
        "elapsed_s": round(time.time() - t0, 1),
        "strategies": results,
    }
    print("\n=== CV ===")
    print(json.dumps(summary, indent=2))
    print("=== END CV ===")

    out = ROOT / "experiments" / "cv_results.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(f"appended to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
