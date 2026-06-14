"""Snapshot-ensemble CV — does averaging the saturated-val plateau epochs beat
the single best-epoch pick?

Motivation (journal iters 3–5): val saturates at acc=1.0 by ~epoch 9, then a
whole *plateau* of epochs each sit at val=1.0 but generalise differently to the
tiny test fold. The loop picks ONE of them (``val_acc_loss``: lowest val-loss)
and that single pick is the documented source of epoch-selection variance. EMA
(iter 3) tried to average them but HURT — because ``AveragedModel(use_buffers)``
also averaged BN running stats on a short run. This experiment averages the
*outputs* (softmax) of the model at its best-N plateau epochs instead — no BN
averaging, the textbook fix for snapshot variance.

Design = the cv_eval pattern: train the full budget ONCE per fold (training path
byte-identical to ``cv_eval._train_fold`` so determinism is preserved), but
record per-epoch test **softmax** (not just argmax). Then *offline*, for the
canonical ``val_acc_loss`` selection:
  - find the production-selected epoch e* and the early-stop window it lived in;
  - rank the window's epochs by the val_acc_loss metric (e* is rank 1);
  - ensemble = average softmax over the top-N of them, argmax, pool over folds.
N=1 == the single production pick → must reproduce the iter-5 CV 0.9864 exactly
(a determinism sanity check), so any N>1 delta is a real ensemble effect, not
training drift. No test label ever feeds selection → leak-free.

Usage:
    .venv/bin/python -m experiments.snapshot_eval --k 5 --seed 42 --tag snap
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

STRATEGY = "val_acc_loss"          # the canonical selection (journal iter 5)
ENSEMBLE_SIZES = [1, 2, 3, 5]      # N=1 is the single-pick sanity baseline


@torch.no_grad()
def _eval_collect(model, loader, criterion, device, want_probs=False):
    """No-grad eval pass → (loss, acc, y_pred, y_true[, probs])."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    y_pred, y_true, probs = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss_sum += criterion(out, labels).item() * images.size(0)
        pred = out.argmax(1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
        y_pred.extend(pred.cpu().tolist())
        y_true.extend(labels.cpu().tolist())
        if want_probs:
            probs.extend(out.softmax(1).cpu().tolist())
    acc = correct / max(total, 1)
    if want_probs:
        return loss_sum / max(total, 1), acc, y_pred, y_true, probs
    return loss_sum / max(total, 1), acc, y_pred, y_true


def _train_fold(cfg, device, splits):
    """Train one fold the FULL budget, recording per-epoch val_acc/val_loss and
    the test-fold **softmax** at every epoch. Training path mirrors
    cv_eval._train_fold exactly (same model/optim/sched/seed) → determinism."""
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
        te_loss, te_acc, te_pred, te_true, te_probs = _eval_collect(
            model, loaders["test"], criterion, device, want_probs=True)
        if y_true_ref is None:
            y_true_ref = te_true
        epochs_log.append({"va_acc": va_acc, "va_loss": va_loss,
                           "te_acc": te_acc, "te_pred": te_pred, "te_probs": te_probs})
    return epochs_log, y_true_ref


def _metric(e):
    return T._selection_metric(e["va_acc"], e["va_loss"], e["va_acc"], e["va_loss"],
                               True, strategy=STRATEGY)


def _select_window(epochs_log, patience):
    """Replay production selection + early-stop. Return (best_idx, stop_idx):
    the checkpointed epoch and the last epoch the run executed before stopping.
    The ensemble pool is epochs[0..stop_idx] — exactly what production trained."""
    best_metric, best_idx, no_improve, stop_idx = -1.0, 0, 0, len(epochs_log) - 1
    for idx, e in enumerate(epochs_log):
        if _metric(e) > best_metric:
            best_metric, best_idx, no_improve = _metric(e), idx, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                stop_idx = idx
                break
    return best_idx, stop_idx


def _ensemble_pred(epochs_log, idxs):
    """Average softmax over the given epoch indices → argmax per test image."""
    import numpy as np
    stack = np.mean([np.asarray(epochs_log[i]["te_probs"]) for i in idxs], axis=0)
    return stack.argmax(1).tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="snapshot")
    p.add_argument("--epochs", type=int, default=None, help="override epoch budget (smoke tests)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Config.load()
    if args.epochs is not None:
        base.train.epochs = args.epochs
    patience = base.train.early_stopping_patience
    samples = ImageFolder(str(base.raw_dir)).samples
    folds = component_kfold(samples, args.k, seed=args.seed)
    print(f"Snapshot CV | device={device} | backbone={base.model.backbone} | k={args.k} "
          f"| seed={args.seed} | fold sizes={[len(f) for f in folds]} | epochs={base.train.epochs}")

    fold_logs, fold_true = [], []
    t0 = time.time()
    for i in range(args.k):
        test_idx = folds[i]
        val_idx = folds[(i + 1) % args.k]
        train_idx = [ix for j in range(args.k) if j not in (i, (i + 1) % args.k) for ix in folds[j]]
        cfg = Config.load()
        cfg.data.seed = args.seed
        cfg.train.epochs = base.train.epochs
        T.set_seed(args.seed)
        elog, ytrue = _train_fold(cfg, device, (train_idx, val_idx, test_idx))
        fold_logs.append(elog)
        fold_true.append(ytrue)
        best_idx, stop_idx = _select_window(elog, patience)
        print(f"  fold {i}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
              f"| pick=ep{best_idx+1} window=ep1..{stop_idx+1} | {time.time()-t0:.0f}s")
        if device == "cuda":
            torch.cuda.empty_cache()

    # Offline: snapshot ensembles of size N over the best plateau epochs.
    results = {}
    for n in ENSEMBLE_SIZES:
        pooled_pred, pooled_true, per_fold = [], [], []
        for i in range(args.k):
            best_idx, stop_idx = _select_window(fold_logs[i], patience)
            window = list(range(0, stop_idx + 1))
            ranked = sorted(window, key=lambda ix: _metric(fold_logs[i][ix]), reverse=True)
            chosen = ranked[:min(n, len(ranked))]
            pred = _ensemble_pred(fold_logs[i], chosen)
            true = fold_true[i]
            pooled_pred.extend(pred)
            pooled_true.extend(true)
            per_fold.append({"acc": round(accuracy_score(true, pred), 4),
                             "f1": round(f1_score(true, pred, average="macro", zero_division=0), 4),
                             "n_ens": len(chosen)})
        accs = [f["acc"] for f in per_fold]
        std = statistics.stdev if len(accs) > 1 else (lambda x: 0.0)
        results[f"ensemble_{n}"] = {
            "pooled_acc": round(accuracy_score(pooled_true, pooled_pred), 4),
            "pooled_macro_f1": round(f1_score(pooled_true, pooled_pred, average="macro", zero_division=0), 4),
            "per_fold_acc_std": round(std(accs), 4),
            "per_fold": per_fold,
        }

    base_acc = results["ensemble_1"]["pooled_acc"]
    summary = {
        "tag": args.tag, "backbone": base.model.backbone, "k": args.k, "seed": args.seed,
        "n_images": len(samples), "strategy": STRATEGY, "epochs_budget": base.train.epochs,
        "patience": patience, "elapsed_s": round(time.time() - t0, 1),
        "ensembles": results,
    }
    print("\n=== SNAPSHOT CV ===")
    print(json.dumps(summary, indent=2))
    print(f"sanity: ensemble_1 pooled_acc={base_acc} (expect 0.9864 from iter-5 CV)")
    print("=== END SNAPSHOT CV ===")

    out = ROOT / "experiments" / "snapshot_results.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(f"appended to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
