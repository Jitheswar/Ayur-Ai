"""Multi-seed evaluation — how good is the model REALLY, and how noisy is the metric?

Iteration 3 showed the single-split test number is noise-limited (1 test error =
0.0036, and an identical config+seed gave 0.9964 then 0.9783). This harness
retrains the *current* config from scratch under several seeds — each seed
reshuffles the dedup component split and re-inits the classifier head — and
reports mean ± std of held-out test accuracy and macro-F1. That distribution,
not a single lucky draw, is the honest metric.

With determinism now enabled (src.train.set_seed), each seed is individually
reproducible, so the spread reported here is genuine split/init variance.

Usage:
    .venv/bin/python -m experiments.multiseed_eval --seeds 42,0,1,2,3 --tag mseed_resnet18
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from src.config import ROOT, Config
from src.data import build_dataloaders
from src.model import load_checkpoint
import src.train as T


@torch.no_grad()
def _eval_test(cfg, device):
    from sklearn.metrics import accuracy_score, f1_score

    model, class_names, meta = load_checkpoint(cfg.checkpoint_path, device)
    cfg.data.image_size = meta["image_size"]
    loaders, _ = build_dataloaders(cfg)
    y_true, y_pred = [], []
    for images, labels in loaders["test"]:
        y_pred.extend(model(images.to(device)).argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1, len(y_true)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,0,1,2,3", help="comma-separated integer seeds")
    p.add_argument("--tag", default="mseed")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Config.load()
    print(f"Multiseed eval | device={device} | backbone={base.model.backbone} | "
          f"select_on={base.train.select_on} | ema={base.train.ema} | seeds={seeds}")

    rows = []
    for seed in seeds:
        cfg = Config.load()
        cfg.data.seed = seed
        T.set_seed(seed)
        t0 = time.time()
        res = T.train_model(cfg, device, verbose=False)
        acc, f1, n = _eval_test(cfg, device)
        dt = time.time() - t0
        rows.append({
            "seed": seed, "test_acc": round(acc, 4), "test_macro_f1": round(f1, 4),
            "n_test": n, "best_epoch": res["best_epoch"], "train_s": round(dt, 1),
        })
        print(f"  seed {seed:>3}: test_acc={acc:.4f}  macro_f1={f1:.4f}  "
              f"n_test={n}  best_ep={res['best_epoch']}  ({dt:.0f}s)")
        if device == "cuda":
            torch.cuda.empty_cache()

    accs = [r["test_acc"] for r in rows]
    f1s = [r["test_macro_f1"] for r in rows]
    stdev = statistics.stdev if len(accs) > 1 else (lambda x: 0.0)
    summary = {
        "tag": args.tag,
        "backbone": base.model.backbone,
        "select_on": base.train.select_on,
        "ema": base.train.ema,
        "seeds": seeds,
        "test_acc_mean": round(statistics.mean(accs), 4),
        "test_acc_std": round(stdev(accs), 4),
        "test_acc_min": min(accs), "test_acc_max": max(accs),
        "macro_f1_mean": round(statistics.mean(f1s), 4),
        "macro_f1_std": round(stdev(f1s), 4),
        "macro_f1_min": min(f1s), "macro_f1_max": max(f1s),
        "per_seed": rows,
    }
    print("\n=== MULTISEED ===")
    print(json.dumps(summary, indent=2))
    print("=== END MULTISEED ===")

    out = ROOT / "experiments" / "multiseed.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(f"appended to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
