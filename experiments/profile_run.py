"""Reusable train + evaluate + profile harness for the experiment loop.

Runs the project's *actual* training and evaluation code paths (so any change
to src/ is reflected), then reports a consistent block of metrics:

    * test accuracy + macro-F1 on the held-out test split (the METRIC)
    * wall-clock training time
    * peak GPU memory (allocated + reserved) during training
    * peak host RAM (self + worker children)
    * single-image inference latency on GPU and CPU (warm, averaged)

Everything is driven by config.yaml + a fixed seed, so results are comparable
across iterations. Usage:

    .venv/bin/python -m experiments.profile_run            # defaults from config
    .venv/bin/python -m experiments.profile_run --skip-train   # eval existing ckpt

A JSON summary is appended to experiments/runs.jsonl and also printed between
clear === PROFILE === markers for easy log scraping.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch

from src.config import ROOT, Config
from src.data import build_dataloaders, inference_transform
from src.model import load_checkpoint


def _host_peak_gb() -> float:
    """Peak RSS (self + children) in GB. ru_maxrss is KiB on Linux."""
    self_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (self_kb + child_kb) / (1024 * 1024)


@torch.no_grad()
def _latency_ms(model, image_size: int, device: str, n: int = 50, warmup: int = 10) -> float:
    """Average single-image forward latency (transform excluded) in ms."""
    model.eval().to(device)
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000.0


@torch.no_grad()
def _evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score

    y_true, y_pred = [], []
    for images, labels in loader:
        preds = model(images.to(device)).argmax(1).cpu()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, f1, len(y_true)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true", help="evaluate the existing checkpoint only")
    p.add_argument("--tag", default="run", help="label recorded in the JSONL summary")
    args = p.parse_args()

    cfg = Config.load()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_time = None
    peak_gpu_alloc = peak_gpu_reserved = None
    host_peak = None

    if not args.skip_train:
        # Import lazily so --skip-train never pays the training import cost.
        import src.train as train_mod

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        sys.argv = ["src.train"]          # use config defaults, no CLI overrides
        train_mod.main()
        train_time = time.time() - t0
        if device == "cuda":
            peak_gpu_alloc = torch.cuda.max_memory_allocated() / 1e9
            peak_gpu_reserved = torch.cuda.max_memory_reserved() / 1e9
        host_peak = _host_peak_gb()

    # ---- evaluate the (just-saved) checkpoint on the test split -------------
    model, class_names, meta = load_checkpoint(cfg.checkpoint_path, device)
    cfg.data.image_size = meta["image_size"]
    loaders, _ = build_dataloaders(cfg)
    acc, f1, n_test = _evaluate(model, loaders["test"], device)

    lat_gpu = _latency_ms(model, meta["image_size"], "cuda") if device == "cuda" else None
    lat_cpu = _latency_ms(model, meta["image_size"], "cpu")

    summary = {
        "tag": args.tag,
        "seed": cfg.data.seed,
        "backbone": meta["backbone"],
        "image_size": meta["image_size"],
        "test_acc": round(acc, 4),
        "test_macro_f1": round(f1, 4),
        "n_test": n_test,
        "train_time_s": round(train_time, 1) if train_time is not None else None,
        "peak_gpu_alloc_gb": round(peak_gpu_alloc, 3) if peak_gpu_alloc is not None else None,
        "peak_gpu_reserved_gb": round(peak_gpu_reserved, 3) if peak_gpu_reserved is not None else None,
        "host_peak_gb": round(host_peak, 2) if host_peak is not None else None,
        "latency_gpu_ms": round(lat_gpu, 2) if lat_gpu is not None else None,
        "latency_cpu_ms": round(lat_cpu, 2),
    }

    out = ROOT / "experiments" / "runs.jsonl"
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary) + "\n")

    print("\n=== PROFILE ===")
    print(json.dumps(summary, indent=2))
    print("=== END PROFILE ===")


if __name__ == "__main__":
    main()
