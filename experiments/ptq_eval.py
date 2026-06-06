"""INT8 post-training quantization (PTQ) — the compression lever (journal iter 9).

Eval-only and CPU-only, so it is immune to the GPU thermal throttling that made
training-based experiments expensive late in the loop. Measures the *quality
cost* of INT8 static quantization against the fp32 served checkpoint, plus the
on-device payoff (model size on disk, single-image CPU latency).

Static (not dynamic) PTQ via FX graph mode: dynamic quant only touches Linear
layers, but a CNN's compute is in the convs — static quant with a calibration
pass quantizes conv+linear, which is the meaningful comparison here.

Usage:
    .venv/bin/python -m experiments.ptq_eval
"""

from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from src.config import Config
from src.data import build_dataloaders
from src.model import load_checkpoint


@torch.no_grad()
def _eval(model, loader):
    y_true, y_pred = [], []
    for images, labels in loader:
        y_pred.extend(model(images).argmax(1).tolist())
        y_true.extend(labels.tolist())
    return (round(accuracy_score(y_true, y_pred), 4),
            round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            len(y_true))


def _size_mb(model) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        torch.save(model.state_dict(), f.name)
        return round(Path(f.name).stat().st_size / 1e6, 2)


@torch.no_grad()
def _cpu_latency_ms(model, n=50) -> float:
    x = torch.randn(1, 3, 224, 224)
    for _ in range(5):  # warmup
        model(x)
    t0 = time.time()
    for _ in range(n):
        model(x)
    return round((time.time() - t0) / n * 1000, 2)


def main() -> None:
    torch.backends.quantized.engine = "x86"
    device = "cpu"  # quantized inference is CPU-only; keep fp32 on CPU too for a fair latency compare
    cfg = Config.load()

    model_fp32, class_names, meta = load_checkpoint(cfg.checkpoint_path, device)
    cfg.data.image_size = meta["image_size"]
    model_fp32.eval()
    loaders, _ = build_dataloaders(cfg)

    # ---- fp32 baseline ----
    acc32, f1_32, n_test = _eval(model_fp32, loaders["test"])
    size32, lat32 = _size_mb(model_fp32), _cpu_latency_ms(model_fp32)
    print(f"fp32  : test_acc={acc32}  macro_f1={f1_32}  size={size32}MB  cpu_lat={lat32}ms  (n_test={n_test})")

    # ---- INT8 static PTQ (FX graph mode) ----
    qmap = get_default_qconfig_mapping("x86")
    example = torch.randn(1, 3, 224, 224)
    prepared = prepare_fx(copy.deepcopy(model_fp32), qmap, example_inputs=example)
    # Calibrate on a few hundred training images (collect activation ranges).
    with torch.no_grad():
        seen = 0
        for images, _ in loaders["train"]:
            prepared(images)
            seen += images.size(0)
            if seen >= 256:
                break
    model_int8 = convert_fx(prepared)
    model_int8.eval()

    acc8, f1_8, _ = _eval(model_int8, loaders["test"])
    size8, lat8 = _size_mb(model_int8), _cpu_latency_ms(model_int8)
    print(f"int8  : test_acc={acc8}  macro_f1={f1_8}  size={size8}MB  cpu_lat={lat8}ms  (calib_imgs={seen})")

    print("\n=== PTQ TRADEOFF (int8 vs fp32) ===")
    print(f"  quality cost : Δacc={acc8-acc32:+.4f}  Δmacro_f1={f1_8-f1_32:+.4f}")
    print(f"  size         : {size32}MB -> {size8}MB  ({size8/size32:.2f}x, -{(1-size8/size32)*100:.0f}%)")
    print(f"  cpu latency  : {lat32}ms -> {lat8}ms  ({lat32/lat8:.2f}x speedup)")


if __name__ == "__main__":
    main()
