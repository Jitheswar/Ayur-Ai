"""Transfer-learning model builder and checkpoint I/O.

A pretrained ImageNet backbone (ResNet / MobileNet / EfficientNet) has its
classifier head replaced by a small head sized to the number of plant classes.
Optionally the backbone can be frozen so only the head trains.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_BACKBONES = ["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"]


def build_model(
    backbone: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """Create a CNN with a fresh classification head for ``num_classes``."""
    backbone = backbone.lower()
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone {backbone!r}. Choose from {SUPPORTED_BACKBONES}."
        )

    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = net.fc.in_features
        net.fc = _head(in_features, num_classes, dropout)
        feature_params = (p for n, p in net.named_parameters() if not n.startswith("fc."))
    elif backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = net.fc.in_features
        net.fc = _head(in_features, num_classes, dropout)
        feature_params = (p for n, p in net.named_parameters() if not n.startswith("fc."))
    elif backbone == "mobilenet_v2":
        net = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = _head(in_features, num_classes, dropout)
        feature_params = (p for n, p in net.named_parameters() if not n.startswith("classifier."))
    else:  # efficientnet_b0
        net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = _head(in_features, num_classes, dropout)
        feature_params = (p for n, p in net.named_parameters() if not n.startswith("classifier."))

    if freeze_backbone:
        for p in feature_params:
            p.requires_grad = False

    return net


def _head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    class_names: list[str],
    cfg,
    extra: dict | None = None,
) -> None:
    """Persist weights together with everything needed to rebuild the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "backbone": cfg.model.backbone,
        "image_size": cfg.data.image_size,
        "dropout": cfg.model.dropout,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Rebuild a model from a checkpoint. Returns (model, class_names, meta)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Train one first: python -m src.train"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = build_model(
        backbone=ckpt["backbone"],
        num_classes=len(class_names),
        pretrained=False,
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    meta = {"image_size": ckpt.get("image_size", 224), "backbone": ckpt["backbone"]}
    return model, class_names, meta
