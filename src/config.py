"""Typed configuration loaded from ``config.yaml`` with CLI overrides.

Usage::

    from src.config import Config
    cfg = Config.load()                         # read config.yaml
    cfg = Config.load(overrides={"train.epochs": 30})

Every nested field is reachable as ``cfg.train.epochs`` etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Project root = the directory that contains this package's parent.
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class DataCfg:
    raw_dir: str = "data/raw"
    image_size: int = 224
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42


@dataclass
class ModelCfg:
    backbone: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout: float = 0.2


@dataclass
class TrainCfg:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    early_stopping_patience: int = 5
    checkpoint_path: str = "models/best_model.pt"


@dataclass
class NlpCfg:
    knowledge_base: str = "data/knowledge_base/plants.json"


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    nlp: NlpCfg = field(default_factory=NlpCfg)

    # -- absolute-path helpers -------------------------------------------
    def abspath(self, relative: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(relative)
        return p if p.is_absolute() else (ROOT / p)

    @property
    def raw_dir(self) -> Path:
        return self.abspath(self.data.raw_dir)

    @property
    def checkpoint_path(self) -> Path:
        return self.abspath(self.train.checkpoint_path)

    @property
    def knowledge_base_path(self) -> Path:
        return self.abspath(self.nlp.knowledge_base)

    # -- construction ----------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_CONFIG_PATH,
        overrides: dict[str, Any] | None = None,
    ) -> "Config":
        raw: dict[str, Any] = {}
        path = Path(path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}

        cfg = cls(
            data=_from_dict(DataCfg, raw.get("data", {})),
            model=_from_dict(ModelCfg, raw.get("model", {})),
            train=_from_dict(TrainCfg, raw.get("train", {})),
            nlp=_from_dict(NlpCfg, raw.get("nlp", {})),
        )
        if overrides:
            cfg.apply_overrides(overrides)
        return cfg

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply dotted-key overrides like ``{"train.epochs": 30}``."""
        for dotted, value in overrides.items():
            if value is None:
                continue
            section, _, key = dotted.partition(".")
            target = getattr(self, section, None)
            if target is None or not hasattr(target, key):
                raise KeyError(f"Unknown config key: {dotted!r}")
            setattr(target, key, value)


def _from_dict(dc_type, data: dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys."""
    assert is_dataclass(dc_type)
    known = {f.name for f in fields(dc_type)}
    filtered = {k: v for k, v in (data or {}).items() if k in known}
    return dc_type(**filtered)
