from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IDSConfig:
    label_col: str = "label"
    positive_label: str = "attack"
    negative_label: str = "normal"
    random_state: int = 42


DEFAULT_CONFIG = IDSConfig()

