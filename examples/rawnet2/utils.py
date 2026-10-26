"""Shared helpers for the RawNet2 training and evaluation entry points."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from anchoradd import AnchorBank, select_anchor_indices

from .data import AudioManifestDataset


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_anchor_bank(
    dataset: AudioManifestDataset,
    *,
    bank_size: int,
    seed: int,
) -> AnchorBank:
    indices = select_anchor_indices(len(dataset), bank_size, seed)
    rows = [dataset[index] for index in indices]
    return AnchorBank(
        waveforms=torch.stack([row["waveform"] for row in rows]).float(),
        sample_ids=tuple(str(row["sample_id"]) for row in rows),
    )
