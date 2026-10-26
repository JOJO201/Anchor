"""Small, auditable utilities for fixed train-only anchor banks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def select_anchor_indices(num_items: int, bank_size: int, seed: int) -> list[int]:
    """Select a deterministic subset without touching global RNG state."""

    if num_items <= 0:
        raise ValueError("num_items must be positive")
    if not 1 <= bank_size <= num_items:
        raise ValueError("bank_size must be in [1, num_items]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(num_items, generator=generator)[:bank_size].tolist()


@dataclass(frozen=True)
class AnchorBank:
    """In-memory waveforms selected exclusively from the training split."""

    waveforms: Tensor
    sample_ids: tuple[str, ...]
    source_split: str = "train"

    def __post_init__(self) -> None:
        if self.source_split != "train":
            raise ValueError("AnchorADD anchors must come from the training split")
        if not isinstance(self.waveforms, Tensor) or self.waveforms.ndim < 2:
            raise ValueError("waveforms must be shaped [K,...]")
        if self.waveforms.shape[0] < 2:
            raise ValueError("an anchor bank must contain at least two samples")
        if self.waveforms.shape[0] != len(self.sample_ids):
            raise ValueError("sample_ids must match the anchor count")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique")
        if not self.waveforms.is_floating_point():
            raise ValueError("anchor waveforms must be floating point")
        if not torch.isfinite(self.waveforms).all():
            raise ValueError("anchor waveforms must be finite")

    def __len__(self) -> int:
        return int(self.waveforms.shape[0])

    def one(self, index: int, *, device: torch.device | str) -> Tensor:
        return self.waveforms[index : index + 1].to(device=device)

    def fixed(self, count: int, *, device: torch.device | str) -> Tensor:
        if not 2 <= count <= len(self):
            raise ValueError("count must be between 2 and the bank size")
        return self.waveforms[:count].to(device=device)
