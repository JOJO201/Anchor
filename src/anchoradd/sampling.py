"""Stateless rolling-anchor decisions for reproducible training."""

from __future__ import annotations

import hashlib
import math


def _uniform(seed: int, epoch: int, step: int, stream: int) -> float:
    material = f"anchoradd|{seed}|{epoch}|{step}|{stream}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return integer / float(2**64)


def stateless_apply_decision(delta: float, seed: int, epoch: int, step: int) -> bool:
    """Return whether to use an anchored view for a mini-batch."""

    if not math.isfinite(float(delta)) or not 0.0 <= float(delta) <= 1.0:
        raise ValueError("delta must be in [0, 1]")
    return _uniform(seed, epoch, step, stream=0) < float(delta)


def stateless_anchor_index(bank_size: int, seed: int, epoch: int, step: int) -> int:
    """Choose a deterministic rolling-anchor index for one mini-batch."""

    if bank_size <= 0:
        raise ValueError("bank_size must be positive")
    return min(int(_uniform(seed, epoch, step, stream=1) * bank_size), bank_size - 1)
