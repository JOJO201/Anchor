"""Anchor-conditioned feature transformation used by AnchorADD."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def initialise_identity_adapter(adapter: nn.Module, alpha: float = 1.0) -> None:
    """Initialise a 1x1 convolution so both AnchorADD views reconstruct ``q``.

    The AnchorADD representation is ``[c, q - alpha*c]``.  Initialising the
    two channel blocks as ``[alpha*I, I]`` maps this representation to ``q``.
    The same adapter maps the null-anchor representation ``[0, q]`` to ``q``.
    """

    if not isinstance(adapter, _CONV_TYPES):
        raise TypeError("adapter must be Conv1d, Conv2d, or Conv3d")
    if adapter.groups != 1:
        raise ValueError("adapter.groups must be 1")
    if any(size != 1 for size in adapter.kernel_size):
        raise ValueError("adapter must use a 1x1 (or 1x1x1) kernel")
    channels = adapter.out_channels
    if adapter.in_channels != 2 * channels:
        raise ValueError("adapter must map 2C input channels to C outputs")
    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")

    with torch.no_grad():
        adapter.weight.zero_()
        if adapter.bias is not None:
            adapter.bias.zero_()
        index = torch.arange(channels, device=adapter.weight.device)
        spatial = (0,) * (adapter.weight.ndim - 2)
        adapter.weight[(index, index, *spatial)] = float(alpha)
        adapter.weight[(index, channels + index, *spatial)] = 1.0


class AnchorConditioner(nn.Module):
    """Construct and project an AnchorADD feature view.

    Features must be channels-first tensors shaped ``[B, C, ...]``.  A null
    anchor uses ``[0, q]`` and therefore traverses the same trainable adapter
    as an anchored view.
    """

    def __init__(
        self,
        adapter: nn.Module,
        *,
        alpha: float = 1.0,
        initialise_as_identity: bool = True,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(alpha)):
            raise ValueError("alpha must be finite")
        self.adapter = adapter
        self.alpha = float(alpha)
        if initialise_as_identity:
            initialise_identity_adapter(self.adapter, self.alpha)

    def forward(self, query: Tensor, anchor: Tensor | None = None) -> Tensor:
        if not isinstance(query, Tensor) or query.ndim < 3:
            raise ValueError("query must be a channels-first tensor [B,C,...]")
        if anchor is None:
            anchor = torch.zeros_like(query)
        elif anchor.shape != query.shape:
            raise ValueError(
                "query and anchor features must have identical shapes; "
                f"got {tuple(query.shape)} and {tuple(anchor.shape)}"
            )
        pair = torch.cat((anchor, query - self.alpha * anchor), dim=1)
        return self.adapter(pair)
