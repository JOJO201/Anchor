"""A compact RawNet2 integration example for AnchorADD.

The RawNet2 architecture is adapted from the ASVspoof 2021 baseline released
by EURECOM.  See ``LICENSE.RawNet2`` and ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from anchoradd import AnchorADD, AnchorConditioner


class SincConv1d(nn.Module):
    """Fixed mel-spaced Sinc filters used by the RawNet2 front end."""

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        *,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__()
        if out_channels <= 0 or kernel_size <= 1 or sample_rate <= 0:
            raise ValueError("invalid SincConv configuration")
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.sample_rate = int(sample_rate)
        self.register_buffer("filters", self._make_filters(), persistent=True)

    @staticmethod
    def _to_mel(hz: Tensor) -> Tensor:
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    @staticmethod
    def _to_hz(mel: Tensor) -> Tensor:
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)

    def _make_filters(self) -> Tensor:
        nyquist = self.sample_rate / 2.0
        mel_limits = self._to_mel(torch.tensor([0.0, nyquist]))
        mel_edges = torch.linspace(
            mel_limits[0], mel_limits[1], self.out_channels + 1
        )
        hz_edges = self._to_hz(mel_edges)
        half = (self.kernel_size - 1) / 2.0
        support = torch.arange(-half, half + 1.0)
        window = torch.hamming_window(self.kernel_size, periodic=False)
        filters = []
        for lower, upper in zip(hz_edges[:-1], hz_edges[1:]):
            high = (2.0 * upper / self.sample_rate) * torch.sinc(
                2.0 * upper * support / self.sample_rate
            )
            low = (2.0 * lower / self.sample_rate) * torch.sinc(
                2.0 * lower * support / self.sample_rate
            )
            filters.append(window * (high - low))
        return torch.stack(filters, dim=0).unsqueeze(1)

    def forward(self, waveforms: Tensor) -> Tensor:
        return F.conv1d(waveforms, self.filters)


class RawNet2FrontEnd(nn.Module):
    def __init__(self, sinc_filters: int, first_conv: int, sample_rate: int) -> None:
        super().__init__()
        self.sinc = SincConv1d(
            sinc_filters, first_conv, sample_rate=sample_rate
        )
        self.batch_norm = nn.BatchNorm1d(sinc_filters)
        self.activation = nn.SELU(inplace=True)

    def forward(self, waveforms: Tensor) -> Tensor:
        if waveforms.ndim != 2:
            raise ValueError("RawNet2 expects waveforms shaped [B,T]")
        features = self.sinc(waveforms.unsqueeze(1))
        features = F.max_pool1d(torch.abs(features), 3)
        return self.activation(self.batch_norm(features))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, first: bool) -> None:
        super().__init__()
        self.first = bool(first)
        self.batch_norm1 = None if first else nn.BatchNorm1d(in_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.pool = nn.MaxPool1d(3)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.shortcut(inputs)
        hidden = inputs
        if self.batch_norm1 is not None:
            hidden = self.activation(self.batch_norm1(hidden))
        hidden = self.conv1(hidden)
        hidden = self.conv2(self.activation(self.batch_norm2(hidden)))
        return self.pool(hidden + residual)


class AttentionResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, first: bool) -> None:
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels, first=first)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.attention = nn.Linear(out_channels, out_channels)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.residual(inputs)
        weights = torch.sigmoid(
            self.attention(self.pool(hidden).squeeze(-1))
        ).unsqueeze(-1)
        return hidden * weights + weights


class RawNet2BackEnd(nn.Module):
    """RawNet2 residual/attention/GRU classifier operating on front features."""

    def __init__(
        self,
        front_channels: int,
        residual_channels: int,
        gru_hidden: int,
        gru_layers: int,
        fc_hidden: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        pairs = [
            (front_channels, front_channels),
            (front_channels, front_channels),
            (front_channels, residual_channels),
            (residual_channels, residual_channels),
            (residual_channels, residual_channels),
            (residual_channels, residual_channels),
        ]
        self.blocks = nn.ModuleList(
            AttentionResidualBlock(source, target, first=index == 0)
            for index, (source, target) in enumerate(pairs)
        )
        self.batch_norm = nn.BatchNorm1d(residual_channels)
        self.activation = nn.SELU(inplace=True)
        self.gru = nn.GRU(
            input_size=residual_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.fc1 = nn.Linear(gru_hidden, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        hidden = features
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.activation(self.batch_norm(hidden)).transpose(1, 2)
        self.gru.flatten_parameters()
        hidden, _ = self.gru(hidden)
        return self.fc2(self.fc1(hidden[:, -1]))


def build_anchoradd_rawnet2(config: Mapping[str, Any]) -> AnchorADD:
    """Build RawNet2 with AnchorADD inserted after its Sinc front end."""

    model = config.get("model", config)
    method = config.get("anchoradd", {})
    front_channels = int(model.get("sinc_filters", 20))
    front = RawNet2FrontEnd(
        sinc_filters=front_channels,
        first_conv=int(model.get("first_conv", 1024)),
        sample_rate=int(model.get("sample_rate", 16_000)),
    )
    back = RawNet2BackEnd(
        front_channels=front_channels,
        residual_channels=int(model.get("residual_channels", 128)),
        gru_hidden=int(model.get("gru_hidden", 1024)),
        gru_layers=int(model.get("gru_layers", 3)),
        fc_hidden=int(model.get("fc_hidden", 1024)),
        num_classes=int(model.get("num_classes", 2)),
    )
    adapter = nn.Conv1d(2 * front_channels, front_channels, 1, bias=True)
    conditioner = AnchorConditioner(
        adapter,
        alpha=float(method.get("alpha", 1.0)),
        initialise_as_identity=True,
    )
    return AnchorADD(front, back, conditioner, positive_class=1)
