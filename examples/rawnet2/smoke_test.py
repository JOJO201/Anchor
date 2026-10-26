"""CPU/GPU smoke test for the complete RawNet2 + AnchorADD path."""

from __future__ import annotations

import argparse

import torch

from .model import build_anchoradd_rawnet2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    config = {
        "model": {
            "sample_rate": 16_000,
            "first_conv": 31,
            "sinc_filters": 4,
            "residual_channels": 8,
            "gru_hidden": 8,
            "gru_layers": 1,
            "fc_hidden": 8,
            "num_classes": 2,
        },
        "anchoradd": {"alpha": 1.0},
    }
    model = build_anchoradd_rawnet2(config).to(device)
    waveforms = torch.randn(2, 3_200, device=device)
    anchors = torch.randn(3, 3_200, device=device)
    null_logits = model(waveforms)
    anchored_logits = model(waveforms, anchors[:1])
    output = model.predict(waveforms, anchors)
    assert null_logits.shape == (2, 2)
    assert anchored_logits.shape == (2, 2)
    assert output["mean_probability"].shape == (2, 2)
    assert output["per_anchor_probability"].shape == (3, 2, 2)
    assert output["uncertainty"].shape == (2,)
    assert torch.isfinite(output["uncertainty"]).all()
    print("RawNet2 + AnchorADD smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
