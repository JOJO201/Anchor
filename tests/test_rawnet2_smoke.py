import torch

from examples.rawnet2.model import build_anchoradd_rawnet2


def test_rawnet2_anchoradd_forward_and_predict() -> None:
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
    model = build_anchoradd_rawnet2(config)
    waveforms = torch.randn(2, 3_200)
    anchors = torch.randn(3, 3_200)
    assert model(waveforms, anchors[:1]).shape == (2, 2)
    output = model.predict(waveforms, anchors)
    assert output["per_anchor_probability"].shape == (3, 2, 2)
