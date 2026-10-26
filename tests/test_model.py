import torch
from torch import nn

from anchoradd import AnchorADD, AnchorConditioner


def _model() -> AnchorADD:
    encoder = nn.Conv1d(1, 2, 1)
    classifier = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(2, 2))
    conditioner = AnchorConditioner(nn.Conv1d(4, 2, 1), alpha=1.0)
    return AnchorADD(encoder, classifier, conditioner)


def test_multi_anchor_prediction_shapes_and_restores_state() -> None:
    model = _model()
    model.train()
    inputs = torch.randn(3, 1, 20)
    anchors = torch.randn(5, 1, 20)
    output = model.predict(inputs, anchors)
    assert model.training
    assert output["mean_probability"].shape == (3, 2)
    assert output["per_anchor_probability"].shape == (5, 3, 2)
    assert output["uncertainty"].shape == (3,)
    assert torch.allclose(output["mean_probability"].sum(dim=1), torch.ones(3))
