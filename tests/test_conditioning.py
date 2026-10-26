import torch
from torch import nn

from anchoradd import AnchorConditioner


def test_identity_initialisation_reconstructs_query() -> None:
    conditioner = AnchorConditioner(nn.Conv1d(6, 3, 1), alpha=0.7)
    query = torch.randn(4, 3, 11)
    anchor = torch.randn(4, 3, 11)
    assert torch.allclose(conditioner(query), query, atol=1.0e-6)
    assert torch.allclose(conditioner(query, anchor), query, atol=1.0e-6)
