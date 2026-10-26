"""Backbone-agnostic building blocks for AnchorADD."""

from .anchor_bank import AnchorBank, select_anchor_indices
from .conditioning import AnchorConditioner, initialise_identity_adapter
from .metrics import compute_binary_metrics, predictive_entropy
from .model import AnchorADD
from .sampling import stateless_anchor_index, stateless_apply_decision

__all__ = [
    "AnchorADD",
    "AnchorBank",
    "AnchorConditioner",
    "compute_binary_metrics",
    "initialise_identity_adapter",
    "predictive_entropy",
    "select_anchor_indices",
    "stateless_anchor_index",
    "stateless_apply_decision",
]

__version__ = "0.1.0"
