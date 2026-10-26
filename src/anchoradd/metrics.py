"""Detection, calibration, and uncertainty-error metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


EPS = 1.0e-12


def predictive_entropy(probability: np.ndarray | Iterable[float]) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _ece(labels: np.ndarray, probability: np.ndarray, bins: int) -> float:
    predicted = (probability >= 0.5).astype(np.int64)
    confidence = np.maximum(probability, 1.0 - probability)
    correct = (predicted == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        upper_closed = index == bins - 1
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if upper_closed
            else confidence < edges[index + 1]
        )
        if np.any(mask):
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if np.unique(labels).size != 2:
        return None
    return float(roc_auc_score(labels, scores))


def compute_binary_metrics(
    labels: np.ndarray | Iterable[int],
    probability_bonafide: np.ndarray | Iterable[float],
    *,
    uncertainty: np.ndarray | Iterable[float] | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Compute the paper-facing binary metrics.

    Labels use ``spoof=0`` and ``bonafide=1``; larger scores therefore mean
    more likely bona fide.  When supplied, ``uncertainty`` should be the
    AnchorADD cross-anchor probability variance.
    """

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability_bonafide, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("labels and probabilities must be non-empty 1-D arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must use spoof=0 and bonafide=1")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite values in [0,1]")
    if np.unique(y).size != 2:
        raise ValueError("EER requires both spoof and bonafide examples")
    if ece_bins < 2:
        raise ValueError("ece_bins must be at least 2")

    fpr, tpr, thresholds = roc_curve(y, p, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[index] + fnr[index]) / 2.0)
    predicted = (p >= 0.5).astype(np.int64)
    errors = (predicted != y).astype(np.int64)
    clipped = np.clip(p, EPS, 1.0 - EPS)
    entropy = predictive_entropy(p)

    result: dict[str, Any] = {
        "n": int(y.size),
        "n_spoof": int((y == 0).sum()),
        "n_bonafide": int((y == 1).sum()),
        "eer": eer,
        "eer_percent": 100.0 * eer,
        "eer_threshold": float(thresholds[index]),
        "accuracy_at_0_5": float((predicted == y).mean()),
        "nll": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
        "brier": float(np.square(p - y).mean()),
        "ece": _ece(y, p, ece_bins),
        "predictive_entropy_error_auroc": _safe_auroc(errors, entropy),
    }

    if uncertainty is not None:
        u = np.asarray(uncertainty, dtype=np.float64)
        if u.shape != y.shape or not np.isfinite(u).all():
            raise ValueError("uncertainty must be finite and match labels")
        result["anchor_variance_error_auroc"] = _safe_auroc(errors, u)
        result["mean_anchor_variance"] = float(u.mean())
    return result
