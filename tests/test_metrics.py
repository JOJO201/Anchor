import numpy as np

from anchoradd import compute_binary_metrics


def test_perfect_binary_predictions() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.01, 0.1, 0.9, 0.99])
    uncertainty = np.asarray([0.1, 0.2, 0.2, 0.1])
    result = compute_binary_metrics(labels, scores, uncertainty=uncertainty)
    assert result["eer"] == 0.0
    assert result["accuracy_at_0_5"] == 1.0
    assert result["anchor_variance_error_auroc"] is None
