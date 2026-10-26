"""Backbone-agnostic AnchorADD wrapper and multi-anchor inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .conditioning import AnchorConditioner


class AnchorADD(nn.Module):
    """Apply AnchorADD between a feature extractor and a classifier.

    ``feature_extractor`` must return channels-first features ``[B,C,...]``;
    ``classifier`` must map conditioned features to class logits ``[B,L]``.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        conditioner: AnchorConditioner,
        *,
        positive_class: int = 1,
    ) -> None:
        super().__init__()
        if positive_class < 0:
            raise ValueError("positive_class must be non-negative")
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.conditioner = conditioner
        self.positive_class = int(positive_class)

    @staticmethod
    def _validate_inputs(inputs: Tensor, *, name: str) -> Tensor:
        if not isinstance(inputs, Tensor) or inputs.ndim < 2:
            raise ValueError(f"{name} must be a batched tensor")
        if inputs.shape[0] < 1:
            raise ValueError(f"{name} cannot have an empty batch")
        return inputs

    @staticmethod
    def _validate_features(features: Tensor, batch_size: int) -> Tensor:
        if not isinstance(features, Tensor) or features.ndim < 3:
            raise ValueError(
                "feature_extractor must return a channels-first tensor [B,C,...]"
            )
        if features.shape[0] != batch_size:
            raise ValueError("feature_extractor changed the batch dimension")
        return features

    @staticmethod
    def _validate_logits(logits: Tensor, batch_size: int) -> Tensor:
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise ValueError("classifier must return logits shaped [B,num_classes]")
        if logits.shape[0] != batch_size:
            raise ValueError("classifier changed the batch dimension")
        return logits

    def forward(self, inputs: Tensor, anchor: Tensor | None = None) -> Tensor:
        """Evaluate one null-anchor or anchored view.

        A single anchor ``[1,...]`` is shared across the mini-batch, matching
        the rolling-anchor training protocol.  Per-example anchors ``[B,...]``
        are also accepted for integration experiments.
        """

        inputs = self._validate_inputs(inputs, name="inputs")
        batch_size = inputs.shape[0]

        if anchor is None:
            query_features = self._validate_features(
                self.feature_extractor(inputs), batch_size
            )
            conditioned = self.conditioner(query_features)
        else:
            if anchor.ndim == inputs.ndim - 1:
                anchor = anchor.unsqueeze(0)
            anchor = self._validate_inputs(anchor, name="anchor")
            if anchor.shape[1:] != inputs.shape[1:]:
                raise ValueError("anchor and inputs must have the same sample shape")
            anchor = anchor.to(device=inputs.device, dtype=inputs.dtype)
            if anchor.shape[0] == 1:
                anchor = anchor.expand(batch_size, *anchor.shape[1:])
            elif anchor.shape[0] != batch_size:
                raise ValueError("anchor batch must contain either 1 or B samples")

            joint = torch.cat((inputs, anchor), dim=0)
            features = self._validate_features(
                self.feature_extractor(joint), 2 * batch_size
            )
            query_features, anchor_features = features.split(batch_size, dim=0)
            conditioned = self.conditioner(query_features, anchor_features)

        return self._validate_logits(self.classifier(conditioned), batch_size)

    @torch.no_grad()
    def predict(self, inputs: Tensor, anchors: Tensor | Sequence[Tensor]) -> dict[str, Tensor]:
        """Aggregate predictions from fixed anchors and estimate uncertainty.

        The returned uncertainty is the unbiased sample variance (``ddof=1``)
        of the positive-class probability across anchor-conditioned views.
        """

        inputs = self._validate_inputs(inputs, name="inputs")
        if isinstance(anchors, Sequence) and not isinstance(anchors, Tensor):
            if not anchors:
                raise ValueError("anchors cannot be empty")
            anchors = torch.stack(
                [a.unsqueeze(0) if a.ndim == inputs.ndim - 1 else a for a in anchors],
                dim=0,
            ).squeeze(1)
        if not isinstance(anchors, Tensor):
            raise TypeError("anchors must be a tensor or a sequence of tensors")
        if anchors.ndim == inputs.ndim - 1:
            anchors = anchors.unsqueeze(0)
        anchors = self._validate_inputs(anchors, name="anchors")
        if anchors.shape[0] < 2:
            raise ValueError("at least two anchors are required to estimate variance")
        if anchors.shape[1:] != inputs.shape[1:]:
            raise ValueError("anchors and inputs must have the same sample shape")
        anchors = anchors.to(device=inputs.device, dtype=inputs.dtype)

        was_training = self.training
        self.eval()
        try:
            batch_size = inputs.shape[0]
            anchor_count = anchors.shape[0]
            joint = torch.cat((inputs, anchors), dim=0)
            features = self._validate_features(
                self.feature_extractor(joint), batch_size + anchor_count
            )
            query_features = features[:batch_size]
            anchor_features = features[batch_size:]

            probabilities = []
            for anchor_feature in anchor_features:
                expanded = anchor_feature.unsqueeze(0).expand_as(query_features)
                logits = self._validate_logits(
                    self.classifier(self.conditioner(query_features, expanded)),
                    batch_size,
                )
                probabilities.append(torch.softmax(logits, dim=-1))
            per_anchor = torch.stack(probabilities, dim=0)
            if self.positive_class >= per_anchor.shape[-1]:
                raise ValueError("positive_class is outside the classifier output")
            mean_probability = per_anchor.mean(dim=0)
            uncertainty = per_anchor[..., self.positive_class].var(
                dim=0, unbiased=True
            )
            return {
                "mean_probability": mean_probability,
                "positive_probability": mean_probability[:, self.positive_class],
                "uncertainty": uncertainty,
                "per_anchor_probability": per_anchor,
            }
        finally:
            self.train(was_training)
