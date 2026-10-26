# Adding a backbone

Choose a channels-first feature boundary `phi(x)` with shape `[B,C,...]`, then
split the detector into `feature_extractor` and `classifier` modules.

```python
from torch import nn
from anchoradd import AnchorADD, AnchorConditioner

adapter = nn.Conv1d(2 * channels, channels, kernel_size=1)
conditioner = AnchorConditioner(adapter, alpha=1.0)
model = AnchorADD(feature_extractor, classifier, conditioner)
```

Training with one shared anchor:

```python
logits = model(query_waveforms, anchor_waveform)  # anchor batch is [1,...]
```

Fixed-bank inference:

```python
output = model.predict(query_waveforms, fixed_train_anchors)
probability = output["positive_probability"]
uncertainty = output["uncertainty"]
```

The feature extractor and classifier must return tensors.  The classifier must
return unnormalised logits shaped `[B,num_classes]`.
