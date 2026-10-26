# RawNet2 example

This example demonstrates the single supported integration shipped with the
public AnchorADD package.  It separates RawNet2 into:

1. a SincConv front end producing channels-first features;
2. the generic AnchorADD conditioner and `1x1` projection; and
3. the original residual-attention/GRU classification pattern.

Anchors are deterministically selected from the training manifest.  One rolling
anchor is shared by each anchored training mini-batch.  The selected train-only
bank is saved in the checkpoint so evaluation cannot accidentally draw anchors
from a development or test set.

From the repository root:

```bash
python -m examples.rawnet2.smoke_test --device cpu
python -m examples.rawnet2.train --help
python -m examples.rawnet2.evaluate --help
```

See the root README for complete commands and the manifest schema.
