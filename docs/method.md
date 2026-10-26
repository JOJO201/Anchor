# Method contract

For query features `q = phi(x)` and anchor features `c = phi(a)`, AnchorADD uses
the paired representation `[c, q - alpha*c]`.  A shared trainable adapter maps
the pair from `2C` channels back to `C` channels before the unchanged detector
back end.

The implementation enforces the following contract:

- anchors originate from the training split only;
- one rolling anchor is shared by a training mini-batch;
- the anchored transform is applied with probability `delta`;
- evaluation uses at least two fixed anchors from the saved train-only bank;
- class probabilities, rather than logits, are averaged across anchors;
- uncertainty is the unbiased sample variance of the bona-fide probability.

`AnchorConditioner` also supports a null anchor.  It constructs `[0,q]` and
passes it through the same adapter, which avoids changing the module graph when
performing controlled with/without-anchor comparisons.
