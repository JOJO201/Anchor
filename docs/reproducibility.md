# Reproducibility notes

## Frozen method semantics

- label mapping: spoof `0`, bona fide `1`;
- prediction: mean class probability over fixed anchors;
- method uncertainty: unbiased variance of bona-fide probability;
- anchor source: training split only;
- default example values: `alpha=1`, `delta=1`, `K=10`;
- RawNet2 waveform length: 64,600 samples at 16 kHz.

## Data augmentation

The paper uses RawBoost algorithm 5 for training.  Its two test-time profiles
use algorithm 3 (DA1) and algorithm 7 (DA2).  RawBoost is an external method and
is not copied into this minimal repository.  To reproduce those profiles, apply
the corresponding waveform transform in `AudioManifestDataset` before the
fixed-length crop and record the algorithm in the experiment configuration.

DA1 and DA2 are test-time profiles of one checkpoint when the training protocol
is otherwise identical; do not train separate checkpoints merely to relabel the
evaluation profile.

## Metric naming

`anchor_variance_error_auroc` evaluates the uncertainty proposed by AnchorADD.
`predictive_entropy_error_auroc` evaluates entropy of the mean prediction.  They
are not interchangeable and should never both be reported under the ambiguous
label `AUC`.

min-tDCF is only meaningful when a compatible ASV protocol and ASV scores are
available.  ASVspoof 2021 subset t-DCF values must be labelled as subset metrics,
not official full-partition challenge scores.
