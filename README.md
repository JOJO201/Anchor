# AnchorADD

AnchorADD is a backbone-agnostic multi-view method for robust audio deepfake
detection and uncertainty estimation.  This repository contains the method
itself and one runnable RawNet2 integration example.  Other detector baselines,
datasets, checkpoints, and experiment outputs are intentionally not bundled.

## Method

Let `phi(x)` be an intermediate representation of an input waveform and let
`c = phi(a)` be the representation of a training-set anchor.  AnchorADD forms

```text
z(x, c) = [c, phi(x) - alpha * c]
```

and maps the concatenated representation back to the detector feature width
with a shared trainable projection.  During training, one anchor from a
train-only bank is shared by a mini-batch with probability `delta`.  At test
time, the detector evaluates `K` fixed anchors.  The mean class probability is
the final prediction; the unbiased sample variance of the bona-fide probability
is the AnchorADD uncertainty score.

The projection is initialised as `[alpha*I, I]`, so both the anchored view
`[c, phi(x)-alpha*c]` and the null-anchor view `[0, phi(x)]` initially reconstruct
`phi(x)`.

## Repository layout

```text
src/anchoradd/          reusable AnchorADD implementation
examples/rawnet2/       the only included detector example
configs/                example configuration
scripts/                manifest preparation
tests/                  core and end-to-end smoke tests
docs/                   method and integration notes
```

## Installation

Python 3.10 or newer is recommended.  Install a PyTorch build appropriate for
your CPU/CUDA environment, then install this project:

```bash
pip install -e ".[example,dev]"
```

The cleaned package targets Python 3.10+ and PyTorch 2.1+.  Record the exact
Python, PyTorch, CUDA, seed, and configuration used for each paper result.  The
core package does not require CUDA.

Run the test suite and the RawNet2 smoke test:

```bash
pytest -q
python -m examples.rawnet2.smoke_test --device cpu
```

## Data manifests

Audio is not included.  The example accepts JSONL manifests with one record per
line:

```json
{"sample_id":"LA_T_1138215","audio_path":"ASVspoof2019_LA_train/flac/LA_T_1138215.flac","label":"spoof","split":"train"}
```

Build ASVspoof 2019 LA manifests relative to a common `LA` directory:

```bash
python scripts/build_manifest.py \
  --protocol /path/to/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  --audio-prefix ASVspoof2019_LA_train/flac \
  --audio-root /path/to/LA \
  --split train \
  --output manifests/asvspoof2019_la_train.jsonl

python scripts/build_manifest.py \
  --protocol /path/to/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  --audio-prefix ASVspoof2019_LA_dev/flac \
  --audio-root /path/to/LA \
  --split dev \
  --output manifests/asvspoof2019_la_dev.jsonl
```

## RawNet2 example

Train:

```bash
python -m examples.rawnet2.train \
  --config configs/rawnet2_asvspoof2019.yaml \
  --train-manifest manifests/asvspoof2019_la_train.jsonl \
  --dev-manifest manifests/asvspoof2019_la_dev.jsonl \
  --audio-root /path/to/LA \
  --output-dir runs/rawnet2_anchoradd
```

Evaluate:

```bash
python -m examples.rawnet2.evaluate \
  --checkpoint runs/rawnet2_anchoradd/best.pt \
  --manifest manifests/asvspoof2019_la_eval.jsonl \
  --audio-root /path/to/LA \
  --output-dir runs/rawnet2_anchoradd/eval
```

Evaluation writes `predictions.csv` and `metrics.json`.  Internal labels are
`spoof=0` and `bonafide=1`, and larger scores mean more likely bona fide.

## Metrics

The evaluation script reports EER, NLL, Brier score, 10-bin ECE, predictive-
entropy error AUROC, and anchor-variance error AUROC.  The two uncertainty
AUROCs are deliberately named separately: the AnchorADD uncertainty defined by
the method is cross-anchor probability variance.

min-tDCF requires a paired ASV protocol and score file and is therefore not
computed by this detector-only example.

## Adding another detector

Split the detector into a feature extractor and classifier, insert a `2C -> C`
adapter at a channels-first feature boundary, and construct `AnchorADD` as shown
in [`docs/adding_a_backbone.md`](docs/adding_a_backbone.md).

## Scope and reproducibility

The example implements the core AnchorADD protocol without vendoring RawBoost.
The paper's augmentation settings and the distinction between method uncertainty
and predictive entropy are documented in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Attribution and licensing

The RawNet2 example is adapted from the EURECOM ASVspoof baseline and retains
its MIT notice in `examples/rawnet2/LICENSE.RawNet2`.  See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

A repository-wide license and final citation metadata must be selected by the
authors before public release; see [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).
