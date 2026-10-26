# Release checklist

- [ ] Decide and add a repository-wide license for the authors' AnchorADD code.
- [ ] Add final author names, paper venue/year, and BibTeX or `CITATION.cff`.
- [ ] Confirm whether public release is permitted during double-blind review.
- [ ] Run `pytest -q` in a clean environment.
- [ ] Run `python -m examples.rawnet2.smoke_test --device cpu`.
- [ ] Scan for absolute paths, credentials, datasets, checkpoints, and logs.
- [ ] Verify the paper and README use the same `alpha`, `delta`, `K`, label
      mapping, uncertainty definition, and data-augmentation protocol.
- [ ] Tag the exact code revision used for final reported results.
