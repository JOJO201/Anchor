# Third-party notices

## RawNet2 / ASVspoof baseline

The code in `examples/rawnet2/model.py` is a modified, reorganised adaptation of
the RawNet2 baseline distributed by EURECOM for ASVspoof.  The original work is
available from the ASVspoof 2021 challenge repository:

<https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2>

The original copyright notice and MIT license are preserved in
`examples/rawnet2/LICENSE.RawNet2`.  Modifications include separating the front
end and classifier, modernising device handling, returning logits, and inserting
the generic AnchorADD conditioning module.
