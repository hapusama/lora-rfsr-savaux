# Stage-2 Adaptive Island Notes

## Goal

Keep the previous stable island Viterbi as the safe baseline, and only let a
more aggressive low-confidence Viterbi profile operate when Stage-1 has almost
no strict anchors.

## Implemented Variant

The new entry point lives in `stage2_variants.py`:

```text
stable island
  -> strict anchors + island-bounded Viterbi

adaptive aggressive island
  -> run stable island
  -> if strict_anchor_ratio <= 0.03, also run aggressive island
  -> accept aggressive only when change_count <= 4 and change_ratio <= 0.12
```

The aggressive profile raises the anchor threshold, widens candidate search,
allows unanchored Viterbi, and weakens the top1 prior. The arbitration step is
important: without it, packets with 5+ changed symbols can drift into a worse
global path.

## Key Results

`0_0_0_10_14_16`, 5 SNRs, 3 seeds, 165 packets:

```text
Savaux hard SER      0.26649
v1 forward VTB SER   0.21420   rescue/break/net = 395/93/+302
v2 bidirectional SER 0.26009   rescue/break/net = 47/10/+37
stable island SER    0.25281   rescue/break/net = 96/17/+79
adaptive island SER  0.25160   rescue/break/net = 103/17/+86
```

Three datasets (`*_8`, `*_16`, `*_32`), 5 SNRs, 3 seeds, 420 packets:

```text
Savaux hard SER      0.19605
v1 forward VTB SER   0.18102   rescue/break/net = 491/270/+221
v2 bidirectional SER 0.19272   rescue/break/net = 70/21/+49
stable island SER    0.18912   rescue/break/net = 147/45/+102
adaptive island SER  0.18857   rescue/break/net = 158/48/+110
```

CRC-valid rate did not improve over stable island in this run, but adaptive did
not reduce it either.

## Takeaway

This is not a 1 dB gain yet. It is a controlled improvement over the stable
island baseline: the DP gets extra authority only in packets where Stage-1 has
near-zero reliable anchors, and the previous best stable island path remains
the fallback whenever the aggressive path changes too much.
