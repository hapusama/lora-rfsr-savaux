# Handoff: Phase-Line Work and Savaux-Based Stage-1 Redesign

Date: 2026-06-20

## Current Goal

The goal is to improve low-SNR LoRa payload FFT-bin selection.

The current phase-line work tries to solve this problem:

```text
At -24 / -25 dB, the correct FFT bin is often not the strongest energy peak.
If Stage 1 only keeps energy Top-L bins, the correct bin may be excluded.
Once that happens, Stage 2 cannot recover it, even if the phase path selector is good.
```

The original motivation of the phase method is still valid:

```text
When energy Top-1 is unreliable, packet-local phase can help break the Top-1 decision.
This is most useful for very weak symbols where the correct bin is close, but not selected by energy alone.
```

## What Has Been Changed So Far

Work was done in:

```text
weak_decoder/phase_line/configs.py
weak_decoder/phase_line/selector.py
weak_decoder/phase_line/_eval/
weak_decoder/phase_line/PHASE_PROPOSAL_RESCUE_REPORT_2026-06-20.md
```

The main changes are:

1. Stage 1 no longer only keeps fused-energy Top-L.
   It can add limited phase/coherence-aided candidates.

2. Some high-confidence symbols are hard-decoded.
   These symbols are used as phase anchors for the second stage.

3. Top-L can be widened adaptively.
   The selector keeps `top_l=24` normally, but may widen to 28 or 40 when the packet has enough reliable anchors.

4. Wrong hard locks are guarded.
   Hard anchors can be softened only under narrow packet-level conditions.

5. The second stage uses a first-order phase path search.
   It combines local energy, offset coherence, and phase smoothness.

6. A packet-level fallback was added.
   If the chosen phase path has abnormal phase jumps, the selector can fall back to the older v3/local-coherence result.

## Current Effect

On dataset:

```text
0_0_0_10_14_16
```

For -24 / -25 dB:

```text
Old phase DP:
  -24 dB SER = 0.283
  -25 dB SER = 0.410

Current phase-line selector:
  -24 dB SER = 0.229
  -25 dB SER = 0.345
```

Compared with current v3:

```text
v3:
  -24 dB SER = 0.296
  -25 dB SER = 0.436

current phase-line:
  -24 dB SER = 0.229
  -25 dB SER = 0.345
```

So the current phase-line direction does help.

But it still does not beat the Savaux oversampled baseline:

```text
Savaux oversampled:
  -24 dB SER = 0.117
  -25 dB SER = 0.249

current phase-line:
  -24 dB SER = 0.229
  -25 dB SER = 0.345
```

## Main Finding

The biggest remaining problem is still Stage 1 candidate recall.

Current candidate recall:

```text
-24 dB:
  Stage-1 Top-L recall       = 0.862
  Viterbi candidate recall   = 0.852

-25 dB:
  Stage-1 Top-L recall       = 0.766
  Viterbi candidate recall   = 0.766
```

This means:

```text
At -25 dB, about 23% of correct bins are not even available to Stage 2.
The phase path selector cannot choose a bin that was never proposed.
```

## Things Already Tried That Did Not Work

These ideas were tested and should not be repeated blindly:

1. Add many extra coherence-ranked candidates.
   Result: worsened SER.

2. Let all hard anchors drive phase proposal.
   Result: unstable; some wrong hard anchors damage nearby symbols.

3. Globally soften hard anchors to Top-2 or Top-3.
   Result: adds too many distractors.

4. Use the currently selected path to do another local phase rescue.
   Result: worsened SER.

5. Stronger generic sliding-window phase refinement.
   Result: often reinforces the wrong selected path.

6. Keep increasing Top-L.
   Result: recall improves, but extra wrong candidates also increase; Stage 2 then chooses smoother but wrong paths.

## Clearer Version of the New Direction

The next direction should be:

```text
Use Savaux-style oversampled demodulation as the new Stage 1 foundation.
Then use phase only for the remaining ambiguous symbols.
```

The reasoning is:

1. Savaux's method depends heavily on good frame synchronization.
   If frame sync is not accurate, the oversampled branches are not correctly aligned and coherent combining loses gain.

2. In this project, frame sync is already available.
   Instead of treating phase-line selection as a replacement for Savaux, use it as an extra tool after a stronger Savaux-style Stage 1.

3. Stage 1 should first make the best possible per-symbol observation:

```text
frame sync
-> sample-offset branches
-> phase alignment between branches
-> coherent combining
-> high-confidence direct decisions
-> candidate set for uncertain symbols
```

4. Stage 2 should not try to repair everything.
   It should focus on symbols where coherent combining still leaves ambiguity:

```text
If a symbol is clearly decided after coherent combining:
  hard-decode it and use it as a phase anchor.

If a symbol is still ambiguous:
  keep several candidates and let the phase path selector choose.
```

In simple terms:

```text
First make the strongest possible synchronized, oversampled FFT observation.
Then use packet-local phase to handle only the few symbols that remain uncertain.
```

## Proposed Next Design

### Stage 1: Savaux-Style Candidate Generation

Replace the current energy-only / fused multi-offset proposal with a stronger first stage:

```text
Input:
  synchronized packet
  payload symbol start
  CFO / timing information
  oversampled IQ

For each payload symbol:
  1. split the oversampled symbol into sample-offset branches
  2. dechirp each branch
  3. compute branch FFTs using the Savaux formula
  4. phase-align the branches
  5. coherently combine the branches
  6. produce:
       - combined spectrum
       - branch spectra
       - branch phase agreement
       - Top-K candidate bins
       - confidence metrics
```

Important point:

```text
The phase alignment between branches must be designed carefully.
This is not just summing magnitudes.
The complex branch outputs need to be aligned before coherent addition.
```

### Stage 1 Hard Decisions

After coherent combining:

```text
If one bin is clearly stronger:
  select it directly
  mark it as a hard anchor

If several bins are close:
  keep them as candidates
```

The hard-anchor criteria should use:

```text
peak margin
peak-to-median ratio
branch phase agreement
possibly packet-level consistency
```

### Stage 2: Phase Selection for Ambiguous Symbols

For uncertain symbols:

```text
Use the current phase-line / first-order path selector.
But now its candidate set comes from Savaux-style coherent combining,
not from weak fused-energy Top-L.
```

The second stage should use:

```text
hard anchors from Stage 1
candidate energy after coherent combining
candidate branch phase agreement
packet-local phase smoothness
```

The expected role of phase becomes narrower and more realistic:

```text
Phase does not need to beat Savaux alone.
Phase should improve Savaux when a few low-SNR symbols remain ambiguous.
```

## Implementation Notes

Relevant existing code:

```text
weak_decoder/baselines/savaux_oversampled/paper_oversampled_demod.py
weak_decoder/structured_path_demod.py
weak_decoder/adaptive_path_demod.py
weak_decoder/timing_path_demod.py
scripts/experiments/structured_paths/
weak_decoder/phase_line/selector.py
```

Useful current phase-line entry point:

```text
select_phase_viterbi_path(...)
```

Likely implementation path:

1. Add a diagnostic script under `weak_decoder/phase_line/_eval/`.
   It should compare candidate recall from:

```text
current fused-energy Top-L
current phase-line proposal
Savaux combined spectrum Top-K
Savaux combined spectrum + branch phase agreement
```

2. Test only -24 / -25 dB first.

3. If Savaux Top-K gives much better recall, build a new Stage-1 evidence object inside `phase_line`.

4. Feed that evidence into the current phase path selector.

5. Only then tune the second-stage phase weights.

## Success Criteria

The next version should report all of these:

```text
Stage-1 candidate recall
Viterbi candidate recall
final SER
hard-anchor count
hard-anchor error rate
SER versus v3
SER versus Savaux oversampled
```

The target is:

```text
At -24 / -25 dB, match or beat Savaux oversampled,
while showing that phase helps on the remaining ambiguous symbols.
```

## Short Summary

Current work proves that phase helps:

```text
-24 dB: 0.283 -> 0.229
-25 dB: 0.410 -> 0.345
```

But it also proves that the current first stage is too weak:

```text
too many correct bins are missing before Stage 2 starts.
```

The next step should not be more blind phase smoothing.

The better next step is:

```text
use Savaux-style synchronized oversampled coherent combining for Stage 1,
hard-decode clear symbols,
then use the phase-line selector only for symbols that remain uncertain.
```
