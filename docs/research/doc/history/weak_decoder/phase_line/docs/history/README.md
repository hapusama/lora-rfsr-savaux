# Historical Phase-Line Reports

These reports record earlier attempts before the current guarded Savaux Stage-1
selector.  They are kept for evidence and design traceability, but they are not
the current recommended path.

## Timeline

```text
2026-06-19
  Phase-only and sliding-window phase-line attempts were explored.
  Result: not stable enough; candidate recall was the main blocker.

2026-06-19
  Viterbi/DP candidate coverage was diagnosed.
  Result: low-SNR Top-L recall limited what any Stage-2 phase selector could fix.

2026-06-20 daytime
  Phase-aware proposal rescue, adaptive Top-L, hard anchors, and path arbiter
  variants were tested.
  Result: useful diagnostics, but still less stable than a stronger Stage 1.

2026-06-20 evening
  Savaux synchronized oversampled Stage 1 plus guarded phase takeover became
  the active implementation.  See SAVAUX_STAGE1_SELECTOR_REPORT_2026-06-20.md.

2026-06-26 to 2026-06-27
  Stage-2 bidirectional rerank, adaptive island, and anchor-bounded island
  variants were split into separate variant folders.
  Result: useful fallback/arbiter evidence, but still weaker than the v1
  one-order DP baseline on the measured weak-packet fixtures.
```

## Archived Files

```text
PHASE_LINE_STOP_REPORT_2026-06-19.md
  Stop report for the early sliding-window/local phase-line direction.

PHASE_PATH_ABLATION_REPORT_2026-06-19.md
  First-order/second-order phase Viterbi ablations on multi-offset Top-L
  candidates.

DP_CANDIDATE_COVERAGE_REPORT_2026-06-19.md
  Candidate recall diagnosis showing Stage-1 Top-L coverage as the bottleneck.

VITERBI_DP_NOTES_2026-06-19.md
  Implementation notes for the Viterbi/DP selector in selector.py.

PHASE_PROPOSAL_RESCUE_REPORT_2026-06-20.md
  Proposal rescue and adaptive Top-L experiments before switching to Savaux
  Stage 1.

SAVAUX_STAGE1_SELECTOR_REPORT_2026-06-20.md
  Final Savaux Stage-1 selector report for the guarded selector generation.

STAGE2_BIDIRECTIONAL_RERANK_REPORT_2026-06-26.md
  Bidirectional rerank fallback and arbiter experiment report.

STAGE2_ADAPTIVE_ISLAND_REPORT_2026-06-27.md
  Adaptive island experiment report before the newer island-DP reconstruction
  branch.
```

## Current Entry Points

Return to:

```text
../../README.md
../README.md
SAVAUX_STAGE1_SELECTOR_REPORT_2026-06-20.md
```
