# Stage-2 Bidirectional Anchor Rerank Report (2026-06-26)

## Scope

This iteration adds a v2 Stage-2 selector under `weak_decoder/phase_line/`:

```text
Savaux Stage-1 Top-L evidence
-> strict reliable anchors
-> bidirectional interpolated phase profile
-> independent rerank of non-anchor symbols
```

The original forward Viterbi selector is left intact.

## Implemented Selector

New public entry points:

```python
select_anchor_bounded_bidirectional_rerank_path(...)
select_savaux_bidirectional_rerank_path(...)
```

The selector locks strict anchors using:

```text
margin >= 2.50 dB
peak_to_median >= 12.0 dB
branch_agreement >= 0.90
```

Locked anchor phases are unwrapped and linearly interpolated/extrapolated over
the payload.  Non-anchor candidates are scored independently:

```text
Score_i(k) = wE * E_i(k) + wC * C_i(k) + wP * Phi_i(k)

Phi_i(k) = exp(-(wrap(angle(Y_i[k]) - phase_profile_i) / sigma)^2)
```

Default v2 weights are intentionally conservative:

```text
wE = 0.85
wC = 0.05
wP = 0.10
sigma = 0.50 pi
min_switch_gain = 0.04
```

## Target 165-Packet Result

Command:

```powershell
python weak_decoder/phase_line/_eval/diagnose_stage2_v2_bidirectional.py `
  --dataset 0_0_0_10_14_16 `
  --snrs -26 -25.5 -25 -24.5 -24 `
  --seeds 42 43 44 `
  --top-l 16 --stage1-top-k 24 `
  --output-dir weak_decoder/phase_line/_eval/stage2_v2_bidirectional_default_gain004_165
```

Overall packet-average SER:

| Selector | SER | Rescue | Break | Net |
|---|---:|---:|---:|---:|
| Savaux hard | 0.2665 | - | - | - |
| v1 unguarded Viterbi | 0.2142 | 395 | 93 | +302 |
| v1 guarded | 0.2653 | 7 | 0 | +7 |
| v2 bidirectional | 0.2601 | 47 | 10 | +37 |

Packet wins/ties/losses vs Savaux hard:

```text
v2 = 35 / 124 / 6
```

## Three-Dataset Stability Probe

Command:

```powershell
python weak_decoder/phase_line/_eval/diagnose_stage2_v2_bidirectional.py `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -22 -23 -24 -25 -26 `
  --seeds 42 43 44 `
  --top-l 16 --stage1-top-k 24 `
  --output-dir weak_decoder/phase_line/_eval/stage2_v2_bidirectional_gain004_3datasets_5snr_3seed
```

Overall packet-average SER:

| Selector | SER | Rescue | Break | Net |
|---|---:|---:|---:|---:|
| Savaux hard | 0.1961 | - | - | - |
| v1 unguarded Viterbi | 0.1810 | 491 | 270 | +221 |
| v1 guarded | 0.1928 | 48 | 0 | +48 |
| v2 bidirectional | 0.1927 | 70 | 21 | +49 |

Packet wins/ties/losses vs Savaux hard:

```text
v2 = 54 / 353 / 13
```

## Island-Constrained Viterbi Extension

The follow-up variant keeps the bidirectional v2 selector intact and adds a
separate entry point:

```python
select_anchor_bounded_island_viterbi_path(...)
select_savaux_island_viterbi_path(...)
```

It uses the same strict anchor idea, but instead of reranking every non-anchor
symbol independently, it slices the payload into low-confidence islands between
locked anchors and runs Viterbi only inside each island.  Anchor symbols are
never placed in the candidate set, so island Viterbi cannot directly rewrite
them.

Default island settings after the 165-packet tuning pass:

```text
anchor: margin >= 2.50 dB, peak_to_median >= 12.0 dB, branch_agreement >= 0.90
wE = 0.75
wC = 0.10
wProfile = 0.15
top1_bonus = 0.08
transition_weight = 1.0
boundary_weight = 1.0
```

Target 165-packet matrix, output directory
`_eval/stage2_island_top1bonus008_165`:

| Selector | SER | Rescue | Break | Net |
|---|---:|---:|---:|---:|
| Savaux hard | 0.2665 | - | - | - |
| v1 unguarded Viterbi | 0.2142 | 395 | 93 | +302 |
| v1 guarded | 0.2653 | 7 | 0 | +7 |
| v2 bidirectional | 0.2601 | 47 | 10 | +37 |
| island Viterbi | 0.2528 | 96 | 17 | +79 |

Three-dataset stability probe, output directory
`_eval/stage2_island_default_3datasets_5snr_3seed`:

| Selector | SER | Rescue | Break | Net |
|---|---:|---:|---:|---:|
| Savaux hard | 0.1961 | - | - | - |
| v1 unguarded Viterbi | 0.1810 | 491 | 270 | +221 |
| v1 guarded | 0.1928 | 48 | 0 | +48 |
| v2 bidirectional | 0.1927 | 70 | 21 | +49 |
| island Viterbi | 0.1891 | 147 | 45 | +102 |

Island Viterbi is therefore a better controlled rescue layer than independent
v2 rerank: it roughly doubles net rescue in the stability probe while keeping
break far below unguarded Viterbi.  It still does not beat unguarded Viterbi on
pure SER, which means the remaining gap is likely in the local candidate
feature rather than the sequence topology alone.

## Anchor Reliability Diagnostic

On `0_0_0_10_14_16`, SNR `-26..-24 dB`, seeds `42/43/44`, the current
forward Stage-2 configuration marks many symbols as high confidence:

| Anchor class | Count | Wrong top1 | Wrong rate |
|---|---:|---:|---:|
| high confidence | 3279 | 191 | 5.82% |
| strict hard anchor | 1493 | 3 | 0.20% |

The high-confidence wrong rate is strongly SNR dependent:

| SNR | Wrong high-confidence top1 |
|---:|---:|
| -26.0 dB | 59 / 465 = 12.7% |
| -25.5 dB | 51 / 548 = 9.3% |
| -25.0 dB | 39 / 661 = 5.9% |
| -24.5 dB | 26 / 754 = 3.5% |
| -24.0 dB | 16 / 851 = 1.9% |

This validates the v2 design choice to keep
`bidirectional_lock_high_confidence=False`: high-confidence symbols can help as
soft evidence, but they are not reliable enough to serve as immutable global
anchors at low SNR.  Even strict anchors are not mathematically perfect, so v2
also prunes anchor phase outliers before building the bidirectional profile.

## Takeaway

The bidirectional anchor idea is valid as a low-break rescue layer, but not as a
strong replacement for unguarded Viterbi.  A naive strong phase profile caused
heavy breakage; the useful operating point is conservative and energy-dominant.
The remaining bottleneck is that raw FFT-bin phase is still not clean enough to
serve as a hard global constraint.
