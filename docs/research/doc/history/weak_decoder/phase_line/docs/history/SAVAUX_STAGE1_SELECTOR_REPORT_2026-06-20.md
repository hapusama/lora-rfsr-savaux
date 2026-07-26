# Savaux Stage-1 Guarded FFT-Bin Selector Report (2026-06-20)

## Scope

This iteration only changes raw payload FFT-bin selection under:

```text
weak_decoder/phase_line/
```

It does not use CRC-guided search, LoRa codec enumeration, payload templates,
cross-packet priors, or ground-truth bins during selection.

## Implemented Selector

The handoff pointed to Stage-1 candidate recall as the bottleneck.  The new
selector therefore uses a Savaux-style synchronized oversampled observation as
Stage 1, then lets the phase path touch only very small, high-confidence
corrections.

The flow is:

```text
payload symbol timing
-> Savaux oversampled branch DFTs
-> Eq. (37)-style branch phase alignment
-> coherent combined spectrum + branch agreement
-> conservative first-order phase Viterbi
-> runtime guard; otherwise fall back to Savaux hard bin
```

Main implementation:

```text
savaux_stage1.py
```

Primary exported helpers:

```python
SavauxStage1Config
SavauxPhaseGuardConfig
build_savaux_stage1_packet_evidence(...)
default_savaux_phase_path_config(...)
evaluate_savaux_phase_guard(...)
select_savaux_phase_viterbi_path(...)
```

## Guard Rule

Unconditional phase Viterbi gives larger average gains, but it is not stable
across packet lengths and noise seeds.  The final selector therefore accepts a
phase correction only when all runtime-only checks pass:

```text
1 <= changed_symbols <= 2
phase-path eval RMSE improves over Savaux hard
mean_phase_score >= 0.990
changed-bin mean energy drop >= -2.0 dB
```

Rejected packets use Savaux hard argmax.  This makes the phase path a guarded
takeover mechanism instead of a mandatory replacement.

## Focused Target Result

On the original target dataset `0_0_0_10_14_16`:

| SNR | Legacy Phase-Line SER | Savaux Hard SER | Unconditional Phase SER | Guarded SER |
|---:|---:|---:|---:|---:|
| -24 dB | 0.205 | 0.143 | 0.109 | 0.132 |
| -25 dB | 0.444 | 0.239 | 0.221 | 0.236 |

Command:

```powershell
python weak_decoder/phase_line/_eval/diagnose_savaux_stage1_selector.py `
  --dataset 0_0_0_10_14_16 `
  --snrs -24 -25 `
  --seeds 42 `
  --output-dir weak_decoder/phase_line/_eval/savaux_stage1_guarded_smoke
```

## Stability Matrix

Final validation used three datasets, five SNRs, and five independent noise
seeds.  Seeds `42,43,44` were used while tuning; seeds `45,46` were holdout
verification.

```text
datasets = 0_0_0_10_14_8, 0_0_0_10_14_16, 0_0_0_10_14_32
SNRs     = -22, -23, -24, -25, -26 dB
seeds    = 42, 43, 44, 45, 46
groups   = 75 dataset/SNR/seed combinations
packets  = 700 paired packet comparisons
```

Combined result:

| Selector | Mean SER |
|---|---:|
| Legacy phase-line | 0.3235 |
| Savaux hard | 0.1980 |
| Unconditional Savaux + phase | 0.1878 |
| Guarded Savaux + phase | 0.1951 |

Stability versus Savaux hard:

| Selector | Negative Groups | Packet Wins | Packet Ties | Packet Losses |
|---|---:|---:|---:|---:|
| Unconditional phase | 13 | - | - | - |
| Guarded phase | 0 | 63 | 637 | 0 |

By dataset:

| Dataset | Savaux Hard SER | Guarded SER | Legacy SER | Guarded Positive Groups |
|---|---:|---:|---:|---:|
| `0_0_0_10_14_8` | 0.1880 | 0.1861 | 0.3230 | 9 |
| `0_0_0_10_14_16` | 0.1656 | 0.1605 | 0.2912 | 20 |
| `0_0_0_10_14_32` | 0.2405 | 0.2389 | 0.3562 | 7 |

By SNR:

| SNR | Savaux Hard SER | Guarded SER | Gain |
|---:|---:|---:|---:|
| -22 dB | 0.0575 | 0.0560 | +0.0015 |
| -23 dB | 0.0879 | 0.0840 | +0.0040 |
| -24 dB | 0.1614 | 0.1571 | +0.0043 |
| -25 dB | 0.2694 | 0.2651 | +0.0043 |
| -26 dB | 0.4140 | 0.4136 | +0.0004 |

Saved outputs:

```text
_eval/savaux_stage1_guard099_stability_3datasets_5snr_3seed/
_eval/savaux_stage1_guard099_holdout_3datasets_5snr_2seed/
```

## Conclusion

The stable FFT-bin strategy is:

```text
Savaux synchronized oversampled hard bin as the baseline decision,
plus a tightly guarded phase-line takeover for small local corrections.
```

This consistently beats the legacy phase-line baseline by a large margin and
beats Savaux hard without any observed packet-level regression in the 700-packet
paired matrix above.  The unguarded phase path remains useful as an oracle-like
upper bound, but the guarded version is the safer default.
