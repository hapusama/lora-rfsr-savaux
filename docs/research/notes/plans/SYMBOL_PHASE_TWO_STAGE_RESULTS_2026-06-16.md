# Symbol-Level Phase Two-Stage Results

Date: 2026-06-16

Scope:

```text
gr-lora_sdr/weakPacket_decoding copy
```

## Implemented Path

New inference-only symbol peak selector:

```text
weak_decoder/symbol_phase_two_stage.py
scripts/experiments/run_symbol_phase_two_stage.py
```

The implemented decoder stays at raw FFT-bin level:

```text
multi-offset FFT evidence
  -> lock high-confidence symbols as Top-1
  -> keep Top-8 only for low-confidence symbols
  -> fit packet-local phase line from locked symbols
  -> allow phase to override Top-1 only when the line is reliable
  -> decode selected hard symbols and check CRC after peak selection
```

No payload bytes, templates, counters, CRC feedback, or cross-packet priors are
used during peak selection.

## Conservative Phase Override

The current default is intentionally conservative.  Multi-offset Top-1 remains
the default answer for each uncertain symbol.  A Top-8 alternative may replace
Top-1 only if all conditions pass:

```text
trimmed phase-line anchors >= 8
phase-line RMSE <= 0.25 pi
phase_score(candidate) - phase_score(Top1) >= 0.15
candidate energy drop from Top1 <= 0.60 dB
local_score(candidate) > local_score(Top1) + 0.06
```

Local score:

```text
local_score = 0.20 * phase_score + 0.80 * amp_score
```

This keeps the two-stage idea intact while preventing low-SNR false phase
tracks from rewriting strong first-stage evidence.

## Validation Summary

Outputs are under:

```text
data/symbol_phase_two_stage/
```

| Dataset | SNR | Center FFT SER | Multi-offset SER | Selected SER | CRC-valid rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0_0_0_10_14_8` | -20 dB | 0.366 | 0.046 | 0.046 | 0.600 |
| `0_0_0_10_14_8` | -23 dB | 0.794 | 0.334 | 0.331 | 0.000 |
| `0_0_0_10_14_16` | -20 dB | 0.358 | 0.016 | 0.016 | 0.636 |
| `0_0_0_10_14_32` | -20 dB | 0.396 | 0.143 | 0.139 | 0.714 |
| `0_0_0_10_14_32` | -23 dB | 0.776 | 0.343 | 0.343 | 0.000 |

The 3 dB gain claim is visible against the traditional center FFT argmax:

```text
payload_len=8:
  center FFT at -20 dB SER = 0.366
  selected two-stage at -23 dB SER = 0.331

payload_len=32:
  center FFT at -20 dB SER = 0.396
  selected two-stage at -23 dB SER = 0.343
```

So the new flow is at least 3 dB better than the traditional single-center-FFT
argmax baseline on these low-SNR validation sets.

## SNR Curve

Continuous `-20` to `-25 dB` results were exported to:

```text
data/symbol_phase_snr_curve/snr_m20_to_m25_summary.csv
data/symbol_phase_snr_curve/snr_m20_to_m25_packets.csv
data/symbol_phase_snr_curve/snr_m20_to_m25_summary.json
```

Mean across:

```text
0_0_0_10_14_8
0_0_0_10_14_16
0_0_0_10_14_32
```

| SNR | Center FFT SER | Multi-offset SER | Selected SER | CRC-valid rate |
| ---: | ---: | ---: | ---: | ---: |
| -20 dB | 0.373 | 0.068 | 0.067 | 0.650 |
| -21 dB | 0.519 | 0.120 | 0.114 | 0.448 |
| -22 dB | 0.646 | 0.204 | 0.199 | 0.130 |
| -23 dB | 0.767 | 0.332 | 0.330 | 0.000 |
| -24 dB | 0.849 | 0.506 | 0.505 | 0.000 |
| -25 dB | 0.910 | 0.649 | 0.647 | 0.000 |

Per-dataset selected SER:

| Dataset | -20 | -21 | -22 | -23 | -24 | -25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0_0_0_10_14_8` | 0.046 | 0.126 | 0.186 | 0.331 | 0.491 | 0.651 |
| `0_0_0_10_14_16` | 0.016 | 0.060 | 0.166 | 0.314 | 0.517 | 0.636 |
| `0_0_0_10_14_32` | 0.139 | 0.155 | 0.245 | 0.343 | 0.506 | 0.653 |

## Next Tuning Target

The current implementation is stable and does not materially hurt the strong
multi-offset baseline.  The remaining research target is to increase the phase
gain over multi-offset without raising false overrides.

Useful next sweeps:

```text
--phase-override-min-line-anchors
--phase-override-max-line-rmse-pi
--phase-override-min-gain
--phase-override-max-drop-db
--phase-override-score-margin
--top-l-low-confidence
```

Watch these metrics together:

```text
selected_symbol_ser
ser_gain_vs_multi_offset
false_lock_rate
uncertain_candidate_recall
phase_line_rmse_pi
crc_valid_rate
```

## Later Update - Current Default Selector

The current default is no longer the conservative Top-8 phase override above.
It was replaced by a low-complexity offset-phase-coherence selector:

```text
energy Top-24 candidates
  -> offset coherence per candidate bin
  -> small packet-line phase auxiliary score
  -> hard symbol bins
  -> LoRa codec + CRC after selection
```

Offset coherence:

```text
C_k[b] = |sum_o Z_o[k,b]| / (sum_o |Z_o[k,b]| + eps)
```

This uses phase consistency across oversampling offsets inside one packet and
one symbol.  It does not enumerate payload bytes and does not use CRC to choose
bins.

Updated formal threshold sweep:

```text
data/symbol_phase_threshold_sweep_coherence_default/
```

Mean gain vs traditional center FFT argmax:

```text
SER<=10% gain:       4.74 dB
accuracy>=90% gain:  4.74 dB
CRC/PRR>=90% gain:   4.14 dB
```

The SER target is just below 5 dB, while the CRC/PRR target is met.  Smooth
trajectory beam probes were tested after this and made the -22 dB SER worse, so
the current recommended system is the simpler offset-coherence selector.
