# Handoff: Offset-Coherence Ablation Experiments

Date: 2026-06-17

Scope:

```text
gr-lora_sdr/weakPacket_decoding copy
```

## Current System To Preserve

The current best low-complexity decoder is symbol-level, not byte-enumeration
or codec-beam search:

```text
header-first packet/timing/header parameters
  -> payload symbol FFT evidence
  -> multi-offset FFT energy fusion
  -> high-confidence symbol Top-1 lock
  -> low-confidence energy Top-24 candidates
  -> offset-phase coherence scoring
  -> small packet-line phase auxiliary score
  -> hard symbol bins
  -> LoRa codec decode
  -> CRC as final PRR metric only
```

It does not use:

```text
payload byte prior
CRC feedback for candidate choice
cross-packet or retransmission combining
application template / counter prior
```

Current default selector:

```text
selection_mode = coherence
top_l_low_confidence = 24
smooth_phase_weight = 0.05
smooth_amp_weight = 0.50
smooth_coherence_weight = 0.90
smooth_max_energy_drop_db = 20
```

Core phase feature:

```text
C_k[b] = |sum_o Z_o[k,b]| / (sum_o |Z_o[k,b]| + eps)
```

where `Z_o[k,b]` is the complex FFT value for payload symbol `k`, raw bin `b`,
and oversampling offset `o`.

Interpretation:

```text
Correct weak LoRa peaks tend to be complex-phase consistent across offsets.
Noise peaks can be strong in one offset but are less coherent across offsets.
```

## Current Formal Result

Formal sweep output:

```text
data/symbol_phase_threshold_sweep_coherence_default/
```

Mean across:

```text
0_0_0_10_14_8
0_0_0_10_14_16
0_0_0_10_14_32
```

Threshold gains vs traditional center FFT argmax:

| Metric | Center threshold | Selected threshold | Gain |
| --- | ---: | ---: | ---: |
| SER <= 10% | -17.23 dB | -21.97 dB | 4.74 dB |
| accuracy >= 90% | -17.23 dB | -21.97 dB | 4.74 dB |
| CRC/PRR >= 90% | -15.17 dB | -19.31 dB | 4.14 dB |
| CRC/PRR >= 80% | -15.94 dB | -20.31 dB | 4.37 dB |
| CRC/PRR >= 50% | -17.11 dB | -21.62 dB | 4.51 dB |

Mean SNR curve:

| SNR | Center SER | Multi SER | Selected SER | Selected accuracy | Selected CRC/PRR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -20 dB | 0.373 | 0.068 | 0.044 | 0.956 | 0.858 |
| -21 dB | 0.519 | 0.120 | 0.060 | 0.940 | 0.668 |
| -22 dB | 0.646 | 0.204 | 0.101 | 0.899 | 0.397 |
| -23 dB | 0.767 | 0.332 | 0.184 | 0.816 | 0.114 |
| -24 dB | 0.849 | 0.506 | 0.314 | 0.686 | 0.000 |
| -25 dB | 0.910 | 0.649 | 0.446 | 0.554 | 0.000 |

Main status:

```text
CRC/PRR target is met with margin.
SER target is very close to 5 dB but currently 4.74 dB.
Further smooth-beam attempts increased complexity and worsened -22 dB SER.
```

## Goal Of Next Work

Run a clean ablation study to quantify which component creates the gain:

```text
1. center FFT argmax baseline
2. multi-offset energy fusion
3. high-confidence Top-1 locking
4. candidate size Top-L
5. offset-phase coherence
6. packet-line phase auxiliary score
7. optional coherence-candidate expansion
8. optional smooth path/trajectory beam
```

The desired output is a paper-style table that can answer:

```text
How much gain comes from multi-offset evidence?
How much additional gain comes from offset-phase coherence?
Does packet-line phase help beyond offset coherence?
Does high-confidence locking prevent damage?
Is Top-24 really the best low-complexity candidate size?
```

## Primary Script

Use:

```text
scripts/experiments/run_symbol_phase_threshold_sweep.py
```

This script reports, for center, multi-offset, and selected:

```text
symbol SER
symbol accuracy
CRC valid rate / PRR
threshold_table.csv
gain_table.csv
per_packet_metrics.csv
snr_curve_summary.csv
summary.json
```

Recommended SNR range for formal ablation:

```powershell
--snr-start -12 --snr-stop -26 --snr-step -1
```

Fast pre-screen range:

```powershell
--snr-start -20 --snr-stop -23 --snr-step -1
```

## Ablation Matrix

### A0 - Traditional center FFT argmax

Already included by every sweep as `center`.

This is the standard LoRa PHY comparison:

```text
fixed center/downsample phase
  -> dechirp
  -> N=2^SF FFT
  -> argmax energy bin
```

### A1 - Multi-offset energy argmax

Already included by every sweep as `multi`.

This isolates the gain from using all oversampling offsets but no second-stage
coherence re-ranking.

### A2 - Energy-only selected path

Purpose:

```text
Check whether the selected-path machinery itself helps, without phase coherence.
Expected to be close to multi-offset argmax.
```

Command:

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --selection-mode coherence `
  --top-l-low-confidence 24 `
  --smooth-phase-weight 0.0 `
  --smooth-amp-weight 1.0 `
  --smooth-coherence-weight 0.0 `
  --smooth-max-energy-drop-db 20 `
  --snr-start -12 --snr-stop -26 --snr-step -1 `
  --output-dir data\ablation_energy_only_top24
```

### A3 - Offset coherence only

Purpose:

```text
Measure how far phase coherence can go without amplitude protection.
Expected to help some ambiguous symbols but may select coherent noise peaks.
```

Command:

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --selection-mode coherence `
  --top-l-low-confidence 24 `
  --smooth-phase-weight 0.0 `
  --smooth-amp-weight 0.0 `
  --smooth-coherence-weight 1.0 `
  --smooth-max-energy-drop-db 20 `
  --snr-start -12 --snr-stop -26 --snr-step -1 `
  --output-dir data\ablation_coherence_only_top24
```

### A4 - Energy + offset coherence, no packet-line phase

Purpose:

```text
Isolate the main offset-coherence gain without the global packet phase-line term.
```

Command:

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --selection-mode coherence `
  --top-l-low-confidence 24 `
  --smooth-phase-weight 0.0 `
  --smooth-amp-weight 0.50 `
  --smooth-coherence-weight 0.90 `
  --smooth-max-energy-drop-db 20 `
  --snr-start -12 --snr-stop -26 --snr-step -1 `
  --output-dir data\ablation_amp_coherence_no_line
```

### A5 - Current default: energy + offset coherence + small packet-line phase

Purpose:

```text
Main method.  This is the current recommended result.
```

Command:

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --snr-start -12 --snr-stop -26 --snr-step -1 `
  --output-dir data\ablation_current_default
```

### A6 - Packet-line phase only

Purpose:

```text
Show that global packet-line phase alone is not the main gain source.
```

Command:

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --selection-mode coherence `
  --top-l-low-confidence 24 `
  --smooth-phase-weight 1.0 `
  --smooth-amp-weight 0.0 `
  --smooth-coherence-weight 0.0 `
  --smooth-max-energy-drop-db 20 `
  --snr-start -12 --snr-stop -26 --snr-step -1 `
  --output-dir data\ablation_packet_line_only
```

### A7 - Top-L candidate size

Purpose:

```text
Find the smallest candidate size that keeps most of the gain.
```

Run:

```powershell
foreach ($L in 8,16,24,32) {
  python scripts\experiments\run_symbol_phase_threshold_sweep.py `
    --selection-mode coherence `
    --top-l-low-confidence $L `
    --smooth-phase-weight 0.05 `
    --smooth-amp-weight 0.50 `
    --smooth-coherence-weight 0.90 `
    --smooth-max-energy-drop-db 20 `
    --snr-start -12 --snr-stop -26 --snr-step -1 `
    --output-dir "data\ablation_topL_$L"
}
```

Expected:

```text
Top-8 is likely too narrow around -22 dB.
Top-24 is currently the best low-complexity tradeoff.
Top-32 may not improve and can admit more false coherent peaks.
```

### A8 - High-confidence locking ablation

Purpose:

```text
Check whether locking reliable symbols prevents phase/coherence overcorrection.
```

Disable most locks by making the lock thresholds impossible:

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --selection-mode coherence `
  --top-l-low-confidence 24 `
  --lock-margin-db 999 `
  --lock-peak-to-median-db 999 `
  --smooth-phase-weight 0.05 `
  --smooth-amp-weight 0.50 `
  --smooth-coherence-weight 0.90 `
  --smooth-max-energy-drop-db 20 `
  --snr-start -12 --snr-stop -26 --snr-step -1 `
  --output-dir data\ablation_no_high_conf_lock
```

Expected:

```text
If SER worsens, high-confidence locking is protecting already-correct strong
symbols from unnecessary re-ranking.
```

### A9 - Coherence-candidate expansion

Purpose:

```text
Test whether adding coherence Top-C candidates beyond energy Top-L helps.
```

Command:

```powershell
foreach ($C in 32,64,128) {
  python scripts\experiments\run_symbol_phase_threshold_sweep.py `
    --selection-mode coherence `
    --top-l-low-confidence 24 `
    --coherence-candidate-top-l $C `
    --smooth-phase-weight 0.05 `
    --smooth-amp-weight 0.50 `
    --smooth-coherence-weight 0.90 `
    --smooth-max-energy-drop-db 20 `
    --snr-start -20 --snr-stop -23 --snr-step -1 `
    --output-dir "data\ablation_coherence_candidate_top$C"
}
```

Current probes suggested this can hurt by admitting coherent noise peaks.  Keep
it as an ablation, not as the default, unless the full sweep proves otherwise.

### A10 - Smooth trajectory beam

Purpose:

```text
Document that extra trajectory complexity does not currently justify itself.
```

Command:

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --selection-mode smooth `
  --top-l-low-confidence 24 `
  --smooth-phase-weight 0.05 `
  --smooth-amp-weight 0.50 `
  --smooth-coherence-weight 0.90 `
  --smooth-slope-penalty 0.00 `
  --smooth-curvature-penalty 0.10 `
  --smooth-max-energy-drop-db 20 `
  --smooth-min-line-anchors 2 `
  --smooth-min-locked-ratio 0 `
  --smooth-max-line-rmse-pi 1 `
  --beam-width 256 `
  --snr-start -20 --snr-stop -23 --snr-step -1 `
  --output-dir data\ablation_smooth_beam_probe
```

Previous probes worsened -22 dB SER, so this should be reported as a negative
result unless retuning changes the picture.

## Metrics To Compare

Primary:

```text
SER <= 10% threshold
accuracy >= 90% threshold
CRC/PRR >= 90% threshold
```

Secondary:

```text
CRC/PRR >= 80%
CRC/PRR >= 50%
selected SER at -20, -21, -22, -23 dB
mean_locked_ratio
mean_uncertain_candidate_recall
mean_selected_offset_coherence
false_lock_rate
```

Important interpretation:

```text
CRC valid rate is PRR-like packet success.
CRC is not allowed to choose candidates.
```

## Suggested Paper-Style Ablation Table

Build a table like:

| Variant | Multi-offset energy | Top-L | Locking | Offset coherence | Packet line | SER gain | CRC90 gain |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| center argmax | no | 1 | no | no | no | 0.00 | 0.00 |
| multi argmax | yes | 1 | no | no | no | TBD | TBD |
| energy-only selected | yes | 24 | yes | no | no | TBD | TBD |
| coherence only | yes | 24 | yes | yes | no | TBD | TBD |
| amp + coherence | yes | 24 | yes | yes | no | TBD | TBD |
| current default | yes | 24 | yes | yes | small | 4.74 | 4.14 |
| no high-conf lock | yes | 24 | no | yes | small | TBD | TBD |
| smooth beam | yes | 24 | yes | yes | smooth | TBD | TBD |

Use `gain_table.csv` from each output directory to fill `SER gain` and
`CRC90 gain`.

## Expected Main Claim

If the ablation follows current evidence, the clean claim should be:

```text
Most of the traditional-PHY gain comes from multi-offset FFT evidence.
The additional gain over multi-offset comes from offset-phase coherence, which
uses complex phase consistency across oversampling offsets to reject incoherent
noise peaks inside the symbol-level candidate set.
Global packet-line phase is useful as a small auxiliary term, but it is not the
dominant source of the current gain.
```

## Watchouts

1. Do not compare against an oversampled or multi-offset method and call it
   "traditional LoRa PHY".  The traditional baseline is center/downsample FFT
   argmax.
2. Do not use CRC-valid candidates to choose bins.  CRC remains a final metric.
3. Do not reintroduce payload byte enumeration as the main path.
4. Do not overclaim `SER >= 5 dB`; current verified mean is `4.74 dB`.
5. If a complex method only improves one packet but worsens mean threshold, keep
   it as a negative ablation rather than default behavior.

## Minimal Next-Step Checklist

```text
1. Run A2-A6 full sweeps.
2. Run A7 Top-L full sweeps.
3. Run A8 no-lock full sweep.
4. Run A9/A10 only on -20..-23 unless a probe looks promising.
5. Create one combined ablation CSV/table from all gain_table.csv files.
6. Update THRESHOLD_GAIN_EVALUATION_2026-06-16.md with the final ablation table.
```

