# Local Window Phase Selector

Date: 2026-06-18

Scope: `weak_decoder/symbol_phase_two_stage.py`

## Claim

The old packet phase-line idea should not be used as a strong global linear
model.  A wrong selector can produce a very high global R2 while tracing a bad
phase path.  The more defensible PHY-only use of phase is local smoothness:

```text
FFT-bin candidates are still chosen only from per-symbol FFT evidence.
Phase only gives a small local smoothness vote among Top-L candidates.
No codec, CRC, payload byte prior, counter prior, or cross-packet information is
used to choose bins.
```

## Implemented Mode

New selection mode:

```text
--selection-mode window
```

Core behavior:

1. Build per-symbol Top-L candidates from multi-offset FFT evidence.
2. Keep high-confidence locked symbols at Top-1.
3. For each symbol, estimate a local phase prediction from nearby locked
   anchors within `window_anchor_span`.
4. If local anchors are unavailable or too inconsistent, fall back to the header
   phase line; if that is also unavailable, ignore phase for that symbol.
5. Score candidates by:

```text
score =
  window_amp_weight       * normalized FFT evidence
+ window_coherence_weight * offset coherence
+ window_phase_weight     * local phase residual score
- optional slope/curvature penalties
```

Current conservative defaults:

```text
window_phase_weight = 0.05
window_amp_weight = 0.50
window_coherence_weight = 0.90
window_slope_weight = 0.00
window_curvature_weight = 0.00
window_size = 5
window_anchor_span = 8.0
window_anchor_min = 2
window_anchor_max_rmse_pi = 0.40
window_min_locked_ratio = 0.10
```

These defaults intentionally keep phase as a weak auxiliary term.  Strong phase
weights were observed to follow false smooth trajectories.

## Why This Is Different From Global Phase Line

The previous global line metric can prefer an internally consistent but wrong
sequence.  The local-window mode does not trust a single packet-wide line as a
truth model.  It asks a narrower question:

```text
Does this candidate continue the nearby trusted-symbol phase neighborhood?
```

This keeps the method within FFT-bin selection while making it less sensitive
to global slope mismatch and high-R2 false paths.

## Quick Probe

Input:

```text
data/low_snr_gt_bin/0_0_0_10_14_16_m22_m27_sto_input/
data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols.csv
```

Commands:

```powershell
python scripts\experiments\run_symbol_phase_two_stage.py `
  -i data\low_snr_gt_bin\0_0_0_10_14_16_m22_m27_sto_input\0_0_0_10_14_16_snr_m23dB.bin `
  -s data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv `
  -o data\window_phase_local_probe_m23\window_default_results.csv `
  --summary-json data\window_phase_local_probe_m23\window_default_summary.json `
  --selection-mode window --beam-width 64
```

Compare to:

```powershell
python scripts\experiments\run_symbol_phase_two_stage.py `
  -i data\low_snr_gt_bin\0_0_0_10_14_16_m22_m27_sto_input\0_0_0_10_14_16_snr_m23dB.bin `
  -s data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv `
  -o data\window_phase_local_probe_m23\coherence_default_results.csv `
  --summary-json data\window_phase_local_probe_m23\coherence_default_summary.json `
  --selection-mode coherence
```

Small probe results on one noisy IQ set:

| SNR | coherence selected SER | window selected SER | CRC both |
| ---: | ---: | ---: | ---: |
| -22 | 0.060 | 0.057 | 0.364 |
| -23 | 0.166 | 0.161 | 0.000 |
| -24 | 0.296 | 0.288 | 0.000 |

Interpretation:

```text
The current local-window phase vote is modest, but it is not just dead code:
with conservative weights it gives small SER improvements in this probe without
using codec/CRC selection.  Aggressive phase weights were harmful, which supports
the decision to avoid global or strong phase-line control.
```

## Full Existing Noisy-IQ Batch

Batch runner:

```text
scripts/experiments/run_local_window_phase_batch.py
```

Command:

```powershell
python scripts\experiments\run_local_window_phase_batch.py `
  --output-dir data\local_window_phase_batch_guarded_v2_m22_m27
```

Summary:

| SNR | coherence SER | window SER | window_guarded SER |
| ---: | ---: | ---: | ---: |
| -22 | 0.0597 | 0.0571 | 0.0571 |
| -23 | 0.1662 | 0.1610 | 0.1532 |
| -24 | 0.2961 | 0.2909 | 0.2935 |
| -25 | 0.4364 | 0.4338 | 0.4312 |
| -26 | 0.5870 | 0.5870 | 0.5870 |
| -27 | 0.7299 | 0.7299 | 0.7299 |

The current guarded local-window phase vote is small but consistently safe on
this existing noisy-IQ batch.  It helps the transition region without hurting the
low-locked tail because it falls back to coherence when `locked_ratio < 0.10`.
This is not yet a large paper-level gain, but it is a clean FFT-bin-only
mechanism worth refining.

## Guarded Repair Note

A stricter `window_guarded` mode starts from the existing coherence selection and
only allows local phase to repair a low-confidence symbol when the phase gain is
large and energy/coherence losses are bounded.

The first guarded prototype was worse because it accidentally started from the
multi-offset Top-1 path instead of the coherence-selected path.  After fixing
that baseline and adding `window_min_locked_ratio`, `window_guarded` became the
best mode in the `m22..m27` batch above.

Recommended next default candidate: `--selection-mode window_guarded`.

## Next Work

Recommended next sweep:

```powershell
python scripts\experiments\run_symbol_phase_threshold_sweep.py `
  --selection-mode window `
  --snr-start -20 --snr-stop -24 --snr-step -1 `
  --output-dir data\ablation_local_window_phase_probe_m20_m24
```

Useful ablations:

```text
window_phase_weight: 0.00, 0.02, 0.05, 0.10
window_anchor_span: 4, 8, 12
window_anchor_max_rmse_pi: 0.25, 0.40, 0.60
window_degree: 0, 1
```

Also test adaptive gating based on local anchor count and local anchor RMSE; the
current locked-ratio fallback is deliberately simple.

Do not evaluate this as a codec-assisted decoder.  The intended claim is only a
PHY-layer FFT-bin selection improvement.
