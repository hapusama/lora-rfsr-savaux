# Island DP Reconstruction Variant

This folder holds the next Stage-2 direction: lock high-confidence Stage-1
symbols as anchors, split the remaining low-confidence symbols into islands,
and run constrained Viterbi only inside those islands.  The DP state is now
two-dimensional:

```text
(candidate_raw_bin, oversampling_branch)
```

High-confidence anchors remain immutable; low-confidence islands may move
between oversampling branches, with branch jumps penalized against the
anchor-interpolated fractional-STO trend.

The implementation entry point is:

```python
select_island_reconstruction_viterbi_path(...)
```

The current best validated baseline remains:

```text
Savaux over-sampled coherent Stage-1 + v1 one-order phase DP
```

This variant is intentionally separate because it should be benchmarked as a
new research branch, not silently mixed into the winning v1 path.

## Evidence Needed From Stage 1

Minimum existing evidence:

- `center_spectra`: complex combined spectrum per symbol.
- `evidence_powers`: Stage-1 power likelihood per raw bin.
- `offset_coherences`: currently `branch_phase_agreement`, used as a proxy for
  branch consistency and anchor reliability.

New useful evidence now exposed by `savaux_stage1.py`:

- `branch_spectra`: per-oversampling-branch complex FFT spectra.
- `dechirped_symbols`: optional retained dechirped oversampled IQ per symbol
  when `SavauxStage1Config(retain_dechirped_symbols=True)` is used.

The selector uses `branch_spectra` to pick anchor branches and to keep a useful
fallback path.  When `dechirped_symbols` is available, each `(bin, branch)` state
is scored by candidate-specific two-segment reconstruction: split at the LoRa
wrap point, compensate the suffix with the local fractional-STO estimate, and
use the stitched complex phase for the island transition penalty.

## Current Default Profile And Evidence Fusion

The default island profile is now the pure anchor-locked JTRD path:

- High-confidence Stage-1 anchors are still hard locked.
- Low-confidence islands run 2D `(bin, branch)` DP only when both left and
  right anchors exist.
- The v1 one-order DP path is no longer used as a runtime prior.  It remains a
  benchmark only.
- Per-branch framesync timing can be passed as
  `branch_residual_sto_chips[symbol][branch]`; when present, two-segment
  reconstruction uses the branch-specific instantaneous STO for the selected
  state.

This keeps the algorithm aligned with the physical model: locked anchors protect
the high-confidence base, while low-confidence islands are allowed to move
without a changed-symbol count guard.

The strongest FFT-bin-only direction found so far is not the raw island DP
alone.  It is dual Stage-1 evidence fusion:

```text
old center phase + product_norm(old power, residual-STO-corrected power)
```

This is implemented in `dual_evidence.py`.  It still only changes FFT-bin
evidence and path selection; it does not use payload codec or CRC feedback.

Validated 32-symbol product-fusion command:

```powershell
python weak_decoder\phase_line\variants\island_dp_reconstruction\evaluate_island_dp.py `
  --datasets 0_0_0_10_14_32 `
  --snrs -25 -26 -27 -28 -29 `
  --seeds 42 43 44 45 46 `
  --max-packets 10 `
  --v1-top-l 16 `
  --fusion-mode product_norm `
  --fusion-corrected-weight 0.5 `
  --output-dir weak_decoder\phase_line\variants\island_dp_reconstruction\_eval\fusion_product32_seed42_46_snr25_29
```

Result on the current local fixtures:

```text
groups                 : 25
packets                : 175
symbols                : 6125
v1 errors              : 534
corrected-v1 errors    : 523
product-fusion errors  : 515
raw-island errors      : 532
fusion-island errors   : 515
product-fusion net     : +19 symbols vs v1
fusion-island net      : +19 symbols vs v1
fusion locked E3 count : 0
fusion rescue/break    : 30 / 11 vs v1
island over fusion     : 2 rescue / 2 break, net 0
```

An auxiliary island experiment can also pass old/corrected power rows into the
island candidate generator:

```powershell
python weak_decoder\phase_line\variants\island_dp_reconstruction\evaluate_island_dp.py `
  --datasets 0_0_0_10_14_32 `
  --snrs -25 -26 -27 -28 -29 `
  --seeds 42 43 44 45 46 `
  --max-packets 10 `
  --fusion-mode product_norm `
  --anchor-margin-db 3.5 `
  --anchor-peak-to-median-db 11.0 `
  --anchor-min-coherence 0.92 `
  --allow-third-bin-when-baseline-differs `
  --third-bin-penalty 0.0 `
  --baseline-bonus 0.0 `
  --island-accept-margin 0.0 `
  --auxiliary-top-l 40 `
  --auxiliary-energy-weight 0.25 `
  --use-aux-evidence `
  --output-dir weak_decoder\phase_line\variants\island_dp_reconstruction\_eval\aux_product32_seed42_46_snr25_29
```

That run reached 512 errors, net +22 symbols vs v1 and +3 over product fusion.
However, a short low-SNR check at -30..-32 dB / seeds 42..43 was not stable:
v1 243 errors, product fusion 236 errors, aux fusion-island 241 errors.  So aux
island is kept as an experiment knob rather than the recommended profile.

## Current Diagnosis

The main bottleneck is local evidence, not the island DP transition:

- With product fusion, only 21/515 remaining errors were in default locked
  anchors; stricter anchors reduced locked errors to 0.
- For fusion-v1 misses, GT recall in power candidate sets was limited:
  Top40 union of fusion/old/corrected powers covered 320/515 misses, Top128
  covered 412/515.
- Among GT-present Top128 union candidates, simple power scores still ranked GT
  poorly: best observed rank-1 count was only 22/412.
- Two-segment reconstruction did not fix ranking.  On a seed42..43 / -25..-27
  probe, reconstruction quality alone had median GT rank 52; branch-power and
  mixed scores were only modestly better than power.
- Independent per-branch phase DP is not a rescue path.  On the same probe,
  branch-path SER was 18%-23%, and only 3/101 fusion misses were corrected by
  any branch path.

So the current Python result is a small but real FFT-bin-only improvement, not
the desired 1-2 dB gain yet.  The next useful research target is a stronger
candidate-local score or Stage-1 branch alignment feature that can move GT from
Top10/Top40 into Top1 more often.

## Multi-Origin Sampling-Phase Fusion

`multi_origin_evidence.py` adds the most recent FFT-bin-only experiment.  It
generalizes the original center-sampling mouthfeel by building Stage-1 evidence
for all sampling origins `0..os_factor-1`, then fusing their per-bin powers.
By default, the complex phase used by the phase DP still comes from one
configured `phase_origin`.  A newer optional phase mouthfeel,
`--multi-phase-mode weighted_unit`, also fuses the per-bin complex phase across
origins, using each origin's normalized bin power as the phase weight.  This
keeps the phase used by the DP consistent with the multi-origin power evidence.

The raw multi-origin product fusion has much higher rescue but also too many
breaks:

```text
v1 errors      : 534
dual errors    : 515
multi errors   : 532
multi rescue/break vs v1: 52 / 50
```

Because of that, the module also includes a conservative gate:

```python
arbitrate_dual_vs_multi_origin_path(...)
```

The gate chooses the multi-origin path only when its path-quality metrics beat
the dual-evidence path.  Current conservative setting:

```text
multi_trajectory_score - dual_trajectory_score >= 0.003267
```

Validated current-best command:

```powershell
python weak_decoder\phase_line\variants\island_dp_reconstruction\evaluate_multi_origin.py `
  --datasets 0_0_0_10_14_32 `
  --snrs -25 -26 -27 -28 -29 `
  --seeds 42 43 44 45 46 `
  --max-packets 10 `
  --multi-mode sum_norm `
  --multi-phase-mode weighted_unit `
  --enable-gate `
  --enable-symbol-gate `
  --symbol-min-trajectory-gain 0.005618281741596176 `
  --symbol-min-old-power-margin -999 `
  --symbol-min-old-multi-norm 0.7303704876428979 `
  --output-dir weak_decoder\phase_line\variants\island_dp_reconstruction\_eval\multi_sum_weighted_unit_symbolgate_oldnorm32_seed42_46_snr25_29
```

Current best result on the 32-symbol fixture:

```text
groups              : 25
symbols             : 6125
v1 errors           : 534
dual-fusion errors  : 515
raw multi errors    : 515
gated multi errors  : 497
gated net vs v1     : +37 symbols
gated net vs dual   : +18 symbols
gated rescue/break vs v1  : 46 / 9
gated rescue/break vs dual: 21 / 3
symbol-gated changes vs dual: 68 symbols
```

The newest recommended symbol gate came from
`diagnose_dual_multi_symbol_gate.py`, not from CRC or codec feedback.  The
diagnostic rebuilds dual and multi FFT-bin paths, records only symbols where
the two paths disagree, and sweeps runtime evidence features with a train/test
split.

The current 497-error run adds optional symbol-level arbitration on top of the
packet gate.  For symbols where dual and multi disagree, it accepts the multi
bin only when the packet-level trajectory gain is strong enough and the multi
bin is itself strong in the original old-center Stage-1 power:

```text
multi_trajectory_score - dual_trajectory_score >= 0.005618281741596176
old_power[multi_bin] / old_power_peak >= 0.7303704876428979
```

This is still FFT-bin evidence only; no CRC, codec, payload template, or GT
feedback is used.

Holdout check for that gate:

```text
train seeds 42-44: dual 308 -> gated 299, rescue/break 12 / 3
test  seeds 45-46: dual 207 -> gated 198, rescue/break  9 / 0
all   seeds 42-46: dual 515 -> gated 497, rescue/break 21 / 3
```

This is cleaner than the previous margin-only symbol gate, which also reached
497 errors but used 81 changes and had 23 / 5 rescue/break vs dual.  The
old-norm gate uses 68 changes and keeps test-set breaks at zero in this split.

This is the current best Python result in this folder, but it is still a small
SER reduction rather than a 1-2 dB horizontal SNR shift.

`--multi-corrected` was also tested with the same gate.  It regressed to 514
errors, so the current best keeps multi-origin as plain `sum_norm` power fusion
and leaves residual-STO correction to the dual-evidence baseline.  Alternate
fixed phase origins were tested too: phase origin 1 tied at 501 errors, phase
origin 3 gave 502, and phase origin 0 regressed to 514.  The new
`weighted_unit` phase fusion improves this fixed-origin phase mouthfeel from
501 to 499 errors.

Cross-dataset check with the same gate was run per dataset to avoid a long
single command timeout:

```powershell
python weak_decoder\phase_line\variants\island_dp_reconstruction\evaluate_multi_origin.py `
  --datasets <one of: 0_0_0_10_14_8, 0_0_0_10_14_16, 0_0_0_10_14_32> `
  --snrs -25 -26 -27 -28 -29 `
  --seeds 42 43 44 45 46 `
  --max-packets 10 `
  --multi-mode sum_norm `
  --multi-phase-mode weighted_unit `
  --enable-gate `
  --enable-symbol-gate `
  --symbol-min-trajectory-gain 0.005618281741596176 `
  --symbol-min-old-power-margin -999 `
  --symbol-min-old-multi-norm 0.7303704876428979 `
  --output-dir weak_decoder\phase_line\variants\island_dp_reconstruction\_eval\multi_sum_weighted_unit_symbolgate_oldnorm_<dataset>_seed42_46_snr25_29
```

```text
dataset 0_0_0_10_14_8:
  v1 / dual / multi / gated errors: 0 / 0 / 0 / 0

dataset 0_0_0_10_14_16:
  v1 / dual / multi / gated errors: 0 / 0 / 10 / 0

dataset 0_0_0_10_14_32:
  v1 / dual / multi / gated errors: 534 / 515 / 515 / 497

all three datasets:
  v1 / dual / multi / gated errors: 534 / 515 / 525 / 497
  gated rescue/break vs dual: 21 / 3, changes 68
```

The current gain is therefore concentrated on the 32-symbol weak fixture.  The
8/16-symbol fixtures in this SNR slice are already saturated at zero v1 errors,
so they provide little rescue room and mainly expose break risk.

Robust origin-fusion modes were also checked on the 32-symbol fixture with the
same gate:

```text
sum_norm gated          : 501 errors
sum_norm + weighted_unit phase gated: 499 errors
sum_norm + weighted_unit phase + symbol gate: 497 errors
median_norm gated       : 505 errors
trimmed_mean_norm gated : 505 errors
top2_mean_norm gated    : 574 errors
```

So the current best remains plain `sum_norm` for power, but with
`weighted_unit` phase fusion and the optional symbol gate.  Taking the largest
origins is too vulnerable to noise peaks; median/trimmed fusion is safer but
gives up too much rescue.

The weighted-unit symbol-level oracle is still much lower at 475 errors
relative to the dual/multi pair, so there is theoretical room left.  The current
simple two-feature symbol gate captures only a small part of that room:
23 rescues and 5 breaks versus dual.  This reinforces the diagnosis that
stronger local evidence is needed before more path heuristics will buy a real
1-2 dB shift.
