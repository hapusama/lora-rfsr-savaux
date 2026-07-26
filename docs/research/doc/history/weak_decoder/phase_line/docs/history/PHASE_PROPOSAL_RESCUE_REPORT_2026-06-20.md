# Phase-aware proposal rescue report (2026-06-20)

## Goal

This iteration targets the Stage-1 bottleneck observed at low SNR: energy-only
Top-L proposal sets miss many ground-truth FFT bins, so Stage-2 phase DP cannot
recover them.

The implementation stays inside `weak_decoder/phase_line/` and changes only the
phase-line selector path.

## Implemented changes

### 1. Gated phase/coherence proposal rescue

Stage 1 now keeps the normal multi-offset energy Top-L as the base candidate
list, then selectively replaces weak tail candidates with bins that have much
stronger offset phase-coherence.

The replacement is gated:

```text
candidate_coherence > tail_candidate_coherence + phase_proposal_coherence_min_gain
```

Default rescue settings:

```text
phase_proposal_coherence_rescue_count = 12
phase_proposal_coherence_min_gain     = 0.10
```

This avoids the earlier failure mode where blindly cutting the energy tail
improved some packets but damaged others.

### 2. Energy backup inside first-order Viterbi

For first-order phase DP, the actual Viterbi candidate set also appends the
original energy Top-L bins as a backup. This means Stage 1 can expose a
phase-aware Top-L set for recall while Stage 2 does not lose the old
energy-only escape path.

The backup is limited to first-order DP. Second-order DP keeps the compact
Top-L set because it is slower and has not been the best-performing path.

### 3. Adaptive Top-L widening

The default path now keeps `top_l=24` as the base width. It has two packet-local
widening gates:

```text
adaptive_top_l_high             = 40
adaptive_top_l_anchor_threshold = 10
adaptive_top_l_mid              = 28
adaptive_top_l_mid_anchor_threshold = 8
adaptive_top_l_mid_max_anchor_rmse_pi = 0.30
```

This is a packet-local confidence gate, not an SNR label.  It keeps the `-25 dB`
behavior unchanged on the current dataset while recovering a small amount of
`-24 dB` selection headroom.  The mid-width gate handles packets with 8-9 hard
anchors only when those anchors form a reasonably clean phase line.

### 4. High-confidence hard anchors

`PhasePathSelectorConfig` now has separate hard-anchor thresholds:

```text
hard_anchor_top_k
hard_anchor_margin_db
hard_anchor_peak_to_median_db
hard_anchor_min_coherence
```

When a symbol is genuinely far ahead in energy, Viterbi can restrict it to
Top-1 and treat it as a hard local anchor. These thresholds are intentionally
stricter than the older soft high-confidence fields.

The current default uses coherence plus peak-to-median rather than energy
margin:

```text
hard_anchor_margin_db          = 0.0
hard_anchor_peak_to_median_db  = 6.0
hard_anchor_min_coherence      = 0.85
```

This is important at `-24/-25 dB`: the Top-1 vs Top-2 margin is often small,
but a high offset phase-coherence plus a clear peak-to-median separation is a
much better low-SNR reliability signal.

### 5. Anchor-only sliding-window refine

A local sliding-window phase-line post-refine was first tested against the full
DP path and reduced performance because it reinforced wrong neighbors.  The
current default is narrower: fit the local window only from hard anchors, then
allow a small number of candidate switches when the local anchor line agrees
with energy/coherence evidence.

```text
sliding_window_refine_enabled = True
sliding_window_radius         = 6
sliding_window_min_anchors    = 4
sliding_window_max_rmse_pi    = 0.25
sliding_window_phase_weight   = 0.15
sliding_window_energy_weight  = 0.35
sliding_window_coherence_weight = 0.50
```

This implements the "sliding window" direction without trusting the whole
selected path as anchors.

### 6. Anchor phase bias and anchor-slope transition

The final path selector now uses hard anchors in two conservative ways:

```text
anchor_phase_bias_weight = 0.03
anchor_phase_bias_span   = 8.0
anchor_slope_weight      = 0.05
anchor_slope_max_rmse_pi = 0.25
```

`anchor_phase_bias` adds a small local-score bonus to candidates that are close
to a local phase line fitted from nearby hard anchors.  `anchor_slope_weight`
estimates a payload-local phase slope from hard anchors and uses it as the
first-order transition reference.  This is more faithful to the observed
packet-local phase trajectory than penalizing every adjacent phase increment
toward zero.

### 7. Adaptive phase relaxation

Selectable-miss diagnostics showed that simply strengthening sliding-window
phase scoring is unsafe: for most selectable misses, the wrong selected path is
also more locally phase-consistent than the GT bin.  The useful signal is more
specific: when a packet has many hard anchors, a non-zero payload phase slope is
more trustworthy, and the global first-order penalty toward zero can be relaxed.

The default now applies this only when hard-anchor count is at least 10:

```text
adaptive_phase_relax_anchor_threshold       = 10
adaptive_phase_relax_first_order_weight     = 0.10
adaptive_phase_relax_anchor_slope_weight    = 0.08
adaptive_phase_relax_anchor_slope_max_rmse_pi = 0.35
```

Together with adaptive Top-L widening, this lowered `-24 dB` DP1 SER without
changing the `-25 dB` result.

### 8. Gated hard-anchor phase proposal

Using every hard anchor as a phase-proposal anchor was not stable, but the
packet-level diagnostics showed a narrower useful regime: a medium number of
hard anchors whose phases form a very clean local line.  The selector now has a
separate gate:

```text
phase_proposal_gated_hard_anchors = True
phase_proposal_gated_hard_anchor_min_count = 5
phase_proposal_gated_hard_anchor_max_count = 8
phase_proposal_gated_hard_anchor_max_rmse_pi = 0.12
```

Only in that regime are hard anchors allowed to participate in Stage-1
phase-aware proposal.  This rescued two `-25 dB` packets in the current dataset
without reopening the earlier high-anchor-count failure cases.

### 9. Adaptive hard-anchor softening

Hard anchors are mostly reliable, but at `-24/-25 dB` a few low-margin hard
anchors are wrong and absolute Top-1 locking makes them unrecoverable.  A
global Top-2/Top-3 relaxation hurts, so the final selector only softens hard
anchors when the packet has many hard anchors and their phase line is still
coherent:

```text
adaptive_hard_anchor_softening_enabled = True
adaptive_hard_anchor_softening_min_count = 10
adaptive_hard_anchor_softening_max_rmse_pi = 0.50
adaptive_hard_anchor_softening_top_k = 3
adaptive_hard_anchor_softening_max_margin_db = 0.50
```

This keeps normal hard anchors at Top-1, but lets low-margin hard anchors keep
up to three candidates in packets where the surrounding anchors are trustworthy.
It improved the `-24 dB` point and left `-25 dB` unchanged.

### 10. Packet-local path arbiter

The remaining failures include packets whose true payload phase is not very
smooth.  In those cases the first-order DP can over-smooth and choose a path
that is internally consistent but wrong.  A small packet-local arbiter now
checks the selected DP path before final reporting:

```text
path_arbiter_enabled = True
path_arbiter_first_abs_pi_threshold = 0.35
path_arbiter_second_abs_pi_threshold = 0.42
```

If the DP-selected path has unusually large first- or second-difference phase
motion, the selector reruns the older local-coherence/v3 selector and uses that
path instead.  The gate uses only runtime path statistics, not GT or CRC.  It
helped the high-phase-motion packets where enforcing a global first-order
smoothness prior was too strong.

## Verification

Dataset:

```text
0_0_0_10_14_16
```

Command:

```powershell
python scripts/experiments/phase_line/run_phase_path_ablation.py `
  --datasets 0_0_0_10_14_16 `
  --snr-start -24 --snr-stop -25 --snr-step -1 `
  --output-dir weak_decoder/phase_line/_eval/phase_path_ablation_m24_m25_path_arbiter `
  --dp-energy-weight 0.35 `
  --dp-coherence-weight 0.40 `
  --dp-rank-weight 0.0 `
  --second-lambda 0.05 `
  --anchor-bonus 0.05
```

### Summary

The main SER numbers below come from `run_phase_path_ablation.py`.  With
adaptive Top-L, older fixed-width Top-L recall diagnostics are no longer a
single clean number, so the candidate-recall discussion is separated below.

| SNR | Old DP1 SER | Adaptive Top-L/relax | Gated + adaptive-soft | Final with path arbiter |
|---:|---:|---:|---:|---:|
| -24 dB | 0.283 | 0.242 | 0.234 | 0.229 |
| -25 dB | 0.410 | 0.366 | 0.353 | 0.345 |

The final `run_phase_path_ablation.py` output is:

```text
-24 dB: topL=0.865 v3=0.296 line=0.649 dp1=0.229 dp1h=0.231 dp2=0.234 dp2a=0.234
-25 dB: topL=0.749 v3=0.436 line=0.683 dp1=0.345 dp1h=0.358 dp2=0.405 dp2a=0.405
```

Candidate-coverage verification:

```text
-24 dB: topL=0.862 vitCand=0.852 dp1_ser=0.229 dp1_selectable_miss=39 dp1_gt_missing=45
-25 dB: topL=0.766 vitCand=0.766 dp1_ser=0.345 dp1_selectable_miss=46 dp1_gt_missing=83
```

## Current gap

This still does not beat the existing Savaux oversampled baseline on the same
dataset. The saved baseline table reports approximately:

| SNR | Savaux paper SER | New DP1 SER |
|---:|---:|---:|
| -24 dB | 0.117 | 0.229 |
| -25 dB | 0.249 | 0.345 |

So this iteration proves the Stage-1 phase/coherence proposal idea helps, but
it does not yet satisfy the final "stronger than Savaux oversampled" target.

## Next technical target

The remaining errors are now a mix of:

```text
candidate still missing
candidate present but DP chooses a smoother or locally stronger wrong bin
```

The next useful direction is not more blind single-point phase residual. The
evidence still supports a path-level approach:

1. keep the gated phase/coherence proposal rescue;
2. make the second stage aware of rescued-candidate provenance;
3. improve candidate proposal beyond energy/coherence Top-L, because many
   remaining selectable misses have GT candidate rank above 10;
4. avoid blindly strengthening sliding-window phase scoring, because the current
   selected wrong path is often more locally phase-consistent than GT.

### Top-L and adaptive probe

Increasing `top_l` does improve candidate recall, but it also introduces more
selectable distractors.  The adaptive row uses `top_l=24` by default and widens
only when hard-anchor count is at least 10:

| Setting | -24 effective recall | -24 DP1 SER | -25 effective recall | -25 DP1 SER |
|---|---:|---:|---:|---:|
| 24 | 0.883 | 0.255 | 0.784 | 0.366 |
| 28 | 0.899 | 0.252 | 0.795 | 0.369 |
| 32 | 0.904 | 0.249 | 0.805 | 0.374 |
| 40 | 0.925 | 0.242 | 0.834 | 0.377 |
| adaptive 24->40, anchors >= 10 | 0.909 | 0.244 | 0.790 | 0.366 |
| two-stage adaptive 24->28/40 | 0.865* | 0.242 | 0.745* | 0.366 |
| final gated + adaptive-soft | 0.865* | 0.234 | 0.749* | 0.353 |
| final + path arbiter | 0.865* | 0.229 | 0.749* | 0.345 |

So the base width remains 24, with adaptive widening only when the packet has
enough hard anchors to absorb the extra distractors.

`*` The starred recall numbers are from the ablation script's uncertain-symbol
candidate metric, not the full fixed-width Top-L recall.  They are kept here
only to show that the final SER improvement is coming from packet-level gating,
not a single fixed candidate width.

### Negative result: extra coherence candidate append

I also tested appending a small number of additional coherence-ranked candidates
to the Viterbi candidate list instead of replacing the energy Top-L tail.  It
is configurable through:

```text
extra_coherence_candidates
extra_coherence_min_gain
```

The best probes did not improve the final curve:

| Setting | -24 DP1 SER | -25 DP1 SER |
|---|---:|---:|
| final default | 0.229 | 0.345 |
| extra coherence 4, gain 0.05 | 0.244 | 0.364 |
| extra coherence 8, gain 0.05 | 0.244 | 0.364 |

So this remains off by default.  The extra candidates increase search space but
do not solve the path-selection ambiguity.

### Negative result: ungated hard anchors as phase-proposal anchors

A direct reading of the design suggests that hard anchors should also drive
phase-aware candidate proposal.  I tested exactly that with
`phase_proposal_use_hard_anchors=True`.  It was not stable as a global rule:

| Setting | -24 DP1 SER | -25 DP1 SER |
|---|---:|---:|
| final gated default | 0.229 | 0.345 |
| all hard anchors drive proposal | 0.239 | 0.371 |

The reason is that hard anchors are good enough to lock themselves and estimate
a coarse slope, but their raw center-FFT phase is not reliable enough to replace
energy candidates elsewhere.  The final implementation only enables hard-anchor
proposal under the medium-count / very-low-RMSE gate described above.

### Negative result: globally softening hard anchors

Wrong hard anchors tend to have low Top-1 margins and sit close to the hard
threshold, but globally relaxing every suspicious hard anchor to Top-2/Top-3
also introduces more distractors.

| Setting | -24 DP1 SER | -25 DP1 SER |
|---|---:|---:|
| final default | 0.229 | 0.345 |
| hard_soft2_m05 | 0.247 | 0.361 |
| hard_soft3_m05 | 0.242 | 0.361 |

The landed version therefore applies softening only when the packet has enough
hard anchors and their anchor phase line is not too noisy.

### Negative result: top40-only rescue candidates

Increasing fixed Top-L still adds true bins, but the extra true bins rarely
have a runtime score advantage over the already selected wrong bin:

| SNR | GT only in energy Top40 | hard-anchor phase says GT better | energy says GT better | coherence says GT better |
|---:|---:|---:|---:|---:|
| -24 dB | 6 | 1 / 3 predicted | 2 | 1 |
| -25 dB | 19 | 1 / 9 predicted | 1 | 5 |

This is why the final selector does not blindly append all Top40 bins to the
DP graph.  The extra candidates mostly expand the set of smooth-looking
distractors.

### Negative result: random second-stage weight search

A deterministic 35-trial search around energy/coherence/phase/sliding weights
did not find a better two-SNR tradeoff than the final default:

| Variant | -24 DP1 SER | -25 DP1 SER | worst SER |
|---|---:|---:|---:|
| final adaptive-soft before arbiter | 0.234 | 0.353 | 0.353 |
| best non-default (`coh16_gain005`) | 0.236 | 0.356 | 0.356 |
| best random trial | 0.236 | 0.364 | 0.364 |

### Negative result: selected-path phase rescue

I also tested whether the current selected DP path can be used as a local phase
predictor to recover bins missing from the compact Viterbi candidate graph.  A
diagnostic first ranked missing GT bins inside wider energy preselection sets:

| SNR | missing GT candidates | GT in energy Top128 | GT path-rank <= 8 in Top128 |
|---:|---:|---:|---:|
| -24 dB | 57 | 0.754 | 0.421 |
| -25 dB | 90 | 0.622 | 0.222 |

The signal exists, but it is weak and asymmetric.  A follow-up local refiner
that switched selected bins using path-phase + energy + coherence consistently
worsened SER:

| Setting | -24 SER | -25 SER |
|---|---:|---:|
| final default | 0.229 | 0.345 |
| best selected-path rescue probe | 0.239 | 0.371 |
| aggressive selected-path rescue probe | 0.278+ | 0.405+ |

This confirms the earlier sliding-window warning: once the selected path is
wrong, fitting local phase to that path tends to reinforce the wrong trajectory
instead of rescuing the missing true bin.

### Positive diagnostic: path arbiter

The path arbiter was found by comparing a small set of internal selector paths:
current DP1, second-order DP, proposal-off DP, stronger-phase DP, and the older
v3/local-coherence selector.  A simple runtime gate based on the current path's
mean absolute phase jumps captured most of the available non-GT gain:

| Gate | -24 SER | -25 SER |
|---|---:|---:|
| always current | 0.234 | 0.353 |
| v3 when first_abs_pi > 0.35 | 0.229 | 0.345 |
| v3 when second_abs_pi > 0.42 | 0.229 | 0.345 |
| small internal oracle over tested paths | 0.226 | 0.343 |

### Negative result: bin-dependent phase correction

I also tested whether the raw FFT peak phase needs a deterministic correction
of the form:

```text
corrected_phase = raw_phase - beta * raw_bin
```

Oracle fitting with GT can reduce phase-line RMSE, but non-GT versions are not
stable enough to land in the selector:

| SNR | Anchor-estimated beta rank1 | No-beta rank1 | Fixed beta result |
|---:|---:|---:|---|
| -24 dB | 0.517 | 0.529 | worsened rank1 / net rank balance |
| -25 dB | 0.383 | 0.383 | no rank1 gain, more worsened ranks |

The conclusion is that bin-dependent correction is an oracle-looking signal in
this setup: useful for explaining residuals, not yet safe as a runtime selector
feature.

### Negative result: blind sliding-window strengthening

The selectable-miss diagnostic in `_eval/selectable_miss_diagnostic/` explains
why a stronger generic sliding window is risky:

| SNR | selectable misses | mean GT rank | path-line says GT better | hard-anchor line says GT better |
|---:|---:|---:|---:|---:|
| -24 dB | 39 | 12.3 | 0.436 | 0.273 |
| -25 dB | 46 | 15.1 | 0.391 | 0.500 |

For most selectable misses, the locally fitted path phase line prefers the
already-selected wrong bin.  That is why the landed change relaxes first-order
phase only under a hard-anchor-count gate instead of simply increasing local
phase weight everywhere.
