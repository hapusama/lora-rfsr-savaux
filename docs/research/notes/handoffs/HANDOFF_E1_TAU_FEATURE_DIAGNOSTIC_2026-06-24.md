# Handoff: Phase-Line E1 Tau Feature Diagnostic

Date: 2026-06-24

This handoff is for continuing work under:

```text
gr-lora_sdr\weakPacket_decoding copy
```

The immediate user request before handoff was to run an E1-symbol diagnostic comparing selected vs GT:

```text
selected_tau_hat - gt_tau_hat
selected_tau_residual - gt_tau_residual
selected_tau_gain - gt_tau_gain
selected_corrected_coherence - gt_corrected_coherence
selected_energy - gt_energy
```

The previous turn was interrupted before the diagnostic script was written. No long experiment was intentionally left running.

## Current Best System

The current "current system" is in:

```text
weak_decoder\phase_line
```

The best path being used is:

```text
Savaux stage-1 candidate/evidence generation
-> Phase-Line first-order Viterbi selector
```

Key config entry:

```text
weak_decoder\phase_line\savaux_stage1.py
default_savaux_phase_path_config(top_l=16)
```

Important defaults in that config at last inspection:

```text
top_l = 16
energy_weight = 0.65
coherence_weight = 0.15
phase_local_weight = 0.15
rank_weight = 0.0
first_order_weight = 0.08
second_order_weight = 0.0
hard_anchor_top_k = 1
hard_anchor_soft_top_k = 4
high_confidence_top_k = 4
anchor_phase_bias_weight = 0.16
sliding_window_refine_enabled = False
path_arbiter_enabled = False
phase_proposal_enabled = False
```

Noise-peak penalty and reliable-top1 protection fields exist in `configs.py`, but they are not enabled by `default_savaux_phase_path_config()` unless explicitly changed.

## Key Files

Core implementation:

```text
weak_decoder\phase_line\savaux_stage1.py
weak_decoder\phase_line\selector.py
weak_decoder\phase_line\configs.py
```

Current evaluation and diagnostic scripts:

```text
weak_decoder\phase_line\_eval\diagnose_savaux_stage1_selector.py
weak_decoder\phase_line\_eval\sweep_e1_local_score_weights.py
weak_decoder\phase_line\_eval\plot_current_stage2_candidate_coverage.py
weak_decoder\phase_line\_eval\plot_gt_amplitude_path.py
weak_decoder\phase_line\_eval\plot_sto_phase_correction.py
```

Useful helpers to import:

```python
from diagnose_savaux_stage1_selector import (
    _payload_reference_power,
    _payload_starts_and_abs,
)

from run_phase_line_threshold_sweep import _dataset_paths, _load_metadata
from run_two_stage_weak_decoder import load_packets
```

The `_eval` scripts already set:

```python
THIS = Path(__file__).resolve()
WEAK_ROOT = THIS.parents[3]
EXPERIMENT_DIR = WEAK_ROOT / "scripts" / "experiments"
PHASE_EXPERIMENT_DIR = EXPERIMENT_DIR / "phase_line"
```

Use the same path setup for the new diagnostic script.

## Known Empirical State

Soft anchor was already useful. Earlier results:

```text
E3: 6.97% -> 0.66%
SER: 0.1468 -> 0.1281
```

So the current bottleneck moved from "GT had no chance because anchor locked it out" to mainly:

```text
E1: GT entered candidates, but local_score chose selected instead.
```

The next experiment should focus on whether E1 is caused by high-energy noise peaks being trusted too much.

## E1 Definition to Keep Consistent

Use the taxonomy already implemented in:

```text
weak_decoder\phase_line\_eval\sweep_e1_local_score_weights.py
```

Relevant logic:

```python
if gt_cand is None:
    ...
if gt_cand is not None and selected_cand is not None:
    if float(gt_cand.local_score) < float(selected_cand.local_score) - 1e-12:
        return "E1_gt_in_candidates_local_score_loss"
    return "E2_gt_in_candidates_transition_loss"
```

For this new diagnostic, only include symbols where:

```text
selected != GT
GT is in actual Viterbi candidates
selected_cand.local_score > gt_cand.local_score
```

This is stricter than "GT in stage1 Top-40". It is the actual E1 definition.

## Candidate Preparation Gotcha

Do not just use `stage1.top_bins`.

The actual Viterbi candidates are affected by:

```text
high confidence top-k
hard anchor / soft anchor
adaptive top_l logic
anchor phase bias
max_energy_drop_db
```

The selector sequence to mirror is in:

```text
weak_decoder\phase_line\selector.py
select_phase_viterbi_path()
```

Private helper imports are acceptable for diagnostics:

```python
from weak_decoder.phase_line.selector import (
    _anchor_phase_line_rmse_pi,
    _apply_anchor_phase_bias,
    _augment_evidences_with_phase_proposals,
    _is_hard_anchor,
    _is_high_confidence,
    _viterbi_candidates,
)
```

Because `default_savaux_phase_path_config()` currently has `phase_proposal_enabled=False`, proposal augmentation should be a no-op, but keeping the call makes the script robust.

You can also copy/adapt `_effective_selector_state()` from:

```text
weak_decoder\phase_line\_eval\sweep_e1_local_score_weights.py
```

Just remember that for exact current behavior, anchor phase bias must be applied before comparing `local_score`.

## Stage-1 Evidence Setup

Use current Savaux stage1:

```python
stage1 = build_savaux_stage1_packet_evidence(
    samples=noisy,
    start_samples=starts,
    sf=int(packet["sf"]),
    os_factor=int(packet["os_factor"]),
    abs_indices=abs_indices,
    cfo_int=int(packet["cfo_int"]),
    cfo_frac=float(packet["cfo_frac"]),
    header_start_sample=int(packet["header_start_sample"]),
    residual_sto_chips=_payload_residual_sto(packet),  # optional, correction is off by default
    config=SavauxStage1Config(
        cfo_correction_mode="continuous",
        top_k=24,
        branch_agreement_power=0.0,
        residual_sto_phase_correction=False,
    ),
)
```

Use payload-window signal power when adding AWGN. Do not fall back to full-IQ mean unless necessary:

```python
signal_power, _ = _payload_reference_power(samples, packets)
```

This was important in the GT amplitude plotting work; full-IQ power gave misleading low-SNR behavior.

## Tau Feature Definitions for the New Experiment

For each E1 symbol, compute these for both selected and GT bins.

Suggested definitions:

```text
tau_hat:
    residual STO value in [-0.5, 0.5) that maximizes corrected coherent amplitude

tau_residual:
    wrap_half(tau_hat - expected_tau)
    where expected_tau = symbol["sfo_cum_before"], fallback symbol["sto_frac"]

tau_gain_db:
    20 * log10(abs(corrected_value_at_tau_hat) / abs(no_correction_value_at_tau0))

corrected_coherence:
    after residual-STO phase-jump correction and Savaux branch alignment:
    abs(sum(aligned_branch_values)) / (sum(abs(aligned_branch_values)) + eps)

energy:
    stage1.evidence_power[raw_bin]
    Also output energy_score if useful.
```

Important: the tau correction being diagnosed is the chirp-internal wrap phase correction, not the normal Savaux branch alignment. The two corrections are separate:

```text
dechirp
-> branch q extraction
-> candidate-k wrap split
-> residual STO phase-jump correction
-> Savaux branch alignment exp(-j2pi qk/(NR))
-> coherent sum
```

## How to Compute Corrected Coherence

`savaux_stage1.py` already has:

```python
_prepare_dechirped_oversampled_symbol()
_sto_phase_corrected_bin_value()
```

But `_sto_phase_corrected_bin_value()` returns only the combined complex value. For the diagnostic, duplicate its logic locally and also keep per-branch aligned values.

Pseudo-code:

```python
def corrected_value_and_coherence(dechirped, raw_bin, os_factor, tau, phase_sign):
    full = np.asarray(dechirped, dtype=np.complex64)
    n_bins = full.size // os_factor
    k = raw_bin % n_bins
    tau = wrap_half(tau)
    residual_phase = exp(1j * sign * 2*pi*tau)
    p = np.arange(n_bins)
    kernel = exp(-2j*pi*k*p/n_bins)

    aligned = []
    for q in range(os_factor):
        branch = full[q::os_factor]
        values = branch * kernel

        # Paper/Savaux wrap correction already used by current implementation.
        if k != 0:
            values[p >= n_bins - k] *= exp(2j*pi*q/os_factor)

        # Residual STO chirp-internal phase jump.
        cut = ceil(n_bins - k + tau - q/os_factor)
        if cut < n_bins:
            values[cut:] *= residual_phase

        branch_value = sum(values) / sqrt(n_bins)
        branch_weight = exp(-2j*pi*q*k/(n_bins * os_factor))
        aligned.append(branch_weight * branch_value)

    combined = sum(aligned)
    coherence = abs(combined) / (sum(abs(v) for v in aligned) + 1e-30)
    return combined, coherence
```

Use the same `paper_start_sample` convention as `build_savaux_symbol_evidence()`:

```python
origin_shift = os_factor // 2
paper_start = start_sample + origin_shift
paper_header_start = header_start_sample + origin_shift
dechirped = _prepare_dechirped_oversampled_symbol(
    samples=noisy,
    start_sample=paper_start,
    sf=sf,
    os_factor=os_factor,
    cfo_int=cfo_int,
    cfo_frac=cfo_frac,
    header_start_sample=paper_header_start,
    cfo_correction_mode="continuous",
)
```

## Suggested New Script

Create:

```text
weak_decoder\phase_line\_eval\diagnose_e1_tau_features.py
```

Suggested CLI:

```text
--dataset 0_0_0_10_14_16
--snrs -26 -25.5 -25 -24.5 -24
--seeds 42 43 44
--top-l 16
--stage1-top-k 24
--tau-grid-step 0.03125
--output-dir data\baseline_comparison\e1_tau_feature_diagnostics
```

The script should write:

```text
e1_tau_feature_rows.csv
e1_tau_feature_summary.csv
manifest.json
```

Suggested per-symbol row columns:

```text
dataset
target_snr_db
noise_seed
packet_index
payload_symbol_index
gt_bin
selected_bin
selected_local_score
gt_local_score
selected_energy
gt_energy
selected_energy_score
gt_energy_score
selected_minus_gt_energy_db
selected_tau_hat
gt_tau_hat
selected_minus_gt_tau_hat
expected_tau
selected_tau_residual
gt_tau_residual
selected_minus_gt_abs_tau_residual
selected_tau_gain_db
gt_tau_gain_db
selected_minus_gt_tau_gain_db
selected_corrected_coherence
gt_corrected_coherence
selected_minus_gt_corrected_coherence
selected_rank
gt_rank
top1_bin
top1_margin_db
top1_peak_to_median_db
top1_coherence_score
```

Suggested summary columns per SNR:

```text
e1_count
selected_energy_higher_rate
mean_selected_minus_gt_energy_db
gt_tau_residual_smaller_rate
mean_selected_abs_tau_residual
mean_gt_abs_tau_residual
gt_tau_gain_higher_rate
mean_selected_minus_gt_tau_gain_db
gt_corrected_coherence_higher_rate
mean_selected_minus_gt_corrected_coherence
```

## Commands to Run

Compile:

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe -m py_compile "D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding copy\weak_decoder\phase_line\_eval\diagnose_e1_tau_features.py"
```

Run low-SNR diagnostic:

```powershell
D:\mysoft2\miniconda3\envs\gr-lora\python.exe "D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding copy\weak_decoder\phase_line\_eval\diagnose_e1_tau_features.py" `
  --dataset 0_0_0_10_14_16 `
  --snrs -26 -25.5 -25 -24.5 -24 `
  --seeds 42 43 44 `
  --top-l 16 `
  --stage1-top-k 24 `
  --output-dir "D:\Desktop\proj\gr-lora_sdr\weakPacket_decoding copy\data\baseline_comparison\e1_tau_feature_diagnostics"
```

## How to Interpret the Result

If E1 rows show:

```text
selected_energy >> gt_energy
but gt_corrected_coherence > selected_corrected_coherence
or gt_tau_gain > selected_tau_gain
or abs(gt_tau_residual) < abs(selected_tau_residual)
```

then E1 is likely high-energy noise peaks. The right next move is not simply increasing phase weight; it is energy confidence calibration or suspicious selected-bin penalty.

Good feature candidates:

```text
corrected_coherence
tau_gain_db
abs(tau_residual)
energy high but corrected_coherence low
energy high but tau_residual far from expected_tau
```

If selected beats GT on all tau/coherence features too, then this tau diagnostic probably will not help local_score directly.

## Residual STO Phase Correction Status

An optional residual STO chirp-wrap phase correction exists in:

```text
weak_decoder\phase_line\savaux_stage1.py
```

Config fields:

```text
residual_sto_phase_correction
residual_sto_phase_sign
residual_sto_preselect_factor
residual_sto_update_power
```

It is currently off in the default path.

Earlier low-SNR test summary:

```text
phase-only plus correction improved some cases:
  example: packet10 at -25 dB seed 42, SER 0.2857 -> 0.1714

but averaged across the tested low-SNR set it was slightly worse:
  mean SER delta about +0.00052
```

So do not globally enable it yet. Use it as a diagnostic feature first.

## GT Amplitude Plot Status

Script:

```text
weak_decoder\phase_line\_eval\plot_gt_amplitude_path.py
```

It already plots GT-bin amplitude path using current Savaux stage1 evidence.

Important answer already given to the user:

```text
These plots use Savaux stage-1 coherent multi-offset/branch combination.
They do not use the optional residual STO chirp-wrap phase correction.
branch_agreement_power = 0.0, so evidence power is pure coherent combined power.
```

All-packet low-SNR amplitude summary from the corrected payload-power run:

```text
SNR -26.0: SER 0.3481, amp_std 2.303, |diff| 2.530, |2nd diff| 4.507, r2 0.017, rank1 0.621, rank4 0.766
SNR -25.5: SER 0.2753, amp_std 2.165, |diff| 2.388, |2nd diff| 4.251, rank1 0.683, rank4 0.839
SNR -25.0: SER 0.1948, amp_std 2.035, |diff| 2.254, |2nd diff| 4.008, rank1 0.761, rank4 0.875
SNR -24.5: SER 0.1403, amp_std 1.912, |diff| 2.128, |2nd diff| 3.779, rank1 0.808, rank4 0.917
SNR -24.0: SER 0.0909, amp_std 1.798, |diff| 2.009, |2nd diff| 3.563, rank1 0.857, rank4 0.948
```

Conclusion: amplitude after coherent stage1 is useful but not smooth enough to be a strong Viterbi transition by itself. It is better as a weak reliability feature or noise-peak penalty input.

## First Offset Coherent Combine Sanity Check

A quick comparison showed Savaux coherent combine is much better than single center FFT:

```text
SNR -26.0:
  center rank1/rank4/rank16 = 0.047 / 0.143 / 0.278
  savaux rank1/rank4/rank16 = 0.621 / 0.766 / 0.878

SNR -25.0:
  center rank1/rank4/rank16 = 0.099 / 0.213 / 0.382
  savaux rank1/rank4/rank16 = 0.761 / 0.875 / 0.940

SNR -24.0:
  center rank1/rank4/rank16 = 0.166 / 0.306 / 0.494
  savaux rank1/rank4/rank16 = 0.857 / 0.948 / 0.979
```

So the current first-stage coherent combine is definitely doing real work.

## Data Cleanup State

The user asked to delete reproducible ablation clutter under:

```text
data
```

A previous cleanup kept only the main/reproducible data directories, including:

```text
ablation_current_default
ablation_offset_coherence_summary
baseline_comparison
loratrimmer_baseline
phase_opportunity_space
symbol_phase_two_stage
two_stage_weak_decoder
weak_preamble_detections
weak_sync_chain
README.md
```

Do not restore deleted experiment-output clutter unless the user asks.

## Hi2LoRa Takeaway

The user asked whether Hi2LoRa is inspiring for this work. The useful takeaway was:

```text
Hi2LoRa treats hardware/channel imperfections as features, not only as nuisance parameters.
For this decoder, residual CFO/STO/SFO related quantities can be used as candidate reliability features.
```

This directly motivates the current E1 tau feature diagnostic.

## Recommended Next Step

Implement only the E1 tau diagnostic first. Do not start another weight sweep yet.

After the summary CSV exists, decide:

```text
If GT has better tau/coherence diagnostics in many E1 rows:
    add energy confidence calibration or suspicious selected-bin penalty.

If GT does not look better:
    tau features are not the right local_score fix.
```
