# Savaux+Codec pruning diagnostics, 2026-06-18

## Purpose

The current Savaux+codec engineering path improves AWGN CRC thresholds over the pure Savaux paper baseline, but it still does not consistently reach the requested 2-3 dB margin over Savaux. This note records the pruning diagnostics and the experimental search paths tried on 2026-06-18.

## Diagnostic Tool

Script:

`scripts/experiments/structured_paths/diagnose_codec_gt_path.py`

It is evaluation-only and uses synthetic/noisy captures with clean header-first symbols as ground truth. It does not feed GT into the decoder. It reports whether the GT path survives:

1. Savaux per-symbol Top-K evidence,
2. row-wise Hamming nibble candidates,
3. per-block candidates,
4. global block beam,
5. final CRC candidate window.

## Key Findings

Dataset/SNR used for the focused diagnostic:

`0_0_0_10_14_8`, `SNR=-23 dB`, packets `5` and `6`.

Default decoder:

- GT raw bins were present in Savaux Top-K for all payload symbols.
- Correct row nibbles were present for all rows.
- The GT path was pruned later, at block/global candidate stages.

Packet 5:

- Missing block value ranks: `5:1,17,1,1,3`.
- This means the correct block is blocked by one outlier symbol whose demod-symbol value rank is 17, while the other symbols in that block are strong.
- Plain Top-M symbol seeding with `M=6` or `M=8` cannot cover this packet.

Packet 6:

- All GT blocks can be present with symbol seeding.
- Exact GT blocks reconstruct the correct payload and CRC:
  - `exact_blocks_available=1`
  - `exact_block_payload_matches_gt=1`
  - `exact_block_crc_valid=1`
  - `exact_block_rank_cost≈11.394`
- However score/rank beams still failed to include that exact full-block combination in the final candidate set under practical state limits.

## Experimental Paths Added

All of these are default-off knobs in `TwoStageWeakConfig`:

- `block_symbol_seed_top_m`
- `block_symbol_seed_deep_top_l`
- `block_symbol_seed_max_deep_positions`
- `block_symbol_seed_quota`
- `block_symbol_seed_max_combinations`
- `global_rank_diverse_top_r`
- `global_rank_diverse_beam_width`
- `global_rank_cost_max`
- `global_rank_cost_state_limit`

These are exposed in:

- `scripts/experiments/structured_paths/run_savaux_codec_sweep.py`
- `scripts/experiments/structured_paths/run_real_iq_crc_probe.py`
- `scripts/experiments/structured_paths/run_real_iq_batch_crc_probe.py`
- `scripts/experiments/structured_paths/diagnose_codec_gt_path.py`

## Negative/Unstable Result

Small AWGN regression:

Output directories:

- `data/savaux_codec_seed6_probe_m23_m24_len8`
- `data/savaux_codec_seed6_rankcost_probe_m23_m24_len8`

Command shape:

- dataset: `0_0_0_10_14_8`
- SNR: `-23, -24 dB`
- `block_symbol_seed_top_m=6`
- `block_symbol_seed_quota=64`
- optional rank-cost search: `global_rank_cost_max=12.5`, `global_rank_cost_state_limit=20000`

Result:

| SNR | Savaux paper SER | Savaux+codec SER | Savaux CRC | codec CRC | fix | break |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -23 | 0.051 | 0.051 | 0.500 | 0.500 | 0.00 | 0.00 |
| -24 | 0.140 | 0.149 | 0.300 | 0.400 | 0.40 | 0.30 |

Interpretation:

The new search paths can introduce additional CRC-valid candidates, but in this small regression they also introduce false or harmful selections. This is not stable enough to make default. The conservative default should stay with the previously validated `earliest_rank` codec policy and keep the new seed/rank-cost paths as diagnostic/experimental switches only.

## Reliability Calibration: CRC Beam Rank

After the negative seed-search result, the accepted CRC rescue rank was tightened from `256` to `64`.

Probe outputs:

- `data/savaux_codec_seed6_rank64_probe_m23_m24_len8`
- `data/savaux_codec_seed6_rank128_probe_m23_m24_len8`
- `data/savaux_codec_default_rank64_probe_m23_m24_len8`
- `data/savaux_codec_default_rank64_all_m22_m26`
- `data/savaux_codec_default_rank128_all_m22_m26`

Focused `0_0_0_10_14_8`, SNR `-23,-24 dB`:

- Seed search with rank `256` produced a harmful CRC candidate at `-24 dB`:
  - `beam_rank=193`
  - `selected_evidence_margin≈0.646`
  - `symbol_fix_count=0`
  - `symbol_break_count=3`
- Seed search with rank `64` or `128` rejected that candidate and matched the paper baseline at `-24 dB`.
- Default codec with rank `64` still produced useful rescues:
  - `-23 dB`: Savaux CRC `0.5`, codec CRC `0.7`
  - `-24 dB`: Savaux CRC `0.3`, codec CRC `0.5`
  - no observed symbol breaks in the probe.

Three-dataset AWGN check, SNR `-22..-26 dB`:

`data/savaux_codec_default_rank64_all_m22_m26`

Mean CRC rates:

| SNR | Savaux paper | codec rank<=64 |
| ---: | ---: | ---: |
| -22 | 0.650 | 0.952 |
| -23 | 0.370 | 0.792 |
| -24 | 0.256 | 0.539 |
| -25 | 0.033 | 0.350 |
| -26 | 0.000 | 0.097 |

Mean SER:

| SNR | Savaux paper | codec rank<=64 |
| ---: | ---: | ---: |
| -22 | 0.063 | 0.045 |
| -23 | 0.097 | 0.072 |
| -24 | 0.158 | 0.139 |
| -25 | 0.277 | 0.244 |
| -26 | 0.410 | 0.400 |

The rank `64` curve matches the previous rank `256` curve in this SNR region while rejecting the observed seed-search false positive. Therefore rank `64` is now the safer default for codec CRC rescue.

## High-Rank Evidence Gate

The plain rank `64` cutoff is safe, but it can reject rare true rescues. One observed full-sweep rescue in `data/savaux_codec_policy_earliest_m12_m27` had:

- dataset `0_0_0_10_14_8`
- SNR `-23 dB`
- packet `7`
- `beam_rank=141`
- `selected_evidence_margin≈6.935`
- `symbol_fix_count=6`
- `symbol_break_count=0`

The harmful seed-search false positive was much less convincing:

- dataset `0_0_0_10_14_8`
- SNR `-24 dB`
- packet `6`
- `beam_rank=193`
- `selected_evidence_margin≈0.646`
- `symbol_fix_count=0`
- `symbol_break_count=3`

A two-tier CRC gate was added:

- accept normal CRC candidates up to `crc_candidate_max_beam_rank=64`;
- allow later CRC candidates up to `crc_candidate_high_rank_max_beam_rank=256` only if their evidence margin over the hard-argmax fallback is at least `crc_candidate_high_rank_min_evidence_margin=4.0`.

Implemented in:

- `weak_decoder/two_stage_weak_decoder.py`
- `scripts/experiments/structured_paths/run_savaux_codec_sweep.py`
- `scripts/experiments/structured_paths/run_real_iq_crc_probe.py`
- `scripts/experiments/structured_paths/run_real_iq_batch_crc_probe.py`
- `scripts/experiments/structured_paths/diagnose_codec_gt_path.py`

Small focused probes:

- `data/savaux_codec_highrank_gate_probe_m23_m24_len8`
- `data/savaux_codec_seed6_highrank_gate_probe_m23_m24_len8`

On `0_0_0_10_14_8`, SNR `-23,-24 dB`:

| variant | -23 CRC | -24 CRC | observed symbol break |
| --- | ---: | ---: | ---: |
| rank64 default | 0.700 | 0.500 | 0 |
| high-rank evidence gate | 0.800 | 0.500 | 0 |
| seed6 + high-rank gate | 0.500 | 0.300 | 0 |

The high-rank gate recovered the rank-141 true rescue and still rejected the seed-search rank-193 false positive. Seed search remained unstable/unhelpful because it changed the candidate pool and pushed good candidates out of the final window.

Three-dataset AWGN check:

`data/savaux_codec_highrank_gate_all_m22_m26`

Mean CRC rates:

| SNR | Savaux paper | rank64 codec | high-rank gate codec |
| ---: | ---: | ---: | ---: |
| -22 | 0.650 | 0.952 | 0.952 |
| -23 | 0.370 | 0.792 | 0.825 |
| -24 | 0.256 | 0.539 | 0.539 |
| -25 | 0.033 | 0.350 | 0.350 |
| -26 | 0.000 | 0.097 | 0.097 |

Mean SER:

| SNR | Savaux paper | rank64 codec | high-rank gate codec |
| ---: | ---: | ---: | ---: |
| -22 | 0.063 | 0.045 | 0.045 |
| -23 | 0.097 | 0.072 | 0.066 |
| -24 | 0.158 | 0.139 | 0.139 |
| -25 | 0.277 | 0.244 | 0.244 |
| -26 | 0.410 | 0.400 | 0.400 |

No symbol-break rows were observed in this three-dataset run. The mean SER threshold gain over Savaux improved slightly from about `0.370 dB` to `0.416 dB`; the mean CRC50 gain over Savaux stayed about `1.67 dB`.

Rank-cost final states were also probed:

- `data/savaux_codec_rankcost_highrank_gate_probe_m23_m24_len8`
- `data/savaux_codec_seed6_rankcost_highrank_gate_probe_m23_m24_len8`

Rank-cost without seed matched the high-rank gate result and did not add extra net gain. Rank-cost with seed still fell back to the Savaux hard result in the focused probe. Therefore rank-cost and seed search should remain default-off diagnostics.

## CRC-State Search Negative Result

A default-off CRC-state backend was added as another diagnostic path. It tries to merge partial paths by byte/CRC state while block candidates are appended, so it can search CRC-valid combinations without relying only on a global score beam.

Implemented switches:

- `crc_state_search_block_top_r`
- `crc_state_search_state_limit`
- `crc_state_search_keep_per_key`
- `crc_state_search_max_candidates`
- `crc_state_search_min_evidence_margin`

The first implementation missed the explicit-header tail nibbles when initializing the CRC/byte state. That was fixed so the CRC-state stream starts from the header tail before payload-block nibbles are appended.

Focused diagnostic:

`data/codec_gt_path_diag_crcstate_top32_p6_after_tailfix`

Dataset/SNR/packet:

- `0_0_0_10_14_8`
- `SNR=-23 dB`
- packet `6`

Result:

- exact GT blocks are available and reconstruct a CRC-valid payload:
  - `exact_blocks_available=1`
  - `exact_block_crc_valid=1`
  - `exact_block_rank_cost≈11.394`
- CRC-state search produced `22` CRC-valid candidate states.
- None matched the GT payload.
- The evidence gate accepted `0` CRC-state candidates.
- Selected output stayed the hard-argmax fallback.

Interpretation:

CRC-state search can find CRC-valid candidates, but in this focused weak-packet case they are mostly false CRC states under the current block-candidate ordering. It is also slow: the small two-SNR sweep with this backend exceeded a 15-minute timeout. This path should stay default-off and is not a productive direction unless paired with a much stronger independent reliability score.

## Current Practical Conclusion

For real IQ:

- Pure Savaux paper OS combining is currently stronger than the existing current-selected path in the SF11/TP2 transition region.
- Savaux+codec matches Savaux on real IQ because successful real packets are mostly already Savaux-hard CRC valid.

For AWGN:

- The validated Savaux+codec path improves CRC threshold over pure Savaux by about 1.6-1.7 dB. The high-rank evidence gate is a small safe improvement, but the new pruning-expansion experiments do not yet push this to a stable 2-3 dB.

Next promising work should focus less on widening combinatorial search and more on better likelihood calibration or an independent reliability model that can reject false CRC candidates before selection.

## Stop Point For Structured Non-Uniform Paths

The original structured/non-uniform oversampling idea has now been tried in several constrained forms:

- fixed Savaux branches as the paper baseline;
- modular linear paths;
- piecewise-constant paths;
- periodic paths;
- smooth/adaptive phase-consistency paths;
- fractional timing paths;
- packet-shared timing paths;
- block seed and rank-cost searches driven by non-hard symbol alternatives;
- CRC-state search over block candidates.

The physically meaningful path versions are valuable as diagnostics and as paper discussion material, but the measured decoder gain is not stable enough to replace the Savaux OSR baseline. The reliable engineering gain currently comes from using Savaux's oversampled evidence as soft likelihoods and applying local LoRa PHY constraints/CRC carefully. The best validated AWGN improvement remains roughly `1.6-1.7 dB` in CRC threshold over pure Savaux, plus a small safe gain from the high-rank evidence gate. The requested stable `2-3 dB` advantage over Savaux has not been demonstrated.
