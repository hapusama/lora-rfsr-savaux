# Two-stage weak packet decoder

This implementation starts after weak sync and header-first have already
succeeded. It is PHY-only and does not use a payload template, counter pattern,
or cross-packet application prior.

## Algorithm

1. Keep the full complex FFT evidence `Z[k,b]` for each payload symbol.
   The runner can use the original center-sample FFT evidence, the newer
   `multi-offset` evidence, or `phase-gated` evidence.
2. Build raw-bin and demod-symbol likelihoods, plus Top-K evaluation metrics.
3. Work at the payload interleaver-block level instead of mapping one symbol
   error directly to one byte error.
4. Deinterleave symbol likelihoods into bit soft evidence for each Hamming
   codeword row.
5. Enumerate the 16 legal nibble/codeword candidates per codeword and keep a
   configurable candidate list.
6. Beam-search block candidates, then re-encode each payload candidate through
   the local LoRa PHY codec to project it back to theoretical payload bins.
7. Score projected bins against the FFT-bin likelihood. In `phase-gated` mode,
   that likelihood is computed before the codec search from:

   ```text
   oversampled/multi-offset FFT evidence -> Top-M energy screening
     -> one-sided phase-consistency bonus inside the screened bins
   ```

   The phase trend is estimated within the current packet only, normally from
   early high-energy payload anchors. It does not use payload bytes, payload
   templates, counters, or any other packet.
8. Re-encode each surviving payload candidate and optionally score its
   packet-local phase trajectory with
   `phase_guided_demod._score_payload_symbol_prior_candidate()`. This is the
   stronger candidate-level phase-aware score:

   ```text
   score =
     0.25 * mean(0.5 + 0.5*cos(phase_residual))
   + 0.50 * exp(-(line_rmse_pi / 0.30)^2)
   + 0.20 * mean(normalized_bin_power)
   + 0.05 * mean(profile_score)
   ```

   It is still single-packet and blind: the candidate symbols come from the
   local codec beam, not from known payload bytes or retransmissions.
9. Prefer CRC-valid beam candidates. If no beam candidate passes CRC, the
   default behavior is to abstain from destructive repair and return the
   active-evidence hard-decision path with
   `selected_source=argmax_fallback`. In `center` mode this is the traditional
   FFT argmax path; in `phase-gated` mode it is the phase-gated top-1 path.

## Main Files

- `weak_decoder/two_stage_weak_decoder.py`
- `scripts/experiments/run_two_stage_weak_decoder.py`
- `scripts/experiments/evaluate_phase_bin_metric.py`
- `scripts/experiments/evaluate_codec_bin_metric.py`

## Example Commands

Clean sanity:

```powershell
python "gr-lora_sdr/weakPacket_decoding copy/scripts/experiments/run_two_stage_weak_decoder.py" `
  -i "gr-lora_sdr/data/USRP_IQ/0_0_0_10_14_16.bin" `
  -s "gr-lora_sdr/weakPacket_decoding copy/data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols_netidvalid.csv" `
  -o "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/clean_all_results.csv" `
  --summary-json "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/clean_all_summary.json" `
  --phase-weight 0.0
```

Low-SNR payload-only evaluation:

```powershell
python "gr-lora_sdr/weakPacket_decoding copy/scripts/experiments/run_two_stage_weak_decoder.py" `
  -i "gr-lora_sdr/weakPacket_decoding copy/data/low_snr_gt_bin/0_0_0_10_14_16/0_0_0_10_14_16_snr_m15dB.bin" `
  -s "gr-lora_sdr/weakPacket_decoding copy/data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols_netidvalid.csv" `
  -o "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/snr_m15_all_fallback_results.csv" `
  --summary-json "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/snr_m15_all_fallback_summary.json" `
  --fft-evidence-mode multi-offset `
  --phase-weight 0.0
```

`multi-offset` fuses all oversampling phases for each payload symbol:

```text
score[b] = sum_offset |FFT_offset[b]|^2 / max_b |FFT_offset[b]|^2
```

This is the current best bin-candidate metric. It does not assume the true bin
is near the center-offset argmax; it uses the fact that a real LoRa tone remains
stable across the oversampled chip phases while many noise peaks do not.

`phase-gated` keeps that oversampled FFT screen and then applies the
phase-aware peak score from `weak_decoder/candidate_pruning.py`:

```text
S[b] = log1p(E[b] / noise_floor)
     + lambda * q_line * amp_gate[b] * phase_bonus[b]
```

where `E[b]` is usually the multi-offset evidence, `phase_bonus[b]` is a
one-sided bonus from the residual to the packet-local phase line, and `q_line`
turns the bonus down when the line quality is poor. This is intentionally not a
phase-only decoder; phase consistency matters most after oversampled FFT has
already reduced the candidate space.

Example phase-gated command:

```powershell
python "gr-lora_sdr/weakPacket_decoding copy/scripts/experiments/run_two_stage_weak_decoder.py" `
  -i "gr-lora_sdr/weakPacket_decoding copy/data/low_snr_gt_bin/0_0_0_10_14_16/0_0_0_10_14_16_snr_m20dB.bin" `
  -s "gr-lora_sdr/weakPacket_decoding copy/data/weak_sync_chain/header_first/0_0_0_10_14_16_header_first_symbols.csv" `
  -o "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/phase_gated_snr_m20_all_results.csv" `
  --summary-json "gr-lora_sdr/weakPacket_decoding copy/data/two_stage_weak_decoder/phase_gated_snr_m20_all_summary.json" `
  --fft-evidence-mode phase-gated `
  --phase-weight 0.0 `
  --trajectory-score-weight 2.0 `
  --phase-gated-weight 0.2 `
  --phase-gated-width-pi 0.5
```

## Current Validation

Parameters: `phase_weight=0.0`, `nibble_candidates=4`,
`row_beam_width=256`, `block_candidate_limit=64`,
`global_beam_width=64`.

Center FFT evidence:

| Dataset | Packets | Center argmax symbol SER | Two-stage symbol SER | CRC-valid rate | Top-K recall | Fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 11 | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.000 |
| -10 dB | 11 | 0.0000 | 0.0000 | 1.000 | 1.000 | 0.000 |
| -15 dB | 11 | 0.0026 | 0.0000 | 1.000 | 1.000 | 0.000 |
| -20 dB | 11 | 0.3584 | 0.3584 | 0.000 | 0.927 | 1.000 |

Multi-offset FFT evidence:

| Dataset | Packets | Multi-offset argmax symbol SER | Two-stage symbol SER | CRC-valid rate | Top-K recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 11 | 0.0000 | 0.0000 | 1.000 | 1.000 |
| -15 dB | 11 | 0.0000 | 0.0000 | 1.000 | 1.000 |
| -20 dB | 11 | 0.0156 | 0.0000 | 1.000 | 1.000 |
| len8 -23 dB | 10 | 0.3343 | 0.3171 | 0.200 | 0.909 |
| len32 -23 dB | 7 | 0.3429 | 0.3224 | 0.143 | 0.865 |

Phase-gated evidence, using multi-offset energy screening, early-payload phase
trend, and candidate trajectory scoring (`trajectory_score_weight=2.0`,
`phase_gated_weight=0.2`, `phase_gated_width_pi=0.5`):

| Dataset | Packets | Center argmax symbol SER | Phase-gated argmax symbol SER | Two-stage symbol SER | CRC-valid rate | Top-K recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| -20 dB | 11 | 0.3584 | 0.0234 | 0.0000 | 1.000 | 1.000 |
| 33-byte -23 dB | 11 | 0.7325 | 0.3117 | 0.3117 | 0.000 | 0.945 |
| len8 -23 dB | 10 | 0.7943 | 0.3286 | 0.3029 | 0.300 | 0.909 |
| len32 -23 dB | 7 | 0.7755 | 0.3388 | 0.3388 | 0.000 | 0.861 |

On the 33-byte -20 dB set, bin-candidate recall changes as follows:

| Bin metric | Top-1 recall | Top-4 recall | Top-32 recall |
| --- | ---: | ---: | ---: |
| center FFT amplitude | 0.642 | 0.777 | 0.927 |
| multi-offset FFT evidence | 0.984 | 1.000 | 1.000 |

Current conclusion: the center FFT argmax assumption is not reliable at low SNR.
Oversampled FFT evidence is the first large gain: on the 33-byte -20 dB set the
traditional center argmax SER is 0.3584, while the phase-gated two-stage path
recovers every tested packet without payload-byte priors or cross-packet
information. The phase-consistency term is still essential, but it must be used
as a gated low-SNR discriminator inside the oversampled candidates. The -23 dB
results show both sides of that boundary: it rescues the len8 case slightly, but
when the candidate coverage is too low or the phase trend is poor, the decoder
should fall back to energy evidence instead of letting phase-only noise dominate.
