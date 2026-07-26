# LiteNap-Savaux clean-GT added-noise experiment

## Scope

This experiment evaluates the `D=4`, `R=4` receiver proposed in
`doc/litemap+savaux.md` against the existing paper-faithful Savaux
oversampled detector.

The implementation contains:

- true sub-Nyquist views with `N/D` physical samples per `(q, d)` branch;
- wrap-aware candidate steering compatible with the existing Savaux equations;
- coherent fusion over selected oversampling phases `q` and downsampling views `d`;
- an optional training-free two-segment phase-jump reranker;
- a shared payload-GT-independent `+/-1 bin` calibration from the explicit header.

`K=D=4` uses every original sample and is numerically equivalent to the existing
Savaux detector. The observed maximum spectrum difference was
`1.922192e-06`.

The optional phase-jump reranker is an engineering diagnostic inspired by
LiteNap. It is not a complete reproduction of LiteNap's transmitter-specific
hardware phase fingerprint and preamble calibration.

## Data protocol

- Capture: `USRP_collector/data/branch4_fixed/high_snr/sf10_bw125_fs500_pre32_sw34_r001.bin`
- Configuration: SF10, BW 125 kHz, sample rate 500 ksample/s, OSR 4
- Evaluation set: 17 clean-synchronized packets, 833 payload symbols
- Ground truth: clean high-SNR, 10/10-consistent FFT-bin consensus
- Noise: copied payload symbols with complex AWGN added afterward
- Added SNR: clean payload reference power divided by added noise power
- Seeds: 42, 43, 44

The AWGN generator follows `noisy_iq`: the requested total complex-noise power
is divided equally between I and Q. All methods receive the same noisy symbol.
Clean timing, CFO, and SFO metadata are retained, so this is a controlled
symbol-demodulation comparison rather than an end-to-end noisy synchronization
test. Payload ground truth is used only after each method makes a hard-bin
decision.

One synchronized packet had a packet-wide `-1 bin` residual. The common
explicit-header calibration detected it with consensus `1.0` and applied a
`+1 bin` correction to every method. Clean Savaux, K1, and K2 then all reached
`0/833` errors without excluding the packet.

## Results

| Added SNR | Savaux, 4096 samples | Savaux + phase | K2, 2048 samples | K1, 1024 samples |
|---:|---:|---:|---:|---:|
| -22 dB | 1.60% | 4.32% | 22.05% | 64.55% |
| -24 dB | 11.20% | 18.53% | 47.58% | 83.31% |
| -26 dB | 33.17% | 43.98% | 72.51% | 92.44% |
| -28 dB | 60.02% | 71.55% | 87.76% | 97.20% |

These are 2,499 decisions per noisy condition: 833 symbols times three seeds.
Original Savaux is best at every tested added-SNR. K2 and K1 remain useful as
explicit sample-rate/complexity trade-offs, but they do not improve decoding
under white added noise.

The same-sample phase reranker also regresses:

| Added SNR | Fixes vs Savaux | Breaks vs Savaux |
|---:|---:|---:|
| -22 dB | 3 | 71 |
| -24 dB | 27 | 210 |
| -26 dB | 54 | 324 |
| -28 dB | 53 | 341 |

Candidate diagnostics show that most low-SNR failures already select the wrong
alias family. A phase fingerprint can only rerank candidates inside the selected
alias family, so it cannot repair those errors. On this USRP capture the
training-free phase jump is also not stable enough to replace LiteNap's
device-specific calibration.

## Reproduction

Run from `weakPacket_decoding`:

```powershell
python -B -m weak_decoder.os_lora.experiments.evaluate_litenap_savaux `
  --snrs -22 -24 -26 -28 `
  --seeds 42 43 44 `
  --batch-size 32 `
  --fingerprint-weight 0.05 `
  --output-dir data\experiments\litenap_savaux_clean_gt_20260724 `
  --verify-savaux-symbols 8
```

Generated artifacts:

- `RESULTS.md`: readable complete comparison
- `summary.csv`: aggregate comparison across seeds
- `summary_by_seed.csv`: reproducibility by seed
- `symbols.csv`: per-symbol decisions and phase diagnostics
- `config.json`: frozen input paths, parameters, and header calibrations

## Conclusion

The experiment validates the polyphase implementation and the proposed
sample-budget ladder, but it does not demonstrate a SER improvement over
Savaux. In white added noise, discarding samples loses signal energy, while
using all `d` phases reduces exactly to full Savaux. A further LiteNap claim
would require real transmitter-specific phase fingerprints, preamble
calibration, and evaluation on COTS-transmitted captures.
