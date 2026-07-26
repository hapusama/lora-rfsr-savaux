# Paper Oversampled LoRa Demod Baseline

Date: 2026-06-17

Scope: `D:/Desktop/proj/gr-lora_sdr/weakPacket_decoding copy`

## Source

Baseline paper:

```text
Vincent Savaux,
"A Low-Complexity Demodulation for Oversampled LoRa Signal"
TechRxiv preprint, DOI: 10.36227/techrxiv.16657063
```

Implementation source boundary:

```text
Only Eq. (34)-(37) is implemented:
  split the OSR symbol into R branches
  compute the branch-specific DFT Y^(q,R)[k]
  combine branches with exp(-j 2*pi*q*k/(N*R))
  select argmax_k |combined[k]|^2
```

This baseline deliberately does not use:

```text
offset coherence
packet-line phase
Top-L candidate locking
payload byte/template prior
cross-packet prior
CRC-guided bin selection
```

## Added Files

```text
weak_decoder/baselines/savaux_oversampled/paper_oversampled_demod.py
scripts/experiments/baselines/savaux_oversampled/run_paper_oversampled_baseline.py
```

The module implements the paper demodulation metric. The runner evaluates it
against the existing center-branch chip-rate FFT argmax using the same
header-first timing and clean `raw_fft_bin` GT CSV.

## Local Sampling Convention

The header-first CSV stores symbol starts used by the local chip-center FFT
path. That path samples:

```text
start_sample + os_factor//2 + os_factor*n
```

Therefore the runner defaults to:

```text
--paper-origin-shift os_factor//2
```

This aligns the paper branch origin with the same local chip-center convention.
Use `--paper-origin-shift 0` only when `start_sample` is already the intended
oversampled branch origin.

## Sanity Checks

Compile:

```powershell
python -m py_compile `
  "weakPacket_decoding copy\weak_decoder\baselines\savaux_oversampled\paper_oversampled_demod.py" `
  "weakPacket_decoding copy\scripts\experiments\baselines\savaux_oversampled\run_paper_oversampled_baseline.py"
```

Synthetic chirp check:

```text
SF=7, OSR=4, symbol ids 0/1/2/17/63/126/127 all return the expected raw FFT bin.
```

Clean packet check:

```powershell
python "weakPacket_decoding copy\scripts\experiments\baselines\savaux_oversampled\run_paper_oversampled_baseline.py" `
  -i "data\USRP_IQ\0_0_0_10_14_16.bin" `
  -g "weakPacket_decoding copy\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv" `
  --max-packets 3 `
  --output "weakPacket_decoding copy\data\paper_oversampled_baseline\sanity_clean_3_packets.csv" `
  --symbols-output "weakPacket_decoding copy\data\paper_oversampled_baseline\sanity_clean_3_symbols.csv" `
  --summary-json "weakPacket_decoding copy\data\paper_oversampled_baseline\sanity_clean_3.summary.json"
```

Result:

```text
packets=3
center_symbol_ser=0.000
paper_symbol_ser=0.000
paper_crc_valid_rate=1.000
```

## Baseline Curve On Existing Noisy IQ

Input:

```text
data/low_snr_gt_bin/0_0_0_10_14_16_m22_m27_sto_input/
```

Summary output:

```text
data/paper_oversampled_baseline/0_0_0_10_14_16_m22_m27_summary.csv
```

Current results:

```text
SNR   packets  center_SER  paper_OSR_SER  paper_crc_rate
-22   11       0.603       0.005          0.818
-23   11       0.732       0.049          0.273
-24   11       0.834       0.104          0.000
-25   11       0.912       0.229          0.000
-26   11       0.951       0.371          0.000
-27   11       0.969       0.558          0.000
```

Interpretation:

```text
This is now the paper-only OSR baseline to beat.
It already explains why OSR evidence is extremely strong in this project.
Future enhancements should be compared against this curve, not just center FFT.
```
