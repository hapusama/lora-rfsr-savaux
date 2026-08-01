# RF Super-Resolution OTA LoRa Dataset

Over-the-air (OTA) LoRa dataset for RF super-resolution research. Each sample pairs a
received OTA signal (noisy, low-quality) with a clean reference signal (the transmitted
waveform), enabling supervised learning of signal quality improvement.

This dataset accompanies the MobiSys '26 paper:
> **RF Super Resolution: A Deep Learning Approach to Spatial Enhancement for LoRa**
> Andreas Kuster, Huatao Xu, Rui Tan, Mo Li — ACM MobiSys 2026
> DOI: https://doi.org/10.1145/3745756.3809216

## LoRa Parameters

| Parameter        | Value          |
|------------------|----------------|
| Center frequency | 923 MHz (AS923)|
| Spreading factor | 12             |
| Bandwidth        | 125 kHz        |
| Sample rate      | 2 MSPS         |
| Coding rate      | 4/8            |
| Payload length   | 16 bytes       |
| Preamble symbols | 8              |

## Directory Structure

```
rfsr_db/
  ota/          OTA received signals — model inputs  (complex64 IQ, .cfile)
  reference/    Clean transmitted signals — labels   (complex64 IQ, .cfile)
  metadata/     Per-packet JSON files with LoRa parameters and SNR annotations
```

## File Formats

**IQ files (`.cfile`)** — raw binary, interleaved float32 (I, Q, I, Q, …), complex64.
Each file contains one LoRa packet with 500,000-sample zero-padding on each side:

```
| zeros (500k samples) | LoRa packet | zeros (500k samples) |
```

Total length: `500,000 + 3,434,456 + 500,000 = 4,434,456` complex64 samples per file.

Load example (Python):
```python
import numpy as np
signal = np.fromfile("ota/exp0_000000_rxg24_0_fulltrim.cfile", dtype=np.complex64)
```

**OTA filename convention:** `expN_XXXXXX_rxgYY_Z_fulltrim.cfile`
- `expN`   — experiment index (exp0–exp12, different recording sessions / distances)
- `XXXXXX` — packet index (zero-padded 6 digits)
- `rxgYY`  — receiver gain in dB (e.g. `rxg24` = 24 dB)
- `Z`      — repetition index (always 0)

**Reference filename convention:** `signalout_XXXXXX_fulltrim.cfile`
- `XXXXXX` — packet index (mod 100), links to the corresponding `metadata/XXXXXX.json`

## Metadata JSON

Each `metadata/XXXXXX.json` describes one unique LoRa packet (payload + parameters) and
lists all OTA captures of that packet with their measured SNR:

```json
{
  "payload": [...],
  "center_freq": 923000000.0,
  "sf": 12,
  "bw": 125000.0,
  "sample_rate": 2000000.0,
  "num_samples": 3434456,
  ...,
  "files": [
    ["ota/exp0_000000_rxg24_0_fulltrim.cfile", -15.73],
    ...
  ]
}
```

SNR is estimated in-band (125 kHz window) using Welch's PSD method.

## Experimental Setup

Data was collected on the NTU university campus (Singapore). A fixed transmitter
(Ettus Research B205mini) was placed in an indoor lab. An Ettus Research N210 receiver
was moved to 13 distinct locations to capture diverse propagation conditions, including
indoor Line-of-Sight (LoS), short- and mid-range Non-Line-of-Sight (NLoS) through
dense urban building blocks, and an 800 m outdoor over-the-hill path. Transmit power
and receiver RF gain were varied systematically to sweep the SNR range.

Each experiment group (`expN`) corresponds to one receiver placement. Within each
group, the USRP receiver gain (`rxgYY`) is stepped to produce captures across a range
of SNR conditions:

| Experiment | Receiver gain range |
|------------|---------------------|
| exp0       | 0 – 30 dB           |
| exp1       | 0 – 30 dB           |
| exp2       | 6 – 30 dB           |
| exp3       | 0 – 30 dB           |
| exp4       | 0 – 30 dB           |
| exp5       | 21 – 30 dB          |
| exp6       | 40 – 70 dB          |
| exp7       | 40 – 70 dB          |
| exp8       | 49 – 70 dB          | 
| exp9       | 40 – 67 dB          |
| exp10      | 0 – 30 dB           |
| exp11      | 0 – 21 dB           |
| exp12      | 0 dB                |

The receiver gain steps are 3 dB apart. Higher receiver gain with a fixed transmitter
yields lower effective SNR (more thermal noise amplified relative to signal power).
exp6–exp9 used a higher transmit power regime (receiver gain offset +40 dB relative to
the other groups). exp2 and exp5 have restricted gain ranges due to link conditions at
those locations. exp11 is capped at 21 dB. exp12 sweeps channel variation over time at
a fixed gain rather than varying the gain setting.

## Dataset Statistics

- OTA received files:  9,528
- Reference files:       100
- Metadata files:        100  (one per unique packet, indexed 000000–000099)
- SNR range: approximately −35 dB to +10 dB
