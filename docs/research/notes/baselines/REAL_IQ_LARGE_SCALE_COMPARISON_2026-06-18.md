# Real IQ large-scale baseline comparison, 2026-06-18

## Scope

Data source: `gr-lora_sdr/data/USRP_IQ`.

The parameter note in that directory says the real USRP captures use:

- `samp-rate=500000`
- `BW=125000`
- `OS=4`
- `center-freq=487.7e6`
- `sync-word=0x34`
- `crc-mode=0`

The filename convention is:

`experiment_corridor_position_SF_TP_preamble.bin`

These real captures do not include byte-level or symbol-level ground truth, so the comparison uses packet CRC/PRR on header-valid packets. It does not report SER/BER.

## Methods

- `traditional_fft`: single center-window FFT argmax.
- `current_selected`: current phase/threshold weak decoder selection path.
- `savaux_paper`: paper-only oversampled LoRa demodulation baseline.
- `savaux_codec`: Savaux OSR evidence plus local codec/CRC beam. This is not the paper baseline; it is kept in the table as an engineering variant.

## Cross-SF/TP Fast Screen

Output directory:

`data/baseline_comparison/real_iq_crc_batch_param4_fastscreen_sample20m_hop2_probe3`

Run shape:

- all `.bin` files were grouped by `(SF, TP)`;
- up to 4 evenly spaced captures were selected per group;
- sync scanned the first 20,000,000 samples;
- up to 3 paired header-valid packets were CRC-probed per capture.

| SF | TP | captures | header-valid packets | FFT | current | Savaux paper | Savaux+codec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 14 | 3 | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 11 | 2 | 4 | 12 | 0.917 | 0.917 | 1.000 | 1.000 |
| 12 | 2 | 4 | 11 | 0.091 | 0.091 | 0.091 | 0.091 |
| 12 | 6 | 4 | 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | 10 | 4 | 12 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | 14 | 4 | 12 | 0.000 | 0.000 | 0.000 | 0.000 |

Packet-level difference in this fast screen:

- `1_0_15_11_2_16.bin`, packet 3: FFT/current failed, Savaux paper and Savaux+codec passed.
- `savaux_codec` had no extra CRC-valid packets beyond `savaux_paper` in this real-IQ screen.

## SF11/TP2 Fast Full Set

Output directory:

`data/baseline_comparison/real_iq_crc_batch_sf11_all_fastscreen_sample20m_hop2_probe3`

Run shape:

- all 17 captures under `lab1_sf11_TP2`;
- first 20,000,000 samples;
- up to 3 paired packets per capture.

| SF | TP | captures | header-valid packets | FFT | current | Savaux paper | Savaux+codec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 2 | 17 | 45 | 0.911 | 0.911 | 0.956 | 0.956 |

Packet-level differences:

- `1_0_15_11_2_16.bin`, packet 3: FFT/current failed, Savaux paper passed.
- `1_0_16_11_2_16.bin`, packet 0: FFT/current failed, Savaux paper passed.
- `savaux_codec` again matched `savaux_paper`; it did not add extra real-IQ CRC passes in this run.

## Weak SF11 Deep Check

Output directory:

`data/baseline_comparison/real_iq_crc_batch_sf11_weak_fullcheck`

Capture:

`lab1_sf11_TP2/1_0_16_11_2_16.bin`

Full header-valid packet check:

| method | CRC-valid packets |
| --- | ---: |
| traditional FFT | 0/4 |
| multi-offset argmax | 1/4 |
| current selected | 1/4 |
| Savaux paper | 2/4 |
| Savaux+codec | 2/4 |

## Additional Large Runs

### SF12/TP10 full-group partial deep screen

Output directory:

`data/baseline_comparison/real_iq_crc_batch_all_fastscreen_sample20m_hop2_probe3_rank64`

This run used the reliable sync settings from the cross-SF/TP fast screen:

- first 20,000,000 samples;
- `hop-chirps=2`;
- `min-periodic-peaks=6`;
- `max-events=8`;
- `align-step-samples=8`;
- `frame-step-samples=8`;
- `header-max-frames=8`;
- up to 3 paired packets per capture;
- `crc-candidate-max-beam-rank=64` for the local Savaux+codec extension.

The full all-capture job was stopped by the one-hour command timeout, but it had already written completed per-capture rows. The useful completed part is a full SF12/TP10 group plus SF10/TP14 and one SF12/TP14 capture:

| SF | TP | captures | header-valid packets | FFT | current | Savaux paper | Savaux+codec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 14 | 3 | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| 12 | 10 | 17 | 51 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | 14 | 1 | 3 | 0.000 | 0.000 | 0.000 | 0.000 |

This reinforces the earlier observation that these SF12/TP10 windows are not separating payload demodulators yet: after header detection, all compared payload methods fail CRC on the checked packets.

### Discarded param8 screen

Output directory:

`data/baseline_comparison/real_iq_crc_batch_param8_fastscreen_sample20m_hop2_probe3`

This run is not used for conclusions. It omitted the reliable sync-search settings above, and most SF11/SF12 captures produced zero header-valid packets even though the earlier calibrated fast screen found valid packets. It is kept only as a negative run showing that the real-IQ comparison is sensitive to sync-search settings.

## Takeaways

The real-IQ comparison currently has three regimes:

1. SF10/TP14 is easy: all methods pass.
2. SF11/TP2 is the useful transition region: Savaux paper OS combining is consistently stronger than center FFT/current selection by about 2 packets in 45 in the fast full-set run, and by 2/4 vs 1/4 or 0/4 on the weakest deep-check file.
3. SF12 TP2/6/10/14 mostly fails for all methods after header detection, so these captures are not yet useful for distinguishing payload demodulators without improving synchronization/frame selection or finding better packet windows.

For real IQ, the pure Savaux paper baseline is currently stronger than the existing current-selected path in the SF11 transition region. The local `savaux_codec` extension improves AWGN simulation thresholds, but on these real captures it mostly preserves Savaux hard CRC successes and does not exceed the pure Savaux baseline yet.
