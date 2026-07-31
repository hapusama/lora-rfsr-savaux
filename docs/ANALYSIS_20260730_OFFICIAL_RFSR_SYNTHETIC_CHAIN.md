# Published RF-SR checkpoint synthetic go/no-go audit

Date: 2026-07-30

## Scope

The vendored upstream tree contains the published checkpoints and synthetic example,
but no official OTA IQ files. This audit therefore makes only a local synthetic claim:

1. reproduce the published `encode -> AWGN -> 250 kS/s -> RF-SR -> decode` path;
2. compare ordinary FFT, Savaux, and Savaux+GLS on the same fixed packet bank;
3. distinguish noise before RF-SR from noise added after clean RF-SR and FrameSync.

No dataset was downloaded and no model was trained. The main matrix contains four
SF12 packets, 40 scored symbols per packet, and paired nested noise at `-18`, `-20`,
`-22`, and `-24 dB`. A FrameSync failure counts all 40 symbols as end-to-end errors.

Raw result:
`data/results/official_rfsr_synthetic_chain_20260730.json`

## Official synthetic decode reproduction

| SNR | Payload+CRC successes |
| ---: | ---: |
| -18 dB | 2/2 |
| -20 dB | 2/2 |
| -22 dB | 1/2 |
| -24 dB | 0/2 |

The local published synthetic checkpoint and upstream decoder therefore reproduce the
author example, but a single `-22 dB` success is not a stable packet-level result.

## Clean frontend diagnostics

| Frontend | Output power | Gain vs native | FrameSync integer CFO | Header start |
| --- | ---: | ---: | ---: | ---: |
| Native 1 MS/s | 1.000000 | 0.000 dB | 0 | 406404 |
| Published interpolation | 0.062500 | -12.041 dB | -1 | 406396 |
| Published synthetic RF-SR | 0.807582 | -0.928 dB | -1 | 406396 |
| Published OTA RF-SR | 0.062500 | -12.041 dB | -1 | 406396 |

All clean methods synchronize and have zero SER. The interpolation/RF-SR frontend
introduces a deterministic coupled timing/integer-bin offset relative to native IQ;
the independent FrameSync estimate compensates it.

The published OTA checkpoint behaves like interpolation on unit-amplitude synthetic
input. Its residual-network parameter norms are about `1e-4` to `1e-3`, with maximum
individual magnitudes below `2.9e-5`; it is effectively a near-zero residual model in
this domain. It cannot serve as evidence for OTA waveform recovery without OTA IQ.

## Noise before RF-SR

At `-18`, `-20`, and `-22 dB`, all methods synchronize all four packets and all three
demodulators have zero errors. The discriminating point is `-24 dB`:

| Frontend | Sync | FFT e2e SER | Savaux e2e SER | GLS e2e SER |
| --- | ---: | ---: | ---: | ---: |
| Native 1 MS/s | 3/4 | 46/160 = 28.75% | 40/160 = 25.00% | 40/160 = 25.00% |
| Published interpolation | 2/4 | 84/160 = 52.50% | 80/160 = 50.00% | 80/160 = 50.00% |
| Published synthetic RF-SR | 4/4 | 6/160 = 3.75% | 2/160 = 1.25% | 1/160 = 0.625% |
| Published OTA RF-SR | 2/4 | 84/160 = 52.50% | 80/160 = 50.00% | 80/160 = 50.00% |

The end-to-end RF-SR gain is real in this small synthetic audit, but it is primarily a
synchronization/denoising gain. Conditional on successful synchronization:

| Frontend | FFT conditional SER | Savaux conditional SER | GLS conditional SER |
| --- | ---: | ---: | ---: |
| Native 1 MS/s | 6/120 = 5.00% | 0/120 | 0/120 |
| Published interpolation | 4/80 = 5.00% | 0/80 | 0/80 |
| Published synthetic RF-SR | 6/160 = 3.75% | 2/160 = 1.25% | 1/160 = 0.625% |

Savaux's median margin increase at `-24 dB` is `+8.75 dB` for native IQ, `+3.14 dB`
for interpolation, and only `+1.82 dB` for synthetic RF-SR. The held-out pre-RF-SR
noise also has much stronger mean inter-branch correlation after reconstruction:

| Frontend | Mean absolute off-diagonal branch correlation |
| --- | ---: |
| Native 1 MS/s | 0.024-0.030 |
| Published interpolation | 0.280-0.303 |
| Published synthetic RF-SR | 0.410-0.493 |
| Published OTA RF-SR | 0.280-0.303 |

GLS changes two RF-SR Savaux errors to one error at only one operating point. A
one-symbol difference out of 160 is not evidence of a stable GLS gain.

## Noise after clean FrameSync

Using the same absolute post-frontend noise power is badly confounded by frontend
gain. At `-24 dB`, interpolation/OTA ordinary FFT SER is 100% and Savaux SER is
56.25%, while synthetic RF-SR ordinary FFT SER is 20% and Savaux SER is zero. This
mostly reflects the `-12.04 dB` versus `-0.93 dB` output gains above.

After matching noise power to each clean frontend output, all four methods are nearly
identical: at `-24 dB`, ordinary FFT SER is 5.625% for native/interpolation/OTA and
6.25% for synthetic RF-SR; Savaux and GLS are zero for every method. At `-18` through
`-22 dB`, every method is error-free.

This post-frontend test cannot prove recovered branch diversity. Adding white noise
after RF-SR creates eight new independent 1 MS/s noise samples per chip by construction;
Savaux can combine that artificial diversity even when the clean waveform came from a
250 kS/s deterministic interpolation.

## Decision

The local evidence supports this narrower claim:

> The published synthetic RF-SR checkpoint can improve weak-packet synchronization
> and ordinary/Savaux end-to-end SER relative to interpolation on its own synthetic
> generation path.

It does not yet support the intended high-sampling claim:

> A deterministic 250 kS/s-to-1 MS/s RF-SR frontend restores native polyphase branch
> diversity that Savaux/GLS can exploit as new observations.

The strongest evidence against that interpretation is the higher reconstructed branch
correlation, the much smaller Savaux margin gain than native IQ, and zero conditional
Savaux errors on synchronized native/interpolated packets versus two RF-SR errors.

This is therefore a conditional go for a synchronization/denoising frontend, but still
a no-go (or at minimum unproven) for the `RF-SR -> Savaux -> GLS` branch-diversity story.
The four-packet matrix is a feasibility audit, not a publication-quality performance
curve.
