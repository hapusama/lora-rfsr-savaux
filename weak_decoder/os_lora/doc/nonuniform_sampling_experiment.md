# Non-Uniform Sampling Experiment Log

This note records the first runnable `os_lora` experiment for non-uniform
oversampling evidence.  The implementation is intentionally limited to the
symbol evidence layer: no CRC voting, no packet template, no phase-line prior,
and no cross-packet rescue logic.

## Implementation

- `weak_decoder/os_lora/system/nonuniform_sampling.py`
  - builds a deterministic bank of oversampling-offset patterns `c_b[p]`;
  - scores Savaux top-k raw FFT bins with each pattern;
  - reports best-pattern, mean-pattern, coherent-pattern, and
    covariance-whitened coherent scores.
- `weak_decoder/os_lora/experiments/evaluate_nonuniform_sampling.py`
  - loads the existing header-first datasets through `weak_decoder.baselines.common`;
  - compares non-uniform scores against Savaux oversampled evidence;
  - writes per-packet metrics plus CSV/JSON summaries.

## Smoke Sweep

Command:

```powershell
python "weakPacket_decoding\weak_decoder\os_lora\experiments\evaluate_nonuniform_sampling.py" `
  --datasets 0_0_0_10_14_16 `
  --snrs -22 -23 -24 `
  --seeds 42 `
  --max-packets 2 `
  --top-k 8 `
  --output-dir "weakPacket_decoding\data\tmp_os_lora_sweep"
```

Result:

| SNR dB | Symbols | GT in Savaux Top-8 | Savaux SER | Best Pattern SER | Mean Pattern SER | Coherent SER | Whitened SER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -22 | 70 | 1.000 | 0.0000 | 0.1429 | 0.0000 | 0.0000 | 0.0000 |
| -23 | 70 | 1.000 | 0.0143 | 0.3000 | 0.0286 | 0.0143 | 0.0143 |
| -24 | 70 | 1.000 | 0.1143 | 0.4857 | 0.1286 | 0.1143 | 0.1143 |

## Current Read

The simple pattern bank does not yet beat Savaux on this small sweep.  The
true bin is already inside Savaux Top-8 for every tested symbol, so the
bottleneck is not candidate coverage here.  The best single non-uniform pattern
strongly overfits noise.  Equal-power averaging is slightly worse than Savaux
at lower SNR.  Coherent and covariance-whitened combining track Savaux but do
not create a stable correction yet.

The useful direction is therefore not "pick the best weird branch".  If this
idea is worth pursuing, the next step should treat the non-uniform branches as
correlated linear observations and design the projection/weights from a signal
and noise model, probably tied to fractional timing or SFO/CFO uncertainty.

## Literature Sanity Check

Classic non-uniform sampling work does support useful reconstruction or
sub-Nyquist acquisition, but usually by adding a signal model, multiple
well-posed channels, or spectral sparsity:

- Landau's density conditions are the old warning sign: sampling density is
  tied to the signal space measure, so a pattern alone does not create extra
  independent degrees of freedom.
- Papoulis generalized sampling shows that multiple linear-system outputs can
  reconstruct a bandlimited signal, but the channel set must be well posed.
- Periodic non-uniform sampling for multiband signals can approach Landau-rate
  reconstruction when the support model is known or sparse.
- Random demodulation / compressed sensing helps sparse bandlimited signals by
  mixing and solving a model-based recovery problem, not by selecting the
  strongest random branch independently.

For LoRa, this points toward a generalized least-squares or sparse one-tone
projection over candidate bins and timing/CFO uncertainty, with an explicit
noise covariance.  It does not support the naive idea that many non-uniform
branches can be treated as many independent coherent replicas.

References:

- H. J. Landau, "Necessary density conditions for sampling and interpolation of certain entire functions," Acta Mathematica, 1967. https://doi.org/10.1007/BF02395039
- A. Papoulis, "Generalized sampling expansion," IEEE Transactions on Circuits and Systems, 1977. https://epubs.siam.org/doi/10.1137/0144043
- R. Venkataramani and Y. Bresler, "Optimal sub-Nyquist nonuniform sampling and reconstruction for multiband signals," IEEE Transactions on Signal Processing, 2001. https://experts.illinois.edu/en/publications/optimal-sub-nyquist-nonuniform-sampling-and-reconstruction-for-mu/
- J. A. Tropp, J. N. Laska, M. F. Duarte, J. K. Romberg, and R. G. Baraniuk, "Beyond Nyquist: Efficient Sampling of Sparse Bandlimited Signals," IEEE Transactions on Information Theory, 2010. https://users.cms.caltech.edu/~jtropp/papers/TLDRB10-Beyond-Nyquist.pdf

## Matrix Search Pass

Added `weak_decoder/os_lora/experiments/analyze_nonuniform_matrix.py` and expanded the
pattern bank builder with these families:

- `fixed`: the classical OSR fixed branches.
- `basic`: fixed + linear, alternating, and piecewise patterns.
- `periodic`: fixed + periodic non-uniform motifs.
- `dither`: fixed + sparse offset-dither motifs.
- `random` / `balanced_random`: fixed + random or offset-balanced random
  patterns.
- `*_only`: the same non-uniform banks with the fixed branches removed.
- `search`: a larger mixed bank.

The matrix metric is:

```text
effective_replicas = a.H C^+ a / N
```

where `a` is the ideal target response vector and `C = S S.H` is the white-noise
covariance of the pattern outputs.  This is the number of independent OSR
replicas implied by the bank after the correlation between reused samples is
accounted for.

Command:

```powershell
python "weakPacket_decoding\weak_decoder\os_lora\experiments\analyze_nonuniform_matrix.py" `
  --raw-bins 1 61 301 653 900 1023 `
  --bank-kinds fixed basic periodic dither random balanced_random search `
  --random-count 64 `
  --leakage-bins 64 `
  --output-dir "weakPacket_decoding\data\tmp_os_lora_matrix"
```

Key result:

| Bank | Pattern Count | Matrix Rank | Effective Replicas | Target Response |
| --- | ---: | ---: | ---: | ---: |
| fixed | 4 | 4 | 4.000000 | all patterns have amplitude 32 and phase 0 |
| basic | 60 | 20 | 4.000000 | all patterns have amplitude 32 and phase 0 |
| periodic | 72 | 41 | 4.000000 | all patterns have amplitude 32 and phase 0 |
| dither | 208 | 55 | 4.000000 | all patterns have amplitude 32 and phase 0 |
| random | 68 | 68 | 4.000000 | all patterns have amplitude 32 and phase 0 |
| balanced_random | 68 | 68 | 4.000000 | all patterns have amplitude 32 and phase 0 |
| search | 371 | 148 | 4.000000 | all patterns have amplitude 32 and phase 0 |

Pure non-uniform banks also confirm the same ceiling:

| Bank | Pattern Count | Matrix Rank | Effective Replicas |
| --- | ---: | ---: | ---: |
| basic_only | 56 | 20 | 4.000000 |
| periodic_only | 68 | 40 | 4.000000 |
| dither_only | 204 | 55 | 4.000000 |
| random_only | 128 | 128 | 3.911460 |
| balanced_random_only | 128 | 128 | 3.914396 |
| search_only | 431 | 212 | 4.000000 |

This answers the "can it be coherent?" question more carefully:

- Yes, with the LoRa/Savaux wrap-tail correction, a non-uniform pattern output
  is coherent on an ideal target symbol.  In the tested bins, every pattern
  produced amplitude `sqrt(N)=32` and near-zero phase.
- No, those coherent outputs are not independent replicas.  They are correlated
  linear measurements of the same `NR` samples.  Once `C` is included, the
  effective independent replica count does not exceed `R=4`.

## Additional SER Search

Commands swept `fixed`, `basic_only`, `random_only`, `balanced_random_only`,
and `search_only` on two packets from `0_0_0_10_14_16`.

At `top-k=8`, `snr=-24,-25`, seed 42:

| Bank | SNR | Savaux SER | Best Pattern | Mean | Coherent | Whitened/GLS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | -24 | 0.1143 | 0.5714 | 0.2286 | 0.1143 | 0.1143 |
| fixed | -25 | 0.2286 | 0.7286 | 0.3714 | 0.2286 | 0.2286 |
| basic_only | -24 | 0.1143 | 0.5000 | 0.1286 | 0.1143 | 0.1143 |
| basic_only | -25 | 0.2286 | 0.6714 | 0.2714 | 0.2286 | 0.2286 |
| random_only | -24 | 0.1143 | 0.3571 | 0.1000 | 0.1143 | 0.1143 |
| random_only | -25 | 0.2286 | 0.4857 | 0.2571 | 0.2429 | 0.2571 |
| balanced_random_only | -24 | 0.1143 | 0.3429 | 0.1286 | 0.1286 | 0.1286 |
| balanced_random_only | -25 | 0.2286 | 0.4429 | 0.2143 | 0.2143 | 0.2143 |
| search_only | -24 | 0.1143 | 0.3714 | 0.1714 | 0.1429 | 0.1143 |
| search_only | -25 | 0.2286 | 0.5714 | 0.2857 | 0.2714 | 0.2286 |

Random-bank seed sweep at `snr=-25` did not find a stable winner.  Some mean or
coherent aggregations tie Savaux, but others break more symbols than they fix.
The best-pattern rule remains consistently bad because it selects noise peaks.

At `snr=-26`, `top-k=16`, seed 42:

| Bank | Savaux SER | Best Pattern | Mean | Coherent | Whitened/GLS |
| --- | ---: | ---: | ---: | ---: | ---: |
| basic_only | 0.4429 | 0.8143 | 0.4571 | 0.4429 | 0.4429 |
| random_only | 0.4429 | 0.7571 | 0.4286 | 0.4286 | 0.4429 |
| balanced_random_only | 0.4429 | 0.7143 | 0.4857 | 0.4571 | 0.4571 |

The only observed improvement is the `random_only` mean/coherent result at
`-26 dB`, which fixes one net symbol out of 70.  That is an empirical
candidate-reranking effect, not evidence for more than four independent
coherent replicas.

Current conclusion: non-uniform sampling can form coherent LoRa-corrected
measurements, but the matrix covariance keeps the coherent SNR ceiling at the
oversampling factor.  The remaining useful path is not "more branches for more
energy"; it is a deliberately biased reranker or regularized projection that
accepts some non-ML behavior to trade breaks for fixes at very low SNR.

## Full-Spectrum FFT Coherent Sum Pass

Added `weak_decoder/os_lora/experiments/evaluate_pattern_fft_coherence.py` to move from
candidate-only scoring back to a decoder-like full-spectrum test.  For each
payload symbol, it now computes:

- `plain_coherent`: take each constructed non-uniform N-sample branch, run a
  plain FFT, and coherently average the spectra.
- `lora_coherent`: compute the LoRa/Savaux-corrected pattern spectrum for every
  bin, then coherently average.
- `lora_mean_power`: average pattern powers per bin.
- `lora_best_power`: take the maximum pattern power per bin.

The script reports SER plus GT-bin rank, GT/next-bin margin, and GT/median-floor
ratio.

Smoke command:

```powershell
python "weakPacket_decoding\weak_decoder\os_lora\experiments\evaluate_pattern_fft_coherence.py" `
  --datasets 0_0_0_10_14_16 `
  --snrs -24 `
  --seeds 42 `
  --max-packets 1 `
  --bank-kinds fixed basic_only random_only `
  --random-count 16 `
  --max-patterns 32 `
  --output-dir "weakPacket_decoding\data\tmp_pattern_fft_smoke"
```

Smoke result:

| Bank | SNR | Savaux SER | Plain Coherent | LoRa Coherent | Mean Power | Best Power |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | -24 | 0.1143 | 0.6286 | 0.1143 | 0.4000 | 0.8000 |
| basic_only_first32 | -24 | 0.1143 | 0.6286 | 0.1143 | 0.1429 | 0.8286 |
| random_only | -24 | 0.1143 | 0.6857 | 0.2286 | 0.2286 | 0.7143 |

Small sweep command:

```powershell
python "weakPacket_decoding\weak_decoder\os_lora\experiments\evaluate_pattern_fft_coherence.py" `
  --datasets 0_0_0_10_14_16 `
  --snrs -25 -26 `
  --seeds 42 `
  --max-packets 1 `
  --bank-kinds fixed basic_only dither_only random_only balanced_random_only `
  --random-count 16 `
  --max-patterns 32 `
  --output-dir "weakPacket_decoding\data\tmp_pattern_fft_sweep"
```

Key SER result:

| Bank | SNR | Savaux | Plain Coherent | LoRa Coherent | Mean Power | Best Power |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | -25 | 0.2857 | 0.7143 | 0.2857 | 0.5714 | 0.8571 |
| fixed | -26 | 0.4857 | 0.8000 | 0.4857 | 0.8000 | 0.8857 |
| basic_only_first32 | -25 | 0.2857 | 0.7143 | 0.2857 | 0.3714 | 0.8857 |
| basic_only_first32 | -26 | 0.4857 | 0.8000 | 0.4857 | 0.5143 | 0.8857 |
| dither_only_first32 | -25 | 0.2857 | 0.8000 | 0.7143 | 0.7429 | 0.8286 |
| dither_only_first32 | -26 | 0.4857 | 0.8571 | 0.8286 | 0.8571 | 0.9429 |
| random_only | -25 | 0.2857 | 0.7143 | 0.3714 | 0.4000 | 0.8286 |
| random_only | -26 | 0.4857 | 0.7429 | 0.5714 | 0.6000 | 0.9143 |
| balanced_random_only | -25 | 0.2857 | 0.7714 | 0.4000 | 0.4000 | 0.8286 |
| balanced_random_only | -26 | 0.4857 | 0.7714 | 0.6000 | 0.5714 | 0.9143 |

GT-bin diagnostics did not show a hidden win.  For `basic_only_first32`, the
LoRa-coherent spectrum exactly tracks Savaux: GT rank, GT/next-bin margin, and
GT/median-floor ratio are the same to numerical precision.  Random banks do
produce some fixes, but more breaks; their GT margin and GT rank are worse on
average.  The best-pattern spectrum remains a noise-peak selector.

Random seed sweep at `snr=-26`, one packet, 16 patterns:

| Bank Seed | Savaux | Plain Coherent | LoRa Coherent | Mean Power | Best Power |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4857 | 0.7429 | 0.5714 | 0.6000 | 0.9143 |
| 1 | 0.4857 | 0.8857 | 0.5429 | 0.5429 | 0.8571 |
| 2 | 0.4857 | 0.8571 | 0.5714 | 0.6000 | 0.9143 |
| 3 | 0.4857 | 0.8571 | 0.5714 | 0.6000 | 0.8857 |
| 4 | 0.4857 | 0.8857 | 0.5429 | 0.6000 | 0.8571 |
| 5 | 0.4857 | 0.8571 | 0.5714 | 0.5714 | 0.8571 |

Readout: directly constructed-sample FFTs are not aligned enough for LoRa.  The
LoRa-corrected coherent spectra can reproduce Savaux from non-uniform branches,
but the tested banks do not make the GT bin higher relative to the competing
peaks or the median floor.  This further supports treating non-uniform sampling
as a possible biased reranker, not as a stronger coherent-energy combiner.

## Literature-Inspired Reranker Pass

The next pass used four external ideas, translated back into the current matrix
view:

- Lomb-Scargle / least-squares periodogram: non-uniform samples should be scored
  as a candidate-specific projection or fit, not as a plain FFT on fake uniform
  positions.
- Periodic non-uniform sampling for multiband signals: the sampling pattern is a
  matrix-design problem, and noise/error sensitivity matters as much as nominal
  reconstruction.
- Capon / adaptive spectral estimation: when false peaks are structured, use
  data-dependent covariance or consistency to suppress sidelobes instead of only
  summing fixed windows.
- Coherence-pattern guided sparse recovery: in a highly coherent dictionary,
  the useful move is often conservative candidate rejection/reranking rather
  than blindly selecting the largest local proxy.

Implemented code:

- `pattern_coherence_weighted_power(...)` in
  `weak_decoder/os_lora/system/nonuniform_sampling.py`.
- `adaptive_gls_spectrum_power(...)` in the same file.
- `stable_pattern`, `gated_mean_pattern`, `gated_stable_pattern`, and
  `gated_consensus_pattern` in
  `weak_decoder/os_lora/experiments/evaluate_nonuniform_sampling.py`.
- `lora_stable_power` and `lora_adaptive_gls` in
  `weak_decoder/os_lora/experiments/evaluate_pattern_fft_coherence.py`.

The most useful score so far is:

```text
coherence_ratio[k] = |mean_b Y_b[k]|^2 / mean_b |Y_b[k]|^2
stable_score[k] = savaux_power[k] * coherence_ratio[k]
```

This keeps Savaux as the main energy detector, then downweights candidates whose
non-uniform pattern outputs disagree in phase/amplitude.  It is a false-peak
consistency penalty, not an extra-SNR combiner.

Candidate-only command, two packets:

```powershell
python "weakPacket_decoding\weak_decoder\os_lora\experiments\evaluate_nonuniform_sampling.py" `
  --datasets 0_0_0_10_14_16 `
  --snrs -24 -25 -26 `
  --seeds 1 2 3 42 `
  --max-packets 2 `
  --top-k 16 `
  --bank-kind random_only `
  --random-count 32 `
  --bank-seed 2 `
  --stable-exponent 1.0 `
  --gate-margin-db 1.0 `
  --gate-switch-ratio 1.1 `
  --output-dir "weakPacket_decoding\data\tmp_nonuniform_gcons_seed_sweep_2pkt"
```

Aggregate result over 840 payload symbols:

| Method | Errors | SER | Fix vs Savaux | Break vs Savaux | Changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Savaux | 236 | 0.280952 | 0 | 0 | 0 |
| Stable pattern | 233 | 0.277381 | 15 | 12 | 71 |
| Gated stable | 234 | 0.278571 | 5 | 3 | 21 |
| Gated consensus | 233 | 0.277381 | 3 | 0 | 5 |

Default-dataset smoke, one packet per dataset:

```powershell
python "weakPacket_decoding\weak_decoder\os_lora\experiments\evaluate_nonuniform_sampling.py" `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -25 -26 `
  --seeds 1 2 42 `
  --max-packets 1 `
  --top-k 16 `
  --bank-kind random_only `
  --random-count 32 `
  --bank-seed 2 `
  --stable-exponent 1.0 `
  --gate-margin-db 1.0 `
  --gate-switch-ratio 1.1 `
  --output-dir "weakPacket_decoding\data\tmp_nonuniform_gcons_default_datasets"
```

Aggregate result over 630 payload symbols:

| Method | Errors | SER | Fix vs Savaux | Break vs Savaux | Changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Savaux | 309 | 0.490476 | 0 | 0 | 0 |
| Stable pattern | 307 | 0.487302 | 8 | 6 | 70 |
| Gated consensus | 309 | 0.490476 | 1 | 1 | 8 |

Additional checks:

- `balanced_random_only` was worse on the same default-dataset smoke
  (`stable_pattern` 312 errors vs Savaux 309).
- `top-k=24` produced the same aggregate as `top-k=16`, so the current
  bottleneck is not candidate-pool width.
- Stronger consistency exponents hurt: `stable_exponent=0.5, 1.5, 2.0` gave
  310, 312, and 315 errors respectively on the 630-symbol default smoke.

Current status: the first reproducible lower-SER variant is
`random_only + stable_score`, with a very small gain.  The safer
`gated_consensus` variant can produce a no-break result on the 2-packet
`0_0_0_10_14_16` sweep, but it only ties Savaux on the default-dataset smoke.
This is promising enough to keep, but not strong enough to declare the
non-uniform demod solved.

## Gated Hybrid Mean Reranker

Candidate diagnostics showed why `stable_score` only helps a little.  On the
default 630-symbol diagnostic set, Savaux is wrong on 309 symbols, and the GT
bin is still present in the Savaux top-16 on 166 of those errors.  However, the
GT candidate often has lower pattern consistency than the Savaux false peak, so
consistency alone cannot recover most of the available headroom.

The better score is a geometric blend between Savaux power and mean pattern
power:

```text
hybrid_score[k] = SavauxPower[k] * (MeanPatternPower[k] / SavauxPower[k])^beta
                = SavauxPower[k]^(1-beta) * MeanPatternPower[k]^beta
```

The current best small-grid setting is:

- `beta = 0.75`
- `bank_kind = random_only`
- `random_count = 32`
- `bank_seed = 2`
- only allow switching within Savaux top-3
- only switch if the hybrid score beats the Savaux candidate by `1.1x`
- only switch when the Savaux top-1/top-2 margin is at most `1.5 dB`

This is now implemented as `gated_hybrid_pattern` in
`evaluate_nonuniform_sampling.py`.

Default-dataset, two-packet command:

```powershell
python "weakPacket_decoding\weak_decoder\os_lora\experiments\evaluate_nonuniform_sampling.py" `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --snrs -25 -26 `
  --seeds 1 2 42 `
  --max-packets 2 `
  --top-k 16 `
  --bank-kind random_only `
  --random-count 32 `
  --bank-seed 2 `
  --stable-exponent 1.0 `
  --hybrid-mean-beta 0.75 `
  --hybrid-max-rank 3 `
  --gate-margin-db 1.5 `
  --gate-switch-ratio 1.1 `
  --output-dir "weakPacket_decoding\data\tmp_nonuniform_gated_hybrid_default_2pkt"
```

Aggregate result over 1260 payload symbols:

| Method | Errors | SER | Fix vs Savaux | Break vs Savaux | Changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Savaux | 466 | 0.369841 | 0 | 0 | 0 |
| Stable pattern | 469 | 0.372222 | 16 | 19 | 112 |
| Hybrid mean, ungated | 485 | 0.384921 | 33 | 52 | 209 |
| Gated hybrid | 457 | 0.362698 | 14 | 5 | 85 |
| Gated consensus | 464 | 0.368254 | 2 | 0 | 9 |

Baseline comparison on the same datasets/SNRs/seeds/packet count:

| Method | Errors | SER |
| --- | ---: | ---: |
| Savaux oversampled | 466 | 0.369841 |
| LoRaTrimmer | 602 | 0.477778 |
| SymFEC | 987 | 0.783333 |
| UniChirp | 715 | 0.567460 |
| Gated hybrid non-uniform | 457 | 0.362698 |

Readout: the ungated hybrid score is too aggressive, but the gated version is
the strongest non-uniform result so far.  It wins by 9 net symbols over Savaux
on this 1260-symbol smoke and also beats the other standalone baselines on the
same noisy realizations.  The gain is still modest; this is a viable reranker,
not yet a dramatic new detector.

## Stop Condition for Local Reranker Tuning

After widening the validation, the local non-uniform reranker branch no longer
looks like a high-upside path.

With `-24/-25/-26 dB`, three datasets, three packets per dataset, and seeds
`1 2 42`:

| Method | Errors | SER | Net vs Savaux |
| --- | ---: | ---: | ---: |
| Savaux | 837 / 2835 | 0.295238 | 0 |
| gated hybrid, beta=0.75 | 833 / 2835 | 0.293827 | +4 symbols |
| gated hybrid, beta=0.5, consistency alpha=0.25 | 825 / 2835 | 0.291005 | +12 symbols |

With the more moderate `-22/-23/-24 dB` range, three datasets, three packets per
dataset, and seeds `1 2 42`:

| Method | Errors | SER | Net vs Savaux |
| --- | ---: | ---: | ---: |
| Savaux | 384 / 2835 | 0.135450 | 0 |
| gated hybrid, beta=0.5, consistency alpha=0.25 | 381 / 2835 | 0.134392 | +3 symbols |

This is not the kind of gain we are looking for.  It is not a
`SER 0.2 -> 0.1` jump; it is a small ranking perturbation.  The current
evidence says:

- Non-uniform pattern reranking can sometimes fix individual Savaux mistakes.
- The same mechanism also breaks some previously correct symbols.
- The net gain is too small to justify continued threshold/parameter tuning.

Decision: stop local `stable` / `hybrid` / `gated_hybrid` parameter chasing.
Keep the code as a diagnostic and negative result, but do not treat this branch
as the main route to a stronger demodulator.

Only a qualitatively different approach is worth another round, for example:

- estimating timing/CFO/SFO better before Savaux rather than reranking after it;
- using redundancy across coded payload symbols or decoder constraints in a
  principled soft-decision loop;
- learning an interference/noise model from many symbols, not from one symbol's
  non-uniform pattern bank;
- designing a new observation model that changes the front-end information,
  instead of recombining the same oversampled samples.

## Timing-Origin Sanity Check

Because the local reranker gain was too small, we checked whether the Savaux
baseline was simply using a bad fixed oversampling origin.  This is a more
plausible high-upside failure mode than top-k reranking: if the receiver is
sampling the wrong branch, a corrected origin could produce a large SER jump.

Sweep setup:

- datasets: `0_0_0_10_14_8`, `0_0_0_10_14_16`, `0_0_0_10_14_32`
- SNRs: `-22`, `-23`, `-24 dB`
- seeds: `1`, `2`, `42`
- max packets per dataset: `3`
- tested fixed origins: `0`, `1`, `2`, `3`

Aggregate payload SER:

| Origin shift | Errors | SER |
| ---: | ---: | ---: |
| 0 | 1884 / 2835 | 0.664550 |
| 1 | 587 / 2835 | 0.207055 |
| 2 | 384 / 2835 | 0.135450 |
| 3 | 591 / 2835 | 0.208466 |

The current baseline already uses `origin_shift = os_factor // 2 = 2`, which is
the best fixed origin by a large margin.  This rules out a simple "wrong
oversampling branch" explanation for the remaining SER.  Any meaningful next
step must change more than the fixed OSR origin.

## Mid-SNR Baseline and Top-K Oracle

The goal was updated to avoid over-focusing on very low SNR.  We therefore
checked `-22/-23/-24 dB` directly.

Baseline comparison on three datasets, three packets per dataset, seeds
`1 2 42`, total 2835 payload symbols:

| Method | Errors | SER |
| --- | ---: | ---: |
| Savaux oversampled | 384 / 2835 | 0.135450 |
| LoRaTrimmer | 522 / 2835 | 0.184127 |
| SymFEC | 1110 / 2835 | 0.391534 |
| UniChirp | 718 / 2835 | 0.253263 |

The non-uniform gated-hybrid reranker with the best mid-SNR setting only reached
381 / 2835 errors, SER 0.134392.  This is a +3-symbol net gain over Savaux, so
it is not meaningful.

However, the Savaux top-k oracle shows a different kind of headroom:

| Selector | Errors | SER |
| --- | ---: | ---: |
| Savaux top-1 | 384 / 2835 | 0.135450 |
| Oracle within top-2 | 317 / 2835 | 0.111817 |
| Oracle within top-4 | 268 / 2835 | 0.094533 |
| Oracle within top-8 | 240 / 2835 | 0.084656 |
| Oracle within top-16 | 212 / 2835 | 0.074780 |

Readout:

- The current non-uniform pattern scores are too weak to realize this oracle
  gain.
- But a stronger soft-decision / coding-constraint method might have enough
  room to matter, especially on datasets where the GT bin is often in the
  Savaux top-k.
- The hard dataset `0_0_0_10_14_32` has much less top-k headroom; even top-16
  oracle remains at 210 / 945 errors, SER 0.222222.

Conclusion: stop non-uniform reranker tuning.  If we continue, the only
defensible next branch is not "more patterns"; it is a top-k soft-decision
decoder or payload-level constraint method that can exploit the candidates
Savaux already surfaces.

Useful references used for this pass:

- Lomb-Scargle periodogram documentation:
  https://docs.astropy.org/en/stable/timeseries/lombscargle.html
- Venkataramani and Bresler, "Optimal Sub-Nyquist Nonuniform Sampling and
  Reconstruction for Multiband Signals":
  https://mandolinraman.github.io/pubs/optimal-syb-Nyquist-sampling.pdf
- Capon, "High-Resolution Frequency-Wavenumber Spectrum Analysis":
  https://epsc.wustl.edu/~ggeuler/reading/cam_noise_biblio/capon_1969-ieee-high-resolution_frequency-wavenumber_spectrum_analysis.pdf
- Mishali and Eldar, "From Theory to Practice: Sub-Nyquist Sampling of Sparse
  Wideband Analog Signals": https://arxiv.org/abs/0902.4291
- Fannjiang and Liao, "Coherence-Pattern Guided Compressive Sensing with
  Unresolved Grids": https://arxiv.org/abs/1106.5177

## Empirical Noise Covariance Pass

This pass answers a narrower question: is the original sample noise close enough
to white that the earlier AWGN matrix conclusion can be treated as an empirical
fact?

Using the notation in `符号参考.md`, do not introduce a new `A_m` symbol.  For
one off-packet noise window `w`, keep the same dechirped sequence notation:

```math
z_\nu^{(w)}[n],
\qquad
n=0,\dots,NR-1 .
```

For pattern `b` and candidate bin `m`, the measured noise-only pattern output is
the same LoRa-corrected matched projection as before:

```math
Y_{b,\nu}^{(w)}[m]
=
\frac{1}{\sqrt N}
\sum_{p=0}^{N-1}
z_\nu^{(w)}[Rp+c_b[p]]
e^{-j2\pi m(Rp+c_b[p])/(NR)}
\tau_m(p,c_b[p]) .
```

The empirical pattern covariance is then indexed directly by two pattern
indices `b,b'`:

```math
\widehat C_{bb'}[m]
=
\frac{1}{W-1}
\sum_{w=1}^{W}
\left(
Y_{b,\nu}^{(w)}[m]-\bar Y_{b,\nu}[m]
\right)
\left(
Y_{b',\nu}^{(w)}[m]-\bar Y_{b',\nu}[m]
\right)^* .
```

For an ideal LoRa symbol whose true bin is `m`, the signal-only branch response
is:

```math
Y_{b,\mathrm{sig}}[m]
=
\frac{1}{\sqrt N}
\sum_{p=0}^{N-1}
z_m[Rp+c_b[p]]
e^{-j2\pi m(Rp+c_b[p])/(NR)}
\tau_m(p,c_b[p]) .
```

With the LoRa correction, this is coherent on the ideal target:

```math
Y_{b,\mathrm{sig}}[m]
=
\sqrt N
\quad
\text{for every tested pattern } b .
```

The AWGN model used earlier is only the special case:

```math
\mathbb E[\nu[n]\nu^*[n']]
=
\sigma^2\delta[n-n'] .
```

In the more general colored-noise case, write the sample-domain noise
correlation as:

```math
R_\nu[n,n']
=
\mathbb E[\nu[n]\nu^*[n']] .
```

Then the pattern covariance is explicitly:

```math
C_{bb'}[m]
=
\frac{1}{N}
\sum_{p=0}^{N-1}
\sum_{p'=0}^{N-1}
R_\nu[n_{b,p},n_{b',p'}]
e^{-j2\pi m(n_{b,p}-n_{b',p'})/(NR)}
\tau_m(p,c_b[p])
\tau_m^*(p',c_{b'}[p']) ,
```

where:

```math
n_{b,p}=Rp+c_b[p],
\qquad
n_{b',p'}=Rp'+c_{b'}[p'] .
```

White noise is what turns `R_nu[n,n']` into `sigma^2 delta[n-n']`.  That is the
extra assumption behind the old `E_m <= R` ceiling.  The condition that the
ideal signal response is inside the usable nonzero covariance subspace should
instead be checked using the measured `C_{bb'}[m]`.

Implementation:

- `weak_decoder/os_lora/experiments/analyze_empirical_noise_covariance.py`
- off-packet windows are selected outside the header-first packet intervals;
- each window has length `NR`, is dechirped into `z[n]`, and is projected using
  the LoRa-corrected pattern kernel already used by `pattern_bin_values`;
- the script reports raw noise correlation, empirical pattern covariance,
  range residuals for the ideal branch responses `Y_{b,sig}[m]`, and

```math
E_m
=
\frac{
\widehat{\sigma}_{\mathrm{off}}^2
\sum_b\sum_{b'}
Y_{b,\mathrm{sig}}^*[m]\,
\widehat C_{bb'}^+[m]\,
Y_{b',\mathrm{sig}}[m]
}{N}.
```

To avoid trusting sample-starved near-null directions, it also reports a
shrunk covariance:

```math
\widehat C_{bb',\lambda}[m]
=
(1-\lambda)\widehat C_{bb'}[m]
+
\lambda\widehat{\sigma}_{\mathrm{off}}^2
C_{bb'}^{\mathrm{white}}[m] .
```

Command:

```powershell
python "weakPacket_decoding\weak_decoder\os_lora\experiments\analyze_empirical_noise_covariance.py" `
  --datasets 0_0_0_10_14_8 0_0_0_10_14_16 0_0_0_10_14_32 `
  --raw-bins 61 301 653 900 `
  --bank-kinds fixed basic_only random_only balanced_random_only `
  --random-count 64 `
  --max-patterns 64 `
  --max-windows 512 `
  --output-dir "weakPacket_decoding\data\tmp_empirical_noise_cov_<dataset>_512"
```

Raw off-packet noise:

| Dataset | Windows | Raw Power | Offset Var CV | Mean Lag Corr | Max Lag Corr |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0_0_0_10_14_8` | 512 | 1.532e-6 | 0.0064 | 0.252 | 0.755 |
| `0_0_0_10_14_16` | 512 | 5.051e-6 | 0.0015 | 0.110 | 0.655 |
| `0_0_0_10_14_32` | 512 | 8.643e-6 | 0.0015 | 0.091 | 0.653 |

The offset variances are almost equal, but the temporal correlation is large.
This is not iid white sample noise.

Aggregated pattern-covariance readout over three datasets and four candidate
bins:

| Bank | Cases | Raw-Power Median `E_m` | Raw-Power Max `E_m` | Raw `E_m>4` | Shrink `lambda=0.2` Median | Shrink `lambda=0.2` Max | Shrink `E_m>4` | White-Model Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed` | 12 | 11.110 | 64.703 | 10 | 8.122 | 14.494 | 10 | 4.000 |
| `basic_only` | 12 | 13.704 | 50.452 | 10 | 8.659 | 16.180 | 10 | 4.000 |
| `random_only` | 12 | 18.783 | 49.806 | 11 | 10.599 | 18.035 | 10 | 3.822 |
| `balanced_random_only` | 12 | 20.342 | 47.501 | 10 | 10.256 | 18.092 | 10 | 3.828 |

Readout:

- The white model still gives the old result: `fixed` and structured banks sit
  at `4`, while the finite random-only banks are slightly below `4`.
- The empirical covariance does not look white.  Some candidate bins are much
  noisier than the raw average, while others are much quieter.
- With empirical colored covariance, `E_m` can exceed `4`, even after 20%
  shrinkage toward the AWGN covariance.  This is not extra independent replicas
  created by sampling patterns; it is frequency/color selectivity in the real
  background noise.
- Several non-uniform banks have large range residuals under strict empirical
  covariance, especially on `0_0_0_10_14_8`.  That means an unregularized
  `C^+` can overfit near-null empirical directions and should not be used as a
  decoder weight without diagonal loading or shrinkage.

Conclusion:

The earlier AWGN result remains correct as a theory statement:

```math
C_{bb'}[m]=C_{bb'}^{\mathrm{white}}[m]
\quad\Rightarrow\quad
E_m\le R .
```

But the actual off-packet IQ background is colored:

```math
R_\nu[n,n']
\ne
\sigma^2\delta[n-n'],
\qquad
C_{bb'}[m]\ne C_{bb'}^{\mathrm{white}}[m] .
```

Therefore the real-data route is no longer "more non-uniform patterns give more
independent coherent copies."  The defensible route is:

1. estimate `C_{bb'}[m]` or a stable shrinkage version of it from off-packet /
   neighboring-symbol data;
2. use a regularized GLS / adaptive matched filter in pattern space;
3. validate by SER, because empirical `E_m>4` may come from real colored-noise
   suppression or from finite-sample covariance overfitting.
