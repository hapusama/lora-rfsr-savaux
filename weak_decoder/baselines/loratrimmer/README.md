# LoRaTrimmer Baseline

This package contains a paper-style implementation of the core LoRaTrimmer
symbol metric:

> Jialuo Du, Yunhao Liu, Yidong Ren, Li Liu, and Zhichao Cao.
> "LoRaTrimmer: Optimal Energy Condensation with Chirp Trimming for
> LoRa Weak Signal Decoding." ACM MobiCom 2024.

The implementation follows Sec. 3.1 of the paper and the authors' public
prototype at <https://github.com/LoRaTrimmer/LoRaTrimmer>. The paper PDF used
for this implementation is <https://cse.msu.edu/~caozc/papers/mobicom24-du.pdf>.
For every candidate raw FFT bin `k`, it trims the symbol at the expected chirp
wrap time:

```text
t_k = (1 - k / 2**SF) * T
```

Then it computes the non-coherent decision metric:

```text
score[k] = |X1_k|**2 + |X2_k|**2
```

where `X1_k` is the projection of the pre-wrap segment and `X2_k` is the
projection of the post-wrap segment. This bypasses the unknown phase jump
between the two chirp pieces.

## Files

```text
paper_loratrimmer_demod.py  Core matrix metric and one-symbol demod API
__init__.py                 Public exports
```

## Minimal Usage

```python
from weak_decoder.baselines.loratrimmer import demod_loratrimmer_symbol

result = demod_loratrimmer_symbol(
    samples=iq,
    start_sample=symbol_start,
    sf=10,
    os_factor=8,
    ldro=False,
)

print(result.raw_fft_bin, result.symbol_value, result.peak_margin_db)
```

For synchronized packet experiments, `cfo_correction_mode="symbol"` or
`"continuous"` can be used to apply the same optional CFO pre-compensation
style as the other local baselines before computing the unchanged LoRaTrimmer
metric. The pure paper assumption is `cfo_correction_mode="none"`.

This baseline intentionally stops at per-symbol raw-bin selection. It does not
use phase-line selection, offset coherence, payload priors, or CRC-guided
search.
