# LiteNap-Savaux fine-SNR error-mode analysis

## Question

For a downsampling factor `D`, a failed full-bin decision has two mutually
exclusive causes:

1. `wrong_group`: the predicted bin is correct modulo `N/D`, but the selected
   full-frequency alias group is wrong.
2. `wrong_alias_bin`: even the predicted modulo-`N/D` bin is wrong.

The first cause can in principle be repaired by a perfect LiteNap fingerprint.
The second cannot, because the correct full-frequency bin is not in the
selected alias family.

## Protocol

- SF10, `N=1024`, OSR `R=4`, downsampling factor `D=4`
- 17 clean-synchronized packets and 833 payload symbols
- added-SNR grid from `-16` to `-28 dB` in `1 dB` steps
- seeds 42, 43, and 44, giving 2,499 decisions per noisy condition
- clean timing/CFO/SFO and explicit-header residual-bin calibration
- phase reranking disabled to isolate spectral alias and group failures

For every method and symbol:

```text
pred == GT
    -> correct
pred != GT and pred mod 256 == GT mod 256
    -> wrong_group
otherwise
    -> wrong_alias_bin
```

## Selected results

`Alias-error share` is the fraction of all method errors caused by
`wrong_alias_bin`. `Oracle floor` is the SER remaining if an ideal group
fingerprint repairs every `wrong_group` decision.

| Added SNR | K1 SER | K1 alias-error share | K1 oracle floor | K2 SER | K2 alias-error share | K2 oracle floor |
|---:|---:|---:|---:|---:|---:|---:|
| -16 dB | 2.64% | 74.2% | 1.96% | 0.00% | n/a | 0.00% |
| -18 dB | 13.37% | 82.3% | 11.00% | 0.56% | 85.7% | 0.48% |
| -20 dB | 36.73% | 89.9% | 33.01% | 5.52% | 94.2% | 5.20% |
| -22 dB | 62.71% | 93.7% | 58.78% | 20.97% | 97.9% | 20.53% |
| -24 dB | 81.71% | 97.0% | 79.23% | 46.42% | 98.1% | 45.54% |
| -26 dB | 92.28% | 98.0% | 90.40% | 72.11% | 98.4% | 70.95% |
| -28 dB | 95.76% | 98.8% | 94.60% | 88.76% | 98.5% | 87.40% |

Across the complete fine-SNR grid:

| Method | Total errors | Wrong group | Wrong alias bin | Alias-error share |
|---|---:|---:|---:|---:|
| K1 | 18,144 | 767 | 17,377 | 95.77% |
| K2 | 10,617 | 200 | 10,417 | 98.12% |

Among the `wrong_alias_bin` decisions, 91.73% for K1 and 93.14% for K2 are
more than eight circular modulo-256 bins away from the correct alias bin.
These are broad noise-peak failures rather than a residual adjacent-bin
CFO/timing offset.

## Conclusion

Cause 2, an incorrect aliased bin, is the dominant failure mode at every
meaningful point on the waterfall. Group selection is already relatively
reliable when the aliased bin is correct, especially for K2.

Consequently, a perfect LiteNap group fingerprint has a small oracle gain and
cannot recover the observed low-SNR gap. The next detector improvement must
increase coarse alias-bin reliability, for example through more coherent
sample energy, repeated-symbol/packet combining, or codec-aware soft
likelihoods. Improving only the alias-group classifier does not address the
main failure mode.

## Reproduction

Run from `weakPacket_decoding`:

```powershell
python -B -m weak_decoder.os_lora.experiments.evaluate_litenap_savaux `
  --snrs -16 -17 -18 -19 -20 -21 -22 -23 -24 -25 -26 -27 -28 `
  --seeds 42 43 44 `
  --batch-size 64 `
  --fingerprint-weight 0 `
  --output-dir data\experiments\litenap_savaux_error_modes_fine_snr_20260724 `
  --verify-savaux-symbols 8
```

The complete per-SNR decomposition is in `ERROR_MODES.md`; `symbols.csv`
contains each hard decision, error mode, and signed modulo-256 bin offset.
