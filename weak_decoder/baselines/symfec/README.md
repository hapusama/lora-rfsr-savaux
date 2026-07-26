# Sym-FEC Baseline

This package contains a local, paper-inspired implementation of:

> Weiwei Chen, Xianjin Xia, Shuai Wang, Xianjun Deng, Jiehong Wu, and
> Caishi Huang. "Sym-FEC: Enhancing Error Correction in LoRa PHY With a
> Symbol-Level FEC Decoder." IEEE Transactions on Mobile Computing, 2026.
> DOI: `10.1109/TMC.2025.3608893`.

Public paper pages describe Sym-FEC as a symbol-level FEC decoder for LoRa PHY.
Its key idea is signal copy retrieval: spectrum evidence from received symbols
is linked with the coding correlations inside a LoRa coding block. The reported
hardware gain is about 2.3 dB to 3 dB over the traditional LoRa decoder, with no
transmitter-side modification.

## Local Scope

The full IEEE PDF was not accessible from this environment, so this is not a
claim of bit-for-bit reproduction of the authors' private equations. It is an
audit-friendly baseline that implements the public algorithm shape:

1. Convert every received symbol FFT spectrum into likelihoods over
   gr-lora_sdr demod symbol values.
2. Convert each symbol's likelihoods into bit/copy likelihoods using the LoRa
   diagonal deinterleaver relation.
3. Select valid Hamming codewords per coding block.
4. Refine codeword choices with exact reconstructed block-symbol spectrum
   scores.
5. Reconstruct payload symbols and run the existing local payload/CRC codec only
   after symbol selection.

It intentionally does not use payload templates, counters, cross-packet priors,
or CRC-guided symbol selection.

## Files

```text
paper_symfec_decoder.py   Core symbol-level FEC baseline
run_symfec_baseline.py    Runner for existing IQ + header-first symbol CSV
__init__.py               Public exports
```

## Minimal Usage

```python
from weak_decoder.baselines.symfec import (
    SymFECConfig,
    decode_symfec_payload_from_spectra,
)

result = decode_symfec_payload_from_spectra(
    spectra_or_powers=payload_spectra,
    sf=10,
    cr=1,
    ldro=False,
    config=SymFECConfig(),
    header_symbol_values=header_symbols,
)

print(result.selected_raw_fft_bins, result.crc_valid)
```

## Current Input Structure Runner

The runner consumes the same synchronized inputs as the local baseline scripts:

```powershell
python "gr-lora_sdr\weakPacket_decoding\weak_decoder\baselines\symfec\run_symfec_baseline.py" `
    -i "<capture>.bin" `
    -s "<header_first_symbols.csv>" `
    -o "<output_packets.csv>" `
    --summary-json "<output_summary.json>" `
    --evidence-mode center
```

Use `--evidence-mode multi-offset` when the IQ file is oversampled and you want
to fuse all oversampling phases before the Sym-FEC-style block decoder.
