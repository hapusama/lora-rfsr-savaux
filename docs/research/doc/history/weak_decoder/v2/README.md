# v2 - Symbol-Level Two-Stage Phase Selector

This folder labels the first clean PHY-only symbol selector. It avoids
payload-byte enumeration and works directly on payload FFT-bin candidates.

## Scope

```text
multi-offset FFT evidence
high-confidence Top-1 locking
low-confidence Top-L candidate sets
packet-local phase-line consistency
normal LoRa PHY decode and CRC only after bin selection
```

## Main implementation files

```text
../candidate_pruning.py
../symbol_phase_two_stage.py
../phase_guided_demod.py
```

## Current status

Historical PHY-only baseline. This version established the right abstraction:
select one FFT bin per symbol first; do not let CRC, payload templates,
counters, or cross-packet priors participate in candidate selection.

## Import aliases

```python
from weak_decoder.v2 import symbol_phase_two_stage
from weak_decoder.v2 import candidate_pruning
```
