# v1 - Legacy Phase/Codec Beam Branch

This folder labels the early weak-packet decoder line. It does not contain a
separate copy of the code; the Python files here are import aliases to the
top-level implementation modules.

## Scope

```text
phase-guided residual scoring
blind payload search
codec/byte candidate enumeration
two-stage codec/CRC beam diagnostics
```

## Main implementation files

```text
../phase_guided_demod.py
../two_stage_weak_decoder.py
../blind_payload_search.py
../blind_payload_decoder.py
```

## Current status

Historical diagnostic branch. Keep it for ablation and failure analysis, but do
not use it as the paper mainline for the current PHY-only phase-assisted
demodulation story.

## Import aliases

```python
from weak_decoder.v1 import phase_guided_demod
from weak_decoder.v1 import two_stage_weak_decoder
```
