# v3 - Current Phase-Assisted PHY Selector

This folder labels the current research direction: phase-assisted weak LoRa
payload demodulation at the physical-symbol level.

## Scope

```text
Top-L FFT-bin candidates per payload symbol
energy and multi-offset coherence as primary evidence
packet-local smooth circular phase trajectory as auxiliary prior
CRC only as final validation, not search feedback
```

## Main implementation files

```text
../symbol_phase_two_stage.py
../candidate_pruning.py
../phase_guided_demod.py
```

## Current status

Active research line. The clean problem statement is:

```text
Top-L candidate sequence selection under an unknown smooth circular phase
trajectory prior.
```

Do not force preamble/sync/SFD and payload onto a single absolute phase line.
Use preamble/sync/SFD for timing and sync quality. Use header/payload,
especially payload symbols, to estimate the packet-local data-section
trajectory.

## Import aliases

```python
from weak_decoder.v3 import symbol_phase_two_stage
from weak_decoder.v3 import candidate_pruning
from weak_decoder.v3 import phase_guided_demod
```
