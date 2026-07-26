# RFSuperResolution provenance

- Upstream repository: `https://github.com/AndreasKuster/RFSuperResolution`
- Upstream base commit: `00c135947f855790f458fdc25ae9533c70d77849`
- Vendored on: 2026-07-26
- Public checkpoints: copied from the upstream tracked files

This directory captures the working source used during the Windows
reproduction. Relative to the base commit, the source checkout contained local
changes in:

- `example/example.py`
- `rfsr/PHY.py`
- `rfsr/interp.py`
- `rfsr/nn/dataset.py`
- `rfsr/nn/nn.py`

Those working-tree changes were included deliberately so the consolidated
repository represents the tested state rather than silently losing local
reproduction fixes and documentation. The untracked Windows smoke-training
checkpoint and its loss history were not copied.

Future upstream updates should be imported as an explicit commit and recorded
here. Do not replace this directory without reviewing the upstream license and
the `rfsr/PHY.py` exception described in the upstream README.
