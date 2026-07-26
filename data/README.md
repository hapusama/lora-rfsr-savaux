# Dataset layout

IQ data is intentionally separated from Git history. The repository stores
code, manifests and small metadata; local disks, object storage or a mounted
cloud data volume store samples.

## Directories

- `raw/`: immutable captures, normally GNU Radio `complex64` (`<c8`) files.
- `processed/`: synchronized frames, paired windows, cached tensors and other
  reproducible intermediates.
- `results/`: large CSV files, plots and evaluation output.
- `manifests/`: small CSV/JSON metadata that may be committed.

Use paths relative to the repository in manifests. On another machine, either
restore the same layout or mount/link its data volume at `data/raw`.

## Capture groups

Keep receiver gain, bandwidth and all other receiver settings fixed while
collecting a comparison group:

```text
branch4_fixed/high_snr/
branch4_fixed/low_snr/
branch4_fixed/noise_only/
branch4_fixed/interference/
```

For RF-SR evaluation, record whether low-rate and high-rate IQ are:

1. simultaneous captures from a shared event;
2. deterministically downsampled from the same native high-rate capture; or
3. unrelated captures with only statistical comparability.

Only the first two support sample- or symbol-paired supervised evaluation.

## Integrity and upload

Create a SHA-256 before transfer:

```powershell
Get-FileHash .\data\raw\capture.bin -Algorithm SHA256
```

Linux:

```bash
sha256sum data/raw/capture.bin
rsync -avP data/raw/ user@server:/workspace/lora-rfsr-savaux/data/raw/
```

After upload, compute the hash again. Keep the raw file immutable and generate
derived artifacts under `processed/`.

See `manifests/example.csv` for the minimum recommended fields.
