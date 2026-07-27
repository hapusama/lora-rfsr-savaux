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

## Ideal PHY references

Generate ideal references from the complete 33-byte STM32 `[TX Frame]`
records, not from the 20-byte application payload:

```powershell
python tools/generate_reference_phy.py `
  --uart-log data/raw/packet_reference.txt `
  --output-root D:\rfsr_db `
  --limit 2
```

The smoke-test command writes:

```text
rfsr_db/
├── reference/
│   ├── signalout_000000_fulltrim.cfile
│   └── signalout_000001_fulltrim.cfile
└── metadata/
    ├── 000000.json
    └── 000001.json
```

Inspect and align those first two references against high-SNR OTA IQ before
running the command without `--limit`. The generator emits 1 MS/s little-endian
`complex64` by default, adds the standard explicit PHY header and the
SX1276-air CRC convention already validated by this workspace's receiver
(`crc_mode=grlora`), and deliberately does not add the upstream RF-SR private
four-byte application header, artificial CFO, AWGN, or channel effects. It
prepends 10,000 zero samples and no trailing zeros to match RF-SR
`PHY.encode()`; pass `--leading-silence-samples 0` only when a packet-only
waveform is required. The alternative full-payload CRC-16 comparison mode is
not used for this STM32 dataset.

Plot every preamble/sync/SFD/header/payload symbol as an independent STFT
panel:

```powershell
python tools/plot_reference_phy_stft.py `
  --input D:\rfsr_db\reference\signalout_000000_fulltrim.cfile
```

Use repeated `--section` options for a smaller plot set, for example
`--section header --section payload`. The script reads the paired metadata
automatically and writes paginated PNG files under `D:\rfsr_db\stft\`.
