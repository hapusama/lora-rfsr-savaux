"""Direct loader for the public RF-SR OTA archive.

The public archive is already packetized at 2 MS/s.  For the strict RF-SR
task one item is constructed from a single physical capture as

``OTA 250 kS/s -> the same OTA capture at 1 MS/s``.

This keeps receiver gain, CFO/SFO, channel response, and noise in one
coordinate system and makes hard preservation of the observed polyphase
samples well-defined.  The paired ideal reference remains available as an
explicit diagnostic target and supplies payload/symbol truth, but it is not
silently mixed into the strict task.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from rfsr.PHY import (
    lora_chirp,
    lora_header,
    lora_header_init,
    lora_payload,
    lora_payload_init,
)


SF = 12
N_BINS = 1 << SF
BW_HZ = 125_000
HIGH_RATE_HZ = 2_000_000
OUTPUT_RATE_HZ = 1_000_000
LOW_RATE_HZ = 250_000
HIGH_OS_FACTOR = HIGH_RATE_HZ // BW_HZ
OUTPUT_OS_FACTOR = OUTPUT_RATE_HZ // BW_HZ
SUPER_RESOLUTION_FACTOR = OUTPUT_RATE_HZ // LOW_RATE_HZ
HIGH_SAMPLES_PER_SYMBOL = N_BINS * HIGH_OS_FACTOR
OUTPUT_SAMPLES_PER_SYMBOL = N_BINS * OUTPUT_OS_FACTOR
BRANCH_FFT_BINS = OUTPUT_SAMPLES_PER_SYMBOL // SUPER_RESOLUTION_FACTOR
OFFICIAL_GUARD_HIGH = 500_000
OFFICIAL_RISE_HIGH = 100
OFFICIAL_PREAMBLE_SYMBOLS = 8
OFFICIAL_DATA_SYMBOL_OFFSET = 12.25
OFFICIAL_PAYLOAD_BYTES = 16


@dataclass(frozen=True)
class OfficialOTARecord:
    """One downloaded physical OTA capture and its paired public truth."""

    reference_id: int
    ota_path: Path
    reference_path: Path
    metadata_path: Path
    snr_db: float
    raw_symbol_k: tuple[int, ...]
    expected_high_samples: int


def _raw_symbol_k(metadata: dict[str, object]) -> tuple[int, ...]:
    """Reproduce the legacy transmitter's header and payload chirp values."""

    payload = np.asarray(metadata["payload"], dtype=np.uint8)
    if payload.shape != (OFFICIAL_PAYLOAD_BYTES,):
        raise ValueError(
            "official payload must contain exactly "
            f"{OFFICIAL_PAYLOAD_BYTES} bytes"
        )
    length = np.uint16(4 + payload.size)
    _, header_payload_bits = lora_header_init(SF, 0)
    payload_bits, payload_symbol_count = lora_payload_init(
        SF,
        length,
        1,
        4,
        header_payload_bits,
        int(metadata["dst"]),
        int(metadata["src"]),
        int(metadata["seqn"]),
        payload,
    )
    header, payload_offset = lora_header(
        SF, length, 4, 1, payload_bits, 0
    )
    body = lora_payload(
        SF, 4, payload_symbol_count, payload_bits, payload_offset
    )
    return tuple(int(round(float(value))) for value in (*header, *body))


def deterministic_reference_splits(
    reference_ids: Iterable[int],
    *,
    seed: int = 42,
    max_reference_ids: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Make a deterministic 60/20/20 split by reference/payload identity.

    A public reference ID is reused by many experiments and receiver gains.
    Splitting OTA filenames would therefore leak the same transmitted packet
    into train and test.  The reference ID is the smallest safe group.
    """

    values = sorted({int(value) for value in reference_ids})
    if max_reference_ids is not None:
        limit = int(max_reference_ids)
        if limit < 3:
            raise ValueError("max_reference_ids must be at least 3")
    else:
        limit = None
    shuffled = list(values)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    if limit is not None:
        shuffled = shuffled[:limit]
    if len(shuffled) < 3:
        raise ValueError("official OTA splitting requires at least 3 references")

    validation_count = max(1, int(round(len(shuffled) * 0.2)))
    test_count = max(1, int(round(len(shuffled) * 0.2)))
    while validation_count + test_count >= len(shuffled):
        if validation_count >= test_count:
            validation_count -= 1
        else:
            test_count -= 1
    return {
        "test": tuple(sorted(shuffled[:test_count])),
        "validation": tuple(
            sorted(shuffled[test_count : test_count + validation_count])
        ),
        "train": tuple(sorted(shuffled[test_count + validation_count :])),
    }


def _validate_metadata(metadata: dict[str, object], path: Path) -> None:
    expected = {
        "sf": SF,
        "cr": 4,
        "enable_crc": 1,
        "implicit_header": 0,
        "preamble_bits": OFFICIAL_PREAMBLE_SYMBOLS,
    }
    for field, value in expected.items():
        if int(metadata[field]) != int(value):
            raise ValueError(f"unexpected {field} in {path}: {metadata[field]}")
    if int(round(float(metadata["bw"]))) != BW_HZ:
        raise ValueError(f"unexpected bandwidth in {path}")
    if int(round(float(metadata["sample_rate"]))) != HIGH_RATE_HZ:
        raise ValueError(f"unexpected sample rate in {path}")


def scan_official_ota_records(
    root: str | Path,
    *,
    split: str,
    split_seed: int = 42,
    max_reference_ids: int | None = None,
    snr_range: tuple[float, float] = (-35.0, 15.0),
) -> tuple[list[OfficialOTARecord], dict[str, tuple[int, ...]]]:
    """Index downloaded files without requiring a generated CSV manifest."""

    dataset_root = Path(root).expanduser().resolve()
    metadata_paths = sorted((dataset_root / "metadata").glob("*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"no metadata/*.json under {dataset_root}")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")

    metadata_rows: list[tuple[int, Path, dict[str, object]]] = []
    for path in metadata_paths:
        try:
            reference_id = int(path.stem)
        except ValueError as exc:
            raise ValueError(f"metadata filename is not a reference ID: {path}") from exc
        metadata = json.loads(path.read_text(encoding="utf-8"))
        _validate_metadata(metadata, path)
        metadata_rows.append((reference_id, path, metadata))

    splits = deterministic_reference_splits(
        (row[0] for row in metadata_rows),
        seed=int(split_seed),
        max_reference_ids=max_reference_ids,
    )
    selected = set(splits[split])
    minimum_snr, maximum_snr = map(float, snr_range)
    if minimum_snr > maximum_snr:
        raise ValueError("snr_range minimum must not exceed maximum")

    records: list[OfficialOTARecord] = []
    for reference_id, metadata_path, metadata in metadata_rows:
        if reference_id not in selected:
            continue
        reference_path = (
            dataset_root
            / "reference"
            / f"signalout_{reference_id:06d}_fulltrim.cfile"
        )
        expected_samples = (
            int(metadata["num_samples"]) + 2 * OFFICIAL_GUARD_HIGH
        )
        expected_bytes = expected_samples * np.dtype("<c8").itemsize
        if not reference_path.is_file():
            raise FileNotFoundError(f"missing official reference: {reference_path}")
        if reference_path.stat().st_size != expected_bytes:
            raise ValueError(f"official reference size mismatch: {reference_path}")
        symbols = _raw_symbol_k(metadata)

        # Some public metadata lists the same path twice.  Deduplicate it,
        # but reject contradictory SNR annotations.
        files: dict[str, float] = {}
        for relative, raw_snr in metadata.get("files", []):
            relative = str(relative)
            snr_db = float(raw_snr)
            if relative in files and not math.isclose(
                files[relative], snr_db, rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError(f"conflicting SNR values for {relative}")
            files[relative] = snr_db
        for relative, snr_db in sorted(files.items()):
            ota_path = (dataset_root / relative).resolve()
            try:
                ota_path.relative_to(dataset_root)
            except ValueError as exc:
                raise ValueError(f"OTA path escapes dataset root: {relative}") from exc
            if not ota_path.is_file() or not minimum_snr <= snr_db <= maximum_snr:
                continue
            if ota_path.stat().st_size != expected_bytes:
                raise ValueError(f"official OTA size mismatch: {ota_path}")
            records.append(
                OfficialOTARecord(
                    reference_id=reference_id,
                    ota_path=ota_path,
                    reference_path=reference_path,
                    metadata_path=metadata_path,
                    snr_db=float(snr_db),
                    raw_symbol_k=symbols,
                    expected_high_samples=expected_samples,
                )
            )
    if not records:
        raise ValueError(
            f"official OTA {split} split has no downloaded captures in SNR range"
        )
    return records, splits


def _signed_legacy_k(value: int) -> int:
    return int((int(value) + N_BINS // 2) % N_BINS - N_BINS // 2)


def _four_branch_power_numpy(symbol: np.ndarray, downchirp: np.ndarray) -> np.ndarray:
    values = np.asarray(symbol, dtype=np.complex64)
    if values.shape != (OUTPUT_SAMPLES_PER_SYMBOL,):
        raise ValueError("one output-rate LoRa symbol is required")
    dechirped = values * downchirp
    branches = dechirped.reshape(BRANCH_FFT_BINS, SUPER_RESOLUTION_FACTOR)
    spectra = np.fft.fft(branches, axis=0)
    return np.sum(np.abs(spectra) ** 2, axis=1, dtype=np.float64)


class OfficialOTASymbolDataset(Dataset):
    """Return several known-symbol windows from each physical OTA capture.

    ``x`` has shape ``[S, 2, 8192]`` and ``y`` has shape
    ``[S, 2, 32768]``.  Every row obeys ``x == y[..., ::4]`` in the default
    received-target mode.  ``correct_bins`` are the public symbol truth shifted
    by a capture-specific CFO estimate from all eight preamble chirps.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        split_seed: int = 42,
        max_reference_ids: int | None = None,
        snr_range: tuple[float, float] = (-35.0, 15.0),
        symbols_per_capture: int = 4,
        target_source: str = "received",
        task_min_snr_db: float = -20.0,
    ):
        self.root = Path(root).expanduser().resolve()
        self.split = str(split)
        self.split_seed = int(split_seed)
        self.symbols_per_capture = int(symbols_per_capture)
        self.target_source = str(target_source)
        self.task_min_snr_db = float(task_min_snr_db)
        if not 1 <= self.symbols_per_capture <= 8:
            raise ValueError("symbols_per_capture must be in [1, 8]")
        if self.target_source not in {"received", "reference"}:
            raise ValueError("target_source must be received or reference")
        self.records, self.reference_splits = scan_official_ota_records(
            self.root,
            split=self.split,
            split_seed=self.split_seed,
            max_reference_ids=max_reference_ids,
            snr_range=snr_range,
        )
        self.epoch = 0
        upchirp, _ = lora_chirp(
            +1, 0, BW_HZ, N_BINS, OUTPUT_OS_FACTOR, 0, 0
        )
        self._downchirp = np.conjugate(upchirp).astype(np.complex64)
        self._cfo_cache: dict[Path, int] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _source_path(self, record: OfficialOTARecord) -> Path:
        return (
            record.ota_path
            if self.target_source == "received"
            else record.reference_path
        )

    def _estimate_cfo_bin(self, record: OfficialOTARecord) -> int:
        source_path = self._source_path(record)
        cached = self._cfo_cache.get(source_path)
        if cached is not None:
            return cached
        source = np.memmap(source_path, dtype=np.dtype("<c8"), mode="r")
        start = OFFICIAL_GUARD_HIGH + OFFICIAL_RISE_HIGH
        power = np.zeros(BRANCH_FFT_BINS, dtype=np.float64)
        for symbol_index in range(OFFICIAL_PREAMBLE_SYMBOLS):
            high_start = start + symbol_index * HIGH_SAMPLES_PER_SYMBOL
            symbol = np.asarray(
                source[
                    high_start : high_start + HIGH_SAMPLES_PER_SYMBOL : 2
                ],
                dtype=np.complex64,
            )
            power += _four_branch_power_numpy(symbol, self._downchirp)
        result = int(np.argmax(power))
        self._cfo_cache[source_path] = result
        return result

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[int(index)]
        # Epoch-dependent selection gives new symbols and ADC/polyphase views
        # without pretending they are independent physical packets.
        sequence = np.random.SeedSequence(
            [self.split_seed, self.epoch, int(index), record.reference_id]
        )
        rng = np.random.default_rng(sequence)
        symbol_count = len(record.raw_symbol_k)
        selected = rng.choice(
            symbol_count, size=self.symbols_per_capture, replace=False
        )
        phases = rng.integers(0, 8, size=self.symbols_per_capture)
        source = np.memmap(
            self._source_path(record), dtype=np.dtype("<c8"), mode="r"
        )
        cfo_bin = self._estimate_cfo_bin(record)
        data_start = int(
            OFFICIAL_GUARD_HIGH
            + OFFICIAL_RISE_HIGH
            + OFFICIAL_DATA_SYMBOL_OFFSET * HIGH_SAMPLES_PER_SYMBOL
        )

        x_rows: list[torch.Tensor] = []
        y_rows: list[torch.Tensor] = []
        correct_bins: list[int] = []
        for symbol_index, phase in zip(selected.tolist(), phases.tolist()):
            high_start = (
                data_start
                + int(symbol_index) * HIGH_SAMPLES_PER_SYMBOL
                + int(phase)
            )
            y_complex = np.asarray(
                source[
                    high_start : high_start + HIGH_SAMPLES_PER_SYMBOL : 2
                ],
                dtype=np.complex64,
            ).copy()
            if y_complex.size != OUTPUT_SAMPLES_PER_SYMBOL:
                raise ValueError(f"truncated symbol in {record.ota_path}")
            x_complex = np.asarray(
                y_complex[::SUPER_RESOLUTION_FACTOR], dtype=np.complex64
            ).copy()
            x_rows.append(
                torch.from_numpy(
                    np.stack((x_complex.real, x_complex.imag), axis=0)
                )
            )
            y_rows.append(
                torch.from_numpy(
                    np.stack((y_complex.real, y_complex.imag), axis=0)
                )
            )
            frequency_offset = -_signed_legacy_k(
                record.raw_symbol_k[int(symbol_index)]
            )
            correct_bins.append(
                int((cfo_bin + frequency_offset) % BRANCH_FFT_BINS)
            )

        # Phase zero is an architectural invariant, not a learnable output.
        # Excluding it keeps the waveform-loss scale identical to synthetic
        # pretraining and avoids counting a constant epsilon-only term.
        valid_mask = torch.ones(
            self.symbols_per_capture,
            OUTPUT_SAMPLES_PER_SYMBOL,
            dtype=torch.float32,
        )
        valid_mask[:, ::SUPER_RESOLUTION_FACTOR] = 0.0
        return {
            "x": torch.stack(x_rows),
            "y": torch.stack(y_rows),
            "correct_bins": torch.tensor(correct_bins, dtype=torch.long),
            "task_mask": torch.full(
                (self.symbols_per_capture,),
                float(record.snr_db >= self.task_min_snr_db),
                dtype=torch.float32,
            ),
            "valid_mask": valid_mask,
            "snr_db": torch.tensor(record.snr_db, dtype=torch.float32),
            "reference_id": torch.tensor(record.reference_id, dtype=torch.long),
            "ota_path": str(record.ota_path),
        }


__all__ = [
    "BRANCH_FFT_BINS",
    "OfficialOTARecord",
    "OfficialOTASymbolDataset",
    "OUTPUT_SAMPLES_PER_SYMBOL",
    "SUPER_RESOLUTION_FACTOR",
    "deterministic_reference_splits",
    "scan_official_ota_records",
]
