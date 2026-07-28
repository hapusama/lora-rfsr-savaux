"""Manifest-backed OTA dataset for the local RF-SR experiment layout."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATASET_ROOT = (
    REPOSITORY_ROOT / "data" / "reference_phy" / "rfsr_db"
)


def _resolve_relative(root: Path, value: str, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{label} must be relative to {root}, got {value!r}.")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} escapes the dataset root: {value!r}."
        ) from exc
    return resolved


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _split_groups(
    rows: list[dict[str, object]],
    *,
    training: bool,
    test_split: float,
    seed: int,
) -> list[dict[str, object]]:
    if not 0.0 <= test_split < 1.0:
        raise ValueError(f"test_split must be in [0, 1), got {test_split}.")
    groups = sorted({str(row["split_group"]) for row in rows})
    if not groups:
        return []

    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    test_count = int(round(len(shuffled) * test_split))
    if test_split > 0.0 and len(shuffled) > 1:
        test_count = min(len(shuffled) - 1, max(1, test_count))
    test_groups = set(shuffled[:test_count])
    selected_groups = (
        set(groups) - test_groups if training else test_groups
    )
    return [
        row
        for row in rows
        if str(row["split_group"]) in selected_groups
    ]


class OTALoRaDataset(Dataset):
    """Load aligned 250 kS/s input and 1 MS/s reference pairs.

    The trim pipeline stores two 1 MS/s ADC phases per physical packet.
    ``views.csv`` exposes four 250 kS/s polyphase views of each stored file.
    All manifest paths are relative to ``dataset_root`` and every physical
    packet stays wholly in either the train or test split.
    """

    def __init__(
        self,
        oversampling: int = 4,
        snr_range: tuple[float, float] = (-20.0, 20.0),
        training: bool = True,
        trim: bool = False,
        return_snr: bool = False,
        downsampling: int = 8,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        test_split: float = 0.2,
        split_seed: int = 42,
        missing_snr_db: float = 0.0,
    ):
        self.return_snr = bool(return_snr)
        self.training = bool(training)
        self.OSF = int(oversampling)
        self.DSF = int(downsampling)
        self.dataset_root = Path(dataset_root).expanduser().resolve()

        if trim:
            raise ValueError(
                "The manifest dataset is already fulltrim-aligned; the legacy "
                "model3 trim mode is not supported."
            )
        if self.OSF != 4 or self.DSF != 8:
            raise ValueError(
                "The current OTA manifest contract is fixed at DSF=8 and "
                "OSF=4 (250 kS/s input -> 1 MS/s label)."
            )

        manifest_path = self.dataset_root / "manifests" / "views.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing OTA views manifest: {manifest_path}. Run "
                "tools/build_rfsr_ota_dataset.py trim first."
            )

        minimum_snr, maximum_snr = map(float, snr_range)
        if minimum_snr > maximum_snr:
            raise ValueError("snr_range minimum must not exceed maximum.")

        records: list[dict[str, object]] = []
        for raw in _read_manifest(manifest_path):
            ota_path = _resolve_relative(
                self.dataset_root, raw["ota_path"], label="ota_path"
            )
            reference_path = _resolve_relative(
                self.dataset_root,
                raw["reference_path"],
                label="reference_path",
            )
            if not ota_path.is_file():
                raise FileNotFoundError(f"missing OTA IQ file: {ota_path}")
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"missing reference IQ file: {reference_path}"
                )

            metadata_path = (
                self.dataset_root / "metadata" / f"{ota_path.stem}.json"
            )
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"missing OTA metadata file: {metadata_path}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema") != "lora-rfsr-ota-view":
                raise ValueError(
                    f"unsupported OTA metadata schema: {metadata_path}"
                )

            raw_snr = metadata.get("alignment", {}).get(
                "grlora_snr_db", ""
            )
            try:
                snr_db = float(raw_snr)
            except (TypeError, ValueError):
                snr_db = float(missing_snr_db)
            if not math.isfinite(snr_db):
                snr_db = float(missing_snr_db)
            if not minimum_snr <= snr_db <= maximum_snr:
                continue

            stored_samples = int(raw["target_samples"])
            input_samples = int(raw["input_samples"])
            if ota_path.stat().st_size != stored_samples * 8:
                raise ValueError(
                    f"OTA size does not match views.csv: {ota_path}"
                )
            if reference_path.stat().st_size != stored_samples * 8:
                raise ValueError(
                    "reference size does not match views.csv: "
                    f"{reference_path}"
                )
            if stored_samples != self.OSF * input_samples:
                raise ValueError(
                    f"manifest length ratio is not OSF={self.OSF}: "
                    f"{raw['view_id']}"
                )

            records.append(
                {
                    "view_id": raw["view_id"],
                    "split_group": raw["split_group"],
                    "ota_path": ota_path,
                    "reference_path": reference_path,
                    "lowrate_phase": int(raw["lowrate_phase_1m"]),
                    "stored_samples": stored_samples,
                    "input_samples": input_samples,
                    "snr_db": snr_db,
                }
            )

        self.records = _split_groups(
            records,
            training=self.training,
            test_split=float(test_split),
            seed=int(split_seed),
        )
        self.size = len(self.records)
        if not self.records:
            split_name = "training" if self.training else "test"
            raise ValueError(
                f"OTA {split_name} split is empty under {self.dataset_root}."
            )
        print(
            f"OTALoRaDataset contains {self.size} manifest views "
            f"({'train' if self.training else 'test'} split)."
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        record = self.records[index]
        ota = np.memmap(
            record["ota_path"], dtype=np.dtype("<c8"), mode="r"
        )
        reference = np.memmap(
            record["reference_path"], dtype=np.dtype("<c8"), mode="r"
        )
        phase = int(record["lowrate_phase"])
        signal = np.asarray(ota[phase::4], dtype=np.complex64)
        label = np.asarray(reference, dtype=np.complex64)
        if signal.size != int(record["input_samples"]):
            raise ValueError(
                f"unexpected input length for {record['view_id']}: "
                f"{signal.size}"
            )
        if label.size != self.OSF * signal.size:
            raise ValueError(
                f"input/reference length mismatch for {record['view_id']}."
            )

        x = torch.from_numpy(
            np.stack((signal.real, signal.imag), axis=0).copy()
        )
        y = torch.from_numpy(
            np.stack((label.real, label.imag), axis=0).copy()
        )
        if self.return_snr:
            snr = torch.tensor(
                [float(record["snr_db"])], dtype=torch.float32
            )
            return x, y, snr
        return x, y
