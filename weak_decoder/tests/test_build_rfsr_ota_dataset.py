from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tools import build_rfsr_ota_dataset as builder


def write_rows(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def descriptor(*, sample_rate_hz: int = 200) -> builder.CaptureDescriptor:
    return builder.CaptureDescriptor(
        experiment_id=3,
        session_id=2,
        location_id="lab1",
        condition="highsnr",
        run_id=1,
        sf=7,
        bandwidth_hz=25,
        sample_rate_hz=sample_rate_hz,
        preamble_symbols=8,
        sync_word=0x12,
        cr=1,
        phy_crc=True,
        center_frequency_hz=487_700_000,
        rx_gain_db=20,
        pay_len=1,
    )


def catalog_row(
    reference_id: int,
    frame: bytes,
    reference_path: Path,
    *,
    reference_samples: int = 16,
    leading_samples: int = 4,
    tx_period_ms: int = 6000,
) -> dict[str, object]:
    return {
        "reference_id": reference_id,
        "uart_seq": reference_id,
        "uart_round": 0,
        "payload_id": reference_id,
        "frame_hex": builder.spaced_hex(frame),
        "app_payload_hex": builder.spaced_hex(frame),
        "frame_bytes": len(frame),
        "sf": 7,
        "bandwidth_hz": 25,
        "cr": 1,
        "preamble_symbols": 8,
        "sync_word": 0x12,
        "phy_crc": 1,
        "ldro": 0,
        "sample_rate_hz": 100,
        "samples_per_symbol": 512,
        "leading_silence_samples": leading_samples,
        "trailing_silence_samples": 0,
        "reference_samples": reference_samples,
        "header_symbols": 1,
        "payload_symbols": 1,
        "tx_period_ms": tx_period_ms,
        "source_reference_path": str(reference_path),
        "dataset_reference_path": (
            f"reference/signalout_{reference_id:06d}_fulltrim.cfile"
        ),
        "reference_sha256": builder.sha256_file(reference_path),
        "reference_metadata_path": "",
        "uart_source_path": "",
        "uart_source_sha256": "",
    }


class CaptureNamingTest(unittest.TestCase):
    def test_canonical_capture_name_round_trips(self) -> None:
        expected = replace(
            descriptor(sample_rate_hz=2_000_000),
            pay_len=33,
        )
        name = expected.canonical_filename

        self.assertEqual(
            name,
            "rxcap_exp003_sess002_loclab1_condhighsnr_run001_"
            "sf7_bw25_fs2000000_pre8_sw12_cr45_crc1_fc487700000_rxg20.cfile",
        )
        parsed = builder.descriptor_from_mapping(
            builder.parse_capture_filename(name)
        )
        self.assertEqual(parsed, expected)

    def test_output_root_must_stay_inside_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_data = Path(temp_dir) / "data"
            fake_data.mkdir()
            with mock.patch.object(builder, "DATA_ROOT", fake_data.resolve()):
                self.assertEqual(
                    builder.ensure_inside_data(fake_data / "rfsr_db"),
                    (fake_data / "rfsr_db").resolve(),
                )
                with self.assertRaises(ValueError):
                    builder.ensure_inside_data(Path(temp_dir) / "outside")


class AssociationTest(unittest.TestCase):
    def test_crc_anchors_infer_invalid_middle_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            output_root = data_root / "reference_phy" / "rfsr_db"
            output_root.mkdir(parents=True)
            capture_path = root / "capture.cfile"
            np.zeros(10, dtype=np.complex64).tofile(capture_path)

            references = []
            rows = []
            for reference_id in range(3):
                reference = root / f"ref{reference_id}.cfile"
                np.zeros(16, dtype=np.complex64).tofile(reference)
                references.append(reference)
                rows.append(
                    catalog_row(
                        reference_id,
                        bytes([0x40, reference_id]),
                        reference,
                    )
                )
            catalog_path = output_root / "manifests" / "reference_catalog.csv"
            write_rows(catalog_path, builder.CATALOG_FIELDS, rows)

            detections_path = output_root / "detections.csv"
            detection_rows = [
                {
                    "detection_index": 0,
                    "start_sample_2m": 1_000,
                    "crc_valid": 1,
                    "decoded_frame_hex": "40 00",
                },
                {
                    "detection_index": 1,
                    "start_sample_2m": 2_205,
                    "crc_valid": 0,
                    "decoded_frame_hex": "",
                },
                {
                    "detection_index": 2,
                    "start_sample_2m": 3_400,
                    "crc_valid": 1,
                    "decoded_frame_hex": "40 02",
                },
            ]
            write_rows(detections_path, builder.DETECTION_FIELDS, detection_rows)

            with mock.patch.object(builder, "DATA_ROOT", data_root.resolve()):
                packets_path = builder.associate_packets(
                    capture_path,
                    descriptor(),
                    catalog_path,
                    detections_path,
                    output_root,
                    recover_missing=False,
                )
            packets = builder.read_csv(packets_path)
            middle = next(
                row for row in packets if row["detection_index"] == "1"
            )
            self.assertEqual(middle["reference_id"], "1")
            self.assertEqual(middle["association_method"], "neighbor_inferred_high")
            self.assertEqual(middle["status"], "accepted")


class TrimTest(unittest.TestCase):
    def test_trim_writes_two_1m_phases_and_eight_logical_views(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            output_root = data_root / "reference_phy" / "rfsr_db"
            output_root.mkdir(parents=True)

            reference = np.zeros(16, dtype=np.complex64)
            reference[4:] = 1 + 0.5j
            reference_path = root / "reference_source.cfile"
            reference.tofile(reference_path)
            catalog_path = output_root / "manifests" / "reference_catalog.csv"
            write_rows(
                catalog_path,
                builder.CATALOG_FIELDS,
                [catalog_row(0, b"\x40\x00", reference_path)],
            )

            raw = (
                np.arange(100, dtype=np.float32)
                + 1j * np.arange(100, dtype=np.float32)[::-1]
            ).astype(np.complex64)
            capture_path = root / "capture.cfile"
            raw.tofile(capture_path)
            packets_path = output_root / "packets.csv"
            packet = {field: "" for field in builder.PACKET_FIELDS}
            packet.update(
                {
                    "physical_packet_uid": "exp003_sess002_run001:seg00:slot000000",
                    "capture_uid": "exp003_sess002_run001",
                    "capture_packet_index": 0,
                    "sequence_segment": 0,
                    "schedule_slot": 0,
                    "detection_index": 0,
                    "start_sample_2m": 20,
                    "predicted_start_sample_2m": 20,
                    "timing_residual_samples": 0,
                    "reference_id": 0,
                    "association_method": "crc_exact",
                    "association_confidence": 1.0,
                    "status": "accepted",
                    "crc_valid": 1,
                    "decoded_frame_hex": "40 00",
                    "catalog_frame_exact": 1,
                }
            )
            write_rows(packets_path, builder.PACKET_FIELDS, [packet])

            with mock.patch.object(builder, "DATA_ROOT", data_root.resolve()):
                generated = builder.trim_packets(
                    capture_path,
                    descriptor(),
                    catalog_path,
                    packets_path,
                    output_root,
                    reference_mode="copy",
                    compute_capture_hash=False,
                )
                validation_path = builder.validate_dataset(output_root)

            self.assertEqual(len(generated), 2)
            phase0 = np.fromfile(generated[0], dtype=np.dtype("<c8"))
            phase1 = np.fromfile(generated[1], dtype=np.dtype("<c8"))
            crop = raw[12:44]
            np.testing.assert_array_equal(phase0, crop[0::2])
            np.testing.assert_array_equal(phase1, crop[1::2])
            self.assertEqual(phase0.size, 16)
            self.assertEqual(phase0[0::4].size, 4)

            views = builder.read_csv(output_root / "manifests" / "views.csv")
            self.assertEqual(len(views), 8)
            self.assertEqual(
                {int(row["combined_decimation_phase_2m"]) for row in views},
                set(range(8)),
            )
            validation = builder.read_csv(validation_path)
            self.assertEqual(len(validation), 2)
            self.assertTrue(all(row["valid"] == "1" for row in validation))

            metadata = json.loads(
                (output_root / "metadata" / "exp3_000000_rxg20_1_fulltrim.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["reference"]["reference_id"], 0)
            self.assertEqual(
                metadata["view"]["combined_decimation_phases_2m"],
                [1, 3, 5, 7],
            )


if __name__ == "__main__":
    unittest.main()
