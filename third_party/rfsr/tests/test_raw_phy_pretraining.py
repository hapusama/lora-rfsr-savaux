from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


RFSR_ROOT = Path(__file__).resolve().parents[1]
if str(RFSR_ROOT) not in sys.path:
    sys.path.insert(0, str(RFSR_ROOT))

from rfsr.PHY import (  # noqa: E402
    apply_hi2lora_sto,
    encode_random_raw_phy,
    encode_raw_phy,
)
from rfsr.nn.dataset import (  # noqa: E402
    ReferencePhyPretrainingDataset,
    SyntheticLoRaDataset,
)


FRAME0 = bytes.fromhex(
    "40 44 33 22 11 00 01 00 58 "
    "00 00 B9 88 3A 2D 11 8A A8 70 66 A9 9F 80 48 26 E7 52 29 AF "
    "78 56 34 12"
)


def _write_reference(
    root: Path,
    payload_id: int,
    samples: np.ndarray,
    *,
    frame_bytes: int = 4,
    sf: int = 7,
    cr: int = 4,
    preamble_symbols: int = 5,
    ldro: bool = False,
    leading_silence_samples: int = 0,
) -> None:
    reference_dir = root / "reference"
    metadata_dir = root / "metadata"
    reference_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{payload_id:06d}"
    iq_relative = f"reference/signalout_{stem}_fulltrim.cfile"
    np.asarray(samples, dtype=np.dtype("<c8")).tofile(root / iq_relative)
    metadata = {
        "schema": "lora-rfsr-reference",
        "schema_version": 2,
        "reference_kind": "ideal_tx_complex_baseband",
        "alignment_status": "not_aligned_to_ota",
        "packet": {
            "payload_id": payload_id,
            "frame_bytes": frame_bytes,
        },
        "phy": {
            "sample_rate_hz": 1_000_000,
            "sf": sf,
            "bandwidth_hz": 125_000,
            "cr": cr,
            "phy_crc": True,
            "explicit_header": True,
            "preamble_symbols": preamble_symbols,
            "sync_word": 0x12,
            "ldro": ldro,
            "crc_mode": "grlora",
            "leading_silence_samples": leading_silence_samples,
            "trailing_silence_samples": 0,
        },
        "iq": {
            "relative_path": iq_relative,
            "dtype": "<c8",
            "complex_samples": int(samples.size),
            "awgn_added": False,
        },
    }
    (metadata_dir / f"{stem}.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


class RawPhyEncodingTest(unittest.TestCase):
    def test_raw_frame_symbols_and_iq_match_validated_reference(self) -> None:
        encoded = encode_raw_phy(
            FRAME0,
            1_000_000,
            SF=12,
            BW=125_000,
            cr=4,
            enable_crc=1,
            preamble_bits=16,
            sync_word=0x12,
            ldro=True,
            crc_mode="grlora",
            leading_silence_samples=10_000,
        )

        self.assertEqual(
            encoded.header_symbol_ids,
            (2349, 2749, 961, 3757, 2761, 889, 3237, 1433),
        )
        self.assertEqual(
            encoded.payload_symbol_ids[:8],
            (3293, 313, 1433, 2865, 3469, 3985, 3165, 1669),
        )
        self.assertEqual(
            encoded.payload_symbol_ids[-8:],
            (73, 53, 25, 3077, 3589, 1281, 1, 125),
        )
        self.assertEqual(len(encoded.payload_symbol_ids), 56)
        self.assertEqual(encoded.samples.size, 2_770_704)
        self.assertTrue(np.all(encoded.samples[:10_000] == 0))

    def test_random_raw_payload_is_seeded_and_changes_symbols(self) -> None:
        kwargs = {
            "SF": 7,
            "BW": 125_000,
            "cr": 4,
            "enable_crc": 1,
            "preamble_bits": 5,
            "sync_word": 0x12,
            "ldro": False,
            "crc_mode": "grlora",
            "leading_silence_samples": 0,
        }
        first = encode_random_raw_phy(4, 1_000_000, seed=7, **kwargs)
        repeated = encode_random_raw_phy(4, 1_000_000, seed=7, **kwargs)
        different = encode_random_raw_phy(4, 1_000_000, seed=8, **kwargs)

        self.assertEqual(first.payload, repeated.payload)
        self.assertTrue(np.array_equal(first.samples, repeated.samples))
        self.assertNotEqual(first.payload, different.payload)
        self.assertNotEqual(
            first.payload_symbol_ids,
            different.payload_symbol_ids,
        )

    def test_cfo_rotates_packet_without_changing_zero_cfo_reference(self) -> None:
        kwargs = {
            "SF": 7,
            "BW": 125_000,
            "cr": 4,
            "enable_crc": 1,
            "preamble_bits": 5,
            "sync_word": 0x12,
            "ldro": False,
            "crc_mode": "grlora",
            "leading_silence_samples": 16,
        }
        clean = encode_raw_phy(bytes(4), 1_000_000, **kwargs)
        shifted = encode_raw_phy(
            bytes(4),
            1_000_000,
            cfo_hz=1_500.0,
            **kwargs,
        )
        self.assertEqual(clean.cfo_hz, 0.0)
        self.assertEqual(shifted.cfo_hz, 1_500.0)
        self.assertTrue(np.all(clean.samples[:16] == 0))
        self.assertTrue(np.all(shifted.samples[:16] == 0))
        self.assertFalse(np.array_equal(clean.samples[17:], shifted.samples[17:]))
        np.testing.assert_allclose(
            np.abs(clean.samples),
            np.abs(shifted.samples),
            rtol=0,
            atol=3e-7,
        )

    def test_hi2lora_sto_frequency_drift_and_wrap_phase(self) -> None:
        sf = 7
        bandwidth_hz = 125_000
        sample_rate_hz = 1_000_000
        decimation = 4
        preamble_symbols = 5
        leading_samples = 32
        tau_chips = 0.25
        encoded = encode_raw_phy(
            bytes(4),
            sample_rate_hz,
            SF=sf,
            BW=bandwidth_hz,
            cr=4,
            enable_crc=1,
            preamble_bits=preamble_symbols,
            sync_word=0x12,
            ldro=False,
            crc_mode="grlora",
            leading_silence_samples=leading_samples,
        )
        sto_kwargs = {
            "fs": sample_rate_hz,
            "BW": bandwidth_hz,
            "SF": sf,
            "output_decimation": decimation,
            "preamble_bits": preamble_symbols,
            "leading_silence_samples": leading_samples,
        }
        clean = apply_hi2lora_sto(encoded.samples, **sto_kwargs)
        np.testing.assert_array_equal(
            clean,
            encoded.samples[::decimation],
        )
        shifted = apply_hi2lora_sto(
            encoded.samples,
            initial_sto_chips=tau_chips,
            **sto_kwargs,
        )

        chips_per_symbol = 1 << sf
        input_samples_per_chip = (
            sample_rate_hz // bandwidth_hz // decimation
        )
        input_samples_per_symbol = (
            chips_per_symbol * input_samples_per_chip
        )
        packet_start = leading_samples // decimation

        # 第二个 preamble chirp 的前一个 chirp 仍是 z=0，可避开包起点。
        preamble_start = packet_start + input_samples_per_symbol
        preamble_ratio = (
            shifted[
                preamble_start:
                preamble_start + input_samples_per_symbol
            ]
            * np.conj(
                clean[
                    preamble_start:
                    preamble_start + input_samples_per_symbol
                ]
            )
        )
        phase_steps = np.angle(
            preamble_ratio[1:] * np.conj(preamble_ratio[:-1])
        )
        expected_step = (
            -2.0 * np.pi * tau_chips / input_samples_per_symbol
        )
        self.assertAlmostEqual(
            float(np.median(phase_steps)),
            expected_step,
            places=6,
        )

        # sync1 的 z=8，去掉式 (2d) 的线性项后，回绕两侧相差 2πτ。
        sync_symbol_id = 8
        sync_start = (
            packet_start
            + preamble_symbols * input_samples_per_symbol
        )
        sync_ratio = (
            shifted[sync_start:sync_start + input_samples_per_symbol]
            * np.conj(
                clean[sync_start:sync_start + input_samples_per_symbol]
            )
        )
        input_index = np.arange(
            input_samples_per_symbol,
            dtype=np.float64,
        )
        detrended = sync_ratio * np.exp(
            2j * np.pi * tau_chips * input_index
            / input_samples_per_symbol
        )
        fold = (
            (chips_per_symbol - sync_symbol_id)
            * input_samples_per_chip
        )
        phase_before = np.mean(detrended[10:fold - 4])
        phase_after = np.mean(detrended[fold + 4:-10])
        wrap_phase = float(
            np.angle(phase_after * np.conj(phase_before))
        )
        self.assertAlmostEqual(
            wrap_phase,
            2.0 * np.pi * tau_chips,
            places=6,
        )

        drifting = apply_hi2lora_sto(
            encoded.samples,
            initial_sto_chips=0.1,
            sto_slope_chips_per_symbol=0.02,
            **sto_kwargs,
        )
        for symbol_index in (1, 2, 3):
            start = (
                packet_start
                + symbol_index * input_samples_per_symbol
            )
            ratio = (
                drifting[start:start + input_samples_per_symbol]
                * np.conj(clean[start:start + input_samples_per_symbol])
            )
            step = np.median(
                np.angle(ratio[1:] * np.conj(ratio[:-1]))
            )
            estimated_tau = (
                -float(step) * input_samples_per_symbol / (2.0 * np.pi)
            )
            expected_tau = 0.1 + 0.02 * symbol_index
            self.assertAlmostEqual(estimated_tau, expected_tau, places=3)


class ReferencePhyPretrainingDatasetTest(unittest.TestCase):
    def test_snr_range_and_fixed_snr(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            samples = np.exp(1j * np.arange(16)).astype(np.complex64)
            _write_reference(root, 0, samples)

            ranged = ReferencePhyPretrainingDataset(
                root,
                oversampling=4,
                size=1,
                snr_range=(-12.0, -8.0),
                seed=7,
                expected_sf=7,
                random_payload=False,
            )
            x, y, snr = ranged[0]
            self.assertEqual(tuple(x.shape), (2, 4))
            self.assertEqual(tuple(y.shape), (2, 16))
            self.assertGreaterEqual(float(snr.item()), -12.0)
            self.assertLessEqual(float(snr.item()), -8.0)

            fixed = ReferencePhyPretrainingDataset(
                root,
                oversampling=4,
                size=1,
                snr_range=(-15.0, -15.0),
                seed=7,
                expected_sf=7,
                random_payload=False,
            )
            _, _, fixed_snr = fixed[0]
            self.assertAlmostEqual(float(fixed_snr.item()), -15.0)

    def test_random_mode_caches_initially_generated_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = encode_raw_phy(
                bytes(4),
                1_000_000,
                SF=7,
                BW=125_000,
                cr=4,
                enable_crc=1,
                preamble_bits=5,
                sync_word=0x12,
                ldro=False,
                crc_mode="grlora",
                leading_silence_samples=0,
            )
            _write_reference(root, 0, template.samples)
            dataset = ReferencePhyPretrainingDataset(
                root,
                oversampling=4,
                size=2,
                snr_range=(-10.0, -10.0),
                expected_sf=7,
                seed=7,
                random_payload=True,
                cfo_range_hz=(-1_500.0, -1_500.0),
            )

            first_x, first_y, _ = dataset[0]
            second_x, second_y, _ = dataset[0]
            _, other_y, _ = dataset[1]
            self.assertEqual(tuple(first_y.shape), tuple(second_y.shape))
            np.testing.assert_array_equal(first_x.numpy(), second_x.numpy())
            np.testing.assert_array_equal(first_y.numpy(), second_y.numpy())
            self.assertFalse(np.array_equal(first_y.numpy(), other_y.numpy()))
            self.assertEqual(dataset.last_cfo_hz, -1_500.0)

            zero_cfo_dataset = ReferencePhyPretrainingDataset(
                root,
                oversampling=4,
                size=2,
                snr_range=(-10.0, -10.0),
                expected_sf=7,
                seed=7,
                random_payload=True,
                cfo_range_hz=(0.0, 0.0),
            )
            zero_cfo_x, zero_cfo_y, _ = zero_cfo_dataset[0]
            np.testing.assert_array_equal(first_y.numpy(), zero_cfo_y.numpy())
            self.assertFalse(
                np.array_equal(first_x.numpy(), zero_cfo_x.numpy())
            )

    def test_upstream_dataset_also_accepts_fixed_snr(self) -> None:
        dataset = SyntheticLoRaDataset(
            oversampling=4,
            size=1,
            payload_length=2,
            downsampling=8,
            SF=7,
            BW=125_000,
            snr_range=(-9.0, -9.0),
        )
        _, _, snr = dataset[0]
        self.assertAlmostEqual(float(snr.item()), -9.0)

    def test_sto_changes_only_input_and_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = encode_raw_phy(
                bytes(4),
                1_000_000,
                SF=7,
                BW=125_000,
                cr=4,
                enable_crc=1,
                preamble_bits=5,
                sync_word=0x12,
                ldro=False,
                crc_mode="grlora",
                leading_silence_samples=0,
            )
            _write_reference(root, 0, template.samples)
            common = {
                "oversampling": 4,
                "size": 1,
                "snr_range": (-10.0, -10.0),
                "expected_sf": 7,
                "seed": 11,
                "random_payload": True,
                "cfo_range_hz": (0.0, 0.0),
                "sto_initial_range_chips": (0.25, 0.25),
                "sto_slope_range_chips_per_symbol": (0.02, 0.02),
            }
            with_sto = ReferencePhyPretrainingDataset(
                root,
                sto_enabled=True,
                **common,
            )
            without_sto = ReferencePhyPretrainingDataset(
                root,
                sto_enabled=False,
                **common,
            )

            sto_x, sto_y, _ = with_sto[0]
            clean_x, clean_y, _ = without_sto[0]
            np.testing.assert_array_equal(sto_y.numpy(), clean_y.numpy())
            self.assertFalse(np.array_equal(sto_x.numpy(), clean_x.numpy()))
            self.assertEqual(with_sto.last_initial_sto_chips, 0.25)
            self.assertEqual(
                with_sto.last_sto_slope_chips_per_symbol,
                0.02,
            )
            self.assertEqual(without_sto.last_initial_sto_chips, 0.0)
            self.assertEqual(
                without_sto.last_sto_slope_chips_per_symbol,
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
