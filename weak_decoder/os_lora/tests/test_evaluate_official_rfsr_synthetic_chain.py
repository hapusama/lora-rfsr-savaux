"""Tests for the local published-checkpoint synthetic/official OTA audit."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "evaluate_official_rfsr_synthetic_chain.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_official_rfsr_synthetic_chain", SCRIPT
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load evaluation script: {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluateOfficialRfsrSyntheticChainTests(unittest.TestCase):
    def test_method_selection_skips_unused_frontends(self) -> None:
        class UnusedFrontend:
            def interpolate(self, samples: np.ndarray) -> np.ndarray:
                raise AssertionError("unused interpolation frontend was called")

            def enhance(self, samples: np.ndarray, snr_db: float) -> np.ndarray:
                raise AssertionError("unused synthetic frontend was called")

        class OtaFrontend:
            def __init__(self) -> None:
                self.calls = 0

            def enhance(self, samples: np.ndarray, snr_db: float) -> np.ndarray:
                self.calls += 1
                return np.repeat(samples, 4).astype(np.complex64)

        ota = OtaFrontend()
        high = np.arange(32, dtype=np.float32).astype(np.complex64)
        outputs = MODULE._method_outputs(
            high,
            UnusedFrontend(),
            ota,
            snr_db=-18.0,
            methods=("native_1msps", "official_ota_rfsr"),
        )
        self.assertEqual(tuple(outputs), ("native_1msps", "official_ota_rfsr"))
        self.assertEqual(ota.calls, 1)
        self.assertEqual(outputs["native_1msps"].size, 16)
        self.assertEqual(outputs["official_ota_rfsr"].size, 16)

    def test_official_coarse_cfo_centering_derotates_integer_bin(self) -> None:
        sample_count = 64
        sample_rate_hz = 64_000.0
        coarse_bin = 5
        sample_index = np.arange(sample_count, dtype=np.float64)
        samples = np.asarray(
            np.exp(2j * np.pi * coarse_bin * sample_index / sample_count),
            dtype=np.complex64,
        )
        config = SimpleNamespace(
            chirp_samples=sample_count,
            sample_rate_hz=sample_rate_hz,
        )
        initial = SimpleNamespace(
            frame_location=SimpleNamespace(preamble_ref_bin=coarse_bin)
        )
        final = SimpleNamespace(frame_location=None)

        with mock.patch.object(
            MODULE,
            "run_single_packet_sync",
            side_effect=(initial, final),
        ) as synchronize:
            prepared = MODULE.prepare_samples_and_sync(
                samples,
                config,
                coarse_cfo_centering=True,
            )

        self.assertIs(prepared.initial_result, initial)
        self.assertIs(prepared.result, final)
        self.assertEqual(prepared.coarse_cfo_bin, coarse_bin)
        self.assertEqual(prepared.coarse_cfo_hz, 5_000.0)
        np.testing.assert_allclose(prepared.samples, 1.0 + 0.0j, atol=2e-6)
        np.testing.assert_array_equal(synchronize.call_args_list[0].args[0], samples)
        np.testing.assert_array_equal(
            synchronize.call_args_list[1].args[0], prepared.samples
        )

    def test_tracked_sync_reuses_candidate_without_full_search(self) -> None:
        @dataclass(frozen=True)
        class Result:
            status: str
            event_count: int
            frame_location: object
            frame_sync: object
            error: str | None = None

        sample_count = 64
        coarse_bin = 5
        samples = np.asarray(
            np.exp(
                2j
                * np.pi
                * coarse_bin
                * np.arange(sample_count, dtype=np.float64)
                / sample_count
            ),
            dtype=np.complex64,
        )
        config = SimpleNamespace(
            sf=7,
            bw_hz=8_000.0,
            sample_rate_hz=64_000.0,
            center_frequency_hz=923_000_000.0,
            preamble_symbols=8,
            sync_word=0x12,
            detection_chirps=4,
            bin_tolerance=2,
            frame_sync_bin0_tolerance=0,
            chirp_samples=sample_count,
        )
        location = SimpleNamespace(valid=True)
        clean_result = Result("ok", 1, location, SimpleNamespace(valid=True))
        clean = MODULE.PreparedSync(
            samples,
            clean_result,
            clean_result,
            coarse_bin,
            5_000.0,
        )
        validated = SimpleNamespace(valid=True)

        with mock.patch.object(
            MODULE,
            "run_grlora_frame_sync_validation",
            return_value=validated,
        ) as validate, mock.patch.object(
            MODULE,
            "run_single_packet_sync",
            side_effect=AssertionError("full packet search should be skipped"),
        ):
            prepared = MODULE.prepare_samples_with_tracked_sync(
                samples,
                config,
                clean,
            )

        self.assertEqual(prepared.result.status, "ok")
        self.assertIs(prepared.result.frame_sync, validated)
        np.testing.assert_allclose(prepared.samples, 1.0 + 0.0j, atol=2e-6)
        self.assertEqual(validate.call_count, 1)

        invalid_result = Result(
            "sync_invalid",
            1,
            SimpleNamespace(valid=False),
            SimpleNamespace(valid=False),
        )
        invalid_clean = clean._replace(
            result=invalid_result,
            initial_result=invalid_result,
        )
        with mock.patch.object(
            MODULE,
            "run_grlora_frame_sync_validation",
            side_effect=AssertionError("an unlocked packet cannot be tracked"),
        ), mock.patch.object(
            MODULE,
            "run_single_packet_sync",
            side_effect=AssertionError("cold-start fallback should be skipped"),
        ):
            invalid = MODULE.prepare_samples_with_tracked_sync(
                samples,
                config,
                invalid_clean,
            )
        self.assertIs(invalid.result, invalid_result)

    def test_official_mode_rejects_metadata_only_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata").mkdir()
            (root / "metadata" / "000000.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "metadata-only"):
                MODULE.load_official_ota_packets(root, 1)

    def test_official_legacy_symbol_mapping_matches_old_modulator(self) -> None:
        from rfsr.PHY import lora_packet
        from weak_decoder.chirp import (
            bin_to_grlora_symbol,
            build_downchirp,
        )

        metadata = {
            "payload": list(range(MODULE.PAYLOAD_BYTES)),
            "sf": MODULE.SF,
            "cr": MODULE.CR,
            "enable_crc": 1,
            "implicit_header": 0,
            "src": 0,
            "dst": 1,
            "seqn": 7,
        }
        expected = MODULE._official_expected_symbols(metadata)
        _, header_ids, payload_ids = lora_packet(
            MODULE.BW_HZ,
            MODULE.OS_FACTOR,
            MODULE.SF,
            MODULE.N_BINS - 8,
            MODULE.N_BINS - 16,
            MODULE.PREAMBLE_SYMBOLS,
            0,
            MODULE.CR,
            1,
            metadata["src"],
            metadata["dst"],
            metadata["seqn"],
            np.asarray(metadata["payload"], dtype=np.uint8),
            50e-6,
            0,
            0,
        )
        downchirp = build_downchirp(MODULE.SF)

        def demodulate(
            old_symbol_id: float, *, is_header: bool, ldro: bool
        ) -> int:
            from rfsr.PHY import lora_chirp

            chirp, _ = lora_chirp(
                +1,
                old_symbol_id,
                MODULE.BW_HZ,
                MODULE.N_BINS,
                MODULE.OS_FACTOR,
            )
            chip_rate = chirp[MODULE.OS_FACTOR // 2 :: MODULE.OS_FACTOR]
            raw_bin = int(np.argmax(np.abs(np.fft.fft(chip_rate * downchirp)) ** 2))
            return bin_to_grlora_symbol(
                raw_bin,
                sf=MODULE.SF,
                is_header=is_header,
                ldro=ldro,
            )

        actual_header = [
            demodulate(value, is_header=True, ldro=False)
            for value in header_ids
        ]
        actual_payload = [
            demodulate(value, is_header=False, ldro=MODULE.LDRO)
            for value in payload_ids
        ]
        self.assertEqual(actual_header, expected["header"])
        self.assertEqual(actual_payload, expected["payload"])

    def test_complex_awgn_uses_requested_power(self) -> None:
        first = MODULE.complex_awgn(1_000_000, 3.5, np.random.default_rng(71))
        second = MODULE.complex_awgn(1_000_000, 3.5, np.random.default_rng(71))
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(np.mean(np.abs(first) ** 2)), 3.5, delta=0.02)

    def test_snr_is_measured_after_decimation_on_active_iq(self) -> None:
        clean = np.zeros(32, dtype=np.complex64)
        clean[8:24] = 1.0 + 0.0j
        noisy = np.asarray(clean + (0.5 + 0.0j), dtype=np.complex64)
        measured = MODULE.measure_decimated_snr(
            clean,
            noisy,
            decimation=2,
            leading_silence_high_rate=8,
            trailing_silence_high_rate=8,
            measured_sample_rate_hz=250_000,
        )
        self.assertEqual(measured["sample_rate_hz"], 250_000)
        self.assertEqual(measured["active_sample_count"], 8)
        self.assertAlmostEqual(measured["signal_power"], 1.0)
        self.assertAlmostEqual(measured["noise_power"], 0.25)
        self.assertAlmostEqual(measured["snr_db"], 10.0 * np.log10(4.0))

    def test_failed_sync_contributes_unit_end_to_end_ser(self) -> None:
        failed = {
            "stage": "pre_rfsr",
            "snr_db": -24.0,
            "packet_index": 0,
            "method": "official_synthetic_rfsr",
            "sync": {"synchronized": False},
            "decoders": {
                name: MODULE._empty_decoder_score(40) for name in MODULE.DECODERS
            },
        }
        summary = MODULE.summarize_rows([failed])[0]
        self.assertEqual(summary["sync_success_rate"], 0.0)
        self.assertEqual(
            summary["decoders"]["ordinary_fft"]["end_to_end_ser"], 1.0
        )
        self.assertIsNone(
            summary["decoders"]["ordinary_fft"]["conditional_ser"]
        )
        self.assertIsNone(summary["input_snr_measurement"])

    def test_summary_accepts_a_single_selected_decoder(self) -> None:
        row = {
            "stage": "pre_rfsr",
            "snr_db": -18.0,
            "packet_index": 0,
            "method": "official_ota_rfsr",
            "sync": {"synchronized": True},
            "decoders": {
                "savaux": {
                    "symbol_count": 40,
                    "symbol_errors": 1,
                    "median_peak_margin_db": 2.0,
                }
            },
        }
        summary = MODULE.summarize_rows([row])[0]
        self.assertEqual(tuple(summary["decoders"]), ("savaux",))
        self.assertEqual(summary["decoders"]["savaux"]["end_to_end_ser"], 0.025)

    def test_summary_reports_measured_250ksps_snr_range(self) -> None:
        def row(packet_index: int, measured_snr_db: float) -> dict:
            return {
                "stage": "pre_rfsr",
                "snr_db": -24.0,
                "packet_index": packet_index,
                "method": "official_synthetic_rfsr",
                "sync": {"synchronized": True},
                "input_snr_measurement": {
                    "sample_rate_hz": 250_000,
                    "snr_db": measured_snr_db,
                },
                "decoders": {
                    name: {
                        "symbol_count": 40,
                        "symbol_errors": 0,
                        "median_peak_margin_db": 1.0,
                    }
                    for name in MODULE.DECODERS
                },
            }

        summary = MODULE.summarize_rows([row(0, -23.9), row(1, -24.1)])[0]
        measured = summary["input_snr_measurement"]
        self.assertEqual(measured["sample_rate_hz"], 250_000)
        self.assertEqual(measured["measurement_count"], 2)
        self.assertAlmostEqual(measured["median_snr_db"], -24.0)
        self.assertAlmostEqual(measured["min_snr_db"], -24.1)
        self.assertAlmostEqual(measured["max_snr_db"], -23.9)

    def test_batched_savaux_matches_paper_symbol_api(self) -> None:
        from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
            paper_oversampled_spectrum,
        )
        from weak_decoder.chirp import build_upchirp

        for os_factor in (2, MODULE.OS_FACTOR):
            with self.subTest(os_factor=os_factor):
                one = build_upchirp(MODULE.SF, 137, os_factor)
                two = build_upchirp(MODULE.SF, 3001, os_factor)
                samples = np.concatenate((one, two)).astype(np.complex64)
                starts = [0, MODULE.N_BINS * os_factor]
                combined, branches = MODULE._batch_savaux_spectra(
                    samples,
                    starts,
                    0,
                    0.0,
                    os_factor=os_factor,
                )
                for index, start in enumerate(starts):
                    expected_combined, expected_branches, _ = (
                        paper_oversampled_spectrum(
                            samples,
                            start,
                            MODULE.SF,
                            os_factor,
                            cfo_correction_mode="symbol",
                        )
                    )
                    np.testing.assert_allclose(
                        combined[index], expected_combined, rtol=2e-6, atol=5e-6
                    )
                    for branch_index in range(os_factor):
                        np.testing.assert_array_equal(
                            branches[index, :, branch_index],
                            expected_branches[branch_index],
                        )


if __name__ == "__main__":
    unittest.main()
