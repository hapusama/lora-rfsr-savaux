"""Tests for the local published-checkpoint synthetic go/no-go audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

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

        one = build_upchirp(MODULE.SF, 137, MODULE.OS_FACTOR)
        two = build_upchirp(MODULE.SF, 3001, MODULE.OS_FACTOR)
        samples = np.concatenate((one, two)).astype(np.complex64)
        starts = [0, MODULE.N_BINS * MODULE.OS_FACTOR]
        combined, branches = MODULE._batch_savaux_spectra(samples, starts, 0, 0.0)
        for index, start in enumerate(starts):
            expected_combined, expected_branches, _ = paper_oversampled_spectrum(
                samples,
                start,
                MODULE.SF,
                MODULE.OS_FACTOR,
                cfo_correction_mode="symbol",
            )
            np.testing.assert_allclose(
                combined[index], expected_combined, rtol=2e-6, atol=5e-6
            )
            for branch_index in range(MODULE.OS_FACTOR):
                np.testing.assert_array_equal(
                    branches[index, :, branch_index],
                    expected_branches[branch_index],
                )


if __name__ == "__main__":
    unittest.main()
