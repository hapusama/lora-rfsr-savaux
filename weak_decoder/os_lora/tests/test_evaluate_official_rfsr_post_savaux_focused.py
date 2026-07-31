"""Tests for the focused official OTA post-FrameSync Savaux comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "evaluate_official_rfsr_post_savaux_focused.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_official_rfsr_post_savaux_focused", SCRIPT
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load focused evaluation script: {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EvaluateOfficialRfsrPostSavauxFocusedTests(unittest.TestCase):
    def test_batch_savaux_matches_paper_api_at_native_and_rfsr_os(self) -> None:
        from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
            paper_oversampled_spectrum,
        )
        from weak_decoder.chirp import build_upchirp

        for os_factor in (2, 8):
            with self.subTest(os_factor=os_factor):
                first = build_upchirp(12, 137, os_factor)
                second = build_upchirp(12, 3001, os_factor)
                samples = np.concatenate((first, second)).astype(np.complex64)
                starts = [0, (1 << 12) * os_factor]
                actual = MODULE.batch_savaux_spectra(
                    samples,
                    starts,
                    sf=12,
                    os_factor=os_factor,
                    cfo_int=0,
                    cfo_frac=0.0,
                )
                for index, start in enumerate(starts):
                    expected, _, _ = paper_oversampled_spectrum(
                        samples,
                        start,
                        12,
                        os_factor,
                        cfo_correction_mode="symbol",
                    )
                    np.testing.assert_allclose(
                        actual[index], expected, rtol=2e-6, atol=5e-6
                    )

    def test_paired_summary_uses_only_common_clean_sync_attempts(self) -> None:
        def score(included: bool, errors: int) -> dict:
            return {
                "included": included,
                "symbol_count": 40,
                "symbol_errors": errors,
            }

        rows = [
            {
                "methods": {
                    "rfsr": {"score": score(True, 1)},
                    "native": {"score": score(True, 7)},
                }
            },
            {
                "methods": {
                    "rfsr": {"score": score(True, 0)},
                    "native": {"score": score(False, 40)},
                }
            },
        ]
        summary = MODULE._paired_summary(rows, "rfsr", "native")
        self.assertEqual(summary["common_packet_attempts"], 1)
        self.assertEqual(summary["symbol_count_per_method"], 40)
        self.assertEqual(summary["left_symbol_errors"], 1)
        self.assertEqual(summary["right_symbol_errors"], 7)
        self.assertEqual(summary["left_better_attempts"], 1)
        self.assertEqual(summary["right_better_attempts"], 0)

    def test_250k_noise_is_paired_with_every_fourth_1m_sample(self) -> None:
        noise = np.arange(20, dtype=np.float32).astype(np.complex64)
        np.testing.assert_array_equal(
            MODULE.paired_unit_noise(noise, "native_250ksps"),
            np.asarray([0, 4, 8, 12, 16]),
        )
        np.testing.assert_array_equal(
            MODULE.paired_unit_noise(noise, "official_ota_rfsr_1msps"), noise
        )


if __name__ == "__main__":
    unittest.main()
