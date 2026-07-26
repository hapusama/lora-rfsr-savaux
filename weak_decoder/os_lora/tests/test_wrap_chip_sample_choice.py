"""Tests for the single-chip LoRa wrap sample-choice diagnostic."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    _oversampled_downchirp,
)
from weak_decoder.chirp import build_upchirp
from weak_decoder.os_lora.experiments.evaluate_wrap_chip_sample_choice import (
    gt_segment_metrics,
    wrap_chip_choice_spectrum,
)


class WrapChipSampleChoiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sf = 6
        self.os_factor = 4
        self.n_bins = 1 << self.sf
        self.downchirp = _oversampled_downchirp(self.sf, self.os_factor, 0, 0.0)

    def _dechirped(self, raw_bin: int) -> np.ndarray:
        return np.asarray(
            build_upchirp(self.sf, raw_bin, self.os_factor) * self.downchirp,
            dtype=np.complex64,
        )

    def test_q0_choice_is_exactly_the_ordinary_fixed_branch_fft(self) -> None:
        symbol = self._dechirped(17)
        expected = np.fft.fft(symbol[0:: self.os_factor]) / np.sqrt(self.n_bins)
        for delta in (-1, 0, 1):
            actual = wrap_chip_choice_spectrum(
                symbol, self.sf, self.os_factor, delta, selected_offset=0
            )
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_exact_phase_translation_preserves_ideal_candidate_decisions(self) -> None:
        for raw_bin in range(self.n_bins):
            symbol = self._dechirped(raw_bin)
            for delta in (-1, 0, 1):
                for offset in range(self.os_factor):
                    with self.subTest(raw_bin=raw_bin, delta=delta, offset=offset):
                        spectrum = wrap_chip_choice_spectrum(
                            symbol, self.sf, self.os_factor, delta, offset
                        )
                        self.assertEqual(raw_bin, int(np.argmax(np.abs(spectrum) ** 2)))

    def test_ideal_gt_head_tail_metrics_do_not_depend_on_selected_offset(self) -> None:
        raw_bin = 17
        symbol = self._dechirped(raw_bin)
        for delta in (-1, 0, 1):
            baseline = gt_segment_metrics(
                symbol, self.sf, self.os_factor, raw_bin, delta, selected_offset=0
            )
            for offset in range(1, self.os_factor):
                actual = gt_segment_metrics(
                    symbol, self.sf, self.os_factor, raw_bin, delta, offset
                )
                for key in (
                    "head_energy",
                    "tail_energy",
                    "total_energy",
                    "head_tail_coherence",
                    "head_tail_phase_deg",
                ):
                    self.assertAlmostEqual(float(baseline[key]), float(actual[key]), places=5)


if __name__ == "__main__":
    unittest.main()
