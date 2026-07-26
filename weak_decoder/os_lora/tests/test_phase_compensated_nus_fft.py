"""Tests for coarse phase-compensated NUS ordinary FFT processing."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    _oversampled_downchirp,
)
from weak_decoder.chirp import build_upchirp
from weak_decoder.os_lora.experiments.evaluate_phase_compensated_nus_fft import (
    _two_sided_sign_p,
    phase_compensated_fft_spectra,
    phase_compensated_nus_powers,
)
from weak_decoder.os_lora.system.nonuniform_sampling import build_pattern_bank


class PhaseCompensatedNusFftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sf = 6
        self.os_factor = 4
        self.n_bins = 1 << self.sf
        self.bank = build_pattern_bank(self.sf, self.os_factor, kind="canonical_only")
        self.downchirp = _oversampled_downchirp(self.sf, self.os_factor, 0, 0.0)

    def _dechirped_symbol(self, raw_bin: int) -> np.ndarray:
        return np.asarray(
            build_upchirp(self.sf, int(raw_bin), self.os_factor) * self.downchirp,
            dtype=np.complex64,
        )

    def test_known_bin_translation_aligns_all_patterns_to_q0(self) -> None:
        for raw_bin in (0, 1, 17, 32, 63):
            with self.subTest(raw_bin=raw_bin):
                spectra = phase_compensated_fft_spectra(
                    self._dechirped_symbol(raw_bin), self.bank, raw_bin
                )
                selected = np.argmax(np.abs(spectra) ** 2, axis=1)
                np.testing.assert_array_equal(selected, np.full(len(self.bank.names), raw_bin))
                reference = np.broadcast_to(spectra[0][None, :], spectra.shape)
                np.testing.assert_allclose(spectra, reference, atol=3e-6, rtol=3e-6)

    def test_plain_coarse_then_one_compensated_fft_recovers_ideal_bins(self) -> None:
        for raw_bin in range(self.n_bins):
            with self.subTest(raw_bin=raw_bin):
                _plain, compensated, estimates = phase_compensated_nus_powers(
                    self._dechirped_symbol(raw_bin), self.bank, iterations=2
                )
                self.assertEqual(2, len(compensated))
                self.assertEqual(raw_bin, int(np.argmax(compensated[0])))
                self.assertEqual(raw_bin, estimates[1])
                self.assertEqual(raw_bin, estimates[2])

    def test_large_paired_sign_test_is_numerically_stable(self) -> None:
        value = _two_sided_sign_p(4000, 3840)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
