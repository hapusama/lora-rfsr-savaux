"""Regression tests for the LiteNap-Savaux sub-Nyquist detector."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    _oversampled_downchirp,
    _paper_branch_spectrum,
    combine_paper_branch_spectra,
)
from weak_decoder.chirp import build_upchirp
from weak_decoder.os_lora.system.litenap_savaux import (
    candidate_phase_jump_scores,
    choose_downsample_phases,
    combine_subnyquist_components,
    selected_sample_count,
    subnyquist_component_spectra,
    subnyquist_component_spectra_batch,
)


class LiteNapSavauxTests(unittest.TestCase):
    def test_full_downsample_phase_set_matches_savaux(self) -> None:
        sf = 5
        os_factor = 4
        downsample_factor = 4
        n_bins = 1 << sf
        rng = np.random.default_rng(7)
        dechirped = (
            rng.normal(size=n_bins * os_factor)
            + 1j * rng.normal(size=n_bins * os_factor)
        ).astype(np.complex64)

        components = subnyquist_component_spectra(
            dechirped, sf, os_factor, downsample_factor
        )
        actual = combine_subnyquist_components(components)
        branches = tuple(
            _paper_branch_spectrum(
                dechirped[q::os_factor], sf, os_factor, q
            )
            for q in range(os_factor)
        )
        expected = combine_paper_branch_spectra(branches, os_factor)

        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    def test_savaux_phases_resolve_single_view_alias(self) -> None:
        sf = 7
        os_factor = 4
        downsample_factor = 4
        raw_bin = 93
        aliased_bin = raw_bin % ((1 << sf) // downsample_factor)
        dechirped = (
            build_upchirp(sf, raw_bin, os_factor)
            * _oversampled_downchirp(sf, os_factor)
        ).astype(np.complex64)
        components = subnyquist_component_spectra(
            dechirped, sf, os_factor, downsample_factor
        )

        single = combine_subnyquist_components(
            components,
            branch_phases=(0,),
            downsample_phases=(0,),
        )
        enhanced = combine_subnyquist_components(
            components,
            downsample_phases=(0,),
        )

        self.assertEqual(aliased_bin, int(np.argmax(np.abs(single) ** 2)))
        self.assertEqual(raw_bin, int(np.argmax(np.abs(enhanced) ** 2)))

    def test_sample_budgets_and_phase_selection(self) -> None:
        sf = 10
        os_factor = 4
        downsample_factor = 4
        self.assertEqual(
            (0, 1),
            choose_downsample_phases(os_factor, downsample_factor, 2),
        )
        self.assertEqual(
            256,
            selected_sample_count(
                sf,
                os_factor,
                downsample_factor,
                branch_phases=(0,),
                downsample_phases=(0,),
            ),
        )
        self.assertEqual(
            1024,
            selected_sample_count(
                sf,
                os_factor,
                downsample_factor,
                downsample_phases=(0,),
            ),
        )
        self.assertEqual(
            4096,
            selected_sample_count(sf, os_factor, downsample_factor),
        )

    def test_phase_jump_scores_are_finite(self) -> None:
        sf = 6
        os_factor = 4
        downsample_factor = 4
        n_bins = 1 << sf
        rng = np.random.default_rng(11)
        dechirped = (
            rng.normal(size=n_bins * os_factor)
            + 1j * rng.normal(size=n_bins * os_factor)
        ).astype(np.complex64)
        scores = candidate_phase_jump_scores(
            dechirped,
            sf,
            os_factor,
            downsample_factor,
            candidate_bins=(3, 19, 35, 51),
            downsample_phases=(0, 1),
        )
        self.assertEqual((4,), scores.shape)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0.0))

    def test_batch_spectra_match_single_symbol_calls(self) -> None:
        sf = 5
        os_factor = 4
        downsample_factor = 4
        n_bins = 1 << sf
        rng = np.random.default_rng(19)
        symbols = (
            rng.normal(size=(3, n_bins * os_factor))
            + 1j * rng.normal(size=(3, n_bins * os_factor))
        ).astype(np.complex64)
        actual = subnyquist_component_spectra_batch(
            symbols, sf, os_factor, downsample_factor
        )
        expected = np.stack(
            [
                subnyquist_component_spectra(
                    symbol, sf, os_factor, downsample_factor
                )
                for symbol in symbols
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
