"""Unit tests for the structure-preserving oversampled detector."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from weak_decoder.baselines.common import noise_samples
from weak_decoder.chirp import build_upchirp
from weak_decoder.os_lora.experiments.evaluate_oversampled_glrt import (
    _select_active_branch_steering,
)
from weak_decoder.os_lora.system.oversampled_glrt import (
    DualPeakObservation,
    aligned_branch_observations,
    branch_gls_scores,
    coherent_combination_ratio,
    estimate_branch_noise_model,
    estimate_branch_steering,
    estimate_header_bin_correction,
    estimate_pair_noise_model,
    extract_full_rate_dechirped,
    fit_fold_timing_model,
    fit_phase_line,
    full_rate_spectrum,
    KnownDualPeakPair,
    identity_branch_noise_model,
    rerank_coherent_fold_candidates,
    rerank_dual_peak_candidates,
)


class OversampledGLRTTests(unittest.TestCase):
    def test_coherent_combination_ratio_tracks_phase_alignment(self) -> None:
        aligned, _combined, _power = coherent_combination_ratio(1.0, 1.0j, np.pi / 2.0)
        cancelled, _combined, _power = coherent_combination_ratio(1.0, -1.0j, np.pi / 2.0)
        one_peak, _combined, _power = coherent_combination_ratio(1.0, 0.0, 0.0)
        self.assertAlmostEqual(aligned, 1.0)
        self.assertAlmostEqual(cancelled, 0.0)
        self.assertAlmostEqual(one_peak, 0.5)

    def test_coherent_fold_rerank_uses_k_and_k_minus_n(self) -> None:
        sf = 5
        os_factor = 4
        n_bins = 1 << sf
        length = n_bins * os_factor
        branch_scores = np.zeros(n_bins, dtype=np.float64)
        branch_scores[7] = 1.0
        branch_scores[11] = 0.99
        spectrum = np.zeros(length, dtype=np.complex128)
        spectrum[7] = 1.0
        spectrum[7 + length - n_bins] = -1.0
        spectrum[11] = 1.0
        spectrum[11 + length - n_bins] = 1.0
        dechirped = (np.fft.ifft(spectrum) * np.sqrt(length)).astype(np.complex64)
        result = rerank_coherent_fold_candidates(
            dechirped,
            branch_scores,
            sf,
            os_factor,
            phase_rad=0.0,
            top_l=2,
            coherence_weight=1.0,
            selection_mode="joint",
        )
        self.assertEqual(result.branch_selected_bin, 7)
        self.assertEqual(result.selected_bin, 11)

    def test_coherent_fold_rerank_applies_candidate_timing_phase(self) -> None:
        from weak_decoder.os_lora.system.oversampled_glrt import fold_pair_steering

        sf = 5
        os_factor = 4
        n_bins = 1 << sf
        length = n_bins * os_factor
        branch_bin = 7
        coherent_bin = 11
        timing = 0.17
        branch_scores = np.zeros(n_bins, dtype=np.float64)
        branch_scores[branch_bin] = 1.0
        branch_scores[coherent_bin] = 0.99
        spectrum = np.zeros(length, dtype=np.complex128)
        for candidate, sign in ((branch_bin, -1.0), (coherent_bin, 1.0)):
            steering = fold_pair_steering(candidate, sf, os_factor, timing)
            relative_phase = np.angle(steering[1] * np.conjugate(steering[0]))
            spectrum[candidate] = 1.0
            spectrum[candidate + length - n_bins] = sign * np.exp(1j * relative_phase)
        dechirped = (np.fft.ifft(spectrum) * np.sqrt(length)).astype(np.complex64)
        result = rerank_coherent_fold_candidates(
            dechirped,
            branch_scores,
            sf,
            os_factor,
            phase_rad=1.23,
            timing_offset_chips=timing,
            top_l=2,
            selection_mode="confidence_gate",
            min_coherence_gain=0.30,
            max_override_loss_db=0.15,
        )
        self.assertEqual(result.branch_selected_bin, branch_bin)
        self.assertEqual(result.selected_bin, coherent_bin)

    def test_white_gate_rejects_estimated_branch_steering(self) -> None:
        estimated = np.asarray((1.0, 0.8 + 0.2j, 1.1 - 0.1j, 0.9 + 0.05j))
        selected = _select_active_branch_steering(
            identity_branch_noise_model(4),
            estimated,
            1.0,
            "preamble_header",
            0.8,
        )
        self.assertIsNone(selected)

    def test_header_group_consensus_recovers_minus_one_bin_offset(self) -> None:
        observed = tuple(4 * value + 4 for value in range(8))
        result = estimate_header_bin_correction(
            observed,
            1024,
            minimum_consensus=0.75,
        )
        self.assertEqual(result.residual_bins, -1)
        self.assertEqual(result.correction_bins, 1)
        self.assertEqual(result.consensus, 1.0)

    def test_header_group_consensus_rejects_inconsistent_residuals(self) -> None:
        observed = (1, 6, 11, 16, 17, 22, 27, 32)
        result = estimate_header_bin_correction(
            observed,
            1024,
            minimum_consensus=0.75,
        )
        self.assertEqual(result.correction_bins, 0)
        self.assertLess(result.consensus, 0.75)

    def test_lowpass_noise_is_reproducible_and_sample_correlated(self) -> None:
        clean = np.zeros(32768, dtype=np.complex64)
        first = noise_samples(
            clean,
            0.0,
            123,
            1.0,
            noise_shape="lowpass",
            os_factor=4,
            filter_taps=65,
        )
        second = noise_samples(
            clean,
            0.0,
            123,
            1.0,
            noise_shape="lowpass",
            os_factor=4,
            filter_taps=65,
        )
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(np.mean(np.abs(first) ** 2)), 1.0, places=5)
        lag_one = np.mean(first[1:] * np.conjugate(first[:-1]))
        self.assertGreater(float(abs(lag_one)), 0.7)

    def test_ar1_noise_has_requested_colored_structure(self) -> None:
        noise = noise_samples(
            np.zeros(32768, dtype=np.complex64),
            0.0,
            321,
            1.0,
            noise_shape="ar1",
            os_factor=4,
            filter_taps=65,
            color_magnitude=0.85,
            color_phase_rad=0.7,
        )
        self.assertAlmostEqual(float(np.mean(np.abs(noise) ** 2)), 1.0, places=5)
        lag_one = np.mean(noise[1:] * np.conjugate(noise[:-1]))
        self.assertGreater(float(abs(lag_one)), 0.8)
        self.assertAlmostEqual(float(np.angle(lag_one)), 0.7, delta=0.04)

    def test_aligned_branch_sum_is_savaux_combination(self) -> None:
        sf = 6
        os_factor = 4
        rng = np.random.default_rng(19)
        samples = (
            rng.normal(size=(1 << sf) * os_factor)
            + 1j * rng.normal(size=(1 << sf) * os_factor)
        ).astype(np.complex64)
        combined, branches, _ = paper_oversampled_spectrum(samples, 0, sf, os_factor)
        observations = aligned_branch_observations(branches, os_factor)
        np.testing.assert_allclose(
            np.sum(observations, axis=1), combined, rtol=2e-5, atol=2e-5
        )

    def test_white_branch_gls_has_savaux_argmax(self) -> None:
        sf = 6
        os_factor = 4
        rng = np.random.default_rng(23)
        samples = (
            rng.normal(size=(1 << sf) * os_factor)
            + 1j * rng.normal(size=(1 << sf) * os_factor)
        ).astype(np.complex64)
        combined, branches, _ = paper_oversampled_spectrum(samples, 0, sf, os_factor)
        result = branch_gls_scores(
            branches, os_factor, identity_branch_noise_model(os_factor)
        )
        self.assertEqual(result.selected_bin, int(np.argmax(np.abs(combined) ** 2)))
        np.testing.assert_allclose(
            result.scores * os_factor,
            np.abs(combined) ** 2,
            rtol=2e-5,
            atol=2e-5,
        )

    def test_branch_covariance_is_only_osr_by_osr(self) -> None:
        sf = 5
        os_factor = 4
        rng = np.random.default_rng(29)
        windows = (
            rng.normal(size=(20, (1 << sf) * os_factor))
            + 1j * rng.normal(size=(20, (1 << sf) * os_factor))
        ).astype(np.complex64)
        model = estimate_branch_noise_model(
            windows,
            sf,
            os_factor,
            training_bins=(0, 7, 15, 23),
            diagonal_loading=0.05,
        )
        self.assertEqual(model.covariance.shape, (os_factor, os_factor))
        self.assertEqual(model.inverse_covariance.shape, (os_factor, os_factor))
        self.assertEqual(model.snapshot_count, 80)
        diagonal = np.real(np.diag(model.covariance))
        self.assertLess(float(np.max(diagonal) / np.min(diagonal)), 1.8)

    def test_candidate_wise_covariance_is_bank_of_small_matrices(self) -> None:
        sf = 5
        os_factor = 4
        n_bins = 1 << sf
        rng = np.random.default_rng(31)
        windows = (
            rng.normal(size=(24, n_bins * os_factor))
            + 1j * rng.normal(size=(24, n_bins * os_factor))
        ).astype(np.complex64)
        model = estimate_branch_noise_model(
            windows,
            sf,
            os_factor,
            diagonal_loading=0.2,
            covariance_mode="per_bin",
        )
        self.assertEqual(model.covariance.shape, (n_bins, os_factor, os_factor))
        self.assertEqual(
            model.inverse_covariance.shape, (n_bins, os_factor, os_factor)
        )
        self.assertEqual(np.asarray(model.information).shape, (n_bins,))

    def test_candidate_wise_pair_covariance_is_bank_of_two_by_two_matrices(self) -> None:
        sf = 5
        os_factor = 4
        n_bins = 1 << sf
        rng = np.random.default_rng(37)
        windows = (
            rng.normal(size=(24, n_bins * os_factor))
            + 1j * rng.normal(size=(24, n_bins * os_factor))
        ).astype(np.complex64)
        model = estimate_pair_noise_model(
            windows,
            sf,
            os_factor,
            diagonal_loading=0.2,
            covariance_mode="per_bin",
        )
        self.assertEqual(model.covariance.shape, (n_bins, 2, 2))
        self.assertEqual(model.inverse_covariance.shape, (n_bins, 2, 2))
        self.assertEqual(model.snapshot_count, 24)

    def test_rank_one_branch_steering_recovery(self) -> None:
        steering = np.asarray((1.0, 0.8 + 0.2j, 1.1 - 0.1j, 0.9 + 0.05j))
        gains = np.exp(1j * np.linspace(-1.0, 1.2, 12))
        observations = gains[:, None] * steering[None, :]
        estimate = estimate_branch_steering(observations)
        correlation = abs(np.vdot(estimate.steering, steering)) / (
            np.linalg.norm(estimate.steering) * np.linalg.norm(steering)
        )
        self.assertGreater(float(correlation), 0.999999)
        self.assertGreater(estimate.rank_one_fraction, 0.999999)

    def test_dual_peak_consistency_prefers_exact_candidate(self) -> None:
        sf = 7
        os_factor = 4
        n_bins = 1 << sf
        true_bin = 43
        symbol = build_upchirp(sf, true_bin, os_factor)
        dechirped = extract_full_rate_dechirped(
            symbol, 0, sf, os_factor, cfo_correction_mode="none"
        )
        branch_scores = np.zeros(n_bins, dtype=np.float64)
        branch_scores[true_bin] = 1.0
        branch_scores[true_bin + 1] = 0.98
        result = rerank_dual_peak_candidates(
            dechirped,
            branch_scores,
            sf,
            os_factor,
            phase_rad=0.0,
            top_l=2,
            consistency_weight=2.0,
            max_branch_loss_db=3.0,
        )
        by_bin = {row.raw_bin: row for row in result.candidate_scores}
        self.assertEqual(result.selected_bin, true_bin)
        self.assertGreater(
            by_bin[true_bin].consistency, by_bin[true_bin + 1].consistency
        )
        self.assertGreater(by_bin[true_bin].consistency, 0.999)
        self.assertEqual(
            int(np.argmax(np.abs(full_rate_spectrum(dechirped))[:n_bins])), true_bin
        )

    def test_confidence_gate_only_overrides_near_tie(self) -> None:
        sf = 7
        os_factor = 4
        n_bins = 1 << sf
        true_bin = 43
        dechirped = extract_full_rate_dechirped(
            build_upchirp(sf, true_bin, os_factor),
            0,
            sf,
            os_factor,
            cfo_correction_mode="none",
        )
        scores = np.zeros(n_bins, dtype=np.float64)
        scores[true_bin + 1] = 1.0
        scores[true_bin] = 0.999
        result = rerank_dual_peak_candidates(
            dechirped,
            scores,
            sf,
            os_factor,
            phase_rad=0.0,
            top_l=2,
            selection_mode="confidence_gate",
            min_consistency_gain=0.1,
            max_override_loss_db=0.1,
        )
        self.assertEqual(result.selected_bin, true_bin)
        scores[true_bin] = 0.5
        guarded = rerank_dual_peak_candidates(
            dechirped,
            scores,
            sf,
            os_factor,
            phase_rad=0.0,
            top_l=2,
            selection_mode="confidence_gate",
            min_consistency_gain=0.1,
            max_override_loss_db=0.1,
        )
        self.assertEqual(guarded.selected_bin, true_bin + 1)

    def test_diagnostic_only_fold_gate_preserves_branch_decision(self) -> None:
        sf = 7
        os_factor = 4
        n_bins = 1 << sf
        true_bin = 43
        branch_bin = true_bin + 1
        dechirped = extract_full_rate_dechirped(
            build_upchirp(sf, true_bin, os_factor),
            0,
            sf,
            os_factor,
            cfo_correction_mode="none",
        )
        scores = np.zeros(n_bins, dtype=np.float64)
        scores[branch_bin] = 1.0
        scores[true_bin] = 0.999
        result = rerank_dual_peak_candidates(
            dechirped,
            scores,
            sf,
            os_factor,
            phase_rad=0.0,
            top_l=2,
            selection_mode="confidence_gate",
            min_consistency_gain=0.1,
            max_override_loss_db=0.1,
            allow_override=False,
        )
        self.assertEqual(result.selected_bin, branch_bin)
        self.assertEqual(len(result.candidate_scores), 2)

    def test_wrapped_phase_line_fit(self) -> None:
        slope = 0.42
        intercept = 2.9
        observations = []
        for index in range(8):
            phase = float(np.angle(np.exp(1j * (slope * index + intercept))))
            observations.append(
                DualPeakObservation(index, 100, 0.0, phase, 2.0, 1.0, 0.9)
            )
        model = fit_phase_line(observations)
        self.assertAlmostEqual(model.slope_rad_per_symbol, slope, places=10)
        for index in (0, 5, 11):
            residual = np.angle(
                np.exp(1j * (model.predict(index) - (slope * index + intercept)))
            )
            self.assertAlmostEqual(float(residual), 0.0, places=10)

    def test_fixed_phase_slope_uses_circular_intercept(self) -> None:
        slope = -0.037
        intercept = 3.08
        observations = []
        for index in range(8):
            phase = float(np.angle(np.exp(1j * (slope * index + intercept))))
            observations.append(
                DualPeakObservation(index, 200, 0.0, phase, 3.0, 2.0, 0.95)
            )
        model = fit_phase_line(
            observations, fixed_slope_rad_per_symbol=slope
        )
        self.assertAlmostEqual(model.slope_rad_per_symbol, slope, places=12)
        residual = np.angle(np.exp(1j * (model.intercept_rad - intercept)))
        self.assertAlmostEqual(float(residual), 0.0, places=12)

    def test_exact_fold_timing_grid_recovers_synthetic_offset(self) -> None:
        sf = 6
        os_factor = 4
        timing = 0.173
        observations = []
        for index, raw_bin in enumerate((7, 19, 31, 47)):
            from weak_decoder.os_lora.system.oversampled_glrt import fold_pair_steering

            pair = fold_pair_steering(raw_bin, sf, os_factor, timing)
            observations.append(
                KnownDualPeakPair(index, raw_bin, 0.0, complex(pair[0]), complex(pair[1]))
            )
        model = fit_fold_timing_model(
            observations, sf, os_factor, grid_points=1001
        )
        self.assertAlmostEqual(model.offset_chips_at_reference, timing, delta=0.002)
        self.assertGreater(model.mean_consistency, 0.999999)


if __name__ == "__main__":
    unittest.main()
