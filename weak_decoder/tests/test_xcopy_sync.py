from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.chirp import build_upchirp
from weak_decoder.synchronization.frame_locator import sync_word_to_symbols
from weak_decoder.synchronization.xcopy_sync import (
    XCopyConfig,
    XCopySoftFrameCandidate,
    _soft_frame_candidate_usable,
    run_xcopy_paper_sync,
    run_xcopy_sync,
    scan_periodic_preamble,
    xcopy_raw_symbol_rows,
)


def _build_test_frame(
    sf: int,
    os_factor: int,
    preamble_symbols: int,
    sync_word: int,
    payload_symbols: int,
) -> np.ndarray:
    up = build_upchirp(sf, symbol_id=0, os_factor=os_factor)
    down = np.conjugate(up).astype(np.complex64)
    sync1, sync2 = sync_word_to_symbols(sync_word)
    parts = [
        np.tile(up, preamble_symbols),
        build_upchirp(sf, symbol_id=sync1, os_factor=os_factor),
        build_upchirp(sf, symbol_id=sync2, os_factor=os_factor),
        down,
        down,
        down[: up.size // 4],
    ]
    parts.extend(
        build_upchirp(sf, symbol_id=7 + 3 * index, os_factor=os_factor)
        for index in range(payload_symbols)
    )
    return np.concatenate(parts).astype(np.complex64)


class XCopySyncTest(unittest.TestCase):
    def _config(self, **overrides: object) -> XCopyConfig:
        values: dict[str, object] = {
            "sf": 6,
            "bw": 31_250.0,
            "samp_rate": 125_000.0,
            "preamble_symbols": 8,
            "sync_word": 0x12,
            "retransmit_period_samples": 7_000,
            "payload_symbols": 4,
            "detection_chirps": 4,
            "phase_hop_samples": 256,
            "min_detection_peak_to_median": 6.0,
            "alignment_search_samples": 12,
            "alignment_decimation": 4,
            "max_relative_cfo_hz": 20.0,
            "min_alignment_peak_to_median": 12.0,
            "alignment_timing_model_tolerance": 1.5,
            "alignment_cfo_model_tolerance_hz": 1.5,
            "min_aligned_copies": 4,
            "center_freq": 487.7e6,
        }
        values.update(overrides)
        return XCopyConfig(**values)

    def test_periodic_noise_is_rejected(self) -> None:
        rng = np.random.default_rng(20260724)
        noise = (
            rng.normal(size=7_000 * 7) + 1j * rng.normal(size=7_000 * 7)
        ).astype(np.complex64)

        detection = scan_periodic_preamble(noise, self._config())

        self.assertFalse(detection.detected)
        self.assertIsNone(detection.coarse_preamble_phase_sample)

    def test_soft_boundary_requires_preamble_and_sfd_evidence(self) -> None:
        config = self._config(preamble_symbols=8)
        base = {
            "rank": 1,
            "preamble_start_sample": 0,
            "sfd_start_sample": 0,
            "data_start_sample": 0,
            "score": 1.0,
            "preamble_score": 1.0,
            "sfd_score": 1.0,
            "sync_word_bonus": 10.0,
            "preamble_ref_bin": 0,
            "sync1_bin": 0,
            "sync2_bin": 0,
            "sync1_distance": 0,
            "sync2_distance": 0,
            "sfd1_bin": 0,
            "sfd2_bin": 0,
            "mean_preamble_confidence_db": 0.0,
            "mean_sfd_confidence_db": 0.0,
            "coarse_cfo_bins": 0.0,
            "hard_grlora_pattern_valid": False,
        }
        weak = XCopySoftFrameCandidate(
            **base,
            preamble_stable_count=2,
            sfd_bin_distance=0,
        )
        strong = XCopySoftFrameCandidate(
            **base,
            preamble_stable_count=4,
            sfd_bin_distance=1,
        )

        self.assertFalse(_soft_frame_candidate_usable(weak, config))
        self.assertTrue(_soft_frame_candidate_usable(strong, config))

    def test_aligns_independent_retransmissions_and_locates_frame(self) -> None:
        rng = np.random.default_rng(34)
        config = self._config()
        frame = _build_test_frame(
            config.sf,
            config.os_factor,
            config.preamble_symbols,
            config.sync_word,
            config.payload_symbols,
        )
        copy_count = 7
        phase = 1_024
        total_samples = phase + copy_count * config.retransmit_period_samples
        capture = (
            0.2
            * (
                rng.normal(size=total_samples)
                + 1j * rng.normal(size=total_samples)
            )
        ).astype(np.complex64)

        for index in range(copy_count):
            delay = index - 3
            cfo_hz = 8.0 + 0.6 * index
            phase_rad = rng.uniform(-np.pi, np.pi)
            start = phase + index * config.retransmit_period_samples + delay
            time_s = np.arange(frame.size, dtype=np.float64) / config.samp_rate
            rotated = frame * np.exp(1j * (2.0 * np.pi * cfo_hz * time_s + phase_rad))
            capture[start : start + frame.size] += rotated.astype(np.complex64)

        result = run_xcopy_sync(capture, config)

        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(result.aligned_copy_count, 4)
        self.assertIsNotNone(result.frame_location)
        self.assertTrue(result.frame_location.valid)
        self.assertIsNotNone(result.frame_sync)
        self.assertTrue(result.frame_sync.valid)
        delays = [
            item.relative_delay_samples
            for item in result.alignments
            if item.included and not item.is_reference
        ]
        self.assertGreater(len(set(delays)), 1)

    def test_paper_detector_aligns_packets_and_exports_raw_symbols(self) -> None:
        rng = np.random.default_rng(20260725)
        config = self._config(
            detection_peak_fraction=0.3,
            max_copies=6,
        )
        frame = _build_test_frame(
            config.sf,
            config.os_factor,
            config.preamble_symbols,
            config.sync_word,
            config.payload_symbols,
        )
        copy_count = 6
        first_start = 1_024
        total_samples = first_start + copy_count * config.retransmit_period_samples
        capture = (
            0.16
            * (
                rng.normal(size=total_samples)
                + 1j * rng.normal(size=total_samples)
            )
        ).astype(np.complex64)
        for index in range(copy_count):
            start = first_start + index * config.retransmit_period_samples + index - 2
            time_s = np.arange(frame.size, dtype=np.float64) / config.samp_rate
            cfo_hz = 3.0 + 0.4 * index
            phase_rad = rng.uniform(-np.pi, np.pi)
            copy = frame * np.exp(
                1j * (2.0 * np.pi * cfo_hz * time_s + phase_rad)
            )
            capture[start : start + frame.size] += copy.astype(np.complex64)

        result = run_xcopy_paper_sync(capture, config)
        rows = xcopy_raw_symbol_rows(capture, result, header_symbols=0)

        self.assertIn(result.status, {"ok", "ok_soft_boundary"})
        self.assertGreaterEqual(len(result.packet_detections), 4)
        self.assertGreaterEqual(result.aligned_copy_count, 4)
        self.assertEqual(
            len(rows),
            result.aligned_copy_count * config.payload_symbols,
        )
        self.assertTrue(all(row["raw_symbol_start_sample"] >= 0 for row in rows))
        self.assertTrue(
            all(row["boundary_refinement_source"] for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
