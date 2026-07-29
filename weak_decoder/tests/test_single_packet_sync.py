"""Tests for reusable one-packet detection and synchronization."""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.chirp import build_upchirp
from weak_decoder.synchronization import (
    SinglePacketSyncConfig,
    run_single_packet_sync,
    sync_word_to_symbols,
)


class SinglePacketSyncTests(unittest.TestCase):
    def test_detects_frame_without_a_supplied_packet_boundary(self) -> None:
        sf = 6
        os_factor = 4
        preamble = 8
        sync_word = 0x12
        up = build_upchirp(sf, 0, os_factor)
        down = np.conjugate(up).astype(np.complex64)
        sync1, sync2 = sync_word_to_symbols(sync_word)
        frame = np.concatenate(
            (
                np.tile(up, preamble),
                build_upchirp(sf, sync1, os_factor),
                build_upchirp(sf, sync2, os_factor),
                down,
                down,
                down[: up.size // 4],
                build_upchirp(sf, 17, os_factor),
            )
        ).astype(np.complex64)
        samples = np.concatenate((np.zeros(73, dtype=np.complex64), frame))
        result = run_single_packet_sync(
            samples,
            SinglePacketSyncConfig(
                sf=sf,
                bw_hz=31_250.0,
                sample_rate_hz=125_000.0,
                center_frequency_hz=487_700_000.0,
                preamble_symbols=preamble,
                sync_word=sync_word,
                scan_chirps=16,
            ),
        )
        self.assertTrue(result.synchronized, result)
        self.assertIsNotNone(result.frame_sync)
        self.assertLessEqual(
            abs(int(result.frame_sync.fine_preamble_start_sample) - 73),
            os_factor,
        )


if __name__ == "__main__":
    unittest.main()
