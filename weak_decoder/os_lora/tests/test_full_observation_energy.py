"""验证 full-observation 能量审计中的 GT/假峰提取。"""

from __future__ import annotations

import unittest

import numpy as np

from weak_decoder.os_lora.experiments.evaluate_full_colored_ml_awgn import (
    _groundtruth_energy_metrics,
)


class FullObservationEnergyTests(unittest.TestCase):
    def test_groundtruth_energy_metrics_are_signed(self) -> None:
        scores = np.asarray([[2.0, 8.0, 4.0], [9.0, 3.0, 6.0]], dtype=np.float64)
        groundtruth = np.asarray([1, 1], dtype=np.int64)

        gt_score, false_score, margin_db = _groundtruth_energy_metrics(scores, groundtruth)

        np.testing.assert_allclose(gt_score, [8.0, 3.0])
        np.testing.assert_allclose(false_score, [4.0, 9.0])
        self.assertAlmostEqual(float(margin_db[0]), 10.0 * np.log10(2.0))
        self.assertAlmostEqual(float(margin_db[1]), 10.0 * np.log10(1.0 / 3.0))

    def test_groundtruth_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _groundtruth_energy_metrics(np.ones((2, 3)), np.asarray([0]))


if __name__ == "__main__":
    unittest.main()
