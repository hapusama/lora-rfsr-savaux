from __future__ import annotations

import unittest

from weak_decoder.os_lora.experiments.evaluate_litenap_savaux import (
    _alias_error_mode,
    _signed_alias_bin_delta,
)


class LiteNapErrorModeTests(unittest.TestCase):
    def test_correct_full_bin(self) -> None:
        self.assertEqual(
            _alias_error_mode(803, 803, 1024, 4),
            "correct",
        )

    def test_correct_alias_bin_but_wrong_group(self) -> None:
        self.assertEqual(
            _alias_error_mode(547, 803, 1024, 4),
            "wrong_group",
        )

    def test_wrong_alias_bin(self) -> None:
        self.assertEqual(
            _alias_error_mode(802, 803, 1024, 4),
            "wrong_alias_bin",
        )

    def test_factor_must_divide_bin_count(self) -> None:
        with self.assertRaises(ValueError):
            _alias_error_mode(1, 1, 1024, 3)

    def test_signed_alias_delta_uses_shortest_circular_distance(self) -> None:
        self.assertEqual(
            _signed_alias_bin_delta(0, 255, 1024, 4),
            1,
        )
        self.assertEqual(
            _signed_alias_bin_delta(255, 0, 1024, 4),
            -1,
        )
        self.assertEqual(
            _signed_alias_bin_delta(547, 803, 1024, 4),
            0,
        )


if __name__ == "__main__":
    unittest.main()
