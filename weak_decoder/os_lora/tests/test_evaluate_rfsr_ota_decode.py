"""不启动 GNU Radio 的 RFSR OTA 评估入口单元测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "evaluate_rfsr_ota_decode.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rfsr_ota_decode", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - 静态路径损坏时的保护。
    raise RuntimeError(f"cannot load evaluation script: {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Args:
    def __init__(
        self,
        *,
        points=None,
        start=None,
        stop=None,
        step=None,
        include_raw=False,
    ) -> None:
        self.extra_snr_db = points
        self.extra_snr_start_db = start
        self.extra_snr_stop_db = stop
        self.extra_snr_step_db = step
        self.include_raw_ota = include_raw


class EvaluateRfsrOtaDecodeTests(unittest.TestCase):
    def test_worker_count_and_explicit_packet_selection(self) -> None:
        available = MODULE.available_cpu_count()
        self.assertEqual(MODULE.resolve_worker_count(1, 100), 1)
        self.assertEqual(
            MODULE.resolve_worker_count(available + 10, 3),
            min(available, 3),
        )
        records = [
            {"split_group": "packet-a", "adc_phase": 0, "lowrate_phase": 0},
            {"split_group": "packet-b", "adc_phase": 0, "lowrate_phase": 0},
        ]
        self.assertEqual(
            MODULE._select_canonical_packet_indices(
                records, ["packet-b", "packet-a"], 1
            ),
            [1],
        )
        with self.assertRaisesRegex(ValueError, "packet-c"):
            MODULE._select_canonical_packet_indices(
                records, ["packet-c"], None
            )

    def test_descending_half_db_grid_includes_endpoint(self) -> None:
        points = MODULE.resolve_extra_snr_points(
            _Args(start=-14.0, stop=-16.0, step=-0.5)
        )
        self.assertEqual(points, [-14.0, -14.5, -15.0, -15.5, -16.0])

    def test_explicit_points_can_include_raw_ota(self) -> None:
        points = MODULE.resolve_extra_snr_points(
            _Args(points=[-18.0, -18.0, -19.0], include_raw=True)
        )
        self.assertEqual(points, [None, -18.0, -19.0])

    def test_manifest_conditioning_preserves_training_input(self) -> None:
        self.assertEqual(
            MODULE._rfsr_conditioning_snr(-1.5, -19.0, "manifest"),
            -1.5,
        )
        self.assertEqual(
            MODULE._rfsr_conditioning_snr(-1.5, -19.0, "minimum"),
            -19.0,
        )

    def test_symbol_specs_follow_raw_phy_data_boundary(self) -> None:
        metadata = {
            "phy": {"samples_per_symbol": 32, "preamble_symbols": 16},
            "iq": {"leading_silence_samples": 10},
            "symbols": {"header_ids": [1, 5], "payload_ids": [9, 13]},
        }
        values = MODULE._symbol_specs(metadata, "all", 4)
        self.assertEqual([row["gt_bin"] for row in values], [1, 5, 9, 13])
        self.assertEqual([row["start_sample"] for row in values], [658, 690, 722, 754])


if __name__ == "__main__":
    unittest.main()
