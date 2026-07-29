"""干净输出固定 FrameSync 后再加噪的 Savaux SER 入口单元测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "evaluate_rfsr_savaux_ser.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rfsr_savaux_ser", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load evaluation script: {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _Args:
    extra_snr_db = None
    extra_snr_start_db = -18.0
    extra_snr_stop_db = -19.0
    extra_snr_step_db = -0.5
    include_clean_output = True


class _SeedArgs:
    noise_seed = [20260728]
    noise_seed_count = 3


class EvaluateRfsrSavauxSerTests(unittest.TestCase):
    def test_paired_methods_receive_identical_awgn(self) -> None:
        first = np.ones(128, dtype=np.complex64)
        second = np.full(128, 2.0 + 1.0j, dtype=np.complex64)
        noisy_first = MODULE._add_extra_awgn(
            first,
            -5.0,
            np.random.default_rng(123),
            reference_power=0.25,
        )
        noisy_second = MODULE._add_extra_awgn(
            second,
            -5.0,
            np.random.default_rng(123),
            reference_power=0.25,
        )
        np.testing.assert_allclose(
            noisy_first - first,
            noisy_second - second,
            rtol=0.0,
            atol=2e-7,
        )

    def test_snr_grid_and_clean_output_condition(self) -> None:
        self.assertEqual(
            MODULE.resolve_extra_snr_points(_Args()),
            [None, -18.0, -18.5, -19.0],
        )
        self.assertEqual(MODULE._condition_label(None), "no_extra_noise")
        self.assertEqual(
            MODULE.resolve_noise_seeds(_SeedArgs()),
            [20260728, 20260729, 20260730],
        )

    def test_worker_count_respects_visible_cpu_and_task_count(self) -> None:
        available = MODULE.available_cpu_count()
        self.assertGreaterEqual(available, 1)
        self.assertEqual(MODULE.resolve_worker_count(0, 1), 1)
        self.assertEqual(MODULE.resolve_worker_count(1, 100), 1)
        self.assertEqual(
            MODULE.resolve_worker_count(available + 10, 3),
            min(available, 3),
        )
        with self.assertRaisesRegex(ValueError, "workers"):
            MODULE.resolve_worker_count(-1, 3)

    def test_reference_ids_are_converted_to_hard_symbols(self) -> None:
        metadata = {
            "packet": {"frame_bytes": 33},
            "phy": {"ldro": True, "phy_crc": True, "cr": 4},
            "symbols": {
                "header_ids": [1, 5, 4093],
                "payload_ids": [1, 5, 9, 4093],
            },
        }
        self.assertEqual(
            MODULE.reference_demod_symbols(metadata, sf=12),
            {"header": [0, 1, 1023], "payload": [0, 1, 2, 1023]},
        )

    def test_padding_only_payload_block_is_not_scored(self) -> None:
        metadata = {
            "packet": {"frame_bytes": 33},
            "phy": {"ldro": True, "phy_crc": True, "cr": 4},
            "symbols": {
                "header_ids": [1] * 8,
                "payload_ids": list(range(1, 57)),
            },
        }
        expected = MODULE.reference_demod_symbols(metadata, sf=12)
        self.assertEqual(len(expected["header"]), 8)
        self.assertEqual(len(expected["payload"]), 48)

    def test_split_manifest_proves_physical_packets_are_disjoint(self) -> None:
        datasets = {
            "train": SimpleNamespace(records=[{"split_group": "packet-a"}]),
            "validation": SimpleNamespace(records=[{"split_group": "packet-b"}]),
            "test": SimpleNamespace(records=[{"split_group": "packet-c"}]),
        }
        manifest = MODULE._split_manifest(datasets, seed=42, max_groups=3)
        self.assertTrue(manifest["disjoint"])
        self.assertEqual(
            manifest["physical_packet_counts"],
            {"train": 1, "validation": 1, "test": 1},
        )
        self.assertEqual(manifest["overlaps"]["train_test"], [])

        datasets["test"].records.append({"split_group": "packet-a"})
        with self.assertRaisesRegex(RuntimeError, "leakage"):
            MODULE._split_manifest(datasets, seed=42, max_groups=3)

    def test_missing_decisions_after_sync_count_as_symbol_errors(self) -> None:
        expected = {"header": [3, 4], "payload": [7, 8, 9]}
        demod = SimpleNamespace(
            symbols=(
                SimpleNamespace(stage="payload", stage_symbol_index=0, symbol_value=7),
                SimpleNamespace(stage="payload", stage_symbol_index=1, symbol_value=99),
            )
        )
        score = MODULE.score_demodulation(demod, expected, "payload")
        self.assertEqual(score["symbol_count"], 3)
        self.assertEqual(score["symbol_errors"], 2)
        self.assertEqual(score["missing_symbols"], 1)
        self.assertTrue(score["included"])
        self.assertFalse(score["complete"])

        failed = MODULE.score_demodulation(None, expected, "payload")
        self.assertEqual(failed["symbol_errors"], 3)
        self.assertEqual(failed["ser"], 1.0)

        compact = MODULE.score_demodulation(
            demod, expected, "payload", include_symbol_details=False
        )
        self.assertEqual(compact["symbol_errors"], 2)
        self.assertFalse(compact["symbol_details_included"])
        self.assertEqual(compact["symbols"], [])

    def test_clean_framesync_failure_is_excluded_from_ser_summary(self) -> None:
        rows = [
            {
                "methods": {
                    "rfsr_1msps": {
                        "clean_sync": {"synchronized": True},
                        "demod_after_noise": {"header_valid": True},
                        "score": {
                            "included": True,
                            "symbol_count": 10,
                            "symbol_errors": 2,
                            "complete": True,
                        },
                    }
                }
            },
            {
                "methods": {
                    "rfsr_1msps": {
                        "clean_sync": {"synchronized": False},
                        "demod_after_noise": {"header_valid": False},
                        "score": MODULE.excluded_sync_score("payload"),
                    }
                }
            },
        ]
        summary = MODULE._summary(rows, ["rfsr_1msps"])["rfsr_1msps"]
        self.assertEqual(summary["packet_count"], 2)
        self.assertEqual(summary["clean_synchronized_packets"], 1)
        self.assertEqual(summary["ser_packet_count"], 1)
        self.assertEqual(summary["symbol_count"], 10)
        self.assertEqual(summary["symbol_errors"], 2)
        self.assertEqual(summary["ser"], 0.2)
        self.assertEqual(
            MODULE.excluded_sync_score("payload")["exclusion_reason"],
            "clean_framesync_failed",
        )

    def test_paired_rfsr_native_comparison_uses_common_sync_subset(self) -> None:
        def result(errors: int, *, included: bool = True) -> dict:
            return {
                "clean_sync": {"synchronized": included},
                "demod_after_noise": {"header_valid": included},
                "score": {
                    "included": included,
                    "symbol_count": 10 if included else 0,
                    "symbol_errors": errors if included else 0,
                    "complete": included,
                    "ser": float(errors / 10) if included else None,
                },
            }

        paired_row = {
            "physical_packet_uid": "packet-a",
            "methods": {
                "rfsr_1msps": result(1),
                "native_1msps": result(3),
            },
        }
        unpaired_row = {
            "physical_packet_uid": "packet-b",
            "methods": {
                "rfsr_1msps": result(0, included=False),
                "native_1msps": result(0),
            },
        }
        comparison = MODULE._paired_method_comparison(
            [paired_row, unpaired_row], "rfsr_1msps", "native_1msps"
        )
        self.assertEqual(comparison["common_clean_sync_packet_attempts"], 1)
        self.assertEqual(comparison["common_unique_physical_packets"], 1)
        self.assertAlmostEqual(comparison["left_ser"], 0.1)
        self.assertAlmostEqual(comparison["right_ser"], 0.3)
        self.assertAlmostEqual(
            comparison["ser_difference_left_minus_right"], -0.2
        )
        self.assertEqual(comparison["lower_ser_method"], "rfsr_1msps")

        aggregate = MODULE._aggregate_by_snr(
            [
                {
                    "extra_snr_db": -10.0,
                    "noise_seed": 1,
                    "packets": [paired_row],
                },
                {
                    "extra_snr_db": -10.0,
                    "noise_seed": 2,
                    "packets": [unpaired_row],
                },
            ],
            ["rfsr_1msps", "native_1msps"],
        )[0]
        self.assertEqual(aggregate["noise_seeds"], [1, 2])
        self.assertEqual(aggregate["unique_physical_packets"], 2)
        self.assertEqual(aggregate["packet_attempts"], 2)
        self.assertEqual(
            aggregate["paired_rfsr_vs_native"][
                "common_clean_sync_packet_attempts"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
