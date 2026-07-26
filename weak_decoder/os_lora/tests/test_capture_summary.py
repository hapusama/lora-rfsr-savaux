"""真实 capture 汇总表的字段与数值回归测试。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from weak_decoder.os_lora.experiments.evaluate_real_capture_gls import (
    _capture_summary_row,
    _capture_sync_metrics,
    _packet_gt_snr_rows,
    _upsert_capture_summary,
)


class CaptureSummaryTests(unittest.TestCase):
    """验证同步汇总、GT-bin 包级 SNR 和跨 capture 表更新行为。"""

    def test_sync_metrics(self) -> None:
        """检测次数、strict 次数和秒时刻应来自同一批同步记录。"""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sync.csv"
            path.write_text(
                "detected_start_sample,grlora_framesync_valid\n"
                "10,1\n"
                "30,0\n"
                "50,true\n",
                encoding="utf-8",
            )
            metrics = _capture_sync_metrics(path, sample_rate=10.0)
        self.assertEqual(3, metrics["detection_count"])
        self.assertEqual(2, metrics["strict_sync_count"])
        self.assertEqual("1.000000|3.000000|5.000000", metrics["packet_times_s"])
        self.assertEqual("1.000000|5.000000", metrics["strict_packet_times_s"])

    def test_packet_gt_snr_uses_energy_weighted_symbols(self) -> None:
        """包级 SNR 应先累加 GT 与剩余 bin 能量，再计算二者的 dB 比值。"""

        symbols = [
            {
                "packet_index": 3,
                "frame_index": 7,
                "payload_symbol_index": 0,
                "start_sample": 100,
                "dechirp_gt_bin_power": 4.0,
                "dechirp_residual_fft_energy": 1.0,
                "dechirp_total_fft_energy": 5.0,
            },
            {
                "packet_index": 3,
                "frame_index": 7,
                "payload_symbol_index": 1,
                "start_sample": 108,
                "dechirp_gt_bin_power": 6.0,
                "dechirp_residual_fft_energy": 3.0,
                "dechirp_total_fft_energy": 9.0,
            },
        ]
        rows = _packet_gt_snr_rows(symbols, capture="low_test", sample_rate=10.0)
        self.assertEqual(1, len(rows))
        self.assertEqual(2, rows[0]["symbol_count"])
        self.assertAlmostEqual(10.0 / 14.0, rows[0]["dechirp_gt_energy_ratio"])
        self.assertAlmostEqual(10.0 * np.log10(10.0 / 4.0), rows[0]["dechirp_gt_packet_snr_db"])

    def test_capture_row_and_upsert(self) -> None:
        """三种 SER 应进入同一行，重复组合应更新而不是追加。"""

        args = argparse.Namespace(
            test_name="low_test",
            training_name="low_train",
            summary_gls_method="gls_crossfit",
        )
        summaries = [
            {"method": "ordinary_fft", "frame_count": 2, "symbol_count": 4, "errors": 1, "ser": 0.25},
            {"method": "savaux", "frame_count": 2, "symbol_count": 4, "errors": 0, "ser": 0.0},
            {"method": "gls_crossfit", "frame_count": 2, "symbol_count": 4, "errors": 0, "ser": 0.0},
            {"method": "gls_offpacket", "frame_count": 2, "symbol_count": 4, "errors": 1, "ser": 0.25},
        ]
        packet_snr_rows = [
            {"dechirp_gt_packet_snr_db": -12.0},
            {"dechirp_gt_packet_snr_db": -8.0},
        ]
        row = _capture_summary_row(
            args=args,
            summary_rows=summaries,
            symbol_rows=[{"dechirp_pnr_db": 10.0}, {"dechirp_pnr_db": 14.0}],
            sync_metrics={
                "packet_times_s": "1.0|2.0",
                "strict_packet_times_s": "1.0",
                "detection_count": 2,
                "strict_sync_count": 1,
            },
            packet_snr_rows=packet_snr_rows,
            sample_rate=8.0,
            bandwidth=2.0,
            capture_sample_count=80,
        )
        self.assertEqual("1/4", row["fft_errors"])
        self.assertEqual(0.25, row["fft_ser"])
        self.assertEqual(0.0, row["savaux_ser"])
        self.assertEqual(0.0, row["gls_ser"])
        self.assertEqual(12.0, row["dechirp_pnr_db"])
        self.assertEqual(-10.0, row["dechirp_gt_packet_snr_db"])
        self.assertEqual("-12.000000|-8.000000", row["dechirp_gt_packet_snr_values_db"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "captures.csv"
            # 模拟旧版汇总表，更新后旧 SNR 字段应从表头中彻底移除。
            path.write_text(
                "capture,training_capture,gls_method,input_inband_snr_db\n"
                "low_test,low_train,gls_crossfit,-10.0\n",
                encoding="utf-8",
            )
            _upsert_capture_summary(path, row)
            changed = dict(row)
            changed["gls_ser"] = 0.5
            _upsert_capture_summary(path, changed)
            with path.open("r", encoding="utf-8", newline="") as handle:
                saved = list(csv.DictReader(handle))
        self.assertEqual(1, len(saved))
        self.assertEqual("0.5", saved[0]["gls_ser"])
        self.assertNotIn("input_inband_snr_db", saved[0])


if __name__ == "__main__":
    unittest.main()
