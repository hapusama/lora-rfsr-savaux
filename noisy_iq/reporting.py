"""Metadata and sweep summary writers.

中文说明：这一层只负责把每一步实验结果压平成 JSON / CSV，
不参与功率估计和 IQ 写入。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import csv_value, json_safe


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write strict JSON metadata with NumPy and NaN values normalized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False 保证输出是标准 JSON；NaN/Inf 会先被 json_safe 转成 null。
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(metadata), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def append_measurement_record(
    records: list[dict[str, Any]],
    *,
    kind: str,
    step_index: int,
    source_file: Path,
    output_file: Path,
    noise_power_db_relative: float | None,
    added_noise_power: float,
    seed: int | None,
    measurement: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Flatten one gr-lora_sdr measurement into a summary row."""
    summary = measurement.get("grlora_snr_db_summary", {})
    payload_check = measurement.get("payload_check", {}) or {}
    # CSV 保持一行一个 sweep step，便于后续 pandas/Excel 直接画曲线。
    records.append(
        {
            "kind": kind,
            "step_index": step_index,
            "source_file": str(source_file),
            "output_file": str(output_file),
            "noise_power_db_relative": noise_power_db_relative,
            "added_noise_power": float(added_noise_power),
            "seed": "" if seed is None else int(seed),
            "detected_packets": int(measurement.get("detected_packets", 0)),
            "decoded_payload_packets": int(measurement.get("decoded_payload_packets", 0)),
            "expected_packet_count": int(payload_check.get("expected_packet_count", 0)),
            "crc_valid_packets": int(payload_check.get("crc_valid_packets", 0)),
            "crc_invalid_packets": int(payload_check.get("crc_invalid_packets", 0)),
            "correct_payload_packets": int(payload_check.get("correct_payload_packets", 0)),
            "wrong_payload_packets": int(payload_check.get("wrong_payload_packets", 0)),
            "missed_detection_packets": int(payload_check.get("missed_detection_packets", 0)),
            "missed_correct_payload_packets": int(payload_check.get("missed_correct_payload_packets", 0)),
            "compared_payload_packets": int(payload_check.get("compared_payload_packets", 0)),
            "compared_byte_count": int(payload_check.get("compared_byte_count", 0)),
            "compared_bit_count": int(payload_check.get("compared_bit_count", 0)),
            "byte_error_count": int(payload_check.get("byte_error_count", 0)),
            "bit_error_count": int(payload_check.get("bit_error_count", 0)),
            "ber": payload_check.get("ber", float("nan")),
            "all_expected_payloads_correct": bool(payload_check.get("all_expected_payloads_correct", False)),
            "grlora_snr_count": summary.get("count", 0),
            "grlora_snr_mean": summary.get("mean", float("nan")),
            "grlora_snr_median": summary.get("median", float("nan")),
            "grlora_snr_std": summary.get("std", float("nan")),
            "grlora_snr_min": summary.get("min", float("nan")),
            "grlora_snr_max": summary.get("max", float("nan")),
            "sf": int(args.sf),
            "preamble_len": int(args.preamble_len),
            "samp_rate": float(args.samp_rate),
            "bw": float(args.bw),
            "sync_word": f"0x{int(args.sync_word):02x}",
        }
    )


def write_sweep_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a compact one-row-per-step CSV for Excel or pandas."""
    fieldnames = [
        "kind",
        "step_index",
        "source_file",
        "output_file",
        "noise_power_db_relative",
        "added_noise_power",
        "seed",
        "detected_packets",
        "decoded_payload_packets",
        "expected_packet_count",
        "crc_valid_packets",
        "crc_invalid_packets",
        "correct_payload_packets",
        "wrong_payload_packets",
        "missed_detection_packets",
        "missed_correct_payload_packets",
        "compared_payload_packets",
        "compared_byte_count",
        "compared_bit_count",
        "byte_error_count",
        "bit_error_count",
        "ber",
        "all_expected_payloads_correct",
        "grlora_snr_count",
        "grlora_snr_mean",
        "grlora_snr_median",
        "grlora_snr_std",
        "grlora_snr_min",
        "grlora_snr_max",
        "sf",
        "preamble_len",
        "samp_rate",
        "bw",
        "sync_word",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_value(record.get(key, "")) for key in fieldnames})


@dataclass
class SweepReporter:
    """Owns summary records and writes JSON/CSV artifacts.

    runner 只管调用 append/write_summary，不需要知道 CSV 字段细节。
    """

    records: list[dict[str, Any]] = field(default_factory=list)

    def append(
        self,
        *,
        kind: str,
        step_index: int,
        source_file: Path,
        output_file: Path,
        noise_power_db_relative: float | None,
        added_noise_power: float,
        seed: int | None,
        measurement: dict[str, Any],
        args: argparse.Namespace,
    ) -> None:
        append_measurement_record(
            self.records,
            kind=kind,
            step_index=step_index,
            source_file=source_file,
            output_file=output_file,
            noise_power_db_relative=noise_power_db_relative,
            added_noise_power=added_noise_power,
            seed=seed,
            measurement=measurement,
            args=args,
        )

    def write_summary(
        self,
        *,
        summary_json: Path,
        summary_csv: Path,
        metadata: dict[str, Any],
    ) -> None:
        write_metadata(summary_json, {**metadata, "records": self.records})
        write_sweep_summary_csv(summary_csv, self.records)
