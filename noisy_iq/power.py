"""Reference-power and gr-lora_sdr SNR measurement logic.

中文说明：这里负责“用什么功率作为加噪参考”和“如何复用 gr-lora_sdr
检测结果做 SNR / payload 统计”。真正写 IQ 文件的逻辑不在这里。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .capture import compare_payloads, expected_payloads_from_args
from .detector import GrloraPacketDetector
from .iq_file import (
    estimate_block_powers,
    mean_power,
    mean_power_chunked,
    mean_power_outside_ranges,
    normalize_ranges,
    sum_power,
)
from .utils import db10, summarize_values


@dataclass
class GrloraMeasurementService:
    """Run packet detection and convert packet metadata into summary measurements.

    这个服务只做测量汇总，方便 runner 复用 clean 文件和 noisy 文件的同一套逻辑。
    """

    args: argparse.Namespace

    def detect(self, path: Path) -> list[dict[str, Any]]:
        return GrloraPacketDetector(self.args).detect(path.resolve())

    def measure_snr(self, path: Path) -> dict[str, Any]:
        verify_args = argparse.Namespace(**vars(self.args))
        verify_args.input = path
        packets = GrloraPacketDetector(verify_args).detect(path.resolve())
        return self.measurement_from_packets(path, packets)

    def measurement_from_packets(self, path: Path, packets: list[dict[str, Any]]) -> dict[str, Any]:
        # frame_sync 给出的 snr_db 是 gr-lora_sdr 内部估计值，这里只做有限值统计。
        snr_values = [float(packet.get("grlora_snr_db", float("nan"))) for packet in packets]
        decoded_packets = [
            packet
            for packet in packets
            if packet.get("decoded_payload_available") or packet.get("decoded_payload_hex")
        ]
        expected_payloads = expected_payloads_from_args(self.args)
        payload_check = compare_payloads(decoded_packets, expected_payloads) if expected_payloads else {}
        if payload_check:
            payload_check["missed_detection_packets"] = int(
                max(0, payload_check["expected_packet_count"] - len(packets))
            )
        return {
            "file": str(path.resolve()),
            "detected_packets": int(len(packets)),
            "decoded_payload_packets": int(len(decoded_packets)),
            "grlora_snr_db_summary": summarize_values(snr_values),
            "payload_check": payload_check,
            "packet_measurements": packets,
        }

    @staticmethod
    def clean_measurement_from_packet_power(input_path: Path, power_info: dict[str, Any]) -> dict[str, Any]:
        """Reuse packet-mode detection results so the clean input is not decoded twice."""
        packets = power_info.get("packet_ranges", [])
        return {
            "file": str(input_path.resolve()),
            "detected_packets": int(power_info.get("packet_count", 0)),
            "decoded_payload_packets": int(
                sum(
                    1
                    for packet in packets
                    if packet.get("decoded_payload_available") or packet.get("decoded_payload_hex")
                )
            ),
            "grlora_snr_db_summary": power_info.get("grlora_snr_db_summary", summarize_values([])),
            "payload_check": power_info.get("payload_check", {}),
            "packet_measurements": power_info.get("packet_ranges", []),
        }


@dataclass
class ReferencePowerEstimator:
    """Estimate the signal reference power used to scale added noise.

    支持 total、active、window、packet 四种模式；输出的 signal_power
    只用于缩放“额外加入的噪声”，不等价于目标 SNR。
    """

    args: argparse.Namespace
    measurement_service: GrloraMeasurementService | None = None

    def __post_init__(self) -> None:
        if self.measurement_service is None:
            self.measurement_service = GrloraMeasurementService(self.args)

    def estimate(self, samples: np.ndarray) -> dict[str, Any]:
        total_power = mean_power_chunked(samples, self.args.chunk_samples, self.args.sample_limit)

        if self.args.power_mode == "packet":
            # packet 模式最严谨：先用 gr-lora_sdr 找包边界，再按完整包范围估计功率。
            packet_power = self._estimate_packet_power(samples)
            packet_power.update(
                {
                    "total_power": float(total_power),
                    "total_power_db": db10(float(total_power)),
                    "noise_floor_power": float(packet_power["existing_noise_power"]),
                    "noise_floor_power_db": packet_power["existing_noise_power_db"],
                    "active_mean_power": float(packet_power["packet_mean_power"]),
                    "active_mean_power_db": packet_power["packet_mean_power_db"],
                    "active_blocks": 0,
                    "total_blocks": 0,
                }
            )
            return packet_power

        if self.args.power_mode == "total":
            # total 模式最快，不区分包和空闲间隔，适合 dry-run 或粗略 sweep。
            signal_power = total_power
            noise_power = 0.0 if self.args.ignore_existing_noise else float("nan")
            current_snr_db = float("nan")
            active_blocks = 0
            total_blocks = 0
            noise_floor = float("nan")
            active_mean_power = total_power
        elif self.args.power_mode == "window":
            if self.args.signal_start is None or self.args.signal_samples is None:
                raise ValueError("--power-mode window requires --signal-start and --signal-samples.")
            start = max(0, int(self.args.signal_start))
            stop = min(samples.size, start + int(self.args.signal_samples))
            if stop <= start:
                raise ValueError("The requested signal window is empty.")
            block_powers = estimate_block_powers(samples, self.args.block_samples, self.args.sample_limit)
            noise_floor = float(np.percentile(block_powers, self.args.noise_percentile))
            # window 模式由用户指定包所在窗口，仍用低分位块功率估计已有底噪。
            active_mean_power = mean_power(samples[start:stop])
            noise_power = 0.0 if self.args.ignore_existing_noise else noise_floor
            signal_power = active_mean_power if self.args.ignore_existing_noise else active_mean_power - noise_power
            current_snr_db = db10(signal_power / noise_power) if noise_power > 0.0 else float("nan")
            active_blocks = 1
            total_blocks = int(block_powers.size)
        else:
            block_powers = estimate_block_powers(samples, self.args.block_samples, self.args.sample_limit)
            noise_floor = float(np.percentile(block_powers, self.args.noise_percentile))
            threshold = noise_floor * (10.0 ** (self.args.active_threshold_db / 10.0))
            # active 模式把明显高于底噪的块视为“有包/有信号”的块。
            active = block_powers > threshold
            active_blocks = int(np.count_nonzero(active))
            total_blocks = int(block_powers.size)
            if active_blocks == 0:
                active_mean_power = total_power
                signal_power = total_power if self.args.ignore_existing_noise else max(total_power - noise_floor, 0.0)
            else:
                active_mean_power = float(np.mean(block_powers[active], dtype=np.float64))
                signal_power = active_mean_power if self.args.ignore_existing_noise else active_mean_power - noise_floor
            noise_power = 0.0 if self.args.ignore_existing_noise else noise_floor
            current_snr_db = db10(signal_power / noise_power) if noise_power > 0.0 else float("nan")

        if not np.isfinite(signal_power) or signal_power <= 0.0:
            raise ValueError(
                "Estimated non-positive signal power. Try --power-mode total, "
                "--ignore-existing-noise, or a manual --power-mode window."
            )

        return {
            "power_mode": self.args.power_mode,
            "total_power": float(total_power),
            "total_power_db": db10(float(total_power)),
            "signal_power": float(signal_power),
            "signal_power_db": db10(float(signal_power)),
            "existing_noise_power": float(noise_power),
            "existing_noise_power_db": db10(float(noise_power)) if np.isfinite(noise_power) else float("nan"),
            "current_snr_db": float(current_snr_db),
            "noise_floor_power": float(noise_floor),
            "noise_floor_power_db": db10(float(noise_floor)) if np.isfinite(noise_floor) else float("nan"),
            "active_mean_power": float(active_mean_power),
            "active_mean_power_db": db10(float(active_mean_power)),
            "active_blocks": int(active_blocks),
            "total_blocks": int(total_blocks),
        }

    def _estimate_packet_power(self, samples: np.ndarray) -> dict[str, Any]:
        packets = self.measurement_service.detect(Path(self.args.input).resolve())
        limit = samples.size if self.args.sample_limit is None else min(samples.size, int(self.args.sample_limit))
        # 先裁剪/合并包范围，再计算包内和包外功率，避免重叠包重复计数。
        packet_ranges = normalize_ranges(
            [(packet["packet_start_sample"], packet["packet_end_sample"]) for packet in packets],
            limit,
        )
        if not packet_ranges:
            raise ValueError(
                "gr-lora_sdr did not publish any valid packet ranges. "
                "Check --sf/--samp-rate/--bw/--sync-word/--preamble-len, or use --power-mode active/window."
            )

        packet_power_sum = 0.0
        packet_sample_count = 0
        packet_records = []
        for packet in packets:
            start = max(0, min(int(packet["packet_start_sample"]), limit))
            end = max(start, min(int(packet["packet_end_sample"]), limit))
            if end <= start:
                continue
            current_sum, current_count = sum_power(samples[start:end])
            current_power = current_sum / current_count if current_count else float("nan")
            packet_power_sum += current_sum
            packet_sample_count += current_count
            record = dict(packet)
            record["packet_start_sample"] = int(start)
            record["packet_end_sample"] = int(end)
            record["packet_samples"] = int(current_count)
            record["packet_mean_power"] = float(current_power)
            record["packet_mean_power_db"] = db10(float(current_power))
            packet_records.append(record)

        if packet_sample_count == 0:
            raise ValueError("Detected packet ranges are empty after applying --sample-limit.")

        packet_mean_power = packet_power_sum / packet_sample_count
        outside_power, outside_count = mean_power_outside_ranges(samples, packet_ranges, self.args.sample_limit)
        noise_power = 0.0 if self.args.ignore_existing_noise or not np.isfinite(outside_power) else outside_power
        signal_power = packet_mean_power if self.args.ignore_existing_noise else packet_mean_power - noise_power
        if signal_power <= 0.0:
            raise ValueError(
                "Packet-level signal power is non-positive after subtracting packet-outside noise. "
                "Try --ignore-existing-noise or inspect detected packet ranges."
            )

        grlora_snr_values = [float(packet.get("grlora_snr_db", float("nan"))) for packet in packet_records]
        packet_power_values = [float(packet.get("packet_mean_power", float("nan"))) for packet in packet_records]
        decoded_packets = [
            packet
            for packet in packet_records
            if packet.get("decoded_payload_available") or packet.get("decoded_payload_hex")
        ]
        expected_payloads = expected_payloads_from_args(self.args)
        payload_check = compare_payloads(decoded_packets, expected_payloads) if expected_payloads else {}
        if payload_check:
            payload_check["missed_detection_packets"] = int(
                max(0, payload_check["expected_packet_count"] - len(packet_records))
            )
        current_snr_db = db10(signal_power / noise_power) if noise_power > 0.0 else float("nan")

        return {
            "power_mode": "packet",
            "signal_power": float(signal_power),
            "signal_power_db": db10(float(signal_power)),
            "existing_noise_power": float(noise_power),
            "existing_noise_power_db": db10(float(noise_power)) if np.isfinite(noise_power) else float("nan"),
            "current_snr_db": float(current_snr_db),
            "packet_mean_power": float(packet_mean_power),
            "packet_mean_power_db": db10(float(packet_mean_power)),
            "packet_count": int(len(packet_records)),
            "packet_total_samples": int(packet_sample_count),
            "outside_noise_samples": int(outside_count),
            "packet_power_summary": summarize_values(packet_power_values),
            "grlora_snr_db_summary": summarize_values(grlora_snr_values),
            "payload_check": payload_check,
            "packet_ranges": packet_records,
        }


def estimate_reference_power(samples: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    return ReferencePowerEstimator(args).estimate(samples)


def measure_grlora_snr(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    return GrloraMeasurementService(args).measure_snr(path)


def clean_measurement_from_packet_power(input_path: Path, power_info: dict[str, Any]) -> dict[str, Any]:
    return GrloraMeasurementService.clean_measurement_from_packet_power(input_path, power_info)
