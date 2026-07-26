"""Top-level orchestration for noisy IQ sweeps.

中文说明：NoisyIqSweep 是总调度对象，负责串起参数解析后的流程：
读 IQ -> 估计参考功率 -> 逐步加噪 -> 调 gr-lora_sdr 测量 -> 写汇总。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np

from .capture import (
    build_noise_power_db_values,
    check_output_collisions,
    decoded_payload_hexes_from_packets,
    normalize_payload_hexes,
    planned_output_paths,
    resolve_capture_parameters,
    validate_capture_args,
)
from .constants import DEFAULT_OUTPUT_ROOT
from .iq_file import IqCapture
from .power import GrloraMeasurementService, ReferencePowerEstimator
from .reporting import SweepReporter, write_metadata
from .utils import db10, format_db_value


@dataclass
class NoisyIqSweep:
    """Generate noisy IQ files and measure them with gr-lora_sdr.

    这个类尽量只做流程编排，把具体算法委托给 capture/iq_file/power/reporting。
    """

    args: argparse.Namespace
    groundtruth_info: dict[str, Any] = field(default_factory=dict)

    def run(self) -> int:
        input_path = Path(self.args.input).resolve()
        self.args.input = input_path
        # 先把所有参数解析成稳定状态，后面的 detector/power 都读取同一个 args。
        capture_metadata = resolve_capture_parameters(self.args, input_path)
        noise_power_db_values = build_noise_power_db_values(self.args)
        output_dir = self._resolve_output_dir(input_path)

        validate_capture_args(self.args)
        outputs = planned_output_paths(input_path, output_dir, noise_power_db_values)
        if not self.args.dry_run:
            # 长时间跑 GNU Radio 前先检查输出冲突，避免最后一步才发现文件已存在。
            check_output_collisions(outputs, self.args.overwrite, input_path)

        with IqCapture.open(input_path) as capture:
            process_samples = capture.processed_sample_count(self.args.sample_limit)
            measurement_service = GrloraMeasurementService(self.args)
            clean_groundtruth_measurement = self._prepare_groundtruth(measurement_service, input_path)
            # signal_power 是后续每个噪声步的缩放基准，不直接代表最终文件 SNR。
            power_info = ReferencePowerEstimator(self.args, measurement_service).estimate(capture.samples)
            noise_reference_power = float(power_info["signal_power"])
            if not np.isfinite(noise_reference_power) or noise_reference_power <= 0.0:
                raise ValueError("Estimated non-positive reference power; cannot scale added noise.")

            self._print_header(
                input_path,
                output_dir,
                capture.samples.size,
                process_samples,
                capture_metadata,
                power_info,
                noise_reference_power,
                noise_power_db_values,
            )

            reporter = SweepReporter()
            if not self.args.dry_run:
                # 非 dry-run 会把 clean 输入也测一遍，作为 noisy 文件的对照行。
                self._measure_clean_input(
                    reporter,
                    measurement_service,
                    input_path,
                    power_info,
                    clean_groundtruth_measurement,
                )

            for index, (noise_power_db, out_path, meta_path) in enumerate(outputs, start=1):
                self._run_step(
                    index=index,
                    noise_power_db=noise_power_db,
                    out_path=out_path,
                    meta_path=meta_path,
                    capture=capture,
                    input_path=input_path,
                    process_samples=process_samples,
                    capture_metadata=capture_metadata,
                    power_info=power_info,
                    noise_reference_power=noise_reference_power,
                    measurement_service=measurement_service,
                    reporter=reporter,
                )

            if not self.args.dry_run:
                summary_json = output_dir / f"{input_path.stem}_noise_sweep_summary.json"
                summary_csv = output_dir / f"{input_path.stem}_noise_sweep_summary.csv"
                # 汇总文件保留全局配置和每一步测量记录，方便之后复现实验。
                reporter.write_summary(
                    summary_json=summary_json,
                    summary_csv=summary_csv,
                    metadata={
                        "input_file": str(input_path),
                        "output_dir": str(output_dir),
                        "capture_metadata": capture_metadata,
                        "noise_power_db_values": noise_power_db_values,
                        "noise_reference_power": float(noise_reference_power),
                        "noise_reference_power_db": db10(float(noise_reference_power)),
                        "noise_reference_mode": self.args.power_mode,
                        "groundtruth": self.groundtruth_info,
                    },
                )
                print(f"Summary JSON: {summary_json}")
                print(f"Summary CSV:  {summary_csv}")

        return 0

    def _resolve_output_dir(self, input_path: Path) -> Path:
        if self.args.output_dir is not None:
            return Path(self.args.output_dir).resolve()
        return (DEFAULT_OUTPUT_ROOT / input_path.stem).resolve()

    def _prepare_groundtruth(
        self,
        measurement_service: GrloraMeasurementService,
        input_path: Path,
    ) -> dict[str, Any] | None:
        if self.args.no_expected_payload_check:
            self.args.expected_payload_source = "disabled"
            self.groundtruth_info = {
                "mode": "disabled",
                "source_file": str(input_path),
                "expected_payload_hex": [],
            }
            return None

        if self.args.expected_payload_hex is not None:
            payload_hexes = normalize_payload_hexes(list(self.args.expected_payload_hex))
            self.args.expected_payload_hex = payload_hexes
            self.args.expected_payload_source = "cli"
            self.groundtruth_info = {
                "mode": "cli",
                "source_file": str(input_path),
                "expected_packet_count": int(len(payload_hexes)),
                "expected_payload_hex": payload_hexes,
            }
            return None

        if self.args.dry_run:
            self.args.expected_payload_source = "dry_run_not_decoded"
            self.groundtruth_info = {
                "mode": "dry_run_not_decoded",
                "source_file": str(input_path),
                "expected_payload_hex": [],
            }
            return None

        print("[GROUNDTRUTH] decoding clean input without added noise...")
        packets = measurement_service.detect(input_path)
        decoded_packets = [
            packet
            for packet in packets
            if packet.get("decoded_payload_available") or packet.get("decoded_payload_hex")
        ]
        payload_hexes = decoded_payload_hexes_from_packets(decoded_packets)
        detected_count = len(packets)
        decoded_count = len(decoded_packets)
        if not payload_hexes:
            raise RuntimeError(
                "Clean input did not produce any decoded payloads, so no groundtruth can be derived. "
                "Check --sf/--samp-rate/--bw/--sync-word/--preamble-len, or pass --expected-payload-hex."
            )
        if decoded_count != detected_count:
            raise RuntimeError(
                "Clean input groundtruth is incomplete: "
                f"detected {detected_count} packet(s) but decoded {decoded_count} payload(s). "
                "Fix the clean decode settings first, or pass --expected-payload-hex explicitly."
            )

        crc_valid_count = sum(1 for packet in decoded_packets if bool(packet.get("crc_valid", False)))
        self.args.expected_payload_hex = payload_hexes
        self.args.expected_payload_source = "clean_decode"
        self.groundtruth_info = {
            "mode": "clean_decode",
            "source_file": str(input_path),
            "detected_packets": int(detected_count),
            "decoded_payload_packets": int(decoded_count),
            "crc_valid_packets": int(crc_valid_count),
            "crc_invalid_packets": int(max(0, decoded_count - crc_valid_count)),
            "expected_packet_count": int(len(payload_hexes)),
            "expected_payload_hex": payload_hexes,
        }

        clean_measurement = measurement_service.measurement_from_packets(input_path, packets)
        payload_check = clean_measurement.get("payload_check", {}) or {}
        if not payload_check.get("all_expected_payloads_correct", False):
            raise RuntimeError("Clean decoded payloads could not be verified against the derived groundtruth.")

        print(
            "[GROUNDTRUTH] clean decode accepted: "
            f"detected={detected_count}, payloads={decoded_count}, "
            f"crc_valid={crc_valid_count}/{decoded_count}"
        )
        return clean_measurement

    def _print_header(
        self,
        input_path: Path,
        output_dir: Path,
        total_samples: int,
        process_samples: int,
        capture_metadata: dict[str, Any],
        power_info: dict[str, Any],
        noise_reference_power: float,
        noise_power_db_values: list[float],
    ) -> None:
        print(f"Input: {input_path}")
        print(f"Samples: {total_samples} total, {process_samples} to process")
        print(f"Output directory: {output_dir}")
        print(
            "Resolved capture parameters: "
            f"sf={self.args.sf} ({capture_metadata['resolved_sf_source']}), "
            f"preamble_len={self.args.preamble_len} ({capture_metadata['resolved_preamble_len_source']}), "
            f"samp_rate={self.args.samp_rate:.0f}, bw={self.args.bw:.0f}, "
            f"sync_word=0x{int(self.args.sync_word):02x}"
        )
        print(
            "Noise-step reference: "
            f"mode={self.args.power_mode}, power={noise_reference_power:.6e} "
            f"({db10(noise_reference_power):.2f} dB)"
        )
        if self.groundtruth_info:
            print(
                "Groundtruth: "
                f"mode={self.groundtruth_info.get('mode', '')}, "
                f"expected_packets={self.groundtruth_info.get('expected_packet_count', 0)}"
            )
        if self.args.power_mode == "packet":
            grlora_snr = power_info.get("grlora_snr_db_summary", {})
            print(
                "Packet power estimate: "
                f"{power_info.get('packet_count', 0)} packet(s), "
                f"packet_mean={power_info.get('packet_mean_power', float('nan')):.6e} "
                f"({power_info.get('packet_mean_power_db', float('nan')):.2f} dB), "
                f"grlora_snr_median={grlora_snr.get('median', float('nan')):.2f} dB"
            )
        if power_info["total_blocks"]:
            print(f"Active blocks: {power_info['active_blocks']}/{power_info['total_blocks']}")

        print(
            "Noise steps (relative dB, not target SNR): "
            + ", ".join(format_db_value(value) for value in noise_power_db_values)
        )

    def _measure_clean_input(
        self,
        reporter: SweepReporter,
        measurement_service: GrloraMeasurementService,
        input_path: Path,
        power_info: dict[str, Any],
        precomputed_measurement: dict[str, Any] | None = None,
    ) -> None:
        if precomputed_measurement is not None:
            clean_measurement = precomputed_measurement
        else:
            clean_measurement = (
                measurement_service.clean_measurement_from_packet_power(input_path, power_info)
                if self.args.power_mode == "packet"
                else measurement_service.measure_snr(input_path)
            )
        clean_summary = clean_measurement["grlora_snr_db_summary"]
        reporter.append(
            kind="clean",
            step_index=0,
            source_file=input_path,
            output_file=input_path,
            noise_power_db_relative=None,
            added_noise_power=0.0,
            seed=None,
            measurement=clean_measurement,
            args=self.args,
        )
        clean_payload = clean_measurement.get("payload_check", {}) or {}
        print(
            "[MEASURE] clean input: "
            f"detected={clean_measurement['detected_packets']}, "
            f"decoded={clean_measurement.get('decoded_payload_packets', 0)}, "
            f"correct={clean_payload.get('correct_payload_packets', 0)}/"
            f"{clean_payload.get('expected_packet_count', 0)}, "
            f"wrong={clean_payload.get('wrong_payload_packets', 0)}, "
            f"miss_detect={clean_payload.get('missed_detection_packets', 0)}, "
            f"ber={float(clean_payload.get('ber', float('nan'))):.3g}, "
            f"grlora_snr_median={clean_summary['median']:.2f} dB"
        )

    def _run_step(
        self,
        *,
        index: int,
        noise_power_db: float,
        out_path: Path,
        meta_path: Path,
        capture: IqCapture,
        input_path: Path,
        process_samples: int,
        capture_metadata: dict[str, Any],
        power_info: dict[str, Any],
        noise_reference_power: float,
        measurement_service: GrloraMeasurementService,
        reporter: SweepReporter,
    ) -> None:
        # 命令行里的 noise_power_db 是“相对参考信号功率”的加噪强度，不是目标 SNR。
        added_noise_power = noise_reference_power * (10.0 ** (float(noise_power_db) / 10.0))
        seed = self.args.seed + (index - 1) if self.args.independent_noise else self.args.seed
        metadata = self._build_step_metadata(
            input_path=input_path,
            out_path=out_path,
            noise_power_db=noise_power_db,
            noise_reference_power=noise_reference_power,
            added_noise_power=added_noise_power,
            seed=seed,
            process_samples=process_samples,
            capture_metadata=capture_metadata,
            power_info=power_info,
        )

        status = "DRY-RUN" if self.args.dry_run else "WRITE"
        print(
            f"[{status}] step {index:02d}, noise_rel={format_db_value(noise_power_db):>8} dB -> {out_path.name}; "
            f"add_noise_power={added_noise_power:.6e}"
        )
        if self.args.dry_run:
            return

        # 先生成 noisy bin，再用同一套 gr-lora_sdr 测量逻辑读回验证。
        capture.write_noisy(
            out_path,
            added_noise_power,
            seed,
            self.args.chunk_samples,
            self.args.sample_limit,
            self.args.overwrite,
        )
        measurement = measurement_service.measure_snr(out_path)
        metadata["grlora_snr_measurement"] = measurement
        summary = measurement["grlora_snr_db_summary"]
        payload_check = measurement.get("payload_check", {}) or {}
        reporter.append(
            kind="noisy",
            step_index=index,
            source_file=input_path,
            output_file=out_path,
            noise_power_db_relative=float(noise_power_db),
            added_noise_power=float(added_noise_power),
            seed=seed,
            measurement=measurement,
            args=self.args,
        )
        print(
            "[MEASURE] "
            f"{out_path.name}: detected={measurement['detected_packets']}, "
            f"decoded={measurement.get('decoded_payload_packets', 0)}, "
            f"correct={payload_check.get('correct_payload_packets', 0)}/"
            f"{payload_check.get('expected_packet_count', 0)}, "
            f"wrong={payload_check.get('wrong_payload_packets', 0)}, "
            f"miss_detect={payload_check.get('missed_detection_packets', 0)}, "
            f"ber={float(payload_check.get('ber', float('nan'))):.3g}, "
            f"grlora_snr_median={summary['median']:.2f} dB"
        )
        write_metadata(meta_path, metadata)

    def _build_step_metadata(
        self,
        *,
        input_path: Path,
        out_path: Path,
        noise_power_db: float,
        noise_reference_power: float,
        added_noise_power: float,
        seed: int,
        process_samples: int,
        capture_metadata: dict[str, Any],
        power_info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "input_file": str(input_path),
            "output_file": str(out_path),
            "format": "raw numpy.complex64 / GNU Radio gr_complex",
            "noise_power_db_relative": float(noise_power_db),
            "noise_reference_power": float(noise_reference_power),
            "noise_reference_power_db": db10(float(noise_reference_power)),
            "noise_reference_mode": self.args.power_mode,
            "added_noise_power": float(added_noise_power),
            "added_noise_sigma_per_iq_component": math.sqrt(added_noise_power / 2.0)
            if added_noise_power > 0.0
            else 0.0,
            "seed": int(seed),
            "sample_limit": self.args.sample_limit,
            "processed_samples": int(process_samples),
            "capture_metadata": capture_metadata,
            "groundtruth": self.groundtruth_info,
            "args": {
                "power_mode": self.args.power_mode,
                "sf": self.args.sf,
                "bw": self.args.bw,
                "samp_rate": self.args.samp_rate,
                "cr": self.args.cr,
                "pay_len": self.args.pay_len,
                "has_crc": self.args.has_crc,
                "impl_head": self.args.impl_head,
                "center_freq": self.args.center_freq,
                "sync_word": self.args.sync_word,
                "preamble_len": self.args.preamble_len,
                "ldro_mode": self.args.ldro_mode,
                "crc_mode": self.args.crc_mode,
                "expected_payload_hex": self.args.expected_payload_hex,
                "expected_payload_source": getattr(self.args, "expected_payload_source", ""),
                "no_expected_payload_check": self.args.no_expected_payload_check,
                "noise_percentile": self.args.noise_percentile,
                "active_threshold_db": self.args.active_threshold_db,
                "block_samples": self.args.block_samples,
                "chunk_samples": self.args.chunk_samples,
                "ignore_existing_noise": self.args.ignore_existing_noise,
                "independent_noise": self.args.independent_noise,
                "noise_power_db": self.args.noise_power_db,
                "noise_start_db": self.args.noise_start_db,
                "noise_stop_db": self.args.noise_stop_db,
                "noise_step_db": self.args.noise_step_db,
            },
            "power_estimate": power_info,
        }


def main(args: argparse.Namespace) -> int:
    """Run a noisy IQ sweep from parsed CLI arguments."""
    return NoisyIqSweep(args).run()
