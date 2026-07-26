#!/usr/bin/env python3
"""Generate near-threshold IQ files for decode failure and detection failure.

中文说明：
这个脚本复用 ``noisy_iq`` 包里的加噪、参考功率估计和 gr-lora_sdr 测量逻辑，
自动搜索两个“刚好失败”的噪声点：

- decode failure：还能检测到一些包，但预期 payload 已经不能全部正确恢复。
- detect failure：已经完全检测不到包。

最终会分别保存两个 raw complex64 二进制 IQ 文件，并写同名 JSON 记录搜索结果。
"""
# python .\gr-lora_sdr\weakPacket_decoding\scripts\experiments\make_failure_limit_iq.py --samp-rate 500000 --bw 125000 --sync-word 0x34 --coarse-start-db -30 --coarse-stop-db 20 --coarse-step-db 2 --refine-iterations 6 --overwrite

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


WEAKPACKET_ROOT = Path(__file__).resolve().parents[2]
if str(WEAKPACKET_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAKPACKET_ROOT))

from noisy_iq.capture import (
    expected_payloads_from_args,
    resolve_capture_parameters,
    validate_capture_args,
)
from noisy_iq.constants import DEFAULT_INPUT, DEFAULT_OUTPUT_ROOT
from noisy_iq.iq_file import IqCapture
from noisy_iq.power import GrloraMeasurementService, ReferencePowerEstimator
from noisy_iq.reporting import write_metadata
from noisy_iq.utils import db10, db_to_label, format_db_value, parse_int_auto


@dataclass
class SearchRecord:
    """One measured noise point during the threshold search."""

    noise_power_db_relative: float
    added_noise_power: float
    seed: int
    measurement: dict[str, Any]
    detected_packets: int
    expected_packets: int
    decoded_payload_packets: int
    correct_payload_packets: int
    full_detection_ok: bool
    detection_ok: bool
    decode_ok: bool

    @property
    def grlora_snr_median(self) -> float:
        summary = self.measurement.get("grlora_snr_db_summary", {}) or {}
        return float(summary.get("median", float("nan")))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "noise_power_db_relative": float(self.noise_power_db_relative),
            "added_noise_power": float(self.added_noise_power),
            "seed": int(self.seed),
            "detected_packets": int(self.detected_packets),
            "expected_packets": int(self.expected_packets),
            "decoded_payload_packets": int(self.decoded_payload_packets),
            "correct_payload_packets": int(self.correct_payload_packets),
            "full_detection_ok": bool(self.full_detection_ok),
            "detection_ok": bool(self.detection_ok),
            "decode_ok": bool(self.decode_ok),
            "grlora_snr_median": float(self.grlora_snr_median),
            "measurement": self.measurement,
        }


@dataclass
class FailureLimitSearcher:
    """Search and save noisy IQ files around failure thresholds."""

    args: argparse.Namespace

    def run(self) -> int:
        input_path = Path(self.args.input).resolve()
        self.args.input = input_path
        capture_metadata = resolve_capture_parameters(self.args, input_path)
        validate_capture_args(self.args)

        output_dir = self._resolve_output_dir(input_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._check_final_collisions(input_path, output_dir)

        with IqCapture.open(input_path) as capture:
            measurement_service = GrloraMeasurementService(self.args)
            power_info = ReferencePowerEstimator(self.args, measurement_service).estimate(capture.samples)
            reference_power = float(power_info["signal_power"])
            if not np.isfinite(reference_power) or reference_power <= 0.0:
                raise ValueError("Estimated non-positive reference power; cannot search thresholds.")

            expected_packets = len(expected_payloads_from_args(self.args))
            if expected_packets == 0:
                print(
                    "[WARN] No expected payload list is active; decode failure will use a weaker "
                    "decoded-packet-count rule. Passing --expected-payload-hex is recommended."
                )

            print(f"Input: {input_path}")
            print(f"Output directory: {output_dir}")
            print(
                "Resolved capture parameters: "
                f"sf={self.args.sf} ({capture_metadata['resolved_sf_source']}), "
                f"preamble_len={self.args.preamble_len} ({capture_metadata['resolved_preamble_len_source']}), "
                f"samp_rate={self.args.samp_rate:.0f}, bw={self.args.bw:.0f}, "
                f"sync_word=0x{int(self.args.sync_word):02x}"
            )
            print(
                "Noise reference: "
                f"mode={self.args.power_mode}, power={reference_power:.6e} "
                f"({db10(reference_power):.2f} dB)"
            )

            tmp_path = output_dir / f"{input_path.stem}_threshold_search_tmp.bin"
            history: list[SearchRecord] = []
            try:
                decode_bracket, detect_bracket = self._coarse_search(
                    capture=capture,
                    measurement_service=measurement_service,
                    tmp_path=tmp_path,
                    reference_power=reference_power,
                    expected_packets=expected_packets,
                    history=history,
                )

                decode_record = self._refine_boundary(
                    target="decode",
                    bracket=decode_bracket,
                    capture=capture,
                    measurement_service=measurement_service,
                    tmp_path=tmp_path,
                    reference_power=reference_power,
                    expected_packets=expected_packets,
                    history=history,
                )
                detect_record = self._refine_boundary(
                    target="detect",
                    bracket=detect_bracket,
                    capture=capture,
                    measurement_service=measurement_service,
                    tmp_path=tmp_path,
                    reference_power=reference_power,
                    expected_packets=expected_packets,
                    history=history,
                )

                self._save_limit_file(
                    kind="decode_failure",
                    record=decode_record,
                    capture=capture,
                    input_path=input_path,
                    output_dir=output_dir,
                    reference_power=reference_power,
                    power_info=power_info,
                    capture_metadata=capture_metadata,
                    history=history,
                )
                self._save_limit_file(
                    kind="detect_failure",
                    record=detect_record,
                    capture=capture,
                    input_path=input_path,
                    output_dir=output_dir,
                    reference_power=reference_power,
                    power_info=power_info,
                    capture_metadata=capture_metadata,
                    history=history,
                )
            finally:
                tmp_path.unlink(missing_ok=True)

        return 0

    def _resolve_output_dir(self, input_path: Path) -> Path:
        if self.args.output_dir is not None:
            return Path(self.args.output_dir).resolve()
        return (DEFAULT_OUTPUT_ROOT / input_path.stem / "failure_limits").resolve()

    def _check_final_collisions(self, input_path: Path, output_dir: Path) -> None:
        for kind in ("decode_failure", "detect_failure"):
            for suffix in (".bin", ".json"):
                path = output_dir / f"{input_path.stem}_{kind}_limit{suffix}"
                if path.exists() and not self.args.overwrite:
                    raise FileExistsError(f"{path} already exists; pass --overwrite to replace it.")

    def _coarse_search(
        self,
        *,
        capture: IqCapture,
        measurement_service: GrloraMeasurementService,
        tmp_path: Path,
        reference_power: float,
        expected_packets: int,
        history: list[SearchRecord],
    ) -> tuple[tuple[SearchRecord, SearchRecord], tuple[SearchRecord, SearchRecord]]:
        decode_bracket: tuple[SearchRecord, SearchRecord] | None = None
        detect_bracket: tuple[SearchRecord, SearchRecord] | None = None
        last_decode_good: SearchRecord | None = None
        last_detect_good: SearchRecord | None = None

        for noise_db in self._coarse_noise_values():
            record = self._evaluate_noise_point(
                noise_db=noise_db,
                capture=capture,
                measurement_service=measurement_service,
                tmp_path=tmp_path,
                reference_power=reference_power,
                expected_packets=expected_packets,
            )
            history.append(record)
            self._print_record("COARSE", record)

            if record.detection_ok:
                last_detect_good = record
            elif detect_bracket is None and last_detect_good is not None:
                detect_bracket = (last_detect_good, record)

            if record.decode_ok:
                last_decode_good = record
            elif decode_bracket is None and last_decode_good is not None:
                # 粗扫步长较大时，可能直接从“全解码成功”跳到“检测也失败”。
                # 仍然先建立 decode bracket，后续二分会尽量找“检测仍成功但解码失败”的点。
                decode_bracket = (last_decode_good, record)

            if decode_bracket is not None and detect_bracket is not None:
                return decode_bracket, detect_bracket

        if decode_bracket is None:
            raise RuntimeError(
                "Could not bracket decode failure. Try increasing --coarse-stop-db, "
                "or check that clean/low-noise packets decode correctly."
            )
        if detect_bracket is None:
            raise RuntimeError(
                "Could not bracket detection failure. Try increasing --coarse-stop-db."
            )
        return decode_bracket, detect_bracket

    def _refine_boundary(
        self,
        *,
        target: str,
        bracket: tuple[SearchRecord, SearchRecord],
        capture: IqCapture,
        measurement_service: GrloraMeasurementService,
        tmp_path: Path,
        reference_power: float,
        expected_packets: int,
        history: list[SearchRecord],
    ) -> SearchRecord:
        low, high = bracket
        best_failure = high
        best_detection_ok_failure: SearchRecord | None = None
        for _ in range(int(self.args.refine_iterations)):
            mid_db = (low.noise_power_db_relative + high.noise_power_db_relative) / 2.0
            record = self._evaluate_noise_point(
                noise_db=mid_db,
                capture=capture,
                measurement_service=measurement_service,
                tmp_path=tmp_path,
                reference_power=reference_power,
                expected_packets=expected_packets,
            )
            history.append(record)
            self._print_record(f"REFINE-{target}", record)

            if target == "decode":
                if record.decode_ok:
                    low = record
                else:
                    high = record
                    best_failure = record
                    # 解码失败样本最好仍然能检测到一些包，这样能区分“有检测但整体解码失败”。
                    if record.detection_ok:
                        best_detection_ok_failure = record
            else:
                if record.detection_ok:
                    low = record
                else:
                    high = record
                    best_failure = record

        if target == "decode" and best_detection_ok_failure is not None:
            return best_detection_ok_failure
        return best_failure

    def _evaluate_noise_point(
        self,
        *,
        noise_db: float,
        capture: IqCapture,
        measurement_service: GrloraMeasurementService,
        tmp_path: Path,
        reference_power: float,
        expected_packets: int,
    ) -> SearchRecord:
        added_noise_power = reference_power * (10.0 ** (float(noise_db) / 10.0))
        capture.write_noisy(
            tmp_path,
            added_noise_power,
            int(self.args.seed),
            int(self.args.chunk_samples),
            self.args.sample_limit,
            overwrite=True,
        )
        measurement = measurement_service.measure_snr(tmp_path)
        payload_check = measurement.get("payload_check", {}) or {}
        detected_packets = int(measurement.get("detected_packets", 0))
        decoded_payload_packets = int(measurement.get("decoded_payload_packets", 0))
        expected = int(payload_check.get("expected_packet_count", expected_packets))
        correct_packets = int(payload_check.get("correct_payload_packets", 0))

        if expected > 0:
            full_detection_ok = detected_packets >= expected
            # 多包 capture 中，少检测几个包和完全检测不到是两种不同现象；
            # 这里把“检测失败极限”留给 detected_packets == 0。
            detection_ok = detected_packets > 0
            decode_ok = bool(payload_check.get("all_expected_payloads_correct", False))
        else:
            full_detection_ok = detected_packets > 0
            detection_ok = detected_packets > 0
            decode_ok = detection_ok and decoded_payload_packets >= detected_packets

        return SearchRecord(
            noise_power_db_relative=float(noise_db),
            added_noise_power=float(added_noise_power),
            seed=int(self.args.seed),
            measurement=measurement,
            detected_packets=detected_packets,
            expected_packets=expected,
            decoded_payload_packets=decoded_payload_packets,
            correct_payload_packets=correct_packets,
            full_detection_ok=bool(full_detection_ok),
            detection_ok=bool(detection_ok),
            decode_ok=bool(decode_ok),
        )

    def _save_limit_file(
        self,
        *,
        kind: str,
        record: SearchRecord,
        capture: IqCapture,
        input_path: Path,
        output_dir: Path,
        reference_power: float,
        power_info: dict[str, Any],
        capture_metadata: dict[str, Any],
        history: list[SearchRecord],
    ) -> None:
        label = db_to_label(record.noise_power_db_relative)
        bin_path = output_dir / f"{input_path.stem}_{kind}_limit.bin"
        json_path = output_dir / f"{input_path.stem}_{kind}_limit.json"
        if not self.args.overwrite and (bin_path.exists() or json_path.exists()):
            raise FileExistsError(f"{bin_path} or {json_path} already exists; pass --overwrite.")

        capture.write_noisy(
            bin_path,
            record.added_noise_power,
            record.seed,
            int(self.args.chunk_samples),
            self.args.sample_limit,
            overwrite=True,
        )
        metadata = {
            "kind": kind,
            "input_file": str(input_path),
            "output_file": str(bin_path),
            "format": "raw numpy.complex64 / GNU Radio gr_complex",
            "noise_power_label": f"{label}dB",
            "noise_reference_power": float(reference_power),
            "noise_reference_power_db": db10(float(reference_power)),
            "noise_reference_mode": self.args.power_mode,
            "selected_record": record.to_metadata(),
            "capture_metadata": capture_metadata,
            "power_estimate": power_info,
            "search": {
                "coarse_start_db": float(self.args.coarse_start_db),
                "coarse_stop_db": float(self.args.coarse_stop_db),
                "coarse_step_db": float(self.args.coarse_step_db),
                "refine_iterations": int(self.args.refine_iterations),
                "history": [item.to_metadata() for item in history],
            },
            "args": vars(self.args),
        }
        write_metadata(json_path, metadata)
        print(
            f"[SAVE] {kind}: {bin_path} "
            f"(noise_rel={format_db_value(record.noise_power_db_relative)} dB, "
            f"median_snr={record.grlora_snr_median:.2f} dB)"
        )

    def _coarse_noise_values(self) -> list[float]:
        start = float(self.args.coarse_start_db)
        stop = float(self.args.coarse_stop_db)
        step = float(self.args.coarse_step_db)
        if step == 0.0:
            raise ValueError("--coarse-step-db must not be 0.")
        if (stop - start) * step < 0.0:
            raise ValueError("--coarse-step-db sign must move from start toward stop.")

        values = []
        current = start
        epsilon = abs(step) * 1e-9
        if step > 0.0:
            while current <= stop + epsilon:
                values.append(round(current, 10))
                current += step
        else:
            while current >= stop - epsilon:
                values.append(round(current, 10))
                current += step
        if not values:
            raise ValueError("No coarse search points were generated.")
        return values

    @staticmethod
    def _print_record(stage: str, record: SearchRecord) -> None:
        print(
            f"[{stage}] noise_rel={format_db_value(record.noise_power_db_relative):>8} dB, "
            f"det={record.detected_packets}/{record.expected_packets}, "
            f"decoded={record.decoded_payload_packets}, "
            f"correct={record.correct_payload_packets}/{record.expected_packets}, "
            f"decode_ok={record.decode_ok}, full_detect={record.full_detection_ok}, "
            f"any_detect={record.detection_ok}, "
            f"median_snr={record.grlora_snr_median:.2f} dB"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search for near-threshold noisy IQ files where gr-lora_sdr first fails "
            "payload decoding and packet detection."
        )
    )
    parser.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT, help=f"Input IQ file. Default: {DEFAULT_INPUT}")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: weakPacket_decoding/data/noisy_iq/<input-stem>/failure_limits",
    )
    parser.add_argument("--coarse-start-db", type=float, default=-30.0, help="Coarse search start noise dB.")
    parser.add_argument("--coarse-stop-db", type=float, default=20.0, help="Coarse search stop noise dB.")
    parser.add_argument("--coarse-step-db", type=float, default=2.0, help="Coarse search step in dB.")
    parser.add_argument("--refine-iterations", type=int, default=6, help="Binary-refinement iterations per boundary.")

    parser.add_argument("--power-mode", choices=("packet", "active", "total", "window"), default="total")
    parser.add_argument("--sf", type=int, default=None, help="LoRa spreading factor. Default: infer from filename.")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa bandwidth in Hz.")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ sample rate in Hz.")
    parser.add_argument("--cr", type=int, default=1, help="LoRa coding-rate index.")
    parser.add_argument("--pay-len", type=int, default=255, help="Fallback payload length.")
    parser.add_argument("--has-crc", action="store_true", default=True, help="Packet has PHY CRC. Default: enabled.")
    parser.add_argument("--no-crc", action="store_false", dest="has_crc", help="Packet has no PHY CRC.")
    parser.add_argument("--impl-head", action="store_true", default=False, help="Use implicit header mode.")
    parser.add_argument("--soft-decoding", action="store_true", default=False, help="Enable gr-lora_sdr soft decoding.")
    parser.add_argument("--center-freq", type=float, default=487.7e6, help="RF center frequency.")
    parser.add_argument("--sync-word", type=parse_int_auto, default=0x34, help="LoRa sync word, decimal or hex.")
    parser.add_argument("--preamble-len", type=int, default=None, help="Expected preamble length. Default: infer.")
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO mode: 0 off, 1 on, 2 auto.")
    parser.add_argument("--crc-mode", type=int, choices=[0, 1], default=0, help="0=GRLORA, 1=SX1276.")
    parser.add_argument("--expected-payload-hex", type=str, nargs="+", default=None)
    parser.add_argument("--no-expected-payload-check", action="store_true")
    parser.add_argument("--print-header", action="store_true", default=False)
    parser.add_argument("--print-grlora", action="store_true", default=False)

    parser.add_argument("--noise-percentile", type=float, default=10.0)
    parser.add_argument("--active-threshold-db", type=float, default=6.0)
    parser.add_argument("--block-samples", type=int, default=32768)
    parser.add_argument("--chunk-samples", type=int, default=1_000_000)
    parser.add_argument("--signal-start", type=int, default=None)
    parser.add_argument("--signal-samples", type=int, default=None)
    parser.add_argument("--ignore-existing-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    return FailureLimitSearcher(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
