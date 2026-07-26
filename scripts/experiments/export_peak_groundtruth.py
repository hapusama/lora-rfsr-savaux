#!/usr/bin/env python3
"""导出 gr-lora_sdr fft_demod 内部的 peak 级 groundtruth。"""
# D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\export_peak_groundtruth.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin -o gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_8_peak_gt.csv --sf 10 --bw 125000 --samp-rate 500000 --cr 1 --center-freq 487.7e6 --sync-word 0x34 --preamble-len 8 --ldro-mode 2 --crc-mode 0
from __future__ import annotations

import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import sys
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from noisy_iq.detector import cleanup_file_source_path, prepare_file_source_path


def parse_int_auto(value: str) -> int:
    return int(value, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "运行 gr-lora_sdr 接收链，并监听 fft_demod 的 peak_candidates "
            "消息端口，生成符号/peak 级 groundtruth CSV。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="raw complex64 IQ 文件。")
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出 peak groundtruth CSV。")
    parser.add_argument("--sf", "--spreading-factor", type=int, default=7, help="LoRa SF，默认 7。")
    parser.add_argument("--bw", "--bandwidth", type=float, default=125000.0, help="LoRa 带宽 Hz，默认 125000。")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ 采样率 Hz，默认 500000。")
    parser.add_argument("--cr", "--coding-rate", type=int, default=1, help="编码率索引，默认 1。")
    parser.add_argument("--pay-len", type=int, default=255, help="隐式头 fallback payload 长度。")
    parser.add_argument("--has-crc", action="store_true", default=True, help="发送端带 PHY CRC，默认开启。")
    parser.add_argument("--no-crc", action="store_false", dest="has_crc", help="发送端不带 PHY CRC。")
    parser.add_argument("--impl-head", action="store_true", default=False, help="隐式头模式。")
    parser.add_argument("--soft-decoding", action="store_true", default=True, help="用 soft decoding 链路，默认开启。")
    parser.add_argument("--hard-decoding", action="store_false", dest="soft_decoding", help="使用 hard decoding 链路。")
    parser.add_argument("--center-freq", type=float, default=868.1e6, help="RF 中心频率，用于 SFO 估计，默认 868.1e6。")
    parser.add_argument("--sync-word", type=parse_int_auto, default=0x34, help="同步字，默认 0x34。")
    parser.add_argument("--preamble-len", type=int, default=16, help="前导码长度/同步触发参数，默认 16。")
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO：0 关，1 开，2 自动。")
    parser.add_argument("--crc-mode", type=int, choices=[0, 1], default=0, help="0=GRLORA，1=SX1276。")
    parser.add_argument("--print-header", action="store_true", default=False, help="打印 header_decoder 信息。")
    parser.add_argument("--max-log-approx", action="store_true", default=True, help="soft LLR 使用 max-log 近似，默认开启。")
    parser.add_argument("--no-max-log-approx", action="store_false", dest="max_log_approx", help="soft LLR 不使用 max-log 近似。")
    parser.add_argument(
        "--include-header",
        action="store_true",
        default=True,
        help="写出 PHY header 符号。当前默认开启，保留该参数是为了兼容旧命令。",
    )
    parser.add_argument(
        "--payload-only",
        action="store_false",
        dest="include_header",
        help="只写 payload chirp label，不写 PHY header 符号。",
    )
    parser.add_argument(
        "--min-label-confidence-db",
        type=float,
        default=6.0,
        help="label_reliable 判定阈值，只打标不丢行，默认 6 dB。",
    )
    parser.add_argument("--summary-output", type=Path, default=None, help="可选：另存每个包的 label 摘要 CSV。")
    parser.add_argument(
        "--consensus-output",
        type=Path,
        default=None,
        help="可选：按 symbol 位置汇总重复帧，输出可直接用于 SER 的 FFT-bin ground truth。",
    )
    return parser.parse_args()


def _pmt_value(pmt, msg, key: str, default: Any = None) -> Any:
    value = pmt.dict_ref(msg, pmt.intern(key), pmt.PMT_NIL)
    if pmt.is_null(value):
        return default
    try:
        return pmt.to_python(value)
    except Exception:
        return default


def _u16vector(pmt, value) -> list[int]:
    if hasattr(pmt, "is_u16vector") and pmt.is_u16vector(value):
        return [int(item) for item in pmt.u16vector_elements(value)]
    return []


def _f32vector(pmt, value) -> list[float]:
    if hasattr(pmt, "is_f32vector") and pmt.is_f32vector(value):
        return [float(item) for item in pmt.f32vector_elements(value)]
    return []


def _c32vector(pmt, value) -> dict[str, list[float]]:
    if hasattr(pmt, "is_c32vector") and pmt.is_c32vector(value):
        values = list(pmt.c32vector_elements(value))
        return {
            "real": [float(item.real) for item in values],
            "imag": [float(item.imag) for item in values],
        }
    return {"real": [], "imag": []}


def export_peak_groundtruth(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from gnuradio import blocks, gr
        import gnuradio.lora_sdr as lora_sdr
        import pmt
    except ImportError as exc:
        raise RuntimeError(
            "需要在包含 GNU Radio 和 gr-lora_sdr 的 conda 环境中运行，"
            "例如 conda activate gr-lora。"
        ) from exc

    class PeakCandidateSink(gr.basic_block):
        """收集 fft_demod 发布的逐符号 Top-K peak 候选。"""

        def __init__(self):
            gr.basic_block.__init__(self, name="peak_candidate_sink", in_sig=None, out_sig=None)
            self.records: list[dict[str, Any]] = []
            self._lock = threading.Lock()
            self.message_port_register_in(pmt.intern("peak_candidates"))
            self.set_msg_handler(pmt.intern("peak_candidates"), self.handle_peak_candidates)

        def handle_peak_candidates(self, msg):
            if not pmt.is_dict(msg):
                return
            bins_pmt = pmt.dict_ref(msg, pmt.intern("candidate_bins"), pmt.PMT_NIL)
            symbols_pmt = pmt.dict_ref(msg, pmt.intern("candidate_symbols"), pmt.PMT_NIL)
            values_pmt = pmt.dict_ref(msg, pmt.intern("candidate_values"), pmt.PMT_NIL)
            powers_pmt = pmt.dict_ref(msg, pmt.intern("candidate_powers"), pmt.PMT_NIL)
            phases_pmt = pmt.dict_ref(msg, pmt.intern("candidate_phases"), pmt.PMT_NIL)
            record = {
                "frame_count": int(_pmt_value(pmt, msg, "frame_count", -1)),
                "symbol_index": int(_pmt_value(pmt, msg, "symbol_index", -1)),
                "is_header": bool(_pmt_value(pmt, msg, "is_header", False)),
                "sf": int(_pmt_value(pmt, msg, "sf", args.sf)),
                "cr": int(_pmt_value(pmt, msg, "cr", args.cr)),
                "ldro": bool(_pmt_value(pmt, msg, "ldro", False)),
                "samples_per_symbol": int(_pmt_value(pmt, msg, "samples_per_symbol", 1 << args.sf)),
                "top_k": int(_pmt_value(pmt, msg, "top_k", 0)),
                "hard_bin": int(_pmt_value(pmt, msg, "hard_bin", -1)),
                "hard_symbol": int(_pmt_value(pmt, msg, "hard_symbol", -1)),
                "confidence_db": float(_pmt_value(pmt, msg, "confidence_db", float("nan"))),
                "total_power": float(_pmt_value(pmt, msg, "total_power", float("nan"))),
                "noise_power_est": float(_pmt_value(pmt, msg, "noise_power_est", float("nan"))),
                "cfo_int": int(_pmt_value(pmt, msg, "cfo_int", 0)),
                "cfo_frac": float(_pmt_value(pmt, msg, "cfo_frac", 0.0)),
                "candidate_bins": _u16vector(pmt, bins_pmt),
                "candidate_symbols": _u16vector(pmt, symbols_pmt),
                "candidate_values": _c32vector(pmt, values_pmt),
                "candidate_powers": _f32vector(pmt, powers_pmt),
                "candidate_phases": _f32vector(pmt, phases_pmt),
            }
            with self._lock:
                self.records.append(record)

    class PeakGroundtruthTopBlock(gr.top_block):
        """完整接收链，只额外监听 fft_demod 的 peak 候选消息。"""

        def __init__(self, file_source_path: Path):
            gr.top_block.__init__(self, "LoRa Peak Groundtruth Exporter", catch_exceptions=True)
            os_factor = int(round(float(args.samp_rate) / float(args.bw)))
            min_buf = int(os_factor * ((1 << int(args.sf)) + 2))

            self.file_source = blocks.file_source(gr.sizeof_gr_complex, str(file_source_path), False, 0, 0)
            self.file_source.set_min_output_buffer(min_buf)
            self.frame_sync = lora_sdr.frame_sync(
                int(args.center_freq),
                int(args.bw),
                int(args.sf),
                bool(args.impl_head),
                [int(args.sync_word)],
                os_factor,
                int(args.preamble_len),
            )
            self.fft_demod = lora_sdr.fft_demod(bool(args.soft_decoding), bool(args.max_log_approx))
            self.gray_mapping = lora_sdr.gray_mapping(bool(args.soft_decoding))
            self.deinterleaver = lora_sdr.deinterleaver(bool(args.soft_decoding))
            self.hamming_dec = lora_sdr.hamming_dec(bool(args.soft_decoding))
            self.header_decoder = lora_sdr.header_decoder(
                bool(args.impl_head),
                int(args.cr),
                int(args.pay_len),
                bool(args.has_crc),
                int(args.ldro_mode),
                bool(args.print_header),
            )
            self.dewhitening = lora_sdr.dewhitening()
            crc_mode = lora_sdr.Crc_mode.SX1276 if int(args.crc_mode) == 1 else lora_sdr.Crc_mode.GRLORA
            self.crc_verif = lora_sdr.crc_verif(0, False, crc_mode)
            self.peak_sink = PeakCandidateSink()

            self.connect((self.file_source, 0), (self.frame_sync, 0))
            self.connect((self.frame_sync, 0), (self.fft_demod, 0))
            self.connect((self.fft_demod, 0), (self.gray_mapping, 0))
            self.connect((self.gray_mapping, 0), (self.deinterleaver, 0))
            self.connect((self.deinterleaver, 0), (self.hamming_dec, 0))
            self.connect((self.hamming_dec, 0), (self.header_decoder, 0))
            self.connect((self.header_decoder, 0), (self.dewhitening, 0))
            self.connect((self.dewhitening, 0), (self.crc_verif, 0))
            self.msg_connect((self.header_decoder, "frame_info"), (self.frame_sync, "frame_info"))
            self.msg_connect((self.fft_demod, "peak_candidates"), (self.peak_sink, "peak_candidates"))

    staged_path = prepare_file_source_path(args.input)
    try:
        tb = PeakGroundtruthTopBlock(staged_path)
        tb.start()
        tb.wait()
        records = list(tb.peak_sink.records)
    finally:
        cleanup_file_source_path(staged_path)

    records.sort(key=lambda item: (item["frame_count"], item["symbol_index"]))
    annotate_records(records, min_label_confidence_db=float(args.min_label_confidence_db))
    return {
        "input_file": str(Path(args.input).resolve()),
        "format": "gr-lora_sdr fft_demod peak_candidates",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output"
        },
        "record_count": len(records),
        "payload_record_count": sum(1 for record in records if not bool(record["is_header"])),
        "records": records,
    }


def annotate_records(records: list[dict[str, Any]], min_label_confidence_db: float = 6.0) -> None:
    """给 gr-lora_sdr 原始 peak 消息补上 packet/payload 级 label 索引。"""

    records.sort(key=lambda item: (item["frame_count"], item["symbol_index"]))
    frame_counts = sorted({int(record["frame_count"]) for record in records})
    packet_index_by_frame = {frame_count: index for index, frame_count in enumerate(frame_counts)}
    total_by_frame = {frame_count: 0 for frame_count in frame_counts}
    header_by_frame = {frame_count: 0 for frame_count in frame_counts}
    payload_by_frame = {frame_count: 0 for frame_count in frame_counts}
    for record in records:
        frame_count = int(record["frame_count"])
        total_by_frame[frame_count] += 1
        if bool(record["is_header"]):
            header_by_frame[frame_count] += 1
        else:
            payload_by_frame[frame_count] += 1

    payload_seen = {frame_count: 0 for frame_count in frame_counts}
    for record in records:
        frame_count = int(record["frame_count"])
        is_header = bool(record["is_header"])
        payload_chirp_index = ""
        if not is_header:
            payload_chirp_index = payload_seen[frame_count]
            payload_seen[frame_count] += 1

        values = record.get("candidate_values", {})
        real_values = values.get("real", [])
        imag_values = values.get("imag", [])
        powers = record.get("candidate_powers", [])
        phases = record.get("candidate_phases", [])
        confidence_db = float(record.get("confidence_db", float("nan")))
        record["packet_index"] = int(packet_index_by_frame[frame_count])
        record["frame_symbol_index"] = int(record["symbol_index"])
        record["payload_chirp_index"] = payload_chirp_index
        record["packet_total_symbols"] = int(total_by_frame[frame_count])
        record["packet_header_symbols"] = int(header_by_frame[frame_count])
        record["packet_payload_chirps"] = int(payload_by_frame[frame_count])
        record["label_fft_bin"] = int(record["hard_bin"])
        record["label_symbol"] = int(record["hard_symbol"])
        record["label_real"] = real_values[0] if real_values else ""
        record["label_imag"] = imag_values[0] if imag_values else ""
        record["label_power"] = powers[0] if powers else ""
        record["label_phase"] = phases[0] if phases else ""
        record["label_confidence_db"] = confidence_db
        record["label_reliable"] = int(confidence_db >= float(min_label_confidence_db))


def _records_for_output(payload: dict[str, Any], include_header: bool) -> list[dict[str, Any]]:
    records = list(payload["records"])
    if include_header:
        return records
    return [record for record in records if not bool(record["is_header"])]


def write_peak_csv(path: Path, payload: dict[str, Any], include_header: bool = False) -> dict[str, Any]:
    """把嵌套的 Top-K peak 记录展开成便于训练/评估使用的 CSV。"""

    records = _records_for_output(payload, include_header=include_header)
    max_top_k = max((int(record.get("top_k", 0)) for record in records), default=0)
    fixed_fields = [
        "input_file",
        "packet_index",
        "frame_count",
        "frame_symbol_index",
        "payload_chirp_index",
        "packet_total_symbols",
        "packet_header_symbols",
        "packet_payload_chirps",
        "is_header",
        "sf",
        "cr",
        "ldro",
        "samples_per_symbol",
        "label_fft_bin",
        "label_symbol",
        "label_real",
        "label_imag",
        "label_power",
        "label_phase",
        "label_confidence_db",
        "label_reliable",
        "top_k",
        "hard_bin",
        "hard_symbol",
        "confidence_db",
        "total_power",
        "noise_power_est",
        "cfo_int",
        "cfo_frac",
    ]
    candidate_fields = []
    for rank in range(1, max_top_k + 1):
        candidate_fields.extend(
            [
                f"top{rank}_bin",
                f"top{rank}_symbol",
                f"top{rank}_real",
                f"top{rank}_imag",
                f"top{rank}_power",
                f"top{rank}_phase",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fixed_fields + candidate_fields)
        writer.writeheader()
        for record in records:
            row = {
                "input_file": payload["input_file"],
                "packet_index": record["packet_index"],
                "frame_count": record["frame_count"],
                "frame_symbol_index": record["frame_symbol_index"],
                "payload_chirp_index": record["payload_chirp_index"],
                "packet_total_symbols": record["packet_total_symbols"],
                "packet_header_symbols": record["packet_header_symbols"],
                "packet_payload_chirps": record["packet_payload_chirps"],
                "is_header": int(bool(record["is_header"])),
                "sf": record["sf"],
                "cr": record["cr"],
                "ldro": int(bool(record["ldro"])),
                "samples_per_symbol": record["samples_per_symbol"],
                "label_fft_bin": record["label_fft_bin"],
                "label_symbol": record["label_symbol"],
                "label_real": record["label_real"],
                "label_imag": record["label_imag"],
                "label_power": record["label_power"],
                "label_phase": record["label_phase"],
                "label_confidence_db": record["label_confidence_db"],
                "label_reliable": record["label_reliable"],
                "top_k": record["top_k"],
                "hard_bin": record["hard_bin"],
                "hard_symbol": record["hard_symbol"],
                "confidence_db": record["confidence_db"],
                "total_power": record["total_power"],
                "noise_power_est": record["noise_power_est"],
                "cfo_int": record["cfo_int"],
                "cfo_frac": record["cfo_frac"],
            }
            values = record.get("candidate_values", {})
            real_values = values.get("real", [])
            imag_values = values.get("imag", [])
            bins = record.get("candidate_bins", [])
            symbols = record.get("candidate_symbols", [])
            powers = record.get("candidate_powers", [])
            phases = record.get("candidate_phases", [])
            for index in range(max_top_k):
                rank = index + 1
                row[f"top{rank}_bin"] = bins[index] if index < len(bins) else ""
                row[f"top{rank}_symbol"] = symbols[index] if index < len(symbols) else ""
                row[f"top{rank}_real"] = real_values[index] if index < len(real_values) else ""
                row[f"top{rank}_imag"] = imag_values[index] if index < len(imag_values) else ""
                row[f"top{rank}_power"] = powers[index] if index < len(powers) else ""
                row[f"top{rank}_phase"] = phases[index] if index < len(phases) else ""
            writer.writerow(row)
    os.replace(tmp_path, path)
    return {
        "wrote_rows": len(records),
        "include_header": bool(include_header),
        "max_top_k": int(max_top_k),
    }


def write_summary_csv(path: Path, payload: dict[str, Any]) -> None:
    """写出每个 gr-lora_sdr 解码帧对应的 payload label 摘要。"""

    records = list(payload["records"])
    frame_counts = sorted({int(record["frame_count"]) for record in records})
    fields = [
        "input_file",
        "packet_index",
        "frame_count",
        "total_symbols",
        "header_symbols",
        "payload_chirps",
        "first_payload_frame_symbol_index",
        "last_payload_frame_symbol_index",
        "min_payload_confidence_db",
        "mean_payload_confidence_db",
        "all_payload_labels_reliable",
        "cfo_int",
        "cfo_frac",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for packet_index, frame_count in enumerate(frame_counts):
            packet_records = [record for record in records if int(record["frame_count"]) == frame_count]
            payload_records = [record for record in packet_records if not bool(record["is_header"])]
            confidences = [float(record["label_confidence_db"]) for record in payload_records]
            first_payload = payload_records[0]["frame_symbol_index"] if payload_records else ""
            last_payload = payload_records[-1]["frame_symbol_index"] if payload_records else ""
            writer.writerow(
                {
                    "input_file": payload["input_file"],
                    "packet_index": packet_index,
                    "frame_count": frame_count,
                    "total_symbols": len(packet_records),
                    "header_symbols": sum(1 for record in packet_records if bool(record["is_header"])),
                    "payload_chirps": len(payload_records),
                    "first_payload_frame_symbol_index": first_payload,
                    "last_payload_frame_symbol_index": last_payload,
                    "min_payload_confidence_db": min(confidences) if confidences else "",
                    "mean_payload_confidence_db": (sum(confidences) / len(confidences)) if confidences else "",
                    "all_payload_labels_reliable": int(all(int(record["label_reliable"]) for record in payload_records)) if payload_records else 0,
                    "cfo_int": packet_records[0]["cfo_int"] if packet_records else "",
                    "cfo_frac": packet_records[0]["cfo_frac"] if packet_records else "",
                }
            )
    os.replace(tmp_path, path)


def write_consensus_csv(path: Path, payload: dict[str, Any]) -> None:
    """把固定帧的重复观测汇总成一行一个 symbol 的 FFT-bin 真值。"""

    records = list(payload["records"])
    symbol_indices = sorted({int(record["frame_symbol_index"]) for record in records})
    fields = [
        "input_file",
        "frame_symbol_index",
        "stage",
        "stage_symbol_index",
        "support_frames",
        "groundtruth_fft_bin",
        "groundtruth_symbol",
        "agreement_count",
        "agreement_ratio",
        "unique_fft_bin_count",
        "unique_fft_bins",
        "min_confidence_db",
        "mean_confidence_db",
        "all_source_labels_reliable",
        "consensus_reliable",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_symbol_index in symbol_indices:
            items = [
                record
                for record in records
                if int(record["frame_symbol_index"]) == frame_symbol_index
            ]
            bin_counts = Counter(int(record["label_fft_bin"]) for record in items)
            groundtruth_bin, agreement_count = max(
                bin_counts.items(), key=lambda item: (item[1], -item[0])
            )
            matching_symbols = [
                int(record["label_symbol"])
                for record in items
                if int(record["label_fft_bin"]) == groundtruth_bin
            ]
            symbol_counts = Counter(matching_symbols)
            groundtruth_symbol = max(
                symbol_counts.items(), key=lambda item: (item[1], -item[0])
            )[0]
            confidences = [float(record["label_confidence_db"]) for record in items]
            all_reliable = all(int(record["label_reliable"]) for record in items)
            agreement_ratio = float(agreement_count) / float(len(items))
            is_header = bool(items[0]["is_header"])
            stage_symbol_index = (
                frame_symbol_index
                if is_header
                else int(items[0]["payload_chirp_index"])
            )
            writer.writerow(
                {
                    "input_file": payload["input_file"],
                    "frame_symbol_index": frame_symbol_index,
                    "stage": "header" if is_header else "payload",
                    "stage_symbol_index": stage_symbol_index,
                    "support_frames": len(items),
                    "groundtruth_fft_bin": groundtruth_bin,
                    "groundtruth_symbol": groundtruth_symbol,
                    "agreement_count": agreement_count,
                    "agreement_ratio": agreement_ratio,
                    "unique_fft_bin_count": len(bin_counts),
                    "unique_fft_bins": " ".join(str(value) for value in sorted(bin_counts)),
                    "min_confidence_db": min(confidences),
                    "mean_confidence_db": sum(confidences) / len(confidences),
                    "all_source_labels_reliable": int(all_reliable),
                    "consensus_reliable": int(all_reliable and agreement_count == len(items)),
                }
            )
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    payload = export_peak_groundtruth(args)
    stats = write_peak_csv(args.output, payload, include_header=bool(args.include_header))
    if args.summary_output is not None:
        write_summary_csv(args.summary_output, payload)
    if args.consensus_output is not None:
        write_consensus_csv(args.consensus_output, payload)
    print(f"raw_records={payload['record_count']}")
    print(f"payload_records={payload['payload_record_count']}")
    print(f"wrote_rows={stats['wrote_rows']}")
    print(f"include_header={int(stats['include_header'])}")
    print(f"wrote={args.output}")
    if args.summary_output is not None:
        print(f"wrote_summary={args.summary_output}")
    if args.consensus_output is not None:
        print(f"wrote_consensus={args.consensus_output}")


if __name__ == "__main__":
    main()
