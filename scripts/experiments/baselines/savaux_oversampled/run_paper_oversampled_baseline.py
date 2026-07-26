#!/usr/bin/env python3
"""评估 Savaux oversampled LoRa 解调 baseline。

这个 runner 的职责刻意很窄：读取本项目 header-first CSV 中已经同步且
header-valid 的 symbol，然后比较：

* center/osr1 argmax：现有单中心采样 branch 的 chip-rate dechirp+FFT；
* paper_osr_argmax：Savaux 论文里的 branch DFT + 相位合并规则。

它不使用 offset coherence、packet-line phase、Top-L lock、payload template、
cross-packet prior 或 CRC-guided bin selection。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[4]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, bin_to_grlora_symbol, signed_fft_bin  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    demod_paper_oversampled_symbol,
)
from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paper-only oversampled LoRa demodulation baseline on header-first symbols."
    )
    parser.add_argument("-i", "--input-iq", type=Path, required=True, help="complex64 IQ file.")
    parser.add_argument(
        "-g",
        "--gt-symbol-csv",
        type=Path,
        required=True,
        help="clean header-first symbol CSV used for timing/header and GT raw_fft_bin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="per-packet output CSV. Default: data/paper_oversampled_baseline/<iq-stem>_paper_osr_packets.csv",
    )
    parser.add_argument(
        "--symbols-output",
        type=Path,
        default=None,
        help="per-symbol output CSV. Default: same stem as --output with _symbols suffix.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="optional aggregate summary JSON.",
    )
    parser.add_argument("--packet", type=int, default=None, help="optional packet_index filter.")
    parser.add_argument("--max-packets", type=int, default=None, help="optional packet limit after sorting.")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument(
        "--cfo-correction-mode",
        choices=("none", "symbol", "continuous"),
        default="continuous",
        help=(
            "none is the pure paper assumption; symbol/continuous only apply the "
            "receiver-estimated CFO before the same paper metric."
        ),
    )
    parser.add_argument(
        "--paper-origin-shift",
        type=int,
        default=None,
        help=(
            "sample shift applied before the paper OSR branch split. Default is "
            "os_factor//2 to match header-first chip-center sampling. Use 0 when "
            "start_sample is already the desired oversampled branch origin."
        ),
    )
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    if value == "":
        return int(default)
    return int(float(value))


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return float(default)
    return float(value)


def _first_int(row: dict[str, str], keys: tuple[str, ...], default: int = 0) -> int:
    for key in keys:
        if str(row.get(key, "")).strip() != "":
            return _int(row, key, default)
    return int(default)


def load_packets(symbol_csv: Path, packet_filter: int | None, max_packets: int | None) -> list[dict[str, Any]]:
    """从 header-first symbol CSV 读取可评估的包。

    这里使用 clean header-first CSV 提供三类信息：
    1. 每个 payload symbol 的 start_sample；
    2. header 解码参数和 payload codec 需要的 header symbols；
    3. clean 条件下的 raw_fft_bin，作为 SER 评估 GT。
    """

    with symbol_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    packets: dict[int, dict[str, Any]] = {}
    for row in rows:
        packet_index = _int(row, "packet_index", -1)
        if packet_index < 0:
            continue
        if packet_filter is not None and packet_index != int(packet_filter):
            continue
        packet = packets.setdefault(
            packet_index,
            {
                "packet_index": packet_index,
                "frame_index": _int(row, "frame_index", packet_index),
                "event_index": _int(row, "event_index", packet_index),
                "header_symbols": [],
                "payload_symbols": [],
                "header_start_sample": None,
                "sf": _int(row, "sf", 10),
                "bw": _float(row, "bw", 125000.0),
                "os_factor": _int(row, "os_factor", 4),
                "cfo_int": _int(row, "cfo_int", 0),
                "cfo_frac": _float(row, "cfo_frac", 0.0),
                "payload_len": _int(row, "payload_len", 0),
                "cr": _first_int(row, ("payload_cr", "cr"), 1),
                "has_crc": bool(_first_int(row, ("payload_has_crc", "has_crc"), 1)),
                "ldro": bool(_first_int(row, ("payload_ldro", "ldro"), 0)),
                "header_valid": bool(_int(row, "header_valid", 0)),
            },
        )

        stage = str(row.get("stage", "")).strip().lower()
        stage_symbol_index = _int(row, "stage_symbol_index", -1)
        if stage == "header":
            if stage_symbol_index == 0:
                packet["header_start_sample"] = _int(row, "start_sample", 0)
            if 0 <= stage_symbol_index < 8:
                packet["header_symbols"].append((stage_symbol_index, _int(row, "symbol_value", 0)))
        elif stage == "payload":
            packet["payload_symbols"].append(
                {
                    "payload_symbol_index": stage_symbol_index,
                    "start_sample": _int(row, "start_sample", 0),
                    "gt_raw_fft_bin": _int(row, "raw_fft_bin", -1),
                }
            )

    valid_packets: list[dict[str, Any]] = []
    for packet in packets.values():
        if not packet["header_valid"] or packet["header_start_sample"] is None:
            continue
        packet["header_symbols"] = [
            value for _, value in sorted(packet["header_symbols"], key=lambda item: item[0])
        ]
        packet["payload_symbols"].sort(key=lambda item: int(item["payload_symbol_index"]))
        if len(packet["header_symbols"]) == 8 and packet["payload_symbols"]:
            valid_packets.append(packet)
    valid_packets.sort(key=lambda item: int(item["packet_index"]))
    if max_packets is not None:
        valid_packets = valid_packets[: int(max_packets)]
    return valid_packets


def _center_symbol_spectrum(
    samples: np.ndarray,
    packet: dict[str, Any],
    start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    """中心采样 baseline：只取每个 chip 的中心样点并做普通 chip-rate FFT。"""

    sf = int(packet["sf"])
    n_bins = 1 << sf
    os_factor = int(packet["os_factor"])
    center_offset = os_factor // 2
    indexes = int(start_sample) + center_offset + os_factor * np.arange(n_bins, dtype=np.int64)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")

    mode = str(cfo_correction_mode)
    use_cfo = mode in {"symbol", "continuous"}
    cfo_int = int(packet["cfo_int"]) if use_cfo else 0
    cfo_frac = float(packet["cfo_frac"]) if use_cfo else 0.0
    symbol = np.asarray(samples[indexes], dtype=np.complex64)
    if mode == "continuous":
        cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
        relative_chip_start = float(int(start_sample) - int(packet["header_start_sample"])) / float(os_factor)
        cfo_phase = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        symbol = (symbol * np.exp(-1j * cfo_phase)).astype(np.complex64)
    downchirp = build_downchirp(sf, cfo_int=cfo_int, cfo_frac=cfo_frac)
    return np.fft.fft((symbol * downchirp).astype(np.complex64)).astype(np.complex64)


def _rank_of(power: np.ndarray, raw_bin: int) -> int:
    gt = int(raw_bin)
    if gt < 0 or gt >= power.size:
        return -1
    order = np.argsort(np.asarray(power, dtype=np.float64))[::-1]
    matches = np.where(order == gt)[0]
    return int(matches[0] + 1) if matches.size else -1


def _ser(raw_bins: Sequence[int], gt_bins: Sequence[int], sf: int, ldro: bool) -> tuple[float, float, int]:
    raw_errors = 0
    symbol_errors = 0
    compared = 0
    for pred, gt in zip(raw_bins, gt_bins):
        if int(gt) < 0:
            continue
        raw_errors += int(int(pred) != int(gt))
        pred_symbol = bin_to_grlora_symbol(int(pred), sf=sf, is_header=False, ldro=ldro)
        gt_symbol = bin_to_grlora_symbol(int(gt), sf=sf, is_header=False, ldro=ldro)
        symbol_errors += int(pred_symbol != gt_symbol)
        compared += 1
    if compared <= 0:
        return 0.0, 0.0, 0
    return float(raw_errors / compared), float(symbol_errors / compared), int(compared)


def _decode_payload(packet: dict[str, Any], raw_bins: Sequence[int], crc_mode: str, ldro_mode: int) -> dict[str, Any]:
    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    payload_symbols = tuple(
        bin_to_grlora_symbol(int(raw_bin), sf=sf, is_header=False, ldro=ldro)
        for raw_bin in raw_bins
    )
    try:
        decoded = decode_explicit_frame_symbols(
            header_symbol_values=tuple(packet["header_symbols"]),
            payload_symbol_values=payload_symbols,
            sf=sf,
            bw=float(packet["bw"]),
            ldro_mode=int(ldro_mode),
            crc_mode=str(crc_mode),
        )
        return {
            "payload_hex": decoded.payload.payload_bytes.hex(),
            "crc_valid": int(decoded.payload.crc_valid),
            "crc_computed": f"0x{decoded.payload.crc_computed:04x}",
            "crc_received": f"0x{decoded.payload.crc_received:04x}",
            "decode_error": "",
        }
    except Exception as exc:
        return {
            "payload_hex": "",
            "crc_valid": 0,
            "crc_computed": "",
            "crc_received": "",
            "decode_error": f"{type(exc).__name__}: {exc}",
        }


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else 0.0


def evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """对一个 packet 同时跑 center baseline 和 Savaux paper baseline。"""

    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    ldro = bool(packet["ldro"])
    center_bins: list[int] = []
    paper_bins: list[int] = []
    gt_bins: list[int] = []
    symbol_rows: list[dict[str, Any]] = []
    # header-first 的 start_sample 搭配 center branch 使用 start + os_factor//2。
    # 为了让 paper OSR 全分支处理与同一个采样原点对齐，默认也把 branch origin
    # 平移到 chip-center；如果外部数据已经给出 branch origin，可传 0 覆盖。
    paper_origin_shift = int(args.paper_origin_shift) if args.paper_origin_shift is not None else os_factor // 2
    paper_header_start = int(packet["header_start_sample"]) + paper_origin_shift

    for symbol in packet["payload_symbols"]:
        start_sample = int(symbol["start_sample"])
        paper_start_sample = start_sample + paper_origin_shift
        gt_bin = int(symbol["gt_raw_fft_bin"])
        center_spectrum = _center_symbol_spectrum(
            samples=samples,
            packet=packet,
            start_sample=start_sample,
            cfo_correction_mode=str(args.cfo_correction_mode),
        )
        center_power = np.abs(center_spectrum).astype(np.float64) ** 2
        center_bin = int(np.argmax(center_power))
        center_peak_power = float(center_power[center_bin])

        paper = demod_paper_oversampled_symbol(
            # 这是唯一的论文 baseline 调用点；这里没有传入 coherence/phase-line/CRC 信息。
            samples=samples,
            start_sample=paper_start_sample,
            sf=sf,
            os_factor=os_factor,
            is_header=False,
            ldro=ldro,
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=paper_header_start,
            cfo_correction_mode=str(args.cfo_correction_mode),
        )
        paper_power = np.abs(paper.combined_spectrum).astype(np.float64) ** 2

        center_bins.append(center_bin)
        paper_bins.append(int(paper.raw_fft_bin))
        gt_bins.append(gt_bin)

        center_symbol = bin_to_grlora_symbol(center_bin, sf=sf, is_header=False, ldro=ldro)
        paper_symbol = int(paper.symbol_value)
        gt_symbol = bin_to_grlora_symbol(gt_bin, sf=sf, is_header=False, ldro=ldro)
        symbol_rows.append(
            {
                "packet_index": int(packet["packet_index"]),
                "frame_index": int(packet["frame_index"]),
                "event_index": int(packet["event_index"]),
                "payload_symbol_index": int(symbol["payload_symbol_index"]),
                "start_sample": start_sample,
                "paper_start_sample": paper_start_sample,
                "paper_origin_shift": paper_origin_shift,
                "sf": sf,
                "os_factor": os_factor,
                "cfo_correction_mode": str(args.cfo_correction_mode),
                "gt_raw_fft_bin": gt_bin,
                "gt_signed_fft_bin": signed_fft_bin(gt_bin, 1 << sf),
                "gt_symbol_value": int(gt_symbol),
                "center_raw_fft_bin": center_bin,
                "center_signed_fft_bin": signed_fft_bin(center_bin, 1 << sf),
                "center_symbol_value": int(center_symbol),
                "paper_raw_fft_bin": int(paper.raw_fft_bin),
                "paper_signed_fft_bin": int(paper.signed_fft_bin),
                "paper_symbol_value": paper_symbol,
                "center_raw_hit": int(center_bin == gt_bin),
                "paper_raw_hit": int(int(paper.raw_fft_bin) == gt_bin),
                "center_symbol_hit": int(center_symbol == gt_symbol),
                "paper_symbol_hit": int(paper_symbol == gt_symbol),
                "center_gt_rank": _rank_of(center_power, gt_bin),
                "paper_gt_rank": _rank_of(paper_power, gt_bin),
                "center_peak_power": center_peak_power,
                "paper_peak_power": float(paper.peak_power),
                "paper_peak_margin_db": float(paper.peak_margin_db),
                "paper_total_power": float(paper.total_power),
            }
        )

    center_raw_ser, center_symbol_ser, compared = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    paper_raw_ser, paper_symbol_ser, _ = _ser(paper_bins, gt_bins, sf=sf, ldro=ldro)
    center_decode = _decode_payload(packet, center_bins, args.crc_mode, args.ldro_mode)
    paper_decode = _decode_payload(packet, paper_bins, args.crc_mode, args.ldro_mode)
    row = {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "sf": sf,
        "bw": float(packet["bw"]),
        "os_factor": os_factor,
        "payload_len": int(packet["payload_len"]),
        "cr": int(packet["cr"]),
        "has_crc": int(bool(packet["has_crc"])),
        "ldro": int(ldro),
        "cfo_correction_mode": str(args.cfo_correction_mode),
        "paper_origin_shift": int(paper_origin_shift),
        "symbol_count": int(len(gt_bins)),
        "gt_compared_symbols": int(compared),
        "center_raw_ser": float(center_raw_ser),
        "center_symbol_ser": float(center_symbol_ser),
        "paper_raw_ser": float(paper_raw_ser),
        "paper_symbol_ser": float(paper_symbol_ser),
        "paper_symbol_ser_gain_vs_center": float(center_symbol_ser - paper_symbol_ser),
        "center_crc_valid": int(center_decode["crc_valid"]),
        "paper_crc_valid": int(paper_decode["crc_valid"]),
        "paper_crc_gain_vs_center": int(paper_decode["crc_valid"]) - int(center_decode["crc_valid"]),
        "center_mean_gt_rank": _mean([row["center_gt_rank"] for row in symbol_rows]),
        "paper_mean_gt_rank": _mean([row["paper_gt_rank"] for row in symbol_rows]),
        "center_payload_hex": center_decode["payload_hex"],
        "paper_payload_hex": paper_decode["payload_hex"],
        "center_crc_computed": center_decode["crc_computed"],
        "center_crc_received": center_decode["crc_received"],
        "paper_crc_computed": paper_decode["crc_computed"],
        "paper_crc_received": paper_decode["crc_received"],
        "center_decode_error": center_decode["decode_error"],
        "paper_decode_error": paper_decode["decode_error"],
    }
    return row, symbol_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_summary(packet_rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_iq": str(args.input_iq),
        "gt_symbol_csv": str(args.gt_symbol_csv),
        "packet_count": int(len(packet_rows)),
        "crc_mode": str(args.crc_mode),
        "cfo_correction_mode": str(args.cfo_correction_mode),
        "paper_origin_shift": (
            "os_factor//2" if args.paper_origin_shift is None else int(args.paper_origin_shift)
        ),
        "center_symbol_ser": _mean([row["center_symbol_ser"] for row in packet_rows]),
        "paper_symbol_ser": _mean([row["paper_symbol_ser"] for row in packet_rows]),
        "paper_symbol_ser_gain_vs_center": _mean(
            [row["paper_symbol_ser_gain_vs_center"] for row in packet_rows]
        ),
        "center_crc_valid_rate": _mean([row["center_crc_valid"] for row in packet_rows]),
        "paper_crc_valid_rate": _mean([row["paper_crc_valid"] for row in packet_rows]),
        "paper_crc_gain_vs_center": _mean([row["paper_crc_gain_vs_center"] for row in packet_rows]),
        "uses_offset_coherence": False,
        "uses_packet_line_phase": False,
        "uses_top_l_locking": False,
        "uses_crc_guided_selection": False,
        "algorithm_source": "Savaux, A Low-Complexity Demodulation for Oversampled LoRa Signal, Eq. (34)-(37)",
    }


def main() -> int:
    args = parse_args()
    input_iq = args.input_iq.resolve()
    symbol_csv = args.gt_symbol_csv.resolve()
    output = args.output
    if output is None:
        output = WEAK_ROOT / "data" / "paper_oversampled_baseline" / f"{input_iq.stem}_paper_osr_packets.csv"
    symbols_output = args.symbols_output or output.with_name(output.stem + "_symbols.csv")
    summary_json = args.summary_json or output.with_suffix(".summary.json")

    samples = np.fromfile(input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {input_iq}")
    packets = load_packets(symbol_csv, packet_filter=args.packet, max_packets=args.max_packets)
    if not packets:
        raise SystemExit("no header-valid packets found in GT symbol CSV")

    packet_rows: list[dict[str, Any]] = []
    all_symbol_rows: list[dict[str, Any]] = []
    for packet in packets:
        print(f"packet {packet['packet_index']}: paper oversampled baseline", flush=True)
        packet_row, symbol_rows = evaluate_packet(samples, packet, args)
        packet_rows.append(packet_row)
        all_symbol_rows.extend(symbol_rows)
        print(
            "  center_ser={:.3f} paper_ser={:.3f} center_crc={} paper_crc={}".format(
                float(packet_row["center_symbol_ser"]),
                float(packet_row["paper_symbol_ser"]),
                int(packet_row["center_crc_valid"]),
                int(packet_row["paper_crc_valid"]),
            ),
            flush=True,
        )

    write_csv(output, packet_rows)
    write_csv(symbols_output, all_symbol_rows)
    summary = build_summary(packet_rows, args)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote={output}")
    print(f"wrote_symbols={symbols_output}")
    print(f"wrote_summary={summary_json}")
    print(
        "summary: packets={packet_count} center_ser={center_symbol_ser:.3f} "
        "paper_ser={paper_symbol_ser:.3f} crc={paper_crc_valid_rate:.3f}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
