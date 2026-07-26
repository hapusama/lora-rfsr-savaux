#!/usr/bin/env python3
"""从同步候选开始，执行 header-first FFT demod 和 payload 一致性检查。"""
# D:\mysoft2\miniconda3\envs\gr-lora\python.exe weakPacket_decoding\scripts\run_header_first_demod.py -i data\USRP_IQ\0_0_0_10_14_16.bin -s weakPacket_decoding\data\weak_sync_chain\sync_chain\0_0_0_10_14_16_sync_chain.csv -o weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_symbols.csv --frames-output weakPacket_decoding\data\weak_sync_chain\header_first\0_0_0_10_14_16_header_first_frames.csv --sf 10 --bw 125000 --samp-rate 500000 --ldro-mode 2
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[1]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.decoding.header_first_demod import (  # noqa: E402
    HeaderDecodeResult,
    SymbolDemodResult,
    decode_explicit_header,
    demod_symbol_sequence,
)
from weak_decoder.synchronization.preamble_detector import load_complex64_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 run_weak_sync_chain.py 输出的 gr-lora framesync 候选，"
            "执行 header-first FFT demod。默认只处理 grlora_framesync_valid==1 的 frame。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="raw complex64 IQ 文件。")
    parser.add_argument("-s", "--sync-csv", type=Path, required=True, help="run_weak_sync_chain.py 输出的 sync_chain CSV。")
    parser.add_argument("-o", "--output", type=Path, required=True, help="逐 symbol FFT peak 输出 CSV。")
    parser.add_argument("--frames-output", type=Path, default=None, help="可选：逐 frame header 解码摘要 CSV。")
    parser.add_argument("--sf", type=int, default=10, help="LoRa SF，默认 10。")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa BW Hz，默认 125000。")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ 采样率 Hz，默认 500000。")
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO 模式：0 关，1 开，2 自动，默认 2。")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理多少个被选中的同步候选。")
    parser.add_argument(
        "--frame-filter",
        choices=("framesync-valid", "netid-valid", "frame-valid", "all"),
        default="framesync-valid",
        help=(
            "同步候选筛选方式：framesync-valid 只处理最终同步有效候选；"
            "netid-valid 只要求 netID 成组检查通过；frame-valid 只要求粗帧定界通过；"
            "all 会把所有检测候选都送进 FFT demod。默认 framesync-valid。"
        ),
    )
    parser.add_argument(
        "--include-invalid-header",
        action="store_true",
        default=False,
        help="header checksum 失败时仍保留该 frame 的 8 个 header symbol 输出。",
    )
    parser.add_argument(
        "--invalid-header-payload-policy",
        choices=("skip", "mode"),
        default="skip",
        help=(
            "header 无效时是否继续解 payload：skip 表示跳过 payload；"
            "mode 表示使用有效 header 中最常见的 payload symbol 数继续解调，"
            "适合做同一 bin 文件内的 payload FFT bin 一致性检查。默认 skip。"
        ),
    )
    parser.add_argument(
        "--consistency-output",
        type=Path,
        default=None,
        help="可选：输出 payload FFT bin 逐 symbol 位置的一致性检查 CSV。",
    )
    parser.add_argument(
        "--cfo-correction-mode",
        choices=("symbol", "continuous"),
        default="continuous",
        help=(
            "CFO 补偿模式：continuous 默认按整帧连续 chip 时间补偿公共 CFO 相位；"
            "symbol 为旧 gr-lora_sdr-like 口径，只在每个 symbol 内用 CFO downchirp 聚峰。"
            "默认 continuous。"
        ),
    )
    return parser.parse_args()


def _to_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return int(default)
    return int(float(value))


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return float(default)
    return float(value)


def _valid_flag(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "0")).strip() in {"1", "true", "True"}


def _passes_frame_filter(row: dict[str, str], frame_filter: str) -> bool:
    if frame_filter == "framesync-valid":
        return _valid_flag(row, "grlora_framesync_valid")
    if frame_filter == "netid-valid":
        return _valid_flag(row, "grlora_netid_valid")
    if frame_filter == "frame-valid":
        return _valid_flag(row, "frame_valid")
    if frame_filter == "all":
        return True
    raise ValueError(f"unknown frame filter: {frame_filter}")


def _header_start_sample(row: dict[str, str], os_factor: int) -> int:
    """优先使用 gr-lora 精同步后的 data 起点；这个位置就是 PHY header 第 0 个 symbol 起点。"""

    for key in ("header_start_sample", "grlora_fine_payload_start_sample"):
        if row.get(key, "") != "":
            return _to_int(row, key)
    if row.get("grlora_synced_payload_start_sample", "") != "":
        synced_start = _to_int(row, "grlora_synced_payload_start_sample")
        cfo_int = _to_int(row, "grlora_cfo_int_est")
        netid_offset = _to_int(row, "grlora_netid_offset") if _valid_flag(row, "grlora_netid_valid") else 0
        sto_correction = _to_int(row, "grlora_payload_sto_sample_correction")
        return int(synced_start + int(os_factor) * cfo_int - int(os_factor) * netid_offset - sto_correction)
    if row.get("located_payload_start_sample", "") != "":
        return _to_int(row, "located_payload_start_sample")
    raise ValueError("sync CSV row does not contain a usable header start sample.")


def load_sync_rows(path: Path, max_frames: int | None, os_factor: int, frame_filter: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if _passes_frame_filter(row, frame_filter)]
    rows.sort(key=lambda item: _header_start_sample(item, os_factor=os_factor))
    if max_frames is not None:
        rows = rows[: int(max_frames)]
    return rows


def _join_ints(values: tuple[int, ...] | list[int]) -> str:
    return " ".join(str(int(item)) for item in values)


def frame_summary_row(
    source_row: dict[str, str],
    frame_index: int,
    header_start_sample: int,
    header: HeaderDecodeResult | None,
    error: str = "",
) -> dict[str, object]:
    base = {
        "frame_index": int(frame_index),
        "packet_index": source_row.get("packet_index", ""),
        "event_index": source_row.get("event_index", ""),
        "header_start_sample": int(header_start_sample),
        "source_grlora_framesync_valid": int(_valid_flag(source_row, "grlora_framesync_valid")),
        "source_frame_valid": int(_valid_flag(source_row, "frame_valid")),
        "source_grlora_netid_valid": source_row.get("grlora_netid_valid", ""),
        "source_grlora_cfo_int": source_row.get("grlora_cfo_int_est", ""),
        "source_grlora_cfo_frac": source_row.get("grlora_cfo_frac_est", ""),
        "source_grlora_payload_sto_frac": source_row.get("grlora_payload_sto_frac_est", ""),
        "source_grlora_sfo_hat": source_row.get("grlora_sfo_hat", ""),
        "source_grlora_branch_sample_phases": source_row.get("grlora_branch_sample_phases", ""),
        "source_grlora_branch_valid": source_row.get("grlora_branch_valid", ""),
        "source_grlora_branch_payload_sto_frac": source_row.get("grlora_branch_payload_sto_frac_est", ""),
        "source_grlora_branch_payload_sto_sample_correction": source_row.get(
            "grlora_branch_payload_sto_sample_correction", ""
        ),
        "source_grlora_branch_sfo_hat": source_row.get("grlora_branch_sfo_hat", ""),
        "source_grlora_branch_sfo_cum_initial": source_row.get("grlora_branch_sfo_cum_initial", ""),
        "error": error,
    }
    if header is None:
        base.update(
            {
                "header_valid": 0,
                "payload_len": "",
                "cr": "",
                "has_crc": "",
                "ldro": "",
                "payload_symbol_count": "",
                "total_symbol_count": "",
                "header_checksum_received": "",
                "header_checksum_computed": "",
                "header_error": "",
                "header_nibbles": "",
                "gray_symbols": "",
                "codewords": "",
                "decoded_nibbles": "",
            }
        )
        return base

    base.update(
        {
            "header_valid": int(header.header_valid),
            "payload_len": int(header.payload_len),
            "cr": int(header.cr),
            "has_crc": int(header.has_crc),
            "ldro": int(header.ldro),
            "payload_symbol_count": int(header.payload_symbol_count),
            "total_symbol_count": int(header.total_symbol_count),
            "header_checksum_received": int(header.header_checksum_received),
            "header_checksum_computed": int(header.header_checksum_computed),
            "header_error": int(header.header_error),
            "header_nibbles": _join_ints(header.header_nibbles),
            "gray_symbols": _join_ints(header.gray_symbols),
            "codewords": _join_ints(header.codewords),
            "decoded_nibbles": _join_ints(header.decoded_nibbles),
        }
    )
    return base


def symbol_row(
    source_row: dict[str, str],
    frame_index: int,
    header: HeaderDecodeResult | None,
    symbol: SymbolDemodResult,
    sf: int,
    bw: float,
    os_factor: int,
) -> dict[str, object]:
    return {
        "frame_index": int(frame_index),
        "packet_index": source_row.get("packet_index", ""),
        "event_index": source_row.get("event_index", ""),
        "stage": symbol.stage,
        "frame_symbol_index": symbol.frame_symbol_index,
        "stage_symbol_index": symbol.stage_symbol_index,
        "start_sample": symbol.start_sample,
        "sf": int(sf),
        "bw": float(bw),
        "os_factor": int(os_factor),
        "cfo_int": source_row.get("grlora_cfo_int_est", ""),
        "cfo_frac": source_row.get("grlora_cfo_frac_est", ""),
        "cfo_correction_mode": symbol.cfo_correction_mode,
        "cfo_common_phase_rad": symbol.cfo_common_phase_rad,
        "cfo_common_phase_pi": symbol.cfo_common_phase_rad / np.pi,
        "sto_frac": source_row.get("grlora_payload_sto_frac_est", ""),
        "sfo_hat": source_row.get("grlora_sfo_hat", ""),
        "sfo_cum_before": symbol.sfo_cum_before,
        "source_grlora_branch_sample_phases": source_row.get("grlora_branch_sample_phases", ""),
        "source_grlora_branch_valid": source_row.get("grlora_branch_valid", ""),
        "source_grlora_branch_payload_sto_frac": source_row.get("grlora_branch_payload_sto_frac_est", ""),
        "source_grlora_branch_payload_sto_sample_correction": source_row.get(
            "grlora_branch_payload_sto_sample_correction", ""
        ),
        "source_grlora_branch_sfo_hat": source_row.get("grlora_branch_sfo_hat", ""),
        "source_grlora_branch_sfo_cum_initial": source_row.get("grlora_branch_sfo_cum_initial", ""),
        "sfo_sample_adjust_after": symbol.sfo_sample_adjust_after,
        "raw_fft_bin": symbol.raw_fft_bin,
        "signed_fft_bin": symbol.signed_fft_bin,
        "symbol_value": symbol.symbol_value,
        "peak_real": symbol.peak_real,
        "peak_imag": symbol.peak_imag,
        "peak_amp": symbol.peak_amp,
        "peak_power": symbol.peak_power,
        "peak_phase": symbol.peak_phase,
        "peak_margin_db": symbol.peak_margin_db,
        "total_power": symbol.total_power,
        "header_valid": "" if header is None else int(header.header_valid),
        "payload_len": "" if header is None else header.payload_len,
        "payload_cr": "" if header is None else header.cr,
        "payload_has_crc": "" if header is None else int(header.has_crc),
        "payload_ldro": "" if header is None else int(header.ldro),
    }


def payload_consistency_rows(symbol_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """按 payload symbol 位置检查所有候选包的 FFT bin 是否一致。"""

    by_symbol: dict[int, list[dict[str, object]]] = {}
    for row in symbol_rows:
        if row.get("stage") != "payload":
            continue
        by_symbol.setdefault(int(row.get("stage_symbol_index", 0)), []).append(row)

    rows: list[dict[str, object]] = []
    for stage_symbol_index in sorted(by_symbol):
        items = by_symbol[stage_symbol_index]
        raw_values = [int(item["raw_fft_bin"]) for item in items]
        signed_values = [int(item["signed_fft_bin"]) for item in items]
        symbol_values = [int(item["symbol_value"]) for item in items]

        raw_counter = Counter(raw_values)
        signed_counter = Counter(signed_values)
        symbol_counter = Counter(symbol_values)
        mode_raw, mode_raw_count = raw_counter.most_common(1)[0]
        mode_signed, mode_signed_count = signed_counter.most_common(1)[0]
        mode_symbol, mode_symbol_count = symbol_counter.most_common(1)[0]

        mismatch_items = [item for item in items if int(item["raw_fft_bin"]) != int(mode_raw)]
        rows.append(
            {
                "payload_symbol_index": int(stage_symbol_index),
                "frame_count": len(items),
                "mode_raw_fft_bin": int(mode_raw),
                "mode_signed_fft_bin": int(mode_signed),
                "mode_symbol_value": int(mode_symbol),
                "mode_raw_count": int(mode_raw_count),
                "mode_symbol_count": int(mode_symbol_count),
                "raw_mismatch_count": len(mismatch_items),
                "raw_unique_count": len(raw_counter),
                "symbol_unique_count": len(symbol_counter),
                "unique_raw_fft_bins": _join_ints(sorted(raw_counter)),
                "unique_symbol_values": _join_ints(sorted(symbol_counter)),
                "mismatch_frame_indices": _join_ints([int(item["frame_index"]) for item in mismatch_items]),
                "mismatch_packet_indices": _join_ints([int(item["packet_index"]) for item in mismatch_items if str(item.get("packet_index", "")) != ""]),
                "mismatch_event_indices": _join_ints([int(item["event_index"]) for item in mismatch_items if str(item.get("event_index", "")) != ""]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    ratio = float(args.samp_rate) / float(args.bw)
    os_factor = int(round(ratio))
    if os_factor <= 0 or abs(ratio - os_factor) > 1e-6:
        raise ValueError(f"--samp-rate / --bw must be an integer, got {ratio}.")

    samples = load_complex64_file(args.input)
    sync_rows = load_sync_rows(args.sync_csv, args.max_frames, os_factor=os_factor, frame_filter=args.frame_filter)
    symbol_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    prepared_frames: list[dict[str, object]] = []

    for frame_index, row in enumerate(sync_rows):
        frame_sf = _to_int(row, "sf", args.sf)
        frame_bw = _to_float(row, "bw", args.bw)
        frame_os_factor = _to_int(row, "os_factor", os_factor)
        header_start = _header_start_sample(row, os_factor=frame_os_factor)
        cfo_int = _to_int(row, "grlora_cfo_int_est")
        cfo_frac = _to_float(row, "grlora_cfo_frac_est")
        sfo_hat = _to_float(row, "grlora_sfo_hat")
        sfo_cum_initial = _to_float(row, "grlora_sfo_cum_initial")

        try:
            header_symbols = demod_symbol_sequence(
                samples=samples,
                header_start_sample=header_start,
                sf=frame_sf,
                os_factor=frame_os_factor,
                cfo_int=cfo_int,
                cfo_frac=cfo_frac,
                sfo_hat=sfo_hat,
                sfo_cum_initial=sfo_cum_initial,
                header_count=8,
                payload_count=0,
                payload_ldro=False,
                cfo_correction_mode=args.cfo_correction_mode,
            )
            header = decode_explicit_header(
                [item.symbol_value for item in header_symbols],
                sf=frame_sf,
                bw=frame_bw,
                ldro_mode=args.ldro_mode,
            )
            prepared_frames.append(
                {
                    "row": row,
                    "frame_index": frame_index,
                    "frame_sf": frame_sf,
                    "frame_bw": frame_bw,
                    "frame_os_factor": frame_os_factor,
                    "header_start": header_start,
                    "cfo_int": cfo_int,
                    "cfo_frac": cfo_frac,
                    "sfo_hat": sfo_hat,
                    "sfo_cum_initial": sfo_cum_initial,
                    "header": header,
                    "header_symbols": header_symbols,
                    "error": "",
                }
            )
        except Exception as exc:
            prepared_frames.append(
                {
                    "row": row,
                    "frame_index": frame_index,
                    "frame_sf": frame_sf,
                    "frame_bw": frame_bw,
                    "frame_os_factor": frame_os_factor,
                    "header_start": header_start,
                    "cfo_int": cfo_int,
                    "cfo_frac": cfo_frac,
                    "sfo_hat": sfo_hat,
                    "sfo_cum_initial": sfo_cum_initial,
                    "header": None,
                    "header_symbols": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    valid_payload_counts = [
        int(item["header"].payload_symbol_count)
        for item in prepared_frames
        if item.get("header") is not None and item["header"].header_valid
    ]
    fallback_payload_count = 0
    if args.invalid_header_payload_policy == "mode" and valid_payload_counts:
        fallback_payload_count = Counter(valid_payload_counts).most_common(1)[0][0]

    for item in prepared_frames:
        row = item["row"]
        frame_index = int(item["frame_index"])
        frame_sf = int(item["frame_sf"])
        frame_bw = float(item["frame_bw"])
        frame_os_factor = int(item["frame_os_factor"])
        header_start = int(item["header_start"])
        cfo_int = int(item["cfo_int"])
        cfo_frac = float(item["cfo_frac"])
        sfo_hat = float(item["sfo_hat"])
        sfo_cum_initial = float(item["sfo_cum_initial"])
        header = item["header"]
        error = str(item.get("error", ""))

        if error:
            frame_rows.append(frame_summary_row(row, frame_index, header_start, None, error=error))
            continue

        frame_rows.append(frame_summary_row(row, frame_index, header_start, header))
        if header is None:
            continue

        payload_count = int(header.payload_symbol_count) if header.header_valid else int(fallback_payload_count)
        payload_ldro = bool(header.ldro) if header.header_valid else False

        if header.header_valid or payload_count > 0:
            try:
                symbols = demod_symbol_sequence(
                    samples=samples,
                    header_start_sample=header_start,
                    sf=frame_sf,
                    os_factor=frame_os_factor,
                    cfo_int=cfo_int,
                    cfo_frac=cfo_frac,
                    sfo_hat=sfo_hat,
                    sfo_cum_initial=sfo_cum_initial,
                    header_count=8,
                    payload_count=payload_count,
                    payload_ldro=payload_ldro,
                    cfo_correction_mode=args.cfo_correction_mode,
                )
                for symbol in symbols:
                    symbol_rows.append(symbol_row(row, frame_index, header, symbol, frame_sf, frame_bw, frame_os_factor))
            except Exception as exc:
                frame_rows[-1]["error"] = f"{type(exc).__name__}: {exc}"
        elif args.include_invalid_header:
            for symbol in item["header_symbols"]:
                symbol_rows.append(symbol_row(row, frame_index, header, symbol, frame_sf, frame_bw, frame_os_factor))

    symbol_fields = [
        "frame_index",
        "packet_index",
        "event_index",
        "stage",
        "frame_symbol_index",
        "stage_symbol_index",
        "start_sample",
        "sf",
        "bw",
        "os_factor",
        "cfo_int",
        "cfo_frac",
        "cfo_correction_mode",
        "cfo_common_phase_rad",
        "cfo_common_phase_pi",
        "sto_frac",
        "sfo_hat",
        "sfo_cum_before",
        "source_grlora_branch_sample_phases",
        "source_grlora_branch_valid",
        "source_grlora_branch_payload_sto_frac",
        "source_grlora_branch_payload_sto_sample_correction",
        "source_grlora_branch_sfo_hat",
        "source_grlora_branch_sfo_cum_initial",
        "sfo_sample_adjust_after",
        "raw_fft_bin",
        "signed_fft_bin",
        "symbol_value",
        "peak_real",
        "peak_imag",
        "peak_amp",
        "peak_power",
        "peak_phase",
        "peak_margin_db",
        "total_power",
        "header_valid",
        "payload_len",
        "payload_cr",
        "payload_has_crc",
        "payload_ldro",
    ]
    frame_fields = [
        "frame_index",
        "packet_index",
        "event_index",
        "header_start_sample",
        "source_frame_valid",
        "source_grlora_framesync_valid",
        "source_grlora_netid_valid",
        "source_grlora_cfo_int",
        "source_grlora_cfo_frac",
        "source_grlora_payload_sto_frac",
        "source_grlora_sfo_hat",
        "source_grlora_branch_sample_phases",
        "source_grlora_branch_valid",
        "source_grlora_branch_payload_sto_frac",
        "source_grlora_branch_payload_sto_sample_correction",
        "source_grlora_branch_sfo_hat",
        "source_grlora_branch_sfo_cum_initial",
        "header_valid",
        "payload_len",
        "cr",
        "has_crc",
        "ldro",
        "payload_symbol_count",
        "total_symbol_count",
        "header_checksum_received",
        "header_checksum_computed",
        "header_error",
        "header_nibbles",
        "gray_symbols",
        "codewords",
        "decoded_nibbles",
        "error",
    ]
    write_csv(args.output, symbol_rows, symbol_fields)
    frames_output = args.frames_output or args.output.with_name(args.output.stem + "_frames.csv")
    write_csv(frames_output, frame_rows, frame_fields)
    consistency_output = args.consistency_output
    if consistency_output is not None:
        consistency_fields = [
            "payload_symbol_index",
            "frame_count",
            "mode_raw_fft_bin",
            "mode_signed_fft_bin",
            "mode_symbol_value",
            "mode_raw_count",
            "mode_symbol_count",
            "raw_mismatch_count",
            "raw_unique_count",
            "symbol_unique_count",
            "unique_raw_fft_bins",
            "unique_symbol_values",
            "mismatch_frame_indices",
            "mismatch_packet_indices",
            "mismatch_event_indices",
        ]
        write_csv(consistency_output, payload_consistency_rows(symbol_rows), consistency_fields)

    valid_headers = sum(int(row.get("header_valid", 0)) for row in frame_rows)
    payload_rows = sum(1 for row in symbol_rows if row.get("stage") == "payload")
    print(f"selected_candidates={len(sync_rows)}")
    print(f"frame_filter={args.frame_filter}")
    print(f"header_valid={valid_headers}/{len(frame_rows)}")
    if fallback_payload_count:
        print(f"fallback_payload_symbol_count={fallback_payload_count}")
    print(f"symbol_rows={len(symbol_rows)}")
    print(f"payload_rows={payload_rows}")
    print(f"wrote={args.output}")
    print(f"wrote_frames={frames_output}")
    if consistency_output is not None:
        print(f"wrote_consistency={consistency_output}")


if __name__ == "__main__":
    main()
