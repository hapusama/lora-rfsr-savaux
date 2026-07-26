#!/usr/bin/env python3
"""运行弱包前导码滑窗检测。"""
# D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\detect_weak_preamble.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin -o gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_8_events.csv --windows-csv gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_8_windows.csv --sf 10 --bw 125000 --samp-rate 500000 --win-chirps 4 --min-periodic-peaks 6 --bin-tol 2
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


WEAK_ROOT = Path(__file__).resolve().parents[1]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.synchronization.preamble_detector import (  # noqa: E402
    DetectionEvent,
    PreambleDetectorConfig,
    WindowPeak,
    detect_preamble_runs,
    load_complex64_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按多个 chirp 的 dechirp+FFT 累加能量扫描前导码，"
            "只做检测，不做 CFO/STO 精估计和 payload 解码。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="raw complex64 IQ 文件。")
    parser.add_argument("-o", "--output", type=Path, required=True, help="检测事件 CSV 输出路径。")
    parser.add_argument("--windows-csv", type=Path, default=None, help="可选：输出每个滑窗的 peak 观测。")
    parser.add_argument("--sf", type=int, required=True, help="LoRa spreading factor。")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa 带宽 Hz，默认 125000。")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ 采样率 Hz，默认 500000。")
    parser.add_argument("--win-chirps", type=int, default=4, help="每个检测窗口累加多少个 chirp，默认 4。")
    parser.add_argument(
        "--hop-samples",
        type=int,
        default=None,
        help="滑窗步长，默认等于一个过采样 chirp 长度。",
    )
    parser.add_argument(
        "--min-periodic-peaks",
        type=int,
        default=6,
        help="至少连续多少个窗口 peak bin 稳定才报检测，默认 6。",
    )
    parser.add_argument("--bin-tol", type=int, default=2, help="peak bin 允许的循环距离偏差，默认 2。")
    parser.add_argument("--sample-limit", type=int, default=None, help="只扫描前 N 个 IQ sample，默认全文件。")
    parser.add_argument("--max-windows", type=int, default=None, help="最多扫描多少个窗口，便于快速调试。")
    return parser.parse_args()


def write_windows_csv(path: Path, windows: list[WindowPeak], config: PreambleDetectorConfig) -> None:
    """写出逐滑窗 peak 观测，方便肉眼检查 peak bin 是否成串稳定。"""

    fields = [
        "window_index",
        "start_sample",
        "end_sample",
        "peak_bin",
        "peak_bin_div_os",
        "peak_power",
        "second_power",
        "total_power",
        "confidence_db",
        "peak_share",
        "valid",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in windows:
            writer.writerow(
                {
                    "window_index": item.window_index,
                    "start_sample": item.start_sample,
                    "end_sample": item.end_sample,
                    "peak_bin": item.peak_bin,
                    "peak_bin_div_os": item.peak_bin / float(config.os_factor),
                    "peak_power": item.peak_power,
                    "second_power": item.second_power,
                    "total_power": item.total_power,
                    "confidence_db": item.confidence_db,
                    "peak_share": item.peak_share,
                    "valid": int(item.valid),
                }
            )


def write_events_csv(path: Path, events: list[DetectionEvent], config: PreambleDetectorConfig) -> None:
    """写出检测事件，每行是一段连续稳定 peak-bin 窗口串。"""

    fields = [
        "event_index",
        "start_sample",
        "end_sample",
        "first_window_index",
        "last_window_index",
        "window_count",
        "reference_bin",
        "reference_bin_div_os",
        "bin_min",
        "bin_max",
        "mean_peak_power",
        "mean_confidence_db",
        "max_peak_share",
        "sf",
        "bw",
        "samp_rate",
        "os_factor",
        "chirp_samples",
        "win_chirps",
        "hop_samples",
        "min_periodic_peaks",
        "bin_tol",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in events:
            writer.writerow(
                {
                    "event_index": item.event_index,
                    "start_sample": item.start_sample,
                    "end_sample": item.end_sample,
                    "first_window_index": item.first_window_index,
                    "last_window_index": item.last_window_index,
                    "window_count": item.window_count,
                    "reference_bin": item.reference_bin,
                    "reference_bin_div_os": item.reference_bin / float(config.os_factor),
                    "bin_min": item.bin_min,
                    "bin_max": item.bin_max,
                    "mean_peak_power": item.mean_peak_power,
                    "mean_confidence_db": item.mean_confidence_db,
                    "max_peak_share": item.max_peak_share,
                    "sf": config.sf,
                    "bw": config.bw,
                    "samp_rate": config.samp_rate,
                    "os_factor": config.os_factor,
                    "chirp_samples": config.chirp_samples,
                    "win_chirps": config.win_chirps,
                    "hop_samples": config.resolved_hop_samples,
                    "min_periodic_peaks": config.min_periodic_peaks,
                    "bin_tol": config.bin_tol,
                }
            )


def main() -> None:
    args = parse_args()
    config = PreambleDetectorConfig(
        sf=args.sf,
        bw=args.bw,
        samp_rate=args.samp_rate,
        win_chirps=args.win_chirps,
        hop_samples=args.hop_samples,
        min_periodic_peaks=args.min_periodic_peaks,
        bin_tol=args.bin_tol,
    )
    config.validate()

    samples = load_complex64_file(args.input)
    try:
        windows, events = detect_preamble_runs(
            samples,
            config,
            sample_limit=args.sample_limit,
            max_windows=args.max_windows,
        )
        write_events_csv(args.output, events, config)
        if args.windows_csv is not None:
            write_windows_csv(args.windows_csv, windows, config)
    finally:
        mmap_handle = getattr(samples, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    print(f"windows={len(windows)}")
    print(f"detections={len(events)}")
    print(f"chirp_samples={config.chirp_samples}")
    print(f"hop_samples={config.resolved_hop_samples}")
    print(f"wrote={args.output}")
    if args.windows_csv is not None:
        print(f"wrote_windows={args.windows_csv}")


if __name__ == "__main__":
    main()
