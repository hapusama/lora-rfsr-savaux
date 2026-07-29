#!/usr/bin/env python3
"""弱包同步链路入口：前导码检测、SFD 帧定界、gr-lora_sdr 风格粗同步验证。"""
# D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\run_weak_sync_chain.py -i gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_16.bin -o gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\sync_chain\0_0_0_10_14_16_sync_chain.csv --events-csv gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_16_events.csv --windows-csv gr-lora_sdr\weakPacket_decoding\data\weak_preamble_detections\0_0_0_10_14_16_windows.csv --bw 125000 --samp-rate 500000 --center-freq 487.7e6 --sync-word 0x34 --preamble-len 16 --win-chirps 4 --hop-chirps 1 --min-periodic-peaks 12 --frame-min-preamble-peaks 12 --stft-dir gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_16_stft --framesync-peaks-csv gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\framesync_peaks\0_0_0_10_14_16_framesync_peaks.csv --framesync-spectrum-dir gr-lora_sdr\weakPacket_decoding\data\weak_sync_chain\0_0_0_10_14_16_framesync_spectrum --framesync-spectrum-bin-span 96 --framesync-spectrum-chirps 8
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[1]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_upchirp, signed_fft_bin  # noqa: E402
from weak_decoder.synchronization.frame_locator import (  # noqa: E402
    FrameLocation,
    FrameLocatorConfig,
    locate_frame_from_event,
)
from weak_decoder.synchronization.grlora_frame_sync import (  # noqa: E402
    FrameSyncPeak,
    GrloraFrameSyncResult,
    run_grlora_frame_sync_validation,
)
from weak_decoder.synchronization.preamble_detector import (  # noqa: E402
    DetectionEvent,
    PreambleDetectorConfig,
    WindowPeak,
    detect_preamble_runs,
    load_complex64_file,
)
from weak_decoder.synchronization.single_packet import align_event_start  # noqa: E402


def parse_int_auto(text: str) -> int:
    """解析十进制或 0x 前缀十六进制整数。"""

    return int(str(text), 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "一体化弱包帧同步链路：滑窗检测前导码，搜索 sync word + SFD 做帧定界，"
            "再按 gr-lora_sdr 的 k_hat 粗同步方式验证前导码 peak 是否回到 bin0。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="raw complex64 IQ 文件。")
    parser.add_argument("-o", "--output", type=Path, required=True, help="帧同步结果 CSV。")
    parser.add_argument("--events-csv", type=Path, default=None, help="可选：另存检测事件 CSV。")
    parser.add_argument("--windows-csv", type=Path, default=None, help="可选：另存逐滑窗 peak CSV。")
    parser.add_argument("--framesync-peaks-csv", type=Path, default=None, help="可选：另存同步后逐符号 peak CSV。")
    parser.add_argument("--framesync-spectrum-dir", type=Path, default=None, help="可选：保存同步后前导码平均 FFT 频谱图和 CSV。")
    parser.add_argument("--framesync-spectrum-bin-span", type=int, default=96, help="前导码平均频谱图展示 bin0 附近正负多少个 bin。")
    parser.add_argument("--framesync-spectrum-chirps", type=int, default=8, help="平均频谱使用多少个同步后的前导码 upchirp。")
    parser.add_argument("--sf", type=int, default=None, help="LoRa SF。默认从文件名第 4 段推断。")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa 带宽 Hz，默认 125000。")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ 采样率 Hz，默认 500000。")
    parser.add_argument("--center-freq", type=float, default=487.7e6, help="RF 中心频率 Hz，用于由 CFO 推 SFO，默认 487.7e6。")
    parser.add_argument("--sync-word", type=parse_int_auto, default=0x34, help="LoRa sync word，默认 0x34。")
    parser.add_argument("--preamble-len", type=float, default=None, help="前导码 upchirp 数。默认从文件名最后一段推断。")
    parser.add_argument("--win-chirps", type=int, default=2, help="检测窗口内 chirp 数，默认 2。")
    parser.add_argument("--hop-chirps", type=float, default=1.0, help="滑窗步长，单位 chirp，默认 1。")
    parser.add_argument("--hop-samples", type=int, default=None, help="滑窗步长，单位 sample；给出后覆盖 --hop-chirps。")
    parser.add_argument(
        "--min-periodic-peaks",
        type=int,
        default=None,
        help="连续稳定窗口数。默认 preamble_len - win_chirps + 1。",
    )
    parser.add_argument("--bin-tol", type=int, default=2, help="检测阶段 peak bin 循环距离容差，默认 2。")
    parser.add_argument("--sample-limit", type=int, default=None, help="只扫描前 N 个 sample。")
    parser.add_argument("--max-windows", type=int, default=None, help="最多扫描多少个滑窗。")
    parser.add_argument("--max-events", type=int, default=None, help="最多处理多少个检测事件。")
    parser.add_argument(
        "--min-event-gap-chirps",
        type=float,
        default=None,
        help="检测事件去重间隔，单位 chirp；默认 preamble_len。",
    )
    parser.add_argument("--align-search-chirps", type=float, default=1.0, help="chirp 起点粗对齐搜索半径，单位 chirp。")
    parser.add_argument("--align-step-samples", type=int, default=1, help="chirp 起点粗对齐步长，单位 sample。")
    parser.add_argument("--align-chirps", type=int, default=None, help="粗对齐评分使用的 upchirp 数，默认 min(4, preamble_len)。")
    parser.add_argument("--frame-search-samples", type=int, default=None, help="SFD 定位搜索半径，默认 Ns/8。")
    parser.add_argument("--frame-step-samples", type=int, default=1, help="SFD 定位搜索步长，默认 1 sample。")
    parser.add_argument("--frame-preamble-bin-tol", type=int, default=2, help="帧定位阶段前导码稳定 bin 容差。")
    parser.add_argument("--frame-sync-bin-tol", type=int, default=4, help="sync word 相对 bin 容差。")
    parser.add_argument("--frame-sfd-bin-tol", type=int, default=4, help="两个 SFD downchirp bin 稳定容差。")
    parser.add_argument("--frame-min-preamble-peaks", type=int, default=None, help="帧定位阶段最少稳定前导码符号数。")
    parser.add_argument("--frame-symbol-search-span", type=int, default=2, help="SFD 定位额外搜索前后多少个整 chirp。")
    parser.add_argument("--framesync-bin0-tol", type=int, default=0, help="gr-lora 粗同步后前导码落入 bin0 的容差。")
    parser.add_argument("--stft-dir", type=Path, default=None, help="可选：保存 preamble+sync+SFD STFT 验证图。")
    return parser.parse_args(argv)


def infer_params_from_filename(path: Path) -> tuple[int | None, int | None]:
    """按 experiment_corridor_position_sf_txpower_preamble 的命名约定推断参数。"""

    parts = path.stem.split("_")
    sf = None
    preamble_len = None
    if len(parts) >= 6:
        try:
            sf = int(parts[3])
        except ValueError:
            sf = None
        try:
            preamble_len = int(parts[-1])
        except ValueError:
            preamble_len = None
    return sf, preamble_len


def resolve_positive_int(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required and could not be inferred from the filename.")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def resolve_chirp_samples(sf: int, bw: float, samp_rate: float) -> tuple[int, int]:
    ratio = float(samp_rate) / float(bw)
    os_factor = int(round(ratio))
    if os_factor <= 0 or not math.isclose(ratio, os_factor, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"samp_rate / bw must be an integer, got {ratio:.9g}.")
    return int((1 << int(sf)) * os_factor), os_factor


def select_spaced_events(
    events: list[DetectionEvent],
    chirp_samples: int,
    min_gap_chirps: float,
    max_events: int | None,
) -> list[DetectionEvent]:
    """按起点间隔做轻量去重，避免同一段前导码被重复处理。"""

    selected: list[DetectionEvent] = []
    min_gap_samples = int(round(float(min_gap_chirps) * int(chirp_samples)))
    last_start: int | None = None
    for event in sorted(events, key=lambda item: item.start_sample):
        if last_start is not None and event.start_sample - last_start < min_gap_samples:
            continue
        selected.append(event)
        last_start = int(event.start_sample)
        if max_events is not None and len(selected) >= int(max_events):
            break
    return selected


def write_windows_csv(path: Path, windows: list[WindowPeak], config: PreambleDetectorConfig) -> None:
    """写出检测阶段逐滑窗结果。"""

    fields = [
        "window_index",
        "start_sample",
        "end_sample",
        "peak_bin",
        "peak_signed_bin",
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
                    "peak_signed_bin": signed_fft_bin(item.peak_bin, config.chirp_samples) if item.peak_bin >= 0 else "",
                    "peak_power": item.peak_power,
                    "second_power": item.second_power,
                    "total_power": item.total_power,
                    "confidence_db": item.confidence_db,
                    "peak_share": item.peak_share,
                    "valid": int(item.valid),
                }
            )


def write_events_csv(path: Path, events: list[DetectionEvent], config: PreambleDetectorConfig) -> None:
    """写出检测事件。"""

    fields = [
        "event_index",
        "start_sample",
        "end_sample",
        "first_window_index",
        "last_window_index",
        "window_count",
        "reference_bin",
        "reference_signed_bin",
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
                    "reference_signed_bin": signed_fft_bin(item.reference_bin, config.chirp_samples),
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


def write_chain_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """写出一体化帧同步链路结果。"""

    fields = [
        "packet_index",
        "event_index",
        "detected_start_sample",
        "detected_end_sample",
        "detected_window_count",
        "detected_reference_bin",
        "detected_reference_signed_bin",
        "detected_mean_confidence_db",
        "aligned_start_sample",
        "align_offset_samples",
        "align_peak_bin",
        "align_peak_signed_bin",
        "align_confidence_db",
        "align_peak_share",
        "frame_valid",
        "located_preamble_start_sample",
        "located_sfd_start_sample",
        "located_payload_start_sample",
        "locator_score",
        "preamble_ref_bin",
        "preamble_ref_signed_bin",
        "preamble_stable_count",
        "sync1_bin",
        "sync2_bin",
        "sync1_expected_bin",
        "sync2_expected_bin",
        "sync1_distance",
        "sync2_distance",
        "sfd1_bin",
        "sfd2_bin",
        "sfd_bin_distance",
        "mean_preamble_confidence_db",
        "mean_sfd_confidence_db",
        "grlora_framesync_valid",
        "grlora_coarse_offset_chips",
        "grlora_coarse_offset_samples",
        "grlora_synced_preamble_start_sample",
        "grlora_synced_sfd_start_sample",
        "grlora_synced_payload_start_sample",
        "grlora_fine_preamble_start_sample",
        "grlora_fine_payload_start_sample",
        "grlora_preamble_peak_mean_signed_bin",
        "grlora_preamble_peak_max_abs_signed_bin",
        "grlora_preamble_bin0_count",
        "grlora_preamble_peak_count",
        "grlora_sync1_peak_signed_bin",
        "grlora_sync2_peak_signed_bin",
        "grlora_sync1_expected_signed_bin",
        "grlora_sync2_expected_signed_bin",
        "grlora_sync1_distance",
        "grlora_sync2_distance",
        "grlora_sfd1_peak_signed_bin",
        "grlora_sfd2_peak_signed_bin",
        "grlora_sfd_mean_signed_bin",
        "grlora_up_symbols_used",
        "grlora_cfo_frac_est",
        "grlora_sto_frac_initial",
        "grlora_sto_frac_refined",
        "grlora_sto_frac_used",
        "grlora_sto_sample_correction",
        "grlora_cfo_int_est",
        "grlora_down_val_signed_bin",
        "grlora_cfo_total_est",
        "grlora_cfo_hz_est",
        "grlora_sfo_hat",
        "grlora_sfo_samples_per_symbol",
        "grlora_clk_off",
        "grlora_fs_p",
        "grlora_netid_sto_frac_est",
        "grlora_payload_sto_frac_est",
        "grlora_payload_sto_sample_correction",
        "grlora_netid1_est",
        "grlora_netid2_est",
        "grlora_netid_offset",
        "grlora_netid_valid",
        "grlora_sfo_cum_initial",
        "grlora_fine_preamble_peak_mean_signed_bin",
        "grlora_fine_preamble_peak_max_abs_signed_bin",
        "grlora_fine_preamble_bin0_count",
        "grlora_fine_preamble_peak_count",
        "grlora_center_sample_phase",
        "grlora_branch_sample_phases",
        "grlora_branch_valid",
        "grlora_branch_down_val_valid",
        "grlora_branch_cfo_frac_est",
        "grlora_branch_sto_frac_initial",
        "grlora_branch_sto_frac_refined",
        "grlora_branch_sto_frac_used",
        "grlora_branch_sto_sample_correction",
        "grlora_branch_cfo_int_est",
        "grlora_branch_down_val_signed_bin",
        "grlora_branch_cfo_total_est",
        "grlora_branch_cfo_hz_est",
        "grlora_branch_sfo_hat",
        "grlora_branch_sfo_samples_per_symbol",
        "grlora_branch_clk_off",
        "grlora_branch_fs_p",
        "grlora_branch_netid_sto_frac_est",
        "grlora_branch_payload_sto_frac_est",
        "grlora_branch_payload_sto_sample_correction",
        "grlora_branch_netid1_est",
        "grlora_branch_netid2_est",
        "grlora_branch_netid_offset",
        "grlora_branch_netid_valid",
        "grlora_branch_sfo_cum_initial",
        "grlora_branch_fine_preamble_start_sample",
        "grlora_branch_fine_payload_start_sample",
        "grlora_spectrum_chirps",
        "grlora_spectrum_raw_peak_signed_bin",
        "grlora_spectrum_peak_signed_bin",
        "grlora_spectrum_peak_relative_db",
        "grlora_spectrum_bin0_relative_db",
        "grlora_spectrum_bin0_is_peak",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_framesync_peaks_csv(path: Path, rows: list[tuple[int, int, FrameSyncPeak]]) -> None:
    """写出 gr-lora 粗同步后每个验证符号的 peak。"""

    fields = [
        "packet_index",
        "event_index",
        "stage",
        "symbol_index",
        "start_sample",
        "peak_bin",
        "signed_peak_bin",
        "expected_signed_bin",
        "distance_to_expected",
        "peak_power",
        "second_power",
        "confidence_db",
        "peak_share",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for packet_index, event_index, peak in rows:
            writer.writerow(
                {
                    "packet_index": int(packet_index),
                    "event_index": int(event_index),
                    "stage": peak.stage,
                    "symbol_index": peak.symbol_index,
                    "start_sample": peak.start_sample,
                    "peak_bin": peak.peak_bin,
                    "signed_peak_bin": peak.signed_peak_bin,
                    "expected_signed_bin": "" if peak.expected_signed_bin is None else peak.expected_signed_bin,
                    "distance_to_expected": "" if peak.distance_to_expected is None else peak.distance_to_expected,
                    "peak_power": peak.peak_power,
                    "second_power": peak.second_power,
                    "confidence_db": peak.confidence_db,
                    "peak_share": peak.peak_share,
                }
            )


def _join_vector(values: list[object] | tuple[object, ...]) -> str:
    return "|".join(str(item) for item in values)


def _branch_vector(frame_sync: GrloraFrameSyncResult, attr: str) -> str:
    return _join_vector([getattr(item, attr) for item in frame_sync.branch_sync_estimates])


def result_to_row(
    packet_index: int,
    event: DetectionEvent,
    alignment: dict[str, float | int],
    frame_location: FrameLocation,
    frame_sync: GrloraFrameSyncResult,
    detector_config: PreambleDetectorConfig,
) -> dict[str, object]:
    """合并检测、帧定界、gr-lora 粗同步验证三层结果。"""

    return {
        "packet_index": int(packet_index),
        "event_index": int(event.event_index),
        "detected_start_sample": int(event.start_sample),
        "detected_end_sample": int(event.end_sample),
        "detected_window_count": int(event.window_count),
        "detected_reference_bin": int(event.reference_bin),
        "detected_reference_signed_bin": signed_fft_bin(event.reference_bin, detector_config.chirp_samples),
        "detected_mean_confidence_db": float(event.mean_confidence_db),
        "aligned_start_sample": alignment["aligned_start_sample"],
        "align_offset_samples": alignment["align_offset_samples"],
        "align_peak_bin": alignment["align_peak_bin"],
        "align_peak_signed_bin": alignment["align_peak_signed_bin"],
        "align_confidence_db": alignment["align_confidence_db"],
        "align_peak_share": alignment["align_peak_share"],
        "frame_valid": int(frame_location.valid),
        "located_preamble_start_sample": int(frame_location.preamble_start_sample),
        "located_sfd_start_sample": int(frame_location.sfd_start_sample),
        "located_payload_start_sample": int(frame_location.payload_start_sample),
        "locator_score": float(frame_location.score),
        "preamble_ref_bin": int(frame_location.preamble_ref_bin),
        "preamble_ref_signed_bin": signed_fft_bin(frame_location.preamble_ref_bin, detector_config.chirp_samples),
        "preamble_stable_count": int(frame_location.preamble_stable_count),
        "sync1_bin": int(frame_location.sync1_bin),
        "sync2_bin": int(frame_location.sync2_bin),
        "sync1_expected_bin": int(frame_location.sync1_expected_bin),
        "sync2_expected_bin": int(frame_location.sync2_expected_bin),
        "sync1_distance": int(frame_location.sync1_distance),
        "sync2_distance": int(frame_location.sync2_distance),
        "sfd1_bin": int(frame_location.sfd1_bin),
        "sfd2_bin": int(frame_location.sfd2_bin),
        "sfd_bin_distance": int(frame_location.sfd_bin_distance),
        "mean_preamble_confidence_db": float(frame_location.mean_preamble_confidence_db),
        "mean_sfd_confidence_db": float(frame_location.mean_sfd_confidence_db),
        "grlora_framesync_valid": int(frame_sync.valid),
        "grlora_coarse_offset_chips": frame_sync.coarse_offset_chips,
        "grlora_coarse_offset_samples": frame_sync.coarse_offset_samples,
        "grlora_synced_preamble_start_sample": frame_sync.synced_preamble_start_sample,
        "grlora_synced_sfd_start_sample": frame_sync.synced_sfd_start_sample,
        "grlora_synced_payload_start_sample": frame_sync.synced_payload_start_sample,
        "grlora_fine_preamble_start_sample": frame_sync.fine_preamble_start_sample,
        "grlora_fine_payload_start_sample": frame_sync.fine_payload_start_sample,
        "grlora_preamble_peak_mean_signed_bin": frame_sync.preamble_peak_mean_signed_bin,
        "grlora_preamble_peak_max_abs_signed_bin": frame_sync.preamble_peak_max_abs_signed_bin,
        "grlora_preamble_bin0_count": frame_sync.preamble_bin0_count,
        "grlora_preamble_peak_count": frame_sync.preamble_peak_count,
        "grlora_sync1_peak_signed_bin": frame_sync.sync1_peak_signed_bin,
        "grlora_sync2_peak_signed_bin": frame_sync.sync2_peak_signed_bin,
        "grlora_sync1_expected_signed_bin": frame_sync.sync1_expected_signed_bin,
        "grlora_sync2_expected_signed_bin": frame_sync.sync2_expected_signed_bin,
        "grlora_sync1_distance": frame_sync.sync1_distance,
        "grlora_sync2_distance": frame_sync.sync2_distance,
        "grlora_sfd1_peak_signed_bin": frame_sync.sfd1_peak_signed_bin,
        "grlora_sfd2_peak_signed_bin": frame_sync.sfd2_peak_signed_bin,
        "grlora_sfd_mean_signed_bin": frame_sync.sfd_mean_signed_bin,
        "grlora_up_symbols_used": frame_sync.up_symbols_used,
        "grlora_cfo_frac_est": frame_sync.cfo_frac_est,
        "grlora_sto_frac_initial": frame_sync.sto_frac_initial,
        "grlora_sto_frac_refined": frame_sync.sto_frac_refined,
        "grlora_sto_frac_used": frame_sync.sto_frac_used,
        "grlora_sto_sample_correction": frame_sync.sto_sample_correction,
        "grlora_cfo_int_est": frame_sync.cfo_int_est,
        "grlora_down_val_signed_bin": frame_sync.down_val_signed_bin,
        "grlora_cfo_total_est": frame_sync.cfo_total_est,
        "grlora_cfo_hz_est": frame_sync.cfo_hz_est,
        "grlora_sfo_hat": frame_sync.sfo_hat,
        "grlora_sfo_samples_per_symbol": frame_sync.sfo_samples_per_symbol,
        "grlora_clk_off": frame_sync.clk_off,
        "grlora_fs_p": frame_sync.fs_p,
        "grlora_netid_sto_frac_est": frame_sync.netid_sto_frac_est,
        "grlora_payload_sto_frac_est": frame_sync.payload_sto_frac_est,
        "grlora_payload_sto_sample_correction": frame_sync.payload_sto_sample_correction,
        "grlora_netid1_est": frame_sync.netid1_est,
        "grlora_netid2_est": frame_sync.netid2_est,
        "grlora_netid_offset": frame_sync.netid_offset,
        "grlora_netid_valid": int(frame_sync.netid_valid),
        "grlora_sfo_cum_initial": frame_sync.sfo_cum_initial,
        "grlora_fine_preamble_peak_mean_signed_bin": frame_sync.fine_preamble_peak_mean_signed_bin,
        "grlora_fine_preamble_peak_max_abs_signed_bin": frame_sync.fine_preamble_peak_max_abs_signed_bin,
        "grlora_fine_preamble_bin0_count": frame_sync.fine_preamble_bin0_count,
        "grlora_fine_preamble_peak_count": frame_sync.fine_preamble_peak_count,
        "grlora_center_sample_phase": int(detector_config.os_factor / 2),
        "grlora_branch_sample_phases": _branch_vector(frame_sync, "sample_phase"),
        "grlora_branch_valid": _join_vector([int(item.valid) for item in frame_sync.branch_sync_estimates]),
        "grlora_branch_down_val_valid": _join_vector([int(item.down_val_valid) for item in frame_sync.branch_sync_estimates]),
        "grlora_branch_cfo_frac_est": _branch_vector(frame_sync, "cfo_frac_est"),
        "grlora_branch_sto_frac_initial": _branch_vector(frame_sync, "sto_frac_initial"),
        "grlora_branch_sto_frac_refined": _branch_vector(frame_sync, "sto_frac_refined"),
        "grlora_branch_sto_frac_used": _branch_vector(frame_sync, "sto_frac_used"),
        "grlora_branch_sto_sample_correction": _branch_vector(frame_sync, "sto_sample_correction"),
        "grlora_branch_cfo_int_est": _branch_vector(frame_sync, "cfo_int_est"),
        "grlora_branch_down_val_signed_bin": _branch_vector(frame_sync, "down_val_signed_bin"),
        "grlora_branch_cfo_total_est": _branch_vector(frame_sync, "cfo_total_est"),
        "grlora_branch_cfo_hz_est": _branch_vector(frame_sync, "cfo_hz_est"),
        "grlora_branch_sfo_hat": _branch_vector(frame_sync, "sfo_hat"),
        "grlora_branch_sfo_samples_per_symbol": _branch_vector(frame_sync, "sfo_samples_per_symbol"),
        "grlora_branch_clk_off": _branch_vector(frame_sync, "clk_off"),
        "grlora_branch_fs_p": _branch_vector(frame_sync, "fs_p"),
        "grlora_branch_netid_sto_frac_est": _branch_vector(frame_sync, "netid_sto_frac_est"),
        "grlora_branch_payload_sto_frac_est": _branch_vector(frame_sync, "payload_sto_frac_est"),
        "grlora_branch_payload_sto_sample_correction": _branch_vector(frame_sync, "payload_sto_sample_correction"),
        "grlora_branch_netid1_est": _branch_vector(frame_sync, "netid1_est"),
        "grlora_branch_netid2_est": _branch_vector(frame_sync, "netid2_est"),
        "grlora_branch_netid_offset": _branch_vector(frame_sync, "netid_offset"),
        "grlora_branch_netid_valid": _join_vector([int(item.netid_valid) for item in frame_sync.branch_sync_estimates]),
        "grlora_branch_sfo_cum_initial": _branch_vector(frame_sync, "sfo_cum_initial"),
        "grlora_branch_fine_preamble_start_sample": _branch_vector(frame_sync, "fine_preamble_start_sample"),
        "grlora_branch_fine_payload_start_sample": _branch_vector(frame_sync, "fine_payload_start_sample"),
    }


def write_frame_stft_plot(
    samples: np.ndarray,
    frame_location: FrameLocation,
    detector_config: PreambleDetectorConfig,
    preamble_len: float,
    output_path: Path,
) -> None:
    """把物理帧定界出的 preamble+sync+SFD 区间画成 STFT 验证图。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = int(frame_location.preamble_start_sample)
    stop = int(frame_location.payload_start_sample)
    if start < 0 or stop <= start or stop > samples.size:
        raise ValueError("invalid STFT sample range.")

    segment = np.asarray(samples[start:stop], dtype=np.complex64)
    chirp_samples = detector_config.chirp_samples
    nperseg = min(max(128, chirp_samples // 16), max(128, segment.size // 4))
    nperseg = min(nperseg, segment.size)
    noverlap = int(round(nperseg * 0.75))
    hop = max(1, nperseg - noverlap)
    nfft = max(1024, int(2 ** math.ceil(math.log2(max(nperseg * 4, 2)))))
    starts = np.arange(0, segment.size - nperseg + 1, hop, dtype=np.int64)
    if starts.size == 0:
        starts = np.asarray([0], dtype=np.int64)

    window = np.hanning(nperseg).astype(np.float32)
    frames = np.empty((starts.size, nperseg), dtype=np.complex64)
    for row, offset in enumerate(starts):
        frames[row, :] = segment[offset : offset + nperseg] * window

    spec = np.fft.fftshift(np.fft.fft(frames, n=nfft, axis=1), axes=1)
    spec_db = 20.0 * np.log10(np.maximum(np.abs(spec).T, 1e-12))
    spec_db -= float(np.max(spec_db))
    times_ms = (starts + nperseg / 2.0) / detector_config.samp_rate * 1e3
    freqs_khz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / detector_config.samp_rate)) / 1e3

    fig, ax = plt.subplots(figsize=(12, 4), dpi=160)
    image = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        extent=[times_ms[0], times_ms[-1], freqs_khz[0], freqs_khz[-1]],
        cmap="viridis",
        vmin=-75,
        vmax=0,
    )
    symbol_ms = chirp_samples / detector_config.samp_rate * 1e3
    boundaries = [
        (float(preamble_len) * symbol_ms, "sync"),
        ((float(preamble_len) + 2.0) * symbol_ms, "SFD"),
        ((float(preamble_len) + 4.0) * symbol_ms, "quarter"),
        ((float(preamble_len) + 4.25) * symbol_ms, "payload"),
    ]
    for x_ms, label in boundaries:
        ax.axvline(x_ms, color="white", linewidth=0.9, alpha=0.85)
        ax.text(x_ms, freqs_khz[-1] * 0.92, label, color="white", fontsize=8, rotation=90, va="top")

    for idx in range(1, int(math.floor(float(preamble_len) + 4.25)) + 1):
        ax.axvline(idx * symbol_ms, color="white", linewidth=0.35, alpha=0.35)

    ax.set_title(
        "Located LoRa preamble + sync + SFD "
        f"(packet event {frame_location.event_index}, valid={int(frame_location.valid)})"
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (kHz)")
    cbar = fig.colorbar(image, ax=ax, pad=0.01)
    cbar.set_label("Relative power (dB)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _power_to_relative_db(power: np.ndarray) -> np.ndarray:
    """把功率谱归一化成相对 dB，最大值为 0 dB。"""

    values = np.asarray(power, dtype=np.float64)
    peak = float(np.max(values)) if values.size else 0.0
    if peak <= 0.0:
        return np.full(values.shape, -300.0, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(values, 1e-300) / peak)


def write_framesync_preamble_spectrum(
    samples: np.ndarray,
    frame_location: FrameLocation,
    frame_sync: GrloraFrameSyncResult,
    detector_config: PreambleDetectorConfig,
    preamble_len: float,
    packet_index: int,
    output_png: Path,
    output_csv: Path,
    bin_span: int,
    chirp_count: int,
) -> dict[str, object]:
    """画出 gr-lora 同步前后前若干个前导码 dechirp+FFT 的平均功率谱。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fft_len = detector_config.chirp_samples
    use_chirps = min(int(chirp_count), int(round(float(preamble_len))))
    if use_chirps <= 0:
        raise ValueError("framesync spectrum needs at least one preamble chirp.")

    upchirp = build_upchirp(
        detector_config.sf,
        symbol_id=0,
        os_factor=detector_config.os_factor,
    )
    down_ref = np.conjugate(upchirp).astype(np.complex64)

    def average_power_from_start(start_sample: int) -> np.ndarray:
        spectra = []
        for chirp_index in range(use_chirps):
            start = int(start_sample + chirp_index * fft_len)
            stop = start + fft_len
            if start < 0 or stop > samples.size:
                raise ValueError(f"packet {packet_index} does not have enough preamble samples.")
            segment = np.asarray(samples[start:stop], dtype=np.complex64)
            spectra.append(np.fft.fft(segment * down_ref))
        return np.mean(np.abs(np.asarray(spectra)) ** 2, axis=0, dtype=np.float64)

    raw_power = average_power_from_start(int(frame_location.preamble_start_sample))
    synced_power = average_power_from_start(int(frame_sync.synced_preamble_start_sample))
    raw_rel_db = _power_to_relative_db(raw_power)
    synced_rel_db = _power_to_relative_db(synced_power)
    bin_index = np.arange(fft_len, dtype=np.int64)
    signed_bins = np.asarray([signed_fft_bin(int(item), fft_len) for item in bin_index], dtype=np.int64)
    order = np.argsort(signed_bins)
    signed_sorted = signed_bins[order]
    raw_db_sorted = raw_rel_db[order]
    synced_db_sorted = synced_rel_db[order]
    span = max(1, int(bin_span))
    mask = (signed_sorted >= -span) & (signed_sorted <= span)

    raw_peak_bin = int(np.argmax(raw_power))
    raw_peak_signed = signed_fft_bin(raw_peak_bin, fft_len)
    synced_peak_bin = int(np.argmax(synced_power))
    synced_peak_signed = signed_fft_bin(synced_peak_bin, fft_len)
    synced_bin0_db = float(synced_rel_db[0])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "packet_index",
                "chirps",
                "bin_index",
                "signed_bin",
                "raw_avg_power",
                "raw_relative_db",
                "raw_peak_signed_bin",
                "synced_avg_power",
                "synced_relative_db",
                "synced_peak_signed_bin",
                "synced_bin0_relative_db",
            ],
        )
        writer.writeheader()
        for idx in range(fft_len):
            writer.writerow(
                {
                    "packet_index": int(packet_index),
                    "chirps": int(use_chirps),
                    "bin_index": int(bin_index[idx]),
                    "signed_bin": int(signed_bins[idx]),
                    "raw_avg_power": float(raw_power[idx]),
                    "raw_relative_db": float(raw_rel_db[idx]),
                    "raw_peak_signed_bin": int(raw_peak_signed),
                    "synced_avg_power": float(synced_power[idx]),
                    "synced_relative_db": float(synced_rel_db[idx]),
                    "synced_peak_signed_bin": int(synced_peak_signed),
                    "synced_bin0_relative_db": synced_bin0_db,
                }
            )

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), dpi=160, sharex=True)
    axes[0].plot(signed_sorted[mask], raw_db_sorted[mask], color="#1f77b4", linewidth=1.2)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.75)
    axes[0].axvline(raw_peak_signed, color="#d62728", linestyle=":", linewidth=0.9)
    axes[0].set_ylabel("Relative power (dB)")
    axes[0].set_title(
        "Before gr-lora framesync: "
        f"{use_chirps} preamble chirps avg, peak signed bin {raw_peak_signed}"
    )
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(signed_sorted[mask], synced_db_sorted[mask], color="#2ca02c", linewidth=1.2)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.75)
    axes[1].axvline(synced_peak_signed, color="#d62728", linestyle=":", linewidth=0.9)
    axes[1].set_title(
        "After gr-lora coarse timing sync: "
        f"{use_chirps} preamble chirps avg, peak signed bin {synced_peak_signed}"
    )
    axes[1].set_xlabel("Signed FFT bin")
    axes[1].set_ylabel("Relative power (dB)")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(
        f"Packet {packet_index:03d} preamble dechirp+FFT spectrum comparison",
        fontsize=11,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)

    return {
        "grlora_spectrum_chirps": int(use_chirps),
        "grlora_spectrum_raw_peak_signed_bin": int(raw_peak_signed),
        "grlora_spectrum_peak_signed_bin": int(synced_peak_signed),
        "grlora_spectrum_peak_relative_db": float(synced_rel_db[synced_peak_bin]),
        "grlora_spectrum_bin0_relative_db": synced_bin0_db,
        "grlora_spectrum_bin0_is_peak": int(synced_peak_signed == 0),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    inferred_sf, inferred_preamble_len = infer_params_from_filename(args.input)
    sf = resolve_positive_int(args.sf if args.sf is not None else inferred_sf, "sf")
    resolved_preamble_len = args.preamble_len if args.preamble_len is not None else inferred_preamble_len
    if resolved_preamble_len is None:
        raise ValueError("preamble_len is required and could not be inferred from the filename.")
    preamble_len = float(resolved_preamble_len)
    if preamble_len <= 0.0:
        raise ValueError("preamble_len must be positive and could not be inferred from the filename.")

    chirp_samples, _ = resolve_chirp_samples(sf, args.bw, args.samp_rate)
    hop_samples = int(args.hop_samples) if args.hop_samples is not None else int(round(args.hop_chirps * chirp_samples))
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive.")
    min_periodic_peaks = (
        int(args.min_periodic_peaks)
        if args.min_periodic_peaks is not None
        else max(2, int(round(preamble_len)) - int(args.win_chirps) + 1)
    )
    align_chirps = int(args.align_chirps) if args.align_chirps is not None else max(1, min(4, int(round(preamble_len))))
    frame_search_samples = (
        int(args.frame_search_samples)
        if args.frame_search_samples is not None
        else max(1, int(round(chirp_samples / 8)))
    )
    min_event_gap_chirps = float(args.min_event_gap_chirps) if args.min_event_gap_chirps is not None else float(preamble_len)

    detector_config = PreambleDetectorConfig(
        sf=sf,
        bw=args.bw,
        samp_rate=args.samp_rate,
        win_chirps=args.win_chirps,
        hop_samples=hop_samples,
        min_periodic_peaks=min_periodic_peaks,
        bin_tol=args.bin_tol,
    )
    detector_config.validate()
    locator_config = FrameLocatorConfig(
        preamble_len=preamble_len,
        sync_word=args.sync_word,
        search_radius_samples=frame_search_samples,
        step_samples=args.frame_step_samples,
        preamble_bin_tol=args.frame_preamble_bin_tol,
        sync_bin_tol=args.frame_sync_bin_tol,
        sfd_bin_tol=args.frame_sfd_bin_tol,
        min_preamble_peaks=args.frame_min_preamble_peaks,
        symbol_search_span=args.frame_symbol_search_span,
    )
    locator_config.validate()

    samples = load_complex64_file(args.input)
    rows: list[dict[str, object]] = []
    framesync_peak_rows: list[tuple[int, int, FrameSyncPeak]] = []
    try:
        windows, events = detect_preamble_runs(
            samples,
            detector_config,
            sample_limit=args.sample_limit,
            max_windows=args.max_windows,
        )
        selected_events = select_spaced_events(
            events,
            detector_config.chirp_samples,
            min_event_gap_chirps,
            args.max_events,
        )
        search_radius = int(round(float(args.align_search_chirps) * detector_config.chirp_samples))
        for packet_index, event in enumerate(selected_events):
            alignment = align_event_start(
                samples,
                event,
                detector_config,
                search_radius_samples=search_radius,
                step_samples=args.align_step_samples,
                align_chirps=align_chirps,
            )
            frame_location = locate_frame_from_event(
                samples,
                event,
                detector_config,
                locator_config,
                coarse_start_sample=int(alignment["aligned_start_sample"]),
            )
            frame_sync = run_grlora_frame_sync_validation(
                samples,
                frame_location,
                detector_config,
                preamble_len,
                args.sync_word,
                bin0_tol=args.framesync_bin0_tol,
                center_freq=args.center_freq,
            )
            row = result_to_row(
                packet_index,
                event,
                alignment,
                frame_location,
                frame_sync,
                detector_config,
            )
            if args.framesync_spectrum_dir is not None:
                stem = f"packet_{packet_index:03d}_event_{event.event_index:03d}_framesync_preamble_spectrum"
                spectrum_summary = write_framesync_preamble_spectrum(
                    samples,
                    frame_location,
                    frame_sync,
                    detector_config,
                    preamble_len,
                    packet_index,
                    args.framesync_spectrum_dir / f"{stem}.png",
                    args.framesync_spectrum_dir / f"{stem}.csv",
                    args.framesync_spectrum_bin_span,
                    args.framesync_spectrum_chirps,
                )
                row.update(spectrum_summary)
            rows.append(row)
            framesync_peak_rows.extend(
                (packet_index, int(event.event_index), peak)
                for peak in frame_sync.peaks
            )
            if args.stft_dir is not None:
                write_frame_stft_plot(
                    samples,
                    frame_location,
                    detector_config,
                    preamble_len,
                    args.stft_dir / f"packet_{packet_index:03d}_event_{event.event_index:03d}_stft.png",
                )

        write_chain_csv(args.output, rows)
        if args.events_csv is not None:
            write_events_csv(args.events_csv, events, detector_config)
        if args.windows_csv is not None:
            write_windows_csv(args.windows_csv, windows, detector_config)
        if args.framesync_peaks_csv is not None:
            write_framesync_peaks_csv(args.framesync_peaks_csv, framesync_peak_rows)
    finally:
        mmap_handle = getattr(samples, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    valid_frames = sum(int(row["frame_valid"]) for row in rows)
    valid_fsync = sum(int(row["grlora_framesync_valid"]) for row in rows)
    max_abs_bins = [
        int(row["grlora_preamble_peak_max_abs_signed_bin"])
        for row in rows
        if row.get("grlora_preamble_peak_max_abs_signed_bin") != ""
    ]
    max_abs_bin = max(max_abs_bins) if max_abs_bins else ""
    fine_max_abs_bins = [
        int(row["grlora_fine_preamble_peak_max_abs_signed_bin"])
        for row in rows
        if row.get("grlora_fine_preamble_peak_max_abs_signed_bin") != ""
    ]
    fine_max_abs_bin = max(fine_max_abs_bins) if fine_max_abs_bins else ""

    print(f"windows={len(windows)}")
    print(f"detections={len(events)}")
    print(f"selected_packets={len(rows)}")
    print(f"frame_valid={valid_frames}/{len(rows)}")
    print(f"grlora_framesync_valid={valid_fsync}/{len(rows)}")
    print(f"grlora_preamble_max_abs_signed_bin={max_abs_bin}")
    print(f"grlora_fine_preamble_max_abs_signed_bin={fine_max_abs_bin}")
    print(f"sf={sf}")
    print(f"preamble_len={preamble_len:g}")
    print(f"center_freq={args.center_freq:g}")
    print(f"chirp_samples={detector_config.chirp_samples}")
    print(f"win_chirps={detector_config.win_chirps}")
    print(f"hop_samples={detector_config.resolved_hop_samples}")
    print(f"min_periodic_peaks={detector_config.min_periodic_peaks}")
    print(f"align_chirps={align_chirps}")
    print(f"frame_search_samples={frame_search_samples}")
    print(f"frame_symbol_search_span={args.frame_symbol_search_span}")
    print(f"sync_word=0x{int(args.sync_word):02x}")
    print(f"wrote={args.output}")
    if args.stft_dir is not None:
        print(f"wrote_stft_dir={args.stft_dir}")
    if args.events_csv is not None:
        print(f"wrote_events={args.events_csv}")
    if args.windows_csv is not None:
        print(f"wrote_windows={args.windows_csv}")
    if args.framesync_peaks_csv is not None:
        print(f"wrote_framesync_peaks={args.framesync_peaks_csv}")
    if args.framesync_spectrum_dir is not None:
        print(f"wrote_framesync_spectrum_dir={args.framesync_spectrum_dir}")


if __name__ == "__main__":
    main()
