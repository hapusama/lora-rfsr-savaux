#!/usr/bin/env python3
"""在冻结 USRP payload 上比较 Savaux 与 full colored-ML。

实验只在已经冻结的 payload symbol 切片上原位加入复 AWGN，不重新同步，也不
重新生成 GT。Savaux 与 colored-ML 使用同一个 noisy symbol、同一个完整 LoRa
候选集合和同一个精确过采样模板。

任意 4096x4096 协方差无法由当前少量独立窗口无结构地可靠估计，因此这里把
``R_v`` 明确限制为 WSS/circulant 模型。其谱由当前 packet 之前已经解出的 clean
header residual 估计，再与已知的 added-AWGN ``sigma^2 I`` 相加。输出中始终把
它标记为 ``header_residual_circulant``，不把它冒充已知的真实自然噪声协方差。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from weak_decoder.baselines.common import load_packets, write_csv
from weak_decoder.chirp import build_upchirp
from weak_decoder.os_lora.experiment_support.noise_windows import off_packet_starts


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
GR_LORA_ROOT = WEAK_ROOT.parent
DEFAULT_EXPERIMENT = WEAK_ROOT / "data" / "experiments" / "real_iq_payload_noise_cfr_nus_20260720"
DEFAULT_DATASETS = ("0_0_0_10_14_8", "0_0_0_10_14_16", "0_0_0_10_14_32")
COLLECTOR_ROOT = WEAK_ROOT / "USRP_collector" / "data" / "branch4_fixed" / "low_snr"
COLLECTOR_LOW1_IQ = COLLECTOR_ROOT / "sf10_bw125_fs500_pre32_sw34_low1.bin"
COLLECTOR_LOW4_IQ = COLLECTOR_ROOT / "sf10_bw125_fs500_pre32_sw34_low4.bin"
COLLECTOR_LOW1_SYNC = WEAK_ROOT / "data" / "experiments" / "real_low_snr_20260717" / "low1_win4" / "sync.csv"
COLLECTOR_LOW4_SYMBOLS = (
    WEAK_ROOT
    / "data"
    / "experiments"
    / "real_low_snr_20260718_low4_low6"
    / "low4_win4"
    / "fft_symbols.csv"
)
COLLECTOR_GT = (
    WEAK_ROOT
    / "data"
    / "groundtruth"
    / "branch4_fixed"
    / "high_snr"
    / "sf10_bw125_fs500_pre32_sw34_r001_fft_bin_groundtruth.csv"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-set",
        choices=("usrp_iq", "collector_low4"),
        default="usrp_iq",
        help="旧的冻结 USRP_IQ，或连续 low4（用独立 low1 包外噪声估协方差）。",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snrs", nargs="+", type=float, default=[-18, -20, -22, -24, -26, -28])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--source-experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "experiments" / "full_colored_ml_awgn_20260721",
    )
    parser.add_argument("--psd-smoothing-bins", type=int, default=17)
    parser.add_argument("--psd-floor-fraction", type=float, default=0.05)
    parser.add_argument("--max-symbols-per-dataset", type=int, default=0)
    parser.add_argument("--collector-training-windows", type=int, default=256)
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="只审计原始自然 IQ，不生成任何 added-AWGN trial。",
    )
    parser.add_argument(
        "--save-candidate-scores",
        action="store_true",
        help="保存每个 symbol 的全部候选 Lambda 谱和协方差模型信息。",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _signal_reference_powers(path: Path) -> dict[str, float]:
    output: dict[str, float] = {}
    for row in _read_csv(path):
        output[str(row["dataset"])] = float(row["signal_reference_power"])
    return output


def _payload_cases(path: Path, datasets: set[str]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(path):
        dataset = str(row["dataset"])
        if dataset not in datasets:
            continue
        output[dataset].append(
            {
                "dataset": dataset,
                "packet_index": int(row["packet_index"]),
                "frame_index": int(row["frame_index"]),
                "event_index": int(row["event_index"]),
                "payload_symbol_index": int(row["payload_symbol_index"]),
                "frame_symbol_index": int(row["frame_symbol_index"]),
                "start_sample": int(row["corrected_start_sample"]),
                "source_start_sample": int(row["source_start_sample"]),
                "gt_bin": int(row["gt_bin"]),
                "clean_margin_db": float(row["clean_recomputed_margin_db"]),
            }
        )
    for rows in output.values():
        rows.sort(key=lambda item: (item["packet_index"], item["payload_symbol_index"]))
    return dict(output)


def _external_groundtruth(path: Path) -> dict[int, int]:
    output: dict[int, int] = {}
    for row in _read_csv(path):
        output[int(row["frame_symbol_index"])] = int(row["groundtruth_fft_bin"])
    if not output:
        raise RuntimeError(f"no external ground truth found in {path}")
    return output


def _cases_from_packets(
    dataset: str,
    packets: Sequence[dict[str, Any]],
    groundtruth: dict[int, int],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for packet in packets:
        # 与正式 Savaux 评估一致：CSV 的 branch-0 起点向 chip 中心移动 R/2 样点，
        # 再从该位置展开完整 OSR symbol。旧 USRP_IQ 的 frozen cases 已经做过此修正。
        origin_shift = int(packet["os_factor"]) // 2
        for symbol in packet["payload_symbols"]:
            frame_symbol_index = int(symbol["frame_symbol_index"])
            if frame_symbol_index not in groundtruth:
                raise KeyError(f"ground truth has no frame_symbol_index={frame_symbol_index}")
            cases.append(
                {
                    "dataset": dataset,
                    "packet_index": int(packet["packet_index"]),
                    "frame_index": int(packet["frame_index"]),
                    "event_index": int(packet["event_index"]),
                    "payload_symbol_index": int(symbol["payload_symbol_index"]),
                    "frame_symbol_index": frame_symbol_index,
                    "start_sample": int(symbol["start_sample"]) + origin_shift,
                    "source_start_sample": int(symbol["start_sample"]),
                    "gt_bin": int(groundtruth[frame_symbol_index]),
                    "clean_margin_db": float("nan"),
                }
            )
    return cases


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((int(start), int(stop)))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], int(stop)))
    return merged


def _sync_signal_intervals(
    sync_path: Path,
    sample_count: int,
    symbol_samples: int,
    before_chirps: float = 8.0,
    after_chirps: float = 110.0,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for row in _read_csv(sync_path):
        text = str(row.get("detected_start_sample", "")).strip()
        if not text:
            continue
        detected = int(float(text))
        intervals.append(
            (
                max(0, detected - int(round(float(before_chirps) * symbol_samples))),
                min(sample_count, detected + int(round(float(after_chirps) * symbol_samples))),
            )
        )
    if not intervals:
        raise RuntimeError(f"no detected events found in {sync_path}")
    return _merge_intervals(intervals)


def _collector_noise_psd(
    samples: np.ndarray,
    sync_path: Path,
    sf: int,
    os_factor: int,
    max_windows: int,
    smoothing_bins: int,
    floor_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """从连续 low1 的检测保护区之外估计真实包外噪声 PSD。"""

    length = (1 << int(sf)) * int(os_factor)
    intervals = _sync_signal_intervals(sync_path, int(samples.size), length)
    starts = off_packet_starts(
        sample_count=int(samples.size),
        window_len=length,
        intervals=intervals,
        max_windows=int(max_windows),
        seed=41,
    )
    if not starts:
        raise RuntimeError("no collector low1 off-packet windows available")
    matrix = np.asarray(
        [samples[int(start) : int(start) + length] for start in starts], dtype=np.complex128
    )
    matrix -= np.mean(matrix, axis=1, keepdims=True)
    spectra = np.fft.fft(matrix, axis=1)
    raw_psd = np.mean(np.abs(spectra) ** 2, axis=0) / float(length)
    smoothed = _smooth_circular(raw_psd, int(smoothing_bins))
    floor = max(float(np.median(smoothed)) * max(float(floor_fraction), 0.0), 1e-30)
    psd = np.maximum(smoothed, floor).astype(np.float64)
    autocorrelation = np.fft.ifft(psd)
    offset_powers = [
        float(np.mean(np.abs(matrix[:, offset:: int(os_factor)]) ** 2))
        for offset in range(int(os_factor))
    ]
    positive = np.maximum(psd, max(float(np.mean(psd)), 1e-30) * 1e-15)
    p05, p95 = np.percentile(positive, (5.0, 95.0))
    stats: dict[str, Any] = {
        "training_noise_window_count": int(matrix.shape[0]),
        "training_noise_power": float(np.mean(np.abs(matrix) ** 2)),
        "regularized_psd_mean": float(np.mean(psd)),
        "regularized_psd_color_cv": float(np.std(psd) / max(float(np.mean(psd)), 1e-30)),
        "regularized_psd_flatness": float(
            np.exp(np.mean(np.log(positive))) / max(float(np.mean(positive)), 1e-30)
        ),
        "regularized_psd_p95_to_p05_db": float(
            10.0 * math.log10((float(p95) + 1e-30) / (float(p05) + 1e-30))
        ),
        "training_noise_lag1_abs_correlation": float(
            abs(autocorrelation[1]) / max(abs(autocorrelation[0]), 1e-30)
        ),
        "training_noise_osr_offset_power_cv": float(
            np.std(offset_powers) / max(float(np.mean(offset_powers)), 1e-30)
        ),
    }
    for offset, value in enumerate(offset_powers):
        stats[f"training_noise_q{offset}_power"] = float(value)
    return psd, stats


def _packet_header_zero_rates(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
) -> dict[int, float]:
    output: dict[int, float] = {}
    for packet in packets:
        length = (1 << int(packet["sf"])) * int(packet["os_factor"])
        rates = []
        for symbol in packet["header_symbols"]:
            start = int(symbol["start_sample"])
            rates.append(float(np.mean(np.asarray(samples[start : start + length]) == 0)))
        if rates:
            output[int(packet["packet_index"])] = float(np.mean(rates))
    return output


def _payload_reference_power(
    samples: np.ndarray,
    cases: Sequence[dict[str, Any]],
    symbol_samples: int,
) -> float:
    total = 0.0
    count = 0
    for case in cases:
        start = int(case["start_sample"])
        chunk = np.asarray(samples[start : start + int(symbol_samples)], dtype=np.complex64)
        total += float(np.sum(np.abs(chunk).astype(np.float64) ** 2))
        count += int(chunk.size)
    if count <= 0:
        raise RuntimeError("no collector payload samples available for signal reference")
    return float(total / count)


def _smooth_circular(values: np.ndarray, width: int) -> np.ndarray:
    size = max(1, int(width))
    if size <= 1:
        return np.asarray(values, dtype=np.float64)
    if size % 2 == 0:
        size += 1
    half = size // 2
    output = np.zeros_like(np.asarray(values, dtype=np.float64))
    for shift in range(-half, half + 1):
        output += np.roll(values, shift)
    return output / float(size)


def _cfo_vector(sf: int, os_factor: int, cfo_int: int, cfo_frac: float) -> np.ndarray:
    """返回使 raw template 可写成 ``g * s_k`` 的公共单位模向量。"""

    n_bins = 1 << int(sf)
    length = n_bins * int(os_factor)
    n = np.arange(length, dtype=np.float64)
    base = build_upchirp(sf, 0, os_factor).astype(np.complex128)
    reference = build_upchirp(sf, int(cfo_int), os_factor).astype(np.complex128)
    fractional = np.exp(2j * np.pi * float(cfo_frac) * n / float(length))
    return np.asarray(reference * fractional / base, dtype=np.complex64)


def _shift_correlations(
    values: np.ndarray,
    reference_fft: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    """用一次循环相关得到全部 LoRa cyclic-shift 候选的复匹配输出。"""

    matrix = np.asarray(values, dtype=np.complex128)
    spectra = np.fft.fft(matrix, axis=1)
    correlation = np.fft.ifft(np.conj(reference_fft)[None, :] * spectra, axis=1)
    return np.asarray(correlation[:, candidate_indices], dtype=np.complex128)


def _decisions_and_margins(power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(power, dtype=np.float64)
    selected = np.argmax(values, axis=1).astype(np.int64)
    if values.shape[1] <= 1:
        return selected, np.full(values.shape[0], float("inf"), dtype=np.float64)
    top_two = np.partition(values, -2, axis=1)[:, -2:]
    top_two.sort(axis=1)
    margins = 10.0 * np.log10((top_two[:, 1] + 1e-30) / (top_two[:, 0] + 1e-30))
    return selected, margins


def _groundtruth_energy_metrics(
    scores: np.ndarray,
    groundtruth_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 GT 分数、最大错误候选分数和有符号 GT-vs-false margin。"""

    values = np.asarray(scores, dtype=np.float64)
    groundtruth = np.asarray(groundtruth_bins, dtype=np.int64)
    if values.ndim != 2 or groundtruth.shape != (values.shape[0],):
        raise ValueError("candidate score / groundtruth shape mismatch")
    if values.shape[1] <= 1:
        gt_score = values[np.arange(values.shape[0]), groundtruth]
        return gt_score, np.zeros_like(gt_score), np.full_like(gt_score, float("inf"))
    rows = np.arange(values.shape[0], dtype=np.int64)
    gt_score = values[rows, groundtruth]
    masked = values.copy()
    masked[rows, groundtruth] = -np.inf
    false_score = np.max(masked, axis=1)
    margin_db = 10.0 * np.log10((gt_score + 1e-30) / (false_score + 1e-30))
    return gt_score, false_score, margin_db


def _two_sided_sign_p(fixes: int, breaks: int) -> float:
    discordant = int(fixes) + int(breaks)
    if discordant <= 0:
        return 1.0
    smaller = min(int(fixes), int(breaks))
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1)) / float(2**discordant)
    return float(min(1.0, 2.0 * tail))


def _margin(power: np.ndarray) -> float:
    values = np.asarray(power, dtype=np.float64)
    if values.size <= 1:
        return float("inf")
    top = np.partition(values, -2)[-2:]
    return float(10.0 * math.log10((float(np.max(top)) + 1e-30) / (float(np.min(top)) + 1e-30)))


def _header_residual_psd(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    base_templates: np.ndarray,
    smoothing_bins: int,
    floor_fraction: float,
) -> tuple[np.ndarray, dict[str, Any], dict[int, float]]:
    """用 clean header 的模板拟合残差估 WSS/circulant 协方差谱。"""

    if not packets:
        raise RuntimeError("no packets available for header covariance estimation")
    sf = int(packets[0]["sf"])
    os_factor = int(packets[0]["os_factor"])
    length = (1 << sf) * os_factor
    residuals: list[np.ndarray] = []
    packet_zero_rates: dict[int, list[float]] = defaultdict(list)
    for packet in packets:
        g = _cfo_vector(sf, os_factor, int(packet["cfo_int"]), float(packet["cfo_frac"]))
        for symbol in packet["header_symbols"]:
            start = int(symbol["start_sample"])
            stop = start + length
            if start < 0 or stop > int(samples.size):
                continue
            observed = np.asarray(samples[start:stop], dtype=np.complex64)
            raw_bin = int(symbol["raw_fft_bin"])
            template = np.asarray(g * base_templates[raw_bin], dtype=np.complex64)
            alpha = complex(np.vdot(template, observed) / max(float(np.vdot(template, template).real), 1e-30))
            residuals.append(np.asarray(observed - alpha * template, dtype=np.complex64))
            packet_zero_rates[int(packet["packet_index"])].append(float(np.mean(observed == 0)))
    if len(residuals) < 2:
        raise RuntimeError("at least two clean header residuals are required")
    matrix = np.asarray(residuals, dtype=np.complex128)
    # 与普通经验协方差相同，先从每个样点位置减去跨 header 的样本均值。
    matrix -= np.mean(matrix, axis=0, keepdims=True)
    raw_power = float(np.mean(np.abs(matrix) ** 2))
    spectra = np.fft.fft(matrix, axis=1)
    raw_psd = np.mean(np.abs(spectra) ** 2, axis=0) / float(length)
    smoothed = _smooth_circular(raw_psd, int(smoothing_bins))
    floor = max(float(np.median(smoothed)) * max(float(floor_fraction), 0.0), 1e-30)
    psd = np.maximum(smoothed, floor).astype(np.float64)
    autocorrelation = np.fft.ifft(psd)
    offset_powers = [
        float(np.mean(np.abs(matrix[:, offset::os_factor]) ** 2))
        for offset in range(os_factor)
    ]
    positive = np.maximum(psd, max(float(np.mean(psd)), 1e-30) * 1e-15)
    p05, p95 = np.percentile(positive, (5.0, 95.0))
    stats: dict[str, Any] = {
        "header_residual_count": int(matrix.shape[0]),
        "header_residual_power": raw_power,
        "regularized_psd_mean": float(np.mean(psd)),
        "regularized_psd_color_cv": float(np.std(psd) / max(float(np.mean(psd)), 1e-30)),
        "regularized_psd_flatness": float(
            np.exp(np.mean(np.log(positive))) / max(float(np.mean(positive)), 1e-30)
        ),
        "regularized_psd_p95_to_p05_db": float(
            10.0 * math.log10((float(p95) + 1e-30) / (float(p05) + 1e-30))
        ),
        "header_residual_lag1_abs_correlation": float(
            abs(autocorrelation[1]) / max(abs(autocorrelation[0]), 1e-30)
        ),
        "header_residual_osr_offset_power_cv": float(
            np.std(offset_powers) / max(float(np.mean(offset_powers)), 1e-30)
        ),
    }
    for offset, value in enumerate(offset_powers):
        stats[f"header_residual_q{offset}_power"] = float(value)
    mean_packet_zero = {
        int(packet): float(np.mean(values)) for packet, values in packet_zero_rates.items() if values
    }
    return psd, stats, mean_packet_zero


def _count_zeros(samples: np.ndarray, chunk_samples: int = 4_000_000) -> tuple[int, int]:
    zero_count = 0
    total = int(samples.size)
    for start in range(0, total, int(chunk_samples)):
        chunk = np.asarray(samples[start : min(total, start + int(chunk_samples))])
        zero_count += int(np.count_nonzero(chunk == 0))
    return zero_count, total


def _quality_row(
    dataset: str,
    samples: np.ndarray,
    cases: Sequence[dict[str, Any]],
    packets: Sequence[dict[str, Any]],
    header_zero_by_packet: dict[int, float],
    residual_stats: dict[str, Any],
    covariance_source: str,
) -> dict[str, Any]:
    sf = int(packets[0]["sf"])
    os_factor = int(packets[0]["os_factor"])
    length = (1 << sf) * os_factor
    file_zeros, file_samples = _count_zeros(samples)
    payload_by_packet: dict[int, list[float]] = defaultdict(list)
    q_zero_counts = np.zeros(os_factor, dtype=np.int64)
    q_sample_counts = np.zeros(os_factor, dtype=np.int64)
    for case in cases:
        start = int(case["start_sample"])
        observed = np.asarray(samples[start : start + length])
        payload_by_packet[int(case["packet_index"])].append(float(np.mean(observed == 0)))
        for q in range(os_factor):
            branch = observed[q::os_factor]
            q_zero_counts[q] += int(np.count_nonzero(branch == 0))
            q_sample_counts[q] += int(branch.size)
    paired_header: list[float] = []
    paired_payload: list[float] = []
    for packet, values in payload_by_packet.items():
        if packet in header_zero_by_packet and values:
            paired_header.append(float(header_zero_by_packet[packet]))
            paired_payload.append(float(np.mean(values)))
    if len(paired_header) >= 2 and np.std(paired_header) > 0.0 and np.std(paired_payload) > 0.0:
        transfer_corr = float(np.corrcoef(paired_header, paired_payload)[0, 1])
    else:
        transfer_corr = float("nan")
    row: dict[str, Any] = {
        "dataset": dataset,
        "file_samples": int(file_samples),
        "file_exact_zero_fraction": float(file_zeros / max(1, file_samples)),
        "header_exact_zero_fraction": float(np.mean(paired_header)) if paired_header else float("nan"),
        "payload_exact_zero_fraction": float(np.mean(paired_payload)) if paired_payload else float("nan"),
        "header_to_payload_packet_zero_rate_correlation": transfer_corr,
        "paired_packet_count": int(len(paired_header)),
        "covariance_source": str(covariance_source),
        **residual_stats,
    }
    for q in range(os_factor):
        row[f"payload_q{q}_exact_zero_fraction"] = float(
            q_zero_counts[q] / max(1, int(q_sample_counts[q]))
        )
    return row


def _covariance_row(
    dataset: str,
    snr_db: float | None,
    native_psd: np.ndarray,
    signal_reference_power: float,
    covariance_model: str,
) -> dict[str, Any]:
    added_power = (
        0.0
        if snr_db is None
        else float(signal_reference_power / (10.0 ** (float(snr_db) / 10.0)))
    )
    total_psd = np.asarray(native_psd, dtype=np.float64) + added_power
    autocorrelation = np.fft.ifft(total_psd)
    diagonal = max(abs(autocorrelation[0]), 1e-30)
    offdiag = float(np.sqrt(np.sum(np.abs(autocorrelation[1:]) ** 2)))
    native_power = float(np.mean(native_psd))
    return {
        "dataset": dataset,
        "snr_label": "clean" if snr_db is None else f"{float(snr_db):g}",
        "added_snr_db": "" if snr_db is None else float(snr_db),
        "signal_reference_power": float(signal_reference_power),
        "estimated_covariance_power": native_power,
        "added_awgn_power": float(added_power),
        "added_to_estimated_covariance_power_db": (
            float("-inf")
            if added_power <= 0.0
            else float(10.0 * math.log10(added_power / max(native_power, 1e-30)))
        ),
        "total_covariance_color_cv": float(
            np.std(total_psd) / max(float(np.mean(total_psd)), 1e-30)
        ),
        "total_lag1_abs_correlation": float(abs(autocorrelation[1]) / diagonal),
        "total_offdiag_to_diag_frobenius": float(offdiag / diagonal),
        "total_covariance_condition_number": float(
            np.max(total_psd) / max(float(np.min(total_psd)), 1e-30)
        ),
        "covariance_model": str(covariance_model),
    }


def _evaluate_dataset(
    dataset_index: int,
    dataset: str,
    samples: np.ndarray,
    cases: Sequence[dict[str, Any]],
    packets: Sequence[dict[str, Any]],
    signal_reference_power: float,
    native_psd: np.ndarray,
    base_templates: np.ndarray,
    snrs: Sequence[float],
    seeds: Sequence[int],
    covariance_model: str,
    candidate_score_blocks: list[dict[str, np.ndarray]] | None = None,
) -> list[dict[str, Any]]:
    sf = int(packets[0]["sf"])
    os_factor = int(packets[0]["os_factor"])
    n_bins = 1 << sf
    length = n_bins * os_factor
    base_reference = np.asarray(base_templates[0], dtype=np.complex128)
    base_reference_fft = np.fft.fft(base_reference)
    candidate_indices = np.mod(-np.arange(n_bins, dtype=np.int64) * os_factor, length)
    by_packet_meta = {int(packet["packet_index"]): packet for packet in packets}
    cases_by_packet: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_packet[int(case["packet_index"])].append(case)

    output: list[dict[str, Any]] = []
    points: list[float | None] = [None] + [float(value) for value in snrs]
    for packet_index, packet_cases in sorted(cases_by_packet.items()):
        packet = by_packet_meta.get(int(packet_index))
        if packet is None:
            raise KeyError(f"{dataset}: packet {packet_index} has cases but no frozen metadata")
        g = _cfo_vector(sf, os_factor, int(packet["cfo_int"]), float(packet["cfo_frac"]))
        clean = np.asarray(
            [samples[int(case["start_sample"]) : int(case["start_sample"]) + length] for case in packet_cases],
            dtype=np.complex64,
        )
        gt = np.asarray([int(case["gt_bin"]) for case in packet_cases], dtype=np.int64)

        # colored-ML 分母只依赖 packet CFO/template 与 R_v；先计算所有候选的频域能量。
        raw_templates = np.asarray(base_templates * g[None, :], dtype=np.complex64)
        template_fft = np.fft.fft(raw_templates, axis=1)
        template_spectral_power = np.asarray(np.abs(template_fft) ** 2, dtype=np.float32)
        del raw_templates, template_fft

        for point_index, snr_db in enumerate(points):
            added_power = (
                0.0
                if snr_db is None
                else float(signal_reference_power / (10.0 ** (float(snr_db) / 10.0)))
            )
            total_psd = np.asarray(native_psd, dtype=np.float64) + added_power
            inverse_psd = 1.0 / np.maximum(total_psd, 1e-30)
            denominator = (
                np.asarray(template_spectral_power, dtype=np.float64) @ inverse_psd / float(length)
            )
            denominator = np.maximum(denominator, 1e-30)
            template_energy = (
                np.sum(np.asarray(template_spectral_power, dtype=np.float64), axis=1) / float(length)
            )
            colored_noise_energy = (
                np.asarray(template_spectral_power, dtype=np.float64) @ total_psd / float(length)
            )
            savaux_information = template_energy**2 / np.maximum(colored_noise_energy, 1e-30)
            full_information = denominator
            predicted_snr_gain = full_information / np.maximum(savaux_information, 1e-30)
            predicted_snr_gain_db = 10.0 * np.log10(np.maximum(predicted_snr_gain, 1e-30))
            point_seeds: Iterable[int] = (0,) if snr_db is None else seeds
            for seed in point_seeds:
                if snr_db is None:
                    noisy = clean.astype(np.complex64, copy=True)
                else:
                    seed_sequence = np.random.SeedSequence(
                        [int(seed), int(dataset_index), int(point_index), int(packet_index)]
                    )
                    rng = np.random.default_rng(seed_sequence)
                    sigma = math.sqrt(added_power / 2.0)
                    noise = (
                        rng.normal(0.0, sigma, clean.shape)
                        + 1j * rng.normal(0.0, sigma, clean.shape)
                    ).astype(np.complex64)
                    noisy = np.asarray(clean + noise, dtype=np.complex64)

                # Savaux 等价于完整过采样理想模板的白噪声 matched filter。
                cfo_corrected = np.asarray(noisy * np.conj(g)[None, :], dtype=np.complex128)
                savaux_corr = _shift_correlations(
                    cfo_corrected, base_reference_fft, candidate_indices
                )
                savaux_power = np.abs(savaux_corr) ** 2 / float(length)
                savaux_bin, savaux_margin = _decisions_and_margins(savaux_power)

                # full colored-ML: R^-1 先作用于完整 NR 样点，再与全部模板相关。
                whitened_raw = np.fft.ifft(
                    np.fft.fft(noisy.astype(np.complex128), axis=1) * inverse_psd[None, :],
                    axis=1,
                )
                whitened_corrected = whitened_raw * np.conj(g)[None, :]
                colored_corr = _shift_correlations(
                    whitened_corrected, base_reference_fft, candidate_indices
                )
                colored_power = np.abs(colored_corr) ** 2 / denominator[None, :]
                colored_bin, colored_margin = _decisions_and_margins(colored_power)

                # 两个能量都按同一个 R_v 标定成零假设均值为 1 的 Lambda。
                # 这里只归一化 Savaux 的输出能量，不改变其原始 hard decision。
                savaux_output_noise = colored_noise_energy / np.maximum(template_energy, 1e-30)
                savaux_lambda = savaux_power / np.maximum(savaux_output_noise[None, :], 1e-30)
                colored_lambda = colored_power
                savaux_gt, savaux_false, savaux_gt_margin = _groundtruth_energy_metrics(
                    savaux_lambda, gt
                )
                colored_gt, colored_false, colored_gt_margin = _groundtruth_energy_metrics(
                    colored_lambda, gt
                )

                if candidate_score_blocks is not None:
                    candidate_score_blocks.append(
                        {
                            "dataset": np.full(len(packet_cases), dataset, dtype=f"<U{max(1, len(dataset))}"),
                            "packet_index": np.full(len(packet_cases), int(packet_index), dtype=np.int64),
                            "frame_index": np.asarray(
                                [int(case["frame_index"]) for case in packet_cases], dtype=np.int64
                            ),
                            "payload_symbol_index": np.asarray(
                                [int(case["payload_symbol_index"]) for case in packet_cases], dtype=np.int64
                            ),
                            "frame_symbol_index": np.asarray(
                                [int(case["frame_symbol_index"]) for case in packet_cases], dtype=np.int64
                            ),
                            "start_sample": np.asarray(
                                [int(case["start_sample"]) for case in packet_cases], dtype=np.int64
                            ),
                            "gt_bin": gt.astype(np.int64, copy=True),
                            "snr_db": np.full(
                                len(packet_cases), np.nan if snr_db is None else float(snr_db), dtype=np.float64
                            ),
                            "seed": np.full(len(packet_cases), int(seed), dtype=np.int64),
                            "savaux_lambda": np.asarray(savaux_lambda, dtype=np.float32),
                            "colored_ml_lambda": np.asarray(colored_lambda, dtype=np.float32),
                            "savaux_information": np.broadcast_to(
                                savaux_information.astype(np.float32), (len(packet_cases), n_bins)
                            ).copy(),
                            "full_colored_information": np.broadcast_to(
                                full_information.astype(np.float32), (len(packet_cases), n_bins)
                            ).copy(),
                            "predicted_snr_gain_db": np.broadcast_to(
                                predicted_snr_gain_db.astype(np.float32), (len(packet_cases), n_bins)
                            ).copy(),
                        }
                    )

                for local_index, case in enumerate(packet_cases):
                    savaux_ok = int(savaux_bin[local_index]) == int(gt[local_index])
                    colored_ok = int(colored_bin[local_index]) == int(gt[local_index])
                    output.append(
                        {
                            "dataset": dataset,
                            "packet_index": int(packet_index),
                            "frame_index": int(case["frame_index"]),
                            "payload_symbol_index": int(case["payload_symbol_index"]),
                            "frame_symbol_index": int(case["frame_symbol_index"]),
                            "start_sample": int(case["start_sample"]),
                            "gt_bin": int(gt[local_index]),
                            "snr_label": "clean" if snr_db is None else f"{float(snr_db):g}",
                            "added_snr_db": "" if snr_db is None else float(snr_db),
                            "seed": int(seed),
                            "added_awgn_power": float(added_power),
                            "savaux_bin": int(savaux_bin[local_index]),
                            "savaux_correct": int(savaux_ok),
                            "savaux_margin_db": float(savaux_margin[local_index]),
                            "savaux_gt_lambda": float(savaux_gt[local_index]),
                            "savaux_max_false_lambda": float(savaux_false[local_index]),
                            "savaux_gt_false_margin_db": float(savaux_gt_margin[local_index]),
                            # R=sigma^2 I 时 full ML 与 Savaux 严格同判，直接记录此恒等校验。
                            "white_ml_bin": int(savaux_bin[local_index]),
                            "white_ml_matches_savaux": 1,
                            "colored_ml_bin": int(colored_bin[local_index]),
                            "colored_ml_correct": int(colored_ok),
                            "colored_ml_margin_db": float(colored_margin[local_index]),
                            "colored_ml_gt_lambda": float(colored_gt[local_index]),
                            "colored_ml_max_false_lambda": float(colored_false[local_index]),
                            "colored_ml_gt_false_margin_db": float(colored_gt_margin[local_index]),
                            "colored_minus_savaux_gt_lambda": float(
                                colored_gt[local_index] - savaux_gt[local_index]
                            ),
                            "colored_minus_savaux_gt_false_margin_db": float(
                                colored_gt_margin[local_index] - savaux_gt_margin[local_index]
                            ),
                            "model_gt_full_information": float(full_information[int(gt[local_index])]),
                            "model_gt_savaux_information": float(
                                savaux_information[int(gt[local_index])]
                            ),
                            "model_gt_full_to_savaux_snr_gain_db": float(
                                predicted_snr_gain_db[int(gt[local_index])]
                            ),
                            "colored_ml_fix_vs_savaux": int((not savaux_ok) and colored_ok),
                            "colored_ml_break_vs_savaux": int(savaux_ok and (not colored_ok)),
                            "covariance_model": str(covariance_model),
                        }
                    )
        del template_spectral_power
    return output


def _summary_rows(symbol_rows: Sequence[dict[str, Any]], aggregate_dataset: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in symbol_rows:
        dataset = "ALL" if aggregate_dataset else str(row["dataset"])
        groups[(dataset, str(row["snr_label"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, label), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], 1e9 if item[0][1] == "clean" else -float(item[0][1]))
    ):
        count = len(rows)
        savaux_errors = sum(1 - int(row["savaux_correct"]) for row in rows)
        colored_errors = sum(1 - int(row["colored_ml_correct"]) for row in rows)
        fixes = sum(int(row["colored_ml_fix_vs_savaux"]) for row in rows)
        breaks = sum(int(row["colored_ml_break_vs_savaux"]) for row in rows)
        seed_differences: list[int] = []
        for seed in sorted({int(row["seed"]) for row in rows}):
            subset = [row for row in rows if int(row["seed"]) == seed]
            seed_savaux = sum(1 - int(row["savaux_correct"]) for row in subset)
            seed_colored = sum(1 - int(row["colored_ml_correct"]) for row in subset)
            seed_differences.append(int(seed_savaux - seed_colored))
        seed_wins = sum(value > 0 for value in seed_differences)
        seed_losses = sum(value < 0 for value in seed_differences)
        seed_ties = sum(value == 0 for value in seed_differences)
        packet_differences: list[int] = []
        for packet in sorted({(str(row["dataset"]), int(row["packet_index"])) for row in rows}):
            subset = [
                row
                for row in rows
                if (str(row["dataset"]), int(row["packet_index"])) == packet
            ]
            packet_savaux = sum(1 - int(row["savaux_correct"]) for row in subset)
            packet_colored = sum(1 - int(row["colored_ml_correct"]) for row in subset)
            packet_differences.append(int(packet_savaux - packet_colored))
        packet_wins = sum(value > 0 for value in packet_differences)
        packet_losses = sum(value < 0 for value in packet_differences)
        packet_ties = sum(value == 0 for value in packet_differences)
        savaux_gt_lambda = np.asarray([float(row["savaux_gt_lambda"]) for row in rows])
        colored_gt_lambda = np.asarray([float(row["colored_ml_gt_lambda"]) for row in rows])
        savaux_gt_margin = np.asarray(
            [float(row["savaux_gt_false_margin_db"]) for row in rows]
        )
        colored_gt_margin = np.asarray(
            [float(row["colored_ml_gt_false_margin_db"]) for row in rows]
        )
        margin_delta = colored_gt_margin - savaux_gt_margin
        margin_tolerance = 1e-9
        output.append(
            {
                "dataset": dataset,
                "snr_label": label,
                "added_snr_db": "" if label == "clean" else float(label),
                "decision_count": int(count),
                "trial_seed_count": int(len({int(row["seed"]) for row in rows})),
                "savaux_errors": int(savaux_errors),
                "savaux_ser": float(savaux_errors / max(1, count)),
                "white_ml_errors": int(savaux_errors),
                "white_ml_ser": float(savaux_errors / max(1, count)),
                "white_ml_decision_mismatches_vs_savaux": 0,
                "colored_ml_errors": int(colored_errors),
                "colored_ml_ser": float(colored_errors / max(1, count)),
                "colored_ml_fixes_vs_savaux": int(fixes),
                "colored_ml_breaks_vs_savaux": int(breaks),
                "colored_ml_paired_sign_p": _two_sided_sign_p(fixes, breaks),
                "seed_wins_ties_losses": f"{seed_wins}|{seed_ties}|{seed_losses}",
                "seed_level_sign_p": _two_sided_sign_p(seed_wins, seed_losses),
                "packet_wins_ties_losses": f"{packet_wins}|{packet_ties}|{packet_losses}",
                "packet_level_sign_p": _two_sided_sign_p(packet_wins, packet_losses),
                "savaux_gt_lambda_mean": float(np.mean(savaux_gt_lambda)),
                "savaux_gt_lambda_median": float(np.median(savaux_gt_lambda)),
                "colored_ml_gt_lambda_mean": float(np.mean(colored_gt_lambda)),
                "colored_ml_gt_lambda_median": float(np.median(colored_gt_lambda)),
                "savaux_gt_false_margin_db_mean": float(np.mean(savaux_gt_margin)),
                "savaux_gt_false_margin_db_median": float(np.median(savaux_gt_margin)),
                "colored_ml_gt_false_margin_db_mean": float(np.mean(colored_gt_margin)),
                "colored_ml_gt_false_margin_db_median": float(np.median(colored_gt_margin)),
                "colored_minus_savaux_margin_db_mean": float(np.mean(margin_delta)),
                "colored_minus_savaux_margin_db_median": float(np.median(margin_delta)),
                "colored_margin_improved_tied_worse": (
                    f"{int(np.sum(margin_delta > margin_tolerance))}|"
                    f"{int(np.sum(np.abs(margin_delta) <= margin_tolerance))}|"
                    f"{int(np.sum(margin_delta < -margin_tolerance))}"
                ),
                "model_gt_full_to_savaux_snr_gain_db_mean": float(
                    np.mean([float(row["model_gt_full_to_savaux_snr_gain_db"]) for row in rows])
                ),
                "model_gt_full_to_savaux_snr_gain_db_median": float(
                    np.median([float(row["model_gt_full_to_savaux_snr_gain_db"]) for row in rows])
                ),
            }
        )
    return output


def _write_candidate_score_archive(
    output_dir: Path,
    blocks: Sequence[dict[str, np.ndarray]],
) -> None:
    """保存完整候选谱，并生成便于人工检查的逐 bin 汇总。"""

    if not blocks:
        return
    scalar_keys = (
        "dataset",
        "packet_index",
        "frame_index",
        "payload_symbol_index",
        "frame_symbol_index",
        "start_sample",
        "gt_bin",
        "snr_db",
        "seed",
    )
    matrix_keys = (
        "savaux_lambda",
        "colored_ml_lambda",
        "savaux_information",
        "full_colored_information",
        "predicted_snr_gain_db",
    )
    archive: dict[str, np.ndarray] = {
        key: np.concatenate([block[key] for block in blocks], axis=0) for key in scalar_keys
    }
    archive.update(
        {key: np.concatenate([block[key] for block in blocks], axis=0) for key in matrix_keys}
    )
    archive["candidate_bin"] = np.arange(archive["savaux_lambda"].shape[1], dtype=np.int64)
    np.savez_compressed(output_dir / "candidate_scores.npz", **archive)

    savaux = np.asarray(archive["savaux_lambda"], dtype=np.float64)
    colored = np.asarray(archive["colored_ml_lambda"], dtype=np.float64)
    gain_db = np.asarray(archive["predicted_snr_gain_db"], dtype=np.float64)
    groundtruth = np.asarray(archive["gt_bin"], dtype=np.int64)
    candidate_rows: list[dict[str, Any]] = []
    for raw_bin in range(savaux.shape[1]):
        gt_mask = groundtruth == raw_bin
        candidate_rows.append(
            {
                "raw_fft_bin": int(raw_bin),
                "trial_count": int(savaux.shape[0]),
                "groundtruth_count": int(np.sum(gt_mask)),
                "savaux_lambda_mean": float(np.mean(savaux[:, raw_bin])),
                "savaux_lambda_median": float(np.median(savaux[:, raw_bin])),
                "colored_ml_lambda_mean": float(np.mean(colored[:, raw_bin])),
                "colored_ml_lambda_median": float(np.median(colored[:, raw_bin])),
                "colored_minus_savaux_lambda_mean": float(
                    np.mean(colored[:, raw_bin] - savaux[:, raw_bin])
                ),
                "model_full_to_savaux_snr_gain_db_mean": float(np.mean(gain_db[:, raw_bin])),
                "gt_savaux_lambda_mean": (
                    float(np.mean(savaux[gt_mask, raw_bin])) if np.any(gt_mask) else ""
                ),
                "gt_colored_ml_lambda_mean": (
                    float(np.mean(colored[gt_mask, raw_bin])) if np.any(gt_mask) else ""
                ),
            }
        )
    write_csv(output_dir / "candidate_summary.csv", candidate_rows)


def _write_energy_results(
    path: Path,
    summary_rows: Sequence[dict[str, Any]],
    symbol_rows: Sequence[dict[str, Any]],
) -> None:
    """单独报告候选能量与 margin，不用 SER 代替能量结论。"""

    lines = [
        "# Full-observation candidate-energy audit",
        "",
        "Savaux 和 full colored-ML 使用同一个完整 OSR symbol 与全部候选模板。",
        "Savaux 能量按 `s^H R_v s` 标定，colored-ML 按 `s^H R_v^-1 s` 标定，",
        "因此两者在各自的噪声输出上都是 H0 均值为 1 的无量纲 Lambda。",
        "Savaux hard decision 仍使用原始匹配能量，不使用该标定量重新判决。",
        "",
        "| dataset | condition | symbols | Savaux GT Lambda | colored GT Lambda | Savaux margin | colored margin | margin delta | model SNR gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summary_rows:
        dataset = str(summary["dataset"])
        label = str(summary["snr_label"])
        rows = [
            row
            for row in symbol_rows
            if str(row["dataset"]) == dataset and str(row["snr_label"]) == label
        ]
        margin_delta = np.asarray(
            [float(row["colored_minus_savaux_gt_false_margin_db"]) for row in rows]
        )
        gt_delta = np.asarray(
            [float(row["colored_minus_savaux_gt_lambda"]) for row in rows]
        )
        improved = int(np.sum(margin_delta > 1e-9))
        worse = int(np.sum(margin_delta < -1e-9))
        packet_means = []
        for packet in sorted({int(row["packet_index"]) for row in rows}):
            values = [
                float(row["colored_minus_savaux_gt_false_margin_db"])
                for row in rows
                if int(row["packet_index"]) == packet
            ]
            packet_means.append(float(np.mean(values)))
        gain_values = np.asarray(
            [float(row["model_gt_full_to_savaux_snr_gain_db"]) for row in rows]
        )
        lines.append(
            f"| {dataset} | {label} | {len(rows)} "
            f"| {float(summary['savaux_gt_lambda_mean']):.6g} "
            f"| {float(summary['colored_ml_gt_lambda_mean']):.6g} "
            f"| {float(summary['savaux_gt_false_margin_db_mean']):.6f} dB "
            f"| {float(summary['colored_ml_gt_false_margin_db_mean']):.6f} dB "
            f"| {float(summary['colored_minus_savaux_margin_db_mean']):+.6f} dB "
            f"| {float(summary['model_gt_full_to_savaux_snr_gain_db_mean']):+.6f} dB |"
        )
        lines.extend(
            [
                "",
                f"- GT Lambda increased/decreased: {int(np.sum(gt_delta > 0.0))}/{int(np.sum(gt_delta < 0.0))}.",
                f"- GT-vs-false margin improved/worsened: {improved}/{worse}; symbol-level sign p={_two_sided_sign_p(improved, worse):.6g}.",
                f"- Packet mean margin improved/tied/worsened: {sum(v > 0.0 for v in packet_means)}/{sum(v == 0.0 for v in packet_means)}/{sum(v < 0.0 for v in packet_means)}.",
                f"- Model GT-bin SNR gain range: {float(np.min(gain_values)):.6f} to {float(np.max(gain_values)):.6f} dB.",
                f"- Hard decisions: Savaux errors={summary['savaux_errors']}, colored-ML errors={summary['colored_ml_errors']}, fixes/breaks={summary['colored_ml_fixes_vs_savaux']}/{summary['colored_ml_breaks_vs_savaux']}.",
            ]
        )
    lines.extend(
        [
            "",
            "The full per-symbol, per-candidate arrays are stored in `candidate_scores.npz`;",
            "`candidate_summary.csv` contains a 1024-bin aggregate view.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results(
    path: Path,
    global_rows: Sequence[dict[str, Any]],
    quality_rows: Sequence[dict[str, Any]],
    covariance_rows: Sequence[dict[str, Any]],
    source_set: str,
) -> None:
    covariance_description = (
        "clean header residual 的 WSS/circulant PSD"
        if str(source_set) == "usrp_iq"
        else "独立 continuous low1 包外噪声的 WSS/circulant PSD"
    )
    lines = [
        "# Full colored-ML / Savaux added-AWGN audit",
        "",
        "本实验使用冻结的真实 USRP payload 切片和冻结 GT；AWGN 只在内存中的 symbol 副本上加入。",
        "`white_ml` 使用已知的 added-AWGN 协方差 `sigma^2 I`，因此在数学和实现上都与 Savaux 同判。",
        f"`colored_ml` 使用{covariance_description}，再加 `sigma^2 I`；它是结构化估计下的 full-sample ML，不是已知真协方差 oracle。",
        "",
        "## Global SER",
        "",
        "| added SNR | decisions | Savaux errors | colored-ML errors | fixes / breaks | decision p | seed W/T/L | seed p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(
            f"| {row['snr_label']} | {row['decision_count']} | {row['savaux_errors']} "
            f"| {row['colored_ml_errors']} | {row['colored_ml_fixes_vs_savaux']} / "
            f"{row['colored_ml_breaks_vs_savaux']} | {float(row['colored_ml_paired_sign_p']):.3g} "
            f"| {str(row['seed_wins_ties_losses']).replace('|', '/')} "
            f"| {float(row['seed_level_sign_p']):.3g} |"
        )
    lines.extend(
        [
            "",
            "## IQ storage audit",
            "",
            "| dataset | whole-file exact zeros | header zeros | payload zeros | header→payload corr |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in quality_rows:
        lines.append(
            f"| {row['dataset']} | {float(row['file_exact_zero_fraction']):.4f} "
            f"| {float(row['header_exact_zero_fraction']):.4f} "
            f"| {float(row['payload_exact_zero_fraction']):.4f} "
            f"| {float(row['header_to_payload_packet_zero_rate_correlation']):.3f} |"
        )
    if str(source_set) == "usrp_iq":
        lines.extend(
            [
                "",
                "这些旧 `USRP_IQ` 文件包含大段精确零填充，packet 内也存在精确零样点；因此不能从文件的所谓 packet 外窗口直接估计自然噪声协方差。header residual 同时包含接收噪声、模板/信道失配与零样点缺失，只能作为明确标注的上界诊断。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "`collector_low4` 是连续 IQ；其 colored covariance 只由独立 low1 capture 的检测保护区外噪声窗口估计，low4 payload GT 不参与协方差。",
            ]
        )
    lines.extend(
        [
            "",
            "## Why the covariance becomes diagonal",
            "",
            "对每个 added-SNR 点，总协方差模型是 `R_total = R_estimated + sigma^2 I`。下面的 color CV 越接近 0，矩阵越接近缩放单位阵。",
            "",
            "| dataset | added SNR | AWGN / residual (dB) | color CV | lag-1 corr | offdiag/diag Fro |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in covariance_rows:
        if str(row["snr_label"]) == "clean":
            continue
        lines.append(
            f"| {row['dataset']} | {row['snr_label']} "
            f"| {float(row['added_to_estimated_covariance_power_db']):.2f} "
            f"| {float(row['total_covariance_color_cv']):.4g} "
            f"| {float(row['total_lag1_abs_correlation']):.4g} "
            f"| {float(row['total_offdiag_to_diag_frobenius']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `colored_ml > Savaux` 只说明完整样点中有可利用的二阶统计结构；它不自动证明 NUS。",
            "- 若 weak-error 区域由 added AWGN 主导，`sigma^2 I` 会淹没原始有色项，full colored-ML 应收敛到 Savaux。",
        ]
    )
    if str(source_set) == "usrp_iq":
        lines.append(
            "- packet 内精确零样点属于采样缺失/数据门控或硬件链路问题，不是普通传播噪声。若它在 header 到 payload 间可预测，它可能形成 NUS 可作用的 `(p,q)` 可靠性结构，但必须先用连续、未门控的新采集复核。"
        )
    else:
        lines.append(
            "- 独立 low1 包外噪声能检验一般时间有色性；要支持 NUS，还必须进一步证明这种异常能定位到可预测的 `(p,q)` 采样可靠性，而不只是一般 WSS color。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snrs = [] if bool(args.clean_only) else [float(value) for value in args.snrs]
    symbol_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    candidate_score_blocks: list[dict[str, np.ndarray]] | None = (
        [] if bool(args.save_candidate_scores) else None
    )
    config = {
        "source_set": str(args.source_set),
        "added_snr_db": snrs,
        "seeds": [int(value) for value in args.seeds],
        "output_dir": str(output_dir),
        "psd_smoothing_bins": int(args.psd_smoothing_bins),
        "psd_floor_fraction": float(args.psd_floor_fraction),
        "max_symbols_per_dataset": int(args.max_symbols_per_dataset),
        "noise": "in-memory independent complex AWGN on copied payload symbols",
        "clean_only": bool(args.clean_only),
        "candidate_scores_saved": bool(args.save_candidate_scores),
        "lambda_normalization": (
            "Savaux: |s^H z|^2/(s^H R_v s); colored-ML: "
            "|s^H R_v^-1 z|^2/(s^H R_v^-1 s); both H0-normalized"
        ),
    }

    if str(args.source_set) == "usrp_iq":
        datasets = tuple(str(value) for value in args.datasets)
        source_experiment = args.source_experiment.resolve()
        source_demod = source_experiment / "source_demod"
        signal_powers = _signal_reference_powers(source_experiment / "source_summary.csv")
        all_cases = _payload_cases(source_experiment / "source_symbol_cases.csv", set(datasets))
        covariance_model = "header_residual_circulant_plus_known_awgn_identity"
        config.update(
            {
                "datasets": list(datasets),
                "source_experiment": str(source_experiment),
                "synchronization": "frozen corrected_start_sample and packet CFO; no noisy resync",
                "gt": "frozen audited payload gt_bin; scoring only",
                "full_colored_ml_covariance": covariance_model,
            }
        )
        for dataset_index, dataset in enumerate(datasets):
            if dataset not in all_cases:
                raise KeyError(f"no frozen payload cases found for {dataset}")
            if dataset not in signal_powers:
                raise KeyError(f"no signal reference power found for {dataset}")
            symbol_path = source_demod / f"{dataset}_header_first_symbols.csv"
            iq_path = GR_LORA_ROOT / "data" / "USRP_IQ" / f"{dataset}.bin"
            packets = load_packets(symbol_path)
            cases = list(all_cases[dataset])
            if int(args.max_symbols_per_dataset) > 0:
                cases = cases[: int(args.max_symbols_per_dataset)]
            samples = np.memmap(iq_path, dtype=np.complex64, mode="r")
            sf = int(packets[0]["sf"])
            os_factor = int(packets[0]["os_factor"])
            base_templates = np.stack(
                [build_upchirp(sf, raw_bin, os_factor) for raw_bin in range(1 << sf)]
            ).astype(np.complex64)
            native_psd, residual_stats, header_zero_by_packet = _header_residual_psd(
                samples=samples,
                packets=packets,
                base_templates=base_templates,
                smoothing_bins=int(args.psd_smoothing_bins),
                floor_fraction=float(args.psd_floor_fraction),
            )
            quality_rows.append(
                _quality_row(
                    dataset,
                    samples,
                    cases,
                    packets,
                    header_zero_by_packet,
                    residual_stats,
                    covariance_source="clean_header_residual_circulant",
                )
            )
            for snr_db in [None] + snrs:
                covariance_rows.append(
                    _covariance_row(
                        dataset,
                        snr_db,
                        native_psd,
                        float(signal_powers[dataset]),
                        covariance_model,
                    )
                )
            symbol_rows.extend(
                _evaluate_dataset(
                    dataset_index=dataset_index,
                    dataset=dataset,
                    samples=samples,
                    cases=cases,
                    packets=packets,
                    signal_reference_power=float(signal_powers[dataset]),
                    native_psd=native_psd,
                    base_templates=base_templates,
                    snrs=snrs,
                    seeds=[int(value) for value in args.seeds],
                    covariance_model=covariance_model,
                    candidate_score_blocks=candidate_score_blocks,
                )
            )
            del base_templates, samples
    else:
        dataset = "collector_low4_train_low1"
        covariance_model = "independent_low1_offpacket_circulant_plus_known_awgn_identity"
        packets = load_packets(COLLECTOR_LOW4_SYMBOLS)
        cases = _cases_from_packets(dataset, packets, _external_groundtruth(COLLECTOR_GT))
        if int(args.max_symbols_per_dataset) > 0:
            cases = cases[: int(args.max_symbols_per_dataset)]
        samples = np.memmap(COLLECTOR_LOW4_IQ, dtype=np.complex64, mode="r")
        training_samples = np.memmap(COLLECTOR_LOW1_IQ, dtype=np.complex64, mode="r")
        sf = int(packets[0]["sf"])
        os_factor = int(packets[0]["os_factor"])
        length = (1 << sf) * os_factor
        base_templates = np.stack(
            [build_upchirp(sf, raw_bin, os_factor) for raw_bin in range(1 << sf)]
        ).astype(np.complex64)
        native_psd, noise_stats = _collector_noise_psd(
            samples=training_samples,
            sync_path=COLLECTOR_LOW1_SYNC,
            sf=sf,
            os_factor=os_factor,
            max_windows=int(args.collector_training_windows),
            smoothing_bins=int(args.psd_smoothing_bins),
            floor_fraction=float(args.psd_floor_fraction),
        )
        signal_reference_power = _payload_reference_power(samples, cases, length)
        quality_rows.append(
            _quality_row(
                dataset,
                samples,
                cases,
                packets,
                _packet_header_zero_rates(samples, packets),
                noise_stats,
                covariance_source="independent_low1_natural_offpacket_circulant",
            )
        )
        for snr_db in [None] + snrs:
            covariance_rows.append(
                _covariance_row(
                    dataset,
                    snr_db,
                    native_psd,
                    signal_reference_power,
                    covariance_model,
                )
            )
        symbol_rows.extend(
            _evaluate_dataset(
                dataset_index=0,
                dataset=dataset,
                samples=samples,
                cases=cases,
                packets=packets,
                signal_reference_power=signal_reference_power,
                native_psd=native_psd,
                base_templates=base_templates,
                snrs=snrs,
                seeds=[int(value) for value in args.seeds],
                covariance_model=covariance_model,
                candidate_score_blocks=candidate_score_blocks,
            )
        )
        config.update(
            {
                "datasets": [dataset],
                "test_iq": str(COLLECTOR_LOW4_IQ),
                "test_symbols": str(COLLECTOR_LOW4_SYMBOLS),
                "training_iq": str(COLLECTOR_LOW1_IQ),
                "training_sync": str(COLLECTOR_LOW1_SYNC),
                "groundtruth": str(COLLECTOR_GT),
                "collector_training_windows": int(args.collector_training_windows),
                "signal_reference_power": float(signal_reference_power),
                "synchronization": "frozen low4 symbol start and packet CFO; no noisy resync",
                "gt": "independent high-SNR fixed-frame ground truth; scoring only",
                "full_colored_ml_covariance": covariance_model,
            }
        )
        del base_templates, samples, training_samples

    summary = _summary_rows(symbol_rows, aggregate_dataset=False)
    global_summary = _summary_rows(symbol_rows, aggregate_dataset=True)
    write_csv(output_dir / "symbols.csv", symbol_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "global_summary.csv", global_summary)
    write_csv(output_dir / "covariance_summary.csv", covariance_rows)
    write_csv(output_dir / "data_quality.csv", quality_rows)
    if candidate_score_blocks is not None:
        _write_candidate_score_archive(output_dir, candidate_score_blocks)
        _write_energy_results(output_dir / "ENERGY_RESULTS.md", summary, symbol_rows)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_results(
        output_dir / "RESULTS.md",
        global_summary,
        quality_rows,
        covariance_rows,
        source_set=str(args.source_set),
    )
    print(f"Wrote {len(symbol_rows)} paired decisions to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
