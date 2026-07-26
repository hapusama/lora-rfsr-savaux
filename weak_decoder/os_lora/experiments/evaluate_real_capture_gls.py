#!/usr/bin/env python3
"""在真实 IQ 采集上使用外部 FFT-bin GT 评估 Savaux 与 GLS。

本入口与 ``evaluate_low_complexity_gls.py`` 的主要区别是：这里不会向 IQ
额外叠加人工噪声。pattern 选择与固定 GLS 协方差都从另一份训练 capture 的
包外窗口中学习；high-SNR GT 只在同步完成后用于 payload FFT-bin 评分，不会
参与同步、CFO/STO/SFO 估计、候选裁剪或最终判决。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from weak_decoder.baselines.common import load_packets, write_csv
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    _oversampled_downchirp,
    paper_oversampled_spectrum,
)
from weak_decoder.os_lora.experiment_support.noise_windows import (
    empirical_covariance as _empirical_covariance,
    off_packet_starts as _off_packet_starts,
)
from weak_decoder.os_lora.system.noise import select_background_bins as _background_bins
from weak_decoder.os_lora.system.nonuniform_sampling import (
    NonuniformPatternBank,
    build_pattern_bank,
    estimate_pattern_noise_covariance,
    lora_branch_color_mismatch,
    matrix_free_crossfit_gls_spectrum_power,
    pattern_bank_split_spectra,
    pattern_bin_values,
    prepare_dechirped_symbol,
    select_pattern_subset,
    select_pattern_subset_by_information,
    target_response_vector,
)


def _parse_args() -> argparse.Namespace:
    """解析真实 capture、跨 capture 训练和 GLS 评估参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    # 测试 capture：IQ、该 IQ 自己产生的 symbol CSV 和 sync CSV 必须配套。
    parser.add_argument("--test-iq", type=Path, required=True)
    parser.add_argument("--test-symbols", type=Path, required=True)
    parser.add_argument("--test-sync", type=Path, required=True)
    # 训练 capture 只提供包外噪声和 pattern 选择依据，不能与测试 capture 混用。
    parser.add_argument("--training-iq", type=Path, required=True)
    parser.add_argument("--training-symbols", type=Path, required=True)
    parser.add_argument("--training-sync", type=Path, required=True)
    # GT 只保存固定帧每个 frame_symbol_index 对应的正式 FFT bin。
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-name", default="test")
    parser.add_argument("--training-name", default="training")
    parser.add_argument("--candidate-kind", default="canonical")
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--pattern-count", type=int, default=8)
    # 每个包外窗口长度恰好为一个过采样 LoRa symbol。
    parser.add_argument("--training-windows", type=int, default=256)
    parser.add_argument("--training-bins", type=int, default=8)
    parser.add_argument("--training-seed", type=int, default=41)
    # 从每个检测事件附近排除一整段区域，避免前导码、SFD、header 和 payload
    # 泄漏到“纯噪声”窗口中。
    parser.add_argument("--exclude-before-chirps", type=float, default=8.0)
    parser.add_argument("--exclude-after-chirps", type=float, default=110.0)
    parser.add_argument("--gls-loading", type=float, default=0.05)
    parser.add_argument("--exclude-top", type=int, default=8)
    parser.add_argument("--exclude-guard-bins", type=int, default=1)
    parser.add_argument("--crossfit-folds", type=int, default=4)
    parser.add_argument("--cg-iterations", type=int, default=8)
    parser.add_argument("--cg-tolerance", type=float, default=0.0)
    parser.add_argument("--noise-max-lag", type=int, default=64)
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=None,
        help="IQ 采样率；不指定时按 BW×OSR 推导。",
    )
    parser.add_argument(
        "--summary-gls-method",
        choices=("gls_crossfit", "gls_offpacket"),
        default="gls_crossfit",
        help="capture_summary.csv 中 gls_ser 对应的 GLS 版本。",
    )
    parser.add_argument(
        "--capture-summary-csv",
        type=Path,
        default=None,
        help="可选的跨 capture 汇总表；相同测试/训练组合会被更新。",
    )
    return parser.parse_args()


def _groundtruth_bins(path: Path) -> dict[int, int]:
    """读取正式 GT，建立 ``frame_symbol_index -> FFT bin`` 映射。"""

    out: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index = int(row["frame_symbol_index"])
            out[index] = int(row["groundtruth_fft_bin"])
    if not out:
        raise RuntimeError(f"no ground-truth rows found in {path}")
    return out


def _load_packets_with_external_gt(symbol_path: Path, gt: dict[int, int]) -> list[dict[str, Any]]:
    """加载测试帧，并仅在离线评分字段 ``gt_bin`` 中注入外部 GT。

    ``raw_fft_bin``、同步起点和 CFO 等字段仍来自当前 low-SNR capture；这里不会
    按 GT 修正任何接收机中间量。
    """

    packets = load_packets(symbol_path)
    for packet in packets:
        for symbol in packet["payload_symbols"]:
            index = int(symbol["frame_symbol_index"])
            if index not in gt:
                raise KeyError(f"ground truth has no frame_symbol_index={index}")
            symbol["gt_bin"] = int(gt[index])
    return packets


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠的样点区间，便于快速排除所有疑似信号区域。"""

    merged: list[tuple[int, int]] = []
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((int(start), int(stop)))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], int(stop)))
    return merged


def _signal_intervals_from_sync(
    sync_path: Path,
    sample_count: int,
    chirp_samples: int,
    before_chirps: float,
    after_chirps: float,
) -> list[tuple[int, int]]:
    """由同步 CSV 的粗检测起点构造需要从噪声训练中排除的区间。

    即使某个事件没有通过最终 frame sync，也照样排除；这样可以避免把真实但
    同步失败的弱包误当作包外噪声。
    """

    intervals: list[tuple[int, int]] = []
    with sync_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = str(row.get("detected_start_sample", "")).strip()
            if not text:
                continue
            detected = int(float(text))
            start = max(0, detected - int(round(float(before_chirps) * chirp_samples)))
            stop = min(sample_count, detected + int(round(float(after_chirps) * chirp_samples)))
            intervals.append((start, stop))
    if not intervals:
        raise RuntimeError(f"no detected events found in {sync_path}")
    return _merge_intervals(intervals)


def _offpacket_windows(
    samples: np.ndarray,
    sync_path: Path,
    sf: int,
    os_factor: int,
    max_windows: int,
    seed: int,
    before_chirps: float,
    after_chirps: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """在所有信号保护区之外随机抽取单-symbol长度的包外 IQ 窗口。"""

    window_len = (1 << int(sf)) * int(os_factor)
    intervals = _signal_intervals_from_sync(
        sync_path=sync_path,
        sample_count=int(samples.size),
        chirp_samples=window_len,
        before_chirps=before_chirps,
        after_chirps=after_chirps,
    )
    starts = _off_packet_starts(
        sample_count=int(samples.size),
        window_len=window_len,
        intervals=intervals,
        max_windows=int(max_windows),
        seed=int(seed),
    )
    if not starts:
        raise RuntimeError(f"no off-packet windows available for {sync_path}")
    windows = np.asarray(
        [samples[int(start): int(start) + window_len] for start in starts],
        dtype=np.complex64,
    )
    return windows, starts


def _csv_truth(value: Any) -> bool:
    """把同步 CSV 中常见的布尔写法统一转换为 ``bool``。"""

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _capture_sync_metrics(sync_path: Path, sample_rate: float) -> dict[str, Any]:
    """从同步 CSV 汇总检测次数、strict sync 次数与包检测时刻。"""

    if float(sample_rate) <= 0.0:
        raise ValueError("sample_rate must be positive")
    detected_starts: list[int] = []
    strict_starts: list[int] = []
    with sync_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = str(row.get("detected_start_sample", "")).strip()
            if not text:
                continue
            start = int(float(text))
            detected_starts.append(start)
            if _csv_truth(row.get("grlora_framesync_valid", "0")):
                strict_starts.append(start)
    if not detected_starts:
        raise RuntimeError(f"no detected events found in {sync_path}")

    def seconds_text(starts: Sequence[int]) -> str:
        return "|".join(f"{float(start) / float(sample_rate):.6f}" for start in starts)

    return {
        "packet_times_s": seconds_text(detected_starts),
        "strict_packet_times_s": seconds_text(strict_starts),
        "detection_count": int(len(detected_starts)),
        "strict_sync_count": int(len(strict_starts)),
    }


def _dechirp_pnr_db(branch_spectrum: np.ndarray, guard_bins: int) -> float:
    """计算传统单 branch dechirp FFT 的峰值对背景中位数比。"""

    power = np.abs(np.asarray(branch_spectrum)).astype(np.float64) ** 2
    if power.size == 0:
        return float("nan")
    background = _background_bins(power, exclude_top=1, guard_bins=int(guard_bins))
    if background.size == 0:
        return float("nan")
    noise_floor = float(np.median(power[background]))
    return float(10.0 * math.log10((float(np.max(power)) + 1e-30) / (noise_floor + 1e-30)))


def _training_covariance(
    windows: np.ndarray,
    sf: int,
    os_factor: int,
    bank: NonuniformPatternBank,
    bin_count: int,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    """估计候选 pattern 输出的经验噪声协方差。

    每个包外 IQ 窗口先按正式解调链 dechirp，再在若干代表性 raw bin 上计算
    所有 pattern 的复输出向量 ``y``。最终对堆叠后的向量估计
    ``C = E[(y-mean(y))(y-mean(y))^H]``。
    """

    n_bins = 1 << int(sf)
    # 在整个 FFT 频带上等间隔取 bin，避免协方差只代表某个局部频段。
    bins = tuple(
        int(value)
        for value in np.linspace(
            0,
            n_bins,
            num=min(max(1, int(bin_count)), n_bins),
            endpoint=False,
            dtype=np.int64,
        )
    )
    downchirp = _oversampled_downchirp(int(sf), int(os_factor), 0, 0.0)
    vectors: list[np.ndarray] = []
    for window in windows:
        # 噪声窗口也必须经过和在线 detector 相同的 dechirp/pattern 投影。
        dechirped = np.asarray(window * downchirp, dtype=np.complex64)
        for raw_bin in bins:
            vectors.append(pattern_bin_values(dechirped, int(raw_bin), bank))
    matrix = np.asarray(vectors, dtype=np.complex128)
    return _empirical_covariance(matrix), bins, int(matrix.shape[0])


def _subset_covariance(
    full_bank: NonuniformPatternBank,
    selected_bank: NonuniformPatternBank,
    covariance: np.ndarray,
) -> np.ndarray:
    """从完整候选协方差中提取最终入选 pattern 对应的子矩阵。"""

    by_name = {str(name): index for index, name in enumerate(full_bank.names)}
    indices = [by_name[str(name)] for name in selected_bank.names]
    return np.asarray(covariance[np.ix_(indices, indices)], dtype=np.complex128)


def _fixed_gls_power(
    spectra: np.ndarray,
    covariance: np.ndarray,
    bank: NonuniformPatternBank,
    loading: float,
) -> np.ndarray:
    """使用训练 capture 的固定协方差计算所有候选 bin 的 GLS 功率。

    对每个候选 bin 的 pattern 观测向量 ``y``，使用统一权重
    ``w = C^-1 a``，评分为 ``|a^H C^-1 y|^2 / (a^H C^-1 a)``。
    对角加载用于限制有限噪声窗口下协方差求逆的病态程度。
    """

    cov = np.asarray(covariance, dtype=np.complex128)
    mean_power = max(float(np.real(np.trace(cov))) / max(1, cov.shape[0]), 1e-30)
    # 加载量按 pattern 平均噪声功率缩放，使参数不依赖 IQ 的绝对幅度。
    loaded = cov + np.eye(cov.shape[0], dtype=np.complex128) * float(loading) * mean_power
    target = target_response_vector(bank)
    try:
        inverse_target = np.linalg.solve(loaded, target)
    except np.linalg.LinAlgError:
        inverse_target = np.linalg.pinv(loaded, rcond=1e-10) @ target
    denominator = max(float(np.real(target.conj().T @ inverse_target)), 1e-30)
    projection = inverse_target.conj().T @ np.asarray(spectra, dtype=np.complex128)
    return np.asarray(np.abs(projection).astype(np.float64) ** 2 / denominator, dtype=np.float64)


def _noise_statistics(
    windows: np.ndarray,
    os_factor: int,
    max_lag: int,
    selected_covariance: np.ndarray,
    selected_bank: NonuniformPatternBank,
    training_bins: Sequence[int],
) -> dict[str, Any]:
    """从时域、频域和 pattern 域三个角度检查包外噪声是否为白噪声。

    输出包括复自相关、白噪声 3-sigma 参考、PSD 谱平坦度、OSR 四个 offset
    的功率离散程度，以及经验 pattern 协方差对白噪声理论形状的失配。
    """

    values = np.asarray(windows, dtype=np.complex128)
    # 每个窗口单独去直流，避免不同时间段的 DC 漂移直接主导相关性。
    values = values - np.mean(values, axis=1, keepdims=True)
    power = float(np.mean(np.abs(values) ** 2))
    lag_values: list[float] = []
    # 跨窗口累加同一 lag 的复自相关，但不把两个窗口的边界拼接在一起。
    for lag in range(1, min(int(max_lag), values.shape[1] - 1) + 1):
        left = values[:, :-lag]
        right = values[:, lag:]
        numerator = complex(np.sum(right * np.conj(left), dtype=np.complex128))
        denominator = max(float(np.sum(np.abs(left) ** 2, dtype=np.float64)), 1e-30)
        lag_values.append(float(abs(numerator / denominator)))
    pair_count = int(values.shape[0] * max(1, values.shape[1] - 1))
    # 独立白噪声下样本相关系数的量级约为 1/sqrt(N)，这里给出 3-sigma 参考。
    white_3sigma = float(3.0 / math.sqrt(max(1, pair_count)))
    # 对多个窗口的 periodogram 取平均后再计算谱平坦度；越接近 1 越像白噪声。
    spectra = np.fft.fft(values, axis=1)
    psd = np.mean(np.abs(spectra).astype(np.float64) ** 2, axis=0)
    positive = np.maximum(psd, max(float(np.mean(psd)), 1e-30) * 1e-15)
    flatness = float(np.exp(np.mean(np.log(positive))) / max(float(np.mean(positive)), 1e-30))
    p05, p95 = np.percentile(positive, (5.0, 95.0))
    # 检查 OSR=4 的四个固定采样相位是否存在明显功率不均衡。
    offset_powers = [float(np.mean(np.abs(values[:, offset:: int(os_factor)]) ** 2)) for offset in range(int(os_factor))]
    # 将白噪声理论协方差缩放到与经验矩阵相同 trace，再比较矩阵形状。
    white_covariance = estimate_pattern_noise_covariance(selected_bank, training_bins)
    empirical = np.asarray(selected_covariance, dtype=np.complex128)
    empirical_trace = max(float(np.real(np.trace(empirical))), 1e-30)
    white_trace = max(float(np.real(np.trace(white_covariance))), 1e-30)
    scaled_white = white_covariance * (empirical_trace / white_trace)
    covariance_mismatch = float(
        np.linalg.norm(empirical - scaled_white, ord="fro")
        / max(float(np.linalg.norm(scaled_white, ord="fro")), 1e-30)
    )
    return {
        "window_count": int(values.shape[0]),
        "samples_per_window": int(values.shape[1]),
        "raw_noise_power": power,
        "mean_abs_autocorrelation_lags_1_to_max": float(np.mean(lag_values)) if lag_values else 0.0,
        "max_abs_autocorrelation_lags_1_to_max": float(np.max(lag_values)) if lag_values else 0.0,
        "lag1_abs_autocorrelation": float(lag_values[0]) if lag_values else 0.0,
        "white_3sigma_autocorrelation_reference": white_3sigma,
        "significant_lag_count": int(sum(value > white_3sigma for value in lag_values)),
        "tested_lag_count": int(len(lag_values)),
        "psd_spectral_flatness": flatness,
        "psd_p95_to_p05_db": float(10.0 * math.log10((float(p95) + 1e-30) / (float(p05) + 1e-30))),
        "osr_offset_power_cv": float(np.std(offset_powers) / max(float(np.mean(offset_powers)), 1e-30)),
        "selected_pattern_covariance_white_mismatch": covariance_mismatch,
    }


def _evaluate(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    selected_bank: NonuniformPatternBank,
    selected_covariance: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """在完全相同的 payload symbols 上比较四种 FFT-bin 判决器。

    四种方法共享当前 low-SNR capture 自己估计的同步起点和 CFO。外部 GT 只在
    得到 hard bin 后计算 correct/error，不会进入任何方法的候选集合或评分。
    """

    methods = ("ordinary_fft", "savaux", "gls_crossfit", "gls_offpacket")
    symbol_rows: list[dict[str, Any]] = []
    elapsed = {method: 0.0 for method in methods}
    for packet in packets:
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        # 与现有 Savaux baseline 保持一致：从每个 chip 的中心采样相位定义
        # symbol 起点，再展开完整的 OSR=4 symbol。
        origin_shift = os_factor // 2
        header_start = int(packet["header_start_sample"]) + origin_shift
        for symbol in packet["payload_symbols"]:
            start = int(symbol["start_sample"]) + origin_shift
            gt_bin = int(symbol["gt_bin"])
            # 普通 FFT 已由 header-first 链计算，这里直接复用 CSV 中的 hard bin。
            begin = time.perf_counter()
            ordinary_bin = int(symbol["raw_fft_bin"])
            elapsed["ordinary_fft"] += time.perf_counter() - begin
            # Savaux 将 OSR 个固定 branch 分别做论文定义的 DFT，再进行相位合并。
            begin = time.perf_counter()
            savaux_spectrum, savaux_branches, _phase = paper_oversampled_spectrum(
                samples=samples,
                start_sample=start,
                sf=sf,
                os_factor=os_factor,
                cfo_int=int(packet["cfo_int"]),
                cfo_frac=float(packet["cfo_frac"]),
                header_start_sample=header_start,
                cfo_correction_mode="continuous",
            )
            savaux_power = np.abs(savaux_spectrum).astype(np.float64) ** 2
            savaux_bin = int(np.argmax(savaux_power))
            # PNR 使用传统单 branch dechirp FFT，不使用 Savaux 合并或 GT bin。
            dechirp_pnr_db = _dechirp_pnr_db(
                savaux_branches[0],
                guard_bins=int(args.exclude_guard_bins),
            )
            # 用传统单 branch 的 dechirp FFT 计算 GT-bin 能量。GT 只在三种方法
            # 都完成 hard-bin 判决后参与离线质量评估，不会反向影响解调结果。
            traditional_power = np.abs(savaux_branches[0]).astype(np.float64) ** 2
            dechirp_gt_bin_power = float(traditional_power[gt_bin])
            dechirp_total_fft_energy = float(np.sum(traditional_power, dtype=np.float64))
            dechirp_residual_fft_energy = max(
                0.0,
                dechirp_total_fft_energy - dechirp_gt_bin_power,
            )
            dechirp_gt_symbol_snr_db = float(
                10.0
                * math.log10(
                    (dechirp_gt_bin_power + 1e-30)
                    / (dechirp_residual_fft_energy + 1e-30)
                )
            )
            elapsed["savaux"] += time.perf_counter() - begin
            # 排除最强候选及其邻域，用剩余 bin 估计当前 symbol 的背景协方差。
            background = _background_bins(
                savaux_power,
                exclude_top=int(args.exclude_top),
                guard_bins=int(args.exclude_guard_bins),
            )
            # 非均匀 pattern bank 与 Savaux 使用同一段已补偿 IQ，不重新同步。
            dechirped = prepare_dechirped_symbol(
                samples=samples,
                start_sample=start,
                sf=sf,
                os_factor=os_factor,
                cfo_int=int(packet["cfo_int"]),
                cfo_frac=float(packet["cfo_frac"]),
                header_start_sample=header_start,
                cfo_correction_mode="continuous",
            )
            spectra, _head, _tail = pattern_bank_split_spectra(dechirped, selected_bank)
            # Cross-fit GLS：每一折的候选 bin 使用其他折背景 bin 估计的协方差，
            # 防止待评分 bin 同时参与自己的协方差估计。
            begin = time.perf_counter()
            crossfit = matrix_free_crossfit_gls_spectrum_power(
                spectra,
                covariance_bins=background,
                diagonal_loading=float(args.gls_loading),
                folds=int(args.crossfit_folds),
                max_iterations=int(args.cg_iterations),
                tolerance=float(args.cg_tolerance),
            )
            crossfit_bin = int(np.argmax(crossfit.power))
            elapsed["gls_crossfit"] += time.perf_counter() - begin
            # Fixed GLS：完全使用另一份 training capture 的包外协方差 C。
            begin = time.perf_counter()
            offpacket_bin = int(
                np.argmax(
                    _fixed_gls_power(
                        spectra,
                        covariance=selected_covariance,
                        bank=selected_bank,
                        loading=float(args.gls_loading),
                    )
                )
            )
            elapsed["gls_offpacket"] += time.perf_counter() - begin
            top_two = np.partition(savaux_power, -2)[-2:]
            savaux_margin_db = float(
                10.0 * math.log10((float(np.max(top_two)) + 1e-30) / (float(np.min(top_two)) + 1e-30))
            )
            row: dict[str, Any] = {
                "packet_index": int(packet["packet_index"]),
                "frame_index": int(packet["frame_index"]),
                "frame_symbol_index": int(symbol["frame_symbol_index"]),
                "payload_symbol_index": int(symbol["payload_symbol_index"]),
                "start_sample": int(symbol["start_sample"]),
                "gt_bin": gt_bin,
                "ordinary_fft_bin": ordinary_bin,
                "savaux_bin": savaux_bin,
                "gls_crossfit_bin": crossfit_bin,
                "gls_offpacket_bin": offpacket_bin,
                "dechirp_pnr_db": dechirp_pnr_db,
                "dechirp_gt_bin_power": dechirp_gt_bin_power,
                "dechirp_total_fft_energy": dechirp_total_fft_energy,
                "dechirp_residual_fft_energy": dechirp_residual_fft_energy,
                "dechirp_gt_symbol_snr_db": dechirp_gt_symbol_snr_db,
                "savaux_margin_db": savaux_margin_db,
                "branch_color_mismatch": float(lora_branch_color_mismatch(savaux_branches, background)),
            }
            # correct 字段只负责事后评估；上面的所有判决在此之前已经结束。
            for method in methods:
                selected = int(row[f"{method}_bin"])
                row[f"{method}_correct"] = int(selected == gt_bin)
            symbol_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        errors = sum(not bool(row[f"{method}_correct"]) for row in symbol_rows)
        by_frame: dict[int, int] = {}
        for row in symbol_rows:
            frame = int(row["packet_index"])
            by_frame.setdefault(frame, 0)
            by_frame[frame] += int(not bool(row[f"{method}_correct"]))
        # FFT-bin FER：一帧 49 个 payload symbols 中任意一个出错，整帧记错。
        frame_errors = sum(value > 0 for value in by_frame.values())
        # fixes/breaks 是与 Savaux 的配对比较，比只看两行 SER 更容易发现
        # “修复一些 symbol、同时又破坏另一些 symbol”的情况。
        fixes = sum(
            int(row["savaux_bin"]) != int(row["gt_bin"])
            and int(row[f"{method}_bin"]) == int(row["gt_bin"])
            for row in symbol_rows
        )
        breaks = sum(
            int(row["savaux_bin"]) == int(row["gt_bin"])
            and int(row[f"{method}_bin"]) != int(row["gt_bin"])
            for row in symbol_rows
        )
        summary_rows.append(
            {
                "test_name": str(args.test_name),
                "training_name": str(args.training_name),
                "method": method,
                "pattern_count": 0 if method in {"ordinary_fft", "savaux"} else len(selected_bank.names),
                "frame_count": int(len(by_frame)),
                "frame_errors": int(frame_errors),
                "fer": float(frame_errors / max(1, len(by_frame))),
                "symbol_count": int(len(symbol_rows)),
                "errors": int(errors),
                "ser": float(errors / max(1, len(symbol_rows))),
                "fixes_vs_savaux": int(fixes),
                "breaks_vs_savaux": int(breaks),
                "elapsed_seconds": float(elapsed[method]),
                "mean_milliseconds_per_symbol": float(1000.0 * elapsed[method] / max(1, len(symbol_rows))),
            }
        )
    return summary_rows, symbol_rows


def _packet_gt_snr_rows(
    symbol_rows: Sequence[dict[str, Any]],
    capture: str,
    sample_rate: float,
) -> list[dict[str, Any]]:
    """把 GT-bin dechirp 能量按包加权汇总为逐包 SNR。

    每个 symbol 的权重由其 FFT 总能量自然决定。等价地，先把一个包内所有
    symbol 的 GT-bin 能量相加，再除以其余 ``2^SF - 1`` 个 bin 的总能量：
    ``SNR_packet = 10*log10(sum(E_gt) / sum(E_other))``。
    """

    if float(sample_rate) <= 0.0:
        raise ValueError("sample_rate must be positive")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in symbol_rows:
        grouped.setdefault(int(row["packet_index"]), []).append(row)

    packet_rows: list[dict[str, Any]] = []
    for packet_index, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: int(item["payload_symbol_index"]))
        gt_energy = float(sum(float(item["dechirp_gt_bin_power"]) for item in ordered))
        residual_energy = float(
            sum(float(item["dechirp_residual_fft_energy"]) for item in ordered)
        )
        total_energy = float(sum(float(item["dechirp_total_fft_energy"]) for item in ordered))
        packet_snr_db = float(
            10.0 * math.log10((gt_energy + 1e-30) / (residual_energy + 1e-30))
        )
        payload_start_sample = min(int(item["start_sample"]) for item in ordered)
        packet_rows.append(
            {
                "capture": str(capture),
                "packet_index": int(packet_index),
                "frame_index": int(ordered[0]["frame_index"]),
                "payload_start_sample": int(payload_start_sample),
                "payload_start_time_s": float(payload_start_sample) / float(sample_rate),
                "symbol_count": int(len(ordered)),
                "dechirp_gt_bin_energy": gt_energy,
                "dechirp_residual_fft_energy": residual_energy,
                "dechirp_total_fft_energy": total_energy,
                "dechirp_gt_energy_ratio": float(gt_energy / max(total_energy, 1e-30)),
                "dechirp_gt_packet_snr_db": packet_snr_db,
            }
        )
    return packet_rows


def _capture_summary_row(
    args: argparse.Namespace,
    summary_rows: Sequence[dict[str, Any]],
    symbol_rows: Sequence[dict[str, Any]],
    sync_metrics: dict[str, Any],
    packet_snr_rows: Sequence[dict[str, Any]],
    sample_rate: float,
    bandwidth: float,
    capture_sample_count: int,
) -> dict[str, Any]:
    """把同步、dechirp 质量与三种解调器统计压缩成一行 capture 记录。"""

    by_method = {str(row["method"]): row for row in summary_rows}
    required = {"ordinary_fft", "savaux", str(args.summary_gls_method)}
    missing = sorted(required - set(by_method))
    if missing:
        raise KeyError(f"summary rows are missing methods: {missing}")
    fft = by_method["ordinary_fft"]
    savaux = by_method["savaux"]
    gls = by_method[str(args.summary_gls_method)]
    crossfit = by_method.get("gls_crossfit")
    offpacket = by_method.get("gls_offpacket")
    pnr_values = [
        float(row["dechirp_pnr_db"])
        for row in symbol_rows
        if math.isfinite(float(row.get("dechirp_pnr_db", float("nan"))))
    ]
    pnr_median = float(np.median(pnr_values)) if pnr_values else float("nan")
    pnr_mean = float(np.mean(pnr_values)) if pnr_values else float("nan")
    packet_snr_values = [
        float(packet["dechirp_gt_packet_snr_db"])
        for packet in packet_snr_rows
        if math.isfinite(float(packet.get("dechirp_gt_packet_snr_db", float("nan"))))
    ]
    packet_snr_median = (
        float(np.median(packet_snr_values)) if packet_snr_values else float("nan")
    )
    packet_snr_mean = float(np.mean(packet_snr_values)) if packet_snr_values else float("nan")
    symbol_count = int(fft["symbol_count"])
    row: dict[str, Any] = {
        "capture": str(args.test_name),
        "training_capture": str(args.training_name),
        "packet_times_s": str(sync_metrics["packet_times_s"]),
        "strict_packet_times_s": str(sync_metrics["strict_packet_times_s"]),
        "detection_count": int(sync_metrics["detection_count"]),
        "strict_sync_count": int(sync_metrics["strict_sync_count"]),
        "evaluated_frame_count": int(fft["frame_count"]),
        "evaluated_symbol_count": symbol_count,
        "capture_duration_s": float(capture_sample_count) / float(sample_rate),
        "sample_rate_hz": float(sample_rate),
        "input_bandwidth_hz": float(bandwidth),
        # 一行代表整个 capture，因此主字段取逐包 SNR 中位数；逐包值另存 packet_snr.csv。
        "dechirp_gt_packet_snr_db": packet_snr_median,
        "dechirp_gt_packet_snr_mean_db": packet_snr_mean,
        "dechirp_gt_packet_snr_min_db": (
            float(np.min(packet_snr_values)) if packet_snr_values else float("nan")
        ),
        "dechirp_gt_packet_snr_max_db": (
            float(np.max(packet_snr_values)) if packet_snr_values else float("nan")
        ),
        "dechirp_gt_packet_snr_values_db": "|".join(
            f"{value:.6f}" for value in packet_snr_values
        ),
        "packet_snr_count": int(len(packet_snr_values)),
        "dechirp_pnr_db": pnr_median,
        "dechirp_pnr_mean_db": pnr_mean,
        "fft_errors": f"{int(fft['errors'])}/{symbol_count}",
        "fft_error_count": int(fft["errors"]),
        "fft_ser": float(fft["ser"]),
        "savaux_error_count": int(savaux["errors"]),
        "savaux_ser": float(savaux["ser"]),
        "gls_method": str(args.summary_gls_method),
        "gls_error_count": int(gls["errors"]),
        "gls_ser": float(gls["ser"]),
        "gls_crossfit_ser": "" if crossfit is None else float(crossfit["ser"]),
        "gls_offpacket_ser": "" if offpacket is None else float(offpacket["ser"]),
    }
    return row


def _upsert_capture_summary(path: Path, row: dict[str, Any]) -> None:
    """向跨 capture 表中插入或更新同一测试/训练/GLS 组合。"""

    existing: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    key = (
        str(row["capture"]),
        str(row["training_capture"]),
        str(row["gls_method"]),
    )
    updated: list[dict[str, Any]] = []
    replaced = False
    for old in existing:
        old_key = (
            str(old.get("capture", "")),
            str(old.get("training_capture", "")),
            str(old.get("gls_method", "")),
        )
        if old_key == key:
            if not replaced:
                updated.append(dict(row))
                replaced = True
            continue
        # 共享表的字段以当前版本为准，避免已经删除的旧指标继续残留为空白列。
        updated.append({field: old.get(field, "") for field in row})
    if not replaced:
        updated.append(dict(row))
    write_csv(path, updated)


def main() -> int:
    """执行跨 capture pattern 训练、真实 IQ 评估并写出可复核产物。"""

    args = _parse_args()
    # 第一阶段：读取正式 GT，并把它附加到训练/测试 symbol 记录的评分字段。
    gt = _groundtruth_bins(args.groundtruth.resolve())
    test_packets = _load_packets_with_external_gt(args.test_symbols.resolve(), gt)
    training_packets = _load_packets_with_external_gt(args.training_symbols.resolve(), gt)
    if not test_packets or not training_packets:
        raise RuntimeError("test and training symbol CSVs must both contain payload frames")
    sf = int(test_packets[0]["sf"])
    os_factor = int(test_packets[0]["os_factor"])
    if sf != int(training_packets[0]["sf"]) or os_factor != int(training_packets[0]["os_factor"]):
        raise ValueError("training and test captures must have matching SF and OSR")
    bandwidth = float(test_packets[0]["bw"])
    sample_rate = (
        float(args.sample_rate)
        if args.sample_rate is not None
        else bandwidth * float(os_factor)
    )
    if sample_rate <= 0.0 or bandwidth <= 0.0 or bandwidth > sample_rate:
        raise ValueError("sample rate and bandwidth are inconsistent")
    # memmap 避免把两份数百 MB IQ 同时完整复制到内存。
    test_samples = np.memmap(args.test_iq.resolve(), dtype=np.complex64, mode="r")
    training_samples = np.memmap(args.training_iq.resolve(), dtype=np.complex64, mode="r")
    sync_metrics = _capture_sync_metrics(args.test_sync.resolve(), sample_rate)
    # 第二阶段：只从 training capture 的检测事件之外抽取噪声窗口。
    training_windows, starts = _offpacket_windows(
        samples=training_samples,
        sync_path=args.training_sync.resolve(),
        sf=sf,
        os_factor=os_factor,
        max_windows=int(args.training_windows),
        seed=int(args.training_seed),
        before_chirps=float(args.exclude_before_chirps),
        after_chirps=float(args.exclude_after_chirps),
    )
    # 第三阶段：构造候选 bank；candidate_limit=0 表示保留该 kind 的全部候选。
    full_bank = build_pattern_bank(sf, os_factor, kind=str(args.candidate_kind))
    candidate_bank = (
        select_pattern_subset(full_bank, int(args.candidate_limit), "diverse")
        if int(args.candidate_limit) > 0
        else full_bank
    )
    covariance, training_bins, snapshot_count = _training_covariance(
        training_windows,
        sf=sf,
        os_factor=os_factor,
        bank=candidate_bank,
        bin_count=int(args.training_bins),
    )
    # 按 a^H C^-1 a 的边际信息增益，从候选 bank 中离线选择指定数量的 pattern。
    selection = select_pattern_subset_by_information(
        candidate_bank,
        covariance,
        max_patterns=int(args.pattern_count),
        diagonal_loading=float(args.gls_loading),
    )
    selected_bank = selection.bank
    selected_covariance = _subset_covariance(candidate_bank, selected_bank, covariance)
    # 第四阶段：在 test capture 上独立运行所有 detector，再用 GT 统一评分。
    summary_rows, symbol_rows = _evaluate(
        samples=test_samples,
        packets=test_packets,
        selected_bank=selected_bank,
        selected_covariance=selected_covariance,
        args=args,
    )
    packet_snr_rows = _packet_gt_snr_rows(
        symbol_rows=symbol_rows,
        capture=str(args.test_name),
        sample_rate=sample_rate,
    )
    capture_summary = _capture_summary_row(
        args=args,
        summary_rows=summary_rows,
        symbol_rows=symbol_rows,
        sync_metrics=sync_metrics,
        packet_snr_rows=packet_snr_rows,
        sample_rate=sample_rate,
        bandwidth=bandwidth,
        capture_sample_count=int(test_samples.size),
    )
    # 噪声颜色指标只描述 training capture 的包外窗口，和测试 SER 分开记录。
    noise_stats = _noise_statistics(
        training_windows,
        os_factor=os_factor,
        max_lag=int(args.noise_max_lag),
        selected_covariance=selected_covariance,
        selected_bank=selected_bank,
        training_bins=training_bins,
    )
    noise_stats.update(
        {
            "capture": str(args.training_name),
            "snapshot_count": int(snapshot_count),
            "offpacket_start_count": int(len(starts)),
        }
    )
    selection_rows = [
        {
            "training_name": str(args.training_name),
            "test_name": str(args.test_name),
            "selection_rank": rank,
            "candidate_index": int(index),
            "pattern_name": str(candidate_bank.names[index]),
            "marginal_information": float(gain),
        }
        for rank, (index, gain) in enumerate(
            zip(selection.indices, selection.marginal_information, strict=True),
            start=1,
        )
    ]
    # 第五阶段：逐 symbol、汇总、pattern 选择与噪声统计分别落盘，避免只有
    # 一行 SER 而无法追溯某个方法具体修复/破坏了哪些 symbol。
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "capture_summary.csv", [capture_summary])
    write_csv(output_dir / "packet_snr.csv", packet_snr_rows)
    write_csv(output_dir / "symbols.csv", symbol_rows)
    write_csv(output_dir / "selection.csv", selection_rows)
    write_csv(output_dir / "noise_summary.csv", [noise_stats])
    if args.capture_summary_csv is not None:
        _upsert_capture_summary(args.capture_summary_csv.resolve(), capture_summary)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "test_packet_count": len(test_packets),
            "training_packet_count": len(training_packets),
            "candidate_count": len(candidate_bank.names),
            "selected_patterns": list(selected_bank.names),
            "training_bins": list(training_bins),
            "training_snapshots": int(snapshot_count),
            "sample_rate_hz": float(sample_rate),
            "input_bandwidth_hz": float(bandwidth),
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = " ".join(f"{row['method']}={row['ser']:.6f}" for row in summary_rows)
    print(f"{args.test_name} trained_on={args.training_name}: {compact}")
    print(
        f"capture_metrics detections={capture_summary['detection_count']} "
        f"strict_sync={capture_summary['strict_sync_count']} "
        f"dechirp_gt_packet_snr_db={capture_summary['dechirp_gt_packet_snr_db']:.3f} "
        f"dechirp_pnr_db={capture_summary['dechirp_pnr_db']:.3f}"
    )
    print(f"training_offpacket_windows={len(starts)} snapshots={snapshot_count}")
    print(f"noise_psd_flatness={noise_stats['psd_spectral_flatness']:.6f}")
    print(f"noise_max_abs_autocorrelation={noise_stats['max_abs_autocorrelation_lags_1_to_max']:.6f}")
    print(f"wrote={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
