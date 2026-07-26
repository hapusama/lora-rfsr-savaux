"""包外噪声窗口与经验协方差的共享工具。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def active_intervals(
    packets: Sequence[dict[str, Any]],
    guard_samples: int,
) -> list[tuple[int, int]]:
    """返回所有已知 LoRa symbol 占用区间合并后的有序列表。"""

    intervals: list[tuple[int, int]] = []
    for packet in packets:
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        symbol_samples = (1 << sf) * os_factor
        symbols = list(packet.get("header_symbols", [])) + list(packet.get("payload_symbols", []))
        for symbol in symbols:
            start = int(symbol["start_sample"]) - int(guard_samples)
            stop = int(symbol["start_sample"]) + symbol_samples + int(guard_samples)
            intervals.append((max(0, start), max(0, stop)))
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for start, stop in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    return merged


def _overlaps(intervals: Sequence[tuple[int, int]], start: int, stop: int) -> bool:
    """用二分查找判断一个窗口是否与任一活动区间重叠。"""

    lo = 0
    hi = len(intervals)
    while lo < hi:
        mid = (lo + hi) // 2
        if intervals[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(intervals):
        return False
    return intervals[lo][0] < stop


def off_packet_starts(
    sample_count: int,
    window_len: int,
    intervals: Sequence[tuple[int, int]],
    max_windows: int,
    seed: int,
) -> tuple[int, ...]:
    """确定性地抽取不与已知数据包重叠的窗口起点。"""

    rng = np.random.default_rng(int(seed))
    starts: list[int] = []
    seen: set[int] = set()
    max_start = int(sample_count) - int(window_len)
    attempts = 0
    max_attempts = max(10000, int(max_windows) * 200)
    while len(starts) < int(max_windows) and attempts < max_attempts:
        attempts += 1
        start = int(rng.integers(0, max_start + 1))
        start = (start // int(window_len)) * int(window_len)
        if start in seen:
            continue
        stop = start + int(window_len)
        if stop > int(sample_count) or _overlaps(intervals, start, stop):
            continue
        seen.add(start)
        starts.append(start)
    starts.sort()
    return tuple(starts)


def empirical_covariance(vectors: np.ndarray) -> np.ndarray:
    """对复向量样本计算去均值后的 Hermitian 经验协方差。"""

    values = np.asarray(vectors, dtype=np.complex128)
    values = values - np.mean(values, axis=0, keepdims=True)
    denom = max(1, values.shape[0] - 1)
    covariance = values.T @ values.conj() / float(denom)
    return (covariance + covariance.conj().T) * 0.5


def covariance_correlation_stats(covariance: np.ndarray) -> tuple[float, float, float]:
    """返回对角功率变异系数及非对角相关系数的均值和最大值。"""

    cov = np.asarray(covariance, dtype=np.complex128)
    if cov.size == 0:
        return 0.0, 0.0, 0.0
    diag = np.maximum(np.real(np.diag(cov)), 1e-30)
    diag_cv = float(np.std(diag) / max(float(np.mean(diag)), 1e-30))
    if cov.shape[0] <= 1:
        return diag_cv, 0.0, 0.0
    scale = np.sqrt(diag[:, None] * diag[None, :])
    corr = cov / scale
    mask = ~np.eye(cov.shape[0], dtype=bool)
    off = np.abs(corr[mask])
    return diag_cv, float(np.mean(off)), float(np.max(off))


def covariance_range_residuals(
    covariance: np.ndarray,
    target: np.ndarray,
    rconds: Sequence[float],
) -> dict[str, float | int]:
    """计算目标向量到不同截断阈值协方差列空间的相对残差。"""

    cov = np.asarray(covariance, dtype=np.complex128)
    response = np.asarray(target, dtype=np.complex128)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(np.real(eigvals), 0.0)
    max_eig = float(np.max(eigvals)) if eigvals.size else 0.0
    output: dict[str, float | int] = {"eig_max": max_eig}
    response_norm = max(float(np.linalg.norm(response)), 1e-30)
    for rcond in rconds:
        keep = eigvals > max(max_eig, 1e-300) * float(rcond)
        if np.any(keep):
            basis = eigvecs[:, keep]
            projection = basis @ (basis.conj().T @ response)
            residual = float(np.linalg.norm(response - projection) / response_norm)
            rank = int(np.sum(keep))
        else:
            residual = 1.0
            rank = 0
        key = f"{float(rcond):.0e}".replace("-", "m")
        output[f"rank_{key}"] = rank
        output[f"range_residual_{key}"] = residual
    return output


def effective_replicas_with_noise_power(
    covariance: np.ndarray,
    target: np.ndarray,
    n_bins: int,
    noise_power: float,
    rcond: float,
) -> float:
    """按给定原始噪声功率计算 GLS 的等效独立副本数。"""

    cov = np.asarray(covariance, dtype=np.complex128)
    inverse = np.linalg.pinv(cov, rcond=float(rcond))
    value = complex(target.conj().T @ inverse @ target)
    return float(float(noise_power) * np.real(value) / float(n_bins))


def raw_noise_stats(windows: np.ndarray, os_factor: int, max_lag: int) -> dict[str, float]:
    """计算原始噪声功率、采样相位功率差异和时域相关性。"""

    values = np.asarray(windows, dtype=np.complex128)
    values = values - np.mean(values)
    power = float(np.mean(np.abs(values) ** 2))
    output: dict[str, float] = {"raw_noise_power": power}
    offset_variances = []
    flat = values.reshape(-1, values.shape[-1])
    for offset in range(int(os_factor)):
        offset_variances.append(
            float(np.mean(np.abs(flat[:, offset:: int(os_factor)]) ** 2))
        )
    if offset_variances:
        output["offset_var_mean"] = float(np.mean(offset_variances))
        output["offset_var_cv"] = float(
            np.std(offset_variances) / max(float(np.mean(offset_variances)), 1e-30)
        )
    lag_correlations = []
    concatenated = flat.reshape(-1)
    concatenated = concatenated - np.mean(concatenated)
    denominator = max(float(np.mean(np.abs(concatenated) ** 2)), 1e-30)
    for lag in range(1, min(int(max_lag), concatenated.size - 1) + 1):
        correlation = (
            np.mean(concatenated[lag:] * np.conj(concatenated[:-lag])) / denominator
        )
        lag_correlations.append(abs(complex(correlation)))
    if lag_correlations:
        output["lag_corr_mean_abs"] = float(np.mean(lag_correlations))
        output["lag_corr_max_abs"] = float(np.max(lag_correlations))
    return output


__all__ = [
    "active_intervals",
    "covariance_correlation_stats",
    "covariance_range_residuals",
    "effective_replicas_with_noise_power",
    "empirical_covariance",
    "off_packet_starts",
    "raw_noise_stats",
]
