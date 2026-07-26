"""pattern 协方差训练与包外噪声重采样工具。"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ...baselines.savaux_oversampled.paper_oversampled_demod import _oversampled_downchirp
from ..system.nonuniform_sampling import (
    NonuniformPatternBank,
    effective_replica_count,
    pattern_bin_values,
    pattern_noise_signatures,
    target_response_vector,
)
from .noise_windows import active_intervals, empirical_covariance, off_packet_starts


def estimate_offpacket_pattern_covariance(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    bank: NonuniformPatternBank,
    max_windows: int,
    bin_count: int,
    guard_symbols: float,
    shrinkage: float,
    seed: int,
) -> tuple[np.ndarray, float, float, float, int, tuple[int, ...]]:
    """用包外窗口估计 pattern 协方差及多种等效副本数。"""

    sf = int(bank.sf)
    os_factor = int(bank.os_factor)
    n_bins = 1 << sf
    window_len = n_bins * os_factor
    guard_samples = int(round(float(guard_symbols) * window_len))
    intervals = active_intervals(packets, guard_samples=guard_samples)
    starts = off_packet_starts(
        int(np.asarray(samples).size),
        window_len,
        intervals,
        int(max_windows),
        int(seed),
    )
    if not starts:
        raise RuntimeError("no off-packet windows available for covariance estimation")
    count = min(max(1, int(bin_count)), n_bins)
    bins = tuple(
        int(value)
        for value in np.linspace(0, n_bins, num=count, endpoint=False, dtype=np.int64)
    )
    downchirp = _oversampled_downchirp(
        sf=sf,
        os_factor=os_factor,
        cfo_int=0,
        cfo_frac=0.0,
    )
    train_vectors: list[np.ndarray] = []
    test_vectors: list[np.ndarray] = []
    raw_power_sum = 0.0
    raw_count = 0
    for window_index, start in enumerate(starts):
        chunk = np.asarray(
            samples[int(start): int(start) + window_len],
            dtype=np.complex64,
        )
        dechirped = (chunk * downchirp).astype(np.complex64)
        raw_power_sum += float(np.sum(np.abs(dechirped).astype(np.float64) ** 2))
        raw_count += int(dechirped.size)
        for raw_bin in bins:
            values = pattern_bin_values(dechirped, int(raw_bin), bank)
            if window_index % 2 == 0:
                train_vectors.append(values)
            else:
                test_vectors.append(values)
    raw_noise_power = raw_power_sum / float(max(1, raw_count))
    empirical = empirical_covariance(np.asarray(train_vectors, dtype=np.complex128))
    test_covariance = empirical_covariance(np.asarray(test_vectors, dtype=np.complex128))
    white = np.zeros_like(empirical)
    for raw_bin in bins:
        signatures = pattern_noise_signatures(bank, int(raw_bin))
        white += signatures @ signatures.conj().T
    white /= float(max(1, len(bins)))
    alpha = float(np.clip(shrinkage, 0.0, 1.0))
    covariance = (1.0 - alpha) * empirical + alpha * raw_noise_power * white
    mean_diag = float(np.real(np.trace(covariance)) / max(1, covariance.shape[0]))
    covariance += (
        np.eye(covariance.shape[0], dtype=np.complex128) * max(mean_diag, 1e-30) * 1e-6
    )
    target = target_response_vector(bank)
    inverse = np.linalg.pinv(covariance, rcond=1e-4)
    empirical_replicas = float(
        raw_noise_power * np.real(target.conj().T @ inverse @ target) / float(n_bins)
    )
    weights = inverse @ target
    test_noise = float(np.real(weights.conj().T @ test_covariance @ weights))
    cross_validated_replicas = float(
        raw_noise_power
        * abs(complex(weights.conj().T @ target)) ** 2
        / (float(n_bins) * max(test_noise, 1e-30))
    )
    white_replicas = float(
        np.median([effective_replica_count(bank, raw_bin) for raw_bin in bins])
    )
    return (
        covariance,
        white_replicas,
        empirical_replicas,
        cross_validated_replicas,
        len(starts),
        bins,
    )


def bootstrap_offpacket_noise(
    clean: np.ndarray,
    packets: Sequence[dict[str, Any]],
    snr_db: float | None,
    seed: int,
    reference_power: float,
    max_source_windows: int,
    guard_symbols: float,
) -> np.ndarray:
    """从包外窗口重采样噪声并按目标 SNR 叠加到输入 IQ。"""

    if snr_db is None:
        return np.asarray(clean, dtype=np.complex64)
    if not packets:
        raise ValueError("packets are required for off-packet bootstrap noise")
    sf = int(packets[0]["sf"])
    os_factor = int(packets[0]["os_factor"])
    window_len = (1 << sf) * os_factor
    guard_samples = int(round(float(guard_symbols) * window_len))
    intervals = active_intervals(packets, guard_samples=guard_samples)
    starts = off_packet_starts(
        int(np.asarray(clean).size),
        window_len,
        intervals,
        int(max_source_windows),
        int(seed) + 7919,
    )
    if not starts:
        raise RuntimeError("no off-packet windows available for bootstrap noise")
    source = np.asarray(
        [
            np.asarray(clean[int(start): int(start) + window_len], dtype=np.complex64)
            for start in starts
        ],
        dtype=np.complex64,
    )
    source = source - np.mean(source, axis=1, keepdims=True, dtype=np.complex128)
    source_power = np.mean(np.abs(source).astype(np.float64) ** 2, axis=1)
    source = source / np.sqrt(np.maximum(source_power, 1e-30))[:, None]
    rng = np.random.default_rng(int(seed))
    generated = np.empty(np.asarray(clean).size, dtype=np.complex64)
    for out_start in range(0, generated.size, window_len):
        source_index = int(rng.integers(0, source.shape[0]))
        shift = int(rng.integers(0, window_len))
        phase = np.exp(2j * np.pi * float(rng.random()))
        block = np.roll(source[source_index], shift) * phase
        count = min(window_len, generated.size - out_start)
        generated[out_start: out_start + count] = np.asarray(block[:count], dtype=np.complex64)
    generated_power = float(np.mean(np.abs(generated).astype(np.float64) ** 2))
    target_noise_power = float(reference_power) / (10.0 ** (float(snr_db) / 10.0))
    scale = math.sqrt(target_noise_power / max(generated_power, 1e-30))
    return (np.asarray(clean, dtype=np.complex64) + generated * scale).astype(np.complex64)


__all__ = ["bootstrap_offpacket_noise", "estimate_offpacket_pattern_covariance"]
