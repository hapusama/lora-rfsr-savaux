"""带物理约束的 LoRa 非均匀 timing-path 证据。

这里的路径集合用于描述一个过采样 LoRa symbol 内部的小数采样相位：

    t[p] = R*p + center + tau0 + slope*(p/(N-1) - 0.5)

其中 R 是过采样倍数。相比任意枚举 q[p]，该搜索空间更小，也更符合物理意义：
它表示残余 STO 或 symbol 内部的 timing drift。对每个候选 raw FFT bin，证据由
插值后的过采样 dechirped 样点进行非均匀匹配求和得到。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from ..baselines.savaux_oversampled.paper_oversampled_demod import (
    _oversampled_downchirp,
    paper_oversampled_spectrum,
)
from ..chirp import bin_to_grlora_symbol, signed_fft_bin


CfoCorrectionMode = Literal["none", "symbol", "continuous"]


@dataclass(frozen=True)
class TimingPathCandidate:
    raw_fft_bin: int
    savaux_power: float
    timing_path_power: float
    timing_path_gain: float
    best_tau0: float
    best_slope: float
    composite_score: float


@dataclass(frozen=True)
class TimingPathDemodResult:
    raw_fft_bin: int
    signed_fft_bin: int
    symbol_value: int
    savaux_raw_fft_bin: int
    selected_by_path_override: bool
    candidate_bins: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    candidate_savaux_powers: tuple[float, ...]
    candidate_timing_powers: tuple[float, ...]
    candidate_timing_gains: tuple[float, ...]
    candidate_tau0: tuple[float, ...]
    candidate_slope: tuple[float, ...]


def _validate_os_factor(os_factor: int) -> int:
    value = int(os_factor)
    if value <= 0:
        raise ValueError(f"os_factor must be positive, got {os_factor}")
    return value


def _top_bins(power: np.ndarray, top_k: int) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.int64)
    k = min(max(1, int(top_k)), values.size)
    if k >= values.size:
        return np.argsort(values)[::-1].astype(np.int64)
    partial = np.argpartition(values, -k)[-k:]
    order = partial[np.argsort(values[partial])[::-1]]
    return order.astype(np.int64)


def _prepare_dechirped_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    header_start_sample: int | None,
    cfo_correction_mode: CfoCorrectionMode,
) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    start = int(start_sample)
    pad = 2
    stop = start + n_bins * os_value
    if start - pad < 0 or stop + pad > np.asarray(samples).size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")

    mode = str(cfo_correction_mode)
    if mode not in {"none", "symbol", "continuous"}:
        raise ValueError(f"unknown CFO correction mode: {cfo_correction_mode}")
    symbol = np.asarray(samples[start - pad : stop + pad], dtype=np.complex64)
    use_cfo_int = int(cfo_int) if mode in {"symbol", "continuous"} else 0
    use_cfo_frac = float(cfo_frac) if mode in {"symbol", "continuous"} else 0.0
    if mode == "continuous":
        if header_start_sample is None:
            raise ValueError("header_start_sample is required for continuous CFO correction")
        cfo_total = float(cfo_int) + float(cfo_frac)
        relative_chip_start = float(start - int(header_start_sample)) / float(os_value)
        cfo_common_phase_rad = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        symbol = (symbol * np.exp(-1j * cfo_common_phase_rad)).astype(np.complex64)

    downchirp = _oversampled_downchirp(
        sf=sf,
        os_factor=os_value,
        cfo_int=use_cfo_int,
        cfo_frac=use_cfo_frac,
    )
    padded_downchirp = np.pad(downchirp, (pad, pad), mode="edge")
    return (symbol * padded_downchirp).astype(np.complex64)


def _interp_complex(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    x = np.asarray(positions, dtype=np.float64)
    x0 = np.floor(x).astype(np.int64)
    frac = x - x0
    x0 = np.clip(x0, 0, values.size - 2)
    return (
        (1.0 - frac) * values[x0].astype(np.complex128)
        + frac * values[x0 + 1].astype(np.complex128)
    ).astype(np.complex64)


def _timing_path_value(
    dechirped_padded: np.ndarray,
    candidate_bin: int,
    sf: int,
    os_factor: int,
    tau0: float,
    slope: float,
) -> complex:
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    k = int(candidate_bin)
    p = np.arange(n_bins, dtype=np.float64)
    if n_bins > 1:
        drift = float(slope) * (p / float(n_bins - 1) - 0.5)
    else:
        drift = 0.0
    # 加 2 是为了补偿 _prepare_dechirped_symbol 在开头添加的两个 padding 样点。
    positions = 2.0 + os_value * p + (os_value // 2) + float(tau0) + drift
    picked = _interp_complex(dechirped_padded, positions)
    q_frac = (os_value // 2) + float(tau0) + drift
    q_norm = q_frac / float(os_value)
    kernel = np.exp(-2j * np.pi * float(k) * p / float(n_bins))
    if k != 0:
        tail = p >= float(n_bins - k)
        tail_phase = np.ones(n_bins, dtype=np.complex128)
        tail_phase[tail] = np.exp(2j * np.pi * q_norm[tail])
    else:
        tail_phase = 1.0
    branch_weight = np.exp(-2j * np.pi * q_norm * float(k) / float(n_bins))
    return complex(np.sum(picked * kernel * tail_phase * branch_weight) / math.sqrt(float(n_bins)))


def score_timing_path_candidates(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    candidate_bins: Sequence[int],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "none",
    tau_grid: Sequence[float] = (-0.5, -0.25, 0.0, 0.25, 0.5),
    slope_grid: Sequence[float] = (-0.5, 0.0, 0.5),
    path_gain_power: float = 0.20,
    slope_penalty_power: float = 0.10,
    savaux_power: np.ndarray | None = None,
) -> tuple[TimingPathCandidate, ...]:
    """使用线性小数 timing path 为候选 bin 评分。"""

    os_value = _validate_os_factor(os_factor)
    dechirped = _prepare_dechirped_symbol(
        samples=samples,
        start_sample=int(start_sample),
        sf=sf,
        os_factor=os_value,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    power_lookup = None if savaux_power is None else np.asarray(savaux_power, dtype=np.float64)
    out: list[TimingPathCandidate] = []
    for raw_bin in candidate_bins:
        bin_i = int(raw_bin)
        best_power = -1.0
        best_tau = 0.0
        best_slope = 0.0
        for tau0 in tau_grid:
            for slope in slope_grid:
                value = _timing_path_value(
                    dechirped_padded=dechirped,
                    candidate_bin=bin_i,
                    sf=sf,
                    os_factor=os_value,
                    tau0=float(tau0),
                    slope=float(slope),
                )
                path_power = float(abs(value) ** 2)
                if path_power > best_power:
                    best_power = path_power
                    best_tau = float(tau0)
                    best_slope = float(slope)
        savaux_bin_power = float(power_lookup[bin_i]) if power_lookup is not None else best_power
        gain = float(best_power / (savaux_bin_power + 1e-30))
        slope_factor = 1.0 / (1.0 + abs(best_slope))
        composite = float(
            savaux_bin_power
            * max(gain, 1e-12) ** float(path_gain_power)
            * slope_factor ** float(slope_penalty_power)
        )
        out.append(
            TimingPathCandidate(
                raw_fft_bin=bin_i,
                savaux_power=savaux_bin_power,
                timing_path_power=float(best_power),
                timing_path_gain=gain,
                best_tau0=best_tau,
                best_slope=best_slope,
                composite_score=composite,
            )
        )
    return tuple(out)


def score_fixed_timing_path_candidates(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    candidate_bins: Sequence[int],
    tau0: float,
    slope: float,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "none",
    path_gain_power: float = 0.20,
    slope_penalty_power: float = 0.10,
    savaux_power: np.ndarray | None = None,
) -> tuple[TimingPathCandidate, ...]:
    """在一条共享的小数 timing path 上为候选 bin 评分。

    与 ``score_timing_path_candidates`` 不同，这里不会为每个候选分别搜索最优路径。
    调用方可以为每个 packet 只估计一次 ``tau0``/``slope``，再让所有 payload
    symbols 复用该路径，从而明显降低对逐 symbol 噪声的过拟合风险。
    """

    os_value = _validate_os_factor(os_factor)
    dechirped = _prepare_dechirped_symbol(
        samples=samples,
        start_sample=int(start_sample),
        sf=sf,
        os_factor=os_value,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    power_lookup = None if savaux_power is None else np.asarray(savaux_power, dtype=np.float64)
    out: list[TimingPathCandidate] = []
    for raw_bin in candidate_bins:
        bin_i = int(raw_bin)
        value = _timing_path_value(
            dechirped_padded=dechirped,
            candidate_bin=bin_i,
            sf=sf,
            os_factor=os_value,
            tau0=float(tau0),
            slope=float(slope),
        )
        path_power = float(abs(value) ** 2)
        savaux_bin_power = float(power_lookup[bin_i]) if power_lookup is not None else path_power
        gain = float(path_power / (savaux_bin_power + 1e-30))
        slope_factor = 1.0 / (1.0 + abs(float(slope)))
        composite = float(
            savaux_bin_power
            * max(gain, 1e-12) ** float(path_gain_power)
            * slope_factor ** float(slope_penalty_power)
        )
        out.append(
            TimingPathCandidate(
                raw_fft_bin=bin_i,
                savaux_power=savaux_bin_power,
                timing_path_power=path_power,
                timing_path_gain=gain,
                best_tau0=float(tau0),
                best_slope=float(slope),
                composite_score=composite,
            )
        )
    return tuple(out)


def demod_timing_path_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    is_header: bool = False,
    ldro: bool = False,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "none",
    candidate_top_k: int = 16,
    tau_grid: Sequence[float] = (-0.5, -0.25, 0.0, 0.25, 0.5),
    slope_grid: Sequence[float] = (-0.5, 0.0, 0.5),
    path_gain_power: float = 0.20,
    slope_penalty_power: float = 0.10,
    override_margin_db: float = 0.10,
    min_savaux_rel_db: float = -4.0,
    min_path_gain: float = 1.02,
) -> TimingPathDemodResult:
    """结合 Savaux Top-K 与 timing-path 证据解调一个 symbol。"""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    savaux_spectrum, _branches, _phase = paper_oversampled_spectrum(
        samples=samples,
        start_sample=int(start_sample),
        sf=sf,
        os_factor=os_value,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    savaux_power_all = np.abs(savaux_spectrum).astype(np.float64) ** 2
    savaux_bin = int(np.argmax(savaux_power_all))
    candidate_bins = _top_bins(savaux_power_all, int(candidate_top_k))
    if savaux_bin not in set(int(v) for v in candidate_bins):
        candidate_bins = np.concatenate([np.asarray([savaux_bin], dtype=np.int64), candidate_bins])

    candidates = score_timing_path_candidates(
        samples=samples,
        start_sample=int(start_sample),
        sf=sf,
        os_factor=os_value,
        candidate_bins=tuple(int(v) for v in candidate_bins),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
        tau_grid=tau_grid,
        slope_grid=slope_grid,
        path_gain_power=float(path_gain_power),
        slope_penalty_power=float(slope_penalty_power),
        savaux_power=savaux_power_all,
    )
    savaux_idx = int(np.where(candidate_bins == savaux_bin)[0][0])
    best_idx = savaux_idx
    best_score = float(candidates[savaux_idx].composite_score)
    override_margin = 10.0 ** (float(override_margin_db) / 10.0)
    min_rel = 10.0 ** (float(min_savaux_rel_db) / 10.0)
    max_savaux = float(np.max([item.savaux_power for item in candidates])) if candidates else 0.0
    for idx, item in enumerate(candidates):
        if int(item.raw_fft_bin) == savaux_bin:
            continue
        if float(item.savaux_power / (max_savaux + 1e-30)) < min_rel:
            continue
        if float(item.timing_path_gain) < float(min_path_gain):
            continue
        if float(item.composite_score) > best_score * override_margin:
            best_idx = int(idx)
            best_score = float(item.composite_score)

    raw_bin = int(candidates[best_idx].raw_fft_bin)
    symbol_value = bin_to_grlora_symbol(raw_bin, sf=sf, is_header=is_header, ldro=ldro)
    return TimingPathDemodResult(
        raw_fft_bin=raw_bin,
        signed_fft_bin=signed_fft_bin(raw_bin, n_bins),
        symbol_value=int(symbol_value),
        savaux_raw_fft_bin=savaux_bin,
        selected_by_path_override=bool(raw_bin != savaux_bin),
        candidate_bins=tuple(int(item.raw_fft_bin) for item in candidates),
        candidate_scores=tuple(float(item.composite_score) for item in candidates),
        candidate_savaux_powers=tuple(float(item.savaux_power) for item in candidates),
        candidate_timing_powers=tuple(float(item.timing_path_power) for item in candidates),
        candidate_timing_gains=tuple(float(item.timing_path_gain) for item in candidates),
        candidate_tau0=tuple(float(item.best_tau0) for item in candidates),
        candidate_slope=tuple(float(item.best_slope) for item in candidates),
    )


__all__ = [
    "TimingPathCandidate",
    "TimingPathDemodResult",
    "demod_timing_path_symbol",
    "score_fixed_timing_path_candidates",
    "score_timing_path_candidates",
]
