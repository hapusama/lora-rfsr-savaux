"""结构化非均匀过采样路径解调实验。

本模块刻意把非均匀偏移路径作为 Savaux OSR baseline 固定偏移 branch 之外的
实验扩展。对于路径 ``q[p]`` 和候选 raw FFT bin ``k``，这里使用与论文相同的
branch 相位模型，但只沿选定的采样路径计算：

    n[p] = R * p + q[p]

路径分数只作为 Savaux Top-K 小候选集中的额外证据。这并不意味着任意选择的
逐 chip 路径能够产生相互独立的 LoRa 观测。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
class StructuredPath:
    name: str
    offsets: np.ndarray


@dataclass(frozen=True)
class StructuredPathCandidate:
    raw_fft_bin: int
    savaux_power: float
    path_power: float
    path_ratio: float
    best_path_name: str
    composite_score: float


@dataclass(frozen=True)
class StructuredPathDemodResult:
    raw_fft_bin: int
    signed_fft_bin: int
    symbol_value: int
    savaux_raw_fft_bin: int
    score: float
    savaux_power: float
    path_power: float
    path_ratio: float
    best_path_name: str
    candidate_bins: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    candidate_savaux_powers: tuple[float, ...]
    candidate_path_powers: tuple[float, ...]
    candidate_path_ratios: tuple[float, ...]
    selected_by_path_override: bool


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


@lru_cache(maxsize=32)
def structured_paths(sf: int, os_factor: int) -> tuple[StructuredPath, ...]:
    """生成一组紧凑的结构化偏移路径。

    针对 OSR=4，路径集合被刻意限制为低复杂度形式：固定偏移、模线性路径、
    两种短周期路径以及两段常值路径；重复路径会被删除。
    """

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    p = np.arange(n_bins, dtype=np.int64)
    out: list[StructuredPath] = []
    seen: set[bytes] = set()

    def add(name: str, offsets: np.ndarray) -> None:
        vec = np.asarray(offsets, dtype=np.int16)
        if vec.size != n_bins:
            raise ValueError("path length mismatch")
        vec = np.mod(vec, os_value).astype(np.int16, copy=False)
        key = vec.tobytes()
        if key in seen:
            return
        seen.add(key)
        out.append(StructuredPath(name=name, offsets=vec.astype(np.int64)))

    for b in range(os_value):
        add(f"fixed_b{b}", np.full(n_bins, b, dtype=np.int64))

    for a in range(os_value):
        for b in range(os_value):
            add(f"linear_a{a}_b{b}", a * p + b)

    for q0 in range(os_value):
        for q1 in range(os_value):
            add(f"period2_{q0}{q1}", np.where((p % 2) == 0, q0, q1))

    for split_num, split_den in ((1, 4), (1, 2), (3, 4)):
        split = int(round(n_bins * split_num / split_den))
        for q0 in range(os_value):
            for q1 in range(os_value):
                offsets = np.empty(n_bins, dtype=np.int64)
                offsets[:split] = q0
                offsets[split:] = q1
                add(f"piece{split_num}of{split_den}_{q0}{q1}", offsets)

    return tuple(out)


def _prepare_dechirped_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    header_start_sample: int | None,
    cfo_correction_mode: CfoCorrectionMode,
) -> tuple[np.ndarray, float]:
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    start = int(start_sample)
    stop = start + n_bins * os_value
    if start < 0 or stop > np.asarray(samples).size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")

    mode = str(cfo_correction_mode)
    if mode not in {"none", "symbol", "continuous"}:
        raise ValueError(f"unknown CFO correction mode: {cfo_correction_mode}")
    symbol = np.asarray(samples[start:stop], dtype=np.complex64)
    cfo_common_phase_rad = 0.0
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
    return (symbol * downchirp).astype(np.complex64), cfo_common_phase_rad


def _path_candidate_value(
    dechirped: np.ndarray,
    candidate_bin: int,
    path_offsets: np.ndarray,
    sf: int,
    os_factor: int,
) -> complex:
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    k = int(candidate_bin)
    p = np.arange(n_bins, dtype=np.float64)
    q = np.asarray(path_offsets, dtype=np.int64)
    if q.size != n_bins:
        raise ValueError("path length mismatch")
    indexes = os_value * np.arange(n_bins, dtype=np.int64) + q
    picked = np.asarray(dechirped[indexes], dtype=np.complex64)

    kernel = np.exp(-2j * np.pi * float(k) * p / float(n_bins))
    if k != 0:
        tail = p >= float(n_bins - k)
        phase_tail = np.ones(n_bins, dtype=np.complex128)
        phase_tail[tail] = np.exp(2j * np.pi * q[tail].astype(np.float64) / float(os_value))
    else:
        phase_tail = 1.0
    branch_weight = np.exp(
        -2j * np.pi * q.astype(np.float64) * float(k) / float(n_bins * os_value)
    )
    return complex(np.sum(picked * kernel * phase_tail * branch_weight) / math.sqrt(float(n_bins)))


def _structured_path_scores(
    dechirped: np.ndarray,
    candidate_bins: Sequence[int],
    paths: Sequence[StructuredPath],
    sf: int,
    os_factor: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    powers: list[float] = []
    ratios: list[float] = []
    best_names: list[str] = []
    for raw_bin in candidate_bins:
        values = np.asarray(
            [
                _path_candidate_value(
                    dechirped=dechirped,
                    candidate_bin=int(raw_bin),
                    path_offsets=path.offsets,
                    sf=sf,
                    os_factor=os_factor,
                )
                for path in paths
            ],
            dtype=np.complex64,
        )
        path_power = np.abs(values).astype(np.float64) ** 2
        best_idx = int(np.argmax(path_power))
        mean_power = float(np.mean(path_power)) if path_power.size else 0.0
        powers.append(float(path_power[best_idx]))
        ratios.append(float(path_power[best_idx] / (mean_power + 1e-30)))
        best_names.append(paths[best_idx].name if paths else "")
    return (
        np.asarray(powers, dtype=np.float64),
        np.asarray(ratios, dtype=np.float64),
        tuple(best_names),
    )


def score_structured_path_candidates(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    candidate_bins: Sequence[int],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "none",
    path_ratio_power: float = 0.20,
    savaux_power: np.ndarray | None = None,
) -> tuple[StructuredPathCandidate, ...]:
    """使用紧凑的结构化偏移路径集合为 Savaux Top-K bin 评分。

    这是 structured-path 实验的软证据版本。函数本身不选择 hard symbol，
    而是返回每个候选的证据，供调用方融合进已有的似然或 codec 解码器。
    """

    os_value = _validate_os_factor(os_factor)
    dechirped, _common_phase = _prepare_dechirped_symbol(
        samples=samples,
        start_sample=int(start_sample),
        sf=sf,
        os_factor=os_value,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    bins = tuple(int(v) for v in candidate_bins)
    paths = structured_paths(sf, os_value)
    path_powers, path_ratios, best_path_names = _structured_path_scores(
        dechirped=dechirped,
        candidate_bins=bins,
        paths=paths,
        sf=sf,
        os_factor=os_value,
    )
    power_lookup = None if savaux_power is None else np.asarray(savaux_power, dtype=np.float64)
    out: list[StructuredPathCandidate] = []
    for idx, raw_bin in enumerate(bins):
        savaux_bin_power = (
            float(power_lookup[int(raw_bin)])
            if power_lookup is not None
            else float(path_powers[idx])
        )
        ratio = float(path_ratios[idx])
        composite = float(
            savaux_bin_power
            * max(ratio, 1e-12) ** float(path_ratio_power)
        )
        out.append(
            StructuredPathCandidate(
                raw_fft_bin=int(raw_bin),
                savaux_power=savaux_bin_power,
                path_power=float(path_powers[idx]),
                path_ratio=ratio,
                best_path_name=str(best_path_names[idx]),
                composite_score=composite,
            )
        )
    return tuple(out)


def demod_structured_path_symbol(
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
    path_ratio_power: float = 0.20,
    override_margin_db: float = 0.20,
    min_savaux_rel_db: float = -4.0,
) -> StructuredPathDemodResult:
    """使用 Savaux Top-K 上的结构化路径证据解调一个 symbol。"""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    savaux_spectrum, _branches, _cfo_phase = paper_oversampled_spectrum(
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

    dechirped, _common_phase = _prepare_dechirped_symbol(
        samples=samples,
        start_sample=int(start_sample),
        sf=sf,
        os_factor=os_value,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    paths = structured_paths(sf, os_value)
    path_powers, path_ratios, best_path_names = _structured_path_scores(
        dechirped=dechirped,
        candidate_bins=tuple(int(v) for v in candidate_bins),
        paths=paths,
        sf=sf,
        os_factor=os_value,
    )
    candidate_savaux_powers = savaux_power_all[candidate_bins]
    max_savaux = float(np.max(candidate_savaux_powers)) if candidate_savaux_powers.size else 0.0
    savaux_rel = candidate_savaux_powers / (max_savaux + 1e-30)
    score = candidate_savaux_powers * np.power(np.maximum(path_ratios, 1e-12), float(path_ratio_power))

    savaux_candidate_index = int(np.where(candidate_bins == savaux_bin)[0][0])
    best_index = int(savaux_candidate_index)
    best_score = float(score[best_index])
    override_margin = 10.0 ** (float(override_margin_db) / 10.0)
    min_rel = 10.0 ** (float(min_savaux_rel_db) / 10.0)
    for idx, raw_bin in enumerate(candidate_bins):
        if int(raw_bin) == savaux_bin:
            continue
        if float(savaux_rel[idx]) < min_rel:
            continue
        if float(score[idx]) > best_score * override_margin:
            best_index = int(idx)
            best_score = float(score[idx])

    raw_bin = int(candidate_bins[best_index])
    symbol_value = bin_to_grlora_symbol(raw_bin, sf=sf, is_header=is_header, ldro=ldro)
    return StructuredPathDemodResult(
        raw_fft_bin=raw_bin,
        signed_fft_bin=signed_fft_bin(raw_bin, n_bins),
        symbol_value=int(symbol_value),
        savaux_raw_fft_bin=savaux_bin,
        score=float(score[best_index]),
        savaux_power=float(candidate_savaux_powers[best_index]),
        path_power=float(path_powers[best_index]),
        path_ratio=float(path_ratios[best_index]),
        best_path_name=best_path_names[best_index],
        candidate_bins=tuple(int(v) for v in candidate_bins),
        candidate_scores=tuple(float(v) for v in score),
        candidate_savaux_powers=tuple(float(v) for v in candidate_savaux_powers),
        candidate_path_powers=tuple(float(v) for v in path_powers),
        candidate_path_ratios=tuple(float(v) for v in path_ratios),
        selected_by_path_override=bool(raw_bin != savaux_bin),
    )


__all__ = [
    "StructuredPath",
    "StructuredPathCandidate",
    "StructuredPathDemodResult",
    "demod_structured_path_symbol",
    "score_structured_path_candidates",
    "structured_paths",
]
