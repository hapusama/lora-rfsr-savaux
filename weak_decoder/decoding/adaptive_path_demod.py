"""Adaptive structured oversampling-path evidence for LoRa symbols.

Savaux's OSR demodulator coherently combines all fixed oversampling branches.
This module explores a stricter form of the user's non-uniform-path idea:
for a candidate raw FFT bin k, compensate every oversampled sample to the
candidate's common phase frame, then search for a *smooth* offset path q[p].

The dynamic program deliberately penalizes frequent offset changes.  It is not
free 4^N enumeration, and it is only used as secondary evidence over a small
Savaux Top-K candidate set.
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
class AdaptivePathCandidate:
    raw_fft_bin: int
    savaux_power: float
    fixed_best_power: float
    adaptive_path_power: float
    adaptive_path_gain: float
    switch_rate: float
    mean_offset: float
    dp_projection_score: float
    composite_score: float


@dataclass(frozen=True)
class AdaptivePathDemodResult:
    raw_fft_bin: int
    signed_fft_bin: int
    symbol_value: int
    savaux_raw_fft_bin: int
    selected_by_path_override: bool
    candidate_bins: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    candidate_savaux_powers: tuple[float, ...]
    candidate_path_powers: tuple[float, ...]
    candidate_path_gains: tuple[float, ...]
    candidate_switch_rates: tuple[float, ...]
    selected_path_offsets: tuple[int, ...]


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
    stop = start + n_bins * os_value
    if start < 0 or stop > np.asarray(samples).size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")

    mode = str(cfo_correction_mode)
    if mode not in {"none", "symbol", "continuous"}:
        raise ValueError(f"unknown CFO correction mode: {cfo_correction_mode}")
    symbol = np.asarray(samples[start:stop], dtype=np.complex64)
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
    return (symbol * downchirp).astype(np.complex64)


def candidate_phase_matrix(
    dechirped: np.ndarray,
    candidate_bin: int,
    sf: int,
    os_factor: int,
) -> np.ndarray:
    """Return z[p, q] compensated into candidate-bin common phase."""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    k = int(candidate_bin)
    p = np.arange(n_bins, dtype=np.float64)[:, None]
    q = np.arange(os_value, dtype=np.float64)[None, :]
    indexes = os_value * np.arange(n_bins, dtype=np.int64)[:, None] + np.arange(
        os_value,
        dtype=np.int64,
    )[None, :]
    picked = np.asarray(dechirped[indexes], dtype=np.complex64)
    kernel = np.exp(-2j * np.pi * float(k) * p / float(n_bins))
    tail = np.ones((n_bins, os_value), dtype=np.complex128)
    if k != 0:
        tail[p[:, 0] >= float(n_bins - k), :] = np.exp(2j * np.pi * q / float(os_value))
    branch_weight = np.exp(-2j * np.pi * q * float(k) / float(n_bins * os_value))
    return (picked * kernel * tail * branch_weight / math.sqrt(float(n_bins))).astype(np.complex64)


def _circular_distance(a: int, b: int, os_factor: int) -> int:
    delta = abs(int(a) - int(b)) % int(os_factor)
    return int(min(delta, int(os_factor) - delta))


def _transition_penalty_matrix(os_factor: int, switch_cost: float, step_cost: float) -> np.ndarray:
    os_value = _validate_os_factor(os_factor)
    prev = np.arange(os_value, dtype=np.int64)[:, None]
    cur = np.arange(os_value, dtype=np.int64)[None, :]
    delta = np.abs(prev - cur) % os_value
    circular = np.minimum(delta, os_value - delta).astype(np.float64)
    switch = (prev != cur).astype(np.float64)
    return switch * float(switch_cost) + circular * float(step_cost)


def _best_smooth_path(
    z: np.ndarray,
    switch_penalty: float,
    step_penalty: float,
) -> tuple[np.ndarray, float, float, complex]:
    """Find a smooth path in q[p] with dynamic programming."""

    mat = np.asarray(z, dtype=np.complex64)
    n_bins, os_value = mat.shape
    reference = complex(np.sum(mat, dtype=np.complex128))
    if abs(reference) <= 1e-30:
        fixed_values = np.sum(mat, axis=0, dtype=np.complex128)
        reference = complex(fixed_values[int(np.argmax(np.abs(fixed_values)))])
    unit = np.exp(-1j * math.atan2(reference.imag, reference.real))
    projection = np.real(mat.astype(np.complex128) * unit).astype(np.float64)
    scale = float(np.percentile(np.abs(projection), 75)) + 1e-12
    switch_cost = float(switch_penalty) * scale
    step_cost = float(step_penalty) * scale
    transition_penalty = _transition_penalty_matrix(os_value, switch_cost, step_cost)

    dp = np.empty((n_bins, os_value), dtype=np.float64)
    back = np.zeros((n_bins, os_value), dtype=np.int16)
    dp[0, :] = projection[0, :]
    for p_idx in range(1, n_bins):
        prev = dp[p_idx - 1]
        scores = prev[:, None] - transition_penalty
        best_prev = np.argmax(scores, axis=0)
        dp[p_idx, :] = projection[p_idx, :] + scores[best_prev, np.arange(os_value)]
        back[p_idx, :] = best_prev.astype(np.int16)

    offsets = np.empty(n_bins, dtype=np.int64)
    offsets[-1] = int(np.argmax(dp[-1]))
    for p_idx in range(n_bins - 1, 0, -1):
        offsets[p_idx - 1] = int(back[p_idx, offsets[p_idx]])
    selected = mat[np.arange(n_bins, dtype=np.int64), offsets]
    coherent = complex(np.sum(selected, dtype=np.complex128))
    switch_rate = float(np.mean(offsets[1:] != offsets[:-1])) if n_bins > 1 else 0.0
    return offsets, float(np.max(dp[-1])), switch_rate, coherent


def score_adaptive_path_candidates(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    candidate_bins: Sequence[int],
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "none",
    switch_penalty: float = 0.40,
    step_penalty: float = 0.10,
    path_gain_power: float = 0.18,
    switch_penalty_power: float = 0.40,
    savaux_power: np.ndarray | None = None,
) -> tuple[tuple[AdaptivePathCandidate, ...], tuple[np.ndarray, ...]]:
    """Score candidate bins with smooth non-uniform path evidence."""

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
    candidates: list[AdaptivePathCandidate] = []
    paths: list[np.ndarray] = []
    power_lookup = None if savaux_power is None else np.asarray(savaux_power, dtype=np.float64)
    for raw_bin in candidate_bins:
        bin_i = int(raw_bin)
        z = candidate_phase_matrix(
            dechirped=dechirped,
            candidate_bin=bin_i,
            sf=sf,
            os_factor=os_value,
        )
        fixed_values = np.sum(z, axis=0, dtype=np.complex128)
        fixed_power = np.abs(fixed_values).astype(np.float64) ** 2
        fixed_best_power = float(np.max(fixed_power)) if fixed_power.size else 0.0
        offsets, dp_score, switch_rate, coherent = _best_smooth_path(
            z,
            switch_penalty=float(switch_penalty),
            step_penalty=float(step_penalty),
        )
        path_power = float(abs(coherent) ** 2)
        path_gain = float(path_power / (fixed_best_power + 1e-30))
        if power_lookup is None:
            savaux_bin_power = float(abs(np.sum(z, dtype=np.complex128)) ** 2)
        else:
            savaux_bin_power = float(power_lookup[bin_i])
        smooth_factor = max(0.05, 1.0 - float(switch_rate))
        composite = float(
            savaux_bin_power
            * max(path_gain, 1e-12) ** float(path_gain_power)
            * smooth_factor ** float(switch_penalty_power)
        )
        candidates.append(
            AdaptivePathCandidate(
                raw_fft_bin=bin_i,
                savaux_power=savaux_bin_power,
                fixed_best_power=fixed_best_power,
                adaptive_path_power=path_power,
                adaptive_path_gain=path_gain,
                switch_rate=float(switch_rate),
                mean_offset=float(np.mean(offsets)) if offsets.size else 0.0,
                dp_projection_score=float(dp_score),
                composite_score=composite,
            )
        )
        paths.append(offsets)
    return tuple(candidates), tuple(paths)


def demod_adaptive_path_symbol(
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
    switch_penalty: float = 0.40,
    step_penalty: float = 0.10,
    path_gain_power: float = 0.18,
    switch_penalty_power: float = 0.40,
    override_margin_db: float = 0.15,
    min_savaux_rel_db: float = -4.0,
    min_path_gain: float = 1.05,
    max_switch_rate: float = 0.25,
) -> AdaptivePathDemodResult:
    """Demodulate one symbol using Savaux Top-K plus adaptive smooth paths."""

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

    candidates, paths = score_adaptive_path_candidates(
        samples=samples,
        start_sample=int(start_sample),
        sf=sf,
        os_factor=os_value,
        candidate_bins=tuple(int(v) for v in candidate_bins),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
        switch_penalty=float(switch_penalty),
        step_penalty=float(step_penalty),
        path_gain_power=float(path_gain_power),
        switch_penalty_power=float(switch_penalty_power),
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
        if float(item.adaptive_path_gain) < float(min_path_gain):
            continue
        if float(item.switch_rate) > float(max_switch_rate):
            continue
        if float(item.composite_score) > best_score * override_margin:
            best_idx = int(idx)
            best_score = float(item.composite_score)

    raw_bin = int(candidates[best_idx].raw_fft_bin)
    symbol_value = bin_to_grlora_symbol(raw_bin, sf=sf, is_header=is_header, ldro=ldro)
    return AdaptivePathDemodResult(
        raw_fft_bin=raw_bin,
        signed_fft_bin=signed_fft_bin(raw_bin, n_bins),
        symbol_value=int(symbol_value),
        savaux_raw_fft_bin=savaux_bin,
        selected_by_path_override=bool(raw_bin != savaux_bin),
        candidate_bins=tuple(int(item.raw_fft_bin) for item in candidates),
        candidate_scores=tuple(float(item.composite_score) for item in candidates),
        candidate_savaux_powers=tuple(float(item.savaux_power) for item in candidates),
        candidate_path_powers=tuple(float(item.adaptive_path_power) for item in candidates),
        candidate_path_gains=tuple(float(item.adaptive_path_gain) for item in candidates),
        candidate_switch_rates=tuple(float(item.switch_rate) for item in candidates),
        selected_path_offsets=tuple(int(v) for v in paths[best_idx]),
    )


__all__ = [
    "AdaptivePathCandidate",
    "AdaptivePathDemodResult",
    "candidate_phase_matrix",
    "demod_adaptive_path_symbol",
    "score_adaptive_path_candidates",
]
