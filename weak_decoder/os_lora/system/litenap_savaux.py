"""Wrap-aware sub-Nyquist LoRa detection using Savaux polyphase diversity.

LiteNap obtains an aliased FFT bin from one sample every ``D`` Nyquist chips
and resolves the missing frequency chunk from transmitter phase fingerprints.
This module adds a deterministic path that uses the ``R`` oversampling phases
available in an SDR capture to separate the alias candidates before the
optional phase-fingerprint reranker.

For ``K == D`` downsampling phases, the component sum is the original Savaux
oversampled spectrum, up to floating-point summation order. Smaller ``K``
values genuinely discard samples and expose the accuracy/complexity tradeoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PhaseFingerprintRerankResult:
    """Phase-jump reranking diagnostics for one aliased candidate family."""

    raw_fft_bin: int
    alias_bin: int
    candidate_bins: tuple[int, ...]
    spectral_log_scores: tuple[float, ...]
    phase_jump_scores: tuple[float, ...]
    combined_scores: tuple[float, ...]


def _validate_configuration(
    sf: int,
    os_factor: int,
    downsample_factor: int,
) -> tuple[int, int, int]:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    factor = int(downsample_factor)
    if os_value <= 0:
        raise ValueError(f"os_factor must be positive, got {os_factor}")
    if factor <= 0 or n_bins % factor:
        raise ValueError(
            f"downsample_factor must be a positive divisor of {n_bins}, got {downsample_factor}"
        )
    return n_bins, os_value, factor


def _normalize_phases(
    phases: Sequence[int] | None,
    size: int,
    name: str,
) -> tuple[int, ...]:
    values = tuple(range(int(size))) if phases is None else tuple(int(value) for value in phases)
    if not values:
        raise ValueError(f"{name} must contain at least one phase")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate phases: {values}")
    if any(value < 0 or value >= int(size) for value in values):
        raise ValueError(f"{name} must be within [0, {int(size)}), got {values}")
    return values


def choose_downsample_phases(
    os_factor: int,
    downsample_factor: int,
    view_count: int,
) -> tuple[int, ...]:
    """Choose phases that minimize worst-case simplified alias coherence."""

    os_value = int(os_factor)
    factor = int(downsample_factor)
    count = int(view_count)
    if os_value <= 0 or factor <= 0:
        raise ValueError("os_factor and downsample_factor must be positive")
    if count <= 0 or count > factor:
        raise ValueError(f"view_count must be within [1, {factor}], got {view_count}")

    best: tuple[int, ...] | None = None
    best_coherence = float("inf")
    q = np.arange(os_value, dtype=np.float64) / float(os_value)
    for candidate in combinations(range(factor), count):
        worst = 0.0
        positions = np.concatenate(
            [float(phase) + q for phase in candidate]
        )
        for alias_chunk in range(1, factor):
            steering = np.exp(
                2j * np.pi * float(alias_chunk) * positions / float(factor)
            )
            worst = max(worst, float(abs(np.mean(steering))))
        if worst < best_coherence - 1e-12:
            best = tuple(int(value) for value in candidate)
            best_coherence = worst
    if best is None:
        raise RuntimeError("failed to select downsampling phases")
    return best


@lru_cache(maxsize=4)
def _component_matched_filter_bank(
    sf: int,
    os_factor: int,
    downsample_factor: int,
    wrap_correction: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one ``N x (N/D)`` matched-filter matrix per ``(q, d)``."""

    n_bins, os_value, factor = _validate_configuration(
        sf, os_factor, downsample_factor
    )
    samples_per_component = n_bins // factor
    candidates = np.arange(n_bins, dtype=np.float64)[:, None]
    component_matrices: list[np.ndarray] = []
    component_indexes: list[np.ndarray] = []

    for q in range(os_value):
        fractional_q = float(q) / float(os_value)
        for d in range(factor):
            p_int = d + factor * np.arange(samples_per_component, dtype=np.int64)
            p = p_int.astype(np.float64)[None, :]
            sample_time = p + fractional_q
            matrix = np.exp(
                -2j * np.pi * candidates * sample_time / float(n_bins)
            )
            if bool(wrap_correction) and q != 0:
                tail = (candidates > 0.0) & (p >= (float(n_bins) - candidates))
                matrix *= np.where(
                    tail,
                    np.exp(2j * np.pi * float(q) / float(os_value)),
                    1.0,
                )
            matrix = (matrix / math.sqrt(float(n_bins))).astype(np.complex64)
            component_matrices.append(matrix)
            component_indexes.append(os_value * p_int + q)

    matrices = np.stack(component_matrices).reshape(
        os_value, factor, n_bins, samples_per_component
    )
    indexes = np.stack(component_indexes).reshape(
        os_value, factor, samples_per_component
    )
    matrices.setflags(write=False)
    indexes.setflags(write=False)
    return matrices, indexes


def subnyquist_component_spectra(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    downsample_factor: int,
    *,
    wrap_correction: bool = True,
) -> np.ndarray:
    """Return candidate spectra with shape ``(R, D, N)``.

    Component ``[q, d]`` uses the samples ``R * (D*m + d) + q``. The matched
    filters include the Savaux fractional-sample steering and, by default, the
    candidate-dependent phase correction after the LoRa frequency wrap.
    """

    n_bins, os_value, factor = _validate_configuration(
        sf, os_factor, downsample_factor
    )
    symbol = np.asarray(dechirped, dtype=np.complex64)
    if symbol.ndim != 1 or symbol.size != n_bins * os_value:
        raise ValueError(
            f"dechirped symbol has {symbol.size} samples, expected {n_bins * os_value}"
        )
    matrices, indexes = _component_matched_filter_bank(
        int(sf), os_value, factor, bool(wrap_correction)
    )
    observations = symbol[indexes]
    spectra = np.matmul(matrices, observations[..., None])[..., 0]
    return np.asarray(spectra, dtype=np.complex64)


def subnyquist_component_spectra_batch(
    dechirped_symbols: np.ndarray,
    sf: int,
    os_factor: int,
    downsample_factor: int,
    *,
    wrap_correction: bool = True,
) -> np.ndarray:
    """Batch version returning spectra with shape ``(B, R, D, N)``."""

    n_bins, os_value, factor = _validate_configuration(
        sf, os_factor, downsample_factor
    )
    symbols = np.asarray(dechirped_symbols, dtype=np.complex64)
    if symbols.ndim != 2 or symbols.shape[1] != n_bins * os_value:
        raise ValueError(
            "dechirped_symbols must have shape "
            f"(batch, {n_bins * os_value}), got {symbols.shape}"
        )
    matrices, indexes = _component_matched_filter_bank(
        int(sf), os_value, factor, bool(wrap_correction)
    )
    observations = symbols[:, indexes]
    output = np.empty(
        (symbols.shape[0], os_value, factor, n_bins), dtype=np.complex64
    )
    for q in range(os_value):
        for d in range(factor):
            output[:, q, d, :] = (
                observations[:, q, d, :] @ matrices[q, d].T
            ).astype(np.complex64)
    return output


def combine_subnyquist_components(
    component_spectra: np.ndarray,
    *,
    branch_phases: Sequence[int] | None = None,
    downsample_phases: Sequence[int] | None = None,
) -> np.ndarray:
    """Coherently combine a selected set of Savaux and downsampling phases."""

    components = np.asarray(component_spectra, dtype=np.complex64)
    if components.ndim != 3:
        raise ValueError("component_spectra must have shape (R, D, N)")
    os_value, factor, n_bins = components.shape
    q_values = _normalize_phases(branch_phases, os_value, "branch_phases")
    d_values = _normalize_phases(
        downsample_phases, factor, "downsample_phases"
    )
    combined = np.zeros(n_bins, dtype=np.complex128)
    for q in q_values:
        for d in d_values:
            combined += components[q, d].astype(np.complex128)
    return combined.astype(np.complex64)


def litenap_savaux_spectrum(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    downsample_factor: int,
    *,
    branch_phases: Sequence[int] | None = None,
    downsample_phases: Sequence[int] | None = None,
    wrap_correction: bool = True,
) -> np.ndarray:
    """Compute one wrap-aware LiteNap-Savaux candidate spectrum."""

    components = subnyquist_component_spectra(
        dechirped=dechirped,
        sf=sf,
        os_factor=os_factor,
        downsample_factor=downsample_factor,
        wrap_correction=wrap_correction,
    )
    return combine_subnyquist_components(
        components,
        branch_phases=branch_phases,
        downsample_phases=downsample_phases,
    )


def selected_sample_count(
    sf: int,
    os_factor: int,
    downsample_factor: int,
    *,
    branch_phases: Sequence[int] | None = None,
    downsample_phases: Sequence[int] | None = None,
) -> int:
    """Return the number of physical IQ samples used by a detector."""

    n_bins, os_value, factor = _validate_configuration(
        sf, os_factor, downsample_factor
    )
    q_values = _normalize_phases(branch_phases, os_value, "branch_phases")
    d_values = _normalize_phases(
        downsample_phases, factor, "downsample_phases"
    )
    return int(len(q_values) * len(d_values) * (n_bins // factor))


def _selected_layout(
    sf: int,
    os_factor: int,
    downsample_factor: int,
    branch_phases: Sequence[int],
    downsample_phases: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_bins, os_value, factor = _validate_configuration(
        sf, os_factor, downsample_factor
    )
    samples_per_component = n_bins // factor
    raw_indexes: list[int] = []
    chip_indexes: list[int] = []
    branch_indexes: list[int] = []
    for q in branch_phases:
        for d in downsample_phases:
            p = d + factor * np.arange(samples_per_component, dtype=np.int64)
            raw_indexes.extend((os_value * p + int(q)).tolist())
            chip_indexes.extend(p.tolist())
            branch_indexes.extend([int(q)] * samples_per_component)
    order = np.argsort(np.asarray(raw_indexes, dtype=np.int64))
    return (
        np.asarray(raw_indexes, dtype=np.int64)[order],
        np.asarray(chip_indexes, dtype=np.int64)[order],
        np.asarray(branch_indexes, dtype=np.int64)[order],
    )


def candidate_phase_jump_scores(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    downsample_factor: int,
    candidate_bins: Sequence[int],
    *,
    branch_phases: Sequence[int] | None = None,
    downsample_phases: Sequence[int] | None = None,
    min_segment_samples: int = 8,
    wrap_correction: bool = True,
) -> np.ndarray:
    """Score a persistent phase jump at each candidate's predicted wrap time.

    The statistic is a two-segment complex-mean F-like ratio. It is intended
    only as a LiteNap hardware-fingerprint diagnostic; it does not use payload
    ground truth or clean-symbol templates.
    """

    n_bins, os_value, factor = _validate_configuration(
        sf, os_factor, downsample_factor
    )
    q_values = _normalize_phases(branch_phases, os_value, "branch_phases")
    d_values = _normalize_phases(
        downsample_phases, factor, "downsample_phases"
    )
    symbol = np.asarray(dechirped, dtype=np.complex64)
    if symbol.ndim != 1 or symbol.size != n_bins * os_value:
        raise ValueError("dechirped symbol length mismatch")
    raw, p, q = _selected_layout(
        sf, os_value, factor, q_values, d_values
    )
    observations = symbol[raw].astype(np.complex128)
    scores = np.zeros(len(tuple(candidate_bins)), dtype=np.float64)

    for output_index, raw_bin in enumerate(candidate_bins):
        k = int(raw_bin) % n_bins
        if k == 0:
            continue
        before = p < (n_bins - k)
        after = ~before
        n_before = int(np.count_nonzero(before))
        n_after = int(np.count_nonzero(after))
        if n_before < int(min_segment_samples) or n_after < int(min_segment_samples):
            continue
        time = p.astype(np.float64) + q.astype(np.float64) / float(os_value)
        weight = np.exp(-2j * np.pi * float(k) * time / float(n_bins))
        if bool(wrap_correction):
            weight[after] *= np.exp(
                2j * np.pi * q[after].astype(np.float64) / float(os_value)
            )
        residual = observations * weight
        mean_before = complex(np.mean(residual[before]))
        mean_after = complex(np.mean(residual[after]))
        within = float(
            np.sum(np.abs(residual[before] - mean_before) ** 2)
            + np.sum(np.abs(residual[after] - mean_after) ** 2)
        )
        between = (
            float(n_before * n_after)
            / float(n_before + n_after)
            * abs(mean_before - mean_after) ** 2
        )
        noise_scale = within / float(max(1, n_before + n_after - 2))
        scores[output_index] = float(between / (noise_scale + 1e-30))
    return scores


def rerank_with_phase_fingerprint(
    spectral_power: np.ndarray,
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    downsample_factor: int,
    *,
    branch_phases: Sequence[int] | None = None,
    downsample_phases: Sequence[int] | None = None,
    fingerprint_weight: float = 0.25,
    min_segment_samples: int = 8,
    wrap_correction: bool = True,
) -> PhaseFingerprintRerankResult:
    """Rerank the ``D`` candidates in the strongest aliased FFT family."""

    n_bins, os_value, factor = _validate_configuration(
        sf, os_factor, downsample_factor
    )
    power = np.asarray(spectral_power, dtype=np.float64)
    if power.ndim != 1 or power.size != n_bins:
        raise ValueError(f"spectral_power must contain {n_bins} candidates")
    if float(fingerprint_weight) < 0.0:
        raise ValueError("fingerprint_weight must be non-negative")

    samples_per_component = n_bins // factor
    folded = np.sum(power.reshape(factor, samples_per_component), axis=0)
    alias_bin = int(np.argmax(folded))
    candidates = tuple(
        int(alias_bin + chunk * samples_per_component) for chunk in range(factor)
    )
    jump_scores = candidate_phase_jump_scores(
        dechirped=dechirped,
        sf=sf,
        os_factor=os_value,
        downsample_factor=factor,
        candidate_bins=candidates,
        branch_phases=branch_phases,
        downsample_phases=downsample_phases,
        min_segment_samples=min_segment_samples,
        wrap_correction=wrap_correction,
    )
    candidate_power = np.asarray([power[index] for index in candidates], dtype=np.float64)
    scale = max(float(np.max(candidate_power)), 1e-30)
    spectral_log = np.log((candidate_power + 1e-30) / scale)
    combined = spectral_log + float(fingerprint_weight) * np.log1p(jump_scores)
    selected = int(candidates[int(np.argmax(combined))])
    return PhaseFingerprintRerankResult(
        raw_fft_bin=selected,
        alias_bin=alias_bin,
        candidate_bins=candidates,
        spectral_log_scores=tuple(float(value) for value in spectral_log),
        phase_jump_scores=tuple(float(value) for value in jump_scores),
        combined_scores=tuple(float(value) for value in combined),
    )


__all__ = [
    "PhaseFingerprintRerankResult",
    "candidate_phase_jump_scores",
    "choose_downsample_phases",
    "combine_subnyquist_components",
    "litenap_savaux_spectrum",
    "rerank_with_phase_fingerprint",
    "selected_sample_count",
    "subnyquist_component_spectra",
    "subnyquist_component_spectra_batch",
]
