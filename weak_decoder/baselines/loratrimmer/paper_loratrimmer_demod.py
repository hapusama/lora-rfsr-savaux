"""LoRaTrimmer paper-style symbol demodulation baseline.

This module implements the core metric from:

    Jialuo Du, Yunhao Liu, Yidong Ren, Li Liu, and Zhichao Cao.
    "LoRaTrimmer: Optimal Energy Condensation with Chirp Trimming for
    LoRa Weak Signal Decoding." ACM MobiCom 2024.

The paper's Sec. 3.1 trims the dechirped symbol differently for every
candidate initial frequency. For candidate bin k, the symbol is split at
the wrap time t_k = (1 - k / 2**SF) * T. The first segment is projected on
frequency k, the second segment is projected on k - B, and the decision
statistic is the non-coherent sum:

    |X1_k|**2 + |X2_k|**2.

The authors' public prototype expresses the same operation as two dense
matrices, named dataE1/dataE2 in their main.py. The functions below keep
that matrix form for auditability, but also expose a simple symbol-level
API that matches the local weak_decoder baseline style.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal

import numpy as np

from ...chirp import build_upchirp, bin_to_grlora_symbol, signed_fft_bin


CfoCorrectionMode = Literal["none", "symbol", "continuous"]


@dataclass(frozen=True)
class LoRaTrimmerMatrices:
    """Precomputed LoRaTrimmer projection matrices for one SF/OSR pair."""

    sf: int
    os_factor: int
    n_bins: int
    n_samples: int
    split_samples: np.ndarray
    front_matrix: np.ndarray
    tail_matrix: np.ndarray


@dataclass(frozen=True)
class LoRaTrimmerDemodResult:
    """Single-symbol result for the LoRaTrimmer baseline."""

    raw_fft_bin: int
    signed_fft_bin: int
    symbol_value: int
    peak_front_real: float
    peak_front_imag: float
    peak_tail_real: float
    peak_tail_imag: float
    peak_front_power: float
    peak_tail_power: float
    peak_power: float
    peak_margin_db: float
    total_power: float
    split_sample: int
    metric: np.ndarray
    front_projection: np.ndarray
    tail_projection: np.ndarray
    os_factor: int
    cfo_correction_mode: str
    cfo_common_phase_rad: float


def _validate_sf(sf: int) -> int:
    value = int(sf)
    if value <= 0:
        raise ValueError(f"sf must be positive, got {sf}")
    return value


def _validate_os_factor(os_factor: int) -> int:
    value = int(os_factor)
    if value <= 0:
        raise ValueError(f"os_factor must be positive, got {os_factor}")
    return value


def _validate_cfo_mode(mode: CfoCorrectionMode | str) -> str:
    value = str(mode)
    if value not in {"none", "symbol", "continuous"}:
        raise ValueError(f"unknown CFO correction mode: {mode}")
    return value


@lru_cache(maxsize=32)
def build_loratrimmer_matrices(sf: int, os_factor: int = 1) -> LoRaTrimmerMatrices:
    """Build the two paper/prototype projection matrices.

    ``front_matrix[k]`` contains the downchirp samples used before the
    frequency wrap of candidate bin ``k``. ``tail_matrix[k]`` contains the
    wrapped tail samples. Multiplying them by the received symbol computes
    the paper's X1_k and X2_k values.

    The indexing mirrors the authors' public ``gen_constants`` routine:
    ``time_shift = k / N * M`` and ``time_split = M - time_shift``, where
    N is ``2**SF`` and M is ``N * os_factor``.
    """

    sf_value = _validate_sf(sf)
    os_value = _validate_os_factor(os_factor)
    n_bins = 1 << sf_value
    n_samples = n_bins * os_value
    downchirp = np.conjugate(build_upchirp(sf=sf_value, symbol_id=0, os_factor=os_value))

    front = np.zeros((n_bins, n_samples), dtype=np.complex64)
    tail = np.zeros((n_bins, n_samples), dtype=np.complex64)
    split_samples = np.empty(n_bins, dtype=np.int64)
    for raw_bin in range(n_bins):
        time_shift = int(raw_bin * n_samples / n_bins)
        time_split = int(n_samples - time_shift)
        split_samples[raw_bin] = time_split
        front[raw_bin, :time_split] = downchirp[time_shift:]
        if raw_bin != 0:
            tail[raw_bin, time_split:] = downchirp[:time_shift]

    return LoRaTrimmerMatrices(
        sf=sf_value,
        os_factor=os_value,
        n_bins=n_bins,
        n_samples=n_samples,
        split_samples=split_samples,
        front_matrix=front,
        tail_matrix=tail,
    )


def _extract_symbol(
    samples: np.ndarray,
    start_sample: int,
    matrices: LoRaTrimmerMatrices,
) -> np.ndarray:
    start = int(start_sample)
    stop = start + int(matrices.n_samples)
    source = np.asarray(samples, dtype=np.complex64)
    if start < 0 or stop > source.size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")
    return np.asarray(source[start:stop], dtype=np.complex64)


def _apply_cfo_correction(
    symbol: np.ndarray,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    start_sample: int,
    header_start_sample: int | None,
    cfo_correction_mode: CfoCorrectionMode | str,
) -> tuple[np.ndarray, float]:
    """Apply the same optional CFO convention used by local baselines."""

    mode = _validate_cfo_mode(cfo_correction_mode)
    if mode == "none":
        return np.asarray(symbol, dtype=np.complex64), 0.0

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    n = np.arange(np.asarray(symbol).size, dtype=np.float64)
    cfo_total = float(cfo_int) + float(cfo_frac)

    # Local frame-sync helpers model integer and fractional CFO as an
    # additional tone over the symbol duration. Removing it before the
    # LoRaTrimmer projection lets this baseline be compared on synchronized
    # captures without changing the paper metric itself.
    corrected = np.asarray(symbol, dtype=np.complex64) * np.exp(
        -2j * np.pi * cfo_total * n / float(n_bins * os_value)
    )

    common_phase = 0.0
    if mode == "continuous":
        if header_start_sample is None:
            raise ValueError("header_start_sample is required for continuous CFO correction")
        relative_chip_start = float(int(start_sample) - int(header_start_sample)) / float(os_value)
        common_phase = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        corrected = corrected * np.exp(-1j * common_phase)

    return corrected.astype(np.complex64), common_phase


def loratrimmer_metric_from_symbol(
    symbol_samples: np.ndarray,
    sf: int,
    os_factor: int = 1,
    return_projections: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray, np.ndarray, LoRaTrimmerMatrices]
):
    """Compute the LoRaTrimmer metric for one symbol-length array."""

    matrices = build_loratrimmer_matrices(sf=int(sf), os_factor=int(os_factor))
    symbol = np.asarray(symbol_samples, dtype=np.complex64)
    if symbol.size != matrices.n_samples:
        raise ValueError(f"symbol has {symbol.size} samples, expected {matrices.n_samples}")

    front = matrices.front_matrix @ symbol
    tail = matrices.tail_matrix @ symbol
    metric = (
        np.abs(front).astype(np.float64) ** 2
        + np.abs(tail).astype(np.float64) ** 2
    )
    if return_projections:
        return metric.astype(np.float64), front.astype(np.complex64), tail.astype(np.complex64), matrices
    return metric.astype(np.float64)


def loratrimmer_metric(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int = 1,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "none",
    return_projections: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray, np.ndarray, LoRaTrimmerMatrices, float]
):
    """Compute LoRaTrimmer scores for every raw FFT bin at one symbol."""

    matrices = build_loratrimmer_matrices(sf=int(sf), os_factor=int(os_factor))
    symbol = _extract_symbol(samples, start_sample=start_sample, matrices=matrices)
    corrected, cfo_common_phase_rad = _apply_cfo_correction(
        symbol=symbol,
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        start_sample=int(start_sample),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
    )
    metric, front, tail, _matrices = loratrimmer_metric_from_symbol(
        corrected,
        sf=int(sf),
        os_factor=int(os_factor),
        return_projections=True,
    )
    if return_projections:
        return metric, front, tail, matrices, float(cfo_common_phase_rad)
    return metric


def demod_loratrimmer_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int = 1,
    is_header: bool = False,
    ldro: bool = False,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "none",
) -> LoRaTrimmerDemodResult:
    """Demodulate one LoRa symbol using the LoRaTrimmer metric."""

    metric, front, tail, matrices, cfo_common_phase_rad = loratrimmer_metric(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        cfo_correction_mode=cfo_correction_mode,
        return_projections=True,
    )

    raw_bin = int(np.argmax(metric))
    peak_power = float(metric[raw_bin])
    second_power = float(np.partition(metric, -2)[-2]) if metric.size > 1 else 0.0
    symbol_value = bin_to_grlora_symbol(
        raw_bin,
        sf=int(sf),
        is_header=bool(is_header),
        ldro=bool(ldro),
    )

    peak_front = complex(front[raw_bin])
    peak_tail = complex(tail[raw_bin])
    return LoRaTrimmerDemodResult(
        raw_fft_bin=raw_bin,
        signed_fft_bin=signed_fft_bin(raw_bin, 1 << int(sf)),
        symbol_value=int(symbol_value),
        peak_front_real=float(peak_front.real),
        peak_front_imag=float(peak_front.imag),
        peak_tail_real=float(peak_tail.real),
        peak_tail_imag=float(peak_tail.imag),
        peak_front_power=float(abs(peak_front) ** 2),
        peak_tail_power=float(abs(peak_tail) ** 2),
        peak_power=peak_power,
        peak_margin_db=float(10.0 * math.log10((peak_power + 1e-30) / (second_power + 1e-30))),
        total_power=float(np.sum(metric, dtype=np.float64)),
        split_sample=int(matrices.split_samples[raw_bin]),
        metric=np.asarray(metric, dtype=np.float64),
        front_projection=np.asarray(front, dtype=np.complex64),
        tail_projection=np.asarray(tail, dtype=np.complex64),
        os_factor=int(os_factor),
        cfo_correction_mode=str(cfo_correction_mode),
        cfo_common_phase_rad=float(cfo_common_phase_rad),
    )
