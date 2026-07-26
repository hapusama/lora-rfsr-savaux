"""UniChirp paper-style dual-peak demodulation baseline.

This module implements the symbol-level mechanism described in:

    "UniChirp: Unwrapping In-Chirp Phase Misalignment for Weak LoRa
    Signal Demodulation."

The paper models the phase jump between the two dechirped chirp segments as
a packet-local linear trend, then coherently combines the two oversampled FFT
peaks:

    X_agg[k] = X[k] + X[k + (M - N)] * exp(-j * phi_hat[n])

where N=2**SF and M=N*OSR.  The implementation below keeps the algorithm
FFT-bin-only: it uses preamble/header observations to fit the phase trend, and
does not consult payload bytes, CRC, templates, or ground-truth payload bins.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np

from ...chirp import bin_to_grlora_symbol, signed_fft_bin
from ..savaux_oversampled.paper_oversampled_demod import _oversampled_downchirp


CfoCorrectionMode = Literal["none", "symbol", "continuous"]


@dataclass(frozen=True)
class UniChirpDemodConfig:
    """Runtime knobs for the paper-style UniChirp metric."""

    cfo_correction_mode: CfoCorrectionMode = "continuous"
    origin_shift_samples: int | None = None
    enable_bandlimit_filter: bool = True
    filter_bandwidth_scale: float = 1.0
    min_dual_peak_ratio: float = 1e-3
    robust_fit_max_residual_rad: float = 0.75 * math.pi
    min_fit_observations: int = 2


@dataclass(frozen=True)
class UniChirpTrainingSymbol:
    """Known non-payload symbol used to estimate the phase jump trend."""

    start_sample: int
    raw_fft_bin: int
    abs_symbol_index: float
    source: str = "training"


@dataclass(frozen=True)
class UniChirpPhaseObservation:
    """One measured dual-peak phase offset for phase-model fitting."""

    abs_symbol_index: float
    raw_fft_bin: int
    phase_rad: float
    primary_power: float
    secondary_power: float
    quality: float
    source: str


@dataclass(frozen=True)
class UniChirpPhaseModel:
    """Linear packet-local phase-jump model, phi(i)=slope*i+intercept."""

    slope_rad_per_symbol: float
    intercept_rad: float
    observation_count: int
    rmse_rad: float
    source: str

    def predict(self, abs_symbol_index: float) -> float:
        return float(self.slope_rad_per_symbol * float(abs_symbol_index) + self.intercept_rad)


@dataclass(frozen=True)
class UniChirpDemodResult:
    """Single-symbol UniChirp demodulation result."""

    raw_fft_bin: int
    signed_fft_bin: int
    symbol_value: int
    peak_real: float
    peak_imag: float
    peak_amp: float
    peak_power: float
    peak_phase: float
    peak_margin_db: float
    total_power: float
    phase_rad: float
    primary_bin: int
    secondary_bin: int
    primary_power: float
    secondary_power: float
    metric: np.ndarray
    combined_spectrum: np.ndarray
    primary_spectrum: np.ndarray
    secondary_spectrum: np.ndarray
    full_spectrum: np.ndarray
    os_factor: int
    cfo_correction_mode: str


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


def _origin_shift(os_factor: int, config: UniChirpDemodConfig | None) -> int:
    cfg = config or UniChirpDemodConfig()
    os_value = _validate_os_factor(os_factor)
    if cfg.origin_shift_samples is None:
        return int(os_value // 2)
    shift = int(cfg.origin_shift_samples)
    if shift < 0 or shift >= os_value:
        raise ValueError(f"origin_shift_samples must be in [0, {os_value}), got {shift}")
    return shift


def _wrap_phase_rad(value: float) -> float:
    return float(np.angle(np.exp(1j * float(value))))


def _bandlimit_symbol(symbol: np.ndarray, os_factor: int, bandwidth_scale: float) -> np.ndarray:
    """Apply the paper's band-limited prefilter to one symbol window."""

    os_value = _validate_os_factor(os_factor)
    scale = float(bandwidth_scale)
    if os_value <= 1 or scale <= 0.0:
        return np.asarray(symbol, dtype=np.complex64)
    values = np.asarray(symbol, dtype=np.complex64)
    freqs = np.fft.fftfreq(values.size)
    cutoff = min(0.5, 0.5 * scale / float(os_value))
    spectrum = np.fft.fft(values)
    spectrum[np.abs(freqs) > cutoff] = 0.0
    return np.fft.ifft(spectrum).astype(np.complex64)


def _extract_corrected_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
    header_start_sample: int | None,
    config: UniChirpDemodConfig | None,
) -> np.ndarray:
    cfg = config or UniChirpDemodConfig()
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    n_samples = n_bins * os_value
    start = int(start_sample) + _origin_shift(os_value, cfg)
    stop = start + n_samples
    source = np.asarray(samples, dtype=np.complex64)
    if start < 0 or stop > source.size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")

    symbol = np.asarray(source[start:stop], dtype=np.complex64)
    if bool(cfg.enable_bandlimit_filter):
        symbol = _bandlimit_symbol(symbol, os_factor=os_value, bandwidth_scale=float(cfg.filter_bandwidth_scale))

    mode = _validate_cfo_mode(cfg.cfo_correction_mode)
    use_cfo_int = int(cfo_int) if mode in {"symbol", "continuous"} else 0
    use_cfo_frac = float(cfo_frac) if mode in {"symbol", "continuous"} else 0.0
    if mode == "continuous":
        if header_start_sample is None:
            raise ValueError("header_start_sample is required for continuous CFO correction")
        shifted_header = int(header_start_sample) + _origin_shift(os_value, cfg)
        cfo_total = float(cfo_int) + float(cfo_frac)
        relative_chip_start = float(start - shifted_header) / float(os_value)
        common_phase = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        symbol = (symbol * np.exp(-1j * common_phase)).astype(np.complex64)

    downchirp = _oversampled_downchirp(
        sf=int(sf),
        os_factor=os_value,
        cfo_int=use_cfo_int,
        cfo_frac=use_cfo_frac,
    )
    return (symbol * downchirp).astype(np.complex64)


def unichirp_full_spectrum(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    config: UniChirpDemodConfig | None = None,
) -> np.ndarray:
    """Return the full oversampled FFT spectrum after dechirping."""

    dechirped = _extract_corrected_symbol(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        config=config,
    )
    return (np.fft.fft(dechirped) / math.sqrt(float(dechirped.size))).astype(np.complex64)


def observe_unichirp_phase(
    samples: np.ndarray,
    symbol: UniChirpTrainingSymbol,
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    config: UniChirpDemodConfig | None = None,
) -> UniChirpPhaseObservation | None:
    """Measure one primary-vs-secondary phase offset for a known symbol."""

    cfg = config or UniChirpDemodConfig()
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    n_samples = n_bins * os_value
    raw_bin = int(symbol.raw_fft_bin) % n_bins
    secondary_bin = int(raw_bin + n_samples - n_bins)
    spectrum = unichirp_full_spectrum(
        samples=samples,
        start_sample=int(symbol.start_sample),
        sf=int(sf),
        os_factor=os_value,
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        config=cfg,
    )
    primary = complex(spectrum[raw_bin])
    secondary = complex(spectrum[secondary_bin])
    primary_power = float(abs(primary) ** 2)
    secondary_power = float(abs(secondary) ** 2)
    larger = max(primary_power, secondary_power)
    smaller = min(primary_power, secondary_power)
    if larger <= 0.0 or smaller / (larger + 1e-30) < float(cfg.min_dual_peak_ratio):
        return None
    quality = float(2.0 * math.sqrt(primary_power * secondary_power) / (primary_power + secondary_power + 1e-30))
    phase = float(np.angle(secondary * np.conj(primary)))
    return UniChirpPhaseObservation(
        abs_symbol_index=float(symbol.abs_symbol_index),
        raw_fft_bin=raw_bin,
        phase_rad=phase,
        primary_power=primary_power,
        secondary_power=secondary_power,
        quality=quality,
        source=str(symbol.source),
    )


def _fit_phase_line(
    observations: Sequence[UniChirpPhaseObservation],
    max_residual_rad: float,
    min_fit_observations: int,
) -> tuple[float, float, int, float]:
    usable = [item for item in observations if float(item.quality) > 0.0]
    if not usable:
        return 0.0, 0.0, 0, 0.0
    usable.sort(key=lambda item: float(item.abs_symbol_index))
    if len(usable) == 1 or len(usable) < int(min_fit_observations):
        return 0.0, float(usable[0].phase_rad), len(usable), 0.0

    def solve(items: Sequence[UniChirpPhaseObservation]) -> tuple[float, float, float]:
        x = np.asarray([float(item.abs_symbol_index) for item in items], dtype=np.float64)
        y = np.unwrap(np.asarray([float(item.phase_rad) for item in items], dtype=np.float64))
        w = np.asarray([max(1e-6, float(item.quality)) for item in items], dtype=np.float64)
        x0 = float(np.average(x, weights=w))
        y0 = float(np.average(y, weights=w))
        dx = x - x0
        dy = y - y0
        denom = float(np.sum(w * dx * dx))
        slope = 0.0 if denom <= 1e-30 else float(np.sum(w * dx * dy) / denom)
        intercept = float(y0 - slope * x0)
        residual = np.asarray([_wrap_phase_rad(item.phase_rad - (slope * item.abs_symbol_index + intercept)) for item in items])
        rmse = float(math.sqrt(float(np.mean(residual * residual)))) if residual.size else 0.0
        return slope, intercept, rmse

    slope, intercept, _rmse = solve(usable)
    residuals = [
        abs(_wrap_phase_rad(float(item.phase_rad) - (slope * float(item.abs_symbol_index) + intercept)))
        for item in usable
    ]
    kept = [
        item
        for item, residual in zip(usable, residuals)
        if float(residual) <= float(max_residual_rad)
    ]
    if len(kept) >= max(2, int(min_fit_observations)) and len(kept) < len(usable):
        slope, intercept, _rmse = solve(kept)
        usable = kept
    residual = np.asarray(
        [
            _wrap_phase_rad(float(item.phase_rad) - (slope * float(item.abs_symbol_index) + intercept))
            for item in usable
        ],
        dtype=np.float64,
    )
    rmse = float(math.sqrt(float(np.mean(residual * residual)))) if residual.size else 0.0
    return float(slope), float(intercept), int(len(usable)), rmse


def build_unichirp_phase_model(
    samples: np.ndarray,
    training_symbols: Sequence[UniChirpTrainingSymbol],
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    config: UniChirpDemodConfig | None = None,
) -> tuple[UniChirpPhaseModel, tuple[UniChirpPhaseObservation, ...]]:
    """Fit the preamble/header-guided linear phase compensation model."""

    cfg = config or UniChirpDemodConfig()
    observations: list[UniChirpPhaseObservation] = []
    for symbol in training_symbols:
        try:
            observation = observe_unichirp_phase(
                samples=samples,
                symbol=symbol,
                sf=int(sf),
                os_factor=int(os_factor),
                cfo_int=int(cfo_int),
                cfo_frac=float(cfo_frac),
                header_start_sample=header_start_sample,
                config=cfg,
            )
        except ValueError:
            observation = None
        if observation is not None:
            observations.append(observation)

    slope, intercept, count, rmse = _fit_phase_line(
        observations,
        max_residual_rad=float(cfg.robust_fit_max_residual_rad),
        min_fit_observations=int(cfg.min_fit_observations),
    )
    if observations:
        sources = "+".join(sorted({item.source for item in observations}))
    else:
        sources = "zero"
    return (
        UniChirpPhaseModel(
            slope_rad_per_symbol=float(slope),
            intercept_rad=float(intercept),
            observation_count=int(count),
            rmse_rad=float(rmse),
            source=sources,
        ),
        tuple(observations),
    )


def unichirp_metric_from_spectrum(
    full_spectrum: np.ndarray,
    sf: int,
    os_factor: int,
    phase_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute UniChirp metric from one full oversampled FFT spectrum."""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    n_samples = n_bins * os_value
    spectrum = np.asarray(full_spectrum, dtype=np.complex64)
    if spectrum.size != n_samples:
        raise ValueError(f"full_spectrum has {spectrum.size} bins, expected {n_samples}")
    primary_bins = np.arange(n_bins, dtype=np.int64)
    secondary_bins = primary_bins + int(n_samples - n_bins)
    primary = np.asarray(spectrum[primary_bins], dtype=np.complex64)
    secondary = np.asarray(spectrum[secondary_bins], dtype=np.complex64)
    combined = primary + secondary * np.exp(-1j * float(phase_rad))
    metric = np.abs(combined).astype(np.float64) ** 2
    return metric.astype(np.float64), combined.astype(np.complex64), primary, secondary


def unichirp_metric(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    phase_rad: float,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    config: UniChirpDemodConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute UniChirp scores for every raw FFT bin at one symbol."""

    full = unichirp_full_spectrum(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        config=config,
    )
    metric, combined, primary, secondary = unichirp_metric_from_spectrum(
        full,
        sf=int(sf),
        os_factor=int(os_factor),
        phase_rad=float(phase_rad),
    )
    return metric, combined, primary, secondary, full


def demod_unichirp_symbol(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    phase_rad: float,
    is_header: bool = False,
    ldro: bool = False,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    config: UniChirpDemodConfig | None = None,
) -> UniChirpDemodResult:
    """Demodulate one LoRa symbol using UniChirp dual-peak fusion."""

    cfg = config or UniChirpDemodConfig()
    metric, combined, primary, secondary, full = unichirp_metric(
        samples=samples,
        start_sample=int(start_sample),
        sf=int(sf),
        os_factor=int(os_factor),
        phase_rad=float(phase_rad),
        cfo_int=int(cfo_int),
        cfo_frac=float(cfo_frac),
        header_start_sample=header_start_sample,
        config=cfg,
    )
    raw_bin = int(np.argmax(metric))
    peak = complex(combined[raw_bin])
    peak_power = float(metric[raw_bin])
    second_power = float(np.partition(metric, -2)[-2]) if metric.size > 1 else 0.0
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    secondary_bin = int(raw_bin + n_bins * (os_value - 1))
    symbol_value = bin_to_grlora_symbol(raw_bin, sf=int(sf), is_header=bool(is_header), ldro=bool(ldro))
    return UniChirpDemodResult(
        raw_fft_bin=raw_bin,
        signed_fft_bin=signed_fft_bin(raw_bin, n_bins),
        symbol_value=int(symbol_value),
        peak_real=float(peak.real),
        peak_imag=float(peak.imag),
        peak_amp=float(abs(peak)),
        peak_power=peak_power,
        peak_phase=float(math.atan2(peak.imag, peak.real)),
        peak_margin_db=float(10.0 * math.log10((peak_power + 1e-30) / (second_power + 1e-30))),
        total_power=float(np.sum(metric, dtype=np.float64)),
        phase_rad=float(phase_rad),
        primary_bin=raw_bin,
        secondary_bin=secondary_bin,
        primary_power=float(abs(complex(primary[raw_bin])) ** 2),
        secondary_power=float(abs(complex(secondary[raw_bin])) ** 2),
        metric=np.asarray(metric, dtype=np.float64),
        combined_spectrum=np.asarray(combined, dtype=np.complex64),
        primary_spectrum=np.asarray(primary, dtype=np.complex64),
        secondary_spectrum=np.asarray(secondary, dtype=np.complex64),
        full_spectrum=np.asarray(full, dtype=np.complex64),
        os_factor=os_value,
        cfo_correction_mode=str(cfg.cfo_correction_mode),
    )
