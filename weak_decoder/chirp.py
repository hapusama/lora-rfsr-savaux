"""LoRa chirp and FFT helpers matching gr-lora_sdr's normalized formula."""

from __future__ import annotations

import numpy as np


def positive_mod(value: int, modulus: int) -> int:
    """Return value modulo modulus in [0, modulus)."""
    return int((int(value) % int(modulus) + int(modulus)) % int(modulus))


def signed_fft_bin(bin_index: int, fft_len: int) -> int:
    """把循环 FFT bin 转成带符号 bin，便于观察峰值相对 bin0 的偏移。"""

    value = positive_mod(int(bin_index), int(fft_len))
    half = int(fft_len) // 2
    return int(value - int(fft_len) if value > half else value)


def build_upchirp(sf: int, symbol_id: int = 0, os_factor: int = 1) -> np.ndarray:
    """Build one gr-lora_sdr-compatible upchirp.

    The formula mirrors include/gnuradio/lora_sdr/utilities.h::build_upchirp.
    ``symbol_id=0`` is the reference upchirp; other ids are cyclic shifts in
    LoRa bin/chip coordinates.
    """
    sf = int(sf)
    os_factor = int(os_factor)
    n_bins = 1 << sf
    symbol_id = positive_mod(symbol_id, n_bins)
    n = np.arange(n_bins * os_factor, dtype=np.float64)
    fold = n_bins * os_factor - symbol_id * os_factor
    slope = n * n / (2.0 * n_bins) / (os_factor * os_factor)
    linear_before = (symbol_id / n_bins - 0.5) * n / os_factor
    linear_after = (symbol_id / n_bins - 1.5) * n / os_factor
    phase = np.where(n < fold, slope + linear_before, slope + linear_after)
    return np.exp(2j * np.pi * phase).astype(np.complex64)


def build_downchirp(sf: int, cfo_int: int = 0, cfo_frac: float = 0.0) -> np.ndarray:
    """Build the downchirp used by gr-lora_sdr fft_demod for one frame."""
    n_bins = 1 << int(sf)
    n = np.arange(n_bins, dtype=np.float64)
    upchirp = build_upchirp(sf, positive_mod(cfo_int, n_bins), os_factor=1)
    correction = np.exp(-2j * np.pi * float(cfo_frac) * n / n_bins)
    return (np.conjugate(upchirp) * correction).astype(np.complex64)


def dechirp_fft(symbol_samples: np.ndarray, downchirp: np.ndarray) -> np.ndarray:
    """Return complex FFT bins after dechirping one symbol."""
    if symbol_samples.size != downchirp.size:
        raise ValueError(
            f"symbol has {symbol_samples.size} samples, expected {downchirp.size}"
        )
    dechirped = np.asarray(symbol_samples, dtype=np.complex64) * downchirp
    return np.fft.fft(dechirped).astype(np.complex64)


def bin_to_grlora_symbol(bin_index: int, sf: int, is_header: bool = False, ldro: bool = False) -> int:
    """Map an FFT bin index to the hard symbol value used by gr-lora_sdr."""
    n_bins = 1 << int(sf)
    symbol = positive_mod(int(bin_index) - 1, n_bins)
    if is_header or ldro:
        symbol //= 4
    return int(symbol)


def gray_value(value: int) -> int:
    """Return the reflected Gray-coded value used in gr-lora_sdr LLR grouping."""
    value = int(value)
    return int(value ^ (value >> 1))
