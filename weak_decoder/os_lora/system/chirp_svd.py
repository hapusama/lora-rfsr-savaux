"""过采样 LoRa symbol 的 ChirpSVD 系统实现。

核心对象是将一个已经 dechirp 的过采样 symbol 重排为 ``N x R`` 矩阵：

    X[p, q] = z[p * R + q]

理想 LoRa tone 完成 dechirp 后，该矩阵具有明显的低秩结构。本模块提供两种
保守的利用方式：

* 将首个左奇异向量作为 chip-time tone 评分；
* 重建 rank-r 去噪矩阵，再在去噪后的 branch 上运行已有 Savaux 过采样频谱。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from ...baselines.savaux_oversampled.paper_oversampled_demod import (
    _paper_branch_spectrum,
    combine_paper_branch_spectra,
)


@dataclass(frozen=True)
class ChirpSVDSpectra:
    """一个 dechirped 过采样 symbol 完成 SVD 后得到的频谱集合。"""

    singular_values: tuple[float, ...]
    rank1_ratio: float
    rank2_ratio: float
    svd_left_spectrum: np.ndarray
    rank_savaux_spectra: dict[int, np.ndarray]


def _validate_shape(dechirped: np.ndarray, sf: int, os_factor: int) -> tuple[np.ndarray, int, int]:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    if os_value <= 0:
        raise ValueError(f"os_factor must be positive, got {os_factor}")
    symbol = np.asarray(dechirped, dtype=np.complex64)
    expected = n_bins * os_value
    if symbol.size != expected:
        raise ValueError(f"dechirped symbol has {symbol.size} samples, expected {expected}")
    return symbol, n_bins, os_value


def chirp_matrix(dechirped: np.ndarray, sf: int, os_factor: int) -> np.ndarray:
    """返回一个 dechirped symbol 对应的 ``N x R`` ChirpSVD 矩阵。"""

    symbol, n_bins, os_value = _validate_shape(dechirped, sf, os_factor)
    return np.asarray(symbol.reshape(n_bins, os_value), dtype=np.complex128)


def low_rank_chirp_matrix(matrix: np.ndarray, rank: int) -> np.ndarray:
    """返回 ``matrix`` 的 rank-r 截断 SVD 重建结果。"""

    mat = np.asarray(matrix, dtype=np.complex128)
    if mat.ndim != 2:
        raise ValueError("matrix must be 2-D")
    max_rank = min(mat.shape)
    keep = max(1, min(int(rank), max_rank))
    u, s, vh = np.linalg.svd(mat, full_matrices=False)
    return np.asarray((u[:, :keep] * s[:keep]) @ vh[:keep, :], dtype=np.complex128)


def savaux_spectrum_from_dechirped_matrix(matrix: np.ndarray, sf: int, os_factor: int) -> np.ndarray:
    """在已经 dechirp 的矩阵上运行现有 Savaux branch 频谱计算。"""

    mat = np.asarray(matrix, dtype=np.complex128)
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    if mat.shape != (n_bins, os_value):
        raise ValueError(f"matrix shape {mat.shape} does not match {(n_bins, os_value)}")
    branch_spectra = tuple(
        _paper_branch_spectrum(
            dechirped_branch=np.asarray(mat[:, q], dtype=np.complex64),
            sf=int(sf),
            os_factor=os_value,
            branch_index=q,
        )
        for q in range(os_value)
    )
    return combine_paper_branch_spectra(branch_spectra, os_factor=os_value).astype(np.complex64)


def chirp_svd_spectra(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    ranks: Sequence[int] = (1, 2),
) -> ChirpSVDSpectra:
    """计算一个 dechirped 过采样 symbol 的 ChirpSVD 频谱。"""

    matrix = chirp_matrix(dechirped, sf, os_factor)
    n_bins = 1 << int(sf)
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    singular = np.asarray(s, dtype=np.float64)
    total_energy = float(np.sum(singular ** 2))
    rank1_ratio = float((singular[0] ** 2) / max(total_energy, 1e-30)) if singular.size else 0.0
    rank2_ratio = (
        float(np.sum(singular[:2] ** 2) / max(total_energy, 1e-30))
        if singular.size
        else 0.0
    )

    if singular.size:
        # u[:, 0] 是单位范数，因此用 sigma_1 恢复尺度；该归一化在不改变
        # argmax 的前提下，使不同 SF 的频谱尺度可比较。
        svd_left = (singular[0] * np.fft.fft(u[:, 0]) / math.sqrt(float(n_bins))).astype(np.complex64)
    else:
        svd_left = np.zeros(n_bins, dtype=np.complex64)

    rank_spectra: dict[int, np.ndarray] = {}
    for rank in ranks:
        keep = max(1, min(int(rank), min(matrix.shape)))
        reconstructed = np.asarray((u[:, :keep] * s[:keep]) @ vh[:keep, :], dtype=np.complex128)
        rank_spectra[int(rank)] = savaux_spectrum_from_dechirped_matrix(
            reconstructed,
            sf=int(sf),
            os_factor=int(os_factor),
        )

    return ChirpSVDSpectra(
        singular_values=tuple(float(v) for v in singular),
        rank1_ratio=rank1_ratio,
        rank2_ratio=rank2_ratio,
        svd_left_spectrum=svd_left,
        rank_savaux_spectra=rank_spectra,
    )


__all__ = [
    "ChirpSVDSpectra",
    "chirp_matrix",
    "chirp_svd_spectra",
    "low_rank_chirp_matrix",
    "savaux_spectrum_from_dechirped_matrix",
]
