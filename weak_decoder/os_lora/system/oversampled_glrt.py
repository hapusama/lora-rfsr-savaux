"""保持 LoRa 结构的过采样 GLRT/GLS 算法库。

该模块保存此前 OS-LoRa 研究形成的可复用核心算法，目前不接入
``RFSR -> FrameSync -> Savaux`` 主链。保留它是为了后续需要时能够在不恢复历史
实验脚本的情况下重新比较 GLS，但当前结果不得宣称使用了这里的增强方法。

算法包含两类观察：

* Savaux polyphase branch 观察。每个候选 bin 只有 ``OSR`` 维，因此噪声模型是
  很小的 ``OSR x OSR`` 协方差，可用普通 GLS 白化；不会构造整符号长度的巨型
  ``RN x RN`` 协方差。
* 完整采样率 dechirp 后相隔 ``Fs-BW`` 的双分量观察。它利用相位对齐后的相干/
  非相干功率关系重排少量候选，属于后续历史消融，不是当前 Savaux 主链的一部分。

本模块刻意不接收 payload ground truth。branch 噪声协方差只能从包外纯噪声窗
估计；频率、相位和折返时序模型只能由前导码、头部等 payload 之前的已知结构拟合。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal, Sequence

import numpy as np

from ...chirp import build_upchirp


CfoCorrectionMode = Literal["none", "symbol", "continuous"]
PairAmplitudeModel = Literal["fold", "equal", "exact"]
RerankMode = Literal["weighted", "confidence_gate"]
CoherentRerankMode = Literal["joint", "coherence", "confidence_gate"]


# ---------------------------------------------------------------------------
# Savaux branch 观察与低维噪声协方差
# ---------------------------------------------------------------------------


def _validate_os_factor(os_factor: int) -> int:
    value = int(os_factor)
    if value <= 0:
        raise ValueError(f"os_factor must be positive, got {os_factor}")
    return value


def _validate_windows(noise_windows: np.ndarray, symbol_samples: int) -> np.ndarray:
    """把噪声样本规范为“窗口数 x 单符号采样数”，并要求至少两个快照。"""

    windows = np.asarray(noise_windows, dtype=np.complex64)
    if windows.ndim == 1:
        if windows.size % int(symbol_samples) != 0:
            raise ValueError("flat noise input is not an integer number of symbols")
        windows = windows.reshape(-1, int(symbol_samples))
    if windows.ndim != 2 or windows.shape[1] != int(symbol_samples):
        raise ValueError(
            f"noise_windows must have shape (count, {symbol_samples}), got {windows.shape}"
        )
    if windows.shape[0] < 2:
        raise ValueError("at least two noise windows are required")
    return windows


def _regularized_covariance(
    snapshots: np.ndarray,
    diagonal_loading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """估计 Hermitian 协方差，加入对角加载并返回其伪逆。"""

    values = np.asarray(snapshots, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("snapshots must be a two-dimensional matrix with at least two rows")
    centered = values - np.mean(values, axis=0, keepdims=True)
    # 每一行是一份复数列向量观察 y，因此 C_ij = E[y_i conj(y_j)]。
    covariance = centered.T @ centered.conj() / float(max(1, centered.shape[0] - 1))
    covariance = 0.5 * (covariance + covariance.conj().T)
    dimension = int(covariance.shape[0])
    scale = max(float(np.real(np.trace(covariance))) / float(dimension), 1e-30)
    loading = max(0.0, float(diagonal_loading))
    loaded = covariance + loading * scale * np.eye(dimension, dtype=np.complex128)
    inverse = np.linalg.pinv(loaded, hermitian=True)
    return loaded.astype(np.complex128), inverse.astype(np.complex128)


def _oversampled_downchirp(
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
) -> np.ndarray:
    """生成同时补偿整数和小数 CFO 的完整过采样 downchirp。"""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    length = n_bins * os_value
    n = np.arange(length, dtype=np.float64)
    reference = build_upchirp(sf=int(sf), symbol_id=int(cfo_int), os_factor=os_value)
    fractional = np.exp(-2j * np.pi * float(cfo_frac) * n / float(length))
    return (np.conjugate(reference) * fractional).astype(np.complex64)


@lru_cache(maxsize=64)
def _candidate_dft_kernels(
    sf: int,
    os_factor: int,
    branch_index: int,
    candidate_bins: tuple[int, ...],
) -> np.ndarray:
    """为指定候选生成 Savaux Eq.36 的 branch-specific DFT 行。"""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    q = int(branch_index)
    if not 0 <= q < os_value:
        raise ValueError(f"branch_index must be in [0, {os_value})")
    bins = np.asarray(candidate_bins, dtype=np.int64) % n_bins
    p = np.arange(n_bins, dtype=np.float64)[None, :]
    k = bins.astype(np.float64)[:, None]
    kernel = np.exp(-2j * np.pi * k * p / float(n_bins))
    if q:
        tail = (k > 0.0) & (p >= (float(n_bins) - k))
        kernel = kernel * np.where(
            tail,
            np.exp(2j * np.pi * float(q) / float(os_value)),
            1.0,
        )
    return kernel.astype(np.complex64)


def aligned_branch_observations(
    branch_spectra: Sequence[np.ndarray],
    os_factor: int,
) -> np.ndarray:
    """按 Savaux 确定性相位项对齐 branch，返回候选 ``k`` 的 OSR 维观察。"""

    os_value = _validate_os_factor(os_factor)
    spectra = tuple(np.asarray(item, dtype=np.complex64) for item in branch_spectra)
    if len(spectra) != os_value:
        raise ValueError(f"got {len(spectra)} branch spectra, expected {os_value}")
    if not spectra:
        raise ValueError("at least one branch spectrum is required")
    n_bins = int(spectra[0].size)
    if any(item.ndim != 1 or item.size != n_bins for item in spectra):
        raise ValueError("all branch spectra must be one-dimensional and equally sized")
    raw = np.stack(spectra, axis=1).astype(np.complex128)
    k = np.arange(n_bins, dtype=np.float64)[:, None]
    q = np.arange(os_value, dtype=np.float64)[None, :]
    phase = np.exp(-2j * np.pi * k * q / float(n_bins * os_value))
    return (raw * phase).astype(np.complex64)


@dataclass(frozen=True)
class BranchNoiseModel:
    """Savaux branch 的低维噪声协方差、伪逆和 steering 信息。"""

    covariance: np.ndarray
    inverse_covariance: np.ndarray
    steering: np.ndarray
    information: float | np.ndarray
    snapshot_count: int
    training_bins: tuple[int, ...]
    diagonal_loading: float


def identity_branch_noise_model(os_factor: int) -> BranchNoiseModel:
    """构造白噪声单位模型；在该模型下 branch GLS 退化为普通 Savaux。"""

    os_value = _validate_os_factor(os_factor)
    covariance = np.eye(os_value, dtype=np.complex128)
    steering = np.ones(os_value, dtype=np.complex128)
    return BranchNoiseModel(
        covariance=covariance,
        inverse_covariance=covariance.copy(),
        steering=steering,
        information=float(os_value),
        snapshot_count=0,
        training_bins=tuple(),
        diagonal_loading=0.0,
    )


def estimate_branch_noise_model(
    noise_windows: np.ndarray,
    sf: int,
    os_factor: int,
    training_bins: Sequence[int] | None = None,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    diagonal_loading: float = 0.05,
    covariance_mode: Literal["pooled", "per_bin"] = "pooled",
) -> BranchNoiseModel:
    """从纯噪声窗口估计 pooled 或逐候选 ``OSR x OSR`` branch 协方差。

    ``pooled`` 把多个训练 bin 的快照合并成一个协方差；``per_bin`` 为每个候选
    保存一个小矩阵，形状是 ``(N, OSR, OSR)``。两种模式都不会构造
    ``RN x RN`` 的整符号协方差。
    """

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    length = n_bins * os_value
    windows = _validate_windows(noise_windows, length)
    mode = str(covariance_mode)
    if mode not in {"pooled", "per_bin"}:
        raise ValueError(f"unknown covariance_mode: {covariance_mode}")
    if mode == "per_bin":
        bins = tuple(range(n_bins))
    elif training_bins is None:
        count = min(16, n_bins)
        bins = tuple(int(value) for value in np.linspace(0, n_bins, count, endpoint=False))
    else:
        bins = tuple(dict.fromkeys(int(value) % n_bins for value in training_bins))
    if not bins:
        raise ValueError("training_bins must contain at least one candidate")

    downchirp = _oversampled_downchirp(sf, os_value, cfo_int, cfo_frac)
    dechirped = windows * downchirp[None, :]
    values = np.empty((windows.shape[0], len(bins), os_value), dtype=np.complex64)
    normalization = math.sqrt(float(n_bins))
    bin_array = np.asarray(bins, dtype=np.float64)
    for q in range(os_value):
        kernels = _candidate_dft_kernels(int(sf), os_value, q, bins)
        branch = np.asarray(dechirped[:, q::os_value], dtype=np.complex64)
        projected = np.einsum("kn,wn->wk", kernels, branch, optimize=True) / normalization
        alignment = np.exp(-2j * np.pi * float(q) * bin_array / float(n_bins * os_value))
        values[:, :, q] = projected * alignment[None, :]
    steering = np.ones(os_value, dtype=np.complex128)
    if mode == "pooled":
        snapshots = values.reshape(-1, os_value)
        covariance, inverse = _regularized_covariance(snapshots, diagonal_loading)
        inverse_steering = inverse @ steering
        information: float | np.ndarray = float(
            max(np.real(np.vdot(steering, inverse_steering)), 1e-30)
        )
        snapshot_count = int(snapshots.shape[0])
    else:
        centered = values.astype(np.complex128) - np.mean(
            values.astype(np.complex128), axis=0, keepdims=True
        )
        covariance = np.einsum(
            "wki,wkj->kij", centered, centered.conj(), optimize=True
        ) / float(max(1, centered.shape[0] - 1))
        covariance = 0.5 * (covariance + covariance.conj().transpose(0, 2, 1))
        scale = np.maximum(
            np.real(np.trace(covariance, axis1=1, axis2=2)) / float(os_value),
            1e-30,
        )
        covariance = covariance + max(0.0, float(diagonal_loading)) * scale[:, None, None] * np.eye(
            os_value, dtype=np.complex128
        )[None, :, :]
        inverse = np.linalg.pinv(covariance, hermitian=True)
        inverse_steering = np.einsum("krs,s->kr", inverse, steering, optimize=True)
        information = np.maximum(
            np.real(np.einsum("r,kr->k", steering.conj(), inverse_steering)),
            1e-30,
        )
        snapshot_count = int(values.shape[0])
    return BranchNoiseModel(
        covariance=covariance,
        inverse_covariance=inverse,
        steering=steering,
        information=information,
        snapshot_count=snapshot_count,
        training_bins=bins,
        diagonal_loading=float(diagonal_loading),
    )


@dataclass(frozen=True)
class BranchGLSResult:
    """全部候选的 GLS 分数、最佳 bin 和 Top-L 诊断。"""

    scores: np.ndarray
    selected_bin: int
    top_candidates: tuple[int, ...]
    observations: np.ndarray


@dataclass(frozen=True)
class BranchSteeringEstimate:
    """由包内已知符号估计出的 branch 响应及其可信度。"""

    steering: np.ndarray
    rank_one_fraction: float
    observation_count: int


@dataclass(frozen=True)
class HeaderBinCalibration:
    """整包共享的残余整数 bin 偏移估计。"""

    correction_bins: int
    consensus: float
    observation_count: int
    residual_bins: int


def estimate_header_bin_correction(
    observed_bins: Sequence[int],
    n_bins: int,
    minimum_consensus: float = 0.75,
) -> HeaderBinCalibration:
    """利用显式头 ``raw_bin = 1 (mod 4)`` 约束校准正负一个 bin 的偏差。

    这里只接受头部已知结构，不读取 payload 真值。多个头部符号对同一余数达成
    足够共识时才启用修正，避免单个错误头部 bin 把整包拖偏。
    """

    if int(n_bins) <= 0 or int(n_bins) % 4:
        raise ValueError("n_bins must be a positive multiple of four")
    observed = tuple(int(value) % int(n_bins) for value in observed_bins)
    if not observed:
        return HeaderBinCalibration(0, 0.0, 0, 0)
    residuals = [
        int(((raw_bin - 1 + 2) % 4) - 2)
        for raw_bin in observed
    ]
    counts = {value: residuals.count(value) for value in (-2, -1, 0, 1)}
    mode = max(counts, key=lambda value: (counts[value], -abs(value)))
    consensus = float(counts[mode] / len(residuals))
    correction = (
        -int(mode)
        if abs(int(mode)) == 1 and consensus >= float(minimum_consensus)
        else 0
    )
    return HeaderBinCalibration(correction, consensus, len(residuals), int(mode))


def estimate_branch_steering(
    observations: np.ndarray,
) -> BranchSteeringEstimate:
    """用包内已知符号的秩一 SVD 估计 packet-local branch 响应。

    每个已知符号先按自身范数归一化，避免大能量符号主导估计；随后取第一右奇异
    向量作为 branch steering，并统一其公共相位和范数。``rank_one_fraction``
    表示观察能量中可由一个公共 branch 响应解释的比例。
    """

    values = np.asarray(observations, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("observations must have shape (symbols, branches)")
    row_norm = np.linalg.norm(values, axis=1)
    usable = row_norm > 1e-30
    if not np.any(usable):
        steering = np.ones(values.shape[1], dtype=np.complex128)
        return BranchSteeringEstimate(steering, 0.0, 0)
    normalized = values[usable] / row_norm[usable, None]
    _u, singular, vh = np.linalg.svd(normalized, full_matrices=False)
    steering = np.asarray(vh[0], dtype=np.complex128)
    mean_phase = float(np.angle(np.sum(steering)))
    steering = steering * np.exp(-1j * mean_phase)
    steering = steering * math.sqrt(float(steering.size)) / max(
        float(np.linalg.norm(steering)), 1e-30
    )
    rank_one_fraction = float(
        singular[0] ** 2 / max(float(np.sum(singular**2)), 1e-30)
    )
    return BranchSteeringEstimate(
        steering=steering,
        rank_one_fraction=rank_one_fraction,
        observation_count=int(np.sum(usable)),
    )


def branch_gls_scores(
    branch_spectra: Sequence[np.ndarray],
    os_factor: int,
    noise_model: BranchNoiseModel | None = None,
    top_l: int = 8,
    steering: np.ndarray | None = None,
) -> BranchGLSResult:
    """使用 OSR 维 GLS 统计量给全部 LoRa 候选打分。"""

    # 先应用 Savaux 的确定性 branch 相位对齐，再以 C^-1 对 steering 和观察
    # 做匹配；分母 information 用于消除不同协方差尺度带来的分数偏置。

    observations = aligned_branch_observations(branch_spectra, os_factor=os_factor)
    model = noise_model or identity_branch_noise_model(os_factor)
    active_steering = (
        np.asarray(model.steering, dtype=np.complex128)
        if steering is None
        else np.asarray(steering, dtype=np.complex128)
    )
    if active_steering.shape != (int(os_factor),):
        raise ValueError("branch steering dimension does not match os_factor")
    inverse = np.asarray(model.inverse_covariance, dtype=np.complex128)
    if inverse.ndim == 2:
        if inverse.shape != (int(os_factor), int(os_factor)):
            raise ValueError("branch noise model dimension does not match os_factor")
        inverse_steering = inverse @ active_steering
        information = float(
            max(np.real(np.vdot(active_steering, inverse_steering)), 1e-30)
        )
        projected = observations @ np.conjugate(inverse_steering)
    elif inverse.ndim == 3:
        expected = (observations.shape[0], int(os_factor), int(os_factor))
        if inverse.shape != expected:
            raise ValueError(
                f"candidate-wise branch covariance has shape {inverse.shape}, expected {expected}"
            )
        inverse_steering = np.einsum(
            "krs,s->kr", inverse, active_steering, optimize=True
        )
        information = np.maximum(
            np.real(
                np.einsum(
                    "r,kr->k", active_steering.conj(), inverse_steering, optimize=True
                )
            ),
            1e-30,
        )
        projected = np.einsum(
            "kr,kr->k", np.conjugate(inverse_steering), observations, optimize=True
        )
    else:
        raise ValueError("branch inverse covariance must be 2-D or 3-D")
    scores = np.abs(projected).astype(np.float64) ** 2 / information
    order = np.argsort(scores)[::-1]
    keep = max(1, min(int(top_l), int(scores.size)))
    candidates = tuple(int(value) for value in order[:keep])
    return BranchGLSResult(scores, int(candidates[0]), candidates, observations)


# ---------------------------------------------------------------------------
# 完整采样率双峰历史消融；当前 RFSR + FrameSync + Savaux 主链不调用
# ---------------------------------------------------------------------------


def extract_full_rate_dechirped(
    samples: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    header_start_sample: int | None = None,
    cfo_correction_mode: CfoCorrectionMode = "continuous",
) -> np.ndarray:
    """截取一个完整采样率符号，并按接收机约定补偿 CFO。

    ``symbol`` 模式只修正符号内部的频率斜率；``continuous`` 还根据该符号相对
    header 起点的位置补偿公共相位，因此跨符号相位模型必须使用后者。
    """

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    length = n_bins * os_value
    start = int(start_sample)
    stop = start + length
    source = np.asarray(samples)
    if start < 0 or stop > source.size:
        raise ValueError(f"symbol at {start_sample} exceeds input sample range")
    mode = str(cfo_correction_mode)
    if mode not in {"none", "symbol", "continuous"}:
        raise ValueError(f"unknown CFO correction mode: {cfo_correction_mode}")
    symbol = np.asarray(source[start:stop], dtype=np.complex64)
    use_cfo_int = int(cfo_int) if mode in {"symbol", "continuous"} else 0
    use_cfo_frac = float(cfo_frac) if mode in {"symbol", "continuous"} else 0.0
    if mode == "continuous":
        if header_start_sample is None:
            raise ValueError("header_start_sample is required for continuous CFO correction")
        relative_chip_start = float(start - int(header_start_sample)) / float(os_value)
        cfo_total = float(cfo_int) + float(cfo_frac)
        common_phase = 2.0 * math.pi * cfo_total * relative_chip_start / float(n_bins)
        symbol = (symbol * np.exp(-1j * common_phase)).astype(np.complex64)
    downchirp = _oversampled_downchirp(sf, os_value, use_cfo_int, use_cfo_frac)
    return (symbol * downchirp).astype(np.complex64)


def full_rate_spectrum(dechirped: np.ndarray) -> np.ndarray:
    """计算单位能量归一化的完整采样率 DFT。"""

    values = np.asarray(dechirped, dtype=np.complex64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("dechirped must be a non-empty one-dimensional array")
    return (np.fft.fft(values) / math.sqrt(float(values.size))).astype(np.complex64)


def fractional_dft(dechirped: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    """只在少量小数 bin 位置直接计算归一化 DFT，避免整段零填充 FFT。"""

    values = np.asarray(dechirped, dtype=np.complex64)
    frequencies = np.asarray(tuple(float(value) for value in bins), dtype=np.float64)
    if values.ndim != 1 or frequencies.ndim != 1:
        raise ValueError("dechirped and bins must be one-dimensional")
    n = np.arange(values.size, dtype=np.float64)
    kernel = np.exp(-2j * np.pi * frequencies[:, None] * n[None, :] / float(values.size))
    return (kernel @ values.astype(np.complex128) / math.sqrt(float(values.size))).astype(np.complex64)


def estimate_fractional_peak_offset(
    spectrum: np.ndarray,
    raw_bin: int,
    max_abs_offset: float = 0.5,
) -> tuple[float, float]:
    """用主峰左右三个 bin 的抛物线插值估计残余小数 bin 偏差。"""

    values = np.asarray(spectrum)
    length = int(values.size)
    center = int(raw_bin) % length
    power = np.abs(values[[((center - 1) % length), center, ((center + 1) % length)]]) ** 2
    left, middle, right = (float(item) for item in power)
    denominator = left - 2.0 * middle + right
    offset = 0.0 if abs(denominator) <= 1e-30 else 0.5 * (left - right) / denominator
    limit = max(0.0, float(max_abs_offset))
    return float(np.clip(offset, -limit, limit)), middle


@dataclass(frozen=True)
class LinearFrequencyModel:
    """小数频偏随符号序号线性变化的拟合模型。"""

    slope_bins_per_symbol: float
    intercept_bins: float
    observation_count: int
    rmse_bins: float

    def predict(self, symbol_index: float) -> float:
        return float(self.slope_bins_per_symbol * float(symbol_index) + self.intercept_bins)


@dataclass(frozen=True)
class LinearPhaseModel:
    """双峰相位差随符号序号线性变化的拟合模型。"""

    slope_rad_per_symbol: float
    intercept_rad: float
    observation_count: int
    rmse_rad: float

    def predict(self, symbol_index: float) -> float:
        return float(self.slope_rad_per_symbol * float(symbol_index) + self.intercept_rad)


def _weighted_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float, np.ndarray]:
    """加权最小二乘直线，返回斜率、截距和逐点残差。"""

    safe = np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)
    x0 = float(np.average(x, weights=safe))
    y0 = float(np.average(y, weights=safe))
    dx = x - x0
    denominator = float(np.sum(safe * dx * dx))
    slope = 0.0 if denominator <= 1e-30 else float(np.sum(safe * dx * (y - y0)) / denominator)
    intercept = float(y0 - slope * x0)
    return slope, intercept, y - (slope * x + intercept)


def fit_frequency_line(
    symbol_indices: Sequence[float],
    offsets: Sequence[float],
    weights: Sequence[float] | None = None,
    fixed_slope_bins_per_symbol: float | None = None,
) -> LinearFrequencyModel:
    """拟合频偏轨迹；可固定 SFO 已知时对应的理论斜率。"""

    x = np.asarray(symbol_indices, dtype=np.float64)
    y = np.asarray(offsets, dtype=np.float64)
    if x.size != y.size or x.size == 0:
        raise ValueError("symbol_indices and offsets must have the same non-zero length")
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.size != x.size:
        raise ValueError("weights length does not match observations")
    if fixed_slope_bins_per_symbol is None:
        slope, intercept, residual = _weighted_line(x, y, w)
    else:
        slope = float(fixed_slope_bins_per_symbol)
        safe = np.maximum(w, 1e-9)
        intercept = float(np.average(y - slope * x, weights=safe))
        residual = y - (slope * x + intercept)
    rmse = float(math.sqrt(float(np.average(residual * residual, weights=np.maximum(w, 1e-9)))))
    return LinearFrequencyModel(slope, intercept, int(x.size), rmse)


@dataclass(frozen=True)
class DualPeakObservation:
    """一个已知符号的双峰功率、相位及质量观察。"""

    symbol_index: float
    raw_bin: int
    fractional_offset: float
    phase_rad: float
    primary_power: float
    secondary_power: float
    quality: float


@dataclass(frozen=True)
class KnownDualPeakPair:
    """payload 前一个已知符号的复数折返双峰观察。"""

    symbol_index: float
    raw_bin: int
    fractional_offset: float
    primary: complex
    secondary: complex

    @property
    def pair(self) -> np.ndarray:
        return np.asarray((self.primary, self.secondary), dtype=np.complex128)


@dataclass(frozen=True)
class FoldTimingModel:
    """供精确折返双峰 steering 使用的包内定时轨迹。"""

    reference_symbol_index: float
    offset_chips_at_reference: float
    slope_chips_per_symbol: float
    observation_count: int
    mean_consistency: float
    grid_contrast: float

    def predict(self, symbol_index: float) -> float:
        return float(
            self.offset_chips_at_reference
            + self.slope_chips_per_symbol
            * (float(symbol_index) - self.reference_symbol_index)
        )


def observe_dual_peak(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    raw_bin: int,
    fractional_offset: float,
    symbol_index: float,
) -> DualPeakObservation:
    """在 ``k`` 与 ``k + (R-1)N`` 处观察一个候选的完整采样率双峰。"""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    length = n_bins * os_value
    candidate = int(raw_bin) % n_bins
    locations = (
        float(candidate) + float(fractional_offset),
        float(candidate + length - n_bins) + float(fractional_offset),
    )
    primary, secondary = (complex(value) for value in fractional_dft(dechirped, locations))
    primary_power = float(abs(primary) ** 2)
    secondary_power = float(abs(secondary) ** 2)
    quality = float(2.0 * math.sqrt(primary_power * secondary_power) / (primary_power + secondary_power + 1e-30))
    return DualPeakObservation(
        float(symbol_index),
        candidate,
        float(fractional_offset),
        float(np.angle(secondary * np.conjugate(primary))),
        primary_power,
        secondary_power,
        quality,
    )


def observe_known_dual_peak_pair(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    raw_bin: int,
    fractional_offset: float,
    symbol_index: float,
) -> KnownDualPeakPair:
    """返回已知符号在两个完整采样率频点上的复数分量。"""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    length = n_bins * os_value
    candidate = int(raw_bin) % n_bins
    pair = fractional_dft(
        dechirped,
        (
            float(candidate) + float(fractional_offset),
            float(candidate + length - n_bins) + float(fractional_offset),
        ),
    )
    return KnownDualPeakPair(
        float(symbol_index),
        candidate,
        float(fractional_offset),
        complex(pair[0]),
        complex(pair[1]),
    )


def fit_phase_line(
    observations: Sequence[DualPeakObservation],
    min_quality: float = 0.02,
    max_residual_rad: float = 0.75 * math.pi,
    fixed_slope_rad_per_symbol: float | None = None,
) -> LinearPhaseModel:
    """从 payload 前的已知符号稳健拟合展开后的相位直线。

    先按 ``quality`` 丢弃双峰过弱的观察，再对相位展开并加权拟合；首次拟合后
    会剔除相位残差过大的离群点。若理论斜率已知，则只拟合截距。
    """

    usable = [item for item in observations if float(item.quality) >= float(min_quality)]
    if not usable:
        return LinearPhaseModel(0.0, 0.0, 0, float("inf"))
    usable.sort(key=lambda item: float(item.symbol_index))

    def solve(items: Sequence[DualPeakObservation]) -> tuple[float, float, np.ndarray]:
        x = np.asarray([item.symbol_index for item in items], dtype=np.float64)
        wrapped_y = np.asarray([item.phase_rad for item in items], dtype=np.float64)
        w = np.asarray([max(item.quality, 1e-6) for item in items], dtype=np.float64)
        if fixed_slope_rad_per_symbol is None:
            y = np.unwrap(wrapped_y)
            return _weighted_line(x, y, w)
        slope = float(fixed_slope_rad_per_symbol)
        detrended = wrapped_y - slope * x
        resultant = np.sum(w * np.exp(1j * detrended))
        intercept = float(np.angle(resultant)) if abs(resultant) > 0.0 else 0.0
        residual = np.angle(np.exp(1j * (wrapped_y - (slope * x + intercept))))
        return slope, intercept, residual

    slope, intercept, residual = solve(usable)
    wrapped = np.angle(np.exp(1j * residual))
    kept = [item for item, error in zip(usable, np.abs(wrapped), strict=True) if float(error) <= float(max_residual_rad)]
    if len(kept) >= 2 and len(kept) < len(usable):
        usable = kept
        slope, intercept, residual = solve(usable)
        wrapped = np.angle(np.exp(1j * residual))
    weights = np.asarray([max(item.quality, 1e-6) for item in usable], dtype=np.float64)
    rmse = float(math.sqrt(float(np.average(wrapped * wrapped, weights=weights))))
    return LinearPhaseModel(float(slope), float(intercept), int(len(usable)), rmse)


@dataclass(frozen=True)
class PairNoiseModel:
    """完整采样率双峰的 2x2 噪声协方差模型。"""

    covariance: np.ndarray
    inverse_covariance: np.ndarray
    snapshot_count: int
    diagonal_loading: float
    training_bins: tuple[int, ...] = tuple()


def identity_pair_noise_model() -> PairNoiseModel:
    """构造白噪声下的单位双峰协方差。"""

    covariance = np.eye(2, dtype=np.complex128)
    return PairNoiseModel(covariance, covariance.copy(), 0, 0.0)


def _pair_inverse_for_candidate(
    model: PairNoiseModel,
    raw_bin: int,
) -> np.ndarray:
    inverse = np.asarray(model.inverse_covariance, dtype=np.complex128)
    if inverse.ndim == 2 and inverse.shape == (2, 2):
        return inverse
    if inverse.ndim == 3 and inverse.shape[1:] == (2, 2):
        return inverse[int(raw_bin) % int(inverse.shape[0])]
    raise ValueError("pair inverse covariance must have shape (2, 2) or (N, 2, 2)")


def fold_pair_steering(
    raw_bin: int,
    sf: int,
    os_factor: int,
    timing_offset_chips: float,
) -> np.ndarray:
    """计算两个 dechirp chirp 分段对双峰的精确响应。

    与只按长度写成 ``[N-k, k]`` 的近似不同，这里保留每个有限分段泄漏到另一
    分量的交叉项；折返点和分段相位跳变都随候选 bin、定时偏差变化。公共复数
    比例会在后续 GLRT 中消去，所以这里刻意不估计它。
    """

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    length = n_bins * os_value
    candidate = int(raw_bin) % n_bins
    tau = float(timing_offset_chips)
    boundary = int(
        np.clip(
            math.ceil(float(os_value) * (float(n_bins - candidate) + tau)),
            0,
            length,
        )
    )
    phase_jump = np.exp(2j * np.pi * tau)

    def root_sum(start: int, count: int, sign: float) -> complex:
        # 整数 OSR 的单位根每 R 点重复，完整周期之和为零，因此最多只需实际
        # 计算 R-1 项，长符号也不会使这里的开销随 N 增长。
        remainder = int(count) % os_value
        if remainder == 0:
            return 0.0j
        indexes = (int(start) + np.arange(remainder, dtype=np.float64)) % os_value
        return complex(np.sum(np.exp(sign * 2j * np.pi * indexes / float(os_value))))

    primary = complex(boundary) + phase_jump * root_sum(
        boundary, length - boundary, -1.0
    )
    secondary = root_sum(0, boundary, 1.0) + phase_jump * complex(length - boundary)
    steering = np.asarray((primary, secondary), dtype=np.complex128)
    norm = float(np.linalg.norm(steering))
    return steering if norm <= 1e-30 else steering / norm


def _pair_consistency(
    pair: np.ndarray,
    steering: np.ndarray,
    inverse_covariance: np.ndarray,
) -> float:
    inverse = np.asarray(inverse_covariance, dtype=np.complex128)
    response = np.asarray(pair, dtype=np.complex128)
    template = np.asarray(steering, dtype=np.complex128)
    template_energy = max(float(np.real(np.vdot(template, inverse @ template))), 1e-30)
    response_energy = max(float(np.real(np.vdot(response, inverse @ response))), 1e-30)
    projection = complex(np.vdot(template, inverse @ response))
    return float(np.clip(abs(projection) ** 2 / (template_energy * response_energy), 0.0, 1.0))


def fit_fold_timing_model(
    observations: Sequence[KnownDualPeakPair],
    sf: int,
    os_factor: int,
    slope_chips_per_symbol: float = 0.0,
    pair_noise_model: PairNoiseModel | None = None,
    grid_points: int = 401,
) -> FoldTimingModel:
    """用候选相关的双峰 steering 网格搜索 payload 前定时偏差。

    每个网格点都在所有已知符号上计算白化后的模板一致性，选择平均一致性最高
    的定时偏差；``grid_contrast`` 用最佳值与中位数之差描述峰是否清晰。
    """

    usable = tuple(observations)
    if not usable:
        return FoldTimingModel(0.0, 0.0, float(slope_chips_per_symbol), 0, 0.0, 0.0)
    reference = float(np.mean([item.symbol_index for item in usable]))
    model = pair_noise_model or identity_pair_noise_model()
    count = max(21, int(grid_points))
    grid = np.linspace(-0.5, 0.5, count, endpoint=False, dtype=np.float64)
    scores = np.empty(count, dtype=np.float64)
    slope = float(slope_chips_per_symbol)
    for grid_index, offset in enumerate(grid):
        consistencies = []
        for item in usable:
            tau = float(offset + slope * (item.symbol_index - reference))
            steering = fold_pair_steering(item.raw_bin, sf, os_factor, tau)
            inverse = _pair_inverse_for_candidate(model, item.raw_bin)
            consistencies.append(_pair_consistency(item.pair, steering, inverse))
        scores[grid_index] = float(np.mean(consistencies))
    best_index = int(np.argmax(scores))
    best_offset = float(grid[best_index])
    contrast = float(scores[best_index] - np.median(scores))
    return FoldTimingModel(
        reference_symbol_index=reference,
        offset_chips_at_reference=best_offset,
        slope_chips_per_symbol=slope,
        observation_count=len(usable),
        mean_consistency=float(scores[best_index]),
        grid_contrast=contrast,
    )


def estimate_pair_noise_model(
    noise_windows: np.ndarray,
    sf: int,
    os_factor: int,
    training_bins: Sequence[int] | None = None,
    cfo_int: int = 0,
    cfo_frac: float = 0.0,
    diagonal_loading: float = 0.05,
    covariance_mode: Literal["pooled", "per_bin"] = "pooled",
) -> PairNoiseModel:
    """从纯噪声窗估计 pooled 或逐候选的 2x2 双峰协方差。"""

    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    length = n_bins * os_value
    windows = _validate_windows(noise_windows, length)
    mode = str(covariance_mode)
    if mode not in {"pooled", "per_bin"}:
        raise ValueError(f"unknown pair covariance mode: {covariance_mode}")
    if mode == "per_bin":
        bins = tuple(range(n_bins))
    elif training_bins is None:
        count = min(16, n_bins)
        bins = tuple(int(value) for value in np.linspace(0, n_bins, count, endpoint=False))
    else:
        bins = tuple(dict.fromkeys(int(value) % n_bins for value in training_bins))
    if not bins:
        raise ValueError("training_bins must contain at least one candidate")
    downchirp = _oversampled_downchirp(sf, os_value, cfo_int, cfo_frac)
    spectra = np.fft.fft(windows * downchirp[None, :], axis=1) / math.sqrt(float(length))
    indexes = np.asarray(bins, dtype=np.int64)
    pairs = np.stack(
        (spectra[:, indexes], spectra[:, indexes + length - n_bins]), axis=-1
    )
    if mode == "pooled":
        snapshots = pairs.reshape(-1, 2)
        covariance, inverse = _regularized_covariance(
            snapshots, diagonal_loading
        )
        snapshot_count = int(snapshots.shape[0])
    else:
        centered = pairs.astype(np.complex128) - np.mean(
            pairs.astype(np.complex128), axis=0, keepdims=True
        )
        covariance = np.einsum(
            "wki,wkj->kij", centered, centered.conj(), optimize=True
        ) / float(max(1, centered.shape[0] - 1))
        covariance = 0.5 * (
            covariance + covariance.conj().transpose(0, 2, 1)
        )
        scale = np.maximum(
            np.real(np.trace(covariance, axis1=1, axis2=2)) / 2.0,
            1e-30,
        )
        covariance = covariance + max(
            0.0, float(diagonal_loading)
        ) * scale[:, None, None] * np.eye(2, dtype=np.complex128)[None, :, :]
        inverse = np.linalg.pinv(covariance, hermitian=True)
        snapshot_count = int(pairs.shape[0])
    return PairNoiseModel(
        covariance,
        inverse,
        snapshot_count,
        float(diagonal_loading),
        bins,
    )


def _pair_steering(
    raw_bin: int,
    sf: int,
    os_factor: int,
    phase_rad: float,
    amplitude_model: PairAmplitudeModel,
    timing_offset_chips: float | None,
) -> np.ndarray:
    n_bins = 1 << int(sf)
    candidate = int(raw_bin) % int(n_bins)
    if str(amplitude_model) == "exact":
        tau = (
            float(timing_offset_chips)
            if timing_offset_chips is not None
            else float(phase_rad) / (2.0 * math.pi)
        )
        return fold_pair_steering(candidate, sf, os_factor, tau)
    if str(amplitude_model) == "equal":
        primary_amplitude, secondary_amplitude = 1.0, 1.0
    elif str(amplitude_model) == "fold":
        primary_amplitude = float(n_bins - candidate) / float(n_bins)
        secondary_amplitude = float(candidate) / float(n_bins)
    else:
        raise ValueError(f"unknown pair amplitude model: {amplitude_model}")
    return np.asarray(
        [primary_amplitude, secondary_amplitude * np.exp(1j * float(phase_rad))],
        dtype=np.complex128,
    )


@dataclass(frozen=True)
class DualPeakCandidateScore:
    """一个 Top-L 候选在 branch 与双峰域中的完整诊断分数。"""

    raw_bin: int
    branch_score: float
    branch_loss_db: float
    consistency: float
    matched_power: float
    primary_power: float
    secondary_power: float
    observed_phase_rad: float
    predicted_phase_rad: float
    phase_residual_rad: float
    combined_log_score: float


@dataclass(frozen=True)
class DualPeakRerankResult:
    """双峰重排的最终选择及全部候选诊断。"""

    selected_bin: int
    branch_selected_bin: int
    candidate_scores: tuple[DualPeakCandidateScore, ...]
    top_candidates: tuple[int, ...]
    fractional_offset: float
    phase_rad: float


def coherent_combination_ratio(
    primary: complex,
    secondary: complex,
    predicted_phase_rad: float,
) -> tuple[float, complex, float]:
    """对齐第二个峰后，返回归一化相干功率、合并复数值和非相干功率。

    比值为 ``|A1 + A2 exp(-j phi)|^2 / (2 (|A1|^2 + |A2|^2))``，范围
    为 ``[0, 1]``。归一化去掉了候选绝对能量，只留下第二阶段需要的相位相干证据。
    """

    first = complex(primary)
    aligned_second = complex(secondary) * np.exp(-1j * float(predicted_phase_rad))
    combined = first + aligned_second
    incoherent_power = float(abs(first) ** 2 + abs(aligned_second) ** 2)
    if incoherent_power <= 1e-30:
        return 0.0, complex(combined), incoherent_power
    ratio = float(abs(combined) ** 2 / (2.0 * incoherent_power))
    return float(np.clip(ratio, 0.0, 1.0)), complex(combined), incoherent_power


def rerank_coherent_fold_candidates(
    dechirped: np.ndarray,
    branch_scores: np.ndarray,
    sf: int,
    os_factor: int,
    phase_rad: float,
    fractional_offset: float = 0.0,
    timing_offset_chips: float | None = None,
    top_l: int = 8,
    coherence_weight: float = 0.30,
    max_branch_loss_db: float = 3.0,
    selection_mode: CoherentRerankMode = "joint",
    min_coherence_gain: float = 0.30,
    max_override_loss_db: float = 0.15,
    allow_override: bool = True,
) -> DualPeakRerankResult:
    """按相位对齐后的双峰相干性重排 branch Top-L 候选。

    候选 ``k`` 的两个完整采样率 DFT 位置为 ``k+f`` 与
    ``k+(R-1)N+f``，二者实际频率相差 ``Fs-BW``。有定时估计时，使用有限折返
    响应中随候选变化的相对相位；否则退回 ``phase_rad``。这个变体不使用双峰
    协方差或折返幅度模板。
    """

    scores = np.asarray(branch_scores, dtype=np.float64)
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    length = n_bins * os_value
    values = np.asarray(dechirped, dtype=np.complex64)
    if values.size != length:
        raise ValueError(f"dechirped has {values.size} samples, expected {length}")
    if scores.ndim != 1 or scores.size != n_bins:
        raise ValueError(f"branch_scores must have length {n_bins}")
    keep = max(1, min(int(top_l), n_bins))
    order = np.argsort(scores)[::-1][:keep]
    branch_best = int(order[0])
    best_score = max(float(scores[branch_best]), 1e-30)
    locations = np.asarray(
        [
            location
            for candidate_value in order
            for location in (
                float(int(candidate_value)) + float(fractional_offset),
                float(int(candidate_value) + length - n_bins)
                + float(fractional_offset),
            )
        ],
        dtype=np.float64,
    )
    pairs = fractional_dft(values, locations).reshape(-1, 2)
    rows: list[DualPeakCandidateScore] = []
    for row_index, candidate_value in enumerate(order):
        candidate = int(candidate_value)
        primary, secondary = (complex(item) for item in pairs[row_index])
        candidate_phase = float(phase_rad)
        if timing_offset_chips is not None:
            phase_steering = fold_pair_steering(
                candidate,
                int(sf),
                os_value,
                float(timing_offset_chips),
            )
            candidate_phase = float(
                np.angle(phase_steering[1] * np.conjugate(phase_steering[0]))
            )
        coherence, combined, incoherent_power = coherent_combination_ratio(
            primary,
            secondary,
            candidate_phase,
        )
        branch_ratio = max(float(scores[candidate]) / best_score, 1e-30)
        branch_loss_db = float(10.0 * math.log10(branch_ratio))
        eligible = branch_loss_db >= -abs(float(max_branch_loss_db))
        joint_score = (
            math.log(branch_ratio)
            + max(0.0, float(coherence_weight))
            * math.log(max(float(coherence), 1e-9))
            if eligible
            else float("-inf")
        )
        observed_phase = float(np.angle(secondary * np.conjugate(primary)))
        residual = float(
            np.angle(np.exp(1j * (observed_phase - candidate_phase)))
        )
        rows.append(
            DualPeakCandidateScore(
                raw_bin=candidate,
                branch_score=float(scores[candidate]),
                branch_loss_db=branch_loss_db,
                consistency=coherence,
                matched_power=float(abs(combined) ** 2),
                primary_power=float(abs(primary) ** 2),
                secondary_power=float(abs(secondary) ** 2),
                observed_phase_rad=observed_phase,
                predicted_phase_rad=candidate_phase,
                phase_residual_rad=residual,
                combined_log_score=float(joint_score),
            )
        )
    if not bool(allow_override):
        selected = branch_best
    elif str(selection_mode) == "joint":
        selected = max(rows, key=lambda item: item.combined_log_score).raw_bin
    elif str(selection_mode) == "coherence":
        eligible_rows = [
            item
            for item in rows
            if item.branch_loss_db >= -abs(float(max_branch_loss_db))
        ]
        selected = max(eligible_rows, key=lambda item: item.consistency).raw_bin
    elif str(selection_mode) == "confidence_gate":
        first = rows[0]
        alternatives = [
            item
            for item in rows[1:]
            if item.branch_loss_db >= -abs(float(max_override_loss_db))
            and item.consistency >= first.consistency + float(min_coherence_gain)
        ]
        selected = (
            max(alternatives, key=lambda item: item.consistency).raw_bin
            if alternatives
            else first.raw_bin
        )
    else:
        raise ValueError(f"unknown coherent selection mode: {selection_mode}")
    return DualPeakRerankResult(
        selected_bin=int(selected),
        branch_selected_bin=branch_best,
        candidate_scores=tuple(rows),
        top_candidates=tuple(int(value) for value in order),
        fractional_offset=float(fractional_offset),
        phase_rad=float(phase_rad),
    )


def rerank_dual_peak_candidates(
    dechirped: np.ndarray,
    branch_scores: np.ndarray,
    sf: int,
    os_factor: int,
    phase_rad: float,
    fractional_offset: float = 0.0,
    pair_noise_model: PairNoiseModel | None = None,
    top_l: int = 8,
    consistency_weight: float = 1.0,
    max_branch_loss_db: float = 6.0,
    amplitude_model: PairAmplitudeModel = "fold",
    timing_offset_chips: float | None = None,
    selection_mode: RerankMode = "weighted",
    min_consistency_gain: float = 0.12,
    max_override_loss_db: float = 0.10,
    allow_override: bool = True,
) -> DualPeakRerankResult:
    """用白化后的双峰一致性重排 branch-GLS 提议。

    ``allow_override=False`` 时仍计算并返回全部完整采样率诊断，但最终保持
    branch 域判决。白噪声下 Savaux 合并本身已经是匹配滤波充分统计量，因此
    第二次投影不能被误当成独立证据。
    """

    scores = np.asarray(branch_scores, dtype=np.float64)
    n_bins = 1 << int(sf)
    os_value = _validate_os_factor(os_factor)
    length = n_bins * os_value
    values = np.asarray(dechirped, dtype=np.complex64)
    if values.size != length:
        raise ValueError(f"dechirped has {values.size} samples, expected {length}")
    if scores.ndim != 1 or scores.size != n_bins:
        raise ValueError(f"branch_scores must have length {n_bins}")
    model = pair_noise_model or identity_pair_noise_model()
    keep = max(1, min(int(top_l), n_bins))
    order = np.argsort(scores)[::-1][:keep]
    branch_best = int(order[0])
    best_score = max(float(scores[branch_best]), 1e-30)
    pair_locations: list[float] = []
    for candidate_value in order:
        candidate = int(candidate_value)
        pair_locations.extend(
            (
                float(candidate) + float(fractional_offset),
                float(candidate + length - n_bins) + float(fractional_offset),
            )
        )
    extracted_pairs = fractional_dft(values, pair_locations).reshape(-1, 2)
    rows: list[DualPeakCandidateScore] = []
    for row_index, candidate_value in enumerate(order):
        candidate = int(candidate_value)
        pair = extracted_pairs[row_index].astype(np.complex128)
        inverse = _pair_inverse_for_candidate(model, candidate)
        steering = _pair_steering(
            candidate,
            int(sf),
            os_value,
            phase_rad,
            amplitude_model,
            timing_offset_chips,
        )
        inverse_steering = inverse @ steering
        steering_information = max(float(np.real(np.vdot(steering, inverse_steering))), 1e-30)
        pair_energy = max(float(np.real(np.vdot(pair, inverse @ pair))), 1e-30)
        projection = complex(np.vdot(steering, inverse @ pair))
        matched_power = float(abs(projection) ** 2 / steering_information)
        consistency = float(np.clip(abs(projection) ** 2 / (steering_information * pair_energy), 0.0, 1.0))
        branch_loss_db = float(10.0 * math.log10((float(scores[candidate]) + 1e-30) / best_score))
        eligible = branch_loss_db >= -abs(float(max_branch_loss_db))
        combined = (
            math.log((float(scores[candidate]) + 1e-30) / best_score)
            + float(consistency_weight) * math.log(max(consistency, 1e-9))
            if eligible
            else float("-inf")
        )
        primary, secondary = (complex(item) for item in pair)
        observed = float(np.angle(secondary * np.conjugate(primary)))
        steering_phase = float(np.angle(steering[1] * np.conjugate(steering[0])))
        residual = float(np.angle(np.exp(1j * (observed - steering_phase))))
        rows.append(
            DualPeakCandidateScore(
                candidate,
                float(scores[candidate]),
                branch_loss_db,
                consistency,
                matched_power,
                float(abs(primary) ** 2),
                float(abs(secondary) ** 2),
                observed,
                steering_phase,
                residual,
                float(combined),
            )
        )
    if not bool(allow_override):
        selected = rows[0].raw_bin
    elif str(selection_mode) == "confidence_gate":
        first = rows[0]
        alternatives = [
            item
            for item in rows[1:]
            if item.branch_loss_db >= -abs(float(max_override_loss_db))
            and item.consistency >= first.consistency + float(min_consistency_gain)
        ]
        selected = (
            max(alternatives, key=lambda item: item.consistency).raw_bin
            if alternatives
            else first.raw_bin
        )
    elif str(selection_mode) == "weighted":
        selected = max(rows, key=lambda item: item.combined_log_score).raw_bin
    else:
        raise ValueError(f"unknown selection_mode: {selection_mode}")
    return DualPeakRerankResult(
        int(selected),
        branch_best,
        tuple(rows),
        tuple(int(value) for value in order),
        float(fractional_offset),
        float(phase_rad),
    )
