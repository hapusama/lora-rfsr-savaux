"""仿照 gr-lora_sdr 的 frame_sync 同步估计逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..chirp import build_upchirp, positive_mod, signed_fft_bin
from .frame_locator import FrameLocation, sync_word_to_symbols
from .preamble_detector import PreambleDetectorConfig, circular_bin_distance


@dataclass(frozen=True)
class FrameSyncPeak:
    """同步后某个符号的 dechirp+FFT 主峰观测。"""

    stage: str
    symbol_index: int
    start_sample: int
    peak_bin: int
    signed_peak_bin: int
    expected_signed_bin: int | None
    distance_to_expected: int | None
    peak_power: float
    second_power: float
    confidence_db: float
    peak_share: float


@dataclass(frozen=True)
class GrloraBranchSyncEstimate:
    sample_phase: int
    valid: bool
    down_val_valid: bool
    cfo_frac_est: float
    sto_frac_initial: float
    sto_frac_refined: float
    sto_frac_used: float
    sto_sample_correction: int
    cfo_int_est: int
    down_val_signed_bin: int
    cfo_total_est: float
    cfo_hz_est: float
    sfo_hat: float
    sfo_samples_per_symbol: float
    clk_off: float
    fs_p: float
    netid_sto_frac_est: float
    payload_sto_frac_est: float
    payload_sto_sample_correction: int
    netid1_est: int
    netid2_est: int
    netid_offset: int
    netid_valid: bool
    sfo_cum_initial: float
    fine_preamble_start_sample: int
    fine_payload_start_sample: int


@dataclass(frozen=True)
class GrloraFrameSyncResult:
    """gr-lora_sdr 风格同步后的帧边界、同步参数与验证结果。"""

    event_index: int
    preamble_ref_bin: int
    preamble_ref_signed_bin: int
    coarse_offset_chips: int
    coarse_offset_samples: int
    synced_preamble_start_sample: int
    synced_sfd_start_sample: int
    synced_payload_start_sample: int
    fine_preamble_start_sample: int
    fine_payload_start_sample: int
    preamble_peak_mean_signed_bin: float
    preamble_peak_max_abs_signed_bin: int
    preamble_bin0_count: int
    preamble_peak_count: int
    sync1_peak_signed_bin: int
    sync2_peak_signed_bin: int
    sync1_expected_signed_bin: int
    sync2_expected_signed_bin: int
    sync1_distance: int
    sync2_distance: int
    sfd1_peak_signed_bin: int
    sfd2_peak_signed_bin: int
    sfd_mean_signed_bin: float
    up_symbols_used: int
    cfo_frac_est: float
    sto_frac_initial: float
    sto_frac_refined: float
    sto_frac_used: float
    sto_sample_correction: int
    cfo_int_est: int
    down_val_signed_bin: int
    cfo_total_est: float
    cfo_hz_est: float
    sfo_hat: float
    sfo_samples_per_symbol: float
    clk_off: float
    fs_p: float
    netid_sto_frac_est: float
    payload_sto_frac_est: float
    payload_sto_sample_correction: int
    netid1_est: int
    netid2_est: int
    netid_offset: int
    netid_valid: bool
    sfo_cum_initial: float
    fine_preamble_peak_mean_signed_bin: float
    fine_preamble_peak_max_abs_signed_bin: int
    fine_preamble_bin0_count: int
    fine_preamble_peak_count: int
    valid: bool
    branch_sync_estimates: tuple[GrloraBranchSyncEstimate, ...]
    peaks: tuple[FrameSyncPeak, ...]


def _grlora_round(value: float) -> int:
    """复刻 C++ frame_sync_impl::my_roundf 的四舍五入规则。"""

    number = float(value)
    if number > 0.0:
        return int(number + 0.5)
    return int(math.ceil(number - 0.5))


def _wrap_half(value: float) -> float:
    """把小数 STO 包装到 [-0.5, 0.5) 附近，便于观察和后续采样。"""

    wrapped = (float(value) + 0.5) % 1.0 - 0.5
    return float(wrapped)


def _measure_peak(
    samples: np.ndarray,
    start_sample: int,
    reference: np.ndarray,
    stage: str,
    symbol_index: int,
    expected_signed_bin: int | None,
) -> FrameSyncPeak | None:
    n = int(reference.size)
    start = int(start_sample)
    stop = start + n
    if start < 0 or stop > samples.size:
        return None

    segment = np.asarray(samples[start:stop], dtype=np.complex64)
    spectrum = np.fft.fft(segment * reference)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power, dtype=np.float64))
    if total_power <= 0.0:
        return None

    peak_bin = int(np.argmax(power))
    peak_power = float(power[peak_bin])
    second_power = float(np.partition(power, -2)[-2]) if power.size > 1 else 0.0
    signed_peak = signed_fft_bin(peak_bin, n)
    if expected_signed_bin is None:
        distance = None
    else:
        expected_bin = positive_mod(int(expected_signed_bin), n)
        distance = circular_bin_distance(peak_bin, expected_bin, n)

    return FrameSyncPeak(
        stage=str(stage),
        symbol_index=int(symbol_index),
        start_sample=start,
        peak_bin=peak_bin,
        signed_peak_bin=int(signed_peak),
        expected_signed_bin=expected_signed_bin,
        distance_to_expected=distance,
        peak_power=peak_power,
        second_power=second_power,
        confidence_db=float(10.0 * math.log10((peak_power + 1e-30) / (second_power + 1e-30))),
        peak_share=float(peak_power / total_power),
    )


def _extract_chip_rate_chirps(
    samples: np.ndarray,
    start_sample: int,
    detector_config: PreambleDetectorConfig,
    symbol_count: int,
    sample_correction: int = 0,
    sample_phase: int | None = None,
) -> np.ndarray:
    """从过采样 IQ 里按 gr-lora_sdr 的方式抽成 BW 采样率的 chirp 矩阵。"""

    n_bins = int(detector_config.n_bins)
    os_factor = int(detector_config.os_factor)
    total_chips = int(symbol_count) * n_bins
    if total_chips <= 0:
        raise ValueError("symbol_count must be positive.")

    phase = int(os_factor / 2) if sample_phase is None else int(sample_phase)
    if phase < 0 or phase >= os_factor:
        raise ValueError(f"sample_phase must be in [0, {os_factor}), got {sample_phase}.")

    first = int(start_sample) + phase - int(sample_correction)
    chip_offsets = first + os_factor * np.arange(total_chips, dtype=np.int64)
    if int(chip_offsets[0]) < 0 or int(chip_offsets[-1]) >= samples.size:
        raise ValueError("not enough samples to extract chip-rate chirps.")
    return np.asarray(samples[chip_offsets], dtype=np.complex64).reshape(int(symbol_count), n_bins)


def _estimate_cfo_frac_bernier(
    chip_chirps: np.ndarray,
    downchirp: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """复刻 estimate_CFO_frac_Bernier：用相邻前导码峰值相位差估计 CFO_frac。"""

    chirps = np.asarray(chip_chirps, dtype=np.complex64)
    n_symbols, n_bins = chirps.shape
    if n_symbols < 2:
        raise ValueError("CFO_frac estimation needs at least two upchirps.")

    spectra = np.fft.fft(chirps * downchirp[np.newaxis, :], axis=1)
    power = np.abs(spectra) ** 2
    k0 = np.argmax(power, axis=1).astype(np.int64)
    peak_power = power[np.arange(n_symbols), k0]
    idx_max = int(k0[int(np.argmax(peak_power))])

    four_cum = np.sum(spectra[:-1, idx_max] * np.conjugate(spectra[1:, idx_max]))
    cfo_frac = float(-np.angle(four_cum) / (2.0 * math.pi))

    flat_n = np.arange(n_symbols * n_bins, dtype=np.float64)
    correction = np.exp(-2j * np.pi * cfo_frac * flat_n / n_bins).astype(np.complex64)
    corrected = (chirps.reshape(-1) * correction).reshape(n_symbols, n_bins)
    return cfo_frac, corrected.astype(np.complex64), idx_max


def _estimate_sto_frac(
    chip_chirps: np.ndarray,
    downchirp: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    """复刻 estimate_STO_frac：2N FFT 累加能量后用三点插值估计 STO_frac。"""

    chirps = np.asarray(chip_chirps, dtype=np.complex64)
    _, n_bins = chirps.shape
    dechirped = chirps * downchirp[np.newaxis, :]
    spectra = np.fft.fft(dechirped, n=2 * n_bins, axis=1)
    power = np.sum(np.abs(spectra) ** 2, axis=0, dtype=np.float64)
    k0 = int(np.argmax(power))

    y_minus = float(power[(k0 - 1) % (2 * n_bins)])
    y0 = float(power[k0])
    y_plus = float(power[(k0 + 1) % (2 * n_bins)])
    u = 64.0 * n_bins / 406.5506497
    v = u * 2.4674
    denom = u * (y_plus + y_minus) + v * y0
    wa = 0.0 if abs(denom) <= 1e-300 else (y_plus - y_minus) / denom
    ka = wa * n_bins / math.pi
    k_residual = math.fmod((k0 + ka) / 2.0, 1.0)
    sto_frac = float(k_residual - (1.0 if k_residual > 0.5 else 0.0))
    return sto_frac, power, k0


def _apply_flat_cfo(chip_chirps: np.ndarray, cfo_bins: float) -> np.ndarray:
    """对连续前导码样点乘上 CFO 补偿，相位不在 chirp 边界重置。"""

    chirps = np.asarray(chip_chirps, dtype=np.complex64)
    n_symbols, n_bins = chirps.shape
    flat_n = np.arange(n_symbols * n_bins, dtype=np.float64)
    correction = np.exp(-2j * np.pi * float(cfo_bins) * flat_n / n_bins).astype(np.complex64)
    return (chirps.reshape(-1) * correction).reshape(n_symbols, n_bins).astype(np.complex64)


def _apply_symbol_cfo(symbol_chips: np.ndarray, cfo_frac: float) -> np.ndarray:
    """对单个符号乘 gr-lora_sdr 的 CFO_frac 补偿向量。"""

    symbol = np.asarray(symbol_chips, dtype=np.complex64)
    n_bins = int(symbol.size)
    n = np.arange(n_bins, dtype=np.float64)
    correction = np.exp(-2j * np.pi * float(cfo_frac) * n / n_bins).astype(np.complex64)
    return (symbol * correction).astype(np.complex64)


def _sfo_correction_vector(
    symbol_count: int,
    n_bins: int,
    bw: float,
    fs_p: float,
) -> np.ndarray:
    """构造 gr-lora_sdr 用于前导码的 SFO 相位补偿向量。"""

    total = int(symbol_count) * int(n_bins)
    n = np.arange(total, dtype=np.float64)
    q = np.floor(n / int(n_bins))
    r = np.mod(n, int(n_bins))
    fs = float(bw)
    bw_value = float(bw)
    fs_p_value = float(fs_p)
    phase = (
        (r**2) / (2.0 * int(n_bins)) * ((bw_value / fs_p_value) ** 2 - (bw_value / fs) ** 2)
        + (
            q * ((bw_value / fs_p_value) ** 2 - (bw_value / fs_p_value))
            + bw_value / 2.0 * (1.0 / fs - 1.0 / fs_p_value)
        )
        * r
    )
    return np.exp(-2j * np.pi * phase).astype(np.complex64)


def _measure_chip_peak(symbol_chips: np.ndarray, reference: np.ndarray) -> tuple[int, int]:
    spectrum = np.fft.fft(np.asarray(symbol_chips, dtype=np.complex64) * reference)
    peak_bin = int(np.argmax(np.abs(spectrum) ** 2))
    return peak_bin, signed_fft_bin(peak_bin, int(reference.size))


def _measure_lora_symbol(symbol_chips: np.ndarray, reference: np.ndarray) -> int:
    """按 gr-lora_sdr get_symbol_val 的方式返回循环 FFT bin。"""

    spectrum = np.fft.fft(np.asarray(symbol_chips, dtype=np.complex64) * reference)
    return int(np.argmax(np.abs(spectrum) ** 2))


def _estimate_net_ids(
    samples: np.ndarray,
    synced_preamble_start: int,
    detector_config: PreambleDetectorConfig,
    preamble_symbols: int,
    cfo_int: int,
    cfo_frac: float,
    netid_sto_frac: float,
    downchirp: np.ndarray,
    sample_phase: int | None = None,
) -> tuple[int, int]:
    """复刻 frame_sync 对两个 sync word 符号的 STO/CFO 校正和解调。"""

    n_bins = int(detector_config.n_bins)
    os_factor = int(detector_config.os_factor)
    netid_start = int(synced_preamble_start) + int(preamble_symbols) * detector_config.chirp_samples
    # gr-lora_sdr 在 net_id_samp 中使用 0.25 个符号的缓冲；换成绝对坐标后等价于这里的整数 CFO 采样偏移。
    netid_chirps = _extract_chip_rate_chirps(
        samples,
        netid_start + int(cfo_int) * os_factor,
        detector_config,
        2,
        sample_correction=_grlora_round(float(netid_sto_frac) * os_factor),
        sample_phase=sample_phase,
    )
    netid_chirps = _apply_flat_cfo(netid_chirps, float(cfo_int))
    n = np.arange(n_bins, dtype=np.float64)
    cfo_frac_corr = np.exp(-2j * np.pi * float(cfo_frac) * n / n_bins).astype(np.complex64)
    netid_chirps = (netid_chirps * cfo_frac_corr[np.newaxis, :]).astype(np.complex64)
    return (
        _measure_lora_symbol(netid_chirps[0], downchirp),
        _measure_lora_symbol(netid_chirps[1], downchirp),
    )


def _fine_preamble_bin_stats(
    corrected_chirps: np.ndarray,
    downchirp: np.ndarray,
    bin0_tol: int,
) -> tuple[float, int, int, int]:
    spectra = np.fft.fft(np.asarray(corrected_chirps, dtype=np.complex64) * downchirp[np.newaxis, :], axis=1)
    peak_bins = np.argmax(np.abs(spectra) ** 2, axis=1).astype(np.int64)
    signed = [signed_fft_bin(int(item), int(downchirp.size)) for item in peak_bins]
    return (
        float(np.mean(signed, dtype=np.float64)),
        int(max(abs(item) for item in signed)),
        int(sum(abs(item) <= int(bin0_tol) for item in signed)),
        int(len(signed)),
    )


def cfo_int_from_down_val(signed_down_val: int) -> int:
    """复刻 gr-lora_sdr 对 SFD downchirp 解调值除以 2 得到整数 CFO 的逻辑。"""

    return int(math.floor(float(signed_down_val) / 2.0))


def _estimate_branch_sync(
    samples: np.ndarray,
    synced_preamble_start: int,
    synced_sfd_start: int,
    synced_payload_start: int,
    detector_config: PreambleDetectorConfig,
    preamble_symbols: int,
    up_symbols_used: int,
    sync1_expected: int,
    sync2_expected: int,
    down_ref_chip: np.ndarray,
    up_ref_chip: np.ndarray,
    sample_phase: int,
    bin0_tol: int,
    center_freq: float,
    fallback_cfo_int: int = 0,
    fallback_down_val_signed_bin: int = 0,
) -> GrloraBranchSyncEstimate:
    n_bins = int(detector_config.n_bins)
    os_factor = int(detector_config.os_factor)

    default = {
        "sample_phase": int(sample_phase),
        "valid": False,
        "down_val_valid": False,
        "cfo_frac_est": 0.0,
        "sto_frac_initial": 0.0,
        "sto_frac_refined": 0.0,
        "sto_frac_used": 0.0,
        "sto_sample_correction": 0,
        "cfo_int_est": int(fallback_cfo_int),
        "down_val_signed_bin": int(fallback_down_val_signed_bin),
        "cfo_total_est": 0.0,
        "cfo_hz_est": 0.0,
        "sfo_hat": 0.0,
        "sfo_samples_per_symbol": 0.0,
        "clk_off": 0.0,
        "fs_p": float(detector_config.bw),
        "netid_sto_frac_est": 0.0,
        "payload_sto_frac_est": 0.0,
        "payload_sto_sample_correction": 0,
        "netid1_est": -1,
        "netid2_est": -1,
        "netid_offset": 0,
        "netid_valid": False,
        "sfo_cum_initial": 0.0,
        "fine_preamble_start_sample": int(synced_preamble_start),
        "fine_payload_start_sample": int(synced_payload_start),
    }

    try:
        chip_preamble = _extract_chip_rate_chirps(
            samples,
            synced_preamble_start,
            detector_config,
            up_symbols_used,
            sample_correction=0,
            sample_phase=sample_phase,
        )
        cfo_frac_est, cfo_frac_corrected, _ = _estimate_cfo_frac_bernier(chip_preamble, down_ref_chip)
        sto_frac_initial, _, _ = _estimate_sto_frac(cfo_frac_corrected, down_ref_chip)
        sto_initial_sample_correction = _grlora_round(sto_frac_initial * os_factor)

        cfo_int_est = int(fallback_cfo_int)
        down_val_signed_bin = int(fallback_down_val_signed_bin)
        down_val_valid = False
        try:
            sfd2_chips = _extract_chip_rate_chirps(
                samples,
                synced_sfd_start + detector_config.chirp_samples,
                detector_config,
                1,
                sample_correction=sto_initial_sample_correction,
                sample_phase=sample_phase,
            )[0]
            sfd2_corr = _apply_symbol_cfo(sfd2_chips, cfo_frac_est)
            _, down_val_signed_bin = _measure_chip_peak(sfd2_corr, up_ref_chip)
            cfo_int_est = cfo_int_from_down_val(down_val_signed_bin)
            down_val_valid = True
        except ValueError:
            pass

        cfo_total_est = float(cfo_int_est + cfo_frac_est)
        cfo_hz_est = float(cfo_total_est * float(detector_config.bw) / n_bins)
        sfo_hat = float(cfo_total_est * float(detector_config.bw) / float(center_freq))
        sfo_samples_per_symbol = float(sfo_hat * os_factor)
        clk_off = float(sfo_hat / n_bins)
        fs_p = float(detector_config.bw * (1.0 - clk_off))

        refined_flat = cfo_frac_corrected.reshape(-1)
        refined_flat = np.roll(refined_flat, -positive_mod(cfo_int_est, n_bins))
        refined = refined_flat.reshape(up_symbols_used, n_bins)
        refined = _apply_flat_cfo(refined, cfo_int_est)
        sfo_corr = _sfo_correction_vector(up_symbols_used, n_bins, detector_config.bw, fs_p)
        refined = (refined.reshape(-1) * sfo_corr).reshape(up_symbols_used, n_bins).astype(np.complex64)
        sto_frac_refined, _, _ = _estimate_sto_frac(refined, down_ref_chip)
        diff_sto_frac = float(sto_frac_initial - sto_frac_refined)
        if abs(diff_sto_frac) <= float(os_factor - 1) / float(os_factor):
            sto_frac_used = float(sto_frac_refined)
        else:
            sto_frac_used = float(sto_frac_initial)

        sto_sample_correction = _grlora_round(sto_frac_used * os_factor)
        netid_sto_frac_est = _wrap_half(sto_frac_used + sfo_hat * preamble_symbols)
        payload_sto_frac_est = _wrap_half(sto_frac_used + sfo_hat * (preamble_symbols + 4.25))
        payload_sto_sample_correction = _grlora_round(payload_sto_frac_est * os_factor)

        try:
            netid1_est, netid2_est = _estimate_net_ids(
                samples,
                synced_preamble_start,
                detector_config,
                preamble_symbols,
                cfo_int_est,
                cfo_frac_est,
                netid_sto_frac_est,
                down_ref_chip,
                sample_phase=sample_phase,
            )
        except ValueError:
            netid1_est = -1
            netid2_est = -1
        netid_offset = int(netid1_est - int(sync1_expected))
        netid_valid = (
            netid1_est >= 0
            and netid2_est >= 0
            and abs(netid_offset) <= 2
            and positive_mod(netid2_est - netid_offset, n_bins) == positive_mod(int(sync2_expected), n_bins)
        )
        sfo_cum_initial = float(
            (payload_sto_frac_est * os_factor - payload_sto_sample_correction)
            / os_factor
        )
        fine_preamble_start = int(synced_preamble_start + int(sample_phase) - int(os_factor / 2) - sto_sample_correction)
        fine_payload_start = int(
            synced_payload_start
            + int(sample_phase)
            - int(os_factor / 2)
            + os_factor * int(cfo_int_est)
            - (os_factor * netid_offset if netid_valid else 0)
            - payload_sto_sample_correction
        )

        fine_valid = False
        try:
            fine_chirps = _build_corrected_preamble_chirps(
                samples,
                synced_preamble_start,
                detector_config,
                up_symbols_used,
                sto_sample_correction,
                cfo_int_est,
                cfo_frac_est,
                fs_p,
                sample_phase=sample_phase,
            )
            _, fine_max_abs, fine_bin0_count, fine_count = _fine_preamble_bin_stats(
                fine_chirps,
                down_ref_chip,
                bin0_tol,
            )
            fine_valid = fine_bin0_count == fine_count and fine_max_abs <= int(bin0_tol)
        except ValueError:
            fine_valid = False

        return GrloraBranchSyncEstimate(
            sample_phase=int(sample_phase),
            valid=bool(fine_valid and netid_valid),
            down_val_valid=bool(down_val_valid),
            cfo_frac_est=float(cfo_frac_est),
            sto_frac_initial=float(sto_frac_initial),
            sto_frac_refined=float(sto_frac_refined),
            sto_frac_used=float(sto_frac_used),
            sto_sample_correction=int(sto_sample_correction),
            cfo_int_est=int(cfo_int_est),
            down_val_signed_bin=int(down_val_signed_bin),
            cfo_total_est=float(cfo_total_est),
            cfo_hz_est=float(cfo_hz_est),
            sfo_hat=float(sfo_hat),
            sfo_samples_per_symbol=float(sfo_samples_per_symbol),
            clk_off=float(clk_off),
            fs_p=float(fs_p),
            netid_sto_frac_est=float(netid_sto_frac_est),
            payload_sto_frac_est=float(payload_sto_frac_est),
            payload_sto_sample_correction=int(payload_sto_sample_correction),
            netid1_est=int(netid1_est),
            netid2_est=int(netid2_est),
            netid_offset=int(netid_offset),
            netid_valid=bool(netid_valid),
            sfo_cum_initial=float(sfo_cum_initial),
            fine_preamble_start_sample=int(fine_preamble_start),
            fine_payload_start_sample=int(fine_payload_start),
        )
    except ValueError:
        return GrloraBranchSyncEstimate(**default)


def _build_corrected_preamble_chirps(
    samples: np.ndarray,
    synced_preamble_start: int,
    detector_config: PreambleDetectorConfig,
    chirp_count: int,
    sto_sample_correction: int,
    cfo_int: int,
    cfo_frac: float,
    fs_p: float,
    sample_phase: int | None = None,
) -> np.ndarray:
    """按给定 STO/CFO/SFO 估计量生成 gr-lora_sdr 风格校正后的 chip-rate 前导码。"""

    n_bins = int(detector_config.n_bins)
    count = int(chirp_count)
    chirps = _extract_chip_rate_chirps(
        samples,
        int(synced_preamble_start),
        detector_config,
        count,
        sample_correction=int(sto_sample_correction),
        sample_phase=sample_phase,
    )
    flat = chirps.reshape(-1)
    flat = np.roll(flat, -positive_mod(int(cfo_int), n_bins))
    chirps = flat.reshape(count, n_bins)

    # gr-lora_sdr 对整数 CFO 使用连续相位补偿，对小数 CFO 在每个符号内重复同一补偿向量。
    chirps = _apply_flat_cfo(chirps, float(cfo_int))
    n = np.arange(n_bins, dtype=np.float64)
    cfo_frac_corr = np.exp(-2j * np.pi * float(cfo_frac) * n / n_bins).astype(np.complex64)
    chirps = (chirps * cfo_frac_corr[np.newaxis, :]).astype(np.complex64)
    sfo_corr = _sfo_correction_vector(count, n_bins, detector_config.bw, float(fs_p))
    return (chirps.reshape(-1) * sfo_corr).reshape(count, n_bins).astype(np.complex64)


def build_grlora_corrected_preamble_chirps(
    samples: np.ndarray,
    frame_sync: GrloraFrameSyncResult,
    detector_config: PreambleDetectorConfig,
    chirp_count: int,
) -> np.ndarray:
    """按最终 STO/CFO/SFO 估计量生成 gr-lora_sdr 风格校正后的 chip-rate 前导码。"""

    return _build_corrected_preamble_chirps(
        samples,
        frame_sync.synced_preamble_start_sample,
        detector_config,
        chirp_count,
        frame_sync.sto_sample_correction,
        frame_sync.cfo_int_est,
        frame_sync.cfo_frac_est,
        frame_sync.fs_p,
    )


def run_grlora_frame_sync_validation(
    samples: np.ndarray,
    frame_location: FrameLocation,
    detector_config: PreambleDetectorConfig,
    preamble_len: float,
    sync_word: int,
    bin0_tol: int = 0,
    center_freq: float = 487.7e6,
) -> GrloraFrameSyncResult:
    """把帧定界结果按 gr-lora_sdr 流程做粗同步、CFO/STO/SFO 估计和验证。"""

    detector_config.validate()
    preamble_symbols = int(round(float(preamble_len)))
    if preamble_symbols <= 0:
        raise ValueError("preamble_len must be positive.")
    if float(center_freq) <= 0.0:
        raise ValueError("center_freq must be positive.")

    chirp_samples = detector_config.chirp_samples
    os_factor = detector_config.os_factor
    n_bins = detector_config.n_bins
    ref_signed = signed_fft_bin(frame_location.preamble_ref_bin, chirp_samples)

    # gr-lora_sdr 的 k_hat 粗同步会把 upchirp 主峰对应的时间偏移挪回 bin0。
    coarse_offset_chips = -int(ref_signed)
    coarse_offset_samples = int(coarse_offset_chips * os_factor)
    synced_preamble_start = int(frame_location.preamble_start_sample + coarse_offset_samples)
    synced_sfd_start = int(synced_preamble_start + (preamble_symbols + 2) * chirp_samples)
    synced_payload_start = int(round(synced_sfd_start + 2.25 * chirp_samples))

    upchirp_os = build_upchirp(
        detector_config.sf,
        symbol_id=0,
        os_factor=detector_config.os_factor,
    )
    down_ref_os = np.conjugate(upchirp_os).astype(np.complex64)
    up_ref_os = upchirp_os.astype(np.complex64)
    upchirp_chip = build_upchirp(detector_config.sf, symbol_id=0, os_factor=1)
    down_ref_chip = np.conjugate(upchirp_chip).astype(np.complex64)
    up_ref_chip = upchirp_chip.astype(np.complex64)
    sync1_expected, sync2_expected = sync_word_to_symbols(sync_word)

    peaks: list[FrameSyncPeak] = []
    for symbol_index in range(preamble_symbols):
        peak = _measure_peak(
            samples,
            synced_preamble_start + symbol_index * chirp_samples,
            down_ref_os,
            "preamble",
            symbol_index,
            0,
        )
        if peak is not None:
            peaks.append(peak)

    sync1_peak = _measure_peak(
        samples,
        synced_preamble_start + preamble_symbols * chirp_samples,
        down_ref_os,
        "sync",
        preamble_symbols,
        int(sync1_expected),
    )
    sync2_peak = _measure_peak(
        samples,
        synced_preamble_start + (preamble_symbols + 1) * chirp_samples,
        down_ref_os,
        "sync",
        preamble_symbols + 1,
        int(sync2_expected),
    )
    sfd1_peak = _measure_peak(
        samples,
        synced_sfd_start,
        up_ref_os,
        "sfd",
        preamble_symbols + 2,
        None,
    )
    sfd2_peak = _measure_peak(
        samples,
        synced_sfd_start + chirp_samples,
        up_ref_os,
        "sfd",
        preamble_symbols + 3,
        None,
    )

    required = [sync1_peak, sync2_peak, sfd1_peak, sfd2_peak]
    if any(item is None for item in required):
        raise ValueError(f"event {frame_location.event_index} cannot be validated after frame sync.")
    peaks.extend(item for item in required if item is not None)

    preamble_peaks = [item for item in peaks if item.stage == "preamble"]
    if not preamble_peaks:
        raise ValueError(f"event {frame_location.event_index} has no valid synced preamble peak.")

    preamble_signed = [item.signed_peak_bin for item in preamble_peaks]
    bin0_count = sum(abs(item) <= int(bin0_tol) for item in preamble_signed)
    sfd_signed = [sfd1_peak.signed_peak_bin, sfd2_peak.signed_peak_bin]  # type: ignore[union-attr]
    sync1_distance = int(sync1_peak.distance_to_expected)  # type: ignore[union-attr]
    sync2_distance = int(sync2_peak.distance_to_expected)  # type: ignore[union-attr]

    up_symbols_used = max(2, min(preamble_symbols, preamble_symbols - 4))
    chip_preamble = _extract_chip_rate_chirps(
        samples,
        synced_preamble_start,
        detector_config,
        up_symbols_used,
        sample_correction=0,
    )
    cfo_frac_est, cfo_frac_corrected, _ = _estimate_cfo_frac_bernier(chip_preamble, down_ref_chip)
    sto_frac_initial, _, _ = _estimate_sto_frac(cfo_frac_corrected, down_ref_chip)

    sto_initial_sample_correction = _grlora_round(sto_frac_initial * os_factor)
    sfd2_chips = _extract_chip_rate_chirps(
        samples,
        synced_sfd_start + chirp_samples,
        detector_config,
        1,
        sample_correction=sto_initial_sample_correction,
    )[0]
    sfd2_corr = _apply_symbol_cfo(sfd2_chips, cfo_frac_est)
    _, down_val_signed_bin = _measure_chip_peak(sfd2_corr, up_ref_chip)
    cfo_int_est = cfo_int_from_down_val(down_val_signed_bin)

    cfo_total_est = float(cfo_int_est + cfo_frac_est)
    cfo_hz_est = float(cfo_total_est * float(detector_config.bw) / n_bins)
    sfo_hat = float(cfo_total_est * float(detector_config.bw) / float(center_freq))
    sfo_samples_per_symbol = float(sfo_hat * os_factor)
    clk_off = float(sfo_hat / n_bins)
    fs_p = float(detector_config.bw * (1.0 - clk_off))

    refined_flat = cfo_frac_corrected.reshape(-1)
    refined_flat = np.roll(refined_flat, -positive_mod(cfo_int_est, n_bins))
    refined = refined_flat.reshape(up_symbols_used, n_bins)
    refined = _apply_flat_cfo(refined, cfo_int_est)
    sfo_corr = _sfo_correction_vector(up_symbols_used, n_bins, detector_config.bw, fs_p)
    refined = (refined.reshape(-1) * sfo_corr).reshape(up_symbols_used, n_bins).astype(np.complex64)
    sto_frac_refined, _, _ = _estimate_sto_frac(refined, down_ref_chip)
    diff_sto_frac = float(sto_frac_initial - sto_frac_refined)
    if abs(diff_sto_frac) <= float(os_factor - 1) / float(os_factor):
        sto_frac_used = float(sto_frac_refined)
    else:
        sto_frac_used = float(sto_frac_initial)

    sto_sample_correction = _grlora_round(sto_frac_used * os_factor)
    netid_sto_frac_est = _wrap_half(sto_frac_used + sfo_hat * preamble_symbols)
    payload_sto_frac_est = _wrap_half(sto_frac_used + sfo_hat * (preamble_symbols + 4.25))
    payload_sto_sample_correction = _grlora_round(payload_sto_frac_est * os_factor)
    netid1_est, netid2_est = _estimate_net_ids(
        samples,
        synced_preamble_start,
        detector_config,
        preamble_symbols,
        cfo_int_est,
        cfo_frac_est,
        netid_sto_frac_est,
        down_ref_chip,
    )
    netid_offset = int(netid1_est - int(sync1_expected))
    netid_valid = (
        abs(netid_offset) <= 2
        and positive_mod(netid2_est - netid_offset, n_bins) == positive_mod(int(sync2_expected), n_bins)
    )
    sfo_cum_initial = float(
        (payload_sto_frac_est * os_factor - payload_sto_sample_correction)
        / os_factor
    )
    fine_preamble_start = int(synced_preamble_start - sto_sample_correction)
    fine_payload_start = int(
        synced_payload_start
        + os_factor * int(cfo_int_est)
        - (os_factor * netid_offset if netid_valid else 0)
        - payload_sto_sample_correction
    )

    fine_chirps = _build_corrected_preamble_chirps(
        samples,
        synced_preamble_start,
        detector_config,
        up_symbols_used,
        sto_sample_correction,
        cfo_int_est,
        cfo_frac_est,
        fs_p,
    )
    fine_mean, fine_max_abs, fine_bin0_count, fine_count = _fine_preamble_bin_stats(
        fine_chirps,
        down_ref_chip,
        bin0_tol,
    )
    branch_sync_estimates = tuple(
        _estimate_branch_sync(
            samples=samples,
            synced_preamble_start=synced_preamble_start,
            synced_sfd_start=synced_sfd_start,
            synced_payload_start=synced_payload_start,
            detector_config=detector_config,
            preamble_symbols=preamble_symbols,
            up_symbols_used=up_symbols_used,
            sync1_expected=int(sync1_expected),
            sync2_expected=int(sync2_expected),
            down_ref_chip=down_ref_chip,
            up_ref_chip=up_ref_chip,
            sample_phase=phase,
            bin0_tol=bin0_tol,
            center_freq=float(center_freq),
            fallback_cfo_int=int(cfo_int_est),
            fallback_down_val_signed_bin=int(down_val_signed_bin),
        )
        for phase in range(os_factor)
    )

    # 最终同步有效性只要求两件事：
    # 1) 前导码经 CFO/STO/SFO 校正后全部回到 bin0；
    # 2) 两个 sync word / netID 作为一组满足 gr-lora_sdr 风格的共同偏移检查。
    # sync1_distance/sync2_distance 继续保留为调试字段，但不再要求绝对精确为 0。
    valid = bin0_count == len(preamble_peaks) and netid_valid

    return GrloraFrameSyncResult(
        event_index=int(frame_location.event_index),
        preamble_ref_bin=int(frame_location.preamble_ref_bin),
        preamble_ref_signed_bin=int(ref_signed),
        coarse_offset_chips=int(coarse_offset_chips),
        coarse_offset_samples=int(coarse_offset_samples),
        synced_preamble_start_sample=synced_preamble_start,
        synced_sfd_start_sample=synced_sfd_start,
        synced_payload_start_sample=synced_payload_start,
        fine_preamble_start_sample=fine_preamble_start,
        fine_payload_start_sample=fine_payload_start,
        preamble_peak_mean_signed_bin=float(np.mean(preamble_signed, dtype=np.float64)),
        preamble_peak_max_abs_signed_bin=int(max(abs(item) for item in preamble_signed)),
        preamble_bin0_count=int(bin0_count),
        preamble_peak_count=len(preamble_peaks),
        sync1_peak_signed_bin=int(sync1_peak.signed_peak_bin),  # type: ignore[union-attr]
        sync2_peak_signed_bin=int(sync2_peak.signed_peak_bin),  # type: ignore[union-attr]
        sync1_expected_signed_bin=int(sync1_expected),
        sync2_expected_signed_bin=int(sync2_expected),
        sync1_distance=sync1_distance,
        sync2_distance=sync2_distance,
        sfd1_peak_signed_bin=int(sfd1_peak.signed_peak_bin),  # type: ignore[union-attr]
        sfd2_peak_signed_bin=int(sfd2_peak.signed_peak_bin),  # type: ignore[union-attr]
        sfd_mean_signed_bin=float(np.mean(sfd_signed, dtype=np.float64)),
        up_symbols_used=int(up_symbols_used),
        cfo_frac_est=float(cfo_frac_est),
        sto_frac_initial=float(sto_frac_initial),
        sto_frac_refined=float(sto_frac_refined),
        sto_frac_used=float(sto_frac_used),
        sto_sample_correction=int(sto_sample_correction),
        cfo_int_est=int(cfo_int_est),
        down_val_signed_bin=int(down_val_signed_bin),
        cfo_total_est=float(cfo_total_est),
        cfo_hz_est=float(cfo_hz_est),
        sfo_hat=float(sfo_hat),
        sfo_samples_per_symbol=float(sfo_samples_per_symbol),
        clk_off=float(clk_off),
        fs_p=float(fs_p),
        netid_sto_frac_est=float(netid_sto_frac_est),
        payload_sto_frac_est=float(payload_sto_frac_est),
        payload_sto_sample_correction=int(payload_sto_sample_correction),
        netid1_est=int(netid1_est),
        netid2_est=int(netid2_est),
        netid_offset=int(netid_offset),
        netid_valid=bool(netid_valid),
        sfo_cum_initial=float(sfo_cum_initial),
        fine_preamble_peak_mean_signed_bin=float(fine_mean),
        fine_preamble_peak_max_abs_signed_bin=int(fine_max_abs),
        fine_preamble_bin0_count=int(fine_bin0_count),
        fine_preamble_peak_count=int(fine_count),
        valid=bool(valid),
        branch_sync_estimates=branch_sync_estimates,
        peaks=tuple(peaks),
    )
