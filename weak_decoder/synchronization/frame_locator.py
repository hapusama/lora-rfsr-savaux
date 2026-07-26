"""基于 sync word 和 SFD 的 LoRa 帧边界定位。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..chirp import build_upchirp, positive_mod
from .preamble_detector import DetectionEvent, PreambleDetectorConfig, circular_bin_distance


@dataclass(frozen=True)
class SymbolPeak:
    """一个候选符号边界上的 dechirp+FFT peak 观测。"""

    symbol_index: int
    start_sample: int
    direction: str
    peak_bin: int
    peak_power: float
    second_power: float
    total_power: float
    confidence_db: float
    peak_share: float


@dataclass(frozen=True)
class FrameLocatorConfig:
    """SFD 帧定位所需的参数。"""

    preamble_len: float
    sync_word: int = 0x34
    search_radius_samples: int = 512
    step_samples: int = 1
    preamble_bin_tol: int = 2
    sync_bin_tol: int = 4
    sfd_bin_tol: int = 4
    min_preamble_peaks: int | None = None
    symbol_search_span: int = 2

    def validate(self) -> None:
        if self.preamble_len <= 0:
            raise ValueError("preamble_len must be positive.")
        if abs(self.preamble_len - round(self.preamble_len)) > 1e-9:
            raise ValueError("frame locator currently expects an integer preamble_len.")
        if self.search_radius_samples < 0:
            raise ValueError("search_radius_samples must be non-negative.")
        if self.step_samples <= 0:
            raise ValueError("step_samples must be positive.")
        if self.preamble_bin_tol < 0 or self.sync_bin_tol < 0 or self.sfd_bin_tol < 0:
            raise ValueError("bin tolerances must be non-negative.")
        if self.min_preamble_peaks is not None and self.min_preamble_peaks <= 0:
            raise ValueError("min_preamble_peaks must be positive.")
        if self.symbol_search_span < 0:
            raise ValueError("symbol_search_span must be non-negative.")


@dataclass(frozen=True)
class FrameLocation:
    """一个检测事件对应的精细帧定位结果。"""

    event_index: int
    coarse_start_sample: int
    preamble_start_sample: int
    sfd_start_sample: int
    payload_start_sample: int
    preamble_ref_bin: int
    preamble_stable_count: int
    sync1_bin: int
    sync2_bin: int
    sync1_expected_bin: int
    sync2_expected_bin: int
    sync1_distance: int
    sync2_distance: int
    sfd1_bin: int
    sfd2_bin: int
    sfd_bin_distance: int
    mean_preamble_confidence_db: float
    mean_sfd_confidence_db: float
    score: float
    valid: bool


def sync_word_to_symbols(sync_word: int) -> tuple[int, int]:
    """复刻 gr-lora_sdr frame_sync 的 sync word 到两个符号值的映射。"""

    value = int(sync_word) & 0xFF
    return ((value & 0xF0) >> 4) << 3, (value & 0x0F) << 3


def _measure_peak(
    samples: np.ndarray,
    start_sample: int,
    reference: np.ndarray,
    symbol_index: int,
    direction: str,
) -> SymbolPeak | None:
    n = reference.size
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
    confidence_db = 10.0 * math.log10((peak_power + 1e-30) / (second_power + 1e-30))
    return SymbolPeak(
        symbol_index=int(symbol_index),
        start_sample=start,
        direction=direction,
        peak_bin=peak_bin,
        peak_power=peak_power,
        second_power=second_power,
        total_power=total_power,
        confidence_db=float(confidence_db),
        peak_share=float(peak_power / total_power),
    )


def _best_reference_bin(peaks: list[SymbolPeak], fft_len: int, tol: int) -> tuple[int, int]:
    best_bin = int(peaks[0].peak_bin)
    best_count = -1
    best_power = -1.0
    for candidate in peaks:
        close = [
            item
            for item in peaks
            if circular_bin_distance(item.peak_bin, candidate.peak_bin, fft_len) <= int(tol)
        ]
        count = len(close)
        power = float(sum(item.peak_power for item in close))
        if count > best_count or (count == best_count and power > best_power):
            best_bin = int(candidate.peak_bin)
            best_count = count
            best_power = power
    return best_bin, best_count


def _relative_bin_distance(observed: int, reference: int, expected_delta: int, fft_len: int) -> int:
    relative = positive_mod(int(observed) - int(reference), int(fft_len))
    return circular_bin_distance(relative, int(expected_delta), int(fft_len))


def _score_candidate(
    up_peaks: list[SymbolPeak],
    down_peaks: list[SymbolPeak],
    detector_config: PreambleDetectorConfig,
    locator_config: FrameLocatorConfig,
) -> FrameLocation:
    fft_len = detector_config.chirp_samples
    preamble_len = int(round(locator_config.preamble_len))
    sync1_expected, sync2_expected = sync_word_to_symbols(locator_config.sync_word)

    preamble_peaks = up_peaks[:preamble_len]
    sync1_peak = up_peaks[preamble_len]
    sync2_peak = up_peaks[preamble_len + 1]
    sfd1_peak = down_peaks[0]
    sfd2_peak = down_peaks[1]

    ref_bin, stable_count = _best_reference_bin(
        preamble_peaks,
        fft_len,
        locator_config.preamble_bin_tol,
    )
    sync1_distance = _relative_bin_distance(sync1_peak.peak_bin, ref_bin, sync1_expected, fft_len)
    sync2_distance = _relative_bin_distance(sync2_peak.peak_bin, ref_bin, sync2_expected, fft_len)
    sfd_bin_distance = circular_bin_distance(sfd1_peak.peak_bin, sfd2_peak.peak_bin, fft_len)

    mean_preamble_conf = float(np.mean([item.confidence_db for item in preamble_peaks], dtype=np.float64))
    mean_sfd_conf = float(np.mean([item.confidence_db for item in down_peaks], dtype=np.float64))
    mean_preamble_share = float(np.mean([item.peak_share for item in preamble_peaks], dtype=np.float64))
    mean_sfd_share = float(np.mean([item.peak_share for item in down_peaks], dtype=np.float64))

    min_preamble_peaks = (
        int(locator_config.min_preamble_peaks)
        if locator_config.min_preamble_peaks is not None
        else max(3, preamble_len - 2)
    )
    valid = (
        stable_count >= min_preamble_peaks
        and sync1_distance <= locator_config.sync_bin_tol
        and sync2_distance <= locator_config.sync_bin_tol
        and sfd_bin_distance <= locator_config.sfd_bin_tol
    )

    sync_margin = (
        max(0, locator_config.sync_bin_tol - sync1_distance + 1)
        + max(0, locator_config.sync_bin_tol - sync2_distance + 1)
    )
    sfd_margin = max(0, locator_config.sfd_bin_tol - sfd_bin_distance + 1)
    score = (
        100.0 * stable_count / max(1, preamble_len)
        + 8.0 * mean_preamble_conf
        + 40.0 * mean_preamble_share
        + 10.0 * sync_margin
        + 5.0 * sfd_margin
        + 4.0 * mean_sfd_conf
        + 30.0 * mean_sfd_share
        - 4.0 * (sync1_distance + sync2_distance)
        - 2.0 * sfd_bin_distance
    )

    sfd_start = int(sfd1_peak.start_sample)
    payload_start = int(round(sfd_start + 2.25 * detector_config.chirp_samples))
    return FrameLocation(
        event_index=-1,
        coarse_start_sample=0,
        preamble_start_sample=int(preamble_peaks[0].start_sample),
        sfd_start_sample=sfd_start,
        payload_start_sample=payload_start,
        preamble_ref_bin=int(ref_bin),
        preamble_stable_count=int(stable_count),
        sync1_bin=int(sync1_peak.peak_bin),
        sync2_bin=int(sync2_peak.peak_bin),
        sync1_expected_bin=int(sync1_expected),
        sync2_expected_bin=int(sync2_expected),
        sync1_distance=int(sync1_distance),
        sync2_distance=int(sync2_distance),
        sfd1_bin=int(sfd1_peak.peak_bin),
        sfd2_bin=int(sfd2_peak.peak_bin),
        sfd_bin_distance=int(sfd_bin_distance),
        mean_preamble_confidence_db=mean_preamble_conf,
        mean_sfd_confidence_db=mean_sfd_conf,
        score=float(score),
        valid=bool(valid),
    )


def locate_frame_from_event(
    samples: np.ndarray,
    event: DetectionEvent,
    detector_config: PreambleDetectorConfig,
    locator_config: FrameLocatorConfig,
    coarse_start_sample: int | None = None,
) -> FrameLocation:
    """在粗检测事件附近搜索 sync word + SFD，输出帧边界。"""

    detector_config.validate()
    locator_config.validate()
    preamble_len = int(round(locator_config.preamble_len))
    chirp_samples = detector_config.chirp_samples
    base_start = int(event.start_sample if coarse_start_sample is None else coarse_start_sample)
    required = int(math.ceil((preamble_len + 4.25) * chirp_samples))

    upchirp = build_upchirp(detector_config.sf, symbol_id=0, os_factor=detector_config.os_factor)
    down_ref = np.conjugate(upchirp).astype(np.complex64)
    up_ref = upchirp.astype(np.complex64)

    candidate_starts: list[int] = []
    for symbol_offset in range(-int(locator_config.symbol_search_span), int(locator_config.symbol_search_span) + 1):
        center = base_start + symbol_offset * chirp_samples
        start_min = max(0, center - int(locator_config.search_radius_samples))
        start_max = min(samples.size - required, center + int(locator_config.search_radius_samples))
        if start_max < start_min:
            continue
        candidate_starts.extend(range(start_min, start_max + 1, int(locator_config.step_samples)))
    candidate_starts = sorted(set(candidate_starts))
    if not candidate_starts:
        raise ValueError(f"event {event.event_index} does not have enough samples for frame location.")

    best: FrameLocation | None = None
    for candidate_start in candidate_starts:
        up_peaks: list[SymbolPeak] = []
        down_peaks: list[SymbolPeak] = []

        for symbol_index in range(preamble_len + 2):
            peak = _measure_peak(
                samples,
                candidate_start + symbol_index * chirp_samples,
                down_ref,
                symbol_index,
                "up",
            )
            if peak is None:
                break
            up_peaks.append(peak)
        if len(up_peaks) != preamble_len + 2:
            continue

        sfd_start = candidate_start + (preamble_len + 2) * chirp_samples
        for sfd_index in range(2):
            peak = _measure_peak(
                samples,
                sfd_start + sfd_index * chirp_samples,
                up_ref,
                preamble_len + 2 + sfd_index,
                "down",
            )
            if peak is None:
                break
            down_peaks.append(peak)
        if len(down_peaks) != 2:
            continue

        candidate = _score_candidate(up_peaks, down_peaks, detector_config, locator_config)
        candidate = FrameLocation(
            event_index=int(event.event_index),
            coarse_start_sample=base_start,
            preamble_start_sample=candidate.preamble_start_sample,
            sfd_start_sample=candidate.sfd_start_sample,
            payload_start_sample=candidate.payload_start_sample,
            preamble_ref_bin=candidate.preamble_ref_bin,
            preamble_stable_count=candidate.preamble_stable_count,
            sync1_bin=candidate.sync1_bin,
            sync2_bin=candidate.sync2_bin,
            sync1_expected_bin=candidate.sync1_expected_bin,
            sync2_expected_bin=candidate.sync2_expected_bin,
            sync1_distance=candidate.sync1_distance,
            sync2_distance=candidate.sync2_distance,
            sfd1_bin=candidate.sfd1_bin,
            sfd2_bin=candidate.sfd2_bin,
            sfd_bin_distance=candidate.sfd_bin_distance,
            mean_preamble_confidence_db=candidate.mean_preamble_confidence_db,
            mean_sfd_confidence_db=candidate.mean_sfd_confidence_db,
            score=candidate.score,
            valid=candidate.valid,
        )
        if (
            best is None
            or (candidate.valid and not best.valid)
            or (candidate.valid == best.valid and candidate.score > best.score)
        ):
            best = candidate

    if best is None:
        raise ValueError(f"event {event.event_index} has no valid frame-location candidate.")
    return best
