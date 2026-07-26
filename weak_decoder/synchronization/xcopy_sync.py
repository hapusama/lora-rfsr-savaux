"""XCopy-style synchronization for repeated LoRa PHY packets.

The implementation follows the two observations used by XCopy:

* repeated preamble chirps permit long-window dechirp detection; and
* two time-aligned retransmissions cancel to a tone after conjugate
  multiplication, exposing relative timing, CFO, and phase.

The module provides both the paper-style per-packet detector and a Branch4
shortcut that uses the known retransmission period.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from ..chirp import (
    bin_to_grlora_symbol,
    build_downchirp,
    build_upchirp,
    positive_mod,
    signed_fft_bin,
)
from ..decoding.header_first_demod import decode_explicit_header
from .frame_locator import (
    FrameLocation,
    FrameLocatorConfig,
    locate_frame_from_event,
    sync_word_to_symbols,
)
from .grlora_frame_sync import GrloraFrameSyncResult, run_grlora_frame_sync_validation
from .preamble_detector import (
    DetectionEvent,
    PreambleDetectorConfig,
    WindowPeak,
    circular_bin_distance,
    find_periodic_peak_runs,
    scan_preamble_windows,
)


@dataclass(frozen=True)
class XCopyConfig:
    sf: int
    bw: float
    samp_rate: float
    preamble_symbols: int
    sync_word: int
    retransmit_period_samples: int
    payload_symbols: int
    detection_chirps: int = 8
    phase_hop_samples: int | None = None
    min_detection_peak_to_median: float = 6.0
    detection_mad_scale: float = 8.0
    detection_peak_fraction: float = 0.5
    min_detection_run: int = 2
    alignment_search_samples: int = 32
    alignment_decimation: int = 8
    max_relative_cfo_hz: float = 100.0
    min_alignment_peak_to_median: float = 25.0
    alignment_timing_model_tolerance: float = 3.0
    alignment_cfo_model_tolerance_hz: float = 3.0
    min_aligned_copies: int = 4
    pre_roll_chirps: int = 2
    post_roll_chirps: int = 1
    max_copies: int | None = None
    center_freq: float = 487.7e6
    soft_frame_top_k: int = 5
    soft_frame_search_span_chirps: int | None = None

    @property
    def n_bins(self) -> int:
        return 1 << int(self.sf)

    @property
    def os_factor(self) -> int:
        ratio = float(self.samp_rate) / float(self.bw)
        rounded = int(round(ratio))
        if rounded <= 0 or not math.isclose(ratio, rounded, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"samp_rate / bw must be an integer, got {ratio:.9g}.")
        return rounded

    @property
    def chirp_samples(self) -> int:
        return self.n_bins * self.os_factor

    @property
    def resolved_phase_hop_samples(self) -> int:
        return int(self.phase_hop_samples or self.chirp_samples)

    @property
    def frame_samples(self) -> int:
        symbols = float(self.preamble_symbols) + 4.25 + float(self.payload_symbols)
        return int(round(symbols * self.chirp_samples))

    @property
    def resolved_post_roll_chirps(self) -> int:
        # The first selected long-window detection can precede the first
        # complete preamble chirp.  Retain enough tail for the whole frame.
        return max(int(self.post_roll_chirps), int(self.detection_chirps) + 1)

    @property
    def output_samples(self) -> int:
        margin = int(self.pre_roll_chirps + self.resolved_post_roll_chirps) * self.chirp_samples
        return self.frame_samples + margin

    def validate(self) -> None:
        if self.sf <= 0 or self.bw <= 0.0 or self.samp_rate <= 0.0:
            raise ValueError("sf, bw, and samp_rate must be positive.")
        if self.preamble_symbols <= 0 or self.payload_symbols < 0:
            raise ValueError("invalid preamble or payload symbol count.")
        if not 1 <= self.detection_chirps <= self.preamble_symbols:
            raise ValueError("detection_chirps must fit inside the preamble.")
        if self.min_detection_peak_to_median <= 0.0 or self.detection_mad_scale < 0.0:
            raise ValueError("detection thresholds must be non-negative.")
        if not 0.0 <= self.detection_peak_fraction <= 1.0:
            raise ValueError("detection_peak_fraction must be in [0, 1].")
        if self.min_detection_run <= 0:
            raise ValueError("min_detection_run must be positive.")
        if self.retransmit_period_samples <= self.output_samples:
            raise ValueError("retransmission period must exceed one output frame.")
        if self.resolved_phase_hop_samples <= 0:
            raise ValueError("phase_hop_samples must be positive.")
        if self.alignment_search_samples < 0 or self.alignment_decimation <= 0:
            raise ValueError("invalid alignment search settings.")
        if self.max_relative_cfo_hz <= 0.0:
            raise ValueError("max_relative_cfo_hz must be positive.")
        if self.min_aligned_copies < 2:
            raise ValueError("min_aligned_copies must be at least two.")
        if self.pre_roll_chirps < 0 or self.post_roll_chirps < 0:
            raise ValueError("roll chirp counts must be non-negative.")
        if self.soft_frame_top_k <= 0:
            raise ValueError("soft_frame_top_k must be positive.")
        if self.soft_frame_search_span_chirps is not None and self.soft_frame_search_span_chirps < 0:
            raise ValueError("soft_frame_search_span_chirps must be non-negative.")
        _ = self.os_factor


@dataclass(frozen=True)
class XCopyDetectionBin:
    phase_index: int
    phase_sample: int
    copy_count: int
    peak_bin: int
    signed_peak_bin: int
    peak_power: float
    median_power: float
    peak_to_median: float
    selected: bool = False


@dataclass(frozen=True)
class XCopyDetection:
    detected: bool
    coarse_preamble_phase_sample: int | None
    best_phase_sample: int
    best_peak_to_median: float
    threshold: float
    noise_median: float
    noise_mad: float
    run_length: int
    bins: tuple[XCopyDetectionBin, ...]


@dataclass(frozen=True)
class XCopyPacketDetection:
    detection_index: int
    event_start_sample: int
    event_end_sample: int
    coarse_preamble_start_sample: int
    selected_window_start_sample: int
    selected_peak_bin: int
    selected_signed_peak_bin: int
    selected_peak_power: float
    selected_confidence_db: float
    selected_peak_share: float
    window_count: int
    timing_correction_samples: int
    score: float


@dataclass(frozen=True)
class XCopyAlignment:
    copy_index: int
    transmission_index: int
    nominal_start_sample: int
    relative_delay_samples: int
    relative_cfo_hz: float
    relative_phase_rad: float
    peak_to_median: float
    tone_bin: float
    included: bool
    is_reference: bool


@dataclass(frozen=True)
class XCopySoftFrameCandidate:
    rank: int
    preamble_start_sample: int
    sfd_start_sample: int
    data_start_sample: int
    score: float
    preamble_score: float
    sfd_score: float
    sync_word_bonus: float
    preamble_ref_bin: int
    preamble_stable_count: int
    sync1_bin: int
    sync2_bin: int
    sync1_distance: int
    sync2_distance: int
    sfd1_bin: int
    sfd2_bin: int
    sfd_bin_distance: int
    mean_preamble_confidence_db: float
    mean_sfd_confidence_db: float
    coarse_cfo_bins: float
    hard_grlora_pattern_valid: bool


@dataclass(frozen=True)
class XCopySyncResult:
    status: str
    config: XCopyConfig
    detection: XCopyDetection
    alignments: tuple[XCopyAlignment, ...]
    reference_copy_index: int | None
    combined_iq: np.ndarray | None
    frame_location: FrameLocation | None
    frame_sync: GrloraFrameSyncResult | None
    soft_frame_candidates: tuple[XCopySoftFrameCandidate, ...] = ()
    packet_detections: tuple[XCopyPacketDetection, ...] = ()

    @property
    def aligned_copy_count(self) -> int:
        return sum(int(item.included) for item in self.alignments)

    @property
    def selected_soft_frame(self) -> XCopySoftFrameCandidate | None:
        return next(
            (
                candidate
                for candidate in self.soft_frame_candidates
                if _soft_frame_candidate_usable(candidate, self.config)
            ),
            None,
        )


def _soft_frame_candidate_usable(
    candidate: XCopySoftFrameCandidate,
    config: XCopyConfig,
) -> bool:
    """Require primary preamble/SFD evidence without gating on sync word."""

    min_stable = min(
        int(config.preamble_symbols),
        max(3, int(math.ceil(0.5 * config.preamble_symbols))),
    )
    max_sfd_distance = max(6, int(config.n_bins // 128))
    return bool(
        candidate.preamble_stable_count >= min_stable
        and candidate.sfd_bin_distance <= max_sfd_distance
    )


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def _tone_band_indices(nfft: int, sample_rate: float, max_frequency_hz: float) -> np.ndarray:
    max_bin = min(
        nfft // 2,
        max(1, int(math.ceil(float(max_frequency_hz) * nfft / float(sample_rate)))),
    )
    return np.concatenate(
        (
            np.arange(0, max_bin + 1, dtype=np.int64),
            np.arange(nfft - max_bin, nfft, dtype=np.int64),
        )
    )


def _parabolic_peak(power: np.ndarray, peak_bin: int) -> float:
    n = int(power.size)
    y_minus = float(power[(peak_bin - 1) % n])
    y0 = float(power[peak_bin])
    y_plus = float(power[(peak_bin + 1) % n])
    denominator = y_minus - 2.0 * y0 + y_plus
    offset = 0.0 if abs(denominator) <= 1e-30 else 0.5 * (y_minus - y_plus) / denominator
    return float(peak_bin + float(np.clip(offset, -0.5, 0.5)))


def scan_periodic_preamble(samples: np.ndarray, config: XCopyConfig) -> XCopyDetection:
    """Fold scheduled retransmissions and search for a repeated preamble."""

    config.validate()
    chirp_samples = config.chirp_samples
    window_samples = config.detection_chirps * chirp_samples
    period = int(config.retransmit_period_samples)
    hop = config.resolved_phase_hop_samples
    reference = np.tile(
        np.conjugate(build_upchirp(config.sf, symbol_id=0, os_factor=config.os_factor)),
        config.detection_chirps,
    ).astype(np.complex64)

    rows: list[XCopyDetectionBin] = []
    for phase_index, phase in enumerate(range(0, period, hop)):
        starts = list(range(int(phase), int(samples.size) - window_samples + 1, period))
        if config.max_copies is not None and len(starts) > int(config.max_copies):
            starts = starts[-int(config.max_copies) :]
        accumulated: np.ndarray | None = None
        for start in starts:
            block = np.asarray(samples[start : start + window_samples], dtype=np.complex64)
            spectrum = np.fft.fft(block * reference)
            power = np.abs(spectrum) ** 2
            accumulated = power if accumulated is None else accumulated + power
        if accumulated is None:
            continue
        peak_bin = int(np.argmax(accumulated))
        peak_power = float(accumulated[peak_bin])
        median_power = float(np.median(accumulated))
        score = peak_power / (median_power + 1e-30)
        rows.append(
            XCopyDetectionBin(
                phase_index=phase_index,
                phase_sample=int(phase),
                copy_count=len(starts),
                peak_bin=peak_bin,
                signed_peak_bin=signed_fft_bin(peak_bin, accumulated.size),
                peak_power=peak_power,
                median_power=median_power,
                peak_to_median=float(score),
            )
        )

    if not rows:
        raise ValueError("capture is too short for periodic preamble detection.")

    scores = np.asarray([item.peak_to_median for item in rows], dtype=np.float64)
    noise_median = float(np.median(scores))
    noise_mad = float(np.median(np.abs(scores - noise_median)))
    robust_threshold = noise_median + config.detection_mad_scale * 1.4826 * noise_mad
    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])
    threshold = max(
        float(config.min_detection_peak_to_median),
        float(robust_threshold),
        float(config.detection_peak_fraction) * best_score,
    )
    selected = scores >= threshold

    left = best_index
    while left > 0 and bool(selected[left - 1]):
        left -= 1
    right = best_index
    while right + 1 < len(rows) and bool(selected[right + 1]):
        right += 1
    run_length = right - left + 1
    detected = bool(best_score >= threshold and run_length >= config.min_detection_run)
    coarse_phase = int(rows[left].phase_sample) if detected else None
    marked = tuple(
        replace(item, selected=bool(detected and left <= index <= right))
        for index, item in enumerate(rows)
    )
    return XCopyDetection(
        detected=detected,
        coarse_preamble_phase_sample=coarse_phase,
        best_phase_sample=int(rows[best_index].phase_sample),
        best_peak_to_median=best_score,
        threshold=float(threshold),
        noise_median=noise_median,
        noise_mad=noise_mad,
        run_length=int(run_length if detected else 0),
        bins=marked,
    )


def scan_xcopy_packet_preambles(
    samples: np.ndarray,
    config: XCopyConfig,
) -> tuple[XCopyPacketDetection, ...]:
    """Detect retransmissions individually with XCopy's long preamble window."""

    config.validate()
    fully_contained_windows = (
        int(config.preamble_symbols) - int(config.detection_chirps) + 1
    )
    detector = PreambleDetectorConfig(
        sf=config.sf,
        bw=config.bw,
        samp_rate=config.samp_rate,
        win_chirps=config.detection_chirps,
        hop_samples=config.chirp_samples,
        min_periodic_peaks=max(
            int(config.min_detection_run),
            int(math.ceil(0.6 * fully_contained_windows)),
        ),
        bin_tol=3,
    )
    windows = _scan_xcopy_long_windows(samples, detector)
    events = _find_xcopy_long_peak_runs(windows, detector)
    detections: list[XCopyPacketDetection] = []
    for event in events:
        event_windows = windows[event.first_window_index : event.last_window_index + 1]
        if not event_windows:
            continue
        max_power = max(float(item.peak_power) for item in event_windows)
        plateau_threshold = float(config.detection_peak_fraction) * max_power
        selected = next(
            (item for item in event_windows if float(item.peak_power) >= plateau_threshold),
            max(event_windows, key=lambda item: item.peak_power),
        )
        signed_peak = signed_fft_bin(selected.peak_bin, detector.window_samples)
        timing_correction = int(
            round(
                float(signed_peak)
                * float(config.os_factor)
                / float(config.detection_chirps)
            )
        )
        coarse_start = int(selected.start_sample - timing_correction)
        score = float(
            selected.peak_power
            / max(float(selected.second_power), 1e-30)
        )
        detections.append(
            XCopyPacketDetection(
                detection_index=len(detections),
                event_start_sample=int(event.start_sample),
                event_end_sample=int(event.end_sample),
                coarse_preamble_start_sample=coarse_start,
                selected_window_start_sample=int(selected.start_sample),
                selected_peak_bin=int(selected.peak_bin),
                selected_signed_peak_bin=int(signed_peak),
                selected_peak_power=float(selected.peak_power),
                selected_confidence_db=float(selected.confidence_db),
                selected_peak_share=float(selected.peak_share),
                window_count=int(event.window_count),
                timing_correction_samples=timing_correction,
                score=score,
            )
        )

    # One long stable run should represent one packet. Merge overlapping
    # detections conservatively without using the retransmission period.
    merged: list[XCopyPacketDetection] = []
    min_gap = max(config.frame_samples // 2, config.preamble_symbols * config.chirp_samples)
    for item in sorted(detections, key=lambda value: value.coarse_preamble_start_sample):
        if (
            merged
            and item.coarse_preamble_start_sample - merged[-1].coarse_preamble_start_sample < min_gap
        ):
            if item.score > merged[-1].score:
                merged[-1] = replace(item, detection_index=merged[-1].detection_index)
            continue
        merged.append(replace(item, detection_index=len(merged)))

    valid = [
        item
        for item in merged
        if item.coarse_preamble_start_sample - config.alignment_search_samples >= 0
        and item.coarse_preamble_start_sample + config.frame_samples + config.alignment_search_samples
        <= samples.size
    ]
    if config.max_copies is not None and len(valid) > int(config.max_copies):
        valid = valid[-int(config.max_copies) :]
    return tuple(replace(item, detection_index=index) for index, item in enumerate(valid))


def _scan_xcopy_long_windows(
    samples: np.ndarray,
    config: PreambleDetectorConfig,
) -> list[WindowPeak]:
    """Scan with one FFT over the complete multi-chirp detection window.

    Tiling the base downchirp across the long window preserves the complex
    phase between repeated preamble chirps.  This is the enlarged-window FFT
    used by XCopy, rather than a non-coherent sum of per-chirp powers.
    """

    config.validate()
    if samples.size < config.window_samples:
        return []

    chirp_samples = config.chirp_samples
    win_chirps = int(config.win_chirps)
    long_samples = int(config.window_samples)
    hop = int(config.resolved_hop_samples)
    downchirp = np.conjugate(
        build_upchirp(config.sf, symbol_id=0, os_factor=config.os_factor)
    ).astype(np.complex64)
    long_downchirp = np.tile(downchirp, win_chirps)
    peaks: list[WindowPeak] = []
    max_start = int(samples.size - long_samples)
    for window_index, start in enumerate(range(0, max_start + 1, hop)):
        window = np.asarray(samples[start : start + long_samples], dtype=np.complex64)
        spectrum = np.fft.fft(window * long_downchirp)
        energy = np.abs(spectrum) ** 2
        total_power = float(np.sum(energy, dtype=np.float64))
        if total_power <= 0.0:
            peaks.append(
                WindowPeak(
                    window_index=window_index,
                    start_sample=int(start),
                    end_sample=int(start + long_samples),
                    peak_bin=-1,
                    peak_power=0.0,
                    second_power=0.0,
                    total_power=0.0,
                    confidence_db=float("nan"),
                    peak_share=float("nan"),
                    valid=False,
                )
            )
            continue
        peak_bin = int(np.argmax(energy))
        peak_power = float(energy[peak_bin])
        second_power = (
            float(np.partition(energy, -2)[-2]) if energy.size > 1 else 0.0
        )
        peaks.append(
            WindowPeak(
                window_index=window_index,
                start_sample=int(start),
                end_sample=int(start + long_samples),
                peak_bin=peak_bin,
                peak_power=peak_power,
                second_power=second_power,
                total_power=total_power,
                confidence_db=float(
                    10.0
                    * math.log10((peak_power + 1e-30) / (second_power + 1e-30))
                ),
                peak_share=float(peak_power / total_power),
                valid=True,
            )
        )
    return peaks


def _find_xcopy_long_peak_runs(
    windows: list[WindowPeak],
    config: PreambleDetectorConfig,
) -> list[DetectionEvent]:
    """Group stable peaks whose bin coordinates use the long FFT length."""

    events: list[DetectionEvent] = []
    run: list[WindowPeak] = []
    reference_bin: int | None = None
    n_fft_bins = int(config.window_samples)
    bin_tol = int(config.bin_tol) * int(config.win_chirps)

    def close_run() -> None:
        nonlocal run, reference_bin
        if len(run) >= int(config.min_periodic_peaks) and reference_bin is not None:
            bins = [item.peak_bin for item in run]
            events.append(
                DetectionEvent(
                    event_index=len(events),
                    start_sample=int(run[0].start_sample),
                    end_sample=int(run[-1].end_sample),
                    first_window_index=int(run[0].window_index),
                    last_window_index=int(run[-1].window_index),
                    window_count=len(run),
                    reference_bin=int(reference_bin),
                    bin_min=int(min(bins)),
                    bin_max=int(max(bins)),
                    mean_peak_power=float(
                        np.mean([item.peak_power for item in run], dtype=np.float64)
                    ),
                    mean_confidence_db=float(
                        np.mean(
                            [item.confidence_db for item in run],
                            dtype=np.float64,
                        )
                    ),
                    max_peak_share=float(
                        np.nanmax([item.peak_share for item in run])
                    ),
                )
            )
        run = []
        reference_bin = None

    for window in windows:
        if not window.valid:
            close_run()
            continue
        if not run:
            run = [window]
            reference_bin = int(window.peak_bin)
            continue
        assert reference_bin is not None
        if circular_bin_distance(window.peak_bin, reference_bin, n_fft_bins) <= bin_tol:
            run.append(window)
        else:
            close_run()
            run = [window]
            reference_bin = int(window.peak_bin)
    close_run()
    return events


def _nominal_copy_starts(
    sample_count: int,
    coarse_phase: int,
    config: XCopyConfig,
) -> list[tuple[int, int]]:
    period = int(config.retransmit_period_samples)
    required = config.frame_samples + config.alignment_search_samples
    first_index = 0
    while coarse_phase + first_index * period < 0:
        first_index += 1
    starts: list[tuple[int, int]] = []
    transmission_index = first_index
    while True:
        start = int(coarse_phase + transmission_index * period)
        if start + required > sample_count:
            break
        if start - config.alignment_search_samples >= 0:
            starts.append((transmission_index, start))
        transmission_index += 1
    if config.max_copies is not None and len(starts) > int(config.max_copies):
        starts = starts[-int(config.max_copies) :]
    return starts


def _tone_measurement(
    reference: np.ndarray,
    candidate: np.ndarray,
    config: XCopyConfig,
    decimation: int | None = None,
) -> tuple[float, float, float, float]:
    decimation_value = int(decimation or config.alignment_decimation)
    product = reference[::decimation_value] * np.conjugate(candidate[::decimation_value])
    nfft = _next_power_of_two(int(product.size))
    spectrum = np.fft.fft(product, n=nfft)
    power = np.abs(spectrum) ** 2
    decimated_rate = float(config.samp_rate) / decimation_value
    band = _tone_band_indices(nfft, decimated_rate, config.max_relative_cfo_hz)
    peak_bin = int(band[int(np.argmax(power[band]))])
    score = float(power[peak_bin] / (float(np.median(power)) + 1e-30))
    fractional_bin = _parabolic_peak(power, peak_bin)
    if fractional_bin > nfft / 2.0:
        fractional_bin -= nfft
    frequency_hz = float(fractional_bin * decimated_rate / nfft)
    time_s = decimation_value * np.arange(product.size, dtype=np.float64) / float(config.samp_rate)
    phasor = np.exp(-2j * np.pi * frequency_hz * time_s)
    phase_rad = float(np.angle(np.sum(product * phasor)))
    return score, frequency_hz, phase_rad, float(fractional_bin)


def _choose_reference(
    samples: np.ndarray,
    starts: list[tuple[int, int]],
    config: XCopyConfig,
) -> int:
    length = config.frame_samples
    decimation = config.alignment_decimation
    segments = [
        np.asarray(samples[start : start + length : decimation], dtype=np.complex64)
        for _, start in starts
    ]
    nfft = _next_power_of_two(segments[0].size)
    rate = float(config.samp_rate) / decimation
    band = _tone_band_indices(nfft, rate, config.max_relative_cfo_hz)
    scores: list[list[float]] = [[] for _ in starts]
    for lhs in range(len(starts)):
        for rhs in range(lhs + 1, len(starts)):
            power = np.abs(np.fft.fft(segments[lhs] * np.conjugate(segments[rhs]), n=nfft)) ** 2
            score = float(np.max(power[band]) / (float(np.median(power)) + 1e-30))
            scores[lhs].append(score)
            scores[rhs].append(score)

    def utility(index: int) -> float:
        strongest = sorted(scores[index], reverse=True)[: min(5, len(scores[index]))]
        return float(sum(strongest))

    return int(max(range(len(starts)), key=utility))


def _measure_alignments(
    samples: np.ndarray,
    starts: list[tuple[int, int]],
    reference_index: int,
    config: XCopyConfig,
) -> list[XCopyAlignment]:
    length = config.frame_samples
    _, reference_start = starts[reference_index]
    reference = np.asarray(samples[reference_start : reference_start + length], dtype=np.complex64)
    alignments: list[XCopyAlignment] = []
    for copy_index, (transmission_index, nominal_start) in enumerate(starts):
        if copy_index == reference_index:
            alignments.append(
                XCopyAlignment(
                    copy_index=copy_index,
                    transmission_index=transmission_index,
                    nominal_start_sample=nominal_start,
                    relative_delay_samples=0,
                    relative_cfo_hz=0.0,
                    relative_phase_rad=0.0,
                    peak_to_median=float("inf"),
                    tone_bin=0.0,
                    included=True,
                    is_reference=True,
                )
            )
            continue

        radius = int(config.alignment_search_samples)
        if radius <= 64:
            stages = [(-radius, radius, 1, int(config.alignment_decimation))]
            best: tuple[float, int, float, float, float] | None = None
            for lower, upper, step, decimation in stages:
                for delay in range(int(lower), int(upper) + 1, int(step)):
                    start = nominal_start + delay
                    candidate = np.asarray(samples[start : start + length], dtype=np.complex64)
                    score, cfo_hz, phase_rad, tone_bin = _tone_measurement(
                        reference,
                        candidate,
                        config,
                        decimation=decimation,
                    )
                    row = (score, delay, cfo_hz, phase_rad, tone_bin)
                    if best is None or row[0] > best[0]:
                        best = row
        else:
            # The coarse estimator can be wrong by an integer chirp, while
            # its within-chirp timing remains sample-level.  Evaluate every
            # integer-chirp hypothesis with a small raw-sample neighbourhood.
            # Decimation never goes below BW here: larger values alias the
            # Eq. (4) tone and can hide the true timing peak.
            search_decimation = min(
                int(config.alignment_decimation),
                int(config.os_factor),
            )
            chirp_samples = int(config.chirp_samples)
            chirp_hypotheses = range(
                -int(math.ceil(radius / chirp_samples)),
                int(math.ceil(radius / chirp_samples)) + 1,
            )
            best = None
            for chirp_offset in chirp_hypotheses:
                center = chirp_offset * chirp_samples
                for residual in range(-8, 9, max(1, search_decimation)):
                    delay = center + residual
                    if delay < -radius or delay > radius:
                        continue
                    start = nominal_start + delay
                    candidate = np.asarray(samples[start : start + length], dtype=np.complex64)
                    score, cfo_hz, phase_rad, tone_bin = _tone_measurement(
                        reference,
                        candidate,
                        config,
                        decimation=search_decimation,
                    )
                    row = (score, delay, cfo_hz, phase_rad, tone_bin)
                    if best is None or row[0] > best[0]:
                        best = row

            assert best is not None
            fine_center = int(best[1])
            fine_best = best
            for delay in range(
                max(-radius, fine_center - max(4, search_decimation)),
                min(radius, fine_center + max(4, search_decimation)) + 1,
            ):
                start = nominal_start + delay
                candidate = np.asarray(samples[start : start + length], dtype=np.complex64)
                score, cfo_hz, phase_rad, tone_bin = _tone_measurement(
                    reference,
                    candidate,
                    config,
                    decimation=search_decimation,
                )
                row = (score, delay, cfo_hz, phase_rad, tone_bin)
                if row[0] > fine_best[0]:
                    fine_best = row

            # Re-measure the selected delay at the raw sample rate.  This is
            # the paper's full-resolution whole-packet conjugate FFT and is
            # also the score used for copy grouping.
            final_delay = int(fine_best[1])
            final_start = nominal_start + final_delay
            final_candidate = np.asarray(
                samples[final_start : final_start + length],
                dtype=np.complex64,
            )
            final_score, final_cfo, final_phase, final_tone_bin = _tone_measurement(
                reference,
                final_candidate,
                config,
                decimation=1,
            )
            best = (
                final_score,
                final_delay,
                final_cfo,
                final_phase,
                final_tone_bin,
            )
        assert best is not None
        alignments.append(
            XCopyAlignment(
                copy_index=copy_index,
                transmission_index=transmission_index,
                nominal_start_sample=nominal_start,
                relative_delay_samples=int(best[1]),
                relative_cfo_hz=float(best[2]),
                relative_phase_rad=float(best[3]),
                peak_to_median=float(best[0]),
                tone_bin=float(best[4]),
                included=False,
                is_reference=False,
            )
        )
    return alignments


def _select_consistent_alignments(
    alignments: list[XCopyAlignment],
    config: XCopyConfig,
) -> list[XCopyAlignment]:
    candidates = [
        item
        for item in alignments
        if item.is_reference or item.peak_to_median >= config.min_alignment_peak_to_median
    ]
    reference = next(item for item in candidates if item.is_reference)
    if len(candidates) < 2:
        return [replace(item, included=item.is_reference) for item in alignments]

    best_inliers: set[int] = {reference.copy_index}
    best_objective = (1, 0.0)
    model_pairs = [
        (candidates[lhs], candidates[rhs])
        for lhs in range(len(candidates))
        for rhs in range(lhs + 1, len(candidates))
        if candidates[lhs].transmission_index != candidates[rhs].transmission_index
    ]
    for lhs, rhs in model_pairs:
        index_delta = rhs.transmission_index - lhs.transmission_index
        timing_slope = (rhs.relative_delay_samples - lhs.relative_delay_samples) / index_delta
        cfo_slope = (rhs.relative_cfo_hz - lhs.relative_cfo_hz) / index_delta
        inliers: set[int] = set()
        weight = 0.0
        for item in candidates:
            relative_index = item.transmission_index - lhs.transmission_index
            timing_prediction = lhs.relative_delay_samples + timing_slope * relative_index
            cfo_prediction = lhs.relative_cfo_hz + cfo_slope * relative_index
            timing_error = abs(item.relative_delay_samples - timing_prediction)
            cfo_error = abs(item.relative_cfo_hz - cfo_prediction)
            if (
                timing_error <= config.alignment_timing_model_tolerance
                and cfo_error <= config.alignment_cfo_model_tolerance_hz
            ):
                inliers.add(item.copy_index)
                if not item.is_reference:
                    weight += math.log1p(item.peak_to_median)
        if reference.copy_index not in inliers:
            continue
        objective = (len(inliers), weight)
        if objective > best_objective:
            best_objective = objective
            best_inliers = inliers

    return [replace(item, included=item.copy_index in best_inliers) for item in alignments]


def _combine_aligned_copies(
    samples: np.ndarray,
    alignments: list[XCopyAlignment],
    config: XCopyConfig,
) -> np.ndarray:
    pre_roll = config.pre_roll_chirps * config.chirp_samples
    length = config.output_samples
    relative_time = (
        np.arange(length, dtype=np.float64) - float(pre_roll)
    ) / float(config.samp_rate)
    combined = np.zeros(length, dtype=np.complex128)
    count = 0
    for item in alignments:
        if not item.included:
            continue
        start = item.nominal_start_sample - pre_roll + item.relative_delay_samples
        stop = start + length
        if start < 0 or stop > samples.size:
            continue
        segment = np.asarray(samples[start:stop], dtype=np.complex64)
        correction = np.exp(
            1j
            * (
                2.0 * np.pi * item.relative_cfo_hz * relative_time
                + item.relative_phase_rad
            )
        )
        combined += segment * correction
        count += 1
    if count < config.min_aligned_copies:
        raise ValueError(f"only {count} aligned copies are available for combining.")
    return (combined / float(count)).astype(np.complex64)


def _circular_reference_bin(peaks: np.ndarray, power: np.ndarray, fft_len: int, tolerance: int) -> tuple[int, int]:
    best_bin = int(peaks[0])
    best_count = -1
    best_power = -1.0
    for peak_bin in peaks:
        distances = np.asarray(
            [circular_bin_distance(int(value), int(peak_bin), fft_len) for value in peaks],
            dtype=np.int64,
        )
        mask = distances <= int(tolerance)
        count = int(np.count_nonzero(mask))
        close_power = float(np.sum(power[mask], dtype=np.float64))
        if count > best_count or (count == best_count and close_power > best_power):
            best_bin = int(peak_bin)
            best_count = count
            best_power = close_power
    return best_bin, best_count


def _peak_statistics(power: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    peak_bins = np.argmax(power, axis=1).astype(np.int64)
    rows = np.arange(power.shape[0], dtype=np.int64)
    peak_power = power[rows, peak_bins].astype(np.float64)
    second_power = np.partition(power, -2, axis=1)[:, -2].astype(np.float64)
    total_power = np.sum(power, axis=1, dtype=np.float64)
    confidence_db = 10.0 * np.log10((peak_power + 1e-30) / (second_power + 1e-30))
    peak_share = peak_power / np.maximum(total_power, 1e-30)
    return peak_bins, peak_power, confidence_db, peak_share


def _soft_frame_candidate_at(
    samples: np.ndarray,
    start_sample: int,
    config: XCopyConfig,
) -> XCopySoftFrameCandidate | None:
    chirp_samples = config.chirp_samples
    preamble_symbols = int(config.preamble_symbols)
    required = int(math.ceil((preamble_symbols + 4.25) * chirp_samples))
    start = int(start_sample)
    if start < 0 or start + required > samples.size:
        return None

    upchirp = build_upchirp(config.sf, symbol_id=0, os_factor=config.os_factor)
    down_reference = np.conjugate(upchirp).astype(np.complex64)
    up_count = preamble_symbols + 2
    up_block = np.asarray(
        samples[start : start + up_count * chirp_samples],
        dtype=np.complex64,
    ).reshape(up_count, chirp_samples)
    up_power = np.abs(np.fft.fft(up_block * down_reference[np.newaxis, :], axis=1)) ** 2
    up_bins, up_peak_power, up_confidence, up_share = _peak_statistics(up_power)

    sfd_start = start + up_count * chirp_samples
    down_block = np.asarray(
        samples[sfd_start : sfd_start + 2 * chirp_samples],
        dtype=np.complex64,
    ).reshape(2, chirp_samples)
    down_power = np.abs(np.fft.fft(down_block * upchirp[np.newaxis, :], axis=1)) ** 2
    down_bins, _, down_confidence, down_share = _peak_statistics(down_power)

    preamble_bins = up_bins[:preamble_symbols]
    preamble_power = up_peak_power[:preamble_symbols]
    reference_bin, stable_count = _circular_reference_bin(
        preamble_bins,
        preamble_power,
        chirp_samples,
        tolerance=3,
    )
    sync1_expected, sync2_expected = sync_word_to_symbols(config.sync_word)
    sync1_distance = circular_bin_distance(
        positive_mod(int(up_bins[preamble_symbols]) - reference_bin, chirp_samples),
        sync1_expected,
        chirp_samples,
    )
    sync2_distance = circular_bin_distance(
        positive_mod(int(up_bins[preamble_symbols + 1]) - reference_bin, chirp_samples),
        sync2_expected,
        chirp_samples,
    )
    sfd_distance = circular_bin_distance(int(down_bins[0]), int(down_bins[1]), chirp_samples)

    mean_preamble_confidence = float(np.mean(up_confidence[:preamble_symbols], dtype=np.float64))
    mean_preamble_share = float(np.mean(up_share[:preamble_symbols], dtype=np.float64))
    mean_sfd_confidence = float(np.mean(down_confidence, dtype=np.float64))
    mean_sfd_share = float(np.mean(down_share, dtype=np.float64))

    preamble_score = (
        100.0 * stable_count / max(1, preamble_symbols)
        + 8.0 * mean_preamble_confidence
        + 40.0 * mean_preamble_share
    )
    # SFD consistency is primary frame-boundary evidence, but it remains
    # bounded and continuous instead of becoming a hard rejection.
    sfd_score = (
        30.0 * math.exp(-float(sfd_distance) / 4.0)
        + 4.0 * mean_sfd_confidence
        + 30.0 * mean_sfd_share
    )
    # The known sync word is only a bounded ranking bonus. Large distances
    # cannot outweigh a strong preamble transition and SFD pair.
    sync_word_bonus = 5.0 * (
        math.exp(-float(sync1_distance) / 8.0)
        + math.exp(-float(sync2_distance) / 8.0)
    )
    score = preamble_score + sfd_score + sync_word_bonus

    signed_up = signed_fft_bin(reference_bin, chirp_samples)
    signed_down = [
        signed_fft_bin(int(down_bins[index]), chirp_samples)
        for index in range(2)
    ]
    coarse_cfo = 0.5 * (float(signed_up) + float(np.mean(signed_down)))
    hard_valid = bool(
        stable_count >= max(3, preamble_symbols - 4)
        and sync1_distance <= 6
        and sync2_distance <= 6
        and sfd_distance <= 6
    )
    return XCopySoftFrameCandidate(
        rank=-1,
        preamble_start_sample=start,
        sfd_start_sample=sfd_start,
        data_start_sample=int(round(sfd_start + 2.25 * chirp_samples)),
        score=float(score),
        preamble_score=float(preamble_score),
        sfd_score=float(sfd_score),
        sync_word_bonus=float(sync_word_bonus),
        preamble_ref_bin=int(reference_bin),
        preamble_stable_count=int(stable_count),
        sync1_bin=int(up_bins[preamble_symbols]),
        sync2_bin=int(up_bins[preamble_symbols + 1]),
        sync1_distance=int(sync1_distance),
        sync2_distance=int(sync2_distance),
        sfd1_bin=int(down_bins[0]),
        sfd2_bin=int(down_bins[1]),
        sfd_bin_distance=int(sfd_distance),
        mean_preamble_confidence_db=mean_preamble_confidence,
        mean_sfd_confidence_db=mean_sfd_confidence,
        coarse_cfo_bins=float(coarse_cfo),
        hard_grlora_pattern_valid=hard_valid,
    )


def locate_xcopy_soft_frame_candidates(
    combined: np.ndarray,
    config: XCopyConfig,
) -> tuple[XCopySoftFrameCandidate, ...]:
    """Rank absolute frame boundaries without hard sync-word gating."""

    chirp_samples = config.chirp_samples
    base_start = config.pre_roll_chirps * chirp_samples
    span_chirps = (
        int(config.soft_frame_search_span_chirps)
        if config.soft_frame_search_span_chirps is not None
        else max(2, int(config.detection_chirps))
    )
    span_samples = span_chirps * chirp_samples
    coarse_step = max(config.os_factor * 2, chirp_samples // 8)
    start_min = max(0, base_start - span_samples)
    required = int(math.ceil((config.preamble_symbols + 4.25) * chirp_samples))
    start_max = min(combined.size - required, base_start + span_samples)
    if start_max < start_min:
        return ()

    coarse: list[XCopySoftFrameCandidate] = []
    for start in range(start_min, start_max + 1, coarse_step):
        candidate = _soft_frame_candidate_at(combined, start, config)
        if candidate is not None:
            coarse.append(candidate)
    if not coarse:
        return ()

    basin_centers: list[int] = []
    for candidate in sorted(coarse, key=lambda item: item.score, reverse=True):
        if all(abs(candidate.preamble_start_sample - center) >= chirp_samples // 2 for center in basin_centers):
            basin_centers.append(candidate.preamble_start_sample)
        if len(basin_centers) >= max(config.soft_frame_top_k * 2, 4):
            break

    refined: list[XCopySoftFrameCandidate] = []
    fine_step = max(1, config.os_factor)
    for center in basin_centers:
        local: list[XCopySoftFrameCandidate] = []
        for start in range(
            max(start_min, center - coarse_step),
            min(start_max, center + coarse_step) + 1,
            fine_step,
        ):
            candidate = _soft_frame_candidate_at(combined, start, config)
            if candidate is not None:
                local.append(candidate)
        if not local:
            continue
        local_best = max(local, key=lambda item: item.score)
        sample_refined = [
            candidate
            for start in range(
                max(start_min, local_best.preamble_start_sample - fine_step),
                min(start_max, local_best.preamble_start_sample + fine_step) + 1,
            )
            if (candidate := _soft_frame_candidate_at(combined, start, config)) is not None
        ]
        refined.append(max(sample_refined, key=lambda item: item.score))

    selected: list[XCopySoftFrameCandidate] = []
    for candidate in sorted(refined, key=lambda item: item.score, reverse=True):
        if all(
            abs(candidate.preamble_start_sample - item.preamble_start_sample) >= chirp_samples // 2
            for item in selected
        ):
            selected.append(candidate)
        if len(selected) >= config.soft_frame_top_k:
            break
    return tuple(replace(item, rank=index + 1) for index, item in enumerate(selected))


def _split_cfo_bins(value: float) -> tuple[int, float]:
    integer = int(math.floor(float(value) + 0.5))
    return integer, float(value) - integer


def _apply_header_bin_consensus(
    rows: list[dict[str, object]],
    config: XCopyConfig,
    payload_len: int,
    coding_rate: int,
    has_crc: bool,
) -> None:
    """Report a header-based bin gauge without filtering or mutating copies."""

    by_copy: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_copy.setdefault(int(row["copy_index"]), []).append(row)

    best_delta = 0
    best_objective = (-1, -1, 0)
    for delta in range(-4, 5):
        expected_matches = 0
        valid_headers = 0
        for copy_rows in by_copy.values():
            header_rows = sorted(
                (row for row in copy_rows if row["stage"] == "header"),
                key=lambda row: int(row["frame_symbol_index"]),
            )
            if len(header_rows) != 8:
                continue
            values = [
                bin_to_grlora_symbol(
                    positive_mod(int(row["raw_fft_bin"]) + delta, config.n_bins),
                    config.sf,
                    is_header=True,
                    ldro=False,
                )
                for row in header_rows
            ]
            decoded = decode_explicit_header(values, config.sf, config.bw, ldro_mode=2)
            if decoded.header_valid:
                valid_headers += 1
            if (
                decoded.header_valid
                and decoded.payload_len == int(payload_len)
                and decoded.cr == int(coding_rate)
                and decoded.has_crc == bool(has_crc)
            ):
                expected_matches += 1
        objective = (expected_matches, valid_headers, -abs(delta))
        if objective > best_objective:
            best_objective = objective
            best_delta = delta

    for copy_rows in by_copy.values():
        header_rows = sorted(
            (row for row in copy_rows if row["stage"] == "header"),
            key=lambda row: int(row["frame_symbol_index"]),
        )
        corrected_values = [
            bin_to_grlora_symbol(
                positive_mod(int(row["raw_fft_bin"]) + best_delta, config.n_bins),
                config.sf,
                is_header=True,
                ldro=False,
            )
            for row in header_rows
        ]
        decoded = (
            decode_explicit_header(corrected_values, config.sf, config.bw, ldro_mode=2)
            if len(corrected_values) == 8
            else None
        )
        for row in copy_rows:
            original_bin = int(row["raw_fft_bin"])
            original_cfo_int = int(row["cfo_int"])
            row["raw_fft_bin_uncorrected"] = original_bin
            row["cfo_int_uncorrected"] = original_cfo_int
            row["bin_offset_correction"] = 0
            row["header_consensus_suggested_bin_correction"] = int(best_delta)
            row["header_consensus_expected_matches"] = int(best_objective[0])
            row["header_consensus_valid_count"] = int(best_objective[1])
            row["header_valid"] = int(decoded.header_valid) if decoded is not None else 0


def xcopy_raw_symbol_rows(
    samples: np.ndarray,
    result: XCopySyncResult,
    header_symbols: int = 8,
    payload_len: int = 33,
    coding_rate: int = 3,
    has_crc: bool = True,
    ldro: bool = False,
) -> list[dict[str, object]]:
    """Map the selected XCopy boundary back to uncombined raw retransmissions."""

    candidate = result.selected_soft_frame
    if candidate is None:
        return []
    config = result.config
    chirp_samples = config.chirp_samples
    pre_roll = config.pre_roll_chirps * chirp_samples
    n_bins = config.n_bins
    bin_hz = config.bw / n_bins
    frame_sync_consistent = bool(
        result.frame_sync is not None
        and abs(
            int(result.frame_sync.fine_payload_start_sample)
            - int(candidate.data_start_sample)
        )
        <= chirp_samples // 2
    )
    common_cfo = (
        float(result.frame_sync.cfo_total_est)
        if frame_sync_consistent
        else float(candidate.coarse_cfo_bins)
    )
    common_sfo = float(result.frame_sync.sfo_hat) if frame_sync_consistent else 0.0
    common_sfo_cum = (
        float(result.frame_sync.sfo_cum_initial) if frame_sync_consistent else 0.0
    )
    combined_frame_start = (
        int(result.frame_sync.fine_preamble_start_sample)
        if frame_sync_consistent
        else int(candidate.preamble_start_sample)
    )
    combined_data_start = (
        int(result.frame_sync.fine_payload_start_sample)
        if frame_sync_consistent
        else int(candidate.data_start_sample)
    )
    boundary_refinement_source = (
        "grlora_estimate_without_gate"
        if frame_sync_consistent
        else "xcopy_soft"
    )

    rows: list[dict[str, object]] = []
    for alignment in result.alignments:
        if not alignment.included:
            continue
        raw_segment_start = (
            alignment.nominal_start_sample
            - pre_roll
            + alignment.relative_delay_samples
        )
        raw_data_start = raw_segment_start + combined_data_start
        copy_cfo = common_cfo - alignment.relative_cfo_hz / bin_hz
        cfo_int, cfo_frac = _split_cfo_bins(copy_cfo)
        downchirp = build_downchirp(config.sf, cfo_int=cfo_int, cfo_frac=cfo_frac)
        cursor = int(raw_data_start)
        sfo_cum = common_sfo_cum
        for frame_symbol_index in range(config.payload_symbols):
            stage = "header" if frame_symbol_index < int(header_symbols) else "payload"
            stage_index = (
                frame_symbol_index
                if stage == "header"
                else frame_symbol_index - int(header_symbols)
            )
            indexes = (
                cursor
                + int(config.os_factor / 2)
                + config.os_factor * np.arange(n_bins, dtype=np.int64)
            )
            if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
                break
            symbol = np.asarray(samples[indexes], dtype=np.complex64)
            spectrum = np.fft.fft(symbol * downchirp)
            power = np.abs(spectrum) ** 2
            raw_bin = int(np.argmax(power))
            second = float(np.partition(power, -2)[-2])
            peak = float(power[raw_bin])
            rows.append(
                {
                    "frame_index": int(alignment.copy_index),
                    "packet_index": int(alignment.copy_index),
                    "event_index": int(alignment.transmission_index),
                    "copy_index": int(alignment.copy_index),
                    "transmission_index": int(alignment.transmission_index),
                    "stage": stage,
                    "frame_symbol_index": int(frame_symbol_index),
                    "stage_symbol_index": int(stage_index),
                    "start_sample": int(cursor),
                    "raw_symbol_start_sample": int(cursor),
                    "raw_frame_start_sample": int(
                        raw_segment_start + combined_frame_start
                    ),
                    "raw_data_start_sample": int(raw_data_start),
                    "soft_frame_start_sample": int(
                        raw_segment_start + candidate.preamble_start_sample
                    ),
                    "soft_data_start_sample": int(
                        raw_segment_start + candidate.data_start_sample
                    ),
                    "boundary_refinement_source": boundary_refinement_source,
                    "sf": int(config.sf),
                    "bw": float(config.bw),
                    "os_factor": int(config.os_factor),
                    "cfo_int": int(cfo_int),
                    "cfo_frac": float(cfo_frac),
                    "sto_frac": 0.0,
                    "sfo_hat": float(common_sfo),
                    "sfo_cum_before": float(sfo_cum),
                    "sfo_sample_adjust_after": 0,
                    "raw_fft_bin": int(raw_bin),
                    "symbol_value": int(
                        bin_to_grlora_symbol(
                            raw_bin,
                            config.sf,
                            is_header=stage == "header",
                            ldro=bool(ldro),
                        )
                    ),
                    "peak_margin_db": float(
                        10.0 * math.log10((peak + 1e-30) / (second + 1e-30))
                    ),
                    "header_valid": 0,
                    "payload_len": int(payload_len),
                    "payload_cr": int(coding_rate),
                    "payload_has_crc": int(bool(has_crc)),
                    "payload_ldro": int(bool(ldro)),
                    "relative_delay_samples": int(alignment.relative_delay_samples),
                    "relative_cfo_hz": float(alignment.relative_cfo_hz),
                    "relative_phase_rad": float(alignment.relative_phase_rad),
                    "xcopy_alignment_score": float(alignment.peak_to_median),
                    "soft_boundary_score": float(candidate.score),
                    "soft_boundary_rank": int(candidate.rank),
                    "soft_hard_pattern_valid": int(candidate.hard_grlora_pattern_valid),
                }
            )

            step = chirp_samples
            sample_adjust = 0
            threshold = 1.0 / (2.0 * config.os_factor)
            if abs(sfo_cum) > threshold:
                sign = -1 if math.copysign(1.0, sfo_cum) < 0.0 else 1
                step -= sign
                sample_adjust = -sign
                sfo_cum -= sign * (1.0 / config.os_factor)
                rows[-1]["sfo_sample_adjust_after"] = int(sample_adjust)
            sfo_cum += common_sfo
            cursor += step
    _apply_header_bin_consensus(
        rows,
        config,
        payload_len=payload_len,
        coding_rate=coding_rate,
        has_crc=has_crc,
    )
    return rows


def _locate_combined_frame(
    combined: np.ndarray,
    config: XCopyConfig,
) -> tuple[FrameLocation, GrloraFrameSyncResult | None]:
    pre_roll = config.pre_roll_chirps * config.chirp_samples
    detector = PreambleDetectorConfig(
        sf=config.sf,
        bw=config.bw,
        samp_rate=config.samp_rate,
        win_chirps=min(config.detection_chirps, config.preamble_symbols),
        hop_samples=config.chirp_samples,
        min_periodic_peaks=2,
        bin_tol=2,
    )
    event = DetectionEvent(
        event_index=0,
        start_sample=pre_roll,
        end_sample=min(combined.size, pre_roll + config.preamble_symbols * config.chirp_samples),
        first_window_index=0,
        last_window_index=0,
        window_count=config.preamble_symbols,
        reference_bin=0,
        bin_min=0,
        bin_max=0,
        mean_peak_power=0.0,
        mean_confidence_db=0.0,
        max_peak_share=0.0,
    )
    coarse_locator = FrameLocatorConfig(
        preamble_len=float(config.preamble_symbols),
        sync_word=config.sync_word,
        search_radius_samples=max(32, config.chirp_samples // 8),
        step_samples=max(1, config.os_factor * 2),
        preamble_bin_tol=3,
        sync_bin_tol=6,
        sfd_bin_tol=6,
        min_preamble_peaks=max(3, config.preamble_symbols - 4),
        # A partially filled long detection window may fire several chirps
        # before the first complete preamble chirp.  Cover that ambiguity here;
        # the second pass below refines only the winning neighborhood.
        symbol_search_span=max(2, config.detection_chirps),
    )
    coarse = locate_frame_from_event(combined, event, detector, coarse_locator)
    fine_event = replace(event, start_sample=coarse.preamble_start_sample)
    fine_locator = replace(
        coarse_locator,
        search_radius_samples=max(8, config.os_factor * 4),
        step_samples=1,
        symbol_search_span=0,
    )
    fine = locate_frame_from_event(
        combined,
        fine_event,
        detector,
        fine_locator,
        coarse_start_sample=coarse.preamble_start_sample,
    )
    frame_sync = None
    try:
        frame_sync = run_grlora_frame_sync_validation(
            combined,
            fine,
            detector,
            float(config.preamble_symbols),
            config.sync_word,
            center_freq=config.center_freq,
        )
    except ValueError:
        frame_sync = None
    return fine, frame_sync


def run_xcopy_sync(samples: np.ndarray, config: XCopyConfig) -> XCopySyncResult:
    """Run periodic detection, Eq.(4) alignment, combining, and frame location."""

    detection = scan_periodic_preamble(samples, config)
    if not detection.detected or detection.coarse_preamble_phase_sample is None:
        return XCopySyncResult(
            status="preamble_not_detected",
            config=config,
            detection=detection,
            alignments=(),
            reference_copy_index=None,
            combined_iq=None,
            frame_location=None,
            frame_sync=None,
        )

    starts = _nominal_copy_starts(
        int(samples.size),
        int(detection.coarse_preamble_phase_sample),
        config,
    )
    if len(starts) < config.min_aligned_copies:
        return XCopySyncResult(
            status="not_enough_scheduled_copies",
            config=config,
            detection=detection,
            alignments=(),
            reference_copy_index=None,
            combined_iq=None,
            frame_location=None,
            frame_sync=None,
        )

    reference_index = _choose_reference(samples, starts, config)
    alignments = _measure_alignments(samples, starts, reference_index, config)
    alignments = _select_consistent_alignments(alignments, config)
    if sum(int(item.included) for item in alignments) < config.min_aligned_copies:
        return XCopySyncResult(
            status="not_enough_consistent_copies",
            config=config,
            detection=detection,
            alignments=tuple(alignments),
            reference_copy_index=reference_index,
            combined_iq=None,
            frame_location=None,
            frame_sync=None,
        )

    combined = _combine_aligned_copies(samples, alignments, config)
    soft_candidates = locate_xcopy_soft_frame_candidates(combined, config)
    try:
        frame_location, frame_sync = _locate_combined_frame(combined, config)
    except ValueError:
        frame_location = None
        frame_sync = None
    if frame_location is not None and frame_location.valid:
        status = "ok"
    elif any(_soft_frame_candidate_usable(item, config) for item in soft_candidates):
        status = "ok_soft_boundary"
    else:
        status = "combined_frame_not_located"
    return XCopySyncResult(
        status=status,
        config=config,
        detection=detection,
        alignments=tuple(alignments),
        reference_copy_index=reference_index,
        combined_iq=combined,
        frame_location=frame_location,
        frame_sync=frame_sync,
        soft_frame_candidates=soft_candidates,
    )


def run_xcopy_paper_sync(samples: np.ndarray, config: XCopyConfig) -> XCopySyncResult:
    """Run the paper-faithful per-packet detector before Eq. (4) alignment."""

    paper_config = replace(
        config,
        alignment_search_samples=max(
            int(config.alignment_search_samples),
            2 * config.chirp_samples,
        ),
    )
    packet_detections = scan_xcopy_packet_preambles(samples, paper_config)
    bins = tuple(
        XCopyDetectionBin(
            phase_index=item.detection_index,
            phase_sample=item.coarse_preamble_start_sample,
            copy_count=1,
            peak_bin=item.selected_peak_bin,
            signed_peak_bin=item.selected_signed_peak_bin,
            peak_power=item.selected_peak_power,
            median_power=item.selected_peak_power / max(item.score, 1e-30),
            peak_to_median=item.score,
            selected=True,
        )
        for item in packet_detections
    )
    best = max(packet_detections, key=lambda item: item.score) if packet_detections else None
    detection = XCopyDetection(
        detected=len(packet_detections) >= paper_config.min_aligned_copies,
        coarse_preamble_phase_sample=(
            int(packet_detections[0].coarse_preamble_start_sample)
            if packet_detections
            else None
        ),
        best_phase_sample=int(best.coarse_preamble_start_sample) if best is not None else 0,
        best_peak_to_median=float(best.score) if best is not None else 0.0,
        # Paper mode gates on the stable same-frequency run above.  The
        # peak/second-peak ratio is retained as a diagnostic, not a second
        # hidden packet-acceptance threshold.
        threshold=0.0,
        noise_median=0.0,
        noise_mad=0.0,
        run_length=len(packet_detections),
        bins=bins,
    )
    if len(packet_detections) < paper_config.min_aligned_copies:
        return XCopySyncResult(
            status="not_enough_individually_detected_packets",
            config=paper_config,
            detection=detection,
            alignments=(),
            reference_copy_index=None,
            combined_iq=None,
            frame_location=None,
            frame_sync=None,
            packet_detections=packet_detections,
        )

    starts = [
        (item.detection_index, item.coarse_preamble_start_sample)
        for item in packet_detections
    ]
    reference_index = max(
        range(len(packet_detections)),
        key=lambda index: packet_detections[index].score,
    )
    alignments = _measure_alignments(
        samples,
        starts,
        reference_index,
        paper_config,
    )
    # XCopy groups copies by the Eq. (4) whole-packet peak itself. A linear
    # timing/CFO trajectory is specific to the periodic Branch4 shortcut and
    # is deliberately not required in paper-faithful mode.
    alignments = [
        replace(
            item,
            included=bool(
                item.is_reference
                or item.peak_to_median >= paper_config.min_alignment_peak_to_median
            ),
        )
        for item in alignments
    ]
    if sum(int(item.included) for item in alignments) < paper_config.min_aligned_copies:
        return XCopySyncResult(
            status="not_enough_eq4_grouped_copies",
            config=paper_config,
            detection=detection,
            alignments=tuple(alignments),
            reference_copy_index=reference_index,
            combined_iq=None,
            frame_location=None,
            frame_sync=None,
            packet_detections=packet_detections,
        )

    combined = _combine_aligned_copies(samples, alignments, paper_config)
    soft_candidates = locate_xcopy_soft_frame_candidates(combined, paper_config)
    try:
        frame_location, frame_sync = _locate_combined_frame(combined, paper_config)
    except ValueError:
        frame_location = None
        frame_sync = None
    if frame_location is not None and frame_location.valid:
        status = "ok"
    elif any(_soft_frame_candidate_usable(item, paper_config) for item in soft_candidates):
        status = "ok_soft_boundary"
    else:
        status = "combined_frame_not_located"
    return XCopySyncResult(
        status=status,
        config=paper_config,
        detection=detection,
        alignments=tuple(alignments),
        reference_copy_index=reference_index,
        combined_iq=combined,
        frame_location=frame_location,
        frame_sync=frame_sync,
        soft_frame_candidates=soft_candidates,
        packet_detections=packet_detections,
    )
