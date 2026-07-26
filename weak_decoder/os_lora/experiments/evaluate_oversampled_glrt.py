#!/usr/bin/env python3
"""Evaluate branch-GLS plus full-rate dual-peak reranking against baselines.

Two input modes are supported:

* named clean captures with optional injected AWGN (``--datasets``);
* an explicitly synchronized capture with optional external frozen GT.

All methods see the same IQ realization. Payload GT is attached only after
demodulation for SER bookkeeping; covariance and phase/frequency models use
off-packet samples and pre-payload symbols respectively.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.common import (  # noqa: E402
    DEFAULT_DATASETS,
    dataset_paths,
    load_packets,
    noise_samples,
    signal_reference_power,
    write_csv,
)
from weak_decoder.baselines.run_ser_comparison import (  # noqa: E402
    _evaluate_loratrimmer_packet,
    _evaluate_symfec_packet,
)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.baselines.symfec import SymFECConfig  # noqa: E402
from weak_decoder.baselines.unichirp.evaluate_unichirp import (  # noqa: E402
    _evaluate_unichirp_packet,
)
from weak_decoder.baselines.unichirp.paper_unichirp_demod import (  # noqa: E402
    UniChirpDemodConfig,
)
from weak_decoder.os_lora.experiment_support.noise_windows import (  # noqa: E402
    active_intervals,
    covariance_correlation_stats,
    off_packet_starts,
)
from weak_decoder.os_lora.system.oversampled_glrt import (  # noqa: E402
    BranchNoiseModel,
    FoldTimingModel,
    HeaderBinCalibration,
    LinearFrequencyModel,
    LinearPhaseModel,
    PairNoiseModel,
    aligned_branch_observations,
    branch_gls_scores,
    rerank_coherent_fold_candidates,
    estimate_branch_noise_model,
    estimate_branch_steering,
    estimate_fractional_peak_offset,
    estimate_header_bin_correction,
    estimate_pair_noise_model,
    extract_full_rate_dechirped,
    fit_frequency_line,
    fit_fold_timing_model,
    fit_phase_line,
    full_rate_spectrum,
    identity_branch_noise_model,
    observe_dual_peak,
    observe_known_dual_peak_pair,
    rerank_dual_peak_candidates,
)


METHODS = (
    "ordinary_fft",
    "savaux",
    "savaux_header",
    "branch_gls",
    "branch_shrinkage",
    "savaux_dual",
    "proposed",
    "unichirp",
    "symfec",
    "loratrimmer",
)

METHOD_LABELS = {
    "ordinary_fft": "Nyquist FFT",
    "savaux": "Savaux",
    "savaux_header": "Savaux + header calibration",
    "branch_gls": "Branch GLS",
    "branch_shrinkage": "Branch shrinkage",
    "savaux_dual": "Legacy Savaux + pair GLRT",
    "proposed": "Branch GLS + coherent ratio",
    "unichirp": "UniChirp",
    "symfec": "Sym-FEC",
    "loratrimmer": "LoRaTrimmer",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_32"])
    parser.add_argument("--input-iq", type=Path, default=None)
    parser.add_argument("--symbols", type=Path, default=None)
    parser.add_argument("--groundtruth", type=Path, default=None)
    parser.add_argument("--name", default="explicit_capture")
    parser.add_argument("--noise-iq", type=Path, default=None)
    parser.add_argument("--noise-sync", type=Path, default=None)
    parser.add_argument("--snrs", nargs="*", type=float, default=[-22.0, -23.0, -24.0, -25.0, -26.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--noise-shape",
        choices=("white", "lowpass", "ar1"),
        default="white",
        help="Shape injected noise at ADC rate; ar1 is a colored-noise stress case.",
    )
    parser.add_argument("--noise-filter-taps", type=int, default=129)
    parser.add_argument("--noise-color-magnitude", type=float, default=0.85)
    parser.add_argument("--noise-color-phase-rad", type=float, default=0.7)
    parser.add_argument("--max-packets", type=int, default=10)
    parser.add_argument(
        "--preamble-len",
        type=int,
        default=None,
        help="Override the preamble length; otherwise infer it from the IQ name.",
    )
    parser.add_argument("--noise-windows", type=int, default=128)
    parser.add_argument("--noise-training-bins", type=int, default=16)
    parser.add_argument("--noise-seed", type=int, default=4107)
    parser.add_argument("--branch-loading", type=float, default=0.50)
    parser.add_argument(
        "--gls-score-weight",
        type=float,
        default=0.25,
        help=(
            "Convex GLS weight used only by the branch_shrinkage ablation; "
            "the proposed receiver uses the pure GLS statistic."
        ),
    )
    parser.add_argument(
        "--branch-covariance-mode",
        choices=("pooled", "per_bin"),
        default="per_bin",
    )
    parser.add_argument(
        "--min-branch-correlation",
        type=float,
        default=0.05,
        help="Use exact Savaux/white weights below this mean branch correlation.",
    )
    parser.add_argument("--min-branch-diagonal-cv", type=float, default=0.05)
    parser.add_argument("--pair-loading", type=float, default=0.10)
    parser.add_argument(
        "--pair-covariance-mode",
        choices=("pooled", "per_bin"),
        default="per_bin",
    )
    parser.add_argument("--top-l", type=int, default=8)
    parser.add_argument(
        "--branch-steering",
        choices=("ideal", "header", "preamble", "preamble_header"),
        default="preamble_header",
    )
    parser.add_argument("--min-steering-rank-one-fraction", type=float, default=0.80)
    parser.add_argument("--disable-header-bin-calibration", action="store_true")
    parser.add_argument("--min-header-bin-consensus", type=float, default=0.75)
    parser.add_argument("--consistency-weight", type=float, default=0.30)
    parser.add_argument(
        "--coherence-weight",
        type=float,
        default=0.30,
        help="Weight of the normalized coherent-fold ratio in the joint log score.",
    )
    parser.add_argument(
        "--coherent-rerank-mode",
        choices=("joint", "coherence", "confidence_gate"),
        default="confidence_gate",
        help=(
            "Use the largest coherence among near-tied GLS candidates by "
            "default; coherence-only and joint-log modes are ablations."
        ),
    )
    parser.add_argument("--min-coherence-gain", type=float, default=0.30)
    parser.add_argument(
        "--max-coherence-override-loss-db",
        type=float,
        default=0.15,
    )
    parser.add_argument("--max-branch-loss-db", type=float, default=3.0)
    parser.add_argument("--phase-rmse-scale", type=float, default=0.60)
    parser.add_argument("--min-phase-observations", type=int, default=4)
    parser.add_argument(
        "--frequency-mode", choices=("integer", "constant", "line"), default="constant"
    )
    parser.add_argument(
        "--phase-slope-mode", choices=("free", "sfo-pi", "sfo-2pi"), default="free"
    )
    parser.add_argument(
        "--amplitude-model", choices=("exact", "fold", "equal"), default="exact"
    )
    parser.add_argument(
        "--timing-slope-mode", choices=("zero", "minus-sfo", "plus-sfo"), default="zero"
    )
    parser.add_argument("--timing-contrast-scale", type=float, default=0.02)
    parser.add_argument(
        "--rerank-mode",
        choices=("confidence_gate", "weighted"),
        default="confidence_gate",
    )
    parser.add_argument("--min-consistency-gain", type=float, default=0.12)
    parser.add_argument("--max-override-loss-db", type=float, default=0.10)
    parser.add_argument("--min-model-reliability", type=float, default=0.92)
    parser.add_argument(
        "--allow-white-fold-overrides",
        action="store_true",
        help=(
            "Permit fold-based decision changes even when off-packet branch "
            "noise is statistically white. By default the full-rate module "
            "remains diagnostic-only in that matched-filter regime."
        ),
    )
    parser.add_argument("--disable-pair-whitening", action="store_true")
    parser.add_argument("--skip-loratrimmer", action="store_true")
    parser.add_argument("--skip-symfec", action="store_true")
    parser.add_argument("--skip-unichirp", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "experiments" / "oversampled_glrt",
    )
    return parser.parse_args()


def _external_gt(path: Path) -> dict[int, int]:
    output: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output[int(row["frame_symbol_index"])] = int(row["groundtruth_fft_bin"])
    if not output:
        raise RuntimeError(f"no ground-truth rows found in {path}")
    return output


def _attach_gt(packets: Sequence[dict[str, Any]], gt: dict[int, int]) -> None:
    for packet in packets:
        for symbol in packet["payload_symbols"]:
            frame_index = int(symbol["frame_symbol_index"])
            if frame_index not in gt:
                raise KeyError(f"ground truth has no frame_symbol_index={frame_index}")
            symbol["gt_bin"] = int(gt[frame_index])


def _infer_preamble_len(iq_path: Path, dataset: str) -> int:
    match = re.search(r"(?:^|_)pre(\d+)(?:_|$)", iq_path.stem, flags=re.IGNORECASE)
    if match is not None:
        return int(match.group(1))
    final = str(dataset).rsplit("_", maxsplit=1)[-1]
    return int(final) if final.isdigit() else 8


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for start, stop in sorted(intervals):
        if stop <= start:
            continue
        if not output or start > output[-1][1]:
            output.append((int(start), int(stop)))
        else:
            output[-1] = (output[-1][0], max(output[-1][1], int(stop)))
    return output


def _sync_intervals(
    path: Path,
    sample_count: int,
    symbol_samples: int,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text = str(row.get("detected_start_sample", "")).strip()
            if not text:
                continue
            start = int(float(text))
            intervals.append(
                (
                    max(0, start - 8 * symbol_samples),
                    min(int(sample_count), start + 110 * symbol_samples),
                )
            )
    if not intervals:
        raise RuntimeError(f"no detected events found in {path}")
    return _merge_intervals(intervals)


def _noise_windows(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    sf: int,
    os_factor: int,
    count: int,
    seed: int,
    sync_path: Path | None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    length = (1 << int(sf)) * int(os_factor)
    intervals = (
        _sync_intervals(sync_path, int(samples.size), length)
        if sync_path is not None
        else active_intervals(packets, guard_samples=16 * length)
    )
    starts = off_packet_starts(
        sample_count=int(samples.size),
        window_len=length,
        intervals=intervals,
        max_windows=int(count),
        seed=int(seed),
    )
    if len(starts) < 2:
        raise RuntimeError("fewer than two off-packet noise windows are available")
    windows = np.asarray(
        [samples[start : start + length] for start in starts], dtype=np.complex64
    )
    return windows, starts


def _absolute_header_index(packet: dict[str, Any], index: int) -> float:
    return float(packet.get("preamble_len", 8.0)) + 4.25 + float(index)


def _absolute_payload_index(packet: dict[str, Any], index: int) -> float:
    return float(packet.get("preamble_len", 8.0)) + 12.25 + float(index)


def _training_models(
    samples: np.ndarray,
    packet: dict[str, Any],
    frequency_mode: str,
    phase_slope_mode: str,
    timing_slope_mode: str,
    pair_noise_model: PairNoiseModel | None,
) -> tuple[LinearFrequencyModel, LinearPhaseModel, FoldTimingModel]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    origin_shift = os_factor // 2
    header_start = int(packet["header_start_sample"]) + origin_shift
    indexes: list[float] = []
    offsets: list[float] = []
    weights: list[float] = []
    dechirped_symbols: list[tuple[np.ndarray, int, float, str]] = []
    symbol_samples = (1 << sf) * os_factor
    preamble_len = int(packet.get("preamble_len", 8))
    frame_start = int(
        round(
            float(packet["header_start_sample"])
            - (float(preamble_len) + 4.25) * float(symbol_samples)
        )
    )
    training_entries: list[tuple[int, int, float, str]] = [
        (frame_start + index * symbol_samples, 0, float(index), "preamble")
        for index in range(preamble_len)
    ]
    training_entries.extend(
        (
            int(symbol["start_sample"]),
            int(symbol["raw_fft_bin"]),
            _absolute_header_index(packet, int(symbol["stage_symbol_index"])),
            "header",
        )
        for symbol in packet["header_symbols"]
    )
    for start_sample, raw_bin, index, source in training_entries:
        try:
            dechirped = extract_full_rate_dechirped(
                samples,
                int(start_sample) + origin_shift,
                sf,
                os_factor,
                int(packet["cfo_int"]),
                float(packet["cfo_frac"]),
                header_start,
                "continuous",
            )
        except ValueError:
            continue
        offset, peak_power = estimate_fractional_peak_offset(
            full_rate_spectrum(dechirped), int(raw_bin)
        )
        indexes.append(index)
        offsets.append(offset)
        weights.append(peak_power)
        dechirped_symbols.append((dechirped, int(raw_bin), index, source))
    if not indexes:
        return (
            LinearFrequencyModel(0.0, 0.0, 0, float("inf")),
            LinearPhaseModel(0.0, 0.0, 0, float("inf")),
            FoldTimingModel(0.0, 0.0, 0.0, 0, 0.0, 0.0),
        )
    frequency = fit_frequency_line(
        indexes,
        offsets,
        weights,
        fixed_slope_bins_per_symbol=0.0 if str(frequency_mode) == "constant" else None,
    )
    observations = [
        observe_dual_peak(
            dechirped,
            sf,
            os_factor,
            int(raw_bin),
            float(np.clip(frequency.predict(index), -0.5, 0.5)),
            index,
        )
        for dechirped, raw_bin, index, _source in dechirped_symbols
    ]
    sfo_hat = float(
        packet["header_symbols"][0].get(
            "sfo_hat", packet["payload_symbols"][0].get("sfo_hat", 0.0)
        )
    )
    if str(phase_slope_mode) == "sfo-pi":
        fixed_phase_slope: float | None = -math.pi * sfo_hat
    elif str(phase_slope_mode) == "sfo-2pi":
        fixed_phase_slope = -2.0 * math.pi * sfo_hat
    else:
        fixed_phase_slope = None
    phase = fit_phase_line(
        observations, fixed_slope_rad_per_symbol=fixed_phase_slope
    )
    if str(timing_slope_mode) == "minus-sfo":
        timing_slope = -sfo_hat
    elif str(timing_slope_mode) == "plus-sfo":
        timing_slope = sfo_hat
    else:
        timing_slope = 0.0
    complex_observations = [
        observe_known_dual_peak_pair(
            dechirped,
            sf,
            os_factor,
            int(raw_bin),
            float(np.clip(frequency.predict(index), -0.5, 0.5)),
            index,
        )
        for dechirped, raw_bin, index, source in dechirped_symbols
        if source == "header"
    ]
    timing = fit_fold_timing_model(
        complex_observations,
        sf,
        os_factor,
        slope_chips_per_symbol=timing_slope,
        pair_noise_model=None,
    )
    return frequency, phase, timing


def _phase_reliability(
    model: LinearPhaseModel,
    minimum_observations: int,
    rmse_scale: float,
) -> float:
    if model.observation_count <= 0 or not math.isfinite(model.rmse_rad):
        return 0.0
    count_factor = min(1.0, float(model.observation_count) / max(1.0, float(minimum_observations)))
    scale = max(float(rmse_scale), 1e-6)
    return float(count_factor * math.exp(-((float(model.rmse_rad) / scale) ** 2)))


def _select_active_branch_steering(
    branch_model: BranchNoiseModel,
    estimated_steering: np.ndarray,
    steering_quality: float,
    steering_mode: str,
    minimum_rank_one_fraction: float,
) -> np.ndarray | None:
    """Keep exact Savaux steering when the white-noise gate is active."""

    if int(branch_model.snapshot_count) <= 0:
        return None
    if str(steering_mode) not in {"header", "preamble", "preamble_header"}:
        return None
    if float(steering_quality) < float(minimum_rank_one_fraction):
        return None
    return np.asarray(estimated_steering, dtype=np.complex128)


def _header_branch_steering(
    samples: np.ndarray,
    packet: dict[str, Any],
    include_preamble: bool = False,
    include_header: bool = True,
) -> tuple[np.ndarray, float, int]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    origin_shift = os_factor // 2
    header_start = int(packet["header_start_sample"]) + origin_shift
    observations: list[np.ndarray] = []
    if include_preamble:
        symbol_samples = (1 << sf) * os_factor
        preamble_len = int(packet.get("preamble_len", 8))
        frame_start = int(
            round(
                float(packet["header_start_sample"])
                - (float(preamble_len) + 4.25) * float(symbol_samples)
            )
        )
        for preamble_index in range(preamble_len):
            try:
                _combined, branches, _ = paper_oversampled_spectrum(
                    samples,
                    frame_start + preamble_index * symbol_samples + origin_shift,
                    sf,
                    os_factor,
                    int(packet["cfo_int"]),
                    float(packet["cfo_frac"]),
                    header_start,
                    "continuous",
                )
            except ValueError:
                continue
            matrix = aligned_branch_observations(branches, os_factor)
            observations.append(matrix[0])
    if include_header:
        for symbol in packet["header_symbols"]:
            try:
                _combined, branches, _ = paper_oversampled_spectrum(
                    samples,
                    int(symbol["start_sample"]) + origin_shift,
                    sf,
                    os_factor,
                    int(packet["cfo_int"]),
                    float(packet["cfo_frac"]),
                    header_start,
                    "continuous",
                )
            except ValueError:
                continue
            matrix = aligned_branch_observations(branches, os_factor)
            observations.append(matrix[int(symbol["raw_fft_bin"])])
    if not observations:
        return np.ones(os_factor, dtype=np.complex128), 0.0, 0
    estimate = estimate_branch_steering(np.stack(observations, axis=0))
    return estimate.steering, estimate.rank_one_fraction, estimate.observation_count


def _header_bin_calibration(
    samples: np.ndarray,
    packet: dict[str, Any],
    minimum_consensus: float,
) -> HeaderBinCalibration:
    """Estimate a packet-wide +/-1 bin correction from explicit-header groups."""

    sf = int(packet["sf"])
    n_bins = 1 << sf
    os_factor = int(packet["os_factor"])
    origin_shift = os_factor // 2
    header_start = int(packet["header_start_sample"]) + origin_shift
    observed_bins: list[int] = []
    for symbol in packet["header_symbols"]:
        try:
            combined, _branches, _ = paper_oversampled_spectrum(
                samples,
                int(symbol["start_sample"]) + origin_shift,
                sf,
                os_factor,
                int(packet["cfo_int"]),
                float(packet["cfo_frac"]),
                header_start,
                "continuous",
            )
        except ValueError:
            continue
        observed = int(np.argmax(np.abs(combined).astype(np.float64) ** 2))
        observed_bins.append(observed)
    return estimate_header_bin_correction(
        observed_bins,
        n_bins,
        minimum_consensus,
    )


def _timing_reliability(
    model: FoldTimingModel,
    minimum_observations: int,
    contrast_scale: float,
) -> float:
    if model.observation_count <= 0:
        return 0.0
    count_factor = min(
        1.0,
        float(model.observation_count) / max(1.0, float(minimum_observations)),
    )
    contrast_factor = float(
        np.clip(model.grid_contrast / max(float(contrast_scale), 1e-9), 0.0, 1.0)
    )
    return float(count_factor * contrast_factor * np.clip(model.mean_consistency, 0.0, 1.0))


def _model_rows(
    branch: BranchNoiseModel,
    pair: PairNoiseModel,
) -> list[dict[str, Any]]:
    branch_covariance = np.asarray(branch.covariance)
    if branch_covariance.ndim == 3:
        branch_stats = np.asarray(
            [covariance_correlation_stats(item) for item in branch_covariance],
            dtype=np.float64,
        )
        branch_cv, branch_mean_corr, branch_max_corr = (
            float(np.mean(branch_stats[:, 0])),
            float(np.mean(branch_stats[:, 1])),
            float(np.max(branch_stats[:, 2])),
        )
    else:
        branch_cv, branch_mean_corr, branch_max_corr = covariance_correlation_stats(
            branch_covariance
        )
    pair_covariance = np.asarray(pair.covariance)
    if pair_covariance.ndim == 3:
        pair_stats = np.asarray(
            [covariance_correlation_stats(item) for item in pair_covariance],
            dtype=np.float64,
        )
        pair_cv, pair_mean_corr, pair_max_corr = (
            float(np.mean(pair_stats[:, 0])),
            float(np.mean(pair_stats[:, 1])),
            float(np.max(pair_stats[:, 2])),
        )
    else:
        pair_cv, pair_mean_corr, pair_max_corr = covariance_correlation_stats(
            pair_covariance
        )
    return [
        {
            "branch_dimension": int(branch.covariance.shape[-1]),
            "branch_covariance_count": int(
                branch.covariance.shape[0] if branch.covariance.ndim == 3 else 1
            ),
            "branch_snapshots": int(branch.snapshot_count),
            "branch_diagonal_cv": branch_cv,
            "branch_mean_abs_correlation": branch_mean_corr,
            "branch_max_abs_correlation": branch_max_corr,
            "pair_dimension": int(pair.covariance.shape[-1]),
            "pair_covariance_count": int(
                pair.covariance.shape[0] if pair.covariance.ndim == 3 else 1
            ),
            "pair_snapshots": int(pair.snapshot_count),
            "pair_diagonal_cv": pair_cv,
            "pair_mean_abs_correlation": pair_mean_corr,
            "pair_max_abs_correlation": pair_max_corr,
        }
    ]


def _evaluate_group(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    branch_models: dict[int, BranchNoiseModel],
    pair_models: dict[int, PairNoiseModel | None],
    args: argparse.Namespace,
    dataset: str,
    snr_db: float | None,
    seed: int,
    fold_override_active: dict[int, bool],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    symbol_rows: list[dict[str, Any]] = []
    totals = {method: {"errors": 0, "symbols": 0, "fixes": 0, "breaks": 0} for method in METHODS}
    unichirp_config = UniChirpDemodConfig()
    symfec_config = SymFECConfig()
    for packet in packets:
        packet_index = int(packet["packet_index"])
        branch_model = branch_models[packet_index]
        pair_model = pair_models[packet_index]
        packet_fold_override_active = bool(fold_override_active[packet_index])
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        origin_shift = os_factor // 2
        header_start = int(packet["header_start_sample"]) + origin_shift
        frequency_model, phase_model, timing_model = _training_models(
            samples,
            packet,
            str(args.frequency_mode),
            str(args.phase_slope_mode),
            str(args.timing_slope_mode),
            pair_model,
        )
        phase_reliability = _phase_reliability(
            phase_model,
            int(args.min_phase_observations),
            float(args.phase_rmse_scale),
        )
        legacy_reliability = (
            _timing_reliability(
                timing_model,
                int(args.min_phase_observations),
                float(args.timing_contrast_scale),
            )
            if str(args.amplitude_model) == "exact"
            else phase_reliability
        )
        adaptive_weight = float(args.consistency_weight) * legacy_reliability
        coherence_reliability = _timing_reliability(
            timing_model,
            int(args.min_phase_observations),
            float(args.timing_contrast_scale),
        )
        adaptive_coherence_weight = (
            float(args.coherence_weight) * coherence_reliability
        )
        effective_consistency_gain = (
            float(args.min_consistency_gain)
            if legacy_reliability >= float(args.min_model_reliability)
            else float("inf")
        )
        estimated_steering, steering_quality, steering_count = _header_branch_steering(
            samples,
            packet,
            include_preamble=str(args.branch_steering)
            in {"preamble", "preamble_header"},
            include_header=str(args.branch_steering)
            in {"header", "preamble_header"},
        )
        active_steering = _select_active_branch_steering(
            branch_model,
            estimated_steering,
            steering_quality,
            str(args.branch_steering),
            float(args.min_steering_rank_one_fraction),
        )
        header_calibration = (
            _header_bin_calibration(
                samples,
                packet,
                float(args.min_header_bin_consensus),
            )
            if not bool(args.disable_header_bin_calibration)
            else HeaderBinCalibration(0, 0.0, 0, 0)
        )
        header_bin_correction = int(header_calibration.correction_bins)
        header_bin_consensus = float(header_calibration.consensus)
        header_bin_observations = int(header_calibration.observation_count)
        packet_symbol_count = 0
        for symbol in packet["payload_symbols"]:
            try:
                combined, branches, _ = paper_oversampled_spectrum(
                    samples,
                    int(symbol["start_sample"]) + origin_shift,
                    sf,
                    os_factor,
                    int(packet["cfo_int"]),
                    float(packet["cfo_frac"]),
                    header_start,
                    "continuous",
                )
                dechirped = extract_full_rate_dechirped(
                    samples,
                    int(symbol["start_sample"]) + origin_shift,
                    sf,
                    os_factor,
                    int(packet["cfo_int"]),
                    float(packet["cfo_frac"]),
                    header_start,
                    "continuous",
                )
            except ValueError:
                continue
            savaux_power = np.abs(combined).astype(np.float64) ** 2
            savaux_bin = int(np.argmax(savaux_power))
            ordinary_bin = int(np.argmax(np.abs(branches[0]).astype(np.float64) ** 2))
            gls = branch_gls_scores(
                branches,
                os_factor,
                noise_model=branch_model,
                top_l=int(args.top_l),
                steering=active_steering,
            )
            gls_weight = float(np.clip(args.gls_score_weight, 0.0, 1.0))
            normalized_gls = gls.scores / max(float(np.max(gls.scores)), 1e-30)
            normalized_savaux = savaux_power / max(float(np.max(savaux_power)), 1e-30)
            hybrid_scores = (
                gls_weight * normalized_gls
                + (1.0 - gls_weight) * normalized_savaux
            )
            hybrid_bin = int(np.argmax(hybrid_scores))
            absolute_index = _absolute_payload_index(
                packet, int(symbol["payload_symbol_index"])
            )
            predicted_frequency = (
                float(np.clip(frequency_model.predict(absolute_index), -0.5, 0.5))
                if str(args.frequency_mode) in {"constant", "line"}
                else 0.0
            )
            predicted_phase = phase_model.predict(absolute_index)
            predicted_timing = timing_model.predict(absolute_index)
            savaux_dual = rerank_dual_peak_candidates(
                dechirped,
                savaux_power,
                sf,
                os_factor,
                predicted_phase,
                predicted_frequency,
                pair_model,
                int(args.top_l),
                adaptive_weight,
                float(args.max_branch_loss_db),
                str(args.amplitude_model),
                predicted_timing,
                str(args.rerank_mode),
                effective_consistency_gain,
                float(args.max_override_loss_db),
                allow_override=packet_fold_override_active,
            )
            proposed = rerank_coherent_fold_candidates(
                dechirped,
                gls.scores,
                sf,
                os_factor,
                predicted_phase,
                predicted_frequency,
                predicted_timing,
                int(args.top_l),
                adaptive_coherence_weight,
                float(args.max_branch_loss_db),
                str(args.coherent_rerank_mode),
                float(args.min_coherence_gain),
                float(args.max_coherence_override_loss_db),
                allow_override=(
                    packet_fold_override_active
                    and int(timing_model.observation_count)
                    >= int(args.min_phase_observations)
                ),
            )
            selected = {
                "ordinary_fft": ordinary_bin,
                "savaux": savaux_bin,
                "savaux_header": int(
                    (savaux_bin + header_bin_correction) % (1 << sf)
                ),
                "branch_gls": int(gls.selected_bin),
                "branch_shrinkage": hybrid_bin,
                "savaux_dual": int(savaux_dual.selected_bin),
                "proposed": int(
                    (int(proposed.selected_bin) + header_bin_correction) % (1 << sf)
                ),
            }
            gt = int(symbol["gt_bin"])
            for method, raw_bin in selected.items():
                totals[method]["errors"] += int(raw_bin != gt)
                totals[method]["symbols"] += 1
                totals[method]["fixes"] += int(savaux_bin != gt and raw_bin == gt)
                totals[method]["breaks"] += int(savaux_bin == gt and raw_bin != gt)
            gls_order = np.argsort(gls.scores)[::-1]
            savaux_order = np.argsort(savaux_power)[::-1]
            selected_detail = next(
                item
                for item in proposed.candidate_scores
                if item.raw_bin == proposed.selected_bin
            )
            symbol_rows.append(
                {
                    "dataset": dataset,
                    "snr_db": "" if snr_db is None else float(snr_db),
                    "seed": int(seed),
                    "packet_index": int(packet["packet_index"]),
                    "payload_symbol_index": int(symbol["payload_symbol_index"]),
                    "frame_symbol_index": int(symbol["frame_symbol_index"]),
                    "gt_bin": gt,
                    "ordinary_fft_bin": ordinary_bin,
                    "savaux_bin": savaux_bin,
                    "savaux_header_bin": selected["savaux_header"],
                    "branch_gls_bin": int(gls.selected_bin),
                    "branch_shrinkage_bin": hybrid_bin,
                    "savaux_dual_bin": int(savaux_dual.selected_bin),
                    "proposed_observed_bin": int(proposed.selected_bin),
                    "proposed_bin": selected["proposed"],
                    "header_bin_correction": header_bin_correction,
                    "header_bin_consensus": header_bin_consensus,
                    "header_bin_observations": header_bin_observations,
                    "header_bin_residual": int(header_calibration.residual_bins),
                    "savaux_top_l_contains_gt": int(gt in savaux_order[: int(args.top_l)]),
                    "gls_top_l_contains_gt": int(gt in gls_order[: int(args.top_l)]),
                    "phase_observations": int(phase_model.observation_count),
                    "phase_rmse_rad": float(phase_model.rmse_rad),
                    "phase_reliability": coherence_reliability,
                    "adaptive_consistency_weight": adaptive_coherence_weight,
                    "effective_min_consistency_gain": effective_consistency_gain,
                    "fold_override_active": int(packet_fold_override_active),
                    "predicted_phase_rad": float(predicted_phase),
                    "predicted_fractional_bin": float(predicted_frequency),
                    "predicted_timing_offset_chips": float(predicted_timing),
                    "timing_offset_at_reference_chips": float(
                        timing_model.offset_chips_at_reference
                    ),
                    "timing_slope_chips_per_symbol": float(
                        timing_model.slope_chips_per_symbol
                    ),
                    "timing_fit_consistency": float(timing_model.mean_consistency),
                    "timing_grid_contrast": float(timing_model.grid_contrast),
                    "frequency_rmse_bins": float(frequency_model.rmse_bins),
                    "branch_steering_rank_one_fraction": float(steering_quality),
                    "branch_steering_observations": int(steering_count),
                    "branch_steering_active": int(active_steering is not None),
                    "selected_pair_consistency": float(selected_detail.consistency),
                    "selected_pair_phase_residual_rad": float(selected_detail.phase_residual_rad),
                    "gls_candidates": "|".join(str(value) for value in proposed.top_candidates),
                    "gls_candidate_consistency": "|".join(
                        f"{item.consistency:.8g}" for item in proposed.candidate_scores
                    ),
                    "gls_candidate_loss_db": "|".join(
                        f"{item.branch_loss_db:.8g}" for item in proposed.candidate_scores
                    ),
                    "gls_candidate_matched_power": "|".join(
                        f"{item.matched_power:.8g}" for item in proposed.candidate_scores
                    ),
                    "gls_candidate_primary_power": "|".join(
                        f"{item.primary_power:.8g}" for item in proposed.candidate_scores
                    ),
                    "gls_candidate_secondary_power": "|".join(
                        f"{item.secondary_power:.8g}" for item in proposed.candidate_scores
                    ),
                    "gls_candidate_phase_residual_rad": "|".join(
                        f"{item.phase_residual_rad:.8g}" for item in proposed.candidate_scores
                    ),
                }
            )
            packet_symbol_count += 1

        if packet_symbol_count <= 0:
            continue
        if not bool(args.skip_unichirp):
            result = _evaluate_unichirp_packet(
                samples, packet, unichirp_config, training_source="preamble_header"
            )
            totals["unichirp"]["errors"] += int(result["unichirp_err"])
            totals["unichirp"]["symbols"] += int(result["symbol_count"])
        if not bool(args.skip_symfec):
            result = _evaluate_symfec_packet(samples, packet, symfec_config)
            totals["symfec"]["errors"] += int(result["symfec_err"])
            totals["symfec"]["symbols"] += int(result["symbol_count"])
        if not bool(args.skip_loratrimmer):
            result = _evaluate_loratrimmer_packet(samples, packet)
            totals["loratrimmer"]["errors"] += int(result["loratrimmer_err"])
            totals["loratrimmer"]["symbols"] += int(result["symbol_count"])
    return symbol_rows, totals


def _summary_rows(
    totals: dict[str, dict[str, int]],
    dataset: str,
    snr_db: float | None,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        values = totals[method]
        symbols = int(values["symbols"])
        if symbols <= 0:
            continue
        errors = int(values["errors"])
        rows.append(
            {
                "dataset": dataset,
                "snr_db": "" if snr_db is None else float(snr_db),
                "seed": int(seed),
                "method": method,
                "errors": errors,
                "symbol_count": symbols,
                "ser": float(errors / symbols),
                "fixes_vs_savaux": int(values["fixes"]),
                "breaks_vs_savaux": int(values["breaks"]),
            }
        )
    return rows


def _aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["snr_db"]), str(row["method"]))
        item = grouped.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "snr_db": row["snr_db"],
                "method": row["method"],
                "errors": 0,
                "symbol_count": 0,
                "fixes_vs_savaux": 0,
                "breaks_vs_savaux": 0,
            },
        )
        for field in ("errors", "symbol_count", "fixes_vs_savaux", "breaks_vs_savaux"):
            item[field] += int(row[field])
    output = list(grouped.values())
    for row in output:
        row["ser"] = float(row["errors"] / max(1, row["symbol_count"]))
    output.sort(key=lambda row: (str(row["dataset"]), str(row["snr_db"]), METHODS.index(str(row["method"]))))
    return output


def _plot_ser(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    """Write one compact SER-vs-SNR PNG per named dataset."""

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional presentation output
        print(f"plot skipped: {exc}")
        return
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        selected = [
            row
            for row in rows
            if str(row["dataset"]) == dataset and str(row["snr_db"]).strip()
        ]
        if not selected:
            continue
        figure, axis = plt.subplots(figsize=(8.4, 5.0))
        for method in METHODS:
            values = sorted(
                (row for row in selected if str(row["method"]) == method),
                key=lambda row: float(row["snr_db"]),
            )
            if not values:
                continue
            axis.plot(
                [float(row["snr_db"]) for row in values],
                [float(row["ser"]) for row in values],
                marker="o",
                linewidth=2.0 if method in {"savaux", "proposed"} else 1.2,
                label=METHOD_LABELS.get(method, method),
            )
        axis.set_xlabel("Injected SNR (dB)")
        axis.set_ylabel("Payload SER")
        axis.set_title(dataset)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        safe_name = "".join(character if character.isalnum() else "_" for character in dataset)
        figure.savefig(output_dir / f"ser_{safe_name}.png", dpi=160)
        plt.close(figure)


def _dataset_inputs(args: argparse.Namespace) -> list[tuple[str, Path, Path, Path | None]]:
    if args.input_iq is not None or args.symbols is not None:
        if args.input_iq is None or args.symbols is None:
            raise ValueError("--input-iq and --symbols must be provided together")
        return [(str(args.name), args.input_iq.resolve(), args.symbols.resolve(), args.groundtruth.resolve() if args.groundtruth else None)]
    return [
        (dataset, *dataset_paths(dataset), None)
        for dataset in args.datasets
    ]


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, Any]] = []
    all_symbols: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    for dataset, iq_path, symbol_path, gt_path in _dataset_inputs(args):
        packets_all = load_packets(symbol_path)
        preamble_len = (
            int(args.preamble_len)
            if args.preamble_len is not None
            else _infer_preamble_len(iq_path, dataset)
        )
        if preamble_len <= 0:
            raise ValueError("preamble length must be positive")
        for packet in packets_all:
            packet["preamble_len"] = int(preamble_len)
        if gt_path is not None:
            _attach_gt(packets_all, _external_gt(gt_path))
        packets = packets_all[: int(args.max_packets)] if int(args.max_packets) > 0 else packets_all
        if not packets:
            raise RuntimeError(f"no payload packets found for {dataset}")
        clean = np.memmap(iq_path, dtype=np.complex64, mode="r")
        reference_power, _, _ = signal_reference_power(clean, packets, "packet", None)
        snr_values: tuple[float | None, ...] = tuple(args.snrs) if args.snrs else (None,)
        seeds = tuple(args.seeds) if args.snrs else (int(args.seeds[0]),)
        for snr_db in snr_values:
            for seed in seeds:
                samples = noise_samples(
                    clean,
                    snr_db,
                    int(seed),
                    reference_power,
                    noise_shape=str(args.noise_shape),
                    os_factor=int(packets[0]["os_factor"]),
                    filter_taps=int(args.noise_filter_taps),
                    color_magnitude=float(args.noise_color_magnitude),
                    color_phase_rad=float(args.noise_color_phase_rad),
                )
                if args.noise_iq is not None:
                    noise_source = np.memmap(args.noise_iq.resolve(), dtype=np.complex64, mode="r")
                    sync_path = args.noise_sync.resolve() if args.noise_sync is not None else None
                else:
                    noise_source = samples
                    sync_path = None
                windows, starts = _noise_windows(
                    noise_source,
                    packets_all,
                    int(packets[0]["sf"]),
                    int(packets[0]["os_factor"]),
                    int(args.noise_windows),
                    int(args.noise_seed) + int(seed),
                    sync_path,
                )
                n_bins = 1 << int(packets[0]["sf"])
                bin_count = max(1, min(int(args.noise_training_bins), n_bins))
                training_bins = tuple(
                    int(value) for value in np.linspace(0, n_bins, bin_count, endpoint=False)
                )
                branch_models: dict[int, BranchNoiseModel] = {}
                pair_models: dict[int, PairNoiseModel | None] = {}
                fold_override_active: dict[int, bool] = {}
                for packet in packets:
                    packet_index = int(packet["packet_index"])
                    estimated_branch_model = estimate_branch_noise_model(
                        windows,
                        int(packet["sf"]),
                        int(packet["os_factor"]),
                        training_bins,
                        cfo_int=int(packet["cfo_int"]),
                        cfo_frac=float(packet["cfo_frac"]),
                        diagonal_loading=float(args.branch_loading),
                        covariance_mode=str(args.branch_covariance_mode),
                    )
                    estimated_pair = estimate_pair_noise_model(
                        windows,
                        int(packet["sf"]),
                        int(packet["os_factor"]),
                        training_bins,
                        cfo_int=int(packet["cfo_int"]),
                        cfo_frac=float(packet["cfo_frac"]),
                        diagonal_loading=float(args.pair_loading),
                        covariance_mode=str(args.pair_covariance_mode),
                    )
                    cov = _model_rows(estimated_branch_model, estimated_pair)[0]
                    if str(args.branch_covariance_mode) == "per_bin":
                        gate_branch_model = estimate_branch_noise_model(
                            windows,
                            int(packet["sf"]),
                            int(packet["os_factor"]),
                            training_bins,
                            cfo_int=int(packet["cfo_int"]),
                            cfo_frac=float(packet["cfo_frac"]),
                            diagonal_loading=0.0,
                            covariance_mode="pooled",
                        )
                        gate_cov = _model_rows(
                            gate_branch_model, estimated_pair
                        )[0]
                    else:
                        gate_cov = cov
                    gate_mean_correlation = float(
                        gate_cov["branch_mean_abs_correlation"]
                    )
                    gate_diagonal_cv = float(gate_cov["branch_diagonal_cv"])
                    branch_gls_active = bool(
                        gate_mean_correlation
                        >= float(args.min_branch_correlation)
                        or gate_diagonal_cv
                        >= float(args.min_branch_diagonal_cv)
                    )
                    branch_models[packet_index] = (
                        estimated_branch_model
                        if branch_gls_active
                        else identity_branch_noise_model(int(packet["os_factor"]))
                    )
                    pair_models[packet_index] = (
                        None if bool(args.disable_pair_whitening) else estimated_pair
                    )
                    fold_override_active[packet_index] = bool(
                        branch_gls_active or args.allow_white_fold_overrides
                    )
                    cov.update(
                        {
                            "dataset": dataset,
                            "snr_db": "" if snr_db is None else float(snr_db),
                            "seed": int(seed),
                            "packet_index": packet_index,
                            "cfo_int": int(packet["cfo_int"]),
                            "cfo_frac": float(packet["cfo_frac"]),
                            "offpacket_windows": int(len(starts)),
                            "branch_gls_active": int(branch_gls_active),
                            "branch_gate_diagonal_cv": gate_diagonal_cv,
                            "branch_gate_mean_abs_correlation": gate_mean_correlation,
                            "min_branch_correlation": float(
                                args.min_branch_correlation
                            ),
                            "min_branch_diagonal_cv": float(
                                args.min_branch_diagonal_cv
                            ),
                        }
                    )
                    covariance_rows.append(cov)
                symbol_rows, totals = _evaluate_group(
                    samples,
                    packets,
                    branch_models,
                    pair_models,
                    args,
                    dataset,
                    snr_db,
                    int(seed),
                    fold_override_active,
                )
                all_symbols.extend(symbol_rows)
                group_summary = _summary_rows(totals, dataset, snr_db, int(seed))
                all_summary.extend(group_summary)
                compact = " ".join(
                    f"{row['method']}={row['ser']:.4f}" for row in group_summary
                )
                print(f"{dataset} snr={snr_db} seed={seed}: {compact}", flush=True)
                del samples
    aggregate = _aggregate(all_summary)
    write_csv(output_dir / "summary_by_seed.csv", all_summary)
    write_csv(output_dir / "summary.csv", aggregate)
    write_csv(output_dir / "symbols.csv", all_symbols)
    write_csv(output_dir / "covariance.csv", covariance_rows)
    _plot_ser(aggregate, output_dir)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
