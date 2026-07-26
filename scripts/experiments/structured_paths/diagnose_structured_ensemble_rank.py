#!/usr/bin/env python3
"""Diagnose whether structured non-uniform paths improve GT-bin evidence.

This is intentionally symbol-level: before spending time on packet-level CRC
rescues, check whether the structured path ensemble actually moves the clean
GT raw FFT bin upward in the Savaux Top-K score list.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = WEAK_ROOT / "scripts" / "experiments"
SAVAUX_RUNNER_DIR = EXPERIMENTS_DIR / "baselines" / "savaux_oversampled"
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(SAVAUX_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(SAVAUX_RUNNER_DIR))

from run_paper_oversampled_baseline import load_packets as load_savaux_packets  # noqa: E402
from run_savaux_current_threshold_sweep import DEFAULT_DATASETS, _dataset_paths, _payload_reference_power  # noqa: E402
from run_symbol_phase_threshold_sweep import _snr_values, _write_csv  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import paper_oversampled_spectrum  # noqa: E402
from weak_decoder.decoding.structured_path_demod import score_structured_path_candidates  # noqa: E402
from weak_decoder.decoding.timing_path_demod import (  # noqa: E402
    score_fixed_timing_path_candidates,
    score_timing_path_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank-diagnose structured OSR path ensemble.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snr-start", type=float, default=-24.0)
    parser.add_argument("--snr-stop", type=float, default=-24.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "structured_ensemble_rank")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--independent-noise", action="store_true")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol", "none"), default="continuous")
    parser.add_argument("--paper-origin-shift", type=int, default=None)
    parser.add_argument("--structured-path-top-k", type=int, default=8)
    parser.add_argument("--structured-path-ratio-power", type=float, default=0.20)
    parser.add_argument("--structured-score-mix", type=float, default=0.20)
    parser.add_argument("--structured-score-clip-db", type=float, default=1.5)
    parser.add_argument("--evidence-mode", choices=("structured", "timing", "packet_timing"), default="structured")
    parser.add_argument("--timing-tau-grid", default="-0.5,0,0.5")
    parser.add_argument("--timing-slope-grid", default="-0.5,0,0.5")
    parser.add_argument("--timing-path-gain-power", type=float, default=0.20)
    parser.add_argument("--timing-slope-penalty-power", type=float, default=0.10)
    parser.add_argument("--packet-timing-anchor-count", type=int, default=8)
    parser.add_argument("--rank-k", type=int, default=8)
    return parser.parse_args()


def _float_grid(text: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(text).split(",") if item.strip() != "")


def _rank(scores: np.ndarray, raw_bin: int) -> int:
    bin_i = int(raw_bin)
    if bin_i < 0 or bin_i >= scores.size:
        return -1
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    matches = np.where(order == bin_i)[0]
    return int(matches[0] + 1) if matches.size else -1


def _avg(rows: Sequence[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def _savaux_symbol_power(
    samples: np.ndarray,
    packet: dict[str, Any],
    symbol: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int, int]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    origin_shift = int(args.paper_origin_shift) if args.paper_origin_shift is not None else os_factor // 2
    start_sample = int(symbol["start_sample"]) + origin_shift
    paper_header_start = int(packet["header_start_sample"]) + origin_shift
    spectrum, _branches, _phase = paper_oversampled_spectrum(
        samples=samples,
        start_sample=start_sample,
        sf=sf,
        os_factor=os_factor,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=paper_header_start,
        cfo_correction_mode=str(args.cfo_correction_mode),
    )
    return np.abs(spectrum).astype(np.float64) ** 2, start_sample, paper_header_start, os_factor


def _packet_timing_params(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, float]:
    """Estimate one shared timing path for a packet from confident hard bins."""

    anchors: list[tuple[float, dict[str, Any], int, np.ndarray]] = []
    for symbol in packet["payload_symbols"]:
        power, _start, _header_start, _os = _savaux_symbol_power(samples, packet, symbol, args)
        order = np.argsort(power)[::-1]
        if order.size < 2:
            continue
        margin = float(np.log(power[int(order[0])] + 1e-30) - np.log(power[int(order[1])] + 1e-30))
        anchors.append((margin, symbol, int(order[0]), power))
    anchors.sort(key=lambda item: item[0], reverse=True)
    use_anchors = anchors[: max(1, int(args.packet_timing_anchor_count))]
    if not use_anchors:
        return 0.0, 0.0

    best_tau = 0.0
    best_slope = 0.0
    best_score = float("-inf")
    tau_grid = _float_grid(args.timing_tau_grid)
    slope_grid = _float_grid(args.timing_slope_grid)
    for tau0 in tau_grid:
        for slope in slope_grid:
            score = 0.0
            for _margin, symbol, hard_bin, power in use_anchors:
                candidates = score_fixed_timing_path_candidates(
                    samples=samples,
                    start_sample=int(symbol["start_sample"]) + (int(packet["os_factor"]) // 2),
                    sf=int(packet["sf"]),
                    os_factor=int(packet["os_factor"]),
                    candidate_bins=(int(hard_bin),),
                    tau0=float(tau0),
                    slope=float(slope),
                    cfo_int=int(packet["cfo_int"]),
                    cfo_frac=float(packet["cfo_frac"]),
                    header_start_sample=int(packet["header_start_sample"]) + (int(packet["os_factor"]) // 2),
                    cfo_correction_mode=str(args.cfo_correction_mode),
                    path_gain_power=float(args.timing_path_gain_power),
                    slope_penalty_power=float(args.timing_slope_penalty_power),
                    savaux_power=power,
                )
                if candidates:
                    score += math.log(float(candidates[0].timing_path_power) + 1e-30)
            if score > best_score:
                best_score = float(score)
                best_tau = float(tau0)
                best_slope = float(slope)
    return best_tau, best_slope


def _score_symbol(
    samples: np.ndarray,
    packet: dict[str, Any],
    symbol: dict[str, Any],
    args: argparse.Namespace,
    packet_timing: tuple[float, float] | None = None,
) -> dict[str, Any]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    origin_shift = int(args.paper_origin_shift) if args.paper_origin_shift is not None else os_factor // 2
    start_sample = int(symbol["start_sample"]) + origin_shift
    paper_header_start = int(packet["header_start_sample"]) + origin_shift
    gt_bin = int(symbol["gt_raw_fft_bin"])

    spectrum, _branches, _phase = paper_oversampled_spectrum(
        samples=samples,
        start_sample=start_sample,
        sf=sf,
        os_factor=os_factor,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=paper_header_start,
        cfo_correction_mode=str(args.cfo_correction_mode),
    )
    power = np.abs(spectrum).astype(np.float64) ** 2
    savaux_score = np.log(power + 1e-30)
    fused_score = savaux_score.copy()
    top_bins = np.argsort(power)[::-1][: int(args.structured_path_top_k)]
    if str(args.evidence_mode) == "structured":
        candidates = score_structured_path_candidates(
            samples=samples,
            start_sample=start_sample,
            sf=sf,
            os_factor=os_factor,
            candidate_bins=tuple(int(v) for v in top_bins),
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=paper_header_start,
            cfo_correction_mode=str(args.cfo_correction_mode),
            path_ratio_power=float(args.structured_path_ratio_power),
            savaux_power=power,
        )
        candidate_score = {
            int(item.raw_fft_bin): float(item.composite_score)
            for item in candidates
        }
        candidate_ratio = {
            int(item.raw_fft_bin): float(item.path_ratio)
            for item in candidates
        }
        candidate_name = {
            int(item.raw_fft_bin): str(item.best_path_name)
            for item in candidates
        }
    elif str(args.evidence_mode) == "timing":
        candidates = score_timing_path_candidates(
            samples=samples,
            start_sample=start_sample,
            sf=sf,
            os_factor=os_factor,
            candidate_bins=tuple(int(v) for v in top_bins),
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=paper_header_start,
            cfo_correction_mode=str(args.cfo_correction_mode),
            tau_grid=_float_grid(args.timing_tau_grid),
            slope_grid=_float_grid(args.timing_slope_grid),
            path_gain_power=float(args.timing_path_gain_power),
            slope_penalty_power=float(args.timing_slope_penalty_power),
            savaux_power=power,
        )
        candidate_score = {
            int(item.raw_fft_bin): float(item.composite_score)
            for item in candidates
        }
        candidate_ratio = {
            int(item.raw_fft_bin): float(item.timing_path_gain)
            for item in candidates
        }
        candidate_name = {
            int(item.raw_fft_bin): f"tau{item.best_tau0:g}_slope{item.best_slope:g}"
            for item in candidates
        }
    else:
        tau0, slope = packet_timing if packet_timing is not None else (0.0, 0.0)
        candidates = score_fixed_timing_path_candidates(
            samples=samples,
            start_sample=start_sample,
            sf=sf,
            os_factor=os_factor,
            candidate_bins=tuple(int(v) for v in top_bins),
            tau0=float(tau0),
            slope=float(slope),
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=paper_header_start,
            cfo_correction_mode=str(args.cfo_correction_mode),
            path_gain_power=float(args.timing_path_gain_power),
            slope_penalty_power=float(args.timing_slope_penalty_power),
            savaux_power=power,
        )
        candidate_score = {
            int(item.raw_fft_bin): float(item.composite_score)
            for item in candidates
        }
        candidate_ratio = {
            int(item.raw_fft_bin): float(item.timing_path_gain)
            for item in candidates
        }
        candidate_name = {
            int(item.raw_fft_bin): f"packet_tau{item.best_tau0:g}_slope{item.best_slope:g}"
            for item in candidates
        }
    mix = max(0.0, min(1.0, float(args.structured_score_mix)))
    clip = math.log(10.0) * max(0.0, float(args.structured_score_clip_db)) / 10.0
    gt_path_ratio = float("nan")
    gt_path_name = ""
    for raw_bin, score in candidate_score.items():
        path_log = math.log(float(score) + 1e-30)
        bonus = path_log - float(fused_score[raw_bin])
        bonus = max(-clip, min(clip, bonus))
        fused_score[raw_bin] = float(fused_score[raw_bin]) + mix * bonus
        if raw_bin == gt_bin:
            gt_path_ratio = float(candidate_ratio[raw_bin])
            gt_path_name = str(candidate_name[raw_bin])

    savaux_rank = _rank(savaux_score, gt_bin)
    fused_rank = _rank(fused_score, gt_bin)
    savaux_argmax = int(np.argmax(savaux_score))
    fused_argmax = int(np.argmax(fused_score))
    return {
        "payload_symbol_index": int(symbol["payload_symbol_index"]),
        "gt_raw_fft_bin": gt_bin,
        "savaux_argmax_bin": savaux_argmax,
        "fused_argmax_bin": fused_argmax,
        "savaux_gt_rank": savaux_rank,
        "fused_gt_rank": fused_rank,
        "rank_delta": int(savaux_rank - fused_rank) if savaux_rank > 0 and fused_rank > 0 else "",
        "savaux_gt_topk": int(0 < savaux_rank <= int(args.rank_k)),
        "fused_gt_topk": int(0 < fused_rank <= int(args.rank_k)),
        "savaux_correct": int(savaux_argmax == gt_bin),
        "fused_correct": int(fused_argmax == gt_bin),
        "gt_in_structured_topk": int(gt_bin in candidate_score),
        "gt_path_ratio": gt_path_ratio,
        "gt_path_name": gt_path_name,
        "evidence_mode": str(args.evidence_mode),
    }


def main() -> int:
    args = parse_args()
    snrs = _snr_values(args.snr_start, args.snr_stop, args.snr_step)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(args.datasets):
        paths = _dataset_paths(str(dataset))
        clean_path = paths["iq"]
        symbol_csv = paths["symbols"]
        clean_samples = np.fromfile(clean_path, dtype=np.complex64)
        packets = load_savaux_packets(symbol_csv, packet_filter=None, max_packets=None)
        ref_power, _sample_count, _packet_count = _payload_reference_power(clean_samples, symbol_csv)
        unit_noise: np.ndarray | None = None
        if not bool(args.independent_noise):
            rng = np.random.default_rng(int(args.seed) + 100000 * int(dataset_index))
            noise_i = rng.normal(0.0, 1.0, size=clean_samples.size).astype(np.float32)
            noise_q = rng.normal(0.0, 1.0, size=clean_samples.size).astype(np.float32)
            unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)
        for snr_db in snrs:
            snr_linear = 10.0 ** (float(snr_db) / 10.0)
            noise_power = ref_power / snr_linear
            if bool(args.independent_noise):
                rng = np.random.default_rng(int(args.seed) + 100000 * int(dataset_index) + int(round(abs(snr_db) * 100)))
                noise = (
                    rng.normal(scale=math.sqrt(noise_power / 2.0), size=clean_samples.shape)
                    + 1j * rng.normal(scale=math.sqrt(noise_power / 2.0), size=clean_samples.shape)
                ).astype(np.complex64)
            else:
                assert unit_noise is not None
                noise = (unit_noise * math.sqrt(noise_power / 2.0)).astype(np.complex64)
            noisy = (clean_samples + noise).astype(np.complex64)
            rows_for_curve: list[dict[str, Any]] = []
            for packet in packets:
                packet_timing = (
                    _packet_timing_params(noisy, packet, args)
                    if str(args.evidence_mode) == "packet_timing"
                    else None
                )
                for symbol in packet["payload_symbols"]:
                    row = _score_symbol(noisy, packet, symbol, args, packet_timing=packet_timing)
                    row.update(
                        {
                            "dataset": dataset,
                            "target_snr_db": float(snr_db),
                            "packet_index": int(packet["packet_index"]),
                            "packet_timing_tau0": "" if packet_timing is None else packet_timing[0],
                            "packet_timing_slope": "" if packet_timing is None else packet_timing[1],
                        }
                    )
                    symbol_rows.append(row)
                    rows_for_curve.append(row)
            summary_rows.append(
                {
                    "dataset": dataset,
                    "target_snr_db": float(snr_db),
                    "symbol_count": len(rows_for_curve),
                    "savaux_correct_rate": _avg(rows_for_curve, "savaux_correct"),
                    "fused_correct_rate": _avg(rows_for_curve, "fused_correct"),
                    "savaux_gt_topk_rate": _avg(rows_for_curve, "savaux_gt_topk"),
                    "fused_gt_topk_rate": _avg(rows_for_curve, "fused_gt_topk"),
                    "gt_in_structured_topk_rate": _avg(rows_for_curve, "gt_in_structured_topk"),
                    "mean_savaux_gt_rank": _avg(rows_for_curve, "savaux_gt_rank"),
                    "mean_fused_gt_rank": _avg(rows_for_curve, "fused_gt_rank"),
                    "mean_rank_delta": _avg(rows_for_curve, "rank_delta"),
                    "mean_gt_path_ratio": _avg(rows_for_curve, "gt_path_ratio"),
                }
            )
            print(
                f"{dataset} snr={snr_db:5.1f} "
                f"argmax {summary_rows[-1]['savaux_correct_rate']:.3f}->{summary_rows[-1]['fused_correct_rate']:.3f} "
                f"top{args.rank_k} {summary_rows[-1]['savaux_gt_topk_rate']:.3f}->{summary_rows[-1]['fused_gt_topk_rate']:.3f}"
            )

    mean_rows = []
    for snr_db in snrs:
        rows = [row for row in summary_rows if float(row["target_snr_db"]) == float(snr_db)]
        mean_rows.append(
            {
                "dataset": "mean_of_datasets",
                "target_snr_db": float(snr_db),
                "symbol_count": sum(int(row["symbol_count"]) for row in rows),
                "savaux_correct_rate": _avg(rows, "savaux_correct_rate"),
                "fused_correct_rate": _avg(rows, "fused_correct_rate"),
                "savaux_gt_topk_rate": _avg(rows, "savaux_gt_topk_rate"),
                "fused_gt_topk_rate": _avg(rows, "fused_gt_topk_rate"),
                "gt_in_structured_topk_rate": _avg(rows, "gt_in_structured_topk_rate"),
                "mean_savaux_gt_rank": _avg(rows, "mean_savaux_gt_rank"),
                "mean_fused_gt_rank": _avg(rows, "mean_fused_gt_rank"),
                "mean_rank_delta": _avg(rows, "mean_rank_delta"),
                "mean_gt_path_ratio": _avg(rows, "mean_gt_path_ratio"),
            }
        )
    _write_csv(output_dir / "symbol_rank_metrics.csv", symbol_rows)
    _write_csv(output_dir / "rank_summary.csv", summary_rows + mean_rows)
    print(f"wrote={output_dir / 'rank_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
