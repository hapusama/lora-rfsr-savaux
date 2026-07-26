#!/usr/bin/env python3
"""Paired sweep for adaptive smooth non-uniform OSR path demodulation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
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

from run_paper_oversampled_baseline import _decode_payload, _ser, load_packets as load_savaux_packets  # noqa: E402
from run_savaux_current_threshold_sweep import (  # noqa: E402
    DEFAULT_DATASETS,
    _current_default_args,
    _dataset_paths,
    _mean_of_datasets,
    _payload_reference_power,
    _threshold_tables,
)
from run_symbol_phase_threshold_sweep import (  # noqa: E402
    _evaluate_packet_methods as evaluate_current_packet,
    _snr_values,
    _threshold_from_curve,
    _write_csv,
)
from run_symbol_phase_two_stage import build_config  # noqa: E402
from run_two_stage_weak_decoder import load_packets as load_current_packets  # noqa: E402
from weak_decoder.decoding.adaptive_path_demod import demod_adaptive_path_symbol  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    demod_paper_oversampled_symbol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate adaptive smooth OSR paths against Savaux/current.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snr-start", type=float, default=-22.0)
    parser.add_argument("--snr-stop", type=float, default=-26.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "adaptive_path_sweep")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--independent-noise", action="store_true")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol", "none"), default="continuous")
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--paper-origin-shift", type=int, default=None)
    parser.add_argument("--candidate-top-k", type=int, default=16)
    parser.add_argument("--switch-penalty", type=float, default=0.40)
    parser.add_argument("--step-penalty", type=float, default=0.10)
    parser.add_argument("--path-gain-power", type=float, default=0.18)
    parser.add_argument("--switch-penalty-power", type=float, default=0.40)
    parser.add_argument("--override-margin-db", type=float, default=0.15)
    parser.add_argument("--min-savaux-rel-db", type=float, default=-4.0)
    parser.add_argument("--min-path-gain", type=float, default=1.05)
    parser.add_argument("--max-switch-rate", type=float, default=0.25)
    return parser.parse_args()


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


def _adaptive_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        crc_mode=str(args.crc_mode),
        cfo_correction_mode=str(args.cfo_correction_mode),
        ldro_mode=int(args.ldro_mode),
        paper_origin_shift=args.paper_origin_shift,
        candidate_top_k=int(args.candidate_top_k),
        switch_penalty=float(args.switch_penalty),
        step_penalty=float(args.step_penalty),
        path_gain_power=float(args.path_gain_power),
        switch_penalty_power=float(args.switch_penalty_power),
        override_margin_db=float(args.override_margin_db),
        min_savaux_rel_db=float(args.min_savaux_rel_db),
        min_path_gain=float(args.min_path_gain),
        max_switch_rate=float(args.max_switch_rate),
    )


def _evaluate_savaux_and_adaptive_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    ldro = bool(packet["ldro"])
    origin_shift = int(args.paper_origin_shift) if args.paper_origin_shift is not None else os_factor // 2
    paper_header_start = int(packet["header_start_sample"]) + origin_shift

    paper_bins: list[int] = []
    adaptive_bins: list[int] = []
    gt_bins: list[int] = []
    override_count = 0
    fix_count = 0
    break_count = 0
    path_gains: list[float] = []
    switch_rates: list[float] = []

    for symbol in packet["payload_symbols"]:
        start_sample = int(symbol["start_sample"]) + origin_shift
        gt_bin = int(symbol["gt_raw_fft_bin"])
        paper = demod_paper_oversampled_symbol(
            samples=samples,
            start_sample=start_sample,
            sf=sf,
            os_factor=os_factor,
            is_header=False,
            ldro=ldro,
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=paper_header_start,
            cfo_correction_mode=str(args.cfo_correction_mode),
        )
        adaptive = demod_adaptive_path_symbol(
            samples=samples,
            start_sample=start_sample,
            sf=sf,
            os_factor=os_factor,
            is_header=False,
            ldro=ldro,
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=paper_header_start,
            cfo_correction_mode=str(args.cfo_correction_mode),
            candidate_top_k=int(args.candidate_top_k),
            switch_penalty=float(args.switch_penalty),
            step_penalty=float(args.step_penalty),
            path_gain_power=float(args.path_gain_power),
            switch_penalty_power=float(args.switch_penalty_power),
            override_margin_db=float(args.override_margin_db),
            min_savaux_rel_db=float(args.min_savaux_rel_db),
            min_path_gain=float(args.min_path_gain),
            max_switch_rate=float(args.max_switch_rate),
        )
        paper_bin = int(paper.raw_fft_bin)
        adaptive_bin = int(adaptive.raw_fft_bin)
        paper_bins.append(paper_bin)
        adaptive_bins.append(adaptive_bin)
        gt_bins.append(gt_bin)
        if bool(adaptive.selected_by_path_override):
            override_count += 1
            if paper_bin != gt_bin and adaptive_bin == gt_bin:
                fix_count += 1
            if paper_bin == gt_bin and adaptive_bin != gt_bin:
                break_count += 1
        if adaptive.candidate_path_gains:
            selected_idx = adaptive.candidate_bins.index(adaptive_bin)
            path_gains.append(float(adaptive.candidate_path_gains[selected_idx]))
            switch_rates.append(float(adaptive.candidate_switch_rates[selected_idx]))

    paper_raw_ser, paper_symbol_ser, compared = _ser(paper_bins, gt_bins, sf=sf, ldro=ldro)
    adaptive_raw_ser, adaptive_symbol_ser, _ = _ser(adaptive_bins, gt_bins, sf=sf, ldro=ldro)
    paper_decode = _decode_payload(packet, paper_bins, args.crc_mode, args.ldro_mode)
    adaptive_decode = _decode_payload(packet, adaptive_bins, args.crc_mode, args.ldro_mode)
    return {
        "paper_raw_ser": float(paper_raw_ser),
        "paper_symbol_ser": float(paper_symbol_ser),
        "adaptive_raw_ser": float(adaptive_raw_ser),
        "adaptive_symbol_ser": float(adaptive_symbol_ser),
        "adaptive_gain_vs_savaux": float(paper_symbol_ser - adaptive_symbol_ser),
        "gt_compared_symbols": int(compared),
        "paper_crc_valid": int(paper_decode["crc_valid"]),
        "adaptive_crc_valid": int(adaptive_decode["crc_valid"]),
        "adaptive_override_count": int(override_count),
        "adaptive_fix_count": int(fix_count),
        "adaptive_break_count": int(break_count),
        "mean_path_gain": float(np.mean(path_gains)) if path_gains else 0.0,
        "mean_switch_rate": float(np.mean(switch_rates)) if switch_rates else 0.0,
    }


def _paired_packet_row(
    dataset: str,
    snr_db: float,
    current_row: dict[str, Any],
    adaptive_row: dict[str, Any],
) -> dict[str, Any]:
    traditional = float(current_row["center_symbol_ser"])
    multi = float(current_row["multi_symbol_ser"])
    current = float(current_row["selected_symbol_ser"])
    savaux = float(adaptive_row["paper_symbol_ser"])
    adaptive = float(adaptive_row["adaptive_symbol_ser"])
    row = {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_index": int(current_row["packet_index"]),
        "frame_index": int(current_row["frame_index"]),
        "event_index": int(current_row["event_index"]),
        "payload_len": int(current_row["payload_len"]),
        "symbol_count": int(current_row["symbol_count"]),
        "traditional_fft_symbol_ser": traditional,
        "multi_offset_argmax_symbol_ser": multi,
        "current_selected_symbol_ser": current,
        "savaux_paper_symbol_ser": savaux,
        "adaptive_path_symbol_ser": adaptive,
        "current_gain_vs_traditional_fft": traditional - current,
        "savaux_gain_vs_traditional_fft": traditional - savaux,
        "adaptive_gain_vs_traditional_fft": traditional - adaptive,
        "savaux_gain_vs_current": current - savaux,
        "adaptive_gain_vs_current": current - adaptive,
        "adaptive_gain_vs_savaux": savaux - adaptive,
        "traditional_fft_crc_valid": int(current_row["center_crc_valid"]),
        "multi_offset_argmax_crc_valid": int(current_row["multi_crc_valid"]),
        "current_selected_crc_valid": int(current_row["selected_crc_valid"]),
        "savaux_paper_crc_valid": int(adaptive_row["paper_crc_valid"]),
        "adaptive_path_crc_valid": int(adaptive_row["adaptive_crc_valid"]),
        "adaptive_override_count": int(adaptive_row["adaptive_override_count"]),
        "adaptive_fix_count": int(adaptive_row["adaptive_fix_count"]),
        "adaptive_break_count": int(adaptive_row["adaptive_break_count"]),
        "adaptive_mean_path_gain": float(adaptive_row["mean_path_gain"]),
        "adaptive_mean_switch_rate": float(adaptive_row["mean_switch_rate"]),
    }
    values = {
        "traditional_fft": traditional,
        "multi_offset_argmax": multi,
        "current_selected": current,
        "savaux_paper": savaux,
        "adaptive_path": adaptive,
    }
    row["winner_by_symbol_ser"] = min(values, key=values.get)
    return row


def _summarize(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float) -> dict[str, Any]:
    traditional = _avg(rows, "traditional_fft_symbol_ser")
    multi = _avg(rows, "multi_offset_argmax_symbol_ser")
    current = _avg(rows, "current_selected_symbol_ser")
    savaux = _avg(rows, "savaux_paper_symbol_ser")
    adaptive = _avg(rows, "adaptive_path_symbol_ser")
    return {
        "dataset": dataset,
        "target_snr_db": float(snr_db),
        "packet_count": int(len(rows)),
        "traditional_fft_symbol_ser": traditional,
        "traditional_fft_symbol_accuracy": 1.0 - traditional,
        "traditional_fft_crc_valid_rate": _avg(rows, "traditional_fft_crc_valid"),
        "multi_offset_argmax_symbol_ser": multi,
        "multi_offset_argmax_symbol_accuracy": 1.0 - multi,
        "multi_offset_argmax_crc_valid_rate": _avg(rows, "multi_offset_argmax_crc_valid"),
        "current_selected_symbol_ser": current,
        "current_selected_symbol_accuracy": 1.0 - current,
        "current_selected_crc_valid_rate": _avg(rows, "current_selected_crc_valid"),
        "savaux_paper_symbol_ser": savaux,
        "savaux_paper_symbol_accuracy": 1.0 - savaux,
        "savaux_paper_crc_valid_rate": _avg(rows, "savaux_paper_crc_valid"),
        "adaptive_path_symbol_ser": adaptive,
        "adaptive_path_symbol_accuracy": 1.0 - adaptive,
        "adaptive_path_crc_valid_rate": _avg(rows, "adaptive_path_crc_valid"),
        "adaptive_gain_vs_current": current - adaptive,
        "adaptive_gain_vs_savaux": savaux - adaptive,
        "adaptive_crc_gain_vs_current": _avg(rows, "adaptive_path_crc_valid")
        - _avg(rows, "current_selected_crc_valid"),
        "adaptive_crc_gain_vs_savaux": _avg(rows, "adaptive_path_crc_valid")
        - _avg(rows, "savaux_paper_crc_valid"),
        "mean_adaptive_override_count": _avg(rows, "adaptive_override_count"),
        "mean_adaptive_fix_count": _avg(rows, "adaptive_fix_count"),
        "mean_adaptive_break_count": _avg(rows, "adaptive_break_count"),
        "mean_adaptive_path_gain": _avg(rows, "adaptive_mean_path_gain"),
        "mean_adaptive_switch_rate": _avg(rows, "adaptive_mean_switch_rate"),
    }


def _threshold_tables_with_adaptive(summary_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_thresholds, base_gains = _threshold_tables(summary_rows)
    specs = (
        ("SER<=10%", "symbol_ser", 0.10, False),
        ("accuracy>=90%", "symbol_accuracy", 0.90, True),
        ("CRC/PRR>=80%", "crc_valid_rate", 0.80, True),
        ("CRC/PRR>=50%", "crc_valid_rate", 0.50, True),
    )
    datasets = sorted({str(row["dataset"]) for row in summary_rows})
    extra_thresholds: list[dict[str, Any]] = []
    for dataset in datasets:
        curve = [row for row in summary_rows if str(row["dataset"]) == dataset]
        for metric_name, suffix, target, higher_is_better in specs:
            key = f"adaptive_path_{suffix}"
            threshold, status = _threshold_from_curve(curve, key, target, higher_is_better)
            extra_thresholds.append(
                {
                    "dataset": dataset,
                    "method": "adaptive_path",
                    "metric": metric_name,
                    "threshold_snr_db": "" if threshold is None else float(threshold),
                    "status": status,
                }
            )
    thresholds = base_thresholds + extra_thresholds
    by_key = {(row["dataset"], row["method"], row["metric"]): row for row in thresholds}
    gains = list(base_gains)
    for dataset in datasets:
        for baseline in ("traditional_fft", "current_selected", "savaux_paper"):
            for metric_name, _suffix, _target, _higher in specs:
                try:
                    method_thr = float(by_key[(dataset, "adaptive_path", metric_name)]["threshold_snr_db"])
                    base_thr = float(by_key[(dataset, baseline, metric_name)]["threshold_snr_db"])
                    gain = base_thr - method_thr
                    status = "ok"
                except (TypeError, ValueError, KeyError):
                    method_thr = float("nan")
                    base_thr = float("nan")
                    gain = float("nan")
                    status = "missing_threshold"
                gains.append(
                    {
                        "dataset": dataset,
                        "method": "adaptive_path",
                        "baseline": baseline,
                        "metric": metric_name,
                        "method_threshold_snr_db": "" if not math.isfinite(method_thr) else method_thr,
                        "baseline_threshold_snr_db": "" if not math.isfinite(base_thr) else base_thr,
                        "gain_db": "" if not math.isfinite(gain) else gain,
                        "status": status,
                    }
                )
    return thresholds, gains


def main() -> int:
    args = parse_args()
    snrs = _snr_values(args.snr_start, args.snr_stop, args.snr_step)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_args = _current_default_args(args)
    adaptive_args = _adaptive_args(args)
    current_config = build_config(current_args)

    manifest_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset_index, dataset in enumerate(args.datasets):
        paths = _dataset_paths(str(dataset))
        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        reference_power, reference_sample_count, reference_packet_count = _payload_reference_power(
            samples,
            paths["symbols"],
        )
        current_packets = load_current_packets(paths["symbols"], None)
        savaux_packets = load_savaux_packets(paths["symbols"], packet_filter=None, max_packets=None)
        savaux_by_packet = {int(packet["packet_index"]): packet for packet in savaux_packets}
        paired_packet_ids = sorted(set(current_packets).intersection(savaux_by_packet))
        dataset_seed = int(args.seed) + 100000 * int(dataset_index)
        unit_noise: np.ndarray | None = None
        if not bool(args.independent_noise):
            rng = np.random.default_rng(dataset_seed)
            noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
            unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)

        manifest_rows.append(
            {
                "dataset": str(dataset),
                "input_iq": str(paths["iq"]),
                "gt_symbol_csv": str(paths["symbols"]),
                "iq_complex64_samples": int(samples.size),
                "reference_power": float(reference_power),
                "reference_power_db": float(10.0 * math.log10(reference_power)),
                "reference_sample_count": int(reference_sample_count),
                "reference_packet_count": int(reference_packet_count),
                "paired_packet_count": int(len(paired_packet_ids)),
                "seed": int(dataset_seed),
                "candidate_top_k": int(args.candidate_top_k),
                "switch_penalty": float(args.switch_penalty),
                "step_penalty": float(args.step_penalty),
                "path_gain_power": float(args.path_gain_power),
                "switch_penalty_power": float(args.switch_penalty_power),
                "override_margin_db": float(args.override_margin_db),
                "min_savaux_rel_db": float(args.min_savaux_rel_db),
                "min_path_gain": float(args.min_path_gain),
                "max_switch_rate": float(args.max_switch_rate),
            }
        )

        for step_index, snr_db in enumerate(snrs):
            noise_power = reference_power * (10.0 ** (-float(snr_db) / 10.0))
            sigma = math.sqrt(float(noise_power) / 2.0)
            if unit_noise is None:
                rng = np.random.default_rng(dataset_seed + step_index)
                noise_i = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                noise_q = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
                noisy = (samples + (noise_i + 1j * noise_q)).astype(np.complex64, copy=False)
            else:
                noisy = (samples + sigma * unit_noise).astype(np.complex64, copy=False)

            rows: list[dict[str, Any]] = []
            for packet_id in paired_packet_ids:
                current_row = evaluate_current_packet(
                    noisy,
                    current_packets[int(packet_id)],
                    current_args,
                    current_config,
                )
                adaptive_row = _evaluate_savaux_and_adaptive_packet(
                    noisy,
                    savaux_by_packet[int(packet_id)],
                    adaptive_args,
                )
                row = _paired_packet_row(str(dataset), float(snr_db), current_row, adaptive_row)
                rows.append(row)
                packet_rows.append(row)
            summary = _summarize(rows, str(dataset), float(snr_db))
            summary_rows.append(summary)
            print(
                f"{dataset} snr={snr_db:>6.1f} "
                f"current={summary['current_selected_symbol_ser']:.3f} "
                f"savaux={summary['savaux_paper_symbol_ser']:.3f} "
                f"adaptive={summary['adaptive_path_symbol_ser']:.3f} "
                f"fix={summary['mean_adaptive_fix_count']:.2f} "
                f"break={summary['mean_adaptive_break_count']:.2f} "
                f"switch={summary['mean_adaptive_switch_rate']:.3f}",
                flush=True,
            )

    summary_rows.extend(_mean_of_datasets(summary_rows, snrs))
    threshold_rows, gain_rows = _threshold_tables_with_adaptive(summary_rows)
    _write_csv(output_dir / "manifest.csv", manifest_rows)
    _write_csv(output_dir / "per_packet_metrics.csv", packet_rows)
    _write_csv(output_dir / "snr_curve_summary.csv", summary_rows)
    _write_csv(output_dir / "threshold_table.csv", threshold_rows)
    _write_csv(output_dir / "gain_table.csv", gain_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "snr_values": snrs,
                "datasets": list(args.datasets),
                "manifest_rows": manifest_rows,
                "summary_rows": summary_rows,
                "threshold_rows": threshold_rows,
                "gain_rows": gain_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote={output_dir / 'snr_curve_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
