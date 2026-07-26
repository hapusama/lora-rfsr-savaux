#!/usr/bin/env python3
"""Paired SNR sweep for Savaux OSR baseline and the current selector.

The script uses clean header-first symbol CSV files as packet timing and symbol
GT, adds synthetic AWGN in memory, and evaluates every method on the exact same
noisy IQ realization:

* traditional center-sample FFT argmax,
* multi-offset argmax,
* current symbol-phase/coherence selector,
* paper-only Savaux oversampled demodulation.
"""

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


SCRIPT_DIR = Path(__file__).resolve().parent
WEAK_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS_DIR = WEAK_ROOT / "scripts" / "experiments"
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_paper_oversampled_baseline import (  # noqa: E402
    evaluate_packet as evaluate_savaux_packet,
    load_packets as load_savaux_packets,
)
from run_symbol_phase_threshold_sweep import (  # noqa: E402
    _evaluate_packet_methods as evaluate_current_packet,
    _snr_values,
    _threshold_from_curve,
    _write_csv,
)
from run_symbol_phase_two_stage import build_config  # noqa: E402
from run_two_stage_weak_decoder import load_packets as load_current_packets  # noqa: E402


DEFAULT_DATASETS = (
    "0_0_0_10_14_8",
    "0_0_0_10_14_16",
    "0_0_0_10_14_32",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a paired SNR sweep for center FFT, current selector, and Savaux OSR."
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--snr-start", type=float, default=-12.0)
    parser.add_argument("--snr-stop", type=float, default=-27.0)
    parser.add_argument("--snr-step", type=float, default=-1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT
        / "data"
        / "baseline_comparison"
        / "savaux_current_threshold_sweep",
    )
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--independent-noise", action="store_true")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol", "none"), default="continuous")
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument(
        "--paper-origin-shift",
        type=int,
        default=None,
        help="Default keeps the Savaux branch origin aligned to header-first chip centers.",
    )
    return parser.parse_args()


def _current_default_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        crc_mode=str(args.crc_mode),
        cfo_correction_mode=str(args.cfo_correction_mode),
        preamble_len=8.0,
        ldro_mode=int(args.ldro_mode),
        top_l_low_confidence=24,
        lock_margin_db=1.5,
        lock_peak_to_median_db=5.0,
        lock_phase_score=0.35,
        min_locked_for_line=4,
        line_trim_frac=0.25,
        phase_model="linear",
        selection_mode="coherence",
        beam_width=128,
        trajectory_rmse_scale_pi=0.30,
        trajectory_phase_weight=0.20,
        trajectory_line_weight=0.00,
        trajectory_amp_weight=0.80,
        trajectory_profile_weight=0.00,
        phase_override_min_gain=0.15,
        phase_override_max_drop_db=0.60,
        phase_override_score_margin=0.06,
        phase_override_min_line_anchors=8,
        phase_override_max_line_rmse_pi=0.25,
        coherence_weight=0.0,
        coherence_candidate_top_l=0,
        lock_min_coherence=0.0,
        smooth_phase_weight=0.05,
        smooth_amp_weight=0.50,
        smooth_coherence_weight=0.90,
        smooth_slope_penalty=0.05,
        smooth_curvature_penalty=0.10,
        smooth_max_energy_drop_db=20.0,
        smooth_min_line_anchors=4,
        smooth_min_locked_ratio=0.0,
        smooth_max_line_rmse_pi=float("inf"),
    )


def _savaux_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        crc_mode=str(args.crc_mode),
        cfo_correction_mode=str(args.cfo_correction_mode),
        ldro_mode=int(args.ldro_mode),
        paper_origin_shift=args.paper_origin_shift,
    )


def _dataset_paths(dataset: str) -> dict[str, Path]:
    return {
        "iq": WEAK_ROOT.parent / "data" / "USRP_IQ" / f"{dataset}.bin",
        "symbols": WEAK_ROOT
        / "data"
        / "weak_sync_chain"
        / "header_first"
        / f"{dataset}_header_first_symbols.csv",
    }


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    if value == "":
        return int(default)
    return int(float(value))


def _payload_reference_power(samples: np.ndarray, symbol_csv: Path) -> tuple[float, int, int]:
    total_power = 0.0
    total_count = 0
    packet_ids: set[int] = set()
    with symbol_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("stage", "")).strip().lower() != "payload":
                continue
            if _int(row, "header_valid", 0) != 1:
                continue
            sf = _int(row, "sf", 10)
            os_factor = _int(row, "os_factor", 4)
            start = _int(row, "start_sample", 0)
            n_bins = 1 << sf
            indexes = start + os_factor // 2 + os_factor * np.arange(n_bins, dtype=np.int64)
            if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
                raise ValueError(f"payload symbol exceeds IQ range: {symbol_csv} start={start}")
            values = samples[indexes]
            total_power += float(np.sum(np.abs(values) ** 2, dtype=np.float64))
            total_count += int(values.size)
            packet_ids.add(_int(row, "packet_index", -1))
    if total_count <= 0:
        raise ValueError(f"no header-valid payload symbols found: {symbol_csv}")
    reference_power = total_power / float(total_count)
    if not math.isfinite(reference_power) or reference_power <= 0.0:
        raise ValueError(f"invalid reference power {reference_power}: {symbol_csv}")
    return float(reference_power), int(total_count), len(packet_ids)


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


def _summarize(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float) -> dict[str, Any]:
    traditional = _avg(rows, "traditional_fft_symbol_ser")
    multi = _avg(rows, "multi_offset_argmax_symbol_ser")
    current = _avg(rows, "current_selected_symbol_ser")
    savaux = _avg(rows, "savaux_paper_symbol_ser")
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
        "current_gain_vs_traditional_fft": traditional - current,
        "savaux_gain_vs_traditional_fft": traditional - savaux,
        "savaux_gain_vs_current": current - savaux,
        "current_crc_gain_vs_traditional_fft": _avg(rows, "current_selected_crc_valid")
        - _avg(rows, "traditional_fft_crc_valid"),
        "savaux_crc_gain_vs_traditional_fft": _avg(rows, "savaux_paper_crc_valid")
        - _avg(rows, "traditional_fft_crc_valid"),
        "savaux_crc_gain_vs_current": _avg(rows, "savaux_paper_crc_valid")
        - _avg(rows, "current_selected_crc_valid"),
        "mean_locked_ratio": _avg(rows, "current_locked_ratio"),
        "mean_false_lock_rate": _avg(rows, "current_false_lock_rate"),
        "mean_uncertain_candidate_recall": _avg(rows, "current_uncertain_candidate_recall"),
        "mean_selected_offset_coherence": _avg(rows, "current_mean_selected_offset_coherence"),
    }


def _mean_of_datasets(summary_rows: Sequence[dict[str, Any]], snr_values: Sequence[float]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    for snr_db in snr_values:
        group = [
            row
            for row in summary_rows
            if str(row["dataset"]) != "mean_of_datasets"
            and abs(float(row["target_snr_db"]) - float(snr_db)) < 1e-9
        ]
        if not group:
            continue
        total_packets = sum(int(row["packet_count"]) for row in group)
        aggregate: dict[str, Any] = {
            "dataset": "mean_of_datasets",
            "target_snr_db": float(snr_db),
            "packet_count": int(total_packets),
        }
        for key in group[0]:
            if key in aggregate or key in {"dataset", "target_snr_db", "packet_count"}:
                continue
            aggregate[key] = float(np.mean([float(row[key]) for row in group]))
        aggregates.append(aggregate)
    return aggregates


def _threshold_tables(summary_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = (
        ("SER<=10%", "symbol_ser", 0.10, False),
        ("accuracy>=90%", "symbol_accuracy", 0.90, True),
        ("CRC/PRR>=90%", "crc_valid_rate", 0.90, True),
        ("CRC/PRR>=80%", "crc_valid_rate", 0.80, True),
        ("CRC/PRR>=50%", "crc_valid_rate", 0.50, True),
    )
    methods = (
        "traditional_fft",
        "multi_offset_argmax",
        "current_selected",
        "savaux_paper",
    )
    datasets = sorted({str(row["dataset"]) for row in summary_rows})
    thresholds: list[dict[str, Any]] = []
    for dataset in datasets:
        curve = [row for row in summary_rows if str(row["dataset"]) == dataset]
        for method in methods:
            for metric_name, suffix, target, higher_is_better in specs:
                key = f"{method}_{suffix}"
                threshold, status = _threshold_from_curve(curve, key, target, higher_is_better)
                thresholds.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric_name,
                        "threshold_snr_db": "" if threshold is None else float(threshold),
                        "status": status,
                    }
                )

    gains: list[dict[str, Any]] = []
    by_key = {(row["dataset"], row["method"], row["metric"]): row for row in thresholds}
    comparisons = (
        ("multi_offset_argmax", "traditional_fft"),
        ("current_selected", "traditional_fft"),
        ("savaux_paper", "traditional_fft"),
        ("savaux_paper", "current_selected"),
    )
    for dataset in datasets:
        for method, baseline in comparisons:
            for metric_name, _suffix, _target, _higher in specs:
                method_row = by_key[(dataset, method, metric_name)]
                base_row = by_key[(dataset, baseline, metric_name)]
                try:
                    method_thr = float(method_row["threshold_snr_db"])
                    base_thr = float(base_row["threshold_snr_db"])
                    gain = base_thr - method_thr
                    status = "ok"
                except (TypeError, ValueError):
                    method_thr = float("nan")
                    base_thr = float("nan")
                    gain = float("nan")
                    status = "missing_threshold"
                gains.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "baseline": baseline,
                        "metric": metric_name,
                        "method_threshold_snr_db": "" if not math.isfinite(method_thr) else method_thr,
                        "baseline_threshold_snr_db": "" if not math.isfinite(base_thr) else base_thr,
                        "gain_db": "" if not math.isfinite(gain) else gain,
                        "status": status,
                    }
                )
    return thresholds, gains


def _winner(row: dict[str, Any]) -> str:
    values = {
        "traditional_fft": float(row["traditional_fft_symbol_ser"]),
        "multi_offset_argmax": float(row["multi_offset_argmax_symbol_ser"]),
        "current_selected": float(row["current_selected_symbol_ser"]),
        "savaux_paper": float(row["savaux_paper_symbol_ser"]),
    }
    return min(values, key=values.get)


def _paired_packet_row(
    dataset: str,
    snr_db: float,
    current_row: dict[str, Any],
    savaux_row: dict[str, Any],
) -> dict[str, Any]:
    current = float(current_row["selected_symbol_ser"])
    savaux = float(savaux_row["paper_symbol_ser"])
    traditional = float(current_row["center_symbol_ser"])
    multi = float(current_row["multi_symbol_ser"])
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
        "current_gain_vs_traditional_fft": traditional - current,
        "savaux_gain_vs_traditional_fft": traditional - savaux,
        "savaux_gain_vs_current": current - savaux,
        "traditional_fft_crc_valid": int(current_row["center_crc_valid"]),
        "multi_offset_argmax_crc_valid": int(current_row["multi_crc_valid"]),
        "current_selected_crc_valid": int(current_row["selected_crc_valid"]),
        "savaux_paper_crc_valid": int(savaux_row["paper_crc_valid"]),
        "current_locked_ratio": float(current_row.get("locked_ratio", 0.0)),
        "current_false_lock_rate": float(current_row.get("false_lock_rate", 0.0)),
        "current_uncertain_candidate_recall": float(current_row.get("uncertain_candidate_recall", 0.0)),
        "current_mean_selected_offset_coherence": float(
            current_row.get("mean_selected_offset_coherence", 0.0)
        ),
    }
    row["winner_by_symbol_ser"] = _winner(row)
    return row


def main() -> int:
    args = parse_args()
    snrs = _snr_values(args.snr_start, args.snr_stop, args.snr_step)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current_args = _current_default_args(args)
    savaux_args = _savaux_args(args)
    current_config = build_config(current_args)

    manifest_rows: list[dict[str, Any]] = []
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset_index, dataset in enumerate(args.datasets):
        paths = _dataset_paths(str(dataset))
        if not paths["iq"].exists():
            raise FileNotFoundError(paths["iq"])
        if not paths["symbols"].exists():
            raise FileNotFoundError(paths["symbols"])

        samples = np.fromfile(paths["iq"], dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"empty IQ file: {paths['iq']}")
        reference_power, reference_sample_count, reference_packet_count = _payload_reference_power(
            samples, paths["symbols"]
        )
        current_packets = load_current_packets(paths["symbols"], None)
        savaux_packets = load_savaux_packets(paths["symbols"], packet_filter=None, max_packets=None)
        savaux_by_packet = {int(packet["packet_index"]): packet for packet in savaux_packets}
        paired_packet_ids = sorted(set(current_packets).intersection(savaux_by_packet))
        if not paired_packet_ids:
            raise ValueError(f"no paired packets available for {dataset}")

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
                "independent_noise": int(bool(args.independent_noise)),
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
                savaux_row, _symbol_rows = evaluate_savaux_packet(
                    noisy,
                    savaux_by_packet[int(packet_id)],
                    savaux_args,
                )
                row = _paired_packet_row(str(dataset), float(snr_db), current_row, savaux_row)
                rows.append(row)
                packet_rows.append(row)

            summary = _summarize(rows, str(dataset), float(snr_db))
            summary_rows.append(summary)
            print(
                f"{dataset} snr={snr_db:>6.1f} "
                f"fft={summary['traditional_fft_symbol_ser']:.3f} "
                f"multi={summary['multi_offset_argmax_symbol_ser']:.3f} "
                f"current={summary['current_selected_symbol_ser']:.3f} "
                f"savaux={summary['savaux_paper_symbol_ser']:.3f} "
                f"winner={_winner(summary)}",
                flush=True,
            )

    summary_rows.extend(_mean_of_datasets(summary_rows, snrs))
    threshold_rows, gain_rows = _threshold_tables(summary_rows)

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
    print(f"wrote={output_dir / 'threshold_table.csv'}")
    print(f"wrote={output_dir / 'gain_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
