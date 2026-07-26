#!/usr/bin/env python3
"""Diagnose whether adaptive path evidence ranks GT above Savaux argmax."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

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
from run_savaux_current_threshold_sweep import _dataset_paths, _payload_reference_power  # noqa: E402
from weak_decoder.decoding.adaptive_path_demod import score_adaptive_path_candidates  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import paper_oversampled_spectrum  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive-path GT-vs-argmax score diagnostic.")
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snr-db", type=float, default=-24.0)
    parser.add_argument("--output-dir", type=Path, default=WEAK_ROOT / "data" / "adaptive_path_oracle")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--candidate-top-k", type=int, default=32)
    parser.add_argument("--switch-penalty", type=float, default=0.40)
    parser.add_argument("--step-penalty", type=float, default=0.10)
    parser.add_argument("--path-gain-power", type=float, default=0.18)
    parser.add_argument("--switch-penalty-power", type=float, default=0.40)
    parser.add_argument("--cfo-correction-mode", choices=("continuous", "symbol", "none"), default="continuous")
    parser.add_argument("--paper-origin-shift", type=int, default=None)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    paths = _dataset_paths(str(args.dataset))
    samples = np.fromfile(paths["iq"], dtype=np.complex64)
    reference_power, _count, _packets = _payload_reference_power(samples, paths["symbols"])
    rng = np.random.default_rng(int(args.seed))
    noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
    unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)
    noise_power = reference_power * (10.0 ** (-float(args.snr_db) / 10.0))
    sigma = math.sqrt(float(noise_power) / 2.0)
    noisy = (samples + sigma * unit_noise).astype(np.complex64, copy=False)

    rows: list[dict[str, Any]] = []
    for packet in load_savaux_packets(paths["symbols"], packet_filter=None, max_packets=None):
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        origin_shift = int(args.paper_origin_shift) if args.paper_origin_shift is not None else os_factor // 2
        paper_header_start = int(packet["header_start_sample"]) + origin_shift
        for symbol in packet["payload_symbols"]:
            gt_bin = int(symbol["gt_raw_fft_bin"])
            start_sample = int(symbol["start_sample"]) + origin_shift
            spectrum, _branches, _phase = paper_oversampled_spectrum(
                samples=noisy,
                start_sample=start_sample,
                sf=sf,
                os_factor=os_factor,
                cfo_int=int(packet["cfo_int"]),
                cfo_frac=float(packet["cfo_frac"]),
                header_start_sample=paper_header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
            power = np.abs(spectrum).astype(np.float64) ** 2
            savaux_bin = int(np.argmax(power))
            order = np.argsort(power)[::-1]
            gt_rank = int(np.where(order == gt_bin)[0][0] + 1)
            candidate_bins = list(int(v) for v in order[: int(args.candidate_top_k)])
            if gt_bin not in candidate_bins:
                candidate_bins.append(gt_bin)
            if savaux_bin not in candidate_bins:
                candidate_bins.append(savaux_bin)
            candidates, _paths = score_adaptive_path_candidates(
                samples=noisy,
                start_sample=start_sample,
                sf=sf,
                os_factor=os_factor,
                candidate_bins=candidate_bins,
                cfo_int=int(packet["cfo_int"]),
                cfo_frac=float(packet["cfo_frac"]),
                header_start_sample=paper_header_start,
                cfo_correction_mode=str(args.cfo_correction_mode),
                switch_penalty=float(args.switch_penalty),
                step_penalty=float(args.step_penalty),
                path_gain_power=float(args.path_gain_power),
                switch_penalty_power=float(args.switch_penalty_power),
                savaux_power=power,
            )
            by_bin = {int(item.raw_fft_bin): item for item in candidates}
            gt_item = by_bin[gt_bin]
            arg_item = by_bin[savaux_bin]
            rows.append(
                {
                    "packet_index": int(packet["packet_index"]),
                    "payload_symbol_index": int(symbol["payload_symbol_index"]),
                    "gt_bin": gt_bin,
                    "savaux_bin": savaux_bin,
                    "savaux_hit": int(gt_bin == savaux_bin),
                    "gt_savaux_rank": gt_rank,
                    "gt_in_topk": int(gt_rank <= int(args.candidate_top_k)),
                    "gt_score": float(gt_item.composite_score),
                    "arg_score": float(arg_item.composite_score),
                    "gt_score_over_arg_db": float(10.0 * math.log10((gt_item.composite_score + 1e-30) / (arg_item.composite_score + 1e-30))),
                    "gt_path_gain": float(gt_item.adaptive_path_gain),
                    "arg_path_gain": float(arg_item.adaptive_path_gain),
                    "gt_switch_rate": float(gt_item.switch_rate),
                    "arg_switch_rate": float(arg_item.switch_rate),
                    "would_fix": int(gt_bin != savaux_bin and gt_item.composite_score > arg_item.composite_score),
                    "would_break": int(gt_bin == savaux_bin and gt_item.composite_score < arg_item.composite_score),
                }
            )

    summary = {
        "dataset": str(args.dataset),
        "snr_db": float(args.snr_db),
        "symbols": len(rows),
        "savaux_ser": float(np.mean([1 - int(row["savaux_hit"]) for row in rows])) if rows else 0.0,
        "gt_topk_recall": float(np.mean([int(row["gt_in_topk"]) for row in rows])) if rows else 0.0,
        "wrong_symbols": int(sum(1 - int(row["savaux_hit"]) for row in rows)),
        "would_fix_wrong_symbols": int(sum(int(row["would_fix"]) for row in rows)),
        "mean_gt_score_over_arg_db_wrong": float(
            np.mean([float(row["gt_score_over_arg_db"]) for row in rows if not int(row["savaux_hit"])])
        ) if any(not int(row["savaux_hit"]) for row in rows) else 0.0,
        "mean_gt_switch_wrong": float(
            np.mean([float(row["gt_switch_rate"]) for row in rows if not int(row["savaux_hit"])])
        ) if any(not int(row["savaux_hit"]) for row in rows) else 0.0,
        "mean_arg_switch_wrong": float(
            np.mean([float(row["arg_switch_rate"]) for row in rows if not int(row["savaux_hit"])])
        ) if any(not int(row["savaux_hit"]) for row in rows) else 0.0,
    }
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / f"{args.dataset}_snr_{int(args.snr_db)}_adaptive_oracle.csv", rows)
    (out / f"{args.dataset}_snr_{int(args.snr_db)}_adaptive_oracle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
