#!/usr/bin/env python3
"""导出候选级非均匀采样重排序诊断。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.common import (  # noqa: E402
    dataset_paths,
    load_packets,
    noise_samples,
    signal_reference_power,
    snr_values,
    write_csv,
)
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    build_pattern_bank,
    savaux_top_bins_for_symbol,
    score_nonuniform_candidates,
)


def _rank(power: np.ndarray, raw_bin: int) -> int:
    values = np.asarray(power, dtype=np.float64)
    b = int(raw_bin)
    if b < 0 or b >= values.size:
        return 0
    return int(np.sum(values > values[b]) + 1)


def _top_margin_db(power: np.ndarray) -> float:
    values = np.asarray(power, dtype=np.float64)
    if values.size < 2:
        return 0.0
    top2 = np.argpartition(values, -2)[-2:]
    ordered = top2[np.argsort(values[top2])[::-1]]
    return float(10.0 * np.log10((values[ordered[0]] + 1e-30) / (values[ordered[1]] + 1e-30)))


def _circular_distance(a: int, b: int, size: int) -> int:
    delta = abs(int(a) - int(b)) % int(size)
    return int(min(delta, int(size) - delta))


def _candidate_rows_for_symbol(
    samples: np.ndarray,
    packet: dict[str, Any],
    symbol: dict[str, Any],
    args: argparse.Namespace,
    bank: Any,
) -> list[dict[str, Any]]:
    sf = int(packet["sf"])
    n_bins = 1 << sf
    os_factor = int(packet["os_factor"])
    origin_shift = int(args.origin_shift) if args.origin_shift is not None else os_factor // 2
    start = int(symbol["start_sample"]) + origin_shift
    header_start = int(packet["header_start_sample"]) + origin_shift
    gt_bin = int(symbol["gt_bin"])
    candidate_bins, savaux_power = savaux_top_bins_for_symbol(
        samples=samples,
        start_sample=start,
        sf=sf,
        os_factor=os_factor,
        top_k=int(args.top_k),
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=header_start,
        cfo_correction_mode=str(args.cfo_correction_mode),
    )
    if not candidate_bins:
        return []
    savaux_bin = int(candidate_bins[0])
    savaux_margin_db = _top_margin_db(savaux_power)
    score_rows = score_nonuniform_candidates(
        samples=samples,
        start_sample=start,
        sf=sf,
        os_factor=os_factor,
        candidate_bins=candidate_bins,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=header_start,
        cfo_correction_mode=str(args.cfo_correction_mode),
        bank=bank,
        savaux_power=savaux_power,
        stable_exponent=float(args.stable_exponent),
        hybrid_mean_beta=float(args.hybrid_mean_beta),
    )

    rows: list[dict[str, Any]] = []
    ordered = {int(raw_bin): idx + 1 for idx, raw_bin in enumerate(candidate_bins)}
    for item in score_rows:
        raw_bin = int(item.raw_fft_bin)
        mean_power = float(item.mean_pattern_power)
        coherent_power = float(item.coherent_pattern_power)
        stable_power = float(item.stable_pattern_power)
        savaux_bin_power = float(item.savaux_power)
        consistency = float(np.clip(coherent_power / (mean_power + 1e-30), 0.0, 1.0))
        rows.append(
            {
                "dataset": str(args.current_dataset),
                "snr_db": "" if args.current_snr is None else float(args.current_snr),
                "seed": int(args.current_seed),
                "packet_index": int(packet["packet_index"]),
                "payload_symbol_index": int(symbol["payload_symbol_index"]),
                "gt_bin": int(gt_bin),
                "raw_fft_bin": raw_bin,
                "is_gt": int(raw_bin == gt_bin),
                "is_savaux": int(raw_bin == savaux_bin),
                "savaux_ok": int(savaux_bin == gt_bin),
                "candidate_rank": int(ordered.get(raw_bin, 0)),
                "gt_savaux_rank": int(_rank(savaux_power, gt_bin)),
                "savaux_margin_db": float(savaux_margin_db),
                "distance_to_gt": int(_circular_distance(raw_bin, gt_bin, n_bins)),
                "distance_to_savaux": int(_circular_distance(raw_bin, savaux_bin, n_bins)),
                "savaux_power": savaux_bin_power,
                "log_savaux_power": float(math.log(savaux_bin_power + 1e-30)),
                "best_pattern_power": float(item.best_pattern_power),
                "mean_pattern_power": mean_power,
                "coherent_pattern_power": coherent_power,
                "whitened_pattern_power": float(item.whitened_pattern_power),
                "stable_pattern_power": stable_power,
                "hybrid_mean_power": float(item.hybrid_mean_power),
                "pattern_consistency": consistency,
                "mean_gain_vs_savaux": float(mean_power / (savaux_bin_power + 1e-30)),
                "coherent_gain_vs_savaux": float(item.coherent_gain_vs_savaux),
                "whitened_gain_vs_savaux": float(item.whitened_gain_vs_savaux),
                "stable_gain_vs_savaux": float(item.stable_gain_vs_savaux),
                "hybrid_gain_vs_savaux": float(item.hybrid_gain_vs_savaux),
            }
        )
    return rows


def _summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["snr_db"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    methods = {
        "savaux": "savaux_power",
        "mean_pattern": "mean_pattern_power",
        "coherent_pattern": "coherent_pattern_power",
        "whitened_pattern": "whitened_pattern_power",
        "stable_pattern": "stable_pattern_power",
        "hybrid_mean": "hybrid_mean_power",
    }
    out: list[dict[str, Any]] = []
    for (dataset, snr_db, seed), items in sorted(grouped.items()):
        by_symbol: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in items:
            key = (int(row["packet_index"]), int(row["payload_symbol_index"]))
            by_symbol.setdefault(key, []).append(row)
        summary: dict[str, Any] = {
            "dataset": dataset,
            "snr_db": snr_db,
            "seed": int(seed),
            "symbol_count": int(len(by_symbol)),
            "gt_in_topk": int(sum(any(int(r["is_gt"]) for r in group) for group in by_symbol.values())),
        }
        for name, metric in methods.items():
            err = 0
            fix = 0
            break_count = 0
            changes = 0
            for group in by_symbol.values():
                best = max(group, key=lambda r: float(r[metric]))
                savaux = next((r for r in group if int(r["is_savaux"])), group[0])
                if not int(best["is_gt"]):
                    err += 1
                if int(best["raw_fft_bin"]) != int(savaux["raw_fft_bin"]):
                    changes += 1
                if not int(savaux["is_gt"]) and int(best["is_gt"]):
                    fix += 1
                if int(savaux["is_gt"]) and not int(best["is_gt"]):
                    break_count += 1
            summary[f"{name}_ser"] = float(err / max(1, len(by_symbol)))
            if name != "savaux":
                summary[f"{name}_fix_vs_savaux"] = int(fix)
                summary[f"{name}_break_vs_savaux"] = int(break_count)
                summary[f"{name}_changes_vs_savaux"] = int(changes)
        out.append(summary)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_8", "0_0_0_10_14_16", "0_0_0_10_14_32"])
    parser.add_argument("--snrs", nargs="*", type=float, default=[-25.0, -26.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 42])
    parser.add_argument("--max-packets", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--bank-kind", default="random_only")
    parser.add_argument("--random-count", type=int, default=32)
    parser.add_argument("--bank-seed", type=int, default=2)
    parser.add_argument("--stable-exponent", type=float, default=1.0)
    parser.add_argument("--hybrid-mean-beta", type=float, default=0.75)
    parser.add_argument("--origin-shift", type=int, default=None)
    parser.add_argument("--cfo-correction-mode", choices=("none", "symbol", "continuous"), default="continuous")
    parser.add_argument(
        "--signal-reference-mode",
        choices=("packet", "payload", "header_payload", "whole"),
        default="packet",
    )
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "candidate_diagnostics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = dataset_paths(str(dataset))
        clean = np.fromfile(iq_path, dtype=np.complex64)
        packets = load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        if not packets:
            continue
        reference_power, reference_samples, reference_packets = signal_reference_power(
            samples=clean,
            packets=packets,
            mode=str(args.signal_reference_mode),
            explicit_power=args.signal_reference_power,
        )
        print(
            f"{dataset}: reference_power={reference_power:.6g} "
            f"samples={reference_samples} packets={reference_packets}",
            flush=True,
        )
        sf = int(packets[0]["sf"])
        os_factor = int(packets[0]["os_factor"])
        bank = build_pattern_bank(
            sf,
            os_factor,
            kind=str(args.bank_kind),
            random_count=int(args.random_count),
            seed=int(args.bank_seed),
        )
        for seed in args.seeds:
            for snr_db in snr_values(args.snrs):
                samples = noise_samples(clean, snr_db, int(seed), reference_power)
                args.current_dataset = str(dataset)
                args.current_seed = int(seed)
                args.current_snr = snr_db
                for packet in packets:
                    for symbol in packet["payload_symbols"]:
                        all_rows.extend(_candidate_rows_for_symbol(samples, packet, symbol, args, bank))
                print(f"{dataset} snr={snr_db} seed={seed}: candidates={len(all_rows)}", flush=True)
    out_dir = Path(args.output_dir).resolve()
    summaries = _summaries(all_rows)
    write_csv(out_dir / "candidate_metrics.csv", all_rows)
    write_csv(out_dir / "summary.csv", summaries)
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
