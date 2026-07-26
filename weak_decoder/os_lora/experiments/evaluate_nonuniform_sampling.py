#!/usr/bin/env python3
"""对比评估非均匀 OSR 采样与 Savaux OSR 的判决证据。"""

from __future__ import annotations

import argparse
import json
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
    err_count,
    load_packets,
    noise_samples,
    payload_gt_bins,
    signal_reference_power,
    snr_values,
    sum_rows,
    write_csv,
)
from weak_decoder.os_lora.experiment_support.pattern_training import (  # noqa: E402
    bootstrap_offpacket_noise as _offpacket_bootstrap_samples,
    estimate_offpacket_pattern_covariance as _offpacket_pattern_covariance,
)
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    NonuniformPatternBank,
    build_pattern_bank,
    savaux_top_bins_for_symbol,
    score_nonuniform_candidates,
    select_pattern_subset,
)


METHODS = (
    "savaux",
    "best_pattern",
    "mean_pattern",
    "coherent_pattern",
    "whitened_pattern",
    "stable_pattern",
    "hybrid_mean_pattern",
    "gated_mean_pattern",
    "gated_stable_pattern",
    "gated_consensus_pattern",
    "gated_hybrid_pattern",
)


def _select_by_metric(rows: Sequence[Any], metric: str) -> int:
    if not rows:
        return 0
    return int(max(rows, key=lambda item: float(getattr(item, metric))).raw_fft_bin)


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


def _select_gated(
    rows: Sequence[Any],
    metric: str,
    savaux_bin: int,
    savaux_margin_db: float,
    gate_margin_db: float,
    switch_ratio: float,
    min_consistency_delta: float = -1.0,
) -> int:
    if not rows or float(savaux_margin_db) > float(gate_margin_db):
        return int(savaux_bin)
    best = max(rows, key=lambda item: float(getattr(item, metric)))
    savaux_row = next((row for row in rows if int(row.raw_fft_bin) == int(savaux_bin)), None)
    if savaux_row is None:
        return int(best.raw_fft_bin)
    best_score = float(getattr(best, metric))
    savaux_score = float(getattr(savaux_row, metric))
    if float(min_consistency_delta) > -1.0:
        best_consistency = float(best.coherent_pattern_power / (best.mean_pattern_power + 1e-30))
        savaux_consistency = float(savaux_row.coherent_pattern_power / (savaux_row.mean_pattern_power + 1e-30))
        if best_consistency < savaux_consistency + float(min_consistency_delta):
            return int(savaux_bin)
    if best_score > savaux_score * float(switch_ratio):
        return int(best.raw_fft_bin)
    return int(savaux_bin)


def _select_gated_consensus(
    rows: Sequence[Any],
    savaux_bin: int,
    savaux_margin_db: float,
    gate_margin_db: float,
    switch_ratio: float,
) -> int:
    if not rows or float(savaux_margin_db) > float(gate_margin_db):
        return int(savaux_bin)
    mean_best = max(rows, key=lambda item: float(item.mean_pattern_power))
    stable_best = max(rows, key=lambda item: float(item.stable_pattern_power))
    if int(mean_best.raw_fft_bin) != int(stable_best.raw_fft_bin):
        return int(savaux_bin)
    savaux_row = next((row for row in rows if int(row.raw_fft_bin) == int(savaux_bin)), None)
    if savaux_row is None:
        return int(mean_best.raw_fft_bin)
    mean_ok = float(mean_best.mean_pattern_power) > float(savaux_row.mean_pattern_power) * float(switch_ratio)
    stable_ok = float(stable_best.stable_pattern_power) > float(savaux_row.stable_pattern_power) * float(switch_ratio)
    if mean_ok and stable_ok:
        return int(mean_best.raw_fft_bin)
    return int(savaux_bin)


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    bank: NonuniformPatternBank,
    noise_covariance: np.ndarray | None,
) -> dict[str, Any]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    origin_shift = int(args.origin_shift) if args.origin_shift is not None else os_factor // 2
    header_start = int(packet["header_start_sample"]) + origin_shift
    selected = {method: [] for method in METHODS}
    gt_bins = list(payload_gt_bins(packet))
    gt_in_topk = 0
    gt_ranks: list[int] = []
    best_names: list[str] = []
    gains: dict[str, list[float]] = {
        "best_pattern_gain_vs_savaux": [],
        "coherent_gain_vs_savaux": [],
        "whitened_gain_vs_savaux": [],
        "stable_gain_vs_savaux": [],
        "hybrid_gain_vs_savaux": [],
    }

    for symbol in packet["payload_symbols"]:
        start = int(symbol["start_sample"]) + origin_shift
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
        savaux_bin = int(candidate_bins[0]) if candidate_bins else 0
        savaux_margin_db = _top_margin_db(savaux_power)
        selected["savaux"].append(savaux_bin)
        if gt_bin in set(candidate_bins):
            gt_in_topk += 1
        gt_ranks.append(_rank(savaux_power, gt_bin))

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
            hybrid_consistency_alpha=float(args.hybrid_consistency_alpha),
            noise_covariance=noise_covariance,
        )
        selected["best_pattern"].append(_select_by_metric(score_rows, "best_pattern_power"))
        selected["mean_pattern"].append(_select_by_metric(score_rows, "mean_pattern_power"))
        selected["coherent_pattern"].append(_select_by_metric(score_rows, "coherent_pattern_power"))
        selected["whitened_pattern"].append(_select_by_metric(score_rows, "whitened_pattern_power"))
        selected["stable_pattern"].append(_select_by_metric(score_rows, "stable_pattern_power"))
        selected["hybrid_mean_pattern"].append(_select_by_metric(score_rows, "hybrid_mean_power"))
        selected["gated_mean_pattern"].append(
            _select_gated(
                score_rows,
                "mean_pattern_power",
                savaux_bin,
                savaux_margin_db,
                float(args.gate_margin_db),
                float(args.gate_switch_ratio),
                -1.0,
            )
        )
        selected["gated_stable_pattern"].append(
            _select_gated(
                score_rows,
                "stable_pattern_power",
                savaux_bin,
                savaux_margin_db,
                float(args.gate_margin_db),
                float(args.gate_switch_ratio),
                -1.0,
            )
        )
        selected["gated_consensus_pattern"].append(
            _select_gated_consensus(
                score_rows,
                savaux_bin,
                savaux_margin_db,
                float(args.gate_margin_db),
                float(args.gate_switch_ratio),
            )
        )
        hybrid_rows = score_rows[: max(1, int(args.hybrid_max_rank))]
        selected["gated_hybrid_pattern"].append(
            _select_gated(
                hybrid_rows,
                "hybrid_mean_power",
                savaux_bin,
                savaux_margin_db,
                float(args.gate_margin_db),
                float(args.gate_switch_ratio),
                float(args.hybrid_min_consistency_delta),
            )
        )
        if score_rows:
            savaux_row = next((row for row in score_rows if int(row.raw_fft_bin) == savaux_bin), score_rows[0])
            best_names.append(str(savaux_row.best_pattern_name))
            gains["best_pattern_gain_vs_savaux"].append(float(savaux_row.best_pattern_gain_vs_savaux))
            gains["coherent_gain_vs_savaux"].append(float(savaux_row.coherent_gain_vs_savaux))
            gains["whitened_gain_vs_savaux"].append(float(savaux_row.whitened_gain_vs_savaux))
            gains["stable_gain_vs_savaux"].append(float(savaux_row.stable_gain_vs_savaux))
            gains["hybrid_gain_vs_savaux"].append(float(savaux_row.hybrid_gain_vs_savaux))

    out: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "symbol_count": int(len(gt_bins)),
        "gt_topk_recall": float(gt_in_topk / max(1, len(gt_bins))),
        "mean_gt_savaux_rank": float(np.mean(gt_ranks)) if gt_ranks else 0.0,
        "pattern_count": int(len(bank.names)),
        "bank_kind": str(bank.kind),
        "mode_best_pattern_on_savaux": max(set(best_names), key=best_names.count) if best_names else "",
    }
    for key, values in gains.items():
        out[f"mean_{key}"] = float(np.mean(values)) if values else 0.0

    savaux_bins = selected["savaux"]
    for method in METHODS:
        err, compared = err_count(selected[method], gt_bins)
        out[f"{method}_err"] = int(err)
        out[f"{method}_ser"] = float(err / max(1, compared))
        if method != "savaux":
            fix = 0
            break_count = 0
            changes = 0
            for idx in range(min(len(gt_bins), len(selected[method]), len(savaux_bins))):
                cand = int(selected[method][idx])
                sav = int(savaux_bins[idx])
                gt = int(gt_bins[idx])
                if cand != sav:
                    changes += 1
                if sav != gt and cand == gt:
                    fix += 1
                if sav == gt and cand != gt:
                    break_count += 1
            out[f"{method}_changes_vs_savaux"] = int(changes)
            out[f"{method}_fix_vs_savaux"] = int(fix)
            out[f"{method}_break_vs_savaux"] = int(break_count)
    return out


def _summary(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float | None, seed: int) -> dict[str, Any]:
    symbols = sum_rows(rows, "symbol_count")
    out: dict[str, Any] = {
        "dataset": dataset,
        "snr_db": "" if snr_db is None else float(snr_db),
        "seed": int(seed),
        "packet_count": int(len(rows)),
        "symbol_count": int(symbols),
        "bank_kind": str(rows[0]["bank_kind"]) if rows else "",
        "pattern_count": int(rows[0]["pattern_count"]) if rows else 0,
        "covariance_source": str(rows[0].get("covariance_source", "white")) if rows else "",
        "noise_mode": str(rows[0].get("noise_mode", "awgn")) if rows else "",
        "white_effective_replicas": float(rows[0].get("white_effective_replicas", 0.0)) if rows else 0.0,
        "empirical_effective_replicas": (
            float(rows[0].get("empirical_effective_replicas", 0.0)) if rows else 0.0
        ),
        "cross_validated_effective_replicas": (
            float(rows[0].get("cross_validated_effective_replicas", 0.0)) if rows else 0.0
        ),
        "offpacket_windows": int(rows[0].get("offpacket_windows", 0)) if rows else 0,
        "mean_gt_topk_recall": float(np.mean([float(row["gt_topk_recall"]) for row in rows])) if rows else 0.0,
        "mean_gt_savaux_rank": float(np.mean([float(row["mean_gt_savaux_rank"]) for row in rows])) if rows else 0.0,
    }
    for method in METHODS:
        err = sum_rows(rows, f"{method}_err")
        out[f"{method}_ser"] = float(err / max(1, symbols))
    for method in METHODS:
        if method == "savaux":
            continue
        out[f"{method}_fix_vs_savaux"] = sum_rows(rows, f"{method}_fix_vs_savaux")
        out[f"{method}_break_vs_savaux"] = sum_rows(rows, f"{method}_break_vs_savaux")
        out[f"{method}_changes_vs_savaux"] = sum_rows(rows, f"{method}_changes_vs_savaux")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_16"])
    parser.add_argument("--snrs", nargs="*", type=float, default=[-22.0, -23.0, -24.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-packets", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--noise-mode", choices=("awgn", "offpacket_bootstrap"), default="awgn")
    parser.add_argument("--bootstrap-source-windows", type=int, default=256)
    parser.add_argument(
        "--bank-kind",
        choices=(
            "fixed",
            "basic",
            "periodic",
            "dither",
            "multiscale",
            "quadratic",
            "bitmix",
            "random",
            "balanced_random",
            "search",
            "super_search",
            "basic_only",
            "periodic_only",
            "dither_only",
            "multiscale_only",
            "quadratic_only",
            "bitmix_only",
            "random_only",
            "balanced_random_only",
            "search_only",
            "super_search_only",
        ),
        default="basic",
    )
    parser.add_argument("--random-count", type=int, default=64)
    parser.add_argument("--bank-seed", type=int, default=0)
    parser.add_argument("--max-patterns", type=int, default=32)
    parser.add_argument("--subset-strategy", choices=("head", "diverse"), default="diverse")
    parser.add_argument("--covariance-source", choices=("white", "offpacket"), default="white")
    parser.add_argument("--offpacket-windows", type=int, default=64)
    parser.add_argument("--offpacket-bins", type=int, default=16)
    parser.add_argument("--offpacket-shrinkage", type=float, default=0.2)
    parser.add_argument("--offpacket-guard-symbols", type=float, default=2.0)
    parser.add_argument("--stable-exponent", type=float, default=1.0)
    parser.add_argument("--hybrid-mean-beta", type=float, default=0.75)
    parser.add_argument("--hybrid-consistency-alpha", type=float, default=0.0)
    parser.add_argument("--hybrid-max-rank", type=int, default=3)
    parser.add_argument("--hybrid-min-consistency-delta", type=float, default=-1.0)
    parser.add_argument("--gate-margin-db", type=float, default=1.5)
    parser.add_argument("--gate-switch-ratio", type=float, default=1.0)
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
        default=WEAK_ROOT / "data" / "os_lora" / "nonuniform_sampling",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = dataset_paths(str(dataset))
        clean = np.fromfile(iq_path, dtype=np.complex64)
        all_packets = load_packets(symbol_path)
        packets = all_packets
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        if not packets:
            continue
        sf = int(packets[0]["sf"])
        os_factor = int(packets[0]["os_factor"])
        bank = build_pattern_bank(
            sf,
            os_factor,
            kind=str(args.bank_kind),
            random_count=int(args.random_count),
            seed=int(args.bank_seed),
        )
        bank = select_pattern_subset(bank, int(args.max_patterns), str(args.subset_strategy))
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
        for seed in args.seeds:
            for snr_db in snr_values(args.snrs):
                if str(args.noise_mode) == "offpacket_bootstrap":
                    samples = _offpacket_bootstrap_samples(
                        clean=clean,
                        packets=all_packets,
                        snr_db=snr_db,
                        seed=int(seed),
                        reference_power=reference_power,
                        max_source_windows=int(args.bootstrap_source_windows),
                        guard_symbols=float(args.offpacket_guard_symbols),
                    )
                else:
                    samples = noise_samples(clean, snr_db, int(seed), reference_power)
                covariance: np.ndarray | None = None
                white_replicas = 0.0
                empirical_replicas = 0.0
                cross_validated_replicas = 0.0
                offpacket_windows = 0
                covariance_bins: tuple[int, ...] = tuple()
                if str(args.covariance_source) == "offpacket":
                    (
                        covariance,
                        white_replicas,
                        empirical_replicas,
                        cross_validated_replicas,
                        offpacket_windows,
                        covariance_bins,
                    ) = _offpacket_pattern_covariance(
                        samples=samples,
                        packets=all_packets,
                        bank=bank,
                        max_windows=int(args.offpacket_windows),
                        bin_count=int(args.offpacket_bins),
                        guard_symbols=float(args.offpacket_guard_symbols),
                        shrinkage=float(args.offpacket_shrinkage),
                        seed=int(args.bank_seed) + 1009 * int(seed),
                    )
                rows = []
                for packet in packets:
                    row = _evaluate_packet(
                        samples=samples,
                        packet=packet,
                        args=args,
                        bank=bank,
                        noise_covariance=covariance,
                    )
                    row.update(
                        {
                            "dataset": str(dataset),
                            "snr_db": "" if snr_db is None else float(snr_db),
                            "seed": int(seed),
                            "covariance_source": str(args.covariance_source),
                            "noise_mode": str(args.noise_mode),
                            "white_effective_replicas": float(white_replicas),
                            "empirical_effective_replicas": float(empirical_replicas),
                            "cross_validated_effective_replicas": float(cross_validated_replicas),
                            "offpacket_windows": int(offpacket_windows),
                            "covariance_bins": " ".join(str(v) for v in covariance_bins),
                        }
                    )
                    rows.append(row)
                    packet_rows.append(row)
                summary = _summary(rows, str(dataset), snr_db, int(seed))
                summary_rows.append(summary)
                print(
                    f"{dataset} snr={snr_db} seed={seed}: "
                    f"savaux={summary['savaux_ser']:.4f} "
                    f"best={summary['best_pattern_ser']:.4f} "
                    f"mean={summary['mean_pattern_ser']:.4f} "
                    f"coh={summary['coherent_pattern_ser']:.4f} "
                    f"gls={summary['whitened_pattern_ser']:.4f} "
                    f"stable={summary['stable_pattern_ser']:.4f} "
                    f"hybrid={summary['hybrid_mean_pattern_ser']:.4f} "
                    f"gmean={summary['gated_mean_pattern_ser']:.4f} "
                    f"gstable={summary['gated_stable_pattern_ser']:.4f} "
                    f"gcons={summary['gated_consensus_pattern_ser']:.4f} "
                    f"ghybrid={summary['gated_hybrid_pattern_ser']:.4f} "
                    f"eff_train={summary['empirical_effective_replicas']:.3f} "
                    f"eff_cv={summary['cross_validated_effective_replicas']:.3f}",
                    flush=True,
                )

    write_csv(out_dir / "packet_metrics.csv", packet_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
