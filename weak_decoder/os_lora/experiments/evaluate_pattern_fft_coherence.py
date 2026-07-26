#!/usr/bin/env python3
"""评估非均匀 pattern FFT 的全频谱相干合并。"""

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
    err_count,
    load_packets,
    noise_samples,
    signal_reference_power,
    snr_values,
    write_csv,
)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.os_lora.experiment_support.pattern_training import (  # noqa: E402
    bootstrap_offpacket_noise as _offpacket_bootstrap_samples,
    estimate_offpacket_pattern_covariance as _offpacket_pattern_covariance,
)
from weak_decoder.os_lora.system.noise import select_background_bins as _background_bins  # noqa: E402
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    NonuniformPatternBank,
    adaptive_gls_spectrum_power,
    build_pattern_bank,
    crossfit_gls_spectrum_power,
    matrix_free_crossfit_gls_spectrum_power,
    pattern_bank_spectra,
    pattern_coherence_weighted_power,
    plain_pattern_fft_spectra,
    prepare_dechirped_symbol,
    select_pattern_subset,
)


METHODS = (
    "savaux",
    "plain_coherent",
    "lora_coherent",
    "lora_mean_power",
    "lora_best_power",
    "lora_stable_power",
    "lora_adaptive_gls",
    "lora_crossfit_gls",
    "lora_cg_gls",
    "lora_offpacket_gls",
)


def _fixed_gls_spectrum_power(pattern_spectra: np.ndarray, covariance: np.ndarray | None) -> np.ndarray:
    spectra = np.asarray(pattern_spectra, dtype=np.complex128)
    if spectra.ndim != 2 or spectra.shape[0] == 0:
        return np.zeros(spectra.shape[-1] if spectra.ndim == 2 else 0, dtype=np.float64)
    if covariance is None:
        return np.abs(np.mean(spectra, axis=0)).astype(np.float64) ** 2
    cov = np.asarray(covariance, dtype=np.complex128)
    target = np.ones(spectra.shape[0], dtype=np.complex128)
    inverse_target = np.linalg.pinv(cov, rcond=1e-4) @ target
    denom = max(float(np.real(target.conj().T @ inverse_target)), 1e-30)
    projection = inverse_target.conj().T @ spectra
    return np.asarray(np.abs(projection).astype(np.float64) ** 2 / denom, dtype=np.float64)


def _power_metrics(power: np.ndarray, gt_bin: int, guard_bins: int = 1) -> dict[str, float | int]:
    values = np.asarray(power, dtype=np.float64)
    gt = int(gt_bin) % int(values.size)
    selected = int(np.argmax(values)) if values.size else 0
    gt_power = float(values[gt]) if values.size else 0.0
    rank = int(np.sum(values > gt_power) + 1) if values.size else 0
    mask = np.ones(values.size, dtype=bool)
    for delta in range(-int(guard_bins), int(guard_bins) + 1):
        mask[(gt + delta) % values.size] = False
    floor_values = values[mask] if np.any(mask) else values
    floor_median = float(np.median(floor_values)) if floor_values.size else 0.0
    other = values.copy()
    other[gt] = -np.inf
    next_power = float(np.max(other)) if other.size else 0.0
    return {
        "selected_bin": selected,
        "gt_rank": rank,
        "gt_power": gt_power,
        "gt_margin_db": float(10.0 * math.log10((gt_power + 1e-30) / (next_power + 1e-30))),
        "gt_floor_db": float(10.0 * math.log10((gt_power + 1e-30) / (floor_median + 1e-30))),
    }


def _evaluate_symbol(
    samples: np.ndarray,
    packet: dict[str, Any],
    symbol: dict[str, Any],
    bank: NonuniformPatternBank,
    origin_shift: int,
    cfo_correction_mode: str,
    stable_exponent: float,
    adaptive_load: float,
    adaptive_exclude_top: int,
    adaptive_guard_bins: int,
    crossfit_folds: int,
    matrix_free_iterations: int,
    matrix_free_tolerance: float,
    offpacket_covariance: np.ndarray | None,
) -> dict[str, Any]:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    start = int(symbol["start_sample"]) + int(origin_shift)
    header_start = int(packet["header_start_sample"]) + int(origin_shift)
    gt_bin = int(symbol["gt_bin"])

    savaux_spectrum, _branches, _phase = paper_oversampled_spectrum(
        samples=samples,
        start_sample=start,
        sf=sf,
        os_factor=os_factor,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=header_start,
        cfo_correction_mode=cfo_correction_mode,
    )
    dechirped = prepare_dechirped_symbol(
        samples=samples,
        start_sample=start,
        sf=sf,
        os_factor=os_factor,
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=header_start,
        cfo_correction_mode=cfo_correction_mode,
    )
    plain_spectra = plain_pattern_fft_spectra(dechirped=dechirped, bank=bank)
    lora_spectra = pattern_bank_spectra(dechirped=dechirped, bank=bank)
    savaux_power = np.abs(savaux_spectrum).astype(np.float64) ** 2
    adaptive_background = _background_bins(
        savaux_power,
        exclude_top=int(adaptive_exclude_top),
        guard_bins=int(adaptive_guard_bins),
    )

    matrix_free_result = matrix_free_crossfit_gls_spectrum_power(
        lora_spectra,
        covariance_bins=adaptive_background,
        diagonal_loading=float(adaptive_load),
        folds=int(crossfit_folds),
        max_iterations=int(matrix_free_iterations),
        tolerance=float(matrix_free_tolerance),
    )
    spectra_power: dict[str, np.ndarray] = {
        "savaux": savaux_power,
        "plain_coherent": np.abs(np.mean(plain_spectra, axis=0)).astype(np.float64) ** 2,
        "lora_coherent": np.abs(np.mean(lora_spectra, axis=0)).astype(np.float64) ** 2,
        "lora_mean_power": np.mean(np.abs(lora_spectra).astype(np.float64) ** 2, axis=0),
        "lora_best_power": np.max(np.abs(lora_spectra).astype(np.float64) ** 2, axis=0),
        "lora_stable_power": pattern_coherence_weighted_power(
            lora_spectra,
            reference_power=savaux_power,
            exponent=float(stable_exponent),
        ),
        "lora_adaptive_gls": adaptive_gls_spectrum_power(
            lora_spectra,
            covariance_bins=adaptive_background,
            diagonal_loading=float(adaptive_load),
        ),
        "lora_crossfit_gls": crossfit_gls_spectrum_power(
            lora_spectra,
            covariance_bins=adaptive_background,
            diagonal_loading=float(adaptive_load),
            folds=int(crossfit_folds),
        ),
        "lora_cg_gls": matrix_free_result.power,
        "lora_offpacket_gls": _fixed_gls_spectrum_power(lora_spectra, offpacket_covariance),
    }

    out: dict[str, Any] = {
        "packet_index": int(packet["packet_index"]),
        "payload_symbol_index": int(symbol["payload_symbol_index"]),
        "gt_bin": int(gt_bin),
        "bank_kind": str(bank.kind),
        "pattern_count": int(len(bank.names)),
    }
    for method, power in spectra_power.items():
        metrics = _power_metrics(power, gt_bin)
        out[f"{method}_selected_bin"] = int(metrics["selected_bin"])
        out[f"{method}_ok"] = int(metrics["selected_bin"]) == int(gt_bin)
        out[f"{method}_gt_rank"] = int(metrics["gt_rank"])
        out[f"{method}_gt_power"] = float(metrics["gt_power"])
        out[f"{method}_gt_margin_db"] = float(metrics["gt_margin_db"])
        out[f"{method}_gt_floor_db"] = float(metrics["gt_floor_db"])
    return out


def _evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    bank: NonuniformPatternBank,
    args: argparse.Namespace,
    offpacket_covariance: np.ndarray | None,
) -> list[dict[str, Any]]:
    os_factor = int(packet["os_factor"])
    origin_shift = int(args.origin_shift) if args.origin_shift is not None else os_factor // 2
    rows: list[dict[str, Any]] = []
    for symbol in packet["payload_symbols"]:
        rows.append(
            _evaluate_symbol(
                samples=samples,
                packet=packet,
                symbol=symbol,
                bank=bank,
                origin_shift=origin_shift,
                cfo_correction_mode=str(args.cfo_correction_mode),
                stable_exponent=float(args.stable_exponent),
                adaptive_load=float(args.adaptive_load),
                adaptive_exclude_top=int(args.adaptive_exclude_top),
                adaptive_guard_bins=int(args.adaptive_guard_bins),
                crossfit_folds=int(args.crossfit_folds),
                matrix_free_iterations=int(args.matrix_free_iterations),
                matrix_free_tolerance=float(args.matrix_free_tolerance),
                offpacket_covariance=offpacket_covariance,
            )
        )
    return rows


def _summary(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float | None, seed: int) -> dict[str, Any]:
    symbols = int(len(rows))
    gt_bins = [int(row["gt_bin"]) for row in rows]
    out: dict[str, Any] = {
        "dataset": dataset,
        "snr_db": "" if snr_db is None else float(snr_db),
        "seed": int(seed),
        "bank_kind": str(rows[0]["bank_kind"]) if rows else "",
        "pattern_count": int(rows[0]["pattern_count"]) if rows else 0,
        "symbol_count": symbols,
        "noise_mode": str(rows[0].get("noise_mode", "awgn")) if rows else "",
        "covariance_source": str(rows[0].get("covariance_source", "white")) if rows else "",
        "white_effective_replicas": float(rows[0].get("white_effective_replicas", 0.0)) if rows else 0.0,
        "empirical_effective_replicas": (
            float(rows[0].get("empirical_effective_replicas", 0.0)) if rows else 0.0
        ),
        "cross_validated_effective_replicas": (
            float(rows[0].get("cross_validated_effective_replicas", 0.0)) if rows else 0.0
        ),
    }
    savaux_selected = [int(row["savaux_selected_bin"]) for row in rows]
    for method in METHODS:
        selected = [int(row[f"{method}_selected_bin"]) for row in rows]
        err, compared = err_count(selected, gt_bins)
        out[f"{method}_ser"] = float(err / max(1, compared))
        out[f"{method}_mean_gt_rank"] = float(np.mean([int(row[f"{method}_gt_rank"]) for row in rows])) if rows else 0.0
        out[f"{method}_mean_gt_margin_db"] = (
            float(np.mean([float(row[f"{method}_gt_margin_db"]) for row in rows])) if rows else 0.0
        )
        out[f"{method}_mean_gt_floor_db"] = (
            float(np.mean([float(row[f"{method}_gt_floor_db"]) for row in rows])) if rows else 0.0
        )
        if method != "savaux":
            fix = 0
            break_count = 0
            changes = 0
            for idx in range(min(len(selected), len(savaux_selected), len(gt_bins))):
                cand = int(selected[idx])
                sav = int(savaux_selected[idx])
                gt = int(gt_bins[idx])
                if cand != sav:
                    changes += 1
                if sav != gt and cand == gt:
                    fix += 1
                if sav == gt and cand != gt:
                    break_count += 1
            out[f"{method}_fix_vs_savaux"] = int(fix)
            out[f"{method}_break_vs_savaux"] = int(break_count)
            out[f"{method}_changes_vs_savaux"] = int(changes)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_16"])
    parser.add_argument("--snrs", nargs="*", type=float, default=[-24.0, -25.0, -26.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-packets", type=int, default=2)
    parser.add_argument("--noise-mode", choices=("awgn", "offpacket_bootstrap"), default="awgn")
    parser.add_argument("--bootstrap-source-windows", type=int, default=256)
    parser.add_argument(
        "--bank-kinds",
        nargs="+",
        default=["fixed", "basic_only", "dither_only", "random_only", "balanced_random_only"],
    )
    parser.add_argument("--random-count", type=int, default=64)
    parser.add_argument("--bank-seed", type=int, default=0)
    parser.add_argument("--max-patterns", type=int, default=96)
    parser.add_argument("--subset-strategy", choices=("head", "diverse"), default="diverse")
    parser.add_argument("--stable-exponent", type=float, default=1.0)
    parser.add_argument("--adaptive-load", type=float, default=0.05)
    parser.add_argument("--adaptive-exclude-top", type=int, default=8)
    parser.add_argument("--adaptive-guard-bins", type=int, default=1)
    parser.add_argument("--crossfit-folds", type=int, default=4)
    parser.add_argument("--matrix-free-iterations", type=int, default=4)
    parser.add_argument("--matrix-free-tolerance", type=float, default=0.0)
    parser.add_argument("--covariance-source", choices=("white", "offpacket"), default="white")
    parser.add_argument("--offpacket-windows", type=int, default=128)
    parser.add_argument("--offpacket-bins", type=int, default=16)
    parser.add_argument("--offpacket-shrinkage", type=float, default=0.2)
    parser.add_argument("--offpacket-guard-symbols", type=float, default=2.0)
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
        default=WEAK_ROOT / "data" / "os_lora" / "pattern_fft_coherence",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    symbol_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = dataset_paths(str(dataset))
        clean = np.fromfile(iq_path, dtype=np.complex64)
        all_packets = load_packets(symbol_path)
        packets = all_packets
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
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
        for bank_kind in args.bank_kinds:
            if not packets:
                continue
            sf = int(packets[0]["sf"])
            os_factor = int(packets[0]["os_factor"])
            bank = build_pattern_bank(
                sf,
                os_factor,
                kind=str(bank_kind),
                random_count=int(args.random_count),
                seed=int(args.bank_seed),
            )
            bank = select_pattern_subset(bank, int(args.max_patterns), str(args.subset_strategy))
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
                    if str(args.covariance_source) == "offpacket":
                        (
                            covariance,
                            white_replicas,
                            empirical_replicas,
                            cross_validated_replicas,
                            _offpacket_windows,
                            _covariance_bins,
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
                    rows: list[dict[str, Any]] = []
                    for packet in packets:
                        rows.extend(
                            _evaluate_packet(
                                samples=samples,
                                packet=packet,
                                bank=bank,
                                args=args,
                                offpacket_covariance=covariance,
                            )
                        )
                    for row in rows:
                        row.update(
                            {
                                "dataset": str(dataset),
                                "snr_db": "" if snr_db is None else float(snr_db),
                                "seed": int(seed),
                                "noise_mode": str(args.noise_mode),
                                "covariance_source": str(args.covariance_source),
                                "white_effective_replicas": float(white_replicas),
                                "empirical_effective_replicas": float(empirical_replicas),
                                "cross_validated_effective_replicas": float(cross_validated_replicas),
                            }
                        )
                        symbol_rows.append(row)
                    summary = _summary(rows, str(dataset), snr_db, int(seed))
                    summary_rows.append(summary)
                    write_csv(out_dir / "summary.csv", summary_rows)
                    print(
                        f"{dataset} {bank.kind} snr={snr_db} seed={seed}: "
                        f"savaux={summary['savaux_ser']:.4f} "
                        f"plain={summary['plain_coherent_ser']:.4f} "
                        f"lora_coh={summary['lora_coherent_ser']:.4f} "
                        f"meanp={summary['lora_mean_power_ser']:.4f} "
                        f"bestp={summary['lora_best_power_ser']:.4f} "
                        f"stable={summary['lora_stable_power_ser']:.4f} "
                        f"adgls={summary['lora_adaptive_gls_ser']:.4f} "
                        f"xfgls={summary['lora_crossfit_gls_ser']:.4f} "
                        f"cggls={summary['lora_cg_gls_ser']:.4f} "
                        f"offgls={summary['lora_offpacket_gls_ser']:.4f} "
                        f"eff_cv={summary['cross_validated_effective_replicas']:.3f}",
                        flush=True,
                    )
    write_csv(out_dir / "symbol_metrics.csv", symbol_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
