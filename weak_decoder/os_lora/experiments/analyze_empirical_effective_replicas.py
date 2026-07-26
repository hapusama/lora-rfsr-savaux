#!/usr/bin/env python3
"""使用包外协方差和目标响应检查估计有效副本数。"""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.common import dataset_paths, load_packets, write_csv  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import _oversampled_downchirp  # noqa: E402
from weak_decoder.chirp import build_upchirp  # noqa: E402
from weak_decoder.os_lora.experiment_support.noise_windows import (  # noqa: E402
    active_intervals as _active_intervals,
    covariance_correlation_stats as _corr_stats,
    covariance_range_residuals as _range_residuals,
    effective_replicas_with_noise_power as _effective_replicas_with_noise_power,
    empirical_covariance as _empirical_covariance,
    off_packet_starts as _off_packet_starts,
    raw_noise_stats as _raw_noise_stats,
)
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    NonuniformPatternBank,
    build_pattern_bank,
    pattern_bin_values,
    pattern_noise_signatures,
    prepare_dechirped_symbol,
    select_pattern_subset,
)


def _top_gt_bins(packets: Sequence[dict[str, Any]], count: int) -> tuple[int, ...]:
    hist: Counter[int] = Counter()
    for packet in packets:
        for symbol in packet.get("payload_symbols", []):
            hist[int(symbol["gt_bin"])] += 1
    return tuple(int(item) for item, _hits in hist.most_common(max(1, int(count))))


def _raw_noise_power(samples: np.ndarray, starts: Sequence[int], window_len: int, downchirp: np.ndarray) -> float:
    windows = np.asarray(
        [(samples[int(start): int(start) + window_len] * downchirp).astype(np.complex64) for start in starts],
        dtype=np.complex64,
    )
    return float(_raw_noise_stats(windows, os_factor=1, max_lag=1)["raw_noise_power"])


def _offpacket_covariance(
    samples: np.ndarray,
    starts: Sequence[int],
    sf: int,
    os_factor: int,
    raw_bin: int,
    bank: NonuniformPatternBank,
    downchirp: np.ndarray,
) -> np.ndarray:
    n_bins = 1 << int(sf)
    window_len = n_bins * int(os_factor)
    vectors: list[np.ndarray] = []
    for start in starts:
        chunk = np.asarray(samples[int(start): int(start) + window_len], dtype=np.complex64)
        dechirped = (chunk * downchirp).astype(np.complex64)
        vectors.append(pattern_bin_values(dechirped=dechirped, raw_fft_bin=int(raw_bin), bank=bank))
    return _empirical_covariance(np.asarray(vectors, dtype=np.complex128))


def _synthetic_target(sf: int, os_factor: int, raw_bin: int, bank: NonuniformPatternBank) -> np.ndarray:
    downchirp = _oversampled_downchirp(sf=int(sf), os_factor=int(os_factor), cfo_int=0, cfo_frac=0.0)
    ideal = build_upchirp(sf=int(sf), symbol_id=int(raw_bin), os_factor=int(os_factor)) * downchirp
    return pattern_bin_values(dechirped=ideal, raw_fft_bin=int(raw_bin), bank=bank)


def _packet_target(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    sf: int,
    os_factor: int,
    raw_bin: int,
    bank: NonuniformPatternBank,
) -> tuple[np.ndarray | None, int, float, float]:
    n_bins = 1 << int(sf)
    hard = np.full(len(bank.offsets), math.sqrt(float(n_bins)), dtype=np.complex128)
    normalized: list[np.ndarray] = []
    raw_vectors: list[np.ndarray] = []
    for packet in packets:
        origin_shift = int(os_factor) // 2
        header_start = int(packet["header_start_sample"]) + origin_shift
        for symbol in packet.get("payload_symbols", []):
            if int(symbol["gt_bin"]) != int(raw_bin):
                continue
            try:
                dechirped = prepare_dechirped_symbol(
                    samples=samples,
                    start_sample=int(symbol["start_sample"]) + origin_shift,
                    sf=int(sf),
                    os_factor=int(os_factor),
                    cfo_int=int(packet["cfo_int"]),
                    cfo_frac=float(packet["cfo_frac"]),
                    header_start_sample=header_start,
                    cfo_correction_mode="continuous",
                )
            except ValueError:
                continue
            values = pattern_bin_values(dechirped=dechirped, raw_fft_bin=int(raw_bin), bank=bank)
            # 使用硬目标约定 a = sqrt(N) * 1，消除未知复 symbol 幅度 x。
            hard_energy = max(float(np.real(hard.conj().T @ hard)), 1e-30)
            x_hat = complex(hard.conj().T @ values) / hard_energy
            if abs(x_hat) <= 1e-12:
                continue
            normalized.append(np.asarray(values / x_hat, dtype=np.complex128))
            raw_vectors.append(np.asarray(values, dtype=np.complex128))
    if not normalized:
        return None, 0, 0.0, 0.0
    matrix = np.asarray(normalized, dtype=np.complex128)
    target = np.mean(matrix, axis=0)
    residual = float(np.linalg.norm(target - hard) / max(float(np.linalg.norm(hard)), 1e-30))
    phase_std = float(np.std(np.unwrap(np.angle(target)))) if target.size else 0.0
    raw_power = float(np.mean(np.abs(np.asarray(raw_vectors, dtype=np.complex128)) ** 2)) if raw_vectors else 0.0
    return target, int(matrix.shape[0]), residual, max(phase_std, raw_power * 0.0)


def _eff(cov: np.ndarray, target: np.ndarray, n_bins: int, raw_noise_power: float, rcond: float) -> float:
    return _effective_replicas_with_noise_power(cov, target, n_bins, raw_noise_power, rcond)


def _analyze_one(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    starts: Sequence[int],
    sf: int,
    os_factor: int,
    raw_bin: int,
    bank: NonuniformPatternBank,
    raw_noise_power: float,
    shrinkage: float,
    downchirp: np.ndarray,
) -> dict[str, Any]:
    n_bins = 1 << int(sf)
    cov = _offpacket_covariance(samples, starts, sf, os_factor, raw_bin, bank, downchirp)
    signatures = pattern_noise_signatures(bank, int(raw_bin))
    white_cov = signatures @ signatures.conj().T
    shrunk = (1.0 - float(shrinkage)) * cov + float(shrinkage) * float(raw_noise_power) * white_cov

    target_hard = np.full(len(bank.offsets), math.sqrt(float(n_bins)), dtype=np.complex128)
    target_synth = _synthetic_target(sf, os_factor, raw_bin, bank)
    packet_target, packet_count, packet_resid, packet_phase_std = _packet_target(
        samples, packets, sf, os_factor, raw_bin, bank
    )

    synth_resid = float(np.linalg.norm(target_synth - target_hard) / max(float(np.linalg.norm(target_hard)), 1e-30))
    range_hard = _range_residuals(cov, target_hard, (1e-3, 1e-6))
    range_synth = _range_residuals(cov, target_synth, (1e-3, 1e-6))
    diag_cv, mean_abs_corr, max_abs_corr = _corr_stats(cov)
    row: dict[str, Any] = {
        "raw_bin": int(raw_bin),
        "bank_kind": str(bank.kind),
        "pattern_count": int(len(bank.offsets)),
        "offpacket_windows": int(len(starts)),
        "packet_target_count": int(packet_count),
        "diag_cv": float(diag_cv),
        "mean_abs_corr": float(mean_abs_corr),
        "max_abs_corr": float(max_abs_corr),
        "target_synth_vs_hard_residual": synth_resid,
        "target_packet_vs_hard_residual": float(packet_resid),
        "target_packet_phase_std": float(packet_phase_std),
        "range_hard_residual_1e_m3": range_hard["range_residual_1em03"],
        "range_hard_residual_1e_m6": range_hard["range_residual_1em06"],
        "range_synth_residual_1e_m3": range_synth["range_residual_1em03"],
        "range_synth_residual_1e_m6": range_synth["range_residual_1em06"],
        "eff_white_hard": _eff(float(raw_noise_power) * white_cov, target_hard, n_bins, raw_noise_power, 1e-10),
        "eff_empirical_hard": _eff(cov, target_hard, n_bins, raw_noise_power, 1e-3),
        "eff_shrink_hard": _eff(shrunk, target_hard, n_bins, raw_noise_power, 1e-3),
        "eff_empirical_synth": _eff(cov, target_synth, n_bins, raw_noise_power, 1e-3),
        "eff_shrink_synth": _eff(shrunk, target_synth, n_bins, raw_noise_power, 1e-3),
    }
    if packet_target is not None:
        range_packet = _range_residuals(cov, packet_target, (1e-3, 1e-6))
        row.update(
            {
                "range_packet_residual_1e_m3": range_packet["range_residual_1em03"],
                "range_packet_residual_1e_m6": range_packet["range_residual_1em06"],
                "eff_empirical_packet": _eff(cov, packet_target, n_bins, raw_noise_power, 1e-3),
                "eff_shrink_packet": _eff(shrunk, packet_target, n_bins, raw_noise_power, 1e-3),
            }
        )
    else:
        row.update(
            {
                "range_packet_residual_1e_m3": "",
                "range_packet_residual_1e_m6": "",
                "eff_empirical_packet": "",
                "eff_shrink_packet": "",
            }
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_8", "0_0_0_10_14_16", "0_0_0_10_14_32"])
    parser.add_argument("--raw-bins", nargs="*", type=int, default=None)
    parser.add_argument("--top-gt-bins", type=int, default=6)
    parser.add_argument("--bank-kinds", nargs="+", default=["fixed", "basic_only", "random_only", "balanced_random_only"])
    parser.add_argument("--random-count", type=int, default=64)
    parser.add_argument("--bank-seed", type=int, default=0)
    parser.add_argument("--max-patterns", type=int, default=64)
    parser.add_argument("--subset-strategy", choices=("head", "diverse"), default="diverse")
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--window-seed", type=int, default=1234)
    parser.add_argument("--guard-symbols", type=float, default=2.0)
    parser.add_argument("--shrinkage", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "empirical_effective_replicas",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    out_dir = Path(args.output_dir).resolve()
    for dataset in args.datasets:
        iq_path, symbol_path = dataset_paths(str(dataset))
        samples = np.fromfile(iq_path, dtype=np.complex64)
        packets = load_packets(symbol_path)
        if not packets:
            continue
        sf = int(packets[0]["sf"])
        os_factor = int(packets[0]["os_factor"])
        n_bins = 1 << sf
        window_len = n_bins * os_factor
        guard_samples = int(round(float(args.guard_symbols) * window_len))
        intervals = _active_intervals(packets, guard_samples=guard_samples)
        starts = _off_packet_starts(samples.size, window_len, intervals, int(args.max_windows), int(args.window_seed))
        if not starts:
            raise RuntimeError(f"no off-packet windows for {dataset}")
        downchirp = _oversampled_downchirp(sf=sf, os_factor=os_factor, cfo_int=0, cfo_frac=0.0)
        raw_noise_power = _raw_noise_power(samples, starts, window_len, downchirp)
        raw_bins = tuple(int(v) for v in args.raw_bins) if args.raw_bins else _top_gt_bins(packets, int(args.top_gt_bins))
        dataset_rows.append(
            {
                "dataset": str(dataset),
                "sf": int(sf),
                "os_factor": int(os_factor),
                "offpacket_windows": int(len(starts)),
                "raw_noise_power": float(raw_noise_power),
                "raw_bins": " ".join(str(v) for v in raw_bins),
            }
        )
        print(
            f"{dataset}: windows={len(starts)} raw_noise_power={raw_noise_power:.6g} "
            f"bins={' '.join(str(v) for v in raw_bins)}",
            flush=True,
        )
        for bank_kind in args.bank_kinds:
            bank = build_pattern_bank(
                sf,
                os_factor,
                kind=str(bank_kind),
                random_count=int(args.random_count),
                seed=int(args.bank_seed),
            )
            bank = select_pattern_subset(bank, int(args.max_patterns), str(args.subset_strategy))
            for raw_bin in raw_bins:
                row = _analyze_one(
                    samples=samples,
                    packets=packets,
                    starts=starts,
                    sf=sf,
                    os_factor=os_factor,
                    raw_bin=int(raw_bin),
                    bank=bank,
                    raw_noise_power=raw_noise_power,
                    shrinkage=float(args.shrinkage),
                    downchirp=downchirp,
                )
                row["dataset"] = str(dataset)
                rows.append(row)
                print(
                    f"{dataset} {bank.kind:24s} k={int(raw_bin):4d} "
                    f"white={row['eff_white_hard']:.3f} emp={row['eff_empirical_hard']:.3f} "
                    f"shrink={row['eff_shrink_hard']:.3f} packet={row['eff_shrink_packet']}",
                    flush=True,
                )
    write_csv(out_dir / "dataset_metrics.csv", dataset_rows)
    write_csv(out_dir / "effective_replica_metrics.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
