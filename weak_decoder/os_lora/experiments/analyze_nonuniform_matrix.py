#!/usr/bin/env python3
"""从 H_b(m, K) 矩阵视角分析非均匀采样 pattern。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.common import write_csv  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import _oversampled_downchirp  # noqa: E402
from weak_decoder.chirp import build_upchirp  # noqa: E402
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    NonuniformPatternBank,
    build_pattern_bank,
    effective_replica_count,
    pattern_bin_values,
    pattern_noise_signatures,
    select_pattern_subset,
    target_response_vector,
)


def _rank_and_condition(covariance: np.ndarray) -> tuple[int, float]:
    eig = np.linalg.eigvalsh(np.asarray(covariance, dtype=np.complex128))
    eig = np.maximum(np.real(eig), 0.0)
    max_eig = float(np.max(eig)) if eig.size else 0.0
    tol = max(max_eig, 1.0) * 1e-9
    positive = eig[eig > tol]
    rank = int(positive.size)
    if positive.size == 0:
        return rank, float("inf")
    return rank, float(max_eig / float(np.min(positive)))


def _max_abs_corr(covariance: np.ndarray) -> float:
    cov = np.asarray(covariance, dtype=np.complex128)
    if cov.size == 0 or cov.shape[0] <= 1:
        return 0.0
    diag = np.sqrt(np.maximum(np.real(np.diag(cov)), 1e-30))
    corr = cov / (diag[:, None] * diag[None, :])
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.max(np.abs(corr[mask]))) if np.any(mask) else 0.0


def _wrong_bins(n_bins: int, raw_bin: int, count: int) -> tuple[int, ...]:
    if count <= 0:
        return tuple()
    neighbors = [raw_bin + delta for delta in (-4, -3, -2, -1, 1, 2, 3, 4)]
    spread = np.linspace(0, n_bins - 1, num=min(count, n_bins), dtype=np.int64).tolist()
    out: list[int] = []
    seen = {int(raw_bin) % n_bins}
    for value in neighbors + spread:
        item = int(value) % n_bins
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= count:
            break
    return tuple(out)


def _normalized_leakage(
    sf: int,
    os_factor: int,
    raw_bin: int,
    bank: NonuniformPatternBank,
    leakage_bins: int,
) -> tuple[float, float]:
    signatures = pattern_noise_signatures(bank, raw_bin)
    covariance = signatures @ signatures.conj().T
    target = target_response_vector(bank)
    pinv = np.linalg.pinv(covariance, rcond=1e-10)
    denom = complex(target.conj().T @ pinv @ target)
    if abs(denom) <= 1e-30:
        return 0.0, 0.0
    weights = pinv @ target / denom
    leaks: list[float] = []
    for wrong_bin in _wrong_bins(1 << int(sf), int(raw_bin), int(leakage_bins)):
        symbol = build_upchirp(sf, wrong_bin, os_factor)
        dechirped = symbol * _oversampled_downchirp(sf, os_factor, 0, 0.0)
        values = pattern_bin_values(dechirped=dechirped, raw_fft_bin=raw_bin, bank=bank)
        leaks.append(float(abs(complex(weights.conj().T @ values))))
    if not leaks:
        return 0.0, 0.0
    return float(np.max(leaks)), float(np.percentile(np.asarray(leaks, dtype=np.float64), 95.0))


def _analyze_one(
    sf: int,
    os_factor: int,
    raw_bin: int,
    bank_kind: str,
    random_count: int,
    bank_seed: int,
    leakage_bins: int,
    max_patterns: int,
    subset_strategy: str,
) -> dict[str, object]:
    bank = build_pattern_bank(
        sf,
        os_factor,
        kind=bank_kind,
        random_count=random_count,
        seed=bank_seed,
    )
    bank = select_pattern_subset(bank, int(max_patterns), str(subset_strategy))
    signatures = pattern_noise_signatures(bank, raw_bin)
    covariance = signatures @ signatures.conj().T
    rank, condition = _rank_and_condition(covariance)
    effective = effective_replica_count(bank, raw_bin)

    symbol = build_upchirp(sf, raw_bin, os_factor)
    dechirped = symbol * _oversampled_downchirp(sf, os_factor, 0, 0.0)
    target_values = pattern_bin_values(dechirped=dechirped, raw_fft_bin=raw_bin, bank=bank)
    target_amp = np.abs(target_values).astype(np.float64)
    target_phase = np.unwrap(np.angle(target_values))
    max_leak, p95_leak = _normalized_leakage(
        sf=sf,
        os_factor=os_factor,
        raw_bin=raw_bin,
        bank=bank,
        leakage_bins=leakage_bins,
    )

    return {
        "sf": int(sf),
        "os_factor": int(os_factor),
        "raw_bin": int(raw_bin),
        "bank_kind": str(bank.kind),
        "pattern_count": int(len(bank.names)),
        "matrix_rank": int(rank),
        "condition_number": float(condition),
        "max_abs_noise_corr": _max_abs_corr(covariance),
        "effective_replicas": float(effective),
        "target_amp_min": float(np.min(target_amp)) if target_amp.size else 0.0,
        "target_amp_mean": float(np.mean(target_amp)) if target_amp.size else 0.0,
        "target_amp_max": float(np.max(target_amp)) if target_amp.size else 0.0,
        "target_phase_std": float(np.std(target_phase)) if target_phase.size else 0.0,
        "normalized_max_leakage": float(max_leak),
        "normalized_p95_leakage": float(p95_leak),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--os-factor", type=int, default=4)
    parser.add_argument("--raw-bins", nargs="+", type=int, default=[1, 61, 301, 653, 900, 1023])
    parser.add_argument(
        "--bank-kinds",
        nargs="+",
        default=["fixed", "basic", "periodic", "dither", "random", "balanced_random", "search"],
    )
    parser.add_argument("--random-count", type=int, default=64)
    parser.add_argument("--bank-seed", type=int, default=0)
    parser.add_argument("--leakage-bins", type=int, default=128)
    parser.add_argument("--max-patterns", type=int, default=64)
    parser.add_argument("--subset-strategy", choices=("head", "diverse"), default="diverse")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "matrix_analysis",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for bank_kind in args.bank_kinds:
        for raw_bin in args.raw_bins:
            row = _analyze_one(
                sf=int(args.sf),
                os_factor=int(args.os_factor),
                raw_bin=int(raw_bin),
                bank_kind=str(bank_kind),
                random_count=int(args.random_count),
                bank_seed=int(args.bank_seed),
                leakage_bins=int(args.leakage_bins),
                max_patterns=int(args.max_patterns),
                subset_strategy=str(args.subset_strategy),
            )
            rows.append(row)
            print(
                f"{bank_kind:15s} k={int(raw_bin):4d} "
                f"patterns={row['pattern_count']:4d} "
                f"rank={row['matrix_rank']:4d} "
                f"eff={row['effective_replicas']:.6f} "
                f"target_amp={row['target_amp_mean']:.3f} "
                f"leak95={row['normalized_p95_leakage']:.3e}",
                flush=True,
            )
    out_dir = Path(args.output_dir).resolve()
    write_csv(out_dir / "matrix_metrics.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
