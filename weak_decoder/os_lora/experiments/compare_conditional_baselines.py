"""在同一份 IQ 上比较条件式 LoRa GLS 与独立 baseline。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

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
    write_csv,
)
from weak_decoder.baselines.run_ser_comparison import _evaluate_group  # noqa: E402
from weak_decoder.os_lora.experiment_support.conditional_gls import (  # noqa: E402
    build_selected_pattern_banks as _build_selected_banks,
    estimate_training_covariance as _training_covariance,
    evaluate_conditional_condition as _evaluate_conditional_condition,
)
from weak_decoder.os_lora.experiment_support.pattern_training import (  # noqa: E402
    bootstrap_offpacket_noise as _offpacket_bootstrap_samples,
)
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    build_pattern_bank,
    select_pattern_subset,
)


METHODS = (
    "conditional_lora",
    "savaux_oversampled",
    "loratrimmer",
    "symfec",
    "unichirp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snrs", nargs="+", type=float, default=[-23.0, -24.0, -25.0])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-packets", type=int, default=2)
    parser.add_argument("--noise-mode", choices=("offpacket_bootstrap", "awgn"), default="offpacket_bootstrap")
    parser.add_argument("--candidate-kind", default="canonical")
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--pattern-count", type=int, default=8)
    parser.add_argument("--training-seed", type=int, default=41)
    parser.add_argument("--training-snr", type=float, default=-24.0)
    parser.add_argument("--training-windows", type=int, default=64)
    parser.add_argument("--training-bins", type=int, default=8)
    parser.add_argument("--bootstrap-source-windows", type=int, default=256)
    parser.add_argument("--guard-symbols", type=float, default=2.0)
    parser.add_argument("--exclude-top", type=int, default=8)
    parser.add_argument("--exclude-guard-bins", type=int, default=1)
    parser.add_argument("--gls-loading", type=float, default=0.05)
    parser.add_argument("--crossfit-folds", type=int, default=2)
    parser.add_argument("--cg-iterations", type=int, default=4)
    parser.add_argument("--cg-tolerance", type=float, default=0.0)
    parser.add_argument("--wrap-consistency-exponent", type=float, default=0.5)
    parser.add_argument("--wrap-minimum-segment", type=int, default=16)
    parser.add_argument("--savaux-gate-margin-db", type=float, default=0.5)
    parser.add_argument("--branch-color-threshold", type=float, default=0.107121)
    parser.add_argument("--skip-loratrimmer", action="store_true")
    parser.add_argument("--lora-leakage-weight", type=float, default=0.0)
    parser.add_argument("--runtime-lora-weight", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "conditional_vs_baselines",
    )
    return parser.parse_args()


def _condition_samples(
    clean: np.ndarray,
    packets: list[dict[str, Any]],
    reference_power: float,
    snr_db: float,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if str(args.noise_mode) == "awgn":
        return noise_samples(clean, float(snr_db), int(seed), float(reference_power))
    return _offpacket_bootstrap_samples(
        clean=clean,
        packets=packets,
        snr_db=float(snr_db),
        seed=int(seed),
        reference_power=float(reference_power),
        max_source_windows=int(args.bootstrap_source_windows),
        guard_symbols=float(args.guard_symbols),
    )


def main() -> int:
    args = parse_args()
    iq_path, symbol_path = dataset_paths(str(args.dataset))
    clean = np.fromfile(iq_path, dtype=np.complex64)
    all_packets = load_packets(symbol_path)
    packets = all_packets[: int(args.max_packets)] if int(args.max_packets) > 0 else all_packets
    if not packets:
        raise RuntimeError(f"no payload packets found for {args.dataset}")
    reference_power, _samples, _packet_count = signal_reference_power(clean, packets, "packet", None)

    full_bank = build_pattern_bank(
        int(packets[0]["sf"]),
        int(packets[0]["os_factor"]),
        kind=str(args.candidate_kind),
    )
    candidate_bank = (
        select_pattern_subset(full_bank, int(args.candidate_limit), "diverse")
        if int(args.candidate_limit) > 0
        else full_bank
    )
    if int(args.pattern_count) >= len(candidate_bank.names):
        covariance = np.eye(len(candidate_bank.names), dtype=np.complex128)
        training_snapshots = 0
    else:
        covariance, training_snapshots = _training_covariance(
            clean,
            all_packets,
            reference_power,
            candidate_bank,
            args,
        )
    banks, selection_rows = _build_selected_banks(
        candidate_bank,
        covariance,
        [int(args.pattern_count)],
        float(args.gls_loading),
        0.0,
    )

    rows: list[dict[str, Any]] = []
    for seed in args.test_seeds:
        for snr_db in args.snrs:
            samples = _condition_samples(
                clean,
                all_packets,
                reference_power,
                float(snr_db),
                int(seed),
                args,
            )
            detector_rows, _diagnostics = _evaluate_conditional_condition(samples, packets, banks, args)
            detector = next(row for row in detector_rows if row["method"] != "savaux")
            detector_savaux = next(row for row in detector_rows if row["method"] == "savaux")
            baseline = _evaluate_group(
                samples=samples,
                packets=packets,
                include_loratrimmer=not bool(args.skip_loratrimmer),
            )
            symbol_count = int(baseline["symbol_count"])
            if int(detector["symbol_count"]) != symbol_count:
                raise RuntimeError("conditional detector and baselines compared different symbol counts")
            if int(detector_savaux["errors"]) != int(baseline["savaux_oversampled_err"]):
                raise RuntimeError("Savaux implementations disagree on the shared IQ realization")
            row: dict[str, Any] = {
                "dataset": str(args.dataset),
                "noise_mode": str(args.noise_mode),
                "snr_db": float(snr_db),
                "seed": int(seed),
                "packet_count": int(baseline["packet_count"]),
                "symbol_count": symbol_count,
                "candidate_count": len(candidate_bank.names),
                "pattern_count": int(args.pattern_count),
                "training_snapshots": int(training_snapshots),
                "symfec_crc_valid_count": int(baseline["symfec_crc_valid_count"]),
                "conditional_lora_err": int(detector["errors"]),
                "conditional_lora_ser": float(detector["ser"]),
            }
            for method in METHODS[1:]:
                errors = int(baseline[f"{method}_err"])
                row[f"{method}_err"] = errors
                row[f"{method}_ser"] = "" if errors < 0 else float(errors / max(1, symbol_count))
            rows.append(row)
            print(
                f"{args.dataset} snr={snr_db:g} seed={seed}: "
                + " ".join(
                    f"{method}={float(row[f'{method}_ser']):.4f}"
                    for method in METHODS
                    if row.get(f"{method}_ser", "") != ""
                ),
                flush=True,
            )

    summary: list[dict[str, Any]] = []
    for snr_db in args.snrs:
        condition = [row for row in rows if float(row["snr_db"]) == float(snr_db)]
        symbol_count = sum(int(row["symbol_count"]) for row in condition)
        out: dict[str, Any] = {
            "dataset": str(args.dataset),
            "noise_mode": str(args.noise_mode),
            "snr_db": float(snr_db),
            "symbol_count": int(symbol_count),
            "candidate_count": len(candidate_bank.names),
            "pattern_count": int(args.pattern_count),
        }
        for method in METHODS:
            values = [
                int(row[f"{method}_err"])
                for row in condition
                if row.get(f"{method}_err", "") != "" and int(row[f"{method}_err"]) >= 0
            ]
            errors = sum(values)
            out[f"{method}_err"] = int(errors)
            out[f"{method}_ser"] = float(errors / max(1, symbol_count)) if values else ""
        summary.append(out)

    output_dir = Path(args.output_dir).resolve()
    write_csv(output_dir / "selection.csv", selection_rows)
    write_csv(output_dir / "summary_by_seed.csv", rows)
    write_csv(output_dir / "summary_by_snr.csv", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
