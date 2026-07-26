#!/usr/bin/env python3
"""Compare Savaux with three OSR-dimensional branch-GLS covariance scopes.

The four detectors use the same synchronized noisy symbols and the same ideal
Savaux branch steering.  They differ only in how the ``R x R`` covariance is
obtained:

* Savaux: equal coherent branch combining;
* fixed off-packet GLS: one covariance from noise-only windows;
* packet GLS: one covariance from payload null bins pooled within a packet;
* symbol null-bin GLS: one covariance from the current symbol's null bins.

Ground truth is used only after all four decisions have been made.  No
``RN x RN`` covariance, score blending, header steering, or reranking is used.
"""

from __future__ import annotations

import argparse
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
    write_csv,
)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.os_lora.experiment_support.noise_windows import (  # noqa: E402
    active_intervals,
    covariance_correlation_stats,
    off_packet_starts,
)
from weak_decoder.os_lora.system.noise import select_background_bins  # noqa: E402
from weak_decoder.os_lora.system.oversampled_glrt import (  # noqa: E402
    BranchNoiseModel,
    aligned_branch_observations,
    branch_gls_scores,
    estimate_branch_noise_model,
)


METHODS = (
    "savaux",
    "gls_offpacket_fixed",
    "gls_packet_null",
    "gls_symbol_null",
)

METHOD_LABELS = {
    "savaux": "Savaux",
    "gls_offpacket_fixed": "Fixed off-packet GLS",
    "gls_packet_null": "Per-packet null-bin GLS",
    "gls_symbol_null": "Per-symbol null-bin GLS",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_32"])
    parser.add_argument(
        "--snrs",
        nargs="*",
        type=float,
        default=[-22.0, -23.0, -24.0, -25.0, -26.0],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--noise-windows", type=int, default=128)
    parser.add_argument("--noise-training-bins", type=int, default=32)
    parser.add_argument("--noise-seed", type=int, default=4107)
    parser.add_argument("--exclude-top", type=int, default=8)
    parser.add_argument("--guard-bins", type=int, default=1)
    parser.add_argument("--diagonal-loading", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT
        / "data"
        / "experiments"
        / "branch_covariance_scopes",
    )
    return parser.parse_args()


def _regularized_branch_model(
    snapshots: np.ndarray,
    diagonal_loading: float,
    training_bins: Sequence[int] = (),
) -> BranchNoiseModel:
    values = np.asarray(snapshots, dtype=np.complex128)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("branch covariance requires at least two snapshots")
    dimension = int(values.shape[1])
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = centered.T @ centered.conj() / float(values.shape[0] - 1)
    covariance = 0.5 * (covariance + covariance.conj().T)
    scale = max(float(np.real(np.trace(covariance))) / float(dimension), 1e-30)
    covariance += (
        max(0.0, float(diagonal_loading))
        * scale
        * np.eye(dimension, dtype=np.complex128)
    )
    inverse = np.linalg.pinv(covariance, hermitian=True)
    steering = np.ones(dimension, dtype=np.complex128)
    information = float(max(np.real(np.vdot(steering, inverse @ steering)), 1e-30))
    return BranchNoiseModel(
        covariance=covariance,
        inverse_covariance=inverse,
        steering=steering,
        information=information,
        snapshot_count=int(values.shape[0]),
        training_bins=tuple(int(value) for value in training_bins),
        diagonal_loading=float(diagonal_loading),
    )


def _offpacket_windows(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    sf: int,
    os_factor: int,
    count: int,
    seed: int,
) -> np.ndarray:
    symbol_samples = (1 << int(sf)) * int(os_factor)
    starts = off_packet_starts(
        sample_count=int(samples.size),
        window_len=symbol_samples,
        intervals=active_intervals(packets, guard_samples=16 * symbol_samples),
        max_windows=int(count),
        seed=int(seed),
    )
    if len(starts) < 2:
        raise RuntimeError("fewer than two off-packet windows are available")
    return np.asarray(
        [samples[start : start + symbol_samples] for start in starts],
        dtype=np.complex64,
    )


def _training_bins(n_bins: int, count: int) -> tuple[int, ...]:
    use = min(max(1, int(count)), int(n_bins))
    return tuple(
        int(value)
        for value in np.linspace(0, int(n_bins), use, endpoint=False, dtype=np.int64)
    )


def _null_snapshots(
    observations: np.ndarray,
    savaux_power: np.ndarray,
    exclude_top: int,
    guard_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    background = select_background_bins(
        savaux_power,
        exclude_top=int(exclude_top),
        guard_bins=int(guard_bins),
    )
    if background.size < 2:
        raise RuntimeError("null-bin selection produced fewer than two snapshots")
    return np.asarray(observations[background], dtype=np.complex128), background


def _covariance_row(
    dataset: str,
    snr_db: float | None,
    seed: int,
    packet_index: int | str,
    symbol_index: int | str,
    scope: str,
    model: BranchNoiseModel,
) -> dict[str, Any]:
    diagonal_cv, mean_correlation, max_correlation = covariance_correlation_stats(
        np.asarray(model.covariance)
    )
    eigenvalues = np.linalg.eigvalsh(np.asarray(model.covariance, dtype=np.complex128))
    condition = float(
        np.max(eigenvalues) / max(float(np.min(eigenvalues)), 1e-30)
    )
    return {
        "dataset": dataset,
        "snr_db": "" if snr_db is None else float(snr_db),
        "seed": int(seed),
        "packet_index": packet_index,
        "payload_symbol_index": symbol_index,
        "scope": scope,
        "dimension": int(model.covariance.shape[-1]),
        "snapshots": int(model.snapshot_count),
        "diagonal_loading": float(model.diagonal_loading),
        "diagonal_cv": float(diagonal_cv),
        "mean_abs_correlation": float(mean_correlation),
        "max_abs_correlation": float(max_correlation),
        "condition_number": condition,
    }


def _evaluate_realization(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    fixed_model: BranchNoiseModel,
    args: argparse.Namespace,
    dataset: str,
    snr_db: float | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    symbol_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = [
        _covariance_row(
            dataset, snr_db, seed, "", "", "offpacket_fixed", fixed_model
        )
    ]
    for packet in packets:
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        origin_shift = os_factor // 2
        header_start = int(packet["header_start_sample"]) + origin_shift
        records: list[dict[str, Any]] = []
        packet_snapshots: list[np.ndarray] = []
        for symbol in packet["payload_symbols"]:
            try:
                combined, branches, _ = paper_oversampled_spectrum(
                    samples=samples,
                    start_sample=int(symbol["start_sample"]) + origin_shift,
                    sf=sf,
                    os_factor=os_factor,
                    cfo_int=int(packet["cfo_int"]),
                    cfo_frac=float(packet["cfo_frac"]),
                    header_start_sample=header_start,
                    cfo_correction_mode="continuous",
                )
            except ValueError:
                continue
            savaux_power = np.abs(combined).astype(np.float64) ** 2
            observations = aligned_branch_observations(branches, os_factor)
            null_values, background = _null_snapshots(
                observations,
                savaux_power,
                int(args.exclude_top),
                int(args.guard_bins),
            )
            packet_snapshots.append(null_values)
            records.append(
                {
                    "symbol": symbol,
                    "branches": branches,
                    "savaux_power": savaux_power,
                    "null_values": null_values,
                    "background": background,
                }
            )
        if not records:
            continue
        packet_model = _regularized_branch_model(
            np.concatenate(packet_snapshots, axis=0),
            float(args.diagonal_loading),
        )
        covariance_rows.append(
            _covariance_row(
                dataset,
                snr_db,
                seed,
                int(packet["packet_index"]),
                "",
                "packet_null",
                packet_model,
            )
        )
        for record in records:
            symbol = record["symbol"]
            branches = record["branches"]
            savaux_power = record["savaux_power"]
            symbol_model = _regularized_branch_model(
                record["null_values"],
                float(args.diagonal_loading),
                record["background"],
            )
            decisions = {
                "savaux": int(np.argmax(savaux_power)),
                "gls_offpacket_fixed": int(
                    branch_gls_scores(
                        branches, os_factor, noise_model=fixed_model, top_l=1
                    ).selected_bin
                ),
                "gls_packet_null": int(
                    branch_gls_scores(
                        branches, os_factor, noise_model=packet_model, top_l=1
                    ).selected_bin
                ),
                "gls_symbol_null": int(
                    branch_gls_scores(
                        branches, os_factor, noise_model=symbol_model, top_l=1
                    ).selected_bin
                ),
            }
            gt = int(symbol["gt_bin"])
            row: dict[str, Any] = {
                "dataset": dataset,
                "snr_db": "" if snr_db is None else float(snr_db),
                "seed": int(seed),
                "packet_index": int(packet["packet_index"]),
                "payload_symbol_index": int(symbol["payload_symbol_index"]),
                "frame_symbol_index": int(symbol["frame_symbol_index"]),
                "gt_bin": gt,
                "null_bin_count": int(record["background"].size),
            }
            for method, decision in decisions.items():
                row[f"{method}_bin"] = int(decision)
                row[f"{method}_error"] = int(decision != gt)
            symbol_rows.append(row)
            covariance_rows.append(
                _covariance_row(
                    dataset,
                    snr_db,
                    seed,
                    int(packet["packet_index"]),
                    int(symbol["payload_symbol_index"]),
                    "symbol_null",
                    symbol_model,
                )
            )
    return symbol_rows, covariance_rows


def _summary_by_seed(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["snr_db"]), int(row["seed"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (dataset, _snr_text, seed), selected in groups.items():
        snr_value = selected[0]["snr_db"]
        savaux_errors = sum(int(row["savaux_error"]) for row in selected)
        for method in METHODS:
            errors = sum(int(row[f"{method}_error"]) for row in selected)
            fixes = sum(
                int(row["savaux_error"] and not row[f"{method}_error"])
                for row in selected
            )
            breaks = sum(
                int(not row["savaux_error"] and row[f"{method}_error"])
                for row in selected
            )
            output.append(
                {
                    "dataset": dataset,
                    "snr_db": snr_value,
                    "seed": seed,
                    "method": method,
                    "errors": int(errors),
                    "symbol_count": int(len(selected)),
                    "ser": float(errors / max(1, len(selected))),
                    "savaux_errors": int(savaux_errors),
                    "fixes_vs_savaux": int(fixes),
                    "breaks_vs_savaux": int(breaks),
                }
            )
    output.sort(key=lambda row: (str(row["dataset"]), str(row["snr_db"]), int(row["seed"]), METHODS.index(str(row["method"]))))
    return output


def _aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["snr_db"]), str(row["method"]))
        item = groups.setdefault(
            key,
            {
                "dataset": row["dataset"],
                "snr_db": row["snr_db"],
                "method": row["method"],
                "errors": 0,
                "symbol_count": 0,
                "fixes_vs_savaux": 0,
                "breaks_vs_savaux": 0,
            },
        )
        for field in ("errors", "symbol_count", "fixes_vs_savaux", "breaks_vs_savaux"):
            item[field] += int(row[field])
    output = list(groups.values())
    for row in output:
        row["ser"] = float(row["errors"] / max(1, row["symbol_count"]))
        row["net_fixes_vs_savaux"] = int(
            row["fixes_vs_savaux"] - row["breaks_vs_savaux"]
        )
    output.sort(key=lambda row: (str(row["dataset"]), str(row["snr_db"]), METHODS.index(str(row["method"]))))
    return output


def _plot(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"plot skipped: {exc}")
        return
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        selected = [
            row
            for row in rows
            if str(row["dataset"]) == dataset and str(row["snr_db"]).strip()
        ]
        if not selected:
            continue
        figure, axis = plt.subplots(figsize=(7.6, 4.8))
        for method in METHODS:
            values = sorted(
                [row for row in selected if str(row["method"]) == method],
                key=lambda row: float(row["snr_db"]),
            )
            axis.plot(
                [float(row["snr_db"]) for row in values],
                [float(row["ser"]) for row in values],
                marker="o",
                linewidth=1.8,
                label=METHOD_LABELS[method],
            )
        axis.set_xlabel("Injected SNR (dB)")
        axis.set_ylabel("Payload SER")
        axis.set_title(dataset)
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"ser_{dataset}.png", dpi=160)
        plt.close(figure)


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_symbols: list[dict[str, Any]] = []
    all_covariances: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = dataset_paths(str(dataset))
        packets = load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        if not packets:
            raise RuntimeError(f"no packets found for {dataset}")
        clean = np.memmap(iq_path, dtype=np.complex64, mode="r")
        reference_power, _, _ = signal_reference_power(clean, packets, "packet", None)
        sf_values = {int(packet["sf"]) for packet in packets}
        os_values = {int(packet["os_factor"]) for packet in packets}
        if len(sf_values) != 1 or len(os_values) != 1:
            raise ValueError("one comparison run requires a common SF and OSR")
        sf = next(iter(sf_values))
        os_factor = next(iter(os_values))
        n_bins = 1 << sf
        snr_values: list[float | None] = list(float(value) for value in args.snrs)
        if bool(args.include_clean):
            snr_values.insert(0, None)
        for snr_db in snr_values:
            seeds = (int(args.seeds[0]),) if snr_db is None else tuple(int(v) for v in args.seeds)
            for seed in seeds:
                samples = noise_samples(
                    clean,
                    snr_db,
                    seed,
                    reference_power,
                    noise_shape="white",
                    os_factor=os_factor,
                )
                windows = _offpacket_windows(
                    samples,
                    packets,
                    sf,
                    os_factor,
                    int(args.noise_windows),
                    int(args.noise_seed),
                )
                fixed_model = estimate_branch_noise_model(
                    windows,
                    sf,
                    os_factor,
                    training_bins=_training_bins(n_bins, int(args.noise_training_bins)),
                    diagonal_loading=float(args.diagonal_loading),
                    covariance_mode="pooled",
                )
                symbol_rows, covariance_rows = _evaluate_realization(
                    samples,
                    packets,
                    fixed_model,
                    args,
                    str(dataset),
                    snr_db,
                    seed,
                )
                all_symbols.extend(symbol_rows)
                all_covariances.extend(covariance_rows)
                summary = _summary_by_seed(symbol_rows)
                print(
                    f"{dataset} snr={'clean' if snr_db is None else f'{snr_db:g}'} seed={seed}: "
                    + " ".join(
                        f"{row['method']}={row['ser']:.4f}"
                        for row in summary
                    ),
                    flush=True,
                )
    by_seed = _summary_by_seed(all_symbols)
    aggregate = _aggregate(by_seed)
    write_csv(output_dir / "symbols.csv", all_symbols)
    write_csv(output_dir / "covariance.csv", all_covariances)
    write_csv(output_dir / "summary_by_seed.csv", by_seed)
    write_csv(output_dir / "summary.csv", aggregate)
    _plot(aggregate, output_dir)
    (output_dir / "config.txt").write_text(
        "\n".join(
            (
                f"datasets={' '.join(str(value) for value in args.datasets)}",
                f"snrs={' '.join(str(value) for value in args.snrs)}",
                f"seeds={' '.join(str(value) for value in args.seeds)}",
                f"include_clean={int(bool(args.include_clean))}",
                f"max_packets={int(args.max_packets)}",
                f"noise_windows={int(args.noise_windows)}",
                f"noise_training_bins={int(args.noise_training_bins)}",
                f"exclude_top={int(args.exclude_top)}",
                f"guard_bins={int(args.guard_bins)}",
                f"diagonal_loading={float(args.diagonal_loading)}",
                "noise_shape=white",
                "steering=ideal_ones",
            )
        )
        + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
