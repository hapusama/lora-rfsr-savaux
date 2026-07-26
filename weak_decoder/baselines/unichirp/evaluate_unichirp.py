#!/usr/bin/env python3
"""Evaluate the standalone UniChirp baseline.

This runner is symbol-level and FFT-bin-only.  UniChirp uses preamble and/or
header symbols to fit its packet-local phase model, then demodulates payload
symbols with dual-peak coherent fusion.  It intentionally does not import or
compare against retired phase-line experiments.
"""

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
    DEFAULT_DATASETS,
    dataset_paths,
    err_count,
    load_packets,
    mean_rows,
    noise_samples,
    payload_gt_bins,
    snr_values,
    sum_rows,
    write_csv,
)
from weak_decoder.baselines.unichirp import (  # noqa: E402
    UniChirpDemodConfig,
    UniChirpTrainingSymbol,
    build_unichirp_phase_model,
    demod_unichirp_symbol,
)


def _payload_abs_index(packet: dict[str, Any], payload_symbol_index: int) -> float:
    return float(packet.get("preamble_len", 8.0)) + 12.25 + float(payload_symbol_index)


def _header_abs_index(packet: dict[str, Any], header_symbol_index: int) -> float:
    return float(packet.get("preamble_len", 8.0)) + 4.25 + float(header_symbol_index)


def _training_symbols(packet: dict[str, Any], source: str) -> tuple[UniChirpTrainingSymbol, ...]:
    if source == "none":
        return tuple()
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    symbol_samples = (1 << sf) * os_factor
    preamble_len = int(round(float(packet.get("preamble_len", 8.0))))
    header_start = int(packet["header_start_sample"])
    items: list[UniChirpTrainingSymbol] = []
    if source in {"preamble", "preamble_header"}:
        preamble_start = int(round(header_start - (float(preamble_len) + 4.25) * symbol_samples))
        for idx in range(preamble_len):
            items.append(
                UniChirpTrainingSymbol(
                    start_sample=int(preamble_start + idx * symbol_samples),
                    raw_fft_bin=0,
                    abs_symbol_index=float(idx),
                    source="preamble",
                )
            )
    if source in {"header", "preamble_header"}:
        for symbol in packet["header_symbols"]:
            items.append(
                UniChirpTrainingSymbol(
                    start_sample=int(symbol["start_sample"]),
                    raw_fft_bin=int(symbol["raw_fft_bin"]),
                    abs_symbol_index=_header_abs_index(packet, int(symbol["stage_symbol_index"])),
                    source="header",
                )
            )
    return tuple(items)


def _evaluate_unichirp_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    config: UniChirpDemodConfig,
    training_source: str,
) -> dict[str, Any]:
    payload = list(packet["payload_symbols"])
    gt_bins = payload_gt_bins(packet)
    phase_model, observations = build_unichirp_phase_model(
        samples=samples,
        training_symbols=_training_symbols(packet, training_source),
        sf=int(packet["sf"]),
        os_factor=int(packet["os_factor"]),
        cfo_int=int(packet["cfo_int"]),
        cfo_frac=float(packet["cfo_frac"]),
        header_start_sample=int(packet["header_start_sample"]),
        config=config,
    )
    selected: list[int] = []
    margins: list[float] = []
    primary_powers: list[float] = []
    secondary_powers: list[float] = []
    for item in payload:
        abs_index = _payload_abs_index(packet, int(item["payload_symbol_index"]))
        result = demod_unichirp_symbol(
            samples=samples,
            start_sample=int(item["start_sample"]),
            sf=int(packet["sf"]),
            os_factor=int(packet["os_factor"]),
            phase_rad=phase_model.predict(abs_index),
            ldro=bool(packet["ldro"]),
            cfo_int=int(packet["cfo_int"]),
            cfo_frac=float(packet["cfo_frac"]),
            header_start_sample=int(packet["header_start_sample"]),
            config=config,
        )
        selected.append(int(result.raw_fft_bin))
        margins.append(float(result.peak_margin_db))
        primary_powers.append(float(result.primary_power))
        secondary_powers.append(float(result.secondary_power))
    errors, compared = err_count(selected, gt_bins)
    return {
        "packet_index": int(packet["packet_index"]),
        "symbol_count": int(compared),
        "unichirp_err": int(errors),
        "unichirp_ser": float(errors / max(1, compared)),
        "unichirp_phase_observations": int(len(observations)),
        "unichirp_phase_fit_count": int(phase_model.observation_count),
        "unichirp_phase_rmse_rad": float(phase_model.rmse_rad),
        "unichirp_phase_slope_rad": float(phase_model.slope_rad_per_symbol),
        "unichirp_phase_intercept_rad": float(phase_model.intercept_rad),
        "unichirp_phase_source": str(phase_model.source),
        "unichirp_mean_margin_db": float(np.mean(margins)) if margins else 0.0,
        "unichirp_mean_primary_power": float(np.mean(primary_powers)) if primary_powers else 0.0,
        "unichirp_mean_secondary_power": float(np.mean(secondary_powers)) if secondary_powers else 0.0,
    }


def _summary(rows: Sequence[dict[str, Any]], dataset: str, snr_db: float | None, seed: int) -> dict[str, Any]:
    symbols = sum_rows(rows, "symbol_count")
    unichirp_err = sum_rows(rows, "unichirp_err")
    return {
        "dataset": dataset,
        "snr_db": "" if snr_db is None else float(snr_db),
        "seed": int(seed),
        "packet_count": int(len(rows)),
        "symbol_count": int(symbols),
        "unichirp_ser": float(unichirp_err / max(1, symbols)),
        "unichirp_err": int(unichirp_err),
        "mean_unichirp_phase_fit_count": mean_rows(rows, "unichirp_phase_fit_count"),
        "mean_unichirp_phase_rmse_rad": mean_rows(rows, "unichirp_phase_rmse_rad"),
        "mean_unichirp_margin_db": mean_rows(rows, "unichirp_mean_margin_db"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_32"])
    parser.add_argument("--snrs", nargs="*", type=float, default=[-25.0, -26.0, -27.0, -28.0, -29.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-packets", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=THIS_FILE.parent / "_eval" / "unichirp")
    parser.add_argument("--signal-reference-power", type=float, default=None)
    parser.add_argument(
        "--training-source",
        choices=["preamble", "header", "preamble_header", "none"],
        default="preamble_header",
    )
    parser.add_argument("--disable-bandlimit-filter", action="store_true")
    parser.add_argument("--filter-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--min-dual-peak-ratio", type=float, default=1e-3)
    parser.add_argument("--list-default-datasets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.list_default_datasets):
        print(" ".join(DEFAULT_DATASETS))
        return 0

    config = UniChirpDemodConfig(
        enable_bandlimit_filter=not bool(args.disable_bandlimit_filter),
        filter_bandwidth_scale=float(args.filter_bandwidth_scale),
        min_dual_peak_ratio=float(args.min_dual_peak_ratio),
    )
    out_dir = Path(args.output_dir).resolve()
    packet_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in args.datasets:
        iq_path, symbol_path = dataset_paths(str(dataset))
        if not iq_path.exists():
            raise FileNotFoundError(iq_path)
        if not symbol_path.exists():
            raise FileNotFoundError(symbol_path)
        clean = np.fromfile(iq_path, dtype=np.complex64)
        packets = load_packets(symbol_path)
        if int(args.max_packets) > 0:
            packets = packets[: int(args.max_packets)]
        for seed in args.seeds:
            for snr_db in snr_values(args.snrs):
                samples = noise_samples(clean, snr_db, int(seed), args.signal_reference_power)
                rows: list[dict[str, Any]] = []
                for packet in packets:
                    row = _evaluate_unichirp_packet(
                        samples=samples,
                        packet=packet,
                        config=config,
                        training_source=str(args.training_source),
                    )
                    row.update(
                        {
                            "dataset": str(dataset),
                            "snr_db": "" if snr_db is None else float(snr_db),
                            "seed": int(seed),
                        }
                    )
                    rows.append(row)
                    packet_rows.append(row)
                summary = _summary(rows, str(dataset), snr_db, int(seed))
                summary_rows.append(summary)
                print(
                    f"{dataset} snr={snr_db} seed={seed}: "
                    f"unichirp_ser={summary['unichirp_ser']:.4f} "
                    f"unichirp_err={summary['unichirp_err']}",
                    flush=True,
                )

    write_csv(out_dir / "packet_metrics.csv", packet_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
