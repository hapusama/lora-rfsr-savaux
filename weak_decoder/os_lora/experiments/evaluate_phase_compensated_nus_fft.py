#!/usr/bin/env python3
"""Evaluate coarse phase-compensated NUS followed by ordinary dechirp FFTs.

Each NUS pattern selects one OSR sample per chip.  A plain FFT first supplies a
coarse bin without payload ground truth.  That bin is then used to translate
the selected samples to the q=0 polyphase reference, including the LoRa wrap
phase.  The translated N-sample sequences are processed only by ordinary FFTs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from weak_decoder.baselines.common import load_packets, write_csv
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from weak_decoder.os_lora.system.nonuniform_sampling import (
    NonuniformPatternBank,
    build_pattern_bank,
    pattern_bank_spectra,
    pattern_sample_matrix,
    plain_pattern_fft_spectra,
    prepare_dechirped_symbol,
)


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
COLLECTOR_ROOT = WEAK_ROOT / "USRP_collector" / "data" / "branch4_fixed" / "low_snr"
DEFAULT_TEST_IQ = COLLECTOR_ROOT / "sf10_bw125_fs500_pre32_sw34_low4.bin"
DEFAULT_TEST_SYMBOLS = (
    WEAK_ROOT
    / "data"
    / "experiments"
    / "real_low_snr_20260718_low4_low6"
    / "low4_win4"
    / "fft_symbols.csv"
)
DEFAULT_GT = (
    WEAK_ROOT
    / "data"
    / "groundtruth"
    / "branch4_fixed"
    / "high_snr"
    / "sf10_bw125_fs500_pre32_sw34_r001_fft_bin_groundtruth.csv"
)
DEFAULT_OUTPUT = WEAK_ROOT / "data" / "experiments" / "phase_compensated_nus_fft_20260721"

METHODS = (
    "savaux",
    "plain_nus_mean",
    "phase_fft_1",
    "phase_fft_2",
    "exact_nus_coherent",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-iq", type=Path, default=DEFAULT_TEST_IQ)
    parser.add_argument("--test-symbols", type=Path, default=DEFAULT_TEST_SYMBOLS)
    parser.add_argument("--groundtruth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bank-kind", default="canonical_only")
    parser.add_argument("--snrs", nargs="*", type=float, default=[-3.0, -6.0, -9.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 21)))
    parser.add_argument("--max-symbols", type=int, default=0)
    return parser.parse_args()


def _groundtruth_bins(path: Path) -> dict[int, int]:
    output: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output[int(row["frame_symbol_index"])] = int(row["groundtruth_fft_bin"])
    if not output:
        raise RuntimeError(f"no ground truth rows found in {path}")
    return output


def _load_packets_with_external_gt(symbol_path: Path, groundtruth: dict[int, int]) -> list[dict[str, Any]]:
    packets = load_packets(symbol_path)
    for packet in packets:
        for symbol in packet["payload_symbols"]:
            frame_symbol_index = int(symbol["frame_symbol_index"])
            if frame_symbol_index not in groundtruth:
                raise KeyError(f"ground truth has no frame_symbol_index={frame_symbol_index}")
            symbol["gt_bin"] = int(groundtruth[frame_symbol_index])
    return packets


def phase_compensated_fft_spectra(
    dechirped: np.ndarray,
    bank: NonuniformPatternBank,
    compensation_bin: int,
) -> np.ndarray:
    """Translate selected samples to q=0, then apply ordinary N-point FFTs.

    The compensation bin is a receiver estimate, not ground truth.  For a
    correct estimate the phase factor removes both fractional-chip timing and
    the candidate's LoRa wrap discontinuity from every NUS pattern.
    """

    n_bins = 1 << int(bank.sf)
    os_factor = int(bank.os_factor)
    k = int(compensation_bin) % n_bins
    samples = pattern_sample_matrix(dechirped=dechirped, bank=bank).astype(np.complex128)
    if samples.size == 0:
        return np.zeros((0, n_bins), dtype=np.complex128)
    offsets = np.stack(
        [np.mod(np.asarray(item, dtype=np.int64), os_factor) for item in bank.offsets]
    ).astype(np.float64)
    p = np.arange(n_bins, dtype=np.float64)[None, :]

    correction = np.exp(-2j * np.pi * offsets * float(k) / float(n_bins * os_factor))
    if k != 0:
        tail = p >= float(n_bins - k)
        correction *= np.where(tail, np.exp(2j * np.pi * offsets / float(os_factor)), 1.0)
    corrected = samples * correction
    return np.asarray(np.fft.fft(corrected, axis=1) / math.sqrt(float(n_bins)), dtype=np.complex128)


def phase_compensated_nus_powers(
    dechirped: np.ndarray,
    bank: NonuniformPatternBank,
    iterations: int = 2,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[int, ...]]:
    """Return the plain coarse power and iterative compensated FFT powers."""

    plain = plain_pattern_fft_spectra(dechirped=dechirped, bank=bank)
    if plain.size == 0:
        n_bins = 1 << int(bank.sf)
        return np.zeros(n_bins, dtype=np.float64), tuple(), tuple()
    coarse_power = np.mean(np.abs(plain).astype(np.float64) ** 2, axis=0)
    estimate = int(np.argmax(coarse_power))
    powers: list[np.ndarray] = []
    estimates: list[int] = [estimate]
    for _ in range(max(0, int(iterations))):
        spectra = phase_compensated_fft_spectra(dechirped, bank, estimate)
        power = np.abs(np.mean(spectra, axis=0)).astype(np.float64) ** 2
        powers.append(np.asarray(power, dtype=np.float64))
        estimate = int(np.argmax(power))
        estimates.append(estimate)
    return np.asarray(coarse_power, dtype=np.float64), tuple(powers), tuple(estimates)


def _score(power: np.ndarray, gt_bin: int) -> dict[str, float | int]:
    values = np.asarray(power, dtype=np.float64)
    gt = int(gt_bin) % int(values.size)
    selected = int(np.argmax(values))
    gt_power = float(values[gt])
    false = values.copy()
    false[gt] = -np.inf
    max_false = float(np.max(false))
    guard = np.ones(values.size, dtype=bool)
    for delta in (-1, 0, 1):
        guard[(gt + delta) % values.size] = False
    floor = float(np.median(values[guard])) if np.any(guard) else float(np.mean(values))
    mean_energy = max(float(np.mean(values)), 1e-30)
    return {
        "bin": selected,
        "correct": int(selected == gt),
        "gt_power": gt_power,
        "gt_to_mean_energy": float(gt_power / mean_energy),
        "gt_margin_db": float(10.0 * math.log10((gt_power + 1e-30) / (max_false + 1e-30))),
        "gt_floor_db": float(10.0 * math.log10((gt_power + 1e-30) / (floor + 1e-30))),
    }


def _payload_reference_power(samples: np.ndarray, packets: Sequence[dict[str, Any]]) -> float:
    total = 0.0
    count = 0
    for packet in packets:
        length = (1 << int(packet["sf"])) * int(packet["os_factor"])
        origin_shift = int(packet["os_factor"]) // 2
        for symbol in packet["payload_symbols"]:
            start = int(symbol["start_sample"]) + origin_shift
            chunk = np.asarray(samples[start : start + length], dtype=np.complex128)
            total += float(np.sum(np.abs(chunk) ** 2, dtype=np.float64))
            count += int(chunk.size)
    if count <= 0:
        raise RuntimeError("no payload samples available")
    return float(total / count)


def _add_awgn(
    clean: np.ndarray,
    added_power: float,
    seed: int,
    snr_index: int,
    packet_index: int,
    payload_symbol_index: int,
) -> np.ndarray:
    if added_power <= 0.0:
        return np.asarray(clean, dtype=np.complex64).copy()
    sequence = np.random.SeedSequence(
        [int(seed), int(snr_index), int(packet_index), int(payload_symbol_index)]
    )
    rng = np.random.default_rng(sequence)
    sigma = math.sqrt(float(added_power) / 2.0)
    noise = rng.normal(0.0, sigma, clean.size) + 1j * rng.normal(0.0, sigma, clean.size)
    return np.asarray(clean + noise, dtype=np.complex64)


def _two_sided_sign_p(fixes: int, breaks: int) -> float:
    discordant = int(fixes) + int(breaks)
    if discordant <= 0:
        return 1.0
    smaller = min(int(fixes), int(breaks))
    log_two = math.log(2.0)
    log_terms = [
        math.lgamma(discordant + 1.0)
        - math.lgamma(index + 1.0)
        - math.lgamma(discordant - index + 1.0)
        - discordant * log_two
        for index in range(smaller + 1)
    ]
    largest = max(log_terms)
    tail = math.exp(largest) * math.fsum(math.exp(value - largest) for value in log_terms)
    return float(min(1.0, 2.0 * tail))


def _summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    labels = sorted({str(row["snr_label"]) for row in rows}, key=lambda value: (value != "clean", value))
    for label in labels:
        subset = [row for row in rows if str(row["snr_label"]) == label]
        for method in METHODS:
            errors = sum(1 - int(row[f"{method}_correct"]) for row in subset)
            fixes = sum(
                int(row["savaux_correct"]) == 0 and int(row[f"{method}_correct"]) == 1
                for row in subset
            )
            breaks = sum(
                int(row["savaux_correct"]) == 1 and int(row[f"{method}_correct"]) == 0
                for row in subset
            )
            output.append(
                {
                    "snr_label": label,
                    "added_snr_db": "" if label == "clean" else float(label),
                    "method": method,
                    "decisions": len(subset),
                    "errors": int(errors),
                    "ser": float(errors / max(1, len(subset))),
                    "fixes_vs_savaux": int(fixes),
                    "breaks_vs_savaux": int(breaks),
                    "paired_sign_p": _two_sided_sign_p(fixes, breaks),
                    "mean_gt_to_mean_energy": float(
                        np.mean([float(row[f"{method}_gt_to_mean_energy"]) for row in subset])
                    ),
                    "mean_gt_margin_db": float(
                        np.mean([float(row[f"{method}_gt_margin_db"]) for row in subset])
                    ),
                    "mean_gt_floor_db": float(
                        np.mean([float(row[f"{method}_gt_floor_db"]) for row in subset])
                    ),
                }
            )
    return output


def _write_results(path: Path, summaries: Sequence[dict[str, Any]], bank: NonuniformPatternBank) -> None:
    lines = [
        "# Phase-compensated NUS ordinary-FFT audit",
        "",
        f"Pattern bank: `{bank.kind}` ({len(bank.names)} patterns).",
        "The compensation bin is obtained from the plain NUS FFT bank; payload GT is scoring-only.",
        "`mean_gt_to_mean_energy` is GT power divided by the mean candidate power, not an H0-normalized Lambda.",
        "",
        "| added SNR | method | decisions | errors | fixes / breaks | GT/mean energy | GT margin | GT/floor |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        label = str(row["snr_label"])
        lines.append(
            f"| {label} | {row['method']} | {row['decisions']} | {row['errors']} "
            f"| {row['fixes_vs_savaux']} / {row['breaks_vs_savaux']} "
            f"| {float(row['mean_gt_to_mean_energy']):.4f} "
            f"| {float(row['mean_gt_margin_db']):+.4f} dB "
            f"| {float(row['mean_gt_floor_db']):.4f} dB |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    packets = _load_packets_with_external_gt(
        args.test_symbols.resolve(), _groundtruth_bins(args.groundtruth.resolve())
    )
    samples = np.memmap(args.test_iq.resolve(), dtype=np.complex64, mode="r")
    if not packets:
        raise RuntimeError("no payload packets available")
    sf = int(packets[0]["sf"])
    os_factor = int(packets[0]["os_factor"])
    length = (1 << sf) * os_factor
    bank = build_pattern_bank(sf, os_factor, kind=str(args.bank_kind))
    if not bank.offsets:
        raise RuntimeError("pattern bank is empty")
    reference_power = _payload_reference_power(samples, packets)

    cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for packet in packets:
        for symbol in packet["payload_symbols"]:
            cases.append((packet, symbol))
    if int(args.max_symbols) > 0:
        cases = cases[: int(args.max_symbols)]

    rows: list[dict[str, Any]] = []
    conditions: list[float | None] = [None] + [float(value) for value in args.snrs]
    for snr_index, snr_db in enumerate(conditions):
        seeds: Sequence[int] = (0,) if snr_db is None else tuple(int(value) for value in args.seeds)
        added_power = (
            0.0
            if snr_db is None
            else float(reference_power / (10.0 ** (float(snr_db) / 10.0)))
        )
        for seed in seeds:
            for packet, symbol in cases:
                origin_shift = os_factor // 2
                start = int(symbol["start_sample"]) + origin_shift
                header_start = int(packet["header_start_sample"]) + origin_shift
                clean = np.asarray(samples[start : start + length], dtype=np.complex64)
                noisy = _add_awgn(
                    clean,
                    added_power,
                    int(seed),
                    snr_index,
                    int(packet["packet_index"]),
                    int(symbol["payload_symbol_index"]),
                )
                relative_header_start = int(header_start - start)
                savaux_spectrum, _branches, _phase = paper_oversampled_spectrum(
                    samples=noisy,
                    start_sample=0,
                    sf=sf,
                    os_factor=os_factor,
                    cfo_int=int(packet["cfo_int"]),
                    cfo_frac=float(packet["cfo_frac"]),
                    header_start_sample=relative_header_start,
                    cfo_correction_mode="continuous",
                )
                dechirped = prepare_dechirped_symbol(
                    samples=noisy,
                    start_sample=0,
                    sf=sf,
                    os_factor=os_factor,
                    cfo_int=int(packet["cfo_int"]),
                    cfo_frac=float(packet["cfo_frac"]),
                    header_start_sample=relative_header_start,
                    cfo_correction_mode="continuous",
                )
                plain_power, phase_powers, estimates = phase_compensated_nus_powers(
                    dechirped, bank, iterations=2
                )
                if len(phase_powers) != 2 or len(estimates) != 3:
                    raise RuntimeError("phase compensation did not produce two iterations")
                exact_spectra = pattern_bank_spectra(dechirped=dechirped, bank=bank)
                powers = {
                    "savaux": np.abs(savaux_spectrum).astype(np.float64) ** 2,
                    "plain_nus_mean": plain_power,
                    "phase_fft_1": phase_powers[0],
                    "phase_fft_2": phase_powers[1],
                    "exact_nus_coherent": np.abs(np.mean(exact_spectra, axis=0)).astype(np.float64) ** 2,
                }
                gt_bin = int(symbol["gt_bin"])
                row: dict[str, Any] = {
                    "snr_label": "clean" if snr_db is None else f"{float(snr_db):g}",
                    "added_snr_db": "" if snr_db is None else float(snr_db),
                    "seed": int(seed),
                    "packet_index": int(packet["packet_index"]),
                    "payload_symbol_index": int(symbol["payload_symbol_index"]),
                    "frame_symbol_index": int(symbol["frame_symbol_index"]),
                    "gt_bin": gt_bin,
                    "coarse_bin": int(estimates[0]),
                    "phase_fft_1_estimate": int(estimates[1]),
                    "phase_fft_2_estimate": int(estimates[2]),
                }
                for method, power in powers.items():
                    for key, value in _score(power, gt_bin).items():
                        row[f"{method}_{key}"] = value
                rows.append(row)
        snr_label = "clean" if snr_db is None else f"{float(snr_db):g}"
        print(f"snr={snr_label} rows={len(seeds) * len(cases)}", flush=True)

    summaries = _summaries(rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "symbols.csv", rows)
    write_csv(output_dir / "summary.csv", summaries)
    _write_results(output_dir / "RESULTS.md", summaries, bank)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "test_iq": str(args.test_iq.resolve()),
        "test_symbols": str(args.test_symbols.resolve()),
        "groundtruth": str(args.groundtruth.resolve()),
        "output_dir": str(output_dir),
        "sf": sf,
        "os_factor": os_factor,
        "symbol_count": len(cases),
        "signal_reference_power": reference_power,
        "pattern_names": list(bank.names),
        "gt_usage": "scoring only; coarse and compensation bins are receiver estimates",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} symbol trials to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
