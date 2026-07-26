#!/usr/bin/env python3
"""Probe one-sample NUS changes around the candidate-specific LoRa wrap chip."""

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
from weak_decoder.os_lora.system.nonuniform_sampling import prepare_dechirped_symbol


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
DEFAULT_OUTPUT = WEAK_ROOT / "data" / "experiments" / "wrap_chip_sample_choice_20260722"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-iq", type=Path, default=DEFAULT_TEST_IQ)
    parser.add_argument("--test-symbols", type=Path, default=DEFAULT_TEST_SYMBOLS)
    parser.add_argument("--groundtruth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chip-deltas", nargs="+", type=int, default=[-1, 0, 1])
    parser.add_argument("--snrs", nargs="*", type=float, default=[])
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


def wrap_chip_choice_spectrum(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    chip_delta: int,
    selected_offset: int,
) -> np.ndarray:
    """Use fixed q=0 except one sample near each candidate's own wrap chip.

    The selected sample is translated to the q=0 phase reference before it
    replaces the corresponding term in the ordinary q=0 dechirp FFT.
    Candidate k=0 has no in-symbol wrap and is left unchanged.
    """

    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    symbol = np.asarray(dechirped, dtype=np.complex128)
    if symbol.size != n_bins * os_value:
        raise ValueError("dechirped symbol length mismatch")
    q = int(selected_offset) % os_value
    baseline = np.asarray(np.fft.fft(symbol[0::os_value]) / math.sqrt(float(n_bins)), dtype=np.complex128)
    if q == 0:
        return baseline

    candidates = np.arange(n_bins, dtype=np.int64)
    wrap = n_bins - candidates
    chip = wrap + int(chip_delta)
    valid = (candidates > 0) & (chip >= 0) & (chip < n_bins)
    if not np.any(valid):
        return baseline

    k = candidates[valid].astype(np.float64)
    p = chip[valid].astype(np.int64)
    old_sample = symbol[os_value * p]
    new_sample = symbol[os_value * p + q]
    correction = np.exp(-2j * np.pi * k * float(q) / float(n_bins * os_value))
    post_wrap = p >= wrap[valid]
    correction *= np.where(post_wrap, np.exp(2j * np.pi * float(q) / float(os_value)), 1.0)
    kernel = np.exp(-2j * np.pi * k * p.astype(np.float64) / float(n_bins))
    output = baseline.copy()
    output[valid] += (new_sample * correction - old_sample) * kernel / math.sqrt(float(n_bins))
    return output


def gt_segment_metrics(
    dechirped: np.ndarray,
    sf: int,
    os_factor: int,
    gt_bin: int,
    chip_delta: int,
    selected_offset: int,
) -> dict[str, float | int]:
    """Measure GT-aligned head/tail coherence after one sample replacement."""

    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    k = int(gt_bin) % n_bins
    wrap = n_bins - k
    chip = wrap + int(chip_delta)
    if k == 0 or chip < 0 or chip >= n_bins:
        return {
            "segment_valid": 0,
            "head_energy": float("nan"),
            "tail_energy": float("nan"),
            "total_energy": float("nan"),
            "head_tail_coherence": float("nan"),
            "head_tail_phase_deg": float("nan"),
        }

    p = np.arange(n_bins, dtype=np.int64)
    offsets = np.zeros(n_bins, dtype=np.int64)
    offsets[chip] = int(selected_offset) % os_value
    picked = np.asarray(dechirped[os_value * p + offsets], dtype=np.complex128)
    correction = np.exp(
        -2j * np.pi * float(k) * offsets.astype(np.float64) / float(n_bins * os_value)
    )
    post_wrap = p >= wrap
    correction[post_wrap] *= np.exp(
        2j * np.pi * offsets[post_wrap].astype(np.float64) / float(os_value)
    )
    kernel = np.exp(-2j * np.pi * float(k) * p.astype(np.float64) / float(n_bins))
    aligned = picked * correction * kernel
    head = complex(np.sum(aligned[:wrap], dtype=np.complex128))
    tail = complex(np.sum(aligned[wrap:], dtype=np.complex128))
    head_length = wrap
    tail_length = n_bins - wrap
    coherence_denom = max((abs(head) + abs(tail)) ** 2, 1e-30)
    return {
        "segment_valid": 1,
        "head_energy": float(abs(head) ** 2 / max(1, head_length)),
        "tail_energy": float(abs(tail) ** 2 / max(1, tail_length)),
        "total_energy": float(abs(head + tail) ** 2 / float(n_bins)),
        "head_tail_coherence": float(abs(head + tail) ** 2 / coherence_denom),
        "head_tail_phase_deg": float(np.degrees(np.angle(head * np.conjugate(tail)))),
    }


def _score(power: np.ndarray, gt_bin: int) -> dict[str, float | int]:
    values = np.asarray(power, dtype=np.float64)
    gt = int(gt_bin) % int(values.size)
    selected = int(np.argmax(values))
    gt_power = float(values[gt])
    false = values.copy()
    false[gt] = -np.inf
    max_false = float(np.max(false))
    return {
        "bin": selected,
        "correct": int(selected == gt),
        "gt_power": gt_power,
        "gt_to_mean_energy": float(gt_power / max(float(np.mean(values)), 1e-30)),
        "gt_margin_db": float(10.0 * math.log10((gt_power + 1e-30) / (max_false + 1e-30))),
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
    return float(total / max(1, count))


def _add_awgn(
    clean: np.ndarray,
    added_power: float,
    seed: int,
    condition_index: int,
    packet_index: int,
    payload_symbol_index: int,
) -> np.ndarray:
    if added_power <= 0.0:
        return np.asarray(clean, dtype=np.complex64).copy()
    sequence = np.random.SeedSequence(
        [int(seed), int(condition_index), int(packet_index), int(payload_symbol_index)]
    )
    rng = np.random.default_rng(sequence)
    sigma = math.sqrt(float(added_power) / 2.0)
    noise = rng.normal(0.0, sigma, clean.size) + 1j * rng.normal(0.0, sigma, clean.size)
    return np.asarray(clean + noise, dtype=np.complex64)


def _finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {(str(row["snr_label"]), int(row["chip_delta"]), int(row["selected_offset"])) for row in rows},
        key=lambda item: (item[0] != "clean", item[0], item[1], item[2]),
    )
    for snr_label, chip_delta, selected_offset in keys:
        subset = [
            row
            for row in rows
            if str(row["snr_label"]) == snr_label
            and int(row["chip_delta"]) == chip_delta
            and int(row["selected_offset"]) == selected_offset
        ]
        errors = sum(1 - int(row["choice_correct"]) for row in subset)
        fixed_errors = sum(1 - int(row["fixed_q0_correct"]) for row in subset)
        savaux_errors = sum(1 - int(row["savaux_correct"]) for row in subset)
        fixes = sum(
            int(row["fixed_q0_correct"]) == 0 and int(row["choice_correct"]) == 1
            for row in subset
        )
        breaks = sum(
            int(row["fixed_q0_correct"]) == 1 and int(row["choice_correct"]) == 0
            for row in subset
        )
        output.append(
            {
                "snr_label": snr_label,
                "added_snr_db": "" if snr_label == "clean" else float(snr_label),
                "chip_delta": chip_delta,
                "selected_offset": selected_offset,
                "decisions": len(subset),
                "fixed_q0_errors": int(fixed_errors),
                "choice_errors": int(errors),
                "savaux_errors": int(savaux_errors),
                "fixes_vs_fixed_q0": int(fixes),
                "breaks_vs_fixed_q0": int(breaks),
                "mean_gt_to_mean_energy": _finite_mean(
                    [float(row["choice_gt_to_mean_energy"]) for row in subset]
                ),
                "mean_gt_margin_db": _finite_mean(
                    [float(row["choice_gt_margin_db"]) for row in subset]
                ),
                "mean_gt_energy_gain_db": _finite_mean(
                    [float(row["gt_energy_gain_db"]) for row in subset]
                ),
                "gt_energy_improved_symbols": sum(float(row["gt_energy_gain_db"]) > 0.0 for row in subset),
                "segment_valid_count": sum(int(row["segment_valid"]) for row in subset),
                "mean_head_energy_gain_db": _finite_mean(
                    [float(row["head_energy_gain_db"]) for row in subset]
                ),
                "mean_tail_energy_gain_db": _finite_mean(
                    [float(row["tail_energy_gain_db"]) for row in subset]
                ),
                "mean_total_segment_gain_db": _finite_mean(
                    [float(row["total_segment_gain_db"]) for row in subset]
                ),
                "mean_head_tail_coherence_delta": _finite_mean(
                    [float(row["head_tail_coherence_delta"]) for row in subset]
                ),
                "mean_abs_phase_error_reduction_deg": _finite_mean(
                    [float(row["abs_phase_error_reduction_deg"]) for row in subset]
                ),
            }
        )
    return output


def _write_results(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# LoRa wrap-chip sample-choice audit",
        "",
        "All chips use fixed q=0 except one candidate-specific chip near p_wrap=N-k.",
        "The changed sample is phase-translated back to q=0 before candidate scoring.",
        "GT is used only for energy and head/tail diagnostics after every candidate score is formed.",
        "",
        "| SNR | delta | q | fixed errors | choice errors | Savaux errors | fixes/breaks | GT energy gain | head gain | tail gain | coherence delta | phase-error reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['snr_label']} | {row['chip_delta']} | {row['selected_offset']} "
            f"| {row['fixed_q0_errors']} | {row['choice_errors']} | {row['savaux_errors']} "
            f"| {row['fixes_vs_fixed_q0']}/{row['breaks_vs_fixed_q0']} "
            f"| {float(row['mean_gt_energy_gain_db']):+.6f} dB "
            f"| {float(row['mean_head_energy_gain_db']):+.6f} dB "
            f"| {float(row['mean_tail_energy_gain_db']):+.6f} dB "
            f"| {float(row['mean_head_tail_coherence_delta']):+.3e} "
            f"| {float(row['mean_abs_phase_error_reduction_deg']):+.6f} deg |"
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
    n_bins = 1 << sf
    length = n_bins * os_factor
    reference_power = _payload_reference_power(samples, packets)
    cases = [
        (packet, symbol)
        for packet in packets
        for symbol in packet["payload_symbols"]
    ]
    if int(args.max_symbols) > 0:
        cases = cases[: int(args.max_symbols)]

    rows: list[dict[str, Any]] = []
    conditions: list[float | None] = [None] + [float(value) for value in args.snrs]
    for condition_index, snr_db in enumerate(conditions):
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
                    condition_index,
                    int(packet["packet_index"]),
                    int(symbol["payload_symbol_index"]),
                )
                relative_header_start = header_start - start
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
                fixed_spectrum = wrap_chip_choice_spectrum(dechirped, sf, os_factor, 0, 0)
                gt_bin = int(symbol["gt_bin"])
                fixed_score = _score(np.abs(fixed_spectrum) ** 2, gt_bin)
                savaux_score = _score(np.abs(savaux_spectrum) ** 2, gt_bin)
                baseline_segments = {
                    delta: gt_segment_metrics(dechirped, sf, os_factor, gt_bin, delta, 0)
                    for delta in args.chip_deltas
                }
                for chip_delta in args.chip_deltas:
                    baseline_segment = baseline_segments[int(chip_delta)]
                    for selected_offset in range(os_factor):
                        spectrum = wrap_chip_choice_spectrum(
                            dechirped, sf, os_factor, int(chip_delta), selected_offset
                        )
                        choice_score = _score(np.abs(spectrum) ** 2, gt_bin)
                        segments = gt_segment_metrics(
                            dechirped,
                            sf,
                            os_factor,
                            gt_bin,
                            int(chip_delta),
                            selected_offset,
                        )

                        def gain_db(key: str) -> float:
                            before = float(baseline_segment[key])
                            after = float(segments[key])
                            if not np.isfinite(before) or not np.isfinite(after):
                                return float("nan")
                            return float(10.0 * math.log10((after + 1e-30) / (before + 1e-30)))

                        row: dict[str, Any] = {
                            "snr_label": "clean" if snr_db is None else f"{float(snr_db):g}",
                            "added_snr_db": "" if snr_db is None else float(snr_db),
                            "seed": int(seed),
                            "packet_index": int(packet["packet_index"]),
                            "payload_symbol_index": int(symbol["payload_symbol_index"]),
                            "frame_symbol_index": int(symbol["frame_symbol_index"]),
                            "gt_bin": gt_bin,
                            "chip_delta": int(chip_delta),
                            "selected_offset": int(selected_offset),
                            "fixed_q0_bin": int(fixed_score["bin"]),
                            "fixed_q0_correct": int(fixed_score["correct"]),
                            "savaux_bin": int(savaux_score["bin"]),
                            "savaux_correct": int(savaux_score["correct"]),
                            **{f"choice_{key}": value for key, value in choice_score.items()},
                            **segments,
                            "gt_energy_gain_db": float(
                                10.0
                                * math.log10(
                                    (float(choice_score["gt_power"]) + 1e-30)
                                    / (float(fixed_score["gt_power"]) + 1e-30)
                                )
                            ),
                            "head_energy_gain_db": gain_db("head_energy"),
                            "tail_energy_gain_db": gain_db("tail_energy"),
                            "total_segment_gain_db": gain_db("total_energy"),
                            "head_tail_coherence_delta": float(
                                float(segments["head_tail_coherence"])
                                - float(baseline_segment["head_tail_coherence"])
                            ),
                            "abs_phase_error_reduction_deg": float(
                                abs(float(baseline_segment["head_tail_phase_deg"]))
                                - abs(float(segments["head_tail_phase_deg"]))
                            ),
                        }
                        rows.append(row)
        label = "clean" if snr_db is None else f"{float(snr_db):g}"
        print(f"snr={label} trials={len(seeds) * len(cases)}", flush=True)

    summaries = _summaries(rows)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "symbols.csv", rows)
    write_csv(output_dir / "summary.csv", summaries)
    _write_results(output_dir / "RESULTS.md", summaries)
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
        "fixed_uniform_offset": 0,
        "gt_usage": "scoring and head/tail diagnostics only",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
