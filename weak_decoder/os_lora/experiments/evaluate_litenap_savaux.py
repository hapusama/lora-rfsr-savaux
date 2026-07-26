#!/usr/bin/env python3
"""Compare LiteNap-Savaux sub-Nyquist detectors with original Savaux.

The clean high-SNR capture supplies synchronized symbol boundaries and
consensus FFT-bin ground truth. Controlled complex AWGN is added only to
copied payload-symbol observations. Every detector sees the same noisy
realization, and ground truth is used only after hard-bin decisions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from weak_decoder.baselines.common import load_packets, noise_samples, write_csv
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
    paper_oversampled_spectrum,
)
from weak_decoder.os_lora.system.litenap_savaux import (
    choose_downsample_phases,
    rerank_with_phase_fingerprint,
    selected_sample_count,
    subnyquist_component_spectra_batch,
)
from weak_decoder.os_lora.system.nonuniform_sampling import (
    prepare_dechirped_symbol,
)
from weak_decoder.os_lora.system.oversampled_glrt import (
    HeaderBinCalibration,
    estimate_header_bin_correction,
)


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
CAPTURE_STEM = "sf10_bw125_fs500_pre32_sw34_r001"
DEFAULT_IQ = (
    WEAK_ROOT
    / "USRP_collector"
    / "data"
    / "branch4_fixed"
    / "high_snr"
    / f"{CAPTURE_STEM}.bin"
)
DEFAULT_SYMBOLS = (
    WEAK_ROOT
    / "data"
    / "groundtruth"
    / "branch4_fixed"
    / "high_snr"
    / f"{CAPTURE_STEM}_fft_symbols.csv"
)
DEFAULT_GT = (
    WEAK_ROOT
    / "data"
    / "groundtruth"
    / "branch4_fixed"
    / "high_snr"
    / f"{CAPTURE_STEM}_fft_bin_groundtruth.csv"
)
DEFAULT_OUTPUT = (
    WEAK_ROOT / "data" / "experiments" / "litenap_savaux_clean_gt_20260724"
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    branch_phases: tuple[int, ...]
    downsample_phases: tuple[int, ...]
    phase_fingerprint: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-iq", type=Path, default=DEFAULT_IQ)
    parser.add_argument("--symbols", type=Path, default=DEFAULT_SYMBOLS)
    parser.add_argument("--groundtruth", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--snrs",
        nargs="*",
        type=float,
        default=[-22.0, -24.0, -26.0],
        help=(
            "Signal-to-added-AWGN ratios in dB. Clean is always evaluated first. "
            "This equals the negative of noisy_iq noise_power_db_relative."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument(
        "--view-counts",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Numbers K of downsampling phases used with all Savaux phases.",
    )
    parser.add_argument(
        "--fingerprint-weight",
        type=float,
        default=0.05,
        help="Weight of the training-free LiteNap phase-jump statistic; 0 disables it.",
    )
    parser.add_argument("--fingerprint-min-segment", type=int, default=8)
    parser.add_argument(
        "--min-header-bin-consensus",
        type=float,
        default=0.75,
        help=(
            "Minimum explicit-header modulo-four consensus for applying a "
            "packet-wide +/-1 residual-bin correction."
        ),
    )
    parser.add_argument("--origin-shift", type=int, default=None)
    parser.add_argument("--max-packets", type=int, default=0)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--verify-savaux-symbols",
        type=int,
        default=8,
        help="Assert full-view numerical equivalence against the existing Savaux implementation.",
    )
    return parser.parse_args()


def _groundtruth_bins(path: Path) -> dict[int, int]:
    output: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("stage", "")).strip().lower() != "payload":
                continue
            if str(row.get("consensus_reliable", "1")).strip() not in {
                "1",
                "true",
                "True",
            }:
                continue
            output[int(row["frame_symbol_index"])] = int(row["groundtruth_fft_bin"])
    if not output:
        raise RuntimeError(f"no reliable payload ground truth found in {path}")
    return output


def _load_cases(
    symbol_path: Path,
    groundtruth: dict[int, int],
    max_packets: int,
    max_symbols: int,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    packets = [
        packet
        for packet in load_packets(symbol_path)
        if bool(packet.get("header_valid", False))
    ]
    if int(max_packets) > 0:
        packets = packets[: int(max_packets)]
    cases: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for packet in packets:
        for symbol in packet["payload_symbols"]:
            frame_symbol_index = int(symbol["frame_symbol_index"])
            if frame_symbol_index not in groundtruth:
                raise KeyError(
                    f"ground truth has no frame_symbol_index={frame_symbol_index}"
                )
            symbol["gt_bin"] = int(groundtruth[frame_symbol_index])
            cases.append((packet, symbol))
    if int(max_symbols) > 0:
        cases = cases[: int(max_symbols)]
    if not cases:
        raise RuntimeError("no synchronized payload symbols available")
    return packets, cases


def _method_specs(
    sf: int,
    os_factor: int,
    downsample_factor: int,
    view_counts: Sequence[int],
    fingerprint_weight: float,
) -> tuple[MethodSpec, ...]:
    all_q = tuple(range(int(os_factor)))
    all_d = tuple(range(int(downsample_factor)))
    specs = [
        MethodSpec("savaux", all_q, all_d),
        MethodSpec("litenap_single", (0,), (0,)),
    ]
    if float(fingerprint_weight) > 0.0:
        specs.append(
            MethodSpec("savaux_phase", all_q, all_d, True)
        )
        specs.append(
            MethodSpec("litenap_single_phase", (0,), (0,), True)
        )
    seen = {0, int(downsample_factor)}
    for count in sorted(int(value) for value in view_counts):
        if count <= 0 or count > int(downsample_factor):
            raise ValueError(
                f"view counts must be within [1, {downsample_factor}], got {count}"
            )
        if count in seen:
            continue
        seen.add(count)
        phases = choose_downsample_phases(
            os_factor, downsample_factor, count
        )
        specs.append(MethodSpec(f"litenap_savaux_k{count}", all_q, phases))
        if float(fingerprint_weight) > 0.0:
            specs.append(
                MethodSpec(
                    f"litenap_savaux_k{count}_phase",
                    all_q,
                    phases,
                    True,
                )
            )
    return tuple(specs)


def _payload_reference_power(
    samples: np.ndarray,
    cases: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    origin_shift: int,
) -> float:
    total = 0.0
    count = 0
    for packet, symbol in cases:
        length = (1 << int(packet["sf"])) * int(packet["os_factor"])
        start = int(symbol["start_sample"]) + int(origin_shift)
        chunk = np.asarray(samples[start : start + length], dtype=np.complex64)
        if chunk.size != length:
            continue
        total += float(np.sum(np.abs(chunk).astype(np.float64) ** 2))
        count += int(chunk.size)
    if count <= 0:
        raise RuntimeError("no complete payload observations for reference power")
    return float(total / count)


def _packet_header_bin_calibration(
    samples: np.ndarray,
    packet: dict[str, Any],
    origin_shift: int,
    minimum_consensus: float,
) -> HeaderBinCalibration:
    """Estimate a packet-wide residual integer-bin offset without payload GT."""

    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    header_start = int(packet["header_start_sample"]) + int(origin_shift)
    observed_bins: list[int] = []
    for symbol in packet["header_symbols"]:
        combined, _branches, _phase = paper_oversampled_spectrum(
            samples,
            int(symbol["start_sample"]) + int(origin_shift),
            sf,
            os_factor,
            int(packet["cfo_int"]),
            float(packet["cfo_frac"]),
            header_start,
            "continuous",
        )
        observed_bins.append(
            int(np.argmax(np.abs(combined).astype(np.float64) ** 2))
        )
    return estimate_header_bin_correction(
        observed_bins,
        1 << sf,
        float(minimum_consensus),
    )


def _derived_noise_seed(
    seed: int,
    condition_index: int,
    packet_index: int,
    payload_symbol_index: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(condition_index),
            int(packet_index),
            int(payload_symbol_index),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _spectrum_power(spectrum: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(spectrum)).astype(np.float64) ** 2


def _combine_batch(
    components: np.ndarray,
    spec: MethodSpec,
) -> np.ndarray:
    selected = components[
        :,
        np.asarray(spec.branch_phases, dtype=np.int64)[:, None],
        np.asarray(spec.downsample_phases, dtype=np.int64)[None, :],
        :,
    ]
    return np.sum(selected, axis=(1, 2), dtype=np.complex128).astype(
        np.complex64
    )


def _score_power(power: np.ndarray, gt_bin: int) -> dict[str, float | int]:
    values = np.asarray(power, dtype=np.float64)
    gt = int(gt_bin) % int(values.size)
    selected = int(np.argmax(values))
    false = values.copy()
    false[gt] = -np.inf
    max_false = float(np.max(false))
    gt_power = float(values[gt])
    rank = int(1 + np.count_nonzero(values > gt_power))
    return {
        "bin": selected,
        "correct": int(selected == gt),
        "gt_rank": rank,
        "gt_margin_db": float(
            10.0
            * math.log10((gt_power + 1e-30) / (max_false + 1e-30))
        ),
    }


def _alias_error_mode(
    selected_bin: int,
    gt_bin: int,
    n_bins: int,
    downsample_factor: int,
) -> str:
    """Classify a hard-decision error into alias-bin or alias-group failure."""

    if int(n_bins) <= 0 or int(downsample_factor) <= 0:
        raise ValueError("n_bins and downsample_factor must be positive")
    if int(n_bins) % int(downsample_factor):
        raise ValueError("downsample_factor must divide n_bins")
    selected = int(selected_bin) % int(n_bins)
    groundtruth = int(gt_bin) % int(n_bins)
    if selected == groundtruth:
        return "correct"
    alias_bins = int(n_bins) // int(downsample_factor)
    if selected % alias_bins == groundtruth % alias_bins:
        return "wrong_group"
    return "wrong_alias_bin"


def _signed_alias_bin_delta(
    selected_bin: int,
    gt_bin: int,
    n_bins: int,
    downsample_factor: int,
) -> int:
    """Return the shortest signed offset between modulo-N/D alias bins."""

    if int(n_bins) <= 0 or int(downsample_factor) <= 0:
        raise ValueError("n_bins and downsample_factor must be positive")
    if int(n_bins) % int(downsample_factor):
        raise ValueError("downsample_factor must divide n_bins")
    alias_bins = int(n_bins) // int(downsample_factor)
    selected = int(selected_bin) % alias_bins
    groundtruth = int(gt_bin) % alias_bins
    delta = (selected - groundtruth) % alias_bins
    if delta > alias_bins // 2:
        delta -= alias_bins
    return int(delta)


def _two_sided_sign_p(fixes: int, breaks: int) -> float:
    discordant = int(fixes) + int(breaks)
    if discordant <= 0:
        return 1.0
    smaller = min(int(fixes), int(breaks))
    log_terms = [
        math.lgamma(discordant + 1.0)
        - math.lgamma(index + 1.0)
        - math.lgamma(discordant - index + 1.0)
        - discordant * math.log(2.0)
        for index in range(smaller + 1)
    ]
    largest = max(log_terms)
    tail = math.exp(largest) * math.fsum(
        math.exp(value - largest) for value in log_terms
    )
    return float(min(1.0, 2.0 * tail))


def _summaries(
    rows: Sequence[dict[str, Any]],
    methods: Sequence[MethodSpec],
    sf: int,
    os_factor: int,
    downsample_factor: int,
    *,
    by_seed: bool,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["snr_label"]),
            int(row["seed"]) if by_seed else None,
        )
        groups.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    full_samples = int(os_factor) * (1 << int(sf))
    for (snr_label, seed), subset in groups.items():
        for spec in methods:
            errors = sum(1 - int(row[f"{spec.name}_correct"]) for row in subset)
            wrong_group_errors = sum(
                str(row[f"{spec.name}_error_mode"]) == "wrong_group"
                for row in subset
            )
            wrong_alias_bin_errors = sum(
                str(row[f"{spec.name}_error_mode"]) == "wrong_alias_bin"
                for row in subset
            )
            if errors != wrong_group_errors + wrong_alias_bin_errors:
                raise AssertionError(
                    f"{spec.name} error-mode counts do not partition errors"
                )
            alias_bin_correct_decisions = (
                len(subset) - wrong_alias_bin_errors
            )
            far_alias_bin_errors = sum(
                str(row[f"{spec.name}_error_mode"])
                == "wrong_alias_bin"
                and abs(int(row[f"{spec.name}_alias_bin_delta"])) > 8
                for row in subset
            )
            fixes = sum(
                int(row["savaux_correct"]) == 0
                and int(row[f"{spec.name}_correct"]) == 1
                for row in subset
            )
            breaks = sum(
                int(row["savaux_correct"]) == 1
                and int(row[f"{spec.name}_correct"]) == 0
                for row in subset
            )
            sample_count = selected_sample_count(
                sf,
                os_factor,
                downsample_factor,
                branch_phases=spec.branch_phases,
                downsample_phases=spec.downsample_phases,
            )
            output.append(
                {
                    "snr_label": snr_label,
                    "added_snr_db": ""
                    if snr_label == "clean"
                    else float(snr_label),
                    "seed": "" if seed is None else int(seed),
                    "method": spec.name,
                    "decisions": int(len(subset)),
                    "errors": int(errors),
                    "ser": float(errors / max(1, len(subset))),
                    "wrong_group_errors": int(wrong_group_errors),
                    "wrong_alias_bin_errors": int(
                        wrong_alias_bin_errors
                    ),
                    "wrong_group_fraction_of_errors": float(
                        wrong_group_errors / max(1, errors)
                    ),
                    "wrong_alias_bin_fraction_of_errors": float(
                        wrong_alias_bin_errors / max(1, errors)
                    ),
                    "oracle_group_fixed_ser": float(
                        wrong_alias_bin_errors / max(1, len(subset))
                    ),
                    "maximum_group_fix_ser_gain": float(
                        wrong_group_errors / max(1, len(subset))
                    ),
                    "alias_bin_accuracy": float(
                        1.0
                        - wrong_alias_bin_errors
                        / max(1, len(subset))
                    ),
                    "group_error_rate_given_correct_alias_bin": float(
                        wrong_group_errors
                        / max(1, alias_bin_correct_decisions)
                    ),
                    "far_alias_bin_errors_abs_gt8": int(
                        far_alias_bin_errors
                    ),
                    "far_fraction_of_wrong_alias_bin_errors": float(
                        far_alias_bin_errors
                        / max(1, wrong_alias_bin_errors)
                    ),
                    "fixes_vs_savaux": int(fixes),
                    "breaks_vs_savaux": int(breaks),
                    "paired_sign_p": _two_sided_sign_p(fixes, breaks),
                    "mean_gt_margin_db": float(
                        np.mean(
                            [
                                float(row[f"{spec.name}_gt_margin_db"])
                                for row in subset
                            ]
                        )
                    ),
                    "physical_samples": int(sample_count),
                    "sample_fraction_vs_savaux": float(
                        sample_count / full_samples
                    ),
                    "branch_phases": "|".join(
                        str(value) for value in spec.branch_phases
                    ),
                    "downsample_phases": "|".join(
                        str(value) for value in spec.downsample_phases
                    ),
                    "uses_phase_fingerprint": int(spec.phase_fingerprint),
                }
            )
    output.sort(
        key=lambda row: (
            str(row["snr_label"]) != "clean",
            float(row["added_snr_db"])
            if row["added_snr_db"] != ""
            else 0.0,
            str(row["method"]),
            int(row["seed"]) if row["seed"] != "" else -1,
        )
    )
    return output


def _write_results(
    path: Path,
    summary: Sequence[dict[str, Any]],
    *,
    symbol_count: int,
    packet_count: int,
    reference_power: float,
    savaux_max_abs_error: float,
) -> None:
    summary_by_key = {
        (str(row["snr_label"]), str(row["method"])): row
        for row in summary
    }
    available_methods = {
        str(row["method"]) for row in summary
    }
    noisy_labels = sorted(
        {
            str(row["snr_label"])
            for row in summary
            if str(row["snr_label"]) != "clean"
        },
        key=float,
        reverse=True,
    )
    headline_candidates = (
        ("savaux", "Savaux"),
        ("savaux_phase", "Savaux + phase"),
        ("litenap_savaux_k2", "K2 (1/2 samples)"),
        ("litenap_savaux_k1", "K1 (1/4 samples)"),
    )
    headline_methods = tuple(
        (method, label)
        for method, label in headline_candidates
        if method in available_methods
    )
    savaux_is_best = all(
        float(summary_by_key[(label, "savaux")]["ser"])
        <= min(
            float(row["ser"])
            for row in summary
            if str(row["snr_label"]) == label
        )
        for label in noisy_labels
    )
    lines = [
        "# LiteNap-Savaux clean-GT added-noise comparison",
        "",
        (
            f"Source set: {packet_count} clean-synchronized packets, "
            f"{symbol_count} payload symbols."
        ),
        (
            "Complex AWGN uses the noisy_iq convention: I/Q variance is half "
            "the requested total added-noise power."
        ),
        (
            "Clean consensus FFT bins are scoring-only; synchronization, CFO, "
            "header-bin calibration, alias selection, steering, and phase-jump "
            "scores do not read payload GT."
        ),
        f"Payload reference power: `{reference_power:.9g}`.",
        (
            "Full-view component spectrum versus original Savaux maximum "
            f"absolute error: `{savaux_max_abs_error:.6g}`."
        ),
        "",
        "## Outcome",
        "",
        (
            "Original Savaux has the lowest SER at every added-noise condition."
            if savaux_is_best
            else "At least one proposed method improves on original Savaux."
        ),
        (
            "K1/K2 therefore provide sample-rate and computation trade-offs in "
            "this AWGN experiment, not a decoding gain over full-sample Savaux."
        ),
        (
            "The phase variants are a training-free phase-jump diagnostic, not "
            "a full reproduction of LiteNap's transmitter-specific, "
            "preamble-calibrated hardware fingerprint."
        ),
        "",
        "| added SNR | "
        + " | ".join(label for _method, label in headline_methods)
        + " |",
        "|---:|"
        + "|".join("---:" for _method, _label in headline_methods)
        + "|",
    ]
    for label in noisy_labels:
        cells = []
        for method, _display_name in headline_methods:
            row = summary_by_key[(label, method)]
            cells.append(f"{100.0 * float(row['ser']):.2f}%")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Full comparison",
            "",
        "| added SNR | method | samples | fraction | errors / decisions | SER | fixes / breaks | paired p | GT margin |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['snr_label']} | {row['method']} "
            f"| {row['physical_samples']} "
            f"| {float(row['sample_fraction_vs_savaux']):.3f} "
            f"| {row['errors']} / {row['decisions']} "
            f"| {float(row['ser']):.6f} "
            f"| {row['fixes_vs_savaux']} / {row['breaks_vs_savaux']} "
            f"| {float(row['paired_sign_p']):.4g} "
            f"| {float(row['mean_gt_margin_db']):+.3f} dB |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `savaux` uses all `R*N` samples.",
            "- `savaux_phase` uses the same samples and only adds LiteNap phase-jump reranking.",
            "- `litenap_single` uses one `N/D` observation and retains alias ambiguity.",
            "- `litenap_savaux_kK` uses all R oversampling phases for K downsampling views.",
            "- `_phase` variants add the training-free phase-jump timing statistic; no clean waveform template is used.",
            "- Every method shares the same explicit-header modulo-four residual-bin calibration.",
            "- Added SNR is signal-reference power divided by added AWGN power, not the final measured capture SNR.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_error_mode_results(
    path: Path,
    summary: Sequence[dict[str, Any]],
) -> None:
    selected_methods = {
        "litenap_savaux_k1",
        "litenap_savaux_k2",
        "litenap_savaux_k1_phase",
        "litenap_savaux_k2_phase",
    }
    rows = [
        row
        for row in summary
        if str(row["snr_label"]) != "clean"
        and str(row["method"]) in selected_methods
    ]
    rows.sort(
        key=lambda row: (
            -float(row["added_snr_db"]),
            str(row["method"]),
        )
    )
    lines = [
        "# LiteNap-Savaux error-mode decomposition",
        "",
        (
            "`wrong_group` means the aliased bin is correct modulo `N/D`, "
            "but the full-frequency alias group is wrong."
        ),
        (
            "`wrong_alias_bin` means even the modulo-`N/D` aliased bin is "
            "wrong. The two modes partition all hard-decision errors."
        ),
        "",
        "| added SNR | method | SER | oracle group floor | wrong group | wrong alias bin | group share | group error given alias | far alias share |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['snr_label']} | {row['method']} "
            f"| {float(row['ser']):.6f} "
            f"| {float(row['oracle_group_fixed_ser']):.6f} "
            f"| {row['wrong_group_errors']} "
            f"| {row['wrong_alias_bin_errors']} "
            f"| {float(row['wrong_group_fraction_of_errors']):.3f} "
            f"| {float(row['group_error_rate_given_correct_alias_bin']):.3f} "
            f"| {float(row['far_fraction_of_wrong_alias_bin_errors']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    if float(args.fingerprint_weight) < 0.0:
        raise ValueError("fingerprint-weight must be non-negative")

    input_iq = args.input_iq.resolve()
    symbol_path = args.symbols.resolve()
    groundtruth_path = args.groundtruth.resolve()
    groundtruth = _groundtruth_bins(groundtruth_path)
    packets, cases = _load_cases(
        symbol_path,
        groundtruth,
        int(args.max_packets),
        int(args.max_symbols),
    )
    samples = np.memmap(input_iq, dtype=np.complex64, mode="r")
    packet0 = cases[0][0]
    sf = int(packet0["sf"])
    os_factor = int(packet0["os_factor"])
    factor = int(args.downsample_factor)
    n_bins = 1 << sf
    symbol_samples = n_bins * os_factor
    origin_shift = (
        int(args.origin_shift)
        if args.origin_shift is not None
        else os_factor // 2
    )
    methods = _method_specs(
        sf,
        os_factor,
        factor,
        args.view_counts,
        float(args.fingerprint_weight),
    )
    packet_calibrations = {
        int(packet["packet_index"]): _packet_header_bin_calibration(
            samples,
            packet,
            origin_shift,
            float(args.min_header_bin_consensus),
        )
        for packet in packets
    }
    reference_power = _payload_reference_power(
        samples, cases, origin_shift
    )

    rows: list[dict[str, Any]] = []
    conditions: list[float | None] = [None] + [
        float(value) for value in args.snrs
    ]
    verify_remaining = max(0, int(args.verify_savaux_symbols))
    savaux_max_abs_error = 0.0

    for condition_index, snr_db in enumerate(conditions):
        seeds: Sequence[int] = (
            (0,) if snr_db is None else tuple(int(value) for value in args.seeds)
        )
        for seed in seeds:
            for batch_start in range(0, len(cases), int(args.batch_size)):
                batch_cases = cases[
                    batch_start : batch_start + int(args.batch_size)
                ]
                noisy_symbols: list[np.ndarray] = []
                dechirped_symbols: list[np.ndarray] = []
                for packet, symbol in batch_cases:
                    start = int(symbol["start_sample"]) + origin_shift
                    clean = np.asarray(
                        samples[start : start + symbol_samples],
                        dtype=np.complex64,
                    )
                    if clean.size != symbol_samples:
                        raise ValueError(
                            f"incomplete symbol at sample {start}"
                        )
                    noise_seed = _derived_noise_seed(
                        int(seed),
                        condition_index,
                        int(packet["packet_index"]),
                        int(symbol["payload_symbol_index"]),
                    )
                    noisy = noise_samples(
                        clean,
                        snr_db,
                        noise_seed,
                        reference_power,
                        noise_shape="white",
                        os_factor=os_factor,
                    )
                    relative_header_start = (
                        int(packet["header_start_sample"])
                        + origin_shift
                        - start
                    )
                    dechirped = prepare_dechirped_symbol(
                        noisy,
                        start_sample=0,
                        sf=sf,
                        os_factor=os_factor,
                        cfo_int=int(packet["cfo_int"]),
                        cfo_frac=float(packet["cfo_frac"]),
                        header_start_sample=relative_header_start,
                        cfo_correction_mode="continuous",
                    )
                    noisy_symbols.append(noisy)
                    dechirped_symbols.append(dechirped)

                dechirped_batch = np.stack(dechirped_symbols)
                components = subnyquist_component_spectra_batch(
                    dechirped_batch,
                    sf,
                    os_factor,
                    factor,
                    wrap_correction=True,
                )
                regular_spectra = {
                    spec.name: _combine_batch(components, spec)
                    for spec in methods
                    if not spec.phase_fingerprint
                }

                for local_index, (packet, symbol) in enumerate(batch_cases):
                    gt_bin = int(symbol["gt_bin"])
                    calibration = packet_calibrations[
                        int(packet["packet_index"])
                    ]
                    bin_correction = int(calibration.correction_bins)
                    uncorrected_gt_bin = (
                        gt_bin - bin_correction
                    ) % n_bins
                    row: dict[str, Any] = {
                        "snr_label": "clean"
                        if snr_db is None
                        else f"{float(snr_db):g}",
                        "added_snr_db": ""
                        if snr_db is None
                        else float(snr_db),
                        "noise_power_db_relative": ""
                        if snr_db is None
                        else -float(snr_db),
                        "seed": int(seed),
                        "packet_index": int(packet["packet_index"]),
                        "payload_symbol_index": int(
                            symbol["payload_symbol_index"]
                        ),
                        "frame_symbol_index": int(
                            symbol["frame_symbol_index"]
                        ),
                        "start_sample": int(symbol["start_sample"]),
                        "gt_bin": gt_bin,
                        "header_bin_correction": bin_correction,
                        "header_bin_consensus": float(
                            calibration.consensus
                        ),
                        "header_bin_observations": int(
                            calibration.observation_count
                        ),
                        "header_bin_residual": int(
                            calibration.residual_bins
                        ),
                    }

                    power_by_method: dict[str, np.ndarray] = {}
                    for spec in methods:
                        if spec.phase_fingerprint:
                            continue
                        power = _spectrum_power(
                            regular_spectra[spec.name][local_index]
                        )
                        power_by_method[spec.name] = power
                        scored = _score_power(
                            power, uncorrected_gt_bin
                        )
                        scored["bin"] = (
                            int(scored["bin"]) + bin_correction
                        ) % n_bins
                        for key, value in scored.items():
                            row[f"{spec.name}_{key}"] = value

                    for spec in methods:
                        if not spec.phase_fingerprint:
                            continue
                        base_name = spec.name.removesuffix("_phase")
                        base_power = power_by_method[base_name]
                        reranked = rerank_with_phase_fingerprint(
                            base_power,
                            dechirped_symbols[local_index],
                            sf,
                            os_factor,
                            factor,
                            branch_phases=spec.branch_phases,
                            downsample_phases=spec.downsample_phases,
                            fingerprint_weight=float(
                                args.fingerprint_weight
                            ),
                            min_segment_samples=int(
                                args.fingerprint_min_segment
                            ),
                            wrap_correction=True,
                        )
                        selected = (
                            int(reranked.raw_fft_bin) + bin_correction
                        ) % n_bins
                        row[f"{spec.name}_bin"] = selected
                        row[f"{spec.name}_correct"] = int(
                            selected == gt_bin
                        )
                        row[f"{spec.name}_gt_rank"] = int(
                            row[f"{base_name}_gt_rank"]
                        )
                        row[f"{spec.name}_gt_margin_db"] = float(
                            row[f"{base_name}_gt_margin_db"]
                        )
                        row[f"{spec.name}_alias_bin"] = int(
                            (
                                int(reranked.alias_bin)
                                + bin_correction
                            )
                            % n_bins
                        )
                        row[f"{spec.name}_candidate_bins"] = "|".join(
                            str((int(value) + bin_correction) % n_bins)
                            for value in reranked.candidate_bins
                        )
                        row[f"{spec.name}_phase_jump_scores"] = "|".join(
                            f"{value:.9g}"
                            for value in reranked.phase_jump_scores
                        )
                        row[f"{spec.name}_spectral_log_scores"] = "|".join(
                            f"{value:.9g}"
                            for value in reranked.spectral_log_scores
                        )
                        row[f"{spec.name}_combined_scores"] = "|".join(
                            f"{value:.9g}"
                            for value in reranked.combined_scores
                        )

                    for spec in methods:
                        mode = _alias_error_mode(
                            int(row[f"{spec.name}_bin"]),
                            gt_bin,
                            n_bins,
                            factor,
                        )
                        row[f"{spec.name}_error_mode"] = mode
                        row[f"{spec.name}_alias_bin_correct"] = int(
                            mode != "wrong_alias_bin"
                        )
                        row[f"{spec.name}_alias_bin_delta"] = (
                            _signed_alias_bin_delta(
                                int(row[f"{spec.name}_bin"]),
                                gt_bin,
                                n_bins,
                                factor,
                            )
                        )

                    if verify_remaining > 0:
                        relative_header_start = (
                            int(packet["header_start_sample"])
                            + origin_shift
                            - (
                                int(symbol["start_sample"])
                                + origin_shift
                            )
                        )
                        expected, _branches, _phase = (
                            paper_oversampled_spectrum(
                                noisy_symbols[local_index],
                                start_sample=0,
                                sf=sf,
                                os_factor=os_factor,
                                cfo_int=int(packet["cfo_int"]),
                                cfo_frac=float(packet["cfo_frac"]),
                                header_start_sample=relative_header_start,
                                cfo_correction_mode="continuous",
                            )
                        )
                        actual = regular_spectra["savaux"][
                            local_index
                        ]
                        max_error = float(
                            np.max(np.abs(actual - expected))
                        )
                        savaux_max_abs_error = max(
                            savaux_max_abs_error, max_error
                        )
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            rtol=2e-4,
                            atol=2e-4,
                        )
                        verify_remaining -= 1
                    rows.append(row)

            label = "clean" if snr_db is None else f"{float(snr_db):g}"
            print(
                f"snr={label} seed={seed} symbols={len(cases)}",
                flush=True,
            )

    summary = _summaries(
        rows,
        methods,
        sf,
        os_factor,
        factor,
        by_seed=False,
    )
    summary_by_seed = _summaries(
        rows,
        methods,
        sf,
        os_factor,
        factor,
        by_seed=True,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "symbols.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "summary_by_seed.csv", summary_by_seed)
    _write_results(
        output_dir / "RESULTS.md",
        summary,
        symbol_count=len(cases),
        packet_count=len(
            {int(packet["packet_index"]) for packet, _symbol in cases}
        ),
        reference_power=reference_power,
        savaux_max_abs_error=savaux_max_abs_error,
    )
    _write_error_mode_results(
        output_dir / "ERROR_MODES.md",
        summary,
    )
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "input_iq": str(input_iq),
        "symbols": str(symbol_path),
        "groundtruth": str(groundtruth_path),
        "output_dir": str(output_dir),
        "sf": sf,
        "os_factor": os_factor,
        "origin_shift": origin_shift,
        "symbol_count": len(cases),
        "signal_reference_power": reference_power,
        "savaux_max_abs_error": savaux_max_abs_error,
        "header_bin_calibrations": {
            str(packet_index): {
                "correction_bins": int(calibration.correction_bins),
                "consensus": float(calibration.consensus),
                "observation_count": int(
                    calibration.observation_count
                ),
                "residual_bins": int(calibration.residual_bins),
            }
            for packet_index, calibration in packet_calibrations.items()
        },
        "methods": [
            {
                "name": spec.name,
                "branch_phases": list(spec.branch_phases),
                "downsample_phases": list(
                    spec.downsample_phases
                ),
                "phase_fingerprint": spec.phase_fingerprint,
            }
            for spec in methods
        ],
        "gt_usage": "scoring only",
        "noise_protocol": (
            "payload-copy complex AWGN; I/Q variance = total added power / 2; "
            "noise_power_db_relative = -added_snr_db"
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} trials to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
