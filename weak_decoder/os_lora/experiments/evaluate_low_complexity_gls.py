#!/usr/bin/env python3
"""评估协方差选出的 pattern bank 与无矩阵 cross-fit GLS。"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
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
from weak_decoder.os_lora.experiment_support.conditional_gls import (  # noqa: E402
    build_selected_pattern_banks as _build_selected_banks,
    estimate_training_covariance as _training_covariance,
    evaluate_conditional_condition as _evaluate_conditional_condition,
)
from weak_decoder.os_lora.experiment_support.pattern_training import (  # noqa: E402
    bootstrap_offpacket_noise as _offpacket_bootstrap_samples,
)
from weak_decoder.os_lora.system.noise import select_background_bins as _background_bins  # noqa: E402
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    NonuniformPatternBank,
    build_pattern_bank,
    conditional_lora_gls_detect,
    crossfit_weighted_spectrum,
    lora_branch_color_mismatch,
    lora_interbin_leakage_covariance,
    lora_phase_law_consistency,
    lora_wrap_consistency_power,
    matrix_free_crossfit_gls_spectrum_power,
    pattern_bank_split_spectra,
    pattern_covariance_color_mismatch,
    prepare_dechirped_symbol,
    select_pattern_subset,
)


def _filter_candidate_bank(
    bank: NonuniformPatternBank,
    widths: Sequence[int] | None,
    structures: Sequence[str] | None,
) -> NonuniformPatternBank:
    """按名称筛选多尺度候选，保证设计消融实验可以复现。"""

    allowed_widths = None if widths is None else {int(value) for value in widths}
    allowed_structures = None if structures is None else {str(value) for value in structures}
    if allowed_widths is None and allowed_structures is None:
        return bank
    selected: list[int] = []
    for index, name in enumerate(bank.names):
        match = re.fullmatch(r"(block|multiscale)_w(\d+)_s\d+_b\d+", str(name))
        if match is None:
            continue
        structure = str(match.group(1))
        width = int(match.group(2))
        if allowed_widths is not None and width not in allowed_widths:
            continue
        if allowed_structures is not None and structure not in allowed_structures:
            continue
        selected.append(index)
    if not selected:
        raise ValueError("candidate design filter removed every pattern")
    return NonuniformPatternBank(
        names=tuple(bank.names[index] for index in selected),
        offsets=tuple(bank.offsets[index] for index in selected),
        os_factor=int(bank.os_factor),
        sf=int(bank.sf),
        kind=f"{bank.kind}_filtered",
    )


def _test_samples(
    clean: np.ndarray,
    packets: Sequence[dict[str, Any]],
    reference_power: float,
    snr_db: float,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if str(args.noise_mode) == "offpacket_bootstrap":
        return _offpacket_bootstrap_samples(
            clean=clean,
            packets=packets,
            snr_db=float(snr_db),
            seed=int(seed),
            reference_power=float(reference_power),
            max_source_windows=int(args.bootstrap_source_windows),
            guard_symbols=float(args.guard_symbols),
        )
    return noise_samples(clean, float(snr_db), int(seed), float(reference_power))


def _evaluate_condition(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    banks: dict[str, NonuniformPatternBank],
    lora_priors: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_banks: dict[str, str] = {name: name for name in banks}
    if float(args.wrap_consistency_exponent) > 0.0:
        for name in banks:
            if name.startswith("information_"):
                method_banks[f"{name}_wrap"] = name
                if float(args.savaux_gate_margin_db) >= 0.0:
                    method_banks[f"{name}_wrap_gated"] = name
                    if args.branch_color_threshold is not None:
                        method_banks[f"{name}_wrap_color_gated"] = name
                        method_banks[f"{name}_conditional_lora"] = name
                    if args.lora_guard_margin_db is not None:
                        method_banks[f"{name}_wrap_lora_guarded"] = name
    methods = ("savaux", *method_banks.keys())
    selected: dict[str, list[int]] = {method: [] for method in methods}
    gt_bins: list[int] = []
    elapsed: dict[str, float] = {method: 0.0 for method in methods}
    triggered: dict[str, int] = {method: 0 for method in methods}
    screened: dict[str, int] = {method: 0 for method in methods}
    accepted: dict[str, int] = {method: 0 for method in methods}
    rejected: dict[str, int] = {method: 0 for method in methods}
    diagnostics: list[dict[str, Any]] = []
    symbol_index = 0
    for packet_index, packet in enumerate(packets):
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        origin_shift = os_factor // 2
        header_start = int(packet["header_start_sample"]) + origin_shift
        for symbol in packet["payload_symbols"]:
            start = int(symbol["start_sample"]) + origin_shift
            gt_bins.append(int(symbol["gt_bin"]))
            begin = time.perf_counter()
            savaux_spectrum, savaux_branches, _phase = paper_oversampled_spectrum(
                samples=samples,
                start_sample=start,
                sf=sf,
                os_factor=os_factor,
                cfo_int=int(packet["cfo_int"]),
                cfo_frac=float(packet["cfo_frac"]),
                header_start_sample=header_start,
                cfo_correction_mode="continuous",
            )
            savaux_power = np.abs(savaux_spectrum).astype(np.float64) ** 2
            elapsed["savaux"] += time.perf_counter() - begin
            savaux_bin = int(np.argmax(savaux_power))
            selected["savaux"].append(savaux_bin)
            if savaux_power.size >= 2:
                top_two = np.partition(savaux_power, -2)[-2:]
                savaux_margin_db = float(10.0 * np.log10((float(top_two[-1]) + 1e-30) / (float(top_two[-2]) + 1e-30)))
            else:
                savaux_margin_db = float("inf")
            background = _background_bins(
                savaux_power,
                exclude_top=int(args.exclude_top),
                guard_bins=int(args.exclude_guard_bins),
            )
            color_begin = time.perf_counter()
            branch_color_mismatch = lora_branch_color_mismatch(savaux_branches, background)
            branch_color_elapsed = time.perf_counter() - color_begin
            dechirped = prepare_dechirped_symbol(
                samples=samples,
                start_sample=start,
                sf=sf,
                os_factor=os_factor,
                cfo_int=int(packet["cfo_int"]),
                cfo_frac=float(packet["cfo_frac"]),
                header_start_sample=header_start,
                cfo_correction_mode="continuous",
            )
            for name, bank in banks.items():
                conditional_name = f"{name}_conditional_lora"
                if conditional_name in selected:
                    conditional_begin = time.perf_counter()
                    conditional_result = conditional_lora_gls_detect(
                        dechirped=dechirped,
                        savaux_spectrum=savaux_spectrum,
                        savaux_branch_spectra=savaux_branches,
                        bank=bank,
                        covariance_bins=background,
                        savaux_margin_db=float(args.savaux_gate_margin_db),
                        branch_color_threshold=float(args.branch_color_threshold),
                        diagonal_loading=float(args.gls_loading),
                        folds=int(args.crossfit_folds),
                        max_iterations=int(args.cg_iterations),
                        tolerance=float(args.cg_tolerance),
                        wrap_consistency_exponent=float(args.wrap_consistency_exponent),
                        wrap_minimum_segment=int(args.wrap_minimum_segment),
                    )
                    elapsed[conditional_name] += time.perf_counter() - conditional_begin
                    selected[conditional_name].append(int(conditional_result.raw_fft_bin))
                    screened[conditional_name] += int(conditional_result.screened)
                    triggered[conditional_name] += int(conditional_result.backend_ran)
                begin = time.perf_counter()
                spectra, head_spectra, tail_spectra = pattern_bank_split_spectra(dechirped, bank)
                spectra_elapsed = time.perf_counter() - begin
                gls_begin = time.perf_counter()
                gls_result = matrix_free_crossfit_gls_spectrum_power(
                    spectra,
                    covariance_bins=background,
                    diagonal_loading=float(args.gls_loading),
                    folds=int(args.crossfit_folds),
                    max_iterations=int(args.cg_iterations),
                    tolerance=float(args.cg_tolerance),
                    prior_covariance=lora_priors.get(name),
                    prior_weight=(
                        float(args.runtime_lora_weight) if name.startswith("lora_information_") else 0.0
                    ),
                )
                score = gls_result.power
                gls_elapsed = time.perf_counter() - gls_begin
                method_elapsed = spectra_elapsed + gls_elapsed
                elapsed[name] += method_elapsed
                selected[name].append(int(np.argmax(score)))
                wrap_name = f"{name}_wrap"
                if wrap_name in selected:
                    wrap_begin = time.perf_counter()
                    weighted_head = crossfit_weighted_spectrum(head_spectra, gls_result.inverse_targets)
                    weighted_tail = crossfit_weighted_spectrum(tail_spectra, gls_result.inverse_targets)
                    wrap_score = lora_wrap_consistency_power(
                        weighted_head,
                        weighted_tail,
                        base_power=score,
                        exponent=float(args.wrap_consistency_exponent),
                        minimum_segment=int(args.wrap_minimum_segment),
                    )
                    wrap_elapsed = time.perf_counter() - wrap_begin
                    elapsed[wrap_name] += method_elapsed + wrap_elapsed
                    wrap_bin = int(np.argmax(wrap_score))
                    selected[wrap_name].append(wrap_bin)
                    gated_name = f"{name}_wrap_gated"
                    if gated_name in selected:
                        use_fallback = savaux_margin_db <= float(args.savaux_gate_margin_db)
                        selected[gated_name].append(wrap_bin if use_fallback else savaux_bin)
                        if use_fallback:
                            triggered[gated_name] += 1
                            elapsed[gated_name] += method_elapsed + wrap_elapsed
                    color_gated_name = f"{name}_wrap_color_gated"
                    if color_gated_name in selected:
                        ambiguous = savaux_margin_db <= float(args.savaux_gate_margin_db)
                        run_backend = (
                            ambiguous
                            and branch_color_mismatch >= float(args.branch_color_threshold)
                        )
                        selected[color_gated_name].append(wrap_bin if run_backend else savaux_bin)
                        if ambiguous:
                            screened[color_gated_name] += 1
                            elapsed[color_gated_name] += branch_color_elapsed
                        if run_backend:
                            triggered[color_gated_name] += 1
                            elapsed[color_gated_name] += method_elapsed + wrap_elapsed
                        if ambiguous and wrap_bin != savaux_bin:
                            if run_backend:
                                accepted[color_gated_name] += 1
                            else:
                                rejected[color_gated_name] += 1
                        if (
                            conditional_name in selected
                            and selected[conditional_name][-1] != selected[color_gated_name][-1]
                        ):
                            raise RuntimeError("conditional detector disagrees with color-gated ablation")
                    guarded_name = f"{name}_wrap_lora_guarded"
                    if guarded_name in selected:
                        use_fallback = savaux_margin_db <= float(args.savaux_gate_margin_db)
                        guarded_bin = savaux_bin
                        if use_fallback:
                            consistency = lora_phase_law_consistency(
                                dechirped,
                                (savaux_bin, wrap_bin),
                                sf=sf,
                                os_factor=os_factor,
                                segment_count=int(args.lora_guard_segments),
                            )
                            consistency_margin_db = float(
                                10.0
                                * np.log10(
                                    (float(consistency[1]) + 1e-30)
                                    / (float(consistency[0]) + 1e-30)
                                )
                            )
                            accept_proposal = (
                                wrap_bin == savaux_bin
                                or consistency_margin_db >= float(args.lora_guard_margin_db)
                            )
                            guarded_bin = wrap_bin if accept_proposal else savaux_bin
                            triggered[guarded_name] += 1
                            if wrap_bin != savaux_bin:
                                diagnostics.append(
                                    {
                                        "method": str(guarded_name),
                                        "packet_index": int(packet_index),
                                        "symbol_index": int(symbol_index),
                                        "gt_bin": int(gt_bins[-1]),
                                        "savaux_bin": int(savaux_bin),
                                        "proposal_bin": int(wrap_bin),
                                        "savaux_correct": int(savaux_bin == int(gt_bins[-1])),
                                        "proposal_correct": int(wrap_bin == int(gt_bins[-1])),
                                        "savaux_margin_db": float(savaux_margin_db),
                                        "savaux_consistency": float(consistency[0]),
                                        "proposal_consistency": float(consistency[1]),
                                        "consistency_margin_db": float(consistency_margin_db),
                                        "color_mismatch": pattern_covariance_color_mismatch(
                                            spectra,
                                            background,
                                            bank,
                                        ),
                                        "branch_color_mismatch": float(branch_color_mismatch),
                                        "accepted": int(accept_proposal),
                                    }
                                )
                                if accept_proposal:
                                    accepted[guarded_name] += 1
                                else:
                                    rejected[guarded_name] += 1
                            elapsed[guarded_name] += method_elapsed + time.perf_counter() - wrap_begin
                        selected[guarded_name].append(guarded_bin)
            symbol_index += 1

    symbol_count = len(gt_bins)
    savaux_selected = selected["savaux"]
    rows: list[dict[str, Any]] = []
    for method in methods:
        errors = sum(int(value) != int(gt) for value, gt in zip(selected[method], gt_bins, strict=True))
        fixes = sum(
            int(savaux) != int(gt) and int(value) == int(gt)
            for value, savaux, gt in zip(selected[method], savaux_selected, gt_bins, strict=True)
        )
        breaks = sum(
            int(savaux) == int(gt) and int(value) != int(gt)
            for value, savaux, gt in zip(selected[method], savaux_selected, gt_bins, strict=True)
        )
        rows.append(
            {
                "method": str(method),
                "pattern_count": 0 if method == "savaux" else len(banks[method_banks[method]].names),
                "symbol_count": int(symbol_count),
                "errors": int(errors),
                "ser": float(errors / max(1, symbol_count)),
                "fixes_vs_savaux": int(fixes),
                "breaks_vs_savaux": int(breaks),
                "elapsed_seconds": float(elapsed[method]),
                "mean_milliseconds_per_symbol": float(1000.0 * elapsed[method] / max(1, symbol_count)),
                "triggered_symbols": int(triggered[method]),
                "trigger_rate": float(triggered[method] / max(1, symbol_count)),
                "screened_symbols": int(screened[method]),
                "screen_rate": float(screened[method] / max(1, symbol_count)),
                "accepted_proposals": int(accepted[method]),
                "rejected_proposals": int(rejected[method]),
            }
        )
    return rows, diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--training-dataset", default=None)
    parser.add_argument("--snrs", nargs="+", type=float, default=[-23.0, -24.0, -25.0])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-packets", type=int, default=2)
    parser.add_argument("--candidate-kind", default="canonical")
    parser.add_argument("--candidate-widths", nargs="+", type=int, default=None)
    parser.add_argument(
        "--candidate-structures",
        nargs="+",
        choices=("block", "multiscale"),
        default=None,
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=0,
        help="Deterministically pre-compress the offline candidate bank; 0 keeps all candidates.",
    )
    parser.add_argument("--pattern-counts", nargs="+", type=int, default=[8])
    parser.add_argument("--training-seed", type=int, default=41)
    parser.add_argument("--training-snr", type=float, default=-24.0)
    parser.add_argument("--training-windows", type=int, default=64)
    parser.add_argument("--training-bins", type=int, default=8)
    parser.add_argument("--noise-mode", choices=("awgn", "offpacket_bootstrap"), default="offpacket_bootstrap")
    parser.add_argument("--bootstrap-source-windows", type=int, default=256)
    parser.add_argument("--guard-symbols", type=float, default=2.0)
    parser.add_argument("--exclude-top", type=int, default=8)
    parser.add_argument("--exclude-guard-bins", type=int, default=1)
    parser.add_argument("--gls-loading", type=float, default=0.05)
    parser.add_argument("--lora-leakage-weight", type=float, default=0.0)
    parser.add_argument("--runtime-lora-weight", type=float, default=0.0)
    parser.add_argument("--wrap-consistency-exponent", type=float, default=0.0)
    parser.add_argument("--wrap-minimum-segment", type=int, default=16)
    parser.add_argument("--savaux-gate-margin-db", type=float, default=-1.0)
    parser.add_argument("--branch-color-threshold", type=float, default=None)
    parser.add_argument("--lora-guard-margin-db", type=float, default=None)
    parser.add_argument("--lora-guard-segments", type=int, default=8)
    parser.add_argument("--crossfit-folds", type=int, default=4)
    parser.add_argument("--cg-iterations", type=int, default=8)
    parser.add_argument("--cg-tolerance", type=float, default=0.0)
    parser.add_argument("--conditional-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "low_complexity_gls",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    iq_path, symbol_path = dataset_paths(str(args.dataset))
    clean = np.fromfile(iq_path, dtype=np.complex64)
    all_packets = load_packets(symbol_path)
    packets = all_packets[: int(args.max_packets)] if int(args.max_packets) > 0 else all_packets
    if not packets:
        raise RuntimeError(f"no payload packets found for {args.dataset}")
    reference_power, _samples, _packets = signal_reference_power(clean, packets, "packet", None)
    training_dataset = str(args.training_dataset or args.dataset)
    if training_dataset == str(args.dataset):
        training_clean = clean
        training_all_packets = all_packets
        training_reference_power = reference_power
    else:
        training_iq_path, training_symbol_path = dataset_paths(training_dataset)
        training_clean = np.fromfile(training_iq_path, dtype=np.complex64)
        training_all_packets = load_packets(training_symbol_path)
        training_reference_packets = (
            training_all_packets[: int(args.max_packets)]
            if int(args.max_packets) > 0
            else training_all_packets
        )
        if not training_reference_packets:
            raise RuntimeError(f"no payload packets found for {training_dataset}")
        training_reference_power, _samples, _packets = signal_reference_power(
            training_clean,
            training_reference_packets,
            "packet",
            None,
        )
        if (
            int(training_reference_packets[0]["sf"]) != int(packets[0]["sf"])
            or int(training_reference_packets[0]["os_factor"]) != int(packets[0]["os_factor"])
        ):
            raise ValueError("training and test datasets must use the same SF and OSR")
    full_candidate_bank = build_pattern_bank(
        int(packets[0]["sf"]),
        int(packets[0]["os_factor"]),
        kind=str(args.candidate_kind),
    )
    filtered_candidate_bank = _filter_candidate_bank(
        full_candidate_bank,
        args.candidate_widths,
        args.candidate_structures,
    )
    candidate_bank = (
        select_pattern_subset(filtered_candidate_bank, int(args.candidate_limit), "diverse")
        if int(args.candidate_limit) > 0
        else filtered_candidate_bank
    )
    positive_pattern_counts = [int(value) for value in args.pattern_counts if int(value) > 0]
    fixed_bank_selection = (
        bool(positive_pattern_counts)
        and min(positive_pattern_counts) >= len(candidate_bank.names)
        and float(args.lora_leakage_weight) == 0.0
    )
    if fixed_bank_selection:
        covariance = np.eye(len(candidate_bank.names), dtype=np.complex128)
        snapshot_count = 0
        training_covariance_seconds = 0.0
    else:
        training_begin = time.perf_counter()
        covariance, snapshot_count = _training_covariance(
            training_clean,
            training_all_packets,
            training_reference_power,
            candidate_bank,
            args,
        )
        training_covariance_seconds = time.perf_counter() - training_begin
    selection_begin = time.perf_counter()
    banks, selection_rows = _build_selected_banks(
        candidate_bank,
        covariance,
        args.pattern_counts,
        float(args.gls_loading),
        float(args.lora_leakage_weight),
    )
    pattern_selection_seconds = time.perf_counter() - selection_begin
    lora_priors = {
        name: lora_interbin_leakage_covariance(bank)
        for name, bank in banks.items()
        if name.startswith("lora_information_")
    }
    for selection_row in selection_rows:
        selection_row["training_dataset"] = training_dataset
        selection_row["test_dataset"] = str(args.dataset)
        selection_row["full_candidate_count"] = len(full_candidate_bank.names)
        selection_row["filtered_candidate_count"] = len(filtered_candidate_bank.names)
        selection_row["candidate_count"] = len(candidate_bank.names)
        selection_row["training_covariance_seconds"] = float(training_covariance_seconds)
        selection_row["pattern_selection_seconds"] = float(pattern_selection_seconds)
        selection_row["selection_source"] = "fixed_bank" if fixed_bank_selection else "offpacket_covariance"
    summary_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for seed in args.test_seeds:
        for snr_db in args.snrs:
            samples = _test_samples(clean, all_packets, reference_power, float(snr_db), int(seed), args)
            if bool(args.conditional_only):
                rows, diagnostics = _evaluate_conditional_condition(samples, packets, banks, args)
            else:
                rows, diagnostics = _evaluate_condition(samples, packets, banks, lora_priors, args)
            for row in rows:
                row.update(
                    {
                        "dataset": str(args.dataset),
                        "training_dataset": training_dataset,
                        "snr_db": float(snr_db),
                        "seed": int(seed),
                        "training_seed": int(args.training_seed),
                        "training_snr": float(args.training_snr),
                        "training_snapshots": int(snapshot_count),
                        "full_candidate_count": len(full_candidate_bank.names),
                        "filtered_candidate_count": len(filtered_candidate_bank.names),
                        "candidate_count": len(candidate_bank.names),
                        "training_covariance_seconds": float(training_covariance_seconds),
                        "pattern_selection_seconds": float(pattern_selection_seconds),
                        "selection_source": "fixed_bank" if fixed_bank_selection else "offpacket_covariance",
                        "cg_iterations": int(args.cg_iterations),
                        "lora_leakage_weight": float(args.lora_leakage_weight),
                        "runtime_lora_weight": float(args.runtime_lora_weight),
                        "wrap_consistency_exponent": float(args.wrap_consistency_exponent),
                        "savaux_gate_margin_db": float(args.savaux_gate_margin_db),
                        "branch_color_threshold": (
                            "" if args.branch_color_threshold is None else float(args.branch_color_threshold)
                        ),
                        "lora_guard_margin_db": (
                            "" if args.lora_guard_margin_db is None else float(args.lora_guard_margin_db)
                        ),
                        "lora_guard_segments": int(args.lora_guard_segments),
                    }
                )
                summary_rows.append(row)
            for diagnostic in diagnostics:
                diagnostic.update(
                    {
                        "dataset": str(args.dataset),
                        "training_dataset": training_dataset,
                        "noise_mode": str(args.noise_mode),
                        "snr_db": float(snr_db),
                        "seed": int(seed),
                        "lora_guard_segments": int(args.lora_guard_segments),
                    }
                )
                diagnostic_rows.append(diagnostic)
            compact = " ".join(f"{row['method']}={row['ser']:.4f}" for row in rows)
            print(f"{args.dataset} snr={snr_db} seed={seed}: {compact}", flush=True)
            write_csv(Path(args.output_dir).resolve() / "summary.csv", summary_rows)
            write_csv(Path(args.output_dir).resolve() / "proposals.csv", diagnostic_rows)

    output_dir = Path(args.output_dir).resolve()
    write_csv(output_dir / "selection.csv", selection_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "proposals.csv", diagnostic_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
