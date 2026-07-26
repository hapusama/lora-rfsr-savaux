"""多个离线实验共享的条件 GLS 训练与评估流程。"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from ...baselines.savaux_oversampled.paper_oversampled_demod import (
    _oversampled_downchirp,
    paper_oversampled_spectrum,
)
from ..system.noise import select_background_bins
from ..system.nonuniform_sampling import (
    NonuniformPatternBank,
    conditional_lora_gls_detect,
    lora_interbin_leakage_covariance,
    pattern_bin_values,
    prepare_dechirped_symbol,
    select_pattern_subset,
    select_pattern_subset_by_information,
)
from .noise_windows import active_intervals, empirical_covariance, off_packet_starts
from .pattern_training import bootstrap_offpacket_noise


def estimate_training_covariance(
    clean: np.ndarray,
    packets: Sequence[dict[str, Any]],
    reference_power: float,
    candidate_bank: NonuniformPatternBank,
    args: Any,
) -> tuple[np.ndarray, int]:
    """按照实验参数构造 pattern 选择所需的训练协方差。"""

    sf = int(candidate_bank.sf)
    os_factor = int(candidate_bank.os_factor)
    n_bins = 1 << sf
    window_len = n_bins * os_factor
    samples = bootstrap_offpacket_noise(
        clean=clean,
        packets=packets,
        snr_db=float(args.training_snr),
        seed=int(args.training_seed),
        reference_power=float(reference_power),
        max_source_windows=int(args.bootstrap_source_windows),
        guard_symbols=float(args.guard_symbols),
    )
    intervals = active_intervals(
        packets,
        int(round(float(args.guard_symbols) * window_len)),
    )
    starts = off_packet_starts(
        samples.size,
        window_len,
        intervals,
        int(args.training_windows),
        int(args.training_seed) + 7919,
    )
    if not starts:
        raise RuntimeError("no off-packet windows available for pattern selection")
    bin_count = min(max(1, int(args.training_bins)), n_bins)
    bins = tuple(
        int(value)
        for value in np.linspace(0, n_bins, num=bin_count, endpoint=False, dtype=np.int64)
    )
    downchirp = _oversampled_downchirp(sf, os_factor, 0, 0.0)
    vectors: list[np.ndarray] = []
    for start in starts:
        dechirped = np.asarray(
            samples[start: start + window_len] * downchirp,
            dtype=np.complex64,
        )
        for raw_bin in bins:
            vectors.append(pattern_bin_values(dechirped, raw_bin, candidate_bank))
    return empirical_covariance(np.asarray(vectors, dtype=np.complex128)), len(vectors)


def build_selected_pattern_banks(
    candidate_bank: NonuniformPatternBank,
    covariance: np.ndarray,
    pattern_counts: Sequence[int],
    loading: float,
    lora_leakage_weight: float,
) -> tuple[dict[str, NonuniformPatternBank], list[dict[str, Any]]]:
    """构造 Hamming 与协方差信息增益两类 pattern 子集。"""

    banks: dict[str, NonuniformPatternBank] = {}
    selection_rows: list[dict[str, Any]] = []
    design_covariances: list[tuple[str, np.ndarray]] = [("information", covariance)]
    if float(lora_leakage_weight) > 0.0:
        leakage = lora_interbin_leakage_covariance(candidate_bank)
        noise_power = float(np.real(np.trace(covariance)) / max(1, covariance.shape[0]))
        leakage_power = float(np.real(np.trace(leakage)) / max(1, leakage.shape[0]))
        scaled_leakage = leakage * noise_power / max(leakage_power, 1e-30)
        design_covariances.append(
            ("lora_information", covariance + float(lora_leakage_weight) * scaled_leakage)
        )
    for count in sorted({int(value) for value in pattern_counts if int(value) > 0}):
        diverse = select_pattern_subset(candidate_bank, count, "diverse")
        banks[f"hamming_{count}"] = diverse
        for selection_name, design_covariance in design_covariances:
            information = select_pattern_subset_by_information(
                candidate_bank,
                design_covariance,
                max_patterns=count,
                diagonal_loading=float(loading),
            )
            banks[f"{selection_name}_{count}"] = information.bank
            for rank, (index, gain) in enumerate(
                zip(information.indices, information.marginal_information, strict=True),
                start=1,
            ):
                selection_rows.append(
                    {
                        "selection_method": str(selection_name),
                        "pattern_count": int(count),
                        "selection_rank": int(rank),
                        "candidate_index": int(index),
                        "pattern_name": str(candidate_bank.names[index]),
                        "marginal_information": float(gain),
                        "lora_leakage_weight": float(lora_leakage_weight),
                    }
                )
    return banks, selection_rows


def evaluate_conditional_condition(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    banks: dict[str, NonuniformPatternBank],
    args: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """只评估可部署检测器，确保每次提前退出都由物理判据触发。"""

    if args.branch_color_threshold is None:
        raise ValueError("--conditional-only requires --branch-color-threshold")
    detector_banks = {
        name: bank for name, bank in banks.items() if name.startswith("information_")
    }
    methods = ("savaux", *(f"{name}_conditional_lora" for name in detector_banks))
    selected: dict[str, list[int]] = {method: [] for method in methods}
    elapsed: dict[str, float] = {method: 0.0 for method in methods}
    screened: dict[str, int] = {method: 0 for method in methods}
    triggered: dict[str, int] = {method: 0 for method in methods}
    gt_bins: list[int] = []
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
            background = select_background_bins(
                savaux_power,
                exclude_top=int(args.exclude_top),
                guard_bins=int(args.exclude_guard_bins),
            )
            for name, bank in detector_banks.items():
                method = f"{name}_conditional_lora"

                def lazy_dechirp() -> np.ndarray:
                    """仅在条件检测器实际触发后生成 dechirped symbol。"""

                    return prepare_dechirped_symbol(
                        samples=samples,
                        start_sample=start,
                        sf=sf,
                        os_factor=os_factor,
                        cfo_int=int(packet["cfo_int"]),
                        cfo_frac=float(packet["cfo_frac"]),
                        header_start_sample=header_start,
                        cfo_correction_mode="continuous",
                    )

                begin = time.perf_counter()
                result = conditional_lora_gls_detect(
                    dechirped=lazy_dechirp,
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
                elapsed[method] += time.perf_counter() - begin
                selected[method].append(int(result.raw_fft_bin))
                screened[method] += int(result.screened)
                triggered[method] += int(result.backend_ran)
                if result.backend_ran:
                    gt_bin = int(gt_bins[-1])
                    proposal_bin = int(result.raw_fft_bin)
                    gt_rank = 1 + int(np.count_nonzero(savaux_power > savaux_power[gt_bin]))
                    proposal_rank = 1 + int(
                        np.count_nonzero(savaux_power > savaux_power[proposal_bin])
                    )
                    diagnostics.append(
                        {
                            "method": str(method),
                            "packet_index": int(packet_index),
                            "symbol_index": int(symbol_index),
                            "gt_bin": gt_bin,
                            "savaux_bin": savaux_bin,
                            "proposal_bin": proposal_bin,
                            "gt_savaux_rank": int(gt_rank),
                            "proposal_savaux_rank": int(proposal_rank),
                            "savaux_correct": int(savaux_bin == gt_bin),
                            "proposal_correct": int(proposal_bin == gt_bin),
                            "savaux_margin_db": float(result.savaux_margin_db),
                            "branch_color_mismatch": float(result.branch_color_mismatch),
                            "backend_ran": 1,
                        }
                    )
            symbol_index += 1

    symbol_count = len(gt_bins)
    savaux_selected = selected["savaux"]
    rows: list[dict[str, Any]] = []
    for method in methods:
        errors = sum(
            int(value) != int(gt)
            for value, gt in zip(selected[method], gt_bins, strict=True)
        )
        fixes = sum(
            int(savaux) != int(gt) and int(value) == int(gt)
            for value, savaux, gt in zip(
                selected[method], savaux_selected, gt_bins, strict=True
            )
        )
        breaks = sum(
            int(savaux) == int(gt) and int(value) != int(gt)
            for value, savaux, gt in zip(
                selected[method], savaux_selected, gt_bins, strict=True
            )
        )
        bank_name = method.removesuffix("_conditional_lora")
        rows.append(
            {
                "method": str(method),
                "pattern_count": (
                    0 if method == "savaux" else len(detector_banks[bank_name].names)
                ),
                "symbol_count": int(symbol_count),
                "errors": int(errors),
                "ser": float(errors / max(1, symbol_count)),
                "fixes_vs_savaux": int(fixes),
                "breaks_vs_savaux": int(breaks),
                "elapsed_seconds": float(elapsed[method]),
                "mean_milliseconds_per_symbol": float(
                    1000.0 * elapsed[method] / max(1, symbol_count)
                ),
                "triggered_symbols": int(triggered[method]),
                "trigger_rate": float(triggered[method] / max(1, symbol_count)),
                "screened_symbols": int(screened[method]),
                "screen_rate": float(screened[method] / max(1, symbol_count)),
                "accepted_proposals": 0,
                "rejected_proposals": 0,
            }
        )
    return rows, diagnostics


__all__ = [
    "build_selected_pattern_banks",
    "estimate_training_covariance",
    "evaluate_conditional_condition",
]
