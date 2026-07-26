"""绘制入选非均匀 pattern 的类 FFT LoRa 匹配滤波频谱。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.common import (  # noqa: E402
    dataset_paths,
    load_packets,
    signal_reference_power,
)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.os_lora.experiment_support.pattern_training import (  # noqa: E402
    bootstrap_offpacket_noise as _offpacket_bootstrap_samples,
)
from weak_decoder.os_lora.system.noise import select_background_bins as _background_bins  # noqa: E402
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    NonuniformPatternBank,
    build_pattern_bank,
    crossfit_weighted_spectrum,
    lora_wrap_consistency_power,
    matrix_free_crossfit_gls_spectrum_power,
    pattern_bank_split_spectra,
    plain_pattern_fft_spectra,
    prepare_dechirped_symbol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="0_0_0_10_14_16")
    parser.add_argument("--snr", type=float, default=-23.0)
    parser.add_argument("--no-added-noise", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packet-index", type=int, default=0)
    parser.add_argument("--payload-symbol-index", type=int, default=32)
    parser.add_argument("--max-reference-packets", type=int, default=2)
    parser.add_argument("--bootstrap-source-windows", type=int, default=256)
    parser.add_argument("--guard-symbols", type=float, default=2.0)
    parser.add_argument("--exclude-top", type=int, default=8)
    parser.add_argument("--exclude-guard-bins", type=int, default=1)
    parser.add_argument("--gls-loading", type=float, default=0.05)
    parser.add_argument("--crossfit-folds", type=int, default=2)
    parser.add_argument("--cg-iterations", type=int, default=4)
    parser.add_argument("--wrap-exponent", type=float, default=0.5)
    parser.add_argument("--wrap-minimum-segment", type=int, default=16)
    parser.add_argument(
        "--plot-half-width",
        type=int,
        default=0,
        help="Show only GT bin +/- this many bins; 0 shows the full spectrum.",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "compact96_final_full" / "selection.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "matched_filter_spectrum" / "matched_filter_spectrum.png",
    )
    return parser.parse_args()


def selected_bank(sf: int, os_factor: int, selection_csv: Path) -> NonuniformPatternBank:
    with Path(selection_csv).open("r", newline="", encoding="utf-8") as handle:
        names = tuple(
            row["pattern_name"]
            for row in csv.DictReader(handle)
            if int(row["pattern_count"]) == 8 and row["selection_method"] == "information"
        )
    if len(names) != 8:
        raise RuntimeError(f"expected 8 selected patterns, got {len(names)}")
    full = build_pattern_bank(sf, os_factor, kind="multiscale")
    by_name = {name: offsets for name, offsets in zip(full.names, full.offsets, strict=True)}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(f"selected patterns are absent from multiscale bank: {missing}")
    return NonuniformPatternBank(
        names=names,
        offsets=tuple(by_name[name] for name in names),
        os_factor=os_factor,
        sf=sf,
        kind="compact96_information_8",
    )


def normalized_db(power: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    values = np.maximum(np.asarray(power, dtype=np.float64), 0.0)
    peak = max(float(np.max(values)), 1e-30)
    return np.maximum(10.0 * np.log10(values / peak + 1e-30), floor_db)


def mark_bins(ax: plt.Axes, gt_bin: int, selected_bin: int) -> None:
    ax.axvline(gt_bin, color="tab:green", linewidth=1.5, linestyle="--", label=f"GT bin {gt_bin}")
    if selected_bin != gt_bin:
        ax.axvline(
            selected_bin,
            color="tab:red",
            linewidth=1.2,
            linestyle=":",
            label=f"selected {selected_bin}",
        )


def main() -> int:
    args = parse_args()
    iq_path, symbol_path = dataset_paths(str(args.dataset))
    clean = np.fromfile(iq_path, dtype=np.complex64)
    all_packets = load_packets(symbol_path)
    reference_packets = all_packets[: max(1, int(args.max_reference_packets))]
    reference_power, _sample_count, _packet_count = signal_reference_power(
        clean,
        reference_packets,
        "packet",
        None,
    )
    if bool(args.no_added_noise):
        samples = np.asarray(clean, dtype=np.complex64)
        condition_label = "original IQ, no added noise"
    else:
        samples = _offpacket_bootstrap_samples(
            clean=clean,
            packets=all_packets,
            snr_db=float(args.snr),
            seed=int(args.seed),
            reference_power=float(reference_power),
            max_source_windows=int(args.bootstrap_source_windows),
            guard_symbols=float(args.guard_symbols),
        )
        condition_label = f"SNR {args.snr:g} dB"

    packet = all_packets[int(args.packet_index)]
    symbol = packet["payload_symbols"][int(args.payload_symbol_index)]
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    origin_shift = os_factor // 2
    start = int(symbol["start_sample"]) + origin_shift
    header_start = int(packet["header_start_sample"]) + origin_shift
    gt_bin = int(symbol["gt_bin"])

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
    background = _background_bins(
        savaux_power,
        exclude_top=int(args.exclude_top),
        guard_bins=int(args.exclude_guard_bins),
    )
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
    bank = selected_bank(sf, os_factor, Path(args.selection_csv))
    fixed_branch = np.asarray(dechirped[0::os_factor], dtype=np.complex128)
    ordinary_dft_spectrum = np.fft.fft(fixed_branch) / np.sqrt(float(1 << sf))
    ordinary_dft_power = np.abs(ordinary_dft_spectrum).astype(np.float64) ** 2
    plain_spectra = plain_pattern_fft_spectra(dechirped, bank)
    lora_spectra, head_spectra, tail_spectra = pattern_bank_split_spectra(dechirped, bank)
    plain_power = np.abs(np.mean(plain_spectra, axis=0)).astype(np.float64) ** 2
    matched_coherent_power = np.abs(np.mean(lora_spectra, axis=0)).astype(np.float64) ** 2

    gls = matrix_free_crossfit_gls_spectrum_power(
        lora_spectra,
        covariance_bins=background,
        diagonal_loading=float(args.gls_loading),
        folds=int(args.crossfit_folds),
        max_iterations=int(args.cg_iterations),
        tolerance=0.0,
    )
    weighted_head = crossfit_weighted_spectrum(head_spectra, gls.inverse_targets)
    weighted_tail = crossfit_weighted_spectrum(tail_spectra, gls.inverse_targets)
    final_power = lora_wrap_consistency_power(
        weighted_head,
        weighted_tail,
        base_power=gls.power,
        exponent=float(args.wrap_exponent),
        minimum_segment=int(args.wrap_minimum_segment),
    )

    savaux_bin = int(np.argmax(savaux_power))
    ordinary_dft_bin = int(np.argmax(ordinary_dft_power))
    plain_bin = int(np.argmax(plain_power))
    first_plain_bin = int(np.argmax(np.abs(plain_spectra[0]).astype(np.float64) ** 2))
    first_matched_bin = int(np.argmax(np.abs(lora_spectra[0]).astype(np.float64) ** 2))
    coherent_bin = int(np.argmax(matched_coherent_power))
    gls_bin = int(np.argmax(gls.power))
    final_bin = int(np.argmax(final_power))
    bins = np.arange(1 << sf)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    panels = axes.ravel()

    panels[0].plot(bins, normalized_db(ordinary_dft_power), color="tab:blue", linewidth=0.9)
    mark_bins(panels[0], gt_bin, ordinary_dft_bin)
    panels[0].set_title(f"Ordinary DFT: fixed branch q=0 (selected {ordinary_dft_bin})")

    panels[1].plot(bins, normalized_db(savaux_power), color="tab:cyan", linewidth=0.9)
    mark_bins(panels[1], gt_bin, savaux_bin)
    panels[1].set_title(f"Savaux: 4 corrected branches (selected {savaux_bin})")

    panels[2].plot(bins, normalized_db(plain_power), color="tab:orange", linewidth=0.9)
    mark_bins(panels[2], gt_bin, plain_bin)
    panels[2].set_title(f"8 nonuniform patterns, naive DFT sum (selected {plain_bin})")

    first_plain_power = np.abs(plain_spectra[0]).astype(np.float64) ** 2
    first_matched_power = np.abs(lora_spectra[0]).astype(np.float64) ** 2
    panels[3].plot(
        bins,
        normalized_db(first_plain_power),
        color="tab:orange",
        linewidth=0.9,
        label=f"naive DFT (peak {first_plain_bin})",
    )
    panels[3].plot(
        bins,
        normalized_db(first_matched_power),
        color="tab:green",
        linewidth=1.0,
        label=f"LoRa matched (peak {first_matched_bin})",
    )
    panels[3].axvline(gt_bin, color="tab:green", linewidth=1.5, linestyle="--", label=f"GT bin {gt_bin}")
    panels[3].set_title(f"Same pattern comparison: {bank.names[0]}")

    for index, spectrum in enumerate(lora_spectra):
        panels[4].plot(
            bins,
            normalized_db(np.abs(spectrum).astype(np.float64) ** 2),
            color="0.65",
            linewidth=0.55,
            alpha=0.45,
            label="individual patterns" if index == 0 else None,
        )
    panels[4].plot(
        bins,
        normalized_db(matched_coherent_power),
        color="tab:purple",
        linewidth=1.15,
        label="equal coherent sum",
    )
    mark_bins(panels[4], gt_bin, coherent_bin)
    panels[4].set_title(f"LoRa-aware matched-filter bank (selected {coherent_bin})")

    panels[5].plot(bins, normalized_db(gls.power), color="tab:cyan", linewidth=0.8, label="GLS")
    panels[5].plot(
        bins,
        normalized_db(final_power),
        color="tab:purple",
        linewidth=1.15,
        label="GLS + wrap consistency",
    )
    mark_bins(panels[5], gt_bin, final_bin)
    panels[5].set_title(f"Adaptive matched-filter spectrum (GLS {gls_bin}, final {final_bin})")

    if int(args.plot_half_width) > 0:
        left_bin = max(0, gt_bin - int(args.plot_half_width))
        right_bin = min((1 << sf) - 1, gt_bin + int(args.plot_half_width))
    else:
        left_bin = 0
        right_bin = (1 << sf) - 1
    for ax in panels:
        ax.set_xlim(left_bin, right_bin)
        ax.set_ylim(-60, 2)
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower left", fontsize=8)
    axes[0, 0].set_ylabel("Normalized power (dB)")
    axes[1, 0].set_ylabel("Normalized power (dB)")
    for ax in axes[1, :]:
        ax.set_xlabel("Candidate raw FFT bin")
    fig.suptitle(
        f"Nonuniform LoRa matched-filter spectra | {args.dataset} | "
        f"{condition_label} | packet {args.packet_index}, payload symbol {args.payload_symbol_index}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"output={output}")
    print(f"gt_bin={gt_bin}")
    print(f"ordinary_dft_bin={ordinary_dft_bin}")
    print(f"savaux_bin={savaux_bin}")
    print(f"plain_fft_bin={plain_bin}")
    print(f"first_pattern_plain_bin={first_plain_bin}")
    print(f"first_pattern_matched_bin={first_matched_bin}")
    print(f"matched_coherent_bin={coherent_bin}")
    print(f"gls_bin={gls_bin}")
    print(f"final_bin={final_bin}")
    print(f"cg_iterations={','.join(str(value) for value in gls.iterations)}")
    print("patterns=" + ",".join(bank.names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
