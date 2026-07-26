#!/usr/bin/env python3
"""Measure whether ground-truth LoRa interleaver blocks are phase coherent.

This is an oracle diagnostic, not a decoder.  Payload ground truth is used only
to select the correct complex FFT/branch-GLS coefficient and to estimate one
linear phase slope per coding block.  The same oracle slope is then applied to
the hard-decision block and random valid LoRa coding blocks to test whether the
coherence is discriminative rather than merely achievable after phase fitting.

No ``RN x RN`` covariance is constructed.  Colored-noise simulations may use
the existing candidate-wise bank of ``R x R`` branch covariance matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from weak_decoder.baselines.symfec.paper_symfec_decoder import (  # noqa: E402
    _codewords_to_payload_symbol_values,
)
from weak_decoder.decoding.payload_codec import encode_hamming_nibble  # noqa: E402
from weak_decoder.os_lora.system.oversampled_glrt import (  # noqa: E402
    BranchNoiseModel,
    branch_gls_scores,
    estimate_branch_noise_model,
    identity_branch_noise_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["0_0_0_10_14_32"])
    parser.add_argument("--input-iq", type=Path, default=None)
    parser.add_argument("--symbols", type=Path, default=None)
    parser.add_argument("--groundtruth", type=Path, default=None)
    parser.add_argument("--name", default="explicit_capture")
    parser.add_argument("--snrs", nargs="*", type=float, default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[53])
    parser.add_argument(
        "--noise-shape", choices=("white", "lowpass", "ar1"), default="white"
    )
    parser.add_argument("--noise-filter-taps", type=int, default=65)
    parser.add_argument("--noise-color-magnitude", type=float, default=0.85)
    parser.add_argument("--noise-color-phase-rad", type=float, default=0.7)
    parser.add_argument("--branch-loading", type=float, default=0.50)
    parser.add_argument(
        "--branch-covariance-mode", choices=("pooled", "per_bin"), default="per_bin"
    )
    parser.add_argument("--noise-windows", type=int, default=128)
    parser.add_argument("--max-packets", type=int, default=7)
    parser.add_argument("--random-valid-blocks", type=int, default=64)
    parser.add_argument("--phase-grid-points", type=int, default=4097)
    parser.add_argument("--random-seed", type=int, default=7301)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT
        / "data"
        / "experiments"
        / "interleaver_block_coherence_oracle",
    )
    return parser.parse_args()


def _external_gt(path: Path) -> dict[int, int]:
    output: dict[int, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            output[int(row["frame_symbol_index"])] = int(row["groundtruth_fft_bin"])
    if not output:
        raise RuntimeError(f"no ground-truth rows found in {path}")
    return output


def _attach_gt(packets: Sequence[dict[str, Any]], gt: dict[int, int]) -> None:
    for packet in packets:
        for symbol in packet["payload_symbols"]:
            frame_index = int(symbol["frame_symbol_index"])
            if frame_index not in gt:
                raise KeyError(f"ground truth has no frame_symbol_index={frame_index}")
            symbol["gt_bin"] = int(gt[frame_index])


def _dataset_inputs(args: argparse.Namespace) -> list[tuple[str, Path, Path, Path | None]]:
    if args.input_iq is not None or args.symbols is not None:
        if args.input_iq is None or args.symbols is None:
            raise ValueError("--input-iq and --symbols must be provided together")
        return [
            (
                str(args.name),
                args.input_iq.resolve(),
                args.symbols.resolve(),
                args.groundtruth.resolve() if args.groundtruth is not None else None,
            )
        ]
    return [(str(dataset), *dataset_paths(str(dataset)), None) for dataset in args.datasets]


def _coherence(values: Sequence[complex], slope_rad_per_symbol: float = 0.0) -> float:
    z = np.asarray(values, dtype=np.complex128)
    if z.size == 0:
        return 0.0
    indexes = np.arange(z.size, dtype=np.float64)
    aligned = z * np.exp(-1j * float(slope_rad_per_symbol) * indexes)
    denominator = float(z.size) * float(np.sum(np.abs(z) ** 2))
    if denominator <= 1e-30:
        return 0.0
    return float(np.clip(abs(np.sum(aligned)) ** 2 / denominator, 0.0, 1.0))


def _best_linear_coherence(
    values: Sequence[complex],
    phase_grid: np.ndarray,
    phase_kernels: np.ndarray | None = None,
) -> tuple[float, float]:
    z = np.asarray(values, dtype=np.complex128)
    if z.size == 0 or float(np.sum(np.abs(z) ** 2)) <= 1e-30:
        return 0.0, 0.0
    if phase_kernels is None:
        indexes = np.arange(z.size, dtype=np.float64)
        kernels = np.exp(-1j * phase_grid[:, None] * indexes[None, :])
    else:
        kernels = np.asarray(phase_kernels, dtype=np.complex128)
        if kernels.shape != (phase_grid.size, z.size):
            raise ValueError("phase kernel dimensions do not match the block")
    numerator = np.abs(kernels @ z) ** 2
    denominator = float(z.size) * float(np.sum(np.abs(z) ** 2))
    coherence = numerator / max(denominator, 1e-30)
    selected = int(np.argmax(coherence))
    return float(np.clip(coherence[selected], 0.0, 1.0)), float(phase_grid[selected])


def _complex_gls_spectrum(
    gls_observations: np.ndarray,
    model: BranchNoiseModel,
    steering: np.ndarray | None = None,
) -> np.ndarray:
    observations = np.asarray(gls_observations, dtype=np.complex128)
    active = (
        np.asarray(model.steering, dtype=np.complex128)
        if steering is None
        else np.asarray(steering, dtype=np.complex128)
    )
    inverse = np.asarray(model.inverse_covariance, dtype=np.complex128)
    if inverse.ndim == 2:
        inverse_steering = inverse @ active
        information = max(float(np.real(np.vdot(active, inverse_steering))), 1e-30)
        projected = observations @ np.conjugate(inverse_steering)
        return (projected / information).astype(np.complex128)
    if inverse.ndim == 3:
        inverse_steering = np.einsum("krs,s->kr", inverse, active, optimize=True)
        information = np.maximum(
            np.real(np.einsum("r,kr->k", active.conj(), inverse_steering, optimize=True)),
            1e-30,
        )
        projected = np.einsum(
            "kr,kr->k", np.conjugate(inverse_steering), observations, optimize=True
        )
        return (projected / information).astype(np.complex128)
    raise ValueError("branch inverse covariance must be 2-D or 3-D")


def _random_valid_bins(
    sf: int,
    cr: int,
    ldro: bool,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    sf_app = int(sf) - 2 if bool(ldro) else int(sf)
    nibbles = rng.integers(0, 16, size=sf_app)
    codewords = tuple(encode_hamming_nibble(int(value), cr_app=int(cr)) for value in nibbles)
    symbol_values = _codewords_to_payload_symbol_values(
        codewords, sf=int(sf), cr=int(cr), ldro=bool(ldro)
    )
    divisor = 4 if bool(ldro) else 1
    n_bins = 1 << int(sf)
    return tuple(int((int(value) * divisor + 1) % n_bins) for value in symbol_values)


def _branch_model_for_group(
    clean: np.ndarray,
    samples: np.ndarray,
    sf: int,
    os_factor: int,
    snr_db: float | None,
    args: argparse.Namespace,
) -> BranchNoiseModel:
    if snr_db is None or str(args.noise_shape) == "white":
        return identity_branch_noise_model(int(os_factor))
    length = (1 << int(sf)) * int(os_factor)
    count = min(int(args.noise_windows), int(samples.size) // length)
    if count < 2:
        return identity_branch_noise_model(int(os_factor))
    stop = count * length
    injected_noise = (
        np.asarray(samples[:stop], dtype=np.complex64)
        - np.asarray(clean[:stop], dtype=np.complex64)
    ).reshape(count, length)
    return estimate_branch_noise_model(
        injected_noise,
        int(sf),
        int(os_factor),
        diagonal_loading=float(args.branch_loading),
        covariance_mode=str(args.branch_covariance_mode),
    )


def _block_rows(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    model: BranchNoiseModel,
    dataset: str,
    snr_db: float | None,
    seed: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    phase_grid = np.linspace(
        -math.pi,
        math.pi,
        max(257, int(args.phase_grid_points)),
        endpoint=False,
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(args.random_seed) + 1009 * int(seed))
    rows: list[dict[str, Any]] = []
    for packet in packets:
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        block_len = int(packet["cr"]) + 4
        phase_kernels = np.exp(
            -1j
            * phase_grid[:, None]
            * np.arange(block_len, dtype=np.float64)[None, :]
        )
        origin_shift = os_factor // 2
        header_start = int(packet["header_start_sample"]) + origin_shift
        spectra: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        gt_bins: list[int] = []
        hard_bins: list[int] = []
        for symbol in packet["payload_symbols"]:
            try:
                _combined, branches, _phase = paper_oversampled_spectrum(
                    samples,
                    int(symbol["start_sample"]) + origin_shift,
                    sf,
                    os_factor,
                    int(packet["cfo_int"]),
                    float(packet["cfo_frac"]),
                    header_start,
                    "continuous",
                )
            except ValueError:
                continue
            gls = branch_gls_scores(
                branches,
                os_factor,
                noise_model=model,
                top_l=8,
            )
            complex_spectrum = _complex_gls_spectrum(gls.observations, model)
            spectra.append(complex_spectrum)
            scores.append(np.asarray(gls.scores, dtype=np.float64))
            gt_bins.append(int(symbol["gt_bin"]) % (1 << sf))
            hard_bins.append(int(gls.selected_bin))
        complete = len(spectra) - len(spectra) % block_len
        for offset in range(0, complete, block_len):
            block_spectra = spectra[offset : offset + block_len]
            block_scores = scores[offset : offset + block_len]
            block_gt = tuple(gt_bins[offset : offset + block_len])
            block_hard = tuple(hard_bins[offset : offset + block_len])
            gt_values = tuple(
                complex(block_spectra[index][raw_bin])
                for index, raw_bin in enumerate(block_gt)
            )
            gt_best_coherence, gt_slope = _best_linear_coherence(
                gt_values, phase_grid, phase_kernels
            )

            def append_hypothesis(kind: str, trial: int, bins: Sequence[int]) -> None:
                values = tuple(
                    complex(block_spectra[index][int(raw_bin)])
                    for index, raw_bin in enumerate(bins)
                )
                own_best, own_slope = _best_linear_coherence(
                    values, phase_grid, phase_kernels
                )
                ranks = []
                for index, raw_bin in enumerate(bins):
                    order = np.argsort(block_scores[index])[::-1]
                    ranks.append(int(np.flatnonzero(order == int(raw_bin))[0]) + 1)
                rows.append(
                    {
                        "dataset": dataset,
                        "snr_db": "" if snr_db is None else float(snr_db),
                        "seed": int(seed),
                        "packet_index": int(packet["packet_index"]),
                        "block_index": int(offset // block_len),
                        "block_symbol_offset": int(offset),
                        "sf": sf,
                        "cr": int(packet["cr"]),
                        "ldro": int(bool(packet["ldro"])),
                        "block_len": block_len,
                        "hypothesis": kind,
                        "trial": int(trial),
                        "hard_errors": int(
                            sum(int(left) != int(right) for left, right in zip(block_hard, block_gt))
                        ),
                        "raw_coherence": _coherence(values),
                        "gt_slope_coherence": _coherence(values, gt_slope),
                        "best_linear_coherence": own_best,
                        "best_slope_rad_per_symbol": own_slope,
                        "gt_slope_rad_per_symbol": gt_slope,
                        "mean_selected_rank": float(np.mean(ranks)),
                        "max_selected_rank": int(max(ranks)),
                        "selected_bins": "|".join(str(int(value)) for value in bins),
                    }
                )

            append_hypothesis("gt", 0, block_gt)
            append_hypothesis("hard", 0, block_hard)
            for trial in range(max(0, int(args.random_valid_blocks))):
                append_hypothesis(
                    "random_valid",
                    trial,
                    _random_valid_bins(sf, int(packet["cr"]), bool(packet["ldro"]), rng),
                )
    return rows


def _quantile(values: Sequence[float], probability: float) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    return float(np.quantile(array, float(probability))) if array.size else float("nan")


def _summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["snr_db"]), str(row["hypothesis"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (dataset, snr_db, hypothesis), items in sorted(groups.items()):
        summary: dict[str, Any] = {
            "dataset": dataset,
            "snr_db": snr_db,
            "hypothesis": hypothesis,
            "rows": len(items),
            "blocks": len(
                {
                    (int(item["seed"]), int(item["packet_index"]), int(item["block_index"]))
                    for item in items
                }
            ),
        }
        for field in ("raw_coherence", "gt_slope_coherence", "best_linear_coherence"):
            values = [float(item[field]) for item in items]
            summary[f"{field}_q10"] = _quantile(values, 0.10)
            summary[f"{field}_median"] = _quantile(values, 0.50)
            summary[f"{field}_q90"] = _quantile(values, 0.90)
        output.append(summary)
    return output


def _paired_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: dict[tuple[str, str, int, int, int], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["snr_db"]),
            int(row["seed"]),
            int(row["packet_index"]),
            int(row["block_index"]),
        )
        blocks.setdefault(key, {}).setdefault(str(row["hypothesis"]), []).append(row)
    output: list[dict[str, Any]] = []
    for key, hypotheses in sorted(blocks.items()):
        if not all(name in hypotheses for name in ("gt", "hard", "random_valid")):
            continue
        gt = hypotheses["gt"][0]
        hard = hypotheses["hard"][0]
        random_rows = hypotheses["random_valid"]
        gt_fixed = float(gt["gt_slope_coherence"])
        hard_fixed = float(hard["gt_slope_coherence"])
        random_fixed = np.asarray(
            [float(item["gt_slope_coherence"]) for item in random_rows], dtype=np.float64
        )
        random_best = np.asarray(
            [float(item["best_linear_coherence"]) for item in random_rows], dtype=np.float64
        )
        output.append(
            {
                "dataset": key[0],
                "snr_db": key[1],
                "seed": key[2],
                "packet_index": key[3],
                "block_index": key[4],
                "hard_errors": int(gt["hard_errors"]),
                "gt_fixed_coherence": gt_fixed,
                "hard_fixed_coherence": hard_fixed,
                "gt_beats_hard_fixed": int(gt_fixed > hard_fixed + 1e-12),
                "gt_best_coherence": float(gt["best_linear_coherence"]),
                "hard_best_coherence": float(hard["best_linear_coherence"]),
                "gt_beats_hard_own_best": int(
                    float(gt["best_linear_coherence"])
                    > float(hard["best_linear_coherence"]) + 1e-12
                ),
                "random_trials": int(random_fixed.size),
                "gt_beats_random_fixed_fraction": float(np.mean(gt_fixed > random_fixed)),
                "gt_beats_random_own_best_fraction": float(
                    np.mean(float(gt["best_linear_coherence"]) > random_best)
                ),
                "random_fixed_max": float(np.max(random_fixed)),
                "random_own_best_max": float(np.max(random_best)),
            }
        )
    return output


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset, iq_path, symbol_path, gt_path in _dataset_inputs(args):
        packets_all = load_packets(symbol_path)
        if gt_path is not None:
            _attach_gt(packets_all, _external_gt(gt_path))
        packets = (
            packets_all[: int(args.max_packets)]
            if int(args.max_packets) > 0
            else packets_all
        )
        if not packets:
            raise RuntimeError(f"no payload packets found for {dataset}")
        clean = np.memmap(iq_path, dtype=np.complex64, mode="r")
        reference_power, _count, _packets = signal_reference_power(
            clean, packets, "packet", None
        )
        snr_values: tuple[float | None, ...] = (
            tuple(float(value) for value in args.snrs) if args.snrs else (None,)
        )
        seeds = tuple(int(value) for value in args.seeds) if args.snrs else (int(args.seeds[0]),)
        for snr_db in snr_values:
            for seed in seeds:
                samples = noise_samples(
                    clean,
                    snr_db,
                    seed,
                    reference_power,
                    noise_shape=str(args.noise_shape),
                    os_factor=int(packets[0]["os_factor"]),
                    filter_taps=int(args.noise_filter_taps),
                    color_magnitude=float(args.noise_color_magnitude),
                    color_phase_rad=float(args.noise_color_phase_rad),
                )
                model = _branch_model_for_group(
                    clean,
                    samples,
                    int(packets[0]["sf"]),
                    int(packets[0]["os_factor"]),
                    snr_db,
                    args,
                )
                group_rows = _block_rows(
                    samples, packets, model, dataset, snr_db, seed, args
                )
                rows.extend(group_rows)
                gt = [item for item in group_rows if item["hypothesis"] == "gt"]
                hard = [item for item in group_rows if item["hypothesis"] == "hard"]
                print(
                    f"{dataset} snr={snr_db} seed={seed}: blocks={len(gt)} "
                    f"gt_fixed_median={np.median([item['gt_slope_coherence'] for item in gt]):.4f} "
                    f"hard_fixed_median={np.median([item['gt_slope_coherence'] for item in hard]):.4f}",
                    flush=True,
                )
                del samples
    summary = _summary_rows(rows)
    paired = _paired_rows(rows)
    write_csv(output_dir / "hypotheses.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_blocks.csv", paired)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
