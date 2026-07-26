#!/usr/bin/env python3
"""Probe the five RF-SR/Savaux signal paths on one synchronized LoRa symbol.

This is the minimal integration entry, not the final PER sweep. It deliberately
uses the existing Savaux spectrum and branch-GLS functions without modifying
their implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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

from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    paper_oversampled_spectrum,
)
from weak_decoder.os_lora.experiment_support.noise_windows import (  # noqa: E402
    covariance_correlation_stats,
)
from weak_decoder.os_lora.system.oversampled_glrt import (  # noqa: E402
    BranchNoiseModel,
    branch_gls_scores,
    estimate_branch_noise_model,
    identity_branch_noise_model,
)
from weak_decoder.rf_super_resolution import (  # noqa: E402
    DEFAULT_OTA_CHECKPOINT,
    RFSRFrontendConfig,
    RFSuperResolutionFrontend,
    default_rfsr_repo_root,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-low-iq", type=Path, required=True)
    parser.add_argument("--start-low", type=int, required=True)
    parser.add_argument("--sf", type=int, default=12)
    parser.add_argument("--bw", type=float, default=125_000.0)
    parser.add_argument("--input-rate", type=float, default=250_000.0)
    parser.add_argument("--output-rate", type=float, default=1_000_000.0)
    parser.add_argument("--gt-bin", type=int, default=None)
    parser.add_argument("--cfo-int", type=int, default=0)
    parser.add_argument("--cfo-frac", type=float, default=0.0)
    parser.add_argument(
        "--cfo-correction-mode",
        choices=("none", "symbol", "continuous"),
        default="symbol",
    )
    parser.add_argument("--header-start-low", type=int, default=None)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument(
        "--rfsr-repo",
        type=Path,
        default=default_rfsr_repo_root(),
        help="RFSR source tree (default: bundled third_party/rfsr)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-name", default=DEFAULT_OTA_CHECKPOINT)
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-low", type=int, default=68)
    parser.add_argument("--chunk-input-samples", type=int, default=65_536)
    parser.add_argument("--noise-low-iq", type=Path, default=None)
    parser.add_argument("--noise-low-offset", type=int, default=0)
    parser.add_argument("--noise-windows", type=int, default=32)
    parser.add_argument("--branch-loading", type=float, default=0.5)
    parser.add_argument("--noise-training-bins", type=int, default=16)
    parser.add_argument("--top-l", type=int, default=8)
    parser.add_argument("--native-high-iq", type=Path, default=None)
    parser.add_argument("--start-high", type=int, default=None)
    parser.add_argument("--header-start-high", type=int, default=None)
    parser.add_argument("--noise-high-iq", type=Path, default=None)
    parser.add_argument("--noise-high-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _integer_osr(sample_rate: float, bw: float, label: str) -> int:
    ratio = float(sample_rate) / float(bw)
    rounded = int(round(ratio))
    if rounded <= 0 or not math.isclose(ratio, rounded, abs_tol=1e-9):
        raise ValueError(f"{label} sample-rate / BW must be a positive integer, got {ratio}")
    return rounded


def _memmap(path: Path) -> np.memmap:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if resolved.stat().st_size % np.dtype(np.complex64).itemsize:
        raise ValueError(f"complex64 IQ file has an invalid byte length: {resolved}")
    return np.memmap(resolved, dtype=np.complex64, mode="r")


def _true_false_ratio_db(scores: np.ndarray, gt_bin: int | None) -> float | None:
    if gt_bin is None:
        return None
    power = np.asarray(scores, dtype=np.float64)
    gt = int(gt_bin) % int(power.size)
    false = power.copy()
    false[gt] = -np.inf
    strongest_false = float(np.max(false)) if false.size > 1 else 0.0
    return float(10.0 * math.log10((float(power[gt]) + 1e-30) / (strongest_false + 1e-30)))


def _metric_row(
    method: str,
    scores: np.ndarray,
    selected_bin: int,
    gt_bin: int | None,
) -> dict[str, Any]:
    selected = int(selected_bin)
    return {
        "method": str(method),
        "selected_bin": selected,
        "gt_bin": None if gt_bin is None else int(gt_bin),
        "correct": None if gt_bin is None else int(selected == int(gt_bin)),
        "true_to_strongest_false_db": _true_false_ratio_db(scores, gt_bin),
    }


def _spectrum(
    samples: np.ndarray,
    start: int,
    sf: int,
    os_factor: int,
    args: argparse.Namespace,
    header_start: int | None,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    combined, branches, _ = paper_oversampled_spectrum(
        samples=samples,
        start_sample=int(start),
        sf=int(sf),
        os_factor=int(os_factor),
        cfo_int=int(args.cfo_int),
        cfo_frac=float(args.cfo_frac),
        header_start_sample=None if header_start is None else int(header_start),
        cfo_correction_mode=str(args.cfo_correction_mode),
    )
    return combined, branches


def _frontend_view(
    source: np.ndarray,
    start: int,
    symbol_input_samples: int,
    context: int,
    frontend: RFSuperResolutionFrontend,
    mode: str,
    snr_db: float,
    header_start: int | None,
) -> tuple[np.ndarray, int, int | None]:
    left = max(0, int(start) - int(context))
    right = min(int(source.size), int(start) + int(symbol_input_samples) + int(context))
    if int(start) < 0 or int(start) + int(symbol_input_samples) > int(source.size):
        raise ValueError("requested low-rate symbol exceeds the input IQ file")
    transformed = frontend.transform(
        np.asarray(source[left:right], dtype=np.complex64),
        mode=mode,
        snr_db=float(snr_db),
    )
    up = int(frontend.config.upsample_factor)
    local_start = (int(start) - left) * up
    local_header = None if header_start is None else (int(header_start) - left) * up
    return transformed, local_start, local_header


def _noise_windows(
    path: Path | None,
    offset: int,
    count: int,
    symbol_samples: int,
) -> np.ndarray | None:
    if path is None:
        return None
    source = _memmap(path)
    start = int(offset)
    stop = start + int(count) * int(symbol_samples)
    if start < 0 or stop > int(source.size):
        raise ValueError(f"noise IQ contains fewer than {count} requested windows")
    return np.asarray(source[start:stop], dtype=np.complex64).reshape(
        int(count), int(symbol_samples)
    )


def _frontend_noise_windows(
    windows: np.ndarray | None,
    frontend: RFSuperResolutionFrontend,
    mode: str,
    snr_db: float,
) -> np.ndarray | None:
    if windows is None:
        return None
    # The CLI selects one contiguous noise segment, so preserve continuity
    # through the interpolation FIR and split it only after transformation.
    transformed = frontend.transform(
        windows.reshape(-1), mode=mode, snr_db=snr_db
    )
    return transformed.reshape(int(windows.shape[0]), -1).astype(np.complex64)


def _noise_model(
    windows: np.ndarray | None,
    sf: int,
    os_factor: int,
    args: argparse.Namespace,
) -> tuple[BranchNoiseModel, dict[str, Any]]:
    if windows is None:
        model = identity_branch_noise_model(os_factor)
        return model, {
            "source": "identity",
            "snapshot_count": 0,
            "diagonal_cv": 0.0,
            "mean_abs_correlation": 0.0,
            "max_abs_correlation": 0.0,
        }
    n_bins = 1 << int(sf)
    count = max(1, min(int(args.noise_training_bins), n_bins))
    training_bins: Sequence[int] = tuple(
        int(value) for value in np.linspace(0, n_bins, count, endpoint=False)
    )
    model = estimate_branch_noise_model(
        windows,
        sf=int(sf),
        os_factor=int(os_factor),
        training_bins=training_bins,
        cfo_int=int(args.cfo_int),
        cfo_frac=float(args.cfo_frac),
        diagonal_loading=float(args.branch_loading),
        covariance_mode="pooled",
    )
    diagonal_cv, mean_correlation, max_correlation = covariance_correlation_stats(
        model.covariance
    )
    return model, {
        "source": "noise_iq",
        "snapshot_count": int(model.snapshot_count),
        "diagonal_cv": float(diagonal_cv),
        "mean_abs_correlation": float(mean_correlation),
        "max_abs_correlation": float(max_correlation),
    }


def main() -> int:
    args = _parse_args()
    sf = int(args.sf)
    input_osr = _integer_osr(args.input_rate, args.bw, "input")
    output_osr = _integer_osr(args.output_rate, args.bw, "output")
    if not math.isclose(
        float(args.output_rate) / float(args.input_rate), 4.0, abs_tol=1e-9
    ):
        raise ValueError("the published RF-SR path requires 250 kSPS -> 1 MSPS (4x)")
    if output_osr != input_osr * 4:
        raise ValueError("output OSR must equal four times input OSR")

    frontend = RFSuperResolutionFrontend(
        RFSRFrontendConfig(
            repo_root=args.rfsr_repo,
            checkpoint=args.checkpoint,
            checkpoint_name=str(args.checkpoint_name),
            model_variant=args.model_variant,
            device=str(args.device),
            chunk_input_samples=int(args.chunk_input_samples),
            overlap_input_samples=max(
                int(args.context_low),
                RFSuperResolutionFrontend.minimum_overlap_input_samples,
            ),
        )
    )
    low = _memmap(args.input_low_iq)
    n_bins = 1 << sf
    low_symbol_samples = n_bins * input_osr
    high_symbol_samples = n_bins * output_osr
    gt_bin = None if args.gt_bin is None else int(args.gt_bin) % n_bins

    _low_combined, low_branches = _spectrum(
        low,
        int(args.start_low),
        sf,
        input_osr,
        args,
        args.header_start_low,
    )
    rows = [
        _metric_row(
            "native_low_lora",
            np.abs(low_branches[0]).astype(np.float64) ** 2,
            int(np.argmax(np.abs(low_branches[0]) ** 2)),
            gt_bin,
        )
    ]

    low_noise = _noise_windows(
        args.noise_low_iq,
        int(args.noise_low_offset),
        int(args.noise_windows),
        low_symbol_samples,
    )
    noise_report: dict[str, Any] = {}
    views: dict[str, tuple[np.ndarray, int, int | None]] = {}
    models: dict[str, BranchNoiseModel] = {}
    for name, mode in (("interpolation", "interpolation"), ("rfsr", "rfsr")):
        view = _frontend_view(
            low,
            int(args.start_low),
            low_symbol_samples,
            int(args.context_low),
            frontend,
            mode,
            float(args.snr_db),
            args.header_start_low,
        )
        views[name] = view
        transformed_noise = _frontend_noise_windows(
            low_noise, frontend, mode, float(args.snr_db)
        )
        models[name], noise_report[name] = _noise_model(
            transformed_noise, sf, output_osr, args
        )

    interp_iq, interp_start, interp_header = views["interpolation"]
    interp_combined, _interp_branches = _spectrum(
        interp_iq, interp_start, sf, output_osr, args, interp_header
    )
    interp_power = np.abs(interp_combined).astype(np.float64) ** 2
    rows.append(
        _metric_row(
            "interpolation_savaux",
            interp_power,
            int(np.argmax(interp_power)),
            gt_bin,
        )
    )

    rfsr_iq, rfsr_start, rfsr_header = views["rfsr"]
    rfsr_combined, rfsr_branches = _spectrum(
        rfsr_iq, rfsr_start, sf, output_osr, args, rfsr_header
    )
    rfsr_ordinary_power = np.abs(rfsr_branches[0]).astype(np.float64) ** 2
    rows.append(
        _metric_row(
            "rfsr_lora",
            rfsr_ordinary_power,
            int(np.argmax(rfsr_ordinary_power)),
            gt_bin,
        )
    )
    rfsr_gls = branch_gls_scores(
        rfsr_branches,
        output_osr,
        noise_model=models["rfsr"],
        top_l=int(args.top_l),
    )
    rows.append(
        _metric_row(
            "rfsr_savaux_branch_gls",
            rfsr_gls.scores,
            int(rfsr_gls.selected_bin),
            gt_bin,
        )
    )

    if args.native_high_iq is not None:
        if args.start_high is None:
            raise ValueError("--start-high is required with --native-high-iq")
        high = _memmap(args.native_high_iq)
        high_combined, _high_branches = _spectrum(
            high,
            int(args.start_high),
            sf,
            output_osr,
            args,
            args.header_start_high,
        )
        high_noise = _noise_windows(
            args.noise_high_iq,
            int(args.noise_high_offset),
            int(args.noise_windows),
            high_symbol_samples,
        )
        _high_model, noise_report["native_high"] = _noise_model(
            high_noise, sf, output_osr, args
        )
        high_power = np.abs(high_combined).astype(np.float64) ** 2
        rows.append(
            _metric_row(
                "native_high_savaux",
                high_power,
                int(np.argmax(high_power)),
                gt_bin,
            )
        )
    else:
        rows.append(
            {
                "method": "native_high_savaux",
                "status": "not_run: --native-high-iq was not provided",
            }
        )

    result = {
        "scope": "single synchronized symbol smoke probe",
        "rates": {
            "input_sps": float(args.input_rate),
            "output_sps": float(args.output_rate),
            "bandwidth_hz": float(args.bw),
            "input_osr": input_osr,
            "output_osr": output_osr,
        },
        "rfsr": asdict(frontend.provenance),
        "methods": rows,
        "branch_noise": noise_report,
        "notes": [
            "interpolation_savaux and rfsr_savaux_branch_gls share the author's exact polyphase stage",
            "PER/SER aggregation belongs in the full synchronized-packet experiment",
            "the high-rate arm is valid only for independently acquired ADC IQ",
        ],
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
