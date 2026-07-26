#!/usr/bin/env python3
"""低信噪比 wrong-bin 对照实验。

该脚本复用 run_low_snr_gt_bin_experiment.py 生成的 noisy IQ。对每个 payload
symbol 重算一次 corrected chip-rate FFT，然后同时读取 GT bin、低 SNR argmax、
除 GT 外最高峰 wrong_peak，以及若干人为错误 bin 的相位/幅度/能量占比。

目标是验证：低 SNR 下错误 bin 是否也能呈现类似 GT bin 的相位 residual 曲线。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


# 当前文件位于 weakPacket_decoding/scripts/experiments/。
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, dechirp_fft, positive_mod, signed_fft_bin  # noqa: E402


@dataclass(frozen=True)
class PayloadGt:
    frame_index: int
    packet_index: int
    event_index: int
    payload_symbol_index: int
    frame_symbol_index: int
    start_sample: int
    header_start_sample: int
    sf: int
    os_factor: int
    cfo_int: int
    cfo_frac: float
    sto_frac: float
    sfo_hat: float
    gt_raw_fft_bin: int


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    kind: str
    value: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-SNR payload phase/amplitude comparison for GT bins and intentionally wrong bins."
    )
    parser.add_argument(
        "-d",
        "--low-snr-dir",
        type=Path,
        required=True,
        help="Directory containing *_snr_mXXdB.bin files from run_low_snr_gt_bin_experiment.py.",
    )
    parser.add_argument(
        "-g",
        "--gt-symbol-csv",
        type=Path,
        required=True,
        help="Clean header-first symbol CSV. Payload rows with header_valid=1 provide GT raw_fft_bin.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <low-snr-dir>/wrong_bin_control.",
    )
    parser.add_argument(
        "--input-stem",
        type=str,
        default=None,
        help="Input stem used to find noisy bins. Default is inferred from the GT CSV stem.",
    )
    parser.add_argument("--snr-db", type=float, nargs="*", default=None, help="Optional SNR filter.")
    parser.add_argument("--packet", type=int, action="append", default=None, help="Optional packet_index filter.")
    parser.add_argument(
        "--cfo-correction-mode",
        choices=("symbol", "continuous"),
        default="continuous",
        help="FFT correction mode before reading bins. Default: continuous.",
    )
    parser.add_argument(
        "--offset-bins",
        type=str,
        default="1,-1,4,-4,16,-16,64,256",
        help="Comma-separated wrong-bin offsets relative to the clean GT bin.",
    )
    parser.add_argument(
        "--fixed-bins",
        type=str,
        default="",
        help="Comma-separated fixed raw FFT bins. If empty, --fixed-fractions is used.",
    )
    parser.add_argument(
        "--fixed-fractions",
        type=str,
        default="0,0.25,0.5,0.75",
        help="Fixed-bin fractions of FFT length, used only when --fixed-bins is empty.",
    )
    parser.add_argument(
        "--plot-wrong-count",
        type=int,
        default=5,
        help="Besides GT and wrong_peak, plot this many highest-risk wrong-bin candidates.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI.")
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    if value == "":
        return int(default)
    return int(float(value))


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return float(default)
    return float(value)


def _parse_int_list(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(float(item.strip())) for item in text.split(",") if item.strip()]


def _parse_float_list(text: str) -> list[float]:
    if not text.strip():
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _snr_label(snr_db: float) -> str:
    sign = "m" if float(snr_db) < 0 else "p"
    value = abs(float(snr_db))
    if abs(value - round(value)) < 1e-9:
        text = f"{int(round(value)):02d}"
    else:
        text = f"{value:.1f}".replace(".", "p")
    return f"snr_{sign}{text}dB"


def _infer_input_stem(gt_csv: Path) -> str:
    stem = gt_csv.stem
    for suffix in (
        "_header_first_symbols_continuous_cfo",
        "_header_first_symbols_framesync_valid_consistency",
        "_header_first_symbols",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _group_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (_int(row, "frame_index", -1), _int(row, "packet_index", -1), _int(row, "event_index", -1))


def load_payload_gt(path: Path, packet_filter: set[int] | None) -> list[PayloadGt]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    header_start_by_key: dict[tuple[int, int, int], int] = {}
    for row in rows:
        if row.get("stage") != "header":
            continue
        if _int(row, "stage_symbol_index", -1) == 0:
            header_start_by_key[_group_key(row)] = _int(row, "start_sample")

    symbols: list[PayloadGt] = []
    for row in rows:
        if row.get("stage") != "payload":
            continue
        if _int(row, "header_valid", 0) != 1:
            continue
        packet_index = _int(row, "packet_index", -1)
        if packet_filter is not None and packet_index not in packet_filter:
            continue

        key = _group_key(row)
        header_start_sample = header_start_by_key.get(key)
        if header_start_sample is None:
            frame_symbol_index = _int(row, "frame_symbol_index")
            samples_per_symbol = (1 << _int(row, "sf", 10)) * _int(row, "os_factor", 4)
            header_start_sample = _int(row, "start_sample") - frame_symbol_index * samples_per_symbol

        symbols.append(
            PayloadGt(
                frame_index=_int(row, "frame_index", -1),
                packet_index=packet_index,
                event_index=_int(row, "event_index", -1),
                payload_symbol_index=_int(row, "stage_symbol_index", -1),
                frame_symbol_index=_int(row, "frame_symbol_index", -1),
                start_sample=_int(row, "start_sample"),
                header_start_sample=int(header_start_sample),
                sf=_int(row, "sf", 10),
                os_factor=_int(row, "os_factor", 4),
                cfo_int=_int(row, "cfo_int", 0),
                cfo_frac=_float(row, "cfo_frac", 0.0),
                sto_frac=_float(row, "sto_frac", 0.0),
                sfo_hat=_float(row, "sfo_hat", 0.0),
                gt_raw_fft_bin=_int(row, "raw_fft_bin", -1),
            )
        )

    symbols.sort(key=lambda item: (item.packet_index, item.payload_symbol_index))
    if not symbols:
        raise ValueError("No GT payload symbols found. Need stage=payload and header_valid=1.")
    return symbols


def find_noisy_bins(low_snr_dir: Path, input_stem: str, snr_filter: Iterable[float] | None) -> list[tuple[float, Path]]:
    pattern = re.compile(rf"^{re.escape(input_stem)}_snr_([mp])(\d+(?:p\d+)?)dB\.bin$")
    wanted = None if snr_filter is None else {round(float(value), 6) for value in snr_filter}
    found: list[tuple[float, Path]] = []
    for path in low_snr_dir.glob(f"{input_stem}_snr_*dB.bin"):
        match = pattern.match(path.name)
        if not match:
            continue
        sign, value_text = match.groups()
        value = float(value_text.replace("p", "."))
        snr_db = -value if sign == "m" else value
        if wanted is not None and round(float(snr_db), 6) not in wanted:
            continue
        found.append((float(snr_db), path))
    found.sort(key=lambda item: item[0], reverse=True)
    if not found:
        raise FileNotFoundError(f"No noisy IQ bins found under {low_snr_dir} for stem {input_stem}.")
    return found


def symbol_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value / 2) + os_value * np.arange(n_bins, dtype=np.int64)


def corrected_spectrum(
    samples: np.ndarray,
    symbol: PayloadGt,
    downchirps: dict[tuple[int, int, float], np.ndarray],
    cfo_correction_mode: str,
) -> tuple[np.ndarray, float]:
    sf = int(symbol.sf)
    n_bins = 1 << sf
    indexes = symbol_indexes(symbol.start_sample, sf=symbol.sf, os_factor=symbol.os_factor)
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"Symbol exceeds IQ range at start_sample={symbol.start_sample}.")
    chip_symbol = np.asarray(samples[indexes], dtype=np.complex64)
    cfo_total = float(symbol.cfo_int) + float(symbol.cfo_frac)
    if cfo_correction_mode == "continuous":
        relative_chip_start = float(symbol.start_sample - symbol.header_start_sample) / float(symbol.os_factor)
        cfo_common_phase_rad = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
        chip_symbol = (chip_symbol * np.exp(-1j * cfo_common_phase_rad)).astype(np.complex64)
    else:
        cfo_common_phase_rad = 0.0

    key = (sf, int(symbol.cfo_int), float(symbol.cfo_frac))
    downchirp = downchirps.get(key)
    if downchirp is None:
        downchirp = build_downchirp(symbol.sf, cfo_int=symbol.cfo_int, cfo_frac=symbol.cfo_frac)
        downchirps[key] = downchirp
    return dechirp_fft(chip_symbol, downchirp), cfo_common_phase_rad


def build_candidate_specs(n_bins: int, offset_bins: str, fixed_bins: str, fixed_fractions: str) -> list[CandidateSpec]:
    specs = [
        CandidateSpec("gt", "gt", 0),
        CandidateSpec("argmax", "argmax", 0),
        CandidateSpec("wrong_peak", "wrong_peak", 0),
    ]
    seen = {("gt", 0), ("argmax", 0), ("wrong_peak", 0)}
    for offset in _parse_int_list(offset_bins):
        if positive_mod(offset, n_bins) == 0:
            continue
        key = ("offset", positive_mod(offset, n_bins))
        if key in seen:
            continue
        seen.add(key)
        sign = "+" if int(offset) > 0 else ""
        specs.append(CandidateSpec(f"off{sign}{offset}", "offset", int(offset)))

    fixed_values = _parse_int_list(fixed_bins)
    if not fixed_values:
        fixed_values = [int(round(frac * n_bins)) for frac in _parse_float_list(fixed_fractions)]
    for fixed in fixed_values:
        fixed = positive_mod(fixed, n_bins)
        key = ("fixed", fixed)
        if key in seen:
            continue
        seen.add(key)
        specs.append(CandidateSpec(f"fix{fixed:04d}", "fixed", int(fixed)))
    return specs


def second_peak_excluding(power: np.ndarray, excluded_bin: int) -> int:
    masked = np.asarray(power, dtype=np.float64).copy()
    masked[int(excluded_bin)] = -np.inf
    return int(np.argmax(masked))


def target_bin_for_spec(
    spec: CandidateSpec,
    gt_bin: int,
    argmax_bin: int,
    wrong_peak_bin: int,
    n_bins: int,
) -> int:
    if spec.kind == "gt":
        return int(gt_bin)
    if spec.kind == "argmax":
        return int(argmax_bin)
    if spec.kind == "wrong_peak":
        return int(wrong_peak_bin)
    if spec.kind == "offset":
        return positive_mod(int(gt_bin) + int(spec.value), n_bins)
    if spec.kind == "fixed":
        return positive_mod(int(spec.value), n_bins)
    raise ValueError(f"Unknown candidate kind: {spec.kind}")


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    rmse = float(math.sqrt(np.mean((y - pred) ** 2)))
    return coef, pred, r2, rmse


def quadratic_fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    coef = np.polyfit(x, y, deg=2)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def compute_feature_rows(
    samples: np.ndarray,
    symbols: list[PayloadGt],
    target_snr_db: float,
    file_name: str,
    cfo_correction_mode: str,
    specs_by_bins: dict[int, list[CandidateSpec]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    downchirps: dict[tuple[int, int, float], np.ndarray] = {}
    for symbol in symbols:
        spectrum, cfo_common_phase_rad = corrected_spectrum(
            samples=samples,
            symbol=symbol,
            downchirps=downchirps,
            cfo_correction_mode=cfo_correction_mode,
        )
        n_bins = int(spectrum.size)
        gt_bin = int(symbol.gt_raw_fft_bin)
        if not (0 <= gt_bin < n_bins):
            raise ValueError(f"GT bin out of range: {gt_bin}")
        power = np.abs(spectrum) ** 2
        total_power = float(np.sum(power, dtype=np.float64))
        argmax_bin = int(np.argmax(power))
        wrong_peak_bin = second_peak_excluding(power, gt_bin)
        gt_power = float(power[gt_bin])
        gt_amp = float(abs(complex(spectrum[gt_bin])))
        argmax_power = float(power[argmax_bin])
        specs = specs_by_bins[n_bins]

        for spec in specs:
            target_bin = target_bin_for_spec(
                spec,
                gt_bin=gt_bin,
                argmax_bin=argmax_bin,
                wrong_peak_bin=wrong_peak_bin,
                n_bins=n_bins,
            )
            peak = complex(spectrum[target_bin])
            target_power = float(power[target_bin])
            target_amp = float(abs(peak))
            target_rank = int(1 + np.sum(power > target_power))
            rows.append(
                {
                    "file_name": file_name,
                    "target_snr_db": float(target_snr_db),
                    "cfo_correction_mode": cfo_correction_mode,
                    "frame_index": int(symbol.frame_index),
                    "packet_index": int(symbol.packet_index),
                    "event_index": int(symbol.event_index),
                    "payload_symbol_index": int(symbol.payload_symbol_index),
                    "frame_symbol_index": int(symbol.frame_symbol_index),
                    "start_sample": int(symbol.start_sample),
                    "header_start_sample": int(symbol.header_start_sample),
                    "sf": int(symbol.sf),
                    "os_factor": int(symbol.os_factor),
                    "cfo_int": int(symbol.cfo_int),
                    "cfo_frac": float(symbol.cfo_frac),
                    "cfo_common_phase_rad": float(cfo_common_phase_rad),
                    "sto_frac": float(symbol.sto_frac),
                    "sfo_hat": float(symbol.sfo_hat),
                    "candidate_label": spec.label,
                    "candidate_kind": spec.kind,
                    "candidate_value": int(spec.value),
                    "gt_raw_fft_bin": gt_bin,
                    "gt_signed_fft_bin": signed_fft_bin(gt_bin, n_bins),
                    "argmax_raw_fft_bin": argmax_bin,
                    "wrong_peak_raw_fft_bin": wrong_peak_bin,
                    "target_raw_fft_bin": int(target_bin),
                    "target_signed_fft_bin": signed_fft_bin(target_bin, n_bins),
                    "target_is_gt_bin": int(target_bin == gt_bin),
                    "target_is_argmax_bin": int(target_bin == argmax_bin),
                    "argmax_is_gt_bin": int(argmax_bin == gt_bin),
                    "target_rank": target_rank,
                    "target_real": float(peak.real),
                    "target_imag": float(peak.imag),
                    "target_amp": target_amp,
                    "target_power": target_power,
                    "target_phase": float(math.atan2(peak.imag, peak.real)),
                    "target_energy_ratio": float(target_power / total_power) if total_power > 0.0 else float("nan"),
                    "gt_amp": gt_amp,
                    "gt_power": gt_power,
                    "gt_energy_ratio": float(gt_power / total_power) if total_power > 0.0 else float("nan"),
                    "argmax_power": argmax_power,
                    "target_amp_vs_gt_db": float(20.0 * math.log10((target_amp + 1e-30) / (gt_amp + 1e-30))),
                    "target_power_vs_gt_db": float(10.0 * math.log10((target_power + 1e-30) / (gt_power + 1e-30))),
                    "target_power_vs_argmax_db": float(10.0 * math.log10((target_power + 1e-30) / (argmax_power + 1e-30))),
                    "total_fft_energy": total_power,
                }
            )
    add_phase_columns(rows)
    return rows


def add_phase_columns(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["target_snr_db"]), int(row["packet_index"]), str(row["candidate_label"]))].append(row)

    for items in grouped.values():
        items.sort(key=lambda item: int(item["payload_symbol_index"]))
        k = np.asarray([int(item["payload_symbol_index"]) for item in items], dtype=np.float64)
        phase = np.asarray([float(item["target_phase"]) for item in items], dtype=np.float64)
        unwrap = np.unwrap(phase)
        coef, fit, r2, rmse = linear_fit(k, unwrap)
        residual = np.angle(np.exp(1j * (unwrap - fit)))
        quad_r2 = quadratic_fit_r2(k, residual)
        for index, item in enumerate(items):
            item["target_phase_unwrap"] = float(unwrap[index])
            item["phase_linear_fit"] = float(fit[index])
            item["phase_residual"] = float(residual[index])
            item["group_phase_slope_pi_per_symbol"] = float(coef[0] / math.pi)
            item["group_linear_fit_r2"] = float(r2)
            item["group_linear_fit_rmse_pi"] = float(rmse / math.pi)
            item["group_residual_quad_r2"] = float(quad_r2)


def summarize(feature_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        grouped[(float(row["target_snr_db"]), int(row["packet_index"]), str(row["candidate_label"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (target_snr_db, packet_index, label), items in sorted(grouped.items()):
        items.sort(key=lambda item: int(item["payload_symbol_index"]))
        first = items[0]
        amp = np.asarray([float(item["target_amp"]) for item in items], dtype=np.float64)
        er = np.asarray([float(item["target_energy_ratio"]) for item in items], dtype=np.float64)
        residual = np.asarray([float(item["phase_residual"]) for item in items], dtype=np.float64)
        ranks = np.asarray([float(item["target_rank"]) for item in items], dtype=np.float64)
        hit_gt = np.asarray([int(item["target_is_gt_bin"]) for item in items], dtype=np.float64)
        hit_argmax = np.asarray([int(item["target_is_argmax_bin"]) for item in items], dtype=np.float64)
        argmax_gt = np.asarray([int(item["argmax_is_gt_bin"]) for item in items], dtype=np.float64)
        summary.append(
            {
                "target_snr_db": float(target_snr_db),
                "packet_index": int(packet_index),
                "event_index": int(first["event_index"]),
                "frame_index": int(first["frame_index"]),
                "candidate_label": label,
                "candidate_kind": first["candidate_kind"],
                "candidate_value": int(first["candidate_value"]),
                "payload_symbol_count": len(items),
                "target_gt_hit_rate": float(np.mean(hit_gt)),
                "target_argmax_hit_rate": float(np.mean(hit_argmax)),
                "argmax_gt_hit_rate": float(np.mean(argmax_gt)),
                "phase_slope_pi_per_symbol": float(first["group_phase_slope_pi_per_symbol"]),
                "linear_fit_r2": float(first["group_linear_fit_r2"]),
                "linear_fit_rmse_pi": float(first["group_linear_fit_rmse_pi"]),
                "residual_std_pi": float(np.std(residual) / math.pi),
                "residual_peak_to_peak_pi": float((np.max(residual) - np.min(residual)) / math.pi),
                "residual_quad_r2": float(first["group_residual_quad_r2"]),
                "target_amp_mean": float(np.mean(amp)),
                "target_amp_std": float(np.std(amp)),
                "target_amp_cv": float(np.std(amp) / np.mean(amp)) if np.mean(amp) > 0.0 else float("nan"),
                "target_energy_ratio_mean": float(np.mean(er)),
                "target_energy_ratio_std": float(np.std(er)),
                "target_rank_mean": float(np.mean(ranks)),
                "target_rank_median": float(np.median(ranks)),
                "target_rank_min": int(np.min(ranks)),
                "target_rank_max": int(np.max(ranks)),
                "target_top8_hit_rate": float(np.mean(ranks <= 8)),
                "target_top16_hit_rate": float(np.mean(ranks <= 16)),
                "target_top32_hit_rate": float(np.mean(ranks <= 32)),
                "target_amp_vs_gt_db_mean": finite_mean(float(item["target_amp_vs_gt_db"]) for item in items),
                "target_power_vs_gt_db_mean": finite_mean(float(item["target_power_vs_gt_db"]) for item in items),
                "target_power_vs_argmax_db_mean": finite_mean(float(item["target_power_vs_argmax_db"]) for item in items),
                "gt_energy_ratio_mean": finite_mean(float(item["gt_energy_ratio"]) for item in items),
                "cfo_frac": float(first["cfo_frac"]),
                "sto_frac": float(first["sto_frac"]),
                "sfo_hat": float(first["sfo_hat"]),
            }
        )
    return summary


def choose_plot_labels(summary_rows: list[dict[str, Any]], target_snr_db: float, packet_index: int, wrong_count: int) -> list[str]:
    packet_items = [
        item
        for item in summary_rows
        if float(item["target_snr_db"]) == float(target_snr_db) and int(item["packet_index"]) == int(packet_index)
    ]
    labels = ["gt"]
    if any(str(item["candidate_label"]) == "wrong_peak" for item in packet_items):
        labels.append("wrong_peak")
    if any(str(item["candidate_label"]) == "argmax" for item in packet_items):
        labels.append("argmax")

    wrong = [
        item
        for item in packet_items
        if str(item["candidate_label"]) not in set(labels) and float(item["target_gt_hit_rate"]) == 0.0
    ]
    wrong.sort(
        key=lambda item: (
            float(item["residual_quad_r2"]),
            float(item["linear_fit_r2"]),
            float(item["target_energy_ratio_mean"]),
        ),
        reverse=True,
    )
    labels.extend(str(item["candidate_label"]) for item in wrong[: int(wrong_count)])
    return labels


def plot_packet(
    feature_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    target_snr_db: float,
    packet_index: int,
    labels: list[str],
    out_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        if float(row["target_snr_db"]) != float(target_snr_db):
            continue
        if int(row["packet_index"]) != int(packet_index):
            continue
        label = str(row["candidate_label"])
        if label in labels:
            rows_by_label[label].append(row)
    for items in rows_by_label.values():
        items.sort(key=lambda item: int(item["payload_symbol_index"]))

    summary_by_label = {
        str(item["candidate_label"]): item
        for item in summary_rows
        if float(item["target_snr_db"]) == float(target_snr_db) and int(item["packet_index"]) == int(packet_index)
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6), dpi=int(dpi))
    axes = axes.ravel()
    cmap = plt.get_cmap("tab10")
    color_by_label = {"gt": "black", "wrong_peak": "#d62728", "argmax": "#ff7f0e"}

    for index, label in enumerate(labels):
        items = rows_by_label.get(label, [])
        if not items:
            continue
        summary = summary_by_label[label]
        k = np.asarray([int(item["payload_symbol_index"]) for item in items], dtype=np.float64)
        residual = np.asarray([float(item["phase_residual"]) for item in items], dtype=np.float64)
        unwrap = np.asarray([float(item["target_phase_unwrap"]) for item in items], dtype=np.float64)
        amp = np.asarray([float(item["target_amp"]) for item in items], dtype=np.float64)
        er = np.asarray([float(item["target_energy_ratio"]) for item in items], dtype=np.float64)
        rank = np.asarray([float(item["target_rank"]) for item in items], dtype=np.float64)
        color = color_by_label.get(label, cmap(index % 10))
        linewidth = 2.35 if label == "gt" else 1.2
        alpha = 0.95 if label in {"gt", "wrong_peak"} else 0.72
        legend = (
            f"{label} qR2={float(summary['residual_quad_r2']):.2f} "
            f"ER={float(summary['target_energy_ratio_mean']):.2e} "
            f"rank~{float(summary['target_rank_median']):.0f}"
        )
        axes[0].plot(k, residual / math.pi, "o-", markersize=2.8, linewidth=linewidth, alpha=alpha, color=color, label=legend)
        axes[1].plot(k, amp, "o-", markersize=2.8, linewidth=linewidth, alpha=alpha, color=color, label=legend)
        axes[2].plot(k, np.maximum(er, 1e-18), "o-", markersize=2.8, linewidth=linewidth, alpha=alpha, color=color, label=legend)
        axes[3].plot(k, rank, "o-", markersize=2.8, linewidth=linewidth, alpha=alpha, color=color, label=legend)

    axes[0].axhline(0.0, color="black", linewidth=0.75)
    axes[0].set_title("phase residual after linear detrend")
    axes[0].set_ylabel("residual / pi")

    axes[1].set_title("target-bin amplitude")
    axes[1].set_ylabel("amplitude")

    axes[2].set_title("target-bin energy ratio")
    axes[2].set_xlabel("payload symbol index")
    axes[2].set_ylabel("target_power / total_fft_energy")
    axes[2].set_yscale("log")

    axes[3].set_title("target-bin rank, lower is better")
    axes[3].set_xlabel("payload symbol index")
    axes[3].set_ylabel("rank")
    axes[3].set_yscale("log")

    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.6)
    axes[0].legend(loc="best", fontsize=7)

    first_summary = next(
        item
        for item in summary_rows
        if float(item["target_snr_db"]) == float(target_snr_db) and int(item["packet_index"]) == int(packet_index)
    )
    fig.suptitle(
        f"Packet {packet_index} low-SNR wrong-bin control | event {first_summary['event_index']} "
        f"| SNR={target_snr_db:.1f} dB",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


FEATURE_FIELDS = [
    "file_name",
    "target_snr_db",
    "cfo_correction_mode",
    "frame_index",
    "packet_index",
    "event_index",
    "payload_symbol_index",
    "frame_symbol_index",
    "start_sample",
    "header_start_sample",
    "sf",
    "os_factor",
    "cfo_int",
    "cfo_frac",
    "cfo_common_phase_rad",
    "sto_frac",
    "sfo_hat",
    "candidate_label",
    "candidate_kind",
    "candidate_value",
    "gt_raw_fft_bin",
    "gt_signed_fft_bin",
    "argmax_raw_fft_bin",
    "wrong_peak_raw_fft_bin",
    "target_raw_fft_bin",
    "target_signed_fft_bin",
    "target_is_gt_bin",
    "target_is_argmax_bin",
    "argmax_is_gt_bin",
    "target_rank",
    "target_real",
    "target_imag",
    "target_amp",
    "target_power",
    "target_phase",
    "target_phase_unwrap",
    "phase_linear_fit",
    "phase_residual",
    "target_energy_ratio",
    "gt_amp",
    "gt_power",
    "gt_energy_ratio",
    "argmax_power",
    "target_amp_vs_gt_db",
    "target_power_vs_gt_db",
    "target_power_vs_argmax_db",
    "total_fft_energy",
    "group_phase_slope_pi_per_symbol",
    "group_linear_fit_r2",
    "group_linear_fit_rmse_pi",
    "group_residual_quad_r2",
]


SUMMARY_FIELDS = [
    "target_snr_db",
    "packet_index",
    "event_index",
    "frame_index",
    "candidate_label",
    "candidate_kind",
    "candidate_value",
    "payload_symbol_count",
    "target_gt_hit_rate",
    "target_argmax_hit_rate",
    "argmax_gt_hit_rate",
    "phase_slope_pi_per_symbol",
    "linear_fit_r2",
    "linear_fit_rmse_pi",
    "residual_std_pi",
    "residual_peak_to_peak_pi",
    "residual_quad_r2",
    "target_amp_mean",
    "target_amp_std",
    "target_amp_cv",
    "target_energy_ratio_mean",
    "target_energy_ratio_std",
    "target_rank_mean",
    "target_rank_median",
    "target_rank_min",
    "target_rank_max",
    "target_top8_hit_rate",
    "target_top16_hit_rate",
    "target_top32_hit_rate",
    "target_amp_vs_gt_db_mean",
    "target_power_vs_gt_db_mean",
    "target_power_vs_argmax_db_mean",
    "gt_energy_ratio_mean",
    "cfo_frac",
    "sto_frac",
    "sfo_hat",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    low_snr_dir = args.low_snr_dir.resolve()
    gt_csv = args.gt_symbol_csv.resolve()
    input_stem = args.input_stem or _infer_input_stem(gt_csv)
    output_dir = (args.output_dir or (low_snr_dir / "wrong_bin_control")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    packet_filter = set(args.packet) if args.packet is not None else None
    symbols = load_payload_gt(gt_csv, packet_filter=packet_filter)
    n_bins_values = sorted({1 << int(symbol.sf) for symbol in symbols})
    specs_by_bins = {
        n_bins: build_candidate_specs(
            n_bins=n_bins,
            offset_bins=args.offset_bins,
            fixed_bins=args.fixed_bins,
            fixed_fractions=args.fixed_fractions,
        )
        for n_bins in n_bins_values
    }

    noisy_bins = find_noisy_bins(low_snr_dir=low_snr_dir, input_stem=input_stem, snr_filter=args.snr_db)
    all_rows: list[dict[str, Any]] = []
    for snr_db, path in noisy_bins:
        samples = np.fromfile(path, dtype=np.complex64)
        if samples.size == 0:
            raise ValueError(f"Empty noisy IQ file: {path}")
        rows = compute_feature_rows(
            samples=samples,
            symbols=symbols,
            target_snr_db=snr_db,
            file_name=str(path),
            cfo_correction_mode=args.cfo_correction_mode,
            specs_by_bins=specs_by_bins,
        )
        all_rows.extend(rows)
        label = _snr_label(snr_db)
        write_csv(output_dir / f"{input_stem}_{label}_wrong_bin_features.csv", rows, FEATURE_FIELDS)
        print(f"[{label}] symbols={len(symbols)}, feature_rows={len(rows)}")

    summary_rows = summarize(all_rows)
    write_csv(output_dir / f"{input_stem}_low_snr_wrong_bin_features_all.csv", all_rows, FEATURE_FIELDS)
    write_csv(output_dir / f"{input_stem}_low_snr_wrong_bin_summary.csv", summary_rows, SUMMARY_FIELDS)

    plot_count = 0
    packet_indices = sorted({symbol.packet_index for symbol in symbols})
    for snr_db, _ in noisy_bins:
        label = _snr_label(snr_db)
        for packet_index in packet_indices:
            labels = choose_plot_labels(summary_rows, target_snr_db=snr_db, packet_index=packet_index, wrong_count=args.plot_wrong_count)
            out_path = output_dir / "plots" / label / f"packet_{packet_index:03d}_wrong_bin_phase_amp_control.png"
            plot_packet(
                feature_rows=all_rows,
                summary_rows=summary_rows,
                target_snr_db=snr_db,
                packet_index=packet_index,
                labels=labels,
                out_path=out_path,
                dpi=args.dpi,
            )
            plot_count += 1

    print(f"summary={output_dir / f'{input_stem}_low_snr_wrong_bin_summary.csv'}")
    print(f"features={output_dir / f'{input_stem}_low_snr_wrong_bin_features_all.csv'}")
    print(f"plots={output_dir / 'plots'}")
    print(f"wrote_pngs={plot_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
