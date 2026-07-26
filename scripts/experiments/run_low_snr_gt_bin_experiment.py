#!/usr/bin/env python3
"""低信噪比 GT-bin 相位/幅度特征实验。

这个脚本只做“非解码链”的消融验证：给 clean IQ 人为加入复高斯白噪声，
然后使用 clean header-first symbol CSV 里已经确认的 payload raw_fft_bin
作为 GT bin，从 noisy IQ 的 corrected FFT 结果里强行读取该 bin 的复数值。

注意：这里不重新做弱检测、不重新做 framesync，也不把低 SNR 下的 argmax
当真值；目标是观察正确 bin 在低 SNR 下的相位/幅度轨迹是否仍有结构。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np


# 当前文件位于 weakPacket_decoding/scripts/experiments/，所以 parents[2]
# 才是 weakPacket_decoding 根目录。加入 sys.path 后可直接导入 weak_decoder。
WEAK_ROOT = Path(__file__).resolve().parents[2]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, dechirp_fft, signed_fft_bin  # noqa: E402


@dataclass(frozen=True)
class GtPayloadSymbol:
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
    clean_peak_amp: float
    clean_peak_phase: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "给 clean IQ 加 AWGN，并用 clean header-first CSV 的 payload raw_fft_bin "
            "作为 GT bin，导出低 SNR 下的相位/幅度特征。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="clean complex64 IQ .bin 文件。")
    parser.add_argument(
        "-g",
        "--gt-symbol-csv",
        type=Path,
        required=True,
        help="clean header-first symbol CSV；其中 header_valid=1 的 payload 行提供 GT raw_fft_bin。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "low_snr_gt_bin",
        help="输出目录：保存 noisy IQ、feature CSV、summary 和 plots。",
    )
    parser.add_argument(
        "--target-snr-db",
        type=float,
        nargs="+",
        default=[-10.0, -15.0, -20.0],
        help=(
            "目标加噪 SNR，单位 dB。SNR 相对于 clean payload symbol 采样功率定义；"
            "默认：-10 -15 -20。"
        ),
    )
    parser.add_argument(
        "--cfo-correction-mode",
        choices=("symbol", "continuous"),
        default="continuous",
        help="读取 GT bin 前使用的 FFT CFO 补偿模式，默认 continuous。",
    )
    parser.add_argument("--packet", type=int, default=None, help="可选：只处理指定 packet_index。")
    parser.add_argument(
        "--seed",
        type=int,
        default=20260531,
        help="随机种子基准；independent-noise 模式下每个 SNR 使用 seed + step_index。",
    )
    parser.add_argument(
        "--independent-noise",
        action="store_true",
        default=False,
        help="每个 SNR 档使用不同的单位噪声 realization。",
    )
    parser.add_argument(
        "--no-write-noisy-bin",
        action="store_true",
        default=False,
        help="不保存 noisy IQ .bin，只导出特征 CSV 和图。",
    )
    parser.add_argument("--overwrite", action="store_true", default=False, help="允许覆盖已有输出。")
    parser.add_argument("--no-plots", action="store_true", default=False, help="Skip diagnostic PNG generation.")
    parser.add_argument("--dpi", type=int, default=220, help="PNG DPI，默认 220。")
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


def _snr_label(snr_db: float) -> str:
    sign = "m" if float(snr_db) < 0 else "p"
    value = abs(float(snr_db))
    if abs(value - round(value)) < 1e-9:
        text = f"{int(round(value)):02d}"
    else:
        text = f"{value:.1f}".replace(".", "p")
    return f"snr_{sign}{text}dB"


def _group_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (_int(row, "frame_index", -1), _int(row, "packet_index", -1), _int(row, "event_index", -1))


def _symbol_indexes(start_sample: int, sf: int, os_factor: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    return int(start_sample) + int(os_value / 2) + os_value * np.arange(n_bins, dtype=np.int64)


def load_gt_payload_symbols(path: Path, packet_filter: int | None) -> list[GtPayloadSymbol]:
    """从 clean header-first symbol CSV 中读取可作为 GT 的 payload 符号。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    header_start_by_key: dict[tuple[int, int, int], int] = {}
    for row in rows:
        if row.get("stage") != "header":
            continue
        if _int(row, "stage_symbol_index", -1) != 0:
            continue
        header_start_by_key[_group_key(row)] = _int(row, "start_sample")

    payload_symbols: list[GtPayloadSymbol] = []
    for row in rows:
        if row.get("stage") != "payload":
            continue
        if _int(row, "header_valid", 0) != 1:
            continue
        packet_index = _int(row, "packet_index", -1)
        if packet_filter is not None and packet_index != int(packet_filter):
            continue

        key = _group_key(row)
        header_start_sample = header_start_by_key.get(key)
        if header_start_sample is None:
            frame_symbol_index = _int(row, "frame_symbol_index")
            samples_per_symbol = (1 << _int(row, "sf")) * _int(row, "os_factor")
            header_start_sample = _int(row, "start_sample") - frame_symbol_index * samples_per_symbol

        payload_symbols.append(
            GtPayloadSymbol(
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
                clean_peak_amp=_float(row, "peak_amp"),
                clean_peak_phase=_float(row, "peak_phase"),
            )
        )

    payload_symbols.sort(key=lambda item: (item.packet_index, item.payload_symbol_index))
    if not payload_symbols:
        raise ValueError("No GT payload rows found. Need stage=payload and header_valid=1.")
    return payload_symbols


def estimate_payload_reference_power(samples: np.ndarray, symbols: list[GtPayloadSymbol]) -> float:
    """用 GT payload 符号采样点估计信号参考功率，用于定义目标 SNR。"""
    total_power = 0.0
    total_count = 0
    for item in symbols:
        indexes = _symbol_indexes(item.start_sample, item.sf, item.os_factor)
        if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
            raise ValueError(f"GT symbol exceeds IQ range at start_sample={item.start_sample}.")
        values = samples[indexes]
        total_power += float(np.sum(np.abs(values) ** 2, dtype=np.float64))
        total_count += int(values.size)
    if total_count <= 0:
        raise ValueError("No samples available for reference-power estimation.")
    reference_power = total_power / float(total_count)
    if not np.isfinite(reference_power) or reference_power <= 0.0:
        raise ValueError(f"Invalid reference power: {reference_power}")
    return float(reference_power)


def add_awgn(samples: np.ndarray, noise_power: float, seed: int) -> np.ndarray:
    """给整段 complex64 IQ 加复高斯白噪声。"""
    rng = np.random.default_rng(int(seed))
    sigma = math.sqrt(float(noise_power) / 2.0)
    noise_i = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
    noise_q = rng.normal(0.0, sigma, size=samples.size).astype(np.float32)
    return (samples + (noise_i + 1j * noise_q)).astype(np.complex64, copy=False)


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    rmse = float(math.sqrt(np.mean((y - pred) ** 2)))
    return coef, pred, r2, rmse


def _quadratic_fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    coef = np.polyfit(x, y, deg=2)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")


def export_features_for_snr(
    noisy_samples: np.ndarray,
    symbols: list[GtPayloadSymbol],
    target_snr_db: float,
    signal_reference_power: float,
    added_noise_power: float,
    seed: int,
    file_name: str,
    cfo_correction_mode: str,
) -> list[dict[str, Any]]:
    """在某一档 SNR 下重算 FFT，并强行读取每个 payload 符号的 GT bin。"""
    downchirps: dict[tuple[int, int, float], np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for item in symbols:
        n_bins = 1 << int(item.sf)
        if not (0 <= int(item.gt_raw_fft_bin) < n_bins):
            raise ValueError(f"GT bin out of range: {item.gt_raw_fft_bin}")

        indexes = _symbol_indexes(item.start_sample, item.sf, item.os_factor)
        symbol = np.asarray(noisy_samples[indexes], dtype=np.complex64)
        cfo_total = float(item.cfo_int) + float(item.cfo_frac)
        if cfo_correction_mode == "continuous":
            relative_chip_start = float(item.start_sample - item.header_start_sample) / float(item.os_factor)
            cfo_common_phase_rad = float(2.0 * math.pi * cfo_total * relative_chip_start / n_bins)
            symbol = (symbol * np.exp(-1j * cfo_common_phase_rad)).astype(np.complex64)
        else:
            cfo_common_phase_rad = 0.0

        downchirp_key = (int(item.sf), int(item.cfo_int), float(item.cfo_frac))
        downchirp = downchirps.get(downchirp_key)
        if downchirp is None:
            downchirp = build_downchirp(item.sf, cfo_int=item.cfo_int, cfo_frac=item.cfo_frac)
            downchirps[downchirp_key] = downchirp

        spectrum = dechirp_fft(symbol, downchirp)
        power = np.abs(spectrum) ** 2
        noisy_argmax_bin = int(np.argmax(power))
        gt_bin = int(item.gt_raw_fft_bin)
        gt_value = complex(spectrum[gt_bin])
        gt_power = float(power[gt_bin])
        total_fft_energy = float(np.sum(power, dtype=np.float64))
        sorted_desc = np.sort(power)[::-1]
        second_power = float(sorted_desc[1]) if sorted_desc.size > 1 else 0.0
        gt_rank = int(np.where(np.argsort(power)[::-1] == gt_bin)[0][0] + 1)
        selected_value = complex(spectrum[noisy_argmax_bin])
        selected_power = float(power[noisy_argmax_bin])

        rows.append(
            {
                "file_name": file_name,
                "target_snr_db": float(target_snr_db),
                "signal_reference_power": float(signal_reference_power),
                "added_noise_power": float(added_noise_power),
                "seed": int(seed),
                "cfo_correction_mode": cfo_correction_mode,
                "frame_index": int(item.frame_index),
                "packet_index": int(item.packet_index),
                "event_index": int(item.event_index),
                "payload_symbol_index": int(item.payload_symbol_index),
                "frame_symbol_index": int(item.frame_symbol_index),
                "start_sample": int(item.start_sample),
                "header_start_sample": int(item.header_start_sample),
                "sf": int(item.sf),
                "os_factor": int(item.os_factor),
                "cfo_int": int(item.cfo_int),
                "cfo_frac": float(item.cfo_frac),
                "cfo_common_phase_rad": float(cfo_common_phase_rad),
                "sto_frac": float(item.sto_frac),
                "sfo_hat": float(item.sfo_hat),
                "gt_raw_fft_bin": gt_bin,
                "gt_signed_fft_bin": signed_fft_bin(gt_bin, n_bins),
                "noisy_argmax_bin": noisy_argmax_bin,
                "is_argmax_correct": int(noisy_argmax_bin == gt_bin),
                "gt_bin_rank": gt_rank,
                "gt_bin_real": float(gt_value.real),
                "gt_bin_imag": float(gt_value.imag),
                "gt_bin_amp": float(abs(gt_value)),
                "gt_bin_power": gt_power,
                "gt_bin_phase": float(math.atan2(gt_value.imag, gt_value.real)),
                "gt_peak_energy_ratio": float(gt_power / total_fft_energy) if total_fft_energy > 0.0 else float("nan"),
                "gt_to_argmax_power_db": float(10.0 * math.log10((gt_power + 1e-30) / (selected_power + 1e-30))),
                "selected_peak_amp": float(abs(selected_value)),
                "selected_peak_power": selected_power,
                "selected_peak_phase": float(math.atan2(selected_value.imag, selected_value.real)),
                "selected_peak_margin_db": float(
                    10.0 * math.log10((selected_power + 1e-30) / (second_power + 1e-30))
                ),
                "total_fft_energy": total_fft_energy,
                "clean_gt_peak_amp": float(item.clean_peak_amp),
                "clean_gt_peak_phase": float(item.clean_peak_phase),
            }
        )

    add_packet_phase_columns(rows)
    return rows


def add_packet_phase_columns(rows: list[dict[str, Any]]) -> None:
    """按 packet 对 GT-bin phase 做 unwrap、线性拟合和 residual 诊断。"""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["packet_index"])].append(row)
    for packet_rows in grouped.values():
        packet_rows.sort(key=lambda item: int(item["payload_symbol_index"]))
        k = np.asarray([int(row["payload_symbol_index"]) for row in packet_rows], dtype=np.float64)
        phase = np.asarray([float(row["gt_bin_phase"]) for row in packet_rows], dtype=np.float64)
        phase_unwrap = np.unwrap(phase)
        fit_coef, fit_line, fit_r2, fit_rmse = _linear_fit(k, phase_unwrap)
        residual = np.angle(np.exp(1j * (phase_unwrap - fit_line)))
        quad_r2 = _quadratic_fit_r2(k, residual)
        for index, row in enumerate(packet_rows):
            row["gt_bin_phase_unwrap"] = float(phase_unwrap[index])
            row["phase_linear_fit"] = float(fit_line[index])
            row["phase_linear_residual"] = float(residual[index])
            row["packet_phase_slope_pi_per_symbol"] = float(fit_coef[0] / math.pi)
            row["packet_phase_linear_r2"] = float(fit_r2)
            row["packet_phase_linear_rmse_pi"] = float(fit_rmse / math.pi)
            row["packet_residual_quad_r2"] = float(quad_r2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["target_snr_db"]), int(row["packet_index"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (target_snr_db, packet_index), packet_rows in sorted(grouped.items()):
        packet_rows.sort(key=lambda item: int(item["payload_symbol_index"]))
        event_index = int(packet_rows[0]["event_index"])
        frame_index = int(packet_rows[0]["frame_index"])
        argmax_correct = np.asarray([int(row["is_argmax_correct"]) for row in packet_rows], dtype=np.float64)
        gt_amp = np.asarray([float(row["gt_bin_amp"]) for row in packet_rows], dtype=np.float64)
        gt_er = np.asarray([float(row["gt_peak_energy_ratio"]) for row in packet_rows], dtype=np.float64)
        gt_rank = np.asarray([int(row["gt_bin_rank"]) for row in packet_rows], dtype=np.float64)
        residual = np.asarray([float(row["phase_linear_residual"]) for row in packet_rows], dtype=np.float64)
        summary.append(
            {
                "target_snr_db": float(target_snr_db),
                "packet_index": int(packet_index),
                "event_index": event_index,
                "frame_index": frame_index,
                "payload_symbol_count": len(packet_rows),
                "argmax_correct_rate": float(np.mean(argmax_correct)),
                "gt_bin_amp_mean": float(np.mean(gt_amp)),
                "gt_bin_amp_std": float(np.std(gt_amp)),
                "gt_peak_energy_ratio_mean": float(np.mean(gt_er)),
                "gt_peak_energy_ratio_std": float(np.std(gt_er)),
                "phase_slope_pi_per_symbol": float(packet_rows[0]["packet_phase_slope_pi_per_symbol"]),
                "phase_linear_r2": float(packet_rows[0]["packet_phase_linear_r2"]),
                "phase_linear_rmse_pi": float(packet_rows[0]["packet_phase_linear_rmse_pi"]),
                "phase_residual_std_pi": float(np.std(residual) / math.pi),
                "phase_residual_peak_to_peak_pi": float((np.max(residual) - np.min(residual)) / math.pi),
                "phase_residual_quad_r2": float(packet_rows[0]["packet_residual_quad_r2"]),
                "mean_gt_bin_rank": float(np.mean(gt_rank)),
                "gt_top8_recall": float(np.mean(gt_rank <= 8)),
                "gt_top16_recall": float(np.mean(gt_rank <= 16)),
                "gt_top32_recall": float(np.mean(gt_rank <= 32)),
            }
        )
    return summary


def plot_packet_diagnostics(packet_rows: list[dict[str, Any]], out_path: Path, dpi: int) -> None:
    """绘制单个 packet 的 GT-bin wrapped/unwrap/residual/幅度能量四联图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    packet_rows = sorted(packet_rows, key=lambda item: int(item["payload_symbol_index"]))
    k = np.asarray([int(row["payload_symbol_index"]) for row in packet_rows], dtype=np.float64)
    phase = np.asarray([float(row["gt_bin_phase"]) for row in packet_rows], dtype=np.float64)
    phase_unwrap = np.asarray([float(row["gt_bin_phase_unwrap"]) for row in packet_rows], dtype=np.float64)
    phase_fit = np.asarray([float(row["phase_linear_fit"]) for row in packet_rows], dtype=np.float64)
    residual = np.asarray([float(row["phase_linear_residual"]) for row in packet_rows], dtype=np.float64)
    amp = np.asarray([float(row["gt_bin_amp"]) for row in packet_rows], dtype=np.float64)
    energy_ratio = np.asarray([float(row["gt_peak_energy_ratio"]) for row in packet_rows], dtype=np.float64)
    argmax_correct = np.asarray([int(row["is_argmax_correct"]) for row in packet_rows], dtype=np.int32)

    first = packet_rows[0]
    packet_index = int(first["packet_index"])
    event_index = int(first["event_index"])
    target_snr_db = float(first["target_snr_db"])
    linear_r2 = float(first["packet_phase_linear_r2"])
    quad_r2 = float(first["packet_residual_quad_r2"])
    slope = float(first["packet_phase_slope_pi_per_symbol"])
    argmax_rate = float(np.mean(argmax_correct))
    mean_er = float(np.mean(energy_ratio))

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.2), dpi=int(dpi))
    axes = axes.ravel()

    marker_style = {"marker": "o", "markersize": 3.0, "linewidth": 1.1}
    axes[0].plot(k, phase / math.pi, **marker_style)
    axes[0].set_title("GT-bin wrapped phase")
    axes[0].set_ylabel("phase / pi")
    axes[0].set_ylim(-1.05, 1.05)

    axes[1].plot(k, phase_unwrap / math.pi, **marker_style, label="unwrap")
    axes[1].plot(k, phase_fit / math.pi, "-", linewidth=2.0, label=f"linear fit, slope={slope:.3f} pi/sym")
    axes[1].set_title(f"GT-bin unwrap phase, R2={linear_r2:.4f}")
    axes[1].legend()

    axes[2].plot(k, residual / math.pi, **marker_style)
    axes[2].axhline(0.0, color="black", linewidth=0.75)
    axes[2].set_title(f"residual after linear detrend, quad R2={quad_r2:.4f}")
    axes[2].set_xlabel("payload symbol index")
    axes[2].set_ylabel("residual / pi")

    colors = np.where(argmax_correct == 1, "#1f77b4", "#d62728")
    axes[3].scatter(k, amp, c=colors, s=22, label="GT-bin amp")
    axes[3].plot(k, amp, color="#1f77b4", linewidth=0.85, alpha=0.55)
    ax_er = axes[3].twinx()
    ax_er.plot(k, energy_ratio, color="#ff7f0e", linewidth=1.0, alpha=0.75, label="energy ratio")
    axes[3].set_title(f"GT-bin amplitude / energy ratio, argmax acc={argmax_rate:.2f}")
    axes[3].set_xlabel("payload symbol index")
    axes[3].set_ylabel("amplitude")
    ax_er.set_ylabel("energy ratio")

    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.6)

    title = (
        f"Packet {packet_index} GT-bin low-SNR diagnostics | event {event_index} | "
        f"SNR={target_snr_db:.1f} dB | mean ER={mean_er:.3f}"
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_all(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> list[Path]:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(float(row["target_snr_db"]), int(row["packet_index"]))].append(row)

    paths: list[Path] = []
    for (target_snr_db, packet_index), packet_rows in sorted(grouped.items()):
        event_index = int(packet_rows[0]["event_index"])
        label = _snr_label(target_snr_db)
        out_path = (
            out_dir
            / "plots"
            / label
            / f"packet_{packet_index:03d}_event_{event_index:03d}_gt_bin_phase_amp_diagnostics.png"
        )
        plot_packet_diagnostics(packet_rows, out_path=out_path, dpi=dpi)
        paths.append(out_path)
    return paths


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    gt_csv = args.gt_symbol_csv.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = np.fromfile(input_path, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"Empty IQ file: {input_path}")

    gt_symbols = load_gt_payload_symbols(gt_csv, packet_filter=args.packet)
    signal_reference_power = estimate_payload_reference_power(samples, gt_symbols)

    metadata = {
        "input": str(input_path),
        "gt_symbol_csv": str(gt_csv),
        "payload_symbol_count": len(gt_symbols),
        "packet_count": len({item.packet_index for item in gt_symbols}),
        "signal_reference_power": signal_reference_power,
        "signal_reference_power_db": 10.0 * math.log10(signal_reference_power),
        "target_snr_db": [float(value) for value in args.target_snr_db],
        "cfo_correction_mode": args.cfo_correction_mode,
        "seed": int(args.seed),
        "independent_noise": bool(args.independent_noise),
    }
    (out_dir / f"{input_path.stem}_low_snr_gt_bin_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_rows: list[dict[str, Any]] = []
    unit_noise: np.ndarray | None = None
    base_seed = int(args.seed)
    if not args.independent_noise:
        rng = np.random.default_rng(base_seed)
        noise_i = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        noise_q = rng.normal(0.0, 1.0, size=samples.size).astype(np.float32)
        unit_noise = (noise_i + 1j * noise_q).astype(np.complex64)

    for step_index, snr_db in enumerate(args.target_snr_db):
        label = _snr_label(float(snr_db))
        noise_power = signal_reference_power * (10.0 ** (-float(snr_db) / 10.0))
        seed = base_seed + step_index
        if unit_noise is None:
            noisy_samples = add_awgn(samples, noise_power=noise_power, seed=seed)
        else:
            sigma = math.sqrt(float(noise_power) / 2.0)
            noisy_samples = (samples + sigma * unit_noise).astype(np.complex64, copy=False)

        noisy_bin_path = out_dir / f"{input_path.stem}_{label}.bin"
        if not args.no_write_noisy_bin:
            if noisy_bin_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing file: {noisy_bin_path}")
            noisy_samples.tofile(noisy_bin_path)

        feature_rows = export_features_for_snr(
            noisy_samples=noisy_samples,
            symbols=gt_symbols,
            target_snr_db=float(snr_db),
            signal_reference_power=signal_reference_power,
            added_noise_power=float(noise_power),
            seed=seed,
            file_name=str(noisy_bin_path if not args.no_write_noisy_bin else input_path),
            cfo_correction_mode=args.cfo_correction_mode,
        )
        feature_csv = out_dir / f"{input_path.stem}_{label}_gt_bin_features.csv"
        write_csv(feature_csv, feature_rows)
        all_rows.extend(feature_rows)
        print(
            f"[SNR {snr_db:.1f} dB] rows={len(feature_rows)}, "
            f"noise_power={noise_power:.6e}, features={feature_csv}"
        )

    all_feature_csv = out_dir / f"{input_path.stem}_low_snr_gt_bin_features_all.csv"
    write_csv(all_feature_csv, all_rows)
    summary_rows = summarize_rows(all_rows)
    summary_csv = out_dir / f"{input_path.stem}_low_snr_gt_bin_summary.csv"
    write_csv(summary_csv, summary_rows)
    plot_paths = [] if args.no_plots else plot_all(all_rows, out_dir=out_dir, dpi=args.dpi)

    print(f"signal_reference_power={signal_reference_power:.6e}")
    print(f"wrote_metadata={out_dir / f'{input_path.stem}_low_snr_gt_bin_metadata.json'}")
    print(f"wrote_all_features={all_feature_csv}")
    print(f"wrote_summary={summary_csv}")
    if args.no_plots:
        print("wrote_plots=0 (skipped)")
    else:
        print(f"wrote_plots={len(plot_paths)} under {out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
