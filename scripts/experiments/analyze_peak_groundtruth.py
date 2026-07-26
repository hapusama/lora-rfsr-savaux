#!/usr/bin/env python3
"""分析 peak groundtruth CSV 中 hard peak 的相位和幅度趋势。"""
# D:\mysoft2\miniconda3\envs\gr-lora\python.exe gr-lora_sdr\weakPacket_decoding\scripts\experiments\analyze_peak_groundtruth.py -i gr-lora_sdr\weakPacket_decoding\data\peak_groundtruth\0_0_0_10_14_8_peak_gt.csv

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 export_peak_groundtruth.py 导出的 CSV，绘制每包 hard peak 的相位/幅度趋势。"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="peak groundtruth CSV，例如 data/peak_groundtruth/xxx_peak_gt.csv。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录。默认在输入 CSV 同目录下创建 <stem>_plots。",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="只分析指定 frame_count。默认分析全部帧。",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="相位/幅度趋势的移动平均窗口，默认 1，也就是不平滑。",
    )
    parser.add_argument(
        "--payload-only",
        action="store_true",
        help="只画 payload/data 符号，跳过 symbol_index 0..7 的 header。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="输出 PNG 的 DPI，默认 220。",
    )
    return parser.parse_args()


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, "")))
    except ValueError:
        return default


def load_rows(path: Path, frame: int | None, payload_only: bool) -> list[dict[str, object]]:
    """读取 CSV，并抽取 hard peak/top1 peak 的核心观测量。"""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            frame_count = _int(raw, "frame_count", -1)
            is_header = bool(_int(raw, "is_header", 0))
            if frame is not None and frame_count != frame:
                continue
            if payload_only and is_header:
                continue
            power = _float(raw, "top1_power")
            rows.append(
                {
                    "input_file": raw.get("input_file", ""),
                    "frame_count": frame_count,
                    "symbol_index": _int(raw, "symbol_index", -1),
                    "is_header": is_header,
                    "hard_bin": _int(raw, "hard_bin", -1),
                    "hard_symbol": _int(raw, "hard_symbol", -1),
                    "phase": _float(raw, "top1_phase"),
                    "phase_pi": _float(raw, "top1_phase") / math.pi,
                    "power": power,
                    "amplitude": math.sqrt(power) if power >= 0.0 else float("nan"),
                    "confidence_db": _float(raw, "confidence_db"),
                    "cfo_int": _int(raw, "cfo_int", 0),
                    "cfo_frac": _float(raw, "cfo_frac", 0.0),
                }
            )
    rows.sort(key=lambda item: (int(item["frame_count"]), int(item["symbol_index"])))
    return rows


def group_by_frame(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["frame_count"])].append(row)
    return dict(sorted(grouped.items()))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均；窗口过小时直接返回原始值。"""
    window = int(window)
    if window <= 1 or values.size < 2:
        return values.copy()
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def circular_phase_average(phases: np.ndarray, window: int) -> np.ndarray:
    """对相位做圆周移动平均，避免 -pi/pi 跳变把趋势拉坏。"""
    window = int(window)
    if window <= 1 or phases.size < 2:
        return phases.copy()
    real = moving_average(np.cos(phases), window)
    imag = moving_average(np.sin(phases), window)
    return np.arctan2(imag, real)


def write_analysis_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """写出精简后的相位/幅度观测表。"""
    fields = [
        "frame_count",
        "symbol_index",
        "is_header",
        "hard_bin",
        "hard_symbol",
        "phase",
        "phase_pi",
        "amplitude",
        "power",
        "confidence_db",
        "cfo_int",
        "cfo_frac",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _plot_panel(draw, box, x_values, series, y_min, y_max, title, y_label, x_label="Symbol"):
    """用 Pillow 画一个简洁线图面板。"""
    from PIL import ImageFont

    left, top, right, bottom = box
    font = ImageFont.load_default()
    axis_color = (30, 30, 30)
    grid_color = (220, 220, 220)
    draw.rectangle([left, top, right, bottom], outline=axis_color, width=1)
    draw.text((left, top - 22), title, fill=axis_color, font=font)
    draw.text((left, bottom + 8), x_label, fill=axis_color, font=font)
    draw.text((left - 55, top + 8), y_label, fill=axis_color, font=font)

    if not x_values:
        return
    x_min = min(x_values)
    x_max = max(x_values)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0

    def map_x(value):
        return left + int(round((float(value) - x_min) / (x_max - x_min) * (right - left)))

    def map_y(value):
        return bottom - int(round((float(value) - y_min) / (y_max - y_min) * (bottom - top)))

    for tick in np.linspace(y_min, y_max, 5):
        y = map_y(tick)
        draw.line([(left, y), (right, y)], fill=grid_color, width=1)
        draw.text((left - 46, y - 6), f"{tick:.2g}", fill=(80, 80, 80), font=font)
    for tick in np.linspace(x_min, x_max, 5):
        x = map_x(tick)
        draw.line([(x, top), (x, bottom)], fill=grid_color, width=1)
        draw.text((x - 10, bottom + 8), f"{tick:.0f}", fill=(80, 80, 80), font=font)

    legend_y = top + 8
    for name, values, color, marker in series:
        points = [(map_x(x), map_y(y)) for x, y in zip(x_values, values) if np.isfinite(y)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=2 if marker is None else 1)
        if marker is not None:
            for px, py in points:
                draw.rectangle([px - 2, py - 2, px + 2, py + 2], outline=color, fill=(255, 255, 255))
        draw.line([(right - 120, legend_y + 5), (right - 95, legend_y + 5)], fill=color, width=2)
        draw.text((right - 90, legend_y), name, fill=axis_color, font=font)
        legend_y += 16


def plot_frame(frame_count: int, rows: list[dict[str, object]], out_path: Path, smooth_window: int, dpi: int) -> None:
    """绘制单个 packet 的 hard peak 相位和幅度趋势。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    del smooth_window
    x = np.asarray([int(row["symbol_index"]) for row in rows], dtype=np.float64)
    phase = np.asarray([float(row["phase"]) for row in rows], dtype=np.float64)
    phase_pi = phase / math.pi
    amplitude = np.asarray([float(row["amplitude"]) for row in rows], dtype=np.float64)
    is_header = np.asarray([bool(row["is_header"]) for row in rows], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=int(dpi))
    fig.suptitle(f"Frame {frame_count} hard FFT peak after frame_sync", fontsize=11)

    if np.any(is_header):
        boundary = float(np.max(x[is_header]) + 0.5)
    else:
        boundary = None

    axes[0].plot(
        x,
        phase_pi,
        color="#d95f02",
        marker="D",
        markersize=3.2,
        markerfacecolor="white",
        linewidth=0.9,
        label="phase/pi",
    )
    if boundary is not None:
        axes[0].axvline(boundary, color="#777777", linestyle="--", linewidth=0.8, label="header/data boundary")
    axes[0].set_title("Phase of selected FFT peak", fontsize=9)
    axes[0].set_xlabel("Symbol")
    axes[0].set_ylabel("Phase / pi")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, color="#d9d9d9", linewidth=0.5)
    axes[0].legend(fontsize=7, loc="best")

    axes[1].plot(
        x,
        amplitude,
        color="#d95f02",
        marker="D",
        markersize=3.2,
        markerfacecolor="white",
        linewidth=0.9,
        label="amplitude",
    )
    if boundary is not None:
        axes[1].axvline(boundary, color="#777777", linestyle="--", linewidth=0.8, label="header/data boundary")
    axes[1].set_title("Amplitude of selected FFT peak", fontsize=9)
    axes[1].set_xlabel("Symbol")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, color="#d9d9d9", linewidth=0.5)
    axes[1].legend(fontsize=7, loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(grouped: dict[int, list[dict[str, object]]], out_path: Path, dpi: int) -> None:
    """绘制所有 packet 的相位/幅度叠加图，用来快速看包间一致性。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=int(dpi))
    fig.suptitle("All frames hard FFT peak trends after frame_sync", fontsize=11)
    for frame_count, rows in grouped.items():
        x = np.asarray([int(row["symbol_index"]) for row in rows], dtype=np.float64)
        phase_pi = np.asarray([float(row["phase_pi"]) for row in rows], dtype=np.float64)
        amplitude = np.asarray([float(row["amplitude"]) for row in rows], dtype=np.float64)
        axes[0].plot(x, phase_pi, marker=".", linewidth=0.85, alpha=0.8, label=f"frame {frame_count}")
        axes[1].plot(x, amplitude, marker=".", linewidth=0.85, alpha=0.8, label=f"frame {frame_count}")

    axes[0].set_title("Phase overlay", fontsize=9)
    axes[0].set_xlabel("Symbol")
    axes[0].set_ylabel("Phase / pi")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, color="#d9d9d9", linewidth=0.5)
    axes[1].set_title("Amplitude overlay", fontsize=9)
    axes[1].set_xlabel("Symbol")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, color="#d9d9d9", linewidth=0.5)
    if len(grouped) <= 10:
        axes[0].legend(fontsize=7, loc="best")
        axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.input.with_name(args.input.stem + "_plots")
    rows = load_rows(args.input, args.frame, args.payload_only)
    if not rows:
        raise SystemExit("没有读到可分析的 peak 行，请检查输入 CSV 或 --frame/--payload-only 参数。")

    grouped = group_by_frame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_csv = out_dir / "phase_amplitude_analysis.csv"
    write_analysis_csv(analysis_csv, rows)
    for frame_count, frame_rows in grouped.items():
        plot_frame(
            frame_count,
            frame_rows,
            out_dir / f"frame_{frame_count:03d}_phase_amplitude.png",
            args.smooth_window,
            args.dpi,
        )
    if len(grouped) > 1:
        plot_overlay(grouped, out_dir / "all_frames_overlay.png", args.dpi)

    print(f"frames={len(grouped)} rows={len(rows)}")
    print(f"wrote={analysis_csv}")
    print(f"wrote_plots={out_dir}")


if __name__ == "__main__":
    main()
