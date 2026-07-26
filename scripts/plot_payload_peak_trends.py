#!/usr/bin/env python3
"""绘制 header-first FFT demod 后 payload peak 的相位/幅度趋势。"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 run_header_first_demod.py 导出的逐 symbol CSV，"
            "只使用 header_valid=1 的 payload symbol，绘制每个 packet 的 FFT peak 相位和幅度趋势。"
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="header-first symbol CSV，例如 *_header_first_symbols.csv。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="每个 packet 的 PNG 输出目录。默认写到输入文件旁边的 <stem>_payload_peak_trends/。",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=None,
        help="可选：输出绘图用的精简 CSV。",
    )
    parser.add_argument(
        "--packet",
        type=int,
        default=None,
        help="只画指定 packet_index。默认画全部有效 packet。",
    )
    parser.add_argument(
        "--phase-unit",
        choices=("pi", "rad"),
        default="pi",
        help="相位纵轴单位，默认 pi。",
    )
    parser.add_argument(
        "--unwrap-phase",
        action="store_true",
        default=False,
        help="对每个 packet 的相位做 unwrap。默认保留 [-pi, pi] 原始相位。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="输出 PNG 的 DPI，默认 220。",
    )
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return int(default)
    return int(float(value))


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return float(default)
    return float(value)


def load_payload_rows(path: Path, packet: int | None) -> list[dict[str, object]]:
    """读取 payload peak 行；只保留 header checksum 通过的帧。"""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if raw.get("stage") != "payload":
                continue
            if _int(raw, "header_valid", 0) != 1:
                continue
            packet_index = _int(raw, "packet_index", -1)
            if packet is not None and packet_index != int(packet):
                continue
            rows.append(
                {
                    "frame_index": _int(raw, "frame_index", -1),
                    "packet_index": packet_index,
                    "event_index": _int(raw, "event_index", -1),
                    "payload_symbol_index": _int(raw, "stage_symbol_index", -1),
                    "raw_fft_bin": _int(raw, "raw_fft_bin", -1),
                    "signed_fft_bin": _int(raw, "signed_fft_bin", 0),
                    "symbol_value": _int(raw, "symbol_value", -1),
                    "peak_amp": _float(raw, "peak_amp"),
                    "peak_phase": _float(raw, "peak_phase"),
                    "peak_phase_pi": _float(raw, "peak_phase") / math.pi,
                    "peak_margin_db": _float(raw, "peak_margin_db"),
                }
            )
    rows.sort(key=lambda item: (int(item["packet_index"]), int(item["payload_symbol_index"])))
    return rows


def group_by_packet(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["packet_index"])].append(row)
    return dict(sorted(grouped.items()))


def write_analysis_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """写出绘图用的精简表，方便人工查具体 packet / symbol。"""

    fields = [
        "frame_index",
        "packet_index",
        "event_index",
        "payload_symbol_index",
        "raw_fft_bin",
        "signed_fft_bin",
        "symbol_value",
        "peak_amp",
        "peak_phase",
        "peak_phase_pi",
        "peak_margin_db",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_one_packet(
    packet_index: int,
    packet_rows: list[dict[str, object]],
    out_path: Path,
    phase_unit: str,
    unwrap_phase: bool,
    dpi: int,
) -> None:
    """绘制单个 packet 的 payload peak 相位和幅度趋势。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([int(row["payload_symbol_index"]) for row in packet_rows], dtype=np.float64)
    phase = np.asarray([float(row["peak_phase"]) for row in packet_rows], dtype=np.float64)
    if unwrap_phase:
        phase = np.unwrap(phase)
    phase_y = phase / math.pi if phase_unit == "pi" else phase
    amp = np.asarray([float(row["peak_amp"]) for row in packet_rows], dtype=np.float64)
    event_indices = sorted({int(row["event_index"]) for row in packet_rows})
    frame_indices = sorted({int(row["frame_index"]) for row in packet_rows})

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.4), dpi=int(dpi), sharex=True)
    line_color = "#1f77b4"
    marker_style = {
        "marker": "o",
        "markersize": 3.2,
        "markerfacecolor": "white",
        "linewidth": 1.05,
        "alpha": 0.92,
        "color": line_color,
    }

    axes[0].plot(x, phase_y, **marker_style)
    axes[1].plot(x, amp, **marker_style)

    axes[0].set_title("Payload selected FFT peak phase")
    axes[0].set_ylabel("Phase / pi" if phase_unit == "pi" else "Phase (rad)")
    if not unwrap_phase and phase_unit == "pi":
        axes[0].set_ylim(-1.05, 1.05)
    axes[0].grid(True, color="#dddddd", linewidth=0.55)

    axes[1].set_title("Payload selected FFT peak amplitude")
    axes[1].set_xlabel("Payload symbol index")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, color="#dddddd", linewidth=0.55)

    title = (
        f"Packet {packet_index} payload FFT-demod selected peak trends"
        f" | frame {frame_indices[0] if frame_indices else '?'}"
        f" | event {event_indices[0] if event_indices else '?'}"
    )
    if unwrap_phase:
        title += " (phase unwrapped)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_payload_trends(
    grouped: dict[int, list[dict[str, object]]],
    out_dir: Path,
    phase_unit: str,
    unwrap_phase: bool,
    dpi: int,
) -> list[Path]:
    """把每个 packet 分别保存为一个 PNG。"""

    if not grouped:
        raise ValueError("没有可绘制的 payload 行；请检查输入 CSV 是否包含 header_valid=1 的 payload symbol。")

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for packet_index, packet_rows in grouped.items():
        event_index = int(packet_rows[0]["event_index"]) if packet_rows else -1
        out_path = out_dir / f"packet_{packet_index:03d}_event_{event_index:03d}_payload_peak_trends.png"
        plot_one_packet(
            packet_index=packet_index,
            packet_rows=packet_rows,
            out_path=out_path,
            phase_unit=phase_unit,
            unwrap_phase=unwrap_phase,
            dpi=dpi,
        )
        paths.append(out_path)
    return paths


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.input.with_name(args.input.stem + "_payload_peak_trends")
    analysis_path = args.analysis_output

    rows = load_payload_rows(args.input, packet=args.packet)
    grouped = group_by_packet(rows)
    out_paths = plot_payload_trends(
        grouped,
        out_dir=out_dir,
        phase_unit=args.phase_unit,
        unwrap_phase=args.unwrap_phase,
        dpi=args.dpi,
    )
    if analysis_path is not None:
        write_analysis_csv(analysis_path, rows)

    print(f"packets={len(grouped)}")
    print(f"payload_rows={len(rows)}")
    print(f"wrote_dir={out_dir}")
    print(f"wrote_pngs={len(out_paths)}")
    if analysis_path is not None:
        print(f"wrote_analysis={analysis_path}")


if __name__ == "__main__":
    main()
