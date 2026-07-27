#!/usr/bin/env python3
"""按 LoRa 符号边界绘制 reference cfile 的逐符号 STFT。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy import signal


_REFERENCE_NAME_RE = re.compile(r"^signalout_(\d{6})_fulltrim$")
_SECTION_ORDER = ("preamble", "sync", "sfd", "header", "payload")


@dataclass(frozen=True)
class SymbolSpan:
    """一个独立 STFT panel 对应的样本区间。"""

    section: str
    section_index: int
    label: str
    start_sample: int
    stop_sample: int
    symbol_id: int | None = None

    @property
    def sample_count(self) -> int:
        return int(self.stop_sample - self.start_sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 raw complex64 reference 和配对 metadata，严格按 LoRa 符号边界"
            "分别计算 STFT，并分页输出 PNG。"
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="输入 signalout_XXXXXX_fulltrim.cfile。",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        type=Path,
        default=None,
        help="配对 JSON；省略时从 ../metadata/XXXXXX.json 自动查找。",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="PNG 输出目录；默认写到数据根目录的 stft/<cfile stem>/。",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=_SECTION_ORDER,
        default=None,
        help=(
            "只绘制指定部分，可重复传入；默认绘制 preamble、sync、SFD、"
            "header 和 payload。"
        ),
    )
    parser.add_argument(
        "--symbols-per-page",
        type=int,
        default=12,
        help="每张 PNG 的 panel 数，默认 12。",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="每页 panel 列数，默认 3。",
    )
    parser.add_argument(
        "--window-samples",
        type=int,
        default=256,
        help="每个独立 STFT 的 Hann 窗长度，默认 256 样本。",
    )
    parser.add_argument(
        "--overlap-samples",
        type=int,
        default=224,
        help="相邻 STFT 窗重叠样本数，默认 224。",
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=2048,
        help="STFT FFT 点数，默认 2048；只影响图像频率网格。",
    )
    parser.add_argument(
        "--dynamic-range-db",
        type=float,
        default=70.0,
        help="每个 panel 相对自身峰值显示的动态范围，默认 70 dB。",
    )
    parser.add_argument(
        "--frequency-limit-hz",
        type=float,
        default=None,
        help="纵轴正负频率范围；默认使用 0.65 * LoRa BW。",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG DPI，默认 160。")
    return parser.parse_args()


def infer_metadata_path(iq_path: Path) -> Path:
    """按照 reference/ 与 metadata/ 的配对目录约定推导 JSON 路径。"""

    match = _REFERENCE_NAME_RE.fullmatch(iq_path.stem)
    if match is None:
        raise ValueError(
            "无法从文件名推导 metadata；请显式传入 --metadata。"
        )
    payload_id = match.group(1)
    if iq_path.parent.name == "reference":
        return iq_path.parent.parent / "metadata" / f"{payload_id}.json"
    return iq_path.parent / "metadata" / f"{payload_id}.json"


def load_metadata(path: Path) -> dict[str, object]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "lora-rfsr-reference":
        raise ValueError(f"不支持的 metadata schema：{metadata.get('schema')!r}。")
    return metadata


def _append_full_symbols(
    spans: list[SymbolSpan],
    section: str,
    label_prefix: str,
    count: int,
    cursor: int,
    samples_per_symbol: int,
    symbol_ids: Iterable[int] | None = None,
) -> int:
    ids = list(symbol_ids) if symbol_ids is not None else [None] * int(count)
    if len(ids) != int(count):
        raise ValueError(
            f"{section} symbol 数量不一致：count={count}, ids={len(ids)}。"
        )
    for index, symbol_id in enumerate(ids):
        stop = int(cursor + samples_per_symbol)
        spans.append(
            SymbolSpan(
                section=section,
                section_index=index,
                label=f"{label_prefix}{index:02d}",
                start_sample=int(cursor),
                stop_sample=stop,
                symbol_id=None if symbol_id is None else int(symbol_id),
            )
        )
        cursor = stop
    return int(cursor)


def build_symbol_spans(metadata: dict[str, object], sample_count: int) -> list[SymbolSpan]:
    """从 metadata 重建 packet 内所有完整/部分 LoRa 符号边界。"""

    phy = metadata["phy"]
    symbols = metadata["symbols"]
    iq = metadata["iq"]
    if not isinstance(phy, dict) or not isinstance(symbols, dict) or not isinstance(iq, dict):
        raise ValueError("metadata 缺少 phy/symbols/iq 对象。")

    samples_per_symbol = int(phy["samples_per_symbol"])
    preamble_count = int(phy["preamble_symbols"])
    leading = int(iq.get("leading_silence_samples", 0))
    trailing = int(iq.get("trailing_silence_samples", 0))
    sync_word = int(phy["sync_word"])
    header_ids = [int(value) for value in symbols["header_ids"]]
    payload_ids = [int(value) for value in symbols["payload_ids"]]

    spans: list[SymbolSpan] = []
    cursor = int(leading)
    cursor = _append_full_symbols(
        spans,
        section="preamble",
        label_prefix="PRE",
        count=preamble_count,
        cursor=cursor,
        samples_per_symbol=samples_per_symbol,
        symbol_ids=[0] * preamble_count,
    )
    sync_ids = [
        ((sync_word >> 4) & 0x0F) << 3,
        (sync_word & 0x0F) << 3,
    ]
    cursor = _append_full_symbols(
        spans,
        section="sync",
        label_prefix="SYNC",
        count=2,
        cursor=cursor,
        samples_per_symbol=samples_per_symbol,
        symbol_ids=sync_ids,
    )

    # SFD 包含两个完整 downchirp 和最后一个 1/4 downchirp。
    for index, fraction in enumerate((1.0, 1.0, 0.25)):
        length = int(round(samples_per_symbol * fraction))
        stop = int(cursor + length)
        spans.append(
            SymbolSpan(
                section="sfd",
                section_index=index,
                label=f"SFD{index:02d}" if index < 2 else "SFD02 (1/4)",
                start_sample=int(cursor),
                stop_sample=stop,
            )
        )
        cursor = stop

    cursor = _append_full_symbols(
        spans,
        section="header",
        label_prefix="HDR",
        count=len(header_ids),
        cursor=cursor,
        samples_per_symbol=samples_per_symbol,
        symbol_ids=header_ids,
    )
    cursor = _append_full_symbols(
        spans,
        section="payload",
        label_prefix="PAY",
        count=len(payload_ids),
        cursor=cursor,
        samples_per_symbol=samples_per_symbol,
        symbol_ids=payload_ids,
    )

    expected_samples = int(cursor + trailing)
    declared_samples = int(iq["complex_samples"])
    if declared_samples != int(sample_count):
        raise ValueError(
            f"cfile 样本数与 metadata 不一致：file={sample_count}, "
            f"metadata={declared_samples}。"
        )
    if expected_samples != int(sample_count):
        raise ValueError(
            f"符号边界合计与 cfile 不一致：expected={expected_samples}, "
            f"file={sample_count}。"
        )
    return spans


def symbol_stft(
    samples: np.ndarray,
    sample_rate_hz: float,
    window_samples: int,
    overlap_samples: int,
    nfft: int,
    dynamic_range_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只在一个符号内部计算 STFT，并返回中心化频率和相对幅度。"""

    if samples.size < 2:
        raise ValueError("symbol 样本数不足，无法计算 STFT。")
    window = min(int(window_samples), int(samples.size))
    overlap = min(int(overlap_samples), window - 1)
    fft_size = max(int(nfft), window)
    frequencies, times, spectrum = signal.stft(
        samples,
        fs=float(sample_rate_hz),
        window="hann",
        nperseg=window,
        noverlap=overlap,
        nfft=fft_size,
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(spectrum)
    peak = float(np.max(magnitude))
    if peak <= 0.0:
        relative_db = np.full(magnitude.shape, -float(dynamic_range_db))
    else:
        floor = peak * 10.0 ** (-float(dynamic_range_db) / 20.0)
        relative_db = 20.0 * np.log10(np.maximum(magnitude, floor) / peak)
    return (
        np.fft.fftshift(frequencies) / 1_000.0,
        times * 1_000.0,
        np.fft.fftshift(relative_db, axes=0),
    )


def render_pages(
    samples: np.ndarray,
    spans: list[SymbolSpan],
    metadata: dict[str, object],
    output_dir: Path,
    input_stem: str,
    section_tag: str,
    symbols_per_page: int,
    columns: int,
    window_samples: int,
    overlap_samples: int,
    nfft: int,
    dynamic_range_db: float,
    frequency_limit_hz: float,
    dpi: int,
) -> list[Path]:
    """分页绘图；每个 axes 严格对应一个 LoRa 符号。"""

    phy = metadata["phy"]
    if not isinstance(phy, dict):
        raise ValueError("metadata.phy 必须是对象。")
    sample_rate_hz = float(phy["sample_rate_hz"])
    rows = int(math.ceil(int(symbols_per_page) / int(columns)))
    page_count = int(math.ceil(len(spans) / int(symbols_per_page)))
    output_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(page_count):
        page_spans = spans[
            page_index * int(symbols_per_page) :
            (page_index + 1) * int(symbols_per_page)
        ]
        fig, axes = plt.subplots(
            rows,
            int(columns),
            figsize=(5.2 * int(columns), 3.6 * rows),
            squeeze=False,
            constrained_layout=True,
        )
        flat_axes = list(axes.flat)
        last_mesh = None
        for axis, span in zip(flat_axes, page_spans):
            segment = samples[span.start_sample : span.stop_sample]
            frequencies_khz, times_ms, relative_db = symbol_stft(
                segment,
                sample_rate_hz=sample_rate_hz,
                window_samples=window_samples,
                overlap_samples=overlap_samples,
                nfft=nfft,
                dynamic_range_db=dynamic_range_db,
            )
            last_mesh = axis.pcolormesh(
                times_ms,
                frequencies_khz,
                relative_db,
                shading="auto",
                cmap="viridis",
                vmin=-float(dynamic_range_db),
                vmax=0.0,
                rasterized=True,
            )
            symbol_text = (
                "" if span.symbol_id is None else f" | id={span.symbol_id}"
            )
            axis.set_title(
                f"{span.label}{symbol_text}\n"
                f"samples [{span.start_sample}, {span.stop_sample})",
                fontsize=9,
            )
            axis.set_ylim(
                -float(frequency_limit_hz) / 1_000.0,
                float(frequency_limit_hz) / 1_000.0,
            )
            axis.set_xlabel("Time within symbol (ms)")
            axis.set_ylabel("Baseband frequency (kHz)")
            axis.grid(False)
        for axis in flat_axes[len(page_spans) :]:
            axis.set_visible(False)

        fig.suptitle(
            f"{input_stem} | symbol-wise STFT | "
            f"SF{int(phy['sf'])} BW={int(phy['bandwidth_hz']) / 1_000:g} kHz "
            f"Fs={int(phy['sample_rate_hz']) / 1_000:g} kSPS | "
            f"page {page_index + 1}/{page_count}",
            fontsize=13,
        )
        if last_mesh is not None:
            colorbar = fig.colorbar(
                last_mesh,
                ax=[axis for axis in flat_axes if axis.get_visible()],
                shrink=0.92,
                pad=0.02,
            )
            colorbar.set_label("Relative magnitude (dB)")

        output_path = (
            output_dir
            / f"{input_stem}_stft_{section_tag}_p{page_index + 1:03d}.png"
        )
        fig.savefig(output_path, dpi=int(dpi))
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def main() -> int:
    args = parse_args()
    if int(args.symbols_per_page) < 1 or int(args.columns) < 1:
        raise ValueError("--symbols-per-page 和 --columns 必须为正整数。")
    if int(args.window_samples) < 2:
        raise ValueError("--window-samples 必须至少为 2。")
    if not 0 <= int(args.overlap_samples) < int(args.window_samples):
        raise ValueError("--overlap-samples 必须满足 0 <= overlap < window。")
    if float(args.dynamic_range_db) <= 0:
        raise ValueError("--dynamic-range-db 必须为正数。")

    iq_path = args.input.expanduser().resolve()
    metadata_path = (
        args.metadata.expanduser().resolve()
        if args.metadata is not None
        else infer_metadata_path(iq_path)
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else iq_path.parent.parent / "stft" / iq_path.stem
    )
    metadata = load_metadata(metadata_path)
    iq = metadata["iq"]
    phy = metadata["phy"]
    if not isinstance(iq, dict) or not isinstance(phy, dict):
        raise ValueError("metadata 缺少 iq/phy 对象。")
    if str(iq.get("dtype", "")) != "<c8":
        raise ValueError(f"当前只支持 metadata dtype=<c8，得到 {iq.get('dtype')!r}。")

    samples = np.fromfile(iq_path, dtype=np.dtype("<c8"))
    spans = build_symbol_spans(metadata, sample_count=int(samples.size))
    selected_sections = tuple(args.section or _SECTION_ORDER)
    selected = [span for span in spans if span.section in selected_sections]
    if not selected:
        raise ValueError("所选 section 没有可绘制的符号。")
    section_tag = (
        "all"
        if selected_sections == _SECTION_ORDER
        else "-".join(section for section in _SECTION_ORDER if section in selected_sections)
    )
    frequency_limit_hz = (
        float(args.frequency_limit_hz)
        if args.frequency_limit_hz is not None
        else 0.65 * float(phy["bandwidth_hz"])
    )
    if frequency_limit_hz <= 0:
        raise ValueError("--frequency-limit-hz 必须为正数。")

    outputs = render_pages(
        samples=samples,
        spans=selected,
        metadata=metadata,
        output_dir=output_dir,
        input_stem=iq_path.stem,
        section_tag=section_tag,
        symbols_per_page=int(args.symbols_per_page),
        columns=int(args.columns),
        window_samples=int(args.window_samples),
        overlap_samples=int(args.overlap_samples),
        nfft=int(args.nfft),
        dynamic_range_db=float(args.dynamic_range_db),
        frequency_limit_hz=frequency_limit_hz,
        dpi=int(args.dpi),
    )
    print(f"input={iq_path}")
    print(f"metadata={metadata_path}")
    print(f"selected_symbols={len(selected)}/{len(spans)}")
    print(f"leading_silence_samples={int(iq.get('leading_silence_samples', 0))}")
    for output in outputs:
        print(f"wrote={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
