#!/usr/bin/env python3
"""通过 UHD 兼容的 USRP 采集原始 complex64 IQ 数据。

输出文件是不带文件头的 GNU Radio ``gr_complex`` 样点流。在常见的小端序主机上，
可以直接用下面的方式读取::

    iq = numpy.fromfile(path, dtype=numpy.complex64)

默认参数与当前固定帧 SX1276 实验一致：中心频率 487.7 MHz、采样率 500 ksample/s、
LoRa 带宽 125 kHz（过采样倍数 OSR=4）。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time

try:
    from gnuradio import blocks
    from gnuradio import gr
    from gnuradio import uhd
except ImportError as exc:
    raise SystemExit(
        "当前 Python 环境中没有 GNU Radio UHD 支持。请在采集机上激活已安装 "
        "GNU Radio 和 UHD 的 Python/Conda 环境。"
    ) from exc


COMPLEX64_BYTES = 8


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"应为正数，实际输入为 {value!r}")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(f"应为非负数，实际输入为 {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过 USRP 将原始 complex64 IQ 数据保存到 .bin 文件。"
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="原始 IQ .bin 输出路径。")
    parser.add_argument(
        "--duration",
        type=nonnegative_float,
        default=60.0,
        help="调谐稳定后实际保存的秒数；设为 0 时持续采集到按下 Ctrl+C（默认：60）。",
    )
    parser.add_argument(
        "--settle-time",
        type=nonnegative_float,
        default=1.0,
        help="USRP 开始输出后先丢弃的秒数，用于避开启动暂态（默认：1）。",
    )
    parser.add_argument("-f", "--center-freq", type=positive_float, default=487.7e6)
    parser.add_argument("-r", "--samp-rate", type=positive_float, default=500e3)
    parser.add_argument(
        "--lora-bandwidth",
        type=positive_float,
        default=125e3,
        help="LoRa 带宽，仅用于检查 OSR 和记录元数据（默认：125 kHz）。",
    )
    parser.add_argument(
        "--rf-bandwidth",
        type=positive_float,
        default=500e3,
        help="请求设置的 USRP 模拟前端带宽（默认：500 kHz）。",
    )
    parser.add_argument("-g", "--gain", type=float, default=20.0, help="手动接收增益，单位 dB。")
    parser.add_argument("--antenna", default="RX2", help="USRP 接收天线端口（默认：RX2）。")
    parser.add_argument(
        "--device-args",
        default="",
        help='UHD 设备参数，例如 "type=b200" 或 "serial=XXXXXXXX"。',
    )
    parser.add_argument("--channel", type=int, default=0, help="USRP 接收通道编号（默认：0）。")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="元数据 JSON 路径（默认：<输出文件>.json）。",
    )
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有的输出文件。")
    return parser.parse_args()


class USRPIQCollector(gr.top_block):
    def __init__(self, args: argparse.Namespace, sample_count: int | None, skip_count: int):
        super().__init__("USRP 原始 IQ 采集器", catch_exceptions=True)

        self.source = uhd.usrp_source(
            args.device_args,
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=[int(args.channel)],
            ),
        )
        self.source.set_samp_rate(float(args.samp_rate))
        self.source.set_center_freq(float(args.center_freq), int(args.channel))
        self.source.set_antenna(str(args.antenna), int(args.channel))
        self.source.set_gain(float(args.gain), int(args.channel))
        self.source.set_bandwidth(float(args.rf_bandwidth), int(args.channel))

        self.skip = blocks.skiphead(gr.sizeof_gr_complex, int(skip_count))
        self.sink = blocks.file_sink(gr.sizeof_gr_complex, str(args.output), False)
        self.sink.set_unbuffered(False)

        if sample_count is None:
            self.connect(self.source, self.skip, self.sink)
            self.head = None
        else:
            self.head = blocks.head(gr.sizeof_gr_complex, int(sample_count))
            self.connect(self.source, self.skip, self.head, self.sink)

    def actual_settings(self, channel: int) -> dict[str, float | str | int]:
        return {
            "center_freq_hz": float(self.source.get_center_freq(channel)),
            "samp_rate_sps": float(self.source.get_samp_rate()),
            "gain_db": float(self.source.get_gain(channel)),
            "rf_bandwidth_hz": float(self.source.get_bandwidth(channel)),
            "antenna": str(self.source.get_antenna(channel)),
            "channel": int(channel),
        }


def metadata_path(args: argparse.Namespace) -> Path:
    if args.metadata is not None:
        return args.metadata
    return Path(f"{args.output}.json")


def ensure_paths(args: argparse.Namespace, meta_path: Path) -> None:
    args.output = args.output.expanduser().resolve()
    meta_path = meta_path.expanduser().resolve()
    args.metadata = meta_path
    for path in (args.output, meta_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"拒绝覆盖已有文件 {path}；如需覆盖，请添加 --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)


def validate_osr(args: argparse.Namespace) -> int:
    ratio = float(args.samp_rate) / float(args.lora_bandwidth)
    rounded = int(round(ratio))
    if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "为适配弱包解码器，samp-rate / lora-bandwidth 必须为正整数；"
            f"当前为 {args.samp_rate:g} / {args.lora_bandwidth:g} = {ratio:.9g}"
        )
    return rounded


def main() -> int:
    args = parse_args()
    meta_path = metadata_path(args)
    ensure_paths(args, meta_path)
    os_factor = validate_osr(args)

    sample_count = None
    if float(args.duration) > 0.0:
        sample_count = int(round(float(args.duration) * float(args.samp_rate)))
    skip_count = int(round(float(args.settle_time) * float(args.samp_rate)))

    collector = USRPIQCollector(args, sample_count=sample_count, skip_count=skip_count)
    actual = collector.actual_settings(int(args.channel))
    actual_rate = float(actual["samp_rate_sps"])
    if sample_count is not None and not math.isclose(actual_rate, float(args.samp_rate), rel_tol=1e-9):
        # Head 模块的样点数由请求采样率计算。若 USRP 将采样率调整为其他值，
        # 直接停止，避免实际采集时长与元数据不一致。
        raise RuntimeError(
            f"USRP 将采样率从 {args.samp_rate:g} 调整为 {actual_rate:g}；"
            "请使用设备实际支持的采样率重新运行"
        )

    expected_size = None if sample_count is None else sample_count * COMPLEX64_BYTES
    print("USRP IQ 采集")
    print(f"  输出文件       : {args.output}")
    print(f"  中心频率       : {actual['center_freq_hz'] / 1e6:.6f} MHz")
    print(f"  采样率         : {actual_rate / 1e3:.3f} ksample/s")
    print(f"  LoRa 带宽/OSR  : {args.lora_bandwidth / 1e3:.3f} kHz / {os_factor}")
    print(f"  RF 前端带宽    : {actual['rf_bandwidth_hz'] / 1e3:.3f} kHz")
    print(f"  增益/天线      : {actual['gain_db']:.1f} dB / {actual['antenna']}")
    if sample_count is None:
        print("  采集时长       : 直到按下 Ctrl+C")
    else:
        print(f"  采集时长       : {args.duration:.3f} s（{sample_count} 个样点）")
        print(f"  预计文件大小   : {expected_size / (1024 ** 2):.2f} MiB")
    print(f"  启动丢弃时长   : {args.settle_time:.3f} s")

    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    started_utc = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    try:
        collector.start()
        if sample_count is not None:
            while not stop_requested:
                if collector.head is not None and collector.head.nitems_written(0) >= sample_count:
                    break
                time.sleep(0.1)
        else:
            while not stop_requested:
                time.sleep(0.2)
    finally:
        collector.stop()
        collector.wait()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

    elapsed = time.monotonic() - started_monotonic
    file_size = args.output.stat().st_size if args.output.exists() else 0
    if file_size % COMPLEX64_BYTES != 0:
        raise RuntimeError(f"输出大小 {file_size} 字节未按 complex64 样点对齐")
    written_samples = file_size // COMPLEX64_BYTES

    metadata = {
        "format": "raw complex64 interleaved IQ, no header",
        "dtype": "numpy.complex64",
        "sample_bytes": COMPLEX64_BYTES,
        "output_file": str(args.output),
        "output_size_bytes": int(file_size),
        "written_samples": int(written_samples),
        "recorded_duration_s": float(written_samples / actual_rate) if actual_rate > 0 else 0.0,
        "wall_elapsed_s": float(elapsed),
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "requested": {
            "center_freq_hz": float(args.center_freq),
            "samp_rate_sps": float(args.samp_rate),
            "gain_db": float(args.gain),
            "rf_bandwidth_hz": float(args.rf_bandwidth),
            "antenna": str(args.antenna),
            "channel": int(args.channel),
            "device_args": str(args.device_args),
            "duration_s": float(args.duration),
            "settle_time_s": float(args.settle_time),
        },
        "actual": actual,
        "lora": {
            "center_freq_hz": float(args.center_freq),
            "bandwidth_hz": float(args.lora_bandwidth),
            "oversampling_factor": int(os_factor),
            "expected_sf": 10,
            "expected_cr": "4/7",
            "expected_preamble_symbols": 32,
            "expected_phy_payload_bytes": 33,
            "expected_tx_period_s": 3.0,
            "fixed_fcnt": 1,
            "sync_word": "0x34",
            "explicit_header": True,
            "payload_crc": True,
            "tx_power_dbm": 2,
        },
        "interrupted": bool(stop_requested),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"IQ 已保存       : {args.output}（{written_samples} 个样点，{file_size} 字节）")
    print(f"元数据已保存    : {args.metadata}")
    if sample_count is not None and written_samples != sample_count and not stop_requested:
        print(
            f"警告：请求采集 {sample_count} 个样点，实际写入 {written_samples} 个",
            file=sys.stderr,
        )
        return 2
    return 130 if stop_requested and sample_count is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
