#!/usr/bin/env python3
"""从 STM32 UART 清单生成逐包 LoRa reference。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "reference_phy"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weak_decoder.rf_super_resolution.reference_phy import (  # noqa: E402
    parse_uart_reference_log,
    phy_config_from_uart,
    write_reference_packet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "解析 STM32 [TX Frame]，生成不含 RF-SR 私有四字节头的 "
            "raw complex64 LoRa reference。"
        )
    )
    parser.add_argument(
        "--uart-log",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "packet_reference.txt",
        help="包含 [TX Payload]/[TX Frame]/[TX PHY] 的 UART 日志。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "输出根目录；脚本会在其下创建 reference/ 和 metadata/，"
            f"默认：{DEFAULT_OUTPUT_ROOT}。"
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1_000_000,
        help=(
            "reference 采样率，单位 samples/s，默认 1000000。若直接使用 "
            "RF-SR 原版 OTA loader，应先确认其 2 MSPS 磁盘输入约定。"
        ),
    )
    parser.add_argument(
        "--leading-silence-samples",
        type=int,
        default=10_000,
        help=(
            "packet 前置 complex64 零样本数，默认 10000，与 RF-SR "
            "PHY.encode() 对齐；传 0 可恢复 packet-only 输出。"
        ),
    )
    parser.add_argument(
        "--payload-id",
        type=int,
        action="append",
        default=None,
        help="只生成指定 payload ID；可重复传入多个 ID。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只生成筛选结果中的前 N 包；烟测建议使用 1 或 2。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖同 ID 的现有 reference 和 metadata。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    uart_log = parse_uart_reference_log(args.uart_log)
    config = phy_config_from_uart(
        uart_log,
        sample_rate_hz=int(args.sample_rate),
        leading_silence_samples=int(args.leading_silence_samples),
    )
    packets = list(uart_log.first_reference_cycle())
    if args.payload_id:
        # 始终只从第一轮唯一 ID 中选择，避免重复发送覆盖同一 reference。
        selected = set(int(value) for value in args.payload_id)
        known = {packet.payload_id for packet in packets}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"UART 第一轮不包含 payload ID：{unknown}。")
        packets = [packet for packet in packets if packet.payload_id in selected]
    if args.limit is not None:
        if int(args.limit) < 1:
            raise ValueError("--limit 必须为正整数。")
        packets = packets[: int(args.limit)]

    print(
        "UART 第一轮："
        f"{len(uart_log.first_reference_cycle())} 包，"
        f"SHA256={uart_log.source_sha256}"
    )
    print(
        "PHY: "
        f"SF{config.sf} BW={config.bandwidth_hz} CR=4/{config.cr + 4} "
        f"preamble={config.preamble_symbols} sync=0x{config.sync_word:02X} "
        f"Fs={config.sample_rate_hz} LDRO={int(config.ldro)} "
        f"CRC={config.crc_mode} "
        f"leading_zeros={config.leading_silence_samples} trailing_zeros=0"
    )
    total_bytes = 0
    for packet in packets:
        iq_path, metadata_path = write_reference_packet(
            uart_log=uart_log,
            packet=packet,
            config=config,
            output_root=args.output_root,
            overwrite=bool(args.overwrite),
        )
        size = iq_path.stat().st_size
        total_bytes += size
        print(
            f"id={packet.payload_id:03d} seq={packet.seq:03d} "
            f"bytes={size} iq={iq_path} metadata={metadata_path}"
        )
    print(
        f"已生成 {len(packets)} 个 reference，"
        f"合计 {total_bytes / (1024 ** 3):.4f} GiB。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
