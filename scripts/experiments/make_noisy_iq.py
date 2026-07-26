#!/usr/bin/env python3
"""Generate lower-SNR LoRa IQ captures by adding complex AWGN.

The implementation lives in ``weakPacket_decoding/noisy_iq`` so the detector,
power estimator, IQ writer, and reporting code can be tested or reused
independently. This file stays as a thin script entry point.

中文说明：这个文件只保留命令行入口，真正的加噪、测量、写 metadata
等逻辑都拆到了旁边的 ``noisy_iq`` 包里，后续调试时可以按模块看。
"""
# python .\gr-lora_sdr\weakPacket_decoding\scripts\experiments\make_noisy_iq.py -i .\gr-lora_sdr\data\USRP_IQ\0_0_0_10_14_8.bin --samp-rate 500000 --bw 125000 --sync-word 0x34 --noise-power-db 10 15 25 30 --overwrite --preamble-len 8
from __future__ import annotations

from pathlib import Path
import sys


WEAKPACKET_ROOT = Path(__file__).resolve().parents[2]
# 直接运行 scripts/experiments/make_noisy_iq.py 时，Python 默认只认识当前目录；
# 这里把 weakPacket_decoding 放进 sys.path，保证可以导入 noisy_iq 包。
if str(WEAKPACKET_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAKPACKET_ROOT))

from noisy_iq.cli import parse_args
from noisy_iq.runner import main


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
