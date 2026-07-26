"""Shared paths and defaults for the noisy-IQ sweep tools.

中文说明：这里集中放默认路径和默认 sweep 参数，避免脚本、runner、
detector 之间互相硬编码路径。
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
WEAKPACKET_ROOT = PACKAGE_DIR.parent
GRLORA_ROOT = WEAKPACKET_ROOT.parent

# 默认输入仍然指向当前弱包实验最常用的 USRP IQ 捕获。
DEFAULT_INPUT = GRLORA_ROOT / "data" / "USRP_IQ" / "0_0_0_10_14_8.bin"
DEFAULT_OUTPUT_ROOT = WEAKPACKET_ROOT / "data" / "noisy_iq"
# GNU Radio 的 file_source 在 Windows 中文路径下不总是稳定；
# detector 会在这里放一个 ASCII hardlink 给流图读取。
FILE_SOURCE_STAGING_DIR = WEAKPACKET_ROOT / "_file_source_staging"

COMPLEX64_BYTES = 8
DEFAULT_NOISE_START_DB = -30.0
DEFAULT_NOISE_STOP_DB = 0.0
DEFAULT_NOISE_STEP_DB = 5.0

# Groundtruth is derived from the clean input at runtime; keep this only for old imports.
DEFAULT_EXPECTED_PAYLOAD_HEXES: list[str] = []
