"""Shared paths and defaults for the noisy-IQ sweep tools.

中文说明：这里集中放默认路径和默认 sweep 参数，避免脚本、runner、
detector 之间互相硬编码路径。
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
WEAKPACKET_ROOT = PACKAGE_DIR.parent

# Keep every default data path inside this repository.  Callers may still pass
# another input explicitly, but copying only lora-rfsr-savaux/ is sufficient
# for the documented workflow.
DEFAULT_INPUT = (
    WEAKPACKET_ROOT
    / "data"
    / "raw"
    / "ota"
    / (
        "rxcap_exp000_sess000_loclab1_condhighsnr_run000_sf12_"
        "bw125000_fs2000000_pre16_sw12_cr48_crc1_fc487700000_"
        "rxg20.cfile"
    )
)
DEFAULT_OUTPUT_ROOT = WEAKPACKET_ROOT / "data" / "noisy_iq"
# GNU Radio 的 file_source 在 Windows 中文路径下不总是稳定；
# detector 会在这里放一个 ASCII hardlink 给流图读取。
FILE_SOURCE_STAGING_DIR = WEAKPACKET_ROOT / "data" / "_file_source_staging"

COMPLEX64_BYTES = 8
DEFAULT_NOISE_START_DB = -30.0
DEFAULT_NOISE_STOP_DB = 0.0
DEFAULT_NOISE_STEP_DB = 5.0

# Groundtruth is derived from the clean input at runtime; keep this only for old imports.
DEFAULT_EXPECTED_PAYLOAD_HEXES: list[str] = []
