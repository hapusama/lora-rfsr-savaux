"""Command-line interface for the noisy-IQ sweep tool.

中文说明：这里只定义命令行参数，不做实际业务逻辑；这样参数说明再长，
也不会把加噪/测量流程挤在同一个脚本里。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .constants import (
    DEFAULT_INPUT,
    DEFAULT_NOISE_START_DB,
    DEFAULT_NOISE_STEP_DB,
    DEFAULT_NOISE_STOP_DB,
)
from .utils import parse_int_auto


def parse_args() -> argparse.Namespace:
    """Parse CLI options for noisy IQ generation and measurement."""
    # 参数保持和旧脚本兼容，便于继续使用 README 里的原命令。
    parser = argparse.ArgumentParser(
        description=(
            "Add complex AWGN to raw complex64 LoRa IQ captures step by step, "
            "then measure each output with gr-lora_sdr."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input raw complex64 IQ file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: weakPacket_decoding/data/noisy_iq/<input-stem>",
    )
    parser.add_argument(
        "--noise-power-db",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Explicit added-noise powers in dB relative to the chosen reference power. "
            "This is NOT target SNR. If omitted, use --noise-start-db/--noise-stop-db/--noise-step-db."
        ),
    )
    parser.add_argument(
        "--noise-start-db",
        type=float,
        default=DEFAULT_NOISE_START_DB,
        help=f"First added-noise power in dB relative to the reference. Default: {DEFAULT_NOISE_START_DB}.",
    )
    parser.add_argument(
        "--noise-stop-db",
        type=float,
        default=DEFAULT_NOISE_STOP_DB,
        help=f"Last added-noise power in dB relative to the reference. Default: {DEFAULT_NOISE_STOP_DB}.",
    )
    parser.add_argument(
        "--noise-step-db",
        type=float,
        default=DEFAULT_NOISE_STEP_DB,
        help=f"Step size for added-noise power in dB. Default: {DEFAULT_NOISE_STEP_DB}.",
    )
    parser.add_argument(
        "--power-mode",
        choices=("packet", "active", "total", "window"),
        default="total",
        help=(
            "Reference used only to scale the added-noise steps. packet uses gr-lora_sdr aligned "
            "whole-packet ranges; active estimates packet blocks above the noise floor; total uses "
            "the whole-file mean power; window uses --signal-start/--signal-samples. Default: total."
        ),
    )
    parser.add_argument("--sf", type=int, default=None, help="LoRa spreading factor. Default: infer from filename.")
    parser.add_argument("--bw", type=float, default=125000.0, help="LoRa bandwidth in Hz. Default: 125000.")
    parser.add_argument("--samp-rate", type=float, default=500000.0, help="IQ sample rate in Hz. Default: 500000.")
    parser.add_argument("--cr", type=int, default=1, help="LoRa coding-rate index used by gr-lora_sdr. Default: 1.")
    parser.add_argument(
        "--pay-len",
        type=int,
        default=255,
        help="Fallback payload length for implicit header or missing header metadata. Default: 255.",
    )
    parser.add_argument(
        "--has-crc",
        action="store_true",
        default=True,
        help="Packet has PHY CRC. Default: enabled.",
    )
    parser.add_argument("--no-crc", action="store_false", dest="has_crc", help="Packet has no PHY CRC.")
    parser.add_argument("--impl-head", action="store_true", default=False, help="Use implicit header mode.")
    parser.add_argument("--soft-decoding", action="store_true", default=False, help="Enable gr-lora_sdr soft decoding.")
    parser.add_argument(
        "--center-freq",
        type=float,
        default=487.7e6,
        help="RF center frequency for gr-lora_sdr SFO estimation. Default: 487.7e6.",
    )
    parser.add_argument(
        "--sync-word",
        type=parse_int_auto,
        default=0x34,
        help="LoRa sync word, decimal or hex. Default: 0x34.",
    )
    parser.add_argument(
        "--preamble-len",
        type=int,
        default=None,
        help="Expected preamble upchirp count / frame_sync trigger parameter. Default: infer from filename.",
    )
    parser.add_argument("--ldro-mode", type=int, default=2, help="LDRO mode: 0 off, 1 on, 2 auto. Default: 2.")
    parser.add_argument(
        "--crc-mode",
        type=int,
        choices=[0, 1],
        default=0,
        help="CRC algorithm mode for payload verification: 0=GRLORA, 1=SX1276. Default: 0.",
    )
    parser.add_argument(
        "--expected-payload-hex",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Override expected clean decoded payloads as hex strings. "
            "Default: auto-decode the clean input first and use those payloads as groundtruth."
        ),
    )
    parser.add_argument(
        "--no-expected-payload-check",
        action="store_true",
        help="Disable clean groundtruth decoding and only measure detection/SNR.",
    )
    parser.add_argument(
        "--print-header",
        action="store_true",
        default=False,
        help="Let gr-lora_sdr header_decoder print decoded PHY headers while detecting packet ranges.",
    )
    parser.add_argument(
        "--print-grlora",
        action="store_true",
        default=False,
        help="Let gr-lora_sdr fft_demod print demodulator info while detecting packet ranges.",
    )
    parser.add_argument(
        "--noise-percentile",
        type=float,
        default=10.0,
        help="Percentile of block powers used as noise floor in active/window modes.",
    )
    parser.add_argument(
        "--active-threshold-db",
        type=float,
        default=6.0,
        help="A block is active when its power is this many dB above the estimated noise floor.",
    )
    parser.add_argument(
        "--block-samples",
        type=int,
        default=32768,
        help="Block size for active/noise-floor power estimation.",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=1_000_000,
        help="Samples per streaming write chunk.",
    )
    parser.add_argument("--signal-start", type=int, default=None, help="Start sample for --power-mode window.")
    parser.add_argument("--signal-samples", type=int, default=None, help="Number of samples for --power-mode window.")
    parser.add_argument(
        "--ignore-existing-noise",
        action="store_true",
        help="In packet/active/window reference modes, do not subtract estimated existing noise from the reference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260503,
        help="Random seed. By default the same unit-noise realization is scaled for every noise step.",
    )
    parser.add_argument(
        "--independent-noise",
        action="store_true",
        help="Use a different random realization for each noise step.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Only process the first N samples. Mainly useful for smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate powers and print planned outputs only.")

    return parser.parse_args()
