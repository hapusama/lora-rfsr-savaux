"""从原始 complex64 IQ 执行前导码检测和 gr-lora 风格 frame sync。

这是当前 ``weak_decoder`` 的标准 FFT-demod 前端入口。它复用经过真实 IQ
验证的 ``scripts.run_weak_sync_chain``，但默认参数直接取自 Branch4 固定帧 profile。
输出 CSV 可继续交给 ``scripts/run_header_first_demod.py`` 或 OS-LoRa/GLS 链路。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


WEAK_ROOT = Path(__file__).resolve().parent.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from scripts.run_weak_sync_chain import main as run_sync_chain  # noqa: E402
from weak_decoder.branch4_profile import BRANCH4_PROFILE  # noqa: E402


def parse_int_auto(text: str) -> int:
    return int(str(text), 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    profile = BRANCH4_PROFILE
    parser = argparse.ArgumentParser(
        description="读取 raw complex64 IQ，完成前导码检测、帧定界以及 CFO/STO/SFO frame sync。"
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="输入 complex64 .bin 文件。")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="同步结果 CSV；默认写到 weakPacket_decoding/data/frontend/<bin名>_sync.csv。",
    )
    parser.add_argument("--sf", type=int, default=profile.sf)
    parser.add_argument("--bw", type=float, default=profile.bandwidth_hz)
    parser.add_argument("--samp-rate", type=float, default=profile.sample_rate_sps)
    parser.add_argument("--center-freq", type=float, default=profile.center_freq_hz)
    parser.add_argument("--sync-word", type=parse_int_auto, default=profile.sync_word)
    parser.add_argument("--preamble-len", type=int, default=profile.preamble_symbols)
    parser.add_argument("--win-chirps", type=int, default=4, help="检测窗口中的 chirp 数，默认 4。")
    parser.add_argument("--align-step-samples", type=int, default=1, help="粗 chirp 边界搜索步长，默认 1。")
    parser.add_argument("--frame-step-samples", type=int, default=1, help="SFD/frame 搜索步长，默认 1。")
    parser.add_argument("--max-packets", type=int, default=None, help="最多处理多少个检测事件。")
    parser.add_argument("--sample-limit", type=int, default=None, help="只扫描前 N 个 IQ 样点。")
    parser.add_argument("--events-csv", type=Path, default=None, help="可选：保存粗检测事件。")
    parser.add_argument("--windows-csv", type=Path, default=None, help="可选：保存全部滑窗结果，文件可能较大。")
    parser.add_argument("--framesync-peaks-csv", type=Path, default=None, help="可选：保存同步验证 peak。")
    parser.add_argument("--diagnostics-dir", type=Path, default=None, help="可选：保存 STFT 和前导码频谱图。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (WEAK_ROOT / "data" / "frontend" / f"{input_path.stem}_sync.csv").resolve()
    )

    forwarded = [
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--sf",
        str(args.sf),
        "--bw",
        str(args.bw),
        "--samp-rate",
        str(args.samp_rate),
        "--center-freq",
        str(args.center_freq),
        "--sync-word",
        hex(int(args.sync_word)),
        "--preamble-len",
        str(args.preamble_len),
        "--win-chirps",
        str(args.win_chirps),
        "--align-step-samples",
        str(args.align_step_samples),
        "--frame-step-samples",
        str(args.frame_step_samples),
    ]
    if args.max_packets is not None:
        forwarded.extend(("--max-events", str(args.max_packets)))
    if args.sample_limit is not None:
        forwarded.extend(("--sample-limit", str(args.sample_limit)))
    for flag, path in (
        ("--events-csv", args.events_csv),
        ("--windows-csv", args.windows_csv),
        ("--framesync-peaks-csv", args.framesync_peaks_csv),
    ):
        if path is not None:
            forwarded.extend((flag, str(path.expanduser().resolve())))
    if args.diagnostics_dir is not None:
        diagnostics = args.diagnostics_dir.expanduser().resolve()
        forwarded.extend(("--stft-dir", str(diagnostics / "stft")))
        forwarded.extend(("--framesync-spectrum-dir", str(diagnostics / "preamble_spectrum")))

    run_sync_chain(forwarded)
    print(f"frontend_sync_csv={output_path}")
    print(
        "next_fft_demod=python scripts/run_header_first_demod.py "
        f"--input \"{input_path}\" --sync-csv \"{output_path}\" ..."
    )


if __name__ == "__main__":
    main()
