#!/usr/bin/env python3
"""对公开 RF-SR checkpoint 做合成或官方 OTA 端到端审计。

默认模式保留固定合成包实验。传入 ``--official-ota-root`` 时，工具改为读取
RFSR-OTA Dataverse 的 ``metadata/*.json`` 与 ``ota/*.cfile``，并使用 metadata
中的 payload、src、dst 和 seqn 重建该数据集实际使用的旧版 PHY symbol 真值。
工具不训练模型，也不会自行下载数据。

实验评估三种噪声位置：

* ``pre_rfsr``：在 2 MS/s 加入 AWGN，抽取到 250 kS/s，运行前端，再从含噪
  前端输出重新估计 FrameSync；
* ``post_framesync_common_power``：冻结干净 FrameSync，在所有前端的 1 MS/s
  输出上加入相同绝对功率的 AWGN；
* ``post_framesync_gain_matched``：复用同一份归一化噪声 realization，并根据
  各前端的干净输出功率缩放到指定 SNR。

同步失败时，该包全部预期符号都计为端到端符号错误。同时只在同步成功包上报告
条件 SER，从而区分同步失败和解调失败。

官方 OTA 的快速重复实验可传 ``--fast-official``。该预设只比较 Native 与官方
OTA-RFSR、只运行 Savaux、裁剪大部分 guard，并像已锁定的 GNU Radio 流式接收机
一样复用干净包的粗 CFO 和候选帧位置；它仍会在每个含噪版本上重新验证精细
CFO/STO/SFO。默认模式保持独立冷启动同步，适合较慢但严格的审计。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable, NamedTuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RFSR_ROOT = REPO_ROOT / "third_party" / "rfsr"
for _path in (REPO_ROOT, RFSR_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rfsr import awgn, decode, encode  # noqa: E402
from rfsr.PHY import (  # noqa: E402
    encode_raw_phy,
    lora_header,
    lora_header_init,
    lora_payload,
    lora_payload_init,
)
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    _wrapped_tail_dft_batch,
)
from weak_decoder.chirp import bin_to_grlora_symbol  # noqa: E402
from weak_decoder.decoding.header_first_demod import (  # noqa: E402
    demod_symbol_sequence,
)
from weak_decoder.os_lora.system.oversampled_glrt import (  # noqa: E402
    BranchNoiseModel,
    branch_gls_scores,
    estimate_branch_noise_model,
    identity_branch_noise_model,
)
from weak_decoder.rf_super_resolution.frontend import (  # noqa: E402
    DEFAULT_OTA_CHECKPOINT,
    DEFAULT_SYNTHETIC_CHECKPOINT,
    RFSRFrontendConfig,
    RFSuperResolutionFrontend,
)
from weak_decoder.synchronization.single_packet import (  # noqa: E402
    SinglePacketSyncConfig,
    run_single_packet_sync,
)
from weak_decoder.synchronization.grlora_frame_sync import (  # noqa: E402
    run_grlora_frame_sync_validation,
)
from weak_decoder.synchronization.preamble_detector import (  # noqa: E402
    PreambleDetectorConfig,
)


SF = 12
BW_HZ = 125_000
HIGH_RATE_HZ = 2_000_000
LOW_RATE_HZ = 250_000
OUTPUT_RATE_HZ = 1_000_000
OS_FACTOR = OUTPUT_RATE_HZ // BW_HZ
N_BINS = 1 << SF
PREAMBLE_SYMBOLS = 8
SYNC_WORD = 0x12
PAYLOAD_BYTES = 16
CR = 4
LDRO = True
LEADING_SILENCE_HIGH = 10_000
TRAILING_SILENCE_HIGH = 64
OFFICIAL_OTA_GUARD_HIGH = 500_000
FAST_OFFICIAL_CONTEXT_CHIRPS = 2

METHODS = (
    "native_1msps",
    "official_interpolation",
    "official_synthetic_rfsr",
    "official_ota_rfsr",
)
DIRECT_250K_METHOD = "native_250ksps"
DECODERS = ("ordinary_fft", "savaux", "savaux_gls")
NOISE_STAGES = (
    "pre_rfsr",
    "post_framesync_common_power",
    "post_framesync_gain_matched",
)


class EvaluationPacket(NamedTuple):
    """一条合成或官方 OTA 物理包及其独立评分契约。"""

    packet_id: str
    expected: dict[str, list[int]]
    leading_guard_high: int
    trailing_guard_high: int
    center_frequency_hz: float
    reference_power: float
    source_snr_db: float | None
    samples: np.ndarray | None = None
    source_path: Path | None = None
    metadata_path: Path | None = None

    def load_samples(self) -> np.ndarray:
        if self.samples is not None:
            return np.asarray(self.samples, dtype=np.complex64)
        if self.source_path is None:
            raise RuntimeError(f"packet {self.packet_id} has no IQ source")
        return np.fromfile(self.source_path, dtype=np.dtype("<c8"))


class PreparedSync(NamedTuple):
    """FrameSync 使用的 IQ，以及官方 OTA 整数 CFO 居中诊断。"""

    samples: np.ndarray
    result: Any
    initial_result: Any
    coarse_cfo_bin: int
    coarse_cfo_hz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packets",
        type=int,
        default=8,
        help="合成包数量，或官方 OTA 模式下按路径排序后使用的文件上限。",
    )
    parser.add_argument(
        "--official-ota-root",
        type=Path,
        help=(
            "官方 RFSR-OTA 子集根目录；应包含 metadata/*.json 和 "
            "ota/*.cfile。未指定时运行原有合成实验。"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help=(
            "要评估的前端路径。官方 OTA 对比通常只需 "
            "native_1msps official_ota_rfsr。"
        ),
    )
    parser.add_argument(
        "--decoders",
        nargs="+",
        choices=DECODERS,
        default=list(DECODERS),
        help=(
            "要计分的解调器。只测本文 Savaux 时传 --decoders savaux，"
            "可避免无关 GLS/ordinary FFT 计算。"
        ),
    )
    parser.add_argument(
        "--trim-official-guards",
        action="store_true",
        help=(
            "官方 OTA 模式下裁掉大部分零填充 guard，但为流式检测和卷积前端"
            "各保留两条 chirp 上下文；默认完整模式保留原始 guard。"
        ),
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=NOISE_STAGES,
        default=list(NOISE_STAGES),
        help="要评估的噪声位置；clean 基线始终保留。",
    )
    parser.add_argument(
        "--include-direct-250k-baseline",
        "--include-direct-250k-post-baseline",
        dest="include_direct_250k_baseline",
        action="store_true",
        help=(
            "额外在原始 250 kS/s IQ 上以 OS2 独立同步，并评估 clean 与所选"
            "噪声阶段；不插值、不经过 RF-SR。旧 post-baseline 参数名仍可用。"
        ),
    )
    parser.add_argument(
        "--reuse-clean-coarse-cfo",
        action="store_true",
        help=(
            "前置噪声时复用同一录波干净路径估出的整数 CFO，只重新运行最终 "
            "FrameSync；模拟流式接收机保持频率跟踪状态。"
        ),
    )
    parser.add_argument(
        "--reuse-clean-frame-location",
        action="store_true",
        help=(
            "前置噪声时复用干净包的候选帧位置，仅重新运行 gr-lora 风格的 "
            "CFO/STO/SFO 验证；模拟已锁定的流式接收机。"
        ),
    )
    parser.add_argument(
        "--fast-official",
        action="store_true",
        help=(
            "官方 OTA 快速预设：Native 对 OTA-RFSR、Savaux、前置噪声与"
            "增益匹配后置噪声、裁 guard，并复用干净流式同步状态。"
        ),
    )
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=[-18.0, -20.0, -22.0, -24.0],
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--noise-seed", type=int, default=20260731)
    parser.add_argument("--official-decode-trials", type=int, default=2)
    parser.add_argument("--gls-noise-windows", type=int, default=32)
    parser.add_argument("--gls-training-bins", type=int, default=16)
    parser.add_argument("--gls-loading", type=float, default=0.50)
    parser.add_argument("--chunk-input-samples", type=int, default=1_000_000)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch 设备；'auto' 会在 CUDA 可用时自动选择 CUDA。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "results"
        / "official_rfsr_synthetic_chain_20260730.json",
    )
    args = parser.parse_args()
    if args.fast_official:
        if args.official_ota_root is None:
            parser.error("--fast-official requires --official-ota-root")
        args.methods = ["native_1msps", "official_ota_rfsr"]
        args.decoders = ["savaux"]
        args.stages = ["pre_rfsr", "post_framesync_gain_matched"]
        args.trim_official_guards = True
        args.reuse_clean_coarse_cfo = True
        args.reuse_clean_frame_location = True
    if int(args.packets) <= 0:
        parser.error("--packets must be positive")
    if int(args.official_decode_trials) < 0:
        parser.error("--official-decode-trials cannot be negative")
    if int(args.gls_noise_windows) < 2:
        parser.error("--gls-noise-windows must be at least two")
    if int(args.gls_training_bins) <= 0:
        parser.error("--gls-training-bins must be positive")
    if not args.snrs:
        parser.error("--snrs must contain at least one value")
    if not args.methods:
        parser.error("--methods must contain at least one method")
    if not args.decoders:
        parser.error("--decoders must contain at least one decoder")
    if not args.stages:
        parser.error("--stages must contain at least one stage")
    return args


def _active_slice(
    samples: np.ndarray,
    *,
    leading_guard: int,
    trailing_guard: int,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex64)
    start = int(leading_guard)
    stop = int(values.size) - int(trailing_guard)
    if start < 0 or int(trailing_guard) < 0 or stop <= start:
        raise ValueError("guard intervals leave no active packet samples")
    return values[start:stop]


def _finite_positive_power(samples: np.ndarray, *, label: str) -> float:
    power = float(np.mean(np.abs(samples).astype(np.float64) ** 2))
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError(f"{label} power must be positive and finite")
    return power


def _legacy_symbol_to_receiver_value(value: float, reduced_rate: bool) -> int:
    """把官方旧 PHY 的 k 值转换成 gr-lora_sdr 接收机硬判决值。"""

    # 旧调制器 lora_chirp(+1, k) 与 gr-lora_sdr 的 symbol_id=(K-k)%K
    # 生成同一条 chirp；接收机随后执行 (FFT-bin-1)%K，并在 header/LDRO
    # 条件下除以四。
    raw_symbol_id = (N_BINS - int(round(float(value)))) % N_BINS
    divisor = 4 if bool(reduced_rate) else 1
    return int(((raw_symbol_id - 1) % N_BINS) // divisor)


def _official_expected_symbols(metadata: dict[str, Any]) -> dict[str, list[int]]:
    """按官方数据生成器的 4-byte 私有头语义重建 symbol 真值。"""

    payload = np.asarray(metadata["payload"], dtype=np.uint8)
    if payload.ndim != 1 or payload.size != PAYLOAD_BYTES:
        raise ValueError(
            f"official OTA payload must contain {PAYLOAD_BYTES} bytes, "
            f"got shape {payload.shape}"
        )
    sf = int(metadata["sf"])
    cr = int(metadata["cr"])
    crc = int(metadata["enable_crc"])
    implicit = int(metadata["implicit_header"])
    if sf != SF or cr != CR or crc != 1 or implicit != 0:
        raise ValueError(
            "official OTA PHY must use SF12, CR4, CRC, and explicit header"
        )

    length = np.uint16(4 + payload.size)
    _, header_payload_bits = lora_header_init(sf, implicit)
    payload_bits, payload_symbol_count = lora_payload_init(
        sf,
        length,
        crc,
        cr,
        header_payload_bits,
        int(metadata["dst"]),
        int(metadata["src"]),
        int(metadata["seqn"]),
        payload,
    )
    header_ids, payload_offset = lora_header(
        sf,
        length,
        cr,
        crc,
        payload_bits,
        0,
    )
    payload_ids = lora_payload(
        sf,
        cr,
        payload_symbol_count,
        payload_bits,
        payload_offset,
    )
    return {
        "header": [
            _legacy_symbol_to_receiver_value(value, True) for value in header_ids
        ],
        "payload": [
            _legacy_symbol_to_receiver_value(value, LDRO) for value in payload_ids
        ],
    }


def _official_metadata_file_snr(
    metadata: dict[str, Any], relative_path: str
) -> float:
    matches = {
        float(snr_db)
        for value, snr_db in metadata["files"]
        if str(value) == str(relative_path)
    }
    if len(matches) != 1:
        raise ValueError(
            f"metadata must contain exactly one SNR for {relative_path}, got {matches}"
        )
    return matches.pop()


def load_official_ota_packets(root: Path, limit: int) -> list[EvaluationPacket]:
    """加载并严格校验一个已下载的官方 RFSR-OTA 子集。"""

    dataset_root = Path(root).expanduser().resolve()
    metadata_paths = sorted((dataset_root / "metadata").glob("*.json"))
    ota_paths = sorted((dataset_root / "ota").glob("*.cfile"))
    if not metadata_paths:
        raise FileNotFoundError(f"no metadata/*.json under {dataset_root}")
    if not ota_paths:
        raise FileNotFoundError(
            f"no ota/*.cfile under {dataset_root}; a Dataverse metadata-only "
            "archive is not sufficient"
        )

    metadata_by_relative_path: dict[str, tuple[Path, dict[str, Any]]] = {}
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for relative_path, _ in metadata.get("files", []):
            key = str(relative_path)
            previous = metadata_by_relative_path.get(key)
            if previous is not None and previous[0] != metadata_path:
                raise ValueError(f"OTA path appears in multiple metadata files: {key}")
            metadata_by_relative_path[key] = (metadata_path, metadata)

    packets: list[EvaluationPacket] = []
    for ota_path in ota_paths[: int(limit)]:
        relative_path = ota_path.relative_to(dataset_root).as_posix()
        match = metadata_by_relative_path.get(relative_path)
        if match is None:
            raise ValueError(f"official OTA file has no metadata reference: {ota_path}")
        metadata_path, metadata = match
        if int(round(float(metadata["sample_rate"]))) != HIGH_RATE_HZ:
            raise ValueError(f"official OTA file is not 2 MS/s: {ota_path}")
        if int(round(float(metadata["bw"]))) != BW_HZ:
            raise ValueError(f"official OTA file is not BW125: {ota_path}")
        if int(metadata["preamble_bits"]) != PREAMBLE_SYMBOLS:
            raise ValueError(f"unexpected official OTA preamble: {ota_path}")
        expected_samples = (
            int(metadata["num_samples"]) + 2 * OFFICIAL_OTA_GUARD_HIGH
        )
        if ota_path.stat().st_size != expected_samples * np.dtype("<c8").itemsize:
            raise ValueError(
                f"official OTA byte length mismatch for {ota_path}: "
                f"expected {expected_samples * 8}, got {ota_path.stat().st_size}"
            )
        mapped = np.memmap(ota_path, dtype=np.dtype("<c8"), mode="r")
        active = _active_slice(
            mapped,
            leading_guard=OFFICIAL_OTA_GUARD_HIGH,
            trailing_guard=OFFICIAL_OTA_GUARD_HIGH,
        )
        reference_power = _finite_positive_power(
            active, label=f"official OTA {ota_path.name} active"
        )
        del active, mapped
        packets.append(
            EvaluationPacket(
                packet_id=ota_path.stem,
                expected=_official_expected_symbols(metadata),
                leading_guard_high=OFFICIAL_OTA_GUARD_HIGH,
                trailing_guard_high=OFFICIAL_OTA_GUARD_HIGH,
                center_frequency_hz=float(metadata["center_freq"]),
                reference_power=reference_power,
                source_snr_db=_official_metadata_file_snr(
                    metadata, relative_path
                ),
                source_path=ota_path,
                metadata_path=metadata_path,
            )
        )
    if not packets:
        raise RuntimeError("official OTA subset contains no selected packets")
    return packets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_device(requested: str) -> str:
    import torch

    if str(requested) != "auto":
        return str(requested)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _frontend(checkpoint_name: str, device: str, chunk: int) -> RFSuperResolutionFrontend:
    return RFSuperResolutionFrontend(
        RFSRFrontendConfig(
            repo_root=RFSR_ROOT,
            checkpoint_name=checkpoint_name,
            device=str(device),
            chunk_input_samples=int(chunk),
            overlap_input_samples=68,
        )
    )


def complex_awgn(
    count: int,
    noise_power: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成圆对称复 AWGN，使 E[|n|^2] 等于 ``noise_power``。"""

    sigma = math.sqrt(max(float(noise_power), 0.0) / 2.0)
    return np.asarray(
        sigma * (rng.standard_normal(int(count)) + 1j * rng.standard_normal(int(count))),
        dtype=np.complex64,
    )


def snr_noise_power(reference_power: float, snr_db: float) -> float:
    return float(reference_power) / (10.0 ** (float(snr_db) / 10.0))


def measure_decimated_snr(
    clean_high_rate: np.ndarray,
    noisy_high_rate: np.ndarray,
    *,
    decimation: int,
    leading_silence_high_rate: int,
    trailing_silence_high_rate: int,
    measured_sample_rate_hz: int,
) -> dict[str, float | int]:
    """在抽取后的有效包区间测量信号功率、噪声功率和 SNR。"""

    clean = np.asarray(clean_high_rate, dtype=np.complex64)
    noisy = np.asarray(noisy_high_rate, dtype=np.complex64)
    if clean.ndim != 1 or noisy.ndim != 1 or clean.shape != noisy.shape:
        raise ValueError("clean and noisy IQ must be equally sized one-dimensional arrays")
    factor = int(decimation)
    leading = int(leading_silence_high_rate)
    trailing = int(trailing_silence_high_rate)
    if factor <= 0:
        raise ValueError("decimation must be positive")
    if leading < 0 or trailing < 0 or leading + trailing >= clean.size:
        raise ValueError("silence intervals leave no active packet samples")

    clean_low = clean[::factor]
    noisy_low = noisy[::factor]
    active_high_stop = int(clean.size) - trailing
    # 只保留满足 leading <= k*factor < active_high_stop 的抽取后索引 k。
    active_start = (leading + factor - 1) // factor
    active_stop = (active_high_stop + factor - 1) // factor
    clean_active = clean_low[active_start:active_stop]
    noise_active = noisy_low[active_start:active_stop] - clean_active
    if clean_active.size == 0:
        raise ValueError("decimated active packet interval is empty")

    signal_power = float(np.mean(np.abs(clean_active).astype(np.float64) ** 2))
    measured_noise_power = float(
        np.mean(np.abs(noise_active).astype(np.float64) ** 2)
    )
    if signal_power <= 0.0 or measured_noise_power <= 0.0:
        raise ValueError("measured signal and noise powers must be positive")
    return {
        "sample_rate_hz": int(measured_sample_rate_hz),
        "active_sample_count": int(clean_active.size),
        "signal_power": signal_power,
        "noise_power": measured_noise_power,
        "snr_db": float(10.0 * math.log10(signal_power / measured_noise_power)),
    }


def _method_outputs(
    high_rate_samples: np.ndarray,
    synthetic_frontend: RFSuperResolutionFrontend,
    ota_frontend: RFSuperResolutionFrontend,
    snr_db: float,
    methods: Iterable[str] = METHODS,
) -> dict[str, np.ndarray]:
    high = np.asarray(high_rate_samples, dtype=np.complex64)
    low = high[:: HIGH_RATE_HZ // LOW_RATE_HZ]
    selected = tuple(dict.fromkeys(str(method) for method in methods))
    outputs: dict[str, np.ndarray] = {}
    if "native_1msps" in selected:
        outputs["native_1msps"] = np.asarray(
            high[:: HIGH_RATE_HZ // OUTPUT_RATE_HZ], dtype=np.complex64
        )
    if "official_interpolation" in selected:
        outputs["official_interpolation"] = synthetic_frontend.interpolate(low)
    if "official_synthetic_rfsr" in selected:
        outputs["official_synthetic_rfsr"] = synthetic_frontend.enhance(
            low, snr_db=snr_db
        )
    if "official_ota_rfsr" in selected:
        outputs["official_ota_rfsr"] = ota_frontend.enhance(low, snr_db=snr_db)
    if set(outputs) != set(selected):
        raise ValueError(f"unsupported method selection: {selected}")
    return outputs


def _expected_demod_symbols(symbol_ids: Iterable[int], reduced_rate: bool) -> list[int]:
    divisor = 4 if bool(reduced_rate) else 1
    return [((int(value) - 1) % N_BINS) // divisor for value in symbol_ids]


def _expected_symbols(encoded: Any) -> dict[str, list[int]]:
    return {
        "header": _expected_demod_symbols(encoded.header_symbol_ids, True),
        "payload": _expected_demod_symbols(encoded.payload_symbol_ids, LDRO),
    }


def _sync_config(
    center_frequency_hz: float = 915_000_000.0,
    sample_rate_hz: float = OUTPUT_RATE_HZ,
) -> SinglePacketSyncConfig:
    return SinglePacketSyncConfig(
        sf=SF,
        bw_hz=BW_HZ,
        sample_rate_hz=float(sample_rate_hz),
        center_frequency_hz=float(center_frequency_hz),
        preamble_symbols=PREAMBLE_SYMBOLS,
        sync_word=SYNC_WORD,
        scan_chirps=24,
    )


def _sync_report(result: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": str(result.status),
        "synchronized": bool(result.synchronized),
        "event_count": int(result.event_count),
        "error": result.error,
    }
    if result.frame_sync is not None:
        frame_sync = result.frame_sync
        report.update(
            {
                "header_start_sample": int(frame_sync.fine_payload_start_sample),
                "cfo_int_bins": int(frame_sync.cfo_int_est),
                "cfo_frac_bins": float(frame_sync.cfo_frac_est),
                "sfo_chips_per_symbol": float(frame_sync.sfo_hat),
                "sto_fractional_chips": float(frame_sync.sto_frac_used),
            }
        )
    return report


def _center_integer_cfo(
    samples: np.ndarray,
    config: SinglePacketSyncConfig,
    coarse_bin: int,
) -> tuple[np.ndarray, float]:
    values = np.asarray(samples, dtype=np.complex64)
    fft_length = int(config.chirp_samples)
    coarse_bin = int(coarse_bin) % fft_length
    if coarse_bin > fft_length // 2:
        coarse_bin -= fft_length
    if coarse_bin == 0:
        return values, 0.0
    coarse_hz = float(
        coarse_bin * float(config.sample_rate_hz) / float(fft_length)
    )
    sample_index = np.arange(values.size, dtype=np.float64)
    rotation = np.exp(
        -2j
        * np.pi
        * coarse_hz
        * sample_index
        / float(config.sample_rate_hz)
    )
    centered = np.asarray(values * rotation, dtype=np.complex64)
    return centered, coarse_hz


def prepare_samples_and_sync(
    samples: np.ndarray,
    config: SinglePacketSyncConfig,
    *,
    coarse_cfo_centering: bool,
) -> PreparedSync:
    """先检测官方 OTA 的大整数 CFO，去旋转后再执行最终 FrameSync。"""

    values = np.asarray(samples, dtype=np.complex64)
    initial = run_single_packet_sync(values, config)
    if not bool(coarse_cfo_centering) or initial.frame_location is None:
        return PreparedSync(values, initial, initial, 0, 0.0)

    fft_length = int(config.chirp_samples)
    coarse_bin = int(initial.frame_location.preamble_ref_bin) % fft_length
    if coarse_bin > fft_length // 2:
        coarse_bin -= fft_length
    centered, coarse_hz = _center_integer_cfo(values, config, coarse_bin)
    result = run_single_packet_sync(centered, config)
    return PreparedSync(centered, result, initial, coarse_bin, coarse_hz)


def prepare_samples_with_reused_coarse_cfo(
    samples: np.ndarray,
    config: SinglePacketSyncConfig,
    clean_preparation: PreparedSync,
) -> PreparedSync:
    """复用流式接收机已经获得的整数 CFO，再独立执行含噪最终同步。"""

    coarse_bin = int(clean_preparation.coarse_cfo_bin)
    centered, coarse_hz = _center_integer_cfo(samples, config, coarse_bin)
    result = run_single_packet_sync(centered, config)
    return PreparedSync(
        centered,
        result,
        clean_preparation.initial_result,
        coarse_bin,
        coarse_hz,
    )


def prepare_samples_with_tracked_sync(
    samples: np.ndarray,
    config: SinglePacketSyncConfig,
    clean_preparation: PreparedSync,
) -> PreparedSync:
    """复用已锁定帧候选，只重估含噪 IQ 的精同步状态。"""

    clean_result = clean_preparation.result
    if (
        clean_result.frame_location is None
        or not clean_result.frame_location.valid
    ):
        centered, coarse_hz = _center_integer_cfo(
            samples,
            config,
            int(clean_preparation.coarse_cfo_bin),
        )
        return PreparedSync(
            centered,
            clean_result,
            clean_preparation.initial_result,
            int(clean_preparation.coarse_cfo_bin),
            coarse_hz,
        )

    coarse_bin = int(clean_preparation.coarse_cfo_bin)
    centered, coarse_hz = _center_integer_cfo(samples, config, coarse_bin)
    detector = PreambleDetectorConfig(
        sf=int(config.sf),
        bw=float(config.bw_hz),
        samp_rate=float(config.sample_rate_hz),
        win_chirps=int(config.detection_chirps),
        hop_samples=int(config.chirp_samples),
        min_periodic_peaks=max(
            2,
            int(config.preamble_symbols) - int(config.detection_chirps) + 1,
        ),
        bin_tol=int(config.bin_tolerance),
    )
    frame_sync = run_grlora_frame_sync_validation(
        centered,
        clean_result.frame_location,
        detector,
        float(config.preamble_symbols),
        int(config.sync_word),
        bin0_tol=int(config.frame_sync_bin0_tolerance),
        center_freq=float(config.center_frequency_hz),
    )
    status = "ok" if frame_sync.valid else "sync_invalid"
    tracked_result = replace(
        clean_result,
        status=status,
        event_count=1,
        frame_sync=frame_sync,
        error=None,
    )
    return PreparedSync(
        centered,
        tracked_result,
        clean_preparation.initial_result,
        coarse_bin,
        coarse_hz,
    )


def _empty_decoder_score(symbol_count: int) -> dict[str, Any]:
    return {
        "symbol_count": int(symbol_count),
        "symbol_errors": int(symbol_count),
        "ser": 1.0,
        "median_peak_margin_db": None,
    }


def _score_values(
    actual: list[int],
    expected: list[int],
    margins: list[float],
) -> dict[str, Any]:
    errors = sum(int(left != right) for left, right in zip(actual, expected))
    errors += max(0, len(expected) - len(actual))
    return {
        "symbol_count": int(len(expected)),
        "symbol_errors": int(errors),
        "ser": float(errors / len(expected)) if expected else None,
        "median_peak_margin_db": (
            float(np.median(np.asarray(margins, dtype=np.float64))) if margins else None
        ),
    }


def _batch_savaux_spectra(
    samples: np.ndarray,
    starts: list[int],
    cfo_int: int,
    cfo_frac: float,
    os_factor: int = OS_FACTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """用一次批量 FFT 调度计算整包所有符号的论文 Eq.36/37。"""

    values = np.asarray(samples, dtype=np.complex64)
    os_value = int(os_factor)
    if os_value <= 0:
        raise ValueError("os_factor must be positive")
    symbol_samples = N_BINS * os_value
    symbols = np.stack(
        [values[int(start) : int(start) + symbol_samples] for start in starts],
        axis=0,
    )
    if symbols.shape != (len(starts), symbol_samples):
        raise ValueError("one or more Savaux symbols exceed the input sample range")

    sample_index = np.arange(symbol_samples, dtype=np.float64)
    from weak_decoder.chirp import build_upchirp

    reference = build_upchirp(
        sf=SF,
        symbol_id=int(cfo_int),
        os_factor=os_value,
    )
    fractional = np.exp(
        -2j * np.pi * float(cfo_frac) * sample_index / float(symbol_samples)
    )
    downchirp = np.asarray(np.conjugate(reference) * fractional, dtype=np.complex64)
    # 连续 CFO 的公共相位对每个符号只是一个标量，在 |Eq.37|^2 和 GLS 投影的
    # 模平方中都会抵消，因此这里无需显式应用。
    dechirped = np.asarray(symbols * downchirp[None, :], dtype=np.complex64)
    branch_samples = dechirped.reshape(len(starts), N_BINS, os_value)
    spectra = np.fft.fft(branch_samples, axis=1) / math.sqrt(float(N_BINS))

    if os_value > 1:
        # 复用论文实现中精确的 chirp-z 尾部校正，并把所有符号的 q>0 branch
        # 合并到 branch-count 轴上进行批处理。
        tail_input = np.transpose(branch_samples[:, :, 1:], (1, 0, 2)).reshape(
            N_BINS, len(starts) * (os_value - 1)
        )
        tails = _wrapped_tail_dft_batch(tail_input).reshape(
            N_BINS, len(starts), os_value - 1
        )
        tails = np.transpose(tails, (1, 0, 2))
        wrap_phases = np.exp(
            2j * np.pi * np.arange(1, os_value, dtype=np.float64) / os_value
        )
        spectra[:, :, 1:] += (
            (wrap_phases - 1.0)[None, None, :] * tails
        ) / math.sqrt(float(N_BINS))

    bins = np.arange(N_BINS, dtype=np.float64)[None, :, None]
    branches = np.arange(os_value, dtype=np.float64)[None, None, :]
    alignment = np.exp(
        -2j * np.pi * bins * branches / float(N_BINS * os_value)
    )
    aligned = np.asarray(spectra * alignment, dtype=np.complex64)
    combined = np.asarray(np.sum(aligned, axis=2), dtype=np.complex64)
    return combined, np.asarray(spectra, dtype=np.complex64)


def evaluate_decoders(
    samples: np.ndarray,
    sync_result: Any,
    expected: dict[str, list[int]],
    noise_model: BranchNoiseModel,
    decoder_names: Iterable[str] = DECODERS,
    os_factor: int = OS_FACTOR,
) -> dict[str, dict[str, Any]]:
    """在共享同一份 FrameSync 结果后，对全部预期符号计分。"""

    selected_decoders = tuple(
        dict.fromkeys(str(decoder) for decoder in decoder_names)
    )
    if not set(selected_decoders).issubset(DECODERS):
        raise ValueError(f"unsupported decoder selection: {selected_decoders}")
    os_value = int(os_factor)
    if os_value <= 0:
        raise ValueError("os_factor must be positive")
    expected_values = list(expected["header"]) + list(expected["payload"])
    count = len(expected_values)
    if not sync_result.synchronized or sync_result.frame_sync is None:
        return {
            name: _empty_decoder_score(count) for name in selected_decoders
        }

    frame_sync = sync_result.frame_sync
    try:
        ordinary = demod_symbol_sequence(
            samples=np.asarray(samples, dtype=np.complex64),
            header_start_sample=int(frame_sync.fine_payload_start_sample),
            sf=SF,
            os_factor=os_value,
            cfo_int=int(frame_sync.cfo_int_est),
            cfo_frac=float(frame_sync.cfo_frac_est),
            sfo_hat=float(frame_sync.sfo_hat),
            sfo_cum_initial=float(frame_sync.sfo_cum_initial),
            header_count=len(expected["header"]),
            payload_count=len(expected["payload"]),
            payload_ldro=LDRO,
            cfo_correction_mode="continuous",
        )
    except ValueError:
        return {
            name: _empty_decoder_score(count) for name in selected_decoders
        }

    ordinary_values = [int(item.symbol_value) for item in ordinary]
    ordinary_margins = [float(item.peak_margin_db) for item in ordinary]
    try:
        combined_batch, branch_batch = _batch_savaux_spectra(
            samples,
            [int(item.start_sample) + os_value // 2 for item in ordinary],
            cfo_int=int(frame_sync.cfo_int_est),
            cfo_frac=float(frame_sync.cfo_frac_est),
            os_factor=os_value,
        )
    except ValueError:
        scores = {
            "ordinary_fft": _score_values(
                ordinary_values, expected_values, ordinary_margins
            )
        }
        return {
            name: scores.get(name, _empty_decoder_score(count))
            for name in selected_decoders
        }

    savaux_values: list[int] = []
    gls_values: list[int] = []
    savaux_margins: list[float] = []
    gls_margins: list[float] = []
    for index in range(len(ordinary)):
        is_header = index < len(expected["header"])
        savaux_power = np.abs(combined_batch[index]).astype(np.float64) ** 2
        savaux_bin = int(np.argmax(savaux_power))
        savaux_second = float(np.partition(savaux_power, -2)[-2])
        savaux_margins.append(
            float(
                10.0
                * math.log10(
                    (float(savaux_power[savaux_bin]) + 1e-30)
                    / (savaux_second + 1e-30)
                )
            )
        )
        savaux_values.append(
            bin_to_grlora_symbol(
                savaux_bin,
                sf=SF,
                is_header=is_header,
                ldro=bool(LDRO and not is_header),
            )
        )
        if "savaux_gls" in selected_decoders:
            gls = branch_gls_scores(
                tuple(branch_batch[index, :, q] for q in range(os_value)),
                os_factor=os_value,
                noise_model=noise_model,
                top_l=8,
            )
            gls_order = np.argsort(gls.scores)[::-1]
            gls_second = float(gls.scores[gls_order[1]])
            gls_margins.append(
                float(
                    10.0
                    * math.log10(
                        (float(gls.scores[gls.selected_bin]) + 1e-30)
                        / (gls_second + 1e-30)
                    )
                )
            )
            gls_values.append(
                bin_to_grlora_symbol(
                    int(gls.selected_bin),
                    sf=SF,
                    is_header=is_header,
                    ldro=bool(LDRO and not is_header),
                )
            )

    scores = {
        "ordinary_fft": _score_values(
            ordinary_values, expected_values, ordinary_margins
        ),
        "savaux": _score_values(savaux_values, expected_values, savaux_margins),
    }
    if "savaux_gls" in selected_decoders:
        scores["savaux_gls"] = _score_values(
            gls_values, expected_values, gls_margins
        )
    return {name: scores[name] for name in selected_decoders}


def _covariance_diagnostics(model: BranchNoiseModel) -> dict[str, Any]:
    covariance = np.asarray(model.covariance, dtype=np.complex128)
    if covariance.ndim != 2:
        raise ValueError("this audit expects a pooled branch covariance")
    diagonal = np.maximum(np.real(np.diag(covariance)), 1e-30)
    correlation = covariance / np.sqrt(diagonal[:, None] * diagonal[None, :])
    mask = ~np.eye(correlation.shape[0], dtype=bool)
    return {
        "snapshot_count": int(model.snapshot_count),
        "mean_offdiagonal_abs_correlation": float(np.mean(np.abs(correlation[mask]))),
        "max_offdiagonal_abs_correlation": float(np.max(np.abs(correlation[mask]))),
        "condition_number": float(np.linalg.cond(covariance)),
    }


def estimate_pre_rfsr_noise_models(
    snr_db: float,
    reference_power: float,
    synthetic_frontend: RFSuperResolutionFrontend,
    ota_frontend: RFSuperResolutionFrontend,
    args: argparse.Namespace,
    methods: Iterable[str] = METHODS,
) -> tuple[dict[str, BranchNoiseModel], dict[str, dict[str, Any]]]:
    """用留出的纯噪声 IQ 为每种前端估计对应的 branch 协方差。"""

    symbol_samples = N_BINS * OS_FACTOR
    output_count = int(args.gls_noise_windows) * symbol_samples
    high_count = output_count * (HIGH_RATE_HZ // OUTPUT_RATE_HZ)
    seed = int(args.noise_seed) + 10_000 + int(round((float(snr_db) + 100.0) * 10.0))
    rng = np.random.default_rng(seed)
    noise_power = snr_noise_power(reference_power, snr_db)
    high_noise = complex_awgn(high_count, noise_power, rng)
    views = _method_outputs(
        high_noise,
        synthetic_frontend=synthetic_frontend,
        ota_frontend=ota_frontend,
        snr_db=snr_db,
        methods=methods,
    )
    training_bins = tuple(
        int(value)
        for value in np.linspace(
            0,
            N_BINS,
            min(int(args.gls_training_bins), N_BINS),
            endpoint=False,
        )
    )
    models: dict[str, BranchNoiseModel] = {}
    reports: dict[str, dict[str, Any]] = {}
    for name, values in views.items():
        windows = np.asarray(values, dtype=np.complex64).reshape(
            int(args.gls_noise_windows), symbol_samples
        )
        model = estimate_branch_noise_model(
            windows,
            sf=SF,
            os_factor=OS_FACTOR,
            training_bins=training_bins,
            diagonal_loading=float(args.gls_loading),
            covariance_mode="pooled",
        )
        models[name] = model
        reports[name] = _covariance_diagnostics(model)
    return models, reports


def _active_power(
    samples: np.ndarray,
    *,
    leading_guard_high: int = LEADING_SILENCE_HIGH,
    trailing_guard_high: int = TRAILING_SILENCE_HIGH,
    sample_rate_hz: int = OUTPUT_RATE_HZ,
) -> float:
    rate = int(sample_rate_hz)
    if rate <= 0 or HIGH_RATE_HZ % rate != 0:
        raise ValueError("sample_rate_hz must be a positive divisor of HIGH_RATE_HZ")
    factor = HIGH_RATE_HZ // rate
    leading = (int(leading_guard_high) + factor - 1) // factor
    active_high_stop = int(np.asarray(samples).size) * factor - int(
        trailing_guard_high
    )
    stop = (active_high_stop + factor - 1) // factor
    active = np.asarray(samples, dtype=np.complex64)[leading:stop]
    return _finite_positive_power(active, label="frontend active output")


def _row(
    *,
    stage: str,
    snr_db: float | None,
    packet_index: int,
    method: str,
    sync_result: Any,
    scores: dict[str, dict[str, Any]],
    input_snr_measurement: dict[str, float | int] | None = None,
    packet_id: str | None = None,
    source_snr_db: float | None = None,
    added_noise_power: float | None = None,
    sync_preparation: PreparedSync | None = None,
) -> dict[str, Any]:
    return {
        "stage": str(stage),
        "target_snr_db": None if snr_db is None else float(snr_db),
        "snr_db": None if snr_db is None else float(snr_db),
        "input_snr_measurement": input_snr_measurement,
        "packet_index": int(packet_index),
        "packet_id": packet_id,
        "source_snr_db": (
            None if source_snr_db is None else float(source_snr_db)
        ),
        "added_noise_power": (
            None if added_noise_power is None else float(added_noise_power)
        ),
        "coarse_cfo_centering": (
            None
            if sync_preparation is None
            else {
                "coarse_cfo_bin": int(sync_preparation.coarse_cfo_bin),
                "coarse_cfo_hz": float(sync_preparation.coarse_cfo_hz),
                "initial_sync": _sync_report(sync_preparation.initial_result),
            }
        ),
        "method": str(method),
        "sync": _sync_report(sync_result),
        "decoders": scores,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["snr_db"], row["method"])].append(row)

    summary: list[dict[str, Any]] = []
    for (stage, snr_db, method), items in sorted(
        groups.items(),
        key=lambda value: (
            str(value[0][0]),
            -math.inf if value[0][1] is None else float(value[0][1]),
            str(value[0][2]),
        ),
    ):
        synchronized = [item for item in items if bool(item["sync"]["synchronized"])]
        measured_input_snrs = [
            float(item["input_snr_measurement"]["snr_db"])
            for item in items
            if item.get("input_snr_measurement") is not None
        ]
        measured_input_snr_summary = (
            None
            if not measured_input_snrs
            else {
                "sample_rate_hz": LOW_RATE_HZ,
                "measurement_count": len(measured_input_snrs),
                "median_snr_db": float(np.median(measured_input_snrs)),
                "min_snr_db": float(np.min(measured_input_snrs)),
                "max_snr_db": float(np.max(measured_input_snrs)),
            }
        )
        decoder_summary: dict[str, Any] = {}
        decoder_names = tuple(items[0]["decoders"])
        if any(tuple(item["decoders"]) != decoder_names for item in items):
            raise ValueError("grouped rows use inconsistent decoder selections")
        for decoder in decoder_names:
            all_scores = [item["decoders"][decoder] for item in items]
            conditional_scores = [item["decoders"][decoder] for item in synchronized]
            all_count = sum(int(score["symbol_count"]) for score in all_scores)
            all_errors = sum(int(score["symbol_errors"]) for score in all_scores)
            conditional_count = sum(
                int(score["symbol_count"]) for score in conditional_scores
            )
            conditional_errors = sum(
                int(score["symbol_errors"]) for score in conditional_scores
            )
            margins = [
                float(score["median_peak_margin_db"])
                for score in conditional_scores
                if score["median_peak_margin_db"] is not None
            ]
            decoder_summary[decoder] = {
                "end_to_end_symbol_count": int(all_count),
                "end_to_end_symbol_errors": int(all_errors),
                "end_to_end_ser": (
                    None if all_count == 0 else float(all_errors / all_count)
                ),
                "conditional_symbol_count": int(conditional_count),
                "conditional_symbol_errors": int(conditional_errors),
                "conditional_ser": (
                    None
                    if conditional_count == 0
                    else float(conditional_errors / conditional_count)
                ),
                "median_packet_peak_margin_db": (
                    None if not margins else float(np.median(margins))
                ),
            }
        summary.append(
            {
                "stage": stage,
                "target_snr_db": snr_db,
                "snr_db": snr_db,
                "input_snr_measurement": measured_input_snr_summary,
                "method": method,
                "packet_count": len(items),
                "synchronized_packets": len(synchronized),
                "sync_success_rate": float(len(synchronized) / len(items)),
                "decoders": decoder_summary,
            }
        )
    return summary


def _official_decode_reproduction(
    synthetic_frontend: RFSuperResolutionFrontend,
    snrs: list[float],
    trials: int,
    seed: int,
) -> list[dict[str, Any]]:
    """使用确定性 payload，严格复现上游示例的输入输出契约。"""

    results: list[dict[str, Any]] = []
    for snr_index, snr_db in enumerate(snrs):
        successes = 0
        rows: list[dict[str, Any]] = []
        for trial in range(int(trials)):
            trial_seed = int(seed) + 50_000 + snr_index * 1_000 + trial
            np.random.seed(trial_seed)
            payload = np.random.randint(0, 255, size=PAYLOAD_BYTES, dtype=np.uint8)
            clean_signal = encode(
                915e6,
                SF,
                BW_HZ,
                payload,
                HIGH_RATE_HZ,
                0,
                1,
                trial % 256,
                CR,
                1,
                0,
                PREAMBLE_SYMBOLS,
            )
            signal = awgn(clean_signal, float(snr_db))
            input_snr_measurement = measure_decimated_snr(
                clean_signal,
                signal,
                decimation=HIGH_RATE_HZ // LOW_RATE_HZ,
                leading_silence_high_rate=LEADING_SILENCE_HIGH,
                trailing_silence_high_rate=0,
                measured_sample_rate_hz=LOW_RATE_HZ,
            )
            output = synthetic_frontend.enhance(signal[::8], snr_db=float(snr_db))
            try:
                decoded = decode(output, SF, BW_HZ, OUTPUT_RATE_HZ)
            except Exception as exc:  # 上游解码器在部分失败包上会直接抛出异常。
                decoded = []
                decode_error = f"{type(exc).__name__}: {exc}"
            else:
                decode_error = None
            valid = bool(
                len(decoded) == 1
                and int(decoded[0].hdr_ok) == 1
                and int(decoded[0].crc_ok) == 1
                and int(decoded[0].src) == 0
                and int(decoded[0].dst) == 1
                and int(decoded[0].seqn) == trial % 256
                and np.array_equal(np.asarray(decoded[0].payload), payload)
            )
            successes += int(valid)
            rows.append(
                {
                    "trial": trial,
                    "seed": trial_seed,
                    "decoded_packet_count": len(decoded),
                    "payload_crc_match": valid,
                    "decode_error": decode_error,
                    "target_snr_db": float(snr_db),
                    "input_snr_measurement": input_snr_measurement,
                }
            )
        results.append(
            {
                "snr_db": float(snr_db),
                "trials": int(trials),
                "successes": int(successes),
                "packet_success_rate": (
                    None if int(trials) == 0 else float(successes / int(trials))
                ),
                "rows": rows,
            }
        )
    return results


def _build_synthetic_packets(args: argparse.Namespace) -> list[EvaluationPacket]:
    rng = np.random.default_rng(int(args.seed))
    encoded_packets = [
        encode_raw_phy(
            rng.integers(0, 256, size=PAYLOAD_BYTES, dtype=np.uint8),
            HIGH_RATE_HZ,
            SF=SF,
            BW=BW_HZ,
            cr=CR,
            enable_crc=1,
            implicit_header=0,
            preamble_bits=PREAMBLE_SYMBOLS,
            sync_word=SYNC_WORD,
            ldro=LDRO,
            crc_mode="grlora",
            leading_silence_samples=LEADING_SILENCE_HIGH,
            trailing_silence_samples=TRAILING_SILENCE_HIGH,
        )
        for _ in range(int(args.packets))
    ]
    packets: list[EvaluationPacket] = []
    for index, encoded in enumerate(encoded_packets):
        active = _active_slice(
            encoded.samples,
            leading_guard=LEADING_SILENCE_HIGH,
            trailing_guard=TRAILING_SILENCE_HIGH,
        )
        packets.append(
            EvaluationPacket(
                packet_id=f"synthetic_{index:06d}",
                expected=_expected_symbols(encoded),
                leading_guard_high=LEADING_SILENCE_HIGH,
                trailing_guard_high=TRAILING_SILENCE_HIGH,
                center_frequency_hz=915_000_000.0,
                reference_power=_finite_positive_power(
                    active, label=f"synthetic packet {index}"
                ),
                source_snr_db=None,
                samples=np.asarray(encoded.samples, dtype=np.complex64),
            )
        )
    return packets


def main() -> None:
    started_at = perf_counter()
    args = parse_args()
    device = _resolve_device(args.device)
    official_ota_mode = args.official_ota_root is not None
    selected_methods = tuple(dict.fromkeys(str(value) for value in args.methods))
    selected_decoders = tuple(dict.fromkeys(str(value) for value in args.decoders))
    selected_stages = tuple(dict.fromkeys(str(value) for value in args.stages))
    packets = (
        load_official_ota_packets(args.official_ota_root, int(args.packets))
        if official_ota_mode
        else _build_synthetic_packets(args)
    )
    reference_power = float(
        np.median([packet.reference_power for packet in packets])
    )
    print(
        f"device={device} mode={'official_ota' if official_ota_mode else 'synthetic'} "
        f"packets={len(packets)}",
        flush=True,
    )
    synthetic_frontend = _frontend(
        DEFAULT_SYNTHETIC_CHECKPOINT, device, int(args.chunk_input_samples)
    )
    ota_frontend = _frontend(
        DEFAULT_OTA_CHECKPOINT, device, int(args.chunk_input_samples)
    )

    decode_reproduction = (
        []
        if official_ota_mode
        else _official_decode_reproduction(
            synthetic_frontend,
            [float(value) for value in args.snrs],
            int(args.official_decode_trials),
            int(args.seed),
        )
    )

    pre_noise_models: dict[float, dict[str, BranchNoiseModel]] = {}
    covariance_reports: dict[str, dict[str, dict[str, Any]]] = {}
    post_noise_model = identity_branch_noise_model(OS_FACTOR)
    direct_250k_os_factor = LOW_RATE_HZ // BW_HZ
    direct_250k_noise_model = identity_branch_noise_model(
        direct_250k_os_factor
    )
    if "savaux_gls" in selected_decoders and "pre_rfsr" in selected_stages:
        print("estimating pre-RFSR GLS noise models", flush=True)
        for snr_db in args.snrs:
            models, report = estimate_pre_rfsr_noise_models(
                float(snr_db),
                reference_power,
                synthetic_frontend,
                ota_frontend,
                args,
                methods=selected_methods,
            )
            pre_noise_models[float(snr_db)] = models
            covariance_reports[str(float(snr_db))] = report
    else:
        pre_noise_models = {
            float(snr_db): {
                method: post_noise_model for method in selected_methods
            }
            for snr_db in args.snrs
        }
        covariance_reports = {"disabled": {}}

    rows: list[dict[str, Any]] = []
    clean_output_diagnostics: list[dict[str, Any]] = []
    for packet_index, packet in enumerate(packets):
        print(
            f"packet {packet_index + 1}/{len(packets)} {packet.packet_id}",
            flush=True,
        )
        expected = packet.expected
        clean_high = packet.load_samples()
        packet_for_eval = packet
        if official_ota_mode and bool(args.trim_official_guards):
            context_high = int(
                FAST_OFFICIAL_CONTEXT_CHIRPS
                * N_BINS
                * (HIGH_RATE_HZ // BW_HZ)
            )
            source_count = int(clean_high.size)
            start = max(0, int(packet.leading_guard_high) - context_high)
            stop = min(
                source_count,
                source_count - int(packet.trailing_guard_high) + context_high,
            )
            clean_high = np.asarray(
                clean_high[start:stop],
                dtype=np.complex64,
            )
            packet_for_eval = packet._replace(
                leading_guard_high=int(packet.leading_guard_high) - start,
                trailing_guard_high=source_count - stop,
            )
        sync_config = _sync_config(packet.center_frequency_hz)
        clean_methods = _method_outputs(
            clean_high,
            synthetic_frontend,
            ota_frontend,
            snr_db=0.0,
            methods=selected_methods,
        )
        clean_preparations = {
            method: prepare_samples_and_sync(
                values,
                sync_config,
                coarse_cfo_centering=official_ota_mode,
            )
            for method, values in clean_methods.items()
        }
        clean_methods = {
            method: preparation.samples
            for method, preparation in clean_preparations.items()
        }
        clean_sync = {
            method: preparation.result
            for method, preparation in clean_preparations.items()
        }
        direct_250k_values: np.ndarray | None = None
        direct_250k_preparation: PreparedSync | None = None
        direct_250k_power: float | None = None
        direct_250k_config: SinglePacketSyncConfig | None = None
        native_power = (
            _active_power(
                clean_methods["native_1msps"],
                leading_guard_high=packet_for_eval.leading_guard_high,
                trailing_guard_high=packet_for_eval.trailing_guard_high,
            )
            if "native_1msps" in clean_methods
            else None
        )
        clean_powers = {
            method: _active_power(
                values,
                leading_guard_high=packet_for_eval.leading_guard_high,
                trailing_guard_high=packet_for_eval.trailing_guard_high,
            )
            for method, values in clean_methods.items()
        }
        for method, values in clean_methods.items():
            power = clean_powers[method]
            clean_scores = evaluate_decoders(
                values,
                clean_sync[method],
                expected,
                post_noise_model,
                decoder_names=selected_decoders,
            )
            rows.append(
                _row(
                    stage="clean",
                    snr_db=None,
                    packet_index=packet_index,
                    method=method,
                    sync_result=clean_sync[method],
                    scores=clean_scores,
                    packet_id=packet.packet_id,
                    source_snr_db=packet.source_snr_db,
                    sync_preparation=clean_preparations[method],
                )
            )
            clean_output_diagnostics.append(
                {
                    "packet_index": packet_index,
                    "packet_id": packet.packet_id,
                    "source_snr_db": packet.source_snr_db,
                    "method": method,
                    "active_power": power,
                    "gain_vs_native_db": (
                        None
                        if native_power is None
                        else float(
                            10.0
                            * math.log10((power + 1e-30) / (native_power + 1e-30))
                        )
                    ),
                    "coarse_cfo_bin": int(
                        clean_preparations[method].coarse_cfo_bin
                    ),
                    "coarse_cfo_hz": float(
                        clean_preparations[method].coarse_cfo_hz
                    ),
                    "initial_sync": _sync_report(
                        clean_preparations[method].initial_result
                    ),
                    "sync": _sync_report(clean_sync[method]),
                }
            )

        if bool(args.include_direct_250k_baseline):
            direct_250k_config = _sync_config(
                packet.center_frequency_hz,
                sample_rate_hz=LOW_RATE_HZ,
            )
            direct_250k_preparation = prepare_samples_and_sync(
                np.asarray(
                    clean_high[:: HIGH_RATE_HZ // LOW_RATE_HZ],
                    dtype=np.complex64,
                ),
                direct_250k_config,
                coarse_cfo_centering=official_ota_mode,
            )
            direct_250k_values = direct_250k_preparation.samples
            direct_250k_power = _active_power(
                direct_250k_values,
                leading_guard_high=packet_for_eval.leading_guard_high,
                trailing_guard_high=packet_for_eval.trailing_guard_high,
                sample_rate_hz=LOW_RATE_HZ,
            )
            direct_250k_scores = evaluate_decoders(
                direct_250k_values,
                direct_250k_preparation.result,
                expected,
                direct_250k_noise_model,
                decoder_names=selected_decoders,
                os_factor=direct_250k_os_factor,
            )
            rows.append(
                _row(
                    stage="clean",
                    snr_db=None,
                    packet_index=packet_index,
                    method=DIRECT_250K_METHOD,
                    sync_result=direct_250k_preparation.result,
                    scores=direct_250k_scores,
                    packet_id=packet.packet_id,
                    source_snr_db=packet.source_snr_db,
                    sync_preparation=direct_250k_preparation,
                )
            )
            clean_output_diagnostics.append(
                {
                    "packet_index": packet_index,
                    "packet_id": packet.packet_id,
                    "source_snr_db": packet.source_snr_db,
                    "method": DIRECT_250K_METHOD,
                    "active_power": direct_250k_power,
                    "gain_vs_native_db": (
                        None
                        if native_power is None
                        else float(
                            10.0
                            * math.log10(
                                (direct_250k_power + 1e-30)
                                / (native_power + 1e-30)
                            )
                        )
                    ),
                    "coarse_cfo_bin": int(
                        direct_250k_preparation.coarse_cfo_bin
                    ),
                    "coarse_cfo_hz": float(
                        direct_250k_preparation.coarse_cfo_hz
                    ),
                    "initial_sync": _sync_report(
                        direct_250k_preparation.initial_result
                    ),
                    "sync": _sync_report(direct_250k_preparation.result),
                }
            )

        for snr_db_value in args.snrs:
            snr_db = float(snr_db_value)
            noise_power = snr_noise_power(packet_for_eval.reference_power, snr_db)

            post_stages = {
                "post_framesync_common_power",
                "post_framesync_gain_matched",
            }.intersection(selected_stages)
            if post_stages:
                post_rng = np.random.default_rng(
                    int(args.noise_seed) + packet_index * 1000
                )
                unit_post_noise = complex_awgn(
                    len(next(iter(clean_methods.values()))), 1.0, post_rng
                )
                for method, clean_values in clean_methods.items():
                    if "post_framesync_common_power" in post_stages:
                        noisy_values = np.asarray(
                            clean_values
                            + unit_post_noise * math.sqrt(noise_power),
                            dtype=np.complex64,
                        )
                        scores = evaluate_decoders(
                            noisy_values,
                            clean_sync[method],
                            expected,
                            post_noise_model,
                            decoder_names=selected_decoders,
                        )
                        rows.append(
                            _row(
                                stage="post_framesync_common_power",
                                snr_db=snr_db,
                                packet_index=packet_index,
                                method=method,
                                sync_result=clean_sync[method],
                                scores=scores,
                                packet_id=packet.packet_id,
                                source_snr_db=packet.source_snr_db,
                                added_noise_power=noise_power,
                                sync_preparation=clean_preparations[method],
                            )
                        )
                    if "post_framesync_gain_matched" in post_stages:
                        matched_noise_power = snr_noise_power(
                            clean_powers[method], snr_db
                        )
                        gain_matched_values = np.asarray(
                            clean_values
                            + unit_post_noise * math.sqrt(matched_noise_power),
                            dtype=np.complex64,
                        )
                        gain_matched_scores = evaluate_decoders(
                            gain_matched_values,
                            clean_sync[method],
                            expected,
                            post_noise_model,
                            decoder_names=selected_decoders,
                        )
                        rows.append(
                            _row(
                                stage="post_framesync_gain_matched",
                                snr_db=snr_db,
                                packet_index=packet_index,
                                method=method,
                                sync_result=clean_sync[method],
                                scores=gain_matched_scores,
                                packet_id=packet.packet_id,
                                source_snr_db=packet.source_snr_db,
                                added_noise_power=matched_noise_power,
                                sync_preparation=clean_preparations[method],
                            )
                        )

                if (
                    direct_250k_values is not None
                    and direct_250k_preparation is not None
                    and direct_250k_power is not None
                ):
                    direct_unit_noise = np.asarray(
                        unit_post_noise[:: OUTPUT_RATE_HZ // LOW_RATE_HZ],
                        dtype=np.complex64,
                    )
                    if direct_unit_noise.shape != direct_250k_values.shape:
                        raise RuntimeError(
                            "paired direct-250k post noise length mismatch"
                        )
                    if "post_framesync_common_power" in post_stages:
                        direct_common_values = np.asarray(
                            direct_250k_values
                            + direct_unit_noise * math.sqrt(noise_power),
                            dtype=np.complex64,
                        )
                        direct_common_scores = evaluate_decoders(
                            direct_common_values,
                            direct_250k_preparation.result,
                            expected,
                            direct_250k_noise_model,
                            decoder_names=selected_decoders,
                            os_factor=direct_250k_os_factor,
                        )
                        rows.append(
                            _row(
                                stage="post_framesync_common_power",
                                snr_db=snr_db,
                                packet_index=packet_index,
                                method=DIRECT_250K_METHOD,
                                sync_result=direct_250k_preparation.result,
                                scores=direct_common_scores,
                                packet_id=packet.packet_id,
                                source_snr_db=packet.source_snr_db,
                                added_noise_power=noise_power,
                                sync_preparation=direct_250k_preparation,
                            )
                        )
                    if "post_framesync_gain_matched" in post_stages:
                        direct_matched_noise_power = snr_noise_power(
                            direct_250k_power, snr_db
                        )
                        direct_matched_values = np.asarray(
                            direct_250k_values
                            + direct_unit_noise
                            * math.sqrt(direct_matched_noise_power),
                            dtype=np.complex64,
                        )
                        direct_matched_scores = evaluate_decoders(
                            direct_matched_values,
                            direct_250k_preparation.result,
                            expected,
                            direct_250k_noise_model,
                            decoder_names=selected_decoders,
                            os_factor=direct_250k_os_factor,
                        )
                        rows.append(
                            _row(
                                stage="post_framesync_gain_matched",
                                snr_db=snr_db,
                                packet_index=packet_index,
                                method=DIRECT_250K_METHOD,
                                sync_result=direct_250k_preparation.result,
                                scores=direct_matched_scores,
                                packet_id=packet.packet_id,
                                source_snr_db=packet.source_snr_db,
                                added_noise_power=direct_matched_noise_power,
                                sync_preparation=direct_250k_preparation,
                            )
                        )

            if "pre_rfsr" in selected_stages:
                pre_rng = np.random.default_rng(
                    int(args.noise_seed) + 1_000_000 + packet_index * 1000
                )
                paired_high_noise = complex_awgn(
                    len(clean_high), noise_power, pre_rng
                )
                noisy_high = np.asarray(
                    clean_high + paired_high_noise, dtype=np.complex64
                )
                input_snr_measurement = measure_decimated_snr(
                    clean_high,
                    noisy_high,
                    decimation=HIGH_RATE_HZ // LOW_RATE_HZ,
                    leading_silence_high_rate=packet_for_eval.leading_guard_high,
                    trailing_silence_high_rate=packet_for_eval.trailing_guard_high,
                    measured_sample_rate_hz=LOW_RATE_HZ,
                )
                noisy_methods = _method_outputs(
                    noisy_high,
                    synthetic_frontend,
                    ota_frontend,
                    snr_db=snr_db,
                    methods=selected_methods,
                )
                for method, noisy_values in noisy_methods.items():
                    preparation = (
                        prepare_samples_with_tracked_sync(
                            noisy_values,
                            sync_config,
                            clean_preparations[method],
                        )
                        if official_ota_mode and args.reuse_clean_frame_location
                        else prepare_samples_with_reused_coarse_cfo(
                            noisy_values,
                            sync_config,
                            clean_preparations[method],
                        )
                        if official_ota_mode and args.reuse_clean_coarse_cfo
                        else prepare_samples_and_sync(
                            noisy_values,
                            sync_config,
                            coarse_cfo_centering=official_ota_mode,
                        )
                    )
                    noisy_values = preparation.samples
                    sync_result = preparation.result
                    scores = evaluate_decoders(
                        noisy_values,
                        sync_result,
                        expected,
                        pre_noise_models[snr_db][method],
                        decoder_names=selected_decoders,
                    )
                    rows.append(
                        _row(
                            stage="pre_rfsr",
                            snr_db=snr_db,
                            packet_index=packet_index,
                            method=method,
                            sync_result=sync_result,
                            scores=scores,
                            input_snr_measurement=input_snr_measurement,
                            packet_id=packet.packet_id,
                            source_snr_db=packet.source_snr_db,
                            added_noise_power=noise_power,
                            sync_preparation=preparation,
                        )
                    )

                if (
                    direct_250k_values is not None
                    and direct_250k_preparation is not None
                    and direct_250k_config is not None
                ):
                    direct_noisy_values = np.asarray(
                        noisy_high[:: HIGH_RATE_HZ // LOW_RATE_HZ],
                        dtype=np.complex64,
                    )
                    direct_noisy_preparation = (
                        prepare_samples_with_tracked_sync(
                            direct_noisy_values,
                            direct_250k_config,
                            direct_250k_preparation,
                        )
                        if official_ota_mode and args.reuse_clean_frame_location
                        else prepare_samples_with_reused_coarse_cfo(
                            direct_noisy_values,
                            direct_250k_config,
                            direct_250k_preparation,
                        )
                        if official_ota_mode and args.reuse_clean_coarse_cfo
                        else prepare_samples_and_sync(
                            direct_noisy_values,
                            direct_250k_config,
                            coarse_cfo_centering=official_ota_mode,
                        )
                    )
                    direct_pre_scores = evaluate_decoders(
                        direct_noisy_preparation.samples,
                        direct_noisy_preparation.result,
                        expected,
                        direct_250k_noise_model,
                        decoder_names=selected_decoders,
                        os_factor=direct_250k_os_factor,
                    )
                    rows.append(
                        _row(
                            stage="pre_rfsr",
                            snr_db=snr_db,
                            packet_index=packet_index,
                            method=DIRECT_250K_METHOD,
                            sync_result=direct_noisy_preparation.result,
                            scores=direct_pre_scores,
                            input_snr_measurement=input_snr_measurement,
                            packet_id=packet.packet_id,
                            source_snr_db=packet.source_snr_db,
                            added_noise_power=noise_power,
                            sync_preparation=direct_noisy_preparation,
                        )
                    )

    synthetic_checkpoint = RFSR_ROOT / "checkpoints" / DEFAULT_SYNTHETIC_CHECKPOINT
    ota_checkpoint = RFSR_ROOT / "checkpoints" / DEFAULT_OTA_CHECKPOINT
    packet_inventory = [
        {
            "packet_index": index,
            "packet_id": packet.packet_id,
            "source_path": (
                None if packet.source_path is None else str(packet.source_path)
            ),
            "source_sha256": (
                None if packet.source_path is None else _sha256(packet.source_path)
            ),
            "metadata_path": (
                None if packet.metadata_path is None else str(packet.metadata_path)
            ),
            "source_snr_db": packet.source_snr_db,
            "active_input_power": packet.reference_power,
            "leading_guard_samples_high_rate": packet.leading_guard_high,
            "trailing_guard_samples_high_rate": packet.trailing_guard_high,
            "expected_header_symbols": len(packet.expected["header"]),
            "expected_payload_symbols": len(packet.expected["payload"]),
        }
        for index, packet in enumerate(packets)
    ]
    runtime_seconds = float(perf_counter() - started_at)
    output = {
        "schema": "official-rfsr-chain-audit",
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime_seconds,
        "scope": {
            "claim": (
                "published checkpoints on a local official RFSR-OTA subset"
                if official_ota_mode
                else "published checkpoints on the vendored synthetic generation path"
            ),
            "input_mode": "official_ota" if official_ota_mode else "synthetic",
            "official_ota_dataset_used": official_ota_mode,
            "official_ota_root": (
                str(Path(args.official_ota_root).expanduser().resolve())
                if official_ota_mode
                else None
            ),
            "download_performed_by_script": False,
            "training_performed": False,
            "limitation": (
                "This is a small explicitly inventoried official OTA subset, not the full "
                "9,528-capture dataset. Injected SNR is relative to each unmodified OTA "
                "active-interval power and is not a replacement for the metadata Welch SNR. "
                + (
                    "Fast tracked-sync mode measures an already-locked streaming receiver; "
                    "it is not a cold-start packet-acquisition test."
                    if args.reuse_clean_frame_location
                    else ""
                )
                if official_ota_mode
                else "No official OTA IQ is used in synthetic mode."
            ),
        },
        "configuration": {
            "packets_requested": int(args.packets),
            "packets": len(packets),
            "methods": list(selected_methods),
            "decoders": list(selected_decoders),
            "noise_stages": list(selected_stages),
            "direct_250ksps_baseline": bool(
                args.include_direct_250k_baseline
            ),
            "fast_official": bool(args.fast_official),
            "official_guards_trimmed": bool(
                official_ota_mode and args.trim_official_guards
            ),
            "clean_coarse_cfo_reused_for_pre_rfsr": bool(
                official_ota_mode and args.reuse_clean_coarse_cfo
            ),
            "clean_frame_location_reused_for_pre_rfsr": bool(
                official_ota_mode and args.reuse_clean_frame_location
            ),
            "target_injected_snrs_db": [float(value) for value in args.snrs],
            "snrs_db": [float(value) for value in args.snrs],
            "seed": int(args.seed),
            "noise_seed": int(args.noise_seed),
            "sf": SF,
            "bandwidth_hz": BW_HZ,
            "high_rate_hz": HIGH_RATE_HZ,
            "low_rate_hz": LOW_RATE_HZ,
            "output_rate_hz": OUTPUT_RATE_HZ,
            "payload_bytes": PAYLOAD_BYTES,
            "cr": CR,
            "ldro": LDRO,
            "preamble_symbols": PREAMBLE_SYMBOLS,
            "sync_word": SYNC_WORD,
            "gls_calibration_reference_power": reference_power,
            "per_packet_noise_reference": "unmodified active 2 MS/s input power",
            "gls_noise_windows": int(args.gls_noise_windows),
            "gls_training_bins": int(args.gls_training_bins),
            "gls_diagonal_loading": float(args.gls_loading),
            "device": device,
        },
        "checkpoints": {
            "official_synthetic": {
                "path": str(synthetic_checkpoint.relative_to(REPO_ROOT)),
                "sha256": _sha256(synthetic_checkpoint),
            },
            "official_ota": {
                "path": str(ota_checkpoint.relative_to(REPO_ROOT)),
                "sha256": _sha256(ota_checkpoint),
            },
        },
        "noise_placement": {
            "official_ota_coarse_cfo_centering": (
                "before final FrameSync, detect the preamble FFT peak on each frontend "
                "output, derotate that integer-bin CFO, then rerun FrameSync; the same "
                "centered clean output and frozen final FrameSync are used for post noise"
                if official_ota_mode
                else "disabled"
            ),
            "pre_rfsr": (
                "paired AWGN at 2 MS/s -> decimation -> measured SNR at 250 kS/s -> "
                "frontend -> reuse the clean candidate frame location -> independently "
                "validate noisy CFO/STO/SFO"
                if official_ota_mode and args.reuse_clean_frame_location
                else "paired AWGN at 2 MS/s -> decimation -> measured SNR at 250 kS/s "
                "active packet interval -> frontend -> noisy FrameSync"
            ),
            "snr_reporting": (
                "snr_db/target_snr_db is added-AWGN SNR relative to the unmodified "
                "packet active power; source_snr_db is the official metadata Welch SNR; "
                "input_snr_measurement.snr_db is measured between unmodified and "
                "noise-added 250 kS/s IQ after decimation"
            ),
            "post_framesync_common_power": (
                "clean frontend -> clean frozen FrameSync -> identical absolute AWGN at 1 MS/s"
            ),
            "post_framesync_gain_matched": (
                "clean frontend -> clean frozen FrameSync -> common normalized AWGN "
                "scaled to each frontend's clean output power"
            ),
            "direct_250ksps_baseline": (
                "raw 250 kS/s IQ with OS2 FrameSync/decoding and no interpolation; "
                "pre noise is inherited from the common noisy 2 MS/s input, and post "
                "noise is every fourth sample of the paired 1 MS/s realization"
                if args.include_direct_250k_baseline
                else "disabled"
            ),
            "failed_sync_scoring": "all expected symbols count as errors",
            "cross_snr_pairing": (
                "each packet reuses one normalized noise realization across SNRs"
            ),
            "post_gls_noise_model": "identity; white post-frontend AWGN",
            "pre_gls_noise_model": (
                "pooled 8x8 covariance from held-out pure noise passed through each frontend"
                if "savaux_gls" in selected_decoders
                else "disabled because savaux_gls is not selected"
            ),
        },
        "official_decode_reproduction": decode_reproduction,
        "packet_inventory": packet_inventory,
        "clean_output_diagnostics": clean_output_diagnostics,
        "pre_rfsr_covariance_diagnostics": covariance_reports,
        "summary": summarize_rows(rows),
        "packet_rows": rows,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {output_path} runtime_seconds={runtime_seconds:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
