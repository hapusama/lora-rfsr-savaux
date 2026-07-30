#!/usr/bin/env python3
"""在留出的 OTA 包上评估 RFSR、Savaux/GLS 与完整 LoRa 解码链。

RFSR 始终位于接收链最前端：输入是未经 CFO、SFO、增益或幅度校正的
250 kS/s OTA IQ。完整包指标通过 GNU Radio/gr-lora_sdr 的真实解码链得到；
可选的 Savaux/branch-GLS 指标只做逐符号 FFT-bin 诊断，使用 manifest 中的
包边界和下游 detector 已估计的 CFO。它不会把真值、CFO 或 SFO 反馈给 RFSR。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DECODE_METHODS = (
    "low_250ksps",
    "interpolation_1msps",
    "rfsr_1msps",
    "native_1msps",
)
SAVAUX_METHODS = (
    "interpolation_1msps",
    "rfsr_1msps",
    "native_1msps",
)


def available_cpu_count() -> int:
    """返回考虑 affinity 和 cgroup quota 后的实际可用 CPU 数。"""

    candidates = [max(1, int(os.cpu_count() or 1))]
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(max(1, len(os.sched_getaffinity(0))))
        except OSError:
            pass

    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        quota_text, period_text = cpu_max.read_text(encoding="ascii").split()
        if quota_text != "max":
            quota = int(quota_text)
            period = int(period_text)
            if quota > 0 and period > 0:
                candidates.append(max(1, math.ceil(quota / period)))
    except (OSError, ValueError):
        pass
    return min(candidates)


def resolve_worker_count(requested: int, task_count: int) -> int:
    """把解码线程数限制在实际 CPU 和 GNU Radio 任务数以内。"""

    count = int(requested)
    if count < 0:
        raise ValueError("--workers must be zero (automatic) or positive")
    available = available_cpu_count()
    target = available if count == 0 else count
    return max(1, min(target, available, max(1, int(task_count))))


def parse_args() -> argparse.Namespace:
    """解析 OTA、噪声扫描和可选 Savaux/GLS 诊断参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ota-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ota-max-groups", type=int, default=None)
    parser.add_argument("--ota-split-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--packet-uid",
        action="append",
        default=None,
        help=(
            "只评估指定 test 物理包 UID；可重复传入。用于让不同 checkpoint "
            "在共同 held-out 物理包上做严格配对比较。"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "并行运行完整 GNU Radio 解码链的独立进程数；1（默认）保持历史串行"
            "行为，0 自动使用当前容器可用 CPU。RFSR CUDA 推理始终留在主进程。"
        ),
    )
    parser.add_argument(
        "--rfsr-snr-conditioning",
        choices=("manifest", "minimum"),
        default="manifest",
        help=(
            "RFSR 的 SNR 条件输入：manifest 沿用 OTA 训练时的 detector SNR；"
            "minimum 把它与额外人工 SNR 取较小值，用于单独消融。"
        ),
    )
    # 保留旧的单点写法，并允许多次给出该参数组成一个离散扫描点集合。
    parser.add_argument(
        "--extra-snr-db",
        type=float,
        action="append",
        default=None,
        help="相对收到 OTA 波形总功率叠加复 AWGN 的单个 SNR 点；可重复给出。",
    )
    parser.add_argument(
        "--extra-snr-start-db",
        type=float,
        default=None,
        help="额外 AWGN 网格的起点；必须与 stop/step 同时指定。",
    )
    parser.add_argument(
        "--extra-snr-stop-db",
        type=float,
        default=None,
        help="额外 AWGN 网格的终点（包含）；可小于起点。",
    )
    parser.add_argument(
        "--extra-snr-step-db",
        type=float,
        default=None,
        help="额外 AWGN 网格的步长，符号必须从起点指向终点。",
    )
    parser.add_argument(
        "--include-raw-ota",
        action="store_true",
        help="在指定额外噪声点时，也额外评估不叠加人工噪声的原始 OTA。",
    )
    parser.add_argument("--noise-seed", type=int, default=20260728)
    parser.add_argument(
        "--savaux-symbol-count",
        type=int,
        default=0,
        help=(
            "每个包额外评估多少个符号的 Savaux/branch-GLS；0 表示关闭。"
            "SF12 的论文 Savaux DFT 计算量较大，建议先用 1 到 4。"
        ),
    )
    parser.add_argument(
        "--savaux-symbol-kind",
        choices=("header", "payload", "all"),
        default="payload",
        help="Savaux/GLS 逐符号诊断选择 header、payload 或二者。",
    )
    parser.add_argument(
        "--savaux-gls-extra-snr-db",
        type=float,
        action="append",
        default=None,
        help=(
            "仅在这些额外 SNR 点运行 Savaux/GLS；未给出时对所有评估条件运行。"
            "原始 OTA 条件不能用此参数单独选择。"
        ),
    )
    parser.add_argument(
        "--savaux-noise-windows",
        type=int,
        default=3,
        help="用于 branch-GLS 的包前噪声窗口数；少于 2 个时回退为白噪声 GLS。",
    )
    parser.add_argument(
        "--savaux-noise-training-bins",
        type=int,
        default=8,
        help="branch-GLS 协方差训练使用的均匀分布 FFT bin 数。",
    )
    parser.add_argument(
        "--savaux-branch-loading",
        type=float,
        default=0.5,
        help="branch-GLS 包前噪声协方差的对角加载系数。",
    )
    parser.add_argument("--savaux-top-l", type=int, default=8)
    parser.add_argument("--sf", type=int, default=12)
    parser.add_argument("--bw", type=float, default=125e3)
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--payload-length", type=int, default=33)
    parser.add_argument("--preamble-symbols", type=int, default=16)
    parser.add_argument("--sync-word", type=int, default=18)
    parser.add_argument("--center-frequency-hz", type=float, default=487700000.0)
    parser.add_argument("--ldro", type=int, choices=(0, 1, 2), default=1)
    return parser.parse_args()


def _append_unique(points: list[float | None], value: float | None) -> None:
    """保留命令行顺序去重，避免重复执行完全相同的噪声条件。"""

    if value is None:
        if None not in points:
            points.append(None)
        return
    if not any(item is not None and math.isclose(item, value, abs_tol=1e-12) for item in points):
        points.append(float(value))


def resolve_extra_snr_points(args: argparse.Namespace) -> list[float | None]:
    """把单点和起止步长两种 CLI 形式统一成稳定的评估条件列表。"""

    grid_fields = (
        args.extra_snr_start_db,
        args.extra_snr_stop_db,
        args.extra_snr_step_db,
    )
    supplied_grid_fields = sum(value is not None for value in grid_fields)
    if supplied_grid_fields not in {0, 3}:
        raise ValueError(
            "--extra-snr-start-db、--extra-snr-stop-db 和 --extra-snr-step-db "
            "必须同时指定"
        )

    points: list[float | None] = []
    for value in args.extra_snr_db or ():
        _append_unique(points, float(value))
    if supplied_grid_fields:
        start = float(args.extra_snr_start_db)
        stop = float(args.extra_snr_stop_db)
        step = float(args.extra_snr_step_db)
        if not math.isfinite(start) or not math.isfinite(stop) or not math.isfinite(step):
            raise ValueError("额外 SNR 网格参数必须是有限数")
        if step == 0.0 or (stop - start) * step < 0.0:
            raise ValueError("额外 SNR 网格步长必须从起点指向终点且不能为 0")
        # 加半个步长的容差，确保 -13 到 -17、步长 -0.5 包含 -17。
        limit = stop + math.copysign(abs(step) * 0.5, step)
        value = start
        while (value <= limit) if step > 0.0 else (value >= limit):
            _append_unique(points, float(value))
            value += step
    if not points:
        points.append(None)
    elif args.include_raw_ota:
        points.insert(0, None)
    return points


def _condition_label(extra_snr_db: float | None) -> str:
    """生成可用于 JSON 和临时文件名的条件标签。"""

    if extra_snr_db is None:
        return "raw_ota"
    value = f"{float(extra_snr_db):g}".replace("-", "neg").replace(".", "p")
    return f"extra_snr_{value}db"


def _normalize_hex(value: object) -> str:
    return "".join(str(value).split()).upper()


def _complex_from_channels(value) -> np.ndarray:
    return np.asarray(value[0].numpy() + 1j * value[1].numpy(), dtype=np.complex64)


def _select_canonical_packet_indices(
    records: list[dict[str, Any]],
    requested_uids: Iterable[str] | None,
    limit: int | None,
) -> list[int]:
    """选择 canonical q0/ADC0 视图，并拒绝不属于当前 test split 的 UID。"""

    canonical = {
        str(record["split_group"]): index
        for index, record in enumerate(records)
        if int(record["adc_phase"]) == 0
        and int(record["lowrate_phase"]) == 0
    }
    if requested_uids:
        requested = list(dict.fromkeys(str(uid) for uid in requested_uids))
        missing = [uid for uid in requested if uid not in canonical]
        if missing:
            raise ValueError(
                "requested --packet-uid is not a canonical packet in the "
                f"current held-out test split: {missing}"
            )
        selected = [canonical[uid] for uid in requested]
    else:
        selected = list(canonical.values())
    if limit is not None:
        selected = selected[: int(limit)]
    return selected


def _extra_awgn_power(samples: np.ndarray, snr_db: float | None) -> float:
    """计算额外 AWGN 功率；None 表示原始 OTA，不增加人工噪声。"""

    if snr_db is None:
        return 0.0
    power = float(np.mean(np.abs(np.asarray(samples, dtype=np.complex64)) ** 2))
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("cannot add AWGN to zero or non-finite IQ power")
    return float(power / (10.0 ** (float(snr_db) / 10.0)))


def _add_extra_awgn(
    samples: np.ndarray,
    snr_db: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """在完整 1 MS/s OTA 上叠加测试专用复 AWGN，再由各支路共享。"""

    output = np.asarray(samples, dtype=np.complex64).copy()
    noise_power = _extra_awgn_power(output, snr_db)
    if noise_power == 0.0:
        return output
    noise = rng.normal(size=output.size) + 1j * rng.normal(size=output.size)
    noise *= math.sqrt(noise_power / 2.0)
    return np.asarray(output + noise.astype(np.complex64), dtype=np.complex64)


def _packet_rng(noise_seed: int, packet_ordinal: int) -> np.random.Generator:
    """让同一包在不同 SNR 点复用同一标准高斯噪声序列，只改变幅度。"""

    return np.random.default_rng(np.random.SeedSequence((int(noise_seed), int(packet_ordinal))))


def _rfsr_conditioning_snr(
    source_snr_db: float,
    extra_snr_db: float | None,
    mode: str,
) -> float:
    """确定 RFSR 的条件输入；无论哪种模式都不会改动输入 IQ。"""

    if str(mode) == "manifest" or extra_snr_db is None:
        return float(source_snr_db)
    if str(mode) == "minimum":
        return float(min(float(source_snr_db), float(extra_snr_db)))
    raise ValueError(f"unknown RFSR SNR conditioning mode: {mode}")


def _detector_args(args: argparse.Namespace, sample_rate_hz: float) -> argparse.Namespace:
    """构造与 OTA manifest 生成阶段一致的 GNU Radio LoRa 参数。"""

    return argparse.Namespace(
        sf=int(args.sf),
        bw=float(args.bw),
        samp_rate=float(sample_rate_hz),
        cr=int(args.cr),
        pay_len=int(args.payload_length),
        has_crc=True,
        impl_head=False,
        soft_decoding=False,
        center_freq=float(args.center_frequency_hz),
        sync_word=int(args.sync_word),
        preamble_len=int(args.preamble_symbols),
        ldro_mode=int(args.ldro),
        crc_mode=0,
        print_header=False,
        print_grlora=False,
    )


def _decode_cfile(
    *,
    samples: np.ndarray,
    sample_rate_hz: float,
    expected_frame_hex: str,
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """把一条 IQ 支路接到完整 gr-lora_sdr 链，并按 CRC 与完整帧评分。"""

    np.asarray(samples, dtype=np.dtype("<c8")).tofile(path)
    return _decode_existing_cfile(
        path=path,
        expected_frame_hex=expected_frame_hex,
        detector_options=vars(_detector_args(args, sample_rate_hz)),
    )


def _decode_existing_cfile(
    *,
    path: Path,
    expected_frame_hex: str,
    detector_options: dict[str, Any],
) -> dict[str, Any]:
    """在独立进程中读取已落盘 IQ，避免共享 GNU Radio/CUDA 运行时状态。"""

    # 延迟导入使 spawn 子进程只初始化 GNU Radio，不初始化父进程的 CUDA context。
    from noisy_iq.detector import run_grlora_packet_detector

    packets = run_grlora_packet_detector(
        Path(path), argparse.Namespace(**detector_options)
    )
    expected = _normalize_hex(expected_frame_hex)
    decoded = [
        {
            "crc_valid": bool(item.get("crc_valid", False)),
            "frame_hex": _normalize_hex(item.get("decoded_payload_hex", "")),
        }
        for item in packets
    ]
    return {
        "detected_packets": len(decoded),
        "crc_valid_packets": sum(item["crc_valid"] for item in decoded),
        "expected_frame_match": any(
            item["crc_valid"] and item["frame_hex"] == expected for item in decoded
        ),
        "decoded_packets": decoded,
    }


def _metadata_path(ota_root: Path, record: dict[str, object]) -> Path:
    return ota_root / "metadata" / f"{Path(str(record['ota_path'])).stem}.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_metadata(ota_root: Path, ota_metadata: dict[str, Any]) -> dict[str, Any]:
    """由 OTA manifest 的 reference_id 找到对应理想 PHY metadata。"""

    reference_id = int(ota_metadata["reference"]["reference_id"])
    path = ota_root.parent / "metadata" / f"{reference_id:06d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing reference PHY metadata: {path}")
    return _load_json(path)


def _symbol_specs(
    reference_metadata: dict[str, Any], kind: str, count: int
) -> list[dict[str, Any]]:
    """按 raw PHY 格式重建 data symbol 起点；符号 ID 只在最终评分时使用。"""

    phy = reference_metadata["phy"]
    symbols = reference_metadata["symbols"]
    iq = reference_metadata["iq"]
    samples_per_symbol = int(phy["samples_per_symbol"])
    # raw PHY 的数据区位于 preamble、两个 sync、两个完整 downchirp 及 1/4 SFD 之后。
    header_start = (
        int(iq["leading_silence_samples"])
        + (int(phy["preamble_symbols"]) + 4) * samples_per_symbol
        + samples_per_symbol // 4
    )
    candidates: list[dict[str, Any]] = []
    if kind in {"header", "all"}:
        candidates.extend(
            {
                "section": "header",
                "section_index": index,
                "start_sample": header_start + index * samples_per_symbol,
                "gt_bin": int(value),
            }
            for index, value in enumerate(symbols["header_ids"])
        )
    if kind in {"payload", "all"}:
        payload_start = header_start + len(symbols["header_ids"]) * samples_per_symbol
        candidates.extend(
            {
                "section": "payload",
                "section_index": index,
                "start_sample": payload_start + index * samples_per_symbol,
                "gt_bin": int(value),
            }
            for index, value in enumerate(symbols["payload_ids"])
        )
    return candidates[: int(count)]


def _manifest_cfo_bins(ota_metadata: dict[str, Any], sf: int, bw_hz: float) -> tuple[int, float]:
    """把下游 detector 的 Hz 单位 CFO 换成 Savaux 所需的整数/小数 bin。"""

    raw_value = ota_metadata.get("alignment", {}).get("estimated_cfo_hz", "")
    try:
        cfo_hz = float(raw_value)
    except (TypeError, ValueError):
        cfo_hz = 0.0
    if not math.isfinite(cfo_hz):
        cfo_hz = 0.0
    total_bins = cfo_hz * float(1 << int(sf)) / float(bw_hz)
    cfo_int = int(round(total_bins))
    return cfo_int, float(total_bins - cfo_int)


def _leading_offpacket_noise(
    selected: Iterable[int],
    dataset,
    ota_root: Path,
    extra_snr_db: float | None,
    noise_seed: int,
    extra_noise_power: float,
) -> np.ndarray:
    """拼接不同留出包的真实包前噪声，供 GLS 估计输出支路协方差。

    这部分只读取 manifest 标记的包前 off-packet 区域，既不包含 payload，也不
    读取 reference IQ。跨包拼接只用于噪声统计，随后会重新切成完整 LoRa symbol
    长度的窗口。
    """

    pieces: list[np.ndarray] = []
    seen_paths: set[Path] = set()
    for index in selected:
        record = dataset.records[index]
        ota_path = Path(record["ota_path"])
        if ota_path in seen_paths:
            continue
        seen_paths.add(ota_path)
        metadata = _load_json(_metadata_path(ota_root, record))
        leading = int(metadata["ota"]["leading_real_off_packet_samples"])
        samples = np.memmap(ota_path, dtype=np.dtype("<c8"), mode="r")
        pieces.append(np.asarray(samples[:leading], dtype=np.complex64))
    if not pieces:
        raise RuntimeError("no off-packet noise is available for Savaux/GLS calibration")
    output = np.concatenate(pieces).astype(np.complex64, copy=False)
    if extra_snr_db is None or extra_noise_power <= 0.0:
        return output
    rng = np.random.default_rng(np.random.SeedSequence((int(noise_seed), 99173)))
    noise = rng.normal(size=output.size) + 1j * rng.normal(size=output.size)
    noise *= math.sqrt(float(extra_noise_power) / 2.0)
    return np.asarray(output + noise.astype(np.complex64), dtype=np.complex64)


def _estimate_savaux_noise_models(
    *,
    offpacket_high: np.ndarray,
    frontend,
    rfsr_snr_db: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """为三个 1 MS/s 支路估计 OSR x OSR GLS 噪声模型。

    协方差只由包前噪声构成。没有两个完整窗口时，显式回退为 identity model；
    该模型下 GLS 与普通 Savaux 数学等价，结果中会记录这一点。
    """

    from weak_decoder.os_lora.system.oversampled_glrt import (
        estimate_branch_noise_model,
        identity_branch_noise_model,
    )

    symbol_samples = (1 << int(args.sf)) * 8
    available = int(np.asarray(offpacket_high).size) // symbol_samples
    count = min(int(args.savaux_noise_windows), available)
    if count < 2:
        models = {name: identity_branch_noise_model(8) for name in SAVAUX_METHODS}
        report = {
            name: {
                "source": "identity_fallback",
                "window_count": int(count),
                "snapshot_count": 0,
                "reason": "fewer than two complete off-packet LoRa-symbol windows",
            }
            for name in SAVAUX_METHODS
        }
        return models, report

    high = np.asarray(offpacket_high[: count * symbol_samples], dtype=np.complex64)
    low = high[::4].copy()
    views = {
        "interpolation_1msps": frontend.interpolate(low),
        "rfsr_1msps": frontend.enhance(low, float(rfsr_snr_db)),
        "native_1msps": high,
    }
    n_bins = 1 << int(args.sf)
    bin_count = max(1, min(int(args.savaux_noise_training_bins), n_bins))
    training_bins = tuple(
        int(value) for value in np.linspace(0, n_bins, bin_count, endpoint=False)
    )
    models: dict[str, Any] = {}
    report: dict[str, dict[str, Any]] = {}
    for name, values in views.items():
        windows = np.asarray(values, dtype=np.complex64).reshape(count, symbol_samples)
        model = estimate_branch_noise_model(
            windows,
            sf=int(args.sf),
            os_factor=8,
            training_bins=training_bins,
            diagonal_loading=float(args.savaux_branch_loading),
            covariance_mode="pooled",
        )
        models[name] = model
        report[name] = {
            "source": "heldout_offpacket_noise",
            "window_count": int(count),
            "snapshot_count": int(model.snapshot_count),
            "training_bins": list(training_bins),
            "diagonal_loading": float(model.diagonal_loading),
        }
    return models, report


def _evaluate_savaux_gls(
    *,
    method_samples: dict[str, np.ndarray],
    symbol_specs: list[dict[str, Any]],
    cfo_int: int,
    cfo_frac: float,
    noise_models: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    """复用 os_lora 的论文 Savaux 频谱及 branch-GLS，在同一符号上配对评分。"""

    from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (
        paper_oversampled_spectrum,
    )
    from weak_decoder.os_lora.system.oversampled_glrt import branch_gls_scores

    summary = {
        name: {
            "symbol_count": 0,
            "savaux_correct": 0,
            "branch_gls_correct": 0,
        }
        for name in SAVAUX_METHODS
    }
    rows: list[dict[str, Any]] = []
    for spec in symbol_specs:
        for name in SAVAUX_METHODS:
            values = method_samples[name]
            start = int(spec["start_sample"])
            if start < 0 or start + (1 << int(args.sf)) * 8 > values.size:
                raise ValueError(f"{name} does not contain requested Savaux symbol at {start}")
            # CFO 校正属于 RFSR 后面的普通 Savaux 接收机；这里没有 SFO 校正，
            # 也不对网络输入做任何旋转、时间重采样或幅度归一化。
            combined, branches, _ = paper_oversampled_spectrum(
                samples=values,
                start_sample=start,
                sf=int(args.sf),
                os_factor=8,
                cfo_int=int(cfo_int),
                cfo_frac=float(cfo_frac),
                cfo_correction_mode="symbol",
            )
            savaux_power = np.abs(combined).astype(np.float64) ** 2
            savaux_bin = int(np.argmax(savaux_power))
            gls = branch_gls_scores(
                branches,
                8,
                noise_model=noise_models[name],
                top_l=int(args.savaux_top_l),
            )
            gt_bin = int(spec["gt_bin"])
            savaux_correct = int(savaux_bin == gt_bin)
            gls_correct = int(int(gls.selected_bin) == gt_bin)
            summary[name]["symbol_count"] += 1
            summary[name]["savaux_correct"] += savaux_correct
            summary[name]["branch_gls_correct"] += gls_correct
            rows.append(
                {
                    "method": name,
                    "section": str(spec["section"]),
                    "section_index": int(spec["section_index"]),
                    "start_sample_1msps": start,
                    "gt_bin": gt_bin,
                    "savaux_bin": savaux_bin,
                    "savaux_correct": savaux_correct,
                    "branch_gls_bin": int(gls.selected_bin),
                    "branch_gls_correct": gls_correct,
                    "branch_gls_top_candidates": list(gls.top_candidates),
                }
            )
    return summary, rows


def _condition_runs_savaux(args: argparse.Namespace, extra_snr_db: float | None) -> bool:
    if int(args.savaux_symbol_count) <= 0:
        return False
    requested = args.savaux_gls_extra_snr_db
    if not requested:
        return True
    if extra_snr_db is None:
        return False
    return any(math.isclose(float(value), float(extra_snr_db), abs_tol=1e-12) for value in requested)


def _decode_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        method: {
            "expected_frame_matches": sum(
                int(row["methods"][method]["expected_frame_match"]) for row in rows
            ),
            "crc_valid_packets": sum(
                int(row["methods"][method]["crc_valid_packets"]) for row in rows
            ),
        }
        for method in DECODE_METHODS
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if int(args.workers) < 0:
        raise ValueError("--workers must be zero (automatic) or positive")
    if int(args.savaux_symbol_count) < 0:
        raise ValueError("--savaux-symbol-count must be non-negative")
    if int(args.savaux_noise_windows) < 1:
        raise ValueError("--savaux-noise-windows must be positive")

    # GNU Radio 必须先于 PyTorch CUDA 载入。grlora Conda 环境里这样会固定较新的
    # libstdc++，避免 PyTorch 先载入系统旧版库后缺少 GNU Radio 3.10 的 GLIBCXX 符号。
    try:
        from gnuradio import gr  # noqa: F401
        import gnuradio.lora_sdr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "decoder evaluation requires GNU Radio and gr-lora_sdr; run it "
            "in the grlora Conda environment."
        ) from exc

    from rfsr.nn.ota_dataset import OTALoRaDataset
    from weak_decoder.rf_super_resolution import (
        RFSRFrontendConfig,
        RFSuperResolutionFrontend,
        default_rfsr_repo_root,
    )

    ota_root = args.ota_root.expanduser().resolve()
    dataset = OTALoRaDataset(
        dataset_root=ota_root,
        split="test",
        split_seed=args.ota_split_seed,
        max_groups=args.ota_max_groups,
        target_source="received",
        return_snr=True,
    )
    frontend = RFSuperResolutionFrontend(
        RFSRFrontendConfig(
            repo_root=default_rfsr_repo_root(),
            checkpoint=args.checkpoint,
            device=args.device,
        )
    )
    # 只取 q0/ADC0，因此每个物理发送包只被完整解码一次；其他 phase 绝不进入
    # 训练/测试交叉边界，也不被当作独立 packet 计数。
    selected = _select_canonical_packet_indices(
        dataset.records, args.packet_uid, args.limit
    )
    if not selected:
        raise RuntimeError("no q0/ADC-phase-0 packet available in test split")
    worker_count = resolve_worker_count(
        int(args.workers), len(selected) * len(DECODE_METHODS)
    )
    print(
        json.dumps(
            {
                "parallel_backend": "process_pool_spawn",
                "available_cpu_count": available_cpu_count(),
                "worker_count": worker_count,
                "rfsr_inference": "single_main_process",
                "gnuradio_decode": "parallel_independent_branches",
            }
        )
    )
    # GLS 协方差可额外使用同一 test split 内 ADC1 的包前噪声；它们和 q0 一样
    # 属于完全留出的物理包，但不会进入 packet 成功率或逐符号真值统计。
    selected_groups = {str(dataset.records[index]["split_group"]) for index in selected}
    noise_calibration_indices = [
        index
        for index, record in enumerate(dataset.records)
        if str(record["split_group"]) in selected_groups
        and int(record["lowrate_phase"]) == 0
    ]

    conditions: list[dict[str, Any]] = []
    for extra_snr_db in resolve_extra_snr_points(args):
        rows: list[dict[str, Any]] = []
        savaux_inputs: list[dict[str, Any]] = []
        added_noise_powers: list[float] = []
        with tempfile.TemporaryDirectory(prefix="rfsr-ota-decode-") as temporary:
            temporary_root = Path(temporary)
            prepared_rows: list[dict[str, Any]] = []
            for ordinal, index in enumerate(selected):
                x, y, snr = dataset[index]
                del x  # 高率 received target 与 q0 低率输入严格同相位，下面直接再抽取 low。
                record = dataset.records[index]
                source_high = _complex_from_channels(y)
                added_noise_powers.append(_extra_awgn_power(source_high, extra_snr_db))
                high = _add_extra_awgn(source_high, extra_snr_db, _packet_rng(args.noise_seed, ordinal))
                low = high[::4].copy()
                source_snr_db = float(snr.item())
                conditioning_snr_db = _rfsr_conditioning_snr(
                    source_snr_db,
                    extra_snr_db,
                    args.rfsr_snr_conditioning,
                )
                interpolated = frontend.interpolate(low)
                rfsr = frontend.enhance(low, conditioning_snr_db)
                methods = (
                    ("low_250ksps", low, 250e3),
                    ("interpolation_1msps", interpolated, 1e6),
                    ("rfsr_1msps", rfsr, 1e6),
                    ("native_1msps", high, 1e6),
                )
                ota_metadata = _load_json(_metadata_path(ota_root, record))
                expected_frame_hex = str(ota_metadata["packet"]["expected_frame_hex"])
                decode_inputs: list[tuple[str, Path, dict[str, Any]]] = []
                for name, values, sample_rate_hz in methods:
                    path = temporary_root / f"packet{ordinal:03d}_{name}.cfile"
                    np.asarray(values, dtype=np.dtype("<c8")).tofile(path)
                    decode_inputs.append(
                        (
                            name,
                            path,
                            vars(_detector_args(args, sample_rate_hz)),
                        )
                    )
                prepared_rows.append(
                    {
                        "physical_packet_uid": record["split_group"],
                        "view_id": record["view_id"],
                        "source_snr_db": source_snr_db,
                        "rfsr_conditioning_snr_db": conditioning_snr_db,
                        "expected_frame_hex": expected_frame_hex,
                        "decode_inputs": decode_inputs,
                    }
                )
                if _condition_runs_savaux(args, extra_snr_db):
                    reference_metadata = _reference_metadata(ota_root, ota_metadata)
                    savaux_inputs.append(
                        {
                            "physical_packet_uid": str(record["split_group"]),
                            "symbol_specs": _symbol_specs(
                                reference_metadata,
                                args.savaux_symbol_kind,
                                int(args.savaux_symbol_count),
                            ),
                            "cfo": _manifest_cfo_bins(ota_metadata, args.sf, args.bw),
                            "method_samples": {
                                "interpolation_1msps": interpolated,
                                "rfsr_1msps": rfsr,
                                "native_1msps": high,
                            },
                        }
                    )

            if worker_count == 1:
                for prepared in prepared_rows:
                    decode_inputs = prepared.pop("decode_inputs")
                    prepared["methods"] = {
                        name: _decode_existing_cfile(
                            path=path,
                            expected_frame_hex=str(prepared["expected_frame_hex"]),
                            detector_options=detector_options,
                        )
                        for name, path, detector_options in decode_inputs
                    }
                    rows.append(prepared)
            else:
                # spawn 子进程不会继承已经初始化的 CUDA context；每个进程只加载
                # GNU Radio 并读取父进程写好的 cfile，隔离其非线程安全 C++ 状态。
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("spawn"),
                ) as executor:
                    futures: list[tuple[dict[str, Any], str, Any]] = []
                    for prepared in prepared_rows:
                        for name, path, detector_options in prepared["decode_inputs"]:
                            future = executor.submit(
                                _decode_existing_cfile,
                                path=path,
                                expected_frame_hex=str(
                                    prepared["expected_frame_hex"]
                                ),
                                detector_options=detector_options,
                            )
                            futures.append((prepared, name, future))
                    for prepared, name, future in futures:
                        prepared.setdefault("methods", {})[name] = future.result()
                for prepared in prepared_rows:
                    prepared.pop("decode_inputs")
                    rows.append(prepared)

        condition: dict[str, Any] = {
            "label": _condition_label(extra_snr_db),
            "extra_snr_db": extra_snr_db,
            "packet_count": len(rows),
            "summary": _decode_summary(rows),
            "packets": rows,
        }
        if savaux_inputs:
            # 所有留出包的包前噪声只用于 GLS 协方差；中位额外噪声功率与当前条件
            # 匹配，但不会复用任何 active symbol 或真值 bin。
            offpacket = _leading_offpacket_noise(
                noise_calibration_indices,
                dataset,
                ota_root,
                extra_snr_db,
                args.noise_seed,
                float(np.median(added_noise_powers)),
            )
            conditioning_values = [
                float(row["rfsr_conditioning_snr_db"]) for row in rows
            ]
            noise_models, noise_report = _estimate_savaux_noise_models(
                offpacket_high=offpacket,
                frontend=frontend,
                rfsr_snr_db=float(np.median(conditioning_values)),
                args=args,
            )
            method_summary = {
                name: {
                    "symbol_count": 0,
                    "savaux_correct": 0,
                    "branch_gls_correct": 0,
                }
                for name in SAVAUX_METHODS
            }
            symbol_rows: list[dict[str, Any]] = []
            for packet in savaux_inputs:
                cfo_int, cfo_frac = packet["cfo"]
                packet_summary, packet_rows = _evaluate_savaux_gls(
                    method_samples=packet["method_samples"],
                    symbol_specs=packet["symbol_specs"],
                    cfo_int=int(cfo_int),
                    cfo_frac=float(cfo_frac),
                    noise_models=noise_models,
                    args=args,
                )
                for name, values in packet_summary.items():
                    for key, value in values.items():
                        method_summary[name][key] += int(value)
                for row in packet_rows:
                    symbol_rows.append(
                        {"physical_packet_uid": packet["physical_packet_uid"], **row}
                    )
            condition["savaux_branch_gls"] = {
                "scope": (
                    "symbol-level diagnostic with manifest packet boundary and "
                    "downstream detector CFO; no RFSR-side CFO/SFO correction"
                ),
                "noise_models": noise_report,
                "summary": method_summary,
                "symbols": symbol_rows,
            }
        conditions.append(condition)
        print(json.dumps({"condition": condition["label"], "summary": condition["summary"]}, indent=2))

    payload: dict[str, Any] = {
        "schema": "lora-rfsr-ota-decoder-evaluation-v2",
        "test_split": "physical-packet 6:2:2, held-out test",
        "evaluation_waveform": (
            "received OTA; RFSR runs before packet detection and FrameSync"
        ),
        "rfsr_snr_conditioning": str(args.rfsr_snr_conditioning),
        "packet_count_per_condition": len(selected),
        "evaluated_test_packets": [
            str(dataset.records[index]["split_group"]) for index in selected
        ],
        "execution": {
            "parallel_backend": "process_pool_spawn",
            "available_cpu_count": available_cpu_count(),
            "worker_count": worker_count,
            "rfsr_inference": "single_main_process",
            "gnuradio_decode": "parallel_independent_branches",
        },
        "rfsr": frontend.provenance.__dict__,
        "conditions": conditions,
    }
    # 单条件保留旧版顶层字段，已有分析脚本无需调整也能读取这次结果。
    if len(conditions) == 1:
        only = conditions[0]
        payload.update(
            {
                "extra_snr_db": only["extra_snr_db"],
                "packet_count": only["packet_count"],
                "summary": only["summary"],
                "packets": only["packets"],
            }
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote decoder evaluation: {output}")


if __name__ == "__main__":
    main()
