#!/usr/bin/env python3
"""评估干净 RFSR 输出固定同步后、不同加噪条件下的 Savaux 符号错误率。

每个 held-out OTA 包先在不额外加噪的条件下完成降采样、RFSR 和 FrameSync，
随后固定这份 CFO/STO/SFO 与符号起点信息。不同 SNR 的复高斯白噪声只加入
已经完成 RFSR 的 1 MS/s 波形，Savaux 直接复用干净包的 FrameSync，不重新检测
或同步。参考真值只在全部硬判决结束后加载并用于评分。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


METHODS = ("rfsr_1msps", "interpolation_1msps", "native_1msps")


def available_cpu_count() -> int:
    """返回当前进程实际可用的 CPU 数，而不是宿主机的总 CPU 数。

    AutoDL/容器场景中 ``os.cpu_count()`` 可能看见宿主机全部逻辑核；优先取
    Linux affinity，再用 cgroup v2/v1 quota 收紧上限，避免自动并行过度抢占。
    """

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

    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        quota = int(quota_path.read_text(encoding="ascii").strip())
        period = int(period_path.read_text(encoding="ascii").strip())
        if quota > 0 and period > 0:
            candidates.append(max(1, math.ceil(quota / period)))
    except (OSError, ValueError):
        pass
    return min(candidates)


def resolve_worker_count(requested: int, task_count: int) -> int:
    """把用户请求的线程数限制在当前任务量和可用 CPU 范围内。"""

    count = int(requested)
    if count < 0:
        raise ValueError("--workers must be zero (automatic) or positive")
    if int(task_count) < 1:
        return 1
    available = available_cpu_count()
    target = available if count == 0 else count
    return max(1, min(target, available, int(task_count)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ota-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ota-max-groups", type=int, default=100)
    parser.add_argument("--ota-split-seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "CPU 线程数；0（默认）表示自动使用当前容器可用 CPU 与任务数的较小值。"
        ),
    )
    parser.add_argument(
        "--save-symbol-details",
        action="store_true",
        help=(
            "在 JSON 中保存每个 symbol 的 Savaux 决策和逐符号评分；默认只保存 "
            "SER 汇总，以避免大规模 SNR 网格生成数十 MB 的 JSON。"
        ),
    )
    parser.add_argument(
        "--allow-unbound-checkpoint",
        action="store_true",
        help="允许评估没有训练 split sidecar 的旧 checkpoint。",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=METHODS,
        default=None,
        help="默认只评估 rfsr_1msps；可重复传入以增加配对对照支路。",
    )
    parser.add_argument("--extra-snr-db", type=float, action="append", default=None)
    parser.add_argument("--extra-snr-start-db", type=float, default=None)
    parser.add_argument("--extra-snr-stop-db", type=float, default=None)
    parser.add_argument("--extra-snr-step-db", type=float, default=None)
    parser.add_argument(
        "--include-clean-output",
        "--include-raw-ota",
        dest="include_clean_output",
        action="store_true",
        help="在加噪网格前加入 RFSR 后不额外加噪的基准条件。",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        action="append",
        default=None,
        help="复 AWGN seed；可重复传入。默认从 20260728 开始。",
    )
    parser.add_argument(
        "--noise-seed-count",
        type=int,
        default=1,
        help="从单个 --noise-seed 起连续使用多少个 seed。",
    )
    parser.add_argument(
        "--ser-section",
        choices=("payload", "header", "all"),
        default="payload",
    )
    parser.add_argument("--sync-detection-chirps", type=int, default=4)
    parser.add_argument("--sync-scan-chirps", type=int, default=32)
    parser.add_argument("--sf", type=int, default=12)
    parser.add_argument("--bw", type=float, default=125e3)
    parser.add_argument("--preamble-symbols", type=int, default=16)
    parser.add_argument("--sync-word", type=int, default=0x12)
    parser.add_argument("--center-frequency-hz", type=float, default=487700000.0)
    parser.add_argument("--ldro", type=int, choices=(0, 1, 2), default=1)
    return parser.parse_args()


def _append_unique(points: list[float | None], value: float | None) -> None:
    """向噪声条件列表追加一个尚未出现的点；None 表示不额外加噪。"""

    if value is None:
        if None not in points:
            points.append(None)
        return
    if not any(
        item is not None and math.isclose(item, value, abs_tol=1e-12)
        for item in points
    ):
        points.append(float(value))


def resolve_extra_snr_points(args: argparse.Namespace) -> list[float | None]:
    """合并离散 SNR 参数和等间隔网格，并按命令行顺序去重。"""

    grid = (
        args.extra_snr_start_db,
        args.extra_snr_stop_db,
        args.extra_snr_step_db,
    )
    supplied = sum(value is not None for value in grid)
    if supplied not in {0, 3}:
        raise ValueError("extra SNR start/stop/step must be specified together")
    points: list[float | None] = []
    for value in args.extra_snr_db or ():
        _append_unique(points, float(value))
    if supplied:
        start, stop, step = map(float, grid)
        if not all(math.isfinite(value) for value in (start, stop, step)):
            raise ValueError("extra SNR grid values must be finite")
        if step == 0.0 or (stop - start) * step < 0.0:
            raise ValueError("extra SNR step must point from start to stop")
        limit = stop + math.copysign(abs(step) * 0.5, step)
        value = start
        while (value <= limit) if step > 0.0 else (value >= limit):
            _append_unique(points, value)
            value += step
    if not points:
        points.append(None)
    elif bool(args.include_clean_output):
        points.insert(0, None)
    return points


def resolve_noise_seeds(args: argparse.Namespace) -> list[int]:
    """解析显式 seed 列表，或从一个起点展开连续 seed。"""

    count = int(args.noise_seed_count)
    if count < 1:
        raise ValueError("--noise-seed-count must be positive")
    explicit = [int(value) for value in (args.noise_seed or [20260728])]
    if count > 1 and len(explicit) != 1:
        raise ValueError(
            "--noise-seed-count > 1 requires exactly one --noise-seed"
        )
    if count > 1:
        return list(range(explicit[0], explicit[0] + count))
    return list(dict.fromkeys(explicit))


def _condition_label(extra_snr_db: float | None) -> str:
    """把噪声条件转换为适合写入 JSON 的稳定名称。"""

    if extra_snr_db is None:
        return "no_extra_noise"
    value = f"{float(extra_snr_db):g}".replace("-", "neg").replace(".", "p")
    return f"extra_snr_{value}db"


def _complex_from_channels(value: Any) -> np.ndarray:
    """把数据集返回的 I/Q 双通道张量还原为一维复数 IQ。"""

    return np.asarray(
        value[0].numpy() + 1j * value[1].numpy(), dtype=np.complex64
    )


def _extra_awgn_power(samples: np.ndarray, snr_db: float | None) -> float:
    """按给定参考波形的平均功率计算额外复噪声功率。"""

    if snr_db is None:
        return 0.0
    # 正式比较传入原生 1 MS/s OTA；函数默认形式也供独立单元测试使用。
    power = float(np.mean(np.abs(np.asarray(samples, dtype=np.complex64)) ** 2))
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("cannot add AWGN to zero or non-finite IQ power")
    return float(power / (10.0 ** (float(snr_db) / 10.0)))


def _add_extra_awgn(
    samples: np.ndarray,
    snr_db: float | None,
    rng: np.random.Generator,
    *,
    reference_power: float | None = None,
) -> np.ndarray:
    """在 1 MS/s 干净输出上叠加复 AWGN，可指定跨支路公共参考功率。"""

    output = np.asarray(samples, dtype=np.complex64).copy()
    if reference_power is None:
        noise_power = _extra_awgn_power(output, snr_db)
    elif snr_db is None:
        noise_power = 0.0
    else:
        if not math.isfinite(reference_power) or float(reference_power) <= 0.0:
            raise ValueError("reference_power must be positive and finite")
        noise_power = float(
            float(reference_power) / (10.0 ** (float(snr_db) / 10.0))
        )
    if noise_power == 0.0:
        return output
    noise = rng.normal(size=output.size) + 1j * rng.normal(size=output.size)
    noise *= math.sqrt(noise_power / 2.0)
    return np.asarray(output + noise.astype(np.complex64), dtype=np.complex64)


def _packet_rng(noise_seed: int, packet_ordinal: int) -> np.random.Generator:
    # 同一包在不同 SNR 条件下复用同一标准噪声实现，只改变缩放系数，便于配对比较。
    return np.random.default_rng(
        np.random.SeedSequence((int(noise_seed), int(packet_ordinal)))
    )


def _physical_groups(dataset: Any) -> list[str]:
    """从某个数据集 split 中提取排序后的物理包 UID。"""

    return sorted({str(record["split_group"]) for record in dataset.records})


def _split_manifest(datasets: dict[str, Any], seed: int, max_groups: int) -> dict[str, Any]:
    """生成本次评估的数据划分记录，并检查物理包是否跨 split 泄漏。"""

    # split_group 是物理包 UID；同一包的所有 ADC/polyphase views 必须同组。
    groups = {split: _physical_groups(dataset) for split, dataset in datasets.items()}
    sets = {split: set(values) for split, values in groups.items()}
    overlaps = {
        "train_validation": sorted(sets["train"] & sets["validation"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "validation_test": sorted(sets["validation"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError("physical packet leakage across OTA splits")
    return {
        "algorithm": "deterministic physical-packet 6:2:2",
        "split_seed": int(seed),
        "max_groups": int(max_groups),
        "physical_packet_counts": {
            split: len(values) for split, values in groups.items()
        },
        "disjoint": True,
        "overlaps": overlaps,
        "train": groups["train"],
        "validation": groups["validation"],
        "test": groups["test"],
    }


def _load_json(path: Path) -> dict[str, Any]:
    """以 UTF-8 读取一个 JSON 对象。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_split_binding(
    checkpoint: Path,
    evaluated_split: dict[str, Any],
    *,
    allow_unbound: bool,
) -> dict[str, Any]:
    """确认评估重建的 split 与 checkpoint 训练时保存的 split 完全一致。"""

    from rfsr.nn.ota_dataset import checkpoint_split_manifest_path

    # checkpoint 必须与训练时落盘的 split sidecar 一致，防止换 seed 后误测。
    checkpoint_path = checkpoint.expanduser().resolve()
    manifest_path = checkpoint_split_manifest_path(checkpoint_path)
    if not manifest_path.is_file():
        if not allow_unbound:
            raise FileNotFoundError(
                "checkpoint has no split manifest; refusing unverifiable "
                f"held-out evaluation: {manifest_path}. Pass "
                "--allow-unbound-checkpoint only for a documented legacy model."
            )
        return {
            "status": "unbound_legacy_checkpoint",
            "verified": False,
            "manifest_path": str(manifest_path),
        }

    training = _load_json(manifest_path)
    errors: list[str] = []
    expected_scalars = {
        "split_seed": evaluated_split["split_seed"],
        "max_groups": evaluated_split["max_groups"],
        "target_source": "received",
    }
    for key, expected in expected_scalars.items():
        if training.get(key) != expected:
            errors.append(
                f"{key}: checkpoint={training.get(key)!r}, evaluation={expected!r}"
            )
    for split in ("train", "validation", "test"):
        if training.get(split) != evaluated_split[split]:
            errors.append(f"{split} physical packet UIDs differ")
    if errors:
        raise RuntimeError(
            "evaluation split does not match checkpoint training binding: "
            + "; ".join(errors)
        )
    return {
        "status": "verified",
        "verified": True,
        "manifest_path": str(manifest_path),
        "schema": training.get("schema"),
    }


def _reference_metadata(
    ota_root: Path, record: dict[str, object]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取 OTA 包记录及其关联的理想参考包元数据。"""

    ota_path = ota_root / "metadata" / f"{Path(str(record['ota_path'])).stem}.json"
    ota_metadata = _load_json(ota_path)
    reference_id = int(ota_metadata["reference"]["reference_id"])
    reference_path = ota_root.parent / "metadata" / f"{reference_id:06d}.json"
    return ota_metadata, _load_json(reference_path)


def reference_demod_symbols(
    metadata: dict[str, Any], sf: int
) -> dict[str, list[int]]:
    """把调制器符号 ID 转成接收机硬判决值，并只保留真值唯一的部分。"""

    from weak_decoder.decoding.payload_codec import (
        reencoded_payload_known_prefix_symbols,
    )

    n_bins = 1 << int(sf)
    ldro = bool(metadata["phy"]["ldro"])

    def convert(values: Iterable[int], reduced_rate: bool) -> list[int]:
        divisor = 4 if reduced_rate else 1
        return [((int(value) - 1) % n_bins) // divisor for value in values]

    symbols = metadata["symbols"]
    payload = convert(symbols["payload_ids"], ldro)
    known_payload_count = reencoded_payload_known_prefix_symbols(
        payload_len=int(metadata["packet"]["frame_bytes"]),
        has_crc=bool(metadata["phy"]["phy_crc"]),
        sf=int(sf),
        cr=int(metadata["phy"]["cr"]),
        ldro=ldro,
    )
    return {
        "header": convert(symbols["header_ids"], True),
        # 最后一个不完整 interleaver block 含发射机相关 padding，无法仅由
        # frame bytes 唯一恢复，因此不把这些符号作为可靠 SER 真值。
        "payload": payload[:known_payload_count],
    }


def score_demodulation(
    demod: Any | None,
    expected: dict[str, list[int]],
    section: str,
    *,
    include_symbol_details: bool = True,
) -> dict[str, Any]:
    """逐符号比较解调结果与参考值，返回错误位置和汇总 SER。"""

    # 本函数只用于干净输出已经通过 FrameSync 的包；加噪后若 Savaux 判决缺失，
    # 缺失位置仍算 symbol error，避免只挑完整解调包报告结果。
    stages = ("header", "payload") if section == "all" else (section,)
    expected_by_key = {
        (stage, index): int(value)
        for stage in stages
        for index, value in enumerate(expected[stage])
    }
    actual_by_key: dict[tuple[str, int], int] = {}
    if demod is not None:
        actual_by_key = {
            (str(item.stage), int(item.stage_symbol_index)): int(item.symbol_value)
            for item in demod.symbols
            if str(item.stage) in stages
        }
    errors = 0
    missing = 0
    rows: list[dict[str, Any]] = []
    for key, expected_value in expected_by_key.items():
        actual = actual_by_key.get(key)
        is_missing = actual is None
        is_error = is_missing or int(actual) != expected_value
        missing += int(is_missing)
        errors += int(is_error)
        if include_symbol_details:
            rows.append(
                {
                    "stage": key[0],
                    "stage_symbol_index": key[1],
                    "expected_symbol": expected_value,
                    "selected_symbol": actual,
                    "missing": is_missing,
                    "error": is_error,
                }
            )
    count = len(expected_by_key)
    return {
        "included": True,
        "exclusion_reason": None,
        "section": section,
        "symbol_count": count,
        "symbol_errors": errors,
        "missing_symbols": missing,
        "ser": None if count == 0 else float(errors / count),
        "complete": bool(demod is not None and missing == 0),
        "symbol_details_included": bool(include_symbol_details),
        "symbols": rows,
    }


def excluded_sync_score(section: str) -> dict[str, Any]:
    """为干净输出未通过 FrameSync 的包生成明确的非 SER 样本记录。"""

    return {
        "included": False,
        "exclusion_reason": "clean_framesync_failed",
        "section": str(section),
        "symbol_count": 0,
        "symbol_errors": 0,
        "missing_symbols": 0,
        "ser": None,
        "complete": False,
        "symbol_details_included": False,
        "symbols": [],
    }


def _sync_report(result: Any) -> dict[str, Any]:
    """把同步对象压缩成可序列化的状态和 CFO/STO/SFO 诊断信息。"""

    report: dict[str, Any] = {
        "status": str(result.status),
        "event_count": int(result.event_count),
        "synchronized": bool(result.synchronized),
        "error": result.error,
    }
    if result.frame_sync is not None:
        report.update(
            {
                "header_start_sample": int(result.frame_sync.fine_payload_start_sample),
                "cfo_int_bins": int(result.frame_sync.cfo_int_est),
                "cfo_frac_bins": float(result.frame_sync.cfo_frac_est),
                "sfo_chips_per_symbol": float(result.frame_sync.sfo_hat),
                "sto_fractional_chips": float(result.frame_sync.sto_frac_used),
            }
        )
    return report


def _demod_report(
    result: Any | None, *, include_symbol_details: bool = True
) -> dict[str, Any]:
    """序列化头部解析结果和每一个 Savaux 硬判决。"""

    if result is None:
        return {"status": "not_run", "header_valid": False}
    report = {
        "status": str(result.status),
        "header_valid": bool(result.header_valid),
        "decoded_payload_length": int(result.decoded_payload_length),
        "decoded_cr": int(result.decoded_cr),
        "decoded_has_crc": bool(result.decoded_has_crc),
        "decoded_ldro": bool(result.decoded_ldro),
        "decoded_payload_symbol_count": int(result.decoded_payload_symbol_count),
        "decision_count": len(result.symbols),
        "symbol_details_included": bool(include_symbol_details),
        "decisions": (
            [asdict(item) for item in result.symbols]
            if include_symbol_details
            else []
        ),
        "error": result.error,
    }
    return report


def _summary(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    """汇总干净包同步率，并只用干净 FrameSync 成功包计算加噪后 SER。"""

    output: dict[str, Any] = {}
    for method in methods:
        results = [row["methods"][method] for row in rows]
        included = [item for item in results if bool(item["score"]["included"])]
        count = sum(int(item["score"]["symbol_count"]) for item in included)
        errors = sum(int(item["score"]["symbol_errors"]) for item in included)
        synchronized = sum(
            int(item["clean_sync"]["synchronized"]) for item in results
        )
        output[method] = {
            "packet_count": len(results),
            "clean_synchronized_packets": synchronized,
            "clean_sync_success_rate": (
                None if not results else float(synchronized / len(results))
            ),
            "ser_packet_count": len(included),
            "header_valid_packets_after_noise": sum(
                int(item["demod_after_noise"]["header_valid"]) for item in results
            ),
            "complete_packets_after_noise": sum(
                int(item["score"]["complete"]) for item in results
            ),
            "symbol_count": count,
            "symbol_errors": errors,
            "ser": None if count == 0 else float(errors / count),
        }
    return output


def _paired_method_comparison(
    rows: list[dict[str, Any]],
    left_method: str,
    right_method: str,
) -> dict[str, Any]:
    """只在两支路 clean FrameSync 都成功的相同尝试上比较条件 SER。"""

    paired = [
        row
        for row in rows
        if bool(row["methods"][left_method]["score"]["included"])
        and bool(row["methods"][right_method]["score"]["included"])
    ]
    left_count = sum(
        int(row["methods"][left_method]["score"]["symbol_count"])
        for row in paired
    )
    right_count = sum(
        int(row["methods"][right_method]["score"]["symbol_count"])
        for row in paired
    )
    left_errors = sum(
        int(row["methods"][left_method]["score"]["symbol_errors"])
        for row in paired
    )
    right_errors = sum(
        int(row["methods"][right_method]["score"]["symbol_errors"])
        for row in paired
    )
    left_ser = None if left_count == 0 else float(left_errors / left_count)
    right_ser = None if right_count == 0 else float(right_errors / right_count)

    left_better = 0
    right_better = 0
    ties = 0
    for row in paired:
        left_packet_ser = row["methods"][left_method]["score"]["ser"]
        right_packet_ser = row["methods"][right_method]["score"]["ser"]
        if left_packet_ser is None or right_packet_ser is None:
            continue
        if math.isclose(
            float(left_packet_ser), float(right_packet_ser), abs_tol=1e-15
        ):
            ties += 1
        elif float(left_packet_ser) < float(right_packet_ser):
            left_better += 1
        else:
            right_better += 1

    difference = (
        None
        if left_ser is None or right_ser is None
        else float(left_ser - right_ser)
    )
    if difference is None:
        lower_ser_method = None
        higher_ser_method = None
    elif math.isclose(difference, 0.0, abs_tol=1e-15):
        lower_ser_method = "tie"
        higher_ser_method = "tie"
    elif difference < 0.0:
        lower_ser_method = left_method
        higher_ser_method = right_method
    else:
        lower_ser_method = right_method
        higher_ser_method = left_method
    return {
        "left_method": left_method,
        "right_method": right_method,
        "common_clean_sync_packet_attempts": len(paired),
        "common_unique_physical_packets": len(
            {str(row["physical_packet_uid"]) for row in paired}
        ),
        "left_symbol_count": left_count,
        "left_symbol_errors": left_errors,
        "left_ser": left_ser,
        "right_symbol_count": right_count,
        "right_symbol_errors": right_errors,
        "right_ser": right_ser,
        "ser_difference_left_minus_right": difference,
        "lower_ser_method": lower_ser_method,
        "higher_ser_method": higher_ser_method,
        "packet_attempts_left_better": left_better,
        "packet_attempts_right_better": right_better,
        "packet_attempts_tied": ties,
    }


def _aggregate_by_snr(
    conditions: list[dict[str, Any]], methods: list[str]
) -> list[dict[str, Any]]:
    """把多个 noise seed 的逐条件结果按 SNR 合并成正式比较表。"""

    grouped: dict[float | None, list[dict[str, Any]]] = {}
    for condition in conditions:
        key = condition["extra_snr_db"]
        grouped.setdefault(key, []).append(condition)

    output: list[dict[str, Any]] = []
    for extra_snr_db, group in grouped.items():
        rows = [row for condition in group for row in condition["packets"]]
        item: dict[str, Any] = {
            "label": _condition_label(extra_snr_db),
            "extra_snr_db": extra_snr_db,
            "noise_seeds": [
                int(condition["noise_seed"])
                for condition in group
                if condition["noise_seed"] is not None
            ],
            "unique_physical_packets": len(
                {str(row["physical_packet_uid"]) for row in rows}
            ),
            "packet_attempts": len(rows),
            "summary": _summary(rows, methods),
        }
        if "rfsr_1msps" in methods and "native_1msps" in methods:
            item["paired_rfsr_vs_native"] = _paired_method_comparison(
                rows, "rfsr_1msps", "native_1msps"
            )
        output.append(item)
    return output


def _run_clean_sync_task(
    task: tuple[int, str, np.ndarray, Any, Any],
) -> tuple[int, str, Any]:
    """在线程池中执行一条干净波形的独立 FrameSync。"""

    packet_index, method, samples, sync_config, sync_runner = task
    return int(packet_index), str(method), sync_runner(samples, sync_config)


def _demodulate_packet_after_noise(
    task: tuple[
        int,
        int,
        dict[str, Any],
        float | None,
        int | None,
        list[str],
        int,
        float,
        int,
        int,
        Any,
        bool,
    ],
) -> tuple[int, int, dict[str, Any], dict[str, Any | None]]:
    """对一个包的全部对照支路加同一噪声并执行 Savaux。

    任务粒度刻意设为“一个包、一个 SNR/seed、全部 method”。这样同一物理包的
    两条支路仍在一个线程中生成同一标准复噪声，而线程之间只共享只读 IQ 与冻结的
    FrameSync，不发生状态竞争。
    """

    (
        condition_index,
        packet_index,
        packet,
        extra_snr_db,
        noise_seed,
        methods,
        sf,
        bw_hz,
        os_factor,
        ldro_mode,
        demodulator,
        include_symbol_details,
    ) = task
    record = packet["record"]
    reference_power = float(packet["noise_reference_power"])
    noise_power = (
        0.0
        if extra_snr_db is None
        else float(reference_power / (10.0 ** (float(extra_snr_db) / 10.0)))
    )
    method_rows: dict[str, Any] = {}
    demodulations: dict[str, Any | None] = {}
    for method in methods:
        clean_state = packet["methods"][method]
        clean_values = clean_state["samples"]
        clean_sync = clean_state["sync"]
        # 无额外噪声基准不需要无意义复制整条 1 MS/s 波形。
        noisy_values = (
            clean_values
            if extra_snr_db is None
            else _add_extra_awgn(
                clean_values,
                extra_snr_db,
                _packet_rng(
                    int(noise_seed), int(packet["ordinal"])
                ),
                reference_power=reference_power,
            )
        )
        demod = None
        if clean_sync.synchronized and clean_sync.frame_sync is not None:
            demod = demodulator(
                noisy_values,
                clean_sync.frame_sync,
                sf=int(sf),
                bw_hz=float(bw_hz),
                os_factor=int(os_factor),
                ldro_mode=int(ldro_mode),
            )
        demodulations[method] = demod
        method_rows[method] = {
            "clean_sync": _sync_report(clean_sync),
            "added_noise_power": noise_power,
            "demod_after_noise": _demod_report(
                demod, include_symbol_details=include_symbol_details
            ),
        }
    row = {
        "physical_packet_uid": str(record["split_group"]),
        "view_id": str(record["view_id"]),
        "source_snr_db": float(packet["source_snr_db"]),
        "rfsr_conditioning_snr_db": float(packet["source_snr_db"]),
        "noise_reference_power": reference_power,
        "methods": method_rows,
    }
    return int(condition_index), int(packet_index), row, demodulations


def main() -> None:
    """执行完整评估流程并将所有复现实验信息写入一个 JSON 文件。"""

    started_at = perf_counter()
    args = parse_args()
    if args.ota_max_groups is None or int(args.ota_max_groups) < 3:
        raise ValueError("--ota-max-groups must be at least 3")
    if args.limit is not None and int(args.limit) < 1:
        raise ValueError("--limit must be positive")
    if int(args.workers) < 0:
        raise ValueError("--workers must be zero (automatic) or positive")
    methods = list(dict.fromkeys(args.method or ["rfsr_1msps"]))
    noise_seeds = resolve_noise_seeds(args)

    # 延迟导入 PyTorch/RFSR，使纯参数与评分单元测试无需初始化模型。
    from rfsr.nn.ota_dataset import OTALoRaDataset
    # 直接指向当前实现，使本入口不依赖 os_lora 顶层的 GLS 兼容 API。
    from weak_decoder.os_lora.system.synchronized_savaux import (
        demod_synchronized_savaux,
    )
    from weak_decoder.rf_super_resolution import (
        RFSRFrontendConfig,
        RFSuperResolutionFrontend,
        default_rfsr_repo_root,
    )
    from weak_decoder.synchronization import (
        SinglePacketSyncConfig,
        run_single_packet_sync,
    )

    ota_root = args.ota_root.expanduser().resolve()
    # 用与微调完全相同的 seed/max_groups 重建物理包划分，并与 sidecar 交叉校验。
    datasets = {
        split: OTALoRaDataset(
            dataset_root=ota_root,
            split=split,
            split_seed=int(args.ota_split_seed),
            max_groups=int(args.ota_max_groups),
            target_source="received",
            return_snr=True,
        )
        for split in ("train", "validation", "test")
    }
    split_manifest = _split_manifest(
        datasets, int(args.ota_split_seed), int(args.ota_max_groups)
    )
    checkpoint_binding = _checkpoint_split_binding(
        args.checkpoint,
        split_manifest,
        allow_unbound=bool(args.allow_unbound_checkpoint),
    )
    test_dataset = datasets["test"]
    # 每个物理包只评估一个 canonical ADC0/q0 视图，避免把同包视图当独立样本。
    selected = [
        index
        for index, record in enumerate(test_dataset.records)
        if int(record["adc_phase"]) == 0 and int(record["lowrate_phase"]) == 0
    ]
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    if not selected:
        raise RuntimeError("test split contains no canonical ADC0/q0 packet views")

    # RFSR frontend 负责长序列分块推理，并记录 checkpoint SHA/代码版本。
    frontend = RFSuperResolutionFrontend(
        RFSRFrontendConfig(
            repo_root=default_rfsr_repo_root(),
            checkpoint=args.checkpoint,
            device=str(args.device),
        )
    )
    sync_config = SinglePacketSyncConfig(
        sf=int(args.sf),
        bw_hz=float(args.bw),
        sample_rate_hz=1e6,
        center_frequency_hz=float(args.center_frequency_hz),
        preamble_symbols=int(args.preamble_symbols),
        sync_word=int(args.sync_word),
        detection_chirps=int(args.sync_detection_chirps),
        scan_chirps=int(args.sync_scan_chirps),
    )

    # 第一阶段：每个物理包只做一次不含额外噪声的 RFSR。单张 GPU 上多个 Python
    # 进程通常只会争抢显存与 CUDA context，因此这部分明确保持单进程顺序推理。
    prepare_started_at = perf_counter()
    prepared_packets: list[dict[str, Any]] = []
    for ordinal, index in enumerate(selected):
        _, y, snr = test_dataset[index]
        record = test_dataset.records[index]
        source_high = _complex_from_channels(y)
        clean_low = source_high[::4].copy()
        source_snr_db = float(snr.item())

        clean_samples: dict[str, np.ndarray] = {}
        if "rfsr_1msps" in methods:
            clean_samples["rfsr_1msps"] = frontend.enhance(
                clean_low, source_snr_db
            )
        if "interpolation_1msps" in methods:
            clean_samples["interpolation_1msps"] = frontend.interpolate(clean_low)
        if "native_1msps" in methods:
            clean_samples["native_1msps"] = source_high
        sample_lengths = {int(values.size) for values in clean_samples.values()}
        if len(sample_lengths) != 1:
            raise RuntimeError(
                "paired method outputs must have equal lengths for shared AWGN"
            )

        clean_methods: dict[str, dict[str, Any]] = {}
        for method in methods:
            values = clean_samples[method]
            clean_methods[method] = {"samples": values}
        prepared_packets.append(
            {
                "ordinal": ordinal,
                "record": record,
                "source_snr_db": source_snr_db,
                # 两条支路共享原生 1 MS/s OTA 的参考功率，后续得到完全相同的
                # AWGN 功率；不能按 RFSR 输出功率单独缩放，否则比较会失真。
                "noise_reference_power": float(
                    np.mean(np.abs(source_high) ** 2)
                ),
                "methods": clean_methods,
            }
        )
    rfsr_prepare_seconds = perf_counter() - prepare_started_at

    # 展开全部条件后按实际任务量选择线程数。这里的可见 CPU 数已考虑 affinity 和
    # cgroup 配额；--workers 0 不会误把宿主机的全部核都拿来开线程。
    condition_specs: list[tuple[float | None, int | None]] = []
    for extra_snr_db in resolve_extra_snr_points(args):
        if extra_snr_db is None:
            # 不加噪基准与 seed 无关，只需解调一次，不能按 seed 重复扩大样本数。
            condition_specs.append((None, None))
        else:
            condition_specs.extend(
                (float(extra_snr_db), int(noise_seed))
                for noise_seed in noise_seeds
            )

    clean_sync_task_count = len(prepared_packets) * len(methods)
    savaux_task_count = len(prepared_packets) * len(condition_specs)
    worker_count = resolve_worker_count(
        int(args.workers), max(clean_sync_task_count, savaux_task_count)
    )
    print(
        json.dumps(
            {
                "parallel_backend": "thread_pool",
                "available_cpu_count": available_cpu_count(),
                "worker_count": worker_count,
                "clean_framesync_tasks": clean_sync_task_count,
                "savaux_packet_tasks": savaux_task_count,
                "rfsr_inference": "single_process",
            }
        )
    )

    # 第二阶段：RFSR 输出已经固定，干净 FrameSync 完全是 CPU/NumPy 工作。不同包
    # 与不同 method 彼此无共享可写状态，因此可按包并行；得到的同步结果仍只算一次。
    sync_started_at = perf_counter()
    clean_sync_tasks = [
        (packet_index, method, packet["methods"][method]["samples"], sync_config, run_single_packet_sync)
        for packet_index, packet in enumerate(prepared_packets)
        for method in methods
    ]
    if worker_count == 1:
        clean_sync_results = [
            _run_clean_sync_task(task) for task in clean_sync_tasks
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            clean_sync_results = list(
                executor.map(_run_clean_sync_task, clean_sync_tasks)
            )
    for packet_index, method, clean_sync in clean_sync_results:
        prepared_packets[packet_index]["methods"][method]["sync"] = clean_sync
    clean_sync_seconds = perf_counter() - sync_started_at

    # 第三阶段：只在已经完成 RFSR 和 FrameSync 的 1 MS/s 波形上加噪。每个任务
    # 包含一个物理包的全部 method，确保配对支路收到完全相同的标准复噪声实现。
    conditions: list[dict[str, Any]] = [
        {
            "label": _condition_label(extra_snr_db),
            "extra_snr_db": extra_snr_db,
            "noise_seed": noise_seed,
            "packets": [None] * len(prepared_packets),
        }
        for extra_snr_db, noise_seed in condition_specs
    ]
    demod_objects: dict[tuple[int, int, str], Any | None] = {}
    demod_started_at = perf_counter()
    demod_tasks = [
        (
            condition_index,
            packet_index,
            packet,
            extra_snr_db,
            noise_seed,
            methods,
            int(args.sf),
            float(args.bw),
            sync_config.os_factor,
            int(args.ldro),
            demod_synchronized_savaux,
            bool(args.save_symbol_details),
        )
        for condition_index, (extra_snr_db, noise_seed) in enumerate(condition_specs)
        for packet_index, packet in enumerate(prepared_packets)
    ]
    if worker_count == 1:
        demod_results = [
            _demodulate_packet_after_noise(task) for task in demod_tasks
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            demod_results = list(
                executor.map(_demodulate_packet_after_noise, demod_tasks)
            )
    for condition_index, packet_index, row, demodulations in demod_results:
        conditions[condition_index]["packets"][packet_index] = row
        for method, demod in demodulations.items():
            demod_objects[(condition_index, packet_index, method)] = demod
    savaux_demodulation_seconds = perf_counter() - demod_started_at

    # 第四阶段：所有 SNR、所有支路的硬判决完成后才加载 reference，防止真值进入
    # FrameSync 或 Savaux。clean sync 失败包保留诊断记录，但明确排除在 SER 外。
    scoring_started_at = perf_counter()
    for packet_index, packet in enumerate(prepared_packets):
        ota_metadata, reference_metadata = _reference_metadata(
            ota_root, packet["record"]
        )
        expected = reference_demod_symbols(reference_metadata, int(args.sf))
        reference_id = int(ota_metadata["reference"]["reference_id"])
        for condition_index, condition in enumerate(conditions):
            row = condition["packets"][packet_index]
            row["reference_id"] = reference_id
            for method in methods:
                method_row = row["methods"][method]
                clean_sync = packet["methods"][method]["sync"]
                demod = demod_objects.pop((condition_index, packet_index, method))
                method_row["score"] = (
                    score_demodulation(
                        demod,
                        expected,
                        str(args.ser_section),
                        include_symbol_details=bool(args.save_symbol_details),
                    )
                    if clean_sync.synchronized
                    else excluded_sync_score(str(args.ser_section))
                )

    for condition in conditions:
        condition["summary"] = _summary(condition["packets"], methods)

    aggregate_by_snr = _aggregate_by_snr(conditions, methods)
    scoring_and_aggregation_seconds = perf_counter() - scoring_started_at
    for aggregate in aggregate_by_snr:
        console = {
            "snr": aggregate["label"],
            "packet_attempts": aggregate["packet_attempts"],
            "ser": {
                method: aggregate["summary"][method]["ser"]
                for method in methods
            },
        }
        paired = aggregate.get("paired_rfsr_vs_native")
        if paired is not None:
            console["rfsr_minus_native"] = paired[
                "ser_difference_left_minus_right"
            ]
            console["lower_ser_method"] = paired["lower_ser_method"]
        print(json.dumps(console))

    # 输出同时保存逐符号证据、汇总指标、split 和模型 provenance，便于复核。
    payload = {
        "schema": "lora-rfsr-clean-sync-then-noisy-savaux-ser-v5",
        "data_contract": (
            "held-out received OTA -> clean q0 250 kS/s -> RFSR 1 MS/s -> "
            "clean packet detection/CFO-STO-SFO FrameSync once -> freeze FrameSync -> "
            "add identical native-power-referenced AWGN to clean method outputs -> "
            "Savaux with frozen FrameSync; "
            "SER uses clean-FrameSync-success packets only; GT loaded after all decisions"
        ),
        "noise_injection_stage": "after_rfsr_and_clean_framesync_before_savaux",
        "noise_power_reference": (
            "per-packet mean power of native received 1 MS/s OTA before extra AWGN; "
            "same complex AWGN realization and power used for paired methods"
        ),
        "framesync_policy": "estimate_once_on_clean_method_output_and_freeze_across_snr",
        "ser_section": str(args.ser_section),
        "symbol_details_included": bool(args.save_symbol_details),
        "ser_reference_policy": (
            "score only symbols uniquely determined by frame bytes; exclude "
            "the final transmitter-specific padding-only interleaver block"
        ),
        "methods": methods,
        "noise_seeds": noise_seeds,
        "split_manifest": split_manifest,
        "checkpoint_split_binding": checkpoint_binding,
        "evaluated_test_packets": [
            str(test_dataset.records[index]["split_group"]) for index in selected
        ],
        "rfsr_snr_conditioning": "manifest_source_snr_before_extra_awgn",
        "rfsr": asdict(frontend.provenance),
        "sync_config": asdict(sync_config),
        "execution": {
            "parallel_backend": "thread_pool",
            "available_cpu_count": available_cpu_count(),
            "worker_count": worker_count,
            "rfsr_inference": "single_process",
            "timings_seconds": {
                "rfsr_prepare": rfsr_prepare_seconds,
                "clean_framesync": clean_sync_seconds,
                "savaux_demodulation": savaux_demodulation_seconds,
                "scoring_and_aggregation": scoring_and_aggregation_seconds,
                "before_json_write_total": perf_counter() - started_at,
            },
        },
        "conditions": conditions,
        "aggregate_by_snr": aggregate_by_snr,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote synchronized Savaux SER evaluation: {output}")


if __name__ == "__main__":
    main()
