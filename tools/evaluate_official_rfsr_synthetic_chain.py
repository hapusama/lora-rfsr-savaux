#!/usr/bin/env python3
"""在合成 LoRa IQ 上对公开 RF-SR checkpoint 做 go/no-go 审计。

仓库内置的上游代码不包含公开 OTA IQ 数据集，因此本工具严格限制结论范围：
先复现作者本地的合成解码链，再在固定合成包集合上比较普通 FFT、Savaux 和
branch GLS。整个过程不下载数据，也不训练模型。

实验评估三种噪声位置：

* ``pre_rfsr``：在 2 MS/s 加入 AWGN，抽取到 250 kS/s，运行前端，再从含噪
  前端输出重新估计 FrameSync；
* ``post_framesync_common_power``：冻结干净 FrameSync，在所有前端的 1 MS/s
  输出上加入相同绝对功率的 AWGN；
* ``post_framesync_gain_matched``：复用同一份归一化噪声 realization，并根据
  各前端的干净输出功率缩放到指定 SNR。

同步失败时，该包全部预期符号都计为端到端符号错误。同时只在同步成功包上报告
条件 SER，从而区分同步失败和解调失败。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RFSR_ROOT = REPO_ROOT / "third_party" / "rfsr"
for _path in (REPO_ROOT, RFSR_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rfsr import awgn, decode, encode  # noqa: E402
from rfsr.PHY import encode_raw_phy  # noqa: E402
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

METHODS = (
    "native_1msps",
    "official_interpolation",
    "official_synthetic_rfsr",
    "official_ota_rfsr",
)
DECODERS = ("ordinary_fft", "savaux", "savaux_gls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, default=8)
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
    return args


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
) -> dict[str, np.ndarray]:
    high = np.asarray(high_rate_samples, dtype=np.complex64)
    low = high[:: HIGH_RATE_HZ // LOW_RATE_HZ]
    return {
        "native_1msps": np.asarray(
            high[:: HIGH_RATE_HZ // OUTPUT_RATE_HZ], dtype=np.complex64
        ),
        "official_interpolation": synthetic_frontend.interpolate(low),
        "official_synthetic_rfsr": synthetic_frontend.enhance(low, snr_db=snr_db),
        "official_ota_rfsr": ota_frontend.enhance(low, snr_db=snr_db),
    }


def _expected_demod_symbols(symbol_ids: Iterable[int], reduced_rate: bool) -> list[int]:
    divisor = 4 if bool(reduced_rate) else 1
    return [((int(value) - 1) % N_BINS) // divisor for value in symbol_ids]


def _expected_symbols(encoded: Any) -> dict[str, list[int]]:
    return {
        "header": _expected_demod_symbols(encoded.header_symbol_ids, True),
        "payload": _expected_demod_symbols(encoded.payload_symbol_ids, LDRO),
    }


def _sync_config() -> SinglePacketSyncConfig:
    return SinglePacketSyncConfig(
        sf=SF,
        bw_hz=BW_HZ,
        sample_rate_hz=OUTPUT_RATE_HZ,
        center_frequency_hz=915_000_000,
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
) -> tuple[np.ndarray, np.ndarray]:
    """用一次批量 FFT 调度计算整包所有符号的论文 Eq.36/37。"""

    values = np.asarray(samples, dtype=np.complex64)
    symbol_samples = N_BINS * OS_FACTOR
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
        os_factor=OS_FACTOR,
    )
    fractional = np.exp(
        -2j * np.pi * float(cfo_frac) * sample_index / float(symbol_samples)
    )
    downchirp = np.asarray(np.conjugate(reference) * fractional, dtype=np.complex64)
    # 连续 CFO 的公共相位对每个符号只是一个标量，在 |Eq.37|^2 和 GLS 投影的
    # 模平方中都会抵消，因此这里无需显式应用。
    dechirped = np.asarray(symbols * downchirp[None, :], dtype=np.complex64)
    branch_samples = dechirped.reshape(len(starts), N_BINS, OS_FACTOR)
    spectra = np.fft.fft(branch_samples, axis=1) / math.sqrt(float(N_BINS))

    if OS_FACTOR > 1:
        # 复用论文实现中精确的 chirp-z 尾部校正，并把所有符号的 q>0 branch
        # 合并到 branch-count 轴上进行批处理。
        tail_input = np.transpose(branch_samples[:, :, 1:], (1, 0, 2)).reshape(
            N_BINS, len(starts) * (OS_FACTOR - 1)
        )
        tails = _wrapped_tail_dft_batch(tail_input).reshape(
            N_BINS, len(starts), OS_FACTOR - 1
        )
        tails = np.transpose(tails, (1, 0, 2))
        wrap_phases = np.exp(
            2j * np.pi * np.arange(1, OS_FACTOR, dtype=np.float64) / OS_FACTOR
        )
        spectra[:, :, 1:] += (
            (wrap_phases - 1.0)[None, None, :] * tails
        ) / math.sqrt(float(N_BINS))

    bins = np.arange(N_BINS, dtype=np.float64)[None, :, None]
    branches = np.arange(OS_FACTOR, dtype=np.float64)[None, None, :]
    alignment = np.exp(
        -2j * np.pi * bins * branches / float(N_BINS * OS_FACTOR)
    )
    aligned = np.asarray(spectra * alignment, dtype=np.complex64)
    combined = np.asarray(np.sum(aligned, axis=2), dtype=np.complex64)
    return combined, np.asarray(spectra, dtype=np.complex64)


def evaluate_decoders(
    samples: np.ndarray,
    sync_result: Any,
    expected: dict[str, list[int]],
    noise_model: BranchNoiseModel,
) -> dict[str, dict[str, Any]]:
    """在共享同一份 FrameSync 结果后，对全部预期符号计分。"""

    expected_values = list(expected["header"]) + list(expected["payload"])
    count = len(expected_values)
    if not sync_result.synchronized or sync_result.frame_sync is None:
        return {name: _empty_decoder_score(count) for name in DECODERS}

    frame_sync = sync_result.frame_sync
    try:
        ordinary = demod_symbol_sequence(
            samples=np.asarray(samples, dtype=np.complex64),
            header_start_sample=int(frame_sync.fine_payload_start_sample),
            sf=SF,
            os_factor=OS_FACTOR,
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
        return {name: _empty_decoder_score(count) for name in DECODERS}

    ordinary_values = [int(item.symbol_value) for item in ordinary]
    ordinary_margins = [float(item.peak_margin_db) for item in ordinary]
    try:
        combined_batch, branch_batch = _batch_savaux_spectra(
            samples,
            [int(item.start_sample) + OS_FACTOR // 2 for item in ordinary],
            cfo_int=int(frame_sync.cfo_int_est),
            cfo_frac=float(frame_sync.cfo_frac_est),
        )
    except ValueError:
        return {
            "ordinary_fft": _score_values(
                ordinary_values, expected_values, ordinary_margins
            ),
            "savaux": _empty_decoder_score(count),
            "savaux_gls": _empty_decoder_score(count),
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
        gls = branch_gls_scores(
            tuple(branch_batch[index, :, q] for q in range(OS_FACTOR)),
            os_factor=OS_FACTOR,
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

    return {
        "ordinary_fft": _score_values(
            ordinary_values, expected_values, ordinary_margins
        ),
        "savaux": _score_values(savaux_values, expected_values, savaux_margins),
        "savaux_gls": _score_values(gls_values, expected_values, gls_margins),
    }


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


def _active_power(samples: np.ndarray) -> float:
    active_start = LEADING_SILENCE_HIGH // (HIGH_RATE_HZ // OUTPUT_RATE_HZ)
    active_stop = -TRAILING_SILENCE_HIGH // (HIGH_RATE_HZ // OUTPUT_RATE_HZ)
    active = np.asarray(samples, dtype=np.complex64)[active_start:active_stop]
    return float(np.mean(np.abs(active).astype(np.float64) ** 2))


def _row(
    *,
    stage: str,
    snr_db: float | None,
    packet_index: int,
    method: str,
    sync_result: Any,
    scores: dict[str, dict[str, Any]],
    input_snr_measurement: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    return {
        "stage": str(stage),
        "target_snr_db": None if snr_db is None else float(snr_db),
        "snr_db": None if snr_db is None else float(snr_db),
        "input_snr_measurement": input_snr_measurement,
        "packet_index": int(packet_index),
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
        for decoder in DECODERS:
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


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    print(f"device={device}", flush=True)
    synthetic_frontend = _frontend(
        DEFAULT_SYNTHETIC_CHECKPOINT, device, int(args.chunk_input_samples)
    )
    ota_frontend = _frontend(
        DEFAULT_OTA_CHECKPOINT, device, int(args.chunk_input_samples)
    )

    decode_reproduction = _official_decode_reproduction(
        synthetic_frontend,
        [float(value) for value in args.snrs],
        int(args.official_decode_trials),
        int(args.seed),
    )

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
    reference_power = float(
        np.mean(
            [
                np.mean(
                    np.abs(
                        packet.samples[
                            LEADING_SILENCE_HIGH:-TRAILING_SILENCE_HIGH
                        ]
                    ).astype(np.float64)
                    ** 2
                )
                for packet in encoded_packets
            ]
        )
    )

    print("estimating pre-RFSR GLS noise models", flush=True)
    pre_noise_models: dict[float, dict[str, BranchNoiseModel]] = {}
    covariance_reports: dict[str, dict[str, dict[str, Any]]] = {}
    for snr_db in args.snrs:
        models, report = estimate_pre_rfsr_noise_models(
            float(snr_db),
            reference_power,
            synthetic_frontend,
            ota_frontend,
            args,
        )
        pre_noise_models[float(snr_db)] = models
        covariance_reports[str(float(snr_db))] = report
    post_noise_model = identity_branch_noise_model(OS_FACTOR)

    rows: list[dict[str, Any]] = []
    clean_output_diagnostics: list[dict[str, Any]] = []
    sync_config = _sync_config()
    for packet_index, encoded in enumerate(encoded_packets):
        print(f"packet {packet_index + 1}/{len(encoded_packets)}", flush=True)
        expected = _expected_symbols(encoded)
        clean_high = np.asarray(encoded.samples, dtype=np.complex64)
        clean_methods = _method_outputs(
            clean_high,
            synthetic_frontend,
            ota_frontend,
            snr_db=0.0,
        )
        clean_sync = {
            method: run_single_packet_sync(values, sync_config)
            for method, values in clean_methods.items()
        }
        native_power = _active_power(clean_methods["native_1msps"])
        clean_powers = {
            method: _active_power(values) for method, values in clean_methods.items()
        }
        for method, values in clean_methods.items():
            power = clean_powers[method]
            clean_scores = evaluate_decoders(
                values,
                clean_sync[method],
                expected,
                post_noise_model,
            )
            rows.append(
                _row(
                    stage="clean",
                    snr_db=None,
                    packet_index=packet_index,
                    method=method,
                    sync_result=clean_sync[method],
                    scores=clean_scores,
                )
            )
            clean_output_diagnostics.append(
                {
                    "packet_index": packet_index,
                    "method": method,
                    "active_power": power,
                    "gain_vs_native_db": float(
                        10.0 * math.log10((power + 1e-30) / (native_power + 1e-30))
                    ),
                    "sync": _sync_report(clean_sync[method]),
                }
            )

        for snr_db_value in args.snrs:
            snr_db = float(snr_db_value)
            noise_power = snr_noise_power(reference_power, snr_db)

            post_rng = np.random.default_rng(
                int(args.noise_seed) + packet_index * 1000
            )
            unit_post_noise = complex_awgn(
                len(clean_methods["native_1msps"]), 1.0, post_rng
            )
            for method, clean_values in clean_methods.items():
                paired_post_noise = np.asarray(
                    unit_post_noise * math.sqrt(noise_power), dtype=np.complex64
                )
                noisy_values = np.asarray(
                    clean_values + paired_post_noise, dtype=np.complex64
                )
                scores = evaluate_decoders(
                    noisy_values,
                    clean_sync[method],
                    expected,
                    post_noise_model,
                )
                rows.append(
                    _row(
                        stage="post_framesync_common_power",
                        snr_db=snr_db,
                        packet_index=packet_index,
                        method=method,
                        sync_result=clean_sync[method],
                        scores=scores,
                    )
                )
                matched_noise_power = snr_noise_power(clean_powers[method], snr_db)
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
                )
                rows.append(
                    _row(
                        stage="post_framesync_gain_matched",
                        snr_db=snr_db,
                        packet_index=packet_index,
                        method=method,
                        sync_result=clean_sync[method],
                        scores=gain_matched_scores,
                    )
                )

            pre_rng = np.random.default_rng(
                int(args.noise_seed) + 1_000_000 + packet_index * 1000
            )
            paired_high_noise = complex_awgn(len(clean_high), noise_power, pre_rng)
            noisy_high = np.asarray(clean_high + paired_high_noise, dtype=np.complex64)
            input_snr_measurement = measure_decimated_snr(
                clean_high,
                noisy_high,
                decimation=HIGH_RATE_HZ // LOW_RATE_HZ,
                leading_silence_high_rate=LEADING_SILENCE_HIGH,
                trailing_silence_high_rate=TRAILING_SILENCE_HIGH,
                measured_sample_rate_hz=LOW_RATE_HZ,
            )
            noisy_methods = _method_outputs(
                noisy_high,
                synthetic_frontend,
                ota_frontend,
                snr_db=snr_db,
            )
            for method, noisy_values in noisy_methods.items():
                sync_result = run_single_packet_sync(noisy_values, sync_config)
                scores = evaluate_decoders(
                    noisy_values,
                    sync_result,
                    expected,
                    pre_noise_models[snr_db][method],
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
                    )
                )

    synthetic_checkpoint = RFSR_ROOT / "checkpoints" / DEFAULT_SYNTHETIC_CHECKPOINT
    ota_checkpoint = RFSR_ROOT / "checkpoints" / DEFAULT_OTA_CHECKPOINT
    output = {
        "schema": "official-rfsr-synthetic-chain-audit",
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "claim": "published checkpoints on the vendored synthetic generation path",
            "official_ota_dataset_used": False,
            "download_performed": False,
            "training_performed": False,
            "limitation": (
                "The vendored upstream tree contains checkpoints and code but no official "
                "OTA IQ examples; this is not an official OTA-dataset reproduction."
            ),
        },
        "configuration": {
            "packets": int(args.packets),
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
            "trailing_guard_samples_high_rate": TRAILING_SILENCE_HIGH,
            "reference_signal_power": reference_power,
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
            "pre_rfsr": (
                "paired AWGN at 2 MS/s -> decimation -> measured SNR at 250 kS/s "
                "active packet interval -> frontend -> noisy FrameSync"
            ),
            "snr_reporting": (
                "snr_db/target_snr_db is the requested injection point; "
                "input_snr_measurement.snr_db is measured from clean and noisy "
                "250 kS/s IQ after decimation"
            ),
            "post_framesync_common_power": (
                "clean frontend -> clean frozen FrameSync -> identical absolute AWGN at 1 MS/s"
            ),
            "post_framesync_gain_matched": (
                "clean frontend -> clean frozen FrameSync -> common normalized AWGN "
                "scaled to each frontend's clean output power"
            ),
            "failed_sync_scoring": "all expected symbols count as errors",
            "cross_snr_pairing": (
                "each packet reuses one normalized noise realization across SNRs"
            ),
            "post_gls_noise_model": "identity; white post-frontend AWGN",
            "pre_gls_noise_model": (
                "pooled 8x8 covariance from held-out pure noise passed through each frontend"
            ),
        },
        "official_decode_reproduction": decode_reproduction,
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
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
