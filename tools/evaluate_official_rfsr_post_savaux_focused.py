#!/usr/bin/env python3
"""聚焦比较官方 OTA 上 Native 与 OTA-RFSR 的后置噪声 Savaux 曲线。

每条方法先在干净波形上独立完成 FrameSync，再冻结同步参数。后续复 AWGN 只
加入已完成前端处理的波形，因此本工具衡量的是波形表示对 Savaux 解调的帮助，
不衡量含噪 FrameSync 或 RFSR 对新增噪声的去噪能力。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for _path in (REPO_ROOT, TOOLS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_official_rfsr_synthetic_chain as audit  # noqa: E402
from weak_decoder.baselines.savaux_oversampled.paper_oversampled_demod import (  # noqa: E402
    _wrapped_tail_dft_batch,
)
from weak_decoder.chirp import bin_to_grlora_symbol, build_upchirp  # noqa: E402
from weak_decoder.decoding.header_first_demod import (  # noqa: E402
    demod_symbol_sequence,
)


METHODS = (
    "native_250ksps",
    "native_1msps",
    "official_ota_rfsr_1msps",
)
METHOD_SAMPLE_RATES = {
    "native_250ksps": audit.LOW_RATE_HZ,
    "native_1msps": audit.OUTPUT_RATE_HZ,
    "official_ota_rfsr_1msps": audit.OUTPUT_RATE_HZ,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-ota-root", type=Path, required=True)
    parser.add_argument("--packets", type=int, default=4)
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=float,
        default=[-14, -15, -16, -17, -18, -20, -22, -24, -25],
    )
    parser.add_argument(
        "--noise-seeds",
        nargs="+",
        type=int,
        default=[20260731, 20260732, 20260733],
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-input-samples", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if int(args.packets) <= 0:
        parser.error("--packets must be positive")
    if not args.snrs:
        parser.error("--snrs must contain at least one value")
    if not args.noise_seeds:
        parser.error("--noise-seeds must contain at least one value")
    if int(args.workers) <= 0:
        parser.error("--workers must be positive")
    return args


def _active_output_slice(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    leading_guard_high: int,
    trailing_guard_high: int,
) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex64)
    factor = audit.HIGH_RATE_HZ // int(sample_rate_hz)
    start = (int(leading_guard_high) + factor - 1) // factor
    high_stop = values.size * factor - int(trailing_guard_high)
    stop = (high_stop + factor - 1) // factor
    if start < 0 or stop <= start or stop > values.size:
        raise ValueError("guard intervals leave no active output samples")
    return values[start:stop]


def _active_power(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    leading_guard_high: int,
    trailing_guard_high: int,
) -> float:
    active = _active_output_slice(
        samples,
        sample_rate_hz=sample_rate_hz,
        leading_guard_high=leading_guard_high,
        trailing_guard_high=trailing_guard_high,
    )
    return audit._finite_positive_power(active, label="focused Savaux output")


def paired_unit_noise(unit_1m: np.ndarray, method: str) -> np.ndarray:
    """让 250 kS/s 支路复用 1 MS/s 噪声在共同采样时刻的 realization。"""

    values = np.asarray(unit_1m, dtype=np.complex64)
    if method == "native_250ksps":
        return values[::4]
    if method in {"native_1msps", "official_ota_rfsr_1msps"}:
        return values
    raise ValueError(f"unsupported focused method: {method}")


def batch_savaux_spectra(
    samples: np.ndarray,
    starts: list[int],
    *,
    sf: int,
    os_factor: int,
    cfo_int: int,
    cfo_frac: float,
) -> np.ndarray:
    """批量计算任意整数 OS 下的 Savaux Eq.36/37 合并频谱。"""

    values = np.asarray(samples, dtype=np.complex64)
    n_bins = 1 << int(sf)
    os_value = int(os_factor)
    symbol_samples = n_bins * os_value
    symbols = np.stack(
        [values[int(start) : int(start) + symbol_samples] for start in starts],
        axis=0,
    )
    if symbols.shape != (len(starts), symbol_samples):
        raise ValueError("one or more Savaux symbols exceed the input sample range")

    sample_index = np.arange(symbol_samples, dtype=np.float64)
    reference = build_upchirp(
        sf=int(sf), symbol_id=int(cfo_int), os_factor=os_value
    )
    fractional = np.exp(
        -2j * np.pi * float(cfo_frac) * sample_index / float(symbol_samples)
    )
    downchirp = np.asarray(np.conjugate(reference) * fractional, dtype=np.complex64)
    dechirped = np.asarray(symbols * downchirp[None, :], dtype=np.complex64)
    branches = dechirped.reshape(len(starts), n_bins, os_value)
    spectra = np.fft.fft(branches, axis=1) / math.sqrt(float(n_bins))

    if os_value > 1:
        tail_input = np.transpose(branches[:, :, 1:], (1, 0, 2)).reshape(
            n_bins, len(starts) * (os_value - 1)
        )
        tails = _wrapped_tail_dft_batch(tail_input).reshape(
            n_bins, len(starts), os_value - 1
        )
        tails = np.transpose(tails, (1, 0, 2))
        wrap_phases = np.exp(
            2j * np.pi * np.arange(1, os_value, dtype=np.float64) / os_value
        )
        spectra[:, :, 1:] += (
            (wrap_phases - 1.0)[None, None, :] * tails
        ) / math.sqrt(float(n_bins))

    bins = np.arange(n_bins, dtype=np.float64)[None, :, None]
    branch_ids = np.arange(os_value, dtype=np.float64)[None, None, :]
    alignment = np.exp(
        -2j * np.pi * bins * branch_ids / float(n_bins * os_value)
    )
    return np.asarray(np.sum(spectra * alignment, axis=2), dtype=np.complex64)


def evaluate_savaux(
    samples: np.ndarray,
    sync_result: Any,
    expected: dict[str, list[int]],
    *,
    os_factor: int,
) -> dict[str, Any]:
    expected_values = list(expected["header"]) + list(expected["payload"])
    count = len(expected_values)
    if not sync_result.synchronized or sync_result.frame_sync is None:
        return {
            "included": False,
            "symbol_count": count,
            "symbol_errors": count,
            "ser": None,
            "packet_success": False,
            "median_peak_margin_db": None,
        }

    frame_sync = sync_result.frame_sync
    try:
        ordinary = demod_symbol_sequence(
            samples=np.asarray(samples, dtype=np.complex64),
            header_start_sample=int(frame_sync.fine_payload_start_sample),
            sf=audit.SF,
            os_factor=int(os_factor),
            cfo_int=int(frame_sync.cfo_int_est),
            cfo_frac=float(frame_sync.cfo_frac_est),
            sfo_hat=float(frame_sync.sfo_hat),
            sfo_cum_initial=float(frame_sync.sfo_cum_initial),
            header_count=len(expected["header"]),
            payload_count=len(expected["payload"]),
            payload_ldro=audit.LDRO,
            cfo_correction_mode="continuous",
        )
        combined = batch_savaux_spectra(
            samples,
            [int(item.start_sample) + int(os_factor) // 2 for item in ordinary],
            sf=audit.SF,
            os_factor=int(os_factor),
            cfo_int=int(frame_sync.cfo_int_est),
            cfo_frac=float(frame_sync.cfo_frac_est),
        )
    except ValueError:
        return {
            "included": True,
            "symbol_count": count,
            "symbol_errors": count,
            "ser": 1.0,
            "packet_success": False,
            "median_peak_margin_db": None,
        }

    actual: list[int] = []
    margins: list[float] = []
    for index, spectrum in enumerate(combined):
        power = np.abs(spectrum).astype(np.float64) ** 2
        selected = int(np.argmax(power))
        second = float(np.partition(power, -2)[-2])
        margins.append(
            float(
                10.0
                * math.log10(
                    (float(power[selected]) + 1e-30) / (second + 1e-30)
                )
            )
        )
        is_header = index < len(expected["header"])
        actual.append(
            bin_to_grlora_symbol(
                selected,
                sf=audit.SF,
                is_header=is_header,
                ldro=bool(audit.LDRO and not is_header),
            )
        )
    errors = sum(left != right for left, right in zip(actual, expected_values))
    errors += max(0, count - len(actual))
    return {
        "included": True,
        "symbol_count": count,
        "symbol_errors": int(errors),
        "ser": float(errors / count),
        "packet_success": errors == 0,
        "median_peak_margin_db": float(np.median(margins)) if margins else None,
    }


def _sync_config(packet: audit.EvaluationPacket, sample_rate_hz: int) -> Any:
    return audit.SinglePacketSyncConfig(
        sf=audit.SF,
        bw_hz=audit.BW_HZ,
        sample_rate_hz=int(sample_rate_hz),
        center_frequency_hz=packet.center_frequency_hz,
        preamble_symbols=audit.PREAMBLE_SYMBOLS,
        sync_word=audit.SYNC_WORD,
        scan_chirps=24,
    )


def _condition_rows(
    prepared: dict[str, Any],
    noise_seed: int,
    snrs: list[float],
) -> list[dict[str, Any]]:
    packet_index = int(prepared["packet_index"])
    reference_1m = prepared["methods"]["native_1msps"]["samples"]
    rng = np.random.default_rng(int(noise_seed) + packet_index * 1_000_003)
    unit_1m = audit.complex_awgn(len(reference_1m), 1.0, rng)
    rows: list[dict[str, Any]] = []
    for snr_db in snrs:
        methods: dict[str, Any] = {}
        for method in METHODS:
            state = prepared["methods"][method]
            unit_noise = paired_unit_noise(unit_1m, method)
            noise_power = audit.snr_noise_power(state["active_power"], snr_db)
            noisy = np.asarray(
                state["samples"] + unit_noise * math.sqrt(noise_power),
                dtype=np.complex64,
            )
            noise_active = _active_output_slice(
                noisy - state["samples"],
                sample_rate_hz=state["sample_rate_hz"],
                leading_guard_high=prepared["leading_guard_high"],
                trailing_guard_high=prepared["trailing_guard_high"],
            )
            measured_noise_power = float(
                np.mean(np.abs(noise_active).astype(np.float64) ** 2)
            )
            measured_snr_db = float(
                10.0
                * math.log10(state["active_power"] / measured_noise_power)
            )
            methods[method] = {
                "sample_rate_hz": state["sample_rate_hz"],
                "clean_sync": audit._sync_report(state["sync"].result),
                "target_snr_db": float(snr_db),
                "measured_snr_db": measured_snr_db,
                "added_noise_power": noise_power,
                "score": evaluate_savaux(
                    noisy,
                    state["sync"].result,
                    prepared["expected"],
                    os_factor=state["sample_rate_hz"] // audit.BW_HZ,
                ),
            }
        rows.append(
            {
                "packet_index": packet_index,
                "packet_id": prepared["packet_id"],
                "source_snr_db": prepared["source_snr_db"],
                "noise_seed": int(noise_seed),
                "snr_db": float(snr_db),
                "methods": methods,
            }
        )
    return rows


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    scores = [row["methods"][method]["score"] for row in rows]
    included = [score for score in scores if score["included"]]
    count = sum(int(score["symbol_count"]) for score in included)
    errors = sum(int(score["symbol_errors"]) for score in included)
    return {
        "packet_attempts": len(rows),
        "included_packet_attempts": len(included),
        "clean_sync_rate": float(len(included) / len(rows)) if rows else None,
        "symbol_count": count,
        "symbol_errors": errors,
        "conditional_ser": float(errors / count) if count else None,
        "packet_success_rate": (
            float(sum(bool(score["packet_success"]) for score in included) / len(included))
            if included
            else None
        ),
    }


def _paired_summary(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    common = [
        row
        for row in rows
        if row["methods"][left]["score"]["included"]
        and row["methods"][right]["score"]["included"]
    ]
    left_errors = [row["methods"][left]["score"]["symbol_errors"] for row in common]
    right_errors = [row["methods"][right]["score"]["symbol_errors"] for row in common]
    symbol_count = sum(
        int(row["methods"][left]["score"]["symbol_count"]) for row in common
    )
    return {
        "left": left,
        "right": right,
        "common_packet_attempts": len(common),
        "symbol_count_per_method": symbol_count,
        "left_symbol_errors": int(sum(left_errors)),
        "right_symbol_errors": int(sum(right_errors)),
        "left_ser": float(sum(left_errors) / symbol_count) if symbol_count else None,
        "right_ser": float(sum(right_errors) / symbol_count) if symbol_count else None,
        "ser_difference_left_minus_right": (
            float((sum(left_errors) - sum(right_errors)) / symbol_count)
            if symbol_count
            else None
        ),
        "left_better_attempts": sum(a < b for a, b in zip(left_errors, right_errors)),
        "right_better_attempts": sum(a > b for a, b in zip(left_errors, right_errors)),
        "tied_attempts": sum(a == b for a, b in zip(left_errors, right_errors)),
    }


def summarize(rows: list[dict[str, Any]], snrs: list[float]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for snr_db in snrs:
        selected = [row for row in rows if row["snr_db"] == float(snr_db)]
        output.append(
            {
                "snr_db": float(snr_db),
                "methods": {
                    method: _method_summary(selected, method) for method in METHODS
                },
                "paired_rfsr_vs_native_250ksps": _paired_summary(
                    selected, "official_ota_rfsr_1msps", "native_250ksps"
                ),
                "paired_rfsr_vs_native_1msps": _paired_summary(
                    selected, "official_ota_rfsr_1msps", "native_1msps"
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    device = audit._resolve_device(args.device)
    packets = audit.load_official_ota_packets(
        args.official_ota_root, int(args.packets)
    )
    frontend = audit._frontend(
        audit.DEFAULT_OTA_CHECKPOINT,
        device,
        int(args.chunk_input_samples),
    )
    prepared: list[dict[str, Any]] = []
    for packet_index, packet in enumerate(packets):
        print(f"prepare {packet_index + 1}/{len(packets)} {packet.packet_id}", flush=True)
        high = packet.load_samples()
        low = np.asarray(high[::8], dtype=np.complex64)
        method_samples = {
            "native_250ksps": low,
            "native_1msps": np.asarray(high[::2], dtype=np.complex64),
            "official_ota_rfsr_1msps": frontend.enhance(
                low, snr_db=float(packet.source_snr_db or 0.0)
            ),
        }
        methods: dict[str, Any] = {}
        for method, values in method_samples.items():
            sample_rate_hz = METHOD_SAMPLE_RATES[method]
            sync = audit.prepare_samples_and_sync(
                values,
                _sync_config(packet, sample_rate_hz),
                coarse_cfo_centering=True,
            )
            methods[method] = {
                "samples": sync.samples,
                "sync": sync,
                "sample_rate_hz": sample_rate_hz,
                "active_power": _active_power(
                    sync.samples,
                    sample_rate_hz=sample_rate_hz,
                    leading_guard_high=packet.leading_guard_high,
                    trailing_guard_high=packet.trailing_guard_high,
                ),
            }
        prepared.append(
            {
                "packet_index": packet_index,
                "packet_id": packet.packet_id,
                "source_path": str(packet.source_path),
                "source_sha256": audit._sha256(packet.source_path),
                "source_snr_db": packet.source_snr_db,
                "leading_guard_high": packet.leading_guard_high,
                "trailing_guard_high": packet.trailing_guard_high,
                "expected": packet.expected,
                "methods": methods,
            }
        )

    snrs = [float(value) for value in args.snrs]
    tasks = [
        (packet, int(seed), snrs)
        for packet in prepared
        for seed in args.noise_seeds
    ]
    if int(args.workers) == 1:
        nested_rows = [_condition_rows(*task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
            nested_rows = list(executor.map(lambda item: _condition_rows(*item), tasks))
    rows = [row for group in nested_rows for row in group]
    summary = summarize(rows, snrs)
    for item in summary:
        pair250 = item["paired_rfsr_vs_native_250ksps"]
        pair1m = item["paired_rfsr_vs_native_1msps"]
        print(
            json.dumps(
                {
                    "snr_db": item["snr_db"],
                    "rfsr_ser": pair250["left_ser"],
                    "native_250k_ser": pair250["right_ser"],
                    "native_1m_ser": pair1m["right_ser"],
                }
            ),
            flush=True,
        )

    clean_diagnostics = []
    for packet in prepared:
        for method in METHODS:
            state = packet["methods"][method]
            clean_diagnostics.append(
                {
                    "packet_index": packet["packet_index"],
                    "packet_id": packet["packet_id"],
                    "method": method,
                    "sample_rate_hz": state["sample_rate_hz"],
                    "active_power": state["active_power"],
                    "sync": audit._sync_report(state["sync"].result),
                    "coarse_cfo_hz": state["sync"].coarse_cfo_hz,
                }
            )
    payload = {
        "schema": "official-rfsr-focused-post-savaux",
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim": (
            "paired gain-matched post-FrameSync Savaux comparison on official OTA IQ"
        ),
        "noise_injection_stage": "after clean frontend and frozen clean FrameSync",
        "noise_pairing": (
            "one 1 MS/s unit AWGN realization per packet/seed; 250 kS/s uses every "
            "fourth noise sample; each method scales it to its active output power"
        ),
        "configuration": {
            "official_ota_root": str(Path(args.official_ota_root).resolve()),
            "packets": len(packets),
            "snrs_db": snrs,
            "noise_seeds": [int(value) for value in args.noise_seeds],
            "methods": list(METHODS),
            "workers": int(args.workers),
            "device": device,
            "symbols_per_packet": 40,
        },
        "checkpoint": asdict(frontend.provenance),
        "packet_inventory": [
            {
                key: packet[key]
                for key in (
                    "packet_index",
                    "packet_id",
                    "source_path",
                    "source_sha256",
                    "source_snr_db",
                )
            }
            for packet in prepared
        ],
        "clean_diagnostics": clean_diagnostics,
        "summary": summary,
        "rows": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
