"""Shared utilities for standalone baseline evaluators.

The baseline package should not depend on retired phase-line experiments.
This module owns the small amount of dataset loading, AWGN generation, and
SER bookkeeping that several baseline runners need.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[2]
GR_LORA_ROOT = WEAK_ROOT.parent

DEFAULT_DATASETS = ("0_0_0_10_14_8", "0_0_0_10_14_16", "0_0_0_10_14_32")


def _fir_filter_complex(
    samples: np.ndarray,
    taps: np.ndarray,
    block_samples: int = 1 << 18,
) -> np.ndarray:
    """Apply a causal complex FIR with bounded-memory overlap-save FFTs."""

    values = np.asarray(samples, dtype=np.complex64)
    coefficients = np.asarray(taps, dtype=np.complex128)
    if coefficients.ndim != 1 or coefficients.size < 1:
        raise ValueError("taps must be a non-empty vector")
    if coefficients.size == 1:
        return (values * coefficients[0]).astype(np.complex64)
    block = max(1, int(block_samples))
    overlap = int(coefficients.size - 1)
    fft_size = 1 << int(
        math.ceil(math.log2(float(block + 2 * overlap)))
    )
    frequency_response = np.fft.fft(coefficients, fft_size)
    history = np.zeros(overlap, dtype=np.complex64)
    output = np.empty(values.size, dtype=np.complex64)
    for start in range(0, int(values.size), block):
        stop = min(int(values.size), start + block)
        current = values[start:stop]
        extended = np.concatenate((history, current))
        filtered = np.fft.ifft(
            np.fft.fft(extended, fft_size) * frequency_response
        )
        output[start:stop] = filtered[overlap : overlap + current.size].astype(
            np.complex64
        )
        history = extended[-overlap:].copy()
    return output


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    return int(float(value))


def as_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    return float(value)


def as_float_vector(value: str | None) -> tuple[float, ...]:
    text = "" if value is None else str(value).strip()
    if not text:
        return ()
    out: list[float] = []
    for part in text.replace(",", " ").replace(";", " ").split():
        try:
            out.append(float(part))
        except ValueError:
            continue
    return tuple(out)


def maybe_set_branch_vectors(packet: dict[str, Any], row: dict[str, str]) -> None:
    if not packet.get("branch_sfo_hat"):
        values = as_float_vector(row.get("source_grlora_branch_sfo_hat"))
        if values:
            packet["branch_sfo_hat"] = values
    if not packet.get("branch_sfo_cum_initial"):
        values = as_float_vector(row.get("source_grlora_branch_sfo_cum_initial"))
        if values:
            packet["branch_sfo_cum_initial"] = values


def dataset_paths(dataset: str) -> tuple[Path, Path]:
    iq = GR_LORA_ROOT / "data" / "USRP_IQ" / f"{dataset}.bin"
    symbols = WEAK_ROOT / "data" / "weak_sync_chain" / "header_first" / f"{dataset}_header_first_symbols.csv"
    return iq, symbols


def load_packets(symbol_csv: Path) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    with symbol_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("stage", "")) not in {"header", "payload"}:
                continue
            packet_index = as_int(row.get("packet_index"))
            packet = grouped.setdefault(
                packet_index,
                {
                    "packet_index": packet_index,
                    "frame_index": as_int(row.get("frame_index")),
                    "event_index": as_int(row.get("event_index")),
                    "sf": as_int(row.get("sf")),
                    "bw": as_float(row.get("bw")),
                    "os_factor": as_int(row.get("os_factor"), 1),
                    "cfo_int": as_int(row.get("cfo_int")),
                    "cfo_frac": as_float(row.get("cfo_frac")),
                    "header_valid": parse_bool(str(row.get("header_valid", "0"))),
                    "payload_len": as_int(row.get("payload_len")),
                    "cr": as_int(row.get("payload_cr")),
                    "has_crc": parse_bool(str(row.get("payload_has_crc", "0"))),
                    "ldro": parse_bool(str(row.get("payload_ldro", "0"))),
                    "preamble_len": 8.0,
                    "header_symbols": [],
                    "payload_symbols": [],
                    "header_start_sample": None,
                    "branch_sfo_hat": (),
                    "branch_sfo_cum_initial": (),
                },
            )
            maybe_set_branch_vectors(packet, row)
            stage = str(row.get("stage", ""))
            symbol = {
                "stage_symbol_index": as_int(row.get("stage_symbol_index")),
                "frame_symbol_index": as_int(row.get("frame_symbol_index")),
                "start_sample": as_int(row.get("start_sample")),
                "raw_fft_bin": as_int(row.get("raw_fft_bin")),
                "symbol_value": as_int(row.get("symbol_value")),
                "sto_frac": as_float(row.get("sto_frac")),
                "sfo_hat": as_float(row.get("sfo_hat")),
                "sfo_cum_before": as_float(row.get("sfo_cum_before")),
                "sfo_sample_adjust_after": as_int(row.get("sfo_sample_adjust_after")),
            }
            if stage == "header":
                if packet["header_start_sample"] is None:
                    packet["header_start_sample"] = int(symbol["start_sample"])
                packet["header_symbols"].append(symbol)
            elif stage == "payload":
                symbol["payload_symbol_index"] = int(symbol["stage_symbol_index"])
                symbol["gt_bin"] = int(symbol["raw_fft_bin"])
                packet["payload_symbols"].append(symbol)
    packets = [item for item in grouped.values() if item["payload_symbols"]]
    packets.sort(key=lambda item: int(item["packet_index"]))
    for packet in packets:
        if packet["header_start_sample"] is None:
            packet["header_start_sample"] = int(packet["payload_symbols"][0]["start_sample"])
    return packets


def parse_snr_grid(start: float, stop: float, step: float) -> list[float]:
    if step == 0.0:
        raise ValueError("snr step must be non-zero")
    values: list[float] = []
    cur = float(start)
    if step < 0:
        while cur >= float(stop) - 1e-9:
            values.append(round(cur, 6))
            cur += float(step)
    else:
        while cur <= float(stop) + 1e-9:
            values.append(round(cur, 6))
            cur += float(step)
    return values


def snr_values(values: Sequence[float]) -> tuple[float | None, ...]:
    if not values:
        return (None,)
    return tuple(float(v) for v in values)


def noise_samples(
    clean: np.ndarray,
    snr_db: float | None,
    seed: int,
    signal_reference_power: float | None,
    noise_shape: str = "white",
    os_factor: int = 1,
    filter_taps: int = 129,
    color_magnitude: float = 0.85,
    color_phase_rad: float = 0.7,
) -> np.ndarray:
    """Add reproducible white or receiver-bandlimited complex Gaussian noise."""

    if snr_db is None:
        return np.asarray(clean, dtype=np.complex64)
    signal_power = (
        float(signal_reference_power)
        if signal_reference_power is not None
        else float(np.mean(np.abs(clean).astype(np.float64) ** 2))
    )
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    rng = np.random.default_rng(int(seed))
    shape = str(noise_shape)
    if shape not in {"white", "lowpass", "ar1"}:
        raise ValueError(f"unknown noise shape: {noise_shape}")
    noise = (
        rng.normal(0.0, np.sqrt(0.5), clean.size).astype(np.float32)
        + 1j * rng.normal(0.0, np.sqrt(0.5), clean.size).astype(np.float32)
    ).astype(np.complex64)
    if shape in {"lowpass", "ar1"}:
        os_value = int(os_factor)
        if shape == "lowpass" and os_value <= 1:
            raise ValueError("lowpass noise requires os_factor > 1")
        count = max(9, int(filter_taps))
        if count % 2 == 0:
            count += 1
        if shape == "lowpass":
            half = (count - 1) / 2.0
            indexes = np.arange(count, dtype=np.float64) - half
            cutoff_cycles_per_sample = 0.5 / float(os_value)
            taps = (
                2.0
                * cutoff_cycles_per_sample
                * np.sinc(2.0 * cutoff_cycles_per_sample * indexes)
                * np.hamming(count)
            ).astype(np.complex128)
        else:
            magnitude = float(color_magnitude)
            if not 0.0 <= magnitude < 1.0:
                raise ValueError("color_magnitude must be in [0, 1)")
            indexes = np.arange(count, dtype=np.float64)
            pole = magnitude * np.exp(1j * float(color_phase_rad))
            taps = np.power(pole, indexes).astype(np.complex128)
        taps /= np.sqrt(float(np.sum(np.abs(taps) ** 2)))
        noise = _fir_filter_complex(noise, taps)
        measured = float(np.mean(np.abs(noise).astype(np.float64) ** 2))
        if not math.isfinite(measured) or measured <= 0.0:
            raise RuntimeError("lowpass noise shaper produced invalid power")
        noise *= np.float32(1.0 / np.sqrt(measured))
    noise *= np.float32(np.sqrt(noise_power))
    return (np.asarray(clean, dtype=np.complex64) + noise).astype(np.complex64)


def payload_gt_bins(packet: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(item["gt_bin"]) for item in packet["payload_symbols"])


def err_count(selected: Sequence[int], gt_bins: Sequence[int]) -> tuple[int, int]:
    count = min(len(selected), len(gt_bins))
    errors = sum(int(selected[idx]) != int(gt_bins[idx]) for idx in range(count))
    return int(errors), int(count)


def signal_reference_power(
    samples: np.ndarray,
    packets: Sequence[dict[str, Any]],
    mode: str,
    explicit_power: float | None,
) -> tuple[float, int, int]:
    if explicit_power is not None:
        return float(explicit_power), 0, len(packets)
    if mode == "whole":
        return float(np.mean(np.abs(samples).astype(np.float64) ** 2)), int(samples.size), len(packets)

    total_power = 0.0
    total_count = 0
    packet_ids: set[int] = set()
    for packet in packets:
        sf = int(packet["sf"])
        os_factor = int(packet["os_factor"])
        symbol_samples = (1 << sf) * os_factor
        symbols: list[dict[str, Any]] = []
        if mode in {"packet", "header_payload"}:
            symbols.extend(packet.get("header_symbols", []))
        symbols.extend(packet.get("payload_symbols", []))
        for symbol in symbols:
            start = int(symbol["start_sample"])
            stop = start + symbol_samples
            if start < 0 or stop > int(samples.size):
                continue
            chunk = np.asarray(samples[start:stop], dtype=np.complex64)
            total_power += float(np.sum(np.abs(chunk).astype(np.float64) ** 2))
            total_count += int(chunk.size)
            packet_ids.add(int(packet["packet_index"]))
    if total_count <= 0:
        raise ValueError(f"no packet-active samples found for signal reference mode {mode!r}")
    power = total_power / float(total_count)
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError(f"invalid signal reference power {power}")
    return float(power), int(total_count), len(packet_ids)


def sum_rows(rows: Sequence[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0)) for row in rows))


def mean_rows(rows: Sequence[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else 0.0


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
