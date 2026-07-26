#!/usr/bin/env python3
"""Run the Sym-FEC-style baseline on local header-first packet inputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


WEAK_ROOT = Path(__file__).resolve().parents[3]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.chirp import build_downchirp, bin_to_grlora_symbol, signed_fft_bin  # noqa: E402
from weak_decoder.baselines.symfec import (  # noqa: E402
    SymFECConfig,
    decode_symfec_payload_from_spectra,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Sym-FEC-style symbol-level LoRa FEC baseline."
    )
    parser.add_argument("-i", "--input-iq", type=Path, required=True, help="complex64 IQ file")
    parser.add_argument(
        "-s",
        "--symbol-csv",
        type=Path,
        required=True,
        help="header-first symbols CSV with synchronized header/payload rows",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="per-packet CSV output")
    parser.add_argument("--symbols-output", type=Path, default=None, help="optional per-symbol CSV")
    parser.add_argument("--summary-json", type=Path, default=None, help="optional summary JSON")
    parser.add_argument("--packet", type=int, default=None, help="optional packet_index filter")
    parser.add_argument("--max-packets", type=int, default=None, help="optional packet limit")
    parser.add_argument("--no-gt", action="store_true", help="ignore raw_fft_bin columns as GT")
    parser.add_argument("--crc-mode", choices=("grlora", "sx1276"), default="grlora")
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument(
        "--cfo-correction-mode",
        choices=("none", "symbol", "continuous"),
        default="continuous",
        help="CFO pre-compensation before building each symbol spectrum",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=("center", "multi-offset"),
        default="center",
        help="symbol spectrum used by Sym-FEC; multi-offset fuses all OSR phases",
    )
    parser.add_argument("--score-floor-db", type=float, default=30.0)
    parser.add_argument("--bit-metric", choices=("max", "logsumexp"), default="max")
    parser.add_argument("--codeword-candidates", type=int, default=4)
    parser.add_argument("--refine-iterations", type=int, default=2)
    parser.add_argument("--bit-score-weight", type=float, default=1.0)
    parser.add_argument("--exact-symbol-weight", type=float, default=1.0)
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    if value == "":
        return int(default)
    return int(float(value))


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = str(row.get(key, "")).strip()
    if value == "":
        return float(default)
    return float(value)


def _first_int(row: dict[str, str], keys: tuple[str, ...], default: int = 0) -> int:
    for key in keys:
        if str(row.get(key, "")).strip() != "":
            return _int(row, key, default)
    return int(default)


def load_packets(
    symbol_csv: Path,
    packet_filter: int | None,
    max_packets: int | None,
    use_gt: bool,
) -> list[dict[str, Any]]:
    """Read the current header-first CSV structure used by weak_decoder."""

    with symbol_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    packets: dict[int, dict[str, Any]] = {}
    for row in rows:
        packet_index = _int(row, "packet_index", -1)
        if packet_index < 0:
            continue
        if packet_filter is not None and packet_index != int(packet_filter):
            continue

        packet = packets.setdefault(
            packet_index,
            {
                "packet_index": packet_index,
                "frame_index": _int(row, "frame_index", packet_index),
                "event_index": _int(row, "event_index", packet_index),
                "header_symbols": [],
                "payload_symbols": [],
                "header_start_sample": None,
                "sf": _int(row, "sf", 10),
                "bw": _float(row, "bw", 125000.0),
                "os_factor": _int(row, "os_factor", 1),
                "cfo_int": _int(row, "cfo_int", 0),
                "cfo_frac": _float(row, "cfo_frac", 0.0),
                "payload_len": _int(row, "payload_len", 0),
                "cr": _first_int(row, ("payload_cr", "cr"), 1),
                "has_crc": bool(_first_int(row, ("payload_has_crc", "has_crc"), 1)),
                "ldro": bool(_first_int(row, ("payload_ldro", "ldro"), 0)),
                "header_valid": bool(_int(row, "header_valid", 0)),
            },
        )

        stage = str(row.get("stage", "")).strip().lower()
        stage_symbol_index = _int(row, "stage_symbol_index", -1)
        if stage == "header":
            if stage_symbol_index == 0:
                packet["header_start_sample"] = _int(row, "start_sample", 0)
            if 0 <= stage_symbol_index < 8:
                packet["header_symbols"].append((stage_symbol_index, _int(row, "symbol_value", 0)))
        elif stage == "payload":
            packet["payload_symbols"].append(
                {
                    "payload_symbol_index": stage_symbol_index,
                    "start_sample": _int(row, "start_sample", 0),
                    "gt_raw_fft_bin": _int(row, "raw_fft_bin", -1) if use_gt else -1,
                }
            )

    valid_packets: list[dict[str, Any]] = []
    for packet in packets.values():
        if not packet["header_valid"] or packet["header_start_sample"] is None:
            continue
        packet["header_symbols"] = [
            value for _, value in sorted(packet["header_symbols"], key=lambda item: item[0])
        ]
        packet["payload_symbols"].sort(key=lambda item: int(item["payload_symbol_index"]))
        if len(packet["header_symbols"]) == 8 and packet["payload_symbols"]:
            valid_packets.append(packet)
    valid_packets.sort(key=lambda item: int(item["packet_index"]))
    if max_packets is not None:
        valid_packets = valid_packets[: int(max_packets)]
    return valid_packets


def _symbol_sample_indexes(start_sample: int, sf: int, os_factor: int, offset: int) -> np.ndarray:
    n_bins = 1 << int(sf)
    return int(start_sample) + int(offset) + int(os_factor) * np.arange(n_bins, dtype=np.int64)


def _apply_continuous_cfo_phase(
    symbol: np.ndarray,
    start_sample: int,
    sf: int,
    os_factor: int,
    cfo_total: float,
    header_start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    if str(cfo_correction_mode) != "continuous":
        return np.asarray(symbol, dtype=np.complex64)
    n_bins = 1 << int(sf)
    relative_chip_start = float(int(start_sample) - int(header_start_sample)) / float(os_factor)
    cfo_phase = float(2.0 * math.pi * float(cfo_total) * relative_chip_start / n_bins)
    return (np.asarray(symbol, dtype=np.complex64) * np.exp(-1j * cfo_phase)).astype(np.complex64)


def extract_center_spectrum(
    samples: np.ndarray,
    packet: dict[str, Any],
    start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    mode = str(cfo_correction_mode)
    use_cfo = mode in {"symbol", "continuous"}
    cfo_int = int(packet["cfo_int"]) if use_cfo else 0
    cfo_frac = float(packet["cfo_frac"]) if use_cfo else 0.0
    cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
    indexes = _symbol_sample_indexes(
        start_sample=start_sample,
        sf=sf,
        os_factor=os_factor,
        offset=os_factor // 2,
    )
    if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
        raise ValueError(f"symbol at {start_sample} exceeds IQ range")
    symbol = np.asarray(samples[indexes], dtype=np.complex64)
    symbol = _apply_continuous_cfo_phase(
        symbol,
        start_sample=start_sample,
        sf=sf,
        os_factor=os_factor,
        cfo_total=cfo_total,
        header_start_sample=int(packet["header_start_sample"]),
        cfo_correction_mode=mode,
    )
    downchirp = build_downchirp(sf, cfo_int=cfo_int, cfo_frac=cfo_frac)
    return np.fft.fft((symbol * downchirp).astype(np.complex64)).astype(np.complex64)


def extract_multi_offset_spectrum(
    samples: np.ndarray,
    packet: dict[str, Any],
    start_sample: int,
    cfo_correction_mode: str,
) -> np.ndarray:
    sf = int(packet["sf"])
    os_factor = int(packet["os_factor"])
    mode = str(cfo_correction_mode)
    use_cfo = mode in {"symbol", "continuous"}
    cfo_int = int(packet["cfo_int"]) if use_cfo else 0
    cfo_frac = float(packet["cfo_frac"]) if use_cfo else 0.0
    cfo_total = float(packet["cfo_int"]) + float(packet["cfo_frac"])
    downchirp = build_downchirp(sf, cfo_int=cfo_int, cfo_frac=cfo_frac)
    n_bins = 1 << sf
    fused_power = np.zeros(n_bins, dtype=np.float64)
    center_spectrum: np.ndarray | None = None
    for offset in range(os_factor):
        indexes = _symbol_sample_indexes(
            start_sample=start_sample,
            sf=sf,
            os_factor=os_factor,
            offset=offset,
        )
        if int(indexes[0]) < 0 or int(indexes[-1]) >= samples.size:
            raise ValueError(f"symbol at {start_sample} offset {offset} exceeds IQ range")
        symbol = np.asarray(samples[indexes], dtype=np.complex64)
        symbol = _apply_continuous_cfo_phase(
            symbol,
            start_sample=start_sample,
            sf=sf,
            os_factor=os_factor,
            cfo_total=cfo_total,
            header_start_sample=int(packet["header_start_sample"]),
            cfo_correction_mode=mode,
        )
        spectrum = np.fft.fft((symbol * downchirp).astype(np.complex64)).astype(np.complex64)
        power = np.abs(spectrum).astype(np.float64) ** 2
        fused_power += power / (float(np.max(power)) + 1e-30)
        if offset == os_factor // 2:
            center_spectrum = spectrum
    if center_spectrum is None:
        center_spectrum = np.ones(n_bins, dtype=np.complex64)
    phase = np.exp(1j * np.angle(center_spectrum))
    return (np.sqrt(fused_power).astype(np.float64) * phase).astype(np.complex64)


def _argmax_bins(spectra: Sequence[np.ndarray]) -> tuple[int, ...]:
    return tuple(int(np.argmax(np.abs(spec).astype(np.float64) ** 2)) for spec in spectra)


def _ser(raw_bins: Sequence[int], gt_bins: Sequence[int], sf: int, ldro: bool) -> tuple[float, float, int]:
    compared = 0
    raw_errors = 0
    symbol_errors = 0
    for pred, gt in zip(raw_bins, gt_bins):
        gt_i = int(gt)
        if gt_i < 0:
            continue
        pred_i = int(pred)
        raw_errors += int(pred_i != gt_i)
        pred_symbol = bin_to_grlora_symbol(pred_i, sf=int(sf), is_header=False, ldro=bool(ldro))
        gt_symbol = bin_to_grlora_symbol(gt_i, sf=int(sf), is_header=False, ldro=bool(ldro))
        symbol_errors += int(pred_symbol != gt_symbol)
        compared += 1
    if compared <= 0:
        return 0.0, 0.0, 0
    return float(raw_errors / compared), float(symbol_errors / compared), int(compared)


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def build_config(args: argparse.Namespace) -> SymFECConfig:
    return SymFECConfig(
        score_floor_db=float(args.score_floor_db),
        bit_metric=str(args.bit_metric),
        codeword_candidates=int(args.codeword_candidates),
        refine_iterations=int(args.refine_iterations),
        bit_score_weight=float(args.bit_score_weight),
        exact_symbol_weight=float(args.exact_symbol_weight),
    )


def evaluate_packet(
    samples: np.ndarray,
    packet: dict[str, Any],
    args: argparse.Namespace,
    config: SymFECConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    center_spectra: list[np.ndarray] = []
    evidence_spectra: list[np.ndarray] = []
    gt_bins: list[int] = []
    starts: list[int] = []

    for item in packet["payload_symbols"]:
        start_sample = int(item["start_sample"])
        try:
            center = extract_center_spectrum(
                samples=samples,
                packet=packet,
                start_sample=start_sample,
                cfo_correction_mode=str(args.cfo_correction_mode),
            )
            if str(args.evidence_mode) == "multi-offset":
                evidence = extract_multi_offset_spectrum(
                    samples=samples,
                    packet=packet,
                    start_sample=start_sample,
                    cfo_correction_mode=str(args.cfo_correction_mode),
                )
            else:
                evidence = center
        except ValueError:
            continue
        center_spectra.append(center)
        evidence_spectra.append(evidence)
        gt_bins.append(int(item["gt_raw_fft_bin"]))
        starts.append(start_sample)

    sf = int(packet["sf"])
    ldro = bool(packet["ldro"])
    center_bins = _argmax_bins(center_spectra)
    result = decode_symfec_payload_from_spectra(
        spectra_or_powers=evidence_spectra,
        sf=sf,
        cr=int(packet["cr"]),
        ldro=ldro,
        config=config,
        header_symbol_values=tuple(packet["header_symbols"]),
        bw=float(packet["bw"]),
        ldro_mode=int(args.ldro_mode),
        payload_len=int(packet["payload_len"]),
        has_crc=bool(packet["has_crc"]),
        crc_mode=str(args.crc_mode),
    )
    symfec_bins = tuple(result.selected_raw_fft_bins)
    center_raw_ser, center_symbol_ser, compared = _ser(center_bins, gt_bins, sf=sf, ldro=ldro)
    symfec_raw_ser, symfec_symbol_ser, _ = _ser(symfec_bins, gt_bins, sf=sf, ldro=ldro)

    payload_hex = ""
    crc_computed = ""
    crc_received = ""
    decoded_payload_len = 0
    if result.payload_decode is not None:
        payload_hex = result.payload_decode.payload_bytes.hex()
        crc_computed = f"0x{result.payload_decode.crc_computed:04x}"
        crc_received = f"0x{result.payload_decode.crc_received:04x}"
        decoded_payload_len = len(result.payload_decode.payload_bytes)

    changed_count = sum(
        int(a != b)
        for a, b in zip(result.selected_symbol_values, result.argmax_symbol_values)
    )
    block_count = len(result.blocks)
    packet_row = {
        "packet_index": int(packet["packet_index"]),
        "frame_index": int(packet["frame_index"]),
        "event_index": int(packet["event_index"]),
        "sf": sf,
        "bw": float(packet["bw"]),
        "os_factor": int(packet["os_factor"]),
        "payload_len": int(packet["payload_len"]),
        "decoded_payload_len": int(decoded_payload_len),
        "cr": int(packet["cr"]),
        "has_crc": int(bool(packet["has_crc"])),
        "ldro": int(ldro),
        "symbol_count": int(len(result.selected_raw_fft_bins)),
        "block_count": int(block_count),
        "gt_compared_symbols": int(compared),
        "evidence_mode": str(args.evidence_mode),
        "cfo_correction_mode": str(args.cfo_correction_mode),
        "center_raw_ser": float(center_raw_ser),
        "center_symbol_ser": float(center_symbol_ser),
        "symfec_raw_ser": float(symfec_raw_ser),
        "symfec_symbol_ser": float(symfec_symbol_ser),
        "symfec_symbol_ser_gain_vs_center": float(center_symbol_ser - symfec_symbol_ser),
        "symfec_changed_symbols": int(changed_count),
        "symfec_changed_ratio": float(changed_count / max(1, len(result.selected_symbol_values))),
        "center_crc_valid": "",
        "symfec_crc_valid": int(result.crc_valid),
        "symfec_payload_hex": payload_hex,
        "symfec_crc_computed": crc_computed,
        "symfec_crc_received": crc_received,
        "decode_error": result.decode_error,
    }

    symbol_rows: list[dict[str, Any]] = []
    for idx, raw_bin in enumerate(result.selected_raw_fft_bins):
        gt_bin = gt_bins[idx] if idx < len(gt_bins) else -1
        selected_symbol = int(result.selected_symbol_values[idx])
        argmax_bin = int(result.argmax_raw_fft_bins[idx])
        gt_symbol = (
            bin_to_grlora_symbol(gt_bin, sf=sf, is_header=False, ldro=ldro)
            if int(gt_bin) >= 0
            else -1
        )
        symbol_rows.append(
            {
                "packet_index": int(packet["packet_index"]),
                "frame_index": int(packet["frame_index"]),
                "event_index": int(packet["event_index"]),
                "payload_symbol_index": idx,
                "start_sample": starts[idx] if idx < len(starts) else "",
                "sf": sf,
                "os_factor": int(packet["os_factor"]),
                "gt_raw_fft_bin": gt_bin,
                "gt_signed_fft_bin": signed_fft_bin(gt_bin, 1 << sf) if int(gt_bin) >= 0 else "",
                "gt_symbol_value": gt_symbol,
                "center_raw_fft_bin": center_bins[idx] if idx < len(center_bins) else "",
                "center_symbol_value": (
                    bin_to_grlora_symbol(center_bins[idx], sf=sf, is_header=False, ldro=ldro)
                    if idx < len(center_bins)
                    else ""
                ),
                "symfec_raw_fft_bin": int(raw_bin),
                "symfec_signed_fft_bin": signed_fft_bin(raw_bin, 1 << sf),
                "symfec_symbol_value": selected_symbol,
                "symfec_argmax_raw_fft_bin": argmax_bin,
                "symfec_argmax_symbol_value": int(result.argmax_symbol_values[idx]),
                "symfec_changed_from_argmax": int(selected_symbol != int(result.argmax_symbol_values[idx])),
                "center_raw_hit": int(idx < len(center_bins) and int(center_bins[idx]) == int(gt_bin))
                if int(gt_bin) >= 0
                else "",
                "symfec_raw_hit": int(int(raw_bin) == int(gt_bin)) if int(gt_bin) >= 0 else "",
                "symfec_symbol_hit": int(selected_symbol == int(gt_symbol)) if int(gt_symbol) >= 0 else "",
            }
        )
    return packet_row, symbol_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_summary(rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_iq": str(args.input_iq),
        "symbol_csv": str(args.symbol_csv),
        "packet_count": int(len(rows)),
        "evidence_mode": str(args.evidence_mode),
        "cfo_correction_mode": str(args.cfo_correction_mode),
        "center_symbol_ser": _mean(rows, "center_symbol_ser"),
        "symfec_symbol_ser": _mean(rows, "symfec_symbol_ser"),
        "symfec_symbol_ser_gain_vs_center": _mean(rows, "symfec_symbol_ser_gain_vs_center"),
        "symfec_crc_valid_rate": _mean(rows, "symfec_crc_valid"),
        "symfec_changed_ratio": _mean(rows, "symfec_changed_ratio"),
        "parameters": {
            "score_floor_db": float(args.score_floor_db),
            "bit_metric": str(args.bit_metric),
            "codeword_candidates": int(args.codeword_candidates),
            "refine_iterations": int(args.refine_iterations),
            "bit_score_weight": float(args.bit_score_weight),
            "exact_symbol_weight": float(args.exact_symbol_weight),
            "uses_crc_guided_selection": False,
            "uses_payload_template": False,
            "uses_cross_packet_prior": False,
        },
        "algorithm_source": (
            "Sym-FEC public description: symbol-level FEC decoder with signal "
            "copy retrieval across LoRa coding blocks"
        ),
    }


def main() -> int:
    args = parse_args()
    samples = np.fromfile(args.input_iq, dtype=np.complex64)
    if samples.size == 0:
        raise ValueError(f"empty IQ file: {args.input_iq}")

    packets = load_packets(
        args.symbol_csv,
        packet_filter=args.packet,
        max_packets=args.max_packets,
        use_gt=not bool(args.no_gt),
    )
    if not packets:
        raise SystemExit("no header-valid packets found in symbol CSV")

    config = build_config(args)
    packet_rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    for packet in packets:
        print(f"packet {packet['packet_index']}: Sym-FEC-style baseline", flush=True)
        packet_row, packet_symbol_rows = evaluate_packet(samples, packet, args, config)
        packet_rows.append(packet_row)
        symbol_rows.extend(packet_symbol_rows)
        print(
            "  center_ser={:.3f} symfec_ser={:.3f} crc={} changed={}/{}".format(
                float(packet_row["center_symbol_ser"]),
                float(packet_row["symfec_symbol_ser"]),
                int(packet_row["symfec_crc_valid"]),
                int(packet_row["symfec_changed_symbols"]),
                int(packet_row["symbol_count"]),
            ),
            flush=True,
        )

    write_csv(args.output, packet_rows)
    symbols_output = args.symbols_output
    if symbols_output is not None:
        write_csv(symbols_output, symbol_rows)

    summary = build_summary(packet_rows, args)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote={args.output}")
    if symbols_output is not None:
        print(f"wrote_symbols={symbols_output}")
    if args.summary_json is not None:
        print(f"wrote_summary={args.summary_json}")
    print(
        "summary: packets={packet_count} center_ser={center_symbol_ser:.3f} "
        "symfec_ser={symfec_symbol_ser:.3f} crc={symfec_crc_valid_rate:.3f}".format(**summary)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
