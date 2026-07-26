#!/usr/bin/env python3
"""Verify local payload codec alignment against header-first symbol CSV.

The dynamic residual decoder needs a trusted local PHY codec.  This script
checks that clean, CRC-valid hard symbols can be decoded into bytes and
re-encoded back to the deterministic payload-symbol prefix.  The final partial
interleaver block may contain padding codewords that are not constrained by the
decoded payload/CRC bytes, so it is reported separately.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any


WEAK_ROOT = Path(__file__).resolve().parent.parent
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.decoding.payload_codec import (
    decode_explicit_frame_symbols,
    encode_explicit_frame_symbols,
    reencoded_payload_known_prefix_symbols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify payload_codec.py against known-good symbol CSV."
    )
    parser.add_argument(
        "-g",
        "--gt-symbol-csv",
        type=Path,
        default=(
            WEAK_ROOT
            / "data"
            / "weak_sync_chain"
            / "header_first"
            / "0_0_0_10_14_16_header_first_symbols.csv"
        ),
    )
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bw", type=float, default=125000.0)
    parser.add_argument("--ldro-mode", type=int, default=2)
    parser.add_argument("--max-packets", type=int, default=None)
    parser.add_argument("-o", "--output-csv", type=Path, default=None)
    return parser.parse_args()


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = str(row.get(key, "")).strip()
    return int(float(value)) if value else int(default)


def read_symbols(path: Path) -> dict[tuple[int, int], dict[str, list[int]]]:
    result: dict[tuple[int, int], dict[str, list[int]]] = {}
    with path.resolve().open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if _int(row, "header_valid", 0) != 1:
                continue
            stage = str(row.get("stage", "")).strip()
            if stage not in {"header", "payload"}:
                continue
            key = (_int(row, "packet_index", -1), _int(row, "event_index", -1))
            value = _int(row, "symbol_value", -1)
            if key[0] < 0 or key[1] < 0 or value < 0:
                continue
            result.setdefault(key, {"header": [], "payload": []})[stage].append(value)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.resolve().parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.resolve().open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    by_packet = read_symbols(args.gt_symbol_csv)
    keys = sorted(by_packet)
    if args.max_packets is not None:
        keys = keys[: int(args.max_packets)]

    rows: list[dict[str, Any]] = []
    for packet_index, event_index in keys:
        symbols = by_packet[(packet_index, event_index)]
        if len(symbols["header"]) != 8 or not symbols["payload"]:
            continue
        decoded = decode_explicit_frame_symbols(
            symbols["header"],
            symbols["payload"],
            sf=int(args.sf),
            bw=float(args.bw),
            ldro_mode=int(args.ldro_mode),
        )
        enc_header, enc_payload = encode_explicit_frame_symbols(
            decoded.payload.payload_bytes,
            sf=int(args.sf),
            cr=int(decoded.header.cr),
            has_crc=bool(decoded.header.has_crc),
            ldro=bool(decoded.header.ldro),
        )
        known_prefix = reencoded_payload_known_prefix_symbols(
            payload_len=int(decoded.header.payload_len),
            has_crc=bool(decoded.header.has_crc),
            sf=int(args.sf),
            cr=int(decoded.header.cr),
            ldro=bool(decoded.header.ldro),
        )
        header_mismatches = sum(
            1 for got, exp in zip(enc_header, symbols["header"]) if got != exp
        )
        prefix_mismatches = sum(
            1
            for got, exp in zip(enc_payload[:known_prefix], symbols["payload"][:known_prefix])
            if got != exp
        )
        suffix_len = max(0, min(len(enc_payload), len(symbols["payload"])) - known_prefix)
        suffix_mismatches = sum(
            1
            for got, exp in zip(enc_payload[known_prefix:], symbols["payload"][known_prefix:])
            if got != exp
        )
        rows.append({
            "packet_index": packet_index,
            "event_index": event_index,
            "payload_len": decoded.header.payload_len,
            "payload_symbols": len(symbols["payload"]),
            "known_prefix_symbols": known_prefix,
            "crc_valid": int(decoded.payload.crc_valid),
            "header_mismatches": header_mismatches,
            "known_prefix_mismatches": prefix_mismatches,
            "padding_suffix_symbols": suffix_len,
            "padding_suffix_mismatches": suffix_mismatches,
            "payload_hex": decoded.payload.payload_bytes.hex(),
            "crc_hex": decoded.payload.crc_bytes.hex(),
        })

    if args.output_csv:
        write_csv(args.output_csv, rows)

    total = len(rows)
    crc_ok = sum(1 for row in rows if int(row["crc_valid"]) == 1)
    header_ok = sum(1 for row in rows if int(row["header_mismatches"]) == 0)
    prefix_checked = sum(int(row["known_prefix_symbols"]) for row in rows)
    prefix_bad = sum(int(row["known_prefix_mismatches"]) for row in rows)
    suffix_checked = sum(int(row["padding_suffix_symbols"]) for row in rows)
    suffix_bad = sum(int(row["padding_suffix_mismatches"]) for row in rows)

    print(f"Packets checked: {total}")
    print(f"CRC valid: {crc_ok}/{total}")
    print(f"Header re-encode exact: {header_ok}/{total}")
    print(
        f"Deterministic payload prefix mismatches: "
        f"{prefix_bad}/{prefix_checked}"
    )
    print(
        f"Padding suffix mismatches: {suffix_bad}/{suffix_checked} "
        f"(reported, not byte-constrained)"
    )
    if args.output_csv:
        print(f"Output: {args.output_csv.resolve()}")
    return 0 if total and crc_ok == total and prefix_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
