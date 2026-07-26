"""Capture naming, CLI validation, and expected-payload helpers.

中文说明：这一层只处理“实验文件名/命令行参数/期望 payload”这些
和信号处理无关的配置问题。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from .utils import db_to_label, parse_payload_hex


def parse_capture_metadata_value(value: str) -> int | str:
    """Extract the first integer from a filename field, preserving text otherwise."""
    match = re.search(r"-?\d+", str(value))
    if match is None:
        return str(value)
    return int(match.group(0))


def parse_capture_metadata(path: Path) -> dict[str, Any]:
    """Parse experiment metadata from the first six underscore-separated fields."""
    # 文件名约定：实验号_走廊号_位置号_SF_发射功率_Preamble长度.bin
    # 后面即使追加 _noise_rel_...，前六段仍然能还原原始参数。
    parts = [part for part in re.split(r"[_-]", Path(path).stem) if part != ""]
    keys = ["experiment_id", "corridor_id", "position_id", "sf", "tx_power_dbm", "preamble_len"]
    metadata: dict[str, Any] = {
        "filename": Path(path).name,
        "filename_parts": parts,
        "parsed": False,
        "experiment_id": "",
        "corridor_id": "",
        "position_id": "",
        "tx_power_dbm": "",
        "filename_sf": "",
        "filename_tx_power_dbm": "",
        "filename_preamble_len": "",
    }
    if len(parts) < 6:
        return metadata

    for key, value in zip(keys, parts[:6]):
        parsed = parse_capture_metadata_value(value)
        if key == "sf":
            metadata["filename_sf"] = parsed
        elif key == "tx_power_dbm":
            metadata["tx_power_dbm"] = parsed
            metadata["filename_tx_power_dbm"] = parsed
        elif key == "preamble_len":
            metadata["filename_preamble_len"] = parsed
        else:
            metadata[key] = parsed
    metadata["parsed"] = isinstance(metadata["filename_sf"], int) and isinstance(
        metadata["filename_preamble_len"], int
    )
    return metadata


@dataclass
class CaptureParameterResolver:
    """Resolve SF and preamble length from CLI overrides or the capture filename.

    命令行参数优先；没有显式传入时再从文件名里推断。
    """

    args: argparse.Namespace
    input_path: Path

    def resolve(self) -> dict[str, Any]:
        metadata = parse_capture_metadata(self.input_path)

        if self.args.sf is None:
            filename_sf = metadata.get("filename_sf", "")
            if not isinstance(filename_sf, int):
                raise ValueError(
                    f"Cannot infer SF from filename {self.input_path.name}; pass --sf explicitly."
                )
            self.args.sf = int(filename_sf)
            sf_source = "filename"
        else:
            self.args.sf = int(self.args.sf)
            sf_source = "cli"

        if self.args.preamble_len is None:
            filename_preamble_len = metadata.get("filename_preamble_len", "")
            if not isinstance(filename_preamble_len, int):
                raise ValueError(
                    f"Cannot infer preamble length from filename {self.input_path.name}; "
                    "pass --preamble-len explicitly."
                )
            self.args.preamble_len = int(filename_preamble_len)
            preamble_source = "filename"
        else:
            self.args.preamble_len = int(self.args.preamble_len)
            preamble_source = "cli"

        metadata.update(
            {
                "resolved_sf": int(self.args.sf),
                "resolved_sf_source": sf_source,
                "resolved_preamble_len": int(self.args.preamble_len),
                "resolved_preamble_len_source": preamble_source,
            }
        )
        self.args.capture_metadata = metadata
        return metadata


def resolve_capture_parameters(args: argparse.Namespace, input_path: Path) -> dict[str, Any]:
    return CaptureParameterResolver(args=args, input_path=input_path).resolve()


def build_noise_power_db_values(args: argparse.Namespace) -> list[float]:
    """Build the list of added-noise powers in dB relative to the reference power."""
    if args.noise_power_db is not None:
        # 显式列表用于精确复现实验点，比如 -17.5、-18.0 这种细扫。
        values = [float(value) for value in args.noise_power_db]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("--noise-power-db values must be finite numbers.")
        return values

    start = float(args.noise_start_db)
    stop = float(args.noise_stop_db)
    step = float(args.noise_step_db)
    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("--noise-start-db/--noise-stop-db/--noise-step-db must be finite numbers.")
    if step == 0.0:
        raise ValueError("--noise-step-db must not be 0.")
    if (stop - start) * step < 0.0:
        raise ValueError("--noise-step-db sign must move from --noise-start-db toward --noise-stop-db.")

    values = []
    current = start
    epsilon = abs(step) * 1e-9
    # 用 epsilon 避免浮点加法导致最后一个 stop 点被意外跳过。
    if step > 0.0:
        while current <= stop + epsilon:
            values.append(round(current, 10))
            current += step
    else:
        while current >= stop - epsilon:
            values.append(round(current, 10))
            current += step
    if not values:
        raise ValueError("No noise power steps were generated.")
    return values


def validate_capture_args(args: argparse.Namespace) -> None:
    """Validate parameters that directly affect GNU Radio and file output."""
    if not 5 <= int(args.sf) <= 12:
        raise ValueError(f"LoRa SF must be in [5, 12], got {args.sf}.")
    if int(args.preamble_len) <= 0:
        raise ValueError(f"--preamble-len must be positive, got {args.preamble_len}.")
    if float(args.bw) <= 0.0 or float(args.samp_rate) <= 0.0:
        raise ValueError("--bw and --samp-rate must be positive.")
    os_factor = float(args.samp_rate) / float(args.bw)
    rounded_os_factor = round(os_factor)
    if rounded_os_factor <= 0 or not math.isclose(os_factor, rounded_os_factor, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            "--samp-rate must be an integer multiple of --bw for gr-lora_sdr "
            f"(got samp_rate/bw={os_factor:.9g})."
        )
    if args.block_samples <= 0 or args.chunk_samples <= 0:
        raise ValueError("--block-samples and --chunk-samples must be positive.")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise ValueError("--sample-limit must be positive when provided.")


def normalize_payload_hexes(payload_hexes: list[str] | tuple[str, ...]) -> list[str]:
    """Return canonical lower-case payload hex strings."""
    return [parse_payload_hex(value).hex() for value in payload_hexes]


def decoded_payload_hexes_from_packets(decoded_packets: list[dict[str, Any]]) -> list[str]:
    """Extract canonical payload hex strings from decoded packet metadata."""
    payload_hexes = []
    for packet in decoded_packets:
        if not (packet.get("decoded_payload_available") or packet.get("decoded_payload_hex")):
            continue
        payload_hexes.append(parse_payload_hex(str(packet.get("decoded_payload_hex", ""))).hex())
    return payload_hexes


def expected_payloads_from_args(args: argparse.Namespace) -> list[bytes]:
    """Return payload groundtruth used to judge whether decoded packets are correct."""
    if getattr(args, "no_expected_payload_check", False):
        return []
    payload_hexes = getattr(args, "expected_payload_hex", None)
    if payload_hexes is None:
        return []
    return [parse_payload_hex(value) for value in payload_hexes]


def payload_error_counts(decoded_payload: bytes, expected_payload: bytes) -> dict[str, int]:
    """Return byte/bit error counts between two payload byte strings."""
    common_len = min(len(decoded_payload), len(expected_payload))
    byte_errors = sum(
        1
        for decoded_byte, expected_byte in zip(
            decoded_payload[:common_len], expected_payload[:common_len]
        )
        if decoded_byte != expected_byte
    )
    bit_errors = sum(
        (int(decoded_byte) ^ int(expected_byte)).bit_count()
        for decoded_byte, expected_byte in zip(
            decoded_payload[:common_len], expected_payload[:common_len]
        )
    )
    length_delta = abs(len(decoded_payload) - len(expected_payload))
    return {
        "byte_errors": int(byte_errors + length_delta),
        "bit_errors": int(bit_errors + 8 * length_delta),
        "compared_bytes": int(max(len(decoded_payload), len(expected_payload))),
        "compared_bits": int(8 * max(len(decoded_payload), len(expected_payload))),
        "length_delta_bytes": int(length_delta),
    }


def compare_payloads(
    decoded_packets: list[dict[str, Any]],
    expected_payloads: list[bytes],
) -> dict[str, Any]:
    """Compare decoded packet payloads against the expected payload list."""
    decoded = []
    for packet in decoded_packets:
        payload_hex = packet.get("decoded_payload_hex", "")
        if not payload_hex:
            continue
        try:
            payload = parse_payload_hex(str(payload_hex))
        except ValueError:
            payload = b""
        decoded.append((packet, payload))

    unmatched_expected = list(range(len(expected_payloads)))
    matches = []
    exact_matched_decoded_indexes: set[int] = set()
    # 这里按“每个期望 payload 最多匹配一次”处理，避免重复解出同一个包时虚高正确率。
    for decoded_index, (_, payload) in enumerate(decoded):
        match_pos = None
        for expected_index in list(unmatched_expected):
            if payload == expected_payloads[expected_index]:
                match_pos = expected_index
                break
        if match_pos is None:
            continue
        unmatched_expected.remove(match_pos)
        exact_matched_decoded_indexes.add(decoded_index)
        matches.append(
            {
                "decoded_index": int(decoded_index),
                "expected_index": int(match_pos),
                "payload_hex": payload.hex(),
            }
        )

    byte_error_count = 0
    bit_error_count = 0
    compared_byte_count = 0
    compared_bit_count = 0
    compared_payload_packets = 0
    wrong_decoded_indexes = []
    wrong_pairs = []
    nearest_unmatched_expected = list(unmatched_expected)

    for match in matches:
        _, payload = decoded[int(match["decoded_index"])]
        expected_payload = expected_payloads[int(match["expected_index"])]
        counts = payload_error_counts(payload, expected_payload)
        byte_error_count += counts["byte_errors"]
        bit_error_count += counts["bit_errors"]
        compared_byte_count += counts["compared_bytes"]
        compared_bit_count += counts["compared_bits"]
        compared_payload_packets += 1

    for decoded_index, (_, payload) in enumerate(decoded):
        if decoded_index in exact_matched_decoded_indexes:
            continue
        wrong_decoded_indexes.append(decoded_index)
        candidate_indexes = nearest_unmatched_expected or list(range(len(expected_payloads)))
        if not candidate_indexes:
            continue
        nearest_expected = min(
            candidate_indexes,
            key=lambda expected_index: (
                payload_error_counts(payload, expected_payloads[expected_index])["bit_errors"],
                expected_index,
            ),
        )
        if nearest_expected in nearest_unmatched_expected:
            nearest_unmatched_expected.remove(nearest_expected)
        counts = payload_error_counts(payload, expected_payloads[nearest_expected])
        byte_error_count += counts["byte_errors"]
        bit_error_count += counts["bit_errors"]
        compared_byte_count += counts["compared_bytes"]
        compared_bit_count += counts["compared_bits"]
        compared_payload_packets += 1
        wrong_pairs.append(
            {
                "decoded_index": int(decoded_index),
                "nearest_expected_index": int(nearest_expected),
                "byte_errors": int(counts["byte_errors"]),
                "bit_errors": int(counts["bit_errors"]),
                "compared_bytes": int(counts["compared_bytes"]),
                "compared_bits": int(counts["compared_bits"]),
                "length_delta_bytes": int(counts["length_delta_bytes"]),
                "payload_hex": payload.hex(),
            }
        )

    crc_valid_count = sum(1 for packet in decoded_packets if bool(packet.get("crc_valid", False)))
    correct_count = len(matches)
    wrong_count = len(wrong_decoded_indexes)
    missed_correct = max(0, len(expected_payloads) - correct_count)
    ber = bit_error_count / compared_bit_count if compared_bit_count > 0 else float("nan")

    return {
        "expected_packet_count": int(len(expected_payloads)),
        "decoded_payload_count": int(len(decoded)),
        "crc_valid_packets": int(crc_valid_count),
        "crc_invalid_packets": int(max(0, len(decoded_packets) - crc_valid_count)),
        "correct_payload_packets": int(correct_count),
        "wrong_payload_packets": int(wrong_count),
        "missed_correct_payload_packets": int(missed_correct),
        "compared_payload_packets": int(compared_payload_packets),
        "compared_byte_count": int(compared_byte_count),
        "compared_bit_count": int(compared_bit_count),
        "byte_error_count": int(byte_error_count),
        "bit_error_count": int(bit_error_count),
        "ber": float(ber),
        "all_expected_payloads_correct": bool(
            len(expected_payloads) > 0 and correct_count == len(expected_payloads) and wrong_count == 0
        ),
        "matches": matches,
        "wrong_pairs": wrong_pairs,
        "unmatched_expected_indexes": [int(index) for index in unmatched_expected],
        "wrong_decoded_indexes": [int(index) for index in wrong_decoded_indexes],
    }


def planned_output_paths(
    input_path: Path,
    output_dir: Path,
    noise_power_db_values: list[float],
) -> list[tuple[float, Path, Path]]:
    """Compute the bin/json path pair for each noise step."""
    outputs = []
    for noise_power_db in noise_power_db_values:
        label = db_to_label(float(noise_power_db))
        outputs.append(
            (
                float(noise_power_db),
                output_dir / f"{input_path.stem}_noise_rel_{label}dB.bin",
                output_dir / f"{input_path.stem}_noise_rel_{label}dB.json",
            )
        )
    return outputs


def check_output_collisions(
    outputs: list[tuple[float, Path, Path]],
    overwrite: bool,
    input_path: Path,
) -> None:
    """Fail early on output collisions before expensive GNU Radio measurement starts."""
    for _, bin_path, meta_path in outputs:
        if bin_path.resolve() == input_path.resolve():
            raise ValueError(f"Refusing to overwrite the input file: {bin_path}")
        if not overwrite and (bin_path.exists() or meta_path.exists()):
            raise FileExistsError(
                f"{bin_path} or {meta_path} already exists; pass --overwrite to replace existing outputs."
            )
