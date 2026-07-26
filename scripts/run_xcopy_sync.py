#!/usr/bin/env python3
"""Run XCopy-style synchronization on a repeated-packet complex64 capture."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from weak_decoder.synchronization.xcopy_sync import (
    XCopyConfig,
    XCopySyncResult,
    run_xcopy_paper_sync,
    run_xcopy_sync,
    xcopy_raw_symbol_rows,
)


BRANCH4_MEASURED_PERIOD_SAMPLES = 1_500_365


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "对重复固定帧采集执行XCopy逐包检测、整包共轭精对齐、相干合并和原始payload导出。"
        )
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="raw complex64 IQ文件。")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="诊断和合并IQ输出目录。")
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--bw", type=float, default=125e3)
    parser.add_argument("--samp-rate", type=float, default=500e3)
    parser.add_argument("--center-freq", type=float, default=487.7e6)
    parser.add_argument("--preamble-symbols", type=int, default=32)
    parser.add_argument("--sync-word", type=lambda value: int(value, 0), default=0x34)
    parser.add_argument("--payload-symbols", type=int, default=57)
    parser.add_argument(
        "--detection-mode",
        choices=("paper", "periodic"),
        default="paper",
        help="paper scans each retransmission independently; periodic uses the Branch4 timer prior.",
    )
    parser.add_argument(
        "--period-samples",
        type=int,
        default=BRANCH4_MEASURED_PERIOD_SAMPLES,
        help="重传间隔样点数；Branch4高SNR实测默认1500365。",
    )
    parser.add_argument(
        "--detection-chirps",
        type=int,
        default=4,
        help="XCopy uses a four-chirp detection window by default.",
    )
    parser.add_argument("--phase-hop-samples", type=int, default=None)
    parser.add_argument("--min-detection-score", type=float, default=6.0)
    parser.add_argument("--detection-mad-scale", type=float, default=8.0)
    parser.add_argument(
        "--detection-peak-fraction",
        type=float,
        default=None,
        help="First-plateau fraction; defaults to 0.3 in paper mode and 0.5 in periodic mode.",
    )
    parser.add_argument("--min-detection-run", type=int, default=2)
    parser.add_argument("--alignment-search-samples", type=int, default=32)
    parser.add_argument("--alignment-decimation", type=int, default=8)
    parser.add_argument("--max-relative-cfo-hz", type=float, default=100.0)
    parser.add_argument("--min-alignment-score", type=float, default=25.0)
    parser.add_argument("--min-aligned-copies", type=int, default=4)
    parser.add_argument("--max-copies", type=int, default=None)
    parser.add_argument("--soft-frame-top-k", type=int, default=5)
    parser.add_argument("--soft-frame-search-span-chirps", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _prepare_output(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    expected = (
        path / "periodic_scan.csv",
        path / "copies.csv",
        path / "summary.json",
        path / "combined_iq.bin",
        path / "combined_sync.csv",
        path / "soft_frame_candidates.csv",
        path / "aligned_raw_symbols.csv",
        path / "aligned_raw_sync.csv",
        path / "packet_detections.csv",
    )
    existing = [item for item in expected if item.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(item) for item in existing)
        raise FileExistsError(f"refusing to overwrite existing outputs: {joined}")
    if overwrite:
        for item in existing:
            item.unlink()


def _write_scan(path: Path, result: XCopySyncResult) -> None:
    fields = [
        "phase_index",
        "phase_sample",
        "copy_count",
        "peak_bin",
        "signed_peak_bin",
        "peak_power",
        "median_power",
        "peak_to_median",
        "selected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in result.detection.bins:
            writer.writerow(asdict(item))


def _write_packet_detections(path: Path, result: XCopySyncResult) -> None:
    if not result.packet_detections:
        return
    rows = [asdict(item) for item in result.packet_detections]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_copies(path: Path, result: XCopySyncResult) -> None:
    fields = [
        "copy_index",
        "transmission_index",
        "nominal_start_sample",
        "relative_delay_samples",
        "relative_cfo_hz",
        "relative_phase_rad",
        "peak_to_median",
        "tone_bin",
        "included",
        "is_reference",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in result.alignments:
            row = asdict(item)
            row["included"] = int(item.included)
            row["is_reference"] = int(item.is_reference)
            writer.writerow(row)


def _write_combined_sync(path: Path, result: XCopySyncResult) -> None:
    """Write one frame row compatible with run_header_first_demod.py."""

    frame = result.frame_location
    frame_sync = result.frame_sync
    if frame is None or frame_sync is None:
        return

    row: dict[str, object] = {
        "packet_index": 0,
        "event_index": int(frame.event_index),
        "frame_valid": int(frame.valid),
        "located_preamble_start_sample": int(frame.preamble_start_sample),
        "located_sfd_start_sample": int(frame.sfd_start_sample),
        "located_payload_start_sample": int(frame.payload_start_sample),
        "header_start_sample": int(frame_sync.fine_payload_start_sample),
    }
    frame_sync_values = asdict(frame_sync)
    for key, value in frame_sync_values.items():
        if key in {"branch_sync_estimates", "peaks"}:
            continue
        row[f"grlora_{key}"] = int(value) if isinstance(value, bool) else value
    row["grlora_framesync_valid"] = int(frame_sync.valid)

    branches = frame_sync.branch_sync_estimates
    if branches:
        for key in asdict(branches[0]):
            values = [getattr(branch, key) for branch in branches]
            row[f"grlora_branch_{key}"] = "|".join(
                str(int(value) if isinstance(value, bool) else value) for value in values
            )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_soft_frame_candidates(path: Path, result: XCopySyncResult) -> None:
    if not result.soft_frame_candidates:
        return
    rows = [asdict(item) for item in result.soft_frame_candidates]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_raw_symbols(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_raw_sync(path: Path, result: XCopySyncResult, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    by_copy: dict[int, dict[str, object]] = {}
    for row in rows:
        copy_index = int(row["copy_index"])
        by_copy.setdefault(
            copy_index,
            {
                "packet_index": copy_index,
                "event_index": int(row["transmission_index"]),
                "detected_start_sample": int(row["raw_frame_start_sample"]),
                "header_start_sample": int(row["raw_data_start_sample"]),
                "frame_valid": int(row["soft_hard_pattern_valid"]),
                "grlora_framesync_valid": 0,
                "grlora_netid_valid": 0,
                "grlora_fine_payload_start_sample": int(row["raw_data_start_sample"]),
                "grlora_synced_payload_start_sample": int(row["raw_data_start_sample"]),
                "grlora_cfo_int_est": int(row["cfo_int"]),
                "grlora_cfo_frac_est": float(row["cfo_frac"]),
                "grlora_sfo_hat": float(row["sfo_hat"]),
                "grlora_sfo_cum_initial": float(row["sfo_cum_before"]),
                "grlora_payload_sto_frac_est": float(row["sto_frac"]),
                "grlora_payload_sto_sample_correction": 0,
                "grlora_netid_offset": 0,
                "xcopy_alignment_valid": 1,
                "xcopy_alignment_score": float(row["xcopy_alignment_score"]),
                "xcopy_soft_boundary_score": float(row["soft_boundary_score"]),
                "xcopy_soft_boundary_rank": int(row["soft_boundary_rank"]),
                "relative_delay_samples": int(row["relative_delay_samples"]),
                "relative_cfo_hz": float(row["relative_cfo_hz"]),
                "relative_phase_rad": float(row["relative_phase_rad"]),
            },
        )
    output_rows = list(by_copy.values())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _summary(
    result: XCopySyncResult,
    input_path: Path,
    detection_mode: str,
) -> dict[str, object]:
    frame = asdict(result.frame_location) if result.frame_location is not None else None
    frame_sync = None
    if result.frame_sync is not None:
        frame_sync = {
            "valid": bool(result.frame_sync.valid),
            "cfo_total_est": float(result.frame_sync.cfo_total_est),
            "cfo_hz_est": float(result.frame_sync.cfo_hz_est),
            "sfo_hat": float(result.frame_sync.sfo_hat),
            "synced_preamble_start_sample": int(result.frame_sync.synced_preamble_start_sample),
            "synced_payload_start_sample": int(result.frame_sync.synced_payload_start_sample),
        }
    return _json_safe(
        {
            "status": result.status,
            "input": str(input_path.resolve()),
            "detection_mode": str(detection_mode),
            "config": asdict(result.config),
            "detection": {
                key: value
                for key, value in asdict(result.detection).items()
                if key != "bins"
            },
            "reference_copy_index": result.reference_copy_index,
            "scheduled_copy_count": len(result.alignments),
            "aligned_copy_count": result.aligned_copy_count,
            "individually_detected_packet_count": len(result.packet_detections),
            "frame_location": frame,
            "grlora_frame_sync": frame_sync,
            "soft_frame_candidates": [
                asdict(item) for item in result.soft_frame_candidates
            ],
            "combined_iq": "combined_iq.bin" if result.combined_iq is not None else None,
            "aligned_raw_symbols": (
                "aligned_raw_symbols.csv"
                if result.selected_soft_frame is not None
                else None
            ),
        }
    )


def main() -> None:
    args = parse_args()
    _prepare_output(args.output_dir, args.overwrite)
    sample_count = args.input.stat().st_size // np.dtype(np.complex64).itemsize
    if args.input.stat().st_size % np.dtype(np.complex64).itemsize:
        raise ValueError("input is not a headerless complex64 IQ file.")
    limit = sample_count if args.sample_limit is None else min(sample_count, int(args.sample_limit))
    samples = np.memmap(args.input, dtype=np.complex64, mode="r", shape=(limit,))
    config = XCopyConfig(
        sf=args.sf,
        bw=args.bw,
        samp_rate=args.samp_rate,
        preamble_symbols=args.preamble_symbols,
        sync_word=args.sync_word,
        retransmit_period_samples=args.period_samples,
        payload_symbols=args.payload_symbols,
        detection_chirps=args.detection_chirps,
        phase_hop_samples=args.phase_hop_samples,
        min_detection_peak_to_median=args.min_detection_score,
        detection_mad_scale=args.detection_mad_scale,
        detection_peak_fraction=(
            float(args.detection_peak_fraction)
            if args.detection_peak_fraction is not None
            else (0.3 if args.detection_mode == "paper" else 0.5)
        ),
        min_detection_run=args.min_detection_run,
        alignment_search_samples=args.alignment_search_samples,
        alignment_decimation=args.alignment_decimation,
        max_relative_cfo_hz=args.max_relative_cfo_hz,
        min_alignment_peak_to_median=args.min_alignment_score,
        min_aligned_copies=args.min_aligned_copies,
        max_copies=args.max_copies,
        center_freq=args.center_freq,
        soft_frame_top_k=args.soft_frame_top_k,
        soft_frame_search_span_chirps=args.soft_frame_search_span_chirps,
    )
    try:
        result = (
            run_xcopy_paper_sync(samples, config)
            if args.detection_mode == "paper"
            else run_xcopy_sync(samples, config)
        )
        raw_symbol_rows = xcopy_raw_symbol_rows(samples, result)
    finally:
        mmap_handle = getattr(samples, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    _write_scan(args.output_dir / "periodic_scan.csv", result)
    _write_packet_detections(args.output_dir / "packet_detections.csv", result)
    _write_copies(args.output_dir / "copies.csv", result)
    if result.combined_iq is not None:
        result.combined_iq.tofile(args.output_dir / "combined_iq.bin")
    _write_combined_sync(args.output_dir / "combined_sync.csv", result)
    _write_soft_frame_candidates(args.output_dir / "soft_frame_candidates.csv", result)
    _write_raw_symbols(args.output_dir / "aligned_raw_symbols.csv", raw_symbol_rows)
    _write_raw_sync(args.output_dir / "aligned_raw_sync.csv", result, raw_symbol_rows)
    summary = _summary(result, args.input, args.detection_mode)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"status={result.status}")
    print(f"detection_mode={args.detection_mode}")
    print(f"detection_score={result.detection.best_peak_to_median:.6g}")
    print(f"detection_threshold={result.detection.threshold:.6g}")
    print(f"coarse_phase={result.detection.coarse_preamble_phase_sample}")
    print(f"scheduled_copies={len(result.alignments)}")
    print(f"individually_detected_packets={len(result.packet_detections)}")
    print(f"aligned_copies={result.aligned_copy_count}")
    if result.frame_location is not None:
        print(f"frame_valid={int(result.frame_location.valid)}")
        print(f"combined_preamble_start={result.frame_location.preamble_start_sample}")
        print(f"combined_payload_start={result.frame_location.payload_start_sample}")
    if result.frame_sync is not None:
        print(f"grlora_framesync_valid={int(result.frame_sync.valid)}")
    if result.selected_soft_frame is not None:
        print(f"soft_boundary_score={result.selected_soft_frame.score:.6g}")
        print(f"soft_boundary_start={result.selected_soft_frame.preamble_start_sample}")
        print(f"soft_boundary_hard_pattern={int(result.selected_soft_frame.hard_grlora_pattern_valid)}")
        print(f"raw_symbol_rows={len(raw_symbol_rows)}")
    print(f"wrote={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
