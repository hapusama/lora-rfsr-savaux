#!/usr/bin/env python3
"""Paired official-OTA FFT/Savaux/GLS evaluation for task-aware RF-SR."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from scipy import signal
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RFSR_ROOT = REPO_ROOT / "third_party" / "rfsr"
for value in (REPO_ROOT, RFSR_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from rfsr.nn.official_ota_dataset import (  # noqa: E402
    OFFICIAL_GUARD_HIGH,
    OfficialOTARecord,
    scan_official_ota_records,
)
from rfsr.nn.task_model import (  # noqa: E402
    TaskAwareRFSRFrontend,
    load_task_aware_checkpoint,
)
from weak_decoder.os_lora.system.oversampled_glrt import (  # noqa: E402
    BranchNoiseModel,
    estimate_branch_noise_model,
)

# Reuse the locked packet synchronization and decoder scoring implementation;
# this script only supplies a new frontend and leakage-safe packet inventory.
from tools.evaluate_official_rfsr_synthetic_chain import (  # noqa: E402
    BW_HZ,
    HIGH_RATE_HZ,
    N_BINS,
    OUTPUT_RATE_HZ,
    _center_integer_cfo,
    _official_expected_symbols,
    _sync_config,
    evaluate_decoders,
    prepare_samples_and_sync,
    prepare_samples_with_tracked_sync,
)


METHODS = ("interpolation_1msps", "task_rfsr_1msps", "native_1msps")
DECODERS = ("ordinary_fft", "savaux", "savaux_gls")
TRIM_CONTEXT_CHIRPS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test"
    )
    parser.add_argument("--max-reference-ids", type=int)
    parser.add_argument("--minimum-snr-db", type=float, default=-35.0)
    parser.add_argument("--maximum-snr-db", type=float, default=15.0)
    parser.add_argument("--captures-per-reference", type=int, default=2)
    parser.add_argument(
        "--test-reference-limit",
        type=int,
        help="Limit reference IDs in the selected evaluation split.",
    )
    parser.add_argument(
        "--selection",
        choices=("random", "target_snr"),
        default="random",
    )
    parser.add_argument("--selection-snr-db", type=float, default=-15.0)
    parser.add_argument("--calibration-captures", type=int, default=24)
    parser.add_argument("--chunk-input-samples", type=int, default=131_072)
    parser.add_argument("--residual-strength", type=float, default=1.0)
    parser.add_argument(
        "--decode-residual-strength",
        type=float,
        help="Optional residual strength used for decoding after sync locks.",
    )
    parser.add_argument("--gls-training-bins", type=int, default=64)
    parser.add_argument("--gls-loading", type=float, default=0.05)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--permutation-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--sync-mode",
        choices=("common_native", "independent"),
        default="common_native",
        help=(
            "Reuse the native packet location/CFO candidate for all frontends "
            "or run a full search independently for each method."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Per-packet/method score cache; defaults beside --output.",
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.captures_per_reference < 1 or args.calibration_captures < 2:
        parser.error("capture counts are too small")
    if not 0.0 <= args.residual_strength <= 1.0:
        parser.error("residual-strength must be in [0, 1]")
    if (
        args.decode_residual_strength is not None
        and not 0.0 <= args.decode_residual_strength <= 1.0
    ):
        parser.error("decode-residual-strength must be in [0, 1]")
    return args


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _select_balanced_records(
    records: Iterable[OfficialOTARecord],
    *,
    per_reference: int,
    seed: int,
    selection: str = "random",
    selection_snr_db: float = -15.0,
) -> list[OfficialOTARecord]:
    grouped: dict[int, list[OfficialOTARecord]] = defaultdict(list)
    for record in records:
        grouped[int(record.reference_id)].append(record)
    selected: list[OfficialOTARecord] = []
    for reference_id in sorted(grouped):
        values = sorted(grouped[reference_id], key=lambda item: str(item.ota_path))
        if selection == "random":
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(reference_id)])
            )
            indexes = rng.permutation(len(values))[: int(per_reference)]
        elif selection == "target_snr":
            indexes = sorted(
                range(len(values)),
                key=lambda index: (
                    abs(values[index].snr_db - float(selection_snr_db)),
                    str(values[index].ota_path),
                ),
            )[: int(per_reference)]
        else:
            raise ValueError(f"unsupported selection strategy: {selection}")
        selected.extend(values[int(index)] for index in indexes)
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_key(
    *,
    record: OfficialOTARecord,
    method: str,
    checkpoint_sha256: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    stat = record.ota_path.stat()
    return {
        "schema": "official-task-rfsr-method-cache-v1",
        "packet_id": record.ota_path.stem,
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "method": str(method),
        "checkpoint_sha256": checkpoint_sha256,
        "residual_strength": args.residual_strength,
        "decode_residual_strength": args.decode_residual_strength,
        "sync_mode": args.sync_mode,
        "gls_training_bins": args.gls_training_bins,
        "gls_loading": args.gls_loading,
        "gls_calibration_captures": args.calibration_captures,
        "split_seed": args.split_seed,
        "max_reference_ids": args.max_reference_ids,
        "minimum_snr_db": args.minimum_snr_db,
        "maximum_snr_db": args.maximum_snr_db,
        "decoders": list(DECODERS),
    }


def _cache_path(cache_dir: Path, record: OfficialOTARecord, method: str) -> Path:
    return cache_dir / record.ota_path.stem / f"{method}.json"


def _read_cached_rows(
    path: Path, expected_key: dict[str, object]
) -> list[dict[str, object]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stored_key = payload.get("key")
    # Interpolation and native rows are checkpoint-independent.  Reuse them
    # across RF-SR ablations; only the learned frontend must be recomputed.
    if (
        isinstance(stored_key, dict)
        and expected_key.get("method") != "task_rfsr_1msps"
    ):
        stored_key = dict(stored_key)
        stored_key["checkpoint_sha256"] = expected_key["checkpoint_sha256"]
        stored_key["residual_strength"] = expected_key["residual_strength"]
        stored_key["decode_residual_strength"] = expected_key[
            "decode_residual_strength"
        ]
    elif isinstance(stored_key, dict):
        stored_key = dict(stored_key)
        if (
            float(expected_key.get("residual_strength", 1.0)) == 1.0
            and "residual_strength" not in stored_key
        ):
            stored_key["residual_strength"] = 1.0
        if (
            expected_key.get("decode_residual_strength") is None
            and "decode_residual_strength" not in stored_key
        ):
            stored_key["decode_residual_strength"] = None
    if stored_key != expected_key:
        # Accept caches written by the immediately preceding schema revision
        # only for that revision's fixed default inventory.  Non-default SNR
        # filters or reference limits must never reuse it.
        legacy_key = dict(expected_key)
        for field in (
            "max_reference_ids",
            "minimum_snr_db",
            "maximum_snr_db",
        ):
            legacy_key.pop(field, None)
        legacy_defaults = (
            expected_key.get("max_reference_ids") is None
            and float(expected_key.get("minimum_snr_db", -35.0)) == -35.0
            and float(expected_key.get("maximum_snr_db", 15.0)) == 15.0
        )
        if not legacy_defaults or stored_key != legacy_key:
            return None
    rows = payload.get("rows")
    return rows if isinstance(rows, list) else None


def _write_cached_rows(
    path: Path,
    key: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps({"key": key, "rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _trim_official_high(samples: np.ndarray) -> tuple[np.ndarray, int, int]:
    values = np.asarray(samples, dtype=np.complex64)
    context = int(
        TRIM_CONTEXT_CHIRPS * N_BINS * (HIGH_RATE_HZ // BW_HZ)
    )
    start = int(OFFICIAL_GUARD_HIGH - context)
    stop = int(values.size - OFFICIAL_GUARD_HIGH + context)
    if start < 0 or stop > values.size or start % 8 or stop % 8:
        raise ValueError("official trim is not aligned to the 250 kS/s grid")
    return (
        np.asarray(values[start:stop], dtype=np.complex64),
        context,
        context,
    )


def _method_outputs(
    high: np.ndarray, frontend: TaskAwareRFSRFrontend
) -> dict[str, np.ndarray]:
    values = np.asarray(high, dtype=np.complex64)
    low = np.asarray(values[::8], dtype=np.complex64)
    interpolation = np.asarray(
        signal.resample_poly(low, 4, 1), dtype=np.complex64
    )
    task = frontend.enhance(low)
    native = np.asarray(values[::2], dtype=np.complex64)
    expected = 4 * low.size
    outputs = {
        "interpolation_1msps": interpolation[:expected],
        "task_rfsr_1msps": task[:expected],
        "native_1msps": native[:expected],
    }
    if any(item.size != expected for item in outputs.values()):
        raise RuntimeError("frontend output lengths differ")
    return outputs


def _estimate_gls_models(
    records: list[OfficialOTARecord],
    frontend: TaskAwareRFSRFrontend,
    *,
    capture_limit: int,
    training_bins: int,
    loading: float,
) -> tuple[dict[str, BranchNoiseModel], dict[str, object]]:
    symbol_samples = N_BINS * (OUTPUT_RATE_HZ // BW_HZ)
    windows: dict[str, list[np.ndarray]] = {name: [] for name in METHODS}
    for record in records[: int(capture_limit)]:
        raw = np.memmap(record.ota_path, dtype=np.dtype("<c8"), mode="r")
        # One complete output-rate symbol from the leading off-packet guard.
        high_noise = np.asarray(raw[: 2 * symbol_samples], dtype=np.complex64)
        for method, values in _method_outputs(high_noise, frontend).items():
            windows[method].append(np.asarray(values[:symbol_samples]))
    bins = tuple(
        int(value)
        for value in np.linspace(
            0,
            N_BINS,
            min(int(training_bins), N_BINS),
            endpoint=False,
        )
    )
    models: dict[str, BranchNoiseModel] = {}
    report: dict[str, object] = {}
    for method, rows in windows.items():
        model = estimate_branch_noise_model(
            np.stack(rows),
            sf=12,
            os_factor=8,
            training_bins=bins,
            diagonal_loading=float(loading),
            covariance_mode="pooled",
        )
        models[method] = model
        covariance = np.asarray(model.covariance, dtype=np.complex128)
        diagonal = np.maximum(np.real(np.diag(covariance)), 1e-30)
        correlation = covariance / np.sqrt(diagonal[:, None] * diagonal[None, :])
        off_diagonal = ~np.eye(covariance.shape[0], dtype=bool)
        report[method] = {
            "snapshot_count": model.snapshot_count,
            "condition_number": float(np.linalg.cond(covariance)),
            "mean_offdiagonal_abs_correlation": float(
                np.mean(np.abs(correlation[off_diagonal]))
            ),
        }
    return models, report


def paired_cluster_statistics(
    rows: list[dict[str, object]],
    *,
    decoder: str,
    candidate: str = "task_rfsr_1msps",
    baseline: str = "interpolation_1msps",
    bootstrap_repetitions: int = 20_000,
    permutation_repetitions: int = 100_000,
    seed: int = 0,
) -> dict[str, object]:
    """Reference-ID cluster bootstrap and paired sign-flip test."""

    by_key = {
        (int(row["reference_id"]), str(row["packet_id"]), str(row["method"])): row
        for row in rows
        if str(row["decoder"]) == str(decoder)
    }
    reference_ids = sorted(
        {
            key[0]
            for key in by_key
            if key[2] == candidate
            and (key[0], key[1], baseline) in by_key
        }
    )
    cluster_values: list[tuple[int, int, int, int]] = []
    for reference_id in reference_ids:
        candidate_errors = baseline_errors = 0
        candidate_count = baseline_count = 0
        packet_ids = sorted(
            {
                key[1]
                for key in by_key
                if key[0] == reference_id and key[2] == candidate
                and (reference_id, key[1], baseline) in by_key
            }
        )
        for packet_id in packet_ids:
            left = by_key[(reference_id, packet_id, candidate)]
            right = by_key[(reference_id, packet_id, baseline)]
            candidate_errors += int(left["symbol_errors"])
            candidate_count += int(left["symbol_count"])
            baseline_errors += int(right["symbol_errors"])
            baseline_count += int(right["symbol_count"])
        cluster_values.append(
            (candidate_errors, candidate_count, baseline_errors, baseline_count)
        )
    if len(cluster_values) < 2:
        raise ValueError("paired statistics require at least two reference clusters")
    values = np.asarray(cluster_values, dtype=np.float64)

    def difference(selected: np.ndarray) -> float:
        left = float(selected[:, 0].sum() / selected[:, 1].sum())
        right = float(selected[:, 2].sum() / selected[:, 3].sum())
        return left - right

    observed = difference(values)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(bootstrap_repetitions), dtype=np.float64)
    for index in range(bootstrap.size):
        chosen = rng.integers(0, len(values), size=len(values))
        bootstrap[index] = difference(values[chosen])
    cluster_differences = values[:, 0] / values[:, 1] - values[:, 2] / values[:, 3]
    permutation = np.empty(int(permutation_repetitions), dtype=np.float64)
    batch = 10_000
    for start in range(0, permutation.size, batch):
        stop = min(permutation.size, start + batch)
        signs = rng.choice((-1.0, 1.0), size=(stop - start, len(values)))
        permutation[start:stop] = np.mean(
            signs * cluster_differences[None, :], axis=1
        )
    p_one_sided = float(
        (1 + np.count_nonzero(permutation <= np.mean(cluster_differences)))
        / (permutation.size + 1)
    )
    ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "reference_clusters": len(values),
        "candidate_ser": float(values[:, 0].sum() / values[:, 1].sum()),
        "baseline_ser": float(values[:, 2].sum() / values[:, 3].sum()),
        "ser_difference_candidate_minus_baseline": observed,
        "cluster_bootstrap_95pct_ci": [float(ci_low), float(ci_high)],
        "paired_sign_flip_p_one_sided_improvement": p_one_sided,
        "significant_improvement_0p05": bool(ci_high < 0.0 and p_one_sided < 0.05),
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for method in METHODS:
        result[method] = {}
        for decoder in DECODERS:
            selected = [
                row
                for row in rows
                if row["method"] == method and row["decoder"] == decoder
            ]
            errors = sum(int(row["symbol_errors"]) for row in selected)
            count = sum(int(row["symbol_count"]) for row in selected)
            margins = [
                float(row["median_peak_margin_db"])
                for row in selected
                if row["median_peak_margin_db"] is not None
            ]
            result[method][decoder] = {
                "packets": len(selected),
                "synchronized_packets": sum(bool(row["synchronized"]) for row in selected),
                "symbol_errors": errors,
                "symbol_count": count,
                "ser": float(errors / count),
                "median_packet_peak_margin_db": (
                    float(np.median(margins)) if margins else None
                ),
            }
    return result


def main() -> None:
    args = parse_args()
    started = perf_counter()
    device = _device(args.device)
    model, checkpoint = load_task_aware_checkpoint(args.checkpoint, device)
    frontend = TaskAwareRFSRFrontend(
        model,
        device=device,
        chunk_input_samples=args.chunk_input_samples,
        residual_strength=args.residual_strength,
    )
    decode_strength = (
        args.residual_strength
        if args.decode_residual_strength is None
        else float(args.decode_residual_strength)
    )
    decode_frontend = (
        frontend
        if decode_strength == float(args.residual_strength)
        else TaskAwareRFSRFrontend(
            model,
            device=device,
            chunk_input_samples=args.chunk_input_samples,
            residual_strength=decode_strength,
        )
    )
    snr_range = (args.minimum_snr_db, args.maximum_snr_db)
    evaluation_records, splits = scan_official_ota_records(
        args.official_root,
        split=args.split,
        split_seed=args.split_seed,
        max_reference_ids=args.max_reference_ids,
        snr_range=snr_range,
    )
    calibration_records, _ = scan_official_ota_records(
        args.official_root,
        split=("validation" if args.split == "test" else "train"),
        split_seed=args.split_seed,
        max_reference_ids=args.max_reference_ids,
        snr_range=snr_range,
    )
    bound_ids = set(
        checkpoint.get("split_manifest", {})
        .get("reference_ids", {})
        .get(args.split, [])
    )
    if bound_ids and bound_ids != set(splits[args.split]):
        raise RuntimeError("evaluation split does not match checkpoint binding")
    if args.test_reference_limit is not None:
        limited_ids = set(
            list(splits[args.split])[: int(args.test_reference_limit)]
        )
        if len(limited_ids) < 2:
            raise ValueError("test-reference-limit must retain at least two IDs")
        evaluation_records = [
            record
            for record in evaluation_records
            if record.reference_id in limited_ids
        ]
    selected = _select_balanced_records(
        evaluation_records,
        per_reference=args.captures_per_reference,
        seed=args.seed,
        selection=args.selection,
        selection_snr_db=args.selection_snr_db,
    )
    gls_models, gls_report = _estimate_gls_models(
        calibration_records,
        decode_frontend,
        capture_limit=args.calibration_captures,
        training_bins=args.gls_training_bins,
        loading=args.gls_loading,
    )
    selected_reference_count = len(
        {record.reference_id for record in selected}
    )
    print(
        f"device={device} split={args.split} "
        f"references={selected_reference_count} captures={len(selected)}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    checkpoint_sha256 = _sha256(args.checkpoint.expanduser().resolve())
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else args.output.expanduser().resolve().with_suffix("").with_name(
            args.output.expanduser().resolve().stem + "_cache"
        )
    )
    cache_hits = 0
    cache_misses = 0
    sync_config = _sync_config(923_000_000.0)
    for packet_index, record in enumerate(selected):
        print(
            f"packet {packet_index + 1}/{len(selected)} {record.ota_path.name} "
            f"source_snr={record.snr_db:.2f} dB",
            flush=True,
        )
        raw = np.memmap(record.ota_path, dtype=np.dtype("<c8"), mode="r")
        high, _, _ = _trim_official_high(raw)
        metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
        expected = _official_expected_symbols(metadata)
        outputs = _method_outputs(high, frontend)
        decode_task_output = (
            outputs["task_rfsr_1msps"]
            if decode_frontend is frontend
            else _method_outputs(high, decode_frontend)["task_rfsr_1msps"]
        )
        pending: dict[str, tuple[Path, dict[str, object]]] = {}
        for method in METHODS:
            cache_path = _cache_path(cache_dir, record, method)
            key = _cache_key(
                record=record,
                method=method,
                checkpoint_sha256=checkpoint_sha256,
                args=args,
            )
            cached = None if args.no_cache else _read_cached_rows(cache_path, key)
            if cached is not None:
                rows.extend(cached)
                cache_hits += 1
                print(f"  {method}: cache hit", flush=True)
            else:
                pending[method] = (cache_path, key)
                cache_misses += 1
        if not pending:
            continue

        native_preparation = None
        if args.sync_mode == "common_native":
            native_preparation = prepare_samples_and_sync(
                outputs["native_1msps"],
                sync_config,
                coarse_cfo_centering=True,
            )
        for method in METHODS:
            if method not in pending:
                continue
            output = outputs[method]
            if args.sync_mode == "common_native":
                assert native_preparation is not None
                preparation = (
                    native_preparation
                    if method == "native_1msps"
                    else prepare_samples_with_tracked_sync(
                        output, sync_config, native_preparation
                    )
                )
            else:
                preparation = prepare_samples_and_sync(
                    output, sync_config, coarse_cfo_centering=True
                )
            decode_samples = preparation.samples
            if method == "task_rfsr_1msps" and decode_frontend is not frontend:
                decode_samples, _ = _center_integer_cfo(
                    decode_task_output,
                    sync_config,
                    int(preparation.coarse_cfo_bin),
                )
            scores = evaluate_decoders(
                decode_samples,
                preparation.result,
                expected,
                gls_models[method],
                decoder_names=DECODERS,
            )
            method_rows: list[dict[str, object]] = []
            for decoder, score in scores.items():
                method_rows.append(
                    {
                        "packet_index": packet_index,
                        "packet_id": record.ota_path.stem,
                        "reference_id": record.reference_id,
                        "source_snr_db": record.snr_db,
                        "method": method,
                        "decoder": decoder,
                        "synchronized": bool(
                            preparation.result.synchronized
                            and preparation.result.frame_sync is not None
                        ),
                        **score,
                    }
                )
            rows.extend(method_rows)
            if not args.no_cache:
                cache_path, key = pending[method]
                _write_cached_rows(cache_path, key, method_rows)
                print(f"  {method}: cached", flush=True)

    paired = {
        decoder: paired_cluster_statistics(
            rows,
            decoder=decoder,
            bootstrap_repetitions=args.bootstrap_repetitions,
            permutation_repetitions=args.permutation_repetitions,
            seed=args.seed + index,
        )
        for index, decoder in enumerate(DECODERS)
    }
    report = {
        "schema": "official-task-rfsr-paired-evaluation-v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "official_root": str(args.official_root.expanduser().resolve()),
        "split_seed": args.split_seed,
        "evaluation_split": args.split,
        "evaluation_reference_ids": sorted(
            {record.reference_id for record in selected}
        ),
        "captures_per_reference": args.captures_per_reference,
        "selection": args.selection,
        "selection_snr_db": args.selection_snr_db,
        "sync_mode": args.sync_mode,
        "residual_strength": args.residual_strength,
        "decode_residual_strength": decode_strength,
        "selected_capture_count": len(selected),
        "cache": {
            "directory": str(cache_dir),
            "hits": cache_hits,
            "misses": cache_misses,
        },
        "gls_calibration": gls_report,
        "aggregate": _aggregate(rows),
        "paired_task_vs_interpolation": paired,
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["aggregate"], sort_keys=True), flush=True)
    print(json.dumps(paired, sort_keys=True), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
