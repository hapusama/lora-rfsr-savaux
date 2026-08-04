#!/usr/bin/env python3
"""Fast symbol-level gate for official task-aware RF-SR checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RFSR_ROOT = REPO_ROOT / "third_party" / "rfsr"
for value in (REPO_ROOT, RFSR_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from rfsr.nn.official_ota_dataset import OfficialOTASymbolDataset  # noqa: E402
from rfsr.nn.task_loss import TaskAwareRFSRLoss  # noqa: E402
from rfsr.nn.task_model import (  # noqa: E402
    input_rms_scale,
    linear_polyphase_baseline,
    load_task_aware_checkpoint,
    sinc_polyphase_baseline,
)
from tools.train_official_task_rfsr import _device, _loader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for every checkpoint to compare.",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--capture-limit", type=int, default=160)
    parser.add_argument("--symbols-per-capture", type=int, default=4)
    parser.add_argument("--minimum-snr-db", type=float, default=-35.0)
    parser.add_argument("--maximum-snr-db", type=float, default=15.0)
    parser.add_argument("--task-minimum-snr-db", type=float, default=-20.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_checkpoints(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"checkpoint must be NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"empty or duplicate checkpoint name: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = path
    return result


@torch.inference_mode()
def _evaluate_method(
    *,
    loader,
    objective: TaskAwareRFSRLoss,
    device: torch.device,
    predictor,
) -> dict[str, float | int]:
    waveform_sum = concentration_sum = margin_sum = 0.0
    hard_max = 0.0
    exact = within_one = task_symbols = all_symbols = batches = 0
    for batch in loader:
        x = batch["x"].flatten(0, 1).to(device, non_blocking=True)
        y = batch["y"].flatten(0, 1).to(device, non_blocking=True)
        correct = batch["correct_bins"].flatten().to(device, non_blocking=True)
        task_mask = batch["task_mask"].flatten().to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].flatten(0, 1).to(device, non_blocking=True)
        scale = input_rms_scale(x)
        normalized_x = x / scale
        normalized_y = y / scale
        prediction = predictor(normalized_x).float()
        power = objective.savaux_coherent_power(prediction)
        concentration, margin = objective.task_losses(power, correct, task_mask)
        waveform = objective.waveform_loss(prediction, normalized_y, valid_mask)
        peaks = power.argmax(dim=1)
        distance = torch.minimum(
            (peaks - correct).remainder(4096),
            (correct - peaks).remainder(4096),
        )
        active = task_mask > 0
        exact += int(((distance == 0) & active).sum())
        within_one += int(((distance <= 1) & active).sum())
        task_symbols += int(active.sum())
        all_symbols += int(prediction.shape[0])
        waveform_sum += float(waveform)
        concentration_sum += float(concentration)
        margin_sum += float(margin)
        hard_max = max(
            hard_max,
            float((prediction[..., ::4] - normalized_x).abs().max()),
        )
        batches += 1
    return {
        "batches": batches,
        "all_symbols": all_symbols,
        "task_symbols": task_symbols,
        "waveform_charbonnier_missing_phases": waveform_sum / batches,
        "savaux_concentration": concentration_sum / batches,
        "savaux_margin_loss": margin_sum / batches,
        "savaux_exact_bin_accuracy": exact / task_symbols,
        "savaux_within_one_bin_accuracy": within_one / task_symbols,
        "hard_consistency_max": hard_max,
    }


def main() -> None:
    args = parse_args()
    checkpoints = _parse_checkpoints(args.checkpoint)
    device = _device(args.device)
    dataset = OfficialOTASymbolDataset(
        args.official_root,
        split=args.split,
        split_seed=args.split_seed,
        snr_range=(args.minimum_snr_db, args.maximum_snr_db),
        symbols_per_capture=args.symbols_per_capture,
        target_source="received",
        task_min_snr_db=args.task_minimum_snr_db,
    )
    dataset.records = dataset.records[: int(args.capture_limit)]
    loader = _loader(
        dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=0,
    )
    objective = TaskAwareRFSRLoss(spectral_mode="savaux").to(device)
    report: dict[str, object] = {
        "schema": "official-task-rfsr-symbol-gate-v1",
        "split": args.split,
        "capture_count": len(dataset),
        "symbols_per_capture": args.symbols_per_capture,
        "methods": {},
        "checkpoints": {},
    }
    predictors = {
        "linear_interpolation": linear_polyphase_baseline,
        "sinc_interpolation": sinc_polyphase_baseline,
    }
    for name, predictor in predictors.items():
        report["methods"][name] = _evaluate_method(
            loader=loader, objective=objective, device=device, predictor=predictor
        )
        print(name, json.dumps(report["methods"][name], sort_keys=True), flush=True)
    for name, path in checkpoints.items():
        model, payload = load_task_aware_checkpoint(path, device=device)
        report["methods"][name] = _evaluate_method(
            loader=loader, objective=objective, device=device, predictor=model
        )
        report["checkpoints"][name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "epoch": payload.get("epoch"),
            "model_config": payload.get("model_config"),
        }
        print(name, json.dumps(report["methods"][name], sort_keys=True), flush=True)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
