#!/usr/bin/env python3
"""Train the task-aware RF-SR model on the public official OTA archive."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
RFSR_ROOT = REPO_ROOT / "third_party" / "rfsr"
for value in (REPO_ROOT, RFSR_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from rfsr.nn.official_ota_dataset import OfficialOTASymbolDataset  # noqa: E402
from rfsr.nn.task_loss import TaskAwareRFSRLoss  # noqa: E402
from rfsr.nn.task_model import (  # noqa: E402
    TaskAwarePolyphaseTCN,
    input_rms_scale,
    linear_polyphase_baseline,
    load_task_aware_checkpoint,
    sinc_polyphase_baseline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--pretrained",
        type=Path,
        help="Initialize from a task-aware synthetic-pretraining checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-reference-ids", type=int)
    parser.add_argument("--minimum-snr-db", type=float, default=-35.0)
    parser.add_argument("--maximum-snr-db", type=float, default=15.0)
    parser.add_argument("--task-minimum-snr-db", type=float, default=-20.0)
    parser.add_argument(
        "--target-source",
        choices=("received", "reference"),
        default="received",
        help="Use received for the strict hard-consistent RF-SR task.",
    )
    parser.add_argument("--symbols-per-capture", type=int, default=4)
    parser.add_argument("--train-capture-limit", type=int)
    parser.add_argument("--validation-capture-limit", type=int, default=160)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument(
        "--dilations", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32, 64, 128)
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--baseline", choices=("linear", "sinc"), default="sinc")
    parser.add_argument("--sinc-radius", type=int, default=16)
    parser.add_argument(
        "--spectral-mode",
        choices=("four_branch", "savaux"),
        default="savaux",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--waveform-warmup-epochs", type=int, default=2)
    parser.add_argument("--concentration-weight", type=float, default=0.05)
    parser.add_argument("--margin-weight", type=float, default=0.05)
    parser.add_argument("--correct-radius", type=int, default=1)
    parser.add_argument("--guard-radius", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.waveform_warmup_epochs < 0:
        parser.error("epoch counts must be non-negative and epochs must be positive")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers non-negative")
    return args


def _device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _limit_records(dataset: OfficialOTASymbolDataset, limit: int | None) -> None:
    if limit is None:
        return
    count = int(limit)
    if count < 1:
        raise ValueError("capture limits must be positive")
    dataset.records = dataset.records[:count]


def _loader(
    dataset: OfficialOTASymbolDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        # Workers are recreated each epoch so ``dataset.set_epoch`` reaches
        # them; persistent copies would silently repeat the same symbol views.
        persistent_workers=False,
        generator=generator,
    )


def _flatten(
    batch: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = batch["x"].flatten(0, 1).to(device, non_blocking=True)
    y = batch["y"].flatten(0, 1).to(device, non_blocking=True)
    correct = batch["correct_bins"].flatten().to(device, non_blocking=True)
    task_mask = batch["task_mask"].flatten().to(device, non_blocking=True)
    valid_mask = batch.get("valid_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.flatten(0, 1).to(device, non_blocking=True)
    return x, y, correct, task_mask, valid_mask


def _model_baseline(
    model: TaskAwarePolyphaseTCN, x: torch.Tensor
) -> torch.Tensor:
    if model.baseline == "sinc":
        return sinc_polyphase_baseline(x, radius=model.sinc_radius)
    return linear_polyphase_baseline(x)


def _run_epoch(
    *,
    model: TaskAwarePolyphaseTCN,
    objective: TaskAwareRFSRLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    concentration_weight: float,
    margin_weight: float,
    gradient_clip: float,
    supervise_observed: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {
        "total": 0.0,
        "waveform": 0.0,
        "concentration": 0.0,
        "margin": 0.0,
        "baseline_waveform": 0.0,
        "residual_centered_rms": 0.0,
        "hard_consistency_max": 0.0,
        "gradient_norm": 0.0,
    }
    batches = 0
    for batch in loader:
        x, y, correct, task_mask, valid_mask = _flatten(batch, device)
        scale = input_rms_scale(x)
        x_normalized = x / scale
        y_normalized = y / scale
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        grad_context = torch.enable_grad() if training else torch.no_grad()
        with grad_context, autocast:
            prediction = model(x_normalized)
        # CUDA complex-half FFT is unsupported; the small loss branch remains
        # float32 even when the convolutional trunk uses AMP.
        prediction = prediction.float()
        loss_valid_mask = None if supervise_observed else valid_mask
        terms = objective(
            prediction,
            y_normalized.float(),
            correct,
            task_mask=task_mask,
            concentration_weight=concentration_weight,
            margin_weight=margin_weight,
            valid_mask=loss_valid_mask,
        )
        gradient_norm = 0.0
        if training:
            scaler.scale(terms["total"]).backward()
            scaler.unscale_(optimizer)
            gradient_norm = float(
                clip_grad_norm_(model.parameters(), float(gradient_clip))
            )
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            baseline = _model_baseline(model, x_normalized)
            baseline_waveform = objective.waveform_loss(
                baseline, y_normalized.float(), loss_valid_mask
            )
            residual = prediction - baseline
            centered = residual - residual.mean(dim=-1, keepdim=True)
            residual_rms = centered.square().sum(dim=1).mean().sqrt()
            hard_error = (prediction[..., ::4] - x_normalized).abs().max()
        for name in ("total", "waveform", "concentration", "margin"):
            sums[name] += float(terms[name].detach())
        sums["baseline_waveform"] += float(baseline_waveform)
        sums["residual_centered_rms"] += float(residual_rms)
        sums["hard_consistency_max"] = max(
            sums["hard_consistency_max"], float(hard_error)
        )
        sums["gradient_norm"] += gradient_norm
        batches += 1
    if batches == 0:
        raise RuntimeError("data loader produced no batches")
    for name in sums:
        if name != "hard_consistency_max":
            sums[name] /= batches
    sums["batches"] = float(batches)
    return sums


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    started = perf_counter()
    _seed_everything(args.seed)
    device = _device(args.device)
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    snr_range = (args.minimum_snr_db, args.maximum_snr_db)
    common = dict(
        root=args.official_root,
        split_seed=args.split_seed,
        max_reference_ids=args.max_reference_ids,
        snr_range=snr_range,
        symbols_per_capture=args.symbols_per_capture,
        target_source=args.target_source,
        task_min_snr_db=args.task_minimum_snr_db,
    )
    train_dataset = OfficialOTASymbolDataset(split="train", **common)
    validation_dataset = OfficialOTASymbolDataset(split="validation", **common)
    _limit_records(train_dataset, args.train_capture_limit)
    _limit_records(validation_dataset, args.validation_capture_limit)
    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed + 1,
    )

    pretrained_payload: dict[str, object] | None = None
    if args.pretrained is not None:
        model, pretrained_payload = load_task_aware_checkpoint(
            args.pretrained, device=device
        )
        model_config = dict(pretrained_payload["model_config"])
        if "dilations" in model_config:
            model_config["dilations"] = list(model_config["dilations"])
    else:
        model_config = {
            "channels": int(args.channels),
            "dilations": [int(value) for value in args.dilations],
            "dropout": float(args.dropout),
            "baseline": str(args.baseline),
            "sinc_radius": int(args.sinc_radius),
            "hard_observed": True,
        }
        model = TaskAwarePolyphaseTCN(
            channels=model_config["channels"],
            dilations=tuple(model_config["dilations"]),
            dropout=model_config["dropout"],
            baseline=model_config["baseline"],
            sinc_radius=model_config["sinc_radius"],
            hard_observed=model_config["hard_observed"],
        ).to(device)
    objective = TaskAwareRFSRLoss(
        spectral_mode=args.spectral_mode,
        correct_radius=args.correct_radius,
        guard_radius=args.guard_radius,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    split_manifest = {
        "schema": "official-rfsr-reference-split-v1",
        "algorithm": "deterministic reference/payload identity 6:2:2",
        "split_seed": int(args.split_seed),
        "reference_ids": {
            name: list(values)
            for name, values in train_dataset.reference_splits.items()
        },
        "capture_counts_after_snr_filter_and_limits": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
        },
        "snr_range_db": list(snr_range),
        "target_source": args.target_source,
    }
    _write_json(run_dir / "split_manifest.json", split_manifest)
    _write_json(
        run_dir / "config.json",
        {
            **vars(args),
            "official_root": str(args.official_root.expanduser().resolve()),
            "run_dir": str(run_dir),
            "pretrained": (
                str(args.pretrained.expanduser().resolve())
                if args.pretrained is not None
                else None
            ),
            "device": str(device),
            "model_config": model_config,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "receptive_field_input_samples": model.receptive_field_input_samples,
        },
    )
    print(
        f"device={device} train_captures={len(train_dataset)} "
        f"validation_captures={len(validation_dataset)} "
        f"parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    best_validation = math.inf
    stale_epochs = 0
    history: list[dict[str, object]] = []
    for epoch in range(int(args.epochs)):
        train_dataset.set_epoch(epoch)
        task_enabled = epoch >= int(args.waveform_warmup_epochs)
        concentration_weight = (
            float(args.concentration_weight) if task_enabled else 0.0
        )
        margin_weight = float(args.margin_weight) if task_enabled else 0.0
        train_metrics = _run_epoch(
            model=model,
            objective=objective,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            concentration_weight=concentration_weight,
            margin_weight=margin_weight,
            gradient_clip=args.gradient_clip,
        )
        validation_metrics = _run_epoch(
            model=model,
            objective=objective,
            loader=validation_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            # Score every checkpoint with the final objective so warm-up and
            # task-aware epochs remain directly comparable for model selection.
            concentration_weight=float(args.concentration_weight),
            margin_weight=float(args.margin_weight),
            gradient_clip=args.gradient_clip,
        )
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "task_loss_enabled": task_enabled,
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_seconds": perf_counter() - started,
        }
        history.append(row)
        _write_json(run_dir / "history.json", history)
        print(json.dumps(row, sort_keys=True), flush=True)

        score = float(validation_metrics["total"])
        if score < best_validation:
            best_validation = score
            stale_epochs = 0
            checkpoint = {
                "schema": "task-aware-rfsr-v1",
                "epoch": epoch + 1,
                "model_config": model_config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_config": {
                    "target_source": args.target_source,
                    "input_rate_hz": 250_000,
                    "output_rate_hz": 1_000_000,
                    "normalization": "input complex RMS; same scale for target",
                    "hard_observed_polyphase": bool(model.hard_observed),
                    "waveform_loss": "complex Charbonnier on valid symbols",
                    "spectral_mode": args.spectral_mode,
                    "correct_radius": int(args.correct_radius),
                    "guard_radius": int(args.guard_radius),
                    "concentration_weight": concentration_weight,
                    "margin_weight": margin_weight,
                    "pretrained_checkpoint": (
                        str(args.pretrained.expanduser().resolve())
                        if args.pretrained is not None
                        else None
                    ),
                    "pretrained_epoch": (
                        pretrained_payload.get("epoch")
                        if pretrained_payload is not None
                        else None
                    ),
                },
                "split_manifest": split_manifest,
                "validation_metrics": validation_metrics,
            }
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.early_stop_patience):
                print(f"early stopping after epoch {epoch + 1}", flush=True)
                break

    _write_json(
        run_dir / "summary.json",
        {
            "best_validation_total": best_validation,
            "epochs_completed": len(history),
            "elapsed_seconds": perf_counter() - started,
            "checkpoint": str(run_dir / "best.pt"),
        },
    )


if __name__ == "__main__":
    main()
