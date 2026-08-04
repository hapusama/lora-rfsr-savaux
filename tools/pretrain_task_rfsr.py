#!/usr/bin/env python3
"""Pretrain task-aware RF-SR on random valid PHY payloads and RF channels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
RFSR_ROOT = REPO_ROOT / "third_party" / "rfsr"
for value in (REPO_ROOT, RFSR_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from rfsr.nn.task_loss import TaskAwareRFSRLoss  # noqa: E402
from rfsr.nn.task_model import TaskAwarePolyphaseTCN  # noqa: E402
from rfsr.nn.task_pretraining_dataset import (  # noqa: E402
    TaskAwareSyntheticSymbolDataset,
)
from tools.train_official_task_rfsr import (  # noqa: E402
    _device,
    _loader,
    _run_epoch,
    _seed_everything,
    _write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-items", type=int, default=2_000)
    parser.add_argument("--validation-items", type=int, default=400)
    parser.add_argument("--symbols-per-item", type=int, default=4)
    parser.add_argument("--payload-length", type=int, default=20)
    parser.add_argument("--snr-min-db", type=float, default=-24.0)
    parser.add_argument("--snr-max-db", type=float, default=8.0)
    parser.add_argument("--cfo-max-hz", type=float, default=12_000.0)
    parser.add_argument("--sto-max-output-samples", type=float, default=6.0)
    parser.add_argument("--sfo-max-ppm", type=float, default=25.0)
    parser.add_argument("--maximum-multipath-delay", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument(
        "--dilations", type=int, nargs="+", default=(1, 2, 4, 8, 16, 32, 64, 128)
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--baseline", choices=("linear", "sinc"), default="sinc")
    parser.add_argument("--sinc-radius", type=int, default=16)
    parser.add_argument(
        "--soft-observed",
        action="store_true",
        help="Learn denoising corrections for the observed branch too.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--waveform-warmup-epochs", type=int, default=1)
    parser.add_argument("--concentration-weight", type=float, default=0.05)
    parser.add_argument("--margin-weight", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.train_items < 1 or args.validation_items < 1:
        parser.error("dataset sizes must be positive")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs/batch-size must be positive and workers non-negative")
    return args


def _dataset(args: argparse.Namespace, *, validation: bool) -> TaskAwareSyntheticSymbolDataset:
    return TaskAwareSyntheticSymbolDataset(
        item_count=(args.validation_items if validation else args.train_items),
        symbols_per_item=args.symbols_per_item,
        payload_length=args.payload_length,
        seed=args.seed + (10_000 if validation else 0),
        snr_range_db=(args.snr_min_db, args.snr_max_db),
        cfo_range_hz=(-args.cfo_max_hz, args.cfo_max_hz),
        sto_range_output_samples=(
            -args.sto_max_output_samples,
            args.sto_max_output_samples,
        ),
        sfo_range_ppm=(-args.sfo_max_ppm, args.sfo_max_ppm),
        maximum_multipath_delay=args.maximum_multipath_delay,
    )


def main() -> None:
    args = parse_args()
    started = perf_counter()
    _seed_everything(args.seed)
    device = _device(args.device)
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = _dataset(args, validation=False)
    validation_dataset = _dataset(args, validation=True)
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

    model_config = {
        "channels": int(args.channels),
        "dilations": [int(value) for value in args.dilations],
        "dropout": float(args.dropout),
        "baseline": str(args.baseline),
        "sinc_radius": int(args.sinc_radius),
        "hard_observed": not bool(args.soft_observed),
    }
    model = TaskAwarePolyphaseTCN(
        channels=model_config["channels"],
        dilations=tuple(model_config["dilations"]),
        dropout=model_config["dropout"],
        baseline=model_config["baseline"],
        sinc_radius=model_config["sinc_radius"],
        hard_observed=model_config["hard_observed"],
    ).to(device)
    objective = TaskAwareRFSRLoss(spectral_mode="savaux").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    _write_json(
        run_dir / "config.json",
        {
            **vars(args),
            "run_dir": str(run_dir),
            "device": str(device),
            "model_config": model_config,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "generator": (
                "valid random gr-lora PHY payload symbols; 1 MS/s multipath/"
                "CFO/STO/SFO/gain; high-rate AWGN; exact x=noisy[::4]"
            ),
        },
    )
    print(
        f"device={device} train_items={len(train_dataset)} "
        f"validation_items={len(validation_dataset)} "
        f"parameters={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    best_validation = math.inf
    stale_epochs = 0
    history: list[dict[str, object]] = []
    for epoch in range(int(args.epochs)):
        train_dataset.set_epoch(epoch)
        task_enabled = epoch >= int(args.waveform_warmup_epochs)
        train_metrics = _run_epoch(
            model=model,
            objective=objective,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            concentration_weight=(args.concentration_weight if task_enabled else 0.0),
            margin_weight=(args.margin_weight if task_enabled else 0.0),
            gradient_clip=args.gradient_clip,
            supervise_observed=args.soft_observed,
        )
        validation_metrics = _run_epoch(
            model=model,
            objective=objective,
            loader=validation_loader,
            device=device,
            optimizer=None,
            scaler=scaler,
            use_amp=use_amp,
            concentration_weight=args.concentration_weight,
            margin_weight=args.margin_weight,
            gradient_clip=args.gradient_clip,
            supervise_observed=args.soft_observed,
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
                    "stage": "random-payload physical-channel pretraining",
                    "input_rate_hz": 250_000,
                    "output_rate_hz": 1_000_000,
                    "normalization": "input complex RMS; same scale for target",
                    "hard_observed_polyphase": not args.soft_observed,
                    "waveform_loss": (
                        "complex Charbonnier against clean channel target, "
                        + (
                            "all four phases"
                            if args.soft_observed
                            else "masked on noisy observed phase zero"
                        )
                    ),
                    "spectral_mode": "exact Savaux Eq.36/37 coherent",
                    "concentration_weight": float(args.concentration_weight),
                    "margin_weight": float(args.margin_weight),
                },
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
