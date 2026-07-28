#!/usr/bin/env python3
"""Generate one cached pretraining sample and render its full-packet STFT."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy import signal


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RFSR_ROOT = REPOSITORY_ROOT / "third_party" / "rfsr"
if str(RFSR_ROOT) not in sys.path:
    sys.path.insert(0, str(RFSR_ROOT))

from rfsr.nn.dataset import ReferencePhyPretrainingDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one RF-SR pretraining sample and save full STFTs."
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "reference_phy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pretraining_sample_stft.png"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr-min-db", type=float, default=-22.0)
    parser.add_argument("--snr-max-db", type=float, default=10.0)
    parser.add_argument("--cfo-min-hz", type=float, default=-5_000.0)
    parser.add_argument("--cfo-max-hz", type=float, default=5_000.0)
    parser.add_argument(
        "--with-sto",
        action="store_true",
        help="Include the default initial-STO and SFO-slope perturbations.",
    )
    parser.add_argument("--dynamic-range-db", type=float, default=65.0)
    return parser.parse_args()


def tensor_to_complex(tensor) -> np.ndarray:
    return np.asarray(tensor[0].numpy() + 1j * tensor[1].numpy())


def stft_image(
    samples: np.ndarray,
    sample_rate_hz: float,
    window_samples: int,
    nfft: int,
    dynamic_range_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, spectrum = signal.stft(
        samples,
        fs=sample_rate_hz,
        window="hann",
        nperseg=window_samples,
        noverlap=0,
        nfft=nfft,
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(np.fft.fftshift(spectrum, axes=0))
    peak = float(np.max(magnitude))
    floor = peak * 10.0 ** (-dynamic_range_db / 20.0)
    relative_db = 20.0 * np.log10(np.maximum(magnitude, floor) / peak)
    return (
        times,
        np.fft.fftshift(frequencies),
        relative_db,
    )


def symbol_boundaries_seconds(config: dict[str, object], sample_count: int) -> list[float]:
    sample_rate_hz = float(config["fs"])
    symbol_seconds = (1 << int(config["SF"])) / float(config["BW"])
    packet_start = int(config["leading_silence_samples"]) / sample_rate_hz
    packet_end = (
        sample_count - int(config["trailing_silence_samples"])
    ) / sample_rate_hz

    boundaries = [packet_start]
    cursor = packet_start
    for _ in range(int(config["preamble_bits"]) + 4):
        cursor += symbol_seconds
        boundaries.append(cursor)
    cursor += symbol_seconds / 4.0
    boundaries.append(cursor)
    while cursor + symbol_seconds < packet_end:
        cursor += symbol_seconds
        boundaries.append(cursor)
    return boundaries


def draw_stft(
    axis,
    samples: np.ndarray,
    sample_rate_hz: float,
    window_samples: int,
    nfft: int,
    dynamic_range_db: float,
    boundaries_seconds: list[float],
    title: str,
    bandwidth_hz: float,
):
    times, frequencies, image = stft_image(
        samples,
        sample_rate_hz,
        window_samples,
        nfft,
        dynamic_range_db,
    )
    mesh = axis.imshow(
        image,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=-dynamic_range_db,
        vmax=0.0,
        extent=(times[0], times[-1], frequencies[0] / 1_000.0, frequencies[-1] / 1_000.0),
    )
    for boundary in boundaries_seconds:
        axis.axvline(boundary, color="white", linewidth=0.35, alpha=0.35)
    axis.set_ylim(-0.65 * bandwidth_hz / 1_000.0, 0.65 * bandwidth_hz / 1_000.0)
    axis.set_ylabel("Frequency (kHz)")
    axis.set_title(title, fontsize=11)
    return mesh


def main() -> int:
    args = parse_args()
    if args.snr_min_db > args.snr_max_db:
        raise ValueError("--snr-min-db must be <= --snr-max-db")
    if args.cfo_min_hz > args.cfo_max_hz:
        raise ValueError("--cfo-min-hz must be <= --cfo-max-hz")
    if args.dynamic_range_db <= 0.0:
        raise ValueError("--dynamic-range-db must be positive")

    dataset = ReferencePhyPretrainingDataset(
        reference_root=args.reference_root,
        oversampling=4,
        size=1,
        seed=args.seed,
        snr_range=(args.snr_min_db, args.snr_max_db),
        cfo_range_hz=(args.cfo_min_hz, args.cfo_max_hz),
        sto_enabled=bool(args.with_sto),
    )
    x_tensor, y_tensor, snr_tensor = dataset[0]
    x = tensor_to_complex(x_tensor)
    y = tensor_to_complex(y_tensor)
    config = dataset.raw_phy_config
    boundaries = symbol_boundaries_seconds(config, y.size)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(18, 9),
        sharex=True,
        constrained_layout=True,
    )
    bandwidth_hz = float(config["BW"])
    draw_stft(
        axes[0],
        y,
        float(config["fs"]),
        window_samples=512,
        nfft=1024,
        dynamic_range_db=float(args.dynamic_range_db),
        boundaries_seconds=boundaries,
        title="Target y: ideal 1 MS/s LoRa reference",
        bandwidth_hz=bandwidth_hz,
    )
    mesh = draw_stft(
        axes[1],
        x,
        float(config["fs"]) / dataset.OSF,
        window_samples=128,
        nfft=256,
        dynamic_range_db=float(args.dynamic_range_db),
        boundaries_seconds=boundaries,
        title="Input x: 250 kS/s downsampled, impaired and noisy",
        bandwidth_hz=bandwidth_hz,
    )
    axes[1].set_xlabel("Packet time (s)")
    figure.colorbar(mesh, ax=axes, pad=0.01, label="Relative magnitude (dB)")
    figure.suptitle(
        "Cached RF-SR pretraining sample | "
        f"SNR={float(snr_tensor.item()):.2f} dB | "
        f"CFO={dataset.last_cfo_hz:.1f} Hz | "
        f"STO={dataset.last_initial_sto_chips:.3f} chip | "
        f"SFO slope={dataset.last_sto_slope_chips_per_symbol:.4f} chip/symbol",
        fontsize=13,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
