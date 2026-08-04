"""Random-payload, physically consistent pretraining data for task-aware RF-SR."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset

from rfsr.PHY import encode_raw_phy_symbols


SF = 12
N_BINS = 1 << SF
BW_HZ = 125_000
OUTPUT_OS_FACTOR = 8
OUTPUT_SAMPLES_PER_SYMBOL = N_BINS * OUTPUT_OS_FACTOR
SUPER_RESOLUTION_FACTOR = 4


def _grlora_upchirp(symbol_id: int) -> np.ndarray:
    """Generate one modern gr-lora-compatible OSF=8 symbol."""

    value = int(symbol_id) % N_BINS
    n = np.arange(OUTPUT_SAMPLES_PER_SYMBOL, dtype=np.float64)
    fold = OUTPUT_SAMPLES_PER_SYMBOL - value * OUTPUT_OS_FACTOR
    slope = n * n / (2.0 * N_BINS * OUTPUT_OS_FACTOR**2)
    before = (value / N_BINS - 0.5) * n / OUTPUT_OS_FACTOR
    after = (value / N_BINS - 1.5) * n / OUTPUT_OS_FACTOR
    phase = np.where(n < fold, slope + before, slope + after)
    return np.exp(2j * np.pi * phase).astype(np.complex64)


def _linear_sample(values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Linearly sample a guarded three-symbol neighborhood."""

    left = np.floor(query).astype(np.int64)
    fraction = query - left
    left = np.clip(left, 0, values.size - 2)
    return (
        values[left] * (1.0 - fraction) + values[left + 1] * fraction
    ).astype(np.complex64)


class TaskAwareSyntheticSymbolDataset(Dataset):
    """Create valid random PHY symbols, channelize, then decimate.

    Targets are generated at 1 MS/s with multipath, CFO, fractional STO, SFO,
    gain and carrier phase. AWGN is added on that same high-rate time axis and
    the input is exactly ``noisy_high[::4]``. The clean loss masks phase zero,
    because the model deliberately hard-copies the noisy observed samples.
    """

    def __init__(
        self,
        *,
        item_count: int = 2_000,
        symbols_per_item: int = 4,
        payload_length: int = 20,
        seed: int = 42,
        snr_range_db: tuple[float, float] = (-24.0, 8.0),
        cfo_range_hz: tuple[float, float] = (-12_000.0, 12_000.0),
        sto_range_output_samples: tuple[float, float] = (-6.0, 6.0),
        sfo_range_ppm: tuple[float, float] = (-25.0, 25.0),
        maximum_multipath_delay: int = 8,
    ):
        self.item_count = int(item_count)
        self.symbols_per_item = int(symbols_per_item)
        self.payload_length = int(payload_length)
        self.seed = int(seed)
        self.snr_range_db = tuple(map(float, snr_range_db))
        self.cfo_range_hz = tuple(map(float, cfo_range_hz))
        self.sto_range_output_samples = tuple(
            map(float, sto_range_output_samples)
        )
        self.sfo_range_ppm = tuple(map(float, sfo_range_ppm))
        self.maximum_multipath_delay = int(maximum_multipath_delay)
        self.epoch = 0
        if self.item_count < 1 or not 1 <= self.symbols_per_item <= 8:
            raise ValueError("item_count must be positive and symbols_per_item 1..8")
        if not 1 <= self.payload_length <= 255:
            raise ValueError("payload_length must be in [1, 255]")
        for name, interval in (
            ("snr_range_db", self.snr_range_db),
            ("cfo_range_hz", self.cfo_range_hz),
            ("sto_range_output_samples", self.sto_range_output_samples),
            ("sfo_range_ppm", self.sfo_range_ppm),
        ):
            if interval[0] > interval[1]:
                raise ValueError(f"{name} minimum exceeds maximum")
        if self.maximum_multipath_delay < 0:
            raise ValueError("maximum_multipath_delay must be non-negative")

    def __len__(self) -> int:
        return self.item_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _channelized_symbol(
        self,
        symbol_ids: tuple[int, ...],
        symbol_index: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        current = int(symbol_ids[symbol_index])
        neighbors = (
            int(symbol_ids[(symbol_index - 1) % len(symbol_ids)]),
            current,
            int(symbol_ids[(symbol_index + 1) % len(symbol_ids)]),
        )
        source = np.concatenate([_grlora_upchirp(value) for value in neighbors])

        cfo_hz = float(rng.uniform(*self.cfo_range_hz))
        sto = float(rng.uniform(*self.sto_range_output_samples))
        sfo = float(rng.uniform(*self.sfo_range_ppm)) * 1e-6
        adc_phase = int(rng.integers(0, SUPER_RESOLUTION_FACTOR))
        output_index = np.arange(OUTPUT_SAMPLES_PER_SYMBOL, dtype=np.float64)
        query = (
            OUTPUT_SAMPLES_PER_SYMBOL
            + output_index
            + adc_phase
            - sto
            + sfo * (output_index - OUTPUT_SAMPLES_PER_SYMBOL / 2.0)
        )

        tap_count = int(rng.integers(1, 4))
        if self.maximum_multipath_delay == 0:
            delays = np.zeros(tap_count, dtype=np.int64)
        else:
            delays = np.concatenate(
                (
                    np.zeros(1, dtype=np.int64),
                    rng.integers(
                        1,
                        self.maximum_multipath_delay + 1,
                        size=tap_count - 1,
                        dtype=np.int64,
                    ),
                )
            )
        tap_scale = np.exp(-delays.astype(np.float64) / 3.0)
        taps = tap_scale * np.exp(2j * np.pi * rng.random(tap_count))
        taps /= math.sqrt(float(np.sum(np.abs(taps) ** 2)))
        clean = np.zeros(OUTPUT_SAMPLES_PER_SYMBOL, dtype=np.complex64)
        for delay, tap in zip(delays.tolist(), taps.tolist()):
            clean += np.complex64(tap) * _linear_sample(
                source, query - int(delay)
            )

        gain = 10.0 ** (float(rng.uniform(-6.0, 6.0)) / 20.0)
        carrier_phase = float(rng.uniform(-np.pi, np.pi))
        time_seconds = output_index / 1_000_000.0
        rotation = np.exp(
            1j * (2.0 * np.pi * cfo_hz * time_seconds + carrier_phase)
        ).astype(np.complex64)
        clean = np.asarray(clean * rotation * gain, dtype=np.complex64)

        snr_db = float(rng.uniform(*self.snr_range_db))
        signal_power = float(np.mean(np.abs(clean).astype(np.float64) ** 2))
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        component_std = math.sqrt(max(noise_power, 1e-20) / 2.0)
        noise = component_std * (
            rng.standard_normal(clean.size) + 1j * rng.standard_normal(clean.size)
        )
        noisy = np.asarray(clean + noise, dtype=np.complex64)
        correct_bin = int(round(current + cfo_hz * N_BINS / BW_HZ)) % N_BINS
        return noisy, clean, correct_bin, snr_db

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, int(index)])
        )
        payload = rng.integers(
            0, 256, size=self.payload_length, dtype=np.uint8
        ).tobytes()
        header, body = encode_raw_phy_symbols(
            payload, sf=SF, cr=4, enable_crc=1, ldro=True, crc_mode="grlora"
        )
        symbol_ids = tuple(int(value) for value in header + body)
        selected = rng.choice(
            len(symbol_ids), size=self.symbols_per_item, replace=False
        )

        x_rows: list[torch.Tensor] = []
        y_rows: list[torch.Tensor] = []
        correct_bins: list[int] = []
        snrs: list[float] = []
        for symbol_index in selected.tolist():
            noisy, clean, correct_bin, snr_db = self._channelized_symbol(
                symbol_ids, int(symbol_index), rng
            )
            low = np.asarray(noisy[::SUPER_RESOLUTION_FACTOR], dtype=np.complex64)
            x_rows.append(
                torch.from_numpy(np.stack((low.real, low.imag), axis=0).copy())
            )
            y_rows.append(
                torch.from_numpy(
                    np.stack((clean.real, clean.imag), axis=0).copy()
                )
            )
            correct_bins.append(correct_bin)
            snrs.append(snr_db)

        valid_mask = torch.ones(
            self.symbols_per_item, OUTPUT_SAMPLES_PER_SYMBOL, dtype=torch.float32
        )
        valid_mask[:, ::SUPER_RESOLUTION_FACTOR] = 0.0
        return {
            "x": torch.stack(x_rows),
            "y": torch.stack(y_rows),
            "correct_bins": torch.tensor(correct_bins, dtype=torch.long),
            "task_mask": torch.ones(self.symbols_per_item, dtype=torch.float32),
            "valid_mask": valid_mask,
            "snr_db": torch.tensor(snrs, dtype=torch.float32),
            "payload": payload,
        }


__all__ = ["TaskAwareSyntheticSymbolDataset"]
