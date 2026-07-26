"""Raw complex64 IQ file operations and power helpers.

中文说明：这一层只关心 raw complex64 IQ 文件本身，包括 memmap 读取、
分块功率估计、分块加 AWGN 写出。它不依赖 GNU Radio。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import math
import os

import numpy as np

from .constants import COMPLEX64_BYTES


def load_complex64_memmap(path: Path) -> np.memmap:
    """Open a raw complex64 IQ file as a NumPy memmap."""
    size_bytes = path.stat().st_size
    # GNU Radio gr_complex / numpy.complex64 每个采样点 8 字节，文件大小必须整除。
    if size_bytes % COMPLEX64_BYTES != 0:
        raise ValueError(
            f"{path} size {size_bytes} is not divisible by {COMPLEX64_BYTES}; "
            "expected raw complex64 IQ samples."
        )
    return np.memmap(path, dtype=np.complex64, mode="r")


def iter_chunks(
    samples: np.ndarray,
    chunk_samples: int,
    sample_limit: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield chunks from a large IQ array without copying the whole capture."""
    # IQ 捕获可能几百 MB，后续功率估计和写文件都按块走，避免一次性复制大数组。
    limit = samples.size if sample_limit is None else min(samples.size, int(sample_limit))
    for start in range(0, limit, int(chunk_samples)):
        stop = min(limit, start + int(chunk_samples))
        yield start, samples[start:stop]


def mean_power(samples: np.ndarray) -> float:
    """Mean E[|x|^2] for complex IQ samples."""
    if samples.size == 0:
        raise ValueError("Cannot estimate power from an empty sample array.")
    real = samples.real.astype(np.float64, copy=False)
    imag = samples.imag.astype(np.float64, copy=False)
    # 复 IQ 功率按 I^2 + Q^2 算；float64 累加能减少长文件上的数值误差。
    return float(np.mean(real * real + imag * imag, dtype=np.float64))


def sum_power(samples: np.ndarray) -> tuple[float, int]:
    """Return summed E[|x|^2] and sample count for chunk aggregation."""
    if samples.size == 0:
        return 0.0, 0
    real = samples.real.astype(np.float64, copy=False)
    imag = samples.imag.astype(np.float64, copy=False)
    return float(np.sum(real * real + imag * imag, dtype=np.float64)), int(samples.size)


def mean_power_chunked(
    samples: np.ndarray,
    chunk_samples: int,
    sample_limit: int | None = None,
) -> float:
    """Mean power computed in chunks for large captures."""
    total = 0.0
    count = 0
    for _, chunk in iter_chunks(samples, chunk_samples, sample_limit):
        current_sum, current_count = sum_power(chunk)
        total += current_sum
        count += current_count
    if count == 0:
        raise ValueError("Cannot estimate power from zero samples.")
    return float(total / count)


def estimate_block_powers(
    samples: np.ndarray,
    block_samples: int,
    sample_limit: int | None = None,
) -> np.ndarray:
    """Estimate mean power per fixed-size block."""
    limit = samples.size if sample_limit is None else min(samples.size, int(sample_limit))
    if limit <= 0:
        raise ValueError("Cannot estimate block powers from zero samples.")
    powers = []
    for start in range(0, limit, int(block_samples)):
        stop = min(limit, start + int(block_samples))
        powers.append(mean_power(samples[start:stop]))
    return np.asarray(powers, dtype=np.float64)


def normalize_ranges(ranges: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    """Clip and merge sample ranges."""
    normalized = []
    for start, end in sorted(ranges):
        start = max(0, min(int(start), int(limit)))
        end = max(start, min(int(end), int(limit)))
        if end <= start:
            continue
        if normalized and start <= normalized[-1][1]:
            normalized[-1] = (normalized[-1][0], max(normalized[-1][1], end))
        else:
            normalized.append((start, end))
    return normalized


def mean_power_outside_ranges(
    samples: np.ndarray,
    ranges: list[tuple[int, int]],
    sample_limit: int | None = None,
) -> tuple[float, int]:
    """Mean power outside known packet ranges."""
    limit = samples.size if sample_limit is None else min(samples.size, int(sample_limit))
    ranges = normalize_ranges(ranges, limit)
    # packet 模式下，这里估计“包外”已有底噪，用于从包内功率里扣除。
    total = 0.0
    count = 0
    cursor = 0
    for start, end in ranges:
        if start > cursor:
            current_sum, current_count = sum_power(samples[cursor:start])
            total += current_sum
            count += current_count
        cursor = max(cursor, end)
    if cursor < limit:
        current_sum, current_count = sum_power(samples[cursor:limit])
        total += current_sum
        count += current_count
    if count == 0:
        return float("nan"), 0
    return float(total / count), int(count)


def generate_noisy_file(
    samples: np.ndarray,
    out_path: Path,
    add_noise_power: float,
    seed: int,
    chunk_samples: int,
    sample_limit: int | None,
    overwrite: bool,
) -> None:
    """Stream an AWGN-added IQ file to disk."""
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists; pass --overwrite to replace it.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件，全部成功后再原子替换，避免中途中断留下半个 bin。
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    rng = np.random.default_rng(seed)
    # 复高斯噪声总功率为 E[|n|^2]，I/Q 两路各分到一半方差。
    sigma = math.sqrt(add_noise_power / 2.0) if add_noise_power > 0.0 else 0.0

    with tmp_path.open("wb") as handle:
        for _, chunk in iter_chunks(samples, chunk_samples, sample_limit):
            iq = np.asarray(chunk, dtype=np.complex64)
            if sigma > 0.0:
                noise_i = rng.normal(0.0, sigma, size=iq.size).astype(np.float32)
                noise_q = rng.normal(0.0, sigma, size=iq.size).astype(np.float32)
                iq = (iq + (noise_i + 1j * noise_q)).astype(np.complex64, copy=False)
            iq.tofile(handle)

    os.replace(tmp_path, out_path)


@dataclass
class IqCapture:
    """Object wrapper around one raw complex64 capture.

    把 path 和 samples 绑在一起，runner 里就不用到处传裸数组。
    """

    path: Path
    samples: np.ndarray

    @classmethod
    def open(cls, path: Path) -> "IqCapture":
        return cls(path=path, samples=load_complex64_memmap(path))

    def close(self) -> None:
        """Release the underlying memmap file handle on Windows."""
        mmap_handle = getattr(self.samples, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()

    def __enter__(self) -> "IqCapture":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Windows 下 memmap 会持有文件句柄，显式关闭后临时文件/输出目录才能立即删除。
        self.close()

    def processed_sample_count(self, sample_limit: int | None) -> int:
        return self.samples.size if sample_limit is None else min(self.samples.size, int(sample_limit))

    def mean_power(self, chunk_samples: int, sample_limit: int | None = None) -> float:
        return mean_power_chunked(self.samples, chunk_samples, sample_limit)

    def write_noisy(
        self,
        out_path: Path,
        add_noise_power: float,
        seed: int,
        chunk_samples: int,
        sample_limit: int | None,
        overwrite: bool,
    ) -> None:
        generate_noisy_file(
            self.samples,
            out_path,
            add_noise_power,
            seed,
            chunk_samples,
            sample_limit,
            overwrite,
        )
