"""系统层共用的噪声与背景 FFT-bin 选择工具。"""

from __future__ import annotations

import numpy as np


def select_background_bins(power: np.ndarray, exclude_top: int, guard_bins: int) -> np.ndarray:
    """选择用于估计噪声协方差的背景 FFT bins。

    先排除功率最高的 ``exclude_top`` 个候选，再循环排除每个候选左右
    ``guard_bins`` 个邻居，避免主峰及其谱泄漏进入背景协方差。
    """

    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    exclude_count = min(max(0, int(exclude_top)), int(values.size))
    mask = np.ones(values.size, dtype=bool)
    if exclude_count > 0:
        top = np.argpartition(values, -exclude_count)[-exclude_count:]
        for raw_bin in top:
            for delta in range(-int(guard_bins), int(guard_bins) + 1):
                mask[(int(raw_bin) + delta) % values.size] = False
    return np.flatnonzero(mask).astype(np.int64)


__all__ = ["select_background_bins"]
