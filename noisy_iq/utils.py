"""Small formatting, parsing, and serialization helpers.

中文说明：这里放无状态的小工具函数，避免 capture/reporting/power
之间互相复制格式化代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import re

import numpy as np


def parse_payload_hex(value: str) -> bytes:
    """Parse a payload hex string; spaces, commas and 0x prefixes are accepted."""
    text = str(value).strip().lower().replace("0x", "")
    compact = re.sub(r"[^0-9a-f]", "", text)
    if len(compact) % 2:
        raise ValueError(f"Payload hex string has an odd number of hex digits: {value!r}")
    return bytes.fromhex(compact)


def payload_bytes_to_text(payload: bytes) -> str:
    """Return a compact printable representation for JSON/CSV diagnostics."""
    return payload.decode("ascii", errors="backslashreplace")


def parse_int_auto(value: str) -> int:
    """Parse decimal or 0x-prefixed integer CLI values."""
    return int(str(value), 0)


def db10(value: float) -> float:
    """Convert a linear power ratio to dB."""
    if value <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(value)


def db_to_label(value_db: float) -> str:
    """Convert a dB value into a filename-friendly label."""
    text = f"{value_db:.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def format_db_value(value_db: float) -> str:
    """Format a dB value without hiding fine sweep steps."""
    return f"{float(value_db):.3f}".rstrip("0").rstrip(".")


def summarize_values(values: list[float]) -> dict[str, float | int]:
    """Summarize finite float values for metadata and summary CSV rows."""
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars and NaN/Inf values into strict JSON-safe values."""
    # JSON 标准不接受 NaN/Inf；metadata 写出前统一转成 null。
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def csv_value(value: Any) -> Any:
    """Flatten values that are awkward to write directly to summary CSV."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return f"{value:.9g}" if math.isfinite(value) else ""
    if isinstance(value, Path):
        return str(value)
    return value
