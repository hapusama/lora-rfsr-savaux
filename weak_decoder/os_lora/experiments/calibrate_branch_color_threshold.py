"""在无信号 AWGN 零假设下标定 LoRa branch-color 门限。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


THIS_FILE = Path(__file__).resolve()
WEAK_ROOT = THIS_FILE.parents[3]
if str(WEAK_ROOT) not in sys.path:
    sys.path.insert(0, str(WEAK_ROOT))

from weak_decoder.baselines.common import write_csv  # noqa: E402
from weak_decoder.os_lora.system.noise import select_background_bins as _background_bins  # noqa: E402
from weak_decoder.os_lora.system.nonuniform_sampling import (  # noqa: E402
    build_pattern_bank,
    lora_branch_color_mismatch,
    pattern_bank_spectra,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=int, default=10)
    parser.add_argument("--os-factor", type=int, default=4)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--false-alarm-rate", type=float, default=1e-3)
    parser.add_argument("--exclude-top", type=int, default=8)
    parser.add_argument("--exclude-guard-bins", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WEAK_ROOT / "data" / "os_lora" / "branch_color_awgn_null",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sf = int(args.sf)
    os_factor = int(args.os_factor)
    n_bins = 1 << sf
    trials = max(1, int(args.trials))
    false_alarm_rate = float(args.false_alarm_rate)
    if not 0.0 < false_alarm_rate < 1.0:
        raise ValueError("false-alarm-rate must lie in (0, 1)")
    rng = np.random.default_rng(int(args.seed))
    fixed_bank = build_pattern_bank(sf, os_factor, kind="fixed")
    q = np.arange(os_factor, dtype=np.float64)[:, None]
    k = np.arange(n_bins, dtype=np.float64)[None, :]
    undo_combining_phase = np.exp(2j * np.pi * q * k / float(n_bins * os_factor))
    values = np.empty(trials, dtype=np.float64)
    for trial in range(trials):
        noise = (
            rng.standard_normal(n_bins * os_factor)
            + 1j * rng.standard_normal(n_bins * os_factor)
        ).astype(np.complex64) / np.sqrt(2.0)
        aligned_branches = pattern_bank_spectra(noise, fixed_bank)
        savaux_power = np.abs(np.sum(aligned_branches, axis=0)).astype(np.float64) ** 2
        background = _background_bins(
            savaux_power,
            exclude_top=int(args.exclude_top),
            guard_bins=int(args.exclude_guard_bins),
        )
        paper_branches = aligned_branches * undo_combining_phase
        values[trial] = lora_branch_color_mismatch(paper_branches, background)

    quantile = 1.0 - false_alarm_rate
    threshold = float(np.quantile(values, quantile))
    summary = [
        {
            "sf": sf,
            "os_factor": os_factor,
            "trials": trials,
            "seed": int(args.seed),
            "false_alarm_rate": false_alarm_rate,
            "threshold": threshold,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
            "q999": float(np.quantile(values, 0.999)),
            "maximum": float(np.max(values)),
        }
    ]
    output_dir = Path(args.output_dir).resolve()
    write_csv(output_dir / "summary.csv", summary)
    write_csv(
        output_dir / "null_samples.csv",
        [{"trial": int(index), "branch_color_mismatch": float(value)} for index, value in enumerate(values)],
    )
    print(
        f"SF{sf} OSR{os_factor}: Pfa={false_alarm_rate:g} threshold={threshold:.6f} "
        f"q99={summary[0]['q99']:.6f} q999={summary[0]['q999']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
