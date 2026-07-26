#!/usr/bin/env python3
"""根据候选诊断结果分析 Savaux Top-K 的 oracle 提升上限。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_candidate_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_rows(rows: list[dict[str, str]]) -> dict[tuple[Any, ...], list[dict[str, str]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["snr_db"]),
            int(row["seed"]),
            int(row["packet_index"]),
            int(row["payload_symbol_index"]),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def _summarize(groups: dict[tuple[Any, ...], list[dict[str, str]]], top_ks: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, predicate in (
        ("all", lambda _key: True),
        *(
            (f"dataset={dataset}", lambda key, dataset=dataset: str(key[0]) == dataset)
            for dataset in sorted({str(key[0]) for key in groups})
        ),
        *(
            (f"snr={snr}", lambda key, snr=snr: str(key[1]) == snr)
            for snr in sorted({str(key[1]) for key in groups}, key=float)
        ),
    ):
        keys = [key for key in groups if predicate(key)]
        if not keys:
            continue
        savaux_err = 0
        missing_gt = 0
        oracle_err = {int(k): 0 for k in top_ks}
        fixable = {int(k): 0 for k in top_ks}
        for key in keys:
            rows = groups[key]
            savaux = next((row for row in rows if int(row["is_savaux"])), rows[0])
            gt = next((row for row in rows if int(row["is_gt"])), None)
            savaux_ok = bool(int(savaux["is_gt"]))
            if not savaux_ok:
                savaux_err += 1
            gt_rank = int(gt["candidate_rank"]) if gt is not None else 10**9
            if gt is None:
                missing_gt += 1
            for top_k in top_ks:
                if gt_rank > int(top_k):
                    oracle_err[int(top_k)] += 1
                elif not savaux_ok:
                    fixable[int(top_k)] += 1
        row: dict[str, Any] = {
            "group": label,
            "symbol_count": int(len(keys)),
            "savaux_err": int(savaux_err),
            "savaux_ser": float(savaux_err / max(1, len(keys))),
            "missing_gt_from_candidate_file": int(missing_gt),
        }
        for top_k in top_ks:
            k = int(top_k)
            row[f"top{k}_oracle_err"] = int(oracle_err[k])
            row[f"top{k}_oracle_ser"] = float(oracle_err[k] / max(1, len(keys)))
            row[f"top{k}_savaux_errors_fixable"] = int(fixable[k])
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_metrics", type=Path)
    parser.add_argument("--top-ks", nargs="+", type=int, default=[1, 2, 3, 4, 8, 16])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_path = Path(args.candidate_metrics).resolve()
    rows = _load_candidate_rows(candidate_path)
    summary = _summarize(_group_rows(rows), [int(k) for k in args.top_ks])
    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else candidate_path.parent / "topk_oracle"
    )
    _write_csv(out_dir / "topk_oracle_summary.csv", summary)
    (out_dir / "topk_oracle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in summary:
        print(
            f"{row['group']}: savaux={row['savaux_err']}/{row['symbol_count']} "
            f"ser={row['savaux_ser']:.6f} "
            + " ".join(
                f"top{k}={row[f'top{k}_oracle_err']}/{row['symbol_count']}({row[f'top{k}_oracle_ser']:.6f})"
                for k in args.top_ks
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
