#!/usr/bin/env python3
"""Verify that a copied lora-rfsr-savaux directory is runnable by itself."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPOSITORY_ROOT / "data"
DEFAULT_DATASET_ROOT = DATA_ROOT / "reference_phy" / "rfsr_db"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_local(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"runtime manifest path must be relative: {value}")
    resolved = (root / path).resolve()
    if not inside(resolved, root):
        raise ValueError(f"runtime manifest path escapes {root}: {value}")
    return resolved


def discover_capture() -> Path | None:
    captures = sorted((DATA_ROOT / "raw" / "ota").glob("rxcap_*.cfile"))
    return captures[0] if len(captures) == 1 else None


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def check_python_modules(errors: list[str], modules: tuple[str, ...]) -> None:
    for module in modules:
        add_error(
            errors,
            importlib.util.find_spec(module) is not None,
            f"Python package is not installed: {module}",
        )


def check_base(errors: list[str], dataset_root: Path) -> dict[str, int]:
    required = (
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "tools" / "build_rfsr_ota_dataset.py",
        REPOSITORY_ROOT
        / "third_party"
        / "rfsr"
        / "rfsr"
        / "nn"
        / "ota_dataset.py",
        DATA_ROOT / "raw" / "packet_reference.txt",
    )
    for path in required:
        add_error(errors, path.is_file(), f"missing repository file: {path}")

    catalog_path = dataset_root / "manifests" / "reference_catalog.csv"
    add_error(
        errors,
        catalog_path.is_file(),
        f"missing reference catalog: {catalog_path}",
    )
    reference_count = 0
    if catalog_path.is_file():
        rows = read_csv(catalog_path)
        reference_count = len(rows)
        add_error(errors, bool(rows), f"empty reference catalog: {catalog_path}")
        for row in rows:
            try:
                reference = resolve_local(
                    REPOSITORY_ROOT, row["source_reference_path"]
                )
                metadata = resolve_local(
                    REPOSITORY_ROOT, row["reference_metadata_path"]
                )
            except (KeyError, ValueError) as exc:
                errors.append(f"invalid reference catalog row: {exc}")
                continue
            add_error(
                errors,
                reference.is_file(),
                f"missing reference IQ: {reference}",
            )
            add_error(
                errors,
                metadata.is_file(),
                f"missing reference metadata: {metadata}",
            )

    check_python_modules(errors, ("numpy", "scipy"))
    return {"references": reference_count}


def check_preprocess(
    errors: list[str],
    dataset_root: Path,
    capture: Path | None,
) -> dict[str, int | str]:
    if capture is None:
        errors.append(
            "preprocess mode needs --capture when data/raw/ota does not "
            "contain exactly one capture"
        )
        return {"detections": 0}
    capture = capture.expanduser().resolve()
    add_error(
        errors,
        inside(capture, DATA_ROOT),
        f"capture must be inside this repository's data/: {capture}",
    )
    add_error(errors, capture.is_file(), f"missing capture: {capture}")
    add_error(
        errors,
        capture.suffix == ".cfile" and capture.name.startswith("rxcap_"),
        f"capture does not use the canonical name: {capture.name}",
    )
    sidecar_path = Path(str(capture) + ".json")
    add_error(
        errors, sidecar_path.is_file(), f"missing capture sidecar: {sidecar_path}"
    )
    if not capture.is_file() or not sidecar_path.is_file():
        return {"capture": str(capture), "detections": 0}
    add_error(
        errors,
        capture.stat().st_size % 8 == 0,
        f"capture byte size is not complex64-aligned: {capture}",
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    add_error(
        errors,
        sidecar.get("schema") == "lora-rfsr-usrp-capture",
        f"unsupported capture sidecar schema: {sidecar_path}",
    )
    descriptor = sidecar.get("capture", {})
    try:
        capture_uid = (
            f"exp{int(descriptor['experiment_id']):03d}_"
            f"sess{int(descriptor['session_id']):03d}_"
            f"run{int(descriptor['run_id']):03d}"
        )
    except (KeyError, TypeError, ValueError):
        errors.append(f"incomplete capture descriptor: {sidecar_path}")
        return {"capture": str(capture), "detections": 0}
    detections_path = (
        dataset_root
        / "manifests"
        / "captures"
        / capture_uid
        / "detections.csv"
    )
    add_error(
        errors,
        detections_path.is_file(),
        "missing local detections.csv; run the GNU Radio detect stage on the "
        f"acquisition machine before copying to the server: {detections_path}",
    )
    detections = read_csv(detections_path) if detections_path.is_file() else []
    add_error(
        errors,
        bool(detections),
        f"detections manifest is empty: {detections_path}",
    )
    return {
        "capture": str(capture.relative_to(REPOSITORY_ROOT)),
        "capture_bytes": capture.stat().st_size,
        "detections": len(detections),
    }


def check_train(errors: list[str], dataset_root: Path) -> dict[str, int]:
    check_python_modules(errors, ("torch",))
    views_path = dataset_root / "manifests" / "views.csv"
    add_error(errors, views_path.is_file(), f"missing OTA views: {views_path}")
    views = read_csv(views_path) if views_path.is_file() else []
    add_error(errors, bool(views), f"OTA views manifest is empty: {views_path}")
    physical_packets: set[str] = set()
    for row in views:
        physical_packets.add(row.get("split_group", ""))
        for field in ("ota_path", "reference_path"):
            try:
                path = resolve_local(dataset_root, row[field])
            except (KeyError, ValueError) as exc:
                errors.append(f"invalid {field} in views.csv: {exc}")
                continue
            add_error(errors, path.is_file(), f"missing training IQ: {path}")
    checkpoint = (
        REPOSITORY_ROOT
        / "third_party"
        / "rfsr"
        / "checkpoints"
        / "model_model0v0_bs5_osf4_ds250_lr0.001_wd1e-05.pth"
    )
    add_error(
        errors,
        checkpoint.is_file(),
        f"missing bundled pretrained checkpoint: {checkpoint}",
    )
    return {
        "views": len(views),
        "physical_packets": len(physical_packets - {""}),
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("base", "preprocess", "train"),
        default="base",
    )
    parser.add_argument("--capture", type=Path)
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not inside(dataset_root, DATA_ROOT):
        print(
            json.dumps(
                {
                    "ready": False,
                    "errors": [
                        f"dataset root must stay inside {DATA_ROOT}: "
                        f"{dataset_root}"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    errors: list[str] = []
    counts = check_base(errors, dataset_root)
    if args.mode == "preprocess":
        counts.update(
            check_preprocess(
                errors, dataset_root, args.capture or discover_capture()
            )
        )
    elif args.mode == "train":
        counts.update(check_train(errors, dataset_root))

    print(
        json.dumps(
            {
                "ready": not errors,
                "mode": args.mode,
                "repository_root": str(REPOSITORY_ROOT),
                "dataset_root": str(dataset_root),
                "counts": counts,
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
