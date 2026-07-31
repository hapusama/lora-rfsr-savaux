#!/usr/bin/env python3
"""Download a verified subset or all of the official RFSR-OTA Dataverse files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
from typing import Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PERSISTENT_ID = "doi:10.21979/N9/C6ABM3"
DATAVERSE_ORIGIN = "https://researchdata.ntu.edu.sg"
DATASET_API_URL = (
    f"{DATAVERSE_ORIGIN}/api/datasets/:persistentId/"
    f"?persistentId={PERSISTENT_ID}"
)
ACCESS_URL_TEMPLATE = f"{DATAVERSE_ORIGIN}/api/access/datafile/{{file_id}}"
USER_AGENT = "lora-rfsr-savaux-official-downloader/1.0"


class RemoteFile(NamedTuple):
    file_id: int
    relative_path: Path
    size: int
    checksum_type: str
    checksum: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/rfsr-ota-official"),
        help="Data-disk destination. Do not place the full dataset on the system disk.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--ota-limit",
        type=int,
        default=100,
        help="Number of OTA cfiles to download after path sorting (default: 100).",
    )
    selection.add_argument(
        "--all-ota",
        action="store_true",
        help="Download every OTA cfile (about 315 GiB).",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Also download the 100 high-rate reference files (about 3.3 GiB).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Refetch the Dataverse manifest instead of using the local cache.",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        help="Use an existing Dataverse API JSON file; useful on an offline node.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected size and available space without downloading files.",
    )
    parser.add_argument(
        "--skip-space-check",
        action="store_true",
        help="Allow a download even when current free space is below the selection size.",
    )
    args = parser.parse_args()
    if args.ota_limit is not None and int(args.ota_limit) < 0:
        parser.error("--ota-limit cannot be negative")
    if int(args.workers) <= 0:
        parser.error("--workers must be positive")
    if int(args.retries) <= 0:
        parser.error("--retries must be positive")
    if float(args.timeout) <= 0.0:
        parser.error("--timeout must be positive")
    return args


def _format_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=float(timeout)) as response:
        return json.load(response)


def load_manifest(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    if args.manifest_file is not None:
        path = Path(args.manifest_file).expanduser().resolve()
        return json.loads(path.read_text(encoding="utf-8"))

    cache = root / "dataverse_manifest.json"
    if cache.exists() and not bool(args.refresh_manifest):
        return json.loads(cache.read_text(encoding="utf-8"))

    print(f"fetching manifest: {DATASET_API_URL}", flush=True)
    manifest = _fetch_json(DATASET_API_URL, float(args.timeout))
    temporary = cache.with_suffix(cache.suffix + ".part")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache)
    return manifest


def _safe_relative_path(directory: str, filename: str) -> Path:
    pure = PurePosixPath(directory) / filename if directory else PurePosixPath(filename)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe Dataverse path: {pure}")
    return Path(*pure.parts)


def manifest_files(manifest: dict[str, Any]) -> list[RemoteFile]:
    if manifest.get("status") != "OK":
        raise ValueError(f"Dataverse API did not return OK: {manifest.get('status')}")
    entries = manifest["data"]["latestVersion"]["files"]
    files: list[RemoteFile] = []
    for entry in entries:
        data = entry["dataFile"]
        checksum = data.get("checksum", {})
        files.append(
            RemoteFile(
                file_id=int(data["id"]),
                relative_path=_safe_relative_path(
                    str(entry.get("directoryLabel", "")),
                    str(data["filename"]),
                ),
                size=int(data["filesize"]),
                checksum_type=str(checksum.get("type", "")).upper(),
                checksum=str(checksum.get("value", "")).lower(),
            )
        )
    return sorted(files, key=lambda item: item.relative_path.as_posix())


def select_files(files: list[RemoteFile], args: argparse.Namespace) -> list[RemoteFile]:
    root_and_metadata = [
        item
        for item in files
        if len(item.relative_path.parts) == 1
        or item.relative_path.parts[0] == "metadata"
    ]
    ota = [item for item in files if item.relative_path.parts[0] == "ota"]
    reference = [
        item for item in files if item.relative_path.parts[0] == "reference"
    ]
    selected_ota = ota if bool(args.all_ota) else ota[: int(args.ota_limit)]
    selected = root_and_metadata + selected_ota
    if bool(args.include_reference):
        selected += reference
    return sorted(selected, key=lambda item: item.relative_path.as_posix())


def _hash_file(path: Path, algorithm: str) -> str:
    name = algorithm.lower().replace("-", "")
    digest = hashlib.new(name)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def _valid_file(path: Path, remote: RemoteFile) -> bool:
    if not path.is_file() or path.stat().st_size != remote.size:
        return False
    if not remote.checksum_type or not remote.checksum:
        return True
    return _hash_file(path, remote.checksum_type) == remote.checksum


def _download_once(
    remote: RemoteFile,
    destination: Path,
    timeout: float,
) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > remote.size:
        partial.unlink()
        offset = 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(
        ACCESS_URL_TEMPLATE.format(file_id=remote.file_id),
        headers=headers,
    )
    try:
        response = urlopen(request, timeout=float(timeout))
    except HTTPError as exc:
        if exc.code == 416 and offset == remote.size:
            response = None
        else:
            raise

    if response is not None:
        with response:
            append = bool(offset and getattr(response, "status", 200) == 206)
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)

    if partial.stat().st_size != remote.size:
        raise IOError(
            f"size mismatch for {remote.relative_path}: "
            f"expected {remote.size}, got {partial.stat().st_size}"
        )
    if remote.checksum_type and remote.checksum:
        actual = _hash_file(partial, remote.checksum_type)
        if actual != remote.checksum:
            partial.unlink()
            raise IOError(
                f"{remote.checksum_type} mismatch for {remote.relative_path}: "
                f"expected {remote.checksum}, got {actual}"
            )
    partial.replace(destination)


def download_file(
    remote: RemoteFile,
    root: Path,
    retries: int,
    timeout: float,
) -> tuple[str, RemoteFile]:
    destination = root / remote.relative_path
    if _valid_file(destination, remote):
        return "cached", remote
    last_error: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            _download_once(remote, destination, float(timeout))
            return "downloaded", remote
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt < int(retries):
                time.sleep(min(30.0, 2.0 ** min(attempt - 1, 5)))
    raise RuntimeError(
        f"failed {remote.relative_path} after {retries} attempts: {last_error}"
    )


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args, root)
    selected = select_files(manifest_files(manifest), args)
    selected_bytes = sum(item.size for item in selected)
    complete_bytes = sum(
        item.size for item in selected if _valid_file(root / item.relative_path, item)
    )
    remaining_bytes = selected_bytes - complete_bytes
    free_bytes = shutil.disk_usage(root).free
    print(
        f"selected={len(selected)} total={_format_bytes(selected_bytes)} "
        f"complete={_format_bytes(complete_bytes)} "
        f"remaining={_format_bytes(remaining_bytes)} free={_format_bytes(free_bytes)}",
        flush=True,
    )
    if remaining_bytes > free_bytes and not bool(args.skip_space_check):
        raise SystemExit(
            "insufficient free space; enlarge the AutoDL data disk, reduce --ota-limit, "
            "or pass --skip-space-check only when another mount supplies the space"
        )
    if bool(args.dry_run):
        return

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {
            executor.submit(
                download_file,
                item,
                root,
                int(args.retries),
                float(args.timeout),
            ): item
            for item in selected
        }
        completed = 0
        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                status, _ = future.result()
            except Exception as exc:
                failures.append(f"{item.relative_path}: {exc}")
                print(f"[{completed}/{len(selected)}] FAILED {item.relative_path}: {exc}")
            else:
                print(f"[{completed}/{len(selected)}] {status} {item.relative_path}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(f"{len(failures)} downloads failed; rerun to resume")
    print(f"verified {len(selected)} files under {root}")


if __name__ == "__main__":
    main()
