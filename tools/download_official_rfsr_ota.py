#!/usr/bin/env python3
"""Download a verified subset or all of the official RFSR-OTA Dataverse files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
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
COMPRESSED_SIDECAR_SCHEMA = "rfsr-ota-zstd-v1"
OTA_PACKET_PATTERN = re.compile(r"^exp(?P<experiment>\d+)_(?P<packet>\d+)_rxg")


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
    parser.add_argument(
        "--ota-selection",
        choices=("sorted", "random", "packet-stratified"),
        default="sorted",
        help=(
            "How --ota-limit chooses OTA files (default: sorted). Use random to "
            "avoid filename-order bias, or packet-stratified to maximize distinct "
            "(experiment, packet) coverage before adding extra RX-gain variants."
        ),
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=20260803,
        help="Seed used by --ota-selection random (default: 20260803).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--download-backend",
        choices=("urllib", "curl"),
        default="urllib",
        help=(
            "HTTP client used for file payloads (default: urllib). The curl backend "
            "is more robust with the NTU Dataverse chunked-transfer endpoint."
        ),
    )
    parser.add_argument(
        "--curl-command",
        type=Path,
        help=(
            "Explicit curl executable for --download-backend curl. This can select "
            "an OpenSSL build when the Windows Schannel build is unreliable."
        ),
    )
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
    parser.add_argument(
        "--compress-zstd",
        action="store_true",
        help=(
            "After each cfile passes the official checksum, atomically compress it "
            "with zstd, test the frame, write a checksum sidecar, and remove the raw "
            "cfile. Existing valid raw cfiles are kept for immediate evaluation."
        ),
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=3,
        help="zstd compression level used by --compress-zstd (default: 3).",
    )
    parser.add_argument(
        "--compression-ratio-estimate",
        type=float,
        default=0.35,
        help=(
            "Conservative packed/raw ratio used only for the compressed-mode space "
            "preflight (default: 0.35)."
        ),
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
    if not 1 <= int(args.zstd_level) <= 19:
        parser.error("--zstd-level must be between 1 and 19")
    if not 0.0 < float(args.compression_ratio_estimate) <= 1.0:
        parser.error("--compression-ratio-estimate must be in (0, 1]")
    if bool(args.compress_zstd) and shutil.which("zstd") is None:
        parser.error("--compress-zstd requires a zstd executable on PATH")
    if args.curl_command is not None:
        curl_path = Path(args.curl_command).expanduser()
        if args.download_backend != "curl":
            parser.error("--curl-command requires --download-backend curl")
        if not curl_path.is_file():
            parser.error(f"--curl-command is not a file: {curl_path}")
    elif args.download_backend == "curl" and shutil.which("curl") is None:
        parser.error("--download-backend curl requires a curl executable on PATH")
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
    if bool(args.all_ota):
        selected_ota = ota
    elif args.ota_selection == "packet-stratified":
        generator = random.Random(int(args.selection_seed))
        groups: dict[tuple[str, ...], list[RemoteFile]] = {}
        for item in ota:
            match = OTA_PACKET_PATTERN.match(item.relative_path.name)
            key = (
                (match.group("experiment"), match.group("packet"))
                if match is not None
                else (item.relative_path.as_posix(),)
            )
            groups.setdefault(key, []).append(item)
        limit = min(int(args.ota_limit), len(ota))
        grouped = list(groups.values())
        if limit < len(grouped):
            grouped = generator.sample(grouped, k=limit)
        selected_ota = [generator.choice(group) for group in grouped]
        selected_set = set(selected_ota)
        remaining = [item for item in ota if item not in selected_set]
        selected_ota.extend(
            generator.sample(remaining, k=limit - len(selected_ota))
        )
        selected_ota.sort(key=lambda item: item.relative_path.as_posix())
    elif args.ota_selection == "random":
        generator = random.Random(int(args.selection_seed))
        selected_ota = sorted(
            generator.sample(ota, k=min(int(args.ota_limit), len(ota))),
            key=lambda item: item.relative_path.as_posix(),
        )
    else:
        selected_ota = ota[: int(args.ota_limit)]
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


def _compressed_paths(destination: Path) -> tuple[Path, Path]:
    compressed = Path(f"{destination}.zst")
    sidecar = Path(f"{compressed}.meta.json")
    return compressed, sidecar


def _valid_compressed_file(destination: Path, remote: RemoteFile) -> bool:
    """Validate the atomic sidecar contract without decompressing the payload."""

    compressed, sidecar = _compressed_paths(destination)
    if not compressed.is_file() or not sidecar.is_file():
        return False
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    expected = {
        "schema": COMPRESSED_SIDECAR_SCHEMA,
        "relative_path": remote.relative_path.as_posix(),
        "original_size": remote.size,
        "original_checksum_type": remote.checksum_type,
        "original_checksum": remote.checksum,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return False
    try:
        compressed_size = int(metadata["compressed_size"])
    except (KeyError, TypeError, ValueError):
        return False
    return compressed_size > 0 and compressed.stat().st_size == compressed_size


def _logical_complete(root: Path, remote: RemoteFile) -> bool:
    destination = root / remote.relative_path
    return _valid_file(destination, remote) or _valid_compressed_file(
        destination, remote
    )


def compress_verified_file(
    remote: RemoteFile,
    root: Path,
    *,
    zstd_command: str,
    zstd_level: int,
) -> tuple[str, RemoteFile]:
    """Atomically compress one verified cfile before removing its raw source."""

    destination = root / remote.relative_path
    if destination.suffix.lower() != ".cfile":
        return "kept", remote
    if _valid_compressed_file(destination, remote):
        return "compressed-cached", remote
    if not _valid_file(destination, remote):
        raise RuntimeError(f"refusing to compress unverified file: {destination}")

    compressed, sidecar = _compressed_paths(destination)
    temporary = Path(f"{compressed}.part")
    sidecar_temporary = Path(f"{sidecar}.part")
    for stale in (temporary, sidecar_temporary):
        if stale.exists():
            stale.unlink()

    command = [
        zstd_command,
        f"-{int(zstd_level)}",
        "-T0",
        "-q",
        "-f",
        str(destination),
        "-o",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        subprocess.run(
            [zstd_command, "-t", "-q", str(temporary)],
            check=True,
        )
        compressed_size = temporary.stat().st_size
        metadata = {
            "schema": COMPRESSED_SIDECAR_SCHEMA,
            "relative_path": remote.relative_path.as_posix(),
            "original_size": remote.size,
            "original_checksum_type": remote.checksum_type,
            "original_checksum": remote.checksum,
            "compressed_size": compressed_size,
            "compressed_sha256": _hash_file(temporary, "sha256"),
            "zstd_level": int(zstd_level),
        }
        sidecar_temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(compressed)
        sidecar_temporary.replace(sidecar)
        destination.unlink()
    except Exception:
        for stale in (temporary, sidecar_temporary):
            if stale.exists():
                stale.unlink()
        raise
    return "compressed", remote


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


def _download_once_curl(
    remote: RemoteFile,
    destination: Path,
    timeout: float,
    curl_command: str,
) -> None:
    """Download with curl for Dataverse endpoints that truncate urllib streams."""

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    command = [
        curl_command,
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--http1.1",
        "--connect-timeout",
        str(min(30.0, float(timeout))),
        # Match urllib's inactivity-oriented timeout semantics. A hard
        # --max-time incorrectly aborts complete-but-slow Dataverse streams.
        "--speed-limit",
        "1",
        "--speed-time",
        str(max(1, int(float(timeout)))),
        "--keepalive-time",
        "30",
        "--tcp-nodelay",
        "--output",
        str(partial),
        ACCESS_URL_TEMPLATE.format(file_id=remote.file_id),
    ]
    subprocess.run(command, check=True)
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
    download_backend: str,
    curl_command: str | None,
) -> tuple[str, RemoteFile]:
    destination = root / remote.relative_path
    if _valid_file(destination, remote):
        return "cached", remote
    last_error: Exception | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            if download_backend == "curl":
                if curl_command is None:
                    raise RuntimeError("curl backend selected without a curl executable")
                _download_once_curl(
                    remote,
                    destination,
                    float(timeout),
                    curl_command,
                )
            else:
                _download_once(remote, destination, float(timeout))
            return "downloaded", remote
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            ValueError,
            subprocess.SubprocessError,
        ) as exc:
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
        item.size for item in selected if _logical_complete(root, item)
    )
    remaining_bytes = selected_bytes - complete_bytes
    free_bytes = shutil.disk_usage(root).free
    estimate_text = ""
    required_free_bytes = remaining_bytes
    if bool(args.compress_zstd):
        estimated_packed = int(
            remaining_bytes * float(args.compression_ratio_estimate)
        )
        largest_file = max((item.size for item in selected), default=0)
        # Compression is immediate, so only one raw file and its temporary frame
        # need to coexist beyond the accumulated packed result.
        required_free_bytes = estimated_packed + (2 * largest_file)
        estimate_text = (
            f" packed_estimate={_format_bytes(estimated_packed)}"
            f" peak_estimate={_format_bytes(required_free_bytes)}"
        )
    print(
        f"selected={len(selected)} total={_format_bytes(selected_bytes)} "
        f"complete={_format_bytes(complete_bytes)} "
        f"remaining={_format_bytes(remaining_bytes)} free={_format_bytes(free_bytes)}"
        f"{estimate_text}",
        flush=True,
    )
    if required_free_bytes > free_bytes and not bool(args.skip_space_check):
        raise SystemExit(
            "insufficient free space; enlarge the AutoDL data disk, reduce --ota-limit, "
            "or pass --skip-space-check only when another mount supplies the space"
        )
    if bool(args.dry_run):
        return

    pending = [item for item in selected if not _logical_complete(root, item)]
    failures: list[str] = []
    zstd_command = shutil.which("zstd") if bool(args.compress_zstd) else None
    curl_command: str | None = None
    if args.download_backend == "curl":
        curl_command = (
            str(Path(args.curl_command).expanduser().resolve())
            if args.curl_command is not None
            else shutil.which("curl")
        )
    with ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = {
            executor.submit(
                download_file,
                item,
                root,
                int(args.retries),
                float(args.timeout),
                str(args.download_backend),
                curl_command,
            ): item
            for item in pending
        }
        completed = 0
        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                status, _ = future.result()
            except Exception as exc:
                failures.append(f"{item.relative_path}: {exc}")
                print(f"[{completed}/{len(pending)}] FAILED {item.relative_path}: {exc}")
            else:
                if bool(args.compress_zstd) and item.relative_path.suffix.lower() == ".cfile":
                    try:
                        status, _ = compress_verified_file(
                            item,
                            root,
                            zstd_command=str(zstd_command),
                            zstd_level=int(args.zstd_level),
                        )
                    except Exception as exc:
                        failures.append(f"{item.relative_path}: compression failed: {exc}")
                        print(
                            f"[{completed}/{len(pending)}] FAILED compression "
                            f"{item.relative_path}: {exc}"
                        )
                        continue
                print(f"[{completed}/{len(pending)}] {status} {item.relative_path}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(f"{len(failures)} downloads failed; rerun to resume")
    logical_complete = sum(_logical_complete(root, item) for item in selected)
    print(f"verified {logical_complete}/{len(selected)} files under {root}")


if __name__ == "__main__":
    main()
