#!/usr/bin/env python3
"""从连续 USRP IQ 构建可审计的 RF-SR OTA 训练数据集。

脚本刻意把三类证据分开，避免把“检测到信号”和“知道它是哪一个包”混为一谈：

* IQ 检测只负责确定 packet 在连续 cfile 中的采样点位置；
* CRC 正确的解码帧和发包周期只负责确定 packet 身份；
* UART ``packet_reference.txt`` 是理想 reference payload 的唯一真值来源。

完整流程为：

``init-capture -> catalog -> detect -> associate -> trim -> validate``。

数据目录只分为三层含义：

```
data/
├── raw/                         # 不可修改的原始证据
│   ├── packet_reference.txt     # STM32 串口 ground truth
│   └── ota/*.cfile              # USRP 连续 2 MS/s IQ
├── reference_phy/               # 理想 PHY reference 的原始语料
│   ├── reference/*.cfile        # 1 MS/s 理想发射波形
│   ├── metadata/*.json
│   └── rfsr_db/                 # 本脚本生成的可训练数据库
│       ├── ota/*.cfile          # 每个物理包的两个 1 MS/s 接收相位
│       ├── reference/*.cfile    # 训练标签，通常是硬链接
│       ├── metadata/*.json      # 每个 OTA 文件的配对与裁剪证据
│       └── manifests/           # catalog、检测、关联、视图和校验表
└── ...
```

所有持久化输出都被限制在 ``lora-rfsr-savaux/data`` 内。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (REPO_ROOT / "data").resolve()

# 输入语料与最终数据库的默认位置。reference_phy 是“理想语料”，
# reference_phy/rfsr_db 是“真实 OTA 与理想语料配对后的训练数据库”。
DEFAULT_REFERENCE_ROOT = DATA_ROOT / "reference_phy"
DEFAULT_OUTPUT_ROOT = DEFAULT_REFERENCE_ROOT / "rfsr_db"
DEFAULT_UART_LOG = DATA_ROOT / "raw" / "packet_reference.txt"
DEFAULT_RAW_CAPTURE_DIR = DATA_ROOT / "raw" / "ota"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weak_decoder.rf_super_resolution.reference_phy import (  # noqa: E402
    parse_uart_reference_log,
)


CAPTURE_SCHEMA = "lora-rfsr-usrp-capture"
CAPTURE_SCHEMA_VERSION = 1
OTA_SCHEMA = "lora-rfsr-ota-view"
OTA_SCHEMA_VERSION = 1
COMPLEX_DTYPE = np.dtype("<c8")

# 五张 CSV 的字段顺序固定，便于不同实验批次直接拼接和审计。
# reference_catalog.csv：UART 真值 ID 到理想 reference 文件的映射。
CATALOG_FIELDS = (
    "reference_id",
    "uart_seq",
    "uart_round",
    "payload_id",
    "frame_hex",
    "app_payload_hex",
    "frame_bytes",
    "sf",
    "bandwidth_hz",
    "cr",
    "preamble_symbols",
    "sync_word",
    "phy_crc",
    "ldro",
    "sample_rate_hz",
    "samples_per_symbol",
    "leading_silence_samples",
    "trailing_silence_samples",
    "reference_samples",
    "header_symbols",
    "payload_symbols",
    "tx_period_ms",
    "source_reference_path",
    "dataset_reference_path",
    "reference_sha256",
    "reference_metadata_path",
    "uart_source_path",
    "uart_source_sha256",
)

# detections.csv：GNU Radio 在连续 IQ 中找到的包位置和解码结果。
DETECTION_FIELDS = (
    "detection_index",
    "frame_count",
    "start_sample_2m",
    "preamble_end_sample_2m",
    "estimated_packet_end_sample_2m",
    "header_ok",
    "crc_valid",
    "decoded_frame_hex",
    "decoded_payload_len",
    "grlora_snr_db",
    "cfo_hz",
    "sto",
    "sfo",
    "netid1",
    "netid2",
    "samples_per_symbol",
)

# packets.csv：把检测位置与 reference_id 关联后的最终包级判定。
PACKET_FIELDS = (
    "physical_packet_uid",
    "capture_uid",
    "capture_packet_index",
    "sequence_segment",
    "schedule_slot",
    "detection_index",
    "start_sample_2m",
    "predicted_start_sample_2m",
    "timing_residual_samples",
    "reference_id",
    "association_method",
    "association_confidence",
    "status",
    "crc_valid",
    "decoded_frame_hex",
    "catalog_frame_exact",
    "grlora_snr_db",
    "estimated_cfo_hz",
    "sto",
    "sfo",
    "previous_anchor_detection",
    "next_anchor_detection",
    "schedule_period_samples",
    "local_search_used",
    "local_search_score",
    "notes",
)

# views.csv：一个 1 MS/s OTA 文件可提供的四个 250 kS/s 抽取相位。
VIEW_FIELDS = (
    "view_id",
    "physical_packet_uid",
    "capture_uid",
    "capture_packet_index",
    "reference_id",
    "ota_path",
    "reference_path",
    "adc_phase_2m",
    "lowrate_phase_1m",
    "combined_decimation_phase_2m",
    "timing_offset_samples_1m",
    "input_sample_rate_hz",
    "input_samples",
    "target_sample_rate_hz",
    "target_samples",
    "split_group",
)


def utc_now() -> str:
    """返回带时区的 UTC 时间，用于写入可追溯 metadata。"""

    return datetime.now(timezone.utc).isoformat()


def ensure_inside_data(path: str | Path, *, label: str = "output path") -> Path:
    """确认持久化路径位于项目 ``data/`` 内，防止输出散落到仓库外。"""

    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(DATA_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must stay inside {DATA_ROOT}, got {resolved}."
        ) from exc
    return resolved


def display_path(path: str | Path) -> str:
    """项目内路径写成相对路径；项目外路径才保留绝对形式。"""

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: str | Path) -> str:
    """分块计算大文件 SHA-256，避免把数 GB IQ 一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_write_text(path: Path, text: str) -> None:
    """先写临时文件再原子替换，避免中断后留下半份 JSON/CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """以 UTF-8、可读缩进格式原子写入 JSON。"""

    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """按固定字段顺序原子写入 CSV，缺失值统一写为空字符串。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if value is None else value
                    for key, value in row.items()
                    if key in fieldnames
                }
            )
    os.replace(temporary, path)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """读取 CSV，并把每行返回为字段名到字符串的字典。"""

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    """兼容 CSV 中常见的 0/1、true/false、pass/valid 布尔写法。"""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "valid"}


def finite_or_blank(value: Any) -> float | str:
    """有限浮点数原样返回，缺失、NaN 和无穷值写成 CSV 空值。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def normalize_hex(value: str | bytes | bytearray) -> str:
    """把任意带空格或 0x 的十六进制表示归一化为连续大写字符串。"""

    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex().upper()
    return re.sub(r"[^0-9A-Fa-f]", "", str(value)).upper()


def spaced_hex(value: str | bytes | bytearray) -> str:
    """把十六进制帧格式化成 ``AA BB CC``，便于人工查看。"""

    compact = normalize_hex(value)
    if len(compact) % 2:
        raise ValueError(f"hex string has an odd number of digits: {value!r}")
    return " ".join(compact[index : index + 2] for index in range(0, len(compact), 2))


def parse_int(value: Any, *, default: int | None = None) -> int:
    """读取 CSV 整数字段，同时兼容 ``1`` 和 ``1.0``。"""

    text = str(value).strip()
    if not text:
        if default is None:
            raise ValueError("missing integer value")
        return int(default)
    return int(float(text))


def sanitize_token(value: str, label: str) -> str:
    """清洗文件名标签，只允许小写字母、数字和连字符。"""

    token = str(value).strip().lower()
    if not re.fullmatch(r"[a-z0-9-]+", token):
        raise ValueError(
            f"{label} must contain only lower-case letters, digits, or '-', got {value!r}."
        )
    return token


def gain_token(value: float) -> str:
    """把接收增益转换成文件名安全格式，例如 ``20.5 -> 20p5``。"""

    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}".replace(".", "p").replace("-", "m")


def gain_from_token(value: str) -> float:
    """把文件名中的增益 token 还原为浮点数。"""

    return float(str(value).replace("m", "-").replace("p", "."))


@dataclass(frozen=True)
class CaptureDescriptor:
    """一轮连续 USRP 采集的全部实验、PHY 和射频参数。"""

    experiment_id: int
    session_id: int
    location_id: str
    condition: str
    run_id: int
    sf: int
    bandwidth_hz: int
    sample_rate_hz: int
    preamble_symbols: int
    sync_word: int
    cr: int
    phy_crc: bool
    center_frequency_hz: int
    rx_gain_db: float
    pay_len: int = 33
    ldro_mode: int = 1
    crc_mode: str = "grlora"
    rf_bandwidth_hz: int = 1_000_000
    antenna: str = "RX2"
    agc_enabled: bool = False
    device_args: str = ""

    def validate(self) -> None:
        """检查参数范围，阻止错误 PHY 配置进入文件名和 metadata。"""

        if self.experiment_id < 0 or self.session_id < 0 or self.run_id < 0:
            raise ValueError("experiment/session/run IDs must be non-negative.")
        sanitize_token(self.location_id, "location_id")
        sanitize_token(self.condition, "condition")
        if not 7 <= self.sf <= 12:
            raise ValueError(f"SF must be in [7, 12], got {self.sf}.")
        if self.bandwidth_hz <= 0 or self.sample_rate_hz <= 0:
            raise ValueError("sample rate and bandwidth must be positive.")
        if self.sample_rate_hz % self.bandwidth_hz:
            raise ValueError("sample rate must be an integer multiple of bandwidth.")
        if self.cr not in {1, 2, 3, 4}:
            raise ValueError("CR index must be 1..4.")
        if not 0 <= self.sync_word <= 0xFF:
            raise ValueError("sync word must fit uint8.")
        if self.pay_len <= 0 or self.preamble_symbols <= 0:
            raise ValueError("payload length and preamble length must be positive.")
        if self.crc_mode not in {"grlora", "sx1276"}:
            raise ValueError("crc_mode must be grlora or sx1276.")

    @property
    def capture_uid(self) -> str:
        """返回清单目录使用的短 ID，不包含会重复出现的 PHY 参数。"""

        return (
            f"exp{self.experiment_id:03d}_sess{self.session_id:03d}"
            f"_run{self.run_id:03d}"
        )

    @property
    def canonical_filename(self) -> str:
        """生成自描述的连续 IQ 文件名，后续可从文件名恢复关键参数。"""

        self.validate()
        return (
            f"rxcap_exp{self.experiment_id:03d}"
            f"_sess{self.session_id:03d}"
            f"_loc{sanitize_token(self.location_id, 'location_id')}"
            f"_cond{sanitize_token(self.condition, 'condition')}"
            f"_run{self.run_id:03d}"
            f"_sf{self.sf}"
            f"_bw{self.bandwidth_hz}"
            f"_fs{self.sample_rate_hz}"
            f"_pre{self.preamble_symbols}"
            f"_sw{self.sync_word:02X}"
            f"_cr4{self.cr + 4}"
            f"_crc{int(self.phy_crc)}"
            f"_fc{self.center_frequency_hz}"
            f"_rxg{gain_token(self.rx_gain_db)}.cfile"
        )


_CAPTURE_NAME_RE = re.compile(
    r"^rxcap_exp(?P<experiment_id>\d+)"
    r"_sess(?P<session_id>\d+)"
    r"_loc(?P<location_id>[a-z0-9-]+)"
    r"_cond(?P<condition>[a-z0-9-]+)"
    r"_run(?P<run_id>\d+)"
    r"_sf(?P<sf>\d+)"
    r"_bw(?P<bandwidth_hz>\d+)"
    r"_fs(?P<sample_rate_hz>\d+)"
    r"_pre(?P<preamble_symbols>\d+)"
    r"_sw(?P<sync_word>[0-9A-Fa-f]{2})"
    r"_cr4(?P<cr_denominator>[5-8])"
    r"_crc(?P<phy_crc>[01])"
    r"_fc(?P<center_frequency_hz>\d+)"
    r"_rxg(?P<rx_gain_db>[mp0-9]+)\.cfile$"
)


def parse_capture_filename(path: str | Path) -> dict[str, Any]:
    """从规范 cfile 文件名反解析实验和 PHY 参数；旧文件名返回空字典。"""

    match = _CAPTURE_NAME_RE.fullmatch(Path(path).name)
    if match is None:
        return {}
    values: dict[str, Any] = dict(match.groupdict())
    for key in (
        "experiment_id",
        "session_id",
        "run_id",
        "sf",
        "bandwidth_hz",
        "sample_rate_hz",
        "preamble_symbols",
        "center_frequency_hz",
    ):
        values[key] = int(values[key])
    values["sync_word"] = int(values["sync_word"], 16)
    values["cr"] = int(values.pop("cr_denominator")) - 4
    values["phy_crc"] = bool(int(values["phy_crc"]))
    values["rx_gain_db"] = gain_from_token(values["rx_gain_db"])
    return values


def capture_sidecar_path(capture_path: str | Path) -> Path:
    """返回连续 IQ 的 JSON sidecar 路径：``xxx.cfile.json``。"""

    return Path(str(Path(capture_path)) + ".json")


def descriptor_from_mapping(values: dict[str, Any]) -> CaptureDescriptor:
    """把文件名、sidecar 或 CLI 提供的字段合并为强类型采集描述。"""

    defaults = {
        "pay_len": 33,
        "ldro_mode": 1,
        "crc_mode": "grlora",
        "rf_bandwidth_hz": 1_000_000,
        "antenna": "RX2",
        "agc_enabled": False,
        "device_args": "",
    }
    merged = {**defaults, **values}
    required = {
        item.name
        for item in fields(CaptureDescriptor)
        if item.name not in defaults
    }
    missing = sorted(key for key in required if merged.get(key) is None)
    if missing:
        raise ValueError(
            "capture metadata is incomplete; use a canonical filename, sidecar JSON, "
            f"or CLI overrides for: {', '.join(missing)}"
        )
    descriptor = CaptureDescriptor(
        experiment_id=int(merged["experiment_id"]),
        session_id=int(merged["session_id"]),
        location_id=str(merged["location_id"]),
        condition=str(merged["condition"]),
        run_id=int(merged["run_id"]),
        sf=int(merged["sf"]),
        bandwidth_hz=int(merged["bandwidth_hz"]),
        sample_rate_hz=int(merged["sample_rate_hz"]),
        preamble_symbols=int(merged["preamble_symbols"]),
        sync_word=int(merged["sync_word"]),
        cr=int(merged["cr"]),
        phy_crc=bool(merged["phy_crc"]),
        center_frequency_hz=int(merged["center_frequency_hz"]),
        rx_gain_db=float(merged["rx_gain_db"]),
        pay_len=int(merged["pay_len"]),
        ldro_mode=int(merged["ldro_mode"]),
        crc_mode=str(merged["crc_mode"]).lower(),
        rf_bandwidth_hz=int(merged["rf_bandwidth_hz"]),
        antenna=str(merged["antenna"]),
        agc_enabled=bool(merged["agc_enabled"]),
        device_args=str(merged["device_args"]),
    )
    descriptor.validate()
    return descriptor


def descriptor_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """仅提取 CLI 中显式给出的采集描述字段。"""

    names = {item.name for item in fields(CaptureDescriptor)}
    return {
        name: getattr(args, name)
        for name in names
        if hasattr(args, name) and getattr(args, name) is not None
    }


def load_capture_descriptor(
    capture_path: str | Path,
    args: argparse.Namespace | None = None,
) -> CaptureDescriptor:
    """按“文件名 < sidecar < CLI”的优先级加载采集描述。"""

    capture = Path(capture_path).expanduser().resolve()
    values = parse_capture_filename(capture)
    sidecar = capture_sidecar_path(capture)
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("schema") != CAPTURE_SCHEMA:
            raise ValueError(f"unsupported capture sidecar schema in {sidecar}.")
        values.update(payload.get("capture", {}))
    if args is not None:
        values.update(descriptor_overrides(args))
    return descriptor_from_mapping(values)


def capture_manifest_dir(output_root: Path, descriptor: CaptureDescriptor) -> Path:
    """返回某轮采集专属的清单目录 ``manifests/captures/<capture_uid>``。"""

    return output_root / "manifests" / "captures" / descriptor.capture_uid


def initialize_capture(args: argparse.Namespace) -> Path:
    """阶段 0：生成规范 cfile 名称和 sidecar，但不采集也不创建 IQ 内容。"""

    output_dir = ensure_inside_data(args.output_dir, label="raw capture directory")
    descriptor = descriptor_from_mapping(descriptor_overrides(args))
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = output_dir / descriptor.canonical_filename
    sidecar = capture_sidecar_path(capture_path)
    if capture_path.exists() and not args.overwrite:
        raise FileExistsError(f"capture already exists: {capture_path}")
    if sidecar.exists() and not args.overwrite:
        raise FileExistsError(f"capture sidecar already exists: {sidecar}")
    write_json(
        sidecar,
        {
            "schema": CAPTURE_SCHEMA,
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "capture_path": display_path(capture_path),
            "capture": asdict(descriptor),
            "recording": {
                "started_at_utc": None,
                "stopped_at_utc": None,
                "usrp_serial": None,
                "operator_notes": "",
            },
        },
    )
    print(capture_path)
    return capture_path


def build_reference_catalog(
    uart_path: str | Path,
    reference_root: str | Path,
    output_root: str | Path,
) -> Path:
    """阶段 1：把 UART 真值、reference metadata 和理想 IQ 汇总成目录表。

    输入：
        ``data/raw/packet_reference.txt`` 和
        ``data/reference_phy/{reference,metadata}``。
    输出：
        ``data/reference_phy/rfsr_db/manifests/reference_catalog.csv``。
    """

    root = ensure_inside_data(output_root, label="dataset output root")
    reference_source_root = ensure_inside_data(
        reference_root,
        label="reference source root",
    )
    uart_log = parse_uart_reference_log(uart_path)
    rows: list[dict[str, Any]] = []
    for packet in uart_log.first_reference_cycle():
        reference_id = int(packet.payload_id)
        metadata_path = reference_source_root / "metadata" / f"{reference_id:06d}.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing reference metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_frame = normalize_hex(packet.frame)
        metadata_frame = normalize_hex(metadata["packet"]["frame_hex"])
        if expected_frame != metadata_frame:
            raise ValueError(
                f"reference metadata {metadata_path} does not match UART frame "
                f"for payload ID {reference_id}."
            )
        iq = metadata["iq"]
        phy = metadata["phy"]
        symbols = metadata["symbols"]
        source_reference = reference_source_root / str(iq["relative_path"])
        if not source_reference.is_file():
            raise FileNotFoundError(f"missing reference IQ: {source_reference}")
        expected_bytes = int(iq["complex_samples"]) * COMPLEX_DTYPE.itemsize
        if source_reference.stat().st_size != expected_bytes:
            raise ValueError(
                f"{source_reference} has {source_reference.stat().st_size} bytes, "
                f"expected {expected_bytes}."
            )
        rows.append(
            {
                "reference_id": reference_id,
                "uart_seq": packet.seq,
                "uart_round": packet.round,
                "payload_id": packet.payload_id,
                "frame_hex": spaced_hex(packet.frame),
                "app_payload_hex": spaced_hex(packet.app_payload),
                "frame_bytes": len(packet.frame),
                "sf": int(phy["sf"]),
                "bandwidth_hz": int(phy["bandwidth_hz"]),
                "cr": int(phy["cr"]),
                "preamble_symbols": int(phy["preamble_symbols"]),
                "sync_word": int(phy["sync_word"]),
                "phy_crc": int(bool(phy["phy_crc"])),
                "ldro": int(bool(phy["ldro"])),
                "sample_rate_hz": int(phy["sample_rate_hz"]),
                "samples_per_symbol": int(phy["samples_per_symbol"]),
                "leading_silence_samples": int(phy["leading_silence_samples"]),
                "trailing_silence_samples": int(phy["trailing_silence_samples"]),
                "reference_samples": int(iq["complex_samples"]),
                "header_symbols": int(symbols["header_count"]),
                "payload_symbols": int(symbols["payload_count"]),
                "tx_period_ms": int(metadata["source_uart"]["phy"]["tx_period_ms"]),
                "source_reference_path": display_path(source_reference),
                "dataset_reference_path": (
                    f"reference/signalout_{reference_id:06d}_fulltrim.cfile"
                ),
                "reference_sha256": str(iq["sha256"]).upper(),
                "reference_metadata_path": display_path(metadata_path),
                "uart_source_path": display_path(uart_log.source_path),
                "uart_source_sha256": uart_log.source_sha256,
            }
        )
    catalog_path = root / "manifests" / "reference_catalog.csv"
    write_csv(catalog_path, rows, CATALOG_FIELDS)
    return catalog_path


def load_catalog(path: str | Path) -> list[dict[str, str]]:
    """读取 reference catalog，并拒绝空目录表。"""

    rows = read_csv(path)
    if not rows:
        raise ValueError(f"reference catalog is empty: {path}")
    return rows


def validate_capture_against_catalog(
    descriptor: CaptureDescriptor,
    catalog: Sequence[dict[str, str]],
) -> None:
    """确认采集的 SF/BW/preamble/sync/CR/CRC 与 reference 完全一致。"""

    row = catalog[0]
    checks = (
        ("sf", descriptor.sf, parse_int(row["sf"])),
        ("bandwidth_hz", descriptor.bandwidth_hz, parse_int(row["bandwidth_hz"])),
        ("preamble_symbols", descriptor.preamble_symbols, parse_int(row["preamble_symbols"])),
        ("sync_word", descriptor.sync_word, parse_int(row["sync_word"])),
        ("cr", descriptor.cr, parse_int(row["cr"])),
        ("phy_crc", int(descriptor.phy_crc), parse_int(row["phy_crc"])),
    )
    mismatches = [
        f"{name}: capture={actual}, reference={expected}"
        for name, actual, expected in checks
        if int(actual) != int(expected)
    ]
    if mismatches:
        raise ValueError("capture/reference PHY mismatch: " + "; ".join(mismatches))


def save_capture_manifest(
    capture_path: Path,
    descriptor: CaptureDescriptor,
    output_root: Path,
    *,
    capture_sha256: str | None = None,
) -> Path:
    """记录连续 cfile 的大小、时长、mtime 和可选 SHA-256。"""

    manifest_path = capture_manifest_dir(output_root, descriptor) / "capture.json"
    stat = capture_path.stat()
    existing: dict[str, Any] = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_file = existing.get("file", {})
    reusable_sha256 = None
    if (
        int(existing_file.get("bytes", -1)) == stat.st_size
        and int(existing_file.get("mtime_ns", -1)) == stat.st_mtime_ns
    ):
        reusable_sha256 = existing_file.get("sha256")
    payload = {
        "schema": CAPTURE_SCHEMA,
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "updated_at_utc": utc_now(),
        "capture_path": display_path(capture_path),
        "capture": asdict(descriptor),
        "file": {
            "bytes": stat.st_size,
            "complex_samples": stat.st_size // COMPLEX_DTYPE.itemsize,
            "duration_seconds": (
                stat.st_size
                / COMPLEX_DTYPE.itemsize
                / descriptor.sample_rate_hz
            ),
            "mtime_ns": stat.st_mtime_ns,
            "sha256": capture_sha256 or reusable_sha256,
        },
    }
    write_json(manifest_path, payload)
    return manifest_path


def run_detection(
    capture_path: str | Path,
    descriptor: CaptureDescriptor,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """阶段 2：用 GNU Radio LoRa 接收链扫描连续 IQ。

    这里只回答“哪里检测到包、解出了什么、CRC 是否正确”，不在这一阶段
    决定 reference_id。输出位于当前 capture 的 ``detections.csv``。

    这是唯一需要 GNU Radio 和 ``gnuradio.lora_sdr`` 的阶段；服务器可以直接
    使用采集机已经生成的 CSV，不需要安装 GNU Radio。
    """

    root = ensure_inside_data(output_root, label="dataset output root")
    capture = Path(capture_path).expanduser().resolve()
    if not capture.is_file():
        raise FileNotFoundError(capture)
    if capture.stat().st_size % COMPLEX_DTYPE.itemsize:
        raise ValueError(f"capture byte size is not divisible by 8: {capture}")
    destination = capture_manifest_dir(root, descriptor) / "detections.csv"
    if destination.is_file() and not overwrite:
        print(f"reuse {destination}")
        return destination

    # 延迟导入，保证 catalog/associate/trim/server 在无 GNU Radio 环境可运行。
    try:
        from noisy_iq.detector import run_grlora_packet_detector
    except ImportError as exc:
        raise RuntimeError(
            "detect requires GNU Radio and gr-lora_sdr; run it in the RadioConda "
            "environment used by the USRP collector."
        ) from exc

    # 把 CaptureDescriptor 转成 noisy_iq.detector 所需的接收链参数。
    detector_args = argparse.Namespace(
        sf=descriptor.sf,
        bw=float(descriptor.bandwidth_hz),
        samp_rate=float(descriptor.sample_rate_hz),
        cr=descriptor.cr,
        pay_len=descriptor.pay_len,
        has_crc=descriptor.phy_crc,
        impl_head=False,
        soft_decoding=False,
        center_freq=float(descriptor.center_frequency_hz),
        sync_word=descriptor.sync_word,
        preamble_len=descriptor.preamble_symbols,
        ldro_mode=descriptor.ldro_mode,
        crc_mode=1 if descriptor.crc_mode == "sx1276" else 0,
        print_header=False,
        print_grlora=False,
    )
    detections = run_grlora_packet_detector(capture, detector_args)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(detections):
        decoded = item.get("decoded_payload_hex", "")
        rows.append(
            {
                "detection_index": index,
                "frame_count": item.get("frame_count", ""),
                "start_sample_2m": item.get(
                    "packet_start_sample",
                    item.get("start_sample", ""),
                ),
                "preamble_end_sample_2m": item.get("end_sample", ""),
                "estimated_packet_end_sample_2m": item.get("packet_end_sample", ""),
                "header_ok": int(int(item.get("header_err", 1)) == 0),
                "crc_valid": int(bool(item.get("crc_valid", False))),
                "decoded_frame_hex": spaced_hex(decoded) if decoded else "",
                "decoded_payload_len": item.get("decoded_payload_len", ""),
                "grlora_snr_db": finite_or_blank(item.get("grlora_snr_db")),
                "cfo_hz": finite_or_blank(item.get("cfo")),
                "sto": finite_or_blank(item.get("sto")),
                "sfo": finite_or_blank(item.get("sfo")),
                "netid1": item.get("netid1", ""),
                "netid2": item.get("netid2", ""),
                "samples_per_symbol": item.get("samples_per_symbol", ""),
            }
        )
    write_csv(destination, rows, DETECTION_FIELDS)
    save_capture_manifest(capture, descriptor, root)
    return destination


def parse_receiver_log(
    log_path: str | Path,
    catalog: Sequence[dict[str, str]],
    destination: str | Path,
) -> Path:
    """把 GNU Radio 终端文本日志转成辅助审计表 ``rx_events.csv``。

    该表只用于人工复核终端打印与 catalog 是否一致；packet 的采样点位置仍以
    ``detections.csv`` 为准，理想 payload 仍以 UART catalog 为准。
    """

    source = Path(log_path).expanduser().resolve()
    frame_map = {
        normalize_hex(row["frame_hex"]): parse_int(row["reference_id"])
        for row in catalog
    }
    events: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        match = re.match(r"^rx msg:\s*(.+)$", line.strip(), flags=re.IGNORECASE)
        if match:
            if current is not None:
                events.append(current)
            tokens = [token.strip() for token in match.group(1).split(",")]
            try:
                frame = bytes(int(token, 0) for token in tokens)
                frame_hex = normalize_hex(frame)
            except ValueError:
                frame = b""
                frame_hex = ""
            reference_id = frame_map.get(frame_hex)
            current = {
                "log_event_index": len(events),
                "line_number": line_number,
                "crc_valid": "",
                "decoded_frame_hex": spaced_hex(frame) if frame else "",
                "decoded_frame_bytes": len(frame),
                "decoded_reference_id": (
                    "" if reference_id is None else reference_id
                ),
                "catalog_frame_exact": int(reference_id is not None),
            }
            continue
        if current is None:
            continue
        lowered = line.strip().lower()
        if "crc valid" in lowered:
            current["crc_valid"] = 1
        elif "crc" in lowered and any(
            token in lowered for token in ("invalid", "fail", "wrong")
        ):
            current["crc_valid"] = 0
    if current is not None:
        events.append(current)
    path = ensure_inside_data(destination, label="receiver-log CSV")
    write_csv(
        path,
        events,
        (
            "log_event_index",
            "line_number",
            "crc_valid",
            "decoded_frame_hex",
            "decoded_frame_bytes",
            "decoded_reference_id",
            "catalog_frame_exact",
        ),
    )
    return path


def median(values: Sequence[float]) -> float:
    """以 float64 计算中位数，并拒绝空序列。"""

    if not values:
        raise ValueError("cannot take median of an empty sequence")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def build_anchor_segments(
    detections: Sequence[dict[str, Any]],
    *,
    variants: int,
    nominal_period: float,
) -> list[dict[str, Any]]:
    """用 CRC 精确匹配包建立发包周期锚点和连续时序分段。

    如果相邻 CRC 包的 reference_id 递增关系或采样间隔明显不一致，就开始一个
    新 segment，防止设备复位或录制中断后的时序被错误地连在一起。
    """

    anchors = [item for item in detections if item.get("exact_reference_id") is not None]
    if not anchors:
        return []
    segments: list[dict[str, Any]] = []
    current = {"anchors": [], "index": 0}
    previous: dict[str, Any] | None = None
    previous_slot = 0
    for anchor in anchors:
        if previous is None:
            anchor["anchor_slot"] = 0
            current["anchors"].append(anchor)
            previous = anchor
            continue
        delta_samples = int(anchor["start_sample_2m"]) - int(previous["start_sample_2m"])
        delta_slots = max(1, int(round(delta_samples / nominal_period)))
        expected_id = (
            int(previous["exact_reference_id"]) + delta_slots
        ) % variants
        residual = abs(delta_samples - delta_slots * nominal_period)
        consistent = (
            expected_id == int(anchor["exact_reference_id"])
            and residual <= 0.35 * nominal_period
        )
        if not consistent:
            segments.append(current)
            current = {"anchors": [], "index": len(segments)}
            previous_slot = 0
            anchor["anchor_slot"] = 0
        else:
            previous_slot += delta_slots
            anchor["anchor_slot"] = previous_slot
        current["anchors"].append(anchor)
        previous = anchor
    segments.append(current)

    # 每个 segment 用全部锚点稳健估计实际发包周期和时间轴截距。
    for segment in segments:
        segment_anchors = segment["anchors"]
        period_estimates: list[float] = []
        for left, right in zip(segment_anchors, segment_anchors[1:]):
            slot_delta = int(right["anchor_slot"]) - int(left["anchor_slot"])
            if slot_delta > 0:
                period_estimates.append(
                    (
                        int(right["start_sample_2m"])
                        - int(left["start_sample_2m"])
                    )
                    / slot_delta
                )
        period = median(period_estimates) if period_estimates else float(nominal_period)
        offsets = [
            int(anchor["start_sample_2m"]) - period * int(anchor["anchor_slot"])
            for anchor in segment_anchors
        ]
        segment["period"] = period
        segment["offset"] = median(offsets)
        segment["first_start"] = int(segment_anchors[0]["start_sample_2m"])
        segment["last_start"] = int(segment_anchors[-1]["start_sample_2m"])
        segment["base_reference_id"] = int(segment_anchors[0]["exact_reference_id"])
        segment["base_slot"] = int(segment_anchors[0]["anchor_slot"])
    return segments


def closest_segment(
    start_sample: int,
    segments: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """为一个检测点选择时间上最近的锚点分段。"""

    if not segments:
        return None
    return min(
        segments,
        key=lambda segment: min(
            abs(start_sample - int(segment["first_start"])),
            abs(start_sample - int(segment["last_start"])),
        ),
    )


def anchor_neighbors(
    segment: dict[str, Any],
    slot: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """返回某个 schedule slot 前后最近的 CRC 精确锚点。"""

    previous = None
    following = None
    for anchor in segment["anchors"]:
        anchor_slot = int(anchor["anchor_slot"])
        if anchor_slot <= slot:
            previous = anchor
        if anchor_slot >= slot and following is None:
            following = anchor
    return previous, following


def recover_missing_packet(
    capture: np.memmap,
    predicted_start: int,
    descriptor: CaptureDescriptor,
    catalog_row: dict[str, str],
) -> tuple[int | None, float | None]:
    """在预测时刻附近用项目内 XCopy 前导码检测补找漏检包。

    只在一个包附近的局部窗口搜索，不对 12.48 GB 文件做第二次全局扫描。
    返回确认后的 2 MS/s 起点和检测分数；证据不足时返回 ``None``。
    """

    try:
        from weak_decoder.synchronization.xcopy_sync import (
            XCopyConfig,
            scan_xcopy_packet_preambles,
        )
    except ImportError:
        return None, None

    chirp_samples = (1 << descriptor.sf) * (
        descriptor.sample_rate_hz // descriptor.bandwidth_hz
    )
    reference_rate = parse_int(catalog_row["sample_rate_hz"])
    reference_samples = parse_int(catalog_row["reference_samples"])
    leading_reference = parse_int(catalog_row["leading_silence_samples"])
    packet_samples = int(
        round(
            (reference_samples - leading_reference)
            * descriptor.sample_rate_hz
            / reference_rate
        )
    )
    margin = 2 * chirp_samples
    window_start = max(0, predicted_start - margin)
    window_stop = min(capture.size, predicted_start + packet_samples + margin)
    if window_stop - window_start < packet_samples:
        return None, None
    local = np.asarray(capture[window_start:window_stop])
    config = XCopyConfig(
        sf=descriptor.sf,
        bw=float(descriptor.bandwidth_hz),
        samp_rate=float(descriptor.sample_rate_hz),
        preamble_symbols=descriptor.preamble_symbols,
        sync_word=descriptor.sync_word,
        retransmit_period_samples=int(
            round(
                descriptor.sample_rate_hz
                * parse_int(catalog_row["tx_period_ms"])
                / 1000.0
            )
        ),
        payload_symbols=(
            parse_int(catalog_row["header_symbols"])
            + parse_int(catalog_row["payload_symbols"])
        ),
    )
    detections = scan_xcopy_packet_preambles(local, config)
    if not detections:
        return None, None
    best = min(
        detections,
        key=lambda item: abs(
            window_start
            + int(item.coarse_preamble_start_sample)
            - predicted_start
        ),
    )
    recovered = window_start + int(best.coarse_preamble_start_sample)
    if abs(recovered - predicted_start) > chirp_samples:
        return None, float(best.score)
    return recovered, float(best.score)


def associate_packets(
    capture_path: str | Path,
    descriptor: CaptureDescriptor,
    catalog_path: str | Path,
    detections_path: str | Path,
    output_root: str | Path,
    *,
    recover_missing: bool = True,
    usrp_log: str | Path | None = None,
) -> Path:
    """阶段 3：把 IQ 检测结果关联到 UART reference_id。

    证据优先级：

    1. CRC 通过且完整 33-byte frame 与 catalog 完全相等：``crc_exact``；
    2. CRC 不通过但位于前后 CRC 锚点的正确时隙：按周期推断；
    3. 全局 detector 漏检：只在预测位置附近做一次 XCopy 局部确认；
    4. 仍无法确认：保留为 ``ambiguous``，绝不进入 trim。

    输出为当前 capture 的 ``packets.csv``。可选终端日志另写
    ``rx_events.csv``，不会覆盖上述证据链。
    """

    root = ensure_inside_data(output_root, label="dataset output root")
    capture_path = Path(capture_path).expanduser().resolve()
    catalog = load_catalog(catalog_path)
    validate_capture_against_catalog(descriptor, catalog)
    frame_map = {
        normalize_hex(row["frame_hex"]): parse_int(row["reference_id"])
        for row in catalog
    }
    catalog_by_id = {
        parse_int(row["reference_id"]): row
        for row in catalog
    }
    variants = len(catalog_by_id)
    nominal_period = (
        descriptor.sample_rate_hz
        * parse_int(catalog[0]["tx_period_ms"])
        / 1000.0
    )

    # 先把 CSV 字符串转换成可计算字段，并找出 CRC+catalog 精确锚点。
    detections: list[dict[str, Any]] = []
    for row in read_csv(detections_path):
        start = parse_int(row["start_sample_2m"])
        decoded = normalize_hex(row.get("decoded_frame_hex", ""))
        crc_valid = parse_bool(row.get("crc_valid", ""))
        exact_id = frame_map.get(decoded) if crc_valid else None
        detections.append(
            {
                **row,
                "detection_index": parse_int(row["detection_index"]),
                "start_sample_2m": start,
                "decoded_compact": decoded,
                "crc_bool": crc_valid,
                "exact_reference_id": exact_id,
            }
        )
    detections.sort(key=lambda item: int(item["start_sample_2m"]))
    segments = build_anchor_segments(
        detections,
        variants=variants,
        nominal_period=nominal_period,
    )

    # 给每个已检测事件分配时序 slot 和 reference_id。
    rows: list[dict[str, Any]] = []
    for detection in detections:
        start = int(detection["start_sample_2m"])
        segment = closest_segment(start, segments)
        exact_id = detection.get("exact_reference_id")
        if segment is None:
            rows.append(
                {
                    "sequence_segment": "",
                    "schedule_slot": "",
                    "detection_index": detection["detection_index"],
                    "start_sample_2m": start,
                    "predicted_start_sample_2m": "",
                    "timing_residual_samples": "",
                    "reference_id": "",
                    "association_method": "unanchored",
                    "association_confidence": 0.0,
                    "status": "ambiguous",
                    "crc_valid": int(detection["crc_bool"]),
                    "decoded_frame_hex": detection.get("decoded_frame_hex", ""),
                    "catalog_frame_exact": 0,
                    "grlora_snr_db": detection.get("grlora_snr_db", ""),
                    "estimated_cfo_hz": detection.get("cfo_hz", ""),
                    "sto": detection.get("sto", ""),
                    "sfo": detection.get("sfo", ""),
                    "notes": "no CRC-exact anchor is available",
                }
            )
            continue

        period = float(segment["period"])
        slot = int(round((start - float(segment["offset"])) / period))
        predicted = int(round(float(segment["offset"]) + slot * period))
        residual = start - predicted
        expected_id = (
            int(segment["base_reference_id"])
            + slot
            - int(segment["base_slot"])
        ) % variants
        previous, following = anchor_neighbors(segment, slot)
        tolerance = max(
            2
            * (1 << descriptor.sf)
            * (descriptor.sample_rate_hz // descriptor.bandwidth_hz),
            int(round(0.02 * period)),
        )

        if exact_id is not None:
            method = "crc_exact"
            confidence = 1.0
            status = "accepted"
            reference_id = int(exact_id)
            notes = ""
        elif detection["crc_bool"]:
            method = "crc_catalog_mismatch"
            confidence = 0.0
            status = "rejected"
            reference_id = ""
            notes = "CRC passed but the decoded frame is absent from the UART catalog"
        elif abs(residual) <= tolerance:
            if previous is not None and following is not None and previous is not following:
                method = "neighbor_inferred_high"
                confidence = 0.9
            else:
                method = "schedule_inferred_medium"
                confidence = 0.7
            status = "accepted"
            reference_id = expected_id
            notes = "identity inferred from IQ timing and CRC-exact schedule anchors"
        else:
            method = "timing_outlier"
            confidence = 0.0
            status = "ambiguous"
            reference_id = ""
            notes = f"schedule residual {residual} exceeds tolerance {tolerance}"

        rows.append(
            {
                "sequence_segment": segment["index"],
                "schedule_slot": slot,
                "detection_index": detection["detection_index"],
                "start_sample_2m": start,
                "predicted_start_sample_2m": predicted,
                "timing_residual_samples": residual,
                "reference_id": reference_id,
                "association_method": method,
                "association_confidence": confidence,
                "status": status,
                "crc_valid": int(detection["crc_bool"]),
                "decoded_frame_hex": detection.get("decoded_frame_hex", ""),
                "catalog_frame_exact": int(exact_id is not None),
                "grlora_snr_db": detection.get("grlora_snr_db", ""),
                "estimated_cfo_hz": detection.get("cfo_hz", ""),
                "sto": detection.get("sto", ""),
                "sfo": detection.get("sfo", ""),
                "previous_anchor_detection": (
                    "" if previous is None else previous["detection_index"]
                ),
                "next_anchor_detection": (
                    "" if following is None else following["detection_index"]
                ),
                "schedule_period_samples": period,
                "local_search_used": 0,
                "local_search_score": "",
                "notes": notes,
            }
        )

    # 在锚点覆盖区间内找“理论上应该存在、但 detector 没有事件”的空 slot。
    occupied = {
        (parse_int(row["sequence_segment"]), parse_int(row["schedule_slot"]))
        for row in rows
        if str(row.get("sequence_segment", "")).strip()
        and str(row.get("schedule_slot", "")).strip()
    }
    for segment in segments:
        anchor_slots = [int(anchor["anchor_slot"]) for anchor in segment["anchors"]]
        if len(anchor_slots) < 2:
            continue
        for slot in range(min(anchor_slots), max(anchor_slots) + 1):
            key = (int(segment["index"]), slot)
            if key in occupied:
                continue
            predicted = int(round(float(segment["offset"]) + slot * float(segment["period"])))
            reference_id = (
                int(segment["base_reference_id"])
                + slot
                - int(segment["base_slot"])
            ) % variants
            previous, following = anchor_neighbors(segment, slot)
            rows.append(
                {
                    "sequence_segment": segment["index"],
                    "schedule_slot": slot,
                    "detection_index": "",
                    "start_sample_2m": "",
                    "predicted_start_sample_2m": predicted,
                    "timing_residual_samples": "",
                    "reference_id": reference_id,
                    "association_method": "schedule_prediction",
                    "association_confidence": 0.0,
                    "status": "needs_local_search",
                    "crc_valid": 0,
                    "decoded_frame_hex": "",
                    "catalog_frame_exact": 0,
                    "grlora_snr_db": "",
                    "estimated_cfo_hz": "",
                    "sto": "",
                    "sfo": "",
                    "previous_anchor_detection": (
                        "" if previous is None else previous["detection_index"]
                    ),
                    "next_anchor_detection": (
                        "" if following is None else following["detection_index"]
                    ),
                    "schedule_period_samples": segment["period"],
                    "local_search_used": 0,
                    "local_search_score": "",
                    "notes": "no global detector event in this schedule slot",
                }
            )

    # 对空 slot 做局部 IQ 搜索；搜不到就降级为 ambiguous。
    if recover_missing and any(row["status"] == "needs_local_search" for row in rows):
        capture = np.memmap(capture_path, dtype=COMPLEX_DTYPE, mode="r")
        for row in rows:
            if row["status"] != "needs_local_search":
                continue
            reference_id = parse_int(row["reference_id"])
            recovered, score = recover_missing_packet(
                capture,
                parse_int(row["predicted_start_sample_2m"]),
                descriptor,
                catalog_by_id[reference_id],
            )
            row["local_search_used"] = 1
            row["local_search_score"] = "" if score is None else score
            if recovered is None:
                row["status"] = "ambiguous"
                row["notes"] = "schedule slot was not confirmed by local IQ preamble search"
                continue
            row["start_sample_2m"] = recovered
            row["timing_residual_samples"] = (
                recovered - parse_int(row["predicted_start_sample_2m"])
            )
            row["association_method"] = "schedule_inferred_medium"
            row["association_confidence"] = 0.65
            row["status"] = "accepted"
            row["notes"] = "missing global detection recovered by local IQ preamble search"

    # 同一个时隙如果出现多个候选，优先保留 CRC 精确匹配，其余拒绝。
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if not str(row.get("sequence_segment", "")).strip():
            continue
        key = (parse_int(row["sequence_segment"]), parse_int(row["schedule_slot"]))
        grouped.setdefault(key, []).append(row)
    for same_slot in grouped.values():
        accepted = [row for row in same_slot if row["status"] == "accepted"]
        if len(accepted) <= 1:
            continue
        accepted.sort(
            key=lambda row: (
                0 if row["association_method"] == "crc_exact" else 1,
                abs(parse_int(row.get("timing_residual_samples", 0), default=0)),
            )
        )
        for duplicate in accepted[1:]:
            duplicate["status"] = "rejected"
            duplicate["association_method"] = "duplicate_schedule_slot"
            duplicate["association_confidence"] = 0.0
            duplicate["notes"] = "another detection was selected for this schedule slot"

    # 按物理时间重新编号；physical_packet_uid 用作防止训练集泄漏的 split_group。
    primary = [
        row
        for row in rows
        if row["status"] in {"accepted", "needs_local_search", "ambiguous"}
        and str(row.get("sequence_segment", "")).strip()
        and str(row.get("schedule_slot", "")).strip()
    ]
    primary.sort(
        key=lambda row: parse_int(
            row.get("start_sample_2m")
            or row.get("predicted_start_sample_2m")
        )
    )
    primary_ids = {id(row): index for index, row in enumerate(primary)}
    for row in rows:
        index = primary_ids.get(id(row))
        if index is None:
            row["capture_packet_index"] = ""
            row["physical_packet_uid"] = ""
        else:
            row["capture_packet_index"] = index
            row["physical_packet_uid"] = (
                f"{descriptor.capture_uid}:seg{parse_int(row['sequence_segment']):02d}"
                f":slot{parse_int(row['schedule_slot']):06d}"
            )
        row["capture_uid"] = descriptor.capture_uid

    rows.sort(
        key=lambda row: parse_int(
            row.get("start_sample_2m")
            or row.get("predicted_start_sample_2m"),
            default=2**63 - 1,
        )
    )
    destination = capture_manifest_dir(root, descriptor) / "packets.csv"
    write_csv(destination, rows, PACKET_FIELDS)

    if usrp_log is not None:
        parse_receiver_log(
            usrp_log,
            catalog,
            capture_manifest_dir(root, descriptor) / "rx_events.csv",
        )
    return destination


def ensure_reference_file(
    catalog_row: dict[str, str],
    output_root: Path,
    *,
    mode: str,
) -> Path:
    """把理想 reference 放入最终数据库，优先硬链接以避免重复占用空间。"""

    source = (REPO_ROOT / catalog_row["source_reference_path"]).resolve()
    if not source.is_file():
        source = Path(catalog_row["source_reference_path"]).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reference source does not exist: {source}")
    destination = output_root / catalog_row["dataset_reference_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"existing reference has the wrong size: {destination}")
        return destination
    if mode not in {"auto", "hardlink", "copy"}:
        raise ValueError(f"unsupported reference mode: {mode}")
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return destination
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, destination)
    return destination


def write_complex_file(path: Path, values: np.ndarray, *, overwrite: bool) -> None:
    """原子写入无文件头 little-endian complex64 IQ。"""

    if path.exists() and not overwrite:
        expected_bytes = int(values.size) * COMPLEX_DTYPE.itemsize
        if path.stat().st_size == expected_bytes:
            return
        raise FileExistsError(f"output exists with unexpected size: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    np.asarray(values, dtype=COMPLEX_DTYPE).tofile(temporary)
    os.replace(temporary, path)


def load_or_compute_capture_hash(
    capture_path: Path,
    descriptor: CaptureDescriptor,
    output_root: Path,
    *,
    compute_hash: bool,
) -> str | None:
    """复用 capture.json 中有效的哈希；必要时才重新扫描整个大文件。"""

    manifest_path = capture_manifest_dir(output_root, descriptor) / "capture.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        file_info = payload.get("file", {})
        stat = capture_path.stat()
        if (
            int(file_info.get("bytes", -1)) == stat.st_size
            and int(file_info.get("mtime_ns", -1)) == stat.st_mtime_ns
            and file_info.get("sha256")
        ):
            return str(file_info["sha256"])
    digest = sha256_file(capture_path) if compute_hash else None
    save_capture_manifest(
        capture_path,
        descriptor,
        output_root,
        capture_sha256=digest,
    )
    return digest


def rebuild_views_manifest(output_root: Path) -> Path:
    """从 OTA metadata 重建统一的 ``manifests/views.csv``。

    每个落盘的 1 MS/s OTA phase 枚举 q=0..3 四个逻辑视图。逻辑视图不再
    单独落盘；训练 loader 运行时执行 ``ota[q::4]`` 得到 250 kS/s 输入。
    """

    rows: list[dict[str, Any]] = []
    for metadata_path in sorted((output_root / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != OTA_SCHEMA:
            continue
        view = metadata["view"]
        for low_phase in view["available_lowrate_phases_1m"]:
            adc_phase = int(view["adc_phase_2m"])
            combined = adc_phase + 2 * int(low_phase)
            rows.append(
                {
                    "view_id": f"{metadata['ota_id']}:q{int(low_phase)}",
                    "physical_packet_uid": metadata["physical_packet_uid"],
                    "capture_uid": metadata["capture"]["capture_uid"],
                    "capture_packet_index": metadata["capture_packet_index"],
                    "reference_id": metadata["reference"]["reference_id"],
                    "ota_path": metadata["ota"]["relative_path"],
                    "reference_path": metadata["reference"]["relative_path"],
                    "adc_phase_2m": adc_phase,
                    "lowrate_phase_1m": int(low_phase),
                    "combined_decimation_phase_2m": combined,
                    "timing_offset_samples_1m": combined / 2.0,
                    "input_sample_rate_hz": view["lowrate_sample_rate_hz"],
                    "input_samples": view["lowrate_samples"],
                    "target_sample_rate_hz": view["sample_rate_hz"],
                    "target_samples": view["complex_samples"],
                    "split_group": metadata["split_group"],
                }
            )
    destination = output_root / "manifests" / "views.csv"
    write_csv(destination, rows, VIEW_FIELDS)
    return destination


def trim_packets(
    capture_path: str | Path,
    descriptor: CaptureDescriptor,
    catalog_path: str | Path,
    packets_path: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
    reference_mode: str = "auto",
    compute_capture_hash: bool = True,
) -> list[Path]:
    """阶段 4：裁剪 accepted packet，并构造 2×4 多相训练视图。

    对每个已确认 packet：

    1. 从 2 MS/s 连续 IQ 中向前保留与 reference 零前缀等时长的真实包外 IQ；
    2. 裁出与 1 MS/s reference 覆盖相同物理时间的 2 MS/s 窗口；
    3. 用 ``crop[0::2]``、``crop[1::2]`` 落盘两个 1 MS/s OTA phase；
    4. 在 metadata/views.csv 中为每个 phase 枚举四个 250 kS/s 抽取相位；
    5. 配对同一 reference_id 的理想 1 MS/s 标签。

    只有 ``packets.csv`` 中 ``status=accepted`` 的行会进入训练数据库。
    """

    root = ensure_inside_data(output_root, label="dataset output root")
    capture_path = Path(capture_path).expanduser().resolve()
    catalog = load_catalog(catalog_path)
    validate_capture_against_catalog(descriptor, catalog)
    catalog_by_id = {
        parse_int(row["reference_id"]): row
        for row in catalog
    }
    # 当前契约固定为原始 2 MS/s、落盘标签和 OTA 1 MS/s。
    reference_rate = parse_int(catalog[0]["sample_rate_hz"])
    if descriptor.sample_rate_hz != 2 * reference_rate:
        raise ValueError(
            "the 2x2 polyphase trim contract requires capture rate = 2 * "
            f"reference rate, got {descriptor.sample_rate_hz} and {reference_rate}."
        )
    reference_samples = parse_int(catalog[0]["reference_samples"])
    reference_leading = parse_int(catalog[0]["leading_silence_samples"])
    # reference 的 10,000 个零前缀对应原始 2 MS/s 中 20,000 个真实环境样点。
    source_crop_samples = 2 * reference_samples
    source_leading = 2 * reference_leading
    lowrate_samples = reference_samples // 4
    if reference_samples % 4:
        raise ValueError("reference length must be divisible by four.")

    capture = np.memmap(capture_path, dtype=COMPLEX_DTYPE, mode="r")
    capture_hash = load_or_compute_capture_hash(
        capture_path,
        descriptor,
        root,
        compute_hash=compute_capture_hash,
    )
    generated: list[Path] = []
    trim_issues: list[dict[str, Any]] = []
    for row in read_csv(packets_path):
        # ambiguous/rejected 包保留在审计表中，但绝不裁成训练样本。
        if row.get("status") != "accepted":
            continue
        start_packet = parse_int(row["start_sample_2m"])
        trim_start = start_packet - source_leading
        trim_stop = trim_start + source_crop_samples
        if trim_start < 0 or trim_stop > capture.size:
            trim_issues.append(
                {
                    "physical_packet_uid": row.get("physical_packet_uid", ""),
                    "capture_packet_index": row.get("capture_packet_index", ""),
                    "start_sample_2m": start_packet,
                    "trim_start_2m": trim_start,
                    "trim_stop_2m": trim_stop,
                    "reason": "fixed fulltrim window falls outside the source capture",
                }
            )
            continue
        reference_id = parse_int(row["reference_id"])
        catalog_row = catalog_by_id[reference_id]
        reference_path = ensure_reference_file(
            catalog_row,
            root,
            mode=reference_mode,
        )
        reference = np.memmap(reference_path, dtype=COMPLEX_DTYPE, mode="r")
        if reference.size != reference_samples:
            raise ValueError(f"reference has an unexpected length: {reference_path}")
        if not np.all(reference[:reference_leading] == 0):
            raise ValueError(f"reference leading silence is not zero: {reference_path}")

        crop = capture[trim_start:trim_stop]
        capture_packet_index = parse_int(row["capture_packet_index"])
        # 2 MS/s 偶/奇 ADC 相位拆成两个等长 1 MS/s 文件。
        for adc_phase in (0, 1):
            ota = crop[adc_phase::2]
            if ota.size != reference_samples:
                raise RuntimeError("polyphase output length changed unexpectedly.")
            stem = (
                f"exp{descriptor.experiment_id}_{capture_packet_index:06d}"
                f"_rxg{gain_token(descriptor.rx_gain_db)}_{adc_phase}_fulltrim"
            )
            ota_path = root / "ota" / f"{stem}.cfile"
            metadata_path = root / "metadata" / f"{stem}.json"
            write_complex_file(ota_path, ota, overwrite=overwrite)

            leading = np.asarray(ota[:reference_leading])
            active = np.asarray(ota[reference_leading:])
            leading_power = float(np.mean(np.abs(leading) ** 2, dtype=np.float64))
            active_power = float(np.mean(np.abs(active) ** 2, dtype=np.float64))
            # metadata 保存从原始 capture 到 OTA/reference 的完整证据链。
            metadata = {
                "schema": OTA_SCHEMA,
                "schema_version": OTA_SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "ota_id": stem,
                "physical_packet_uid": row["physical_packet_uid"],
                "capture_packet_index": capture_packet_index,
                "split_group": row["physical_packet_uid"],
                "capture": {
                    "capture_uid": descriptor.capture_uid,
                    "source_path": display_path(capture_path),
                    "source_sha256": capture_hash,
                    "source_bytes": capture_path.stat().st_size,
                    "source_sample_rate_hz": descriptor.sample_rate_hz,
                    "center_frequency_hz": descriptor.center_frequency_hz,
                    "rf_bandwidth_hz": descriptor.rf_bandwidth_hz,
                    "rx_gain_db": descriptor.rx_gain_db,
                    "rx_gain_mode": "manual",
                    "agc_enabled": descriptor.agc_enabled,
                    "antenna": descriptor.antenna,
                    "experiment_id": descriptor.experiment_id,
                    "session_id": descriptor.session_id,
                    "location_id": descriptor.location_id,
                    "condition": descriptor.condition,
                    "run_id": descriptor.run_id,
                },
                "packet": {
                    "decoded_frame_hex": row.get("decoded_frame_hex", ""),
                    "expected_frame_hex": catalog_row["frame_hex"],
                    "crc_valid": parse_bool(row.get("crc_valid", "")),
                },
                "reference": {
                    "reference_id": reference_id,
                    "relative_path": reference_path.relative_to(root).as_posix(),
                    "sha256": catalog_row["reference_sha256"],
                    "sample_rate_hz": reference_rate,
                    "complex_samples": reference_samples,
                    "leading_zero_samples": reference_leading,
                },
                "association": {
                    "method": row["association_method"],
                    "confidence": float(row["association_confidence"]),
                    "catalog_frame_exact": parse_bool(row["catalog_frame_exact"]),
                    "sequence_segment": parse_int(row["sequence_segment"]),
                    "schedule_slot": parse_int(row["schedule_slot"]),
                    "previous_anchor_detection": row.get(
                        "previous_anchor_detection", ""
                    ),
                    "next_anchor_detection": row.get(
                        "next_anchor_detection", ""
                    ),
                },
                "trim": {
                    "detected_preamble_start_sample_2m": start_packet,
                    "start_sample_2m": trim_start,
                    "stop_sample_2m_exclusive": trim_stop,
                    "complex_samples_2m": source_crop_samples,
                    "leading_off_packet_samples_2m": source_leading,
                    "timing_residual_samples_2m": row.get(
                        "timing_residual_samples", ""
                    ),
                },
                "ota": {
                    "relative_path": ota_path.relative_to(root).as_posix(),
                    "dtype": "<c8",
                    "bytes": ota_path.stat().st_size,
                    "complex_samples": reference_samples,
                    "sha256": sha256_file(ota_path),
                    "leading_real_off_packet_samples": reference_leading,
                },
                "view": {
                    "sample_rate_hz": reference_rate,
                    "complex_samples": reference_samples,
                    "adc_phase_2m": adc_phase,
                    "adc_phase_timing_offset_samples_1m": adc_phase / 2.0,
                    "available_lowrate_phases_1m": [0, 1, 2, 3],
                    "combined_decimation_phases_2m": [
                        adc_phase + 2 * phase for phase in range(4)
                    ],
                    "lowrate_sample_rate_hz": reference_rate // 4,
                    "lowrate_samples": lowrate_samples,
                    "lowrate_leading_off_packet_samples": reference_leading // 4,
                    "anti_alias_filter_applied": False,
                },
                "alignment": {
                    "status": "packet_boundary_aligned",
                    "cfo_correction_applied": False,
                    "complex_gain_correction_applied": False,
                    "amplitude_normalization_applied": False,
                    "estimated_cfo_hz": row.get("estimated_cfo_hz", ""),
                    "grlora_snr_db": row.get("grlora_snr_db", ""),
                    "sto": row.get("sto", ""),
                    "sfo": row.get("sfo", ""),
                    "estimated_complex_gain_real": "",
                    "estimated_complex_gain_imag": "",
                },
                "quality_control": {
                    "finite_samples": bool(np.all(np.isfinite(ota))),
                    "leading_off_packet_is_real": bool(np.any(np.abs(leading) > 0)),
                    "leading_off_packet_power": leading_power,
                    "active_packet_power": active_power,
                    "reference_leading_samples_are_zero": True,
                    "expected_output_samples": reference_samples,
                },
            }
            write_json(metadata_path, metadata)
            generated.append(ota_path)

    write_csv(
        capture_manifest_dir(root, descriptor) / "trim_issues.csv",
        trim_issues,
        (
            "physical_packet_uid",
            "capture_packet_index",
            "start_sample_2m",
            "trim_start_2m",
            "trim_stop_2m",
            "reason",
        ),
    )
    rebuild_views_manifest(root)
    if not generated:
        raise RuntimeError(
            "trim produced no OTA files; inspect packets.csv and trim_issues.csv."
        )
    return generated


def validate_dataset(output_root: str | Path) -> Path:
    """阶段 5：逐文件验证长度、前缀、多相视图和源 IQ 重建。

    核心检查包括：

    * OTA/reference 文件存在且没有逃出数据库根目录；
    * 两者都是预期长度，reference 前缀为零而 OTA 前缀是真实环境 IQ；
    * q=0..3 四个 250 kS/s 视图长度一致；
    * OTA phase 能逐样点还原到原始 2 MS/s crop 的偶/奇相位；
    * IQ 不包含 NaN/Inf。

    输出 ``manifests/validation.csv`` 和 ``validation_summary.json``。
    """

    root = ensure_inside_data(output_root, label="dataset output root")
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted((root / "metadata").glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != OTA_SCHEMA:
            continue
        issues: list[str] = []
        warnings: list[str] = []
        ota_path = (root / metadata["ota"]["relative_path"]).resolve()
        reference_path = (root / metadata["reference"]["relative_path"]).resolve()
        for path, label in ((ota_path, "OTA"), (reference_path, "reference")):
            try:
                path.relative_to(root)
            except ValueError:
                issues.append(f"{label} path escapes dataset root")
        if not ota_path.is_file():
            issues.append("OTA file missing")
        if not reference_path.is_file():
            issues.append("reference file missing")
        if issues:
            rows.append(
                {
                    "ota_id": metadata.get("ota_id", metadata_path.stem),
                    "valid": 0,
                    "issues": "; ".join(issues),
                    "warnings": "; ".join(warnings),
                }
            )
            continue

        ota = np.memmap(ota_path, dtype=COMPLEX_DTYPE, mode="r")
        reference = np.memmap(reference_path, dtype=COMPLEX_DTYPE, mode="r")
        expected = int(metadata["view"]["complex_samples"])
        leading = int(metadata["reference"]["leading_zero_samples"])
        if ota.size != expected:
            issues.append(f"OTA samples {ota.size} != {expected}")
        if reference.size != expected:
            issues.append(f"reference samples {reference.size} != {expected}")
        if reference.size >= leading and not np.all(reference[:leading] == 0):
            issues.append("reference leading silence is not zero")
        if ota.size >= leading and not np.any(np.abs(ota[:leading]) > 0):
            issues.append("OTA leading off-packet samples are all zero")
        if not np.all(np.isfinite(ota)):
            issues.append("OTA contains non-finite samples")
        for phase in metadata["view"]["available_lowrate_phases_1m"]:
            if ota[int(phase) :: 4].size != int(metadata["view"]["lowrate_samples"]):
                issues.append(f"low-rate phase {phase} has the wrong length")

        # 原始 capture 仍在项目内时做最严格的逐样点重建；服务器若只保留
        # processed 数据，则记录 warning 而不是伪造通过。
        source_text = metadata["capture"]["source_path"]
        source_path = (
            (REPO_ROOT / source_text).resolve()
            if not Path(source_text).is_absolute()
            else Path(source_text).resolve()
        )
        if source_path.is_file():
            source = np.memmap(source_path, dtype=COMPLEX_DTYPE, mode="r")
            trim_start = int(metadata["trim"]["start_sample_2m"])
            trim_stop = int(metadata["trim"]["stop_sample_2m_exclusive"])
            adc_phase = int(metadata["view"]["adc_phase_2m"])
            source_phase = source[trim_start + adc_phase : trim_stop : 2]
            if source_phase.size != ota.size or not np.array_equal(source_phase, ota):
                issues.append("OTA phase does not reconstruct the source crop")
        else:
            warnings.append("source capture unavailable; source reconstruction skipped")

        rows.append(
            {
                "ota_id": metadata["ota_id"],
                "valid": int(not issues),
                "issues": "; ".join(issues),
                "warnings": "; ".join(warnings),
            }
        )

    validation_path = root / "manifests" / "validation.csv"
    write_csv(validation_path, rows, ("ota_id", "valid", "issues", "warnings"))
    write_json(
        root / "manifests" / "validation_summary.json",
        {
            "schema": "lora-rfsr-dataset-validation",
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "dataset_root": display_path(root),
            "ota_views_checked": len(rows),
            "valid": sum(parse_bool(row["valid"]) for row in rows),
            "invalid": sum(not parse_bool(row["valid"]) for row in rows),
            "warnings": sum(bool(row["warnings"]) for row in rows),
        },
    )
    return validation_path


def add_descriptor_arguments(parser: argparse.ArgumentParser) -> None:
    """给需要读取 capture 的子命令添加统一实验参数。"""

    parser.add_argument("--experiment-id", type=int, help="实验编号。")
    parser.add_argument("--session-id", type=int, help="采集 session 编号。")
    parser.add_argument("--location-id", type=str, help="位置短标签，例如 lab1。")
    parser.add_argument(
        "--condition",
        type=str,
        help="采集条件，例如 highsnr、lowsnr 或 interference。",
    )
    parser.add_argument("--run-id", type=int, help="同条件下的重复轮次。")
    parser.add_argument("--sf", type=int, help="LoRa 扩频因子，7..12。")
    parser.add_argument("--bandwidth-hz", type=int, help="LoRa 带宽，单位 Hz。")
    parser.add_argument(
        "--sample-rate-hz", type=int, help="原始 capture 采样率，单位 sample/s。"
    )
    parser.add_argument(
        "--preamble-symbols", type=int, help="发射端前导码 symbol 数。"
    )
    parser.add_argument(
        "--sync-word",
        type=lambda value: int(value, 0),
        help="LoRa sync word，支持十进制或 0x12。",
    )
    parser.add_argument("--cr", type=int, help="编码率索引：1..4 对应 4/5..4/8。")
    parser.add_argument(
        "--phy-crc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="PHY 是否携带 CRC；可用 --no-phy-crc 关闭。",
    )
    parser.add_argument(
        "--center-frequency-hz", type=int, help="USRP 中心频率，单位 Hz。"
    )
    parser.add_argument("--rx-gain-db", type=float, help="USRP 手动接收增益，单位 dB。")
    parser.add_argument("--pay-len", type=int, help="空口 PHY payload 字节数。")
    parser.add_argument(
        "--ldro-mode", type=int, help="低数据率优化：0 关、1 开、2 自动。"
    )
    parser.add_argument(
        "--crc-mode",
        choices=("grlora", "sx1276"),
        help="接收链使用的 CRC 兼容模式。",
    )
    parser.add_argument(
        "--rf-bandwidth-hz", type=int, help="USRP 模拟前端带宽，单位 Hz。"
    )
    parser.add_argument("--antenna", type=str, help="USRP 接收端口，例如 RX2。")
    parser.add_argument(
        "--agc-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="记录是否启用 AGC；正式实验通常关闭。",
    )
    parser.add_argument("--device-args", type=str, help="UHD device args 留档。")


def add_output_root(parser: argparse.ArgumentParser) -> None:
    """添加最终数据库根目录参数，并统一默认到 reference_phy/rfsr_db。"""

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "最终 OTA 数据库根目录，必须位于 lora-rfsr-savaux/data 内。"
            f"默认：{DEFAULT_OUTPUT_ROOT}"
        ),
    )


def add_capture(parser: argparse.ArgumentParser) -> None:
    """添加连续 IQ 路径和可选的采集参数覆盖项。"""

    parser.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="规范命名的连续 2 MS/s complex64 .cfile。",
    )
    add_descriptor_arguments(parser)


def create_parser() -> argparse.ArgumentParser:
    """创建中文命令行界面；每个子命令对应一个独立处理阶段。"""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init-capture",
        help="阶段 0：生成规范原始文件名和 JSON sidecar。",
    )
    init_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_CAPTURE_DIR,
        help=f"原始 IQ 目录，默认：{DEFAULT_RAW_CAPTURE_DIR}",
    )
    add_descriptor_arguments(init_parser)
    init_parser.set_defaults(
        experiment_id=0,
        session_id=0,
        location_id="lab1",
        condition="highsnr",
        run_id=0,
        sf=12,
        bandwidth_hz=125_000,
        sample_rate_hz=2_000_000,
        preamble_symbols=16,
        sync_word=0x12,
        cr=4,
        phy_crc=True,
        center_frequency_hz=487_700_000,
        rx_gain_db=20.0,
        pay_len=33,
        ldro_mode=1,
        crc_mode="grlora",
        rf_bandwidth_hz=1_000_000,
        antenna="RX2",
        agc_enabled=False,
        device_args="",
    )
    init_parser.add_argument(
        "--overwrite", action="store_true", help="允许覆盖已有 sidecar。"
    )

    catalog_parser = subparsers.add_parser(
        "catalog",
        help="阶段 1：建立 UART ID 到理想 reference 的目录表。",
    )
    catalog_parser.add_argument(
        "--uart-log",
        type=Path,
        default=DEFAULT_UART_LOG,
        help=f"STM32 串口 ground truth，默认：{DEFAULT_UART_LOG}",
    )
    catalog_parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help=f"理想 PHY reference 语料根目录，默认：{DEFAULT_REFERENCE_ROOT}",
    )
    add_output_root(catalog_parser)

    detect_parser = subparsers.add_parser(
        "detect",
        help="阶段 2：GNU Radio 扫描连续 IQ，输出包位置和 CRC。",
    )
    add_capture(detect_parser)
    add_output_root(detect_parser)
    detect_parser.add_argument(
        "--overwrite", action="store_true", help="重新运行检测并覆盖 detections.csv。"
    )

    associate_parser = subparsers.add_parser(
        "associate",
        help="阶段 3：把检测位置关联到 UART reference_id。",
    )
    add_capture(associate_parser)
    add_output_root(associate_parser)
    associate_parser.add_argument(
        "--catalog", type=Path, help="reference_catalog.csv；默认自动定位。"
    )
    associate_parser.add_argument(
        "--detections", type=Path, help="detections.csv；默认按 capture_uid 定位。"
    )
    associate_parser.add_argument(
        "--usrp-log", type=Path, help="可选 GNU Radio 终端日志，仅用于审计。"
    )
    associate_parser.add_argument(
        "--recover-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否对时序中缺失的 slot 做局部 XCopy 搜索。",
    )

    trim_parser = subparsers.add_parser(
        "trim",
        help="阶段 4：为 accepted packet 写两个 1 MS/s fulltrim phase。",
    )
    add_capture(trim_parser)
    add_output_root(trim_parser)
    trim_parser.add_argument(
        "--catalog", type=Path, help="reference_catalog.csv；默认自动定位。"
    )
    trim_parser.add_argument(
        "--packets", type=Path, help="packets.csv；默认按 capture_uid 定位。"
    )
    trim_parser.add_argument(
        "--reference-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="reference 放入数据库的方式；auto 优先硬链接，失败再复制。",
    )
    trim_parser.add_argument(
        "--hash-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否计算原始大 cfile 的 SHA-256。",
    )
    trim_parser.add_argument(
        "--overwrite", action="store_true", help="允许覆盖已有 OTA/metadata。"
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="阶段 5：验证文件、长度、多相视图和源 IQ 重建。",
    )
    add_output_root(validate_parser)

    all_parser = subparsers.add_parser(
        "all",
        help="采集机一键执行 catalog→detect→associate→trim→validate。",
    )
    add_capture(all_parser)
    add_output_root(all_parser)
    all_parser.add_argument(
        "--uart-log", type=Path, default=DEFAULT_UART_LOG,
        help="STM32 串口 ground truth。",
    )
    all_parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="理想 PHY reference 语料根目录。",
    )
    all_parser.add_argument(
        "--usrp-log", type=Path, help="可选 GNU Radio 终端日志。"
    )
    all_parser.add_argument(
        "--recover-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否局部搜索全局 detector 漏掉的时隙。",
    )
    all_parser.add_argument(
        "--reference-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="reference 放入最终数据库的方式。",
    )
    all_parser.add_argument(
        "--hash-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否计算原始 capture SHA-256。",
    )
    all_parser.add_argument(
        "--overwrite", action="store_true", help="覆盖已有阶段输出。"
    )

    server_parser = subparsers.add_parser(
        "server",
        help=(
            "服务器执行 catalog→associate→trim→validate；"
            "必须已有 detections.csv，绝不导入 GNU Radio。"
        ),
    )
    add_capture(server_parser)
    add_output_root(server_parser)
    server_parser.add_argument(
        "--uart-log", type=Path, default=DEFAULT_UART_LOG,
        help="STM32 串口 ground truth。",
    )
    server_parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="理想 PHY reference 语料根目录。",
    )
    server_parser.add_argument(
        "--detections", type=Path, help="采集机预先生成的 detections.csv。"
    )
    server_parser.add_argument(
        "--usrp-log", type=Path, help="可选 GNU Radio 终端日志。"
    )
    server_parser.add_argument(
        "--recover-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否局部搜索全局 detector 漏掉的时隙。",
    )
    server_parser.add_argument(
        "--reference-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="reference 放入最终数据库的方式。",
    )
    server_parser.add_argument(
        "--hash-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否计算原始 capture SHA-256。",
    )
    server_parser.add_argument(
        "--overwrite", action="store_true", help="覆盖已有阶段输出。"
    )
    return parser


def resolved_catalog(args: argparse.Namespace, output_root: Path) -> Path:
    """优先使用 CLI catalog 路径，否则使用数据库内默认路径。"""

    value = getattr(args, "catalog", None)
    return Path(value) if value else output_root / "manifests" / "reference_catalog.csv"


def resolved_detections(
    args: argparse.Namespace,
    output_root: Path,
    descriptor: CaptureDescriptor,
) -> Path:
    """优先使用 CLI detections 路径，否则按 capture_uid 自动定位。"""

    value = getattr(args, "detections", None)
    return (
        Path(value)
        if value
        else capture_manifest_dir(output_root, descriptor) / "detections.csv"
    )


def resolved_packets(
    args: argparse.Namespace,
    output_root: Path,
    descriptor: CaptureDescriptor,
) -> Path:
    """优先使用 CLI packets 路径，否则按 capture_uid 自动定位。"""

    value = getattr(args, "packets", None)
    return (
        Path(value)
        if value
        else capture_manifest_dir(output_root, descriptor) / "packets.csv"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """解析子命令并串联对应阶段；每个阶段均可独立重复运行。"""

    args = create_parser().parse_args(argv)
    if args.command == "init-capture":
        initialize_capture(args)
        return 0

    output_root = ensure_inside_data(args.output_root, label="dataset output root")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "catalog":
        path = build_reference_catalog(
            args.uart_log,
            args.reference_root,
            output_root,
        )
        print(path)
        return 0
    if args.command == "validate":
        path = validate_dataset(output_root)
        print(path)
        return 0

    descriptor = load_capture_descriptor(args.capture, args)
    capture = Path(args.capture).expanduser().resolve()
    if args.command == "detect":
        print(
            run_detection(
                capture,
                descriptor,
                output_root,
                overwrite=args.overwrite,
            )
        )
        return 0
    if args.command == "associate":
        print(
            associate_packets(
                capture,
                descriptor,
                resolved_catalog(args, output_root),
                resolved_detections(args, output_root, descriptor),
                output_root,
                recover_missing=args.recover_missing,
                usrp_log=args.usrp_log,
            )
        )
        return 0
    if args.command == "trim":
        generated = trim_packets(
            capture,
            descriptor,
            resolved_catalog(args, output_root),
            resolved_packets(args, output_root, descriptor),
            output_root,
            overwrite=args.overwrite,
            reference_mode=args.reference_mode,
            compute_capture_hash=args.hash_capture,
        )
        print(f"generated {len(generated)} OTA phase files")
        return 0
    if args.command == "all":
        catalog = build_reference_catalog(
            args.uart_log,
            args.reference_root,
            output_root,
        )
        detections = run_detection(
            capture,
            descriptor,
            output_root,
            overwrite=args.overwrite,
        )
        packets = associate_packets(
            capture,
            descriptor,
            catalog,
            detections,
            output_root,
            recover_missing=args.recover_missing,
            usrp_log=args.usrp_log,
        )
        generated = trim_packets(
            capture,
            descriptor,
            catalog,
            packets,
            output_root,
            overwrite=args.overwrite,
            reference_mode=args.reference_mode,
            compute_capture_hash=args.hash_capture,
        )
        validation = validate_dataset(output_root)
        print(f"generated {len(generated)} OTA phase files")
        print(validation)
        return 0
    if args.command == "server":
        catalog = build_reference_catalog(
            args.uart_log,
            args.reference_root,
            output_root,
        )
        detections = resolved_detections(args, output_root, descriptor)
        if not detections.is_file():
            raise FileNotFoundError(
                "server mode requires a precomputed detections.csv inside "
                f"the copied project: {detections}. Run detect on the "
                "acquisition machine first."
            )
        packets = associate_packets(
            capture,
            descriptor,
            catalog,
            detections,
            output_root,
            recover_missing=args.recover_missing,
            usrp_log=args.usrp_log,
        )
        generated = trim_packets(
            capture,
            descriptor,
            catalog,
            packets,
            output_root,
            overwrite=args.overwrite,
            reference_mode=args.reference_mode,
            compute_capture_hash=args.hash_capture,
        )
        validation = validate_dataset(output_root)
        print(f"generated or verified {len(generated)} OTA phase files")
        print(validation)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
