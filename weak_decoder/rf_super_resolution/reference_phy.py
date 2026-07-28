"""从 SX1276 raw PHY payload 生成可审计的理想 LoRa reference。

RF-SR 上游 ``PHY.encode`` 会添加私有四字节 ``DST/SRC/SEQ/LENGTH`` 包装，而
STM32 UART ``[TX Frame]`` 已经是传给 ``Radio.Send`` 的完整 33 字节，因此这里
不会再次包装。输出波形为：

    RF-SR 前置静默 + preamble + sync word + SFD
    + explicit header + encoded payload/CRC

默认前置 10,000 个零样本以对齐 RF-SR ``PHY.encode`` 的合成数据契约，不添加
后置静默、人工 CFO、AWGN、接收增益或信道响应。与 OTA IQ 做逐样本比较前仍须
完成 timing/CFO/复增益对齐。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RFSR_ROOT = REPO_ROOT / "third_party" / "rfsr"
if str(RFSR_ROOT) not in sys.path:
    sys.path.insert(0, str(RFSR_ROOT))

from weak_decoder.decoding.payload_codec import decode_explicit_frame_symbols


REFERENCE_SCHEMA_VERSION = 2
_HEX_BYTE_RE = re.compile(r"^[0-9A-Fa-f]{2}$")
_PAYLOAD_RE = re.compile(
    r"^\[TX Payload\]\s+seq=(\d+)\s+round=(\d+)\s+id=(\d+)"
    r"\s+app_len=(\d+)\s+data=(.+)$"
)
_FRAME_RE = re.compile(
    r"^\[TX Frame\]\s+seq=(\d+)\s+id=(\d+)"
    r"\s+frame_len=(\d+)\s+data=(.+)$"
)


def _portable_source_path(path: str | Path) -> str:
    """Prefer a repository-relative provenance path when one is available."""

    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


_FIELDS_RE = re.compile(r"^\[TX Frame Fields\]\s+(.+)$")


@dataclass(frozen=True)
class ReferencePhyConfig:
    """理想 complex-baseband reference 的 PHY 与文件封装参数。"""

    sample_rate_hz: int = 1_000_000
    bandwidth_hz: int = 125_000
    sf: int = 12
    cr: int = 4
    preamble_symbols: int = 16
    sync_word: int = 0x12
    explicit_header: bool = True
    phy_crc: bool = True
    crc_mode: str = "grlora"
    ldro: bool = True
    leading_silence_samples: int = 10_000
    trailing_silence_samples: int = 0

    def validate(self) -> None:
        if not 7 <= int(self.sf) <= 12:
            raise ValueError(f"SF must be in [7, 12], got {self.sf}.")
        if int(self.cr) not in {1, 2, 3, 4}:
            raise ValueError(f"CR must be 1..4 (4/5..4/8), got {self.cr}.")
        if int(self.bandwidth_hz) <= 0 or int(self.sample_rate_hz) <= 0:
            raise ValueError("sample rate and bandwidth must be positive.")
        if int(self.sample_rate_hz) % int(self.bandwidth_hz) != 0:
            raise ValueError(
                "sample rate must be an integer multiple of LoRa bandwidth; "
                f"got {self.sample_rate_hz}/{self.bandwidth_hz}."
            )
        if int(self.preamble_symbols) < 5:
            raise ValueError("preamble must contain at least five symbols.")
        if int(self.leading_silence_samples) < 0:
            raise ValueError("leading silence samples must be non-negative.")
        if int(self.trailing_silence_samples) < 0:
            raise ValueError("trailing silence samples must be non-negative.")
        if not 0 <= int(self.sync_word) <= 0xFF:
            raise ValueError(f"sync word must fit uint8, got {self.sync_word}.")
        if not bool(self.explicit_header):
            raise ValueError("reference generator currently supports explicit header only.")
        if str(self.crc_mode).lower() not in {"grlora", "sx1276"}:
            raise ValueError(
                "crc_mode must be 'grlora' (validated SX1276-air convention) "
                "or 'sx1276' (full-payload CRC-16 comparison mode)."
            )

    @property
    def os_factor(self) -> int:
        self.validate()
        return int(self.sample_rate_hz) // int(self.bandwidth_hz)

    @property
    def samples_per_symbol(self) -> int:
        return (1 << int(self.sf)) * self.os_factor


@dataclass(frozen=True)
class UartPacketRecord:
    """One complete UART payload/frame/field triplet."""

    seq: int
    round: int
    payload_id: int
    app_payload: bytes
    frame: bytes
    fields: dict[str, str]


@dataclass(frozen=True)
class UartReferenceLog:
    """Parsed UART manifest plus the one-time PHY/config records."""

    source_path: Path
    source_sha256: str
    packets: tuple[UartPacketRecord, ...]
    phy: dict[str, str]
    reference_config: dict[str, str]
    phy_registers: dict[str, dict[str, str]]

    def first_reference_cycle(self) -> tuple[UartPacketRecord, ...]:
        """Return exactly one complete, ID-ordered reference cycle."""

        variants_text = self.reference_config.get("variants", "")
        if not variants_text:
            raise ValueError("[TX Reference Config] does not contain variants.")
        variants = int(variants_text, 0)
        first_round = [packet for packet in self.packets if int(packet.round) == 0]
        by_id: dict[int, UartPacketRecord] = {}
        for packet in first_round:
            if packet.payload_id in by_id:
                raise ValueError(
                    f"duplicate payload id {packet.payload_id} in UART round 0."
                )
            by_id[packet.payload_id] = packet
        expected_ids = set(range(variants))
        actual_ids = set(by_id)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise ValueError(
                "UART round 0 is not a complete reference cycle: "
                f"missing={missing}, extra={extra}."
            )
        return tuple(by_id[payload_id] for payload_id in range(variants))


@dataclass(frozen=True)
class EncodedReference:
    """Generated IQ plus the exact encoded LoRa symbol IDs."""

    samples: np.ndarray
    header_symbol_ids: tuple[int, ...]
    payload_symbol_ids: tuple[int, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in text.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[str(key)] = str(value)
    return values


def _parse_hex_bytes(text: str, expected_len: int, context: str) -> bytes:
    tokens = text.strip().split()
    invalid = [token for token in tokens if not _HEX_BYTE_RE.fullmatch(token)]
    if invalid:
        raise ValueError(f"{context} contains invalid hex bytes: {invalid}.")
    data = bytes(int(token, 16) for token in tokens)
    if len(data) != int(expected_len):
        raise ValueError(
            f"{context} declares {expected_len} bytes but contains {len(data)}."
        )
    return data


def parse_uart_reference_log(path: str | Path) -> UartReferenceLog:
    """Parse and strictly validate STM32 ``[TX ...]`` UART records."""

    source_path = Path(path).expanduser().resolve()
    lines = source_path.read_text(encoding="utf-8").splitlines()
    payloads: dict[int, tuple[int, int, bytes]] = {}
    frames: dict[int, tuple[int, bytes]] = {}
    fields: dict[int, dict[str, str]] = {}
    phy: dict[str, str] = {}
    reference_config: dict[str, str] = {}
    phy_registers: dict[str, dict[str, str]] = {}

    for line_number, line in enumerate(lines, start=1):
        payload_match = _PAYLOAD_RE.match(line)
        if payload_match:
            seq, round_index, payload_id, app_len = (
                int(payload_match.group(index)) for index in range(1, 5)
            )
            if seq in payloads:
                raise ValueError(f"duplicate [TX Payload] seq={seq} at line {line_number}.")
            payloads[seq] = (
                round_index,
                payload_id,
                _parse_hex_bytes(
                    payload_match.group(5),
                    app_len,
                    f"[TX Payload] seq={seq}",
                ),
            )
            continue

        frame_match = _FRAME_RE.match(line)
        if frame_match:
            seq, payload_id, frame_len = (
                int(frame_match.group(index)) for index in range(1, 4)
            )
            if seq in frames:
                raise ValueError(f"duplicate [TX Frame] seq={seq} at line {line_number}.")
            frames[seq] = (
                payload_id,
                _parse_hex_bytes(
                    frame_match.group(4),
                    frame_len,
                    f"[TX Frame] seq={seq}",
                ),
            )
            continue

        fields_match = _FIELDS_RE.match(line)
        if fields_match:
            row = _parse_key_values(fields_match.group(1))
            if "seq" not in row:
                raise ValueError(f"[TX Frame Fields] lacks seq at line {line_number}.")
            seq = int(row["seq"], 0)
            if seq in fields:
                raise ValueError(
                    f"duplicate [TX Frame Fields] seq={seq} at line {line_number}."
                )
            fields[seq] = row
            continue

        if line.startswith("[TX PHY]"):
            phy = _parse_key_values(line[len("[TX PHY]"):])
        elif line.startswith("[TX Reference Config]"):
            reference_config = _parse_key_values(
                line[len("[TX Reference Config]"):]
            )
        elif line.startswith("[TX PHY Reg "):
            prefix, _, body = line.partition("]")
            phy_registers[prefix[1:]] = _parse_key_values(body)

    seqs = sorted(set(payloads) | set(frames) | set(fields))
    packets: list[UartPacketRecord] = []
    for seq in seqs:
        if seq not in payloads or seq not in frames or seq not in fields:
            raise ValueError(f"incomplete UART triplet for seq={seq}.")
        round_index, payload_id, app_payload = payloads[seq]
        frame_id, frame = frames[seq]
        field_row = fields[seq]
        field_id = int(field_row.get("id", "-1"), 0)
        if payload_id != frame_id or payload_id != field_id:
            raise ValueError(f"payload/frame/fields ID mismatch for seq={seq}.")
        if int(field_row.get("radio_send_len", "-1"), 0) != len(frame):
            raise ValueError(f"radio_send_len mismatch for seq={seq}.")
        if int(field_row.get("app_len", "-1"), 0) != len(app_payload):
            raise ValueError(f"app_len mismatch for seq={seq}.")
        if frame.find(app_payload) < 0:
            raise ValueError(f"application payload is not embedded in frame seq={seq}.")
        packets.append(
            UartPacketRecord(
                seq=seq,
                round=round_index,
                payload_id=payload_id,
                app_payload=app_payload,
                frame=frame,
                fields=field_row,
            )
        )

    if not packets:
        raise ValueError("UART log contains no complete packet records.")
    if not phy:
        raise ValueError("UART log contains no [TX PHY] record.")
    if not reference_config:
        raise ValueError("UART log contains no [TX Reference Config] record.")

    parsed = UartReferenceLog(
        source_path=source_path,
        source_sha256=_sha256_file(source_path),
        packets=tuple(packets),
        phy=phy,
        reference_config=reference_config,
        phy_registers=phy_registers,
    )
    parsed.first_reference_cycle()
    return parsed


def phy_config_from_uart(
    uart_log: UartReferenceLog,
    sample_rate_hz: int = 1_000_000,
    leading_silence_samples: int = 10_000,
) -> ReferencePhyConfig:
    """根据 UART 实测 PHY 字段构造 reference 配置。"""

    phy = uart_log.phy
    cr_text = str(phy["cr"])
    cr_parts = cr_text.split("/", 1)
    if len(cr_parts) != 2 or int(cr_parts[0]) != 4:
        raise ValueError(f"unsupported UART CR representation: {cr_text!r}.")
    header_text = str(phy.get("header", "")).lower()
    config = ReferencePhyConfig(
        sample_rate_hz=int(sample_rate_hz),
        bandwidth_hz=int(phy["bw_hz"], 0),
        sf=int(phy["sf"], 0),
        cr=int(cr_parts[1], 0) - 4,
        preamble_symbols=int(phy["preamble_symbols"], 0),
        sync_word=int(phy["syncword"], 0),
        explicit_header=header_text == "explicit",
        phy_crc=bool(int(phy["phy_crc"], 0)),
        # Existing LoraSTMac/SX1276 captures in this workspace validate with
        # gr-lora_sdr crc_mode=0.  Its historical name is "grlora", while the
        # crc_mode=1 "sx1276" branch is the full-payload CRC-16 comparison
        # mode and is not the default used for these hardware packets.
        crc_mode="grlora",
        ldro=bool(int(phy["ldro"], 0)),
        leading_silence_samples=int(leading_silence_samples),
        trailing_silence_samples=0,
    )
    config.validate()
    if bool(int(phy.get("iq_inverted", "0"), 0)):
        raise ValueError("IQ-inverted reference generation is not implemented.")
    if bool(int(phy.get("freq_hop", "0"), 0)):
        raise ValueError("frequency-hopping reference generation is not implemented.")
    return config


def _modulator_to_demod_symbols(
    symbol_ids: Iterable[int],
    sf: int,
    reduced_rate: bool,
) -> list[int]:
    n_bins = 1 << int(sf)
    divisor = 4 if bool(reduced_rate) else 1
    return [
        ((int(symbol_id) - 1) % n_bins) // divisor
        for symbol_id in symbol_ids
    ]


def encode_reference_phy(
    frame: bytes | bytearray | Iterable[int],
    config: ReferencePhyConfig,
) -> EncodedReference:
    """把一个完整 raw PHY payload 编码成 RF-SR 兼容的 ``complex64`` IQ。"""

    # UART catalog parsing and OTA trimming do not require PyTorch.  Import the
    # RF-SR waveform backend only when a new ideal reference is actually encoded.
    from rfsr.PHY import encode_raw_phy

    config.validate()
    frame_bytes = bytes(int(value) & 0xFF for value in frame)
    if not 1 <= len(frame_bytes) <= 255:
        raise ValueError(f"PHY payload length must be 1..255, got {len(frame_bytes)}.")

    raw_encoding = encode_raw_phy(
        frame_bytes,
        int(config.sample_rate_hz),
        SF=int(config.sf),
        BW=int(config.bandwidth_hz),
        cr=int(config.cr),
        enable_crc=bool(config.phy_crc),
        implicit_header=0,
        preamble_bits=int(config.preamble_symbols),
        sync_word=int(config.sync_word),
        ldro=bool(config.ldro),
        crc_mode=str(config.crc_mode),
        leading_silence_samples=int(config.leading_silence_samples),
        trailing_silence_samples=int(config.trailing_silence_samples),
    )
    header_symbols = raw_encoding.header_symbol_ids
    payload_symbols = raw_encoding.payload_symbol_ids

    # Internal round trip catches header/CRC/whitening/interleaver regressions
    # before any large cfile is written.
    header_demod = _modulator_to_demod_symbols(
        header_symbols,
        sf=int(config.sf),
        reduced_rate=True,
    )
    payload_demod = _modulator_to_demod_symbols(
        payload_symbols,
        sf=int(config.sf),
        reduced_rate=bool(config.ldro),
    )
    decoded = decode_explicit_frame_symbols(
        header_demod,
        payload_demod,
        sf=int(config.sf),
        bw=float(config.bandwidth_hz),
        ldro_mode=1 if bool(config.ldro) else 0,
        crc_mode=str(config.crc_mode),
    )
    if not decoded.header.header_valid:
        raise RuntimeError("generated explicit PHY header failed internal decode.")
    if decoded.payload.payload_bytes != frame_bytes:
        raise RuntimeError("generated PHY symbols did not round-trip to the raw frame.")
    if bool(config.phy_crc) and not decoded.payload.crc_valid:
        raise RuntimeError(
            f"generated PHY CRC failed internal {config.crc_mode} verification."
        )

    return EncodedReference(
        samples=raw_encoding.samples,
        header_symbol_ids=tuple(int(value) for value in header_symbols),
        payload_symbol_ids=tuple(int(value) for value in payload_symbols),
    )


def reference_metadata(
    uart_log: UartReferenceLog,
    packet: UartPacketRecord,
    config: ReferencePhyConfig,
    encoded: EncodedReference,
    output_root: Path,
    iq_path: Path,
    iq_sha256: str,
) -> dict[str, Any]:
    """Build the auditable JSON sidecar for one ideal reference."""

    generator_path = Path(__file__).resolve()
    phy_backend_path = RFSR_ROOT / "rfsr" / "PHY.py"
    return {
        "schema": "lora-rfsr-reference",
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "reference_kind": "ideal_tx_complex_baseband",
        "alignment_status": "not_aligned_to_ota",
        "packet": {
            "seq": int(packet.seq),
            "round": int(packet.round),
            "payload_id": int(packet.payload_id),
            "app_payload_hex": packet.app_payload.hex(" ").upper(),
            "frame_hex": packet.frame.hex(" ").upper(),
            "frame_bytes": len(packet.frame),
            "uart_fields": dict(packet.fields),
        },
        "phy": {
            **asdict(config),
            "cr_text": f"4/{int(config.cr) + 4}",
            "sync_word_hex": f"0x{int(config.sync_word):02X}",
            "os_factor": int(config.os_factor),
            "samples_per_symbol": int(config.samples_per_symbol),
            "source_configured_freq_hz": int(
                uart_log.phy.get("configured_freq_hz", "0"), 0
            ),
            "source_actual_freq_hz": int(
                uart_log.phy.get("actual_freq_hz", "0"), 0
            ),
        },
        "symbols": {
            "header_count": len(encoded.header_symbol_ids),
            "payload_count": len(encoded.payload_symbol_ids),
            "header_ids": list(encoded.header_symbol_ids),
            "payload_ids": list(encoded.payload_symbol_ids),
        },
        "iq": {
            "relative_path": iq_path.relative_to(output_root).as_posix(),
            "dtype": "<c8",
            "complex_samples": int(encoded.samples.size),
            "bytes": int(encoded.samples.nbytes),
            "sha256": str(iq_sha256),
            "leading_silence_samples": int(config.leading_silence_samples),
            "trailing_silence_samples": int(config.trailing_silence_samples),
            "artificial_cfo_hz": 0.0,
            "awgn_added": False,
        },
        "source_uart": {
            "path": _portable_source_path(uart_log.source_path),
            "sha256": str(uart_log.source_sha256),
            "phy": dict(uart_log.phy),
            "reference_config": dict(uart_log.reference_config),
            "phy_registers": dict(uart_log.phy_registers),
        },
        "generator": {
            "module": "weak_decoder.rf_super_resolution.reference_phy",
            "source_path": _portable_source_path(generator_path),
            "source_sha256": _sha256_file(generator_path),
            "phy_backend": "rfsr.PHY.encode_raw_phy",
            "phy_backend_source_path": _portable_source_path(
                phy_backend_path
            ),
            "phy_backend_source_sha256": _sha256_file(phy_backend_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "internal_raw_frame_round_trip": True,
            "ota_waveform_validation": "pending",
        },
    }


def write_reference_packet(
    uart_log: UartReferenceLog,
    packet: UartPacketRecord,
    config: ReferencePhyConfig,
    output_root: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Generate one cfile and JSON sidecar with atomic final replacement."""

    root = Path(output_root).expanduser().resolve()
    reference_dir = root / "reference"
    metadata_dir = root / "metadata"
    reference_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(packet.payload_id):06d}"
    iq_path = reference_dir / f"signalout_{stem}_fulltrim.cfile"
    metadata_path = metadata_dir / f"{stem}.json"
    if not bool(overwrite) and (iq_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"reference output already exists for payload_id={packet.payload_id}; "
            "pass overwrite=True only after verifying the target."
        )

    encoded = encode_reference_phy(packet.frame, config)
    iq_tmp = iq_path.with_name(iq_path.name + ".tmp")
    metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
    try:
        encoded.samples.tofile(iq_tmp)
        iq_sha256 = _sha256_file(iq_tmp)
        metadata = reference_metadata(
            uart_log=uart_log,
            packet=packet,
            config=config,
            encoded=encoded,
            output_root=root,
            iq_path=iq_path,
            iq_sha256=iq_sha256,
        )
        metadata_tmp.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if bool(overwrite):
            iq_tmp.replace(iq_path)
            metadata_tmp.replace(metadata_path)
        else:
            if iq_path.exists() or metadata_path.exists():
                raise FileExistsError(
                    f"reference output appeared during generation for id={packet.payload_id}."
                )
            iq_tmp.replace(iq_path)
            metadata_tmp.replace(metadata_path)
    finally:
        iq_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
    return iq_path, metadata_path
